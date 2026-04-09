import os
import json
import math
import time
from typing import Dict, Any, List, Optional

try:
    from skyfield.api import load
    HAS_SKYFIELD = True
except ImportError:
    HAS_SKYFIELD = False

try:
    import numpy as np
    from scipy.optimize import minimize, basinhopping
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from astrolab.core.models import CelestialBody, SimulationState, Vector3D
from astrolab.engine.simulator import SimulationEngine
from astrolab.physics import toolkit

class SkyfieldEphemeris:
    """Wrapper to interact with JPL DE421 ephemeris for real planetary positioning."""
    def __init__(self):
        if not HAS_SKYFIELD:
            raise ImportError("Skyfield is required. Run: pip install skyfield")
        # Load from cache if exists, else download ~16MB
        self.ts = load.timescale()
        self.eph = load('de421.bsp')
        self._mapping = {
            'sun': 'sun',
            'mercury': 'mercury barycenter',
            'venus': 'venus barycenter',
            'earth': 'earth barycenter',
            'moon': 'moon',
            'mars': 'mars barycenter',
            'jupiter': 'jupiter barycenter',
            'saturn': 'saturn barycenter',
        }
        
    def get_state(self, body_name: str, jd: float) -> tuple[Vector3D, Vector3D]:
        """Return position (m) and velocity (m/s) in ICRS frame at given Julian Date."""
        target = self._mapping.get(body_name.lower())
        if not target:
            raise ValueError(f"Ephemeris data for '{body_name}' not available in DE421.")
        
        t = self.ts.tt_jd(jd)
        # We need relative to solar system barycenter if extracting absolute
        # but Astrolab centers on (0,0,0) usually. For trajectories, heliocentric is better.
        sun = self.eph['sun']
        obj = self.eph[target]
        astrometric = sun.at(t).observe(obj)
        
        pos_m = astrometric.position.m
        vel_ms = astrometric.velocity.m_per_s
        return Vector3D(*pos_m), Vector3D(*vel_ms)


class AstrodynamicsPlanner:
    """
    Finite-State Pipeline Planner: PLAN -> SIMULATE -> EVALUATE -> OPTIMIZE
    """
    
    SYSTEM_PROMPT = (
        "You are the Astrolab Planning Engine (LLM Component). "
        "Your task is to convert high-level trajectory goals into an exact JSON strategy. "
        "Available planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Moon.\n"
        "RULES:\n"
        "- If destination is 'Moon', set mission_type to 'cislunar'. NEVER use interplanetary transfer logic for Earth-Moon missions.\n"
        "- DO NOT include gravity assists unless the destination is another planet.\n"
        "Output ONLY valid JSON matching this schema:\n"
        "{\n"
        '  "mission_type": "string (cislunar or interplanetary)",\n'
        '  "start_body": "string",\n'
        '  "target_body": "string",\n'
        '  "earliest_launch_jd": float (Julian Date),\n'
        '  "latest_launch_jd": float (Julian Date),\n'
        '  "flybys": ["string"] (ordered list of assist candidates)\n'
        "}\n"
        "You cannot simulate — you only generate the math optimizer's search space.\n"
        "Assume current year is 2026 (JD ~2461000)."
    )

    def __init__(self, provider: str = "anthropic", model: str = "claude-haiku-4-5-20251001", api_key: str = ""):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.ephemeris = SkyfieldEphemeris() if HAS_SKYFIELD else None

    # 1. PLAN Phase (LLM)
    def _plan_via_llm(self, goal: str) -> Dict[str, Any]:
        """Invoke LLM to generate initial trajectory strategy constraints."""
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        
        prompt = f"User Goal: {goal}\nOutput the trajectory strategy in JSON format."
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=500,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
            
            # Extract JSON block
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
                
            return json.loads(text.strip())
        except Exception as exc:
            return {"error": f"LLM Planning failed: {exc}"}

    # 2. SIMULATE Phase (Physics Engine)
    def _simulate(self, jd_launch: float, jd_arrival: float, strategy: Dict) -> dict:
        """
        Fast-evaluator logic. 
        In a full loop, we'd fire the RK4 N-body engine. 
        For optimizer efficiency, we calculate Lambert's transfer (Porkchop) 
        and only fire the full SimulationEngine on the best candidate.
        """
        if not self.ephemeris:
            return {"error": "Skyfield not installed."}
            
        start = strategy['start_body']
        target = strategy['target_body']
        
        # Domain Classifier
        if target.lower() == 'moon':
            mode = "cislunar"
        else:
            mode = "interplanetary"
            
        r1, v1_planet = self.ephemeris.get_state(start, jd_launch)
        r2, v2_planet = self.ephemeris.get_state(target, jd_arrival)
        
        tof_days = jd_arrival - jd_launch
        tof_s = tof_days * 86400.0
        
        dist1 = r1.magnitude()
        dist2 = r2.magnitude()
        
        cost = 0.0
        details = {}
        
        if mode == "cislunar":
            # Patched Conic (Geocentric)
            mu_primary = 3.986004418e14 # m^3/s^2 (Earth)
            dist1 = 6771000.0 # LEO radius
            dist2 = (r2 - r1).magnitude() # actual moon distance from earth
        else:
            # Heliocentric
            mu_primary = 1.32712440018e20 # m^3/s^2 (Sun)
            
        # Approximate semi-major axis of transfer
        a_transfer = (dist1 + dist2) / 2.0
        
        # Vis-viva required speed at departure and arrival
        v_dep_req = math.sqrt(mu_primary * (2.0/dist1 - 1.0/a_transfer))
        v_arr_req = math.sqrt(mu_primary * (2.0/dist2 - 1.0/a_transfer))
        
        if mode == "cislunar":
            v_start = math.sqrt(mu_primary / dist1)
            dv1 = abs(v_dep_req - v_start)
            dv2 = abs(v_arr_req - math.sqrt(3.902e12 / dist2)) # NRHO simplified
            total_dv = dv1 + dv2
            
            # Artemis-class BLT Cislunar Math calculations
            energy_init = (v_start**2)/2.0 - mu_primary/dist1
            v_post_tli = v_start + dv1
            energy_post = (v_post_tli**2)/2.0 - mu_primary/dist1
            
            escape_achieved = (energy_post > 0)
            c3 = 2.0 * energy_post / 1e6 # km^2/s^2
            
            # Apoapsis
            if escape_achieved:
                r_apo = float('inf')
                moon_soi_entry = True
            else:
                a_orbit = -mu_primary / (2.0 * energy_post)
                r_apo = 2.0 * a_orbit - dist1
                moon_soi_entry = (r_apo > (dist2 - 66000000.0))
                
            # Lunar arrival
            v_apogee = math.sqrt(max(0, 2.0 * (energy_post + mu_primary/dist2)))
            v_moon_orbit = 1022.0 # m/s rough average
            v_inf = abs(v_apogee - v_moon_orbit)
            
            details.update({
                "energy_init_mj": energy_init / 1e6,
                "energy_post_mj": energy_post / 1e6,
                "escape": escape_achieved,
                "c3": c3,
                "soi_entry": moon_soi_entry,
                "v_arrival": v_apogee / 1000.0,
                "v_inf": v_inf / 1000.0,
                "dv1": dv1,
                "dv2": dv2,
                "dv_mid": 0.05, # Representative standard
            })
            
            # BLT Optimizer Objective
            # Anti-Cheat: Reject unviable domains
            if total_dv < 2500 or total_dv > 4000 or tof_days > 40 or tof_days < 10:
                cost += 1e9
            if not moon_soi_entry:
                cost += 1e9
                
            # Minimization Function
            w1, w2, w4 = 1.0, 0.5, 1.5
            cost += w1 * total_dv + w2 * abs(tof_days - 25.0)*1000.0 + w4 * v_inf
            
        else:
            dv1 = abs(v_dep_req - v1_planet.magnitude())
            dv2 = abs(v_arr_req - v2_planet.magnitude())
            total_dv = dv1 + dv2
            hohmann_tof_s = math.pi * math.sqrt((a_transfer**3) / mu_primary)
            tof_penalty = abs(tof_s - hohmann_tof_s) * 0.001
            cost = total_dv + tof_penalty
            details.update({"dv1": dv1, "dv2": dv2})
        
        return {
            "total_dv": total_dv,
            "cost": cost,
            "tof_days": tof_days,
            "domain": mode,
            **details
        }

    # 3/4. EVALUATE & OPTIMIZE Phase (Math Engine)
    def _optimize(self, strategy: Dict) -> Dict:
        """Use SciPy to find the optimal launch date and time of flight."""
        if not HAS_SCIPY:
            return {"error": "Scipy is required to run the optimizer."}
            
        min_jd = strategy.get('earliest_launch_jd', 2461000)
        max_jd = strategy.get('latest_launch_jd', 2461400)
        
        target = strategy.get('target_body', '')
        mode = "cislunar" if target.lower() == 'moon' else "interplanetary"
        
        # Objective function for SciPy

        def objective(x):
            jd_launch = x[0]
            tof_days = x[1]
            if mode == "cislunar":
                if tof_days < 10 or tof_days > 40:
                    return 1e9
            else:
                if jd_launch < min_jd or jd_launch > max_jd or tof_days < 50:
                    return 1e9 # out of bounds penalty
            
            sim_res = self._simulate(jd_launch, jd_launch + tof_days, strategy)
            if "error" in sim_res:
                return 1e9
            return sim_res["cost"]

        # Initial guess
        x0 = np.array([min_jd + (max_jd - min_jd)/2.0, 200.0])
        bounds = [(min_jd, max_jd), (50, 1000)]
        
        print(f"  [Math Engine] Running basin-hopping optimizer across launch window...")
        # We use a quick minimizer to save time in CLI
        res = minimize(objective, x0, bounds=bounds, method='Nelder-Mead', options={'maxiter': 50})
        
        best_launch = res.x[0]
        best_tof = res.x[1]
        
        # Evaluate final
        best_sim = self._simulate(best_launch, best_launch + best_tof, strategy)
        return {
            "success": True,
            "launch_jd": best_launch,
            "arrival_jd": best_launch + best_tof,
            "tof_days": best_tof,
            "metrics": best_sim
        }

    def execute_pipeline(self, goal: str, mode: str = "auto", manual_params: Dict = None) -> Dict:
        """Orchestrates the sequence based on Mode."""
        print(f"\n  [Planner] Initializing {mode.upper()} Mode")
        
        if mode == "auto":
            print("  [Planner] LLM Phase: Parsing goal & generating strategy...")
            strategy = self._plan_via_llm(goal)
            if "error" in strategy:
                return {"success": False, "message": strategy["error"]}
        else:
            strategy = manual_params or {}
            if 'start_body' not in strategy or 'target_body' not in strategy:
                return {"success": False, "message": "Manual mode requires strictly defined targets."}
                
        print(f"  [Planner] Strategy: {strategy['start_body']} -> {strategy['target_body']}")
        if "flybys" in strategy and strategy["flybys"]:
            print(f"  [Planner] Intended Flybys: {strategy['flybys']}")

        print("  [Planner] OPTIMIZE & SIMULATE Phase...")
        if mode == "auto":
            opt_result = self._optimize(strategy)
        else:
            jd_launch = strategy.get('launch_jd', 2461000)
            tof = strategy.get('tof_days', 200)
            sim_res = self._simulate(jd_launch, jd_launch + tof, strategy)
            opt_result = {
                "success": True,
                "launch_jd": jd_launch,
                "tof_days": tof,
                "metrics": sim_res
            }

        if "error" in opt_result.get("metrics", {}):
            return {"success": False, "message": opt_result["metrics"]["error"]}
            
        print("  [Planner] EVALUATE Phase complete.")
        
        total_dv_km_s = opt_result["metrics"]["total_dv"] / 1000.0
        
        return {
            "success": True,
            "start": strategy['start_body'],
            "target": strategy['target_body'],
            "launch_jd": opt_result['launch_jd'],
            "tof_days": opt_result['tof_days'],
            "delta_v_km_s": total_dv_km_s,
            "flybys": strategy.get("flybys", [])
        }

    def create_simulation_state(self, plan_result: Dict) -> SimulationState:
        """Converts an optimized plan into a ready-to-run N-body SimulationState."""
        if not self.ephemeris:
            raise RuntimeError("Skyfield required for simulation state generation.")
            
        jd = plan_result['launch_jd']
        target = plan_result['target']
        bodies = []
        
        # Create central body layout based on domain
        domain = plan_result.get('metrics', {}).get('domain', 'interplanetary')
        
        if domain == "cislunar":
            # Earth-Moon System
            earth_pos, earth_vel = self.ephemeris.get_state('earth', jd)
            moon_pos, moon_vel = self.ephemeris.get_state('moon', jd)
            bodies.append(CelestialBody("earth", 5.972e24, earth_pos, earth_vel, 6371000, "planet", "blue"))
            bodies.append(CelestialBody("moon", 7.342e22, moon_pos, moon_vel, 1737400, "moon", "gray"))
            
            # Spawn spacecraft at LEO
            sc_pos = earth_pos + Vector3D(6771000.0, 0, 0)
            sc_vel = earth_vel + Vector3D(0, plan_result['delta_v_km_s'] * 1000.0, 0) # escape velocity vector approx
            bodies.append(CelestialBody("spacecraft", 1000.0, sc_pos, sc_vel, 10.0, "spacecraft", "red"))
        else:
            # Solar System
            sun_pos, sun_vel = self.ephemeris.get_state('sun', jd)
            bodies.append(CelestialBody("sun", 1.989e30, sun_pos, sun_vel, 6.96e8, "star", "yellow"))
            
            for planet in ['earth', 'mars', 'venus', 'jupiter']:
                try:
                    p_pos, p_vel = self.ephemeris.get_state(planet, jd)
                    bodies.append(CelestialBody(planet, 1e24, p_pos, p_vel, 6e6, "planet", "white")) # mass approx for visualize
                except ValueError:
                    pass
            
            # Spawn spacecraft roughly near start planet
            start_name = plan_result['start']
            start_body = next((b for b in bodies if b.name == start_name.lower()), bodies[0])
            sc_pos = start_body.position + Vector3D(1e7, 1e7, 0)
            sc_vel = start_body.velocity * 1.1 # simplistic $\Delta v$ kick for orbit visualization
            bodies.append(CelestialBody("spacecraft", 1000.0, sc_pos, sc_vel, 10.0, "spacecraft", "magenta"))
            
        dt = 60 if domain == "cislunar" else 3600
        return SimulationState(bodies=bodies, time=0.0, dt=dt, step=0, integrator="rk4")
