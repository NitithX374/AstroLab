"""
AstroLab CLI — Integration & Unit Test Suite
============================================
Run with:
    python tests/test_astrolab.py
"""

import sys
import math
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from astrolab.core.models import CelestialBody, SimulationState, Vector3D
from astrolab.engine.simulator import SimulationEngine
from astrolab.physics import toolkit
from astrolab.physics.gravity import compute_gravitational_forces, compute_accelerations, G
from astrolab.physics.integrators import (
    EulerIntegrator, RK4Integrator, VelocityVerletIntegrator, get_integrator, INTEGRATORS
)
from astrolab.state.manager import StateManager


# ── helpers ──────────────────────────────────────────────────────────────────

def make_solar_system():
    """Return a StateManager pre-loaded with Sun + Earth."""
    sun = CelestialBody(
        name="sun", mass=1.989e30,
        position=Vector3D(0, 0, 0), velocity=Vector3D(0, 0, 0),
        radius=6.96e8, body_type="star",
    )
    earth = CelestialBody(
        name="earth", mass=5.972e24,
        position=Vector3D(1.496e11, 0, 0), velocity=Vector3D(0, 29_780, 0),
        radius=6.371e6, body_type="planet",
    )
    sm = StateManager()
    sm.add_body(sun)
    sm.add_body(earth)
    return sm


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results = []


def test(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    _results.append(condition)


# ── Vector3D tests ────────────────────────────────────────────────────────────

def test_vector3d():
    print("\n── Vector3D ─────────────────────────────────────────────────")
    v1 = Vector3D(1, 2, 3)
    v2 = Vector3D(4, 5, 6)

    test("addition",       v1 + v2 == Vector3D(5, 7, 9))
    test("subtraction",    v2 - v1 == Vector3D(3, 3, 3))
    test("scalar mul",     v1 * 2  == Vector3D(2, 4, 6))
    test("rmul",           2 * v1  == Vector3D(2, 4, 6))
    test("truediv",        v2 / 2  == Vector3D(2, 2.5, 3))
    test("negation",       -v1     == Vector3D(-1, -2, -3))
    test("dot product",    v1.dot(v2) == 32.0)
    test("cross product",  v1.cross(v2) == Vector3D(-3, 6, -3))
    test("magnitude",      abs(v1.magnitude() - math.sqrt(14)) < 1e-10)
    test("magnitude_sq",   v1.magnitude_sq() == 14.0)
    test("normalized",     abs(v1.normalized().magnitude() - 1.0) < 1e-12)
    test("zero factory",   Vector3D.zero() == Vector3D(0, 0, 0))
    test("from_tuple",     Vector3D.from_tuple((1, 2, 3)) == v1)
    test("to_tuple",       v1.to_tuple() == (1.0, 2.0, 3.0))


# ── CelestialBody tests ────────────────────────────────────────────────────────

def test_celestial_body():
    print("\n── CelestialBody ────────────────────────────────────────────")
    earth = CelestialBody(
        name="earth", mass=5.972e24,
        position=Vector3D(1.496e11, 0, 0),
        velocity=Vector3D(0, 29_780, 0),
        radius=6.371e6,
    )
    test("speed property",       abs(earth.speed - 29_780) < 1e-6)
    test("kinetic_energy prop",  abs(earth.kinetic_energy() - 0.5 * 5.972e24 * 29_780**2) < 1e10)
    test("serialise round-trip", CelestialBody.from_dict(earth.to_dict()).mass == earth.mass)
    test("distance_from_origin", abs(earth.distance_from_origin - 1.496e11) < 1.0)


# ── SimulationState tests ──────────────────────────────────────────────────────

def test_simulation_state():
    print("\n── SimulationState ──────────────────────────────────────────")
    sm = make_solar_system()
    state = sm.state

    test("body_count",       state.body_count() == 2)
    test("get_body found",   state.get_body("earth") is not None)
    test("get_body missing", state.get_body("mars") is None)
    test("body_names",       state.body_names() == ["earth", "sun"])

    # Save/load round-trip
    path = "_test_state_tmp.json"
    state.save(path)
    loaded = SimulationState.load(path)
    os.remove(path)
    test("save/load body count",     loaded.body_count() == 2)
    test("save/load earth mass",     loaded.get_body("earth").mass == 5.972e24)
    test("save/load integrator",     loaded.integrator == state.integrator)


# ── Gravity tests ──────────────────────────────────────────────────────────────

def test_gravity():
    print("\n── Gravity Engine ───────────────────────────────────────────")
    sm = make_solar_system()
    bodies = sm.get_all_bodies()

    forces = compute_gravitational_forces(bodies)
    test("Newton 3rd law",   abs((forces[0] + forces[1]).magnitude()) < 1e6)

    sun_idx   = next(i for i, b in enumerate(bodies) if b.name == "sun")
    earth_idx = next(i for i, b in enumerate(bodies) if b.name == "earth")
    F_expected = G * bodies[sun_idx].mass * bodies[earth_idx].mass / (1.496e11)**2
    F_actual   = forces[earth_idx].magnitude()
    test("force magnitude",  abs(F_actual - F_expected) / F_expected < 0.001,
         f"{F_actual:.4e} N vs {F_expected:.4e} N expected")

    accels = compute_accelerations(bodies)
    for a in accels:
        test("accel is vector", isinstance(a, Vector3D))
        break


# ── Integrator tests ───────────────────────────────────────────────────────────

def test_integrators():
    print("\n── Integrators ──────────────────────────────────────────────")

    # Registry
    test("INTEGRATORS keys",  set(INTEGRATORS.keys()) == {'euler', 'rk4', 'verlet'})
    test("get_integrator rk4",   isinstance(get_integrator('rk4'), RK4Integrator))
    test("get_integrator euler", isinstance(get_integrator('euler'), EulerIntegrator))
    test("get_integrator bad",   _raises(lambda: get_integrator('newton'), ValueError))

    # All integrators should advance position
    for name, integ in INTEGRATORS.items():
        sm = make_solar_system()
        earth_before = sm.get_body("earth").position.magnitude()
        integ.step(sm.get_all_bodies(), dt=3600.0)
        earth_after  = sm.get_body("earth").position.magnitude()
        test(f"{name} advances position", earth_before != earth_after)


def _raises(fn, exc_type) -> bool:
    try:
        fn()
        return False
    except exc_type:
        return True


# ── Toolkit tests ──────────────────────────────────────────────────────────────

def test_toolkit():
    print("\n── Astrophysics Toolkit ─────────────────────────────────────")
    sm    = make_solar_system()
    sun   = sm.get_body("sun")
    earth = sm.get_body("earth")

    ve = toolkit.escape_velocity(earth)
    test("escape velocity (Earth)",
         abs(ve / 1e3 - 11.19) < 0.05,
         f"{ve / 1e3:.4f} km/s")

    T = toolkit.orbital_period(earth, sun)
    test("orbital period (Earth)",
         abs(T / 86_400 - 365.25) < 2.0,
         f"{T / 86_400:.2f} days")

    F, uv = toolkit.gravitational_force(earth, sun)
    test("grav force magnitude",  abs(F - 3.54e22) / 3.54e22 < 0.01, f"{F:.4e} N")
    test("unit vector length",    abs(uv.magnitude() - 1.0) < 1e-12)

    rs = toolkit.schwarzschild_radius(sun)
    test("Schwarzschild radius (Sun)", abs(rs - 2954) < 10, f"{rs:.2f} m")

    e = toolkit.total_system_energy([sun, earth])
    test("KE > 0",   e['kinetic'] > 0)
    test("PE < 0",   e['potential'] < 0)
    test("total = KE + PE", abs(e['total'] - (e['kinetic'] + e['potential'])) < 1.0)

    pts = toolkit.lagrange_points(sun, earth)
    test("Lagrange 5 points",    set(pts.keys()) == {'L1', 'L2', 'L3', 'L4', 'L5'})
    l1_d = (pts['L1'] - sun.position).magnitude()
    test("L1 between Sun and Earth",
         0 < l1_d < (earth.position - sun.position).magnitude(),
         f"L1 at {l1_d:.4e} m from Sun")


# ── Simulation Engine tests ────────────────────────────────────────────────────

def test_simulation_engine():
    print("\n── Simulation Engine ────────────────────────────────────────")
    sm     = make_solar_system()
    engine = SimulationEngine(sm.state)

    # Single step
    e0_total = toolkit.total_system_energy(sm.state.bodies)['total']
    engine.step()
    test("step increments state.step", sm.state.step == 1)
    test("step advances state.time",   sm.state.time == sm.state.dt)

    # run()
    sm2 = make_solar_system()
    sim2 = SimulationEngine(sm2.state)
    result = sim2.run(steps=100)
    test("run returns RunResult",    result.steps_taken == 100)
    test("run elapsed time",         abs(result.elapsed_time - sm2.state.dt * 100) < 1e-6)

    # Energy conservation over 1 year with RK4
    sm3 = make_solar_system()
    sm3.state.dt = 3600.0
    sim3 = SimulationEngine(sm3.state)
    E_before = toolkit.total_system_energy(sm3.state.bodies)['total']
    result3 = sim3.run_with_monitor(steps=8760, energy_interval=100, collision_detection=False)
    E_after  = toolkit.total_system_energy(sm3.state.bodies)['total']
    drift    = abs((E_after - E_before) / E_before) * 100
    test("RK4 energy drift < 0.01% (1yr)",
         drift < 0.01,
         f"{drift:.6f}%")
    test("run_with_monitor has energy_log", len(result3.energy_log) > 0)

    # Integrator hot-swap
    engine.integrator = 'verlet'
    test("integrator hot-swap",  engine.integrator.name == 'verlet')
    test("state reflects swap",  sm.state.integrator == 'verlet')


# ── StateManager tests ────────────────────────────────────────────────────────

def test_state_manager():
    print("\n── StateManager ─────────────────────────────────────────────")
    sm = make_solar_system()
    test("add body returns True",  sm.add_body(CelestialBody("mars", 6.39e23, Vector3D.zero(), Vector3D.zero())))
    test("duplicate add returns False", not sm.add_body(CelestialBody("sun", 1e30, Vector3D.zero(), Vector3D.zero())))
    test("get_body works",         sm.get_body("mars") is not None)
    test("remove_body works",      sm.remove_body("mars"))
    test("remove missing False",   not sm.remove_body("pluto"))
    test("body count after ops",   len(sm.get_all_bodies()) == 2)

    # Export / import
    sm.state.dt = 60.0
    sm.export_state("_manager_test.json")
    sm2 = StateManager()
    sm2.import_state("_manager_test.json")
    os.remove("_manager_test.json")
    test("import body count",   len(sm2.get_all_bodies()) == 2)
    test("import dt preserved", sm2.state.dt == 60.0)
    sm2.clear()
    test("clear empties bodies", len(sm2.get_all_bodies()) == 0)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  AstroLab CLI — Test Suite")
    print("=" * 65)

    test_vector3d()
    test_celestial_body()
    test_simulation_state()
    test_gravity()
    test_integrators()
    test_toolkit()
    test_simulation_engine()
    test_state_manager()

    passed = sum(_results)
    total  = len(_results)
    failed = total - passed

    print("\n" + "=" * 65)
    if failed == 0:
        print(f"  \033[92mAll {total} tests passed.\033[0m")
    else:
        print(f"  \033[91m{failed}/{total} tests FAILED.\033[0m")
    print("=" * 65)
    sys.exit(0 if failed == 0 else 1)
