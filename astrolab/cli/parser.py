"""
AstroLab CLI — Interactive REPL
================================
A hierarchical, Cisco-CLI-inspired command interface built on Python's
``cmd`` standard library module.  No third-party dependencies required.

Commands
--------
create body <name> [key=val ...]  — Add a celestial body
delete body <name>                — Remove a body
edit body <name> [key=val ...]   — Update fields of an existing body
show state | body <n> | energy | integrators
simulate dt=<s> steps=<n> [integrator=rk4] [collisions=on] [log_energy=<n>]
compute escape_velocity | orbital_period | grav_force | energy |
        schwarzschild | lagrange | photon_sphere | isco | redshift | timedilation
blackhole create | info | geodesic | shadow | potential | lensing | redshift | timedilation
set dt=<s> | integrator=<name>    — Adjust simulation parameters
export state file=<path>          — Save state to JSON
import state file=<path>          — Load state from JSON
clear                             — Reset simulation
run <script.astro>                — Execute a batch script
help [command]                    — Show help
exit / quit                       — Quit

Unit shortcuts (parsed at CLI boundary, stored as SI)
------------------------------------------------------
  Distances : m (default), km (*1e3), AU (*1.496e11)
  Speeds    : m/s (default), km/s (*1e3)
  Masses    : kg (default, implicit)
"""

from __future__ import annotations

import cmd
import shlex
import os
from typing import Optional

from astrolab.core.models import CelestialBody, SimulationState, Vector3D
from astrolab.engine.simulator import SimulationEngine
from astrolab.state.manager import StateManager
from astrolab.physics import toolkit
from astrolab.physics.integrators import INTEGRATORS
from astrolab.viz.recorder import TrajectoryRecorder


# ---------------------------------------------------------------------------
# Unit / argument parsing helpers
# ---------------------------------------------------------------------------

def parse_scalar(s: str) -> float:
    """Parse a numeric string with optional unit suffix to SI value."""
    s = s.strip().lower()
    multipliers = [
        ('au',   1.496e11),
        ('km/s', 1e3),
        ('km',   1e3),
        ('m/s',  1.0),
        ('m',    1.0),
        ('kg',   1.0),
    ]
    for suffix, factor in multipliers:
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) * factor
    return float(s)


def parse_vector(s: str) -> Vector3D:
    """
    Parse a vector string like ``(x, y, z)`` or ``x,y,z``.
    Each component may carry a unit suffix.
    """
    s = s.strip().strip('()')
    parts = [p.strip() for p in s.split(',')]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 components (x,y,z), got: {s!r}")
    return Vector3D(parse_scalar(parts[0]), parse_scalar(parts[1]), parse_scalar(parts[2]))


def parse_kwargs(tokens: list) -> dict:
    """Parse ``['key=val', ...]`` into ``{'key': 'val', ...}``."""
    result = {}
    for token in tokens:
        if '=' not in token:
            raise ValueError(f"Expected key=value pair, got: {token!r}")
        k, v = token.split('=', 1)
        result[k.strip().lower()] = v.strip()
    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

BANNER = r"""
   █████╗ ███████╗████████╗██████╗  ██████╗ ██╗      █████╗ ██████╗ 
  ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗██║     ██╔══██╗██╔══██╗
  ███████║███████╗   ██║   ██████╔╝██║   ██║██║     ███████║██████╔╝
  ██╔══██║╚════██║   ██║   ██╔══██╗██║   ██║██║     ██╔══██║██╔══██╗
  ██║  ██║███████║   ██║   ██║  ██║╚██████╔╝███████╗██║  ██║██████╔╝
  ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═════╝ 

   AstroLab CLI  v3.0.0  |  N-Body & General Relativity Simulation
  ─────────────────────────────────────────────────────────
  Type  help         to list commands
        help create  for detailed usage of a specific command
        exit         to quit
"""


def _ruler(width: int = 80) -> str:
    return '  ' + '─' * width


# ---------------------------------------------------------------------------
# CLI class
# ---------------------------------------------------------------------------

class AstroLabCLI(cmd.Cmd):
    """Interactive AstroLab command interpreter."""

    prompt = '\033[96mAstroLab>\033[0m '
    intro  = BANNER

    def __init__(self, state_manager: Optional[StateManager] = None) -> None:
        super().__init__()
        self.manager   = state_manager or StateManager()
        self._engine   = SimulationEngine(self.manager.state)
        self._recorder: Optional[TrajectoryRecorder] = None
        self._gr_engine = None   # Lazy-loaded GR engine
        self._bh_config = None   # Current black hole configuration

    # ── create ───────────────────────────────────────────────────────────────

    def do_create(self, arg: str) -> None:
        """
Create a celestial body and add it to the simulation.

  create body <name> mass=<m> [pos=(x,y,z)] [vel=(vx,vy,vz)]
                               [radius=<r>] [type=<t>] [color=<c>]

Supported unit suffixes
  Mass     : kg (default)
  Distance : m (default) | km | AU
  Speed    : m/s (default) | km/s

Examples
  create body sun   mass=1.989e30 radius=696000000 type=star
  create body earth mass=5.97e24  pos=(1AU,0,0) vel=(0,29.78km/s,0) \\
                                  radius=6371000  type=planet
  create body moon  mass=7.342e22 pos=(1.496e11,384.4e6,0) \\
                                  vel=(0,30802,0) radius=1737400
        """
        tokens = shlex.split(arg)
        if len(tokens) < 3 or tokens[0] != 'body':
            print("  Usage: create body <name> mass=<m> [...]")
            return
        name, *kv_tokens = tokens[1], *tokens[2:]
        try:
            kw = parse_kwargs(tokens[2:])
        except ValueError as exc:
            print(f"  [!] {exc}")
            return

        if 'mass' not in kw:
            print("  [!] 'mass' is required.")
            return

        try:
            mass      = parse_scalar(kw['mass'])
            position  = parse_vector(kw.get('pos', '(0,0,0)'))
            velocity  = parse_vector(kw.get('vel', '(0,0,0)'))
            radius    = parse_scalar(kw.get('radius', '0'))
            body_type = kw.get('type', 'unknown')
            color     = kw.get('color', 'white')
        except ValueError as exc:
            print(f"  [!] Parse error: {exc}")
            return

        body = CelestialBody(
            name=name,
            mass=mass,
            position=position,
            velocity=velocity,
            radius=radius,
            body_type=body_type,
            color=color,
        )

        if self.manager.add_body(body):
            print(f"  [+] Created {body_type.upper()} '{name}'")
            print(f"      mass={mass:.4e} kg  |  radius={radius:.4g} m")
            print(f"      pos={position}  |  vel={velocity}")
        else:
            print(f"  [-] Body '{name}' already exists.")

    # ── delete ───────────────────────────────────────────────────────────────

    def do_delete(self, arg: str) -> None:
        """
Remove a celestial body from the simulation.

  delete body <name>
        """
        tokens = shlex.split(arg)
        if len(tokens) != 2 or tokens[0] != 'body':
            print("  Usage: delete body <name>")
            return
        name = tokens[1]
        if self.manager.remove_body(name):
            print(f"  [+] Body '{name}' removed.")
        else:
            print(f"  [-] Body '{name}' not found.")

    # ── edit ─────────────────────────────────────────────────────────────────

    def do_edit(self, arg: str) -> None:
        """
Edit (update) fields of an existing celestial body in-place.

  edit body <name> [mass=<m>] [pos=(x,y,z)] [vel=(vx,vy,vz)]
                              [radius=<r>]   [type=<t>] [color=<c>]

Only the fields you supply are changed; all others are preserved.
Supports the same unit suffixes as 'create body'.

Examples
  edit body tau   radius=695700000
  edit body earth vel=(0,30000,0)
  edit body sun   mass=1.99e30 type=star color=orange
        """
        tokens = shlex.split(arg)
        if len(tokens) < 3 or tokens[0] != 'body':
            print("  Usage: edit body <name> [mass=...] [pos=...] [vel=...] "
                  "[radius=...] [type=...] [color=...]")
            return

        name = tokens[1]
        body = self.manager.get_body(name)
        if body is None:
            print(f"  [-] Body '{name}' not found.  Use 'show state' to list bodies.")
            return

        try:
            kw = parse_kwargs(tokens[2:])
        except ValueError as exc:
            print(f"  [!] Argument error: {exc}")
            return

        if not kw:
            print("  [!] No fields specified. Nothing to update.")
            print("  Editable fields: mass | pos | vel | radius | type | color")
            return

        changes = []
        try:
            if 'mass' in kw:
                body.mass = parse_scalar(kw['mass'])
                changes.append(f"mass={body.mass:.4e} kg")

            if 'pos' in kw:
                body.position = parse_vector(kw['pos'])
                changes.append(f"pos={body.position}")

            if 'vel' in kw:
                body.velocity = parse_vector(kw['vel'])
                changes.append(f"vel={body.velocity}")

            if 'radius' in kw:
                body.radius = parse_scalar(kw['radius'])
                changes.append(f"radius={body.radius:.4g} m")

            if 'type' in kw:
                body.body_type = kw['type']
                changes.append(f"type={body.body_type}")

            if 'color' in kw:
                body.color = kw['color']
                changes.append(f"color={body.color}")

        except ValueError as exc:
            print(f"  [!] Parse error: {exc}")
            return

        unknown = set(kw) - {'mass', 'pos', 'vel', 'radius', 'type', 'color'}
        if unknown:
            print(f"  [!] Unknown field(s): {', '.join(sorted(unknown))}")
            print("  Editable fields: mass | pos | vel | radius | type | color")

        print(f"  [+] Updated '{name}':")
        for change in changes:
            print(f"      {change}")

    # ── show ─────────────────────────────────────────────────────────────────

    def do_show(self, arg: str) -> None:
        """
Display simulation state.

  show state              — Table of all bodies
  show body <name>        — Detailed view of a single body
  show energy             — KE / PE / total energy of the system
  show integrators        — List available numerical integrators
  show trajectory         — Summary of recorded trajectory data
        """
        tokens = shlex.split(arg) if arg.strip() else []
        if not tokens:
            print("  Usage: show state | body <name> | energy | integrators | trajectory")
            return
 
        sub = tokens[0].lower()
 
        if sub == 'state':
            self._show_state()
        elif sub == 'body' and len(tokens) == 2:
            self._show_body(tokens[1])
        elif sub == 'energy':
            self._show_energy()
        elif sub == 'integrators':
            self._show_integrators()
        elif sub == 'trajectory':
            self._show_trajectory()
        else:
            print("  Unknown. Try: show state | show body <name> | show energy | show integrators | show trajectory")

    def _show_state(self) -> None:
        bodies = self.manager.get_all_bodies()
        if not bodies:
            print("  [i] Simulation is empty.  Use 'create body' to add bodies.")
            return
        state = self.manager.state
        days  = state.time / 86_400
        print(f"\n  Simulation Time: {state.time:.4e} s ({days:.2f} days)  "
              f"| Step: {state.step}  | dt: {state.dt} s  | Integrator: {state.integrator}")
        print(_ruler(110))
        hdr = f"  {'Name':<14} {'Type':<10} {'Mass (kg)':<14} {'Radius (m)':<14} {'Position (m)':<40} Velocity (m/s)"
        print(hdr)
        print(_ruler(110))
        for b in bodies:
            print(f"  {b.name:<14} {b.body_type:<10} {b.mass:<14.4e} "
                  f"{b.radius:<14.4g} {str(b.position):<40} {b.velocity}")
        print()

    def _show_body(self, name: str) -> None:
        b = self.manager.get_body(name)
        if not b:
            print(f"  [-] Body '{name}' not found.")
            return
        speed = b.velocity.magnitude()
        dist  = b.position.magnitude()
        ke    = b.kinetic_energy()
        print(f"\n  ┌─ {b.name}  ({b.body_type})")
        print(f"  │  Mass         : {b.mass:.6e} kg")
        print(f"  │  Radius       : {b.radius:.6g} m")
        print(f"  │  Position     : {b.position}")
        print(f"  │  |position|   : {dist:.4e} m")
        print(f"  │  Velocity     : {b.velocity}")
        print(f"  │  |velocity|   : {speed:.4e} m/s")
        print(f"  │  KE           : {ke:.4e} J")
        print(f"  └─ Color        : {b.color}")
        print()

    def _show_energy(self) -> None:
        bodies = self.manager.get_all_bodies()
        if not bodies:
            print("  [i] No bodies in simulation.")
            return
        e = toolkit.total_system_energy(bodies)
        print(f"\n  ┌─ System Energy")
        print(f"  │  Kinetic    : {e['kinetic']:+.6e} J")
        print(f"  │  Potential  : {e['potential']:+.6e} J")
        print(f"  └─ Total      : {e['total']:+.6e} J")
        print()

    def _show_integrators(self) -> None:
        print("\n  Available integrators:")
        desc = {
            'euler':  '1st-order explicit. Fast, not suitable for long runs.',
            'rk4':    '4th-order Runge-Kutta. Default, best all-round choice.',
            'verlet': 'Velocity Verlet (symplectic). Best for long-term stability.',
        }
        for name in INTEGRATORS:
            active = ' ◀ active' if name == self.manager.state.integrator else ''
            print(f"    {name:<8}  {desc.get(name, '')}{active}")
        print()

    def _show_trajectory(self) -> None:
        if not self._recorder or self._recorder.snapshot_count() == 0:
            print("  [i] No trajectory data recorded.  Use 'simulate ... visualize=on' to record.")
            return
        t0, tf = self._recorder.time_range()
        print(f"\n  ┌─ Recorded Trajectory")
        print(f"  │  Bodies     : {len(self._recorder.get_body_names())}")
        print(f"  │  Snapshots  : {self._recorder.snapshot_count()}")
        print(f"  └─ Time Range : {t0:.2e} s to {tf:.2e} s ({(tf-t0)/86400:.2f} days)")
        print()

    # ── set ──────────────────────────────────────────────────────────────────

    def do_set(self, arg: str) -> None:
        """
Adjust simulation parameters.

  set dt=<seconds>
  set integrator=euler|rk4|verlet

Examples
  set dt=60
  set integrator=verlet
        """
        try:
            kw = parse_kwargs(shlex.split(arg))
        except ValueError as exc:
            print(f"  [!] {exc}")
            return
        for key, val in kw.items():
            if key == 'dt':
                self.manager.state.dt = float(val)
                print(f"  [+] Timestep set to {val} s")
            elif key == 'integrator':
                try:
                    self._engine.integrator = val
                    print(f"  [+] Integrator set to '{val}'")
                except ValueError as exc:
                    print(f"  [!] {exc}")
            else:
                print(f"  [!] Unknown parameter: '{key}'.  Try: dt | integrator")

    # ── simulate ─────────────────────────────────────────────────────────────

    def do_simulate(self, arg: str) -> None:
        """
Advance the simulation forward in time.

  simulate dt=<s> steps=<n> [integrator=rk4|euler|verlet]
                              [collisions=on|off] [log_energy=<n>]

Arguments
  dt           : Timestep in seconds (e.g. 3600 for 1 hour).
  steps        : Number of timesteps.
  integrator   : rk4 (default) | euler | verlet.
  collisions   : on (default) | off.
  log_energy   : Log energy every N steps (0 = disabled).

Examples
  simulate dt=3600 steps=8760 integrator=rk4
  simulate dt=60   steps=1440 integrator=verlet log_energy=100
        """
        try:
            kw = parse_kwargs(shlex.split(arg))
        except ValueError as exc:
            print(f"  [!] {exc}")
            return

        if 'dt' not in kw or 'steps' not in kw:
            print("  [!] 'dt' and 'steps' are required.")
            return

        try:
            dt           = float(kw['dt'])
            steps        = int(float(kw['steps']))
            integrator   = kw.get('integrator', self.manager.state.integrator)
            collisions   = kw.get('collisions', 'on').lower() != 'off'
            log_interval = int(kw.get('log_energy', '0'))
            viz_mode     = kw.get('visualize', 'off').lower()
            output_file  = kw.get('output')
        except ValueError as exc:
            print(f"  [!] Parse error: {exc}")
            return

        if not self.manager.get_all_bodies():
            print("  [!] No bodies. Use 'create body' first.")
            return

        # Apply parameters
        self.manager.state.dt = dt
        try:
            self._engine.integrator = integrator
        except ValueError as exc:
            print(f"  [!] {exc}")
            return

        total_time  = dt * steps
        total_days  = total_time / 86_400
        total_years = total_time / (365.25 * 86_400)
        print(f"\n  Simulating {steps:,} step(s) × {dt} s  "
              f"= {total_time:.3e} s  ({total_days:.1f} days / {total_years:.3f} yr)")
        print(f"  Integrator: {integrator.upper()}  |  Collisions: {'ON' if collisions else 'OFF'}\n")

        # Progress bar
        bar_width = 44
        last_pct  = [-1]

        def on_progress(current: int, total: int) -> None:
            pct    = int(current / total * 100)
            filled = int(bar_width * current / total)
            if pct != last_pct[0]:
                bar = '█' * filled + '░' * (bar_width - filled)
                print(f"\r  [{bar}] {pct:3d}%", end='', flush=True)
                last_pct[0] = pct

        # Prepare recorder if requested
        if viz_mode != 'off':
            self._recorder = TrajectoryRecorder(record_every=1)
        else:
            self._recorder = None

        if viz_mode == 'live':
            try:
                from astrolab.viz.vispy_viz import AstroLabLive
                viewer = AstroLabLive(
                    state=self.manager.state,
                    engine=self._engine,
                    total_steps=steps,
                    recorder=self._recorder
                )
                print("  [*] Launching live visualizer...")
                viewer.run()
                # Live mode completes when window closes
                print(f"\n  [+] Live simulation closed.")
            except ImportError as exc:
                print(f"\n  [!] Could not start live view: {exc}")
                print("      Install dependencies: pip install vispy pyqt5")
            return

        result = self._engine.run_with_monitor(
            steps=steps,
            collision_detection=collisions,
            energy_interval=log_interval,
            progress_callback=on_progress,
            recorder=self._recorder
        )

        # Final bar fill-in and summary
        print(f"\r  [{'█' * bar_width}] 100%")
        print(f"\n  [+] Complete.  Simulated {result.elapsed_time:.3e} s total.")

        if result.collisions:
            print(f"\n  Collision Events ({len(result.collisions)})")
            print(_ruler(70))
            for ev in result.collisions:
                print(f"    t={ev.time:.3e} s  |  '{ev.absorbed}' → absorbed by '{ev.survivor}'  "
                      f"|  mass={ev.new_mass:.3e} kg")

        if result.energy_log:
            e0 = result.energy_log[0]['total']
            ef = result.energy_log[-1]['total']
            drift = abs((ef - e0) / e0) * 100 if e0 != 0.0 else 0.0
            print(f"\n  Energy Monitor: E₀={e0:+.4e} J  |  Ef={ef:+.4e} J  |  drift={drift:.4f}%")
 
        if viz_mode == 'on' and output_file and self._recorder:
            if output_file.endswith('.html'):
                try:
                    from astrolab.viz.plotly_viz import render_html
                    render_html(self._recorder, output_file)
                except ImportError as exc:
                    print(f"  [!] Could not render HTML: {exc}")
            else:
                print(f"  [!] Automatic render only supports .html output. Use 'visualize' for other options.")

        print()

    # ── compute ──────────────────────────────────────────────────────────────

    def do_compute(self, arg: str) -> None:
        """
Run astrophysics toolkit calculations.

Newtonian
  compute escape_velocity  body=<name>
  compute orbital_period   body=<name>   primary=<primary>
  compute grav_force       body1=<name>  body2=<name>
  compute energy
  compute schwarzschild    body=<name>
  compute lagrange         primary=<name> secondary=<name>

General Relativity
  compute photon_sphere    body=<name>
  compute isco             body=<name>  [spin=<a/M>]
  compute redshift         body=<name>  r_emit=<m>  r_obs=<m>
  compute timedilation     body=<name>  r=<m>

Examples
  compute escape_velocity body=earth
  compute photon_sphere body=sun
  compute isco body=sun spin=0.5
  compute redshift body=sun r_emit=5000 r_obs=1e12
  compute timedilation body=sun r=10000
        """
        tokens = shlex.split(arg)
        if not tokens:
            print("  Usage: compute <subcommand> [key=val ...]")
            return

        sub = tokens[0].lower()
        try:
            kw = parse_kwargs(tokens[1:]) if len(tokens) > 1 else {}
        except ValueError as exc:
            print(f"  [!] {exc}")
            return

        if sub == 'escape_velocity':
            b = self._get_body(kw.get('body'))
            if not b: return
            if b.radius <= 0:
                print(f"  [!] '{b.name}' has no radius.  Set it with: "
                      f"create body ... radius=<r>")
                return
            ve = toolkit.escape_velocity(b)
            print(f"\n  Escape velocity of '{b.name}': {ve:.4f} m/s  "
                  f"({ve / 1e3:.4f} km/s)\n")

        elif sub == 'orbital_period':
            orb = self._get_body(kw.get('body'))
            pri = self._get_body(kw.get('primary'))
            if not orb or not pri: return
            T    = toolkit.orbital_period(orb, pri)
            days = T / 86_400
            yrs  = T / (365.25 * 86_400)
            print(f"\n  Orbital period of '{orb.name}' around '{pri.name}':")
            print(f"    {T:.4e} s  |  {days:.2f} days  |  {yrs:.4f} years\n")

        elif sub == 'grav_force':
            b1 = self._get_body(kw.get('body1'))
            b2 = self._get_body(kw.get('body2'))
            if not b1 or not b2: return
            F, uv = toolkit.gravitational_force(b1, b2)
            dist  = (b1.position - b2.position).magnitude()
            print(f"\n  Gravitational force: '{b1.name}' ↔ '{b2.name}'")
            print(f"    Separation : {dist:.4e} m")
            print(f"    Force      : {F:.4e} N")
            print(f"    Direction  : {uv}  (unit vector {b1.name} → {b2.name})\n")

        elif sub == 'energy':
            bodies = self.manager.get_all_bodies()
            if not bodies:
                print("  [i] No bodies."); return
            e = toolkit.total_system_energy(bodies)
            print(f"\n  System Energy")
            print(f"    Kinetic   : {e['kinetic']:+.6e} J")
            print(f"    Potential : {e['potential']:+.6e} J")
            print(f"    Total     : {e['total']:+.6e} J\n")

        elif sub == 'schwarzschild':
            b  = self._get_body(kw.get('body'))
            if not b: return
            rs = toolkit.schwarzschild_radius(b)
            note = ''
            if b.radius > 0:
                ratio = b.radius / rs
                note  = (f"  ⚠  Radius < Rs — this IS a black hole!"
                         if ratio < 1 else
                         f"  (Body is {ratio:.2e}× larger than its Schwarzschild radius)")
            print(f"\n  Schwarzschild radius of '{b.name}': {rs:.4e} m  ({rs * 100:.4g} cm)")
            if note:
                print(f"  {note}")
            print()

        elif sub == 'lagrange':
            pri = self._get_body(kw.get('primary'))
            sec = self._get_body(kw.get('secondary'))
            if not pri or not sec: return
            pts  = toolkit.lagrange_points(pri, sec)
            sep  = (sec.position - pri.position).magnitude()
            print(f"\n  Lagrange Points: '{pri.name}' – '{sec.name}'  "
                  f"(separation: {sep:.4e} m)")
            print(f"  {'Point':<6} {'Position (m)':<44} Distance from primary")
            print(_ruler(72))
            for lbl, pos in pts.items():
                d = (pos - pri.position).magnitude()
                print(f"  {lbl:<6} {str(pos):<44} {d:.4e} m")
            print()

        elif sub == 'photon_sphere':
            b = self._get_body(kw.get('body'))
            if not b: return
            rph = toolkit.photon_sphere_radius(b)
            rs  = toolkit.schwarzschild_radius(b)
            print(f"\n  Photon sphere of '{b.name}': {rph:.4e} m  ({rph / 1e3:.4f} km)")
            print(f"    = 1.5 × Schwarzschild radius ({rs:.4e} m)")
            print()

        elif sub == 'isco':
            b = self._get_body(kw.get('body'))
            if not b: return
            spin = float(kw.get('spin', '0'))
            r_isco = toolkit.isco_radius(b, spin=spin)
            print(f"\n  ISCO of '{b.name}' (spin={spin}): {r_isco:.4e} m  ({r_isco / 1e3:.4f} km)")
            if spin == 0:
                print(f"    = 3 × Schwarzschild radius")
            print()

        elif sub == 'redshift':
            b = self._get_body(kw.get('body'))
            if not b: return
            r_emit = parse_scalar(kw.get('r_emit', '0'))
            r_obs  = parse_scalar(kw.get('r_obs', '0'))
            if r_emit <= 0 or r_obs <= 0:
                print("  [!] Both r_emit and r_obs are required and must be > 0.")
                return
            rs_data = toolkit.gravitational_redshift_at(r_emit, r_obs, b)
            print(f"\n  Gravitational Redshift")
            print(f"    Emitter at  : {r_emit:.4e} m")
            print(f"    Observer at : {r_obs:.4e} m")
            print(f"    Redshift z  : {rs_data['z']:.6f}")
            print(f"    λ_obs/λ_emit: {rs_data['wavelength_ratio']:.6f}")
            print(f"    f_obs/f_emit: {rs_data['frequency_ratio']:.6f}")
            print()

        elif sub == 'timedilation':
            b = self._get_body(kw.get('body'))
            if not b: return
            r = parse_scalar(kw.get('r', '0'))
            if r <= 0:
                print("  [!] 'r' is required and must be > 0."); return
            factor = toolkit.time_dilation_factor(r, b)
            print(f"\n  Time Dilation at r={r:.4e} m from '{b.name}'")
            print(f"    dτ/dt = {factor:.8f}")
            print(f"    Clock runs at {factor * 100:.4f}% of distant clock rate")
            if factor > 0:
                print(f"    1 hour local = {3600.0 / factor:.4f} s distant")
            print()

        else:
            print("  Available: escape_velocity | orbital_period | grav_force | "
                   "energy | schwarzschild | lagrange | photon_sphere | isco | "
                   "redshift | timedilation")
 
    # ── visualize ────────────────────────────────────────────────────────────
 
    def do_visualize(self, arg: str) -> None:
        """
View the recorded trajectory of the last simulation.
 
  visualize orbits   [type=interactive|replay] [output=<file>]
  visualize trajectory load=<path>  — Load recording from JSON
  visualize trajectory save=<path>  — Save recording to JSON
 
Types
  interactive : Renders a 3D Plotly HTML chart (default).
  replay      : Opens a Vispy OpenGL replay window.
 
Examples
  visualize orbits type=interactive output=orbits.html
  visualize orbits type=replay
  visualize trajectory save=my_run.json
        """
        tokens = shlex.split(arg)
        if not tokens:
            print("  Usage: visualize orbits | trajectory [...]")
            return
 
        sub = tokens[0].lower()
        try:
            kw = parse_kwargs(tokens[1:]) if len(tokens) > 1 else {}
        except ValueError as exc:
            print(f"  [!] {exc}"); return
 
        if sub == 'trajectory':
            if 'save' in kw:
                if not self._recorder:
                    print("  [!] No trajectory to save."); return
                self._recorder.save(kw['save'])
                print(f"  [+] Trajectory saved to '{kw['save']}'")
            elif 'load' in kw:
                self._recorder = TrajectoryRecorder.load(kw['load'])
                print(f"  [+] Trajectory loaded from '{kw['load']}'")
            return
 
        if sub != 'orbits':
            print(f"  [!] Unknown subcommand '{sub}'. Try: orbits | trajectory")
            return
 
        if not self._recorder or self._recorder.snapshot_count() == 0:
            print("  [!] No trajectory data. Run a simulation with 'visualize=on' first.")
            return
 
        viz_type = kw.get('type', 'interactive').lower()
        output   = kw.get('output', 'orbits.html')
 
        if viz_type == 'interactive':
            try:
                from astrolab.viz.plotly_viz import render_html
                render_html(self._recorder, output)
            except ImportError as exc:
                print(f"  [!] Error: {exc}")
        elif viz_type == 'replay':
            try:
                from astrolab.viz.vispy_viz import AstroLabReplay
                viewer = AstroLabReplay(self._recorder)
                print("  [*] Opening replay window...")
                viewer.run()
            except ImportError as exc:
                print(f"  [!] Error: {exc}")
        else:
            print(f"  [!] Unknown type '{viz_type}'. Use: interactive | replay")

    # ── export / import ──────────────────────────────────────────────────────

    def do_export(self, arg: str) -> None:
        """
Save simulation state to a JSON file.

  export state file=<path>
        """
        tokens = shlex.split(arg)
        if len(tokens) < 2 or tokens[0] != 'state':
            print("  Usage: export state file=<path>")
            return
        try:
            kw = parse_kwargs(tokens[1:])
        except ValueError as exc:
            print(f"  [!] {exc}"); return
        if 'file' not in kw:
            print("  [!] 'file' is required."); return
        if self.manager.export_state(kw['file']):
            print(f"  [+] State exported → '{kw['file']}'")

    def do_import(self, arg: str) -> None:
        """
Load simulation state from a JSON file.

  import state file=<path>
        """
        tokens = shlex.split(arg)
        if len(tokens) < 2 or tokens[0] != 'state':
            print("  Usage: import state file=<path>")
            return
        try:
            kw = parse_kwargs(tokens[1:])
        except ValueError as exc:
            print(f"  [!] {exc}"); return
        if 'file' not in kw:
            print("  [!] 'file' is required."); return
        if self.manager.import_state(kw['file']):
            print(f"  [+] State loaded ← '{kw['file']}'")
            # Re-sync engine to new state
            self._engine = SimulationEngine(self.manager.state)

    # ── clear ────────────────────────────────────────────────────────────────

    def do_clear(self, arg: str) -> None:
        """Reset the simulation — removes all bodies and resets time.  Usage: clear"""
        self.manager.clear()
        print("  [+] Simulation cleared.")

    # ── run (batch script) ───────────────────────────────────────────────────

    def do_run(self, arg: str) -> None:
        """
Execute commands from a batch script file.

  run <path/to/script.astro>

  Each line is treated as one CLI command.
  Empty lines and lines starting with '#' are ignored.
        """
        path = arg.strip()
        if not path:
            print("  Usage: run <script.astro>")
            return
        if not os.path.exists(path):
            print(f"  [!] File not found: {path!r}")
            return
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
        print(f"  [>] Running script '{path}' ({len(lines)} lines)")
        for lineno, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            print(f"\n  [script:{lineno}] {line}")
            self.onecmd(line)

    # ── blackhole ─────────────────────────────────────────────────────────────

    def do_blackhole(self, arg: str) -> None:
        """
General-relativistic black hole simulation.

  blackhole create   mass=<kg> [spin=<a/M>] [metric=schwarzschild|kerr]
  blackhole info
  blackhole geodesic type=timelike|null r=<r_M> [theta=<rad>] [phi=<rad>]
                     [ur=<val>] [utheta=<val>] [uphi=<val>]
                     [steps=<N>] [affine=<max_lambda>]
  blackhole lensing  impact=<b_M>
  blackhole redshift r_emit=<rM> r_obs=<rM>
  blackhole timedilation r=<rM> [v=<frac_c>]
  blackhole potential L=<L_M> [E=<E>] [type=timelike|null]
  blackhole shadow   [rays=<N>]

  Coordinates are in units of M (geometric mass of the black hole).
  r=10 means 10×M from the singularity.

Examples
  blackhole create mass=1.989e30
  blackhole create mass=4.3e6 spin=0.998 metric=kerr
  blackhole info
  blackhole geodesic type=timelike r=10 uphi=0.02
  blackhole geodesic type=null r=50 ur=-1 uphi=0.1
  blackhole lensing impact=5.5
  blackhole redshift r_emit=3 r_obs=100
  blackhole timedilation r=2.5
  blackhole potential L=4 type=timelike
        """
        tokens = shlex.split(arg)
        if not tokens:
            print("  Usage: blackhole create | info | geodesic | lensing | "
                  "redshift | timedilation | potential | shadow")
            return

        sub = tokens[0].lower()
        try:
            kw = parse_kwargs(tokens[1:]) if len(tokens) > 1 else {}
        except ValueError as exc:
            print(f"  [!] {exc}")
            return

        if sub == 'create':
            self._bh_create(kw)
        elif sub == 'info':
            self._bh_info()
        elif sub == 'geodesic':
            self._bh_geodesic(kw)
        elif sub == 'lensing':
            self._bh_lensing(kw)
        elif sub == 'redshift':
            self._bh_redshift(kw)
        elif sub == 'timedilation':
            self._bh_timedilation(kw)
        elif sub == 'potential':
            self._bh_potential(kw)
        elif sub == 'shadow':
            self._bh_shadow(kw)
        else:
            print(f"  [!] Unknown subcommand '{sub}'.")
            print("  Available: create | info | geodesic | lensing | "
                  "redshift | timedilation | potential | shadow")

    def _ensure_gr_engine(self) -> bool:
        """Check that a GR engine is initialized."""
        if self._gr_engine is None:
            print("  [!] No black hole configured.  Use 'blackhole create' first.")
            return False
        return True

    def _bh_create(self, kw: dict) -> None:
        if 'mass' not in kw:
            print("  [!] 'mass' is required (in kg).")
            print("  Example: blackhole create mass=1.989e30")
            return

        try:
            from astrolab.engine.gr_engine import GRSimulationEngine, BlackHoleConfig
            from astrolab.physics.metrics import create_metric

            mass_kg = parse_scalar(kw['mass'])
            spin = float(kw.get('spin', '0'))
            metric_type = kw.get('metric', 'schwarzschild' if spin == 0 else 'kerr').lower()

            if spin > 0 and metric_type == 'schwarzschild':
                metric_type = 'kerr'
                print("  [i] Spin > 0 detected — switching to Kerr metric.")

            config = BlackHoleConfig(
                mass_kg=mass_kg,
                spin=spin,
                metric_type=metric_type,
            )

            self._bh_config = config
            self._gr_engine = GRSimulationEngine.from_config(config)

            print(f"\n  [+] Black hole created ({metric_type.upper()} metric)")
            print(f"      Mass = {mass_kg:.4e} kg  ({mass_kg / 1.989e30:.4f} M☉)")
            print(f"      M_geom = {config.mass_geometric:.4e} m")
            if spin > 0:
                print(f"      Spin a/M = {spin:.4f}")
            print(f"      Schwarzschild radius = {config.schwarzschild_radius_km:.4f} km")
            print()

        except Exception as exc:
            print(f"  [!] Failed to create black hole: {exc}")

    def _bh_info(self) -> None:
        if not self._ensure_gr_engine():
            return
        print(self._gr_engine.format_info())
        print()

    def _bh_geodesic(self, kw: dict) -> None:
        if not self._ensure_gr_engine():
            return

        try:
            import math as _math

            ptype = kw.get('type', 'timelike').lower()
            r = float(kw.get('r', '10'))
            theta = float(kw.get('theta', str(_math.pi / 2)))
            phi = float(kw.get('phi', '0'))
            ur = float(kw.get('ur', '0'))
            utheta = float(kw.get('utheta', '0'))
            uphi = float(kw.get('uphi', '0.02'))
            max_steps = int(kw.get('steps', '50000'))
            max_affine = float(kw.get('affine', '500'))

            if r <= self._gr_engine.metric.event_horizon()[0] / self._gr_engine.M:
                print(f"  [!] Starting radius r={r}M is inside the event horizon. "
                      f"Must be > {self._gr_engine.metric.event_horizon()[0] / self._gr_engine.M:.2f}M.")
                return

            print(f"\n  Tracing {ptype} geodesic...")
            print(f"    r₀ = {r}M,  θ₀ = {theta:.4f},  φ₀ = {phi:.4f}")
            print(f"    u^r = {ur},  u^θ = {utheta},  u^φ = {uphi}")
            print(f"    Max steps: {max_steps:,},  Max λ: {max_affine}M")
            print()

            result = self._gr_engine.trace_geodesic(
                r=r, theta=theta, phi=phi,
                ur=ur, utheta=utheta, uphi=uphi,
                particle_type=ptype,
                max_steps=max_steps,
                max_affine=max_affine,
            )

            # Print results
            M = self._gr_engine.M
            print(f"  ┌─ Geodesic Result")
            print(f"  │  Termination    : {result.termination_reason.value}")
            print(f"  │  Steps taken    : {result.steps_taken:,}")
            print(f"  │  Trajectory pts : {len(result.trajectory)}")
            print(f"  │  Coord. time    : {result.coordinate_time / M:.4f} M")
            if ptype == 'timelike':
                print(f"  │  Proper time    : {result.proper_time / M:.4f} M")
            print(f"  │")
            print(f"  │  ── Orbital Parameters ──")
            print(f"  │  r_min (periapsis)  : {result.orbital_params.get('r_min', 0) / M:.4f} M")
            print(f"  │  r_max (apoapsis)   : {result.orbital_params.get('r_max', 0) / M:.4f} M")
            print(f"  │  Total Δφ           : {result.orbital_params.get('total_delta_phi', 0):.4f} rad  "
                  f"({_math.degrees(result.orbital_params.get('total_delta_phi', 0)):.2f}°)")
            if 'precession_per_orbit' in result.orbital_params:
                prec = result.orbital_params['precession_per_orbit']
                print(f"  │  Precession/orbit   : {prec:.6f} rad  "
                      f"({_math.degrees(prec) * 3600:.2f} arcsec)")
            print(f"  │")
            print(f"  │  ── Constants of Motion ──")
            ci = result.constants_initial
            cf = result.constants_final
            print(f"  │  E (initial)  : {ci.get('E', 0):+.8f}")
            print(f"  │  E (final)    : {cf.get('E', 0):+.8f}  "
                  f"(drift: {abs(ci.get('E', 0) - cf.get('E', 0)):.2e})")
            print(f"  │  L (initial)  : {ci.get('L', 0):+.8f}")
            print(f"  │  L (final)    : {cf.get('L', 0):+.8f}  "
                  f"(drift: {abs(ci.get('L', 0) - cf.get('L', 0)):.2e})")

            if result.events:
                print(f"  │")
                print(f"  │  ── Events ({len(result.events)}) ──")
                for ev in result.events[:15]:  # Cap display at 15 events
                    print(f"  │  λ={ev.affine_param/M:.4f}M  r={ev.r/M:.4f}M  {ev.description}")
                if len(result.events) > 15:
                    print(f"  │  ... and {len(result.events) - 15} more events")

            print(f"  └─")
            print()

        except Exception as exc:
            print(f"  [!] Geodesic trace failed: {exc}")
            import traceback
            traceback.print_exc()

    def _bh_lensing(self, kw: dict) -> None:
        if not self._ensure_gr_engine():
            return

        impact = float(kw.get('impact', kw.get('b', '5.196')))
        result = self._gr_engine.compute_lensing(impact)

        import math as _math
        print(f"\n  ┌─ Gravitational Lensing")
        print(f"  │  Impact parameter   : {result['impact_parameter_M']:.4f} M")
        print(f"  │  Critical b         : {result['critical_impact_parameter_M']:.4f} M")
        print(f"  │  Captured?          : {'YES — falls into BH' if result['captured'] else 'NO — deflected'}")
        print(f"  │  Weak-field deflect.: {result['weak_field_deflection_deg']:.4f}°  "
              f"({result['weak_field_deflection_rad']:.6f} rad)")
        if not result['captured']:
            einstein = 4.0 / impact  # 4M/b in units where we already divide by M
            print(f"  │  Einstein prediction: {_math.degrees(einstein):.4f}°  (4M/b)")
        print(f"  └─")
        print()

    def _bh_redshift(self, kw: dict) -> None:
        if not self._ensure_gr_engine():
            return

        r_emit = float(kw.get('r_emit', '3'))
        r_obs = float(kw.get('r_obs', '1000'))

        result = self._gr_engine.compute_redshift(r_emit, r_obs)

        print(f"\n  ┌─ Gravitational Redshift")
        print(f"  │  Emitter at   : {r_emit} M")
        print(f"  │  Observer at  : {r_obs} M")
        print(f"  │  Redshift z   : {result.get('z', float('inf')):.8f}")
        print(f"  │  λ_obs/λ_emit : {result.get('wavelength_ratio', 0):.8f}")
        print(f"  │  f_obs/f_emit : {result.get('frequency_ratio', 0):.8f}")
        if result.get('z', 0) > 0:
            print(f"  │  A 500nm photon → {500.0 * result.get('wavelength_ratio', 1):.1f}nm")
        print(f"  └─")
        print()

    def _bh_timedilation(self, kw: dict) -> None:
        if not self._ensure_gr_engine():
            return

        r = float(kw.get('r', '3'))
        v = float(kw.get('v', '0'))

        result = self._gr_engine.compute_time_dilation(r, v_tangential=v)

        print(f"\n  ┌─ Gravitational Time Dilation")
        print(f"  │  Position     : {r} M from singularity")
        if v > 0:
            print(f"  │  Velocity     : {v:.4f} c (tangential)")
        print(f"  │  dτ/dt        : {result['factor']:.10f}")
        print(f"  │  Clock rate   : {result['clock_rate_percent']:.6f}% of distant clock")
        if result['factor'] > 0:
            hrs = result['one_hour_local_equals_s'] / 3600.0
            print(f"  │  1 hour here  = {hrs:.6f} hours for distant observer")
        else:
            print(f"  │  Time is frozen at the event horizon")
        print(f"  └─")
        print()

    def _bh_potential(self, kw: dict) -> None:
        if not self._ensure_gr_engine():
            return

        L = float(kw.get('l', kw.get('L', '4')))
        E = float(kw.get('e', kw.get('E', '1')))
        ptype = kw.get('type', 'timelike').lower()
        r_min = float(kw.get('r_min', '2.5'))
        r_max = float(kw.get('r_max', '30'))

        r_arr, V_arr = self._gr_engine.compute_effective_potential(
            L=L, E=E, r_min=r_min, r_max=r_max,
            particle_type=ptype,
        )

        print(f"\n  ┌─ Effective Potential ({ptype})")
        print(f"  │  Angular momentum L = {L} M")
        print(f"  │  Energy E = {E}")
        print(f"  │")
        print(f"  │  {'r/M':<10} {'V_eff':<20}")
        print(f"  │  {'─' * 30}")

        # Print sampled points
        step = max(1, len(r_arr) // 20)
        for i in range(0, len(r_arr), step):
            print(f"  │  {r_arr[i]:<10.2f} {V_arr[i]:<20.8f}")

        # Find extrema (potential barrier, well)
        import numpy as _np
        dV = _np.diff(V_arr)
        sign_changes = _np.where(_np.diff(_np.sign(dV)))[0]
        if len(sign_changes) > 0:
            print(f"  │")
            print(f"  │  ── Critical Points ──")
            for idx in sign_changes:
                cp_type = "maximum (unstable)" if V_arr[idx+1] > V_arr[idx] else "minimum (stable)"
                print(f"  │  r = {r_arr[idx+1]:.4f} M  V = {V_arr[idx+1]:.8f}  ({cp_type})")

        print(f"  └─")
        print()

    def _bh_shadow(self, kw: dict) -> None:
        if not self._ensure_gr_engine():
            return

        n_rays = int(kw.get('rays', '36'))

        boundary = self._gr_engine.compute_shadow_boundary(n_points=n_rays)

        from astrolab.physics.observables import critical_impact_parameter
        b_crit = critical_impact_parameter(self._gr_engine.metric) / self._gr_engine.M

        print(f"\n  ┌─ Black Hole Shadow")
        print(f"  │  Shadow radius  : {b_crit:.4f} M  (critical impact parameter)")
        print(f"  │  Boundary pts   : {len(boundary)}")
        if self._gr_engine.metric.spin > 0:
            print(f"  │  Shape          : Asymmetric (Kerr, spin={self._gr_engine.metric.spin:.4f})")
        else:
            print(f"  │  Shape          : Circular (Schwarzschild)")
        print(f"  │")
        print(f"  │  {'α (M)':<12} {'β (M)':<12}")
        print(f"  │  {'─' * 24}")
        step = max(1, len(boundary) // 12)
        for i in range(0, len(boundary), step):
            alpha, beta = boundary[i]
            print(f"  │  {alpha:<12.4f} {beta:<12.4f}")
        print(f"  └─")
        print()

    # ── exit ─────────────────────────────────────────────────────────────────

    def do_exit(self, arg: str) -> bool:
        """Exit AstroLab CLI."""
        print("  Ad astra! 🚀")
        return True

    def do_quit(self, arg: str) -> bool:
        """Alias for exit."""
        return self.do_exit(arg)

    def do_EOF(self, arg: str) -> bool:
        print()
        return self.do_exit(arg)

    # ── misc ─────────────────────────────────────────────────────────────────

    def default(self, line: str) -> None:
        print(f"  [?] Unknown command: '{line}'.  Type 'help' for commands.")

    def emptyline(self) -> None:
        pass  # Suppress repeating last command on blank input

    # ── internal ─────────────────────────────────────────────────────────────

    def _get_body(self, name: Optional[str]) -> Optional[CelestialBody]:
        if not name:
            print("  [!] Body name required.")
            return None
        b = self.manager.get_body(name)
        if b is None:
            print(f"  [-] Body '{name}' not found.  Use 'show state' to list bodies.")
        return b
