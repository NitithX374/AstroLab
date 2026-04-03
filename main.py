"""
AstroLab CLI — Entry Point
==========================

Usage
-----
  Interactive REPL:
      python main.py

  One-shot command:
      python main.py  "simulate dt=3600 steps=8760"

  Built-in demo (Earth-Sun system, 1 simulated year):
      python main.py --demo

  Run a batch script:
      python main.py --script examples/earth_sun.astro
"""

from __future__ import annotations

import sys

# ── Windows: reconfigure stdout/stderr to UTF-8 so box-drawing characters
#             and emoji print correctly regardless of the active code page.
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import sys

from astrolab.cli.parser import AstroLabCLI
from astrolab.core.models import CelestialBody, SimulationState, Vector3D
from astrolab.engine.simulator import SimulationEngine
from astrolab.state.manager import StateManager
from astrolab.physics import toolkit
from astrolab.physics.integrators import INTEGRATORS


# ---------------------------------------------------------------------------
# Demo simulation — Earth-Sun system
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """
    Demonstrate AstroLab with an Earth-Sun two-body simulation.

    Physical data (SI):
      Sun   : M = 1.989e30 kg,  R = 6.96e8 m
      Earth : M = 5.972e24 kg,  R = 6.371e6 m,
              a = 1 AU = 1.496e11 m,  v_orb ≈ 29 780 m/s
    """

    print("\n" + "=" * 65)
    print("  AstroLab Demo — Earth-Sun System (1 simulated year, RK4)")
    print("=" * 65)

    # ── Build state ──────────────────────────────────────────────────────────
    sun = CelestialBody(
        name="sun",
        mass=1.989e30,
        position=Vector3D(0.0, 0.0, 0.0),
        velocity=Vector3D(0.0, 0.0, 0.0),
        radius=6.96e8,
        body_type="star",
        color="yellow",
    )
    earth = CelestialBody(
        name="earth",
        mass=5.972e24,
        position=Vector3D(1.496e11, 0.0, 0.0),
        velocity=Vector3D(0.0, 29_780.0, 0.0),
        radius=6.371e6,
        body_type="planet",
        color="blue",
    )

    state = SimulationState(
        bodies=[sun, earth],
        dt=3_600.0,       # 1-hour timestep
        integrator="rk4",
    )

    manager = StateManager(state)
    engine  = SimulationEngine(state)

    # ── Pre-flight analytics ──────────────────────────────────────────────────
    print("\n  Initial State")
    print(f"  {'Body':<10} {'Mass (kg)':<14}  {'Position (m)':<36}  Velocity (m/s)")
    print("  " + "─" * 80)
    for b in state.bodies:
        print(f"  {b.name:<10} {b.mass:<14.4e}  {str(b.position):<36}  {b.velocity}")

    print("\n  Pre-flight Calculations")
    print("  " + "─" * 60)
    ve = toolkit.escape_velocity(earth)
    T  = toolkit.orbital_period(earth, sun)
    F, _ = toolkit.gravitational_force(earth, sun)
    rs = toolkit.schwarzschild_radius(sun)
    e0 = toolkit.total_system_energy(state.bodies)
    print(f"    Earth escape velocity   : {ve / 1e3:.4f} km/s")
    print(f"    Earth orbital period    : {T / 86_400:.2f} days  ({T / (365.25*86_400):.4f} years)")
    print(f"    Earth-Sun gravity force : {F:.4e} N")
    print(f"    Sun Schwarzschild radius: {rs:.4f} m  ({rs * 100:.2f} cm)")
    print(f"    Initial system energy   : {e0['total']:.4e} J")

    # ── Lagrange points ───────────────────────────────────────────────────────
    lpts = toolkit.lagrange_points(sun, earth)
    print("\n  Sun-Earth Lagrange Points")
    print(f"  {'Point':<6} {'Distance from Sun (m)'}")
    print("  " + "─" * 40)
    for lbl, pos in lpts.items():
        d = (pos - sun.position).magnitude()
        print(f"  {lbl:<6} {d:.4e} m  ({d / 1.496e11:.4f} AU)")

    # ── Simulate ──────────────────────────────────────────────────────────────
    STEPS = 8_760   # 8760 × 3600 s ≈ 1 year

    print(f"\n  Simulating {STEPS:,} steps × 3600 s (1 year) ...")

    bar_width = 44
    last_pct  = [-1]

    def progress(current: int, total: int) -> None:
        pct    = int(current / total * 100)
        filled = int(bar_width * current / total)
        if pct != last_pct[0]:
            print(f"\r  [{'█' * filled}{'░' * (bar_width - filled)}] {pct:3d}%",
                  end='', flush=True)
            last_pct[0] = pct

    result = engine.run_with_monitor(
        steps=STEPS,
        collision_detection=True,
        energy_interval=100,
        progress_callback=progress,
    )
    print(f"\r  [{'█' * bar_width}] 100%\n")

    # ── Post-run analytics ────────────────────────────────────────────────────
    ef = toolkit.total_system_energy(state.bodies)
    drift = abs((ef['total'] - e0['total']) / e0['total']) * 100 if e0['total'] != 0 else 0.0
    earth_final = manager.get_body('earth')

    print("  Post-Simulation State")
    print("  " + "─" * 60)
    print(f"    Earth final position    : {earth_final.position}")
    print(f"    Earth final speed       : {earth_final.speed:.2f} m/s")
    print(f"    Earth-Sun distance      : {earth_final.distance_from_origin:.4e} m")
    print(f"    Elapsed sim time        : {result.elapsed_time:.4e} s  "
          f"({result.elapsed_time / 86_400:.2f} days)")
    print(f"    Final system energy     : {ef['total']:.4e} J")
    print(f"    Energy drift (RK4)      : {drift:.6f}%")

    if result.collisions:
        print(f"\n  Collisions: {len(result.collisions)}")
        for ev in result.collisions:
            print(f"    {ev.absorbed} absorbed by {ev.survivor}")
    else:
        print("\n  No collisions detected.")

    print("\n" + "=" * 65)
    print("  Demo complete.  Run 'python main.py' for the interactive CLI.")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="astrolab",
        description="AstroLab CLI — N-body astrophysics simulation platform",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the built-in Earth-Sun demo simulation and exit.",
    )
    parser.add_argument(
        "--script",
        metavar="FILE",
        help="Execute commands from a batch script file and exit.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        help='Single CLI command to execute, then exit (e.g. "show state").',
    )

    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    cli = AstroLabCLI()

    if args.script:
        cli.onecmd(f"run {args.script}")
        return

    if args.command:
        cli.onecmd(args.command)
        return

    # ── Default: interactive REPL ────────────────────────────────────────────
    try:
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\n  Interrupted.  Ad astra! 🚀")


if __name__ == "__main__":
    main()
