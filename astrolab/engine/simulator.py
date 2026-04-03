"""
Simulation Engine
=================
Drives the N-body time-step loop.

The engine is **decoupled from any I/O or CLI layer** — it operates purely
on a ``SimulationState`` object using a chosen integrator strategy.

Public API
----------
SimulationEngine.step()              — advance the state by one timestep
SimulationEngine.run(steps)          — advance by N timesteps
SimulationEngine.run_with_monitor()  — run with energy/collision callbacks

Collision detection
-------------------
When enabled (the default), after every integration step the engine
checks every pair of bodies for overlap (|r₁-r₂| < R₁+R₂).  Colliding
bodies are merged inelastically:

* Momentum conservation:  p⃗ = m₁v⃗₁ + m₂v⃗₂
* Volume conservation:    R_new = (R₁³ + R₂³)^(1/3)
* The smaller body is removed from the state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from astrolab.core.models import CelestialBody, SimulationState, Vector3D
from astrolab.physics.gravity import compute_accelerations
from astrolab.physics.integrators import BaseIntegrator, get_integrator
from astrolab.physics.toolkit import total_system_energy


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CollisionEvent:
    """Record of a merger between two bodies."""
    time:      float   # simulation time [s] when collision occurred
    survivor:  str     # name of the body that absorbed the other
    absorbed:  str     # name of the body that was removed
    new_mass:  float   # combined mass [kg]
    new_radius: float  # volume-conserving merged radius [m]


@dataclass
class RunResult:
    """Returned by ``run()`` and ``run_with_monitor()``."""
    elapsed_time: float                    # total simulated time [s]
    steps_taken:  int                      # number of steps executed
    collisions:   List[CollisionEvent] = field(default_factory=list)
    energy_log:   List[Dict]          = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    N-body simulation engine.

    Parameters
    ----------
    state : SimulationState
        The mutable simulation state.  The engine modifies it in-place.
    """

    def __init__(self, state: SimulationState) -> None:
        self.state: SimulationState = state
        self._integrator: BaseIntegrator = get_integrator(state.integrator)

    # ── Integrator management ────────────────────────────────────────────────

    @property
    def integrator(self) -> BaseIntegrator:
        return self._integrator

    @integrator.setter
    def integrator(self, name_or_instance) -> None:
        if isinstance(name_or_instance, str):
            self._integrator = get_integrator(name_or_instance)
            self.state.integrator = name_or_instance
        elif isinstance(name_or_instance, BaseIntegrator):
            self._integrator = name_or_instance
            self.state.integrator = name_or_instance.name
        else:
            raise TypeError("integrator must be a name string or BaseIntegrator instance.")

    # ── Single step ──────────────────────────────────────────────────────────

    def step(self, collision_detection: bool = True) -> List[CollisionEvent]:
        """
        Advance the simulation by exactly one timestep (``state.dt``).

        Parameters
        ----------
        collision_detection : bool
            When True, check for and merge overlapping bodies after the
            position/velocity update.

        Returns
        -------
        List[CollisionEvent]  — any collisions that occurred this step.
        """
        bodies = self.state.bodies
        dt     = self.state.dt

        # Integrate positions and velocities
        self._integrator.step(bodies, dt)

        # Advance time bookkeeping
        self.state.time += dt
        self.state.step += 1

        # Optional collision detection
        events: List[CollisionEvent] = []
        if collision_detection:
            events = self._detect_and_merge(bodies)

        return events

    # ── Multi-step run ───────────────────────────────────────────────────────

    def run(
        self,
        steps: int,
        collision_detection: bool = True,
    ) -> RunResult:
        """
        Advance the simulation by ``steps`` timesteps.

        Parameters
        ----------
        steps               : int   Number of timesteps to run.
        collision_detection : bool  Enable inelastic collision merging.

        Returns
        -------
        RunResult — summary including any collision events.
        """
        collisions: List[CollisionEvent] = []
        for _ in range(steps):
            events = self.step(collision_detection=collision_detection)
            collisions.extend(events)

        return RunResult(
            elapsed_time=self.state.dt * steps,
            steps_taken=steps,
            collisions=collisions,
        )

    def run_with_monitor(
        self,
        steps: int,
        collision_detection: bool = True,
        energy_interval: int = 0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> RunResult:
        """
        Run the simulation with optional energy monitoring and progress reporting.

        Parameters
        ----------
        steps               : int   Number of timesteps.
        collision_detection : bool  Inelastic collision merging.
        energy_interval     : int   Log energy every N steps (0 = disabled).
        progress_callback   : callable(current_step, total_steps)
                              Called after each step for progress reporting.

        Returns
        -------
        RunResult — includes energy_log list of dicts:
            {'step': int, 'time': float, 'kinetic': float,
             'potential': float, 'total': float}
        """
        collisions: List[CollisionEvent] = []
        energy_log: List[Dict] = []

        for s in range(steps):
            events = self.step(collision_detection=collision_detection)
            collisions.extend(events)

            if energy_interval > 0 and s % energy_interval == 0:
                e = total_system_energy(self.state.bodies)
                energy_log.append({
                    'step': self.state.step,
                    'time': self.state.time,
                    **e,
                })

            if progress_callback is not None:
                progress_callback(s + 1, steps)

        return RunResult(
            elapsed_time=self.state.dt * steps,
            steps_taken=steps,
            collisions=collisions,
            energy_log=energy_log,
        )

    # ── Collision detection ──────────────────────────────────────────────────

    def _detect_and_merge(
        self, bodies: List[CelestialBody]
    ) -> List[CollisionEvent]:
        """
        Detect body overlaps and merge colliding pairs inelastically.

        A collision is flagged when:
            |r₁ - r₂| < R₁ + R₂    (both radii must be > 0)

        Post-collision state:
            m_new    = m₁ + m₂
            v⃗_new   = (m₁v⃗₁ + m₂v⃗₂) / m_new    (momentum conservation)
            r⃗_new   = (m₁r⃗₁ + m₂r⃗₂) / m_new    (centre-of-mass position)
            R_new    = (R₁³ + R₂³)^(1/3)          (volume conservation)

        The smaller body is removed from ``self.state.bodies``.
        """
        events: List[CollisionEvent] = []
        i = 0
        while i < len(bodies):
            j = i + 1
            while j < len(bodies):
                b1, b2 = bodies[i], bodies[j]
                r1, r2 = b1.radius, b2.radius

                # Both bodies need a physical radius for collision to be defined
                if r1 > 0.0 and r2 > 0.0:
                    separation = (b1.position - b2.position).magnitude()
                    if separation < r1 + r2:
                        # Designate the more massive body as survivor
                        survivor, victim = (b1, b2) if b1.mass >= b2.mass else (b2, b1)

                        new_mass = survivor.mass + victim.mass
                        new_vel  = (
                            survivor.velocity * survivor.mass
                            + victim.velocity * victim.mass
                        ) / new_mass
                        new_pos  = (
                            survivor.position * survivor.mass
                            + victim.position * victim.mass
                        ) / new_mass
                        new_rad  = (survivor.radius ** 3 + victim.radius ** 3) ** (1.0 / 3.0)

                        survivor.mass     = new_mass
                        survivor.velocity = new_vel
                        survivor.position = new_pos
                        survivor.radius   = new_rad

                        events.append(CollisionEvent(
                            time=self.state.time,
                            survivor=survivor.name,
                            absorbed=victim.name,
                            new_mass=new_mass,
                            new_radius=new_rad,
                        ))

                        bodies.remove(victim)
                        # Don't increment j — the slot just shifted down
                        continue
                j += 1
            i += 1

        return events
