"""
Numerical Integrators
=====================
Interchangeable integration schemes for advancing an N-body system in time.

All integrators share a common interface defined by ``BaseIntegrator``.
They operate directly on a list of ``CelestialBody`` objects, mutating
``position`` and ``velocity`` in-place each timestep.

Available integrators
---------------------
EulerIntegrator          — 1st-order, explicit, O(dt) error/step.
                           Fast, but accumulates energy error rapidly.
                           Suitable only for demos or very short runs.

RK4Integrator            — 4th-order Runge-Kutta, explicit.
                           Default choice: excellent short-to-medium
                           term accuracy with manageable computation cost.

VelocityVerletIntegrator — 2nd-order, symplectic (time-reversible).
                           Conserves the modified Hamiltonian exactly →
                           energy does not drift in long runs.
                           Recommended for orbit stability studies.

Usage
-----
    integrator = RK4Integrator()
    integrator.step(bodies, dt=3600.0)   # advance one hour
"""

from __future__ import annotations

import abc
from typing import Callable, List

from astrolab.core.models import CelestialBody, Vector3D
from astrolab.physics.gravity import compute_accelerations


# ---------------------------------------------------------------------------
# Abstract base class — defines the contract for every integrator
# ---------------------------------------------------------------------------

class BaseIntegrator(abc.ABC):
    """
    Abstract base for all numerical integrators.

    Subclasses must implement ``step(bodies, dt)`` which advances the
    list of bodies by one timestep *in-place*.
    """

    # Human-readable identifier used in CLI and state files
    name: str = "base"

    @abc.abstractmethod
    def step(self, bodies: List[CelestialBody], dt: float) -> None:
        """
        Advance all bodies by one timestep dt [seconds].

        Parameters
        ----------
        bodies : List[CelestialBody]
            Bodies to integrate.  Mutated in-place.
        dt     : float
            Timestep in seconds.
        """

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    def __repr__(self) -> str:
        return self.__str__()


# ---------------------------------------------------------------------------
# Euler
# ---------------------------------------------------------------------------

class EulerIntegrator(BaseIntegrator):
    """
    Explicit (forward) Euler method — O(dt) local truncation error.

    Algorithm::

        a_n     = F(r_n, v_n) / m
        r_{n+1} = r_n + v_n · dt          ← uses velocity BEFORE update
        v_{n+1} = v_n + a_n · dt

    Note: Uses the velocity at the *start* of the timestep to update
    position (standard forward Euler).  Symplectic Euler (which updates
    v first, then uses the new v for r) would give marginally better
    energy behaviour, but is not this class's intent.
    """

    name = "euler"

    def step(self, bodies: List[CelestialBody], dt: float) -> None:
        accels = compute_accelerations(bodies)
        for i, body in enumerate(bodies):
            # Save current velocity before any mutations
            v_old = body.velocity
            # Update position using old velocity (standard forward Euler)
            body.position = body.position + v_old * dt
            # Update velocity using acceleration at old position
            body.velocity = v_old + accels[i] * dt


# ---------------------------------------------------------------------------
# RK4
# ---------------------------------------------------------------------------

class RK4Integrator(BaseIntegrator):
    """
    Classic 4th-order Runge-Kutta method — O(dt⁴) local truncation error.

    Evaluates the force/derivative four times per timestep to construct
    a weighted average that is exact to 4th order in dt.

    State vector: (r⃗, v⃗) for all bodies simultaneously.

    Algorithm (per body)::

        k1 = f(y_n)
        k2 = f(y_n + dt/2 · k1)
        k3 = f(y_n + dt/2 · k2)
        k4 = f(y_n + dt   · k3)
        y_{n+1} = y_n + dt/6 · (k1 + 2k2 + 2k3 + k4)

    where y = (r, v) and f(y) = (v, a(r)).
    """

    name = "rk4"

    def step(self, bodies: List[CelestialBody], dt: float) -> None:
        n = len(bodies)
        half_dt = dt * 0.5

        # Snapshot the state at time t (immutable reference points)
        r0 = [b.position for b in bodies]
        v0 = [b.velocity for b in bodies]

        def derivs(
            positions: List[Vector3D],
            velocities: List[Vector3D],
        ):
            """Return (dr/dt, dv/dt) = (v, a) at the given state."""
            # Build temporary bodies at the trial state for force evaluation
            trial = [
                CelestialBody(
                    name=bodies[i].name,
                    mass=bodies[i].mass,
                    position=positions[i],
                    velocity=velocities[i],
                )
                for i in range(n)
            ]
            accels = compute_accelerations(trial)
            return velocities, accels  # dr/dt = v,  dv/dt = a

        # ── k1 at t ──────────────────────────────────────────────────────────
        dr1, dv1 = derivs(r0, v0)

        # ── k2 at t + dt/2 ───────────────────────────────────────────────────
        r2 = [r0[i] + dr1[i] * half_dt for i in range(n)]
        v2 = [v0[i] + dv1[i] * half_dt for i in range(n)]
        dr2, dv2 = derivs(r2, v2)

        # ── k3 at t + dt/2 (using k2 slope) ──────────────────────────────────
        r3 = [r0[i] + dr2[i] * half_dt for i in range(n)]
        v3 = [v0[i] + dv2[i] * half_dt for i in range(n)]
        dr3, dv3 = derivs(r3, v3)

        # ── k4 at t + dt ──────────────────────────────────────────────────────
        r4 = [r0[i] + dr3[i] * dt for i in range(n)]
        v4 = [v0[i] + dv3[i] * dt for i in range(n)]
        dr4, dv4 = derivs(r4, v4)

        # ── Weighted average update ───────────────────────────────────────────
        sixth_dt = dt / 6.0
        for i in range(n):
            bodies[i].position = r0[i] + (dr1[i] + dr2[i] * 2 + dr3[i] * 2 + dr4[i]) * sixth_dt
            bodies[i].velocity = v0[i] + (dv1[i] + dv2[i] * 2 + dv3[i] * 2 + dv4[i]) * sixth_dt


# ---------------------------------------------------------------------------
# Velocity Verlet (Störmer–Verlet)
# ---------------------------------------------------------------------------

class VelocityVerletIntegrator(BaseIntegrator):
    """
    Velocity Verlet (Störmer–Verlet) method — 2nd-order, symplectic.

    Being symplectic means the integrator preserves a modified Hamiltonian
    exactly, so total energy does not drift secularly even over millions
    of steps — the gold standard for long-term orbital mechanics.

    Algorithm::

        a_n       = F(r_n) / m
        r_{n+1}   = r_n + v_n · dt + ½ · a_n · dt²
        a_{n+1}   = F(r_{n+1}) / m
        v_{n+1}   = v_n + ½ · (a_n + a_{n+1}) · dt

    Two force evaluations per step (vs. four for RK4), making it
    competitive in cost while offering superior long-term stability.
    """

    name = "verlet"

    def step(self, bodies: List[CelestialBody], dt: float) -> None:
        half_dt    = dt * 0.5
        half_dt_sq = 0.5 * dt * dt

        # ── Step 1: accelerations at r_n ─────────────────────────────────────
        a_curr = compute_accelerations(bodies)

        # ── Step 2: update positions: r_{n+1} = r_n + v_n·dt + ½a_n·dt² ─────
        for i, body in enumerate(bodies):
            body.position = body.position + body.velocity * dt + a_curr[i] * half_dt_sq

        # ── Step 3: accelerations at r_{n+1} ─────────────────────────────────
        a_next = compute_accelerations(bodies)

        # ── Step 4: update velocities: v_{n+1} = v_n + ½(a_n + a_{n+1})·dt ──
        for i, body in enumerate(bodies):
            body.velocity = body.velocity + (a_curr[i] + a_next[i]) * half_dt


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

#: All built-in integrators, keyed by their CLI name.
INTEGRATORS: dict[str, BaseIntegrator] = {
    cls.name: cls()
    for cls in (EulerIntegrator, RK4Integrator, VelocityVerletIntegrator)
}


def get_integrator(name: str) -> BaseIntegrator:
    """
    Retrieve an integrator instance by name.

    Parameters
    ----------
    name : str  One of 'euler', 'rk4', 'verlet'.

    Raises
    ------
    ValueError if the name is not recognised.
    """
    integrator = INTEGRATORS.get(name.lower())
    if integrator is None:
        available = list(INTEGRATORS.keys())
        raise ValueError(
            f"Unknown integrator '{name}'. Available: {available}"
        )
    return integrator
