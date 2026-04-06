"""
Geodesic Integrator
====================
Adaptive-step Runge-Kutta integrator for geodesic equations in curved
spacetime.  Supports both timelike (massive particles) and null (photon)
geodesics with constraint renormalization and event detection.

The integrator uses a **Dormand-Prince RK45** scheme:
  - 5th-order solution for advancing the state
  - 4th-order embedded solution for error estimation
  - Step-size adapted via local truncation error control

References
----------
- Dormand & Prince (1980), J. Comp. Appl. Math.
- Press et al., "Numerical Recipes", §17.2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from astrolab.physics.metrics import SpacetimeMetric
from astrolab.physics.christoffel import (
    geodesic_rhs_optimized,
    compute_norm,
    normalize_4velocity,
    compute_constants_of_motion,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

class ParticleType(Enum):
    """Type of test particle being traced."""
    TIMELIKE = "timelike"   # massive particle,  g_μν u^μ u^ν = −1
    NULL     = "null"       # photon,             g_μν u^μ u^ν =  0


class TerminationReason(Enum):
    """Why the geodesic integration was stopped."""
    HORIZON_CROSSED       = "horizon_crossed"
    PHOTON_SPHERE_CAPTURE = "photon_sphere_captured"
    ESCAPED               = "escaped"
    SINGULARITY           = "singularity_approached"
    MAX_STEPS             = "max_steps_reached"
    MAX_AFFINE            = "max_affine_reached"
    CONSTRAINT_VIOLATION  = "constraint_violation"
    USER_EVENT            = "user_event"


@dataclass
class GeodesicPoint:
    """A single point along a geodesic trajectory."""
    affine_param: float       # Affine parameter λ
    t:           float        # Coordinate time
    r:           float        # Radial coordinate
    theta:       float        # Polar angle
    phi:         float        # Azimuthal angle
    ut:          float        # dt/dλ
    ur:          float        # dr/dλ
    utheta:      float        # dθ/dλ
    uphi:        float        # dφ/dλ
    proper_time: float = 0.0  # Accumulated proper time (timelike only)
    norm:        float = 0.0  # Current g_μν u^μ u^ν (diagnostic)

    def to_dict(self) -> dict:
        return {
            'lambda': self.affine_param,
            't': self.t, 'r': self.r, 'theta': self.theta, 'phi': self.phi,
            'ut': self.ut, 'ur': self.ur, 'utheta': self.utheta, 'uphi': self.uphi,
            'proper_time': self.proper_time, 'norm': self.norm,
        }

    def cartesian(self) -> Tuple[float, float, float]:
        """Convert (r, θ, φ) to Cartesian (x, y, z) for visualization."""
        sin_th = math.sin(self.theta)
        return (
            self.r * sin_th * math.cos(self.phi),
            self.r * sin_th * math.sin(self.phi),
            self.r * math.cos(self.theta),
        )


@dataclass
class EventRecord:
    """Record of a physical event detected during integration."""
    event_type:   str          # e.g., "horizon_crossing", "turning_point"
    affine_param: float
    r:            float
    theta:        float
    description:  str


@dataclass
class GeodesicResult:
    """Complete result of a geodesic integration."""
    trajectory:         List[GeodesicPoint]
    termination_reason: TerminationReason
    events:             List[EventRecord]         = field(default_factory=list)
    proper_time:        float                     = 0.0
    coordinate_time:    float                     = 0.0
    steps_taken:        int                       = 0
    constants_initial:  Dict                      = field(default_factory=dict)
    constants_final:    Dict                      = field(default_factory=dict)
    orbital_params:     Dict                      = field(default_factory=dict)

    def periapsis(self) -> float:
        """Minimum radial coordinate along the trajectory."""
        if not self.trajectory:
            return float('inf')
        return min(p.r for p in self.trajectory)

    def apoapsis(self) -> float:
        """Maximum radial coordinate along the trajectory."""
        if not self.trajectory:
            return 0.0
        return max(p.r for p in self.trajectory)

    def total_deflection_angle(self) -> float:
        """Total change in φ coordinate (for lensing calculations)."""
        if len(self.trajectory) < 2:
            return 0.0
        return abs(self.trajectory[-1].phi - self.trajectory[0].phi)

    def cartesian_trajectory(self) -> List[Tuple[float, float, float]]:
        """Convert full trajectory to Cartesian coordinates."""
        return [p.cartesian() for p in self.trajectory]


# ─────────────────────────────────────────────────────────────────────────────
# Dormand-Prince RK45 Butcher tableau
# ─────────────────────────────────────────────────────────────────────────────

# Nodes (c_i)
_DP_C = np.array([0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])

# Runge-Kutta matrix (a_{ij})
_DP_A = np.array([
    [0,            0,           0,           0,         0,           0,      0],
    [1/5,          0,           0,           0,         0,           0,      0],
    [3/40,         9/40,        0,           0,         0,           0,      0],
    [44/45,       -56/15,       32/9,        0,         0,           0,      0],
    [19372/6561,  -25360/2187,  64448/6561, -212/729,   0,           0,      0],
    [9017/3168,   -355/33,      46732/5247,  49/176,   -5103/18656,  0,      0],
    [35/384,       0,           500/1113,    125/192,  -2187/6784,   11/84,  0],
])

# 5th-order weights (b_i) — these are the same as a[6,:]
_DP_B5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])

# 4th-order weights for error estimation
_DP_B4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])


# ─────────────────────────────────────────────────────────────────────────────
# Integrator
# ─────────────────────────────────────────────────────────────────────────────

class GeodesicIntegrator:
    """
    Adaptive-step RK45 geodesic integrator.

    Traces a test particle or photon through a curved spacetime defined
    by a SpacetimeMetric.  Automatically detects event horizon crossings,
    photon sphere capture, and particle escape.

    Parameters
    ----------
    metric         : SpacetimeMetric
    particle_type  : ParticleType                   TIMELIKE or NULL
    atol           : float             = 1e-10      Absolute error tolerance
    rtol           : float             = 1e-10      Relative error tolerance
    max_step       : float             = 1.0        Maximum step in affine parameter
    min_step       : float             = 1e-14      Minimum step (prevents stalling)
    escape_radius  : float             = 500.0      Stop if r > escape_radius (in M)
    normalize_every: int               = 10         Re-normalize 4-velocity every N steps
    record_every   : int               = 1          Store every Nth point in trajectory
    """

    def __init__(
        self,
        metric: SpacetimeMetric,
        particle_type: ParticleType = ParticleType.TIMELIKE,
        atol: float = 1e-10,
        rtol: float = 1e-10,
        max_step: float = 1.0,
        min_step: float = 1e-14,
        escape_radius: Optional[float] = None,
        normalize_every: int = 10,
        record_every: int = 1,
    ) -> None:
        self.metric = metric
        self.particle_type = particle_type
        self.atol = atol
        self.rtol = rtol
        self.max_step = max_step * metric.M   # Scale by M
        self.min_step = min_step * metric.M
        self.escape_radius = (escape_radius or 500.0) * metric.M
        self.normalize_every = normalize_every
        self.record_every = record_every

        # Target norm for the 4-velocity constraint
        self._target_norm = -1.0 if particle_type == ParticleType.TIMELIKE else 0.0

    def _rhs(self, state: np.ndarray) -> np.ndarray:
        """Evaluate geodesic RHS at a state."""
        return geodesic_rhs_optimized(state, self.metric)

    def _rk45_step(
        self, y: np.ndarray, h: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform one Dormand-Prince RK45 step.

        Returns
        -------
        y5   : 5th-order solution (used for stepping)
        y4   : 4th-order solution (used for error estimate)
        err  : scalar error estimate
        """
        k = np.zeros((7, 8), dtype=np.float64)

        k[0] = self._rhs(y)

        for i in range(1, 7):
            y_trial = y.copy()
            for j in range(i):
                y_trial += h * _DP_A[i, j] * k[j]
            # Clamp theta to (0, π) to avoid coordinate singularity
            y_trial[2] = max(1e-8, min(math.pi - 1e-8, y_trial[2]))
            k[i] = self._rhs(y_trial)

        # 5th-order solution
        y5 = y.copy()
        for i in range(7):
            y5 += h * _DP_B5[i] * k[i]

        # 4th-order solution (for error estimate)
        y4 = y.copy()
        for i in range(7):
            y4 += h * _DP_B4[i] * k[i]

        # Error estimate: |y5 - y4|
        err_vec = np.abs(y5 - y4)
        # Scale by tolerance: max(atol, rtol * |y|)
        scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y5))
        err = float(np.max(err_vec / scale))

        return y5, y4, err

    def integrate(
        self,
        initial_state: np.ndarray,
        max_affine: float = 1000.0,
        max_steps: int = 100_000,
        initial_step: float = 0.01,
    ) -> GeodesicResult:
        """
        Integrate a geodesic from the given initial state.

        Parameters
        ----------
        initial_state : np.ndarray, shape (8,)
            [t, r, θ, φ, u^t, u^r, u^θ, u^φ]
        max_affine    : float
            Maximum affine parameter λ (in units of M).
        max_steps     : int
            Maximum number of integration steps.
        initial_step  : float
            Initial step size in affine parameter (in units of M).

        Returns
        -------
        GeodesicResult
        """
        M = self.metric.M
        max_affine_scaled = max_affine * M
        h = initial_step * M

        # Normalize initial 4-velocity
        y = normalize_4velocity(initial_state.copy(), self.metric, self._target_norm)

        # Compute initial constants of motion
        constants_initial = compute_constants_of_motion(y, self.metric)

        # Horizon radius (outer)
        r_horizon = self.metric.event_horizon()[0]
        r_photon = self.metric.photon_sphere()
        horizon_buffer = r_horizon * 1.01  # Small buffer outside horizon

        trajectory: List[GeodesicPoint] = []
        events: List[EventRecord] = []
        lam = 0.0           # Affine parameter
        proper_time = 0.0
        step_count = 0
        prev_ur = y[5]      # Track sign changes in dr/dλ for turning points

        # Record initial point
        norm0 = compute_norm(y, self.metric)
        trajectory.append(GeodesicPoint(
            affine_param=lam, t=y[0], r=y[1], theta=y[2], phi=y[3],
            ut=y[4], ur=y[5], utheta=y[6], uphi=y[7],
            proper_time=proper_time, norm=norm0,
        ))

        termination = TerminationReason.MAX_STEPS

        while step_count < max_steps and lam < max_affine_scaled:
            # ── Adaptive RK45 step ───────────────────────────────────────────
            y5, _, err = self._rk45_step(y, h)

            if err < 1e-30:
                err = 1e-30  # Avoid division by zero

            if err <= 1.0:
                # Step accepted
                y = y5
                lam += h
                step_count += 1

                # Clamp theta to (0, π)
                y[2] = max(1e-8, min(math.pi - 1e-8, y[2]))

                # Wrap φ to [0, 2π) for cleanliness (optional)
                # y[3] = y[3] % (2.0 * math.pi)

                # ── Periodic 4-velocity renormalization ──────────────────────
                if step_count % self.normalize_every == 0:
                    y = normalize_4velocity(y, self.metric, self._target_norm)

                # ── Proper time accumulation (timelike only) ─────────────────
                if self.particle_type == ParticleType.TIMELIKE:
                    norm = compute_norm(y, self.metric)
                    if norm < 0:
                        proper_time += h * math.sqrt(-norm)

                # ── Record trajectory point ──────────────────────────────────
                if step_count % self.record_every == 0:
                    norm_val = compute_norm(y, self.metric)
                    trajectory.append(GeodesicPoint(
                        affine_param=lam, t=y[0], r=y[1], theta=y[2], phi=y[3],
                        ut=y[4], ur=y[5], utheta=y[6], uphi=y[7],
                        proper_time=proper_time, norm=norm_val,
                    ))

                # ── Event detection ──────────────────────────────────────────

                r_current = y[1]

                # 1. Horizon crossing
                if r_current <= horizon_buffer:
                    events.append(EventRecord(
                        event_type="horizon_crossing",
                        affine_param=lam, r=r_current, theta=y[2],
                        description=f"Crossed event horizon at r={r_current:.6f}M "
                                    f"(r_h={r_horizon:.6f}M)"
                    ))
                    termination = TerminationReason.HORIZON_CROSSED
                    break

                # 2. Singularity approach
                if r_current < 0.01 * M:
                    events.append(EventRecord(
                        event_type="singularity",
                        affine_param=lam, r=r_current, theta=y[2],
                        description=f"Approached singularity at r={r_current:.6e}M"
                    ))
                    termination = TerminationReason.SINGULARITY
                    break

                # 3. Escape
                if r_current > self.escape_radius:
                    events.append(EventRecord(
                        event_type="escape",
                        affine_param=lam, r=r_current, theta=y[2],
                        description=f"Escaped to r={r_current:.2f}M"
                    ))
                    termination = TerminationReason.ESCAPED
                    break

                # 4. Turning points (periapsis/apoapsis)
                ur_current = y[5]
                if prev_ur * ur_current < 0:
                    tp_type = "periapsis" if ur_current > 0 else "apoapsis"
                    events.append(EventRecord(
                        event_type=f"turning_point_{tp_type}",
                        affine_param=lam, r=r_current, theta=y[2],
                        description=f"{tp_type.capitalize()} at r={r_current:.6f}M"
                    ))
                prev_ur = ur_current

                # 5. Photon sphere (null geodesics near r_ph with low dr/dλ)
                if (self.particle_type == ParticleType.NULL and
                    abs(r_current - r_photon) < 0.05 * M and
                    abs(ur_current) < 1e-6):
                    events.append(EventRecord(
                        event_type="photon_sphere_capture",
                        affine_param=lam, r=r_current, theta=y[2],
                        description=f"Captured near photon sphere r={r_current:.6f}M "
                                    f"(r_ph={r_photon:.6f}M)"
                    ))
                    termination = TerminationReason.PHOTON_SPHERE_CAPTURE
                    break

                # 6. Ergosphere entry (Kerr)
                if hasattr(self.metric, 'ergosphere_radius'):
                    r_ergo = self.metric.ergosphere_radius(y[2])
                    if (r_current < r_ergo and r_current > r_horizon and
                        not any(e.event_type == "ergosphere_entry" for e in events)):
                        events.append(EventRecord(
                            event_type="ergosphere_entry",
                            affine_param=lam, r=r_current, theta=y[2],
                            description=f"Entered ergosphere at r={r_current:.6f}M "
                                        f"(r_ergo={r_ergo:.6f}M)"
                        ))

                # 7. Constraint violation check
                if step_count % (self.normalize_every * 10) == 0:
                    norm_check = compute_norm(y, self.metric)
                    expected = self._target_norm
                    if abs(norm_check - expected) > 0.01:
                        events.append(EventRecord(
                            event_type="constraint_warning",
                            affine_param=lam, r=r_current, theta=y[2],
                            description=f"4-velocity norm drifted to {norm_check:.6e} "
                                        f"(expected {expected})"
                        ))

            # ── Step-size control ────────────────────────────────────────────
            # Optimal step factor
            if err > 0:
                factor = 0.9 * err ** (-0.2)  # PI controller: h_new = h * S * err^(-1/5)
            else:
                factor = 5.0  # Very small error — grow aggressively

            factor = max(0.1, min(factor, 5.0))  # Clamp growth/shrink factor
            h = h * factor

            # Enforce bounds
            h = max(self.min_step, min(h, self.max_step))

            # Reduce step near horizon for accuracy
            dist_to_horizon = abs(y[1] - r_horizon)
            if dist_to_horizon < 2.0 * M:
                h = min(h, 0.01 * dist_to_horizon + self.min_step)

        else:
            if lam >= max_affine_scaled:
                termination = TerminationReason.MAX_AFFINE

        # Final record
        if trajectory[-1].affine_param != lam:
            norm_final = compute_norm(y, self.metric)
            trajectory.append(GeodesicPoint(
                affine_param=lam, t=y[0], r=y[1], theta=y[2], phi=y[3],
                ut=y[4], ur=y[5], utheta=y[6], uphi=y[7],
                proper_time=proper_time, norm=norm_final,
            ))

        # Final constants of motion
        constants_final = compute_constants_of_motion(y, self.metric)

        # Orbital parameters
        orbital_params = self._compute_orbital_params(trajectory, events)

        return GeodesicResult(
            trajectory=trajectory,
            termination_reason=termination,
            events=events,
            proper_time=proper_time,
            coordinate_time=y[0] - initial_state[0],
            steps_taken=step_count,
            constants_initial=constants_initial,
            constants_final=constants_final,
            orbital_params=orbital_params,
        )

    def _compute_orbital_params(
        self,
        trajectory: List[GeodesicPoint],
        events: List[EventRecord],
    ) -> Dict:
        """Extract orbital parameters from the integrated trajectory."""
        params: Dict = {}

        if not trajectory:
            return params

        r_values = [p.r for p in trajectory]
        params['r_min'] = min(r_values)
        params['r_max'] = max(r_values)

        # Total azimuthal angle traversed
        total_dphi = abs(trajectory[-1].phi - trajectory[0].phi)
        params['total_delta_phi'] = total_dphi

        # Count turning points (orbits)
        turning_points = [e for e in events if 'turning_point' in e.event_type]
        periapses = [e for e in turning_points if 'periapsis' in e.event_type]
        apoapses = [e for e in turning_points if 'apoapsis' in e.event_type]

        params['num_periapses'] = len(periapses)
        params['num_apoapses'] = len(apoapses)

        # Precession: if completed at least one orbit (peri → peri)
        if len(periapses) >= 2:
            dphi_orbit = abs(periapses[1].affine_param - periapses[0].affine_param)
            # The actual azimuthal advance between periapses
            phi_at_peri = []
            for p in trajectory:
                for per_event in periapses:
                    if abs(p.affine_param - per_event.affine_param) < 1e-10:
                        phi_at_peri.append(p.phi)
            if len(phi_at_peri) >= 2:
                precession = abs(phi_at_peri[1] - phi_at_peri[0]) - 2.0 * math.pi
                params['precession_per_orbit'] = precession

        return params
