"""
Event Detection
================
Physical event detection system for geodesic integration in black hole
spacetimes.  Provides a configurable set of event checkers that can be
attached to the geodesic integrator.

This module defines the event types, detection logic, and summary
formatting independently of the integrator, allowing reuse across
different engine configurations.

Event Categories
----------------
Critical (halt integration):
    - Horizon crossing
    - Singularity approach
    - Photon sphere capture (null geodesics)
    - Escape to infinity

Informational (log and continue):
    - Ergosphere entry/exit
    - ISCO crossing
    - Turning points (periapsis/apoapsis)
    - Constraint violation warnings
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from astrolab.physics.metrics import SpacetimeMetric


# ─────────────────────────────────────────────────────────────────────────────
# Event types
# ─────────────────────────────────────────────────────────────────────────────

class EventSeverity(Enum):
    """Whether an event should halt integration or just be logged."""
    CRITICAL     = auto()   # Stop integration
    WARNING      = auto()   # Log but continue
    INFORMATIONAL = auto()  # Record silently


@dataclass
class DetectedEvent:
    """Full record of a detected physical event."""
    name:         str
    severity:     EventSeverity
    affine_param: float
    coordinate_time: float
    r:            float
    theta:        float
    phi:          float
    description:  str
    data:         Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'severity': self.severity.name,
            'affine_param': self.affine_param,
            'coordinate_time': self.coordinate_time,
            'r': self.r, 'theta': self.theta, 'phi': self.phi,
            'description': self.description,
            'data': self.data,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Event Detector
# ─────────────────────────────────────────────────────────────────────────────

class EventDetector:
    """
    Configurable event detection system for geodesic integration.

    Checks a set of physical conditions at each integration step and
    returns any triggered events.  Critical events signal the integrator
    to halt; informational events are logged for analysis.

    Parameters
    ----------
    metric         : SpacetimeMetric
    horizon_buffer : float   Fractional buffer outside horizon (default 1%)
    escape_radius  : float   Escape threshold in units of M
    sing_threshold : float   Singularity approach threshold in units of M
    enable_isco    : bool    Track ISCO crossing for timelike geodesics
    enable_ergo    : bool    Track ergosphere entry (Kerr only)
    """

    def __init__(
        self,
        metric: SpacetimeMetric,
        horizon_buffer: float = 0.01,
        escape_radius: float = 500.0,
        sing_threshold: float = 0.01,
        enable_isco: bool = True,
        enable_ergo: bool = True,
    ) -> None:
        self.metric = metric
        self.M = metric.M

        self._r_horizon = metric.event_horizon()[0]
        self._r_horizon_buffered = self._r_horizon * (1.0 + horizon_buffer)
        self._r_photon = metric.photon_sphere()
        self._r_isco = metric.isco()
        self._r_escape = escape_radius * self.M
        self._r_sing = sing_threshold * self.M
        self._enable_isco = enable_isco
        self._enable_ergo = enable_ergo

        # State tracking
        self._prev_r: Optional[float] = None
        self._prev_ur: Optional[float] = None
        self._crossed_isco: bool = False
        self._entered_ergo: bool = False
        self._exited_ergo: bool = False

    def reset(self) -> None:
        """Reset tracking state for a new integration."""
        self._prev_r = None
        self._prev_ur = None
        self._crossed_isco = False
        self._entered_ergo = False
        self._exited_ergo = False

    def check(
        self,
        state: np.ndarray,
        affine_param: float,
        is_null: bool = False,
    ) -> List[DetectedEvent]:
        """
        Check all event conditions at the current state.

        Parameters
        ----------
        state        : np.ndarray, shape (8,) — [t, r, θ, φ, u^t, u^r, u^θ, u^φ]
        affine_param : float — current affine parameter
        is_null      : bool — True for photon geodesics

        Returns
        -------
        List[DetectedEvent] — events triggered at this step (may be empty).
        """
        events: List[DetectedEvent] = []
        t, r, theta, phi = state[0], state[1], state[2], state[3]
        ur = state[5]

        # ── 1. Singularity approach ──────────────────────────────────────────
        if r < self._r_sing:
            events.append(DetectedEvent(
                name="singularity_approach",
                severity=EventSeverity.CRITICAL,
                affine_param=affine_param,
                coordinate_time=t,
                r=r, theta=theta, phi=phi,
                description=f"Approached ring singularity at r={r/self.M:.6e}M",
                data={'r_M': r / self.M},
            ))
            self._prev_r = r
            self._prev_ur = ur
            return events  # No need to check further

        # ── 2. Horizon crossing ──────────────────────────────────────────────
        if r <= self._r_horizon_buffered:
            events.append(DetectedEvent(
                name="horizon_crossing",
                severity=EventSeverity.CRITICAL,
                affine_param=affine_param,
                coordinate_time=t,
                r=r, theta=theta, phi=phi,
                description=(
                    f"Crossed event horizon at r={r/self.M:.4f}M "
                    f"(r_h={self._r_horizon/self.M:.4f}M)"
                ),
                data={
                    'r_M': r / self.M,
                    'r_horizon_M': self._r_horizon / self.M,
                },
            ))

        # ── 3. Escape ────────────────────────────────────────────────────────
        if r > self._r_escape:
            events.append(DetectedEvent(
                name="escape",
                severity=EventSeverity.CRITICAL,
                affine_param=affine_param,
                coordinate_time=t,
                r=r, theta=theta, phi=phi,
                description=f"Escaped to r={r/self.M:.1f}M",
                data={'r_M': r / self.M},
            ))

        # ── 4. Turning points ────────────────────────────────────────────────
        if self._prev_ur is not None and self._prev_ur * ur < 0:
            tp_type = "periapsis" if ur > 0 else "apoapsis"
            events.append(DetectedEvent(
                name=f"turning_point_{tp_type}",
                severity=EventSeverity.INFORMATIONAL,
                affine_param=affine_param,
                coordinate_time=t,
                r=r, theta=theta, phi=phi,
                description=f"{tp_type.capitalize()} at r={r/self.M:.4f}M",
                data={'r_M': r / self.M, 'type': tp_type},
            ))

        # ── 5. Photon sphere (null only) ─────────────────────────────────────
        if is_null and abs(r - self._r_photon) < 0.05 * self.M and abs(ur) < 1e-6:
            events.append(DetectedEvent(
                name="photon_sphere_capture",
                severity=EventSeverity.CRITICAL,
                affine_param=affine_param,
                coordinate_time=t,
                r=r, theta=theta, phi=phi,
                description=(
                    f"Captured near photon sphere at r={r/self.M:.4f}M "
                    f"(r_ph={self._r_photon/self.M:.4f}M)"
                ),
                data={
                    'r_M': r / self.M,
                    'r_photon_M': self._r_photon / self.M,
                    'dr_dlambda': ur,
                },
            ))

        # ── 6. ISCO crossing (timelike only) ─────────────────────────────────
        if (self._enable_isco and not is_null and
            not self._crossed_isco and
            self._prev_r is not None and
            self._prev_r > self._r_isco and r <= self._r_isco):
            self._crossed_isco = True
            events.append(DetectedEvent(
                name="isco_crossing",
                severity=EventSeverity.WARNING,
                affine_param=affine_param,
                coordinate_time=t,
                r=r, theta=theta, phi=phi,
                description=(
                    f"Crossed ISCO at r={r/self.M:.4f}M "
                    f"(r_isco={self._r_isco/self.M:.4f}M) — orbit now unstable"
                ),
                data={
                    'r_M': r / self.M,
                    'r_isco_M': self._r_isco / self.M,
                },
            ))

        # ── 7. Ergosphere (Kerr only) ────────────────────────────────────────
        if self._enable_ergo and self.metric.spin > 0:
            r_ergo = self.metric.ergosphere_radius(theta)
            if r < r_ergo and r > self._r_horizon and not self._entered_ergo:
                self._entered_ergo = True
                events.append(DetectedEvent(
                    name="ergosphere_entry",
                    severity=EventSeverity.INFORMATIONAL,
                    affine_param=affine_param,
                    coordinate_time=t,
                    r=r, theta=theta, phi=phi,
                    description=(
                        f"Entered ergosphere at r={r/self.M:.4f}M "
                        f"(r_ergo={r_ergo/self.M:.4f}M)"
                    ),
                    data={
                        'r_M': r / self.M,
                        'r_ergo_M': r_ergo / self.M,
                    },
                ))
            elif r > r_ergo and self._entered_ergo and not self._exited_ergo:
                self._exited_ergo = True
                events.append(DetectedEvent(
                    name="ergosphere_exit",
                    severity=EventSeverity.INFORMATIONAL,
                    affine_param=affine_param,
                    coordinate_time=t,
                    r=r, theta=theta, phi=phi,
                    description=f"Exited ergosphere at r={r/self.M:.4f}M",
                    data={'r_M': r / self.M},
                ))

        self._prev_r = r
        self._prev_ur = ur
        return events

    def format_summary(self, events: List[DetectedEvent]) -> str:
        """Format a human-readable summary of all detected events."""
        if not events:
            return "  No events detected."

        lines = [f"  Events Detected ({len(events)}):"]
        lines.append("  " + "─" * 70)

        icons = {
            EventSeverity.CRITICAL: "🔴",
            EventSeverity.WARNING: "🟡",
            EventSeverity.INFORMATIONAL: "🔵",
        }

        for ev in events:
            icon = icons.get(ev.severity, "⚪")
            lines.append(
                f"    {icon} λ={ev.affine_param:.4e}  |  "
                f"t={ev.coordinate_time:.4e}  |  {ev.description}"
            )

        return "\n".join(lines)
