"""
General Relativity Simulation Engine
=====================================
High-level engine that orchestrates geodesic tracing in black hole
spacetimes.  This is the GR counterpart of the Newtonian
``SimulationEngine`` — it drives the geodesic integrator with a
configured metric and produces structured results.

Public API
----------
GRSimulationEngine.trace_geodesic()  — trace a single particle/photon
GRSimulationEngine.trace_photon_ring() — trace multiple photons for lensing
GRSimulationEngine.compute_shadow()  — map the black hole shadow boundary
GRSimulationEngine.analyze_orbit()   — detailed orbital analysis

Usage
-----
    from astrolab.physics.metrics import create_metric
    from astrolab.engine.gr_engine import GRSimulationEngine
    
    metric = create_metric(mass_kg=1.989e30, metric_type='schwarzschild')
    engine = GRSimulationEngine(metric)
    result = engine.trace_geodesic(r=10, theta=pi/2, phi=0, ur=0, uphi=0.02)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from astrolab.physics.metrics import (
    SpacetimeMetric,
    SchwarzschildMetric,
    KerrMetric,
    create_metric,
    mass_to_geometric,
    C_SI,
    G_SI,
)
from astrolab.physics.christoffel import (
    normalize_4velocity,
    compute_constants_of_motion,
    compute_norm,
)
from astrolab.physics.geodesic_integrator import (
    GeodesicIntegrator,
    GeodesicResult,
    GeodesicPoint,
    ParticleType,
    TerminationReason,
)
from astrolab.physics.observables import (
    gravitational_time_dilation,
    gravitational_redshift,
    deflection_angle_weak_field,
    critical_impact_parameter,
    frame_dragging_velocity,
    perihelion_precession_rate,
    shapiro_delay,
    effective_potential_timelike,
    effective_potential_null,
    compute_potential_curve,
    generate_observables_report,
    redshift_schwarzschild,
    time_dilation_schwarzschild,
    proper_distance_schwarzschild,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BlackHoleConfig:
    """Configuration for a black hole."""
    mass_kg:       float          # Mass in SI kilograms
    mass_geometric: float = 0.0   # Mass in geometric units (metres) ← computed
    spin:          float = 0.0    # Dimensionless spin a/M ∈ [0, 1)
    metric_type:   str = "schwarzschild"

    def __post_init__(self):
        self.mass_geometric = mass_to_geometric(self.mass_kg)
        if self.spin > 0 and self.metric_type == 'schwarzschild':
            self.metric_type = 'kerr'

    @property
    def schwarzschild_radius_m(self) -> float:
        """Event horizon radius in metres (Schwarzschild)."""
        return 2.0 * self.mass_geometric

    @property
    def schwarzschild_radius_km(self) -> float:
        return self.schwarzschild_radius_m / 1e3

    def to_dict(self) -> dict:
        return {
            'mass_kg': self.mass_kg,
            'mass_geometric_m': self.mass_geometric,
            'spin': self.spin,
            'metric_type': self.metric_type,
            'schwarzschild_radius_m': self.schwarzschild_radius_m,
        }


# ─────────────────────────────────────────────────────────────────────────────
# GR Engine
# ─────────────────────────────────────────────────────────────────────────────

class GRSimulationEngine:
    """
    General-relativistic simulation engine for black hole spacetimes.

    Wraps the geodesic integrator with convenient methods for tracing
    particles, photons, and computing observables.

    Parameters
    ----------
    metric : SpacetimeMetric
        The spacetime metric (Schwarzschild or Kerr).
    config : BlackHoleConfig, optional
        Black hole configuration (auto-generated from metric if not given).
    """

    def __init__(
        self,
        metric: SpacetimeMetric,
        config: Optional[BlackHoleConfig] = None,
    ) -> None:
        self.metric = metric
        self.config = config
        self.M = metric.M

    @classmethod
    def from_config(cls, config: BlackHoleConfig) -> 'GRSimulationEngine':
        """Create engine from a BlackHoleConfig."""
        metric = create_metric(
            mass_kg=config.mass_kg,
            metric_type=config.metric_type,
            spin=config.spin,
        )
        return cls(metric, config)

    # ── Geodesic Tracing ─────────────────────────────────────────────────────

    def trace_geodesic(
        self,
        r: float,
        theta: float = math.pi / 2,
        phi: float = 0.0,
        ur: float = 0.0,
        utheta: float = 0.0,
        uphi: float = 0.0,
        particle_type: str = "timelike",
        max_steps: int = 50_000,
        max_affine: float = 500.0,
        initial_step: float = 0.01,
        record_every: int = 1,
    ) -> GeodesicResult:
        """
        Trace a geodesic from specified initial conditions.

        Coordinates are in units of M (black hole mass in geometric units):
            r=10 means 10M from the singularity.

        The initial u^t is automatically computed from the mass-shell
        constraint  g_{μν} u^μ u^ν = κ  (−1 for timelike, 0 for null).

        Parameters
        ----------
        r          : float  Initial radial coordinate (units of M)
        theta      : float  Initial polar angle [rad]  (default: equatorial)
        phi        : float  Initial azimuthal angle [rad]
        ur         : float  Initial dr/dλ
        utheta     : float  Initial dθ/dλ
        uphi       : float  Initial dφ/dλ
        particle_type : str  "timelike" or "null"
        max_steps  : int    Maximum integration steps
        max_affine : float  Maximum affine parameter (units of M)
        initial_step : float  Initial step size (units of M)
        record_every : int  Record every Nth point

        Returns
        -------
        GeodesicResult
        """
        ptype = ParticleType.TIMELIKE if particle_type.lower() == "timelike" else ParticleType.NULL
        target_norm = -1.0 if ptype == ParticleType.TIMELIKE else 0.0

        # Scale coordinates by M
        M = self.M
        r_phys = r * M

        # Build initial state with placeholder u^t
        state = np.array([
            0.0,       # t
            r_phys,    # r
            theta,     # θ
            phi,       # φ
            1.0,       # u^t (placeholder — will be normalized)
            ur,        # u^r
            utheta,    # u^θ
            uphi,      # u^φ
        ], dtype=np.float64)

        # Normalize to satisfy mass-shell constraint
        state = normalize_4velocity(state, self.metric, target_norm)

        # Create integrator
        integrator = GeodesicIntegrator(
            metric=self.metric,
            particle_type=ptype,
            max_step=1.0,
            escape_radius=500.0,
            normalize_every=10,
            record_every=record_every,
        )

        # Integrate
        result = integrator.integrate(
            initial_state=state,
            max_affine=max_affine,
            max_steps=max_steps,
            initial_step=initial_step,
        )

        return result

    def trace_photon(
        self,
        impact_parameter: float,
        r_start: float = 100.0,
        theta: float = math.pi / 2,
        direction: str = "inward",
        max_steps: int = 100_000,
        max_affine: float = 2000.0,
    ) -> GeodesicResult:
        """
        Trace a photon with a given impact parameter b = L/E.

        The photon starts at r_start (in units of M) and moves inward
        toward the black hole.  The impact parameter determines whether
        it is captured (b < b_crit) or deflected (b > b_crit).

        Parameters
        ----------
        impact_parameter : float  b = L/E (units of M)
        r_start          : float  Starting radius (units of M)
        theta            : float  Polar angle
        direction        : str    "inward" or "outward"
        max_steps        : int
        max_affine       : float

        Returns
        -------
        GeodesicResult
        """
        M = self.M
        r = r_start * M
        b = impact_parameter * M

        # For a photon at large r moving inward in the equatorial plane:
        # E = u^t (at infinity, E = 1 for normalization)
        # L = b * E = b
        # u^φ = L / (r² sin²θ) = b / r²  (equatorial: sinθ = 1)
        # u^r from null condition: g_{tt}(u^t)² + g_{rr}(u^r)² + g_{φφ}(u^φ)² = 0

        sin_th = math.sin(theta)
        uphi = b / (r * r * sin_th * sin_th) if abs(sin_th) > 1e-15 else 0.0

        # At large r, g_tt ≈ -1, g_rr ≈ 1, so:
        # (u^r)² ≈ (u^t)² - b² / r²
        # With u^t ≈ 1 at infinity:
        ur_sq = 1.0 - b * b / (r * r)
        if ur_sq < 0:
            ur_sq = 0.0  # Turning point at start — photon can't reach this r

        ur = -math.sqrt(ur_sq) if direction == "inward" else math.sqrt(ur_sq)

        return self.trace_geodesic(
            r=r_start, theta=theta, phi=0.0,
            ur=ur, utheta=0.0, uphi=uphi,
            particle_type="null",
            max_steps=max_steps,
            max_affine=max_affine,
            record_every=1,
        )

    def trace_photon_ring(
        self,
        n_rays: int = 50,
        b_min: float = 1.0,
        b_max: float = 10.0,
        r_start: float = 100.0,
        theta: float = math.pi / 2,
    ) -> List[GeodesicResult]:
        """
        Trace multiple photons at different impact parameters.

        Useful for visualizing gravitational lensing and the black hole
        shadow boundary.

        Parameters
        ----------
        n_rays  : int    Number of photon rays
        b_min   : float  Minimum impact parameter (units of M)
        b_max   : float  Maximum impact parameter (units of M)
        r_start : float  Starting radius (units of M)
        theta   : float  Polar angle

        Returns
        -------
        List[GeodesicResult]
        """
        b_values = np.linspace(b_min, b_max, n_rays)
        results = []

        for b in b_values:
            result = self.trace_photon(
                impact_parameter=float(b),
                r_start=r_start,
                theta=theta,
                max_steps=50_000,
                max_affine=1000.0,
            )
            results.append(result)

        return results

    def compute_shadow_boundary(
        self,
        n_points: int = 100,
    ) -> List[Tuple[float, float]]:
        """
        Compute the apparent boundary of the black hole shadow.

        The shadow is the set of photon impact parameters that lead to
        capture.  Its boundary corresponds to the critical impact
        parameter b_c = 3√3 M (Schwarzschild).

        For Kerr, the shadow is asymmetric and depends on spin.

        Returns
        -------
        List of (alpha, beta) celestial coordinates on the observer's sky,
        in units of M.
        """
        M = self.M
        boundary = []

        if isinstance(self.metric, SchwarzschildMetric):
            # Circular shadow with radius b_c = 3√3 M
            b_c = critical_impact_parameter(self.metric) / M
            for i in range(n_points):
                angle = 2.0 * math.pi * i / n_points
                alpha = b_c * math.cos(angle)
                beta = b_c * math.sin(angle)
                boundary.append((alpha, beta))
        elif isinstance(self.metric, KerrMetric):
            a = self.metric.a / M
            # Parametric shadow boundary for Kerr
            # Using Bardeen's parametrization
            for i in range(n_points):
                r_ph = 2.0 + 2.0 * math.cos(2.0 / 3.0 * math.acos(-a) - 2.0 * math.pi * i / n_points)
                r_ph = max(r_ph * M, self.metric.event_horizon()[0] * 1.01)
                # Celestial coordinates
                xi = -(r_ph * r_ph + (a * M) ** 2) / (a * M) + 2.0 * M * r_ph / (a * M) if abs(a) > 1e-10 else 0
                eta_sq = r_ph ** 2 * (r_ph * (r_ph - 3.0 * M) + 2.0 * a * M * math.sqrt(M * r_ph)) / ((a * M) ** 2) if abs(a) > 1e-10 else 27.0 * M * M
                eta = math.sqrt(max(0, eta_sq))
                boundary.append((xi / M, eta / M))

        return boundary

    # ── Observable Computations ──────────────────────────────────────────────

    def compute_time_dilation(
        self,
        r: float,
        theta: float = math.pi / 2,
        v_tangential: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute time dilation at radius r (in units of M).

        Returns
        -------
        dict with 'factor' (dτ/dt), 'clock_rate' (percentage of distant clock),
             'one_hour_local_equals_s' (how many seconds pass far away)
        """
        r_phys = r * self.M
        factor = gravitational_time_dilation(r_phys, self.metric, theta, v_tangential)

        result = {
            'factor': factor,
            'clock_rate_percent': factor * 100.0,
        }

        if factor > 0:
            result['one_hour_local_equals_s'] = 3600.0 / factor
        else:
            result['one_hour_local_equals_s'] = float('inf')

        return result

    def compute_redshift(
        self,
        r_emit: float,
        r_obs: float = 1000.0,
        theta: float = math.pi / 2,
    ) -> Dict[str, float]:
        """
        Compute gravitational redshift between emitter and observer.

        Both radii in units of M.
        """
        r_e = r_emit * self.M
        r_o = r_obs * self.M
        return gravitational_redshift(r_e, r_o, self.metric, theta)

    def compute_lensing(
        self,
        impact_parameter: float,
    ) -> Dict[str, float]:
        """
        Compute gravitational lensing properties for a photon.

        Parameters
        ----------
        impact_parameter : float  b = L/E (units of M)

        Returns
        -------
        dict with deflection angles, critical parameters, etc.
        """
        M = self.M
        b_phys = impact_parameter * M
        b_crit = critical_impact_parameter(self.metric)

        result = {
            'impact_parameter_M': impact_parameter,
            'critical_impact_parameter_M': b_crit / M,
            'captured': impact_parameter < b_crit / M,
            'weak_field_deflection_rad': deflection_angle_weak_field(b_phys, M),
            'weak_field_deflection_deg': math.degrees(deflection_angle_weak_field(b_phys, M)),
        }

        return result

    def compute_frame_dragging(
        self,
        r: float,
        theta: float = math.pi / 2,
    ) -> Dict[str, float]:
        """Frame-dragging at radius r (units of M)."""
        r_phys = r * self.M
        return frame_dragging_velocity(r_phys, theta, self.metric)

    def compute_effective_potential(
        self,
        L: float,
        E: float = 1.0,
        r_min: float = 2.5,
        r_max: float = 50.0,
        n_points: int = 500,
        particle_type: str = "timelike",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute effective potential curve.

        L in units of M, r range in units of M.
        Returns (r_array/M, V_array).
        """
        L_phys = L * self.M
        is_null = particle_type.lower() == "null"
        a = self.metric.a if hasattr(self.metric, 'a') else 0.0

        r_arr, V_arr = compute_potential_curve(
            L=L_phys, M=self.M,
            r_min=r_min, r_max=r_max,
            n_points=n_points,
            is_null=is_null, a=a, E=E,
        )

        return r_arr / self.M, V_arr

    # ── Black Hole Information ───────────────────────────────────────────────

    def black_hole_info(self) -> Dict:
        """
        Comprehensive information about the black hole.

        Returns a dict with all characteristic radii, both in geometric
        units (M) and SI metres.
        """
        M = self.M
        metric = self.metric
        info: Dict = {}

        info['metric'] = metric.name
        info['mass_geometric_m'] = M
        if self.config:
            info['mass_kg'] = self.config.mass_kg
            info['mass_solar'] = self.config.mass_kg / 1.989e30

        # Event horizon
        horizons = metric.event_horizon()
        info['event_horizon_outer_M'] = horizons[0] / M
        info['event_horizon_outer_m'] = horizons[0]
        info['event_horizon_outer_km'] = horizons[0] / 1e3

        if len(horizons) > 1:
            info['event_horizon_inner_M'] = horizons[1] / M
            info['event_horizon_inner_m'] = horizons[1]

        # Photon sphere
        r_ph = metric.photon_sphere()
        info['photon_sphere_M'] = r_ph / M
        info['photon_sphere_m'] = r_ph
        info['photon_sphere_km'] = r_ph / 1e3

        # ISCO
        r_isco = metric.isco(prograde=True)
        info['isco_prograde_M'] = r_isco / M
        info['isco_prograde_m'] = r_isco
        info['isco_prograde_km'] = r_isco / 1e3

        if metric.spin > 0:
            r_isco_retro = metric.isco(prograde=False)
            info['isco_retrograde_M'] = r_isco_retro / M
            info['isco_retrograde_m'] = r_isco_retro

            # Ergosphere at equator
            r_ergo = metric.ergosphere_radius(math.pi / 2)
            info['ergosphere_equator_M'] = r_ergo / M
            info['ergosphere_equator_m'] = r_ergo

            info['spin'] = metric.spin
            info['spin_dimensional_m'] = metric.a if hasattr(metric, 'a') else 0.0

        # Critical impact parameter
        b_crit = critical_impact_parameter(metric)
        info['critical_impact_parameter_M'] = b_crit / M

        # Shadow angular radius (for distant observer)
        info['shadow_angular_radius_M'] = b_crit / M

        return info

    def format_info(self) -> str:
        """Pretty-print black hole information."""
        info = self.black_hole_info()
        M = self.M

        lines = []
        lines.append(f"\n  ┌─ Black Hole Properties ({info['metric']} Metric)")
        lines.append(f"  │")

        if 'mass_kg' in info:
            lines.append(f"  │  Mass              : {info['mass_kg']:.4e} kg")
        if 'mass_solar' in info:
            lines.append(f"  │  Mass (solar)      : {info['mass_solar']:.4f} M☉")
        lines.append(f"  │  Mass (geometric)  : {M:.4e} m")

        if 'spin' in info:
            lines.append(f"  │  Spin (a/M)        : {info['spin']:.4f}")

        lines.append(f"  │")
        lines.append(f"  │  ── Characteristic Radii ──")
        lines.append(f"  │  Event Horizon     : {info['event_horizon_outer_M']:.4f} M  "
                     f"= {info['event_horizon_outer_km']:.4f} km")

        if 'event_horizon_inner_M' in info:
            lines.append(f"  │  Inner Horizon     : {info['event_horizon_inner_M']:.4f} M  "
                         f"= {info.get('event_horizon_inner_m', 0)/1e3:.4f} km")

        lines.append(f"  │  Photon Sphere     : {info['photon_sphere_M']:.4f} M  "
                     f"= {info['photon_sphere_km']:.4f} km")
        lines.append(f"  │  ISCO (prograde)   : {info['isco_prograde_M']:.4f} M  "
                     f"= {info['isco_prograde_km']:.4f} km")

        if 'isco_retrograde_M' in info:
            lines.append(f"  │  ISCO (retrograde) : {info['isco_retrograde_M']:.4f} M")

        if 'ergosphere_equator_M' in info:
            lines.append(f"  │  Ergosphere (eq.)  : {info['ergosphere_equator_M']:.4f} M")

        lines.append(f"  │  Shadow Radius     : {info['shadow_angular_radius_M']:.4f} M")
        lines.append(f"  │  Critical b        : {info['critical_impact_parameter_M']:.4f} M")

        lines.append(f"  └─")
        return "\n".join(lines)
