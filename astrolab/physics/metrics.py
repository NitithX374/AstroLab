"""
Spacetime Metrics
=================
General-relativistic metric tensors for black hole spacetimes.

All computations use **geometric units** (G = c = 1) where the black hole
mass M sets the fundamental length/time scale:
    r_s = 2M   (Schwarzschild radius)

Coordinate system: Boyer-Lindquist (t, r, θ, φ)
    - Reduces to Schwarzschild coordinates when spin a = 0

Classes
-------
SpacetimeMetric       : Abstract base for all metric implementations
SchwarzschildMetric   : Non-rotating, spherically symmetric (Schwarzschild 1916)
KerrMetric            : Rotating, axially symmetric (Kerr 1963)

References
----------
- Misner, Thorne & Wheeler, "Gravitation" (1973), Ch. 25-33
- Chandrasekhar, "The Mathematical Theory of Black Holes" (1983)
- Carroll, "Spacetime and Geometry" (2004), Ch. 5-6
"""

from __future__ import annotations

import abc
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants for unit conversion
# ─────────────────────────────────────────────────────────────────────────────

G_SI: float = 6.67430e-11       # m³ kg⁻¹ s⁻²
C_SI: float = 2.99792458e8      # m/s

def mass_to_geometric(mass_kg: float) -> float:
    """Convert mass from SI kilograms to geometric length units (metres).
    
    In geometric units (G=c=1), mass has dimensions of length:
        M_geom = G·M / c²
    
    Example: Solar mass → ~1477 m
    """
    return G_SI * mass_kg / C_SI**2


def geometric_to_si_length(length_geom: float, mass_kg: float) -> float:
    """Convert a geometric-unit length back to SI metres.
    
    If the length is expressed as a multiple of M (geometric mass),
    multiply by M_geom = G·M_kg / c².
    """
    return length_geom * mass_to_geometric(mass_kg)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class SpacetimeMetric(abc.ABC):
    """
    Abstract base for a 4D spacetime metric in Boyer-Lindquist coordinates.

    Coordinate ordering: (t, r, θ, φ) → indices (0, 1, 2, 3).

    All methods operate in geometric units with black hole mass M.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable metric name."""

    @property
    @abc.abstractmethod
    def M(self) -> float:
        """Black hole mass in geometric units (length)."""

    @property
    @abc.abstractmethod
    def spin(self) -> float:
        """Dimensionless spin parameter a/M  ∈ [0, 1)."""

    @abc.abstractmethod
    def metric_tensor(self, r: float, theta: float) -> np.ndarray:
        """
        Covariant metric tensor g_μν at (r, θ).

        Returns
        -------
        np.ndarray, shape (4, 4)  — symmetric matrix.
        """

    @abc.abstractmethod
    def inverse_metric(self, r: float, theta: float) -> np.ndarray:
        """
        Contravariant metric tensor g^μν at (r, θ).

        Returns
        -------
        np.ndarray, shape (4, 4)
        """

    @abc.abstractmethod
    def christoffel(self, r: float, theta: float) -> np.ndarray:
        """
        Christoffel symbols of the second kind  Γ^α_{μν}  at (r, θ).

        Returns
        -------
        np.ndarray, shape (4, 4, 4)
            Index convention: result[alpha][mu][nu] = Γ^α_{μν}
        """

    @abc.abstractmethod
    def event_horizon(self) -> Tuple[float, ...]:
        """
        Coordinate radii of event horizons.

        Returns
        -------
        tuple of float — (r_outer,) for Schwarzschild, (r_outer, r_inner) for Kerr.
        """

    @abc.abstractmethod
    def photon_sphere(self) -> float:
        """Radius of the (unstable) circular photon orbit."""

    @abc.abstractmethod
    def isco(self, prograde: bool = True) -> float:
        """Innermost stable circular orbit radius."""

    def is_inside_horizon(self, r: float) -> bool:
        """Check whether coordinate radius r is at or inside the outer horizon."""
        return r <= self.event_horizon()[0]

    def ergosphere_radius(self, theta: float) -> float:
        """
        Outer boundary of the ergosphere (where g_tt = 0).
        For Schwarzschild this coincides with the horizon.
        Overridden for Kerr.
        """
        return self.event_horizon()[0]

    def frame_dragging_omega(self, r: float, theta: float) -> float:
        """
        Frame-dragging angular velocity  ω = -g_{tφ} / g_{φφ}.
        Zero for Schwarzschild; non-zero for Kerr.
        """
        g = self.metric_tensor(r, theta)
        if abs(g[3, 3]) < 1e-30:
            return 0.0
        return -g[0, 3] / g[3, 3]

    def to_dict(self) -> Dict:
        """Serialize metric parameters."""
        return {
            'name': self.name,
            'M': self.M,
            'spin': self.spin,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Schwarzschild Metric
# ─────────────────────────────────────────────────────────────────────────────

class SchwarzschildMetric(SpacetimeMetric):
    """
    Schwarzschild metric — static, spherically symmetric vacuum solution.

    Line element (Boyer-Lindquist / Schwarzschild coordinates):

        ds² = -(1 - 2M/r) dt²  +  (1 - 2M/r)⁻¹ dr²  +  r² dθ²  +  r² sin²θ dφ²

    Parameters
    ----------
    M_geometric : float
        Black hole mass in geometric units (length).  Typically obtained
        via ``mass_to_geometric(mass_kg)``.

    Key radii (in units of M):
        Event horizon : r_h   = 2M
        Photon sphere : r_ph  = 3M
        ISCO          : r_isco = 6M
    """

    def __init__(self, M_geometric: float) -> None:
        if M_geometric <= 0:
            raise ValueError(f"Mass must be positive, got {M_geometric}")
        self._M = M_geometric

    @property
    def name(self) -> str:
        return "Schwarzschild"

    @property
    def M(self) -> float:
        return self._M

    @property
    def spin(self) -> float:
        return 0.0

    # ── Metric tensor ────────────────────────────────────────────────────────

    def metric_tensor(self, r: float, theta: float) -> np.ndarray:
        """
        Covariant Schwarzschild metric g_μν.

        g_tt  = -(1 - 2M/r)
        g_rr  =  (1 - 2M/r)⁻¹
        g_θθ  =  r²
        g_φφ  =  r² sin²θ
        All off-diagonal = 0
        """
        M = self._M
        g = np.zeros((4, 4), dtype=np.float64)

        if r < 1e-15:
            return g  # Singular — return zeros (caller should detect this)

        f = 1.0 - 2.0 * M / r              # lapse function squared
        sin_th = math.sin(theta)

        g[0, 0] = -f                        # g_tt
        g[1, 1] = 1.0 / f if abs(f) > 1e-30 else 1e30  # g_rr
        g[2, 2] = r * r                     # g_θθ
        g[3, 3] = r * r * sin_th * sin_th   # g_φφ

        return g

    def inverse_metric(self, r: float, theta: float) -> np.ndarray:
        """
        Contravariant Schwarzschild metric g^μν (diagonal inverse).
        """
        M = self._M
        gi = np.zeros((4, 4), dtype=np.float64)

        if r < 1e-15:
            return gi

        f = 1.0 - 2.0 * M / r
        sin_th = math.sin(theta)
        r2 = r * r

        gi[0, 0] = -1.0 / f if abs(f) > 1e-30 else -1e30
        gi[1, 1] = f
        gi[2, 2] = 1.0 / r2 if r2 > 1e-30 else 1e30
        gi[3, 3] = 1.0 / (r2 * sin_th * sin_th) if abs(sin_th) > 1e-15 else 1e30

        return gi

    # ── Christoffel symbols (analytic) ───────────────────────────────────────

    def christoffel(self, r: float, theta: float) -> np.ndarray:
        """
        Christoffel symbols Γ^α_{μν} for Schwarzschild, computed analytically.

        Non-zero components (9 independent):
            Γ^t_{tr} = Γ^t_{rt} = M / (r(r-2M))
            Γ^r_{tt} = M(r-2M) / r³
            Γ^r_{rr} = -M / (r(r-2M))
            Γ^r_{θθ} = -(r-2M)
            Γ^r_{φφ} = -(r-2M) sin²θ
            Γ^θ_{rθ} = Γ^θ_{θr} = 1/r
            Γ^θ_{φφ} = -sinθ cosθ
            Γ^φ_{rφ} = Γ^φ_{φr} = 1/r
            Γ^φ_{θφ} = Γ^φ_{φθ} = cosθ/sinθ
        """
        M = self._M
        Gamma = np.zeros((4, 4, 4), dtype=np.float64)

        if r < 1e-15:
            return Gamma

        sin_th = math.sin(theta)
        cos_th = math.cos(theta)
        r2 = r * r
        r3 = r2 * r
        f = 1.0 - 2.0 * M / r         # (r - 2M) / r
        rm2M = r - 2.0 * M             # r - 2M

        if abs(rm2M) < 1e-30:
            return Gamma  # On the horizon — degenerate

        # Γ^t_{tr} = Γ^t_{rt} = M / (r(r - 2M))
        val_ttr = M / (r * rm2M)
        Gamma[0, 0, 1] = val_ttr
        Gamma[0, 1, 0] = val_ttr

        # Γ^r_{tt} = M(r - 2M) / r³
        Gamma[1, 0, 0] = M * rm2M / r3

        # Γ^r_{rr} = -M / (r(r - 2M))
        Gamma[1, 1, 1] = -M / (r * rm2M)

        # Γ^r_{θθ} = -(r - 2M)
        Gamma[1, 2, 2] = -rm2M

        # Γ^r_{φφ} = -(r - 2M) sin²θ
        Gamma[1, 3, 3] = -rm2M * sin_th * sin_th

        # Γ^θ_{rθ} = Γ^θ_{θr} = 1/r
        Gamma[2, 1, 2] = 1.0 / r
        Gamma[2, 2, 1] = 1.0 / r

        # Γ^θ_{φφ} = -sinθ cosθ
        Gamma[2, 3, 3] = -sin_th * cos_th

        # Γ^φ_{rφ} = Γ^φ_{φr} = 1/r
        Gamma[3, 1, 3] = 1.0 / r
        Gamma[3, 3, 1] = 1.0 / r

        # Γ^φ_{θφ} = Γ^φ_{φθ} = cosθ / sinθ
        if abs(sin_th) > 1e-15:
            cot_th = cos_th / sin_th
            Gamma[3, 2, 3] = cot_th
            Gamma[3, 3, 2] = cot_th

        return Gamma

    # ── Characteristic radii ─────────────────────────────────────────────────

    def event_horizon(self) -> Tuple[float]:
        """Schwarzschild event horizon at r = 2M."""
        return (2.0 * self._M,)

    def photon_sphere(self) -> float:
        """Unstable circular photon orbit at r = 3M."""
        return 3.0 * self._M

    def isco(self, prograde: bool = True) -> float:
        """Innermost stable circular orbit at r = 6M (no spin dependence)."""
        return 6.0 * self._M

    def effective_potential_timelike(self, r: float, L: float, M: float = None) -> float:
        """
        Effective potential for massive particle radial motion.

        V_eff(r) = (1 - 2M/r)(1 + L²/r²)

        where L is the specific angular momentum (per unit rest mass).
        """
        if M is None:
            M = self._M
        if r < 1e-15:
            return float('inf')
        return (1.0 - 2.0 * M / r) * (1.0 + L * L / (r * r))

    def effective_potential_null(self, r: float, L: float, M: float = None) -> float:
        """
        Effective potential for photon radial motion.

        V_eff(r) = (1 - 2M/r) L² / r²
        """
        if M is None:
            M = self._M
        if r < 1e-15:
            return float('inf')
        return (1.0 - 2.0 * M / r) * L * L / (r * r)


# ─────────────────────────────────────────────────────────────────────────────
# Kerr Metric
# ─────────────────────────────────────────────────────────────────────────────

class KerrMetric(SpacetimeMetric):
    """
    Kerr metric — stationary, axially symmetric vacuum solution for a
    rotating black hole.

    Line element in Boyer-Lindquist coordinates:

        ds² = -(1 - 2Mr/Σ) dt²  −  (4Mar sin²θ / Σ) dt dφ
              + (Σ/Δ) dr²  +  Σ dθ²
              + (r² + a² + 2Ma²r sin²θ / Σ) sin²θ dφ²

    where:
        Σ = r² + a² cos²θ
        Δ = r² − 2Mr + a²
        a = J/M = spin parameter (dimension of length)

    Parameters
    ----------
    M_geometric : float
        Black hole mass in geometric units (length).
    a_over_M : float
        Dimensionless spin parameter  a/M ∈ [0, 1).
        Extreme Kerr limit: a/M → 1.
    """

    def __init__(self, M_geometric: float, a_over_M: float = 0.0) -> None:
        if M_geometric <= 0:
            raise ValueError(f"Mass must be positive, got {M_geometric}")
        if not (0.0 <= a_over_M < 1.0):
            raise ValueError(
                f"Spin a/M must be in [0, 1), got {a_over_M}. "
                f"a/M = 1 is the extreme Kerr limit (excluded for numerical stability)."
            )
        self._M = M_geometric
        self._a = a_over_M * M_geometric   # dimensional spin: a = (a/M) * M
        self._a_over_M = a_over_M

    @property
    def name(self) -> str:
        return "Kerr"

    @property
    def M(self) -> float:
        return self._M

    @property
    def spin(self) -> float:
        return self._a_over_M

    @property
    def a(self) -> float:
        """Dimensional spin parameter in geometric units."""
        return self._a

    # ── Helper functions ─────────────────────────────────────────────────────

    def _sigma(self, r: float, theta: float) -> float:
        """Σ = r² + a² cos²θ"""
        cos_th = math.cos(theta)
        return r * r + self._a * self._a * cos_th * cos_th

    def _delta(self, r: float) -> float:
        """Δ = r² − 2Mr + a²"""
        return r * r - 2.0 * self._M * r + self._a * self._a

    # ── Metric tensor ────────────────────────────────────────────────────────

    def metric_tensor(self, r: float, theta: float) -> np.ndarray:
        M = self._M
        a = self._a
        g = np.zeros((4, 4), dtype=np.float64)

        if r < 1e-15:
            return g

        sin_th = math.sin(theta)
        cos_th = math.cos(theta)
        sin2 = sin_th * sin_th

        Sigma = r * r + a * a * cos_th * cos_th
        Delta = r * r - 2.0 * M * r + a * a

        if abs(Sigma) < 1e-30:
            return g

        A = (r * r + a * a)**2 - a * a * Delta * sin2

        g[0, 0] = -(1.0 - 2.0 * M * r / Sigma)                # g_tt
        g[0, 3] = -2.0 * M * a * r * sin2 / Sigma              # g_tφ
        g[3, 0] = g[0, 3]                                       # symmetric
        g[1, 1] = Sigma / Delta if abs(Delta) > 1e-30 else 1e30 # g_rr
        g[2, 2] = Sigma                                         # g_θθ
        g[3, 3] = A * sin2 / Sigma                              # g_φφ

        return g

    def inverse_metric(self, r: float, theta: float) -> np.ndarray:
        M = self._M
        a = self._a
        gi = np.zeros((4, 4), dtype=np.float64)

        if r < 1e-15:
            return gi

        sin_th = math.sin(theta)
        cos_th = math.cos(theta)
        sin2 = sin_th * sin_th

        Sigma = r * r + a * a * cos_th * cos_th
        Delta = r * r - 2.0 * M * r + a * a

        if abs(Sigma) < 1e-30:
            return gi

        A = (r * r + a * a)**2 - a * a * Delta * sin2

        if abs(A * sin2) < 1e-30 or abs(Delta) < 1e-30:
            return gi

        # Inverse components via analytic formulas
        gi[0, 0] = -A / (Sigma * Delta)
        gi[0, 3] = -2.0 * M * a * r / (Sigma * Delta)
        gi[3, 0] = gi[0, 3]
        gi[1, 1] = Delta / Sigma
        gi[2, 2] = 1.0 / Sigma
        gi[3, 3] = (Delta - a * a * sin2) / (Sigma * Delta * sin2) if abs(sin2) > 1e-30 else 1e30

        return gi

    # ── Christoffel symbols (numerical via finite differences) ───────────────

    def christoffel(self, r: float, theta: float) -> np.ndarray:
        """
        Christoffel symbols Γ^α_{μν} for the Kerr metric.

        Computed via:
            Γ^α_{μν} = ½ g^{αβ} (∂_μ g_{νβ} + ∂_ν g_{μβ} − ∂_β g_{μν})

        Derivatives are computed numerically using central finite differences
        with respect to (r, θ). The metric is stationary (∂_t = 0) and
        axisymmetric (∂_φ = 0), so only r and θ derivatives are needed.
        """
        h_r = max(1e-6 * abs(r), 1e-10) if r > 0 else 1e-10
        h_th = 1e-6

        # Clamp theta away from poles for numerical stability
        theta_safe = max(min(theta, math.pi - 1e-8), 1e-8)

        # ∂g/∂x^γ for γ ∈ {0=t, 1=r, 2=θ, 3=φ}
        # ∂_t g = 0 (stationary),  ∂_φ g = 0 (axisymmetric)
        dg = np.zeros((4, 4, 4), dtype=np.float64)  # dg[gamma][mu][nu]

        # ∂g/∂r via central difference
        g_plus = self.metric_tensor(r + h_r, theta_safe)
        g_minus = self.metric_tensor(r - h_r, theta_safe)
        dg[1] = (g_plus - g_minus) / (2.0 * h_r)

        # ∂g/∂θ via central difference
        g_plus = self.metric_tensor(r, theta_safe + h_th)
        g_minus = self.metric_tensor(r, theta_safe - h_th)
        dg[2] = (g_plus - g_minus) / (2.0 * h_th)

        # Compute Γ^α_{μν} = ½ g^{αβ} (∂_μ g_{νβ} + ∂_ν g_{μβ} − ∂_β g_{μν})
        gi = self.inverse_metric(r, theta_safe)
        Gamma = np.zeros((4, 4, 4), dtype=np.float64)

        for alpha in range(4):
            for mu in range(4):
                for nu in range(mu, 4):  # exploit symmetry in lower indices
                    val = 0.0
                    for beta in range(4):
                        val += gi[alpha, beta] * (
                            dg[mu][nu][beta] + dg[nu][mu][beta] - dg[beta][mu][nu]
                        )
                    val *= 0.5
                    Gamma[alpha, mu, nu] = val
                    Gamma[alpha, nu, mu] = val  # Γ symmetric in lower indices

        return Gamma

    # ── Characteristic radii ─────────────────────────────────────────────────

    def event_horizon(self) -> Tuple[float, float]:
        """
        Kerr event horizons: r± = M ± √(M² − a²).
        Returns (r_outer, r_inner).
        """
        M, a = self._M, self._a
        disc = M * M - a * a
        if disc < 0:
            raise ValueError(f"Naked singularity: a ({a}) > M ({M})")
        sqrt_disc = math.sqrt(disc)
        return (M + sqrt_disc, M - sqrt_disc)

    def photon_sphere(self) -> float:
        """
        Prograde equatorial circular photon orbit radius for Kerr.
        r_ph = 2M(1 + cos(2/3 · arccos(∓a/M)))
        Uses prograde (−a/M) by default.
        """
        M, a = self._M, self._a
        # Prograde photon orbit
        return 2.0 * M * (1.0 + math.cos(2.0 / 3.0 * math.acos(-a / M)))

    def isco(self, prograde: bool = True) -> float:
        """
        Kerr ISCO radius (Bardeen, Press & Teukolsky 1972).

        r_isco = M (3 + Z₂ ∓ √((3 − Z₁)(3 + Z₁ + 2Z₂)))

        where:
            Z₁ = 1 + (1−a²/M²)^(1/3) [(1+a/M)^(1/3) + (1−a/M)^(1/3)]
            Z₂ = √(3a²/M² + Z₁²)

        − for prograde, + for retrograde.
        """
        M, a = self._M, self._a
        chi = a / M  # dimensionless spin

        z1 = 1.0 + (1.0 - chi * chi) ** (1.0 / 3.0) * (
            (1.0 + chi) ** (1.0 / 3.0) + (1.0 - chi) ** (1.0 / 3.0)
        )
        z2 = math.sqrt(3.0 * chi * chi + z1 * z1)

        if prograde:
            return M * (3.0 + z2 - math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)))
        else:
            return M * (3.0 + z2 + math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)))

    def ergosphere_radius(self, theta: float) -> float:
        """
        Outer boundary of the ergosphere where g_tt = 0:
            r_ergo = M + √(M² − a² cos²θ)
        """
        M, a = self._M, self._a
        cos_th = math.cos(theta)
        disc = M * M - a * a * cos_th * cos_th
        if disc < 0:
            return M  # fallback
        return M + math.sqrt(disc)

    def effective_potential_timelike(self, r: float, E: float, L: float) -> float:
        """
        Effective radial potential for timelike geodesics in Kerr.

        From the radial equation: Σ² (dr/dτ)² = R(r) where
        R(r) = [E(r² + a²) − aL]² − Δ[r² + (L − aE)²]
        
        Returns R(r) / Σ² — positive means radially allowed.
        """
        M, a = self._M, self._a
        Delta = r * r - 2.0 * M * r + a * a
        R = (E * (r * r + a * a) - a * L) ** 2 - Delta * (r * r + (L - a * E) ** 2)
        return R

    def effective_potential_null(self, r: float, E: float, L: float) -> float:
        """
        Effective radial potential for null geodesics in Kerr.

        R(r) = [E(r² + a²) − aL]² − Δ (L − aE)²
        """
        M, a = self._M, self._a
        Delta = r * r - 2.0 * M * r + a * a
        R = (E * (r * r + a * a) - a * L) ** 2 - Delta * (L - a * E) ** 2
        return R


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

METRICS: Dict[str, type] = {
    'schwarzschild': SchwarzschildMetric,
    'kerr': KerrMetric,
}


def create_metric(
    mass_kg: float,
    metric_type: str = 'schwarzschild',
    spin: float = 0.0,
) -> SpacetimeMetric:
    """
    Factory: create a metric from SI mass and optional spin.

    Parameters
    ----------
    mass_kg     : float   Black hole mass in kilograms.
    metric_type : str     'schwarzschild' or 'kerr'.
    spin        : float   Dimensionless spin a/M ∈ [0, 1). Ignored for Schwarzschild.

    Returns
    -------
    SpacetimeMetric instance.
    """
    M_geom = mass_to_geometric(mass_kg)
    mt = metric_type.lower()

    if mt == 'schwarzschild':
        return SchwarzschildMetric(M_geom)
    elif mt == 'kerr':
        return KerrMetric(M_geom, a_over_M=spin)
    else:
        available = list(METRICS.keys())
        raise ValueError(f"Unknown metric '{metric_type}'. Available: {available}")
