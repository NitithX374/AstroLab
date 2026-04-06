"""
GR Observables
===============
Computes physically meaningful observable quantities from geodesic data
and spacetime metrics.

Observables
-----------
- Gravitational time dilation
- Gravitational redshift
- Gravitational lensing (deflection angle)
- Frame-dragging angular velocity
- Orbital precession
- Effective potential (radial classification)
- Impact parameter
- Shapiro time delay
- Proper distance

All functions work in geometric units (G = c = 1) and accept/return
values scaled by the black hole mass M.

References
----------
- Weinberg, "Gravitation and Cosmology" (1972), Ch. 8
- Hartle, "Gravity" (2003), Ch. 9
- Schutz, "A First Course in General Relativity" (2009), Ch. 11
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from astrolab.physics.metrics import SpacetimeMetric, SchwarzschildMetric, KerrMetric


# ─────────────────────────────────────────────────────────────────────────────
# Time Dilation
# ─────────────────────────────────────────────────────────────────────────────

def gravitational_time_dilation(
    r: float,
    metric: SpacetimeMetric,
    theta: float = math.pi / 2,
    v_tangential: float = 0.0,
) -> float:
    """
    Compute gravitational time dilation factor dτ/dt for a static observer.

    For a static observer (dr = dθ = dφ = 0):
        dτ/dt = √(-g_tt)

    With tangential velocity v (fraction of c):
        dτ/dt = √(-g_tt - g_φφ ω²)   where ω = v / (r sinθ)

    For Schwarzschild:
        dτ/dt = √(1 - 2M/r)

    Parameters
    ----------
    r     : float  Radial coordinate in geometric units
    metric: SpacetimeMetric
    theta : float  Polar angle (default: equatorial plane)
    v_tangential : float  Tangential velocity as fraction of c (0 to 1)

    Returns
    -------
    float — dτ/dt ratio. Values < 1 mean time runs slower (closer to BH).
    """
    g = metric.metric_tensor(r, theta)

    if v_tangential == 0.0:
        # Static observer
        if g[0, 0] >= 0:
            return 0.0  # Inside ergosphere or horizon — no static observers
        return math.sqrt(-g[0, 0])
    else:
        # Moving observer with angular velocity
        sin_th = math.sin(theta)
        if abs(sin_th) < 1e-15 or abs(r) < 1e-15:
            return 0.0
        omega = v_tangential / (r * sin_th)
        val = -(g[0, 0] + 2.0 * g[0, 3] * omega + g[3, 3] * omega * omega)
        if val <= 0:
            return 0.0
        return math.sqrt(val)


def time_dilation_schwarzschild(r: float, M: float) -> float:
    """
    Schwarzschild time dilation (simplified, exact formula).

    dτ/dt = √(1 − 2M/r)

    Parameters
    ----------
    r : float  Radial coordinate (geometric units)
    M : float  Black hole mass (geometric units)

    Returns
    -------
    float — time dilation factor
    """
    if r <= 2.0 * M:
        return 0.0
    return math.sqrt(1.0 - 2.0 * M / r)


# ─────────────────────────────────────────────────────────────────────────────
# Gravitational Redshift
# ─────────────────────────────────────────────────────────────────────────────

def gravitational_redshift(
    r_emit: float,
    r_obs: float,
    metric: SpacetimeMetric,
    theta: float = math.pi / 2,
) -> Dict[str, float]:
    """
    Compute gravitational redshift between two radial positions.

    For static observers in a stationary spacetime:
        1 + z = √(g_tt(observer) / g_tt(emitter))

    A photon emitted at r_emit and observed at r_obs experiences:
        z > 0 : redshift  (emitter deeper in potential well)
        z < 0 : blueshift (observer deeper in potential well)

    Parameters
    ----------
    r_emit : float  Emission radius (geometric units)
    r_obs  : float  Observer radius (geometric units)
    metric : SpacetimeMetric
    theta  : float  Polar angle

    Returns
    -------
    dict with 'z' (redshift), 'wavelength_ratio', 'frequency_ratio'
    """
    g_emit = metric.metric_tensor(r_emit, theta)
    g_obs = metric.metric_tensor(r_obs, theta)

    gtt_emit = g_emit[0, 0]
    gtt_obs = g_obs[0, 0]

    # Both must be negative (outside horizon)
    if gtt_emit >= 0 or gtt_obs >= 0:
        return {
            'z': float('inf'),
            'wavelength_ratio': float('inf'),
            'frequency_ratio': 0.0,
            'note': 'Emitter or observer inside horizon/ergosphere — infinite redshift',
        }

    one_plus_z = math.sqrt(gtt_obs / gtt_emit)
    z = one_plus_z - 1.0

    return {
        'z': z,
        'wavelength_ratio': one_plus_z,      # λ_obs / λ_emit
        'frequency_ratio': 1.0 / one_plus_z, # f_obs / f_emit
    }


def redshift_schwarzschild(r_emit: float, r_obs: float, M: float) -> float:
    """
    Schwarzschild redshift (exact formula for static observers).

    1 + z = √((1 − 2M/r_obs) / (1 − 2M/r_emit))
    """
    if r_emit <= 2.0 * M or r_obs <= 2.0 * M:
        return float('inf')
    return math.sqrt((1.0 - 2.0 * M / r_obs) / (1.0 - 2.0 * M / r_emit)) - 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Gravitational Lensing
# ─────────────────────────────────────────────────────────────────────────────

def deflection_angle_weak_field(
    impact_parameter: float,
    M: float,
) -> float:
    """
    Weak-field gravitational deflection angle (Einstein 1915).

    Δφ = 4M / b

    Valid for b >> M (light passing far from the black hole).

    Parameters
    ----------
    impact_parameter : float  b = L/E  (geometric units)
    M                : float  Black hole mass (geometric units)

    Returns
    -------
    float — deflection angle in radians
    """
    if impact_parameter <= 0:
        return float('inf')
    return 4.0 * M / impact_parameter


def critical_impact_parameter(metric: SpacetimeMetric) -> float:
    """
    Critical impact parameter b_c for photon capture.

    For Schwarzschild: b_c = 3√3 M ≈ 5.196 M

    Photons with b < b_c are captured; b > b_c are deflected and escape.
    b = b_c produces the unstable circular photon orbit.

    For Kerr, this depends on the direction (prograde/retrograde) and is
    more complex. This returns the equatorial prograde value.
    """
    M = metric.M

    if isinstance(metric, SchwarzschildMetric):
        return 3.0 * math.sqrt(3.0) * M
    elif isinstance(metric, KerrMetric):
        a = metric.a
        r_ph = metric.photon_sphere()
        # b_c = (r_ph² + a²) / (a + r_ph √(1 - 2M/r_ph + a²/r_ph²))
        # Simplified for prograde equatorial:
        Delta_ph = r_ph * r_ph - 2.0 * M * r_ph + a * a
        if Delta_ph <= 0:
            return 3.0 * math.sqrt(3.0) * M  # Fallback
        return (r_ph * r_ph + a * a) * r_ph / (r_ph * r_ph - a * a) - a if (r_ph * r_ph - a * a) > 0 else 3.0 * math.sqrt(3.0) * M
    else:
        return 3.0 * math.sqrt(3.0) * M


def lensing_magnification(
    theta_E: float,
    beta: float,
) -> Tuple[float, float]:
    """
    Point-source gravitational lensing magnification.

    For a point mass lens, the Einstein ring angle is θ_E.
    The source is at angle β from the lens-observer axis.

    Two images form at angles:
        θ± = ½(β ± √(β² + 4θ_E²))

    Magnification of each image:
        μ± = |θ± / β · dθ±/dβ|

    Parameters
    ----------
    theta_E : float  Einstein angle
    beta    : float  Source-lens angular separation

    Returns
    -------
    (mu_plus, mu_minus) — magnifications of the two images
    """
    if abs(beta) < 1e-15:
        return (float('inf'), float('inf'))  # Perfect alignment → Einstein ring

    u = beta / theta_E
    u2 = u * u

    mu_plus  = 0.5 * (u2 + 2) / (u * math.sqrt(u2 + 4)) + 0.5
    mu_minus = 0.5 * (u2 + 2) / (u * math.sqrt(u2 + 4)) - 0.5

    return (abs(mu_plus), abs(mu_minus))


# ─────────────────────────────────────────────────────────────────────────────
# Frame-Dragging
# ─────────────────────────────────────────────────────────────────────────────

def frame_dragging_velocity(
    r: float,
    theta: float,
    metric: SpacetimeMetric,
) -> Dict[str, float]:
    """
    Compute the frame-dragging angular velocity and linear velocity.

    ω = −g_{tφ} / g_{φφ}

    This is the angular velocity at which local inertial frames are
    dragged by the rotating black hole (Lense-Thirring effect).

    Parameters
    ----------
    r     : float  Radial coordinate (geometric units)
    theta : float  Polar angle
    metric: SpacetimeMetric

    Returns
    -------
    dict with 'omega' (angular velocity), 'v_linear' (linear velocity at r sinθ)
    """
    omega = metric.frame_dragging_omega(r, theta)
    sin_th = math.sin(theta)
    v_linear = omega * r * sin_th  # As fraction of c (geometric units)

    return {
        'omega': omega,
        'v_linear': v_linear,
        'v_fraction_c': abs(v_linear),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Effective Potential
# ─────────────────────────────────────────────────────────────────────────────

def effective_potential_timelike(
    r: float,
    L: float,
    M: float,
    a: float = 0.0,
    E: float = 1.0,
) -> float:
    """
    Effective potential for timelike (massive particle) radial motion.

    Schwarzschild (a = 0):
        V_eff(r) = (1 − 2M/r)(1 + L²/r²)

    The particle moves where E² ≥ V_eff(r).
    Turning points occur at E² = V_eff(r).

    Parameters
    ----------
    r : float  Radial coordinate
    L : float  Specific angular momentum
    M : float  Black hole mass
    a : float  Spin parameter (dimensional)
    E : float  Specific energy (used for Kerr potential)

    Returns
    -------
    float — effective potential value
    """
    if r < 1e-15:
        return float('inf')

    if abs(a) < 1e-15:
        # Schwarzschild
        return (1.0 - 2.0 * M / r) * (1.0 + L * L / (r * r))
    else:
        # Kerr — return R(r) / r⁴ where R(r) is the radial potential
        Delta = r * r - 2.0 * M * r + a * a
        return (E * (r * r + a * a) - a * L) ** 2 - Delta * (r * r + (L - a * E) ** 2)


def effective_potential_null(
    r: float,
    L: float,
    M: float,
    a: float = 0.0,
    E: float = 1.0,
) -> float:
    """
    Effective potential for null (photon) radial motion.

    Schwarzschild (a = 0):
        V_eff(r) = (1 − 2M/r) L² / r²

    Parameters
    ----------
    r : float  Radial coordinate
    L : float  Specific angular momentum (or impact parameter b = L/E)
    M : float  Black hole mass
    a : float  Spin parameter
    E : float  Specific energy

    Returns
    -------
    float — effective potential value
    """
    if r < 1e-15:
        return float('inf')

    if abs(a) < 1e-15:
        return (1.0 - 2.0 * M / r) * L * L / (r * r)
    else:
        Delta = r * r - 2.0 * M * r + a * a
        return (E * (r * r + a * a) - a * L) ** 2 - Delta * (L - a * E) ** 2


def compute_potential_curve(
    L: float,
    M: float,
    r_min: float = 2.5,
    r_max: float = 50.0,
    n_points: int = 500,
    is_null: bool = False,
    a: float = 0.0,
    E: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the effective potential as a function of r.

    Parameters
    ----------
    L       : float  Specific angular momentum
    M       : float  Black hole mass
    r_min   : float  Minimum r (in units of M)
    r_max   : float  Maximum r (in units of M)
    n_points: int    Number of sample points
    is_null : bool   True for photon potential
    a       : float  Spin parameter
    E       : float  Specific energy

    Returns
    -------
    (r_array, V_array)  numpy arrays
    """
    r_arr = np.linspace(r_min * M, r_max * M, n_points)
    func = effective_potential_null if is_null else effective_potential_timelike

    V_arr = np.array([func(r, L, M, a=a, E=E) for r in r_arr])

    return r_arr, V_arr


# ─────────────────────────────────────────────────────────────────────────────
# Orbital Precession
# ─────────────────────────────────────────────────────────────────────────────

def perihelion_precession_rate(
    semi_major: float,
    eccentricity: float,
    M: float,
) -> float:
    """
    GR perihelion precession per orbit (Schwarzschild, weak-field limit).

    δφ = 6π M / (a(1 − e²))

    where a is the semi-major axis and e is the eccentricity (in geometric units).

    This reproduces Mercury's famous 43 arcsec/century when applied with
    solar mass and Mercury's orbital parameters.

    Parameters
    ----------
    semi_major    : float  Semi-major axis (geometric units)
    eccentricity  : float  Orbital eccentricity (0 < e < 1)
    M             : float  Central mass (geometric units)

    Returns
    -------
    float — precession angle per orbit in radians
    """
    if eccentricity >= 1.0 or eccentricity < 0:
        return 0.0
    p = semi_major * (1.0 - eccentricity * eccentricity)
    if p <= 0:
        return float('inf')
    return 6.0 * math.pi * M / p


# ─────────────────────────────────────────────────────────────────────────────
# Shapiro Time Delay
# ─────────────────────────────────────────────────────────────────────────────

def shapiro_delay(
    r_emit: float,
    r_obs: float,
    r_closest: float,
    M: float,
) -> float:
    """
    Shapiro time delay for a signal passing through curved spacetime.

    The excess travel time relative to flat space for a signal that
    passes a mass M at closest approach r_closest:

    Δt = 2M · ln((r_emit + √(r_emit² − r_closest²)) ·
                  (r_obs  + √(r_obs²  − r_closest²)) / r_closest²)
        + M · (√(r_emit² − r_closest²)/r_emit
              + √(r_obs² − r_closest²)/r_obs)

    Simplified (weak-field, r >> M):
    Δt ≈ 2M · [1 + ln(4 r_emit r_obs / r_closest²)]

    Parameters
    ----------
    r_emit    : float  Emitter distance from center
    r_obs     : float  Observer distance from center
    r_closest : float  Closest approach distance of signal
    M         : float  Central mass

    Returns
    -------
    float — time delay in geometric units (multiply by G·M_kg/c³ for seconds)
    """
    if r_closest <= 0 or r_emit <= 0 or r_obs <= 0:
        return 0.0

    # Ensure r_closest < r_emit and r_closest < r_obs
    r_closest = min(r_closest, r_emit, r_obs)

    term1 = r_emit * r_emit - r_closest * r_closest
    term2 = r_obs * r_obs - r_closest * r_closest

    if term1 < 0 or term2 < 0:
        return 0.0

    sqrt1 = math.sqrt(term1)
    sqrt2 = math.sqrt(term2)

    if r_closest * r_closest < 1e-30:
        return 0.0

    delay = 2.0 * M * math.log(
        (r_emit + sqrt1) * (r_obs + sqrt2) / (r_closest * r_closest)
    )

    return delay


# ─────────────────────────────────────────────────────────────────────────────
# Proper Distance
# ─────────────────────────────────────────────────────────────────────────────

def proper_distance_schwarzschild(
    r1: float,
    r2: float,
    M: float,
    n_points: int = 1000,
) -> float:
    """
    Proper (ruler) distance between two radial coordinates in Schwarzschild.

    d_proper = ∫_{r1}^{r2} dr / √(1 − 2M/r)

    This is larger than the coordinate distance (r2 − r1) due to
    spatial curvature near the black hole.

    Computed via numerical integration (trapezoidal rule).

    Parameters
    ----------
    r1, r2 : float  Radial coordinates (r1 < r2, both > 2M)
    M      : float  Black hole mass
    n_points : int  Integration resolution

    Returns
    -------
    float — proper distance in geometric units
    """
    r_min = max(r1, 2.0 * M + 1e-10)
    r_max = max(r2, r_min + 1e-10)

    r_arr = np.linspace(r_min, r_max, n_points)
    dr = r_arr[1] - r_arr[0]

    integrand = np.array([
        1.0 / math.sqrt(1.0 - 2.0 * M / r) if r > 2.0 * M else 0.0
        for r in r_arr
    ])

    return float(np.trapz(integrand, r_arr))


# ─────────────────────────────────────────────────────────────────────────────
# Summary Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_observables_report(
    r: float,
    metric: SpacetimeMetric,
    theta: float = math.pi / 2,
    r_observer: float = None,
) -> Dict[str, float]:
    """
    Compute all observables at a given position for quick reference.

    Parameters
    ----------
    r          : float  Radial coordinate
    metric     : SpacetimeMetric
    theta      : float  Polar angle
    r_observer : float  Observer position (for redshift). Default: 1000M

    Returns
    -------
    dict of observable names → values
    """
    M = metric.M
    if r_observer is None:
        r_observer = 1000.0 * M

    report: Dict[str, float] = {}

    # Time dilation
    report['time_dilation_factor'] = gravitational_time_dilation(r, metric, theta)

    # Redshift (if outside horizon)
    if not metric.is_inside_horizon(r) and not metric.is_inside_horizon(r_observer):
        rs = gravitational_redshift(r, r_observer, metric, theta)
        report['redshift_z'] = rs['z']
        report['frequency_ratio'] = rs['frequency_ratio']

    # Frame-dragging
    if metric.spin > 0:
        fd = frame_dragging_velocity(r, theta, metric)
        report['frame_drag_omega'] = fd['omega']
        report['frame_drag_v_c'] = fd['v_fraction_c']

    # Proper distance to some reference
    if isinstance(metric, SchwarzschildMetric) and not metric.is_inside_horizon(r):
        report['proper_distance_to_horizon'] = proper_distance_schwarzschild(
            2.0 * M + 0.001 * M, r, M
        )

    return report
