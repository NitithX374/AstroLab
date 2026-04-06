"""
Christoffel Symbols & Geodesic Equation
========================================
Builds the right-hand side (RHS) of the geodesic ODE system from
Christoffel symbols, enabling numerical integration of particle/photon
trajectories in curved spacetime.

The geodesic equation:
    d²x^α/dλ² = −Γ^α_{μν} (dx^μ/dλ)(dx^ν/dλ)

State vector convention (8-dimensional):
    y = [t, r, θ, φ, ṫ, ṙ, θ̇, φ̇]
    where dots denote derivatives w.r.t. affine parameter λ.

References
----------
- Carroll, "Spacetime and Geometry", §3.3
- Misner, Thorne & Wheeler, "Gravitation", §25.3
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from astrolab.physics.metrics import SpacetimeMetric


def geodesic_rhs(
    state: np.ndarray,
    metric: SpacetimeMetric,
) -> np.ndarray:
    """
    Compute the right-hand side of the geodesic ODE system.

    Given a state vector y = [x^0, x^1, x^2, x^3, u^0, u^1, u^2, u^3]
    (coordinates + 4-velocity), return dy/dλ.

    dy/dλ = [u^0, u^1, u^2, u^3,             ← dx^α/dλ = u^α
             −Γ^0_{μν} u^μ u^ν,              ← du^α/dλ = geodesic accel
             −Γ^1_{μν} u^μ u^ν,
             −Γ^2_{μν} u^μ u^ν,
             −Γ^3_{μν} u^μ u^ν]

    Parameters
    ----------
    state : np.ndarray, shape (8,)
        [t, r, θ, φ, ṫ, ṙ, θ̇, φ̇]
    metric : SpacetimeMetric
        The spacetime metric providing Christoffel symbols.

    Returns
    -------
    np.ndarray, shape (8,) — time derivatives dy/dλ.
    """
    # Unpack
    r     = state[1]
    theta = state[2]
    u     = state[4:8]  # 4-velocity components [u^t, u^r, u^θ, u^φ]

    # Get Christoffel symbols at current position
    Gamma = metric.christoffel(r, theta)

    # Compute geodesic acceleration: du^α/dλ = −Γ^α_{μν} u^μ u^ν
    accel = np.zeros(4, dtype=np.float64)
    for alpha in range(4):
        for mu in range(4):
            for nu in range(4):
                accel[alpha] -= Gamma[alpha, mu, nu] * u[mu] * u[nu]

    # Assemble RHS
    dydt = np.zeros(8, dtype=np.float64)
    dydt[0:4] = u         # dx^α/dλ = u^α
    dydt[4:8] = accel     # du^α/dλ = geodesic acceleration

    return dydt


def geodesic_rhs_optimized(
    state: np.ndarray,
    metric: SpacetimeMetric,
) -> np.ndarray:
    """
    Optimized RHS using numpy einsum for the contraction Γ^α_{μν} u^μ u^ν.
    Equivalent to geodesic_rhs but faster for repeated calls.
    """
    r     = state[1]
    theta = state[2]
    u     = state[4:8]

    Gamma = metric.christoffel(r, theta)

    # Contract: accel[α] = −Γ^α_{μν} u^μ u^ν
    accel = -np.einsum('amn,m,n->a', Gamma, u, u)

    dydt = np.zeros(8, dtype=np.float64)
    dydt[0:4] = u
    dydt[4:8] = accel

    return dydt


def compute_norm(
    state: np.ndarray,
    metric: SpacetimeMetric,
) -> float:
    """
    Compute the norm of the 4-velocity: g_{μν} u^μ u^ν.

    This should be:
        -1  for massive particles (timelike geodesics)
         0  for photons           (null geodesics)

    Monitoring this quantity provides a check on integration accuracy.

    Parameters
    ----------
    state : np.ndarray, shape (8,)
    metric : SpacetimeMetric

    Returns
    -------
    float — should be ≈ -1 (timelike) or ≈ 0 (null).
    """
    r     = state[1]
    theta = state[2]
    u     = state[4:8]

    g = metric.metric_tensor(r, theta)
    return float(np.einsum('mn,m,n->', g, u, u))


def normalize_4velocity(
    state: np.ndarray,
    metric: SpacetimeMetric,
    target_norm: float,
) -> np.ndarray:
    """
    Re-normalize the 4-velocity to satisfy the mass-shell constraint.

    For a timelike geodesic:  g_{μν} u^μ u^ν = −1
    For a null geodesic:      g_{μν} u^μ u^ν =  0

    Strategy: adjust u^t to satisfy the constraint while keeping the
    spatial components u^r, u^θ, u^φ fixed.  This minimally perturbs
    the trajectory.

    From  g_{tt}(u^t)² + 2 g_{tφ} u^t u^φ + [spatial terms] = target_norm
    we solve the quadratic for u^t.

    Parameters
    ----------
    state       : np.ndarray, shape (8,)
    metric      : SpacetimeMetric
    target_norm : float  -1 for timelike, 0 for null.

    Returns
    -------
    np.ndarray, shape (8,) — state with corrected u^t.
    """
    result = state.copy()
    r     = state[1]
    theta = state[2]
    g = metric.metric_tensor(r, theta)

    u_r   = state[5]
    u_th  = state[6]
    u_phi = state[7]

    # Spatial part of the norm: sum_{i,j ∈ {r,θ,φ}} g_{ij} u^i u^j
    spatial_norm = (
        g[1, 1] * u_r * u_r +
        g[2, 2] * u_th * u_th +
        g[3, 3] * u_phi * u_phi +
        2.0 * g[1, 2] * u_r * u_th +
        2.0 * g[1, 3] * u_r * u_phi +
        2.0 * g[2, 3] * u_th * u_phi
    )

    # Cross terms with u^t:  2 g_{t,i} u^i  (only g_{tφ} is non-zero for Kerr)
    cross = 2.0 * (g[0, 1] * u_r + g[0, 2] * u_th + g[0, 3] * u_phi)

    # Quadratic in u^t:  g_tt (u^t)² + cross · u^t + (spatial_norm - target_norm) = 0
    A = g[0, 0]
    B = cross
    C = spatial_norm - target_norm

    if abs(A) < 1e-30:
        # Degenerate (on horizon) — don't modify
        return result

    disc = B * B - 4.0 * A * C
    if disc < 0:
        # No real solution — constraint violation too large, return unchanged
        return result

    sqrt_disc = math.sqrt(disc)
    # Choose the root that gives u^t > 0 (future-directed)
    ut1 = (-B + sqrt_disc) / (2.0 * A)
    ut2 = (-B - sqrt_disc) / (2.0 * A)

    # Pick the positive root (or the larger one as fallback)
    if ut1 > 0:
        result[4] = ut1
    elif ut2 > 0:
        result[4] = ut2
    else:
        result[4] = max(ut1, ut2)  # Both negative — pick least negative

    return result


def compute_constants_of_motion(
    state: np.ndarray,
    metric: SpacetimeMetric,
) -> dict:
    """
    Compute conserved quantities along a geodesic.

    For stationary, axisymmetric spacetimes, two Killing vectors give:
        E = −g_{tμ} u^μ    (specific energy, conserved by ∂_t symmetry)
        L =  g_{φμ} u^μ    (specific angular momentum, conserved by ∂_φ symmetry)

    For Kerr, the Carter constant Q provides a third integral:
        Q = u_θ² + cos²θ (a²(κ − E²) + L²/sin²θ)
    where κ = −g_{μν} u^μ u^ν (= 1 for timelike, 0 for null).

    Parameters
    ----------
    state  : np.ndarray, shape (8,)
    metric : SpacetimeMetric

    Returns
    -------
    dict with keys: 'E', 'L', 'Q' (Carter constant, Kerr only), 'norm'
    """
    r     = state[1]
    theta = state[2]
    u     = state[4:8]
    g     = metric.metric_tensor(r, theta)

    # Covariant 4-velocity:  u_α = g_{αμ} u^μ
    u_lower = g @ u

    E = -u_lower[0]   # −g_{tμ} u^μ  (positive for future-directed timelike)
    L = u_lower[3]    #  g_{φμ} u^μ

    result = {'E': float(E), 'L': float(L), 'norm': float(np.dot(u_lower, u))}

    # Carter constant for Kerr
    if hasattr(metric, 'a') and metric.a != 0:
        a = metric.a
        cos_th = math.cos(theta)
        sin_th = math.sin(theta)
        kappa = -result['norm']   # +1 for timelike, 0 for null

        u_theta_covariant = u_lower[2]
        if abs(sin_th) > 1e-15:
            Q = (u_theta_covariant ** 2 +
                 cos_th ** 2 * (a * a * (kappa - E * E) + L * L / (sin_th ** 2)))
        else:
            Q = u_theta_covariant ** 2
        result['Q'] = float(Q)

    return result
