"""
Newtonian Gravity Engine
========================
Computes gravitational forces and accelerations for an N-body system.

All physics is in SI units.  The gravitational constant used is:
    G = 6.67430e-11  m³ kg⁻¹ s⁻²

Softening
---------
An optional softening length ε (default 0) can be supplied to avoid
singularities when two point-mass bodies approach very close:
    F = G m1 m2 / (r² + ε²)   ·  r̂

Public API
----------
compute_gravitational_forces(bodies, softening) -> List[Vector3D]
compute_accelerations(bodies, softening)        -> List[Vector3D]
"""

from typing import List

from astrolab.core.models import CelestialBody, Vector3D

# Gravitational constant  [m³ kg⁻¹ s⁻²]
G: float = 6.67430e-11


def compute_gravitational_forces(
    bodies: List[CelestialBody],
    softening: float = 0.0,
) -> List[Vector3D]:
    """
    Compute the net gravitational force acting on each body due to all others.

    Uses Newton's law of universal gravitation in vector form:

        F⃗ᵢⱼ = G mᵢ mⱼ / (|r⃗ᵢⱼ|² + ε²) · r̂ᵢⱼ

    Newton's third law is exploited so only N(N-1)/2 pairs are evaluated.

    Parameters
    ----------
    bodies    : list of CelestialBody objects (mutated in place only by caller).
    softening : plummer softening length ε in metres [default 0].

    Returns
    -------
    List[Vector3D] — net force on each body, same order as `bodies`.
    """
    n = len(bodies)
    forces: List[Vector3D] = [Vector3D.zero() for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            b1 = bodies[i]
            b2 = bodies[j]

            # Displacement vector from b1 → b2
            r_vec = b2.position - b1.position
            dist_sq = r_vec.magnitude_sq() + softening ** 2

            if dist_sq == 0.0:
                continue  # identical positions — skip (shouldn't happen in normal sim)

            dist = dist_sq ** 0.5

            # Scalar force magnitude   F = G m1 m2 / (r² + ε²)
            force_mag = G * b1.mass * b2.mass / dist_sq

            # Force vector (directed from b1 toward b2)
            force_vec = r_vec * (force_mag / dist)

            forces[i] = forces[i] + force_vec   # b1 attracted toward b2
            forces[j] = forces[j] - force_vec   # b2 attracted toward b1 (Newton 3)

    return forces


def compute_accelerations(
    bodies: List[CelestialBody],
    softening: float = 0.0,
) -> List[Vector3D]:
    """
    Compute acceleration a⃗ = F⃗/m for each body.

    Parameters
    ----------
    bodies    : list of CelestialBody objects.
    softening : plummer softening length ε (passed through to force calc).

    Returns
    -------
    List[Vector3D] — acceleration vector for each body [m s⁻²].
    """
    forces = compute_gravitational_forces(bodies, softening)
    return [
        forces[i] / body.mass if body.mass > 0.0 else Vector3D.zero()
        for i, body in enumerate(bodies)
    ]
