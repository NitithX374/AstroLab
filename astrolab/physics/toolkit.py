"""
Astrophysics Toolkit
====================
Pure-function scientific calculations. All values use SI units internally.

Functions:
  - escape_velocity(body)
  - orbital_period(body, primary)
  - gravitational_force(b1, b2)
  - kinetic_energy(body)
  - potential_energy(b1, b2)
  - total_system_energy(bodies)
  - schwarzschild_radius(body)
  - lagrange_points(primary, secondary)
"""

import math
from typing import Dict, List, Tuple
from astrolab.core.models import CelestialBody, Vector3D

# Gravitational constant (m^3 kg^-1 s^-2)
G = 6.67430e-11
# Speed of light (m/s)
C = 2.99792458e8


def escape_velocity(body: CelestialBody) -> float:
    """
    Compute the surface escape velocity of a body.
    Requires body.radius to be set (>0).
    v_e = sqrt(2 * G * M / R)
    Returns: escape velocity in m/s
    """
    if body.radius <= 0:
        raise ValueError(f"Body '{body.name}' has no valid radius set.")
    return math.sqrt(2 * G * body.mass / body.radius)


def orbital_period(orbiting: CelestialBody, primary: CelestialBody) -> float:
    """
    Compute Keplerian orbital period using the current separation.
    T = 2*pi * sqrt(a^3 / (G * M))  where a = separation distance.
    Returns: orbital period in seconds
    """
    sep = (orbiting.position - primary.position).magnitude()
    if sep == 0:
        raise ValueError("Bodies are co-located; cannot compute orbital period.")
    return 2 * math.pi * math.sqrt(sep**3 / (G * primary.mass))


def gravitational_force(b1: CelestialBody, b2: CelestialBody) -> Tuple[float, Vector3D]:
    """
    Compute gravitational force magnitude and direction vector from b1 toward b2.
    F = G * m1 * m2 / r^2
    Returns: (force_magnitude_N, unit_vector_b1_to_b2)
    """
    r_vec = b2.position - b1.position
    dist = r_vec.magnitude()
    if dist == 0:
        raise ValueError("Bodies are co-located; gravitational force is undefined.")
    force_mag = G * b1.mass * b2.mass / dist**2
    unit_vec = r_vec / dist
    return force_mag, unit_vec


def kinetic_energy(body: CelestialBody) -> float:
    """
    KE = 0.5 * m * v^2
    Returns: kinetic energy in Joules
    """
    v = body.velocity.magnitude()
    return 0.5 * body.mass * v**2


def potential_energy(b1: CelestialBody, b2: CelestialBody) -> float:
    """
    Gravitational potential energy between two bodies.
    U = -G * m1 * m2 / r
    Returns: potential energy in Joules (negative value)
    """
    dist = (b1.position - b2.position).magnitude()
    if dist == 0:
        return float('-inf')
    return -G * b1.mass * b2.mass / dist


def total_system_energy(bodies: List[CelestialBody]) -> Dict[str, float]:
    """
    Compute the total kinetic, potential, and mechanical energy of the system.
    Returns: {'kinetic': KE, 'potential': PE, 'total': KE+PE}
    """
    ke = sum(kinetic_energy(b) for b in bodies)
    pe = 0.0
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            pe += potential_energy(bodies[i], bodies[j])
    return {'kinetic': ke, 'potential': pe, 'total': ke + pe}


def schwarzschild_radius(body: CelestialBody) -> float:
    """
    Compute the Schwarzschild radius (event horizon radius for a black hole).
    Rs = 2 * G * M / c^2
    Returns: Schwarzschild radius in meters
    """
    return 2 * G * body.mass / C**2


def lagrange_points(primary: CelestialBody, secondary: CelestialBody) -> Dict[str, Vector3D]:
    """
    Approximate the 5 Lagrange points of a two-body system.
    Primary and secondary are assumed to be in the same orbital plane (z=0).

    Uses the mass ratio and separation to estimate L1-L5 positions.
    Returns: dict mapping 'L1'..'L5' to Vector3D positions (approximated)
    """
    M1 = primary.mass
    M2 = secondary.mass
    mu = M2 / (M1 + M2)   # mass ratio

    r_vec = secondary.position - primary.position
    d = r_vec.magnitude()
    if d == 0:
        raise ValueError("Bodies are co-located; Lagrange points undefined.")

    # Unit vector from primary to secondary
    u = r_vec / d
    # Perpendicular unit vector (in XY plane)
    u_perp = Vector3D(-u.y, u.x, 0.0)

    # Center of mass position
    com = primary.position + r_vec * mu

    # L1: between primary and secondary, closer to secondary
    l1_dist = d * (1 - (mu / 3) ** (1 / 3))
    L1 = primary.position + u * l1_dist

    # L2: beyond secondary (away from primary)
    l2_dist = d * (1 + (mu / 3) ** (1 / 3))
    L2 = primary.position + u * l2_dist

    # L3: opposite side of primary from secondary
    l3_dist = d * (1 + 5 * mu / 12)
    L3 = primary.position - u * l3_dist

    # L4: 60° ahead of secondary in orbit (equilateral triangle)
    # Rotate r_vec 60° toward the direction of motion
    half_d = d / 2.0
    h = d * math.sqrt(3) / 2.0
    L4 = com + (secondary.position - primary.position) * (-0.5 / d * d) + u_perp * h * (1 - mu)
    # Simplified: equilateral triangle apex positions
    L4 = primary.position + u * (d / 2.0) + u_perp * (d * math.sqrt(3) / 2.0)
    L5 = primary.position + u * (d / 2.0) - u_perp * (d * math.sqrt(3) / 2.0)

    return {'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4, 'L5': L5}
