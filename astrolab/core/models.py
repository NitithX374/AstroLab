"""
Core Data Models
================
All quantities are strictly in SI units:
  - distance  : metres (m)
  - mass      : kilograms (kg)
  - velocity  : metres per second (m/s)
  - time      : seconds (s)

Classes
-------
Vector3D       : Immutable-style 3D vector with full operator overloads
CelestialBody  : A gravitational body with physical properties
SimulationState: Full snapshot of a simulation at a point in time
"""

import math
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Vector3D
# ---------------------------------------------------------------------------

@dataclass
class Vector3D:
    """
    Immutable-style 3D Cartesian vector.

    All arithmetic operators return new Vector3D instances so the original
    is never mutated — safe to pass around freely.
    """
    x: float
    y: float
    z: float

    # ── Arithmetic ──────────────────────────────────────────────────────────

    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> 'Vector3D':
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> 'Vector3D':
        if scalar == 0.0:
            raise ZeroDivisionError("Cannot divide a vector by zero.")
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> 'Vector3D':
        return Vector3D(-self.x, -self.y, -self.z)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector3D):
            return NotImplemented
        return (self.x == other.x and self.y == other.y and self.z == other.z)

    # ── Linear algebra ──────────────────────────────────────────────────────

    def dot(self, other: 'Vector3D') -> float:
        """Dot product: self · other"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector3D') -> 'Vector3D':
        """Cross product: self × other"""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def magnitude(self) -> float:
        """Euclidean (L2) norm: |v|"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def magnitude_sq(self) -> float:
        """Squared magnitude: |v|² (avoids sqrt, useful for comparisons)"""
        return self.x**2 + self.y**2 + self.z**2

    def normalized(self) -> 'Vector3D':
        """Return the unit vector in the same direction, or zero if |v|=0."""
        mag = self.magnitude()
        if mag == 0.0:
            return Vector3D(0.0, 0.0, 0.0)
        return self / mag

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> 'Vector3D':
        return cls(float(t[0]), float(t[1]), float(t[2]))

    @classmethod
    def zero(cls) -> 'Vector3D':
        """Factory: return the zero vector."""
        return cls(0.0, 0.0, 0.0)

    # ── Display ──────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return f"({self.x:g}, {self.y:g}, {self.z:g})"

    def __repr__(self) -> str:
        return f"Vector3D({self.x!r}, {self.y!r}, {self.z!r})"


# ---------------------------------------------------------------------------
# CelestialBody
# ---------------------------------------------------------------------------

@dataclass
class CelestialBody:
    """
    Represents any gravitational body — star, planet, moon, asteroid, etc.

    Parameters
    ----------
    name      : str   Unique human-readable identifier.
    mass      : float Mass in kilograms [kg].
    position  : Vector3D   3D position in metres [m].
    velocity  : Vector3D   3D velocity in metres per second [m/s].
    radius    : float Physical radius in metres [m].
                      Required for collision detection and escape velocity.
    body_type : str   Descriptive category (star | planet | moon | asteroid |
                      black_hole | comet | unknown).
    color     : str   Hint for visualisation hooks; has no physics effect.
    """

    name:      str
    mass:      float
    position:  Vector3D
    velocity:  Vector3D
    radius:    float = 0.0
    body_type: str   = "unknown"
    color:     str   = "white"

    # ── Derived properties ───────────────────────────────────────────────────

    @property
    def speed(self) -> float:
        """Current scalar speed |v| in m/s."""
        return self.velocity.magnitude()

    @property
    def distance_from_origin(self) -> float:
        """Distance from coordinate origin |r| in metres."""
        return self.position.magnitude()

    def kinetic_energy(self) -> float:
        """KE = ½mv²  [J]"""
        return 0.5 * self.mass * self.speed**2

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "name":      self.name,
            "mass":      self.mass,
            "position":  self.position.to_tuple(),
            "velocity":  self.velocity.to_tuple(),
            "radius":    self.radius,
            "body_type": self.body_type,
            "color":     self.color,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CelestialBody':
        return cls(
            name=data['name'],
            mass=data['mass'],
            position=Vector3D.from_tuple(data['position']),
            velocity=Vector3D.from_tuple(data['velocity']),
            radius=data.get('radius', 0.0),
            body_type=data.get('body_type', 'unknown'),
            color=data.get('color', 'white'),
        )

    # ── Display ──────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return (
            f"CelestialBody(name={self.name!r}, mass={self.mass:.3e} kg, "
            f"pos={self.position}, vel={self.velocity})"
        )


# ---------------------------------------------------------------------------
# SimulationState
# ---------------------------------------------------------------------------

@dataclass
class SimulationState:
    """
    A complete, self-contained snapshot of the simulation at a single moment.

    Parameters
    ----------
    bodies    : List[CelestialBody]  All bodies in the system.
    time      : float                Elapsed simulation time in seconds [s].
    dt        : float                Current timestep in seconds [s].
    step      : int                  Number of steps completed so far.
    integrator: str                  Name of the active integrator.
    """

    bodies:     List[CelestialBody]
    time:       float = 0.0
    dt:         float = 60.0
    step:       int   = 0
    integrator: str   = "rk4"

    # ── Convenience accessors ────────────────────────────────────────────────

    def get_body(self, name: str) -> Optional[CelestialBody]:
        """Return the body with the given name, or None if not found."""
        for b in self.bodies:
            if b.name == name:
                return b
        return None

    def body_names(self) -> List[str]:
        """Return a sorted list of all body names."""
        return sorted(b.name for b in self.bodies)

    def body_count(self) -> int:
        return len(self.bodies)

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "time":       self.time,
            "dt":         self.dt,
            "step":       self.step,
            "integrator": self.integrator,
            "bodies":     [b.to_dict() for b in self.bodies],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SimulationState':
        return cls(
            bodies=[CelestialBody.from_dict(bd) for bd in data['bodies']],
            time=data.get('time', 0.0),
            dt=data.get('dt', 60.0),
            step=data.get('step', 0),
            integrator=data.get('integrator', 'rk4'),
        )

    def save(self, filepath: str) -> None:
        """Persist state to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'SimulationState':
        """Load state from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    # ── Display ──────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        days = self.time / 86_400
        return (
            f"SimulationState(bodies={self.body_count()}, "
            f"time={days:.2f} days, step={self.step}, dt={self.dt}s, "
            f"integrator={self.integrator!r})"
        )


# ---------------------------------------------------------------------------
# GeodesicTrajectory  (General Relativity)
# ---------------------------------------------------------------------------

@dataclass
class GeodesicTrajectoryPoint:
    """
    A single point on a geodesic in curved spacetime.

    Stores Boyer-Lindquist coordinates (t, r, θ, φ) and the corresponding
    4-velocity components, along with diagnostics.
    """
    affine_param: float          # Affine parameter λ
    t:            float          # Coordinate time
    r:            float          # Radial coordinate (geometric units)
    theta:        float          # Polar angle [rad]
    phi:          float          # Azimuthal angle [rad]
    ut:           float = 0.0   # dt/dλ
    ur:           float = 0.0   # dr/dλ
    utheta:       float = 0.0   # dθ/dλ
    uphi:         float = 0.0   # dφ/dλ
    proper_time:  float = 0.0   # Accumulated proper time
    norm:         float = 0.0   # g_μν u^μ u^ν (diagnostic)

    @property
    def x(self) -> float:
        """Cartesian x = r sinθ cosφ"""
        return self.r * math.sin(self.theta) * math.cos(self.phi)

    @property
    def y(self) -> float:
        """Cartesian y = r sinθ sinφ"""
        return self.r * math.sin(self.theta) * math.sin(self.phi)

    @property
    def z(self) -> float:
        """Cartesian z = r cosθ"""
        return self.r * math.cos(self.theta)

    def cartesian(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_dict(self) -> Dict:
        return {
            'lambda': self.affine_param,
            't': self.t, 'r': self.r, 'theta': self.theta, 'phi': self.phi,
            'ut': self.ut, 'ur': self.ur, 'utheta': self.utheta, 'uphi': self.uphi,
            'proper_time': self.proper_time, 'norm': self.norm,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'GeodesicTrajectoryPoint':
        return cls(
            affine_param=data.get('lambda', 0.0),
            t=data.get('t', 0.0),
            r=data.get('r', 0.0),
            theta=data.get('theta', math.pi / 2),
            phi=data.get('phi', 0.0),
            ut=data.get('ut', 0.0),
            ur=data.get('ur', 0.0),
            utheta=data.get('utheta', 0.0),
            uphi=data.get('uphi', 0.0),
            proper_time=data.get('proper_time', 0.0),
            norm=data.get('norm', 0.0),
        )
