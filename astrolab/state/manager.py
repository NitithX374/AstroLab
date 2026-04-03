"""
State Manager
=============
Thin wrapper around ``SimulationState`` providing a dictionary-style
interface for body lookup, plus JSON import/export helpers.

The engine operates directly on ``SimulationState``; the manager is
used by the CLI layer to bridge user commands and simulation state.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from astrolab.core.models import CelestialBody, SimulationState, Vector3D


class StateManager:
    """
    Manages the lifecycle of a ``SimulationState``.

    Provides:
    - O(1) body lookup by name via an internal index
    - add / remove helpers that keep the index consistent
    - JSON save / load via ``SimulationState.save()`` / ``SimulationState.load()``
    """

    def __init__(self, state: Optional[SimulationState] = None) -> None:
        if state is None:
            state = SimulationState(bodies=[], time=0.0, dt=60.0)
        self.state: SimulationState = state
        self._index: Dict[str, CelestialBody] = {b.name: b for b in state.bodies}

    # ── Body management ──────────────────────────────────────────────────────

    def add_body(self, body: CelestialBody) -> bool:
        """
        Add a body.  Returns False (no-op) if a body with the same name
        already exists.
        """
        if body.name in self._index:
            return False
        self.state.bodies.append(body)
        self._index[body.name] = body
        return True

    def remove_body(self, name: str) -> bool:
        """Remove a body by name.  Returns False if not found."""
        body = self._index.pop(name, None)
        if body is None:
            return False
        self.state.bodies.remove(body)
        return True

    def get_body(self, name: str) -> Optional[CelestialBody]:
        """Return the body with the given name, or None."""
        return self._index.get(name)

    def get_all_bodies(self) -> List[CelestialBody]:
        """Return a copy of the body list (order matches insertion order)."""
        return list(self.state.bodies)

    def clear(self) -> None:
        """Remove all bodies and reset simulation time."""
        self.state.bodies.clear()
        self._index.clear()
        self.state.time = 0.0
        self.state.step = 0

    # ── Persistence ──────────────────────────────────────────────────────────

    def export_state(self, filepath: str) -> bool:
        """Save state to a JSON file.  Returns True on success."""
        try:
            self.state.save(filepath)
            return True
        except Exception as exc:
            print(f"  [!] Export failed: {exc}")
            return False

    def import_state(self, filepath: str) -> bool:
        """Load state from a JSON file.  Returns True on success."""
        if not os.path.exists(filepath):
            print(f"  [!] File not found: {filepath!r}")
            return False
        try:
            new_state = SimulationState.load(filepath)
            self.state.bodies     = new_state.bodies
            self.state.time       = new_state.time
            self.state.dt         = new_state.dt
            self.state.step       = new_state.step
            self.state.integrator = new_state.integrator
            self._index = {b.name: b for b in self.state.bodies}
            return True
        except Exception as exc:
            print(f"  [!] Import failed: {exc}")
            return False

    # ── Display ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"StateManager(bodies={len(self._index)}, time={self.state.time:.2e}s)"
