"""
Trajectory Recorder
===================
Lightweight data collector that hooks into the simulation engine and
stores per-step body positions for later visualization.

Designed to be zero-dependency — no NumPy, no plotting imports.
All data is stored as plain Python lists.

Usage
-----
    recorder = TrajectoryRecorder(record_every=10)

    # hook into engine
    engine.run_with_monitor(steps=8760, recorder=recorder)

    # later — hand to a visualizer
    from astrolab.viz.plotly_viz import render_html
    render_html(recorder, 'orbits.html')
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from astrolab.core.models import CelestialBody


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_CSS_TO_HEX: Dict[str, str] = {
    'white':   '#FFFFFF', 'yellow':  '#FFD700', 'orange':  '#FFA500',
    'blue':    '#4488FF', 'red':     '#FF4444', 'green':   '#44FF44',
    'grey':    '#AAAAAA', 'gray':    '#AAAAAA', 'purple':  '#AA00AA',
    'brown':   '#AA8833', 'cyan':    '#00CCCC', 'pink':    '#FF88AA',
    'magenta': '#FF00FF', 'gold':    '#FFD700', 'silver':  '#C0C0C0',
}

_TYPE_COLORS: Dict[str, str] = {
    'star':       '#FFD700',
    'planet':     '#4488FF',
    'moon':       '#AAAAAA',
    'asteroid':   '#AA8833',
    'black_hole': '#CC00CC',
    'comet':      '#88FFEE',
    'unknown':    '#FFFFFF',
}


def resolve_display_color(body: CelestialBody) -> str:
    """
    Return a CSS hex color string for *body*, resolving named colors and
    falling back to a type-based palette.
    """
    c = body.color.strip().lower()
    if c in _CSS_TO_HEX:
        return _CSS_TO_HEX[c]
    if c.startswith('#') or c.startswith('rgb'):
        return body.color
    return _TYPE_COLORS.get(body.body_type, '#FFFFFF')


# ---------------------------------------------------------------------------
# TrajectoryRecorder
# ---------------------------------------------------------------------------

class TrajectoryRecorder:
    """
    Records body states at regular simulation steps.

    Parameters
    ----------
    record_every : int
        Store one snapshot every N steps (default 1 = every step).
        Use higher values for large simulations to limit memory.
    max_snapshots : int
        Hard cap on stored snapshots (oldest are discarded).
        0 = unlimited.

    Data format (internal)
    ----------------------
    _snapshots  : list of::

        {
            'time':   float,             # simulation time [s]
            'bodies': [                  # list of body states
                {'name': str, 'x': float, 'y': float, 'z': float, 'speed': float},
                ...
            ]
        }

    _meta : dict of::

        body_name -> {
            'color': str,   # resolved hex CSS color
            'type':  str,   # body_type string
            'mass':  float,
            'radius': float,
        }
    """

    def __init__(
        self,
        record_every: int = 1,
        max_snapshots: int = 0,
    ) -> None:
        self.record_every  = max(1, record_every)
        self.max_snapshots = max_snapshots

        self._snapshots: List[Dict] = []
        self._meta:      Dict[str, Dict] = {}
        self._call_count: int = 0          # counts every record() call

    # ── Recording ────────────────────────────────────────────────────────────

    def record(self, bodies: List[CelestialBody], time: float) -> None:
        """
        Store the current state of *bodies* at simulation time *time*.

        Call this once per simulation step (or let the engine call it).
        Recording is skipped automatically when ``_call_count % record_every != 0``.
        """
        self._call_count += 1
        if self._call_count % self.record_every != 0:
            return

        # Update static metadata for each body (catches mass changes from merges)
        for body in bodies:
            self._meta[body.name] = {
                'color':  resolve_display_color(body),
                'type':   body.body_type,
                'mass':   body.mass,
                'radius': body.radius,
            }

        snap = {
            'time': time,
            'bodies': [
                {
                    'name':  b.name,
                    'x':     b.position.x,
                    'y':     b.position.y,
                    'z':     b.position.z,
                    'speed': b.speed,
                }
                for b in bodies
            ],
        }
        self._snapshots.append(snap)

        # Enforce cap
        if self.max_snapshots > 0 and len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(0)

    def reset(self) -> None:
        """Clear all recorded data."""
        self._snapshots.clear()
        self._meta.clear()
        self._call_count = 0

    # ── Accessors ─────────────────────────────────────────────────────────────

    def snapshot_count(self) -> int:
        return len(self._snapshots)

    def get_snapshot(self, idx: int) -> Dict:
        return self._snapshots[idx]

    def get_body_names(self) -> List[str]:
        """All body names that appear in at least one snapshot."""
        return list(self._meta.keys())

    def get_body_meta(self, name: str) -> Dict:
        """Return metadata dict for *name*, or {} if unknown."""
        return self._meta.get(name, {})

    def get_trajectory(
        self, name: str
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """
        Return five parallel lists for *name*:
            times, xs, ys, zs, speeds
        All in SI units (seconds, metres, m/s).
        """
        times, xs, ys, zs, speeds = [], [], [], [], []
        for snap in self._snapshots:
            for entry in snap['bodies']:
                if entry['name'] == name:
                    times.append(snap['time'])
                    xs.append(entry['x'])
                    ys.append(entry['y'])
                    zs.append(entry['z'])
                    speeds.append(entry['speed'])
                    break
        return times, xs, ys, zs, speeds

    def time_range(self) -> Tuple[float, float]:
        """Return (t_start, t_end) in seconds."""
        if not self._snapshots:
            return (0.0, 0.0)
        return (self._snapshots[0]['time'], self._snapshots[-1]['time'])

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save recorder data to a JSON file."""
        data = {
            'meta':      self._meta,
            'snapshots': self._snapshots,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'TrajectoryRecorder':
        """Load a previously saved recorder from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        r = cls()
        r._meta      = data['meta']
        r._snapshots = data['snapshots']
        return r

    # ── Display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        t0, tf = self.time_range()
        return (
            f"TrajectoryRecorder("
            f"bodies={len(self._meta)}, "
            f"snapshots={len(self._snapshots)}, "
            f"t=[{t0:.2e}..{tf:.2e}] s)"
        )
