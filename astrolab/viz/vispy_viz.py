"""
Vispy Real-Time Orbit Viewer
=============================
Two distinct viewing modes, both using Vispy's OpenGL scene graph.

Modes
-----
AstroLabLive   — Drives the simulation from inside Vispy's event loop.
                 The physics engine runs on a timer callback; the window
                 updates in real time as bodies move.

AstroLabReplay — Plays back a completed TrajectoryRecorder recording
                 like a movie: scrub through time, pause/rewind.

Controls (both modes)
---------------------
  SPACE         Pause / resume
  R             Rewind to start (Replay) / restart sim (Live)
  +/-           Speed up / slow down playback
  Scroll        Zoom
  Left-drag     Orbit camera
  Right-drag    Pan camera
  Q / Esc       Close window

Requires
--------
    pip install vispy pyqt5        (Windows / Linux)
    pip install vispy pyopengl     (macOS fallback)

Notes
-----
- All positions are converted from SI metres → Astronomical Units (AU)
  internally for display so the scene fits nicely at default zoom.
- Vispy needs a Qt backend.  If PyQt5 is not found it falls back to
  any available backend automatically.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from astrolab.viz.recorder import TrajectoryRecorder
    from astrolab.core.models import SimulationState
    from astrolab.engine.simulator import SimulationEngine

# Metres → AU
AU: float = 1.496e11

# Marker sizes (pixels) by body type
_BODY_SIZE: Dict[str, int] = {
    'star':       22,
    'black_hole': 16,
    'planet':     11,
    'moon':        7,
    'asteroid':    5,
    'comet':        5,
    'unknown':      8,
}

# Max trail length (number of recorded points shown per body)
_MAX_TRAIL: int = 600

# Background colour
_BG_COLOR: str = '#000010'


def _pick_backend() -> None:
    """Try to set a sensible Vispy backend for the current platform."""
    import sys
    backends = ['pyqt5', 'pyqt6', 'pyside2', 'pyside6', 'pyqt4', 'wx', 'glfw']
    for b in backends:
        try:
            import vispy
            vispy.use(b)
            return
        except Exception:
            continue


# ---------------------------------------------------------------------------
# AstroLabLive — real-time simulation in Vispy event loop
# ---------------------------------------------------------------------------

class AstroLabLive:
    """
    Run the simulation inside Vispy's event loop.

    The physics engine is ticked by a Vispy ``app.Timer``; the window
    updates every ``interval`` seconds.

    Parameters
    ----------
    state            : SimulationState  — shared mutable state
    engine           : SimulationEngine — will be stepped by the timer
    total_steps      : int              — stop after this many steps
    steps_per_frame  : int              — engine steps per timer tick
    recorder         : TrajectoryRecorder — optional; pass in to record trajectory
    fps              : int              — target render frames per second
    """

    def __init__(
        self,
        state:           'SimulationState',
        engine:          'SimulationEngine',
        total_steps:     int,
        steps_per_frame: int = 10,
        recorder:        Optional['TrajectoryRecorder'] = None,
        fps:             int = 30,
    ) -> None:
        _pick_backend()
        from vispy import app, scene
        import numpy as np
        self._np = np

        self.state           = state
        self.engine          = engine
        self.total_steps     = total_steps
        self.steps_per_frame = steps_per_frame
        self.recorder        = recorder
        self._steps_done     = 0
        self._paused         = False
        self._app            = app

        # ── Canvas + camera ─────────────────────────────────────────────────
        self.canvas = scene.SceneCanvas(
            title='AstroLab — Live Simulation',
            size=(1280, 800),
            bgcolor=_BG_COLOR,
            keys='interactive',
            show=False,
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45, distance=4.0, elevation=25, azimuth=30, up='z',
        )

        # Axis reference lines (X=red, Y=green, Z=blue, 1 AU each)
        _add_axes(self.view.scene, length=1.5)

        # ── Per-body visuals ─────────────────────────────────────────────────
        self._trail_buf: Dict[str, List] = {b.name: [] for b in state.bodies}
        self._lines:   Dict[str, object] = {}
        self._markers: Dict[str, object] = {}

        for body in state.bodies:
            c = _body_color(body)
            self._lines[body.name] = scene.visuals.Line(
                parent=self.view.scene, color=c, width=1.5, connect='strip', method='gl',
            )
            self._markers[body.name] = scene.visuals.Markers(
                parent=self.view.scene,
            )

        # Keyboard / timer
        self.canvas.events.key_press.connect(self._on_key)
        self._timer = app.Timer(interval=1.0 / fps, connect=self._on_tick, start=False)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_key(self, event) -> None:
        k = event.key.name if hasattr(event.key, 'name') else str(event.key)
        if k == 'Space':
            self._paused = not self._paused
        elif k in ('Q', 'Escape'):
            self._timer.stop()
            self._app.quit()
        elif k == 'R':
            # Restart — rebuild everything
            self._steps_done = 0
            for n in self._trail_buf:
                self._trail_buf[n].clear()

    def _on_tick(self, event) -> None:
        if self._paused:
            return
        if self._steps_done >= self.total_steps:
            self._timer.stop()
            self.canvas.title = (
                'AstroLab — Simulation Complete  '
                '(SPACE=pause  Q=quit)'
            )
            return

        steps = min(self.steps_per_frame, self.total_steps - self._steps_done)
        self.engine.run(steps=steps, collision_detection=True)
        self._steps_done += steps

        if self.recorder is not None:
            self.recorder.record(self.state.bodies, self.state.time)

        # Update trail buffers
        for body in self.state.bodies:
            pos = self._np.array(
                [body.position.x, body.position.y, body.position.z],
                dtype=self._np.float32
            ) / AU
            buf = self._trail_buf.get(body.name)
            if buf is None:
                self._trail_buf[body.name] = []
                buf = self._trail_buf[body.name]
            buf.append(pos)
            if len(buf) > _MAX_TRAIL:
                buf.pop(0)

        self._update_visuals()

        # Update title HUD
        days = self.state.time / 86_400
        pct  = self._steps_done * 100 // self.total_steps
        status = '⏸ PAUSED  ' if self._paused else ''
        self.canvas.title = (
            f'AstroLab Live — {status}{days:.1f} days  |  {pct}%  '
            f'(SPACE=pause  Q=quit)'
        )

    def _update_visuals(self) -> None:
        np = self._np
        from vispy import scene

        for name, buf in self._trail_buf.items():
            # Trail line
            if len(buf) >= 2:
                pts = np.array(buf, dtype=np.float32)
                self._lines[name].set_data(pts)

            # Current position marker
            if buf:
                body = next((b for b in self.state.bodies if b.name == name), None)
                if body is None:
                    continue
                pos   = np.array([buf[-1]], dtype=np.float32)
                sz    = _BODY_SIZE.get(body.body_type, 8)
                color = _body_color(body)
                self._markers[name].set_data(
                    pos, face_color=color, size=sz,
                    edge_color='white', edge_width=0.5,
                )

    # ── Entry ─────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Show the window and start the event loop (blocks until closed)."""
        self.canvas.show()
        self._timer.start()
        self._app.run()


# ---------------------------------------------------------------------------
# AstroLabReplay — playback of a recorded trajectory
# ---------------------------------------------------------------------------

class AstroLabReplay:
    """
    Replay a completed ``TrajectoryRecorder`` recording like a movie.

    Parameters
    ----------
    recorder    : TrajectoryRecorder
    fps         : int   Target playback frames per second.
    speed       : int   Snapshots to advance per frame (1 = real-time,
                        higher = time-lapse).
    """

    def __init__(
        self,
        recorder: 'TrajectoryRecorder',
        fps:      int = 30,
        speed:    int = 1,
    ) -> None:
        _pick_backend()
        from vispy import app, scene
        import numpy as np
        self._np = np

        self.recorder   = recorder
        self.fps        = fps
        self.speed      = speed
        self._snap_idx  = 0
        self._paused    = False
        self._app       = app

        body_names = recorder.get_body_names()

        # ── Canvas + camera ─────────────────────────────────────────────────
        self.canvas = scene.SceneCanvas(
            title='AstroLab — Orbit Replay',
            size=(1280, 800),
            bgcolor=_BG_COLOR,
            keys='interactive',
            show=False,
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45, distance=4.0, elevation=25, azimuth=30, up='z',
        )
        _add_axes(self.view.scene, length=1.5)

        # ── Per-body visuals ─────────────────────────────────────────────────
        self._trail_buf: Dict[str, List]  = {n: [] for n in body_names}
        self._lines:     Dict[str, object] = {}
        self._markers:   Dict[str, object] = {}

        for name in body_names:
            meta  = recorder.get_body_meta(name)
            color = meta.get('color', '#FFFFFF')
            self._lines[name]   = scene.visuals.Line(
                parent=self.view.scene, color=color, width=1.5,
                connect='strip', method='gl',
            )
            self._markers[name] = scene.visuals.Markers(parent=self.view.scene)

        # ── Timer + keyboard ─────────────────────────────────────────────────
        self.canvas.events.key_press.connect(self._on_key)
        self._timer = app.Timer(interval=1.0 / fps, connect=self._on_tick, start=False)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_key(self, event) -> None:
        k = event.key.name if hasattr(event.key, 'name') else str(event.key)
        if k == 'Space':
            self._paused = not self._paused
        elif k in ('Q', 'Escape'):
            self._timer.stop()
            self._app.quit()
        elif k == 'R':
            self._snap_idx = 0
            for n in self._trail_buf:
                self._trail_buf[n].clear()
        elif k in ('Equal', '+'):      # Speed up
            self.speed = min(self.speed * 2, 512)
        elif k in ('Minus', '-'):      # Slow down
            self.speed = max(1, self.speed // 2)

    def _on_tick(self, event) -> None:
        if self._paused:
            return

        total = self.recorder.snapshot_count()
        if self._snap_idx >= total:
            self._timer.stop()
            self.canvas.title = (
                'AstroLab — Replay Complete  '
                '(R=rewind  Q=quit)'
            )
            return

        end = min(self._snap_idx + self.speed, total)
        for i in range(self._snap_idx, end):
            snap = self.recorder.get_snapshot(i)
            for entry in snap['bodies']:
                name = entry['name']
                if name not in self._trail_buf:
                    self._trail_buf[name] = []
                pos = self._np.array(
                    [entry['x'], entry['y'], entry['z']], dtype=self._np.float32
                ) / AU
                self._trail_buf[name].append(pos)
                if len(self._trail_buf[name]) > _MAX_TRAIL:
                    self._trail_buf[name].pop(0)

        self._snap_idx = end
        self._update_visuals()

        # HUD
        snap  = self.recorder.get_snapshot(self._snap_idx - 1)
        days  = snap['time'] / 86_400
        pct   = self._snap_idx * 100 // total
        spd_s = f'×{self.speed}'
        tag   = '⏸ PAUSED  ' if self._paused else ''
        self.canvas.title = (
            f'AstroLab Replay — {tag}{days:.1f} days  |  {pct}%  |  {spd_s}  '
            f'(SPACE  R=rewind  +/-=speed  Q=quit)'
        )

    def _update_visuals(self) -> None:
        np = self._np
        for name, buf in self._trail_buf.items():
            if len(buf) >= 2:
                pts = np.array(buf, dtype=np.float32)
                self._lines[name].set_data(pts)

            if buf:
                meta  = self.recorder.get_body_meta(name)
                color = meta.get('color', '#FFFFFF')
                btype = meta.get('type', 'unknown')
                sz    = _BODY_SIZE.get(btype, 8)
                pos   = np.array([buf[-1]], dtype=np.float32)
                self._markers[name].set_data(
                    pos, face_color=color, size=sz,
                    edge_color='white', edge_width=0.5,
                )

    # ── Entry ─────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Show the window and start the playback loop (blocks until closed)."""
        self.canvas.show()
        self._timer.start()
        self._app.run()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _body_color(body) -> str:
    """Return a hex color string for *body*."""
    from astrolab.viz.recorder import resolve_display_color
    return resolve_display_color(body)


def _add_axes(parent, length: float = 2.0) -> None:
    """Draw XYZ axis lines in the scene (X=red, Y=green, Z=blue)."""
    from vispy import scene
    import numpy as np

    axes = [
        ([(0, 0, 0), (length, 0, 0)], '#FF4444'),
        ([(0, 0, 0), (0, length, 0)], '#44FF44'),
        ([(0, 0, 0), (0, 0, length)], '#4444FF'),
    ]
    for pts, clr in axes:
        scene.visuals.Line(
            pos=np.array(pts, dtype=np.float32),
            color=clr, width=1,
            parent=parent,
        )
