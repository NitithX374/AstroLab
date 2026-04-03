"""
Plotly Interactive Orbit Visualizer
====================================
Renders a TrajectoryRecorder's data as a self-contained interactive 3D
HTML file using Plotly.

The output is a **single portable HTML file** — share it with anyone,
open it in any browser, no server or Python install required.

Features
--------
- 3D interactive orbit view (rotate / zoom / pan)
- Hover any point for: body name, elapsed days, position (AU), speed (km/s)
- Click legend entry to show/hide individual bodies
- Final-position sphere + body name label
- Dark space theme

Requires
--------
    pip install plotly

Usage (standalone)
------------------
    from astrolab.viz.recorder import TrajectoryRecorder
    from astrolab.viz.plotly_viz import render_html

    recorder = TrajectoryRecorder.load('trajectory.json')
    render_html(recorder, 'orbits.html')
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astrolab.viz.recorder import TrajectoryRecorder

# Metres per Astronomical Unit
AU: float = 1.496e11

# Body-type size map (marker pixel size)
_MARKER_SIZE: dict[str, int] = {
    'star':       14,
    'black_hole': 12,
    'planet':     8,
    'moon':       6,
    'asteroid':   5,
    'comet':      5,
    'unknown':    7,
}


def render_html(
    recorder: 'TrajectoryRecorder',
    output_path: str,
    title: str = 'AstroLab — Orbit Viewer',
    unit: str = 'AU',
) -> None:
    """
    Generate a self-contained 3D interactive HTML orbit plot.

    Parameters
    ----------
    recorder    : TrajectoryRecorder
    output_path : str   Path for the output HTML file.
    title       : str   Plot title (shown at top of page).
    unit        : str   'AU' or 'm' — axis units.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for HTML rendering.\n"
            "Install with:  pip install plotly"
        )

    scale  = 1.0 / AU if unit == 'AU' else 1.0
    ax_lbl = unit

    fig = go.Figure()

    body_names = recorder.get_body_names()
    if not body_names:
        print("  [!] Recorder has no data.")
        return

    t_start, t_end = recorder.time_range()
    duration_days  = (t_end - t_start) / 86_400

    for name in body_names:
        times, xs, ys, zs, speeds = recorder.get_trajectory(name)
        if not xs:
            continue

        meta  = recorder.get_body_meta(name)
        color = meta.get('color', '#FFFFFF')
        btype = meta.get('type', 'unknown')
        msize = _MARKER_SIZE.get(btype, 7)

        xs_s     = [x * scale for x in xs]
        ys_s     = [y * scale for y in ys]
        zs_s     = [z * scale for z in zs]
        days     = [t / 86_400 for t in times]
        spd_kms  = [s / 1e3    for s in speeds]

        # ── Orbit trail ──────────────────────────────────────────────────────
        fig.add_trace(go.Scatter3d(
            x=xs_s, y=ys_s, z=zs_s,
            mode='lines',
            name=name,
            line=dict(color=color, width=2),
            customdata=list(zip(
                days, spd_kms,
                xs_s, ys_s, zs_s
            )),
            hovertemplate=(
                f'<b>{name}</b><br>'
                f'Day: %{{customdata[0]:.1f}}<br>'
                f'Speed: %{{customdata[1]:.2f}} km/s<br>'
                f'X: %{{customdata[2]:.4f}} {ax_lbl}<br>'
                f'Y: %{{customdata[3]:.4f}} {ax_lbl}<br>'
                f'Z: %{{customdata[4]:.4f}} {ax_lbl}'
                '<extra></extra>'
            ),
        ))

        # ── Final-position marker ─────────────────────────────────────────────
        fig.add_trace(go.Scatter3d(
            x=[xs_s[-1]], y=[ys_s[-1]], z=[zs_s[-1]],
            mode='markers+text',
            name=f'{name} (final)',
            marker=dict(
                size=msize,
                color=color,
                symbol='circle',
                line=dict(color='white', width=1),
                opacity=1.0,
            ),
            text=[name],
            textposition='top center',
            textfont=dict(color='white', size=11, family='monospace'),
            showlegend=False,
            hovertemplate=(
                f'<b>{name}</b> — final position<br>'
                f'Day {days[-1]:.1f}<br>'
                f'Speed: {spd_kms[-1]:.2f} km/s'
                '<extra></extra>'
            ),
        ))

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        template='plotly_dark',
        title=dict(
            text=f'{title}<br>'
                 f'<sup>{len(body_names)} bod{"ies" if len(body_names)>1 else "y"}  '
                 f'·  {duration_days:.1f} days simulated</sup>',
            font=dict(size=18, color='white'),
            x=0.5, xanchor='center',
        ),
        scene=dict(
            xaxis=dict(
                title=f'X ({ax_lbl})',
                gridcolor='#1a1a2e', zerolinecolor='#444', backgroundcolor='#000010',
                showbackground=True,
            ),
            yaxis=dict(
                title=f'Y ({ax_lbl})',
                gridcolor='#1a1a2e', zerolinecolor='#444', backgroundcolor='#000010',
                showbackground=True,
            ),
            zaxis=dict(
                title=f'Z ({ax_lbl})',
                gridcolor='#1a1a2e', zerolinecolor='#444', backgroundcolor='#000010',
                showbackground=True,
            ),
            bgcolor='#000010',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor='#000010',
        font=dict(color='white', family='monospace'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.6)',
            bordercolor='#444',
            borderwidth=1,
            font=dict(size=12),
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        hoverlabel=dict(
            bgcolor='#111122',
            bordercolor='#555',
            font=dict(family='monospace', size=12),
        ),
    )

    # Write to file
    fig.write_html(
        output_path,
        include_plotlyjs='cdn',    # small file size; requires internet for CDN
        full_html=True,
        config={
            'displaylogo': False,
            'modeBarButtonsToRemove': ['toImage'],
            'scrollZoom': True,
        },
    )
    print(f"  [+] Plotly orbit chart saved → '{output_path}'")
    print(f"      Open in any browser — rotate, zoom, hover for details.")
