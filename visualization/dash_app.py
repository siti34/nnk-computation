"""Plotly Dash 3D visualization for HTO mechanical axis computation.

Interactive web application for visualising pipeline results including:
- 3D scatter plot of marker clusters, anatomical landmarks, joint centres
- Mechanical axis, joint lines, and local coordinate system axes
- Frame-by-frame animation of dynamic trials
- Real-time angle readouts and time-series plots

Launch standalone:
    python -m visualization.dash_app
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``src`` is importable when
# running this module directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.angles import HTOAngles
from src.data_loader import (
    FEMORAL_MARKERS,
    TIBIAL_MARKERS,
    TrialData,
    load_all_trials,
)
from src.pipeline import (
    DynamicTrialResult,
    PipelineResults,
    run_pipeline,
)
from src.rigid_body import FramePose, RigidBodyReference, transform_point_to_global

# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------
_FEMORAL_COLOUR = "rgb(30, 100, 220)"    # blue
_TIBIAL_COLOUR = "rgb(30, 180, 80)"      # green
_JC_COLOUR = "rgb(220, 40, 40)"          # red
_LANDMARK_COLOUR = "rgb(230, 200, 40)"   # yellow
_MECH_AXIS_COLOUR = "rgb(220, 40, 40)"   # red
_FEM_JL_COLOUR = "rgb(255, 160, 40)"     # orange
_TIB_JL_COLOUR = "rgb(40, 210, 220)"     # cyan
_DIST_TIB_JL_COLOUR = "rgb(180, 80, 220)"  # purple
_LCS_X_COLOUR = "rgb(220, 30, 30)"
_LCS_Y_COLOUR = "rgb(30, 180, 30)"
_LCS_Z_COLOUR = "rgb(30, 30, 220)"

_LCS_LENGTH = 30.0  # mm

# Layout style constants
_SIDEBAR_STYLE: Dict[str, Any] = {
    "width": "250px",
    "position": "fixed",
    "top": "0",
    "left": "0",
    "bottom": "0",
    "padding": "20px 15px",
    "backgroundColor": "#1e1e2f",
    "color": "#e0e0e0",
    "overflowY": "auto",
    "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    "fontSize": "13px",
    "boxShadow": "2px 0 8px rgba(0,0,0,0.3)",
    "zIndex": "1000",
}

_MAIN_STYLE: Dict[str, Any] = {
    "marginLeft": "270px",
    "padding": "10px 20px",
    "backgroundColor": "#121212",
    "minHeight": "100vh",
    "fontFamily": "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    "color": "#e0e0e0",
}

_LABEL_STYLE: Dict[str, Any] = {
    "fontWeight": "600",
    "marginTop": "14px",
    "marginBottom": "4px",
    "display": "block",
    "fontSize": "12px",
    "color": "#aaa",
    "textTransform": "uppercase",
    "letterSpacing": "0.5px",
}

_ANGLE_BOX_STYLE: Dict[str, Any] = {
    "backgroundColor": "#1e1e2f",
    "borderRadius": "6px",
    "padding": "12px 16px",
    "marginTop": "8px",
    "fontSize": "13px",
    "lineHeight": "1.6",
    "border": "1px solid #333",
}


# ===================================================================
# Helper: precompute all frame data for every trial (+ static)
# ===================================================================

def _static_frame_data(
    results: PipelineResults,
    trials: Dict[str, TrialData],
) -> Dict[str, Any]:
    """Build the data dict for the static trial (single frame)."""

    static_trial = trials.get("static")

    # --- Cluster marker positions (average across static frames) ----
    fem_markers: Dict[str, List[float]] = {}
    tib_markers: Dict[str, List[float]] = {}

    if static_trial is not None:
        for m in FEMORAL_MARKERS:
            traj = static_trial.markers.get(m)
            if traj is not None:
                valid = ~np.any(np.isnan(traj), axis=1)
                if np.any(valid):
                    fem_markers[m] = np.mean(traj[valid], axis=0).tolist()

        for m in TIBIAL_MARKERS:
            traj = static_trial.markers.get(m)
            if traj is not None:
                valid = ~np.any(np.isnan(traj), axis=1)
                if np.any(valid):
                    tib_markers[m] = np.mean(traj[valid], axis=0).tolist()

    # --- Landmarks (global, from pipeline) --------------------------
    landmarks: Dict[str, List[float]] = {
        k: v.tolist() for k, v in results.landmarks_global.items()
    }

    # --- Joint centres -----------------------------------------------
    hjc = results.hjc.position.tolist()
    kjc = results.kjc.tolist()
    ajc = results.ajc.tolist()

    # --- LCS data (centroid + rotation columns for axes) -------------
    fem_lcs = {
        "origin": results.femoral_ref.centroid.tolist(),
        "x": results.femoral_ref.rotation[:, 0].tolist(),
        "y": results.femoral_ref.rotation[:, 1].tolist(),
        "z": results.femoral_ref.rotation[:, 2].tolist(),
    }
    tib_lcs = {
        "origin": results.tibial_ref.centroid.tolist(),
        "x": results.tibial_ref.rotation[:, 0].tolist(),
        "y": results.tibial_ref.rotation[:, 1].tolist(),
        "z": results.tibial_ref.rotation[:, 2].tolist(),
    }

    # --- Angles -------------------------------------------------------
    sa = results.static_angles
    angles = {
        "hka": round(sa.hka, 2),
        "mldfa": round(sa.mldfa, 2),
        "mpta": round(sa.mpta, 2),
        "jlca": round(sa.jlca, 2),
        "mldta": round(sa.mldta, 2) if sa.mldta is not None else None,
        "knee_offset_mm": round(sa.knee_offset_mm, 2),
    }

    return {
        "fem_markers": fem_markers,
        "tib_markers": tib_markers,
        "landmarks": landmarks,
        "hjc": hjc,
        "kjc": kjc,
        "ajc": ajc,
        "fem_lcs": fem_lcs,
        "tib_lcs": tib_lcs,
        "angles": angles,
    }


def _dynamic_frames_data(
    dyn: DynamicTrialResult,
    results: PipelineResults,
    trial: TrialData,
) -> List[Dict[str, Any]]:
    """Pre-compute per-frame visualisation data for a dynamic trial.

    Returns a list of dicts (one per frame), each with the same keys as
    ``_static_frame_data`` output.
    """

    frames: List[Dict[str, Any]] = []

    for fi in range(dyn.n_frames):
        fp = dyn.femoral_poses[fi]
        tp = dyn.tibial_poses[fi]

        # ----- cluster markers (raw from trial) ---------------------
        fem_markers: Dict[str, List[float]] = {}
        for m in FEMORAL_MARKERS:
            traj = trial.markers.get(m)
            if traj is not None and not np.any(np.isnan(traj[fi])):
                fem_markers[m] = traj[fi].tolist()

        tib_markers: Dict[str, List[float]] = {}
        for m in TIBIAL_MARKERS:
            traj = trial.markers.get(m)
            if traj is not None and not np.any(np.isnan(traj[fi])):
                tib_markers[m] = traj[fi].tolist()

        # ----- landmarks (global, per-frame) -------------------------
        landmarks: Dict[str, List[float]] = {}
        for lm_name, arr in dyn.landmarks_per_frame.items():
            pos = arr[fi]
            if not np.any(np.isnan(pos)):
                landmarks[lm_name] = pos.tolist()

        # ----- joint centres -----------------------------------------
        hjc_f = dyn.hjc_per_frame[fi]
        kjc_f = dyn.kjc_per_frame[fi]
        ajc_f = dyn.ajc_per_frame[fi]

        hjc = hjc_f.tolist() if not np.any(np.isnan(hjc_f)) else None
        kjc = kjc_f.tolist() if not np.any(np.isnan(kjc_f)) else None
        ajc = ajc_f.tolist() if not np.any(np.isnan(ajc_f)) else None

        # ----- LCS axes ----------------------------------------------
        fem_lcs = None
        if fp is not None:
            fem_lcs = {
                "origin": fp.origin.tolist(),
                "x": fp.rotation[:, 0].tolist(),
                "y": fp.rotation[:, 1].tolist(),
                "z": fp.rotation[:, 2].tolist(),
            }

        tib_lcs = None
        if tp is not None:
            tib_lcs = {
                "origin": tp.origin.tolist(),
                "x": tp.rotation[:, 0].tolist(),
                "y": tp.rotation[:, 1].tolist(),
                "z": tp.rotation[:, 2].tolist(),
            }

        # ----- angles ------------------------------------------------
        ang = dyn.angles_per_frame[fi]
        if ang is not None:
            angles = {
                "hka": round(ang.hka, 2),
                "mldfa": round(ang.mldfa, 2),
                "mpta": round(ang.mpta, 2),
                "jlca": round(ang.jlca, 2),
                "mldta": round(ang.mldta, 2) if ang.mldta is not None else None,
                "knee_offset_mm": round(ang.knee_offset_mm, 2),
            }
        else:
            angles = {
                "hka": None, "mldfa": None, "mpta": None,
                "jlca": None, "mldta": None, "knee_offset_mm": None,
            }

        frames.append({
            "fem_markers": fem_markers,
            "tib_markers": tib_markers,
            "landmarks": landmarks,
            "hjc": hjc,
            "kjc": kjc,
            "ajc": ajc,
            "fem_lcs": fem_lcs,
            "tib_lcs": tib_lcs,
            "angles": angles,
        })

    return frames


def _precompute_all(
    results: PipelineResults,
    trials: Dict[str, TrialData],
) -> Dict[str, Any]:
    """Pre-compute visualisation data for all trials.

    Returns
    -------
    dict
        ``{"static": <frame_data>, "<trial_name>": [<frame_data>, ...], ...}``
    """
    store: Dict[str, Any] = {}

    # Static
    store["static"] = [_static_frame_data(results, trials)]

    # Dynamic trials
    for trial_name, dyn in results.dynamic_results.items():
        trial = trials.get(trial_name)
        if trial is None:
            continue
        store[trial_name] = _dynamic_frames_data(dyn, results, trial)

    # Also precompute HKA time-series per dynamic trial
    hka_series: Dict[str, List[Optional[float]]] = {}
    for trial_name, dyn in results.dynamic_results.items():
        hka_series[trial_name] = [
            round(a.hka, 2) if a is not None else None
            for a in dyn.angles_per_frame
        ]
    store["_hka_series"] = hka_series

    return store


# ===================================================================
# Build Plotly 3D figure from a single frame data dict
# ===================================================================

def _build_3d_figure(
    fdata: Dict[str, Any],
    show_fem_markers: bool = True,
    show_tib_markers: bool = True,
    show_lcs: bool = True,
    show_mech_axis: bool = True,
    show_joint_lines: bool = True,
    show_joint_centres: bool = True,
    show_landmarks: bool = True,
) -> go.Figure:
    """Create a Plotly 3D figure from pre-computed frame data."""

    traces: List[go.Scatter3d] = []

    # ---- femoral markers -------------------------------------------
    if show_fem_markers and fdata["fem_markers"]:
        names = list(fdata["fem_markers"].keys())
        pts = np.array([fdata["fem_markers"][n] for n in names])
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers+text",
            marker=dict(size=5, color=_FEMORAL_COLOUR, symbol="circle"),
            text=names,
            textposition="top center",
            textfont=dict(size=9, color=_FEMORAL_COLOUR),
            name="Femoral markers",
            legendgroup="fem",
            hovertemplate="%{text}<br>x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<extra></extra>",
        ))

    # ---- tibial markers --------------------------------------------
    if show_tib_markers and fdata["tib_markers"]:
        names = list(fdata["tib_markers"].keys())
        pts = np.array([fdata["tib_markers"][n] for n in names])
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers+text",
            marker=dict(size=5, color=_TIBIAL_COLOUR, symbol="circle"),
            text=names,
            textposition="top center",
            textfont=dict(size=9, color=_TIBIAL_COLOUR),
            name="Tibial markers",
            legendgroup="tib",
            hovertemplate="%{text}<br>x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<extra></extra>",
        ))

    # ---- anatomical landmarks (diamonds) ---------------------------
    if show_landmarks and fdata["landmarks"]:
        lm_names = list(fdata["landmarks"].keys())
        lm_pts = np.array([fdata["landmarks"][n] for n in lm_names])
        traces.append(go.Scatter3d(
            x=lm_pts[:, 0], y=lm_pts[:, 1], z=lm_pts[:, 2],
            mode="markers+text",
            marker=dict(size=6, color=_LANDMARK_COLOUR, symbol="diamond"),
            text=lm_names,
            textposition="bottom center",
            textfont=dict(size=9, color=_LANDMARK_COLOUR),
            name="Landmarks",
            legendgroup="lm",
            hovertemplate="%{text}<br>x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<extra></extra>",
        ))

    # ---- joint centres (large red spheres) --------------------------
    if show_joint_centres:
        jc_names: List[str] = []
        jc_pts: List[List[float]] = []
        for label, pt in [("HJC", fdata["hjc"]),
                          ("KJC", fdata["kjc"]),
                          ("AJC", fdata["ajc"])]:
            if pt is not None:
                jc_names.append(label)
                jc_pts.append(pt)
        if jc_pts:
            arr = np.array(jc_pts)
            traces.append(go.Scatter3d(
                x=arr[:, 0], y=arr[:, 1], z=arr[:, 2],
                mode="markers+text",
                marker=dict(size=8, color=_JC_COLOUR, symbol="circle",
                            line=dict(width=1, color="white")),
                text=jc_names,
                textposition="middle right",
                textfont=dict(size=10, color="white"),
                name="Joint centres",
                legendgroup="jc",
                hovertemplate="%{text}<br>x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<extra></extra>",
            ))

    # ---- mechanical axis (thick red line HJC->KJC->AJC) ------------
    if show_mech_axis:
        mech_pts = [fdata["hjc"], fdata["kjc"], fdata["ajc"]]
        mech_pts = [p for p in mech_pts if p is not None]
        if len(mech_pts) >= 2:
            arr = np.array(mech_pts)
            traces.append(go.Scatter3d(
                x=arr[:, 0], y=arr[:, 1], z=arr[:, 2],
                mode="lines",
                line=dict(width=6, color=_MECH_AXIS_COLOUR),
                name="Mechanical axis",
                legendgroup="mech",
                hoverinfo="skip",
            ))

    # ---- joint lines ------------------------------------------------
    if show_joint_lines and fdata["landmarks"]:
        lm = fdata["landmarks"]
        # Distal femoral joint line: LFEC -- MFEC
        if "LFEC" in lm and "MFEC" in lm:
            pts = np.array([lm["LFEC"], lm["MFEC"]])
            traces.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="lines",
                line=dict(width=4, color=_FEM_JL_COLOUR),
                name="Distal femoral JL",
                legendgroup="jl",
                hoverinfo="skip",
            ))
        # Proximal tibial joint line: LTP -- MTP
        if "LTP" in lm and "MTP" in lm:
            pts = np.array([lm["LTP"], lm["MTP"]])
            traces.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="lines",
                line=dict(width=4, color=_TIB_JL_COLOUR),
                name="Proximal tibial JL",
                legendgroup="jl",
                hoverinfo="skip",
            ))
        # Distal tibial joint line: LM -- MM
        if "LM" in lm and "MM" in lm:
            pts = np.array([lm["LM"], lm["MM"]])
            traces.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="lines",
                line=dict(width=4, color=_DIST_TIB_JL_COLOUR),
                name="Distal tibial JL",
                legendgroup="jl",
                hoverinfo="skip",
            ))

    # ---- LCS axes (short coloured lines from centroid) --------------
    if show_lcs:
        for lcs_data, grp_name in [
            (fdata.get("fem_lcs"), "Femoral LCS"),
            (fdata.get("tib_lcs"), "Tibial LCS"),
        ]:
            if lcs_data is None:
                continue
            origin = np.array(lcs_data["origin"])
            for axis_key, colour, axis_label in [
                ("x", _LCS_X_COLOUR, "X"),
                ("y", _LCS_Y_COLOUR, "Y"),
                ("z", _LCS_Z_COLOUR, "Z"),
            ]:
                direction = np.array(lcs_data[axis_key])
                tip = origin + _LCS_LENGTH * direction
                traces.append(go.Scatter3d(
                    x=[origin[0], tip[0]],
                    y=[origin[1], tip[1]],
                    z=[origin[2], tip[2]],
                    mode="lines",
                    line=dict(width=3, color=colour),
                    name=f"{grp_name} {axis_label}",
                    legendgroup=f"lcs_{grp_name}",
                    showlegend=(axis_key == "x"),  # one legend entry per cluster
                    hoverinfo="skip",
                ))

    # ---- layout -----------------------------------------------------
    # Gather all visible points to set axis range
    all_pts: List[List[float]] = []
    for mk_dict in [fdata["fem_markers"], fdata["tib_markers"], fdata["landmarks"]]:
        if mk_dict:
            all_pts.extend(mk_dict.values())
    for p in [fdata["hjc"], fdata["kjc"], fdata["ajc"]]:
        if p is not None:
            all_pts.append(p)

    if all_pts:
        arr = np.array(all_pts)
        centre = arr.mean(axis=0)
        extent = max(np.ptp(arr, axis=0)) / 2 + 50  # padding
        x_range = [centre[0] - extent, centre[0] + extent]
        y_range = [centre[1] - extent, centre[1] + extent]
        z_range = [centre[2] - extent, centre[2] + extent]
    else:
        x_range = y_range = z_range = [-200, 200]

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X (mm)", range=x_range, backgroundcolor="#181828",
                       gridcolor="#333", showbackground=True),
            yaxis=dict(title="Y (mm)", range=y_range, backgroundcolor="#181828",
                       gridcolor="#333", showbackground=True),
            zaxis=dict(title="Z (mm)", range=z_range, backgroundcolor="#181828",
                       gridcolor="#333", showbackground=True),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        font=dict(color="#e0e0e0"),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            bgcolor="rgba(30,30,47,0.85)",
            font=dict(size=10, color="#ccc"),
        ),
        height=600,
    )
    return fig


def _build_hka_timeseries_fig(
    hka_values: List[Optional[float]],
    current_frame: int,
) -> go.Figure:
    """Build a small HKA time-series line chart with a vertical marker
    at the current frame."""
    n = len(hka_values)
    xs = list(range(n))
    ys = [v if v is not None else None for v in hka_values]

    fig = go.Figure()

    # Main HKA line
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color=_MECH_AXIS_COLOUR, width=2),
        name="HKA",
        hovertemplate="Frame %{x}<br>HKA = %{y:.1f} deg<extra></extra>",
    ))

    # Reference line at 180 deg (neutral)
    fig.add_hline(y=180, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                  annotation_text="180 (neutral)", annotation_font_color="#888")

    # Current frame marker
    if 0 <= current_frame < n and hka_values[current_frame] is not None:
        fig.add_trace(go.Scatter(
            x=[current_frame], y=[hka_values[current_frame]],
            mode="markers",
            marker=dict(size=10, color="white", symbol="circle",
                        line=dict(width=2, color=_MECH_AXIS_COLOUR)),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        height=180,
        margin=dict(l=45, r=10, t=25, b=30),
        paper_bgcolor="#1e1e2f",
        plot_bgcolor="#1e1e2f",
        font=dict(color="#ccc", size=10),
        xaxis=dict(title="Frame", gridcolor="#333"),
        yaxis=dict(title="HKA (deg)", gridcolor="#333"),
        showlegend=False,
    )
    return fig


# ===================================================================
# Build angle readout HTML
# ===================================================================

def _angle_readout_children(angles: Dict[str, Any]) -> List:
    """Return Dash html components for the angle readout panel."""

    def _fmt(val, unit="deg"):
        if val is None:
            return "N/A"
        return f"{val:.1f}" if isinstance(val, float) else f"{val}"

    hka_val = angles.get("hka")
    if hka_val is not None:
        if abs(hka_val - 180) < 3:
            alignment = "neutral"
            align_colour = "#4caf50"
        elif hka_val < 180:
            alignment = "varus"
            align_colour = "#ff9800"
        else:
            alignment = "valgus"
            align_colour = "#2196f3"
    else:
        alignment = ""
        align_colour = "#888"

    rows = [
        html.Div([
            html.Span("HKA: ", style={"fontWeight": "700", "color": "#fff"}),
            html.Span(f"{_fmt(hka_val)}  ", style={"color": "#fff"}),
            html.Span(alignment, style={"color": align_colour, "fontWeight": "600"}),
        ], style={"marginBottom": "4px"}),
        html.Div(f"mLDFA: {_fmt(angles.get('mldfa'))}   (norm 85-90)"),
        html.Div(f"MPTA:  {_fmt(angles.get('mpta'))}   (norm 85-90)"),
        html.Div(f"JLCA:  {_fmt(angles.get('jlca'))}   (norm 0-2)"),
        html.Div(f"mLDTA: {_fmt(angles.get('mldta'))}   (norm ~89)"),
        html.Div(
            f"Knee offset: {_fmt(angles.get('knee_offset_mm'))} mm",
            style={"marginTop": "4px", "color": "#aaa"},
        ),
    ]
    return rows


# ===================================================================
# create_app  -- main entry point
# ===================================================================

def create_app(
    results: PipelineResults,
    trials: Dict[str, TrialData],
) -> Dash:
    """Create and return a configured Dash application.

    Parameters
    ----------
    results : PipelineResults
        Output of ``run_pipeline``.
    trials : Dict[str, TrialData]
        Loaded trial data (the same dict used by the pipeline).

    Returns
    -------
    Dash
        A fully configured Dash application ready for ``.run_server()``.
    """

    # ----- precompute ------------------------------------------------
    precomputed = _precompute_all(results, trials)

    # Available trial names for the dropdown
    trial_options = [{"label": "Static", "value": "static"}]
    for tn in sorted(results.dynamic_results.keys()):
        trial_options.append({"label": tn.replace("_", " ").title(), "value": tn})

    # Max frames across all trials (for slider)
    max_frames_map: Dict[str, int] = {"static": 1}
    for tn, dyn in results.dynamic_results.items():
        max_frames_map[tn] = dyn.n_frames

    # ----- app -------------------------------------------------------
    app = Dash(
        __name__,
        title="HTO Mechanical Axis Viewer",
        update_title=None,
    )

    # ----- layout ----------------------------------------------------
    app.layout = html.Div([
        # Precomputed data store
        dcc.Store(id="precomputed-store", data=precomputed),
        dcc.Store(id="max-frames-map", data=max_frames_map),
        dcc.Store(id="is-playing", data=False),

        # ---- sidebar -------------------------------------------------
        html.Div([
            html.H3("HTO Viewer", style={"marginTop": "0", "color": "#fff",
                                          "borderBottom": "1px solid #444",
                                          "paddingBottom": "10px"}),

            # Trial selector
            html.Label("Trial", style=_LABEL_STYLE),
            dcc.Dropdown(
                id="trial-dropdown",
                options=trial_options,
                value="static",
                clearable=False,
                style={"color": "#222", "fontSize": "12px"},
            ),

            # Frame slider
            html.Label("Frame", style=_LABEL_STYLE),
            dcc.Slider(
                id="frame-slider",
                min=0, max=0, step=1, value=0,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
            ),

            # Play / Pause + speed
            html.Div([
                html.Button("Play", id="play-btn",
                            style={"marginRight": "8px", "padding": "4px 14px",
                                   "cursor": "pointer", "borderRadius": "4px",
                                   "border": "1px solid #555", "backgroundColor": "#2a2a40",
                                   "color": "#ddd", "fontSize": "12px"}),
                html.Label("Speed (ms):", style={"fontSize": "11px", "marginRight": "4px"}),
                dcc.Input(id="speed-input", type="number", value=100, min=20, max=2000,
                          step=10,
                          style={"width": "55px", "fontSize": "11px", "padding": "2px 4px",
                                 "backgroundColor": "#2a2a40", "color": "#ddd",
                                 "border": "1px solid #555", "borderRadius": "4px"}),
            ], style={"marginTop": "10px", "display": "flex", "alignItems": "center"}),

            # Interval component for animation
            dcc.Interval(id="animation-interval", interval=100, n_intervals=0, disabled=True),

            # Toggle checkboxes
            html.Label("Show / Hide", style=_LABEL_STYLE),
            dcc.Checklist(
                id="visibility-checks",
                options=[
                    {"label": " Femoral markers", "value": "fem_markers"},
                    {"label": " Tibial markers", "value": "tib_markers"},
                    {"label": " Landmarks", "value": "landmarks"},
                    {"label": " LCS axes", "value": "lcs"},
                    {"label": " Mechanical axis", "value": "mech_axis"},
                    {"label": " Joint lines", "value": "joint_lines"},
                    {"label": " Joint centres", "value": "joint_centres"},
                ],
                value=["fem_markers", "tib_markers", "landmarks", "lcs",
                       "mech_axis", "joint_lines", "joint_centres"],
                style={"lineHeight": "1.9", "fontSize": "12px"},
                inputStyle={"marginRight": "6px"},
            ),
        ], style=_SIDEBAR_STYLE),

        # ---- main area -----------------------------------------------
        html.Div([
            # 3D viewport
            dcc.Graph(id="scatter3d", style={"height": "600px"}),

            # Bottom panel: angle readout + HKA time series
            html.Div([
                # Angle readout (left)
                html.Div([
                    html.H4("Angle Readout", style={"margin": "0 0 6px 0", "fontSize": "14px",
                                                      "color": "#aaa"}),
                    html.Div(id="angle-readout", style=_ANGLE_BOX_STYLE),
                ], style={"flex": "1", "minWidth": "220px", "marginRight": "16px"}),

                # HKA time-series (right)
                html.Div([
                    html.H4("HKA Over Frames", style={"margin": "0 0 6px 0", "fontSize": "14px",
                                                        "color": "#aaa"}),
                    dcc.Graph(id="hka-timeseries",
                              style={"height": "180px"},
                              config={"displayModeBar": False}),
                ], style={"flex": "2", "minWidth": "300px"}),
            ], style={"display": "flex", "marginTop": "12px", "alignItems": "flex-start"}),
        ], style=_MAIN_STYLE),
    ])

    # ==================================================================
    # Callbacks
    # ==================================================================

    # --- Update slider max when trial changes -------------------------
    @app.callback(
        [Output("frame-slider", "max"),
         Output("frame-slider", "value")],
        Input("trial-dropdown", "value"),
        State("max-frames-map", "data"),
    )
    def update_slider_range(trial_name, mf_map):
        max_f = mf_map.get(trial_name, 1) - 1
        return max(max_f, 0), 0

    # --- Play / Pause toggle ------------------------------------------
    @app.callback(
        [Output("is-playing", "data"),
         Output("play-btn", "children"),
         Output("animation-interval", "disabled"),
         Output("animation-interval", "interval")],
        Input("play-btn", "n_clicks"),
        [State("is-playing", "data"),
         State("speed-input", "value")],
        prevent_initial_call=True,
    )
    def toggle_play(n_clicks, playing, speed):
        new_playing = not playing
        label = "Pause" if new_playing else "Play"
        interval_ms = max(int(speed) if speed else 100, 20)
        return new_playing, label, not new_playing, interval_ms

    # --- Advance frame on interval tick -------------------------------
    @app.callback(
        Output("frame-slider", "value", allow_duplicate=True),
        Input("animation-interval", "n_intervals"),
        [State("frame-slider", "value"),
         State("frame-slider", "max"),
         State("is-playing", "data")],
        prevent_initial_call=True,
    )
    def advance_frame(n_intervals, current_frame, max_frame, playing):
        if not playing:
            return current_frame
        next_frame = current_frame + 1
        if next_frame > max_frame:
            next_frame = 0
        return next_frame

    # --- Main render callback: 3D figure + angles + HKA chart ---------
    @app.callback(
        [Output("scatter3d", "figure"),
         Output("angle-readout", "children"),
         Output("hka-timeseries", "figure")],
        [Input("frame-slider", "value"),
         Input("visibility-checks", "value")],
        [State("trial-dropdown", "value"),
         State("precomputed-store", "data")],
    )
    def render(frame_idx, vis_checks, trial_name, store):
        # Get frame data
        trial_frames = store.get(trial_name, store.get("static", [{}]))
        frame_idx = min(frame_idx, len(trial_frames) - 1)
        frame_idx = max(frame_idx, 0)
        fdata = trial_frames[frame_idx]

        # Build 3D figure
        fig3d = _build_3d_figure(
            fdata,
            show_fem_markers="fem_markers" in vis_checks,
            show_tib_markers="tib_markers" in vis_checks,
            show_lcs="lcs" in vis_checks,
            show_mech_axis="mech_axis" in vis_checks,
            show_joint_lines="joint_lines" in vis_checks,
            show_joint_centres="joint_centres" in vis_checks,
            show_landmarks="landmarks" in vis_checks,
        )

        # Angle readout
        angles = fdata.get("angles", {})
        readout = _angle_readout_children(angles)

        # HKA time-series
        hka_series_all = store.get("_hka_series", {})
        if trial_name in hka_series_all:
            hka_vals = hka_series_all[trial_name]
        else:
            # Static: single value
            hka_val = angles.get("hka")
            hka_vals = [hka_val]

        hka_fig = _build_hka_timeseries_fig(hka_vals, frame_idx)

        return fig3d, readout, hka_fig

    # --- Also re-render when trial changes ----------------------------
    @app.callback(
        [Output("scatter3d", "figure", allow_duplicate=True),
         Output("angle-readout", "children", allow_duplicate=True),
         Output("hka-timeseries", "figure", allow_duplicate=True)],
        Input("trial-dropdown", "value"),
        [State("frame-slider", "value"),
         State("visibility-checks", "value"),
         State("precomputed-store", "data")],
        prevent_initial_call=True,
    )
    def render_on_trial_change(trial_name, frame_idx, vis_checks, store):
        trial_frames = store.get(trial_name, store.get("static", [{}]))
        frame_idx = min(frame_idx, len(trial_frames) - 1)
        frame_idx = max(frame_idx, 0)
        fdata = trial_frames[frame_idx]

        fig3d = _build_3d_figure(
            fdata,
            show_fem_markers="fem_markers" in vis_checks,
            show_tib_markers="tib_markers" in vis_checks,
            show_lcs="lcs" in vis_checks,
            show_mech_axis="mech_axis" in vis_checks,
            show_joint_lines="joint_lines" in vis_checks,
            show_joint_centres="joint_centres" in vis_checks,
            show_landmarks="landmarks" in vis_checks,
        )

        angles = fdata.get("angles", {})
        readout = _angle_readout_children(angles)

        hka_series_all = store.get("_hka_series", {})
        if trial_name in hka_series_all:
            hka_vals = hka_series_all[trial_name]
        else:
            hka_val = angles.get("hka")
            hka_vals = [hka_val]
        hka_fig = _build_hka_timeseries_fig(hka_vals, frame_idx)

        return fig3d, readout, hka_fig

    return app


# ===================================================================
# Standalone entry point
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HTO Mechanical Axis 3D Viewer",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to the directory containing trial CSV files. "
             "If not provided, uses the default data location.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for the Dash server (default: 8050).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Dash in debug mode.",
    )
    args = parser.parse_args()

    # Resolve data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Try common default locations relative to the project root
        candidates = [
            _PROJECT_ROOT / "data",
            _PROJECT_ROOT / "Data",
            _PROJECT_ROOT / "trial_data",
        ]
        data_dir = None
        for c in candidates:
            if c.is_dir():
                data_dir = c
                break
        if data_dir is None:
            print("ERROR: No data directory found. Provide --data-dir argument.")
            print("  Searched:", [str(c) for c in candidates])
            sys.exit(1)

    print(f"Data directory: {data_dir}")

    # Default landmark mapping (adjust to match your experimental setup)
    landmark_mapping = {
        "digitizer_1": ["LFEC", "MFEC", "LTP"],
        "digitizer_2": ["MTP", "LM", "MM"],
    }

    print("Running pipeline...")
    results = run_pipeline(data_dir, landmark_mapping)

    print("Loading trials for visualisation...")
    trials = load_all_trials(data_dir)

    print("Creating Dash app...")
    app = create_app(results, trials)

    print(f"\nStarting server on http://localhost:{args.port}")
    app.run_server(debug=args.debug, port=args.port)
