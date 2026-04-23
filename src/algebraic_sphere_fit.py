"""Algebraic sphere-fit for joint center estimation.

Implements the linearized least-squares sphere-fitting approach used in the
Mathematica reference notebook.  For N markers, each tracing an arc on a
sphere centred at the joint, the method solves:

    x*A + y*B + z*C + D_j = -(x^2 + y^2 + z^2)

for every data point (x, y, z) of marker j.  This yields a single shared
centre = (-A/2, -B/2, -C/2) and per-marker radii derived from D_j.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .data_loader import TrialData


def algebraic_sphere_fit_common_center(
    trajectories: Dict[str, np.ndarray],
    marker_names: List[str],
) -> Tuple[np.ndarray, Dict[str, float], float]:
    """Algebraic sphere-fit with a common centre and per-marker radii.

    Args:
        trajectories: ``{marker_name: (N_frames, 3)}`` arrays.  NaN rows
            are silently skipped.
        marker_names: Ordered list of marker names to include.

    Returns:
        (center, per_marker_radii, residual_std)
            center: (3,) fitted sphere centre.
            per_marker_radii: ``{name: radius_mm}`` for each marker.
            residual_std: Std-dev of the algebraic residuals (quality metric).
    """
    n_markers = len(marker_names)
    A_rows: List[np.ndarray] = []
    b_rows: List[float] = []

    for j, name in enumerate(marker_names):
        traj = trajectories[name]
        valid = ~np.any(np.isnan(traj), axis=1)
        pts = traj[valid]

        for point in pts:
            x, y, z = point
            row = np.zeros(3 + n_markers)
            row[0] = x
            row[1] = y
            row[2] = z
            row[3 + j] = 1.0
            A_rows.append(row)
            b_rows.append(-(x**2 + y**2 + z**2))

    if not A_rows:
        raise ValueError("No valid data points for sphere fitting.")

    A_mat = np.array(A_rows)
    b_vec = np.array(b_rows)

    print(f"  Algebraic sphere-fit: {len(A_rows)} data points "
          f"from {n_markers} markers")

    coeffs, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

    A_coef, B_coef, C_coef = coeffs[0], coeffs[1], coeffs[2]
    center = np.array([-A_coef / 2, -B_coef / 2, -C_coef / 2])

    # Per-marker radii
    center_norm_sq = (A_coef / 2)**2 + (B_coef / 2)**2 + (C_coef / 2)**2
    per_marker_radii: Dict[str, float] = {}
    for j, name in enumerate(marker_names):
        D_j = coeffs[3 + j]
        r_sq = center_norm_sq - D_j
        per_marker_radii[name] = float(np.sqrt(max(r_sq, 0.0)))

    # Residual quality
    fitted = A_mat @ coeffs
    residual_std = float(np.std(b_vec - fitted))

    return center, per_marker_radii, residual_std


def compute_joint_center(
    rotation_trials: List[TrialData],
    marker_names: List[str],
    label: str = "Joint",
) -> Tuple[np.ndarray, Dict[str, float], float]:
    """Compute a joint centre from rotation-trial marker trajectories.

    Pools marker data across all supplied trials, then runs the algebraic
    sphere-fit with a common centre and per-marker radii.

    Args:
        rotation_trials: Dynamic trials where the markers trace arcs.
        marker_names: Marker names belonging to the cluster.
        label: Display label for console output.

    Returns:
        (center, per_marker_radii, residual_std)
    """
    # Pool trajectories across trials
    combined: Dict[str, List[np.ndarray]] = {m: [] for m in marker_names}
    for trial in rotation_trials:
        for marker in marker_names:
            traj = trial.get_marker(marker)
            valid = ~np.any(np.isnan(traj), axis=1)
            if np.any(valid):
                combined[marker].append(traj[valid])

    # Stack per-marker
    stacked: Dict[str, np.ndarray] = {}
    for name in marker_names:
        if combined[name]:
            stacked[name] = np.vstack(combined[name])
        else:
            raise ValueError(f"No valid data for marker '{name}'.")

    print(f"\n  {label} — algebraic sphere-fit on "
          f"{[n for n in marker_names]}:")

    center, radii, residual_std = algebraic_sphere_fit_common_center(
        stacked, marker_names)

    print(f"  {label} center (sphere-fit, global) = "
          f"{center.round(3)}")
    for name, r in radii.items():
        print(f"    {name}: radius = {r:.2f} mm")
    print(f"  Residual std = {residual_std:.3f}")

    return center, radii, residual_std
