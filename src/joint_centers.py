"""Joint center computation: HJC, KJC, AJC.

Provides algorithms for estimating the Hip Joint Center (HJC) via the
pivot/sphere-fit method, and computing the Knee Joint Center (KJC)
and Ankle Joint Center (AJC) from digitized anatomical landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .data_loader import TrialData
from .utils import average_positions, inter_marker_distances, rms_error


@dataclass
class HJCResult:
    """Result of Hip Joint Center estimation.

    Attributes:
        position: (3,) estimated HJC position in global coordinates.
        radius: Fitted sphere radius (distance from HJC to markers) in mm.
        residual_std: Standard deviation of sphere-fit residuals in mm.
        per_marker_centers: Dict of individual sphere centers per marker.
        per_marker_radii: Dict of individual sphere radii per marker.
    """
    position: np.ndarray
    radius: float
    residual_std: float
    per_marker_centers: Optional[Dict[str, np.ndarray]] = None
    per_marker_radii: Optional[Dict[str, float]] = None


def _sphere_residuals(params: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Residual function for sphere fitting.

    Args:
        params: [cx, cy, cz, r] — sphere center and radius.
        points: (N, 3) array of points on the sphere surface.

    Returns:
        (N,) residuals: distance_to_center - radius.
    """
    center = params[:3]
    r = params[3]
    distances = np.linalg.norm(points - center, axis=1)
    return distances - r


def fit_sphere(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Fit a sphere to a set of 3D points using least-squares optimization.

    Args:
        points: (N, 3) array of 3D points lying on a sphere surface.

    Returns:
        Tuple of (center, radius, residual_std):
            center: (3,) sphere center coordinates.
            radius: Sphere radius in mm.
            residual_std: Standard deviation of fit residuals in mm.
    """
    # Filter NaN rows
    valid = ~np.any(np.isnan(points), axis=1)
    pts = points[valid]

    if len(pts) < 4:
        raise ValueError("Need at least 4 valid points for sphere fitting.")

    # Initial estimate
    initial_center = np.mean(pts, axis=0)
    initial_radius = np.mean(np.linalg.norm(pts - initial_center, axis=1))
    initial_params = np.append(initial_center, initial_radius)

    result = least_squares(_sphere_residuals, initial_params, args=(pts,))

    center = result.x[:3]
    radius = result.x[3]
    residuals = _sphere_residuals(result.x, pts)
    residual_std = np.std(residuals)

    return center, abs(radius), residual_std


def _compute_rigidity_mask(
    trial: TrialData,
    marker_names: List[str],
    ref_distances: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Compute a boolean mask of frames passing the rigidity check.

    Args:
        trial: Trial data.
        marker_names: Marker names in the cluster.
        ref_distances: Reference pairwise inter-marker distances.
        threshold: Max acceptable RMS deviation (mm).

    Returns:
        (N_frames,) boolean mask — True for frames that pass.
    """
    n_frames = trial.get_marker(marker_names[0]).shape[0]
    mask = np.zeros(n_frames, dtype=bool)

    for i in range(n_frames):
        positions = []
        valid = True
        for name in marker_names:
            pos = trial.get_marker(name)[i]
            if np.any(np.isnan(pos)):
                valid = False
                break
            positions.append(pos)
        if not valid:
            continue
        cur_dists = inter_marker_distances(np.array(positions))
        rms = rms_error(np.abs(cur_dists - ref_distances))
        if rms <= threshold:
            mask[i] = True

    return mask


def compute_hjc(rotation_trials: List[TrialData],
                marker_names: List[str],
                method: str = "pooled",
                reference: Optional["RigidBodyReference"] = None,
                rigidity_threshold: float = 1.0) -> HJCResult:
    """Estimate Hip Joint Center using the pivot/sphere-fit method.

    During hip rotation trials, each femoral marker traces an arc on a sphere
    centered at the hip joint center. By fitting spheres to these trajectories,
    we can estimate the HJC.

    Args:
        rotation_trials: List of TrialData from rotation/pivot dynamic trials.
        marker_names: Femoral marker names (e.g., ["F1", "F2", "F3", "F4", "FC"]).
        method: "pooled" (fit one sphere to ALL marker data) or
                "per_marker" (fit separate spheres, take consensus center).
        reference: Optional RigidBodyReference for rigidity-based frame gating.
            When provided, frames where the cluster deformation exceeds
            rigidity_threshold are excluded before sphere fitting.
        rigidity_threshold: Max RMS inter-marker distance deviation (mm)
            for a frame to be included. Only used when reference is provided.

    Returns:
        HJCResult with estimated HJC position and quality metrics.
    """
    # Collect trajectories from all rotation trials
    all_trajectories: Dict[str, List[np.ndarray]] = {m: [] for m in marker_names}
    total_frames = 0
    kept_frames = 0

    for trial in rotation_trials:
        n_frames = trial.get_marker(marker_names[0]).shape[0]

        # Build per-frame validity mask
        if reference is not None:
            rigidity_mask = _compute_rigidity_mask(
                trial, marker_names, reference.reference_distances,
                rigidity_threshold)
            total_frames += n_frames
            kept_frames += int(np.sum(rigidity_mask))
        else:
            rigidity_mask = np.ones(n_frames, dtype=bool)

        for marker in marker_names:
            traj = trial.get_marker(marker)
            # Filter NaN frames AND rigidity-failed frames
            nan_valid = ~np.any(np.isnan(traj), axis=1)
            combined = nan_valid & rigidity_mask
            if np.any(combined):
                all_trajectories[marker].append(traj[combined])

    if reference is not None:
        pct = (kept_frames / total_frames * 100) if total_frames else 0
        print(f"  Rigidity gating: kept {kept_frames}/{total_frames} frames "
              f"({pct:.1f}%, threshold={rigidity_threshold} mm)")

    if method == "pooled":
        return _hjc_pooled(all_trajectories, marker_names)
    elif method == "per_marker":
        return _hjc_per_marker(all_trajectories, marker_names)
    else:
        raise ValueError(f"Unknown HJC method: {method}. Use 'pooled' or 'per_marker'.")


def _hjc_pooled(trajectories: Dict[str, List[np.ndarray]],
                marker_names: List[str]) -> HJCResult:
    """Pool all marker trajectories and fit a single sphere."""
    all_points = []
    for marker in marker_names:
        for traj in trajectories[marker]:
            all_points.append(traj)

    if not all_points:
        raise ValueError("No valid trajectory data for HJC estimation.")

    pooled = np.vstack(all_points)
    print(f"  HJC sphere-fit: {len(pooled)} total points from "
          f"{len(marker_names)} markers")

    center, radius, residual_std = fit_sphere(pooled)

    # Also compute per-marker centers for diagnostics
    per_marker_centers = {}
    per_marker_radii = {}
    for marker in marker_names:
        marker_pts = np.vstack(trajectories[marker]) if trajectories[marker] else None
        if marker_pts is not None and len(marker_pts) >= 4:
            mc, mr, _ = fit_sphere(marker_pts)
            per_marker_centers[marker] = mc
            per_marker_radii[marker] = mr

    return HJCResult(
        position=center,
        radius=radius,
        residual_std=residual_std,
        per_marker_centers=per_marker_centers,
        per_marker_radii=per_marker_radii,
    )


def _hjc_per_marker(trajectories: Dict[str, List[np.ndarray]],
                    marker_names: List[str]) -> HJCResult:
    """Fit separate spheres per marker, consensus center is the average."""
    centers = []
    radii = []
    per_marker_centers = {}
    per_marker_radii = {}

    for marker in marker_names:
        if not trajectories[marker]:
            continue
        marker_pts = np.vstack(trajectories[marker])
        if len(marker_pts) < 4:
            continue
        mc, mr, _ = fit_sphere(marker_pts)
        centers.append(mc)
        radii.append(mr)
        per_marker_centers[marker] = mc
        per_marker_radii[marker] = mr
        print(f"    {marker}: center={mc.round(2)}, radius={mr:.2f} mm")

    if not centers:
        raise ValueError("Could not fit sphere for any marker.")

    centers = np.array(centers)
    consensus = np.mean(centers, axis=0)
    spread = np.std(np.linalg.norm(centers - consensus, axis=1))

    print(f"  HJC consensus: {consensus.round(2)} mm, spread={spread:.2f} mm")

    return HJCResult(
        position=consensus,
        radius=np.mean(radii),
        residual_std=spread,
        per_marker_centers=per_marker_centers,
        per_marker_radii=per_marker_radii,
    )


def compute_kjc(lfec: np.ndarray, mfec: np.ndarray) -> np.ndarray:
    """Compute Knee Joint Center as midpoint of femoral epicondyles.

    Args:
        lfec: (3,) lateral femoral epicondyle position.
        mfec: (3,) medial femoral epicondyle position.

    Returns:
        (3,) knee joint center position.
    """
    kjc = (lfec + mfec) / 2.0
    epicondylar_width = np.linalg.norm(lfec - mfec)
    print(f"  KJC: {kjc.round(2)} mm (epicondylar width: {epicondylar_width:.2f} mm)")
    return kjc


def compute_ajc(lm: np.ndarray, mm: np.ndarray) -> np.ndarray:
    """Compute Ankle Joint Center as midpoint of malleoli.

    Args:
        lm: (3,) lateral malleolus position.
        mm: (3,) medial malleolus position.

    Returns:
        (3,) ankle joint center position.
    """
    ajc = (lm + mm) / 2.0
    bimalleolar_width = np.linalg.norm(lm - mm)
    print(f"  AJC: {ajc.round(2)} mm (bimalleolar width: {bimalleolar_width:.2f} mm)")
    return ajc
