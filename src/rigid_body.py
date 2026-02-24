"""Rigid body operations: LCS construction, Kabsch tracking, rigidity validation.

Provides the core algorithms for establishing local coordinate systems (LCS)
from rigid marker clusters and tracking them through dynamic motion capture frames
using the Kabsch (SVD-based) alignment algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from itertools import permutations as _permutations

from .utils import (
    average_positions,
    centroid,
    ensure_right_handed,
    inter_marker_distances,
    normalize,
    rms_error,
)


@dataclass
class RigidBodyReference:
    """Reference configuration of a rigid marker cluster from the static trial.

    Attributes:
        marker_names: Ordered list of marker names in the cluster.
        centroid: (3,) global centroid position in reference pose.
        rotation: (3, 3) rotation matrix defining the LCS orientation.
        local_coords: Dict mapping marker name to (3,) local coordinates.
        reference_distances: Pairwise inter-marker distances for rigidity check.
    """
    marker_names: List[str]
    centroid: np.ndarray
    rotation: np.ndarray
    local_coords: Dict[str, np.ndarray]
    reference_distances: np.ndarray


@dataclass
class FramePose:
    """Pose of a rigid body at a single dynamic frame.

    Attributes:
        origin: (3,) global position of the LCS origin (centroid).
        rotation: (3, 3) rotation matrix (global LCS orientation).
        fit_rmse: RMS error of the Kabsch fit (mm).
    """
    origin: np.ndarray
    rotation: np.ndarray
    fit_rmse: float


def build_reference(marker_trajectories: Dict[str, np.ndarray],
                    marker_names: List[str]) -> RigidBodyReference:
    """Build the reference configuration for a rigid cluster from the static trial.

    Averages marker positions across all static frames, computes the centroid,
    establishes a local coordinate system via SVD, and stores each marker's
    local coordinates.

    Algorithm:
        1. Average each marker's trajectory across frames -> p_i^ref
        2. Compute centroid c^ref = mean(p_i^ref)
        3. Center markers: q_i = p_i^ref - c^ref
        4. SVD on Q = [q1; q2; ...; qN] (Nx3 matrix)
        5. R_ref = Vt.T (columns are LCS axes)
        6. Ensure right-handedness
        7. Local coords: p_i^local = R_ref^T @ q_i

    Args:
        marker_trajectories: Dict mapping marker names to (N_frames, 3) arrays.
        marker_names: List of marker names belonging to this cluster.

    Returns:
        RigidBodyReference containing the reference configuration.
    """
    # Step 1: Average positions across frames
    ref_positions = {}
    for name in marker_names:
        if name not in marker_trajectories:
            raise KeyError(f"Marker '{name}' not found in trajectory data.")
        ref_positions[name] = average_positions(marker_trajectories[name])

    # Stack into (N_markers, 3) matrix
    positions_array = np.array([ref_positions[name] for name in marker_names])

    # Step 2: Compute centroid
    c_ref = centroid(positions_array)

    # Step 3: Center the markers
    centered = positions_array - c_ref  # (N_markers, 3)

    # Step 4: SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Step 5: LCS axes from right singular vectors
    # Vt rows are the principal directions; Vt.T columns are the axes
    R_ref = Vt.T  # (3, 3)

    # Step 6: Ensure right-handedness
    R_ref = ensure_right_handed(R_ref)

    # Step 7: Compute local coordinates for each marker
    local_coords = {}
    for name, pos in ref_positions.items():
        local_coords[name] = R_ref.T @ (pos - c_ref)

    # Reference pairwise distances for rigidity validation
    ref_distances = inter_marker_distances(positions_array)

    return RigidBodyReference(
        marker_names=marker_names,
        centroid=c_ref,
        rotation=R_ref,
        local_coords=local_coords,
        reference_distances=ref_distances,
    )


def kabsch_align(reference_local: np.ndarray,
                 current_centered: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute optimal rotation via Kabsch algorithm.

    Finds R that minimizes sum ||R @ q_i^ref - q_i^current||^2.

    Algorithm:
        H = Q_ref^T @ Q_current  (3x3 cross-covariance)
        U, S, Vt = SVD(H)
        d = det(Vt^T @ U^T)
        R = Vt^T @ diag(1, 1, d) @ U^T

    Args:
        reference_local: (N, 3) local coordinates of markers (from reference).
        current_centered: (N, 3) centered current marker positions.

    Returns:
        Tuple of (R, rmse) where R is (3, 3) rotation and rmse is fit error in mm.
    """
    # Cross-covariance matrix
    H = reference_local.T @ current_centered  # (3, 3)

    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (not reflection)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, d])

    R = Vt.T @ sign_matrix @ U.T

    # Compute RMSE
    transformed = (R @ reference_local.T).T  # (N, 3)
    residuals = np.linalg.norm(transformed - current_centered, axis=1)
    rmse = rms_error(residuals)

    return R, rmse


def track_frame(reference: RigidBodyReference,
                current_positions: Dict[str, np.ndarray]) -> FramePose:
    """Compute the rigid body pose at a single frame using Kabsch alignment.

    Args:
        reference: Reference configuration from static trial.
        current_positions: Dict mapping marker names to (3,) positions at this frame.

    Returns:
        FramePose with the global LCS origin, rotation, and fit quality.
    """
    # Collect marker positions in the same order as reference
    available_markers = [n for n in reference.marker_names
                         if n in current_positions
                         and not np.any(np.isnan(current_positions[n]))]

    if len(available_markers) < 3:
        raise ValueError(
            f"Need at least 3 visible markers for rigid body tracking, "
            f"got {len(available_markers)}."
        )

    # Build arrays
    ref_local = np.array([reference.local_coords[n] for n in available_markers])
    cur_global = np.array([current_positions[n] for n in available_markers])

    # Compute current centroid
    c_current = centroid(cur_global)

    # Center current positions
    cur_centered = cur_global - c_current

    # Kabsch alignment
    R, rmse = kabsch_align(ref_local, cur_centered)

    return FramePose(
        origin=c_current,
        rotation=R @ reference.rotation,
        fit_rmse=rmse,
    )


def track_dynamic_trial(reference: RigidBodyReference,
                        marker_trajectories: Dict[str, np.ndarray],
                        ) -> List[Optional[FramePose]]:
    """Track rigid body pose through all frames of a dynamic trial.

    Args:
        reference: Reference configuration from static trial.
        marker_trajectories: Dict mapping marker names to (N_frames, 3) arrays.

    Returns:
        List of FramePose (one per frame). None entries indicate frames
        where tracking failed (insufficient visible markers).
    """
    # Determine number of frames from any marker
    sample_marker = reference.marker_names[0]
    n_frames = marker_trajectories[sample_marker].shape[0]

    poses: List[Optional[FramePose]] = []
    for frame_idx in range(n_frames):
        # Extract single-frame positions
        frame_positions = {}
        for name in reference.marker_names:
            if name in marker_trajectories:
                pos = marker_trajectories[name][frame_idx]
                frame_positions[name] = pos

        try:
            pose = track_frame(reference, frame_positions)
            poses.append(pose)
        except ValueError:
            poses.append(None)

    return poses


def transform_point_to_global(local_point: np.ndarray,
                              pose: FramePose) -> np.ndarray:
    """Transform a point from the rigid body's local coordinates to global.

    Args:
        local_point: (3,) point in the cluster's local coordinate frame.
        pose: Current frame pose of the rigid body.

    Returns:
        (3,) point in global coordinates.
    """
    # The pose rotation already incorporates R(t) @ R_ref
    # local_point is in the reference LCS, so we need:
    # p_global = R_kabsch @ local_point + origin
    # But pose.rotation = R_kabsch @ R_ref, and local_point = R_ref^T @ (p - c_ref)
    # So: p_global = R_kabsch @ R_ref @ R_ref^T @ (p - c_ref) + c_current
    #             = R_kabsch @ (p_ref - c_ref) + c_current
    # Which means we need R_kabsch alone, not pose.rotation.
    # Let's use the reference rotation to go back:
    # local_point is stored as R_ref^T @ (p - c_ref)
    # So p_in_ref_frame = R_ref @ local_point = (p - c_ref)
    # Then p_global = R_kabsch @ (p - c_ref) + c_current
    # Since pose.rotation = R_kabsch @ R_ref:
    # R_kabsch = pose.rotation @ R_ref^T  ... but we don't store R_ref in pose
    #
    # Simpler: work directly with the stored transform.
    # pose.rotation columns ARE the global LCS axes.
    # local_point is expressed in the reference LCS.
    # So: p_global = pose.rotation @ local_point + pose.origin
    # This works because pose.rotation = R_kabsch @ R_ref, and
    # local_point = R_ref^T @ (p_ref - c_ref).
    return pose.rotation @ local_point + pose.origin


def validate_rigidity(marker_trajectories: Dict[str, np.ndarray],
                      reference: RigidBodyReference,
                      threshold_mm: float = 0.5) -> Dict:
    """Validate the rigid body assumption across all frames.

    Computes inter-marker distances at each frame and compares them
    to the reference distances.

    Args:
        marker_trajectories: Dict mapping marker names to (N_frames, 3) arrays.
        reference: Reference configuration with expected distances.
        threshold_mm: Maximum acceptable RMS deviation in mm.

    Returns:
        Dict with keys:
            'rms_per_frame': (N_frames,) RMS distance deviation per frame
            'max_deviation': Maximum deviation across all frames
            'mean_rms': Mean RMS across all frames
            'passes': Whether mean_rms < threshold_mm
            'flagged_frames': Indices of frames exceeding threshold
    """
    marker_names = reference.marker_names
    sample_traj = marker_trajectories[marker_names[0]]
    n_frames = sample_traj.shape[0]

    rms_per_frame = np.zeros(n_frames)
    max_deviation = 0.0

    for frame_idx in range(n_frames):
        positions = []
        valid = True
        for name in marker_names:
            pos = marker_trajectories[name][frame_idx]
            if np.any(np.isnan(pos)):
                valid = False
                break
            positions.append(pos)

        if not valid:
            rms_per_frame[frame_idx] = np.nan
            continue

        positions = np.array(positions)
        current_distances = inter_marker_distances(positions)
        deviations = np.abs(current_distances - reference.reference_distances)
        rms = rms_error(deviations)
        rms_per_frame[frame_idx] = rms
        max_deviation = max(max_deviation, np.max(deviations))

    valid_frames = ~np.isnan(rms_per_frame)
    mean_rms = np.mean(rms_per_frame[valid_frames]) if np.any(valid_frames) else np.nan
    flagged = np.where(rms_per_frame > threshold_mm)[0]

    return {
        "rms_per_frame": rms_per_frame,
        "max_deviation": max_deviation,
        "mean_rms": mean_rms,
        "passes": mean_rms < threshold_mm,
        "flagged_frames": flagged,
    }


def detect_marker_swap(
    reference: RigidBodyReference,
    trial_markers: Dict[str, np.ndarray],
    error_threshold: float = 5.0,
) -> Optional[Dict[str, str]]:
    """Detect if marker labels are swapped between a trial and the reference.

    Compares inter-marker distances at the first valid frame against
    reference distances under all possible permutations of marker labels.

    Args:
        reference: Reference configuration from the static trial.
        trial_markers: Dict mapping marker names to (N, 3) trajectories.
        error_threshold: Max mean distance error (mm) for a valid match.

    Returns:
        None if no swap detected (identity permutation is best).
        Dict mapping {trial_label: correct_label} if a swap improves fit.
    """
    marker_names = reference.marker_names

    # Find first frame with all markers visible
    n_frames = trial_markers[marker_names[0]].shape[0]
    frame = None
    for f in range(min(n_frames, 100)):
        all_valid = True
        for name in marker_names:
            if np.any(np.isnan(trial_markers[name][f])):
                all_valid = False
                break
        if all_valid:
            frame = f
            break

    if frame is None:
        return None

    ref_dists = reference.reference_distances

    # Check identity first
    identity_positions = np.array(
        [trial_markers[n][frame] for n in marker_names])
    identity_dists = inter_marker_distances(identity_positions)
    identity_err = np.mean(np.abs(identity_dists - ref_dists))

    if identity_err < error_threshold:
        return None  # No swap needed

    # Try all permutations
    best_perm = None
    best_err = identity_err

    for perm in _permutations(range(len(marker_names))):
        perm_names = [marker_names[p] for p in perm]
        positions = np.array(
            [trial_markers[n][frame] for n in perm_names])
        dists = inter_marker_distances(positions)
        err = np.mean(np.abs(dists - ref_dists))
        if err < best_err:
            best_err = err
            best_perm = perm

    if best_perm is None or best_err > error_threshold:
        return None

    # Build relabeling map: {trial_label: should_be_label}
    perm_names = [marker_names[p] for p in best_perm]
    relabel_map = {}
    for orig, perm_name in zip(marker_names, perm_names):
        if orig != perm_name:
            relabel_map[perm_name] = orig

    return relabel_map


def apply_marker_relabeling(
    markers: Dict[str, np.ndarray],
    relabel_map: Dict[str, str],
) -> Dict[str, np.ndarray]:
    """Apply a marker relabeling to trajectory data.

    Args:
        markers: Dict mapping marker names to (N, 3) trajectories.
        relabel_map: Dict mapping {wrong_label: correct_label}.

    Returns:
        New dict with corrected marker labels.
    """
    result = dict(markers)
    # Collect the data for swapped markers
    temp = {}
    for wrong, correct in relabel_map.items():
        if wrong in result:
            temp[correct] = result[wrong]

    result.update(temp)
    return result
