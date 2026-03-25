"""Digitizer landmark registration with auto-segmentation.

Handles the registration of anatomical landmarks from digitizer trials.
Each trial may contain multiple landmarks digitized sequentially — the
digitizer tip (DT) is held stationary on each landmark for a period.

The auto-segmentation algorithm detects stationary contact periods
from the DT velocity profile and extracts one landmark position per segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_loader import TrialData
from .rigid_body import RigidBodyReference
from .utils import average_positions


@dataclass
class DetectedSegment:
    """A detected stationary segment in the digitizer tip trajectory.

    Attributes:
        start_frame: Index of the first frame in the segment.
        end_frame: Index of the last frame (inclusive).
        position: (3,) averaged DT position during the segment.
        std: (3,) per-axis standard deviation during the segment (mm).
        n_frames: Number of frames in the segment.
    """
    start_frame: int
    end_frame: int
    position: np.ndarray
    std: np.ndarray
    n_frames: int


def compute_velocity(trajectory: np.ndarray,
                     sampling_rate: float) -> np.ndarray:
    """Compute frame-to-frame velocity magnitude of a marker trajectory.

    Args:
        trajectory: (N, 3) marker positions.
        sampling_rate: Sampling rate in Hz.

    Returns:
        (N-1,) velocity magnitudes in mm/s.
    """
    displacements = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(displacements, axis=1) * sampling_rate
    return speeds


def detect_stationary_segments(
    trajectory: np.ndarray,
    sampling_rate: float,
    velocity_threshold: float = 2.0,
    min_duration_s: float = 0.3,
    position_std_threshold: float = 1.0,
) -> List[DetectedSegment]:
    """Detect stationary segments in a digitizer tip trajectory.

    A segment is stationary when the DT velocity stays below a threshold
    for at least a minimum duration, and position standard deviation is low.

    Args:
        trajectory: (N, 3) DT marker positions.
        sampling_rate: Sampling rate in Hz.
        velocity_threshold: Max velocity in mm/s to be considered stationary.
        min_duration_s: Minimum segment duration in seconds.
        position_std_threshold: Max per-axis std (mm) for a valid segment.

    Returns:
        List of DetectedSegment objects, ordered by start frame.
    """
    min_frames = int(min_duration_s * sampling_rate)
    speeds = compute_velocity(trajectory, sampling_rate)

    # Pad speeds to match trajectory length (first frame has no velocity)
    speeds = np.concatenate([[speeds[0]], speeds])

    # Find frames below velocity threshold
    is_stationary = speeds < velocity_threshold

    # Find contiguous runs of stationary frames
    segments: List[DetectedSegment] = []
    in_segment = False
    seg_start = 0

    for i in range(len(is_stationary)):
        if is_stationary[i] and not in_segment:
            seg_start = i
            in_segment = True
        elif not is_stationary[i] and in_segment:
            # End of segment
            seg_end = i - 1
            seg_length = seg_end - seg_start + 1
            if seg_length >= min_frames:
                _add_segment(segments, trajectory, seg_start, seg_end,
                             position_std_threshold)
            in_segment = False

    # Handle segment running to end of data
    if in_segment:
        seg_end = len(is_stationary) - 1
        seg_length = seg_end - seg_start + 1
        if seg_length >= min_frames:
            _add_segment(segments, trajectory, seg_start, seg_end,
                         position_std_threshold)

    return segments


def _add_segment(segments: List[DetectedSegment], trajectory: np.ndarray,
                 start: int, end: int, std_threshold: float) -> None:
    """Add a segment if its position standard deviation is acceptable."""
    segment_data = trajectory[start:end + 1]

    # Filter NaN frames
    valid = ~np.any(np.isnan(segment_data), axis=1)
    if not np.any(valid):
        return

    valid_data = segment_data[valid]
    pos = np.mean(valid_data, axis=0)
    std = np.std(valid_data, axis=0)

    if np.all(std < std_threshold):
        segments.append(DetectedSegment(
            start_frame=start,
            end_frame=end,
            position=pos,
            std=std,
            n_frames=end - start + 1,
        ))


def _strip_rest_segments(segments: List[DetectedSegment],
                         distance_threshold: float = 20.0,
                         rest_position: Optional[np.ndarray] = None,
                         ) -> List[DetectedSegment]:
    """Remove rest-position segments at the start and end of a trial.

    The digitizer typically rests at its holder before and after digitizing,
    producing stationary segments at the start/end that are NOT landmarks.

    Detection strategies (in order):
    1. If rest_position is provided, strip segments within distance_threshold of it.
    2. If first and last segment positions are within distance_threshold, strip both.
    3. If the first segment starts at frame 0, check if it matches the digitizer
       holder area (far from any typical bone marker positions).

    Args:
        segments: Ordered list of detected segments.
        distance_threshold: Max distance (mm) to consider as the same position.
        rest_position: Known rest position of the digitizer holder.

    Returns:
        Filtered list with rest segments removed.
    """
    if len(segments) < 2:
        return segments

    result = list(segments)

    # Strategy 1: known rest position
    if rest_position is not None:
        result = [s for s in result
                  if np.linalg.norm(s.position - rest_position) > distance_threshold]
        return result

    # Strategy 2: first and last match each other
    first_pos = result[0].position
    last_pos = result[-1].position
    dist = np.linalg.norm(first_pos - last_pos)

    if dist < distance_threshold:
        return result[1:-1]

    # Strategy 3: first segment starts at frame 0 — very likely a rest position
    # (the digitizer was resting in its holder when recording started)
    if result[0].start_frame == 0 and len(result) > 2:
        other_positions = np.array([s.position for s in result[1:]])
        other_mean = np.mean(other_positions, axis=0)
        first_dist = np.linalg.norm(result[0].position - other_mean)
        if first_dist > 100.0:
            result = result[1:]

    return result


def register_landmarks(
    trial: TrialData,
    landmark_names: List[str],
    velocity_threshold: float = 15.0,
    min_duration_s: float = 0.5,
    position_std_threshold: float = 1.5,
    strip_rest: bool = True,
    skip_first: int = 0,
    skip_last: int = 0,
    rest_position: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Register anatomical landmarks from a digitizer trial.

    Auto-segments stationary contact periods and assigns them to
    landmark names in order.

    Args:
        trial: Loaded digitizer trial data.
        landmark_names: Ordered list of landmark names. The i-th detected
            stationary segment is assigned to the i-th name.
        velocity_threshold: mm/s threshold for stationarity detection.
        min_duration_s: Minimum contact duration in seconds.
        position_std_threshold: Max positional std (mm) per axis.
        strip_rest: If True, auto-remove rest-position segments at
            start/end of trial (detected as first/last segments with
            nearly identical positions).
        skip_first: Number of segments to skip from the start (after
            auto-stripping). Use when auto-strip misses a rest segment.
        skip_last: Number of segments to skip from the end.
        rest_position: Known rest position of the digitizer holder (3,).
            If provided, any segment within 20mm of this position is removed.

    Returns:
        Tuple of (landmarks, landmark_stds):
            landmarks: Dict mapping landmark names to (3,) global positions.
            landmark_stds: Dict mapping landmark names to (3,) per-axis std (mm).

    Raises:
        ValueError: If number of detected segments doesn't match landmark count.
    """
    dt_trajectory = trial.get_marker("DT")

    segments = detect_stationary_segments(
        dt_trajectory,
        trial.sampling_rate,
        velocity_threshold=velocity_threshold,
        min_duration_s=min_duration_s,
        position_std_threshold=position_std_threshold,
    )

    if strip_rest:
        n_before = len(segments)
        segments = _strip_rest_segments(segments, rest_position=rest_position)
        if len(segments) < n_before:
            print(f"    Stripped {n_before - len(segments)} rest segment(s)")

    # Manual skip overrides
    if skip_first > 0 or skip_last > 0:
        end = len(segments) - skip_last if skip_last > 0 else len(segments)
        segments = segments[skip_first:end]
        if skip_first > 0 or skip_last > 0:
            print(f"    Skipped {skip_first} first + {skip_last} last segment(s)")

    if len(segments) != len(landmark_names):
        raise ValueError(
            f"Detected {len(segments)} stationary segments but expected "
            f"{len(landmark_names)} landmarks ({landmark_names}). "
            f"Adjust velocity_threshold ({velocity_threshold} mm/s) or "
            f"min_duration_s ({min_duration_s} s) parameters."
        )

    landmarks = {}
    landmark_stds = {}
    for seg, name in zip(segments, landmark_names):
        landmarks[name] = seg.position
        landmark_stds[name] = seg.std
        print(f"  Landmark '{name}': position={seg.position.round(2)} mm, "
              f"std={seg.std.round(3)} mm, frames={seg.start_frame}-{seg.end_frame}")

    return landmarks, landmark_stds


def express_landmark_in_lcs(
    landmark_global: np.ndarray,
    reference: RigidBodyReference,
) -> np.ndarray:
    """Express a global landmark position in a rigid body's local coordinate system.

    Args:
        landmark_global: (3,) landmark position in global coordinates.
        reference: Reference configuration of the rigid cluster.

    Returns:
        (3,) landmark position in the cluster's local coordinate frame.
    """
    return reference.rotation.T @ (landmark_global - reference.centroid)


def register_and_localize(
    trial: TrialData,
    landmark_names: List[str],
    femoral_ref: RigidBodyReference,
    tibial_ref: RigidBodyReference,
    femoral_landmarks: List[str],
    tibial_landmarks: List[str],
    **kwargs,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Register landmarks and express them in the appropriate cluster LCS.

    Args:
        trial: Digitizer trial data.
        landmark_names: Ordered list of all landmark names in this trial.
        femoral_ref: Femoral cluster reference configuration.
        tibial_ref: Tibial cluster reference configuration.
        femoral_landmarks: Names of landmarks belonging to the femoral segment
            (e.g., ["LFEC", "MFEC"]).
        tibial_landmarks: Names of landmarks belonging to the tibial segment
            (e.g., ["LTP", "MTP", "LM", "MM"]).
        **kwargs: Additional arguments passed to register_landmarks().

    Returns:
        Tuple of (global_landmarks, local_landmarks) dicts.
        global_landmarks: {name: (3,) global position}
        local_landmarks: {name: (3,) local position in appropriate cluster LCS}
    """
    global_landmarks, _ = register_landmarks(trial, landmark_names, **kwargs)

    local_landmarks = {}
    for name, pos in global_landmarks.items():
        if name in femoral_landmarks:
            local_landmarks[name] = express_landmark_in_lcs(pos, femoral_ref)
        elif name in tibial_landmarks:
            local_landmarks[name] = express_landmark_in_lcs(pos, tibial_ref)
        else:
            # Unknown association — store in global only
            print(f"  Warning: Landmark '{name}' not assigned to femoral or "
                  f"tibial cluster. Stored in global coordinates only.")

    return global_landmarks, local_landmarks


def get_segment_diagnostics(trial: TrialData,
                            velocity_threshold: float = 2.0,
                            min_duration_s: float = 0.3) -> Dict:
    """Get diagnostic information about detected segments for debugging.

    Useful for tuning segmentation parameters.

    Args:
        trial: Digitizer trial data.
        velocity_threshold: mm/s threshold.
        min_duration_s: Minimum duration in seconds.

    Returns:
        Dict with 'segments', 'velocity_profile', 'dt_trajectory'.
    """
    dt = trial.get_marker("DT")
    speeds = compute_velocity(dt, trial.sampling_rate)

    segments = detect_stationary_segments(
        dt, trial.sampling_rate,
        velocity_threshold=velocity_threshold,
        min_duration_s=min_duration_s,
    )

    return {
        "segments": segments,
        "velocity_profile": speeds,
        "dt_trajectory": dt,
        "n_segments": len(segments),
        "sampling_rate": trial.sampling_rate,
    }
