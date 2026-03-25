"""End-to-end pipeline orchestrator for HTO mechanical axis computation.

Coordinates all processing steps: data loading, LCS construction,
landmark registration, joint center computation, angle calculation,
and dynamic trial tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .angles import HTOAngles, compute_all_angles, print_angles
from .data_loader import (
    ALL_MARKERS,
    FEMORAL_MARKERS,
    TIBIAL_MARKERS,
    TrialData,
    load_all_trials,
    load_trial,
)
from .digitizer import register_landmarks, express_landmark_in_lcs
from .joint_centers import HJCResult, compute_ajc, compute_hjc, compute_kjc
from .rigid_body import (
    FramePose,
    RigidBodyReference,
    apply_marker_relabeling,
    build_reference,
    detect_marker_swap,
    track_dynamic_trial,
    transform_point_to_global,
    validate_rigidity,
)

# Landmark-to-cluster assignment
FEMORAL_LANDMARK_NAMES = ["LFEC", "MFEC"]
TIBIAL_LANDMARK_NAMES = ["LTP", "MTP", "LM", "MM"]


@dataclass
class PipelineResults:
    """Container for all pipeline outputs.

    Attributes:
        femoral_ref: Femoral cluster reference configuration.
        tibial_ref: Tibial cluster reference configuration.
        landmarks_global: Anatomical landmarks in global coords (static).
        landmarks_local: Anatomical landmarks in cluster LCS.
        hjc: HJC estimation result.
        kjc: (3,) Knee joint center.
        ajc: (3,) Ankle joint center.
        static_angles: HTO angles from the static trial.
        dynamic_results: Per-trial dynamic tracking results.
        rigidity_validation: Per-trial rigidity check results.
    """
    femoral_ref: RigidBodyReference
    tibial_ref: RigidBodyReference
    landmarks_global: Dict[str, np.ndarray]
    landmarks_local: Dict[str, np.ndarray]
    hjc: HJCResult
    kjc: np.ndarray
    ajc: np.ndarray
    static_angles: HTOAngles
    dynamic_results: Dict[str, "DynamicTrialResult"] = field(default_factory=dict)
    rigidity_validation: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class DynamicTrialResult:
    """Results from processing a single dynamic trial.

    Attributes:
        trial_name: Descriptive name of the trial.
        n_frames: Number of frames.
        femoral_poses: Per-frame femoral cluster poses.
        tibial_poses: Per-frame tibial cluster poses.
        hjc_per_frame: (N, 3) HJC positions per frame.
        kjc_per_frame: (N, 3) KJC positions per frame.
        ajc_per_frame: (N, 3) AJC positions per frame.
        angles_per_frame: Per-frame HTO angles.
        landmarks_per_frame: Per-frame landmark positions in global coords.
    """
    trial_name: str
    n_frames: int
    femoral_poses: List[Optional[FramePose]]
    tibial_poses: List[Optional[FramePose]]
    hjc_per_frame: np.ndarray
    kjc_per_frame: np.ndarray
    ajc_per_frame: np.ndarray
    angles_per_frame: List[Optional[HTOAngles]]
    landmarks_per_frame: Dict[str, np.ndarray]


def run_pipeline(
    data_dir: str | Path,
    landmark_mapping: Dict[str, List[str]],
    hjc_method: str = "pooled",
    segmentation_params: Optional[Dict] = None,
) -> PipelineResults:
    """Run the complete HTO mechanical axis computation pipeline.

    Args:
        data_dir: Path to the directory containing trial CSV files.
        landmark_mapping: Dict mapping digitizer trial names (as used by
            load_all_trials, e.g., "digitizer_1", "digitizer_2") to
            ordered lists of landmark names. Example:
            {"digitizer_1": ["LFEC", "MFEC", "LTP"],
             "digitizer_2": ["MTP", "LM", "MM"]}
        hjc_method: Method for HJC computation ("pooled" or "per_marker").
        segmentation_params: Optional dict of parameters for digitizer
            auto-segmentation (velocity_threshold, min_duration_s, etc.).

    Returns:
        PipelineResults with all computed outputs.
    """
    data_dir = Path(data_dir)
    seg_params = segmentation_params or {}

    # =========================================================================
    # Step 1: Load all trials
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Loading trial data")
    print("=" * 60)
    trials = load_all_trials(data_dir)

    if "static" not in trials:
        raise ValueError("Static trial is required but not found.")

    # =========================================================================
    # Step 2: Build reference LCS for both clusters
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Building reference LCS from static trial")
    print("=" * 60)
    static = trials["static"]

    print("\n  Femoral cluster (F1, F2, F3, F4, FC):")
    femoral_ref = build_reference(static.markers, FEMORAL_MARKERS)
    print(f"    Centroid: {femoral_ref.centroid.round(2)} mm")
    print(f"    LCS orientation:\n{femoral_ref.rotation.round(4)}")

    print("\n  Tibial cluster (T1, T2, T3, T4, TC):")
    tibial_ref = build_reference(static.markers, TIBIAL_MARKERS)
    print(f"    Centroid: {tibial_ref.centroid.round(2)} mm")
    print(f"    LCS orientation:\n{tibial_ref.rotation.round(4)}")

    # =========================================================================
    # Step 2b: Detect and correct marker label swaps in non-static trials
    # =========================================================================
    print("\n  Checking for marker label swaps...")
    for trial_name, trial in trials.items():
        if trial_name == "static":
            continue
        for cluster_name, ref, markers in [
            ("femoral", femoral_ref, FEMORAL_MARKERS),
            ("tibial", tibial_ref, TIBIAL_MARKERS),
        ]:
            subset = trial.get_markers_subset(markers)
            relabel = detect_marker_swap(ref, subset)
            if relabel:
                print(f"    {trial_name} {cluster_name}: SWAP detected "
                      f"{relabel} — correcting")
                corrected = apply_marker_relabeling(trial.markers, relabel)
                trial.markers.update(corrected)

    # =========================================================================
    # Step 3: Register anatomical landmarks from digitizer trials
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Registering anatomical landmarks")
    print("=" * 60)

    all_global_landmarks: Dict[str, np.ndarray] = {}
    all_local_landmarks: Dict[str, np.ndarray] = {}

    # Accumulate all measurements per landmark for weighted averaging
    _landmark_measurements: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    for trial_name, landmark_names in landmark_mapping.items():
        if trial_name not in trials:
            print(f"  Warning: Trial '{trial_name}' not found, skipping.")
            continue

        print(f"\n  Processing {trial_name} -> {landmark_names}")
        trial = trials[trial_name]

        # Register landmarks (auto-segmentation)
        global_lms, global_stds = register_landmarks(
            trial, landmark_names, **seg_params)

        for name in global_lms:
            if name not in _landmark_measurements:
                _landmark_measurements[name] = []
            _landmark_measurements[name].append(
                (global_lms[name], global_stds[name]))

    # Merge landmarks — inverse-variance weighted average when repeated
    print("\n  Merging repeated landmarks (inverse-variance weighting):")
    for name, measurements in _landmark_measurements.items():
        if len(measurements) == 1:
            all_global_landmarks[name] = measurements[0][0]
        else:
            positions = np.array([m[0] for m in measurements])
            stds = np.array([m[1] for m in measurements])
            # Inverse-variance weighting per axis
            variances = np.maximum(stds ** 2, 1e-6)
            weights = 1.0 / variances
            weight_sum = np.sum(weights, axis=0)
            weighted_avg = np.sum(positions * weights, axis=0) / weight_sum

            naive_avg = np.mean(positions, axis=0)
            dist = np.linalg.norm(positions[0] - positions[-1])
            shift = np.linalg.norm(weighted_avg - naive_avg)
            rel_weights = weights / weight_sum
            print(f"    '{name}': {len(measurements)} measurements, "
                  f"spread={dist:.2f} mm, "
                  f"shift from naive avg={shift:.3f} mm")
            for i, (_, m_std) in enumerate(measurements):
                w = rel_weights[i]
                print(f"      meas {i+1}: std={m_std.round(3)}, "
                      f"weight=[{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}]")
            all_global_landmarks[name] = weighted_avg

    # Express in appropriate cluster LCS
    for name, pos in all_global_landmarks.items():
        if name in FEMORAL_LANDMARK_NAMES:
            all_local_landmarks[name] = express_landmark_in_lcs(
                pos, femoral_ref)
        elif name in TIBIAL_LANDMARK_NAMES:
            all_local_landmarks[name] = express_landmark_in_lcs(
                pos, tibial_ref)

    # Estimate LFEC if not digitized (mirror MFEC along medial-lateral axis)
    if "LFEC" not in all_global_landmarks:
        if "MFEC" in all_global_landmarks and \
           "LTP" in all_global_landmarks and "MTP" in all_global_landmarks:
            from .utils import normalize as _normalize
            mfec = all_global_landmarks["MFEC"]
            ltp = all_global_landmarks["LTP"]
            mtp = all_global_landmarks["MTP"]
            knee_center = (ltp + mtp) / 2.0
            # Mirror MFEC across the sagittal plane through knee_center.
            # Only flip the medial-lateral component, preserving the
            # proximal-distal level so LFEC stays at the epicondylar height.
            ml_dir = _normalize(ltp - mtp)
            offset = mfec - knee_center
            ml_component = np.dot(offset, ml_dir)
            lfec_est = mfec - 2.0 * ml_component * ml_dir
            all_global_landmarks["LFEC"] = lfec_est
            all_local_landmarks["LFEC"] = express_landmark_in_lcs(
                lfec_est, femoral_ref)
            print(f"\n  LFEC estimated (mirror MFEC along M-L axis):")
            print(f"    LFEC: {lfec_est.round(2)} mm")

    # Verify all required landmarks were found
    required = ["LFEC", "MFEC", "LTP", "MTP"]
    missing = [r for r in required if r not in all_global_landmarks]
    if missing:
        raise ValueError(
            f"Missing required landmarks: {missing}. "
            f"Found: {list(all_global_landmarks.keys())}")

    # =========================================================================
    # Step 4: Compute HJC from rotation trials
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Computing Hip Joint Center (HJC)")
    print("=" * 60)

    rotation_trial_names = ["rotation_1", "rotation_2", "rotation_3"]
    rotation_trials = [trials[n] for n in rotation_trial_names if n in trials]

    if not rotation_trials:
        raise ValueError("No rotation trials found for HJC computation.")

    print(f"  Using {len(rotation_trials)} rotation trial(s)")
    hjc_result = compute_hjc(rotation_trials, FEMORAL_MARKERS, method=hjc_method,
                             reference=femoral_ref)
    print(f"  HJC position: {hjc_result.position.round(2)} mm")
    print(f"  Sphere radius: {hjc_result.radius:.2f} mm")
    print(f"  Residual std: {hjc_result.residual_std:.3f} mm")

    # =========================================================================
    # Step 5: Compute KJC and AJC
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Computing KJC and AJC")
    print("=" * 60)

    kjc = compute_kjc(all_global_landmarks["LFEC"], all_global_landmarks["MFEC"])

    if "LM" in all_global_landmarks and "MM" in all_global_landmarks:
        ajc = compute_ajc(all_global_landmarks["LM"], all_global_landmarks["MM"])
    else:
        print("  Warning: Malleoli landmarks not found. AJC will be estimated.")
        ajc = _estimate_ajc_fallback(kjc, tibial_ref)

    # =========================================================================
    # Step 6: Compute static angles
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Computing HTO angles (static)")
    print("=" * 60)

    static_angles = compute_all_angles(
        hjc=hjc_result.position,
        kjc=kjc,
        ajc=ajc,
        lfec=all_global_landmarks["LFEC"],
        mfec=all_global_landmarks["MFEC"],
        ltp=all_global_landmarks["LTP"],
        mtp=all_global_landmarks["MTP"],
        lm=all_global_landmarks.get("LM"),
        mm=all_global_landmarks.get("MM"),
    )
    print_angles(static_angles)

    results = PipelineResults(
        femoral_ref=femoral_ref,
        tibial_ref=tibial_ref,
        landmarks_global=all_global_landmarks,
        landmarks_local=all_local_landmarks,
        hjc=hjc_result,
        kjc=kjc,
        ajc=ajc,
        static_angles=static_angles,
    )

    # =========================================================================
    # Step 7: Process dynamic trials
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Processing dynamic trials")
    print("=" * 60)

    dynamic_trial_names = [
        "left_right_1", "left_right_2",
        "up_down_1", "up_down_2",
        "rotation_1", "rotation_2", "rotation_3",
    ]

    for trial_name in dynamic_trial_names:
        if trial_name not in trials:
            continue

        print(f"\n  Processing {trial_name}...")
        trial = trials[trial_name]

        try:
            dyn_result = _process_dynamic_trial(
                trial_name, trial,
                femoral_ref, tibial_ref,
                all_local_landmarks,
                hjc_result.position,
            )
            results.dynamic_results[trial_name] = dyn_result
            print(f"    Processed {dyn_result.n_frames} frames")

            # Rigidity validation
            for cluster_name, ref, markers in [
                ("femoral", femoral_ref, FEMORAL_MARKERS),
                ("tibial", tibial_ref, TIBIAL_MARKERS),
            ]:
                marker_trajs = trial.get_markers_subset(markers)
                val = validate_rigidity(marker_trajs, ref)
                key = f"{trial_name}_{cluster_name}"
                results.rigidity_validation[key] = val
                status = "PASS" if val["passes"] else "FAIL"
                print(f"    Rigidity {cluster_name}: {status} "
                      f"(mean RMS={val['mean_rms']:.3f} mm)")

        except Exception as e:
            print(f"    Error processing {trial_name}: {e}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)

    return results


def _process_dynamic_trial(
    trial_name: str,
    trial: TrialData,
    femoral_ref: RigidBodyReference,
    tibial_ref: RigidBodyReference,
    local_landmarks: Dict[str, np.ndarray],
    static_hjc: np.ndarray,
) -> DynamicTrialResult:
    """Process a single dynamic trial: track poses and compute per-frame angles."""

    n_frames = trial.n_frames

    # Track rigid body poses
    femoral_markers = trial.get_markers_subset(FEMORAL_MARKERS)
    tibial_markers = trial.get_markers_subset(TIBIAL_MARKERS)

    femoral_poses = track_dynamic_trial(femoral_ref, femoral_markers)
    tibial_poses = track_dynamic_trial(tibial_ref, tibial_markers)

    # Compute per-frame landmarks and joint centers
    hjc_per_frame = np.full((n_frames, 3), np.nan)
    kjc_per_frame = np.full((n_frames, 3), np.nan)
    ajc_per_frame = np.full((n_frames, 3), np.nan)
    landmarks_per_frame: Dict[str, np.ndarray] = {
        name: np.full((n_frames, 3), np.nan) for name in local_landmarks
    }
    angles_per_frame: List[Optional[HTOAngles]] = [None] * n_frames

    for i in range(n_frames):
        f_pose = femoral_poses[i]
        t_pose = tibial_poses[i]

        if f_pose is None or t_pose is None:
            continue

        # Transform landmarks to global
        frame_landmarks = {}
        for name, local_pos in local_landmarks.items():
            if name in FEMORAL_LANDMARK_NAMES:
                gpos = transform_point_to_global(local_pos, f_pose)
            else:
                gpos = transform_point_to_global(local_pos, t_pose)
            frame_landmarks[name] = gpos
            landmarks_per_frame[name][i] = gpos

        # HJC: use static HJC (it doesn't move with the femoral cluster in
        # the same way for a phantom model)
        # For a real patient, HJC would be re-estimated or the femoral cluster
        # would be tracking the femur relative to a stationary pelvis.
        hjc_per_frame[i] = static_hjc

        # KJC and AJC from transformed landmarks
        if "LFEC" in frame_landmarks and "MFEC" in frame_landmarks:
            kjc_per_frame[i] = (frame_landmarks["LFEC"] + frame_landmarks["MFEC"]) / 2

        if "LM" in frame_landmarks and "MM" in frame_landmarks:
            ajc_per_frame[i] = (frame_landmarks["LM"] + frame_landmarks["MM"]) / 2

        # Compute angles for this frame
        if (not np.any(np.isnan(hjc_per_frame[i])) and
            not np.any(np.isnan(kjc_per_frame[i])) and
            not np.any(np.isnan(ajc_per_frame[i])) and
            "LFEC" in frame_landmarks and "MFEC" in frame_landmarks and
            "LTP" in frame_landmarks and "MTP" in frame_landmarks):

            try:
                angles_per_frame[i] = compute_all_angles(
                    hjc=hjc_per_frame[i],
                    kjc=kjc_per_frame[i],
                    ajc=ajc_per_frame[i],
                    lfec=frame_landmarks["LFEC"],
                    mfec=frame_landmarks["MFEC"],
                    ltp=frame_landmarks["LTP"],
                    mtp=frame_landmarks["MTP"],
                    lm=frame_landmarks.get("LM"),
                    mm=frame_landmarks.get("MM"),
                )
            except (ValueError, ZeroDivisionError):
                pass

    return DynamicTrialResult(
        trial_name=trial_name,
        n_frames=n_frames,
        femoral_poses=femoral_poses,
        tibial_poses=tibial_poses,
        hjc_per_frame=hjc_per_frame,
        kjc_per_frame=kjc_per_frame,
        ajc_per_frame=ajc_per_frame,
        angles_per_frame=angles_per_frame,
        landmarks_per_frame=landmarks_per_frame,
    )


def _estimate_ajc_fallback(kjc: np.ndarray,
                           tibial_ref: RigidBodyReference,
                           tibial_length_mm: float = 380.0) -> np.ndarray:
    """Fallback AJC estimation when malleoli are not available.

    Estimates AJC by projecting along the tibial axis direction.
    """
    # Use the tibial cluster's principal axis (Z of LCS) as tibial direction
    tibial_axis = tibial_ref.rotation[:, 2]  # Third column
    # Ensure it points distally (away from KJC)
    to_kjc = kjc - tibial_ref.centroid
    if np.dot(tibial_axis, to_kjc) > 0:
        tibial_axis = -tibial_axis

    ajc = kjc + tibial_length_mm * tibial_axis
    print(f"  AJC (estimated): {ajc.round(2)} mm "
          f"(tibial length={tibial_length_mm} mm)")
    return ajc


def extract_angle_time_series(
    dynamic_result: DynamicTrialResult,
    sampling_rate: float,
) -> Dict[str, np.ndarray]:
    """Extract angle time series from a dynamic trial result.

    Returns a dict with keys 'time', 'hka', 'mldfa', 'mpta', 'jlca', 'mldta',
    each as (N,) arrays. NaN where angles could not be computed.
    """
    n = dynamic_result.n_frames
    time = np.arange(n) / sampling_rate

    hka = np.full(n, np.nan)
    mldfa = np.full(n, np.nan)
    mpta = np.full(n, np.nan)
    jlca = np.full(n, np.nan)
    mldta = np.full(n, np.nan)

    for i, angles in enumerate(dynamic_result.angles_per_frame):
        if angles is not None:
            hka[i] = angles.hka
            mldfa[i] = angles.mldfa
            mpta[i] = angles.mpta
            jlca[i] = angles.jlca
            if angles.mldta is not None:
                mldta[i] = angles.mldta

    return {
        "time": time,
        "hka": hka,
        "mldfa": mldfa,
        "mpta": mpta,
        "jlca": jlca,
        "mldta": mldta,
    }
