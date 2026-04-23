"""Algebraic sphere-fit pipeline for joint centre estimation.

Mirrors the Mathematica reference approach:
  - HJC from femoral markers via algebraic sphere-fit (common centre, per-marker radii)
  - KJC from tibial markers via the same algebraic sphere-fit
  - AJC from malleoli midpoint (unchanged)
  - All other landmarks and angles from the original pipeline

Outputs ``outputs/landmarks_2.json`` for comparison with the original
``outputs/landmarks.json`` and the Mathematica reference values.

Usage:
    python main_algebraic.py --data-dir data/04-Feb-Trials/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.algebraic_sphere_fit import compute_joint_center
from src.angles import compute_all_angles, print_angles
from src.data_loader import (
    FEMORAL_MARKERS,
    TIBIAL_MARKERS,
    load_all_trials,
)
from src.digitizer import register_landmarks, express_landmark_in_lcs
from src.joint_centers import compute_ajc, compute_kjc
from src.rigid_body import build_reference
from src.utils import normalize


FEMORAL_LANDMARK_NAMES = ["LFEC", "MFEC"]
TIBIAL_LANDMARK_NAMES = ["LTP", "MTP", "LM", "MM"]


def get_default_landmark_mapping() -> Dict[str, List[str]]:
    return {
        "digitizer_1": ["MFEC", "MM", "LM"],
        "digitizer_2": ["MFEC", "MTP", "LTP", "MM"],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Algebraic sphere-fit pipeline (Mathematica reference)")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--landmark-config", type=str, default=None)
    parser.add_argument("--velocity-threshold", type=float, default=15.0)
    parser.add_argument("--min-contact-duration", type=float, default=0.5)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    # Landmark mapping
    if args.landmark_config:
        with open(args.landmark_config) as f:
            landmark_mapping = json.load(f)
    else:
        landmark_mapping = get_default_landmark_mapping()

    # =========================================================================
    # Step 1: Load all trials
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Loading trial data")
    print("=" * 60)
    trials = load_all_trials(data_dir)

    if "static" not in trials:
        print("Error: Static trial is required.")
        return 1

    # =========================================================================
    # Step 2: Build reference LCS
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Building reference LCS from static trial")
    print("=" * 60)
    static = trials["static"]

    femoral_ref = build_reference(static.markers, FEMORAL_MARKERS)
    tibial_ref = build_reference(static.markers, TIBIAL_MARKERS)

    # =========================================================================
    # Step 3: Register anatomical landmarks
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Registering anatomical landmarks")
    print("=" * 60)

    seg_params = {
        "velocity_threshold": args.velocity_threshold,
        "min_duration_s": args.min_contact_duration,
    }

    all_global_landmarks: Dict[str, np.ndarray] = {}

    for trial_name, lm_names in landmark_mapping.items():
        if trial_name not in trials:
            print(f"  Warning: Trial '{trial_name}' not found, skipping.")
            continue

        print(f"\n  Processing {trial_name} -> {lm_names}")
        trial = trials[trial_name]
        global_lms = register_landmarks(trial, lm_names, **seg_params)

        for name, pos in global_lms.items():
            if name in all_global_landmarks:
                prev = all_global_landmarks[name]
                all_global_landmarks[name] = (prev + pos) / 2.0
            else:
                all_global_landmarks[name] = pos

    # Estimate LFEC if not digitized
    if "LFEC" not in all_global_landmarks:
        if ("MFEC" in all_global_landmarks and
                "LTP" in all_global_landmarks and
                "MTP" in all_global_landmarks):
            mfec = all_global_landmarks["MFEC"]
            ltp = all_global_landmarks["LTP"]
            mtp = all_global_landmarks["MTP"]
            knee_center = (ltp + mtp) / 2.0
            ml_dir = normalize(ltp - mtp)
            offset = mfec - knee_center
            ml_component = np.dot(offset, ml_dir)
            lfec_est = mfec - 2.0 * ml_component * ml_dir
            all_global_landmarks["LFEC"] = lfec_est
            print(f"\n  LFEC estimated (mirror MFEC): {lfec_est.round(2)} mm")

    # =========================================================================
    # Step 4: Algebraic sphere-fit for HJC and KJC
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Algebraic sphere-fit — HJC (femoral) & KJC (tibial)")
    print("=" * 60)

    rotation_trial_names = ["rotation_1", "rotation_2", "rotation_3"]
    rotation_trials = [trials[n] for n in rotation_trial_names if n in trials]

    if not rotation_trials:
        print("Error: No rotation trials found.")
        return 1

    print(f"  Using {len(rotation_trials)} rotation trial(s)")

    # HJC — sphere-fit on femoral markers
    hjc, hjc_radii, hjc_res = compute_joint_center(
        rotation_trials, FEMORAL_MARKERS, label="HJC")

    # KJC — sphere-fit on tibial markers
    kjc_sf, kjc_radii, kjc_res = compute_joint_center(
        rotation_trials, TIBIAL_MARKERS, label="KJC")

    # Also compute epicondyle-midpoint KJC for comparison
    kjc_epi = compute_kjc(
        all_global_landmarks["LFEC"], all_global_landmarks["MFEC"])
    print(f"\n  KJC (epicondyle midpoint): {kjc_epi.round(3)}")
    print(f"  KJC (sphere-fit, tibial):  {kjc_sf.round(3)}")

    # Use sphere-fit KJC as primary (matches Mathematica)
    kjc = kjc_sf

    # =========================================================================
    # Step 5: AJC
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Computing AJC")
    print("=" * 60)

    if "LM" in all_global_landmarks and "MM" in all_global_landmarks:
        ajc = compute_ajc(
            all_global_landmarks["LM"], all_global_landmarks["MM"])
    else:
        print("  Warning: Malleoli not available.")
        ajc = np.full(3, np.nan)

    # =========================================================================
    # Step 6: Compute angles
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Computing HTO angles (static, algebraic centres)")
    print("=" * 60)

    static_angles = compute_all_angles(
        hjc=hjc,
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

    # =========================================================================
    # Step 7: Export landmarks_2.json
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Exporting results")
    print("=" * 60)

    landmarks_dict = {
        name: {"x": round(float(pos[0]), 3),
               "y": round(float(pos[1]), 3),
               "z": round(float(pos[2]), 3)}
        for name, pos in all_global_landmarks.items()
    }
    landmarks_dict["HJC"] = {
        "x": round(float(hjc[0]), 3),
        "y": round(float(hjc[1]), 3),
        "z": round(float(hjc[2]), 3),
    }
    landmarks_dict["KJC_sphere_fit"] = {
        "x": round(float(kjc_sf[0]), 3),
        "y": round(float(kjc_sf[1]), 3),
        "z": round(float(kjc_sf[2]), 3),
    }
    landmarks_dict["KJC_epicondyle"] = {
        "x": round(float(kjc_epi[0]), 3),
        "y": round(float(kjc_epi[1]), 3),
        "z": round(float(kjc_epi[2]), 3),
    }
    landmarks_dict["AJC"] = {
        "x": round(float(ajc[0]), 3),
        "y": round(float(ajc[1]), 3),
        "z": round(float(ajc[2]), 3),
    }

    # Angles
    angles_dict = {
        "HKA_deg": round(static_angles.hka, 2),
        "mLDFA_deg": round(static_angles.mldfa, 2),
        "MPTA_deg": round(static_angles.mpta, 2),
        "JLCA_deg": round(static_angles.jlca, 2),
        "mLDTA_deg": round(static_angles.mldta, 2)
        if static_angles.mldta else None,
        "knee_offset_mm": round(static_angles.knee_offset_mm, 2),
    }

    output = {
        "landmarks": landmarks_dict,
        "static_angles": angles_dict,
        "hjc_per_marker_radii": {
            k: round(v, 2) for k, v in hjc_radii.items()},
        "kjc_per_marker_radii": {
            k: round(v, 2) for k, v in kjc_radii.items()},
    }

    out_path = output_dir / "landmarks_2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  -> {out_path}")

    # Print comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON: Algebraic pipeline vs Mathematica reference")
    print("=" * 60)
    print(f"  Python HJC (algebraic): {hjc.round(3)}")
    print(f"  Mathematica HJC:        [-242.298, 36.369, 217.166]")
    print()
    print(f"  Python KJC (sphere-fit): {kjc_sf.round(3)}")
    print(f"  Mathematica KJC:         [47.542, 585.659, 100.009]")
    print()
    print(f"  Python KJC (epicondyle): {kjc_epi.round(3)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
