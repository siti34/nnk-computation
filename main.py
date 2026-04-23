"""CLI entry point for HTO Mechanical Axis Computation Pipeline.

Usage:
    python main.py --data-dir data/04-Feb-Trials/
    python main.py --data-dir data/04-Feb-Trials/ --visualize
    python main.py --data-dir data/04-Feb-Trials/ --output-dir outputs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HTO Mechanical Axis Computation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data-dir data/04-Feb-Trials/
  python main.py --data-dir data/04-Feb-Trials/ --visualize
  python main.py --data-dir data/04-Feb-Trials/ --output-dir outputs/ --hjc-method per_marker
        """,
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to directory containing trial CSV files.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directory for output files (default: outputs/).",
    )
    parser.add_argument(
        "--landmark-config", type=str, default=None,
        help="Path to JSON file mapping digitizer trials to landmark names. "
             "If not provided, uses default mapping.",
    )
    parser.add_argument(
        "--hjc-method", type=str, choices=["pooled", "per_marker"],
        default="per_marker",
        help="HJC estimation method (default: per_marker).",
    )
    parser.add_argument(
        "--velocity-threshold", type=float, default=15.0,
        help="Digitizer velocity threshold in mm/s (default: 15.0).",
    )
    parser.add_argument(
        "--min-contact-duration", type=float, default=0.5,
        help="Minimum digitizer contact duration in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Launch Plotly Dash 3D visualization after pipeline.",
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="Port for Dash visualization server (default: 8050).",
    )
    return parser.parse_args(argv)


def get_default_landmark_mapping() -> Dict[str, List[str]]:
    """Default mapping of digitizer trials to landmark names.

    Based on spatial analysis of detected landmarks:
    - digitizer_1 (Digitizer_trial.csv): MFEC, MM, LM
      (1 knee point then 2 ankle points)
    - digitizer_2 (Digitizer_trial 1.csv): MFEC (repeat), MTP, LTP, MM (repeat)
      (repeated MFEC, then 2 knee points, then repeated MM)

    LFEC is estimated from the available landmarks (see pipeline).
    When a landmark appears in multiple trials, positions are averaged.
    Override with --landmark-config if your protocol differs.
    """
    return {
        "digitizer_1": ["MFEC", "MM", "LM"],
        "digitizer_2": ["MFEC", "MTP", "LTP", "MM"],
    }


def export_results(results, output_dir: Path) -> None:
    """Export pipeline results to CSV and JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export static angles
    angles = results.static_angles
    angles_dict = {
        "HKA_deg": round(angles.hka, 2),
        "mLDFA_deg": round(angles.mldfa, 2),
        "MPTA_deg": round(angles.mpta, 2),
        "JLCA_deg": round(angles.jlca, 2),
        "mLDTA_deg": round(angles.mldta, 2) if angles.mldta else None,
        "knee_offset_mm": round(angles.knee_offset_mm, 2),
    }

    with open(output_dir / "static_angles.json", "w") as f:
        json.dump(angles_dict, f, indent=2)
    print(f"  Static angles -> {output_dir / 'static_angles.json'}")

    # Export landmarks
    landmarks_dict = {
        name: {"x": round(float(pos[0]), 3),
               "y": round(float(pos[1]), 3),
               "z": round(float(pos[2]), 3)}
        for name, pos in results.landmarks_global.items()
    }
    landmarks_dict["HJC"] = {
        "x": round(float(results.hjc.position[0]), 3),
        "y": round(float(results.hjc.position[1]), 3),
        "z": round(float(results.hjc.position[2]), 3),
    }
    landmarks_dict["KJC"] = {
        "x": round(float(results.kjc[0]), 3),
        "y": round(float(results.kjc[1]), 3),
        "z": round(float(results.kjc[2]), 3),
    }
    landmarks_dict["AJC"] = {
        "x": round(float(results.ajc[0]), 3),
        "y": round(float(results.ajc[1]), 3),
        "z": round(float(results.ajc[2]), 3),
    }

    with open(output_dir / "landmarks.json", "w") as f:
        json.dump(landmarks_dict, f, indent=2)
    print(f"  Landmarks -> {output_dir / 'landmarks.json'}")

    # Export dynamic angle time series
    from src.pipeline import extract_angle_time_series

    for trial_name, dyn_result in results.dynamic_results.items():
        ts = extract_angle_time_series(dyn_result, 100.0)
        df = pd.DataFrame(ts)
        csv_path = output_dir / f"angles_{trial_name}.csv"
        df.to_csv(csv_path, index=False, float_format="%.3f")
        print(f"  {trial_name} angles -> {csv_path}")

    # Export rigidity validation summary
    rigidity_summary = {}
    for key, val in results.rigidity_validation.items():
        rigidity_summary[key] = {
            "mean_rms_mm": round(float(val["mean_rms"]), 4),
            "max_deviation_mm": round(float(val["max_deviation"]), 4),
            "passes": bool(val["passes"]),
            "flagged_frame_count": len(val["flagged_frames"]),
        }

    with open(output_dir / "rigidity_validation.json", "w") as f:
        json.dump(rigidity_summary, f, indent=2)
    print(f"  Rigidity validation -> {output_dir / 'rigidity_validation.json'}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.is_dir():
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    # Load landmark mapping
    if args.landmark_config:
        config_path = Path(args.landmark_config)
        if not config_path.exists():
            print(f"Error: Landmark config not found: {config_path}")
            return 1
        with open(config_path) as f:
            landmark_mapping = json.load(f)
    else:
        landmark_mapping = get_default_landmark_mapping()
        print("Using default landmark mapping:")
        for trial, lms in landmark_mapping.items():
            print(f"  {trial}: {lms}")
        print("(Override with --landmark-config path/to/config.json)\n")

    # Run pipeline
    from src.pipeline import run_pipeline

    segmentation_params = {
        "velocity_threshold": args.velocity_threshold,
        "min_duration_s": args.min_contact_duration,
    }

    try:
        results = run_pipeline(
            data_dir=data_dir,
            landmark_mapping=landmark_mapping,
            hjc_method=args.hjc_method,
            segmentation_params=segmentation_params,
        )
    except Exception as e:
        print(f"\nPipeline error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Export results
    print(f"\nExporting results to {output_dir}/")
    export_results(results, output_dir)

    # Launch visualization if requested
    if args.visualize:
        print(f"\nLaunching 3D visualization on http://localhost:{args.port}")
        from src.data_loader import load_all_trials
        from visualization.dash_app import create_app

        trials = load_all_trials(data_dir)
        app = create_app(results, trials)
        app.run(debug=False, port=args.port)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())