"""Data loading for Vicon motion capture CSV files.

Handles dual-structure CSVs exported from Vicon Nexus that contain
force plate data (top section) followed by marker trajectory data
(bottom section, starting after the 'Trajectories' keyword).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TrialData:
    """Container for marker trajectory data from a single trial."""

    filepath: Path
    sampling_rate: float
    marker_names: List[str]
    markers: Dict[str, np.ndarray]  # {marker_name: (N, 3) array}
    n_frames: int

    def get_marker(self, name: str) -> np.ndarray:
        """Get Nx3 trajectory for a marker, raising KeyError if missing."""
        if name not in self.markers:
            available = ", ".join(sorted(self.markers.keys()))
            raise KeyError(
                f"Marker '{name}' not found. Available: {available}"
            )
        return self.markers[name]

    def get_markers_subset(self, names: List[str]) -> Dict[str, np.ndarray]:
        """Get trajectories for a list of markers."""
        return {name: self.get_marker(name) for name in names}


# Marker groups for the experimental setup
FEMORAL_MARKERS = ["F1", "F2", "F3", "F4", "FC"]
TIBIAL_MARKERS = ["T1", "T2", "T3", "T4", "TC"]
DIGITIZER_MARKERS = ["D1", "D2", "D3", "D4", "DT"]
ALL_MARKERS = DIGITIZER_MARKERS + FEMORAL_MARKERS + TIBIAL_MARKERS


def _find_trajectory_section(filepath: Path) -> int:
    """Find the line number where the 'Trajectories' section begins.

    Scans the file for a line containing exactly 'Trajectories' (case-sensitive).

    Returns:
        Line number (0-indexed) of the 'Trajectories' keyword.

    Raises:
        ValueError: If no trajectory section found in the file.
    """
    with open(filepath, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if line.strip() == "Trajectories":
                return i
    raise ValueError(
        f"No 'Trajectories' section found in {filepath}. "
        "Ensure this is a Vicon-exported CSV with marker trajectory data."
    )


def _parse_marker_header(header_line: str) -> List[str]:
    """Parse marker names from the CSV header line.

    Expects format: ',,Subject1:D3,,,Subject1:D4,,,...'
    Each marker name appears once, followed by two empty columns (for Y, Z).

    Returns:
        Ordered list of marker names (e.g., ['D3', 'D4', 'D2', ...]).
    """
    parts = header_line.strip().rstrip(",").split(",")
    marker_names = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Match 'Subject1:MarkerName' or 'Subject 1:MarkerName'
        match = re.match(r"(?:Subject\s*\d+:)?(.+)", part)
        if match:
            name = match.group(1).strip()
            if name not in ("Frame", "Sub Frame", "X", "Y", "Z", "mm"):
                marker_names.append(name)
    return marker_names


def load_trial(filepath: str | Path) -> TrialData:
    """Load marker trajectory data from a Vicon-exported CSV file.

    Handles dual-structure CSVs where force plate data appears first,
    followed by marker trajectories after the 'Trajectories' keyword.

    Args:
        filepath: Path to the CSV file.

    Returns:
        TrialData containing marker trajectories and metadata.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Trial file not found: {filepath}")

    # Find trajectory section
    traj_line = _find_trajectory_section(filepath)

    # Read the relevant lines
    with open(filepath, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # Line layout after 'Trajectories':
    # traj_line + 0: "Trajectories"
    # traj_line + 1: sampling rate (e.g., "100")
    # traj_line + 2: marker names header
    # traj_line + 3: "Frame,Sub Frame,X,Y,Z,X,Y,Z,..."
    # traj_line + 4: units ",,mm,mm,mm,..."
    # traj_line + 5+: data rows

    sampling_rate = float(lines[traj_line + 1].strip())
    marker_names = _parse_marker_header(lines[traj_line + 2])

    # Read data starting from the data rows (skip 5 header lines)
    data_start = traj_line + 5
    data_lines = lines[data_start:]

    # Parse numeric data
    rows = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        try:
            row = [float(x) if x.strip() else np.nan for x in parts]
            rows.append(row)
        except ValueError:
            continue

    if not rows:
        raise ValueError(f"No data rows found in trajectory section of {filepath}")

    data = np.array(rows)

    # First two columns are Frame and Sub Frame
    # Remaining columns are X,Y,Z triplets for each marker
    n_frames = data.shape[0]
    marker_data = data[:, 2:]  # Skip Frame, Sub Frame

    # Build marker dictionary
    markers: Dict[str, np.ndarray] = {}
    for i, name in enumerate(marker_names):
        col_start = i * 3
        col_end = col_start + 3
        if col_end <= marker_data.shape[1]:
            markers[name] = marker_data[:, col_start:col_end].copy()

    return TrialData(
        filepath=filepath,
        sampling_rate=sampling_rate,
        marker_names=marker_names,
        markers=markers,
        n_frames=n_frames,
    )


def load_all_trials(data_dir: str | Path) -> Dict[str, TrialData]:
    """Load all trial CSV files from a directory.

    Returns:
        Dictionary mapping descriptive trial names to TrialData objects.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory not found: {data_dir}")

    # Map descriptive names to filenames
    trial_files = {
        "static": "Static_Trial01.csv",
        "digitizer_1": "Digitizer_trial.csv",
        "digitizer_2": "Digitizer_trial 1.csv",
        "rotation_1": "Dynamic Trial.csv",
        "rotation_2": "Dynamic Trial 1.csv",
        "rotation_3": "Dynamic Trial 2.csv",
        "left_right_1": "Dynamic Trial_Left_Right_Motion.csv",
        "left_right_2": "Dynamic Trial_Left_Right_Motion 1.csv",
        "up_down_1": "Dynamic Trial_Up_Down_Motion.csv",
        "up_down_2": "Dynamic Trial_Up_Down_Motion 1.csv",
    }

    trials: Dict[str, TrialData] = {}
    for name, filename in trial_files.items():
        fpath = data_dir / filename
        if fpath.exists():
            try:
                trials[name] = load_trial(fpath)
                print(f"  Loaded {name}: {filename} "
                      f"({trials[name].n_frames} frames, "
                      f"{trials[name].sampling_rate} Hz, "
                      f"{len(trials[name].marker_names)} markers)")
            except (ValueError, Exception) as e:
                print(f"  Warning: Could not load {filename}: {e}")
        else:
            print(f"  Skipped {name}: {filename} (file not found)")

    return trials
