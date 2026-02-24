"""Shared math utilities for biomechanical computations.

Provides helper functions for vector operations, projections,
angle calculations, and coordinate frame operations used across
the pipeline modules.
"""

from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector or array of vectors to unit length.

    Args:
        v: Vector (3,) or array of vectors (N, 3).

    Returns:
        Unit vector(s) with same shape as input.

    Raises:
        ValueError: If any vector has zero magnitude.
    """
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            raise ValueError("Cannot normalize zero-length vector.")
        return v / norm
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Cannot normalize zero-length vector(s).")
    return v / norms


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between two 3D vectors in degrees.

    Args:
        v1, v2: 3D vectors.

    Returns:
        Angle in degrees [0, 180].
    """
    v1n = normalize(v1)
    v2n = normalize(v2)
    cos_angle = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def signed_angle_in_plane(v1: np.ndarray, v2: np.ndarray,
                          plane_normal: np.ndarray) -> float:
    """Compute signed angle from v1 to v2 in a plane defined by its normal.

    Positive angle = v1 rotates counter-clockwise toward v2
    when viewed from the direction the normal points.

    Args:
        v1, v2: 3D vectors (will be projected onto the plane).
        plane_normal: Normal vector of the reference plane.

    Returns:
        Signed angle in degrees (-180, 180].
    """
    n = normalize(plane_normal)
    # Project vectors onto the plane
    v1p = project_onto_plane(v1, n)
    v2p = project_onto_plane(v2, n)
    v1p = normalize(v1p)
    v2p = normalize(v2p)

    cos_angle = np.clip(np.dot(v1p, v2p), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    # Determine sign using cross product
    cross = np.cross(v1p, v2p)
    if np.dot(cross, n) < 0:
        angle = -angle

    return np.degrees(angle)


def project_onto_plane(v: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Project a vector onto a plane defined by its normal.

    Args:
        v: Vector to project.
        plane_normal: Unit normal of the plane.

    Returns:
        Projected vector lying in the plane.
    """
    n = normalize(plane_normal)
    return v - np.dot(v, n) * n


def project_point_onto_plane(point: np.ndarray, plane_point: np.ndarray,
                             plane_normal: np.ndarray) -> np.ndarray:
    """Project a 3D point onto a plane.

    Args:
        point: Point to project.
        plane_point: Any point on the plane.
        plane_normal: Unit normal of the plane.

    Returns:
        Projected point on the plane.
    """
    n = normalize(plane_normal)
    d = np.dot(point - plane_point, n)
    return point - d * n


def point_to_line_distance(point: np.ndarray, line_start: np.ndarray,
                           line_end: np.ndarray) -> float:
    """Compute perpendicular distance from a point to a line.

    Args:
        point: 3D point.
        line_start, line_end: Two points defining the line.

    Returns:
        Perpendicular distance in same units as input.
    """
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        return np.linalg.norm(point - line_start)
    line_unit = line_vec / line_len
    t = np.dot(point - line_start, line_unit)
    closest = line_start + t * line_unit
    return np.linalg.norm(point - closest)


def ensure_right_handed(R: np.ndarray) -> np.ndarray:
    """Ensure a 3x3 rotation matrix is right-handed.

    If det(R) < 0, flips the third column to make the system right-handed.

    Args:
        R: 3x3 matrix.

    Returns:
        Right-handed 3x3 matrix.
    """
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    return R


def centroid(points: np.ndarray) -> np.ndarray:
    """Compute the centroid of a set of 3D points.

    Args:
        points: (N, 3) array of points.

    Returns:
        (3,) centroid vector.
    """
    return np.mean(points, axis=0)


def rms_error(errors: np.ndarray) -> float:
    """Compute root-mean-square of an error array.

    Args:
        errors: 1D array of error values.

    Returns:
        RMS value.
    """
    return np.sqrt(np.mean(errors ** 2))


def inter_marker_distances(positions: np.ndarray) -> np.ndarray:
    """Compute all pairwise distances between markers.

    Args:
        positions: (N, 3) array where N is the number of markers.

    Returns:
        (N*(N-1)/2,) array of pairwise distances.
    """
    from scipy.spatial.distance import pdist
    return pdist(positions)


def average_positions(trajectories: np.ndarray) -> np.ndarray:
    """Average marker positions across frames.

    Args:
        trajectories: (N_frames, 3) array for a single marker.

    Returns:
        (3,) averaged position.
    """
    # Filter out NaN frames
    valid = ~np.any(np.isnan(trajectories), axis=1)
    if not np.any(valid):
        raise ValueError("All frames contain NaN values.")
    return np.mean(trajectories[valid], axis=0)
