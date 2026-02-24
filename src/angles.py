"""HTO clinical angle computation.

Computes the mechanical axis alignment angles used in High Tibial Osteotomy
(HTO) planning:
    - HKA: Hip-Knee-Ankle angle (primary alignment metric)
    - mLDFA: mechanical Lateral Distal Femoral Angle
    - MPTA: Medial Proximal Tibial Angle
    - JLCA: Joint Line Convergence Angle
    - mLDTA: mechanical Lateral Distal Tibial Angle

All angles are computed in the frontal plane defined by the femoral
anatomical coordinate frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .utils import (
    angle_between_vectors,
    normalize,
    project_onto_plane,
    signed_angle_in_plane,
)


@dataclass
class HTOAngles:
    """Container for all HTO-relevant clinical angles.

    All angles are in degrees. Convention follows Paley's deformity analysis.

    Attributes:
        hka: Hip-Knee-Ankle angle. 180° = neutral, <180° = varus, >180° = valgus.
        mldfa: mechanical Lateral Distal Femoral Angle. Normal ~87°.
        mpta: Medial Proximal Tibial Angle. Normal ~87°.
        jlca: Joint Line Convergence Angle. Normal 0-2°.
        mldta: mechanical Lateral Distal Tibial Angle. Normal ~89°.
        knee_offset_mm: Perpendicular distance from KJC to mechanical axis (mm).
        frontal_normal: (3,) normal vector of the frontal plane used for projections.
    """
    hka: float
    mldfa: float
    mpta: float
    jlca: float
    mldta: Optional[float]
    knee_offset_mm: float
    frontal_normal: np.ndarray


def _build_frontal_plane_normal(hjc: np.ndarray, kjc: np.ndarray,
                                lfec: np.ndarray, mfec: np.ndarray) -> np.ndarray:
    """Construct the frontal plane normal from anatomical landmarks.

    The frontal (coronal) plane contains the mechanical axis and the
    medial-lateral direction. Its normal points anteriorly.

    Construction:
        1. Mechanical axis direction: v_mech = HJC - KJC (proximal direction)
        2. Medial-lateral direction: v_ml = LFEC - MFEC
        3. Frontal normal = v_mech x v_ml (points anteriorly for right-hand rule)
        4. Normalize

    Args:
        hjc, kjc: Joint centers defining the mechanical axis.
        lfec, mfec: Epicondyles defining the medial-lateral axis.

    Returns:
        (3,) unit normal vector of the frontal plane (anterior direction).
    """
    v_mech = hjc - kjc
    v_ml = lfec - mfec
    normal = np.cross(v_mech, v_ml)
    return normalize(normal)


def compute_hka(hjc: np.ndarray, kjc: np.ndarray, ajc: np.ndarray,
                frontal_normal: np.ndarray) -> float:
    """Compute the Hip-Knee-Ankle (HKA) angle.

    The HKA is the angle at the knee between the femoral and tibial
    mechanical axes, measured in the frontal plane.

    Convention:
        - 180° = perfect neutral alignment
        - < 180° = varus (bow-legged)
        - > 180° = valgus (knock-kneed)

    Args:
        hjc, kjc, ajc: Joint center positions.
        frontal_normal: Unit normal of the frontal plane.

    Returns:
        HKA angle in degrees.
    """
    # Femoral mechanical axis: KJC to HJC direction (distal to proximal)
    v_femur = hjc - kjc
    # Tibial mechanical axis: KJC to AJC direction (proximal to distal)
    v_tibia = ajc - kjc

    # Project onto frontal plane
    v_femur_p = project_onto_plane(v_femur, frontal_normal)
    v_tibia_p = project_onto_plane(v_tibia, frontal_normal)

    # The HKA is the deviation from a straight line (180°).
    # Compute the signed angle between the femoral distal direction (-v_femur)
    # and the tibial distal direction (v_tibia) in the frontal plane.
    # For a straight leg, -v_femur and v_tibia are parallel → deviation = 0° → HKA = 180°.
    deviation = signed_angle_in_plane(-v_femur_p, v_tibia_p, frontal_normal)
    hka = 180.0 + deviation

    return hka


def compute_mldfa(hjc: np.ndarray, kjc: np.ndarray,
                  lfec: np.ndarray, mfec: np.ndarray,
                  frontal_normal: np.ndarray) -> float:
    """Compute the mechanical Lateral Distal Femoral Angle (mLDFA).

    Angle between the femoral mechanical axis and the distal femoral
    joint line, measured on the lateral side in the frontal plane.

    Normal: 85-90° (typically 87°).

    Args:
        hjc, kjc: Joint centers defining the femoral mechanical axis.
        lfec, mfec: Epicondyles defining the distal femoral joint line.
        frontal_normal: Frontal plane normal.

    Returns:
        mLDFA in degrees.
    """
    # Femoral mechanical axis: HJC → KJC (proximal to distal)
    v_mech = kjc - hjc
    # Distal femoral joint line: MFEC → LFEC (medial to lateral)
    v_joint = lfec - mfec

    # Project onto frontal plane
    v_mech_p = project_onto_plane(v_mech, frontal_normal)
    v_joint_p = project_onto_plane(v_joint, frontal_normal)

    # mLDFA is measured on the lateral side
    # It is the acute angle between the mechanical axis (pointing distally)
    # and the joint line (pointing laterally)
    angle = angle_between_vectors(v_mech_p, v_joint_p)

    # The lateral angle is the one we want
    # If the joint line points laterally relative to the mech axis,
    # the angle measured should be ~87°
    return angle


def compute_mpta(kjc: np.ndarray, ajc: np.ndarray,
                 ltp: np.ndarray, mtp: np.ndarray,
                 frontal_normal: np.ndarray) -> float:
    """Compute the Medial Proximal Tibial Angle (MPTA).

    Angle between the tibial mechanical axis and the proximal tibial
    joint line, measured on the medial side in the frontal plane.

    Normal: 85-90° (typically 87°).

    Args:
        kjc, ajc: Joint centers defining the tibial mechanical axis.
        ltp, mtp: Tibial plateau points defining the proximal tibial joint line.
        frontal_normal: Frontal plane normal.

    Returns:
        MPTA in degrees.
    """
    # Tibial mechanical axis: KJC → AJC (proximal to distal)
    v_mech = ajc - kjc
    # Proximal tibial joint line: LTP → MTP (lateral to medial)
    v_joint = mtp - ltp

    # Project onto frontal plane
    v_mech_p = project_onto_plane(v_mech, frontal_normal)
    v_joint_p = project_onto_plane(v_joint, frontal_normal)

    # MPTA is measured on the medial side
    angle = angle_between_vectors(v_mech_p, v_joint_p)

    return angle


def compute_jlca(lfec: np.ndarray, mfec: np.ndarray,
                 ltp: np.ndarray, mtp: np.ndarray,
                 frontal_normal: np.ndarray) -> float:
    """Compute the Joint Line Convergence Angle (JLCA).

    Angle between the distal femoral joint line and the proximal
    tibial joint line in the frontal plane.

    Normal: 0-2°.

    Args:
        lfec, mfec: Femoral epicondyles (distal femoral joint line).
        ltp, mtp: Tibial plateaus (proximal tibial joint line).
        frontal_normal: Frontal plane normal.

    Returns:
        JLCA in degrees.
    """
    # Distal femoral joint line: MFEC → LFEC
    v_femoral = lfec - mfec
    # Proximal tibial joint line: MTP → LTP
    v_tibial = ltp - mtp

    # Project onto frontal plane
    v_femoral_p = project_onto_plane(v_femoral, frontal_normal)
    v_tibial_p = project_onto_plane(v_tibial, frontal_normal)

    angle = angle_between_vectors(v_femoral_p, v_tibial_p)

    return angle


def compute_mldta(kjc: np.ndarray, ajc: np.ndarray,
                  lm: np.ndarray, mm: np.ndarray,
                  frontal_normal: np.ndarray) -> float:
    """Compute the mechanical Lateral Distal Tibial Angle (mLDTA).

    Angle between the tibial mechanical axis and the distal tibial
    joint line (defined by malleoli), measured on the lateral side.

    Normal: ~89°.

    Args:
        kjc, ajc: Joint centers defining the tibial mechanical axis.
        lm, mm: Malleoli defining the distal tibial joint line.
        frontal_normal: Frontal plane normal.

    Returns:
        mLDTA in degrees.
    """
    # Tibial mechanical axis: KJC → AJC (proximal to distal)
    v_mech = ajc - kjc
    # Distal tibial joint line: MM → LM (medial to lateral)
    v_joint = lm - mm

    # Project onto frontal plane
    v_mech_p = project_onto_plane(v_mech, frontal_normal)
    v_joint_p = project_onto_plane(v_joint, frontal_normal)

    angle = angle_between_vectors(v_mech_p, v_joint_p)

    return angle


def _compute_knee_offset(hjc: np.ndarray, kjc: np.ndarray,
                         ajc: np.ndarray) -> float:
    """Compute perpendicular distance from KJC to the mechanical axis line.

    Args:
        hjc, kjc, ajc: Joint centers.

    Returns:
        Offset distance in mm (positive = lateral).
    """
    from .utils import point_to_line_distance
    return point_to_line_distance(kjc, hjc, ajc)


def compute_all_angles(
    hjc: np.ndarray,
    kjc: np.ndarray,
    ajc: np.ndarray,
    lfec: np.ndarray,
    mfec: np.ndarray,
    ltp: np.ndarray,
    mtp: np.ndarray,
    lm: Optional[np.ndarray] = None,
    mm: Optional[np.ndarray] = None,
) -> HTOAngles:
    """Compute all HTO clinical angles from joint centers and landmarks.

    Args:
        hjc: (3,) Hip joint center.
        kjc: (3,) Knee joint center.
        ajc: (3,) Ankle joint center.
        lfec: (3,) Lateral femoral epicondyle.
        mfec: (3,) Medial femoral epicondyle.
        ltp: (3,) Lateral tibial plateau.
        mtp: (3,) Medial tibial plateau.
        lm: (3,) Lateral malleolus (optional).
        mm: (3,) Medial malleolus (optional).

    Returns:
        HTOAngles dataclass with all computed angles.
    """
    # Build frontal plane
    frontal_normal = _build_frontal_plane_normal(hjc, kjc, lfec, mfec)

    hka = compute_hka(hjc, kjc, ajc, frontal_normal)
    mldfa = compute_mldfa(hjc, kjc, lfec, mfec, frontal_normal)
    mpta = compute_mpta(kjc, ajc, ltp, mtp, frontal_normal)
    jlca = compute_jlca(lfec, mfec, ltp, mtp, frontal_normal)

    mldta = None
    if lm is not None and mm is not None:
        mldta = compute_mldta(kjc, ajc, lm, mm, frontal_normal)

    knee_offset = _compute_knee_offset(hjc, kjc, ajc)

    return HTOAngles(
        hka=hka,
        mldfa=mldfa,
        mpta=mpta,
        jlca=jlca,
        mldta=mldta,
        knee_offset_mm=knee_offset,
        frontal_normal=frontal_normal,
    )


def print_angles(angles: HTOAngles) -> None:
    """Print a formatted summary of HTO angles."""
    print("\n=== HTO Clinical Angles ===")
    print(f"  HKA:    {angles.hka:.1f}°  "
          f"({'neutral' if abs(angles.hka - 180) < 3 else 'varus' if angles.hka < 180 else 'valgus'})")
    print(f"  mLDFA:  {angles.mldfa:.1f}°  (normal: 85-90°)")
    print(f"  MPTA:   {angles.mpta:.1f}°  (normal: 85-90°)")
    print(f"  JLCA:   {angles.jlca:.1f}°  (normal: 0-2°)")
    if angles.mldta is not None:
        print(f"  mLDTA:  {angles.mldta:.1f}°  (normal: ~89°)")
    else:
        print(f"  mLDTA:  N/A (malleoli landmarks not available)")
    print(f"  Knee offset: {angles.knee_offset_mm:.2f} mm from mechanical axis")
    print("===========================\n")
