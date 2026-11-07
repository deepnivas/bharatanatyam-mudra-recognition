"""
src/features.py - Feature engineering, normalization, and visualization for hand landmarks.

Provides:
- Translation & scale normalization for 21 MediaPipe hand landmarks.
- Coordinate transformation (wrist origin, scale normalization).
- Modular feature vector extraction for both offline dataset processing and real-time webcam inference.
- Landmark drawing and visualization tools for 21-point hand skeletons.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union
import cv2
import numpy as np

# MediaPipe Hand Landmark Topology: 21 points (0..20)
# (0: WRIST, 1..4: THUMB, 5..8: INDEX, 9..12: MIDDLE, 13..16: RING, 17..20: PINKY)
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    # Palm base & knuckle loop
    (0, 1),
    (0, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (0, 17),
    # Thumb
    (1, 2),
    (2, 3),
    (3, 4),
    # Index finger
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle finger
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring finger
    (13, 14),
    (14, 15),
    (15, 16),
    # Pinky finger
    (17, 18),
    (18, 19),
    (19, 20),
)

LANDMARK_NAMES: Tuple[str, ...] = (
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
)


def get_landmark_column_names() -> List[str]:
    """Returns the list of 63 landmark column names:
    ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', ..., 'x21', 'y21', 'z21'].
    Landmarks are 1-indexed (1 to 21) where landmark 1 corresponds to wrist (index 0).
    """
    cols: List[str] = []
    for i in range(1, 22):
        cols.extend([f"x{i}", f"y{i}", f"z{i}"])
    return cols


def normalize_landmarks(
    landmarks: np.ndarray,
    scale_mode: str = "max_distance",
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalizes hand landmarks for translation and scale invariance.
    
    Transformation steps:
    1. Origin translation: Landmark 0 (wrist) becomes origin (0, 0, 0).
       p'_i = p_i - p_0  for i in 0..20.
    2. Scale normalization:
       - 'max_distance': Divides by maximum Euclidean distance from the wrist to any landmark.
         Scale factor = max_i ||p'_i||_2.
         Guarantees all points lie within the unit sphere [0, 1].
       - 'palm_size': Divides by Euclidean distance from wrist (0) to middle MCP (9).
    
    Args:
        landmarks: np.ndarray of shape (21, 3) or (63,).
        scale_mode: 'max_distance' (default) or 'palm_size'.
        eps: Small epsilon to prevent division by zero.
        
    Returns:
        np.ndarray of shape (21, 3) containing normalized coordinates.
    """
    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.ndim == 1 and pts.size == 63:
        pts = pts.reshape(21, 3)
    elif pts.shape != (21, 3):
        raise ValueError(f"Expected landmarks array of shape (21, 3) or (63,), got {pts.shape}")

    # 1. Translation normalization: wrist (index 0) becomes origin
    wrist = pts[0].copy()
    centered = pts - wrist

    # 2. Scale normalization
    if scale_mode == "max_distance":
        # Euclidean distance from wrist to all 21 points
        dists = np.linalg.norm(centered, axis=1)
        scale = float(np.max(dists))
    elif scale_mode == "palm_size":
        # Distance from wrist (0) to Middle MCP joint (9)
        scale = float(np.linalg.norm(centered[9]))
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}. Use 'max_distance' or 'palm_size'.")

    if scale > eps:
        normalized = centered / scale
    else:
        normalized = centered

    return normalized


def extract_features_from_mediapipe_result(
    result: Any,
    scale_mode: str = "max_distance",
    flatten: bool = True,
) -> Optional[np.ndarray]:
    """Extracts normalized feature vector from a MediaPipe HandLandmarkerResult.
    
    Designed for plug-and-play reuse in real-time webcam inference and dataset pipelines.
    
    Args:
        result: HandLandmarkerResult from MediaPipe Tasks API.
        scale_mode: Normalization scale mode ('max_distance' or 'palm_size').
        flatten: If True, returns (63,) 1D array; if False, returns (21, 3) array.
        
    Returns:
        np.ndarray if hand is detected, else None.
    """
    if result is None or not getattr(result, "hand_landmarks", None):
        return None

    if len(result.hand_landmarks) == 0:
        return None

    first_hand = result.hand_landmarks[0]
    if len(first_hand) != 21:
        return None

    # Extract raw normalized coordinates (0..1)
    pts = np.array([[lm.x, lm.y, lm.z] for lm in first_hand], dtype=np.float32)
    normalized = normalize_landmarks(pts, scale_mode=scale_mode)

    return normalized.flatten() if flatten else normalized


def draw_hand_landmarks_on_image(
    image_bgr: np.ndarray,
    landmarks: Union[Any, np.ndarray],
    draw_connections: bool = True,
    draw_points: bool = True,
    connection_color: Tuple[int, int, int] = (255, 200, 0),  # Cyan / Bright Blue-Green in BGR
    point_color: Tuple[int, int, int] = (0, 100, 255),       # Vibrant Orange-Red
    point_radius: int = 4,
    line_thickness: int = 2,
) -> np.ndarray:
    """Overlays the 21-point skeleton and connections onto an image.
    
    Args:
        image_bgr: Target image in BGR format.
        landmarks: Either a list of MediaPipe NormalizedLandmark objects or (21, 3)/(21, 2) numpy array.
        draw_connections: Whether to draw bone connection lines.
        draw_points: Whether to draw keypoint circles.
        connection_color: BGR line color.
        point_color: BGR joint circle color.
        point_radius: Joint circle radius in pixels.
        line_thickness: Connection line thickness in pixels.
        
    Returns:
        Annotated BGR image copy.
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]

    # Convert landmarks to pixel coordinates (x_px, y_px)
    coords: List[Tuple[int, int]] = []

    if isinstance(landmarks, np.ndarray):
        pts = landmarks.reshape(21, -1)
        for i in range(21):
            px = int(np.clip(pts[i, 0] * w, 0, w - 1))
            py = int(np.clip(pts[i, 1] * h, 0, h - 1))
            coords.append((px, py))
    elif isinstance(landmarks, (list, tuple)):
        # List of MediaPipe NormalizedLandmark objects or objects with .x, .y
        for lm in landmarks:
            if hasattr(lm, "x") and hasattr(lm, "y"):
                px = int(np.clip(lm.x * w, 0, w - 1))
                py = int(np.clip(lm.y * h, 0, h - 1))
            elif isinstance(lm, (tuple, list)):
                px = int(np.clip(lm[0] * w, 0, w - 1))
                py = int(np.clip(lm[1] * h, 0, h - 1))
            else:
                continue
            coords.append((px, py))

    if len(coords) < 21:
        return out

    # 1. Draw connections
    if draw_connections:
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = coords[start_idx]
            pt2 = coords[end_idx]
            cv2.line(out, pt1, pt2, connection_color, line_thickness, cv2.LINE_AA)

    # 2. Draw landmark joints
    if draw_points:
        for i, (px, py) in enumerate(coords):
            # Wrist gets slightly larger marker
            r = point_radius + 2 if i == 0 else point_radius
            # Outer colored ring
            cv2.circle(out, (px, py), r, point_color, -1, cv2.LINE_AA)
            # Inner white center for sharp visibility
            cv2.circle(out, (px, py), max(1, r - 2), (255, 255, 255), -1, cv2.LINE_AA)

    return out


# ==============================================================================
# Milestone V4: Joint-Angle and Geometric Feature Engineering
# ==============================================================================

JOINT_ANGLE_TRIPLETS: Tuple[Tuple[str, int, int, int], ...] = (
    # Thumb: (Wrist-CMC-MCP, CMC-MCP-IP, MCP-IP-TIP)
    ("thumb_cmc_mcp", 0, 1, 2),
    ("thumb_mcp_ip", 1, 2, 3),
    ("thumb_ip_tip", 2, 3, 4),
    # Index: (Wrist-MCP-PIP, MCP-PIP-DIP, PIP-DIP-TIP)
    ("index_wrist_mcp", 0, 5, 6),
    ("index_mcp_pip", 5, 6, 7),
    ("index_pip_dip", 6, 7, 8),
    # Middle: (Wrist-MCP-PIP, MCP-PIP-DIP, PIP-DIP-TIP)
    ("middle_wrist_mcp", 0, 9, 10),
    ("middle_mcp_pip", 9, 10, 11),
    ("middle_pip_dip", 10, 11, 12),
    # Ring: (Wrist-MCP-PIP, MCP-PIP-DIP, PIP-DIP-TIP)
    ("ring_wrist_mcp", 0, 13, 14),
    ("ring_mcp_pip", 13, 14, 15),
    ("ring_pip_dip", 14, 15, 16),
    # Pinky: (Wrist-MCP-PIP, MCP-PIP-DIP, PIP-DIP-TIP)
    ("pinky_wrist_mcp", 0, 17, 18),
    ("pinky_mcp_pip", 17, 18, 19),
    ("pinky_pip_dip", 18, 19, 20),
)

FINGERTIP_INDICES: Tuple[int, ...] = (4, 8, 12, 16, 20)  # Thumb, Index, Middle, Ring, Pinky


def compute_angle_between_points(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-7) -> float:
    """Computes angle at joint B formed by vectors BA and BC, normalized to [0, 1] by dividing by pi."""
    u = a - b
    v = c - b
    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))
    if norm_u < eps or norm_v < eps:
        return 0.0
    cosine = float(np.dot(u, v) / (norm_u * norm_v))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.arccos(cosine) / np.pi)


def compute_joint_angles(landmarks_21_3: np.ndarray) -> np.ndarray:
    """Computes 15 normalized finger joint angles in [0, 1] across all five fingers."""
    pts = np.asarray(landmarks_21_3, dtype=np.float32).reshape(21, 3)
    angles: List[float] = []
    for _, a, b, c in JOINT_ANGLE_TRIPLETS:
        angles.append(compute_angle_between_points(pts[a], pts[b], pts[c]))
    return np.array(angles, dtype=np.float32)


def compute_palm_and_fingertip_distances(landmarks_21_3: np.ndarray) -> np.ndarray:
    """Computes 20 normalized Euclidean distances:
    - 5 distances from fingertips to Palm Center
    - 5 distances from fingertips to Wrist (origin)
    - 10 pairwise inter-fingertip distances
    """
    pts = np.asarray(landmarks_21_3, dtype=np.float32).reshape(21, 3)
    palm_center = (pts[0] + pts[5] + pts[9] + pts[17]) / 4.0
    tips = FINGERTIP_INDICES

    dists: List[float] = []
    # 1. 5 Fingertip to Palm Center
    for t in tips:
        dists.append(float(np.linalg.norm(pts[t] - palm_center)))

    # 2. 5 Fingertip to Wrist (origin)
    for t in tips:
        dists.append(float(np.linalg.norm(pts[t])))

    # 3. 10 Pairwise inter-fingertip distances
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            dists.append(float(np.linalg.norm(pts[tips[i]] - pts[tips[j]])))

    return np.array(dists, dtype=np.float32)


def compute_pinch_and_loop_features(landmarks_21_3: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Computes 7 geometric features targeting pinch gestures (Bhramara vs Katakamukha):
    - thumb-index fingertip distance
    - thumb-middle fingertip distance
    - index-middle fingertip distance
    - pinch triangle perimeter
    - pinch triangle area (cross product)
    - thumb-index to thumb-middle distance ratio
    - thumb-index orientation angle
    """
    pts = np.asarray(landmarks_21_3, dtype=np.float32).reshape(21, 3)
    p4 = pts[4]   # THUMB_TIP
    p8 = pts[8]   # INDEX_TIP
    p12 = pts[12] # MIDDLE_TIP

    d4_8 = float(np.linalg.norm(p4 - p8))
    d4_12 = float(np.linalg.norm(p4 - p12))
    d8_12 = float(np.linalg.norm(p8 - p12))
    perimeter = d4_8 + d4_12 + d8_12

    # Triangle area formed by thumb, index, middle fingertips
    cross_prod = np.cross(p8 - p4, p12 - p4)
    area = 0.5 * float(np.linalg.norm(cross_prod))

    # Ratio of thumb-index distance to thumb-middle distance
    ratio = float(d4_8 / (d4_12 + eps))

    # Vector orientation between thumb distal segment and index distal segment
    u_th = p4 - pts[3]
    u_idx = p8 - pts[7]
    norm_th = float(np.linalg.norm(u_th))
    norm_idx = float(np.linalg.norm(u_idx))
    if norm_th > eps and norm_idx > eps:
        cos_ang = float(np.clip(np.dot(u_th, u_idx) / (norm_th * norm_idx), -1.0, 1.0))
        orient_angle = float(np.arccos(cos_ang) / np.pi)
    else:
        orient_angle = 0.0

    return np.array([d4_8, d4_12, d8_12, perimeter, area, ratio, orient_angle], dtype=np.float32)


def compute_flexion_and_dispersion_features(landmarks_21_3: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Computes 7 flexion and dispersion features targeting curled fingers (Kangula vs Mukula):
    - index finger extension ratio (tip-to-mcp / segment lengths)
    - middle finger extension ratio
    - ring finger extension ratio
    - pinky finger extension ratio
    - ring-to-pinky extension distance ratio
    - max pairwise fingertip dispersion
    - mean pairwise fingertip dispersion
    """
    pts = np.asarray(landmarks_21_3, dtype=np.float32).reshape(21, 3)

    # Extension ratios for 4 fingers: Index (5..8), Middle (9..12), Ring (13..16), Pinky (17..20)
    ext_ratios: List[float] = []
    for mcp, pip, dip, tip in [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]:
        span = float(np.linalg.norm(pts[tip] - pts[mcp]))
        segs = (
            float(np.linalg.norm(pts[pip] - pts[mcp]))
            + float(np.linalg.norm(pts[dip] - pts[pip]))
            + float(np.linalg.norm(pts[tip] - pts[dip]))
        )
        ext_ratios.append(float(span / (segs + eps)))

    # Ratio of ring fingertip distance to pinky fingertip distance from wrist
    ring_dist = float(np.linalg.norm(pts[16]))
    pinky_dist = float(np.linalg.norm(pts[20]))
    ring_to_pinky = float(ring_dist / (pinky_dist + eps))

    # Pairwise fingertip dispersion
    tips = FINGERTIP_INDICES
    pair_dists: List[float] = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            pair_dists.append(float(np.linalg.norm(pts[tips[i]] - pts[tips[j]])))

    max_disp = float(np.max(pair_dists))
    mean_disp = float(np.mean(pair_dists))

    return np.array(ext_ratios + [ring_to_pinky, max_disp, mean_disp], dtype=np.float32)


def get_v4_geometric_feature_names() -> List[str]:
    """Returns the names of all 49 engineered geometric features."""
    names: List[str] = []
    # 1. 15 joint angles
    for prefix, _, _, _ in JOINT_ANGLE_TRIPLETS:
        names.append(f"angle_{prefix}")
    # 2. 5 fingertip to palm center
    for t_name in ("thumb", "index", "middle", "ring", "pinky"):
        names.append(f"dist_{t_name}_to_palm_center")
    # 3. 5 fingertip to wrist
    for t_name in ("thumb", "index", "middle", "ring", "pinky"):
        names.append(f"dist_{t_name}_to_wrist")
    # 4. 10 pairwise tip distances
    tip_names = ("th", "idx", "mid", "rng", "pnk")
    for i in range(len(tip_names)):
        for j in range(i + 1, len(tip_names)):
            names.append(f"dist_tip_{tip_names[i]}_{tip_names[j]}")
    # 5. 7 pinch & loop features
    names.extend([
        "pinch_d4_8", "pinch_d4_12", "pinch_d8_12",
        "pinch_triangle_perimeter", "pinch_triangle_area",
        "pinch_d4_8_to_d4_12_ratio", "pinch_thumb_index_angle"
    ])
    # 6. 7 flexion & dispersion features
    names.extend([
        "flexion_index_ratio", "flexion_middle_ratio",
        "flexion_ring_ratio", "flexion_pinky_ratio",
        "ring_to_pinky_dist_ratio",
        "fingertip_dispersion_max", "fingertip_dispersion_mean"
    ])
    return names


def extract_v4_features(
    landmarks: np.ndarray,
    feature_config: str = "full",
) -> np.ndarray:
    """Extracts landmark feature representation according to experimental configuration:
    - 'control': 63 raw normalized landmark coordinates (V3 baseline)
    - 'angles': 63 coordinates + 15 joint angles = 78 features
    - 'angles_distances': 63 coordinates + 15 angles + 20 distances = 98 features
    - 'full': 63 coordinates + 49 geometric features = 112 features
    """
    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.ndim == 1 and pts.size == 63:
        pts = pts.reshape(21, 3)
    elif pts.shape != (21, 3):
        raise ValueError(f"Expected shape (21, 3) or (63,), got {pts.shape}")

    raw_63 = pts.flatten()

    if feature_config == "control":
        return raw_63

    angles = compute_joint_angles(pts)
    if feature_config == "angles":
        return np.concatenate([raw_63, angles])

    distances = compute_palm_and_fingertip_distances(pts)
    if feature_config == "angles_distances":
        return np.concatenate([raw_63, angles, distances])

    # Full suite: angles, distances, pinch/loop, flexion/dispersion
    pinch = compute_pinch_and_loop_features(pts)
    flexion = compute_flexion_and_dispersion_features(pts)
    return np.concatenate([raw_63, angles, distances, pinch, flexion])

