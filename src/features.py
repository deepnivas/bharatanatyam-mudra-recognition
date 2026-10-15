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
