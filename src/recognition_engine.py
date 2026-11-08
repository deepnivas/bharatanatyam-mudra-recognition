#!/usr/bin/env python3
"""
src/recognition_engine.py - Unified Multi-Model Recognition & Comparison Engine (Parts 3 & 4).

Features:
- Encapsulates MediaPipe HandLandmarker Tasks API (in IMAGE mode).
- Supports dynamic switching between all 5 model generations (V1, V2, V3, V4, V5).
- Automatically routes the correct feature representation:
    * V1 (Baseline SVM): 63 raw normalized coordinates
    * V2 (Augmented SVM): 63 raw normalized coordinates
    * V3 (Cascade SVM): 63 raw normalized coordinates
    * V4 (Geometric SVM): 98 features (63 coords + 15 angles + 20 distances)
    * V5 (MLP Neural Net): 98 features (63 coords + 15 angles + 20 distances)
- Computes palm normal orientation and 3D direction vectors.
- Applies temporal smoothing (rolling majority vote) and confidence gating.
- Compares predicted mudra against the user-selected expected mudra.
"""

from __future__ import annotations

from collections import Counter, deque
import logging
from pathlib import Path
import sys
import time
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import cv2
import joblib
import numpy as np

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from audit_dataset import MediaPipeHandDetector
from features import (
    draw_hand_landmarks_on_image,
    extract_v4_features,
    normalize_landmarks,
)

logger = logging.getLogger("RecognitionEngine")


class MudraRecognitionResult:
    """Encapsulates the complete result of a recognition cycle."""

    def __init__(
        self,
        hand_detected: bool,
        predicted_mudra: str = "",
        confidence: float = 0.0,
        raw_confidence: float = 0.0,
        landmarks_21_3: Optional[np.ndarray] = None,
        normalized_63: Optional[np.ndarray] = None,
        features_98: Optional[np.ndarray] = None,
        palm_orientation: Optional[Dict[str, Any]] = None,
        active_model_name: str = "V1",
        expected_mudra: str = "",
        is_expected_match: bool = False,
        smoothed: bool = True,
    ) -> None:
        self.hand_detected = hand_detected
        self.predicted_mudra = predicted_mudra
        self.confidence = confidence
        self.raw_confidence = raw_confidence
        self.landmarks_21_3 = landmarks_21_3
        self.normalized_63 = normalized_63
        self.features_98 = features_98
        self.palm_orientation = palm_orientation or {}
        self.active_model_name = active_model_name
        self.expected_mudra = expected_mudra
        self.is_expected_match = is_expected_match
        self.smoothed = smoothed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hand_detected": self.hand_detected,
            "predicted_mudra": self.predicted_mudra,
            "confidence": round(self.confidence, 4),
            "raw_confidence": round(self.raw_confidence, 4),
            "active_model_name": self.active_model_name,
            "expected_mudra": self.expected_mudra,
            "is_expected_match": self.is_expected_match,
            "palm_orientation": self.palm_orientation,
        }


class MudraRecognitionEngine:
    """Unified Recognition Engine capable of multi-model inference and pose comparison."""

    AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
        "V1": {
            "name": "V1 Baseline SVM",
            "path": "models/final_svm_model.joblib",
            "feature_config": "control",
            "feature_dim": 63,
        },
        "V2": {
            "name": "V2 Augmented Balanced SVM",
            "path": "models/augmented_svm_balanced.joblib",
            "feature_config": "control",
            "feature_dim": 63,
        },
        "V3": {
            "name": "V3 Cascade Unweighted SVM",
            "path": "models/v3_svm_unweighted.joblib",
            "feature_config": "control",
            "feature_dim": 63,
        },
        "V4": {
            "name": "V4 Geometric SVM",
            "path": "models/v4_svm_model.joblib",
            "feature_config": "angles_distances",
            "feature_dim": 98,
        },
        "V5": {
            "name": "V5 Selected MLP Neural Network",
            "path": "models/v5_experiment/v5_selected_mlp_model.joblib",
            "feature_config": "angles_distances",
            "feature_dim": 98,
        },
    }

    def __init__(
        self,
        default_model: str = "V4",
        min_detection_confidence: float = 0.5,
        smoothing_window: int = 7,
        confidence_threshold: float = 0.35,
        hand_landmarker_path: Path = Path("models/hand_landmarker.task"),
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.confidence_threshold = confidence_threshold
        self.hand_landmarker_path = hand_landmarker_path
        self.active_model_key = default_model if default_model in self.AVAILABLE_MODELS else "V4"

        # Initialize detector
        self.detector = MediaPipeHandDetector(
            model_path=self.hand_landmarker_path,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
        )

        # Cache of loaded models
        self._loaded_models: Dict[str, Any] = {}
        self._load_model(self.active_model_key)

        # Temporal smoother
        self.window_size = smoothing_window
        self.history: Deque[Tuple[str, float]] = deque(maxlen=smoothing_window)

    def _load_model(self, model_key: str) -> Any:
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(self.AVAILABLE_MODELS.keys())}")

        if model_key not in self._loaded_models:
            cfg = self.AVAILABLE_MODELS[model_key]
            m_path = Path(cfg["path"])
            if not m_path.is_file():
                # Fallback to alternative paths if needed
                alt_path = Path("models/final_svm_model.joblib")
                logger.warning(f"Model file {m_path} not found. Falling back to {alt_path}")
                m_path = alt_path
            logger.info(f"Loading {cfg['name']} from {m_path}...")
            self._loaded_models[model_key] = joblib.load(m_path)

        return self._loaded_models[model_key]

    def set_active_model(self, model_key: str) -> None:
        """Dynamically switches the active classification model."""
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model key: {model_key}")
        if model_key != self.active_model_key:
            self._load_model(model_key)
            self.active_model_key = model_key
            self.clear_smoothing()
            logger.info(f"Switched active model to: {self.AVAILABLE_MODELS[model_key]['name']}")

    def clear_smoothing(self) -> None:
        """Clears the temporal smoothing buffer."""
        self.history.clear()

    @staticmethod
    def compute_palm_orientation(pts_21_3: np.ndarray) -> Dict[str, Any]:
        """Calculates 3D palm normal vector and wrist-to-middle direction."""
        wrist = pts_21_3[0]
        index_mcp = pts_21_3[5]
        pinky_mcp = pts_21_3[17]
        middle_mcp = pts_21_3[9]

        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist

        # Normal vector perpendicular to palm plane
        normal = np.cross(v1, v2)
        norm_mag = float(np.linalg.norm(normal))
        if norm_mag > 1e-6:
            normal = normal / norm_mag
        else:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Direction vector from wrist to middle finger base
        hand_dir = middle_mcp - wrist
        dir_mag = float(np.linalg.norm(hand_dir))
        if dir_mag > 1e-6:
            hand_dir = hand_dir / dir_mag
        else:
            hand_dir = np.array([0.0, -1.0, 0.0], dtype=np.float32)

        # Classify primary facing direction
        # In MediaPipe camera coords: +X is right, +Y is down, +Z is towards camera / depth
        z_facing = "Facing Camera" if normal[2] < 0 else "Facing Away / Rear"

        return {
            "palm_normal": [round(float(x), 4) for x in normal],
            "hand_direction": [round(float(x), 4) for x in hand_dir],
            "primary_orientation": z_facing,
        }

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        expected_mudra: str = "",
    ) -> Tuple[MudraRecognitionResult, np.ndarray]:
        """Processes a single BGR camera frame.

        Returns:
            Tuple of (MudraRecognitionResult, annotated_display_frame)
        """
        display_frame = bgr_frame.copy()
        detected, result, det_err = self.detector.detect(bgr_frame)

        if not detected or result is None or not result.hand_landmarks:
            self.clear_smoothing()
            return MudraRecognitionResult(
                hand_detected=False,
                active_model_name=self.AVAILABLE_MODELS[self.active_model_key]["name"],
                expected_mudra=expected_mudra,
            ), display_frame

        first_hand = result.hand_landmarks[0]
        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in first_hand], dtype=np.float32)

        # Normalize landmarks: translation (wrist=0) + max_distance scale
        norm_pts = normalize_landmarks(raw_pts, scale_mode="max_distance")
        norm_63 = norm_pts.flatten()

        # Extract features according to active model specification
        active_cfg = self.AVAILABLE_MODELS[self.active_model_key]
        feature_cfg_name = active_cfg["feature_config"]

        if feature_cfg_name == "control":
            input_vector = norm_63.reshape(1, -1)
            feat_98 = extract_v4_features(norm_pts, feature_config="angles_distances")
        else:
            feat_98 = extract_v4_features(norm_pts, feature_config="angles_distances")
            input_vector = feat_98.reshape(1, -1)

        # Run model prediction
        model = self._load_model(self.active_model_key)
        raw_pred = str(model.predict(input_vector)[0])
        probabilities = model.predict_proba(input_vector)[0]
        raw_conf = float(np.max(probabilities))

        # Temporal smoothing
        self.history.append((raw_pred, raw_conf))
        classes = [p for p, _ in self.history]
        counts = Counter(classes)
        majority_pred, count = counts.most_common(1)[0]
        majority_confs = [c for p, c in self.history if p == majority_pred]
        smoothed_conf = float(np.mean(majority_confs))

        # Apply confidence gating
        final_pred = majority_pred if smoothed_conf >= self.confidence_threshold else "Uncertain"

        # Compute palm orientation
        palm_info = self.compute_palm_orientation(norm_pts)

        # Expected mudra comparison
        is_match = False
        if expected_mudra and final_pred != "Uncertain":
            # Direct match or prefix matching (e.g., '1-Pataka' matches 'Pataka')
            is_match = (
                final_pred.lower() == expected_mudra.lower()
                or final_pred.split("-")[-1].lower() == expected_mudra.split("-")[-1].lower()
            )

        # Overlay 21-joint skeleton
        display_frame = draw_hand_landmarks_on_image(
            display_frame,
            first_hand,
            draw_connections=True,
            draw_points=True,
            connection_color=(255, 215, 0),  # Cyan
            point_color=(0, 100, 255),       # Vibrant orange
            point_radius=4,
            line_thickness=2,
        )

        res = MudraRecognitionResult(
            hand_detected=True,
            predicted_mudra=final_pred,
            confidence=smoothed_conf,
            raw_confidence=raw_conf,
            landmarks_21_3=norm_pts,
            normalized_63=norm_63,
            features_98=feat_98,
            palm_orientation=palm_info,
            active_model_name=active_cfg["name"],
            expected_mudra=expected_mudra,
            is_expected_match=is_match,
            smoothed=True,
        )

        return res, display_frame

    def close(self) -> None:
        """Closes detector resources."""
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()
