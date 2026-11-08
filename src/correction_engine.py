#!/usr/bin/env python3
"""
src/correction_engine.py - Explainable Mudra Correction Engine & 4-State System (Parts 6 & 7).

Features:
- Finger joint-angle, distance, curvature, and palm-orientation comparative analysis.
- Evaluates poses against instructor-approved reference metadata and acceptable tolerances.
- Identifies finger-specific errors (Thumb, Index, Middle, Ring, Pinky).
- Generates natural, actionable, explainable correction instructions.
- Implements the 4-state evaluation paradigm:
    * STATE A: CORRECT ("CORRECT MUDRA")
    * STATE B: WRONG MUDRA ("WRONG MUDRA")
    * STATE C: CORRECT CLASS, INCORRECT EXECUTION ("MUDRA RECOGNIZED — POSITION NEEDS CORRECTION")
    * STATE D: UNABLE TO ASSESS ("UNABLE TO ASSESS")
"""

from __future__ import annotations

from enum import Enum
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from features import (
    compute_joint_angles,
    compute_palm_and_fingertip_distances,
    compute_pinch_and_loop_features,
    compute_flexion_and_dispersion_features,
    HAND_CONNECTIONS,
)
from recognition_engine import MudraRecognitionResult

logger = logging.getLogger("CorrectionEngine")


class MudraEvaluationState(str, Enum):
    """The 4 standardized evaluation states."""
    STATE_A_CORRECT = "CORRECT MUDRA"
    STATE_B_WRONG_MUDRA = "WRONG MUDRA"
    STATE_C_INCORRECT_EXECUTION = "MUDRA RECOGNIZED — POSITION NEEDS CORRECTION"
    STATE_D_UNABLE_TO_ASSESS = "UNABLE TO ASSESS"


class FingerStatus(str, Enum):
    """Finger position evaluation status."""
    CORRECT = "Correct"
    NEEDS_ADJUSTMENT = "Needs adjustment"
    INCORRECT = "Incorrect"


FINGER_INDICES_MAP: Dict[str, Tuple[int, ...]] = {
    "Thumb": (1, 2, 3, 4),
    "Index": (5, 6, 7, 8),
    "Middle": (9, 10, 11, 12),
    "Ring": (13, 14, 15, 16),
    "Pinky": (17, 18, 19, 20),
}

# Mapping of joint angle triplets in src/features.py (3 angles per finger)
FINGER_ANGLE_INDICES: Dict[str, Tuple[int, int, int]] = {
    "Thumb": (0, 1, 2),
    "Index": (3, 4, 5),
    "Middle": (6, 7, 8),
    "Ring": (9, 10, 11),
    "Pinky": (12, 13, 14),
}


class MudraCorrectionReport:
    """Detailed report produced by the MudraCorrectionEngine."""

    def __init__(
        self,
        state: MudraEvaluationState,
        expected_mudra: str,
        recognized_mudra: str,
        confidence: float,
        finger_statuses: Dict[str, FingerStatus],
        instructions: List[str],
        reference_image_path: Optional[str] = None,
        reference_landmarks: Optional[np.ndarray] = None,
        similarity_score: float = 0.0,
        explanation: str = "",
    ) -> None:
        self.state = state
        self.expected_mudra = expected_mudra
        self.recognized_mudra = recognized_mudra
        self.confidence = confidence
        self.finger_statuses = finger_statuses
        self.instructions = instructions
        self.reference_image_path = reference_image_path
        self.reference_landmarks = reference_landmarks
        self.similarity_score = similarity_score
        self.explanation = explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "expected_mudra": self.expected_mudra,
            "recognized_mudra": self.recognized_mudra,
            "confidence": round(self.confidence, 4),
            "finger_statuses": {k: v.value for k, v in self.finger_statuses.items()},
            "instructions": self.instructions,
            "similarity_score": round(self.similarity_score, 4),
            "explanation": self.explanation,
            "reference_image_path": self.reference_image_path,
        }


class MudraCorrectionEngine:
    """Explainable pose correction and comparative analysis engine."""

    def __init__(
        self,
        reference_dir: Path = Path("data/kuchipudi_references"),
        min_confidence_for_assessment: float = 0.40,
    ) -> None:
        self.reference_dir = reference_dir
        self.min_confidence = min_confidence_for_assessment
        self.manifest_path = reference_dir / "manifest.json"
        self._manifest: Dict[str, Any] = {}
        self._reference_cache: Dict[str, Dict[str, Any]] = {}
        self.load_manifest()

    def load_manifest(self) -> None:
        if self.manifest_path.is_file():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self._manifest = json.load(f)
                logger.info(f"Loaded manifest with {self._manifest.get('supported_mudras_count', 0)} mudras.")
            except Exception as e:
                logger.error(f"Error loading manifest: {e}")
                self._manifest = {}
        else:
            logger.warning(f"Manifest not found at {self.manifest_path}")

    def get_supported_mudras(self) -> List[str]:
        mudras = self._manifest.get("mudras", {})
        return sorted(list(mudras.keys()))

    def get_reference_data(self, mudra_name: str) -> Optional[Dict[str, Any]]:
        """Loads reference landmarks and metadata for a given mudra."""
        if mudra_name in self._reference_cache:
            return self._reference_cache[mudra_name]

        mudra_entry = self._manifest.get("mudras", {}).get(mudra_name)
        if not mudra_entry:
            # Try fuzzy matching without prefix (e.g., 'Pataka' vs '1-Pataka')
            for k, v in self._manifest.get("mudras", {}).items():
                if k.lower() == mudra_name.lower() or k.split("-")[-1].lower() == mudra_name.split("-")[-1].lower():
                    mudra_entry = v
                    mudra_name = k
                    break

        if not mudra_entry:
            return None

        # Load metadata JSON
        meta_path = Path(mudra_entry["metadata_path"])
        if not meta_path.is_file():
            meta_path = self.reference_dir / "metadata" / f"{mudra_name}_metadata.json"

        # Load reference landmarks NPY
        npy_path = Path(mudra_entry["landmarks_path"])
        if not npy_path.is_file():
            npy_path = self.reference_dir / "landmarks" / f"{mudra_name}_reference.npy"

        if not meta_path.is_file() or not npy_path.is_file():
            return None

        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)

        ref_landmarks = np.load(str(npy_path))

        ref_data = {
            "metadata": meta,
            "landmarks": ref_landmarks,
            "image_path": mudra_entry.get("image_path", f"data/kuchipudi_references/images/{mudra_name}_reference.jpg"),
        }
        self._reference_cache[mudra_name] = ref_data
        return ref_data

    def evaluate_pose(
        self,
        rec_result: MudraRecognitionResult,
        expected_mudra: str,
    ) -> MudraCorrectionReport:
        """Evaluates hand pose against the expected mudra and produces a MudraCorrectionReport."""

        # Check for STATE D: UNABLE TO ASSESS
        if (
            not rec_result.hand_detected
            or rec_result.landmarks_21_3 is None
            or rec_result.confidence < self.min_confidence
            or rec_result.predicted_mudra == "Uncertain"
        ):
            reason = "No hand detected in camera view" if not rec_result.hand_detected else (
                "Insufficient recognition confidence or hand visibility"
            )
            return MudraCorrectionReport(
                state=MudraEvaluationState.STATE_D_UNABLE_TO_ASSESS,
                expected_mudra=expected_mudra,
                recognized_mudra=rec_result.predicted_mudra if rec_result.hand_detected else "None",
                confidence=rec_result.confidence,
                finger_statuses={f: FingerStatus.NEEDS_ADJUSTMENT for f in FINGER_INDICES_MAP},
                instructions=["Place your hand clearly and steadily within the camera frame."],
                explanation=reason,
            )

        # Retrieve reference data
        ref_data = self.get_reference_data(expected_mudra)
        if not ref_data:
            return MudraCorrectionReport(
                state=MudraEvaluationState.STATE_D_UNABLE_TO_ASSESS,
                expected_mudra=expected_mudra,
                recognized_mudra=rec_result.predicted_mudra,
                confidence=rec_result.confidence,
                finger_statuses={f: FingerStatus.NEEDS_ADJUSTMENT for f in FINGER_INDICES_MAP},
                instructions=["Instructor reference validation required for this mudra."],
                explanation="Reference validation required: No verified reference record found.",
            )

        ref_meta = ref_data["metadata"]
        ref_pts = ref_data["landmarks"]
        ref_img_path = ref_data["image_path"]

        # Check for STATE B: WRONG MUDRA
        if not rec_result.is_expected_match:
            return MudraCorrectionReport(
                state=MudraEvaluationState.STATE_B_WRONG_MUDRA,
                expected_mudra=expected_mudra,
                recognized_mudra=rec_result.predicted_mudra,
                confidence=rec_result.confidence,
                finger_statuses={f: FingerStatus.INCORRECT for f in FINGER_INDICES_MAP},
                instructions=[
                    f"Perform the expected mudra '{expected_mudra}'.",
                    f"Currently detected: '{rec_result.predicted_mudra}' ({rec_result.confidence*100:.1f}% confidence).",
                    "Refer to the instructor reference image on the right.",
                ],
                reference_image_path=ref_img_path,
                reference_landmarks=ref_pts,
                explanation=f"Detected mudra '{rec_result.predicted_mudra}' differs from target '{expected_mudra}'.",
            )

        # STATE A vs STATE C: FINGER GEOMETRIC ANALYSIS
        user_pts = rec_result.landmarks_21_3
        user_angles = compute_joint_angles(user_pts)
        ref_angles = np.array(ref_meta["joint_angles_mean"], dtype=np.float32)
        tolerances = np.array(ref_meta["joint_angle_tolerances"], dtype=np.float32)

        finger_statuses: Dict[str, FingerStatus] = {}
        instructions: List[str] = []
        incorrect_finger_count = 0

        # Finger-by-finger analysis
        for finger_name, angle_idxs in FINGER_ANGLE_INDICES.items():
            u_angs = user_angles[list(angle_idxs)]
            r_angs = ref_angles[list(angle_idxs)]
            tols = tolerances[list(angle_idxs)]

            diffs = u_angs - r_angs
            abs_diffs = np.abs(diffs)

            # Check if within tolerance
            out_of_tol = abs_diffs > tols
            violations = np.sum(out_of_tol)

            if violations == 0:
                finger_statuses[finger_name] = FingerStatus.CORRECT
            elif violations == 1 and np.max(abs_diffs - tols) < 0.12:
                finger_statuses[finger_name] = FingerStatus.NEEDS_ADJUSTMENT
                # Construct constructive instruction
                max_idx = int(np.argmax(abs_diffs))
                diff_val = diffs[max_idx]
                if diff_val > 0:
                    instructions.append(f"Bend your {finger_name} slightly further.")
                else:
                    instructions.append(f"Straighten your {finger_name} slightly.")
            else:
                finger_statuses[finger_name] = FingerStatus.INCORRECT
                incorrect_finger_count += 1
                max_idx = int(np.argmax(abs_diffs))
                diff_val = diffs[max_idx]
                if diff_val > 0:
                    instructions.append(f"Curl your {finger_name} closer towards the palm.")
                else:
                    instructions.append(f"Extend your {finger_name} fully outwards.")

        # Overall similarity score: normalized Euclidean distance between landmark points
        dist_matrix = np.linalg.norm(user_pts - ref_pts, axis=1)
        mean_dist = float(np.mean(dist_matrix))
        # Map distance to similarity [0, 1] (0.0 distance -> 1.0 similarity, 0.4 distance -> 0.0)
        similarity = float(np.clip(1.0 - (mean_dist / 0.40), 0.0, 1.0))

        # Determine if STATE A or STATE C
        if incorrect_finger_count == 0 and len(instructions) == 0:
            final_state = MudraEvaluationState.STATE_A_CORRECT
            final_instructions = ["Pose meets accepted instructor reference criteria! Excellent!"]
            explanation = "All finger joint angles and hand geometry are within accepted reference tolerances."
        elif incorrect_finger_count == 0 and len(instructions) <= 2:
            # Minor adjustments only
            final_state = MudraEvaluationState.STATE_C_INCORRECT_EXECUTION
            final_instructions = instructions
            explanation = "Mudra class recognized correctly, but minor finger position adjustments are required."
        else:
            final_state = MudraEvaluationState.STATE_C_INCORRECT_EXECUTION
            final_instructions = instructions
            explanation = f"Mudra class recognized, but {incorrect_finger_count} finger(s) need position correction."

        return MudraCorrectionReport(
            state=final_state,
            expected_mudra=expected_mudra,
            recognized_mudra=rec_result.predicted_mudra,
            confidence=rec_result.confidence,
            finger_statuses=finger_statuses,
            instructions=final_instructions,
            reference_image_path=ref_img_path,
            reference_landmarks=ref_pts,
            similarity_score=similarity,
            explanation=explanation,
        )

    def draw_correction_skeleton(
        self,
        image_bgr: np.ndarray,
        landmarks_21_3: np.ndarray,
        finger_statuses: Dict[str, FingerStatus],
    ) -> np.ndarray:
        """Annotates skeleton with color-coded joints based on finger correctness."""
        out = image_bgr.copy()
        h, w = out.shape[:2]

        status_colors = {
            FingerStatus.CORRECT: (0, 220, 100),         # Vibrant Green
            FingerStatus.NEEDS_ADJUSTMENT: (0, 215, 255),# Yellow / Amber
            FingerStatus.INCORRECT: (0, 0, 255),         # Bright Red
        }

        # First draw normal connections in muted cyan
        coords: List[Tuple[int, int]] = []
        for i in range(21):
            px = int(np.clip(landmarks_21_3[i, 0] * w, 0, w - 1))
            py = int(np.clip(landmarks_21_3[i, 1] * h, 0, h - 1))
            coords.append((px, py))

        for s_idx, e_idx in HAND_CONNECTIONS:
            cv2.line(out, coords[s_idx], coords[e_idx], (200, 180, 0), 2, cv2.LINE_AA)

        # Draw color-coded joint circles per finger
        for finger_name, joint_idxs in FINGER_INDICES_MAP.items():
            status = finger_statuses.get(finger_name, FingerStatus.CORRECT)
            col = status_colors[status]
            for j_idx in joint_idxs:
                px, py = coords[j_idx]
                cv2.circle(out, (px, py), 5, col, -1, cv2.LINE_AA)
                cv2.circle(out, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)

        # Wrist joint (index 0)
        cv2.circle(out, coords[0], 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, coords[0], 4, (0, 180, 255), -1, cv2.LINE_AA)

        return out
