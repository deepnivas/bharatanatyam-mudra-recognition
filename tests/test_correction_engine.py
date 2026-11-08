#!/usr/bin/env python3
"""
tests/test_correction_engine.py - Tests for Explainable Correction Engine & 4-State System (Parts 6 & 7).
"""

import numpy as np
import pytest
from src.correction_engine import (
    MudraCorrectionEngine,
    MudraEvaluationState,
    FingerStatus,
)
from src.recognition_engine import MudraRecognitionResult
from src.import_kuchipudi_reference import import_instructor_reference
from pathlib import Path


def test_state_d_unable_to_assess_no_hand():
    engine = MudraCorrectionEngine()
    res = MudraRecognitionResult(hand_detected=False)
    report = engine.evaluate_pose(res, expected_mudra="1-Pataka")

    assert report.state == MudraEvaluationState.STATE_D_UNABLE_TO_ASSESS
    assert "detected" in report.explanation.lower() or "camera" in report.explanation.lower()


def test_state_d_unable_to_assess_low_confidence():
    engine = MudraCorrectionEngine()
    dummy_pts = np.random.rand(21, 3).astype(np.float32)
    res = MudraRecognitionResult(
        hand_detected=True,
        predicted_mudra="1-Pataka",
        confidence=0.25,  # Below default 0.40 threshold
        landmarks_21_3=dummy_pts,
    )
    report = engine.evaluate_pose(res, expected_mudra="1-Pataka")
    assert report.state == MudraEvaluationState.STATE_D_UNABLE_TO_ASSESS


def test_state_b_wrong_mudra():
    engine = MudraCorrectionEngine()
    dummy_pts = np.random.rand(21, 3).astype(np.float32)
    res = MudraRecognitionResult(
        hand_detected=True,
        predicted_mudra="10-Shikara",
        confidence=0.85,
        landmarks_21_3=dummy_pts,
        is_expected_match=False,
    )
    report = engine.evaluate_pose(res, expected_mudra="1-Pataka")

    assert report.state == MudraEvaluationState.STATE_B_WRONG_MUDRA
    assert report.expected_mudra == "1-Pataka"
    assert report.recognized_mudra == "10-Shikara"
    assert "differs" in report.explanation.lower()


def test_reference_validation_required_for_unverified():
    """Ensure STATE A cannot be issued using unverified reference data (Audit Rule)."""
    engine = MudraCorrectionEngine()
    ref_data = engine.get_reference_data("1-Pataka")
    assert ref_data is not None

    res = MudraRecognitionResult(
        hand_detected=True,
        predicted_mudra="1-Pataka",
        confidence=0.95,
        landmarks_21_3=ref_data["landmarks"],
        is_expected_match=True,
    )
    report = engine.evaluate_pose(res, expected_mudra="1-Pataka")

    # Because default records are pending_instructor_validation, State A must be blocked
    assert report.state == MudraEvaluationState.STATE_D_UNABLE_TO_ASSESS
    assert report.explanation == "REFERENCE VALIDATION REQUIRED"


def test_state_a_correct_with_verified_instructor_reference():
    """Ensure STATE A can be issued once an instructor-verified reference is imported."""
    engine = MudraCorrectionEngine()
    ref_data = engine.get_reference_data("1-Pataka")
    assert ref_data is not None

    try:
        # Simulate importing verified reference
        img_path = Path("reports/webcam_debug_frame.jpg")
        if not img_path.is_file():
            img_path = Path("reports/webcam_sample.jpg")
        if not img_path.is_file():
            # Create dummy image
            img_path.parent.mkdir(parents=True, exist_ok=True)
            import cv2
            cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))

        npy_path = Path("data/kuchipudi_references/landmarks/1-Pataka_reference.npy")
        import_instructor_reference(
            mudra_name="1-Pataka",
            image_path=img_path,
            landmarks_npy_path=npy_path,
            instructor_name="Guru Test Instructor",
            lineage_notes="Kuchipudi Test Lineage",
        )

        # Clear cache
        engine._reference_cache.clear()
        engine.load_manifest()

        res = MudraRecognitionResult(
            hand_detected=True,
            predicted_mudra="1-Pataka",
            confidence=0.95,
            landmarks_21_3=ref_data["landmarks"],
            is_expected_match=True,
        )
        report = engine.evaluate_pose(res, expected_mudra="1-Pataka")

        assert report.state == MudraEvaluationState.STATE_A_CORRECT
        assert report.has_instructor_tolerances is True
        assert report.validation_status == "instructor_verified"
    finally:
        # Reset 1-Pataka back to pending_instructor_validation after test
        from src.build_kuchipudi_references import build_reference_library
        build_reference_library(Path("data/processed/train_landmarks.csv"), Path("data/kuchipudi_references"))


def test_draw_correction_skeleton_output():
    engine = MudraCorrectionEngine()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    pts = np.random.rand(21, 3).astype(np.float32)
    statuses = {f: FingerStatus.CORRECT for f in ["Thumb", "Index", "Middle", "Ring", "Pinky"]}

    annotated = engine.draw_correction_skeleton(img, pts, statuses)
    assert annotated.shape == (400, 400, 3)
    assert not np.array_equal(annotated, img)
