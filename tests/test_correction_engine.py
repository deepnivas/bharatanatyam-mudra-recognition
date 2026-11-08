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
    assert any("reference" in inst.lower() for inst in report.instructions)


def test_state_a_correct_exact_reference():
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

    assert report.state == MudraEvaluationState.STATE_A_CORRECT
    assert report.similarity_score >= 0.99
    assert all(s == FingerStatus.CORRECT for s in report.finger_statuses.values())
    assert "excellent" in report.instructions[0].lower() or "criteria" in report.instructions[0].lower()


def test_state_c_incorrect_execution_minor_adjustment():
    engine = MudraCorrectionEngine()
    ref_data = engine.get_reference_data("1-Pataka")
    assert ref_data is not None

    ref_lms = ref_data["landmarks"]
    # Create slight deviation: bend index finger slightly
    mod_lms = ref_lms.copy()
    mod_lms[6, 1] += 0.05  # PIP
    mod_lms[7, 1] += 0.08  # DIP

    res = MudraRecognitionResult(
        hand_detected=True,
        predicted_mudra="1-Pataka",
        confidence=0.88,
        landmarks_21_3=mod_lms,
        is_expected_match=True,
    )
    report = engine.evaluate_pose(res, expected_mudra="1-Pataka")

    assert report.state == MudraEvaluationState.STATE_C_INCORRECT_EXECUTION
    # Index should need adjustment, others correct
    assert report.finger_statuses["Index"] in (FingerStatus.NEEDS_ADJUSTMENT, FingerStatus.INCORRECT)
    assert "index" in " ".join(report.instructions).lower()
    assert report.similarity_score > 0.9


def test_finger_status_reporting_completeness():
    engine = MudraCorrectionEngine()
    ref_data = engine.get_reference_data("1-Pataka")
    assert ref_data is not None

    res = MudraRecognitionResult(
        hand_detected=True,
        predicted_mudra="1-Pataka",
        confidence=0.9,
        landmarks_21_3=ref_data["landmarks"],
        is_expected_match=True,
    )
    report = engine.evaluate_pose(res, expected_mudra="1-Pataka")

    for finger in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
        assert finger in report.finger_statuses
        assert report.finger_statuses[finger] in [
            FingerStatus.CORRECT,
            FingerStatus.NEEDS_ADJUSTMENT,
            FingerStatus.INCORRECT,
        ]


def test_draw_correction_skeleton_output():
    engine = MudraCorrectionEngine()
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    pts = np.random.rand(21, 3).astype(np.float32)
    statuses = {f: FingerStatus.CORRECT for f in ["Thumb", "Index", "Middle", "Ring", "Pinky"]}

    annotated = engine.draw_correction_skeleton(img, pts, statuses)
    assert annotated.shape == (400, 400, 3)
    # Should not be identical to blank image (some drawing happened)
    assert not np.array_equal(annotated, img)


def test_instruction_generation_not_empty():
    engine = MudraCorrectionEngine()
    ref_data = engine.get_reference_data("1-Pataka")
    assert ref_data is not None

    mod_lms = ref_data["landmarks"].copy()
    mod_lms[6, 1] += 0.1
    mod_lms[7, 1] += 0.15

    res = MudraRecognitionResult(
        hand_detected=True,
        predicted_mudra="1-Pataka",
        confidence=0.9,
        landmarks_21_3=mod_lms,
        is_expected_match=True,
    )
    report = engine.evaluate_pose(res, expected_mudra="1-Pataka")
    assert len(report.instructions) >= 1
    assert all(isinstance(inst, str) and len(inst) > 5 for inst in report.instructions)