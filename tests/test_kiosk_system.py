#!/usr/bin/env python3
"""
tests/test_kiosk_system.py - Integration tests for Interactive Kiosk Application (Part 8).
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.kiosk_app import InteractiveMudraKiosk
from src.correction_engine import MudraEvaluationState


def test_kiosk_initialization():
    kiosk = InteractiveMudraKiosk(camera_index=None, default_model="V4", enable_audio=False)
    assert kiosk.current_model_key == "V4"
    assert len(kiosk.supported_mudras) == 30
    assert len(kiosk.buttons) >= 5
    assert kiosk.current_expected_mudra == "1-Pataka"


def test_kiosk_navigation():
    kiosk = InteractiveMudraKiosk(camera_index=None, default_model="V4", enable_audio=False)
    initial_mudra = kiosk.current_expected_mudra

    kiosk.next_mudra()
    assert kiosk.current_expected_mudra != initial_mudra

    kiosk.previous_mudra()
    assert kiosk.current_expected_mudra == initial_mudra


def test_kiosk_model_cycling():
    kiosk = InteractiveMudraKiosk(camera_index=None, default_model="V1", enable_audio=False)
    assert kiosk.current_model_key == "V1"

    kiosk.cycle_model()
    assert kiosk.current_model_key == "V2"

    kiosk.cycle_model()
    assert kiosk.current_model_key == "V3"

    kiosk.cycle_model()
    assert kiosk.current_model_key == "V4"

    kiosk.cycle_model()
    assert kiosk.current_model_key == "V5"

    kiosk.cycle_model()
    assert kiosk.current_model_key == "V1"


def test_kiosk_frame_rendering():
    kiosk = InteractiveMudraKiosk(camera_index=None, default_model="V4", enable_audio=False)
    dummy_cam = np.zeros((480, 640, 3), dtype=np.uint8)

    rec_res, _ = kiosk.rec_engine.process_frame(dummy_cam, expected_mudra="1-Pataka")
    report = kiosk.corr_engine.evaluate_pose(rec_res, expected_mudra="1-Pataka")

    canvas = kiosk.render_kiosk_frame(dummy_cam, rec_res, report, fps=30.0)
    assert canvas.shape == (kiosk.screen_h, kiosk.screen_w, 3)
    assert canvas.dtype == np.uint8


def test_kiosk_audio_manager():
    kiosk = InteractiveMudraKiosk(camera_index=None, default_model="V4", enable_audio=False)
    # Verify non-blocking notification doesn't raise errors when disabled
    kiosk.audio_manager.notify_state(MudraEvaluationState.STATE_A_CORRECT)
    kiosk.audio_manager.notify_state(MudraEvaluationState.STATE_B_WRONG_MUDRA, "Wrong Mudra")
    kiosk.audio_manager.notify_state(MudraEvaluationState.STATE_C_INCORRECT_EXECUTION, "Adjust index")
