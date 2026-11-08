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


@pytest.fixture
def kiosk_instance():
    kiosk = InteractiveMudraKiosk(camera_index=None, default_model="V4", enable_audio=False)
    yield kiosk
    kiosk.close()


def test_kiosk_initialization(kiosk_instance):
    assert kiosk_instance.current_model_key == "V4"
    assert len(kiosk_instance.supported_mudras) == 30
    assert len(kiosk_instance.buttons) >= 5
    assert kiosk_instance.current_expected_mudra == "1-Pataka"


def test_kiosk_navigation(kiosk_instance):
    initial_mudra = kiosk_instance.current_expected_mudra

    kiosk_instance.next_mudra()
    assert kiosk_instance.current_expected_mudra != initial_mudra

    kiosk_instance.previous_mudra()
    assert kiosk_instance.current_expected_mudra == initial_mudra


def test_kiosk_model_cycling(kiosk_instance):
    kiosk_instance.current_model_key = "V1"
    kiosk_instance.rec_engine.set_active_model("V1")
    assert kiosk_instance.current_model_key == "V1"

    kiosk_instance.cycle_model()
    assert kiosk_instance.current_model_key == "V2"

    kiosk_instance.cycle_model()
    assert kiosk_instance.current_model_key == "V3"

    kiosk_instance.cycle_model()
    assert kiosk_instance.current_model_key == "V4"

    kiosk_instance.cycle_model()
    assert kiosk_instance.current_model_key == "V5"

    kiosk_instance.cycle_model()
    assert kiosk_instance.current_model_key == "V1"


def test_kiosk_frame_rendering(kiosk_instance):
    dummy_cam = np.zeros((480, 640, 3), dtype=np.uint8)

    rec_res, _ = kiosk_instance.rec_engine.process_frame(dummy_cam, expected_mudra="1-Pataka")
    report = kiosk_instance.corr_engine.evaluate_pose(rec_res, expected_mudra="1-Pataka")

    canvas = kiosk_instance.render_kiosk_frame(dummy_cam, rec_res, report, fps=30.0)
    assert canvas.shape == (kiosk_instance.screen_h, kiosk_instance.screen_w, 3)
    assert canvas.dtype == np.uint8


def test_kiosk_audio_manager(kiosk_instance):
    kiosk_instance.audio_manager.notify_state(MudraEvaluationState.STATE_A_CORRECT)
    kiosk_instance.audio_manager.notify_state(MudraEvaluationState.STATE_B_WRONG_MUDRA, "Wrong Mudra")
    kiosk_instance.audio_manager.notify_state(MudraEvaluationState.STATE_C_INCORRECT_EXECUTION, "Adjust index")
