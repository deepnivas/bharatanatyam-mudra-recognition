#!/usr/bin/env python3
"""
tests/test_recognition_engine.py - Tests for Unified Multi-Model Recognition Engine (Parts 3 & 4).
"""

import numpy as np
import pytest
from src.recognition_engine import MudraRecognitionEngine


def test_model_switching_and_availability():
    engine = MudraRecognitionEngine(default_model="V4")
    assert engine.active_model_key == "V4"

    # Switch across all 5 models
    for key in ["V1", "V2", "V3", "V4", "V5"]:
        engine.set_active_model(key)
        assert engine.active_model_key == key
        assert key in engine.AVAILABLE_MODELS
        assert engine._load_model(key) is not None

    engine.close()


def test_palm_orientation_calculation():
    # Construct a flat palm lying on the XY plane facing +Z
    flat_pts = np.zeros((21, 3), dtype=np.float32)
    flat_pts[0] = [0.0, 0.0, 0.0]     # Wrist
    flat_pts[5] = [-0.2, -0.4, 0.0]   # Index MCP
    flat_pts[9] = [0.0, -0.5, 0.0]    # Middle MCP
    flat_pts[17] = [0.2, -0.4, 0.0]   # Pinky MCP

    orientation = MudraRecognitionEngine.compute_palm_orientation(flat_pts)
    assert "palm_normal" in orientation
    assert "hand_direction" in orientation
    assert "primary_orientation" in orientation
    assert len(orientation["palm_normal"]) == 3


def test_expected_mudra_matching():
    engine = MudraRecognitionEngine(default_model="V4")
    # Feed empty frame to verify no-crash and expected matching logic
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res, annotated = engine.process_frame(dummy_frame, expected_mudra="1-Pataka")

    assert res.hand_detected is False
    assert res.expected_mudra == "1-Pataka"
    assert res.is_expected_match is False
    engine.close()
