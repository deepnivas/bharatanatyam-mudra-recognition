#!/usr/bin/env python3
"""
tests/test_reference_library.py - Unit tests for Reference Pose Library (Part 5).
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.correction_engine import MudraCorrectionEngine


def test_manifest_integrity():
    manifest_path = Path("data/kuchipudi_references/manifest.json")
    assert manifest_path.is_file(), "Manifest file must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["supported_mudras_count"] == 30, "Must support all 30 canonical mudras"
    assert len(data["mudras"]) == 30, "Mudras dictionary must have 30 entries"

    for mudra_id, entry in data["mudras"].items():
        assert "traditional_name" in entry
        assert "metadata_path" in entry
        assert "image_path" in entry
        assert "landmarks_path" in entry
        assert entry["validated"] is True

        # Check landmarks NPY file
        npy_path = Path(entry["landmarks_path"])
        assert npy_path.is_file(), f"Landmarks NPY missing: {npy_path}"
        arr = np.load(str(npy_path))
        assert arr.shape == (21, 3), f"Landmarks shape must be (21, 3), got {arr.shape}"

        # Check metadata JSON file
        meta_path = Path(entry["metadata_path"])
        assert meta_path.is_file(), f"Metadata JSON missing: {meta_path}"
        with open(meta_path, "r", encoding="utf-8") as mf:
            meta = json.load(mf)
        assert meta["mudra_id"] == mudra_id
        assert len(meta["joint_angles_mean"]) == 15, "Must have 15 joint angle means"
        assert len(meta["joint_angle_tolerances"]) == 15, "Must have 15 joint angle tolerances"


def test_correction_engine_reference_loading():
    engine = MudraCorrectionEngine()
    supported = engine.get_supported_mudras()
    assert len(supported) == 30

    ref_pataka = engine.get_reference_data("1-Pataka")
    assert ref_data_is_valid(ref_pataka)

    ref_bhramara = engine.get_reference_data("24-Bhramara")
    assert ref_data_is_valid(ref_bhramara)


def ref_data_is_valid(ref_data):
    if not ref_data:
        return False
    if "metadata" not in ref_data or "landmarks" not in ref_data:
        return False
    if ref_data["landmarks"].shape != (21, 3):
        return False
    return True
