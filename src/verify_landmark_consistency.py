#!/usr/bin/env python3
"""
verify_landmark_consistency.py - Verify coordinate invariance and consistency for V3.

Verifies that uniform upscaling (1.5x) preserves the exact coordinate system and that
the resulting normalized 63-dimensional hand landmark features are invariant between
raw images and 1.5x upscaled images.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

# Add src to path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from audit_dataset import MediaPipeHandDetector
from audit_dataset_merge import load_image_unicode
from features import get_landmark_column_names, normalize_landmarks

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("LandmarkConsistency")


def verify_consistency(
    image_paths: List[Path],
    model_path: Path,
    output_report_path: Path,
) -> Dict[str, Any]:
    """Tests landmark consistency on sample images at 1.0x (raw) vs 1.5x (upscaled)."""
    logger.info("Initializing MediaPipe Hand Detectors for consistency test...")
    det_05 = MediaPipeHandDetector(model_path=model_path, num_hands=1, min_hand_detection_confidence=0.5)
    det_03 = MediaPipeHandDetector(model_path=model_path, num_hands=1, min_hand_detection_confidence=0.3)
    det_05.start()
    det_03.start()

    results: List[Dict[str, Any]] = []
    mae_features: List[float] = []
    max_err_features: List[float] = []
    cosine_sims: List[float] = []

    try:
        for idx, p in enumerate(image_paths):
            img_raw, _ = load_image_unicode(p)
            if img_raw is None:
                continue

            # Stage 1: Raw 1.0x detection at conf=0.5
            det_raw, res_raw, _ = det_05.detect(img_raw)
            if not det_raw or not res_raw or not res_raw.hand_landmarks:
                continue

            pts_raw = np.array([[lm.x, lm.y, lm.z] for lm in res_raw.hand_landmarks[0]], dtype=np.float32)

            # Stage 2: 1.5x upscaled detection at conf=0.3
            h, w = img_raw.shape[:2]
            img_up = cv2.resize(img_raw, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_LINEAR)
            det_up, res_up, _ = det_03.detect(img_up)
            if not det_up or not res_up or not res_up.hand_landmarks:
                continue

            pts_up = np.array([[lm.x, lm.y, lm.z] for lm in res_up.hand_landmarks[0]], dtype=np.float32)

            # Compare raw normalized image-space coordinates [x, y] in [0, 1]
            raw_coord_diff = np.abs(pts_raw[:, :2] - pts_up[:, :2])
            raw_coord_mae = float(np.mean(raw_coord_diff))

            # Apply full feature normalization (wrist translation + max distance scaling)
            feat_raw = normalize_landmarks(pts_raw).flatten()
            feat_up = normalize_landmarks(pts_up).flatten()

            feat_diff = np.abs(feat_raw - feat_up)
            feat_mae = float(np.mean(feat_diff))
            feat_max = float(np.max(feat_diff))

            # Cosine similarity
            dot = np.dot(feat_raw, feat_up)
            norm_prod = (np.linalg.norm(feat_raw) * np.linalg.norm(feat_up)) + 1e-12
            cos_sim = float(dot / norm_prod)

            mae_features.append(feat_mae)
            max_err_features.append(feat_max)
            cosine_sims.append(cos_sim)

            results.append({
                "image": p.name,
                "raw_coord_mae": raw_coord_mae,
                "normalized_feature_mae": feat_mae,
                "normalized_feature_max_err": feat_max,
                "cosine_similarity": cos_sim,
            })

            if len(results) >= 50:
                break

    finally:
        det_05.close()
        det_03.close()

    summary = {
        "num_tested_samples": len(results),
        "mean_raw_coord_mae": float(np.mean([r["raw_coord_mae"] for r in results])),
        "mean_feature_mae": float(np.mean(mae_features)),
        "median_feature_mae": float(np.median(mae_features)),
        "p95_feature_mae": float(np.percentile(mae_features, 95)),
        "max_feature_err_observed": float(np.max(max_err_features)),
        "mean_cosine_similarity": float(np.mean(cosine_sims)),
        "min_cosine_similarity": float(np.min(cosine_sims)),
        "is_consistent": bool(np.mean(cosine_sims) > 0.99 and np.mean(mae_features) < 0.05),
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": results[:20]}, f, indent=2)

    logger.info("=" * 70)
    logger.info("LANDMARK COORDINATE CONSISTENCY VERIFICATION SUMMARY")
    logger.info(f"Tested Samples:            {summary['num_tested_samples']}")
    logger.info(f"Mean Raw Coord MAE:        {summary['mean_raw_coord_mae']:.6f} (in [0, 1] normalized space)")
    logger.info(f"Mean Feature Vector MAE:   {summary['mean_feature_mae']:.6f}")
    logger.info(f"Median Feature Vector MAE: {summary['median_feature_mae']:.6f}")
    logger.info(f"P95 Feature Vector MAE:    {summary['p95_feature_mae']:.6f}")
    logger.info(f"Mean Cosine Similarity:    {summary['mean_cosine_similarity']:.6f}")
    logger.info(f"Min Cosine Similarity:     {summary['min_cosine_similarity']:.6f}")
    logger.info(f"Coordinate Invariance:     {'CONFIRMED (PASS)' if summary['is_consistent'] else 'FAIL'}")
    logger.info("=" * 70)

    return summary


def main():
    root = Path(r"C:\Users\Home\Bharatanatyam-Mudra-Dataset")
    model_path = Path("models/hand_landmarker.task")
    output_path = Path("reports/v3_experiment/landmark_consistency_verification.json")

    # Sample images from multiple classes
    images: List[Path] = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            imgs = list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
            if imgs:
                images.extend(imgs[:5])

    verify_consistency(images, model_path, output_path)


if __name__ == "__main__":
    main()
