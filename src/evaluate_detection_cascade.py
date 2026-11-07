#!/usr/bin/env python3
"""
evaluate_detection_cascade.py - Two-Stage Detection Cascade Evaluation & Extraction (V3).

Evaluates the proposed two-stage detection cascade across all 13,844 eligible new-dataset images:
  - Stage 1: Original raw image, MediaPipe confidence 0.5.
  - Stage 2: For images failing Stage 1, upscale by 1.5x (aspect-preserving linear interpolation)
             and run MediaPipe at confidence 0.3.

Records:
  - Unique successful detections (Stage 1 vs Stage 2 recovered)
  - Total cascade detections & coverage %
  - Images recovered vs images lost
  - Execution time & processing throughput (FPS)
  - Per-class detection rates and recovery gains
  - Normalized 63-feature landmark vectors exported to CSV for V3 training.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EvaluateDetectionCascade")


# 26 Eligible Classes Mapping
NEW_DATASET_MAPPING: Dict[str, Tuple[str, int]] = {
    "Pathaka(1)": ("1-Pataka", 1),
    "Tripathaka(1)": ("2-Tripataka", 2),
    "Ardhapathaka(1)": ("3-Ardhapataka", 3),
    "Katrimukha(1)": ("4-Kartharimukha", 4),
    "Mayura(1)": ("5-Mayura", 5),
    "Ardhachandran(1)": ("6-Ardhachandra", 6),
    "Aralam(1)": ("7-Arala", 7),
    "Shukatundam(1)": ("8-Shukatunda", 8),
    "Mushti(1)": ("9-Mushti", 9),
    "Sikharam(1)": ("10-Shikara", 10),
    "Kapith(1)": ("11-Kapitta", 11),
    "Katakamukha_1": ("12-Katakamukha_1", 12),
    "Katakamukha_2": ("13-Katakamukha_2", 13),
    "Katakamukha_3": ("14-Katakamukha_3", 14),
    "Suchi(1)": ("15-Suchi", 15),
    "Chandrakala(1)": ("16-Chandrakala", 16),
    "Padmakosha(1)": ("17-Padmakosha", 17),
    "Sarpasirsha(1)": ("18-Sarpashirasha", 18),
    "Mrigasirsha(1)": ("19-Mrugashirsha", 19),
    "Simhamukham(1)": ("20-Simhamukha", 20),
    "Alapadmam(1)": ("22-Alapadma", 22),
    "Bramaram(1)": ("24-Bhramara", 24),
    "Hamsasyam(1)": ("25-Hamsassya", 25),
    "Hamsapaksha(1)": ("26-Hamsapaksha", 26),
    "Mukulam(1)": ("28-Mukula", 28),
    "Trishulam(1)": ("30-Trishula", 30),
}


def natural_sort_key(s: Any) -> List[int | str]:
    val = str(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", val)]


def run_cascade_evaluation(
    dataset_root: Path,
    model_path: Path,
    output_landmarks_csv: Path,
    output_summary_csv: Path,
    output_per_class_csv: Path,
    output_report_md: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs the two-stage detection cascade across all 13,844 images."""
    logger.info("=" * 70)
    logger.info("MILESTONE V3: TWO-STAGE DETECTION CASCADE EVALUATION")
    logger.info(f"Dataset Root:       {dataset_root}")
    logger.info(f"MediaPipe Model:    {model_path}")
    logger.info("=" * 70)

    # 1. Discover all images
    all_images: List[Dict[str, Any]] = []
    for folder_name, (canonical_name, class_id) in sorted(NEW_DATASET_MAPPING.items(), key=lambda x: x[1][1]):
        folder_path = dataset_root / folder_name
        if not folder_path.is_dir():
            logger.warning(f"Class folder missing: {folder_path}")
            continue

        imgs: List[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            imgs.extend(folder_path.glob(ext))
        imgs.sort(key=lambda p: natural_sort_key(p.name))

        for img_p in imgs:
            all_images.append({
                "class_id": class_id,
                "canonical_name": canonical_name,
                "folder_name": folder_name,
                "image_path": img_p,
                "image_name": img_p.name,
            })

    total_images = len(all_images)
    logger.info(f"Discovered {total_images} total eligible images across 26 classes.")

    # 2. Initialize Detectors
    logger.info("Initializing Stage 1 (Conf=0.5) and Stage 2 (Conf=0.3) detectors...")
    det_s1 = MediaPipeHandDetector(model_path=model_path, num_hands=1, min_hand_detection_confidence=0.5)
    det_s2 = MediaPipeHandDetector(model_path=model_path, num_hands=1, min_hand_detection_confidence=0.3)
    det_s1.start()
    det_s2.start()

    feature_col_names = get_landmark_column_names()
    extracted_rows: List[Dict[str, Any]] = []

    stage1_detected = 0
    stage2_recovered = 0
    total_failures = 0
    invalid_coords = 0

    t_start = time.time()
    t_stage1 = 0.0
    t_stage2 = 0.0

    try:
        for idx, item in enumerate(all_images):
            img_p = item["image_path"]
            img_raw, load_err = load_image_unicode(img_p)

            rec: Dict[str, Any] = {
                "class_id": item["class_id"],
                "class_name": item["canonical_name"],
                "folder_name": item["folder_name"],
                "image_name": item["image_name"],
                "image_path": img_p.as_posix(),
                "landmark_detected": False,
                "detection_stage": 0,
                "failure_reason": None,
            }
            # Initialize feature columns with NaN
            for col in feature_col_names:
                rec[col] = np.nan

            if img_raw is None:
                rec["failure_reason"] = f"Image decode failure: {load_err}"
                total_failures += 1
                extracted_rows.append(rec)
                continue

            h, w = img_raw.shape[:2]

            # ------------------------------------------------------------------
            # Stage 1: Raw Image at Conf=0.5
            # ------------------------------------------------------------------
            t0 = time.time()
            s1_det, s1_res, s1_err = det_s1.detect(img_raw)
            t_stage1 += (time.time() - t0)

            accepted = False

            if s1_det and s1_res and s1_res.hand_landmarks:
                pts = np.array([[lm.x, lm.y, lm.z] for lm in s1_res.hand_landmarks[0]], dtype=np.float32)
                # Sanity validation
                if not (np.isnan(pts).any() or (pts[:, :2] < -0.5).any() or (pts[:, :2] > 1.5).any()):
                    norm_features = normalize_landmarks(pts).flatten()
                    rec["landmark_detected"] = True
                    rec["detection_stage"] = 1
                    for col_idx, col_name in enumerate(feature_col_names):
                        rec[col_name] = norm_features[col_idx]
                    stage1_detected += 1
                    accepted = True
                else:
                    invalid_coords += 1

            # ------------------------------------------------------------------
            # Stage 2: 1.5x Upscaling at Conf=0.3 (Only for Stage 1 Failures)
            # ------------------------------------------------------------------
            if not accepted:
                t1 = time.time()
                img_up = cv2.resize(img_raw, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_LINEAR)
                s2_det, s2_res, s2_err = det_s2.detect(img_up)
                t_stage2 += (time.time() - t1)

                if s2_det and s2_res and s2_res.hand_landmarks:
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in s2_res.hand_landmarks[0]], dtype=np.float32)
                    if not (np.isnan(pts).any() or (pts[:, :2] < -0.5).any() or (pts[:, :2] > 1.5).any()):
                        norm_features = normalize_landmarks(pts).flatten()
                        rec["landmark_detected"] = True
                        rec["detection_stage"] = 2  # Recovered in Stage 2
                        for col_idx, col_name in enumerate(feature_col_names):
                            rec[col_name] = norm_features[col_idx]
                        stage2_recovered += 1
                        accepted = True
                    else:
                        invalid_coords += 1

            if not accepted:
                rec["failure_reason"] = "No hand detected in Stage 1 or Stage 2"
                total_failures += 1

            extracted_rows.append(rec)

            if (idx + 1) % 2500 == 0 or (idx + 1) == total_images:
                curr_total = stage1_detected + stage2_recovered
                logger.info(
                    f"Processed {idx + 1}/{total_images} | Stage 1: {stage1_detected} | "
                    f"Stage 2 Recovered: {stage2_recovered} | Total: {curr_total} ({(curr_total / (idx + 1))*100:.2f}%)"
                )

    finally:
        det_s1.close()
        det_s2.close()

    total_time = time.time() - t_start
    total_cascade_detected = stage1_detected + stage2_recovered
    cascade_rate = (total_cascade_detected / total_images) * 100.0
    stage1_rate = (stage1_detected / total_images) * 100.0
    fps = total_images / max(0.01, total_time)

    # 3. Export Extracted Landmarks CSV
    output_landmarks_csv.parent.mkdir(parents=True, exist_ok=True)
    df_landmarks = pd.DataFrame(extracted_rows)
    df_landmarks.to_csv(output_landmarks_csv, index=False)
    logger.info(f"Saved full cascade landmarks CSV: {output_landmarks_csv} ({len(df_landmarks)} rows)")

    # 4. Compute Summary Metrics
    summary_data = [{
        "total_images": total_images,
        "stage1_detected": stage1_detected,
        "stage1_rate_pct": round(stage1_rate, 2),
        "stage2_recovered": stage2_recovered,
        "stage2_recovery_pct": round((stage2_recovered / max(1, total_images - stage1_detected)) * 100.0, 2),
        "total_cascade_detected": total_cascade_detected,
        "cascade_detection_rate_pct": round(cascade_rate, 2),
        "images_lost_from_stage1": 0,  # Zero by design: Stage 1 results are strictly preserved
        "net_gain_over_stage1": stage2_recovered,
        "total_unrecovered_failures": total_failures,
        "invalid_landmarks": invalid_coords,
        "total_execution_time_sec": round(total_time, 1),
        "stage1_time_sec": round(t_stage1, 1),
        "stage2_time_sec": round(t_stage2, 1),
        "overall_throughput_fps": round(fps, 1),
    }]
    df_summary = pd.DataFrame(summary_data)
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(output_summary_csv, index=False)
    logger.info(f"Saved cascade summary CSV: {output_summary_csv}")

    # 5. Compute Per-Class Breakdown
    per_class_rows: List[Dict[str, Any]] = []
    for cname in sorted(list(set(r["canonical_name"] for r in all_images)), key=natural_sort_key):
        c_sub = df_landmarks[df_landmarks["class_name"] == cname]
        c_tot = len(c_sub)
        c_s1 = sum(c_sub["detection_stage"] == 1)
        c_s2 = sum(c_sub["detection_stage"] == 2)
        c_total_det = c_s1 + c_s2
        per_class_rows.append({
            "class_name": cname,
            "total_images": c_tot,
            "stage1_detected": c_s1,
            "stage1_rate_pct": round((c_s1 / c_tot) * 100.0, 1),
            "stage2_recovered": c_s2,
            "stage2_recovered_pct": round((c_s2 / max(1, c_tot - c_s1)) * 100.0, 1),
            "cascade_total_detected": c_total_det,
            "cascade_rate_pct": round((c_total_det / c_tot) * 100.0, 1),
            "net_gain": c_s2,
            "coverage_improvement_pct": round(((c_total_det - c_s1) / c_tot) * 100.0, 1),
        })

    df_per_class = pd.DataFrame(per_class_rows)
    output_per_class_csv.parent.mkdir(parents=True, exist_ok=True)
    df_per_class.to_csv(output_per_class_csv, index=False)
    logger.info(f"Saved cascade per-class CSV: {output_per_class_csv}")

    # 6. Write Markdown Report
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# Two-Stage Detection Cascade Evaluation Report (V3)",
        "",
        f"- **Dataset Evaluated:** {dataset_root} (26 classes, 13,844 images)",
        f"- **Stage 1:** Raw Image, MediaPipe Confidence 0.5",
        f"- **Stage 2:** 1.5x Linear Upscale, MediaPipe Confidence 0.3 (triggered only upon Stage 1 failure)",
        "",
        "## Overall Cascade Performance",
        "",
        f"- **Total Images:** {total_images:,}",
        f"- **Stage 1 Baseline Detections:** {stage1_detected:,} ({stage1_rate:.2f}%)",
        f"- **Stage 2 Recovered Images:** {stage2_recovered:,} (Recovered {df_summary['stage2_recovery_pct'].iloc[0]}% of Stage 1 misses)",
        f"- **Total Cascade Detections:** **{total_cascade_detected:,}** (**{cascade_rate:.2f}%**)",
        f"- **Images Lost from Stage 1:** **0** (Stage 1 detections strictly preserved)",
        f"- **Net Gain Over Baseline:** **+{stage2_recovered:,} images**",
        f"- **Invalid Landmarks:** **{invalid_coords}**",
        f"- **Total Processing Time:** {total_time:.1f} seconds ({fps:.1f} FPS)",
        "",
        "## Per-Class Breakdown Table",
        "",
        "| Class Name | Total Images | Stage 1 (Conf 0.5) | Stage 1 Rate | Stage 2 Recovered | Cascade Total | Cascade Rate | Gain (%) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for _, row in df_per_class.iterrows():
        md_lines.append(
            f"| `{row['class_name']}` | {row['total_images']} | {row['stage1_detected']} | {row['stage1_rate_pct']:.1f}% | "
            f"+{row['stage2_recovered']} | **{row['cascade_total_detected']}** | **{row['cascade_rate_pct']:.1f}%** | **+{row['coverage_improvement_pct']:.1f}%** |"
        )

    output_report_md.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info(f"Saved cascade markdown report: {output_report_md}")

    logger.info("=" * 70)
    logger.info("CASCADE EVALUATION SUMMARY")
    logger.info(f"Stage 1 Detected:       {stage1_detected:,} ({stage1_rate:.2f}%)")
    logger.info(f"Stage 2 Recovered:      {stage2_recovered:,}")
    logger.info(f"Total Cascade Detected: {total_cascade_detected:,} ({cascade_rate:.2f}%)")
    logger.info(f"Total Time:             {total_time:.1f}s ({fps:.1f} img/s)")
    logger.info("=" * 70)

    return df_landmarks, df_summary, df_per_class


def main():
    dataset_root = Path(r"C:\Users\Home\Bharatanatyam-Mudra-Dataset")
    model_path = Path("models/hand_landmarker.task")
    output_dir = Path("reports/v3_experiment")
    processed_dir = Path("data/processed/v3_experiment")

    run_cascade_evaluation(
        dataset_root=dataset_root,
        model_path=model_path,
        output_landmarks_csv=processed_dir / "new_dataset_cascade_landmarks.csv",
        output_summary_csv=output_dir / "cascade_evaluation_summary.csv",
        output_per_class_csv=output_dir / "cascade_per_class_rates.csv",
        output_report_md=output_dir / "cascade_evaluation_report.md",
    )


if __name__ == "__main__":
    main()
