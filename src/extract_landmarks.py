#!/usr/bin/env python3
"""
src/extract_landmarks.py - Milestone 2: Bharatanatyam Mudra Landmark Extraction.

Extracts normalized 21-point hand landmarks using MediaPipe HandLandmarker Tasks API:
- Processes Training/ and Testing/ splits independently without mixing.
- Translation invariance: landmark 0 (wrist) positioned at origin (0, 0, 0).
- Scale invariance: normalized by maximum Euclidean distance from wrist.
- Failed detections are preserved explicitly with landmark_detected=False.
- Exports processed datasets to data/processed/train_landmarks.csv and test_landmarks.csv.
- Exports summary metrics to reports/landmark_extraction_summary.csv.
- Generates 21-point skeleton overlay visualization to reports/landmark_samples.jpg.

Reuses MediaPipeHandDetector from Milestone 1 and features from src/features.py.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# Add src to sys.path to allow imports from local directory or package
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from audit_dataset import (
    MediaPipeHandDetector,
    find_dataset_splits,
    get_class_folders,
    get_image_files,
    natural_sort_key,
    verify_image,
)
from features import (
    draw_hand_landmarks_on_image,
    get_landmark_column_names,
    normalize_landmarks,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, total=None, leave=True):
        return iterable


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("LandmarkExtraction")


def create_skeleton_visualization_grid(
    samples: Dict[str, Tuple[np.ndarray, Any]],
    output_path: Path,
    cols: int = 6,
    thumb_size: Tuple[int, int] = (280, 280),
    label_height: int = 36,
) -> None:
    """Builds a contact sheet grid showing the 21-point hand skeletons overlaid on representative images.
    
    Args:
        samples: Mapping of class_name -> (bgr_image, landmarks_list).
        output_path: Target path for the output JPEG.
        cols: Number of columns in the grid.
        thumb_size: (width, height) for each cell thumbnail.
        label_height: Height of label banner at bottom of each cell.
    """
    sorted_classes = sorted(samples.keys(), key=natural_sort_key)
    n = len(sorted_classes)
    if n == 0:
        logger.warning("No samples available to render landmark contact sheet.")
        return

    rows = (n + cols - 1) // cols
    tw, th = thumb_size
    cell_w = tw
    cell_h = th + label_height
    padding = 8
    header_height = 54

    grid_w = cols * cell_w + (cols + 1) * padding
    grid_h = header_height + rows * cell_h + (rows + 1) * padding

    # Canvas with dark background
    canvas = np.full((grid_h, grid_w, 3), 30, dtype=np.uint8)

    # Header banner
    cv2.putText(
        canvas,
        "Bharatanatyam Mudras - MediaPipe 21-Point Hand Skeleton Extraction",
        (padding + 12, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for idx, cls_name in enumerate(sorted_classes):
        r = idx // cols
        c = idx % cols
        x0 = padding + c * (cell_w + padding)
        y0 = header_height + padding + r * (cell_h + padding)

        img, lms = samples[cls_name]
        # Overlay landmarks on image
        annotated = draw_hand_landmarks_on_image(
            img,
            lms,
            draw_connections=True,
            draw_points=True,
            connection_color=(255, 215, 0),  # Bright cyan/blue-green
            point_color=(0, 90, 255),        # Bright orange/red
            point_radius=4,
            line_thickness=2,
        )

        thumb = cv2.resize(annotated, (tw, th), interpolation=cv2.INTER_AREA)

        # Place thumbnail
        canvas[y0 : y0 + th, x0 : x0 + tw] = thumb

        # Label bar below thumbnail
        label_y0 = y0 + th
        label_y1 = label_y0 + label_height
        canvas[label_y0:label_y1, x0 : x0 + tw] = (18, 18, 18)

        text = f"{cls_name} (21 pts)"
        font_scale = 0.44
        thickness = 1
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = x0 + max(2, (tw - text_size[0]) // 2)
        ty = label_y0 + (label_height + text_size[1]) // 2 - 2

        cv2.putText(
            canvas,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (240, 240, 240),
            thickness,
            cv2.LINE_AA,
        )

        # Border
        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (65, 65, 65), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if success:
        logger.info(f"Landmark visualization grid saved to: {output_path}")
    else:
        logger.error(f"Failed to save landmark visualization to: {output_path}")


def extract_landmarks(
    dataset_root: Path,
    model_path: Path,
    processed_dir: Path,
    reports_dir: Path,
    samples_output: Path,
    num_hands: int = 1,
    min_confidence: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extracts MediaPipe landmarks for both Training and Testing splits, applies normalization,
    and saves processed CSV datasets and summary reports.
    """
    logger.info("Initializing MediaPipe Landmark Extraction (Milestone 2)")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Processed output directory: {processed_dir}")

    splits = find_dataset_splits(dataset_root)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    landmark_cols = get_landmark_column_names()
    csv_header = ["split", "class_id", "class_name", "image_path", "landmark_detected"] + landmark_cols

    # Collect representative samples for visualization
    vis_samples: Dict[str, Tuple[np.ndarray, Any]] = {}

    split_dfs: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, Any]] = []

    split_stats: Dict[str, Dict[str, int]] = {
        "Training": {"total": 0, "success": 0, "failed": 0},
        "Testing": {"total": 0, "success": 0, "failed": 0},
    }

    t0 = time.time()
    total_images_processed = 0

    with MediaPipeHandDetector(
        model_path=model_path,
        num_hands=num_hands,
        min_hand_detection_confidence=min_confidence,
    ) as detector:
        for split_name in ["Training", "Testing"]:
            split_dir = splits[split_name]
            class_folders = get_class_folders(split_dir)
            split_rows: List[Dict[str, Any]] = []

            logger.info(f"Processing split '{split_name}' ({len(class_folders)} classes)...")

            for cf in class_folders:
                class_name = cf.name
                # Parse class_id from prefix (e.g. '1-Pataka' -> 1)
                prefix_str = class_name.split("-")[0]
                class_id = int(prefix_str) if prefix_str.isdigit() else 0

                image_files = get_image_files(cf)
                class_total = len(image_files)
                class_success = 0
                class_failed = 0

                desc = f"{split_name} | {class_name}"
                for img_path in tqdm(image_files, desc=desc, leave=False):
                    total_images_processed += 1
                    # Store POSIX relative path for portability
                    try:
                        rel_path = img_path.relative_to(Path.cwd()).as_posix()
                    except ValueError:
                        rel_path = img_path.as_posix()

                    is_valid, img, err = verify_image(img_path)
                    if not is_valid or img is None:
                        logger.warning(f"Unreadable image: {img_path} ({err})")
                        row = {
                            "split": split_name,
                            "class_id": class_id,
                            "class_name": class_name,
                            "image_path": rel_path,
                            "landmark_detected": False,
                        }
                        for col in landmark_cols:
                            row[col] = np.nan
                        split_rows.append(row)
                        class_failed += 1
                        continue

                    # Run MediaPipe HandLandmarker in IMAGE mode
                    detected, result, det_err = detector.detect(img)

                    if detected and result and result.hand_landmarks:
                        first_hand = result.hand_landmarks[0]
                        # 21 raw normalized landmarks
                        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in first_hand], dtype=np.float32)

                        # Apply wrist origin subtraction and hand scale normalization
                        norm_pts = normalize_landmarks(raw_pts, scale_mode="max_distance")
                        flat_norm = norm_pts.flatten()

                        row = {
                            "split": split_name,
                            "class_id": class_id,
                            "class_name": class_name,
                            "image_path": rel_path,
                            "landmark_detected": True,
                        }
                        for col_name, val in zip(landmark_cols, flat_norm):
                            row[col_name] = round(float(val), 6)

                        split_rows.append(row)
                        class_success += 1

                        # Store first detected sample for each class from Training for visualization
                        if split_name == "Training" and class_name not in vis_samples:
                            vis_samples[class_name] = (img, first_hand)
                    else:
                        # Preserve sample as failed detection - never silently skip
                        row = {
                            "split": split_name,
                            "class_id": class_id,
                            "class_name": class_name,
                            "image_path": rel_path,
                            "landmark_detected": False,
                        }
                        for col in landmark_cols:
                            row[col] = np.nan

                        split_rows.append(row)
                        class_failed += 1

                # Class summary stats
                class_rate = round((class_success / class_total * 100.0) if class_total > 0 else 0.0, 2)
                summary_rows.append(
                    {
                        "class": class_name,
                        "split": split_name,
                        "total_images": class_total,
                        "successful_landmarks": class_success,
                        "failed_landmarks": class_failed,
                        "detection_rate": round(class_success / class_total, 4) if class_total > 0 else 0.0,
                        "detection_rate_pct": class_rate,
                    }
                )

                split_stats[split_name]["total"] += class_total
                split_stats[split_name]["success"] += class_success
                split_stats[split_name]["failed"] += class_failed

            # Create DataFrame for this split
            df_split = pd.DataFrame(split_rows, columns=csv_header)
            split_dfs[split_name] = df_split

    elapsed = time.time() - t0
    logger.info(f"Landmark extraction completed for {total_images_processed} images in {elapsed:.2f}s.")

    # Save processed CSV files
    train_csv_path = processed_dir / "train_landmarks.csv"
    test_csv_path = processed_dir / "test_landmarks.csv"

    split_dfs["Training"].to_csv(train_csv_path, index=False)
    logger.info(f"Saved Training landmarks to: {train_csv_path} ({len(split_dfs['Training'])} rows)")

    split_dfs["Testing"].to_csv(test_csv_path, index=False)
    logger.info(f"Saved Testing landmarks to: {test_csv_path} ({len(split_dfs['Testing'])} rows)")

    # Overall summary rows for landmark_extraction_summary.csv
    for s_name in ["Training", "Testing"]:
        s_tot = split_stats[s_name]["total"]
        s_suc = split_stats[s_name]["success"]
        s_fai = split_stats[s_name]["failed"]
        s_rate = (s_suc / s_tot * 100.0) if s_tot > 0 else 0.0
        summary_rows.append(
            {
                "class": f"--- {s_name} TOTAL ---",
                "split": s_name,
                "total_images": s_tot,
                "successful_landmarks": s_suc,
                "failed_landmarks": s_fai,
                "detection_rate": round(s_suc / s_tot, 4) if s_tot > 0 else 0.0,
                "detection_rate_pct": round(s_rate, 2),
            }
        )

    comb_tot = split_stats["Training"]["total"] + split_stats["Testing"]["total"]
    comb_suc = split_stats["Training"]["success"] + split_stats["Testing"]["success"]
    comb_fai = split_stats["Training"]["failed"] + split_stats["Testing"]["failed"]
    comb_rate = (comb_suc / comb_tot * 100.0) if comb_tot > 0 else 0.0
    summary_rows.append(
        {
            "class": "=== COMBINED TOTAL ===",
            "split": "All",
            "total_images": comb_tot,
            "successful_landmarks": comb_suc,
            "failed_landmarks": comb_fai,
            "detection_rate": round(comb_suc / comb_tot, 4) if comb_tot > 0 else 0.0,
            "detection_rate_pct": round(comb_rate, 2),
        }
    )

    df_summary = pd.DataFrame(summary_rows)
    # Reorder columns matching requirement 8
    df_summary_export = df_summary[
        ["class", "split", "total_images", "successful_landmarks", "failed_landmarks", "detection_rate"]
    ]
    summary_csv_path = reports_dir / "landmark_extraction_summary.csv"
    df_summary_export.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved extraction summary to: {summary_csv_path}")

    # Generate visualization grid
    create_skeleton_visualization_grid(
        samples=vis_samples,
        output_path=samples_output,
        cols=6,
        thumb_size=(280, 280),
        label_height=36,
    )

    # Print Final Summary matching Requirement 10
    tr_tot = split_stats["Training"]["total"]
    tr_suc = split_stats["Training"]["success"]
    tr_fai = split_stats["Training"]["failed"]
    tr_pct = (tr_suc / tr_tot * 100.0) if tr_tot > 0 else 0.0

    te_tot = split_stats["Testing"]["total"]
    te_suc = split_stats["Testing"]["success"]
    te_fai = split_stats["Testing"]["failed"]
    te_pct = (te_suc / te_tot * 100.0) if te_tot > 0 else 0.0

    print("\n" + "=" * 45)
    print("LANDMARK EXTRACTION")
    print("Training:")
    print(f"Images: {tr_tot}")
    print(f"Successful: {tr_suc}")
    print(f"Failed: {tr_fai}")
    print(f"Detection rate: {tr_pct:.2f}%\n")
    print("Testing:")
    print(f"Images: {te_tot}")
    print(f"Successful: {te_suc}")
    print(f"Failed: {te_fai}")
    print(f"Detection rate: {te_pct:.2f}%")
    print("=" * 45 + "\n")

    return split_dfs["Training"], split_dfs["Testing"], df_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 2: MediaPipe Landmark Extraction & Normalization for Bharatanatyam Mudras",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/Dance-V1.0/Kokul1984-Dance-1c60726"),
        help="Path to root of dataset containing Training/ and Testing/ folders",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/hand_landmarker.task"),
        help="Path to MediaPipe hand_landmarker.task model",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to save train_landmarks.csv and test_landmarks.csv",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to save landmark_extraction_summary.csv",
    )
    parser.add_argument(
        "--samples-output",
        type=Path,
        default=Path("reports/landmark_samples.jpg"),
        help="Path to save skeleton visualization grid",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=1,
        help="Number of hands to detect (1 for single-hand Asamyuta mudras)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum hand detection confidence threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        extract_landmarks(
            dataset_root=args.dataset_root,
            model_path=args.model_path,
            processed_dir=args.processed_dir,
            reports_dir=args.reports_dir,
            samples_output=args.samples_output,
            num_hands=args.num_hands,
            min_confidence=args.min_confidence,
        )
    except Exception as e:
        logger.exception(f"Landmark extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
