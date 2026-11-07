#!/usr/bin/env python3
"""
src/analyze_detection_failures.py - MediaPipe Hand Detection Failure Analysis.

Analyzes detection failures across the Bharatanatyam Mudra dataset:
1. Reuses core components from src/audit_dataset.py:
   - MediaPipeHandDetector (configured in IMAGE mode, 1 hand, confidence 0.5)
   - find_dataset_splits, get_class_folders, get_image_files
   - verify_image (OpenCV integrity verification)
2. Separates failure types:
   - UNREADABLE_IMAGE: OpenCV decode or file corruption failures
   - MEDIAPIPE_EXCEPTION: Exceptions raised during landmark detection
   - NO_HAND_DETECTED: MediaPipe executed successfully but detected 0 hands
3. Exports:
   - reports/mediapipe_audit/detection_failures.csv (detailed itemized list)
   - reports/mediapipe_audit/failure_summary.csv (class-by-class & split statistics)
   - reports/mediapipe_audit/failure_contact_sheet.jpg (annotated thumbnails with class & filename)
4. Manages contact sheet dimensions by splitting into multiple sheets when needed.
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

# Add src to sys.path to ensure module resolution when running directly or as module
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Reuse required components from audit_dataset.py (Requirement 1)
from audit_dataset import (
    MediaPipeHandDetector,
    find_dataset_splits,
    get_class_folders,
    get_image_files,
    natural_sort_key,
    verify_image,
)

try:
    from tqdm import tqdm
except ImportError:
    # Graceful fallback if tqdm is not installed
    def tqdm(iterable, desc=None, total=None, unit="it"):  # type: ignore
        return iterable


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FailureAnalysis")


def draw_centered_text(
    canvas: np.ndarray,
    text: str,
    x0: int,
    tw: int,
    y: int,
    font_scale: float = 0.38,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    """Draws centered text within a horizontal slot [x0, x0 + tw], truncating or shrinking if needed."""
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    while text_size[0] > (tw - 8) and font_scale > 0.25:
        font_scale -= 0.03
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    if text_size[0] > (tw - 8):
        while len(text) > 3 and text_size[0] > (tw - 8):
            text = text[:-4] + "..."
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    tx = x0 + max(2, (tw - text_size[0]) // 2)
    cv2.putText(
        canvas,
        text,
        (tx, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def create_failure_contact_sheets(
    failures: List[Dict[str, Any]],
    output_dir: Path,
    base_filename: str = "failure_contact_sheet.jpg",
    cols: int = 6,
    max_images_per_sheet: int = 60,
    thumb_size: Tuple[int, int] = (200, 200),
    label_height: int = 46,
) -> List[Path]:
    """Generates readable failure contact sheets with class and filename labels.
    
    Splits failures across multiple sheets if total count exceeds max_images_per_sheet
    to maintain manageable image dimensions. Guarantees that failure_contact_sheet.jpg
    is always produced.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_failures = len(failures)
    primary_output_path = output_dir / base_filename
    saved_paths: List[Path] = []

    # If no failures occurred, output a clean placeholder sheet
    if n_failures == 0:
        empty_w, empty_h = 800, 360
        sheet = np.full((empty_h, empty_w, 3), 32, dtype=np.uint8)
        cv2.putText(
            sheet,
            "MediaPipe Hand Detection Failure Analysis",
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            "No detection failures detected. All images processed successfully!",
            (40, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 230, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(primary_output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        logger.info(f"Zero failures recorded. Placeholder saved to: {primary_output_path}")
        return [primary_output_path]

    total_sheets = (n_failures + max_images_per_sheet - 1) // max_images_per_sheet
    tw, th = thumb_size
    cell_w = tw
    cell_h = th + label_height
    padding = 6
    header_height = 54

    for sheet_idx in range(total_sheets):
        start_idx = sheet_idx * max_images_per_sheet
        end_idx = min(start_idx + max_images_per_sheet, n_failures)
        sheet_failures = failures[start_idx:end_idx]
        count_in_sheet = len(sheet_failures)

        rows = (count_in_sheet + cols - 1) // cols
        grid_w = cols * cell_w + (cols + 1) * padding
        grid_h = header_height + rows * cell_h + (rows + 1) * padding

        sheet = np.full((grid_h, grid_w, 3), 30, dtype=np.uint8)

        # Header banner
        header_text = (
            f"MediaPipe Detection Failures - Sheet {sheet_idx + 1} of {total_sheets} "
            f"(Items {start_idx + 1}-{end_idx} of {n_failures})"
        )
        cv2.putText(
            sheet,
            header_text,
            (padding + 10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for idx, rec in enumerate(sheet_failures):
            r = idx // cols
            c = idx % cols
            x0 = padding + c * (cell_w + padding)
            y0 = header_height + padding + r * (cell_h + padding)

            full_path = Path(rec["full_path"])
            failure_type = rec["failure_type"]

            # Load thumbnail or generate fallback tile
            if failure_type == "UNREADABLE_IMAGE":
                thumb = np.zeros((th, tw, 3), dtype=np.uint8)
                cv2.putText(
                    thumb,
                    "CORRUPT",
                    (tw // 5, th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                border_color = (0, 0, 240)  # Red border for unreadable
            else:
                is_valid, img, _ = verify_image(full_path)
                if not is_valid or img is None:
                    thumb = np.zeros((th, tw, 3), dtype=np.uint8)
                    cv2.putText(
                        thumb,
                        "ERR",
                        (tw // 3, th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    thumb = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

                if failure_type == "MEDIAPIPE_EXCEPTION":
                    border_color = (0, 0, 240)  # Red border
                    # Corner badge
                    cv2.rectangle(thumb, (tw - 56, 4), (tw - 4, 22), (0, 0, 180), -1)
                    cv2.putText(
                        thumb,
                        "EXC",
                        (tw - 48, 17),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    # NO_HAND_DETECTED
                    border_color = (40, 120, 230)  # Coral / orange border
                    # Corner badge
                    cv2.rectangle(thumb, (tw - 68, 4), (tw - 4, 22), (20, 80, 190), -1)
                    cv2.putText(
                        thumb,
                        "NO HAND",
                        (tw - 64, 17),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.32,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            # Insert thumbnail
            sheet[y0 : y0 + th, x0 : x0 + tw] = thumb

            # Label area below thumbnail
            label_y0 = y0 + th
            label_y1 = label_y0 + label_height
            sheet[label_y0:label_y1, x0 : x0 + tw] = (18, 18, 18)

            # Class label (Line 1)
            split_abbr = "TR" if rec["split"].lower().startswith("train") else "TE"
            class_display = f"[{split_abbr}] {rec['class']}"
            draw_centered_text(
                sheet,
                class_display,
                x0,
                tw,
                label_y0 + 17,
                font_scale=0.38,
                color=(240, 240, 240),
                thickness=1,
            )

            # Filename label (Line 2)
            filename_display = rec["filename"]
            draw_centered_text(
                sheet,
                filename_display,
                x0,
                tw,
                label_y0 + 35,
                font_scale=0.34,
                color=(180, 180, 180),
                thickness=1,
            )

            # Border around cell
            cv2.rectangle(sheet, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), border_color, 1)

        # File naming:
        # Sheet 0 is always saved as primary_output_path (failure_contact_sheet.jpg)
        # If multiple sheets, also save individual numbered copies (failure_contact_sheet_1.jpg, etc.)
        if sheet_idx == 0:
            target_path = primary_output_path
            cv2.imwrite(str(target_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            saved_paths.append(target_path)
            logger.info(f"Saved primary contact sheet: {target_path}")

            if total_sheets > 1:
                numbered_path = output_dir / f"failure_contact_sheet_1{primary_output_path.suffix}"
                cv2.imwrite(str(numbered_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                saved_paths.append(numbered_path)
        else:
            sheet_num = sheet_idx + 1
            numbered_path = output_dir / f"failure_contact_sheet_{sheet_num}{primary_output_path.suffix}"
            cv2.imwrite(str(numbered_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            saved_paths.append(numbered_path)
            logger.info(f"Saved additional contact sheet: {numbered_path}")

    return saved_paths


def analyze_detection_failures(
    dataset_root: Path,
    model_path: Path,
    output_dir: Path,
    confidence: float = 0.5,
    max_images_per_sheet: int = 60,
    cols: int = 6,
    num_hands: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Path]]:
    """Runs landmark detection failure audit across Training and Testing splits.
    
    Args:
        dataset_root: Root dataset folder containing Training/ and Testing/ splits.
        model_path: Path to MediaPipe hand_landmarker.task model.
        output_dir: Directory where CSV reports and contact sheets will be saved.
        confidence: Detection, presence, and tracking confidence (default: 0.5).
        max_images_per_sheet: Maximum thumbnails per contact sheet.
        cols: Number of columns in contact sheet.
        num_hands: Maximum hands to detect (1 for single hand mudras).
        
    Returns:
        Tuple: (df_failures, df_summary, list_of_contact_sheet_paths)
    """
    dataset_root = Path(dataset_root).resolve()
    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Model path:   {model_path}")
    logger.info(f"Output dir:   {output_dir}")
    logger.info(f"Confidence:   {confidence} (detection / presence / tracking)")

    # 1. Discover dataset splits and classes (Reuse from audit_dataset.py)
    splits = find_dataset_splits(dataset_root)
    split_class_files: Dict[str, Dict[str, List[Path]]] = {}
    total_dataset_images = 0

    for split_name, split_path in splits.items():
        class_folders = get_class_folders(split_path)
        split_class_files[split_name] = {}
        for class_dir in class_folders:
            img_files = get_image_files(class_dir)
            split_class_files[split_name][class_dir.name] = img_files
            total_dataset_images += len(img_files)

    logger.info(
        f"Found {len(splits)} splits, "
        f"{sum(len(c) for c in split_class_files.values())} class folders, "
        f"{total_dataset_images} total images."
    )

    # 2. Initialize MediaPipeHandDetector with exact audit parameters (Requirement 2)
    # IMAGE mode, 1 hand, min_confidence=0.5
    failure_records: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    total_processed = 0
    total_detected = 0
    total_failed = 0
    total_unreadable = 0
    total_exceptions = 0
    total_no_hand = 0

    t_start = time.time()

    with MediaPipeHandDetector(
        model_path=model_path,
        num_hands=num_hands,
        min_hand_detection_confidence=confidence,
        min_hand_presence_confidence=confidence,
        min_tracking_confidence=confidence,
    ) as detector:
        for split_name in ["Training", "Testing"]:
            if split_name not in split_class_files:
                continue

            classes_dict = split_class_files[split_name]
            split_total = 0
            split_readable = 0
            split_detected = 0
            split_failed = 0
            split_unreadable = 0
            split_exceptions = 0
            split_no_hand = 0

            logger.info(f"Analyzing {split_name} split ({len(classes_dict)} classes)...")

            for class_name, img_paths in classes_dict.items():
                cls_total = len(img_paths)
                cls_readable = 0
                cls_detected = 0
                cls_failed = 0
                cls_unreadable = 0
                cls_exceptions = 0
                cls_no_hand = 0

                desc = f"{split_name} | {class_name}"
                for img_path in tqdm(img_paths, desc=desc, leave=False):
                    total_processed += 1
                    split_total += 1

                    # Image integrity check with OpenCV (Requirement 9: handle unreadable separately)
                    is_valid, img, read_err = verify_image(img_path)
                    if not is_valid or img is None:
                        cls_unreadable += 1
                        cls_failed += 1
                        split_unreadable += 1
                        split_failed += 1
                        total_unreadable += 1
                        total_failed += 1

                        failure_records.append(
                            {
                                "split": split_name,
                                "class": class_name,
                                "filename": img_path.name,
                                "full_path": str(img_path.resolve()),
                                "failure_type": "UNREADABLE_IMAGE",
                                "error": read_err or "OpenCV failed to decode image",
                            }
                        )
                        continue

                    cls_readable += 1
                    split_readable += 1

                    # MediaPipe Hand Landmark Detection (Requirement 9: handle exceptions separately)
                    try:
                        detected, _, det_err = detector.detect(img)
                        if det_err is not None:
                            cls_exceptions += 1
                            cls_failed += 1
                            split_exceptions += 1
                            split_failed += 1
                            total_exceptions += 1
                            total_failed += 1

                            failure_records.append(
                                {
                                    "split": split_name,
                                    "class": class_name,
                                    "filename": img_path.name,
                                    "full_path": str(img_path.resolve()),
                                    "failure_type": "MEDIAPIPE_EXCEPTION",
                                    "error": det_err,
                                }
                            )
                        elif not detected:
                            cls_no_hand += 1
                            cls_failed += 1
                            split_no_hand += 1
                            split_failed += 1
                            total_no_hand += 1
                            total_failed += 1

                            failure_records.append(
                                {
                                    "split": split_name,
                                    "class": class_name,
                                    "filename": img_path.name,
                                    "full_path": str(img_path.resolve()),
                                    "failure_type": "NO_HAND_DETECTED",
                                    "error": "No hand detected above confidence threshold",
                                }
                            )
                        else:
                            cls_detected += 1
                            split_detected += 1
                            total_detected += 1
                    except Exception as exc:
                        cls_exceptions += 1
                        cls_failed += 1
                        split_exceptions += 1
                        split_failed += 1
                        total_exceptions += 1
                        total_failed += 1

                        failure_records.append(
                            {
                                "split": split_name,
                                "class": class_name,
                                "filename": img_path.name,
                                "full_path": str(img_path.resolve()),
                                "failure_type": "MEDIAPIPE_EXCEPTION",
                                "error": str(exc),
                            }
                        )

                # Class level summary record
                cls_fail_rate = (cls_failed / cls_total * 100.0) if cls_total > 0 else 0.0
                summary_rows.append(
                    {
                        "split": split_name,
                        "class": class_name,
                        "total_images": cls_total,
                        "readable_images": cls_readable,
                        "hands_detected": cls_detected,
                        "total_failures": cls_failed,
                        "failure_rate_pct": round(cls_fail_rate, 2),
                        "no_hand_detected": cls_no_hand,
                        "unreadable_images": cls_unreadable,
                        "mediapipe_exceptions": cls_exceptions,
                    }
                )

            # Split overall summary row
            split_fail_rate = (split_failed / split_total * 100.0) if split_total > 0 else 0.0
            summary_rows.append(
                {
                    "split": split_name,
                    "class": f"--- {split_name} OVERALL ---",
                    "total_images": split_total,
                    "readable_images": split_readable,
                    "hands_detected": split_detected,
                    "total_failures": split_failed,
                    "failure_rate_pct": round(split_fail_rate, 2),
                    "no_hand_detected": split_no_hand,
                    "unreadable_images": split_unreadable,
                    "mediapipe_exceptions": split_exceptions,
                }
            )

    elapsed = time.time() - t_start

    # Grand combined overall row
    grand_fail_rate = (total_failed / total_processed * 100.0) if total_processed > 0 else 0.0
    grand_readable = total_processed - total_unreadable
    summary_rows.append(
        {
            "split": "All",
            "class": "=== COMBINED OVERALL ===",
            "total_images": total_processed,
            "readable_images": grand_readable,
            "hands_detected": total_detected,
            "total_failures": total_failed,
            "failure_rate_pct": round(grand_fail_rate, 2),
            "no_hand_detected": total_no_hand,
            "unreadable_images": total_unreadable,
            "mediapipe_exceptions": total_exceptions,
        }
    )

    # 3. Export DataFrames to CSV (Requirement 4 & 5)
    # Failure CSV columns: split, class, filename, full path, failure type, error
    failure_columns = ["split", "class", "filename", "full_path", "failure_type", "error"]
    df_failures = pd.DataFrame(failure_records, columns=failure_columns)
    failures_csv_path = output_dir / "detection_failures.csv"
    df_failures.to_csv(failures_csv_path, index=False)
    logger.info(f"Saved detection failures CSV to: {failures_csv_path} ({len(df_failures)} rows)")

    # Summary CSV
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "failure_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved failure summary CSV to: {summary_csv_path} ({len(df_summary)} rows)")

    # 4. Generate contact sheets (Requirement 4 & 6)
    contact_sheet_paths = create_failure_contact_sheets(
        failures=failure_records,
        output_dir=output_dir,
        base_filename="failure_contact_sheet.jpg",
        cols=cols,
        max_images_per_sheet=max_images_per_sheet,
        thumb_size=(200, 200),
        label_height=46,
    )

    # 5. Print comprehensive console summary (Requirement 10)
    grand_det_rate = (total_detected / total_processed * 100.0) if total_processed > 0 else 0.0
    print("\n" + "=" * 65)
    print("          MEDIAPIPE DETECTION FAILURE AUDIT SUMMARY")
    print("=" * 65)
    print(f"Total Images Processed:       {total_processed:>6}")
    print(f"Hands Successfully Detected:  {total_detected:>6}  ({grand_det_rate:.2f}%)")
    print(f"Total Detection Failures:     {total_failed:>6}  ({grand_fail_rate:.2f}%)")
    print("-" * 65)
    print("Failure Breakdown by Type:")
    print(f"  - No Hand Detected:         {total_no_hand:>6}")
    print(f"  - Unreadable / Corrupt:     {total_unreadable:>6}")
    print(f"  - MediaPipe Exceptions:     {total_exceptions:>6}")
    print("-" * 65)
    for row in summary_rows:
        if row["class"].startswith("---"):
            split_lbl = row["split"]
            print(
                f"{split_lbl} Split Failures:        {row['total_failures']:>6} / {row['total_images']:<6} "
                f"({row['failure_rate_pct']:.2f}% fail rate)"
            )
    print("-" * 65)
    print(f"Processing Elapsed Time:      {elapsed:.2f} seconds")
    print("Generated Artifacts:")
    print(f"  1. Failures CSV:    {failures_csv_path}")
    print(f"  2. Summary CSV:     {summary_csv_path}")
    for i, p in enumerate(contact_sheet_paths, start=3):
        print(f"  {i}. Contact Sheet:  {p}")
    print("=" * 65 + "\n")

    return df_failures, df_summary, contact_sheet_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MediaPipe hand detection failures on the Bharatanatyam Mudra dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/Dance-V1.0/Kokul1984-Dance-1c60726"),
        help="Path to root of the dataset containing Training/ and Testing/ folders",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/hand_landmarker.task"),
        help="Path to MediaPipe hand_landmarker.task model file",
    )
    parser.add_argument(
        "--output-dir",
        "--reports-dir",
        dest="output_dir",
        type=Path,
        default=Path("reports/mediapipe_audit"),
        help="Output directory where CSV reports and contact sheets will be saved",
    )
    parser.add_argument(
        "--confidence",
        "--min-confidence",
        dest="confidence",
        type=float,
        default=0.5,
        help="MediaPipe detection, presence, and tracking confidence threshold",
    )
    parser.add_argument(
        "--max-images-per-sheet",
        type=int,
        default=60,
        help="Maximum thumbnails per contact sheet to keep image dimensions manageable",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=6,
        help="Number of thumbnail columns in contact sheet grid",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=1,
        help="Maximum number of hands to detect (1 for single-hand mudras)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        analyze_detection_failures(
            dataset_root=args.dataset_root,
            model_path=args.model_path,
            output_dir=args.output_dir,
            confidence=args.confidence,
            max_images_per_sheet=args.max_images_per_sheet,
            cols=args.cols,
            num_hands=args.num_hands,
        )
    except Exception as e:
        logger.exception(f"Failure analysis halted due to an unhandled exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
