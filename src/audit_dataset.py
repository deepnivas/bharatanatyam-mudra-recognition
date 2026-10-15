#!/usr/bin/env python3
"""
src/audit_dataset.py - Milestone 1: Bharatanatyam Mudra Dataset Audit & Landmark Validation.

Performs:
1. Dataset structure audit (Training & Testing splits, 30 Asamyuta Mudra classes).
2. Image integrity verification using OpenCV (JPG/JPEG/PNG).
3. Hand landmark detection verification using MediaPipe Tasks API (IMAGE mode).
4. Comprehensive statistical reporting and CSV export.
5. Contact sheet generation of representative class samples.

Modular design enables direct reuse of MediaPipeHandDetector for downstream training pipeline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    # Graceful fallback if tqdm is not installed
    def tqdm(iterable, desc=None, total=None, unit="it"):
        return iterable


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DatasetAudit")


def natural_sort_key(key_input: Any) -> list[int | str]:
    """Provides natural / numerical sorting for strings and Path objects.
    Ensures '2-Tripataka' comes before '10-Shikara'.
    """
    s = key_input.name if isinstance(key_input, Path) else str(key_input)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


class MediaPipeHandDetector:
    """Encapsulates MediaPipe Tasks API HandLandmarker in IMAGE mode.
    
    Modular and reusable across audit and training pipelines.
    """

    def __init__(
        self,
        model_path: Path | str,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MediaPipe hand landmarker model not found at: {self.model_path}")

        self.num_hands = num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        self._options = vision.HandLandmarkerOptions(
            base_options=self._base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._detector: Optional[vision.HandLandmarker] = None

    def __enter__(self) -> "MediaPipeHandDetector":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def start(self) -> None:
        """Initializes the HandLandmarker task."""
        if self._detector is None:
            self._detector = vision.HandLandmarker.create_from_options(self._options)

    def close(self) -> None:
        """Releases detector resources."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def detect(self, image_bgr: np.ndarray) -> Tuple[bool, Optional[Any], Optional[str]]:
        """Runs detection on a BGR image array.
        
        Returns:
            Tuple: (hand_detected, landmark_result, error_message)
        """
        if self._detector is None:
            self.start()

        if image_bgr is None or image_bgr.size == 0:
            return False, None, "Invalid empty image buffer provided"

        try:
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = self._detector.detect(mp_image)

            hand_detected = bool(result.hand_landmarks and len(result.hand_landmarks) > 0)
            return hand_detected, result, None
        except Exception as e:
            return False, None, str(e)

    def extract_landmark_array(self, image_bgr: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
        """Extracts normalized landmarks as a numpy array of shape (21, 3).
        Designed for downstream feature extraction.
        
        Returns:
            Tuple: (hand_detected, landmarks_array, error_message)
        """
        detected, result, err = self.detect(image_bgr)
        if not detected or result is None or not result.hand_landmarks:
            return False, None, err

        first_hand = result.hand_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in first_hand], dtype=np.float32)
        return True, pts, None


def verify_image(image_path: Path) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
    """Verifies that an image file can be read and decoded cleanly with OpenCV.
    
    Returns:
        Tuple: (is_valid, bgr_image, error_message)
    """
    if not image_path.is_file():
        return False, None, f"File does not exist: {image_path}"

    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            return False, None, "OpenCV failed to decode image (cv2.imread returned None)"
        if img.size == 0:
            return False, None, "Decoded image has zero size"
        if len(img.shape) != 3 or img.shape[2] != 3:
            return False, None, f"Unexpected image shape: {img.shape}"
        return True, img, None
    except Exception as e:
        return False, None, f"Exception while reading image: {e}"


def find_dataset_splits(dataset_root: Path) -> Dict[str, Path]:
    """Locates and validates Training and Testing directories under dataset root."""
    root = dataset_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root directory not found: {root}")

    splits = {}
    for split_name in ["Training", "Testing"]:
        split_path = root / split_name
        if not split_path.is_dir():
            raise FileNotFoundError(f"Expected split directory not found: {split_path}")
        splits[split_name] = split_path

    return splits


def get_class_folders(split_dir: Path) -> List[Path]:
    """Returns naturally sorted list of class subdirectories."""
    folders = [p for p in split_dir.iterdir() if p.is_dir()]
    return sorted(folders, key=natural_sort_key)


def get_image_files(class_dir: Path) -> List[Path]:
    """Returns naturally sorted list of image files in a class folder."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    return sorted(files, key=natural_sort_key)


def print_image_count_table(split_class_counts: Dict[str, Dict[str, int]]) -> None:
    """Prints image count table matching required format."""
    print("\n" + "=" * 55)
    print(f"{'Split':<12}{'Class':<28}{'Images':>10}")
    print("=" * 55)
    for split, class_counts in split_class_counts.items():
        for class_name, count in class_counts.items():
            print(f"{split:<12}{class_name:<28}{count:>10}")
    print("=" * 55 + "\n")


def print_landmark_report_table(records: List[Dict[str, Any]]) -> None:
    """Prints detection summary table to console."""
    print("\n" + "=" * 80)
    print(f"{'Split':<10}{'Class':<24}{'Total':>7}{'Read':>7}{'Detected':>10}{'No Hand':>10}{'Rate (%)':>10}")
    print("=" * 80)
    for r in records:
        print(
            f"{r['split']:<10}{r['class']:<24}{r['total_images']:>7}{r['readable_images']:>7}"
            f"{r['hands_detected']:>10}{r['no_hands_detected']:>10}{r['detection_rate_pct']:>10.2f}"
        )
    print("=" * 80 + "\n")


def create_contact_sheet(
    sample_images: Dict[str, Path],
    output_path: Path,
    cols: int = 6,
    thumb_size: Tuple[int, int] = (240, 240),
    label_height: int = 36,
) -> None:
    """Builds a contact sheet grid of representative mudra images with labels.
    
    Args:
        sample_images: Mapping from class_name to representative image path.
        output_path: Path to save the contact sheet JPEG.
        cols: Number of columns in the grid.
        thumb_size: (width, height) of each thumbnail image.
        label_height: Height of the label banner below each thumbnail.
    """
    sorted_classes = sorted(sample_images.keys(), key=natural_sort_key)
    n_classes = len(sorted_classes)
    if n_classes == 0:
        logger.warning("No sample images provided for contact sheet.")
        return

    rows = (n_classes + cols - 1) // cols
    tw, th = thumb_size
    cell_w = tw
    cell_h = th + label_height
    padding = 6
    header_height = 50

    grid_w = cols * cell_w + (cols + 1) * padding
    grid_h = header_height + rows * cell_h + (rows + 1) * padding

    # Create dark background canvas
    sheet = np.full((grid_h, grid_w, 3), 32, dtype=np.uint8)

    # Draw header banner
    header_text = "Bharatanatyam Mudra Dataset - 30 Asamyuta Mudras (Sample Grid)"
    cv2.putText(
        sheet,
        header_text,
        (padding + 10, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for idx, cls_name in enumerate(sorted_classes):
        r = idx // cols
        c = idx % cols
        x0 = padding + c * (cell_w + padding)
        y0 = header_height + padding + r * (cell_h + padding)

        img_path = sample_images[cls_name]
        is_valid, img, _ = verify_image(img_path)
        if not is_valid or img is None:
            # Fallback black thumbnail if unreadable
            thumb = np.zeros((th, tw, 3), dtype=np.uint8)
            cv2.putText(thumb, "ERR", (tw // 3, th // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            thumb = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

        # Place image thumbnail
        sheet[y0 : y0 + th, x0 : x0 + tw] = thumb

        # Label bar below thumbnail
        label_y0 = y0 + th
        label_y1 = label_y0 + label_height
        sheet[label_y0:label_y1, x0 : x0 + tw] = (20, 20, 20)

        # Draw centered class name
        display_text = cls_name
        font_scale = 0.42
        thickness = 1
        text_size, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = x0 + max(2, (tw - text_size[0]) // 2)
        ty = label_y0 + (label_height + text_size[1]) // 2 - 2

        cv2.putText(
            sheet,
            display_text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (230, 230, 230),
            thickness,
            cv2.LINE_AA,
        )

        # Thin border around cell
        cv2.rectangle(sheet, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (70, 70, 70), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if success:
        logger.info(f"Contact sheet successfully saved to: {output_path}")
    else:
        logger.error(f"Failed to write contact sheet to: {output_path}")


def audit_dataset(
    dataset_root: Path,
    model_path: Path,
    reports_dir: Path,
    samples_file: Path,
    num_hands: int = 1,
    min_confidence: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestrates dataset counting, image validation, MediaPipe landmark detection,
    and report generation.
    """
    logger.info(f"Starting Dataset Audit (Milestone 1)")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"MediaPipe model: {model_path}")

    # 1. Locate splits
    splits = find_dataset_splits(dataset_root)
    logger.info(f"Discovered splits: {list(splits.keys())}")

    # 2. Count image files in every class folder
    split_class_counts: Dict[str, Dict[str, int]] = {}
    split_class_files: Dict[str, Dict[str, List[Path]]] = {}
    representative_samples: Dict[str, Path] = {}

    for split_name, split_path in splits.items():
        split_class_counts[split_name] = {}
        split_class_files[split_name] = {}
        class_folders = get_class_folders(split_path)

        for cf in class_folders:
            imgs = get_image_files(cf)
            split_class_counts[split_name][cf.name] = len(imgs)
            split_class_files[split_name][cf.name] = imgs

            # Collect representative image from Training for the contact sheet
            if split_name == "Training" and imgs and cf.name not in representative_samples:
                representative_samples[cf.name] = imgs[0]

    # 3. Print Image Count Table
    print_image_count_table(split_class_counts)

    # 4 & 5 & 6. Verify with OpenCV and run MediaPipe Tasks HandLandmarker
    dataset_audit_rows: List[Dict[str, Any]] = []
    landmark_report_rows: List[Dict[str, Any]] = []
    unreadable_files: List[Tuple[Path, str]] = []
    detection_failures: List[Tuple[str, str, Path, str]] = []

    total_images_processed = 0
    t0 = time.time()

    with MediaPipeHandDetector(
        model_path=model_path,
        num_hands=num_hands,
        min_hand_detection_confidence=min_confidence,
    ) as detector:
        for split_name in ["Training", "Testing"]:
            split_files_dict = split_class_files[split_name]

            split_total_imgs = 0
            split_readable_imgs = 0
            split_unreadable_imgs = 0
            split_hands_detected = 0
            split_no_hands_detected = 0

            logger.info(f"Auditing split: {split_name} ({len(split_files_dict)} classes)...")

            for class_name, img_paths in split_files_dict.items():
                class_total = len(img_paths)
                class_readable = 0
                class_unreadable = 0
                class_detected = 0
                class_not_detected = 0

                desc = f"{split_name} | {class_name}"
                for img_path in tqdm(img_paths, desc=desc, leave=False):
                    total_images_processed += 1
                    # 4. Check image with OpenCV
                    is_valid, img, err = verify_image(img_path)

                    if not is_valid or img is None:
                        class_unreadable += 1
                        unreadable_files.append((img_path, err or "Decode error"))
                        logger.warning(f"Unreadable image detected: {img_path} - Reason: {err}")
                        continue

                    class_readable += 1

                    # 6. Run HandLandmarker in IMAGE mode
                    detected, _, det_err = detector.detect(img)
                    if det_err:
                        detection_failures.append((split_name, class_name, img_path, det_err))
                        logger.warning(f"MediaPipe error on {img_path}: {det_err}")

                    if detected:
                        class_detected += 1
                    else:
                        class_not_detected += 1

                # Class level stats
                det_rate = (class_detected / class_readable * 100.0) if class_readable > 0 else 0.0

                dataset_audit_rows.append(
                    {
                        "split": split_name,
                        "class": class_name,
                        "total_images": class_total,
                        "readable_images": class_readable,
                        "unreadable_images": class_unreadable,
                        "status": "PASS" if class_unreadable == 0 else "FAIL_CORRUPTED",
                    }
                )

                landmark_report_rows.append(
                    {
                        "split": split_name,
                        "class": class_name,
                        "total_images": class_total,
                        "readable_images": class_readable,
                        "hands_detected": class_detected,
                        "no_hands_detected": class_not_detected,
                        "detection_rate": round(det_rate / 100.0, 4),
                        "detection_rate_pct": round(det_rate, 2),
                    }
                )

                split_total_imgs += class_total
                split_readable_imgs += class_readable
                split_unreadable_imgs += class_unreadable
                split_hands_detected += class_detected
                split_no_hands_detected += class_not_detected

            # Split Overall summary record
            split_det_rate = (
                (split_hands_detected / split_readable_imgs * 100.0) if split_readable_imgs > 0 else 0.0
            )
            dataset_audit_rows.append(
                {
                    "split": split_name,
                    "class": f"--- {split_name} TOTAL ---",
                    "total_images": split_total_imgs,
                    "readable_images": split_readable_imgs,
                    "unreadable_images": split_unreadable_imgs,
                    "status": "PASS" if split_unreadable_imgs == 0 else "FAIL_CORRUPTED",
                }
            )
            landmark_report_rows.append(
                {
                    "split": split_name,
                    "class": f"--- {split_name} OVERALL ---",
                    "total_images": split_total_imgs,
                    "readable_images": split_readable_imgs,
                    "hands_detected": split_hands_detected,
                    "no_hands_detected": split_no_hands_detected,
                    "detection_rate": round(split_det_rate / 100.0, 4),
                    "detection_rate_pct": round(split_det_rate, 2),
                }
            )

    elapsed = time.time() - t0
    logger.info(f"Processed {total_images_processed} images in {elapsed:.2f}s ({total_images_processed/elapsed:.1f} fps).")

    # Combined dataset total
    df_audit = pd.DataFrame(dataset_audit_rows)
    df_landmark = pd.DataFrame(landmark_report_rows)

    grand_total_imgs = sum(
        r["total_images"] for r in dataset_audit_rows if not r["class"].startswith("---")
    )
    grand_readable_imgs = sum(
        r["readable_images"] for r in dataset_audit_rows if not r["class"].startswith("---")
    )
    grand_unreadable_imgs = sum(
        r["unreadable_images"] for r in dataset_audit_rows if not r["class"].startswith("---")
    )
    grand_detected = sum(
        r["hands_detected"] for r in landmark_report_rows if not r["class"].startswith("---")
    )
    grand_not_detected = sum(
        r["no_hands_detected"] for r in landmark_report_rows if not r["class"].startswith("---")
    )
    grand_det_rate = (grand_detected / grand_readable_imgs * 100.0) if grand_readable_imgs > 0 else 0.0

    # Add grand total rows
    dataset_audit_rows.append(
        {
            "split": "All",
            "class": "=== COMBINED TOTAL ===",
            "total_images": grand_total_imgs,
            "readable_images": grand_readable_imgs,
            "unreadable_images": grand_unreadable_imgs,
            "status": "PASS" if grand_unreadable_imgs == 0 else "FAIL_CORRUPTED",
        }
    )
    landmark_report_rows.append(
        {
            "split": "All",
            "class": "=== COMBINED OVERALL ===",
            "total_images": grand_total_imgs,
            "readable_images": grand_readable_imgs,
            "hands_detected": grand_detected,
            "no_hands_detected": grand_not_detected,
            "detection_rate": round(grand_det_rate / 100.0, 4),
            "detection_rate_pct": round(grand_det_rate, 2),
        }
    )

    df_audit = pd.DataFrame(dataset_audit_rows)
    df_landmark = pd.DataFrame(landmark_report_rows)

    # 7. Print landmark summary table
    print_landmark_report_table(landmark_report_rows)

    # Report unreadable images if any
    if unreadable_files:
        logger.error(f"Found {len(unreadable_files)} unreadable image(s):")
        for u_path, u_err in unreadable_files:
            logger.error(f"  - {u_path}: {u_err}")
    else:
        logger.info("Image Integrity Check: 100% of images were successfully read and validated by OpenCV.")

    # 8. Save reports to CSV
    reports_dir.mkdir(parents=True, exist_ok=True)
    audit_csv_path = reports_dir / "dataset_audit.csv"
    landmark_csv_path = reports_dir / "landmark_detection_report.csv"

    df_audit.to_csv(audit_csv_path, index=False)
    logger.info(f"Dataset audit saved to: {audit_csv_path}")

    df_landmark.to_csv(landmark_csv_path, index=False)
    logger.info(f"Landmark detection report saved to: {landmark_csv_path}")

    # 9. Create contact sheet of representative samples
    create_contact_sheet(
        sample_images=representative_samples,
        output_path=samples_file,
        cols=6,
        thumb_size=(240, 240),
        label_height=36,
    )

    # Print Final Summary Banner
    training_summary = [r for r in landmark_report_rows if r["class"] == "--- Training OVERALL ---"][0]
    testing_summary = [r for r in landmark_report_rows if r["class"] == "--- Testing OVERALL ---"][0]

    print("\n" + "=" * 60)
    print("                 MILESTONE 1 AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total Images Found:           {grand_total_imgs}")
    print(f"Successfully Read Images:     {grand_readable_imgs}")
    print(f"Unreadable Images:            {grand_unreadable_imgs}")
    print(f"Total Hand Landmark Detections: {grand_detected}")
    print(f"Total No-Hand Detections:      {grand_not_detected}")
    print(f"Training Overall Detection Rate: {training_summary['detection_rate_pct']:.2f}% ({training_summary['hands_detected']}/{training_summary['readable_images']})")
    print(f"Testing Overall Detection Rate:  {testing_summary['detection_rate_pct']:.2f}% ({testing_summary['hands_detected']}/{testing_summary['readable_images']})")
    print(f"Combined Detection Rate:         {grand_det_rate:.2f}% ({grand_detected}/{grand_readable_imgs})")
    print("=" * 60)
    print(f"Reports Generated:")
    print(f"  1. {audit_csv_path}")
    print(f"  2. {landmark_csv_path}")
    print(f"  3. {samples_file}")
    print("=" * 60 + "\n")

    return df_audit, df_landmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 1: Bharatanatyam Mudra Dataset Audit & MediaPipe Hand Landmark Validation",
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
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where CSV reports should be saved",
    )
    parser.add_argument(
        "--samples-file",
        type=Path,
        default=Path("reports/dataset_samples.jpg"),
        help="Output path for the contact sheet image",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=1,
        help="Maximum number of hands to detect (1 for single-hand mudras)",
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
        audit_dataset(
            dataset_root=args.dataset_root,
            model_path=args.model_path,
            reports_dir=args.reports_dir,
            samples_file=args.samples_file,
            num_hands=args.num_hands,
            min_confidence=args.min_confidence,
        )
    except Exception as e:
        logger.exception(f"Audit failed with unhandled exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
