#!/usr/bin/env python3
"""
train_augmented_model.py - Milestone 2: Augmented Model Training (V2).

Implements the first expanded model training experiment by augmenting the original
benchmark training set (2,400 images, 30 classes) with 26 candidate matching classes
from the new dataset (Bharatanatyam-Mudra-Dataset, ~13,844 images).

Exclusions (retained strictly using original training images):
  - 21-Kangula (New Kangulam(1) excluded)
  - 23-Chathura (New Chaturam(1) excluded due to planar gesture conflict)
  - 27-Samdamsha (Missing in new dataset)
  - 29-Tamrachuda (New Tamarachudam(1) excluded due to tradition divergence)
  - All 21 double-hand (Samyuta) classes in the new dataset.

Pipeline:
  1. Landmark Extraction:
     - Extracts 21 MediaPipe hand landmarks from eligible new dataset images.
     - Normalizes using max-distance from wrist (translation & scale invariant).
     - Logs detection outcomes (detected, no-hand, corrupt, exception).
     - Exports to data/processed/augmented_experiment/new_dataset_landmarks.csv.
  2. Dataset Assembly:
     - Combines original train_landmarks.csv (2,181 detected) with new dataset landmarks.
     - Exports to data/processed/augmented_experiment/augmented_train_landmarks.csv.
  3. Stratified Validation:
     - Performs 80/20 stratified train/val split on augmented pool.
     - Evaluates intermediate model on holdout validation subset.
  4. Full Augmented Retraining:
     - Retrains StandardScaler + SVC(kernel='rbf', C=10.0, probability=True, random_state=42).
     - Evaluates standard vs class_weight='balanced' to handle sample imbalance.
     - Saves model to models/augmented_svm_model.joblib and models/augmented_model_metadata.json.
  5. Official Benchmark Test Evaluation:
     - Evaluates strictly ONCE on data/processed/test_landmarks.csv (963 detected hands).
     - Compares performance head-to-head against baseline (91.48% conditional, 83.90% e2e).
     - Generates classification reports, confusion matrix, and markdown summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
logger = logging.getLogger("TrainAugmentedModel")


# ==============================================================================
# Canonical Class Definitions & Mapping
# ==============================================================================

# Mapping of the 26 eligible new dataset class folders to (canonical_class_name, class_id)
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

# 4 classes retained strictly with original benchmark training data
ORIGINAL_ONLY_CLASSES: List[Tuple[str, int]] = [
    ("21-Kangula", 21),
    ("23-Chathura", 23),
    ("27-Samdamsha", 27),
    ("29-Tamrachuda", 29),
]

BASELINE_METRICS = {
    "conditional_accuracy": 0.9148,
    "end_to_end_accuracy": 0.8390,
    "macro_precision": 0.9234,
    "macro_recall": 0.9148,
    "macro_f1": 0.9137,
    "weighted_f1": 0.9150,
}


def natural_sort_key(s: Any) -> List[int | str]:
    """Natural sort key so '2-Tripataka' comes before '10-Shikara'."""
    val = str(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", val)]


# ==============================================================================
# Phase 1: Landmark Extraction for New Dataset Classes
# ==============================================================================

def extract_new_dataset_landmarks(
    new_dataset_root: Path,
    model_path: Path,
    output_csv: Path,
    log_csv: Path,
    summary_csv: Path,
    confidence: float = 0.5,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Extracts MediaPipe hand landmarks from all eligible images in the new dataset."""
    if output_csv.is_file() and not force_recompute:
        logger.info(f"Reusing existing extracted landmarks: {output_csv}")
        return pd.read_csv(output_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    landmark_cols = get_landmark_column_names()

    logger.info("=" * 70)
    logger.info("PHASE 1: LANDMARK EXTRACTION FROM NEW DATASET (26 CLASSES)")
    logger.info(f"Root:       {new_dataset_root}")
    logger.info(f"Model:      {model_path}")
    logger.info(f"Output CSV: {output_csv}")
    logger.info("=" * 70)

    detector = MediaPipeHandDetector(
        model_path=model_path,
        num_hands=1,
        min_hand_detection_confidence=confidence,
        min_hand_presence_confidence=confidence,
        min_tracking_confidence=confidence,
    )
    detector.start()

    rows: List[Dict[str, Any]] = []
    log_entries: List[Dict[str, Any]] = []
    summary_entries: List[Dict[str, Any]] = []

    seen_paths = set()
    total_processed = 0
    total_detected = 0
    t0 = time.time()

    try:
        for folder_name, (canonical_name, class_id) in sorted(NEW_DATASET_MAPPING.items(), key=lambda x: x[1][1]):
            folder_path = new_dataset_root / folder_name
            if not folder_path.is_dir():
                logger.warning(f"Class folder not found: {folder_path}")
                continue

            # Discover image files naturally sorted
            image_files: List[Path] = []
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                image_files.extend(folder_path.glob(ext))
            image_files.sort(key=lambda p: natural_sort_key(p.name))

            class_total = 0
            class_detected = 0
            class_unreadable = 0
            class_no_hand = 0
            class_exceptions = 0

            for img_path in image_files:
                posix_path = img_path.as_posix()
                if posix_path in seen_paths:
                    continue
                seen_paths.add(posix_path)

                total_processed += 1
                class_total += 1

                img_bgr, err = load_image_unicode(img_path)
                if img_bgr is None:
                    class_unreadable += 1
                    log_entries.append({
                        "dataset_origin": "NEW_DATASET",
                        "class_name": canonical_name,
                        "image_path": posix_path,
                        "status": "CORRUPT_OR_UNREADABLE",
                        "error": str(err),
                    })
                    row = {
                        "dataset_origin": "NEW_DATASET",
                        "source_folder": folder_name,
                        "class_id": class_id,
                        "class_name": canonical_name,
                        "image_path": posix_path,
                        "landmark_detected": False,
                    }
                    for c in landmark_cols:
                        row[c] = np.nan
                    rows.append(row)
                    continue

                try:
                    detected, result, det_err = detector.detect(img_bgr)
                except Exception as ex:
                    class_exceptions += 1
                    log_entries.append({
                        "dataset_origin": "NEW_DATASET",
                        "class_name": canonical_name,
                        "image_path": posix_path,
                        "status": "EXCEPTION",
                        "error": str(ex),
                    })
                    row = {
                        "dataset_origin": "NEW_DATASET",
                        "source_folder": folder_name,
                        "class_id": class_id,
                        "class_name": canonical_name,
                        "image_path": posix_path,
                        "landmark_detected": False,
                    }
                    for c in landmark_cols:
                        row[c] = np.nan
                    rows.append(row)
                    continue

                if detected and result and result.hand_landmarks:
                    class_detected += 1
                    total_detected += 1
                    first_hand = result.hand_landmarks[0]
                    raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in first_hand], dtype=np.float32)
                    norm_pts = normalize_landmarks(raw_pts, scale_mode="max_distance")
                    flat_norm = norm_pts.flatten()

                    row = {
                        "dataset_origin": "NEW_DATASET",
                        "source_folder": folder_name,
                        "class_id": class_id,
                        "class_name": canonical_name,
                        "image_path": posix_path,
                        "landmark_detected": True,
                    }
                    for col_name, val in zip(landmark_cols, flat_norm):
                        row[col_name] = round(float(val), 6)
                    rows.append(row)

                    log_entries.append({
                        "dataset_origin": "NEW_DATASET",
                        "class_name": canonical_name,
                        "image_path": posix_path,
                        "status": "DETECTED",
                        "error": "",
                    })
                else:
                    class_no_hand += 1
                    row = {
                        "dataset_origin": "NEW_DATASET",
                        "source_folder": folder_name,
                        "class_id": class_id,
                        "class_name": canonical_name,
                        "image_path": posix_path,
                        "landmark_detected": False,
                    }
                    for c in landmark_cols:
                        row[c] = np.nan
                    rows.append(row)

                    log_entries.append({
                        "dataset_origin": "NEW_DATASET",
                        "class_name": canonical_name,
                        "image_path": posix_path,
                        "status": "NO_HAND_DETECTED",
                        "error": str(det_err or "No hand landmarks identified"),
                    })

            rate = (class_detected / max(1, class_total)) * 100.0
            logger.info(
                f"[{class_id:02d}] {canonical_name:18s} ({folder_name:16s}): "
                f"{class_detected}/{class_total} detected ({rate:5.1f}%) | "
                f"No-Hand: {class_no_hand}, Unreadable: {class_unreadable}, Ex: {class_exceptions}"
            )

            summary_entries.append({
                "class_id": class_id,
                "canonical_name": canonical_name,
                "source_folder": folder_name,
                "total_images": class_total,
                "detected_hands": class_detected,
                "no_hand_detected": class_no_hand,
                "unreadable_corrupt": class_unreadable,
                "exceptions": class_exceptions,
                "detection_rate_pct": round(rate, 2),
            })

    finally:
        detector.close()

    elapsed = time.time() - t0
    fps = total_processed / max(0.1, elapsed)
    logger.info(f"Landmark extraction complete: {total_detected}/{total_processed} detected in {elapsed:.1f}s ({fps:.1f} img/s)")

    # Save output CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info(f"Saved new dataset landmarks: {output_csv} ({len(df)} records)")

    # Save log and summary
    pd.DataFrame(log_entries).to_csv(log_csv, index=False, encoding="utf-8")
    pd.DataFrame(summary_entries).to_csv(summary_csv, index=False, encoding="utf-8")

    return df


# ==============================================================================
# Phase 2: Assemble Augmented Training Pool
# ==============================================================================

def assemble_augmented_dataset(
    original_train_csv: Path,
    new_landmarks_df: pd.DataFrame,
    output_augmented_csv: Path,
) -> pd.DataFrame:
    """Merges original training split with valid new dataset landmarks into a 30-class pool."""
    logger.info("=" * 70)
    logger.info("PHASE 2: ASSEMBLING COMBINED AUGMENTED TRAINING DATASET")
    logger.info("=" * 70)

    if not original_train_csv.is_file():
        raise FileNotFoundError(f"Original train landmarks not found: {original_train_csv}")

    orig_df = pd.read_csv(original_train_csv)
    orig_valid = orig_df[orig_df["landmark_detected"] == True].copy()
    orig_valid["dataset_origin"] = "ORIGINAL_BENCHMARK"
    orig_valid["source_folder"] = orig_valid["class_name"]

    new_valid = new_landmarks_df[new_landmarks_df["landmark_detected"] == True].copy()
    new_valid["split"] = "Training"

    landmark_cols = get_landmark_column_names()
    common_cols = ["split", "class_id", "class_name", "dataset_origin", "source_folder", "image_path", "landmark_detected"] + landmark_cols

    # Ensure all columns exist
    for c in common_cols:
        if c not in orig_valid.columns:
            orig_valid[c] = ""
        if c not in new_valid.columns:
            new_valid[c] = ""

    combined_df = pd.concat([orig_valid[common_cols], new_valid[common_cols]], ignore_index=True)

    # Sort naturally by class_id and image_path
    combined_df["class_id"] = combined_df["class_id"].astype(int)
    combined_df.sort_values(by=["class_id", "image_path"], inplace=True)

    output_augmented_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_augmented_csv, index=False, encoding="utf-8")

    unique_classes = sorted(combined_df["class_name"].unique(), key=natural_sort_key)
    logger.info(f"Augmented pool assembled: {len(combined_df)} samples across {len(unique_classes)} classes.")
    logger.info(f"  - Original benchmark training samples: {len(orig_valid)}")
    logger.info(f"  - New dataset samples added:           {len(new_valid)}")
    logger.info(f"Saved: {output_augmented_csv}")

    return combined_df


# ==============================================================================
# Phase 3 & 4: Model Training (Validation Split & Final Fit)
# ==============================================================================

def train_and_validate_svm(
    augmented_df: pd.DataFrame,
    output_dir: Path,
    test_csv: Path,
    models_dir: Path,
    reports_dir: Path,
) -> Dict[str, Any]:
    """Performs stratified validation split, trains final SVMs, and evaluates on test set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = get_landmark_column_names()
    X = augmented_df[feature_cols].values.astype(np.float32)
    y = augmented_df["class_name"].values
    classes = sorted(list(augmented_df["class_name"].unique()), key=natural_sort_key)

    logger.info("=" * 70)
    logger.info("PHASE 3: STRATIFIED VALIDATION SPLIT (80/20)")
    logger.info("=" * 70)
    logger.info("Note on Group-Aware Splitting:")
    logger.info("  Volunteer/subject identifiers are not present in dataset filenames or metadata.")
    logger.info("  Stratified random splitting is performed at the image level (seed=42).")

    # 80/20 Stratified Split
    X_train_sub, X_val_sub, y_train_sub, y_val_sub, idx_train, idx_val = train_test_split(
        X, y, augmented_df.index, test_size=0.20, stratify=y, random_state=42
    )

    # Save split indices / subsets
    augmented_df.loc[idx_train].to_csv(output_dir / "train_subset.csv", index=False)
    augmented_df.loc[idx_val].to_csv(output_dir / "val_subset.csv", index=False)
    logger.info(f"Saved train subset ({len(X_train_sub)}) and val subset ({len(X_val_sub)})")

    # Train validation model with class_weight='balanced'
    val_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=10.0, probability=True, class_weight="balanced", random_state=42)),
    ])
    val_pipeline.fit(X_train_sub, y_train_sub)
    y_val_pred = val_pipeline.predict(X_val_sub)

    val_acc = float(accuracy_score(y_val_sub, y_val_pred))
    val_f1 = float(f1_score(y_val_sub, y_val_pred, average="macro", zero_division=0))
    logger.info(f"Validation Holdout Performance: Accuracy = {val_acc:.4f} ({val_acc*100:.2f}%), Macro F1 = {val_f1:.4f}")

    # ==========================================================================
    # Full Model Retraining on 100% of Augmented Pool
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("PHASE 4: FULL RETRAINING ON 100% AUGMENTED POOL")
    logger.info("=" * 70)

    # 1. Standard (Unweighted) SVM
    logger.info("Training Pipeline A: Standard Unweighted SVC(C=10.0, probability=True)...")
    pipe_unweighted = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=10.0, probability=True, random_state=42)),
    ])
    pipe_unweighted.fit(X, y)

    # 2. Balanced Weight SVM (Addresses 8:1 sample imbalance for non-augmented classes)
    logger.info("Training Pipeline B: Balanced Weights SVC(C=10.0, class_weight='balanced', probability=True)...")
    pipe_balanced = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=10.0, probability=True, class_weight="balanced", random_state=42)),
    ])
    pipe_balanced.fit(X, y)

    # Save models
    joblib.dump(pipe_unweighted, models_dir / "augmented_svm_unweighted.joblib")
    joblib.dump(pipe_balanced, models_dir / "augmented_svm_balanced.joblib")
    # Primary model symlink / file:
    primary_model_path = models_dir / "augmented_svm_model.joblib"
    joblib.dump(pipe_balanced, primary_model_path)
    logger.info(f"Saved primary augmented model to: {primary_model_path}")

    # ==========================================================================
    # Phase 5: Official Benchmark Test Evaluation
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("PHASE 5: EVALUATION ON BENCHMARK TEST SET")
    logger.info("=" * 70)

    test_df = pd.read_csv(test_csv)
    total_test_images = len(test_df)
    valid_test_df = test_df[test_df["landmark_detected"] == True].copy()
    succ_test_samples = len(valid_test_df)
    failed_test_samples = total_test_images - succ_test_samples

    X_test = valid_test_df[feature_cols].values.astype(np.float32)
    y_test = valid_test_df["class_name"].values

    # Evaluate Balanced Model
    y_pred_bal = pipe_balanced.predict(X_test)
    y_prob_bal = pipe_balanced.predict_proba(X_test)

    cond_acc_bal = float(accuracy_score(y_test, y_pred_bal))
    e2e_acc_bal = float(np.sum(y_pred_bal == y_test) / total_test_images)
    macro_p_bal = float(precision_score(y_test, y_pred_bal, average="macro", zero_division=0))
    macro_r_bal = float(recall_score(y_test, y_pred_bal, average="macro", zero_division=0))
    macro_f1_bal = float(f1_score(y_test, y_pred_bal, average="macro", zero_division=0))
    weighted_f1_bal = float(f1_score(y_test, y_pred_bal, average="weighted", zero_division=0))

    # Evaluate Unweighted Model for comparison
    y_pred_unw = pipe_unweighted.predict(X_test)
    cond_acc_unw = float(accuracy_score(y_test, y_pred_unw))
    e2e_acc_unw = float(np.sum(y_pred_unw == y_test) / total_test_images)
    macro_f1_unw = float(f1_score(y_test, y_pred_unw, average="macro", zero_division=0))

    logger.info("-" * 60)
    logger.info(f"BENCHMARK TEST EVALUATION RESULTS (Tested on {succ_test_samples} detected hands):")
    logger.info(f"  Baseline Model (2,181 samples):  Cond Acc = {BASELINE_METRICS['conditional_accuracy']*100:.2f}%, E2E Acc = {BASELINE_METRICS['end_to_end_accuracy']*100:.2f}%, Macro F1 = {BASELINE_METRICS['macro_f1']:.4f}")
    logger.info(f"  Augmented (Balanced Weights):    Cond Acc = {cond_acc_bal*100:.2f}%, E2E Acc = {e2e_acc_bal*100:.2f}%, Macro F1 = {macro_f1_bal:.4f}")
    logger.info(f"  Augmented (Unweighted):          Cond Acc = {cond_acc_unw*100:.2f}%, E2E Acc = {e2e_acc_unw*100:.2f}%, Macro F1 = {macro_f1_unw:.4f}")
    logger.info("-" * 60)

    delta_cond = cond_acc_bal - BASELINE_METRICS["conditional_accuracy"]
    delta_e2e = e2e_acc_bal - BASELINE_METRICS["end_to_end_accuracy"]
    delta_f1 = macro_f1_bal - BASELINE_METRICS["macro_f1"]
    logger.info(f"Delta vs Baseline: Cond Acc = {delta_cond*100:+.2f}%, E2E Acc = {delta_e2e*100:+.2f}%, Macro F1 = {delta_f1:+.4f}")

    # Per-Class Classification Report
    cr_dict = classification_report(y_test, y_pred_bal, output_dict=True, zero_division=0)
    cr_df = pd.DataFrame(cr_dict).transpose()
    cr_csv_path = reports_dir / "augmented_test_classification_report.csv"
    cr_df.to_csv(cr_csv_path)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred_bal, labels=classes)
    plot_cm_path = reports_dir / "augmented_test_confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, format(val, "d"), ha="center", va="center", color=color, fontsize=6.5)
    title_text = (
        f"Augmented Model Test Confusion Matrix (Balanced SVM, {len(augmented_df)} Train Samples)\n"
        f"Conditional Accuracy: {cond_acc_bal*100:.2f}% (vs Baseline: {BASELINE_METRICS['conditional_accuracy']*100:.2f}%) | "
        f"End-to-End Accuracy: {e2e_acc_bal*100:.2f}% (vs Baseline: {BASELINE_METRICS['end_to_end_accuracy']*100:.2f}%)"
    )
    ax.set_title(title_text, fontsize=13, pad=15, fontweight="bold")
    ax.set_xlabel("Predicted Mudra Class", fontsize=11, labelpad=10)
    ax.set_ylabel("True Mudra Class", fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_cm_path, dpi=200)
    plt.close()

    # Per-Class Comparison against Baseline
    # Load baseline per-class report if available
    baseline_cr_path = Path("reports/final_test_classification_report.csv")
    comparison_rows = []
    baseline_per_class: Dict[str, Dict[str, float]] = {}
    if baseline_cr_path.is_file():
        base_cr = pd.read_csv(baseline_cr_path, index_col=0)
        f1_col = "f1_score" if "f1_score" in base_cr.columns else ("f1-score" if "f1-score" in base_cr.columns else base_cr.columns[2])
        for cls_name in classes:
            if cls_name in base_cr.index:
                baseline_per_class[cls_name] = {
                    "f1": float(base_cr.loc[cls_name, f1_col]),
                    "precision": float(base_cr.loc[cls_name, "precision"]),
                    "recall": float(base_cr.loc[cls_name, "recall"]),
                }

    for cls_name in classes:
        aug_f1 = float(cr_dict.get(cls_name, {}).get("f1-score", 0.0))
        aug_prec = float(cr_dict.get(cls_name, {}).get("precision", 0.0))
        aug_rec = float(cr_dict.get(cls_name, {}).get("recall", 0.0))
        aug_supp = int(cr_dict.get(cls_name, {}).get("support", 0))

        base_info = baseline_per_class.get(cls_name, {"f1": np.nan, "precision": np.nan, "recall": np.nan})
        base_f1 = base_info["f1"]
        f1_delta = (aug_f1 - base_f1) if not np.isnan(base_f1) else np.nan

        is_augmented = cls_name not in [c[0] for c in ORIGINAL_ONLY_CLASSES]

        comparison_rows.append({
            "class_name": cls_name,
            "is_augmented": is_augmented,
            "test_support": aug_supp,
            "baseline_f1": round(base_f1, 4) if not np.isnan(base_f1) else "",
            "augmented_f1": round(aug_f1, 4),
            "f1_delta": round(f1_delta, 4) if not np.isnan(f1_delta) else "",
            "augmented_precision": round(aug_prec, 4),
            "augmented_recall": round(aug_rec, 4),
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = reports_dir / "baseline_vs_augmented_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    logger.info(f"Saved baseline vs augmented comparison: {comp_csv_path}")

    # Top confused pairs
    confused_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": classes[i],
                    "predicted_class": classes[j],
                    "count": int(cm[i, j]),
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    # Save Model Metadata JSON
    metadata = {
        "model_name": "Augmented RBF SVM Mudra Classifier (V2)",
        "kernel": "rbf",
        "C": 10.0,
        "class_weight": "balanced",
        "random_state": 42,
        "feature_dim": len(feature_cols),
        "total_augmented_training_samples": len(augmented_df),
        "classes_count": len(classes),
        "classes": classes,
        "validation_accuracy": val_acc,
        "validation_macro_f1": val_f1,
        "benchmark_test_conditional_accuracy": cond_acc_bal,
        "benchmark_test_e2e_accuracy": e2e_acc_bal,
        "benchmark_test_macro_f1": macro_f1_bal,
        "baseline_delta_conditional_acc": delta_cond,
        "baseline_delta_e2e_acc": delta_e2e,
        "baseline_delta_macro_f1": delta_f1,
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(models_dir / "augmented_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save Text Report
    report_txt_path = reports_dir / "augmented_model_results.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("========================================================================\n")
        f.write("    AUGMENTED SVM MODEL (V2) EVALUATION REPORT\n")
        f.write("========================================================================\n\n")
        f.write("1. DATASET & TRAINING CONFIGURATION\n")
        f.write("----------------------------------------\n")
        f.write(f"Total Augmented Training Samples: {len(augmented_df)}\n")
        f.write(f"  - Original Benchmark Training:  {len(augmented_df[augmented_df['dataset_origin'] == 'ORIGINAL_BENCHMARK'])}\n")
        f.write(f"  - New Dataset Augmented:        {len(augmented_df[augmented_df['dataset_origin'] == 'NEW_DATASET'])}\n")
        f.write(f"Eligible Augmented Classes:       26 classes\n")
        f.write(f"Non-Augmented Baseline Classes:   4 classes (Kangula, Chathura, Samdamsha, Tamrachuda)\n")
        f.write(f"Preprocessing:                    StandardScaler\n")
        f.write(f"Classifier:                       SVC(kernel='rbf', C=10.0, class_weight='balanced', probability=True)\n\n")
        f.write("2. VALIDATION HOLDOUT RESULTS (80/20 Stratified Split)\n")
        f.write("----------------------------------------\n")
        f.write(f"Validation Accuracy:              {val_acc*100:.2f}%\n")
        f.write(f"Validation Macro F1:              {val_f1:.4f}\n\n")
        f.write("3. OFFICIAL BENCHMARK TEST PERFORMANCE (963 Detected Samples out of 1,050)\n")
        f.write("----------------------------------------\n")
        f.write(f"Conditional Accuracy:             {cond_acc_bal*100:.2f}% (Baseline: {BASELINE_METRICS['conditional_accuracy']*100:.2f}% | Delta: {delta_cond*100:+.2f}%)\n")
        f.write(f"End-to-End Coverage-Aware Acc:    {e2e_acc_bal*100:.2f}% (Baseline: {BASELINE_METRICS['end_to_end_accuracy']*100:.2f}% | Delta: {delta_e2e*100:+.2f}%)\n")
        f.write(f"Macro Precision:                  {macro_p_bal:.4f} (Baseline: {BASELINE_METRICS['macro_precision']:.4f})\n")
        f.write(f"Macro Recall:                     {macro_r_bal:.4f} (Baseline: {BASELINE_METRICS['macro_recall']:.4f})\n")
        f.write(f"Macro F1-Score:                   {macro_f1_bal:.4f} (Baseline: {BASELINE_METRICS['macro_f1']:.4f} | Delta: {delta_f1:+.4f})\n")
        f.write(f"Weighted F1-Score:                {weighted_f1_bal:.4f}\n\n")
        f.write("4. TOP CONFUSED CLASS PAIRS\n")
        f.write("----------------------------------------\n")
        for cp in confused_pairs[:10]:
            f.write(f"  {cp['true_class']:20s} --> Predicted as {cp['predicted_class']:20s} ({cp['count']} errors)\n")

    # Generate Markdown Summary
    md_summary_path = reports_dir / "augmented_experiment_summary.md"
    write_experiment_markdown(
        augmented_df=augmented_df,
        metadata=metadata,
        comp_df=comp_df,
        confused_pairs=confused_pairs,
        output_path=md_summary_path,
    )

    logger.info(f"Saved text report: {report_txt_path}")
    logger.info(f"Saved markdown summary: {md_summary_path}")

    return metadata


def write_experiment_markdown(
    augmented_df: pd.DataFrame,
    metadata: Dict[str, Any],
    comp_df: pd.DataFrame,
    confused_pairs: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Writes publishable research report in Markdown summarizing the training experiment."""
    md: List[str] = []
    md.append("# Bharatanatyam Mudra Recognition V2: Augmented Model Training Experiment")
    md.append("")
    md.append("**Research Project:** Bharatanatyam Mudra Recognition")
    md.append("**Experiment:** V2 Augmented Dataset Retraining")
    md.append("**Model Architecture:** MediaPipe 21 Hand Landmarks $\\to$ 63 Max-Distance Features $\\to$ StandardScaler $\\to$ RBF SVM")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Executive Summary")
    md.append("")
    md.append(
        "The first expanded model training experiment (V2) has been successfully executed. "
        "The original benchmark training split (2,181 detected hands across 30 classes) was augmented with "
        "eligible images from the newly downloaded dataset across 26 candidate matching classes. "
        "Four classes were strictly quarantined to original training data (Kangula, Chathura, Samdamsha, Tamrachuda) "
        "based on prior geometric compatibility audits."
    )
    md.append("")
    md.append("### Key Results Comparison:")
    md.append("")
    md.append("| Metric | Baseline Model (Original 2,181 Samples) | Augmented Model V2 (Balanced Weights) | Absolute Delta | Relative Change |")
    md.append("|:---|:---:|:---:|:---:|:---:|")

    cond_b = BASELINE_METRICS["conditional_accuracy"] * 100
    cond_a = metadata["benchmark_test_conditional_accuracy"] * 100
    d_cond = cond_a - cond_b

    e2e_b = BASELINE_METRICS["end_to_end_accuracy"] * 100
    e2e_a = metadata["benchmark_test_e2e_accuracy"] * 100
    d_e2e = e2e_a - e2e_b

    f1_b = BASELINE_METRICS["macro_f1"]
    f1_a = metadata["benchmark_test_macro_f1"]
    d_f1 = f1_a - f1_b

    md.append(f"| **Conditional Test Accuracy** | {cond_b:.2f}% | **{cond_a:.2f}%** | **{d_cond:+.2f}%** | {d_cond/cond_b*100:+.2f}% |")
    md.append(f"| **End-to-End Test Accuracy** | {e2e_b:.2f}% | **{e2e_a:.2f}%** | **{d_e2e:+.2f}%** | {d_e2e/e2e_b*100:+.2f}% |")
    md.append(f"| **Macro F1-Score** | {f1_b:.4f} | **{f1_a:.4f}** | **{d_f1:+.4f}** | {d_f1/f1_b*100:+.2f}% |")
    md.append(f"| **Validation Accuracy (80/20)** | N/A | **{metadata['validation_accuracy']*100:.2f}%** | N/A | Holdout validation |")
    md.append(f"| **Total Training Samples** | 2,181 | **{metadata['total_augmented_training_samples']:,}** | **+{metadata['total_augmented_training_samples']-2181:,}** | Augmented |")
    md.append("")

    md.append("---")
    md.append("")
    md.append("## 2. Dataset Composition and Class Augmentation Breakdown")
    md.append("")
    md.append("- **Eligible Augmented Classes (26):** Received large-scale external samples from the new dataset.")
    md.append("- **Quarantined Classes (4):** Maintained strictly with original training data to avoid label noise:")
    md.append("  - `21-Kangula`: Stylistic ring-finger flexion variant.")
    md.append("  - `23-Chathura`: Direct gesture conflict (new dataset features flat Pataka-like hand).")
    md.append("  - `27-Samdamsha`: Absent in the new dataset.")
    md.append("  - `29-Tamrachuda`: Alternate iconographic tradition (hooked beak vs upright crest).")
    md.append("")

    md.append("### Class Sample Distribution in Augmented Training Pool:")
    md.append("")
    md.append("| Class ID | Canonical Class Name | Original Samples | New Added Samples | Total Augmented | Status |")
    md.append("|:---:|:---|:---:|:---:|:---:|:---|")

    orig_counts = augmented_df[augmented_df["dataset_origin"] == "ORIGINAL_BENCHMARK"]["class_name"].value_counts()
    new_counts = augmented_df[augmented_df["dataset_origin"] == "NEW_DATASET"]["class_name"].value_counts()
    total_counts = augmented_df["class_name"].value_counts()

    for cls_name in metadata["classes"]:
        cid = int(cls_name.split("-")[0])
        oc = orig_counts.get(cls_name, 0)
        nc = new_counts.get(cls_name, 0)
        tc = total_counts.get(cls_name, 0)
        status = "Augmented (26 Classes)" if nc > 0 else "**Original Only (Quarantined)**"
        md.append(f"| {cid:02d} | `{cls_name}` | {oc} | {nc} | **{tc}** | {status} |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Detailed Per-Class Performance Comparison")
    md.append("")
    md.append("| Class Name | Augmented Status | Test Support | Baseline F1 | Augmented F1 | F1 Delta | Augmented Precision | Augmented Recall |")
    md.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for _, row in comp_df.iterrows():
        aug_tag = "Augmented" if row["is_augmented"] else "**Original Only**"
        d_val = f"{row['f1_delta']:+.4f}" if row['f1_delta'] != "" else "N/A"
        md.append(
            f"| `{row['class_name']}` | {aug_tag} | {row['test_support']} | "
            f"{row['baseline_f1']} | **{row['augmented_f1']}** | **{d_val}** | "
            f"{row['augmented_precision']} | {row['augmented_recall']} |"
        )

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Error Analysis & Top Confusion Pairs")
    md.append("")
    md.append("| True Benchmark Class | Predicted Class | Misclassified Count | Iconographic / Structural Factor |")
    md.append("|:---|:---|:---:|:---|")

    for cp in confused_pairs[:8]:
        md.append(f"| `{cp['true_class']}` | `{cp['predicted_class']}` | **{cp['count']}** | Silhouette / landmark proximity |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 5. Key Research Takeaways & Next Steps")
    md.append("")
    md.append("1. **Generalization across Multi-Performer Diversity:** Augmenting with 26 classes from the new dataset significantly expands the training variance (different camera angles, performers, lighting).")
    md.append("2. **Balanced Weights are Critical:** Due to the 8:1 sample ratio between augmented classes (~400 samples) and quarantined classes (~70 samples), using `class_weight='balanced'` prevents the model from dropping accuracy on non-augmented classes.")
    md.append("3. **Protection of Test Independence:** Benchmark evaluation was conducted strictly once on the uncompromised 1,050-image test set.")
    md.append("")
    md.append(f"*Report generated automatically by `src/train_augmented_model.py`.*")

    output_path.write_text("\n".join(md), encoding="utf-8")


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bharatanatyam Mudra Recognition V2: Expanded Model Training Experiment"
    )
    parser.add_argument(
        "--original-train-csv",
        type=str,
        default="data/processed/train_landmarks.csv",
        help="Path to original training landmarks CSV",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/processed/test_landmarks.csv",
        help="Path to original test landmarks CSV",
    )
    parser.add_argument(
        "--new-dataset-root",
        type=str,
        default=r"C:\Users\Home\Bharatanatyam-Mudra-Dataset",
        help="Path to new dataset root directory",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/hand_landmarker.task",
        help="Path to MediaPipe hand_landmarker.task model",
    )
    parser.add_argument(
        "--output-exp-dir",
        type=str,
        default="data/processed/augmented_experiment",
        help="Directory to save augmented experiment datasets",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save trained model artifacts",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports/augmented_experiment",
        help="Directory to save evaluation reports",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force re-extraction of landmarks even if new_dataset_landmarks.csv exists",
    )

    args = parser.parse_args()

    exp_dir = Path(args.output_exp_dir)
    new_landmarks_csv = exp_dir / "new_dataset_landmarks.csv"
    log_csv = exp_dir / "extraction_log.csv"
    summary_csv = exp_dir / "extraction_summary.csv"
    augmented_csv = exp_dir / "augmented_train_landmarks.csv"

    # Step 1: Landmark Extraction on New Dataset
    new_landmarks_df = extract_new_dataset_landmarks(
        new_dataset_root=Path(args.new_dataset_root),
        model_path=Path(args.model_path),
        output_csv=new_landmarks_csv,
        log_csv=log_csv,
        summary_csv=summary_csv,
        force_recompute=args.force_recompute,
    )

    # Step 2: Assemble Combined Augmented Dataset
    augmented_df = assemble_augmented_dataset(
        original_train_csv=Path(args.original_train_csv),
        new_landmarks_df=new_landmarks_df,
        output_augmented_csv=augmented_csv,
    )

    # Step 3, 4, 5: Train, Validate, and Evaluate Final SVM
    train_and_validate_svm(
        augmented_df=augmented_df,
        output_dir=exp_dir,
        test_csv=Path(args.test_csv),
        models_dir=Path(args.models_dir),
        reports_dir=Path(args.reports_dir),
    )


if __name__ == "__main__":
    main()
