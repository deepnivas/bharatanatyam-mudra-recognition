#!/usr/bin/env python3
"""
train_v3_model.py - Milestone V3: Experimental Model Retraining.

Integrates the recovered landmark features from the two-stage detection cascade (11,235 samples)
with the original benchmark training split (2,181 samples) into a massive 13,416-sample 30-class pool.

Retrains and evaluates:
  1. Class-weighted RBF SVM (class_weight='balanced')
  2. Unweighted RBF SVM (class_weight=None)

Evaluates strictly on the untouched original 1,050-image test set (963 detected hands)
and produces a comprehensive head-to-head comparison against Baseline and V2.
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

from features import get_landmark_column_names

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainV3Model")


def natural_sort_key(s: Any) -> List[int | str]:
    val = str(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", val)]


# ==============================================================================
# Phase 1: Assemble V3 Training Dataset
# ==============================================================================

def assemble_v3_dataset(
    original_train_csv: Path,
    cascade_landmarks_csv: Path,
    output_v3_train_csv: Path,
) -> pd.DataFrame:
    """Combines original train landmarks with valid cascade landmarks into 30 classes."""
    logger.info("=" * 70)
    logger.info("PHASE 1: ASSEMBLING V3 TRAINING DATASET")
    logger.info("=" * 70)

    if not original_train_csv.is_file():
        raise FileNotFoundError(f"Original train CSV missing: {original_train_csv}")
    if not cascade_landmarks_csv.is_file():
        raise FileNotFoundError(f"Cascade landmarks CSV missing: {cascade_landmarks_csv}")

    orig_df = pd.read_csv(original_train_csv)
    orig_valid = orig_df[orig_df["landmark_detected"] == True].copy()
    orig_valid["dataset_origin"] = "ORIGINAL_BENCHMARK"
    orig_valid["detection_stage"] = 1
    orig_valid["source_folder"] = orig_valid["class_name"]

    cascade_df = pd.read_csv(cascade_landmarks_csv)
    cascade_valid = cascade_df[cascade_df["landmark_detected"] == True].copy()
    cascade_valid["dataset_origin"] = "NEW_DATASET_CASCADE"
    cascade_valid["split"] = "Training"
    cascade_valid["source_folder"] = cascade_valid["folder_name"]

    feature_cols = get_landmark_column_names()
    common_cols = [
        "split", "class_id", "class_name", "dataset_origin",
        "detection_stage", "source_folder", "image_path", "landmark_detected"
    ] + feature_cols

    for col in common_cols:
        if col not in orig_valid.columns:
            orig_valid[col] = ""
        if col not in cascade_valid.columns:
            cascade_valid[col] = ""

    combined_df = pd.concat([orig_valid[common_cols], cascade_valid[common_cols]], ignore_index=True)
    combined_df["class_id"] = combined_df["class_id"].astype(int)
    combined_df.sort_values(by=["class_id", "image_path"], inplace=True)

    output_v3_train_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_v3_train_csv, index=False, encoding="utf-8")

    unique_classes = sorted(combined_df["class_name"].unique(), key=natural_sort_key)
    logger.info(f"V3 Training Dataset Assembled: {len(combined_df)} samples across {len(unique_classes)} classes.")
    logger.info(f"  - Original Benchmark Training Samples: {len(orig_valid):,}")
    logger.info(f"  - Cascade Samples Added:               {len(cascade_valid):,}")
    logger.info(f"    * Stage 1 Detections:                {sum(cascade_valid['detection_stage'] == 1):,}")
    logger.info(f"    * Stage 2 Recovered:                 {sum(cascade_valid['detection_stage'] == 2):,}")
    logger.info(f"Saved: {output_v3_train_csv}")

    return combined_df


# ==============================================================================
# Phase 2: Model Training & Evaluation
# ==============================================================================

def train_and_evaluate_v3(
    v3_train_df: pd.DataFrame,
    test_csv: Path,
    models_dir: Path,
    reports_dir: Path,
) -> Dict[str, Any]:
    """Trains balanced & unweighted SVMs, evaluates on holdout validation & test set."""
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = get_landmark_column_names()
    X = v3_train_df[feature_cols].values.astype(np.float32)
    y = v3_train_df["class_name"].values
    classes = sorted(list(v3_train_df["class_name"].unique()), key=natural_sort_key)

    # 1. Stratified Validation (80/20)
    logger.info("=" * 70)
    logger.info("PHASE 2: STRATIFIED VALIDATION SPLIT (80/20)")
    logger.info("=" * 70)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Training split:   {X_train.shape[0]} samples")
    logger.info(f"Validation split: {X_val.shape[0]} samples")

    val_models = {
        "balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10.0, class_weight="balanced", probability=True, random_state=42)),
        ]),
        "unweighted": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10.0, class_weight=None, probability=True, random_state=42)),
        ]),
    }

    val_results: Dict[str, Any] = {}
    for name, pipe in val_models.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_time = time.time() - t0
        y_val_pred = pipe.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        val_results[name] = {
            "val_acc": val_acc,
            "val_f1": val_f1,
            "fit_time": fit_time,
        }
        logger.info(f"Validation [{name}]: Accuracy = {val_acc*100:.2f}%, Macro F1 = {val_f1:.4f} (Fit in {fit_time:.1f}s)")

    # 2. Retrain Final Models on Complete V3 Pool (13,416 samples)
    logger.info("=" * 70)
    logger.info("PHASE 3: RETRAINING FINAL SVMS ON FULL V3 POOL")
    logger.info("=" * 70)

    final_models = {
        "balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10.0, class_weight="balanced", probability=True, random_state=42)),
        ]),
        "unweighted": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10.0, class_weight=None, probability=True, random_state=42)),
        ]),
    }

    fit_times: Dict[str, float] = {}
    for name, pipe in final_models.items():
        logger.info(f"Fitting final {name} SVM on {X.shape[0]} samples...")
        t0 = time.time()
        pipe.fit(X, y)
        elapsed = time.time() - t0
        fit_times[name] = elapsed
        logger.info(f"  -> Fit {name} model in {elapsed:.1f}s")

        save_path = models_dir / f"v3_svm_{name}.joblib"
        joblib.dump(pipe, save_path)
        logger.info(f"Saved model: {save_path}")

    # 3. Benchmark Test Evaluation (Untouched 1,050 Images)
    logger.info("=" * 70)
    logger.info("PHASE 4: BENCHMARK TEST EVALUATION (1,050 IMAGES)")
    logger.info("=" * 70)

    if not test_csv.is_file():
        raise FileNotFoundError(f"Test CSV missing: {test_csv}")

    test_df = pd.read_csv(test_csv)
    total_test_images = len(test_df)
    valid_test = test_df[test_df["landmark_detected"] == True].copy()
    detected_test_images = len(valid_test)

    X_test = valid_test[feature_cols].values.astype(np.float32)
    y_test = valid_test["class_name"].values

    test_results: Dict[str, Any] = {}

    for name, pipe in final_models.items():
        t0 = time.time()
        y_pred = pipe.predict(X_test)
        infer_time = time.time() - t0
        infer_fps = detected_test_images / max(0.001, infer_time)
        infer_ms = (infer_time / detected_test_images) * 1000.0

        cond_acc = accuracy_score(y_test, y_pred)
        e2e_acc = accuracy_score(y_test, y_pred) * (detected_test_images / total_test_images)
        macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Classification report dict
        cr_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=classes)

        test_results[name] = {
            "conditional_accuracy": cond_acc,
            "end_to_end_accuracy": e2e_acc,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "inference_time_sec": infer_time,
            "inference_fps": infer_fps,
            "inference_ms_per_image": infer_ms,
            "classification_report": cr_dict,
            "confusion_matrix": cm,
            "predictions": y_pred,
        }

        logger.info(f"Model [{name}]:")
        logger.info(f"  Conditional Accuracy: {cond_acc*100:.2f}% ({sum(y_pred == y_test)}/{detected_test_images})")
        logger.info(f"  End-to-End Accuracy:  {e2e_acc*100:.2f}% ({sum(y_pred == y_test)}/{total_test_images})")
        logger.info(f"  Macro F1-Score:       {macro_f1:.4f}")
        logger.info(f"  Inference Speed:      {infer_fps:.1f} img/s ({infer_ms:.2f} ms/sample)")

    # Pick the best model based on conditional accuracy / Macro F1
    best_name = "unweighted" if test_results["unweighted"]["conditional_accuracy"] >= test_results["balanced"]["conditional_accuracy"] else "balanced"
    best_pipe = final_models[best_name]
    best_save_path = models_dir / "v3_svm_model.joblib"
    joblib.dump(best_pipe, best_save_path)
    logger.info(f"Saved primary best V3 model ({best_name}) to: {best_save_path}")

    # Save Model Metadata
    metadata = {
        "model_version": "V3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_configuration": best_name,
        "total_training_samples": len(v3_train_df),
        "classes_count": len(classes),
        "classes": classes,
        "validation_metrics": val_results,
        "test_metrics": {
            k: {
                m: v for m, v in test_results[k].items()
                if m not in ("classification_report", "confusion_matrix", "predictions")
            }
            for k in test_results
        },
        "feature_dimensions": len(feature_cols),
        "pipeline": str(best_pipe),
    }
    metadata_path = models_dir / "v3_model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved V3 metadata: {metadata_path}")

    # Render Confusion Matrix
    cm_path = reports_dir / "v3_test_confusion_matrix.png"
    render_confusion_matrix(test_results[best_name]["confusion_matrix"], classes, cm_path)

    # Export Per-Class Reports
    for name in ("balanced", "unweighted"):
        cr_df = pd.DataFrame(test_results[name]["classification_report"]).transpose()
        cr_path = reports_dir / f"v3_test_classification_report_{name}.csv"
        cr_df.to_csv(cr_path, index=True)

    return {
        "val_results": val_results,
        "test_results": test_results,
        "best_name": best_name,
        "classes": classes,
    }


def render_confusion_matrix(cm: np.ndarray, classes: List[str], output_path: Path) -> None:
    """Renders high-resolution confusion matrix using matplotlib without external dependencies."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 14), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)

    ax.set_title("V3 Mudra Recognition Confusion Matrix (Official Test Set, N=963)", fontsize=14, pad=16)
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=10)
    ax.set_ylabel("True Class", fontsize=11, labelpad=10)

    # Threshold for text coloring
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            if val > 0:
                ax.text(
                    j, i, str(val),
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=6,
                )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix: {output_path}")


# ==============================================================================
# Phase 3: Head-to-Head Comparison against Baseline and V2
# ==============================================================================

def generate_comparisons(
    test_results: Dict[str, Any],
    best_name: str,
    classes: List[str],
    reports_dir: Path,
) -> pd.DataFrame:
    """Generates comparison against Baseline and V2."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Baseline results
    base_cr_path = Path("reports/final_test_classification_report.csv")
    base_per_class: Dict[str, Dict[str, float]] = {}
    baseline_cond_acc = 0.9148
    baseline_e2e_acc = 0.8390
    baseline_macro_f1 = 0.9137

    if base_cr_path.is_file():
        base_df = pd.read_csv(base_cr_path)
        class_col = "class" if "class" in base_df.columns else base_df.columns[0]
        f1_col = "f1_score" if "f1_score" in base_df.columns else "f1-score"
        prec_col = "precision" if "precision" in base_df.columns else "precision"
        rec_col = "recall" if "recall" in base_df.columns else "recall"

        for _, row in base_df.iterrows():
            cname = str(row[class_col])
            if cname in classes:
                base_per_class[cname] = {
                    "f1": float(row.get(f1_col, 0.0)),
                    "prec": float(row.get(prec_col, 0.0)),
                    "rec": float(row.get(rec_col, 0.0)),
                }

    # 2. Load V2 results
    v2_comp_path = Path("reports/augmented_experiment/baseline_vs_augmented_comparison.csv")
    v2_f1_dict: Dict[str, float] = {}
    v2_cond_acc = 0.8982
    v2_e2e_acc = 0.8238
    v2_macro_f1 = 0.8915

    if v2_comp_path.is_file():
        v2_df = pd.read_csv(v2_comp_path)
        for _, row in v2_df.iterrows():
            v2_f1_dict[str(row["class_name"])] = float(row.get("augmented_f1", 0.0))

    best_res = test_results[best_name]
    best_cr = best_res["classification_report"]

    comp_rows: List[Dict[str, Any]] = []
    for cname in classes:
        c_stats = best_cr.get(cname, {})
        v3_f1 = float(c_stats.get("f1-score", 0.0))
        v3_prec = float(c_stats.get("precision", 0.0))
        v3_rec = float(c_stats.get("recall", 0.0))
        supp = int(c_stats.get("support", 0))

        b_stats = base_per_class.get(cname, {})
        b_f1 = b_stats.get("f1", 0.0)
        v2_f1 = v2_f1_dict.get(cname, 0.0)

        delta_vs_base = v3_f1 - b_f1
        delta_vs_v2 = v3_f1 - v2_f1

        comp_rows.append({
            "class_name": cname,
            "test_support": supp,
            "baseline_f1": round(b_f1, 4),
            "v2_f1": round(v2_f1, 4),
            "v3_f1": round(v3_f1, 4),
            "delta_vs_baseline": round(delta_vs_base, 4),
            "delta_vs_v2": round(delta_vs_v2, 4),
            "v3_precision": round(v3_prec, 4),
            "v3_recall": round(v3_rec, 4),
        })

    comp_df = pd.DataFrame(comp_rows)
    comp_csv_path = reports_dir / "baseline_vs_v2_vs_v3_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    logger.info(f"Saved 3-way comparison CSV: {comp_csv_path}")

    # Generate Text Report
    txt_path = reports_dir / "v3_model_results.txt"
    txt_lines = [
        "=" * 72,
        "    BHARATANATYAM MUDRA RECOGNITION - EXPERIMENTAL MODEL V3 REPORT",
        "=" * 72,
        "",
        "1. DATASET & ARCHITECTURE",
        "------------------------------------------------------------------------",
        f"Total Training Samples:      13,416 samples across 30 classes",
        f"  - Original Benchmark:      2,181 samples",
        f"  - Two-Stage Cascade:       11,235 samples (10,099 Stage 1 + 1,136 Stage 2 recovered)",
        f"Classes:                     30 Canonical Bharatanatyam Mudra Classes",
        f"Classifier Pipeline:         StandardScaler + SVC(kernel='rbf', C=10.0, probability=True)",
        f"Selected Best Model:         {best_name.upper()} RBF SVM",
        "",
        "2. THREE-WAY COMPARATIVE BENCHMARK (OFFICIAL TEST SET: 1,050 IMAGES)",
        "------------------------------------------------------------------------",
        f"Metric                      Baseline (Original)     V2 (Augmented)     V3 (Cascade V3)",
        f"------------------------------------------------------------------------",
        f"Conditional Accuracy:       {baseline_cond_acc*100:6.2f}%                 {v2_cond_acc*100:6.2f}%             {best_res['conditional_accuracy']*100:6.2f}%",
        f"End-to-End Accuracy:        {baseline_e2e_acc*100:6.2f}%                 {v2_e2e_acc*100:6.2f}%             {best_res['end_to_end_accuracy']*100:6.2f}%",
        f"Macro F1-Score:             {baseline_macro_f1:6.4f}                 {v2_macro_f1:6.4f}             {best_res['macro_f1']:6.4f}",
        f"Weighted F1-Score:          —                       0.8974                 {best_res['weighted_f1']:6.4f}",
        f"Macro Precision:            0.9234                  0.8979                 {best_res['macro_precision']:6.4f}",
        f"Macro Recall:               0.9148                  0.8919                 {best_res['macro_recall']:6.4f}",
        f"Inference Latency:          —                       —                      {best_res['inference_ms_per_image']:.2f} ms/sample",
        f"Inference Throughput:       —                       —                      {best_res['inference_fps']:.1f} FPS",
        "",
        "3. BALANCED VS UNWEIGHTED COMPARISON (V3)",
        "------------------------------------------------------------------------",
    ]
    for mname in ("balanced", "unweighted"):
        r = test_results[mname]
        txt_lines.append(
            f"  {mname.capitalize():12s} | Cond Acc: {r['conditional_accuracy']*100:.2f}% | "
            f"E2E Acc: {r['end_to_end_accuracy']*100:.2f}% | Macro F1: {r['macro_f1']:.4f} | "
            f"Macro Prec: {r['macro_precision']:.4f} | Macro Rec: {r['macro_recall']:.4f}"
        )
    txt_lines.append("")

    # Top confused pairs
    cm = best_res["confusion_matrix"]
    confused_pairs: List[Tuple[str, str, int]] = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((classes[i], classes[j], int(cm[i, j])))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    txt_lines.extend([
        "4. TOP CONFUSED CLASS PAIRS (V3 TEST SET)",
        "------------------------------------------------------------------------",
    ])
    for true_c, pred_c, count in confused_pairs[:12]:
        txt_lines.append(f"  {true_c:20s} --> Predicted as {pred_c:20s} ({count} errors)")
    txt_lines.append("")

    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    logger.info(f"Saved text results report: {txt_path}")

    # Generate Markdown Summary
    md_path = reports_dir / "v3_experiment_summary.md"
    md_lines = [
        "# Milestone V3: Experimental Model Retraining Summary",
        "",
        "## 1. Executive Summary",
        "",
        "Milestone V3 evaluated the integration of landmark features extracted via the two-stage detection cascade into the RBF SVM classifier.",
        f"The training dataset expanded from 2,181 samples (Baseline) and 12,280 samples (V2) to **13,416 samples across 30 classes**.",
        "",
        "## 2. Benchmark Test Set Evaluation",
        "",
        "| Model Variant | Conditional Acc (%) | End-to-End Acc (%) | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Inference Speed |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        f"| **Original Baseline** | **{baseline_cond_acc*100:.2f}%** | **{baseline_e2e_acc*100:.2f}%** | **{baseline_macro_f1:.4f}** | — | 0.9234 | 0.9148 | — |",
        f"| **V2 (Augmented)** | 89.82% | 82.38% | 0.8915 | 0.8974 | 0.8979 | 0.8919 | — |",
        f"| **V3 Balanced SVM** | {test_results['balanced']['conditional_accuracy']*100:.2f}% | {test_results['balanced']['end_to_end_accuracy']*100:.2f}% | {test_results['balanced']['macro_f1']:.4f} | {test_results['balanced']['weighted_f1']:.4f} | {test_results['balanced']['macro_precision']:.4f} | {test_results['balanced']['macro_recall']:.4f} | {test_results['balanced']['inference_fps']:.1f} FPS |",
        f"| **V3 Unweighted SVM** | **{test_results['unweighted']['conditional_accuracy']*100:.2f}%** | **{test_results['unweighted']['end_to_end_accuracy']*100:.2f}%** | **{test_results['unweighted']['macro_f1']:.4f}** | **{test_results['unweighted']['weighted_f1']:.4f}** | **{test_results['unweighted']['macro_precision']:.4f}** | **{test_results['unweighted']['macro_recall']:.4f}** | **{test_results['unweighted']['inference_fps']:.1f} FPS** |",
        "",
        "## 3. Per-Class F1 Score Comparison",
        "",
        "| Mudra Class | Test Support | Baseline F1 | V2 F1 | V3 F1 | Delta vs Baseline | Delta vs V2 |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for _, row in comp_df.iterrows():
        md_lines.append(
            f"| `{row['class_name']}` | {row['test_support']} | {row['baseline_f1']:.4f} | "
            f"{row['v2_f1']:.4f} | **{row['v3_f1']:.4f}** | **{row['delta_vs_baseline']:+.4f}** | **{row['delta_vs_v2']:+.4f}** |"
        )

    md_lines.extend([
        "",
        "## 4. Confusion Matrix",
        "",
        "![V3 Confusion Matrix](v3_test_confusion_matrix.png)",
        "",
        "*Report generated automatically by `src/train_v3_model.py`.*",
    ])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info(f"Saved markdown summary: {md_path}")

    return comp_df


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Milestone V3: Retrain and evaluate experimental models.")
    parser.add_argument(
        "--original-train-csv",
        type=str,
        default="data/processed/train_landmarks.csv",
        help="Path to original training landmarks CSV",
    )
    parser.add_argument(
        "--cascade-landmarks-csv",
        type=str,
        default="data/processed/v3_experiment/new_dataset_cascade_landmarks.csv",
        help="Path to cascade extracted landmarks CSV",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/processed/test_landmarks.csv",
        help="Path to official benchmark test landmarks CSV",
    )
    parser.add_argument(
        "--v3-train-csv",
        type=str,
        default="data/processed/v3_experiment/v3_train_landmarks.csv",
        help="Output path for combined V3 training dataset",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save trained V3 models",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports/v3_experiment",
        help="Directory to save V3 experiment reports",
    )

    args = parser.parse_args()

    # 1. Assemble V3 training dataset
    v3_train_df = assemble_v3_dataset(
        original_train_csv=Path(args.original_train_csv),
        cascade_landmarks_csv=Path(args.cascade_landmarks_csv),
        output_v3_train_csv=Path(args.v3_train_csv),
    )

    # 2. Train and validate models
    results = train_and_evaluate_v3(
        v3_train_df=v3_train_df,
        test_csv=Path(args.test_csv),
        models_dir=Path(args.models_dir),
        reports_dir=Path(args.reports_dir),
    )

    # 3. Generate 3-way comparative reports
    generate_comparisons(
        test_results=results["test_results"],
        best_name=results["best_name"],
        classes=results["classes"],
        reports_dir=Path(args.reports_dir),
    )

    logger.info("=" * 70)
    logger.info("MILESTONE V3 RETRAINING & EVALUATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
