#!/usr/bin/env python3
"""
train_v5_mlp.py - Milestone V5: Neural Network (MLP) Classifier Training & Benchmark.

Evaluates Multi-Layer Perceptron (MLP) neural networks across different epoch limits
and feature representations on the Bharatanatyam Mudra Recognition project.

Feature Variants:
  - Model A: 63 normalized landmark coordinates (raw landmark baseline).
  - Model B: 98 features (63 coordinates + 35 geometric features: 15 joint angles + 20 distances)
             extracted strictly via extract_v4_features() from src/features.py.

Architecture:
  - Input: 63 or 98 features
  - Hidden Layer 1: 128 neurons (ReLU)
  - Hidden Layer 2: 64 neurons (ReLU)
  - Output: 30 classes (Softmax / Cross-Entropy)
  - Optimizer: Adam (lr=0.001, batch_size=64, random_state=42)

Epoch Experiments:
  - E1 (50 epochs), E2 (100 epochs), E3 (200 epochs), E4 (300 epochs)
  - Evaluated on an 80/20 stratified validation split.
  - Best model selected strictly based on Validation Macro F1.

Final Evaluation:
  - Retrained on complete 13,416-sample training pool.
  - Evaluated once on untouched official 1,050-image test set (963 detected hands).
  - 5-Way Comparison: V1 vs V2 vs V3 vs V4 vs V5 MLP.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add src to sys.path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from features import extract_v4_features, get_landmark_column_names

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainV5MLP")


def natural_sort_key(s: Any) -> List[int | str]:
    val = str(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", val)]


# ==============================================================================
# Feature Extraction
# ==============================================================================

def extract_dataset_features(
    df: pd.DataFrame,
    feature_config: str,
) -> Tuple[np.ndarray, List[str]]:
    """Extracts features using existing extract_v4_features() from src/features.py.
    
    Args:
        df: DataFrame containing 63 landmark coordinates (x1..z21).
        feature_config: 'control' (Model A, 63 coords) or 'angles_distances' (Model B, 98 features).
        
    Returns:
        X: (N, D) numpy array
        feature_names: List of D feature names
    """
    logger.info(f"Extracting features for configuration '{feature_config}'...")
    raw_cols = get_landmark_column_names()
    raw_coords = df[raw_cols].values.astype(np.float32)
    n_samples = len(df)
    pts_all = raw_coords.reshape(n_samples, 21, 3)

    features_list: List[np.ndarray] = []
    for i in range(n_samples):
        feat = extract_v4_features(pts_all[i], feature_config=feature_config)
        features_list.append(feat)

    X = np.array(features_list, dtype=np.float32)
    logger.info(f"Extracted feature matrix: shape {X.shape}")
    return X


# ==============================================================================
# Phase 1: 80/20 Stratified Validation & Epoch Sweep (8 Experiments)
# ==============================================================================

def run_epoch_experiments(
    X_train_full_A: np.ndarray,
    X_train_full_B: np.ndarray,
    y_train_full: np.ndarray,
    epochs_list: List[int],
    models_dir: Path,
    reports_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, List[float]], str, Dict[str, Any]]:
    """Trains 8 experimental MLP models across 4 epoch settings for Model A and Model B.
    
    Uses an 80/20 stratified validation split of the training pool.
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: 80/20 STRATIFIED VALIDATION & EPOCH EXPERIMENTS")
    logger.info("=" * 70)
    logger.info("Validation Strategy: Image-level stratified 80/20 train/val split.")
    logger.info("(Dancer/performer identity labels are not annotated in dataset metadata).")

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Split indices deterministically
    indices = np.arange(len(y_train_full))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full,
    )
    logger.info(f"Training split:   {len(train_idx)} samples")
    logger.info(f"Validation split: {len(val_idx)} samples")

    y_train_split = y_train_full[train_idx]
    y_val_split = y_train_full[val_idx]
    classes = sorted(list(np.unique(y_train_full)), key=natural_sort_key)

    variants = [
        ("Model_A_Coords63", X_train_full_A, 63),
        ("Model_B_Geometric98", X_train_full_B, 98),
    ]

    experiment_records: List[Dict[str, Any]] = []
    loss_curves: Dict[str, List[float]] = {}
    validation_models: Dict[str, Pipeline] = {}
    val_cr_records: Dict[str, Dict[str, Any]] = {}

    for var_name, X_full, n_feats in variants:
        X_train_sub = X_full[train_idx]
        X_val_sub = X_full[val_idx]

        for epochs in epochs_list:
            exp_id = f"{var_name}_E{epochs}"
            logger.info(f"Running [{exp_id}] ({n_feats} features, {epochs} max epochs)...")

            # Create pipeline: StandardScaler fitted ONLY on train portion
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=0.001,
                        batch_size=64,
                        random_state=42,
                        max_iter=epochs,
                        tol=1e-9,
                        n_iter_no_change=epochs,
                        early_stopping=False,
                    ),
                ),
            ])

            t0 = time.time()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                pipe.fit(X_train_sub, y_train_split)
            fit_time = time.time() - t0

            mlp_clf: MLPClassifier = pipe.named_steps["mlp"]
            actual_iters = mlp_clf.n_iter_
            loss_history = list(mlp_clf.loss_curve_)
            final_loss = float(mlp_clf.loss_)
            loss_curves[exp_id] = loss_history

            # Validation inference
            t0 = time.time()
            y_val_pred = pipe.predict(X_val_sub)
            infer_time = time.time() - t0
            val_fps = len(y_val_pred) / max(0.0001, infer_time)

            val_acc = accuracy_score(y_val_split, y_val_pred)
            val_macro_prec = precision_score(y_val_split, y_val_pred, average="macro", zero_division=0)
            val_macro_rec = recall_score(y_val_split, y_val_pred, average="macro", zero_division=0)
            val_macro_f1 = f1_score(y_val_split, y_val_pred, average="macro", zero_division=0)
            val_weighted_f1 = f1_score(y_val_split, y_val_pred, average="weighted", zero_division=0)

            cr_dict = classification_report(y_val_split, y_val_pred, output_dict=True, zero_division=0)
            val_cr_records[exp_id] = cr_dict
            validation_models[exp_id] = pipe

            # Save individual experimental model
            model_save_path = models_dir / f"mlp_{var_name}_e{epochs}.joblib"
            joblib.dump(pipe, model_save_path)

            logger.info(
                f"  -> {exp_id}: Val Acc={val_acc*100:.2f}% | Val Macro F1={val_macro_f1:.4f} | "
                f"Final Loss={final_loss:.4f} | Actual Iterations={actual_iters}/{epochs} | Fit Time={fit_time:.1f}s"
            )

            experiment_records.append({
                "experiment_id": exp_id,
                "variant": var_name,
                "features_used": n_feats,
                "configured_max_epochs": epochs,
                "actual_epochs_trained": actual_iters,
                "final_training_loss": round(final_loss, 5),
                "validation_accuracy_pct": round(val_acc * 100.0, 2),
                "validation_macro_precision": round(val_macro_prec, 4),
                "validation_macro_recall": round(val_macro_rec, 4),
                "validation_macro_f1": round(val_macro_f1, 4),
                "validation_weighted_f1": round(val_weighted_f1, 4),
                "fit_time_seconds": round(fit_time, 2),
                "val_inference_fps": round(val_fps, 1),
                "model_path": str(model_save_path),
            })

    exp_df = pd.DataFrame(experiment_records)
    # Sort strictly by Validation Macro F1 descending
    exp_df.sort_values(by="validation_macro_f1", ascending=False, inplace=True)

    summary_csv = reports_dir / "mlp_epoch_experiments_summary.csv"
    reports_dir.mkdir(parents=True, exist_ok=True)
    exp_df.to_csv(summary_csv, index=False)
    logger.info(f"Saved epoch comparison summary CSV: {summary_csv}")

    # Export per-class validation metrics for all 8 experiments
    val_per_class_rows: List[Dict[str, Any]] = []
    for exp_id, cr_dict in val_cr_records.items():
        for cname in classes:
            c_metrics = cr_dict.get(cname, {})
            val_per_class_rows.append({
                "experiment_id": exp_id,
                "class_name": cname,
                "precision": round(float(c_metrics.get("precision", 0.0)), 4),
                "recall": round(float(c_metrics.get("recall", 0.0)), 4),
                "f1_score": round(float(c_metrics.get("f1-score", 0.0)), 4),
                "validation_support": int(c_metrics.get("support", 0)),
            })
    val_pc_df = pd.DataFrame(val_per_class_rows)
    val_pc_csv = reports_dir / "mlp_validation_metrics.csv"
    val_pc_df.to_csv(val_pc_csv, index=False)
    logger.info(f"Saved per-class validation metrics CSV: {val_pc_csv}")

    # Identify Winning Model
    best_row = exp_df.iloc[0]
    best_exp_id = str(best_row["experiment_id"])
    logger.info("=" * 70)
    logger.info(f"SELECTED BEST MLP CONFIGURATION (VIA VALIDATION MACRO F1): [{best_exp_id}]")
    logger.info(
        f"  Variant:              {best_row['variant']}\n"
        f"  Features:             {best_row['features_used']}\n"
        f"  Selected Epochs:      {best_row['configured_max_epochs']}\n"
        f"  Validation Macro F1:  {best_row['validation_macro_f1']:.4f}\n"
        f"  Validation Accuracy:  {best_row['validation_accuracy_pct']:.2f}%\n"
        f"  Training Loss:        {best_row['final_training_loss']:.5f}"
    )
    logger.info("=" * 70)

    # Plot training loss curves
    plot_loss_curves(loss_curves, reports_dir / "mlp_loss_curves.png")

    return exp_df, loss_curves, best_exp_id, best_row.to_dict()


def plot_loss_curves(loss_curves: Dict[str, List[float]], output_path: Path) -> None:
    """Plots training loss curves for all 8 MLP experiments in a high-res comparison chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    colors_a = ["#3498db", "#2980b9", "#1f618d", "#154360"]
    colors_b = ["#e67e22", "#d35400", "#ba4a00", "#a04000"]

    epochs_list = [50, 100, 200, 300]

    # Panel 1: Model A (63 Coords)
    for idx, e in enumerate(epochs_list):
        key = f"Model_A_Coords63_E{e}"
        if key in loss_curves:
            curve = loss_curves[key]
            ax1.plot(range(1, len(curve) + 1), curve, label=f"E{e} (Loss={curve[-1]:.4f})", color=colors_a[idx], lw=2)

    ax1.set_title("Model A: 63 Landmark Coordinates (Loss Curves)", fontsize=12, pad=10)
    ax1.set_xlabel("Epoch / Iteration", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Training Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: Model B (98 Geometric)
    for idx, e in enumerate(epochs_list):
        key = f"Model_B_Geometric98_E{e}"
        if key in loss_curves:
            curve = loss_curves[key]
            ax2.plot(range(1, len(curve) + 1), curve, label=f"E{e} (Loss={curve[-1]:.4f})", color=colors_b[idx], lw=2)

    ax2.set_title("Model B: 98 Geometric Features (Loss Curves)", fontsize=12, pad=10)
    ax2.set_xlabel("Epoch / Iteration", fontsize=10)
    ax2.set_ylabel("Cross-Entropy Training Loss", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    plt.suptitle("Milestone V5: MLP Training Loss Convergence Across Epoch Limits", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved loss curves plot: {output_path}")


# ==============================================================================
# Phase 2: Final Retraining on 100% Training Pool & Official Test Benchmark
# ==============================================================================

def run_final_retraining_and_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    total_test_images: int,
    best_info: Dict[str, Any],
    classes: List[str],
    models_dir: Path,
    reports_dir: Path,
) -> Dict[str, Any]:
    """Retrains chosen best MLP configuration on complete 13,416-sample pool and evaluates on test set."""
    logger.info("=" * 70)
    logger.info("PHASE 2: FINAL RETRAINING ON FULL 13,416 TRAINING POOL & BENCHMARK TESTING")
    logger.info("=" * 70)

    is_model_b = "Model_B" in best_info["variant"]
    feat_cfg = "angles_distances" if is_model_b else "control"
    selected_epochs = int(best_info["configured_max_epochs"])

    logger.info(f"Retraining winning configuration: {best_info['variant']} with {selected_epochs} epochs...")

    # Extract features on complete training set and test set
    X_train_full = extract_dataset_features(train_df, feature_config=feat_cfg)
    y_train_full = train_df["class_name"].values

    X_test_valid = extract_dataset_features(test_df, feature_config=feat_cfg)
    y_test_valid = test_df["class_name"].values
    detected_test_count = len(y_test_valid)

    # Instantiate full final pipeline
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                batch_size=64,
                random_state=42,
                max_iter=selected_epochs,
                tol=1e-9,
                n_iter_no_change=selected_epochs,
                early_stopping=False,
            ),
        ),
    ])

    # Fit final pipeline on 100% of training data
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        final_pipeline.fit(X_train_full, y_train_full)
    final_fit_time = time.time() - t0

    mlp_clf: MLPClassifier = final_pipeline.named_steps["mlp"]
    final_train_loss = float(mlp_clf.loss_)
    final_actual_iters = mlp_clf.n_iter_

    # Save final selected model
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = models_dir / "v5_selected_mlp_model.joblib"
    joblib.dump(final_pipeline, final_model_path)
    model_size_bytes = os.path.getsize(final_model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)

    logger.info(
        f"Final fit completed in {final_fit_time:.1f}s | Final Loss: {final_train_loss:.5f} | "
        f"Actual Iterations: {final_actual_iters}/{selected_epochs}"
    )
    logger.info(f"Saved final selected MLP model: {final_model_path} ({model_size_mb:.2f} MB)")

    # --------------------------------------------------------------------------
    # Official Test Set Evaluation (963 detected hands out of 1,050)
    # --------------------------------------------------------------------------
    logger.info(f"Evaluating final MLP on untouched test set ({detected_test_count} detected hands)...")

    # Measure classifier-only inference latency
    t0 = time.time()
    y_test_pred = final_pipeline.predict(X_test_valid)
    infer_time = time.time() - t0
    y_test_proba = final_pipeline.predict_proba(X_test_valid)

    infer_fps = detected_test_count / max(0.0001, infer_time)
    latency_ms = (infer_time / detected_test_count) * 1000.0

    # Primary performance metrics
    cond_acc = accuracy_score(y_test_valid, y_test_pred)
    # End-to-end accuracy: strictly counting 87 missed hand detections as errors
    correct_count = sum(y_test_pred == y_test_valid)
    e2e_acc = correct_count / total_test_images

    macro_prec = precision_score(y_test_valid, y_test_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test_valid, y_test_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test_valid, y_test_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test_valid, y_test_pred, average="weighted", zero_division=0)

    logger.info("=" * 70)
    logger.info("V5 MLP TEST SET EVALUATION RESULTS:")
    logger.info(f"  Conditional Accuracy: {cond_acc*100:.2f}% ({correct_count}/{detected_test_count})")
    logger.info(f"  End-to-End Accuracy:  {e2e_acc*100:.2f}% ({correct_count}/{total_test_images})")
    logger.info(f"  Macro Precision:      {macro_prec:.4f}")
    logger.info(f"  Macro Recall:         {macro_rec:.4f}")
    logger.info(f"  Macro F1-Score:       {macro_f1:.4f}")
    logger.info(f"  Weighted F1-Score:    {weighted_f1:.4f}")
    logger.info(f"  Classifier Latency:   {latency_ms:.2f} ms/sample ({infer_fps:.1f} FPS)")
    logger.info("=" * 70)

    # Classification report & Confusion matrix
    cr_dict = classification_report(y_test_valid, y_test_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test_valid, y_test_pred, labels=classes)

    # Render confusion matrix
    cm_path = reports_dir / "v5_test_confusion_matrix.png"
    render_confusion_matrix(cm, classes, cm_path)

    # Save Model Metadata
    metadata = {
        "model_version": "V5_MLP",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_experiment_id": best_info["experiment_id"],
        "architecture": {
            "input_features": int(best_info["features_used"]),
            "hidden_layers": [128, 64],
            "activation": "relu",
            "solver": "adam",
            "learning_rate_init": 0.001,
            "batch_size": 64,
            "configured_epochs": selected_epochs,
            "actual_epochs_trained": final_actual_iters,
            "final_training_loss": float(final_train_loss),
        },
        "dataset_sizes": {
            "training_samples": len(X_train_full),
            "test_detected_samples": detected_test_count,
            "test_total_samples": total_test_images,
        },
        "validation_metrics": {
            "validation_accuracy_pct": float(best_info["validation_accuracy_pct"]),
            "validation_macro_f1": float(best_info["validation_macro_f1"]),
            "validation_weighted_f1": float(best_info["validation_weighted_f1"]),
        },
        "test_metrics": {
            "conditional_accuracy_pct": round(float(cond_acc) * 100.0, 2),
            "end_to_end_accuracy_pct": round(float(e2e_acc) * 100.0, 2),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "classifier_latency_ms": round(float(latency_ms), 3),
            "classifier_inference_fps": round(float(infer_fps), 1),
            "model_size_mb": round(float(model_size_mb), 2),
        },
        "pipeline": str(final_pipeline),
    }
    metadata_path = models_dir / "v5_model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved V5 model metadata: {metadata_path}")

    # Build 5-Way Milestone Progression Table
    bench_rows = [
        {
            "model_generation": "V1_Original_Baseline",
            "classifier_type": "RBF SVM (C=10.0)",
            "features_used": "63 raw coordinates",
            "feature_dim": 63,
            "training_samples": 2181,
            "conditional_acc_pct": 91.48,
            "e2e_acc_pct": 83.90,
            "macro_precision": 0.9234,
            "macro_recall": 0.9148,
            "macro_f1": 0.9137,
            "latency_ms": 0.35,
            "inference_fps": 2850.0,
            "model_size_mb": 0.58,
        },
        {
            "model_generation": "V2_Augmented_Baseline",
            "classifier_type": "RBF SVM (Balanced)",
            "features_used": "63 raw coordinates",
            "feature_dim": 63,
            "training_samples": 12280,
            "conditional_acc_pct": 89.82,
            "e2e_acc_pct": 82.38,
            "macro_precision": 0.8979,
            "macro_recall": 0.8919,
            "macro_f1": 0.8915,
            "latency_ms": 0.36,
            "inference_fps": 2750.0,
            "model_size_mb": 2.84,
        },
        {
            "model_generation": "V3_Cascade_Control",
            "classifier_type": "RBF SVM (Unweighted)",
            "features_used": "63 raw coordinates",
            "feature_dim": 63,
            "training_samples": 13416,
            "conditional_acc_pct": 90.97,
            "e2e_acc_pct": 83.43,
            "macro_precision": 0.9103,
            "macro_recall": 0.9049,
            "macro_f1": 0.9037,
            "latency_ms": 0.35,
            "inference_fps": 2873.9,
            "model_size_mb": 3.05,
        },
        {
            "model_generation": "V4_Geometric_SVM",
            "classifier_type": "RBF SVM (C=15.0)",
            "features_used": "98 angles + distances",
            "feature_dim": 98,
            "training_samples": 13416,
            "conditional_acc_pct": 90.97,
            "e2e_acc_pct": 83.43,
            "macro_precision": 0.9099,
            "macro_recall": 0.9052,
            "macro_f1": 0.9033,
            "latency_ms": 0.31,
            "inference_fps": 3180.7,
            "model_size_mb": 5.54,
        },
        {
            "model_generation": f"V5_MLP_{best_info['variant']}_E{selected_epochs}",
            "classifier_type": f"MLP (128-64, Adam, {selected_epochs}ep)",
            "features_used": f"{best_info['features_used']} features",
            "feature_dim": int(best_info["features_used"]),
            "training_samples": 13416,
            "conditional_acc_pct": round(float(cond_acc) * 100.0, 2),
            "e2e_acc_pct": round(float(e2e_acc) * 100.0, 2),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "macro_f1": round(float(macro_f1), 4),
            "latency_ms": round(float(latency_ms), 3),
            "inference_fps": round(float(infer_fps), 1),
            "model_size_mb": round(float(model_size_mb), 2),
        },
    ]
    bench_df = pd.DataFrame(bench_rows)
    bench_csv = reports_dir / "v1_to_v5_benchmark_comparison.csv"
    bench_df.to_csv(bench_csv, index=False)
    logger.info(f"Saved 5-way benchmark comparison CSV: {bench_csv}")

    # Build Per-Class F1 Comparison Table
    base_cr_path = Path("reports/final_test_classification_report.csv")
    base_f1_dict: Dict[str, float] = {}
    if base_cr_path.is_file():
        b_df = pd.read_csv(base_cr_path)
        c_col = "class" if "class" in b_df.columns else b_df.columns[0]
        f_col = "f1_score" if "f1_score" in b_df.columns else "f1-score"
        for _, r in b_df.iterrows():
            base_f1_dict[str(r[c_col])] = float(r.get(f_col, 0.0))

    v4_comp_path = Path("reports/v4_experiment/v4_per_class_f1_comparison.csv")
    v4_f1_dict: Dict[str, float] = {}
    if v4_comp_path.is_file():
        v4_df = pd.read_csv(v4_comp_path)
        for _, r in v4_df.iterrows():
            v4_f1_dict[str(r["class_name"])] = float(r.get("v4_f1", 0.0))

    per_class_rows: List[Dict[str, Any]] = []
    for cname in classes:
        c_stats = cr_dict.get(cname, {})
        v5_f1 = float(c_stats.get("f1-score", 0.0))
        v5_prec = float(c_stats.get("precision", 0.0))
        v5_rec = float(c_stats.get("recall", 0.0))
        supp = int(c_stats.get("support", 0))

        b_f1 = base_f1_dict.get(cname, 0.0)
        v4_f1 = v4_f1_dict.get(cname, 0.0)

        per_class_rows.append({
            "class_name": cname,
            "support": supp,
            "baseline_f1": round(b_f1, 4),
            "v4_f1": round(v4_f1, 4),
            "v5_mlp_f1": round(v5_f1, 4),
            "delta_vs_baseline": round(v5_f1 - b_f1, 4),
            "delta_vs_v4": round(v5_f1 - v4_f1, 4),
            "v5_precision": round(v5_prec, 4),
            "v5_recall": round(v5_rec, 4),
        })
    per_class_df = pd.DataFrame(per_class_rows)

    # Write Markdown Research Report
    report_md_path = reports_dir / "v5_mlp_experiment_report.md"
    write_v5_markdown_report(
        bench_df=bench_df,
        per_class_df=per_class_df,
        best_info=best_info,
        selected_epochs=selected_epochs,
        final_actual_iters=final_actual_iters,
        final_train_loss=final_train_loss,
        output_path=report_md_path,
    )
    logger.info(f"Saved V5 research report: {report_md_path}")

    return {
        "bench_df": bench_df,
        "per_class_df": per_class_df,
        "cond_acc": cond_acc,
        "e2e_acc": e2e_acc,
        "macro_f1": macro_f1,
    }


def render_confusion_matrix(cm: np.ndarray, classes: List[str], output_path: Path) -> None:
    """Renders high-resolution confusion matrix for V5 MLP."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 14), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)

    ax.set_title("V5 MLP Neural Network Confusion Matrix (Official Test Set, N=963)", fontsize=14, pad=16)
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=10)
    ax.set_ylabel("True Class", fontsize=11, labelpad=10)

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


def write_v5_markdown_report(
    bench_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    best_info: Dict[str, Any],
    selected_epochs: int,
    final_actual_iters: int,
    final_train_loss: float,
    output_path: Path,
) -> None:
    """Generates comprehensive Markdown report for Milestone V5."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Milestone V5 — Neural Network (MLP) Classifier Research Report",
        "",
        "**Research Project:** Bharatanatyam Mudra Recognition  ",
        "**Milestone:** V5 — Multi-Layer Perceptron (MLP) Neural Network Training & Epoch Benchmark  ",
        "**Script:** `src/train_v5_mlp.py`  ",
        f"**Selected Winning Configuration:** `{best_info['experiment_id']}`  ",
        f"**Configured Epochs:** {selected_epochs} (Actual Epochs Run: {final_actual_iters})  ",
        f"**Final Retraining Loss:** {final_train_loss:.5f}  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Five-Generation Milestone Comparison",
        "",
        "Milestone V5 evaluated whether deep feedforward neural architectures (MLPClassifier) trained across",
        "varying epoch limits (50, 100, 200, 300) can outperform the established RBF SVM models on normalized",
        "hand landmark coordinates (Model A, 63 features) and engineered geometric features (Model B, 98 features).",
        "",
        "### Complete Five-Generation Progression Table",
        "",
        "| Generation | Architecture & Features | Training Samples | Conditional Acc (%) | End-to-End Acc (%) | Macro F1 | Inference Speed | Model Size |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for _, row in bench_df.iterrows():
        b_flag = " **(V5 MLP Champion)**" if "V5" in row["model_generation"] else ""
        lines.append(
            f"| **{row['model_generation']}**{b_flag} | {row['features_used']} ({row['classifier_type']}) | "
            f"{row['training_samples']:,} | **{row['conditional_acc_pct']:.2f}%** | "
            f"**{row['e2e_acc_pct']:.2f}%** | **{row['macro_f1']:.4f}** | "
            f"{row['inference_fps']:.1f} FPS ({row['latency_ms']:.2f} ms) | {row['model_size_mb']:.2f} MB |"
        )

    lines.extend([
        "",
        "> [!NOTE]",
        "> **Classifier Latency vs Webcam Pipeline:**",
        "> The inference latency reported above represents raw classifier feedforward time ($0.1–0.4$ ms per sample,",
        "> corresponding to $>2,500$ FPS). Real-time webcam pipeline throughput is constrained by MediaPipe landmark",
        "> detection (~30–50 FPS), which is measured independently in deployment testing.",
        "",
        "---",
        "",
        "## 2. Validation Methodology & Model Selection",
        "",
        "- **Validation Strategy:** 80/20 Image-Level Stratified Train/Val Split (10,732 training / 2,684 validation samples).",
        "- **Performer Independence:** Dataset metadata does not include individual dancer IDs or session tags; therefore,",
        "  validation is strictly image-level stratified by class distribution.",
        "- **Selection Criterion:** Winning model was selected strictly on **Validation Macro F1** across the 8 experiments,",
        "  ensuring zero test-set contamination.",
        "",
        "### Epoch Experiments Convergence Curves",
        "",
        "![MLP Loss Curves](mlp_loss_curves.png)",
        "",
        "---",
        "",
        "## 3. Official Benchmark Test Set Per-Class Results",
        "",
        "| Mudra Class | Test Support | Baseline F1 (V1) | V4 F1 | V5 MLP F1 | Delta vs Baseline | Delta vs V4 | V5 Precision | V5 Recall |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for _, row in per_class_df.iterrows():
        lines.append(
            f"| `{row['class_name']}` | {row['support']} | {row['baseline_f1']:.4f} | "
            f"{row['v4_f1']:.4f} | **{row['v5_mlp_f1']:.4f}** | **{row['delta_vs_baseline']:+.4f}** | "
            f"**{row['delta_vs_v4']:+.4f}** | {row['v5_precision']:.4f} | {row['v5_recall']:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Test Set Confusion Matrix",
        "",
        "![V5 Confusion Matrix](v5_test_confusion_matrix.png)",
        "",
        "*Report generated automatically by `src/train_v5_mlp.py`.*",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Milestone V5: Neural Network (MLP) Classifier Training.")
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/processed/v3_experiment/v3_train_landmarks.csv",
        help="Path to V3 combined training landmarks CSV",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/processed/test_landmarks.csv",
        help="Path to official benchmark test landmarks CSV",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/v5_experiment",
        help="Directory to save V5 experimental models",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports/v5_experiment",
        help="Directory to save V5 reports and plots",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default="50,100,200,300",
        help="Comma-separated epoch limits to test (default: 50,100,200,300)",
    )

    args = parser.parse_args()

    train_path = Path(args.train_csv)
    test_path = Path(args.test_csv)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    epochs_list = [int(e.strip()) for e in args.epochs.split(",") if e.strip()]

    # 1. Load data
    train_df = pd.read_csv(train_path)
    train_valid = train_df[train_df["landmark_detected"] == True].copy()
    y_train = train_valid["class_name"].values
    classes = sorted(list(train_valid["class_name"].unique()), key=natural_sort_key)

    test_df = pd.read_csv(test_path)
    total_test_images = len(test_df)
    test_valid = test_df[test_df["landmark_detected"] == True].copy()

    # 2. Extract features
    X_train_A = extract_dataset_features(train_valid, feature_config="control")
    X_train_B = extract_dataset_features(train_valid, feature_config="angles_distances")

    # 3. Phase 1: 80/20 Stratified Validation & Epoch Sweep
    exp_df, loss_curves, best_exp_id, best_info = run_epoch_experiments(
        X_train_full_A=X_train_A,
        X_train_full_B=X_train_B,
        y_train_full=y_train,
        epochs_list=epochs_list,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )

    # 4. Phase 2: Final Retraining & Official Test Benchmark
    results = run_final_retraining_and_test(
        train_df=train_valid,
        test_df=test_valid,
        total_test_images=total_test_images,
        best_info=best_info,
        classes=classes,
        models_dir=models_dir,
        reports_dir=reports_dir,
    )

    logger.info("=" * 70)
    logger.info("MILESTONE V5 MLP EXPERIMENTS COMPLETE")
    logger.info(f"Conditional Accuracy: {results['cond_acc']*100:.2f}%")
    logger.info(f"End-to-End Accuracy:  {results['e2e_acc']*100:.2f}%")
    logger.info(f"Macro F1 Score:       {results['macro_f1']:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
