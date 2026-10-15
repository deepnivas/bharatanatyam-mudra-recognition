#!/usr/bin/env python3
"""
src/train_models.py - Milestone 3: Classifier Training and Validation Evaluation.

Trains three classifiers (SVM, Random Forest, MLP) on Bharatanatyam mudra landmarks:
1. Loads train_landmarks.csv, filtering for landmark_detected == True.
2. Performs stratified 80% train / 20% validation split (random_state=42).
3. Trains three scikit-learn pipelines with feature scalers fitted strictly on training portion:
   - SVM (RBF kernel, probability=True)
   - Random Forest (balanced class weights)
   - MLP (multi-layer perceptron with early stopping)
4. Evaluates all models on validation split:
   - Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1
5. Saves validation comparison table and confusion matrix plots:
   - reports/validation_model_comparison.csv
   - reports/confusion_matrix_svm.png
   - reports/confusion_matrix_random_forest.png
   - reports/confusion_matrix_mlp.png
   - reports/model_comparison.png
6. Selects best model based on validation Macro F1 and saves:
   - models/svm_model.joblib
   - models/random_forest_model.joblib
   - models/mlp_model.joblib
   - models/best_model.joblib
   - models/model_metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.svm import SVC

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainModels")


def natural_sort_key(s: Any) -> List[int | str]:
    """Natural numerical sort key so '2-Tripataka' precedes '10-Shikara'."""
    val = str(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", val)]


def load_training_data(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Loads landmark features from train_landmarks.csv filtering for detected hands."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Training data not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} total rows from {csv_path}")

    valid_mask = df["landmark_detected"] == True
    valid_df = df[valid_mask].copy()
    logger.info(f"Filtered {len(valid_df)} valid landmark rows (ignored {len(df) - len(valid_df)} failed detections)")

    # Extract 63 coordinate feature columns: x1, y1, z1, ..., x21, y21, z21
    feature_cols = [c for c in valid_df.columns if c.startswith(("x", "y", "z")) and c[1:].isdigit()]
    if len(feature_cols) != 63:
        raise ValueError(f"Expected 63 landmark feature columns, found {len(feature_cols)}")

    X = valid_df[feature_cols].values.astype(np.float32)
    y = valid_df["class_name"].values

    classes = sorted(list(valid_df["class_name"].unique()), key=natural_sort_key)
    logger.info(f"Feature matrix shape: {X.shape}, Target classes: {len(classes)}")

    return X, y, feature_cols, classes


def build_model_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """Builds standard sklearn pipelines for SVM, Random Forest, and MLP."""
    pipelines = {
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=5.0,
                        kernel="rbf",
                        gamma="scale",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        criterion="gini",
                        max_depth=None,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 128),
                        activation="relu",
                        solver="adam",
                        alpha=0.0001,
                        max_iter=500,
                        early_stopping=True,
                        n_iter_no_change=15,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    return pipelines


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    title: str,
    output_path: Path,
    cmap: str = "Blues",
) -> None:
    """Plots and saves a high-resolution confusion matrix."""
    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        ax=ax,
        annot_kws={"size": 6.5},
    )
    ax.set_title(title, fontsize=13, pad=15, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=10)
    ax.set_ylabel("True Class", fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrix plot: {output_path}")


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generates bar chart comparing validation Accuracy and Macro F1 across models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    models = comparison_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    acc_vals = comparison_df["accuracy"].values * 100
    f1_vals = comparison_df["macro_f1"].values * 100

    rects1 = ax.bar(x - width / 2, acc_vals, width, label="Accuracy (%)", color="#2b5c8f")
    rects2 = ax.bar(x + width / 2, f1_vals, width, label="Macro F1 (%)", color="#e27c38")

    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Validation Performance Comparison (Accuracy vs Macro F1)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="medium")
    ax.legend(frameon=True, fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved model comparison plot: {output_path}")


def analyze_confusions(
    cm: np.ndarray,
    classes: List[str],
    top_n: int = 5,
) -> List[Tuple[str, str, int]]:
    """Identifies top confused class pairs (true_class, predicted_class, count)."""
    confusions = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confusions.append((classes[i], classes[j], int(cm[i, j])))
    confusions.sort(key=lambda x: x[2], reverse=True)
    return confusions[:top_n]


def train_and_evaluate(
    data_path: Path,
    models_dir: Path,
    reports_dir: Path,
    random_state: int = 42,
) -> Tuple[Dict[str, Pipeline], str, pd.DataFrame]:
    """Orchestrates model training, validation evaluation, comparison, and saving."""
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X, y, feature_cols, classes = load_training_data(data_path)

    # 2. Stratified 80/20 train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state,
    )
    logger.info(f"Stratified Split: Training={len(X_train)} samples, Validation={len(X_val)} samples")

    # 3. Build model pipelines
    pipelines = build_model_pipelines(random_state=random_state)

    comparison_records: List[Dict[str, Any]] = []
    cms: Dict[str, np.ndarray] = {}
    reports: Dict[str, Dict[str, Any]] = {}

    cm_color_map = {
        "SVM": "Blues",
        "Random Forest": "Greens",
        "MLP": "Purples",
    }
    cm_file_map = {
        "SVM": "confusion_matrix_svm.png",
        "Random Forest": "confusion_matrix_random_forest.png",
        "MLP": "confusion_matrix_mlp.png",
    }
    joblib_file_map = {
        "SVM": "svm_model.joblib",
        "Random Forest": "random_forest_model.joblib",
        "MLP": "mlp_model.joblib",
    }

    # 4. Train and evaluate each model on validation set
    for name, pipeline in pipelines.items():
        logger.info(f"Training {name} pipeline on {len(X_train)} samples...")
        pipeline.fit(X_train, y_train)

        # Predict on validation set
        y_pred = pipeline.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        logger.info(f"[{name}] Validation Acc: {acc:.4f}, Macro-F1: {macro_f1:.4f}, Weighted-F1: {weighted_f1:.4f}")

        comparison_records.append(
            {
                "model": name,
                "accuracy": round(acc, 4),
                "macro_precision": round(prec, 4),
                "macro_recall": round(rec, 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
            }
        )

        cm = confusion_matrix(y_val, y_pred, labels=classes)
        cms[name] = cm
        reports[name] = classification_report(y_val, y_pred, labels=classes, output_dict=True, zero_division=0)

        # Plot confusion matrix
        cm_path = reports_dir / cm_file_map[name]
        plot_confusion_matrix(
            cm,
            classes,
            title=f"Validation Confusion Matrix - {name} (Acc: {acc*100:.2f}%, Macro F1: {macro_f1*100:.2f}%)",
            output_path=cm_path,
            cmap=cm_color_map[name],
        )

        # Save individual model pipeline
        model_save_path = models_dir / joblib_file_map[name]
        joblib.dump(pipeline, model_save_path)
        logger.info(f"Saved {name} model pipeline to: {model_save_path}")

    # 5. Export validation comparison CSV
    df_comparison = pd.DataFrame(comparison_records)
    comparison_csv_path = reports_dir / "validation_model_comparison.csv"
    df_comparison.to_csv(comparison_csv_path, index=False)
    logger.info(f"Saved validation comparison table: {comparison_csv_path}")

    # 6. Generate model comparison bar plot
    comp_plot_path = reports_dir / "model_comparison.png"
    plot_model_comparison(df_comparison, comp_plot_path)

    # 7. Model selection (based on validation Macro F1)
    best_row = df_comparison.sort_values(by="macro_f1", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_macro_f1 = best_row["macro_f1"]
    best_acc = best_row["accuracy"]
    logger.info(
        f"Selected Best Model based on Validation Macro F1: {best_model_name} "
        f"(Macro F1: {best_macro_f1:.4f}, Acc: {best_acc:.4f})"
    )

    # Save best model copy
    best_pipeline = pipelines[best_model_name]
    best_model_path = models_dir / "best_model.joblib"
    joblib.dump(best_pipeline, best_model_path)
    logger.info(f"Saved best model pipeline to: {best_model_path}")

    # Identify weakest classes and top confused pairs on validation
    best_report = reports[best_model_name]
    class_f1s = []
    for cls in classes:
        if cls in best_report:
            class_f1s.append((cls, best_report[cls]["f1-score"]))
    class_f1s.sort(key=lambda x: x[1])
    weakest_classes = class_f1s[:5]

    top_confusions = analyze_confusions(cms[best_model_name], classes, top_n=5)

    # Save metadata
    metadata = {
        "best_model_name": best_model_name,
        "selection_metric": "macro_f1",
        "validation_metrics": {
            "accuracy": float(best_acc),
            "macro_f1": float(best_macro_f1),
            "macro_precision": float(best_row["macro_precision"]),
            "macro_recall": float(best_row["macro_recall"]),
            "weighted_f1": float(best_row["weighted_f1"]),
        },
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "classes": classes,
        "feature_cols": feature_cols,
        "weakest_validation_classes": [{"class": c, "f1_score": round(score, 4)} for c, score in weakest_classes],
        "top_confused_pairs": [
            {"true": t, "predicted": p, "count": cnt} for t, p, cnt in top_confusions
        ],
    }
    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to: {metadata_path}")

    # Print validation summary table
    print("\n" + "=" * 65)
    print("           VALIDATION PERFORMANCE COMPARISON")
    print("=" * 65)
    print(f"{'Model':<16}{'Accuracy':>10}{'Macro Prec':>12}{'Macro Rec':>12}{'Macro F1':>12}")
    print("-" * 65)
    for r in comparison_records:
        print(f"{r['model']:<16}{r['accuracy']:>10.4f}{r['macro_precision']:>12.4f}{r['macro_recall']:>12.4f}{r['macro_f1']:>12.4f}")
    print("=" * 65)
    print(f"Selected Best Model: {best_model_name} (Macro F1 = {best_macro_f1:.4f})")
    print(f"Weakest Validation Mudras:")
    for c, s in weakest_classes:
        print(f"  - {c}: F1 = {s:.4f}")
    print(f"Top Confused Pairs on Validation:")
    for t, p, cnt in top_confusions:
        print(f"  - True: {t} -> Predicted: {p} ({cnt} samples)")
    print("=" * 65 + "\n")

    return pipelines, best_model_name, df_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 3: Classifier Training and Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/train_landmarks.csv"),
        help="Path to train_landmarks.csv",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save trained model artifacts",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to save comparison CSV and confusion matrix plots",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        train_and_evaluate(
            data_path=args.data_path,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
            random_state=args.random_state,
        )
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
