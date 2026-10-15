#!/usr/bin/env python3
"""
src/train_final_model.py - Final SVM Retraining on Full Training Set and Test Evaluation.

Steps:
1. Loads train_landmarks.csv, keeping all 2,181 successful landmark rows (landmark_detected == True).
2. Retrains the selected SVM pipeline on ALL 2,181 training samples:
   - StandardScaler fitted on the full 2,181 samples.
   - SVC(kernel='rbf', C=10.0, probability=True, random_state=42).
3. Saves the final model to models/final_svm_model.joblib.
4. Evaluates ONLY ONCE on test_landmarks.csv (963 successful samples out of 1050 total).
5. Calculates:
   - Conditional accuracy, Macro precision, Macro recall, Macro F1, Weighted F1
   - Per-class precision, recall, F1-score
   - End-to-end coverage-aware accuracy = correct predictions / 1050
   - Weakest 5 mudras and top confused class pairs
6. Generates:
   - reports/final_model_results.txt
   - reports/final_test_classification_report.csv
   - reports/final_test_confusion_matrix.png
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainFinalModel")


def natural_sort_key(s: Any) -> List[int | str]:
    """Natural sort key so '2-Tripataka' comes before '10-Shikara'."""
    val = str(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", val)]


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, int, int, int, List[str], List[str]]:
    """Loads CSV, extracts 63 landmark features for landmark_detected==True, returns statistics."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    total_samples = len(df)
    valid_df = df[df["landmark_detected"] == True].copy()
    successful_samples = len(valid_df)
    failed_samples = total_samples - successful_samples

    feature_cols = [c for c in valid_df.columns if c.startswith(("x", "y", "z")) and c[1:].isdigit()]
    if len(feature_cols) != 63:
        raise ValueError(f"Expected 63 landmark feature columns, found {len(feature_cols)}")

    X = valid_df[feature_cols].values.astype(np.float32)
    y = valid_df["class_name"].values
    classes = sorted(list(valid_df["class_name"].unique()), key=natural_sort_key)

    return X, y, total_samples, successful_samples, failed_samples, feature_cols, classes


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    output_path: Path,
    cond_acc: float,
    e2e_acc: float,
) -> None:
    """Plots and saves the final test confusion matrix."""
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        ax=ax,
        annot_kws={"size": 6.5},
    )
    title_text = (
        f"Final Test Confusion Matrix - Retrained SVM (RBF, C=10.0, Full 2,181 Train Samples)\n"
        f"Conditional Accuracy: {cond_acc*100:.2f}% | End-to-End Coverage-Aware Accuracy: {e2e_acc*100:.2f}%"
    )
    ax.set_title(title_text, fontsize=13, pad=15, fontweight="bold")
    ax.set_xlabel("Predicted Mudra Class", fontsize=11, labelpad=10)
    ax.set_ylabel("True Mudra Class", fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved final test confusion matrix plot to: {output_path}")


def main_pipeline(
    train_path: Path,
    test_path: Path,
    models_dir: Path,
    reports_dir: Path,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load full training dataset
    X_train, y_train, total_train, succ_train, fail_train, feature_cols, train_classes = load_dataset(train_path)
    logger.info(f"Training Data: {succ_train} successful samples loaded (out of {total_train} total images)")

    # 2. Build final SVM pipeline
    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )

    logger.info("Fitting final SVM pipeline (StandardScaler + SVC(C=10.0)) on ALL 2,181 training samples...")
    final_pipeline.fit(X_train, y_train)

    # 3. Save final model
    final_model_path = models_dir / "final_svm_model.joblib"
    joblib.dump(final_pipeline, final_model_path)
    logger.info(f"Saved final retrained SVM model pipeline to: {final_model_path}")

    # Also update best_model.joblib with this final version
    best_model_path = models_dir / "best_model.joblib"
    joblib.dump(final_pipeline, best_model_path)

    # 4. Load test dataset
    X_test, y_test, total_test, succ_test, fail_test, _, test_classes = load_dataset(test_path)
    logger.info(f"Test Data: {succ_test} successful samples, {fail_test} failed detections, {total_test} total images")

    # 5. Evaluate ONCE on test data
    y_pred = final_pipeline.predict(X_test)

    # 6. Compute metrics
    cond_acc = float(accuracy_score(y_test, y_pred))
    cond_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    cond_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    cond_macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    cond_weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    correct_predictions = int(np.sum(y_pred == y_test))
    e2e_coverage_aware_acc = float(correct_predictions / total_test)

    # 7. Classification report per-class
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=test_classes,
        output_dict=True,
        zero_division=0,
    )

    report_rows = []
    class_metrics_list = []
    for cls in test_classes:
        if cls in report_dict:
            m = report_dict[cls]
            prec = round(float(m["precision"]), 4)
            rec = round(float(m["recall"]), 4)
            f1 = round(float(m["f1-score"]), 4)
            supp = int(m["support"])
            report_rows.append(
                {
                    "class": cls,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1,
                    "support": supp,
                }
            )
            class_metrics_list.append((cls, f1, prec, rec, supp))

    # Summary rows in CSV
    report_rows.append(
        {
            "class": "macro avg",
            "precision": round(cond_prec, 4),
            "recall": round(cond_rec, 4),
            "f1_score": round(cond_macro_f1, 4),
            "support": succ_test,
        }
    )
    report_rows.append(
        {
            "class": "weighted avg",
            "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "f1_score": round(cond_weighted_f1, 4),
            "support": succ_test,
        }
    )

    df_report = pd.DataFrame(report_rows)
    report_csv_path = reports_dir / "final_test_classification_report.csv"
    df_report.to_csv(report_csv_path, index=False)
    logger.info(f"Saved classification report CSV: {report_csv_path}")

    # 8. Confusion matrix & top confusions
    cm = confusion_matrix(y_test, y_pred, labels=test_classes)
    cm_plot_path = reports_dir / "final_test_confusion_matrix.png"
    plot_confusion_matrix(cm, test_classes, cm_plot_path, cond_acc, e2e_coverage_aware_acc)

    confused_pairs = []
    for i in range(len(test_classes)):
        for j in range(len(test_classes)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((test_classes[i], test_classes[j], int(cm[i, j])))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    # Weakest 5 mudras
    class_metrics_list.sort(key=lambda x: x[1])
    weakest_5 = class_metrics_list[:5]
    strongest_5 = sorted(class_metrics_list, key=lambda x: x[1], reverse=True)[:5]

    # 9. Write reports/final_model_results.txt
    txt_path = reports_dir / "final_model_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("    FINAL RETRAINED SVM MODEL EVALUATION REPORT (FULL 2,181 TRAINING SAMPLES)\n")
        f.write("=" * 72 + "\n\n")

        f.write("1. MODEL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write("Model:                  SVM (Support Vector Machine)\n")
        f.write("Kernel:                 RBF (Radial Basis Function)\n")
        f.write("C:                      10.0\n")
        f.write("Probability:            True\n")
        f.write("Random State:           42\n")
        f.write("Preprocessing:          StandardScaler (fitted strictly on full training set)\n")
        f.write(f"Feature Count:          {len(feature_cols)}\n")
        f.write(f"Training Samples:       {succ_train} (100% of successful training landmarks)\n\n")

        f.write("2. TEST DATASET SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Test Images:      {total_test}\n")
        f.write(f"Successful Detections:  {succ_test}\n")
        f.write(f"MediaPipe Failures:     {fail_test}\n\n")

        f.write("3. FINAL TEST PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write("A. Conditional Classifier Performance (evaluated on 963 detected hands):\n")
        f.write(f"   - Conditional Accuracy:     {cond_acc:.4f} ({cond_acc*100:.2f}%)\n")
        f.write(f"   - Macro Precision:          {cond_prec:.4f} ({cond_prec*100:.2f}%)\n")
        f.write(f"   - Macro Recall:             {cond_rec:.4f} ({cond_rec*100:.2f}%)\n")
        f.write(f"   - Macro F1:                 {cond_macro_f1:.4f} ({cond_macro_f1*100:.2f}%)\n")
        f.write(f"   - Weighted F1:              {cond_weighted_f1:.4f} ({cond_weighted_f1*100:.2f}%)\n\n")

        f.write("B. End-to-End Coverage-Aware Performance (evaluated on all 1,050 test images):\n")
        f.write(f"   - End-to-end Coverage-Aware Accuracy: {e2e_coverage_aware_acc:.4f} ({e2e_coverage_aware_acc*100:.2f}%)\n")
        f.write(f"   - Correctly Classified:               {correct_predictions} / {total_test}\n")
        f.write(f"   - Detection Failure Penalty:          {fail_test} / {total_test} ({fail_test/total_test*100:.2f}%)\n\n")

        f.write("4. CLASS-LEVEL PERFORMANCE HIGHLIGHTS\n")
        f.write("-" * 40 + "\n")
        f.write("Strongest Mudras (Top F1):\n")
        for cls, f1, p, r, s in strongest_5:
            f.write(f"   - {cls:<20}: F1={f1:.4f} | Prec={p:.4f} | Rec={r:.4f} | Supp={s}\n")

        f.write("\nWeakest 5 Mudras (Lowest F1):\n")
        for cls, f1, p, r, s in weakest_5:
            f.write(f"   - {cls:<20}: F1={f1:.4f} | Prec={p:.4f} | Rec={r:.4f} | Supp={s}\n")

        f.write("\nTop Confused Mudra Pairs:\n")
        for true_c, pred_c, cnt in confused_pairs[:8]:
            f.write(f"   - True: {true_c} -> Predicted: {pred_c} ({cnt} samples)\n")

        f.write("\n" + "=" * 72 + "\n")

    logger.info(f"Saved final results text report to: {txt_path}")

    # 10. Update models/model_metadata.json
    metadata = {
        "model": "SVM RBF",
        "kernel": "rbf",
        "C": 10.0,
        "probability": True,
        "random_state": 42,
        "feature count": 63,
        "training samples": succ_train,
        "successful test samples": succ_test,
        "total test samples": total_test,
        "MediaPipe test failures": fail_test,
        "conditional_metrics": {
            "accuracy": round(cond_acc, 4),
            "macro_precision": round(cond_prec, 4),
            "macro_recall": round(cond_rec, 4),
            "macro_f1": round(cond_macro_f1, 4),
            "weighted_f1": round(cond_weighted_f1, 4),
        },
        "end_to_end_coverage_aware_accuracy": round(e2e_coverage_aware_acc, 4),
        "correct_predictions": correct_predictions,
        "weakest_5_mudras": [
            {"class": cls, "f1_score": f1, "precision": p, "recall": r, "support": s}
            for cls, f1, p, r, s in weakest_5
        ],
        "top_confused_pairs": [
            {"true": t, "predicted": p, "count": cnt}
            for t, p, cnt in confused_pairs[:8]
        ],
        "classes": test_classes,
        "model_file": "models/final_svm_model.joblib",
    }

    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Updated metadata JSON at: {metadata_path}")

    # 11. Console output
    print("\n" + "=" * 60)
    print("FINAL RETRAINED SVM MODEL EVALUATION (FULL 2,181 TRAINING SAMPLES)")
    print("=" * 60)
    print(f"Training samples used:          {succ_train}")
    print(f"Successful test samples:        {succ_test}")
    print(f"Failed MediaPipe detections:    {fail_test}")
    print(f"Total test images:              {total_test}\n")

    print("CONDITIONAL CLASSIFIER PERFORMANCE (on 963 detected hands):")
    print(f"Conditional accuracy:           {cond_acc*100:.2f}%")
    print(f"Macro precision:                {cond_prec*100:.2f}%")
    print(f"Macro recall:                   {cond_rec*100:.2f}%")
    print(f"Macro F1:                       {cond_macro_f1*100:.2f}%")
    print(f"Weighted F1:                    {cond_weighted_f1*100:.2f}%\n")

    print("END-TO-END COVERAGE-AWARE PERFORMANCE (on 1,050 total images):")
    print(f"End-to-end coverage-aware accuracy: {e2e_coverage_aware_acc*100:.2f}% ({correct_predictions}/{total_test})\n")

    print("WEAKEST 5 MUDRAS:")
    for cls, f1, p, r, s in weakest_5:
        print(f"  - {cls:<20}: F1 = {f1*100:.2f}% (Prec: {p*100:.2f}%, Rec: {r*100:.2f}%, Support: {s})")

    print("\nTOP CONFUSION PAIRS:")
    for true_c, pred_c, cnt in confused_pairs[:5]:
        print(f"  - True: {true_c} -> Predicted: {pred_c} ({cnt} samples)")

    print("\nGENERATED ARTIFACTS:")
    print(f"  - {final_model_path}")
    print(f"  - {txt_path}")
    print(f"  - {report_csv_path}")
    print(f"  - {cm_plot_path}")
    print(f"  - {metadata_path}")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain SVM on full training set and perform final evaluation on test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("data/processed/train_landmarks.csv"),
        help="Path to train_landmarks.csv",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/processed/test_landmarks.csv"),
        help="Path to test_landmarks.csv",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to save evaluation reports",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        main_pipeline(
            train_path=args.train_data,
            test_path=args.test_data,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
        )
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)
