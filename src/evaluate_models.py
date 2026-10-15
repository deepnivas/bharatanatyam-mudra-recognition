#!/usr/bin/env python3
"""
src/evaluate_models.py - Milestone 3: Final Test Evaluation and Coverage-Aware Analysis.

Evaluates the selected best model on the independent test split:
1. Loads test_landmarks.csv (1,050 total images).
2. Filters for landmark_detected == True (963 successfully detected test images).
3. Evaluates conditional classifier performance:
   - Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1
4. Computes End-to-end coverage-aware accuracy:
   - Treats the 87 MediaPipe detection failures as misclassifications:
     overall_correct / 1050
5. Exports reports and visualizations:
   - reports/final_test_results.txt
   - reports/test_confusion_matrix.png
   - reports/test_classification_report.csv
6. Prints the required final Milestone 3 summary.
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

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EvaluateModels")


def natural_sort_key(s: Any) -> List[int | str]:
    """Natural numerical sort key so '2-Tripataka' precedes '10-Shikara'."""
    val = str(s)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", val)]


def load_test_data(
    csv_path: Path,
) -> Tuple[np.ndarray, np.ndarray, int, int, int, List[str], List[str]]:
    """Loads test data, records total samples and detection failures, and returns valid features."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Test data not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    total_samples = len(df)
    valid_mask = df["landmark_detected"] == True
    valid_df = df[valid_mask].copy()

    successful_samples = len(valid_df)
    failed_detections = total_samples - successful_samples

    logger.info(
        f"Loaded Test set: {total_samples} total images | {successful_samples} successful MediaPipe detections | "
        f"{failed_detections} failed detections"
    )

    feature_cols = [c for c in valid_df.columns if c.startswith(("x", "y", "z")) and c[1:].isdigit()]
    if len(feature_cols) != 63:
        raise ValueError(f"Expected 63 landmark feature columns, found {len(feature_cols)}")

    X_test = valid_df[feature_cols].values.astype(np.float32)
    y_test = valid_df["class_name"].values

    classes = sorted(list(valid_df["class_name"].unique()), key=natural_sort_key)

    return X_test, y_test, total_samples, successful_samples, failed_detections, feature_cols, classes


def plot_test_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    output_path: Path,
    model_name: str,
    cond_acc: float,
    e2e_acc: float,
) -> None:
    """Plots and saves the test confusion matrix."""
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
        f"Test Confusion Matrix - {model_name}\n"
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
    logger.info(f"Saved test confusion matrix: {output_path}")


def evaluate_test_set(
    test_csv_path: Path,
    model_path: Path,
    metadata_path: Path,
    val_comparison_path: Path,
    reports_dir: Path,
) -> Dict[str, Any]:
    """Runs final evaluation on the test set using the selected best model."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load test data
    X_test, y_test, total_test, succ_test, fail_test, _, test_classes = load_test_data(test_csv_path)

    # 2. Load model and metadata
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found at: {model_path}. Run train_models.py first.")

    model = joblib.load(model_path)
    logger.info(f"Loaded model from: {model_path}")

    model_name = "Best Model"
    val_metrics = {}
    train_samples = 1744
    val_samples = 437

    if metadata_path.is_file():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            model_name = metadata.get("best_model_name", "Best Model")
            val_metrics = metadata.get("validation_metrics", {})
            train_samples = metadata.get("training_samples", 1744)
            val_samples = metadata.get("validation_samples", 437)

    # Load validation comparison table for summary
    val_comp_df = None
    if val_comparison_path.is_file():
        val_comp_df = pd.read_csv(val_comparison_path)

    # 3. Predict on valid test samples
    y_pred = model.predict(X_test)

    # 4. Conditional Metrics (on 963 detected test images)
    cond_acc = accuracy_score(y_test, y_pred)
    cond_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    cond_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    cond_macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cond_weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    correct_predictions = int(np.sum(y_pred == y_test))

    # 5. End-to-end Coverage-Aware Accuracy (correct / 1050 total test images)
    # The 87 MediaPipe failure images are treated as not correctly classified
    e2e_coverage_aware_acc = correct_predictions / float(total_test)

    logger.info(f"Conditional Accuracy: {cond_acc:.4f} ({correct_predictions}/{succ_test})")
    logger.info(f"End-to-end Coverage-Aware Accuracy: {e2e_coverage_aware_acc:.4f} ({correct_predictions}/{total_test})")

    # 6. Detailed classification report
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=test_classes,
        output_dict=True,
        zero_division=0,
    )

    # Convert to CSV format
    report_rows = []
    class_f1_scores = []
    for cls in test_classes:
        if cls in report_dict:
            metrics = report_dict[cls]
            prec = round(metrics["precision"], 4)
            rec = round(metrics["recall"], 4)
            f1 = round(metrics["f1-score"], 4)
            supp = int(metrics["support"])
            report_rows.append(
                {
                    "class": cls,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1,
                    "support": supp,
                }
            )
            class_f1_scores.append((cls, f1, prec, rec, supp))

    # Add summary rows
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
            "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "f1_score": round(cond_weighted_f1, 4),
            "support": succ_test,
        }
    )

    df_test_report = pd.DataFrame(report_rows)
    test_report_csv_path = reports_dir / "test_classification_report.csv"
    df_test_report.to_csv(test_report_csv_path, index=False)
    logger.info(f"Saved test classification report CSV: {test_report_csv_path}")

    # 7. Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=test_classes)
    cm_plot_path = reports_dir / "test_confusion_matrix.png"
    plot_test_confusion_matrix(
        cm=cm,
        classes=test_classes,
        output_path=cm_plot_path,
        model_name=model_name,
        cond_acc=cond_acc,
        e2e_acc=e2e_coverage_aware_acc,
    )

    # 8. Identify strongest and weakest mudras
    class_f1_scores.sort(key=lambda x: x[1])
    weakest_mudras = class_f1_scores[:5]
    strongest_mudras = sorted(class_f1_scores, key=lambda x: x[1], reverse=True)[:5]

    # Find confused pairs
    confused_pairs = []
    for i in range(len(test_classes)):
        for j in range(len(test_classes)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((test_classes[i], test_classes[j], int(cm[i, j])))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    # 9. Write final_test_results.txt
    txt_path = reports_dir / "final_test_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("       BHARATANATYAM MUDRA RECOGNITION - MILESTONE 3 FINAL TEST REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. DATASET SPLITS & SAMPLE COUNTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training split (80% train):            {train_samples} samples\n")
        f.write(f"Validation split (20% val):            {val_samples} samples\n")
        f.write(f"Total training data (detected hands):  {train_samples + val_samples} samples\n")
        f.write(f"Total test images:                     {total_test} samples\n")
        f.write(f"Successful MediaPipe test detections:  {succ_test} samples\n")
        f.write(f"Failed MediaPipe test detections:      {fail_test} samples\n\n")

        f.write("2. MODEL SELECTION (VALIDATION PERFORMANCE)\n")
        f.write("-" * 40 + "\n")
        if val_comp_df is not None:
            f.write(val_comp_df.to_string(index=False) + "\n\n")
        f.write(f"Selected Model: {model_name}\n")
        f.write(f"Selection Reason: Achieved highest validation Macro F1 score ({val_metrics.get('macro_f1', 0):.4f}) across all 30 classes.\n\n")

        f.write("3. FINAL TEST RESULTS (INDEPENDENT TEST SET)\n")
        f.write("-" * 40 + "\n")
        f.write(f"A. Conditional Classifier Performance (on {succ_test} successfully detected hands):\n")
        f.write(f"   - Accuracy:          {cond_acc:.4f} ({cond_acc*100:.2f}%)\n")
        f.write(f"   - Macro Precision:   {cond_prec:.4f} ({cond_prec*100:.2f}%)\n")
        f.write(f"   - Macro Recall:      {cond_rec:.4f} ({cond_rec*100:.2f}%)\n")
        f.write(f"   - Macro F1:          {cond_macro_f1:.4f} ({cond_macro_f1*100:.2f}%)\n")
        f.write(f"   - Weighted F1:       {cond_weighted_f1:.4f} ({cond_weighted_f1*100:.2f}%)\n\n")

        f.write(f"B. End-to-End Coverage-Aware Performance (across all {total_test} test images):\n")
        f.write(f"   - End-to-end coverage-aware accuracy: {e2e_coverage_aware_acc:.4f} ({e2e_coverage_aware_acc*100:.2f}%)\n")
        f.write(f"   - Total correct predictions:          {correct_predictions} / {total_test}\n")
        f.write(f"   - Upstream landmark failures:         {fail_test} / {total_test} ({fail_test/total_test*100:.2f}%)\n")
        f.write("   Note: MediaPipe failure images are treated as unclassified/incorrect,\n")
        f.write("   accurately reflecting real-world end-to-end system throughput.\n\n")

        f.write("4. CLASS-LEVEL HIGHLIGHTS\n")
        f.write("-" * 40 + "\n")
        f.write("Strongest Mudras (Top F1):\n")
        for cls, f1, p, r, s in strongest_mudras:
            f.write(f"   - {cls:<20}: F1={f1:.4f} | Prec={p:.4f} | Rec={r:.4f} | Supp={s}\n")
        f.write("\nWeakest Mudras (Lowest F1):\n")
        for cls, f1, p, r, s in weakest_mudras:
            f.write(f"   - {cls:<20}: F1={f1:.4f} | Prec={p:.4f} | Rec={r:.4f} | Supp={s}\n")

        f.write("\nTop Commonly Confused Class Pairs on Test Set:\n")
        for true_c, pred_c, cnt in confused_pairs[:8]:
            f.write(f"   - True: {true_c} -> Predicted: {pred_c} ({cnt} samples)\n")

        f.write("\n" + "=" * 70 + "\n")

    logger.info(f"Saved final test report: {txt_path}")

    # Extract metrics for console summary matching Section 8
    val_svm_acc = val_comp_df.loc[val_comp_df["model"] == "SVM", "accuracy"].values[0] if val_comp_df is not None else 0.0
    val_svm_f1 = val_comp_df.loc[val_comp_df["model"] == "SVM", "macro_f1"].values[0] if val_comp_df is not None else 0.0

    val_rf_acc = val_comp_df.loc[val_comp_df["model"] == "Random Forest", "accuracy"].values[0] if val_comp_df is not None else 0.0
    val_rf_f1 = val_comp_df.loc[val_comp_df["model"] == "Random Forest", "macro_f1"].values[0] if val_comp_df is not None else 0.0

    val_mlp_acc = val_comp_df.loc[val_comp_df["model"] == "MLP", "accuracy"].values[0] if val_comp_df is not None else 0.0
    val_mlp_f1 = val_comp_df.loc[val_comp_df["model"] == "MLP", "macro_f1"].values[0] if val_comp_df is not None else 0.0

    weakest_names_str = ", ".join([f"{c} (F1: {s:.2f})" for c, s, _, _, _ in weakest_mudras[:3]])

    # 10. Print the exact required Milestone 3 Complete summary
    print("\n" + "=" * 50)
    print("MILESTONE 3 COMPLETE")
    print("=" * 50)
    print(f"Training samples:         {train_samples}")
    print(f"Validation samples:       {val_samples}")
    print(f"Successful test samples:  {succ_test}")
    print(f"Failed test detections:   {fail_test}\n")

    print("SVM:")
    print(f"Accuracy:  {val_svm_acc*100:.2f}%")
    print(f"Macro F1:  {val_svm_f1*100:.2f}%\n")

    print("Random Forest:")
    print(f"Accuracy:  {val_rf_acc*100:.2f}%")
    print(f"Macro F1:  {val_rf_f1*100:.2f}%\n")

    print("MLP:")
    print(f"Accuracy:  {val_mlp_acc*100:.2f}%")
    print(f"Macro F1:  {val_mlp_f1*100:.2f}%\n")

    print(f"Selected model: {model_name}")
    print(f"Reason:         Highest validation Macro F1 score ({val_metrics.get('macro_f1', 0)*100:.2f}%) across all 30 classes.\n")

    print("FINAL TEST RESULTS")
    print(f"Conditional classifier accuracy:   {cond_acc*100:.2f}%")
    print(f"Conditional Macro F1:              {cond_macro_f1*100:.2f}%")
    print(f"Conditional Weighted F1:           {cond_weighted_f1*100:.2f}%\n")

    print(f"End-to-end coverage-aware accuracy: {e2e_coverage_aware_acc*100:.2f}% ({correct_predictions}/{total_test})\n")

    print(f"Weakest mudras: {weakest_names_str}\n")

    print("Generated files:")
    print("  - reports/validation_model_comparison.csv")
    print("  - reports/confusion_matrix_svm.png")
    print("  - reports/confusion_matrix_random_forest.png")
    print("  - reports/confusion_matrix_mlp.png")
    print("  - reports/model_comparison.png")
    print("  - reports/final_test_results.txt")
    print("  - reports/test_confusion_matrix.png")
    print("  - reports/test_classification_report.csv")
    print("  - models/svm_model.joblib")
    print("  - models/random_forest_model.joblib")
    print("  - models/mlp_model.joblib")
    print("  - models/best_model.joblib")
    print("  - models/model_metadata.json")
    print("=" * 50 + "\n")

    return {
        "conditional_accuracy": cond_acc,
        "conditional_macro_f1": cond_macro_f1,
        "conditional_weighted_f1": cond_weighted_f1,
        "e2e_accuracy": e2e_coverage_aware_acc,
        "correct_predictions": correct_predictions,
        "total_test": total_test,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 3: Final Test Evaluation for Selected Mudra Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/processed/test_landmarks.csv"),
        help="Path to test_landmarks.csv",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/best_model.joblib"),
        help="Path to selected best model pipeline",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("models/model_metadata.json"),
        help="Path to model_metadata.json",
    )
    parser.add_argument(
        "--val-comparison",
        type=Path,
        default=Path("reports/validation_model_comparison.csv"),
        help="Path to validation_model_comparison.csv",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to save test evaluation reports",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        evaluate_test_set(
            test_csv_path=args.test_data,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
            val_comparison_path=args.val_comparison,
            reports_dir=args.reports_dir,
        )
    except Exception as e:
        logger.exception(f"Test evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
