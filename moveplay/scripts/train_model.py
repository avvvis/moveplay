"""End-to-end training pipeline for MovePlay activity classifier."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moveplay.config import (
    MODELS_DIR,
    RESULTS_DIR,
    TRAIN_SUBJECTS,
    TEST_SUBJECTS,
    WINDOW_SIZE,
    WINDOW_STEP,
    SAMPLING_RATE_HZ,
)
from moveplay.data import load_subjects
from moveplay.features import build_feature_matrix, feature_names
from moveplay.train import (
    train_random_forest,
    train_logistic_baseline,
    evaluate,
    save_model,
    save_metrics,
)


def print_class_distribution(label, counts: Counter) -> None:
    print(f"  {label}:")
    for cls, n in sorted(counts.items()):
        print(f"    {cls}: {n:,} rows")


def main() -> None:
    print("=" * 60)
    print("  MovePlay – Activity Classifier Training Pipeline")
    print("=" * 60)

    # --- Load raw data ---
    print("\n[1/5] Loading training data ...")
    train_data, train_labels, train_subjects = load_subjects(TRAIN_SUBJECTS)
    print(f"  {len(train_data):,} rows, subjects {TRAIN_SUBJECTS}")
    print_class_distribution("Class distribution", Counter(train_labels))

    print("\n[2/5] Loading test data ...")
    test_data, test_labels, test_subjects = load_subjects(TEST_SUBJECTS)
    print(f"  {len(test_data):,} rows, subjects {TEST_SUBJECTS}")
    print_class_distribution("Class distribution", Counter(test_labels))

    # --- Build feature matrices ---
    print("\n[3/5] Building training feature matrix ...")
    X_train, y_train, _ = build_feature_matrix(
        train_data, train_labels, train_subjects, WINDOW_SIZE, WINDOW_STEP, SAMPLING_RATE_HZ
    )
    print(f"  Shape: {X_train.shape}")
    print_class_distribution("Windows per class", Counter(y_train))

    print("\n[4/5] Building test feature matrix ...")
    X_test, y_test, _ = build_feature_matrix(
        test_data, test_labels, test_subjects, WINDOW_SIZE, WINDOW_STEP, SAMPLING_RATE_HZ
    )
    print(f"  Shape: {X_test.shape}")
    print_class_distribution("Windows per class", Counter(y_test))

    # --- Train and evaluate ---
    print("\n[5/5] Training models ...")

    print("  Training Random Forest ...")
    rf = train_random_forest(X_train, y_train)
    rf_metrics = evaluate(rf, X_test, y_test, name="RandomForest")

    print("  Training Logistic Regression baseline ...")
    lr = train_logistic_baseline(X_train, y_train)
    lr_metrics = evaluate(lr, X_test, y_test, name="LogisticRegression")

    # --- Save artifacts ---
    model_path = MODELS_DIR / "rf_model.joblib"
    save_model(rf, model_path)
    print(f"\nModel saved: {model_path}")

    metrics_path = RESULTS_DIR / "metrics.json"
    save_metrics({"rf": rf_metrics, "lr": lr_metrics}, metrics_path)
    print(f"Metrics saved: {metrics_path}")

    # --- Feature importance ---
    names = feature_names()
    importances = rf.feature_importances_
    top15_idx = np.argsort(importances)[::-1][:15]
    print("\nTop 15 features by RF importance:")
    for rank, idx in enumerate(top15_idx, 1):
        print(f"  {rank:2d}. {names[idx]:<30s}  {importances[idx]:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
