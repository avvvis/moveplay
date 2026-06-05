import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CLASS_NAMES


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    return clf


def train_logistic_baseline(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, name: str = "") -> dict:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=CLASS_NAMES)

    report = classification_report(y_test, y_pred, labels=CLASS_NAMES, output_dict=True)
    report_str = classification_report(y_test, y_pred, labels=CLASS_NAMES)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES).tolist()

    header = f"\n{'='*60}\n{name} Evaluation\n{'='*60}"
    print(header)
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print("\nClassification Report:")
    print(report_str)

    col_width = 12
    print("Confusion Matrix (rows=true, cols=predicted):")
    header_row = " " * col_width + "".join(f"{c:>{col_width}}" for c in CLASS_NAMES)
    print(header_row)
    for i, row_label in enumerate(CLASS_NAMES):
        row_str = f"{row_label:>{col_width}}" + "".join(f"{cm[i][j]:>{col_width}}" for j in range(len(CLASS_NAMES)))
        print(row_str)
    print()

    return {
        "name": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": CLASS_NAMES,
    }


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
