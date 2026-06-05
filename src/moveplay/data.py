import numpy as np
import pandas as pd

from .config import (
    RAW_DATA_DIR,
    SENSOR_COLS,
    LABEL_COL,
    LABEL_MAPPING,
)


def load_subject(subject_id: int) -> tuple:
    path = RAW_DATA_DIR / f"mHealth_subject{subject_id}.log"
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Run `python scripts/download_data.py` to fetch the MHEALTH dataset, "
            "or `python scripts/generate_synthetic.py` for synthetic data."
        )
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=SENSOR_COLS + [LABEL_COL],
    )
    sensor_data = df[SENSOR_COLS].to_numpy(dtype=np.float64)
    labels = df[LABEL_COL].to_numpy(dtype=np.int64)
    return sensor_data, labels


def filter_target_classes(sensor_data: np.ndarray, labels: np.ndarray) -> tuple:
    mask = np.isin(labels, list(LABEL_MAPPING.keys()))
    filtered_data = sensor_data[mask]
    mapped_labels = np.array(
        [LABEL_MAPPING[int(lbl)] for lbl in labels[mask]], dtype=object
    )
    return filtered_data, mapped_labels


def load_subjects(subject_ids: list) -> tuple:
    all_data = []
    all_labels = []
    all_subjects = []

    for sid in subject_ids:
        sensor_data, labels = load_subject(sid)
        filtered_data, mapped_labels = filter_target_classes(sensor_data, labels)
        all_data.append(filtered_data)
        all_labels.append(mapped_labels)
        all_subjects.append(np.full(len(filtered_data), sid, dtype=np.int64))

    sensor_data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    subjects = np.concatenate(all_subjects, axis=0)
    return sensor_data, labels, subjects
