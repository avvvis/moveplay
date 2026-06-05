# MovePlay

MovePlay is a university final-year project that classifies wearable IMU activity data (accelerometer + gyroscope from the right lower arm) into three categories — **walking**, **running**, and **gym** — and will eventually switch a user's Spotify playlist to match. This repository contains the offline ML pipeline: data loading, feature extraction, model training, and evaluation. Real-time inference and Spotify integration are out of scope for this iteration.

## Directory tree

```
moveplay/
├── README.md
├── requirements.txt
├── .gitignore
├── src/moveplay/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   └── train.py
├── scripts/
│   ├── download_data.py
│   ├── generate_synthetic.py
│   └── train_model.py
├── data/raw/           (empty, populated by download_data.py)
├── data/processed/     (empty, reserved for future use)
├── models/             (empty, populated by train_model.py)
└── results/            (empty, populated by train_model.py)
```

## Setup

```bash
pip install -r requirements.txt
```

## Data acquisition

### Option A — Real MHEALTH data (recommended for honest evaluation)

```bash
python scripts/download_data.py
```

Downloads the [MHEALTH dataset](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) from UCI and extracts the 10 subject log files into `data/raw/`. The script is idempotent — if all files are already present it will skip the download.

### Option B — Synthetic data (pipeline sanity-check only)

```bash
python scripts/generate_synthetic.py
```

Generates 10 fake subject files in MHEALTH format with distinct sinusoidal frequency signatures per activity.

> **WARNING:** Synthetic-data results are NOT real evaluation results. The signals are far cleaner than real IMU data and will yield inflated accuracy figures (RF typically ≥ 0.99). Use real data for any meaningful comparison.

## Training

```bash
python scripts/train_model.py
```

Runs the full pipeline: loads data, extracts features, trains a Random Forest and a Logistic Regression baseline, prints evaluation metrics, and saves artifacts.

## Saved artifacts

| File | Description |
|------|-------------|
| `models/rf_model.joblib` | Trained Random Forest classifier (joblib format) |
| `results/metrics.json` | Accuracy, macro F1, classification report, and confusion matrix for both RF and LR |

## Configuration

All constants live in `src/moveplay/config.py` — that is the single source of truth. Key knobs:

| Constant | Value | Meaning |
|----------|-------|---------|
| `SAMPLING_RATE_HZ` | 50 | MHEALTH sampling rate |
| `WINDOW_SIZE` | 128 | Samples per window (2.56 s) |
| `WINDOW_STEP` | 64 | Step between windows (50 % overlap) |
| `TRAIN_SUBJECTS` | `[1..7]` | Subject IDs used for training |
| `TEST_SUBJECTS` | `[8, 9, 10]` | Subject IDs used for testing (disjoint) |
| `SENSOR_COLS` | `[14..19]` | Columns read from each log file |

## Class mapping

| MHEALTH label | Activity | Target class |
|---------------|----------|-------------|
| 4 | Walking | `walking` |
| 10 | Jogging | `running` |
| 11 | Running | `running` |
| 6 | Waist bends forward | `gym` |
| 7 | Frontal elevation of arms | `gym` |
| 8 | Knees bending (crouching) | `gym` |
| 12 | Jump front & back | `gym` |

All other MHEALTH labels (0, 1, 2, 3, 5, 9) are filtered out.

**Note:** The `gym` class collapses four heterogeneous calisthenic activities into one label. This is a simplification that makes the three-class story tractable but means the classifier cannot distinguish between, say, arm raises and squats. This is an interesting limitation worth discussing in a presentation context.

## Sensor channels used

Only the **right lower arm IMU** is used, because it best approximates a phone strapped to the arm (the intended deployment scenario). The chest accelerometer, ECG, left ankle sensor, and magnetometer channels are all ignored.

| Columns (0-indexed) | Sensor | Used |
|---------------------|--------|------|
| 14, 15, 16 | Right lower arm accelerometer x/y/z | Yes |
| 17, 18, 19 | Right lower arm gyroscope x/y/z | Yes |
| 0–13, 20–22 | All other sensors / magnetometers | No |
| 23 | Activity label | Yes (target) |
