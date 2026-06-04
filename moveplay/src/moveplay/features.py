import numpy as np
from scipy import stats
from scipy.fft import rfft

from .config import CHANNEL_NAMES, SAMPLING_RATE_HZ, WINDOW_SIZE, WINDOW_STEP


def find_label_segments(labels: np.ndarray) -> list:
    segments = []
    if len(labels) == 0:
        return segments
    start = 0
    current = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append((start, i, current))
            start = i
            current = labels[i]
    segments.append((start, len(labels), current))
    return segments


def iter_windows(sensor_data: np.ndarray, labels: np.ndarray, window_size: int, step: int):
    segments = find_label_segments(labels)
    for seg_start, seg_end, label in segments:
        seg_len = seg_end - seg_start
        if seg_len < window_size:
            continue
        pos = seg_start
        while pos + window_size <= seg_end:
            yield sensor_data[pos:pos + window_size], label
            pos += step


def extract_features(window: np.ndarray, fs: int = SAMPLING_RATE_HZ) -> np.ndarray:
    n_samples, n_channels = window.shape
    per_axis_stats = []

    for ch in range(n_channels):
        x = window[:, ch]
        mean = np.mean(x)
        std = np.std(x)
        mn = np.min(x)
        mx = np.max(x)
        median = np.median(x)
        rms = np.sqrt(np.mean(x ** 2))
        energy = np.sum(x ** 2) / n_samples
        iqr = stats.iqr(x)

        fft_vals = rfft(x)
        fft_mag = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

        # drop DC bin for dominant frequency/magnitude
        fft_mag_no_dc = fft_mag[1:]
        freqs_no_dc = freqs[1:]

        if len(fft_mag_no_dc) > 0:
            dom_idx = np.argmax(fft_mag_no_dc)
            dom_freq = freqs_no_dc[dom_idx]
            dom_mag = fft_mag_no_dc[dom_idx]
        else:
            dom_freq = 0.0
            dom_mag = 0.0

        spec_energy = np.sum(fft_mag ** 2)

        per_axis_stats.extend([
            mean, std, mn, mx, median, rms, energy, iqr,
            dom_freq, dom_mag, spec_energy,
        ])

    # vector magnitude features
    acc = window[:, 0:3]
    gyro = window[:, 3:6]
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    per_axis_stats.extend([
        np.mean(acc_mag),
        np.std(acc_mag),
        np.mean(gyro_mag),
        np.std(gyro_mag),
    ])

    return np.array(per_axis_stats, dtype=np.float64)


_STAT_NAMES = [
    "mean", "std", "min", "max", "median", "rms", "energy",
    "iqr", "dom_freq", "dom_mag", "spec_energy",
]


def feature_names() -> list:
    names = []
    for ch in CHANNEL_NAMES:
        for stat in _STAT_NAMES:
            names.append(f"{ch}_{stat}")
    names.extend(["acc_mag_mean", "acc_mag_std", "gyro_mag_mean", "gyro_mag_std"])
    return names


def build_feature_matrix(
    sensor_data: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    window_size: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
    fs: int = SAMPLING_RATE_HZ,
) -> tuple:
    X_rows = []
    y_rows = []
    s_rows = []

    for sid in np.unique(subjects):
        mask = subjects == sid
        subj_data = sensor_data[mask]
        subj_labels = labels[mask]

        for window, label in iter_windows(subj_data, subj_labels, window_size, step):
            X_rows.append(extract_features(window, fs=fs))
            y_rows.append(label)
            s_rows.append(sid)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=object)
    s = np.array(s_rows, dtype=np.int64)
    return X, y, s
