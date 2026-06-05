from collections import deque

import joblib
import numpy as np

from .config import MODELS_DIR, SAMPLING_RATE_HZ, WINDOW_SIZE, WINDOW_STEP
from .features import extract_features


class ActivityClassifier:
    """Sliding-window activity classifier for real-time or replayed IMU data."""

    def __init__(self, model_path=None, window_size=WINDOW_SIZE, step=WINDOW_STEP, fs=SAMPLING_RATE_HZ):
        if model_path is None:
            model_path = MODELS_DIR / "rf_model.joblib"
        self.model = joblib.load(model_path)
        self.window_size = window_size
        self.step = step
        self.fs = fs
        self._buffer = deque(maxlen=window_size)
        self._since_last = 0

    def push_sample(self, sample: np.ndarray):
        """Feed one 6-channel sample. Returns predicted class string or None."""
        self._buffer.append(sample)
        self._since_last += 1

        if len(self._buffer) == self.window_size and self._since_last >= self.step:
            window = np.array(self._buffer)
            features = extract_features(window, self.fs)
            prediction = self.model.predict([features])[0]
            self._since_last = 0
            return prediction

        return None

    def reset(self):
        self._buffer.clear()
        self._since_last = 0
