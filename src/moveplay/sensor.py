import time

import numpy as np
import pandas as pd

from .config import RAW_DATA_DIR, SAMPLING_RATE_HZ, SENSOR_COLS, LABEL_COL, LABEL_MAPPING


class BaseSensor:
    """
    Interface all sensor implementations must follow.

    stream() must yield (sample, true_label) where:
      sample      : np.ndarray shape (6,) in channel order acc_x/y/z, gyro_x/y/z
      true_label  : str — known class name or "unknown" for live sensors
    """

    def stream(self):
        raise NotImplementedError


class MockSensor(BaseSensor):
    """
    Replays a recorded MHEALTH subject file as a live IMU stream.
    Useful for testing the full pipeline without real hardware.
    """

    def __init__(self, subject_id: int, speed: float = 1.0):
        self.subject_id = subject_id
        self.speed = speed
        self._fs = SAMPLING_RATE_HZ
        self._sensor, self._labels = self._load(subject_id)

    def _load(self, subject_id: int):
        path = RAW_DATA_DIR / f"mHealth_subject{subject_id}.log"
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                "Run `python scripts/download_data.py` or `python scripts/generate_synthetic.py` first."
            )
        df = pd.read_csv(path, sep=r"\s+", header=None, usecols=SENSOR_COLS + [LABEL_COL])
        sensor = df[SENSOR_COLS].to_numpy(dtype=np.float64)
        labels = df[LABEL_COL].to_numpy(dtype=np.int64)
        return sensor, labels

    def stream(self):
        interval = 1.0 / (self._fs * self.speed)
        for i in range(len(self._sensor)):
            sample = self._sensor[i]
            raw_label = int(self._labels[i])
            true_label = LABEL_MAPPING.get(raw_label, "other")
            yield sample, true_label
            time.sleep(interval)


class RealSensor(BaseSensor):
    """
    Placeholder for a real IMU sensor (phone, wearable, USB dongle, etc.).

    To implement for your hardware:
    1. Connect to the device (BLE, USB serial, TCP socket, etc.)
    2. Override stream() to yield (sample, "unknown") at ~50 Hz
    3. sample must be np.ndarray shape (6,) in this channel order:
         [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    4. Match the units of the MHEALTH training data:
         accelerometer in m/s², gyroscope in deg/s

    Example skeleton:
        def stream(self):
            while True:
                raw = self._read_from_device()          # your hardware call
                sample = self._parse_to_array(raw)      # shape (6,)
                yield sample, "unknown"
    """

    def __init__(self):
        raise NotImplementedError(
            "RealSensor is a placeholder — implement it for your specific hardware.\n"
            "See the class docstring for guidance."
        )

    def stream(self):
        raise NotImplementedError
