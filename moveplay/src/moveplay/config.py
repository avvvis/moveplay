from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

SAMPLING_RATE_HZ = 50
WINDOW_SECONDS = 2.56
WINDOW_SIZE = 128
WINDOW_OVERLAP = 0.5
WINDOW_STEP = 64

ARM_ACC_COLS = [14, 15, 16]
ARM_GYRO_COLS = [17, 18, 19]
SENSOR_COLS = ARM_ACC_COLS + ARM_GYRO_COLS
LABEL_COL = 23

CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

LABEL_MAPPING = {
    4:  "walking",
    10: "running",
    11: "running",
    6:  "gym",
    7:  "gym",
    8:  "gym",
    12: "gym",
}
CLASS_NAMES = ["walking", "running", "gym"]

TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7]
TEST_SUBJECTS = [8, 9, 10]
