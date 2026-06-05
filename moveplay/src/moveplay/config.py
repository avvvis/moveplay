<<<<<<< HEAD
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

=======
from pathlib import Path

>>>>>>> 571e4a36f08214a125d57f3e891c82b1b60acb2c
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
<<<<<<< HEAD

# ---------------------------------------------------------------------------
# Spotify credentials — set these in your .env file (see .env.example)
# ---------------------------------------------------------------------------
SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

# Map each activity class to a Spotify playlist URI.
# Right-click a playlist in Spotify → Share → Copy Spotify URI
ACTIVITY_PLAYLISTS = {
    "walking": os.getenv("PLAYLIST_WALKING", ""),
    "running": os.getenv("PLAYLIST_RUNNING", ""),
    "gym":     os.getenv("PLAYLIST_GYM",     ""),
}
=======
>>>>>>> 571e4a36f08214a125d57f3e891c82b1b60acb2c
