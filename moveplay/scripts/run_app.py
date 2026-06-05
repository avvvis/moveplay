"""
MovePlay — full application entry point.

Sensor → Activity Classifier → Spotify playlist switcher

Usage:
    python scripts/run_app.py                        # mock sensor, no Spotify
    python scripts/run_app.py --spotify              # mock sensor + Spotify
    python scripts/run_app.py --subject 9 --speed 5  # different subject/speed
    python scripts/run_app.py --spotify --speed 1.0  # real-time with Spotify
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moveplay.config import SPOTIFY_CLIENT_ID
from moveplay.inference import ActivityClassifier
from moveplay.sensor import MockSensor

STABILITY_N = 3   # consecutive matching predictions required to announce a change

DISPLAY = {
    "walking": "WALKING  🚶",
    "running": "RUNNING  🏃",
    "gym":     "GYM      💪",
    "other":   "other    ·",
}


def main():
    parser = argparse.ArgumentParser(description="MovePlay activity-aware playlist switcher")
    parser.add_argument("--subject", type=int, default=8,
                        help="Mock sensor subject ID to replay (1–10, default: 8)")
    parser.add_argument("--speed", type=float, default=5.0,
                        help="Replay speed multiplier (1.0 = real-time, default: 5.0)")
    parser.add_argument("--spotify", action="store_true",
                        help="Enable Spotify playlist switching")
    args = parser.parse_args()

    # --- Spotify setup ---
    spotify = None
    if args.spotify:
        if not SPOTIFY_CLIENT_ID:
            print("ERROR: Spotify credentials not configured.")
            print("       Copy .env.example → .env and fill in your credentials.")
            print("       Then run again with --spotify.")
            sys.exit(1)
        from moveplay.spotify import SpotifyController
        spotify = SpotifyController().connect()
    else:
        print("[Spotify] Running in dry-run mode (--spotify not passed, no playlist switching)")

    # --- Sensor + classifier ---
    print(f"\n[Sensor]  Replaying subject {args.subject} at {args.speed}x speed")
    print("[Model]   Loading activity classifier ...")
    sensor = MockSensor(subject_id=args.subject, speed=args.speed)
    classifier = ActivityClassifier()
    print("[Model]   Ready.\n")

    print("=" * 45)
    print("  MovePlay is running  |  Ctrl+C to stop")
    print("=" * 45)

    recent = deque(maxlen=STABILITY_N)
    current_activity = None

    try:
        for sample, _ in sensor.stream():
            pred = classifier.push_sample(sample)
            if pred is None:
                continue

            recent.append(pred)
            if len(recent) < STABILITY_N or len(set(recent)) > 1:
                continue

            stable = recent[0]
            if stable == current_activity or stable == "other":
                continue

            current_activity = stable
            ts = time.strftime("%H:%M:%S")
            print(f"  [{ts}]  {DISPLAY.get(stable, stable)}")

            if spotify:
                spotify.on_activity_change(stable)

    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    main()
