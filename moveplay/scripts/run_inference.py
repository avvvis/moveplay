"""
Run the activity classifier on replayed sensor data (mock real-time inference).

Usage:
    python scripts/run_inference.py                    # subject 8, 10x speed
    python scripts/run_inference.py --subject 9        # different subject
    python scripts/run_inference.py --speed 1.0        # real-time speed
    python scripts/run_inference.py --speed 20         # fast forward
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moveplay.inference import ActivityClassifier
from moveplay.sensor import MockSensor

STABILITY_N = 3   # consecutive matching predictions before announcing a change

ACTIVITY_LABEL = {
    "walking": "WALKING",
    "running": "RUNNING",
    "gym":     "GYM    ",
    "other":   "other  ",
}


def main():
    parser = argparse.ArgumentParser(description="Mock real-time activity inference")
    parser.add_argument("--subject", type=int, default=8,
                        help="Subject ID to replay (1-10, default: 8)")
    parser.add_argument("--speed", type=float, default=10.0,
                        help="Replay speed multiplier — 1.0=real-time, 10.0=10x faster (default: 10)")
    args = parser.parse_args()

    print("\nMovePlay — Mock Real-Time Inference")
    print(f"Subject : {args.subject}   Speed : {args.speed}x   (Ctrl+C to stop)")
    print("-" * 60)

    sensor = MockSensor(subject_id=args.subject, speed=args.speed)
    classifier = ActivityClassifier()

    recent = deque(maxlen=STABILITY_N)
    current_activity = None
    total = 0
    correct = 0
    start = time.time()

    print(f"{'Elapsed':>8}  {'Prediction':<12}  {'True label':<12}  {'Accuracy':>8}")
    print(f"{'-------':>8}  {'-'*12:<12}  {'-'*12:<12}  {'--------':>8}")

    try:
        for sample, true_label in sensor.stream():
            pred = classifier.push_sample(sample)
            if pred is None:
                continue

            total += 1
            if true_label != "other" and pred == true_label:
                correct += 1

            recent.append(pred)

            # only print when we have a stable run of identical predictions
            if len(recent) < STABILITY_N or len(set(recent)) > 1:
                continue

            stable = recent[0]
            if stable == current_activity:
                continue

            current_activity = stable
            elapsed = time.time() - start
            acc = f"{correct / total:.1%}" if total else "  N/A"
            marker = " <-- change" if total > STABILITY_N else ""
            print(f"{elapsed:>7.1f}s  {ACTIVITY_LABEL.get(stable, stable):<12}  "
                  f"{ACTIVITY_LABEL.get(true_label, true_label):<12}  {acc:>8}{marker}")

    except KeyboardInterrupt:
        print("\n(interrupted)")

    elapsed = time.time() - start
    print("-" * 60)
    print(f"Elapsed       : {elapsed:.1f}s")
    print(f"Predictions   : {total}")
    if total:
        print(f"Accuracy      : {correct / total:.1%}  (on labelled windows only)")
    print()


if __name__ == "__main__":
    main()
