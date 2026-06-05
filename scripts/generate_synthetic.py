"""
Generate synthetic MHEALTH-format data for pipeline sanity-checking.

NOTE: Results produced on synthetic data are NOT real evaluation results.
      The synthetic signals are much cleaner than real IMU data and will
      yield inflated accuracy figures.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from moveplay.config import RAW_DATA_DIR, SAMPLING_RATE_HZ

FS = SAMPLING_RATE_HZ
N_SUBJECTS = 10
N_COLS = 24
ARM_COLS = list(range(14, 20))   # indices 14-19
LABEL_COL = 23

ACTIVITIES = [
    # (label, duration_s, base_freq_hz, base_amplitude)
    (0,  30, 0.2, 0.05),
    (4,  60, 1.7, 0.50),
    (6,  30, 0.6, 1.20),
    (7,  30, 0.5, 1.50),
    (8,  30, 1.0, 1.00),
    (10, 60, 2.8, 1.00),
    (11, 60, 3.5, 1.50),
    (12, 30, 1.8, 2.00),
]

RNG_SEED = 0


def generate_subject(subject_id: int, rng: np.random.Generator) -> np.ndarray:
    blocks = []
    for label, duration_s, base_freq, base_amp in ACTIVITIES:
        n_samples = int(duration_s * FS)
        t = np.arange(n_samples) / FS

        freq_jitter = 1.0 + rng.uniform(-0.10, 0.10)
        amp_jitter = 1.0 + rng.uniform(-0.15, 0.15)
        freq = base_freq * freq_jitter
        amp = base_amp * amp_jitter

        block = np.zeros((n_samples, N_COLS), dtype=np.float32)

        # fill non-arm columns with low-amplitude noise
        non_arm = [c for c in range(N_COLS - 1) if c not in ARM_COLS]
        block[:, non_arm] = rng.normal(0, 0.05, size=(n_samples, len(non_arm)))

        # fill arm sensor channels (14-19) with sinusoidal signal + noise
        for col in ARM_COLS:
            phase = rng.uniform(0, 2 * np.pi)
            signal = amp * np.sin(2 * np.pi * freq * t + phase)
            noise = rng.normal(0, 0.02 * amp, size=n_samples)
            block[:, col] = signal + noise

        block[:, LABEL_COL] = label
        blocks.append(block)

    return np.concatenate(blocks, axis=0)


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    for sid in range(1, N_SUBJECTS + 1):
        data = generate_subject(sid, rng)
        out_path = RAW_DATA_DIR / f"mHealth_subject{sid}.log"
        np.savetxt(out_path, data, fmt="%.6f", delimiter="\t")
        print(f"Subject {sid}: {data.shape[0]} rows → {out_path.name}")

    print(f"\nDone. {N_SUBJECTS} synthetic subject files written to {RAW_DATA_DIR}")
    print("NOTE: These are synthetic files for pipeline testing only — not real MHEALTH data.")


if __name__ == "__main__":
    main()
