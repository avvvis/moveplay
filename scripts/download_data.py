"""Download the MHEALTH dataset from the UCI ML Repository into data/raw/."""
import io
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from moveplay.config import RAW_DATA_DIR

URL = "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"
TIMEOUT = 120
N_SUBJECTS = 10


def all_files_present() -> bool:
    return all((RAW_DATA_DIR / f"mHealth_subject{i}.log").exists() for i in range(1, N_SUBJECTS + 1))


def extract_log_files(zip_bytes: bytes, dest: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as outer:
        names = outer.namelist()
        inner_zips = [n for n in names if n.endswith(".zip")]
        if inner_zips:
            inner_name = inner_zips[0]
            print(f"  Found inner zip: {inner_name}")
            inner_bytes = outer.read(inner_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                log_names = [n for n in inner.namelist() if n.endswith(".log")]
                for log_name in log_names:
                    filename = Path(log_name).name
                    out_path = dest / filename
                    out_path.write_bytes(inner.read(log_name))
                    print(f"  Extracted: {filename}")
        else:
            log_names = [n for n in names if n.endswith(".log")]
            for log_name in log_names:
                filename = Path(log_name).name
                out_path = dest / filename
                out_path.write_bytes(outer.read(log_name))
                print(f"  Extracted: {filename}")


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if all_files_present():
        print("All 10 subject files already present — skipping download.")
        return

    print(f"Downloading MHEALTH dataset from {URL} ...")
    response = requests.get(URL, timeout=TIMEOUT)
    response.raise_for_status()
    print(f"  Downloaded {len(response.content) / 1_048_576:.1f} MB")

    print("Extracting files ...")
    extract_log_files(response.content, RAW_DATA_DIR)

    missing = [i for i in range(1, N_SUBJECTS + 1) if not (RAW_DATA_DIR / f"mHealth_subject{i}.log").exists()]
    if missing:
        print(f"WARNING: missing subject files for: {missing}")
    else:
        print(f"Done. All {N_SUBJECTS} subject files in {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
