import os
import urllib.request
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}


def download(url: str, dest: Path):
    print(f"Downloading {dest.name}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> saved to {dest}")


def ensure_models():
    for name, url in MODELS.items():
        dest = MODELS_DIR / name
        if not dest.exists():
            download(url, dest)
        else:
            print(f"{name} already exists")


if __name__ == "__main__":
    ensure_models()
