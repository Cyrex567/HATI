from huggingface_hub import hf_hub_download
import shutil
import os

REPO_ID = "Cyrex567/HATI-Lunar-Models"

files = [
    "scout_yolov8_nano.pt",
    "titan_yolov8_large.pt",
    "APOLLO17_DTM_150CM.tiff"
]

print(f"Downloading assets from {REPO_ID}...")

for file in files:
    print(f" -> Fetching {file}...")
    path = hf_hub_download(repo_id=REPO_ID, filename=file, local_dir=".")
    print(f"    Saved to {path}")

print("Done. You can now run the notebook.")
