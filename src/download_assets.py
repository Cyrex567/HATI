import os
from huggingface_hub import hf_hub_download
import shutil

# --- CONFIGURATION ---
REPO_ID = "Cyrex567/HATI-Lunar-Models"
DIRS = ["data", "models", "ground_segment_data", "flight_segment_logs"]

# Define the mapping: (HuggingFace Filename) -> (Local Folder)
FILES_TO_FETCH = {
    "APOLLO17_DTM_150CM.tiff": "data",
    "titan_yolov8_large.pt": "models",
    "scout_yolov8_nano.pt": "models"
}

def setup_environment():
    print(f" Initializing HATI v1.5 Environment...")
    
    # 1. Create Directory Structure
    for d in DIRS:
        os.makedirs(d, exist_ok=True)
        print(f"   [OK] Created directory: {d}/")

    # 2. Download Assets
    print(f"\n Connecting to Hugging Face Hub: {REPO_ID}")
    for filename, folder in FILES_TO_FETCH.items():
        print(f"   -> Downloading {filename}...")
        try:
            # Download to cache first
            cached_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
            
            # Move to the correct local folder
            destination = os.path.join(folder, filename)
            shutil.copy(cached_path, destination)
            print(f"       Placed in {destination}")
        except Exception as e:
            print(f"       ERROR: Could not download {filename}. {e}")

    print("\n Setup Complete. You can now run 'hati_code.ipynb'.")

if __name__ == "__main__":
    setup_environment()
