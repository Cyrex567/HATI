# HATI v1.5: Asymmetric Split-Segment Architecture for Lunar Hazard Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%26%20Data-blue)](https://huggingface.co/Cyrex567/HATI-Lunar-Models)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

HATI is an autonomous landing hazard detection system designed to solve the "Hardware-Data Deadlock" for future lunar missions.

Traditional monolithic architectures struggle to run high-resolution hazard detection on radiation-hardened flight computers. HATI v1.5 solves this by decoupling the workload:

* **TITAN (Ground Segment):** Uses `YOLOv8-Large` with "Deep Scan" — a deterministic Test-Time Augmentation loop over scales `[4.0x, 3.0x, 2.0x, 1.5x, 1.0x]` (bicubic) — to generate high-fidelity hazard maps from Earth-based servers.
* **SCOUT (Space Segment):** Uses `YOLOv8-Nano` for real-time, low-latency (1.64ms) inference and homing logic onboard the lander.
* **CHALLENGER (Validation):** Apollo 17 LM-7 6-DoF physics simulator with rigid-body inertia, PD attitude control, DPS throttling-gap modeling, and injected hardware latency.

This repository contains the **Monte Carlo Landing Simulation** and the complete pipeline code.

## Performance Validation

Validated on **Apollo 17 Landing Site** data (LRO 1.5m/px resolution).

| Metric | Result | Note |
| :--- | :--- | :--- |
| **Combined Safety Score** | **85%** | Success rate in Monte Carlo blind descent |
| **Boulder Precision** | **99.5%** | Via Ground-based Titan model |
| **Ghost Class** | Detected | Successfully identified eroded craters missed by human labelers |

## Repository Structure

* `HATI_V1_5.ipynb`: The primary research notebook containing the descent physics engine, guidance logic, and detection pipeline.
* `download_assets.py`: Automation script to fetch large weights and maps from the Hugging Face hub.
* `requirements.txt`: Python dependencies.
* `CITATION.cff`: Citation metadata for research use.

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Cyrex567/HATI.git
cd HATI
```

### 2. Install PyTorch for your hardware
PyTorch is *not* pinned in `requirements.txt` because the right wheel
depends on your GPU. Install it first:

| Hardware | Command |
|---|---|
| CPU-only | `pip install torch torchvision` |
| RTX 30 / 40 series (Ampere/Ada) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| RTX 50 series, e.g. 5070 Ti (Blackwell) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |

Then install the rest:
```bash
pip install -r requirements.txt
```

### 3. Download model weights and the LRO DEM
```bash
python src/download_assets.py
```
Pulls `APOLLO17_DTM_150CM.tiff` (288 MB), `titan_yolov8_large.pt` (89.5 MB),
and `scout_yolov8_nano.pt` (6.5 MB) from
[Cyrex567/HATI-Lunar-Models](https://huggingface.co/Cyrex567/HATI-Lunar-Models)
into `data/` and `models/`.

### 4. Run the notebook
Open `src/HATI_V1_5.ipynb` in VS Code / Jupyter and run cells top-to-bottom.
The notebook auto-detects CUDA and uses FP16 + Tensor Cores when a GPU is
present; on CPU it still runs but expect ~30–60 min for the Titan survey.

### 5. (Optional) Run the smoke test
The last cell synthesizes a hazard map and runs a 5-mission LM-7 campaign
in a few seconds. Use it to confirm your environment without the multi-GB
DEM.

