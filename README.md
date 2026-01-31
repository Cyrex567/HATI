# HATI v1.5: Asymmetric Split-Segment Architecture for Lunar Hazard Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%26%20Data-blue)](https://huggingface.co/Cyrex567/HATI-Lunar-Models)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

**HATI is an autonomous landing hazard detection system designed to solve the "Hardware-Data Deadlock" for future lunar missions.

Traditional monolithic architectures struggle to run high-resolution hazard detection on radiation-hardened flight computers. HATI v1.5 solves this by decoupling the workload:

* **TITAN (Ground Segment):** Uses `YOLOv8-Large` with "Deep Scan" (3.0x bicubic upsampling) to generate high-fidelity hazard maps from Earth-based servers.
* **SCOUT (Space Segment):** Uses `YOLOv8-Nano` for real-time, low-latency (1.64ms) inference and homing logic onboard the lander.

This repository contains the **Monte Carlo Landing Simulation** and the complete pipeline code.

## Performance Validation

Validated on **Apollo 17 Landing Site** data (LRO 1.5m/px resolution).

| Metric | Result | Note |
| :--- | :--- | :--- |
| **Combined Safety Score** | **91.7%** | Success rate in Monte Carlo blind descent |
| **Inference Latency** | **1.64 ms** | Validated on Edge TPU / Jetson hardware |
| **Boulder Precision** | **99.5%** | Via Ground-based Titan model |
| **Ghost Class** | Detected | Successfully identified eroded craters missed by human labelers |

## Repository Structure

* `hati_code.ipynb`: The primary research notebook containing the descent physics engine, guidance logic, and detection pipeline.
* `download_assets.py`: Automation script to fetch large weights and maps from the Hugging Face hub.
* `requirements.txt`: Python dependencies.
* `CITATION.cff`: Citation metadata for research use.

## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Cyrex567/YOUR-REPO-NAME.git](https://github.com/Cyrex567/YOUR-REPO-NAME.git)
cd YOUR-REPO-NAME
