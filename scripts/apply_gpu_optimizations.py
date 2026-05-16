"""
Optimize the HATI v1.5 notebook for an RTX 5070 Ti (Blackwell, sm_120,
16 GB VRAM, CUDA 12.8+).

The notebook stays CPU-compatible; GPU is used when available.

Changes applied
---------------
1. Device-setup preamble at the top of cell 0:
     - Detects CUDA, prints device + VRAM, enables cuDNN benchmark
       and TF32 matmul.
     - Exposes DEVICE and USE_HALF for downstream cells.

2. Cell 0 (Titan scan):
     - Move model to DEVICE: YOLO(...).to(DEVICE).
     - Batched TTA: model.predict([orig, flip], half=USE_HALF) instead
       of two sequential calls.
     - Pass imgsz=640 explicitly so YOLO does not re-detect resolution
       per-call.

3. Cell 1 (Scout config):
     - Same device + half migration.

4. Cell 4 (LM-7 simulator):
     - Add USE_LIVE_SCOUT flag. When True, the perception step actually
       calls the Scout YOLO and measures wall-clock latency. When False
       (default), the paper's N(1.64ms, 0.5ms) sample is used.

The script is idempotent.
"""
import json
import re
from pathlib import Path

NB_PATH = Path(r"C:\Máni Mission\HATI V2.0\src\HATI_V1_5.ipynb")


GPU_PREAMBLE = '''# ----- GPU SETUP (CUDA if available, else CPU) -----
# Optimized for NVIDIA RTX 5070 Ti (Blackwell, sm_120, 16 GB VRAM).
# Requires torch built against CUDA 12.8 — see requirements.txt.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = (DEVICE == "cuda")  # FP16 inference on tensor cores
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")  # TF32 on Blackwell
    _props = torch.cuda.get_device_properties(0)
    print(f"[GPU] {_props.name}  |  VRAM {_props.total_memory / 1e9:.1f} GB"
          f"  |  sm_{_props.major}{_props.minor}")
else:
    print("[CPU] No CUDA visible. Titan scan will be much slower; consider running on a GPU host.")

'''


def patch_cell_0(src: str) -> str:
    # Insert GPU preamble right before SETUP section
    setup_marker = "# 1. SETUP\n"
    if "[GPU]" in src:
        return src  # already patched
    src = src.replace(setup_marker, GPU_PREAMBLE + setup_marker)

    # Move titan model to DEVICE
    src = src.replace(
        "    model_titan = YOLO(PATH_MODEL_TITAN)",
        "    model_titan = YOLO(PATH_MODEL_TITAN)\n    model_titan.to(DEVICE)",
    )

    # Batch the orig+flip pair in one predict call.
    old_tta = '''                    # Prepare Batch: [Original, H-Flip]
                    img_list = [base_input, cv2.flip(base_input, 1)]

                    for i, img in enumerate(img_list):
                        # Inference (Low Conf 0.10)
                        results = model_titan.predict(img, conf=0.10, verbose=False, classes=[0, 1])

                        for r in results:'''
    new_tta = '''                    # Prepare Batch: [Original, H-Flip]
                    img_list = [base_input, cv2.flip(base_input, 1)]

                    # Batched inference: orig + flip in a single forward pass.
                    # imgsz fixed so YOLO does not re-letterbox per call.
                    batched = model_titan.predict(
                        img_list, conf=0.10, verbose=False, classes=[0, 1],
                        imgsz=640, half=USE_HALF, device=DEVICE,
                    )

                    for i, r in enumerate(batched):
                        img = img_list[i]
                        for r in [r]:'''
    src = src.replace(old_tta, new_tta)
    return src


def patch_cell_1(src: str) -> str:
    """Scout config — small cell, just loads the model and the Titan map."""
    if ".to(DEVICE)" in src:
        return src
    src = src.replace(
        "    model_scout = YOLO(PATH_MODEL_SCOUT)",
        "    model_scout = YOLO(PATH_MODEL_SCOUT)\n    model_scout.to(DEVICE)",
    )
    return src


def patch_cell_3(src: str) -> str:
    """Block 4 — kinematic Monte Carlo. The mission loop doesn't actually
    call the YOLO model (the Titan map is precomputed). Nothing to GPU-ify."""
    return src


def patch_cell_4(src: str) -> str:
    """LM-7 simulator — add USE_LIVE_SCOUT flag plus real-inference path."""
    if "USE_LIVE_SCOUT" in src:
        return src

    # Insert a flag declaration near the constants block.
    flag_block = '''
# Live Scout inference vs. sampled latency.
#   False (default, recommended for paper-reproducible Monte Carlo):
#     Latency is sampled from N(LATENCY_MU_MS, LATENCY_SIGMA_MS).
#   True (requires GPU + Scout weights loaded as `model_scout`):
#     Each perception step actually runs the Scout YOLO and uses the
#     measured wall-clock latency. Slower, but proves the loop closes.
USE_LIVE_SCOUT = False
'''
    src = src.replace(
        "# Latency injection (paper eq. 6): N(mu=1.64 ms, sigma=0.5 ms)",
        flag_block + "\n# Latency injection (paper eq. 6): N(mu=1.64 ms, sigma=0.5 ms)",
    )

    # Insert the live-scout branch inside the perception block.
    src = src.replace(
        "            latency_log.append(_sample_latency_ms())",
        '''            if USE_LIVE_SCOUT and "model_scout" in globals() and DEVICE == "cuda":
                # Real inference: measure wall-clock latency in ms.
                import time as _t
                _t0 = _t.perf_counter()
                _dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = model_scout.predict(
                    _dummy, conf=0.30, verbose=False,
                    imgsz=640, half=USE_HALF, device=DEVICE,
                )
                latency_log.append((_t.perf_counter() - _t0) * 1000.0)
            else:
                latency_log.append(_sample_latency_ms())''',
    )
    return src


def main():
    if not NB_PATH.exists():
        raise SystemExit(f"Notebook not found: {NB_PATH}")

    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    patchers = {0: patch_cell_0, 1: patch_cell_1, 3: patch_cell_3, 4: patch_cell_4}
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        before = src
        fn = patchers.get(i)
        if fn:
            src = fn(src)
        if src != before:
            cell["source"] = src.splitlines(keepends=True)
            print(f"  -> Optimized cell [{i}]")

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> Saved {NB_PATH.name}")


if __name__ == "__main__":
    main()
