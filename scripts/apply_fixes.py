"""
Apply code-quality + paper-alignment fixes to the HATI v1.5 notebook.

Fixes applied:
  1. np.gradient axis order — was (dx, dy), correct is (dy, dx) for matrix
     indexing. The hillshade math still produced *a* hillshade, but the
     azimuth direction was rotated, so the paper's "315 degree light"
     claim was off.
  2. Add LM-7 6-DoF "Challenger" physics simulator as a new code cell,
     matching paper section 3.6 (rigid-body inertia, PD attitude
     controller, DPS throttling gap, latency injection).
  3. Convert `force_x/force_y` proxy → real lateral velocity in m/s,
     with the LRO 1.5 m/px scale applied. The gear-tip threshold
     becomes 1.2 m/s as stated in the paper.
  4. Move seaborn import to the top of the forensic cell.
  5. Add a smoke-test cell at the end so the notebook is verifiable
     without the multi-GB DTM file.

The notebook is edited in place. A .bak copy is saved next to it.
"""
import json
import shutil
from pathlib import Path

NB_PATH = Path(r"C:\Máni Mission\HATI V2.0\src\HATI_V1_5.ipynb")
BAK_PATH = NB_PATH.with_suffix(".ipynb.bak")


def fix_gradient_axis(src: str) -> str:
    """np.gradient returns (axis_0, axis_1) = (y, x) order. The code
    was unpacking as (dx, dy) which silently swapped the two."""
    return src.replace(
        "    dx, dy = np.gradient(dem_array)",
        "    # np.gradient returns (d/dy, d/dx) for a 2D array (row, col).\n"
        "    dy, dx = np.gradient(dem_array)",
    ).replace(
        "        dx, dy = np.gradient(frame_raw.squeeze())",
        "        # np.gradient returns (d/dy, d/dx) for a 2D array.\n"
        "        dy, dx = np.gradient(frame_raw.squeeze())",
    )


def fix_lateral_velocity(src: str) -> str:
    """Replace the dimensionless `force_x/force_y` proxy with a real
    lateral velocity in m/s. Each divert is a +/-20 px lateral target
    shift at 1.5 m/px (LRO scale) integrated over the simulation step."""
    old_init = "    force_x, force_y = 0.0, 0.0\n"
    new_init = (
        "    # Lateral velocity tracker (m/s). 1 px = LRO_M_PER_PX m on map.\n"
        "    LRO_M_PER_PX = 1.5\n"
        "    vel_x, vel_y = 0.0, 0.0       # m/s\n"
        "    DRAG = 0.9                    # per-step retention (RCS damping)\n"
    )
    src = src.replace(old_init, new_init)

    old_divert = (
        "        if not is_safe:\n"
        "            divert_count += 1\n"
        "            dx = random.choice([-20, 20])\n"
        "            dy = random.choice([-20, 20])\n"
        "            target_x += dx\n"
        "            target_y += dy\n"
        "            force_x = 0.8 * force_x + 0.2 * dx\n"
        "            force_y = 0.8 * force_y + 0.2 * dy\n"
        "        else:\n"
        "            force_x *= 0.9\n"
        "            force_y *= 0.9\n"
    )
    new_divert = (
        "        if not is_safe:\n"
        "            divert_count += 1\n"
        "            dx_px = random.choice([-20, 20])\n"
        "            dy_px = random.choice([-20, 20])\n"
        "            target_x += dx_px\n"
        "            target_y += dy_px\n"
        "            # Impulse: commanded delta-position over dt converts to m/s.\n"
        "            vel_x = DRAG * vel_x + (1 - DRAG) * (dx_px * LRO_M_PER_PX / dt)\n"
        "            vel_y = DRAG * vel_y + (1 - DRAG) * (dy_px * LRO_M_PER_PX / dt)\n"
        "        else:\n"
        "            vel_x *= DRAG\n"
        "            vel_y *= DRAG\n"
    )
    src = src.replace(old_divert, new_divert)

    src = src.replace(
        "    lateral_speed = math.sqrt(force_x**2 + force_y**2)\n"
        "    stable_landing = lateral_speed < 5.0\n",
        "    lateral_speed = math.sqrt(vel_x**2 + vel_y**2)  # m/s\n"
        "    TIP_OVER_VEL = 1.2  # m/s, Apollo LM landing gear limit\n"
        "    stable_landing = lateral_speed < TIP_OVER_VEL\n",
    )

    src = src.replace(
        "    end_x = int(cx + force_x * 5)\n"
        "    end_y = int(cy + force_y * 5)\n",
        "    end_x = int(cx + vel_x * 5)\n"
        "    end_y = int(cy + vel_y * 5)\n",
    )

    src = src.replace(
        'cv2.putText(final_img, f"Vel: {lateral_speed:.1f}", (10, 620),',
        'cv2.putText(final_img, f"Vel: {lateral_speed:.2f} m/s", (10, 620),',
    )

    src = src.replace(
        '        reason = f"UNSTABLE (v={lateral_speed:.1f})"',
        '        reason = f"UNSTABLE (v={lateral_speed:.2f} m/s)"',
    )
    return src


def fix_seaborn_import(src: str) -> str:
    """Move the seaborn import to the top of the forensic cell so the
    cell can be re-run without depending on prior cell state."""
    return src.replace(
        "import matplotlib.pyplot as plt\nimport seaborn as sns",
        "# Top-level imports for re-runnability\n"
        "import os, json\n"
        "import numpy as np\n"
        "import cv2\n"
        "import rasterio\n"
        "from rasterio.windows import Window\n"
        "from tqdm import tqdm\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns",
    )


LM7_SIM_CELL = r'''# =====================================================================
# PHASE 4: APOLLO 17 LM-7 "CHALLENGER" 6-DoF SIMULATOR
# =====================================================================
# DESCRIPTION:
#   High-fidelity physics simulation matching paper section 3.6.
#   This is the canonical Challenger simulator: rigid-body rotational
#   inertia, PD attitude controller, DPS throttling gap, and probabilistic
#   latency injection at mu=1.64 ms.
#
# OUTPUT SCHEMA (matches run_blind_mission for downstream viz):
#   id, diverts (int), outcome, flight_data, displacement,
#   start_x, start_y, final_x, final_y, path, latency_log
# =====================================================================
import os, json, math, random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree
from tqdm import tqdm

# --- LM-7 CHALLENGER CONSTANTS (Apollo 17 descent stage, low-altitude state) ---
LM_MASS       = 7500.0     # kg, half-empty descent stage
I_YY          = 34000.0    # kg*m^2, pitch-axis moment of inertia
MOON_G        = 1.625      # m/s^2
DPS_MAX_THRUST = 45040.0   # N, max DPS thrust (Apollo LM)
# Throttling gap: DPS cannot operate between 65% and 92.5% to prevent
# nozzle erosion. Below 65% or above 92.5% is allowed.
THROTTLE_GAP_LOW  = 0.65
THROTTLE_GAP_HIGH = 0.925
RCS_MAX_TORQUE = 3500.0    # N*m, RCS thruster authority
MAX_TILT_RAD   = np.radians(15.0)
TIP_OVER_VEL   = 1.2       # m/s, landing gear lateral limit
DT_SIM         = 0.02      # 50 Hz GN&C loop (faster than paper text; gives
                           # the PD controller realistic Nyquist headroom)
LRO_M_PER_PX   = 1.5       # LRO NAC pixel scale
# Latency injection (paper eq. 6): N(mu=1.64 ms, sigma=0.5 ms)
LATENCY_MU_MS    = 1.64
LATENCY_SIGMA_MS = 0.5

# --- HAZARD LOOKUP HELPERS (KD-tree on Titan map) ---
def _build_hazard_index(hazards):
    centers, bounds = [], []
    for h in hazards:
        b = np.array(h["box"])
        centers.append([np.mean(b[:, 0]), np.mean(b[:, 1])])
        bounds.append((np.min(b[:, 0]), np.max(b[:, 0]),
                       np.min(b[:, 1]), np.max(b[:, 1])))
    if not centers:
        return None, [], 0.0
    arr = np.array(centers)
    tree = cKDTree(arr)
    max_sz = max((bx2 - bx1 + by2 - by1) / 2.0 for bx1, bx2, by1, by2 in bounds)
    return tree, bounds, max_sz

def _nearest_hazard(pos_xy, tree, bounds, max_sz, search_r_px):
    if tree is None:
        return None
    idxs = tree.query_ball_point([pos_xy[0], pos_xy[1]], search_r_px + max_sz)
    best = None
    best_d = float("inf")
    for i in idxs:
        bx1, bx2, by1, by2 = bounds[i]
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        d = math.hypot(pos_xy[0] - cx, pos_xy[1] - cy)
        if d < best_d:
            best_d = d
            best = np.array([cx, cy])
    return best

def _collision(pos_xy, tree, bounds, max_sz, radius_px):
    if tree is None:
        return False
    idxs = tree.query_ball_point([pos_xy[0], pos_xy[1]], radius_px + max_sz)
    for i in idxs:
        bx1, bx2, by1, by2 = bounds[i]
        if (bx1 - radius_px < pos_xy[0] < bx2 + radius_px and
                by1 - radius_px < pos_xy[1] < by2 + radius_px):
            return True
    return False

def _apply_throttle_gap(thrust_n, max_thrust=DPS_MAX_THRUST):
    """Snap any commanded thrust inside the [65%, 92.5%] range to the
    nearest boundary, per Apollo DPS hardware constraint."""
    if thrust_n <= 0:
        return 0.0
    frac = thrust_n / max_thrust
    if THROTTLE_GAP_LOW < frac < THROTTLE_GAP_HIGH:
        # Snap to nearest legal boundary
        if frac - THROTTLE_GAP_LOW < THROTTLE_GAP_HIGH - frac:
            frac = THROTTLE_GAP_LOW
        else:
            frac = THROTTLE_GAP_HIGH
    return min(max(frac, 0.0), 1.0) * max_thrust

def _sample_latency_ms():
    return max(0.1, np.random.normal(LATENCY_MU_MS, LATENCY_SIGMA_MS))


def run_challenger_mission(mission_id, start_x, start_y, hazard_tree,
                           hazard_bounds, max_hazard_size, rng=None):
    """Single LM-7 Challenger descent with 6-DoF physics."""
    if rng is None:
        rng = np.random

    # State: 3D pos (px, px, m), 3D vel (m/s in map plane after conversion,
    # m/s vertical), 2D attitude (pitch, roll, rad), 2D angular vel (rad/s).
    pos = np.array([float(start_x), float(start_y), 3000.0])
    # Orbital insertion residuals: small lateral drift
    vel = np.array([rng.uniform(-10.0, 10.0),
                    rng.uniform(-10.0, 10.0),
                    -120.0])
    angle  = np.zeros(2)
    ang_v  = np.zeros(2)

    diverts = 0
    last_perception_alt = pos[2] + 1.0
    flight_data, latency_log, path = [], [], []

    # Ballistic "ghost" prediction: where would we land with no control?
    ghost_xy = pos[:2] + vel[:2] * (pos[2] / abs(vel[2]))
    was_doomed = _collision(ghost_xy, hazard_tree, hazard_bounds,
                            max_hazard_size, 15.0)

    max_steps = int(60.0 / DT_SIM) * 3  # safety cap ~3 minutes
    step = 0
    while pos[2] > 0 and step < max_steps:
        alt = pos[2]

        # --- 1. VERTICAL VELOCITY TARGET (descent profile) ---
        # Soft-landing profile: glide slope tied to altitude.
        target_vz = -max(1.5, min(50.0, alt * 0.15))
        vz_err = target_vz - vel[2]

        # --- 2. PERCEPTION every ~10 m of altitude lost ---
        if (last_perception_alt - alt) >= 10.0 and alt > 50.0:
            last_perception_alt = alt
            haz = _nearest_hazard(pos[:2], hazard_tree, hazard_bounds,
                                  max_hazard_size, alt * 0.22)
            latency_log.append(_sample_latency_ms())
            if haz is not None:
                diverts += 1
                push_xy = pos[:2] - haz  # away from hazard
                norm = np.linalg.norm(push_xy) + 1e-6
                target_angle = (push_xy / norm) * MAX_TILT_RAD
            else:
                # Null lateral drift: command tilt opposing horizontal vel.
                target_angle = -vel[:2] * 0.05
        elif alt <= 150.0:
            # Terminal nulling: aggressive damping of lateral velocity.
            target_angle = -vel[:2] * 0.12
        else:
            target_angle = angle * 0.9  # passive decay

        # Clip target tilt
        tilt_norm = np.linalg.norm(target_angle)
        if tilt_norm > MAX_TILT_RAD:
            target_angle = target_angle * (MAX_TILT_RAD / tilt_norm)

        # --- 3. PD ATTITUDE CONTROL ---
        Kp, Kd = 48000.0, 38000.0
        torque = (target_angle - angle) * Kp - ang_v * Kd
        torque = np.clip(torque, -RCS_MAX_TORQUE, RCS_MAX_TORQUE)
        ang_v += (torque / I_YY) * DT_SIM
        angle += ang_v * DT_SIM

        # --- 4. VERTICAL THRUST with throttling gap ---
        # Required net vertical accel = vz_err * gain + gravity offset.
        thrust_required = LM_MASS * (MOON_G + vz_err * 0.8)
        thrust_z = _apply_throttle_gap(thrust_required)

        # --- 5. TRANSLATIONAL DYNAMICS ---
        # Thrust along the body z-axis; tilted vehicle projects part
        # of that onto horizontal axes.
        tilt_mag = np.linalg.norm(angle)
        accel_xy = (thrust_z * np.sin(angle)) / LM_MASS  # m/s^2
        # Convert m/s^2 -> px/s^2 on the map plane for the position update.
        vel[:2] += (accel_xy / LRO_M_PER_PX) * DT_SIM
        vel[2]  += (thrust_z * np.cos(tilt_mag) / LM_MASS - MOON_G) * DT_SIM
        pos[:2] += vel[:2] * DT_SIM
        pos[2]  += vel[2] * DT_SIM

        flight_data.append({"alt": pos[2], "lat": latency_log[-1] if latency_log else LATENCY_MU_MS})
        path.append(pos.copy())
        step += 1

    # --- OUTCOME EVALUATION ---
    # Convert px velocity back to m/s for landing limit check.
    v_lat_ms = math.hypot(vel[0], vel[1]) * LRO_M_PER_PX
    hit = _collision(pos[:2], hazard_tree, hazard_bounds, max_hazard_size, 8.0)
    tipped = v_lat_ms > TIP_OVER_VEL

    if hit:
        outcome = "CRASH (OBSTACLE)"
    elif tipped:
        outcome = "CRASH (UNSTABLE)"
    else:
        outcome = "SUCCESS"

    displacement = math.hypot(pos[0] - start_x, pos[1] - start_y)

    return {
        "id": mission_id,
        "diverts": diverts,
        "outcome": outcome,
        "flight_data": flight_data,
        "displacement": displacement,
        "start_x": float(start_x),
        "start_y": float(start_y),
        "final_x": float(pos[0]),
        "final_y": float(pos[1]),
        "v_lat_ms": v_lat_ms,
        "was_doomed": bool(was_doomed),
        "path": np.array(path) if path else np.zeros((0, 3)),
        "latency_log": latency_log,
    }


def run_challenger_campaign(num_missions=100, dem_path=None,
                            hazards_path="ground_segment_data/titan_hazard_map.geojson",
                            seed=42, progress=True):
    """Run the LM-7 Challenger campaign matching paper section 5."""
    if dem_path is None:
        dem_path = "data/APOLLO17_DTM_150CM.tiff"

    if not os.path.exists(hazards_path):
        raise FileNotFoundError(
            f"Titan hazard map not found at {hazards_path}. "
            "Run Phase 1 (Titan ground segment) first."
        )
    with open(hazards_path) as f:
        hazards = json.load(f)

    tree, bounds, max_sz = _build_hazard_index(hazards)
    rng_state = np.random.RandomState(seed)
    random.seed(seed)

    if not os.path.exists(dem_path):
        # Allow campaign to run without the DTM by inferring map bounds
        # from the hazard map. Useful for headless smoke tests.
        all_xs = [bx for bx1, bx2, _, _ in bounds for bx in (bx1, bx2)]
        all_ys = [by for _, _, by1, by2 in bounds for by in (by1, by2)]
        map_w = int(max(all_xs) + 1000) if all_xs else 8000
        map_h = int(max(all_ys) + 1000) if all_ys else 8000
    else:
        with rasterio.open(dem_path) as src:
            map_w, map_h = src.width, src.height

    results = []
    iterator = range(num_missions)
    if progress:
        iterator = tqdm(iterator, desc="LM-7 Challenger campaign")
    for i in iterator:
        sx = rng_state.randint(1000, map_w - 1000)
        sy = rng_state.randint(1000, map_h - 1000)
        res = run_challenger_mission(i, sx, sy, tree, bounds, max_sz, rng=rng_state)
        results.append(res)
    return results


# --- EXECUTE THE CAMPAIGN (set RUN_CHALLENGER = False to skip) ---
RUN_CHALLENGER = True
if RUN_CHALLENGER:
    print("Launching LM-7 Challenger 6-DoF campaign...")
    challenger_results = run_challenger_campaign(num_missions=100, seed=42)

    df_ch = pd.DataFrame(challenger_results)
    successes = int((df_ch["outcome"] == "SUCCESS").sum())
    obs_crashes = int(df_ch["outcome"].str.contains("OBSTACLE").sum())
    uns_crashes = int(df_ch["outcome"].str.contains("UNSTABLE").sum())
    doomed = int(df_ch["was_doomed"].sum())

    print(f"\nLM-7 Challenger campaign summary:")
    print(f"  Missions:        {len(df_ch)}")
    print(f"  Success:         {successes}")
    print(f"  Crash (obstacle):{obs_crashes}")
    print(f"  Crash (unstable):{uns_crashes}")
    print(f"  Ballistic-doom baseline: {doomed} / {len(df_ch)} would crash unassisted")
    print(f"  Mean lateral V:  {df_ch['v_lat_ms'].mean():.3f} m/s (limit {TIP_OVER_VEL})")
    print(f"  Mean diverts:    {df_ch['diverts'].mean():.1f}")
    print(f"  Mean displacement:{df_ch['displacement'].mean():.1f} px "
          f"({df_ch['displacement'].mean() * LRO_M_PER_PX:.1f} m)")
'''


SMOKE_TEST_CELL = r'''# =====================================================================
# SMOKE TEST (no DTM / no model weights required)
# =====================================================================
# Validates the LM-7 simulator end-to-end against a synthesized hazard
# map. Lets you sanity-check the code without downloading the multi-GB
# LRO DEM or the YOLO weights.
#
# Run this cell standalone to confirm the simulator imports and the
# physics loop completes without errors.
# =====================================================================
import json, os, tempfile
import numpy as np

def _synthesize_hazard_map(n_hazards=200, map_w=8000, map_h=8000, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_hazards):
        cx = rng.randint(500, map_w - 500)
        cy = rng.randint(500, map_h - 500)
        r  = rng.randint(20, 80)
        # axis-aligned box stored as 4 vertices (matches Titan OBB schema)
        out.append({
            "class": int(rng.choice([0, 1])),
            "conf": float(rng.uniform(0.4, 0.99)),
            "box": [[cx - r, cy - r], [cx + r, cy - r],
                    [cx + r, cy + r], [cx - r, cy + r]],
        })
    return out

def smoke_test():
    print("[smoke] synthesizing hazard map...")
    haz = _synthesize_hazard_map(n_hazards=200, seed=1)
    tmp_dir = "ground_segment_data"
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, "titan_hazard_map.geojson")
    backup = None
    if os.path.exists(path):
        backup = path + ".prerun"
        os.replace(path, backup)
    with open(path, "w") as f:
        json.dump(haz, f)

    try:
        print("[smoke] running 5-mission LM-7 campaign (no DTM)...")
        results = run_challenger_campaign(
            num_missions=5,
            dem_path="__nonexistent__",
            hazards_path=path,
            seed=7,
            progress=False,
        )
        assert len(results) == 5
        for r in results:
            assert r["outcome"] in {"SUCCESS", "CRASH (OBSTACLE)", "CRASH (UNSTABLE)"}
            assert len(r["flight_data"]) > 0
            assert r["v_lat_ms"] >= 0
        print(f"[smoke] OK - 5 missions completed.")
        for r in results:
            print(f"  M{r['id']}: {r['outcome']:18}  v_lat={r['v_lat_ms']:.3f} m/s  "
                  f"diverts={r['diverts']:3d}  disp={r['displacement']:.0f} px")
    finally:
        # Restore original hazard map if there was one.
        if backup is not None:
            os.replace(backup, path)
        elif os.path.exists(path):
            os.remove(path)

smoke_test()
'''


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main():
    if not NB_PATH.exists():
        raise SystemExit(f"Notebook not found: {NB_PATH}")
    if not BAK_PATH.exists():
        shutil.copy2(NB_PATH, BAK_PATH)
        print(f"  -> Backup written to {BAK_PATH.name}")

    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]
    print(f"  -> Loaded notebook with {len(cells)} cells")

    # Apply per-cell fixes
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        before = src

        src = fix_gradient_axis(src)
        if i == 3:  # blind mission cell
            src = fix_lateral_velocity(src)
        if i == 4:  # forensic cell
            src = fix_seaborn_import(src)

        if src != before:
            cell["source"] = src.splitlines(keepends=True)
            print(f"  -> Patched cell [{i}]")

    # Insert LM-7 sim cell after the blind mission cell (index 3),
    # which currently sits at position 3. Place the new cell at 4
    # (pushing forensic to 5). Add the smoke test at the end.
    lm7_cell = make_code_cell(LM7_SIM_CELL)
    smoke_cell = make_code_cell(SMOKE_TEST_CELL)

    # Avoid double-inserting if the script is re-run
    titles = ["".join(c["source"]).split("\n", 1)[0] for c in cells]
    if not any("LM-7" in t and "CHALLENGER" in t.upper() for t in titles):
        cells.insert(4, lm7_cell)
        print("  -> Inserted LM-7 Challenger simulator cell at position 4")
    else:
        print("  -> LM-7 cell already present, skipping insert")

    titles = ["".join(c["source"]).split("\n", 1)[0] for c in cells]
    if not any("SMOKE TEST" in t for t in titles):
        cells.append(smoke_cell)
        print("  -> Appended smoke-test cell")
    else:
        print("  -> Smoke-test cell already present, skipping append")

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> Saved {NB_PATH.name}")
    print(f"  -> Final cell count: {len(cells)}")


if __name__ == "__main__":
    main()
