"""Shadow-kinematics obstacle detector -- the solar-sweep hazard map.

Principle: the Sun is the probe. A real obstacle casts a shadow that rotates
around its BASE as the sun azimuth sweeps, with length L = h*cot(elevation). If
we accumulate, over all illuminations, one vote at the UP-SUN base of every
detected shadow region, then:
  * a true boulder/rim votes at the SAME base every frame -> votes CONVERGE into
    a sharp confidence peak, and the length-vs-cot(elev) spread gives its height;
  * a low-albedo patch or a fixed dark streak (which fools a single-frame census)
    does not move/elongate with the sun -> its votes SMEAR and never converge ->
    rejected.

This script proves the detector on a SYNTHETIC scene with known ground truth:
sub-resolution boulders (0.3-3 m, below DEM resolution) + albedo/streak decoys,
rendered under a 0..360 deg azimuth sweep at polar (~5 deg) elevation. It reports
how detection AUC and decoy rejection improve as the sweep fills in, and how well
heights are recovered. The SAME detector then applies to the real ingested sweep.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.measure import label, regionprops

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "athena"
RES = 0.9                    # m/px (matches NOBILE03 ortho)
N, M = 500, 500             # scene size (px) -> 450 m
ELEV = 5.0                   # sun elevation (deg), polar
AMBIENT = 0.08               # shadowed reflectance fraction
SHADOW_FRAC = 0.5            # detector: shadow if DN < 0.5 * local bg
BG_WIN = 45                  # local-background median window (px)
MIN_AREA = 4                 # px
RNG = np.random.default_rng(7)


# ----------------------------------------------------------------- scene
def make_scene():
    """Returns height field h(m), albedo map, and ground-truth dicts."""
    h = np.zeros((N, M), np.float32)
    boulders = []                                    # (row, col, height_m)
    for _ in range(70):
        r, c = RNG.integers(30, N - 30), RNG.integers(30, M - 30)
        ht = float(RNG.uniform(0.3, 3.0))
        rad = 1 if ht < 1.2 else 2                   # sub-resolution footprints
        yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
        bump = (yy * yy + xx * xx <= rad * rad) * ht
        h[r - rad:r + rad + 1, c - rad:c + rad + 1] = np.maximum(
            h[r - rad:r + rad + 1, c - rad:c + rad + 1], bump)
        boulders.append((r, c, ht))

    albedo = np.full((N, M), 1.0, np.float32)
    decoys = []                                      # (row, col)
    for _ in range(25):                              # round dark patches
        r, c = RNG.integers(20, N - 20), RNG.integers(20, M - 20)
        rad = RNG.integers(2, 4)
        yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
        m = yy * yy + xx * xx <= rad * rad
        albedo[r - rad:r + rad + 1, c - rad:c + rad + 1][m] = 0.42
        decoys.append((r, c))
    for _ in range(15):                              # fixed dark STREAKS (worst case)
        r, c = RNG.integers(30, N - 30), RNG.integers(30, M - 30)
        ang = RNG.uniform(0, np.pi); L = RNG.integers(8, 18)
        for t in range(-L, L):
            rr = int(r + t * np.sin(ang)); cc = int(c + t * np.cos(ang))
            if 0 <= rr < N and 0 <= cc < M:
                albedo[rr, cc] = 0.42
        decoys.append((r, c))
    return h, albedo, boulders, decoys


# ----------------------------------------------------------------- render
def cast_shadow(h, az_deg, elev_deg):
    """Ray-march cast shadow on height field h toward the sun azimuth."""
    e = np.radians(elev_deg); az = np.radians(az_deg)
    srow, scol = -np.cos(az), np.sin(az)             # ground unit vec TOWARD sun
    tan_e = np.tan(e)
    max_d = (float(h.max()) * (1.0 / max(tan_e, 1e-3))) + 5.0
    K = int(max_d / RES) + 1
    block = np.full_like(h, -1e9)
    for k in range(1, K + 1):
        d = k * RES
        shifted = ndi.shift(h, shift=(-k * srow, -k * scol), order=1, mode="nearest")
        np.maximum(block, shifted - d * tan_e, out=block)
    return block > h                                 # True where blocked from sun


def render(h, albedo, az, elev):
    sh = cast_shadow(h, az, elev)
    refl = albedo * np.where(sh, AMBIENT, 1.0)
    dn = refl * 200.0 + RNG.normal(0, 4.0, h.shape)  # +read noise
    return np.clip(dn, 0, None).astype(np.float32)


# ----------------------------------------------------------------- detect + vote
def detect(dn):
    bg = ndi.uniform_filter(dn, BG_WIN)   # separable, fast; shadows are sparse
    mask = dn < SHADOW_FRAC * bg
    return ndi.binary_opening(mask, iterations=1)


def vote(mask, az_deg, elev_deg, evidence, height_acc):
    """Vote at the up-sun base of each elongated (cast-shadow-like) region."""
    e = np.radians(elev_deg); az = np.radians(az_deg)
    srow, scol = -np.cos(az), np.sin(az)             # toward sun
    prow, pcol = -scol, srow                         # perpendicular
    tan_e = np.tan(e)
    for p in regionprops(label(mask)):
        if p.area < MIN_AREA:
            continue
        rr = p.coords[:, 0].astype(np.float32); cc = p.coords[:, 1].astype(np.float32)
        proj = rr * srow + cc * scol                 # along sun-anti-sun axis
        perp = rr * prow + cc * pcol
        along = (proj.max() - proj.min()) * RES
        wide = (perp.max() - perp.min()) * RES
        if along < 1.8 * max(wide, RES):             # not elongated anti-solar -> reject
            continue
        i = int(np.argmax(proj))                     # up-sun extreme = caster base
        br, bc = int(rr[i]), int(cc[i])
        evidence[br, bc] += 1.0
        height_acc[br, bc] += along * tan_e          # L*tan(e) = h


def accumulate(h, albedo, az_list):
    evidence = np.zeros((N, M), np.float32)
    height_acc = np.zeros((N, M), np.float32)
    dark = np.zeros((N, M), np.float32)
    frames = {}
    for az in az_list:
        dn = render(h, albedo, az, ELEV)
        mask = detect(dn)
        dark += mask
        vote(mask, az, ELEV, evidence, height_acc)
        frames[az] = (dn, mask)
    conf = ndi.gaussian_filter(evidence, 1.2)        # converge votes -> confidence
    with np.errstate(invalid="ignore"):
        hmap = np.where(evidence > 0, height_acc / np.maximum(evidence, 1e-6), 0)
    return conf, evidence, hmap, dark / len(az_list), frames


# ----------------------------------------------------------------- benchmark
def truth_mask(boulders):
    t = np.zeros((N, M), bool)
    for r, c, _ in boulders:
        t[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3] = True
    return t


def auc(score, truth):
    truth = truth.ravel().astype(bool); s = score.ravel()
    P, Nn = int(truth.sum()), int((~truth).sum())
    if P == 0 or Nn == 0:
        return float("nan")
    o = np.argsort(-s); t = truth[o]
    tpr = np.concatenate([[0], np.cumsum(t) / P]); fpr = np.concatenate([[0], np.cumsum(~t) / Nn])
    return float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))


def main():
    h, albedo, boulders, decoys = make_scene()
    truth = truth_mask(boulders)
    # candidate pixels = local maxima neighbourhoods only would be ideal; here score
    # the smoothed confidence directly over the whole grid (interior, to avoid edges)
    interior = np.zeros((N, M), bool); interior[20:-20, 20:-20] = True

    sweeps = [1, 3, 6, 12, 24, 48]
    print(f"scene: {len(boulders)} sub-res boulders (0.3-3 m, footprint 1-2 px = {RES:.1f}-{2*RES:.1f} m), "
          f"{len(decoys)} albedo/streak decoys | sun elev {ELEV} deg | res {RES} m")
    print(f"{'frames':>7} {'azimuths':>9} {'AUC':>7} {'decoy_conf':>11} {'boulder_conf':>13} {'h_RMSE_m':>9}")
    curve = []
    full = None
    for n in sweeps:
        az_list = list(np.linspace(0, 360, n, endpoint=False))
        conf, evidence, hmap, darkf, frames = accumulate(h, albedo, az_list)
        a = auc(np.where(interior, conf, 0), truth & interior)
        # decoy vs boulder mean confidence (separation)
        dc = float(np.mean([conf[r, c] for r, c in decoys]))
        bc = float(np.mean([conf[r, c] for r, c, _ in boulders]))
        # height recovery at detected boulders
        errs = []
        for r, c, ht in boulders:
            sub = hmap[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3]
            est = sub[sub > 0]
            if est.size:
                errs.append(np.median(est) - ht)
        rmse = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
        curve.append((n, a, dc, bc, rmse))
        print(f"{n:>7} {n:>8}  {a:>6.3f} {dc:>11.2f} {bc:>13.2f} {rmse:>9.2f}")
        if n == sweeps[-1]:
            full = (conf, hmap, darkf, frames, az_list)

    conf, hmap, darkf, frames, az_list = full
    print(f"\nsingle-frame AUC = {curve[0][1]:.3f}  ->  full-sweep AUC = {curve[-1][1]:.3f}  "
          f"(boulder/decoy confidence ratio {curve[-1][3]/max(curve[-1][2],1e-6):.0f}x)")

    # ---- figure
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)
    L = len(az_list)
    show_az = [az_list[0], az_list[L // 3], az_list[2 * L // 3]]   # ~0, 120, 240 -> visible rotation
    for i, az in enumerate(show_az):
        ax = fig.add_subplot(gs[0, i])
        dn, mask = frames[az]
        ax.imshow(dn, cmap="gray", vmin=0, vmax=210)
        ov = np.zeros((*dn.shape, 4)); ov[mask] = (1, 0.2, 0.1, 0.6)
        ax.imshow(ov)
        ax.set_title(f"rendered frame  sun az={az:.0f}°  (shadows in red)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axc = fig.add_subplot(gs[1, 0])
    axc.imshow(conf, cmap="inferno")
    br = np.array([(c, r) for r, c, _ in boulders]); dr = np.array([(c, r) for r, c in decoys])
    axc.scatter(br[:, 0], br[:, 1], s=18, facecolors="none", edgecolors="#27E1D6", lw=0.8, label="true boulder")
    axc.scatter(dr[:, 0], dr[:, 1], s=24, marker="x", c="#5cf", lw=0.8, label="albedo decoy")
    axc.set_title(f"accumulated confidence ({len(az_list)} frames)\nvotes converge on boulders, smear on decoys",
                  fontsize=9); axc.legend(fontsize=7, loc="upper right"); axc.set_xticks([]); axc.set_yticks([])
    axk = fig.add_subplot(gs[1, 1])
    ns = [c[0] for c in curve]; bcs = [c[3] for c in curve]; dcs = [c[2] for c in curve]
    axk.plot(ns, bcs, "o-", color="#1B7A6E", lw=2, label="boulder (signal)")
    axk.plot(ns, dcs, "s-", color="#B23A2E", lw=2, label="decoy (artifact)")
    axk.set_xscale("log"); axk.set_xlabel("frames in sweep"); axk.set_ylabel("mean confidence (accumulated votes)")
    axk.grid(alpha=0.3); axk.legend(fontsize=8, loc="upper left")
    axk.set_title(f"signal converges, decoys stay flat\ndetection AUC {curve[0][1]:.3f} → {curve[-1][1]:.3f}", fontsize=9)
    axh = fig.add_subplot(gs[1, 2])
    te, tr = [], []
    for r, c, ht in boulders:
        sub = hmap[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3]; est = sub[sub > 0]
        if est.size:
            tr.append(ht); te.append(np.median(est))
    axh.scatter(tr, te, s=20, color="#B23A2E", alpha=0.7)
    lim = [0, 3.3]; axh.plot(lim, lim, "k--", lw=1)
    axh.set_xlim(lim); axh.set_ylim(lim); axh.set_xlabel("true height (m)"); axh.set_ylabel("recovered height (m)")
    axh.set_title(f"height recovery from L·tan(e)\nRMSE = {curve[-1][4]:.2f} m", fontsize=9); axh.grid(alpha=0.3)
    fig.suptitle("HATI shadow-kinematics detector — the Sun as probe: sub-resolution obstacle hazard map "
                 "from a solar sweep (synthetic ground-truth benchmark)", fontsize=11, fontweight="bold")
    fig.savefig(OUT / "shadow_kinematics_benchmark.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {OUT/'shadow_kinematics_benchmark.png'}")


if __name__ == "__main__":
    main()
