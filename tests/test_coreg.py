"""Regression guard on the co-registration measurement.

The first real ingest returned (0.00, 0.00) with a nan error on every frame,
which reads as a perfect alignment and is in fact the signature of correlating
against nodata. That was fixed by masking. This file guards the measurement
itself, which had a subtler version of the same problem.

skimage's phase_cross_correlation defaults to normalization='phase'. Phase
normalisation whitens the spectrum and so amplifies the highest-frequency bins.
On imagery that has been resampled -- cam2map interpolates, which low-passes --
those bins carry numerical noise, the correlation peak vanishes, and a planted
shift comes back as exactly (0.00, 0.00). It fails the same way when the two
frames carry different shadows, which for a solar sweep is the normal case.

Under phase normalisation the reported error is also ~1.000 whether the answer
is exact or absent, so it cannot be used to tell the two apart.

These tests plant known shifts and require them back to a fifth of a pixel.

    python tests/test_coreg.py
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation as pcc

TRUTH = (-24.1, -58.3)
TOL = 0.2
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def measure(ref: np.ndarray, mov: np.ndarray) -> tuple[np.ndarray, float]:
    """Exactly what ingest_sweep does: mean-subtract, correlate unnormalised."""
    a = ref - np.nanmean(ref)
    b = mov - np.nanmean(mov)
    sh, err, _ = pcc(a, b, upsample_factor=10, normalization=None)
    return sh, float(err)


def offset(sh: np.ndarray) -> float:
    """How far the recovered shift is from the planted one.

    pcc(ref, mov) returns the shift to APPLY to mov to bring it onto ref, so for
    mov built by shifting ref by d, the expected answer is -d. ingest_sweep feeds
    that same value straight to ndimage.shift, so this also pins the sign.
    """
    return float(np.hypot(sh[0] + TRUTH[0], sh[1] + TRUTH[1]))


def main() -> None:
    rng = np.random.default_rng(1)
    smooth = gaussian_filter(rng.normal(size=(800, 800)), 3)
    sharp = gaussian_filter(rng.normal(size=(800, 800)), 1.0)

    print("co-registration measurement")

    cases = {
        "smooth, noiseless (the cam2map case)":
            nd_shift(smooth, TRUTH, order=3, mode="nearest"),
        "smooth plus sensor noise":
            nd_shift(smooth, TRUTH, order=3, mode="nearest")
            + rng.normal(scale=0.02, size=smooth.shape),
        "sharp texture":
            nd_shift(sharp, TRUTH, order=3, mode="nearest"),
    }
    refs = {"sharp texture": sharp}
    for name, mov in cases.items():
        ref = refs.get(name, smooth)
        sh, err = measure(ref, mov)
        check(f"recovers a planted shift: {name}", offset(sh) < TOL,
              f"off by {offset(sh):.2f} px, error {err:.3f}")

    # a solar sweep correlates frames lit from different directions, so part of
    # the scene flips contrast. The measurement has to survive that.
    flipped = smooth.copy()
    flipped[:, :400] *= -0.6
    sh, err = measure(smooth, nd_shift(flipped, TRUTH, order=3, mode="nearest"))
    check("recovers it when the illumination differs", offset(sh) < TOL,
          f"off by {offset(sh):.2f} px, error {err:.3f}")

    # unnormalised correlation is dominated by the DC term, and NAC DN values sit
    # well above zero, so the mean subtraction is load-bearing rather than tidiness
    sh, err = measure(smooth + 1200.0,
                      nd_shift(smooth, TRUTH, order=3, mode="nearest") + 980.0)
    check("survives a large DC offset between the frames", offset(sh) < TOL,
          f"off by {offset(sh):.2f} px, error {err:.3f}")

    # applying the measured shift must drive the residual to sub-pixel, since that
    # residual is the gate the kinematics claim rests on
    mov = nd_shift(smooth, TRUTH, order=3, mode="nearest")
    sh, _ = measure(smooth, mov)
    aligned = nd_shift(mov, sh, order=1, mode="nearest")
    sh2, _, _ = pcc(smooth - smooth.mean(), aligned - aligned.mean(),
                    upsample_factor=20, normalization=None)
    resid = float(np.hypot(sh2[0], sh2[1]))
    check("applying the shift leaves a sub-pixel residual", resid < 1.0,
          f"residual {resid:.3f} px")

    # and the reason for all of the above: the default would have failed silently
    sh_p, err_p, _ = pcc(smooth, nd_shift(smooth, TRUTH, order=3, mode="nearest"),
                         upsample_factor=10)
    check("the skimage default returns exactly (0,0) here, which is why it is not used",
          abs(sh_p[0]) < 1e-9 and abs(sh_p[1]) < 1e-9,
          f"phase default gave ({sh_p[0]:+.2f},{sh_p[1]:+.2f}) error {err_p:.3f}")

    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
