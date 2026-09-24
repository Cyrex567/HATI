"""Residual scale of the static-plus-polynomial null model, measured from the stack.

After projecting out a static albedo field (shared by all frames) and a spatial
polynomial per frame, what remains is noise plus any structure the null model
does not describe, real relief included. The pooled value is therefore an upper
bound on independent noise, not a noise measurement. The spatial-structure
diagnostics compare the residual with rendered noise so the two can be told
apart.

Patches are chosen from illumination, receiving slope and data support only,
never from detector scores: picking the places where the model already fits
would underestimate the residual.

Per-frame estimator. With n frames, the equal-weight temporal projection and a
p-term spatial projection leave, for frame k, an expected mean square per
remaining pixel of

    m_k = sigma_k^2 (1 - 2/n) + S / n^2,      S = sum_j sigma_j^2,

so sum_k m_k = S (1 - 1/n). Each sigma_k^2 = (m_k - S/n^2) / (1 - 2/n) follows
exactly for n >= 3. This holds for independent noise; structured residuals
enter m_k in the same way and are reported, not removed.
"""
from dataclasses import dataclass, asdict

import numpy as np

from .shadow_likelihood import NuisanceProjector


@dataclass(frozen=True)
class NoiseScaleConfig:
    patch_px: int = 24
    max_slope: float = 0.05        # receiving-plane gradient magnitude, m/m
    min_visibility: float = 0.99   # nominal DEM visibility required in every frame
    spatial_degree: int = 1        # 1: plane, the baseline null; 2: quadratic, the adaptive null
    scales_px: tuple = (2, 4)      # block sizes for the structure ratio

    def __post_init__(self):
        if type(self.patch_px) is not int or self.patch_px < 8:
            raise ValueError('patch_px must be an integer of at least 8')
        if not np.isfinite(self.max_slope) or self.max_slope <= 0:
            raise ValueError('max_slope must be finite and positive')
        if not 0 < self.min_visibility <= 1:
            raise ValueError('min_visibility must lie in (0, 1]')
        if self.spatial_degree not in (1, 2):
            raise ValueError('spatial_degree must be 1 or 2')
        if not self.scales_px or any(type(s) is not int or s < 2 or self.patch_px % s for s in self.scales_px):
            raise ValueError('structure scales must be integers of at least 2 that divide patch_px')


def spatial_terms(degree):
    return 3 if degree == 1 else 6


def select_patches(stack, visibility, slope_row, slope_col, cfg):
    """Non-overlapping square patches accepted by support, illumination and slope."""
    stack = np.asarray(stack, float)
    n, h, w = stack.shape
    p = cfg.patch_px
    finite = np.isfinite(stack).all(axis=0)
    lit = (np.nan_to_num(np.asarray(visibility, float), nan=0.) >= cfg.min_visibility).all(axis=0)
    slope = np.hypot(np.asarray(slope_row, float), np.asarray(slope_col, float))
    flat = np.isfinite(slope) & (slope <= cfg.max_slope)
    accepted = []
    rejected = dict(data=0, illumination=0, slope=0)
    for r in range(0, h - p + 1, p):
        for c in range(0, w - p + 1, p):
            window = np.s_[r:r + p, c:c + p]
            if not finite[window].all():
                rejected['data'] += 1
            elif not lit[window].all():
                rejected['illumination'] += 1
            elif not flat[window].all():
                rejected['slope'] += 1
            else:
                accepted.append((r, c))
    return accepted, rejected


def patch_residual(block, degree):
    """Residual of one (n, p, p) block under the null model, in input units."""
    n = block.shape[0]
    projector = NuisanceProjector(np.ones(block.shape[1:], bool), np.ones(n), spatial_degree=degree)
    return projector.apply(block).reshape(block.shape)


def per_frame_sigma(mean_square):
    """Invert m_k = sigma_k^2 (1 - 2/n) + S/n^2; negative estimates are clipped and flagged."""
    m = np.asarray(mean_square, float)
    n = len(m)
    if n < 3:
        raise ValueError('at least three frames are required')
    total = m.sum() * n / (n - 1)
    variance = (m - total / n**2) / (1 - 2 / n)
    return np.sqrt(np.clip(variance, 0, None)), float(np.sqrt(total / n)), np.flatnonzero(variance < 0).tolist()


def structure(residual, scales):
    """Lag-one correlation and block-variance ratios; independent noise gives ~0 and ~1."""
    r = residual
    horizontal = np.sum(r[..., :, 1:] * r[..., :, :-1]) / np.sqrt(np.sum(r[..., :, 1:]**2) * np.sum(r[..., :, :-1]**2))
    vertical = np.sum(r[..., 1:, :] * r[..., :-1, :]) / np.sqrt(np.sum(r[..., 1:, :]**2) * np.sum(r[..., :-1, :]**2))
    pixel = np.mean(r**2)
    ratios = {}
    for s in scales:
        n, h, w = r.shape
        blocks = r.reshape(n, h // s, s, w // s, s).mean(axis=(2, 4))
        # s^2 * var(block mean) / var(pixel) is 1 for independent pixels and
        # grows with scale when the residual has spatially extended structure.
        ratios[str(s)] = float(s * s * np.mean(blocks**2) / pixel) if pixel > 0 else None
    return dict(lag1_correlation=float((horizontal + vertical) / 2), block_variance_ratio=ratios)


def measure_residual_scale(stack, visibility, slope_row, slope_col, cfg=NoiseScaleConfig()):
    stack = np.asarray(stack, float)
    n = stack.shape[0]
    if n < 3:
        raise ValueError('at least three frames are required')
    patches, rejected = select_patches(stack, visibility, slope_row, slope_col, cfg)
    p = cfg.patch_px
    terms = spatial_terms(cfg.spatial_degree)
    dof = p * p - terms
    energy = np.zeros(n)
    per_patch = []
    residuals = []
    for r, c in patches:
        res = patch_residual(stack[:, r:r + p, c:c + p], cfg.spatial_degree)
        e = np.sum(res.reshape(n, -1)**2, axis=1)
        energy += e
        per_patch.append(dict(row_px=r, col_px=c,
                              pooled_sigma=float(np.sqrt(e.sum() / ((n - 1) * dof)))))
        residuals.append(res)
    result = dict(configuration=asdict(cfg), frames=n, patches=len(patches), rejected_patches=rejected,
                  pixels_per_patch=p * p, spatial_terms=terms, degrees_of_freedom_per_frame=dof * len(patches),
                  per_patch=per_patch)
    if not patches:
        result.update(pooled_sigma=None, per_frame_sigma=None, negative_variance_frames=[],
                      median_patch_sigma=None, structure=None)
        return result
    mean_square = energy / (dof * len(patches))
    sigma, pooled, negative = per_frame_sigma(mean_square)
    result.update(per_frame_mean_square=mean_square.tolist(), per_frame_sigma=sigma.tolist(),
                  pooled_sigma=pooled, negative_variance_frames=negative,
                  median_patch_sigma=float(np.median([q['pooled_sigma'] for q in per_patch])),
                  structure=structure(np.stack(residuals).reshape(-1, p, p), cfg.scales_px),
                  interpretation='Residual scale under the null model, including any unmodelled structure; '
                                 'an upper bound on independent noise.')
    return result


def patch_map(result, shape):
    """Per-patch pooled residual scale painted on the analysis grid, for display."""
    out = np.full(shape, np.nan)
    p = result['configuration']['patch_px']
    for q in result['per_patch']:
        out[q['row_px']:q['row_px'] + p, q['col_px']:q['col_px'] + p] = q['pooled_sigma']
    return out
