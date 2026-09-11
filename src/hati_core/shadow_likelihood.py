"""Experimental image-domain shadow-root search; no learned parameters.

H0 is a static image plus an independent illumination plane per frame.
H1 adds one nonnegative, shared-amplitude moving shadow template. Both are
projected through the SAME nuisance operator. Scores assume independent normal
noise with supplied sigma; they are rankings, not field-calibrated significances.
All frames participate. Incomplete pixels are excluded from every frame rather
than allowing a candidate to select whichever observations support it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class ShadowConfig:
    pixel_m: float = 0.9
    heights_m: tuple = (0.3, 0.5, 1.0, 2.0)
    widths_m: tuple = (0.4, 0.9, 1.8, 3.6)
    radius_px: int = 12
    root_support_px: float = 6.0
    psf_sigma_px: float = 0.6
    registration_sigma_px: float = 0.0
    solar_radius_deg: float = 0.266
    supersample: int = 4
    max_candidates: int = 150
    min_separation_px: int = 3
    min_common_fraction: float = 0.8
    min_identifiability: float = 0.02
    max_contrast: float = 1.0
    root_offsets_px: tuple = (-0.5, 0.0, 0.5)

    def __post_init__(self):
        positive = (self.pixel_m, self.radius_px, self.supersample,
                    self.max_candidates, self.min_separation_px, self.max_contrast)
        if any(not np.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("scales, counts and contrast must be positive")
        if self.radius_px < 4 or any(int(v) != v for v in (
                self.radius_px, self.supersample, self.max_candidates, self.min_separation_px)):
            raise ValueError("radius must be >=4 and counts must be integers")
        if not np.isfinite(self.root_support_px) or not 2 <= self.root_support_px <= self.radius_px:
            raise ValueError("root fitting radius must lie between 2 and patch radius")
        for seq in (self.heights_m, self.widths_m):
            if not seq or any(not np.isfinite(v) or v <= 0 for v in seq):
                raise ValueError("height and width grids must be finite and positive")
        if any(not np.isfinite(v) or v < 0 for v in (
                self.psf_sigma_px, self.registration_sigma_px, self.solar_radius_deg)):
            raise ValueError("blur and solar radius must be finite and nonnegative")
        if not 0 < self.min_common_fraction <= 1 or not 0 < self.min_identifiability <= 1:
            raise ValueError("observability fractions must lie in (0, 1]")
        if not self.root_offsets_px or any(not np.isfinite(v) or abs(v) > 0.5
                                           for v in self.root_offsets_px):
            raise ValueError("root offsets must be within half a pixel")

    def hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


class NuisanceProjector:
    """Exact weighted projection for common support and per-frame scalar noise.

    The temporal static-albedo space and spatial illumination-plane space
    commute on common support. Subtract both projections, adding their overlap
    implicitly through sequential application. No alternating-fit tolerance.
    """
    def __init__(self, common, sigma):
        self.common = np.asarray(common, bool)
        self.sigma = np.asarray(sigma, float)
        if self.common.ndim != 2 or self.common.sum() < 12:
            raise ValueError("insufficient common pixels")
        if self.sigma.ndim != 1 or len(self.sigma) < 3 or np.any(
                ~np.isfinite(self.sigma) | (self.sigma <= 0)):
            raise ValueError("at least three finite positive frame noise scales required")
        rr, cc = np.indices(self.common.shape, dtype=float)
        design = np.stack([np.ones(self.common.sum()),
                           rr[self.common] - rr[self.common].mean(),
                           cc[self.common] - cc[self.common].mean()], axis=1)
        if np.linalg.matrix_rank(design) != 3:
            raise ValueError("common support cannot identify an illumination plane")
        self.q = np.linalg.qr(design, mode="reduced")[0]
        self.u = 1 / self.sigma
        self.u /= np.linalg.norm(self.u)

    def apply(self, stack):
        a = np.asarray(stack, float)[..., self.common] / self.sigma[:, None]
        if not np.isfinite(a).all():
            raise ValueError("nonfinite sample on declared common support")
        a = a - self.u[:, None] * (self.u @ a)[..., None, :]
        return a - (a @ self.q) @ self.q.T


class RegistrationProjector(NuisanceProjector):
    """Whiten a first-order static-albedo displacement covariance.

    A small shift delta changes albedo by G*delta. With isotropic registration
    sigma s, spatial covariance in normalized noise units is I+s^2*G*G'/sigma^2.
    Two singular modes implement its inverse square root without a dense matrix.
    Identical noise scales are required so whitening commutes with removal of
    static albedo. This marginalizes a Gaussian displacement approximation;
    it cannot repair wrong registration peaks or large nonlinear displacements.
    """
    def __init__(self,common,sigma,static_image,registration_sigma_px):
        super().__init__(common,sigma)
        if not np.isfinite(registration_sigma_px) or registration_sigma_px<0:
            raise ValueError('registration sigma must be finite and nonnegative')
        if not np.allclose(self.sigma,self.sigma[0]):
            raise ValueError('registration covariance currently requires equal frame noise')
        static = np.asarray(static_image,float)
        if static.shape != common.shape or not np.isfinite(static).all():
            raise ValueError('static covariance reference must be finite and match the patch')
        gr,gc = np.gradient(ndi.gaussian_filter(static,.6))
        g = np.stack([gr[common],gc[common]],axis=1)*registration_sigma_px/self.sigma[0]
        g -= self.q@(self.q.T@g)
        self.modes,singular,_ = np.linalg.svd(g,full_matrices=False)
        self.attenuation = 1-1/np.sqrt(1+singular**2)

    def apply(self,stack):
        a = super().apply(stack)
        return a-((a@self.modes)*self.attenuation)@self.modes.T


def shadow_template(shape, root, azimuths, elevations, height_m, width_m,
                    cfg, slope_rc=(0.0, 0.0)):
    """Pixel-integrated rectangular shadow on a local receiving plane.

    Azimuth: clockwise from map up. Image rows increase down. slope_rc is
    dz/distance along increasing row and column, in m/m. Width is independent
    of height. Five equal-area solar-disc strips approximate finite-Sun blur;
    a Gaussian approximates optical blur and supplied registration uncertainty.
    Returns coverage and whether any shadow endpoint falls outside the patch.
    Censored templates support a root ranking, not a measured object height.
    """
    az = np.radians(np.asarray(azimuths, float))
    el = np.asarray(elevations, float)
    if az.shape != el.shape or az.ndim != 1 or not np.isfinite(az).all() or not np.isfinite(el).all():
        raise ValueError("invalid illumination arrays")
    if np.any(el <= cfg.solar_radius_deg) or np.any(el >= 90):
        raise ValueError("Sun must be fully above the local horizontal and below zenith")
    if not np.isfinite(slope_rc).all() or len(slope_rc) != 2:
        raise ValueError("receiving-plane slopes must be finite")
    ss = cfg.supersample
    sigma = np.hypot(cfg.psf_sigma_px, cfg.registration_sigma_px) * ss
    pad = int(np.ceil(4 * sigma / ss))
    padded = (shape[0] + 2 * pad, shape[1] + 2 * pad)
    rr = (np.arange(padded[0] * ss) + 0.5) / ss - 0.5 - root[0] - pad
    cc = (np.arange(padded[1] * ss) + 0.5) / ss - 0.5 - root[1] - pad
    rows, cols = rr[:, None] * cfg.pixel_m, cc[None, :] * cfg.pixel_m
    offsets = np.linspace(-0.8, 0.8, 5)
    weights = np.sqrt(1 - offsets**2)
    weights /= weights.sum()
    result, censored = [], False
    for a, e in zip(az, el):
        dr, dc = np.cos(a), -np.sin(a)  # down-Sun
        along, across = rows * dr + cols * dc, -rows * dc + cols * dr
        beta = slope_rc[0] * dr + slope_rc[1] * dc
        cover = np.zeros((len(rr), len(cc)), float)
        for off, weight in zip(offsets, weights):
            denom = np.tan(np.radians(e + off * cfg.solar_radius_deg)) + beta
            if denom <= 0:
                raise ValueError("receiving plane has no finite shadow intersection")
            length = height_m / denom
            er, ec = root[0] + dr * length / cfg.pixel_m, root[1] + dc * length / cfg.pixel_m
            censored |= not (1 <= er < shape[0] - 2 and 1 <= ec < shape[1] - 2)
            censored |= np.hypot(er-(shape[0]-1)/2,ec-(shape[1]-1)/2) > cfg.root_support_px - 0.75
            cover += weight * ((along >= 0) & (along <= length) & (abs(across) <= width_m / 2))
        # Render beyond the patch so a censored shadow does not acquire a false
        # blurred endpoint at the crop boundary.
        if sigma > 0:
            cover = ndi.gaussian_filter(cover, sigma, mode="constant")
        binned = cover.reshape(padded[0], ss, padded[1], ss).mean(axis=(1, 3))
        result.append(binned[pad:pad+shape[0], pad:pad+shape[1]])
    return np.asarray(result), bool(censored)


def fit_template(projector, residual, template, cfg):
    rt = projector.apply(template)
    energy = float(np.sum(rt * rt))
    raw_energy = float(np.sum((template[:, projector.common] / projector.sigma[:, None]) ** 2))
    ident = energy / max(raw_energy, 1e-30)
    if energy <= 1e-12 or ident < cfg.min_identifiability:
        return None
    amplitude = float(np.clip(-np.sum(rt * residual) / energy, 0, cfg.max_contrast))
    # This is the constrained likelihood improvement; do not report an
    # unconstrained z-score when the physical amplitude bound is active.
    delta = max(0.0, float(-2 * amplitude * np.sum(rt * residual) - amplitude**2 * energy))
    frame_delta = np.sum(residual**2 - (residual + amplitude * rt)**2, axis=1)
    return dict(score=float(np.sqrt(delta)), delta_chi2=delta, contrast=amplitude,
                identifiability=ident, frame_delta_chi2=frame_delta.tolist())


def propose_roots(stack, azimuths, sigma, cfg):
    """Directional darkening on temporal residuals; only a proposal stage.

    There is no connected-component size/width gate. The complete subsequent
    fit scores every frame, including contradictory observations.
    """
    common = np.isfinite(stack).all(axis=0)
    if common.sum() < 12:
        return [], False
    # Remove the same nuisance family before selection as before fitting. A
    # per-frame illumination plane must not consume the candidate budget.
    projector = NuisanceProjector(common, sigma)
    residual = np.zeros_like(stack)
    residual[:, common] = projector.apply(stack) * sigma[:, None]
    strength = np.zeros(stack.shape[1:], float)
    for frame, az in zip(residual, np.radians(azimuths)):
        dr, dc = np.cos(az), -np.sin(az)
        ahead = ndi.shift(frame, (-1.5 * dr, -1.5 * dc), order=1, mode="nearest")
        behind = ndi.shift(frame, (1.5 * dr, 1.5 * dc), order=1, mode="nearest")
        strength += np.maximum(behind - ahead, 0)
    edge = cfg.radius_px + 1
    allowed = ndi.binary_erosion(common, iterations=2)
    allowed[:edge] = allowed[-edge:] = False
    allowed[:, :edge] = allowed[:, -edge:] = False
    maxima = strength == ndi.maximum_filter(strength, 2 * cfg.min_separation_px + 1)
    coords = np.argwhere(allowed & maxima & (strength > 1e-10))
    order = sorted(coords.tolist(), key=lambda p: (-strength[tuple(p)], p[0], p[1]))
    # Plateaus must not spend the candidate budget on adjacent equivalent roots.
    selected = []
    for point in order:
        if all(np.linalg.norm(np.subtract(point, p)) >= cfg.min_separation_px for p in selected):
            selected.append(point)
        if len(selected) > cfg.max_candidates:
            break
    return selected[:cfg.max_candidates], len(selected) > cfg.max_candidates


def search_stack(stack, azimuths, elevations, sigma, cfg=None, *, slope_rc=(0., 0.)):
    """Search normalized radiance stack. Returns candidates and explicit status.

    No score threshold is applied and no safety labels are produced. Grid
    heights/widths are template parameters, NOT confidence intervals or
    independently resolved dimensions. Cap and mask exclusions are reported.
    """
    cfg = cfg or ShadowConfig()
    stack = np.asarray(stack, float)
    if stack.ndim != 3 or stack.shape[0] < 3 or min(stack.shape[1:]) < 2 * cfg.radius_px + 5:
        raise ValueError("need >=3 frames and a window larger than the template")
    if len(azimuths) != len(stack) or len(elevations) != len(stack):
        raise ValueError("one measured illumination per frame is required")
    sigma = np.broadcast_to(np.asarray(sigma, float), (len(stack),)).copy()
    if np.any(~np.isfinite(sigma) | (sigma <= 0)):
        raise ValueError("noise sigma must be finite and positive")
    # Validate geometry even if a stationary stack produces no proposals.
    size = 2 * cfg.radius_px + 1
    base = cfg.radius_px
    templates = []
    for dy in cfg.root_offsets_px:
        for dx in cfg.root_offsets_px:
            for h in cfg.heights_m:
                for w in cfg.widths_m:
                    t, censored = shadow_template((size, size), (base + dy, base + dx),
                                                  azimuths, elevations, h, w, cfg, slope_rc)
                    templates.append((dy, dx, h, w, t, censored))
    points, truncated = propose_roots(stack, azimuths, sigma, cfg)
    yy, xx = np.indices((size,size))
    support = np.hypot(yy-base,xx-base) <= cfg.root_support_px
    candidates, excluded = [], 0
    for row, col in points:
        patch = stack[:, row-base:row+base+1, col-base:col+base+1]
        common = np.isfinite(patch).all(axis=0) & support
        fraction = common.sum()/support.sum()
        if fraction < cfg.min_common_fraction:
            excluded += 1
            continue
        projector = NuisanceProjector(common, sigma)
        residual = projector.apply(patch)
        best = None
        for dy, dx, h, w, template, censored in templates:
            fit = fit_template(projector, residual, template, cfg)
            if fit is not None and (best is None or fit["score"] > best["score"]):
                best = dict(fit, row_px=row+dy, col_px=col+dx,
                            template_height_m=h, template_width_m=w,
                            endpoint_censored=censored, common_fraction=float(fraction))
        if best is not None:
            candidates.append(best)
        else:
            excluded += 1
    candidates.sort(key=lambda c: -c["score"])
    # A physical root can generate several nearby proposals. Keep the best fit.
    distinct = []
    for c in candidates:
        if all(np.hypot(c["row_px"]-p["row_px"], c["col_px"]-p["col_px"]) >=
               cfg.min_separation_px for p in distinct):
            distinct.append(c)
    return dict(status="experimental_unvalidated", config=asdict(cfg), config_hash=cfg.hash(),
                candidates=distinct, proposals=len(points), search_truncated=truncated,
                excluded_proposals=excluded, common_fraction=float(np.isfinite(stack).all(axis=0).mean()),
                noise_sigma=sigma.tolist(), slope_rc=list(slope_rc),
                score_interpretation="Constrained Gaussian likelihood ranking; no calibrated field p-value",
                absence_interpretation="No candidate is not evidence of safe terrain")


def gaussian_search_calibration(result, stack, azimuths, elevations, sigma, cfg,
                                *, trials=99, seed=2718, slope_rc=(0., 0.)):
    """Full-search Monte Carlo rank under the explicit independent Gaussian H0.

    With fixed configuration, mask and KNOWN noise, H0 nuisance components are
    projected out before selection and fitting. Rerunning proposals and every
    template on each null draw includes the look-elsewhere effect. Real NAC
    residuals are spatially correlated and sigma is uncertain: these probabilities
    are model checks, not a measured lunar false-positive rate. Do not tune the
    grid after seeing them and then reuse the same calibration.
    """
    if trials < 19:
        raise ValueError("at least 19 trials required; 99 or more recommended")
    if result.get('config_hash') != cfg.hash():
        raise ValueError("calibration must use the searched configuration")
    sigma = np.broadcast_to(np.asarray(sigma,float), (len(stack),))
    mask = np.isfinite(stack)
    rng = np.random.default_rng(seed)
    maxima = []
    for _ in range(trials):
        noise = rng.normal(size=np.shape(stack))*sigma[:,None,None]
        null = search_stack(np.where(mask,noise,np.nan), azimuths,elevations,sigma,cfg,slope_rc=slope_rc)
        maxima.append(max([c['score'] for c in null['candidates']] or [0.]))
    observed = max([c['score'] for c in result['candidates']] or [0.])
    return dict(model="Independent Gaussian noise with fixed known frame sigma and fixed mask",
                scope="Global no-moving-shadow null; complete search rerun; not field calibration",
                trials=trials, seed=seed, null_max_scores=maxima,
                observed_max_score=observed,
                global_p=(1+sum(v >= observed for v in maxima))/(trials+1),
                minimum_attainable_p=1/(trials+1))


def map_sun_azimuth(crs, transform, lat, lon, ground_azimuth):
    """Project a measured local bearing into an arbitrary raster via its Jacobian.

    Handles either pole and rotated affine grids. Angles are map-image angles,
    so a positive column direction is clockwise from image up.
    """
    from pyproj import CRS, Geod, Transformer
    target = CRS.from_user_input(crs)
    source = target.geodetic_crs
    ellipsoid = source.ellipsoid
    geod = Geod(a=ellipsoid.semi_major_metre, b=ellipsoid.semi_minor_metre)
    forward = Transformer.from_crs(source, target, always_xy=True)
    lon1, lat1, _ = geod.fwd(lon, lat, ground_azimuth, 1.0)
    x0, y0 = forward.transform(lon, lat)
    x1, y1 = forward.transform(lon1, lat1)
    c0, r0 = (~transform) * (x0, y0)
    c1, r1 = (~transform) * (x1, y1)
    if not np.isfinite([r0, c0, r1, c1]).all() or np.hypot(r1-r0, c1-c0) < 1e-8:
        raise ValueError("degenerate map bearing")
    return float(np.degrees(np.arctan2(c1-c0, -(r1-r0))) % 360)
