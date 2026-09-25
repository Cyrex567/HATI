"""Multi-image shape from shading on the aligned stack (campaign T14): linearised and Gauss-Newton.

Ratio images remove albedo: Y_k / Ybar - 1 = -grad(h) . w_k + p_k(x) + noise,
where w_k is cot(e_k) times the horizontal direction toward the Sun, minus its
mean over frames, and p_k is a brightness plane per frame. The height h lives on
a coarse grid (grid_px image pixels) with bilinear interpolation and a
second-difference smoothness penalty; together they set the effective
resolution. The system is solved by sparse least squares (LSQR).

Samples darker than dark_ratio times the frame mean are left out of that frame's
equations, so deep cast shadows, rock shadows included, are not fitted as
slopes. The surface is relative: its mean and any plane shared by every frame
are not observed. Linear shading holds for slopes below the Sun elevation, so
steep walls are smoothed and their slopes underestimated.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from .noise_scale import sun_loadings


def _bilinear(shape, grid_px):
    H, W = shape
    Hc, Wc = (H-1)//grid_px+2, (W-1)//grid_px+2
    r, c = np.indices(shape, dtype=float).reshape(2, -1)/grid_px
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    fr, fc = r-r0, c-c0
    rows, cols, vals = [], [], []
    for dr, dc, w in ((0, 0, (1-fr)*(1-fc)), (1, 0, fr*(1-fc)), (0, 1, (1-fr)*fc), (1, 1, fr*fc)):
        rows.append(np.arange(H*W)); cols.append((r0+dr)*Wc+(c0+dc)); vals.append(w)
    matrix = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(H*W, Hc*Wc))
    return matrix, (Hc, Wc)


def _derivative(n, spacing):
    """Central differences, one-sided at the ends."""
    main = np.zeros(n); upper = np.full(n-1, .5); lower = np.full(n-1, -.5)
    d = sparse.diags([lower, main, upper], [-1, 0, 1], shape=(n, n)).tolil()
    d[0, 0], d[0, 1] = -1., 1.
    d[n-1, n-2], d[n-1, n-1] = -1., 1.
    return d.tocsr()/spacing


def _smoothness(shape):
    Hc, Wc = shape
    def second(n):
        return sparse.diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0, 1, 2], shape=(n-2, n))
    def first(n):
        return sparse.diags([-np.ones(n-1), np.ones(n-1)], [0, 1], shape=(n-1, n))
    return sparse.vstack([sparse.kron(second(Hc), sparse.identity(Wc)),
                          sparse.kron(sparse.identity(Hc), second(Wc)),
                          np.sqrt(2)*sparse.kron(first(Hc), first(Wc))]).tocsr()


def solve_sfs(stack, valid, azimuths, elevations, pixel_m, *, grid_px=2, smoothness=1.,
              dark_ratio=.5, iterations=600, tolerance=1e-8, passes=1, shadow_sigma=3.):
    """Fit a relative height field; return it with its predicted shading and a corrected stack.

    corrected = stack - Ybar * predicted relief ratio, on pixels valid in every
    frame; elsewhere the stack is unchanged. The static part of the shading is
    not removed: it cannot be told from albedo and the detector's null absorbs it.
    With passes > 1, samples more than shadow_sigma robust scales darker than
    the previous fit are treated as cast shadows and left out of the next one.
    """
    stack = np.asarray(stack, float)
    n, H, W = stack.shape
    if n < 3 or type(grid_px) is not int or grid_px < 1 or smoothness < 0 or not 0 <= dark_ratio < 1:
        raise ValueError('need three frames, an integer grid and nonnegative smoothness')
    if type(passes) is not int or passes < 1 or not shadow_sigma > 0:
        raise ValueError('passes must be a positive integer and shadow_sigma positive')
    usable = np.asarray(valid, bool) & np.isfinite(stack)
    common = usable.all(axis=0)
    if common.sum() < 100:
        raise ValueError('too few pixels valid in every frame')
    count = usable.sum(axis=0)
    mean = np.where(common, np.where(usable, stack, 0.).sum(axis=0)/np.maximum(count, 1), np.nan)
    common &= mean > 0
    ratio = stack/np.where(common, mean, 1.)[None]-1
    loadings = sun_loadings(azimuths, elevations)
    w = loadings-loadings.mean(axis=0)
    if np.linalg.matrix_rank(w) < 2:
        raise ValueError('illumination directions do not constrain both slope components')
    interp, coarse = _bilinear((H, W), grid_px)
    grad_r = (sparse.kron(_derivative(H, pixel_m), sparse.identity(W))@interp).tocsr()
    grad_c = (sparse.kron(sparse.identity(H), _derivative(W, pixel_m))@interp).tocsr()
    rr, cc = np.indices((H, W), dtype=float)
    rr = ((rr-H/2)/H).ravel(); cc = ((cc-W/2)/W).ravel()
    unknowns = interp.shape[1]
    penalty = _smoothness(coarse)
    regulariser = sparse.vstack([sparse.hstack([np.sqrt(smoothness)*penalty, sparse.csr_matrix((penalty.shape[0], 3*n))]),
                                 sparse.hstack([1e-6*sparse.identity(unknowns), sparse.csr_matrix((unknowns, 3*n))])])

    def solve(kept, start):
        blocks, targets = [], []
        for k in range(n):
            idx = np.flatnonzero(kept[k].ravel())
            slope_part = -(w[k, 0]*grad_r[idx]+w[k, 1]*grad_c[idx])
            plane = sparse.csr_matrix((np.column_stack([np.ones(len(idx)), rr[idx], cc[idx]]).ravel(),
                                       (np.repeat(np.arange(len(idx)), 3), np.tile(np.arange(3*k, 3*k+3), len(idx)))),
                                      shape=(len(idx), 3*n))
            blocks.append(sparse.hstack([slope_part, plane]))
            targets.append(ratio[k].ravel()[idx])
        system = sparse.vstack([sparse.vstack(blocks), regulariser]).tocsr()
        rhs = np.concatenate([*targets, np.zeros(regulariser.shape[0])])
        return lsqr(system, rhs, atol=tolerance, btol=tolerance, iter_lim=iterations, x0=start)

    def evaluate(solution):
        heights, planes = solution[0][:unknowns], solution[0][unknowns:].reshape(n, 3)
        gr, gc = (grad_r@heights).reshape(H, W), (grad_c@heights).reshape(H, W)
        predicted = -(w[:, 0, None, None]*gr[None]+w[:, 1, None, None]*gc[None])
        plane_part = (planes[:, 0, None, None]+planes[:, 1, None, None]*rr.reshape(H, W)[None]
                      + planes[:, 2, None, None]*cc.reshape(H, W)[None])
        return heights, planes, gr, gc, predicted, ratio-plane_part

    kept = np.asarray([common & (stack[k] >= dark_ratio*mean) for k in range(n)])
    dark_kept = kept.copy()
    solution = solve(kept, None)
    for _ in range(passes-1):
        # Samples far darker than the fitted shading are cast shadows, rock shadows
        # included: leave them out and refit, so the surface does not bend to them.
        *_, predicted, before = evaluate(solution)
        left = (before-predicted)[kept]
        scale = 1.4826*float(np.median(np.abs(left-np.median(left))))
        kept = dark_kept & ((before-predicted) >= -shadow_sigma*scale)
        solution = solve(kept, solution[0])
    heights, planes, gr, gc, predicted, before = evaluate(solution)
    h = (interp@heights).reshape(H, W)
    used = kept
    after = before-predicted
    explained = 1-float(np.sum(after[used]**2)/max(np.sum(before[used]**2), 1e-30))
    corrected = np.where(common[None], stack-np.where(common, mean, 0.)[None]*predicted, stack)
    relative = np.where(common, h-h[common].mean(), np.nan)
    slope = np.where(common, np.degrees(np.arctan(np.hypot(gr, gc))), np.nan)
    return dict(height_m=relative, slope_deg=slope, predicted_ratio=np.where(common[None], predicted, np.nan),
                corrected=corrected, common=common, used=used, planes=planes, explained_fraction=explained,
                ratio_rms_before=float(np.sqrt(np.mean(before[used]**2))),
                ratio_rms_after=float(np.sqrt(np.mean(after[used]**2))),
                used_fraction_per_frame=[float(u[common].mean()) for u in used],
                shadow_excluded_fraction=float(1-used[dark_kept].mean()),
                lsqr_stop=int(solution[1]), lsqr_iterations=int(solution[2]),
                configuration=dict(grid_px=grid_px, smoothness=smoothness, dark_ratio=dark_ratio, passes=passes,
                                   shadow_sigma=shadow_sigma, iterations=iterations, tolerance=tolerance, pixel_m=pixel_m))


def _sun_vector(azimuth_deg, elevation_deg):
    """Unit vector toward the Sun in (row, col, up); azimuth clockwise from map up."""
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array([-np.cos(a)*np.cos(e), np.sin(a)*np.cos(e), np.sin(e)])


def lunar_lambert(p, q, sun, L=.5):
    """Lunar-Lambert brightness relative to flat ground, nadir view, and its slope derivatives.

    p and q are dh/drow and dh/dcol in m/m. Returns R, dR/dp and dR/dq, with R
    and its derivatives zero where the surface faces away from the Sun.
    """
    norm = np.sqrt(1+p*p+q*q)
    mu0 = (-p*sun[0]-q*sun[1]+sun[2])/norm
    mu = 1/norm
    flat = 2*L*sun[2]/(sun[2]+1)+(1-L)*sun[2]
    lit = mu0 > 0
    m0 = np.where(lit, mu0, 0.)
    R = (2*L*m0/(m0+mu)+(1-L)*m0)/flat
    dR_dmu0 = (2*L*mu/(m0+mu)**2+(1-L))/flat
    dR_dmu = -2*L*m0/(m0+mu)**2/flat
    dmu0_dp, dmu0_dq = -sun[0]/norm-mu0*p/norm**2, -sun[1]/norm-mu0*q/norm**2
    dmu_dp, dmu_dq = -p/norm**3, -q/norm**3
    return (R, np.where(lit, dR_dmu0*dmu0_dp+dR_dmu*dmu_dp, 0.),
            np.where(lit, dR_dmu0*dmu0_dq+dR_dmu*dmu_dq, 0.))


def solve_sfs_nonlinear(stack, valid, azimuths, elevations, pixel_m, *, grid_px=1, smoothness=1.,
                        dark_ratio=.5, iterations=800, tolerance=1e-8, gauss_newton=6, shadow_sigma=3.,
                        dark_model=.35, lunar_lambert_l=.5, max_correction=2.):
    """Non-linear multi-image shape from shading in log brightness, by Gauss-Newton.

    log Y_k(x) = a(x) + b_k(x) + log R_k(grad h(x)) + noise: a is the per-pixel
    albedo, removed by subtracting the mean over frames; b_k is a brightness
    plane per frame; R_k is the Lunar-Lambert brightness relative to flat ground
    under frame k's Sun. At grazing Sun a slope of one or two degrees already
    moves R far from linear, which the linearised solver cannot follow. Each
    Gauss-Newton step solves the linear solver's sparse problem with slope
    sensitivities evaluated on the current surface, with a step-halving line
    search. Left out of the fit: samples observed dark (below dark_ratio of the
    pixel mean), samples the model puts near self-shadow (R below dark_model),
    and, from the second step, samples more than shadow_sigma robust scales
    darker than the model (cast shadows, rock shadows included).

    Returns the same fields as solve_sfs. The corrected stack divides out each
    frame's predicted shading relative to the mean over frames, clamped to
    max_correction and left untouched where the model predicts self-shadow.
    """
    stack = np.asarray(stack, float)
    n, H, W = stack.shape
    if n < 3 or type(grid_px) is not int or grid_px < 1 or smoothness < 0 or not 0 <= dark_ratio < 1:
        raise ValueError('need three frames, an integer grid and nonnegative smoothness')
    if type(gauss_newton) is not int or gauss_newton < 1 or not shadow_sigma > 0 or not 0 < dark_model < 1 or not max_correction > 1:
        raise ValueError('invalid Gauss-Newton settings')
    usable = np.asarray(valid, bool) & np.isfinite(stack) & (stack > 0)
    common = usable.all(axis=0)
    if common.sum() < 100:
        raise ValueError('too few pixels valid in every frame')
    log_y = np.log(np.where(usable, stack, 1.))
    z = log_y-log_y.mean(axis=0)[None]
    mean = np.where(common, stack.mean(axis=0), np.nan)
    observed = np.asarray([common & (stack[k] >= dark_ratio*mean) for k in range(n)])
    suns = [_sun_vector(a, e) for a, e in zip(azimuths, elevations)]
    interp, coarse = _bilinear((H, W), grid_px)
    grad_r = (sparse.kron(_derivative(H, pixel_m), sparse.identity(W))@interp).tocsr()
    grad_c = (sparse.kron(sparse.identity(H), _derivative(W, pixel_m))@interp).tocsr()
    rr, cc = np.indices((H, W), dtype=float)
    rr = ((rr-H/2)/H).ravel(); cc = ((cc-W/2)/W).ravel()
    unknowns = interp.shape[1]
    penalty = np.sqrt(smoothness)*_smoothness(coarse)
    anchor = sparse.hstack([1e-6*sparse.identity(unknowns), sparse.csr_matrix((unknowns, 3*n))])
    regulariser = sparse.vstack([sparse.hstack([penalty, sparse.csr_matrix((penalty.shape[0], 3*n))]), anchor])

    def model(heights):
        p, q = (grad_r@heights).reshape(H, W), (grad_c@heights).reshape(H, W)
        R, dp, dq = (np.asarray(v) for v in zip(*[lunar_lambert(p, q, s, lunar_lambert_l) for s in suns]))
        rho = np.log(np.maximum(R, dark_model))
        lit = R > dark_model
        return p, q, R, np.where(lit, dp/np.maximum(R, dark_model), 0.), np.where(lit, dq/np.maximum(R, dark_model), 0.), lit, rho

    def planes_image(planes):
        return (planes[:, 0, None]+planes[:, 1, None]*rr[None]+planes[:, 2, None]*cc[None]).reshape(n, H, W)

    def cost(heights, planes, kept):
        *_, rho = model(heights)
        left = (z-(rho-rho.mean(axis=0))-planes_image(planes))[kept]
        smooth = penalty@heights
        return float(left@left+smooth@smooth)

    heights, planes = np.zeros(unknowns), np.zeros((n, 3))
    shadow_keep = np.ones((n, H, W), bool)
    history, lsqr_total, stop = [], 0, 0
    for step_index in range(gauss_newton):
        p, q, R, jr, jc, lit, rho = model(heights)
        kept = observed & lit & shadow_keep
        residual = z-(rho-rho.mean(axis=0))-planes_image(planes)
        jr, jc = jr-jr.mean(axis=0), jc-jc.mean(axis=0)
        blocks, targets = [], []
        for k in range(n):
            idx = np.flatnonzero(kept[k].ravel())
            slope_part = sparse.diags(jr[k].ravel()[idx])@grad_r[idx]+sparse.diags(jc[k].ravel()[idx])@grad_c[idx]
            plane = sparse.csr_matrix((np.column_stack([np.ones(len(idx)), rr[idx], cc[idx]]).ravel(),
                                       (np.repeat(np.arange(len(idx)), 3), np.tile(np.arange(3*k, 3*k+3), len(idx)))),
                                      shape=(len(idx), 3*n))
            blocks.append(sparse.hstack([slope_part, plane]))
            targets.append(residual[k].ravel()[idx])
        system = sparse.vstack([sparse.vstack(blocks), regulariser]).tocsr()
        rhs = np.concatenate([*targets, -(penalty@heights), np.zeros(unknowns)])
        solution = lsqr(system, rhs, atol=tolerance, btol=tolerance, iter_lim=iterations)
        lsqr_total += int(solution[2]); stop = int(solution[1])
        delta_h, delta_p = solution[0][:unknowns], solution[0][unknowns:].reshape(n, 3)
        before = cost(heights, planes, kept)
        step = 1.
        for _ in range(5):
            trial = cost(heights+step*delta_h, planes+step*delta_p, kept)
            if trial < before:
                break
            step /= 2
        else:
            history.append(dict(step=step_index, cost=before, accepted=False)); break
        heights, planes = heights+step*delta_h, planes+step*delta_p
        history.append(dict(step=step_index, cost=trial, relative_change=(before-trial)/before, step_length=step,
                            lsqr_iterations=int(solution[2]), kept_fraction=float(kept[:, common].mean())))
        if step_index == 0:
            # Cast shadows from the first fit: far darker than any modelled shading.
            *_, rho = model(heights)
            left = z-(rho-rho.mean(axis=0))-planes_image(planes)
            values = left[kept]
            scale = 1.4826*float(np.median(np.abs(values-np.median(values))))
            shadow_keep = left >= -shadow_sigma*scale
        elif (before-trial)/before < 1e-3:
            break
    p, q, R, _, _, lit, rho = model(heights)
    kept = observed & lit & shadow_keep
    shading = rho-rho.mean(axis=0)
    before = z-planes_image(planes)
    after = before-shading
    explained = 1-float(np.sum(after[kept]**2)/max(np.sum(before[kept]**2), 1e-30))
    factor = np.clip(np.exp(-shading), 1/max_correction, max_correction)
    corrected = np.where(common[None] & lit, stack*factor, stack)
    h = (interp@heights).reshape(H, W)
    return dict(height_m=np.where(common, h-h[common].mean(), np.nan),
                slope_deg=np.where(common, np.degrees(np.arctan(np.hypot(p, q))), np.nan),
                predicted_ratio=np.where(common[None], np.exp(shading)-1, np.nan), corrected=corrected,
                common=common, used=kept, planes=planes, explained_fraction=explained,
                ratio_rms_before=float(np.sqrt(np.mean(before[kept]**2))), ratio_rms_after=float(np.sqrt(np.mean(after[kept]**2))),
                used_fraction_per_frame=[float(u[common].mean()) for u in kept],
                shadow_excluded_fraction=float(1-shadow_keep[observed].mean()),
                model_shadow_fraction=float(1-lit[:, common].mean()),
                lsqr_stop=stop, lsqr_iterations=lsqr_total, gauss_newton=history,
                configuration=dict(model='nonlinear', grid_px=grid_px, smoothness=smoothness, dark_ratio=dark_ratio,
                                   dark_model=dark_model, shadow_sigma=shadow_sigma, gauss_newton=gauss_newton,
                                   iterations=iterations, tolerance=tolerance, lunar_lambert_l=lunar_lambert_l,
                                   max_correction=max_correction, pixel_m=pixel_m))


def rock_factor(shape, sites, azimuths, elevations, pixel_m, *, seed, supersample=4, window_px=32):
    """Multiplicative brightness of injected rocks (shadow plus lit faces) on flat ground.

    sites: [(row, col, height_m)]. Rendered with the independent relief generator
    without noise, texture or planes, then applied as stack * factor so the real
    albedo and relief stay underneath.
    """
    from .relief_scenes import render_relief
    from .rock_scenes import make_rock
    factor = np.ones((len(azimuths), *shape))
    half = window_px//2
    for i, (r, c, height) in enumerate(sites):
        r0, c0 = int(r)-half, int(c)-half
        if r0 < 0 or c0 < 0 or r0+window_px > shape[0] or c0+window_px > shape[1]:
            raise ValueError('injection window must lie inside the image')
        rock = make_rock(seed+i, (r-r0, c-c0), height, .6, aspect=1.35)
        local = render_relief((window_px, window_px), azimuths, elevations, pixel_m=pixel_m, seed=seed+i, noise=0.,
                              rocks=[rock], supersample=supersample, texture=0., stain=0., frame_plane=0.)
        factor[:, r0:r0+window_px, c0:c0+window_px] *= local['stack']
    return factor
