"""Footprint maxima over every sampled root, without cell replication."""
import numpy as np
from .landing_terrain import buffer_evidence


def buffer_roots(roots, assessed, pixel_m, radius_m, score_scale=8.):
    """Roots are N x 3 (row, col, score) in pixel-centre index coordinates.

    The disk is centred on each output pixel centre. Coverage remains a
    conservative disk test on assessed cells; high evidence survives incomplete
    coverage. This is a finite root bank, not continuous obstacle completeness.
    Ties are resolved by row, then column. Returns the actual controlling root.
    """
    roots = np.asarray(roots, float).reshape(-1, 3)
    assessed = np.asarray(assessed, bool)
    if assessed.ndim != 2 or not np.isfinite([pixel_m, radius_m, score_scale]).all() or pixel_m <= 0 or radius_m < 0 or score_scale <= 0:
        raise ValueError('invalid footprint dimensions or score scale')
    if not np.isfinite(roots).all() or np.any(roots[:, 2] < 0):
        raise ValueError('root coordinates and nonnegative scores must be finite')
    _, complete = buffer_evidence(np.where(assessed, 0., np.nan), pixel_m, radius_m)
    h, w = assessed.shape
    # Highest rank means largest score, with the smallest coordinate winning ties.
    order = np.lexsort((-roots[:, 1], -roots[:, 0], roots[:, 2]))
    roots = roots[order]
    winner = np.full(h*w, -1, dtype=np.int64)
    base = np.floor(roots[:, :2]).astype(int)
    reach = int(np.ceil(radius_m / pixel_m)) + 1
    rank = np.arange(len(roots))
    for dr in range(-reach, reach+1):
        for dc in range(-reach, reach+1):
            rr, cc = base[:, 0]+dr, base[:, 1]+dc
            inside = ((rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                      & (((roots[:, 0]-rr)**2 + (roots[:, 1]-cc)**2)*pixel_m**2 <= radius_m**2+1e-10))
            np.maximum.at(winner, rr[inside]*w+cc[inside], rank[inside])
    found = winner >= 0
    source = np.full((h*w, 3), np.nan)
    source[found] = roots[winner[found]]
    source = source.reshape(h, w, 3)
    peak = source[..., 2] / (source[..., 2]+score_scale)
    index = np.where(complete | (peak >= .5), peak, np.nan)
    return dict(index=index, complete=complete, root_row=source[..., 0],
                root_col=source[..., 1], score=source[..., 2])
