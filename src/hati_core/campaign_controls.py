"""Independent synthetic controls for the saturation campaign.

No call to the detector renderer: rounded cross-section shadows are integrated
on a finer grid and blurred anisotropically. These scenes test model transfer,
not lunar performance. Inputs and random seeds are fixed before fitting.
"""
import numpy as np
from scipy import ndimage as ndi


def render_control(shape, azimuths, elevations, *, pixel_m, seed, noise,
                   kind='caster', height=.3, width=.6, slope_rc=(0., 0.),
                   root=None, supersample=8):
    rng = np.random.default_rng(seed)
    root = root or ((shape[0]-1)/2+.3, (shape[1]-1)/2+.2)
    ss = supersample
    yy, xx = np.indices((shape[0]*ss, shape[1]*ss), dtype=float)
    yy = ((yy+.5)/ss-.5-root[0])*pixel_m
    xx = ((xx+.5)/ss-.5-root[1])*pixel_m
    rows, cols = np.indices(shape, dtype=float)
    texture = .04*ndi.gaussian_filter(rng.normal(size=shape), 1.2)
    stain = .3*np.exp(-((rows-shape[0]*.3)**2+(cols-shape[1]*.7)**2)/8)
    frames = []
    for i, (az, el) in enumerate(zip(azimuths, elevations)):
        dr, dc = np.cos(np.radians(az)), -np.sin(np.radians(az))
        denominator = np.tan(np.radians(el))+slope_rc[0]*dr+slope_rc[1]*dc
        if denominator <= 0:
            raise ValueError('synthetic receiving plane has no finite shadow intersection')
        cast = np.zeros_like(yy)
        if kind in ('caster', 'overlap', 'resolved_ridge'):
            for shift in ((0., 0.), (1.7*pixel_m, 2.1*pixel_m)) if kind == 'overlap' else ((0., 0.),):
                along = (yy-shift[0])*dr+(xx-shift[1])*dc
                across = -(yy-shift[0])*dc+(xx-shift[1])*dr
                extent = width if kind != 'resolved_ridge' else 9*pixel_m
                local_height = height*np.sqrt(np.maximum(0, 1-(across/(extent/2))**2))
                cast = np.maximum(cast, (along >= 0) & (along < local_height/denominator) & (abs(across) < extent/2))
        cast = ndi.gaussian_filter(cast.astype(float), (.52*ss, .68*ss))
        cast = cast.reshape(shape[0], ss, shape[1], ss).mean(axis=(1, 3))
        frame = 1+texture-stain+.0003*i*(rows-cols)-.72*cast
        if kind == 'structured_null':
            frame += .12*np.sin((rows+cols*.7)/3+i*.8)
        white = rng.normal(size=shape)
        colored = ndi.gaussian_filter(rng.normal(size=shape), .8)
        frame += noise*(.8*white+.6*colored/max(colored.std(), 1e-12))
        frames.append(frame)
    return np.asarray(frames)
