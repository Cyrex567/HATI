"""Independent, seeded triangular-rock shadow scenes on a receiving plane.

No detector renderer is imported. Triangle projection, solar-disc quadrature,
pixel integration and anisotropic blur form a deliberately different forward
model. Photometry is illustrative; these are not simulated calibrated NAC DN.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull


NASA_CREDIT = ('NASA/These 3D reconstructed image data were produced at the Lunar Sample Laboratory '
               'Facility for Astromaterials 3D in NASA\'s Acquisition & Curation Office and were funded '
               'by NASA Planetary Data Archiving, Restoration, and Tools Program, Proposal No.: 15-PDART15_2-0041.')


def mesh_hash(vertices, faces):
    return hashlib.sha256(np.asarray(vertices, '<f8').tobytes()+np.asarray(faces, '<i8').tobytes()).hexdigest()


def read_obj(path):
    """Geometry only: ignore materials/external references; support polygon faces."""
    vertices, faces = [], []
    for line in Path(path).read_text(encoding='utf-8', errors='strict').splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'v':
            if len(fields) < 4:
                raise ValueError('OBJ vertex needs three coordinates')
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == 'f':
            indices = [int(x.split('/')[0]) for x in fields[1:]]
            if len(indices) < 3 or 0 in indices:
                raise ValueError('invalid OBJ face')
            indices = [x-1 if x > 0 else len(vertices)+x for x in indices]
            if any(x < 0 or x >= len(vertices) for x in indices):
                raise ValueError('OBJ face references unavailable vertex')
            faces.extend((indices[0], indices[i], indices[i+1]) for i in range(1, len(indices)-1))
    v, f = np.asarray(vertices, float), np.asarray(faces, int)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.isfinite(v).all() or not len(f):
        raise ValueError('finite three-dimensional OBJ geometry required')
    if np.linalg.matrix_rank(v-v.mean(axis=0)) != 3:
        raise ValueError('rock mesh must have three-dimensional extent')
    return v, f


def convex_proxy(vertices, directions=96):
    """Bounded-cost shape proxy; explicitly discards concavities and fine detail."""
    v = np.asarray(vertices, float)
    z = 1-2*(np.arange(directions)+.5)/directions
    a = np.arange(directions)*np.pi*(3-np.sqrt(5))
    rays = np.column_stack([np.sqrt(1-z*z)*np.cos(a), np.sqrt(1-z*z)*np.sin(a), z])
    # Avoid a vertices-by-directions allocation for large source meshes.
    chosen = np.unique([np.argmax(v@ray) for ray in rays])
    points = v[chosen]
    hull = ConvexHull(points)
    return points, hull.simplices


def procedural_mesh(seed):
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(42, 3)); directions /= np.linalg.norm(directions, axis=1)[:, None]
    vertices = directions*rng.uniform(.65, 1.1, (42, 1))
    return vertices, ConvexHull(vertices).simplices


def load_catalog(manifest, *, split='evaluation'):
    """Verify local payloads and require parent-rock separation across splits."""
    path = Path(manifest); catalog = json.loads(path.read_text(encoding='utf-8'))
    if catalog.get('schema_version') != 1:
        raise ValueError('unsupported rock catalog schema')
    groups = {}; hashes = {}; ids = set(); selected = []
    for row in catalog['meshes']:
        for key in ('id', 'parent_rock', 'split', 'path', 'sha256', 'source_url', 'credit', 'geometry_assumptions'):
            if not row.get(key):
                raise ValueError('mesh provenance field missing: '+key)
        if row['id'] in ids or row['split'] not in ('development', 'evaluation'):
            raise ValueError('unique mesh IDs and declared splits required')
        ids.add(row['id'])
        for mapping, key in ((groups, row['parent_rock']), (hashes, row['sha256'])):
            if key in mapping and mapping[key] != row['split']:
                raise ValueError('rock identity or payload appears in both splits')
            mapping[key] = row['split']
        payload = (path.parent/row['path']).resolve()
        if not payload.is_relative_to(path.parent.resolve()):
            raise ValueError('mesh payload must stay inside the catalog folder')
        if hashlib.sha256(payload.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('mesh checksum mismatch')
        if row['split'] == split:
            v, f = read_obj(payload)
            if len(f) > 256:
                v, f = convex_proxy(v)
                reduction = '96-direction convex proxy; concavities and fine detail removed'
            else:
                reduction = 'original triangles'
            selected.append(dict(vertices=v, faces=f, provenance=dict(**row, reduction=reduction)))
    return selected


def make_rock(seed, root, height_m, width_m, *, aspect=1., burial=.1, mesh=None, yaw_deg=None):
    """Height is the exposed vertical maximum; width is pre-yaw body-axis extent.

    Source sample units are deliberately replaced by the requested dimensions.
    Catalog-derived scenes therefore test assumed rescaled shapes, not measured
    lunar boulder size distributions. Placement axes: row, column, up.
    """
    if not np.isfinite([height_m, width_m, aspect, burial, *root]).all() or min(height_m, width_m, aspect) <= 0 or not 0 <= burial < 1:
        raise ValueError('positive dimensions and burial in [0,1) required')
    rng = np.random.default_rng(seed)
    if mesh is None:
        vertices, faces = procedural_mesh(seed)
        provenance = dict(source='procedural_convex_rock', seed=seed, split='synthetic_evaluation')
    else:
        vertices, faces = mesh['vertices'].copy(), mesh['faces'].copy()
        provenance = mesh['provenance']
    v = np.array(vertices, float, copy=True)
    span = np.ptp(v, axis=0)
    if np.any(span <= 0):
        raise ValueError('mesh has zero dimension')
    v[:, :2] = (v[:, :2]-(v[:, :2].min(axis=0)+v[:, :2].max(axis=0))/2)/span[:2]
    v[:, 0] *= width_m*aspect; v[:, 1] *= width_m
    v[:, 2] = ((v[:, 2]-v[:, 2].min())/span[2]-burial)*height_m/(1-burial)
    yaw = float(rng.uniform(0, 360) if yaw_deg is None else yaw_deg)
    a = np.radians(yaw); rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    v[:, :2] = v[:, :2]@rotation.T
    return dict(vertices=v, faces=np.asarray(faces, int), root=np.asarray(root, float),
                truth=dict(seed=seed, root_px=list(map(float, root)), exposed_height_m=float(height_m),
                           body_width_m=float(width_m), body_length_m=float(width_m*aspect),
                           yaw_deg=yaw, burial_fraction=float(burial),
                           placed_mesh_sha256=mesh_hash(v, faces), source=provenance))


def _above_ground(triangle):
    """Clip a triangle against relative z=0; return polygon vertices."""
    out = []
    for start, end in zip(triangle, np.roll(triangle, -1, axis=0)):
        if start[2] >= 0:
            out.append(start)
        if (start[2] >= 0) != (end[2] >= 0):
            out.append(start+(end-start)*(-start[2]/(end[2]-start[2])))
    return out


def _raster_triangle(canvas, points, pixel_m, supersample):
    """Rasterize a projected triangle at physical supersample centres."""
    p = np.asarray(points)*supersample/pixel_m+(supersample-1)/2
    low = np.maximum(0, np.floor(p.min(axis=0)).astype(int))
    high = np.minimum(np.array(canvas.shape)-1, np.ceil(p.max(axis=0)).astype(int))
    if np.any(low > high):
        return
    r, c = np.mgrid[low[0]:high[0]+1, low[1]:high[1]+1]
    a, b, d = p; determinant = (b[1]-d[1])*(a[0]-d[0])+(d[0]-b[0])*(a[1]-d[1])
    if abs(determinant) < 1e-12:
        return
    u = ((b[1]-d[1])*(r-d[0])+(d[0]-b[0])*(c-d[1]))/determinant
    v = ((d[1]-a[1])*(r-d[0])+(a[0]-d[0])*(c-d[1]))/determinant
    canvas[low[0]:high[0]+1, low[1]:high[1]+1] |= (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9)


def render_rocks(shape, azimuths, elevations, rocks, *, pixel_m, seed, noise=.015,
                 slope_rc=(0., 0.), supersample=6, solar_radius_deg=.266,
                 registration_sigma_px=0., structured_null=False):
    """Independent projected-triangle union, including finite-Sun penumbrae.

    Casters cast onto one plane. Rock-body reflectance, mutual illumination,
    multiple scattering and arbitrary terrain intersections are not modelled.
    """
    if len(azimuths) != len(elevations) or len(azimuths) < 1 or not np.isfinite([*azimuths, *elevations, pixel_m, noise, registration_sigma_px]).all():
        raise ValueError('finite geometry and equal illumination lengths required')
    if pixel_m <= 0 or noise < 0 or registration_sigma_px < 0 or type(supersample) is not int or supersample < 1:
        raise ValueError('invalid scene scales')
    if any(e <= solar_radius_deg or e >= 90 for e in elevations):
        raise ValueError('Sun must be above the horizontal and below zenith')
    rng = np.random.default_rng(seed); ss = supersample; pad = 5
    padded = (shape[0]+2*pad, shape[1]+2*pad)
    high_shape = (padded[0]*ss, padded[1]*ss)
    rows, cols = np.indices(shape)
    texture = .04*ndi.gaussian_filter(rng.normal(size=shape), 1.2)
    stain = .2*np.exp(-((rows-shape[0]*.3)**2+(cols-shape[1]*.7)**2)/8)
    frames, coverage, shifts = [], [], []
    # Equal-area centre/ring quadrature varies azimuth as well as elevation.
    disc = [(0., 0.)]+[(.75*np.cos(a), .75*np.sin(a)) for a in np.linspace(0, 2*np.pi, 6, endpoint=False)]
    for i, (az, el) in enumerate(zip(azimuths, elevations)):
        cover = np.zeros(high_shape)
        for da, de in disc:
            angle = np.radians(az+da*solar_radius_deg/max(np.cos(np.radians(el)), .01))
            direction = np.array([np.cos(angle), -np.sin(angle)])
            denom = np.tan(np.radians(el+de*solar_radius_deg))+np.dot(slope_rc, direction)
            if denom <= 0:
                raise ValueError('scene receiving plane has no finite shadow intersection')
            cast = np.zeros(high_shape, bool)
            for rock in rocks:
                v = rock['vertices']; origin = (rock['root']+pad)*pixel_m
                for face in rock['faces']:
                    polygon = _above_ground(v[face])
                    if len(polygon) < 3:
                        continue
                    p = np.asarray(polygon)
                    projected = p[:, :2]+origin+p[:, 2, None]/denom*direction
                    for k in range(1, len(projected)-1):
                        _raster_triangle(cast, projected[[0, k, k+1]], pixel_m, ss)
            cover += cast/len(disc)
        cover = ndi.gaussian_filter(cover, (.52*ss, .68*ss), mode='constant')
        cast = cover.reshape(padded[0], ss, padded[1], ss).mean(axis=(1, 3))[pad:pad+shape[0], pad:pad+shape[1]]
        frame = 1+texture-stain+.0003*i*(rows-cols)-.72*cast
        if structured_null:
            frame += .12*np.sin((rows+.7*cols)/3+.8*i)
        shift = rng.normal(0, registration_sigma_px, 2)
        if registration_sigma_px:
            frame = ndi.shift(frame, shift, order=1, mode='nearest')
        white = rng.normal(size=shape); colored = ndi.gaussian_filter(rng.normal(size=shape), .8)
        frame += noise*(.8*white+.6*colored/max(colored.std(), 1e-12))
        frames.append(frame); coverage.append(cast); shifts.append(shift.tolist())
    return dict(stack=np.asarray(frames), coverage=np.asarray(coverage),
                truth=dict(seed=seed, rocks=[r['truth'] for r in rocks], shape=list(shape), pixel_m=pixel_m,
                           azimuths=list(map(float, azimuths)), elevations=list(map(float, elevations)),
                           slope_rc=list(slope_rc), noise_sigma=noise, supersample=ss,
                           solar_radius_deg=solar_radius_deg, registration_shifts_px=shifts,
                           structured_null=structured_null,
                           renderer='projected_clipped_triangles_disc7_v1',
                           limitations='Planar receiving terrain; simplified normalized radiance; no rock-body photometry or scattering. Rescaled shapes are not a measured lunar size population.'))
