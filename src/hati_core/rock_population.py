"""Planted-rock geometry for calibration, drawn from lunar rock shape statistics.

Every planted rock gets its own exposed height, proportions, burial, yaw and body shape,
so calibration covers the shapes real rocks take instead of one fixed body:

- Exposed height is log-uniform between configured bounds. Every part of the range the
  detector is calibrated over gets rocks; this is a calibration design, not the lunar
  size-frequency distribution, which small rocks dominate.
- Horizontal elongation follows laboratory impact fragments, whose axes scatter around
  a : b : c = 2 : sqrt(2) : 1, so width over length centres on 0.71
  (Fujiwara, Kamimoto and Tsukamoto 1978, Nature 272, 602-603).
- Vertical proportion follows measured Moon rocks: height over maximum diameter
  averaged 0.54 over 445 rocks in Lunokhod, Apollo and LROC NAC images, and their axis
  ratios resemble the impact fragments (Demidov and Basilevsky 2014, Solar System
  Research 48, 324-329). The vertical axis is the shortest, as for a rock at rest.
- Burial is small: the same study found the rocks' penetration into the regolith
  negligible.
- Body shape is a NASA Apollo sample mesh (Astromaterials 3D) from the catalog's
  development split, or a seeded procedural convex body. The evaluation split is never
  drawn for calibration.
- Yaw is uniform.

The spreads around those means are assumptions, declared in RockPrior and recorded with
every rock; replace them when a measured polar distribution is available.
"""
from dataclasses import asdict, dataclass

import numpy as np

SOURCES = ('Fujiwara, Kamimoto and Tsukamoto 1978, Nature 272, 602-603: impact fragment axes about 2 : sqrt(2) : 1',
           'Demidov and Basilevsky 2014, Solar System Research 48, 324-329: lunar rock height over maximum diameter '
           '0.54 +/- 0.03 (445 rocks); negligible penetration into the regolith')


@dataclass(frozen=True)
class RockPrior:
    height_m: tuple = (.15, 2.)              # exposed height, log-uniform
    width_over_length: tuple = (.71, .12)    # mean, sd; impact fragments (b/a)
    height_over_diameter: tuple = (.54, .1)  # mean, sd of exposed height / long axis; lunar rocks
    ratio_limits: tuple = (.3, 1.)           # every axis ratio is clipped to this range
    burial: tuple = (0., .15)                # buried share of the body's height, uniform
    nasa_fraction: float = .5                # share of bodies drawn from the NASA development meshes

    def validate(self):
        lo, hi = self.height_m
        if not 0 < lo < hi:
            raise ValueError('planted height range must be positive and increasing')
        for name in ('width_over_length', 'height_over_diameter'):
            mean, sd = getattr(self, name)
            if not (0 < mean <= 1 and sd >= 0):
                raise ValueError(f'{name} needs a mean in (0, 1] and a non-negative spread')
        if not 0 < self.ratio_limits[0] < self.ratio_limits[1] <= 1:
            raise ValueError('ratio limits must lie in (0, 1]')
        if not 0 <= self.burial[0] <= self.burial[1] < 1:
            raise ValueError('burial must lie in [0, 1)')
        if not 0 <= self.nasa_fraction <= 1:
            raise ValueError('nasa_fraction must lie in [0, 1]')
        return self


def prior_from_config(cfg):
    """The configured prior; keys mirror RockPrior's fields under 'planted_rock_prior'."""
    values = dict(cfg.get('planted_rock_prior') or {})
    unknown = set(values) - set(RockPrior.__dataclass_fields__)
    if unknown:
        raise ValueError('unknown planted_rock_prior keys: '+', '.join(sorted(unknown)))
    return RockPrior(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}).validate()


def sample_population(count, seed, prior=None, meshes=()):
    """count planted-rock specifications; the same seed gives the same rocks.

    Each specification holds the exposed height, body width and length (pre-yaw), burial,
    yaw, a seed for a procedural body, and the index of the NASA mesh when one is used.
    """
    prior = (prior or RockPrior()).validate()
    rng = np.random.default_rng(seed)
    lo, hi = np.log(prior.height_m[0]), np.log(prior.height_m[1])
    low, high = prior.ratio_limits
    rocks = []
    for i in range(int(count)):
        height = float(np.exp(rng.uniform(lo, hi)))
        b_over_a = float(np.clip(rng.normal(*prior.width_over_length), low, high))
        h_over_d = float(np.clip(rng.normal(*prior.height_over_diameter), low, high))
        burial = float(rng.uniform(*prior.burial))
        # The vertical axis stays the shortest: the whole body, buried part included, is no taller than wide.
        h_over_d = min(h_over_d, b_over_a*(1-burial))
        length = height/h_over_d
        width = length*b_over_a
        use_mesh = bool(len(meshes)) and rng.random() < prior.nasa_fraction
        mesh_index = int(rng.integers(len(meshes))) if use_mesh else None
        rocks.append(dict(index=i, seed=int(rng.integers(2**31-1)), height_m=height, width_m=width, length_m=length,
                          aspect=length/width, burial=burial, yaw_deg=float(rng.uniform(0, 360)),
                          height_over_diameter=height/length, width_over_length=b_over_a,
                          mesh_index=mesh_index,
                          shape=(meshes[mesh_index]['provenance']['id'] if use_mesh else 'procedural')))
    return rocks


def at_height(spec, height_m):
    """The same body scaled to another exposed height; its proportions, burial, yaw and shape are kept."""
    if not height_m > 0:
        raise ValueError('height must be positive')
    scale = height_m/spec['height_m']
    return dict(spec, height_m=float(height_m), width_m=spec['width_m']*scale, length_m=spec['length_m']*scale)


def build_rock(spec, root, meshes=()):
    """The make_rock body for one specification, standing at root (row, col in pixels)."""
    from .rock_scenes import make_rock
    mesh = meshes[spec['mesh_index']] if spec.get('mesh_index') is not None else None
    return make_rock(spec['seed'], root, spec['height_m'], spec['width_m'], aspect=spec['aspect'],
                     burial=spec['burial'], mesh=mesh, yaw_deg=spec['yaw_deg'])


def describe(prior, meshes=()):
    """Provenance for results: the prior, its sources and the meshes it may draw."""
    return dict(prior=asdict(prior), sources=list(SOURCES),
                meshes=[m['provenance']['id'] for m in meshes],
                note='Heights are log-uniform for calibration coverage, not a lunar size distribution. '
                     'Spreads around the sourced means are assumptions.')
