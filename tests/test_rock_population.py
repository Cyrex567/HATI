"""Planted-rock population: sourced proportions, reproducible draws, NASA meshes from the development split only."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.hati_core.rock_population import RockPrior, build_rock, prior_from_config, sample_population
from src.hati_core.rock_scenes import load_catalog

CATALOG = ROOT/'data/rock_shapes/apollo_proxy_v2/catalog.json'


class PopulationTests(unittest.TestCase):
    def test_same_seed_same_rocks(self):
        self.assertEqual(sample_population(20, 7), sample_population(20, 7))
        self.assertNotEqual(sample_population(20, 7), sample_population(20, 8))

    def test_heights_cover_the_range_evenly_in_log(self):
        rocks = sample_population(4000, 1, RockPrior(height_m=(.2, 1.6)))
        h = np.array([r['height_m'] for r in rocks])
        self.assertTrue(((h >= .2) & (h <= 1.6)).all())
        # Log-uniform: the median sits at the geometric mean, and each factor of two holds a third of the rocks.
        self.assertAlmostEqual(np.median(h), np.sqrt(.2*1.6), delta=.03)
        for lo in (.2, .4, .8):
            self.assertAlmostEqual(np.mean((h >= lo) & (h < 2*lo)), 1/3, delta=.03)

    def test_proportions_follow_the_sourced_means(self):
        rocks = sample_population(4000, 2)
        hd = np.array([r['height_over_diameter'] for r in rocks])
        wl = np.array([r['width_over_length'] for r in rocks])
        # Lunar rocks: height over maximum diameter 0.54 (Demidov and Basilevsky 2014); the cap that keeps the
        # vertical axis shortest pulls the mean a little lower. Impact fragments: width over length 0.71.
        self.assertTrue(.48 < hd.mean() < .56)
        self.assertAlmostEqual(wl.mean(), .71, delta=.02)
        for r in rocks:
            self.assertAlmostEqual(r['height_m']/r['length_m'], r['height_over_diameter'], places=9)
            self.assertLessEqual(r['height_m']/(1-r['burial']), r['width_m']+1e-9)   # vertical axis shortest
            self.assertLessEqual(r['width_m'], r['length_m']+1e-9)
            self.assertTrue(0 <= r['burial'] <= .15 and 0 <= r['yaw_deg'] < 360)

    def test_bodies_match_their_specifications(self):
        for spec in sample_population(6, 3):
            truth = build_rock(spec, (10., 10.))['truth']
            self.assertAlmostEqual(truth['exposed_height_m'], spec['height_m'])
            self.assertAlmostEqual(truth['body_width_m'], spec['width_m'])
            self.assertAlmostEqual(truth['body_length_m'], spec['length_m'])
            self.assertAlmostEqual(truth['yaw_deg'], spec['yaw_deg'])
            self.assertEqual(truth['source']['source'], 'procedural_convex_rock')

    def test_nasa_meshes_come_from_the_development_split(self):
        meshes = load_catalog(CATALOG, split='development')
        self.assertEqual({m['provenance']['split'] for m in meshes}, {'development'})
        self.assertEqual(len(meshes), 15)
        evaluation = {m['provenance']['parent_rock'] for m in load_catalog(CATALOG, split='evaluation')}
        self.assertIn('10021', evaluation)
        self.assertFalse(evaluation & {m['provenance']['parent_rock'] for m in meshes})   # no rock in both
        rocks = sample_population(200, 4, RockPrior(nasa_fraction=.5), meshes)
        share = np.mean([r['mesh_index'] is not None for r in rocks])
        self.assertTrue(.35 < share < .65)
        nasa = next(r for r in rocks if r['mesh_index'] is not None)
        self.assertGreater(len({r['mesh_index'] for r in rocks if r['mesh_index'] is not None}), 8)   # many bodies drawn
        self.assertEqual(nasa['shape'], meshes[nasa['mesh_index']]['provenance']['id'])
        self.assertEqual(build_rock(nasa, (5., 5.), meshes)['truth']['source']['id'], nasa['shape'])
        self.assertTrue(all(r['mesh_index'] is None for r in sample_population(50, 4, RockPrior(nasa_fraction=0.), meshes)))
        self.assertTrue(all(r['shape'] == 'procedural' for r in sample_population(50, 4)))

    def test_a_body_scaled_to_a_set_height_keeps_its_proportions(self):
        from src.hati_core.rock_population import at_height
        spec = sample_population(1, 5)[0]
        tall = at_height(spec, 1.2)
        self.assertAlmostEqual(tall['height_m'], 1.2)
        self.assertAlmostEqual(tall['height_m']/tall['length_m'], spec['height_over_diameter'])
        self.assertAlmostEqual(tall['width_m']/tall['length_m'], spec['width_m']/spec['length_m'])
        self.assertEqual({k: tall[k] for k in ('aspect', 'burial', 'yaw_deg', 'shape', 'seed')},
                         {k: spec[k] for k in ('aspect', 'burial', 'yaw_deg', 'shape', 'seed')})
        with self.assertRaises(ValueError):
            at_height(spec, 0.)

    def test_configuration_is_checked(self):
        self.assertEqual(prior_from_config({}), RockPrior())
        self.assertEqual(prior_from_config(dict(planted_rock_prior=dict(height_m=[.3, 1.]))).height_m, (.3, 1.))
        for bad in (dict(height_m=[1., .3]), dict(burial=[0., 1.2]), dict(nasa_fraction=2.), dict(width_over_length=[1.5, .1])):
            with self.assertRaises(ValueError):
                prior_from_config(dict(planted_rock_prior=bad))
        with self.assertRaises(ValueError):
            prior_from_config(dict(planted_rock_prior=dict(heights=[.2, 1.])))


if __name__ == '__main__':
    unittest.main()
