"""Independent mesh geometry, deterministic seeds and provenance checks."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.hati_core.rock_scenes import read_obj, make_rock, render_rocks, load_catalog, _above_ground


OBJ = 'v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n'


class RockTests(unittest.TestCase):
    def test_repeatable_truth_pixels_and_independence_from_detector_renderer(self):
        rock = make_rock(44, (20.3, 20.2), .6, .8, aspect=1.4)
        args = ((41, 41), [0, 90, 180, 270], [5]*4, [rock])
        with patch('src.hati_core.shadow_likelihood.shadow_template', side_effect=AssertionError('renderer coupling')):
            a = render_rocks(*args, pixel_m=.9, seed=3, supersample=2)
            b = render_rocks(*args, pixel_m=.9, seed=3, supersample=2)
        np.testing.assert_array_equal(a['stack'], b['stack'])
        self.assertEqual(a['truth'], b['truth'])
        self.assertAlmostEqual(rock['vertices'][:, 2].max(), .6)
        self.assertLess(rock['vertices'][:, 2].min(), 0)
        self.assertGreater(a['coverage'][0].sum(), 0)

    def test_shadow_moves_down_sun_and_lowers_with_increased_elevation(self):
        rock = make_rock(23, (24, 24), .8, .8)
        out = render_rocks((49, 49), [0, 180, 0], [4, 4, 12], [rock], pixel_m=.9,
                           seed=2, noise=0, supersample=3, solar_radius_deg=0)
        r, c = np.indices((49, 49)); cover = out['coverage']
        centroids = [(r*t).sum()/t.sum() for t in cover]
        self.assertGreater(centroids[0], 24)
        self.assertLess(centroids[1], 24)
        self.assertLess(centroids[2], centroids[0])

    def test_buried_triangle_is_clipped_before_projection(self):
        polygon = _above_ground(np.array([[0, 0, -1.], [1, 0, 1.], [0, 1, 1.]]))
        self.assertEqual(len(polygon), 4)
        self.assertTrue(all(p[2] >= 0 for p in polygon))

    def test_catalog_hash_and_parent_split_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); obj = root/'rock.obj'; obj.write_text(OBJ)
            item = dict(id='one', parent_rock='10000', split='evaluation', path='rock.obj',
                        sha256=hashlib.sha256(obj.read_bytes()).hexdigest(), source_url='https://example.org/source',
                        credit='fixture', geometry_assumptions='fixture tetrahedron')
            manifest = root/'catalog.json'
            manifest.write_text(json.dumps(dict(schema_version=1, meshes=[item])))
            self.assertEqual(len(load_catalog(manifest)), 1)
            manifest.write_text(json.dumps(dict(schema_version=1, meshes=[item, dict(item, id='two', split='development')])))
            with self.assertRaisesRegex(ValueError, 'both splits'):
                load_catalog(manifest)
            manifest.write_text(json.dumps(dict(schema_version=1, meshes=[item])))
            obj.write_text(OBJ+'# altered\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_catalog(manifest)

    def test_negative_obj_indices_and_invalid_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'r.obj'; p.write_text(OBJ.replace('f 2 3 4', 'f -3 -2 -1'))
            v, f = read_obj(p)
            self.assertEqual(f[-1].tolist(), [1, 2, 3])
            p.write_text(OBJ.replace('v 0 0 1', 'v 0 0 nan'))
            with self.assertRaises(ValueError):
                read_obj(p)


if __name__ == '__main__':
    unittest.main()
