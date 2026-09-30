"""Sweep morphology classifier (HATI 2.6): geometry, fitting, decisions and a small end-to-end check."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]
from src.hati_core.relief_scenes import lit_fraction, relief_feature, render_relief  # noqa: E402
from src.hati_core.rock_scenes import make_rock  # noqa: E402
from src.hati_core.shadow_likelihood import ShadowConfig  # noqa: E402
from src.hati_core.sweep_classifier import (COMPACT, SweepClassifierConfig, _bowl_shadow, _conformal_quantile,  # noqa: E402
                                            _dome_shadow, _toward_sun, build_banks, calibrate_margins,
                                            clopper_pearson, decide, evaluate_cell, fit_two, relief_components)

AZ = np.array([43.1, 43.5, 69., 73.7, 326.7, 354.6, 6., 24.9])
EL = np.array([4.35, 3.59, 3.47, 3.38, 3.58, 3.63, 3.28, 3.69])
SMALL = SweepClassifierConfig(radius_px=12, support_px=9., rock_heights_m=(.3, .6, 1.2), rock_widths_m=(.6,),
                              rock_offsets_px=(-.5, .5), relief_offsets_px=(0.,), crater_diameters_m=(3.6, 7.2),
                              crater_depth_ratios=(.02, .1), hummock_diameters_m=(3.6, 7.2), hummock_height_ratios=(.02, .1))


def paraboloid(kind, radius, relief, spacing, n):
    x = (np.arange(n)-(n-1)/2)*spacing
    r2 = x[:, None]**2+x[None, :]**2
    inside = r2 < radius**2
    shape = relief*(r2/radius**2-1) if kind == 'bowl' else relief*(1-r2/radius**2)
    return np.where(inside, shape, 0.), x


class GeometryTests(unittest.TestCase):
    def test_closed_form_shadows_match_the_numerical_horizon(self):
        spacing, n = .05, 300
        for kind, radius, relief in (('bowl', 3., .6), ('bowl', 3., .15), ('dome', 3., .3)):
            h, x = paraboloid(kind, radius, relief, spacing, n)
            rows, cols = np.meshgrid(x, x, indexing='ij')
            for az, el in ((30., 3.5), (300., 5.)):
                s = _toward_sun(az)
                u = rows*s[0]+cols*s[1]; v2 = np.maximum(rows**2+cols**2-u*u, 0)
                te = np.tan(np.radians(el))
                analytic = _bowl_shadow(u, v2, radius, relief, te) if kind == 'bowl' else _dome_shadow(u, v2, radius, relief, te)
                numeric = lit_fraction(h, spacing, az, el, solar_radius_deg=1e-6) < .5
                self.assertLess(abs(analytic.mean()-numeric.mean()), .002, (kind, radius, relief, az, el))

    def test_gentle_features_cast_no_shadow_and_steep_ones_do(self):
        u = np.linspace(-6, 6, 241)[:, None]*np.ones((1, 3)); v2 = np.zeros_like(u)
        te = np.tan(np.radians(3.5))
        self.assertFalse(_dome_shadow(u, v2, 2., .05, te).any())       # flanks 2.9 deg
        self.assertTrue(_dome_shadow(u, v2, 2., .3, te).any())         # flanks 16.7 deg
        self.assertFalse(_bowl_shadow(u, v2, 2., .05, te).any())
        self.assertTrue(_bowl_shadow(u, v2, 2., .3, te).any())

    def test_crater_is_dark_toward_the_sun_and_a_hummock_away_from_it(self):
        # Sun in the east: columns increase toward it. At 6 deg the shadow of a 7.2 m, 0.72 m deep bowl
        # reaches 1.7 m past its centre (closed form), so the Sun-side wall is dark and the far wall lit.
        shape, centre = (33, 33), (16., 16.)
        shade, cover = relief_components(shape, centre, 7.2, .1, 'crater', [90.], [6.], .9, .6)
        self.assertGreater(cover[0, 16, 19], .9)                 # 2.7 m toward the Sun, inside the rim
        self.assertLess(cover[0, 16, 13], .1)                    # 2.7 m away from the Sun, inside the rim
        self.assertGreater(shade[0, 16, 13], .5)                 # that wall faces the Sun
        shade, cover = relief_components(shape, centre, 7.2, .1, 'hummock', [90.], [3.5], .9, .6)
        cols = np.indices(shape)[1]-centre[1]
        self.assertLess((cover[0]*cols).sum()/cover[0].sum(), -2.)   # shadow on the far side and beyond
        self.assertGreater(shade[0, 16, 18], .5)                     # Sun-facing flank bright
        self.assertLess(cover[0, 16, 18], .1)


class FittingTests(unittest.TestCase):
    def test_box_constrained_pairs_match_a_general_solver(self):
        from scipy.optimize import lsq_linear
        rng = np.random.default_rng(3)
        x1, x2 = rng.normal(size=(40, 30)), rng.normal(size=(40, 30))
        y = rng.normal(size=30)
        g, c, improvement = fit_two(x1, x2, y, 2., 1.)
        for j in range(40):
            a = np.column_stack([x1[j], -x2[j]])
            reference = lsq_linear(a, y, bounds=([0, 0], [2, 1])).x
            ours = np.array([g[j], c[j]])
            self.assertLess(np.sum((y-a@ours)**2), np.sum((y-a@reference)**2)+1e-9)
            self.assertAlmostEqual(improvement[j], y@y-np.sum((y-a@ours)**2), places=8)

    def test_conformal_quantile_and_binomial_interval(self):
        values = np.arange(1, 20, dtype=float)                   # n = 19
        self.assertEqual(_conformal_quantile(values, .1), 18.)   # rank ceil(20*0.9) = 18
        self.assertEqual(_conformal_quantile(values[:5], .1), float('inf'))
        low, high = clopper_pearson(0, 10)
        self.assertEqual(low, 0.); self.assertAlmostEqual(high, 1-.025**(1/10), places=6)
        low, high = clopper_pearson(10, 10)
        self.assertAlmostEqual(low, .025**(1/10), places=6); self.assertEqual(high, 1.)


def row(gains, sun=None):
    sun = sun or {c: 5. for c in gains}
    return dict(status='assessed', gains=gains, sun_margins=sun)


class DecisionTests(unittest.TestCase):
    margins = dict(none=.5, sun=.3, compactness=.7, pair=dict(boulder=.2, hummock=.2, crater=.2, extended=.2))

    def test_labels_follow_the_declared_rules(self):
        m = self.margins
        self.assertEqual(decide(row(dict(boulder=.1, hummock=.2, crater=0., extended=.4)), m)['label'], 'no_signal')
        self.assertEqual(decide(row(dict(boulder=3., hummock=1., crater=0., extended=-1.)), m)['label'], 'boulder')
        self.assertEqual(decide(row(dict(boulder=0., hummock=0., crater=4., extended=4.5)), m)['label'], 'crater')
        self.assertEqual(decide(row(dict(boulder=0., hummock=.3, crater=.2, extended=3.)), m)['label'], 'extended')
        self.assertEqual(decide(row(dict(boulder=2., hummock=1.9, crater=0., extended=1.)), m)['label'], 'ambiguous')
        stripes = row(dict(boulder=.6, hummock=.9, crater=.1, extended=-.2), sun=dict(boulder=.05, hummock=.06, crater=0., extended=0.))
        self.assertEqual(decide(stripes, m)['label'], 'non_solar_change')

    def test_a_weak_change_that_fails_the_sun_check_is_ambiguous_not_non_solar(self):
        m = dict(self.margins, sun_ratio=.3)
        weak_rock = row(dict(boulder=.55, hummock=.27, crater=.02, extended=-.9),
                        sun=dict(boulder=.28, hummock=.18, crater=0., extended=.2))      # half the gain follows the Sun
        self.assertEqual(decide(weak_rock, m)['label'], 'ambiguous')
        stripes = row(dict(boulder=.6, hummock=.9, crater=.1, extended=-.2), sun=dict(boulder=.05, hummock=.06, crater=0., extended=0.))
        self.assertEqual(decide(stripes, m)['label'], 'non_solar_change')

    def test_calibration_covers_the_declared_errors(self):
        rng = np.random.default_rng(1)
        rows = []
        for truth, gains in (('rock', dict(boulder=3., hummock=1., crater=0., extended=-1.)),
                             ('mound', dict(boulder=1., hummock=4., crater=0., extended=4.5)),
                             ('bowl', dict(boulder=0., hummock=0., crater=4., extended=4.4)),
                             ('ripples', dict(boulder=0., hummock=.2, crater=.1, extended=3.))):
            for _ in range(30):
                noisy = {k: v+rng.normal(0, .3) for k, v in gains.items()}
                rows.append(dict(row(noisy), truth=truth))
        for truth in ('none', 'stripes'):
            for _ in range(30):
                noisy = {k: rng.normal(0, .2) for k in ('boulder', 'hummock', 'crater', 'extended')}
                rows.append(dict(row(noisy, sun={k: rng.normal(0, .1) for k in noisy}), truth=truth))
        margins = calibrate_margins(rows, dict(none=.1, sun=.1, compactness=.1, pair=dict(boulder=.1, hummock=.1, crater=.1, extended=.1)))
        wrong = sum(decide(r, margins)['label'] not in ('ambiguous', 'no_signal') for r in rows if r['truth'] == 'none')
        self.assertLessEqual(wrong, 6)
        self.assertLess(margins['compactness'], 1.)
        self.assertTrue(0. <= margins['sun_ratio'] <= 1.)
        for c in COMPACT:
            self.assertGreaterEqual(margins['pair'][c], 0.)


class EndToEndTests(unittest.TestCase):
    """A small bank on clean, strong scenes: each physical kind must lead with its own hypothesis."""

    @classmethod
    def setUpClass(cls):
        cls.sc = replace(ShadowConfig(pixel_m=.9, registration_sigma_px=.5), radius_px=SMALL.radius_px,
                         root_support_px=SMALL.support_px)
        shape = (2*SMALL.radius_px+1,)*2
        cls.shifts = [1, 4, 7]
        cls.banks = build_banks(shape, AZ, EL, cls.sc, SMALL)
        cls.shuffled = {k: build_banks(shape, np.roll(AZ, k), np.roll(EL, k), cls.sc, SMALL) for k in cls.shifts}

    def evaluate(self, **scene):
        size = 2*SMALL.radius_px+1
        out = render_relief((size, size), AZ, EL, pixel_m=.9, seed=21, noise=.02, supersample=4, **scene)
        return evaluate_cell(out['stack'], np.ones_like(out['stack'], bool), AZ, EL, .02, self.sc, SMALL,
                             banks=self.banks, shuffled_banks=self.shuffled, shifts=self.shifts)

    def test_each_kind_leads_with_its_own_compact_hypothesis(self):
        centre = (SMALL.radius_px+.3, SMALL.radius_px+.2)
        rock = self.evaluate(rocks=[make_rock(4, centre, .6, .6, aspect=1.35)])['gains']
        self.assertEqual(max(COMPACT, key=rock.get), 'boulder')
        self.assertLess(rock['crater'], .2*rock['boulder'])
        bowl = self.evaluate(features=[relief_feature('bowl', 6., 5., seed=3)])['gains']
        self.assertEqual(max(COMPACT, key=bowl.get), 'crater')
        self.assertLess(max(bowl['boulder'], bowl['hummock']), .2*bowl['crater'])
        mound = self.evaluate(features=[relief_feature('mound', 6., 5., seed=3)])['gains']
        self.assertEqual(max(('hummock', 'crater'), key=mound.get), 'hummock')

    def test_blank_ground_gains_nothing_and_stripes_fail_the_sun_check(self):
        blank = self.evaluate()
        self.assertLess(max(blank['gains'].values()), .1)
        stripes = self.evaluate(structured_null=True)
        best = max(stripes['gains'], key=stripes['gains'].get)
        self.assertLess(stripes['sun_margins'][best], .25*max(stripes['gains'][best], 1e-9)+.2)

    def test_banks_for_other_frames_are_refused(self):
        # A frame that loses its support must not leave the remaining frames paired with the wrong Sun.
        size = 2*SMALL.radius_px+1
        out = render_relief((size, size), AZ, EL, pixel_m=.9, seed=5, noise=.02, supersample=4)
        valid = np.ones_like(out['stack'], bool)
        valid[3, :, :SMALL.radius_px] = False                  # half the disc missing in frame 3
        from src.hati_core.sweep_classifier import usable_frames
        frames = usable_frames(out['stack'], valid, SMALL)
        self.assertNotIn(3, frames.tolist())
        with self.assertRaises(ValueError):
            evaluate_cell(out['stack'], valid, AZ, EL, .02, self.sc, SMALL, banks=self.banks,
                          shuffled_banks=self.shuffled, shifts=self.shifts)
        row = evaluate_cell(out['stack'][frames], valid[frames], AZ[frames], EL[frames], .02, self.sc, SMALL, shifts=[1, 3, 5])
        self.assertEqual(row['status'], 'assessed')

    def test_missing_pixels_outside_the_fitting_disc_are_harmless(self):
        size = 2*SMALL.radius_px+1
        out = render_relief((size, size), AZ, EL, pixel_m=.9, seed=5, noise=.02, supersample=4)
        stack = out['stack'].copy()
        stack[:, :3, :3] = np.nan                                  # a corner outside the disc, as at a mask edge
        valid = np.isfinite(stack)
        row = evaluate_cell(stack, valid, AZ, EL, .02, self.sc, SMALL, banks=self.banks,
                            shuffled_banks=self.shuffled, shifts=self.shifts)
        self.assertEqual(row['status'], 'assessed')
        clean = evaluate_cell(out['stack'], np.ones_like(stack, bool), AZ, EL, .02, self.sc, SMALL, banks=self.banks,
                              shuffled_banks=self.shuffled, shifts=self.shifts)
        for name in row['gains']:
            self.assertAlmostEqual(row['gains'][name], clean['gains'][name], places=6)

    def test_withheld_intensities_never_choose_the_model_that_predicts_them(self):
        # For fold k, the chosen template and amplitudes must not depend on frame k's intensities.
        size = 2*SMALL.radius_px+1
        centre = (SMALL.radius_px+.3, SMALL.radius_px+.2)
        out = render_relief((size, size), AZ, EL, pixel_m=.9, seed=5, noise=.02, supersample=4,
                            rocks=[make_rock(6, centre, .6, .6, aspect=1.35)])
        valid = np.ones_like(out['stack'], bool)
        altered = out['stack'].copy()
        altered[2] = altered[2]+np.random.default_rng(9).normal(0, .5, altered[2].shape)
        a = evaluate_cell(out['stack'], valid, AZ, EL, .02, self.sc, SMALL, banks=self.banks,
                          shuffled_banks=self.shuffled, shifts=self.shifts, record_folds=True)
        b = evaluate_cell(altered, valid, AZ, EL, .02, self.sc, SMALL, banks=self.banks,
                          shuffled_banks=self.shuffled, shifts=self.shifts, record_folds=True)
        fold_a = next(f for f in a['folds'] if f['withheld'] == 2)
        fold_b = next(f for f in b['folds'] if f['withheld'] == 2)
        for name in COMPACT:
            self.assertEqual(fold_a['chosen'][name][0], fold_b['chosen'][name][0], name)
            np.testing.assert_allclose(fold_a['chosen'][name][1:], fold_b['chosen'][name][1:], rtol=1e-10, atol=1e-12)
        self.assertNotEqual(a['frame_errors']['null'][2], b['frame_errors']['null'][2])   # the prediction target did change

class HazardQuantityTests(unittest.TestCase):
    def test_hazard_tests_use_the_lower_end_of_the_full_compatible_range(self):
        sys.path[:0] = [str(ROOT/'scripts')]
        from classifier_stage import _hazard
        row = dict(best_parameters=dict(boulder=dict(height_m=.6, width_m=.6),
                                        crater=dict(diameter_m=5.4, relief_m=.54, max_slope_deg=21.8)),
                   compatible_ranges=dict(boulder=dict(height_m=[.2, .6], width_m=[.3, .6]),
                                          crater=dict(diameter_m=[3.6, 5.4], relief_m=[.18, .54], max_slope_deg=[7.6, 21.8])))
        boulder = _hazard('boulder', row, .3, 8.)
        self.assertEqual(boulder['height_range_m'], [.2, .6])
        self.assertFalse(boulder['exceeds_clearance'])          # 0.2 m is compatible, so 0.3 m is not established
        crater = _hazard('crater', row, .3, 8.)
        self.assertEqual(crater['depth_range_m'], [.18, .54])
        self.assertFalse(crater['exceeds_slope_limit'])         # 7.6 deg is compatible with the data
        self.assertEqual(_hazard('extended', row, .3, 8.), {})


class CampaignStageTests(unittest.TestCase):
    def test_t18_on_a_relief_bundle_without_external_calls(self):
        import json
        import tempfile
        from argparse import Namespace
        from dataclasses import asdict
        from unittest.mock import patch
        sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'tests')]
        from saturation_experiments import Experiment, t1, t18
        from test_relief import relief_bundle
        small = asdict(SMALL)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); bundle = out/'input.zip'; relief_bundle(bundle)
            cfg = json.loads((ROOT/'configs/saturation_campaign.json').read_text())
            cfg.update(synthetic_seeds=1, synthetic_locations=[[.5, .5]], classifier=small, classifier_seeds=1,
                       classifier_blanks=4, classifier_planted_sites=1, classifier_athena_cells=3,
                       classifier_planted_kinds=['rock 0.6', 'bowl 6'], relief_touchdown_radius_px=4)
            config = out/'config.json'; config.write_text(json.dumps(cfg))
            def run(stage, fn):
                args = Namespace(stage=stage, bundle=bundle, config=config, output=out/'stages'/stage,
                                 campaign=out, dem=None, thermal=None, held_out=None, rock_catalog=None)
                with patch('subprocess.run', side_effect=AssertionError('external ingestion forbidden')):
                    return fn(Experiment(args))
            run('T1', t1)
            result = run('T18', t18)
            self.assertEqual(result['status'], 'PARTIAL')
            self.assertEqual(set(result['margins']['pair']), {'boulder', 'hummock', 'crater', 'extended'})
            self.assertIn('rock', result['generated_test']['per_truth'])
            self.assertGreater(result['athena_cells'], 0)
            self.assertEqual(sum(result['athena_counts'].values()),
                             sum(1 for r in json.loads((out/'stages/T18/athena_cells.json').read_text()) if r.get('label')))
            planted = json.loads((out/'stages/T18/planted.json').read_text())
            self.assertTrue(all(p['background_label'] is not None for p in planted['summary']))
            self.assertTrue((out/'stages/T18/classifier.png').exists())


if __name__ == '__main__':
    unittest.main()
