"""Offline audit regressions. No network, downloads, or ISIS subprocesses.

Run: python tests/test_audit_regressions.py
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.ndimage import gaussian_filter, shift
from skimage.registration import phase_cross_correlation as pcc

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import ingest_sweep as ingest
import shadow_kinematics_real as kin
from sweep_contract import predates
from src.hati_core import Config, channels, gate


class AuditRegressions(unittest.TestCase):
    def test_processing_chain_and_cache_invalidation_without_running_isis(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            img = work / "M1LE.IMG"
            img.write_bytes(b"test only")
            mapfile = work / "test.map"
            mapfile.write_text("map version 1")
            frame = {"pid": "nac.m1le", "img": img, "url": "https://example.org/M1LE.IMG",
                     "utc": "2024-01-01T00:00:00Z"}
            calls = []
            fail_projection = [False]
            def fake_isis(cmd):
                calls.append(cmd)
                for value in cmd:
                    if value.startswith("to="):
                        Path(value[3:]).write_text("simulated output")
                return (False, "simulated interrupted projection") if (
                    fail_projection[0] and cmd[0] == "cam2map") else (True, "")
            with patch.object(ingest, "isis", side_effect=fake_isis), \
                 patch.object(ingest, "isis_retry", side_effect=fake_isis), \
                 patch.object(ingest, "campt_covers", return_value="inside"), \
                 patch.object(ingest, "lev2_has_site", return_value=True):
                self.assertTrue(ingest.process_frame(frame, work, mapfile))
                programs = [c[0] for c in calls]
                self.assertLess(programs.index("lronaccal"), programs.index("lronacecho"))
                self.assertLess(programs.index("lronacecho"), programs.index("cam2map"))
                self.assertIn("from=" + str(work / "M1LE.echo.cub"), calls[-1])
                count = len(calls)
                self.assertTrue(ingest.process_frame(frame, work, mapfile))
                self.assertEqual(len(calls), count)
                mapfile.write_text("map version 2")
                self.assertTrue(ingest.process_frame(frame, work, mapfile))
                self.assertGreater(len(calls), count)
                fail_projection[0] = True
                self.assertFalse(ingest.process_frame(frame, work, mapfile, rebuild=True))
                self.assertFalse((work / "M1LE.processing.json").exists())

    def test_map_rotation_matches_projected_north_and_east(self):
        from rasterio.warp import transform
        lat, lon, eps = -84.7906, 29.1957, 1e-5
        x, y = transform("+proj=longlat +R=1737400 +no_defs", kin.MAP_CRS,
                         [lon, lon, lon + eps], [lat, lat + eps, lat])
        north = np.array([x[1] - x[0], y[1] - y[0]])
        east = np.array([x[2] - x[0], y[2] - y[0]])
        north /= np.linalg.norm(north)
        east /= np.linalg.norm(east)
        for a in (0, 45, 90, 180, 270):
            direction = north * np.cos(np.radians(a)) + east * np.sin(np.radians(a))
            expected = [np.sin(np.radians(a + lon)), np.cos(np.radians(a + lon))]
            np.testing.assert_allclose(direction, expected, atol=1e-6)

    def test_cutoff_is_exclusive_and_missing_dates_fail(self):
        self.assertTrue(predates("2025-03-05T23:59:59Z"))
        for value in ("2025-03-06T00:00:00Z", "2025-03-09T02:15:41Z", "", None,
                      "2025-03-05T23:59:59"):
            self.assertFalse(predates(value))

    def test_post_landing_frames_never_enter_sibling_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.csv"
            path.write_text("product,utc,sun_elev_deg,sun_az_deg,download_url,emission_deg,margin_m\n"
                            "nac.pre,2024-01-01T00:00:00Z,4,20,pre.img,3,1000\n"
                            "nac.post,2025-03-09T00:00:00Z,4,20,post.img,3,1000\n")
            rows = ingest.load_csv(catalog=path)
            self.assertEqual([r["pid"] for r in rows], ["nac.pre"])
            self.assertNotIn("POST", ingest.ALL_BY_PID)

    def test_empty_selection_fails_with_a_message(self):
        with self.assertRaises(SystemExit):
            ingest.select_frames([], 24, 1, 9, 4)

    def test_bounded_registration_applies_correct_coarse_sign(self):
        ref = gaussian_filter(np.random.default_rng(6).normal(size=(256, 256)), 2)
        mov = shift(ref, (8, -5), order=1, mode="nearest")
        applied = []
        def tracked(a, d, **kw):
            applied.append(d)
            return shift(a, d, **kw)
        result, _, edge = ingest.bounded_shift(ref, mov, 20, pcc, tracked, np)
        np.testing.assert_allclose(applied[0], (-8, 5))
        np.testing.assert_allclose(result, (-8, 5), atol=.2)
        self.assertFalse(edge)

    def test_bounded_registration_cannot_escape_via_refinement(self):
        ref = gaussian_filter(np.random.default_rng(0).normal(size=(256, 256)), 2)
        with self.assertRaises(ValueError):
            ingest.bounded_shift(ref, shift(ref, (60, 0), order=1, mode="nearest"),
                                 20, pcc, shift, np)

    def test_arc_residual_cannot_cancel_opposite_axes(self):
        votes = [(i, 100 + (-1)**i * 4, 100 - (-1)**i * 4) for i in range(4)]
        result = kin.classify(votes, [0, 30, 60, 90])
        self.assertAlmostEqual(result["rms_arc"], math.sqrt(32), places=8)

    def test_stationary_stain_is_a_known_counterexample_not_a_boulder(self):
        dn = np.full((160, 160), 120.)
        dn[70:79, 80] = 20
        frames = [{"mask": kin.detect(dn, .5, 45, 4), "az_map": float(a), "elev": 4}
                  for a in np.linspace(-44, 44, 8)]
        _, ev, _, _, casts = kin.accumulate(frames, dn.shape, 4, 1.8, 3, max_width_px=8)
        self.assertEqual(casts, [0, 0, 1, 1, 1, 1, 0, 0])
        self.assertEqual(int((ev >= 4).sum()), 29)
        # Document a limitation: do not change this assertion to claim albedo rejection.

    def test_shifted_null_preserves_edge_vote_mass(self):
        frames = [{"mask": np.zeros((40, 40), bool), "az_map": 0, "elev": 4,
                   "regions": [(np.arange(1, 10, dtype=float), np.ones(9))]}]
        a = kin.accumulate(frames, (40, 40), 4, 1.8, 3)[1]
        b = kin.accumulate(frames, (40, 40), 4, 1.8, 3, jitter=[(-10, -10)])[1]
        self.assertEqual(float(a.sum()), float(b.sum()))

    def test_injection_integrates_width_and_orientation(self):
        dn = np.full((160, 160), 120.)
        masses = []
        for az in (0, 30, 60, 90):
            image = kin.inject_shadows(dn, [(80.3, 80.2, 1.0)], az, 4, 45)
            masses.append(float(((dn - image) / 90).sum()))
        expected = (2 / kin.RES) * (1 / math.tan(math.radians(4)) / kin.RES)
        np.testing.assert_allclose(masses, expected, rtol=.06)

    def test_recovery_passes_the_real_detector_width(self):
        frames = [{"dn": np.full((80, 80), 120.), "az_map": 0, "elev": 4}]
        args = SimpleNamespace(bg_win=45, shadow_frac=.5, min_area=4,
                               elongation=1.8, vote_radius=3, max_width_px=11)
        fake = (None, np.zeros((80, 80)), None, None, None)
        with patch.object(kin, "accumulate", return_value=fake) as acc:
            kin.recovery(frames, (80, 80), [(40, 40, 1)], args, 1, 6)
        self.assertEqual(acc.call_args.kwargs["max_width_px"], 11)

    def test_gate_has_one_definition_and_unknown_is_excluded(self):
        cfg = Config(scale_m=2, window_m=56, slope_baseline_m=16)
        dem = np.zeros((120, 120), np.float32)
        dem[:, 60:] = 8
        np.testing.assert_array_equal(gate(dem, cfg), gate(channels(dem, cfg), cfg))
        self.assertTrue(gate({"rms_slope": np.array([[np.nan]])})[0, 0])

    def test_channels_honor_valid_mask_and_filter_support(self):
        cfg = Config(scale_m=2, window_m=12, slope_baseline_m=4)
        dem = np.zeros((100, 100), np.float32)
        valid = np.ones(dem.shape, bool)
        valid[50, 50] = False
        out = channels(dem, cfg, valid=valid)
        self.assertTrue(np.isnan(out["rms_slope"][50, 50]))
        self.assertTrue(np.isnan(out["rms_slope"][51, 50]))
        self.assertEqual(out["rms_slope"][10, 10], 0)


if __name__ == "__main__":
    unittest.main()
