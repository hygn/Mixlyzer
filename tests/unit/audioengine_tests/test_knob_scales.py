"""audioengine.controls / audioengine.volume: the knobs' and faders' scales (position <-> dB) the GUI shows and the
engines take, their ends, their middles and their inverses, and the controls' tables the GUI and the engines share."""
import math
import unittest

import numpy as np

from audioengine import controls as c
from audioengine import volume as v

POSITIONS = np.linspace(0.0, 1.0, 1001)


class TrimTests(unittest.TestCase):
    def test_ends_and_middle(self):
        lo, hi = v.TRIM_DB_RANGE
        self.assertEqual((v.trim_knob_to_db(0.0), v.trim_knob_to_db(0.5), v.trim_knob_to_db(1.0)), (lo, 0.0, hi))
        self.assertEqual((v.trim_db_to_knob(lo), v.trim_db_to_knob(0.0), v.trim_db_to_knob(hi)), (0.0, 0.5, 1.0))

    def test_held_to_the_range(self):
        lo, hi = v.TRIM_DB_RANGE
        self.assertEqual(v.trim_db_to_knob(lo - 100), 0.0)
        self.assertEqual(v.trim_db_to_knob(hi + 100), 1.0)
        self.assertEqual(v.trim_knob_to_db(-3.0), lo)
        self.assertEqual(v.trim_knob_to_db(7.0), hi)

    def test_inverse_and_monotonic(self):
        db = np.array([v.trim_knob_to_db(k) for k in POSITIONS])
        self.assertTrue(np.all(np.diff(db) > 0))
        np.testing.assert_allclose([v.trim_db_to_knob(d) for d in db], POSITIONS, atol=1e-12)

    def test_linear_gain(self):
        self.assertEqual(v.trim_dbfs_to_linear(0.0), 1.0)
        self.assertAlmostEqual(v.trim_dbfs_to_linear(-6.0), 10 ** (-6 / 20))
        self.assertLess(v.trim_dbfs_to_linear(-1000.0), 1e-40)
        self.assertEqual(v.trim_dbfs_to_linear(9.0), 1.0)          # held at unity (the Python engine's gain)


class VolumeFaderTests(unittest.TestCase):
    def test_ends(self):
        self.assertEqual(v.slider_percent_to_linear(0, 0.0), 0.0)
        self.assertAlmostEqual(v.slider_percent_to_linear(100, 0.0), 1.0)
        self.assertEqual(v.slider_percent_to_linear(-50, 0.0), 0.0)
        self.assertAlmostEqual(v.slider_percent_to_linear(250, 0.0), 1.0)

    def test_monotonic_and_bounded(self):
        gains = np.array([v.slider_percent_to_linear(p, -3.0) for p in np.linspace(0, 100, 401)])
        self.assertTrue(np.all(np.diff(gains) >= 0))
        self.assertTrue(np.all((gains >= 0) & (gains <= 1)))
        self.assertLessEqual(gains[-1], v.trim_dbfs_to_linear(-3.0) + 1e-12)


class LevelKnobTests(unittest.TestCase):
    def test_unity_and_ends(self):
        self.assertAlmostEqual(c.level_knob_to_db(c.LEVEL_UNITY), 0.0)
        self.assertEqual(c.level_knob_to_db(0.0), c.LEVEL_DB_MIN)
        self.assertAlmostEqual(c.level_knob_to_db(1.0), 40 * math.log10(1 / c.LEVEL_UNITY))   # src/mixer.rs: (k / u)^2
        self.assertEqual(c.level_knob_to_db(-1.0), c.LEVEL_DB_MIN)
        self.assertEqual(c.level_knob_to_db(5.0), c.level_knob_to_db(1.0))
        self.assertEqual(c.level_db_to_knob(c.LEVEL_DB_MIN), 0.0)
        self.assertEqual(c.level_db_to_knob(-1000.0), 0.0)
        self.assertEqual(c.level_db_to_knob(100.0), 1.0)
        self.assertAlmostEqual(c.level_db_to_knob(0.0), c.LEVEL_UNITY)

    def test_inverse_above_the_floor(self):
        db = np.array([c.level_knob_to_db(k) for k in POSITIONS])
        self.assertTrue(np.all(np.diff(db) >= 0))
        above = db > c.LEVEL_DB_MIN
        np.testing.assert_allclose([c.level_db_to_knob(d) for d in db[above]], POSITIONS[above], atol=1e-12)

    def test_booth_eq(self):
        lo, hi = c.BOOTH_EQ_DB
        self.assertEqual((c.booth_eq_knob_to_db(0.0), c.booth_eq_knob_to_db(0.5), c.booth_eq_knob_to_db(1.0)),
                         (lo, 0.0, hi))
        db = np.array([c.booth_eq_knob_to_db(k) for k in POSITIONS])
        self.assertTrue(np.all(np.diff(db) > 0))
        np.testing.assert_allclose([c.booth_eq_db_to_knob(d) for d in db], POSITIONS, atol=1e-12)
        self.assertEqual((c.booth_eq_db_to_knob(-99.0), c.booth_eq_db_to_knob(99.0)), (0.0, 1.0))


class ControlTableTests(unittest.TestCase):
    def test_effects(self):
        self.assertEqual(len(c.COLOR_SLOTS), len(c.COLOR_TYPES))
        self.assertTrue(set(c.COLOR_SLOTS) <= {"pre_eq", "post_eq", "post_fader"})
        self.assertEqual(c.COLOR_OFF, len(c.COLOR_TYPES))                       # "none" right after the effects
        self.assertEqual(len(set(c.COLOR_TYPES)), len(c.COLOR_TYPES))
        self.assertEqual((c.EQ_MODES, c.FADER_CURVES), (("isolator", "eq"), ("default", "steep", "slow")))

    def test_mixer_keys(self):
        self.assertEqual(len(set(c.MIX_KEYS)), len(c.MIX_KEYS))
        self.assertEqual([c.MIX_KEY[k] for k in c.MIX_KEYS], list(range(len(c.MIX_KEYS))))

    def test_deck_actions_fit_under_the_deck_bits(self):
        acts = {name: value for name, value in vars(c).items() if name.startswith("ACT_") and isinstance(value, int)}
        codes = [value for name, value in acts.items() if name not in ("ACT_DECK_SHIFT", "ACT_GUI")]
        self.assertTrue(all(0 < code < 1 << c.ACT_DECK_SHIFT for code in codes))
        self.assertLess(c.ACT_JUMP_A + 7, c.ACT_BEND_DOWN)                       # pads A - H before the next act
        last_jump = c.ACT_BEAT_JUMP + 2 * len(c.BEAT_JUMP_SIZES) - 1
        self.assertGreater(c.ACT_GUI, last_jump)
        ranges = [(c.ACT_JUMP_A, 8), (c.ACT_BEAT_JUMP, 2 * len(c.BEAT_JUMP_SIZES)),
                  (c.ACT_COLOR_SELECT, len(c.COLOR_TYPES))]                       # an action and the codes after it
        taken = [code for start, n in ranges for code in range(start, start + n)]
        others = [code for code in codes if not any(start <= code < start + n for start, n in ranges)]
        self.assertEqual(len(taken + others), len(set(taken + others)), "two actions share a code")

    def test_lamps(self):
        lamps = [value for name, value in vars(c).items() if name.startswith("LAMP_")]
        self.assertTrue(all(value > 26 for value in lamps))                     # 1 - 26 are the JumpCUE A - Z
        self.assertLess(c.LAMP_COLOR + len(c.COLOR_TYPES) - 1, c.LAMP_VINYL)
        self.assertEqual(len(set(lamps)), len(lamps))

    def test_tempo_ranges_and_buffers(self):
        self.assertEqual(list(c.TEMPO_RANGES), sorted(c.TEMPO_RANGES))
        for lo, hi in (c.NATIVE_BUFFER_MS_RANGE, c.PYTHON_BUFFER_MS_RANGE):
            self.assertLess(0, lo)
            self.assertLess(lo, hi)
        self.assertLess(c.PEAK_FLOOR_DBFS, -42.0)                                # below the meter's lowest segment


class DeckReportTests(unittest.TestCase):
    def test_values_and_key(self):
        report = c.DeckReport(1, "1.5", 0.25, float("nan"), float("nan"), 7, 10, 1, knobs=[0, 1, 0.5, 0.5],
                              mix=[1.0, 2.0])
        self.assertIs(report.playing, True)
        self.assertEqual(report.factor, 1.5)
        self.assertTrue(report.vinyl)
        self.assertEqual(report.knobs, (0.0, 1.0, 0.5, 0.5))
        self.assertEqual(report.mix_value("fader_curve"), 2.0)
        self.assertEqual(report.mix_value("master_level", 0.8), 0.8)            # not reported: the default
        self.assertEqual(report.mix_value("no such key", -1.0), -1.0)
        louder = c.DeckReport(1, 1.5, 0.25, 0.0, 0.0, 7, 10, 1, peak=-3.0)
        quieter = c.DeckReport(1, 1.5, 0.25, 0.0, 0.0, 7, 10, 1, peak=-20.0)
        self.assertEqual(louder.key(), quieter.key())                            # the peak is the meters'
        self.assertNotEqual(louder.key(), c.DeckReport(0, 1.5, 0.25, 0.0, 0.0, 7, 10, 1).key())


if __name__ == "__main__":
    unittest.main()
