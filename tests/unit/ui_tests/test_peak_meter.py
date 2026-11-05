"""The channel's peak meter: up at once, down fast enough to follow the music's hits, the peak marker never lagging far."""
import unittest

from unit.ui_tests.qt_env import APP  # noqa: F401  (the application, offscreen)

from ui.deck.track_info_panel import VerticalPeakMeter


class PeakMeterTests(unittest.TestCase):
    def setUp(self):
        self.meter = VerticalPeakMeter()
        self.meter._anim.stop()     # its ticks by hand: 16 ms each

    def _ticks(self, seconds: float) -> None:
        for _ in range(int(seconds * 1000 / self.meter._anim.interval())):
            self.meter._tick()

    def test_up_at_once(self):
        self.meter.set_level_dbfs(-6.0)
        self.assertEqual(self.meter._display_dbfs, -6.0)

    def test_down_a_whole_segment_within_50_ms(self):
        self.meter.set_level_dbfs(-6.0)
        self.meter.set_level_dbfs(-30.0)
        self._ticks(0.05)
        self.assertLessEqual(self.meter._display_dbfs, -9.0)

    def test_the_whole_scale_within_a_second_stopping_at_the_level(self):
        self.meter.set_level_dbfs(0.0)
        self.meter.set_level_dbfs(self.meter.FLOOR_DBFS)
        self._ticks(1.0)
        self.assertEqual(self.meter._display_dbfs, self.meter.FLOOR_DBFS)
        self.meter.set_level_dbfs(-20.0)
        self.meter.set_level_dbfs(-24.0)
        self._ticks(1.0)
        self.assertEqual(self.meter._display_dbfs, -24.0)

    def test_the_marker_falls_no_slower_than_the_bar(self):
        self.assertGreaterEqual(self.meter._peak_fall_db_per_sec, self.meter._fall_db_per_sec)


if __name__ == "__main__":
    unittest.main()
