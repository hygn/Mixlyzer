"""The diagnostics (app.diagnostics, the Diagnostics window): the console's lines kept with their times, each
output's buffer summed from the player's reports, the user's home folder and name left out, every section in the
report, and the window showing it as made (the app sends nothing)."""
import getpass
import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app import diagnostics  # noqa: E402
from core.settings.config import default_cfg  # noqa: E402
from ui.windows.diagnostics_dialog import DiagnosticsDialog  # noqa: E402

APP = QApplication.instance() or QApplication([])


class _Player:
    native = False

    def output_devices(self, api=None):
        return "Speakers (DDJ-400)", ["Speakers (DDJ-400)", "Headphones"], {"Speakers (DDJ-400)": 4, "Headphones": 2}

    def midi_ports(self):
        return ["DDJ-400"]


def _stats(seconds: float, dropout_sec: float, aux_dropout: float) -> dict:
    return {"hist": [1, 2], "seconds": seconds, "mean_ms": 20.0, "dropout_sec": dropout_sec, "device_ms": 10.0,
            "output_api": "wasapi", "exclusive": True, "late": 3, "lowest_ms": 4.5,
            "outputs": {"headphones": {"open": True, "device": "Headphones", "capacity_ms": 40.0, "late": 0,
                                       "observed": {"hist": [1], "seconds": seconds, "mean_ms": 30.0,
                                                    "dropout_sec": aux_dropout}},
                        "booth": None}}


class ConsoleRecordTests(unittest.TestCase):
    def test_whole_lines_kept_the_last_ones(self):
        record = diagnostics.ConsoleRecord(kept=2)
        for text in ("[AudioEngine] master opened", "", "[Player] play", "[Decoder] FFmpeg warm failed rc=1"):
            record.add(text, error=text.startswith("[Decoder]"))
        lines = record.lines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].endswith("  [Player] play"))
        self.assertTrue(lines[1].endswith("! [Decoder] FFmpeg warm failed rc=1"))


class BufferRecordTests(unittest.TestCase):
    def test_played_time_and_dropouts_summed_state_as_last_reported(self):
        record = diagnostics.BufferRecord()
        self.assertEqual(record.lines(), ["no buffer reported yet (nothing played)"])
        record.add(10.0, _stats(0.25, 0.0, 0.0), 40.0)
        record.add(10.0, _stats(0.25, 0.002, 0.001), 40.0)
        record.add(10.0, None, 40.0)
        master, headphones = record.outputs["master"], record.outputs["headphones"]
        self.assertAlmostEqual(master["played_sec"], 0.5)
        self.assertAlmostEqual(master["dropout_sec"], 0.002)
        self.assertAlmostEqual(headphones["dropout_sec"], 0.001)
        self.assertNotIn("booth", record.outputs)
        self.assertNotIn("hist", master["state"])
        self.assertEqual(master["state"]["late"], 3)
        self.assertIn("master: played 0.5 s, dropouts 2.0 ms, buffer 40.0 ms", record.lines())


class RedactTests(unittest.TestCase):
    def test_home_and_name(self):
        home = os.path.expanduser("~")
        user = getpass.getuser()
        text = diagnostics.redact(f"{home}{os.sep}Music{os.sep}a.mp3 | {home.upper()} | {user}'s AirPods")
        self.assertNotIn(home, text)
        self.assertIn("~", text)
        if len(user) >= 3:
            self.assertNotIn(user, text)
            self.assertIn("<user>'s AirPods", text)


class ReportTests(unittest.TestCase):
    def test_every_section(self):
        record = diagnostics.ConsoleRecord()
        record.add("[AudioEngine] master: WASAPI exclusive")
        buffers = diagnostics.BufferRecord()
        buffers.add(10.0, _stats(0.25, 0.0, 0.0), 40.0)
        text = diagnostics.report(default_cfg(), _Player(), buffers, record)
        for title in ("== Mixlyzer", "== System", "== FFmpeg", "== Audio setup", "== Output devices",
                      "== MIDI inputs", "== Output buffers since the start", "== Console (last 1 lines)"):
            self.assertIn(title, text)
        self.assertIn("* Speakers (DDJ-400): 4 channels", text)
        self.assertIn("[AudioEngine] master: WASAPI exclusive", text)
        self.assertNotIn(os.path.expanduser("~"), text)


class DialogTests(unittest.TestCase):
    def test_made_when_shown_and_copied(self):
        made = mock.Mock(return_value="== Mixlyzer\nversion 0.1.0\n")
        dialog = DiagnosticsDialog(made)
        dialog.show()
        self.assertEqual(made.call_count, 1)
        self.assertEqual(dialog.text.toPlainText(), "== Mixlyzer\nversion 0.1.0\n")
        dialog.refresh()
        self.assertEqual(made.call_count, 2)
        dialog.close()


if __name__ == "__main__":
    unittest.main()
