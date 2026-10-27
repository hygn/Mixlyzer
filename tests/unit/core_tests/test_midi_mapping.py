"""core.controls.midi: a controller's messages as rekordbox's MIDI CSV writes them ("900B", "B000H+"), the actions
they are bound to, the user's mapping file and the import of a rekordbox mapping. The CSV rows here are written for
the test (the format only: no vendor's mapping file)."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from audioengine import controls as engine
from core.controls import midi
from ui import shortcuts as ui_shortcuts


class MessageTests(unittest.TestCase):
    def test_kinds(self):
        note = midi.Message.parse("900B")
        self.assertEqual((note.status, note.data1, note.hires, note.direction), (0x90, 0x0B, False, 0))
        self.assertEqual(midi.Message.parse("800B"), note)                    # a note off read as its note on
        self.assertEqual(midi.Message.parse("9f7f").status, 0x9F)            # case and channel 16
        self.assertEqual(midi.Message.parse("  B000  ").status, 0xB0)
        bend = midi.Message.parse("E07F")
        self.assertEqual((bend.status, bend.data1), (0xE0, 0))               # a pitch bend's data 1 not read
        self.assertEqual(midi.Message.parse("A140").status, 0xA1)            # aftertouch

    def test_refused(self):
        for text in (None, "", "90", "900", "90000", "9080", "90FF", "C000", "D000", "F000", "7000", "G000",
                     "900BX", "B000+-", "B000OT", "B000OS", "B000TS", "90 0B", "0x900B"):
            with self.subTest(text=text):
                self.assertIsNone(midi.Message.parse(text))

    def test_options_only_where_they_mean_something(self):
        cc = midi.Message.parse("B01FH+IO")
        self.assertEqual((cc.hires, cc.direction, cc.invert, cc.encoding), (True, 1, True, "O"))
        note = midi.Message.parse("901FH+IO")
        self.assertEqual((note.hires, note.direction, note.invert, note.encoding), (False, 0, True, ""))
        self.assertEqual(midi.Message.parse("B000HH").hires, True)            # repeated: as once
        self.assertEqual(midi.Message.parse("B000--").direction, -1)

    def test_text_round_trip(self):
        """Every message written is read back the same, and written the same again."""
        count = 0
        options = ["", "H", "+", "-", "I", "O", "T", "S", "H+I", "H-T", "+IS", "HIO"]
        for kind, channel, data1, opt in itertools.product("89ABE", "05F", ("00", "1F", "40", "7F"), options):
            m = midi.Message.parse(f"{kind}{channel}{data1}{opt}")
            self.assertIsNotNone(m)
            again = midi.Message.parse(m.text())
            self.assertEqual(again, m)
            self.assertEqual(again.text(), m.text())
            count += 1
        self.assertEqual(count, 5 * 3 * 4 * len(options))

    def test_flags(self):
        self.assertEqual(midi.Message.parse("B000").flags(), 0)
        m = midi.Message.parse("B000H-IS")
        self.assertEqual(m.flags(), engine.BIND_HIRES | engine.BIND_COUNTERCLOCKWISE | engine.BIND_INVERT
                         | engine.BIND_ENC_SIGN << engine.BIND_ENCODING_SHIFT)
        self.assertEqual(midi.Message.parse("B000+").flags(), engine.BIND_CLOCKWISE)
        for enc, code in (("O", engine.BIND_ENC_OFFSET), ("T", engine.BIND_ENC_TWOS), ("S", engine.BIND_ENC_SIGN)):
            self.assertEqual(midi.Message.parse("B000" + enc).flags() >> engine.BIND_ENCODING_SHIFT, code)

    def test_labels(self):
        self.assertEqual(midi.Message.parse("900B").label(), "Ch 1 Note 0x0B")
        self.assertEqual(midi.Message.parse("B000H").label(), "Ch 1 CC 0x00 (14-bit)")
        self.assertEqual(midi.Message.parse("B640+").label(), "Ch 7 CC 0x40 ↻")
        self.assertEqual(midi.Message.parse("E0").label() if midi.Message.parse("E0") else None, None)
        self.assertEqual(midi.Message.parse("EF00").label(), "Ch 16 Pitch bend")


class RelativeStepsTests(unittest.TestCase):
    def test_every_value(self):
        steps = [midi.relative_steps(v) for v in range(128)]
        self.assertEqual((steps[0], steps[64]), (0, 0))                         # no turn
        self.assertEqual(steps[1:32], list(range(1, 32)))                      # two's complement, clockwise
        self.assertEqual(steps[97:128], list(range(-31, 0)))                   # two's complement, back
        self.assertEqual(steps[65:97], list(range(1, 33)))                     # 64 + steps
        self.assertEqual(steps[32:64], list(range(-32, 0)))                    # 64 - steps
        self.assertTrue(all(abs(s) <= 32 for s in steps))

    def test_out_of_range(self):
        for v in (-1, 128, 255, 1000):
            self.assertEqual(midi.relative_steps(v), 0)


class ActionTests(unittest.TestCase):
    """The actions Settings > MIDI shows (ACTIONS_SHOWN): the ones a user binds."""

    def test_ids_unique_and_grouped(self):
        ids = [a.id for a in midi.ACTIONS]
        self.assertEqual(len(ids), len(set(ids)))
        for a in midi.ACTIONS_SHOWN:
            with self.subTest(action=a.id):
                self.assertIn(a.group, midi.GROUPS)
                self.assertIn(a.kind, (midi.BUTTON, midi.FADER, midi.KNOB, midi.TOUCH, midi.METER))
                self.assertTrue(bool(a.code) != bool(a.gui), "the engine's or the GUI's, never both or none")
                if a.kind == midi.KNOB and a.gui:
                    self.assertEqual(len(a.gui), 2)                             # clockwise, counterclockwise
                self.assertTrue(0 <= a.lamp <= 26 or a.lamp >= 32)
                midi.action_label(a)                                            # its text fills in

    def test_engine_codes_unique(self):
        codes = [a.code for a in midi.ACTIONS if a.code]
        self.assertEqual(len(codes), len(set(codes)))
        for a in midi.ACTIONS_SHOWN:
            if a.code:
                self.assertEqual(a.code >> engine.ACT_DECK_SHIFT, 0)            # deck 1's, mixer channel 1's
                self.assertLess(a.code, engine.ACT_GUI)

    def test_shown_ones_are_deck_one(self):
        for a in midi.ACTIONS_SHOWN:
            self.assertNotRegex(a.id, r"^(deck|mixer\.ch)(?!1\.)\d")

    def test_gui_actions_are_shortcuts(self):
        shortcut_ids = {a.id for a in ui_shortcuts.ACTIONS}
        for a in midi.ACTIONS_SHOWN:
            for gui in a.gui:
                with self.subTest(action=a.id, gui=gui):
                    self.assertIn(gui, shortcut_ids)

    def test_gui_codes_round_trip(self):
        for a in midi.ACTIONS_SHOWN:
            self.assertIs(midi.gui_action(midi.gui_code(a)), a)
        self.assertIsNone(midi.gui_action(engine.ACT_GUI - 1))
        self.assertIsNone(midi.gui_action(engine.ACT_GUI + len(midi.ACTIONS)))
        self.assertLess(midi.gui_code(midi.ACTIONS[-1]), 1 << engine.ACT_DECK_SHIFT)

    def test_bindings(self):
        got = midi.bindings({"deck1.play_pause": ["900B", "nonsense", "910B"], "no.such.action": ["9000"],
                             "browser.rotate": ["B640O"]})
        play = midi.BY_ID["deck1.play_pause"]
        rotate = midi.BY_ID["browser.rotate"]
        self.assertEqual(got, [(0x90, 0x0B, 0, play.lamp, play.code), (0x91, 0x0B, 0, play.lamp, play.code),
                               (0xB6, 0x40, engine.BIND_ENC_OFFSET << engine.BIND_ENCODING_SHIFT, 0,
                                midi.gui_code(rotate))])


class MappingFileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.user = self.dir / "midi.json"
        self.shipped = self.dir / "default_midi.json"
        self.shipped.write_text(json.dumps({"deck1.cue": ["900C"]}), encoding="utf-8")
        for name, value in (("USER_FILE", str(self.user)), ("DEFAULT_FILE", str(self.shipped))):
            patcher = mock.patch.object(midi, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_only_changes_written(self):
        midi.save({"deck1.cue": ["900C"], "deck1.play_pause": ["900b"]})
        self.assertEqual(json.loads(self.user.read_text(encoding="utf-8")), {"deck1.play_pause": ["900b"]})
        self.assertEqual(midi.mapping(), {"deck1.cue": ["900C"], "deck1.play_pause": ["900B"]})
        midi.save({"deck1.cue": ["900C"]})
        self.assertFalse(self.user.exists())                                   # nothing changed: no file

    def test_shipped_never_written(self):
        with self.assertRaises(ValueError):
            midi.save({"deck1.cue": ["9000"]}, path=str(self.shipped))
        self.assertEqual(json.loads(self.shipped.read_text(encoding="utf-8")), {"deck1.cue": ["900C"]})

    def test_malformed_files(self):
        for text in ("{", "[1, 2]", '"x"', "﻿{}", ""):
            with self.subTest(text=text):
                self.user.write_text(text, encoding="utf-8")
                self.assertEqual(midi.user_changes(), {})
        self.user.write_bytes(b"\xff\xfe\x00")
        self.assertEqual(midi.user_changes(), {})
        self.user.write_text(json.dumps({"a": "900B", "b": [1], "c": ["900B", "bad", "B0"], "d": []}),
                             encoding="utf-8")
        self.assertEqual(midi.user_changes(), {"c": ["900B"], "d": []})
        self.user.unlink()
        self.assertEqual(midi.user_changes(), {})


CSV_HEADER = "#name,function,type,input,deck1,deck2,deck3,deck4,output,deck1,deck2,deck3,deck4,option,comment\n"


class RekordboxImportTests(unittest.TestCase):
    def read(self, rows: list[str], bom: bool = True):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.csv"
            path.write_text(("﻿" if bom else "") + CSV_HEADER + "\n".join(rows) + "\n", encoding="utf-8")
            return midi.import_rekordbox_csv(str(path))

    def test_deck_one_column_and_channel_offsets(self):
        mapping, skipped = self.read([
            "PlayPause,PlayPause,Button,900B,900B,910B,,,,,,,,,",       # a whole message in deck 1's column
            "Cue,Cue,Button,900C,0,1,,,,,,,,,",                          # a channel offset to the input's
            "TempoSlider,TempoSlider,KnobSliderHiRes,B000,1,,,,,,,,,,",  # 14 bits: H; channel 2
            "Load,Load,Button,9046,9046,,,,,,,,,,",
            "EQHigh,EQHigh,KnobSliderHiRes,B007,B007,,,,,,,,,,",
        ])
        self.assertEqual(mapping, {"deck1.play_pause": ["900B"], "deck1.cue": ["900C"],
                                   "deck1.tempo_fader": ["B100H"], "deck1.load": ["9046"],
                                   "mixer.ch1.eq_high": ["B007H"]})
        self.assertEqual(skipped, [])

    def test_offset_wraps_within_the_status(self):
        mapping, _ = self.read(["Cue,Cue,Button,9F0C,1,,,,,,,,,,"])          # channel 16 + 1: channel 1, a note
        self.assertEqual(mapping, {"deck1.cue": ["900C"]})

    def test_left_out(self):
        mapping, skipped = self.read([
            "@comment,,,,,,,,,,,,,,",
            "",
            "JogTouch,JogTouch+Shift,Button,9067,9067,,,,,,,,,,",       # the SHIFT layer of the same function
            "HeadphonesMix,HeadphonesMix,KnobSlider,B60C,,,,,,,,,,,No function assigned (analog)",
            "Unknown,Unknown,Button,9000,9000,,,,,,,,,,",                 # not a Mixlyzer action
            "PlayPause,PlayPause,Button,910B,,910B,,,,,,,,,",             # deck 2 only
            "Cue,Cue,Button,zz,zz,,,,,,,,,,",                             # not a message
        ], bom=False)
        self.assertEqual(mapping, {})
        self.assertEqual(skipped, sorted({"JogTouch+Shift", "HeadphonesMix", "Unknown", "PlayPause", "Cue"}))

    def test_global_and_output_columns(self):
        mapping, _ = self.read([
            "Browse,Browse,Rotary,B640,,,,,,,,,,,",
            "HeadphonesVolume,HeadphonesVolume,KnobSlider,b60d,,,,,,,,,,,",
            "CFXParameterCH1,CFXParameterCH1,KnobSlider,B617,,,,,,,,,,,",
            "ChannelLevel,ChannelLevel,Indicator,,,,,,B002,0,,,,,",           # deck 1's output: the offset
            "PlayPause,PlayPause,Button,900B,900B,,,,,,,,,,",
            "PlayPause,PlayPause,Button,900B,900B,,,,,,,,,,",                 # the same again: once
        ])
        self.assertEqual(mapping, {"browser.rotate": ["B640"], "mixer.headphones_level": ["B60D"],
                                   "mixer.ch1.color": ["B617"], "mixer.ch1.level_meter": ["B002"],
                                   "deck1.play_pause": ["900B"]})

    def test_short_rows_padded(self):
        mapping, skipped = self.read(["PlayPause,PlayPause,Button,900B,900B"])
        self.assertEqual(mapping, {"deck1.play_pause": ["900B"]})
        self.assertEqual(skipped, [])


if __name__ == "__main__":
    unittest.main()
