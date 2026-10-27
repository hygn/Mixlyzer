"""The settings dialog (offscreen Qt, on `default_cfg()`): every value shown and read back as it was, a page's change
reaching the config it hands on (Apply / OK), what playing changes kept as it is, the Shortcuts page (a key another
action has taken only once agreed, Revert, Revert All, saved as the user's changes alone) and the MIDI page (disabled
on the Python engine; Learn; the actions it does not show kept)."""
import json
import os
import unittest
from dataclasses import replace
from unittest import mock

from unit.ui_tests.qt_env import APP, drain, isolated

from PySide6 import QtGui
from PySide6.QtWidgets import QDialogButtonBox, QMessageBox

from audioengine.controls import booth_eq_db_to_knob, level_db_to_knob
from core.controls import midi, shortcuts
from core.settings.config import cuepointrule, default_cfg, outputconfig
from core.workflows.event_bus import EventBus
from system import os_support
from ui.config.audio_overview import AudioOverview
from ui.config.window import SettingsDialog

_ISOLATION = None


def setUpModule():
    global _ISOLATION
    _ISOLATION = isolated()


def tearDownModule():
    _ISOLATION.close()


def edited_cfg():
    """Every page's values off their defaults (deck.json's too, which the dialog does not edit)."""
    cfg = default_cfg()
    cfg.analysisconfig = replace(cfg.analysisconfig, bpm_min=85, bpm_dynamic=False, dynamic_downbeat=False,
                                 key_dynamic=False, autocue="jumpcue", key_parameter_path="mine/key.npz",
                                 max_parallel_analyses=1)
    cfg.libconfig = replace(cfg.libconfig, libpath="elsewhere/library", local_music_directories=["D:/Music"],
                            write_log=True, rekordbox_sync_enabled=False, rekordbox_ramp_tolerance_ms=2.5,
                            rekordbox_comment_tieups=True, tieup_to_versions=False, tieup_web_search=True)
    names = cfg.viewconfig.phrase_labels
    cfg.viewconfig = replace(cfg.viewconfig, language="ja-JP", display_keystrip=False, waveform_style="3band",
                             env_mid=(300.0, 1800.0), fps=30, reduce_fps_when_occluded=False,
                             phrase_labels=replace(names, chorus=replace(names.chorus, alias="Hook", abbrev="H")))
    cfg.playbackconfig = replace(
        cfg.playbackconfig, default_volume_percent=70, soft_clip=False, audio_engine="python", native_buffer_ms=12,
        python_buffer_ms=120, output_device="Speakers (test)", native_exclusive=True, native_exclusive_rate=48000,
        native_priority="above_normal", native_power_throttling=True, midi_input="Controller (test)",
        midi_jog_ticks=1024, headphones_master_mix=False, booth_level=level_db_to_knob(-6.0),
        booth_eq_hi=booth_eq_db_to_knob(3.0), booth_eq_low=booth_eq_db_to_knob(-4.5), headphones_split=True,
        output_headphones=outputconfig("device", "Phones (test)", "1-2", "wasapi", True, 30, 44100))
    cfg.externalsyncconfig = replace(cfg.externalsyncconfig, enabled=True, osc_host="0.0.0.0", osc_port=40000)
    cfg.cuepointconfig = replace(cfg.cuepointconfig, chorus=cuepointrule(True, True, True))
    cfg.searchconfig = replace(cfg.searchconfig, db_languages=["ja", "zh"], han_runs=False, query_languages=["ko-hanja"],
                               query_keyboard_layout=False, music_index=False, music_listen_window=8.0,
                               music_listen_interval=3.0)
    cfg.deckconfig = replace(cfg.deckconfig, use_timestretch=False, enable_metronome=True, volume_trim_dbfs=-3.0,
                             tempo_range=16, jog_mode="cdj", mixer_eq="eq", master_level=0.6, color_type=2,
                             music_input_device="Mic (test)", metronome_offset_msec=12.0,
                             metronome_beat_pitch_semitones=-2.0, headphones_cue_mix=30)
    return cfg


class SettingsDialogTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.saved = []
        self.bus.sig_setting_saveJsonRequested.connect(self.saved.append)
        self.d = SettingsDialog(self.bus)

    def tearDown(self):
        self.d.close()
        self.d.deleteLater()
        drain()

    def test_pages(self):
        self.assertEqual([self.d.tabs.tabText(i) for i in range(self.d.tabs.count())],
                         ["View", "Audio", "Search", "Library", "Analysis", "Shortcuts", "MIDI", "External Sync"])
        self.assertEqual([self.d.audio_tabs.tabText(i) for i in range(self.d.audio_tabs.count())],
                         ["Overview", "Engine", "Main output", "Headphones", "Booth", "Mixer", "Metronome"])

    def test_defaults_read_back_as_they_are(self):
        cfg = default_cfg()
        self.d.set_config(cfg)
        self.assertEqual(self.d.get_config(), cfg)

    def test_every_value_read_back_as_it_is(self):
        cfg = edited_cfg()
        self.d.set_config(cfg)
        self.assertEqual(self.d.get_config(), cfg)

    def test_what_playing_changes_is_kept_as_it_is(self):
        """The main screen's values (the MT button, the TRIM knob, the toolbar's switches) have no field here."""
        cfg = edited_cfg()
        self.d.set_config(cfg)
        self.d.sp_metronome_offset_msec.setValue(-20.0)              # the one deck value edited here
        deck = self.d.get_config().deckconfig
        self.assertEqual(deck, replace(cfg.deckconfig, metronome_offset_msec=-20.0))

    def test_a_controllers_cue_mix_is_not_undone(self):
        """The headphones' CUE / MASTER mix a controller moved after the dialog was made: kept by OK, shown when the
        dialog opens; one changed here wins."""
        cfg = edited_cfg()
        self.d.set_config(cfg)
        cfg.deckconfig.headphones_cue_mix = 80                        # a controller, as AppWindow._keep_switches
        self.assertEqual(self.d.get_config().deckconfig.headphones_cue_mix, 80)
        self.d.show()
        drain()
        self.assertEqual(self.d.grp_output_headphones.sp_cue_mix.value(), 80)
        self.d.grp_output_headphones.sp_cue_mix.setValue(10)          # changed here
        cfg.deckconfig.headphones_cue_mix = 60                        # and by the controller since
        self.assertEqual(self.d.get_config().deckconfig.headphones_cue_mix, 10)
        self.d.hide()
        self.d.show()
        drain()
        self.assertEqual(self.d.grp_output_headphones.sp_cue_mix.value(), 10)   # the change here not dropped

    def test_a_change_applied_reaches_the_config(self):
        self.d.set_config(default_cfg())
        self.d.sp_bpm_min.setValue(90)
        self.d.cb_key_dynamic.setChecked(False)
        self.d.sp_fps.setValue(45)
        self.d.cmb_waveform_style.setCurrentIndex(self.d.cmb_waveform_style.findData("3band"))
        self.d.sp_booth_level.setValue(-12.0)
        self.d.sp_external_sync_port.setValue(5000)
        self.d.cue_rule_boxes["verse"][0].setChecked(True)
        self.d.btn_box.button(QDialogButtonBox.Apply).click()
        self.assertEqual(len(self.saved), 1)
        cfg = self.saved[0]
        self.assertEqual(cfg.analysisconfig.bpm_min, 90)
        self.assertFalse(cfg.analysisconfig.key_dynamic)
        self.assertEqual(cfg.viewconfig.fps, 45)
        self.assertEqual(cfg.viewconfig.waveform_style, "3band")
        self.assertAlmostEqual(cfg.playbackconfig.booth_level, level_db_to_knob(-12.0))
        self.assertEqual(cfg.externalsyncconfig.osc_port, 5000)
        self.assertTrue(cfg.cuepointconfig.verse.at_start)
        self.assertEqual(self.d.lbl_bpm_range.text(), "Reads 90 - 180 BPM")
        self.assertFalse(os.path.exists(shortcuts.USER_FILE))        # the shortcuts as shipped: no user file
        self.d.btn_box.button(QDialogButtonBox.Cancel).click()
        self.assertEqual(len(self.saved), 1)

    def test_dynamic_downbeat_always_on_with_dynamic_analysis(self):
        cfg = default_cfg()
        cfg.analysisconfig.dynamic_downbeat = False          # the static analysis's choice
        self.d.set_config(cfg)
        self.assertTrue(self.d.cb_dynamic_downbeat.isChecked())
        self.assertFalse(self.d.cb_dynamic_downbeat.isEnabled())
        self.assertFalse(self.d.get_config().analysisconfig.dynamic_downbeat)    # kept while dynamic
        self.d.cb_bpm_dynamic.setChecked(False)
        self.assertTrue(self.d.cb_dynamic_downbeat.isEnabled())
        self.assertFalse(self.d.cb_dynamic_downbeat.isChecked())
        self.d.cb_dynamic_downbeat.setChecked(True)
        self.assertTrue(self.d.get_config().analysisconfig.dynamic_downbeat)

    def test_one_reading_language_stays_checked(self):
        self.d.set_config(default_cfg())
        for cb in self.d.db_language_boxes.values():
            cb.setChecked(False)
        self.assertEqual(len(self.d.get_config().searchconfig.db_languages), 1)

    def test_a_network_address_warned_about(self):
        self.d.set_config(default_cfg())
        self.assertTrue(self.d.lbl_external_sync_network.isHidden())
        self.d.ed_external_sync_host.setText("192.168.0.5")
        self.assertFalse(self.d.lbl_external_sync_network.isHidden())
        self.d.ed_external_sync_host.setText("")
        self.assertTrue(self.d.lbl_external_sync_network.isHidden())
        self.assertEqual(self.d.get_config().externalsyncconfig.osc_host, "127.0.0.1")    # empty: this computer

    def test_the_python_engine_hides_the_native_engines_rows(self):
        self.d.set_config(default_cfg())
        self.d.set_audio_engine(native=False, devices=None)
        self.assertFalse(self.d.btn_restart_engine.isEnabled())
        self.assertTrue(all(w.isHidden() for _label, w in self.d._native_rows))
        devices = mock.Mock(return_value=("Out A", ["Out A", "Out B"], {"Out A": 2, "Out B": 4}))
        self.d.set_audio_engine(native=True, devices=devices)
        self.d._load_output_devices("Out B")
        self.assertTrue(self.d.btn_restart_engine.isEnabled())
        self.assertEqual([self.d.cb_output_device.itemData(i) for i in range(self.d.cb_output_device.count())],
                         ["", "Out A", "Out B"])
        self.assertEqual(self.d.get_config().playbackconfig.output_device, "Out B")


class AudioOverviewTests(unittest.TestCase):
    """Settings > Audio > Overview: the signal path drawn from the engine's buffer stats (synthetic here)."""

    STATS = {"rate": 48000, "output_api": "wasapi", "exclusive": True, "channels": 4, "period_ms": 5.33,
             "device_ms": 6.0, "hist": [0.0, 0.0, 0.1, 0.8, 0.1], "seconds": 0.25, "mean_ms": 0.3, "bucket_ms": 0.1,
             "late": 0, "engine_thread": "mmcss", "output_thread": "mmcss",
             "outputs": {"headphones": {"open": True, "device": "Phones (test)", "output_api": "wasapi", "rate": 44100,
                                        "capacity_ms": 20.0, "device_ms": 3.0, "drift_ppm": 12.5, "late": 0}}}

    def setUp(self):
        self.view = AudioOverview()
        self.view.resize(900, 700)
        self.ctx = {"native": True, "timestretch": True, "main_device": "Speakers (test)", "api_names": {},
                    "routes": {"headphones": outputconfig("device", "Phones (test)", "1-2", "wasapi", True, 20, 0),
                               "booth": default_cfg().playbackconfig.output_booth},
                    "midi": ("", 0), "mix": {}, "deck": {"playing": True, "factor": 1.02}, "soft_clip": True}

    def tearDown(self):
        self.view.deleteLater()
        drain()

    def drawn(self) -> dict:
        self.assertFalse(self.view.canvas.grab().isNull())          # painted without an error
        return self.view.canvas._m

    def test_before_any_stats_on_the_python_engine(self):
        self.view.update_view(None, 0.0, {"native": False})
        m = self.drawn()
        self.assertFalse(m["native"])
        self.assertEqual(m["main"]["readout"], "waiting for playback")
        self.assertFalse(m["headphones"]["routed"])

    def test_the_outputs_on_their_clocks(self):
        self.view.update_view(dict(self.STATS), 20.0, self.ctx)
        m = self.drawn()
        self.assertEqual(m["main"]["state"], "ok")
        self.assertIn("48 kHz", m["main"]["lines"][1])
        self.assertEqual(m["headphones"]["domain"], 2)                    # another device's clock
        self.assertEqual(m["headphones"]["drift"], "+12.5 ppm")
        self.assertEqual(len(m["legend"]), 2)
        self.assertIn("Phones (test)", m["legend"][1][1])
        self.assertEqual(m["deck_lines"][1], "+2.00 %")

    def test_a_block_in_trouble_ringed(self):
        self.view.update_view(dict(self.STATS, late=3), 20.0, self.ctx)
        self.assertEqual(self.drawn()["main"]["state"], "warn")
        self.view.update_view(dict(self.STATS, dropout_sec=0.02), 20.0, self.ctx)
        self.assertEqual(self.drawn()["main"]["state"], "bad")
        self.view.update_view({"output_open": False}, 20.0, self.ctx)
        self.assertEqual(self.drawn()["main"]["state"], "bad")
        stats = dict(self.STATS, outputs={"headphones": {"open": False}})
        self.view.update_view(stats, 20.0, self.ctx)
        self.assertEqual(self.drawn()["headphones"]["state"], "bad")

    def test_the_main_monitor_says_when_the_device_is_not_open(self):
        d = SettingsDialog(EventBus())
        try:
            d.set_config(default_cfg())
            d.show()
            d._on_output_buffer(0.0, {"output_open": False}, 20.0)
            self.assertIn("could not be opened", d.main_monitor.label.text())
        finally:
            d.close()
            d.deleteLater()


class ShortcutPageTests(unittest.TestCase):
    def setUp(self):
        if os.path.exists(shortcuts.USER_FILE):
            os.remove(shortcuts.USER_FILE)
        self.bus = EventBus()
        self.d = SettingsDialog(self.bus)
        self.defaults = shortcuts.defaults()

    def tearDown(self):
        self.d.close()
        self.d.deleteLater()
        drain()
        if os.path.exists(shortcuts.USER_FILE):
            os.remove(shortcuts.USER_FILE)

    def press(self, action_id: str, key: str, slot: int = 0) -> None:
        """The action picked and ``key`` typed into its main (``slot`` 0) / alternative field."""
        self.d.tree_shortcuts.setCurrentItem(self.d._shortcut_items[action_id])
        (self.d.kse_shortcut_main, self.d.kse_shortcut_alt)[slot].setKeySequence(QtGui.QKeySequence(key))
        drain()

    def test_a_free_key_assigned_and_saved_as_a_change(self):
        free = "Ctrl+Alt+Shift+F7"
        self.assertNotIn(free, [k for keys in self.defaults.values() for k in keys])
        self.press("library.reveal", free, slot=1)
        self.assertEqual(self.d._shortcut_keys["library.reveal"], self.defaults["library.reveal"][:1] + [free])
        self.assertTrue(self.d._shortcut_items["library.reveal"].font(1).bold())        # changed: in bold
        changed = []
        self.bus.sig_shortcuts_changed.connect(changed.append)
        self.d._save_shortcut_settings()
        with open(shortcuts.USER_FILE, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh), {"library.reveal": self.d._shortcut_keys["library.reveal"]})
        self.assertEqual(changed[0]["library.reveal"], self.d._shortcut_keys["library.reveal"])

    def test_a_key_in_use_taken_only_once_agreed(self):
        owner, key = "library.edit_track", self.defaults["library.edit_track"][0]
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.No) as asked:
            self.press("library.reveal", key)
        self.assertTrue(asked.called)
        self.assertEqual(self.d._shortcut_keys["library.reveal"], self.defaults["library.reveal"])
        self.assertEqual(self.d._shortcut_keys[owner], self.defaults[owner])
        self.assertEqual(self.d.kse_shortcut_main.keySequence().toString(QtGui.QKeySequence.PortableText),
                         self.defaults["library.reveal"][0])                # the field shows the keys kept
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.press("library.reveal", key)
        self.assertEqual(self.d._shortcut_keys["library.reveal"][0], key)
        self.assertNotIn(key, self.d._shortcut_keys[owner])

    def test_a_key_another_program_holds_asked_about(self):
        key = "Ctrl+Alt+Shift+F8"
        with mock.patch.object(os_support, "hotkey_taken", lambda k: k == key), \
                mock.patch.object(QMessageBox, "question", return_value=QMessageBox.No) as asked:
            self.press("library.reveal", key)
        self.assertIn("Another program", asked.call_args.args[2])
        self.assertNotIn(key, self.d._shortcut_keys["library.reveal"])

    def test_revert_and_revert_all(self):
        self.press("library.reveal", "Ctrl+Alt+Shift+F7")
        self.press("library.edit_track", "Ctrl+Alt+Shift+F9")
        self.d.btn_shortcut_revert.click()                    # the selected one: Edit Track
        self.assertEqual(self.d._shortcut_keys["library.edit_track"], self.defaults["library.edit_track"])
        self.assertNotEqual(self.d._shortcut_keys["library.reveal"], self.defaults["library.reveal"])
        self.d.btn_shortcut_revert_all.click()
        self.assertEqual(self.d._shortcut_keys, self.defaults)

    def test_edits_not_kept_dropped_when_shown_again(self):
        self.press("library.reveal", "Ctrl+Alt+Shift+F7")
        self.d._load_shortcut_settings()                       # as at the dialog's next show
        self.assertEqual(self.d._shortcut_keys["library.reveal"], self.defaults["library.reveal"])

    def test_filter_finds_an_action_by_its_key(self):
        key = self.defaults["library.reveal"][0]
        native = QtGui.QKeySequence(key).toString(QtGui.QKeySequence.NativeText)
        self.d.ed_shortcut_filter.setText(native)
        shown = [a for a, item in self.d._shortcut_items.items() if not item.isHidden()]
        self.assertIn("library.reveal", shown)
        self.assertLess(len(shown), 5)


class MidiPageTests(unittest.TestCase):
    def setUp(self):
        with open(midi.USER_FILE, "w", encoding="utf-8") as fh:
            # deck 2's: an action of an earlier version, none now
            json.dump({"mixer.ch1.trim": ["B010"], "deck1.play_pause": ["900B"], "deck2.play_pause": ["910B"]}, fh)
        self.bus = EventBus()
        self.d = SettingsDialog(self.bus)
        self.learning = []

    def tearDown(self):
        self.d.close()
        self.d.deleteLater()
        drain()
        os.remove(midi.USER_FILE)

    def native(self) -> None:
        self.d.set_midi_access(True, lambda: ["DDJ (test)"], self.learning.append)

    def learn(self, action_id: str, *events, batched: bool = False) -> None:
        """Learn ``action_id`` from ``events`` (status, data 1, data 2): each reported on its own, or all in one
        report (``batched``: as the engine's ring hands over the messages that came between two polls)."""
        self.d.tree_midi.setCurrentItem(self.d._midi_items[action_id])
        self.d.btn_midi_learn.setChecked(True)
        reports = [events] if batched else [(e,) for e in events]
        for report in reports:
            self.d._on_midi_events([(s, d1, d2, 0, 0, 0) for s, d1, d2 in report])
        if self.d._midi_learn_timer.isActive():               # a CC waiting for its LSB / its release
            self.d._midi_learn_timer.stop()
            self.d._finish_pending_learn()

    def test_disabled_on_the_python_engine(self):
        for w in (self.d.cb_midi_input, self.d.tree_midi, self.d.btn_midi_learn, self.d.btn_midi_import,
                  self.d.btn_midi_revert_all):
            self.assertFalse(w.isEnabled())
        self.assertIn("native audio engine", self.d.lbl_midi_note.text())
        self.d._toggle_midi_learn(True)
        self.assertIsNone(self.d._midi_learning)

    def test_every_action_shown(self):
        shown = set(self.d._midi_items)
        self.assertEqual(shown, {a.id for a in midi.ACTIONS})
        self.assertIn("mixer.ch1.trim", shown)

    def test_inputs_listed_with_the_one_wanted(self):
        self.native()
        cfg = default_cfg()
        cfg.playbackconfig.midi_input = "Gone (test)"
        self.d.set_config(cfg)
        self.assertEqual([self.d.cb_midi_input.itemData(i) for i in range(self.d.cb_midi_input.count())],
                         ["", "DDJ (test)", "Gone (test)"])
        self.assertEqual(self.d.get_config().playbackconfig.midi_input, "Gone (test)")

    def test_learn_a_fader_a_button_and_a_knobs_direction(self):
        self.native()
        self.learn("mixer.ch1.volume", (0xB0, 0x13, 64), (0xB0, 0x33, 10))      # MSB and LSB: 14 bits
        self.assertEqual(self.d._midi_map["mixer.ch1.volume"], [midi.Message(0xB0, 0x13, hires=True).text()])
        self.assertEqual(self.learning, [True, False])
        self.learn("deck1.cue", (0x90, 0x0C, 127), (0x90, 0x0C, 0))
        self.assertEqual(self.d._midi_map["deck1.cue"], ["900C"])
        self.learn("deck1.tempo_reset", (0xB0, 0x20, 1))                        # a knob's turn on a button
        self.assertEqual(self.d._midi_map["deck1.tempo_reset"], [midi.Message(0xB0, 0x20, direction=1).text()])
        self.learn("deck1.track_start", (0xB0, 0x21, 127), (0xB0, 0x21, 0))     # a CC button: 127, then 0
        self.assertEqual(self.d._midi_map["deck1.track_start"], ["B021"])

    def test_a_14_bit_fader_learned_from_one_report(self):
        self.native()
        self.learn("mixer.ch1.volume", (0xB0, 0x13, 64), (0xB0, 0x33, 10), batched=True)
        self.assertEqual(self.d._midi_map["mixer.ch1.volume"], [midi.Message(0xB0, 0x13, hires=True).text()])

    def test_a_cc_button_learned_from_one_report(self):
        self.native()
        self.learn("deck1.track_start", (0xB0, 0x21, 127), (0xB0, 0x21, 0), batched=True)
        self.assertEqual(self.d._midi_map["deck1.track_start"], ["B021"])

    def test_a_message_in_use_taken_once_agreed(self):
        self.native()
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.No):
            self.learn("deck1.cue", (0x90, 0x0B, 127))
        self.assertEqual(self.d._midi_map["deck1.cue"], [])
        self.assertEqual(self.d._midi_map["deck1.play_pause"], ["900B"])
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
            self.learn("deck1.cue", (0x90, 0x0B, 127))
        self.assertEqual(self.d._midi_map["deck1.cue"], ["900B"])
        self.assertEqual(self.d._midi_map["deck1.play_pause"], [])

    def test_saved_without_the_entries_of_no_action(self):
        self.native()
        self.learn("deck1.cue", (0x90, 0x0C, 127))
        told = []
        self.bus.sig_midi_changed.connect(lambda: told.append(True))
        self.d._save_midi_settings()
        with open(midi.USER_FILE, encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertEqual(saved["mixer.ch1.trim"], ["B010"])
        self.assertEqual(saved["deck1.cue"], ["900C"])
        self.assertNotIn("deck2.play_pause", saved)
        self.assertEqual(told, [True])

    def test_revert_all_reverts_every_action(self):
        self.native()
        self.d.btn_midi_revert_all.click()
        self.assertEqual(self.d._midi_map["deck1.play_pause"], [])
        self.assertEqual(self.d._midi_map["mixer.ch1.trim"], [])                # TRIM shown now, reverted too


if __name__ == "__main__":
    unittest.main()
