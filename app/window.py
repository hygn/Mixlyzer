import os

from PySide6 import QtCore, QtGui, QtWidgets
import copy

from app.audio_report import print_audio_setup
from app.print_hook import install_print_hook, log_environment_info
from audioengine.native.client import (
    ACT_DECK_SHIFT, ACT_JUMP_A, EQ_MODES, FADER_CURVES, MIDI_EVENT_GUI, MIDI_EVENT_HOTCUE_SET, MIDI_EVENT_RELATIVE,
)
from core import midi, os_support, shortcuts
from core.event_bus import EventBus
from core.model import DataModel
from core.timeline import TimelineCoordinator
from core.player import PlayerController
from core.library_handler import LibraryDB
from core.analysis_lib_handler import FeatureNPZStore
from analyzer_core.global_analyzer import extract_tags
from core.workflows.analysis import AnalysisWorkflow
from core.taskmanager import taskmanager
from core.external_sync import ExternalSyncController
from core.workflows.rekordbox_sync import RekordboxXmlSync
from core.workflows.reading_refresh import LibraryReadingRefresh
from core.workflows.music_index import LibraryMusicIndex
from core.workflows.tieup_web import TieupWebLookup
from .metronome import MetronomeController
from utils.window_visibility import is_window_fully_hidden
from utils.volume import slider_percent_to_linear, trim_dbfs_to_linear, trim_knob_to_db
from utils.phrases import set_phrase_label_names

from ui.pane import MainPane
from ui.deck_controls import block, fill, row_title
from ui.deck_switches import DeckSwitches, MasterLevels
from ui.midi_toggle import MidiToggle
from ui.config.window import SettingsDialog
from ui.workers import WorkersDialog
from ui.oss_support import SupportDialog
from ui.about_dialog import AboutDialog
from ui.shortcuts import ShortcutManager

# a knob's steps acted on from one message at most (a fast turn)
MIDI_STEPS_MAX = 16

from core.workflows.segment_reanalysis import SegmentReanalysisWorkflow
from core.config import config, load_cfg, save_cfg, save_deck, search_settings
from core.beat_geometry import downbeat_beat_indices
from core.i18n import tr
from core.resource_paths import resource_path

class AppWindow(QtWidgets.QMainWindow):
    _sig_metronome_set_beats = QtCore.Signal(object, object, float)
    _WINDOW_VISIBILITY_POLL_MS = 33
    _HIDDEN_REFRESH_FPS = 1
    _HIDDEN_ENTER_POLLS = 5

    def __init__(self, progress=None):
        """``progress(fraction, part)``: how far the window is built (the startup splash)."""
        super().__init__()
        step = progress or (lambda _fraction, _part: None)
        step(0.0, tr("Core objects and playback"))
        os_support.prepare_app_process("org.hygn.mixylzer")
        self.setWindowTitle("Mixlyzer")
        self.setAcceptDrops(True)
        self.setWindowIcon(QtGui.QIcon(str(resource_path("assets/images/mixlyzer.png"))))

        self.cfg = load_cfg()
        set_phrase_label_names(self.cfg.viewconfig.phrase_labels.by_label())
        if self.cfg.libconfig.write_log:
            install_print_hook(log_path=self.cfg.libconfig.logpath)
            log_environment_info(log_path=self.cfg.libconfig.logpath)

        # Core objects
        self.bus = EventBus()
        self.model = DataModel()
        self.tl = TimelineCoordinator(self.bus)
        self.player = PlayerController(
            self.bus,
            model=self.model,
            volume_position=max(0, min(100, int(self.cfg.playbackconfig.default_volume_percent or 100))) / 100.0,
            volume_trim_dbfs=float(self.cfg.deckconfig.volume_trim_dbfs),
            tempo_range=int(self.cfg.deckconfig.tempo_range),
            default_peak_meter_gain_linear=self._default_peak_meter_gain_linear_from_cfg(
                self.cfg
            ),
            engine=self.cfg.playbackconfig.audio_engine,
            native_buffer_ms=float(self.cfg.playbackconfig.native_buffer_ms),
            python_buffer_ms=float(self.cfg.playbackconfig.python_buffer_ms),
            output_device=self.cfg.playbackconfig.output_device,
            native_api=self.cfg.playbackconfig.native_audio_api,
            native_exclusive=self.cfg.playbackconfig.native_exclusive,
            native_exclusive_rate=self.cfg.playbackconfig.native_exclusive_rate,
        )
        self.player.set_refresh_fps(self.cfg.viewconfig.fps)
        self.external_sync = ExternalSyncController(
            self.bus,
            self.model,
            self.cfg.externalsyncconfig,
            poll_fps=self.cfg.viewconfig.fps,
            status_callback=self.statusBar().showMessage,
            load_track_callback=self._load_track_from_external_sync,
            seek_callback=self.player.seek,
            track_exists_callback=self._track_exists_in_library,
            library_path_callback=self._library_path_of_file,
            track_duration_callback=self._track_duration_in_library,
            track_total_samples_callback=self._track_total_samples_in_library,
            failure_callback=self._on_external_sync_failure,
        )
        self.rekordbox_sync = RekordboxXmlSync(cfg_getter=lambda: self.cfg, parent=self)
        self.rekordbox_sync.started.connect(self.bus.sig_rekordbox_sync_started)
        self.rekordbox_sync.progress.connect(self.bus.sig_rekordbox_sync_progress)
        self.rekordbox_sync.finished.connect(self.bus.sig_rekordbox_sync_finished)
        self.rekordbox_sync.failed.connect(self.bus.sig_rekordbox_sync_failed)
        self.reading_refresh = LibraryReadingRefresh(cfg_getter=lambda: self.cfg, parent=self)
        self.reading_refresh.started.connect(self.bus.sig_reading_refresh_started)
        self.reading_refresh.progress.connect(self.bus.sig_reading_refresh_progress)
        self.reading_refresh.finished.connect(self.bus.sig_reading_refresh_finished)
        self.reading_refresh.failed.connect(self.bus.sig_reading_refresh_failed)
        # the library's search takes the new readings
        self.reading_refresh.finished.connect(lambda _request, _result: self._emit_library_update(self.cfg, None))
        self.music_index = LibraryMusicIndex(cfg_getter=lambda: self.cfg, parent=self)
        self.tieup_web = TieupWebLookup(libpath=lambda: self.cfg.libconfig.libpath,
                                        settings=lambda: self.cfg.searchconfig,
                                        enabled=lambda: bool(self.cfg.libconfig.tieup_web_search), parent=self)
        self.tieup_web.tagged.connect(self._on_tieups_tagged)
        self.music_index.started.connect(self.bus.sig_music_index_started)
        self.music_index.progress.connect(self.bus.sig_music_index_progress)
        self.music_index.finished.connect(self.bus.sig_music_index_finished)
        self.music_index.failed.connect(self.bus.sig_music_index_failed)

        # Main UI pane
        step(0.03, tr("Main view, editor and library"))
        self.pane = MainPane(self.bus, self.model, self.tl, self.cfg)
        self.pane.track_info.set_eq_available(self.player.native)   # the Python engine has no mixer
        self.setCentralWidget(self.pane)
        self.pane.install_default()

        step(0.3, tr("Settings"))
        self.cfgwin = SettingsDialog(bus=self.bus)
        # the engine that plays (its buffer shown first) and its output devices
        self.cfgwin.set_audio_engine(self.player.native, self.player.output_devices, self.player.output_channels)
        # a MIDI controller: the native engine's alone
        self.cfgwin.set_midi_access(self.player.native, self.player.midi_ports, self.player.set_midi_learn)
        self.cfgwin.set_external_sync_status(self.external_sync.snapshot)
        self.cfgwin.set_config(self.cfg)
        step(0.6, tr("Dialogs, workflows and metronome"))
        self.workers = WorkersDialog()
        self.workers.set_bus(self.bus)
        self.support_dlg = SupportDialog(self)
        self.about_dialog = AboutDialog(self)

        # the menus' bar at the top (thin); under it one toolbar row in the signal's order: CH 1 (the controller,
        # the deck, its channel), then MASTER (the master's and the headphones' levels)
        tb = self.addToolBar(tr("Mixer"))
        tb.setMovable(False)
        tb.setStyleSheet("QToolBar { padding: 0; spacing: 0; border: none; }")
        tb.layout().setContentsMargins(0, 0, 0, 0)
        rows = QtWidgets.QWidget()
        rows.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        rows_layout = QtWidgets.QVBoxLayout(rows)
        rows_layout.setContentsMargins(0, 2, 0, 2)
        rows_layout.setSpacing(2)
        a_cfg = QtGui.QAction(tr("Settings"), self);  a_cfg.triggered.connect(self.open_cfg)
        a_workers = QtGui.QAction(tr("Workers"), self); a_workers.triggered.connect(self.open_workers)
        a_works = QtGui.QAction(tr("Tie-up Works"), self); a_works.triggered.connect(lambda: self.pane.lib.open_works())
        a_support_tb = QtGui.QAction(tr("Open Source"), self); a_support_tb.triggered.connect(self._open_support)
        a_about = QtGui.QAction(tr("About"), self); a_about.triggered.connect(self._open_about)

        self.bus.sig_stop_requested.connect(self.player.stop)
        self.bus.sig_pause_requested.connect(self.player.pause)
        self.bus.sig_play_requested.connect(self.player.play)
        self.bus.sig_timestretch_toggle_requested.connect(self._on_timestretch_toggle)
        self.bus.sig_metronome_toggle_requested.connect(self._on_metronome_toggle)
        menus = self.menuBar()
        menus.setNativeMenuBar(False)
        font = menus.font()   # a thin bar
        font.setPointSize(9)
        menus.setFont(font)
        menus.setStyleSheet("QMenuBar { padding: 0; } QMenuBar::item { padding: 2px 10px; }")
        for action in (a_cfg, a_works, a_workers, a_support_tb, a_about):
            menus.addAction(action)

        def toolbar_part(lay: QtWidgets.QHBoxLayout, title: str, widgets: list) -> None:
            """A part of the row: its title, then its blocks, widened to the window's right edge with the others
            (deck_controls.fill)."""
            lay.addWidget(row_title(title), 0, QtCore.Qt.AlignVCenter)
            lay.addSpacing(2)
            fill(lay, widgets)

        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(6, 0, 8, 0)
        row_layout.setSpacing(6)
        # CH 1: the MIDI controller's input on / off; the deck's and its channel's switches (jog mode, EQ, fader
        # curve, Sound Color FX, the headphones CUE)
        self.midi_toggle = MidiToggle()
        self.midi_toggle.setEnabled(self.player.native)
        self.midi_toggle.toggled_by_hand.connect(self._toggle_midi)
        self.bus.sig_deck_state.connect(self._show_midi)
        self.deck_switches = DeckSwitches(self.bus)
        self.deck_switches.set_available(self.player.native)
        self.bus.sig_deck_state.connect(self.deck_switches.set_report)
        toolbar_part(row_layout, "CH 1", [block("CONTROLLER", [self.midi_toggle]), self.deck_switches])
        # MASTER: the master's and the headphones' levels
        self.master_levels = MasterLevels(self.bus)
        self.master_levels.set_available(self.player.native)
        self.bus.sig_deck_state.connect(self.master_levels.set_report)
        row_layout.addSpacing(4)
        toolbar_part(row_layout, "MASTER", [self.master_levels])
        rows_layout.addWidget(row)
        tb.addWidget(rows)

        # keyboard shortcuts (assets/shortcuts; Settings > Shortcuts)
        handlers = {**self.pane.shortcut_handlers(), **self.deck_switches.shortcut_handlers(),
                    "playback.restart_engine": self._restart_engine,
                    "playback.midi_input": self._toggle_midi}
        self.bus.sig_engine_restarted.connect(self._on_engine_restarted)
        self.shortcuts = ShortcutManager(self, handlers, self.statusBar().showMessage)
        self.shortcuts.apply(shortcuts.keymap())
        self.bus.sig_shortcuts_changed.connect(self.shortcuts.apply)
        # a MIDI controller (assets/midi; Settings > MIDI): the native engine does the deck's and the mixer's actions
        # itself; the others it passes on (sig_midi_events) are run here as the shortcuts they name
        self._midi_handlers = handlers
        self.bus.sig_midi_events.connect(self._on_midi_events)
        self.bus.sig_midi_changed.connect(self._apply_midi)
        self.bus.sig_deck_state.connect(self._keep_tempo_range)
        self.bus.sig_deck_state.connect(self._keep_trim)
        self.bus.sig_deck_state.connect(self._keep_switches)
        # what the deck keeps (deck.json: the trim, the tempo range, the switches, the levels, the effects picked),
        # written a second after it last changed (a knob turned reports every frame)
        self._deck_save = QtCore.QTimer(self)
        self._deck_save.setSingleShot(True)
        self._deck_save.setInterval(1000)
        self._deck_save.timeout.connect(lambda: save_deck(self.cfg))
        self._apply_midi()


        # events
        self.bus.sig_features_loaded.connect(self._on_features_loaded)
        self.bus.sig_request_load_track.connect(self.load_track)
        self.bus.sig_request_analyze.connect(self.analyze_file)
        self.bus.sig_duration_changed.connect(self._on_duration)
        self.bus.sig_unload_track_requested.connect(self.unload_track)
        self.bus.sig_setting_saveJsonRequested.connect(self._on_settings_save)
        self.bus.sig_rekordbox_sync_requested.connect(self.rekordbox_sync.sync_requested)
        self.bus.sig_rekordbox_sync_track_requested.connect(self.rekordbox_sync.sync_track_requested)
        self.bus.sig_reading_refresh_requested.connect(self.reading_refresh.refresh_requested)
        self.bus.sig_music_index_requested.connect(self.music_index.index_requested)
        self.bus.sig_reanalyze_requested.connect(self.reanalyze_file)

        self.resize(1600, 900)  # minimum target: FHD

        # status
        self._deck_path: str | None = None   # the file the deck holds
        self._external_sync_applied_enabled: bool | None = None
        self._effective_refresh_fps: int | None = None
        self._hidden_visibility_streak = 0
        self._visibility_refresh_enabled = False
        self._visibility_refresh_failed = False
        self._visibility_refresh_timer = QtCore.QTimer(self)
        self._visibility_refresh_timer.setInterval(self._WINDOW_VISIBILITY_POLL_MS)
        self._visibility_refresh_timer.timeout.connect(self._update_refresh_fps_for_visibility)

        # task manager
        self.taskmanager = taskmanager(self.bus)

        status_bar = self.statusBar()
        self.segment_reanalysis_workflow = SegmentReanalysisWorkflow(
            bus=self.bus,
            model=self.model,
            taskmanager=self.taskmanager,
            get_current_path=lambda: self._deck_path,
            track_edit_getter=lambda: getattr(self.pane, "track_edit", None),
            status_callback=status_bar.showMessage,
            parent=self,
        )
        self.analysis_workflow = AnalysisWorkflow(
            taskmanager=self.taskmanager,
            result_callback=self._on_features_ready,
            reanalysis_result_callback=self._on_features_reanalyze,
            status_callback=status_bar.showMessage,
            missing_file_callback=self._on_track_file_missing,
            parent=self,
        )

        self.metro = MetronomeController(
            click_wav_path=self.cfg.deckconfig.metronome_wav_path,
        )
        self.metro.set_downbeat_cycle(1)
        self.metro.set_offset(float(self.cfg.deckconfig.metronome_offset_msec))
        self._apply_click_settings(self.cfg)
        self.metro.moveToThread(self.player.audio_thread())
        self.metro.bind_mixer(self.player.click_mixer)
        self._sig_metronome_set_beats.connect(self.metro.set_beats, QtCore.Qt.QueuedConnection)
        QtCore.QMetaObject.invokeMethod(self.metro, "initialize_audio", QtCore.Qt.QueuedConnection)
        if self.cfg.deckconfig.enable_metronome:
            QtCore.QMetaObject.invokeMethod(self.metro, "start", QtCore.Qt.QueuedConnection)
        else:
            QtCore.QMetaObject.invokeMethod(self.metro, "stop", QtCore.Qt.QueuedConnection)
        self.bus.sig_beatgrid_edited.connect(self._sync_metronome_beats)
        self._apply_timestretch(bool(self.cfg.deckconfig.use_timestretch))
        print_audio_setup(self.cfg, self.player.native)
        # The scratch reader's and cursor poller's numba code, loaded here on the main
        # thread once the window is up (~1 s from the disk cache), so the audio thread
        # never waits for it (already loaded when the app started with app.startup).
        if not self.player.native:
            from audioengine.python.cursor_poller import warm_cursor_poller
            from audioengine.python.dsp import warm_scratch_numba
            QtCore.QTimer.singleShot(0, lambda: (warm_scratch_numba(), warm_cursor_poller()))
        self._apply_external_sync_mode(self.cfg.externalsyncconfig)
        self._apply_visibility_refresh_setting(force=True)
        step(1.0, "")

    # DnD: a file dropped on the deck (the area above the library) loads there, shown by its drop zone while
    # dragged; the library's tracks take a file from elsewhere themselves (analyzed, not loaded)
    @staticmethod
    def _dropped_file(e) -> str | None:
        for url in e.mimeData().urls():
            if url.isLocalFile() and os.path.isfile(url.toLocalFile()):
                return url.toLocalFile()
        return None

    def _over_deck(self, e) -> bool:
        return self.pane.drop_zone_at(self.pane.mapFrom(self, e.position().toPoint()))

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        self.dragMoveEvent(e)

    def dragMoveEvent(self, e: QtGui.QDragMoveEvent):
        if self._dropped_file(e) is not None and self._over_deck(e):
            self.pane.show_drop_zone(True)
            e.acceptProposedAction()
        else:
            self.pane.hide_drop_zone()
            e.ignore()

    def dragLeaveEvent(self, e: QtGui.QDragLeaveEvent):
        self.pane.hide_drop_zone()

    def dropEvent(self, e: QtGui.QDropEvent):
        self.pane.hide_drop_zone()
        path = self._dropped_file(e)
        if path is None or not self._over_deck(e):
            e.ignore()
            return
        e.acceptProposedAction()
        self.load_track(path)

    def _apply_midi(self) -> None:
        """The MIDI input, the mapping and the jog's resolution to the native engine."""
        pcfg = self.cfg.playbackconfig
        self.player.midi_configure(pcfg.midi_input, midi.bindings(midi.mapping()), int(pcfg.midi_jog_ticks),
                                   bool(self.cfg.deckconfig.midi_enabled))
        self._show_midi()

    def _toggle_midi(self) -> None:
        """The MIDI input switched on or off (the toolbar's button, its shortcut), kept in deck.json."""
        if not self.player.native:
            self.statusBar().showMessage(tr("MIDI needs the native audio engine (Settings > Audio > Engine)."))
            return
        deck = self.cfg.deckconfig
        deck.midi_enabled = not deck.midi_enabled
        self.player.set_midi_enabled(deck.midi_enabled)
        self._deck_save.start()
        self._show_midi()
        self.statusBar().showMessage(tr("MIDI input on.") if deck.midi_enabled else tr("MIDI input off."), 3000)

    def _show_midi(self, report=None) -> None:
        """The toolbar's MIDI button: the setting and, from the deck's report, the port's state."""
        if report is not None:
            self._midi_state = int(report.midi_state)
        self.midi_toggle.show_state(bool(self.cfg.deckconfig.midi_enabled),
                                    str(self.cfg.playbackconfig.midi_input or ""), getattr(self, "_midi_state", 0))

    def _on_midi_events(self, events) -> None:
        """The controller's messages passed on to the GUI: a JumpCUE pad on an empty slot (SET), a knob's turn
        (its steps, each a shortcut run), a button pressed (its shortcut run once)."""
        for _status, _d1, d2, flags, code, steps in events:
            if flags & MIDI_EVENT_HOTCUE_SET:
                letter = midi.JUMPCUE_PADS[(int(code) & 0xFFFFFF) - ACT_JUMP_A]
                if int(code) >> ACT_DECK_SHIFT == 0:   # the app's deck: deck 1
                    self._run_midi_handler(f"cue.set.{letter}")
                continue
            if not flags & MIDI_EVENT_GUI:
                continue
            action = midi.gui_action(code)
            if action is None:
                continue
            if action.kind == midi.KNOB:   # clockwise: its first shortcut, counterclockwise: its second
                handler_id = action.gui[0] if steps > 0 else action.gui[-1]
                for _ in range(min(abs(int(steps)), MIDI_STEPS_MAX)):
                    self._run_midi_handler(handler_id)
            elif flags & MIDI_EVENT_RELATIVE:   # a button bound to a knob's one direction: a step a press
                for _ in range(min(abs(int(steps)), MIDI_STEPS_MAX)):
                    self._run_midi_handler(action.gui[0])
            elif d2 > 0:   # pressed (let go: nothing)
                self._run_midi_handler(action.gui[0])

    def _run_midi_handler(self, handler_id: str) -> None:
        handler = self._midi_handlers.get(handler_id)
        message = handler() if handler is not None else None
        if message:
            self.statusBar().showMessage(message, 3000)

    def _keep_tempo_range(self, report) -> None:
        """The tempo range as the deck has it (its button, a key, a controller), kept for the next start."""
        tempo_range = int(report.tempo_range)
        if tempo_range != int(self.cfg.deckconfig.tempo_range):
            self.cfg.deckconfig.tempo_range = tempo_range
            self._deck_save.start()

    def _restart_engine(self) -> None:
        """The native audio engine started again by hand (the shortcut; Settings > Audio > Engine has its button)."""
        if not self.player.native:
            self.statusBar().showMessage(tr("The Python audio engine plays: there is no native engine to restart."))
            return
        self.statusBar().showMessage(tr("Restarting the audio engine..."))
        self.bus.sig_engine_restart_requested.emit()

    def _on_engine_restarted(self, ok: bool) -> None:
        self.statusBar().showMessage(tr("Audio engine restarted.") if ok else
                                     tr("The audio engine did not start; it is tried again every 2 seconds."), 5000)

    def _keep_trim(self, report) -> None:
        """The trim as the deck has it (its TRIM knob, a controller), kept in deck.json: the peak meter's gain follows
        it at once, the file a second after it last moved."""
        trim = round(trim_knob_to_db(report.knobs[0]), 2)
        if abs(trim - float(self.cfg.deckconfig.volume_trim_dbfs)) < 0.005:
            return
        self.cfg.deckconfig.volume_trim_dbfs = trim
        self.pane.apply_playback_config(self.cfg)
        self._deck_save.start()

    def _keep_switches(self, report) -> None:
        """The jog's mode, the EQ, the fader curve, the levels and the effects picked as the deck has them (the
        toolbar, a controller), kept in deck.json (the native engine's: the Python engine has none of them)."""
        if not self.player.native or not report.mix:
            return
        deck = self.cfg.deckconfig
        v = report.mix_value
        kept = {
            "mixer_eq": EQ_MODES[min(1, max(0, int(v("eq_mode"))))],
            "fader_curve": FADER_CURVES[min(2, max(0, int(v("fader_curve"))))],
            "jog_mode": "vinyl" if report.vinyl else "cdj",
            "master_level": round(v("master_level", deck.master_level), 3),
            "headphones_level": round(v("hp_a_level", deck.headphones_level), 3),
            "headphones_cue_mix": int(round(100.0 * v("hp_a_mix", deck.headphones_cue_mix / 100.0))),
            "color_type": int(v("color_type", deck.color_type)),
            "color_param": round(v("color_param", deck.color_param), 3),
        }
        changed = {k: value for k, value in kept.items() if getattr(deck, k) != value}
        if changed:
            for k, value in changed.items():
                setattr(deck, k, value)
            self._deck_save.start()

    def open_cfg(self):
        self.cfgwin.show()
        self.cfgwin.raise_()
        self.cfgwin.activateWindow()

    def open_workers(self):
        # Ensure current tasks are reflected; caller may choose to resend snapshot if needed
        self.workers.show()
        self.workers.raise_()
        self.workers.activateWindow()

    def _load_features_from_store(self, uid: str | None, cfg: config) -> dict | None:
        if not uid:
            return None
        store = FeatureNPZStore(base_dir=cfg.libconfig.libpath, compressed=True)
        try:
            return store.load(uid)
        except (FileNotFoundError, ValueError):
            return None

    def load_track(self, path: str) -> None:
        """The file onto the deck: it plays at once (decoded), its analysis comes when done (a library track's
        stored one at once); the analysis's results alone are applied then, the playback goes on."""
        if self._is_external_sync_active():
            self.statusBar().showMessage(tr("External Sync is enabled. Local track loading is disabled."))
            return
        self._load(path)

    def _load(self, path: str) -> None:
        if not os.path.isfile(path) and self.analysis_workflow.library_track(path) is None:
            return
        track = self.analysis_workflow.library_track(path)
        if track is not None and not os.path.isfile(track.path):
            # a library track whose file is gone: the analysis asks for its new place (loaded here then); the
            # deck keeps its track meanwhile
            self.analysis_workflow.analyze(track.path, listener=self._on_deck_analysis)
            return
        self.segment_reanalysis_workflow.cancel_all(tr("Segment reanalysis canceled (track changed)"))
        self._deck_path = track.path if track is not None else path
        if track is None:
            # a new file: its tags meanwhile, the analysis's results when it is done
            title, artist, album, _comment = extract_tags(path)
            self.model.load({}, {"path": path, "title": title, "artist": artist, "album": album}, 0.0)
            self._emit_loaded()
        self.player.stop()
        self.player.set_source(self._deck_path)
        self.analysis_workflow.analyze(self._deck_path, listener=self._on_deck_analysis)

    def analyze_file(self, path: str) -> None:
        """The file into the library (analyzed), not loaded."""
        self.analysis_workflow.analyze(path)

    def _holds(self, path: str | None) -> bool:
        """The deck's track is the file ``path``."""
        if not path or not self._deck_path:
            return False
        return os.path.normcase(os.path.normpath(path)) == os.path.normcase(os.path.normpath(self._deck_path))

    def _on_deck_analysis(self, result: dict) -> None:
        """A track's analysis (stored or new) for the deck: applied when the deck still holds that track."""
        if self._holds(dict(result.get("properties") or {}).get("path")):
            self._apply_analysis(result)

    def _apply_analysis(self, result: dict) -> None:
        """The analysis's features (the stored ones: the library's standard form) and properties into the model,
        the decoded audio kept (the playback goes on)."""
        properties = dict(result.get("properties") or {})
        features = dict(result.get("features") or {})
        stored = self._load_features_from_store(properties.get("uid"), load_cfg())
        if stored is not None:
            features = stored
        duration = float(features.get("duration_sec") or self.model.duration_sec or 0.0)
        self.model.load(features, properties, duration, keep_audio=True)
        album_art = result.get("album_art")
        if album_art is not None:
            self.model.set_album_art(album_art)
        self._emit_loaded()

    def _emit_loaded(self) -> None:
        self.bus.sig_features_loaded.emit()
        self.bus.sig_properties_loaded.emit()
        self.bus.sig_albumart_loaded.emit()

    def _on_duration(self, duration_sec: float) -> None:
        """The decoded track's length: the views' until the analysis gives its own."""
        if duration_sec > 0.0 and float(self.model.duration_sec or 0.0) <= 0.0 and self._deck_path:
            self.model.duration_sec = float(duration_sec)
            self.model.sig_updated.emit()
            self.bus.sig_properties_loaded.emit()

    def unload_track(self) -> None:
        """Empty the deck: no track in the player or the views."""
        if self._is_external_sync_active():
            self.statusBar().showMessage(tr("External Sync is enabled. Local track loading is disabled."))
            return
        self.player.unload()
        self._deck_path = None
        self.model.load({}, {}, 0.0)
        self._emit_loaded()
        self.statusBar().showMessage(tr("Track unloaded."))

    def reanalyze_file(self, path: str, analysis: dict | None = None):
        """``analysis``: analysis settings for this reanalysis only (the saved ones stay)."""
        self.analysis_workflow.reanalyze(path, analysis)

    def _on_track_file_missing(self, track, force_analyze: bool, analysis: dict | None = None,
                               listener=None) -> None:
        """A library track whose file is gone: say so and let the user set its new path (then the analysis
        asked for, with its settings for this run, or the deck's load that asked)."""
        title = tr("Track File Not Found")
        name = track.title or os.path.basename(track.path)
        answer = QtWidgets.QMessageBox.question(
            self, title,
            tr('Cannot load "{name}": its file is missing.\n{path}\n\nSet the file\'s new location?').format(
                name=name, path=track.path),
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        folder = os.path.dirname(track.path)
        new_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("Locate {name}").format(name=name), folder if os.path.isdir(folder) else "")
        if not new_path:
            return
        new_path = os.path.normpath(new_path)
        cfg = load_cfg()
        db = LibraryDB(os.path.join(cfg.libconfig.libpath, "library.db"))
        db.connect()
        try:
            db.set_path(track.uid, new_path)
            lib_rows = db.list_all()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, title, str(exc))
            return
        finally:
            db.close()
        self._emit_library_update(cfg, lib_rows)
        self._sync_rekordbox_track(track.uid)
        if force_analyze:
            self.reanalyze_file(new_path, analysis)
        elif listener is not None:
            self._load(new_path)
        else:
            self.analyze_file(new_path)

    @QtCore.Slot(dict)
    def _on_features_ready(self, feat: dict):
        """A track's analysis (a new one, or a library track's stored one): the library; the deck takes it itself
        when it asked (its listener)."""
        properties = dict(feat["properties"])
        if feat.get("update_db", False):
            self._emit_library_update(load_cfg(), feat.get("library"))
            self._sync_rekordbox_track(properties.get("uid"))
            self.statusBar().showMessage(tr("Analysis saved to library."))
            if feat.get("new_track") and not feat.get("tieups"):   # a new track its versions gave none: the web
                self.tieup_web.new_track(str(properties.get("uid") or ""))
        self.taskmanager.updatetask(feat["taskid"], tr("Finished"), 1)
        self.taskmanager.rmtask(feat["taskid"])

    def _on_features_reanalyze(self, feat: dict):
        """A full reanalysis: the library, and the deck when it holds the track (its playback goes on)."""
        properties = dict(feat["properties"])
        if feat.get("update_db", False):
            self._emit_library_update(load_cfg(), feat.get("library"))
            self._sync_rekordbox_track(properties.get("uid"))
        if self._holds(properties.get("path")):
            self._apply_analysis(feat)
        self.taskmanager.rmtask(feat["taskid"])
        self.statusBar().showMessage(tr("Analysis saved to library."))

    def _on_tieups_tagged(self, uids: list) -> None:
        """Tracks given a tie-up from the web: the library, the Rekordbox XML's comments."""
        self._emit_library_update(load_cfg(), None)
        for uid in uids:
            self._sync_rekordbox_track(uid)

    def _emit_library_update(self, cfg: config, lib_rows):
        if lib_rows is None:
            l = LibraryDB(os.path.join(cfg.libconfig.libpath, "library.db"))
            l.connect()
            lib_rows = l.list_all()
            l.close()
        self.bus.sig_lib_updated.emit(lib_rows)

    def _sync_rekordbox_track(self, uid: object) -> None:
        track_uid = str(uid or "").strip()
        if track_uid:
            self.bus.sig_rekordbox_sync_track_requested.emit(track_uid)

    def _on_features_loaded(self):
        self.bus.sig_window_changed.emit(self.tl.window_sec)
        self.bus.sig_center_changed.emit(self.tl.center_t)
        self._sync_metronome_beats()

    def _sync_metronome_beats(self):
        beats = None
        downbeats = None
        if self.model.features:
            beats = self.model.features.get("beats_time_sec")
            if beats is not None:
                downbeats = downbeat_beat_indices(
                    beats,
                    self.model.features.get("tempo_segments"),
                )
        self._sig_metronome_set_beats.emit(beats, downbeats, float(self.tl.current_time))

    def _track_exists_in_library(self, path: str) -> bool:
        cfg = load_cfg()
        db = LibraryDB(os.path.join(cfg.libconfig.libpath, "library.db"))
        db.connect()
        try:
            return db.get(path) is not None
        finally:
            db.close()

    def _library_path_of_file(self, path: str) -> str | None:
        """The path the library stores the file ``path`` points to under, if it is in it."""
        cfg = load_cfg()
        db = LibraryDB(os.path.join(cfg.libconfig.libpath, "library.db"))
        db.connect()
        try:
            row = db.find_file(path)
            return row.path if row else None
        finally:
            db.close()

    def _track_duration_in_library(self, path: str) -> float | None:
        cfg = load_cfg()
        db = LibraryDB(os.path.join(cfg.libconfig.libpath, "library.db"))
        db.connect()
        try:
            row = db.get(path)
            return float(row.duration) if row and row.duration is not None else None
        finally:
            db.close()

    def _track_total_samples_in_library(self, path: str) -> int | None:
        cfg = load_cfg()
        db = LibraryDB(os.path.join(cfg.libconfig.libpath, "library.db"))
        db.connect()
        try:
            row = db.get(path)
            return int(row.total_samples) if row and row.total_samples is not None else None
        finally:
            db.close()

    def _load_track_from_external_sync(self, path: str, time_sec: float) -> None:
        exists_in_library = self._track_exists_in_library(path)
        message = tr("External Sync loading: {file}") if exists_in_library else tr("External Sync analyzing: {file}")
        self.statusBar().showMessage(message.format(file=os.path.basename(path)))
        self._load(path)
        self.player.seek(float(time_sec))   # replayed once the track is decoded

    def _on_external_sync_failure(self, message: str) -> None:
        new_cfg = copy.deepcopy(self.cfg)
        new_cfg.externalsyncconfig.enabled = False
        save_cfg(new_cfg)
        self.cfg = new_cfg
        self.cfgwin.set_config(new_cfg)
        self.external_sync.set_config(new_cfg.externalsyncconfig)
        self._apply_external_sync_mode(new_cfg.externalsyncconfig)
        QtWidgets.QMessageBox.warning(
            self,
            tr("External Sync Disabled"),
            message,
        )

    def _on_settings_save(self, _config: config):
        prev_cfg = self.cfg.to_dict()
        prev_deck = self.cfg.deck_dict()
        save_cfg(_config)
        save_deck(_config)
        os.makedirs(_config.libconfig.libpath or ".", exist_ok=True)
        self.cfg = _config
        self.cfgwin.set_config(_config)
        self.external_sync.set_config(_config.externalsyncconfig)
        self.external_sync.set_poll_fps(_config.viewconfig.fps)
        self._apply_external_sync_mode(_config.externalsyncconfig)
        prev_lcfg = prev_cfg.get("libconfig", {})
        new_lcfg = _config.to_dict().get("libconfig", {})
        if any(
            prev_lcfg.get(key) != new_lcfg.get(key)
            for key in ("libpath", "local_music_directories")
        ):
            self.pane.lib.set_config(_config)
        rekordbox_cfg_changed = any(
            prev_lcfg.get(k) != new_lcfg.get(k)
            for k in ("rekordbox_sync_enabled", "rekordbox_xml_path", "rekordbox_ramp_tolerance_ms",
                      "rekordbox_comment_tieups")
        )
        if (
            rekordbox_cfg_changed
            and _config.libconfig.rekordbox_sync_enabled
            and _config.libconfig.rekordbox_xml_path.strip()
        ):
            self.bus.sig_rekordbox_sync_requested.emit(True)
        # search: the query's settings at once; the stored readings' (languages, Japanese text's Han
        # runs): the library read again
        prev_search = search_settings(prev_cfg.get("searchconfig"))
        if prev_search != _config.searchconfig:
            self.pane.lib.set_search_config(_config)
        if prev_search.stored_readings() != _config.searchconfig.stored_readings():
            self.bus.sig_reading_refresh_requested.emit(None)
        prev_vcfg = prev_cfg.get("viewconfig", {})
        new_vcfg = _config.to_dict()["viewconfig"]
        prev_pcfg = prev_cfg.get("playbackconfig", {})
        new_pcfg = _config.to_dict()["playbackconfig"]
        _VIEW_LAYOUT_KEYS = {"display_waveform", "display_beatgrid", "display_keystrip", "display_JumpCUE", "display_phrase"}
        layout_changed = any(prev_vcfg.get(k) != new_vcfg.get(k) for k in _VIEW_LAYOUT_KEYS)
        viewconfig_changed = prev_vcfg != new_vcfg
        playbackconfig_changed = prev_pcfg != new_pcfg or prev_deck != _config.deck_dict()
        peak_meter_config_changed = (
            prev_vcfg.get("use_output_volume_as_peak_meter_input", True)
            != new_vcfg.get("use_output_volume_as_peak_meter_input", True)
        )
        if viewconfig_changed:
            self._apply_visibility_refresh_setting(force=True)
        _WAVE_STYLE_KEYS = ("waveform_style", "mainview_wave_frame_ms_short", "mainview_wave_frame_ms_long",
                            "mainview_wave_filter_order")
        # the waveform's low / mid / high ranges: the overview and the preview sum them from the stored bands
        bands_changed = any(
            [float(x) for x in (prev_vcfg.get(k) or ())] != [float(x) for x in new_vcfg[k]]
            for k in ("env_lo", "env_mid", "env_hi")
        )
        if bands_changed or any(prev_vcfg.get(k) != new_vcfg[k] for k in _WAVE_STYLE_KEYS):
            self.pane.ov.refresh_waveform()
            if not layout_changed:
                self.bus.sig_reload_UI.emit(_config)  # the waveform view takes the style when installed
        if prev_vcfg.get("phrase_labels") != new_vcfg["phrase_labels"]:
            set_phrase_label_names(_config.viewconfig.phrase_labels.by_label())
            track_edit = getattr(self.pane, "track_edit", None)
            if track_edit is not None:
                track_edit.refresh_phrase_label_names()
            self.bus.sig_phrase_segments_updated.emit()
        if playbackconfig_changed:
            if _config.deckconfig.enable_metronome:
                QtCore.QMetaObject.invokeMethod(self.metro, "start", QtCore.Qt.QueuedConnection)
            else:
                QtCore.QMetaObject.invokeMethod(self.metro, "stop", QtCore.Qt.QueuedConnection)
            QtCore.QMetaObject.invokeMethod(
                self.metro,
                "set_soundfile",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, _config.deckconfig.metronome_wav_path),
            )
            QtCore.QMetaObject.invokeMethod(
                self.metro,
                "set_offset",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(float, float(_config.deckconfig.metronome_offset_msec)),
            )
            self._apply_timestretch(bool(_config.deckconfig.use_timestretch))
            self._apply_click_settings(_config)
        if playbackconfig_changed or peak_meter_config_changed:
            self.pane.apply_playback_config(_config)
        if layout_changed:
            self.bus.sig_reload_UI.emit(_config)
        elif prev_cfg.get("externalsyncconfig") != _config.to_dict().get("externalsyncconfig"):
            self.bus.sig_reload_UI.emit(_config)
        if any(prev_pcfg.get(k) != new_pcfg.get(k) for k in ("midi_input", "midi_jog_ticks")):
            self._apply_midi()

    def _open_support(self):
        # Non-modal dialog; reuse instance
        self.support_dlg.show()
        self.support_dlg.raise_()
        self.support_dlg.activateWindow()

    def _open_about(self):
        self.about_dialog.show()
        self.about_dialog.raise_()
        self.about_dialog.activateWindow()

    def _is_external_sync_active(self) -> bool:
        return bool(self.cfg.externalsyncconfig.enabled)

    def is_fully_hidden_from_user(self) -> bool:
        return bool(is_window_fully_hidden(self))

    def _initialize_visibility_refresh(self) -> bool:
        try:
            self.is_fully_hidden_from_user()
        except Exception as exc:
            self._visibility_refresh_enabled = False
            self._visibility_refresh_failed = True
            print(f"[WindowVisibility] disabled: {exc}")
            return False
        self._visibility_refresh_enabled = True
        self._visibility_refresh_failed = False
        self._update_refresh_fps_for_visibility(force=True)
        return True

    def _apply_visibility_refresh_setting(self, *, force: bool = False) -> None:
        enabled_by_cfg = bool(getattr(self.cfg.viewconfig, "reduce_fps_when_occluded", True))
        if not enabled_by_cfg:
            self._visibility_refresh_timer.stop()
            self._visibility_refresh_enabled = False
            self._visibility_refresh_failed = False
            self._hidden_visibility_streak = 0
            self._apply_effective_refresh_fps(int(self.cfg.viewconfig.fps), force=True)
            return
        if self._visibility_refresh_enabled:
            self._update_refresh_fps_for_visibility(force=force)
            if not self._visibility_refresh_timer.isActive():
                self._visibility_refresh_timer.start()
            return
        if self._initialize_visibility_refresh():
            if not self._visibility_refresh_timer.isActive():
                self._visibility_refresh_timer.start()
        else:
            self._apply_effective_refresh_fps(int(self.cfg.viewconfig.fps), force=True)

    def _apply_effective_refresh_fps(self, fps: int, *, force: bool = False) -> None:
        fps = max(1, min(240, int(fps)))
        if (not force) and self._effective_refresh_fps == fps:
            return
        self._effective_refresh_fps = fps
        self.player.set_refresh_fps(fps)

    @QtCore.Slot()
    def _update_refresh_fps_for_visibility(self, *, force: bool = False) -> None:
        if not self._visibility_refresh_enabled:
            return
        target_fps = int(self.cfg.viewconfig.fps)
        try:
            is_hidden = self.is_fully_hidden_from_user()
        except Exception as exc:
            if not self._visibility_refresh_failed:
                print(f"[WindowVisibility] disabled during polling: {exc}")
            self._visibility_refresh_failed = True
            self._visibility_refresh_enabled = False
            self._visibility_refresh_timer.stop()
            self._hidden_visibility_streak = 0
            self._apply_effective_refresh_fps(int(self.cfg.viewconfig.fps), force=True)
            return
        if is_hidden:
            self._hidden_visibility_streak += 1
        else:
            self._hidden_visibility_streak = 0
        if self._hidden_visibility_streak >= self._HIDDEN_ENTER_POLLS:
            target_fps = self._HIDDEN_REFRESH_FPS
        self._apply_effective_refresh_fps(target_fps, force=force)

    def _apply_click_settings(self, cfg: config) -> None:
        """The metronome's clicks (volumes / pitches, ducking), the output's soft clip and routing, and the deck as
        deck.json keeps it (trim, switches, levels, the effects picked)."""
        pcfg, deck = cfg.playbackconfig, cfg.deckconfig
        QtCore.QMetaObject.invokeMethod(
            self.metro,
            "set_click_levels",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(float, float(deck.metronome_downbeat_volume_percent) / 100.0),
            QtCore.Q_ARG(float, float(deck.metronome_downbeat_pitch_semitones)),
            QtCore.Q_ARG(float, float(deck.metronome_beat_volume_percent) / 100.0),
            QtCore.Q_ARG(float, float(deck.metronome_beat_pitch_semitones)),
        )
        QtCore.QMetaObject.invokeMethod(
            self.metro, "set_ducking", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(bool, bool(deck.metronome_ducking)),
        )
        self.player.set_soft_clip(bool(pcfg.soft_clip))
        self.player.set_volume_trim(float(deck.volume_trim_dbfs))
        # the switches as last left
        eq = EQ_MODES.index(deck.mixer_eq) if deck.mixer_eq in EQ_MODES else 0
        curve = FADER_CURVES.index(deck.fader_curve) if deck.fader_curve in FADER_CURVES else 0
        self.bus.sig_mixer_requested.emit("eq_mode", float(eq))
        self.bus.sig_mixer_requested.emit("fader_curve", float(curve))
        self.player.set_jog_mode(deck.jog_mode != "cdj")
        self.player.set_routing(pcfg.output_main, pcfg.output_headphones, pcfg.output_booth)
        self.player.set_headphones(bool(pcfg.headphones_master_mix), int(deck.headphones_cue_mix) / 100.0,
                                   float(deck.headphones_level))
        # the levels, the booth's EQ and the effects picked as last left (the effects themselves off)
        for key, value in (("master_level", deck.master_level), ("booth_level", pcfg.booth_level),
                           ("booth_eq_hi", pcfg.booth_eq_hi), ("booth_eq_low", pcfg.booth_eq_low),
                           ("hp_a_split", 1.0 if pcfg.headphones_split else 0.0),
                           ("color_type", deck.color_type), ("color_param", deck.color_param)):
            self.bus.sig_mixer_requested.emit(key, float(value))
        self.player.set_buffer_ms(float(pcfg.native_buffer_ms), float(pcfg.python_buffer_ms))
        self.player.set_output_device(pcfg.output_device)
        self.player.set_output_mode(pcfg.native_audio_api, bool(pcfg.native_exclusive))
        self.player.set_exclusive_rate(int(pcfg.native_exclusive_rate))
        self.player.set_process(pcfg.native_priority, bool(pcfg.native_power_throttling))

    def _on_timestretch_toggle(self, enabled: bool) -> None:
        """The deck's MT button (the time stretch's one place), kept in deck.json."""
        if bool(self.cfg.deckconfig.use_timestretch) == bool(enabled):
            return
        new_cfg = copy.deepcopy(self.cfg)
        new_cfg.deckconfig.use_timestretch = bool(enabled)
        self._on_settings_save(new_cfg)

    def _on_metronome_toggle(self, enabled: bool) -> None:
        """The editor's metronome button (the metronome's on / off's one place), kept in deck.json."""
        if bool(self.cfg.deckconfig.enable_metronome) == bool(enabled):
            return
        new_cfg = copy.deepcopy(self.cfg)
        new_cfg.deckconfig.enable_metronome = bool(enabled)
        self._on_settings_save(new_cfg)

    def _apply_timestretch(self, enabled: bool) -> None:
        if enabled and not self.player.native:
            # Load the stretcher's numba kernels here, on the main thread, so the audio
            # thread never blocks on loading/compiling them (~0.5 s from the disk cache).
            from audioengine.python.timestretch import warm_timestretch_numba
            warm_timestretch_numba()
        self.player.set_timestretch(enabled)
    def _default_volume_linear_from_cfg(self, cfg: config) -> float:
        trim_dbfs = float(cfg.deckconfig.volume_trim_dbfs)
        default_percent = int(cfg.playbackconfig.default_volume_percent or 100)
        return slider_percent_to_linear(default_percent, trim_dbfs)

    def _default_peak_meter_gain_linear_from_cfg(self, cfg: config) -> float:
        trim_dbfs = float(cfg.deckconfig.volume_trim_dbfs)
        use_output_volume = bool(
            getattr(
                getattr(cfg, "viewconfig", None),
                "use_output_volume_as_peak_meter_input",
                True,
            )
        )
        if use_output_volume:
            return self._default_volume_linear_from_cfg(cfg)
        return trim_dbfs_to_linear(trim_dbfs)

    def _apply_external_sync_mode(self, sync_cfg) -> None:
        enabled = bool(sync_cfg.enabled)
        state_changed = (self._external_sync_applied_enabled is None) or (self._external_sync_applied_enabled != enabled)
        self._external_sync_applied_enabled = enabled
        if state_changed:
            self.bus.sig_external_sync_enabled.emit(enabled)
        if enabled:
            if state_changed:
                self.player.pause()
            address = self.external_sync.listening_on()
            if address:
                self.statusBar().showMessage(
                    tr("External Sync enabled: OSC on {address}. Local play/load is locked.").format(address=address))
