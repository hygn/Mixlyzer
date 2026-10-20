import os

from PySide6 import QtCore, QtGui, QtWidgets
import ctypes
import atexit
import copy
import json

from app.print_hook import install_print_hook, log_environment_info
from core.event_bus import EventBus
from core.model import DataModel
from core.timeline import TimelineCoordinator
from core.player import PlayerController
from core.library_handler import LibraryDB
from core.analysis_lib_handler import FeatureNPZStore
from core.workflows.analysis import AnalysisWorkflow
from core.taskmanager import taskmanager
from core.external_sync import ExternalSyncController
from core.workflows.rekordbox_sync import RekordboxXmlSync
from core.workflows.reading_refresh import LibraryReadingRefresh
from .metronome import MetronomeController
from utils.window_visibility import is_window_fully_hidden
from utils.volume import slider_percent_to_linear, trim_dbfs_to_linear
from utils.atomic_io import atomic_write_json
from utils.phrases import set_phrase_label_names

from ui.pane import MainPane
from ui.config.window import SettingsDialog
from ui.workers import WorkersDialog
from ui.oss_support import SupportDialog
from ui.about_dialog import AboutDialog

from core.workflows.segment_reanalysis import SegmentReanalysisWorkflow
from core.config import config, load_cfg, search_settings
from core.beat_geometry import downbeat_beat_indices
from core.i18n import tr

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
        appid = "org.hygn.mixylzer"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
            atexit.register(lambda: ctypes.windll.winmm.timeEndPeriod(1))
        except Exception:
            pass
        self.setWindowTitle("Mixlyzer")
        self.setAcceptDrops(True)
        self.setWindowIcon(QtGui.QIcon("assets/images/mixlyzer.png"))

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
            default_volume_linear=self._default_volume_linear_from_cfg(self.cfg),
            default_peak_meter_gain_linear=self._default_peak_meter_gain_linear_from_cfg(
                self.cfg
            ),
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

        # Main UI pane
        step(0.03, tr("Main view, editor and library"))
        self.pane = MainPane(self.bus, self.model, self.tl, self.cfg)
        self.setCentralWidget(self.pane)
        self.pane.install_default()

        step(0.3, tr("Settings"))
        self.cfgwin = SettingsDialog(bus=self.bus)
        self.cfgwin.set_config(self.cfg)
        step(0.6, tr("Dialogs, workflows and metronome"))
        self.workers = WorkersDialog()
        self.workers.set_bus(self.bus)
        self.support_dlg = SupportDialog(self)
        self.about_dialog = AboutDialog(self)

        # Toolbar
        tb = self.addToolBar(tr("Transport"))
        tb.setMovable(False)
        a_cfg = QtGui.QAction(tr("Settings"), self);  a_cfg.triggered.connect(self.open_cfg)
        a_workers = QtGui.QAction(tr("Workers"), self); a_workers.triggered.connect(self.open_workers)
        a_support_tb = QtGui.QAction(tr("Open Source"), self); a_support_tb.triggered.connect(self._open_support)
        a_about = QtGui.QAction(tr("About"), self); a_about.triggered.connect(self._open_about)

        self.bus.sig_stop_requested.connect(self.player.stop)
        self.bus.sig_pause_requested.connect(self.player.pause)
        self.bus.sig_play_requested.connect(self.player.play)
        self.bus.sig_timestretch_toggle_requested.connect(self._on_timestretch_toggle)
        self.bus.sig_metronome_toggle_requested.connect(self._on_metronome_toggle)
        tb.addActions([a_cfg, a_workers, a_support_tb, a_about])


        # events
        self.bus.sig_features_loaded.connect(self._on_features_loaded)
        self.bus.sig_request_load_track.connect(self.analyze_file)
        self.bus.sig_unload_track_requested.connect(self.unload_track)
        self.bus.sig_setting_saveJsonRequested.connect(self._on_settings_save)
        self.bus.sig_rekordbox_sync_requested.connect(self.rekordbox_sync.sync_requested)
        self.bus.sig_rekordbox_sync_track_requested.connect(self.rekordbox_sync.sync_track_requested)
        self.bus.sig_reading_refresh_requested.connect(self.reading_refresh.refresh_requested)
        self.bus.sig_reanalyze_requested.connect(self.reanalyze_file)

        self.resize(1600, 900)  # minimum target: FHD

        # status
        self._external_sync_pending_seek: float | None = None
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
            get_current_path=lambda: self.analysis_workflow.current_path,
            track_edit_getter=lambda: getattr(self.pane, "track_edit", None),
            status_callback=status_bar.showMessage,
            parent=self,
        )
        self.analysis_workflow = AnalysisWorkflow(
            taskmanager=self.taskmanager,
            result_callback=self._on_features_ready,
            reanalysis_result_callback=self._on_features_reanalyze,
            album_art_callback=self._set_album_art,
            cancel_reanalysis_callback=self.segment_reanalysis_workflow.cancel_all,
            status_callback=status_bar.showMessage,
            missing_file_callback=self._on_track_file_missing,
            parent=self,
        )

        self.metro = MetronomeController(
            click_wav_path=self.cfg.playbackconfig.metronome_wav_path,
        )
        self.metro.set_downbeat_cycle(1)
        self.metro.set_offset(float(getattr(self.cfg.playbackconfig, "metronome_offset_msec", 0.0)))
        self._apply_click_settings(self.cfg.playbackconfig)
        self.metro.moveToThread(self.player.audio_thread())
        self.metro.bind_mixer(self.player.click_mixer)
        self._sig_metronome_set_beats.connect(self.metro.set_beats, QtCore.Qt.QueuedConnection)
        QtCore.QMetaObject.invokeMethod(self.metro, "initialize_audio", QtCore.Qt.QueuedConnection)
        if self.cfg.playbackconfig.enable_metronome:
            QtCore.QMetaObject.invokeMethod(self.metro, "start", QtCore.Qt.QueuedConnection)
        else:
            QtCore.QMetaObject.invokeMethod(self.metro, "stop", QtCore.Qt.QueuedConnection)
        self.bus.sig_beatgrid_edited.connect(self._sync_metronome_beats)
        self._apply_timestretch(bool(self.cfg.playbackconfig.use_timestretch))
        # The scratch reader's and cursor poller's numba code, loaded here on the main
        # thread once the window is up (~1 s from the disk cache), so the audio thread
        # never waits for it (already loaded when the app started with app.startup).
        from core.audio.cursor_poller import warm_cursor_poller
        from core.audio.dsp import warm_scratch_numba
        QtCore.QTimer.singleShot(0, lambda: (warm_scratch_numba(), warm_cursor_poller()))
        self._apply_external_sync_mode(self.cfg.externalsyncconfig)
        self._apply_visibility_refresh_setting(force=True)
        step(1.0, "")

    # DnD
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dragMoveEvent(self, e: QtGui.QDragMoveEvent):
        # A track dragged from the library loads when dropped on the deck, not
        # back on the library.
        over = self.childAt(e.position().toPoint())
        lib = getattr(self.pane, "lib", None)
        if lib is not None and over is not None and (over is lib or lib.isAncestorOf(over)):
            e.ignore()
        elif e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent):
        if self._is_external_sync_active():
            self.statusBar().showMessage(tr("External Sync is enabled. Local track loading is disabled."))
            e.ignore()
            return
        try:
            urls = e.mimeData().urls()
            if not urls: return
            path = urls[0].toLocalFile()
            if os.path.isfile(path):
                self.analyze_file(path)
        except Exception as ex:
            self.statusBar().showMessage(tr("Drop error: {error}").format(error=ex))

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

    def analyze_file(self, path: str):
        if self._is_external_sync_active():
            self.statusBar().showMessage(tr("External Sync is enabled. Local track loading is disabled."))
            return
        self.analysis_workflow.analyze(path)

    def unload_track(self) -> None:
        """Empty the deck: no track in the player or the views."""
        if self._is_external_sync_active():
            self.statusBar().showMessage(tr("External Sync is enabled. Local track loading is disabled."))
            return
        self.player.unload()
        self.analysis_workflow.current_path = None
        self.model.load({}, {}, 0.0)
        self.bus.sig_features_loaded.emit()
        self.bus.sig_properties_loaded.emit()
        self.bus.sig_albumart_loaded.emit()
        self.statusBar().showMessage(tr("Track unloaded."))

    def reanalyze_file(self, path: str):
        self.analysis_workflow.reanalyze(path)

    def _on_track_file_missing(self, track, force_analyze: bool) -> None:
        """A library track whose file is gone: say so and let the user set its new path."""
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
            self.reanalyze_file(new_path)
        else:
            self.analyze_file(new_path)

    @QtCore.Slot(dict)
    def _on_features_ready(self, feat: dict):
        raw_features = dict(feat["features"])
        properties = dict(feat["properties"])
        update_db = feat.get("update_db", False)
        auto_load = bool(feat.get("auto_load", True))
        cfg = load_cfg()
        feat_std = self._load_features_from_store(properties.get("uid"), cfg)
        if feat_std is None:
            feat_std = raw_features
        lib_rows = feat.get("library")
        duration_sec = float(feat_std.get("duration_sec"))
        if update_db:
            self._emit_library_update(cfg, lib_rows)
            self._sync_rekordbox_track(properties.get("uid"))

        if auto_load:
            self._apply_model_update(
                feat_std,
                properties,
                duration_sec,
                stop_playback=True,
                refresh_source=True,
            )
            if self._external_sync_pending_seek is not None:
                self.player.seek(float(self._external_sync_pending_seek))
                self._external_sync_pending_seek = None

        self.taskmanager.updatetask(feat["taskid"], tr("Finished"), 1)
        self.taskmanager.rmtask(feat["taskid"])
        if auto_load:
            self.statusBar().showMessage(tr("Done. Press Play."))
        else:
            self.statusBar().showMessage(tr("Analysis saved to library."))

    def _on_features_reanalyze(self, feat:dict):
        raw_features = dict(feat["features"])
        properties = dict(feat["properties"])
        update_db = feat.get("update_db", False)
        is_current_track = self.player.get_source() == properties.get("path")
        cfg = load_cfg()
        feat_std = self._load_features_from_store(properties.get("uid"), cfg)
        if feat_std is None:
            feat_std = raw_features
        lib_rows = feat.get("library")
        duration_sec = float(feat_std.get("duration_sec"))
        if update_db:
            self._emit_library_update(cfg, lib_rows)
            self._sync_rekordbox_track(properties.get("uid"))
            if is_current_track:
                self._apply_model_update(
                    feat_std,
                    properties,
                    duration_sec,
                    stop_playback=False,
                    refresh_source=False,
                )
        else:
            if is_current_track:
                self._apply_model_update(
                    feat_std,
                    properties,
                    duration_sec,
                    stop_playback=False,
                    refresh_source=False,
                )
        self.taskmanager.rmtask(feat["taskid"])
        self.statusBar().showMessage(tr("Done. Press Play."))

    def _apply_model_update(
        self,
        features: dict,
        properties: dict,
        duration_sec: float,
        *,
        stop_playback: bool,
        refresh_source: bool,
    ) -> None:
        """Update the shared DataModel and notify listeners."""
        self.model.load(features, properties, duration_sec)
        if stop_playback:
            self.player.stop()
        if refresh_source:
            self.player.set_source(properties.get("path"))
        self.bus.sig_features_loaded.emit()
        self.bus.sig_properties_loaded.emit()
        if self.model.album_art:
            self.bus.sig_albumart_loaded.emit()

    def _set_album_art(self, album_art: QtGui.QImage | None) -> None:
        self.model.set_album_art(album_art)
        self.bus.sig_albumart_loaded.emit()

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
        self._external_sync_pending_seek = float(time_sec)
        self.analysis_workflow.analyze(path)

    def _on_external_sync_failure(self, message: str) -> None:
        failed_cfg = self.cfg.to_dict()
        failed_cfg["externalsyncconfig"]["enabled"] = False
        new_cfg = config.from_dict(failed_cfg)
        atomic_write_json("config.json", new_cfg.to_dict(), ensure_ascii=True, indent=None)
        self.cfg = new_cfg
        self.cfgwin.set_config(new_cfg)
        self.external_sync.set_config(new_cfg.externalsyncconfig)
        self._apply_external_sync_mode(new_cfg.externalsyncconfig)
        QtWidgets.QMessageBox.warning(
            self,
            tr("Memory Sync Disabled"),
            message,
        )

    def _on_settings_save(self, _config: config):
        with open("config.json", "r") as f:
            prev_cfg = json.load(f)
        atomic_write_json("config.json", _config.to_dict(), ensure_ascii=True, indent=None)
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
            for k in ("rekordbox_sync_enabled", "rekordbox_xml_path")
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
        playbackconfig_changed = prev_pcfg != new_pcfg
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
            if _config.playbackconfig.enable_metronome:
                QtCore.QMetaObject.invokeMethod(self.metro, "start", QtCore.Qt.QueuedConnection)
            else:
                QtCore.QMetaObject.invokeMethod(self.metro, "stop", QtCore.Qt.QueuedConnection)
            QtCore.QMetaObject.invokeMethod(
                self.metro,
                "set_soundfile",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(str, _config.playbackconfig.metronome_wav_path),
            )
            QtCore.QMetaObject.invokeMethod(
                self.metro,
                "set_offset",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(float, float(_config.playbackconfig.metronome_offset_msec)),
            )
            self._apply_timestretch(bool(_config.playbackconfig.use_timestretch))
            self._apply_click_settings(_config.playbackconfig)
        if playbackconfig_changed or peak_meter_config_changed:
            self.pane.apply_playback_config(_config)
        if layout_changed:
            self.bus.sig_reload_UI.emit(_config)
        elif prev_cfg.get("externalsyncconfig") != _config.to_dict().get("externalsyncconfig"):
            self.bus.sig_reload_UI.emit(_config)

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

    def _apply_click_settings(self, pcfg) -> None:
        """Metronome click volumes / pitches and ducking, and the output soft clip."""
        QtCore.QMetaObject.invokeMethod(
            self.metro,
            "set_click_levels",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(float, float(pcfg.metronome_downbeat_volume_percent) / 100.0),
            QtCore.Q_ARG(float, float(pcfg.metronome_downbeat_pitch_semitones)),
            QtCore.Q_ARG(float, float(pcfg.metronome_beat_volume_percent) / 100.0),
            QtCore.Q_ARG(float, float(pcfg.metronome_beat_pitch_semitones)),
        )
        QtCore.QMetaObject.invokeMethod(
            self.metro, "set_ducking", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(bool, bool(pcfg.metronome_ducking)),
        )
        self.player.set_soft_clip(bool(pcfg.soft_clip))

    def _on_timestretch_toggle(self, enabled: bool) -> None:
        """Deck MT button: same as switching "Use timestretch" in the settings and saving."""
        if bool(self.cfg.playbackconfig.use_timestretch) == bool(enabled):
            return
        new_cfg = copy.deepcopy(self.cfg)
        new_cfg.playbackconfig.use_timestretch = bool(enabled)
        self._on_settings_save(new_cfg)

    def _on_metronome_toggle(self, enabled: bool) -> None:
        """Editor metronome button: same as switching "Enable metronome" in the settings and saving."""
        if bool(self.cfg.playbackconfig.enable_metronome) == bool(enabled):
            return
        new_cfg = copy.deepcopy(self.cfg)
        new_cfg.playbackconfig.enable_metronome = bool(enabled)
        self._on_settings_save(new_cfg)

    def _apply_timestretch(self, enabled: bool) -> None:
        if enabled:
            # Load the stretcher's numba kernels here, on the main thread, so the audio
            # thread never blocks on loading/compiling them (~0.5 s from the disk cache).
            from core.audio.timestretch import warm_timestretch_numba
            warm_timestretch_numba()
        self.player.set_timestretch(enabled)
    def _default_volume_linear_from_cfg(self, cfg: config) -> float:
        playback_cfg = getattr(cfg, "playbackconfig", None)
        trim_dbfs = float(getattr(playback_cfg, "volume_trim_dbfs", -6.0))
        default_percent = int(getattr(playback_cfg, "default_volume_percent", 100) or 100)
        return slider_percent_to_linear(default_percent, trim_dbfs)

    def _default_peak_meter_gain_linear_from_cfg(self, cfg: config) -> float:
        playback_cfg = getattr(cfg, "playbackconfig", None)
        trim_dbfs = float(getattr(playback_cfg, "volume_trim_dbfs", -6.0))
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
            mode_desc = tr("Time Sync") if str(sync_cfg.mode) == "time" else tr("Sample Index Sync")
            target_desc = (
                f"{mode_desc}, process={sync_cfg.memory_process_name or tr('manual')}, "
                f"deck1.path={sync_cfg.memory_deck1.path.offsets}, "
                f"deck2.path={sync_cfg.memory_deck2.path.offsets}"
            )
            self.statusBar().showMessage(
                tr("External Sync enabled via {target}. Local play/load is locked.").format(target=target_desc)
            )
