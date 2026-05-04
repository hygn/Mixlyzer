import os

from PySide6 import QtCore, QtGui, QtWidgets
import ctypes
import atexit
import json

from app.print_hook import install_print_hook, log_environment_info
from core.event_bus import EventBus
from core.model import DataModel, GlobalParams
from core.timeline import TimelineCoordinator
from core.player import PlayerController
from core.library_handler import LibraryDB
from core.analysis_lib_handler import FeatureNPZStore
from core.taskmanager import taskmanager
from core.external_sync import ExternalSyncController
from core.rekordbox_sync import RekordboxXmlSync
from .metronome import MetronomeController
from utils.window_visibility import is_window_fully_hidden
from utils.volume import slider_percent_to_linear

from ui.pane import MainPane
from ui.cfgwindow import SettingsDialog
from ui.workers import WorkersDialog
from ui.oss_support import SupportDialog
from ui.about_dialog import AboutDialog
from views.base import REGISTRY  # to instantiate default views

from core.analysis_worker import AnalysisWorker
from core.segment_reanalysis_manager import SegmentReanalysisManager
from analyzer_core.global_analyzer import getAlbumArt, extract_tags
from core.config import config, analysisconfig, keyconfig, libconfig, viewconfig, load_cfg

class AppWindow(QtWidgets.QMainWindow):
    _sig_metronome_set_beats = QtCore.Signal(object, float)
    _WINDOW_VISIBILITY_POLL_MS = 33
    _HIDDEN_REFRESH_FPS = 1
    _HIDDEN_ENTER_POLLS = 5

    def __init__(self):
        super().__init__()
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
            track_duration_callback=self._track_duration_in_library,
            track_total_samples_callback=self._track_total_samples_in_library,
            failure_callback=self._on_external_sync_failure,
        )
        self.rekordbox_sync = RekordboxXmlSync(cfg_getter=lambda: self.cfg, parent=self)

        # Main UI pane
        self.pane = MainPane(self.bus, self.model, self.tl, self.cfg)
        self.setCentralWidget(self.pane)
        self.pane.install_default()

        self.cfgwin = SettingsDialog(bus=self.bus)
        self.cfgwin.set_config(self.cfg)
        self.workers = WorkersDialog()
        self.workers.set_bus(self.bus)
        self.support_dlg = SupportDialog(self)
        self.about_dialog = AboutDialog(self)

        # Toolbar
        tb = self.addToolBar("Transport")
        a_cfg = QtGui.QAction("Settings", self);  a_cfg.triggered.connect(self.open_cfg)
        a_workers = QtGui.QAction("Workers", self); a_workers.triggered.connect(self.open_workers)
        a_support_tb = QtGui.QAction("Open Source", self); a_support_tb.triggered.connect(self._open_support)
        a_about = QtGui.QAction("About", self); a_about.triggered.connect(self._open_about)

        self.bus.sig_stop_requested.connect(self.player.stop)
        self.bus.sig_pause_requested.connect(self.player.pause)
        self.bus.sig_play_requested.connect(self.player.play)
        tb.addActions([a_cfg, a_workers, a_support_tb, a_about])


        # events
        self.bus.sig_features_loaded.connect(self._on_features_loaded)
        self.bus.sig_request_load_track.connect(self.analyze_file)
        self.bus.sig_setting_saveJsonRequested.connect(self._on_settings_save)
        self.bus.sig_rekordbox_sync_requested.connect(self.rekordbox_sync.sync_requested)
        self.bus.sig_rekordbox_sync_track_requested.connect(self.rekordbox_sync.sync_track_requested)
        self.bus.sig_reanalyze_requested.connect(self.reanalyze_file)

        self.resize(1280, 860)

        # status
        self.current_path: str | None = None
        self._external_sync_pending_seek: float | None = None
        self._external_sync_applied_enabled: bool | None = None
        self._analysis_workers: dict[int, AnalysisWorker] = {}
        self._analysis_context: dict[int, dict] = {}
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
        self.segment_manager = SegmentReanalysisManager(
            bus=self.bus,
            model=self.model,
            taskmanager=self.taskmanager,
            get_current_path=lambda: self.current_path,
            track_edit_getter=lambda: getattr(self.pane, "track_edit", None),
            status_callback=status_bar.showMessage,
            parent=self,
        )

        self.metro = MetronomeController(
            click_wav_path=self.cfg.playbackconfig.metronome_wav_path,
        )
        self.metro.set_downbeat_cycle(1)
        self.metro.moveToThread(self.player.audio_thread())
        self.player.connect_precise_audio_time(self.metro._on_time_changed)
        self._sig_metronome_set_beats.connect(self.metro.set_beats, QtCore.Qt.QueuedConnection)
        QtCore.QMetaObject.invokeMethod(self.metro, "initialize_audio", QtCore.Qt.QueuedConnection)
        if self.cfg.playbackconfig.enable_metronome:
            QtCore.QMetaObject.invokeMethod(self.metro, "start", QtCore.Qt.QueuedConnection)
        else:
            QtCore.QMetaObject.invokeMethod(self.metro, "stop", QtCore.Qt.QueuedConnection)
        self.bus.sig_beatgrid_edited.connect(self._sync_metronome_beats)
        self._apply_external_sync_mode(self.cfg.externalsyncconfig)
        self._apply_visibility_refresh_setting(force=True)

    # DnD
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent):
        if self._is_external_sync_active():
            self.statusBar().showMessage("External Sync is enabled. Local track loading is disabled.")
            e.ignore()
            return
        try:
            urls = e.mimeData().urls()
            if not urls: return
            path = urls[0].toLocalFile()
            if os.path.isfile(path):
                self.analyze_file(path)
        except Exception as ex:
            self.statusBar().showMessage(f"Drop error: {ex}")

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

    def _handle_worker_error(self, taskid: int, message: str) -> None:
        self.taskmanager.rmtask(taskid, message)
        path = self._analysis_context.get(taskid, {}).get("path")
        basename = os.path.basename(path) if path else ""
        prefix = f"Error ({basename})" if basename else "Error"
        self.statusBar().showMessage(f"{prefix}: {message}")
        print(f"[AnalysisWorker] Error for task {taskid}: {message}")
        self._finalize_analysis(taskid)

    def _handle_worker_success(self, taskid: int, payload: dict, finished_slot) -> None:
        try:
            ctx = self._analysis_context.get(taskid, {})
            payload = dict(payload)
            payload.setdefault("auto_load", ctx.get("auto_load", True))
            finished_slot(payload)
        finally:
            self._finalize_analysis(taskid)

    def _on_worker_progress(self, taskid: int, status: str, progress: float) -> None:
        if taskid not in self._analysis_workers:
            return
        self.taskmanager.updatetask(taskid, status, float(progress))

    def _on_worker_status(self, taskid: int, status: str) -> None:
        if not status:
            return
        path = self._analysis_context.get(taskid, {}).get("path")
        basename = os.path.basename(path) if path else ""
        message = f"{status}: {basename}" if basename else status
        self.statusBar().showMessage(message)

    def _finalize_analysis(self, taskid: int) -> None:
        worker = self._analysis_workers.pop(taskid, None)
        if worker is not None:
            worker.stop()
            worker.deleteLater()
        self._analysis_context.pop(taskid, None)

    def analyze_file(self, path: str):
        if self._is_external_sync_active():
            self.statusBar().showMessage("External Sync is enabled. Local track loading is disabled.")
            return
        self._start_analysis(path, force_analyze=False, finished_slot=self._on_features_ready)

    def reanalyze_file(self, path: str):
        self._start_analysis(path, force_analyze=True, finished_slot=self._on_features_reanalyze)

    def _find_inflight_analysis_task(self, path: str) -> int | None:
        norm_path = os.path.normcase(os.path.normpath(path))
        for taskid, ctx in self._analysis_context.items():
            ctx_path = str(ctx.get("path") or "").strip()
            if not ctx_path:
                continue
            if os.path.normcase(os.path.normpath(ctx_path)) == norm_path:
                return taskid
        return None

    def _start_analysis(self, path: str, *, force_analyze: bool, finished_slot):
        inflight_taskid = self._find_inflight_analysis_task(path)
        if inflight_taskid is not None:
            basename = os.path.basename(path)
            self.statusBar().showMessage(f"Analysis already in progress: {basename}")
            return

        self.segment_manager.cancel_all("Segment reanalysis canceled (track changed)")
        self.statusBar().showMessage(f"Analyzing: {os.path.basename(path)}")

        self.current_path = path

        cfg = load_cfg()
        thumb = getAlbumArt(path)
        title, artist, album, _comment = extract_tags(path)
        task_info = self.taskmanager.addtask(
            songname=title,
            thumbnail=thumb,
            status="Loading Track",
            progress=0.0,
        )
        taskid = task_info.taskid

        l = LibraryDB(os.path.join(cfg.libconfig.libpath, f"library.db"))
        l.connect()
        track = l.get(path)
        auto_load = track is not None
        if (track != None) and track.uid and (not force_analyze):
            try:
                f = FeatureNPZStore(base_dir=cfg.libconfig.libpath, compressed=True)
                feat = f.load(track.uid)
                l.close()
                metadata = track.to_meta()
                # Cached library load: set album art immediately.
                self._set_album_art(thumb)
                features_properties = {
                    "features": feat,
                    "properties": metadata,
                    "update_db": False,
                    "taskid": taskid,
                    "auto_load": True,
                }
                self._on_features_ready(features_properties)
                return
            except (FileNotFoundError, ValueError):
                pass
        l.close()

        worker = AnalysisWorker(path, cfg, taskid, force_analyze=force_analyze, parent=self)
        self._analysis_workers[taskid] = worker
        self._analysis_context[taskid] = {
            "path": path,
            "force": force_analyze,
            "auto_load": auto_load,
        }

        worker.progress.connect(lambda status, progress, tid=taskid: self._on_worker_progress(tid, status, progress))
        worker.status.connect(lambda status, tid=taskid: self._on_worker_status(tid, status))
        worker.error.connect(lambda msg, tid=taskid: self._handle_worker_error(tid, msg))
        worker.finished.connect(lambda payload, tid=taskid: self._handle_worker_success(tid, payload, finished_slot))
        worker.start()

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
        gp = GlobalParams(
            analysis_samp_rate=int(feat_std.get("sr")),
            bpm_hop_length=int(feat_std.get("bpm_hop")),
            chroma_hop_length=int(feat_std.get("chroma_hop")),
        )
        duration_sec = float(feat_std.get("duration_sec"))
        if update_db:
            self._emit_library_update(cfg, lib_rows)
            self._sync_rekordbox_track(properties.get("uid"))

        if auto_load:
            self._apply_model_update(
                feat_std,
                properties,
                gp,
                duration_sec,
                stop_playback=True,
                refresh_source=True,
            )
            if self._external_sync_pending_seek is not None:
                self.player.seek(float(self._external_sync_pending_seek))
                self._external_sync_pending_seek = None

        self.taskmanager.updatetask(feat["taskid"], "Finished", 1)
        self.taskmanager.rmtask(feat["taskid"])
        if auto_load:
            self.statusBar().showMessage("Done. Press Play.")
        else:
            self.statusBar().showMessage("Analysis saved to library.")

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
        gp = GlobalParams(
            analysis_samp_rate=int(feat_std.get("sr")),
            bpm_hop_length=int(feat_std.get("bpm_hop")),
            chroma_hop_length=int(feat_std.get("chroma_hop")),
        )
        duration_sec = float(feat_std.get("duration_sec"))
        if update_db:
            self._emit_library_update(cfg, lib_rows)
            self._sync_rekordbox_track(properties.get("uid"))
            if is_current_track:
                self._apply_model_update(
                    feat_std,
                    properties,
                    gp,
                    duration_sec,
                    stop_playback=False,
                    refresh_source=False,
                )
        else:
            if is_current_track:
                self._apply_model_update(
                    feat_std,
                    properties,
                    gp,
                    duration_sec,
                    stop_playback=False,
                    refresh_source=False,
                )
        self.taskmanager.rmtask(feat["taskid"])
        self.statusBar().showMessage("Done. Press Play.")

    def _apply_model_update(
        self,
        features: dict,
        properties: dict,
        gp: GlobalParams,
        duration_sec: float,
        *,
        stop_playback: bool,
        refresh_source: bool,
    ) -> None:
        """Update the shared DataModel and notify listeners."""
        self.model.load(features, properties, gp, duration_sec)
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
        if self.model.features:
            beats = self.model.features.get("beats_time_sec")
        self._sig_metronome_set_beats.emit(beats, float(self.tl.current_time))

    def _track_exists_in_library(self, path: str) -> bool:
        cfg = load_cfg()
        db = LibraryDB(os.path.join(cfg.libconfig.libpath, "library.db"))
        db.connect()
        try:
            return db.get(path) is not None
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
        action = "loading" if exists_in_library else "analyzing"
        self.statusBar().showMessage(f"External Sync {action}: {os.path.basename(path)}")
        self._external_sync_pending_seek = float(time_sec)
        self._start_analysis(path, force_analyze=False, finished_slot=self._on_features_ready)

    def _on_external_sync_failure(self, message: str) -> None:
        failed_cfg = self.cfg.to_dict()
        failed_cfg["externalsyncconfig"]["enabled"] = False
        new_cfg = config.from_dict(failed_cfg)
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(new_cfg.to_dict(), f)
        self.cfg = new_cfg
        self.cfgwin.set_config(new_cfg)
        self.external_sync.set_config(new_cfg.externalsyncconfig)
        self._apply_external_sync_mode(new_cfg.externalsyncconfig)
        QtWidgets.QMessageBox.warning(
            self,
            "Memory Sync Disabled",
            message,
        )

    def _on_settings_save(self, _config: config):
        with open("config.json", "r") as f:
            prev_cfg = json.load(f)
        with open("config.json", "w") as f:
            json.dump(_config.to_dict(), f)
        self.cfg = _config
        self.cfgwin.set_config(_config)
        self.external_sync.set_config(_config.externalsyncconfig)
        self.external_sync.set_poll_fps(_config.viewconfig.fps)
        self._apply_external_sync_mode(_config.externalsyncconfig)
        prev_lcfg = prev_cfg.get("libconfig", {})
        new_lcfg = _config.to_dict().get("libconfig", {})
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
        prev_vcfg = prev_cfg.get("viewconfig", {})
        new_vcfg = _config.to_dict()["viewconfig"]
        prev_pcfg = prev_cfg.get("playbackconfig", {})
        new_pcfg = _config.to_dict()["playbackconfig"]
        _VIEW_LAYOUT_KEYS = {"display_waveform", "display_beatgrid", "display_keystrip", "display_JumpCUE"}
        layout_changed = any(prev_vcfg.get(k) != new_vcfg.get(k) for k in _VIEW_LAYOUT_KEYS)
        viewconfig_changed = prev_vcfg != new_vcfg
        playbackconfig_changed = prev_pcfg != new_pcfg
        if viewconfig_changed:
            self._apply_visibility_refresh_setting(force=True)
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

    def _default_volume_linear_from_cfg(self, cfg: config) -> float:
        playback_cfg = getattr(cfg, "playbackconfig", None)
        trim_dbfs = float(getattr(playback_cfg, "volume_trim_dbfs", -6.0))
        default_percent = int(getattr(playback_cfg, "default_volume_percent", 100) or 100)
        return slider_percent_to_linear(default_percent, trim_dbfs)

    def _apply_external_sync_mode(self, sync_cfg) -> None:
        enabled = bool(sync_cfg.enabled)
        state_changed = (self._external_sync_applied_enabled is None) or (self._external_sync_applied_enabled != enabled)
        self._external_sync_applied_enabled = enabled
        if state_changed:
            self.bus.sig_external_sync_enabled.emit(enabled)
        if enabled:
            if state_changed:
                self.player.pause()
            mode_desc = "Time Sync" if str(sync_cfg.mode) == "time" else "Sample Index Sync"
            target_desc = (
                f"{mode_desc}, process={sync_cfg.memory_process_name or 'manual'}, "
                f"deck1.path={sync_cfg.memory_deck1.path.offsets}, "
                f"deck2.path={sync_cfg.memory_deck2.path.offsets}"
            )
            self.statusBar().showMessage(
                f"External Sync enabled via {target_desc}. "
                f"Local play/load is locked."
            )
