import os

from PySide6 import QtCore, QtGui, QtWidgets
import ctypes
import atexit
import json

from core.event_bus import EventBus
from core.model import DataModel, GlobalParams
from core.timeline import TimelineCoordinator
from core.player import PlayerController
from core.library_handler import LibraryDB
from core.analysis_lib_handler import FeatureNPZStore
from core.taskmanager import taskmanager
from .metronome import MetronomeController

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

        # Core objects
        self.bus = EventBus()
        self.model = DataModel()
        self.tl = TimelineCoordinator(self.bus)
        self.player = PlayerController(self.bus)

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
        self.bus.sig_reanalyze_requested.connect(self.reanalyze_file)

        self.resize(1280, 860)

        # status
        self.current_path: str | None = None
        self._analysis_workers: dict[int, AnalysisWorker] = {}
        self._analysis_context: dict[int, dict] = {}

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

        self.metro = MetronomeController(bus=self.bus, tl=self.tl, model=self.model, click_wav_path=self.cfg.viewconfig.metronome_wav_path)
        self.metro.set_downbeat_cycle(1)
        if self.cfg.viewconfig.enable_metronome: self.metro.start()
        else: self.metro.stop()

    # DnD
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent):
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
        except FileNotFoundError:
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
        self._start_analysis(path, force_analyze=False, finished_slot=self._on_features_ready)

    def reanalyze_file(self, path: str):
        self._start_analysis(path, force_analyze=True, finished_slot=self._on_features_reanalyze)

    def _start_analysis(self, path: str, *, force_analyze: bool, finished_slot):
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
        if (track != None) and (not force_analyze):
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

        if auto_load:
            self._apply_model_update(
                feat_std,
                properties,
                gp,
                duration_sec,
                stop_playback=True,
                refresh_source=True,
            )

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

    def _on_features_loaded(self):
        self.bus.sig_window_changed.emit(self.tl.window_sec)
        self.bus.sig_center_changed.emit(self.tl.center_t)

    def _on_settings_save(self, _config: config):
        with open("config.json", "r") as f:
            prev_cfg = json.load(f)
        with open("config.json", "w") as f:
            json.dump(_config.to_dict(), f)
        if prev_cfg["viewconfig"] != _config.to_dict()["viewconfig"]:
            self.bus.sig_reload_UI.emit(_config)
            if _config.viewconfig.enable_metronome: self.metro.start()
            else: self.metro.stop()
            self.metro.set_soundfile(_config.viewconfig.metronome_wav_path)

    def _open_support(self):
        # Non-modal dialog; reuse instance
        self.support_dlg.show()
        self.support_dlg.raise_()
        self.support_dlg.activateWindow()

    def _open_about(self):
        self.about_dialog.show()
        self.about_dialog.raise_()
        self.about_dialog.activateWindow()
