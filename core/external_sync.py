"""External Sync: Mixlyzer follows another program's deck (the track loaded, its position) and locks its own
transport meanwhile. The deck's state comes as OSC messages (core.osc) over UDP, one per deck as often as the
sender reads it:

    /mixlyzer/deck/time     i deck  s path  T|F loaded  T|F active  d|f seconds
    /mixlyzer/deck/samples  i deck  s path  T|F loaded  T|F active  h|i samples  i rate

``rate`` is the rate the sender counts samples at, 0: the file's own (its total samples). A flag may also come as
an i (0 / 1), for senders without T / F. The deck followed is the lowest-numbered one that is loaded and active;
a deck not heard from for ``STALE_SEC`` is gone. Messages are received on their own thread, which keeps each
deck's newest state; the GUI takes them at the view's frame rate (no queued signal per message).

The receiver listens on this computer only (127.0.0.1) unless the settings name another address; then any
device on that network can drive the deck, so only tracks already in the library are followed (none is
analysed for it). A path must be a local file: network paths (UNC, //host) are refused, also when a link
resolves to one, as opening one sends the user's credentials to that host."""

from __future__ import annotations

import ipaddress
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Optional

from PySide6 import QtCore

from core import osc
from core.config import externalsyncconfig
from core.event_bus import EventBus
from core.i18n import tr
from core.track.model import DataModel

TIME_ADDRESS = "/mixlyzer/deck/time"
SAMPLES_ADDRESS = "/mixlyzer/deck/samples"
STALE_SEC = 1.0      # a deck not heard from for this long is no longer followed
MAX_DECKS = 8
MAX_PATH_CHARS = 4096


@dataclass(frozen=True)
class DeckReport:
    deck: int
    path: str
    loaded: bool
    active: bool
    seconds: Optional[float]     # the position, when sent as time
    samples: Optional[int]       # or as samples, counted at ``rate`` (0: the file's)
    rate: int
    received_at: float           # time.monotonic()


def _flag(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    return None


def deck_report(address: str, args: list, now: float) -> Optional[DeckReport]:
    """The deck state an OSC message carries, None for any other message (or a malformed one)."""
    if address not in (TIME_ADDRESS, SAMPLES_ADDRESS) or len(args) < 5:
        return None
    deck, path, loaded, active = args[0], args[1], _flag(args[2]), _flag(args[3])
    if isinstance(deck, bool) or not isinstance(deck, int) or not 1 <= deck <= MAX_DECKS:
        return None
    if not isinstance(path, str) or len(path) > MAX_PATH_CHARS or loaded is None or active is None:
        return None
    if address == TIME_ADDRESS:
        seconds = args[4]
        if len(args) != 5 or isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            return None
        if not 0.0 <= float(seconds) < 1e7:   # also refuses NaN
            return None
        return DeckReport(deck, path, loaded, active, float(seconds), None, 0, now)
    if len(args) != 6:
        return None
    samples, rate = args[4], args[5]
    if isinstance(samples, bool) or not isinstance(samples, int) or isinstance(rate, bool) or not isinstance(rate, int):
        return None
    if samples < 0 or not 0 <= rate <= 1_000_000:
        return None
    return DeckReport(deck, path, loaded, active, None, samples, rate, now)


def is_loopback(host: str) -> bool:
    try:
        return all(ipaddress.ip_address(info[4][0]).is_loopback
                   for info in socket.getaddrinfo(host, None, type=socket.SOCK_DGRAM))
    except (OSError, ValueError):
        return False


def _network_path(text: str) -> bool:
    """A path naming a file on another computer (Windows UNC in any spelling, //host)."""
    if text.startswith(("\\\\", "//", "\\/", "/\\")):
        return True
    return PureWindowsPath(text).drive.startswith(("\\\\", "//"))


class OscReceiver:
    """A UDP socket on its own thread keeping each deck's newest report."""

    def __init__(self, host: str, port: int) -> None:
        family = socket.getaddrinfo(host, port, type=socket.SOCK_DGRAM)[0][0]
        self._sock = socket.socket(family, socket.SOCK_DGRAM)
        try:
            self._sock.bind((host, port))
        except OSError:
            self._sock.close()
            raise
        self._sock.settimeout(0.2)
        self.address = self._sock.getsockname()[:2]
        self._lock = threading.Lock()
        self._decks: dict[int, DeckReport] = {}
        self._sender = ""
        self._refused = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ExternalSyncOSC", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                packet, sender = self._sock.recvfrom(osc.MAX_PACKET + 1)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            now = time.monotonic()
            try:
                reports = [r for r in (deck_report(a, args, now) for a, args in osc.parse(packet)) if r]
            except osc.OscError:
                reports = []
            with self._lock:
                if not reports:
                    self._refused += 1
                    continue
                for report in reports:
                    self._decks[report.deck] = report
                self._sender = f"{sender[0]}:{sender[1]}"

    def latest(self) -> tuple[dict[int, DeckReport], str, int]:
        """(each deck's newest report, the last sender, packets refused)."""
        with self._lock:
            return dict(self._decks), self._sender, self._refused

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        self._thread.join(timeout=1.0)


class ExternalSyncController(QtCore.QObject):
    """Tracks app state and optionally follows an external deck (OSC)."""

    def __init__(
        self,
        bus: EventBus,
        model: DataModel,
        cfg: externalsyncconfig,
        *,
        poll_fps: int = 60,
        status_callback: Callable[[str], None] | None = None,
        load_track_callback: Callable[[str, float], None] | None = None,
        seek_callback: Callable[[float], None] | None = None,
        track_exists_callback: Callable[[str], bool] | None = None,
        library_path_callback: Callable[[str], str | None] | None = None,
        track_duration_callback: Callable[[str], float | None] | None = None,
        track_total_samples_callback: Callable[[str], int | None] | None = None,
        failure_callback: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.bus = bus
        self.model = model
        self._cfg = cfg
        self._time_sec = 0.0
        self._is_playing = False
        self._status_callback = status_callback
        self._load_track_callback = load_track_callback
        self._seek_callback = seek_callback
        self._track_exists_callback = track_exists_callback
        self._library_path_callback = library_path_callback
        self._track_duration_callback = track_duration_callback
        self._track_total_samples_callback = track_total_samples_callback
        self._failure_callback = failure_callback
        self._missing_path_logged: str | None = None
        self._pending_external_path: str | None = None
        self._pending_external_raw_path: str | None = None
        self._pending_external_time_sec: float = 0.0
        self._analysis_request_inflight: str | None = None
        self._analysis_request_raw_path: str | None = None
        self._load_request_inflight: str | None = None
        self._invalid_external_path_logged: str | None = None
        self._not_in_library_logged: str | None = None
        self._validated_external_paths: dict[str, str] = {}
        self._receiver: OscReceiver | None = None
        self._library_only = False
        self._following: int = 0   # the deck followed (0: none)

        self._poll = QtCore.QTimer(self)
        self.set_poll_fps(int(poll_fps))
        self._poll.timeout.connect(self._poll_external)

        self.bus.sig_time_changed.connect(self._on_time_changed)
        self.bus.sig_playback_status.connect(self._on_playback_status_changed)
        self.model.sig_loaded.connect(self._on_features_loaded)
        self.bus.sig_lib_updated.connect(self._on_library_updated)
        self.model.sig_properties_changed.connect(self.publish_state)
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.close)   # the port let go with the app

        self._apply_polling_state()

    def set_config(self, cfg: externalsyncconfig) -> None:
        listen_changed = (cfg.osc_host, cfg.osc_port) != (self._cfg.osc_host, self._cfg.osc_port)
        self._cfg = cfg
        self._missing_path_logged = None
        self._pending_external_path = None
        self._pending_external_raw_path = None
        self._pending_external_time_sec = 0.0
        self._analysis_request_inflight = None
        self._analysis_request_raw_path = None
        self._load_request_inflight = None
        self._invalid_external_path_logged = None
        self._not_in_library_logged = None
        self._validated_external_paths = {}
        if listen_changed:
            self._close_receiver()
        self._apply_polling_state()
        self.publish_state()

    def is_enabled(self) -> bool:
        return bool(self._cfg.enabled)

    def set_poll_fps(self, fps: int) -> None:
        fps = max(1, min(240, int(fps)))
        interval_ms = max(1, int(round(1000.0 / fps)))
        self._poll.setInterval(interval_ms)

    def close(self) -> None:
        self._poll.stop()
        self._close_receiver()

    def listening_on(self) -> str:
        """host:port being listened on ("" when not)."""
        if self._receiver is None:
            return ""
        host, port = self._receiver.address
        return f"[{host}]:{port}" if ":" in str(host) else f"{host}:{port}"

    def snapshot(self) -> dict[str, Any]:
        props = self.model.properties if isinstance(self.model.properties, dict) else {}
        title = str(props.get("title") or "").strip()
        artist = str(props.get("artist") or "").strip()
        path = str(props.get("path") or "").strip()
        uid = str(props.get("uid") or "").strip()
        duration_sec = float(props.get("duration_sec") or self.model.duration_sec or 0.0)
        decks, sender, refused = self._receiver.latest() if self._receiver is not None else ({}, "", 0)
        now = time.monotonic()
        return {
            "enabled": bool(self._cfg.enabled),
            "osc": {
                "listening": self.listening_on(),
                "library_only": bool(self._library_only),
                "sender": sender,
                "refused_packets": int(refused),
                "following": int(self._following),
                "decks": {
                    n: {"loaded": r.loaded, "active": r.active, "age_sec": now - r.received_at}
                    for n, r in sorted(decks.items())
                },
            },
            "track": {
                "loaded": bool(path),
                "uid": uid,
                "path": path,
                "title": title,
                "artist": artist,
                "duration_sec": duration_sec,
            },
            "transport": {
                "is_playing": bool(self._is_playing),
                "position_sec": float(self._time_sec),
            },
        }

    @QtCore.Slot(float)
    def _on_time_changed(self, sec: float) -> None:
        self._time_sec = max(0.0, float(sec))
        self.publish_state()

    @QtCore.Slot(bool)
    def _on_playback_status_changed(self, is_playing: bool) -> None:
        self._is_playing = bool(is_playing)
        self.publish_state()

    @QtCore.Slot()
    def _on_features_loaded(self) -> None:
        loaded_path = self._loaded_track_path()
        self._load_request_inflight = None
        if loaded_path and loaded_path == self._pending_external_path:
            self._clear_pending_external()
        if loaded_path and loaded_path == self._analysis_request_inflight:
            self._analysis_request_inflight = None
            self._analysis_request_raw_path = None
        self._missing_path_logged = None
        QtCore.QTimer.singleShot(0, self._drive_pending_external)
        self.publish_state()

    @QtCore.Slot(object)
    def _on_library_updated(self, _data) -> None:
        # a file may now be in the library under its own path: look it up again
        self._validated_external_paths.clear()
        if (
            self._analysis_request_inflight
            and self._analysis_request_raw_path
            and self._track_exists_in_library(self._analysis_request_raw_path)
        ):
            self._analysis_request_inflight = None
            self._analysis_request_raw_path = None
        QtCore.QTimer.singleShot(0, self._drive_pending_external)
        self.publish_state()

    @QtCore.Slot()
    def publish_state(self) -> None:
        self.bus.sig_external_sync_state.emit(self.snapshot())

    def _apply_polling_state(self) -> None:
        if not self._cfg.enabled:
            self._poll.stop()
            self._close_receiver()
            return
        if self._receiver is None:
            host, port = str(self._cfg.osc_host or "").strip() or "127.0.0.1", int(self._cfg.osc_port)
            try:
                self._receiver = OscReceiver(host, port)
            except (OSError, ValueError, OverflowError) as exc:
                self._disable_due_to_failure(
                    tr("External Sync could not listen for OSC on {host}:{port} ({error}) and has been disabled.")
                    .format(host=host, port=port, error=exc))
                return
            self._library_only = not is_loopback(host)
            self._log(tr("External Sync listening for OSC on {address}.").format(address=self.listening_on()))
        if not self._poll.isActive():
            self._poll.start()

    def _close_receiver(self) -> None:
        receiver, self._receiver = self._receiver, None
        self._following = 0
        if receiver is not None:
            receiver.close()

    @QtCore.Slot()
    def _poll_external(self) -> None:
        if not self._cfg.enabled or self._receiver is None:
            return
        decks, _sender, _refused = self._receiver.latest()
        now = time.monotonic()
        followed = sorted(n for n, r in decks.items() if r.loaded and r.active and now - r.received_at < STALE_SEC)
        if not followed:
            self._following = 0
            return
        report = decks[followed[0]]
        self._following = report.deck
        path = report.path.strip()
        if report.seconds is not None:
            time_sec = report.seconds
        else:
            time_sec = self._sample_index_to_time(path, float(report.samples), report.rate)
        self._apply_external_deck({"path": path, "loaded": True, "time_sec": time_sec})

    def _apply_external_deck(self, external: dict[str, Any]) -> None:
        path = str(external.get("path") or "").strip()
        loaded = bool(external.get("loaded"))
        time_sec = max(0.0, float(external.get("time_sec") or 0.0))

        if not loaded or not path:
            self._clear_pending_external()
            self._analysis_request_inflight = None
            self._analysis_request_raw_path = None
            self._load_request_inflight = None
            self._missing_path_logged = None
            self._invalid_external_path_logged = None
            return
        validated_path = self._validate_external_track_path(path)
        if not validated_path:
            if self._invalid_external_path_logged != path:
                self._invalid_external_path_logged = path
                self._log(tr("External Sync ignored unsafe track path: {path}").format(path=path))
            self._clear_pending_external()
            self._analysis_request_inflight = None
            self._analysis_request_raw_path = None
            self._load_request_inflight = None
            self._missing_path_logged = None
            return
        self._invalid_external_path_logged = None
        norm_external = validated_path
        loaded_path = self._loaded_track_path()

        if loaded_path and loaded_path == norm_external:
            self._clear_pending_external()
            self._analysis_request_inflight = None
            self._analysis_request_raw_path = None
            self._missing_path_logged = None
            self._load_request_inflight = None
            if self._seek_callback is None:
                return
            self._seek_callback(time_sec)
            return

        self._pending_external_path = norm_external
        self._pending_external_raw_path = validated_path
        self._pending_external_time_sec = time_sec
        self._drive_pending_external()

    def _drive_pending_external(self) -> None:
        pending_path = self._pending_external_path
        pending_raw_path = self._pending_external_raw_path
        if not pending_path or not pending_raw_path:
            return

        loaded_path = self._loaded_track_path()
        if loaded_path and loaded_path == pending_path:
            self._clear_pending_external()
            self._analysis_request_inflight = None
            self._analysis_request_raw_path = None
            self._missing_path_logged = None
            self._load_request_inflight = None
            return

        if self._load_request_inflight or self._analysis_request_inflight:
            return

        if self._track_exists_in_library(pending_raw_path):
            self._missing_path_logged = None
            self._request_track_load(pending_raw_path, self._pending_external_time_sec)
            return

        if self._library_only:   # a sender on the network: no analysis started for it
            if self._not_in_library_logged != pending_path:
                self._not_in_library_logged = pending_path
                self._log(tr("External Sync ignored a track not in the library (OSC from the network): {path}")
                          .format(path=pending_raw_path))
            return
        if self._missing_path_logged != pending_path:
            self._missing_path_logged = pending_path
            self._log(tr("External Sync analyzing external track before load: {path}").format(path=pending_raw_path))
        self._request_track_analysis(pending_raw_path, self._pending_external_time_sec)

    def _request_track_analysis(self, path: str, time_sec: float) -> None:
        if self._analysis_request_inflight == path or self._load_track_callback is None:
            return
        self._analysis_request_inflight = path
        self._analysis_request_raw_path = path
        self._load_track_callback(path, time_sec)

    def _request_track_load(self, path: str, time_sec: float) -> None:
        if self._load_request_inflight == path or self._load_track_callback is None:
            return
        self._load_request_inflight = path
        self._load_track_callback(path, time_sec)

    def _clear_pending_external(self) -> None:
        self._pending_external_path = None
        self._pending_external_raw_path = None
        self._pending_external_time_sec = 0.0

    def _loaded_track_path(self) -> str:
        props = self.model.properties if isinstance(self.model.properties, dict) else {}
        return str(props.get("path") or "").strip()

    def _validate_external_track_path(self, path: str | None) -> str:
        """The track the deck's ``path`` points to, as the app names it: the path the
        library stores that file under (the deck may spell it in another case, with
        other separators or through a link), else the file's resolved path. Empty
        for a path that is not a local file (checked before the file system is
        touched, and again once a link is resolved)."""
        raw = str(path or "").strip()
        if not raw:
            return ""
        cached = self._validated_external_paths.get(raw)
        if cached is not None:
            return cached
        resolved = self._local_file(raw)
        if resolved:
            resolved = self._library_path(resolved) or resolved
        self._validated_external_paths[raw] = resolved
        return resolved

    @staticmethod
    def _local_file(raw: str) -> str:
        if _network_path(raw) or "\x00" in raw:
            return ""
        try:
            expanded = Path(raw).expanduser()
            if not expanded.is_absolute():
                return ""
            candidate = expanded.resolve(strict=False)
            if _network_path(str(candidate)) or not candidate.is_file():
                return ""
        except (OSError, RuntimeError, ValueError):
            return ""
        return str(candidate)

    def _library_path(self, path: str) -> str:
        if self._library_path_callback is None:
            return ""
        try:
            return str(self._library_path_callback(path) or "")
        except Exception:
            return ""

    def _track_exists_in_library(self, path: str) -> bool:
        if self._track_exists_callback is None:
            return False
        try:
            return bool(self._track_exists_callback(path))
        except Exception:
            return False

    def _sample_index_to_time(self, path: str, sample_index: float, rate: int) -> float:
        """A position in samples as seconds: counted at ``rate``, or (0) at the file's own rate, so the index
        over its total samples is the share of its duration; clamped to the track."""
        validated = self._validate_external_track_path(path)
        if not validated:
            return 0.0
        duration_sec = self._resolve_track_duration(validated)
        if duration_sec <= 0.0:
            return 0.0
        if rate > 0:
            return min(max(0.0, sample_index) / float(rate), duration_sec)
        total_samples = self._resolve_track_total_samples(validated)
        if total_samples <= 0.0:
            return 0.0
        clamped_index = max(0.0, min(sample_index, total_samples))
        return (clamped_index / total_samples) * duration_sec

    def _resolve_track_duration(self, path: str) -> float:
        target = str(path or "").strip()
        if not target:
            return 0.0
        loaded_path = self._loaded_track_path()
        if loaded_path and loaded_path == target:
            try:
                return float(self.model.duration_sec or 0.0)
            except Exception:
                return 0.0
        if self._track_duration_callback is not None:
            try:
                return max(0.0, float(self._track_duration_callback(path) or 0.0))
            except Exception:
                return 0.0
        return 0.0

    def _resolve_track_total_samples(self, path: str) -> float:
        target = str(path or "").strip()
        if not target:
            return 0.0
        props = self.model.properties if isinstance(self.model.properties, dict) else {}
        loaded_path = self._loaded_track_path()
        if loaded_path and loaded_path == target:
            try:
                return max(0.0, float(props.get("total_samples") or 0.0))
            except Exception:
                return 0.0
        if self._track_total_samples_callback is not None:
            try:
                return max(0.0, float(self._track_total_samples_callback(path) or 0.0))
            except Exception:
                return 0.0
        return 0.0

    def _disable_due_to_failure(self, message: str) -> None:
        if not bool(self._cfg.enabled):
            return
        self._cfg.enabled = False
        self._close_receiver()
        self._apply_polling_state()
        self.publish_state()
        self.bus.sig_external_sync_enabled.emit(False)
        self._log(message)
        if self._failure_callback is not None:
            try:
                self._failure_callback(message)
            except Exception:
                pass

    def _log(self, message: str) -> None:
        print(f"[ExternalSync] {message}")
        if self._status_callback is not None:
            try:
                self._status_callback(message)
            except Exception:
                pass
