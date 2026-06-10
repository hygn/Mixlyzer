from __future__ import annotations

import ctypes
import csv
import io
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pymem
import pymem.process
import pymem.ressources.structure
from pymem.ptypes import RemotePointer
from PySide6 import QtCore

from core.config import externalsyncconfig, memorydeckconfig, memoryvalueconfig
from core.event_bus import EventBus
from core.model import DataModel
from core.resource_paths import process_denylist_path


class ExternalSyncController(QtCore.QObject):
    """Tracks app state and optionally follows an external memory source."""

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
        self._blocked_process_logged: str | None = None
        self._denylist_unavailable_logged: bool = False
        self._invalid_external_path_logged: str | None = None
        self._validated_external_paths: dict[str, str] = {}
        self._pm: pymem.Pymem | None = None
        self._pm_pid: int = 0
        self._process_list_cache: list[dict[str, int | str]] = []
        self._process_list_cache_ts: float = 0.0

        self._poll = QtCore.QTimer(self)
        self.set_poll_fps(int(poll_fps))
        self._poll.timeout.connect(self._poll_external_memory)

        self.bus.sig_time_changed.connect(self._on_time_changed)
        self.bus.sig_playback_status.connect(self._on_playback_status_changed)
        self.bus.sig_features_loaded.connect(self._on_features_loaded)
        self.bus.sig_lib_updated.connect(self._on_library_updated)
        self.bus.sig_properties_loaded.connect(self.publish_state)

        self._apply_polling_state()

    def set_config(self, cfg: externalsyncconfig) -> None:
        self._cfg = cfg
        self._missing_path_logged = None
        self._pending_external_path = None
        self._pending_external_raw_path = None
        self._pending_external_time_sec = 0.0
        self._analysis_request_inflight = None
        self._analysis_request_raw_path = None
        self._load_request_inflight = None
        self._blocked_process_logged = None
        self._denylist_unavailable_logged = False
        self._invalid_external_path_logged = None
        self._validated_external_paths = {}
        self._process_list_cache = []
        self._process_list_cache_ts = 0.0
        self._close_process_handle()
        self._apply_polling_state()
        self.publish_state()

    def is_enabled(self) -> bool:
        return bool(self._cfg.enabled)

    def set_poll_fps(self, fps: int) -> None:
        fps = max(1, min(240, int(fps)))
        interval_ms = max(1, int(round(1000.0 / fps)))
        self._poll.setInterval(interval_ms)

    def snapshot(self) -> dict[str, Any]:
        props = self.model.properties if isinstance(self.model.properties, dict) else {}
        title = str(props.get("title") or "").strip()
        artist = str(props.get("artist") or "").strip()
        path = str(props.get("path") or "").strip()
        uid = str(props.get("uid") or "").strip()
        duration_sec = float(props.get("duration_sec") or self.model.duration_sec or 0.0)
        return {
            "enabled": bool(self._cfg.enabled),
            "mode": str(self._cfg.mode),
            "memory": {
                "process_name": str(self._cfg.memory_process_name),
                "process_pid": int(self._cfg.memory_process_pid),
                "deck1": self._memory_deck_snapshot(self._cfg.memory_deck1),
                "deck2": self._memory_deck_snapshot(self._cfg.memory_deck2),
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

    def _memory_deck_snapshot(self, deck_cfg: memorydeckconfig) -> dict[str, Any]:
        return {
            "time": self._memory_value_snapshot(deck_cfg.time),
            "sample_index": self._memory_value_snapshot(deck_cfg.sample_index),
            "path": self._memory_value_snapshot(deck_cfg.path),
            "active": self._memory_value_snapshot(deck_cfg.active),
            "loaded": self._memory_value_snapshot(deck_cfg.loaded),
        }

    def _memory_value_snapshot(self, spec: memoryvalueconfig) -> dict[str, Any]:
        return {
            "offsets": str(spec.offsets),
            "type": str(spec.value_type),
            "length": int(spec.length),
            "encoding": str(spec.encoding),
            "bit_pos": int(spec.bit_pos),
            "multiplier": float(spec.multiplier),
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
        should_poll = bool(self._cfg.enabled)
        if should_poll and not self._poll.isActive():
            self._poll.start()
        if not should_poll and self._poll.isActive():
            self._poll.stop()
            self._close_process_handle()

    @QtCore.Slot()
    def _poll_external_memory(self) -> None:
        if not self._cfg.enabled:
            return
        process_name = str(self._cfg.memory_process_name or "").strip()
        if self._is_denied_process_name(process_name):
            self._log_blocked_process(process_name or "configured process")
            return
        pid = self._resolve_target_pid()
        if pid <= 0:
            self._close_process_handle()
            return
        pm = self._ensure_process_handle(pid)
        if pm is None:
            self._disable_due_to_failure(
                f"Failed to open process memory for '{process_name or pid}'. Memory Sync has been disabled."
            )
            return
        try:
            external = self._read_active_deck(pm)
            if not external:
                return
            self._apply_external_deck(external)
        except Exception:
            self._close_process_handle()
            self._disable_due_to_failure("Memory Sync read failed and has been disabled.")

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
                self._log(f"External Sync ignored unsafe track path: {path}")
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

        if self._missing_path_logged != pending_path:
            self._missing_path_logged = pending_path
            self._log(f"External Sync analyzing external track before load: {pending_raw_path}")
        self._request_track_analysis(pending_raw_path, self._pending_external_time_sec)

    def _request_track_analysis(self, path: str, time_sec: float) -> None:
        norm_path = self._normalize_path(path)
        if self._analysis_request_inflight == norm_path or self._load_track_callback is None:
            return
        self._analysis_request_inflight = norm_path
        self._analysis_request_raw_path = path
        self._load_track_callback(path, time_sec)

    def _request_track_load(self, path: str, time_sec: float) -> None:
        norm_path = self._normalize_path(path)
        if self._load_request_inflight == norm_path or self._load_track_callback is None:
            return
        self._load_request_inflight = norm_path
        self._load_track_callback(path, time_sec)

    def _clear_pending_external(self) -> None:
        self._pending_external_path = None
        self._pending_external_raw_path = None
        self._pending_external_time_sec = 0.0

    def _loaded_track_path(self) -> str:
        props = self.model.properties if isinstance(self.model.properties, dict) else {}
        return self._normalize_path(props.get("path"))

    def _normalize_path(self, path: str | None) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        return os.path.normcase(os.path.normpath(raw))

    def _validate_external_track_path(self, path: str | None) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        cached = self._validated_external_paths.get(raw)
        if cached is not None:
            return cached
        if raw.startswith(("\\\\", "//")):
            self._validated_external_paths[raw] = ""
            return ""
        try:
            expanded = Path(raw).expanduser()
        except Exception:
            self._validated_external_paths[raw] = ""
            return ""
        if not expanded.is_absolute():
            self._validated_external_paths[raw] = ""
            return ""
        try:
            candidate = expanded.resolve(strict=False)
        except Exception:
            self._validated_external_paths[raw] = ""
            return ""
        try:
            if not candidate.exists() or not candidate.is_file():
                self._validated_external_paths[raw] = ""
                return ""
        except OSError:
            self._validated_external_paths[raw] = ""
            return ""
        normalized = self._normalize_path(str(candidate))
        self._validated_external_paths[raw] = normalized
        return normalized

    def _track_exists_in_library(self, path: str) -> bool:
        if self._track_exists_callback is None:
            return False
        try:
            return bool(self._track_exists_callback(path))
        except Exception:
            return False

    def _read_active_deck(self, pm: pymem.Pymem) -> dict[str, Any] | None:
        decks: list[tuple[int, memorydeckconfig, bool]] = []
        for deck_no, deck_cfg in (
            (1, self._cfg.memory_deck1),
            (2, self._cfg.memory_deck2),
        ):
            deck_state = self._read_deck_state(pm, deck_cfg)
            if deck_state is None:
                continue
            loaded, active = deck_state
            decks.append((deck_no, deck_cfg, loaded and active))
        if not decks:
            return None
        active_decks = [deck for deck in decks if deck[2]]
        if not active_decks:
            return None
        active_decks.sort(key=lambda deck: int(deck[0]))
        deck_no, deck_cfg, _ = active_decks[0]
        external = self._read_deck_payload(pm, deck_cfg)
        if not external:
            return None
        external["deck_no"] = deck_no
        return external

    def _read_deck_state(self, pm: pymem.Pymem, deck_cfg: memorydeckconfig) -> tuple[bool, bool] | None:
        try:
            raw_loaded = self._read_memory_value(pm, deck_cfg.loaded)
            raw_active = self._read_memory_value(pm, deck_cfg.active)
            return bool(raw_loaded), bool(raw_active)
        except Exception as exc:
            self._disable_due_to_failure(f"Memory Sync deck state read failed: {exc}")
            raise

    def _read_deck_payload(self, pm: pymem.Pymem, deck_cfg: memorydeckconfig) -> dict[str, Any] | None:
        try:
            raw_path = self._read_memory_value(pm, deck_cfg.path)
            path = str(raw_path or "").strip()
            if str(self._cfg.mode) == "sample_index":
                raw_sample_index = self._read_memory_value(pm, deck_cfg.sample_index)
                sample_index = max(0.0, float(raw_sample_index or 0.0))
                time_sec = self._sample_index_to_time(path, sample_index)
            else:
                raw_time = self._read_memory_value(pm, deck_cfg.time)
                sample_index = 0.0
                time_sec = max(0.0, float(raw_time or 0.0) * float(deck_cfg.time.multiplier))
        except Exception as exc:
            self._disable_due_to_failure(f"Memory Sync deck payload read failed: {exc}")
            raise
        return {
            "loaded": True,
            "path": path,
            "time_sec": time_sec,
            "sample_index": sample_index,
            "active": True,
        }

    def _sample_index_to_time(self, path: str, sample_index: float) -> float:
        duration_sec = self._resolve_track_duration(path)
        if duration_sec <= 0.0:
            return 0.0
        total_samples = self._resolve_track_total_samples(path)
        if total_samples <= 0.0:
            return 0.0
        clamped_index = max(0.0, min(sample_index, total_samples))
        return (clamped_index / total_samples) * duration_sec

    def _resolve_track_duration(self, path: str) -> float:
        target = self._normalize_path(path)
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
        sample_source = str(getattr(self._cfg, "total_sample_count_source", "reference_sample_rate"))
        if sample_source == "reference_sample_rate":
            duration_sec = self._resolve_track_duration(path)
            reference_sample_rate = max(1.0, float(getattr(self._cfg, "reference_sample_rate", 44100) or 44100))
            if duration_sec <= 0.0:
                return 0.0
            return max(0.0, float(int(round(duration_sec * reference_sample_rate))))

        target = self._normalize_path(path)
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

    def _read_memory_value(self, pm: pymem.Pymem, spec: memoryvalueconfig):
        address = self._resolve_pointer_chain(pm, spec.offsets)
        if address <= 0:
            raise ValueError("Invalid address")
        value_type = str(spec.value_type)
        if value_type == "str":
            return pm.read_string(
                address=address,
                byte=max(1, int(spec.length)),
                encoding=(spec.encoding or "utf-8"),
            )
        if value_type == "float":
            return pm.read_float(address)
        if value_type == "int":
            return pm.read_int(address)
        if value_type == "bool":
            raw = self._read_bytes(pm, address, 1)
            byte_val = raw[0]
            bit_pos = int(spec.bit_pos)
            if 0 <= bit_pos <= 7:
                return bool((byte_val >> bit_pos) & 0x1)
            return bool(byte_val)
        raise ValueError(f"Unsupported memory value type: {value_type}")

    def _resolve_pointer_chain(self, pm: pymem.Pymem, offsets_text: str) -> int:
        tokens = [token.strip() for token in str(offsets_text).split(",") if token.strip()]
        if not tokens:
            return 0
        is_global = self._is_global_address_token(tokens[0])
        offsets = [self._hex_to_int(token) for token in tokens]
        base = int(offsets[0]) if is_global else (int(pm.base_address) + int(offsets[0]))
        if len(offsets) == 1:
            return base
        remote_pointer = RemotePointer(pm.process_handle, base)
        for idx, offset in enumerate(offsets[1:], start=1):
            if idx < len(offsets) - 1:
                remote_pointer = RemotePointer(pm.process_handle, remote_pointer.value + int(offset))
            else:
                return int(remote_pointer.value) + int(offset)
        return base

    def _read_bytes(self, pm: pymem.Pymem, address: int, size: int) -> bytes:
        return bytes(pm.read_bytes(address, size))

    def _hex_to_int(self, token: str) -> int:
        token = str(token or "").strip()
        if not token:
            return 0
        if token[:1].lower() == "g":
            token = token[1:]
        token = token.lower().replace("0x", "")
        return int(token, 16)

    def _is_global_address_token(self, token: str) -> bool:
        text = str(token or "").strip()
        return text[:1].lower() == "g"

    def _resolve_target_pid(self) -> int:
        if int(self._cfg.memory_process_pid) > 0:
            target_pid = int(self._cfg.memory_process_pid)
            return target_pid
        target_name = str(self._cfg.memory_process_name or "").strip().lower()
        if not target_name:
            return 0
        for proc in self._list_running_processes():
            if str(proc.get("name") or "").strip().lower() == target_name:
                if self._is_denied_process_name(str(proc.get("name") or "")):
                    self._log_blocked_process(str(proc.get("name") or ""))
                    return 0
                return int(proc.get("pid") or 0)
        return 0

    def _process_info_for_pid(self, pid: int) -> dict[str, int | str] | None:
        for proc in self._list_running_processes():
            if int(proc.get("pid") or 0) == int(pid):
                return proc
        return None

    def _list_running_processes(self) -> list[dict[str, int | str]]:
        now = time.monotonic()
        if self._process_list_cache and (now - self._process_list_cache_ts) < 1.0:
            return self._process_list_cache
        try:
            out = subprocess.check_output(
                ["tasklist", "/FO", "CSV", "/NH"],
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            return []
        rows: list[dict[str, int | str]] = []
        for row in csv.reader(io.StringIO(out)):
            if len(row) < 2:
                continue
            try:
                pid = int(str(row[1]).strip())
            except Exception:
                continue
            rows.append({"name": str(row[0]).strip(), "pid": pid})
        self._process_list_cache = rows
        self._process_list_cache_ts = now
        return rows

    def _ensure_process_handle(self, pid: int) -> pymem.Pymem | None:
        if self._pm is not None and self._pm_pid == int(pid):
            return self._pm
        self._close_process_handle()
        try:
            process_access = (
                pymem.ressources.structure.PROCESS.PROCESS_QUERY_INFORMATION.value
                | pymem.ressources.structure.PROCESS.PROCESS_VM_READ.value
            )
            pm = pymem.Pymem()
            pm.process_id = int(pid)
            pm.process_handle = pymem.process.open(
                int(pid),
                debug=False,
                process_access=process_access,
            )
            if not pm.process_handle:
                return None
            proc = self._process_info_for_pid(pid)
            process_name = str(proc.get("name") or self._cfg.memory_process_name or pid) if proc else str(
                self._cfg.memory_process_name or pid
            )
            image_path = self._query_process_image_path(pm.process_handle)
            if self._is_denied_process_name(process_name) or self._is_denied_process_image_path(image_path):
                self._log_blocked_process(process_name or str(pid), image_path=image_path)
                try:
                    ctypes.windll.kernel32.CloseHandle(pm.process_handle)
                except Exception:
                    pass
                return None
            pm.check_wow64()
        except Exception:
            return None
        self._pm = pm
        self._pm_pid = int(pid)
        return pm

    def _query_process_image_path(self, process_handle) -> str:
        if os.name != "nt" or not process_handle:
            return ""
        buffer_len = 32768
        buffer = ctypes.create_unicode_buffer(buffer_len)
        size = ctypes.c_ulong(buffer_len)
        try:
            ok = ctypes.windll.kernel32.QueryFullProcessImageNameW(
                process_handle,
                0,
                buffer,
                ctypes.byref(size),
            )
        except Exception:
            return ""
        if not ok:
            return ""
        return str(buffer.value or "").strip()

    def _close_process_handle(self) -> None:
        pm = self._pm
        self._pm = None
        self._pm_pid = 0
        if pm is None:
            return
        try:
            pm.close_process()
        except Exception:
            pass

    def _disable_due_to_failure(self, message: str) -> None:
        if not bool(self._cfg.enabled):
            return
        self._cfg.enabled = False
        self._close_process_handle()
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

    def _log_blocked_process(self, process_name: str, *, image_path: str = "") -> None:
        name = str(process_name or "").strip()
        normalized = self._normalize_process_name(name)
        path_key = self._normalize_process_image_path(image_path)
        cache_key = f"{normalized}|{path_key}"
        if self._blocked_process_logged == cache_key:
            return
        self._blocked_process_logged = cache_key
        if path_key:
            self._log(f"External Sync blocked for denied process: {name} ({image_path})")
        else:
            self._log(f"External Sync blocked for denied process: {name}")

    def _is_denied_process_name(self, name: str) -> bool:
        payload = self._load_process_denylist_payload()
        if payload is None:
            # Fail closed: if the denylist cannot be read, deny every process
            # rather than silently allowing attachment to protected targets.
            return True
        target = self._normalize_process_name(name)
        if not target:
            return False
        return target in self._denied_process_names(payload)

    def _denied_process_names(self, payload: dict[str, Any]) -> set[str]:
        names: set[str] = set()
        for item in payload.get("blocked_process_names", []):
            text = self._normalize_process_name(item)
            if text:
                names.add(text)
        return names

    def _denied_path_keywords(self, payload: dict[str, Any]) -> tuple[str, ...]:
        keywords: list[str] = []
        for item in payload.get("blocked_path_keywords", []):
            text = self._normalize_process_image_path(item)
            if text:
                keywords.append(text)
        return tuple(keywords)

    def _load_process_denylist_payload(self) -> dict[str, Any] | None:
        cache = getattr(self, "_process_denylist_cache", None)
        if isinstance(cache, dict):
            return cache
        try:
            payload = json.loads(process_denylist_path().read_text(encoding="utf-8"))
        except Exception as exc:
            # Do NOT cache the failure so a transient error (e.g. file briefly
            # locked) can recover on a later poll. Log once per failure streak
            # to avoid flooding the poll loop.
            if not self._denylist_unavailable_logged:
                self._denylist_unavailable_logged = True
                self._log(f"Process denylist unavailable ({exc}); blocking all processes.")
            return None
        if not isinstance(payload, dict):
            payload = {}
        self._process_denylist_cache = payload
        self._denylist_unavailable_logged = False
        return payload

    def _is_denied_process_image_path(self, image_path: str) -> bool:
        payload = self._load_process_denylist_payload()
        if payload is None:
            return True  # Fail closed (see _is_denied_process_name).
        normalized = self._normalize_process_image_path(image_path)
        if not normalized:
            return False
        return any(keyword in normalized for keyword in self._denied_path_keywords(payload))

    def _normalize_process_name(self, name: str) -> str:
        text = str(name or "").strip().lower()
        if text.endswith(".exe"):
            text = text[:-4]
        return text

    def _normalize_process_image_path(self, image_path: str) -> str:
        return str(image_path or "").strip().lower().replace("/", "\\")
