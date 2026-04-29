from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Optional
from urllib.parse import quote
from xml.dom import minidom
from xml.etree import ElementTree as ET

import numpy as np
from PySide6 import QtCore

from core.analysis_lib_handler import FeatureNPZStore
from core.library_handler import TrackRow
from third_party.rekordbox import (
    _compute_average_bpm,
    _extract_jump_marks,
    _extract_tempo_segments,
    _format_date,
    _generate_track_id,
    _get_duration,
    _kind_from_path,
    _safe_scalar,
    _tempo_entries_for_xml,
)
from utils.labels import idx_to_labels


class RekordboxXmlSync(QtCore.QObject):
    def __init__(self, *, cfg_getter: Callable[[], object], parent=None) -> None:
        super().__init__(parent)
        self._cfg_getter = cfg_getter

    @QtCore.Slot(object)
    def sync_incremental(self, rows: object) -> None:
        track_rows = self._coerce_rows(rows)
        if track_rows is None:
            return
        self._sync_incremental_rows(track_rows)

    @QtCore.Slot(bool)
    def sync_requested(self, full_rebuild: bool) -> None:
        cfg = self._cfg_getter()
        if cfg is None:
            return
        db_rows = self._load_library_rows(cfg)
        self._sync(db_rows, full_rebuild=bool(full_rebuild))

    @QtCore.Slot(str)
    def sync_track_requested(self, uid: str) -> None:
        track_uid = str(uid or "").strip()
        if not track_uid:
            return
        cfg = self._cfg_getter()
        if cfg is None:
            return
        libcfg = getattr(cfg, "libconfig", None)
        if libcfg is None:
            return
        enabled = bool(getattr(libcfg, "rekordbox_sync_enabled", False))
        xml_path_text = str(getattr(libcfg, "rekordbox_xml_path", "") or "").strip()
        if not enabled or not xml_path_text:
            return

        xml_path = Path(xml_path_text).expanduser()
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        store = FeatureNPZStore(base_dir=str(libcfg.libpath), compressed=True)
        root, collection, existing = self._load_existing_tree(xml_path, force_new=False)
        row = self._load_track_row(cfg, track_uid)
        existing_node = existing.get(track_uid)
        if existing_node is not None:
            collection.remove(existing_node)

        if row is not None:
            features = self._load_features(store, row.uid)
            track_hash = self._track_hash(row, store)
            node = _build_track_element(
                row,
                features,
                resolve_path=lambda p: self._resolve_audio_path(p),
                track_hash=track_hash,
            )
            collection.append(node)

        collection.set("Entries", str(len(collection.findall("TRACK"))))
        xml_text = _prettify_xml(root)
        xml_path.write_text(xml_text, encoding="utf-8")

    def _sync(self, rows: list[TrackRow], *, full_rebuild: bool) -> None:
        cfg = self._cfg_getter()
        if cfg is None:
            return
        libcfg = getattr(cfg, "libconfig", None)
        if libcfg is None:
            return
        enabled = bool(getattr(libcfg, "rekordbox_sync_enabled", False))
        xml_path_text = str(getattr(libcfg, "rekordbox_xml_path", "") or "").strip()
        if not enabled or not xml_path_text:
            return

        xml_path = Path(xml_path_text).expanduser()
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        store = FeatureNPZStore(base_dir=str(libcfg.libpath), compressed=True)
        resolver = lambda p: self._resolve_audio_path(p)

        root, collection, existing = self._load_existing_tree(xml_path, force_new=full_rebuild)
        ordered_track_nodes: list[ET.Element] = []
        for row in rows:
            track_key = self._track_key_for_row(row)
            track_hash = self._track_hash(row, store)
            prev = existing.get(track_key)
            if (
                prev is not None
                and not full_rebuild
                and str(prev.attrib.get("MixlyzerHash", "")) == track_hash
            ):
                node = prev
            else:
                features = self._load_features(store, row.uid)
                node = _build_track_element(
                    row,
                    features,
                    resolve_path=resolver,
                    track_hash=track_hash,
                )
            ordered_track_nodes.append(node)

        collection.clear()
        collection.set("Entries", str(len(ordered_track_nodes)))
        for node in ordered_track_nodes:
            collection.append(node)

        xml_text = _prettify_xml(root)
        xml_path.write_text(xml_text, encoding="utf-8")

    def _sync_incremental_rows(self, rows: list[TrackRow]) -> None:
        cfg = self._cfg_getter()
        if cfg is None:
            return
        libcfg = getattr(cfg, "libconfig", None)
        if libcfg is None:
            return
        enabled = bool(getattr(libcfg, "rekordbox_sync_enabled", False))
        xml_path_text = str(getattr(libcfg, "rekordbox_xml_path", "") or "").strip()
        if not enabled or not xml_path_text:
            return

        xml_path = Path(xml_path_text).expanduser()
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        store = FeatureNPZStore(base_dir=str(libcfg.libpath), compressed=True)
        resolver = lambda p: self._resolve_audio_path(p)
        root, collection, existing = self._load_existing_tree(xml_path, force_new=False)

        for row in rows:
            track_key = self._track_key_for_row(row)
            existing_node = existing.get(track_key)
            if existing_node is not None:
                collection.remove(existing_node)

            features = self._load_features(store, row.uid)
            track_hash = self._track_hash(row, store)
            node = _build_track_element(
                row,
                features,
                resolve_path=resolver,
                track_hash=track_hash,
            )
            collection.append(node)

        collection.set("Entries", str(len(collection.findall("TRACK"))))
        xml_text = _prettify_xml(root)
        xml_path.write_text(xml_text, encoding="utf-8")

    def _load_existing_tree(
        self,
        xml_path: Path,
        *,
        force_new: bool,
    ) -> tuple[ET.Element, ET.Element, dict[str, ET.Element]]:
        if not force_new and xml_path.exists():
            try:
                root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
                collection = root.find("COLLECTION")
                if collection is not None:
                    existing = {}
                    for track in collection.findall("TRACK"):
                        track_key = self._track_key_for_element(track)
                        if track_key:
                            existing[track_key] = track
                    return root, collection, existing
            except Exception:
                pass
        root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
        ET.SubElement(root, "PRODUCT", Name="Mixlyzer", Version="1.0.0", Company="Mixlyzer")
        collection = ET.SubElement(root, "COLLECTION", Entries="0")
        return root, collection, {}

    @staticmethod
    def _track_key_for_row(row: TrackRow) -> str:
        uid = str(getattr(row, "uid", "") or "").strip()
        if uid:
            return uid
        return _generate_track_id(row)

    @staticmethod
    def _track_key_for_element(track: ET.Element) -> str:
        uid = str(track.attrib.get("MixlyzerUID", "")).strip()
        if uid:
            return uid
        return str(track.attrib.get("TrackID", "")).strip()

    def _load_features(self, store: FeatureNPZStore, uid: Optional[str]) -> dict[str, np.ndarray]:
        if not uid:
            return {}
        try:
            return store.load(uid)
        except Exception:
            return {}

    def _track_hash(self, row: TrackRow, store: FeatureNPZStore) -> str:
        parts = [str(v) for k, v in sorted(asdict(row).items())]
        uid = str(row.uid or "").strip()
        npz_path = Path(store.path(uid)) if uid else None
        if npz_path is not None and npz_path.exists():
            stat = npz_path.stat()
            parts.append(str(stat.st_mtime_ns))
            parts.append(str(stat.st_size))
        digest = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
        return digest

    def _coerce_rows(self, rows: object) -> list[TrackRow] | None:
        if rows is None:
            return []
        result: list[TrackRow] = []
        try:
            iterator: Iterable = rows
        except Exception:
            return None
        for row in iterator:
            if isinstance(row, TrackRow):
                result.append(row)
            elif isinstance(row, dict):
                try:
                    result.append(TrackRow(**row))
                except Exception:
                    continue
        return result

    def _load_library_rows(self, cfg) -> list[TrackRow]:
        from core.library_handler import LibraryDB

        db = LibraryDB(str(Path(cfg.libconfig.libpath) / "library.db"))
        db.connect()
        try:
            return db.list_all()
        finally:
            db.close()

    def _load_track_row(self, cfg, uid: str) -> TrackRow | None:
        from core.library_handler import LibraryDB

        db = LibraryDB(str(Path(cfg.libconfig.libpath) / "library.db"))
        db.connect()
        try:
            return db.get_by_uid(uid)
        finally:
            db.close()

    @staticmethod
    def _resolve_audio_path(path: str | None) -> Optional[Path]:
        raw = str(path or "").strip()
        if not raw:
            return None
        candidate = Path(raw).expanduser()
        return candidate if candidate.exists() else candidate


def _build_track_element(
    track: TrackRow,
    features: dict[str, np.ndarray],
    *,
    resolve_path,
    track_hash: str,
) -> ET.Element:
    tempo_segments = _extract_tempo_segments(features)
    duration = _get_duration(track, features)
    average_bpm = _compute_average_bpm(track, tempo_segments, features)
    position_marks = _extract_jump_marks(features)

    if duration <= 0 and tempo_segments.size:
        duration = float(np.max(tempo_segments[:, 1]))

    resolved_audio = resolve_path(track.path) if track.path else None
    file_size = 0
    if resolved_audio and resolved_audio.exists():
        try:
            file_size = resolved_audio.stat().st_size
        except OSError:
            file_size = 0
    elif track.file_size:
        file_size = int(track.file_size)

    sample_rate = int(_safe_scalar(features.get("sr")) or 44100)
    tempo_global = _safe_scalar(features.get("tempo_global"))
    if not np.isfinite(average_bpm) or average_bpm <= 0:
        average_bpm = tempo_global if tempo_global > 0 else 0.0

    bit_rate = 0
    if file_size > 0 and duration > 0:
        bit_rate = int(round((file_size * 8) / duration / 1000))

    song_title = track.title or (resolved_audio.stem if resolved_audio else "Untitled")
    tonality = ""
    if track.key is not None:
        try:
            tonality = idx_to_labels(track.key)[0]
        except Exception:
            tonality = ""

    fallback_audio = Path(track.path) if track.path else None
    track_attrs = {
        "TrackID": _generate_track_id(track),
        "MixlyzerUID": str(track.uid or ""),
        "Name": song_title,
        "Artist": track.artist or "",
        "Composer": "",
        "Album": track.album or "",
        "Grouping": "",
        "Genre": "",
        "Kind": _kind_from_path(resolved_audio),
        "Size": str(file_size),
        "TotalTime": str(int(round(duration)) if duration > 0 else 0),
        "DiscNumber": "1",
        "TrackNumber": "1",
        "Year": "0",
        "AverageBpm": f"{average_bpm:.2f}" if average_bpm > 0 else "0.00",
        "DateAdded": _format_date(track.added_ts),
        "BitRate": str(bit_rate),
        "SampleRate": str(sample_rate),
        "Comments": track.comment or "",
        "PlayCount": "0",
        "Rating": str(int(track.rating or 0)),
        "Location": _file_url(resolved_audio if resolved_audio is not None else fallback_audio),
        "Remixer": "",
        "Tonality": tonality,
        "Label": "",
        "Mix": "",
        "MixlyzerHash": track_hash,
    }
    track_elem = ET.Element("TRACK", track_attrs)

    time_signature = int(max(1, _safe_scalar(features.get("timesignature")) or 4))
    tempo_entries = _tempo_entries_for_xml(tempo_segments, average_bpm)
    for tempo in tempo_entries:
        ET.SubElement(
            track_elem,
            "TEMPO",
            Inizio=f"{tempo['start']:.3f}",
            Bpm=f"{tempo['bpm']:.2f}",
            Metro=f"{time_signature}/4",
            Battito=str(int(tempo.get("battito", 1))),
        )

    jump_letters = "ABCDEFGH"
    for idx, mark in enumerate(sorted(position_marks, key=lambda m: m["start"])):
        color = mark.get("color", (40, 226, 20))
        slot = mark.get("slot")
        if isinstance(slot, (int, np.integer)) and 0 <= slot < len(jump_letters):
            num_value = int(slot)
            label = mark["name"] or f"Jump {jump_letters[num_value]}"
        else:
            num_value = idx % len(jump_letters)
            label = mark["name"] or f"Jump {jump_letters[num_value]}"
        ET.SubElement(
            track_elem,
            "POSITION_MARK",
            Name=label,
            Type="0",
            Start=f"{mark['start']:.3f}",
            Num=str(num_value),
            Red=str(color[0]),
            Green=str(color[1]),
            Blue=str(color[2]),
        )
    return track_elem


def _file_url(audio_path: Optional[Path]) -> str:
    if audio_path is None:
        return ""
    target = audio_path.resolve()
    url_path = target.as_posix()
    encoded = quote(url_path, safe="/:.")
    if encoded.startswith("/"):
        return f"file://localhost{encoded}"
    return f"file://localhost/{encoded}"


def _prettify_xml(root: ET.Element) -> str:
    _strip_whitespace_only_nodes(root)
    xml_bytes = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ")
    if not pretty.startswith("<?xml"):
        pretty = '<?xml version="1.0" encoding="UTF-8"?>\n' + pretty
    return pretty


def _strip_whitespace_only_nodes(elem: ET.Element) -> None:
    if elem.text is not None and not elem.text.strip():
        elem.text = None
    for child in list(elem):
        _strip_whitespace_only_nodes(child)
        if child.tail is not None and not child.tail.strip():
            child.tail = None
