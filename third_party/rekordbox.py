from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
import math
from typing import Callable, Optional
from urllib.parse import quote
from xml.dom import minidom
from xml.etree import ElementTree as ET

import numpy as np

from core.library_handler import TrackRow
from utils.labels import idx_to_labels
from utils.jump_cues import extract_jump_cue_pairs
from utils.jump_cues import extract_jump_cue_pairs

ResolvePath = Callable[[str | None], Optional[Path]]

__all__ = ["build_rekordbox_xml", "sanitize_filename"]


def build_rekordbox_xml(
    track: TrackRow,
    features: dict[str, np.ndarray],
    *,
    resolve_path: ResolvePath,
    audio_path: Optional[Path] = None,
) -> tuple[str, str]:
    tempo_segments = _extract_tempo_segments(features)
    duration = _get_duration(track, features)
    average_bpm = _compute_average_bpm(track, tempo_segments, features)
    position_marks = _extract_jump_marks(features)

    if duration <= 0 and tempo_segments.size:
        duration = float(np.max(tempo_segments[:, 1]))

    resolved_audio = audio_path if audio_path and audio_path.exists() else None
    if resolved_audio is None and track.path:
        candidate = resolve_path(track.path)
        if candidate:
            resolved_audio = candidate

    file_size = 0
    if resolved_audio and resolved_audio.exists():
        try:
            file_size = resolved_audio.stat().st_size
        except OSError:
            file_size = 0
    elif track.file_size:
        file_size = int(track.file_size)

    sample_rate = 44100
    tempo_global = _safe_scalar(features.get("tempo_global"))
    if not np.isfinite(average_bpm) or average_bpm <= 0:
        average_bpm = tempo_global if tempo_global > 0 else 0.0

    bit_rate = 0
    if file_size > 0 and duration > 0:
        bit_rate = int(round((file_size * 8) / duration / 1000))

    song_title = track.title or (resolved_audio.stem if resolved_audio else "Untitled")
    xml_filename = f"{sanitize_filename(song_title)}_rekordbox.xml"

    track_id = _generate_track_id(track)
    tonality = ""
    if track.key is not None:
        try:
            tonality = idx_to_labels(track.key)[0]
        except Exception:
            tonality = ""

    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(root, "PRODUCT", Name="Mixlyzer", Version="1.0.0", Company="Mixlyzer")
    collection = ET.SubElement(root, "COLLECTION", Entries="1")

    track_attrs = {
        "TrackID": track_id,
        "Name": song_title,
        "Artist": track.artist or "",
        "Composer": "",
        "Album": track.album or "",
        "Grouping": "",
        "Genre": "",
        "Kind": _kind_from_path(resolved_audio or (resolve_path(track.path) if track.path else None)),
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
        "Location": _file_url(resolved_audio, track, resolve_path),
        "Remixer": "",
        "Tonality": tonality,
        "Label": "",
        "Mix": "",
    }
    track_elem = ET.SubElement(collection, "TRACK", track_attrs)

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

    position_marks = _extract_jump_marks(features)
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

    xml_bytes = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ")
    if not pretty.startswith("<?xml"):
        pretty = '<?xml version="1.0" encoding="UTF-8"?>\n' + pretty
    return pretty, xml_filename


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r'[\\/:*?"<>|]+', "_", name).strip()
    return sanitized or "track"


def _extract_tempo_segments(features: dict[str, np.ndarray]) -> np.ndarray:
    """Return tempo segments as (N,3|4) array.

    Format:
      [start, end, bpm] or [start, end, bpm, inizio]
    """
    raw = features.get("tempo_segments")
    if raw is None:
        return np.empty((0, 3), dtype=float)
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 1:
        if arr.size % 4 == 0:
            arr = arr.reshape((-1, 4))
        elif arr.size % 3 == 0:
            arr = arr.reshape((-1, 3))
        else:
            return np.empty((0, 3), dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return arr[:, : min(arr.shape[1], 4)]
    return np.empty((0, 3), dtype=float)


def _extract_jump_marks(features: dict[str, np.ndarray]) -> list[dict[str, object]]:
    """Extract jump cue marks using the normalized jump_cues_np format."""

    pairs = extract_jump_cue_pairs(features)
    if not pairs:
        return []

    marks: list[dict[str, object]] = []
    forward_idx = 0
    backward_idx = 0

    def _append_mark(segment: dict, color: tuple[int, int, int], role: str) -> None:
        nonlocal forward_idx, backward_idx
        label = str(segment.get("label", "")).strip()
        point = segment.get("point")
        if point is None:
            return
        point_val = float(point)
        if not np.isfinite(point_val):
            return
        slot = _slot_index(label)
        if role == "forward":
            forward_idx += 1
            ordinal = forward_idx
        else:
            backward_idx += 1
            ordinal = backward_idx
        if not label:
            if role == "forward":
                label_to_use = f"Jump F{ordinal}"
            else:
                label_to_use = f"Jump B{ordinal}"
        else:
            label_to_use = label
        marks.append({"name": label_to_use, "start": point_val, "color": color, "slot": slot})

    for pair in pairs:
        forward = pair.get("forward") or {}
        backward = pair.get("backward") or {}
        _append_mark(forward, (40, 226, 20), "forward")
        _append_mark(backward, (226, 126, 40), "backward")

    return marks


def _extract_jump_marks(features: dict[str, np.ndarray]) -> list[dict[str, object]]:
    """Extract jump cue marks using the normalized jump_cues_np format."""

    pairs = extract_jump_cue_pairs(features)
    if not pairs:
        return []

    marks: list[dict[str, object]] = []
    forward_idx = 0
    backward_idx = 0

    def _append_mark(segment: dict, color: tuple[int, int, int], role: str) -> None:
        nonlocal forward_idx, backward_idx
        label = str(segment.get("label", "")).strip()
        point = segment.get("point")
        if point is None:
            return
        point_val = float(point)
        if not np.isfinite(point_val):
            return
        slot = _slot_index(label)
        if role == "forward":
            forward_idx += 1
            ordinal = forward_idx
        else:
            backward_idx += 1
            ordinal = backward_idx
        if not label:
            if role == "forward":
                label_to_use = f"Jump F{ordinal}"
            else:
                label_to_use = f"Jump B{ordinal}"
        else:
            label_to_use = label
        marks.append({"name": label_to_use, "start": point_val, "color": color, "slot": slot})

    for pair in pairs:
        forward = pair.get("forward") or {}
        backward = pair.get("backward") or {}
        _append_mark(forward, (40, 226, 20), "forward")
        _append_mark(backward, (226, 126, 40), "backward")

    return marks


def _tempo_entries_for_xml(
    tempo_segments: np.ndarray,
    fallback_bpm: float,
) -> list[dict[str, float]]:
    """Build TEMPO entries describing beat start and bar alignment.

    - Inizio (XML): first beat timestamp (segment start).
    - Battito: relative position of the first beat w.r.t. the downbeat (segment inizio).
    - Bpm: segment bpm (fallback when invalid).
    """
    entries: list[dict[str, float]] = []
    if tempo_segments.size:
        for seg in tempo_segments:
            start, _end, bpm = seg[:3]
            if not np.isfinite(start) or start < 0:
                continue
            bpm_val = float(bpm) if np.isfinite(bpm) and bpm > 0 else float(fallback_bpm if fallback_bpm > 0 else 0.0)
            downbeat = None
            if len(seg) >= 4:
                try:
                    val = float(seg[3])
                    if np.isfinite(val) and val >= 0:
                        downbeat = val
                except Exception:
                    downbeat = None
            period = (60.0 / bpm_val) if bpm_val > 0 else None
            first_beat = float(start)
            if period and np.isfinite(downbeat):
                offset = (start - downbeat) / period
                n = int(math.ceil(offset)) if np.isfinite(offset) else 0
                first_beat = float(downbeat + n * period)
            if downbeat is None or not np.isfinite(downbeat):
                downbeat = first_beat
            battito = 1
            if period and np.isfinite(downbeat) and np.isfinite(first_beat) and period > 0:
                diff = (downbeat - first_beat) / period
                diff_beats = int(round(diff)) % 4
                battito = ((4 - diff_beats) % 4) + 1
            entries.append(
                {
                    "start": first_beat,
                    "bpm": bpm_val,
                    "battito": int(battito),
                }
            )
    if not entries:
        entries.append({"start": 0.0, "bpm": fallback_bpm if fallback_bpm > 0 else 0.0, "battito": 1})
    entries.sort(key=lambda e: e["start"])
    return entries


def _compute_average_bpm(
    track: Optional[TrackRow],
    tempo_segments: np.ndarray,
    features: dict[str, np.ndarray],
) -> float:
    if track and track.bpm:
        return float(track.bpm)
    if tempo_segments.size:
        starts = tempo_segments[:, 0]
        ends = tempo_segments[:, 1]
        bpms = tempo_segments[:, 2]
        durations = np.maximum(ends - starts, 1e-3)
        try:
            return float(np.average(bpms, weights=durations))
        except ZeroDivisionError:
            return float(np.mean(bpms))
    tempo_global = _safe_scalar(features.get("tempo_global"))
    return float(tempo_global) if tempo_global > 0 else 0.0


def _get_duration(track: Optional[TrackRow], features: dict[str, np.ndarray]) -> float:
    if track and track.duration:
        return float(track.duration)
    return float(_safe_scalar(features.get("duration_sec")))


def _safe_scalar(value: Optional[np.ndarray]) -> float:
    if value is None:
        return 0.0
    arr = np.asarray(value)
    try:
        return float(arr.item())
    except Exception:
        try:
            return float(arr.astype(float).flatten()[0])
        except Exception:
            return 0.0


def _generate_track_id(track: Optional[TrackRow]) -> str:
    if track is None:
        return "0"
    source = (track.uid or track.path or "")[:64]
    source = source or "0"
    if source.isdigit():
        return source
    try:
        digest = hashlib.md5(source.encode("utf-8"), usedforsecurity=False).hexdigest()
    except TypeError:
        digest = hashlib.md5(source.encode("utf-8")).hexdigest()
    return str(int(digest[:8], 16))


def _kind_from_path(path: Optional[Path]) -> str:
    suffix = (path.suffix if path else "").upper().lstrip(".")
    return f"{suffix} File" if suffix else ""


def _format_date(timestamp: int | float | None) -> str:
    if not timestamp:
        return ""
    try:
        return datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _file_url(audio_path: Optional[Path], track: TrackRow, resolve_path: ResolvePath) -> str:
    if audio_path and audio_path.exists():
        target = audio_path.resolve()
    else:
        target = resolve_path(track.path if track else "")
        if target is None:
            return ""
    url_path = target.as_posix()
    encoded = quote(url_path, safe="/:.")
    if encoded.startswith("/"):
        return f"file://localhost{encoded}"
    return f"file://localhost/{encoded}"


def _slot_index(label: str | None) -> Optional[int]:
    if not label:
        return None
    upper = label.upper()
    m = re.search(r"\b([A-H])\b", upper)
    if not m:
        m = re.match(r"\s*([A-H])", upper)
    if not m:
        return None
    return ord(m.group(1)) - ord("A")
