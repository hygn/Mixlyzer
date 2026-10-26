from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
import math
from typing import Callable, Iterable, Optional
from urllib.parse import quote
from uuid import UUID
from xml.dom import minidom
from xml.etree import ElementTree as ET

import numpy as np

from core.library.records import TrackRow
from utils.labels import idx_to_labels
from utils.jump_cues import extract_jump_cue_graph
from core.tempo_rows import is_ramp_row, ramp_beat_times, segment_end_bpm
from utils.cue_points import extract_cue_points
from utils.jumpcue_colors import get_cue_point_color

ResolvePath = Callable[[str | None], Optional[Path]]

__all__ = [
    "RekordboxXmlDocument",
    "build_rekordbox_track_element",
    "rekordbox_track_key_for_row",
    "sanitize_filename",
]


def build_rekordbox_track_element(
    track: TrackRow,
    features: dict[str, np.ndarray],
    *,
    resolve_path: ResolvePath,
    audio_path: Optional[Path] = None,
    extra_attributes: dict[str, str] | None = None,
    ramp_tolerance_sec: float = 0.0,
) -> ET.Element:
    """Build the canonical Rekordbox ``TRACK`` element for every export path: the JumpCUEs as hot cues,
    the CUE points as memory cues. ``ramp_tolerance_sec``: how far a tempo ramp's steps may put a beat
    from the ramp's (0: a step per beat; ``_ramp_tempo_entries``)."""

    tempo_segments = _extract_tempo_segments(features)
    duration = _get_duration(track, features)
    average_bpm = _compute_average_bpm(track, tempo_segments, features)
    position_marks = _extract_jump_marks(features)
    memory_cues = _extract_memory_cues(features)

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

    sample_rate = int(_safe_scalar(features.get("sr")) or 44100)
    tempo_global = _safe_scalar(features.get("tempo_global"))
    if not np.isfinite(average_bpm) or average_bpm <= 0:
        average_bpm = tempo_global if tempo_global > 0 else 0.0

    bit_rate = 0
    if file_size > 0 and duration > 0:
        bit_rate = int(round((file_size * 8) / duration / 1000))

    song_title = track.title or (resolved_audio.stem if resolved_audio else "Untitled")
    track_id = _generate_track_id(track)
    tonality = ""
    if track.key is not None:
        try:
            tonality = idx_to_labels(track.key)[0]
        except Exception:
            tonality = ""

    track_attrs = {
        "TrackID": track_id,
        "Name": song_title,
        "Artist": track.artist or "",
        "Composer": "",
        "Album": track.album or "",
        "Grouping": "",
        "Genre": "",
        "Kind": _kind_from_path(
            resolved_audio or (resolve_path(track.path) if track.path else None)
        ),
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
    if extra_attributes:
        track_attrs.update({str(key): str(value) for key, value in extra_attributes.items()})
    track_elem = ET.Element("TRACK", track_attrs)

    tempo_entries = _tempo_entries_for_xml(tempo_segments, average_bpm, ramp_tolerance_sec)
    for tempo in tempo_entries:
        ts_num = int(tempo["time_sig"])
        ET.SubElement(
            track_elem,
            "TEMPO",
            Inizio=f"{tempo['start']:.3f}",
            Bpm=f"{tempo['bpm']:.2f}",
            Metro=f"{ts_num}/4",
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

    # memory cues: no slot (Num -1), in time order
    for cue in memory_cues:
        color = cue["color"]
        ET.SubElement(
            track_elem,
            "POSITION_MARK",
            Name=cue["name"],
            Type="0",
            Start=f"{cue['start']:.3f}",
            Num="-1",
            Red=str(color[0]),
            Green=str(color[1]),
            Blue=str(color[2]),
        )

    return track_elem


def rekordbox_track_key_for_row(track: TrackRow) -> str:
    uid = str(track.uid or "").strip()
    return uid or _generate_track_id(track)


def _rekordbox_track_key_for_element(track: ET.Element) -> str:
    uid = str(track.attrib.get("MixlyzerUID", "")).strip()
    return uid or str(track.attrib.get("TrackID", "")).strip()


class RekordboxXmlDocument:
    """XML document operations shared by one-track export and library sync."""

    def __init__(self, root: ET.Element, collection: ET.Element) -> None:
        self.root = root
        self.collection = collection
        self._reindex()

    @classmethod
    def new(cls) -> "RekordboxXmlDocument":
        root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
        ET.SubElement(
            root,
            "PRODUCT",
            Name="Mixlyzer",
            Version="1.0.0",
            Company="Mixlyzer",
        )
        collection = ET.SubElement(root, "COLLECTION", Entries="0")
        return cls(root, collection)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        force_new: bool = False,
    ) -> "RekordboxXmlDocument":
        xml_path = Path(path)
        if not force_new and xml_path.exists():
            try:
                root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
            except (OSError, ET.ParseError, UnicodeError) as exc:
                raise ValueError(f"Cannot read Rekordbox XML: {xml_path}") from exc
            collection = root.find("COLLECTION")
            if collection is None:
                raise ValueError(f"Rekordbox XML has no COLLECTION: {xml_path}")
            return cls(root, collection)
        return cls.new()

    def _reindex(self) -> None:
        self.existing: dict[str, ET.Element] = {}
        for track in self.collection.findall("TRACK"):
            key = _rekordbox_track_key_for_element(track)
            if key:
                self.existing[key] = track

    def existing_track(self, key: str) -> ET.Element | None:
        return self.existing.get(str(key))

    def remove_track(self, key: str) -> None:
        normalized_key = str(key)
        removed = False
        for track in list(self.collection.findall("TRACK")):
            if _rekordbox_track_key_for_element(track) == normalized_key:
                self.collection.remove(track)
                removed = True
        self.existing.pop(normalized_key, None)
        if removed:
            self._update_entry_count()

    def append_track(self, track: ET.Element) -> None:
        self.collection.append(track)
        key = _rekordbox_track_key_for_element(track)
        if key:
            self.existing[key] = track
        self._update_entry_count()

    def replace_tracks(self, tracks: Iterable[ET.Element]) -> None:
        nodes = list(tracks)
        self.collection.clear()
        self.collection.extend(nodes)
        self._update_entry_count()
        self._reindex()

    @property
    def entry_count(self) -> int:
        return len(self.collection.findall("TRACK"))

    def _update_entry_count(self) -> None:
        self.collection.set("Entries", str(self.entry_count))

    def to_xml(self) -> str:
        _strip_whitespace_only_nodes(self.root)
        _strip_invalid_characters(self.root)
        xml_bytes = ET.tostring(self.root, encoding="utf-8")
        pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ")
        if not pretty.startswith("<?xml"):
            pretty = '<?xml version="1.0" encoding="UTF-8"?>\n' + pretty
        return pretty


def _strip_whitespace_only_nodes(element: ET.Element) -> None:
    if element.text is not None and not element.text.strip():
        element.text = None
    for child in list(element):
        _strip_whitespace_only_nodes(child)
        if child.tail is not None and not child.tail.strip():
            child.tail = None


# what XML 1.0 cannot hold (control characters but tab / line feed / carriage return, lone surrogates, U+FFFE /
# U+FFFF): a tag holding one (a title, a comment) would make the whole document unreadable
_XML_INVALID = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff￾￿]")


def _strip_invalid_characters(element: ET.Element) -> None:
    for key, value in element.attrib.items():
        if _XML_INVALID.search(value):
            element.set(key, _XML_INVALID.sub("", value))
    for name in ("text", "tail"):
        value = getattr(element, name)
        if value and _XML_INVALID.search(value):
            setattr(element, name, _XML_INVALID.sub("", value))
    for child in element:
        _strip_invalid_characters(child)


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r'[\\/:*?"<>|]+', "_", name).strip()
    return sanitized or "track"


def _extract_tempo_segments(features: dict[str, np.ndarray]) -> np.ndarray:
    """Return tempo segments as (N,3|4|5|6) array.

    Format:
      [start, end, bpm] | [start, end, bpm, inizio] | [start, end, bpm, inizio, ts_num]
      | [start, end, bpm, inizio, ts_num, bpm_end] (bpm_end != bpm: linear tempo ramp)
    """
    raw = features.get("tempo_segments")
    if raw is None:
        return np.empty((0, 3), dtype=float)
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 1:
        for width in (6, 5, 4, 3):
            if arr.size % width == 0:
                arr = arr.reshape((-1, width))
                break
        else:
            return np.empty((0, 3), dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return arr[:, : min(arr.shape[1], 6)]
    return np.empty((0, 3), dtype=float)


def _extract_jump_marks(features: dict[str, np.ndarray]) -> list[dict[str, object]]:
    """Extract graph-wise jump cue marks using the normalized jump_cues_np format."""

    cues, _links = extract_jump_cue_graph(features)
    if not cues:
        return []

    marks: list[dict[str, object]] = []
    for cue in cues:
        label = str(cue.get("label", "")).strip()
        point = cue.get("point")
        if point is None:
            continue
        point_val = float(point)
        if not np.isfinite(point_val):
            continue
        color = cue["color"]
        slot = _slot_index(label)
        comment = str(cue.get("comment", "") or "").strip()
        marks.append(
            {
                "name": comment or label or "",
                "start": point_val,
                "color": color,
                "slot": slot,
            }
        )

    return marks


def _extract_memory_cues(features: dict[str, np.ndarray]) -> list[dict[str, object]]:
    """The CUE points as memory cues: their names and colours, in time order."""
    return [
        {
            "name": str(point["label"]).strip(),
            "start": float(point["time_sec"]),
            "color": get_cue_point_color(int(point["color_idx"])),
        }
        for point in extract_cue_points(features)
    ]


def _tempo_entries_for_xml(
    tempo_segments: np.ndarray,
    fallback_bpm: float,
    ramp_tolerance_sec: float = 0.0,
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
            if len(seg) >= 6 and is_ramp_row(seg) and np.isfinite(seg[3]):
                entries.extend(_ramp_tempo_entries(seg, ramp_tolerance_sec))
                continue
            bpm_val = (
                float(bpm)
                if np.isfinite(bpm) and bpm > 0
                else float(fallback_bpm if fallback_bpm > 0 else 0.0)
            )
            downbeat = None
            if len(seg) >= 4:
                try:
                    val = float(seg[3])
                    if np.isfinite(val) and val >= 0:
                        downbeat = val
                except Exception:
                    downbeat = None
            ts_num = 4
            if len(seg) >= 5:
                try:
                    ts_val = int(round(float(seg[4])))
                    if ts_val >= 1:
                        ts_num = ts_val
                except Exception:
                    ts_num = 4
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
                diff_beats = int(round(diff)) % ts_num
                battito = ((ts_num - diff_beats) % ts_num) + 1
            entries.append(
                {
                    "start": first_beat,
                    "bpm": bpm_val,
                    "battito": int(battito),
                    "time_sig": int(ts_num),
                }
            )
    if not entries:
        entries.append(
            {
                "start": 0.0,
                "bpm": fallback_bpm if fallback_bpm > 0 else 0.0,
                "battito": 1,
                "time_sig": 4,
            }
        )
    entries.sort(key=lambda e: e["start"])
    return entries


def _ramp_tempo_entries(seg, tolerance_sec: float = 0.0) -> list[dict[str, float]]:
    """A tempo ramp as steps of steady tempo (Rekordbox has no ramps), one TEMPO entry each. A step
    starts on a beat of the ramp and its BPM reaches the next step's first beat; it takes in the beats
    after its first while every beat it gives stays within ``tolerance_sec`` of the ramp's, one beat at
    least (0: a step per beat). Measured as the XML has them: Inizio in ms, the BPM rounded down to
    0.01 (a step's last beat then never falls just before the next step's first). Battito counts the
    bar from the ramp's downbeat (``inizio``)."""
    start, end, bpm, inizio = (float(v) for v in seg[:4])
    ts_num = int(round(float(seg[4]))) if np.isfinite(seg[4]) and seg[4] >= 1 else 4
    end_bpm = segment_end_bpm(seg)
    # The beats of the ramp plus the first one after it (its tempo line runs on past ``end``).
    reach = end + 2.0 * 60.0 / max(min(bpm, end_bpm), 1e-6)
    line_end = end_bpm + (end_bpm - bpm) * (reach - end) / (end - start) if end > start else end_bpm
    if line_end <= 0:
        line_end = end_bpm
    beats = ramp_beat_times(start, reach, bpm, line_end, inizio, reach)
    count = int(np.count_nonzero(beats < end))
    if count == 0 or beats.size <= count:
        return []
    ref = int(np.argmin(np.abs(beats - inizio)))
    written = np.round(beats[: count + 1], 3)   # the steps' Inizio as written
    out = []
    i = 0
    while i < count:
        j = i + 1
        step_bpm = _step_bpm(written, i, j)
        while j < count:
            longer = _step_bpm(written, i, j + 1)
            if longer <= 0:
                break
            given = written[i] + np.arange(j + 1 - i) * (60.0 / longer)
            if float(np.max(np.abs(given - beats[i : j + 1]))) > tolerance_sec:
                break
            j, step_bpm = j + 1, longer
        if step_bpm > 0:
            out.append(
                {
                    "start": float(written[i]),
                    "bpm": step_bpm,
                    "battito": int((i - ref) % ts_num) + 1,
                    "time_sig": ts_num,
                }
            )
        i = j
    return out


def _step_bpm(written: np.ndarray, first: int, after: int) -> float:
    """The BPM of a step from beat ``first`` reaching beat ``after`` (Inizio as written), rounded down
    to 0.01 as the XML writes it."""
    span = float(written[after] - written[first])
    return math.floor(6000.0 * (after - first) / span) / 100.0 if span > 0 else 0.0


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
        if tempo_segments.shape[1] >= 6:
            # A ramp's mean tempo over its span.
            bpms = 0.5 * (bpms + np.asarray([segment_end_bpm(row) for row in tempo_segments]))
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
    uid = str(track.uid or "").strip()
    if not uid:
        raise ValueError("Track uid is required to generate Rekordbox TrackID.")
    return str(UUID(uid).int)


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
