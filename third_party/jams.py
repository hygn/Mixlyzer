"""A track's marks as a JAMS file (JSON Annotated Music Specification, 0.3: https://github.com/marl/jams), for the
Export tab: the music analysis tools' format, in the extension JAMS-mxl-sbx1 (FORMAT) Mixlyzer writes. A JAMS reader
reads it as any JAMS file; what JAMS has no namespace for is in its own sandboxes and ``blob`` annotations. Its
reference reader, the standard library's alone for other programs to take: examples/jams_mxl_sbx1.py.

- ``file_metadata``: the track's title, artist, album (``release``) and length.
- ``annotations``, each with ``annotation_metadata.annotation_tools`` "Mixlyzer VERSION":
  - ``beat``: each beat of the grid, its value its place in the bar (1: a downbeat; the bars as the editor shows
    them, analyzer.music.beat_geometry; the beats before the first downbeat counted back from it in bars of the
    first bar's length);
  - ``tempo``: each tempo row, its BPM at its start (confidence 1.0); the annotation's sandbox
    ``mixlyzer.bpm_end``, row by row, its BPM at its end (a ramp's; a steady row's its BPM);
  - ``key_mode``: each key row ("A:minor"; an enharmonic key spelled as its signature, the one with fewer
    accidentals: Db, Eb, Ab, Bb, F# major, C#, D#, G#, Bb minor);
  - ``segment_open``: each phrase, its label as stored (INTRO, VERSE, CHORUS ...);
  - ``blob``, its sandbox ``mixlyzer.kind`` "cue_points": each CUE point at its time, its value
    {"name", "comment", "color": "#rrggbb"};
  - ``blob``, ``mixlyzer.kind`` "jump_cues": each JumpCUE at its point, its value {"slot": "A", "name",
    "group": the group's index (the cues of a group jump between each other), "color", "passage": [start, end]}.
  Every observation's duration is 0 but a tempo row's, a key row's and a phrase's; confidence null but a tempo's.
- ``sandbox.mixlyzer``: ``format`` (FORMAT), ``app_version``, ``uid`` (the track in the library; left out when
  not asked for), and ``file``, the audio file itself: ``path`` (left out when not asked for), ``size`` (bytes)
  and ``audio``, its audio as encoded
  (analyzer.audio.decoder.AudioPacketHash: ``packets_sha256``, the SHA-256 of its first audio stream's packets as
  FFmpeg's demuxer gives them, which tags written do not change; ``packets_bytes``; ``codec``, ``sample_rate``,
  ``channels`` as FFmpeg names them; ``samples``, its length in samples as the library has it, null when unknown;
  ``ffmpeg``, the version that read it).

Times in seconds exactly as the library stores them: the beats, the tempo rows and the key rows float64, the
phrases, the CUE points and the JumpCUEs float32, each written as its value exactly (a float32 widened: 88.96031188964844;
a deck's unsaved edit as saving would store it). A row or a mark with a time that is not a number is left out. The
text is ASCII (jams_text)."""
from __future__ import annotations

import json
from typing import Iterable, Optional

import numpy as np

from analyzer.audio.decoder import AudioPacketHash
from analyzer.music.beat_geometry import downbeat_beat_indices
from analyzer.music.key_names import key_accidentals
from analyzer.music.marks import CuePoint, JumpCueGroup, Phrase
from analyzer.music.tempo_rows import segment_end_bpm
from core.marks.colors import get_cue_point_color, get_jumpcue_pair_color
from core.marks.phrases import UNASSIGNED

FORMAT = "JAMS-mxl-sbx1"
JAMS_VERSION = "0.3.5"       # the JAMS schema version written
SANDBOX = "mixlyzer"         # the sandboxes' key

_SHARPS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLATS = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")


def key_mode_value(pitch: int, minor: bool) -> str:
    """A key as JAMS's key_mode namespace writes it ("Eb:major", "C#:minor")."""
    pc = int(pitch) % 12
    names = _FLATS if key_accidentals(pc + 12 * int(bool(minor))) < 0 else _SHARPS
    return f"{names[pc]}:{'minor' if minor else 'major'}"


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _finite(*values) -> bool:
    return all(np.isfinite(float(v)) for v in values)


def _f32(value: float) -> float:
    """A time the library stores in float32 (the phrases, the CUE points, the JumpCUEs: core.analysis_library) as
    that float32 exactly, widened (88.96031188964844): a reader takes it as the very value stored, equal to a beat
    it sits on where the beat has that value. Its shortest decimal (88.96031) is that float32 only to a float32
    reader: a JSON reader's float64 of it is another time."""
    return float(np.float32(value))


def beat_positions(beats: np.ndarray, tempo_segments) -> np.ndarray:
    """Each beat's place in its bar (1: a downbeat); the beats before the first downbeat counted back from it in
    bars of the first bar's length."""
    out = np.zeros(beats.size, dtype=np.int64)
    downs = np.asarray(downbeat_beat_indices(beats, tempo_segments), dtype=np.int64)
    if beats.size == 0:
        return out
    if downs.size == 0:
        return np.arange(beats.size, dtype=np.int64) % 4 + 1
    first = int(downs[1] - downs[0]) if downs.size > 1 else 4
    for i in range(int(downs[0])):
        out[i] = first - (int(downs[0]) - i - 1) % first
    for k, d in enumerate(downs):
        stop = int(downs[k + 1]) if k + 1 < downs.size else beats.size
        out[int(d):stop] = np.arange(1, stop - int(d) + 1)
    return out


def _observation(time: float, duration: float, value, confidence=None) -> dict:
    return {"time": float(time), "duration": max(0.0, float(duration)), "value": value, "confidence": confidence}


def _annotation(namespace: str, data: list[dict], tools: str, sandbox: Optional[dict] = None) -> dict:
    out = {"namespace": namespace, "annotation_metadata": {"annotation_tools": tools}, "data": data}
    if sandbox is not None:
        out["sandbox"] = {SANDBOX: sandbox}
    return out


def marks_of(held, duration_sec: float = 0.0) -> dict:
    """The marks `jams_document` takes, of what holds a track's marks (a StoredAnalysis; the deck's DataModel, as
    edited); its length, else ``duration_sec``."""
    return dict(beats=held.beats, tempo_segments=held.tempo_segments, key_segments=held.key_segments,
                phrases=list(held.phrases), cue_points=list(held.cue_points), jump_cues=list(held.jump_cues),
                duration_sec=float(held.duration_sec or duration_sec or 0.0))


def jams_document(*, beats, tempo_segments, key_segments, phrases: Iterable[Phrase],
                  cue_points: Iterable[CuePoint], jump_cues: Iterable[JumpCueGroup], duration_sec: float,
                  title: str, artist: str, album: str, path: Optional[str], file_size: int, uid: Optional[str],
                  audio: AudioPacketHash, total_samples: Optional[int], app_version: str) -> dict:
    """The JAMS document (FORMAT) of a track's marks (beats; tempo rows [start, end, bpm, inizio, ts, bpm_end]; key
    rows [pitch, minor, start, end]; phrases, UNASSIGNED left out; CUE points; JumpCUE groups) and its file;
    ``path`` / ``uid`` None: left out."""
    tools = f"Mixlyzer {app_version}"
    grid = np.asarray(beats, dtype=float).ravel()
    grid = np.sort(grid[np.isfinite(grid)])
    rows = np.asarray(tempo_segments, dtype=float).reshape(-1, np.shape(tempo_segments)[-1]) \
        if np.size(tempo_segments) else np.empty((0, 6))
    positions = beat_positions(grid, rows)
    beat_data = [_observation(t, 0.0, int(p)) for t, p in zip(grid, positions)]

    tempo_rows = sorted((r for r in rows if r.size >= 3 and _finite(r[0], r[1], r[2])), key=lambda r: r[0])
    tempo_data = [_observation(r[0], r[1] - r[0], float(r[2]), 1.0) for r in tempo_rows]
    bpm_end = [segment_end_bpm(r) for r in tempo_rows]

    keys = np.asarray(key_segments, dtype=float).reshape(-1, 4) if np.size(key_segments) else np.empty((0, 4))
    keys = keys[np.all(np.isfinite(keys), axis=1)]
    key_data = [_observation(start, end - start, key_mode_value(int(pitch), bool(minor)))
                for pitch, minor, start, end in keys[np.argsort(keys[:, 2], kind="stable")]]

    phrase_data = [_observation(_f32(p.start), _f32(p.end) - _f32(p.start), p.label)
                   for p in sorted(phrases, key=lambda p: p.start)
                   if p.label and p.label != UNASSIGNED and _finite(p.start, p.end)]

    cue_data = [_observation(_f32(c.time_sec), 0.0, {"name": c.label, "comment": c.comment,
                                                     "color": _hex(get_cue_point_color(c.color_idx))})
                for c in sorted(cue_points, key=lambda c: c.time_sec) if _finite(c.time_sec)]

    jumps = []
    for gi, group in enumerate(jump_cues):
        for cue in group.cues:
            if _finite(cue.point, cue.start, cue.end):
                jumps.append(_observation(_f32(cue.point), 0.0, {
                    "slot": cue.label, "name": cue.comment, "group": gi,
                    "color": _hex(get_jumpcue_pair_color(group.color_idx)),
                    "passage": [_f32(cue.start), _f32(cue.end)]}))
    jumps.sort(key=lambda o: o["time"])

    file = {
        **({"path": path} if path is not None else {}),
        "size": int(file_size),
        "audio": {
            "packets_sha256": audio.sha256,
            "packets_bytes": int(audio.size),
            "codec": audio.codec,
            "sample_rate": int(audio.sample_rate),
            "channels": audio.channels,
            "samples": int(total_samples) if total_samples else None,
            "ffmpeg": audio.ffmpeg,
        },
    }
    return {
        "file_metadata": {"title": title or "", "artist": artist or "", "release": album or "",
                          "duration": max(0.0, float(duration_sec or 0.0)), "jams_version": JAMS_VERSION,
                          "identifiers": {}},
        "annotations": [
            _annotation("beat", beat_data, tools),
            _annotation("tempo", tempo_data, tools, {"bpm_end": bpm_end}),
            _annotation("key_mode", key_data, tools),
            _annotation("segment_open", phrase_data, tools),
            _annotation("blob", cue_data, tools, {"kind": "cue_points"}),
            _annotation("blob", jumps, tools, {"kind": "jump_cues"}),
        ],
        "sandbox": {SANDBOX: {
            "format": FORMAT,
            "app_version": app_version,
            **({"uid": uid} if uid is not None else {}),
            "file": file,
        }},
    }


def jams_text(document: dict) -> str:
    """The document as the file's text, indented; ASCII, every other character escaped (\\uXXXX) as the jams
    package writes it: it reads a file in the system's code page, which UTF-8 text breaks (cp949, cp932)."""
    return json.dumps(document, ensure_ascii=True, indent=2) + "\n"
