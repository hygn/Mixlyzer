"""A reference reader of JAMS-mxl-sbx1: the JAMS files Mixlyzer's Export tab writes (the writer and the format's
description: third_party/jams.py). Python's standard library alone, nothing of Mixlyzer: another program may take
this file as it is.

    python examples/jams_mxl_sbx1.py TRACK.jams [AUDIO_FILE]

prints what the file holds and, given an audio file, whether it is the audio the file was made of (the hash of its
audio packets, read by the FFmpeg on PATH: the tags may differ, the audio not).

    from jams_mxl_sbx1 import load, matches
    track = load("TRACK.jams")
    track.downbeats, track.tempo[0].bpm, track.keys[0].key, track.jump_groups()
    matches(track, "TRACK.flac")

What it reads: the annotations Mixlyzer wrote (``annotation_metadata.annotation_tools`` "Mixlyzer ..."; another
program's annotations added to the file are left alone), the extension's sandbox (``sandbox.mixlyzer``). A file of
another format, or a value that is not what the format says, is a FormatError: never a guess."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

FORMAT = "JAMS-mxl-sbx1"
SANDBOX = "mixlyzer"
TOOLS = "Mixlyzer"


class FormatError(ValueError):
    """The file is not a JAMS-mxl-sbx1 file, or holds a value the format does not allow."""


@dataclass(frozen=True)
class Beat:
    time: float
    position: int            # its place in its bar: 1 a downbeat


@dataclass(frozen=True)
class TempoRow:
    start: float
    end: float
    bpm: float               # at its start
    bpm_end: float           # at its end: a ramp goes linearly in time from bpm to it; a steady row's is its bpm

    @property
    def is_ramp(self) -> bool:
        return self.bpm_end != self.bpm


@dataclass(frozen=True)
class KeyRow:
    start: float
    end: float
    key: str                 # JAMS key_mode: "Eb:major", "C#:minor"

    @property
    def tonic(self) -> str:
        return self.key.split(":")[0]

    @property
    def minor(self) -> bool:
        return self.key.endswith(":minor")


@dataclass(frozen=True)
class Phrase:
    start: float
    end: float
    label: str               # Mixlyzer's phrase labels: INTRO, VERSE, CHORUS ...


@dataclass(frozen=True)
class CuePoint:
    time: float
    name: str
    comment: str
    color: str               # "#rrggbb"


@dataclass(frozen=True)
class JumpCue:
    time: float              # the point jumped from and to
    slot: str                # "A" ...
    name: str
    group: int               # the cues of a group jump between each other
    color: str
    passage: tuple[float, float]


@dataclass(frozen=True)
class AudioIdentity:
    packets_sha256: str      # the SHA-256 of the first audio stream's packets as FFmpeg's demuxer gives them
    packets_bytes: int
    codec: str
    sample_rate: int
    channels: str
    samples: Optional[int]
    ffmpeg: str              # the FFmpeg that read it


@dataclass(frozen=True)
class Track:
    title: str
    artist: str
    release: str
    duration: float
    app_version: str
    uid: Optional[str]       # the track in the Mixlyzer library it came from (None: left out when exported)
    path: Optional[str]      # its audio file (None: left out when exported)
    size: int
    audio: AudioIdentity
    beats: tuple[Beat, ...]
    tempo: tuple[TempoRow, ...]
    keys: tuple[KeyRow, ...]
    phrases: tuple[Phrase, ...]
    cue_points: tuple[CuePoint, ...]
    jump_cues: tuple[JumpCue, ...]

    @property
    def downbeats(self) -> tuple[float, ...]:
        return tuple(b.time for b in self.beats if b.position == 1)

    def jump_groups(self) -> dict[int, tuple[JumpCue, ...]]:
        """The JumpCUEs by group, each group's by time."""
        out: dict[int, list[JumpCue]] = {}
        for cue in self.jump_cues:
            out.setdefault(cue.group, []).append(cue)
        return {g: tuple(sorted(cues, key=lambda c: c.time)) for g, cues in sorted(out.items())}


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise FormatError(what)


def _obj(value, what: str) -> dict:
    _need(isinstance(value, dict), f"{what}: not an object")
    return value


def _list(value, what: str) -> list:
    _need(isinstance(value, list), f"{what}: not a list")
    return value


def _num(value, what: str) -> float:
    _need(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
          f"{what}: not a number")
    return float(value)


def _time(value, what: str) -> float:
    t = _num(value, what)
    _need(t >= 0.0, f"{what}: negative")
    return t


def _int(value, what: str) -> int:
    _need(isinstance(value, int) and not isinstance(value, bool), f"{what}: not an integer")
    return int(value)


def _str(value, what: str) -> str:
    _need(isinstance(value, str), f"{what}: not a string")
    return value


def _at(value, *keys):
    """``value[k1][k2]...``; None where one is missing or not an object."""
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _ours(annotation: dict) -> bool:
    tools = _at(annotation, "annotation_metadata", "annotation_tools")
    return isinstance(tools, str) and (tools == TOOLS or tools.startswith(TOOLS + " "))


def _annotation(annotations: list, namespace: str, kind: Optional[str] = None) -> dict:
    """Mixlyzer's one annotation of ``namespace`` (a blob: of ``kind``)."""
    found = [a for a in annotations if isinstance(a, dict) and a.get("namespace") == namespace and _ours(a)
             and (kind is None or _at(a, "sandbox", SANDBOX, "kind") == kind)]
    name = namespace + (f" ({kind})" if kind else "")
    _need(len(found) == 1, f"{name}: {len(found)} annotations of Mixlyzer's, not one")
    return found[0]


def _data(annotation: dict, what: str) -> list[tuple[float, float, object]]:
    """(time, duration, value) of each observation (JAMS's sparse form)."""
    out = []
    for i, o in enumerate(_list(annotation.get("data"), f"{what} data")):
        o = _obj(o, f"{what}[{i}]")
        _need({"time", "duration", "value", "confidence"} <= set(o), f"{what}[{i}]: not an observation")
        out.append((_time(o["time"], f"{what}[{i}].time"), _time(o["duration"], f"{what}[{i}].duration"), o["value"]))
    return out


def parse(document: dict) -> Track:
    """The Track of a JAMS-mxl-sbx1 document (as json.load gives it)."""
    document = _obj(document, "the file")
    mx = _obj(_at(document, "sandbox", SANDBOX), f"sandbox.{SANDBOX}")
    _need(mx.get("format") == FORMAT, f"not {FORMAT}: format {mx.get('format')!r}")
    meta = _obj(document.get("file_metadata"), "file_metadata")
    annotations = _list(document.get("annotations"), "annotations")

    beats = tuple(Beat(t, _int(v, "beat value")) for t, _d, v in _data(_annotation(annotations, "beat"), "beat"))
    tempo_annotation = _annotation(annotations, "tempo")
    tempo_data = _data(tempo_annotation, "tempo")
    bpm_end = _list(_at(tempo_annotation, "sandbox", SANDBOX, "bpm_end"), "tempo bpm_end")
    _need(len(bpm_end) == len(tempo_data), "tempo bpm_end: not one a row")
    tempo = tuple(TempoRow(t, t + d, _num(v, "tempo value"), _num(e, "tempo bpm_end"))
                  for (t, d, v), e in zip(tempo_data, bpm_end))
    keys = tuple(KeyRow(t, t + d, _str(v, "key_mode value"))
                 for t, d, v in _data(_annotation(annotations, "key_mode"), "key_mode"))
    phrases = tuple(Phrase(t, t + d, _str(v, "segment_open value"))
                    for t, d, v in _data(_annotation(annotations, "segment_open"), "segment_open"))
    cues = []
    for t, _d, v in _data(_annotation(annotations, "blob", "cue_points"), "cue_points"):
        v = _obj(v, "cue point")
        cues.append(CuePoint(t, _str(v.get("name"), "cue name"), _str(v.get("comment"), "cue comment"),
                             _str(v.get("color"), "cue color")))
    jumps = []
    for t, _d, v in _data(_annotation(annotations, "blob", "jump_cues"), "jump_cues"):
        v = _obj(v, "JumpCUE")
        passage = _list(v.get("passage"), "JumpCUE passage")
        _need(len(passage) == 2, "JumpCUE passage: not [start, end]")
        jumps.append(JumpCue(t, _str(v.get("slot"), "JumpCUE slot"), _str(v.get("name"), "JumpCUE name"),
                             _int(v.get("group"), "JumpCUE group"), _str(v.get("color"), "JumpCUE color"),
                             (_time(passage[0], "JumpCUE passage"), _time(passage[1], "JumpCUE passage"))))

    file = _obj(mx.get("file"), "sandbox file")
    audio = _obj(file.get("audio"), "sandbox file audio")
    digest = _str(audio.get("packets_sha256"), "packets_sha256")
    _need(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "packets_sha256: not a SHA-256")
    samples = audio.get("samples")
    identity = AudioIdentity(digest, _int(audio.get("packets_bytes"), "packets_bytes"),
                             _str(audio.get("codec"), "codec"), _int(audio.get("sample_rate"), "sample_rate"),
                             _str(audio.get("channels"), "channels"),
                             None if samples is None else _int(samples, "samples"), _str(audio.get("ffmpeg"), "ffmpeg"))
    uid, path = mx.get("uid"), file.get("path")
    return Track(
        title=_str(meta.get("title", ""), "title"), artist=_str(meta.get("artist", ""), "artist"),
        release=_str(meta.get("release", ""), "release"), duration=_time(meta.get("duration", 0.0), "duration"),
        app_version=_str(mx.get("app_version", ""), "app_version"),
        uid=None if uid is None else _str(uid, "uid"), path=None if path is None else _str(path, "path"),
        size=_int(file.get("size"), "size"), audio=identity, beats=beats, tempo=tempo, keys=keys, phrases=phrases,
        cue_points=tuple(cues), jump_cues=tuple(jumps))


def load(path: str) -> Track:
    """The Track of a JAMS-mxl-sbx1 file."""
    with open(path, encoding="utf-8") as f:     # the writer writes ASCII; UTF-8 takes an edited file too
        try:
            document = json.load(f)
        except json.JSONDecodeError as exc:
            raise FormatError(f"not JSON: {exc}") from exc
    return parse(document)


def packets_sha256(audio_path: str, ffmpeg: Optional[str] = None) -> Optional[str]:
    """The SHA-256 of the file's first audio stream's packets, as the format names its audio (FFmpeg copies them out:
    -map 0:a:0 -c copy -f data); None when FFmpeg cannot read it. ``ffmpeg``: its path (None: the one on PATH)."""
    exe = ffmpeg or shutil.which("ffmpeg")
    if not exe:
        raise FileNotFoundError("FFmpeg not found on PATH")
    # the file opened by FFmpeg's file protocol alone, by its absolute path: no name taken for a URL or an option
    args = [exe, "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-i", "file:" + os.path.abspath(audio_path),
            "-map", "0:a:0", "-c", "copy", "-f", "data", "-"]
    digest, size = hashlib.sha256(), 0
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        while chunk := proc.stdout.read(1 << 20):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest() if proc.returncode == 0 and size else None


def matches(track: Track, audio_path: str, ffmpeg: Optional[str] = None) -> bool:
    """Whether ``audio_path`` holds the audio ``track`` was made of: tags written since (or another file's tags) do not
    matter, the audio encoded again does."""
    return packets_sha256(audio_path, ffmpeg) == track.audio.packets_sha256


def _report(track: Track) -> str:
    lines = [f"{track.title} / {track.artist}" + (f" / {track.release}" if track.release else ""),
             f"  Mixlyzer {track.app_version}, {track.duration:.1f} s, uid {track.uid or '-'}",
             f"  file {track.path or '-'} ({track.size} bytes; {track.audio.codec} {track.audio.sample_rate} Hz "
             f"{track.audio.channels}, audio {track.audio.packets_sha256[:16]}...)",
             f"  {len(track.beats)} beats, {len(track.downbeats)} bars"]
    lines += [f"  tempo {r.start:7.2f} - {r.end:7.2f} s  {r.bpm:.2f}" + (f" -> {r.bpm_end:.2f}" if r.is_ramp else "")
              + " BPM" for r in track.tempo]
    lines += [f"  key   {k.start:7.2f} - {k.end:7.2f} s  {k.key}" for k in track.keys]
    lines.append("  phrases " + " ".join(p.label for p in track.phrases))
    lines += [f"  CUE   {c.time:7.2f} s  {c.name}" for c in track.cue_points]
    for group, cues in track.jump_groups().items():
        lines.append(f"  JumpCUE group {group}: " + ", ".join(f"{c.slot} {c.time:.2f} s" for c in cues))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):     # a console's code page (cp949, cp932) lacks some titles' characters
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    if not 1 <= len(argv) <= 2:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    try:
        track = load(argv[0])
    except (OSError, FormatError) as exc:
        print(f"{argv[0]}: {exc}", file=sys.stderr)
        return 1
    print(_report(track))
    if len(argv) == 2:
        same = matches(track, argv[1])
        print(f"{argv[1]}: {'the same audio' if same else 'other audio (or not readable)'}")
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
