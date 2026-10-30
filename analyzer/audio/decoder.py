from __future__ import annotations

import os
import re
import subprocess
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from analyzer.runtime import IS_WINDOWS, hidden_subprocess_kwargs

try:
    import soundfile as sf  # lightweight header probe
except Exception:  # pragma: no cover
    sf = None


_ffmpeg_warm_lock = threading.Lock()
_ffmpeg_warmed = False
# an FFmpeg in the folder holding the analyzer (the app's folder; a package's _internal), else the one on PATH
_BUNDLED_FFMPEG = Path(__file__).resolve().parents[2] / ("ffmpeg.exe" if IS_WINDOWS else "ffmpeg")


def _input(path: str) -> list[str]:
    """FFmpeg's input options for the local file ``path``: opened by its file protocol alone, so a name FFmpeg
    would take for another protocol's URL (``pipe:1.mp3``, ``http:...``, ``concat:...`` on Linux and macOS) is still
    the file, and a playlist inside it (HLS, a concat list) opens no URL; absolute, so a name starting with ``-`` is
    no option. FFmpeg opens a long Windows path itself (its extended prefix)."""
    return ["-protocol_whitelist", "file", "-i", "file:" + os.path.abspath(path)]


@lru_cache(maxsize=1)
def _ffmpeg_command() -> str:
    """FFmpeg by its full path: the app's, else the first on PATH (absolute entries only: a bare name or a
    relative entry would take one from the current folder). The app's path when there is none (not found)."""
    def runnable(p: Path) -> bool:
        return p.is_file() and (IS_WINDOWS or os.access(p, os.X_OK))

    if runnable(_BUNDLED_FFMPEG):
        return str(_BUNDLED_FFMPEG)
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        folder = folder.strip().strip('"')
        if folder and os.path.isabs(folder) and runnable(Path(folder) / _BUNDLED_FFMPEG.name):
            return str(Path(folder) / _BUNDLED_FFMPEG.name)
    return str(_BUNDLED_FFMPEG)


def warm_ffmpeg_decoder() -> None:
    """
    Warm the external FFmpeg decoder path once per process.
    This pays process startup / binary paging cost during app startup rather
    than on the first real track decode.
    """
    global _ffmpeg_warmed
    with _ffmpeg_warm_lock:
        if _ffmpeg_warmed:
            return
        print("[Decoder] Warming FFmpeg")
        args = [
            _ffmpeg_command(),
            "-hide_banner",
            "-nostats",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            "0.05",
            "-f",
            "null",
            "-",
        ]
        try:
            proc = subprocess.run(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                **hidden_subprocess_kwargs(),
            )
            if proc.returncode == 0:
                _ffmpeg_warmed = True
                print("[Decoder] FFmpeg warm complete")
            else:
                print(f"[Decoder] FFmpeg warm failed rc={proc.returncode}")
        except Exception as exc:
            print(f"[Decoder] FFmpeg warm exception: {exc}")


def decode_to_memmap(path: str, sr: int, ch: int) -> np.ndarray:
    """
    Decode input media into float32 PCM using FFmpeg and keep the stream in memory.
    Returns a contiguous numpy array shaped [N, ch]; no temporary file is created.
    """
    print("[Decoder] Starting FFmpeg")

    args = [
        _ffmpeg_command(),
        "-hide_banner",
        "-nostats",
        "-v",
        "error",
        *_input(path),
        "-vn",
        "-sn",
        "-map",
        "a:0",
        "-ac",
        str(ch),
        "-ar",
        str(sr),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]

    popen_kwargs = hidden_subprocess_kwargs()

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **popen_kwargs,
    )
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        err_msg = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"FFmpeg decode failed ({proc.returncode}): {err_msg}")

    if not stdout:
        raise RuntimeError("FFmpeg produced no PCM data.")

    data = np.frombuffer(stdout, dtype="<f4")
    if data.size % ch != 0:
        data = data[: (data.size // ch) * ch]

    pcm = data.reshape(-1, ch)
    return np.ascontiguousarray(pcm, dtype=np.float32)


COVER_MAX_BYTES = 32 << 20   # a cover larger than this is not read (no picture is that big)


def _ffmetadata(text: str) -> dict[str, str]:
    """The global section of FFmpeg's ffmetadata output: key (lower case) -> value; ``\\`` escapes ``=``, ``;``,
    ``#``, ``\\`` and the line end."""
    out: dict[str, str] = {}
    if not text.startswith(";FFMETADATA"):
        return out
    body = text.split("\n", 1)[1] if "\n" in text else ""
    key, buf, escaped, comment, at_start = None, [], False, False, True
    for ch in body + "\n":
        if comment:
            comment = ch != "\n"
            at_start = not comment
            continue
        if escaped:
            buf.append(ch)
            escaped = at_start = False
        elif ch == "\\":
            escaped = True
        elif ch == "\n":
            if key is not None:
                out.setdefault(key.strip().lower(), "".join(buf))
            key, buf, at_start = None, [], True
        elif at_start and ch in ";#":
            comment = True
        elif at_start and ch == "[":   # a stream's or a chapter's section: the global one is over
            break
        elif ch == "=" and key is None:
            key, buf = "".join(buf), []
        else:
            buf.append(ch)
            at_start = False
    return out


def read_metadata(path: str) -> dict[str, str]:
    """The file's tags as FFmpeg reads them (for the formats core.tags.writer does not read): its global
    metadata, else its first audio stream's (Ogg keeps the comments there); key (FFmpeg's, lower case) ->
    value. {} when FFmpeg cannot read the file."""
    out: dict[str, str] = {}
    for source in ("0", "0:s:a:0"):
        args = [_ffmpeg_command(), "-hide_banner", "-nostats", "-v", "error", *_input(path),
                "-map_metadata", source, "-f", "ffmetadata", "pipe:1"]
        try:
            proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
                                  timeout=30, **hidden_subprocess_kwargs())
        except (OSError, subprocess.TimeoutExpired):
            return out
        if proc.returncode != 0:
            return out
        for key, value in _ffmetadata(proc.stdout.decode("utf-8", "replace")).items():
            out.setdefault(key, value)
    return out


def read_cover(path: str) -> bytes | None:
    """The file's cover picture as stored (its first picture stream: ID3 APIC, FLAC PICTURE, MP4 covr, ...), as
    FFmpeg reads it; None when it has none or it is over COVER_MAX_BYTES."""
    args = [_ffmpeg_command(), "-hide_banner", "-nostats", "-v", "error", *_input(path),
            "-map", "0:v:0", "-c", "copy", "-frames:v", "1", "-f", "image2pipe", "pipe:1"]
    try:   # one packet copied: no more than the file holds
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False, timeout=30,
                              **hidden_subprocess_kwargs())
    except (OSError, subprocess.TimeoutExpired):
        return None
    data = proc.stdout
    if proc.returncode != 0 or not data or len(data) > COVER_MAX_BYTES:
        return None
    return data


def get_samplerate(path: str) -> int:
    """
    Probe media sample rate without full decode.
    Tries soundfile header read first; falls back to ffmpeg stderr parsing.
    """
    if sf is not None:
        try:
            info = sf.info(path)
            if getattr(info, "samplerate", 0):
                return int(info.samplerate)
        except Exception:
            pass

    sample_rate, _duration_sec = _probe_audio_stream(path)
    return sample_rate


def get_total_samples(path: str) -> int:
    """
    Probe total audio sample frames without full decode.
    Tries soundfile header read first; falls back to ffmpeg stderr parsing.
    """
    if sf is not None:
        try:
            info = sf.info(path)
            frames = int(getattr(info, "frames", 0) or 0)
            if frames > 0:
                return frames
        except Exception:
            pass

    sample_rate, duration_sec = _probe_audio_stream(path)
    if sample_rate <= 0 or duration_sec <= 0.0:
        return 0
    total = int(round(duration_sec * sample_rate))
    return total if total > 0 else 0


def _probe_audio_stream(path: str) -> tuple[int, float]:
    args = [
        _ffmpeg_command(),
        "-hide_banner",
        *_input(path),
    ]
    popen_kwargs = hidden_subprocess_kwargs()
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            **popen_kwargs,
        )
        text = (proc.stderr or b"").decode(errors="ignore")
        if not text:
            return 0, 0.0

        sample_rate_match = re.search(r"(\d+)\s*Hz", text, flags=re.IGNORECASE)
        if not sample_rate_match:
            return 0, 0.0
        sample_rate = int(sample_rate_match.group(1))

        duration_match = re.search(
            r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if not duration_match:
            return sample_rate, 0.0
        hours = int(duration_match.group(1))
        minutes = int(duration_match.group(2))
        seconds = float(duration_match.group(3))
        duration_sec = (hours * 3600.0) + (minutes * 60.0) + seconds
        return sample_rate, duration_sec
    except Exception:
        return 0, 0.0
