from __future__ import annotations

import os
import re
import subprocess
import threading

import numpy as np

try:
    import soundfile as sf  # lightweight header probe
except Exception:  # pragma: no cover
    sf = None


_ffmpeg_warm_lock = threading.Lock()
_ffmpeg_warmed = False


def _ffmpeg_command() -> str:
    root_ffmpeg = os.path.join(os.getcwd(), "ffmpeg.exe")
    if os.name == "nt" and os.path.exists(root_ffmpeg):
        return root_ffmpeg
    return "ffmpeg"


def _windows_subprocess_kwargs() -> dict:
    creationflags = 0
    startupinfo = None
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    return {
        "creationflags": creationflags,
        "startupinfo": startupinfo,
    }


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
                **_windows_subprocess_kwargs(),
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
        "-i",
        path,
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

    popen_kwargs = _windows_subprocess_kwargs()

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
        "-i",
        path,
    ]
    popen_kwargs = _windows_subprocess_kwargs()
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
