from __future__ import annotations

import os
import subprocess

import numpy as np

try:
    import soundfile as sf  # lightweight header probe
except Exception:  # pragma: no cover
    sf = None


def decode_to_memmap(path: str, sr: int, ch: int) -> np.ndarray:
    """
    Decode input media into float32 PCM using FFmpeg and keep the stream in memory.
    Returns a contiguous numpy array shaped [N, ch]; no temporary file is created.
    """
    print("[Decoder] Starting FFmpeg")

    args = [
        "ffmpeg",
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

    creationflags = 0
    startupinfo = None
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
        startupinfo=startupinfo,
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
    Tries soundfile header read first; falls back to ffprobe.
    """
    if sf is not None:
        try:
            info = sf.info(path)
            if getattr(info, "samplerate", 0):
                return int(info.samplerate)
        except Exception:
            pass

    args = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate",
        "-of",
        "default=nw=1:nk=1",
        path,
    ]
    creationflags = 0
    startupinfo = None
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    try:
        out = subprocess.check_output(
            args,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            startupinfo=startupinfo,
        )
        sr_str = out.decode(errors="ignore").strip()
        sr_val = int(sr_str) if sr_str else 0
        if sr_val > 0:
            return sr_val
    except Exception:
        pass
    return 0
