from __future__ import annotations

import os
import subprocess
import threading

import numpy as np

try:
    import soundfile as sf  # lightweight header probe
except Exception:  # pragma: no cover
    sf = None


_ffmpeg_warm_lock = threading.Lock()
_ffmpeg_warmed = False


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
            "ffmpeg",
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
    popen_kwargs = _windows_subprocess_kwargs()
    try:
        out = subprocess.check_output(
            args,
            stderr=subprocess.STDOUT,
            **popen_kwargs,
        )
        sr_str = out.decode(errors="ignore").strip()
        sr_val = int(sr_str) if sr_str else 0
        if sr_val > 0:
            return sr_val
    except Exception:
        pass
    return 0
