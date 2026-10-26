"""A track's audio as the analyses read it: decoded by FFmpeg (SoundFile when FFmpeg fails), at the rate asked for."""
from pathlib import Path

import numpy as np
import soundfile as sf

from analyzer.audio.decoder import decode_to_memmap
from analyzer.audio.resample import resample_audio
from analyzer.errors import DECODE_FAILED, AnalysisError


def fast_load(path: str, target_sr, stereo: bool = False) -> np.ndarray:
    """Decode audio to float32. Mono ``(N,)`` by default, stereo ``(N, 2)`` when
    ``stereo=True`` (channel-duplicated if the source is mono)."""
    path = Path(path)
    sr_req = int(target_sr) if target_sr else 44100
    ch = 2 if stereo else 1
    try:
        pcm = decode_to_memmap(path.as_posix(), sr_req, ch=ch)
        arr = np.array(pcm, dtype=np.float32, copy=False)
        y = arr.reshape(-1, 2) if stereo else arr.reshape(-1)
        sr = sr_req
    except Exception:
        try:
            y, sr = sf.read(path.as_posix(), dtype="float32", always_2d=False)
        except Exception as exc:
            raise AnalysisError(DECODE_FAILED, path=str(path)) from exc
        if stereo:
            if y.ndim == 1:
                y = np.stack([y, y], axis=1)
            elif y.shape[1] == 1:
                y = np.repeat(y, 2, axis=1)
            else:
                y = y[:, :2]
        elif y.ndim > 1:
            y = y.mean(axis=1).astype(np.float32)
    if target_sr and sr != target_sr:
        y = resample_audio(
            y,
            sr,
            target_sr,
            axis=0 if stereo else -1,
            quality="HQ",
        )
    return np.ascontiguousarray(y, dtype=np.float32)
