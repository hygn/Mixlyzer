"""A track's audio rendered for the Export tab, run off the GUI thread (it decodes the file and takes seconds): the
whole track, with a JumpCUE's jump baked in or not (the audio up to the cue jumped from, then on from the cue
jumped to, crossfaded over JUMP_CROSSFADE frames: ui.render.jumprender), at another tempo or not (the native
engine's time stretch, as the deck plays it: core.playback.player.stretch_offline)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from analyzer.audio.decoder import decode_to_memmap, get_samplerate
from audioengine.controls import MIN_RATE
from core.playback.player import stretch_offline
from system.atomic_io import atomic_output_path
from ui.render.jumprender import jump_renderer

JUMP_CROSSFADE = 128       # frames
MAX_SECONDS = 3600         # the longest render (its frames held in memory, as float32, more than once)


class RenderError(ValueError):
    """A render refused before its work: longer than MAX_SECONDS."""


def render_track(path: str, jump: tuple[float, float] | None = None, speed: float = 1.0) -> tuple[np.ndarray, int]:
    """The audio of ``path`` (stereo float32, [frames, 2]) and its sample rate: ``jump`` (seconds jumped from, to)
    baked in, played ``speed`` times as fast (1: as it is; another, at the engine's 44.1 kHz at least: the file's
    rate below it is raised)."""
    rate = int(get_samplerate(path) or 44100)
    if speed != 1.0:
        rate = max(rate, MIN_RATE)
    pcm = np.asarray(decode_to_memmap(path, rate, 2), dtype=np.float32)
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    if pcm.shape[1] == 1:
        pcm = np.repeat(pcm, 2, axis=1)
    if jump is not None:
        last = max(0.0, (pcm.shape[0] - 1) / rate)
        start, dest = (float(np.clip(t, 0.0, last)) for t in jump)
        pcm = np.stack([jump_renderer(np.array(pcm[:, c], copy=True), rate, start, dest, JUMP_CROSSFADE)[0]
                        for c in range(pcm.shape[1])], axis=1)
    if speed != 1.0:
        if pcm.shape[0] / rate / speed > MAX_SECONDS:
            raise RenderError(f"longer than {MAX_SECONDS} s at {speed:g} times the speed")
        pcm = stretch_offline(pcm, rate, speed)
    return np.clip(pcm, -1.0, 1.0).astype(np.float32, copy=False), rate


def write_wav(path: Path, pcm: np.ndarray, rate: int) -> None:
    """``pcm`` as a 16-bit WAV at ``path``, replaced whole (never half written)."""
    with atomic_output_path(path) as temporary:
        sf.write(str(temporary), pcm, rate, format="WAV", subtype="PCM_16")
