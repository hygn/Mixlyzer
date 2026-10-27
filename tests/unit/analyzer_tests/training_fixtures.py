"""Synthetic tracks, frames and caches for the training tests (no library, no audio): the tracks the library's owner
would give a Reoptimize run (analyzer.training.library_tracks), each with an empty stand-in audio file in a temporary
folder, and the optimizers' caches written as their own builders would write them."""
from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

import numpy as np

from analyzer.beat.frame_features import FrameFeatures
from analyzer.training import library_tracks
from analyzer.training.beat_frames import BEAT_FRAME_CACHE_FORMAT, _EXACT, _SCALARS
from analyzer.training.library_tracks import Annotation

SR = 22050
HOP = 256
FRAME_SEC = HOP / SR
_REPO = Path(__file__).resolve().parents[3]


def shipped(name: str) -> str:
    """A shipped model (assets/weights/default_<name>), only ever read."""
    return str(_REPO / "assets" / "weights" / f"default_{name}")


def uid_of(i: int) -> str:
    """A fixed, sortable UUIDv4 for the i-th synthetic track."""
    return str(UUID(int=(i + 1) | (4 << 76) | (2 << 62)))


def steady_beats(duration: float, bpm: float, first: float = 0.25) -> np.ndarray:
    return np.arange(first, duration - 0.05, 60.0 / bpm)


def tempo_rows(beats: np.ndarray, duration: float, bpm: float, ts: int = 4) -> np.ndarray:
    return np.asarray([[0.0, duration, bpm, float(beats[0]), float(ts), bpm]])


def synthetic_frames(n_frames: int, beats: np.ndarray, seed: int) -> FrameFeatures:
    """Frame features with onset pulses on the beats (and noise elsewhere)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) * FRAME_SEC
    pulse = np.zeros(n_frames, np.float32)
    idx = np.round(np.asarray(beats) / FRAME_SEC).astype(int)
    pulse[idx[(idx >= 0) & (idx < n_frames)]] = 1.0

    def m(rows):
        return rng.random((rows, n_frames)).astype(np.float32)
    contour = np.where(rng.random(n_frames) < 0.5, -1, rng.integers(0, 5, n_frames)).astype(np.int64)
    return FrameFeatures(
        sample_rate=SR, hop_length=HOP, duration_sec=float(n_frames * FRAME_SEC), frame_times=t,
        chroma=m(12), bass_chroma=m(12), mfcc=m(20), tonnetz=m(6), onset_bands=m(3) + pulse,
        rms_db=(-30 + 10 * rng.random(n_frames)).astype(np.float32), band_ratios=m(3), flatness=m(1)[0],
        librosa_onset=m(1)[0] + 2 * pulse, harmonic_onset=m(1)[0], percussive_bands=m(3) + pulse,
        percussive_share=m(1)[0], melody_chroma=m(12), melody_onset=m(1)[0],
        melody_pitch=np.where(contour >= 0, 60.0, np.nan), melody_contour=contour,
    )


def write_beat_frames(cache_dir: Path, track: dict, frames: FrameFeatures) -> Path:
    """A beat frame cache as analyzer.training.beat_frames.build_frames writes it."""
    values = {}
    for field in dataclasses.fields(FrameFeatures):
        v = getattr(frames, field.name)
        values[field.name] = np.asarray(v) if field.name in _SCALARS + _EXACT else np.asarray(v, np.float16)
    path = Path(cache_dir) / f"{track['uid']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, cache_format=np.asarray(BEAT_FRAME_CACHE_FORMAT),
                        audio_mtime_ns=np.int64(Path(track["audio_path"]).stat().st_mtime_ns), **values)
    return path


def given_track(folder: Path, i: int, annotation: Annotation | None, *, audio: bool = True, **extra) -> dict:
    """One track as the library's owner gives it (core.analysis_library.jobs.training_tracks)."""
    path = Path(folder) / f"track{i}.wav"
    if audio:
        path.write_bytes(b"RIFF")
    return {"uid": uid_of(i), "title": f"Song {i}", "artist": f"Artist {i}", "audio_path": path,
            "analysis_path": Path(folder) / f"{uid_of(i)}.npz", "annotation": annotation, **extra}


def beat_annotation(duration: float = 20.0, bpm: float = 120.0, **extra) -> Annotation:
    beats = steady_beats(duration, bpm)
    return Annotation(beats=beats, sample_rate=SR, duration_sec=duration,
                      tempo_segments=tempo_rows(beats, duration, bpm), **extra)


def use_tracks(tracks) -> None:
    library_tracks.use(tracks)


def thread_pool(workers=1, *args, **kwargs):
    """In place of analyzer.runtime.spawn_pool: the jobs run in this process."""
    return ThreadPoolExecutor(max_workers=1)


def no_pool(*args, **kwargs):
    raise AssertionError("no process pool was expected (every cache was current)")


def plain_words(text) -> list[str]:
    """The parts of a progress text that are plain strings, not marked for the app's catalogs (analyzer.training.text):
    those reach the user untranslated. A string of digits and punctuation only, as "(3/10)", needs no translation."""
    from analyzer.training.text import Concat, Text
    if isinstance(text, Text):
        return []
    if isinstance(text, Concat):
        return [w for part in text.parts for w in plain_words(part)]
    if isinstance(text, tuple):
        return [w for part in text for w in plain_words(part)]
    s = str(text)
    return [s] if any(c.isalpha() for c in s) and not s.startswith("UUID: ") else []


def numeric_grad(f, P: dict, key: str, picks, eps: float = 1e-5) -> np.ndarray:
    """Central differences of the scalar ``f(P)`` at the flat indices ``picks`` of ``P[key]``."""
    out = []
    flat = P[key].reshape(-1)
    for i in picks:
        old = flat[i]
        flat[i] = old + eps
        hi = f(P)
        flat[i] = old - eps
        lo = f(P)
        flat[i] = old
        out.append((hi - lo) / (2 * eps))
    return np.asarray(out)
