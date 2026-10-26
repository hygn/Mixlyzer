"""Frame features of the library's tracks on the beat tracker's grid (22050 Hz, HPSS, hop 256), cached for
the beat tracker's optimizers: every FrameFeatures field, the features stored as float16."""

from __future__ import annotations

from concurrent.futures import as_completed
import dataclasses
from pathlib import Path
from typing import Callable

import numpy as np

from analyzer.beat.frame_features import FrameFeatures
from analyzer.runtime import spawn_pool
from analyzer.training import SkippedOptimizationTrack, optimizer_track_name, skipped_track
from analyzer.training.onset_training import _auto_cache_workers, _list_beat_tracks
from analyzer.files import atomic_output_path
from analyzer.training.text import msg

ProgressCallback = Callable[[int, str], None]
BEAT_FRAME_CACHE_FORMAT = "mixlyzer_beat_frame_cache_v1"
_SCALARS = ("sample_rate", "hop_length", "duration_sec")
_EXACT = ("frame_times", "melody_contour")       # kept at full precision


def list_grid_tracks(library_dir: Path, ignored: list[SkippedOptimizationTrack]) -> list[dict[str, object]]:
    """The library's tracks with their beat grid: beats and tempo rows (the user's, as stored)."""
    tracks = []
    for track in _list_beat_tracks(library_dir, ignored):
        try:
            rows = np.asarray(track["annotation"].tempo_segments, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            ignored.append(skipped_track(track, msg("invalid tempo rows: {error}", error=exc)))
            continue
        if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 4:
            ignored.append(skipped_track(track, msg("no tempo rows")))
            continue
        track["segments"] = rows
        tracks.append(track)
    return tracks


def cache_path(cache_dir: Path, uid: str) -> Path:
    return Path(cache_dir) / f"{uid}.npz"


def read_frames(path: Path, track: dict[str, object]) -> FrameFeatures | None:
    """The cached frames when they are current (same format, same audio file), else None."""
    if not Path(path).is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(np.asarray(z["cache_format"]).item()) != BEAT_FRAME_CACHE_FORMAT:
                return None
            if int(np.asarray(z["audio_mtime_ns"]).item()) != Path(track["audio_path"]).stat().st_mtime_ns:
                return None
            values = {}
            for field in dataclasses.fields(FrameFeatures):
                v = np.asarray(z[field.name])
                if field.name in _SCALARS:
                    values[field.name] = v.item()
                elif field.name in _EXACT:
                    values[field.name] = v
                else:
                    values[field.name] = v.astype(np.float32)
            return FrameFeatures(**values)
    except Exception:
        # a partial or corrupt cache is a miss and is rebuilt
        return None


def build_frames(track: dict[str, object], cache_dir_text: str) -> str:
    """Process-worker entry point: the frames the full analysis gives the beat tracker."""
    from threadpoolctl import threadpool_limits
    from analyzer.beat.frame_features import extract_frame_features
    from analyzer.beat.general import FRAME_HOP_LENGTH, SAMPLE_RATE
    from analyzer.audio.load import fast_load
    from analyzer.audio.hpss import hpss_audio_with_spectra

    path = cache_path(Path(cache_dir_text), str(track["uid"]))
    with threadpool_limits(limits=1):
        y = fast_load(str(track["audio_path"]), SAMPLE_RATE)
        harmonic, percussive, _ = hpss_audio_with_spectra(y, SAMPLE_RATE)
        frames = extract_frame_features(y, SAMPLE_RATE, percussive, harmonic, hop_length=FRAME_HOP_LENGTH)
    values = {}
    for field in dataclasses.fields(FrameFeatures):
        v = getattr(frames, field.name)
        if field.name in _SCALARS or field.name in _EXACT:
            values[field.name] = np.asarray(v)
        else:
            values[field.name] = np.asarray(v, dtype=np.float16)
    with atomic_output_path(path) as temporary:
        np.savez_compressed(
            temporary,
            cache_format=np.asarray(BEAT_FRAME_CACHE_FORMAT),
            audio_mtime_ns=np.int64(Path(track["audio_path"]).stat().st_mtime_ns),
            **values,
        )
    return str(track["uid"])


def ensure_frames(
    tracks: list[dict[str, object]],
    cache_dir: Path,
    ignored: list[SkippedOptimizationTrack],
    *,
    rebuild: bool = False,
    max_workers: int | None = None,
    progress: ProgressCallback | None = None,
    progress_span: tuple[int, int] = (0, 100),
) -> list[dict[str, object]]:
    """Build the missing or stale caches; returns the tracks whose cache is current."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [t for t in tracks if rebuild or read_frames(cache_path(cache_dir, str(t["uid"])), t) is None]
    failed: set[str] = set()
    if jobs:
        lo, hi = progress_span
        with spawn_pool(_auto_cache_workers(max_workers, len(jobs))) as executor:
            futures = {executor.submit(build_frames, t, str(cache_dir)): t for t in jobs}
            for done, future in enumerate(as_completed(futures), start=1):
                track = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed.add(str(track["uid"]))
                    ignored.append(skipped_track(track, msg("frame feature cache build failed: {error}", error=exc)))
                if progress:
                    progress(lo + int(round((hi - lo) * done / len(jobs))),
                             (msg("Frame features {done}/{total}", done=done, total=len(jobs)), optimizer_track_name(track)))
    return [t for t in tracks if str(t["uid"]) not in failed]
