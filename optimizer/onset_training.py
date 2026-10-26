"""Training data of the onset models: the library's songs with beat grids, a per-song cache of the onset
channels (``learned_onset.onset_channels``) and frame samples with beat targets. Used by the tempo-change
onset optimizer (``refine_onset_optimizer.py``) and the beat models' frame cache (``beat_frames.py``)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from analyzer_core.beat.frame_features import extract_frame_features
from analyzer_core.hpss import hpss_audio
from analyzer_core.beat.learned_onset import ONSET_CHANNEL_NAMES, onset_channels
from foundation.decoder import decode_to_memmap
from optimizer import SkippedOptimizationTrack, library_tracks, skipped_track
from utils.atomic_io import atomic_output_path
from foundation.i18n import tr


ONSET_FEATURE_CACHE_FORMAT = "mixlyzer_onset_feature_cache_v4"

# Frames near an annotated beat get a Gaussian target of this width (in frames).
TARGET_SIGMA_FRAMES = 0.75
TARGET_MIN = 0.01
NEGATIVE_STRIDE = 6
# The frames at these beat subdivisions are where half-beat, 16th and triplet
# grids collect energy; each is always used and weighs SUBDIVISION_WEIGHT frames.
SUBDIVISION_POSITIONS = (1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4)
SUBDIVISION_WEIGHT = 4.0
MIN_BEATS = 16


def _list_beat_tracks(
    library_dir: Path,
    ignored: list[SkippedOptimizationTrack] | None = None,
) -> list[dict[str, object]]:
    """The tracks with a usable beat grid, of the tracks the library's owner gave this run with their stored
    features (optimizer.library_tracks; ``library_dir`` is kept for the callers' signatures)."""
    tracks: list[dict[str, object]] = []
    for track in library_tracks.tracks():
        audio_path = Path(track["audio_path"])
        track.update(audio_path=audio_path)
        stored = track["features"]
        if stored is None:
            if ignored is not None:
                reason = track.get("error")
                ignored.append(skipped_track(track, tr("invalid beat analysis: {error}").format(error=reason)
                                             if reason else tr("analysis NPZ is missing")))
            continue
        if not audio_path.is_file():
            if ignored is not None:
                ignored.append(skipped_track(track, tr("audio file is missing")))
            continue
        try:
            beats = np.asarray(stored["beats_time_sec"], dtype=np.float64)
            sample_rate = int(np.asarray(stored["sr"]).item())
        except (KeyError, ValueError) as exc:
            if ignored is not None:
                ignored.append(skipped_track(track, tr("invalid beat analysis: {error}").format(error=exc)))
            continue
        if beats.ndim != 1 or beats.size < MIN_BEATS or not np.all(np.isfinite(beats)):
            if ignored is not None:
                ignored.append(skipped_track(track, tr("invalid or insufficient beat grid")))
            continue
        track.update(beats=np.sort(beats), sample_rate=sample_rate)
        tracks.append(track)
    return tracks


def _cache_path(cache_dir: Path, uid: str) -> Path:
    return cache_dir / f"{uid}.npz"


def _read_current_cache(
    cache_path: Path, track: dict[str, object], use_hpss: bool, hop_length: int
) -> tuple[np.ndarray, np.ndarray] | None:
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "cache_format",
                "frame_times",
                "channels",
                "channel_names",
                "use_hpss",
                "frame_hop_length",
                "sample_rate",
                "audio_mtime_ns",
            }:
                return None
            frame_times = np.asarray(archive["frame_times"], dtype=np.float64)
            channels = np.asarray(archive["channels"])
            current = bool(
                str(np.asarray(archive["cache_format"]).item()) == ONSET_FEATURE_CACHE_FORMAT
                and tuple(np.asarray(archive["channel_names"]).astype(str)) == ONSET_CHANNEL_NAMES
                and bool(np.asarray(archive["use_hpss"]).item()) == bool(use_hpss)
                and int(np.asarray(archive["frame_hop_length"]).item()) == int(hop_length)
                and int(np.asarray(archive["sample_rate"]).item()) == int(track["sample_rate"])
                and frame_times.ndim == 1
                and channels.shape == (frame_times.size, len(ONSET_CHANNEL_NAMES))
                and channels.dtype == np.dtype(np.float32)
                and np.all(np.isfinite(channels))
                and int(np.asarray(archive["audio_mtime_ns"]).item())
                == Path(track["audio_path"]).stat().st_mtime_ns
            )
            if current:
                return frame_times, np.asarray(channels, dtype=np.float32)
    except Exception:
        # A partial/corrupt cache is equivalent to a miss and is rebuilt.
        pass
    return None


def _build_track_cache(
    track: dict[str, object], cache_dir_text: str, use_hpss: bool, hop_length: int
) -> str:
    """Process-worker entry point for one onset feature cache."""

    cache_path = _cache_path(Path(cache_dir_text), str(track["uid"]))

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - installed with scikit-learn
        threadpool_limits = None

    def build() -> None:
        sample_rate = int(track["sample_rate"])
        # Same input as the analyzer: stereo decode averaged to mono and its
        # HPSS parts.
        stereo = decode_to_memmap(str(track["audio_path"]), sample_rate, 2).reshape(-1, 2)
        audio = np.ascontiguousarray(stereo.mean(axis=1), dtype=np.float32)
        harmonic, percussive = hpss_audio(audio) if use_hpss else (audio, audio)
        frames = extract_frame_features(
            audio, sample_rate, percussive, harmonic, hop_length=int(hop_length)
        )
        with atomic_output_path(cache_path) as temporary:
            np.savez_compressed(
                temporary,
                cache_format=np.asarray(ONSET_FEATURE_CACHE_FORMAT),
                frame_times=frames.frame_times.astype(np.float64),
                channels=onset_channels(frames),
                channel_names=np.asarray(ONSET_CHANNEL_NAMES, dtype="U64"),
                use_hpss=np.bool_(use_hpss),
                frame_hop_length=np.int64(hop_length),
                sample_rate=np.int64(sample_rate),
                audio_mtime_ns=np.int64(Path(track["audio_path"]).stat().st_mtime_ns),
            )

    if threadpool_limits is None:
        build()
    else:
        with threadpool_limits(limits=1):
            build()
    return str(track["uid"])


def beat_targets(frame_times: np.ndarray, beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frames inside the annotated beat range and their Gaussian beat targets."""
    beats = np.asarray(beats, dtype=np.float64)
    period = float(np.median(np.diff(beats)))
    inside = np.flatnonzero(
        (frame_times >= beats[0] - 0.5 * period) & (frame_times <= beats[-1] + 0.5 * period)
    )
    times = frame_times[inside]
    right = np.clip(np.searchsorted(beats, times), 1, beats.size - 1)
    distance = np.minimum(np.abs(times - beats[right - 1]), np.abs(beats[right] - times))
    hop_sec = float(np.median(np.diff(frame_times)))
    target = np.exp(-0.5 * np.square(distance / (TARGET_SIGMA_FRAMES * hop_sec)))
    return inside, target


def _subdivision_frames(frame_times: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Frame nearest to every SUBDIVISION_POSITIONS point between annotated beats."""
    units = (np.arange(beats.size - 1)[:, np.newaxis] + np.asarray(SUBDIVISION_POSITIONS)).ravel()
    times = np.interp(units, np.arange(beats.size, dtype=np.float64), beats)
    return np.unique(np.clip(np.rint(np.interp(times, frame_times, np.arange(frame_times.size))), 0, frame_times.size - 1).astype(np.int64))


def samples_from_inputs(
    inputs: np.ndarray, frame_times: np.ndarray, beats: np.ndarray
) -> dict[str, np.ndarray]:
    """Training samples from per-frame model inputs [T, D].

    kind 0: near a beat (Gaussian target), 1: beat subdivision, 2: other off-beat.
    """
    beats = np.asarray(beats, dtype=np.float64)
    inside, target = beat_targets(frame_times, beats)
    near = target >= TARGET_MIN
    subdivision = np.isin(inside, _subdivision_frames(frame_times, beats)) & ~near
    keep = near | subdivision | (inside % NEGATIVE_STRIDE == 0)
    kind = np.where(near, 0, np.where(subdivision, 1, 2))[keep].astype(np.int8)
    weight = np.select([kind == 0, kind == 1], [1.0, SUBDIVISION_WEIGHT], float(NEGATIVE_STRIDE))
    return {
        "x": inputs[inside[keep]],
        "y": np.where(near[keep], target[keep], 0.0).astype(np.float32),
        "w": weight.astype(np.float32),
        "kind": kind,
    }


def _auto_cache_workers(requested: int | None, jobs: int) -> int:
    if jobs <= 0:
        return 1
    if requested is not None:
        return max(1, min(int(requested), jobs))
    return max(1, min(4, jobs, (os.cpu_count() or 2) // 2))
