"""Beat/bar grid reconstruction and current phrase acoustic feature extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
from scipy.special import expit


EPS = 1e-10


@dataclass(frozen=True)
class MeterSegment:
    """A beat-analysis interval used only to define downbeat-aligned bars."""

    start_sec: float
    end_sec: float
    inizio_sec: float
    meter: int


@dataclass(frozen=True)
class PredictorGrid:
    """Permitted structural input reconstructed from the analysis NPZ."""

    beat_times_sec: np.ndarray
    beat_edges_sec: np.ndarray
    meter_segments: tuple[MeterSegment, ...]
    downbeat_mask: np.ndarray
    beat_in_bar: np.ndarray
    bar_index_of_beat: np.ndarray
    bar_starts_beat: np.ndarray
    bar_ends_beat: np.ndarray
    bar_starts_sec: np.ndarray
    bar_ends_sec: np.ndarray
    bar_meters: np.ndarray
    bar_is_partial: np.ndarray

    @property
    def n_beats(self) -> int:
        return int(self.beat_times_sec.size)

    @property
    def n_bars(self) -> int:
        return int(self.bar_starts_beat.size)

    @property
    def bar_boundary_times_sec(self) -> np.ndarray:
        if self.n_bars == 0:
            return np.asarray([], dtype=np.float64)
        return np.concatenate(
            [self.bar_starts_sec, self.bar_ends_sec[-1:]], axis=0
        ).astype(np.float64)


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 22050
    hop_length: int = 512
    n_fft: int = 2048
    n_mels: int = 48
    n_mfcc: int = 20
    beat_subdivisions: int = 8
    fmin: float = 30.0
    fmax: float | None = 11025.0
    res_type: str = "soxr_hq"



@dataclass
class AcousticFeatures:
    """Feature matrices have shape ``(channels, beats_or_bars)``."""

    beat: dict[str, np.ndarray]
    bar: dict[str, np.ndarray]
    family_beat: dict[str, np.ndarray]
    family_bar: dict[str, np.ndarray]
    audio_duration_sec: float



def _validate_beat_grid(beat_times: np.ndarray) -> np.ndarray:
    beats = np.asarray(beat_times, dtype=np.float64)
    if beats.ndim != 1 or beats.size < 8:
        raise ValueError("beats_time_sec must be a 1-D array with at least 8 beats.")
    if not np.all(np.isfinite(beats)):
        raise ValueError("beats_time_sec contains NaN or infinity.")
    if np.any(np.diff(beats) <= 0.0):
        raise ValueError("beats_time_sec must be strictly increasing.")
    if beats[0] < 0.0:
        raise ValueError("Beat times must be non-negative.")
    return beats


def _parse_meter_segments(tempo_segments: np.ndarray) -> tuple[MeterSegment, ...]:
    rows = np.asarray(tempo_segments, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 5:
        raise ValueError(
            "tempo_segments must have at least five columns: "
            "[start, end, bpm, inizio, meter]."
        )

    parsed: list[MeterSegment] = []
    for row in rows:
        start_sec = float(row[0])
        end_sec = float(row[1])
        inizio_sec = float(row[3])
        meter = int(round(float(row[4])))
        if not all(np.isfinite((start_sec, end_sec, inizio_sec))):
            continue
        if end_sec <= start_sec or meter < 1:
            continue
        parsed.append(
            MeterSegment(
                start_sec=start_sec,
                end_sec=end_sec,
                inizio_sec=inizio_sec,
                meter=meter,
            )
        )

    if not parsed:
        raise ValueError("No valid [start, end, inizio, meter] segment was found.")
    parsed.sort(key=lambda segment: (segment.start_sec, segment.end_sec))
    return tuple(parsed)


def _estimate_final_beat_edge(beats: np.ndarray, segment_end: float) -> float:
    median_ibi = float(np.median(np.diff(beats)))
    estimated = float(beats[-1] + median_ibi)
    if np.isfinite(segment_end) and segment_end > beats[-1]:
        estimated = min(estimated, float(segment_end))
    if estimated <= beats[-1] + 1e-5:
        estimated = float(beats[-1] + median_ibi)
    return estimated


def _reconstruct_bars(
    beats: np.ndarray,
    segments: tuple[MeterSegment, ...],
) -> PredictorGrid:
    n_beats = beats.size
    downbeat_mask = np.zeros(n_beats, dtype=bool)
    beat_meter = np.zeros(n_beats, dtype=np.int16)
    beat_anchor = np.full(n_beats, -1, dtype=np.int32)

    median_ibi = float(np.median(np.diff(beats)))
    tolerance = max(0.08, 0.48 * median_ibi)

    for seg_idx, segment in enumerate(segments):
        anchor = int(np.argmin(np.abs(beats - segment.inizio_sec)))
        anchor_error = abs(float(beats[anchor] - segment.inizio_sec))
        if anchor_error > tolerance:
            raise ValueError(
                f"inizio={segment.inizio_sec:.6f}s is not close to a beat "
                f"(nearest error {anchor_error:.3f}s)."
            )

        indices = np.flatnonzero(
            (beats >= segment.start_sec - tolerance)
            & (beats < segment.end_sec + tolerance)
        )
        if indices.size == 0:
            continue
        beat_meter[indices] = segment.meter
        beat_anchor[indices] = anchor
        phases = np.mod(indices - anchor, segment.meter)
        downbeat_mask[indices[phases == 0]] = True

    # Fill uncovered beats with the nearest defined segment.  Only meter and
    # inizio-derived phase are propagated; BPM and annotations remain unused.
    covered = beat_meter > 0
    if not np.all(covered):
        covered_indices = np.flatnonzero(covered)
        if covered_indices.size == 0:
            raise ValueError("Meter segments do not overlap the beat grid.")
        for beat_idx in np.flatnonzero(~covered):
            nearest = int(covered_indices[np.argmin(np.abs(covered_indices - beat_idx))])
            beat_meter[beat_idx] = beat_meter[nearest]
            beat_anchor[beat_idx] = beat_anchor[nearest]
            if (beat_idx - beat_anchor[beat_idx]) % int(beat_meter[beat_idx]) == 0:
                downbeat_mask[beat_idx] = True

    actual_downbeats = np.flatnonzero(downbeat_mask)
    boundary_starts = np.unique(
        np.concatenate(
            [np.asarray([0], dtype=np.int32), actual_downbeats.astype(np.int32)]
        )
    )
    boundary_starts = boundary_starts[boundary_starts < n_beats]
    boundary_ends = np.concatenate(
        [boundary_starts[1:], np.asarray([n_beats], dtype=np.int32)]
    )

    valid = boundary_ends > boundary_starts
    boundary_starts = boundary_starts[valid]
    boundary_ends = boundary_ends[valid]

    beat_in_bar = np.zeros(n_beats, dtype=np.int16)
    bar_index_of_beat = np.full(n_beats, -1, dtype=np.int32)
    bar_meters = np.zeros(boundary_starts.size, dtype=np.int16)
    bar_is_partial = np.zeros(boundary_starts.size, dtype=bool)

    for bar_idx, (start, end) in enumerate(
        zip(boundary_starts, boundary_ends, strict=True)
    ):
        bar_index_of_beat[start:end] = bar_idx
        beat_in_bar[start:end] = np.arange(end - start, dtype=np.int16)
        meter = int(np.median(beat_meter[start:end]))
        bar_meters[bar_idx] = meter
        bar_is_partial[bar_idx] = (
            not bool(downbeat_mask[start]) or (end - start) != meter
        )

    final_edge = _estimate_final_beat_edge(beats, max(s.end_sec for s in segments))
    beat_edges = np.concatenate([beats, np.asarray([final_edge], dtype=np.float64)])
    bar_starts_sec = beats[boundary_starts]
    bar_ends_sec = beat_edges[boundary_ends]

    return PredictorGrid(
        beat_times_sec=beats,
        beat_edges_sec=beat_edges,
        meter_segments=segments,
        downbeat_mask=downbeat_mask,
        beat_in_bar=beat_in_bar,
        bar_index_of_beat=bar_index_of_beat,
        bar_starts_beat=boundary_starts,
        bar_ends_beat=boundary_ends,
        bar_starts_sec=bar_starts_sec.astype(np.float64),
        bar_ends_sec=bar_ends_sec.astype(np.float64),
        bar_meters=bar_meters,
        bar_is_partial=bar_is_partial,
    )


def build_predictor_grid(
    beat_times_sec: np.ndarray, tempo_segments: np.ndarray
) -> PredictorGrid:
    """Build the predictor grid from in-memory beats + tempo_segments arrays.

    Same as :func:`load_predictor_grid` but without reading an NPZ, for callers
    (e.g. the live analyzer) that already hold these arrays.
    """
    beats = _validate_beat_grid(np.asarray(beat_times_sec, dtype=np.float64))
    segments = _parse_meter_segments(np.asarray(tempo_segments, dtype=np.float64))
    return _reconstruct_bars(beats, segments)


def load_predictor_grid(npz_path: str | Path) -> PredictorGrid:
    """Read only the predictor-permitted fields from a Mixlyzer analysis NPZ.

    No key enumeration is performed.  In particular, this function never
    accesses ``phrase_segments_np.*``, precomputed chroma, key, cue, or image
    fields.
    """

    with np.load(Path(npz_path), allow_pickle=False) as archive:
        try:
            beats = _validate_beat_grid(archive["beats_time_sec"])
        except KeyError as exc:
            raise KeyError("NPZ is missing required field 'beats_time_sec'.") from exc
        try:
            segments = _parse_meter_segments(archive["tempo_segments"])
        except KeyError as exc:
            raise KeyError("NPZ is missing required field 'tempo_segments'.") from exc

    return _reconstruct_bars(beats, segments)


# ---------------------------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------------------------


def _load_audio_stereo(
    audio_path: str | Path,
    config: FeatureConfig,
    *,
    audio_array: np.ndarray | None = None,
    audio_sr: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if audio_array is not None:
        # Use already-decoded audio (no second decode). Accept (N,), (N,2), (2,N).
        arr = np.asarray(audio_array, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] in (1, 2) and arr.shape[0] < arr.shape[1]:
            waveform = arr            # (ch, N)
        elif arr.ndim == 2:
            waveform = arr.T          # (N, ch) -> (ch, N)
        else:
            waveform = arr            # (N,)
        sr = int(audio_sr or config.sample_rate)
        if sr != config.sample_rate:
            waveform = librosa.resample(
                np.ascontiguousarray(waveform), orig_sr=sr,
                target_sr=config.sample_rate, res_type=config.res_type,
            )
            sr = config.sample_rate
    else:
        waveform, sr = librosa.load(
            path=str(audio_path),
            sr=config.sample_rate,
            mono=False,
            dtype=np.float32,
            res_type=config.res_type,
        )
    if waveform.ndim == 1:
        left = waveform
        right = waveform
    elif waveform.ndim == 2:
        left = waveform[0]
        right = waveform[1] if waveform.shape[0] > 1 else waveform[0]
    else:
        raise ValueError(f"Unexpected decoded audio shape: {waveform.shape}")

    length = min(left.size, right.size)
    if length == 0:
        raise ValueError("Decoded audio is empty.")
    left = np.asarray(left[:length], dtype=np.float32)
    right = np.asarray(right[:length], dtype=np.float32)
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    return mid.astype(np.float32), side.astype(np.float32), np.stack([left, right]), int(sr)


def _sync_frames_to_intervals(
    values: np.ndarray,
    frame_times: np.ndarray,
    edges: np.ndarray,
    reducer: str = "mean",
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if x.ndim != 2 or x.shape[1] != frame_times.size:
        raise ValueError(
            f"Frame feature shape {x.shape} does not match {frame_times.size} times."
        )

    result = np.zeros((x.shape[0], edges.size - 1), dtype=np.float64)
    left = np.searchsorted(frame_times, edges[:-1], side="left")
    right = np.searchsorted(frame_times, edges[1:], side="left")
    for index, (lo, hi) in enumerate(zip(left, right, strict=True)):
        if hi <= lo:
            nearest = int(
                np.clip(
                    np.searchsorted(frame_times, 0.5 * (edges[index] + edges[index + 1])),
                    0,
                    frame_times.size - 1,
                )
            )
            block = x[:, nearest : nearest + 1]
        else:
            block = x[:, lo:hi]
        if reducer == "mean":
            result[:, index] = np.mean(block, axis=1)
        elif reducer == "median":
            result[:, index] = np.median(block, axis=1)
        elif reducer == "sum":
            result[:, index] = np.sum(block, axis=1)
        elif reducer == "max":
            result[:, index] = np.max(block, axis=1)
        else:
            raise ValueError(f"Unsupported reducer: {reducer}")
    return result.astype(np.float32)


def _subdivision_profiles(
    onset_envelopes: Sequence[np.ndarray],
    frame_times: np.ndarray,
    beat_edges: np.ndarray,
    subdivisions: int,
) -> np.ndarray:
    n_beats = beat_edges.size - 1
    profiles = np.zeros(
        (len(onset_envelopes), subdivisions, n_beats), dtype=np.float64
    )
    for beat_idx in range(n_beats):
        sub_edges = np.linspace(
            beat_edges[beat_idx], beat_edges[beat_idx + 1], subdivisions + 1
        )
        for channel, envelope in enumerate(onset_envelopes):
            for sub_idx in range(subdivisions):
                lo = int(np.searchsorted(frame_times, sub_edges[sub_idx], side="left"))
                hi = int(
                    np.searchsorted(frame_times, sub_edges[sub_idx + 1], side="left")
                )
                if hi <= lo:
                    nearest = int(
                        np.clip(
                            np.searchsorted(
                                frame_times,
                                0.5 * (sub_edges[sub_idx] + sub_edges[sub_idx + 1]),
                            ),
                            0,
                            frame_times.size - 1,
                        )
                    )
                    profiles[channel, sub_idx, beat_idx] = envelope[nearest]
                else:
                    profiles[channel, sub_idx, beat_idx] = np.sum(envelope[lo:hi])

    normalizer = np.sum(profiles, axis=1, keepdims=True) + EPS
    profiles /= normalizer
    return profiles.reshape(len(onset_envelopes) * subdivisions, n_beats).astype(
        np.float32
    )


def _aggregate_beats_to_bars(
    values: np.ndarray,
    grid: PredictorGrid,
    reducer: str = "mean",
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if x.shape[1] != grid.n_beats:
        raise ValueError("Beat feature length does not match PredictorGrid.")
    result = np.zeros((x.shape[0], grid.n_bars), dtype=np.float64)
    for bar_idx, (start, end) in enumerate(
        zip(grid.bar_starts_beat, grid.bar_ends_beat, strict=True)
    ):
        block = x[:, start:end]
        if reducer == "mean":
            result[:, bar_idx] = np.mean(block, axis=1)
        elif reducer == "median":
            result[:, bar_idx] = np.median(block, axis=1)
        elif reducer == "sum":
            result[:, bar_idx] = np.sum(block, axis=1)
        elif reducer == "max":
            result[:, bar_idx] = np.max(block, axis=1)
        else:
            raise ValueError(f"Unsupported reducer: {reducer}")
    return result.astype(np.float32)


def _safe_log_power(power: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(power, EPS)).astype(np.float32)


def _band_energy(
    power: np.ndarray,
    frequencies: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    if not np.any(mask):
        return np.zeros(power.shape[1], dtype=np.float32)
    return np.mean(power[mask], axis=0).astype(np.float32)


def _pitch_salience(harmonic_magnitude: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    mask = (frequencies >= 90.0) & (frequencies <= 1400.0)
    if not np.any(mask):
        return np.zeros(harmonic_magnitude.shape[1], dtype=np.float32)
    selected = harmonic_magnitude[mask]
    peak = np.max(selected, axis=0)
    mean = np.mean(selected, axis=0)
    salience = np.log1p(peak / np.maximum(mean, EPS))
    return salience.astype(np.float32)


def extract_song_features(
    audio_path: str | Path,
    grid: PredictorGrid,
    config: FeatureConfig = FeatureConfig(),
    *,
    audio_array: np.ndarray | None = None,
    audio_sr: int | None = None,
) -> AcousticFeatures:
    """Extract deterministic beat- and bar-synchronous music features.

    Pass ``audio_array`` (mono ``(N,)`` or stereo ``(N, 2)`` / ``(2, N)``) with
    ``audio_sr`` to feed already-loaded audio instead of reading ``audio_path``.
    """

    mid, side, stereo, sr = _load_audio_stereo(
        audio_path, config, audio_array=audio_array, audio_sr=audio_sr
    )
    duration_sec = mid.size / float(sr)
    tolerance = max(1.0, 2.0 * float(np.median(np.diff(grid.beat_times_sec))))
    if grid.beat_times_sec[-1] > duration_sec + tolerance:
        raise ValueError(
            f"Beat grid ends at {grid.beat_times_sec[-1]:.3f}s but audio ends "
            f"at {duration_sec:.3f}s.  Check that both files belong to one song."
        )

    stft_mid = librosa.stft(
        mid,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        window="hann",
        center=True,
    )
    magnitude = np.abs(stft_mid).astype(np.float32)
    power = np.square(magnitude, dtype=np.float32)
    harmonic_mag, percussive_mag = librosa.decompose.hpss(magnitude)
    harmonic_power = np.square(harmonic_mag, dtype=np.float32)
    percussive_power = np.square(percussive_mag, dtype=np.float32)

    frame_times = librosa.frames_to_time(
        np.arange(magnitude.shape[1]), sr=sr, hop_length=config.hop_length
    )
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=config.n_fft)

    mel_power = librosa.feature.melspectrogram(
        S=power,
        sr=sr,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    mel_db = librosa.power_to_db(mel_power, ref=np.max)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=config.n_mfcc)
    chroma = librosa.feature.chroma_stft(
        S=harmonic_power,
        sr=sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        norm=2,
    )
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    contrast_bands = max(1, min(6, int(math.floor(math.log2((0.5 * sr) / 200.0))) - 1))
    contrast = librosa.feature.spectral_contrast(
        S=magnitude,
        sr=sr,
        n_fft=config.n_fft,
        n_bands=contrast_bands,
    )
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=power)[0]
    rolloff = librosa.feature.spectral_rolloff(
        S=magnitude, sr=sr, roll_percent=0.85
    )[0]

    rms = librosa.feature.rms(S=magnitude, frame_length=config.n_fft)[0]
    harmonic_rms = librosa.feature.rms(
        S=harmonic_mag, frame_length=config.n_fft
    )[0]
    percussive_rms = librosa.feature.rms(
        S=percussive_mag, frame_length=config.n_fft
    )[0]
    side_rms = librosa.feature.rms(
        y=side,
        frame_length=config.n_fft,
        hop_length=config.hop_length,
        center=True,
    )[0]
    mid_rms = librosa.feature.rms(
        y=mid,
        frame_length=config.n_fft,
        hop_length=config.hop_length,
        center=True,
    )[0]

    low = _band_energy(power, frequencies, 30.0, 180.0)
    low_mid = _band_energy(power, frequencies, 180.0, 800.0)
    mid_band = _band_energy(power, frequencies, 800.0, 4000.0)
    high = _band_energy(power, frequencies, 4000.0, 11000.0)
    total_energy = np.mean(power, axis=0)
    pitch_salience = _pitch_salience(harmonic_mag, frequencies)

    onset_full = librosa.onset.onset_strength(
        S=mel_db, sr=sr, hop_length=config.hop_length
    )
    percussive_mel = librosa.feature.melspectrogram(
        S=percussive_power,
        sr=sr,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    onset_percussive = librosa.onset.onset_strength(
        S=librosa.power_to_db(percussive_mel, ref=np.max),
        sr=sr,
        hop_length=config.hop_length,
    )
    high_log = np.log1p(
        np.mean(power[frequencies >= min(5000.0, 0.45 * sr)], axis=0)
    )
    onset_high = np.maximum(
        np.diff(high_log, prepend=high_log[:1]), 0.0
    ).astype(np.float32)

    beat_edges = grid.beat_edges_sec.copy()
    beat_edges[-1] = min(max(beat_edges[-1], grid.beat_times_sec[-1]), duration_sec)
    if beat_edges[-1] <= beat_edges[-2]:
        beat_edges[-1] = beat_edges[-2] + float(np.median(np.diff(grid.beat_times_sec)))

    beat_mel = _sync_frames_to_intervals(mel_db, frame_times, beat_edges, "median")
    beat_mfcc = _sync_frames_to_intervals(mfcc, frame_times, beat_edges, "mean")
    beat_chroma = _sync_frames_to_intervals(chroma, frame_times, beat_edges, "mean")
    beat_tonnetz = _sync_frames_to_intervals(tonnetz, frame_times, beat_edges, "mean")
    beat_contrast = _sync_frames_to_intervals(contrast, frame_times, beat_edges, "mean")
    beat_rhythm = _subdivision_profiles(
        (onset_full, onset_percussive, onset_high),
        frame_times,
        beat_edges,
        config.beat_subdivisions,
    )

    def sync_scalar(value: np.ndarray, reducer: str = "mean") -> np.ndarray:
        return _sync_frames_to_intervals(value, frame_times, beat_edges, reducer)[0]

    log_rms = np.log(np.maximum(sync_scalar(rms), EPS))
    harmonic_log_rms = np.log(np.maximum(sync_scalar(harmonic_rms), EPS))
    percussive_log_rms = np.log(np.maximum(sync_scalar(percussive_rms), EPS))
    low_log_energy = np.log(np.maximum(sync_scalar(low), EPS))
    low_mid_log_energy = np.log(np.maximum(sync_scalar(low_mid), EPS))
    mid_log_energy = np.log(np.maximum(sync_scalar(mid_band), EPS))
    high_log_energy = np.log(np.maximum(sync_scalar(high), EPS))
    total_log_energy = np.log(np.maximum(sync_scalar(total_energy), EPS))
    harmonic_percussive_ratio = harmonic_log_rms - percussive_log_rms
    stereo_width = np.log(
        np.maximum(sync_scalar(side_rms), EPS)
        / np.maximum(sync_scalar(mid_rms), EPS)
    )
    centroid_norm = sync_scalar(centroid) / (0.5 * sr)
    bandwidth_norm = sync_scalar(bandwidth) / (0.5 * sr)
    rolloff_norm = sync_scalar(rolloff) / (0.5 * sr)
    flatness_sync = sync_scalar(flatness)
    pitch_salience_sync = sync_scalar(pitch_salience)
    onset_full_sync = sync_scalar(onset_full, "sum")
    onset_percussive_sync = sync_scalar(onset_percussive, "sum")
    onset_high_sync = sync_scalar(onset_high, "sum")

    mid_ratio = np.exp(mid_log_energy - total_log_energy)
    vocal_proxy = expit(
        0.85 * _robust_z_1d(pitch_salience_sync)
        + 0.70 * _robust_z_1d(mid_ratio)
        + 0.45 * _robust_z_1d(harmonic_percussive_ratio)
        - 0.35 * _robust_z_1d(flatness_sync)
        - 0.20 * _robust_z_1d(onset_percussive_sync)
    ).astype(np.float32)

    beat: dict[str, np.ndarray] = {
        "mel": beat_mel,
        "mfcc": beat_mfcc,
        "chroma": beat_chroma,
        "tonnetz": beat_tonnetz,
        "spectral_contrast": beat_contrast,
        "rhythm_profile": beat_rhythm,
        "log_rms": log_rms[np.newaxis, :].astype(np.float32),
        "harmonic_log_rms": harmonic_log_rms[np.newaxis, :].astype(np.float32),
        "percussive_log_rms": percussive_log_rms[np.newaxis, :].astype(np.float32),
        "low_log_energy": low_log_energy[np.newaxis, :].astype(np.float32),
        "low_mid_log_energy": low_mid_log_energy[np.newaxis, :].astype(np.float32),
        "mid_log_energy": mid_log_energy[np.newaxis, :].astype(np.float32),
        "high_log_energy": high_log_energy[np.newaxis, :].astype(np.float32),
        "harmonic_percussive_ratio": harmonic_percussive_ratio[np.newaxis, :].astype(np.float32),
        "centroid": centroid_norm[np.newaxis, :].astype(np.float32),
        "bandwidth": bandwidth_norm[np.newaxis, :].astype(np.float32),
        "rolloff": rolloff_norm[np.newaxis, :].astype(np.float32),
        "flatness": flatness_sync[np.newaxis, :].astype(np.float32),
        "stereo_width": stereo_width[np.newaxis, :].astype(np.float32),
        "pitch_salience": pitch_salience_sync[np.newaxis, :].astype(np.float32),
        "vocal_proxy": vocal_proxy[np.newaxis, :].astype(np.float32),
        "onset_full": onset_full_sync[np.newaxis, :].astype(np.float32),
        "onset_percussive": onset_percussive_sync[np.newaxis, :].astype(np.float32),
        "onset_high": onset_high_sync[np.newaxis, :].astype(np.float32),
    }

    bar: dict[str, np.ndarray] = {}
    sum_features = {"onset_full", "onset_percussive", "onset_high"}
    median_features = {"mel"}
    for name, values in beat.items():
        reducer = "sum" if name in sum_features else "median" if name in median_features else "mean"
        bar[name] = _aggregate_beats_to_bars(values, grid, reducer)

    family_beat = {
        "timbre": np.vstack(
            [beat["mel"], beat["mfcc"], beat["spectral_contrast"]]
        ),
        "harmony": np.vstack([beat["chroma"], beat["tonnetz"]]),
        "rhythm": np.vstack(
            [
                beat["rhythm_profile"],
                beat["onset_full"],
                beat["onset_percussive"],
                beat["onset_high"],
            ]
        ),
        "texture": np.vstack(
            [
                beat["log_rms"],
                beat["harmonic_log_rms"],
                beat["percussive_log_rms"],
                beat["low_log_energy"],
                beat["low_mid_log_energy"],
                beat["mid_log_energy"],
                beat["high_log_energy"],
                beat["harmonic_percussive_ratio"],
                beat["centroid"],
                beat["bandwidth"],
                beat["rolloff"],
                beat["flatness"],
                beat["stereo_width"],
                beat["pitch_salience"],
                beat["vocal_proxy"],
            ]
        ),
    }
    family_bar = {
        "timbre": np.vstack(
            [bar["mel"], bar["mfcc"], bar["spectral_contrast"]]
        ),
        "harmony": np.vstack([bar["chroma"], bar["tonnetz"]]),
        "rhythm": np.vstack(
            [
                bar["rhythm_profile"],
                bar["onset_full"],
                bar["onset_percussive"],
                bar["onset_high"],
            ]
        ),
        "texture": np.vstack(
            [
                bar["log_rms"],
                bar["harmonic_log_rms"],
                bar["percussive_log_rms"],
                bar["low_log_energy"],
                bar["low_mid_log_energy"],
                bar["mid_log_energy"],
                bar["high_log_energy"],
                bar["harmonic_percussive_ratio"],
                bar["centroid"],
                bar["bandwidth"],
                bar["rolloff"],
                bar["flatness"],
                bar["stereo_width"],
                bar["pitch_salience"],
                bar["vocal_proxy"],
            ]
        ),
    }

    return AcousticFeatures(
        beat=beat,
        bar=bar,
        family_beat=family_beat,
        family_bar=family_bar,
        audio_duration_sec=float(duration_sec),
    )

def _robust_z_1d(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    scale = 1.4826 * mad
    if scale <= 1e-8:
        scale = float(np.std(x))
    if scale <= 1e-8:
        scale = 1.0
    return ((x - median) / scale).astype(np.float64)

