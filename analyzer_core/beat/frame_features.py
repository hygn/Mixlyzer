from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import librosa
import numpy as np

from analyzer_core.beat.melody_contour import extract_melody_contour


# Shared frame grid. Computed once per track before beat tracking: the learned
# onset (beat tracking) and the downbeat model (after pooling to beats) both
# read these frames, so the analysis is not repeated. The analyzer uses the
# configured BPM hop length; learned models that do not record the hop they
# were optimized at were optimized at this one.
FRAME_HOP_LENGTH = 256
FRAME_N_FFT = 2048
FRAME_N_MELS = 64
# Percussive flux bands: kick / snare-body / hats (2.5-5 kHz is left out).
PERCUSSIVE_BAND_EDGES_HZ = ((0.0, 150.0), (150.0, 2500.0), (5000.0, np.inf))
# Melody register of the harmonic part: C4 .. B6.
MELODY_FMIN_NOTE = "C4"
MELODY_OCTAVES = 3
# librosa.feature.chroma_cqt default resolution.
CHROMA_BINS_PER_OCTAVE = 36
BASS_CHROMA_FMIN_HZ = float(librosa.note_to_hz("C1"))


@dataclass(frozen=True)
class FrameFeatures:
    """Frame-level features on one grid; frame ``i`` is centred at ``frame_times[i]``."""

    sample_rate: int
    hop_length: int
    duration_sec: float
    frame_times: np.ndarray       # [T] seconds
    chroma: np.ndarray            # [12, T] CQT chroma
    bass_chroma: np.ndarray       # [12, T] low-register chroma
    mfcc: np.ndarray              # [20, T]
    tonnetz: np.ndarray           # [6, T]
    onset_bands: np.ndarray       # [3, T] positive log-mel flux of the mix (low / mid / high)
    rms_db: np.ndarray            # [T]
    band_ratios: np.ndarray       # [3, T] power share below 180 Hz / 180-2500 Hz / above
    flatness: np.ndarray          # [T] spectral flatness of the mix
    librosa_onset: np.ndarray     # [T] librosa onset strength of the percussive part
    harmonic_onset: np.ndarray    # [T] librosa onset strength of the harmonic part
    percussive_bands: np.ndarray  # [3, T] positive log-mel flux of the percussive part
    percussive_share: np.ndarray  # [T] percussive / (percussive + harmonic) power
    melody_chroma: np.ndarray     # [12, T] chroma of the harmonic part in the melody register
    melody_onset: np.ndarray      # [T] positive log-CQT flux of the melody register (note / syllable starts)
    melody_pitch: np.ndarray      # [T] predominant melody pitch (MIDI semitones), NaN when unvoiced
    melody_contour: np.ndarray    # [T] id of the melody contour owning the frame, -1 when unvoiced

    @property
    def n_frames(self) -> int:
        return int(self.frame_times.size)


def model_frame_hop_length(model: dict) -> int:
    """Frame hop a learned model was trained on."""
    return int(model.get("frame_hop_length", FRAME_HOP_LENGTH))


def robust_standardize_rows(features: np.ndarray) -> np.ndarray:
    median = np.median(features, axis=1, keepdims=True)
    mad = np.median(np.abs(features - median), axis=1, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1e-5)
    return ((features - median) / scale).astype(np.float32)


def _fit_length(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] >= length:
        return values[..., :length]
    pad = [(0, 0)] * (values.ndim - 1) + [(0, length - values.shape[-1])]
    return np.pad(values, pad, mode="edge")


def _positive_flux(log_spec: np.ndarray) -> np.ndarray:
    return np.maximum(np.diff(log_spec, axis=1, prepend=log_spec[:, :1]), 0.0)


def extract_frame_features(
    audio: np.ndarray,
    sample_rate: int,
    percussive: np.ndarray | None = None,
    harmonic: np.ndarray | None = None,
    *,
    hop_length: int = FRAME_HOP_LENGTH,
    n_fft: int = FRAME_N_FFT,
    n_mels: int = FRAME_N_MELS,
) -> FrameFeatures:
    """Analyze mono ``audio`` once on a shared frame grid.

    ``percussive`` / ``harmonic`` are the HPSS parts of ``audio``; without them
    the mix stands in for both (no HPSS).
    """
    audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("audio is empty")
    sample_rate = int(sample_rate)
    percussive = audio if percussive is None else np.asarray(percussive, dtype=np.float32).reshape(-1)
    harmonic = audio if harmonic is None else np.asarray(harmonic, dtype=np.float32).reshape(-1)

    def spectrum(signal: np.ndarray) -> np.ndarray:
        return np.abs(librosa.stft(
            y=signal, n_fft=n_fft, hop_length=hop_length,
            window="hann", center=True, pad_mode="reflect",
        )).astype(np.float32)

    magnitude = spectrum(audio)
    power = np.square(magnitude, dtype=np.float32)
    n_frames = magnitude.shape[1]
    frame_times = librosa.frames_to_time(
        np.arange(n_frames), sr=sample_rate, hop_length=hop_length,
    ).astype(np.float64)

    # The CQTs and the melody contour run on worker threads while the STFT
    # features below are computed (FFTs and scipy filters release the GIL).
    # The workers do no BLAS matrix products: concurrent BLAS calls would change
    # the float rounding from run to run, so the chroma filterbanks run on this
    # thread after the join.
    def mix_cqt() -> tuple[np.ndarray, np.ndarray]:
        # chroma_cqt estimates the tuning of ``audio`` on every call; estimate it once.
        tuning = librosa.estimate_tuning(y=audio, sr=sample_rate, bins_per_octave=CHROMA_BINS_PER_OCTAVE)
        # The CQTs librosa.feature.chroma_cqt computes: C1 + 7 octaves, and the bass register C1 + 3 octaves.
        full = np.abs(librosa.cqt(
            audio, sr=sample_rate, hop_length=hop_length, n_bins=7 * CHROMA_BINS_PER_OCTAVE,
            bins_per_octave=CHROMA_BINS_PER_OCTAVE, tuning=tuning,
        ))
        bass = np.abs(librosa.cqt(
            audio, sr=sample_rate, hop_length=hop_length, fmin=BASS_CHROMA_FMIN_HZ,
            n_bins=3 * CHROMA_BINS_PER_OCTAVE, bins_per_octave=CHROMA_BINS_PER_OCTAVE, tuning=tuning,
        ))
        return full, bass

    def melody() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        melody_cqt = np.abs(librosa.cqt(
            harmonic, sr=sample_rate, hop_length=hop_length,
            fmin=librosa.note_to_hz(MELODY_FMIN_NOTE), n_bins=12 * MELODY_OCTAVES, bins_per_octave=12,
        ))
        melody_onset = _positive_flux(
            librosa.amplitude_to_db(melody_cqt, ref=np.max, top_db=80.0)
        ).mean(axis=0)
        melody_pitch, melody_contour = extract_melody_contour(harmonic, sample_rate, hop_length, n_frames)
        return melody_cqt, melody_onset, melody_pitch, melody_contour

    pool = ThreadPoolExecutor(max_workers=2)
    mix_cqt_job = pool.submit(mix_cqt)
    melody_job = pool.submit(melody)

    mel_power = librosa.feature.melspectrogram(
        S=power, sr=sample_rate, n_mels=n_mels,
        fmin=30.0, fmax=sample_rate / 2.0, norm="slaney", power=2.0,
    )
    log_mel = librosa.power_to_db(mel_power, ref=np.max, top_db=100.0).astype(np.float32)

    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(
            librosa.feature.melspectrogram(S=power, sr=sample_rate, n_mels=n_mels),
            ref=np.max,
        ),
        n_mfcc=20,
    )

    mel_difference = _positive_flux(log_mel)
    band_edges = np.linspace(0, n_mels, 4, dtype=int)
    onset_bands = np.stack(
        [mel_difference[band_edges[i]:band_edges[i + 1]].mean(axis=0) for i in range(3)]
    )
    rms = librosa.feature.rms(S=magnitude, frame_length=n_fft, center=False)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max, top_db=100.0)[0]
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    total_power = np.maximum(power.sum(axis=0), 1e-10)
    band_ratios = np.stack(
        [
            power[frequencies < 180.0].sum(axis=0),
            power[(frequencies >= 180.0) & (frequencies < 2500.0)].sum(axis=0),
            power[frequencies >= 2500.0].sum(axis=0),
        ]
    ) / total_power
    flatness = librosa.feature.spectral_flatness(S=magnitude)[0]

    # HPSS parts: drum flux per band, harmonic (note / chord) onsets, drum share.
    percussive_power = np.square(spectrum(percussive), dtype=np.float32)
    harmonic_power = np.square(spectrum(harmonic), dtype=np.float32)
    percussive_mel = librosa.power_to_db(
        librosa.feature.melspectrogram(S=percussive_power, sr=sample_rate, n_mels=n_mels, fmin=30.0),
        ref=np.max, top_db=100.0,
    )
    percussive_flux = _positive_flux(percussive_mel)
    mel_hz = librosa.mel_frequencies(n_mels=n_mels, fmin=30.0, fmax=sample_rate / 2.0)
    percussive_bands = np.stack(
        [percussive_flux[(mel_hz >= lo) & (mel_hz < hi)].mean(axis=0) for lo, hi in PERCUSSIVE_BAND_EDGES_HZ]
    )
    percussive_total = percussive_power.sum(axis=0)
    percussive_share = percussive_total / np.maximum(percussive_total + harmonic_power.sum(axis=0), 1e-10)

    # librosa onset strength (128-band log-mel flux) from the spectra above.
    def onset_strength(signal_power: np.ndarray) -> np.ndarray:
        return librosa.onset.onset_strength(
            S=librosa.power_to_db(librosa.feature.melspectrogram(S=signal_power, sr=sample_rate)),
            sr=sample_rate, hop_length=hop_length, n_fft=n_fft, center=True,
        )

    librosa_onset = onset_strength(percussive_power)
    harmonic_onset = onset_strength(harmonic_power)

    with pool:
        full_cqt, bass_cqt = mix_cqt_job.result()
        melody_cqt, melody_onset, melody_pitch, melody_contour = melody_job.result()
    # Sharp CQT chroma (not CENS, which over-smooths chord-change timing).
    chroma = librosa.feature.chroma_cqt(C=full_cqt, n_chroma=12, bins_per_octave=CHROMA_BINS_PER_OCTAVE)
    bass_chroma = librosa.feature.chroma_cqt(
        C=bass_cqt, n_chroma=12, fmin=BASS_CHROMA_FMIN_HZ, bins_per_octave=CHROMA_BINS_PER_OCTAVE,
    )
    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sample_rate)
    melody_chroma = librosa.feature.chroma_cqt(C=melody_cqt, n_chroma=12)

    return FrameFeatures(
        sample_rate=sample_rate,
        hop_length=int(hop_length),
        duration_sec=audio.size / float(sample_rate),
        frame_times=frame_times,
        chroma=_fit_length(chroma, n_frames),
        bass_chroma=_fit_length(bass_chroma, n_frames),
        mfcc=_fit_length(mfcc, n_frames),
        tonnetz=_fit_length(tonnetz, n_frames),
        onset_bands=_fit_length(onset_bands, n_frames),
        rms_db=_fit_length(rms_db, n_frames),
        band_ratios=_fit_length(band_ratios, n_frames),
        flatness=_fit_length(flatness, n_frames),
        librosa_onset=_fit_length(librosa_onset, n_frames),
        harmonic_onset=_fit_length(harmonic_onset, n_frames),
        percussive_bands=_fit_length(percussive_bands, n_frames),
        percussive_share=_fit_length(percussive_share, n_frames),
        melody_chroma=_fit_length(melody_chroma, n_frames),
        melody_onset=_fit_length(melody_onset, n_frames),
        melody_pitch=melody_pitch,
        melody_contour=melody_contour,
    )
