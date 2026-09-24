from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d


# Predominant melody without source separation, after MELODIA (Salamon & Gomez
# 2012), kept light: harmonic-summation pitch salience on a whitened CQT of the
# harmonic part, pitch contours linked by continuity, weak contours dropped
# (voicing), and at every frame the most salient contour is the melody.
# The beat-phase model reads note starts and held lengths from it (strong beats
# take long notes: Temperley & Sleator's length rule, Parncutt's durational accent).
CQT_BINS_PER_OCTAVE = 36
CQT_FMIN_NOTE = "C2"
CQT_OCTAVES = 7
F0_MIN_NOTE = "F3"
F0_MAX_NOTE = "C6"
N_HARMONICS = 10
HARMONIC_DECAY = 0.8
PEAKS_PER_FRAME = 5
PEAK_RELATIVE_FLOOR = 0.35        # peaks within 35 % of the frame's strongest salience
LINK_BINS = 2.4                   # 80 cents at 36 bins per octave
GAP_SEC = 0.05
MIN_CONTOUR_SEC = 0.1
VOICING_NU = 0.2                  # keep contours with mean salience >= mean - nu * std


def _salience(harmonic: np.ndarray, sample_rate: int, hop_length: int) -> tuple[np.ndarray, np.ndarray]:
    n_bins = CQT_BINS_PER_OCTAVE * CQT_OCTAVES
    fmin = librosa.note_to_hz(CQT_FMIN_NOTE)
    magnitude = np.abs(librosa.cqt(
        harmonic, sr=sample_rate, hop_length=hop_length, fmin=fmin,
        n_bins=n_bins, bins_per_octave=CQT_BINS_PER_OCTAVE,
    ))
    level = librosa.amplitude_to_db(magnitude, ref=np.max, top_db=80.0)
    envelope = uniform_filter1d(level, size=CQT_BINS_PER_OCTAVE, axis=0, mode="nearest")
    whitened = np.maximum(level - envelope, 0.0)
    frequencies = librosa.cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=CQT_BINS_PER_OCTAVE)
    candidates = np.flatnonzero(
        (frequencies >= librosa.note_to_hz(F0_MIN_NOTE)) & (frequencies <= librosa.note_to_hz(F0_MAX_NOTE))
    )
    salience = np.zeros((candidates.size, whitened.shape[1]))
    for harmonic_number in range(1, N_HARMONICS + 1):
        offset = int(round(CQT_BINS_PER_OCTAVE * np.log2(harmonic_number)))
        salience += HARMONIC_DECAY ** (harmonic_number - 1) * whitened[np.minimum(candidates + offset, n_bins - 1)]
    return salience, frequencies[candidates]


def _frame_peaks(salience: np.ndarray) -> list[np.ndarray]:
    is_peak = (salience >= np.roll(salience, 1, axis=0)) & (salience >= np.roll(salience, -1, axis=0)) & (salience > 0)
    floor = (1.0 - PEAK_RELATIVE_FLOOR) * salience.max(axis=0)
    peaks = []
    for t in range(salience.shape[1]):
        bins = np.flatnonzero(is_peak[:, t] & (salience[:, t] >= floor[t]))
        peaks.append(bins[np.argsort(salience[bins, t])[::-1][:PEAKS_PER_FRAME]])
    return peaks


def _contours(salience: np.ndarray, frame_sec: float) -> list[tuple[list[int], list[float], int]]:
    """Link salience peaks across frames: (bins, saliences, start frame) per contour."""
    gap = int(round(GAP_SEC / frame_sec))
    active: list[list] = []                         # [bins, saliences, start, last_frame]
    finished: list[list] = []
    for t, peaks in enumerate(_frame_peaks(salience)):
        used: set[int] = set()
        for contour in active:
            if t - contour[3] > gap:
                continue
            last = contour[0][-1]
            best = None
            for p in peaks:
                if p in used or abs(p - last) > LINK_BINS:
                    continue
                if best is None or abs(p - last) < abs(best - last):
                    best = p
            if best is not None:
                used.add(int(best))
                contour[0].append(int(best)); contour[1].append(float(salience[best, t])); contour[3] = t
        finished.extend(c for c in active if t - c[3] > gap)
        active = [c for c in active if t - c[3] <= gap]
        for p in peaks:
            if int(p) not in used:
                active.append([[int(p)], [float(salience[p, t])], t, t])
    finished.extend(active)
    min_len = int(round(MIN_CONTOUR_SEC / frame_sec))
    return [(c[0], c[1], c[2]) for c in finished if len(c[0]) >= min_len]


def extract_melody_contour(
    harmonic: np.ndarray, sample_rate: int, hop_length: int, n_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: melody pitch in MIDI semitones (NaN when unvoiced) and the id of its contour (-1)."""
    frame_sec = hop_length / float(sample_rate)
    salience, f0s = _salience(np.asarray(harmonic, dtype=np.float32), sample_rate, hop_length)
    contours = _contours(salience, frame_sec)
    if contours:
        means = np.array([np.mean(c[1]) for c in contours])
        keep = means >= means.mean() - VOICING_NU * means.std()
        contours = [c for c, k in zip(contours, keep) if k]
    T = salience.shape[1]
    best = np.zeros(T); f0_bin = np.full(T, -1); owner = np.full(T, -1)
    for index, (bins, saliences, start) in enumerate(contours):
        strength = float(np.mean(saliences))
        frames = np.arange(start, min(start + len(bins), T))
        take = strength > best[frames]
        best[frames[take]] = strength
        f0_bin[frames[take]] = np.asarray(bins)[: frames.size][take]
        owner[frames[take]] = index
    pitch = np.where(f0_bin >= 0, 12.0 * np.log2(f0s[np.maximum(f0_bin, 0)] / 440.0) + 69.0, np.nan)
    pitch = _fit(pitch, n_frames, np.nan)
    owner = _fit(owner.astype(np.float64), n_frames, -1.0).astype(np.int32)
    return pitch.astype(np.float32), owner


def _fit(values: np.ndarray, length: int, fill: float) -> np.ndarray:
    if values.size >= length:
        return values[:length]
    return np.concatenate([values, np.full(length - values.size, fill)])


def melody_note_starts(
    frame_times: np.ndarray, pitch: np.ndarray, contour: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Melody note starts (s) and held lengths (s).

    A start is a frame where a contour takes the melody after silence or with a
    pitch change >= 0.5 semitone, and keeps it for >= 100 ms; the note is held
    until the contour changes or the pitch moves >= 1 semitone from its start.
    """
    frame_times = np.asarray(frame_times, dtype=np.float64)
    T = min(frame_times.size, pitch.size, contour.size)
    if T < 2:
        return np.zeros(0), np.zeros(0)
    frame_sec = float(np.median(np.diff(frame_times[:T])))
    hold = int(round(0.1 / frame_sec))
    voiced = contour[:T] >= 0
    previous = np.concatenate([[-2], contour[: T - 1]])
    starts, lengths = [], []
    for i in np.flatnonzero(voiced & (previous != contour[:T])):
        if i + hold > T or np.any(contour[i:i + hold] != contour[i]):
            continue
        if i > 0 and voiced[i - 1] and abs(pitch[i] - pitch[i - 1]) < 0.5:
            continue
        k = i
        while k + 1 < T and contour[k + 1] == contour[i] and abs(pitch[k + 1] - pitch[i]) < 1.0:
            k += 1
        starts.append(frame_times[i]); lengths.append((k - i + 1) * frame_sec)
    return np.asarray(starts), np.asarray(lengths)
