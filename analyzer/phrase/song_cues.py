"""Two sets of per-beat cues the phrase model reads beside its features, on its beat grid (structure.PredictorGrid),
from the track's mono mix at SAMPLE_RATE (the analysis's decode: no stems, no second decode):

* vocal_cues: singing-voice cues without separation (after Lehner, Widmer & Sonnleitner, ICASSP 2014): the
  fluctogram (17 overlapping half-octave bands of a 20-cent CQT of the harmonic part, 200 Hz - 3.2 kHz: per frame the
  shift best matching the previous frame, its mean, spread, share of moving bands and maximum), the vocal variance
  (variance of MFCC 2-6 over ~1 s and ~4 s), the vocal band's spectral flatness and its share of the harmonic energy;
  each robust-z'd over the song as the phrase features are.
* style_cues: mixability and song-style cues, NOT normalized within the song (the phrase features all are, so a
  model could not tell an electronic track from a ballad): per beat the kick on the beat and over its bar, the
  harmonic share, the chroma entropy, the mel loop similarity with 4 and 8 beats before, the loudness against the
  song's 95th percentile, the onset density; per song (repeated) its harmonic share, kick regularity, tempo, bar
  loudness range, vocal-band share and share of beats with a strong kick.

Their frame-level part needs no beats (cue_frames: made beside the beat tracking), the rest is per beat. As the
research extracted them for training (research/benchmark/phrase_boundary/vocal_feats.py, style_feats.py): the same
values, by the same operations in the same order, but for the HPSS's two median filters (both cues' HPSS read the
same STFT magnitude): made once, by analyzer.audio.hpss's sliding median (scipy's values, ~20x faster)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analyzer.audio.hpss import hpss_medians

SAMPLE_RATE = 22050
HOP = 512
N_FFT = 2048
VOCAL_NAMES = ("fluct_mean", "fluct_std", "fluct_active", "vocal_var", "flatness_vocal", "vocal_share",
               "fluct_max_band", "vocal_var_slow")
STYLE_NAMES = ("kick_on_beat", "kick_bar", "harmonic_share", "chroma_entropy", "loop4", "loop8", "loud_db",
               "onset_density", "song_harmonic", "song_kick", "song_bpm", "song_dyn_db", "song_vocal_band",
               "song_kick_share")


@dataclass(frozen=True)
class CueFrames:
    """The cues' frame-level part of a song (HOP frames at SAMPLE_RATE)."""
    vocal: np.ndarray            # (frames, 8): the vocal cues per frame
    kick: np.ndarray             # the low band's percussive onset envelope
    onsets: np.ndarray           # 1 on the frames of a picked onset
    harmonic_share: np.ndarray   # the harmonic part's share of the energy
    vocal_band: np.ndarray       # the harmonic vocal band's share of the energy
    entropy: np.ndarray          # the harmonic chroma's entropy
    mel: np.ndarray              # (64, frames) mel dB
    rms: np.ndarray


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=0)
    mad = 1.4826 * np.median(np.abs(x - med), axis=0)
    return (x - med) / np.where(mad > 1e-8, mad, np.std(x, axis=0) + 1e-8)


def _runvar(x, w):
    k = np.ones(w) / w
    m = np.convolve(x, k, "same")
    return np.convolve(x * x, k, "same") - m * m


def cue_frames(y: np.ndarray) -> CueFrames:
    """The frame-level cues of the mono mix ``y`` at SAMPLE_RATE."""
    import librosa
    sr, hop = SAMPLE_RATE, HOP
    y = np.asarray(y, np.float32)
    # one STFT for both: librosa.effects.harmonic's (its defaults) and style's (n_fft 2048, hop 512) are the same
    D = librosa.stft(y)
    S, phase = librosa.magphase(D)
    # librosa.decompose.hpss's median filters, in its arrays' memory layout (empty_like the magnitude: the sums and
    # products after them add in the same order)
    medians = hpss_medians(S)
    harm, perc = np.empty_like(S), np.empty_like(S)
    harm[:], perc[:] = medians
    # vocal: the harmonic part, librosa.effects.harmonic(y, margin=2.0)
    mask = librosa.util.softmask(harm, perc * 2.0, power=2.0, split_zeros=False)
    yh = librosa.istft((S * mask) * phase, dtype=y.dtype, n_fft=N_FFT, length=y.shape[-1])
    C = np.log1p(30 * np.abs(librosa.cqt(yh, sr=sr, hop_length=hop, fmin=200.0, n_bins=240, bins_per_octave=60)))
    T = C.shape[1]
    # fluctogram: per band and frame the shift (bins of 20 cents) best matching the previous frame
    shifts = np.arange(-5, 6)
    flu = np.zeros((17, T))
    for k in range(17):
        a, b = 12 * k + 6, 12 * k + 6 + 30                      # half-octave bands, a 12-bin hop, 5-bin margins
        cur = C[a:b, 1:]
        cur = cur - cur.mean(0)
        best = np.full(T - 1, -np.inf)
        arg = np.zeros(T - 1)
        for s in shifts:
            prev = C[a + s:b + s, :-1]
            prev = prev - prev.mean(0)
            r = (cur * prev).sum(0) / (np.linalg.norm(cur, axis=0) * np.linalg.norm(prev, axis=0) + 1e-9)
            upd = r > best
            best = np.where(upd, r, best)
            arg = np.where(upd, s, arg)
        flu[k, 1:] = np.abs(arg) * (best > 0.5)
    M = np.abs(D)
    power = M ** 2
    Sh = np.abs(librosa.stft(yh, n_fft=N_FFT, hop_length=hop)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    band = (f >= 200) & (f <= 3200)
    flat = np.exp(np.mean(np.log(power[band] + 1e-12), 0)) / (np.mean(power[band], 0) + 1e-12)
    share = Sh[band].sum(0) / (Sh.sum(0) + 1e-12)
    mf = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=power, sr=sr, n_mels=40)),
                              n_mfcc=7)
    n = min(T, power.shape[1], mf.shape[1])
    vv = np.mean([_runvar(mf[i, :n], 43) for i in range(1, 6)], 0)
    vv_slow = np.mean([_runvar(mf[i, :n], 172) for i in range(1, 6)], 0)
    vocal = np.stack([flu[:, :n].mean(0), flu[:, :n].std(0), (flu[:, :n] > 0).mean(0), np.log(vv + 1e-6),
                      flat[:n], share[:n], flu[:, :n].max(0), np.log(vv_slow + 1e-6)], 1)
    # style: librosa.decompose.hpss(M, margin=1.0) of the magnitude
    H = (M * librosa.util.softmask(harm, perc * 1.0, power=2.0, split_zeros=True)) * 1
    P = (M * librosa.util.softmask(perc, harm * 1.0, power=2.0, split_zeros=True)) * 1
    Ts = M.shape[1]
    low = (f >= 30) & (f <= 150)
    kick_env = np.maximum(0, np.diff(np.log1p(100 * (P[low] ** 2).sum(0)), prepend=0))
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(M ** 2), sr=sr)
    peaks = librosa.util.peak_pick(onset, pre_max=3, post_max=3, pre_avg=10, post_avg=10, delta=0.3, wait=4)
    ponset = np.zeros(Ts)
    ponset[peaks[peaks < Ts]] = 1
    hs = (H ** 2).sum(0) / ((M ** 2).sum(0) + 1e-12)
    vs = (H[band] ** 2).sum(0) / ((M ** 2).sum(0) + 1e-12)
    chroma = librosa.feature.chroma_stft(S=H ** 2, sr=sr)
    cp = chroma / (chroma.sum(0, keepdims=True) + 1e-12)
    ent = -(cp * np.log(cp + 1e-12)).sum(0) / np.log(12)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(S=M ** 2, sr=sr, n_mels=64))
    rms = np.sqrt((M ** 2).mean(0))
    return CueFrames(vocal=vocal, kick=kick_env, onsets=ponset, harmonic_share=hs, vocal_band=vs, entropy=ent,
                     mel=mel, rms=rms)


def vocal_cues(frames: CueFrames, beat_times: np.ndarray) -> np.ndarray:
    """(beats, 8) float32 on the grid's beat times (VOCAL_NAMES)."""
    import librosa
    bt = np.asarray(beat_times, float)
    frame = frames.vocal
    n = frame.shape[0]
    ft = librosa.frames_to_time(np.arange(n), sr=SAMPLE_RATE, hop_length=HOP)
    edges = np.r_[bt, bt[-1] + np.median(np.diff(bt))]
    idx = np.searchsorted(ft, edges)
    out = np.stack([frame[idx[i]:max(idx[i + 1], idx[i] + 1)].mean(0) if idx[i] < n else frame[-1]
                    for i in range(bt.size)])
    return _robust_z(out).astype(np.float32)


def style_cues(frames: CueFrames, beat_times: np.ndarray) -> np.ndarray:
    """(beats, 14) float32 on the grid's beat times (STYLE_NAMES)."""
    import librosa
    sr, hop = SAMPLE_RATE, HOP
    bt = np.asarray(beat_times, float)
    T = frames.rms.size
    ft = librosa.frames_to_time(np.arange(T), sr=sr, hop_length=hop)
    kick_env, mel = frames.kick, frames.mel
    n = bt.size
    period = float(np.median(np.diff(bt))) if n > 1 else 0.5
    edges = np.r_[bt, bt[-1] + period]
    idx = np.clip(np.searchsorted(ft, edges), 0, T)
    win = max(1, int(round(0.07 * sr / hop)))                     # +-70 ms around the beat for the kick

    def span(x, i):
        a, b = idx[i], max(idx[i + 1], idx[i] + 1)
        return x[a:b] if a < T else x[-1:]

    kick_on = np.zeros(n)
    for i in range(n):
        c = int(np.clip(np.searchsorted(ft, bt[i]), 0, T - 1))
        local = kick_env[max(0, c - win):c + win + 1].max()
        kick_on[i] = local / (span(kick_env, i).mean() + 1e-6)
    beat_mel = np.stack([mel[:, idx[i]:max(idx[i + 1], idx[i] + 1)].mean(1) if idx[i] < T else mel[:, -1]
                         for i in range(n)])
    bm = beat_mel - beat_mel.mean(1, keepdims=True)
    bm /= np.linalg.norm(bm, axis=1, keepdims=True) + 1e-9

    def lag_sim(k):
        out = np.zeros(n)
        out[k:] = (bm[k:] * bm[:-k]).sum(1)
        out[:k] = out[k:2 * k].mean() if n > 2 * k else 0
        return out

    def per(x):
        return np.array([span(x, i).mean() for i in range(n)])

    beat_rms = per(frames.rms)
    loud = 20 * np.log10(beat_rms + 1e-9) - 20 * np.log10(np.percentile(beat_rms, 95) + 1e-9)
    dens = np.array([span(frames.onsets, i).sum() for i in range(n)])
    kick_bar = np.convolve(kick_on, np.ones(4) / 4, "same")
    harm = per(frames.harmonic_share)
    bars = 20 * np.log10(np.convolve(beat_rms, np.ones(4) / 4, "valid")[::4] + 1e-9)
    song = [harm.mean(), np.median(kick_on), 60.0 / period, np.percentile(bars, 95) - np.percentile(bars, 10),
            per(frames.vocal_band).mean(), float(np.mean(kick_on > 2.0))]
    out = np.column_stack([kick_on, kick_bar, harm, per(frames.entropy), lag_sim(4), lag_sim(8), loud, dens]
                          + [np.full(n, v) for v in song])
    return out.astype(np.float32)
