"""JumpCUE: two places in a track that can be jumped between, either way, each a hot cue.

Two steps. ``audio_context`` reads the audio and the beats alone (the repeats, analyzer.jumpcue.repeats: the passages
that come again, transposed or not; and the vocal context), so the full analysis makes it beside the phrases.
``jump_cue_pairs`` then scores beat pairs and chooses the sure one with the phrases, under the user's rules
(analyzer.jumpcue.pairs): the repeats add candidates, the phrases give the others."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from analyzer.audio.hpss import HPSS_HOP_LENGTH, HPSS_N_FFT, HpssSpectra
from analyzer.jumpcue import pairs as P
from analyzer.jumpcue.repeats import repeats
from analyzer.jumpcue.vocal_context import VocalContext, VocalContextConfig, compute_vocal_context
from analyzer.settings import JumpCueRules


@dataclass(frozen=True)
class JumpContext:
    """What JumpCUE reads of the audio and the beats alone."""
    links: np.ndarray                  # [N, 5] the repeats (analyzer.jumpcue.repeats), best first
    vocal: Optional[VocalContext]      # None: too few beats


def audio_context(y_harm: np.ndarray, y_mix: np.ndarray, sr: int, beats_time: Sequence[float], *,
                  spectra: Optional[HpssSpectra] = None) -> JumpContext:
    """The repeats of the harmonic part and the vocal context of the mix (``spectra``: the mix's
    HPSS spectra: their harmonic magnitude read for the repeats, their magnitude reused when its STFT is the vocal
    context's; without them the harmonic part's own STFT)."""
    beats = np.asarray(beats_time, dtype=float)
    if spectra is not None:
        links = repeats(spectra.harmonic, spectra.sample_rate, spectra.n_fft, spectra.hop_length, beats)
    else:
        from analyzer.audio.spectral import stft
        harmonic = np.abs(stft(y_harm, n_fft=HPSS_N_FFT, hop_length=HPSS_HOP_LENGTH, center=True, pad_mode="constant"))
        links = repeats(harmonic, sr, HPSS_N_FFT, HPSS_HOP_LENGTH, beats)
    if beats.size < 2:
        return JumpContext(links, None)
    vc = VocalContextConfig()
    magnitude = (spectra.magnitude if spectra is not None and (spectra.n_fft, spectra.hop_length) == (vc.n_fft, vc.hop)
                 else None)
    return JumpContext(links, compute_vocal_context(y_mix, sr, beats, magnitude=magnitude, config=vc))


def jump_cue_pairs(context: JumpContext, *, model_path: str, beats_time: np.ndarray, tempo_segments: np.ndarray,
                   phrases: Sequence, frames, key_frames: np.ndarray, rules: JumpCueRules = JumpCueRules(),
                   cue_times: Sequence[float] = ()) -> list[dict]:
    """The automatic JumpCUE, one pair or none: [{forward, backward (each label, start, end, point), lag_beats,
    lag_sec, score, confidence (both the pair's score)}].

    ``model_path``: the pair model (pairs.FORMAT); ``phrases``: the track's (start, end, label): the full analysis's
    own, the editor's as they are; ``frames``: the analysis's frame features; ``key_frames``: the keys' frame chroma;
    ``rules``: where the pair may go (Settings > Analysis > CUE); ``cue_times``: the track's CUE points (s), which the
    pair may not wholly sit on."""
    if context.vocal is None:
        return []
    beats = np.asarray(beats_time, dtype=float)
    feats = P.beat_features(frames, key_frames, beats, context.vocal, context.links)
    t = P.pair_track(beats, tempo_segments, phrases, feats)
    pairs = P.candidates(t)
    if pairs.shape[0] == 0:
        return []
    X, names = P.features(t, pairs)
    models = P.load_models(model_path)
    if names != models.pair_names:
        raise ValueError("the pair model's features differ from the app's")
    scores = P.pair_scores(X, models)
    cue_beats = frozenset(int(np.argmin(np.abs(beats - c))) for c in cue_times)
    chosen = P.choose_pair(t, pairs, scores, rules, cue_beats)
    if chosen is None:
        return []
    p, q, score = chosen
    n = t.n

    def at(i):
        return float(beats[min(max(int(i), 0), n - 1)])

    def point(label, b):
        return {"label": label, "start": at(b), "end": at(b + P.END_BEATS), "point": at(b)}
    print(f"[JumpCUE] pair {int(at(p) // 60)}:{at(p) % 60:04.1f} - {int(at(q) // 60)}:{at(q) % 60:04.1f}, score {score:.3f}")
    return [{"forward": point("B", q), "backward": point("A", p), "lag_beats": float(q - p), "lag_sec": at(q) - at(p),
             "score": score, "confidence": score}]
