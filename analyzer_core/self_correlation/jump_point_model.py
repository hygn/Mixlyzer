"""Learned jump point of a JumpCUE group and the per-candidate scores it uses.

The engine finds a group (passages that repeat) and a jump point on it; the
model then re-chooses, for the whole group at once, the beat to jump on and
the members to keep:

- candidates: the group shifted by d beats (|d| <= shift_beats, any beat;
  downbeats are a feature, not a constraint) x the full group and every
  subset of 2+ members whose pairs all continue alike over the first
  short_beats beats (the feasibility filter);
- features: pair scores (min / mean over the kept members' pairs) of the
  16-beat after-jump score, the short continuation and the before-jump
  agreement, the harmonic context and the short-time splice masking;
  per-beat features (min / mean over the kept members) of the phrase
  boundary probability, MFCC change, onset strength, melody onset, break
  depth and downbeat; the share of members kept; product / square terms;
- score: the features standardized over the group's candidates, weighted
  (assets/weights/default_jumpcue_jump_point.json shipped, see
  optimizer/jumpcue_parameter_optimizer.py).

Sections: feature terms, harmonic context, splice masking, the track-local
score cache shared with the rule stage, and the model.
"""
from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.beat_geometry import downbeat_beat_indices

from .vocal_context import JumpContextConfig, JumpPointScore, VocalContext, jump_point_scores


# ------------------------------------------------------------- feature terms
# The model stays linear in its fitted parameters; product and square terms
# express small, reviewed nonlinearities of features already computed.

def feature_term_value(base: Mapping[str, float], term: str) -> float:
    """Evaluate a base feature, ``a*b`` interaction, or ``a^2`` square."""
    if term in base:
        return float(base[term])
    if term.endswith("^2"):
        name = term[:-2]
        if name not in base:
            raise KeyError(f"unknown JumpCUE feature term: {term}")
        value = float(base[name])
        return value * value
    parts = term.split("*")
    if len(parts) == 2 and all(part in base for part in parts):
        return float(base[parts[0]]) * float(base[parts[1]])
    raise KeyError(f"unknown JumpCUE feature term: {term}")


def feature_vector(base: Mapping[str, float], terms: Sequence[str]) -> np.ndarray:
    """Return terms in model order as a contiguous float vector."""
    return np.ascontiguousarray([feature_term_value(base, term) for term in terms], dtype=float)


# --------------------------------------------------------- harmonic context
# Audio-only musical context of two candidate points: the phrase analyzer's
# beat-synchronous chroma + tonal-centroid features on both sides of the jump.
# No text, tags, filenames or other metadata enter these scores.

SEMANTIC_PAIR_SCORES = (
    "harmony_post",
    "harmony_pre",
    "chroma_post",
    "chroma_pre",
)


def _window(matrix: np.ndarray, beat: int, length: int, before: bool) -> np.ndarray:
    end = int(beat) if before else int(beat) + int(length)
    start = int(beat) - int(length) if before else int(beat)
    if start < 0 or end > matrix.shape[0] or end <= start:
        return np.empty((0, matrix.shape[1]), dtype=float)
    return np.asarray(matrix[start:end], dtype=float)


def context_cosine(
    matrix: np.ndarray,
    beat_a: int,
    beat_b: int,
    length: int = 16,
    *,
    before: bool = False,
    transposition_invariant: bool = False,
) -> float:
    """Mean beat-aligned cosine similarity of two audio-derived contexts.

    With ``transposition_invariant``, ``matrix`` must contain the twelve chroma
    dimensions and the best single pitch-class rotation is used for the whole
    context.  A single rotation preserves harmonic progression while allowing
    repeated sections that were transposed together.
    """
    x = _window(np.asarray(matrix), beat_a, length, before)
    y = _window(np.asarray(matrix), beat_b, length, before)
    if not x.size or x.shape != y.shape:
        return 0.0
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-9)
    if transposition_invariant:
        if x.shape[1] != 12:
            raise ValueError("transposition-invariant context must have 12 chroma columns")
        return float(max(np.mean(np.sum(x * np.roll(y, shift, axis=1), axis=1)) for shift in range(12)))
    return float(np.mean(np.sum(x * y, axis=1)))


def semantic_pair_scores(
    harmony: np.ndarray | None,
    beat_a: int,
    beat_b: int,
    length: int = 16,
    *,
    required: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Direct harmony and transposition-invariant chroma context scores."""
    selected = SEMANTIC_PAIR_SCORES if required is None else tuple(required)
    if harmony is None:
        return {name: 0.0 for name in selected}
    matrix = np.asarray(harmony, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 12:
        return {name: 0.0 for name in selected}
    chroma = matrix[:, :12]
    out: dict[str, float] = {}
    if "harmony_post" in selected:
        out["harmony_post"] = context_cosine(matrix, beat_a, beat_b, length)
    if "harmony_pre" in selected:
        out["harmony_pre"] = context_cosine(matrix, beat_a, beat_b, length, before=True)
    if "chroma_post" in selected:
        out["chroma_post"] = context_cosine(
            chroma, beat_a, beat_b, length, transposition_invariant=True
        )
    if "chroma_pre" in selected:
        out["chroma_pre"] = context_cosine(
            chroma, beat_a, beat_b, length, before=True, transposition_invariant=True
        )
    return out


# ----------------------------------------------------------- splice masking
# Short-time, band-wise masking proxy for a JumpCUE splice: a relative ranking
# feature, not a calibrated hearing threshold, at nominal unity-speed playback.
# The error is the outgoing continuation left by the player's linear crossfade,
# relative to a clean destination; nearby audio and attack excess in the same
# band mask it. Both directions and all channels count. Only short PCM
# snippets and band powers are retained, not the full track.

@dataclass(frozen=True)
class JumpMaskingConfig:
    crossfade_ms: float = 4.0       # PCMFeeder's rate // 250 declick crossfade
    frame_ms: float = 8.0
    hop_ms: float = 2.0
    before_ms: float = 20.0
    after_ms: float = 10.0
    forward_decay_ms: float = 15.0  # masker before the splice
    backward_decay_ms: float = 3.0  # masker after the splice
    masking_margin_db: float = 10.0
    attack_boost: float = 2.0

    def __post_init__(self) -> None:
        positive = (self.crossfade_ms, self.frame_ms, self.hop_ms,
                    self.forward_decay_ms, self.backward_decay_ms)
        nonnegative = (self.before_ms, self.after_ms, self.masking_margin_db, self.attack_boost)
        if any(not np.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("masking durations and decay constants must be positive and finite")
        if any(not np.isfinite(v) or v < 0 for v in nonnegative):
            raise ValueError("masking windows, margin and attack boost must be nonnegative and finite")


@dataclass
class JumpMaskingContext:
    heads: np.ndarray             # [beats, crossfade samples, channels]
    before: np.ndarray            # [beats, channels, bands], temporally weighted power
    after: np.ndarray
    attack: np.ndarray            # destination's power above its pre-beat baseline
    valid: np.ndarray
    bands: np.ndarray             # [bands, rfft bins]
    n_fft: int
    floor: float
    config: JumpMaskingConfig
    _pair_scores: dict[tuple[int, int], float] = field(default_factory=dict, init=False, repr=False)

    def _band_power(self, audio: np.ndarray) -> np.ndarray:
        # Parseval-normalized band power, with one-sided FFT energy weights.
        spec = np.fft.rfft(audio, n=self.n_fft, axis=-2)
        power = spec.real ** 2 + spec.imag ** 2
        power[..., 1:-1, :] *= 2.0
        return np.einsum("...fc,kf->...ck", power, self.bands) / (self.n_fft * audio.shape[-2])

    def pair_score(self, a: int, b: int) -> float:
        """[0, 1], larger = less exposed splice error; worst direction/channel."""
        if not (0 <= a < self.valid.size and 0 <= b < self.valid.size):
            return 0.0
        if not (self.valid[a] and self.valid[b]):
            return 0.0
        key = (min(a, b), max(a, b))
        if key not in self._pair_scores:
            self._pair_scores[key] = self._compute_pair_score(*key)
        return self._pair_scores[key]

    def _compute_pair_score(self, a: int, b: int) -> float:
        n = self.heads.shape[1]
        fade_out = 1.0 - (np.arange(n, dtype=float) + 0.5) / n
        error = self._band_power((self.heads[a] - self.heads[b]) * fade_out[:, None])
        margin = 10.0 ** (-self.config.masking_margin_db / 10.0)
        scores = []
        for source, target in ((a, b), (b, a)):
            masker = np.maximum(self.before[source], self.after[target])
            masker += self.config.attack_boost * self.attack[target]
            exposed = np.sum(error / (margin * masker + self.floor), axis=-1)
            scores.append(float(np.min(1.0 / (1.0 + exposed))))
        return min(scores)


def compute_jump_masking_context(
    audio: np.ndarray,
    sample_rate: int,
    beats_time: Sequence[float],
    *,
    config: JumpMaskingConfig | None = None,
) -> JumpMaskingContext:
    """Pool millisecond windows around each beat, preserving stereo channels.

    Windows entirely before the beat estimate its baseline. A destination
    attack receives extra masking power only near the splice; a transient
    late in the beat does not receive the beat-wide onset feature's credit.
    """
    cfg = config or JumpMaskingConfig()
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError("sample_rate must be positive")
    pcm = np.asarray(audio)
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    if pcm.ndim != 2 or pcm.shape[1] == 0:
        raise ValueError("masking audio must have shape [samples] or [samples, channels]")
    beats = np.asarray(beats_time, dtype=float)
    if beats.ndim != 1 or not np.isfinite(beats).all():
        raise ValueError("beats_time must be a finite one-dimensional array")

    def samples(ms: float) -> int:
        return max(1, int(sr * ms / 1000.0))

    fade, frame, hop = samples(cfg.crossfade_ms), samples(cfg.frame_ms), samples(cfg.hop_ms)
    n_fft = 1 << (max(fade, frame) - 1).bit_length()
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = (0.0, 250.0, 1000.0, 4000.0, np.inf)
    bands = np.array([(freqs >= lo) & (freqs < hi) for lo, hi in zip(edges[:-1], edges[1:])])
    shape = (beats.size, pcm.shape[1], len(edges) - 1)
    ctx = JumpMaskingContext(
        heads=np.zeros((beats.size, fade, pcm.shape[1]), dtype=np.float32),
        before=np.zeros(shape), after=np.zeros(shape), attack=np.zeros(shape),
        valid=np.zeros(beats.size, dtype=bool), bands=bands,
        n_fft=n_fft, floor=1e-20, config=cfg,
    )
    pre_offsets = np.arange(-int(sr * cfg.before_ms / 1000.0) - frame, -frame + 1, hop)
    post_offsets = np.arange(0, int(sr * cfg.after_ms / 1000.0) + 1, hop)
    offsets = np.concatenate((pre_offsets, post_offsets))
    centers_ms = (offsets + frame / 2 - fade / 2) * 1000.0 / sr
    decay = np.where(centers_ms < 0, cfg.forward_decay_ms, cfg.backward_decay_ms)
    weights = np.exp(-np.abs(centers_ms) / decay)
    window = np.hanning(frame) if frame > 2 else np.ones(frame)
    window /= np.sqrt(np.mean(window ** 2))

    for k, t in enumerate(beats):
        start = int(round(t * sr))
        if start + offsets[0] < 0 or start + max(fade, offsets[-1] + frame) > pcm.shape[0]:
            continue
        ctx.valid[k] = True
        ctx.heads[k] = np.nan_to_num(pcm[start:start + fade], nan=0.0, posinf=0.0, neginf=0.0)
        snippets = np.array([pcm[start + o:start + o + frame] for o in offsets], dtype=float)
        snippets = np.nan_to_num(snippets, nan=0.0, posinf=0.0, neginf=0.0)
        powers = ctx._band_power(snippets * window[None, :, None])
        pre = powers[:pre_offsets.size]
        weighted = powers * weights[:, None, None]
        ctx.before[k] = weighted[:pre_offsets.size].max(axis=0)
        ctx.after[k] = weighted[pre_offsets.size:].max(axis=0)
        baseline = np.median(pre, axis=0)
        excess = np.maximum(powers[pre_offsets.size:] - baseline, 0.0)
        ctx.attack[k] = (excess * weights[pre_offsets.size:, None, None]).max(axis=0)

    # Relative numerical floor keeps the feature invariant to a track's gain.
    peak = max(float(ctx.before.max(initial=0.0)), float(ctx.after.max(initial=0.0)))
    ctx.floor = max(peak * 1e-10, 1e-20)
    return ctx


# ---------------------------------------------------------------- score cache
# Track-local features and pair scores shared by the rule stage and the model.
# One instance per immutable VocalContext / beat grid; it does not survive an
# engine run, and score keys include the scoring parameters. The rule stage
# uses a penalized pre-jump score, the model the raw pre-jump agreement, so the
# full JumpPointScore is cached.

class JumpContextCache:
    def __init__(self, ctx: VocalContext):
        self.ctx = ctx
        self._scores: dict[tuple, dict[int, JumpPointScore | None]] = {}
        self._range_scores: dict[tuple, list[JumpPointScore]] = {}
        self._falls: dict[int, np.ndarray] = {}
        self._depths: dict[tuple[int, int], np.ndarray] = {}
        self._short: dict[tuple[int, int, int], float] = {}
        self._features: JumpPointFeatures | None = None
        self._beat_features: dict[str, np.ndarray] | None = None

    def scores(self, a_range: tuple[int, int], lag: int,
               config: JumpContextConfig | None = None) -> list[JumpPointScore]:
        cfg = config or JumpContextConfig()
        parameters = tuple(getattr(cfg, f.name) for f in fields(JumpContextConfig))
        # Unusual configurations can make the local window depend on the
        # requested span. Preserve that behavior rather than reusing scalars.
        if cfg.local_beats > cfg.window_beats:
            key = (parameters, lag, a_range)
            if key not in self._range_scores:
                self._range_scores[key] = jump_point_scores(self.ctx, a_range, lag, cfg)
            return self._range_scores[key]
        first = max(int(a_range[0]), int(lag))
        end = min(int(a_range[1]), self.ctx.beat_mel.shape[0])
        if lag <= 0 or end <= first:
            return []
        cached = self._scores.setdefault((parameters, int(lag)), {})
        missing = [k for k in range(first, end) if k not in cached]
        # Fill contiguous missing spans in one call, preserving the original
        # vectorized acoustic/formant comparisons and scalar score formula.
        i = 0
        while i < len(missing):
            start = missing[i]
            stop = start + 1
            i += 1
            while i < len(missing) and missing[i] == stop:
                stop += 1
                i += 1
            for k in range(start, stop):
                cached[k] = None
            for score in jump_point_scores(self.ctx, (start, stop), lag, cfg):
                cached[score.beat_a] = score
        return [cached[k] for k in range(first, end) if cached[k] is not None]

    def score(self, early: int, late: int) -> JumpPointScore | None:
        scores = self.scores((late, late + 1), late - early)
        return scores[0] if scores else None

    def level_fall(self, ref_beats: int = 8) -> np.ndarray:
        ref = max(1, int(ref_beats))
        if ref not in self._falls:
            level = np.asarray(self.ctx.beat_db, dtype=float)
            fall = np.zeros(level.size)
            for k in range(ref, level.size):
                fall[k] = float(np.median(level[k - ref:k])) - level[k]
            self._falls[ref] = fall
        return self._falls[ref]

    def break_depth(self, ref_beats: int = 8, width: int = 4) -> np.ndarray:
        key = (max(1, int(ref_beats)), max(1, int(width)))
        if key not in self._depths:
            fall = self.level_fall(key[0])
            self._depths[key] = np.array([fall[k:k + key[1]].max() for k in range(fall.size)])
        return self._depths[key]

    def beat_features(self, feats: JumpPointFeatures) -> dict[str, np.ndarray]:
        if self._features is not feats:
            self._features = feats
            self._beat_features = {
                "boundary": feats.boundary, "mfcc_change": feats.mfcc_change, "onset": feats.onset,
                "melody_onset": feats.melody_onset, "break_depth": self.break_depth(),
                "downbeat": feats.downbeat,
            }
        return self._beat_features

    def short_score(self, early: int, late: int, length: int) -> float:
        key = (min(early, late), max(early, late), int(length))
        if key not in self._short:
            mel = self.ctx.beat_mel
            self._short[key] = float(np.mean([mel[early + t] @ mel[late + t] for t in range(length)]))
        return self._short[key]


# --------------------------------------------------------------------- model

PAIR_SCORES = ("long", "short", "pre", "masking", *SEMANTIC_PAIR_SCORES)


@dataclass
class JumpPointFeatures:
    """Per-beat features of a track on its beat grid (``[B]`` arrays)."""

    boundary: np.ndarray       # phrase boundary probability
    mfcc_change: np.ndarray    # MFCC change, 4 beats after vs 4 beats before
    onset: np.ndarray          # onset strength on the beat
    melody_onset: np.ndarray   # melody onset on the beat
    downbeat: np.ndarray       # 1 on downbeats
    masking: Optional[JumpMaskingContext] = None
    harmony: Optional[np.ndarray] = None  # [B, 18] phrase chroma + tonnetz, audio only


def jump_point_beat_features(
    frames,
    beats_time: Sequence[float],
    tempo_segments: Optional[np.ndarray],
    boundary_probability: np.ndarray,
    *,
    audio: Optional[np.ndarray] = None,
    masking_config: Optional[JumpMaskingConfig] = None,
    phrase_feature_z: Optional[np.ndarray] = None,
) -> JumpPointFeatures:
    """Per-beat features from the frame features (``frames``, a FrameFeatures)
    and the phrase boundary probability, on ``beats_time``."""
    bt = np.asarray(beats_time, dtype=float)
    B = bt.size
    ft = np.asarray(frames.frame_times, dtype=float)
    tail = bt[-1] + (bt[-1] - bt[-2]) if B > 1 else bt[-1] + 0.5
    # Beat k spans frames [edges[k], edges[k + 1]).
    edges = np.searchsorted(ft, np.append(bt, tail))

    def span_mean(values: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.clip(a, 0, B)
        b = np.clip(b, 0, B)
        cs = np.concatenate([np.zeros((1,) + values.shape[1:]), np.cumsum(np.nan_to_num(values), axis=0)])
        cnt = (edges[b] - edges[a]).astype(float).reshape((-1,) + (1,) * (values.ndim - 1))
        s = cs[edges[b]] - cs[edges[a]]
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)

    k = np.arange(B)
    onset = span_mean(np.asarray(frames.librosa_onset, dtype=float), k, k + 1)
    melody = span_mean(np.asarray(frames.melody_onset, dtype=float), k, k + 1)
    mf = np.asarray(frames.mfcc[1:], dtype=float).T
    mfcc_change = np.linalg.norm(span_mean(mf, k, k + 4) - span_mean(mf, k - 4, k), axis=1)

    boundary = np.full(B, np.nan)
    prob = np.asarray(boundary_probability, dtype=float).ravel()[:B]
    boundary[: prob.size] = prob

    down = np.zeros(B)
    if tempo_segments is not None and np.asarray(tempo_segments).size:
        idx = np.asarray(downbeat_beat_indices(bt, np.asarray(tempo_segments, dtype=float)), dtype=int)
        down[idx[(idx >= 0) & (idx < B)]] = 1.0
    harmony = None
    if phrase_feature_z is not None:
        phrase_z = np.asarray(phrase_feature_z, dtype=float)
        # Phrase feature families are timbre(73), harmony(18), rhythm(27),
        # texture(15). Reuse the already-computed audio representation.
        if phrase_z.ndim == 2 and phrase_z.shape[0] == B and phrase_z.shape[1] >= 91:
            harmony = np.ascontiguousarray(phrase_z[:, 73:91])
    return JumpPointFeatures(
        boundary=np.nan_to_num(boundary),
        mfcc_change=np.nan_to_num(mfcc_change),
        onset=np.nan_to_num(onset),
        melody_onset=np.nan_to_num(melody),
        downbeat=down,
        masking=(compute_jump_masking_context(audio, frames.sample_rate, bt, config=masking_config)
                 if audio is not None else None),
        harmony=harmony,
    )


class JumpPointModel:
    """Weights of the conditional logit; ``choose`` picks a group's jump."""

    def __init__(self, names: List[str], weights: np.ndarray, short_threshold: float,
                 shift_beats: int, edge_beats: int, short_beats: int):
        self.names = list(names)
        self.weights = np.asarray(weights, dtype=float)
        self.short_threshold = float(short_threshold)
        self.shift_beats = int(shift_beats)
        self.edge_beats = int(edge_beats)
        self.short_beats = int(short_beats)
        self._active_harmony: Optional[np.ndarray] = None
        self._semantic_scores = tuple(
            name for name in SEMANTIC_PAIR_SCORES
            if any(name in feature for feature in self.names)
        )

    @classmethod
    def load(cls, path: str | Path) -> Optional["JumpPointModel"]:
        """The model at ``path``; None when the file is missing or not a model."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if data.get("format") != "mixlyzer-jumpcue-jump-point":
            return None
        return cls(data["features"], np.asarray(data["weights"], dtype=float), data["short_threshold"],
                   data["shift_beats"], data["edge_beats"], data["short_beats"])

    # ------------------------------------------------------------ candidates
    def _beat_features(self, ctx: VocalContext, feats: JumpPointFeatures,
                       cache: Optional[JumpContextCache] = None) -> Dict[str, np.ndarray]:
        cache = cache if cache is not None else JumpContextCache(ctx)
        if cache.ctx is not ctx:
            raise ValueError("JumpCUE cache belongs to another track context")
        return cache.beat_features(feats)

    def _shifts(self, ctx: VocalContext, beat_feats: Dict[str, np.ndarray], points: np.ndarray,
                masking: Optional[JumpMaskingContext] = None,
                cache: Optional[JumpContextCache] = None):
        mel = ctx.beat_mel
        B = mel.shape[0]
        cache = cache if cache is not None else JumpContextCache(ctx)
        if cache.ctx is not ctx:
            raise ValueError("JumpCUE cache belongs to another track context")
        pairs = list(itertools.combinations(range(points.size), 2))
        out = []
        shifts = [d for d in range(-self.shift_beats, self.shift_beats + 1)
                  if points.min() + d >= self.edge_beats and points.max() + d + self.edge_beats < B]
        if not shifts:
            return pairs, out
        # Score all shifts of a pair together. Existing rule-stage scores are
        # reused; only uncovered beats are computed, in contiguous batches.
        for i, j in pairs:
            cache.scores((int(points[j] + shifts[0]), int(points[j] + shifts[-1]) + 1),
                         int(points[j] - points[i]))
        for d in shifts:
            ks = points + d
            pair: Dict[str, Dict[Tuple[int, int], float]] = {name: {} for name in PAIR_SCORES}
            for i, j in pairs:
                s = cache.score(int(ks[i]), int(ks[j]))
                pair["long"][(i, j)] = s.score if s else -1.0
                pair["pre"][(i, j)] = max(s.acoustic_pre, 0.0) * s.lyric_pre if s else 0.0
                pair["short"][(i, j)] = cache.short_score(int(ks[i]), int(ks[j]), self.short_beats)
                pair["masking"][(i, j)] = masking.pair_score(int(ks[i]), int(ks[j])) if masking is not None else 0.0
                if self._active_harmony is not None:
                    semantic = semantic_pair_scores(
                        self._active_harmony, int(ks[i]), int(ks[j]), 16,
                        required=self._semantic_scores,
                    )
                    for name, value in semantic.items():
                        pair[name][(i, j)] = value
            out.append((d, ks, {n: v[ks] for n, v in beat_feats.items()}, pair))
        return pairs, out

    def _features(self, member: Dict[str, np.ndarray], pair, S: Tuple[int, ...], n: int) -> Dict[str, float]:
        Sp = list(itertools.combinations(S, 2))
        f: Dict[str, float] = {}
        for name in PAIR_SCORES:
            values = pair.get(name, {})
            v = np.array([values.get(p, 0.0) for p in Sp])
            f[f"{name}_min"], f[f"{name}_mean"] = float(v.min()), float(v.mean())
        idx = np.asarray(S)
        for name, v in member.items():
            f[f"{name}_min"], f[f"{name}_mean"] = float(v[idx].min()), float(v[idx].mean())
        f["share"] = len(S) / n
        return f

    def choose(
        self,
        ctx: VocalContext,
        feats: JumpPointFeatures,
        points: Sequence[int],
        allowed,
        *,
        cache: Optional[JumpContextCache] = None,
    ) -> Optional[Tuple[int, Tuple[int, ...]]]:
        """Best (shift d, kept member indices) for a group whose members jump
        at beats ``points``; ``allowed(beats)`` says whether jump beats keep the
        point limits. ``cache`` shares rule-stage scores and track features
        across groups; without it, a local cache is created for this call.
        None when no candidate keeps the limits."""
        points = np.asarray(points, dtype=int)
        n = points.size
        if n < 2:
            return None
        cache = cache if cache is not None else JumpContextCache(ctx)
        self._active_harmony = feats.harmony
        try:
            pairs, shifts = self._shifts(
                ctx, self._beat_features(ctx, feats, cache), points, feats.masking, cache
            )
        finally:
            self._active_harmony = None
        rows = []
        for d, ks, member, pair in shifts:
            ok = {p: pair["short"][p] >= self.short_threshold for p in pairs}
            subsets = [S for size in range(n, 1, -1) for S in itertools.combinations(range(n), size)
                       if all(ok[p] for p in itertools.combinations(S, 2))]
            full = tuple(range(n))
            if full not in subsets:
                subsets = [full] + subsets
            for S in subsets:
                rows.append((d, S, ks, self._features(member, pair, S, n)))
        if not rows:
            return None
        X = np.vstack([feature_vector(r[3], self.names) for r in rows])
        sd = X.std(axis=0)
        X = (X - X.mean(axis=0)) / np.where(sd > 1e-9, sd, 1.0)
        score = X @ self.weights
        for i in np.argsort(-score, kind="stable"):
            d, S, ks, _f = rows[int(i)]
            if allowed([int(ks[m]) for m in S]):
                return d, S
        return None
