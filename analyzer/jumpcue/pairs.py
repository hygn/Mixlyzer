"""JumpCUE's pair model, four steps:

  1. candidates (``candidates``): beat pairs along the repeats (analyzer.jumpcue.repeats), between phrases of the same
     family at the same place, and from a chorus's or an interlude's head to the outro's head (a jump into the ending);
  2. features (``features``): per pair, how alike the music is before and after the two beats and how each junction
     sounds, from the per-beat arrays the analysis already has (``beat_features``) and the phrases;
  3. score (``pair_scores``): one gradient-boosting model;
  4. the automatic JumpCUE (``choose_pair``): the best-scored pair the user's rules allow (analyzer.settings.JumpCueRules,
     Settings > Analysis > CUE: the sections of its earlier and its later point, by default the first chorus and the
     last, and how many beats around a section still count as in it; the score it needs; its points' gap, none early in
     the first round, none wholly on CUE points), none under the score asked for.

One pair, the sure one: a second group or a third point was right far less often (the tracks reviewed whole,
2026-11-11). The key is not read: the editor's reanalysis has none, so the model is trained without it. The model is
trained by training/jumpcue_optimizer.py, on the same functions."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np

from analyzer.settings import JumpCueRules

FORMAT = "mixlyzer_jumpcue_pairs_v2"     # v2: the repeats of analyzer.jumpcue.repeats, one model

FAMILY = {"CHORUS": "chorus", "BREAK_CHORUS": "chorus", "VERSE": "verse", "BRIDGE": "bridge", "INTRO": "intro",
          "INTERLUDE": "interlude", "OUTRO": "outro"}
LABELS = ("LEAD_IN", "INTRO", "VERSE", "CHORUS", "BREAK_CHORUS", "BRIDGE", "OUTRO", "INTERLUDE", "SILENCE")
STRUCT_WINDOW = 8          # beats around a phrase start, for the structure candidates
LINK_MARGIN = 8            # beats beyond a link's ends
MIN_LAG = 8                # beats between a pair's points at least
END_BEATS = 16             # a point's passage, shown from the point
SUNG_FAMILIES = ("verse", "chorus", "bridge")
CUE_MOVE_BEATS = 8         # how many beats earlier a pair wholly on CUE points is offered
KEY_FRAME_SEC = 512.0 / 22050.0   # the keys' frame chroma's hop (analyzer.key.chord_key.frame_chroma)


@dataclass
class PairTrack:
    beats: np.ndarray
    bar_phase: np.ndarray          # beats since the bar's downbeat (-1: before the first)
    downbeat: np.ndarray           # bool
    phrase_start: np.ndarray       # [P] beat index
    phrase_label: list
    beat_label: np.ndarray         # [B] label index per beat (-1 none)
    key: np.ndarray                # [B] -1: unknown (the models are trained without the key)
    feats: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.beats.size)


def quarters(values, frame_times, beats):
    """[B, 4] mean of a frame series over each quarter of each beat (the last beat: its median length)."""
    b = np.asarray(beats, dtype=float)
    step = np.diff(b)
    ends = np.append(b[1:], b[-1] + (np.median(step) if step.size else 0.5))
    out = np.zeros((b.size, 4), dtype=np.float32)
    for q in range(4):
        s = b + (ends - b) * q / 4
        e = b + (ends - b) * (q + 1) / 4
        i0 = np.searchsorted(frame_times, s)
        i1 = np.maximum(np.searchsorted(frame_times, e), i0 + 1)
        c = np.concatenate([[0.0], np.cumsum(values)])
        i1 = np.minimum(i1, values.size)
        i0 = np.minimum(i0, i1 - 1)
        out[:, q] = (c[i1] - c[i0]) / np.maximum(i1 - i0, 1)
    return out


def beat_features(frames, key_frames: np.ndarray, beats: np.ndarray, vocal, links) -> dict:
    """The per-beat arrays the pair features read (their dtypes as the models were trained on them): the frame
    features and the keys' frame chroma (its STFT chroma of the harmonic part, blocks 24..35) quartered, the vocal
    context's beats, the repeats ([N, 5]); nothing recomputed."""
    level = frames.rms_db.astype(float)
    ft = frames.frame_times
    low, _mid, high = frames.band_ratios
    cft = np.arange(key_frames.shape[1]) * KEY_FRAME_SEC
    return {
        "links": np.asarray(links if links is not None else [], dtype=np.float64).reshape(-1, 5),
        "v_beat_mel": vocal.beat_mel.astype(np.float16),
        "v_beat_db": vocal.beat_db.astype(np.float32),
        "v_beat_ceps": vocal.beat_ceps.astype(np.float16),
        "q_vocal": quarters(vocal.gap_activity.astype(float), vocal.frame_times, beats),
        "q_level": quarters(level, ft, beats),
        "q_low": quarters(level + 10.0 * np.log10(np.maximum(low.astype(float), 1e-6)), ft, beats),
        "q_high": quarters(level + 10.0 * np.log10(np.maximum(high.astype(float), 1e-6)), ft, beats),
        "q_flat": quarters(frames.flatness.astype(float), ft, beats),
        "q_onset": quarters(frames.librosa_onset.astype(float), ft, beats),
        "chroma": np.stack([quarters(key_frames[24 + k].astype(float), cft, beats).mean(1) for k in range(12)], axis=1),
    }


def pair_track(beats: np.ndarray, tempo_segments: np.ndarray, phrases: Sequence, feats: dict) -> PairTrack:
    """``phrases``: (start, end, label) of the analysis's own phrases."""
    from analyzer.music.beat_geometry import downbeat_beat_indices

    beats = np.asarray(beats, dtype=float)
    starts = np.asarray([float(s) for s, _e, _l in phrases], float)
    ends = np.asarray([float(e) for _s, e, _l in phrases], float)
    labels = [str(lab) for _s, _e, lab in phrases]
    down = np.asarray(sorted(downbeat_beat_indices(beats, np.asarray(tempo_segments, float)).tolist()), int)
    is_down = np.zeros(beats.size, bool)
    is_down[down[(down >= 0) & (down < beats.size)]] = True
    last = np.searchsorted(down, np.arange(beats.size), side="right") - 1
    bar_phase = np.where(last >= 0, np.arange(beats.size) - down[np.maximum(last, 0)], -1)
    p_start = np.asarray([int(np.argmin(np.abs(beats - s))) for s in starts], int)
    beat_label = np.full(beats.size, -1, int)
    for s, e, lab in zip(starts, ends, labels):
        inside = (beats >= s - 1e-6) & (beats < e - 1e-6)
        beat_label[inside] = LABELS.index(lab) if lab in LABELS else -1
    return PairTrack(beats, bar_phase, is_down, p_start, labels, beat_label, np.full(beats.size, -1, int), feats)


def candidates(t, cross_ending: bool = True) -> np.ndarray:
    """[C, 2] beat pairs p < q: along every repeat (its lag, LINK_MARGIN beats beyond its ends), and
    between phrases of the same family (the same place within STRUCT_WINDOW beats of their starts); with
    ``cross_ending``, also a chorus's or an interlude's head to the outro's head; the same place in its bar either
    way, at least MIN_LAG beats apart."""
    n = t.n
    out = set()
    for a0, a1, b0, b1, _s in t.feats["links"]:
        lag = int(round(b0 - a0))
        if lag == 0:
            continue
        lo, hi = int(round(min(a0, a1))) - LINK_MARGIN, int(round(max(a0, a1))) + LINK_MARGIN
        for p in range(max(0, lo), min(n, hi)):
            q = p + lag
            if 0 <= q < n:
                out.add((min(p, q), max(p, q)))
    fam = [FAMILY.get(lab, "") for lab in t.phrase_label]
    for i, j in itertools.combinations(range(len(t.phrase_start)), 2):
        same = bool(fam[i]) and fam[i] == fam[j]
        if not (same or (cross_ending and fam[j] == "outro" and fam[i] in ("chorus", "interlude"))):
            continue
        si, sj = int(t.phrase_start[i]), int(t.phrase_start[j])
        for k in (range(-STRUCT_WINDOW, STRUCT_WINDOW + 1) if same else (0,)):
            p, q = si + k, sj + k
            if 0 <= p < n and 0 <= q < n:
                out.add((min(p, q), max(p, q)))
    arr = np.asarray(sorted(out), int).reshape(-1, 2)
    keep = (arr[:, 1] - arr[:, 0] >= MIN_LAG) & (t.bar_phase[arr[:, 0]] == t.bar_phase[arr[:, 1]])
    return arr[keep]


def _unit(x):
    x = np.asarray(x, float)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def _cos_window(F, p, q, offsets):
    n = F.shape[0]
    vals = []
    for k in offsets:
        a, b = p + k, q + k
        ok = (a >= 0) & (a < n) & (b >= 0) & (b < n)
        v = np.full(p.shape, np.nan)
        v[ok] = np.sum(F[a[ok]] * F[b[ok]], axis=1)
        vals.append(v)
    V = np.stack(vals, 1)
    return np.nanmean(V, 1), np.nanmin(V, 1)


def features(t, pairs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """[C, F] features of ``pairs`` and their names: how alike the music is before and after the two beats (timbre,
    harmony with the best of 12 chroma rotations, the voice's formants), each junction (as the pair's min and max: an
    undirected pair), the relation, the join as heard against the song's own flow, and where the pair ranks among its
    neighbours on its lag."""
    f = t.feats
    p, q = pairs[:, 0], pairs[:, 1]
    n = t.n
    cols: dict[str, np.ndarray] = {}
    pre, post = range(-4, 0), range(0, 4)
    mel = _unit(f["v_beat_mel"].astype(float))
    for name, offs in (("pre", pre), ("post", post)):
        mean, low = _cos_window(mel, p, q, offs)
        cols[f"timbre_{name}"], cols[f"timbre_{name}_min"] = mean, low
    chroma = f["chroma"].astype(float)
    best = np.full(p.shape, -np.inf)
    rot = np.zeros(p.shape, int)
    plain = {}
    for r in range(12):
        C = _unit(np.roll(chroma, r, axis=1))
        A = _unit(chroma)
        vals = []
        for k in itertools.chain(pre, post):
            a, b = np.clip(p + k, 0, n - 1), np.clip(q + k, 0, n - 1)
            vals.append(np.sum(A[a] * C[b], 1))
        s = np.mean(vals, 0)
        if r == 0:
            plain["pre"], _ = _cos_window(A, p, q, pre)
            plain["post"], _ = _cos_window(A, p, q, post)
        better = s > best
        best[better], rot[better] = s[better], r
    cols["harmony_pre"], cols["harmony_post"] = plain["pre"], plain["post"]
    cols["harmony_rotated"] = best
    cols["rotation"] = np.minimum(rot, 12 - rot).astype(float)
    ceps = _unit(f["v_beat_ceps"].astype(float).mean(1))
    cols["voice_pre"], _ = _cos_window(ceps, p, q, pre)
    cols["voice_post"], _ = _cos_window(ceps, p, q, post)

    def junction(i):
        lv, lo = f["q_level"].astype(float), f["q_low"].astype(float)
        prev = np.clip(i - 1, 0, n - 1)
        prev2 = np.clip(i - 2, 0, n - 1)
        before_level = (lv[prev].mean(1) + lv[prev2].mean(1)) / 2
        before_low = (lo[prev].mean(1) + lo[prev2].mean(1)) / 2
        vq = f["q_vocal"]
        dist = np.full(i.shape, 99.0)
        if t.phrase_start.size:
            d = i[:, None] - t.phrase_start[None, :]
            dist = d[np.arange(i.size), np.argmin(np.abs(d), axis=1)].astype(float)
        return {
            "dropout": before_level - lv[i, :2].mean(1),
            "low_dropout": before_low - lo[i, :2].mean(1),
            "flatness": f["q_flat"][i, :2].mean(1).astype(float),
            "onset": f["q_onset"][i, 0].astype(float),
            "voice_across": np.minimum(vq[prev, 3], vq[i, 0]).astype(float),
            "downbeat": t.downbeat[i].astype(float),
            "phrase_dist": np.clip(dist, -16, 16),
            "phrase_head": (np.abs(dist) == 0).astype(float),
        }
    jp, jq = junction(p), junction(q)
    for k in jp:
        cols[f"{k}_min"], cols[f"{k}_max"] = np.minimum(jp[k], jq[k]), np.maximum(jp[k], jq[k])
    for ph in range(4):
        cols[f"bar_phase_{ph}"] = (t.bar_phase[p] == ph).astype(float)
    cols["phrase_dist_same"] = (jp["phrase_dist"] == jq["phrase_dist"]).astype(float)

    lp, lq = t.beat_label[p], t.beat_label[q]
    fam = np.asarray([FAMILY.get(LABELS[x], "") if x >= 0 else "" for x in range(-1, len(LABELS))], object)
    cols["same_label"] = ((lp == lq) & (lp >= 0)).astype(float)
    cols["same_family"] = ((fam[lp + 1] == fam[lq + 1]) & (fam[lp + 1] != "")).astype(float)
    for lab in ("CHORUS", "BREAK_CHORUS", "VERSE", "BRIDGE", "INTERLUDE", "OUTRO"):
        x = LABELS.index(lab)
        cols[f"has_{lab.lower()}"] = ((lp == x) | (lq == x)).astype(float)
    kp, kq = t.key[p], t.key[q]
    cols["key_change"] = ((kp != kq) & (kp >= 0) & (kq >= 0)).astype(float)
    db = f["v_beat_db"].astype(float)
    c = np.concatenate([[0.0], np.cumsum(db)])

    def after(i):
        return (c[np.minimum(i + 16, n)] - c[i]) / np.maximum(np.minimum(i + 16, n) - i, 1)
    cols["level_after_diff"] = np.abs(after(p) - after(q))
    cols["lag_beats"] = (q - p).astype(float)
    cols["late"] = q / max(n - 1, 1)
    lo = f["q_low"].astype(float)
    nxt_p, nxt_q = np.clip(p + 1, 0, n - 1), np.clip(q + 1, 0, n - 1)
    cols["low_next_min"] = (np.minimum(lo[nxt_p].mean(1), lo[nxt_q].mean(1))
                            - np.maximum(lo[np.clip(p - 1, 0, n - 1)].mean(1), lo[np.clip(q - 1, 0, n - 1)].mean(1)))

    def join_excess(a, b):
        pa = np.clip(a - 1, 0, n - 1)
        timbre = np.sum(mel[pa] * mel[a], 1) - np.sum(mel[pa] * mel[b], 1)
        bands = np.stack([f["q_level"], f["q_low"], f["q_high"]], -1).astype(float)
        end = bands[pa, 3]
        quarter = np.abs(end - bands[b, 0]).sum(1) - np.abs(end - bands[a, 0]).sum(1)
        return timbre, quarter
    t_pq, q_pq = join_excess(p, q)
    t_qp, q_qp = join_excess(q, p)
    cols["join_timbre_max"], cols["join_timbre_mean"] = np.maximum(t_pq, t_qp), (t_pq + t_qp) / 2
    cols["join_bands_max"], cols["join_bands_mean"] = np.maximum(q_pq, q_qp), (q_pq + q_qp) / 2
    for name in ("timbre_pre", "timbre_post", "harmony_post", "harmony_rotated", "voice_post"):
        v = cols[name]
        order = np.argsort(np.argsort(np.nan_to_num(v, nan=-9.0)))
        cols[f"{name}_pct"] = order / max(len(v) - 1, 1)
    lag = q - p
    region_keys = ("dropout_max", "low_dropout_max", "phrase_head_max", "join_timbre_mean",
                   "join_bands_mean", "timbre_post", "voice_across_min")
    reg = {k: np.zeros(p.shape) for k in region_keys}
    by_lag: dict[int, list] = {}
    for i, L in enumerate(lag.tolist()):
        by_lag.setdefault(L, []).append(i)
    for L, idx in by_lag.items():
        idx = np.asarray(idx)
        idx = idx[np.argsort(p[idx])]
        pp = p[idx]
        for a, i in enumerate(idx):
            lo_, hi_ = np.searchsorted(pp, pp[a] - 8), np.searchsorted(pp, pp[a] + 8, side="right")
            nb = idx[lo_:hi_]
            for k in region_keys:
                v = cols[k]
                reg[k][i] = (np.sum(v[nb] < v[i]) + 0.5 * np.sum(v[nb] == v[i])) / max(len(nb), 1)
    for k in region_keys:
        cols[f"{k}_in_region"] = reg[k]
    on_link = np.zeros(p.shape)
    for a0, a1, b0, _b1, s in f["links"]:
        lag_ = int(round(b0 - a0))
        lo2, hi2 = min(a0, a1) - LINK_MARGIN, max(a0, a1) + LINK_MARGIN
        hit = ((q - p) == abs(lag_)) & (np.minimum(p, q) >= lo2) & (np.minimum(p, q) <= hi2)
        on_link = np.maximum(on_link, np.where(hit, s, 0.0))
    cols["link_score"] = on_link
    X = np.stack([np.nan_to_num(np.asarray(v, float), nan=0.0) for v in cols.values()], 1)
    return X, list(cols)


@dataclass(frozen=True)
class PairModels:
    """The model of a ``FORMAT`` file: the pair scorer with its feature names."""
    pair: object
    pair_names: list


@lru_cache(maxsize=2)
def load_models(path: str) -> PairModels:
    from analyzer.phrase.phrase_analyzer import _load_numpy_hgb

    with np.load(path, allow_pickle=False) as z:
        if str(z["format"]) != FORMAT:
            raise ValueError(f"not a {FORMAT} file: {path}")
        return PairModels(_load_numpy_hgb(z, "pair"), [str(x) for x in z["feature_names"]])


def pair_scores(X: np.ndarray, models: PairModels) -> np.ndarray:
    """The pairs' scores, 0..1, from their features."""
    return models.pair.predict_proba(X)[:, 1]


def chorus_spans(t) -> list[tuple[int, int]]:
    """The chorus sections in order, each (first beat, end beat): a run of consecutive phrases of the chorus family
    (CHORUS and BREAK_CHORUS are one)."""
    spans: list = []
    for i, lab in enumerate(t.phrase_label):
        s, e = int(t.phrase_start[i]), int(t.phrase_start[i + 1]) if i + 1 < len(t.phrase_start) else t.n
        if FAMILY.get(lab) != "chorus":
            continue
        if spans and i > 0 and FAMILY.get(t.phrase_label[i - 1]) == "chorus":
            spans[-1] = (spans[-1][0], e)
        else:
            spans.append((s, e))
    return spans


def start_beat(t, rules: JumpCueRules) -> int:
    """No point before this: ``rules.first_sung_beats`` past the first sung phrase's head (verse, chorus, bridge); 0 in
    a track with none sung."""
    sung = [int(s) for s, lab in zip(t.phrase_start, t.phrase_label) if FAMILY.get(lab) in SUNG_FAMILIES]
    return min(sung) + int(rules.first_sung_beats) if sung else 0


def off_cue(t, p: int, q: int, cue_beats: frozenset, start: int) -> tuple[int, int]:
    """The pair, when both its points are CUE points (a jump on the mix start is hard to use live), CUE_MOVE_BEATS
    earlier, its lag kept; as it is when it cannot move there (before ``start``, another place in the bar, a CUE point
    there)."""
    if p not in cue_beats or q not in cue_beats:
        return p, q
    a, b = p - CUE_MOVE_BEATS, q - CUE_MOVE_BEATS
    if (a >= start and t.bar_phase[a] == t.bar_phase[p] and t.bar_phase[b] == t.bar_phase[q]
            and a not in cue_beats and b not in cue_beats):
        return a, b
    return p, q


def choose_pair(t, pairs: np.ndarray, scores: np.ndarray, rules: JumpCueRules = JumpCueRules(),
                cue_beats: frozenset = frozenset()) -> tuple[int, int, float] | None:
    """The automatic JumpCUE: (its earlier beat, its later beat, its score), the best-scored pair ``rules`` allows; None
    when there is none, or its score is under ``rules.min_score``.

    A pair the rules allow: its earlier point in ``rules.earlier``'s sections and its later point in ``rules.later``'s
    (a section reaching ``rules.section_margin_beats`` before its head and past its end; both in choruses: the later
    one in a later chorus section), ``rules.min_gap_beats`` apart or more, none before ``start_beat``; one wholly on CUE
    points offered off them (``off_cue``) when ``rules.avoid_cue_points``, on its score there (a pair that is no
    candidate scores 0). On a tie the earlier landing."""
    spans = chorus_spans(t)
    margin = int(rules.section_margin_beats)
    start = start_beat(t, rules)
    score = {(int(a), int(b)): float(s) for (a, b), s in zip(pairs.tolist(), scores)}

    def sections(x: int, where: str) -> list[int]:
        """The chorus sections ``where`` names that hold beat ``x`` ([-1]: anywhere)."""
        if where == "any":
            return [-1]
        held = [k for k, (s, e) in enumerate(spans) if s - margin <= x < e + margin]
        if where == "first_chorus":
            return [0] if 0 in held else []
        if where == "last_chorus":
            return [len(spans) - 1] if spans and len(spans) - 1 in held else []
        return held

    best, best_s = None, 0.0
    for p, q in score:
        if q - p < rules.min_gap_beats or p < start:
            continue
        a, b = off_cue(t, p, q, cue_beats, start) if rules.avoid_cue_points else (p, q)
        ka, kb = sections(a, rules.earlier), sections(b, rules.later)
        if not ka or not kb:
            continue
        if rules.earlier != "any" and rules.later != "any" and not any(j > i for i in ka for j in kb):
            continue
        s = score.get((a, b), 0.0)
        if s > best_s or (s == best_s and best is not None and b < best[1]):
            best, best_s = (a, b), s
    if best is None or best_s < rules.min_score:
        return None
    return best[0], best[1], best_s
