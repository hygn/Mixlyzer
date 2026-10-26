from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks

from analyzer.music.beat_geometry import downbeat_beat_indices

from .jump_point_model import JumpContextCache, JumpPointFeatures, JumpPointModel

from .self_corr_wrapper import (
    SelfCorrelationReport,
    SelfCorrelationV2Config,
    analyze_self_correlation,
    refine_overlap,
)
from .vocal_context import (
    JumpContextConfig,
    VocalContext,
    VocalContextConfig,
    compute_vocal_context,
    jump_point_scores,
)

@dataclass
class JumpCuePoint:
    label: str
    start: float
    end: float
    point: float

    def as_dict(self) -> dict:
        return {"label": self.label, "start": float(self.start), "end": float(self.end), "point": float(self.point)}

@dataclass
class JumpCuePair:
    forward: JumpCuePoint
    backward: JumpCuePoint
    lag_beats: float
    lag_sec: float
    score: float
    confidence: float

    def as_dict(self) -> dict:
        return {
            "forward": self.forward.as_dict(),
            "backward": self.backward.as_dict(),
            "lag_beats": float(self.lag_beats),
            "lag_sec": float(self.lag_sec),
            "score": float(self.score),
            "confidence": float(self.confidence),
        }

@dataclass
class JumpCueConfig:
    self_corr: SelfCorrelationV2Config = SelfCorrelationV2Config
    min_score: float = 0.0
    # A link must repeat at least a hypermeasure (GTTM: meter is felt up to
    # about 2-4 bars) to count as a repeated passage and not a lone similar
    # beat; counted in bars so it does not depend on the tempo.
    min_link_bars: int = 4
    # A link seeds a group: every place in the track that repeats the link's
    # segment (up to max_members), all jumping at the same position in it.
    max_groups: int = 4
    max_members: int = 6
    # Every point becomes a rekordbox Hot Cue (A-H on a CDJ): at most this many
    # points in all, strongest group first.
    hot_cue_slots: int = 8
    # Repeat strength of a further member, from the track's median (0) to the
    # segment itself (1).
    member_min_rel: float = 0.5
    # Any two members can be jumped between, so the weakest pair counts: a
    # further member is kept when, at the group's jump beat, every pair still
    # scores at least this share of the link's own pair.
    member_min_score_ratio: float = 0.75
    # No jump point in the first min_point_time_sec of the track, and every two
    # points (of any groups) at least min_point_gap_sec apart. Groups are placed
    # strongest first; a member that cannot meet this is left out.
    min_point_time_sec: float = 30.0
    min_point_gap_sec: float = 10.0
    # Groups whose members overlap in the track (the same passage reached from
    # two links) are merged: the weaker group is absorbed when at least half of
    # its passages join the stronger one (same weakest-pair rule as members).
    merge_min_overlap_beats: int = 8
    # Members start at the same position in the bar (their shift is a whole
    # number of bars), so a jump keeps the metrical position (Plachouras &
    # Miron 2023; the Infinite Jukebox also matches the beat's place in the
    # bar). A repeat found up to bar_align_beats off is moved onto it.
    bar_align_beats: int = 2
    # Jump beat: a break first. A rest is a grouping boundary (GTTM GPR 2):
    # where the music drops out, what was heard and what comes next are cut
    # apart, so a jump there is heard as a reset, and the beats before the jump
    # agreeing is enough (the better of the two sides scores it). A beat is in
    # a break when the mix level lies break_min_db (a clearly audible step)
    # below the median of the break_ref_beats before; the break counts when
    # its depth is among the break_top_fraction most salient drops of the
    # track in every member. When every member is in such a break at once and
    # that scores min_group_score or more, the jump sits on the first beat of
    # the deepest shared break, on any beat of the bar. Otherwise the
    # best-scoring downbeat (the best-scoring beat when there are no bars).
    break_min_db: float = 3.0
    break_top_fraction: float = 0.05
    break_ref_beats: int = 8
    # The jump beat is searched this far before and after the link's segment.
    offset_margin_beats: int = 8
    # Repeats are also searched window by window over the segment, so a passage
    # that repeats only part of it (the full half of a 落ちサビ) is found.
    candidate_window_beats: int = 32
    # A placed group whose weakest pair scores below this is dropped: a jump
    # between passages that share melody and words but not the arrangement
    # (chorus -> 落ちサビ) scores low on the acoustic context around the jump.
    min_group_score: float = 0.4
    snap_to_beats: bool = True
    beat_snap_tol_sec: float = 0.15
    score_temperature: float = 0.35
    # Vocal / lyric context of the jump point (used when the mix is given).
    vocal: VocalContextConfig = field(default_factory=VocalContextConfig)
    jump_context: JumpContextConfig = field(default_factory=JumpContextConfig)
    # Learned jump point (jump_point_model.py): with the per-beat features and
    # the vocal context, it re-chooses the jump beat and the kept members of
    # every placed group; without them the rules above decide.
    jump_point_model_path: str = ""      # its file; none: the rules decide


@dataclass
class RepeatGroup:
    starts: List[int]       # first beat of every member, ascending
    length: int             # beats
    offset: int             # jump beat within the members
    score: float


@dataclass
class JumpCueResult:
    pairs: List[JumpCuePair] = field(default_factory=list)
    adjacency: Dict[str, List[str]] = field(default_factory=dict)
    report: Optional[SelfCorrelationReport] = None
    vocal: Optional[VocalContext] = None

    def as_dict(self) -> dict:
        return {
            "pairs": [pair.as_dict() for pair in self.pairs],
            "adjacency": {k: list(v) for k, v in self.adjacency.items()},
        }


class JumpCueEngine:
    """
    Build groups of mutually reachable Jump CUEs from self-correlation links.

    Each link seeds a group with every repeat of its segment (A<->B<->C when a
    chorus comes three times); a link whose two segments are already members
    of a group adds nothing. All members of a group jump at the same beat of
    the segment, so every pair of them is returned with identical points.
    """

    def __init__(self, config: Optional[JumpCueConfig] = None) -> None:
        self.config = config or JumpCueConfig()
        # Beats a jump point may sit on (bar starts); None = any beat.
        self._point_beats: Optional[np.ndarray] = None
        # Beat position within its bar (beats since the bar's downbeat); None = unknown.
        self._bar_phase: Optional[np.ndarray] = None
        self._beats_per_bar: int = 4
        # Break depth a break must reach (dB), from the track's most salient drops.
        self._break_floor: float = np.inf
        # Breaks and the vocal context of the track (set per run).
        self._vocal: Optional[VocalContext] = None
        self._break_fall: Optional[np.ndarray] = None
        self._C: Optional[np.ndarray] = None
        # (early, late, length) -> (score after the jump, score before it)
        self._pair_cache: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._score_cache: Optional[JumpContextCache] = None
        self._jump_features: Optional[JumpPointFeatures] = None
        self._jump_model: Optional[JumpPointModel] = (
            JumpPointModel.load(self.config.jump_point_model_path) if self.config.jump_point_model_path else None
        )

    def run(
        self,
        y_harm: np.ndarray,
        sr: int,
        beats_time: Sequence[float],
        *,
        y_mix: Optional[np.ndarray] = None,
        mix_magnitude: Optional[np.ndarray] = None,
        tempo_segments: Optional[np.ndarray] = None,
        jump_point_features: Optional[JumpPointFeatures] = None,
        vocal_context: Optional[VocalContext] = None,
    ) -> JumpCueResult:
        """``y_mix`` (or its STFT magnitude with the vocal config's n_fft / hop)
        enables the vocal / lyric context of the jump points. ``tempo_segments``
        gives the downbeats that jump points prefer outside breaks.
        ``jump_point_features`` (with the vocal context) lets the learned model
        choose the jump points."""
        beats = np.asarray(beats_time, dtype=float)
        if vocal_context is not None and not np.array_equal(vocal_context.beats_time, beats):
            raise ValueError("cached vocal context uses a different beat grid")
        self._jump_features = jump_point_features
        self._score_cache = None
        self._pair_cache = {}
        self._vocal = None
        self._break_fall = None
        self._break_floor = np.inf
        self._C = None
        self._point_beats = None
        self._bar_phase = None
        self._beats_per_bar = 4
        segments = None if tempo_segments is None else np.asarray(tempo_segments, dtype=float)
        if segments is not None and segments.ndim == 2 and segments.shape[0] > 0:
            mask = np.zeros(beats.size, dtype=bool)
            mask[downbeat_beat_indices(beats, segments)] = True
            self._point_beats = mask
            downbeats = np.flatnonzero(mask)
            if downbeats.size:
                last = np.searchsorted(downbeats, np.arange(beats.size), side="right") - 1
                self._bar_phase = np.where(last >= 0, np.arange(beats.size) - downbeats[np.maximum(last, 0)], -1)
            if downbeats.size >= 2:
                self._beats_per_bar = int(max(1, round(float(np.median(np.diff(downbeats))))))
        report, xmat = analyze_self_correlation(
            y_harm=y_harm,
            sr=sr,
            beats_time=beats,
            config=self.config.self_corr,
        )
        vocal = vocal_context
        if vocal is None and (y_mix is not None or mix_magnitude is not None) and beats.size >= 2 and report.links:
            vocal = compute_vocal_context(
                y_mix, sr, beats, magnitude=mix_magnitude, config=self.config.vocal
            )
        pairs = self._build_pairs(report, beats, xmat=xmat, vocal=vocal)
        adjacency = self._build_adjacency(pairs)
        return JumpCueResult(pairs=pairs, adjacency=adjacency, report=report, vocal=vocal)

    def _build_pairs(
        self,
        report: SelfCorrelationReport,
        beats_time: np.ndarray,
        xmat: np.ndarray,
        vocal: Optional[VocalContext] = None,
    ) -> List[JumpCuePair]:
        self._score_cache = JumpContextCache(vocal) if vocal is not None else None
        if not report.links or xmat.ndim != 2 or xmat.shape[1] == 0:
            return []

        cfg = self.config
        X = xmat / (np.linalg.norm(xmat, axis=0, keepdims=True) + 1e-9)
        C = X.T @ X  # beat-column cosine
        n = C.shape[0]
        self._vocal = vocal
        self._break_fall = self._level_fall(vocal)
        self._break_floor = np.inf
        if self._break_fall is not None and self._break_fall.size > 4:
            depth = self._score_cache.break_depth(cfg.break_ref_beats)
            self._break_floor = float(np.quantile(depth, 1.0 - float(cfg.break_top_fraction)))
        self._C = C
        self._pair_cache = {}
        groups: List[RepeatGroup] = []
        # Groups are built without the gap to each other: merging moves their
        # points, and the gap is kept when they are placed (_place_groups).
        none = np.zeros(0)
        for link in report.links:
            link = refine_overlap(xmat, link, beats_time)
            if link.score < cfg.min_score:
                continue
            a0 = int(round(link.a_beat[0]))
            b0 = int(round(link.b_beat[0]))
            length = int(round(link.a_beat[1])) - a0
            if length < cfg.min_link_bars * self._beats_per_bar:
                continue
            if length <= 0 or a0 + length > n or b0 < 0 or b0 + length > n:
                continue
            if any(self._covers_link(g, a0, b0, length) for g in groups):
                continue
            group = self._build_group(C, a0, b0, length, vocal, beats_time, none)
            if group is None or group.score < cfg.min_group_score:
                continue
            groups.append(group)
            starts, offset, score = group.starts, group.offset, group.score
            print(
                f"[JumpCUE] group {length} beats, members "
                + ", ".join(f"{float(beats_time[m]):.1f}s" for m in starts)
                + f", jump at +{offset} beats score={score:.3f}"
            )
            if len(groups) >= 2 * cfg.max_groups:
                break
        if not groups:
            return []
        groups = self._merge_groups(C, groups, vocal, beats_time)
        groups = self._place_groups(C, groups, vocal, beats_time)
        if groups and vocal is not None and self._jump_features is not None and self._jump_model is not None:
            groups = self._model_points(groups, beats_time)
        else:
            groups = self._fit_hot_cues(groups)
        if not groups:
            return []

        raw: List[Tuple[JumpCuePoint, JumpCuePoint, int, float, float]] = []
        label_index = 0
        for group in groups:
            points = []
            for m in group.starts:
                points.append(
                    JumpCuePoint(
                        label=self._label_from_index(label_index),
                        start=self._beat_time(beats_time, m),
                        end=self._beat_time(beats_time, m + group.length),
                        point=self._beat_time(beats_time, m + group.offset),
                    )
                )
                label_index += 1
            beats_in = np.arange(group.length)
            for i in range(len(group.starts)):
                for j in range(i + 1, len(group.starts)):
                    early, late = group.starts[i], group.starts[j]
                    pair_score = float(np.mean(C[early + beats_in, late + beats_in]))
                    raw.append((points[j], points[i], late - early, points[j].start - points[i].start, pair_score))

        confidences = self._softmax_confidence([r[4] for r in raw], cfg.score_temperature)
        return [
            JumpCuePair(
                forward=fwd,
                backward=bwd,
                lag_beats=float(lag_beats),
                lag_sec=float(lag_sec),
                score=float(score),
                confidence=float(conf),
            )
            for (fwd, bwd, lag_beats, lag_sec, score), conf in zip(raw, confidences)
        ]

    @staticmethod
    def _covers_link(group: RepeatGroup, a0: int, b0: int, length: int, tol: int = 2) -> bool:
        """``group`` already holds the link: two of its members overlap the
        link's segments by half or more each, with the same shift between
        them as the link (within ``tol`` beats). A link with another shift
        pairs other passages even when its segments lie inside long members."""
        def covering(start: int) -> List[int]:
            out = []
            for m in group.starts:
                overlap = min(m + group.length, start + length) - max(m, start)
                if overlap >= 0.5 * min(length, group.length):
                    out.append(m)
            return out

        for ma in covering(a0):
            for mb in covering(b0):
                if ma != mb and abs((ma - mb) - (a0 - b0)) <= tol:
                    return True
        return False

    def _repeat_candidates(self, C: np.ndarray, a0: int, b0: int, length: int) -> List[int]:
        """Starts of further repeats of segment ``[a0, a0 + length)``, strongest
        first, at least half a segment from each other and from the link's
        segments. Peaks of the mean cosine along the diagonal, over the whole
        segment and over windows of candidate_window_beats (half-overlapping)
        of it, each relative from the track's median (0) to the window itself
        (1); a window peak gives the start of the segment it lies in."""
        cfg = self.config
        n = C.shape[0]
        gap = max(1, length // 2)
        win = int(cfg.candidate_window_beats)
        windows = [(0, length)]
        if length > win:
            windows += [(w, win) for w in range(0, length - win + 1, max(1, win // 2))]

        best: Dict[int, float] = {}
        for w0, wl in windows:
            span = n - wl + 1
            stripe = np.zeros(span)
            for i in range(wl):
                stripe += C[a0 + w0 + i, i : i + span]
            stripe /= float(wl)
            own = a0 + w0
            away = np.abs(np.arange(span) - own) >= max(1, wl // 2)
            base = float(np.median(stripe[away])) if away.any() else 0.0
            rel = (stripe - base) / max(float(stripe[own]) - base, 1e-9)
            peaks, _ = find_peaks(stripe, distance=max(1, wl // 2))
            for j in peaks:
                start = int(j) - w0
                if rel[j] < cfg.member_min_rel or start < 0 or start + length > n:
                    continue
                best[start] = max(best.get(start, -np.inf), float(rel[j]))

        taken = [a0, b0]
        found: List[int] = []
        for start in sorted(best, key=lambda k: best[k], reverse=True):
            if all(abs(start - m) >= gap for m in taken):
                taken.append(start)
                found.append(start)
        return found

    def _pair_curves(
        self, C: np.ndarray, early: int, late: int, length: int, vocal: Optional[VocalContext]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scores of jumping between ``early + d`` and ``late + d`` for every
        ``d`` in the segment, judged on the beats after the jump and on the
        beats before it: the vocal / lyric context scores, or without them the
        cosine of the two beats (after) and of the two beats before (before)."""
        key = (early, late, length)
        if key in self._pair_cache:
            return self._pair_cache[key]
        after = np.full(length, -np.inf)
        before = np.full(length, -np.inf)
        if vocal is not None:
            cache = self._score_cache
            scores = (cache.scores((late, late + length), late - early, self.config.jump_context)
                      if cache is not None and cache.ctx is vocal else
                      jump_point_scores(vocal, (late, late + length), late - early, self.config.jump_context))
            for sc in scores:
                d = sc.beat_a - late
                after[d] = sc.score
                # Same running-line penalty as the score after the jump.
                penalty = max(sc.acoustic_post, 0.0) * sc.lyric_post - sc.score
                before[d] = max(sc.acoustic_pre, 0.0) * sc.lyric_pre - penalty
        if not np.isfinite(after).any():
            d = np.arange(length)
            after = C[early + d, late + d].astype(float)
            prev = np.maximum(d - 1, 0)
            before = C[early + prev, late + prev].astype(float)
        self._pair_cache[key] = (after, before)
        return after, before

    def _pair_curve(
        self, C: np.ndarray, early: int, late: int, length: int, vocal: Optional[VocalContext]
    ) -> np.ndarray:
        """Score of jumping between ``early + d`` and ``late + d`` (the beats
        after the jump agree), for every ``d`` in the segment."""
        return self._pair_curves(C, early, late, length, vocal)[0]

    def _build_group(
        self,
        C: np.ndarray,
        a0: int,
        b0: int,
        length: int,
        vocal: Optional[VocalContext],
        beats_time: np.ndarray,
        taken: np.ndarray,
    ) -> Optional[RepeatGroup]:
        """The link's two segments plus the further repeats that keep every
        pair a good jump, jumping at the beat where the weakest pair is best.
        Jump beats that break the point limits (track start, gap to ``taken``
        and between the members) are not used; None when the link's own two
        segments have no such beat."""
        cfg = self.config
        curves: Dict[Tuple[int, int, int], np.ndarray] = {}

        def weakest(members: List[int]) -> np.ndarray:
            return self._weakest_curve(C, members, length, vocal, beats_time, taken, curves)

        if a0 == b0:
            return None
        n = C.shape[0]
        b0 = self._align_bar(a0, b0, length, n)
        if b0 is None or b0 == a0:
            return None
        candidates = [
            j for j in (self._align_bar(a0, c, length, n) for c in self._repeat_candidates(C, a0, b0, length))
            if j is not None and j not in (a0, b0)
        ]
        # Widen the segment by offset_margin_beats on each side (within the track).
        margin = int(max(0, cfg.offset_margin_beats))
        head = min(margin, a0, b0)
        tail = max(0, min(margin, n - max(a0, b0) - length))
        a0, b0, length = a0 - head, b0 - head, length + head + tail
        candidates = [j - head for j in candidates if j - head >= 0 and j - head + length <= n]

        members = sorted({a0, b0})
        seed = float(np.max(weakest(members)))
        if not np.isfinite(seed):
            return None
        floor = cfg.member_min_score_ratio * seed
        for j in candidates:
            # A link that is no good jump itself takes no further members.
            if len(members) >= cfg.max_members or seed <= 0.0:
                break
            trial = sorted(members + [j])
            if float(np.max(weakest(trial))) >= floor:
                members = trial
        curve = weakest(members)
        offset, score = self._pick_offset(curve, members)
        return RepeatGroup(starts=members, length=length, offset=offset, score=score)

    def _weakest_curve(
        self,
        C: np.ndarray,
        members: List[int],
        length: int,
        vocal: Optional[VocalContext],
        beats_time: np.ndarray,
        taken: np.ndarray,
        curves: Dict[Tuple[int, int, int], np.ndarray],
    ) -> np.ndarray:
        """Per jump beat ``d`` of the segment: the lowest pair score among the
        members, -inf where a point breaks the limits (before the start time,
        or closer than the gap to ``taken`` or to another member's point)."""
        cfg = self.config
        d = np.arange(length)
        gap = float(cfg.min_point_gap_sec)
        out = np.full(length, np.inf)
        times = [beats_time[m + d] for m in members]
        for i, t in enumerate(times):
            out[t < float(cfg.min_point_time_sec)] = -np.inf
            if taken.size:
                out[np.min(np.abs(t[:, None] - taken[None, :]), axis=1) < gap] = -np.inf
            for j in range(i + 1, len(members)):
                out[np.abs(times[j] - t) < gap] = -np.inf
                key = (members[i], members[j], length)
                if key not in curves:
                    curves[key] = self._pair_curve(C, members[i], members[j], length, vocal)
                out = np.minimum(out, curves[key])
        return out

    def _merge_groups(
        self,
        C: np.ndarray,
        groups: List[RepeatGroup],
        vocal: Optional[VocalContext],
        beats_time: np.ndarray,
    ) -> List[RepeatGroup]:
        """Merge groups that reach the same passage.

        Two groups whose members overlap by merge_min_overlap_beats or more are
        laid on one frame through their most overlapping members (overlapping
        members are the same audio, so the shift between the groups is exact),
        and the segment spans both groups' segments (clipped to the track). The
        stronger group's passages stay; the weaker group's passages join one by
        one while the weakest pair keeps member_min_score_ratio of the weaker
        group's score, or of the stronger group's when the weaker scores 0 or
        less (a passage within the point gap of a member is that member). The weaker group is absorbed when at least half of its
        passages are in the merged group; the rest are dropped. The limits
        against other groups are left to :meth:`_place_groups`. Repeats until
        nothing merges.
        """
        cfg = self.config
        n = C.shape[0]
        none = np.zeros(0)
        curves: Dict[Tuple[int, int, int], np.ndarray] = {}
        groups = sorted(groups, key=lambda g: g.score, reverse=True)

        def same_passage(a: int, b: int) -> bool:
            ta = beats_time[min(max(a, 0), beats_time.size - 1)]
            tb = beats_time[min(max(b, 0), beats_time.size - 1)]
            return abs(float(ta - tb)) < float(cfg.min_point_gap_sec)

        def try_merge(g: RepeatGroup, h: RepeatGroup) -> Optional[RepeatGroup]:
            best = None  # (overlap, g member, h member)
            for gm in g.starts:
                for hm in h.starts:
                    ov = min(gm + g.length, hm + h.length) - max(gm, hm)
                    if ov >= cfg.merge_min_overlap_beats and (best is None or ov > best[0]):
                        best = (ov, gm, hm)
            if best is None:
                return None
            shift = best[2] - best[1]  # h member = g member + shift
            lo = max(min(0, shift), -min(g.starts))
            hi = min(max(g.length, shift + h.length), n - max(g.starts))
            length = hi - lo
            if length < cfg.merge_min_overlap_beats:
                return None
            members = [m + lo for m in g.starts]  # frame origins moved to lo
            # A weaker group that is no good jump itself (score <= 0) gives no
            # reason to lower the stronger group: its passages must then keep
            # the stronger group's own level.
            base = h.score if h.score > 0.0 else g.score
            floor = max(cfg.min_group_score, cfg.member_min_score_ratio * min(g.score, base))
            joined = 0
            for hm in h.starts:
                o = hm - shift + lo
                if any(same_passage(o, m) for m in members):
                    joined += 1
                    continue
                if o < 0 or o + length > n or len(members) >= cfg.max_members:
                    continue
                trial = sorted(members + [o])
                if float(np.max(self._weakest_curve(C, trial, length, vocal, beats_time, none, curves))) >= floor:
                    members = trial
                    joined += 1
            if 2 * joined < len(h.starts):
                print(
                    f"[JumpCUE] merge rejected: {joined}/{len(h.starts)} passages of the group at "
                    f"{float(beats_time[h.starts[0]]):.1f}s join the group at {float(beats_time[g.starts[0]]):.1f}s"
                )
                return None
            curve = self._weakest_curve(C, members, length, vocal, beats_time, none, curves)
            offset, score = self._pick_offset(curve, members)
            if not np.isfinite(score):
                return None
            merged = RepeatGroup(starts=members, length=length, offset=offset, score=score)
            print(
                f"[JumpCUE] merged groups ({g.score:.3f}, {h.score:.3f}) -> {length} beats, members "
                + ", ".join(f"{float(beats_time[m]):.1f}s" for m in members)
                + f", jump at +{offset} beats score={merged.score:.3f}"
            )
            return merged

        changed = True
        while changed:
            changed = False
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    merged = try_merge(groups[i], groups[j])
                    if merged is None:
                        continue
                    rest = [x for k, x in enumerate(groups) if k not in (i, j)]
                    groups = sorted(rest + [merged], key=lambda x: x.score, reverse=True)
                    changed = True
                    break
                if changed:
                    break
        return groups

    def _place_groups(
        self,
        C: np.ndarray,
        groups: List[RepeatGroup],
        vocal: Optional[VocalContext],
        beats_time: np.ndarray,
    ) -> List[RepeatGroup]:
        """Final jump beats, strongest group first, under the point limits
        (track start, gap to the points already placed and between members).
        A group with no allowed jump beat, or whose weakest pair there scores
        below min_group_score, is dropped."""
        curves: Dict[Tuple[int, int, int], np.ndarray] = {}
        placed: List[RepeatGroup] = []
        taken = np.zeros(0)
        for g in sorted(groups, key=lambda x: x.score, reverse=True):
            curve = self._weakest_curve(C, g.starts, g.length, vocal, beats_time, taken, curves)
            offset, score = self._pick_offset(curve, g.starts)
            if not np.isfinite(score):
                print(f"[JumpCUE] group at {float(beats_time[g.starts[0]]):.1f}s dropped: no jump beat within the point limits")
                continue
            if score < self.config.min_group_score:
                print(
                    f"[JumpCUE] group at {float(beats_time[g.starts[0]]):.1f}s dropped: "
                    f"score {score:.3f} < {self.config.min_group_score:.3f}"
                )
                continue
            placed.append(RepeatGroup(starts=g.starts, length=g.length, offset=offset, score=score))
            if len(placed) >= self.config.max_groups:
                break
            taken = np.append(taken, beats_time[np.asarray(g.starts) + offset])
        return placed

    def _model_points(self, groups: List[RepeatGroup], beats_time: np.ndarray) -> List[RepeatGroup]:
        """Placed groups with the jump beat and the kept members chosen by the
        learned model (strongest group first), keeping the point limits: from
        min_point_time_sec on, min_point_gap_sec between all points, and no
        more points than hot_cue_slots in all."""
        cfg = self.config
        taken: List[float] = []
        out: List[RepeatGroup] = []

        def allowed(beats: List[int]) -> bool:
            if len(taken) + len(beats) > int(cfg.hot_cue_slots):
                return False
            times = [self._beat_time(beats_time, k) for k in beats]
            if min(times) < float(cfg.min_point_time_sec):
                return False
            gap = float(cfg.min_point_gap_sec)
            if any(abs(a - b) < gap for a, b in itertools.combinations(times, 2)):
                return False
            return all(abs(t - q) >= gap for t in times for q in taken)

        for g in groups:
            points = [m + g.offset for m in g.starts]
            choice = self._jump_model.choose(self._vocal, self._jump_features, points, allowed,
                                             cache=self._score_cache)
            if choice is None:
                print(f"[JumpCUE] group at {float(beats_time[g.starts[0]]):.1f}s dropped: no model jump within the point limits")
                continue
            d, kept = choice
            starts = [g.starts[i] for i in kept]
            out.append(RepeatGroup(starts=starts, length=g.length, offset=g.offset + d, score=g.score))
            taken += [self._beat_time(beats_time, m + g.offset + d) for m in starts]
            print(f"[JumpCUE] model: {len(starts)}/{len(g.starts)} members, jump shifted {d:+d} beats")
        return out

    def _fit_hot_cues(self, groups: List[RepeatGroup]) -> List[RepeatGroup]:
        """Groups (strongest first) whose points fit in hot_cue_slots; without
        the learned model there is no member choice, so a group that does not
        fit whole is left out."""
        out: List[RepeatGroup] = []
        used = 0
        for g in groups:
            if used + len(g.starts) > int(self.config.hot_cue_slots):
                print(f"[JumpCUE] group of {len(g.starts)} dropped: hot cue slots full ({used}/{self.config.hot_cue_slots})")
                continue
            out.append(g)
            used += len(g.starts)
        return out

    def _align_bar(self, ref: int, start: int, length: int, n: int) -> Optional[int]:
        """``start`` moved (by up to bar_align_beats) to the bar position of
        ``ref``; unchanged without bars, None when no such start fits."""
        phase = self._bar_phase
        if phase is None:
            return start
        want = phase[min(ref, phase.size - 1)]
        tol = int(max(0, self.config.bar_align_beats))
        for step in sorted(range(-tol, tol + 1), key=abs):
            j = start + step
            if 0 <= j and j + length <= n and j < phase.size and phase[j] == want:
                return j
        return None

    def _level_fall(self, vocal: Optional[VocalContext]) -> Optional[np.ndarray]:
        """Per beat: how far the mix level lies below the median of the
        break_ref_beats before it (dB)."""
        if vocal is None or vocal.beat_db.size == 0:
            return None
        if self._score_cache is not None and self._score_cache.ctx is vocal:
            return self._score_cache.level_fall(self.config.break_ref_beats)
        level = np.asarray(vocal.beat_db, dtype=float)
        ref = int(max(1, self.config.break_ref_beats))
        fall = np.zeros(level.size)
        for k in range(ref, level.size):
            fall[k] = float(np.median(level[k - ref:k])) - level[k]
        return fall

    def _break_offset(self, C: np.ndarray, curve: np.ndarray, members: List[int]) -> Optional[Tuple[int, float]]:
        """First beat of the deepest break every member is in at once (reaching
        the track's break depth floor in each), scoring min_group_score or more
        on the better side of the jump, with that score; None if none."""
        cfg = self.config
        fall = self._break_fall
        if fall is None:
            return None
        length = curve.size
        idx = np.clip(np.asarray(members, dtype=int)[:, None] + np.arange(length)[None, :], 0, fall.size - 1)
        falls = fall[idx]
        shared = (falls >= float(cfg.break_min_db)).all(axis=0)
        if not shared.any():
            return None
        weakest = np.full(length, np.inf)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                after, before = self._pair_curves(C, members[i], members[j], length, self._vocal)
                weakest = np.minimum(weakest, np.maximum(after, before))
        best = None
        d = 0
        while d < length:
            if not shared[d]:
                d += 1
                continue
            end = d
            while end + 1 < length and shared[end + 1]:
                end += 1
            # A jump point must keep the limits (finite curve) and score.
            depth = float(falls[:, d:end + 1].max(axis=1).min())
            if (
                np.isfinite(curve[d])
                and weakest[d] >= float(cfg.min_group_score)
                and depth >= self._break_floor
            ):
                if best is None or depth > best[0]:
                    best = (depth, d, float(weakest[d]))
            d = end + 1
        return None if best is None else (best[1], best[2])

    def _pick_offset(self, curve: np.ndarray, members: List[int]) -> Tuple[int, float]:
        """Jump beat (see JumpCueConfig.break_min_db): the start of the deepest
        break shared by every member when it scores min_group_score or more;
        otherwise the best-scoring downbeat (any beat without bars, or when no
        downbeat scores min_group_score). Returns the beat and its score."""
        cfg = self.config
        brk = self._break_offset(self._C, curve, members) if self._C is not None else None
        if brk is not None:
            return brk
        candidates = curve
        if self._point_beats is not None:
            idx = np.clip(np.asarray(members, dtype=int)[:, None] + np.arange(curve.size)[None, :], 0, self._point_beats.size - 1)
            on_bar = np.where(self._point_beats[idx].all(axis=0), curve, -np.inf)
            if np.max(on_bar) >= float(cfg.min_group_score):
                candidates = on_bar
        pick = int(np.argmax(candidates))
        return pick, float(curve[pick])

    @staticmethod
    def _beat_time(beats: np.ndarray, idx: int) -> float:
        if idx < beats.size:
            return float(beats[max(0, idx)])
        gap = float(beats[-1] - beats[-2]) if beats.size >= 2 else 0.5
        return float(beats[-1] + gap * (idx - beats.size + 1))

    @staticmethod
    def _build_adjacency(pairs: Iterable[JumpCuePair]) -> Dict[str, List[str]]:
        adjacency: Dict[str, List[str]] = {}
        for pair in pairs:
            adjacency.setdefault(pair.forward.label, []).append(pair.backward.label)
            adjacency.setdefault(pair.backward.label, []).append(pair.forward.label)
        return adjacency

    @staticmethod
    def _label_from_index(idx: int) -> str:
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        n = len(alphabet)
        if idx < n:
            return alphabet[idx]
        # Extend beyond Z by appending numeric suffix (AA0, AB0, …)
        primary = alphabet[idx % n]
        suffix = idx // n
        return f"{primary}{suffix}"

    @staticmethod
    def _softmax_confidence(scores: Sequence[float], temperature: float) -> List[float]:
        scores = np.asarray([max(0.0, float(v)) for v in scores], dtype=float)
        if not scores.size:
            return []
        temp = max(float(temperature), 1e-6)
        norm = scores - scores.max()
        weights = np.exp(norm / temp)
        denom = float(np.sum(weights)) or 1.0
        return [float(w / denom) for w in weights]

__all__ = [
    "RepeatGroup",
    "JumpCuePoint",
    "JumpCuePair",
    "JumpCueConfig",
    "JumpCueResult",
    "JumpCueEngine",
]

