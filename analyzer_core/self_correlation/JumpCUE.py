from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .self_corr_wrapper import (
    SelfCorrelationReport,
    SelfCorrelationV2Config,
    SimilarLink,
    analyze_self_correlation,
    refine_overlap,
    get_best_jump_offset
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
    min_duration_sec: float = 6.0
    max_pairs: int = 4
    snap_to_beats: bool = True
    beat_snap_tol_sec: float = 0.15
    score_temperature: float = 0.35


@dataclass
class JumpCueResult:
    pairs: List[JumpCuePair] = field(default_factory=list)
    adjacency: Dict[str, List[str]] = field(default_factory=dict)
    report: Optional[SelfCorrelationReport] = None

    def as_dict(self) -> dict:
        return {
            "pairs": [pair.as_dict() for pair in self.pairs],
            "adjacency": {k: list(v) for k, v in self.adjacency.items()},
        }


class JumpCueEngine:
    """
    Build mutually reachable Jump CUE pairs (A<->B, C<->D, …) from self-correlation links.
    """

    def __init__(self, config: Optional[JumpCueConfig] = None) -> None:
        self.config = config or JumpCueConfig()

    def run(
        self,
        y_harm: np.ndarray,
        sr: int,
        beats_time: Sequence[float],
    ) -> JumpCueResult:
        report, xmat = analyze_self_correlation(
            y_harm=y_harm,
            sr=sr,
            beats_time=beats_time,
            config=self.config.self_corr,
        )
        pairs = self._build_pairs(
            report,
            np.asarray(beats_time, dtype=float),
            xmat = xmat
        )
        adjacency = self._build_adjacency(pairs)
        return JumpCueResult(pairs=pairs, adjacency=adjacency, report=report)

    def _build_pairs(
        self,
        report: SelfCorrelationReport,
        beats_time: np.ndarray,
        xmat: np.ndarray
    ) -> List[JumpCuePair]:
        if not report.links:
            return []

        cfg = self.config
        filtered: List[SimilarLink] = []
        jump_time: List[Tuple] = []
        for link in report.links:
            link = refine_overlap(xmat, link, beats_time)
            dur_src = link.a[1] - link.a[0]
            dur_dst = link.b[1] - link.b[0]
            if link.score < cfg.min_score:
                continue
            if dur_src < cfg.min_duration_sec or dur_dst < cfg.min_duration_sec:
                continue
            TjA, TjB = get_best_jump_offset(xmat, link, beats_time)
            filtered.append(link)
            jump_time.append((TjA, TjB))
        if not filtered:
            return []

        confidences = self._softmax_confidence(filtered, cfg.score_temperature)
        pairs: List[JumpCuePair] = []
        for idx, (link, conf, jTime) in enumerate(zip(filtered, confidences, jump_time)):
            label_src, label_dst = self._labels_for_pair(idx)
            forward = JumpCuePoint(label=label_src, start=float(link.a[0]), end=float(link.a[1]), point=jTime[0])
            backward = JumpCuePoint(label=label_dst, start=float(link.b[0]), end=float(link.b[1]), point=jTime[1])
            pair = JumpCuePair(
                forward=forward,
                backward=backward,
                lag_beats=float(link.lag_beats),
                lag_sec=float(link.lag_sec),
                score=float(link.score),
                confidence=float(conf),
            )
            pairs.append(pair)
            if len(pairs) >= cfg.max_pairs:
                break
        return pairs

    @staticmethod
    def _build_adjacency(pairs: Iterable[JumpCuePair]) -> Dict[str, List[str]]:
        adjacency: Dict[str, List[str]] = {}
        for pair in pairs:
            adjacency.setdefault(pair.forward.label, []).append(pair.backward.label)
            adjacency.setdefault(pair.backward.label, []).append(pair.forward.label)
        return adjacency

    @staticmethod
    def _labels_for_pair(pair_index: int) -> Tuple[str, str]:
        base = pair_index * 2
        return JumpCueEngine._label_from_index(base), JumpCueEngine._label_from_index(base + 1)

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
    def _softmax_confidence(links: Sequence[SimilarLink], temperature: float) -> List[float]:
        scores = np.asarray([max(0.0, link.score) for link in links], dtype=float)
        if not scores.size:
            return []
        temp = max(float(temperature), 1e-6)
        norm = scores - scores.max()
        weights = np.exp(norm / temp)
        denom = float(np.sum(weights)) or 1.0
        return [float(w / denom) for w in weights]

    @staticmethod
    def _snap_time(value: float, beats_time: np.ndarray, tol: float) -> float:
        idx = int(np.searchsorted(beats_time, value))
        candidates = []
        if 0 <= idx < beats_time.size:
            candidates.append(beats_time[idx])
        if idx > 0:
            candidates.append(beats_time[idx - 1])
        if idx + 1 < beats_time.size:
            candidates.append(beats_time[idx + 1])
        if not candidates:
            return value
        best = min(candidates, key=lambda x: abs(x - value))
        return float(best) if abs(best - value) <= float(tol) else float(value)

__all__ = [
    "JumpCuePoint",
    "JumpCuePair",
    "JumpCueConfig",
    "JumpCueResult",
    "JumpCueEngine",
]

