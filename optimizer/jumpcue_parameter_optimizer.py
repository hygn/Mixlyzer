from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.model_selection import GroupKFold

from analyzer.jumpcue.jump_point_model import (
    SEMANTIC_PAIR_SCORES,
    feature_vector,
    semantic_pair_scores,
)
from optimizer import model_start as MS
from utils.atomic_io import atomic_output_path, atomic_write_json, atomic_write_text
from foundation.i18n import tr


PAIR_FEATURES = ("long", "short", "pre", "masking")
MEMBER_FEATURES = (
    "boundary", "mfcc_change", "onset", "melody_onset", "break_depth", "downbeat",
)
PAIR_TERMS = (
    "long_min*short_min", "long_mean*short_mean",
    "long_min*pre_min", "long_mean*pre_mean",
    "short_min*pre_min", "short_mean*pre_mean",
    "long_min*share", "short_min*share",
)
STRUCTURE_TERMS = (
    "boundary_min*downbeat_min", "boundary_mean*downbeat_mean",
    "boundary_min*break_depth_min", "boundary_mean*break_depth_mean",
    "mfcc_change_min*onset_min", "mfcc_change_mean*onset_mean",
    "melody_onset_min*boundary_min", "break_depth_mean*downbeat_mean",
)
NONLINEAR_TERMS = (
    "long_min^2", "short_min^2", "pre_mean^2", "boundary_min^2",
    "boundary_mean^2", "mfcc_change_mean^2", "break_depth_mean^2", "share^2",
)
PERCEPTUAL_CUE_TERMS = (
    "boundary_min*mfcc_change_min", "boundary_mean*mfcc_change_mean",
    "boundary_min*onset_min", "boundary_mean*onset_mean",
    "boundary_mean*melody_onset_mean", "onset_mean*melody_onset_mean",
    "mfcc_change_mean*break_depth_mean", "onset_mean*break_depth_mean",
)
ATTENTION_GATE_TERMS = (
    "long_min*boundary_min", "long_mean*boundary_mean",
    "short_min*boundary_min", "short_mean*boundary_mean",
    "short_min*onset_min", "short_mean*onset_mean",
    "pre_min*boundary_min", "pre_mean*boundary_mean",
)
HARMONY_FEATURES = tuple(
    name for base in SEMANTIC_PAIR_SCORES for name in (f"{base}_min", f"{base}_mean")
)
HARMONY_FLOW_TERMS = (
    "harmony_post_min*long_min", "harmony_post_mean*long_mean",
    "harmony_post_min*short_min", "harmony_post_mean*short_mean",
    "chroma_post_min*long_min", "chroma_post_mean*long_mean",
    "harmony_post_min*boundary_min", "harmony_post_mean*boundary_mean",
)
FEATURE_SET_TERMS = {
    "base": (),
    "pair": PAIR_TERMS,
    "structure": STRUCTURE_TERMS,
    "nonlinear": NONLINEAR_TERMS,
    "pair_structure": PAIR_TERMS + STRUCTURE_TERMS,
    "pair_nonlinear": PAIR_TERMS + NONLINEAR_TERMS,
    "structure_nonlinear": STRUCTURE_TERMS + NONLINEAR_TERMS,
    "combined": PAIR_TERMS + STRUCTURE_TERMS + NONLINEAR_TERMS,
    "perceptual_cues": PAIR_TERMS + STRUCTURE_TERMS + PERCEPTUAL_CUE_TERMS,
    "perceptual_attention": PAIR_TERMS + STRUCTURE_TERMS + ATTENTION_GATE_TERMS,
    "perceptual_full": (
        PAIR_TERMS + STRUCTURE_TERMS + PERCEPTUAL_CUE_TERMS + ATTENTION_GATE_TERMS
    ),
    "harmonic": PAIR_TERMS + STRUCTURE_TERMS + HARMONY_FEATURES,
    "harmonic_flow": PAIR_TERMS + STRUCTURE_TERMS + HARMONY_FEATURES + HARMONY_FLOW_TERMS,
    "harmony_post": PAIR_TERMS + STRUCTURE_TERMS + (
        "harmony_post_min", "harmony_post_mean",
    ),
    "chroma_post": PAIR_TERMS + STRUCTURE_TERMS + (
        "chroma_post_min", "chroma_post_mean",
    ),
}


@dataclass
class GroupRows:
    uid: str
    kind: str
    phase: str
    target: tuple[int, tuple[int, ...]]
    choices: list[tuple[int, tuple[int, ...]]]
    raw_X: np.ndarray
    target_idx: int
    positive_idx: np.ndarray
    short_min: np.ndarray
    is_full: np.ndarray
    # The model pick the user replaced (a corrected group's only true negative).
    rejected_idx: int | None = None


def _load_phrase_harmony(cache_dir: Path, uids: set[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for uid in uids:
        path = cache_dir / f"{uid}.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                matrix = np.asarray(archive["feature_z"], dtype=float)
                feature_names = [str(value) for value in archive["feature_names"]]
        except (OSError, KeyError, ValueError):
            continue
        indices = [i for i, name in enumerate(feature_names) if name.startswith("harmony_")]
        if len(indices) == 18 and matrix.ndim == 2:
            out[uid] = np.ascontiguousarray(matrix[:, indices])
    return out


def _candidate_rows(
    group: dict,
    names: list[str],
    candidate_threshold: float,
    harmony_by_uid: dict[str, np.ndarray] | None = None,
) -> GroupRows:
    choices: list[tuple[int, tuple[int, ...]]] = []
    rows: list[list[float]] = []
    short_mins: list[float] = []
    full_flags: list[bool] = []
    n = len(group["members"])
    harmony = None if harmony_by_uid is None else harmony_by_uid.get(str(group["uid"]))
    required_semantic = tuple(
        base for base in SEMANTIC_PAIR_SCORES if any(base in name for name in names)
    )
    needs_harmony = bool(required_semantic)
    target = (int(group["label"][0]), tuple(int(i) for i in group["label"][1]))
    rejected = group.get("rejected")
    if rejected is not None:
        rejected = (int(rejected[0]), tuple(int(i) for i in rejected[1]))
    for cand in group["cands"]:
        pairs = list(group["pairs"])
        semantic_pair: dict[str, dict[tuple[int, int], float]] = {
            name: {} for name in SEMANTIC_PAIR_SCORES
        }
        if needs_harmony:
            points = np.asarray(group["members"], dtype=int) + int(cand["d"])
            for pair_id in pairs:
                a, b = pair_id
                values = semantic_pair_scores(
                    harmony, int(points[a]), int(points[b]), 16,
                    required=required_semantic,
                )
                for name, value in values.items():
                    semantic_pair[name][tuple(pair_id)] = value
        subsets = [
            S for size in range(n, 1, -1)
            for S in __import__("itertools").combinations(range(n), size)
        ]
        full = tuple(range(n))
        for subset in subsets:
            pair_ids = list(__import__("itertools").combinations(subset, 2))
            short_min = float(min(cand["short"][tuple(p)] for p in pair_ids))
            choice = (int(cand["d"]), tuple(int(i) for i in subset))
            if subset != full and short_min < candidate_threshold and choice not in (target, rejected):
                continue
            f: dict[str, float] = {}
            for name in PAIR_FEATURES:
                values = np.asarray([
                    float(cand.get(name, {}).get(tuple(p), 0.0)) for p in pair_ids
                ])
                f[f"{name}_min"] = float(values.min())
                f[f"{name}_mean"] = float(values.mean())
            for name in SEMANTIC_PAIR_SCORES:
                values = np.asarray([semantic_pair[name].get(tuple(p), 0.0) for p in pair_ids])
                f[f"{name}_min"] = float(values.min())
                f[f"{name}_mean"] = float(values.mean())
            idx = np.asarray(subset, dtype=int)
            for name in MEMBER_FEATURES:
                values = np.asarray(cand["member"][name], dtype=float)[idx]
                f[f"{name}_min"] = float(values.min())
                f[f"{name}_mean"] = float(values.mean())
            f["share"] = len(subset) / n
            rows.append(feature_vector(f, names).tolist())
            choices.append(choice)
            short_mins.append(short_min)
            full_flags.append(subset == full)
    if target not in choices:
        raise ValueError(f"annotated choice is not reachable: {group['uid']} {target}")
    target_idx = choices.index(target)
    return GroupRows(
        uid=str(group["uid"]), kind=str(group["kind"]), phase=str(group.get("phase", "")),
        target=target, choices=choices, raw_X=np.asarray(rows, dtype=float),
        target_idx=target_idx, positive_idx=np.asarray([target_idx], dtype=int),
        short_min=np.asarray(short_mins), is_full=np.asarray(full_flags),
        rejected_idx=choices.index(rejected) if rejected in choices and rejected != target else None,
    )


def _eligible_matrix(group: GroupRows, short_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    eligible = group.is_full | (group.short_min >= short_threshold)
    idx = np.flatnonzero(eligible)
    X = group.raw_X[idx]
    sd = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(sd > 1e-9, sd, 1.0)
    return idx, X


def _assign_positive_sets(groups: list[GroupRows], current: dict, names: list[str],
                          short_threshold: float, max_positive: int = 4) -> None:
    """Treat the annotation as one observed valid answer, not the only answer.

    Alternatives keep the annotated member set and must be at least nearly as
    good on the three cached continuity measures.  The cap reflects that only
    a small number of the many seamless points are useful in practice.
    """
    columns = {name: names.index(name) for name in ("long_min", "short_min", "pre_min")}
    for group, row in zip(groups, current["rows"]):
        eligible, _ = _eligible_matrix(group, short_threshold)
        target_values = group.raw_X[group.target_idx]
        same_members = [
            int(i) for i in eligible
            if group.choices[int(i)][1] == group.target[1]
        ]
        valid = []
        for i in same_members:
            values = group.raw_X[i]
            if (values[columns["long_min"]] >= target_values[columns["long_min"]] - 0.05
                    and values[columns["short_min"]] >= target_values[columns["short_min"]] - 0.03
                    and values[columns["pre_min"]] >= target_values[columns["pre_min"]] - 0.07):
                quality = (values[columns["long_min"]] + values[columns["short_min"]]
                           + 0.35 * values[columns["pre_min"]])
                valid.append((float(quality), i))
        valid.sort(reverse=True)
        positives = {group.target_idx}
        positives.update(i for _, i in valid[:max_positive])
        if group.kind == "corrected" and row["pred_idx"] != group.target_idx:
            positives.discard(int(row["pred_idx"]))
        group.positive_idx = np.asarray(sorted(positives), dtype=int)


def _copy_positive_sets(source: list[GroupRows], target: list[GroupRows]) -> None:
    if len(source) != len(target):
        raise ValueError("JumpCUE group count changed while expanding features")
    for old, new in zip(source, target):
        if old.choices != new.choices:
            raise ValueError(f"JumpCUE candidates changed while expanding features: {old.uid}")
        new.positive_idx = old.positive_idx.copy()


def _predict(group: GroupRows, weights: np.ndarray, short_threshold: float) -> int:
    idx, X = _eligible_matrix(group, short_threshold)
    return int(idx[np.argmax(X @ weights)])


def _metrics(groups: list[GroupRows], weights: np.ndarray, short_threshold: float) -> dict:
    exact = acceptable = shift = within1 = target_top3 = 0
    abs_error: list[int] = []
    member_f1: list[float] = []
    by_kind: dict[str, list[bool]] = defaultdict(list)
    acceptable_by_kind: dict[str, list[bool]] = defaultdict(list)
    by_track: dict[str, list[bool]] = defaultdict(list)
    acceptable_by_track: dict[str, list[bool]] = defaultdict(list)
    rows = []
    for g in groups:
        pi = _predict(g, weights, short_threshold)
        pred = g.choices[pi]
        ok = pred == g.target
        acceptable_ok = pi in set(g.positive_idx.tolist())
        ds = abs(pred[0] - g.target[0])
        p, t = set(pred[1]), set(g.target[1])
        f1 = 2 * len(p & t) / max(1, len(p) + len(t))
        exact += ok
        acceptable += acceptable_ok
        shift += pred[0] == g.target[0]
        within1 += ds <= 1
        abs_error.append(ds)
        member_f1.append(f1)
        by_kind[g.kind].append(ok)
        acceptable_by_kind[g.kind].append(acceptable_ok)
        by_track[g.uid].append(ok)
        acceptable_by_track[g.uid].append(acceptable_ok)
        idx, X = _eligible_matrix(g, short_threshold)
        scores = X @ weights
        pred_local = int(np.flatnonzero(idx == pi)[0])
        order = np.argsort(scores)
        top_global = idx[np.argsort(-scores, kind="stable")[:3]]
        target_top3 += bool(np.any(top_global == g.target_idx))
        margin = float(scores[pred_local] - scores[order[-2]]) if len(order) > 1 else 0.0
        rows.append({
            "uid": g.uid, "kind": g.kind, "phase": g.phase, "correct": bool(ok),
            "acceptable": bool(acceptable_ok), "positive_count": int(len(g.positive_idx)),
            "target_shift": g.target[0], "pred_shift": pred[0], "shift_error": pred[0] - g.target[0],
            "target_members": list(g.target[1]), "pred_members": list(pred[1]),
            "member_f1": f1, "margin": margin, "target_idx": g.target_idx, "pred_idx": pi,
            "target_reachable": bool(np.any(idx == g.target_idx)),
        })
    n = max(1, len(groups))
    return {
        "groups": len(groups), "tracks": len(by_track), "exact_accuracy": exact / n,
        "acceptable_accuracy": acceptable / n,
        "observed_target_top3": target_top3 / n,
        "mean_positive_count": float(np.mean([len(g.positive_idx) for g in groups])),
        "short_threshold": float(short_threshold),
        "shift_accuracy": shift / n, "within_1_beat": within1 / n,
        "mean_abs_shift_error": float(np.mean(abs_error)), "member_f1": float(np.mean(member_f1)),
        "track_all_groups_accuracy": float(np.mean([all(v) for v in by_track.values()])),
        "acceptable_track_accuracy": float(np.mean([all(v) for v in acceptable_by_track.values()])),
        "by_kind": {k: {"groups": len(v), "accuracy": float(np.mean(v)),
                        "acceptable_accuracy": float(np.mean(acceptable_by_kind[k]))}
                    for k, v in by_kind.items()},
        "rows": rows,
    }


def _fit(groups: list[GroupRows], base_weights: np.ndarray, hard_uids: set[str],
         *, l2: float, hard_weight: float, corrected_weight: float = 3.0,
         short_threshold: float = -1.0, preference_weight: float = 0.0) -> np.ndarray:
    """Set likelihood of the saved choices; with ``preference_weight``, also a
    logistic preference of the saved choice over the rejected model pick in
    corrected groups (the only known wrong answer; other candidates stay
    unlabeled)."""
    prepared = []
    for g in groups:
        idx, X = _eligible_matrix(g, short_threshold)
        positives = np.flatnonzero(np.isin(idx, g.positive_idx))
        if not len(positives):
            continue
        pair = None
        if preference_weight and g.rejected_idx is not None:
            t = np.flatnonzero(idx == g.target_idx)
            r = np.flatnonzero(idx == g.rejected_idx)
            if t.size and r.size:
                pair = X[int(t[0])] - X[int(r[0])]
        prepared.append((g, X, positives, pair))

    def objective(w):
        loss = 0.5 * l2 * float(w @ w)
        grad = l2 * w
        for g, X, positives, pair in prepared:
            scale = (corrected_weight if g.kind == "corrected" else 1.0) * (hard_weight if g.uid in hard_uids else 1.0)
            scores = X @ w
            prob = np.exp(scores - logsumexp(scores))
            positive_scores = scores[positives]
            positive_prob = np.exp(positive_scores - logsumexp(positive_scores))
            loss += scale * (float(logsumexp(scores)) - float(logsumexp(positive_scores)))
            grad += scale * (prob @ X - positive_prob @ X[positives])
            if pair is not None:
                margin = float(pair @ w)
                loss += preference_weight * float(np.logaddexp(0.0, -margin))
                grad -= preference_weight * float(expit(-margin)) * pair
        return loss, grad
    result = minimize(objective, base_weights, method="L-BFGS-B", jac=True,
                      options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-7})
    if not result.success:
        print(f"[JumpCUE train] optimizer warning: {result.message}")
    return np.asarray(result.x, dtype=float)


def _oof(groups: list[GroupRows], base_weights: np.ndarray, hard_uids: set[str],
         l2: float, hard_weight: float, short_threshold: float,
         corrected_weight: float = 3.0, folds: int = 5,
         preference_weight: float = 0.0) -> tuple[dict, np.ndarray]:
    uids = np.asarray([g.uid for g in groups])
    predictions = np.empty(len(groups), dtype=int)
    target_top3 = np.zeros(len(groups), dtype=bool)
    splitter = GroupKFold(n_splits=min(folds, len(set(uids))))
    for train, valid in splitter.split(np.arange(len(groups)), groups=uids):
        w = _fit([groups[i] for i in train], base_weights, hard_uids,
                 l2=l2, hard_weight=hard_weight, corrected_weight=corrected_weight,
                 short_threshold=short_threshold, preference_weight=preference_weight)
        for i in valid:
            idx, X = _eligible_matrix(groups[i], short_threshold)
            scores = X @ w
            order = np.argsort(-scores, kind="stable")
            predictions[i] = int(idx[order[0]])
            target_top3[i] = bool(np.any(idx[order[:3]] == groups[i].target_idx))
    # Evaluate without mutating the group data.
    exact = []; acceptable = []; shift = []; within = []; ae = []; mf1 = []; track = defaultdict(list)
    acceptable_track = defaultdict(list)
    by_kind: dict[str, list[bool]] = defaultdict(list)
    acceptable_by_kind: dict[str, list[bool]] = defaultdict(list)
    for g, pi in zip(groups, predictions):
        p = g.choices[int(pi)]; t = g.target
        ok = p == t; exact.append(ok); shift.append(p[0] == t[0]); within.append(abs(p[0]-t[0]) <= 1)
        acceptable_ok = int(pi) in set(g.positive_idx.tolist())
        acceptable.append(acceptable_ok)
        ae.append(abs(p[0]-t[0])); ps, ts = set(p[1]), set(t[1]); mf1.append(2*len(ps&ts)/max(1,len(ps)+len(ts)))
        track[g.uid].append(ok)
        acceptable_track[g.uid].append(acceptable_ok)
        by_kind[g.kind].append(ok)
        acceptable_by_kind[g.kind].append(acceptable_ok)
    return ({"exact_accuracy": float(np.mean(exact)), "shift_accuracy": float(np.mean(shift)),
             "acceptable_accuracy": float(np.mean(acceptable)),
             "observed_target_top3": float(np.mean(target_top3)),
             "mean_positive_count": float(np.mean([len(g.positive_idx) for g in groups])),
             "within_1_beat": float(np.mean(within)), "mean_abs_shift_error": float(np.mean(ae)),
             "member_f1": float(np.mean(mf1)),
             "track_all_groups_accuracy": float(np.mean([all(v) for v in track.values()])),
             "acceptable_track_accuracy": float(np.mean([all(v) for v in acceptable_track.values()])),
             "by_kind": {k: {"groups": len(v), "accuracy": float(np.mean(v)),
                              "acceptable_accuracy": float(np.mean(acceptable_by_kind[k]))}
                         for k, v in by_kind.items()}}, predictions)


def _weaknesses(groups: list[GroupRows], current: dict, names: list[str]) -> dict:
    failures = [(g, r) for g, r in zip(groups, current["rows"]) if not r.get("acceptable", r["correct"])]
    patterns = Counter()
    deltas = []
    for g, row in failures:
        pred = g.choices[row["pred_idx"]]
        if pred[0] != g.target[0] and pred[1] != g.target[1]: patterns["shift_and_members"] += 1
        elif pred[0] != g.target[0]: patterns["shift_only"] += 1
        else: patterns["members_only"] += 1
        if abs(row["shift_error"]) % 4 == 0: patterns["bar_multiple_shift"] += 1
        if abs(g.target[0]) >= 24: patterns["target_near_search_edge"] += 1
        idx, X = _eligible_matrix(g, float(current["short_threshold"]))
        ti = np.flatnonzero(idx == g.target_idx)
        pi = np.flatnonzero(idx == row["pred_idx"])
        if len(ti) and len(pi):
            deltas.append(X[int(ti[0])] - X[int(pi[0])])
    mean_delta = np.mean(deltas, axis=0) if deltas else np.zeros(len(names))
    feature_gaps = sorted(
        ({"feature": n, "target_minus_wrong": float(v)} for n, v in zip(names, mean_delta)),
        key=lambda x: abs(x["target_minus_wrong"]), reverse=True,
    )
    return {
        "failures": len(failures),
        "structurally_unreachable": sum(not row["target_reachable"] for _, row in failures),
        "patterns": dict(patterns),
        "largest_feature_gaps": feature_gaps[:10],
    }


def _write_markdown_report(report: dict, path: Path) -> None:
    current = report["current_model"]
    best = report["best_cv"]
    fitted = report["full_fit"]
    weak = report["weaknesses"]
    lines = [
        "# JumpCUE model audit and retraining",
        "",
        "## Result",
        "",
        f"- Annotated set: {current['groups']} groups across {current['tracks']} tracks.",
        f"- Current exact accuracy: {current['exact_accuracy']:.2%}; track-level all-group accuracy: {current['track_all_groups_accuracy']:.2%}.",
        f"- Current acceptable-set accuracy: {current.get('acceptable_accuracy', current['exact_accuracy']):.2%}; "
        f"acceptable track accuracy: {current.get('acceptable_track_accuracy', current['track_all_groups_accuracy']):.2%}.",
        f"- Best 5-fold track-split CV: {best['exact_accuracy']:.2%} exact, {best['shift_accuracy']:.2%} shift, {best['member_f1']:.3f} member F1.",
        f"- Best-CV acceptable-set accuracy: {best.get('acceptable_accuracy', best['exact_accuracy']):.2%}; "
        f"observed-label top-3: {best.get('observed_target_top3', 0.0):.2%}.",
        *( [f"- Best-CV accuracy by label: accepted {best['by_kind']['accepted']['accuracy']:.2%}; "
             f"corrected {best['by_kind']['corrected']['accuracy']:.2%}."] if "by_kind" in best else [] ),
        *( [f"- Best-CV acceptable accuracy by label: accepted "
             f"{best['by_kind']['accepted'].get('acceptable_accuracy', best['by_kind']['accepted']['accuracy']):.2%}; "
             f"corrected {best['by_kind']['corrected'].get('acceptable_accuracy', best['by_kind']['corrected']['accuracy']):.2%}."]
           if "by_kind" in best else [] ),
        f"- Full-data fit: {fitted['exact_accuracy']:.2%} exact; this is training-set fit, not the generalization estimate.",
        f"- Selected parameters: feature_set={best.get('feature_set', 'base')}, "
        f"features={best.get('feature_count', len(best.get('features', [])))}, "
        f"l2={best['l2']:g}, hard_weight={best['hard_weight']:g}, "
        f"corrected_weight={best.get('corrected_weight', 3.0):g}.",
        *( ["- Extra weighting of current errors was tested and rejected by track-split validation."]
           if any(float(t.get("hard_weight", 1.0)) > 1.0 for t in report.get("cv_trials", [])) else [] ),
        "",
        "The reviewed cache predates the masking feature. The audit and fit use the 19 cached features; "
        "the existing masking_min and masking_mean heuristic weights are retained unchanged.",
        f"Labels use set likelihood with at most four continuity-equivalent answers per group "
        f"(mean {current.get('mean_positive_count', 1.0):.2f}); one observed annotation is not treated as the only valid answer.",
        "",
        "## Current weaknesses",
        "",
        f"- {weak['failures']} of {current['groups']} groups fall outside the acceptable set, spanning {report['hard_track_count']} tracks.",
        f"- {weak['structurally_unreachable']} annotated targets are removed by short_threshold before scoring and cannot be fixed by weight fitting.",
        f"- Accepted accuracy is {current['by_kind']['accepted']['accuracy']:.2%}; corrected accuracy is {current['by_kind']['corrected']['accuracy']:.2%}.",
        f"- Accepted acceptable accuracy is {current['by_kind']['accepted'].get('acceptable_accuracy', current['by_kind']['accepted']['accuracy']):.2%}; "
        f"corrected acceptable accuracy is {current['by_kind']['corrected'].get('acceptable_accuracy', current['by_kind']['corrected']['accuracy']):.2%}.",
        f"- Error types: {weak['patterns'].get('shift_only', 0)} shift-only, {weak['patterns'].get('members_only', 0)} members-only, "
        f"{weak['patterns'].get('shift_and_members', 0)} both.",
        "- The largest target-vs-wrong gaps are boundary, MFCC-change and break-depth features. "
        "The old model often preferred a stronger phrase boundary even when the annotation chose a smoother acoustic continuation.",
        "",
        "## Wrong tracks under the previous model",
        "",
        "| Track | UID | Wrong groups | Details |",
        "|---|---|---:|---|",
    ]
    for item in report["hard_tracks"]:
        title = (item["title"] or "(untitled)").replace("|", "\\|").replace("\n", " ")
        details = []
        for error in item["errors"]:
            details.append(
                f"shift {error['pred_shift']:+d}->{error['target_shift']:+d}; "
                f"members {error['pred_members']}->{error['target_members']}"
            )
        lines.append(f"| {title} | `{item['uid']}` | {len(item['errors'])} | {'; '.join(details)} |")
    atomic_write_text(path, "\n".join(lines) + "\n")


def _write_audit_markdown(audit: dict, path: Path, title: str = "JumpCUE model audit") -> None:
    metrics = audit["current_model"]
    weak = audit["weaknesses"]
    lines = [
        f"# {title}", "",
        f"- Exact accuracy: {metrics['exact_accuracy']:.2%} ({metrics['groups']} groups / {metrics['tracks']} tracks).",
        f"- Acceptable-set accuracy: {metrics.get('acceptable_accuracy', metrics['exact_accuracy']):.2%}; "
        f"acceptable track accuracy: {metrics.get('acceptable_track_accuracy', metrics['track_all_groups_accuracy']):.2%}.",
        f"- Shift accuracy: {metrics['shift_accuracy']:.2%}; member F1: {metrics['member_f1']:.3f}.",
        f"- Fully correct tracks: {metrics['track_all_groups_accuracy']:.2%}.",
        f"- Remaining failures: {weak['failures']} groups across {audit['hard_track_count']} tracks.",
        f"- Structurally unreachable targets: {weak['structurally_unreachable']}.",
        f"- Accepted accuracy: {metrics['by_kind']['accepted']['accuracy']:.2%}; "
        f"corrected accuracy: {metrics['by_kind']['corrected']['accuracy']:.2%}.",
        f"- Accepted acceptable accuracy: {metrics['by_kind']['accepted'].get('acceptable_accuracy', metrics['by_kind']['accepted']['accuracy']):.2%}; "
        f"corrected acceptable accuracy: {metrics['by_kind']['corrected'].get('acceptable_accuracy', metrics['by_kind']['corrected']['accuracy']):.2%}.",
        "", "| Track | UID | Wrong groups |", "|---|---|---:|",
    ]
    for item in audit["hard_tracks"]:
        title_text = (item["title"] or "(untitled)").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {title_text} | `{item['uid']}` | {len(item['errors'])} |")
    atomic_write_text(path, "\n".join(lines) + "\n")


def _cache_supported(term: str) -> bool:
    return "masking_" not in term


def _feature_set_names(base_names: list[str], feature_set: str) -> list[str]:
    return list(dict.fromkeys([*base_names, *FEATURE_SET_TERMS[feature_set]]))


def run(dataset: Path, model_path: Path, output: Path, *,
        stage: str = "all", state_dir: Path = Path("featurecache/jumpcue/training_state"),
        phrase_cache_dir: Path = Path("featurecache/phrase"),
        trial_l2: float = 1.0, trial_hard_weight: float = 1.0,
        trial_feature_set: str = "base", trial_corrected_weight: float = 1.0,
        trial_preference_weight: float = 3.0,
        label_model_path: Path | None = None, titles: dict[str, str] | None = None) -> dict:
    """``titles``: uid -> title for the report (the library is not opened here: without them, uids only)."""
    raw = load_cache(dataset)
    raw_groups = list(raw["groups"] if isinstance(raw, dict) else raw)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    # The reviewed cache predates the masking extension. Constant-zero masking
    # columns standardize to zero, so current behavior over the recorded feature set is exact.
    weight_by_name = dict(zip(model["features"], model["weights"]))
    names = [n for n in model["features"] if _cache_supported(n)]
    base_names = [n for n in names if "*" not in n and not n.endswith("^2")]
    current_weights = np.asarray([weight_by_name[n] for n in names], dtype=float)
    threshold = float(model["short_threshold"])
    harmony_by_uid = _load_phrase_harmony(
        phrase_cache_dir, {str(group["uid"]) for group in raw_groups}
    )
    groups = [_candidate_rows(g, names, threshold, harmony_by_uid) for g in raw_groups]
    current = _metrics(groups, current_weights, threshold)
    if label_model_path is None:
        _assign_positive_sets(groups, current, names, threshold)
    else:
        label_model = json.loads(label_model_path.read_text(encoding="utf-8"))
        label_weight_by_name = dict(zip(label_model["features"], label_model["weights"]))
        label_names = [n for n in label_model["features"] if _cache_supported(n)]
        label_threshold = float(label_model["short_threshold"])
        label_groups = [
            _candidate_rows(g, label_names, label_threshold, harmony_by_uid) for g in raw_groups
        ]
        label_weights = np.asarray([label_weight_by_name[n] for n in label_names], dtype=float)
        label_current = _metrics(label_groups, label_weights, label_threshold)
        _assign_positive_sets(label_groups, label_current, label_names, label_threshold)
        _copy_positive_sets(label_groups, groups)
    current = _metrics(groups, current_weights, threshold)
    hard_uids = {r["uid"] for r in current["rows"] if not r["acceptable"]}

    titles = titles or {}
    wrong_tracks = []
    for uid in sorted(hard_uids):
        rs = [r for r in current["rows"] if r["uid"] == uid and not r["acceptable"]]
        wrong_tracks.append({"uid": uid, "title": titles.get(uid, ""), "errors": rs})

    audit = {
        "dataset": str(dataset),
        "label_model": str(label_model_path or model_path),
        "evaluated_features": names,
        "unevaluated_features": [n for n in model["features"] if n not in names],
        "current_model": current | {"rows": None},
        "hard_tracks": wrong_tracks,
        "hard_track_count": len(hard_uids),
        "weaknesses": _weaknesses(groups, current, names),
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    if stage == "audit":
        atomic_write_json(state_dir / "audit.json", audit)
        _write_audit_markdown(audit, state_dir / "audit.md")
        return audit

    if stage == "trial":
        train_names = _feature_set_names(base_names, trial_feature_set)
        train_groups = (groups if train_names == names else
                        [_candidate_rows(g, train_names, threshold, harmony_by_uid) for g in raw_groups])
        if train_groups is not groups:
            _copy_positive_sets(groups, train_groups)
        initial = np.asarray([weight_by_name.get(n, 0.0) for n in train_names], dtype=float)
        score, _ = _oof(train_groups, initial, hard_uids, trial_l2, trial_hard_weight,
                        threshold, corrected_weight=trial_corrected_weight,
                            preference_weight=trial_preference_weight)
        trial = {"feature_set": trial_feature_set, "features": train_names,
                 "feature_count": len(train_names), "l2": trial_l2,
                 "hard_weight": trial_hard_weight,
                 "corrected_weight": trial_corrected_weight,
                 "preference_weight": trial_preference_weight, **score}
        key = (f"trial_{trial_feature_set}_l2_{trial_l2:g}_hard_{trial_hard_weight:g}"
               f"_corrected_{trial_corrected_weight:g}.json")
        atomic_write_json(state_dir / key, trial)
        return {"trial": trial}

    if stage == "sweep":
        train_names = _feature_set_names(base_names, trial_feature_set)
        train_groups = (groups if train_names == names else
                        [_candidate_rows(g, train_names, threshold, harmony_by_uid) for g in raw_groups])
        if train_groups is not groups:
            _copy_positive_sets(groups, train_groups)
        initial = np.asarray([weight_by_name.get(n, 0.0) for n in train_names], dtype=float)
        swept = []
        for l2 in (3.0, 10.0, 30.0):
            score, _ = _oof(train_groups, initial, hard_uids, l2, trial_hard_weight,
                            threshold, corrected_weight=trial_corrected_weight,
                            preference_weight=trial_preference_weight)
            trial = {"feature_set": trial_feature_set, "features": train_names,
                     "feature_count": len(train_names), "l2": l2,
                     "hard_weight": trial_hard_weight,
                     "corrected_weight": trial_corrected_weight,
                 "preference_weight": trial_preference_weight, **score}
            key = (f"trial_{trial_feature_set}_l2_{l2:g}_hard_{trial_hard_weight:g}"
                   f"_corrected_{trial_corrected_weight:g}.json")
            atomic_write_json(state_dir / key, trial)
            swept.append(trial)
        return {"trials": swept}

    if stage == "all":
        trials = []
        for feature_set in FEATURE_SET_TERMS:
            train_names = _feature_set_names(base_names, feature_set)
            train_groups = (groups if train_names == names else
                            [_candidate_rows(g, train_names, threshold, harmony_by_uid) for g in raw_groups])
            if train_groups is not groups:
                _copy_positive_sets(groups, train_groups)
            initial = np.asarray([weight_by_name.get(n, 0.0) for n in train_names], dtype=float)
            for l2 in (1.0, 3.0, 10.0):
                score, _ = _oof(train_groups, initial, hard_uids, l2, 1.0, threshold,
                                corrected_weight=trial_corrected_weight,
                                preference_weight=trial_preference_weight)
                trials.append({"feature_set": feature_set, "features": train_names,
                               "feature_count": len(train_names), "l2": l2,
                               "hard_weight": 1.0,
                               "corrected_weight": trial_corrected_weight,
                 "preference_weight": trial_preference_weight, **score})
    else:
        trials = [json.loads(path.read_text(encoding="utf-8"))
                  for path in sorted(state_dir.glob("trial_*.json"))]
        if not trials:
            raise RuntimeError(f"no completed trials in {state_dir}")

    best = max(trials, key=lambda x: (
        x.get("acceptable_accuracy", x["exact_accuracy"]),
        x.get("acceptable_track_accuracy", x["track_all_groups_accuracy"]),
        x.get("observed_target_top3", 0.0),
        x["member_f1"],
    ))
    train_names = list(best.get("features", base_names))
    train_groups = (groups if train_names == names else
                    [_candidate_rows(g, train_names, threshold, harmony_by_uid) for g in raw_groups])
    if train_groups is not groups:
        _copy_positive_sets(groups, train_groups)
    initial = np.asarray([weight_by_name.get(n, 0.0) for n in train_names], dtype=float)
    final_weights = _fit(train_groups, initial, hard_uids,
                         l2=best["l2"], hard_weight=best["hard_weight"],
                         corrected_weight=best.get("corrected_weight", 3.0),
                         preference_weight=best.get("preference_weight", 0.0),
                         short_threshold=threshold)
    fitted = _metrics(train_groups, final_weights, threshold)

    output_model = dict(model)
    by_name = dict(weight_by_name)
    by_name.update(dict(zip(train_names, final_weights)))
    retained = [n for n in model["features"] if not _cache_supported(n)]
    output_model["features"] = [*train_names, *retained]
    output_model["weights"] = [float(by_name[n]) for n in output_model["features"]]
    output_model["short_threshold"] = threshold
    output_model["feature_terms"] = {
        "feature_set": best.get("feature_set", "base"),
        "syntax": "base, a*b interaction, or a^2 square; evaluated before per-group standardization",
    }
    output_model["label_model"] = {
        "objective": "set likelihood + pairwise preference",
        "max_positive_candidates": 4,
        "policy": "annotated target plus same-member continuity-equivalent candidates; rejected corrected "
                  "prediction excluded; a known rejected pick is preferred against",
    }
    output_model["trained_on"] = {
        "groups": len(groups), "tracks": len({g.uid for g in groups}),
        "reachable_groups": sum(r["target_reachable"] for r in current["rows"]),
        "corrected": sum(g.kind == "corrected" for g in groups),
        "accepted": sum(g.kind == "accepted" for g in groups),
        "hard_tracks": len(hard_uids), "hard_weight": best["hard_weight"], "l2": best["l2"],
        "corrected_weight": best.get("corrected_weight", 3.0),
        "preference_weight": best.get("preference_weight", 0.0),
        "preference_pairs": sum(g.rejected_idx is not None for g in groups),
        "feature_set": best.get("feature_set", "base"), "feature_count": len(train_names),
        "mean_positive_candidates": current.get("mean_positive_count", 1.0),
        "validation": "5-fold GroupKFold by track",
    }
    report = {
        **audit,
        "cv_trials": trials,
        "best_cv": best, "full_fit": fitted | {"rows": None}, "model": output_model,
    }
    atomic_write_json(output, report)
    _write_markdown_report(report, output.with_suffix(".md"))
    atomic_write_json(model_path, output_model)
    return report


# --------------------------------------------------------------------------
# Library optimizer (settings dialog): trains on the user's own library.
#
# A track contributes its saved JumpCUEs ('jump_cues_np') and, when kept, the
# automatic analysis result ('jump_cues_auto_np'). The engine runs on the
# track's current beat grid and every placed group's candidates (shift x
# member subset) are cached. A saved group is the group's label; where the
# automatic result chose differently for the same group, that choice is the
# one known wrong answer (pairwise preference). Every other candidate stays
# unlabeled: a library holds only the jump points its owner uses.
# --------------------------------------------------------------------------

_CACHE_VERSION = 1
_SNAP_TOLERANCE_SEC = 0.1


# The candidate caches as NPZ (never pickle: a cache folder may come from elsewhere): the arrays as their own
# entries, the rest (dicts, with tuple keys too, lists, tuples, numbers, strings) as JSON with references to them.
def _pack(obj, arrays: list):
    if isinstance(obj, np.ndarray):
        arrays.append(obj)
        return {"__a": len(arrays) - 1}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {"__d": [[_pack(k, arrays), _pack(v, arrays)] for k, v in obj.items()]}
    if isinstance(obj, tuple):
        return {"__t": [_pack(v, arrays) for v in obj]}
    if isinstance(obj, list):
        return [_pack(v, arrays) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"not cached: {type(obj).__name__}")


def _unpack(node, archive):
    if isinstance(node, list):
        return [_unpack(v, archive) for v in node]
    if isinstance(node, dict):
        if "__a" in node:
            return np.asarray(archive[f"a{int(node['__a'])}"])
        if "__t" in node:
            return tuple(_unpack(v, archive) for v in node["__t"])
        return {_unpack(k, archive): _unpack(v, archive) for k, v in node["__d"]}
    return node


def save_cache(path: Path, obj) -> None:
    arrays: list = []
    tree = json.dumps(_pack(obj, arrays))
    with atomic_output_path(path) as temporary:
        with temporary.open("wb") as fh:
            np.savez(fh, tree=np.asarray(tree), **{f"a{i}": a for i, a in enumerate(arrays)})


def load_cache(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return _unpack(json.loads(str(archive["tree"])), archive)
_MEMBER_KEYS = ("boundary", "mfcc_change", "onset", "melody_onset", "break_depth", "downbeat")


@dataclass(frozen=True)
class JumpCueOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    phrase_parameter_path: Path
    l2: float = 10.0
    corrected_weight: float = 1.0
    preference_weight: float = 3.0
    # >= 2 adds a track-level cross-validation before the final fit.
    cv_folds: int = 5
    rebuild_cache: bool = False
    max_cache_workers: int | None = None


@dataclass(frozen=True)
class JumpCueOptimizationResult:
    output_path: Path
    track_count: int
    group_count: int
    preference_pairs: int
    label_auc: float
    pick_downbeat: float
    user_above_automatic: float
    cross_validated: bool
    skipped_tracks: tuple = ()


def _analysis_settings(phrase_parameter_path: Path) -> dict:
    from core.config import ANALYSIS_SAMPLE_RATE, default_cfg
    from foundation.resource_paths import project_root

    values = dict(default_cfg().to_dict()["analysisconfig"])
    try:
        loaded = json.loads((project_root() / "config.json").read_text(encoding="utf-8"))
        values.update(loaded.get("analysisconfig", {}))
    except (OSError, ValueError):
        pass
    phrase = Path(phrase_parameter_path)
    return {
        "sr": ANALYSIS_SAMPLE_RATE,
        "use_hpss": True,                  # always on (part of the cache key)
        "bpm_hop_length": int(values.get("bpm_hop_length", 256)),
        "phrase_parameter_path": str(phrase),
        "phrase_mtime": phrase.stat().st_mtime if phrase.exists() else 0.0,
    }


def _cache_key(beats: np.ndarray, segments: np.ndarray, settings: dict) -> str:
    import hashlib

    h = hashlib.sha1()
    h.update(np.ascontiguousarray(beats, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(segments, dtype=np.float64).tobytes())
    h.update(json.dumps({**settings, "version": _CACHE_VERSION}, sort_keys=True).encode())
    return h.hexdigest()


def _record_candidates(audio_path: str, beats: np.ndarray, segments: np.ndarray, settings: dict) -> dict:
    """The production analysis path up to the learned jump point, recording
    the candidates the model scores for every placed group."""
    from analyzer.beat.frame_features import extract_frame_features
    from analyzer.phrase import detect_phrase_segments
    from analyzer.pipeline.track import fast_load
    from analyzer.audio.hpss import hpss_audio_with_spectra
    from analyzer.jumpcue.JumpCUE import JumpCueEngine
    from analyzer.jumpcue.jump_point_model import jump_point_beat_features

    class RecordingEngine(JumpCueEngine):
        def _model_points(self, groups, beats_time):
            model = self._jump_model
            beat_feats = model._beat_features(self._vocal, self._jump_features, self._score_cache)
            self.recorded = []
            for g in groups:
                points = np.asarray([m + g.offset for m in g.starts], dtype=int)
                model._active_harmony = self._jump_features.harmony
                try:
                    pairs, shifts = model._shifts(self._vocal, beat_feats, points, None, self._score_cache)
                finally:
                    model._active_harmony = None
                self.recorded.append({
                    "members": points, "pairs": [tuple(p) for p in pairs],
                    "cands": [{"d": int(d), "member": {k: np.asarray(member[k], dtype=float) for k in _MEMBER_KEYS},
                               "long": dict(pair["long"]), "short": dict(pair["short"]), "pre": dict(pair["pre"])}
                              for d, _ks, member, pair in shifts],
                })
            return super()._model_points(groups, beats_time)

    sr = settings["sr"]
    stereo = fast_load(audio_path, sr, stereo=True)
    mono = np.ascontiguousarray(stereo.mean(axis=1), dtype=np.float32)
    y_harm, y_perc, spectra = hpss_audio_with_spectra(mono, sr)
    frames = extract_frame_features(mono, sr, y_perc, y_harm, hop_length=settings["bpm_hop_length"])
    _segments, prob, feature_z = detect_phrase_segments(
        stereo, sr, beats, segments, model_path=settings["phrase_parameter_path"], hpss=spectra,
        return_boundary_probability=True, return_feature_z=True,
    )
    if prob is None:
        raise ValueError("too few beats for the phrase analysis")
    engine = RecordingEngine()
    features = jump_point_beat_features(frames, beats, segments, prob, phrase_feature_z=feature_z)
    engine.run(
        y_harm=y_harm, sr=sr, beats_time=beats, y_mix=mono, tempo_segments=segments,
        mix_magnitude=(spectra.magnitude if spectra is not None and (spectra.n_fft, spectra.hop_length)
                       == (engine.config.vocal.n_fft, engine.config.vocal.hop) else None),
        jump_point_features=features,
    )
    return {"records": getattr(engine, "recorded", []), "harmony": features.harmony}


def _build_track_cache(track: dict, cache_dir: str, settings: dict) -> str:
    """Spawn-pool entry: (re)build one track's candidate cache."""
    data = _record_candidates(str(track["audio_path"]), track["beats"], track["segments"], settings)
    save_cache(Path(cache_dir) / f"{track['uid']}.npz", {"key": track["cache_key"], **data})
    return str(track["uid"])


def _beat_sets(groups: list[list[float]], beats: np.ndarray) -> list[set[int]]:
    out = []
    for points in groups:
        idx = [int(np.argmin(np.abs(beats - t))) for t in points]
        if all(abs(beats[i] - t) <= _SNAP_TOLERANCE_SEC for i, t in zip(idx, points)) and len(set(idx)) >= 2:
            out.append(set(idx))
    return out


def _choice_for(record: dict, beats: set[int]) -> tuple[int, tuple[int, ...]] | None:
    for cand in record["cands"]:
        S = tuple(i for i, p in enumerate(record["members"]) if int(p) + cand["d"] in beats)
        if len(S) == len(beats):
            return int(cand["d"]), S
    return None


def _library_groups(track: dict, records: list[dict]) -> list[dict]:
    """Labeled groups of one track: each engine group with the saved group it
    can express, and the automatic choice for it when that differs."""
    saved = _beat_sets(track["saved"], track["beats"])
    auto = _beat_sets(track["auto"], track["beats"])
    used_saved: set[int] = set()
    used_auto: set[int] = set()
    out = []
    for record in records:
        label = None
        for i, group in enumerate(saved):
            if i not in used_saved and (choice := _choice_for(record, group)) is not None:
                used_saved.add(i)
                label = choice
                break
        if label is None:
            continue
        rejected = None
        for i, group in enumerate(auto):
            if i not in used_auto and (choice := _choice_for(record, group)) is not None:
                used_auto.add(i)
                rejected = choice if choice != label else None
                break
        out.append({"uid": track["uid"], "kind": "corrected" if rejected else "accepted", "phase": "library",
                    "members": record["members"], "pairs": record["pairs"],
                    "cands": [{**c, "masking": {}} for c in record["cands"]],
                    "label": label, "rejected": rejected})
    return out


def _rank_values(groups: list[GroupRows], weights: np.ndarray, threshold: float, names: list[str]) -> dict:
    """Per-group values: the saved point's rank among the candidates (AUC),
    whether the top pick is on a downbeat, and whether the saved choice
    scores above the automatic one where they differ."""
    down = names.index("downbeat_min") if "downbeat_min" in names else None
    auc, pick_down, above = [], [], []
    for g in groups:
        idx, X = _eligible_matrix(g, threshold)
        s = X @ weights
        t = np.flatnonzero(idx == g.target_idx)
        if not t.size:
            continue
        others = np.delete(s, int(t[0]))
        auc.append(float((np.sum(others < s[t[0]]) + 0.5 * np.sum(others == s[t[0]])) / max(others.size, 1)))
        pick = int(idx[int(np.argmax(s))])
        if down is not None:
            pick_down.append(float(g.raw_X[pick, down] > 0))
        if g.rejected_idx is not None:
            r = np.flatnonzero(idx == g.rejected_idx)
            if r.size:
                above.append(float(s[t[0]] > s[r[0]]))
    return {"auc": auc, "pick_downbeat": pick_down, "user_above_automatic": above}


def _mean_values(values: dict) -> dict:
    return {k: float(np.mean(v)) if v else float("nan") for k, v in values.items()}


def optimize_jumpcue_parameters(
    request: JumpCueOptimizationRequest,
    *,
    feature_progress=None,
    optimize_progress=None,
) -> JumpCueOptimizationResult:
    from concurrent.futures import as_completed

    from core.concurrency.process_pool import create_spawn_process_pool
    from optimizer import library_tracks, optimizer_track_name, skipped_track
    from utils.jump_cues import jump_cue_point_groups

    cache_dir = Path(request.cache_dir)
    output_path = Path(request.output_path)
    MS.check_output(output_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    settings = _analysis_settings(Path(request.phrase_parameter_path))
    skipped: list = []

    tracks: list[dict] = []
    for given in library_tracks.tracks():     # the tracks the library's owner gave this run, with their features
        info = {"uid": str(given["uid"]), "audio_path": Path(given["audio_path"]),
                "title": given["title"], "artist": given["artist"]}
        feats = given["features"]
        try:
            if feats is None:
                continue
            saved = jump_cue_point_groups(feats, "jump_cues_np")
            if not saved:
                continue
            beats = np.asarray(feats["beats_time_sec"], dtype=float)
            segments = np.asarray(feats["tempo_segments"], dtype=float)
            if not info["audio_path"].exists():
                skipped.append(skipped_track(info, tr("Audio file not found")))
                continue
            tracks.append({**info, "beats": beats, "segments": segments, "saved": saved,
                           "auto": jump_cue_point_groups(feats, "jump_cues_auto_np"),
                           "cache_key": _cache_key(beats, segments, settings)})
        except Exception as exc:
            skipped.append(skipped_track(info, tr("Reading the analysis failed: {error}").format(error=exc)))
    if len(tracks) < 5:
        raise RuntimeError(tr("Need at least five tracks with saved JumpCUEs ({count} found).").format(count=len(tracks)))

    def cached(track: dict) -> dict | None:
        path = cache_dir / f"{track['uid']}.npz"
        if request.rebuild_cache or not path.exists():
            return None
        try:
            data = load_cache(path)
        except Exception:
            return None
        return data if isinstance(data, dict) and data.get("key") == track["cache_key"] else None

    total = len(tracks)
    jobs = [t for t in tracks if cached(t) is None]
    done = total - len(jobs)
    if feature_progress is not None:
        feature_progress(int(done * 90 / total),
                         tr("Building {count} JumpCUE candidate caches").format(count=len(jobs)) if jobs
                         else tr("JumpCUE candidate cache is current"))
    if jobs:
        workers = max(1, min(request.max_cache_workers or 2, len(jobs)))
        failed: set[str] = set()
        with create_spawn_process_pool(workers) as executor:
            futures = {executor.submit(_build_track_cache, t, str(cache_dir), settings): t for t in jobs}
            for future in as_completed(futures):
                track = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed.add(track["uid"])
                    skipped.append(skipped_track(track, tr("JumpCUE analysis failed: {error}").format(error=exc)))
                done += 1
                if feature_progress is not None:
                    feature_progress(int(done * 90 / total), f"({done}/{total})\n{optimizer_track_name(track)}")
        tracks = [t for t in tracks if t["uid"] not in failed]

    base_path = MS.base_model(output_path, MS.CURRENT)   # the model in use: the user's, else the shipped one
    model = json.loads(Path(base_path).read_text(encoding="utf-8"))
    names = list(model["features"])
    threshold = float(model["short_threshold"])
    init = np.asarray(model["weights"], dtype=float)
    groups: list[GroupRows] = []
    for track in tracks:
        data = cached(track)
        if data is None:
            continue
        labeled = _library_groups(track, data["records"])
        if not labeled:
            skipped.append(skipped_track(track, tr("No saved JumpCUE group matches an automatically found group")))
            continue
        harmony = {track["uid"]: data["harmony"]} if data.get("harmony") is not None else {}
        groups.extend(_candidate_rows(g, names, threshold, harmony) for g in labeled)
    if feature_progress is not None:
        feature_progress(100, tr("{count} labeled groups").format(count=len(groups)))
    uids = np.asarray([g.uid for g in groups])
    if np.unique(uids).size < 5:
        raise RuntimeError(f"Need labeled groups on at least five tracks ({np.unique(uids).size} found).")
    _assign_positive_sets(groups, _metrics(groups, init, threshold), names, threshold)

    def fit(train: list[GroupRows]) -> np.ndarray:
        return _fit(train, init, set(), l2=request.l2, hard_weight=1.0,
                    corrected_weight=request.corrected_weight, short_threshold=threshold,
                    preference_weight=request.preference_weight)

    folds = int(request.cv_folds)
    cross_validated = folds >= 2 and np.unique(uids).size >= 2 * folds
    if cross_validated:
        pooled: dict = defaultdict(list)
        splits = list(GroupKFold(n_splits=folds).split(np.arange(len(groups)), groups=uids))
        for k, (train, valid) in enumerate(splits, start=1):
            if optimize_progress is not None:
                optimize_progress(int((k - 1) * 80 / folds), tr("Cross-validation fold {fold}/{folds}").format(fold=k, folds=folds))
            w = fit([groups[i] for i in train])
            for key, values in _rank_values([groups[i] for i in valid], w, threshold, names).items():
                pooled[key].extend(values)
        metrics = _mean_values(pooled)
    if optimize_progress is not None:
        optimize_progress(85, tr("Fitting on all tracks"))
    weights = fit(groups)
    if not cross_validated:
        metrics = _mean_values(_rank_values(groups, weights, threshold, names))

    pairs = sum(g.rejected_idx is not None for g in groups)
    out = dict(model)
    out["weights"] = [float(v) for v in weights]
    out["trained_on"] = {
        "groups": len(groups), "tracks": int(np.unique(uids).size),
        "corrected": sum(g.kind == "corrected" for g in groups),
        "accepted": sum(g.kind == "accepted" for g in groups),
        "preference_pairs": int(pairs), "l2": request.l2,
        "corrected_weight": request.corrected_weight, "preference_weight": request.preference_weight,
        "feature_count": len(names),
        "validation": f"{folds}-fold GroupKFold by track" if cross_validated else "none",
        "source": "library (saved JumpCUEs vs the automatic analysis result)",
    }
    out["label_model"] = {
        "objective": "set likelihood + pairwise preference",
        "max_positive_candidates": 4,
        "policy": "saved choice plus same-member continuity-equivalent candidates; where the automatic "
                  "result chose differently, the saved choice is preferred to it",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, out)
    if optimize_progress is not None:
        optimize_progress(100, tr("Done"))
    return JumpCueOptimizationResult(
        output_path=output_path, track_count=int(np.unique(uids).size), group_count=len(groups),
        preference_pairs=int(pairs), label_auc=metrics["auc"], pick_downbeat=metrics["pick_downbeat"],
        user_above_automatic=metrics["user_above_automatic"], cross_validated=bool(cross_validated),
        skipped_tracks=tuple(skipped),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and audit the JumpCUE point model")
    parser.add_argument("--dataset", type=Path, default=Path("featurecache/jumpcue/dataset_v5.npz"))
    parser.add_argument("--model", type=Path, default=Path("assets/weights/jumpcue_jump_point.json"))
    parser.add_argument("--label-model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("featurecache/jumpcue/training_report_current.json"))
    parser.add_argument("--stage", choices=("audit", "trial", "sweep", "finalize", "all"), default="all")
    parser.add_argument("--state-dir", type=Path, default=Path("featurecache/jumpcue/training_state"))
    parser.add_argument("--phrase-cache-dir", type=Path, default=Path("featurecache/phrase"))
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--corrected-weight", type=float, default=1.0)
    parser.add_argument("--preference-weight", type=float, default=3.0,
                        help="weight of the saved choice over a group's rejected model pick")
    parser.add_argument("--feature-set", choices=tuple(FEATURE_SET_TERMS), default="base")
    args = parser.parse_args()
    report = run(args.dataset, args.model, args.output, stage=args.stage,
                 state_dir=args.state_dir, trial_l2=args.l2,
                 phrase_cache_dir=args.phrase_cache_dir,
                 trial_hard_weight=args.hard_weight, trial_feature_set=args.feature_set,
                 trial_corrected_weight=args.corrected_weight,
                 trial_preference_weight=args.preference_weight,
                 label_model_path=args.label_model)
    if args.stage == "audit":
        shown = {k: report[k] for k in ("current_model", "hard_track_count", "weaknesses")}
    elif args.stage in {"trial", "sweep"}:
        shown = report
    else:
        shown = {k: report[k] for k in ("current_model", "hard_track_count", "best_cv", "full_fit", "weaknesses")}
    print(json.dumps(shown, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
