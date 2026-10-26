"""Training of the Phrase LEAD_IN models (see analyzer/phrase/lead_in.py).

The features read pass-1 outputs, so every training track's features come from
the out-of-fold pass-1 models of ``phrase_refinement.fit_two_pass``. Only song
starts are examples: a track has a lead-in when its first annotated segment is
LEAD_IN. The presence threshold is the one that gets the most tracks exactly
right (presence and end) in a track-level K-fold over these features.

Benchmark (benchmark/phrase_boundary/LEADIN_EXPERIMENT.md, 241 library tracks,
10x 5-fold CV): presence AP 0.999, exact end 0.91 of the lead-ins, 0.90 of the
tracks exactly right.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from analyzer.phrase import lead_in as _lead_in

# Tiny models: a few hundred tracks, one row (presence) or five rows (location) each.
PARAMS = dict(learning_rate=0.05, max_iter=150, max_leaf_nodes=4, max_depth=2, min_samples_leaf=8,
              l2_regularization=1.0, early_stopping=False)
THRESHOLD_FOLDS = 5
# Fewer tracks of either kind than this and no lead-in models are fit.
MIN_TRACKS_PER_CLASS = 5


def track_lead_in(features: dict, target: tuple) -> tuple[bool, int]:
    """(has a lead-in, index of its end among the candidates or -1)."""
    bounds, labels = target
    if not labels or str(labels[0]) != _lead_in.LEAD_IN_LABEL or len(bounds) < 3:
        return False, -1
    hit = np.flatnonzero(np.asarray(features["candidates"]) == int(bounds[1]))
    return True, int(hit[0]) if hit.size else -1


def _fit(X, y, seed: int):
    pos = max(float(np.mean(y)), 1e-3)
    w = np.where(y > 0, (1.0 - pos) / pos, 1.0)
    return HistGradientBoostingClassifier(random_state=int(seed), **PARAMS).fit(X, y, sample_weight=w)


def _fit_pair(rows: list[tuple[dict, bool, int]], seed: int) -> dict | None:
    has = np.asarray([h for _f, h, _j in rows], dtype=np.int8)
    if min(int(has.sum()), int(has.size - has.sum())) < MIN_TRACKS_PER_CLASS:
        return None
    located = [(f, j) for f, h, j in rows if h and j >= 0]
    if len(located) < MIN_TRACKS_PER_CLASS:
        return None
    presence = _fit(np.vstack([f["presence"] for f, _h, _j in rows]), has, seed)
    X = np.vstack([f["location"] for f, _j in located])
    y = np.asarray([int(j == c) for _f, j in located for c in range(_lead_in.N_CANDIDATES)], dtype=np.int8)
    ok = ~np.isnan(X[:, 0])
    return {"presence": presence, "location": _fit(X[ok], y[ok], seed + 1)}


def _threshold(rows: list[tuple[dict, bool, int]], seed: int) -> float:
    has = np.asarray([h for _f, h, _j in rows], dtype=np.int8)
    folds = min(THRESHOLD_FOLDS, int(has.sum()), int(has.size - has.sum()))
    scored = []
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=int(seed)).split(np.zeros(has.size), has):
        models = _fit_pair([rows[i] for i in train], seed)
        if models is None:
            continue
        for i in test:
            f, h, j = rows[i]
            n = int(np.asarray(f["candidates"]).size)
            s = float(models["presence"].predict_proba(f["presence"][np.newaxis, :])[0, 1])
            pick = int(np.argmax(models["location"].predict_proba(f["location"][:n])[:, 1])) if n else -1
            scored.append((s, h, h and pick == j))
    if not scored:
        return 0.5
    # Saying "lead-in" at s >= tau: right for a located lead-in, wrong for a track without one.
    taus = sorted({s for s, _h, _ok in scored})
    correct = [sum((s >= t and ok) or (s < t and not h) for s, h, ok in scored) for t in taus]
    return float(taus[int(np.argmax(correct))])


def fit_lead_in(features: dict[int, dict], indices, targets: list, seed: int) -> tuple[dict | None, float]:
    """(models or None, presence threshold) from out-of-fold pass-1 features."""
    rows = [(features[int(i)], *track_lead_in(features[int(i)], targets[int(i)])) for i in indices]
    models = _fit_pair(rows, seed)
    if models is None:
        return None, 0.5
    return models, _threshold(rows, seed + 7)


def lead_in_features(track: dict, boundary_clf, pass_one_bounds, n_mels: int) -> dict:
    cands, loc, presence = _lead_in.lead_in_features(
        track["feature_z"],
        _lead_in.raw_probability_head(boundary_clf, track["boundary_feature_z"], track["downbeat"]),
        track["downbeat"], track["beat_times_sec"], pass_one_bounds, int(n_mels),
    )
    return {"candidates": cands, "location": loc, "presence": presence}


def lead_in_accuracy(targets: list, predictions: list) -> float:
    """Tracks whose lead-in is exactly right: none in both, or the same end beat."""
    ok = 0
    for (tb, tl), (pb, pl) in zip(targets, predictions, strict=True):
        t = int(tb[1]) if tl and str(tl[0]) == _lead_in.LEAD_IN_LABEL and len(tb) > 2 else None
        p = int(pb[1]) if pl and str(pl[0]) == _lead_in.LEAD_IN_LABEL and len(pb) > 2 else None
        ok += t == p
    return ok / max(len(targets), 1)
