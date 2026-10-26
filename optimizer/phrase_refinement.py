"""Training of the Phrase second pass (see analyzer/phrase/refinement.py).

The pass-2 boundary GBM, the pass-2 label GBM and the bar-pair same-label GBM
learn from pass-1 outputs. Those outputs must look like they will at
inference, i.e. come from models that never saw the track, so every training
track's pass-1 outputs and learned similarity are produced out of fold (inner
track-level K-fold) before the pass-2 models are fit.

Benchmark (benchmark/phrase_boundary, 241 library tracks, 5-fold CV): F1 of
the label-change boundaries 0.744 -> 0.778, beat label accuracy 0.876 -> 0.901,
all-boundary F1 unchanged.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from analyzer.phrase.model_features import segment_feature_matrix
from analyzer.phrase.phrase_analyzer import (
    _predict_boundary_probability,
    decode_phrase_boundaries,
    finish_phrase_segments,
    pass_one_context,
    refinement_boundary_matrix,
    refinement_label_matrix,
    similarity_from_inputs,
)
from analyzer.phrase.lead_in import LEAD_IN_VERSION
from analyzer.phrase.refinement import (
    beat_posterior,
    label_change_features,
    phrase_grid_features,
    shift_candidates,
)
from optimizer import phrase_lead_in as _lead_in
from foundation.i18n import tr

ProgressCallback = Callable[[int, str], None]

INNER_FOLDS = 4
# Boundaries between two different labels weigh this much more than ones
# between two same-label phrases in the pass-2 boundary GBM.
CHANGE_WEIGHT = 3.0
NEGATIVE_GUARD_BEATS = 2
PAIR_SAMPLE_FRACTION = 0.3
BOUNDARY2_PARAMS = dict(learning_rate=0.05, max_iter=420, max_leaf_nodes=31, max_depth=3,
                        min_samples_leaf=10, l2_regularization=2.0)
LABEL2_PARAMS = dict(learning_rate=0.05, max_iter=360, max_leaf_nodes=15, max_depth=3,
                     min_samples_leaf=8, l2_regularization=0.5)
PAIR_PARAMS = dict(learning_rate=0.1, max_iter=250, max_leaf_nodes=31, max_depth=6,
                   min_samples_leaf=50, l2_regularization=1.0)
# Label-change GBM: weight of its pairwise term in the label Viterbi.
LABEL_CHANGE_WEIGHT = 1.0
CHANGE_PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, max_depth=4,
                     min_samples_leaf=20, l2_regularization=1.0)
# Bar-shift GBM: changes between two different key labels move to the best of
# these offsets (whole bars, up to 4 either way). Changes into or out of SILENCE
# / BRIDGE matter little for DJ use and are left alone.
KEY_LABELS = ("INTRO", "VERSE", "CHORUS", "BREAK_CHORUS", "INTERLUDE", "OUTRO")
SHIFT_OFFSETS_BEATS = tuple(range(-16, 17, 4))
SHIFT_EDGE_BEATS = 8
SHIFT_PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, max_depth=4,
                    min_samples_leaf=20, l2_regularization=1.0)


def _interior(bounds: np.ndarray, n: int) -> list[tuple[int, int]]:
    """(beat, k) for every interior boundary; k indexes the label it starts."""
    return [(int(b), k) for k, b in enumerate(np.asarray(bounds)) if 0 < int(b) < n]


def bar_labels(bounds: np.ndarray, labels: list[str], bars: np.ndarray) -> np.ndarray:
    seg = np.searchsorted(np.asarray(bounds), bars, side="right") - 1
    return np.asarray(labels, dtype=object)[np.clip(seg, 0, len(labels) - 1)]


def fit_pair_model(prepared: list[dict], indices, targets, seed: int):
    rng = np.random.default_rng(int(seed))
    xs, ys = [], []
    for i in indices:
        inputs = prepared[int(i)]["repeat_inputs"]
        bounds, labels = targets[int(i)]
        lab = bar_labels(bounds, labels, inputs["bars"])
        keep = rng.random(inputs["pair_a"].size) < PAIR_SAMPLE_FRACTION
        xs.append(np.asarray(inputs["pair_X"][keep], dtype=np.float32))
        ys.append((lab[inputs["pair_a"]] == lab[inputs["pair_b"]])[keep].astype(np.int8))
    clf = HistGradientBoostingClassifier(early_stopping=False, random_state=int(seed), **PAIR_PARAMS)
    clf.fit(np.vstack(xs), np.concatenate(ys))
    return clf


def fit_boundary2(prepared: list[dict], indices, targets, extra: dict[int, np.ndarray], seed: int):
    """Pass-2 boundary GBM on [pass-1 boundary features | refinement columns]."""
    xs, ys, ws = [], [], []
    for i in indices:
        track = prepared[int(i)]
        X = np.hstack([np.asarray(track["boundary_feature_z"], dtype=np.float32), extra[int(i)]])
        n = X.shape[0]
        y = np.zeros(n, np.int8)
        w = np.ones(n)
        use = np.asarray(track["valid_mask"], dtype=bool).copy()
        bounds, labels = targets[int(i)]
        for b, k in _interior(bounds, n):
            y[b] = 1
            if 0 < k < len(labels) and labels[k - 1] != labels[k]:
                w[b] = CHANGE_WEIGHT
            guard = np.zeros(n, bool)
            guard[max(0, b - NEGATIVE_GUARD_BEATS):b + NEGATIVE_GUARD_BEATS + 1] = True
            guard[b] = False
            use &= ~guard
        xs.append(X[use])
        ys.append(y[use])
        ws.append(w[use])
    X, y, w = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
    pos = float(np.sum(w * y) / np.sum(w))
    w = w * np.where(y > 0, (1.0 - pos) / max(pos, 1e-6), 1.0)
    w *= w.size / w.sum()
    clf = HistGradientBoostingClassifier(early_stopping=False, random_state=int(seed), **BOUNDARY2_PARAMS)
    clf.fit(X, y, sample_weight=w)
    return clf


def fit_label2(prepared: list[dict], indices, targets, oof: dict[int, dict], seed: int):
    """Pass-2 label GBM on the annotated segmentation with out-of-fold pass-1 context."""
    xs, ys = [], []
    for i in indices:
        track = prepared[int(i)]
        bounds, labels = targets[int(i)]
        xs.append(refinement_label_matrix(track["feature_z"], np.asarray(bounds, np.int32),
                                          track["label_context_beats"], oof[int(i)]["true_beat_post"],
                                          track["repeat_inputs"], oof[int(i)]["S"]))
        ys += [str(label) for label in labels]
    y = np.asarray(ys, dtype=object)
    names, counts = np.unique(y, return_counts=True)
    weight_of = {name: 1.0 / count for name, count in zip(names, counts)}
    w = np.asarray([weight_of[v] for v in y])
    clf = HistGradientBoostingClassifier(early_stopping=False, random_state=int(seed), **LABEL2_PARAMS)
    clf.fit(np.vstack(xs), y, sample_weight=w * w.size / w.sum())
    return clf


def fit_change_model(prepared: list[dict], indices, targets, oof: dict[int, dict], seed: int):
    """P(label changes) between adjacent annotated segments, from out-of-fold pass-1 context."""
    xs, ys = [], []
    for i in indices:
        track = prepared[int(i)]
        bounds, labels = targets[int(i)]
        X = label_change_features(track["feature_z"], np.asarray(bounds, np.int32), track["label_context_beats"],
                                  oof[int(i)]["true_beat_post"], oof[int(i)]["S"], track["repeat_inputs"]["bars"])
        if X.shape[0]:
            xs.append(X)
            ys += [int(str(a) != str(b)) for a, b in zip(labels[:-1], labels[1:])]
    clf = HistGradientBoostingClassifier(early_stopping=False, random_state=int(seed), **CHANGE_PARAMS)
    clf.fit(np.vstack(xs), np.asarray(ys, dtype=np.int8))
    return clf


def fit_shift_model(prepared: list[dict], indices, targets, extra: dict[int, np.ndarray], oof: dict[int, dict],
                    seed: int):
    """Ranks the true key-label change against the same track's beats 1-4 bars away,
    with the phrase grid of the (out-of-fold) pass-1 boundaries."""
    keys = set(KEY_LABELS)
    xs, ys = [], []
    for i in indices:
        track = prepared[int(i)]
        n = int(np.asarray(track["feature_z"]).shape[0])
        bounds, labels = targets[int(i)]
        centers = [b for b, k in _interior(bounds, n)
                   if 0 < k < len(labels) and labels[k - 1] != labels[k]
                   and labels[k - 1] in keys and labels[k] in keys]
        if not centers:
            continue
        X = np.hstack([np.asarray(track["boundary_feature_z"], dtype=np.float32), extra[int(i)],
                       phrase_grid_features(oof[int(i)]["bounds"], n)])
        for c in centers:
            cands = shift_candidates(c, 0, n, n, SHIFT_OFFSETS_BEATS, SHIFT_EDGE_BEATS)
            xs.append(X[cands])
            ys.append((np.asarray(cands) == c).astype(np.int8))
    clf = HistGradientBoostingClassifier(early_stopping=False, random_state=int(seed), **SHIFT_PARAMS)
    clf.fit(np.vstack(xs), np.concatenate(ys))
    return clf


def _pass_one_artifact(boundary_clf, label_model, settings: dict) -> dict:
    return {
        "settings": settings,
        "boundary_clf": boundary_clf,
        "label_clf": label_model.clf,
        "label_labels": list(label_model.labels),
        "label_transition": label_model.transition,
        "label_length_mu": label_model.length_mu,
        "label_length_sigma": label_model.length_sigma,
    }


def pass_one_outputs(track: dict, artifact: dict, pair_clf, true_bounds=None) -> dict:
    """Everything pass 2 reads from pass 1 for one track."""
    settings = artifact["settings"]
    ctx = track["label_context_beats"]
    valid = np.asarray(track["valid_mask"], dtype=bool)
    downbeat = np.asarray(track["downbeat"], dtype=bool)
    p = _predict_boundary_probability(artifact["boundary_clf"], track["boundary_feature_z"], valid)
    bounds = decode_phrase_boundaries(p, valid, downbeat, settings)
    out = {"probability": p, "bounds": bounds,
           "context": pass_one_context(artifact["label_clf"], track["feature_z"], bounds, valid, downbeat, ctx),
           "S": similarity_from_inputs(pair_clf, track["repeat_inputs"]),
           "lead_in": _lead_in.lead_in_features(track, artifact["boundary_clf"], bounds, settings["n_mels"])}
    if true_bounds is not None:
        tb = np.asarray(true_bounds, np.int32)
        post = artifact["label_clf"].predict_proba(
            segment_feature_matrix(np.asarray(track["feature_z"], dtype=np.float64), tb, ctx))
        out["true_beat_post"] = beat_posterior(post, tb, np.asarray(track["feature_z"]).shape[0])
    return out


def refinement_columns(track: dict, out: dict) -> np.ndarray:
    """The pass-2 boundary columns that follow the pass-1 boundary features
    (a zero-width base so only the refinement columns are built)."""
    empty = np.zeros((out["probability"].size, 0), np.float32)
    return refinement_boundary_matrix(empty, out["probability"], out["bounds"], out["context"],
                                      track["repeat_inputs"], out["S"]).astype(np.float32)


def fit_two_pass(
    prepared: list[dict],
    indices: np.ndarray,
    targets: list,
    settings: dict,
    fit_pass_one: Callable,
    seed: int,
    progress: ProgressCallback | None = None,
) -> dict:
    """Fit pass 1 and pass 2 on ``indices``; returns an artifact-shaped dict.

    ``fit_pass_one(prepared, indices, targets, seed) -> (boundary_clf, label_model)``.
    """
    indices = np.asarray(indices)
    say = progress or (lambda _pct, _msg: None)
    oof: dict[int, dict] = {}
    n_inner = min(INNER_FOLDS, indices.size)
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=n_inner).split(indices, groups=indices),
                                                 start=1):
        say(int(round((fold - 1) * 60 / n_inner)),
            tr("Pass-1 outputs out of fold {fold}/{folds}").format(fold=fold, folds=n_inner))
        train = indices[train_idx]
        boundary_clf, label_model = fit_pass_one(prepared, train, targets, seed)
        pair_clf = fit_pair_model(prepared, train, targets, seed + 400)
        artifact = _pass_one_artifact(boundary_clf, label_model, settings)
        for i in indices[test_idx]:
            oof[int(i)] = pass_one_outputs(prepared[int(i)], artifact, pair_clf, targets[int(i)][0])

    say(60, tr("Fitting pass-1 boundary and label GBMs"))
    boundary_clf, label_model = fit_pass_one(prepared, indices, targets, seed)
    say(70, tr("Fitting bar-pair similarity GBM"))
    pair_clf = fit_pair_model(prepared, indices, targets, seed + 400)
    say(78, tr("Fitting pass-2 boundary GBM"))
    extra = {int(i): refinement_columns(prepared[int(i)], oof[int(i)]) for i in indices}
    boundary2 = fit_boundary2(prepared, indices, targets, extra, seed + 200)
    say(88, tr("Fitting pass-2 label GBM"))
    label2 = fit_label2(prepared, indices, targets, oof, seed + 300)
    if [str(c) for c in label2.classes_] != [str(c) for c in label_model.labels]:
        raise RuntimeError("Pass-2 label classes differ from pass-1 label classes")
    say(94, tr("Fitting label-change and bar-shift GBMs"))
    change = fit_change_model(prepared, indices, targets, oof, seed + 500)
    shift = fit_shift_model(prepared, indices, targets, extra, oof, seed + 600)
    say(97, tr("Fitting lead-in GBMs"))
    lead_in, threshold = _lead_in.fit_lead_in({i: oof[i]["lead_in"] for i in oof}, indices, targets, seed + 700)
    if lead_in is not None:
        settings = {**settings, "lead_in_version": LEAD_IN_VERSION, "lead_in_threshold": threshold}
    artifact = _pass_one_artifact(boundary_clf, label_model, settings)
    artifact["refinement"] = {"boundary2": boundary2, "label2": label2, "pair": pair_clf,
                              "change": change, "shift": shift}
    artifact["lead_in"] = lead_in
    return artifact


def predict_track(track: dict, artifact: dict) -> tuple[np.ndarray, list[str]]:
    """Full pipeline (pass 1, lead-in, pass 2) on a prepared track, as the analyzer runs it."""
    settings = artifact["settings"]
    valid = np.asarray(track["valid_mask"], dtype=bool)
    downbeat = np.asarray(track["downbeat"], dtype=bool)
    p = _predict_boundary_probability(artifact["boundary_clf"], track["boundary_feature_z"], valid)
    bounds = decode_phrase_boundaries(p, valid, downbeat, settings)
    # Pass 2 adds its columns to the pass-1 boundary features itself.
    return finish_phrase_segments(artifact, track["feature_z"], track["boundary_feature_z"], valid, downbeat,
                                  track["beat_times_sec"], p, bounds, track["repeat_inputs"])


def _matches(pred: np.ndarray, truth: np.ndarray) -> int:
    return int(np.intersect1d(np.asarray(pred), np.asarray(truth)).size)


def pipeline_metrics(prepared: list[dict], targets: list, predictions: list) -> dict[str, float]:
    """Decoded-pipeline metrics, exact beat: all-boundary F1, F1 of the boundaries
    between two different labels, beat label accuracy and exactly right lead-ins."""
    keys = set(KEY_LABELS)
    hit = n_pred = n_true = 0
    c_hit = c_pred = c_true = 0
    k_hit = k_pred = k_true = 0
    beats = beats_ok = 0

    def changes(bounds, labels, n, key_only=False):
        return np.asarray([b for b, k in _interior(bounds, n)
                           if 0 < k < len(labels) and labels[k - 1] != labels[k]
                           and (not key_only or (labels[k - 1] in keys and labels[k] in keys))])

    for track, (tb, tl), (pb, pl) in zip(prepared, targets, predictions, strict=True):
        n = int(np.asarray(track["feature_z"]).shape[0])
        truth_b = np.asarray([b for b, _ in _interior(tb, n)])
        pred_b = np.asarray([b for b, _ in _interior(pb, n)])
        hit += _matches(pred_b, truth_b)
        n_pred += pred_b.size
        n_true += truth_b.size
        for key_only in (False, True):
            truth_c, pred_c = changes(tb, tl, n, key_only), changes(pb, pl, n, key_only)
            if key_only:
                k_hit += _matches(pred_c, truth_c)
                k_pred += pred_c.size
                k_true += truth_c.size
            else:
                c_hit += _matches(pred_c, truth_c)
                c_pred += pred_c.size
                c_true += truth_c.size
        truth = np.empty(n, dtype=object)
        pred = np.empty(n, dtype=object)
        for (s, e), lab in zip(zip(tb[:-1], tb[1:]), tl):
            truth[int(s):int(e)] = str(lab)
        for (s, e), lab in zip(zip(pb[:-1], pb[1:]), pl):
            pred[int(s):int(e)] = str(lab)
        beats += n
        beats_ok += int(np.sum(truth == pred))
    return {
        "boundary_f1": 2.0 * hit / max(n_pred + n_true, 1),
        "label_change_f1": 2.0 * c_hit / max(c_pred + c_true, 1),
        "key_change_f1": 2.0 * k_hit / max(k_pred + k_true, 1),
        "label_accuracy": beats_ok / max(beats, 1),
        "lead_in_accuracy": _lead_in.lead_in_accuracy(targets, predictions),
    }
