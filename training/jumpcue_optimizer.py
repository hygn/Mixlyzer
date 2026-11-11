"""The JumpCUE pair model (analyzer.jumpcue.pairs) retrained on the library's JumpCUE groups.

Labels (partial: a library holds only the jumps its owner uses):
  - every pair of a saved group (MIN_LAG beats apart or more) is a positive; a reviewed one, weighted
    ``reviewed_weight``, when the user changed or added the group (none of the automatic result's) or judged the whole
    track (``full_review_from``: the tracks added since one; ``full_review_upto``: those added up to one);
  - the pairs of an automatic group the user changed or removed (in no saved group) are known negatives (NEG_WEIGHT);
  - the other candidates are unlabelled, taken as negatives, each track's sharing one weight (UNLABELLED_SHARE).
One model, the pair scorer, on pairs.features of the candidates (and, in training, of the saved pairs that are none).
Reported, held out (folds by song: a track's versions together): the reviewed pairs' ranks among their track's
candidates, and the automatic pair (pairs.choose_pair under the default rules) against the reviewed groups.

The per-track arrays are the analysis's own (analyzer.pipeline.jumpcues.jumpcue_inputs, pairs.beat_features), cached by
the beat grid; the phrases are the track's as stored (the user's, as the editor gives them)."""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from analyzer.files import atomic_save_npz
from analyzer.jumpcue import pairs as P
from training.text import TrainingError, msg

CACHE_VERSION = 3           # 2: the repeats of analyzer.jumpcue.repeats; 3: no phrases, no boundary probability
SNAP_SEC = 0.1             # a saved point this far from a beat at most
NEG_WEIGHT = 2.0           # a pair of an automatic group the user changed or removed
UNLABELLED_SHARE = 200.0   # a track's unlabelled pairs weigh this much in all at most
# the gradient-boosting models: (trees, learning rate, leaves, smallest leaf)
GBM = {"pair": (300, 0.05, 15, 40)}


@dataclass(frozen=True)
class JumpCueOptimizationRequest:
    library_dir: Path
    cache_dir: Path
    output_path: Path
    cv_folds: int = 5                  # the held-out measures' folds: 2 or more
    reviewed_weight: float = 30.0
    # a track's uid (or its start): the tracks added since it, it too, had every group judged (their pairs not saved
    # are known unwanted, and their saved groups reviewed)
    full_review_from: str = ""
    # a track's uid (or its start): the tracks added up to it, it too, had every group judged as well
    full_review_upto: str = ""
    rebuild_cache: bool = False
    max_cache_workers: int | None = None


@dataclass(frozen=True)
class JumpCueOptimizationResult:
    output_path: Path
    track_count: int
    group_count: int                   # saved groups
    reviewed_groups: int
    candidate_pairs: int
    held_out: dict = field(default_factory=dict)
    skipped_tracks: tuple = ()


@dataclass
class _Track:
    uid: str
    song: str                          # its fold group
    t: P.PairTrack
    saved: list                        # [frozenset of beats]
    auto: list
    full: bool

    def reviewed(self, group: frozenset) -> bool:
        return self.full or (bool(self.auto) and group not in self.auto)

    def labels(self) -> tuple[dict, set]:
        """({(p, q): reviewed} of the saved groups' pairs, the known negative pairs)."""
        pos: dict = {}
        for g in self.saved:
            for p, q in itertools.combinations(sorted(g), 2):
                if q - p >= P.MIN_LAG:
                    pos[(p, q)] = pos.get((p, q), False) or self.reviewed(g)
        neg = {pq for g in self.auto if g not in self.saved for pq in itertools.combinations(sorted(g), 2)
               if pq not in pos}
        return pos, neg


def _snap(groups, beats: np.ndarray) -> list:
    out = []
    for g in groups:
        idx = [int(np.argmin(np.abs(beats - t))) for t in g]
        if all(abs(beats[i] - t) <= SNAP_SEC for i, t in zip(idx, g)) and len(set(idx)) >= 2:
            out.append(frozenset(idx))
    return out


def _gbm(kind: str, X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """A ``kind`` model fitted (refused when the examples hold one class: nothing to tell apart)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    if np.unique(y).size < 2:
        raise TrainingError(msg("Too few saved JumpCUE groups to learn the {model} model from.", model=kind))
    trees, rate, leaves, leaf = GBM[kind]
    m = HistGradientBoostingClassifier(max_iter=trees, learning_rate=rate, max_leaf_nodes=leaves,
                                       min_samples_leaf=leaf, l2_regularization=1.0, random_state=0)
    m.fit(X, y, sample_weight=w)
    return m


def _weights(y: np.ndarray, rev: np.ndarray, neg: np.ndarray, reviewed_weight: float) -> np.ndarray:
    w = np.ones(len(y))
    w[rev] = reviewed_weight
    w[neg] = NEG_WEIGHT
    unl = (y == 0) & ~neg
    if unl.any():
        w[unl] *= min(1.0, UNLABELLED_SHARE / unl.sum())
    return w


def _labelled(tr: _Track, pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos, neg = tr.labels()
    keys = [tuple(x) for x in pairs.tolist()]
    return (np.asarray([k in pos for k in keys], float), np.asarray([pos.get(k, False) for k in keys], bool),
            np.asarray([k in neg for k in keys], bool))


# ---------------------------------------------------------------------------------------------------------------------
# the per-track arrays, cached (NPZ: never pickle, a cache folder may come from elsewhere)

def _cache_key(beats: np.ndarray, segments: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(beats, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(segments, dtype=np.float64).tobytes())
    h.update(json.dumps({"version": CACHE_VERSION}, sort_keys=True).encode())
    return h.hexdigest()


def _build_cache(track: dict, cache_dir: str) -> str:
    """Spawn-pool entry: one track's arrays, as the analysis makes them, written to its cache."""
    from analyzer.pipeline.jumpcues import jumpcue_inputs

    beats = np.asarray(track["beats"], float)
    inputs = jumpcue_inputs(str(track["audio_path"]), beats)
    if inputs.context.vocal is None:
        raise ValueError("too few beats")
    feats = P.beat_features(inputs.frames, inputs.key_frames, beats, inputs.context.vocal, inputs.context.links)
    atomic_save_npz(Path(cache_dir) / f"{track['uid']}.npz", {"key": np.asarray(track["cache_key"]), **feats},
                    compressed=False)
    return str(track["uid"])


def _cached(path: Path, key: str) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["key"]) != key:
                return None
            return {k: z[k] for k in z.files}
    except (OSError, ValueError, KeyError):
        return None


def _pair_track(track: dict, arrays: dict) -> P.PairTrack:
    """The track's pair track: its cached arrays on its stored phrases (``track["phrases"]``)."""
    feats = {k: v for k, v in arrays.items() if k != "key"}
    return P.pair_track(track["beats"], track["segments"], track["phrases"], feats)


def _phrases(a) -> list:
    """The stored phrases, (start, end, label) each."""
    if a.phrase_starts is None:
        return []
    return [(float(s), float(e), str(lab)) for s, e, lab in zip(a.phrase_starts, a.phrase_ends, a.phrase_labels)]


# ---------------------------------------------------------------------------------------------------------------------
# the two models

def _pair_rows(tr: _Track) -> dict:
    """The track's candidates (``in_cand``) and its saved pairs that are none, their features and labels."""
    cand = P.candidates(tr.t)
    pos, _neg = tr.labels()
    cand_set = set(map(tuple, cand.tolist()))
    pairs = np.asarray(sorted(cand_set | set(pos)), int).reshape(-1, 2)
    X, names = P.features(tr.t, pairs)
    y, rev, neg = _labelled(tr, pairs)
    return {"pairs": pairs, "in_cand": np.asarray([tuple(x) in cand_set for x in pairs.tolist()], bool), "X": X,
            "names": names, "y": y, "rev": rev, "neg": neg}


def _fit_pairs(rows: list[dict], reviewed_weight: float):
    return _gbm("pair", np.concatenate([r["X"] for r in rows]), np.concatenate([r["y"] for r in rows]),
                np.concatenate([_weights(r["y"], r["rev"], r["neg"], reviewed_weight) for r in rows]))


# ---------------------------------------------------------------------------------------------------------------------
# held-out measures

def _ranks(tr: _Track, pairs: np.ndarray, scores: np.ndarray) -> list[int]:
    """Each reviewed pair's rank among the track's candidates (a pair that is no candidate: never found)."""
    pos, _neg = tr.labels()
    rank = np.empty(len(scores), int)
    rank[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
    index = {tuple(x): i for i, x in enumerate(pairs.tolist())}
    return [int(rank[index[pq]]) if pq in index else 10 ** 6 for pq, rev in pos.items() if rev]


def _section_runs(t) -> np.ndarray:
    """Each beat's section run (a run of consecutive phrases of one family, pairs.FAMILY: a CHORUS and a BREAK_CHORUS
    in it are one chorus; -1 before the first phrase)."""
    run_of = np.full(t.n, -1)
    run, prev = -1, None
    ps = t.phrase_start
    for i, (b0, lab) in enumerate(zip(ps, t.phrase_label)):
        fam = P.FAMILY.get(lab, lab)
        if fam != prev:
            run, prev = run + 1, fam
        run_of[int(b0):int(ps[i + 1]) if i + 1 < len(ps) else t.n] = run
    return run_of


def _placement(tr: _Track, chosen) -> dict:
    """The automatic pair against the reviewed groups: given at all; one of their pairs; the same jump (its two section
    runs a pair of one reviewed group's)."""
    reviewed = [g for g in tr.saved if tr.reviewed(g)]
    if not reviewed:
        return {}
    out = {"given": [chosen is not None]}
    if chosen is not None:
        p, q = int(chosen[0]), int(chosen[1])
        run = _section_runs(tr.t)
        out["exact"] = [any({p, q} <= g for g in reviewed)]
        out["same jump"] = [any((run[p], run[q]) in {(run[a], run[b]) for a, b in itertools.combinations(sorted(g), 2)}
                                for g in reviewed)]
    return out


def _recall(ranks: list[int]) -> dict:
    r = np.asarray(ranks)
    return {f"R@{k}": float(np.mean(r <= k)) if r.size else float("nan") for k in (1, 4, 8)} | {"pairs": int(r.size)}


# ---------------------------------------------------------------------------------------------------------------------

def optimize_jumpcue(request: JumpCueOptimizationRequest, *, feature_progress=None,
                     optimize_progress=None) -> JumpCueOptimizationResult:
    from concurrent.futures import as_completed

    from analyzer.runtime import spawn_pool
    from training import library_tracks, optimizer_track_name, skipped_track

    def say(progress, value, text):
        if progress is not None:
            progress(int(value), text)

    cache_dir = Path(request.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    folds = int(request.cv_folds)
    if folds < 2:
        raise TrainingError(msg("The JumpCUE models learn from held-out scores: cv_folds must be 2 or more."))
    skipped: list = []
    given = library_tracks.tracks()
    def added(uid: str):
        if not uid:
            return None
        found = [t for t in given if str(t["uid"]).startswith(str(uid))]
        if len(found) != 1:
            raise TrainingError(msg("No one track's uid starts with {uid}.", uid=uid))
        return int(found[0].get("added_ts", 0))
    full_from, full_upto = added(request.full_review_from), added(request.full_review_upto)

    tracks: list[dict] = []
    for track in given:
        info = {"uid": str(track["uid"]), "audio_path": Path(track["audio_path"]), "title": track["title"],
                "artist": track["artist"]}
        a = track.get("annotation")
        if a is None or not a.jump_cue_groups:
            continue
        if a.beats is None or a.tempo_segments is None:
            skipped.append(skipped_track(info, msg("No beat grid")))
            continue
        if not info["audio_path"].exists():
            skipped.append(skipped_track(info, msg("Audio file not found")))
            continue
        beats, segments = np.asarray(a.beats, float), np.asarray(a.tempo_segments, float)
        tracks.append({**info, "song": str(track.get("song") or track["uid"]), "beats": beats, "segments": segments,
                       "saved": _snap(a.jump_cue_groups, beats), "auto": _snap(a.jump_cue_auto_groups or [], beats),
                       "phrases": _phrases(a),
                       "full": ((full_from is not None and int(track.get("added_ts", 0)) >= full_from)
                                or (full_upto is not None and int(track.get("added_ts", 0)) <= full_upto)),
                       "cache_key": _cache_key(beats, segments)})

    def cache_path(track):
        return cache_dir / f"{track['uid']}.npz"
    jobs = [t for t in tracks if request.rebuild_cache or _cached(cache_path(t), t["cache_key"]) is None]
    done = len(tracks) - len(jobs)
    say(feature_progress, done * 90 / max(len(tracks), 1),
        msg("Analyzing {count} tracks for JumpCUE", count=len(jobs)) if jobs else msg("The JumpCUE cache is current"))
    failed: set[str] = set()
    if jobs:
        with spawn_pool(max(1, min(request.max_cache_workers or 2, len(jobs)))) as pool:
            futures = {pool.submit(_build_cache, t, str(cache_dir)): t for t in jobs}
            for future in as_completed(futures):
                track = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed.add(track["uid"])
                    skipped.append(skipped_track(track, msg("JumpCUE analysis failed: {error}", error=exc)))
                done += 1
                say(feature_progress, done * 90 / len(tracks), (f"({done}/{len(tracks)})", optimizer_track_name(track)))
    kept: list[_Track] = []
    for track in tracks:
        arrays = None if track["uid"] in failed else _cached(cache_path(track), track["cache_key"])
        if arrays is not None:
            kept.append(_Track(track["uid"], track["song"], _pair_track(track, arrays), track["saved"], track["auto"],
                               track["full"]))
    songs = np.asarray([tr.song for tr in kept])
    if np.unique(songs).size < max(5, folds):
        raise TrainingError(msg("Need saved JumpCUEs on at least {need} songs ({count} found).",
                                need=max(5, folds), count=int(np.unique(songs).size)))
    say(feature_progress, 100, msg("{count} tracks with saved JumpCUEs", count=len(kept)))

    # 1. the pair scorer, held out by song
    rows = [_pair_rows(tr) for tr in kept]
    splits = list(GroupKFold(n_splits=folds).split(np.arange(len(kept)), groups=songs))
    held: list = [None] * len(kept)
    for k, (train, valid) in enumerate(splits, start=1):
        say(optimize_progress, (k - 1) * 70 / folds, msg("Pair scorer: fold {fold}/{folds}", fold=k, folds=folds))
        model = _fit_pairs([rows[i] for i in train], request.reviewed_weight)
        for i in valid:
            X = rows[i]["X"][rows[i]["in_cand"]]
            held[i] = model.predict_proba(X)[:, 1] if len(X) else np.zeros(0)     # a track with no candidate
    cands = [r["pairs"][r["in_cand"]] for r in rows]

    # 2. the automatic pair on the held-out scores
    say(optimize_progress, 70, msg("The automatic pair"))
    placement: dict = {"given": [], "exact": [], "same jump": []}
    for i in range(len(kept)):
        for key, values in _placement(kept[i], P.choose_pair(kept[i].t, cands[i], held[i])).items():
            placement[key].extend(values)
    held_out = {
        "pair scorer": _recall([r for tr, c, s in zip(kept, cands, held) for r in _ranks(tr, c, s)]),
        "automatic pair": {"tracks given one": float(np.mean(placement["given"] or [np.nan])),
                           "a reviewed pair": float(np.mean(placement["exact"] or [np.nan])),
                           "the same jump": float(np.mean(placement["same jump"] or [np.nan]))},
    }

    # the model written: fitted on every track
    say(optimize_progress, 75, msg("Fitting on all tracks"))
    pair_model = _fit_pairs(rows, request.reviewed_weight)
    from analyzer.phrase.phrase_analyzer import _hgb_to_npz_payload
    payload = {"format": np.asarray(P.FORMAT), "feature_names": np.asarray(rows[0]["names"])}
    _hgb_to_npz_payload("pair", pair_model, payload)

    def check(path: Path) -> None:
        """The app's NumPy runtime reads the written file as sklearn scores."""
        app = P.load_models.__wrapped__(str(path))
        X = rows[0]["X"]
        if len(X) and np.max(np.abs(app.pair.predict_proba(X)[:, 1] - pair_model.predict_proba(X)[:, 1])) > 1e-9:
            raise RuntimeError("the app's NumPy runtime differs from the trained model")
    output = Path(request.output_path)
    atomic_save_npz(output, payload, compressed=False, check=check)
    P.load_models.cache_clear()
    say(optimize_progress, 100, msg("Done"))
    return JumpCueOptimizationResult(
        output_path=output, track_count=len(kept), group_count=sum(len(tr.saved) for tr in kept),
        reviewed_groups=sum(tr.reviewed(g) for tr in kept for g in tr.saved),
        candidate_pairs=int(sum(len(c) for c in cands)), held_out=held_out, skipped_tracks=tuple(skipped))
