"""LEAD_IN: the short stretch at the very start of a song, before its first phrase.

The boundary decoder cannot place it (``edge_beats`` and the minimum phrase
distance keep boundaries off the first beats), so after the boundaries a small
presence GBM (one row a song) says whether the song has one and a location GBM
(one row per candidate) picks where it ends. Every annotated lead-in ends on one
of the first downbeats, so those are the candidates. The LEAD_IN label is then
given by this step alone: the label decoder puts it on the first segment when
there is a lead-in and nowhere else.

Features come from the per-beat ``feature_z`` rows, the pass-1 boundary
probability without the edge mask and the pass-1 boundaries, which the
optimizer has for every training track (benchmark/phrase_boundary/LEADIN_EXPERIMENT.md).
"""

from __future__ import annotations

import numpy as np

LEAD_IN_VERSION = 1
LEAD_IN_LABEL = "LEAD_IN"
N_CANDIDATES = 5          # first downbeats a lead-in may end on
N_PRESENCE_CANDIDATES = 3
AFTER_BEATS = 8
# Boundaries closer than this after the lead-in end are dropped with it.
MIN_GAP_BEATS = 4
# feature_z columns (model_features.feature_z_from_acoustic): the rhythm family ends
# with the three onset rows, the texture family is the last 15 rows.
_COLUMNS = {
    "onset_full": -18,
    "onset_percussive": -17,
    "log_rms": -15,
    "harmonic_log_rms": -14,
    "percussive_log_rms": -13,
    "low_log_energy": -12,
    "high_log_energy": -9,
    "flatness": -4,
    "pitch_salience": -2,
    "vocal_proxy": -1,
}
_PRESENCE_PER_CANDIDATE = ("b_log_rms", "a_log_rms", "hit", "d_log_rms", "d_percussive_log_rms",
                           "d_onset_percussive", "d_vocal_proxy", "mel_cos", "grid16")
LOCATION_FEATURES = (
    "j", "k", "t", "p1", "p1_rel", "next_bound", "grid16", "grid32", "hit", "a_log_rms", "mel_cos",
    *(f"{p}_{name}" for name in _COLUMNS for p in ("d", "b")),
)
PRESENCE_FEATURES = (
    "b0", "t_fd", "p1_max", "p1_d0",
    *(f"c{j}_{name}" for j in range(N_PRESENCE_CANDIDATES) for name in _PRESENCE_PER_CANDIDATE),
)


def candidates(downbeat: np.ndarray) -> np.ndarray:
    """Beat indices a lead-in may end on: the first downbeats after beat 0."""
    db = np.flatnonzero(np.asarray(downbeat, dtype=bool))
    return db[db > 0][:N_CANDIDATES].astype(np.int32)


def head_beats(downbeat: np.ndarray) -> int:
    """Beats of pass-1 probability the features read."""
    cands = candidates(downbeat)
    return int(cands[-1]) + AFTER_BEATS + 1 if cands.size else AFTER_BEATS + 1


def _rows(feature_z, p1, beat_times, pass_one_bounds, cands, n_mels) -> list[dict[str, float]]:
    fz = np.asarray(feature_z, dtype=np.float64)
    n = fz.shape[0]
    mel = fz[:, :n_mels]
    interior = [int(b) for b in np.asarray(pass_one_bounds)[1:-1]]
    top = max((float(p1[k]) for k in cands), default=0.0)
    rows = []
    for j, k in enumerate(int(x) for x in cands):
        after = slice(k, min(k + AFTER_BEATS, n))
        nb = next((b for b in interior if b > k), None)
        f = {
            "j": float(j), "k": float(k), "t": float(beat_times[k]), "p1": float(p1[k]),
            "p1_rel": float(p1[k]) - top,
            "next_bound": float(min(nb - k, 64) if nb is not None else 64),
            "grid16": float(nb is not None and (nb - k) % 16 == 0),
            "grid32": float(nb is not None and (nb - k) % 32 == 0),
            "hit": float(fz[k, _COLUMNS["onset_full"]]),
            "a_log_rms": float(np.mean(fz[after, _COLUMNS["log_rms"]])),
        }
        for name, col in _COLUMNS.items():
            before = float(np.mean(fz[:k, col]))
            f[f"d_{name}"] = float(np.mean(fz[after, col])) - before
            f[f"b_{name}"] = before
        a, b = mel[:k].mean(axis=0), mel[after].mean(axis=0)
        f["mel_cos"] = float(1.0 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        rows.append(f)
    return rows


def lead_in_features(
    feature_z: np.ndarray,
    raw_probability_head: np.ndarray,
    downbeat: np.ndarray,
    beat_times_sec: np.ndarray,
    pass_one_bounds: np.ndarray,
    n_mels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(candidate beats, location rows padded to N_CANDIDATES with NaN, presence row).

    ``raw_probability_head``: pass-1 boundary probability of the first
    ``head_beats(downbeat)`` beats, without the edge mask.
    """
    cands = candidates(downbeat)
    p1 = np.zeros(np.asarray(feature_z).shape[0], dtype=np.float64)
    head = np.asarray(raw_probability_head, dtype=np.float64).reshape(-1)[: p1.size]
    p1[: head.size] = head
    rows = _rows(feature_z, p1, beat_times_sec, pass_one_bounds, cands, int(n_mels))
    loc = np.full((N_CANDIDATES, len(LOCATION_FEATURES)), np.nan, dtype=np.float64)
    for j, r in enumerate(rows):
        loc[j] = [r[name] for name in LOCATION_FEATURES]
    pres_rows = rows[:N_PRESENCE_CANDIDATES]
    presence = {
        "b0": float(beat_times_sec[0]),
        "t_fd": pres_rows[0]["t"] if pres_rows else np.nan,
        "p1_max": max((r["p1"] for r in pres_rows), default=np.nan),
        "p1_d0": pres_rows[0]["p1"] if pres_rows else np.nan,
    }
    for j in range(N_PRESENCE_CANDIDATES):
        for name in _PRESENCE_PER_CANDIDATE:
            presence[f"c{j}_{name}"] = pres_rows[j][name] if j < len(pres_rows) else np.nan
    return cands, loc, np.asarray([presence[name] for name in PRESENCE_FEATURES], dtype=np.float64)


def raw_probability_head(boundary_clf, boundary_feature_z: np.ndarray, downbeat: np.ndarray) -> np.ndarray:
    head = min(head_beats(downbeat), int(np.asarray(boundary_feature_z).shape[0]))
    X = np.asarray(boundary_feature_z[:head], dtype=np.float64)
    return np.clip(boundary_clf.predict_proba(X)[:, 1], 1e-6, 1.0 - 1e-6)


def choose_lead_in(models: dict, threshold: float, cands: np.ndarray, loc: np.ndarray,
                   presence: np.ndarray) -> int | None:
    """End beat of the lead-in, or None."""
    if not cands.size:
        return None
    if float(models["presence"].predict_proba(presence[np.newaxis, :])[0, 1]) < float(threshold):
        return None
    score = models["location"].predict_proba(loc[: cands.size])[:, 1]
    return int(cands[int(np.argmax(score))])


def predict_lead_in(artifact: dict, feature_z, boundary_feature_z, downbeat, beat_times_sec,
                    pass_one_bounds) -> int | None:
    """The artifact's lead-in end beat for one track (None: no lead-in or no lead-in models)."""
    models = artifact.get("lead_in")
    if not models:
        return None
    settings = artifact["settings"]
    cands, loc, presence = lead_in_features(
        feature_z,
        raw_probability_head(artifact["boundary_clf"], boundary_feature_z, downbeat),
        downbeat, beat_times_sec, pass_one_bounds, int(settings["n_mels"]),
    )
    return choose_lead_in(models, float(settings["lead_in_threshold"]), cands, loc, presence)


def insert_lead_in(bounds: np.ndarray, end_beat: int | None) -> np.ndarray:
    """Bounds with a first segment [0, end_beat); boundaries up to a bar after it are dropped."""
    b = np.asarray(bounds, dtype=np.int32)
    if end_beat is None or not 0 < int(end_beat) < int(b[-1]) - MIN_GAP_BEATS:
        return b
    k = int(end_beat)
    rest = [int(x) for x in b[1:-1] if int(x) >= k + MIN_GAP_BEATS]
    return np.asarray([0, k, *rest, int(b[-1])], dtype=np.int32)


def label_mask(labels: list[str], n_segments: int, has_lead_in: bool) -> np.ndarray:
    """Allowed (segment, label): LEAD_IN on the first segment exactly when there is a lead-in."""
    allowed = np.ones((n_segments, len(labels)), dtype=bool)
    col = labels.index(LEAD_IN_LABEL)
    allowed[:, col] = False
    if has_lead_in:
        allowed[0] = False
        allowed[0, col] = True
    return allowed
