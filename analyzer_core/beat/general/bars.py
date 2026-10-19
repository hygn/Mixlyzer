"""Bars on the decided beats.

Bar-supported beat phase: of a steady section's grid and the same grid half a beat later, keep the one
whose beats carry a bar structure -- the downbeat activation standing out on every 2nd, 3rd or 4th beat
(musical: beats are nested in bars, bass notes and harmony changes land on the bar line, which lies on a
beat, never between two; the off-beat grid has no beat the bar lines fall on; bars of 2 beats are a
half-time reading -- a BPM range below the music's tempo -- whose beats belong on beats 1 and 3, not on the
backbeat). Only when the moved grid's bar support is in the normal range of right grids (5th percentile of
the library's correct sections).

Bars: a Viterbi over (meter 3 or 4, position in the bar) on the mean log-odds of two downbeat observers
that hear different cues -- the frame TCN's downbeat activation and the beat-level downbeat GBM (harmony,
bass and neighbour-beat differences). An odd bar (restart) costs CB inside a section and nothing where a
new grid section starts (the DJ grid itself restarts there).

to_rows: the production tempo rows ([start, end, bpm, inizio = a downbeat, ts, bpm_end]) for the beats and
downbeats, split wherever the bar length changes, so that ``core.beat_geometry.downbeat_beat_indices``
gives exactly these downbeats.
"""
from __future__ import annotations

import numpy as np

from core.beat_geometry import downbeat_beat_indices
from core.tempo_rows import rebuild_grid_from_segments, split_row

from .decode import is_ramp

MIN_BEATS = 32
START_MARGIN = 1e-3                 # a row starts this long before its first beat (float32 storage)
MIN_SUPPORT = 0.33
CB = 160.0


def bar_support(p):
    best = -np.inf
    for m in (2, 3, 4):
        for ph in range(m):
            on = p[ph::m]
            off = np.delete(p, np.arange(ph, p.size, m))
            if on.size and off.size:
                best = max(best, float(on.mean() - off.mean()))
    return best


def _at(act, t0, hop, t):
    i = np.clip(np.round((np.asarray(t) - t0) / hop).astype(int), 0, act.size - 1)
    return np.array([act[max(0, j - 2):j + 3].max() for j in i])


def bar_check(rows, down, t0, hop):
    rows = np.asarray(rows, float).copy()
    for r in rows:
        if is_ramp(r):
            continue
        P = 60.0 / r[2]
        k = np.arange(np.ceil((r[0] - r[3]) / P - 1e-9), np.floor((r[1] - r[3]) / P - 1e-9) + 1)
        g = r[3] + P * k
        if g.size < MIN_BEATS:
            continue
        half = bar_support(_at(down, t0, hop, g + 0.5 * P))
        if half > bar_support(_at(down, t0, hop, g)) and half >= MIN_SUPPORT:
            r[3] += 0.5 * P
    return rows


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def context(Xs, shifts=(-4, -3, -2, -1, 1, 2, 3, 4)):
    """The beat-level downbeat features with their differences to the neighbouring beats."""
    cols = [Xs]
    for s in shifts:
        sh = np.roll(Xs, -s, axis=0)
        if s > 0:
            sh[-s:] = 0
        else:
            sh[:-s] = 0
        cols.append(Xs - sh)
    return np.concatenate(cols, axis=1)


def section_starts(beats, rows):
    cheap = np.zeros(len(beats), bool)
    for r in np.asarray(rows, float)[1:]:
        i = int(np.searchsorted(beats, r[0] - 1e-3))
        if i < cheap.size:
            cheap[i] = True
    return cheap


CTX_FORMAT = 'mixlyzer.downbeat_ctx.v1'
CTX_PATH = 'assets/weights/downbeat_ctx_hgb.npz'


def load_ctx_model(path: str = CTX_PATH):
    """The contextual downbeat GBM (beat-level downbeat features and their neighbour differences)."""
    from analyzer_core.cue_and_phrase.phrase_analyzer import _load_numpy_hgb
    from core.resource_paths import resource_path
    with np.load(resource_path(path), allow_pickle=False) as z:
        if str(np.asarray(z['format']).item()) != CTX_FORMAT:
            raise ValueError(f'Contextual downbeat model format mismatch: {path}')
        return _load_numpy_hgb(z, 'downbeat_ctx')


def export_ctx_model(classifier, path, training: dict | None = None) -> None:
    """Write a fitted HistGradientBoostingClassifier as the runtime's NumPy artifact."""
    import json
    from pathlib import Path
    from analyzer_core.cue_and_phrase.phrase_analyzer import _hgb_to_npz_payload
    payload = {'format': np.asarray(CTX_FORMAT),
               'training_json': np.asarray(json.dumps(training or {}, ensure_ascii=False, sort_keys=True))}
    _hgb_to_npz_payload('downbeat_ctx', classifier, payload)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def bars_viterbi(pd, cost=CB, cheap=None, cost_at_start=0.0):
    """Bar positions: states (meter 3 or 4, position in the bar); a bar goes on for free, a restart (odd bar,
    meter change) costs ``cost`` (``cost_at_start`` where ``cheap``, a new grid section). True at downbeats."""
    states = [(m, q) for m in (3, 4) for q in range(m)]
    e = np.stack([_logit(pd) if q == 0 else _logit(1 - pd) for m, q in states], axis=1)
    ns = len(states)
    nxt = np.array([states.index((m, (q + 1) % m)) for m, q in states])      # each state has one successor
    prev = np.empty(ns, int)
    prev[nxt] = np.arange(ns)                                                   # ... and one predecessor
    V = e[0].copy()
    back = np.zeros((pd.size, ns), int)
    for i in range(1, pd.size):
        cand = V[prev]
        src = prev.copy()
        g = int(np.argmax(V))
        c = cost_at_start if cheap is not None and cheap[i] else cost
        jump = V[g] - c > cand
        cand = np.where(jump, V[g] - c, cand)
        src = np.where(jump, g, src)
        back[i] = src
        V = cand + e[i]
    st = np.zeros(pd.size, int)
    st[-1] = int(np.argmax(V))
    for i in range(pd.size - 1, 0, -1):
        st[i - 1] = back[i, st[i]]
    return np.array([states[x][1] == 0 for x in st])


def downbeats(beats, rows, down, t0, hop, frames, ctx_model):
    """Downbeat times among the beats: the bars Viterbi on the fused observers."""
    from analyzer_core.beat.downbeat_offset import _downbeat_feature_matrix, _robust_standardize_columns
    b = np.sort(np.asarray(beats, float))
    p = _at(down, t0, hop, b)
    if b.size >= 8 and ctx_model is not None:
        _, F = _downbeat_feature_matrix(frames, b, beat_phase_bins=8, harmonic_phase_bins=8)
        pg = ctx_model.predict_proba(context(_robust_standardize_columns(F)))[:, 1]
        pgb = np.full(b.size, 0.25)
        n = min(b.size, pg.size)
        pgb[:n] = pg[:n]
        p = 1.0 / (1.0 + np.exp(-0.5 * (_logit(p) + _logit(pgb))))
    return b[bars_viterbi(p, CB, section_starts(b, rows), 0.0)]


def _neighbour_beat(row, beat_time, k):
    """The beat k (+1 or -1) steps from ``beat_time`` on the row's own grid (a ramp's tempo line extended)."""
    if not is_ramp(row):
        return float(beat_time + k * 60.0 / row[2])
    start, b0, b1 = float(row[0]), float(row[2]), float(row[5])
    slope = (b1 - b0) / (float(row[1]) - start)

    def phase(x):
        return (b0 * x + 0.5 * slope * x * x) / 60.0
    target = phase(float(beat_time) - start) + k
    disc = b0 * b0 + 2.0 * slope * 60.0 * target
    return float(start + (-b0 + np.sqrt(max(disc, 0.0))) / slope)


def to_rows(rows, beats, downbeat_times):
    """Production tempo rows whose (inizio, ts) give exactly these downbeats: every research row is cut where
    a run of equal-length bars begins; each piece's inizio is a downbeat of its run on the piece's own grid
    (extrapolated when the piece holds no bar line) and ts the run's bar length."""
    b = np.sort(np.asarray(beats, float))
    di = np.flatnonzero(np.isin(np.round(b, 6), np.round(np.asarray(downbeat_times, float), 6)))
    if di.size == 0:
        return np.asarray([np.r_[np.asarray(r, float)[:5], r[5] if len(r) > 5 else r[2]] for r in rows])
    # bar k = [d_k, d_k+1); length of a partial first / last bar = the neighbouring full bar's
    lengths = np.diff(np.r_[di, b.size]).astype(int)
    if lengths.size > 1 and lengths[-1] < lengths[-2]:
        lengths[-1] = lengths[-2]
    first_len = int(lengths[0])

    def bar_of(j):
        """(downbeat index of the bar holding beat j, its length)."""
        k = int(np.searchsorted(di, j, side='right')) - 1
        if k < 0:
            return int(di[0]), first_len                    # pickup before the first bar line
        return int(di[k]), int(lengths[k])

    out = []
    for r in np.asarray(rows, float):
        r = np.r_[r[:5], r[5] if r.size > 5 else r[2]]
        lo = int(np.searchsorted(b, r[0], side='left'))
        hi = int(np.searchsorted(b, r[1], side='left'))
        if hi <= lo:
            out.append(r)
            continue
        # runs of equal bar length inside [lo, hi)
        runs = []
        j = lo
        while j < hi:
            d0, L = bar_of(j)
            k = int(np.searchsorted(di, j, side='right'))       # next bar line after j
            end = hi
            while k < di.size and di[k] < hi:
                if int(lengths[k]) != L:
                    end = int(di[k])
                    break
                k += 1
            runs.append((j, end, d0, L))
            j = end
        pieces = [r]
        for start, _, _, _ in runs[1:]:
            last = pieces.pop()
            left, right = split_row(last, float(b[start]) - 1e-4)
            pieces += [np.asarray(left, float), np.asarray(right, float)]
        for piece, (start, end, d0, L) in zip(pieces, runs):
            piece = np.asarray(piece, float).copy()
            inside = di[(di >= start) & (di < end)]
            if inside.size:
                piece[3] = b[inside[0]]
                piece[4] = float(L)
            else:
                # no bar line inside (a bridge or a short ramp): the reference is the beat just after the piece
                # (or just before it at the track end), one beat off the piece's own grid, and the bar is longer
                # than the piece, so no downbeat falls inside
                n_in = end - start
                piece[4] = float(L if L > n_in else n_in + 1)
                if end < b.size:
                    piece[3] = _neighbour_beat(piece, b[end - 1], +1)
                else:
                    piece[3] = _neighbour_beat(piece, b[start], -1)
            out.append(piece)
    return _join_needless_splits(np.asarray(out), b, di)


def _join_needless_splits(rows, beats, down_idx):
    """Neighbouring steady rows on one grid joined again wherever one row's bar pattern already gives the
    other's downbeats (e.g. a bar starting on the last beat before a tempo change, or a short bridge back to
    the same tempo, needs no row of its own), and a row shorter than a beat handed to the next row when its
    beat is on that row's grid too (a hand-over beat that starts a bar). Each change is kept only when the
    production rules rebuild the same beats and downbeats."""
    rows = [np.asarray(r, float) for r in rows]
    want = np.asarray(down_idx, int)

    def same(cand):
        bb = rebuild_grid_from_segments(np.asarray(cand))
        bb = bb[bb >= 0]
        if bb.size != beats.size or (bb.size and np.max(np.abs(bb - beats)) > 1e-6):
            return False
        got = downbeat_beat_indices(beats, np.asarray(cand))
        return got.size == want.size and bool(np.all(got == want))

    i = 0
    while i + 1 < len(rows):
        p, q = rows[i], rows[i + 1]
        period = 60.0 / p[2]
        on_grid = abs((q[3] - p[3]) / period - round((q[3] - p[3]) / period)) < 1e-6
        if not is_ramp(p) and not is_ramp(q) and abs(p[2] - q[2]) < 1e-9 and on_grid:
            forward, back = p.copy(), q.copy()
            forward[1] = q[1]                       # p's bar pattern over both
            back[0] = p[0]                          # q's bar pattern over both
            joined = next((c for c in (forward, back) if same(rows[:i] + [c] + rows[i + 2:])), None)
            if joined is not None:
                rows = rows[:i] + [joined] + rows[i + 2:]
                continue
        if i + 2 <= len(rows) - 1 and q[1] - q[0] < 60.0 / q[2]:
            # q is shorter than a beat: the next row takes its beat (start moved in front of it)
            r = rows[i + 2]
            for inizio in (r[3], q[3]):
                taken = r.copy()
                taken[0] = q[0]
                taken[3] = inizio
                cand = rows[:i + 1] + [taken] + rows[i + 3:]
                if not is_ramp(r) and same(cand):
                    rows = cand
                    break
            else:
                i += 1
            continue
        i += 1
    # Every steady row starts a clear START_MARGIN before its first beat: the rows are stored as float32
    # (0.1 ms resolution at half an hour), and a start on its first beat could otherwise pass it.
    for i in range(1, len(rows)):
        if is_ramp(rows[i]) or is_ramp(rows[i - 1]):
            continue
        j = int(np.searchsorted(beats, rows[i][0] - 1e-9))
        if j < beats.size and beats[j] - rows[i][0] < START_MARGIN and beats[j] - START_MARGIN > rows[i - 1][0]:
            cand = [r.copy() for r in rows]
            cand[i][0] = cand[i - 1][1] = beats[j] - START_MARGIN
            if same(cand):
                rows = cand
    return np.asarray(rows)
