"""Every track's stored marks are what the app reads them as: beats on their tempo rows, rows covering the track,
keys and phrases ordered within it, CUE points and JumpCUEs in it, their letters slots; the library's index rows the
stored analysis's own; a save of the marks as they are changes nothing."""
import math
import unittest

import numpy as np

from analyzer.music.tempo_rows import MIN_BEAT_GAP_SEC, rebuild_grid_from_segments, with_end_bpm
from core.analysis_library.stored import StoredAnalysis, TrackEdits, annotation_of, apply_edits
from core.library.segments import build_bpm_segments, build_key_segments
from core.marks import jump_cues
from core.marks.phrases import PHRASE_LABELS, normalize_phrase_segments

from . import _library as lib

END_TOLERANCE_SEC = 0.05     # a mark this far past the stored duration is still in the track (the decode's rounding)
GRID_TOLERANCE_SEC = 0.002   # the stored beats against the ones their rows give
ROW_GAP_SEC = 0.002          # rows touch: the editor's grid moves work in 1 ms steps
PHRASE_PAST_END_SEC = 1.0    # the last phrase may end on the bar after the track's end (read cut at the end)


def analyses():
    for uid, arrays in lib.stored().items():
        yield uid, StoredAnalysis.from_arrays(arrays)


@lib.needs_library
class BeatgridTests(unittest.TestCase):
    def test_beats(self):
        problems = []
        for uid, a in analyses():
            beats = np.asarray(a.beats, dtype=float)
            if beats.size == 0:
                continue
            if not np.all(np.isfinite(beats)):
                problems.append((uid, "a beat not finite"))
            elif beats.size > 1 and float(np.min(np.diff(beats))) < MIN_BEAT_GAP_SEC:
                problems.append((uid, f"two beats {float(np.min(np.diff(beats))) * 1000:.1f} ms apart"))
            elif beats[0] < -1e-6 or beats[-1] > a.duration_sec + END_TOLERANCE_SEC:
                problems.append((uid, "a beat outside the track"))
        lib.report(self, problems, "beats")

    def test_rows(self):
        problems = []
        for uid, a in analyses():
            rows = with_end_bpm(a.tempo_segments)
            if rows.size == 0:
                continue
            if not np.all(np.isfinite(rows)):
                problems.append((uid, "a row not finite"))
            elif np.any(rows[:, 1] <= rows[:, 0]) or np.any(rows[:, 2] <= 0) or np.any(rows[:, 5] <= 0):
                problems.append((uid, "a row without length or tempo"))
            elif np.any(np.abs(rows[1:, 0] - rows[:-1, 1]) > ROW_GAP_SEC):
                problems.append((uid, "rows not touching"))
            elif np.any((rows[:, 4] < 1) | (rows[:, 4] > 8) | (rows[:, 4] != np.round(rows[:, 4]))):
                problems.append((uid, "a bar length not 1 - 8 beats"))
            elif a.duration_sec and abs(rows[-1, 1] - a.duration_sec) > END_TOLERANCE_SEC:
                problems.append((uid, f"the rows end {rows[-1, 1] - a.duration_sec:+.3f} s from the track's end"))
        lib.report(self, problems, "tempo rows")

    def test_beats_are_their_rows(self):
        """Away from the rows' junctions (where the tracker hands one grid over to the next, a beat may sit between
        the two rows' own) every stored beat is its row's, and every row's beat is stored: an edit rebuilds the
        beats from the rows, so a beat only the stored array has is lost at the first edit."""
        problems = []
        for uid, a in analyses():
            beats = np.asarray(a.beats, dtype=float)
            rows = with_end_bpm(a.tempo_segments)
            if beats.size < 2 or rows.size == 0:
                continue
            rebuilt = rebuild_grid_from_segments(rows)
            period = 60.0 / rows[:, 2].max()
            junctions = rows[1:, 0]
            away = lambda t: np.min(np.abs(junctions[:, None] - t[None, :]), axis=0) > 2 * period \
                if junctions.size else np.ones(t.size, bool)   # noqa: E731
            stored_only = beats[away(beats)]
            rows_only = rebuilt[away(rebuilt)]
            far = [float(np.min(np.abs(rebuilt - t))) for t in stored_only]
            missing = [float(np.min(np.abs(beats - t))) for t in rows_only]
            worst = max(far + [0.0])
            if worst > GRID_TOLERANCE_SEC:
                where = "past the rows' end" if stored_only[int(np.argmax(far))] > rows[-1, 1] else "inside a row"
                problems.append((uid, f"a stored beat {worst * 1000:.1f} ms from the rows' beats, {where}"))
            elif max(missing + [0.0]) > GRID_TOLERANCE_SEC:
                problems.append((uid, f"a row's beat not stored ({max(missing) * 1000:.1f} ms from any)"))
        lib.report(self, problems, "beats against their tempo rows")


@lib.needs_library
class OtherMarkTests(unittest.TestCase):
    def test_keys(self):
        problems = []
        for uid, a in analyses():
            for which, keys in (("keys", a.key_segments), ("automatic keys", a.key_segments_auto)):
                k = np.asarray(keys if keys is not None else np.empty((0, 4)), dtype=float).reshape(-1, 4)
                if k.size == 0:
                    continue
                pitch, minor, start, end = k.T
                if not np.all(np.isfinite(k)):
                    problems.append((uid, f"{which}: not finite"))
                elif np.any((pitch < 0) | (pitch > 11) | (pitch != np.round(pitch))) or np.any(~np.isin(minor, (0, 1))):
                    problems.append((uid, f"{which}: not a key"))
                elif np.any(end <= start) or np.any(start[1:] < end[:-1] - 1e-6):
                    problems.append((uid, f"{which}: not ordered"))
                elif start[0] < -1e-6 or end[-1] > a.duration_sec + END_TOLERANCE_SEC:
                    problems.append((uid, f"{which}: outside the track"))
        lib.report(self, problems, "keys")

    def test_phrases(self):
        problems = []
        for uid, a in analyses():
            rows = list(a.phrases)
            if not rows:
                continue
            labels = {r.label for r in rows} - set(PHRASE_LABELS)
            if labels:
                problems.append((uid, f"{len(labels)} labels not the app's"))
            again = normalize_phrase_segments(rows)
            if len(again) != len(rows) or any(abs(x.start - y.start) > 1e-4 or abs(x.end - y.end) > 1e-4
                                               for x, y in zip(again, rows)):
                problems.append((uid, "not as the editor stores them (ordered, apart, no slivers)"))
            elif rows[-1].start >= a.duration_sec or rows[-1].end > a.duration_sec + PHRASE_PAST_END_SEC:
                problems.append((uid, f"a phrase {rows[-1].end - a.duration_sec:+.2f} s past the track's end"))
        lib.report(self, problems, "phrases")

    def test_cue_points(self):
        problems = []
        for uid, a in analyses():
            times = [c.time_sec for c in a.cue_points]
            if not all(math.isfinite(t) and -1e-6 <= t <= a.duration_sec + END_TOLERANCE_SEC for t in times):
                problems.append((uid, "a CUE point outside the track"))
            elif times != sorted(times):
                problems.append((uid, "CUE points not in time order"))
        lib.report(self, problems, "CUE points")

    def test_jump_cues(self):
        problems = []
        for uid, a in analyses():
            for which, groups in (("JumpCUEs", a.jump_cues), ("automatic JumpCUEs", a.jump_cues_auto or ())):
                groups = [g for g in groups if g.cues]
                if which == "JumpCUEs" and not jump_cues.has_slot_letters(groups):
                    problems.append((uid, f"{which}: letters not slots A-Z (each once)"))
                cues = [c for g in groups for c in g.cues]
                if any(not (-1e-6 <= c.point <= a.duration_sec + END_TOLERANCE_SEC) for c in cues):
                    problems.append((uid, f"{which}: a point outside the track"))
        lib.report(self, problems, "JumpCUEs")


@lib.needs_library
class IndexRowTests(unittest.TestCase):
    """The library.db segment tables (the transition search's index) against the stored analysis they come from."""

    def rows(self, table, columns):
        with lib.connect() as db:
            out = {}
            for row in db.execute(f"SELECT track_uid, {', '.join(columns)} FROM {table} ORDER BY track_uid, seq_index"):  # noqa: S608
                out.setdefault(row[0], []).append(row[1:])
            return out

    def test_tempo_index(self):
        stored = self.rows("track_bpm_segments", ("start_sec", "end_sec", "bpm", "bpm_end", "time_signature"))
        problems = []
        for uid, a in analyses():
            want = [(r["start_sec"], r["end_sec"], r["bpm"], r["bpm_end"], r["time_signature"])
                    for r in build_bpm_segments(a.tempo_segments)]
            got = stored.get(uid, [])
            if len(want) != len(got) or any(
                    any(abs((x or 0) - (y or 0)) > 1e-4 for x, y in zip(w, g)) or (w[3] is None) != (g[3] is None)
                    for w, g in zip(want, got)):
                problems.append((uid, f"{len(got)} index rows, {len(want)} from the stored analysis"))
        lib.report(self, problems, "tempo index rows not the stored analysis's")

    def test_key_index(self):
        stored = self.rows("track_key_segments", ("start_sec", "end_sec", "key_value"))
        problems = []
        for uid, a in analyses():
            want = [(r["start_sec"], r["end_sec"], r["key_value"]) for r in build_key_segments(a.key_segments)]
            got = stored.get(uid, [])
            if len(want) != len(got) or any(any(abs(x - y) > 1e-4 for x, y in zip(w, g)) for w, g in zip(want, got)):
                problems.append((uid, f"{len(got)} index rows, {len(want)} from the stored analysis"))
        lib.report(self, problems, "key index rows not the stored analysis's")


@lib.needs_library
class SaveTests(unittest.TestCase):
    def test_saving_the_marks_as_they_are_changes_nothing(self):
        """The editor's save of an unchanged track (in memory: nothing is written)."""
        problems = []
        for uid, arrays in lib.stored().items():
            a = StoredAnalysis.from_arrays(arrays)
            edits = TrackEdits(beats=a.beats, tempo_segments=a.tempo_segments, key_segments=a.key_segments,
                               phrases=a.phrases, jump_cues=tuple(g for g in a.jump_cues if g.cues) or None,
                               cue_points=a.cue_points)
            saved = dict(arrays)
            try:
                apply_edits(saved, edits)
            except Exception as exc:
                problems.append((uid, f"apply_edits: {type(exc).__name__}"))
                continue
            b = StoredAnalysis.from_arrays(saved)
            changed = [f for f in ("beats", "tempo_segments", "key_segments")
                       if not np.array_equal(np.asarray(getattr(a, f), float), np.asarray(getattr(b, f), float))]
            if b.phrases != a.phrases:
                changed.append("phrases")
            if b.cue_points != a.cue_points:
                changed.append("cue_points")
            if tuple(g for g in b.jump_cues if g.cues) != tuple(g for g in a.jump_cues if g.cues):
                changed.append("jump_cues")
            if changed:
                problems.append((uid, "changed: " + ", ".join(changed)))
        lib.report(self, problems, "a save without edits changed the stored marks")

    def test_reoptimize_reads_every_track(self):
        problems = []
        for uid, arrays in lib.stored().items():
            try:
                annotation_of(arrays)
            except Exception as exc:
                problems.append((uid, f"annotation_of: {type(exc).__name__}"))
        lib.report(self, problems, "Reoptimize's annotation")


if __name__ == "__main__":
    unittest.main()
