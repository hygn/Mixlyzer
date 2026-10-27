"""The full analysis of each song in songs.json against its ground truth: beats, downbeats, keys, phrase boundaries
and labels, each scored (metrics.py) and held to the song's floor (``expect``: what the analyzer reached when the
floor was set, less a margin; a better analyzer may raise it). Run with the shipped models and the default settings,
so the runner's own retrained models and settings play no part.

The audio is not here: get the release songs.json names (its title, artist and release) and point to the file:

    tests/ground_truth/audio.json        {"<id>": "<path>", ...}   (git-ignored)
    MIXLYZER_GT_AUDIO=<folder>           a folder holding <id>.<ext> files

A file of another length than the one annotated (another edit or version of the song) is skipped. About 20 s of CPU
a song, at a lower priority than the other programs:

    MIXLYZER_GT_TESTS=1 uv run python -m unittest discover -s tests/ground_truth -t tests -p "test_*.py"
    MIXLYZER_GT_REPORT=1   also print each song's scores (to set or raise a floor)
"""
from __future__ import annotations

import json
import os
import unittest
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import metrics

HERE = Path(__file__).resolve().parent
ENABLED = os.environ.get("MIXLYZER_GT_TESTS") == "1"
REPORT = os.environ.get("MIXLYZER_GT_REPORT") == "1"
DURATION_TOLERANCE_SEC = 1.0
SCORES = ("beat_f", "downbeat_f", "key", "phrase_bounds", "phrase_labels")


def songs() -> list[dict]:
    return json.loads((HERE / "songs.json").read_text(encoding="utf-8"))["songs"]


def audio_path(song_id: str) -> Path | None:
    folder = os.environ.get("MIXLYZER_GT_AUDIO")
    if folder:
        found = sorted(p for p in Path(folder).glob(f"{song_id}.*") if p.is_file())
        if found:
            return found[0]
    try:
        path = Path(json.loads((HERE / "audio.json").read_text(encoding="utf-8"))[song_id])
    except (OSError, ValueError, KeyError):
        return None
    return path if path.is_file() else None


@lru_cache(maxsize=1)
def settings():
    """The default analysis with the shipped models (never a retrained one)."""
    from dataclasses import fields, replace

    from analyzer.settings import ModelPaths
    from core.settings.config import default_cfg
    from core.workers.analysis_settings import analysis_settings
    from system.model_files import shipped_path
    base = analysis_settings(default_cfg())
    models = ModelPaths(**{f.name: str(shipped_path(getattr(base.models, f.name))) for f in fields(ModelPaths)})
    return replace(base, models=models, music_index=True)


_ANALYSES: dict[str, tuple[Path, object]] = {}


def analysis_of(song: dict):
    """(the audio's path, the analysis) of a song with audio of its annotated length, analyzed once a run (the search
    test reads the songs' analyses after them); None without."""
    if song["id"] not in _ANALYSES:
        path = audio_path(song["id"])
        if path is None:
            return None
        from analyzer.pipeline.track import analyze_track
        _ANALYSES[song["id"]] = (path, analyze_track(str(path), settings()))
    path, result = _ANALYSES[song["id"]]
    return (path, result) if abs(result.duration_sec - song["duration_sec"]) <= DURATION_TOLERANCE_SEC else None


def scores(song: dict, result) -> dict[str, float]:
    from analyzer.music.beat_geometry import downbeat_times
    gt_beats = np.asarray(song["beats"], dtype=float)
    est_beats = np.asarray(result.beats, dtype=float)
    shift = metrics.offset(gt_beats, est_beats)
    est = est_beats - shift
    gt_down = downbeat_times(gt_beats, np.asarray(song["tempo_segments"], dtype=float))
    est_down = downbeat_times(est_beats, result.tempo_segments)
    est_down = (est_down - shift) if est_down is not None else []
    duration = float(song["duration_sec"])
    gt_bounds = [p[0] for p in song["phrases"][1:]]
    est_bounds = [p.start - shift for p in result.phrases[1:]]
    return {
        "offset_ms": round(shift * 1000.0, 1),
        "beat_f": metrics.f_measure(gt_beats, est),
        "downbeat_f": metrics.f_measure(gt_down if gt_down is not None else [], est_down),
        # keys only from a human source ("keys": its segments and where they come from), never a library's
        **({"key": metrics.key_agreement(song["keys"]["segments"], result.key_segments, duration, shift)}
           if "keys" in song else {}),
        "phrase_bounds": metrics.boundary_f(gt_bounds, est_bounds, gt_beats),
        "phrase_labels": metrics.label_agreement(song["phrases"], [(p.start, p.end, p.label) for p in result.phrases],
                                                 duration, shift),
    }


def _song_test(song: dict):
    def test(self):
        if audio_path(song["id"]) is None:
            self.skipTest(f"no audio for {song['id']} ({song['title']} / {song['release']})")
        done = analysis_of(song)
        if done is None:
            self.skipTest(f"{song['id']}: another length than annotated, {song['duration_sec']:.1f} s "
                          "(another edit of the song)")
        _path, result = done
        self.assertIsNotNone(result.music_index)                      # Search by Music's entry made with it
        got = scores(song, result)
        if REPORT:
            print(f"\n[GT] {song['id']}: " + json.dumps({k: round(v, 4) for k, v in got.items()}))
        for name in SCORES:
            floor = song.get("expect", {}).get(name)
            if floor is not None and name in got:
                with self.subTest(score=name):
                    self.assertGreaterEqual(got[name], floor, f"{song['id']} {name}: {got[name]:.3f} < {floor}")
    test.__doc__ = f"{song['title']} ({song['artist']})"
    return test


@unittest.skipUnless(ENABLED, "the analyzer's ground truth is off (MIXLYZER_GT_TESTS=1)")
class GroundTruthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from system import os_support
        os_support.lower_process_priority()


for _song in songs():
    setattr(GroundTruthTests, f"test_{_song['id'].replace('-', '_')}", _song_test(_song))


QUERY_SEC = 12.0


def _queries(path: Path, duration: float, seed: int) -> dict[str, np.ndarray]:
    """A song's excerpts as Search by Music hears them (mono at its rate): as it is, under noise at 10 dB SNR and
    half the level, and through a phone's band (300 Hz - 3.4 kHz) with a little noise."""
    from scipy.signal import butter, sosfiltfilt

    from analyzer.audio.load import fast_load
    from analyzer.search.music import melody
    y = fast_load(str(path), melody.SR)
    start = int(0.4 * duration * melody.SR)
    clip = y[start:start + int(QUERY_SEC * melody.SR)].astype(np.float64)
    rng = np.random.default_rng(seed)
    rms = float(np.sqrt(np.mean(clip ** 2))) or 1e-9
    noisy = 0.5 * (clip + rng.standard_normal(clip.size) * rms / np.sqrt(10.0))
    phone = sosfiltfilt(butter(4, [300, 3400], "bandpass", fs=melody.SR, output="sos"), clip)
    phone = phone + rng.standard_normal(clip.size) * rms * 0.03
    return {name: q.astype(np.float32) for name, q in (("clean", clip), ("noisy", noisy), ("phone", phone))}


HUM_SEC, HUM_SHIFT = 15.0, 2.0 - 12.0   # a hum: two semitones up, an octave down


def _hum(entry) -> tuple[np.ndarray, float]:
    """A sung melody's stand-in: a sine following the song's centre line (its lead vocal) at 0.1 s frames, over the
    window of HUM_SEC it is voiced most, moved by HUM_SHIFT; and where the window starts (s)."""
    from analyzer.search.music import melody
    step = 0.1
    line = entry.lines[step][0].astype(np.float64)
    voiced = line != melody.UNV
    n = int(HUM_SEC / step)
    counts = np.convolve(voiced, np.ones(n), "valid")
    at = int(np.argmax(counts))
    midi = line[at:at + n] / melody.STEPS + melody.BASE + HUM_SHIFT
    on = voiced[at:at + n]
    per = int(step * melody.SR)
    hz = np.repeat(np.where(on, 440.0 * 2.0 ** ((midi - 69.0) / 12.0), 0.0), per)
    gain = np.convolve(np.repeat(on.astype(np.float64), per), np.ones(441) / 441, "same")   # 20 ms fades
    phase = 2 * np.pi * np.cumsum(hz) / melody.SR
    return (0.3 * gain * np.sin(phase)).astype(np.float32), at * step


@unittest.skipUnless(ENABLED, "the analyzer's ground truth is off (MIXLYZER_GT_TESTS=1)")
class SearchByMusicTests(unittest.TestCase):
    """Search by Music over an index of the songs' own entries (as the analysis made them, stored as the library's
    owner stores them): each song's excerpt finds that song first, at the place it was cut from."""

    def test_each_song_found_by_its_excerpts(self):
        import tempfile

        from analyzer.search.music import melody
        from analyzer.search.music.search import search_audio
        from core.library import music_index
        done = {s["id"]: (s, analysis_of(s)) for s in songs() if audio_path(s["id"]) is not None}
        done = {k: v for k, v in done.items() if v[1] is not None}
        if len(done) < 2:
            self.skipTest("fewer than two songs with audio")
        with tempfile.TemporaryDirectory() as tmp:
            path = music_index.path_for(tmp)
            con = music_index.open_writer(path)
            with con:
                for sid, (_song, (_p, result)) in done.items():
                    music_index.put(con, sid, 0, 0.0, result.music_index)
            con.close()
            landmarks, melodies = music_index.open_search(path)
            try:
                for k, (sid, (song, (audio, result))) in enumerate(sorted(done.items())):
                    at = 0.4 * song["duration_sec"]
                    for name, q in _queries(audio, song["duration_sec"], k).items():
                        with self.subTest(song=sid, query=name):
                            found = search_audio(q, melody.SR, melodies, landmarks)
                            self.assertTrue(found.matches, "nothing found")
                            best = found.matches[0]
                            self.assertEqual((best.uid, found.kind), (sid, "recording"))
                            self.assertLess(abs(best.start_sec - at), 1.0)   # where the excerpt was cut from
                    # hummed in another key and octave: no landmark of the recording, the melody matcher's
                    hum, hum_at = _hum(result.music_index)
                    with self.subTest(song=sid, query="hum"):
                        found = search_audio(hum, melody.SR, melodies, landmarks)
                        self.assertEqual(found.kind, "melody")
                        best = found.matches[0]
                        self.assertEqual(best.uid, sid)
                        folded = (best.transposition + HUM_SHIFT + 6.0) % 12.0 - 6.0   # the song above the hum
                        self.assertLess(abs(folded), 0.5, best.transposition)
                        self.assertLess(abs(best.start_sec - hum_at), 3.0)
            finally:
                landmarks.close()


class FixtureTests(unittest.TestCase):
    """songs.json itself (always run)."""

    def test_songs_well_formed(self):
        from analyzer.music.tempo_rows import rebuild_grid_from_segments
        from core.marks.phrases import PHRASE_LABELS
        ids = [s["id"] for s in songs()]
        self.assertEqual(len(ids), len(set(ids)))
        for song in songs():
            with self.subTest(song=song["id"]):
                self.assertTrue(song["title"] and song["artist"])
                self.assertNotIn("uid", song)
                self.assertNotIn("path", song)
                beats = np.asarray(song["beats"])
                self.assertTrue(np.all(np.diff(beats) > 0))
                self.assertLessEqual(beats[-1], song["duration_sec"] + 0.15)
                rebuilt = rebuild_grid_from_segments(np.asarray(song["tempo_segments"]))
                self.assertGreater(metrics.f_measure(beats, rebuilt, 0.002), 0.99)
                self.assertTrue({p[2] for p in song["phrases"]} <= set(PHRASE_LABELS))
                self.assertTrue(set(song.get("expect", {})) <= set(SCORES))
                self.assertNotIn("key_segments", song)                   # keys: a human source's alone
                if "keys" in song:
                    self.assertTrue(song["keys"].get("source"), "human keys name their source")
                    self.assertTrue(song["keys"].get("segments"))
                else:
                    self.assertNotIn("key", song.get("expect", {}))
                self.assertTrue(all(0.0 <= v <= 1.0 for v in song.get("expect", {}).values()))


if __name__ == "__main__":
    unittest.main()
