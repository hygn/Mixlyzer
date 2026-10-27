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
    return replace(base, models=models, music_index=False)


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
        path = audio_path(song["id"])
        if path is None:
            self.skipTest(f"no audio for {song['id']} ({song['title']} / {song['release']})")
        from analyzer.pipeline.track import analyze_track
        result = analyze_track(str(path), settings())
        if abs(result.duration_sec - song["duration_sec"]) > DURATION_TOLERANCE_SEC:
            self.skipTest(f"{song['id']}: {result.duration_sec:.1f} s, annotated {song['duration_sec']:.1f} s "
                          "(another edit of the song)")
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
