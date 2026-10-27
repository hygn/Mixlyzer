"""The Variants (analyzer.search.variants, core.library.variants) on real releases: several versions of a few songs
(variants.json: each one's title, artist and release, to buy the same), each analyzed in full as the app does, its
fingerprints stored in a variants.db of the test's own, and the relations found held to what the files are:

  - each song's versions are one group, another song (on the same single) in none;
  - two purchases of one master (another product, another codec): the same mix;
  - the instrumental: an instrumental of the original and of every vocal version, it the instrumental side;
  - another vocal (a character's version, another duet): another vocal of the original;
  - the TV size: the same mix, its shorter edit;
  - a part version (one singer's part sung): related to the original, its kind not held.

Where the Variants fall short today, variants.json's needs_improvement names the pair and the check: reported as
needing improvement (a skip saying so), not a failure; once it holds, the test fails until the entry is taken off.

The audio is not here: point to the files as for test_analyzer.py (tests/ground_truth/audio.json, or
MIXLYZER_GT_AUDIO=<folder> of <id>.<ext> files). A file whose length differs from variants.json's by more than
DURATION_TOLERANCE_SEC is another edit: left out. A store may start its copy a little earlier or later or master it
otherwise (tests/README.md): the fingerprint follows the beats, so the start does not matter, but another master may
not be "the same mix" of this one.

Each file is analyzed once, about 20 s of CPU at a lower priority, then kept in cache/test_variants/ (git-ignored)
until the file or the fingerprint's method changes:

    MIXLYZER_GT_TESTS=1 uv run python -m unittest discover -s tests/ground_truth -t tests -p "test_variants.py"
    MIXLYZER_GT_REPORT=1   also print every relation found, its covers and agreement
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import fields
from itertools import combinations
from pathlib import Path

import numpy as np

from .test_analyzer import ENABLED, REPORT, audio_path, settings

HERE = Path(__file__).resolve().parent
CACHE = HERE.parents[1] / "cache" / "test_variants"
DURATION_TOLERANCE_SEC = 2.0


def catalog() -> dict:
    return json.loads((HERE / "variants.json").read_text(encoding="utf-8"))


def _cached_features(path: Path):
    """(beats, overview, length) of a file: the full analysis's (the shipped models, the default settings), kept
    in CACHE under the file's size and time and the fingerprint's method."""
    from analyzer.audio.wave_overview import WaveOverview
    from analyzer.search import variants as fpm
    stat = path.stat()
    stamp = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{fpm.METHOD}"
    cached = CACHE / (hashlib.sha1(stamp.encode("utf-8")).hexdigest() + ".npz")
    names = [f.name for f in fields(WaveOverview)]
    if cached.is_file():
        with np.load(cached, allow_pickle=False) as z:
            return (np.asarray(z["beats"]), WaveOverview(*(np.asarray(z[f"wave_{n}"]) for n in names)),
                    float(z["duration"]))
    from dataclasses import replace
    from analyzer.pipeline.track import analyze_track
    result = analyze_track(str(path), replace(settings(), music_index=False))
    CACHE.mkdir(parents=True, exist_ok=True)
    from analyzer.files import atomic_save_npz
    atomic_save_npz(cached, {"beats": np.asarray(result.beats, dtype=np.float64),
                             "duration": np.float64(result.duration_sec),
                             **{f"wave_{n}": np.asarray(getattr(result.wave_overview, n)) for n in names}})
    return np.asarray(result.beats, dtype=np.float64), result.wave_overview, float(result.duration_sec)


def _kind(pairs: dict, x: str, y: str) -> str | None:
    """The relation of x and y as found, from x's side: "same", "vocal", "x instrumental", "y instrumental", or
    None (not related)."""
    from analyzer.search import variants as fpm
    p = pairs.get((min(x, y), max(x, y)))
    if p is None:
        return None
    if p.kind in (fpm.A_INSTRUMENTAL, fpm.B_INSTRUMENTAL):
        instrumental = p.a if p.kind == fpm.A_INSTRUMENTAL else p.b
        return "x instrumental" if instrumental == x else "y instrumental"
    return p.kind


@unittest.skipUnless(ENABLED, "the tests on real songs are off (MIXLYZER_GT_TESTS=1)")
class VariantsOnReleasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from system import os_support
        os_support.lower_process_priority()
        from analyzer.search import variants as fpm
        from core.library import variants as store
        cls.data = catalog()
        cls.improve = {(e["check"], frozenset(e["pair"])): e for e in cls.data.get("needs_improvement", [])}
        cls.available: dict[str, dict] = {}       # id -> its entry, for the files of the listed length
        cls.left_out: list[str] = []
        entries = [v for s in cls.data["songs"] for v in s["versions"]] + cls.data["others"]
        features = {}
        for entry in entries:
            path = audio_path(entry["id"])
            if path is None:
                cls.left_out.append(f"{entry['id']}: no audio")
                continue
            beats, wave, duration = _cached_features(path)
            if abs(duration - entry["duration_sec"]) > DURATION_TOLERANCE_SEC:
                cls.left_out.append(f"{entry['id']}: {duration:.1f} s, not {entry['duration_sec']:.1f} s")
                continue
            cls.available[entry["id"]] = entry
            features[entry["id"]] = (beats, wave, duration)
        cls._tmp = tempfile.TemporaryDirectory()
        db = os.path.join(cls._tmp.name, "variants.db")
        con = store.open_writer(db)
        try:
            for uid, f in features.items():
                store.put(con, uid, fpm.track_fingerprints(*f))
        finally:
            con.close()
        found = store.pairs(db)
        cls.pairs = {(p.a, p.b): p for p in found}
        from core.search.variants import groups
        cls.groups = groups(found, set(cls.available))
        if REPORT:
            print()
            for line in cls.left_out:
                print(f"[Variants] left out {line}")
            for p in sorted(found, key=lambda p: (p.a, p.b)):
                print(f"[Variants] {p.a} ~ {p.b}: {p.kind}, cover {p.cover_a:.2f} / {p.cover_b:.2f}, "
                      f"full {p.full:.3f}")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _held(self, check: str, x: str, y: str, ok: bool, message: str) -> None:
        """A check of x ~ y. One on variants.json's needs_improvement list that falls short is reported as needing
        improvement (a skip naming it: the state recorded, nothing broken); one that now holds fails until its entry
        is taken off the list (as an expectedFailure's unexpected success)."""
        known = self.improve.get((check, frozenset((x, y))))
        if known is None:
            self.assertTrue(ok, message)
        elif ok:
            self.fail(f"{x} ~ {y} ({check}) now as expected: take it off variants.json's needs_improvement")
        else:
            self.skipTest(f"needs improvement (since {known['since']}): {message}")

    def _expect(self, x: str, y: str, wanted: set[str]) -> None:
        got = _kind(self.pairs, x, y)
        p = self.pairs.get((min(x, y), max(x, y)))
        detail = f" (cover {p.cover_a:.2f} / {p.cover_b:.2f}, full {p.full:.3f})" if p else ""
        self._held("kind", x, y, got in wanted, f"{x} ~ {y}: {got}{detail}, expected {sorted(map(str, wanted))}")

    def test_each_song_one_group(self):
        """Each song's versions are one group; another song in none."""
        for song in self.data["songs"]:
            versions = {v["id"] for v in song["versions"] if v["id"] in self.available}
            if len(versions) < 2:
                continue
            with self.subTest(song=song["song"]):
                holding = [g for g in self.groups if set(g) & versions]
                self.assertEqual([sorted(g) for g in holding], [sorted(versions)])
        for other in self.data["others"]:
            with self.subTest(other=other["id"]):
                self.assertFalse([g for g in self.groups if other["id"] in g])

    def test_versions_of_every_song_found_together(self):
        """Every two versions of one song are related (a group joined through a third is not enough)."""
        for song in self.data["songs"]:
            versions = self._available_versions(song)
            for x, y in combinations(sorted(versions), 2):
                with self.subTest(pair=f"{x} ~ {y}"):
                    self._held("related", x, y, _kind(self.pairs, x, y) is not None, f"{x} ~ {y} not related")

    def test_no_song_is_a_version_of_another(self):
        songs = [self._available_versions(s) for s in self.data["songs"]]
        songs += [{o["id"]} for o in self.data["others"] if o["id"] in self.available]
        for i, j in combinations(range(len(songs)), 2):
            for x in songs[i]:
                for y in songs[j]:
                    with self.subTest(pair=f"{x} ~ {y}"):
                        self.assertIsNone(_kind(self.pairs, x, y))

    def test_the_kinds(self):
        from analyzer.search import variants as fpm
        for song in self.data["songs"]:
            versions = {v["id"]: v for v in song["versions"] if v["id"] in self.available}
            if len(versions) < 2:
                continue
            original = next((i for i, v in versions.items() if v["role"] == "original"), None)
            instrumentals = [i for i, v in versions.items() if v["role"] == "instrumental"]
            for vid, v in versions.items():
                role = v["role"]
                if role == "purchase" and v["of"] in versions:
                    with self.subTest(pair=f"{vid} ~ {v['of']}"):
                        self._expect(vid, v["of"], {fpm.SAME_MIX_KIND})
                if original is None or vid == original:
                    continue
                with self.subTest(pair=f"{vid} ~ {original}"):
                    if role == "instrumental":
                        self._expect(vid, original, {"x instrumental"})
                    elif role == "vocal":
                        self._expect(vid, original, {fpm.OTHER_VOCAL})
                        for inst in instrumentals:
                            self._expect(inst, vid, {"x instrumental"})
                    elif role == "tv_size":
                        self._expect(vid, original, {fpm.SAME_MIX_KIND})
                        p = self.pairs.get((min(vid, original), max(vid, original)))
                        if p is not None:
                            own, other = (p.cover_a, p.cover_b) if p.a == vid else (p.cover_b, p.cover_a)
                            self.assertEqual(fpm.length(own, other), "edit",
                                             f"{vid}: cover {own:.2f}, the original's {other:.2f}")
                    elif role == "part":
                        self.assertIsNotNone(_kind(self.pairs, vid, original), f"{vid} ~ {original} not related")

    def _available_versions(self, song: dict) -> set[str]:
        return {v["id"] for v in song["versions"] if v["id"] in self.available}


if __name__ == "__main__":
    unittest.main()
