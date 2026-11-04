"""training.phrase_stack_optimizer: the beats' labels and boundaries from the annotated phrases, the DS-TCN's torch
module against the shipped arrays and the analysis's NumPy inference of its export, the rare songs repeated, the song
cues' cache, and a small run on synthetic tracks (the phrase features and the GBM members stood in for: the shipped
model's own member) that writes a stack the analysis's loader reads. The torch parts need the training group
(skipped without it)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from analyzer.phrase import dstcn as DS
from analyzer.phrase.phrase_analyzer import load_two_stage_model
from training import phrase_parameter_optimizer as PPO
from training import phrase_stack_optimizer as O
from training.text import TrainingError

from unit.training_tests import training_fixtures as F

RNG_SEED = 20261105
SHIPPED = F.shipped("phrase_analyzer.npz")
HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = unittest.skipUnless(HAS_TORCH, "torch is the training group's (uv sync --group training)")
CYCLE = ("INTRO", "VERSE", "CHORUS", "INTERLUDE", "VERSE", "CHORUS", "OUTRO")


class BeatLabelTests(unittest.TestCase):
    def test_each_phrase_from_its_beat_to_the_next(self):
        bt = np.arange(20) * 0.5
        y, bnd = O.beat_labels(bt, np.array([1.0, 3.98, 7.0]), ["INTRO", "VERSE", "UNKNOWN"])
        self.assertEqual(y[:2].tolist(), [-1, -1])                       # before the first phrase
        self.assertEqual(set(y[2:8].tolist()), {DS.LABELS.index("INTRO")})
        self.assertEqual(set(y[8:14].tolist()), {DS.LABELS.index("VERSE")})  # 50 ms early still its beat
        self.assertEqual(set(y[14:].tolist()), {-1})                     # a label the net has no class for
        self.assertEqual(np.flatnonzero(bnd).tolist(), [2, 8, 14])

    def test_a_phrase_from_the_first_beat_is_no_boundary(self):
        _, bnd = O.beat_labels(np.arange(8) * 0.5, np.array([0.0, 2.0]), ["INTRO", "VERSE"])
        self.assertEqual(np.flatnonzero(bnd).tolist(), [4])


@needs_torch
class NetworkTests(unittest.TestCase):
    def test_state_dict_is_the_shipped_arrays(self):
        with np.load(SHIPPED, allow_pickle=False) as z:
            shipped = {k[len("dstcn.w."):]: z[k].shape for k in z.files if k.startswith("dstcn.w.")}
            d_in = int(z["dstcn.mu"].size) + DS.POSITIONS
        net = O.network(d_in, O.CHANNELS)
        self.assertEqual({k: tuple(v.shape) for k, v in net.state_dict().items()}, shipped)

    def test_exported_arrays_are_the_analysiss_network(self):
        import torch
        torch.manual_seed(RNG_SEED)
        rng = np.random.default_rng(RNG_SEED)
        songs = [song(rng, 120 + 40 * i, i) for i in range(2)]
        net = O.network(songs[0].x.shape[1] + DS.POSITIONS, 16)
        with torch.no_grad():                    # a FiLM away from the identity, so it is checked too
            net.film.weight.normal_(0, 0.05)
        net.eval()
        mu, sd = rng.normal(0, 1, songs[0].x.shape[1]), rng.uniform(0.5, 2, songs[0].x.shape[1])
        arrays = O.export_net(net, mu, sd, 0.5, 16)
        self.assertLess(O.check_export(net, mu, sd, arrays, songs, torch.device("cpu")), O.EXPORT_TOLERANCE)


class RarityTests(unittest.TestCase):
    def test_the_rarest_share_repeated(self):
        rng = np.random.default_rng(RNG_SEED)
        songs = [song(rng, 64, i) for i in range(20)]
        for s in songs[:3]:
            s.x += 8.0                                        # three songs unlike the rest
        r = O.rarity_repeats(songs, np.zeros(songs[0].x.shape[1]), np.ones(songs[0].x.shape[1]))
        self.assertEqual(int((r == O.RARE_REPEATS).sum()), int(round(O.RARE_TOP * 20)))
        self.assertTrue(np.all(r[:3] == O.RARE_REPEATS))


def song(rng, n, i, group=None):
    down = np.arange(n) % 4 == 0
    y, bnd = O.beat_labels(np.arange(n) * 0.5, np.arange(0, n, 32) * 0.5,
                           [CYCLE[k % len(CYCLE)] for k in range(len(range(0, n, 32)))])
    x = rng.normal(0, 1, (n, 155)).astype(np.float32)
    return O.Song(f"s{i}", x, down, DS.positions(down), y, bnd, group or f"g{i}")


def prepared_track(rng, i, n):
    """A track as phrase_parameter_optimizer.prepare_tracks gives it, its features random but for the labels' mark."""
    starts = np.arange(0, n, 32)
    labels = [CYCLE[k % len(CYCLE)] for k in range(starts.size)]
    fz = rng.normal(0, 1, (n, 133)).astype(np.float32)
    for k, a in enumerate(starts):
        fz[a:a + 32, CYCLE.index(labels[k])] += 3.0             # the label is in the features
    return {"uid": F.uid_of(i), "beat_times_sec": np.arange(n) * 0.5, "feature_z": fz,
            "boundary_feature_z": rng.normal(0, 1, (n, 12)).astype(np.float32),
            "downbeat": np.arange(n) % 4 == 0, "phrase_starts_sec": starts * 0.5,
            "phrase_labels": np.asarray(labels)}, (np.r_[starts, n], labels)


def plain_member(path: Path) -> None:
    """The shipped stack's first member as a plain phrase artifact (what a GBM member is written as)."""
    with np.load(SHIPPED, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files if not k.startswith(("m1.", "m2.", "dstcn.", "spec_"))}
    settings = json.loads(str(a["settings_json"].item()))
    for key in ("ensemble_members", "stack_version", "stack_weights"):
        settings.pop(key, None)
    a["settings_json"] = np.asarray(json.dumps(settings))
    np.savez(path, **a)


@needs_torch
class RunTests(unittest.TestCase):
    TRACKS = 6

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cache = self.dir / "cache"
        self.out = self.dir / "weights" / "phrase_analyzer.npz"
        self.out.parent.mkdir(parents=True)
        rng = np.random.default_rng(RNG_SEED)
        made = [prepared_track(rng, i, 96 + 32 * (i % 3)) for i in range(self.TRACKS)]
        self.prepared = [p for p, _ in made]
        self.targets = [t for _, t in made]
        tracks = []
        for i in range(self.TRACKS):
            audio = self.dir / f"{i}.flac"
            audio.write_bytes(b"audio")
            tracks.append({"uid": F.uid_of(i), "title": f"song {i}", "artist": "", "audio_path": audio,
                           "song": F.uid_of(0) if i < 2 else ""})          # the first two: versions of one song
        F.use_tracks(tracks)
        self.cue_calls = []

    def tearDown(self):
        F.library_tracks._tracks = None
        load_two_stage_model.cache_clear()
        self._tmp.cleanup()

    def cues(self, audio_path, beat_times):
        self.cue_calls.append(Path(audio_path).name)
        rng = np.random.default_rng(len(beat_times))
        return rng.normal(0, 1, (len(beat_times), 8)).astype(np.float32), \
            rng.normal(0, 1, (len(beat_times), 14)).astype(np.float32)

    def run_small(self, **request):
        def prepare(*args, **kwargs):
            return PPO.PreparedPhrases([dict(p) for p in self.prepared], list(self.targets), {}, [])

        def member(_artifact, path):
            plain_member(path)
        values = dict(library_dir=self.dir, cache_dir=self.cache, output_path=self.out, members=2, outer_folds=3,
                      channels=8, device="cpu", max_cache_workers=1)
        values.update(request)
        fit = mock.Mock(return_value={})
        with mock.patch.object(O.PPO, "prepare_tracks", prepare), mock.patch.object(O.PPO, "_fit_pipeline", fit), \
                mock.patch.object(O, "export_two_stage_npz_artifact", member), \
                mock.patch.object(O, "song_cues", self.cues), mock.patch.object(O, "spawn_pool", F.thread_pool), \
                mock.patch.object(O, "EPOCHS", 2), mock.patch.object(O, "PATIENCE", 1):
            result = O.optimize_phrase_stack(O.PhraseStackOptimizationRequest(**values))
        return result, fit

    def test_stack_written_and_read_by_the_analysiss_loader(self):
        result, fit = self.run_small()
        self.assertEqual((result.track_count, result.members), (self.TRACKS, 2))
        self.assertGreater(result.specialist_segments, 0)
        self.assertTrue(0.0 <= result.dstcn_out_of_fold_accuracy <= 1.0)
        model = load_two_stage_model(str(self.out))
        self.assertIsNotNone(model.get("stack"))
        with np.load(self.out, allow_pickle=False) as z:
            self.assertEqual(int(z["dstcn.channels"]), 8)                      # the trained DS-TCN, not the shipped
            self.assertTrue(any(k.startswith("spec_") for k in z.files))       # the specialist
            settings = json.loads(str(z["settings_json"].item()))
        self.assertEqual((settings["ensemble_members"], settings["stack_weights"]), (2, list(O.STACK_WEIGHTS)))
        self.assertFalse(any(p.name.startswith(".") for p in self.out.parent.iterdir()))   # the parts removed
        # the members: the second with its inner folds assigned otherwise, every song's features with the columns
        self.assertEqual([c.args[4] for c in fit.call_args_list], [0, 1000])
        self.assertEqual([c.args[6] for c in fit.call_args_list], [None, 1001])
        widths = {np.asarray(p["feature_z"]).shape[1] for p in fit.call_args_list[0].args[0]}
        self.assertEqual(widths, {133 + 8 + len(DS.LABELS) + 1})

    def test_song_cues_cached(self):
        self.run_small()
        self.assertEqual(len(self.cue_calls), self.TRACKS)
        self.cue_calls.clear()
        self.out.unlink()
        self.run_small()
        self.assertEqual(self.cue_calls, [])                  # read back, the audio not decoded again

    def test_too_few_tracks_refused(self):
        self.prepared, self.targets = self.prepared[:2], self.targets[:2]
        with self.assertRaises(TrainingError):
            self.run_small()
        self.assertFalse(self.out.exists())

    def test_output_must_be_an_npz(self):
        with self.assertRaises(ValueError):
            self.run_small(output_path=self.dir / "phrase.json")


if __name__ == "__main__":
    unittest.main()
