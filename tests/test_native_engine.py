"""The native audio engine (audioengine/native): its scratch path renders as the Python engine's, its time stretch
keeps the level and the attacks, and the engine plays through its control block. Skipped without the built engine
(build/audio_engine.py), and the playback test without an output device."""
import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from audioengine.native import client  # noqa: E402

EXE = client.executable()
RATE = 48000


def _music(seconds: float, seed: int = 3) -> np.ndarray:
    """A stereo test signal with onsets: decaying chords and noise bursts every 0.25 s."""
    t = np.arange(int(RATE * seconds)) / RATE
    rng = np.random.default_rng(seed)
    x = np.zeros_like(t)
    for f in (110.0, 220.0, 330.0, 440.0, 660.0):
        x += np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * 0.08
    env = np.exp(-(t % 0.25) * 18.0)
    x = x * (0.4 + 0.6 * env) + rng.standard_normal(t.size) * 0.05 * env
    return np.stack([x, np.roll(x, 37) * 0.9], axis=1).astype(np.float32)


@unittest.skipIf(EXE is None, "the native engine is not built (build/audio_engine.py)")
class NativeMatchesPython(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def _f32(self, name: str, data: np.ndarray) -> str:
        path = os.path.join(self.dir, name)
        np.ascontiguousarray(data).astype("<f4").tofile(path)
        return path

    def test_scratch_path_is_the_same(self):
        from audioengine.python import dsp
        dsp.warm_scratch_numba()
        pcm = _music(4.0)
        events = [(0.0, 2.0 * RATE, 1.0)]
        t, x = 0.02, 0.0
        for stroke, speed in ((1, 900.0), (-1, 1500.0), (1, 300.0)):
            end = t + 0.2
            while t < end:
                x += stroke
                events.append((t, 2.0 * RATE - x * 150.0, 0.0))
                t += 1.0 / speed
            events.append((t + 0.02, 2.0 * RATE - x * 150.0, 1.0))
            t += 0.08
        ev = np.array(events, dtype="<f8")
        ev_path = os.path.join(self.dir, "ev.f64")
        ev.tofile(ev_path)
        out = os.path.join(self.dir, "out.f32")
        subprocess.run([str(EXE), "--scratch-file", self._f32("in.f32", pcm), "--events", ev_path, "--out", out],
                       check=True, capture_output=True)
        rust = np.fromfile(out, "<f4").reshape(-1, 2)

        platter = dsp.ScratchPlatter(RATE)
        platter.reset(ev[0, 1], 0.0)
        blocks, k, now = [], 1, 0.0
        while now < ev[-1, 0] + 0.2:
            while k < len(ev) and ev[k, 0] <= now:
                platter.set_target(ev[k, 1], ev[k, 0], rest=bool(ev[k, 2]))
                k += 1
            start = platter.position
            blocks.append(dsp.read_at(pcm, platter.positions(48, now), previous=start))
            now += 48 / RATE
        python = np.concatenate(blocks)
        self.assertEqual(python.shape, rust.shape)
        np.testing.assert_allclose(rust, python, atol=1e-6)


@unittest.skipIf(EXE is None, "the native engine is not built (build/audio_engine.py)")
class NativeStretch(unittest.TestCase):
    """The time stretch the engine plays (--stretch-file): the low band's vocoder and the high band's
    WSOLA keep the source's level, and every band's attack lands where the time map puts it, together."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def _bursts(self) -> tuple[np.ndarray, np.ndarray]:
        """Noise bursts (sharp onsets, every band) over a pad and a bass; their onsets in frames."""
        rng = np.random.default_rng(5)
        t = np.arange(RATE * 8) / RATE
        pad = sum(0.04 * np.sin(2 * np.pi * f * t + p) for f, p in ((55, 0), (600, 1), (900, 2), (6000, 3)))
        x = np.stack([pad, pad * 0.9], axis=1)
        onsets = (np.arange(0.5, 7.5, 0.37) * RATE).astype(int)
        n = int(0.08 * RATE)
        for i in onsets:
            x[i:i + n] += rng.standard_normal((n, 2)) * np.exp(-np.arange(n) / (0.012 * RATE))[:, None] * 0.5
        return x.astype(np.float32), onsets

    def _render(self, pcm: np.ndarray, factor: float) -> np.ndarray:
        src = os.path.join(self.dir, "st_in.f32")
        out = os.path.join(self.dir, "st_out.f32")
        np.ascontiguousarray(pcm).astype("<f4").tofile(src)
        subprocess.run([str(EXE), "--stretch-file", src, "--out", out, "--factor", str(factor)],
                       check=True, capture_output=True)
        return np.fromfile(out, "<f4").reshape(-1, 2)

    def test_level_and_attacks(self):
        from scipy.signal import butter, sosfiltfilt
        pcm, onsets = self._bursts()
        bands = {"low": butter(4, [1000, 5000], "bandpass", fs=RATE, output="sos"),
                 "high": butter(4, 9000, "highpass", fs=RATE, output="sos")}

        def level(y, sos):
            return 10 * np.log10(np.mean(sosfiltfilt(sos, y.mean(axis=1).astype(np.float64)) ** 2))

        for factor in (0.8, 0.92, 1.25):
            y = self._render(pcm, factor)
            self.assertEqual(len(y), int(np.ceil(len(pcm) / factor - 1e-9)), factor)
            for name, sos in bands.items():
                # the level per output second: the source's, stretched
                self.assertLess(abs(level(y, sos) - level(pcm, sos)), 1.0, (factor, name))
            found = {}
            for name, sos in bands.items():
                env = np.abs(sosfiltfilt(sos, y.mean(axis=1).astype(np.float64)))
                env = np.convolve(env, np.ones(10) / 10, "same")
                at = []
                for i in onsets[1:-1]:
                    c = int(round(i / factor))
                    seg = env[c - int(0.03 * RATE):c + int(0.03 * RATE)]
                    at.append((int(np.argmax(seg > 0.5 * seg.max())) - int(0.03 * RATE)) / RATE * 1000)
                found[name] = np.array(at)
                self.assertLess(abs(np.mean(found[name])), 0.6, (factor, name, found[name]))
            self.assertLess(np.max(np.abs(found["high"] - found["low"])), 1.5, factor)


@unittest.skipIf(EXE is None, "the native engine is not built (build/audio_engine.py)")
class NativePlayback(unittest.TestCase):
    def test_plays_seeks_and_pauses(self):
        engine = client.NativeEngine.start()
        if engine is None:
            self.skipTest("the native engine did not start (no output device?)")
        try:
            rate, ch = engine.rate, engine.channels
            pcm = client.share_pcm(np.zeros((rate * 10, ch), np.float32))
            name = pcm.shm_name.encode()
            engine.send(client.OP_SET_VOLUME, struct.pack("<d", 0.0))   # the channel fader down: silent
            engine.send(client.OP_LOAD, struct.pack("<IIQI", 1, ch, pcm.shape[0], len(name)) + name)
            engine.send(client.OP_PLAY)
            time.sleep(0.5)
            st = engine.state()
            if st.playhead == 0.0:
                time.sleep(1.5)   # the device not opened: tried again once a second
                if engine.state().device_errors > 0:
                    self.skipTest("the output device is held (another app in exclusive mode?)")
                st = engine.state()
            self.assertEqual(st.track_gen, 1)
            self.assertTrue(st.playing)
            self.assertGreater(st.playhead, 0.2 * rate)
            engine.send(client.OP_SEEK, struct.pack("<d", 5.0 * rate))
            time.sleep(0.3)
            self.assertGreater(engine.state().playhead, 5.0 * rate)
            engine.send(client.OP_PAUSE)
            time.sleep(0.1)
            st = engine.state()
            self.assertFalse(st.playing)
            held = st.playhead
            time.sleep(0.2)
            self.assertEqual(engine.state().playhead, held)
            self.assertEqual(engine.state().cmd_done, engine._seq)
        finally:
            engine.close()
        self.assertFalse(engine.alive())

    def test_the_masters_spectrum_while_asked(self):
        """A 1 kHz sine played with the master routed nowhere (the device silent): its spectrum frames peak at
        1 kHz while asked for (op SET_SPECTRUM), none come after."""
        from audioengine.controls import SPECTRUM_BINS, SPECTRUM_FRAMES_PER_SEC, spectrum_edges
        engine = client.NativeEngine.start()
        if engine is None:
            self.skipTest("the native engine did not start (no output device?)")
        try:
            rate, ch = engine.rate, engine.channels
            t = np.arange(rate * 5) / rate
            sine = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
            pcm = client.share_pcm(np.repeat(sine[:, None], ch, axis=1))
            name = pcm.shm_name.encode()
            engine.send(client.OP_SET_ROUTING, struct.pack("<6B", *([client.NO_CHANNEL] * 6)))
            engine.send(client.OP_SET_VOLUME, struct.pack("<d", 1.0))
            engine.send(client.OP_LOAD, struct.pack("<IIQI", 1, ch, pcm.shape[0], len(name)) + name)
            engine.send(client.OP_PLAY)
            engine.send(client.OP_SET_SPECTRUM, struct.pack("<I", 1))
            time.sleep(0.6)
            if engine.state().device_errors > 0 or engine.state().playhead == 0.0:
                self.skipTest("the output device is held (another app in exclusive mode?)")
            engine.spectrum_frames()
            time.sleep(0.4)
            frames = engine.spectrum_frames()
            self.assertGreaterEqual(len(frames), 0.4 * SPECTRUM_FRAMES_PER_SEC - 3)
            self.assertEqual(frames.shape[1], SPECTRUM_BINS)
            edges = np.asarray(spectrum_edges())
            peak = int(np.argmax(frames[-1]))
            self.assertTrue(edges[peak - 1] <= 1000.0 <= edges[peak + 2], (edges[peak], edges[peak + 1]))
            engine.send(client.OP_SET_SPECTRUM, struct.pack("<I", 0))
            time.sleep(0.2)
            engine.spectrum_frames()
            time.sleep(0.3)
            self.assertEqual(len(engine.spectrum_frames()), 0)
        finally:
            engine.close()


@unittest.skipIf(EXE is None, "the native engine is not built (build/audio_engine.py)")
class NativeMalformedCommands(unittest.TestCase):
    def test_counts_and_sizes_are_not_trusted(self):
        """Counts past the payload, a PCM larger than its block, NaN positions: refused or cut to the payload; the
        engine stays up and takes the next command at once."""
        engine = client.NativeEngine.start()
        if engine is None:
            self.skipTest("the native engine did not start (no output device?)")
        try:
            ch = engine.channels
            pcm = client.share_pcm(np.zeros((1000, ch), np.float32))
            name = pcm.shm_name.encode()
            huge = 0xFFFFFFFF
            engine.send(client.OP_CLICK_SAMPLE, struct.pack("<II", huge, huge))
            engine.send(client.OP_TONE_HOLD, struct.pack("<I", huge))
            engine.send(client.OP_CLICK_BEATS, struct.pack("<I", huge) + struct.pack("<d", 1.0))
            engine.send(client.OP_DECK_MARKS, struct.pack("<I", huge))
            engine.send(client.OP_ARM_JUMP, struct.pack("<ddIIdd", float("nan"), 100.0, 1, huge, float("nan"), 50.0))
            for frames in (1 << 62, pcm.shape[0] * 1000):   # past usize when multiplied; past the block
                engine.send(client.OP_LOAD, struct.pack("<IIQI", 7, ch, frames, len(name)) + name)
            engine.send(client.OP_WAKE)
            deadline = time.monotonic() + 2.0
            while engine.state().cmd_done < engine._seq and time.monotonic() < deadline:
                time.sleep(0.01)
            st = engine.state()
            self.assertTrue(engine.alive())
            self.assertEqual(st.cmd_done, engine._seq)
            self.assertEqual(st.track_gen, 7)   # the load taken as "no track"
            self.assertFalse(st.at_end and st.playing)
        finally:
            engine.close()
        self.assertFalse(engine.alive())


if __name__ == "__main__":
    unittest.main()
