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
    """The time stretch the engine plays (--stretch-file): the two vocoders keep the source's level, every band's
    attack lands where the time map puts it, together, with nothing of it ahead; held notes and quiet textures are
    left alone at the hits; slowed down, a vibrato stays steady and a noise stays noise."""

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
                # within a millisecond: its first one the vocoder's own rise and the source's added
                self.assertLess(abs(np.mean(found[name])), 1.0, (factor, name, found[name]))
            # the bands' attacks together: on average within 0.5 ms, each within 2 (a burst's half-level jitters)
            together = found["high"] - found["low"]
            self.assertLess(abs(np.mean(together)), 0.5, (factor, together))
            self.assertLess(np.max(np.abs(together)), 2.0, (factor, together))

    def test_nothing_of_an_attack_ahead_of_it_slowed_down(self):
        """Slowed far down, nothing of an attack is heard ahead of it: the vocoders' long windows smear it back
        (gated, bin by bin); before, the high band's WSOLA put a copy as loud as the attack up to 14 ms early at
        -50 % (a flam, a click before every syllable of a voice). And the attack lands on the time map."""
        from scipy.signal import butter, sosfiltfilt
        pcm, onsets = self._bursts()
        # (the band, the most heard ahead of the attack, how late its half-level point at most and on average: the low
        # band's attack is the source's added on the vocoder's own rise, so its peak, and its half, comes a little later;
        # the highs' most is 3 ms ahead, where the upper band's short window gathers its smear and the gate's 21 ms
        # frames cut it a little less: -26 to -35 dB, -31 dB the median, inside the attack's backward masking; with a
        # 128 ms window above 6.5 kHz it was -35 to -41 dB there, and more of it 5 - 40 ms ahead in real music; 1 - 5 kHz
        # the half-level point jitters with the burst's noise, -1 to +4 ms with the upper band not zero-padded, on
        # average as late)
        bands = {"high": (butter(4, 9000, "highpass", fs=RATE, output="sos"), -25.0, 2.0, 1.0),
                 "low": (butter(4, [1000, 5000], "bandpass", fs=RATE, output="sos"), -15.0, 4.5, 1.5)}
        for factor in (0.5, 0.67):
            y = self._render(pcm, factor)
            for name, (sos, most, late, late_mean) in bands.items():
                env = np.convolve(np.abs(sosfiltfilt(sos, y.mean(axis=1).astype(np.float64))), np.ones(10) / 10, "same")
                ats = []
                for i in onsets[1:-1]:
                    c = int(round(i / factor))
                    attack = env[c - int(0.003 * RATE):c + int(0.03 * RATE)].max()
                    before = env[c - int(0.025 * RATE):c - int(0.003 * RATE)].max()
                    self.assertLess(20 * np.log10(before / attack), most, (factor, name, i))
                    seg = env[c - int(0.03 * RATE):c + int(0.03 * RATE)]
                    ats.append((int(np.argmax(seg > 0.5 * seg.max())) - int(0.03 * RATE)) / RATE * 1000)
                # (a noise burst's half-level crossing jitters from one to the next)
                self.assertLess(np.max(np.abs(ats)), late, (factor, name, ats))
                self.assertLess(abs(np.mean(ats)), late_mean, (factor, name, ats))

    @staticmethod
    def _hits(x: np.ndarray, seconds: float, per_second: float) -> np.ndarray:
        """Drums apart from the 300 Hz - 7 kHz the tests listen to (a 50 Hz kick on every other hit, a hi-hat above
        8 kHz on each) added to ``x``; the hits' times (s)."""
        from scipy.signal import butter, sosfiltfilt
        rng = np.random.default_rng(7)
        m = int(0.15 * RATE)
        k = np.arange(m) / RATE
        kick = 0.6 * np.sin(2 * np.pi * 50 * k) * np.exp(-k / 0.05)
        hat = sosfiltfilt(butter(8, 8000, "highpass", fs=RATE, output="sos"), rng.standard_normal(m)) * np.exp(-k / 0.01) * 0.3
        hits = np.arange(0.3, seconds - 0.3, 1 / per_second)
        for i, at in enumerate(hits):
            a = int(at * RATE)
            x[a:a + m] += (hat + (kick if i % 2 == 0 else 0))[:, None]
        return hits

    def test_held_notes_steady_under_hits(self):
        """A held note under drums stays steady (its harmonics demodulated: their level's movement at 4 - 10 Hz,
        the hits' rate, below -8 dB of it): crossfading to the source's waveform at each hit, or a window shortened
        there, cancelled the note's partials 10 - 25 dB at every hit, heard as a tremolo at the beat's rate."""
        from scipy.signal import butter, sosfiltfilt
        seconds, f0 = 8.0, 233.08
        t = np.arange(int(seconds * RATE)) / RATE
        rng = np.random.default_rng(1)
        note = sum(0.08 / h * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28)) for h in range(1, 17))
        x = np.stack([note, note * 0.95], axis=1)
        self._hits(x, seconds, 4.0)
        lp = butter(4, 40, "lowpass", fs=RATE, output="sos")
        for factor in (0.5, 0.8):
            y = self._render(x.astype(np.float32), factor).mean(axis=1).astype(np.float64)
            tt = np.arange(len(y)) / RATE
            keep = slice(int(0.5 * RATE), len(y) - int(0.5 * RATE))
            depth = []
            for h in range(1, 17):
                amp = np.abs(sosfiltfilt(lp, y * np.exp(-2j * np.pi * f0 * h * tt)))[keep][::RATE // 200]
                a = amp / amp.mean() - 1
                p = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2 / len(a)
                f = np.fft.rfftfreq(len(a), 1 / 200)
                depth.append(p[(f >= 4) & (f < 10)].sum())
            self.assertLess(10 * np.log10(np.mean(depth)), -8.0, factor)

    def test_vibrato_steady_slowed_down(self):
        """A sung note's vibrato (40 cents at 5.5 Hz) slowed down keeps its harmonics' level (demodulated along the
        vibrato itself: their movement at 4 - 10 Hz below -5 dB of it): a window long against the vibrato saw a
        harmonic sweep across its bins, their phases at odds, the level beating at twice the vibrato's rate (the
        "LFO" of a slowed voice, worst at -70 %)."""
        from scipy.signal import butter, sosfiltfilt
        seconds, f0, cents, rate = 8.0, 233.08, 40.0, 5.5
        lp = butter(4, 40, "lowpass", fs=RATE, output="sos")

        def phase(t, speed):
            f = f0 * 2 ** (cents * np.sin(2 * np.pi * rate * speed * t) / 1200)
            return 2 * np.pi * np.cumsum(f) / RATE

        t = np.arange(int(seconds * RATE)) / RATE
        rng = np.random.default_rng(1)
        ph = phase(t, 1.0)
        note = sum(0.08 / h * np.sin(h * ph + rng.uniform(0, 6.28)) for h in range(1, 17))
        x = np.stack([note, note * 0.95], axis=1).astype(np.float32)
        for factor in (0.5, 0.3):
            y = self._render(x, factor).mean(axis=1).astype(np.float64)
            ph_out = phase(np.arange(len(y)) / RATE, factor)
            keep = slice(int(0.5 * RATE), len(y) - int(0.5 * RATE))
            depth = []
            for h in range(1, 17):
                amp = np.abs(sosfiltfilt(lp, y * np.exp(-1j * h * ph_out)))[keep][::RATE // 200]
                a = amp / amp.mean() - 1
                p = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2 / len(a)
                f = np.fft.rfftfreq(len(a), 1 / 200)
                depth.append(p[(f >= 4) & (f < 10)].sum())
            self.assertLess(10 * np.log10(np.mean(depth)), -5.0, factor)

    def test_noise_stays_noise_slowed_down(self):
        """A noise slowed down stays noise (its spectral flatness, 1.5 - 6 kHz, at least 0.75 of the source's): a
        vocoder's bins each held at its phase advance turn a noise into steady sinusoids on the window's grid, a
        metallic ring, the more so the slower and the shorter the window (the upper band's noise is turned at random)."""
        rng = np.random.default_rng(3)
        n = int(8 * RATE)
        f = np.fft.rfftfreq(n, 1 / RATE)
        pink = np.fft.irfft(np.fft.rfft(rng.standard_normal((n, 2)), axis=0) / np.sqrt(np.maximum(f, 20))[:, None], n,
                            axis=0)
        pink = (pink * 0.1 / np.sqrt(np.mean(pink ** 2))).astype(np.float32)

        def flatness(x):
            m = x.mean(axis=1).astype(np.float64)
            k = (len(m) - 2048) // 1024
            idx = np.arange(2048)[None, :] + 1024 * np.arange(k)[:, None]
            p = np.abs(np.fft.rfft(m[idx] * np.hanning(2048), axis=1)) ** 2
            band = p[:, (np.fft.rfftfreq(2048, 1 / RATE) >= 1500) & (np.fft.rfftfreq(2048, 1 / RATE) < 6000)] + 1e-20
            return np.mean(np.exp(np.mean(np.log(band), axis=1)) / np.mean(band, axis=1))

        for factor in (0.7, 0.5, 0.3):
            self.assertGreater(flatness(self._render(pink, factor)) / flatness(pink), 0.75, factor)

    def test_quiet_texture_kept_around_hits(self):
        """A quiet texture (a noise at -36 dBFS: a room, a reverb's tail) keeps its level about the hits (300 -
        6000 Hz, where only it is, within 1.5 dB of its level between them from 100 ms before each to 10 ms before
        it): bins compared one by one, a noise's random half was cut before every hit."""
        from scipy.signal import butter, sosfiltfilt
        seconds = 8.0
        rng = np.random.default_rng(3)
        n = int(seconds * RATE)
        f = np.fft.rfftfreq(n, 1 / RATE)
        pink = np.fft.irfft(np.fft.rfft(rng.standard_normal((n, 2)), axis=0) / np.sqrt(np.maximum(f, 20))[:, None], n,
                            axis=0)
        pink *= 10 ** (-36 / 20) / np.sqrt(np.mean(pink ** 2))
        hits = self._hits(pink, seconds, 4.0)
        band = butter(6, [300, 6000], "bandpass", fs=RATE, output="sos")
        factor = 0.5
        e = sosfiltfilt(band, self._render(pink.astype(np.float32), factor).mean(axis=1).astype(np.float64)) ** 2
        e = np.convolve(e, np.ones(int(0.005 * RATE)) / int(0.005 * RATE), "same")
        rows = []
        for at in hits[2:-2]:
            d = int(at / factor * RATE)
            ref = np.mean(e[d - int(0.2 * RATE):d - int(0.12 * RATE)])
            rows.append([10 * np.log10(e[d - int(s / 1000 * RATE)] / ref) for s in range(10, 101, 5)])
        self.assertGreater(np.min(np.mean(rows, axis=0)), -1.5, np.mean(rows, axis=0))


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
