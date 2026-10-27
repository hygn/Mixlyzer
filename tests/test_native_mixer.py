"""The native engine's mixer by its output PCM (the engine's --mix-file renderer): the channel strip (TRIM, the EQ,
each Sound Color FX and their places in it), the master, the booth and the headphones, each against what it should do,
on synthetic signals (clicks, noise, tones) where a property is measured exactly. Skipped without the built engine
(build/audio_engine.py)."""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from audioengine import controls  # noqa: E402
from audioengine.native import client  # noqa: E402

EXE = client.executable()
SR = 48000
BPM = 120.0
BEAT = 60 / BPM
COLOR = {name: k for k, name in enumerate(controls.COLOR_TYPES)}
TMP = tempfile.mkdtemp(prefix="mixer-test-")


def render(x, sets=None, knobs=None, bpm=BPM, tail=0.0, bus="master", cue=False, block=512, stop_at=-1.0, clip=None):
    """x: (n, 2) float32 -> the mixer's output (m, 2): ``bus`` master, hp (the headphones) or booth; ``cue`` the
    channel's headphones CUE; the deck stopped (its fader down) at ``stop_at`` s; ``clip``: the outputs' last stages
    (master, headphones, booth: controls.CLIPS), else none (the mixer before them)."""
    src = os.path.join(TMP, "in.f32")
    np.ascontiguousarray(x, dtype="<f4").tofile(src)
    out = os.path.join(TMP, "out.f32")
    args = [str(EXE), "--mix-file", src, "--out", out, "--rate", str(SR), "--bpm", str(bpm), "--tail", str(tail),
            "--bus", bus, "--block", str(block), "--stop-at", str(stop_at)]
    if sets:
        args += ["--set", ",".join(f"{k}={v}" for k, v in sets.items())]
    if knobs:
        args += ["--knob", ",".join(f"{k}={v}" for k, v in knobs.items())]
    if cue:
        args += ["--cue"]
    if clip:
        args += ["--clip", ",".join(clip)]
    subprocess.run(args, check=True, capture_output=True)
    return np.fromfile(out, dtype="<f4").reshape(-1, 2)


def db(a, b):
    return 20 * np.log10(max(a, 1e-12) / max(b, 1e-12))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def band_rms(x, lo, hi):
    """The RMS of the mono sum between lo and hi Hz."""
    m = x.mean(axis=1)
    spec = np.fft.rfft(m * np.hanning(len(m)))
    f = np.fft.rfftfreq(len(m), 1 / SR)
    return float(np.sqrt(np.sum(np.abs(spec[(f >= lo) & (f < hi)]) ** 2)))


def centroid(x):
    m = x.mean(axis=1)
    spec = np.abs(np.fft.rfft(m * np.hanning(len(m))))
    f = np.fft.rfftfreq(len(m), 1 / SR)
    return float(np.sum(f * spec) / max(np.sum(spec), 1e-12))


def noise(sec, amp=0.25, seed=1):
    v = (np.random.default_rng(seed).standard_normal(int(sec * SR)) * amp).astype(np.float32)
    return np.stack([v, v], axis=1)


def stereo_noise(sec, amp=0.25, seed=1):
    return (np.random.default_rng(seed).standard_normal((int(sec * SR), 2)) * amp).astype(np.float32)


def tone(hz, sec, amp=0.5):
    v = (amp * np.sin(2 * np.pi * hz * np.arange(int(sec * SR)) / SR)).astype(np.float32)
    return np.stack([v, v], axis=1)


def click(sec, at=0.1, amp=0.8):
    x = np.zeros((int(sec * SR), 2), np.float32)
    x[int(at * SR)] = amp
    return x


def music(sec, seed=3):
    """Something like a track: decaying chords, a kick and noise bursts on a beat, left and right apart."""
    t = np.arange(int(sec * SR)) / SR
    rng = np.random.default_rng(seed)
    env = np.exp(-(t % BEAT) * 12.0)
    x = sum(0.08 * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) for f in (110, 220, 330, 440, 660, 3520))
    x = x * (0.4 + 0.6 * env) + 0.4 * np.sin(2 * np.pi * 50 * t) * env ** 4 + rng.standard_normal(t.size) * 0.05 * env
    return np.stack([x, np.roll(x, 37) * 0.9], axis=1).astype(np.float32)


def peaks_after(y, start, count, min_gap):
    """The ``count`` strongest separated peaks (time s, level) of |mono| after ``start`` s, by time."""
    m = np.abs(y.mean(axis=1)).copy()
    m[: int(start * SR)] = 0
    out = []
    for _ in range(count):
        i = int(np.argmax(m))
        if m[i] <= 1e-6:
            break
        out.append((i / SR, float(m[i])))
        m[max(0, i - int(min_gap * SR)): i + int(min_gap * SR)] = 0
    return sorted(out)


def transfer_db(y, x, seg=4096):
    """The transfer function (dB) from x to y (mono sums), averaged over segments (H = Pxy / Pxx), and its
    frequencies."""
    a, b = x.mean(axis=1), y.mean(axis=1)
    w = np.hanning(seg)
    pxy = pxx = 0
    for k in range(0, len(a) - seg, seg // 2):
        fa, fb = np.fft.rfft(a[k:k + seg] * w), np.fft.rfft(b[k:k + seg] * w)
        pxy = pxy + fb * np.conj(fa)
        pxx = pxx + np.abs(fa) ** 2
    h = np.abs(pxy) / np.maximum(pxx, 1e-20)
    return 20 * np.log10(np.maximum(h, 1e-9)), np.fft.rfftfreq(seg, 1 / SR)


def comb_delay_ms(h_db, f, lo=300.0, hi=12000.0):
    """A comb's delay (ms) from its response's ripple: the strongest period of the log magnitude over frequency
    (its teeth 1 / delay apart), read by an autocorrelation of the ripple."""
    sel = (f >= lo) & (f <= hi)
    r = h_db[sel] - np.convolve(h_db[sel], np.ones(31) / 31, mode="same")
    r = r[20:-20] - r[20:-20].mean()
    ac = np.correlate(r, r, "full")[len(r) - 1:]
    df = f[1] - f[0]
    k0 = int(np.argmax(ac < 0))   # past the main lobe: the ripple's period is the next peak
    k = k0 + int(np.argmax(ac[k0:len(ac) // 2]))
    return 1000.0 / (k * df)


@unittest.skipIf(EXE is None, "the native engine is not built (build/audio_engine.py)")
class Neutral(unittest.TestCase):
    def test_neutral_settings_leave_the_samples_as_they_are(self):
        x = music(6)
        np.testing.assert_array_equal(render(x), x)
        np.testing.assert_array_equal(render(x, sets={"color_type": COLOR["SPACE"]}), x, "Color FX picked, COLOR at the middle")
        np.testing.assert_array_equal(render(x, sets={"color_type": controls.COLOR_OFF}, knobs={"color": 0.1}), x,
                                      "no Color FX, COLOR turned")


@unittest.skipIf(EXE is None, "the native engine is not built (build/audio_engine.py)")
class Strip(unittest.TestCase):
    def test_eq_bands_at_the_bottom(self):
        """An isolator band killed (its LR4 crossovers at 246 / 2484 Hz), the EQ's at -26 dB (shelves at 70 Hz /
        13 kHz, a peak at 1 kHz)."""
        for mode, cases in [(0, [("low", 60, -45), ("mid", 800, -30), ("high", 10000, -50)]),
                            (1, [("low", 30, -20), ("mid", 1000, -20), ("high", 18000, -18)])]:
            for knob, f, floor in cases:
                with self.subTest(mode=controls.EQ_MODES[mode], knob=knob):
                    t = tone(f, 1.5)
                    self.assertLess(db(rms(render(t, sets={"eq_mode": mode}, knobs={knob: 0.0})[SR:]), rms(t[SR:])), floor)

    def test_eq_low_at_the_top_is_6_db(self):
        x = noise(4)
        for mode in (0, 1):
            with self.subTest(mode=controls.EQ_MODES[mode]):
                y = render(x, sets={"eq_mode": mode}, knobs={"low": 1.0})
                self.assertAlmostEqual(db(band_rms(y[SR:], 20, 60), band_rms(x[SR:], 20, 60)), 6.0, delta=1.0)

    def test_trim_at_the_top_is_9_db(self):
        t = tone(1000, 2)
        self.assertAlmostEqual(db(rms(render(t, knobs={"trim": 1.0})[SR:]), rms(t[SR:])), 9.0, delta=0.1)


@unittest.skipIf(EXE is None, "the native engine is not built (build/audio_engine.py)")
class SoundColorFx(unittest.TestCase):
    def test_filter(self):
        x = noise(4)
        y = render(x, sets={"color_type": COLOR["FILTER"]}, knobs={"color": 0.1})
        self.assertLess(db(band_rms(y[SR:], 2000, 16000), band_rms(x[SR:], 2000, 16000)), -40, "left: a low-pass")
        y = render(x, sets={"color_type": COLOR["FILTER"]}, knobs={"color": 0.9})
        self.assertLess(db(band_rms(y[SR:], 20, 500), band_rms(x[SR:], 20, 500)), -30, "right: a high-pass")
        lo = render(x, sets={"color_type": COLOR["FILTER"], "color_param": 0.0}, knobs={"color": 0.25})
        hi = render(x, sets={"color_type": COLOR["FILTER"], "color_param": 1.0}, knobs={"color": 0.25})
        fc = 20000 * (40 / 20000) ** 0.5
        self.assertGreater(db(band_rms(hi[SR:], fc * 0.9, fc * 1.1), band_rms(lo[SR:], fc * 0.9, fc * 1.1)), 10,
                           "PARAMETER: the resonance at the cutoff")

    def test_sweep(self):
        t1, t3, t10 = tone(1000, 3), tone(250, 3), tone(10000, 3)

        def gain(t, knob, param=0.5):
            return db(rms(render(t, sets={"color_type": COLOR["SWEEP"], "color_param": param}, knobs={"color": knob})[SR:]), rms(t[SR:]))
        self.assertLess(gain(t1, 0.1), -30, "left: a notch at 1 kHz")
        self.assertGreater(gain(t3, 0.1), -6)
        self.assertLess(abs(gain(t1, 0.9)), 1, "right: a band-pass at 1 kHz")
        self.assertLess(gain(t3, 0.9), -20)
        self.assertLess(abs(gain(t10, 0.9, 1.0)), 1, "PARAMETER 1: its centre at 10 kHz")

    def test_noise(self):
        s = np.zeros((SR * 3, 2), np.float32)

        def out(param, knob):
            return render(s, sets={"color_type": COLOR["NOISE"], "color_param": param}, knobs={"color": knob})[SR:]
        self.assertEqual(rms(out(0.0, 0.9)), 0, "PARAMETER 0: no noise")
        self.assertAlmostEqual(db(rms(out(1.0, 0.7)), rms(out(0.5, 0.7))), 6.02, delta=0.5, msg="PARAMETER: its volume")
        cl, cr = centroid(out(1.0, 0.1)), centroid(out(1.0, 0.9))
        self.assertTrue(cl < 1000 < 5000 < cr, f"left low-passed, right high-passed: {cl:.0f} / {cr:.0f} Hz")

    def test_crush(self):
        t = tone(440, 3)
        y0 = render(t, sets={"color_type": COLOR["CRUSH"], "color_param": 0.0}, knobs={"color": 0.1})[SR:, 0]
        y1 = render(t, sets={"color_type": COLOR["CRUSH"], "color_param": 1.0}, knobs={"color": 0.1})[SR:, 0]
        l0, l1 = len(np.unique(np.round(y0, 4))), len(np.unique(np.round(y1, 4)))
        e0, e1 = db(rms(y0 - t[SR:, 0]), rms(t[SR:, 0])), db(rms(y1 - t[SR:, 0]), rms(t[SR:, 0]))
        self.assertTrue(l1 < l0 and l1 < 100 and e1 > e0, f"bits and rate down, more with PARAMETER: {l0} -> {l1} levels")
        low = tone(60, 3)
        y = render(low, sets={"color_type": COLOR["CRUSH"], "color_param": 0.5}, knobs={"color": 0.95})
        self.assertLess(db(band_rms(y[SR:], 40, 80), band_rms(low[SR:], 40, 80)), -20, "right: through a high-pass")

    def test_dub_echo(self):
        ratios = []
        for p in (0.2, 0.9):
            y = render(click(4), sets={"color_type": COLOR["DUB ECHO"], "color_param": p}, knobs={"color": 0.05})
            pk = peaks_after(y, 0.15, 3, 0.2)
            ratios.append(pk[1][1] / pk[0][1] if len(pk) > 1 else 0)
            gaps = np.diff([0.1] + [t for t, _ in pk])
            np.testing.assert_allclose(gaps, 0.75 * BEAT, atol=0.01, err_msg="repeats 3/4 beat apart")
        self.assertGreater(ratios[1], ratios[0] * 1.5, "PARAMETER: the feedback")
        el = render(click(3), sets={"color_type": COLOR["DUB ECHO"], "color_param": 0.5}, knobs={"color": 0.05})[int(0.2 * SR):]
        er = render(click(3), sets={"color_type": COLOR["DUB ECHO"], "color_param": 0.5}, knobs={"color": 0.95})[int(0.2 * SR):]
        self.assertGreater(db(band_rms(el, 500, 2000), band_rms(el, 6000, 12000)), 10, "left: the middle range")
        self.assertGreater(db(band_rms(er, 4000, 12000), band_rms(er, 200, 800)), 20, "right: the highs")

    def test_space(self):
        def decay(y):
            m = np.convolve(np.abs(y.mean(axis=1)), np.ones(480) / 480, "same")
            return db(m[int(1.0 * SR)], m[int(0.3 * SR)])
        d0 = decay(render(click(3), sets={"color_type": COLOR["SPACE"], "color_param": 0.0}, knobs={"color": 0.9}))
        d1 = decay(render(click(3), sets={"color_type": COLOR["SPACE"], "color_param": 1.0}, knobs={"color": 0.9}))
        self.assertGreater(d1, d0 + 10, "PARAMETER: the feedback (a longer tail)")
        cl = centroid(render(click(3), sets={"color_type": COLOR["SPACE"]}, knobs={"color": 0.1})[int(0.2 * SR):])
        cr = centroid(render(click(3), sets={"color_type": COLOR["SPACE"]}, knobs={"color": 0.9})[int(0.2 * SR):])
        self.assertLess(cl, cr / 2, "left low, right high")

    def test_jet(self):
        """A flanger's comb turned by hand: 6 ms at the middle to 0.12 ms at the ends, evenly in pitch; the right's
        repeats in phase, the left's inverted (their teeth between the right's)."""
        x = stereo_noise(5)
        for knob in (0.65, 0.8):
            with self.subTest(knob=knob):
                want = 6.0 * (0.12 / 6.0) ** ((knob - 0.5) * 2)
                h, f = transfer_db(render(x, sets={"color_type": COLOR["JET"], "color_param": 0.6}, knobs={"color": knob})[SR:], x[SR:])
                sel = (f > 300) & (f < 8000)
                self.assertLess(abs(comb_delay_ms(h, f) - want) / want, 0.2)
                self.assertGreater(float(np.percentile(h[sel], 95) - np.percentile(h[sel], 5)), 10, "its depth")
        hr, f = transfer_db(render(x, sets={"color_type": COLOR["JET"], "color_param": 0.8}, knobs={"color": 0.65})[SR:], x[SR:])
        hl, _ = transfer_db(render(x, sets={"color_type": COLOR["JET"], "color_param": 0.8}, knobs={"color": 0.35})[SR:], x[SR:])
        tooth = 1000.0 / (6.0 * (0.12 / 6.0) ** 0.3)

        def at(h, hz):
            return float(np.max(h[(f > 0.95 * hz) & (f < 1.05 * hz)]))
        # in theory (feedback 0.81) 11.9 dB and 19.3 dB; the narrow teeth come out lower in 4096-point segments
        self.assertGreater(at(hr, tooth) - at(hl, tooth), 6)
        self.assertGreater(at(hl, 0.5 * tooth) - at(hr, 0.5 * tooth), 6)


@unittest.skipIf(EXE is None, "the native engine is not built (build/audio_engine.py)")
class EffectPlaces(unittest.TestCase):
    """The effects' places in the strip (a DJM-A9's block diagram): NOISE before the EQ, SPACE / DUB ECHO after the
    fader (not in the pre-fader CUE; their tail on after the deck stopped with its fader down)."""

    def test_noise_before_the_eq(self):
        x = stereo_noise(5)
        y = render(x, sets={"color_type": COLOR["NOISE"], "color_param": 1.0}, knobs={"color": 0.9, "low": 0, "mid": 0, "high": 0})
        self.assertLess(db(rms(y[SR:]), rms(x[SR:])), -60, "a full kill takes its noise too")

    def test_space_and_dub_echo_after_the_fader(self):
        x = stereo_noise(5)
        n = len(x)
        for name in ("SPACE", "DUB ECHO"):
            with self.subTest(effect=name):
                sets = {"color_type": COLOR[name], "color_param": 0.6, "hp_master": 0}
                cue = render(x, sets=sets, knobs={"color": 0.95}, bus="hp", cue=True)
                master = render(x, sets=sets, knobs={"color": 0.95})
                self.assertLess(db(rms(cue[:n] - x), rms(x)), -80, "the CUE: the sound as it was")
                self.assertGreater(db(rms(master[:n] - x), rms(x)), -25, "the master: the effect")
                y = render(x[:4 * SR], sets=sets, knobs={"color": 0.95}, stop_at=3.0, tail=1.0)
                quiet = render(x[:4 * SR], stop_at=3.0, tail=1.0)
                after = slice(int(3.05 * SR), int(3.6 * SR))
                self.assertGreater(db(rms(y[after]), rms(x[:3 * SR])), -50, "its tail on after the deck stopped")
                self.assertFalse(np.any(quiet[after]), "nothing at all without the effect")


@unittest.skipIf(EXE is None, "the native engine is not built (build/audio_engine.py)")
class Buses(unittest.TestCase):
    def test_master_level(self):
        t = tone(1000, 2, 0.25)
        for k, want in [(0.8, 0.0), (1.0, 20 * np.log10((1 / 0.8) ** 2)), (0.4, 20 * np.log10(0.25))]:
            with self.subTest(level=k):
                self.assertAlmostEqual(db(rms(render(t, sets={"master_level": k})[SR:]), rms(t[SR:])), want, delta=0.05)

    def test_booth_eq(self):
        x = noise(3)
        y = render(x, sets={"booth_eq_low": 0.0}, bus="booth")
        self.assertAlmostEqual(db(band_rms(y[SR:], 20, 40), band_rms(x[SR:], 20, 40)), -12, delta=1.0)
        y = render(x, sets={"booth_eq_hi": 1.0}, bus="booth")
        self.assertAlmostEqual(db(band_rms(y[SR:], 16000, 20000), band_rms(x[SR:], 16000, 20000)), 6, delta=1.0)

    def test_headphones(self):
        st = stereo_noise(2)
        np.testing.assert_allclose(render(st, cue=True, bus="hp", sets={"hp_mix": 0.0}), st, atol=1e-6,
                                   err_msg="the CUE'd channel, stereo")
        self.assertEqual(rms(render(st, bus="hp", sets={"hp_mix": 0.0})), 0, "no CUE: nothing")
        # MONO SPLIT: the CUE mono left, the master right (its fader down: silent after its first block's fade)
        y = render(st, cue=True, bus="hp", knobs={"fader": 0.0}, sets={"hp_split": 1, "hp_mix": 0.5})
        np.testing.assert_allclose(y[:, 0], st.mean(axis=1), atol=1e-6)
        self.assertEqual(rms(y[1024:, 1]), 0)


class LastStages(unittest.TestCase):
    """Each output's last stage on music pushed over 0 dBFS (TRIM at the top, +9 dB), each output its own."""
    LOUD = {"trim": 1.0}

    def test_each_stage_keeps_its_output_at_0_dbfs(self):
        x = music(4)
        raw = render(x, knobs=self.LOUD)                       # the mixer before its last stage
        self.assertGreater(np.abs(raw).max(), 1.5)
        d = round(controls.LIMITER_DELAY_SEC * SR) - 1         # the limiter's delay in frames
        stages = ("limiter", "tanh", "hard")
        for bus, kind in zip(("master", "hp", "booth"), stages):
            with self.subTest(bus=bus, kind=kind):
                y = render(x, knobs=self.LOUD, bus=bus, sets={"hp_mix": 1.0}, clip=stages)
                self.assertLessEqual(np.abs(y).max(), 1.0)
                ref = render(x, knobs=self.LOUD, bus=bus, sets={"hp_mix": 1.0})
                if kind == "hard":
                    np.testing.assert_array_equal(y, np.clip(ref, -1.0, 1.0))
                elif kind == "tanh":
                    under = np.abs(ref) <= 0.8
                    np.testing.assert_array_equal(y[under], ref[under])
                    self.assertLess(np.abs(y).max(), 1.0)
                else:
                    # a gain on the music d frames late: no sample flat at the ceiling, the shape kept
                    late = ref[: len(ref) - d]
                    g = y[d:, 0][np.abs(late[:, 0]) > 0.3] / late[:, 0][np.abs(late[:, 0]) > 0.3]
                    self.assertGreater(g.min(), 0.0)
                    self.assertLessEqual(g.max(), 1.0 + 1e-6)
                    self.assertLess(int(np.sum(np.diff(y[d:, 0]) == 0) - np.sum(np.diff(late[:, 0]) == 0)), 10)

    def test_the_limiter_leaves_music_under_0_dbfs_as_it_is_late(self):
        x = music(2)
        self.assertLess(np.abs(x).max(), 1.0)
        d = round(controls.LIMITER_DELAY_SEC * SR) - 1
        y = render(x, clip=("limiter", "limiter", "limiter"))
        np.testing.assert_array_equal(y[:d], 0.0)
        np.testing.assert_array_equal(y[d:], x[: len(x) - d])


if __name__ == "__main__":
    unittest.main()
