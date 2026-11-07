"""NativeFeeder's replay to a restarted engine: the settings sent, the deck and the mixer as last reported."""
import struct
import unittest
from unittest import mock

from audioengine import controls
from audioengine.native import client


class _Engine:
    """The calls NativeFeeder makes of an engine, kept (no process, no shared memory)."""

    rate, channels, sent = 48000, 2, 0

    def __init__(self):
        self.calls = []

    def send(self, op, payload=b""):
        self.calls.append((op, bytes(payload)))

    def send_to_deck(self, deck, op, payload=b""):
        self.calls.append((("deck", deck, op), bytes(payload)))

    def state(self):
        return None


def _report(mix):
    return controls.DeckReport(False, 1.0, 1.0, 0.0, 0.0, 0, 0, 0, (0.5, 0.5, 0.5, 0.5), False, -45.0, 0.5, tuple(mix))


class ReplayTests(unittest.TestCase):
    def _mixer_sent(self, engine):
        return {struct.unpack("<Id", p)[0]: struct.unpack("<Id", p)[1]
                for op, p in engine.calls if op == client.OP_SET_MIXER}

    def test_mixer_as_reported_is_sent_again(self):
        engine = _Engine()
        feeder = client.NativeFeeder(engine)
        mix = [0.0] * len(controls.MIX_KEYS)
        mix[controls.MIX_KEY["eq_mode"]] = 1.0          # the EQ, not the isolator
        mix[controls.MIX_KEY["fader_curve"]] = 2.0
        mix[controls.MIX_KEY["hp_mix"]] = 0.3           # a controller's CUE / MASTER mix
        mix[controls.MIX_KEY["color_type"]] = 4.0
        feeder._report = _report(mix)
        engine.calls.clear()
        feeder._replay()
        sent = self._mixer_sent(engine)
        self.assertEqual(sent, dict(enumerate(mix)))

    def test_settings_go_before_the_reported_mixer(self):
        engine = _Engine()
        feeder = client.NativeFeeder(engine)
        feeder.set_mixer("eq_mode", 0.0)
        mix = [0.0] * len(controls.MIX_KEYS)
        mix[controls.MIX_KEY["eq_mode"]] = 1.0          # changed since by a controller
        feeder._report = _report(mix)
        engine.calls.clear()
        feeder._replay()
        eq = [struct.unpack("<Id", p)[1] for op, p in engine.calls
              if op == client.OP_SET_MIXER and struct.unpack("<Id", p)[0] == controls.MIX_KEY["eq_mode"]]
        self.assertEqual(eq[-1], 1.0)

    def test_no_report_no_mixer(self):
        engine = _Engine()
        feeder = client.NativeFeeder(engine)
        engine.calls.clear()
        feeder._replay()
        self.assertEqual(self._mixer_sent(engine), {})

    def test_the_channels_knobs_as_reported_color_too(self):
        engine = _Engine()
        feeder = client.NativeFeeder(engine)
        report = _report([0.0] * len(controls.MIX_KEYS))
        report.knobs, report.color = (0.6, 0.1, 0.5, 0.9), 0.2       # COLOR turned (a controller's or the GUI's)
        feeder._report = report
        engine.calls.clear()
        feeder._replay()
        knobs = {struct.unpack("<Id", p)[0]: struct.unpack("<Id", p)[1]
                 for op, p in engine.calls if op == client.OP_SET_MIXER_KNOB}
        self.assertEqual(knobs, {0: 0.6, 1: 0.1, 2: 0.5, 3: 0.9, 4: 0.2})

    def test_the_channel_cue_as_reported(self):
        engine = _Engine()
        feeder = client.NativeFeeder(engine)
        report = _report([0.0] * len(controls.MIX_KEYS))
        report.channel_cue = True
        feeder._report = report
        engine.calls.clear()
        feeder._replay()
        self.assertIn((client.OP_SET_CHANNEL_CUE, struct.pack("<I", 1)), engine.calls)

    def test_the_key_shift_as_reported_its_fraction_too(self):
        engine = _Engine()
        feeder = client.NativeFeeder(engine)
        report = _report([0.0] * len(controls.MIX_KEYS))
        report.key_shift = 1.25
        feeder._report = report
        engine.calls.clear()
        feeder._replay()
        self.assertIn((client.OP_SET_KEY_SHIFT, struct.pack("<d", 1.25)), engine.calls)

    def test_no_key_shift_none_sent(self):
        engine = _Engine()
        feeder = client.NativeFeeder(engine)
        feeder._report = _report([0.0] * len(controls.MIX_KEYS))
        engine.calls.clear()
        feeder._replay()
        self.assertFalse([op for op, _p in engine.calls if op == client.OP_SET_KEY_SHIFT])

    def test_a_restart_at_another_format_takes_the_new_engine(self):
        """The engine restarted on a device whose format changed meanwhile: the new engine kept (not closed, the dead
        one started again for good), the track decoded at the old rate unloaded and the player told where it was."""
        old = _Engine()
        feeder = client.NativeFeeder(old)
        feeder._pcm, feeder._len = object(), 48000 * 10            # a track loaded at 48 kHz
        feeder._state = mock.Mock(playhead=48000 * 3)               # heard at 3 s
        new = _Engine()
        new.rate = 44100
        told = []
        feeder.format_changed.connect(told.append)
        with mock.patch.object(client.NativeEngine, "start", return_value=new):
            feeder._restart()
        self.assertIs(feeder._engine, new)
        self.assertEqual((feeder.rate, feeder.ch), (44100, 2))
        self.assertIsNone(feeder._pcm)
        self.assertEqual(told, [3.0])
        self.assertFalse([op for op, _p in new.calls if op == client.OP_LOAD])   # nothing at the old rate sent

    def test_a_key_shift_not_a_number_is_not_sent(self):
        engine = _Engine()
        feeder = client.NativeFeeder(engine)
        engine.calls.clear()
        feeder.set_key_shift(float("nan"))
        self.assertFalse([op for op, _p in engine.calls if op == client.OP_SET_KEY_SHIFT])


if __name__ == "__main__":
    unittest.main()
