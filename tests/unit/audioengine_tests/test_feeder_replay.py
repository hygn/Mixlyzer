"""NativeFeeder's replay to a restarted engine: the settings sent, the deck and the mixer as last reported."""
import struct
import unittest

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


if __name__ == "__main__":
    unittest.main()
