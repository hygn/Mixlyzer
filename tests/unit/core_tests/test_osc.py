"""core.playback.osc: External Sync's packets come from the network, so the decoder takes a well-formed OSC 1.0
packet and refuses anything else as a whole (OscError, never another exception). Boundaries of every type, every
truncation of a valid packet and seeded random corruption."""
import math
import random
import struct
import unittest

from core.playback import osc

SEED = 20261027


def bundle(*elements: bytes, time_tag: bytes = bytes(8)) -> bytes:
    return b"#bundle\x00" + time_tag + b"".join(struct.pack(">i", len(e)) + e for e in elements)


class RoundTripTests(unittest.TestCase):
    def test_every_type_at_its_bounds(self):
        args = [0, -1, (1 << 31) - 1, -(1 << 31), 1 << 31, -(1 << 31) - 1, (1 << 63) - 1, -(1 << 63),
                0.0, -0.0, 1.5, float("inf"), float("-inf"), 5e-324, 1.7976931348623157e308,
                "", "a", "abc", "abcd", "abcde", "\u3042\U0001F3B5", True, False]
        [(address, got)] = osc.parse(osc.message("/x", *args))
        self.assertEqual(address, "/x")
        self.assertEqual(got, args)
        self.assertEqual([type(a) for a in got], [type(a) for a in args])
        self.assertEqual(math.copysign(1, got[9]), -1.0)                        # -0.0 kept

    def test_nan(self):
        [(_a, [value])] = osc.parse(osc.message("/x", float("nan")))
        self.assertTrue(math.isnan(value))

    def test_float32(self):
        packet = b"/f\x00\x00" + b",f\x00\x00" + struct.pack(">f", 0.25)
        self.assertEqual(osc.parse(packet), [("/f", [0.25])])

    def test_address_lengths(self):
        """The NUL and its padding: an address of 3 letters takes 4 bytes, of 4 takes 8."""
        for n in range(1, 13):
            address = "/" + "a" * (n - 1)
            packet = osc.message(address, 1)
            self.assertEqual(len(packet) % 4, 0)
            self.assertEqual(osc.parse(packet), [(address, [1])])

    def test_no_type_tags(self):
        self.assertEqual(osc.parse(b"/a\x00\x00"), [("/a", [])])
        self.assertEqual(osc.parse(b"/a\x00\x00,\x00\x00\x00"), [("/a", [])])

    def test_bundles(self):
        one, two = osc.message("/a", 1), osc.message("/b", "x")
        self.assertEqual(osc.parse(bundle()), [])                               # a bundle of nothing
        self.assertEqual(osc.parse(bundle(one, two)), [("/a", [1]), ("/b", ["x"])])
        nested = one
        for _ in range(osc.MAX_DEPTH):
            nested = bundle(nested)
        self.assertEqual(osc.parse(nested), [("/a", [1])])                     # MAX_DEPTH deep: read
        with self.assertRaises(osc.OscError):
            osc.parse(bundle(nested))                                           # one more: refused

    def test_the_size_limit(self):
        head = len(osc.message("/a", "")) - 4                                   # "/a" and ",s": 8 bytes
        packet = osc.message("/a", "x" * (osc.MAX_PACKET - head - 1))           # the string's NUL ends it
        self.assertEqual(len(packet), osc.MAX_PACKET)
        self.assertEqual(len(osc.parse(packet)), 1)
        over = osc.message("/a", "x" * (osc.MAX_PACKET - head))
        self.assertEqual(len(over), osc.MAX_PACKET + 4)
        with self.assertRaises(osc.OscError):
            osc.parse(over)

    def test_message_refuses_what_it_cannot_send(self):
        for bad in (None, b"bytes", [1], 1 + 2j):
            with self.assertRaises(TypeError):
                osc.message("/a", bad)


class RefusalTests(unittest.TestCase):
    def refused(self, packet, why=""):
        with self.subTest(why=why or packet[:24]), self.assertRaises(osc.OscError):
            osc.parse(packet)

    def test_malformed(self):
        good = osc.message("/a", 1, "s", 2.0)
        self.refused(b"", "empty")
        self.refused(bytearray(), "empty bytearray")
        self.refused(b"/a\x00", "not a multiple of 4")
        self.refused(b"/abc", "address without its NUL")
        self.refused(b"/a\x00x", "padding not NUL")
        self.refused(b"a\x00\x00\x00", "address without /")
        self.refused(b"/a\x00\x00i\x00\x00\x00" + bytes(4), "tags without ,")
        self.refused(b"/a\x00\x00,i\x00\x00", "int past the end")
        self.refused(b"/a\x00\x00,d\x00\x00" + bytes(4), "double past the end")
        self.refused(b"/a\x00\x00,s\x00\x00", "string past the end")
        self.refused(b"/a\x00\x00,i\x00\x00" + bytes(8), "bytes after the arguments")
        self.refused(b"/a\x00\x00,b\x00\x00" + bytes(4), "blob: not read")
        for tag in "tcrmINSb[]":
            self.refused(b"/a\x00\x00," + tag.encode() + b"\x00\x00" + bytes(8), f"type {tag}")
        self.refused(b"/\xff\x00\x00", "address not UTF-8")
        self.refused(b"/a\x00\x00,s\x00\x00\xc3\x28\x00\x00", "string not UTF-8")
        self.refused(good + bytes(4), "trailing zeros")

    def test_bundle_sizes(self):
        one = osc.message("/a", 1)
        self.refused(b"#bundle\x00" + bytes(4), "no time tag")
        for size in (0, -4, len(one) + 4, 3, 1 << 30):
            self.refused(b"#bundle\x00" + bytes(8) + struct.pack(">i", size) + one, f"element size {size}")
        self.refused(bundle(one) + b"\x00\x00\x00", "a size cut short")
        self.refused(bundle(b"/a\x00"), "element not a multiple of 4")
        self.refused(bundle(one, b"junk"), "a bad element spoils the bundle")

    def test_every_truncation_refused(self):
        packets = [osc.message("/mixlyzer/deck/time", 1, "C:/x.flac", True, False, 12.5),
                   bundle(osc.message("/a", 1), osc.message("/b", "xyz", 2.0))]
        for packet in packets:
            for cut in range(len(packet)):
                try:
                    got = osc.parse(packet[:cut])
                except osc.OscError:
                    continue
                # a cut on an element's boundary may still be a whole packet: then it is the leading part
                self.assertTrue(got == [] or got[0][0] in ("/a", "/mixlyzer/deck/time"), (cut, got))

    def test_random_corruption_only_refused(self):
        """Seeded fuzzing: flipped, inserted, deleted bytes and random packets parse or raise OscError only."""
        rng = random.Random(SEED)
        seeds = [osc.message("/mixlyzer/deck/samples", 2, "/m/a b.flac", 1, 0, 441000, 44100),
                 osc.message("/x", -5, 1.25, "\u00e9"), bundle(osc.message("/a"), bundle(osc.message("/b", True)))]
        for _ in range(4000):
            data = bytearray(rng.choice(seeds))
            for _ in range(rng.randint(1, 4)):
                kind = rng.random()
                at = rng.randrange(len(data) + 1)
                if kind < 0.5 and data:
                    data[min(at, len(data) - 1)] ^= 1 << rng.randrange(8)
                elif kind < 0.75:
                    data[at:at] = bytes(rng.randrange(256) for _ in range(rng.choice((1, 4))))
                elif data:
                    del data[at:at + rng.choice((1, 4))]
            try:
                osc.parse(bytes(data))
            except osc.OscError:
                pass
        for _ in range(2000):
            junk = bytes(rng.randrange(256) for _ in range(rng.choice((4, 8, 12, 16, 20, 64))))
            try:
                osc.parse(junk)
            except osc.OscError:
                pass


if __name__ == "__main__":
    unittest.main()
