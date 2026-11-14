"""core.local_api's tokens and pairing: a token's scopes bound by the hash Mixlyzer keeps (a scope added to a token is
refused), the pairing's states (closed, open, an app asking; its code typed right, wrong, refused, or its time run
out), one app at a time, the paired apps' file (read back; one that does not read moved aside), the request's checks
(the Host header, a browser's headers, a bearer token)."""
import json
import os
import tempfile
import unittest

from core.local_api import tokens
from core.local_api.pairings import CODE_TRIES, PAIRING_SEC, PairingStore, Phase, Refused, Verdict, clean_name
from core.local_api.protocol import Request, from_browser, host_allowed, json_media_type


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def forged(token: str, scopes: list[str]) -> str:
    """``token`` with its payload's scopes changed (its secret kept)."""
    prefix, _payload, secret = token.split(".")
    token_id = tokens.parse(token)[0]
    payload = json.dumps({"id": token_id, "scopes": scopes}, separators=(",", ":"), sort_keys=True)
    return f"{prefix}.{tokens._b64(payload.encode('ascii'))}.{secret}"


class TokenTests(unittest.TestCase):
    def test_a_token_says_its_scopes(self):
        issued = tokens.issue([tokens.WRITE, tokens.READ, tokens.READ])
        self.assertEqual(issued.scopes, (tokens.READ, tokens.WRITE))
        self.assertEqual(tokens.parse(issued.token), (issued.token_id, issued.scopes))
        self.assertEqual(tokens.digest(issued.token), issued.sha256)
        self.assertLessEqual(len(issued.token), tokens.MAX_TOKEN_CHARS)
        self.assertNotEqual(tokens.issue([tokens.READ]).token, tokens.issue([tokens.READ]).token)

    def test_not_tokens(self):
        issued = tokens.issue([tokens.READ])
        prefix, payload, secret = issued.token.split(".")
        for text in ("", "Bearer", "mxl1..", f"mxl2.{payload}.{secret}", f"{prefix}.{payload}", f"{prefix}.{payload}.ab",
                     f"{prefix}.!!.{secret}", issued.token + "x" * 300, issued.token.replace("m", "ｍ"), None, 5):
            with self.subTest(text=text):
                self.assertIsNone(tokens.parse(text))
        self.assertIsNone(tokens.parse(forged(issued.token, ["write", "read"])))     # not in their order
        self.assertIsNone(tokens.parse(forged(issued.token, ["admin"])))
        with self.assertRaises(ValueError):
            tokens.issue(["admin"])


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "pairings.json")
        self.clock = Clock()
        self.changes = 0
        self.store = PairingStore(self.path, clock=self.clock, on_change=self.changed)
        self.store.load()

    def tearDown(self):
        self.dir.cleanup()

    def changed(self):
        self.changes += 1

    def paired(self, name="Tool", scopes=("read",)):
        self.store.open()
        asking = self.store.ask(name, list(scopes))
        self.assertEqual(self.store.enter(asking.code), Verdict.PAIRED)
        return asking

    def test_closed_until_the_user_opens_it(self):
        with self.assertRaises(Refused) as caught:
            self.store.ask("Tool", ["read"])
        self.assertEqual(caught.exception.code, "pairing_closed")
        self.store.open()
        self.assertIs(self.store.view().phase, Phase.OPEN)
        asking = self.store.ask("  My\x00 Tool​ ", ["read"])
        self.assertEqual(asking.name, "My Tool")
        self.assertRegex(asking.code, r"^\d{4}$")
        self.assertIs(self.store.view().phase, Phase.ASKED)
        self.assertGreater(self.changes, 0)

    def test_the_right_code_pairs_the_token_and_only_it(self):
        asking = self.paired(scopes=("read", "write"))
        token = asking.issued.token
        self.assertEqual(self.store.check(token)[0], Verdict.PAIRED)
        grant = self.store.authenticate(token)
        self.assertEqual((grant.name, grant.scopes), ("Tool", ("read", "write")))
        self.assertIs(self.store.view().phase, Phase.CLOSED)        # one answer per opening
        self.assertEqual(self.store.view().last, (Verdict.PAIRED, "Tool"))
        other = tokens.issue(["read"]).token
        self.assertIsNone(self.store.authenticate(other))
        self.assertEqual(self.store.check(other)[0], Verdict.UNKNOWN)

    def test_a_scope_added_to_a_token_is_refused(self):
        token = self.paired(scopes=("read",)).issued.token
        self.assertIsNone(self.store.authenticate(forged(token, ["read", "write"])))
        self.assertIsNotNone(self.store.authenticate(token))

    def test_one_app_at_a_time(self):
        self.store.open()
        first = self.store.ask("First", ["read"])
        with self.assertRaises(Refused) as caught:
            self.store.ask("Second", ["read"])
        self.assertEqual(caught.exception.code, "pairing_busy")
        self.assertEqual(self.store.check(first.issued.token)[0], Verdict.PENDING)

    def test_wrong_codes_then_refused(self):
        self.store.open()
        asking = self.store.ask("Tool", ["read"])
        wrong = f"{(int(asking.code) + 1) % 10000:04d}"
        for _ in range(CODE_TRIES - 1):
            self.assertEqual(self.store.enter(wrong), Verdict.PENDING)
        self.assertEqual(self.store.enter(wrong), Verdict.DENIED)
        self.assertEqual(self.store.enter(asking.code), Verdict.DENIED)       # too late
        self.assertEqual(self.store.check(asking.issued.token)[0], Verdict.DENIED)
        self.assertIsNone(self.store.authenticate(asking.issued.token))

    def test_refused_by_the_user(self):
        self.store.open()
        asking = self.store.ask("Tool", ["read"])
        self.store.refuse()
        self.assertEqual(self.store.check(asking.issued.token)[0], Verdict.DENIED)
        self.store.open()
        asking = self.store.ask("Tool", ["read"])
        self.store.close()
        self.assertEqual(self.store.check(asking.issued.token)[0], Verdict.DENIED)

    def test_its_time_runs_out(self):
        self.store.open()
        self.clock.now += PAIRING_SEC + 1
        self.assertTrue(self.store.expire_due())
        self.assertIs(self.store.view().phase, Phase.CLOSED)
        self.store.open()
        asking = self.store.ask("Tool", ["read"])
        self.clock.now += PAIRING_SEC + 1
        self.assertEqual(self.store.enter(asking.code), Verdict.EXPIRED)
        self.assertEqual(self.store.check(asking.issued.token)[0], Verdict.EXPIRED)
        self.clock.now += 3600
        self.assertEqual(self.store.check(asking.issued.token)[0], Verdict.UNKNOWN)   # its outcome forgotten

    def test_what_an_app_may_ask(self):
        self.store.open()
        for name, scopes in (("", ["read"]), ("x" * 65, ["read"]), (5, ["read"]), ("Tool", ["write"]),
                             ("Tool", ["admin"]), ("Tool", "read"), ("Tool", 7)):
            with self.subTest(name=name, scopes=scopes):
                with self.assertRaises(Refused) as caught:
                    self.store.ask(name, scopes)
                self.assertEqual(caught.exception.code, "bad_request")
        self.assertIs(self.store.view().phase, Phase.OPEN)
        self.assertEqual(clean_name("テスト\tの　道具"), "テスト の 道具")

    def test_kept_in_the_file_and_unpaired(self):
        token = self.paired(name="道具").issued.token
        with open(self.path, encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn(token.split(".")[2], text)            # its secret never written
        self.assertTrue(text.isascii())
        again = PairingStore(self.path, clock=self.clock)
        again.load()
        self.assertEqual([p.name for p in again.pairings()], ["道具"])
        self.assertEqual(again.authenticate(token).name, "道具")
        self.assertTrue(again.unpair(again.pairings()[0].token_id))
        self.assertIsNone(again.authenticate(token))
        third = PairingStore(self.path)
        third.load()
        self.assertEqual(third.pairings(), [])

    def test_a_file_that_does_not_read_is_moved_aside(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{not json")
        store = PairingStore(self.path)
        store.load()
        self.assertEqual(store.pairings(), [])
        self.assertTrue(store.problem)
        self.assertFalse(os.path.exists(self.path))
        self.assertTrue(any(n.startswith("pairings.json.unreadable-") for n in os.listdir(self.dir.name)))


class ProtocolTests(unittest.TestCase):
    def test_host(self):
        for host in ("127.0.0.1:48700", "localhost:48700", "LOCALHOST:48700", "[::1]:48700"):
            self.assertTrue(host_allowed(host, 48700), host)
        for host in ("127.0.0.1", "127.0.0.1:80", "evil.example:48700", "127.0.0.1.nip.io:48700", ""):
            self.assertFalse(host_allowed(host, 48700), host)

    def test_browser_headers_and_media_type(self):
        self.assertTrue(from_browser({"origin": "null"}))
        self.assertTrue(from_browser({"sec-fetch-site": "cross-site"}))
        self.assertFalse(from_browser({"user-agent": "Mozilla/5.0"}))
        self.assertTrue(json_media_type("application/json; charset=utf-8"))
        self.assertFalse(json_media_type("text/plain"))

    def test_bearer(self):
        self.assertEqual(Request("GET", "/", {"authorization": "Bearer abc"}).bearer(), "abc")
        self.assertEqual(Request("GET", "/", {"authorization": "bearer  abc "}).bearer(), "abc")
        self.assertIsNone(Request("GET", "/", {"authorization": "Basic abc"}).bearer())
        self.assertIsNone(Request("GET", "/", {"authorization": "Bearer "}).bearer())
        self.assertIsNone(Request("GET", "/", {}).bearer())


if __name__ == "__main__":
    unittest.main()
