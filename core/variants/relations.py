"""The variants' relations as the app uses them: a pair of tracks found to be versions of one song (`Pair`, as the
library keeps it: core.library.variants), and the tracks joined by them into songs (`groups`)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pair:
    a: str
    b: str
    kind: str          # fingerprint.relation's: b's to a
    cover_a: float
    cover_b: float
    full: float


def groups(found: list[Pair], uids: set[str]) -> list[list[str]]:
    """The tracks of ``uids`` joined by relations into groups (versions of one song), two tracks or more each."""
    parent: dict[str, str] = {}

    def root(u: str) -> str:
        while parent.setdefault(u, u) != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for p in found:
        if p.a in uids and p.b in uids:
            ra, rb = root(p.a), root(p.b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    members: dict[str, list[str]] = {}
    for u in parent:
        members.setdefault(root(u), []).append(u)
    return [sorted(g) for g in members.values() if len(g) > 1]
