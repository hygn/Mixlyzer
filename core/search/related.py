"""The library's tracks related to the deck's track (the sources panel's Related Tracks): an artist they share, their
album, a tie-up's work. The Variants (core.search.variants) and the Mix Candidates (core.search.mixability) are the
others."""
from __future__ import annotations

from core.tieups.domain import name_key
from core.tieups.web import artist_names, song_title


def artist_keys(title: str, artist: str) -> frozenset[str]:
    """The artists a track credits, as two are compared (name_key): the artist field's names ("A feat. B", "A & B",
    a character's credit with its voice actor) and the title's featured ones (core.tieups.web)."""
    _song, featured = song_title(title or "")
    return frozenset(k for k in (name_key(n) for n in artist_names(artist or "", featured)) if k)


def album_key(album: str) -> str:
    """An album's name as two are compared ("" for none)."""
    return name_key(album)
