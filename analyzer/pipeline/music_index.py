"""Search by Music's index entry of a track analyzed before it was indexed (Index Now): its audio decoded and indexed
as the full analysis indexes it (which reuses its own HPSS mask and melody contours; this makes its own)."""

from analyzer.audio.load import fast_load
from analyzer.search.music.index import SR, IndexEntry, index_audio


def index_track(path: str) -> IndexEntry:
    return index_audio(fast_load(path, target_sr=SR, stereo=True))
