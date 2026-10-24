"""Variants: the library's tracks that are versions of one song, found by a fingerprint of the analysis's stored
features (no audio decoded): the same audio (another encode, copy or rate), another singer's version, the
instrumental / off vocal, an edit (a TV size). Arrangements, remixes and bootlegs are another song's form and are
not variants.

- `fingerprint`: the song fingerprint (Philips-style bits on the beat grid's frames of the overview's 12 bands), its
  index rows and queries, the verification on the stored codes and the relation it reads.
- `store`: `{libpath}/variants.db`, the fingerprints, their hashes and the relations found (as Search by Music's
  index: data made from the analysis, rebuilt whenever its method changes, never the user's own). Written by the
  analysis and by Index Now (core.workers.music_index), each track queried against those already there.

The research behind it (566 tracks of version sets, 1351 tracks queried against each other: no different song
called a version) is `benchmark/fingerprint_research` (not committed).
"""
