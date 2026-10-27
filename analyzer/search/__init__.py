"""Finding the library's songs by their sound; where the indexes are kept is the app's.

- music/: Search by Music, a song found by a hummed, sung or whistled melody or by the song itself played near the
  microphone: each song's index entry (made by the full analysis), the query's analysis, the matching, the
  searches joined over time while listening. Its index is kept in `{libpath}/music_search.db`
  (core.library.music_index).
- variants.py: the Variants, the library's tracks that are versions of one song (the same audio: another encode, copy
  or rate; another vocal; the instrumental / off vocal; an edit, a TV size; not arrangements, remixes or bootlegs:
  another song's form), found by a fingerprint of the analysis's beats and overview bands (no audio): the
  fingerprint, its index rows and queries, the verification and the relation it reads. Its store is
  `{libpath}/variants.db` (core.library.variants), the relations as the app uses them core.search.variants. The
  research behind it (566 tracks of version sets, 1351 tracks queried against each other: no different song called
  a version) is `research/benchmark/fingerprint_research` (not committed).
- mixability.py: the Mix Candidates' bars, where one track's end and another's start sound alike enough to blend
  (energy, rhythm and harmony per bar; the blend lined up on both tracks' phrases). Its store is
  `{libpath}/mixability.db` (core.library.mixability), the candidates as the app ranks them core.search.mixability.
"""
