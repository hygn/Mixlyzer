"""The library: the user's tracks and everything kept about them. One owner reads and writes it; the rest of the app
asks the owner. Where to look:

  for the rest of the app (the only modules imported outside core.library):
    service.py      LibraryService, the owner: its thread runs the jobs one at a time; read / write / call
                    LibrarySnapshot, what the GUI shows of the library, read again after writes
    jobs.py         every operation on the library, by name (jobs.track_tieups, jobs.store_analysis ...)
    records.py      the records passed around: TrackRow, PlaylistRow, IndexTrack, the readings' rule

  inside (used by jobs.py and service.py alone; every SQL statement and library file is here):
    db.py           library.db: tracks, readings, tag sources, playlists, segment tables
    tieups.py       library.db: the tie-up tables (what a tie-up is: core.tieup)
    variants.py     variants.db: the Variants' fingerprints and relations (what they are: core.variants)
    music_index.py  music_search.db: Search by Music's index (the search itself: core.music_search)
    thumbnails.py   thumbs/{uid}.jpg

  beside it, the same owner's: core.analysis_library, the tracks' stored analyses ({uid}.npz; its jobs.py the
  operations, its store.py the files)

tests/test_library_boundary.py fails when code outside the library (core.library, core.analysis_library) imports an
inside module or SQLite.
"""
