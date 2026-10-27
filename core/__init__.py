"""The app's features, a package each:

    settings          config.json and deck.json (config.py)
    library           the library, its one owner and every operation on it (library.db, thumbnails, the indexes)
    analysis_library  each track's stored analysis (its NPZ) and the stored forms of its marks
    track             the deck's loaded track, its one owner (DataModel)
    playback          the player and its engines' feeder, External Sync (OSC)
    controls          a MIDI controller's mapping, the keyboard shortcuts' files
    search            the readings the library is searched by, the Variants' relations, the Mix Candidates
    tieups            tie-ups: their domain and the web lookup
    tags              a track file's tags and cover, read and written
    workers           the worker processes (analysis, reanalysis, Reoptimize, indexing)
    workflows         their lifecycle and queues on the GUI's side, the event bus, the task list
"""
