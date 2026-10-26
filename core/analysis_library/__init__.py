"""The analysis library: each library track's stored analysis, {libpath}/{uid}.npz (its beat grid, tempo rows, keys,
phrases, JumpCUEs, CUE points, the waveform overview). Kept by the library's one owner (core.library.service, the
same thread as library.db's: an analysis stored or a track removed writes both at once); the rest of the app asks
the owner. Below core.library (it imports nothing of it), above core.marks (the marks' vocabulary it stores).
Where to look:

  for the rest of the app:
    jobs.py         every operation on the stored analyses alone, by name (analysis_jobs.analysis ...); the ones
                    that write library.db too are core.library.jobs' (store_analysis, save_edits, remove_tracks)
    stored.py       StoredAnalysis, a track's stored analysis as the app reads it (its fields, never the stored
                    keys: they are known here alone), TrackEdits (the editor's save), Reoptimize's annotations
    and two conversions to the stored form: cue_points.as_stored (CUE points as they read back once stored),
    jump_cues.jump_cue_groups_of_pairs (the detector's JumpCUE pairs as the groups they are stored as)

  inside (used by the jobs, the owner and stored.py alone):
    cue_points.py, jump_cues.py, phrases.py
                    each mark's stored block, made from and read as analyzer.music.marks' types
    store.py        FeatureNPZStore: the file format (flattened keys, NumPy values only, written atomically)

A process (the analysis, Reoptimize) never opens these files: it is given what it reads and hands back what is
stored. tests/test_library_boundary.py fails when code outside the library uses more than the surface above, and
when this package imports core.library.
"""
