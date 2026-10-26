"""The analysis library: each library track's stored analysis, {libpath}/{uid}.npz (its beat grid, tempo rows, keys,
phrases, JumpCUEs, CUE points, the waveform overview). Kept by the library's one owner (core.library.service, the
same thread as library.db's: an analysis stored or a track removed writes both at once); the rest of the app asks
the owner. Where to look:

  for the rest of the app:
    jobs.py         every operation on the stored analyses, by name (analysis_jobs.analysis, save_edits ...)
    stored.py       StoredAnalysis, a track's stored analysis as the app reads it (its fields, never the stored
                    keys: they are known here alone), TrackEdits (the editor's save), Reoptimize's annotations
    cue_points.py, jump_cues.py, phrases.py, jumpcue_colors.py
                    each mark's stored form, read and written as analyzer.music.marks' types (and the phrase labels'
                    and colour presets' tables)

  inside (used by the jobs and the owner alone; every read and write of a {uid}.npz file is here):
    store.py        FeatureNPZStore: the file format (flattened keys, NumPy values only, written atomically)

A process (the analysis, Reoptimize) never opens these files: it is given what it reads and hands back what is
stored. tests/test_library_boundary.py fails when code outside the library imports store.py.
"""
