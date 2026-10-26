"""The analyzer: everything that reads a track's audio and finds its marks, in one directory. It imports no app code
(tests/test_analyzer_boundary.py): the app hands it a file path and its settings (settings.py: AnalysisSettings, every
model the file to read), it hands back typed results and reports its stages and refusals by id (the app words them).
Where to look:

    pipeline/     what is run: track.py the full analysis (analyze_track -> TrackAnalysis), the editor's partial
                  reanalyses each its own file (beatgrid.py a tempo segment, keys.py, phrases.py, jumpcues.py);
                  stages.py the stage ids they report
    audio/        the audio read: FFmpeg's decoder, the analyses' decode (load.py), HPSS, resampling, spectra,
                  conversions, the overview waveform's measurement
    music/        the marks' music: tempo rows (steady and ramps, the editor's grid moves), key rows, the beats and
                  downbeats they make
    beat/         the beat tracker (general/: the frame TCN, decoding, ramps, bars) and its frame features
    key/          the chord-aware key analyzer
    phrase/       phrases and their labels, the LEAD_IN, the CUE points' rules (cue_points.py)
    jumpcue/      JumpCUE detection (self-similarity, the jump point model), AutoCUE's JumpCUE
    settings.py   what an analysis is asked for; errors.py its refusals; files.py a model written whole
    runtime.py    the process: the operating system, console programs without a window, physical cores
"""
