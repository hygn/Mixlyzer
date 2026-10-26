"""The analyzer: everything that reads a track's audio and finds its marks, in one directory. Where to look:

    pipeline/     what is run: track.py the full analysis, the editor's partial reanalyses each its own file
                  (beatgrid.py a tempo segment, keys.py, phrases.py, jumpcues.py)
    audio/        the audio read: FFmpeg's decoder, HPSS, resampling, spectra, conversions
    music/        the marks' music: tempo rows (steady and ramps) and the beats and downbeats they make
    beat/         the beat tracker (general/: the frame TCN, decoding, ramps, bars) and its frame features
    key/          the chord-aware key analyzer
    phrase/       phrases and their labels, the LEAD_IN, the CUE points' rules
    jumpcue/      JumpCUE detection (self-similarity, the jump point model)
    runtime.py    the process's CPU facts (physical cores)
"""
