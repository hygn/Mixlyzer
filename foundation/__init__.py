"""The app's lowest layer: what every other part stands on and that stands on nothing of the app's (no import of
core, analyzer_core, utils, ui, views, app; tests/test_layers.py holds it).

    resource_paths  the bundled files and the app's folder
    os_support      the operating system, detected once, and what differs by it
    i18n            the UI's language and its catalogs (tr)
    decoder         FFmpeg: audio decoded, a file's metadata and cover read
    model_files     a learned model's file: the user's, else the shipped default
    tempo_rows      tempo rows [start, end, bpm, inizio, ts, bpm_end]: steady rows and ramps, the beats they make
    beat_geometry   downbeats and bar positions from the beats and the tempo rows

The layers above, each importing only those below: foundation -> utils -> analyzer_core -> core -> ui / views ->
app.
"""
