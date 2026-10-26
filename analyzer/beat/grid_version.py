"""The version of the beat grids the analysis stores (`analyzer.beat.general.track` and what it reads): raised
whenever a change of the tracker moves the beats it gives. What is made from the stored grids carries it (the
Variants' fingerprint: analyzer.search.variants.METHOD), so a track analyzed by another tracker is not compared
with it. Kept apart from the tracker: the GUI reads it without loading the tracker."""

BEAT_GRID_VERSION = 1
