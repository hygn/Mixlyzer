"""Search the library by music: a hummed, sung or whistled melody, or the song itself played from a phone or a
speaker, picked up by the microphone (or a file).

- A recording of the song itself: spectral-peak landmarks (`landmarks`, Wang 2003): exact, fast, robust to noise and
  a room, not to a changed tempo or pitch.
- A melody from a voice: its pitch (`query`) matched against two reference lines of every song (`melody`: the stereo
  centre's melody and the mix's top line) by a subsequence DTW cascade over all songs (`matcher`).
- A song's index entry (`index`: its reference lines and landmarks) is made by the full analysis (analyzer.pipeline.track)
  and, for tracks analyzed before, by Index Now; the app keeps the entries (`{libpath}/music_search.db`,
  core.library.music_index) and lends them to the search.
- While listening, the searches of the last seconds are joined over time (`listening`).

`search.search_audio` is the one entry: landmarks first, the melody when the recording found no song.
"""
