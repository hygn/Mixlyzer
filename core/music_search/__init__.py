"""Search the library by music: a hummed, sung or whistled melody, or the song itself played from a phone or a
speaker, picked up by the microphone (or a file).

- A recording of the song itself: spectral-peak landmarks (`landmarks`, Wang 2003): exact, fast, robust to noise and
  a room, not to a changed tempo or pitch.
- A melody from a voice: its pitch (`query`) matched against two reference lines of every song (`melody`: the stereo
  centre's melody and the mix's top line) by a subsequence DTW cascade over all songs (`matcher`).
- Their index is `{libpath}/music_search.db` (`store`), apart from library.db: data made from the audio, rebuilt
  whenever its method changes, large (the landmarks), never the user's own. Filled in the background
  (core.workers.music_index).

`search.search_audio` is the one entry: landmarks first, the melody when the recording found no song.
"""
