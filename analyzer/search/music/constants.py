"""What the app's index and the search's modules share, importing nothing: the app reads these without loading the search,
whose kernels are numba's, until a search is asked for."""

SR, HOP = 22050, 256                # the analyzer's rate and hop: its melody extractor's frames
FRAMES = (0.4, 0.2, 0.1)            # the search cascade's frame sizes (s)
MELODY_METHOD = "center+mixtop-2"   # stored with the index: another method re-extracts every song
LANDMARKS_METHOD = "wang-1"
