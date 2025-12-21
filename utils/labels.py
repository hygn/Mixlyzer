CAMELOT_LABELS = [
    "8B","3B","10B","5B","12B","7B","2B","9B","4B","11B","6B","1B",
    "5A","12A","7A","2A","9A","4A","11A","6A","1A","8A","3A","10A"
]
CLASSICAL_LABELS = [
    "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
    "Cm","C#m","Dm","D#m","Em","Fm","F#m","Gm","G#m","Am","A#m","Bm"
]

KEY_DISPLAY_LABELS = [
    f"{cls} ({cam})" for cam, cls in zip(CAMELOT_LABELS, CLASSICAL_LABELS)
]

def idx_to_labels(idx:int) -> tuple[str,str]:
    i = int(idx) % 24
    return CAMELOT_LABELS[i], CLASSICAL_LABELS[i]
