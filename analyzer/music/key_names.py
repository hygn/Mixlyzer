ALPHANUMERIC_LABELS = [
    "8B","3B","10B","5B","12B","7B","2B","9B","4B","11B","6B","1B",
    "5A","12A","7A","2A","9A","4A","11A","6A","1A","8A","3A","10A"
]
CLASSICAL_LABELS = [
    "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
    "Cm","C#m","Dm","D#m","Em","Fm","F#m","Gm","G#m","Am","A#m","Bm"
]

KEY_DISPLAY_LABELS = [
    f"{cls} ({num})" for num, cls in zip(ALPHANUMERIC_LABELS, CLASSICAL_LABELS)
]

def idx_to_labels(idx:int) -> tuple[str,str]:
    i = int(idx) % 24
    return ALPHANUMERIC_LABELS[i], CLASSICAL_LABELS[i]


# a major key's accidentals by its pitch class (C = 0): sharps positive, flats negative (an enharmonic key the one
# with fewer, F# major as 6 sharps)
MAJOR_ACCIDENTALS = (0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5)


def key_accidentals(idx: int) -> int:
    """The key signature of key ``idx`` (pitch + 12 * minor, as KEY_DISPLAY_LABELS): sharps positive, flats
    negative; a minor key its relative major's."""
    i = int(idx) % 24
    return MAJOR_ACCIDENTALS[(i % 12 + 3) % 12 if i >= 12 else i]
