import numpy as np

def speed_to_semitone(speed: float) -> tuple[float, int, float]:
    semitone = float(12 * np.log2(speed))
    nearest = int(round(semitone))
    frac = semitone - nearest
    return semitone, nearest, frac