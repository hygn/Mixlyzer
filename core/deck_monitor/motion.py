"""A deck's position between its messages, steady on screen, read from how its position moves (a position carried on
from the last message by the time since jumped back at each message, and shook where a deck stood still). The
positions heard since the last jump (a CUE, a loop, a seek: a step back, or forward farther than the time between
allows) and within WINDOW_SEC are fitted with a line, so the playhead moves at the deck's own speed and a message's
small lateness does not show; a position unchanged for STILL_SEC is a deck standing still, shown exactly where it
stands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

WINDOW_SEC = 0.5     # the positions a moving deck's line is fitted on
STILL_SEC = 0.03     # unchanged this long: standing still (rekordbox moves its position every 6 ms while it plays)
JUMP_SEC = 0.1       # a step this much beyond what the time between two positions allows: a jump
MAX_SPEED = 4.0      # the fastest a deck is taken to play (faster: a jump)
AHEAD_SEC = 0.25     # carried on past the last position heard at most this far (a sender gone quiet stops there)
KEPT_SEC = WINDOW_SEC + 0.1      # how long a receiver keeps the positions heard


@dataclass(frozen=True)
class Motion:
    position: float      # seconds into the track
    speed: float         # seconds of the track a second (0: standing still)


def motion(points: Sequence[tuple[float, float]], now: float) -> Optional[Motion]:
    """The deck at ``now`` (time.monotonic()), from its positions heard, ``points``: (when, seconds), the oldest
    first. None without any."""
    if not points:
        return None
    last_at, last = points[-1]
    run = [points[-1]]                       # the newest first
    for at, value in reversed(points[:-1]):
        later_at, later = run[-1]
        step, between = later - value, max(0.0, later_at - at)
        if last_at - at > WINDOW_SEC or step < 0.0 or step > MAX_SPEED * between + JUMP_SEC:
            break                            # too old, or a jump: the line starts after it
        run.append((at, value))
    if _still_since(run) <= last_at - STILL_SEC:
        return Motion(last, 0.0)
    run = _since_standing(run)
    if len(run) < 2:
        return Motion(last, 0.0)
    n = len(run)
    mean_t = sum(at - last_at for at, _ in run) / n
    mean_v = sum(v for _, v in run) / n
    var = sum((at - last_at - mean_t) ** 2 for at, _ in run)
    if var <= 1e-12:
        return Motion(last, 0.0)
    slope = sum((at - last_at - mean_t) * (v - mean_v) for at, v in run) / var
    speed = min(max(slope, 0.0), MAX_SPEED)
    at_last = mean_v - speed * mean_t        # the line at the last position's time
    return Motion(at_last + speed * min(max(0.0, now - last_at), AHEAD_SEC), speed)


def _still_since(run: list) -> float:
    """When the newest position was first heard, unchanged since (``run`` newest first)."""
    since, value = run[0]
    for at, v in run[1:]:
        if v != value:
            break
        since = at
    return since


def _since_standing(run: list) -> list:
    """``run`` (newest first) from the end of the last stretch it stood still for STILL_SEC, oldest first: a deck
    that starts moves from where it stood."""
    out = [run[0]]
    i = 1
    while i < len(run):
        j = i
        while j < len(run) and run[j][1] == run[i - 1][1]:
            j += 1
        if j > i and run[i - 1][0] - run[j - 1][0] >= STILL_SEC:
            break                            # it stood still there: its last point is where the moving began
        out.extend(run[i:max(j, i + 1)])
        i = max(j, i + 1)
    out.reverse()
    return out
