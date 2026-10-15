from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np


@dataclass
class Segment:
    """A run of rendered output frames and the track position they carry."""
    out_start: int     # first output frame (counted since the sink was started)
    frames: int
    in_start: float    # input frame at out_start
    step: float        # input frames per output frame; 0.0 = silence holding in_start
    peak_dbfs: float


class OutputTimeline:
    """
    Everything rendered for one sink session, in output frames.

    Rendered audio waits in `pending` until the device accepts it. The frame being heard is
    written - queued, where queued is what the device still holds (bufferSize - bytesFree);
    segments map that output frame back to the track timeline at the speed it was rendered.
    """

    def __init__(self, bytes_per_frame: int):
        self.bpf = int(bytes_per_frame)
        self.pending = bytearray()
        self.written_bytes = 0
        self.rendered_frames = 0
        self.segments: Deque[Segment] = deque()

    def reset(self) -> None:
        self.pending.clear()
        self.written_bytes = 0
        self.rendered_frames = 0
        self.segments.clear()

    def append(self, out: np.ndarray, in_start: float, step: float, peak_dbfs: float) -> None:
        n = int(out.shape[0])
        self.segments.append(Segment(self.rendered_frames, n, float(in_start), float(step), peak_dbfs))
        self.rendered_frames += n
        self.pending += out.astype("<f4", copy=False).tobytes()

    def write_to(self, dev) -> bool:
        """Push pending audio into the device; False while some of it is still waiting."""
        if not self.pending:
            return True
        try:
            written = int(dev.write(bytes(self.pending)))
        except Exception:
            written = 0
        if written > 0:
            del self.pending[:written]
            self.written_bytes += written
        return not self.pending

    def heard_frame(self, queued_bytes: int) -> float:
        """Output frame currently leaving the device."""
        return max(0.0, (self.written_bytes - queued_bytes) / float(self.bpf))

    def segment_at(self, out_frame: float) -> Optional[Segment]:
        for seg in reversed(self.segments):
            if seg.out_start <= out_frame:
                return seg
        return None

    def hold_silence_at(self, in_start: float) -> None:
        """The silence queued while paused holds ``in_start``: the paused position
        moved, and what is heard until new audio follows that silence is the new
        position, not the one it was rendered at."""
        for seg in self.segments:
            if seg.step == 0.0:
                seg.in_start = float(in_start)

    def first_segment(self) -> Optional[Segment]:
        return self.segments[0] if self.segments else None

    def prune(self, heard_frame: float) -> None:
        """Drop segments that finished playing, keeping the one being heard."""
        while len(self.segments) > 1 and self.segments[1].out_start <= heard_frame:
            self.segments.popleft()
