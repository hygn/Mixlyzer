from __future__ import annotations

import os

from PySide6 import QtCore

from core.analysis_lib_handler import FeatureNPZStore
from core.library_handler import LibraryDB
from core.linear_segments import build_bpm_segments, build_key_segments
from utils.cue_points import build_cue_points_np, build_phrase_cue_points
from utils.keystrip import build_keystrip_buffer
from utils.phrases import extract_phrase_segments


class SaveWorker(QtCore.QObject):
    finished = QtCore.Signal(object)

    def save(self, store: FeatureNPZStore, libpath: str, uid, *, beatgrid=None, segments=None, key_segments=None, key_np=None, jump_cues_np=None, jump_cues_extracted=None, jump_cues_auto_np=None, phrase_segments_np=None):
        try:
            if uid is None:
                return
            prev_feat = store.load(uid)
            if beatgrid is not None:
                prev_feat["beats_time_sec"] = beatgrid
            if segments is not None:
                prev_feat["tempo_segments"] = segments
            if key_segments is not None:
                prev_feat["key_segments"] = key_segments
                if key_np is None:
                    duration = prev_feat.get("duration_sec", 0.0)
                    key_np = build_keystrip_buffer(key_segments, duration)
            if key_np is not None:
                prev_feat["key_np"] = key_np
            if jump_cues_np is not None:
                prev_feat["jump_cues_np"] = jump_cues_np
            if jump_cues_extracted is not None:
                prev_feat["jump_cues_extracted"] = jump_cues_extracted
            if jump_cues_auto_np is not None:
                # A JumpCUE reanalysis replaces the stored automatic result.
                for key in [k for k in prev_feat if k.startswith("jump_cues_auto_np.")]:
                    del prev_feat[key]
                prev_feat["jump_cues_auto_np"] = jump_cues_auto_np
            if phrase_segments_np is not None:
                prev_feat["phrase_segments_np"] = phrase_segments_np
                phrase_rows = extract_phrase_segments(
                    {"phrase_segments_np": phrase_segments_np}
                )
                prev_feat["cue_points_np"] = build_cue_points_np(
                    build_phrase_cue_points(phrase_rows)
                )
            store.save(uid, prev_feat)
            db = LibraryDB(os.path.join(libpath, "library.db"))
            db.connect()
            try:
                db.replace_bpm_segments(uid, build_bpm_segments(prev_feat.get("tempo_segments")))
                db.replace_key_segments(uid, build_key_segments(prev_feat.get("key_segments")))
            finally:
                db.close()
        except Exception:
            pass
        finally:
            self.finished.emit(uid)
        return
