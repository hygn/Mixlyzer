from __future__ import annotations

import os

from PySide6 import QtCore

from core.analysis_lib_handler import FeatureNPZStore
from core.library_handler import LibraryDB
from core.linear_segments import build_bpm_segments, build_key_segments


def _replace_block(features: dict, name: str, block: dict) -> None:
    """Replace a stored block (flattened as ``name.<field>``) as a whole, so no
    field of the previous block is left behind."""
    for key in [k for k in features if k.startswith(name + ".")]:
        del features[key]
    features[name] = block


class SaveWorker(QtCore.QObject):
    finished = QtCore.Signal(object)

    def save(self, store: FeatureNPZStore, libpath: str, uid, *, beatgrid=None, segments=None, key_segments=None, jump_cues_np=None, jump_cues_auto_np=None, phrase_segments_np=None, cue_points_np=None):
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
            if jump_cues_np is not None:
                _replace_block(prev_feat, "jump_cues_np", jump_cues_np)
            if jump_cues_auto_np is not None:
                # a JumpCUE reanalysis replaces the stored automatic result
                _replace_block(prev_feat, "jump_cues_auto_np", jump_cues_auto_np)
            if phrase_segments_np is not None:
                _replace_block(prev_feat, "phrase_segments_np", phrase_segments_np)
            if cue_points_np is not None:
                _replace_block(prev_feat, "cue_points_np", cue_points_np)
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
