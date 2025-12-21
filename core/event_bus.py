from PySide6 import QtCore
from core.library_handler import TrackRow
from typing import List

class EventBus(QtCore.QObject):
    # Player/Timeline
    sig_time_changed = QtCore.Signal(float)      # current time (sec)
    sig_duration_changed = QtCore.Signal(float)  # total duration (sec)
    sig_seek_requested = QtCore.Signal(float)    # seek(sec)
    sig_window_changed = QtCore.Signal(float)    # window_sec
    sig_center_changed = QtCore.Signal(float)    # center_t
    sig_playback_status = QtCore.Signal(bool)
    sig_transport_enabled = QtCore.Signal(bool)
    sig_volume_changed = QtCore.Signal(float)
    sig_tempo_factor_changed = QtCore.Signal(float)
    sig_tempo_mode_changed = QtCore.Signal(str)
    sig_stop_requested = QtCore.Signal(bool)
    sig_play_requested = QtCore.Signal(bool)
    sig_pause_requested = QtCore.Signal(bool)
    sig_scrub_begin  = QtCore.Signal()
    sig_scrub_update = QtCore.Signal(float)  # sec
    sig_scrub_end    = QtCore.Signal(float)  # sec

    # Data
    sig_features_loaded = QtCore.Signal()    # notification only; read from shared model
    sig_properties_loaded = QtCore.Signal()  # notification only; read from shared model
    sig_albumart_loaded = QtCore.Signal()    # notification only; read from shared model

    # Library
    sig_lib_updated = QtCore.Signal(object)
    sig_request_load_track = QtCore.Signal(str)
    sig_reanalyze_requested = QtCore.Signal(str)
    sig_segment_reanalyze_requested = QtCore.Signal(object)
    sig_segment_reanalyze_state = QtCore.Signal(bool)

    # Config Window
    sig_setting_saveJsonRequested = QtCore.Signal(object)

    # reload configs
    sig_reload_UI = QtCore.Signal(object)

    # Analysis Tasks
    # WorkersDialog expects these signals to exist and connects unconditionally.
    # Payloads:
    #   - sig_task_created(task_id:int, metadata:object)
    #   - sig_task_progress(task_id:int, progress:float, message:str)
    #   - sig_task_finished(task_id:int, error:str)
    sig_task_created = QtCore.Signal(int, object)
    sig_task_progress = QtCore.Signal(int, float, str)
    sig_task_finished = QtCore.Signal(int, str)

    # Editor signals
    sig_beatgrid_edited = QtCore.Signal()
    # Payload: None or (start_sec: float, end_sec: float)
    sig_key_selection_changed = QtCore.Signal(object)
    sig_key_segments_updated = QtCore.Signal()

    sig_jumpcue_updated = QtCore.Signal()

    sig_jump_arm = QtCore.Signal(object)
    sig_jump_disarm = QtCore.Signal()
