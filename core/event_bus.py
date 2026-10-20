from PySide6 import QtCore

class EventBus(QtCore.QObject):
    # Player/Timeline
    sig_time_changed = QtCore.Signal(float)      # current time (sec)
    sig_duration_changed = QtCore.Signal(float)  # total duration (sec)
    sig_seek_requested = QtCore.Signal(float)    # seek(sec)
    sig_window_changed = QtCore.Signal(float)    # window_sec
    sig_center_changed = QtCore.Signal(float)    # center_t
    sig_playback_status = QtCore.Signal(bool)
    sig_transport_enabled = QtCore.Signal(bool)
    sig_external_sync_enabled = QtCore.Signal(bool)
    sig_external_sync_state = QtCore.Signal(object)
    sig_volume_changed = QtCore.Signal(float)
    sig_peak_meter_gain_changed = QtCore.Signal(float)
    sig_output_peak_dbfs = QtCore.Signal(float)
    # Output buffer: queued ms now, time-continuous observations since the last report
    # (PCMFeeder.take_buffer_stats: hist / seconds / mean_ms) or None, capacity ms
    sig_output_buffer = QtCore.Signal(float, object, float)
    sig_tempo_factor_changed = QtCore.Signal(float)
    sig_tempo_mode_changed = QtCore.Signal(str)
    sig_stop_requested = QtCore.Signal(bool)
    sig_play_requested = QtCore.Signal(bool)
    sig_pause_requested = QtCore.Signal(bool)
    sig_scrub_begin  = QtCore.Signal()
    sig_scrub_update = QtCore.Signal(float)  # sec
    sig_scrub_end    = QtCore.Signal(float)  # sec
    # Scratch: the main view's waveform held and dragged (the deck is silent while held)
    sig_scratch_begin  = QtCore.Signal(float)  # track seconds per physical screen pixel of the drag (0: unknown)
    sig_scratch_update = QtCore.Signal(float, float)  # sec, time.perf_counter() of the pointer event
    sig_scratch_end    = QtCore.Signal(float)  # sec

    # Data
    sig_features_loaded = QtCore.Signal()    # notification only; read from shared model
    sig_properties_loaded = QtCore.Signal()  # notification only; read from shared model
    sig_albumart_loaded = QtCore.Signal()    # notification only; read from shared model

    # Library
    sig_lib_updated = QtCore.Signal(object)
    sig_request_load_track = QtCore.Signal(str)
    sig_unload_track_requested = QtCore.Signal()
    sig_reanalyze_requested = QtCore.Signal(str)
    sig_segment_reanalyze_requested = QtCore.Signal(object)
    sig_segment_reanalyze_state = QtCore.Signal(bool)

    # Config Window
    sig_setting_saveJsonRequested = QtCore.Signal(object)
    sig_rekordbox_sync_requested = QtCore.Signal(object)
    sig_rekordbox_sync_track_requested = QtCore.Signal(str)
    sig_rekordbox_sync_started = QtCore.Signal(object)
    sig_rekordbox_sync_progress = QtCore.Signal(object, int, str)
    sig_rekordbox_sync_finished = QtCore.Signal(object, object)
    sig_rekordbox_sync_failed = QtCore.Signal(object, str)
    # the library's readings read again (Settings > Search)
    sig_reading_refresh_requested = QtCore.Signal(object)
    sig_reading_refresh_started = QtCore.Signal(object)
    sig_reading_refresh_progress = QtCore.Signal(object, int, str)
    sig_reading_refresh_finished = QtCore.Signal(object, object)
    sig_reading_refresh_failed = QtCore.Signal(object, str)

    # reload configs
    sig_reload_UI = QtCore.Signal(object)
    # the keyboard shortcuts saved in the settings (core.shortcuts keymap: action -> key sequences)
    sig_shortcuts_changed = QtCore.Signal(object)

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
    sig_key_segments_updated = QtCore.Signal()
    sig_tone_hold = QtCore.Signal(object)      # MIDI notes held under the music (empty: off)
    sig_phrase_segments_updated = QtCore.Signal()
    sig_cue_points_updated = QtCore.Signal()      # notification only; read from shared model

    sig_jumpcue_updated = QtCore.Signal()
    # The edit pages shown while editing is unlocked (frozenset of views.base.EDIT_PAGES; empty:
    # locked): the views emphasize those pages' visualizations and dim the others'.
    sig_edit_focus_changed = QtCore.Signal(object)

    # Deck MT button: playback.use_timestretch on/off (saved like the settings dialog)
    sig_timestretch_toggle_requested = QtCore.Signal(bool)
    sig_metronome_toggle_requested = QtCore.Signal(bool)

    sig_jump_arm = QtCore.Signal(object)
    sig_jump_disarm = QtCore.Signal()
    sig_jump_fired = QtCore.Signal()  # an armed jump was played

    # UI timing
    sig_ui_draw_interval = QtCore.Signal(float)
