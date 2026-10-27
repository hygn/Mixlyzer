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
    sig_peak_meter_gain_changed = QtCore.Signal(float)
    sig_output_peak_dbfs = QtCore.Signal(float)
    # Output buffer: queued ms now, time-continuous observations since the last report
    # (PCMFeeder.take_buffer_stats: hist / seconds / mean_ms) or None, capacity ms
    sig_output_buffer = QtCore.Signal(float, object, float)
    # the deck's tempo as the player reports it (1: 100 %); notification of a change
    sig_tempo_factor_changed = QtCore.Signal(float)
    sig_scrub_begin  = QtCore.Signal()
    sig_scrub_update = QtCore.Signal(float)  # sec
    sig_scrub_end    = QtCore.Signal(float)  # sec
    # Scratch: the main view's waveform held and dragged (the deck is silent while held)
    sig_scratch_begin  = QtCore.Signal(float)  # track seconds per physical screen pixel of the drag (0: unknown)
    sig_scratch_update = QtCore.Signal(float, float)  # sec, time.perf_counter() of the pointer event
    sig_scratch_end    = QtCore.Signal(float)  # sec

    # The loaded track and its marks as edited are the model's (core.track.model: DataModel.sig_loaded,
    # sig_beatgrid_changed, ...), not the bus's.

    # Library
    sig_lib_updated = QtCore.Signal(object)
    sig_request_load_track = QtCore.Signal(str)   # loaded onto the deck (played at once, analyzed meanwhile)
    sig_request_analyze = QtCore.Signal(str)      # analyzed into the library, not loaded
    sig_unload_track_requested = QtCore.Signal()
    # a track's full reanalysis: its path and analysis settings for this run only (analysisconfig fields; None:
    # the saved settings)
    sig_reanalyze_requested = QtCore.Signal(str, object)
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
    # library.db made to agree with the stored analyses (Settings > Library's Sync Now: core.workflows.library_sync)
    sig_library_sync_requested = QtCore.Signal()
    sig_library_sync_progress = QtCore.Signal(int, int)        # tracks done, of
    sig_library_sync_finished = QtCore.Signal(object)          # core.library.jobs.IndexSync
    sig_library_sync_failed = QtCore.Signal(str)
    # the library packed into a file / restored from one (Settings > Library's Backup: core.workflows.library_backup)
    sig_library_backup_requested = QtCore.Signal(bool)         # the feature caches packed too
    sig_library_restore_requested = QtCore.Signal()
    # the library's readings read again (Settings > Search)
    sig_reading_refresh_requested = QtCore.Signal(object)
    sig_reading_refresh_started = QtCore.Signal(object)
    sig_reading_refresh_progress = QtCore.Signal(object, int, str)
    sig_reading_refresh_finished = QtCore.Signal(object, object)
    sig_reading_refresh_failed = QtCore.Signal(object, str)
    # Search by Music: Index Now (Settings > Search) and its run
    sig_music_index_requested = QtCore.Signal(object)
    sig_music_index_started = QtCore.Signal(object)
    sig_music_index_progress = QtCore.Signal(object, int, str)
    sig_music_index_finished = QtCore.Signal(object, object)
    sig_music_index_failed = QtCore.Signal(object, str)

    # reload configs
    sig_reload_UI = QtCore.Signal(object)
    # the keyboard shortcuts saved in the settings (core.controls.shortcuts keymap: action -> key sequences)
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
    sig_tone_hold = QtCore.Signal(object)      # MIDI notes held under the music (empty: off)
    # the phrase labels' names shown changed (Settings > View): the phrases are drawn again, not changed
    sig_phrase_labels_changed = QtCore.Signal()
    # The edit pages shown while editing is unlocked (frozenset of views.base.EDIT_PAGES; empty:
    # locked): the views emphasize those pages' visualizations and dim the others'.
    sig_edit_focus_changed = QtCore.Signal(object)

    # Deck MT button: playback.use_timestretch on/off (saved like the settings dialog)
    sig_timestretch_toggle_requested = QtCore.Signal(bool)
    sig_metronome_toggle_requested = QtCore.Signal(bool)

    # The deck (the native engine's src/deck.rs, or the Python engine's audioengine/python/deck.py): requests from
    # the GUI, combined there with a MIDI controller's; what the deck then is comes back as sig_deck_state (a
    # DeckReport: playing, factor, volume, cue s, jump_dest s, lamps, tempo_range, midi_state), which the GUI shows
    sig_deck_action = QtCore.Signal(int, bool)   # client.DECK_ACTIONS code, pressed (True) / let go
    sig_tempo_requested = QtCore.Signal(float)   # the tempo fader moved: factor
    sig_volume_requested = QtCore.Signal(float)   # the volume fader moved: 0 .. 1
    # the player's channel knob turned: 0 TRIM, 1 EQ LOW, 2 MID, 3 HIGH; its position 0 .. 1 (the middle 0 dB)
    sig_knob_requested = QtCore.Signal(int, float)
    # the mixer's setting asked for (the toolbar's switches, levels and Sound Color FX): client.MIX_KEYS' key, its value (the
    # native engine's mixer; what it then is comes back in the deck's report, DeckReport.mix)
    sig_mixer_requested = QtCore.Signal(str, float)
    sig_jog_mode_requested = QtCore.Signal(bool)   # the jog wheel's mode asked for: VINYL (True) / CDJ
    sig_channel_cue_requested = QtCore.Signal(bool)   # the deck's channel's headphones CUE asked for (on / off)
    sig_jump_to_requested = QtCore.Signal(float)   # a JumpCUE's JUMP: its point (s)
    sig_call_requested = QtCore.Signal(float)   # a CUE point's CALL: its time (s)
    # the track's marks as edited: (beats s, downbeat flags, CUE points s, JumpCUEs A-Z s or NaN)
    sig_deck_marks = QtCore.Signal(object)
    sig_deck_state = QtCore.Signal(object)
    # the native audio engine started again by hand (Settings > Audio > Engine), and whether it did
    sig_engine_restart_requested = QtCore.Signal()
    sig_engine_restarted = QtCore.Signal(bool)
    # MIDI messages from the controller (status, data1, data2, flags, action): the GUI's actions, and all while
    # learning
    sig_midi_events = QtCore.Signal(object)
    # the MIDI mapping or input saved in the settings
    sig_midi_changed = QtCore.Signal()

    # UI timing
    sig_ui_draw_interval = QtCore.Signal(float)
