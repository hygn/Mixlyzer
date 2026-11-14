import copy
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Literal, Tuple, get_args, get_origin

from system.atomic_io import atomic_write_json


@dataclass
class libconfig:
    libpath: str
    # An empty list means the operating system's default Music location.
    local_music_directories: list[str]
    write_log: bool
    logpath: str
    rekordbox_sync_enabled: bool
    rekordbox_xml_path: str
    # How far (ms) the XML's steps of steady tempo standing in for a tempo ramp may put a beat from the
    # ramp's (0: a step per beat).
    rekordbox_ramp_tolerance_ms: float
    # The track's tie-ups (core.tieups.domain) after each track's comment in the XML: [Title OP; Other ED].
    rekordbox_comment_tieups: bool
    # Their works' titles in this language (core.tieups.domain.TITLE_LANGUAGES; "": tieup_language's).
    rekordbox_tieup_language: str
    # A tie-up added in the Edit tab added to the track's versions (its Variants) too.
    tieup_to_versions: bool
    # A new track its versions gave no tie-up looked up on Wikipedia / Wikidata (core.tieups.web; sends its title and
    # artist there).
    tieup_web_search: bool
    # The language the tie-ups' works are shown in (core.tieups.domain.TITLE_LANGUAGES; "": the UI's), and the one
    # the Sync tab writes them in after a file's comment ("": tieup_language's).
    tieup_language: str
    export_tieup_language: str
    # The editor's save writes the track's rows in library.db too (its BPM, key and segment tables, made from the marks
    # saved: core.library.segments.track_index); off, only its stored analysis (library.db brought up to it by Sync Now).
    sync_stored_analysis: bool
    # The library folder (system.os_support.path_key) whose warning of a sync client's folder the user turned off
    # ("": none; another folder warned again).
    cloud_sync_warning_off: str = ""
    # Serato's tags (the Sync tab's Write Tags, third_party.serato): its hot cues made of the JumpCUEs (A-H in their
    # slots: "jumpcues"), the CUE points (in time order: "cue_points") or both (the CUE points in the slots the
    # JumpCUEs leave: "both"); the file's own Serato hot cues kept in the slots left (else all replaced); the
    # beatgrid locked (BPMLOCK: Serato's analysis leaves it).
    serato_hot_cues: str = "jumpcues"
    serato_keep_cues: bool = True
    serato_bpm_lock: bool = True
    # rekordbox's phrases (the Sync tab, third_party.rekordbox_phrases): the mood they are written in ("high",
    # "mid", "low") and each label's phrase there ("LABEL=choice,...": rekordbox_phrases.map_text; "" or a label
    # left out: the mood's default)
    rekordbox_phrase_mood: str = "mid"
    rekordbox_phrase_map: str = ""
    # The local API (core.local_api): other programs on this computer asking analyses over HTTP on 127.0.0.1 at this
    # port, once paired (Settings > Library > Local Apps); this device's (a restored library keeps them as they are).
    local_api: bool = False
    local_api_port: int = 48700

    def rekordbox_xml(self) -> tuple:
        """What the Rekordbox XML is written with (a change of any writes it again whole)."""
        return (bool(self.rekordbox_sync_enabled), str(self.rekordbox_xml_path or "").strip(),
                float(self.rekordbox_ramp_tolerance_ms), bool(self.rekordbox_comment_tieups),
                str(self.rekordbox_tieup_language or ""))

    def rekordbox_xml_resync(self, previous: "libconfig") -> bool:
        """Whether a save from ``previous`` to these settings writes the XML again whole (AppWindow._on_settings_save
        alone decides it; the settings dialog's Sync Now shows the sync the save starts)."""
        return (previous.rekordbox_xml() != self.rekordbox_xml() and bool(self.rekordbox_sync_enabled)
                and bool(str(self.rekordbox_xml_path or "").strip()))


@dataclass
class phraselabelname:
    alias: str   # shown in place of the label ("" = the label)
    abbrev: str  # the overview's short form ("" = the first letter)


@dataclass
class phraselabelnames:
    """How each phrase label is shown (core.marks.phrases.PHRASE_LABELS, lowercased)."""
    lead_in: phraselabelname
    intro: phraselabelname
    verse: phraselabelname
    chorus: phraselabelname
    break_chorus: phraselabelname
    bridge: phraselabelname
    outro: phraselabelname
    interlude: phraselabelname
    silence: phraselabelname

    def by_label(self) -> dict[str, phraselabelname]:
        """Names keyed by phrase label (uppercase)."""
        return {f.name.upper(): getattr(self, f.name) for f in fields(self)}


@dataclass
class viewconfig:
    language: str  # the UI's (system.i18n.LANGUAGES; "" = the operating system's), applied at the next start
    display_waveform: bool
    display_beatgrid: bool
    display_keystrip: bool
    display_JumpCUE: bool
    display_phrase: bool
    # the overview's and the main view's BPM and keys as heard: the tempo changed, the key shifted (the deck's)
    played_tempo_key: bool
    highlight_changes: bool  # where an edit, undo or reanalysis changed the marks blinks (views.change_flash)
    waveform_style: Literal["rgb", "3band"]  # RGB: colored by the band mix; 3band: rekordbox-style layers
    # low / mid / high ranges (Hz) of the waveforms' colours: the overview and the preview sum them from the stored
    # bands when drawn, the main view waveform filters its audio with them
    env_lo: Tuple[float, float]
    env_mid: Tuple[float, float]
    env_hi: Tuple[float, float]
    # main view waveform: drawing frame (bar) length (ms) of the short views (up to
    # views.timeline.SHORT_WINDOW_MAX_SEC wide) and of the longer ones
    mainview_wave_frame_ms_short: float
    mainview_wave_frame_ms_long: float
    mainview_wave_frame_ms_close: float  # ... of the close views (up to views.timeline.CLOSE_WINDOW_MAX_SEC)
    mainview_wave_filter_order: int  # main view waveform: band filter order
    use_output_volume_as_peak_meter_input: bool
    fps: int
    reduce_fps_when_occluded: bool
    record_img_path: str
    phrase_labels: phraselabelnames


@dataclass
class deckconfig:
    """What playing changes, kept in deck.json (beside config.json, the settings that seldom change): saved as it
    changes (the toolbar, the deck's buttons, a controller, the editor's metronome) and applied at the start. The
    deck's momentary state (FX ON, the EQ and COLOR knobs, CUE) is not kept: every
    start begins with the effects off and the knobs in their middles."""
    use_timestretch: bool  # tempo changes keep the pitch (time stretch) instead of varispeed (the MT button)
    # the metronome (the editor's listening aids; Settings > Audio > Metronome)
    enable_metronome: bool
    metronome_wav_path: str
    metronome_offset_msec: float
    metronome_downbeat_volume_percent: int
    metronome_downbeat_pitch_semitones: float  # the click is played faster (higher) / slower
    metronome_beat_volume_percent: int
    metronome_beat_pitch_semitones: float
    metronome_ducking: bool  # lower the music under each click
    # the channel's TRIM (dB; its knob, Settings > Audio > Mixer), the tempo fader's range (+-%: 6, 10, 16; 100 WIDE:
    # 0 - 200 %), the MIDI input read or not (the toolbar's MIDI button)
    volume_trim_dbfs: float
    tempo_range: int
    midi_enabled: bool
    # the mixer's and the deck's switches: the EQ ("isolator": a band down to a kill; "eq": -26 dB at the bottom),
    # the channel fader's curve ("default", "steep": up fast, "slow"), the jog wheel's mode ("vinyl": its top holds
    # the record; "cdj": a bend)
    mixer_eq: str
    fader_curve: str
    jog_mode: str
    # the track info's time, switched by a click on it: elapsed / total, -remaining / total, elapsed / -remaining
    time_display: Literal["elapsed", "remaining", "elapsed_remaining"]
    # the master's level, a knob position 0 .. 1 (0.8 = 0 dB, client.LEVEL_UNITY)
    master_level: float
    # Sound Color FX as picked (not whether it is on): client.COLOR_TYPES' place (COLOR_OFF none), its PARAMETER
    # (0 .. 1)
    color_type: int
    color_param: float
    # Search by Music's microphone, by its name as the system lists it ("": the system's default input), picked in
    # the library's search bar
    music_input_device: str
    # the Mix Candidates' criteria (core.search.mixability.MixOptions), set at the end of the library's search
    # bar: the tracks to play before the loaded one or after it; B's BPM within mix_tempo_range % of A's, its key
    # compatible, its sound alike; all those on, or any
    mix_previous: bool
    mix_bpm: bool
    mix_key: bool
    mix_audio: bool
    mix_match_all: bool
    mix_tempo_range: float
    # the channel fader where it was last (0 .. 1: its position, the trim and its curve after it)
    volume_position: float
    # the Sync tab's choices as last made: Write Tags' fields checked (core.tags.writer's, "comment_tieups" the
    # tie-ups after the comment, "serato" Serato's tags), its backup ("" none, "tags", "file"); Rekordbox's XML and
    # phrases checked, the phrases' backup (third_party.rekordbox_phrases.BACKUP_KINDS)
    sync_tag_fields: list[str]
    tag_backup: str
    sync_rekordbox_xml: bool
    sync_rekordbox_phrases: bool
    phrase_backup: str
    # the Export tab's JAMS as last checked: the audio file's path and the track's uid written into it
    export_jams_path: bool
    export_jams_uid: bool
    # Deck Monitor (core.deck_monitor): where the decks are received while its window is open, set in that window:
    # the protocol ("osc": OSC Basic and Full, "prodjlink"); OSC's address and port (127.0.0.1 this computer only;
    # another address takes them from that network too); the network interface Pro DJ Link uses (its IPv4 address,
    # "": the first)
    monitor_protocol: Literal["osc", "prodjlink"]
    monitor_host: str
    monitor_port: int
    monitor_interface: str
    # the phrase now and next its decks show, by "section" (each phrase) or by "label" (a run of one label), all four
    # alike: clicked there
    monitor_phrase_unit: Literal["section", "label"]
    # a library track's JumpCUEs and CUE points when one of the sender's hot cues / memory cues differs (OSC Full, Pro
    # DJ Link): "mixlyzer" Mixlyzer's (the sender's where it has none) or "sender" all of the sender's, as the device's
    # buttons play them (none where it has none); set there
    monitor_jump_cues_first: Literal["mixlyzer", "sender"]
    monitor_cue_points_first: Literal["mixlyzer", "sender"]


@dataclass
class playbackconfig:
    # the master's last stage, keeping it at 0 dBFS at most (the native engine's mixer::Clip): a hard clip, a tanh
    # soft clip, or a limiter (1 ms late: its lookahead)
    output_clip: Literal["tanh", "hard", "limiter"]
    # the audio engine: "native" (audioengine/native; the Python one where it is not built or does not start)
    # or "python" (audioengine/python only); applied at the next start
    audio_engine: Literal["native", "python"]
    # each engine's output buffer (ms): audio queued ahead of what is heard, with the time stretch on or off
    native_buffer_ms: int
    python_buffer_ms: int
    output_device: str  # the output device's name as its engine lists it; "" = the system's default
    # the native engine's audio API: "wasapi" its own (Windows), "cpal" cpal's for the system (WASAPI's usual
    # shared stream, ALSA, CoreAudio), "jack" (Linux); one the system has not is cpal's
    native_audio_api: Literal["wasapi", "cpal", "jack"]
    native_exclusive: bool  # WASAPI's exclusive mode (the device to the app alone)
    native_exclusive_rate: int  # the rate in exclusive mode (0: the device's), from the next track loaded on
    # the native engine's priority (Windows its process's class; Linux its threads' real-time / nice; macOS its
    # thread's time constraint / QoS) and whether the system may power-throttle it (Windows EcoQoS; Linux timer
    # slack and uclamp; macOS App Nap and timer coalescing): audioengine/native/src/sched.rs
    native_priority: Literal["normal", "above_normal", "high"]
    native_power_throttling: bool
    # a MIDI controller (the native engine reads it; Settings > MIDI, its mapping in assets/midi): the input port
    # ("": none) and its jog wheel's messages per turn (33 1/3 rpm); whether it is read: deckconfig.midi_enabled
    midi_input: str
    midi_jog_ticks: int
    # the native engine's master: its channel pair on the device ("1-2")
    output_main: str


@dataclass
class analysisconfig:
    # The beat tracker reads every tempo in the octave from here (bpm_min .. 2 x bpm_min).
    bpm_min: int
    # Dynamic analysis: tempo changes within a track (off: one steady grid for the whole track).
    bpm_dynamic: bool
    # Dynamic downbeats of the static analysis: odd bars and meter changes (off: one bar length and
    # phase for the whole track); the dynamic analysis always has them.
    dynamic_downbeat: bool
    # The learned models' files (system.model_files: a missing file is the shipped default_<name>; the training,
    # training/, writes the user's): the onset model that places tempo changes, the beat tracker's frame TCN,
    # metronome critic, decoder costs (JSON, refitted with the beat model they belong to) and contextual downbeat
    # model, the phrase model, the JumpCUE pair models.
    refine_onset_parameter_path: str
    beat_tcn_parameter_path: str
    beat_critic_parameter_path: str
    beat_decoder_parameter_path: str
    downbeat_ctx_parameter_path: str
    phrase_parameter_path: str
    jumpcue_pairs_parameter_path: str
    # Dynamic key analysis: key changes within a track (off: one key for the whole track, the dynamic
    # analysis's key in the intro / outro / interludes).
    key_dynamic: bool
    # The chord-aware key model.
    key_parameter_path: str
    # AutoCUE: what the analysis puts at the LEAD_IN end, an early first
    # downbeat, or 0:00: a CUE point, a JumpCUE of its own, or nothing.
    autocue: Literal["off", "cue_point", "jumpcue"]
    # The automatic JumpCUE, one pair (analyzer.settings.JumpCueRules): the sections its earlier and its later point
    # may be in and the beats around a section still in it, the score it needs, its points' least gap in beats, the
    # beats after the first sung phrase's head kept clear, and a pair wholly on CUE points moved off them.
    jumpcue_earlier: Literal["first_chorus", "chorus", "any"]
    jumpcue_later: Literal["last_chorus", "chorus", "any"]
    jumpcue_section_margin_beats: int
    jumpcue_min_score: float
    jumpcue_min_gap_beats: int
    jumpcue_first_sung_beats: int
    jumpcue_avoid_cue_points: bool
    # Full analyses run at once (each a process of its own with threads of its own); the others wait their turn,
    # the deck's track first.
    max_parallel_analyses: int


@dataclass
class cuepointrule:
    at_start: bool               # a CUE point where a phrase of the label starts
    at_end: bool                 # ... and where it ends
    same_label_boundaries: bool  # also between two phrases of the label (else only next to another label)


@dataclass
class cuepointconfig:
    """CUE points the analysis sets from the phrases, by phrase label (the
    predefined labels, core.marks.phrases.PHRASE_LABELS, lowercased). LEAD_IN has no
    rule: where it ends is analysisconfig.autocue."""
    intro: cuepointrule
    verse: cuepointrule
    chorus: cuepointrule
    break_chorus: cuepointrule
    bridge: cuepointrule
    outro: cuepointrule
    interlude: cuepointrule
    silence: cuepointrule

    def by_label(self) -> dict[str, cuepointrule]:
        """Rules keyed by phrase label (uppercase)."""
        return {f.name.upper(): getattr(self, f.name) for f in fields(self)}


@dataclass
class searchconfig:
    """The library search by how titles / artists / albums read (core.search.reading)."""
    # the languages Han characters (without kana) are read in for the library's stored readings
    # (core.search.reading.HAN_LANGUAGES); a change reads the library again
    db_languages: list[str]
    # Han mixed with other characters found by its Korean Hanja reading too (when Korean Hanja is a
    # language): beside Latin letters / Hangul all of it (the per-language readings), beside kana its
    # runs of han_run_min (2 or more) Han characters, the rest in Japanese (Re:ゼロから始める異世界生活
    # ... isegyesaenghwal); for the stored readings (a change reads the library again) and the query
    han_runs: bool
    han_run_min: int
    # the languages a query's Han characters are read in
    query_languages: list[str]
    query_hangul_korean: bool    # a query's Hangul read as Korean (사회 sahoe: Korean titles, Korean Hanja readings)
    query_hangul_foreign: bool   # ... as Korean spells a foreign word (도쿄 tokyo: Japanese / English titles)
    # a query typed on the 2-set keyboard without the 한/영 key also read as what was meant (tkghl 사회, 무ㅐㅜㄷ anone)
    query_keyboard_layout: bool
    # a query of consonants alone (ㄷㅅㅁㄴ) a 초성 search on each name's initials; and their keys typed without the
    # 한/영 key (etas), its tracks found besides the others
    query_initials: bool
    query_initials_keys: bool
    # the library's names (title, artist, album) sorted as written, or by their readings (core.search.reading
    # sort_key): ABC, 가나다 (a Japanese title as Korean spells it), 五十音 (by its Japanese reading)
    name_sort: Literal["text", "abc", "hangul", "gojuon"]
    # Search by Music (analyzer.search.music): the analysis indexes each track it analyzes (about 6 s more a track);
    # listening, the last music_listen_window seconds are searched every music_listen_interval seconds
    music_index: bool
    music_listen_window: float
    music_listen_interval: float

    def han_run(self) -> int:
        """core.search.reading's ``han_run``: the Han run length read in Korean Hanja beside kana, 0: off."""
        return max(2, int(self.han_run_min)) if self.han_runs else 0

    def stored_readings(self) -> tuple:
        """What the library's stored readings depend on: a change reads the library again."""
        return tuple(self.db_languages), self.han_run()


@dataclass
class config:
    analysisconfig: analysisconfig
    libconfig: libconfig
    viewconfig: viewconfig
    playbackconfig: playbackconfig
    cuepointconfig: cuepointconfig
    searchconfig: searchconfig
    deckconfig: deckconfig   # kept in deck.json, not config.json

    def to_dict(self) -> dict:
        """config.json's content: the settings but the deck's (deck_dict)."""
        payload = asdict(self)
        payload.pop("deckconfig", None)
        return payload

    def deck_dict(self) -> dict:
        """deck.json's content."""
        return asdict(self.deckconfig)

    @classmethod
    def from_dict(cls, payload: dict, deck: dict | None = None):
        """From config.json's content and deck.json's (None: the defaults)."""
        merged = dict(_normalize_config_payload(payload))
        merged["deckconfig"] = deck if isinstance(deck, dict) else asdict(default_cfg().deckconfig)
        return _coerce_dataclass(cls, merged)


def search_settings(payload: dict | None) -> searchconfig:
    """The search settings of a config.json dict, the defaults filling what it lacks."""
    return _coerce_dataclass(searchconfig, {**asdict(default_cfg().searchconfig), **(payload or {})})


def _normalize_config_payload(payload: dict) -> dict:
    """Translate configuration keys from older Mixlyzer releases."""
    normalized = dict(payload) if isinstance(payload, dict) else {}
    raw_lib = normalized.get("libconfig")
    if not isinstance(raw_lib, dict):
        return normalized
    lib = dict(raw_lib)
    if "local_music_directories" not in lib and "local_music_directory" in lib:
        legacy = str(lib.get("local_music_directory") or "").strip()
        # ~/Music was the old hard-coded default. An empty list now asks the
        # OS for its Music location, including a redirected Windows folder.
        lib["local_music_directories"] = [] if legacy in ("", "~/Music") else [legacy]
    lib.pop("local_music_directory", None)
    normalized["libconfig"] = lib
    # the waveform band ranges moved from the analysis settings to the view settings
    analysis, view = normalized.get("analysisconfig"), normalized.get("viewconfig")
    if isinstance(analysis, dict) and isinstance(view, dict):
        analysis, view = dict(analysis), dict(view)
        for key in ("env_lo", "env_mid", "env_hi"):
            if key in analysis:
                view.setdefault(key, analysis.pop(key))
        normalized["analysisconfig"], normalized["viewconfig"] = analysis, view
    return normalized


def _coerce_dataclass(cls, payload: dict):
    src = payload if isinstance(payload, dict) else {}
    kwargs = {}
    for field in fields(cls):
        kwargs[field.name] = _coerce_value(field.type, src.get(field.name))
    return cls(**kwargs)


def _coerce_value(type_hint, value):
    origin = get_origin(type_hint)
    if is_dataclass(type_hint):
        return _coerce_dataclass(type_hint, value)
    if origin is Literal:
        options = get_args(type_hint)
        if value in options:
            return value
        return options[0] if options else value
    if origin in (tuple, Tuple):
        args = get_args(type_hint)
        raw_values = list(value) if isinstance(value, (list, tuple)) else []
        coerced = []
        for idx, arg in enumerate(args):
            raw = raw_values[idx] if idx < len(raw_values) else None
            coerced.append(_coerce_scalar(arg, raw))
        return tuple(coerced)
    if origin is list:
        (item_type,) = get_args(type_hint) or (object,)
        raw_values = value if isinstance(value, (list, tuple)) else []
        return [_coerce_scalar(item_type, item) for item in raw_values]
    return _coerce_scalar(type_hint, value)


def _coerce_scalar(type_hint, value):
    if type_hint is bool:
        return bool(value)
    if type_hint is int:
        try:
            return int(value)
        except Exception:
            return 0
    if type_hint is float:
        try:
            return float(value)
        except Exception:
            return 0.0
    if type_hint is str:
        return "" if value is None else str(value)
    return value



def default_cfg():
    lcfg = libconfig(
        libpath="library",
        local_music_directories=[],
        logpath="mixlyzer.log",
        write_log=False,
        rekordbox_sync_enabled=True,
        rekordbox_xml_path="./rekordbox.xml",
        rekordbox_ramp_tolerance_ms=1.0,
        rekordbox_comment_tieups=False,
        rekordbox_tieup_language="",
        tieup_to_versions=True,
        tieup_web_search=False,
        tieup_language="",
        export_tieup_language="",
        sync_stored_analysis=True,
        cloud_sync_warning_off="",
        serato_hot_cues="jumpcues",
        serato_keep_cues=True,
        serato_bpm_lock=True,
        rekordbox_phrase_mood="mid",
        rekordbox_phrase_map="",
        local_api=False,
        local_api_port=48700,
    )
    acfg = analysisconfig(
        bpm_min=100,
        bpm_dynamic=True,
        dynamic_downbeat=True,
        refine_onset_parameter_path="assets/weights/refine_onset_hgb.npz",
        beat_tcn_parameter_path="assets/weights/beat_tcn.npz",
        beat_critic_parameter_path="assets/weights/beat_critic.npz",
        beat_decoder_parameter_path="assets/weights/beat_decoder.json",
        downbeat_ctx_parameter_path="assets/weights/downbeat_ctx_hgb.npz",
        phrase_parameter_path="assets/weights/phrase_analyzer.npz",
        jumpcue_pairs_parameter_path="assets/weights/jumpcue_pairs.npz",
        key_dynamic=True,
        key_parameter_path="assets/weights/key_chord_cnn.npz",
        autocue="cue_point",
        jumpcue_earlier="first_chorus",
        jumpcue_later="last_chorus",
        jumpcue_section_margin_beats=4,
        jumpcue_min_score=0.9,
        jumpcue_min_gap_beats=48,
        jumpcue_first_sung_beats=20,
        jumpcue_avoid_cue_points=True,
        max_parallel_analyses=4,
    )
    vcfg = viewconfig(
        language="",
        display_waveform=True,
        display_beatgrid=True,
        display_keystrip=True,
        display_JumpCUE=True,
        display_phrase=True,
        played_tempo_key=False,
        highlight_changes=True,
        waveform_style="rgb",
        env_lo=(20.0, 200.0),
        env_mid=(250.0, 1500.0),
        env_hi=(1600.0, 11025.0),
        mainview_wave_frame_ms_short=1.0,
        mainview_wave_frame_ms_long=2.0,
        mainview_wave_frame_ms_close=0.5,
        mainview_wave_filter_order=4,
        use_output_volume_as_peak_meter_input=False,
        fps=60,
        reduce_fps_when_occluded=True,
        record_img_path="assets/images/vinyl.png",
        phrase_labels=phraselabelnames(**{
            f.name: phraselabelname(alias="", abbrev="") for f in fields(phraselabelnames)
        }),
    )
    dcfg = deckconfig(
        use_timestretch=True,
        enable_metronome=False,
        metronome_wav_path="assets/sound/click.wav",
        metronome_offset_msec=0.0,
        metronome_downbeat_volume_percent=60,
        metronome_downbeat_pitch_semitones=7.0,
        metronome_beat_volume_percent=60,
        metronome_beat_pitch_semitones=0.0,
        metronome_ducking=True,
        volume_trim_dbfs=-9.0,
        tempo_range=100,
        midi_enabled=True,
        mixer_eq="isolator",
        fader_curve="default",
        jog_mode="vinyl",
        time_display="elapsed",
        master_level=0.8,
        color_type=5,    # FILTER (client.COLOR_TYPES), as the engine starts
        color_param=0.5,
        music_input_device="",
        mix_previous=False,
        mix_bpm=True,
        mix_key=False,      # keys are a blend's colour, not a gate: the candidates are ranked by how they sound
        mix_audio=True,
        mix_match_all=True,
        mix_tempo_range=8.0,
        volume_position=1.0,
        sync_tag_fields=["title", "artist", "album", "comment"],
        tag_backup="tags",
        sync_rekordbox_xml=True,
        sync_rekordbox_phrases=False,
        phrase_backup="ext",
        export_jams_path=True,
        export_jams_uid=True,
        monitor_protocol="osc",
        monitor_host="127.0.0.1",
        monitor_port=39321,
        monitor_interface="",
        monitor_phrase_unit="section",
        monitor_jump_cues_first="mixlyzer",
        monitor_cue_points_first="mixlyzer",
    )
    pcfg = playbackconfig(
        output_clip="tanh",
        audio_engine="native",
        native_buffer_ms=20,
        python_buffer_ms=80,
        output_device="",
        native_audio_api="wasapi",
        native_exclusive=False,
        native_exclusive_rate=0,
        native_priority="high",
        native_power_throttling=False,
        midi_input="",
        midi_jog_ticks=720,
        output_main="1-2",
    )
    # CUE points at the start of an interlude and of the outro (not between two
    # of them)
    cue_rules = {
        f.name: cuepointrule(at_start=f.name in ("interlude", "outro"), at_end=False, same_label_boundaries=False)
        for f in fields(cuepointconfig)
    }
    # every language the readers have (core.search.reading.HAN_LANGUAGES)
    han_languages = ["ja", "zh", "yue", "ko-hanja", "vi", "zh-tw"]
    scfg = searchconfig(
        db_languages=list(han_languages),
        han_runs=True,
        han_run_min=3,
        query_languages=list(han_languages),
        query_hangul_korean=True,
        query_hangul_foreign=True,
        query_keyboard_layout=True,
        query_initials=True,
        query_initials_keys=True,
        name_sort="text",
        music_index=True,
        music_listen_window=10.0,
        music_listen_interval=2.0,
    )
    return config(
        libconfig=lcfg,
        analysisconfig=acfg,
        viewconfig=vcfg,
        playbackconfig=pcfg,
        cuepointconfig=cuepointconfig(**cue_rules),
        searchconfig=scfg,
        deckconfig=dcfg,
    )


def ui_language() -> str:
    """config.json's UI language (viewconfig.language), only read: set before anything else is done."""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            value = json.load(f).get("viewconfig", {}).get("language", "")
    except (OSError, ValueError, AttributeError):
        return ""
    return value if isinstance(value, str) else ""


CONFIG_FILE = "config.json"   # the settings that seldom change (the settings dialog's)
DECK_FILE = "deck.json"       # what playing changes (deckconfig)

# the settings files load_cfg() could not read, kept aside (their settings then reset to the defaults): (the file's
# name, where it was kept)
_unreadable_kept: list[tuple[str, Path]] = []


def unreadable_settings() -> list[tuple[str, Path]]:
    """The settings files that could not be read and were kept aside (their settings reset): (name, kept as)."""
    return list(_unreadable_kept)


def _keep_unreadable_config(error: Exception, name: str) -> None:
    """The unreadable settings file ``name`` moved aside (<name>.unreadable-<time>), so its settings are not lost."""
    kept = Path(f"{name}.unreadable-{datetime.now():%Y%m%d-%H%M%S}")
    try:
        os.replace(name, kept)
    except OSError as exc:
        print(f"[Config] {name} could not be read ({error}) nor kept aside ({exc})")
        return
    _unreadable_kept.append((name, kept.resolve()))
    print(f"[Config] {name} could not be read ({error}): kept as {kept}, its settings reset to the defaults")


def _merge_defaults(default_dict: dict, loaded_dict: dict) -> dict:
    merged = dict(default_dict)
    for key, val in loaded_dict.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_defaults(merged[key], val)
        else:
            merged[key] = val
    return merged


def restored_settings(current: config, payload: dict, deck: dict, *, libpath: str,
                      local_music_directories: list[str], rekordbox_xml_path: str) -> config:
    """A library package's settings (config.json's and deck.json's content) on this device: the package's, but for
    what describes this device, kept as it is here: the audio (playbackconfig), the microphone, Deck Monitor, the
    local API, and every file path (the log, the models, the metronome's sound, the record's image: a package
    never names where a file is written). The library is ``libpath``; the music folders and the Rekordbox XML are
    the restore's (joined to this device's folders by the user)."""
    defaults = default_cfg()
    new = config.from_dict(_merge_defaults(defaults.to_dict(), _normalize_config_payload(payload)),
                           _merge_defaults(defaults.deck_dict(), deck if isinstance(deck, dict) else {}))
    new.playbackconfig = copy.deepcopy(current.playbackconfig)
    for f in fields(analysisconfig):
        if f.name.endswith("_path"):
            setattr(new.analysisconfig, f.name, getattr(current.analysisconfig, f.name))
    new.viewconfig.record_img_path = current.viewconfig.record_img_path
    new.deckconfig.metronome_wav_path = current.deckconfig.metronome_wav_path
    new.deckconfig.music_input_device = current.deckconfig.music_input_device
    new.deckconfig.monitor_protocol = current.deckconfig.monitor_protocol
    new.deckconfig.monitor_host = current.deckconfig.monitor_host
    new.deckconfig.monitor_port = current.deckconfig.monitor_port
    new.deckconfig.monitor_interface = current.deckconfig.monitor_interface
    new.libconfig.local_api = current.libconfig.local_api
    new.libconfig.local_api_port = current.libconfig.local_api_port
    new.libconfig.libpath = libpath
    new.libconfig.logpath = current.libconfig.logpath
    new.libconfig.local_music_directories = list(local_music_directories)
    new.libconfig.rekordbox_xml_path = rekordbox_xml_path
    return new


def load_cfg() -> config:
    def _ensure_library_dir(cfg: config) -> None:
        lib_path = Path(cfg.libconfig.libpath)
        if not lib_path.exists():
            lib_path.mkdir(parents=True, exist_ok=True)

    defaults = default_cfg()
    loaded: dict = {}
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise TypeError("not a JSON object")
            loaded = _normalize_config_payload(loaded)
    except FileNotFoundError:
        loaded = {}
    except (TypeError, ValueError, AttributeError, KeyError) as exc:   # not JSON or UTF-8, or not settings
        _keep_unreadable_config(exc, CONFIG_FILE)
        loaded = {}
    deck = _load_deck(loaded)
    cfg = config.from_dict(_merge_defaults(defaults.to_dict(), loaded),
                           _merge_defaults(defaults.deck_dict(), deck))
    # written only where the file is not what it reads as (missing, unreadable, a key added or migrated): load_cfg
    # is called to read the settings, by the GUI and by worker processes at once, and a file replaced while
    # another reads it fails on Windows
    if _stored(CONFIG_FILE) != _as_json(cfg.to_dict()):
        save_cfg(cfg)
    if _stored(DECK_FILE) != _as_json(cfg.deck_dict()):
        save_deck(cfg)
    _ensure_library_dir(cfg)
    return cfg


def _load_deck(settings: dict) -> dict:
    """deck.json's content ({}: none, or not readable: kept aside). Where there is none yet, what config.json held
    of it before deck.json was kept (its playbackconfig's keys of deckconfig's), so a first start keeps them."""
    try:
        with open(DECK_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise TypeError("not a JSON object")
        return loaded
    except FileNotFoundError:
        playback = settings.get("playbackconfig")
        names = {f.name for f in fields(deckconfig)}
        return {k: v for k, v in playback.items() if k in names} if isinstance(playback, dict) else {}
    except (TypeError, ValueError, AttributeError, KeyError) as exc:
        _keep_unreadable_config(exc, DECK_FILE)
        return {}


def _stored(path: str) -> object:
    """A settings file's content as JSON, or None (missing, unreadable)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _as_json(payload: dict) -> object:
    """``payload`` as it reads back from its file (tuples as lists)."""
    return json.loads(json.dumps(payload, ensure_ascii=True))


def save_cfg(cfg: config) -> None:
    """config.json: the settings that seldom change (the settings dialog's)."""
    atomic_write_json(CONFIG_FILE, cfg.to_dict(), ensure_ascii=True, indent=None)


def save_deck(cfg: config) -> None:
    """deck.json: what playing changes (deckconfig), written as it changes."""
    atomic_write_json(DECK_FILE, cfg.deck_dict(), ensure_ascii=True, indent=1)
