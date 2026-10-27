"""The audio setup on the console at the start ("[AudioEngine] ..." lines, into the log and the startup splash): the
engine that plays and the settings it was given. What the engine then opened (each device's API, mode, format,
period, buffer and latency, its threads' scheduling, the MIDI input) follows as the engine's own lines
(audioengine/native/src/log.rs). Logs are not translated."""
from __future__ import annotations

from audioengine.native import client
from core.controls import midi

_APIS = {"wasapi": "WASAPI", "cpal": "cpal", "jack": "JACK"}
_PRIORITIES = {"normal": "normal", "above_normal": "above normal", "high": "high"}


def _device(name: str) -> str:
    return f'"{name}"' if name else "the system's default device"


def _api(api: str, exclusive: bool, rate: int | None = None) -> str:
    """The API and mode; ``rate`` the exclusive rate asked for (0: the device's; None: not said)."""
    text = _APIS.get(api, api)
    if api == "wasapi":
        text += " exclusive" if exclusive else " low-latency shared"
        if exclusive and rate is not None:
            text += f" at {rate} Hz" if rate else " at the device's rate"
    return text


def _bus(name: str, out) -> str:
    if out.route == "off":
        return f"{name}: off"
    if out.route == "main_device":
        return f"{name}: channels {out.channels} of the master's device"
    rate = f"{out.rate} Hz" if out.rate else "its device's rate"
    return (f"{name}: {_device(out.device)} channels {out.channels}, {_api(out.audio_api, out.exclusive)}, {rate}, "
            f"buffer {out.buffer_ms} ms (its clock followed)")


def audio_setup_lines(cfg, native: bool) -> list[str]:
    """The lines for the settings ``cfg`` (its playbackconfig and deckconfig) with the engine that plays
    (``native``)."""
    pcfg, deck = cfg.playbackconfig, cfg.deckconfig
    stretch = "time stretch " + ("on" if deck.use_timestretch else "off (varispeed)")
    if not native:
        if pcfg.audio_engine == "python":
            why = "Settings > Audio > Engine"
        elif not client.available():
            why = "the native engine is not built"
        else:
            why = "the native engine did not start"
        return [f"Python engine ({why}): {_device(pcfg.output_device)}, buffer {pcfg.python_buffer_ms} ms, {stretch}; "
                "deck 1 on one output, no mixer, no MIDI"]
    exe = client.executable()
    mapped = len(midi.bindings(midi.mapping()))
    eq = "isolator" if deck.mixer_eq != "eq" else "EQ"
    return [
        f"native engine {exe.name if exe else ''}, protocol {client.VERSION}",
        f"master: {_device(pcfg.output_device)} channels {pcfg.output_main}, "
        f"{_api(pcfg.native_audio_api, pcfg.native_exclusive, pcfg.native_exclusive_rate)}, "
        f"buffer {pcfg.native_buffer_ms} ms, {stretch}",
        _bus("headphones", pcfg.output_headphones)
        + (f", the master mixed in (CUE / MASTER {deck.headphones_cue_mix} %)" if pcfg.headphones_master_mix else ""),
        _bus("booth", pcfg.output_booth),
        f"process: priority {_PRIORITIES.get(pcfg.native_priority, pcfg.native_priority)}, power throttling "
        + ("the system's choice" if pcfg.native_power_throttling else "off"),
        f"mixer: {eq}, channel fader curve {deck.fader_curve}, trim {deck.volume_trim_dbfs:+.1f} dB; last stages: "
        f"master {pcfg.output_clip}, headphones {pcfg.output_headphones.clip}, booth {pcfg.output_booth.clip}",
        "MIDI: " + (f'"{pcfg.midi_input}"' if pcfg.midi_input else "no input")
        + ("" if deck.midi_enabled else " (switched off)")
        + f", {mapped} messages mapped, jog {pcfg.midi_jog_ticks} ticks a turn in {deck.jog_mode.upper()} mode",
    ]


def print_audio_setup(cfg, native: bool) -> None:
    for line in audio_setup_lines(cfg, native):
        print(f"[AudioEngine] {line}")
