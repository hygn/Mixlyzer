"""The player's audio engines: ``native`` (Rust, a process of its own; the default) and ``python`` (the
fallback, in the player's audio thread). core.playback.player picks one; both render the same way."""
