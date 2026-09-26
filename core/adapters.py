from utils.keystrip import build_keystrip_buffer

def normalize_gui_buffers(features: dict) -> dict:
    """GUI buffers derived from the analysis features (the key strip image)."""
    out = dict(features)

    key_np = build_keystrip_buffer(
        out.get("key_segments"),
        out.get("duration_sec"),
    )
    if key_np is not None:
        out["key_np"] = key_np

    return out
