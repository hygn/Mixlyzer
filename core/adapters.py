def normalize_gui_buffers(features: dict) -> dict:
    """GUI buffers derived from the analysis features.

    None at the moment: the key strip is rendered just in time from
    ``key_segments`` (utils.keystrip.render_keystrip); the ``key_np`` image that
    older libraries still hold in their NPZs is deprecated and never read.
    """
    return dict(features)
