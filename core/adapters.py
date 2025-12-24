import numpy as np
from .model import GlobalParams
from utils.wave import build_wave_image, gen_img_mipmap
from utils.keystrip import build_keystrip_buffer

def normalize_gui_buffers(features: dict, gp: GlobalParams) -> dict:
    out = dict(features)

    # Generate Waveform Image
    if out.get("wave_img_np") is None:
        lo, mid, hi, min_env, max_env = out.get("lo_env"), out.get("mid_env"), out.get("hi_env"), out.get("min_env"), out.get("max_env")
        if lo is not None and mid is not None and hi is not None:
            img = build_wave_image(lo, mid, hi, min_env, max_env, height_px=128) # (H,T,3)
            out["wave_img_np"] = img.swapaxes(0, 1) # (T,H,3)
            len_t = img.shape[1]
            ds_scale = int(np.clip(len_t / 4096, 1, None))
            print(f"[Waveform] Time axis size of waveform = {len_t} px")
            print(f"[Waveform] Downscale factor for overview = {ds_scale}x ({int(len_t/ds_scale)} px)")
            out["wave_img_np_preview"] = gen_img_mipmap(img, ds_scale=ds_scale).swapaxes(0, 1)

    env_hop = out.get("env_hop_length", None)
    if env_hop is not None and out.get("lo_env") is not None:
        out["duration_sec"] = float(len(out["lo_env"]) * env_hop / gp.analysis_samp_rate)

    key_np = build_keystrip_buffer(
        out.get("key_segments"),
        out.get("duration_sec"),
    )
    if key_np is not None:
        out["key_np"] = key_np

    # tempo dynamic time axis
    td = out.get("tempo_dynamic")
    if td is not None:
        N = int(np.asarray(td).size)
        t_on = np.linspace(0, N * gp.bpm_hop_length / gp.analysis_samp_rate, N, endpoint=False)
        out["t_on"] = t_on

    return out
