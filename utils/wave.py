import numpy as np
from scipy.ndimage import gaussian_filter1d

def downsample_blur_stride(img: np.ndarray, factor: int, sigma: float = 0.8):
    if factor <= 1:
        return img
    blurred = gaussian_filter1d(img.astype(float), sigma=sigma*factor, axis=1)
    return blurred[:, ::factor, :].astype(np.uint8)

def apply_contrast_stretch(img, low_p=10, high_p=98):
    imgf = img.astype(np.float32) / 255.0
    lo, hi = np.percentile(imgf, [low_p, high_p])
    imgf = np.clip((imgf - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (imgf * 255.0 + 0.5).astype(np.uint8)

def build_wave_image(lo, mid, hi, min_env, max_env, height_px=110, downsample=4, white_threshold=0.05):
    print("[Render] Rendering Waveform image")

    lo = np.nan_to_num(np.asarray(lo, dtype=float), nan=0.0)
    mid = np.nan_to_num(np.asarray(mid, dtype=float), nan=0.0)
    hi  = np.nan_to_num(np.asarray(hi,  dtype=float), nan=0.0)
    min_env  = np.nan_to_num(np.asarray(min_env,  dtype=float), nan=0.0)
    max_env  = np.nan_to_num(np.asarray(max_env,  dtype=float), nan=0.0)

    T = min(lo.size, mid.size, hi.size, min_env.size, max_env.size)
    lo, mid, hi, min_env, max_env = lo[:T], mid[:T], hi[:T], min_env[:T], max_env[:T]
    H = int(height_px)
    half = H // 2

    rgb = np.stack([lo, mid, hi], axis=1)
    magnitudes = np.linalg.norm(rgb, axis=1)
    max_mag = float(magnitudes.max()) if magnitudes.size else 1.0
    scale = np.clip(magnitudes, 1e-8, None)
    rgb_norm = rgb / scale[:, None]
    threshold = white_threshold * max_mag
    low_mask = magnitudes < threshold
    rgb_norm[low_mask] = [1.0, 1.0, 1.0]
    rgb8 = (np.clip(rgb_norm, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    img = np.zeros((H, T, 3), dtype=np.uint8)
    y_top = half - (max_env * half).astype(int)
    y_bot = half - (min_env * half).astype(int)
    y_top = np.clip(y_top, 0, H - 1)
    y_bot = np.clip(y_bot, 0, H - 1)
    for idx in range(T):
        if y_top[idx] < y_bot[idx]:
            y_top[idx], y_bot[idx] = y_bot[idx], y_top[idx]
    mid_row = min(max(0, half), H - 1)
    for t in range(T):
        top = int(max(y_top[t], y_bot[t]))
        bot = int(min(y_top[t], y_bot[t]))
        if top == bot:
            top = bot = mid_row
        img[bot:top + 1, t, :] = rgb8[t]
    if downsample > 1:
        img = downsample_blur_stride(img, downsample)
    img = apply_contrast_stretch(img)
    print("[Render] Rendering Finished")

    return img

def gen_img_mipmap(img_np, ds_scale = 8):
    print("[Render] Generating Mipmap")
    img = downsample_blur_stride(img_np, ds_scale) 
    print("[Render] Generating Mipmap Finished")
    return img
