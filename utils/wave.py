import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration
    njit = None


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


def _render_wave_columns_py(y_top, y_bot, rgb8, height_px: int):
    t_len = int(rgb8.shape[0])
    img = np.zeros((int(height_px), t_len, 3), dtype=np.uint8)
    half = int(height_px) // 2
    mid_row = min(max(0, half), int(height_px) - 1)
    for idx in range(t_len):
        top = int(y_top[idx])
        bot = int(y_bot[idx])
        if top < bot:
            top, bot = bot, top
        if top == bot:
            top = mid_row
            bot = mid_row
        img[bot:top + 1, idx, :] = rgb8[idx]
    return img


if njit is not None:
    @njit(cache=True)
    def _render_wave_columns_numba(y_top, y_bot, rgb8, height_px: int):
        t_len = rgb8.shape[0]
        img = np.zeros((height_px, t_len, 3), dtype=np.uint8)
        half = height_px // 2
        mid_row = half
        if mid_row < 0:
            mid_row = 0
        elif mid_row >= height_px:
            mid_row = height_px - 1
        for idx in range(t_len):
            top = int(y_top[idx])
            bot = int(y_bot[idx])
            if top < bot:
                tmp = top
                top = bot
                bot = tmp
            if top == bot:
                top = mid_row
                bot = mid_row
            r = rgb8[idx, 0]
            g = rgb8[idx, 1]
            b = rgb8[idx, 2]
            for row in range(bot, top + 1):
                img[row, idx, 0] = r
                img[row, idx, 1] = g
                img[row, idx, 2] = b
        return img
else:
    _render_wave_columns_numba = None


def build_wave_image(lo, mid, hi, min_env, max_env, height_px=110, downsample=2, white_threshold=0.05):

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
    active_mask = ~low_mask
    if np.any(active_mask):
        active_rgb = rgb_norm[active_mask]
        min_idx = np.argmin(active_rgb, axis=1)
        active_rgb[np.arange(active_rgb.shape[0]), min_idx] = 0.0
        rgb_norm[active_mask] = active_rgb
    rgb8 = (np.clip(rgb_norm, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    img = np.zeros((H, T, 3), dtype=np.uint8)
    y_top = half - (max_env * half).astype(int)
    y_bot = half - (min_env * half).astype(int)
    y_top = np.clip(y_top, 0, H - 1)
    y_bot = np.clip(y_bot, 0, H - 1)
    render_fn = _render_wave_columns_numba or _render_wave_columns_py
    img = render_fn(y_top.astype(np.int32, copy=False), y_bot.astype(np.int32, copy=False), rgb8, H)
    if downsample > 1:
        img = downsample_blur_stride(img, downsample)
    img = apply_contrast_stretch(img)

    return img

def gen_img_mipmap(img_np, ds_scale = 8):
    print("[Render] Generating Mipmap")
    img = downsample_blur_stride(img_np, ds_scale) 
    print("[Render] Generating Mipmap Finished")
    return img
