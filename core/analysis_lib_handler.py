from __future__ import annotations
import os, json, tempfile, shutil
from typing import Any, Dict, Iterable, Optional, Tuple
import numpy as np

def _ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d

def _atomic_write_bytes(path: str, data: bytes) -> None:
    """Write to a temp file, then atomically replace via os.replace."""
    d = os.path.dirname(path) or "."
    _ensure_dir(d)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise

def _sanitize_key(k: str) -> str:
    """
    Normalize to an np.savez-safe key.
    (allow alnum/underscore/dot; replace others with '_')
    """
    out = []
    for ch in str(k):
        if ch.isalnum() or ch in ("_", "."):
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    return s if s else "k"

def _flatten_dict(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dict → single-level dict with 'a.b.c' keys.
    Lists/tuples/ndarrays/scalars stay as values.
    """
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else f"{k}"
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, key, sep=sep))
        else:
            flat[key] = v
    return flat

def _as_np_compatible(value: Any) -> Optional[np.ndarray]:
    """
    Convert to an np.savez-friendly type.
    - np.ndarray → passthrough
    - scalars (numeric/bool) → 0-d array
    - list/tuple (numeric/bool/ndarray-friendly) → np.asarray
    - otherwise None (skip storing)
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (int, float, bool, np.number)):
        return np.asarray(value)
    if isinstance(value, (list, tuple)):
        # Quick check whether elements form a numeric/bool/ndarray-compatible array
        try:
            arr = np.asarray(value)
            # Skip object dtype
            if arr.dtype == object:
                return None
            return arr
        except Exception:
            return None
    # Do not store strings/binary/other objects as features (put in JSON)
    return None

# ==============================
# Feature (NPZ) Store
# ==============================

class FeatureNPZStore:
    """
    UID-based feature NPZ store/loader.
    - Path pattern: {base_dir}/{uid}.npz
    - Stores only NumPy-compatible values (numeric scalars/arrays/lists/tuples)
    - Flattens nested dicts into 'a.b.c' keys
    """
    def __init__(self, base_dir: str, compressed: bool = True, sep: str = "."):
        self.base_dir = _ensure_dir(base_dir)
        self.compressed = compressed
        self.sep = sep

    def path(self, uid: str) -> str:
        return os.path.join(self.base_dir, f"{uid}.npz")

    def save(self, uid: str, features: Dict[str, Any]) -> str:
        """
        Save features(dict) to NPZ, filtering to storable values.
        Returns storage path.
        """
        if not isinstance(features, dict):
            raise TypeError("features must be a dict")

        # 1) Flatten & sanitize keys
        flat = _flatten_dict(features, sep=self.sep)
        kv: Dict[str, np.ndarray] = {}
        skipped: Dict[str, Any] = {}

        for k, v in flat.items():
            k2 = _sanitize_key(k)
            arr = _as_np_compatible(v)
            if arr is None:
                skipped[k2] = v  # keep for potential logging
                continue
            kv[k2] = arr

        if not kv:
            raise ValueError("No storable feature values (only numeric/bool arrays/lists/scalars are supported).")

        # 2) Serialize NPZ bytes
        # np.savez* only accepts file paths, so temp file -> read bytes -> atomic replace
        tmp_path = os.path.join(self.base_dir, f".tmp_write_{uid}.npz")
        _ensure_dir(self.base_dir)
        if self.compressed:
            np.savez_compressed(tmp_path, **kv)
        else:
            np.savez(tmp_path, **kv)

        with open(tmp_path, "rb") as f:
            data = f.read()
        os.remove(tmp_path)

        # 3) Atomic save
        final = self.path(uid)
        _atomic_write_bytes(final, data)
        return final

    def load(self, uid: str) -> Dict[str, np.ndarray]:
        """
        Load NPZ → dict (returns flattened keys as-is).
        """
        p = self.path(uid)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        with np.load(p, allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}

    def exists(self, uid: str) -> bool:
        return os.path.exists(self.path(uid))

    def delete(self, uid: str) -> None:
        try:
            os.remove(self.path(uid))
        except FileNotFoundError:
            pass

