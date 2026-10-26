"""{libpath}/{uid}.npz: each track's stored analysis (`FeatureNPZStore`), the file format alone. Used by the jobs
(core.analysis_library.jobs, core.library.jobs) on the library's owner's thread, nowhere else."""
from __future__ import annotations
import os
from typing import Any, Dict, Optional
import numpy as np

from system.ids import canonical_uuid4 as _canonical_uuid4
from system.atomic_io import atomic_output_path

def _ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def _validated_uid(uid: str) -> str:
    return _canonical_uuid4(uid)

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
        return os.path.join(self.base_dir, f"{_validated_uid(uid)}.npz")

    def save(self, uid: str, features: Dict[str, Any]) -> str:
        """
        Save features(dict) to NPZ, filtering to storable values.
        Returns storage path.
        """
        if not isinstance(features, dict):
            raise TypeError("features must be a dict")
        safe_uid = _validated_uid(uid)

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
            raise ValueError(
                "No storable feature values "
                "(only numeric/bool arrays/lists/scalars are supported)."
            )

        # 2) Serialize directly to a sibling temporary file, then replace.
        final = self.path(safe_uid)
        with atomic_output_path(final) as temporary:
            if self.compressed:
                np.savez_compressed(temporary, **kv)
            else:
                np.savez(temporary, **kv)
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

    def load_keys(self, uid: str, keys: tuple, prefixes: tuple = ()) -> Optional[Dict[str, np.ndarray]]:
        """Only the stored arrays named in ``keys`` or starting with one of ``prefixes`` (the others are not
        decompressed); None when the track has no stored features."""
        p = self.path(uid)
        if not os.path.exists(p):
            return None
        with np.load(p, allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files if k in keys or k.startswith(prefixes)}

    def exists(self, uid: str) -> bool:
        try:
            return os.path.exists(self.path(uid))
        except ValueError:
            return False

    def delete(self, uid: str) -> None:
        try:
            os.remove(self.path(uid))
        except (FileNotFoundError, ValueError):
            pass
