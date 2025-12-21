from dataclasses import dataclass
from typing import Optional, Callable, Any
from PySide6 import QtCore
from PySide6.QtGui import QImage

@dataclass
class GlobalParams:
    analysis_samp_rate: int
    bpm_hop_length: int
    chroma_hop_length: int


class _ObservableDict(dict):
    """Dict that triggers a callback on mutation with a delta description."""

    def __init__(self, *args, on_change: Callable[[str], None], name: str = "dict", **kwargs):
        self._on_change = on_change
        self._name = name
        super().__init__(*args, **kwargs)

    def _notify(self, desc: str) -> None:
        try:
            self._on_change(f"{self._name}: {desc}")
        except Exception:
            pass

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self._notify(f"set {key} -> {self._fmt_val(value)}")

    def __delitem__(self, key) -> None:
        super().__delitem__(key)
        self._notify(f"deleted {key}")

    def clear(self) -> None:
        super().clear()
        self._notify("cleared")

    def pop(self, k, d=None) -> Any:
        result = super().pop(k, d)
        self._notify(f"popped {k}")
        return result

    def popitem(self) -> Any:
        result = super().popitem()
        self._notify(f"popped item {result[0]}")
        return result

    def setdefault(self, key, default=None) -> Any:
        existed = key in self
        result = super().setdefault(key, default)
        if not existed:
            self._notify(f"setdefault {key} -> {self._fmt_val(default)}")
        return result

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._notify("bulk update")

    @staticmethod
    def _fmt_val(val: Any) -> str:
        try:
            import numpy as _np  # local import to avoid hard dep
            if isinstance(val, _np.ndarray):
                return f"ndarray{val.shape}/{val.dtype}"
        except Exception:
            pass
        return type(val).__name__


class DataModel(QtCore.QObject):
    sig_updated = QtCore.Signal()
    def __init__(self):
        super().__init__()
        self.features: dict = self._wrap_observable({}, name="features")
        self.properties: dict = self._wrap_observable({}, name="properties")
        self.duration_sec: float = 0.0
        self.gp: Optional[GlobalParams] = None
        self.album_art: Optional[QImage] = None

    def _wrap_observable(self, mapping: dict, name: str) -> _ObservableDict:
        if isinstance(mapping, _ObservableDict):
            return mapping
        return _ObservableDict(mapping, on_change=self._notify_changed, name=name)

    def _notify_changed(self, desc: str = "updated") -> None:
        self.sig_updated.emit()
        print(f"[DataModel] {desc}")

    def load(self, features: dict, properties: dict, gp: GlobalParams, duration_sec: float, album_art: Optional[QImage] = None):
        self.features = self._wrap_observable(features, name="features")
        self.properties = self._wrap_observable(properties, name="properties")
        self.gp = gp
        self.duration_sec = duration_sec
        self.album_art = album_art
        self._notify_changed("load")
        print(f"[DataModel] Summary {self.debug_summary()}")

    def set_album_art(self, album_art: Optional[QImage]):
        self.album_art = album_art
        self._notify_changed("album_art set")

    def debug_summary(self) -> str:
        """Return a human-readable snapshot of current model state for debugging."""
        feat_keys = sorted(self.features.keys()) if isinstance(self.features, dict) else []
        props = self.properties if isinstance(self.properties, dict) else {}
        return (
            f"DataModel(features={len(feat_keys)} keys: {feat_keys[:40]}, "
            f"duration={self.duration_sec:.3f}s, "
            f"gp={self.gp}, "
            f"title={props.get('title')}, "
            f"artist={props.get('artist')}, "
            f"album_art={'set' if self.album_art else 'none'})"
        )
