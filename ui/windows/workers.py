from typing import Optional, Dict
from PySide6.QtCore import Qt, Signal, QSize
from PySide6 import QtGui
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QProgressBar,
    QDialogButtonBox, QTabWidget
)
from core.workflows.event_bus import EventBus
from system.i18n import tr, tr_mark
from core.workflows.taskmanager import ANALYSIS, TIEUP_LOOKUP
from ui.widgets.elided_label import ElidedLabel


class WorkersDialog(QDialog):
    """Subwindow showing the tasks in progress, a tab per kind (core.workflows.taskmanager): the analyses, and the tie-ups looked
    up on the web in the background (core.workflows.tieup_web), each tab's title with its count."""

    # Public signals (Dialog re-emits with same payload)
    sig_task_started = Signal(int, object)          # task_id, metadata
    sig_task_progress = Signal(int, float, str)     # task_id, progress(0..1), message
    sig_task_finished = Signal(int, str)    # task_id, error

    TABS = ((ANALYSIS, tr_mark("Analysis")), (TIEUP_LOOKUP, tr_mark("Tie-up Lookup")))

    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("Workers"))
        self.setModal(False)
        self.resize(500, 700)

        self._items: Dict[int, QListWidgetItem] = {}
        self._lists: Dict[int, QListWidget] = {}     # a task's list (its tab's)
        self._bus = None

        # UI: a list per kind
        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.lists = {kind: QListWidget() for kind, _name in self.TABS}
        for kind, _name in self.TABS:
            self.tabs.addTab(self.lists[kind], "")
        self._retitle()

        # Button box
        self.btn_box = QDialogButtonBox(QDialogButtonBox.Close)

        self.btn_box.rejected.connect(self.reject)

        root.addWidget(self.tabs)
        root.addWidget(self.btn_box)

    def _retitle(self) -> None:
        """Each tab's title with its tasks' count."""
        for i, (kind, name) in enumerate(self.TABS):
            n = self.lists[kind].count()
            self.tabs.setTabText(i, tr("{name} ({n})").format(name=tr(name), n=n) if n else tr(name))

    # EventBus wiring
    def set_bus(self, bus) -> None:
        # Disconnect previous bus
        if getattr(self, "_bus", None) is not None:
            self._disconnect_bus_signals(self._bus)
        self._bus = bus
        if bus is None:
            return
        # Connect new bus (signals assumed to exist)
        self._connect_bus_signals(bus)

    def _connect_bus_signals(self, bus:EventBus) -> None:
        bus.sig_task_created.connect(self._on_created)
        bus.sig_task_progress.connect(self._on_progress)
        bus.sig_task_finished.connect(self._on_finished)

    def _disconnect_bus_signals(self, bus) -> None:
        try:
            bus.sig_task_created.disconnect(self._on_created)
        except Exception:
            pass
        try:
            bus.sig_task_progress.disconnect(self._on_progress)
        except Exception:
            pass
        try:
            bus.sig_task_finished.disconnect(self._on_finished)
        except Exception:
            pass

    # Bus handlers
    def _on_created(self, task_id: int, metadata: Optional[object]):
        # Create UI entry and re-emit
        if task_id in self._items:
            return
        self._add_item(task_id, metadata)
        # Apply initial status/progress from task dataclass
        init_status = getattr(metadata, 'status', None)
        init_prog = getattr(metadata, 'progress', None)
        if init_status is not None or init_prog is not None:
            self._on_progress(task_id, float(init_prog or 0.0), str(init_status or ""))
        self.sig_task_started.emit(task_id, metadata)

    def _on_progress(self, task_id: int, progress: float, message: str):
        item = self._items.get(task_id)
        if not item:
            return
        data = item.data(Qt.UserRole) or {}
        bar: QProgressBar = data.get("bar")
        lbl_status: QLabel = data.get("label_status")
        if bar is not None:
            p = max(0.0, min(1.0, float(progress)))
            bar.setValue(int(p * 100))
        if message and lbl_status is not None:
            lbl_status.setText(message)
        self.sig_task_progress.emit(task_id, float(progress), message or "")

    def _on_finished(self, task_id: int, error: Optional[str] = None):
        item = self._items.pop(task_id, None)
        tasks = self._lists.pop(task_id, None)
        if item is not None and tasks is not None:
            tasks.takeItem(tasks.row(item))
            self._retitle()
        self.sig_task_finished.emit(task_id, error or "")

    # Internal UI build
    def _add_item(self, task_id: int, metadata: Optional[object]):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 10, 8, 10)
        lay.setSpacing(8)

        # Top row: thumbnail + text (title, status)
        row = QWidget()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(10)

        # Thumbnail
        thumb_lbl = QLabel()
        thumb_lbl.setFixedSize(64, 64)
        thumb_lbl.setScaledContents(True)
        img = self._extract_pixmap(metadata)
        if img is not None:
            scaled_img = img.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            thumb_lbl.setPixmap(QtGui.QPixmap.fromImage(scaled_img))

        # Text area (title, status)
        text_box = QWidget()
        text_lay = QVBoxLayout(text_box)
        text_lay.setContentsMargins(0, 0, 0, 0)
        text_lay.setSpacing(2)

        name = self._extract_name(metadata) or tr("Task {id}").format(id=task_id)
        # a long title or message is cut to the row (ElidedLabel), never widening it
        name_lbl = ElidedLabel(name)
        # Initial status text (task.status)
        init_status = getattr(metadata, 'status', '')
        status_lbl = ElidedLabel(str(init_status) if init_status is not None else '')
        status_lbl.setStyleSheet("color: gray;")
        text_lay.addWidget(name_lbl)
        text_lay.addWidget(status_lbl)

        row_lay.addWidget(thumb_lbl)
        row_lay.addWidget(text_box, 1)

        # Progress bar
        bar = QProgressBar(); bar.setRange(0, 100); bar.setValue(0)
        bar.setFixedHeight(16)
        # Initial progress (task.progress)
        pv = getattr(metadata, 'progress', None)
        if pv is not None:
            p = max(0.0, min(1.0, float(pv)))
            bar.setValue(int(p * 100))

        lay.addWidget(row)
        lay.addWidget(bar)

        item = QListWidgetItem()
        item.setSizeHint(QSize(0, 96))
        item.setData(Qt.UserRole, {
            "label_name": name_lbl,
            "label_status": status_lbl,
            "bar": bar,
            "thumb": thumb_lbl,
        })
        tasks = self.lists.get(getattr(metadata, "kind", ANALYSIS), self.lists[ANALYSIS])
        tasks.addItem(item)
        tasks.setItemWidget(item, w)
        self._items[task_id] = item
        self._lists[task_id] = tasks
        self._retitle()

    def _extract_name(self, metadata: Optional[object]) -> Optional[str]:
        if metadata is None:
            return None
        v = getattr(metadata, 'songname', None)
        return str(v) if v is not None else None

    def _extract_pixmap(self, metadata: Optional[object]) -> Optional[QtGui.QImage]:
        # If no QImage, generate a simple fallback thumbnail
        if metadata is not None:
            v = getattr(metadata, 'thumbnail', None)
            if isinstance(v, QtGui.QImage) and not v.isNull():
                return v
        return self._make_fallback_image(96, 96)

    def _make_fallback_image(self, w: int, h: int) -> QtGui.QImage:
        img = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
        img.fill(QtGui.QColor("#333"))
        painter = QtGui.QPainter(img)
        try:
            painter.setPen(QtGui.QPen(QtGui.QColor("#666")))
            painter.drawRect(0, 0, w - 1, h - 1)
            painter.setPen(QtGui.QPen(QtGui.QColor("#aaa")))
            painter.drawText(0, 0, w, h, Qt.AlignCenter, tr("No Art"))
        finally:
            painter.end()
        return img

