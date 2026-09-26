import csv
import io
import json
import os
import subprocess

from PySide6 import QtGui
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.config import externalsyncconfig, memorydeckconfig, memoryvalueconfig
from core.resource_paths import process_denylist_path


def make_memory_value_group(
    title: str,
    *,
    include_length: bool = False,
    include_encoding: bool = False,
    include_bit_pos: bool = False,
    include_multiplier: bool = False,
) -> dict:
    group = QGroupBox(title)
    form = QFormLayout(group)
    ed_offsets = QLineEdit()
    ed_offsets.setPlaceholderText("00000000,00,00,00")
    cmb_type = QComboBox()
    cmb_type.addItems(["float", "bool", "str", "int"])
    ed_offsets.setToolTip(
        "Comma-separated hex pointer chain. The first value is an offset from the\n"
        "process's main module (prefix 'g' for an absolute address); each following\n"
        "value is dereferenced and added as an offset."
    )
    cmb_type.setToolTip("Data type stored at the resolved address.")
    widgets = {
        "group": group,
        "offsets": ed_offsets,
        "value_type": cmb_type,
    }
    form.addRow("Offset chain", ed_offsets)
    form.addRow("Type", cmb_type)
    if include_length:
        sp_length = QSpinBox()
        sp_length.setRange(0, 1_000_000)
        sp_length.setToolTip("Bytes to read for the string; it is cut at the first NUL.")
        widgets["length"] = sp_length
        form.addRow("Length", sp_length)
    if include_encoding:
        ed_encoding = QLineEdit()
        ed_encoding.setToolTip("Text encoding of the string, e.g. utf-8 or utf-16-le.")
        widgets["encoding"] = ed_encoding
        form.addRow("Encoding", ed_encoding)
    if include_bit_pos:
        sp_bit_pos = QSpinBox()
        sp_bit_pos.setRange(0, 63)
        sp_bit_pos.setToolTip(
            "Bit of the byte read as the flag (0-7). Values above 7 use the whole byte."
        )
        widgets["bit_pos"] = sp_bit_pos
        form.addRow("Bit position", sp_bit_pos)
    if include_multiplier:
        sp_multiplier = QDoubleSpinBox()
        sp_multiplier.setDecimals(8)
        sp_multiplier.setRange(-1_000_000.0, 1_000_000.0)
        sp_multiplier.setSingleStep(0.01)
        sp_multiplier.setToolTip(
            "The value read is multiplied by this to get seconds (e.g. 0.001 for ms)."
        )
        widgets["multiplier"] = sp_multiplier
        form.addRow("Multiplier", sp_multiplier)
    return widgets


def make_memory_deck_group(title: str) -> dict:
    group = QGroupBox(title)
    layout = QVBoxLayout(group)
    time_spec = make_memory_value_group("Time", include_multiplier=True)
    sample_index_spec = make_memory_value_group("Current Sample Index")
    path_spec = make_memory_value_group(
        "Path", include_length=True, include_encoding=True
    )
    active_spec = make_memory_value_group("Active", include_bit_pos=True)
    loaded_spec = make_memory_value_group("Loaded", include_bit_pos=True)
    time_spec["group"].setToolTip("Playback position of the deck (Time Sync mode).")
    sample_index_spec["group"].setToolTip(
        "Playback position of the deck in samples (Sample Index Sync mode)."
    )
    path_spec["group"].setToolTip("File path of the track loaded on the deck.")
    active_spec["group"].setToolTip("Flag marking the deck as active. Only loaded and active decks are followed.")
    loaded_spec["group"].setToolTip("Flag that is true while a track is loaded on the deck.")
    layout.addWidget(time_spec["group"])
    layout.addWidget(sample_index_spec["group"])
    layout.addWidget(path_spec["group"])
    layout.addWidget(active_spec["group"])
    layout.addWidget(loaded_spec["group"])
    return {
        "group": group,
        "time": time_spec,
        "sample_index": sample_index_spec,
        "path": path_spec,
        "active": active_spec,
        "loaded": loaded_spec,
    }


def build_external_sync_tab(dialog) -> None:
    dialog.tab_external_sync = QWidget()
    root = QVBoxLayout(dialog.tab_external_sync)
    form = QFormLayout()

    dialog.cb_external_sync_enabled = QCheckBox("Enable external sync")
    dialog.cmb_external_sync_mode = QComboBox()
    dialog.cmb_external_sync_mode.addItems(["Time Sync", "Sample Index Sync"])
    dialog.cmb_total_sample_count_source = QComboBox()
    dialog.cmb_total_sample_count_source.addItems(
        ["from reference sample rate", "from file"]
    )
    dialog.sp_reference_sample_rate = QSpinBox()
    dialog.sp_reference_sample_rate.setRange(1, 384000)
    dialog.sp_reference_sample_rate.setSingleStep(100)

    dialog.cmb_memory_process = QComboBox()
    dialog.cmb_memory_process.setEditable(True)
    dialog.cmb_memory_process.setSizeAdjustPolicy(
        QComboBox.AdjustToMinimumContentsLengthWithIcon
    )
    dialog.cmb_memory_process.setMinimumContentsLength(20)
    dialog.cmb_memory_process.view().setTextElideMode(Qt.ElideRight)
    dialog.cmb_memory_process.view().setHorizontalScrollBarPolicy(
        Qt.ScrollBarAlwaysOff
    )
    dialog.btn_memory_process_refresh = QPushButton("Refresh")
    dialog.cb_external_sync_enabled.setToolTip(
        "Follow the track and playback position of external software\n"
        "by reading its memory."
    )
    dialog.cmb_external_sync_mode.setToolTip(
        "Time Sync: read the position as time (scaled by the multiplier).\n"
        "Sample Index Sync: read the position as a sample index."
    )
    dialog.cmb_total_sample_count_source.setToolTip(
        "How the track's total sample count is found to convert a sample index to time.\n"
        "from reference sample rate: duration x reference sample rate.\n"
        "from file: the decoded file's own sample count."
    )
    dialog.sp_reference_sample_rate.setToolTip(
        "Sample rate the external software counts samples at."
    )
    dialog.cmb_memory_process.setToolTip(
        "Process to read memory from. Denylisted processes are blocked."
    )
    dialog.btn_memory_process_refresh.setToolTip("Reload the list of running processes.")
    dialog.deck1_specs = make_memory_deck_group("Deck 1")
    dialog.deck2_specs = make_memory_deck_group("Deck 2")

    dialog.grp_memory = QGroupBox("Memory")
    f_memory = QFormLayout(dialog.grp_memory)
    process_row = QWidget()
    process_row.setToolTip(dialog.cmb_memory_process.toolTip())
    process_row_layout = QHBoxLayout(process_row)
    process_row_layout.setContentsMargins(0, 0, 0, 0)
    process_row_layout.setSpacing(6)
    process_row_layout.addWidget(dialog.cmb_memory_process, 1)
    process_row_layout.addWidget(dialog.btn_memory_process_refresh, 0)
    f_memory.addRow("Process", process_row)
    dialog.lbl_memory_process_warning = QLabel(
        "Warning: attaching memory sync to an unrelated process can trigger "
        "anti-cheat or antivirus false positives. Only target software you trust "
        "and explicitly intend to sync with."
    )
    dialog.lbl_memory_process_warning.setWordWrap(True)
    dialog.lbl_memory_process_warning.setSizePolicy(
        QSizePolicy.Ignored, QSizePolicy.Preferred
    )
    dialog.lbl_memory_process_warning.setStyleSheet("color: #d8a441;")
    f_memory.addRow(dialog.lbl_memory_process_warning)
    dialog.lbl_memory_process_blocked = QLabel("")
    dialog.lbl_memory_process_blocked.setWordWrap(True)
    dialog.lbl_memory_process_blocked.setSizePolicy(
        QSizePolicy.Ignored, QSizePolicy.Preferred
    )
    dialog.lbl_memory_process_blocked.setStyleSheet("color: #d86f41;")
    f_memory.addRow(dialog.lbl_memory_process_blocked)

    dialog.lbl_external_sync_note = QLabel(
        "When enabled, local track loading and transport playback are restricted "
        "so the external software remains the source of truth."
    )
    dialog.lbl_external_sync_note.setWordWrap(True)

    form.addRow(dialog.cb_external_sync_enabled)
    form.addRow("Mode", dialog.cmb_external_sync_mode)
    form.addRow(
        "Total Sample Count Source", dialog.cmb_total_sample_count_source
    )
    form.addRow("Reference Sample Rate", dialog.sp_reference_sample_rate)
    form.addRow(dialog.lbl_external_sync_note)
    root.addLayout(form)

    dialog.memory_scroll = QScrollArea()
    dialog.memory_scroll.setObjectName("memorySettingsScroll")
    dialog.memory_scroll.setWidgetResizable(True)
    dialog.memory_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    dialog.memory_scroll.setMinimumHeight(340)
    dialog.memory_scroll.viewport().setObjectName("memorySettingsViewport")
    dialog.memory_scroll.viewport().setBackgroundRole(QtGui.QPalette.Window)
    dialog.memory_scroll.viewport().setAutoFillBackground(True)
    dialog.memory_scroll_contents = QWidget()
    dialog.memory_scroll_contents.setObjectName("memorySettingsContents")
    dialog.memory_scroll_contents.setSizePolicy(
        QSizePolicy.Expanding, QSizePolicy.Preferred
    )
    dialog.memory_scroll_contents.setBackgroundRole(QtGui.QPalette.Window)
    dialog.memory_scroll_contents.setAutoFillBackground(True)
    memory_scroll_layout = QVBoxLayout(dialog.memory_scroll_contents)
    memory_scroll_layout.setContentsMargins(0, 0, 0, 0)
    memory_scroll_layout.setSpacing(8)
    memory_scroll_layout.addWidget(dialog.grp_memory)
    memory_scroll_layout.addWidget(dialog.deck1_specs["group"])
    memory_scroll_layout.addWidget(dialog.deck2_specs["group"])
    memory_scroll_layout.addStretch(1)
    dialog.memory_scroll.setWidget(dialog.memory_scroll_contents)
    dialog.memory_scroll.setStyleSheet(
        "QScrollArea#memorySettingsScroll, "
        "QWidget#memorySettingsViewport, "
        "QWidget#memorySettingsContents { "
        "background-color: palette(window); "
        "}"
    )
    root.addWidget(dialog.memory_scroll)
    root.addStretch(1)

    dialog.cmb_external_sync_mode.currentIndexChanged.connect(
        dialog._sync_external_sync_mode_ui
    )
    dialog.cmb_total_sample_count_source.currentIndexChanged.connect(
        dialog._sync_external_sync_mode_ui
    )
    dialog.btn_memory_process_refresh.clicked.connect(
        dialog._refresh_memory_processes
    )
    dialog._refresh_memory_processes()
    dialog._sync_external_sync_mode_ui()
    dialog.tabs.addTab(dialog.tab_external_sync, "External Sync")


class ExternalSyncSettingsMixin:
    def _sync_external_sync_mode_ui(self):
        is_time_sync = self.cmb_external_sync_mode.currentIndex() == 0
        use_reference_sample_rate = self.cmb_total_sample_count_source.currentIndex() == 0
        self.memory_scroll.setVisible(True)
        self.cmb_total_sample_count_source.setEnabled(not is_time_sync)
        self.sp_reference_sample_rate.setEnabled((not is_time_sync) and use_reference_sample_rate)
        for deck_widgets in (self.deck1_specs, self.deck2_specs):
            deck_widgets["time"]["group"].setVisible(is_time_sync)
            deck_widgets["sample_index"]["group"].setVisible(not is_time_sync)

    def _refresh_memory_processes(self):
        current_name = self._memory_process_name()
        current_pid = self._memory_process_pid()
        self.cmb_memory_process.blockSignals(True)
        self.cmb_memory_process.clear()
        processes = self._list_running_processes()
        processes.sort(
            key=lambda proc: (
                1 if self._is_denied_process_name(str(proc.get("name") or "")) else 0,
                str(proc.get("name") or "").lower(),
                int(proc.get("pid") or 0),
            )
        )
        model = self.cmb_memory_process.model()
        for proc in processes:
            denied = self._is_denied_process_name(str(proc.get("name") or ""))
            label = f'{proc["name"]} (PID {proc["pid"]})'
            if denied:
                label = f"[Blocked] {label}"
            self.cmb_memory_process.addItem(label, proc)
            row = self.cmb_memory_process.count() - 1
            self.cmb_memory_process.setItemData(row, label, Qt.ToolTipRole)
            item = model.item(row) if hasattr(model, "item") else None
            if item is not None and denied:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled & ~Qt.ItemIsSelectable)
                item.setForeground(QtGui.QColor("#888888"))
        self.cmb_memory_process.blockSignals(False)
        self._set_selected_memory_process(current_name, current_pid)

    def _set_selected_memory_process(self, process_name: str, process_pid: int):
        target_name = (process_name or "").strip().lower()
        if self._is_denied_process_name(target_name):
            self.cmb_memory_process.setCurrentIndex(-1)
            self.cmb_memory_process.setEditText("")
            self.lbl_memory_process_blocked.setText(
                f"Blocked by denylist: {process_name.strip()}"
            )
            return
        target_pid = int(process_pid or 0)
        for idx in range(self.cmb_memory_process.count()):
            data = self.cmb_memory_process.itemData(idx)
            if not isinstance(data, dict):
                continue
            item_name = str(data.get("name") or "").strip().lower()
            item_pid = int(data.get("pid") or 0)
            if target_pid and item_pid == target_pid:
                self.cmb_memory_process.setCurrentIndex(idx)
                self.lbl_memory_process_blocked.setText("")
                return
            if target_name and item_name == target_name:
                self.cmb_memory_process.setCurrentIndex(idx)
                self.lbl_memory_process_blocked.setText("")
                return
        text = process_name.strip() if process_name else ""
        if target_pid:
            text = f"{text} (PID {target_pid})".strip()
        if self._is_denied_process_name(text):
            self.cmb_memory_process.setCurrentIndex(-1)
            self.cmb_memory_process.setEditText("")
            self.lbl_memory_process_blocked.setText(
                f"Blocked by denylist: {process_name.strip()}"
            )
            return
        self.cmb_memory_process.setCurrentIndex(-1)
        self.cmb_memory_process.setEditText(text)
        self.lbl_memory_process_blocked.setText("")

    def _memory_process_name(self) -> str:
        data = self.cmb_memory_process.currentData()
        if isinstance(data, dict):
            name = str(data.get("name") or "").strip()
            return "" if self._is_denied_process_name(name) else name
        text = self.cmb_memory_process.currentText().strip()
        name = text.split(" (PID ", 1)[0].strip()
        if self._is_denied_process_name(name):
            self.lbl_memory_process_blocked.setText(f"Blocked by denylist: {name}")
            return ""
        return name

    def _memory_process_pid(self) -> int:
        data = self.cmb_memory_process.currentData()
        if isinstance(data, dict):
            try:
                return int(data.get("pid") or 0)
            except Exception:
                return 0
        text = self.cmb_memory_process.currentText().strip()
        if " (PID " in text and text.endswith(")"):
            try:
                return int(text.rsplit(" (PID ", 1)[1][:-1])
            except Exception:
                return 0
        return 0

    def _list_running_processes(self) -> list[dict]:
        try:
            if os.name == "nt":
                out = subprocess.check_output(
                    ["tasklist", "/FO", "CSV", "/NH"],
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            else:
                out = subprocess.check_output(
                    ["ps", "-A", "-o", "pid=,comm="],
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
        except Exception:
            return []
        rows = []
        if os.name == "nt":
            for row in csv.reader(io.StringIO(out)):
                if len(row) < 2:
                    continue
                name = str(row[0]).strip()
                pid_text = str(row[1]).strip()
                try:
                    pid = int(pid_text)
                except Exception:
                    continue
                if name:
                    rows.append({"name": name, "pid": pid})
        else:
            for line in out.splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) != 2:
                    continue
                try:
                    pid = int(parts[0])
                except Exception:
                    continue
                if parts[1].strip():
                    rows.append({"name": parts[1].strip(), "pid": pid})
        rows.sort(key=lambda item: (item["name"].lower(), item["pid"]))
        return rows

    def _is_denied_process_name(self, name: str) -> bool:
        deny = self._load_process_denylist()
        if deny is None:
            # Fail closed: if the denylist cannot be read, treat every process
            # as denied so an unsafe target can never be picked.
            return True
        target = self._normalize_process_name(name)
        if not target:
            return False
        return target in deny

    def _load_process_denylist(self) -> set[str] | None:
        cache = getattr(self, "_process_denylist_cache", None)
        if isinstance(cache, set):
            return cache
        try:
            payload = json.loads(process_denylist_path().read_text(encoding="utf-8"))
        except Exception:
            # Not cached, so a transient error recovers on the next call.
            return None
        names: set[str] = set()
        if isinstance(payload, dict):
            for item in payload.get("blocked_process_names", []):
                text = self._normalize_process_name(item)
                if text:
                    names.add(text)
        self._process_denylist_cache = names
        return names

    def _normalize_process_name(self, name: str) -> str:
        text = str(name or "").strip().lower()
        if text.endswith(".exe"):
            text = text[:-4]
        return text
