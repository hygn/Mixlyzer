"""The search bar's part: the text search and the filters, the structured searches."""
from __future__ import annotations

from typing import Optional
from PySide6 import QtCore, QtGui, QtWidgets
from core.marks.phrases import PHRASE_LABELS, phrase_display_name
from core.search import reading
from core.library import jobs
from ui.edit_panel.key_picker import KeyPickButton
from ui.theme import current_theme
from system.i18n import tr
from ui.library.common import READ_COLUMNS, SEARCH_DEBOUNCE_MS


class _FillRowsLayout(QtWidgets.QLayout):
    """Lays its widgets out in rows that always span the full width.

    Items go left to right and wrap when the next one no longer fits; each row's
    spare width is shared out in proportion to the items' preferred widths, so
    the filter cards follow the window width instead of leaving a gap.
    """

    def __init__(self, parent=None, spacing: int = 6):
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QtCore.QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _rows(self, items: list, width: int) -> list[list]:
        """Split ``items`` (in order) into the fewest rows that fit ``width``,
        choosing among those the split whose rows are the most even, so a small
        card never ends up alone on a row while another row is crowded."""
        gap = self.spacing()
        hints = [item.sizeHint().width() for item in items]
        n = len(hints)
        best = None
        for mask in range(1 << max(0, n - 1)):  # bit i: break after item i
            rows, start = [], 0
            for i in range(n):
                if i == n - 1 or mask >> i & 1:
                    rows.append((start, i + 1))
                    start = i + 1
            natural = [sum(hints[a:b]) + gap * (b - a - 1) for a, b in rows]
            if any(w > width and b - a > 1 for w, (a, b) in zip(natural, rows)):
                continue  # only a single oversized card may overflow its row
            score = (len(rows), max(natural) - min(natural), max(natural))
            if best is None or score < best[0]:
                best = (score, rows)
        return [items[a:b] for a, b in best[1]] if best else [items]

    def _widths(self, row: list, width: int) -> list[float]:
        """Grow each card from its preferred width in proportion to it, up to its
        maximum (its inputs stop growing at a sensible size); a row that cannot
        use all of ``width`` keeps the rest at its right end."""
        gap = self.spacing()
        widths = [float(min(item.sizeHint().width(), width)) for item in row]
        limits = [float(max(w, min(self._max_width(item), width))) for item, w in zip(row, widths)]
        spare = width - sum(widths) - gap * (len(row) - 1)
        while spare > 0.5:
            open_ = [i for i in range(len(row)) if widths[i] < limits[i] - 0.5]
            if not open_:
                break
            total = sum(row[i].sizeHint().width() for i in open_) or 1
            used = 0.0
            for i in open_:
                add = min(spare * row[i].sizeHint().width() / total, limits[i] - widths[i])
                widths[i] += add
                used += add
            spare -= used
            if used < 0.5:
                break
        return widths

    @staticmethod
    def _max_width(item: QtWidgets.QLayoutItem) -> int:
        # A growable widget reports no maximum of its own; its layout knows how
        # far the inputs inside can grow before the card would only gain gaps.
        widget = item.widget()
        if widget is not None and widget.layout() is not None:
            return min(widget.maximumWidth(), widget.layout().maximumSize().width())
        return item.maximumSize().width()

    def _arrange(self, rect: QtCore.QRect, apply: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        gap = self.spacing()
        items = [item for item in self._items if not item.isEmpty()]
        if not items:
            return 0
        y = area.y()
        for row in self._rows(items, area.width()):
            height = max(item.sizeHint().height() for item in row)
            if apply:
                x = float(area.x())
                for item, w in zip(row, self._widths(row, area.width())):
                    left = round(x)
                    item.setGeometry(QtCore.QRect(left, y, round(x + w) - left, height))
                    x += w + gap
            y += height + gap
        return y - gap - area.y() + m.top() + m.bottom()


class _MatchToggle(QtWidgets.QAbstractButton):
    """AND | OR as one switch: both words always show, the active half is lit,
    and a click anywhere on it flips the mode (checked = OR)."""

    LABELS = ("AND", "OR")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.TabFocus)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        font = QtGui.QFont(self.font())
        font.setPointSize(9)
        font.setBold(True)
        self.setFont(font)
        self.setAttribute(QtCore.Qt.WA_Hover)
        self.toggled.connect(self._update_tooltip)
        self._update_tooltip()

    def match_all(self) -> bool:
        return not self.isChecked()

    def _update_tooltip(self) -> None:
        self.setToolTip(tr("AND: require every search condition (click to switch)") if self.match_all()
                        else tr("OR: accept any search condition (click to switch)"))

    def sizeHint(self) -> QtCore.QSize:
        fm = QtGui.QFontMetrics(self.font())
        half = max(fm.horizontalAdvance(t) for t in self.LABELS) + 20
        return QtCore.QSize(half * 2, fm.height() + 10)

    def paintEvent(self, _event) -> None:
        theme = current_theme()
        enabled = self.isEnabled()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        frame = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        border = theme.separator if not enabled else (
            theme.border_strong if self.underMouse() or self.hasFocus() else theme.border)
        p.setPen(QtGui.QPen(QtGui.QColor(border), 1))
        p.setBrush(QtGui.QColor(theme.raised if enabled else theme.window))
        p.drawRoundedRect(frame, theme.radius, theme.radius)

        half = frame.width() / 2
        active = 1 if self.isChecked() else 0
        lit = QtCore.QRectF(frame.left() + active * half, frame.top(), half, frame.height())
        p.setPen(QtGui.QPen(QtGui.QColor(theme.accent if enabled else theme.separator), 1))
        p.setBrush(QtGui.QColor(theme.accent_pressed if enabled else theme.raised))
        p.drawRoundedRect(lit, theme.radius, theme.radius)

        p.setFont(self.font())
        for i, text in enumerate(self.LABELS):
            if not enabled:
                color = theme.text_disabled
            elif i == active:
                color = theme.on_accent
            else:
                color = theme.text_dim
            p.setPen(QtGui.QColor(color))
            cell = QtCore.QRectF(frame.left() + i * half, frame.top(), half, frame.height())
            p.drawText(cell, QtCore.Qt.AlignCenter, text)
        p.end()


def grow(widget: QtWidgets.QWidget, width: int) -> None:
    """An input that grows with its card from ``width`` to a capped size,
    so spare row width spreads over the cards instead of one huge field."""
    widget.setMinimumWidth(width)
    widget.setMaximumWidth(round(width * 1.8))
    widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

def compact_spin(spin: QtWidgets.QDoubleSpinBox, lo: float, hi: float, value: float,
                 suffix: str = "") -> QtWidgets.QDoubleSpinBox:
    spin.setRange(lo, hi)
    spin.setDecimals(2)
    spin.setValue(value)
    spin.setSuffix(suffix)
    # At least as wide as the widest value it can show (the global font is large).
    grow(spin, spin.fontMetrics().horizontalAdvance(f"{hi:.2f}{suffix}") + 26)
    spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
    spin.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    return spin

def compact_combo(combo: QtWidgets.QComboBox, width: int = 104) -> None:
    grow(combo, width)
    combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)

def dim_label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setAlignment(QtCore.Qt.AlignCenter)
    label.setStyleSheet(f"color: {current_theme().text_dim}; background: transparent;")
    label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
    return label

# Layout
def separator() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setFixedWidth(1)
    line.setStyleSheet(f"background: {current_theme().separator};")
    return line


class SearchMixin:
    """This module's part of LibraryWidget (mixed into it)."""

    def _build_search(self) -> QtWidgets.QHBoxLayout:
        """The search bar: its first row (returned: the field, the column, the filters' chips, AND / OR) and the filters' editors under it (``filter_bar``)."""
        # Search UI. Row 1: text search, one toggle chip per structured
        # filter and AND / OR. Row 2 holds only the enabled filters' editors,
        # so an idle search bar is a single line and a filter is one click.
        self.e_search = QtWidgets.QLineEdit(placeholderText=tr("Search title, artist, key ..."))
        self.e_search.setToolTip(
            tr("<p>Titles, artists and albums are also found by how they read, in any script: "
               "あのね by anone, ano ne or 아노네 (the reading is set in Edit Track).</p>"))
        self.e_search.setClearButtonEnabled(True)
        self.cb_column = QtWidgets.QComboBox()
        for mode in self.SEARCH_MODES:
            self.cb_column.addItem(self._header_name(mode), mode)
        self.cb_column.setToolTip(tr("Column the text search looks in"))
        self.cb_column.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cb_fuzzy = self._chip(
            tr("Fuzzy"), tr("<p>Also find a similar pronunciation (from six letters of the reading on: about one "
                            "sound in five off). Off: the reading must match, up to its spelling.</p>"))
        self.cb_fuzzy.setChecked(True)


        # Filter toggles (row 1)
        self.cb_phrase_filter = self._chip(
            tr("Phrase"), tr("Tracks with a phrase of this label (optionally at a BPM / key)"))
        self.cb_transition_bpm = self._chip("BPM \u2192", tr("Tracks whose BPM changes from one tempo to another"))
        self.cb_transition_key = self._chip("Key \u2192", tr("Tracks whose key changes from one key to another"))
        self.match_toggle = _MatchToggle()
        self.btn_clear_filters = QtWidgets.QToolButton()
        self.btn_clear_filters.setText(tr("Clear"))
        self.btn_clear_filters.setToolTip(tr("Remove every search condition (text and filters)"))

        # Phrase filter: the label always applies; BPM and key refine it.
        self.cmb_phrase_label = QtWidgets.QComboBox()
        for label in PHRASE_LABELS:
            self.cmb_phrase_label.addItem(phrase_display_name(label), label)
        compact_combo(self.cmb_phrase_label, 112)
        self.cb_phrase_bpm = self._chip("BPM", tr("Also require this BPM inside the phrase"), small=True)
        self.sp_phrase_bpm = compact_spin(QtWidgets.QDoubleSpinBox(), 0.0, 400.0, 125.0)
        self.lbl_phrase_bpm_tol = dim_label("\u00b1")
        self.sp_phrase_bpm_tolerance = compact_spin(
            QtWidgets.QDoubleSpinBox(), 0.0, 100.0, 2.0, "%")
        self.sp_phrase_bpm_tolerance.setToolTip(tr("Phrase BPM tolerance"))
        self.cb_phrase_key = self._chip("Key", tr("Also require this key inside the phrase"), small=True)
        self.cmb_phrase_key = KeyPickButton()
        self.cmb_phrase_key.setToolTip(tr("Key inside the selected phrase (Alphanumeric wheel)"))
        self.cb_phrase_key_compatible = self._chip(
            tr("Compat"), tr("Also match harmonically compatible keys (\u00b11 on the Alphanumeric wheel, relative key)"),
            small=True)

        # BPM transition
        self.sp_from_bpm = compact_spin(QtWidgets.QDoubleSpinBox(), 0.0, 400.0, 125.0)
        self.lbl_bpm_arrow = dim_label("\u2192")
        self.sp_to_bpm = compact_spin(QtWidgets.QDoubleSpinBox(), 0.0, 400.0, 129.0)
        self.lbl_bpm_tol = dim_label("\u00b1")
        self.sp_bpm_tolerance = compact_spin(QtWidgets.QDoubleSpinBox(), 0.0, 100.0, 2.0, "%")
        self.sp_bpm_tolerance.setToolTip(tr("BPM transition tolerance"))
        self.cb_bpm_first_segment = self._chip(
            tr("1st"), tr("Require the transition to start at the first BPM segment"), small=True)

        # Key transition
        self.cmb_from_key = KeyPickButton()
        self.cmb_from_key.setToolTip(tr("Key before the change (Alphanumeric wheel)"))
        self.lbl_arrow = dim_label("\u2192")
        self.cmb_to_key = KeyPickButton()
        self.cmb_to_key.setToolTip(tr("Key after the change (Alphanumeric wheel)"))
        for picker in (self.cmb_phrase_key, self.cmb_from_key, self.cmb_to_key):
            grow(picker, 84)
            # as tall as the text inputs beside it
            picker.setFixedHeight(self.sp_from_bpm.sizeHint().height())
        self.cb_key_first_segment = self._chip(
            tr("1st"), tr("Require the transition to start at the first key segment"), small=True)
        self.cb_key_compatible = self._chip(
            tr("Compat"), tr("Also match harmonically compatible keys at both ends "
                             "(\u00b11 on the Alphanumeric wheel, relative key)"), small=True)
        self.cb_key_compatible.setChecked(True)

        # Shared by both transitions
        self.lbl_min_sec = dim_label("\u2265")
        self.sp_transition_min_duration = compact_spin(
            QtWidgets.QDoubleSpinBox(), 0.0, 600.0, 4.0, "s")
        self.sp_transition_min_duration.setToolTip(
            tr("Minimum duration of both sides of each enabled transition"))

        self.card_phrase = self._card(tr("Phrase"), (
            self.cmb_phrase_label, self.cb_phrase_bpm, self.sp_phrase_bpm,
            self.lbl_phrase_bpm_tol, self.sp_phrase_bpm_tolerance,
            self.cb_phrase_key, self.cmb_phrase_key, self.cb_phrase_key_compatible))
        self.card_bpm = self._card("BPM", (
            self.sp_from_bpm, self.lbl_bpm_arrow, self.sp_to_bpm,
            self.lbl_bpm_tol, self.sp_bpm_tolerance, self.cb_bpm_first_segment))
        self.card_key = self._card("Key", (
            self.cmb_from_key, self.lbl_arrow, self.cmb_to_key,
            self.cb_key_compatible, self.cb_key_first_segment))
        self.card_min_sec = self._card(tr("Min seg"), (self.lbl_min_sec, self.sp_transition_min_duration))
        self.card_min_sec.setToolTip(tr("Minimum duration of both sides of each enabled transition"))

        row_h = self.e_search.sizeHint().height()
        for widget in (self.cb_column, self.cb_fuzzy, self.cb_phrase_filter, self.cb_transition_bpm,
                       self.cb_transition_key, self.match_toggle,
                       self.btn_clear_filters, self.btn_mic, self.btn_music_close, self.music_bar):
            widget.setFixedHeight(row_h)
        self.btn_mic.setFixedWidth(row_h)
        self.btn_music_close.setFixedWidth(row_h)

        primary = QtWidgets.QHBoxLayout()
        primary.setContentsMargins(0, 0, 0, 0)
        primary.setSpacing(4)
        self.e_search.setMinimumWidth(220)
        primary.addWidget(self.btn_mic)
        primary.addWidget(self.btn_music_close)
        primary.addWidget(self.e_search, 1)
        primary.addWidget(self.music_bar, 1)
        primary.addWidget(self.cb_column)
        primary.addWidget(self.cb_fuzzy)
        primary.addSpacing(4)
        primary.addWidget(separator())
        primary.addSpacing(4)
        for chip in (self.cb_phrase_filter, self.cb_transition_bpm, self.cb_transition_key):
            primary.addWidget(chip)
        primary.addSpacing(4)
        primary.addWidget(separator())
        primary.addSpacing(4)
        primary.addWidget(self.match_toggle)
        primary.addWidget(self.btn_clear_filters)

        self.filter_bar = QtWidgets.QWidget()
        filters = _FillRowsLayout(self.filter_bar)
        for card in (self.card_phrase, self.card_bpm, self.card_key, self.card_min_sec):
            filters.addWidget(card)
        return primary

    def _connect_search(self):
        """The search bar's signals: a filter change is applied once they pause (SEARCH_DEBOUNCE_MS)."""
        self._search_timer = QtCore.QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(SEARCH_DEBOUNCE_MS)
        self._search_timer.timeout.connect(self._apply_filter)
        self.e_search.textChanged.connect(self._on_search_changed)
        self.e_search.textChanged.connect(self._sync_filter_ui)
        self.cb_column.currentIndexChanged.connect(self._on_search_changed)
        self.cb_fuzzy.toggled.connect(self._on_search_changed)
        self.cb_phrase_filter.toggled.connect(self._sync_filter_ui)
        self.cb_phrase_filter.toggled.connect(self._on_search_changed)
        self.cmb_phrase_label.currentIndexChanged.connect(self._on_search_changed)
        self.cb_phrase_bpm.toggled.connect(self._sync_filter_ui)
        self.cb_phrase_bpm.toggled.connect(self._on_search_changed)
        self.sp_phrase_bpm.valueChanged.connect(self._on_search_changed)
        self.sp_phrase_bpm_tolerance.valueChanged.connect(self._on_search_changed)
        self.cb_phrase_key.toggled.connect(self._sync_filter_ui)
        self.cb_phrase_key.toggled.connect(self._on_search_changed)
        self.cmb_phrase_key.keyChanged.connect(self._on_search_changed)
        self.cb_phrase_key_compatible.toggled.connect(self._on_search_changed)
        self.cb_key_compatible.toggled.connect(self._on_search_changed)
        self.cb_transition_bpm.toggled.connect(self._sync_filter_ui)
        self.cb_transition_bpm.toggled.connect(self._on_search_changed)
        self.cb_transition_key.toggled.connect(self._sync_filter_ui)
        self.cb_transition_key.toggled.connect(self._on_search_changed)
        self.sp_from_bpm.valueChanged.connect(self._on_search_changed)
        self.sp_to_bpm.valueChanged.connect(self._on_search_changed)
        self.sp_bpm_tolerance.valueChanged.connect(self._on_search_changed)
        self.cb_bpm_first_segment.toggled.connect(self._on_search_changed)
        self.cmb_from_key.keyChanged.connect(self._on_search_changed)
        self.cmb_to_key.keyChanged.connect(self._on_search_changed)
        self.sp_transition_min_duration.valueChanged.connect(self._on_search_changed)
        self.cb_key_first_segment.toggled.connect(self._on_search_changed)
        self.match_toggle.toggled.connect(lambda _on: self._on_search_changed())
        self.btn_clear_filters.clicked.connect(self._clear_filters)

    # Filter
    def _on_search_changed(self, *_args):
        """A filter changed: applied once the changes pause (typing, a held spin box arrow)."""
        self._search_timer.start()

    def _apply_filter(self) -> None:
        """Hide the rows the search and the filters leave out (the rows stay made)."""
        self._search_timer.stop()
        text = self.e_search.text().strip().lower()
        col = str(self.cb_column.currentData() or "All")
        field_map = {
            "#": "_disp_playlist_index",
            "Title": "title",
            "Artist": "artist",
            "Album": "album",
            "Tie-up": "tieup_text",
            "Comment": "comment",
            "Key": "_disp_key_alphanumeric",
            "BPM": "_disp_bpm",
            "Duration": "_disp_duration",
            "Rating": "_disp_rating",
            "Added": "_disp_added",
            "Path": "path",
        }
        conditions: list[set[str]] = []
        if text:
            text_paths: set[str] = set()
            cols = self.HEADERS if col == "All" else [col]
            read_fields = [READ_COLUMNS[name] for name in cols if name in READ_COLUMNS]
            s = self.cfg.searchconfig
            # as typed (not lower-cased): Shift picks the 2-set keyboard's double consonants
            query = reading.query_keys(self.e_search.text().strip(), s.query_languages,
                                       hangul_korean=s.query_hangul_korean, hangul_foreign=s.query_hangul_foreign,
                                       keyboard=s.query_keyboard_layout, han_run=s.han_run()) if read_fields else []
            for t in self._all_tracks:
                for name in cols:
                    attr = field_map.get(name)
                    if not attr:
                        continue
                    val = getattr(t, attr, "")
                    if val and text in str(val).lower():
                        text_paths.add(t.path)
                        break
            if read_fields:
                # by sound: the query read like the readings, normalized, exactly or nearly
                entries = [(t.path, k) for t in self._all_tracks for f in read_fields
                           for k in t.reading_keys.get(f, [])]
                text_paths.update(entries[i][0] for i in reading.find(
                    query, [k for _path, k in entries], similar=self.cb_fuzzy.isChecked()))
            conditions.append(text_paths)

        structured = self._structured_conditions()
        if structured is None:
            return      # applied when the phrase / transition search comes
        conditions.extend(structured)
        if not conditions:
            allowed = None
        elif self.match_toggle.match_all():
            allowed = set.intersection(*conditions)
        else:
            allowed = set.union(*conditions)
        path_col = self.HEADERS.index("Path")
        for r in range(self.table.rowCount()):
            item = self.table.item(r, path_col)
            hidden = allowed is not None and (item is None or item.text() not in allowed)
            if self.table.isRowHidden(r) != hidden:
                self.table.setRowHidden(r, hidden)
        # a hidden row is not acted on: out of the selection
        selection = self.table.selectionModel()
        for index in selection.selectedRows():
            if self.table.isRowHidden(index.row()):
                selection.select(index, QtCore.QItemSelectionModel.Deselect | QtCore.QItemSelectionModel.Rows)

    def _sync_filter_ui(self):
        is_phrase = self.cb_phrase_filter.isChecked()
        is_bpm = self.cb_transition_bpm.isChecked()
        is_key = self.cb_transition_key.isChecked()
        self.card_phrase.setVisible(is_phrase)
        self.card_bpm.setVisible(is_bpm)
        self.card_key.setVisible(is_key)
        self.card_min_sec.setVisible(is_bpm or is_key)
        self.filter_bar.setVisible(is_phrase or is_bpm or is_key)

        phrase_bpm = self.cb_phrase_bpm.isChecked()
        for widget in (self.sp_phrase_bpm, self.lbl_phrase_bpm_tol, self.sp_phrase_bpm_tolerance):
            widget.setEnabled(phrase_bpm)
        phrase_key = self.cb_phrase_key.isChecked()
        self.cmb_phrase_key.setEnabled(phrase_key)
        self.cb_phrase_key_compatible.setEnabled(phrase_key)

        # AND / OR is offered as soon as a filter is on: the text search is the
        # other condition (typed already or about to be).
        has_filter = is_phrase or is_bpm or is_key
        self.match_toggle.setEnabled(has_filter)

    def _clear_filters(self) -> None:
        """Remove every condition: the text and all filter toggles (values stay)."""
        widgets = (
            self.e_search, self.cb_phrase_filter, self.cb_transition_bpm, self.cb_transition_key,
            self.cb_phrase_bpm, self.cb_phrase_key, self.cb_bpm_first_segment,
            self.cb_key_first_segment,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self.e_search.clear()
        for chip in widgets[1:]:
            chip.setChecked(False)
        for widget in widgets:
            widget.blockSignals(False)
        self._sync_filter_ui()
        self._apply_filter()

    def _structured_searches(self) -> tuple:
        """The phrase / transition filters that are on, as the LibraryDB searches they are: (method, args, keyword
        args) each, read from the widgets here and run by the library's owner (jobs.run_searches), one path set
        each."""
        searches = []
        match_all = self.match_toggle.match_all()
        if self.cb_phrase_filter.isChecked():
            bpm_min = bpm_max = None
            if self.cb_phrase_bpm.isChecked():
                center = float(self.sp_phrase_bpm.value())
                scale = max(0.0, float(self.sp_phrase_bpm_tolerance.value())) / 100.0
                # At 0%, still match values that round to the BPM shown by
                # this two-decimal editor (analysis values retain more
                # floating-point precision in the database).
                delta = max(center * scale, 0.5 * 10 ** -self.sp_phrase_bpm.decimals())
                bpm_min = center - delta
                bpm_max = center + delta
            key_value = (
                self.cmb_phrase_key.key()
                if self.cb_phrase_key.isChecked()
                else None
            )
            searches.append(("search_phrase_sections", (str(self.cmb_phrase_label.currentData()),), (
                ("bpm_min", bpm_min), ("bpm_max", bpm_max), ("key_value", key_value),
                ("compatible_keys", self.cb_phrase_key_compatible.isChecked()), ("match_all", match_all))))

        min_duration = float(self.sp_transition_min_duration.value())
        if self.cb_transition_bpm.isChecked():
            center_from = float(self.sp_from_bpm.value())
            center_to = float(self.sp_to_bpm.value())
            tol_pct = max(0.0, float(self.sp_bpm_tolerance.value()))
            scale = tol_pct / 100.0
            from_delta = max(
                center_from * scale, 0.5 * 10 ** -self.sp_from_bpm.decimals())
            to_delta = max(
                center_to * scale, 0.5 * 10 ** -self.sp_to_bpm.decimals())
            searches.append(("search_bpm_transitions", (
                center_from - from_delta, center_from + from_delta, center_to - to_delta, center_to + to_delta), (
                ("min_duration_sec", min_duration), ("require_first_segment", self.cb_bpm_first_segment.isChecked()))))
        if self.cb_transition_key.isChecked():
            searches.append(("search_harmonic_key_transitions", (self.cmb_from_key.key(), self.cmb_to_key.key()), (
                ("min_duration_sec", min_duration), ("require_first_segment", self.cb_key_first_segment.isChecked()),
                ("compatible", self.cb_key_compatible.isChecked()))))
        return tuple(searches)

    def _structured_conditions(self) -> Optional[list[set[str]]]:
        """One path set per phrase / transition filter on; None while their search is on its way (the filter is
        applied again when it comes). The last search's answer is kept until the library changes."""
        searches = self._structured_searches()
        if not searches:
            return []
        if self._structured_found is not None and self._structured_found[0] == searches:
            return self._structured_found[1]
        if self._structured_asked != searches:
            self._structured_asked = searches
            self.library.read(jobs.run_searches, searches, done=lambda found: self._structured_came(searches, found))
        return None

    def _structured_came(self, searches: tuple, found: list) -> None:
        if self._structured_asked == searches:
            self._structured_asked = None
        self._structured_found = (searches, found)
        self._apply_filter()

    def _focus_search(self) -> None:
        self.e_search.setFocus(QtCore.Qt.ShortcutFocusReason)
        self.e_search.selectAll()
