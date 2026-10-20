from PySide6 import QtCore
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QWidget,
)

from core import reading
from core.config import searchconfig
from ui.config.dialogs import ReadingRefreshProgressDialog
from ui.config.scroll import scroll_page

_LANGUAGE_TIPS = {
    "ja": "Japanese, by word (SudachiDict): 東京 toukyou.",
    "zh": "Mandarin pinyin, tones dropped: 東京 dong jing.",
    "yue": "Cantonese Jyutping, tones dropped: 東京 dung ging.",
    "ko-hanja": "Korean Hanja reading (Unihan), spelled as a word: 東京 donggyeong.",
    "vi": "Sino-Vietnamese reading (Unihan), diacritics dropped: 東京 dong kinh.",
    "zh-tw": "Taiwan's Mandarin, kept only where it differs from Mandarin: 乾杯 gan bei.",
}


def _language_boxes(dialog, prefix: str) -> tuple[QWidget, dict[str, QCheckBox]]:
    """A check box per Han language (three a row); at least one stays checked."""
    box = QWidget()
    grid = QGridLayout(box)
    grid.setContentsMargins(0, 0, 0, 0)
    boxes: dict[str, QCheckBox] = {}
    for i, lang in enumerate(reading.HAN_LANGUAGES):
        cb = QCheckBox(reading.SOURCES[lang])
        cb.setToolTip(f"<p>{_LANGUAGE_TIPS[lang]}</p>")
        grid.addWidget(cb, i // 3, i % 3)
        boxes[lang] = cb
        setattr(dialog, f"cb_{prefix}_{lang.replace('-', '_')}", cb)
    _keep_one_checked(list(boxes.values()))
    return box, boxes


def _keep_one_checked(boxes: list[QCheckBox]) -> None:
    def toggled(checked: bool, cb: QCheckBox) -> None:
        if not checked and not any(b.isChecked() for b in boxes):
            cb.setChecked(True)
    for cb in boxes:
        cb.toggled.connect(lambda checked, cb=cb: toggled(checked, cb))


def build_search_tab(dialog) -> None:
    dialog.tab_search = QWidget()
    form = QFormLayout(dialog.tab_search)

    # the library's stored readings
    grp_db = QGroupBox("Library Readings")
    f_db = QFormLayout(grp_db)
    db_box, dialog.db_language_boxes = _language_boxes(dialog, "db")
    db_box.setToolTip(
        "<p>The languages a title, artist or album of Han characters without kana is read in and "
        "stored for the search, one reading each. A change reads the library again when saved.</p>")
    f_db.addRow("Han Characters", db_box)
    dialog.lbl_db_languages_note = QLabel(
        "Kana is read in Japanese, Hangul in Korean, other scripts as written. "
        "Readings set in Edit Track are kept.")
    dialog.lbl_db_languages_note.setWordWrap(True)
    f_db.addRow(dialog.lbl_db_languages_note)
    dialog.btn_reread_library = QPushButton("Re-read Library")
    dialog.btn_reread_library.setToolTip(
        "<p>Save these settings and read every track's title, artist and album again, in the "
        "background. Readings set in Edit Track are kept.</p>")
    f_db.addRow("", dialog.btn_reread_library)

    # the query
    grp_query = QGroupBox("Search Query")
    f_query = QFormLayout(grp_query)
    query_box, dialog.query_language_boxes = _language_boxes(dialog, "query")
    query_box.setToolTip(
        "<p>The languages Han characters in a query are read in: a track is found when any of them "
        "matches any of its readings.</p>")
    f_query.addRow("Han Characters", query_box)
    hangul_box = QWidget()
    hangul_grid = QGridLayout(hangul_box)
    hangul_grid.setContentsMargins(0, 0, 0, 0)
    dialog.cb_query_hangul_korean = QCheckBox("As Korean")
    dialog.cb_query_hangul_korean.setToolTip(
        "<p>Hangul read in Revised Romanization: 사회 finds a Korean title and 社会 by its Korean "
        "Hanja reading.</p>")
    dialog.cb_query_hangul_foreign = QCheckBox("As a foreign word")
    dialog.cb_query_hangul_foreign.setToolTip(
        "<p>Hangul read as Korean spells a foreign word (ㅓ o, ㅡ u, ㅐ e): 도쿄 finds 東京, 아노네 "
        "finds あのね.</p>")
    hangul_grid.addWidget(dialog.cb_query_hangul_korean, 0, 0)
    hangul_grid.addWidget(dialog.cb_query_hangul_foreign, 0, 1)
    hangul_grid.addWidget(QWidget(), 0, 2)
    _keep_one_checked([dialog.cb_query_hangul_korean, dialog.cb_query_hangul_foreign])
    f_query.addRow("Hangul", hangul_box)

    form.addRow(grp_db)
    form.addRow(grp_query)
    dialog.btn_reread_library.clicked.connect(dialog._reread_library_now)
    dialog.tabs.addTab(scroll_page(dialog.tab_search), "Search")


class SearchSettingsMixin:
    def _set_search_config(self, s: searchconfig) -> None:
        for boxes, chosen in ((self.db_language_boxes, reading.han_languages(s.db_languages)),
                              (self.query_language_boxes, reading.han_languages(s.query_languages))):
            # the chosen checked first: one box stays checked throughout
            for lang in sorted(boxes, key=lambda lang: lang not in chosen):
                boxes[lang].setChecked(lang in chosen)
        foreign = bool(s.query_hangul_foreign)
        korean = bool(s.query_hangul_korean) or not foreign
        for cb, on in sorted(((self.cb_query_hangul_korean, korean), (self.cb_query_hangul_foreign, foreign)),
                             key=lambda pair: not pair[1]):
            cb.setChecked(on)

    def _get_search_config(self) -> searchconfig:
        return searchconfig(
            db_languages=[lang for lang, cb in self.db_language_boxes.items() if cb.isChecked()],
            query_languages=[lang for lang, cb in self.query_language_boxes.items() if cb.isChecked()],
            query_hangul_korean=bool(self.cb_query_hangul_korean.isChecked()),
            query_hangul_foreign=bool(self.cb_query_hangul_foreign.isChecked()),
        )

    def _initialize_search_settings(self, bus) -> None:
        self._reading_refresh_dialog: ReadingRefreshProgressDialog | None = None
        if bus is None:
            return
        bus.sig_reading_refresh_started.connect(self._on_reading_refresh_started)
        bus.sig_reading_refresh_progress.connect(self._on_reading_refresh_progress)
        bus.sig_reading_refresh_finished.connect(self._on_reading_refresh_finished)
        bus.sig_reading_refresh_failed.connect(self._on_reading_refresh_failed)

    def _reread_library_now(self) -> None:
        if self._bus is None:
            return
        cfg = self.get_config()
        previous = getattr(self._current_cfg, "searchconfig", None)
        # changed languages: saving them starts the re-read (app.window); else it is asked here
        languages_changed = previous is None or list(previous.db_languages) != cfg.searchconfig.db_languages
        self.saveJsonRequested.emit(cfg)
        if not languages_changed:
            self._bus.sig_reading_refresh_requested.emit(None)

    @QtCore.Slot(object)
    def _on_reading_refresh_started(self, _request) -> None:
        self.btn_reread_library.setEnabled(False)
        if self._reading_refresh_dialog is None:
            self._reading_refresh_dialog = ReadingRefreshProgressDialog(self)
        self._reading_refresh_dialog.set_started()
        self._reading_refresh_dialog.show()
        self._reading_refresh_dialog.raise_()

    @QtCore.Slot(object, int, str)
    def _on_reading_refresh_progress(self, _request, value: int, message: str) -> None:
        if self._reading_refresh_dialog is not None:
            self._reading_refresh_dialog.set_progress(value, message)

    @QtCore.Slot(object, object)
    def _on_reading_refresh_finished(self, _request, result) -> None:
        self.btn_reread_library.setEnabled(True)
        if self._reading_refresh_dialog is not None:
            self._reading_refresh_dialog.set_finished(result)

    @QtCore.Slot(object, str)
    def _on_reading_refresh_failed(self, _request, message: str) -> None:
        self.btn_reread_library.setEnabled(True)
        if self._reading_refresh_dialog is not None:
            self._reading_refresh_dialog.set_failed(message)
