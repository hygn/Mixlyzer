from PySide6 import QtCore
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from core import reading
from core.config import searchconfig
from ui.config.dialogs import MusicIndexProgressDialog, ReadingRefreshProgressDialog
from ui.config.scroll import scroll_page
from foundation.i18n import tr, tr_mark

_LANGUAGE_TIPS = {
    "ja": tr_mark("Japanese, by word (SudachiDict): 東京 toukyou."),
    "zh": tr_mark("Mandarin pinyin, tones dropped: 東京 dong jing."),
    "yue": tr_mark("Cantonese Jyutping, tones dropped: 東京 dung ging."),
    "ko-hanja": tr_mark("Korean Hanja reading (Unihan), spelled as a word: 東京 donggyeong."),
    "vi": tr_mark("Sino-Vietnamese reading (Unihan), diacritics dropped: 東京 dong kinh."),
    "zh-tw": tr_mark("Taiwan's Mandarin, kept only where it differs from Mandarin: 乾杯 gan bei."),
}


def _language_boxes(dialog, prefix: str) -> tuple[QWidget, dict[str, QCheckBox]]:
    """A check box per Han language (three a row); at least one stays checked."""
    box = QWidget()
    grid = QGridLayout(box)
    grid.setContentsMargins(0, 0, 0, 0)
    boxes: dict[str, QCheckBox] = {}
    for i, lang in enumerate(reading.HAN_LANGUAGES):
        cb = QCheckBox(tr(reading.SOURCES[lang]))
        cb.setToolTip(f"<p>{tr(_LANGUAGE_TIPS[lang])}</p>")
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
    grp_db = QGroupBox(tr("Library Readings"))
    f_db = QFormLayout(grp_db)
    db_box, dialog.db_language_boxes = _language_boxes(dialog, "db")
    db_box.setToolTip(
        tr("<p>The languages a title, artist or album of Han characters without kana is read in and "
           "stored for the search, one reading each. A change reads the library again when saved.</p>"))
    f_db.addRow(tr("Han Characters"), db_box)
    # Han mixed with other characters: its runs in Korean Hanja too
    mixed = QWidget()
    row = QHBoxLayout(mixed)
    row.setContentsMargins(0, 0, 0, 0)
    dialog.cb_han_runs = QCheckBox(tr("Han runs in Korean Hanja, from"))
    dialog.sp_han_run_min = QSpinBox()
    dialog.sp_han_run_min.setRange(reading.HAN_RUN_LEAST, 10)
    dialog.sp_han_run_min.setSuffix(tr(" characters"))
    mixed.setToolTip(
        tr("<p>Han characters mixed with other characters are found by their Korean Hanja reading too. "
           "Beside Latin letters or Hangul all of them are read so; beside kana (read in Japanese with it) "
           "the runs of this many or more, the rest staying Japanese (Re:ゼロから始める異世界生活: zero kara "
           "hajimeru isegyesaenghwal; shorter runs there are mostly words read with the kana: 君, 始める). "
           "Needs Korean Hanja checked.</p>"))
    row.addWidget(dialog.cb_han_runs)
    row.addWidget(dialog.sp_han_run_min)
    row.addStretch(1)
    f_db.addRow(tr("Mixed Text"), mixed)

    def sync_han_runs(_checked: bool = False) -> None:
        mixed.setEnabled(dialog.db_language_boxes["ko-hanja"].isChecked())
        dialog.sp_han_run_min.setEnabled(dialog.cb_han_runs.isChecked())
    dialog.db_language_boxes["ko-hanja"].toggled.connect(sync_han_runs)
    dialog.cb_han_runs.toggled.connect(sync_han_runs)
    dialog._sync_han_runs = sync_han_runs
    dialog.lbl_db_languages_note = QLabel(
        tr("Kana is read in Japanese, Hangul in Korean, other scripts as written. "
           "Readings set in Edit Track are kept."))
    dialog.lbl_db_languages_note.setWordWrap(True)
    f_db.addRow(dialog.lbl_db_languages_note)
    dialog.btn_reread_library = QPushButton(tr("Re-read Library"))
    dialog.btn_reread_library.setToolTip(
        tr("<p>Save these settings and read every track's title, artist and album again, in the "
           "background. Readings set in Edit Track are kept.</p>"))
    f_db.addRow("", dialog.btn_reread_library)

    # the query
    grp_query = QGroupBox(tr("Search Query"))
    f_query = QFormLayout(grp_query)
    query_box, dialog.query_language_boxes = _language_boxes(dialog, "query")
    query_box.setToolTip(
        tr("<p>The languages Han characters in a query are read in: a track is found when any of them "
           "matches any of its readings.</p>"))
    f_query.addRow(tr("Han Characters"), query_box)
    hangul_box = QWidget()
    hangul_grid = QGridLayout(hangul_box)
    hangul_grid.setContentsMargins(0, 0, 0, 0)
    dialog.cb_query_hangul_korean = QCheckBox(tr("As Korean"))
    dialog.cb_query_hangul_korean.setToolTip(
        tr("<p>Hangul read in Revised Romanization: 사회 finds a Korean title and 社会 by its Korean "
           "Hanja reading.</p>"))
    dialog.cb_query_hangul_foreign = QCheckBox(tr("As a foreign word"))
    dialog.cb_query_hangul_foreign.setToolTip(
        tr("<p>Hangul read as Korean spells a foreign word (ㅓ o, ㅡ u, ㅐ e): 도쿄 finds 東京, 아노네 "
           "finds あのね.</p>"))
    dialog.cb_query_keyboard_layout = QCheckBox(tr("Typed without 한/영"))
    dialog.cb_query_keyboard_layout.setToolTip(
        tr("<p>A query typed on the 2-set (Dubeolsik) keyboard in the wrong input mode is also read as "
           "meant: tkghl as 사회 (whole syllables, two or more), 무ㅐㅜㄷ as anone.</p>"))
    hangul_grid.addWidget(dialog.cb_query_hangul_korean, 0, 0)
    hangul_grid.addWidget(dialog.cb_query_hangul_foreign, 0, 1)
    hangul_grid.addWidget(dialog.cb_query_keyboard_layout, 0, 2)
    _keep_one_checked([dialog.cb_query_hangul_korean, dialog.cb_query_hangul_foreign])
    f_query.addRow(tr("Hangul"), hangul_box)

    # Search by Music (analyzer.search.music)
    grp_music = QGroupBox(tr("Search by Music"))
    f_music = QFormLayout(grp_music)
    dialog.cb_music_index = QCheckBox(tr("Index tracks when analyzing"))
    dialog.cb_music_index.setToolTip(
        tr("<p>The analysis also indexes each track for Search by Music: its melody (the lead and the top line) "
           "and fingerprints of its sound, about 6 seconds more a track.</p>"))
    f_music.addRow(dialog.cb_music_index)
    dialog.sp_music_window = QDoubleSpinBox()
    dialog.sp_music_window.setRange(4.0, 30.0)
    dialog.sp_music_window.setDecimals(0)
    dialog.sp_music_window.setSuffix(tr(" s"))
    dialog.sp_music_window.setToolTip(
        tr("<p>While listening, the search looks at this many seconds of what was heard last. A melody is "
           "found best from 8 to 15 seconds of it.</p>"))
    f_music.addRow(tr("Listen to the last"), dialog.sp_music_window)
    dialog.sp_music_interval = QDoubleSpinBox()
    dialog.sp_music_interval.setRange(1.0, 10.0)
    dialog.sp_music_interval.setDecimals(1)
    dialog.sp_music_interval.setSingleStep(0.5)
    dialog.sp_music_interval.setSuffix(tr(" s"))
    dialog.sp_music_interval.setToolTip(tr("<p>While listening, a search is made this often.</p>"))
    f_music.addRow(tr("Search every"), dialog.sp_music_interval)
    dialog.lbl_music_index_status = QLabel()
    dialog.lbl_music_index_status.setWordWrap(True)
    f_music.addRow(dialog.lbl_music_index_status)
    dialog.btn_music_index_now = QPushButton(tr("Index Now"))
    dialog.btn_music_index_now.setToolTip(
        tr("<p>Index the library's tracks that are not indexed yet for Search by Music or for the library's "
           "Variants (analyzed before they existed, or with indexing off), in the background; tracks gone from the "
           "library leave the index. The Variants are found from the stored analysis, without reading the audio.</p>"))
    f_music.addRow("", dialog.btn_music_index_now)

    form.addRow(grp_db)
    form.addRow(grp_query)
    form.addRow(grp_music)
    dialog.btn_reread_library.clicked.connect(dialog._reread_library_now)
    dialog.btn_music_index_now.clicked.connect(dialog._music_index_now)
    dialog.tabs.addTab(scroll_page(dialog.tab_search), tr("Search"))


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
        self.cb_query_keyboard_layout.setChecked(bool(s.query_keyboard_layout))
        self.cb_han_runs.setChecked(bool(s.han_runs))
        self.sp_han_run_min.setValue(max(reading.HAN_RUN_LEAST, int(s.han_run_min)))
        self._sync_han_runs()
        self.cb_music_index.setChecked(bool(s.music_index))
        self.sp_music_window.setValue(float(s.music_listen_window))
        self.sp_music_interval.setValue(float(s.music_listen_interval))
        self._show_music_index_status()

    def _get_search_config(self) -> searchconfig:
        return searchconfig(
            db_languages=[lang for lang, cb in self.db_language_boxes.items() if cb.isChecked()],
            han_runs=bool(self.cb_han_runs.isChecked()),
            han_run_min=int(self.sp_han_run_min.value()),
            query_languages=[lang for lang, cb in self.query_language_boxes.items() if cb.isChecked()],
            query_hangul_korean=bool(self.cb_query_hangul_korean.isChecked()),
            query_hangul_foreign=bool(self.cb_query_hangul_foreign.isChecked()),
            query_keyboard_layout=bool(self.cb_query_keyboard_layout.isChecked()),
            music_index=bool(self.cb_music_index.isChecked()),
            music_listen_window=float(self.sp_music_window.value()),
            music_listen_interval=float(self.sp_music_interval.value()),
        )

    def _initialize_search_settings(self, bus) -> None:
        self._reading_refresh_dialog: ReadingRefreshProgressDialog | None = None
        self._music_index_dialog: MusicIndexProgressDialog | None = None
        if bus is None:
            return
        bus.sig_music_index_started.connect(self._on_music_index_started)
        bus.sig_music_index_progress.connect(self._on_music_index_progress)
        bus.sig_music_index_finished.connect(self._on_music_index_finished)
        bus.sig_music_index_failed.connect(self._on_music_index_failed)
        bus.sig_reading_refresh_started.connect(self._on_reading_refresh_started)
        bus.sig_reading_refresh_progress.connect(self._on_reading_refresh_progress)
        bus.sig_reading_refresh_finished.connect(self._on_reading_refresh_finished)
        bus.sig_reading_refresh_failed.connect(self._on_reading_refresh_failed)

    def _reread_library_now(self) -> None:
        if self._bus is None:
            return
        cfg = self.get_config()
        previous = getattr(self._current_cfg, "searchconfig", None)
        # what the stored readings depend on changed: saving it starts the re-read (app.window); else
        # it is asked here
        changed = previous is None or previous.stored_readings() != cfg.searchconfig.stored_readings()
        self.saveJsonRequested.emit(cfg)
        if not changed:
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

    def _show_music_index_status(self) -> None:
        """How many tracks are indexed, read by the library's owner (set_library) and shown when it comes."""
        from core.library import jobs
        library = getattr(self, "_library", None)
        if library is None:
            return
        library.read(jobs.index_status, done=lambda counts: self.lbl_music_index_status.setText(
            tr("{indexed} of {total} library tracks are indexed.").format(indexed=counts[0], total=counts[1])))

    def _music_index_now(self) -> None:
        if self._bus is None:
            return
        self.saveJsonRequested.emit(self.get_config())
        self._bus.sig_music_index_requested.emit(None)

    @QtCore.Slot(object)
    def _on_music_index_started(self, run) -> None:
        self.btn_music_index_now.setEnabled(False)
        if self._music_index_dialog is None:
            self._music_index_dialog = MusicIndexProgressDialog(self)
        self._music_index_dialog.set_started(run)
        self._music_index_dialog.show()
        self._music_index_dialog.raise_()

    @QtCore.Slot(object, int, str)
    def _on_music_index_progress(self, _run, value: int, message: str) -> None:
        if self._music_index_dialog is not None:
            self._music_index_dialog.set_progress(value, message)

    @QtCore.Slot(object, object)
    def _on_music_index_finished(self, _run, summary) -> None:
        self.btn_music_index_now.setEnabled(True)
        if self._music_index_dialog is not None:
            self._music_index_dialog.set_finished(summary)
        self._show_music_index_status()

    @QtCore.Slot(object, str)
    def _on_music_index_failed(self, _run, message: str) -> None:
        self.btn_music_index_now.setEnabled(True)
        if self._music_index_dialog is not None:
            self._music_index_dialog.set_failed(message)
        self._show_music_index_status()
