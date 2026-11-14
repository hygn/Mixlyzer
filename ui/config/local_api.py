"""Settings > Library > Local Apps: the local API (core.local_api) on or off and its port (config.json, saved with
the dialog), what it is doing, Pair an App... (the pairing, `PairingDialog`) and the paired apps (unpaired at once,
asked first: the paired apps' file is the local API's, not a setting)."""
from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from core.local_api.pairings import Phase, Verdict
from core.local_api.tokens import READ, WRITE
from system.i18n import tr, tr_mark, tr_n

SCOPE_NAMES = {READ: tr_mark("Analyze and read"), WRITE: tr_mark("Save to the library")}
REFRESH_MS = 500


def _scopes_text(scopes) -> str:
    return ", ".join(tr(SCOPE_NAMES[s]) for s in scopes if s in SCOPE_NAMES)


def _time_text(iso: str) -> str:
    """A time the paired apps' file keeps (ISO, to the second) as shown: date and minutes."""
    return iso.replace("T", " ")[:16] if iso else tr("Never")


def build_local_apps_page(dialog) -> QtWidgets.QGroupBox:
    """The Local Apps group (its page: ui.config.library)."""
    group = QtWidgets.QGroupBox(tr("Local Apps"))
    form = QtWidgets.QFormLayout(group)
    dialog.cb_local_api = QtWidgets.QCheckBox(tr("Let apps on this computer ask for analyses"))
    dialog.cb_local_api.setToolTip(tr(
        "<p>A paired program on this computer sends a file's path over HTTP (127.0.0.1) and gets its analysis as "
        "JAMS.</p>"))
    dialog.sp_local_api_port = QtWidgets.QSpinBox()
    dialog.sp_local_api_port.setRange(1024, 65535)
    dialog.sp_local_api_port.setToolTip(tr("<p>The port on 127.0.0.1 the apps connect to.</p>"))
    dialog.lbl_local_api_status = QtWidgets.QLabel()
    dialog.lbl_local_api_status.setWordWrap(True)
    dialog.lbl_local_api_status.setTextFormat(QtCore.Qt.PlainText)
    dialog.btn_pair_app = QtWidgets.QPushButton(tr("Pair an App…"))
    dialog.btn_pair_app.setToolTip(tr(
        "<p>For two minutes an app may ask to pair: type the 4-digit code it shows.</p>"))
    dialog.tbl_paired_apps = QtWidgets.QTableWidget(0, 4)
    dialog.tbl_paired_apps.setHorizontalHeaderLabels([tr("App"), tr("Permissions"), tr("Paired"), tr("Last Used")])
    dialog.tbl_paired_apps.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    dialog.tbl_paired_apps.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    dialog.tbl_paired_apps.verticalHeader().setVisible(False)
    dialog.tbl_paired_apps.horizontalHeader().setStretchLastSection(True)
    dialog.tbl_paired_apps.setMinimumHeight(140)
    dialog.btn_unpair_apps = QtWidgets.QPushButton(tr("Unpair"))
    dialog.btn_unpair_apps.setToolTip(tr("<p>The selected apps' tokens stop working at once.</p>"))
    dialog.lbl_local_api_note = QtWidgets.QLabel(tr("The apps reach Mixlyzer while it runs. Mixlyzer takes a full "
                                                    "path to an audio file on this computer, never a network path."))
    dialog.lbl_local_api_note.setWordWrap(True)

    form.addRow(dialog.cb_local_api)
    form.addRow(tr("Port"), dialog.sp_local_api_port)
    form.addRow(dialog.lbl_local_api_status)
    form.addRow("", dialog.btn_pair_app)
    form.addRow(dialog.tbl_paired_apps)
    buttons = QtWidgets.QHBoxLayout()
    buttons.addStretch(1)
    buttons.addWidget(dialog.btn_unpair_apps)
    form.addRow(buttons)
    form.addRow(dialog.lbl_local_api_note)

    dialog.btn_pair_app.clicked.connect(dialog._pair_app)
    dialog.btn_unpair_apps.clicked.connect(dialog._unpair_apps)
    dialog.tbl_paired_apps.itemSelectionChanged.connect(dialog._refresh_unpair)
    return group


class PairingDialog(QtWidgets.QDialog):
    """Pair an App: pairing opened while it is shown; an app asking shows its name and permissions, and the code it
    shows is typed here (wrong: the tries left said); closed before an answer, the pairing is closed (an app asking
    refused). Follows the local API's pairing (`LocalApi.pairing_changed`, and its time every REFRESH_MS)."""

    def __init__(self, api, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._api = api
        self.setWindowTitle(tr("Pair an App"))
        self.setModal(False)
        self.setMinimumWidth(460)
        self.lbl_state = QtWidgets.QLabel()
        self.lbl_state.setWordWrap(True)
        self.lbl_state.setTextFormat(QtCore.Qt.PlainText)        # an app's name is its own text
        self.lbl_warning = QtWidgets.QLabel(tr("The name is the one the app gave: pair only an app you started."))
        self.lbl_warning.setWordWrap(True)
        self.ed_code = QtWidgets.QLineEdit()
        self.ed_code.setMaxLength(4)
        self.ed_code.setValidator(QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"\d{0,4}"), self))
        self.ed_code.setPlaceholderText("0000")
        font = self.ed_code.font()
        font.setPointSize(font.pointSize() + 8)
        font.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, 6)
        self.ed_code.setFont(font)
        self.ed_code.setAlignment(QtCore.Qt.AlignCenter)
        self.ed_code.returnPressed.connect(self._enter)
        self.btn_pair = QtWidgets.QPushButton(tr("Pair"))
        self.btn_pair.setDefault(True)
        self.btn_pair.clicked.connect(self._enter)
        self.btn_refuse = QtWidgets.QPushButton(tr("Refuse"))
        self.btn_refuse.clicked.connect(self._api.store.refuse)
        self.btn_close = QtWidgets.QPushButton(tr("Close"))
        self.btn_close.clicked.connect(self.close)
        self.lbl_code = QtWidgets.QLabel(tr("Code shown in the app"))
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        for button in (self.btn_pair, self.btn_refuse, self.btn_close):
            row.addWidget(button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.lbl_state)
        layout.addWidget(self.lbl_warning)
        layout.addWidget(self.lbl_code)
        layout.addWidget(self.ed_code)
        layout.addLayout(row)
        self._tries_said = ""
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        api.pairing_changed.connect(self.refresh)

    def start(self) -> None:
        """Pairing opened (again), the dialog shown."""
        self._tries_said = ""
        self._api.open_pairing()
        self._timer.start()
        self.refresh()
        self.show()
        self.raise_()
        self.activateWindow()

    @QtCore.Slot()
    def refresh(self) -> None:
        view = self._api.store.view()
        asking = view.asking if view.phase is Phase.ASKED else None
        left = int(round(view.seconds_left))
        time_left = f"{left // 60}:{left % 60:02d}"
        if view.phase is Phase.OPEN:
            text = tr("Waiting for an app to ask to pair ({time} left)…").format(time=time_left)
        elif asking is not None:
            text = tr("{name} asks to pair, with these permissions: {permissions}. Type the code it shows "
                      "({time} left).").format(name=asking.name, permissions=_scopes_text(asking.scopes),
                                              time=time_left)
            if self._tries_said:
                text += "\n" + self._tries_said
        else:
            verdict, name = view.last if view.last else (Verdict.UNKNOWN, "")
            text = {Verdict.PAIRED: tr("{name} is paired.").format(name=name),
                    Verdict.DENIED: tr("The pairing was refused."),
                    Verdict.EXPIRED: tr("The time ran out: pair again to let an app ask.")}.get(
                        verdict, tr("Pairing is closed."))
            self._timer.stop()
        self.lbl_state.setText(text)
        for widget in (self.lbl_warning, self.lbl_code, self.ed_code, self.btn_pair, self.btn_refuse):
            widget.setVisible(asking is not None)
        if asking is not None and not self.ed_code.hasFocus():
            self.ed_code.setFocus()

    def _enter(self) -> None:
        verdict = self._api.store.enter(self.ed_code.text())
        self.ed_code.clear()
        if verdict is Verdict.PENDING:
            asking = self._api.store.view().asking
            left = asking.tries_left if asking is not None else 0
            self._tries_said = tr_n("Wrong code: {count} try left.", "Wrong code: {count} tries left.",
                                    left).format(count=left)
        self.refresh()

    def closeEvent(self, event) -> None:
        """Closed before an answer: the pairing closed (an app asking refused)."""
        self._timer.stop()
        if self._api.store.view().phase is not Phase.CLOSED:
            self._api.close_pairing()
        super().closeEvent(event)


class LocalApiSettingsMixin:
    """The settings dialog's side of the local API (set_local_api: the app's LocalApi)."""

    def set_local_api(self, api) -> None:
        self._local_api = api
        self._pairing_dialog: Optional[PairingDialog] = None
        api.status_changed.connect(self._refresh_local_api)
        api.pairing_changed.connect(self._refresh_local_api)
        self._refresh_local_api()

    def _set_local_api_config(self, lib) -> None:
        self.cb_local_api.setChecked(bool(lib.local_api))
        self.sp_local_api_port.setValue(int(lib.local_api_port))

    def _get_local_api_config(self) -> dict:
        return {"local_api": bool(self.cb_local_api.isChecked()),
                "local_api_port": int(self.sp_local_api_port.value())}

    @QtCore.Slot()
    def _refresh_local_api(self) -> None:
        api = getattr(self, "_local_api", None)
        if api is None:
            self.btn_pair_app.setEnabled(False)
            self.btn_unpair_apps.setEnabled(False)
            return
        status = api.status()
        if status.listening:
            text = tr("Listening on {address}.").format(address=f"127.0.0.1:{status.port}")
        elif status.on and status.error:
            text = tr("Port {port} could not be opened: {error}").format(port=status.port, error=status.error)
        else:
            text = tr("Off. Turn it on and apply to pair an app.")
        if api.store.problem:
            text += "\n" + tr("The paired apps' file could not be read: every app pairs again.")
        self.lbl_local_api_status.setText(text)
        self.btn_pair_app.setEnabled(status.listening)
        selected = {i.data(QtCore.Qt.UserRole) for i in self.tbl_paired_apps.selectedItems() if i.column() == 0}
        self.tbl_paired_apps.setRowCount(0)
        for pairing in api.store.pairings():
            row = self.tbl_paired_apps.rowCount()
            self.tbl_paired_apps.insertRow(row)
            cells = (pairing.name, _scopes_text(pairing.scopes), _time_text(pairing.paired),
                     _time_text(pairing.last_used))
            for column, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(text)
                if column == 0:
                    item.setData(QtCore.Qt.UserRole, pairing.token_id)
                self.tbl_paired_apps.setItem(row, column, item)
            if pairing.token_id in selected:
                self.tbl_paired_apps.selectRow(row)
        self._refresh_unpair()

    @QtCore.Slot()
    def _refresh_unpair(self) -> None:
        self.btn_unpair_apps.setEnabled(bool(self.tbl_paired_apps.selectedItems()))

    def _pair_app(self) -> None:
        api = getattr(self, "_local_api", None)
        if api is None or not api.status().listening:
            return
        if self._pairing_dialog is None:
            self._pairing_dialog = PairingDialog(api, self)
        self._pairing_dialog.start()

    def _unpair_apps(self) -> None:
        api = getattr(self, "_local_api", None)
        rows = sorted({i.row() for i in self.tbl_paired_apps.selectedItems()})
        if api is None or not rows:
            return
        token_ids = [self.tbl_paired_apps.item(r, 0).data(QtCore.Qt.UserRole) for r in rows]
        names = ", ".join(self.tbl_paired_apps.item(r, 0).text() for r in rows)
        answer = QtWidgets.QMessageBox.question(
            self, tr("Unpair"),
            tr_n("Unpair {names}? It pairs again to use Mixlyzer.",
                 "Unpair {count} apps ({names})? They pair again to use Mixlyzer.", len(rows)).format(
                count=len(rows), names=names),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            return
        for token_id in token_ids:
            api.store.unpair(token_id)
