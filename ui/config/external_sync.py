"""Settings > External Sync: the OSC messages another program's deck is followed by (core.external_sync), where
they are received, and what has been received."""

import ipaddress

from PySide6 import QtCore, QtGui
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.external_sync import SAMPLES_ADDRESS, TIME_ADDRESS
from core.i18n import tr
from ui.config.scroll import scroll_page
from ui.live_label import LiveLabel

STATUS_INTERVAL_MS = 500

# the messages as a sender writes them (protocol: not translated)
MESSAGE_FORMAT = (f"{TIME_ADDRESS}     i deck  s path  T|F loaded  T|F active  d|f seconds\n"
                  f"{SAMPLES_ADDRESS}  i deck  s path  T|F loaded  T|F active  h|i samples  i rate")


def _local_address(text: str) -> bool:
    """Whether the address only takes messages from this computer (a literal loopback address, localhost)."""
    text = text.strip()
    if text in ("", "localhost"):
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def build_external_sync_tab(dialog) -> None:
    dialog.tab_external_sync = QWidget()
    root = QVBoxLayout(dialog.tab_external_sync)
    form = QFormLayout()

    dialog.cb_external_sync_enabled = QCheckBox(tr("Enable External Sync"))
    dialog.cb_external_sync_enabled.setToolTip(
        tr("Follow the track and playback position of another program's deck, sent to Mixlyzer as OSC messages."))
    dialog.ed_external_sync_host = QLineEdit()
    dialog.ed_external_sync_host.setPlaceholderText("127.0.0.1")
    dialog.ed_external_sync_host.setToolTip(
        tr("The address the messages are received on. 127.0.0.1 takes them from this computer only; the address "
           "of a network interface (or 0.0.0.0, all of them) takes them from that network too."))
    dialog.sp_external_sync_port = QSpinBox()
    dialog.sp_external_sync_port.setRange(1, 65535)
    dialog.sp_external_sync_port.setToolTip(tr("The UDP port the messages are sent to."))

    dialog.lbl_external_sync_network = QLabel(
        tr("Messages are taken from the network: any device on it can drive the deck. Only tracks already in the "
           "library are followed (none is analyzed for it)."))
    dialog.lbl_external_sync_network.setWordWrap(True)
    dialog.lbl_external_sync_network.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    dialog.lbl_external_sync_network.setStyleSheet("color: #d8a441;")

    dialog.lbl_external_sync_status = LiveLabel("")
    dialog.lbl_external_sync_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    dialog.lbl_external_sync_format = QLabel(MESSAGE_FORMAT)
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
    dialog.lbl_external_sync_format.setFont(font)
    dialog.lbl_external_sync_format.setTextFormat(QtCore.Qt.PlainText)
    dialog.lbl_external_sync_format.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    dialog.lbl_external_sync_format_note = QLabel(
        tr("One message per deck, as often as the sender reads its state. The deck followed is the lowest-numbered "
           "one that is loaded and active; a deck not heard from for a second is no longer followed. rate is the "
           "sample rate the samples are counted at (0: the file's own); a flag may also be sent as an integer "
           "0 / 1. The path must be a file on this computer."))
    dialog.lbl_external_sync_format_note.setWordWrap(True)
    dialog.lbl_external_sync_format_note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    dialog.lbl_external_sync_note = QLabel(
        tr("When enabled, local track loading and transport playback are restricted "
           "so the external software remains the source of truth."))
    dialog.lbl_external_sync_note.setWordWrap(True)

    form.addRow(dialog.cb_external_sync_enabled)
    form.addRow(tr("Address"), dialog.ed_external_sync_host)
    form.addRow(tr("Port"), dialog.sp_external_sync_port)
    form.addRow(dialog.lbl_external_sync_network)
    form.addRow(tr("Status"), dialog.lbl_external_sync_status)
    form.addRow(dialog.lbl_external_sync_note)
    root.addLayout(form)
    root.addWidget(QLabel(tr("Messages")))
    root.addWidget(dialog.lbl_external_sync_format)
    root.addWidget(dialog.lbl_external_sync_format_note)
    root.addStretch(1)

    dialog.ed_external_sync_host.textChanged.connect(dialog._sync_external_sync_address_ui)
    dialog._external_sync_status_timer = QtCore.QTimer(dialog)
    dialog._external_sync_status_timer.setInterval(STATUS_INTERVAL_MS)
    dialog._external_sync_status_timer.timeout.connect(dialog._refresh_external_sync_status)
    dialog._external_sync_status_timer.start()
    dialog._external_sync_status_provider = None
    dialog._sync_external_sync_address_ui()
    dialog.tabs.addTab(scroll_page(dialog.tab_external_sync), tr("External Sync"))


class ExternalSyncSettingsMixin:
    def set_external_sync_status(self, provider) -> None:
        """``provider()`` -> the controller's snapshot (ExternalSyncController.snapshot), shown as the status."""
        self._external_sync_status_provider = provider
        self._refresh_external_sync_status()

    def _sync_external_sync_address_ui(self) -> None:
        self.lbl_external_sync_network.setVisible(not _local_address(self.ed_external_sync_host.text()))

    def _external_sync_host(self) -> str:
        return self.ed_external_sync_host.text().strip() or "127.0.0.1"

    def _refresh_external_sync_status(self) -> None:
        if not self.tab_external_sync.isVisible() or self._external_sync_status_provider is None:
            return
        try:
            state = self._external_sync_status_provider()
        except Exception:
            return
        self.lbl_external_sync_status.setText(_status_text(state))


def _status_text(state: dict) -> str:
    if not state.get("enabled"):
        return tr("Off.")
    sync = state.get("osc") or {}
    listening = str(sync.get("listening") or "")
    if not listening:
        return tr("Not listening.")
    decks = sync.get("decks") or {}
    if not decks:
        return tr("Listening on {address}; nothing received yet.").format(address=listening)
    newest = min(float(d.get("age_sec") or 0.0) for d in decks.values())
    following = int(sync.get("following") or 0)
    if following:
        return tr("Listening on {address}; following deck {deck} (last message {age} s ago, from {sender}).").format(
            address=listening, deck=following, age=f"{newest:.1f}", sender=sync.get("sender") or "?")
    return tr("Listening on {address}; no deck loaded and active (last message {age} s ago, from {sender}).").format(
        address=listening, age=f"{newest:.1f}", sender=sync.get("sender") or "?")
