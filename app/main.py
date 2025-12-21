import sys, os
from PySide6 import QtWidgets, QtGui, QtCore
from app.window import AppWindow
from utils.fonts import load_fonts_and_set_global


def _apply_dark_theme(app: QtWidgets.QApplication) -> None:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(90, 135, 255))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)


def main():
    app = QtWidgets.QApplication(sys.argv)
    _apply_dark_theme(app)
    load_fonts_and_set_global(app)
    w = AppWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
