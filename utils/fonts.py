from pathlib import Path
from PySide6 import QtWidgets, QtGui

def load_fonts_and_set_global(app: QtWidgets.QApplication):
    # 1) Register bundled fonts (all font files under the given directory)
    font_dir = Path("assets/fonts")   # keep original path
    exts = {".ttf", ".otf", ".ttc", ".otc"}
    families = []
    for p in font_dir.rglob("*"):
        if p.suffix.lower() in exts:
            fid = QtGui.QFontDatabase.addApplicationFont(str(p))
            if fid != -1:
                families.extend(QtGui.QFontDatabase.applicationFontFamilies(fid))

    # Deduplicate (preserve order)
    seen = set()
    chain = [f for f in families if not (f in seen or seen.add(f))]

    # 2) Group under an alias
    alias = "Noto Sans Global"
    if chain:
        QtGui.QFont.insertSubstitutions(alias, chain)

    # 3) Set app-wide default font
    font = QtGui.QFont()
    font.setPointSize(11)
    font.setFamilies([alias])   # use alias so chain order becomes the fallback
    font.setWeight(QtGui.QFont.Weight.Normal)
    app.setFont(font)
