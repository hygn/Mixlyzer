from PySide6 import QtWidgets, QtGui
from system import i18n
from system.resource_paths import resource_path
from ui import qt_messages

# the bundled CJK fonts' order in the fallback chain, by the UI's language: a kanji / hanja is drawn in the shapes of
# the first font holding it, so the UI's own comes first (English: Japanese, most of an anime library's titles); the
# Chinese ones last, for the characters the others lack (simplified ones above all). The Latin font leads them all.
_CJK_ORDER = {"ko": ("KR", "JP", "TC", "SC"), "ja": ("JP", "KR", "TC", "SC")}
_CJK_DEFAULT = ("JP", "KR", "TC", "SC")


def font_rank(name: str, language: str) -> int:
    """A font file's place in the chain (``name`` its file name, ``language`` the UI's tag: ko-KR, ja-JP, en-US):
    0 the Latin Noto Sans, then the CJK ones in _CJK_ORDER, then any other."""
    stem = name.split("-", 1)[0]
    if stem == "NotoSans":
        return 0
    order = _CJK_ORDER.get(language.split("-", 1)[0], _CJK_DEFAULT)
    script = stem.removeprefix("NotoSans")
    return 1 + order.index(script) if script in order else 1 + len(order)


def load_fonts_and_set_global(app: QtWidgets.QApplication):
    qt_messages.install()     # the font fallback's notes among the notes left out
    # 1) Register bundled fonts (all font files under the given directory), in the chain's order
    font_dir = resource_path("assets/fonts")   # with the app's resources (a PyInstaller build: _internal)
    exts = {".ttf", ".otf", ".ttc", ".otc"}
    families = []
    files = [p for p in font_dir.rglob("*") if p.suffix.lower() in exts]
    for p in sorted(files, key=lambda p: (font_rank(p.name, i18n.language()), p.name)):
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
