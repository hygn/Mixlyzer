from PySide6 import QtCore, QtGui, QtWidgets


class SupportDialog(QtWidgets.QDialog):
    """Dialog encouraging donations to Python and key dependencies."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Support Python & Open Source")
        self.setModal(False)
        self.resize(560, 520)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)
        intro = QtWidgets.QLabel(
            "This app is built on the shoulders of open-source projects.\nIf you'd like to support the ecosystem, here are some ways."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        # Quick actions
        quick = QtWidgets.QHBoxLayout()
        btn_psf = QtWidgets.QPushButton("Donate to Python (PSF)")
        btn_psf.clicked.connect(lambda: self._open_url("https://www.python.org/psf/donations/"))
        btn_numfocus = QtWidgets.QPushButton("Donate to Scientific Python (NumFOCUS)")
        btn_numfocus.clicked.connect(lambda: self._open_url("https://numfocus.org/donate"))
        quick.addWidget(btn_psf)
        quick.addWidget(btn_numfocus)
        root.addLayout(quick)

        # Dependencies list with links
        deps = self._curated_dependencies()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(inner)
        form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        for name, url in deps:
            roww = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(roww)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(6)
            lbl = QtWidgets.QLabel(name)
            btn = QtWidgets.QPushButton("Open Link")
            btn.clicked.connect(lambda _=False, u=url: self._open_url(u))
            hl.addWidget(lbl)
            hl.addStretch(1)
            hl.addWidget(btn)
            form.addRow(roww)

        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        # Close button
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        # Map Close button to reject for non-modal behavior
        close_btn = btns.button(QtWidgets.QDialogButtonBox.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.reject)
        root.addWidget(btns)

    def _open_url(self, url: str):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))

    def _curated_dependencies(self):
        candidates = [
            ("PySide6 / Shiboken6", "https://doc.qt.io/qtforpython-6/"),
            ("PyYAML", "https://github.com/yaml/pyyaml"),
            ("librosa", "https://github.com/sponsors/librosa"),
            ("audioread", "https://github.com/beetbox/audioread"),
            ("audioop-lts", "https://pypi.org/project/audioop-lts/"),
            ("decorator", "https://github.com/micheles/decorator"),
            ("joblib", "https://joblib.readthedocs.io/"),
            ("lazy_loader", "https://github.com/scientific-python/lazy_loader"),
            ("packaging", "https://github.com/pypa/packaging"),
            ("msgpack", "https://msgpack.org/"),
            ("numba", "https://github.com/numba/numba"),
            ("llvmlite", "https://github.com/numba/llvmlite"),
            ("numpy", "https://numfocus.org/donate-to-numpy"),
            ("pooch", "https://github.com/fatiando/pooch"),
            ("platformdirs", "https://github.com/platformdirs/platformdirs"),
            ("requests", "https://requests.readthedocs.io/"),
            ("certifi", "https://github.com/certifi/python-certifi"),
            ("charset-normalizer", "https://github.com/Ousret/charset_normalizer"),
            ("idna", "https://github.com/kjd/idna"),
            ("urllib3", "https://github.com/urllib3/urllib3"),
            ("scikit-learn", "https://numfocus.org/donate-to-scikit-learn"),
            ("threadpoolctl", "https://github.com/joblib/threadpoolctl"),
            ("scipy", "https://numfocus.org/donate-to-scipy"),
            ("soundfile", "https://github.com/bastibe/python-soundfile"),
            ("cffi", "https://github.com/python-cffi/cffi/"),
            ("pycparser", "https://github.com/eliben/pycparser"),
            ("soxr", "https://github.com/dofuuz/python-soxr"),
            ("typing_extensions", "https://github.com/python/typing_extensions"),
            ("contourpy", "https://github.com/contourpy/contourpy"),
            ("cycler", "https://github.com/matplotlib/cycler"),
            ("fonttools", "https://github.com/fonttools/fonttools"),
            ("kiwisolver", "https://github.com/nucleic/kiwi"),
            ("pillow", "https://github.com/python-pillow/Pillow"),
            ("pyparsing", "https://github.com/pyparsing/pyparsing"),
            ("python-dateutil", "https://github.com/dateutil/dateutil"),
            ("six", "https://github.com/benjaminp/six"),
            ("matplotlib", "https://numfocus.org/donate-to-matplotlib"),
            ("imageio", "https://github.com/imageio/imageio"),
            ("tifffile", "https://github.com/cgohlke/tifffile"),
            ("resampy", "https://github.com/bmcfee/resampy"),
            ("networkx", "https://networkx.org/"),
            ("scikit-image", "https://github.com/scikit-image/scikit-image"),
            ("pymediainfo", "https://github.com/sbraz/pymediainfo"),
            ("pyqtgraph", "https://pyqtgraph.org/"),
            ("tqdm", "https://github.com/tqdm/tqdm"),
            ("colorama", "https://github.com/tartley/colorama"),
            ("FFmpeg", "https://ffmpeg.org/donations.html"),
            ("SQLite", "https://sqlite.org/consortium.html"),
        ]

        # Optionally, filter to only those importable in the current env.
        filtered = []
        detect = {
            "PySide6 / Shiboken6": "PySide6",
            "PyYAML": "yaml",
            "librosa": "librosa",
            "audioread": "audioread",
            "audioop-lts": "audioop",
            "decorator": "decorator",
            "joblib": "joblib",
            "lazy_loader": "lazy_loader",
            "packaging": "packaging",
            "msgpack": "msgpack",
            "numba": "numba",
            "llvmlite": "llvmlite",
            "numpy": "numpy",
            "pooch": "pooch",
            "platformdirs": "platformdirs",
            "requests": "requests",
            "certifi": "certifi",
            "charset-normalizer": "charset_normalizer",
            "idna": "idna",
            "urllib3": "urllib3",
            "scikit-learn": "sklearn",
            "threadpoolctl": "threadpoolctl",
            "scipy": "scipy",
            "soundfile": "soundfile",
            "cffi": "cffi",
            "pycparser": "pycparser",
            "soxr": "soxr",
            "typing_extensions": "typing_extensions",
            "contourpy": "contourpy",
            "cycler": "cycler",
            "fonttools": "fontTools",
            "kiwisolver": "kiwisolver",
            "pillow": "PIL",
            "pyparsing": "pyparsing",
            "python-dateutil": "dateutil",
            "six": "six",
            "matplotlib": "matplotlib",
            "imageio": "imageio",
            "tifffile": "tifffile",
            "resampy": "resampy",
            "networkx": "networkx",
            "scikit-image": "skimage",
            "pymediainfo": "pymediainfo",
            "pyqtgraph": "pyqtgraph",
            "tqdm": "tqdm",
            "colorama": "colorama",
        }
        for name, url in candidates:
            mod = detect.get(name)
            try:
                __import__(mod)
                filtered.append((name, url))
            except Exception:
                # If import fails, still include as it may be optional at runtime
                filtered.append((name, url))
        # Add detected open-source BLAS backend if present (exclude MKL/Accelerate/Unknown)
        blas = self._detect_blas_backend()
        if blas is not None:
            filtered.append(blas)
        return filtered

    def _detect_blas_backend(self):
        """Detect the BLAS backend via NumPy and return only open-source ones.

        Returns:
            Optional[tuple[str, str]]: (display_name, url) for OpenBLAS/BLIS/ATLAS,
            otherwise None for MKL/Accelerate/Unknown.
        """
        try:
            import numpy as _np  # noqa: F401
            from numpy import __config__ as _npconf

            keys = [
                'blas_opt_info', 'openblas_info', 'blis_info',
                'atlas_3_10_blas_threads_info', 'atlas_blas_info',
                'lapack_opt_info',
            ]
            infos = {}
            for k in keys:
                try:
                    d = _npconf.get_info(k)
                except Exception:
                    d = {}
                if d:
                    infos[k] = d

            libs = set()
            for d in infos.values():
                for lib in (d.get('libraries') or []):
                    try:
                        libs.add(str(lib).lower())
                    except Exception:
                        pass
                for arg in (d.get('extra_link_args') or []):
                    s = str(arg).lower()
                    if 'openblas' in s:
                        libs.add('openblas')
                    if 'atlas' in s:
                        libs.add('atlas')
                    if 'blis' in s:
                        libs.add('blis')

            # Fallback: parse __config__.show output
            try:
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    _npconf.show()
                txt = buf.getvalue().lower()
                if 'openblas' in txt:
                    libs.add('openblas')
                if 'atlas' in txt:
                    libs.add('atlas')
                if 'blis' in txt:
                    libs.add('blis')
            except Exception:
                pass

            # Return only open-source ones
            if 'openblas' in libs or 'openblas_info' in infos:
                return ("OpenBLAS", "http://www.openmathlib.org/OpenBLAS/docs/about/")
            if 'blis' in libs or 'blis_info' in infos:
                return ("BLIS", "https://github.com/flame/blis")
            if 'atlas' in libs or any(k.startswith('atlas') for k in infos):
                return ("ATLAS", "http://math-atlas.sourceforge.net/")

            # Non-OSS or unknown: skip
            return None
        except Exception:
            return None
