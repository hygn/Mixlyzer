"""librosa's modules the analysis reaches, which librosa imports at their first use: imported on one thread before the
analysis's threads start (two threads importing one package's modules at once can meet in an import deadlock), and by the
resident analysis processes while they wait (core.workers.analysis, which names them without loading the analysis)."""

LAZY_MODULES = (
    "librosa.core.audio", "librosa.core.constantq", "librosa.core.pitch", "librosa.core.spectrum",
    "librosa.feature.spectral", "librosa.feature.utils", "librosa.filters", "librosa.onset", "librosa.segment",
)
