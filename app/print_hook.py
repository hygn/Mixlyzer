import contextlib
import io
import logging
import multiprocessing as mp
from logging.handlers import RotatingFileHandler
from pathlib import Path
import platform
import sys

_DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "mixlyzer.log"
_LOGGER_NAME = "mixlyzer_logger"


def _resolve_log_path(log_path: str | None) -> Path:
    try:
        return Path(log_path).expanduser().resolve() if log_path else _DEFAULT_LOG_PATH
    except Exception:
        return _DEFAULT_LOG_PATH


def _setup_logger(log_path: str | None) -> logging.Logger:
    resolved = _resolve_log_path(log_path)
    logger = logging.getLogger(_LOGGER_NAME)
    current_path = getattr(logger, "_log_path", None)
    if logger.handlers and current_path == resolved:
        return logger
    if logger.handlers:
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(resolved, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    logger._log_path = resolved
    return logger


class _StreamToLogger:
    def __init__(self, logger: logging.Logger, level: int, tee_stream) -> None:
        self.logger = logger
        self.level = level
        self.tee_stream = tee_stream

    def write(self, buf: str) -> None:
        if buf.strip():
            self.logger.log(self.level, buf.rstrip())
        try:
            if self.tee_stream:
                self.tee_stream.write(buf)
        except Exception:
            pass

    def flush(self) -> None:
        try:
            if self.tee_stream:
                self.tee_stream.flush()
        except Exception:
            pass


def install_print_hook(log_path: str | None = None) -> None:
    """
    Redirect stdout/stderr to a rotating log file while preserving console output.
    Keep this lightweight so it can run before multiprocessing spawn.
    """
    if getattr(install_print_hook, "_installed", False) and getattr(install_print_hook, "_cfg", None) == log_path:
        return
    install_print_hook._cfg = log_path

    resolved_path = _resolve_log_path(log_path)
    _cleanup_log_once_main_process(resolved_path)
    logger = _setup_logger(resolved_path)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = _StreamToLogger(logger, logging.INFO, orig_stdout)
    sys.stderr = _StreamToLogger(logger, logging.ERROR, orig_stderr)
    install_print_hook._installed = True

    if _is_main_process():
        log_environment_info(log_path=resolved_path)


def _cleanup_log_once_main_process(log_path: Path) -> None:
    """Delete existing log only in the main process on first install."""
    if getattr(_cleanup_log_once_main_process, "_done", False):
        return
    if _is_main_process():
        try:
            log_path.unlink(missing_ok=True)
        except Exception:
            pass
    _cleanup_log_once_main_process._done = True


def _capture_numpy_config() -> str:
    try:
        import numpy as np
    except Exception as exc:
        return f"numpy not available: {exc}"

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            np.__config__.show()
    except Exception as exc:
        return f"numpy config unavailable: {exc}"
    return buf.getvalue().strip()


def log_environment_info(log_path: str | None = None) -> None:
    """Emit Python/platform/NumPy build info once in the main process."""
    if getattr(log_environment_info, "_logged", False):
        return
    if not _is_main_process():
        return

    logger = _setup_logger(log_path)
    try:
        sink = logger.info if logger is not None else print
        sink("=== Mixlyzer environment info ===")
        sink("Python: %s", sys.version.replace("\n", " ")) if logger else sink(f"Python: {sys.version.replace(chr(10), ' ')}")
        sink("Executable: %s", sys.executable) if logger else sink(f"Executable: {sys.executable}")
        sink("Platform: %s", platform.platform()) if logger else sink(f"Platform: {platform.platform()}")
        sink("Processor: %s", platform.processor()) if logger else sink(f"Processor: {platform.processor()}")
        sink("Machine: %s", platform.machine()) if logger else sink(f"Machine: {platform.machine()}")
        np_cfg = _capture_numpy_config()
        if np_cfg:
            for line in np_cfg.splitlines():
                sink("NumPy: %s", line) if logger else sink(f"NumPy: {line}")
    except Exception:
        pass
    log_environment_info._logged = True


def _is_main_process() -> bool:
    try:
        return mp.current_process().name == "MainProcess" and mp.parent_process() is None
    except Exception:
        return False
