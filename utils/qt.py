from contextlib import contextmanager

@contextmanager
def block_signals(obj):
    try:
        obj.blockSignals(True); yield
    finally:
        obj.blockSignals(False)