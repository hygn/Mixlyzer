"""system.os_support.lower_process_priority: the calling process really goes below the others (Windows' below normal
class, elsewhere a nice of 10), checked in a child process so the test run keeps its own priority. On 64-bit Windows
the current process's pseudo-handle (-1) must stay 64 bits wide on its way to SetPriorityClass: cut to an int, the
call failed silently and nothing was lowered."""
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CHILD = """
import sys
sys.path.insert(0, {repo!r})
from system import os_support
os_support.lower_process_priority()
if os_support.IS_WINDOWS:
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32")
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.GetPriorityClass.argtypes = [wintypes.HANDLE]
    k.GetPriorityClass.restype = wintypes.DWORD
    print(k.GetPriorityClass(k.GetCurrentProcess()))
else:
    import os
    print(os.nice(0))
"""


class LowerPriorityTests(unittest.TestCase):
    def test_the_process_goes_below_the_others(self):
        out = subprocess.run([sys.executable, "-c", CHILD.format(repo=str(REPO))], capture_output=True, text=True,
                             timeout=60, check=True).stdout.split()
        from system import os_support
        if os_support.IS_WINDOWS:
            self.assertEqual(int(out[-1]), 0x4000)           # BELOW_NORMAL_PRIORITY_CLASS
        else:
            self.assertGreaterEqual(int(out[-1]), 10)


if __name__ == "__main__":
    unittest.main()
