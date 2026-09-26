"""Process statistics for the performance panel: CPU usage and resident memory (no psutil needed)."""

import os
import sys
import time


def process_rss_bytes():
    """Resident memory of this process in bytes, or None if unavailable on this platform."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.windll.psapi
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            info = Counters()
            info.cb = ctypes.sizeof(info)
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(info), info.cb):
                return int(info.WorkingSetSize)
            return None
        with open("/proc/self/statm") as fh:  # Linux
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError):
        return None


class CpuMeter:
    """CPU use of this process since the previous call, as a percentage of all cores."""

    def __init__(self):
        self._cpu = time.process_time()
        self._wall = time.perf_counter()
        self._cores = os.cpu_count() or 1

    def sample(self):
        cpu, wall = time.process_time(), time.perf_counter()
        d_wall = wall - self._wall
        pct = 100.0 * (cpu - self._cpu) / d_wall / self._cores if d_wall > 0 else 0.0
        self._cpu, self._wall = cpu, wall
        return min(max(pct, 0.0), 100.0)


def format_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
