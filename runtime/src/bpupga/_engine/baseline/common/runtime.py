from __future__ import annotations

import threading
from types import TracebackType
from typing import Optional, Type

try:
    import psutil
except ImportError:  # pragma: no cover - optional graceful fallback
    psutil = None


class PeakMemoryMonitor:
    """Sample process RSS in a background thread.

    This captures NumPy/SciPy native allocations, unlike ``tracemalloc``.
    The result is process-level peak resident memory, not a delta from startup.
    """

    def __init__(self, interval_sec: float = 0.05) -> None:
        self.interval_sec = interval_sec
        self.peak_rss_bytes: Optional[int] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def peak_rss_mb(self) -> Optional[float]:
        if self.peak_rss_bytes is None:
            return None
        return self.peak_rss_bytes / (1024.0 * 1024.0)

    def _sample(self) -> None:
        if psutil is None:
            return
        process = psutil.Process()
        while not self._stop_event.is_set():
            rss = int(process.memory_info().rss)
            self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss)
            self._stop_event.wait(self.interval_sec)
        rss = int(process.memory_info().rss)
        self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss)

    def __enter__(self) -> "PeakMemoryMonitor":
        if psutil is not None:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 4))

