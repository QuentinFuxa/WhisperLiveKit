"""Sample process RSS separately from MLX allocator measurements."""

import importlib.util
import threading


class ResourceMonitor:
    interval_s = 0.05

    def __init__(self):
        self.rss_peak_bytes = None
        self.rss_samples = 0
        self.mlx_peak_bytes = None
        self.mlx_active_bytes = None
        self._stop = threading.Event()
        self._thread = None
        self._process = None
        self._mlx = None

    def _sample(self):
        rss = self._process.memory_info().rss
        self.rss_peak_bytes = max(self.rss_peak_bytes or 0, rss)
        self.rss_samples += 1

    def _poll(self):
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self):
        try:
            import psutil
            self._process = psutil.Process()
        except ImportError:
            pass
        if importlib.util.find_spec("mlx"):
            import mlx.core as mx
            if mx.metal.is_available():
                self._mlx = mx
                mx.reset_peak_memory()
        if self._process:
            self._sample()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
            self._sample()
        if self._mlx:
            self.mlx_peak_bytes = self._mlx.get_peak_memory()
            self.mlx_active_bytes = self._mlx.get_active_memory()
