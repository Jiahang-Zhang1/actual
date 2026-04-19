from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import psutil


@dataclass
class SampleSummary:
    cpu_pct_avg: float | None
    cpu_pct_max: float | None
    rss_mb_avg: float | None
    rss_mb_max: float | None
    sample_count: int


class ProcessTreeSampler:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.root = psutil.Process(os.getpid())
        self.samples: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _current_processes(self):
        procs = [self.root]
        try:
            procs.extend(self.root.children(recursive=True))
        except Exception:
            pass
        return procs

    def _prime_cpu(self) -> None:
        for proc in self._current_processes():
            try:
                proc.cpu_percent(interval=None)
            except Exception:
                continue

    def _sample_once(self) -> None:
        total_cpu = 0.0
        total_rss = 0
        for proc in self._current_processes():
            try:
                total_cpu += proc.cpu_percent(interval=None)
                total_rss += proc.memory_info().rss
            except Exception:
                continue

        self.samples.append(
            {
                "ts": time.time(),
                "cpu_pct": total_cpu,
                "rss_mb": total_rss / (1024 * 1024),
            }
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            time.sleep(self.interval_s)

    def start(self) -> None:
        self._prime_cpu()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> SampleSummary:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

        if not self.samples:
            return SampleSummary(None, None, None, None, 0)

        cpu_values = [sample["cpu_pct"] for sample in self.samples]
        rss_values = [sample["rss_mb"] for sample in self.samples]
        return SampleSummary(
            cpu_pct_avg=round(sum(cpu_values) / len(cpu_values), 4),
            cpu_pct_max=round(max(cpu_values), 4),
            rss_mb_avg=round(sum(rss_values) / len(rss_values), 4),
            rss_mb_max=round(max(rss_values), 4),
            sample_count=len(self.samples),
        )
