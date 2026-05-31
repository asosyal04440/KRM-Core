from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class Timer:
    started_at: float

    @classmethod
    def start(cls) -> "Timer":
        return cls(time.perf_counter())

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0
