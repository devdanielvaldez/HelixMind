"""Convert measured sandbox CPU time into durable session credit charges."""
from __future__ import annotations

import math

DEFAULT_CPU_SECONDS_PER_CREDIT = 10.0


def meter_cpu_time(cpu_time_microseconds: int, seconds_per_credit: float = DEFAULT_CPU_SECONDS_PER_CREDIT) -> dict[str, int | float]:
    """Validate process CPU usage and round its configured credit cost upward."""
    if isinstance(cpu_time_microseconds, bool) or not isinstance(cpu_time_microseconds, int) or cpu_time_microseconds < 0:
        raise ValueError("measured CPU time must be a nonnegative integer number of microseconds")
    if isinstance(seconds_per_credit, bool) or not isinstance(seconds_per_credit, (int, float)) or not math.isfinite(seconds_per_credit) or seconds_per_credit <= 0:
        raise ValueError("CPU seconds per credit must be a positive finite number")
    credits = math.ceil(cpu_time_microseconds / (seconds_per_credit * 1_000_000))
    return {
        "cpu_time_microseconds": cpu_time_microseconds,
        "seconds_per_credit": float(seconds_per_credit),
        "credits": credits,
    }
