"""Telemetry ingest pipeline."""

import time
from dataclasses import dataclass

RETRY_LIMIT = 3
BACKOFF_BASE = 0.5


@dataclass
class Frame:
    channel: str
    value: float
    timestamp: float


def exponential_backoff(attempt: int, base: float = BACKOFF_BASE) -> float:
    """Return seconds to wait before retry number `attempt`."""
    return base * (2 ** (attempt - 1))


def retry_on_timeout(func, *args, **kwargs):
    """Call func, retrying transient timeouts with exponential backoff."""
    last_error = None
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            return func(*args, **kwargs)
        except TimeoutError as exc:
            last_error = exc
            time.sleep(exponential_backoff(attempt))
    raise last_error


def normalise_channel(name: str) -> str:
    """Lowercase a channel name and replace separators with underscores."""
    cleaned = name.strip().lower()
    for separator in ("-", " ", "/"):
        cleaned = cleaned.replace(separator, "_")
    return cleaned


def shed_low_priority(frames: list, keep_ratio: float = 0.8) -> list:
    """Drop the lowest priority frames when a queue nears its depth limit."""
    ordered = sorted(frames, key=lambda f: getattr(f, "priority", 0), reverse=True)
    keep = int(len(ordered) * keep_ratio)
    return ordered[:keep]
