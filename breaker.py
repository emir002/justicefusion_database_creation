# breaker.py
import threading, time
__all__ = ["set_breaker", "breaker_open_until"]

_breaker_lock       = threading.Lock()
_breaker_open_until = 0.0           # epoch-seconds; 0 == closed

def set_breaker(until_ts: float) -> None:
    """Trip the breaker so that *all* callers will refuse work until 'until_ts'."""
    global _breaker_open_until
    with _breaker_lock:
        if until_ts > _breaker_open_until:
            _breaker_open_until = until_ts

def breaker_open_until() -> float:
    """Return the timestamp until which calls should be rejected."""
    with _breaker_lock:
        return _breaker_open_until
