# gpio_alert.py — 3-tier alert system for Project Xceed
# Supports: no_belt (continuous), clipped_behind (intermittent), decoy (single beep)

import threading
import time
from gpiozero import OutputDevice

# ── Hardware ─────────────────────────────────────────
BUZZER_PIN = 17
buzzer = OutputDevice(BUZZER_PIN)

# ── Internal state ───────────────────────────────────
_lock            = threading.Lock()
_alert_thread    = None
_stop_event      = threading.Event()
_current_class   = None   # tracks what is currently alerting

# ── Buzz patterns ────────────────────────────────────
# Each pattern is a list of (on_seconds, off_seconds) tuples.
# The pattern loops until alert_off() or a new alert_set() call.

PATTERNS = {
    # no_belt: continuous — buzzer stays on, no gap
    'no_belt':        [(9999, 0)],

    # clipped_behind: intermittent — 0.3s on, 0.3s off
    'clipped_behind': [(0.3, 0.3)],

    # decoy: single short beep, then 2s silence, repeat
    'decoy':          [(0.1, 2.0)],
}


# ── Internal thread runner ───────────────────────────
def _run_pattern(pattern, stop_event):
    while not stop_event.is_set():
        for on_dur, off_dur in pattern:
            if stop_event.is_set():
                break
            buzzer.on()
            # Use wait with timeout so stop_event is checked frequently
            stop_event.wait(timeout=min(on_dur, 0.05))
            if on_dur > 0.05 and not stop_event.is_set():
                remaining = on_dur - 0.05
                stop_event.wait(timeout=remaining)
            if off_dur > 0:
                buzzer.off()
                stop_event.wait(timeout=off_dur)
    buzzer.off()


def _stop_thread():
    """Stop any running alert thread and silence buzzer."""
    global _alert_thread, _stop_event
    _stop_event.set()
    if _alert_thread and _alert_thread.is_alive():
        _alert_thread.join(timeout=1.0)
    buzzer.off()
    _stop_event = threading.Event()
    _alert_thread = None


# ── Public API ───────────────────────────────────────

def alert_set(class_name: str):
    """
    Activate alert pattern for the given class.
    Call this whenever the detected class changes.
    Silently ignored if the same class is already alerting.
    """
    global _alert_thread, _current_class

    with _lock:
        if class_name == _current_class:
            return   # already running correct pattern

        _stop_thread()

        pattern = PATTERNS.get(class_name)
        if pattern is None:
            # proper_belt or unknown — no alert
            _current_class = None
            return

        _current_class = class_name
        _alert_thread = threading.Thread(
            target=_run_pattern,
            args=(pattern, _stop_event),
            daemon=True
        )
        _alert_thread.start()


def alert_off():
    """Silence all alerts immediately."""
    global _current_class
    with _lock:
        _stop_thread()
        _current_class = None


# Legacy shims — kept so existing detect.py calls still work
def alert_on():
    alert_set('no_belt')


def beep(duration=0.5):
    """Single blocking beep — used for startup test only."""
    buzzer.on()
    time.sleep(duration)
    buzzer.off()


# ── Startup self-test ────────────────────────────────
if __name__ == "__main__":
    print("GPIO self-test — 3 tiers")

    print("  proper_belt: silence (2s)")
    alert_off()
    time.sleep(2)

    print("  decoy: short beep pattern (3s)")
    alert_set('decoy')
    time.sleep(3)

    print("  clipped_behind: intermittent (3s)")
    alert_set('clipped_behind')
    time.sleep(3)

    print("  no_belt: continuous (2s)")
    alert_set('no_belt')
    time.sleep(2)

    alert_off()
    print("Self-test complete.")
