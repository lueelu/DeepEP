# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026
# Modified by guofanfeng 2026

"""Optional training Timer capture and offline Chrome/Perfetto export."""


def capture_timer(op, *, enabled=True):
    """Capture one forward and its autograd backward; ordinary calls are unchanged."""
    from .capture import TimerCapture

    return TimerCapture(op, enabled=enabled)
