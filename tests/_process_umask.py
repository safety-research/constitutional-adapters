"""The process umask, read for the suite's umask guard (``tests/conftest.py``) and the tests that pin a caller's umask.

Python has no getter: ``os.umask`` only sets, returning the old value, so the usual read is a set-and-restore round trip
that briefly installs a different mask for every thread in the process. Linux reports the mask in ``/proc/self/status``
(``Umask:``, kernel 4.7+), which reads it without changing it; the round trip is the fallback where that line is absent.
"""

from __future__ import annotations

import os

__all__ = ["current_umask"]


def current_umask() -> int:
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except OSError:
        pass
    mask = os.umask(0o022)
    os.umask(mask)
    return mask
