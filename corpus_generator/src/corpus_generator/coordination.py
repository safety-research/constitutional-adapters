"""The ready-to-post coordination line a generation run prints before it starts.

A site whose shared API pools are coordinated in a chat channel names that channel in ``SISTEER_COORDINATION_CHANNEL``
(the front door's ``config.env``, or the environment). A run that is about to put load on a shared pool then prints one
line to paste there, in the ``<N> <model> <pool> <duration>`` form, and proceeds: posting is never a gate. With the
variable unset or empty no ready-to-post line is printed at all.

This is the twin of the harness's ``sisteer.coordination`` (this package does not depend on the harness);
``tests/test_coordination_channel.py`` holds the two equal.
"""
from __future__ import annotations

import os
from typing import Callable, Mapping, Optional

#: the site variable naming the channel
ENV = "SISTEER_COORDINATION_CHANNEL"
#: the line that closes a printed post
PROCEEDING = "--- (posting is the user's job; proceeding now) ---"


def channel(env: Optional[Mapping[str, str]] = None) -> str:
    """The site's coordination channel, or ``""`` when it names none."""
    return ((os.environ if env is None else env).get(ENV) or "").strip()


def header(note: str = "", env: Optional[Mapping[str, str]] = None) -> str:
    """``--- ready-to-post for <channel>[ (<note>)] ---``, or ``""`` when the site names no channel."""
    ch = channel(env)
    if not ch:
        return ""
    return f"--- ready-to-post for {ch}{f' ({note})' if note else ''} ---"


def post(line: str, echo: Callable[[str], object] = print, *, note: str = "", proceeding: bool = True,
         env: Optional[Mapping[str, str]] = None) -> bool:
    """Print the header, ``line`` and (with ``proceeding``) the closing line through ``echo``, each on its own line.
    Prints nothing when the site names no channel. Returns whether anything was printed."""
    h = header(note, env)
    if not h:
        return False
    echo(h)
    echo(line)
    if proceeding:
        echo(PROCEEDING)
    return True
