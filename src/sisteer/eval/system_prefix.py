"""The authoritative system-prefix composer, in a module that imports NOTHING but the stdlib.

``install_authoritative_system`` was born in :mod:`sisteer.eval.jailbreak`, whose module top imports
``sisteer.generate`` / ``sisteer.methods`` / ``sisteer.models`` and therefore ``torch``. The seedrep ladder's
producer side composes a rung item's re-draw messages with this function INSIDE the Modal app process --
the image's system python, which has no torch (the drivers' venv does) -- so the composer has to be importable
without it (launch F-G). This is a pure move: the function body is byte-for-byte the one ``jailbreak.py``
carried, and ``sisteer.eval.jailbreak.install_authoritative_system`` still names it (re-exported), so every
existing caller and every test that patches it there keeps working.
"""
from __future__ import annotations


def install_authoritative_system(messages: list[dict], system: str) -> list[dict]:
    """Return ``messages`` with ``system`` prepended as the SOLE system message,
    folding any incoming system content into the first user turn.

    HIGH (multi-turn safety): the safety SI is authoritative. An attack-supplied
    system message must NOT be able to REPLACE it — so a carried-in system role
    is demoted into user text (kept, so the model still sees it) and the safety
    SI becomes the one system message the template renders."""
    folded: list[dict] = []
    carried: str | None = None
    for m in messages:
        if m.get("role") == "system":
            carried = f"{carried}\n\n{m['content']}" if carried else m["content"]
        else:
            folded.append(dict(m))
    if carried:
        for m in folded:
            if m.get("role") == "user":
                m["content"] = f"{carried}\n\n{m['content']}"
                break
        else:
            folded.insert(0, {"role": "user", "content": carried})
    return [{"role": "system", "content": system}] + folded
