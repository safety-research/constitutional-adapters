"""Every registry reader names the producing session the same way.

A record's producer block (the emitter, the harvester, the roster adapter), the judging scope's session (which the
judgment id hashes and a claim records) and the worklist's claiming session all read ``emit.SESSION_ENV_VARS`` in order:
the registry's own variable, then the one the session manager hands every session it starts. If one of them read another
variable, a claim and the records filed under it would name the session differently.
"""
from __future__ import annotations

import os
import pathlib
import re

import pytest

from sisteer.registry import emit, harvest, hooks, worklist
from sisteer.registry.adapters import roster

REPO = pathlib.Path(__file__).resolve().parents[1]
#: a direct environment read of a variable whose name ends in SESSION_NAME (os.environ.get / os.environ[...] / os.getenv)
DIRECT_READ = re.compile(r"""(?:environ\.get|environ\[|getenv)\(?\s*["']([A-Z0-9_]*SESSION_NAME)["']""")
#: a module's own copy of the tuple
TUPLE_COPY = re.compile(r"""^SESSION_ENV_VARS\b[^=]*=\s*\(([^)]*)\)""", re.M)

OWN, MANAGER = emit.SESSION_ENV_VARS


def _fallback() -> str:
    return f"{os.environ.get('USER', 'user')}@{os.uname().nodename}"


def _sessions(tmp_path) -> dict:
    scope = hooks.JudgingScope("run-synthetic", "campaign-synthetic", "judge-synthetic", spool=str(tmp_path / "spool"),
                               disabled=True)
    return {
        "emit.producer_block": emit.producer_block("synthetic.py", code_sha="a" * 40)["session"],
        "harvest.harvester_producer": harvest.harvester_producer(str(tmp_path))["session"],
        "roster.producer": roster.producer()["session"],
        "the judging scope's writer": scope.writer.producer["session"],
        "the judging scope's session": scope.session,
        "worklist.default_session": worklist.default_session(),
    }


@pytest.mark.parametrize("own, manager, expected", [
    (None, None, None),
    (None, "synthetic-managed-session", "synthetic-managed-session"),
    ("synthetic-session", None, "synthetic-session"),
    ("synthetic-session", "synthetic-managed-session", "synthetic-session"),
    ("", "synthetic-managed-session", "synthetic-managed-session"),
])
def test_every_reader_names_the_same_session(tmp_path, monkeypatch, own, manager, expected):
    for var, value in ((OWN, own), (MANAGER, manager)):
        if value is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, value)
    got = _sessions(tmp_path)
    # a producer block names no session when none is set; the judging scope and the worklist then name the host's user
    blocks = ("emit.producer_block", "harvest.harvester_producer", "roster.producer", "the judging scope's writer")
    assert got == {k: expected if k in blocks else (expected or _fallback()) for k in got}


def test_the_registrys_own_variable_comes_first():
    assert OWN == "SISTEER_SESSION_NAME" and len(emit.SESSION_ENV_VARS) == 2


def test_no_module_reads_a_session_variable_directly():
    direct, copies = [], []
    for top in ("src", "scripts"):
        for path in sorted((REPO / top).rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            rel = str(path.relative_to(REPO))
            direct += [f"{rel}: {m.group(1)}" for m in DIRECT_READ.finditer(text)]
            copies += [(rel, tuple(re.findall(r"""["']([A-Z0-9_]+)["']""", m.group(1)))) for m in TUPLE_COPY.finditer(text)]
    assert direct == []
    assert copies and all(names == emit.SESSION_ENV_VARS for _rel, names in copies), copies
