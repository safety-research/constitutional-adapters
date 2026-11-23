"""The explicit run list (``RUNS.tsv``) the two explicit-run-list scopes read (``strongreject/scope_runs.py``,
``terra/scope_cpi_runs.py``): its reader, the read-only registry connection, the judge model whose judgments make a listed run a
follow-up, and the path rules that tie a listed store to the store the registry records. Stdlib only, so the StrongREJECT scope still
runs on a cluster node with a bare ``python3``."""
from __future__ import annotations

import os
import sqlite3
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

DEFAULT_REGISTRY = os.path.join(os.path.expanduser("~"), "registry", "registry.sqlite")
#: the judge model whose judgments make a listed run a follow-up (``--allow-follow-up``)
REJUDGE_MODEL = "minimax-m3"

Rules = Sequence[Tuple[str, str]]


class Refused(ValueError):
    pass


def read_tsv(path: str, required: Sequence[str], optional: Sequence[str]) -> Iterator[Tuple[int, Dict[str, Optional[str]]]]:
    """``(row number, row)`` per data row of a tab-separated run list with a header row (blank and ``#`` lines skipped and not counted,
    the header numbered 1; an empty field reads as None). :class:`Refused` for an empty list, a header that lacks a ``required`` column
    or names one that is neither required nor ``optional``, and a row whose field count is not the header's."""
    with open(path, encoding="utf-8") as fh:
        lines = [l.rstrip("\n") for l in fh if l.strip() and not l.startswith("#")]
    if not lines:
        raise Refused(f"{path}: empty")
    hdr = lines[0].split("\t")
    missing = [c for c in required if c not in hdr]
    if missing:
        raise Refused(f"{path}: header lacks {missing} (has {hdr})")
    unknown = sorted(set(hdr) - set(required) - set(optional))
    if unknown:
        raise Refused(f"{path}: unknown columns {unknown}")
    for i, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != len(hdr):
            raise Refused(f"{path}:{i}: {len(parts)} fields, the header has {len(hdr)}")
        yield i, {k: (v if v != "" else None) for k, v in zip(hdr, parts)}


def connect_ro(db: str) -> sqlite3.Connection:
    """A read-only connection to the registry database, rows as :class:`sqlite3.Row`."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=60)
    con.row_factory = sqlite3.Row
    return con


def parse_rules(specs: Sequence[str], flag: str = "--path-rewrite") -> List[Tuple[str, str]]:
    """``FROM=TO`` arguments as prefix rules; :class:`SystemExit` for one without a ``=`` or with an empty side."""
    rules = []
    for s in specs:
        a, sep, b = s.partition("=")
        if not sep or not a or not b:
            raise SystemExit(f"{flag} {s!r}: expected FROM=TO")
        rules.append((a, b))
    return rules


def rewrite(path: str, rules: Rules) -> str:
    """``path`` with the first matching ``FROM`` prefix replaced by its ``TO`` (the paper-row scope's rule): a store read through a
    mirror."""
    for a, b in rules:
        if path.startswith(a):
            return b + path[len(a):]
    return path


def recorded_store(uri: Optional[str], rules: Rules = ()) -> Optional[str]:
    """A ``store_uri`` the registry records (a run's, a judgment's) as the path it is read at here: its ``#fragment`` dropped, ``~``
    expanded, normalized, then ``rules`` applied. Pure string work (nothing is resolved on disk), so it also serves a host that does not
    hold the store. None for an empty uri."""
    if not uri:
        return None
    return rewrite(os.path.normpath(os.path.expanduser(str(uri).split("#", 1)[0])), rules)
