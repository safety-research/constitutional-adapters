#!/usr/bin/env python3
"""The publication scan: every token tier over the publication set, text and binaries alike, plus the attribution tier
in its two scopes. A hit outside the reviewed allowlist (`reproduce/tools/scan_allowlist.tsv`) fails the scan.

The publication set is every file the published repository carries: in a git checkout, what `git ls-files --cached
--others --exclude-standard` lists; in a plain directory (an exported tree), every file under it. Either way the paths
`tests/published_repo.py` names in UNPUBLISHED (the maintainers' `CLAUDE.md`) are left out, and so is `__pycache__`. A
tracked file absent from the working tree (a sparse checkout) is an error, never a silent skip.

Tokens come from ONE table, `TOKEN_TABLE` in tests/front_door_testlib.py, read here by `ast` (never imported or run).
Name-like tokens are stored only as sha256 hashes of their lowercase `[a-z0-9]+` runs joined by one space, so this tool,
the table and the tests hold no word they look for; shapes (a host pattern, a path prefix) stay regexes. A row:

    (tier, "hash", <sha256 hex>, {flags})      n: runs in the token (default 1); head: the sha256 of the first run, which
                                               an n-run row needs; cs: exact case (no lowercasing and no split of a
                                               compound word); sep: the exact separator between runs (or a list of
                                               them), or "loose" (a gap of up to 8 characters that are not letters or
                                               digits); default: a gap of one or two characters, none of them whitespace
    (tier, "shape", <regex>, {flags})          i: case-insensitive; unless: a regex whose overlapping match drops the hit
                                               (the chat role "the user turn" against the attribution shapes); seed: a
                                               fast regex (or a list), one of which every match holds on its line, so
                                               the full regex runs only on those lines (matched against the lowercased
                                               text when `i`); a shape that starts with `^` is matched against each whole
                                               unit (a line, a value). A shape matches within one line, except that the
                                               comment scope also reads each comment passage as one line (below)

Any row may carry `out: True` (the rows every output check enforces, tests/front_door_testlib.assert_no_tokens), and an
attribution row `scopes` (the attribution scopes it applies in; default both).

Scopes:
  tree      every tier but attribution, over every file: text line by line, every path name, and the binaries through
            their decoders (gzip text, the split-gzip SQLite's TEXT cells and schema, Parquet string columns, column names
            and metadata, the pickles inside `.pt` zips and `.pkl` files through `pickletools.genops` (opcodes only;
            nothing is unpickled) with the zips' other small members as printable runs, zstd `.eval` members, safetensors
            headers). Tensor storage members of a `.pt` zip are numbers, not text, and are not read. Any other binary
            is read as its printable runs
  comment   the attribution tier over comment and docstring text: Python COMMENT tokens and module, class and function
            docstrings (in `.py` files and in `.py.txt` copies of Python source); `#` comments in shell, YAML, TOML,
            systemd, env and ini files, Dockerfiles and Makefiles, and in pip requirements and constraints files (by
            pip's own rule); `--` and `/* */` comments outside SQL string literals; Markdown under docs/ and every README;
            notebook markdown cells, code-cell comments and printed output; the comment lines and every row's reason in
            this tool's allowlist. String literals that code reads are never in this scope. A passage is two or more of
            these units on consecutive lines that read as one: comments of one kind, the lines of one docstring or one
            block comment, or lines of one docs page or notebook text, each unit opening its line right after a unit
            that closes its own (a line of code ends a passage, and so does code beside the next comment), up to a blank
            one. Every row of this scope but a `^` shape also reads each passage with the marker every continuation line
            repeats removed (`#`, `--`, a block comment's `*`, a blockquote's `>`) and its lines joined by one space
            (none after a word a hyphen broke); a hit there that a line break splits is reported at its first line, with
            the lines it spans (joined by a newline) as its unit, unless one of those lines holds the same row's hit on
            its own
  leaf      the attribution tier over every free-text leaf of the paper frame (Parquet and SQLite alike; a TEXT cell, or
            each string inside a JSON cell), the registry transcripts and the dimension seeds (every JSON, JSONL and YAML
            file under configs/registry/ and paper/snapshot/, and src/sisteer/registry/decode_policy.json), whether or
            not a policy declares the leaf; a mapping key is a leaf at `<object address>{<key>}`

Allowlist rows (tab-separated): path glob, tier, `row:<token id>` (every hit of that table row) | `line:<sha256 of the
line>` (for a hit a line break splits, of the lines it spans joined by a newline) | `value:<sha256 of the value>`, reason. A data leaf's path is `<snapshot dir or file>#<address>` (address `table.column.key[].key`). Every row
carries a reason; a row that matches nothing on a whole scan is stale and fails it.

The ratchet (`reproduce/tools/scan_ratchet.json`, `--ratchet`): the number of units outside the allowlist per scope,
tier and path, stored under a hash of the three so the file holds no path that names a token. A count may only fall;
`--write-ratchet` records the current counts (and deletes the file when every count is zero), and a recorded count
above the current one fails until it is lowered.

Exit status: 0 no hit outside the allowlist (with --ratchet: none above the ratchet, and the ratchet is current); 1 hits;
2 a usage error, an absent tracked file, or a decoder that cannot run (named; nothing is skipped silently).
"""
from __future__ import annotations

import argparse
import ast
import base64
import bisect
import fnmatch
import gzip
import hashlib
import io
import itertools
import json
import multiprocessing
import os
import pickletools
import re
import sqlite3
import struct
import subprocess
import sys
import tempfile
import tokenize
import warnings
import zipfile
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent.parent
TABLE_FILE = "tests/front_door_testlib.py"
TABLE_NAME = "TOKEN_TABLE"
PUBLISHED_FILE = "tests/published_repo.py"
PUBLISHED_NAME = "UNPUBLISHED"
ALLOWLIST = "reproduce/tools/scan_allowlist.tsv"
RATCHET = "reproduce/tools/scan_ratchet.json"
RATCHET_FORMAT = "scan-ratchet/1"
SKIP_DIRS = ("__pycache__", ".git")
ATTRIBUTION = "attribution"
TIERS = ("personal", "site-label", "site-host", "storage", "network", "region", "org", "vault", "process", "old-name",
         "godmode", ATTRIBUTION)
SCOPES = ("tree", "comment", "leaf")
FLAGS = frozenset({"n", "head", "cs", "sep", "out", "i", "unless", "seed", "scopes"})

# --------------------------------------------------------------------------------------------------- the token table


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()


_RUN = re.compile(r"[A-Za-z0-9]+")


def token_hash(text: str, *, cs: bool = False) -> str:
    """The stored form of a name-like token: its `[A-Za-z0-9]+` runs (lowercased unless `cs`) joined by one space."""
    runs = _RUN.findall(text)
    return sha256_text(" ".join(runs if cs else (r.lower() for r in runs)))


@dataclass(frozen=True)
class Row:
    tier: str
    kind: str          # "hash" | "shape"
    value: str         # the sha256 hex (hash) or the regex (shape)
    n: int = 1
    head: "str | None" = None
    cs: bool = False
    sep: "str | tuple | None" = None
    out: bool = False
    ci: bool = False   # shapes: case-insensitive
    unless: "str | None" = None
    seeds: tuple = ()  # shapes: fast regexes, one of which every match holds on its line
    scopes: tuple = ("comment", "leaf")

    @property
    def token_id(self) -> str:
        return self.value if self.kind == "hash" else sha256_text("shape:" + self.value)

    @property
    def anchored(self) -> bool:
        return self.kind == "shape" and self.value.startswith("^")


def _bad(i: int, msg: str) -> ValueError:
    return ValueError(f"{TABLE_NAME}[{i}]: {msg}")


def parse_rows(table) -> list:
    rows = []
    for i, ent in enumerate(table):
        if not (isinstance(ent, tuple) and len(ent) == 4 and isinstance(ent[3], dict)):
            raise ValueError(f"{TABLE_NAME}[{i}] is not (tier, kind, value, {{flags}}): {ent!r}")
        tier, kind, value, flags = ent
        if tier not in TIERS:
            raise _bad(i, f"unknown tier {tier!r} (one of {', '.join(TIERS)})")
        unknown = set(flags) - FLAGS
        if unknown:
            raise _bad(i, f"unknown flag(s) {sorted(unknown)}")
        scopes = tuple(flags.get("scopes", ("comment", "leaf")))
        if "scopes" in flags and tier != ATTRIBUTION:
            raise _bad(i, "only attribution rows take scopes")
        if not scopes or not set(scopes) <= {"comment", "leaf"}:
            raise _bad(i, f"scopes are comment and leaf, not {scopes!r}")
        if kind == "hash":
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise _bad(i, f"a hash row holds a sha256 hex digest, not {value!r}")
            n = int(flags.get("n", 1))
            head = flags.get("head")
            if n < 1 or (n > 1) != (head is not None) or (head is not None and not re.fullmatch(r"[0-9a-f]{64}", head)):
                raise _bad(i, f"an n-run row (n > 1) carries the sha256 of its first run as `head`, and a one-run row none "
                              f"(n={n}, head={head!r})")
            sep = flags.get("sep")
            if isinstance(sep, list):
                sep = tuple(sep)
            seps = sep if isinstance(sep, tuple) else (sep,)
            if sep is not None and sep != "loose" and not (seps and all(
                    isinstance(x, str) and x and len(x) <= _LOOSE and x != "loose" and not _RUN.search(x) for x in seps)):
                raise _bad(i, f"sep is 'loose', or a short separator (or a list of them) without letters or digits, not {sep!r}")
            for bad in ("i", "unless", "seed"):
                if bad in flags:
                    raise _bad(i, f"`{bad}` is a shape flag")
            rows.append(Row(tier, kind, value, n=n, head=head, cs=bool(flags.get("cs")), sep=sep, out=bool(flags.get("out")),
                            scopes=scopes))
        elif kind == "shape":
            seeds = flags.get("seed", ())
            seeds = (seeds,) if isinstance(seeds, str) else tuple(seeds)
            try:
                rx = re.compile(value)
                if flags.get("unless"):
                    re.compile(flags["unless"])
                for s in seeds:
                    re.compile(s)
            except (re.error, TypeError) as exc:
                raise _bad(i, f"not a regex ({exc})") from exc
            if rx.match(""):
                raise _bad(i, "a shape that matches the empty string matches everywhere")
            for bad in ("n", "head", "cs", "sep"):
                if bad in flags:
                    raise _bad(i, f"`{bad}` is a hash-row flag")
            if not all(isinstance(s, str) and s for s in seeds) or any(re.compile(s).match("") for s in seeds):
                raise _bad(i, f"a seed is a regex that matches something, not {seeds!r}")
            if flags.get("i") and any(re.search(r"(?<!\\)[A-Z]", s) for s in seeds):
                raise _bad(i, "a case-insensitive row's seeds are matched against the lowercased text: no capital letter")
            if value.startswith("^") and seeds:
                raise _bad(i, "a `^` shape is matched against each whole unit and takes no seed")
            rows.append(Row(tier, kind, value, out=bool(flags.get("out")), ci=bool(flags.get("i")), unless=flags.get("unless"),
                            seeds=seeds, scopes=scopes))
        else:
            raise _bad(i, f"kind is 'hash' or 'shape', not {kind!r}")
    ids = [r.token_id for r in rows]
    dup = {x for x in ids if ids.count(x) > 1}
    if dup:
        raise ValueError(f"{TABLE_NAME}: duplicate row(s) {sorted(dup)}")
    return rows


def _literal(path: Path, name: str):
    """The literal assigned to `name` at module level in `path`, by `ast.literal_eval`: the file is never imported."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.value is not None:
            return ast.literal_eval(node.value)
    raise ValueError(f"{path} has no literal {name} assignment")


def load_table(path: Path) -> tuple:
    """`TOKEN_TABLE` from the token source."""
    return _literal(path, TABLE_NAME)


def load_unpublished(root: Path) -> tuple:
    """The repository paths the publication leaves out (`UNPUBLISHED` in tests/published_repo.py)."""
    got = _literal(Path(root) / PUBLISHED_FILE, PUBLISHED_NAME)
    if not (isinstance(got, tuple) and all(isinstance(p, str) and p for p in got)):
        raise ValueError(f"{PUBLISHED_FILE}: {PUBLISHED_NAME} is a tuple of repository paths, not {got!r}")
    return got


# --------------------------------------------------------------------------------------------------- matching
# The matcher works on UTF-8 bytes: a text's runs come from one `translate` and `split`, the runs that start a row are
# found by `bytes.find`, and a shape's full regex runs only on the lines that hold one of its seeds. Offsets are mapped
# back to characters for the units a hit reports. A shape matches within one line (or one line of a value); the comment
# scope also hands its rows each comment passage joined onto one line.

#: the parts of a compound word: `CamelCase` -> Camel, Case; `item42` -> item, 42 (an exact-case row never splits)
_PIECE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_PIECE_MAX = 40            # a longer run is data (a hash, base64), not a compound word: split nothing
_LOOSE = 8
_MEMO_MAX = 1_000_000
_ALNUM = frozenset(b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
_TO_RUNS = bytes(c if c in _ALNUM else 32 for c in range(256))
_RUNB = re.compile(rb"[A-Za-z0-9]+")
_WS = frozenset(b" \t\n\r\x0b\x0c")
#: what batched values are joined with: no letter or digit (no run continues across it), longer than any gap a token may
#: span, and a newline (no shape crosses a line)
SEP = b"\x00\x00\x00\x00\n\x00\x00\x00\x00\x00\x00"
_CHUNK = 1 << 23           # bytes per batched scan
_MISS = object()


def _gap_ok(gap: bytes, sep) -> bool:
    if sep is None:
        return 0 < len(gap) <= 2 and not any(c in _WS for c in gap)
    if sep == "loose":
        return 0 < len(gap) <= _LOOSE
    g = gap.decode("utf-8", "replace")
    return g in sep if isinstance(sep, tuple) else g == sep


def _gap_limit(sep) -> int:
    if sep is None:
        return 2
    if sep == "loose":
        return _LOOSE
    return max(len(x.encode()) for x in sep) if isinstance(sep, tuple) else len(sep.encode())


def _overlaps(spans: list, starts: list, s: int, e: int) -> bool:
    """Whether [s, e) overlaps one of `spans` (sorted, non-overlapping, `starts` their starts)."""
    i = bisect.bisect_left(starts, e)
    return i > 0 and spans[i - 1][1] > s


def _enc(text) -> bytes:
    return text if isinstance(text, bytes) else text.encode("utf-8", "surrogatepass")


def _chars(b: bytes) -> int:
    """The number of characters `b` decodes to (a byte offset as a character offset)."""
    return len(b.decode("utf-8", "replace"))


class Matcher:
    """Finds the rows of a token table in a text. `find(text)` returns every `(row, start, end)` of one unit; `lines(text)`
    every hit of a text as `(line number, line, row, start, end)`; `batch(values)` scans many values at once and returns
    `(index, row, start, end)`; `search(text)` returns the first hit or None (the regex-like face the output checks use).
    Offsets are characters of the unit."""

    def __init__(self, rows):
        self.rows = tuple(rows)
        self._uni_ci: dict = {}
        self._uni_cs: dict = {}
        self._head_of: dict = {}
        free, anch = [], []
        for r in self.rows:
            if r.kind == "hash" and r.n == 1:
                (self._uni_cs if r.cs else self._uni_ci).setdefault(r.value, []).append(r)
            elif r.kind == "hash":
                self._head_of.setdefault((r.cs, r.head), []).append(r)
            else:
                fl = re.IGNORECASE if r.ci else 0
                un = re.compile(r.unless.encode()) if r.unless else None
                rx = re.compile(r.value.encode("utf-8"), fl)
                if r.anchored:
                    anch.append((r, rx, un))
                else:
                    free.append((r, rx, tuple(re.compile(s.encode("utf-8")) for s in r.seeds), un))
        self._free = free
        self._anch = anch
        self._cs_heads = any(cs for cs, _ in self._head_of)
        self._has_hash = bool(self._uni_ci or self._uni_cs or self._head_of)
        self._memo: dict = {}
        self._hit_runs: set = set()

    # ---- runs
    def _info(self, run: bytes):
        got = self._memo.get(run, _MISS)
        if got is not _MISS:
            return got
        low = run.lower()
        hl = hashlib.sha256(low).hexdigest()
        uni = list(self._uni_ci.get(hl, ()))
        if self._uni_cs:
            uni += self._uni_cs.get(hashlib.sha256(run).hexdigest(), ())
        compound = (run != low and not run.isupper()) or (any(48 <= c <= 57 for c in run) and not run.isdigit())
        if self._uni_ci and len(run) <= _PIECE_MAX and compound:
            pieces = _PIECE.findall(run.decode("ascii"))
            if len(pieces) > 1:
                for a in range(len(pieces)):
                    for b in range(a + 1, len(pieces) + 1):
                        if b - a < len(pieces):
                            uni += self._uni_ci.get(sha256_text("".join(pieces[a:b]).lower()), ())
        heads = list(self._head_of.get((False, hl), ()))
        if self._cs_heads:
            heads += self._head_of.get((True, hashlib.sha256(run).hexdigest()), ())
        got = (tuple(dict.fromkeys(uni)), tuple(heads)) if (uni or heads) else None
        self._memo[run] = got
        if got is not None:
            self._hit_runs.add(run)
        return got

    def _live_runs(self, data: bytes) -> set:
        """The runs of `data` that start a hash row (a one-run token, or an n-run row's head)."""
        if len(self._memo) > _MEMO_MAX:
            self._memo.clear()
            self._hit_runs.clear()
        runs = set(data.translate(_TO_RUNS).split())
        for r in runs.difference(self._memo):
            self._info(r)
        return runs & self._hit_runs

    def _tail(self, data: bytes, pos: int, row: Row, head: bytes):
        """The end of `row` when its first run `head` ends at `pos`, else None."""
        parts = [head]
        limit = _gap_limit(row.sep)
        for _ in range(row.n - 1):
            m = _RUNB.search(data, pos, pos + limit + 1)
            if m is None:
                return None
            m = _RUNB.match(data, m.start())
            if not _gap_ok(data[pos:m.start()], row.sep):
                return None
            parts.append(m.group())
            pos = m.end()
        joined = b" ".join(parts if row.cs else [p.lower() for p in parts])
        return pos if hashlib.sha256(joined).hexdigest() == row.value else None

    @staticmethod
    def _shape(data: bytes, r: Row, rx, unless, ls: int, le: int) -> list:
        out, spans, starts = [], None, None
        for m in rx.finditer(data, ls, le):
            s, e = m.span()
            if s == e:
                continue
            if unless is not None:
                if spans is None:
                    spans = [u.span() for u in unless.finditer(data, ls, le)]
                    starts = [a for a, _ in spans]
                if _overlaps(spans, starts, s, e):
                    continue
            out.append((r, s, e))
        return out

    def _find(self, data: bytes, *, anchored: bool) -> list:
        """Every `(row, start, end)` in `data`, byte offsets; `anchored`: the `^` shapes too, line by line."""
        out = []
        if self._has_hash:
            n = len(data)
            for run in self._live_runs(data):
                uni, heads = self._memo[run]
                k = len(run)
                i = data.find(run)
                while i >= 0:
                    j = i + k
                    if (i == 0 or data[i - 1] not in _ALNUM) and (j == n or data[j] not in _ALNUM):
                        out.extend((r, i, j) for r in uni)
                        for r in heads:
                            end = self._tail(data, j, r, run)
                            if end is not None:
                                out.append((r, i, end))
                    i = data.find(run, j)
        low = None
        for r, rx, seeds, unless in self._free:
            if not seeds:
                for ls, le in _line_spans(data, rx):
                    out.extend(self._shape(data, r, rx, unless, ls, le))
                continue
            if r.ci and low is None:
                low = data.lower()
            hay = low if r.ci else data
            done = set()
            for sd in seeds:
                for m in sd.finditer(hay):
                    q = m.start()
                    ls = data.rfind(b"\n", 0, q) + 1
                    if ls in done:
                        continue
                    done.add(ls)
                    le = data.find(b"\n", q)
                    out.extend(self._shape(data, r, rx, unless, ls, len(data) if le < 0 else le))
        if anchored:
            for r, rx, unless in self._anch:
                pos = 0
                while pos <= len(data):
                    le = data.find(b"\n", pos)
                    le = len(data) if le < 0 else le
                    line_end = le - 1 if le > pos and data[le - 1:le] == b"\r" else le
                    out.extend(self._shape(data, r, rx, unless, pos, line_end))
                    pos = le + 1
        return out

    # ---- the faces
    def maybe(self, text) -> bool:
        """Whether `text` holds a hit (a filter for the units a scope must first carve out of a file)."""
        return bool(self._find(_enc(text), anchored=True))

    def seeded(self, text) -> bool:
        """Whether `text` holds the first run of a hash row or a seed of a shape (always, for a shape without seeds): the
        filter before a scope joins units a line break may have split a hit across."""
        if not self.rows:
            return False
        if self._anch or any(not seeds for _r, _rx, seeds, _un in self._free):
            return True
        data, low = _enc(text), None
        if self._has_hash and self._live_runs(data):
            return True
        for r, _rx, seeds, _un in self._free:
            if r.ci and low is None:
                low = data.lower()
            if any(sd.search(low if r.ci else data) for sd in seeds):
                return True
        return False

    def lines(self, text, base: int = 1) -> list:
        """Every hit as `(line number, line, row, start, end)`; offsets are characters within the line, where a hit that
        runs on is cut."""
        data = _enc(text)
        hits = self._find(data, anchored=True)
        if not hits:
            return []
        hits.sort(key=lambda h: (h[1], h[2]))
        out, pos, n = [], 0, base
        cache: dict = {}
        for r, s, e in hits:
            n += data.count(b"\n", pos, s)
            pos = s
            ls = data.rfind(b"\n", 0, s) + 1
            if ls not in cache:
                le = data.find(b"\n", s)
                lb = data[ls:len(data) if le < 0 else le]
                if lb.endswith(b"\r"):
                    lb = lb[:-1]
                cache = {ls: (lb, lb.decode("utf-8", "replace"))}
            lb, unit = cache[ls]
            cs = _chars(lb[:s - ls])
            ce = _chars(lb[:min(e, ls + len(lb)) - ls])
            out.append((n, unit, r, cs, max(ce, cs)))
        return out

    def batch(self, values) -> list:
        """Every hit in `values` (a sequence of strings) as `(index, row, start, end)`: the values are scanned joined by
        SEP, in chunks; a `^` shape is matched against each value on its own."""
        values = values if isinstance(values, (list, tuple)) else list(values)
        out = []
        i0 = 0
        while i0 < len(values):
            enc, size, i1 = [], 0, i0
            while i1 < len(values) and (i1 == i0 or size <= _CHUNK):
                b = _enc(values[i1])
                enc.append(b)
                size += len(b) + len(SEP)
                i1 += 1
            data = SEP.join(enc)
            hits = self._find(data, anchored=False)
            if hits:
                starts = list(itertools.accumulate((len(b) + len(SEP) for b in enc[:-1]), initial=0))
                for r, s, e in hits:
                    k = bisect.bisect_right(starts, s) - 1
                    off, b = s - starts[k], enc[k]
                    if off < len(b):
                        cs = _chars(b[:off])
                        out.append((i0 + k, r, cs, max(cs, _chars(b[:min(e - starts[k], len(b))]))))
            for r, rx, unless in self._anch:
                for k, b in enumerate(enc):
                    for r2, s, e in self._shape(b, r, rx, unless, 0, len(b)):
                        cs = _chars(b[:s])
                        out.append((i0 + k, r2, cs, max(cs, _chars(b[:e]))))
            i0 = i1
        return out

    def find(self, text) -> list:
        """Every hit of one unit as `(row, start, end)` (character offsets); a `^` shape is matched against the whole unit."""
        return [(r, s, e) for _k, r, s, e in self.batch([text.decode("utf-8", "replace") if isinstance(text, bytes) else text])]

    def scan(self, text) -> list:
        return self.find(text)

    def search(self, text):
        hits = self.find(text)
        return min(hits, key=lambda h: (h[1], h[2])) if hits else None


def _line_spans(data: bytes, rx):
    """The lines of `data` an unseeded shape can match on: every line the regex matches somewhere (one pass of the full
    regex over `data`, then each such line whole)."""
    seen = set()
    for m in rx.finditer(data):
        ls = data.rfind(b"\n", 0, m.start()) + 1
        if ls in seen:
            continue
        seen.add(ls)
        le = data.find(b"\n", m.start())
        yield ls, len(data) if le < 0 else le


def hash_row(tier: str, text: str, **flags) -> tuple:
    """The table row for a plain token, as the table's maintainers make it (outside the repository, where the plain words
    live): `(tier, "hash", sha256 of the runs, {n, head, ...})`. Tests use it to plant invented tokens."""
    cs = bool(flags.get("cs"))
    runs = _RUN.findall(text)
    if not runs:
        raise ValueError(f"{text!r} has no [A-Za-z0-9] run")
    f = dict(flags)
    if len(runs) > 1:
        f["n"] = len(runs)
        f["head"] = sha256_text(runs[0] if cs else runs[0].lower())
    return (tier, "hash", token_hash(text, cs=cs), f)


def build_matchers(table) -> dict:
    """{'tree', 'comment', 'wrap', 'leaf', 'out'}: a Matcher per scope (the tree takes every tier but attribution; the two
    attribution scopes take the attribution rows declared for them, and `wrap` the comment rows that also read a comment
    passage across its line breaks, every one but a `^` shape; `out` the rows output checks enforce), plus `rows`."""
    rows = parse_rows(tuple(table))
    return {"tree": Matcher([r for r in rows if r.tier != ATTRIBUTION]),
            "comment": Matcher([r for r in rows if r.tier == ATTRIBUTION and "comment" in r.scopes]),
            "wrap": Matcher([r for r in rows if r.tier == ATTRIBUTION and "comment" in r.scopes and not r.anchored]),
            "leaf": Matcher([r for r in rows if r.tier == ATTRIBUTION and "leaf" in r.scopes]),
            "out": Matcher([r for r in rows if r.out]),
            "rows": rows}


# --------------------------------------------------------------------------------------------------- hits and the allowlist


_UNIT_KEEP = 2000          # a longer unit (a frame cell) is kept as a window around the hit; its hash is of the whole


@dataclass(frozen=True)
class Hit:
    path: str          # repo path; a frame hit names its snapshot directory, so Parquet and SQLite share one address
    where: str         # "L12" (a text line), "path" (the file's name), a leaf address "table.column.key[]", "member:op12" …
    tier: str
    token: str         # the row's token id
    unit: str          # the scanned unit (a line, a value, a pickle string), or a window of a long one
    start: int         # the hit within `unit`
    end: int
    scope: str
    origin: str = ""   # the file a frame value was read from
    unit_hash: str = ""            # sha256 of the whole unit
    cut: tuple = (False, False)    # `unit` is a window: text was cut before / after it

    @classmethod
    def make(cls, path, where, tier, token, unit, start, end, scope, origin=""):
        h = sha256_text(unit)
        if len(unit) <= _UNIT_KEEP:
            return cls(path, where, tier, token, unit, start, end, scope, origin, h)
        a = max(0, start - _UNIT_KEEP // 4)
        b = min(len(unit), a + _UNIT_KEEP)
        return cls(path, where, tier, token, unit[a:b], start - a, min(end, b) - a, scope, origin, h, (a > 0, b < len(unit)))

    def excerpt(self, width: int = 140) -> str:
        a = max(0, self.start - width // 3)
        s = self.unit[a:a + width].replace("\n", "\\n")
        return ("…" if a or self.cut[0] else "") + s + ("…" if a + width < len(self.unit) or self.cut[1] else "")

    def key(self) -> tuple:
        """One unit counts once per scope and tier, however many rows or representations hit it."""
        return (self.scope, self.tier, self.path, self.where, self.unit_hash)


@dataclass
class AllowRow:
    glob: str
    tier: str
    kind: str          # row | line | value
    digest: str
    reason: str
    lineno: int
    used: int = 0

    def matches(self, h: Hit) -> bool:
        if h.tier != self.tier:
            return False
        if "#" in self.glob:
            # a leaf address spells each list level `[]`, which a glob reads literally (`[[]]`), however deep the nesting
            if not fnmatch.fnmatchcase(f"{h.path}#{h.where}", self.glob.replace("[]", "[[]]")):
                return False
        elif not fnmatch.fnmatchcase(h.path, self.glob):
            return False
        if self.kind == "row":
            return h.token == self.digest
        return h.unit_hash == self.digest


def load_allowlist(path: Path) -> list:
    rows = []
    for i, ln in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) != 4:
            raise ValueError(f"{path}:{i}: an allowlist row is 4 tab-separated fields (path glob, tier, match, reason)")
        glob, tier, match, reason = (p.strip() for p in parts)
        if tier not in TIERS:
            raise ValueError(f"{path}:{i}: unknown tier {tier!r}")
        m = re.fullmatch(r"(row|line|value):([0-9a-f]{64})", match)
        if not m:
            raise ValueError(f"{path}:{i}: the match is row:<token id>, line:<sha256> or value:<sha256>, not {match!r}")
        if not reason:
            raise ValueError(f"{path}:{i}: every row carries its reason")
        if m.group(1) == "value" and "#" not in glob:
            raise ValueError(f"{path}:{i}: a value: row names a data unit (<path>#<address>)")
        rows.append(AllowRow(glob, tier, m.group(1), m.group(2), reason, i))
    return rows


def apply_allowlist(hits: list, allow: list) -> list:
    """The hits no allowlist row covers; each row's `used` counts what it covered."""
    left = []
    for h in hits:
        for row in allow:
            if row.matches(h):
                row.used += 1
                break
        else:
            left.append(h)
    return left


def tier_counts(hits: list) -> dict:
    """Units outside the allowlist per tier (a unit hit in two scopes counts in each)."""
    out = {t: 0 for t in TIERS}
    for _scope, tier, _p, _w, _u in {h.key() for h in hits}:
        out[tier] += 1
    return out


# --------------------------------------------------------------------------------------------------- the ratchet


def _site_key(scope: str, tier: str, path: str) -> str:
    return sha256_text(f"{scope}\t{tier}\t{path}")[:24]


def ratchet_of(hits: list) -> dict:
    """The ratchet document for `hits` (the units outside the allowlist)."""
    per = Counter((s, t, p) for s, t, p, _w, _u in {h.key() for h in hits})
    totals = Counter()
    for (s, t, _p), n in per.items():
        totals[f"{s}:{t}"] += n
    return {"format": RATCHET_FORMAT,
            "about": "units outside the allowlist per scope, tier and path (keyed by a hash of the three); a count may only "
                     "fall. Rewrite with reproduce/tools/scrub_scan.py --write-ratchet; the file goes when every count is zero.",
            "totals": dict(sorted(totals.items())),
            "sites": dict(sorted((_site_key(s, t, p), n) for (s, t, p), n in per.items()))}


def load_ratchet(path: Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("format") != RATCHET_FORMAT or not isinstance(doc.get("sites"), dict):
        raise ValueError(f"{path}: not a {RATCHET_FORMAT} document")
    return doc


def check_ratchet(hits: list, doc: "dict | None") -> tuple:
    """`(rose, loose)`: `rose` the (scope, tier, path, now, recorded) whose count is above the ratchet; `loose` the number of
    recorded sites now below their count (the ratchet must be lowered)."""
    per = Counter((s, t, p) for s, t, p, _w, _u in {h.key() for h in hits})
    sites = dict((doc or {}).get("sites", {}))
    now = {}
    rose = []
    for (s, t, p), n in sorted(per.items()):
        k = _site_key(s, t, p)
        now[k] = n
        rec = int(sites.get(k, 0))
        if n > rec:
            rose.append((s, t, p, n, rec))
    loose = sum(1 for k, rec in sites.items() if now.get(k, 0) < int(rec))
    return rose, loose


def write_ratchet(path: Path, hits: list) -> "dict | None":
    doc = ratchet_of(hits)
    if not doc["sites"]:
        if Path(path).exists():
            Path(path).unlink()
        return None
    Path(path).write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return doc


# --------------------------------------------------------------------------------------------------- the publication set


def git_files(root: Path) -> list:
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root, capture_output=True,
                         check=True).stdout.decode("utf-8", "surrogateescape")
    return sorted({p for p in out.split("\0") if p})


def walk_files(root: Path) -> list:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        keep = []
        for d in dirnames:
            full = os.path.join(dirpath, d)
            if d in SKIP_DIRS:
                continue
            if os.path.islink(full):
                out.append(os.path.normpath(os.path.join(rel, d)))
            else:
                keep.append(d)
        dirnames[:] = sorted(keep)
        out.extend(os.path.normpath(os.path.join(rel, f)) for f in filenames)
    return sorted(p.replace(os.sep, "/") for p in out)


def is_checkout(root: Path) -> bool:
    return (Path(root) / ".git").exists()


def publication_set(root: Path, unpublished=None) -> list:
    """Every file the published repository carries, minus UNPUBLISHED and `__pycache__`."""
    root = Path(root)
    unpublished = tuple(load_unpublished(root) if unpublished is None else unpublished)
    files = git_files(root) if is_checkout(root) else walk_files(root)
    return [p for p in files if p not in unpublished and not any(part in SKIP_DIRS for part in p.split("/"))]


def select(files: list, roots: list) -> list:
    if not roots:
        return files
    norm = [r.rstrip("/") for r in roots]
    return [f for f in files if any(f == r or f.startswith(r + "/") or fnmatch.fnmatchcase(f, r) for r in norm)]


# --------------------------------------------------------------------------------------------------- decoders

SPLIT_SQLITE = re.compile(r"^(?P<base>.+\.sqlite)\.gz\.(?P<part>\d{2})$")
ZIP_PICKLES = (".pt", ".pth", ".ckpt")
COMMENT_HASH_SUFFIXES = (".sh", ".bash", ".sbatch", ".yaml", ".yml", ".toml", ".in", ".service", ".timer", ".env", ".cfg",
                         ".conf", ".ini")
COMMENT_HASH_NAMES = ("Makefile", "Dockerfile", ".gitignore", ".gitattributes")
#: Python source, and a committed copy of Python source kept as text (an evidence copy, `<name>.py.txt`)
PYTHON_SUFFIXES = (".py", ".py.txt")
#: pip requirements and constraints files (`requirements.txt`, `<name>-requirements.txt`, `constraints.txt`, and the `.in`
#: inputs a lock is compiled from), whose comments follow pip's own rule
REQUIREMENTS_SUFFIXES = (".txt", ".in")
REQUIREMENTS_STEMS = ("requirements", "constraints")
YAML_SUFFIXES = (".yaml", ".yml")
LEAF_PREFIXES = ("configs/registry/", "paper/snapshot/")
LEAF_FILES = ("src/sisteer/registry/decode_policy.json",)
FRAME_PREFIX = "paper/snapshot/"


class DecodeError(RuntimeError):
    pass


_PRINTABLE = re.compile(rb"[\x20-\x7e\t]{4,}")


def _printable_runs(data: bytes) -> list:
    return [m.group().decode("ascii") for m in _PRINTABLE.finditer(data)]


def _pickle_strings(data: bytes, where: str) -> list:
    """Every string argument of a pickle's opcodes, by `pickletools.genops`: the stream is disassembled, never loaded."""
    out = []
    try:
        for i, (_op, arg, _pos) in enumerate(pickletools.genops(io.BytesIO(data))):
            if isinstance(arg, str):
                out.append((i, arg))
            elif isinstance(arg, (bytes, bytearray)) and arg:
                out.append((i, bytes(arg).decode("utf-8", "replace")))
    except Exception as exc:  # noqa: BLE001  (genops raises ValueError and friends on a stream it cannot walk)
        raise DecodeError(f"{where}: not a pickle stream pickletools can walk ({exc})") from exc
    return out


def _zip_members(path: Path):
    """(name, bytes) of every member of a zip, zstd members (method 93, which this Python's zipfile cannot read) included."""
    with zipfile.ZipFile(path) as zf, open(path, "rb") as fh:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if info.compress_type == 93:
                fh.seek(info.header_offset)
                hdr = fh.read(30)
                if hdr[:4] != b"PK\x03\x04":
                    raise DecodeError(f"{path}: {info.filename}: no local header")
                nlen, xlen = struct.unpack("<HH", hdr[26:30])
                fh.seek(info.header_offset + 30 + nlen + xlen)
                raw = fh.read(info.compress_size)
                try:
                    import zstandard  # noqa: PLC0415  (a locked dependency; only the .eval decoder needs it)
                except ImportError as exc:
                    raise DecodeError(f"{path}: zstd member {info.filename} and no `zstandard` module: {exc}") from exc
                data = zstandard.ZstdDecompressor().decompress(raw, max_output_size=max(info.file_size, 1))
            else:
                data = zf.read(info)
            yield info.filename, data


def _is_tensor_storage(member: str) -> bool:
    return re.search(r"(^|/)data/\d+$", member) is not None


def _json_leaves(v, path: str = ""):
    """(address, string) for every string value of a parsed JSON/YAML document, and for every string mapping key at
    `<object address>{<key>}`: a key is text a reader sees as well."""
    if isinstance(v, dict):
        for k, x in v.items():
            if isinstance(k, str):
                yield f"{path}{{{k}}}", k
            yield from _json_leaves(x, f"{path}.{k}" if path else str(k))
    elif isinstance(v, list):
        for x in v:
            yield from _json_leaves(x, path + "[]")
    elif isinstance(v, str):
        yield path, v


def _value_leaves(prefix: str, value: str):
    """A TEXT cell's leaves: each string of a JSON object or array cell, else the cell itself."""
    t = value.lstrip()
    if t[:1] in ("{", "[") and len(t) > 1:
        try:
            doc = json.loads(value)
        except ValueError:
            doc = None
        if isinstance(doc, (dict, list)):
            for addr, leaf in _json_leaves(doc):
                yield (f"{prefix}.{addr}" if addr and not addr.startswith(("[", "{")) else prefix + addr), leaf
            return
    yield prefix, value


# --------------------------------------------------------------------------------------------------- comment units
# Each producer yields (line number, column, text): a comment or docstring piece on one physical line. With `tagged`, a
# fourth field names the passage the piece can belong to (see _passages): a comment kind, or one docstring or block comment.


def _docstring_starts(tree) -> set:
    """(line, column) of every module, class and function docstring: a walk over statement bodies only (expressions hold
    no definition that can carry one)."""
    out, stack = set(), [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                out.add((first.lineno, first.col_offset))
        for attr in ("body", "orelse", "finalbody", "handlers", "cases"):
            stack.extend(c for c in (getattr(node, attr, None) or ()) if isinstance(c, ast.AST))
    return out


def _tokens(text: str) -> list:
    """(type, string, start) of every token of a Python source: the C tokenizer driven directly where this Python has it
    (the same tokens `tokenize.generate_tokens` returns, without a named tuple per token), else `tokenize` itself."""
    try:
        import _tokenize  # noqa: PLC0415
        it = _tokenize.TokenizerIter(io.StringIO(text).readline, extra_tokens=True)
    except (ImportError, TypeError):
        return [(t.type, t.string, t.start) for t in tokenize.generate_tokens(io.StringIO(text).readline)]
    try:
        return [(t[0], t[1], t[2]) for t in it]
    except SyntaxError as exc:
        raise tokenize.TokenError(str(exc)) from exc


#: a tool directive's code list inside a comment (`noqa: D401`, `type: ignore[attr-defined]`): codes, not prose, masked
#: before the attribution shapes see the comment (any prose beside them is still read)
_DIRECTIVE_CODES = re.compile(r"\b(?:noqa|ruff|flake8|pylint|pyright|mypy)\b\s*:?\s*(?:disable=)?[A-Za-z]*[0-9]+(?:\s*,\s*[A-Za-z]*[0-9]+)*"
                              r"|\btype:\s*ignore\[[^\]]*\]")


def _mask_directives(comment: str) -> str:
    return _DIRECTIVE_CODES.sub(lambda m: " " * len(m.group()), comment) if "#" in comment else comment


def _py_comment_units(text: str, path: str, tagged: bool = False):
    """Every COMMENT token and every module/class/function docstring of a Python source."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # a source's own invalid escapes are its business, not the scan's
            tree = ast.parse(text, filename=path)
    except SyntaxError as exc:
        raise DecodeError(f"{path}: not parseable Python ({exc.msg}, line {exc.lineno}); the comment scope cannot classify it") from exc
    doc_at = _docstring_starts(tree)
    try:
        toks = _tokens(text)
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise DecodeError(f"{path}: not tokenizable ({exc})") from exc
    for ttype, string, start in toks:
        if ttype == tokenize.COMMENT:
            unit = (start[0], start[1], _mask_directives(string))
            yield (*unit, "#") if tagged else unit
        elif ttype == tokenize.STRING and start in doc_at:
            for k, piece in enumerate(string.split("\n")):
                unit = (start[0] + k, (start[1] if k == 0 else 0), piece)
                yield (*unit, ("doc", start)) if tagged else unit


def _hash_comment(line: str, quotes: str = "'\"") -> "int | None":
    """The column of the `#` comment of one line of shell, YAML or TOML: a `#` at the line start or after whitespace,
    outside quotes."""
    if "#" not in line:
        return None
    q = None
    i = 0
    while i < len(line):
        c = line[i]
        if q:
            if c == "\\" and q == '"':
                i += 2
                continue
            if c == q:
                q = None
        elif c in quotes:
            q = c
        elif c == "\\":
            i += 2
            continue
        elif c == "#" and (i == 0 or line[i - 1] in " \t"):
            return i
        i += 1
    return None


_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_YAML_BLOCK = re.compile(r"(?:^|[:\-]\s|^\s*-\s)\s*[|>][+-]?\d*\s*(?:#.*)?$")


def _hash_comment_units(text: str, path: str):
    """Every `#` comment, skipping shell here-document bodies, YAML block scalars and TOML multi-line strings, whose `#`
    is text."""
    is_shell = path.endswith((".sh", ".bash", ".sbatch")) or text.startswith("#!")
    is_yaml = path.endswith(YAML_SUFFIXES)
    is_toml = path.endswith(".toml")
    heredoc = None
    block_indent = None
    toml_ml = None
    for n, line in enumerate(text.split("\n"), 1):
        line = line.rstrip("\r")
        if heredoc is not None:
            if line.strip() == heredoc:
                heredoc = None
            continue
        if block_indent is not None:
            ind = len(line) - len(line.lstrip(" "))
            if not line.strip() or ind > block_indent:
                continue
            block_indent = None
        if toml_ml is not None:
            if toml_ml in line:
                toml_ml = None
            continue
        col = _hash_comment(line, quotes="'\"`" if is_shell else "'\"")
        code = line if col is None else line[:col]
        if col is not None:
            yield n, col, line[col:]
        if is_shell:
            m = _HEREDOC.search(code)
            if m:
                heredoc = m.group(2)
        elif is_yaml and _YAML_BLOCK.search(code.rstrip()):
            block_indent = len(line) - len(line.lstrip(" "))
        elif is_toml:
            for d in ('"""', "'''"):
                if code.count(d) % 2 == 1:
                    toml_ml = d


_PIP_COMMENT = re.compile(r"(?:^|(?<=\s))#")
_PIP_COMMENT_LINE = re.compile(r"\s*#")


def _pip_comment_units(text: str):
    """Every comment of a pip requirements or constraints file, read as pip reads one: a `#` at the line start or after
    whitespace starts a comment that runs to the end of the line, quotes protect nothing, and a `#` inside a URL
    (`...whl#sha256=...`) is not one. A comment on a requirement line that ends in a backslash continues onto the next
    line, which pip then reads as part of the comment; a line holding only a comment never continues."""
    cont = False
    for n, line in enumerate(text.split("\n"), 1):
        line = line.rstrip("\r")
        if cont:
            yield n, 0, line
            cont = line.endswith("\\") and not _PIP_COMMENT_LINE.match(line)
            continue
        m = _PIP_COMMENT.search(line)
        if m is None:
            continue
        yield n, m.start(), line[m.start():]
        cont = line.endswith("\\") and not _PIP_COMMENT_LINE.match(line)


_SQL_PART = re.compile(r"'(?:[^']|'')*'|--[^\n]*|/\*.*?\*/", re.DOTALL)


def _sql_comment_units(text: str, tagged: bool = False):
    """Every `--` and `/* */` comment outside single-quoted string literals."""
    for m in _SQL_PART.finditer(text):
        s = m.group()
        if s.startswith("'"):
            continue
        first = text.count("\n", 0, m.start()) + 1
        col0 = m.start() - (text.rfind("\n", 0, m.start()) + 1)
        tag = "--" if s.startswith("--") else ("blk", m.start())
        for k, piece in enumerate(s.split("\n")):
            unit = (first + k, (col0 if k == 0 else 0), piece)
            yield (*unit, tag) if tagged else unit


def _markdown_units(text: str):
    for n, line in enumerate(text.split("\n"), 1):
        yield n, 0, line.rstrip("\r")


def _allowlist_units(text: str):
    """The prose of the scan's own allowlist: every comment line, and every row's reason (the text after its third tab).
    A reason is read as a comment, so an exemption cannot carry the attribution it exempts elsewhere; the glob and the
    match are not prose."""
    for n, line in enumerate(text.split("\n"), 1):
        line = line.rstrip("\r")
        if line.lstrip().startswith("#"):
            col = line.index("#")
            yield n, col, line[col:]
        elif line.count("\t") >= 3:
            col = sum(len(f) + 1 for f in line.split("\t", 3)[:3])
            yield n, col, line[col:]


def _cell_source(cell: dict) -> str:
    return "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "")


def _output_texts(o: dict) -> list:
    """The printed texts of one notebook output: its stream text, its text and JSON data and its traceback (no image)."""
    texts = []
    if "text" in o:
        texts.append(o["text"])
    for mime, v in (o.get("data") or {}).items():
        if mime.startswith("text/") or mime == "application/json":
            texts.append(json.dumps(v) if isinstance(v, dict) else v)
    if "traceback" in o:
        texts.append(o["traceback"])
    return ["".join(t) if isinstance(t, list) else str(t) for t in texts]


def _notebook_units(doc: dict, *, comments: bool):
    """(where, text) of a notebook: markdown cells, code-cell comments (comment scope) or code (tree scope), and every
    printed output but image data."""
    for ci, cell in enumerate(doc.get("cells", [])):
        src = _cell_source(cell)
        kind = cell.get("cell_type")
        if kind in ("markdown", "raw"):
            for k, ln in enumerate(src.split("\n")):
                yield f"cell{ci}:L{k + 1}", ln
        elif kind == "code":
            if comments:
                try:
                    units = list(_py_comment_units(src, f"cell{ci}"))
                except DecodeError:              # a cell with magics: its `#` comments, line by line
                    units = []
                    for k, ln in enumerate(src.split("\n")):
                        c = _hash_comment(ln)
                        if c is not None:
                            units.append((k + 1, c, ln[c:]))
                for k, _col, piece in units:
                    yield f"cell{ci}:L{k}", piece
            else:
                for k, ln in enumerate(src.split("\n")):
                    yield f"cell{ci}:L{k + 1}", ln
            for oi, o in enumerate(cell.get("outputs", [])):
                for t in _output_texts(o):
                    for k, ln in enumerate(t.split("\n")):
                        yield f"cell{ci}:out{oi}:L{k + 1}", ln


def _notebook_texts(doc: dict):
    """(where prefix, lines, tagged units) of each notebook text the comment scope reads, for its passages: a markdown or
    raw cell, a code cell's comments and docstrings, a printed output."""
    for ci, cell in enumerate(doc.get("cells", [])):
        src = _cell_source(cell)
        kind = cell.get("cell_type")
        lines = [ln.rstrip("\r") for ln in src.split("\n")]
        if kind in ("markdown", "raw"):
            yield f"cell{ci}:", lines, [(k, 0, ln, "md") for k, ln in enumerate(lines, 1)]
        elif kind == "code":
            try:
                units = list(_py_comment_units(src, f"cell{ci}", tagged=True))
            except DecodeError:              # a cell with magics: its `#` comments, line by line
                units = [(k, c, ln[c:], "#") for k, ln in enumerate(lines, 1) if (c := _hash_comment(ln)) is not None]
            yield f"cell{ci}:", lines, units
            for oi, o in enumerate(cell.get("outputs", [])):
                for t in _output_texts(o):
                    out = [ln.rstrip("\r") for ln in t.split("\n")]
                    yield f"cell{ci}:out{oi}:", out, [(k, 0, ln, "out") for k, ln in enumerate(out, 1)]


#: what each continuation line of a passage repeats before its prose, by tag: a comment marker and the blank around it,
#: a block comment's `*`, a Markdown blockquote's `>`; a docstring, a printed output or an allowlist reason only blank
_LEADS = {"#": re.compile(r"[ \t]*(?:#+:?)?[ \t]*"), "--": re.compile(r"[ \t]*(?:--+)?[ \t]*"),
          "blk": re.compile(r"[ \t]*(?:/\*+|\*+(?!/))?[ \t]*"), "md": re.compile(r"[ \t]*(?:>[ \t]*)*"),
          "doc": re.compile(r"[ \t]*"), "out": re.compile(r"[ \t]*"), "row": re.compile(r"[ \t]*")}
#: a line that ends inside a hyphenated word: its next line goes on without a space
_HYPHEN_END = re.compile(r"[A-Za-z0-9]-\Z")


def _passages(units: list, lines: list) -> list:
    """The comment passages of one text, as the comment scope's rows read them across line breaks. `units` are its
    tagged comment units (line number, column, text, tag) in order, `lines` its lines. A passage is two or more units,
    each on the line after the one before and with the same tag (comments of one kind, one docstring, one block comment,
    the lines of one page or notebook text), opening its line right after a unit that closes its own (a line of code, or
    code beside a comment, ends it), up to a unit that is blank once its marker is removed. Each passage is `(text,
    segments)`: the units joined by one space after each continuation loses the marker it repeats (no space after a word
    a hyphen broke), and per unit `(line number, column of its text in the line, offset of that text in the passage,
    offset where the joint before it starts)`."""
    out, run, last = [], [], None

    def close():
        if len(run) > 1:
            out.append(_join(run))
        run.clear()

    for n, col, piece, tag in units:
        piece = piece.rstrip("\r")
        line = lines[n - 1] if 0 < n <= len(lines) else ""
        kind = tag[0] if isinstance(tag, tuple) else tag
        # where its column says (a comment whose lint directive was masked keeps its length and its marker)
        placed = line[col:col + len(piece)] == piece or (
            kind == "#" and len(line) >= col + len(piece) and line[col:col + 1] == piece[:1])
        lead = _LEADS[kind].match(piece).end()
        if not (placed and last is not None and n == last[0] + 1 and tag == last[3] and not line[:col].strip()
                and not lines[last[0] - 1][last[1] + len(last[2]):].strip()):
            close()
        if not placed or not piece[lead:].strip():
            close()
            last = None
            continue
        run.append((n, col, piece, lead))
        last = (n, col, piece, tag)
    close()
    return out


def _join(run: list) -> tuple:
    n, col, piece, _lead = run[0]
    text, segs = piece.rstrip(), [(n, col, 0, 0)]
    for n, col, piece, lead in run[1:]:
        cut = len(text)
        if not _HYPHEN_END.search(text):
            text += " "
        segs.append((n, col + lead, len(text), cut))
        text += piece[lead:].rstrip()
    return text, segs


def _at(segs: list, x: int) -> tuple:
    """(line number, column) of offset `x` of a passage."""
    n, col, off, _cut = segs[bisect.bisect_right([sg[2] for sg in segs], x) - 1]
    return n, col + x - off


def in_comment_scope_markdown(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path.endswith(".md") and (path.startswith("docs/") or name.upper().startswith("README"))


def is_requirements_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name.endswith(REQUIREMENTS_SUFFIXES) and any(stem in name for stem in REQUIREMENTS_STEMS)


def is_hash_comment_file(path: str, text: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path.endswith(COMMENT_HASH_SUFFIXES) or name in COMMENT_HASH_NAMES or name.startswith("Dockerfile") \
        or text.startswith("#!")


def in_leaf_scope(path: str) -> bool:
    return (path.startswith(LEAF_PREFIXES) or path in LEAF_FILES) and path.endswith((".json", ".jsonl", ".yaml", ".yml"))


# --------------------------------------------------------------------------------------------------- the scan


@dataclass
class Report:
    hits: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    scanned: int = 0
    complete: bool = True          # every scope, every decoder, the whole publication set


class Scanner:
    def __init__(self, root: Path, table, *, scopes=SCOPES, binaries: bool = True, frame_sqlite: bool = True, tmpdir=None):
        self.root = Path(root)
        self.m = build_matchers(table)
        self.scopes = tuple(scopes)
        self.binaries = binaries
        self.frame_sqlite = frame_sqlite
        self.tmpdir = tmpdir
        self.report = Report()

    # ---- emit
    def _emit(self, scope, path, where, unit, r, s, e, origin=""):
        self.report.hits.append(Hit.make(path, where, r.tier, r.token_id, unit, s, e, scope, origin))

    def _tree_text(self, path: str, text, *, base: int = 1, prefix: str = ""):
        for n, unit, r, s, e in self.m["tree"].lines(text, base):
            self._emit("tree", path, f"{prefix}L{n}", unit, r, s, e)

    def _batch(self, matcher: str, scope: str, path: str, units: list, origin: str = ""):
        """`units` is a list of (where, text): scanned in one batch."""
        if not units:
            return
        for k, r, s, e in self.m[matcher].batch([u for _, u in units]):
            where, unit = units[k]
            self._emit(scope, path, where, unit, r, s, e, origin)

    # ---- files
    def scan_names(self, files):
        """Every path name of the publication set (tree scope)."""
        missing = [p for p in files if not (self.root / p).is_file() and not (self.root / p).is_symlink()]
        if missing:
            self.report.errors.append(f"{len(missing)} file(s) of the publication set are absent from the working tree (a sparse "
                                      f"checkout?), e.g. {missing[:3]}: scan a full checkout")
        if "tree" in self.scopes:
            for k, r, s, e in self.m["tree"].batch(list(files)):
                self._emit("tree", files[k], "path", files[k], r, s, e)
        return missing

    def scan_files(self, files, *, names: bool = True):
        missing = self.scan_names(files) if names else \
            [p for p in files if not (self.root / p).is_file() and not (self.root / p).is_symlink()]
        sqlite_groups: dict = {}
        for p in files:
            if p in missing:
                continue
            m = SPLIT_SQLITE.match(p)
            if m:
                sqlite_groups.setdefault(m.group("base"), []).append(p)
                continue
            try:
                self.scan_file(p)
            except DecodeError as exc:
                self.report.errors.append(str(exc))
            except (OSError, ValueError, zipfile.BadZipFile, EOFError) as exc:
                self.report.errors.append(f"{p}: cannot be read ({type(exc).__name__}: {exc})")
            self.report.scanned += 1
        for base, parts in sorted(sqlite_groups.items()):
            self.report.scanned += len(parts)
            if not self.binaries or not self.frame_sqlite or not ({"tree", "leaf"} & set(self.scopes)):
                continue
            try:
                self.scan_split_sqlite(base, sorted(parts))
            except DecodeError as exc:
                self.report.errors.append(str(exc))
            except (OSError, sqlite3.Error, zlib.error) as exc:
                self.report.errors.append(f"{base}: the split SQLite cannot be read ({type(exc).__name__}: {exc})")

    def scan_file(self, p: str):
        full = self.root / p
        tree = "tree" in self.scopes
        if full.is_symlink():
            if tree:
                self._tree_text(p, os.readlink(full), prefix="symlink:")
            return
        if p.endswith(".parquet"):
            if self.binaries:
                self.scan_parquet(p)
            return
        if p.endswith(".gz"):
            if self.binaries and tree:
                self.scan_gzip_text(p)
            return
        if p.endswith(ZIP_PICKLES):
            if self.binaries and tree:
                self.scan_zip_pickle(p)
            return
        if p.endswith(".pkl"):
            if self.binaries and tree:
                self._batch("tree", "tree", p, [(f"op{i}", s) for i, s in _pickle_strings(full.read_bytes(), p)])
            return
        if p.endswith(".eval") and zipfile.is_zipfile(full):
            if self.binaries and tree:
                for name, data in _zip_members(full):
                    self._tree_text(p, data, prefix=f"{name}:")
            return
        if p.endswith(".safetensors"):
            if self.binaries and tree:
                with open(full, "rb") as fh:
                    head = fh.read(8)
                    if len(head) < 8:
                        raise DecodeError(f"{p}: shorter than a safetensors header")
                    (n,) = struct.unpack("<Q", head)
                    if n > 1 << 26:
                        raise DecodeError(f"{p}: a safetensors header of {n} bytes is not a header")
                    self._tree_text(p, fh.read(n), prefix="header:")
            return
        data = full.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            if self.binaries and tree:
                self._batch("tree", "tree", p, [(f"run{k}", s) for k, s in enumerate(_printable_runs(data))])
            return
        self.scan_text(p, text, data)

    def scan_gzip_text(self, p: str):
        base = 1
        with gzip.open(self.root / p, "rb") as fh:
            rest = b""
            while True:
                chunk = fh.read(1 << 24)
                if not chunk:
                    break
                buf = rest + chunk
                cut = buf.rfind(b"\n") + 1
                if cut == 0:
                    rest = buf
                    continue
                block, rest = buf[:cut], buf[cut:]
                self._tree_text(p, block, base=base)
                base += block.count(b"\n")
            if rest:
                self._tree_text(p, rest, base=base)

    def scan_zip_pickle(self, p: str):
        full = self.root / p
        if not zipfile.is_zipfile(full):
            self._batch("tree", "tree", p, [(f"op{i}", s) for i, s in _pickle_strings(full.read_bytes(), p)])
            return
        units = []
        for name, data in _zip_members(full):
            if name.endswith(".pkl"):
                units.extend((f"{name}:op{i}", s) for i, s in _pickle_strings(data, f"{p}:{name}"))
            elif not _is_tensor_storage(name) and len(data) <= 1 << 16:
                units.extend((f"{name}:run{k}", s) for k, s in enumerate(_printable_runs(data)))
        self._batch("tree", "tree", p, units)

    def scan_text(self, p: str, text: str, data: "bytes | None" = None):
        if p.endswith(".ipynb"):
            try:
                doc = json.loads(text)
            except ValueError as exc:
                raise DecodeError(f"{p}: not a notebook ({exc})") from exc
            if "tree" in self.scopes:
                units = list(_notebook_units(doc, comments=False))
                units += [(f"metadata:{k}", ln) for k, ln in enumerate(json.dumps(doc.get("metadata", {}), ensure_ascii=False).split("\n"))]
                self._batch("tree", "tree", p, units)
            if "comment" in self.scopes:
                units = list(_notebook_units(doc, comments=True))
                found: dict = {}
                for k, r, s, e in (self.m["comment"].batch([u for _, u in units]) if units else ()):
                    where, unit = units[k]
                    self._emit("comment", p, where, unit, r, s, e)
                    found.setdefault((where, r.token_id), []).append((s, e))
                if self.m["wrap"].rows:
                    for prefix, lines, cell_units in _notebook_texts(doc):
                        cols = {u[0]: u[1] for u in cell_units}
                        own: dict = {}
                        for (where, tok), spans in found.items():
                            num = where[len(prefix) + 1:] if where.startswith(prefix + "L") else ""
                            if num.isdigit() and int(num) in cols:
                                c = cols[int(num)]
                                own[(int(num), tok)] = [(c + a, c + b) for a, b in spans]
                        self._wrap(p, prefix, lines, cell_units, own)
            return
        data = _enc(text) if data is None else data
        if "tree" in self.scopes:
            self._tree_text(p, data)
        if "comment" in self.scopes and self.has_comment_units(p, text):
            single = self.m["comment"].maybe(data)
            wrap = self.m["wrap"].seeded(data)
            units = list(self.comment_units(p, text, tagged=True)) if single or wrap else []
            if units:
                phys = [ln.rstrip("\r") for ln in text.split("\n")]
                found = {}
                for k, r, s, e in (self.m["comment"].batch([u[2] for u in units]) if single else ()):
                    n, col, piece = units[k][:3]
                    line = phys[n - 1] if 0 < n <= len(phys) else piece
                    if line[col:col + len(piece)] != piece:          # a piece not where its column says: report the piece
                        line, col = piece, 0
                    self._emit("comment", p, f"L{n}", line, r, col + s, col + e)
                    found.setdefault((n, r.token_id), []).append((col + s, col + e))
                if wrap:
                    self._wrap(p, "", phys, units, found)
        if "leaf" in self.scopes and in_leaf_scope(p):
            self.scan_leaf_file(p, text)

    def _wrap(self, p: str, prefix: str, lines: list, units: list, found: dict):
        """The comment rows' hits a line break splits inside one comment passage of `units` (see _passages): each at its
        first line, with the lines it spans as its unit, unless one of those lines holds the same row's hit on its own
        (`found`: (line number, token id) -> the columns of such hits on that line)."""
        passages = _passages(units, lines)
        if not passages:
            return
        for k, r, s, e in self.m["wrap"].batch([t for t, _ in passages]):
            segs = passages[k][1]
            if not any(s < cut and e > off for _n, _c, off, cut in segs[1:]):
                continue                                  # within one line, which the line's own scan reads
            (n1, c1), (n2, c2) = _at(segs, s), _at(segs, e - 1)
            span = {n: (c1 if n == n1 else 0, c2 + 1 if n == n2 else len(lines[n - 1])) for n in range(n1, n2 + 1)}
            if any(a < span[n][1] and b > span[n][0] for n in span for a, b in found.get((n, r.token_id), ())):
                continue
            self._emit("comment", p, f"{prefix}L{n1}", "\n".join(lines[n1 - 1:n2]), r, c1,
                       sum(len(lines[n - 1]) + 1 for n in range(n1, n2)) + c2 + 1)

    @staticmethod
    def has_comment_units(p: str, text: str) -> bool:
        return p == ALLOWLIST or p.endswith(PYTHON_SUFFIXES + (".sql",)) or in_comment_scope_markdown(p) \
            or is_requirements_file(p) or is_hash_comment_file(p, text)

    @staticmethod
    def comment_units(p: str, text: str, tagged: bool = False):
        """The comment units of a file as (line number, column, text); `tagged` adds each unit's passage tag."""
        if p == ALLOWLIST:   # a comment line, or a row's reason (a passage of its own)
            units = ((n, c, t, "#" if t.startswith("#") else ("row", n)) for n, c, t in _allowlist_units(text))
        elif p.endswith(PYTHON_SUFFIXES):
            units = _py_comment_units(text, p, tagged=True)
        elif p.endswith(".sql"):
            units = _sql_comment_units(text, tagged=True)
        elif in_comment_scope_markdown(p):
            units = ((n, c, t, "md") for n, c, t in _markdown_units(text))
        elif is_requirements_file(p):
            units = ((n, c, t, "#") for n, c, t in _pip_comment_units(text))
        elif is_hash_comment_file(p, text):
            units = ((n, c, t, "#") for n, c, t in _hash_comment_units(text, p))
        else:
            return
        for u in units:
            yield u if tagged else u[:3]

    def scan_leaf_file(self, p: str, text: str):
        units = []
        if p.endswith(".jsonl"):
            for n, ln in enumerate(text.split("\n"), 1):
                if not ln.strip():
                    continue
                try:
                    doc = json.loads(ln)
                except ValueError as exc:
                    raise DecodeError(f"{p}:{n}: not JSON ({exc})") from exc
                units.extend((f"L{n}:{addr}", v) for addr, v in _json_leaves(doc))
        elif p.endswith(".json"):
            try:
                doc = json.loads(text)
            except ValueError as exc:
                raise DecodeError(f"{p}: not JSON ({exc})") from exc
            units.extend(_json_leaves(doc))
        else:
            try:
                import yaml  # noqa: PLC0415  (a declared dependency; only the leaf scope of YAML seeds needs it)
            except ImportError as exc:
                raise DecodeError(f"{p}: the leaf scope reads YAML and no `yaml` module is importable: {exc}") from exc
            try:
                docs = list(yaml.safe_load_all(text))
            except yaml.YAMLError as exc:
                raise DecodeError(f"{p}: not YAML ({exc})") from exc
            for k, d in enumerate(docs):
                pre = f"doc{k}:" if len(docs) > 1 else ""
                units.extend((pre + addr, v) for addr, v in _json_leaves(d))
        self._batch("leaf", "leaf", p, units)

    # ---- the frame
    @staticmethod
    def _frame_dir(p: str) -> str:
        return p.rsplit("/", 1)[0] if p.startswith(FRAME_PREFIX) else p

    def _frame_values(self, path: str, origin: str, table: str, col: str, values: list):
        """The distinct TEXT values of one column: the whole cell in the tree scope, each leaf in the leaf scope."""
        values = [v for v in values if v]
        if not values:
            return
        if "tree" in self.scopes:
            self._batch("tree", "tree", path, [(f"{table}.{col}", v) for v in values], origin)
        if "leaf" in self.scopes and path.startswith(FRAME_PREFIX):
            leaves = {}
            for v in values:
                for addr, leaf in _value_leaves(f"{table}.{col}", v):
                    if leaf:
                        leaves.setdefault((addr, leaf), None)
            self._batch("leaf", "leaf", path, list(leaves), origin)

    def scan_parquet(self, p: str):
        try:
            import pyarrow as pa  # noqa: PLC0415
            import pyarrow.compute as pc  # noqa: PLC0415
            import pyarrow.parquet as pq  # noqa: PLC0415
        except ImportError as exc:
            raise DecodeError(f"{p}: Parquet needs `pyarrow`, which is not importable: {exc}") from exc
        pf = pq.ParquetFile(self.root / p)
        table = p.rsplit("/", 1)[-1][: -len(".parquet")]
        path = self._frame_dir(p)
        if "tree" in self.scopes:
            units = []
            for k, v in (pf.schema_arrow.metadata or {}).items():
                if k == b"ARROW:schema":
                    try:
                        raw = base64.b64decode(v)
                    except ValueError:
                        raw = v
                    units.extend((f"{table}:schema:{i}", s) for i, s in enumerate(_printable_runs(raw)))
                else:
                    units.append((f"{table}:meta", k.decode("utf-8", "replace") + "=" + v.decode("utf-8", "replace")))
            units.extend((f"{table}:column", name) for name in pf.schema_arrow.names)
            self._batch("tree", "tree", path, units, p)

        def is_text(t):
            return pa.types.is_string(t) or pa.types.is_large_string(t)

        for f in pf.schema_arrow:
            if not (is_text(f.type) or (pa.types.is_dictionary(f.type) and is_text(f.type.value_type))):
                continue
            arr = pf.read(columns=[f.name]).column(f.name)
            if pa.types.is_dictionary(arr.type):
                arr = arr.cast(arr.type.value_type)
            self._frame_values(path, p, table, f.name, pc.unique(arr).to_pylist())

    def scan_split_sqlite(self, base: str, parts: list):
        want = [f"{base}.gz.{i:02d}" for i in range(len(parts))]
        if parts != want:
            raise DecodeError(f"{base}: split parts {parts} are not .gz.00 .. .gz.{len(parts) - 1:02d} in order")
        path = self._frame_dir(base)
        with tempfile.TemporaryDirectory(dir=self.tmpdir, prefix="scrub_scan_") as td:
            db = Path(td) / "frame.sqlite"
            with open(db, "wb") as out:
                d = zlib.decompressobj(16 + zlib.MAX_WBITS)
                for part in parts:
                    with open(self.root / part, "rb") as fh:
                        while True:
                            chunk = fh.read(1 << 22)
                            if not chunk:
                                break
                            out.write(d.decompress(chunk))
                out.write(d.flush())
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
                if "tree" in self.scopes:
                    self._batch("tree", "tree", path, [(f"{t}:ddl", sql) for t, sql in
                                                       conn.execute("SELECT tbl_name, sql FROM sqlite_master WHERE sql IS NOT NULL")],
                                base)
                for t in tables:
                    cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]
                    for c in cols:
                        vals = [v for (v,) in conn.execute(f'SELECT DISTINCT "{c}" FROM "{t}" WHERE typeof("{c}") = \'text\'')]
                        self._frame_values(path, base, t, c, vals)
            finally:
                conn.close()


def _groups(root: Path, files: list) -> list:
    """The files as independent work items, largest first: the parts of one split SQLite together, every binary alone,
    text files in bundles of about 8 MB."""
    sizes = {}
    for p in files:
        try:
            sizes[p] = (root / p).lstat().st_size
        except OSError:
            sizes[p] = 0
    split: dict = {}
    items, bundle, bsize = [], [], 0
    for p in files:
        m = SPLIT_SQLITE.match(p)
        if m:
            split.setdefault(m.group("base"), []).append(p)
        elif p.endswith((".parquet", ".gz", ".pkl", ".eval", ".safetensors") + ZIP_PICKLES) or sizes[p] > 1 << 23:
            items.append([p])
        else:
            bundle.append(p)
            bsize += sizes[p]
            if bsize > 1 << 23:
                items.append(bundle)
                bundle, bsize = [], 0
    if bundle:
        items.append(bundle)
    items.extend(sorted(v) for v in split.values())
    return sorted(items, key=lambda g: -sum(sizes[p] for p in g))


def _work(args):
    root, table, scopes, binaries, frame_sqlite, tmpdir, files = args
    sc = Scanner(Path(root), table, scopes=scopes, binaries=binaries, frame_sqlite=frame_sqlite, tmpdir=tmpdir)
    sc.scan_files(files, names=False)
    return sc.report.hits, sc.report.errors, sc.report.scanned


def default_jobs() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(8, n))


def scan(root: Path, *, roots=(), scopes=SCOPES, binaries=True, frame_sqlite=True, table=None, tokens=None, tmpdir=None,
         files=None, unpublished=None, jobs: int = 1) -> tuple:
    """Run the scan; returns `(scanner, files)`. `table` overrides the token source (tests plant their own), `unpublished`
    the paths left out; `jobs` > 1 scans in that many forked worker processes (where this platform forks)."""
    root = Path(root)
    if table is None:
        table = load_table(Path(tokens) if tokens else root / TABLE_FILE)
    sc = Scanner(root, table, scopes=scopes, binaries=binaries, frame_sqlite=frame_sqlite, tmpdir=tmpdir)
    if files is None:
        files = select(publication_set(root, unpublished), list(roots))
    files = list(files)
    sc.report.complete = not roots and set(scopes) == set(SCOPES) and binaries and frame_sqlite
    if jobs <= 1 or len(files) < 2 or "fork" not in multiprocessing.get_all_start_methods():
        sc.scan_files(files)
        return sc, files
    missing = set(sc.scan_names(files))
    work = [(str(root), tuple(table), tuple(scopes), binaries, frame_sqlite, tmpdir, g)
            for g in _groups(root, [p for p in files if p not in missing])]
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("fork")) as ex:
        for hits, errors, scanned in ex.map(_work, work):
            sc.report.hits.extend(hits)
            sc.report.errors.extend(errors)
            sc.report.scanned += scanned
    sc.report.scanned += len(missing)
    return sc, files


# --------------------------------------------------------------------------------------------------- the command line


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", help="limit to these paths (a file, a directory or a glob); default: the whole publication set")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="the repository or exported tree (default: the checkout holding this tool)")
    ap.add_argument("--tokens", help=f"the token source (default: <root>/{TABLE_FILE})")
    ap.add_argument("--allowlist", help=f"the allowlist (default: <root>/{ALLOWLIST})")
    ap.add_argument("--no-allowlist", action="store_true", help="report every hit, allowlisted or not")
    ap.add_argument("--scope", action="append", choices=SCOPES, help="scan only these scopes (repeatable; default: all three)")
    ap.add_argument("--all-scopes", action="store_true", help="scan all three scopes (the default; for callers that say so)")
    ap.add_argument("--text-only", action="store_true", help="skip every binary decoder (gz, Parquet, SQLite, pickles, .eval, safetensors)")
    ap.add_argument("--no-frame-sqlite", action="store_true", help="skip the split-gzip SQLite (its rows mirror the Parquet set)")
    ap.add_argument("--tmpdir", help="where the split SQLite is reassembled (default: the system temp directory; it needs ~2 GB)")
    ap.add_argument("--jobs", type=int, default=default_jobs(), help="worker processes (default: the usable CPUs, at most 8)")
    ap.add_argument("--ratchet", nargs="?", const="", metavar="FILE",
                    help=f"also hold the per-site counts to the ratchet (default: <root>/{RATCHET}); hits within it pass")
    ap.add_argument("--write-ratchet", nargs="?", const="", metavar="FILE",
                    help=f"record the current per-site counts as the ratchet (default: <root>/{RATCHET}); needs a whole scan")
    ap.add_argument("--counts", action="store_true", help="print the per-tier count of units outside the allowlist as JSON")
    ap.add_argument("--json", action="store_true", help="print the hits outside the allowlist as JSON lines")
    ap.add_argument("--grep", action="store_true", help="grep-style report (path:line:text and a summary line); exit 0 whatever it finds")
    ap.add_argument("--suggest", action="store_true", help="print a candidate allowlist row (reason left blank) for every hit")
    ap.add_argument("--quiet", action="store_true", help="print only the summary")
    a = ap.parse_args(argv)
    if a.all_scopes and a.scope:
        ap.error("--all-scopes and --scope exclude each other")
    root = Path(a.root).resolve()
    scopes = tuple(a.scope) if a.scope else SCOPES
    try:
        sc, _files = scan(root, roots=a.roots, scopes=scopes, binaries=not a.text_only, frame_sqlite=not a.no_frame_sqlite,
                          tokens=a.tokens, tmpdir=a.tmpdir, jobs=a.jobs)
        allow = [] if a.no_allowlist else load_allowlist(Path(a.allowlist) if a.allowlist else root / ALLOWLIST)
    except (ValueError, OSError, SyntaxError, subprocess.CalledProcessError) as exc:
        print(f"scrub_scan: REFUSED: {exc}", file=sys.stderr)
        return 2
    hits = apply_allowlist(sc.report.hits, allow)
    where = " ".join(a.roots) if a.roots else "the publication set"
    whole = sc.report.complete and not sc.report.errors
    if a.write_ratchet is not None:
        if not whole or a.no_allowlist:
            print("scrub_scan: REFUSED: the ratchet records a whole scan (every scope, every decoder, the whole publication set, "
                  "the allowlist applied) with no error", file=sys.stderr)
            for e in sc.report.errors:
                print(f"scrub_scan: ERROR {e}", file=sys.stderr)
            return 2
        path = Path(a.write_ratchet) if a.write_ratchet else root / RATCHET
        doc = write_ratchet(path, hits)
        print(f"scrub_scan: ratchet {'removed (every count is zero)' if doc is None else f'written: {path}'}", file=sys.stderr)
    if a.grep:
        lines = sorted({(h.path, h.where, h.unit) for h in hits if re.fullmatch(r"L\d+", h.where)}, key=lambda x: (x[0], int(x[1][1:])))
        other = sorted({(h.path, h.where, h.unit) for h in hits if not re.fullmatch(r"L\d+", h.where)})
        for p, w, u in lines:
            for k, ln in enumerate(u.split("\n")):                # a hit a line break splits: each line it spans
                print(f"{p}:{int(w[1:]) + k}:{ln}")
        for p, w, u in other:
            print(f"{p}:{w}:{u[:300]}")
        n = len(lines) + len(other)
        print(f"scrub: {n} line(s) under {where}" if n else f"scrub: no identifier under {where}")
        for e in sc.report.errors:
            print(f"scrub: ERROR {e}", file=sys.stderr)
        return 2 if sc.report.errors else 0
    if a.json:
        for h in hits:
            print(json.dumps({"path": h.path, "where": h.where, "tier": h.tier, "token": h.token, "scope": h.scope,
                              "unit_sha256": h.unit_hash, "origin": h.origin, "excerpt": h.excerpt()}, ensure_ascii=False))
    elif a.suggest:
        seen = set()
        for h in hits:
            data = not re.fullmatch(r"L\d+", h.where)
            row = (f"{h.path}#{h.where}" if data else h.path, h.tier, f"{'value' if data else 'line'}:{h.unit_hash}")
            if row not in seen:
                seen.add(row)
                print("\t".join(row) + "\t")
    elif not a.quiet and not a.counts:
        for h in sorted(hits, key=lambda h: (h.path, h.where, h.tier)):
            print(f"{h.path}:{h.where}: {h.tier} [{h.scope}] {h.token[:12]}: {h.excerpt()}")
    counts = tier_counts(hits)
    if a.counts:
        print(json.dumps(counts, sort_keys=True))
    total = sum(counts.values())
    stale = [r for r in allow if not r.used] if whole else []
    print(f"scrub_scan: {sc.report.scanned} file(s) under {where}; {total} unit(s) outside the allowlist"
          + ("" if not total else " (" + ", ".join(f"{t} {n}" for t, n in counts.items() if n) + ")"), file=sys.stderr)
    for r in stale:
        print(f"scrub_scan: STALE allowlist row line {r.lineno} ({r.glob} {r.tier} {r.kind}): it covers nothing", file=sys.stderr)
    for e in sc.report.errors:
        print(f"scrub_scan: ERROR {e}", file=sys.stderr)
    if sc.report.errors:
        return 2
    if a.ratchet is not None:
        rpath = Path(a.ratchet) if a.ratchet else root / RATCHET
        doc = load_ratchet(rpath) if rpath.exists() else None
        rose, loose = check_ratchet(hits, doc)
        for s, t, p, n, rec in rose:
            print(f"scrub_scan: RATCHET {p}: {t} [{s}] {n} unit(s), the ratchet allows {rec}", file=sys.stderr)
        if loose and whole:
            print(f"scrub_scan: RATCHET {loose} site(s) fell below the ratchet: lower it (--write-ratchet)", file=sys.stderr)
        return 1 if rose or (loose and whole) or stale else 0
    return 1 if total or stale else 0


if __name__ == "__main__":
    sys.exit(main())
