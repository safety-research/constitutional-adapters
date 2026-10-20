#!/usr/bin/env python3
"""Compute and certify ``configs/SCRUB_MAP.tsv``: every hash-recorded file the release rewrote, re-derived byte for byte
from its pre-image before a row can exist.

Some files a record names by sha256 (a config a run ran or was graded with, a frozen sha-asserted copy, a harness
manifest, a registry seed) are rewritten by the release: site paths become placeholders, site labels get their neutral
names, and comment text loses process references. A record keeps the pre-release sha256 forever, so a verifier that
re-hashes the file today needs a certified map from that sha to today's. This tool writes the map, and certifies each
row with two checks:

1. **Byte re-derivation.** The pre-image is read from the pre-release commit (``--pre-ref``). Only the declared inputs
   are applied, in this fixed order of stages:

   a. keyed line tables, matched against the pre-image's own lines: ``line_table`` rows replace a block of exact lines
      of one path (each row already proven comment-only by its own checker), and ``header`` rows replace one exact
      comment line a generator used to emit with the line it emits now. A line a ``header`` row wrote is final: no
      text rule of stage (c) touches it.
   b. structured rewrites (``rewrite``): a committed tool, named as ``apply``, run on the files its ``paths`` globs
      name (the shipped-text rewrite of registry seeds)
   c. text rules, in manifest order: ``rename`` (identifier renames) and ``placeholder`` (literal site values to the
      placeholders of ``sisteer.site``), each a table or a committed tool
   d. sha substitution: each embedded pre-image sha256 of another row becomes that row's post-image sha256, except
      under a key that ``historical_keys`` names for that path (a historical record keeps the sha it recorded); a
      prefix of at least eight hex digits is substituted by the same-length prefix

   The result must equal the committed post-image byte for byte. Any other difference fails the row and is printed as
   a unified diff, so an edit to a recorded parameter, an off-table comment edit or a stale embedded sha cannot be
   blessed. Stage (a) must also change comments only: a file that parses (YAML, JSON, JSONL, TOML) parses to the same
   tree before and after it, and in any other file a header row may rewrite ``#`` comment lines (in Markdown, prose
   outside fenced code) only.

2. **Runtime values.** Every substitution a ``placeholder`` table makes in stage (c) is tagged with the text it
   replaced and the text it wrote, trimmed to the part that differs (a rule may spell its context, such as a key). For
   YAML, JSON, JSONL and TOML files the text is parsed three ways (every substitution undone, tagged, and applied) and
   the three trees are walked together: they must have the same shape, and every parsed value that holds a
   substitution must keep its runtime value. A substitution inside a comment has no runtime value and is only counted.
   In any other file every substitution is checked as text, and one that is not a placeholder or a home path may sit
   on a ``#`` comment line only. What a substitution writes says how a reader resolves it:

   - text that holds a placeholder or starts with ``~`` must give back the literal it replaced, through
     ``sisteer.site.expand`` against the site config given by ``--site`` (with the placeholder environment variables
     masked, so only the site config answers). Every file resolves under one cluster: the site's default, unless the
     manifest's ``cluster_of`` binds its path to a named cluster. Both sides are read as a path reader reads them, with
     a leading ``~`` expanded, as ``sisteer.site.expand_path`` does, and a literal's ``$USER`` as the account this
     tool runs as.
   - nothing at all, or the tail of the literal it matched, removes a checkout prefix: the text removed must be an
     absolute directory prefix, and the parsed value must be left a relative path
   - any other text has no value meaning, so it may not change a parsed value

   Renames (stage c ``rename``, tables and ``apply`` tools alike) are checked by what they do to the parsed values. The
   text entering stage (c) and the text after it with every placeholder substitution undone are parsed and walked
   together (both with stage d applied), so every string value or mapping key that differs was changed by a rename. Each
   changed value is split into whitespace-separated tokens and aligned, and every changed span is judged:

   - a span with no path-shaped token on either side is wording (an identifier, a site label, a sentence) and is
     accepted by declaration. A token is path-shaped when it holds a ``/`` next to a letter, digit, ``_``, ``.``, ``-``,
     ``~`` or ``$``, starts with ``~``, or holds a placeholder of the closed set or ``$USER``.
   - a path-shaped span must read the same on the site before and after: each token as a path reader reads it
     (placeholders expanded under the file's cluster, ``$USER`` as this account, a leading ``~`` as its home), with a
     ``<label>:`` host prefix compared through the site's alias table (``site_aliases.toml`` beside the site config,
     ``[site_aliases]`` legacy label = label) and a ``~`` right after it read as the home too
   - a path-shaped span that reads differently passes only under a key that a ``prose`` input declares for the file:
     the value's nearest mapping key, as the release tree spells it, which the file's readers treat as free text
   - a rename that changes a number, a boolean, a null or the parsed shape refuses; a renamed key is judged as a value
     with no prose key

   In a file that does not parse, a changed group of lines must be comment lines (in Markdown, prose outside fenced
   code), or every path-shaped span in it must read the same. The shipped-text rewrites of stage (b) are accepted by
   declaration: they rewrite registry seeds under their own policy, which their own checker certifies.

Which files get a row (computed, never hand-listed): every file of the pre-release tree whose bytes changed and whose
pre-image sha256, or a prefix of it at least eight hex digits long, appears in a record. The records are:

- the committed paper frame of the pre-release tree (every string column of every table)
- the results registry (``--registry``, opened read-only): every text column of tables under 200,000 rows, and the
  sha- and hash-named columns of larger tables
- the pre-release tree itself: a mention inside ``configs/`` under any key, and a mention anywhere under a historical key
- any file given with ``--recorded`` (one sha per line, an optional label after a tab)
- closure: a changed file whose sha another row embeds is a row too, so stage (d) only ever substitutes certified rows

Changed files under the harness inputs (``configs/capeval/manifests/``, the Petri scenario lock, the Petri behaviour
files) also get a row whenever their parsed content changes, because the harness hash covers the parsed manifest rather
than its bytes; a comment-only edit keeps the harness hash and gets no row on that account. A file at a new path is
followed when a ``rename`` input maps its old path to it. Deleted files get no row.

The inputs manifest (``--inputs``, JSON, kept outside the repository with the inputs it names)::

    {"format": "scrub-map-inputs/1",
     "pre_ref": "<commit>",                                   (optional; must equal --pre-ref)
     "inputs": [
       {"role": "line_table", "path": "<file>", "sha256": "<hex>"},
       {"role": "header", "path": "<file>", "sha256": "<hex>", "default_scope": "configs/*"},
       {"role": "rewrite", "apply": "<repo .py>:<function>", "paths": ["configs/registry/**"],
        "files": {"<name>": {"path": "<file>", "sha256": "<hex>"}}},
       {"role": "rename", "apply": "<repo .py>:<loader>.<method>", "path": "<file>", "sha256": "<hex>"},
       {"role": "placeholder", "path": "<file>", "sha256": "<hex>"},
       {"role": "prose", "path": "<file>", "sha256": "<hex>"}],
     "historical_keys": ["<key>", {"key": "<key>", "paths": ["<glob>"]}, ...],   (optional; added to the default set)
     "omit": [{"path": "<pre-release path>", "reason": "<why this file gets no row>"}],     (optional)
     "cluster_of": [{"paths": ["<glob>"], "cluster": "<label>", "reason": "<who reads it there>"}]}  (optional)

A relative ``path`` is relative to the manifest's directory. Every ``sha256`` is checked before anything runs.

- ``line_table`` (JSONL): ``{"path": <pre-release path>, "old": [<exact line>, ...], "new": [<line>, ...]}``; a string
  ``old`` or ``new`` is a one-line block and a null ``new`` deletes the block. A row applies wherever its block occurs
  as consecutive lines of its path, or, with ``"occ": [k, n]``, at the k-th of those places only, and only while the
  pre-image holds exactly n of them (``(path, old, occ)`` is a row's key); rows whose places overlap apply nowhere; a
  written line takes the file's first line ending, and a block's last line keeps the block's own ending when that is
  empty. The table must hold the rows that were applied to the release tree.
- ``prose`` (TSV): ``key<TAB>scope<TAB>reason`` per line, ``#`` comments: a mapping key whose values the readers of
  the files in scope treat as free text, so a rename may reword a path-shaped token in them (check 2); every line
  needs a reason.
- ``header`` (JSONL): ``{"old": <exact line>, "new": <line, may hold \\n; or null>, "scope": "<globs>"}``, or
  ``"paths": [<glob>, ...]`` instead of ``scope``. With neither, the input's ``default_scope`` holds, else every path.
- ``rename`` and ``placeholder`` tables (TSV): ``kind<TAB>match<TAB>replacement[<TAB>scope]`` per line, ``#``
  comments, ``\\t``, ``\\n`` and ``\\\\`` escapes; ``kind`` is ``literal`` (every occurrence), ``word`` (every
  occurrence not inside an identifier) or ``regex`` (Python ``re.sub``, replacement template). Rules apply in file
  order, each to the whole text of a path in scope; a rule without a scope applies to every path.
- ``scope`` is a comma-separated list of globs over repository paths (``*`` crosses ``/``); a glob with a leading
  ``!`` excludes, and the last glob that matches decides, as in ``.gitignore``.
- ``apply`` names a committed function in the release tree: ``<repo .py>:<function>``, called as
  ``fn(text, *, path=<pre-release path>, table=<the input's file>, **files)``, or ``<repo .py>:<loader>.<method>``,
  called as ``loader(<the input's file>).<method>(text, <pre-release path>)``, whose object also renames paths when it
  has ``rewrite_path``. An ``apply`` sees the whole text, and returns it. Its source sha256 joins the commitment below.

``rule_set`` (every row's fifth column) is the sha256 of the canonical inputs manifest: the manifest with every file
reduced to its role, format and sha256, every ``apply`` to its script, function and source sha256, and the effective
historical keys, omitted paths and cluster bindings, serialised as sorted compact JSON. The map's comment header
carries that JSON line itself, so anyone holding the private inputs can check that ``rule_set`` commits to them, and
that this tool, run with them, re-derives every row. The map never names a private path.

    scrub_map.py --pre-ref <commit> --inputs <manifest> --site <site dir> --registry <db>          # compute, then write
    scrub_map.py --verify --pre-ref <commit> --inputs <manifest> --site <site dir> --registry <db> # certify the map

Without ``--post-ref`` the post-images are the working tree of ``--repo`` (default: the current checkout).
``sisteer.site`` is imported from that checkout's ``src/``. Compute writes the map only when every row passes both
checks. ``--verify`` also recomputes the row set and fails on any missing, extra or changed row; ``--rows-only`` skips
that recomputation, and says so. Exit 0: every row certified; 1: a row failed or the row set differs; 2: the inputs or
arguments cannot be used.
"""
from __future__ import annotations

import argparse
import difflib
import fnmatch
import getpass
import hashlib
import importlib
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

COLUMNS: Tuple[str, ...] = ("pre_sha256", "pre_path", "path", "post_sha256", "rule_set", "kind")
KINDS: Tuple[str, ...] = ("config", "harness", "registry_seed", "frozen")
MANIFEST_FORMAT = "scrub-map-inputs/1"
DEFAULT_MAP = "configs/SCRUB_MAP.tsv"
MIN_PREFIX = 8
BIG_TABLE_ROWS = 200_000
MAX_PROBLEMS = 20
ROLES: Tuple[str, ...] = ("line_table", "header", "rewrite", "rename", "placeholder", "prose")
TEXT_ROLES: Tuple[str, ...] = ("rename", "placeholder")
SITE_ALIASES = "site_aliases.toml"

#: Keys whose sha values are historical records (what ran, never recomputed): the release never rewrites them, so
#: stage (d) leaves them alone. Each key holds in the paths its globs match; no globs is every path. ``source_sha256``
#: records data in the capeval task files, while elsewhere (the ifeval matrices) it is a generated pin that moves.
HISTORICAL_KEYS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("campaign_parent_object_sha256", ()), ("serving_config_sha256", ()), ("standup_config_source_sha", ()),
    ("source_sha256", ("configs/capeval/tasks/*",)),
)
#: The first glob a path matches decides its row's kind; anything else is ``config``.
KIND_GLOBS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("registry_seed", ("configs/registry/*",)),
    ("frozen", ("configs/sdfbase/*/frozen/*", "configs/seedrep/source/*")),
    ("harness", ("configs/capeval/manifests/*", "configs/capeval/petri-scenarios.lock.yaml", "configs/capeval/petri/*")),
)
HARNESS_GLOBS: Tuple[str, ...] = dict(KIND_GLOBS)["harness"]
#: In-tree mentions under these prefixes count as records under any key (a config embedding another file's sha).
RECORDING_TREE_PREFIXES: Tuple[str, ...] = ("configs/",)
#: Columns of large registry tables that hash record files, never repository files.
RECORD_HASH_COLUMNS: Tuple[str, ...] = ("record_sha256", "folded_record_sha256", "withdrawn_by_record_sha256")

HEX_TOKEN = re.compile(r"(?<![0-9A-Fa-f])[0-9a-f]{%d,64}(?![0-9A-Fa-f])" % MIN_PREFIX)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
# the key a value on the same line belongs to: `key: <sha>`, `"key": "<sha>"`, `key = <sha>`, `sha256:` prefix allowed
_KEY_BEFORE = re.compile(r"""["']?([A-Za-z_][A-Za-z0-9_.\-]*)["']?\s*[:=]\s*["']?(?:sha256:)?$""")
# a tagged substitution: U+E000, base-20 letters g..z (no hex digit, no line break), U+E001
_MARK_OPEN, _MARK_CLOSE = "\ue000", "\ue001"
_MARK = re.compile("\ue000([g-z]+)\ue001")
_YAML_EXT = (".yaml", ".yml")
_USER_VAR = re.compile(r"\$\{USER\}|\$USER(?![A-Za-z0-9_])")
# check 2 on renames: a token that names a path, the punctuation that may open a token, a `<label>:` host prefix
_PATH_SHAPE = re.compile(r"[\w.~$-]/|/[\w.~$-]")
_TOKEN_LEAD = re.compile(r"""^[("'\[{<`]*""")
_HOST_PREFIX = re.compile(r"^([A-Za-z][A-Za-z0-9_.-]*):(?=[/~$])")
_SPACES = re.compile(r"(\s+)")


class InputsError(Exception):
    """The inputs manifest, an input it names, or the arguments cannot be used (exit 2)."""


# --- small helpers -----------------------------------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def glob_match(path: str, globs: Iterable[str]) -> bool:
    """fnmatch semantics, where ``*`` also crosses ``/`` (so ``configs/registry/*`` is the whole subtree)."""
    return any(fnmatch.fnmatchcase(path, g.replace("**", "*")) for g in globs)


def in_scope(path: str, globs: Sequence[str]) -> bool:
    """The ``.gitignore`` reading of a scope: the last glob that matches decides, ``!`` excludes; no match is out."""
    verdict = False
    for g in globs:
        neg = g.startswith("!")
        if fnmatch.fnmatchcase(path, g[1:] if neg else g):
            verdict = not neg
    return verdict


def parse_scope(text: str, where: str) -> Tuple[str, ...]:
    globs = tuple(g.strip() for g in text.split(",") if g.strip())
    if not globs:
        raise InputsError(f"{where}: an empty scope")
    return globs


def kind_of(path: str) -> str:
    for kind, globs in KIND_GLOBS:
        if glob_match(path, globs):
            return kind
    return "config"


def _encode_mark(n: int) -> str:
    digits = "ghijklmnopqrstuvwxyz"
    s = ""
    while True:
        n, r = divmod(n, 20)
        s = digits[r] + s
        if n == 0:
            return _MARK_OPEN + s + _MARK_CLOSE


def _decode_mark(s: str) -> int:
    n = 0
    for c in s:
        n = n * 20 + (ord(c) - ord("g"))
    return n


def _mark_ids(s: str) -> List[int]:
    return [_decode_mark(x) for x in _MARK.findall(s)]


def _unescape(field_text: str) -> str:
    out, i = [], 0
    while i < len(field_text):
        c = field_text[i]
        if c == "\\" and i + 1 < len(field_text):
            nxt = field_text[i + 1]
            out.append({"t": "\t", "n": "\n", "\\": "\\"}.get(nxt, "\\" + nxt))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


@dataclass
class Line:
    body: str
    end: str            # the line's own terminator ('' on a last line without one)
    final: bool = False  # written by a header row: no text rule of stage (c) touches it


def split_lines(text: str, final: bool = False) -> List[Line]:
    """``str.splitlines(keepends=True)``, each line split into its text and its terminator (every trailing ``\\r`` and
    ``\\n``), the reading the line-table tools use."""
    out = []
    for raw in text.splitlines(keepends=True):
        body = raw.rstrip("\r\n")
        out.append(Line(body, raw[len(body):], final))
    return out


def join_lines(lines: Sequence[Line]) -> str:
    return "".join(ln.body + ln.end for ln in lines)


def unified_diff(want: str, got: str, label: str, limit: int = 120) -> str:
    lines = list(difflib.unified_diff(want.splitlines(keepends=True), got.splitlines(keepends=True),
                                      fromfile=f"re-derived/{label}", tofile=f"committed/{label}", n=2))
    if len(lines) > limit:
        lines = lines[:limit] + [f"... ({len(lines) - limit} more diff lines)\n"]
    return "".join(line if line.endswith("\n") else line + "\n" for line in lines)


# --- trees -------------------------------------------------------------------------------------------------------------

def git(repo: str, *args: str, input_bytes: Optional[bytes] = None) -> bytes:
    r = subprocess.run(["git", "-C", repo, *args], input=input_bytes, capture_output=True)
    if r.returncode != 0:
        raise InputsError(f"git {' '.join(args)} failed in {repo}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


class _CatFile:
    """One ``git cat-file --batch`` process per repository."""

    def __init__(self, repo: str):
        self.proc = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)

    def read(self, spec: str) -> Optional[bytes]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(spec.encode() + b"\n")
        self.proc.stdin.flush()
        header = self.proc.stdout.readline().split()
        if len(header) < 3 or header[-1] == b"missing":
            return None
        data = self.proc.stdout.read(int(header[2]))
        self.proc.stdout.read(1)
        return data

    def close(self) -> None:
        for pipe in (self.proc.stdin, self.proc.stdout):
            if pipe is not None and not pipe.closed:
                pipe.close()
        self.proc.wait()


class GitTree:
    """The blobs of one commit."""

    def __init__(self, repo: str, ref: str):
        self.repo, self.ref = repo, ref
        self.commit = git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
        self.oids: Dict[str, str] = {}
        for entry in git(repo, "ls-tree", "-r", "-z", "--full-tree", self.commit).split(b"\0"):
            if not entry:
                continue
            meta, path = entry.split(b"\t", 1)
            _mode, typ, oid = meta.split()
            if typ == b"blob":
                self.oids[path.decode()] = oid.decode()
        self._cat = _CatFile(repo)
        self.label = self.commit

    def paths(self) -> Iterable[str]:
        return self.oids.keys()

    def has(self, path: str) -> bool:
        return path in self.oids

    def oid(self, path: str) -> Optional[str]:
        return self.oids.get(path)

    def read(self, path: str) -> bytes:
        oid = self.oids.get(path)
        data = self._cat.read(oid) if oid else None
        if data is None:
            raise InputsError(f"{path} is not a blob of {self.commit}")
        return data

    def close(self) -> None:
        self._cat.close()


class DiskTree:
    """The files of a checkout as they are on disk (tracked and untracked-but-not-ignored); a sparse checkout's
    skipped paths are read from the index. A tracked file ``git diff-files`` does not list takes its index blob id;
    every other file is hashed from its bytes."""

    def __init__(self, repo: str):
        self.repo = os.path.abspath(repo)
        self.skipped: Set[str] = set()
        self.files: Set[str] = set()
        self._oid: Dict[str, str] = {}
        dirty = {e.decode() for e in git(self.repo, "diff-files", "--name-only", "-z").split(b"\0") if e}
        for entry in git(self.repo, "ls-files", "-z", "-s").split(b"\0"):
            if entry:
                meta, path = entry.split(b"\t", 1)
                _mode, oid, stage = meta.split()
                if stage == b"0" and path.decode() not in dirty:
                    self._oid[path.decode()] = oid.decode()
        for entry in git(self.repo, "ls-files", "-z", "-v", "--cached").split(b"\0"):
            if not entry:
                continue
            tag, path = entry[:1], entry[2:].decode()
            if tag.islower() and tag != b"s":      # assume-unchanged: diff-files does not look, so hash the bytes
                self._oid.pop(path, None)
            if tag in (b"S", b"s"):
                self.skipped.add(path)
                self.files.add(path)
            elif os.path.lexists(os.path.join(self.repo, path)):
                self.files.add(path)
        for entry in git(self.repo, "ls-files", "-z", "--others", "--exclude-standard").split(b"\0"):
            if entry:
                self.files.add(entry.decode())
        self._cat = _CatFile(self.repo)
        self.label = f"working tree of {self.repo}"

    def paths(self) -> Iterable[str]:
        return self.files

    def has(self, path: str) -> bool:
        return path in self.files

    def read(self, path: str) -> bytes:
        if path in self.skipped:
            data = self._cat.read(f":{path}")
            if data is None:
                raise InputsError(f"{path} is skipped in the sparse checkout and missing from the index")
            return data
        full = os.path.join(self.repo, path)
        if os.path.islink(full):
            return os.readlink(full).encode()
        with open(full, "rb") as fh:
            return fh.read()

    def oid(self, path: str) -> Optional[str]:
        if path not in self.files:
            return None
        if path not in self._oid:
            self._oid[path] = git_blob_id(self.read(path))
        return self._oid[path]

    def close(self) -> None:
        self._cat.close()


# --- the inputs manifest -----------------------------------------------------------------------------------------------

@dataclass
class TextRule:
    kind: str                       # literal | word | regex
    match: str
    replacement: str
    regex: "re.Pattern[str]"
    where: str                      # file:line, for messages
    tagged: bool                    # a placeholder rule: each substitution is tagged for the runtime check
    scope: Optional[Tuple[str, ...]] = None


@dataclass
class BlockRow:
    path: str
    old: Tuple[str, ...]
    new: Tuple[str, ...]
    occ: Optional[Tuple[int, int]] = None   # (k, n): the k-th of exactly n places of ``old``; None: every place


@dataclass
class HeaderRow:
    old: str
    new: Optional[str]
    globs: Optional[Tuple[str, ...]]   # None: every path
    gitignore: bool                    # scope semantics (last match decides) rather than any-match


@dataclass
class Input:
    role: str
    fmt: str
    file: Optional[str] = None              # the input's own file (absolute)
    sha256: Optional[str] = None
    files: Dict[str, Tuple[str, str]] = field(default_factory=dict)   # name -> (absolute path, sha256)
    paths: Tuple[str, ...] = ()
    default_scope: Optional[str] = None
    apply: Optional[str] = None
    apply_sha256: Optional[str] = None
    call: Optional[Callable[[str, str], str]] = None            # (text, pre_path) -> text
    path_call: Optional[Callable[[str], str]] = None            # pre_path -> path, for an apply that renames paths
    rules: List[TextRule] = field(default_factory=list)
    blocks: List[BlockRow] = field(default_factory=list)
    header_rows: List[HeaderRow] = field(default_factory=list)
    prose: List[Tuple[str, Tuple[str, ...], str]] = field(default_factory=list)   # (key, scope, reason)

    def canonical(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"role": self.role, "format": self.fmt}
        if self.sha256:
            out["sha256"] = self.sha256
        if self.files:
            out["files"] = {k: v[1] for k, v in sorted(self.files.items())}
        if self.paths:
            out["paths"] = list(self.paths)
        if self.default_scope:
            out["default_scope"] = self.default_scope
        if self.apply:
            out["apply"] = self.apply
            out["apply_sha256"] = self.apply_sha256
        return out


@dataclass
class Inputs:
    inputs: List[Input]
    historical_keys: Tuple[Tuple[str, Tuple[str, ...]], ...]
    omit: Dict[str, str]
    pre_ref: Optional[str]
    cluster_of: List[Tuple[Tuple[str, ...], str, str]] = field(default_factory=list)   # (globs, cluster, reason)

    def canonical(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"format": MANIFEST_FORMAT, "inputs": [i.canonical() for i in self.inputs],
                               "historical_keys": [[k, list(g)] for k, g in sorted(self.historical_keys)],
                               "omit": sorted(self.omit)}
        if self.cluster_of:
            out["cluster_of"] = [{"paths": list(g), "cluster": c, "reason": r} for g, c, r in self.cluster_of]
        return out

    def historical_for(self, path: str) -> Set[str]:
        """The keys whose sha values are historical records in ``path``."""
        return {k for k, globs in self.historical_keys if not globs or glob_match(path, globs)}

    def historical_text(self) -> str:
        return ", ".join(k + (f" (in {', '.join(g)})" if g else "") for k, g in sorted(self.historical_keys))

    def context_for(self, path: str) -> Optional[str]:
        """The cluster whose block resolves a file's placeholders: the first ``cluster_of`` entry whose globs match
        its path, else None (the site's default cluster)."""
        for globs, cluster, _reason in self.cluster_of:
            if glob_match(path, globs):
                return cluster
        return None

    def prose_for(self, path: str) -> Set[str]:
        """The keys a ``prose`` input declares as free text in ``path`` (a release path)."""
        return {k for inp in self.of_role("prose") for k, scope, _r in inp.prose if in_scope(path, scope)}

    def canonical_json(self) -> str:
        return canonical_json(self.canonical())

    def rule_set(self) -> str:
        return sha256_bytes(self.canonical_json().encode("utf-8"))

    def of_role(self, *roles: str) -> List[Input]:
        return [i for i in self.inputs if i.role in roles]


def _checked_file(base: str, spec: Mapping[str, Any], what: str) -> Tuple[str, str]:
    path, want = spec.get("path"), spec.get("sha256")
    if not isinstance(path, str) or not path:
        raise InputsError(f"{what}: a file needs a path")
    if not isinstance(want, str) or not SHA256.match(want):
        raise InputsError(f"{what}: {path} needs a sha256 of 64 lowercase hex digits")
    full = os.path.normpath(os.path.join(base, os.path.expanduser(path)))
    try:
        with open(full, "rb") as fh:
            got = sha256_bytes(fh.read())
    except OSError as exc:
        raise InputsError(f"{what}: cannot read {full}: {exc}") from exc
    if got != want:
        raise InputsError(f"{what}: {full} hashes to {got}, the manifest declares {want}")
    return full, want


def _load_tsv_rules(path: str, role: str) -> List[TextRule]:
    """``kind<TAB>match<TAB>replacement[<TAB>scope]``; an empty or absent scope is every path."""
    rules = []
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cells = line.split("\t")
            where = f"{role} rules {path}:{n}"
            if len(cells) not in (3, 4):
                raise InputsError(f"{where}: expected kind<TAB>match<TAB>replacement[<TAB>scope], "
                                  f"got {len(cells)} cells")
            kind, match, repl = cells[0].strip(), _unescape(cells[1]), _unescape(cells[2])
            scope = tuple(g.strip() for g in cells[3].split(",") if g.strip()) if len(cells) == 4 else ()
            if not match:
                raise InputsError(f"{where}: an empty match")
            if kind == "literal":
                rx = re.compile(re.escape(match))
            elif kind == "word":
                rx = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(match) + r"(?![A-Za-z0-9_])")
            elif kind == "regex":
                try:
                    rx = re.compile(match)
                except re.error as exc:
                    raise InputsError(f"{where}: bad regex {match!r}: {exc}") from exc
            else:
                raise InputsError(f"{where}: kind {kind!r} is not literal, word or regex")
            rules.append(TextRule(kind, match, repl, rx, where, role == "placeholder", scope or None))
    return rules


def _load_jsonl(path: str, role: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise InputsError(f"{role} table {path}:{n}: not JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise InputsError(f"{role} table {path}:{n}: a row must be a JSON object")
            yield n, obj


def _as_block(value: Any, where: str, name: str, allow_null: bool) -> Tuple[str, ...]:
    if value is None and allow_null:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
        raise InputsError(f"{where}: {name!r} must be a string or a list of strings"
                          + (" (or null)" if allow_null else ""))
    if any("\n" in s or "\r" in s for s in value):
        raise InputsError(f"{where}: a line of {name!r} may not hold a line terminator")
    return tuple(value)


def _load_blocks(path: str) -> List[BlockRow]:
    rows: List[BlockRow] = []
    seen: Dict[Tuple[str, Tuple[str, ...]], Tuple[str, ...]] = {}
    for n, obj in _load_jsonl(path, "line_table"):
        where = f"line table {path}:{n}"
        p = obj.get("path")
        if not isinstance(p, str) or not p:
            raise InputsError(f"{where}: a row needs a path")
        if "new" not in obj:
            raise InputsError(f"{where}: a row needs 'new' (lines, or null to drop the block)")
        old = _as_block(obj.get("old"), where, "old", False)
        if not old:
            raise InputsError(f"{where}: 'old' needs at least one line")
        occ = obj.get("occ")
        if occ is not None:
            if not (isinstance(occ, list) and len(occ) == 2 and all(type(x) is int for x in occ)
                    and 1 <= occ[0] <= occ[1]):
                raise InputsError(f"{where}: 'occ' must be [k, n] with 1 <= k <= n")
            occ = (occ[0], occ[1])
        row = BlockRow(p, old, _as_block(obj["new"], where, "new", True), occ)
        key = (row.path, row.old, row.occ)
        if key in seen:
            if seen[key] != row.new:
                raise InputsError(f"{where}: {p} maps the block starting {old[0]!r} twice, differently")
            continue
        seen[key] = row.new
        rows.append(row)
    return rows


def _load_prose(path: str) -> List[Tuple[str, Tuple[str, ...], str]]:
    """``key<TAB>scope<TAB>reason`` per line; ``#`` comments."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            where = f"prose keys {path}:{n}"
            cells = line.split("\t")
            if len(cells) != 3:
                raise InputsError(f"{where}: expected key<TAB>scope<TAB>reason, got {len(cells)} cells")
            key, reason = cells[0].strip(), cells[2].strip()
            if not key:
                raise InputsError(f"{where}: an empty key")
            if not reason:
                raise InputsError(f"{where}: the key {key!r} needs a reason (why its readers treat it as free text)")
            rows.append((key, parse_scope(cells[1], where), reason))
    return rows


def _load_header(path: str, default_scope: Optional[str]) -> List[HeaderRow]:
    rows: List[HeaderRow] = []
    seen: Dict[Tuple[str, Optional[Tuple[str, ...]], bool], Optional[str]] = {}
    for n, obj in _load_jsonl(path, "header"):
        where = f"header table {path}:{n}"
        old = obj.get("old")
        if not isinstance(old, str) or not old or "\n" in old or "\r" in old:
            raise InputsError(f"{where}: a row needs old (one nonempty line)")
        if "new" not in obj or (obj["new"] is not None and not isinstance(obj["new"], str)):
            raise InputsError(f"{where}: 'new' must be a string (it may hold \\n) or null")
        new = obj["new"]
        if "scope" in obj and "paths" in obj:
            raise InputsError(f"{where}: give scope or paths, not both")
        if "scope" in obj:
            if not isinstance(obj["scope"], str):
                raise InputsError(f"{where}: scope is a comma-separated string of globs")
            globs, gitignore = parse_scope(obj["scope"], where), True
        elif "paths" in obj:
            if not isinstance(obj["paths"], list) or not all(isinstance(g, str) for g in obj["paths"]):
                raise InputsError(f"{where}: paths is a list of globs")
            globs, gitignore = tuple(obj["paths"]) or None, False
        elif default_scope:
            globs, gitignore = parse_scope(default_scope, where), True
        else:
            globs, gitignore = None, False
        key = (old, globs, gitignore)
        if key in seen and seen[key] != new:
            raise InputsError(f"{where}: the line {old!r} is mapped twice, differently")
        seen[key] = new
        rows.append(HeaderRow(old, new, globs, gitignore))
    return rows


def _load_apply(spec: str, inp: Input, tree: Any, repo: str, what: str) -> None:
    """Load an ``apply`` from the release tree into ``inp.call`` (and ``inp.path_call``)."""
    if spec.count(":") != 1:
        raise InputsError(f"{what}: apply {spec!r} must be <repo-relative .py file>:<function> or :<loader>.<method>")
    rel, expr = spec.split(":")
    if os.path.isabs(rel) or ".." in rel.split("/") or not rel.endswith(".py"):
        raise InputsError(f"{what}: apply {spec!r} must name a .py file inside the repository")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$", expr):
        raise InputsError(f"{what}: apply {spec!r}: after the colon, a function or loader.method")
    if not tree.has(rel):
        raise InputsError(f"{what}: apply names {rel}, which the release tree ({tree.label}) does not hold")
    source = tree.read(rel)
    module = types.ModuleType(f"_scrub_map_apply_{len(sys.modules)}")
    module.__file__ = os.path.join(repo, rel)
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    except Exception as exc:  # the tool's own import failure, named
        raise InputsError(f"{what}: cannot load {rel}: {type(exc).__name__}: {exc}") from exc
    inp.apply_sha256 = sha256_bytes(source)
    if "." in expr:
        loader_name, method = expr.split(".")
        loader = getattr(module, loader_name, None)
        if not callable(loader):
            raise InputsError(f"{what}: {rel} has no loader {loader_name}")
        if inp.file is None:
            raise InputsError(f"{what}: apply {spec!r} loads the input's file, so the input needs path and sha256")
        try:
            obj = loader(inp.file)
        except Exception as exc:
            raise InputsError(f"{what}: {rel}:{loader_name} refused {inp.file}: {type(exc).__name__}: {exc}") from exc
        meth = getattr(obj, method, None)
        if not callable(meth):
            raise InputsError(f"{what}: what {loader_name} returns has no method {method}")
        inp.call = lambda text, path, _m=meth: _m(text, path)
        rp = getattr(obj, "rewrite_path", None)
        inp.path_call = rp if callable(rp) else None
    else:
        func = getattr(module, expr, None)
        if not callable(func):
            raise InputsError(f"{what}: {rel} has no function {expr}")
        kwargs = {name: p for name, (p, _sha) in inp.files.items()}
        inp.call = lambda text, path, _f=func, _t=inp.file, _k=kwargs: _f(text, path=path, table=_t, **_k)


def load_inputs(manifest_path: str, *, tree: Any, repo: str) -> Inputs:
    """Read and check the inputs manifest; ``tree`` is the release tree (``apply`` functions are loaded from it)."""
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            man = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise InputsError(f"cannot read the inputs manifest {manifest_path}: {exc}") from exc
    if not isinstance(man, dict) or man.get("format") != MANIFEST_FORMAT:
        raise InputsError(f"{manifest_path}: format must be {MANIFEST_FORMAT!r}")
    unknown = set(man) - {"format", "pre_ref", "inputs", "historical_keys", "omit", "cluster_of"}
    if unknown:
        raise InputsError(f"{manifest_path}: unknown keys {sorted(unknown)}")
    base = os.path.dirname(os.path.abspath(manifest_path))
    inputs: List[Input] = []
    for idx, spec in enumerate(man.get("inputs") or []):
        what = f"{manifest_path} inputs[{idx}]"
        if not isinstance(spec, dict):
            raise InputsError(f"{what}: must be an object")
        bad = set(spec) - {"role", "format", "path", "sha256", "files", "paths", "apply", "apply_sha256",
                           "default_scope"}
        if bad:
            raise InputsError(f"{what}: unknown keys {sorted(bad)}")
        role = spec.get("role")
        if role not in ROLES:
            raise InputsError(f"{what}: role {role!r} is not one of {', '.join(ROLES)}")
        default_fmt = {"line_table": "jsonl", "header": "jsonl", "rename": "tsv", "placeholder": "tsv",
                       "rewrite": "apply", "prose": "tsv"}[role]
        inp = Input(role=role, fmt=spec.get("format") or ("apply" if spec.get("apply") else default_fmt))
        allowed = {"line_table": ("jsonl",), "header": ("jsonl",), "rewrite": ("apply",),
                   "rename": ("tsv", "apply"), "placeholder": ("tsv",), "prose": ("tsv",)}[role]
        if inp.fmt not in allowed:
            hint = (" (placeholder rules must be a table, so each substitution can be checked against the site)"
                    if role == "placeholder" and inp.fmt == "apply" else "")
            raise InputsError(f"{what}: a {role} input is {' or '.join(allowed)}, not {inp.fmt}{hint}")
        if spec.get("path") is not None or spec.get("sha256") is not None:
            inp.file, inp.sha256 = _checked_file(base, spec, what)
        for name, fspec in sorted((spec.get("files") or {}).items()):
            if not isinstance(fspec, dict) or not re.match(r"^[a-z_][a-z0-9_]*$", name) or name in ("path", "table"):
                raise InputsError(f"{what}: files.{name} must be an object with path and sha256, under a lowercase name")
            inp.files[name] = _checked_file(base, fspec, f"{what} files.{name}")
        inp.paths = tuple(spec.get("paths") or ())
        if any(not isinstance(g, str) or os.path.isabs(g) for g in inp.paths):
            raise InputsError(f"{what}: paths must be repository-relative globs")
        inp.default_scope = spec.get("default_scope")
        if inp.default_scope is not None and (role != "header" or not isinstance(inp.default_scope, str)):
            raise InputsError(f"{what}: default_scope is a scope string, on a header input")
        if inp.fmt == "apply":
            if not spec.get("apply"):
                raise InputsError(f"{what}: format apply needs an apply function")
            inp.apply = spec["apply"]
            _load_apply(inp.apply, inp, tree, repo, what)
            if spec.get("apply_sha256") not in (None, inp.apply_sha256):
                raise InputsError(f"{what}: {inp.apply} hashes to {inp.apply_sha256}, the manifest declares "
                                  f"{spec['apply_sha256']}")
            if role == "rewrite" and not inp.paths:
                raise InputsError(f"{what}: a rewrite needs the paths globs it applies to")
        else:
            if spec.get("apply"):
                raise InputsError(f"{what}: apply needs format apply (or no format)")
            if inp.file is None:
                raise InputsError(f"{what}: format {inp.fmt} needs path and sha256")
            if role == "prose":
                inp.prose = _load_prose(inp.file)
            elif inp.fmt == "tsv":
                inp.rules = _load_tsv_rules(inp.file, role)
            elif role == "line_table":
                inp.blocks = _load_blocks(inp.file)
            else:
                inp.header_rows = _load_header(inp.file, inp.default_scope)
        inputs.append(inp)
    if not inputs:
        raise InputsError(f"{manifest_path}: no inputs")
    hk: Set[Tuple[str, Tuple[str, ...]]] = set(HISTORICAL_KEYS)
    for entry in man.get("historical_keys") or []:
        if isinstance(entry, str) and entry:
            hk.add((entry, ()))
        elif (isinstance(entry, dict) and isinstance(entry.get("key"), str) and entry["key"]
              and isinstance(entry.get("paths"), list) and entry["paths"]
              and all(isinstance(g, str) and g for g in entry["paths"])):
            hk.add((entry["key"], tuple(entry["paths"])))
        else:
            raise InputsError(f"{manifest_path}: a historical_keys entry is a key name (every path) or "
                              "{\"key\": <name>, \"paths\": [<globs>]}")
    omit: Dict[str, str] = {}
    for entry in man.get("omit") or []:
        if not isinstance(entry, dict) or not entry.get("path") or not str(entry.get("reason") or "").strip():
            raise InputsError(f"{manifest_path}: every omit entry needs a path and a reason")
        omit[entry["path"]] = str(entry["reason"]).strip()
    cluster_of: List[Tuple[Tuple[str, ...], str, str]] = []
    for entry in man.get("cluster_of") or []:
        globs = entry.get("paths") if isinstance(entry, dict) else None
        if (not isinstance(globs, list) or not globs or not all(isinstance(g, str) and g for g in globs)
                or not isinstance(entry.get("cluster"), str) or not entry["cluster"]
                or not str(entry.get("reason") or "").strip()):
            raise InputsError(f"{manifest_path}: every cluster_of entry needs paths (globs), a cluster and a reason")
        cluster_of.append((tuple(globs), entry["cluster"], str(entry["reason"]).strip()))
    pre_ref = man.get("pre_ref")
    return Inputs(inputs, tuple(sorted(hk)), omit, pre_ref, cluster_of)


# --- applying the declared rules ---------------------------------------------------------------------------------------

@dataclass
class Mark:
    """One tagged substitution: the text it replaced, the text it wrote, and how a reader resolves the new text."""
    literal: str
    replacement: str
    kind: str          # placeholder | home | relocation | text


@dataclass
class Derived:
    """A re-derived text with each tagged substitution still marked; ``after_lines`` is the text stage (a) left,
    ``header_hits`` the lines header rows replaced, with what they wrote, and ``before`` the text that entered stage (c),
    with stage (d) applied (what the renames are judged against)."""
    tagged: str
    marks: List[Mark]
    after_lines: str = ""
    header_hits: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    before: Optional[str] = None

    def post_of(self, s: str) -> str:
        return _MARK.sub(lambda m: self.marks[_decode_mark(m.group(1))].replacement, s)

    def literal_of(self, s: str) -> str:
        return _MARK.sub(lambda m: self.marks[_decode_mark(m.group(1))].literal, s)

    def post_text(self) -> str:
        return self.post_of(self.tagged)

    def literal_text(self) -> str:
        return self.literal_of(self.tagged)


def _call(inp: Input, text: str, path: str) -> str:
    assert inp.call is not None
    try:
        out = inp.call(text, path)
    except InputsError:
        raise
    except Exception as exc:  # the committed tool's own refusal, named
        raise InputsError(f"{inp.apply} failed on {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(out, str):
        raise InputsError(f"{inp.apply} returned {type(out).__name__}, not text, for {path}")
    return out


def rename_path(path: str, inputs: Inputs) -> str:
    """The release path of a pre-release path: ``rename`` tables of format ``tsv`` and ``apply`` objects with
    ``rewrite_path`` rename paths; nothing else does."""
    for inp in inputs.of_role("rename"):
        if inp.path_call is not None:
            path = inp.path_call(path)
        elif inp.fmt == "tsv":
            for rule in inp.rules:
                path = rule.regex.sub(rule.replacement if rule.kind == "regex" else (lambda _m, r=rule.replacement: r),
                                      path)
    return path


def _place_blocks(lines: Sequence[str], rows: Sequence[BlockRow]) -> List[Tuple[int, BlockRow]]:
    """Every place a row's block occurs (a row with ``occ`` (k, n): the k-th place, while there are exactly n); a row with
    any place that overlaps another row's place applies nowhere."""
    placements: List[Tuple[int, int, BlockRow]] = []
    for ri, row in enumerate(rows):
        k = len(row.old)
        hits = [i for i in range(len(lines) - k + 1) if lines[i] == row.old[0] and tuple(lines[i:i + k]) == row.old]
        if row.occ is not None:
            want, total = row.occ
            hits = [hits[want - 1]] if len(hits) == total else []
        placements.extend((i, ri, row) for i in hits)
    placements.sort(key=lambda p: p[0])
    refused: Set[int] = set()
    last_end = -1
    for j, (h, ri, row) in enumerate(placements):
        if h < last_end:
            refused.add(ri)
            refused.add(placements[j - 1][1])
        last_end = max(last_end, h + len(row.old))
    return [(h, row) for h, ri, row in placements if ri not in refused]


def _line_stage(text: str, pre_path: str, inputs: Inputs,
                hits: Optional[List[Tuple[str, Optional[str]]]] = None) -> List[Line]:
    """Stage (a): the line tables' blocks and the header rows, matched against the pre-image's own lines. Each line a
    header row replaced is appended to ``hits`` with what it wrote."""
    lines = split_lines(text)
    rows = [r for inp in inputs.of_role("line_table") for r in inp.blocks if r.path == pre_path]
    seen: Dict[Tuple[Tuple[str, ...], Optional[Tuple[int, int]]], Tuple[str, ...]] = {}
    for r in rows:
        if (r.old, r.occ) in seen and seen[(r.old, r.occ)] != r.new:
            raise InputsError(f"two line tables map the block starting {r.old[0]!r} of {pre_path} two ways")
        seen[(r.old, r.occ)] = r.new
    header: Dict[str, Optional[str]] = {}
    for inp in inputs.of_role("header"):
        for hr in inp.header_rows:
            ok = hr.globs is None or (in_scope(pre_path, hr.globs) if hr.gitignore else glob_match(pre_path, hr.globs))
            if not ok:
                continue
            if hr.old in header and header[hr.old] != hr.new:
                raise InputsError(f"two header rows map the line {hr.old!r} of {pre_path} two ways")
            header[hr.old] = hr.new
    if not rows and not header:
        return lines
    bodies = [ln.body for ln in lines]
    placed = _place_blocks(bodies, list({(r.old, r.occ): r for r in rows}.values())) if rows else []
    term = next((ln.end for ln in lines if ln.end), "\n")
    out: List[Line] = []

    def plain(i: int) -> None:
        ln = lines[i]
        if ln.body not in header:
            out.append(ln)
            return
        new = header[ln.body]
        if hits is not None:
            hits.append((ln.body, new))
        if new is None:
            return
        parts = new.split("\n")
        out.extend(Line(p, "\n", True) for p in parts[:-1])
        out.append(Line(parts[-1], ln.end, True))

    cur = 0
    for h, row in placed:
        for i in range(cur, h):
            plain(i)
        for i in range(h, h + len(row.old)):
            if bodies[i] in header:
                raise InputsError(f"a line table block and a header row both rewrite line {i + 1} of {pre_path}")
        block_end = lines[h + len(row.old) - 1].end
        for j, s in enumerate(row.new):
            out.append(Line(s, term if (j < len(row.new) - 1 or block_end) else block_end))
        cur = h + len(row.old)
    for i in range(cur, len(lines)):
        plain(i)
    return out


_TRIM_STOP = re.compile(r"[A-Za-z0-9_${}~]")


def _trimmed(literal: str, repl: str) -> Tuple[int, int]:
    """How much text a substitution keeps at each end: the common prefix up to the first ``$`` or ``~``, and the common
    suffix of characters that cannot belong to a placeholder or a home prefix. A rule may spell its context (a key, a
    separator); the tag then covers only the text that changes, so the tagged text parses like the other two."""
    p = 0
    while p < len(literal) and p < len(repl) and literal[p] == repl[p] and literal[p] not in "$~":
        p += 1
    s = 0
    while (s < len(literal) - p and s < len(repl) - p and literal[-1 - s] == repl[-1 - s]
           and not _TRIM_STOP.match(literal[-1 - s])):
        s += 1
    return p, s


def value_kind_of(replacement: str) -> str:
    """How a reader resolves what a placeholder-table substitution wrote (the text it changed, trimmed)."""
    if not replacement:
        return "relocation"
    if "$" in replacement:
        return "placeholder"
    if replacement.startswith("~"):
        return "home"
    return "text"


def _sub_rules(text: str, rules: Sequence[TextRule], marks: Optional[List[Mark]]) -> str:
    for rule in rules:
        if not rule.tagged or marks is None:
            if rule.kind == "regex":
                text = rule.regex.sub(rule.replacement, text)
            else:
                text = rule.regex.sub(lambda _m, r=rule.replacement: r, text)
            continue

        def tag(m: "re.Match[str]", rule: TextRule = rule) -> str:
            literal = m.group(0)
            repl = m.expand(rule.replacement) if rule.kind == "regex" else rule.replacement
            if literal == repl:
                return literal
            p, s = _trimmed(literal, repl)
            head, tail = literal[:p], literal[len(literal) - s:]
            lit, new = literal[p:len(literal) - s], repl[p:len(repl) - s]
            if new and "$" not in new and not new.startswith("~") and len(new) < len(lit) and lit.endswith(new):
                # the rule removed a prefix and kept the rest of the text: only the removed prefix is tagged
                lit, new, tail = lit[:len(lit) - len(new)], "", new + tail
            marks.append(Mark(lit, new, value_kind_of(new)))
            return head + _encode_mark(len(marks) - 1) + tail
        text = rule.regex.sub(tag, text)
    return text


def _text_stage(lines: List[Line], inp: Input, path: str, marks: List[Mark]) -> List[Line]:
    """One ``rename`` or ``placeholder`` input. A table skips the lines a header row wrote; an ``apply`` sees the whole
    text, and must keep the line count when the file holds such lines."""
    if inp.call is not None:
        new = split_lines(_call(inp, join_lines(lines), path))
        finals = [ln.final for ln in lines]
        if any(finals):
            if len(new) != len(lines):
                raise InputsError(f"{inp.apply} changed the number of lines of {path}, which holds header-table lines")
            for ln, f in zip(new, finals, strict=True):
                ln.final = f
        return new
    rules = [r for r in inp.rules if r.scope is None or in_scope(path, r.scope)]
    if not rules:
        return lines
    out: List[Line] = []
    run: List[Line] = []

    def flush() -> None:
        if run:
            out.extend(split_lines(_sub_rules(join_lines(run), rules, marks)))
            run.clear()

    for ln in lines:
        if ln.final:
            flush()
            out.append(ln)
        else:
            run.append(ln)
    flush()
    return out


def _sha_stage(text: str, subst: Mapping[str, str], historical: Iterable[str],
               prefixes: Mapping[int, Mapping[str, str]]) -> str:
    hist = set(historical)

    def one(m: "re.Match[str]") -> str:
        token = m.group(0)
        new = subst.get(token) if len(token) == 64 else prefixes.get(len(token), {}).get(token)
        if new is None:
            return token
        line_start = text.rfind("\n", 0, m.start()) + 1
        km = _KEY_BEFORE.search(text, line_start, m.start())
        if km and km.end() == m.start() and km.group(1) in hist:
            return token
        return new if len(token) == 64 else new[:len(token)]

    return HEX_TOKEN.sub(one, text)


def derive(pre: bytes, pre_path: str, inputs: Inputs, subst: Mapping[str, str],
           prefixes: Optional[Mapping[int, Mapping[str, str]]] = None) -> Derived:
    """Apply the declared stages to one pre-image (see the module docstring for the order)."""
    try:
        text = pre.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputsError(f"{pre_path}: the pre-image is not UTF-8 text ({exc}); a mapped file must be text") from exc
    if _MARK_OPEN in text or _MARK_CLOSE in text:
        raise InputsError(f"{pre_path}: the pre-image holds U+E000 or U+E001, which this tool uses to tag substitutions")
    hits: List[Tuple[str, Optional[str]]] = []
    lines = _line_stage(text, pre_path, inputs, hits)
    after_lines = join_lines(lines)
    rewrites = [i for i in inputs.of_role("rewrite") if glob_match(pre_path, i.paths)]
    if rewrites:
        if any(ln.final for ln in lines):
            raise InputsError(f"{pre_path}: a header table and a rewrite both apply")
        text = join_lines(lines)
        for inp in rewrites:
            text = _call(inp, text, pre_path)
        lines = split_lines(text)
    marks: List[Mark] = []
    before = join_lines(lines)
    for inp in inputs.of_role(*TEXT_ROLES):
        lines = _text_stage(lines, inp, pre_path, marks)
    text = join_lines(lines)
    found = _mark_ids(text)
    if sorted(found) != list(range(len(marks))):
        raise InputsError(f"{pre_path}: a rule after a tagged substitution rewrote, dropped or duplicated it")
    hist = inputs.historical_for(pre_path)
    text = _sha_stage(text, subst, hist, prefixes or {})
    return Derived(text, marks, after_lines, hits, _sha_stage(before, subst, hist, prefixes or {}))


# --- the site ----------------------------------------------------------------------------------------------------------

class SiteExpander:
    """Placeholders expanded by ``sisteer.site`` against one site config, with the placeholder environment variables
    masked so only the site config answers. Only placeholders of the closed set are expanded; any other ``$`` stays.
    Both sides of a comparison are read as a path reader reads them: a leading ``~`` is expanded."""

    def __init__(self, site: str, module: Optional[types.ModuleType] = None, repo: Optional[str] = None):
        try:
            if module is not None:
                mod = module
            else:
                src = os.path.join(repo, "src") if repo else None
                if src and os.path.isdir(src) and src not in sys.path:
                    sys.path.insert(0, src)
                mod = importlib.import_module("sisteer.site")
        except ImportError as exc:
            raise InputsError(f"the runtime-value check needs sisteer.site (the placeholder resolver): {exc}") from exc
        for attr in ("NAMES", "PLACEHOLDERS", "load_site", "expand"):
            if not hasattr(mod, attr):
                raise InputsError(f"sisteer.site has no {attr}; this tool expands placeholders through it")
        self.mod = mod
        self.origin = getattr(mod, "__file__", None) or mod.__name__
        masked = {"SISTEER_CONFIG", "SISTEER_SITE_DIR", "SISTEER_CLUSTER_LABEL"}
        for envs, keys, *_rest in mod.PLACEHOLDERS.values():
            masked.update(envs)
            masked.update(keys)
        self.masked = tuple(sorted(masked))
        self.env = {k: v for k, v in os.environ.items() if k not in masked}
        self.home = os.path.expanduser("~")
        cfg = os.path.join(site, "config.env") if os.path.isdir(site) else site
        try:
            self.site = mod.load_site(cfg, env=self.env)
        except Exception as exc:  # the resolver's own refusal, named
            raise InputsError(f"cannot read the site config {cfg}: {exc}") from exc
        self.labels: List[str] = list(getattr(self.site, "labels", []) or [])
        names = "|".join(re.escape(n) for n in mod.NAMES)
        self.pattern = re.compile(r"\$\{(?:%s)\}|\$(?:%s)(?![A-Za-z0-9_])" % (names, names))
        self.cache: Dict[Tuple[str, Optional[str]], Tuple[bool, str]] = {}
        self.alias_file = os.path.join(os.path.dirname(os.path.abspath(cfg)), SITE_ALIASES)
        self.aliases = self._load_aliases(self.alias_file)

    @staticmethod
    def _load_aliases(path: str) -> Dict[str, str]:
        """The site's ``[site_aliases]`` table (legacy label = label), or nothing when the site has no alias file."""
        if not os.path.isfile(path):
            return {}
        import tomllib
        try:
            with open(path, "rb") as fh:
                table = tomllib.load(fh).get("site_aliases", {})
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise InputsError(f"cannot read the site's alias table {path}: {exc}") from exc
        if not isinstance(table, dict) or not all(isinstance(k, str) and isinstance(v, str) and v
                                                  for k, v in table.items()):
            raise InputsError(f"{path}: [site_aliases] maps each legacy label to a label, both strings")
        return dict(table)

    def path_shaped(self, token: str) -> bool:
        """Whether a whitespace-free token names a path: a ``/`` next to a name character, a leading ``~``, or a
        placeholder of the closed set or ``$USER``."""
        return bool(_PATH_SHAPE.search(token) or _TOKEN_LEAD.sub("", token).startswith("~")
                    or self.pattern.search(token) or _USER_VAR.search(token))

    def read_span(self, span: str, context: Optional[str]) -> str:
        """A span of text with every token read as a path reader reads it: a ``<label>:`` host prefix through the alias
        table, then placeholders under ``context``, ``$USER`` and a leading ``~``. Raises LookupError as ``resolve``."""
        out = []
        for part in _SPACES.split(span):
            if not part or part.isspace():
                out.append(part)
                continue
            lead = _TOKEN_LEAD.match(part).group(0)
            rest, host = part[len(lead):], ""
            m = _HOST_PREFIX.match(rest)
            if m:
                host = self.aliases.get(m.group(1), m.group(1)) + ":"
                rest = rest[m.end():]
            out.append(lead + host + self.resolve(rest, context))
        return "".join(out)

    @staticmethod
    def resolve_literal(value: str) -> str:
        """A literal as a path reader saw it: ``$USER`` is the account this tool runs as (the site's user), and a
        leading ``~`` is that account's home."""
        if _USER_VAR.search(value):
            value = _USER_VAR.sub(lambda _m: os.environ.get("USER") or getpass.getuser(), value)
        return os.path.expanduser(value) if value.startswith("~") else value

    def resolve(self, value: str, context: Optional[str]) -> str:
        """``value`` with its placeholders expanded under ``context``, then a leading ``~`` expanded; raises
        LookupError with the resolver's message when a placeholder does not resolve there."""
        def one(m: "re.Match[str]") -> str:
            key = (m.group(0), context)
            if key not in self.cache:
                try:
                    self.cache[key] = (True, self.mod.expand(m.group(0), context, site=self.site, env=self.env))
                except Exception as exc:  # PlaceholderError or a missing block: this context does not resolve it
                    self.cache[key] = (False, str(exc))
            ok, got = self.cache[key]
            if not ok:
                raise LookupError(got)
            return got
        return self.resolve_literal(self.pattern.sub(one, value))

    def match(self, marks: Sequence[Mark], context: Optional[str]) -> Tuple[bool, str]:
        """Whether ``context`` (the file's cluster, None for the default) resolves every mark's new text to its literal;
        on failure, why, for the first mark that does not."""
        ctx = context or "default"
        for mk in marks:
            want = self.resolve_literal(mk.literal)
            try:
                got = self.resolve(mk.replacement, context)
            except LookupError as exc:
                why = f"[{ctx}] {exc}"
            else:
                if got == want:
                    continue
                why = f"[{ctx}] {got!r} is not {want!r}"
            what = "placeholder" if mk.kind == "placeholder" else "home-relative path"
            return False, f"{what} {mk.replacement!r} does not expand to the {mk.literal!r} it replaced: {why}"
        return True, ctx


def _parse_structured(text: str, path: str) -> Any:
    if path.endswith(_YAML_EXT):
        import yaml
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        return list(yaml.load_all(text, Loader=loader))
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if path.endswith(".json"):
        return json.loads(text)
    if path.endswith(".toml"):
        import tomllib
        return tomllib.loads(text)
    return None


def structured(path: str) -> bool:
    return path.endswith(_YAML_EXT + (".json", ".jsonl", ".toml"))


def _same_scalar(a: Any, b: Any) -> bool:
    return type(a) is type(b) and (a == b or (a != a and b != b))


def _first_difference(a: Any, b: Any, where: str = "$") -> Optional[str]:
    """Where two parsed trees first differ (types included), or None when they are equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return f"{where}: keys {list(a)!r} became {list(b)!r}"
        for k in a:
            got = _first_difference(a[k], b[k], f"{where}.{k}")
            if got:
                return got
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{where}: {len(a)} items became {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            got = _first_difference(x, y, f"{where}[{i}]")
            if got:
                return got
        return None
    return None if _same_scalar(a, b) else f"{where}: {a!r} became {b!r}"


def _is_comment_line(line: str) -> bool:
    return line.lstrip().startswith("#")


def comment_check(raw: str, after: str, path: str, header_hits: Sequence[Tuple[str, Optional[str]]]) -> List[str]:
    """Stage (a) may change comments only. A file that parses (YAML, JSON, JSONL, TOML) must parse to the same tree
    before and after the line and header tables; in any other file, every line a header row replaced, and every line it
    wrote, must be a ``#`` comment line. (Line-table rows were each proven comment-only by their own checker.)"""
    if raw == after:
        return []
    if structured(path):
        try:
            before = _parse_structured(raw, path)
        except Exception:  # a template that never parses: fall back to the comment-line rule below
            before = _UNPARSED
        if before is not _UNPARSED:
            try:
                now = _parse_structured(after, path)
            except Exception as exc:
                return [f"the line or header tables leave the file unparseable: {type(exc).__name__}: {exc}"]
            diff = _first_difference(before, now)
            return [f"the line or header tables changed a parsed value: {diff}"] if diff else []
    problems = []
    fenced = _fenced_lines(raw) if path.endswith(".md") else set()
    for old, new in header_hits:
        written = [] if new is None else new.split("\n")
        if path.endswith(".md"):   # Markdown prose: any line outside a fenced code block, never a fence itself
            ok = old not in fenced and not any(w.lstrip().startswith("```") for w in [old] + written)
        else:
            ok = _is_comment_line(old) and all(_is_comment_line(w) for w in written)
        if not ok:
            problems.append(f"a header row rewrote the line {old!r} as {new!r}; "
                            "a header row rewrites comment lines (or Markdown prose) only")
    return problems


def _fenced_lines(text: str) -> Set[str]:
    """The lines of a Markdown text that sit inside a fenced code block (fences included)."""
    out, inside = set(), False
    for ln in text.splitlines():
        if ln.lstrip().startswith("```"):
            out.add(ln)
            inside = not inside
        elif inside:
            out.add(ln)
    return out


_UNPARSED = object()


@dataclass
class Stats:
    checked: int = 0      # placeholder and home-relative substitutions resolved on the site
    comments: int = 0     # substitutions outside every parsed value (comments): no runtime value
    declared: int = 0     # checkout-prefix removals in values, and text on comment lines of files that do not parse
    renamed_resolved: int = 0   # renamed spans that name a path and read the same on the site
    renamed_prose: int = 0      # renamed path-shaped spans that read differently, under a declared prose key
    renamed_wording: int = 0    # renamed spans with no path (and renamed comment lines of files that do not parse)
    context: str = "default"

    def add(self, other: "Stats") -> None:
        self.checked += other.checked
        self.comments += other.comments
        self.declared += other.declared
        self.renamed_resolved += other.renamed_resolved
        self.renamed_prose += other.renamed_prose
        self.renamed_wording += other.renamed_wording


def _directory_prefix(literal: str) -> bool:
    """An absolute directory prefix, the only text an empty replacement may remove from a value."""
    return literal.startswith(("/", "~/")) and literal.endswith("/") and len(literal) > 2


class _Walk:
    def __init__(self, d: Derived, site: SiteExpander, context: Optional[str]):
        self.d, self.site, self.context = d, site, context
        self.problems: List[str] = []
        self.seen: Set[int] = set()
        self.stats = Stats(context=context or "default")

    def fail(self, msg: str) -> None:
        if len(self.problems) < MAX_PROBLEMS:
            self.problems.append(msg)

    def value(self, L: Any, T: str, P: Any, where: str) -> None:
        """A parsed string that holds substitutions: it parses the same three ways and keeps its runtime value."""
        ids = _mark_ids(T)
        self.seen.update(ids)
        if not (isinstance(L, str) and L == self.d.literal_of(T) and isinstance(P, str) and P == self.d.post_of(T)):
            self.fail(f"{where}: a substitution changes how the value parses: {L!r} becomes {P!r}")
            return
        marks = [self.d.marks[i] for i in ids]
        if any(m.kind == "text" for m in marks):
            self.fail(f"{where}: a rule declared as text with no value meaning changed the value {L!r} to {P!r}")
        relocations = [i for i in ids if self.d.marks[i].kind == "relocation"]
        if relocations:
            self.stats.declared += len(relocations)
            first = _MARK.match(T)
            removed = self.d.marks[relocations[0]].literal
            if not _directory_prefix(removed):
                self.fail(f"{where}: an empty replacement may only remove an absolute directory prefix from a value, "
                          f"but it removed {removed!r} ({L!r} becomes {P!r})")
            elif (len(relocations) > 1 or first is None or _decode_mark(first.group(1)) != relocations[0]
                    or not P or P.startswith(("/", "~", "$"))):
                self.fail(f"{where}: a checkout-prefix removal must leave a relative path at the start of the value, "
                          f"but {L!r} becomes {P!r}")
        checkable = [m for m in marks if m.kind in ("placeholder", "home")]
        if checkable:
            self.stats.checked += len(checkable)
            ok, why = self.site.match(checkable, self.context)
            if not ok:
                self.fail(f"{where}: {why}")

    def walk(self, L: Any, T: Any, P: Any, where: str) -> None:
        if len(self.problems) >= MAX_PROBLEMS:
            return
        if isinstance(T, str) and _MARK.search(T):
            self.value(L, T, P, where)
            return
        if isinstance(T, dict):
            if not (isinstance(L, dict) and isinstance(P, dict) and len(L) == len(T) == len(P)):
                self.fail(f"{where}: the mapping parses with another shape after the release")
                return
            for k, tv in T.items():
                if isinstance(k, str) and _MARK.search(k):
                    lk, pk = self.d.literal_of(k), self.d.post_of(k)
                    if lk not in L or pk not in P:
                        self.fail(f"{where}: the key {lk!r} does not parse as {pk!r} after the release")
                        return
                    self.value(lk, k, pk, f"{where}.<key {pk}>")
                else:
                    lk = pk = k
                    if k not in L or k not in P:
                        self.fail(f"{where}: the key {k!r} does not parse the same way after the release")
                        return
                self.walk(L[lk], tv, P[pk], f"{where}.{pk}")
            return
        if isinstance(T, list):
            if not (isinstance(L, list) and isinstance(P, list) and len(L) == len(T) == len(P)):
                self.fail(f"{where}: the list parses with another length after the release")
                return
            for i, (a, t, b) in enumerate(zip(L, T, P, strict=True)):
                self.walk(a, t, b, f"{where}[{i}]")
            return
        if not (_same_scalar(L, T) and _same_scalar(P, T)):
            self.fail(f"{where}: {L!r} parses as {P!r} after the release")


class _RenameWalk:
    """Check 2 on renames: the text entering stage (c) against the text after it with placeholders undone."""

    def __init__(self, site: SiteExpander, context: Optional[str], prose: Set[str], stats: Stats):
        self.site, self.context, self.prose, self.stats = site, context, prose, stats
        self.problems: List[str] = []

    def fail(self, msg: str) -> None:
        if len(self.problems) < MAX_PROBLEMS:
            self.problems.append(msg)

    def text(self, before: str, after: str, where: str, key: Optional[str]) -> None:
        """One renamed string: each changed span that names a path must read the same, or sit under a prose key."""
        tb, ta = _SPACES.split(before), _SPACES.split(after)
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, tb, ta, autojunk=False).get_opcodes():
            if op == "equal":
                continue
            sb, sa = "".join(tb[i1:i2]), "".join(ta[j1:j2])
            if not any(t and not t.isspace() and self.site.path_shaped(t) for t in tb[i1:i2] + ta[j1:j2]):
                self.stats.renamed_wording += 1
                continue
            try:
                rb, ra = self.site.read_span(sb, self.context), self.site.read_span(sa, self.context)
                why = None if rb == ra else f"it reads as {ra!r}, the recorded text as {rb!r}"
            except LookupError as exc:
                why = f"[{self.context or 'default'}] {exc}"
            if why is None:
                self.stats.renamed_resolved += 1
            elif key is not None and key in self.prose:
                self.stats.renamed_prose += 1
            else:
                self.fail(f"{where}: a rename changed the path {sb!r} to {sa!r}: {why}. A rename may move a path only to "
                          "text that reads the same on the site, or reword it under a key a prose input declares for "
                          "this file")

    def walk(self, b: Any, a: Any, where: str, key: Optional[str]) -> None:
        if len(self.problems) >= MAX_PROBLEMS:
            return
        if isinstance(b, dict) and isinstance(a, dict):
            if len(b) != len(a):
                self.fail(f"{where}: a rename changed the mapping's keys: {list(b)!r} became {list(a)!r}")
                return
            for (kb, vb), (ka, va) in zip(b.items(), a.items(), strict=True):
                if kb != ka:
                    if isinstance(kb, str) and isinstance(ka, str):
                        self.text(kb, ka, f"{where}.<key {ka}>", None)
                    else:
                        self.fail(f"{where}: a rename changed the key {kb!r} to {ka!r}")
                self.walk(vb, va, f"{where}.{ka}", ka if isinstance(ka, str) else None)
            return
        if isinstance(b, list) and isinstance(a, list):
            if len(b) != len(a):
                self.fail(f"{where}: a rename changed the list's length: {len(b)} items became {len(a)}")
                return
            for i, (x, y) in enumerate(zip(b, a, strict=True)):
                self.walk(x, y, f"{where}[{i}]", key)
            return
        if isinstance(b, (dict, list)) or isinstance(a, (dict, list)):
            self.fail(f"{where}: a rename changed the value's shape: {b!r} became {a!r}")
            return
        if _same_scalar(b, a):
            return
        if isinstance(b, str) and isinstance(a, str):
            self.text(b, a, where, key)
            return
        self.fail(f"{where}: a rename changed a value that is not text: {b!r} became {a!r}")

    def lines(self, before: str, after: str, path: str) -> None:
        """A file that parses no way: each changed group of lines is comment text, or its paths read the same."""
        md = path.endswith(".md")
        fenced_b, fenced_a = (_fenced_lines(before), _fenced_lines(after)) if md else (set(), set())

        def commentish(line: str, fenced: Set[str]) -> bool:
            return _is_comment_line(line) or (md and line not in fenced)

        bl, al = before.split("\n"), after.split("\n")
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, bl, al, autojunk=False).get_opcodes():
            if op == "equal":
                continue
            # a group of as many lines after as before is judged line by line, any other group as one block
            pairs = ([([bl[i1 + k]], [al[j1 + k]], i1 + k) for k in range(i2 - i1)] if i2 - i1 == j2 - j1
                     else [(bl[i1:i2], al[j1:j2], i1)])
            for gb, ga, at in pairs:
                if all(commentish(x, fenced_b) for x in gb) and all(commentish(x, fenced_a) for x in ga):
                    self.stats.renamed_wording += 1
                    continue
                self.text("\n".join(gb), "\n".join(ga), f"line {at + 1}", None)


def _rename_check(d: Derived, path: str, site: SiteExpander, context: Optional[str], prose: Set[str],
                  stats: Stats) -> List[str]:
    """Check 2 on renames for one row (see the module docstring)."""
    before, after = d.before, d.literal_text()
    if before is None or before == after:
        return []
    w = _RenameWalk(site, context, prose, stats)
    if structured(path):
        parsed: List[Any] = []
        errors: List[Tuple[str, Exception]] = []
        for label, text in (("before the text rules", before), ("after them, placeholders undone", after)):
            try:
                parsed.append(_parse_structured(text, path))
            except Exception as exc:  # yaml.YAMLError, json.JSONDecodeError, tomllib.TOMLDecodeError
                parsed.append(_UNPARSED)
                errors.append((label, exc))
        if not errors:
            w.walk(parsed[0], parsed[1], "$", None)
            return w.problems
        if len(errors) == 1:
            label, exc = errors[0]
            return [f"a rename changes whether the file parses ({label}: {type(exc).__name__}: {exc})"]
    w.lines(before, after, path)
    return w.problems


def runtime_check(d: Derived, path: str, site: SiteExpander, context: Optional[str] = None,
                  prose: Optional[Set[str]] = None) -> Tuple[List[str], Stats]:
    """Check 2 for one row under the file's one cluster (``context``; None is the site's default): every tagged
    substitution in a parsed value keeps its runtime value, and every rename that moves a path in a parsed value reads
    the same on the site or sits under a key in ``prose``."""
    problems, stats = _placeholder_check(d, path, site, context)
    problems.extend(_rename_check(d, path, site, context, prose or set(), stats))
    return problems, stats


def _placeholder_check(d: Derived, path: str, site: SiteExpander,
                       context: Optional[str]) -> Tuple[List[str], Stats]:
    """Check 2 on placeholder tables: every tagged substitution in a parsed value keeps its runtime value."""
    w = _Walk(d, site, context)
    if not d.marks:
        return [], w.stats
    if structured(path):
        parsed: List[Any] = []
        errors: List[Tuple[str, Exception]] = []
        for label, text in (("with every substitution undone", d.literal_text()), ("tagged", d.tagged),
                            ("as committed", d.post_text())):
            try:
                parsed.append(_parse_structured(text, path))
            except Exception as exc:  # yaml.YAMLError, json.JSONDecodeError, tomllib.TOMLDecodeError
                parsed.append(None)
                errors.append((label, exc))
        if not errors:
            w.walk(parsed[0], parsed[1], parsed[2], "$")
            w.stats.comments = len(d.marks) - len(w.seen)
            return w.problems, w.stats
        if len(errors) < 3:
            label, exc = errors[0]
            return [f"the file parses one way but not another ({label}: {type(exc).__name__}: {exc})"], w.stats
        # a file that never parses (a template) is checked line by line, as any other file
    for i, mk in enumerate(d.marks):
        m = re.search(re.escape(_encode_mark(i)), d.tagged)
        line = d.tagged[d.tagged.rfind("\n", 0, m.start()) + 1:m.start()] if m else ""
        if mk.kind in ("placeholder", "home"):
            w.stats.checked += 1
            ok, why = site.match([mk], context)
            if not ok:
                w.fail(why)
        elif _is_comment_line(line):
            w.stats.declared += 1
        else:
            w.fail(f"{mk.literal!r} became {mk.replacement!r} outside a comment of a file this tool cannot parse; "
                   "only a placeholder or a home path may change such a line")
    return w.problems, w.stats


# --- records -----------------------------------------------------------------------------------------------------------

class Candidates:
    """The pre-image shas of changed files, and their prefixes, for filtering what the record scans keep."""

    def __init__(self, shas: Iterable[str]):
        self.full: Set[str] = set(shas)
        self.prefix: Dict[int, Dict[str, Set[str]]] = {}
        for s in self.full:
            for n in range(MIN_PREFIX, 64):
                self.prefix.setdefault(n, {}).setdefault(s[:n], set()).add(s)

    def resolve(self, token: str) -> Set[str]:
        if len(token) == 64:
            return {token} if token in self.full else set()
        return self.prefix.get(len(token), {}).get(token, set())


class Records:
    def __init__(self, cands: Candidates):
        self.cands = cands
        self.sources: Dict[str, Set[str]] = {}

    def add_text(self, text: str, label: str) -> None:
        for token in HEX_TOKEN.findall(text):
            for sha in self.cands.resolve(token):
                self.sources.setdefault(sha, set()).add(label if len(token) == 64 else f"{label} (prefix {len(token)})")


def scan_frame(tree: GitTree, records: Records) -> int:
    """Every string column of every Parquet table of the committed frame (``paper/snapshot/CURRENT``)."""
    if not tree.has("paper/snapshot/CURRENT"):
        raise InputsError(f"{tree.label} has no paper/snapshot/CURRENT; pass --no-frame if it has no frame")
    snap = tree.read("paper/snapshot/CURRENT").decode().strip()
    base = f"paper/snapshot/snapshots/{snap}/"
    tables = sorted(p for p in tree.paths() if p.startswith(base) and p.endswith(".parquet"))
    if not tables:
        raise InputsError(f"{tree.label}: the frame {snap} has no Parquet tables")
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise InputsError(f"reading the frame needs pyarrow: {exc}") from exc
    for p in tables:
        t = pq.read_table(io.BytesIO(tree.read(p)))
        name = p[len(base):-len(".parquet")]
        for col in t.column_names:
            arr = t[col]
            if not (pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type)):
                continue
            hits = pc.filter(arr, pc.match_substring_regex(arr, "[0-9a-f]{%d}" % MIN_PREFIX))
            for v in hits.to_pylist():
                if v:
                    records.add_text(v, f"frame:{name}.{col}")
    return len(tables)


def scan_registry(db: str, records: Records) -> int:
    """The results registry, read-only: every text column of small tables, sha and hash columns of large ones."""
    if not os.path.isfile(db):
        raise InputsError(f"no registry database at {db}")
    con = sqlite3.connect(f"file:{os.path.abspath(db)}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        pattern = "*" + "[0-9a-f]" * MIN_PREFIX + "*"
        for t in tables:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
            try:
                n = con.execute(f'SELECT max(rowid) FROM "{t}"').fetchone()[0] or 0
            except sqlite3.OperationalError:     # WITHOUT ROWID
                n = BIG_TABLE_ROWS
            if n >= BIG_TABLE_ROWS:
                cols = [c for c in cols if re.search("sha|hash", c, re.I) and c not in RECORD_HASH_COLUMNS]
            for c in cols:
                q = f'SELECT "{c}" FROM "{t}" WHERE typeof("{c}") = \'text\' AND "{c}" GLOB ?'
                for (v,) in con.execute(q, (pattern,)):
                    records.add_text(v, f"registry:{t}.{c}")
        return len(tables)
    finally:
        con.close()


def scan_tree(tree: GitTree, records: Records, inputs: "Inputs") -> int:
    """Mentions in the pre-release tree: inside ``configs/`` under any key, elsewhere only under a historical key.
    ``git grep`` first narrows the tree to the text files that hold an eight-digit prefix of some candidate."""
    prefixes = sorted({s[:MIN_PREFIX] for s in records.cands.full})
    if not prefixes:
        return 0
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(prefixes) + "\n")
        pattern_file = fh.name
    try:
        r = subprocess.run(["git", "-C", tree.repo, "grep", "-I", "-l", "-z", "-F", "-f", pattern_file, tree.commit,
                            "--", ".", ":(exclude)paper/snapshot"], capture_output=True)
    finally:
        os.unlink(pattern_file)
    if r.returncode not in (0, 1):
        raise InputsError(f"git grep failed in {tree.repo}: {r.stderr.decode(errors='replace').strip()}")
    head = tree.commit + ":"
    paths = [e.decode()[len(head):] if e.decode().startswith(head) else e.decode()
             for e in r.stdout.split(b"\0") if e]
    for p in paths:
        hist = inputs.historical_for(p)
        text = tree.read(p).decode("utf-8", "replace")
        anywhere = p.startswith(RECORDING_TREE_PREFIXES)
        for m in HEX_TOKEN.finditer(text):
            shas = records.cands.resolve(m.group(0))
            if not shas:
                continue
            line_start = text.rfind("\n", 0, m.start()) + 1
            km = _KEY_BEFORE.search(text, line_start, m.start())
            key = km.group(1) if km and km.end() == m.start() else None
            if anywhere or key in hist:
                label = f"tree:{p}" + (f" ({key})" if key else "")
                for sha in shas:
                    records.sources.setdefault(sha, set()).add(label)
    return len(paths)


def scan_recorded_file(path: str, records: Records) -> None:
    try:
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, start=1):
                if not line.strip() or line.startswith("#"):
                    continue
                token, _, label = line.rstrip("\n").partition("\t")
                if not HEX_TOKEN.fullmatch(token.strip()):
                    raise InputsError(f"{path}:{n}: {token!r} is not a sha (at least {MIN_PREFIX} lowercase hex digits)")
                for sha in records.cands.resolve(token.strip()):
                    records.sources.setdefault(sha, set()).add(f"recorded:{os.path.basename(path)}:{n}"
                                                               + (f" ({label.strip()})" if label.strip() else ""))
    except OSError as exc:
        raise InputsError(f"cannot read {path}: {exc}") from exc


# --- the row set -------------------------------------------------------------------------------------------------------

@dataclass
class Change:
    pre_path: str
    path: str
    pre_sha256: str
    post_sha256: str
    pre: bytes
    post: bytes
    sources: Set[str] = field(default_factory=set)

    @property
    def kind(self) -> str:
        return kind_of(self.path)


def changed_files(pre: GitTree, post: Any, inputs: Inputs) -> Tuple[Dict[str, Change], List[str]]:
    """Every pre-release file whose bytes differ in the release tree (followed through a renamed path); the deleted
    ones are listed apart."""
    changes: Dict[str, Change] = {}
    deleted = []
    for p in sorted(pre.paths()):
        q = p if post.has(p) else None
        if q is None:
            r = rename_path(p, inputs)
            q = r if r != p and post.has(r) else None
        if q is None:
            deleted.append(p)
            continue
        if q == p and pre.oid(p) == post.oid(q):
            continue
        before, after = pre.read(p), post.read(q)
        if before == after:
            continue
        changes[p] = Change(p, q, sha256_bytes(before), sha256_bytes(after), before, after)
    return changes, deleted


def harness_change(ch: Change) -> bool:
    """A harness input whose hash-relevant content changed: parsed content for YAML and JSON, bytes otherwise."""
    if not glob_match(ch.path, HARNESS_GLOBS):
        return False
    if not structured(ch.path):
        return True
    try:
        return _parse_structured(ch.pre.decode("utf-8"), ch.path) != _parse_structured(ch.post.decode("utf-8"), ch.path)
    except Exception:
        return True


def select_rows(changes: Mapping[str, Change], records: Records, inputs: Inputs) -> Tuple[List[Change], Dict[str, str]]:
    """The changed files that get a row: recorded, harness, then the closure over embedded shas. Omitted files are
    returned apart with their reasons; an omit entry that names no candidate refuses."""
    by_sha: Dict[str, List[Change]] = {}
    for ch in changes.values():
        by_sha.setdefault(ch.pre_sha256, []).append(ch)
    chosen: Dict[str, Change] = {}
    for sha, labels in records.sources.items():
        for ch in by_sha.get(sha, []):
            ch.sources |= labels
            chosen[ch.pre_path] = ch
    for ch in changes.values():
        if harness_change(ch):
            ch.sources.add("harness: the parsed content changed")
            chosen[ch.pre_path] = ch
    queue = list(chosen.values())
    while queue:
        row = queue.pop()
        hist = inputs.historical_for(row.pre_path)
        text = row.pre.decode("utf-8", "replace")
        for m in HEX_TOKEN.finditer(text):
            for sha in records.cands.resolve(m.group(0)):
                line_start = text.rfind("\n", 0, m.start()) + 1
                km = _KEY_BEFORE.search(text, line_start, m.start())
                if km and km.end() == m.start() and km.group(1) in hist:
                    continue
                for ch in by_sha.get(sha, []):
                    ch.sources.add(f"embedded by {row.pre_path}")
                    if ch.pre_path not in chosen:
                        chosen[ch.pre_path] = ch
                        queue.append(ch)
    omitted = {}
    for p, reason in inputs.omit.items():
        if p not in chosen:
            raise InputsError(f"omit names {p}, which is not a changed, recorded file of this release")
        omitted[p] = reason
        del chosen[p]
    return sorted(chosen.values(), key=lambda c: (c.path, c.pre_path)), omitted


def substitution_tables(rows: Sequence[Change]) -> Tuple[Dict[str, str], Dict[int, Dict[str, str]]]:
    subst = {r.pre_sha256: r.post_sha256 for r in rows}
    prefixes: Dict[int, Dict[str, str]] = {}
    for n in range(MIN_PREFIX, 64):
        seen: Dict[str, Set[str]] = {}
        for pre, post in subst.items():
            seen.setdefault(pre[:n], set()).add(post)
        prefixes[n] = {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}
    return subst, prefixes


@dataclass
class RowResult:
    change: Change
    problems: List[str]
    stats: Stats = field(default_factory=Stats)


def certify(rows: Sequence[Change], inputs: Inputs, site: SiteExpander) -> List[RowResult]:
    """Both checks for every row."""
    subst, prefixes = substitution_tables(rows)
    results = []
    for ch in rows:
        problems: List[str] = []
        try:
            d = derive(ch.pre, ch.pre_path, inputs, subst, prefixes)
        except InputsError as exc:
            results.append(RowResult(ch, [f"cannot re-derive: {exc}"]))
            continue
        want = d.post_text()
        if want.encode("utf-8") != ch.post:
            problems.append("the committed bytes are not the declared rules applied to the pre-image:\n"
                            + unified_diff(want, ch.post.decode("utf-8", "replace"), ch.path))
        problems.extend(comment_check(ch.pre.decode("utf-8"), d.after_lines, ch.pre_path, d.header_hits))
        found, stats = runtime_check(d, ch.path, site, inputs.context_for(ch.path), inputs.prose_for(ch.path))
        problems.extend(found)
        results.append(RowResult(ch, problems, stats))
    return results


# --- the map file ------------------------------------------------------------------------------------------------------

def render_map(rows: Sequence[Change], inputs: Inputs) -> str:
    rs = inputs.rule_set()
    out = [
        "# Files a record names by sha256 that the release rewrote by declared rules only. Each row was",
        "# re-derived byte for byte from its pre-image, and each placeholder it holds was expanded on a site to",
        "# the literal it replaced, by scripts/release/scrub_map.py --verify, before it was written. A verifier",
        "# that finds a recorded pre_sha256 and today's post_sha256 at path may pass as verified_via_map; no other",
        "# mismatch is certified.",
        f"# rule_set {rs}",
        f"# inputs {inputs.canonical_json()}",
        "\t".join(COLUMNS),
    ]
    for ch in rows:
        out.append("\t".join((ch.pre_sha256, ch.pre_path, ch.path, ch.post_sha256, rs, ch.kind)))
    return "\n".join(out) + "\n"


@dataclass
class MapFile:
    rows: List[Tuple[str, ...]]
    rule_set_line: Optional[str]
    inputs_line: Optional[str]


def parse_map(text: str, source: str) -> MapFile:
    rows: List[Tuple[str, ...]] = []
    header = False
    rs_line = inputs_line = None
    for n, line in enumerate(text.splitlines(), start=1):
        if line.startswith("# rule_set "):
            rs_line = line[len("# rule_set "):].strip()
        elif line.startswith("# inputs "):
            inputs_line = line[len("# inputs "):]
        if not line.strip() or line.startswith("#"):
            continue
        cells = tuple(line.split("\t"))
        if not header:
            if cells != COLUMNS:
                raise InputsError(f"{source}:{n}: the header must be {' '.join(COLUMNS)}, got {list(cells)}")
            header = True
            continue
        if len(cells) != len(COLUMNS):
            raise InputsError(f"{source}:{n}: {len(cells)} cells, expected {len(COLUMNS)}")
        pre_sha, _pre_path, _path, post_sha, rs, kind = cells
        if not all(SHA256.match(x) for x in (pre_sha, post_sha, rs)) or kind not in KINDS:
            raise InputsError(f"{source}:{n}: a malformed row")
        rows.append(cells)
    return MapFile(rows, rs_line, inputs_line)


# --- driver ------------------------------------------------------------------------------------------------------------

def _print_results(results: Sequence[RowResult], out: Any) -> int:
    bad = 0
    total = Stats()
    for r in results:
        total.add(r.stats)
        if r.problems:
            bad += 1
            print(f"FAIL {r.change.path} (pre {r.change.pre_path} {r.change.pre_sha256[:12]}):", file=out)
            for p in r.problems:
                print("  " + p.replace("\n", "\n  ").rstrip(), file=out)
    print(f"substitutions: {total.checked} resolved on the site, {total.comments} in comments (no runtime value), "
          f"{total.declared} declared (checkout-prefix removals, or text in files that do not parse)", file=out)
    print(f"renamed spans: {total.renamed_resolved} name a path that reads the same on the site, {total.renamed_prose} "
          f"reword a path under a declared prose key, {total.renamed_wording} wording with no path", file=out)
    return bad


def run(argv: Optional[Sequence[str]] = None, *, site_module: Optional[types.ModuleType] = None,
        out: Any = None) -> int:
    out = out or sys.stdout
    ap = argparse.ArgumentParser(prog="scrub_map.py", description=__doc__.split("\n")[0])
    ap.add_argument("--verify", action="store_true", help="certify the committed map instead of writing one")
    ap.add_argument("--pre-ref", required=True, help="the pre-release commit every pre-image is read from")
    ap.add_argument("--inputs", required=True, help="the private inputs manifest (JSON)")
    ap.add_argument("--site", required=True,
                    help="the site config directory (or config.env) placeholders expand against")
    ap.add_argument("--repo", default=".", help="the release checkout (default: the current directory)")
    ap.add_argument("--post-ref", default=None, help="read the post-images from this commit, not the working tree")
    ap.add_argument("--map", default=None, help=f"the map to verify (default: {DEFAULT_MAP} of the release tree)")
    ap.add_argument("--out", default=None, help=f"where compute writes the map (default: <repo>/{DEFAULT_MAP})")
    reg = ap.add_mutually_exclusive_group()
    reg.add_argument("--registry", default=None, help="the results registry database, opened read-only")
    reg.add_argument("--no-registry", action="store_true", help="take no records from a registry (said in the output)")
    ap.add_argument("--no-frame", action="store_true",
                    help="take no records from the committed frame (said in the output)")
    ap.add_argument("--recorded", action="append", default=[], help="a file of recorded shas, one per line")
    ap.add_argument("--rows-only", action="store_true",
                    help="--verify: re-derive the map's rows without recomputing the set")
    ap.add_argument("--report", default=None, help="write a JSON report of every row, its sources and its result")
    args = ap.parse_args(argv)
    trees: List[Any] = []
    try:
        repo = git(args.repo, "rev-parse", "--show-toplevel").decode().strip()
        src = os.path.join(repo, "src")
        if os.path.isdir(src) and src not in sys.path:   # sisteer, for apply functions and the resolver alike
            sys.path.insert(0, src)
        pre = GitTree(repo, args.pre_ref)
        trees.append(pre)
        post: Any = GitTree(repo, args.post_ref) if args.post_ref else DiskTree(repo)
        trees.append(post)
        inputs = load_inputs(args.inputs, tree=post, repo=repo)
        manifest_pre = (git(repo, "rev-parse", "--verify", f"{inputs.pre_ref}^{{commit}}").decode().strip()
                        if inputs.pre_ref else pre.commit)
        if manifest_pre != pre.commit:
            raise InputsError(f"the manifest's pre_ref {inputs.pre_ref} is not --pre-ref {args.pre_ref}")
        site = SiteExpander(args.site, site_module, repo)
        rs = inputs.rule_set()
        print(f"pre-ref {pre.commit}; release tree: {post.label}", file=out)
        print(f"rule_set {rs} ({len(inputs.inputs)} inputs; historical keys: {inputs.historical_text()})", file=out)
        unknown = sorted({c for _g, c, _r in inputs.cluster_of} - set(site.labels))
        if unknown:
            raise InputsError(f"cluster_of names {', '.join(unknown)}, which the site config does not "
                              f"(clusters: {', '.join(site.labels) or 'none'})")
        print(f"site {site.site.path} through {site.origin}: clusters {', '.join(site.labels) or 'none'} (a file "
              f"resolves under the default cluster unless cluster_of names one: {len(inputs.cluster_of)} entries); "
              f"home {site.home}; masked from the environment: {', '.join(site.masked)}", file=out)
        print(f"site aliases: {len(site.aliases)} legacy labels"
              + ("" if site.aliases else f" (no {SITE_ALIASES} beside the site config)")
              + f"; prose keys: {sum(len(i.prose) for i in inputs.of_role('prose'))} declared", file=out)
        need_set = not (args.verify and args.rows_only)
        if need_set and not args.registry and not args.no_registry:
            raise InputsError("name the registry (--registry DB) or say --no-registry")
        if args.rows_only and not args.verify:
            raise InputsError("--rows-only belongs to --verify")
        if not args.verify and args.post_ref and not args.out:
            raise InputsError("with --post-ref, name where the map goes with --out")

        changes, deleted = changed_files(pre, post, inputs)
        print(f"{len(changes)} changed files, {len(deleted)} removed or not followed", file=out)
        rows: List[Change] = []
        omitted: Dict[str, str] = {}
        if need_set:
            records = Records(Candidates(ch.pre_sha256 for ch in changes.values()))
            if args.no_frame:
                print("records: the committed frame was NOT read (--no-frame)", file=out)
            else:
                print(f"records: frame, {scan_frame(pre, records)} tables", file=out)
            if args.registry:
                print(f"records: registry {args.registry} (read-only), {scan_registry(args.registry, records)} tables",
                      file=out)
            else:
                print("records: NO registry was read (--no-registry)", file=out)
            print(f"records: pre-release tree, {scan_tree(pre, records, inputs)} files with a candidate "
                  "sha prefix", file=out)
            for f in args.recorded:
                scan_recorded_file(f, records)
            rows, omitted = select_rows(changes, records, inputs)
            for p, why in sorted(omitted.items()):
                print(f"omitted by the manifest: {p}: {why}", file=out)

        if args.verify:
            map_path = args.map or DEFAULT_MAP
            if args.map and os.path.isfile(args.map):
                with open(args.map, encoding="utf-8") as fh:
                    text = fh.read()
            elif post.has(map_path):
                text = post.read(map_path).decode("utf-8")
            else:
                raise InputsError(f"no map at {map_path} in {post.label}")
            mf = parse_map(text, map_path)
            set_problems = []
            if mf.rule_set_line != rs:
                set_problems.append(f"the map's rule_set header is {mf.rule_set_line}, these inputs give {rs}")
            if mf.inputs_line is None or sha256_bytes(mf.inputs_line.encode("utf-8")) != rs:
                set_problems.append("the map's inputs header line does not hash to its rule_set")
            stale = sorted({r[4] for r in mf.rows} - {rs})
            if stale:
                set_problems.append(f"rows carry another rule_set: {', '.join(stale)}")
            if need_set:
                want = {(c.pre_sha256, c.pre_path, c.path, c.post_sha256, rs, c.kind) for c in rows}
                have = set(mf.rows)
                for r in sorted(want - have):
                    set_problems.append(f"missing row: {r[1]} -> {r[2]} ({r[0][:12]} -> {r[3][:12]}, {r[5]})")
                for r in sorted(have - want):
                    set_problems.append(f"row not in the computed set: {r[1]} -> {r[2]} "
                                        f"({r[0][:12]} -> {r[3][:12]}, {r[5]})")
            else:
                print("row set: NOT recomputed (--rows-only); only the map's own rows are re-derived", file=out)
                taken: Set[str] = set()
                for r in mf.rows:
                    ch = changes.get(r[1])
                    if ch is None or (ch.pre_sha256, ch.pre_path, ch.path, ch.post_sha256, ch.kind) != (
                            r[0], r[1], r[2], r[3], r[5]):
                        set_problems.append(f"row {r[1]} -> {r[2]} does not describe a change of this release tree")
                    elif ch.pre_path not in taken:
                        taken.add(ch.pre_path)
                        rows.append(ch)
            results = certify(rows, inputs, site)
            for p in set_problems:
                print(f"FAIL map: {p}", file=out)
            bad = _print_results(results, out)
            ok = not set_problems and bad == 0
            print(f"{'CERTIFIED' if ok else 'REFUSED'}: {len(results) - bad} of {len(results)} rows re-derived and "
                  f"expanded on the site; {len(set_problems)} map problems", file=out)
        else:
            results = certify(rows, inputs, site)
            bad = _print_results(results, out)
            ok = bad == 0
            if ok:
                dest = args.out or os.path.join(repo, DEFAULT_MAP)
                with open(dest, "w", encoding="utf-8") as fh:
                    fh.write(render_map(rows, inputs))
                print(f"WROTE {dest}: {len(rows)} rows, every one re-derived and expanded on the site", file=out)
            else:
                print(f"REFUSED: {bad} of {len(results)} rows failed; no map written", file=out)
        if args.report:
            report = {
                "pre_ref": pre.commit, "rule_set": rs, "inputs": inputs.canonical(), "ok": ok,
                "rows": [{"pre_path": r.change.pre_path, "path": r.change.path, "pre_sha256": r.change.pre_sha256,
                          "post_sha256": r.change.post_sha256, "kind": r.change.kind,
                          "sources": sorted(r.change.sources), "problems": r.problems,
                          "substitutions": {"resolved": r.stats.checked, "in_comments": r.stats.comments,
                                            "declared": r.stats.declared, "cluster": r.stats.context},
                          "renamed": {"path_resolved": r.stats.renamed_resolved, "prose": r.stats.renamed_prose,
                                      "wording": r.stats.renamed_wording}}
                         for r in results],
                "omitted": omitted, "removed_or_not_followed": deleted,
            }
            with open(args.report, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=1, sort_keys=True)
        return 0 if ok else 1
    except InputsError as exc:
        print(f"REFUSED: {exc}", file=out)
        return 2
    finally:
        for t in trees:
            t.close()


if __name__ == "__main__":
    sys.exit(run())
