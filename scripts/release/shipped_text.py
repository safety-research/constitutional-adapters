#!/usr/bin/env python3
"""Rewrite the free text the release ships in data, under one declared field policy.

Three scopes carry free text:

* ``frame``: the committed paper frame (``paper/snapshot/snapshots/<id>/``), every TEXT cell of its SQLite and every string
  inside a JSON cell (the frame's JSON cells are canonical: sorted keys, compact separators, UTF-8 kept)
* ``transcripts``: ``configs/registry/rulings_r2_v*.jsonl``, ``roster_r2_v*.yaml`` and ``sources_r2_v1.yaml``
* ``seeds``: every other data file under ``configs/registry/`` except ``cluster_reads.yaml``

A *leaf* is one string at an address ``<scope>:<table or file>:<column or top-level key>:<JSON path>`` (list indices are written
``[*]`` when an address is matched against a pattern). Every leaf falls in exactly one class:

``identifier``  ids, keys, enum values, hashes, URIs and paths that code joins on: byte-exact apart from the site pass
``read``        a value a reader parses or displays (the reader inventory): byte-exact apart from the site pass; with the policy
                option ``verbatim``, byte-exact outright (a value matched against bytes recorded outside the release)
``sentinel``    a withheld required value: one fixed nonempty string per field
``drop``        session and batch ids: NULL, ``[]`` or the sentinel, as the policy declares
``prose``       every other leaf: the site pass, then the rules file, then the prose table

The *site pass* is the policy's ordered substitution list (site labels, hosts, worktree prefixes and other fixed site rules),
applied to every leaf so a value and every copy of it rewrite alike and joins hold. The policy, the rules file, the prose table
and the comment table are private inputs passed by path; this module names none of them and spells no site, host or session name.

Every writer keeps the file's own format: a JSONL record that does not change keeps its bytes, a YAML file is edited by splicing
only the scalars that change (comments, anchors, aliases and key order stay) and is re-parsed to prove that exactly the declared
values moved, and the frame keeps its tables, row counts, row order and every non-TEXT cell.

Subcommands: ``inventory`` (the reader inventory), ``census`` (every leaf with its class and whether the attribution shapes still
flag it), ``apply`` (rewrite the registry files in a tree and derive the new frame directory) and ``rewrite-file`` (one file,
for a caller that re-derives a postimage from its preimage).
"""
from __future__ import annotations

import argparse
import ast
import copy
import fnmatch
import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

SCOPES = ("frame", "transcripts", "seeds")
CLASSES = ("identifier", "read", "sentinel", "drop", "prose")
REGISTRY_DIR = os.path.join("configs", "registry")
SNAPSHOT_DIR = os.path.join("paper", "snapshot")
#: files under configs/registry that are not this module's: another track owns their data
EXCLUDED_FILES = frozenset({"cluster_reads.yaml"})
TRANSCRIPT_GLOBS = ("rulings_r2_v*.jsonl", "roster_r2_v*.yaml", "sources_r2_v*.yaml")
#: committed copies that must stay byte-identical to a registry file (the decode-policy export ships twice)
MIRRORS: Dict[str, Tuple[str, ...]] = {os.path.join("configs", "registry", "decode_policy.json"):
                                       (os.path.join("src", "sisteer", "registry", "decode_policy.json"),)}
DATA_SUFFIXES = (".yaml", ".yml", ".jsonl", ".json")
JSON_STYLE_FRAME = dict(sort_keys=True, separators=(",", ":"), ensure_ascii=False)
#: a JSON path component for a list index when the address is a pattern
ANY_INDEX = "[*]"

# The attribution shapes: decision ids, gate and lane tags, attributions to a person and process paths. Names (sessions,
# codenames, sites) are not shapes and come from the policy's private token list.
ATTRIBUTION_SHAPES: Dict[str, str] = {
    "decision_id": r"(?<![A-Za-z0-9_])(?:R\d+-)?D\d{1,5}(?:[-.]\d{1,3})?(?:[a-z])?(?![A-Za-z0-9_])",
    # a numbered decision written out in words: the word, optionally `no.`, `number` or `#`, then the number; whitespace or `#` must
    # come before the number, so a dotted key (`x.decisions.<n>`) and the word in plain prose stay clean
    "decision_spelled": r"(?i:\bdecisions?(?:\s+(?:no\.?|number|#)\s*|\s*#\s*|\s+)\d{1,4}\b)",
    "user_attribution": (r"(?i:\bthe user(?:'s|’s)?\b(?!\s+(?:turn|turns|message|messages|prompt|prompts|role|content|query|"
                         r"queries|request|requests|input|inputs|side|asks?|says?|said|wants?|writes?|types?|sends?|"
                         r"supplied|provided|gave|gives|query's|message's)\b)"
                         r"|\buser(?:'s)? (?:rule|ruling|rulings|call|order|decision|override|direction|instruction|authori[sz]ation|"
                         r"sign-off|confirmation|approval|choice|request)\b|\bper the user\b|\bby the user\b|\(user\b)"),
    "upper_user": r"(?<![A-Za-z_])USER(?![A-Za-z_])",
    "relayed": r"(?i:\brelayed by\b|\bsaid to\b|\brelayed to\b)",
    "gate_lane": (r"\b(?:the )?[A-Z]\d{1,3}[a-z]? (?:gate|lane|window|wave|smoke|battery|landing|review|merge|cutover)\b"
                  r"|(?<![A-Za-z0-9_])W\d{1,3}(?![A-Za-z0-9_])|registry-slurm-only"),
    "process_path": r"~/specs/|\.local/state/|~/reviews/|(?<![A-Za-z])romp(?![A-Za-z])|worktrees/|Co-Authored-By",
}


class ShippedTextError(RuntimeError):
    """A rewrite or a check cannot proceed; the message names the address or file."""


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------------------------------------
# addresses
# --------------------------------------------------------------------------------------------------------------------------------


def path_str(path: Sequence[object], pattern: bool = True) -> str:
    out = "$"
    for p in path:
        if isinstance(p, int):
            out += ANY_INDEX if pattern else f"[{p}]"
        else:
            out += "." + str(p)
    return out


def address(scope: str, unit: str, column: str, path: Sequence[object], pattern: bool = True) -> str:
    return f"{scope}:{unit}:{column}:{path_str(path, pattern)}"


def leaf_key(column: str, path: Sequence[object]) -> str:
    """The name a reader would use for the leaf: the last string component of the JSON path, else the column."""
    for p in reversed(path):
        if isinstance(p, str):
            return p
    return column


def unit_scope(name: str) -> Optional[str]:
    """``transcripts`` / ``seeds`` for a data file under configs/registry, None for anything this module does not own."""
    base = os.path.basename(name)
    if base in EXCLUDED_FILES or not base.endswith(DATA_SUFFIXES):
        return None
    if any(fnmatch.fnmatchcase(base, g) for g in TRANSCRIPT_GLOBS):
        return "transcripts"
    return "seeds"


# --------------------------------------------------------------------------------------------------------------------------------
# private inputs
# --------------------------------------------------------------------------------------------------------------------------------


@dataclass
class Rule:
    rid: str
    pattern: "re.Pattern[str]"
    replacement: str
    scopes: Tuple[str, ...]          # empty = every scope

    def applies(self, scope: str) -> bool:
        return not self.scopes or scope in self.scopes


def _unescape_cell(s: str) -> str:
    return s.replace("\\t", "\t").replace("\\n", "\n")


def load_rules(path: Optional[str]) -> List[Rule]:
    """The rules file (TSV: ``id``, ``pattern``, ``replacement``, optional ``scope``, optional ``note``); ``#`` lines and a
    header row starting ``id`` are skipped. Each pattern is a Python ``re`` pattern applied with ``re.sub`` in file order."""
    if not path:
        return []
    rules: List[Rule] = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cells = line.split("\t")
            if n == 1 and cells[0].strip().lower() in ("id", "rule_id"):
                continue
            if len(cells) < 3:
                raise ShippedTextError(f"{path}:{n}: a rule needs id, pattern and replacement (tab-separated)")
            scope_cell = cells[3].strip() if len(cells) > 3 else ""
            scopes = tuple(s.strip() for s in scope_cell.split(",") if s.strip() and s.strip() != "all")
            for s in scopes:
                if s not in SCOPES + ("comments",):
                    raise ShippedTextError(f"{path}:{n}: scope {s!r} is not one of {SCOPES + ('comments',)}")
            try:
                pat = re.compile(_unescape_cell(cells[1]))
            except re.error as exc:
                raise ShippedTextError(f"{path}:{n}: rule {cells[0]!r}: bad pattern ({exc})") from exc
            rules.append(Rule(cells[0].strip(), pat, _unescape_cell(cells[2]), scopes))
    return rules


_TABLE_OLD_KEYS = ("old_sha256", "sha256", "value_sha256", "key")
_TABLE_NEW_KEYS = ("new", "new_value", "rewrite")


@dataclass
class ProseTable:
    by_address: Dict[Tuple[str, str, str], str] = field(default_factory=dict)
    by_sha: Dict[str, str] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)
    rows: int = 0

    def lookup(self, scope: str, addr: str, *shas: str) -> Optional[str]:
        for i, h in enumerate(shas):
            if not h:
                continue
            if i == 0:
                got = self.by_address.get((scope, addr, h))
                if got is not None:
                    return got
            got = self.by_sha.get(h)
            if got is not None:
                return got
        return None


def load_tables(paths: Optional[Sequence[str]]) -> ProseTable:
    """Several prose tables read as one, in order; a later table may not give a value another rewrite."""
    t = ProseTable()
    for p in paths or ():
        load_table(p, into=t)
    return t


def load_table(path: Optional[str], into: Optional[ProseTable] = None) -> ProseTable:
    """The prose table (JSONL rows ``{scope, path, old_sha256, new}``; the alternative field names of the interface note are
    read too). Two rows giving one value two rewrites is recorded as a conflict, which the CLI refuses."""
    t = into if into is not None else ProseTable()
    if not path:
        return t
    with open(path, "r", encoding="utf-8") as fh:
        for n, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            old = next((row[k] for k in _TABLE_OLD_KEYS if row.get(k)), None)
            if old is None and isinstance(row.get("old"), str):
                old = sha256_text(row["old"])
            new = next((row[k] for k in _TABLE_NEW_KEYS if k in row), None)
            if old is None or not isinstance(new, str):
                raise ShippedTextError(f"{path}:{n}: a row needs an old value's sha256 and a new string")
            t.rows += 1
            scope = str(row.get("scope") or "")
            addr = str(row.get("path") or row.get("address") or "")
            if scope and addr:
                t.by_address[(scope, addr, old)] = new
            prev = t.by_sha.get(old)
            if prev is not None and prev != new:
                t.conflicts.append(f"{path}:{n}: sha {old[:12]} rewritten two ways")
                continue
            t.by_sha[old] = new
    return t


@dataclass
class CommentTable:
    """Hand rewrites of comment lines: (file basename, exact original line) -> new line."""
    rows: Dict[Tuple[str, str], str] = field(default_factory=dict)

    def get(self, unit: str, line: str) -> Optional[str]:
        return self.rows.get((unit, line))


def load_comment_table(path: Optional[str]) -> CommentTable:
    t = CommentTable()
    if not path:
        return t
    with open(path, "r", encoding="utf-8") as fh:
        for n, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not all(isinstance(row.get(k), str) for k in ("path", "old", "new")):
                raise ShippedTextError(f"{path}:{n}: a comment row needs path, old and new strings")
            t.rows[(os.path.basename(row["path"]), row["old"])] = row["new"]
    return t


#: a site substitution: (pattern, replacement, scopes it applies to; empty = every scope, ``comments`` included)
SiteSub = Tuple["re.Pattern[str]", str, Tuple[str, ...]]


@dataclass
class Policy:
    """The declared field policy (a private JSON file; see :func:`load_policy` for its keys)."""
    raw: Dict[str, Any]
    sentinel: str
    classes: List[Tuple[str, str, Any]]                   # (address glob, class, option)
    identifier_keys: List["re.Pattern[str]"]
    site_identifier: List[SiteSub]
    site_prose: List[SiteSub]
    names: List[str]
    frame_aliases: Dict[str, str]
    provenance: Dict[str, Any]
    exempt: Dict[Tuple[str, str], str]
    single_token_identifier: bool
    extra_shapes: Dict[str, str] = field(default_factory=dict)
    keys: List[str] = field(default_factory=list)          # address globs of JSON objects whose keys are identifier leaves
    key_map: List[Tuple[str, str, str]] = field(default_factory=list)   # (object address glob, old key, new key)

    def alias(self, scope: str, unit: str, column: str) -> Tuple[str, str]:
        if scope != "frame":
            return unit, column
        tgt = self.frame_aliases.get(f"{unit}.{column}")
        if tgt:
            u, c = tgt.split(".", 1)
            return u, c
        return unit, column


def policy_glob(g: str) -> str:
    """An address glob with its literal list index ``[*]`` escaped, so fnmatch reads it as the three characters it is and not as
    a character class."""
    return g.replace(ANY_INDEX, "[[][*]]")


def _compile_subs(items: Iterable[Sequence[Any]], where: str) -> List[SiteSub]:
    out = []
    for i, it in enumerate(items):
        if len(it) not in (2, 3):
            raise ShippedTextError(f"policy {where}[{i}]: expected [pattern, replacement] or [pattern, replacement, [scope, …]]")
        scopes = tuple(it[2]) if len(it) == 3 else ()
        for s in scopes:
            if s not in SCOPES + ("comments",):
                raise ShippedTextError(f"policy {where}[{i}]: scope {s!r} is not one of {SCOPES + ('comments',)}")
        out.append((re.compile(it[0]), it[1], scopes))
    return out


def load_policy(path: str) -> Policy:
    """Keys: ``sentinel`` (the withheld-value string), ``classes`` (ordered ``[address glob, class, option]``; the first match
    wins; ``option`` is the sentinel or drop value), ``identifier_keys`` (regexes on a leaf's key name), ``site_identifier`` and
    ``site_prose`` (ordered ``[pattern, replacement]`` substitutions, optionally ``[pattern, replacement, [scope, …]]``; the
    identifier list runs on every leaf after the prose list has run on multi-word prose leaves and comments), ``names`` (private
    name tokens the attribution scan also flags), ``extra_shapes`` (``{name: regex}`` shapes the scan adds), ``frame_aliases``
    (``view.column`` -> ``table.column`` whose class and rewrite it shares), ``provenance`` (``paths``, ``keep``, ``withhold``),
    ``exempt`` (``[address, value sha256, reason]`` rows the attribution scan skips), ``single_token_identifier`` (a value with
    no whitespace is an identifier unless a class rule says otherwise; default true), ``keys`` (address globs of JSON objects
    whose keys are themselves values a reader looks up, such as a spelling map: each key takes the identifier site map, so it
    stays equal to the same value spelled elsewhere), ``key_map`` (``[object address glob, old key, new key]`` rows: a key that
    is free text naming its group, renamed where the object sits; every other key is kept)."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    sentinel = raw.get("sentinel")
    if not isinstance(sentinel, str) or not sentinel.strip():
        raise ShippedTextError(f"{path}: `sentinel` must be a nonempty string")
    classes = []
    for i, row in enumerate(raw.get("classes") or []):
        if not (isinstance(row, list) and 2 <= len(row) <= 3 and row[1] in CLASSES):
            raise ShippedTextError(f"{path}: classes[{i}] must be [address glob, one of {CLASSES}, optional value]")
        classes.append((policy_glob(row[0]), row[1], row[2] if len(row) > 2 else None))
    exempt = {}
    for i, row in enumerate(raw.get("exempt") or []):
        if not (isinstance(row, list) and len(row) == 3 and all(isinstance(x, str) and x for x in row)):
            raise ShippedTextError(f"{path}: exempt[{i}] must be [address, value sha256, reason], each nonempty")
        exempt[(row[0], row[1])] = row[2]
    prov = dict(raw.get("provenance") or {})
    prov["paths"] = [policy_glob(g) for g in prov.get("paths") or []]
    return Policy(raw=raw, sentinel=sentinel, classes=classes,
                  identifier_keys=[re.compile(p) for p in raw.get("identifier_keys") or []],
                  site_identifier=_compile_subs(raw.get("site_identifier") or [], "site_identifier"),
                  site_prose=_compile_subs(raw.get("site_prose") or [], "site_prose"),
                  names=[str(n) for n in raw.get("names") or []],
                  frame_aliases=dict(raw.get("frame_aliases") or {}), provenance=prov, exempt=exempt,
                  single_token_identifier=bool(raw.get("single_token_identifier", True)),
                  extra_shapes={str(k): str(v) for k, v in (raw.get("extra_shapes") or {}).items()},
                  keys=[policy_glob(g) for g in raw.get("keys") or []],
                  key_map=[_key_map_row(path, i, r) for i, r in enumerate(raw.get("key_map") or [])])


def _key_map_row(path: str, i: int, row: Any) -> Tuple[str, str, str]:
    if not (isinstance(row, list) and len(row) == 3 and all(isinstance(x, str) and x for x in row)):
        raise ShippedTextError(f"{path}: key_map[{i}] must be [object address glob, old key, new key], each nonempty")
    return policy_glob(row[0]), row[1], row[2]


def key_address(scope: str, unit: str, column: Optional[str], path: Sequence[object], pattern: bool = True) -> str:
    """The address of a mapping KEY: its object's address followed by ``<key>`` (a record's own top-level keys have no column)."""
    if column is None:
        return f"{scope}:{unit}:<key>"
    return address(scope, unit, column, path, pattern) + "<key>"


def load_inventory(path: Optional[str]) -> Dict[str, frozenset]:
    """The reader inventory: ``{scope: [name, …]}`` (the names readers read in that scope)."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    names = raw.get("names", raw)
    return {s: frozenset(v if isinstance(v, list) else list(v)) for s, v in names.items() if s in SCOPES}


# --------------------------------------------------------------------------------------------------------------------------------
# the attribution scan (shapes plus the policy's private names)
# --------------------------------------------------------------------------------------------------------------------------------


class Scanner:
    def __init__(self, names: Sequence[str] = (), extra_shapes: Optional[Mapping[str, str]] = None):
        shapes = dict(ATTRIBUTION_SHAPES)
        shapes.update(extra_shapes or {})
        self.shapes = {k: re.compile(v) for k, v in shapes.items()}
        names = [n for n in names if n]
        self.names = re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
                                + r")(?![A-Za-z0-9_])", re.IGNORECASE) if names else None

        # one combined search decides most leaves; per-shape hits are computed only for a leaf it flags
        self.any_rx = re.compile("|".join(f"(?:{v})" for v in shapes.values()))

    def any(self, text: str) -> bool:
        return bool(self.any_rx.search(text) or (self.names is not None and self.names.search(text)))

    def hits(self, text: str) -> List[Tuple[str, str]]:
        if not self.any(text):
            return []
        out = []
        for k, rx in self.shapes.items():
            m = rx.search(text)
            if m:
                out.append((k, m.group(0)))
        if self.names is not None:
            m = self.names.search(text)
            if m:
                out.append(("name", m.group(0)))
        return out


# --------------------------------------------------------------------------------------------------------------------------------
# the rewrite context
# --------------------------------------------------------------------------------------------------------------------------------


DROP_KEY = object()       # a JSON key the policy removes


@dataclass
class Change:
    addr: str             # concrete address
    cls: str
    old: Optional[str]
    new: Any


class Context:
    """The policy, the inventory, the rules, the prose table and the comment table, plus the memo every writer shares (one
    value, one class key, one rewrite)."""

    def __init__(self, policy: Policy, inventory: Optional[Mapping[str, frozenset]] = None, rules: Sequence[Rule] = (),
                 table: Optional[ProseTable] = None, comments: Optional[CommentTable] = None):
        self.policy = policy
        self.inventory = dict(inventory or {})
        self.rules = list(rules)
        self.table = table or ProseTable()
        self.comments = comments or CommentTable()
        self.scanner = Scanner(policy.names, policy.extra_shapes)
        trig = [rx.pattern for rx, _, _ in policy.site_identifier + policy.site_prose] + [r.pattern.pattern for r in self.rules]
        # every pattern the passes could act on, as one search: a raw cell none of them (nor the scan) matches cannot change,
        # except through a table row keyed on its exact value or a class that rewrites regardless (sentinel, drop)
        self._trigger = re.compile("|".join(f"(?:{p})" for p in trig)) if trig else None
        self._always = ([g for g, cls, _ in policy.classes if cls in ("sentinel", "drop")] + list(policy.provenance.get("paths") or [])
                        + list(policy.keys) + [g for g, _, _ in policy.key_map])
        self._memo: Dict[Tuple[str, str, str], Any] = {}
        self._class_memo: Dict[str, Tuple[str, Any]] = {}
        self.table_hits: Dict[str, int] = {}
        self.unresolved: Dict[Tuple[str, str], Dict[str, Any]] = {}     # (pattern address, sha) -> info: prose still flagged
        self.read_changed: Dict[Tuple[str, str], Dict[str, Any]] = {}   # read leaves the site pass moved

    def may_change(self, scope: str, unit: str, column: str, raw: str) -> bool:
        """False only when no leaf of this raw value can change: no site pattern, rule or scan shape matches the raw text, its
        sha is not a table key, and no sentinel / drop / provenance rule addresses this column."""
        u, c = self.policy.alias(scope, unit, column)
        prefix = f"{scope}:{u}:{c}:"
        if any(fnmatch.fnmatchcase(prefix + "$", g) or g.startswith(prefix) or fnmatch.fnmatchcase(prefix, g.rsplit(":", 1)[0] + ":")
               for g in self._always):
            return True
        if self._trigger is not None and self._trigger.search(raw):
            return True
        if self.scanner.any(raw):
            return True
        return sha256_text(raw) in self.table.by_sha

    # ---- classes ----
    def classify(self, scope: str, unit: str, column: str, path: Sequence[object], value: Optional[str] = None) -> Tuple[str, Any]:
        u, c = self.policy.alias(scope, unit, column)
        pat = address(scope, u, c, path)
        key = pat if value is None or not self.policy.single_token_identifier else pat + ("|1" if _single_token(value) else "|0")
        hit = self._class_memo.get(key)
        if hit is not None:
            return hit
        res = self._classify(scope, u, c, path, pat, value)
        self._class_memo[key] = res
        return res

    def _classify(self, scope, unit, column, path, pat, value) -> Tuple[str, Any]:
        for glob, cls, opt in self.policy.classes:
            if fnmatch.fnmatchcase(pat, glob):
                return cls, opt
        name = leaf_key(column, path)
        if name in self.inventory.get(scope, ()):
            return "read", None
        if any(rx.search(name) for rx in self.policy.identifier_keys):
            return "identifier", None
        if value is not None and self.policy.single_token_identifier and _single_token(value):
            return "identifier", None
        return "prose", None

    # ---- passes ----
    def site_pass(self, s: str, prose: bool, scope: str = "") -> str:
        """The site substitutions for ``scope``: the prose forms first when the value is prose with more than one word (a
        single-token value is never prose grammar, so it takes the identifier forms only), then the identifier forms."""
        if prose and (scope == "comments" or any(ch.isspace() for ch in s)):
            for rx, rep, scopes in self.policy.site_prose:
                if not scopes or scope in scopes:
                    s = rx.sub(rep, s)
        for rx, rep, scopes in self.policy.site_identifier:
            if not scopes or scope in scopes:
                s = rx.sub(rep, s)
        return s

    def rule_pass(self, s: str, scope: str) -> str:
        for r in self.rules:
            if r.applies(scope):
                s = r.pattern.sub(r.replacement, s)
        return s

    def transform(self, scope: str, unit: str, column: str, path: Sequence[object], value: str) -> Tuple[str, Any]:
        """(class, new value) for one string leaf. The new value is a string, None (NULL) or :data:`DROP_KEY`."""
        cls, opt = self.classify(scope, unit, column, path, value)
        u, c = self.policy.alias(scope, unit, column)
        pat = address(scope, u, c, path)
        mk = (cls + "|" + pat if cls in ("sentinel", "drop", "prose") else cls, value, scope)
        if mk in self._memo:
            return cls, self._memo[mk]
        if cls == "sentinel":
            new: Any = self.policy.sentinel if opt is None else opt
        elif cls == "drop":
            new = None if opt is None else (DROP_KEY if opt == "__drop_key__" else opt)
        elif cls in ("identifier", "read"):
            # `verbatim`: a value a loader matches against bytes recorded outside the release (a producer's spelling, a path it
            # opens), so not even the site map may touch it
            new = value if opt == "verbatim" else self.site_pass(value, prose=False, scope=scope)
            if cls == "read" and new != value:
                self.read_changed.setdefault((pat, sha256_text(value)), {"address": pat, "old": value, "new": new})
        else:
            new = self._prose(scope, pat, value)
        self._memo[mk] = new
        return cls, new

    def _prose(self, scope: str, pat: str, value: str) -> str:
        new, _ = self.prose_text(scope, pat, value)
        return new

    def prose_text(self, scope: str, pat: str, value: str) -> Tuple[str, bool]:
        """(new text, whether a table row gave it) for one prose value: the table by the original value's sha, else the site
        pass, the table by that value's sha, the rules, the table by that value's sha; a value the scan still flags with no row
        is recorded in :attr:`unresolved`."""
        h0 = sha256_text(value)
        got = self.table.lookup(scope, pat, h0)
        if got is None:
            s1 = self.site_pass(value, prose=True, scope=scope)
            got = self.table.lookup(scope, pat, "", sha256_text(s1)) if s1 != value else None
            if got is None:
                s2 = self.rule_pass(s1, scope)
                got = self.table.lookup(scope, pat, "", sha256_text(s2)) if s2 != s1 else None
                if got is None:
                    if self.scanner.any(s2):
                        self.unresolved.setdefault((pat, h0), {"scope": scope, "address": pat, "old_sha256": h0, "old": value,
                                                                "after_rules": s2, "hits": self.scanner.hits(s2)})
                    return s2, False
        self.table_hits[h0] = self.table_hits.get(h0, 0) + 1
        return self.site_pass(got, prose=True, scope=scope), True

    # ---- JSON values ----
    def rewrite_json(self, scope: str, unit: str, column: str, obj: Any, path: Tuple[object, ...] = (),
                     changes: Optional[List[Change]] = None) -> Any:
        """A copy of ``obj`` with every string leaf transformed (dict keys are kept); the provenance rule reshapes the blocks
        the policy names."""
        if isinstance(obj, dict):
            if self._is_provenance(scope, unit, column, path):
                return self._provenance(scope, unit, column, obj, path, changes)
            out = {}
            for k, v in obj.items():
                nv = self.rewrite_json(scope, unit, column, v, path + (k,), changes)
                if nv is DROP_KEY:
                    if changes is not None:
                        changes.append(Change(address(scope, unit, column, path + (k,), False), "drop", _short(v), "<key dropped>"))
                    continue
                nk = self.key_for(scope, unit, column, path, k)
                if nk in out:
                    raise ShippedTextError(f"{address(scope, unit, column, path, False)}: two keys become {nk!r}")
                if nk != k and changes is not None:
                    changes.append(Change(key_address(scope, unit, column, path, False), "key", k, nk))
                out[nk] = nv
            return out
        if isinstance(obj, list):
            items = [self.rewrite_json(scope, unit, column, v, path + (i,), changes) for i, v in enumerate(obj)]
            return [v for v in items if v is not DROP_KEY]
        if isinstance(obj, str):
            cls, new = self.transform(scope, unit, column, path, obj)
            if new != obj and changes is not None:
                changes.append(Change(address(scope, unit, column, path, False), cls, obj, new))
            return new
        return obj

    def key_for(self, scope: str, unit: str, column: str, path: Sequence[object], k: Any) -> Any:
        """A key of the object at (scope, unit, column, path) as the release writes it: the identifier site map when the policy's
        ``keys`` names the object, then the policy's ``key_map`` row for it; any other key is kept."""
        if not isinstance(k, str) or not (self.policy.keys or self.policy.key_map):
            return k
        u, c = self.policy.alias(scope, unit, column)
        pat = address(scope, u, c, path)
        nk = k
        if any(fnmatch.fnmatchcase(pat, g) for g in self.policy.keys):
            nk = self.site_pass(k, prose=False, scope=scope)
        for g, old, new in self.policy.key_map:
            if old == k and fnmatch.fnmatchcase(pat, g):
                nk = new
                break
        return nk

    def _is_provenance(self, scope, unit, column, path) -> bool:
        u, c = self.policy.alias(scope, unit, column)
        pat = address(scope, u, c, path)
        return any(fnmatch.fnmatchcase(pat, g) for g in self.policy.provenance.get("paths") or [])

    def _provenance(self, scope, unit, column, block: Dict[str, Any], path, changes) -> Dict[str, Any]:
        keep = set(self.policy.provenance.get("keep") or [])
        withhold = set(self.policy.provenance.get("withhold") or [])
        out = {}
        for k, v in block.items():
            if k in withhold:
                out[k] = self.policy.sentinel
            elif k in keep:
                out[k] = v
            else:
                if changes is not None:
                    changes.append(Change(address(scope, unit, column, path + (k,), False), "sentinel", _short(v), "<key dropped>"))
                continue
            if out[k] != v and changes is not None:
                changes.append(Change(address(scope, unit, column, path + (k,), False), "sentinel", _short(v), out[k]))
        return out

    # ---- comments ----
    def rewrite_comment_line(self, unit: str, line: str, start: int) -> str:
        """A line whose comment starts at ``start``: the comment table's line, else the comment text through the site pass
        (prose form) and the rules (the code part before ``start`` is never touched)."""
        hand = self.comments.get(unit, line)
        if hand is not None:
            if hand[:start] != line[:start]:
                raise ShippedTextError(f"{unit}: a comment-table row changes text before the comment on line {line!r}")
            return hand
        body = line[start:]
        # the prose table may carry the whole line (keyed by the line's sha) or the comment alone (keyed by the comment's sha
        # before or after the site pass and the rules)
        got = self.table.by_sha.get(sha256_text(line))
        if got is not None:
            return got if got.startswith(line[:start]) and got[start:start + 1] == "#" else line[:start] + got
        s1 = self.site_pass(body, prose=True, scope="comments")
        s2 = self.rule_pass(s1, "comments")
        for cand in (body, s1, s2):
            got = self.table.by_sha.get(sha256_text(cand))
            if got is not None:
                return line[:start] + self.site_pass(got, prose=True, scope="comments")
        if self.scanner.any(s2):
            self.unresolved.setdefault((f"comments:{unit}", sha256_text(line)),
                                       {"scope": "comments", "address": f"comments:{unit}", "old_sha256": sha256_text(line),
                                        "old": line, "offset": start, "after_rules": line[:start] + s2, "hits": self.scanner.hits(s2)})
        return line[:start] + s2


def _single_token(v: str) -> bool:
    return bool(v) and not any(ch.isspace() for ch in v)


def _short(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)
    return s


# --------------------------------------------------------------------------------------------------------------------------------
# JSON in TEXT cells
# --------------------------------------------------------------------------------------------------------------------------------


def parse_json_cell(v: str) -> Tuple[bool, Any]:
    """(True, value) when the cell is a JSON object or array, else (False, None)."""
    if not v or v[0] not in "[{" or v[-1] not in "]}":
        return False, None
    try:
        return True, json.loads(v)
    except ValueError:
        return False, None


def dump_frame_json(obj: Any) -> str:
    return json.dumps(obj, **JSON_STYLE_FRAME)


def walk_leaves(obj: Any, path: Tuple[object, ...] = ()) -> Iterator[Tuple[Tuple[object, ...], str]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_leaves(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_leaves(v, path + (i,))
    elif isinstance(obj, str):
        yield path, obj


# --------------------------------------------------------------------------------------------------------------------------------
# JSONL and JSON files
# --------------------------------------------------------------------------------------------------------------------------------


def _jsonl_unit(fname: str, rec: Mapping[str, Any]) -> str:
    return fname + "#header" if rec.get("_header") else fname


def jsonl_records(text: str) -> List[Tuple[str, Any]]:
    out = []
    for line in text.split("\n"):
        if line.strip():
            out.append((line, json.loads(line)))
    return out


def rewrite_jsonl(ctx: Context, fname: str, text: str, scope: str, changes: Optional[List[Change]] = None) -> str:
    """Each record through the policy; a record that does not change keeps its bytes. A changed record is written as
    ``json.dumps(record, ensure_ascii=False)``, the files' own style (checked on every unchanged record)."""
    out_lines = []
    for line in text.split("\n"):
        if not line.strip():
            out_lines.append(line)
            continue
        rec = json.loads(line)
        if json.dumps(rec, ensure_ascii=False) != line:
            raise ShippedTextError(f"{fname}: a record is not in the file's JSON style; refusing to re-serialise it")
        unit = _jsonl_unit(fname, rec)
        new = {}
        for k, v in rec.items():
            nv = ctx.rewrite_json(scope, unit, k, v, (), changes)
            if nv is DROP_KEY:
                continue
            new[k] = nv
        out_lines.append(line if new == rec else json.dumps(new, ensure_ascii=False))
    return "\n".join(out_lines)


def rewrite_json_file(ctx: Context, fname: str, text: str, scope: str, changes: Optional[List[Change]] = None) -> str:
    obj = json.loads(text)
    trailing = "\n" if text.endswith("\n") else ""
    indent, ensure_ascii, sort_keys = _json_file_style(obj, text)
    new = {}
    if not isinstance(obj, dict):
        raise ShippedTextError(f"{fname}: expected a JSON object at the top level")
    for k, v in obj.items():
        nv = ctx.rewrite_json(scope, fname, k, v, (), changes)
        if nv is DROP_KEY:
            continue
        new[k] = nv
    if new == obj:
        return text
    pin = DERIVED_PINS.get(fname)
    if pin is not None:
        pin(new, changes)
    return json.dumps(new, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys) + trailing


def _decode_policy_pin(data: Dict[str, Any], changes: Optional[List[Change]]) -> None:
    """The decode-policy export pins its body in ``header.policy_set_sha256``; a rewritten body gets its pin recomputed with the
    exporter's own function, so the file stays a valid export."""
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
    if os.path.isdir(os.path.join(src, "sisteer")) and src not in sys.path:
        sys.path.insert(0, src)
    from sisteer.registry import decode_policy as DP
    header = data.get("header")
    if not isinstance(header, dict) or "policy_set_sha256" not in header:
        raise ShippedTextError("decode_policy.json: no header.policy_set_sha256 to re-pin")
    old, new = header["policy_set_sha256"], DP.policy_set_sha256(data)
    if new != old:
        header["policy_set_sha256"] = new
        if changes is not None:
            changes.append(Change("seeds:decode_policy.json:header:$.policy_set_sha256", "pin", old, new))


#: files whose body carries a hash of itself, re-pinned after a rewrite
DERIVED_PINS: Dict[str, Callable[[Dict[str, Any], Optional[List[Change]]], None]] = {"decode_policy.json": _decode_policy_pin}


def _json_file_style(obj: Any, text: str) -> Tuple[Optional[int], bool, bool]:
    """(indent, ensure_ascii, sort_keys) that reproduce the file; a file whose keys are sorted stays sorted when a key changes."""
    body = text[:-1] if text.endswith("\n") else text
    for indent in (1, 2, 4, None):
        for ea in (True, False):
            if json.dumps(obj, indent=indent, ensure_ascii=ea) == body:
                return indent, ea, json.dumps(obj, indent=indent, ensure_ascii=ea, sort_keys=True) == body
    raise ShippedTextError("a JSON file is in no known style (indent 1/2/4/none, ascii or not); refusing to re-serialise it")


# --------------------------------------------------------------------------------------------------------------------------------
# YAML: splice the scalars that change, keep everything else
# --------------------------------------------------------------------------------------------------------------------------------

_STR_TAG = "tag:yaml.org,2002:str"
_MERGE_TAG = "tag:yaml.org,2002:merge"


def _yaml():
    import yaml
    return yaml


def compose_yaml(text: str):
    yaml = _yaml()
    return yaml.compose(text, Loader=yaml.SafeLoader)


@dataclass
class YLeaf:
    node: Any
    path: Tuple[object, ...]
    flow: bool


def yaml_leaves(root) -> List[YLeaf]:
    """Every string scalar VALUE node with its path (merge keys walked at the merging level; an aliased node is listed once
    per path it is reachable by)."""
    yaml = _yaml()
    out: List[YLeaf] = []

    def walk(node, path, flow, stack):
        if id(node) in stack:
            return
        if isinstance(node, yaml.MappingNode):
            st = stack | {id(node)}
            for k, v in node.value:
                if k.tag == _MERGE_TAG:
                    subs = v.value if isinstance(v, yaml.SequenceNode) else [v]
                    for sub in subs:
                        walk(sub, path, flow or bool(sub.flow_style), st)
                    continue
                walk(v, path + (str(k.value),), flow or bool(node.flow_style), st)
        elif isinstance(node, yaml.SequenceNode):
            st = stack | {id(node)}
            for i, item in enumerate(node.value):
                walk(item, path + (i,), flow or bool(node.flow_style), st)
        elif isinstance(node, yaml.ScalarNode) and node.tag == _STR_TAG:
            out.append(YLeaf(node, path, flow))

    if root is not None:
        walk(root, (), False, frozenset())
    return out


def yaml_scalar_spans(root) -> List[Tuple[int, int]]:
    """(start, end) character spans of every scalar node, keys included (each node once)."""
    yaml = _yaml()
    spans: Dict[int, Tuple[int, int]] = {}

    def walk(node):
        if id(node) in spans:
            return
        if isinstance(node, yaml.ScalarNode):
            spans[id(node)] = (node.start_mark.index, node.end_mark.index)
            return
        spans[id(node)] = (-1, -1)
        if isinstance(node, yaml.MappingNode):
            for k, v in node.value:
                walk(k)
                walk(v)
        elif isinstance(node, yaml.SequenceNode):
            for item in node.value:
                walk(item)

    if root is not None:
        walk(root)
    return sorted(s for s in spans.values() if s[0] >= 0)


def _plain_safe(s: str, flow: bool) -> bool:
    yaml = _yaml()
    if not s or s != s.strip() or "\n" in s or "\t" in s:
        return False
    if s[0] in "-?:,[]{}#&*!|>'\"%@`" or ": " in s or " #" in s or s.endswith(":"):
        return False
    if flow and any(ch in s for ch in ",[]{}"):
        return False
    try:
        return yaml.safe_load("k: " + s) == {"k": s}
    except Exception:  # noqa: BLE001 -- anything the loader will not read back as this string is not plain-safe
        return False


def _double_quoted(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def _line_indent(text: str, idx: int) -> int:
    ls = text.rfind("\n", 0, idx) + 1
    n = 0
    while ls + n < len(text) and text[ls + n] == " ":
        n += 1
    return n


_YAML_PROPS = re.compile(r"(?:[&!][^\s]*[ \t]+)+")
#: the comment that may follow a block scalar's indicators on its header line
_HEADER_COMMENT = re.compile(r"[ \t]+#.*$")


def scalar_span(text: str, node) -> Tuple[int, int]:
    """(start, end) of a scalar's own text: the node's marks without its ``&anchor`` / ``!tag`` properties, which the marks
    include and which a rewrite must keep."""
    start, end = node.start_mark.index, node.end_mark.index
    m = _YAML_PROPS.match(text, start, end)
    return (m.end() if m else start), end


def emit_scalar(text: str, leaf: YLeaf, new: str) -> str:
    """The source text for ``new`` at ``leaf``'s position, in the original style where that style can carry it: a plain scalar
    stays plain when the value is plain-safe (re-wrapped to the original's continuation indent when the original spanned lines),
    a single-quoted one stays single-quoted, a block scalar stays a block scalar, anything else becomes double-quoted."""
    node = leaf.node
    style = node.style
    start, end = scalar_span(text, node)
    orig = text[start:end]
    if style in ("|", ">"):
        header_end = orig.find("\n")
        indicator = orig[:header_end] if header_end >= 0 else orig
        # the header line's comment (`>   # ...`) is a comment, not part of the value: the new header keeps it
        hm = _HEADER_COMMENT.search(indicator)
        header_comment = hm.group(0) if hm else ""
        body = orig[header_end + 1:] if header_end >= 0 else ""
        first = next((ln for ln in body.split("\n") if ln.strip()), "")
        ind = len(first) - len(first.lstrip(" ")) if first else _line_indent(text, start) + 2
        if style == "|" or "\n" in new.rstrip("\n"):
            chomp = "" if new.endswith("\n") and not new.endswith("\n\n") else ("-" if not new.endswith("\n") else "+")
            lines = new[:-1].split("\n") if new.endswith("\n") and chomp != "+" else new.split("\n")
            if chomp == "+":
                lines = new.split("\n")[:-1]
            out = "|" + chomp + header_comment + "\n" + "\n".join((" " * ind + ln) if ln else "" for ln in lines)
        else:
            # folded, one paragraph: re-fold to the original's width
            chomp = "-" if not new.endswith("\n") else ""
            words = new.rstrip("\n").split(" ")
            width = max(40, max((len(ln) for ln in body.split("\n")), default=80))
            out_lines, cur = [], ""
            for w in words:
                if cur and len(" " * ind + cur + " " + w) > width:
                    out_lines.append(" " * ind + cur)
                    cur = w
                else:
                    cur = w if not cur else cur + " " + w
            out_lines.append(" " * ind + cur)
            out = ">" + chomp + header_comment + "\n" + "\n".join(out_lines)
        # keep the original span's trailing line breaks (the block's end mark sits after them)
        tail = orig[len(orig.rstrip("\n")):]
        return out + tail if not out.endswith("\n") else out
    if style is None and _plain_safe(new, leaf.flow):
        if "\n" in orig and not leaf.flow:
            lines = orig.split("\n")
            cont = lines[1]
            ind = len(cont) - len(cont.lstrip(" "))
            width = max(len(ln) for ln in lines)
            col0 = start - (text.rfind("\n", 0, start) + 1)
            out_lines, cur, cur_start = [], "", col0
            for w in new.split(" "):
                if cur and cur_start + len(cur) + 1 + len(w) > max(width, col0 + 40):
                    out_lines.append(cur)
                    cur, cur_start = w, ind
                else:
                    cur = w if not cur else cur + " " + w
            out_lines.append(cur)
            return ("\n" + " " * ind).join(out_lines)
        return new
    if style == "'" and "\n" not in new:
        return "'" + new.replace("'", "''") + "'"
    return _double_quoted(new)


def _block_header_end(text: str, start: int, end: int) -> Optional[int]:
    """The offset of the line break that ends a block scalar's header line (``|`` or ``>``, its indicators and any comment) when
    the scalar spanning ``start``..``end`` is a block scalar, else None. A parser's scalar span starts at the indicator, so it
    covers the header line's comment, which is a YAML comment and not scalar text."""
    m = _YAML_PROPS.match(text, start, end)
    i = m.end() if m else start
    if i < end and text[i] in "|>":
        nl = text.find("\n", i, end)
        return nl if nl >= 0 else None
    return None


def find_comment_starts(text: str, spans: Sequence[Tuple[int, int]]) -> Dict[int, int]:
    """{line start offset: offset of the ``#`` that opens the line's comment}; a ``#`` inside a scalar never counts (a block
    scalar's header line is not inside it: its comment counts)."""
    covered = bytearray(len(text) + 1)
    for s, e in spans:
        h = _block_header_end(text, s, e)
        if h is not None:
            s = h
        covered[s:e] = b"\x01" * (e - s)
    out: Dict[int, int] = {}
    pos = 0
    for line in text.split("\n"):
        start = pos
        i = line.find("#")
        while i >= 0:
            p = start + i
            if not covered[p] and (i == 0 or line[i - 1] in " \t"):
                out[start] = p
                break
            i = line.find("#", i + 1)
        pos += len(line) + 1
    return out


def rewrite_yaml(ctx: Context, fname: str, text: str, scope: str, changes: Optional[List[Change]] = None,
                 comments: bool = True) -> str:
    """Splice every string scalar the policy rewrites, then (``comments``) every comment the comment pass rewrites, and prove the
    result: the re-composed document has the same shape, and every scalar equals its declared new value."""
    root = compose_yaml(text)
    leaves = yaml_leaves(root)
    # one new value per node (a node reachable by two paths must rewrite alike)
    by_node: Dict[int, Tuple[YLeaf, str, str, str]] = {}
    for lf in leaves:
        column = str(lf.path[0]) if lf.path else ""
        sub = lf.path[1:]
        cls, new = ctx.transform(scope, fname, column, sub, lf.node.value)
        if new is DROP_KEY or new is None:
            new = ctx.policy.sentinel if cls in ("sentinel", "drop") else lf.node.value
        prev = by_node.get(id(lf.node))
        addr = address(scope, fname, column, sub, False)
        if prev is not None and prev[1] != new:
            raise ShippedTextError(f"{fname}: one scalar is reachable as {prev[2]} and {addr} and the policy rewrites it two ways")
        if prev is None:
            by_node[id(lf.node)] = (lf, new, addr, cls)
    edits: List[Tuple[int, int, str]] = []
    for lf, new, addr, cls in by_node.values():
        if new == lf.node.value:
            continue
        rep = emit_scalar(text, lf, new)
        edits.append(scalar_span(text, lf.node) + (rep,))
        if changes is not None:
            changes.append(Change(addr, cls, lf.node.value, new))
    if comments:
        # comments are rewritten against the ORIGINAL lines (the comment table is keyed by them) and spliced with the scalars
        # in one pass: a comment overlaps a scalar only on a block scalar's header line, whose rewrite keeps the comment as it
        # was, so a header comment that changes while its scalar changes too is refused by the splice as an overlap
        edits += _yaml_comment_edits(ctx, fname, text, root, changes)
    out = _splice(text, edits)
    new_by_node = {nid: new for nid, (_, new, _, _) in by_node.items()}
    _verify_yaml(fname, text, out, {lf.path: new_by_node[id(lf.node)] for lf in leaves})
    return out


def _splice(text: str, edits: List[Tuple[int, int, str]]) -> str:
    edits.sort()
    for a, b in zip(edits, edits[1:]):
        if a[1] > b[0]:
            raise ShippedTextError(f"overlapping scalar edits at offsets {a[0]} and {b[0]}")
    parts, pos = [], 0
    for s, e, rep in edits:
        parts.append(text[pos:s])
        parts.append(rep)
        pos = e
    parts.append(text[pos:])
    return "".join(parts)


def _yaml_comment_edits(ctx: Context, fname: str, text: str, root, changes: Optional[List[Change]]) -> List[Tuple[int, int, str]]:
    """(start, end, new comment) for every comment the comment pass rewrites, at the original text's offsets."""
    starts = find_comment_starts(text, yaml_scalar_spans(root))
    edits = []
    pos = 0
    for n, line in enumerate(text.split("\n"), start=1):
        c = starts.get(pos)
        if c is not None:
            off = c - pos
            new = ctx.rewrite_comment_line(fname, line, off)
            if new != line:
                if new[:off] != line[:off]:
                    raise ShippedTextError(f"{fname}:{n}: a comment rewrite changes text before the comment")
                edits.append((c, pos + len(line), new[off:]))
                if changes is not None:
                    changes.append(Change(f"comments:{fname}:{n}", "comment", line, new))
        pos += len(line) + 1
    return edits


def _verify_yaml(fname: str, old_text: str, new_text: str, expect: Mapping[Tuple[object, ...], str]) -> None:
    """Re-compose both texts and walk them together: same shape, same keys and tags, every string scalar equal to its declared
    new value (``expect``, by path) or else to its old value. A merge key's value is walked where its anchor is defined."""
    yaml = _yaml()
    a, b = compose_yaml(old_text), compose_yaml(new_text)

    def walk(x, y, path):
        if type(x) is not type(y):
            raise ShippedTextError(f"{fname}: the rewritten YAML changed shape at {path_str(path, False)}")
        if isinstance(x, yaml.ScalarNode):
            want = expect.get(tuple(path), x.value) if x.tag == _STR_TAG else x.value
            if y.tag != x.tag or y.value != want:
                raise ShippedTextError(f"{fname}: at {path_str(path, False)} the rewritten YAML reads {y.value!r} ({y.tag}), "
                                       f"expected {want!r} ({x.tag})")
            return
        if isinstance(x, yaml.MappingNode):
            if len(x.value) != len(y.value):
                raise ShippedTextError(f"{fname}: a mapping changed size at {path_str(path, False)}")
            for (kx, vx), (ky, vy) in zip(x.value, y.value):
                if kx.value != ky.value or kx.tag != ky.tag:
                    raise ShippedTextError(f"{fname}: a key changed at {path_str(path, False)}: {kx.value!r} -> {ky.value!r}")
                if kx.tag == _MERGE_TAG:
                    continue
                walk(vx, vy, path + (str(kx.value),))
            return
        if len(x.value) != len(y.value):
            raise ShippedTextError(f"{fname}: a list changed length at {path_str(path, False)}")
        for i, (vx, vy) in enumerate(zip(x.value, y.value)):
            walk(vx, vy, path + (i,))

    walk(a, b, ())
    # and the constructed document equals the expected one (anchors, aliases and merges resolved)
    if yaml.safe_load(new_text) is None and yaml.safe_load(old_text) is not None:
        raise ShippedTextError(f"{fname}: the rewritten YAML is empty")


# --------------------------------------------------------------------------------------------------------------------------------
# one file, by kind
# --------------------------------------------------------------------------------------------------------------------------------


def rewrite_text(ctx: Context, rel_path: str, text: str, changes: Optional[List[Change]] = None) -> str:
    """The rewritten text of one configs/registry data file (``rel_path`` names it; only its basename is used)."""
    fname = os.path.basename(rel_path)
    scope = unit_scope(fname)
    if scope is None:
        raise ShippedTextError(f"{rel_path}: not a data file this policy covers")
    if fname.endswith(".jsonl"):
        return rewrite_jsonl(ctx, fname, text, scope, changes)
    if fname.endswith(".json"):
        return rewrite_json_file(ctx, fname, text, scope, changes)
    return rewrite_yaml(ctx, fname, text, scope, changes)


def file_leaves(rel_path: str, text: str) -> Iterator[Tuple[str, str, str, Tuple[object, ...], str]]:
    """(scope, unit, column, path, value) for every string leaf of one data file."""
    fname = os.path.basename(rel_path)
    scope = unit_scope(fname)
    if scope is None:
        return
    if fname.endswith(".jsonl"):
        for _, rec in jsonl_records(text):
            unit = _jsonl_unit(fname, rec)
            for k, v in rec.items():
                for p, s in walk_leaves(v):
                    yield scope, unit, k, p, s
    elif fname.endswith(".json"):
        obj = json.loads(text)
        for k, v in obj.items():
            for p, s in walk_leaves(v):
                yield scope, fname, k, p, s
    else:
        seen = set()
        for lf in yaml_leaves(compose_yaml(text)):
            key = (id(lf.node), lf.path)
            if key in seen:
                continue
            seen.add(key)
            col = str(lf.path[0]) if lf.path else ""
            yield scope, fname, col, lf.path[1:], lf.node.value


def walk_keys(obj: Any, path: Tuple[object, ...] = ()) -> Iterator[Tuple[Tuple[object, ...], str]]:
    """(the object's path, key) for every mapping key inside ``obj``, ``obj``'s own keys included."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, str(k)
            yield from walk_keys(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_keys(v, path + (i,))


def file_keys(rel_path: str, text: str) -> Iterator[Tuple[str, str, Optional[str], Tuple[object, ...], str]]:
    """(scope, unit, column, object path, key) for every mapping key of one data file; a record's own top-level keys come with
    column None (:func:`key_address`)."""
    fname = os.path.basename(rel_path)
    scope = unit_scope(fname)
    if scope is None:
        return
    if fname.endswith(".jsonl") or fname.endswith(".json"):
        recs = [rec for _, rec in jsonl_records(text)] if fname.endswith(".jsonl") else [json.loads(text)]
        for rec in recs:
            unit = _jsonl_unit(fname, rec) if fname.endswith(".jsonl") else fname
            for k, v in rec.items():
                yield scope, unit, None, (), str(k)
                for p, kk in walk_keys(v):
                    yield scope, unit, str(k), p, kk
        return
    yaml = _yaml()

    def walk(node, path, stack):
        if id(node) in stack:
            return
        if isinstance(node, yaml.MappingNode):
            st = stack | {id(node)}
            for k, v in node.value:
                if k.tag == _MERGE_TAG:
                    for sub in (v.value if isinstance(v, yaml.SequenceNode) else [v]):
                        yield from walk(sub, path, st)
                    continue
                if isinstance(k, yaml.ScalarNode):
                    yield path, str(k.value)
                yield from walk(v, path + (str(k.value),), st)
        elif isinstance(node, yaml.SequenceNode):
            st = stack | {id(node)}
            for i, item in enumerate(node.value):
                yield from walk(item, path + (i,), st)

    root = compose_yaml(text)
    if root is not None:
        for p, k in walk(root, (), frozenset()):
            yield (scope, fname, None, (), k) if not p else (scope, fname, str(p[0]), p[1:], k)


def yaml_comment_lines(rel_path: str, text: str) -> Iterator[Tuple[int, str, int]]:
    """(line number, line, comment offset within the line) for every YAML line that carries a comment."""
    fname = os.path.basename(rel_path)
    if not fname.endswith((".yaml", ".yml")):
        return
    starts = find_comment_starts(text, yaml_scalar_spans(compose_yaml(text)))
    pos = 0
    for n, line in enumerate(text.split("\n"), start=1):
        c = starts.get(pos)
        if c is not None:
            yield n, line, c - pos
        pos += len(line) + 1


def registry_files(tree: str) -> List[str]:
    """Every data file this policy covers under ``<tree>/configs/registry``, as repo-relative paths, sorted."""
    d = os.path.join(tree, REGISTRY_DIR)
    return sorted(os.path.join(REGISTRY_DIR, fn) for fn in os.listdir(d) if unit_scope(fn) is not None)


# --------------------------------------------------------------------------------------------------------------------------------
# declared pre-patches: content edits other tracks send for these files, applied to the pre-image before the policy
# --------------------------------------------------------------------------------------------------------------------------------

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def split_patch(patch_text: str) -> Dict[str, List[str]]:
    """{repo-relative path: [hunk lines …]} from a unified diff (``a/``/``b/`` prefixes stripped; header lines dropped)."""
    out: Dict[str, List[str]] = {}
    cur: Optional[str] = None
    for line in patch_text.split("\n"):
        if line.startswith("+++ "):
            tgt = line[4:].split("\t", 1)[0].strip()
            cur = tgt[2:] if tgt.startswith("b/") else tgt
            out.setdefault(cur, [])
        elif line.startswith("--- ") or line.startswith("diff ") or line.startswith("index "):
            continue
        elif cur is not None and (line.startswith("@@") or line[:1] in (" ", "+", "-", "\\")):
            out[cur].append(line)
    return out


def apply_patch_text(text: str, hunks: Sequence[str], where: str = "") -> str:
    """``text`` with one file's unified-diff hunks applied strictly (every context and removed line must match exactly at the
    hunk's position)."""
    lines = text.split("\n")
    out: List[str] = []
    pos = 0
    i = 0
    while i < len(hunks):
        m = _HUNK.match(hunks[i])
        if not m:
            raise ShippedTextError(f"{where}: not a hunk header: {hunks[i]!r}")
        start = int(m.group(1)) - 1 if int(m.group(2) or 1) else int(m.group(1))
        out.extend(lines[pos:start])
        pos = start
        i += 1
        while i < len(hunks) and not hunks[i].startswith("@@"):
            h = hunks[i]
            tag, body = h[:1], h[1:]
            if tag == "\\":
                i += 1
                continue
            if tag in (" ", "-"):
                if pos >= len(lines) or lines[pos] != body:
                    raise ShippedTextError(f"{where}: a pre-patch hunk does not apply at line {pos + 1}")
                if tag == " ":
                    out.append(body)
                pos += 1
            elif tag == "+":
                out.append(body)
            i += 1
    out.extend(lines[pos:])
    return "\n".join(out)


def prepatch_text(text: str, rel_path: str, patch_paths: Sequence[str]) -> str:
    """The pre-image of one file with every declared pre-patch that names it applied, in order."""
    for pp in patch_paths:
        with open(pp, "r", encoding="utf-8") as fh:
            hunks = split_patch(fh.read()).get(rel_path)
        if hunks:
            text = apply_patch_text(text, hunks, f"{pp}: {rel_path}")
    return text


_SCRUB_MAP_CTX: Dict[Tuple[Optional[str], ...], Context] = {}


def scrub_map_rewrite(text: str, *, path: str, table: Optional[str] = None, policy: str, inventory: Optional[str] = None,
                      rules: Optional[str] = None, comments: Optional[str] = None, prepatch: Optional[str] = None) -> str:
    """``scripts/release/scrub_map.py``'s entry point: the text of one configs/registry data file under the declared policy,
    rules, prose table and comment table, exactly as the release writes it, from its pre-release text. ``prepatch`` is the
    declared content edits other tracks sent for these files (one unified diff), applied to the pre-image first."""
    key = (table, policy, inventory, rules, comments)
    if key not in _SCRUB_MAP_CTX:
        _SCRUB_MAP_CTX[key] = Context(load_policy(policy), load_inventory(inventory), load_rules(rules), load_table(table),
                                      load_comment_table(comments))
    if prepatch:
        text = prepatch_text(text, path, [prepatch])
    return rewrite_text(_SCRUB_MAP_CTX[key], path, text)


# --------------------------------------------------------------------------------------------------------------------------------
# the frame
# --------------------------------------------------------------------------------------------------------------------------------


def frame_tables(conn: sqlite3.Connection) -> List[str]:
    return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def frame_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]


def frame_text_values(conn: sqlite3.Connection, table: str, column: str) -> Iterator[Tuple[str, int]]:
    """(distinct TEXT value, row count)."""
    q = f'SELECT "{column}", COUNT(*) FROM "{table}" WHERE typeof("{column}") = \'text\' GROUP BY "{column}"'
    yield from conn.execute(q)


def frame_cell_leaves(table: str, column: str, value: str) -> Iterator[Tuple[Tuple[object, ...], str]]:
    is_json, obj = parse_json_cell(value)
    if is_json:
        yield from walk_leaves(obj)
    else:
        yield (), value


def frame_cell_keys(table: str, column: str, value: str) -> Iterator[Tuple[Tuple[object, ...], str]]:
    """(object path, key) for every mapping key of a JSON cell; a plain cell has none."""
    is_json, obj = parse_json_cell(value)
    if is_json:
        yield from walk_keys(obj)


def rewrite_frame_cell(ctx: Context, table: str, column: str, value: str, changes: Optional[List[Change]] = None) -> Any:
    """The new cell for one TEXT value: a JSON cell is rewritten leaf by leaf and written canonically when anything moved; a
    plain cell is one leaf. Returns a string or None (NULL)."""
    is_json, obj = parse_json_cell(value)
    if is_json:
        new = ctx.rewrite_json("frame", table, column, obj, (), changes)
        if new == obj:
            return value
        return dump_frame_json(new)
    cls, new = ctx.transform("frame", table, column, (), value)
    if new is DROP_KEY:
        new = None
    if new != value and changes is not None:
        changes.append(Change(address("frame", table, column, (), False), cls, value, new))
    return new


def rewrite_frame(ctx: Context, src: str, dst: str, log: Optional[Callable[[str], None]] = None) -> Dict[str, Dict[str, int]]:
    """Copy the frame SQLite ``src`` to ``dst`` and rewrite every TEXT cell under the policy, in place (rows, row order and
    every non-TEXT cell are untouched), then VACUUM. Returns ``{table: {column: changed rows}}``."""
    shutil.copyfile(src, dst)
    conn = sqlite3.connect(dst, isolation_level=None)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    changed: Dict[str, Dict[str, int]] = {}
    try:
        for t in frame_tables(conn):
            for c in frame_columns(conn, t):
                mapping = []
                for v, _n in frame_text_values(conn, t, c):
                    if not ctx.may_change("frame", t, c, v):
                        continue
                    nv = rewrite_frame_cell(ctx, t, c, v)
                    if nv != v:
                        mapping.append((v, nv))
                if not mapping:
                    continue
                conn.execute("BEGIN")
                conn.execute("CREATE TEMP TABLE IF NOT EXISTS _st_map (old TEXT PRIMARY KEY, new TEXT)")
                conn.execute("DELETE FROM _st_map")
                conn.executemany("INSERT INTO _st_map (old, new) VALUES (?, ?)", mapping)
                cur = conn.execute(f'UPDATE "{t}" SET "{c}" = (SELECT new FROM temp._st_map WHERE old = "{t}"."{c}") '
                                   f'WHERE typeof("{c}") = \'text\' AND "{c}" IN (SELECT old FROM temp._st_map)')
                changed.setdefault(t, {})[c] = cur.rowcount
                conn.execute("COMMIT")
                if log:
                    log(f"frame {t}.{c}: {len(mapping)} distinct values, {cur.rowcount} rows")
        conn.execute("DROP TABLE IF EXISTS temp._st_map")
        conn.execute("VACUUM")
    finally:
        conn.close()
    return changed


def frame_leaves(conn: sqlite3.Connection) -> Iterator[Tuple[str, str, str, Tuple[object, ...], str, int]]:
    """(scope, table, column, path, value, rows) for every distinct string leaf of the frame."""
    for t in frame_tables(conn):
        for c in frame_columns(conn, t):
            for v, n in frame_text_values(conn, t, c):
                for p, s in frame_cell_leaves(t, c, v):
                    yield "frame", t, c, p, s, n


# --------------------------------------------------------------------------------------------------------------------------------
# the frame directory: Parquet, gz parts, MANIFEST
# --------------------------------------------------------------------------------------------------------------------------------


def _arrow_storage_class(arrow_type) -> str:
    import pyarrow as pa
    if pa.types.is_integer(arrow_type):
        return "INTEGER"
    if pa.types.is_floating(arrow_type):
        return "REAL"
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return "BLOB"
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "TEXT"
    raise ShippedTextError(f"no SQLite storage class for Parquet type {arrow_type}")


def derive_frame_dir(ctx: Context, old_dir: str, work: str, out_parent: str, built_utc: str, rules_sha256: str,
                     log: Optional[Callable[[str], None]] = None) -> Tuple[str, Dict[str, Any]]:
    """The re-derived frame directory under ``out_parent``: the old SQLite reassembled and rewritten (:func:`rewrite_frame`), each
    table's Parquet written from it with the old file's own column list and types (a table no cell of which moved keeps its old
    Parquet bytes), the SQLite gzip-split as the exporter does, and a MANIFEST that keeps every old key and adds
    ``derived_from``, ``release_rules_sha256`` and the release columns in ``scrubbed``. Returns (directory, manifest)."""
    sys.path.insert(0, os.path.join(_repo_root_of(old_dir), "src"))
    from sisteer.registry import snapshot as SNAP          # the exporter's own writers
    from sisteer.registry.identity import canonical_json
    import pyarrow.parquet as pq

    with open(os.path.join(old_dir, "MANIFEST.json"), "r", encoding="utf-8") as fh:
        old = json.load(fh)
    os.makedirs(work, exist_ok=True)
    old_sqlite = os.path.join(work, "old_" + SNAP.PAPER_FRAME_SQLITE)
    got = SNAP._reassemble_parts([os.path.join(old_dir, p) for p in old["sqlite_parts"]], old_sqlite)
    if got != old["sqlite_sha256"]:
        raise ShippedTextError(f"{old_dir}: the reassembled SQLite hashes {got}, MANIFEST says {old['sqlite_sha256']}")
    building = os.path.join(out_parent, ".building-release")
    if os.path.exists(building):
        shutil.rmtree(building)
    os.makedirs(building)
    new_sqlite = os.path.join(building, SNAP.PAPER_FRAME_SQLITE)
    changed = rewrite_frame(ctx, old_sqlite, new_sqlite, log=log)
    conn = sqlite3.connect(f"file:{new_sqlite}?mode=ro", uri=True)
    try:
        for t in old["parquet_tables"]:
            src_pq = os.path.join(old_dir, f"{t}.parquet")
            dst_pq = os.path.join(building, f"{t}.parquet")
            if t not in changed:
                shutil.copyfile(src_pq, dst_pq)
                continue
            schema = pq.read_schema(src_pq)
            cols = list(schema.names)
            types = {f.name: _arrow_storage_class(f.type) for f in schema}
            SNAP._write_parquet(conn.execute(f"SELECT {', '.join(cols)} FROM {t} ORDER BY rowid"), cols, types, dst_pq)
            if log:
                log(f"parquet {t}: rewritten")
    finally:
        conn.close()
    sqlite_sha = sha256_file(new_sqlite)
    sqlite_bytes = os.path.getsize(new_sqlite)
    parts = SNAP._gzip_split(new_sqlite, building, SNAP.PAPER_FRAME_SQLITE, int(old.get("sqlite_part_bytes") or SNAP.PAPER_FRAME_PART_BYTES))
    os.remove(new_sqlite)
    files: Dict[str, Dict[str, Any]] = {}
    for fn in sorted(os.listdir(building)):
        p = os.path.join(building, fn)
        files[fn] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    head = int(old["ingest_seq"])
    tables = dict(old["tables"])
    built_compact = built_utc.replace("-", "").replace(":", "").replace(".", "")[:15]
    core = canonical_json({"ingest_seq": head, "built_utc": built_utc, "files": files, "tables": tables, "scope": "paper_frame"})
    sha8 = hashlib.sha256(core.encode("utf-8")).hexdigest()[:8]
    snapshot_id = f"{head:012d}-{built_compact}-{sha8}"
    scrubbed = {t: sorted(set(old.get("scrubbed", {}).get(t, [])) | set(changed.get(t, {}))) for t in sorted(set(old.get("scrubbed", {})) | set(changed))}
    manifest = dict(old)
    manifest.update({"snapshot_id": snapshot_id, "built_utc": built_utc, "files": files, "sqlite_sha256": sqlite_sha,
                     "sqlite_bytes": sqlite_bytes, "sqlite_parts": parts, "scrubbed": scrubbed, "derived_from": old["snapshot_id"],
                     "release_rules_sha256": rules_sha256,
                     "release_changed_rows": {t: dict(sorted(cs.items())) for t, cs in sorted(changed.items())}})
    SNAP._refuse_manifest_leak(manifest, SNAP._private_tokens(), (out_parent, work, os.path.expanduser("~")))
    with open(os.path.join(building, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    final = os.path.join(out_parent, snapshot_id)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.rename(building, final)
    return final, manifest


def _repo_root_of(path: str) -> str:
    p = os.path.abspath(path)
    while p != os.sep:
        if os.path.isdir(os.path.join(p, "src", "sisteer")):
            return p
        p = os.path.dirname(p)
    raise ShippedTextError(f"{path}: not inside a checkout (no src/sisteer above it)")


def inputs_sha256(paths: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of ``{role: sha256 of the file, or a list of them}`` for the private inputs (absent ones as
    null)."""
    def one(v):
        if isinstance(v, (list, tuple)):
            return [sha256_file(x) for x in v]
        return sha256_file(v) if v else None
    m = {k: one(v) for k, v in sorted(paths.items())}
    return hashlib.sha256(json.dumps(m, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def input_paths(a) -> Dict[str, Any]:
    """The private inputs a run read, by role (what :func:`inputs_sha256` commits to)."""
    return {"policy": a.policy, "rules": getattr(a, "rules", None), "table": list(getattr(a, "table", None) or []),
            "comments": getattr(a, "comments", None), "inventory": getattr(a, "inventory", None)}


# --------------------------------------------------------------------------------------------------------------------------------
# the reader inventory
# --------------------------------------------------------------------------------------------------------------------------------

#: the readers whose names make a leaf `read`, by scope
READERS: Dict[str, Tuple[str, ...]] = {
    "frame": ("src/sisteer/registry/paper.py", "paper/notebooks/registry_nb.py", "paper/notebooks/make_registry_notebook.py"),
    "transcripts": ("src/sisteer/registry/adapters/roster.py",),
    "seeds": ("src/sisteer/registry/adapters/dimensions.py",),
}
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SQL_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SQL_HINT = re.compile(r"\bSELECT\b[\s\S]*\bFROM\b")


def _reader_names(source: str) -> Dict[str, List[str]]:
    """Names a reader reads, by how: subscript and ``.get`` keys, ``col:`` / ``extra:`` column-map entries, identifiers in SQL
    text and every identifier-shaped string constant (the over-approximation: a key spelled anywhere in a reader is kept)."""
    tree = ast.parse(source)
    found: Dict[str, set] = {"subscript": set(), "get": set(), "column_map": set(), "sql": set(), "constant": set()}
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str) and _NAME.match(sl.value):
                found["subscript"].add(sl.value)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("get", "pop", "setdefault"):
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                v = node.args[0].value
                if _NAME.match(v):
                    found["get"].add(v)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value
            m = re.match(r"^(?:col|extra):([A-Za-z0-9_.]+)$", v)
            if m:
                for part in m.group(1).split("."):
                    found["column_map"].add(part)
            elif _NAME.match(v):
                found["constant"].add(v)
            elif _SQL_HINT.search(v):
                found["sql"].update(w for w in _SQL_WORD.findall(v))
    return {k: sorted(v) for k, v in found.items()}


def build_inventory(tree: str) -> Dict[str, Any]:
    """The reader inventory of ``tree``: per scope, the names its readers read (static pass), plus the dimension identity
    fields (seeds) and the frame columns of the paper views (frame)."""
    sys.path.insert(0, os.path.join(tree, "src"))
    from sisteer.registry import schema_def as SD
    columns = {c for t in SD.TABLE_BY_NAME.values() for c in t.column_names}
    out: Dict[str, Any] = {"readers": {}, "names": {}}
    for scope, files in READERS.items():
        names: set = set()
        for rel in files:
            with open(os.path.join(tree, rel), "r", encoding="utf-8") as fh:
                got = _reader_names(fh.read())
            # words of SQL text count when they are schema columns or snake_case names (prose around a query does not)
            got["sql"] = sorted(w for w in got["sql"] if w in columns or ("_" in w and w.lower() == w))
            out["readers"][rel] = {k: len(v) for k, v in got.items()}
            for v in got.values():
                names.update(v)
        out["names"][scope] = names
    from sisteer.registry import identity as ID
    ident = {n for n, _ in ID.INSTRUMENT_ID_FIELDS} | set(ID.JUDGE_ID_FIELDS)
    out["names"]["seeds"] |= ident
    out["identity_fields"] = sorted(ident)
    out["names"] = {k: sorted(v) for k, v in out["names"].items()}
    return out


# --------------------------------------------------------------------------------------------------------------------------------
# census
# --------------------------------------------------------------------------------------------------------------------------------


def census(ctx: Context, tree: str, frame_sqlite: Optional[str], detail: Optional[Callable[[Dict[str, Any]], None]] = None
           ) -> Dict[str, Dict[str, Any]]:
    """Per address pattern and class: distinct values, rows, how many change, how many the attribution scan flags before and
    after. ``detail`` receives one row per distinct (address, value) that changes or is flagged (YAML comments as class
    ``comment``)."""
    agg: Dict[str, Dict[str, Any]] = {}
    seen = set()

    def leaf(scope, unit, column, path, value, n=1):
        u, c = ctx.policy.alias(scope, unit, column)
        pat = address(scope, u, c, path)
        key = (pat, value)
        if key in seen:
            return
        seen.add(key)
        cls, new = ctx.transform(scope, unit, column, path, value)
        new_s = new if isinstance(new, str) else None
        fo = ctx.scanner.any(value)
        fn = bool(new_s) and ctx.scanner.any(new_s)
        a = agg.setdefault(pat + " | " + cls, {"address": pat, "class": cls, "distinct": 0, "rows": 0, "changed": 0, "flag_old": 0,
                                                "flag_new": 0, "examples": []})
        a["distinct"] += 1
        a["rows"] += n
        a["changed"] += int(new != value)
        a["flag_old"] += int(fo)
        a["flag_new"] += int(fn)
        if (fn or new != value) and len(a["examples"]) < 3:
            a["examples"].append({"old": value[:300], "new": (new_s or repr(new))[:300]})
        if detail is not None and (fo or fn or new != value):
            detail({"scope": scope, "address": pat, "class": cls, "old_sha256": sha256_text(value), "old": value,
                    "new": new if new is not DROP_KEY else "<key dropped>", "changed": new != value, "rows": n,
                    "flag_old": [h[0] for h in ctx.scanner.hits(value)] if fo else [],
                    "flag_new": [h[0] for h in ctx.scanner.hits(new_s)] if fn else []})

    for rel in registry_files(tree):
        with open(os.path.join(tree, rel), "r", encoding="utf-8") as fh:
            text = fh.read()
        for scope, unit, column, path, value in file_leaves(rel, text):
            leaf(scope, unit, column, path, value)
        base = os.path.basename(rel)
        for n, line, off in yaml_comment_lines(rel, text):
            new = ctx.rewrite_comment_line(base, line, off)
            fo, fn = ctx.scanner.any(line[off:]), ctx.scanner.any(new[off:])
            a = agg.setdefault(f"comments:{base} | comment", {"address": f"comments:{base}", "class": "comment", "distinct": 0, "rows": 0,
                                                               "changed": 0, "flag_old": 0, "flag_new": 0, "examples": []})
            a["distinct"] += 1
            a["rows"] += 1
            a["changed"] += int(new != line)
            a["flag_old"] += int(fo)
            a["flag_new"] += int(fn)
            if detail is not None and (fo or fn or new != line):
                detail({"scope": "comments", "address": f"comments:{base}:{n}", "class": "comment", "old_sha256": sha256_text(line),
                        "old": line, "new": new, "changed": new != line, "rows": 1,
                        "flag_old": [h[0] for h in ctx.scanner.hits(line[off:])], "flag_new": [h[0] for h in ctx.scanner.hits(new[off:])]})
    if frame_sqlite:
        conn = sqlite3.connect(f"file:{frame_sqlite}?mode=ro", uri=True)
        try:
            for t in frame_tables(conn):
                for c in frame_columns(conn, t):
                    for v, n in frame_text_values(conn, t, c):
                        if not ctx.may_change("frame", t, c, v):
                            continue
                        for p, sv in frame_cell_leaves(t, c, v):
                            leaf("frame", t, c, p, sv, n)
        finally:
            conn.close()
    return agg


# --------------------------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------------------------


def _ctx_from_args(a) -> Context:
    return Context(load_policy(a.policy), load_inventory(getattr(a, "inventory", None)), load_rules(getattr(a, "rules", None)),
                   load_tables(getattr(a, "table", None)), load_comment_table(getattr(a, "comments", None)))


def _add_inputs(p: argparse.ArgumentParser) -> None:
    p.add_argument("--policy", required=True, help="the field policy (JSON)")
    p.add_argument("--inventory", help="the reader inventory (JSON, from the `inventory` subcommand)")
    p.add_argument("--rules", help="the rules file (TSV)")
    p.add_argument("--table", action="append", default=[], help="a prose table (JSONL); repeat for several, read as one")
    p.add_argument("--comments", help="the comment table (JSONL: path, old, new)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_inv = sub.add_parser("inventory", help="write the reader inventory of a checkout")
    p_inv.add_argument("--tree", required=True)
    p_inv.add_argument("--out", required=True)
    p_cen = sub.add_parser("census", help="every leaf with its class and new value (JSONL)")
    _add_inputs(p_cen)
    p_cen.add_argument("--tree", required=True)
    p_cen.add_argument("--frame-sqlite", help="a reassembled frame SQLite (read-only)")
    p_cen.add_argument("--out", required=True)
    p_app = sub.add_parser("apply", help="rewrite configs/registry in a tree and derive the new frame directory")
    _add_inputs(p_app)
    p_app.add_argument("--tree", required=True, help="the checkout to rewrite (its configs/registry and paper/snapshot)")
    p_app.add_argument("--work", required=True, help="scratch directory for the reassembled SQLite")
    p_app.add_argument("--built-utc", required=True, help="the new frame's build time, YYYY-MM-DDTHH:MM:SS.mmmZ")
    p_app.add_argument("--skip-frame", action="store_true")
    p_app.add_argument("--skip-files", action="store_true")
    p_app.add_argument("--changes", help="write every change as JSONL here")
    p_app.add_argument("--pre-patch", action="append", default=[],
                       help="a unified diff of declared content edits, applied to the files before the policy (repeat)")
    p_one = sub.add_parser("rewrite-file", help="rewrite one configs/registry data file")
    _add_inputs(p_one)
    p_one.add_argument("--path", required=True, help="the file's repo-relative path (its basename selects the scope)")
    p_one.add_argument("--in", dest="inp", required=True)
    p_one.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "inventory":
        inv = build_inventory(a.tree)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(inv, fh, indent=1, sort_keys=True)
        print(json.dumps({s: len(v) for s, v in inv["names"].items()}))
        return 0
    ctx = _ctx_from_args(a)
    if ctx.table.conflicts:
        raise ShippedTextError("the prose table rewrites a value two ways: " + "; ".join(ctx.table.conflicts[:5]))
    if a.cmd == "census":
        n = [0]
        with open(a.out, "w", encoding="utf-8") as fh:
            def detail(r):
                fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
                n[0] += 1
            agg = census(ctx, a.tree, a.frame_sqlite, detail)
        with open(a.out + ".summary.json", "w", encoding="utf-8") as fh:
            json.dump(sorted(agg.values(), key=lambda r: (-r["flag_new"], -r["changed"], r["address"])), fh, indent=1, ensure_ascii=False)
        print(f"census: {n[0]} detail rows -> {a.out}; {len(agg)} address patterns -> {a.out}.summary.json")
        if ctx.unresolved:
            with open(a.out + ".unresolved.jsonl", "w", encoding="utf-8") as fh:
                for r in ctx.unresolved.values():
                    fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        return 0
    if a.cmd == "rewrite-file":
        with open(a.inp, "r", encoding="utf-8") as fh:
            text = fh.read()
        out = rewrite_text(ctx, a.path, text)
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(out)
        return 0
    # apply
    changes: List[Change] = []
    if not a.skip_files:
        for rel in registry_files(a.tree):
            p = os.path.join(a.tree, rel)
            with open(p, "r", encoding="utf-8") as fh:
                text = fh.read()
            patched = prepatch_text(text, rel, a.pre_patch)
            new = rewrite_text(ctx, rel, patched, changes)
            if new != text:
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(new)
                print(f"rewrote {rel}")
            for mirror in MIRRORS.get(rel, ()):
                mp = os.path.join(a.tree, mirror)
                with open(mp, "r", encoding="utf-8") as fh:
                    if fh.read() != text:
                        raise ShippedTextError(f"{mirror} is not byte-identical to {rel}; refusing to rewrite either")
                if new != text:
                    with open(mp, "w", encoding="utf-8") as fh:
                        fh.write(new)
                    print(f"rewrote {mirror} (the byte-identical copy of {rel})")
    if not a.skip_frame:
        snap_root = os.path.join(a.tree, SNAPSHOT_DIR)
        with open(os.path.join(snap_root, "CURRENT"), "r", encoding="utf-8") as fh:
            old_id = fh.read().strip()
        old_dir = os.path.join(snap_root, "snapshots", old_id)
        rules_sha = inputs_sha256(input_paths(a))
        final, manifest = derive_frame_dir(ctx, old_dir, a.work, os.path.join(snap_root, "snapshots"), a.built_utc, rules_sha,
                                           log=print)
        shutil.rmtree(old_dir)
        with open(os.path.join(snap_root, "CURRENT"), "w", encoding="utf-8") as fh:
            fh.write(manifest["snapshot_id"] + "\n")
        print(f"frame: {old_id} -> {manifest['snapshot_id']}")
    if a.changes:
        with open(a.changes, "w", encoding="utf-8") as fh:
            for c in changes:
                fh.write(json.dumps({"address": c.addr, "class": c.cls, "old": c.old, "new": c.new}, ensure_ascii=False) + "\n")
    if ctx.unresolved:
        print(f"WARNING: {len(ctx.unresolved)} prose leaves are still flagged after the rules and the table", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
