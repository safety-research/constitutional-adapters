#!/usr/bin/env python3
"""Apply declared text rules to config files: the one implementation the release and its scrub map share.

The release rewrites site-specific text in ``configs/``: storage paths become placeholders of ``sisteer.site``, and
free text that names a host, a person, a working session or the old repository name gets neutral wording. Each rewrite
is declared as a rule in a table kept outside the repository (its left-hand sides spell the text being removed). This
module applies such tables. The release applies them to the tree, and ``scripts/release/scrub_map.py`` applies the same
tables to each pre-release file when it re-derives the released bytes, either as a placeholder table it reads itself
or through :func:`rewrite`, named as a rename rule set's ``apply`` function.

TABLE (TSV, UTF-8; ``#`` lines and blank lines are ignored)::

    kind <TAB> match <TAB> replacement [<TAB> scope]

- ``kind`` is ``literal`` (every occurrence), ``word`` (every occurrence not inside an identifier) or ``regex``
  (Python ``re.sub`` with a replacement template). ``\\t``, ``\\n`` and ``\\\\`` are escapes in ``match`` and
  ``replacement``, as in the scrub map's own tables, so a table without a scope column reads the same in both.
- ``scope`` is optional: comma-separated globs over repository-relative paths, where ``*`` also crosses ``/``. A glob
  that starts with ``!`` excludes. The last glob that matches a path decides, as in ``.gitignore``, and a path that no
  glob matches is out of scope. A rule without a scope applies to every path.
- Rules apply in file order, each to the whole text, and a rule never spans lines unless its regex says so.

Applying the tables twice gives the same text as applying them once (``--check`` on a rewritten tree finds nothing),
because every replacement is outside every match. The release checks that property on its own output.

    config_text.py --table RULES.tsv [--table MORE.tsv ...] --apply  [--files-from LIST | PATH ...]
    config_text.py --table RULES.tsv [--table MORE.tsv ...] --check  [--files-from LIST | PATH ...]
    config_text.py --table RULES.tsv --digest

``--apply`` rewrites the files in place; ``--check`` exits 1 when any file would change and names each one. Paths are
relative to ``--root`` (default: the checkout this file is in). Stdlib only: the scrub map loads this file by path.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

KINDS = ("literal", "word", "regex")


class TableError(ValueError):
    """A malformed table row, named by file and line."""


@dataclass(frozen=True)
class Rule:
    kind: str
    match: str
    replacement: str
    scope: Tuple[str, ...]
    regex: "re.Pattern[str]"
    where: str

    def applies_to(self, path: str) -> bool:
        return in_scope(path, self.scope)

    def apply(self, text: str) -> str:
        if self.kind == "regex":
            return self.regex.sub(self.replacement, text)
        return self.regex.sub(lambda _m: self.replacement, text)


def in_scope(path: str, globs: Sequence[str]) -> bool:
    """Whether ``path`` is in scope: no globs means every path; otherwise the last glob that matches decides."""
    if not globs:
        return True
    verdict = False
    for g in globs:
        neg = g.startswith("!")
        if fnmatch.fnmatchcase(path, g[1:] if neg else g):
            verdict = not neg
    return verdict


def _unescape(field: str) -> str:
    out, i = [], 0
    while i < len(field):
        c = field[i]
        if c == "\\" and i + 1 < len(field):
            nxt = field[i + 1]
            out.append({"t": "\t", "n": "\n", "\\": "\\"}.get(nxt, "\\" + nxt))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def load_table(path: str) -> List[Rule]:
    """The rules of one table, in file order. A malformed row refuses with its file and line."""
    rules: List[Rule] = []
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cells = line.split("\t")
            where = f"{path}:{n}"
            if len(cells) not in (3, 4):
                raise TableError(f"{where}: expected kind<TAB>match<TAB>replacement[<TAB>scope], got {len(cells)} cells")
            kind, match, repl = cells[0].strip(), _unescape(cells[1]), _unescape(cells[2])
            scope = tuple(g.strip() for g in cells[3].split(",") if g.strip()) if len(cells) == 4 else ()
            if kind not in KINDS:
                raise TableError(f"{where}: kind {kind!r} is not literal, word or regex")
            if not match:
                raise TableError(f"{where}: an empty match")
            if kind == "literal":
                rx = re.compile(re.escape(match))
            elif kind == "word":
                rx = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(match) + r"(?![A-Za-z0-9_])")
            else:
                try:
                    rx = re.compile(match)
                except re.error as exc:
                    raise TableError(f"{where}: bad regex {match!r}: {exc}") from exc
            rules.append(Rule(kind, match, repl, scope, rx, where))
    return rules


def apply_rules(text: str, path: str, rules: Iterable[Rule]) -> str:
    """``text`` (the contents of ``path``) with every in-scope rule applied in order."""
    for rule in rules:
        if rule.applies_to(path):
            text = rule.apply(text)
    return text


_CACHE: dict = {}


def rewrite(text: str, *, path: str, table: str, **_files: str) -> str:
    """The scrub map's ``apply`` entry point: ``text`` of ``path`` with the rules of ``table`` applied."""
    key = (os.path.abspath(table), os.path.getmtime(table))
    if key not in _CACHE:
        _CACHE[key] = load_table(table)
    return apply_rules(text, path, _CACHE[key])


def digest(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _targets(root: str, paths: Sequence[str], files_from: Optional[str]) -> List[str]:
    rels = list(paths)
    if files_from:
        with open(files_from, encoding="utf-8") as fh:
            rels += [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    out = []
    for p in rels:
        ap = os.path.abspath(os.path.join(root, p)) if not os.path.isabs(p) else p
        out.append(os.path.relpath(ap, root).replace(os.sep, "/"))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", action="append", required=True, help="a rules table (TSV); repeat to apply in order")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    help="the checkout the paths are relative to (default: this file's checkout)")
    ap.add_argument("--files-from", help="a file listing one repository-relative path per line")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true", help="rewrite the files in place")
    mode.add_argument("--check", action="store_true", help="exit 1 if any file would change")
    mode.add_argument("--digest", action="store_true", help="print the sha256 of each table")
    ap.add_argument("paths", nargs="*")
    a = ap.parse_args(argv)
    try:
        tables = [load_table(t) for t in a.table]
    except (OSError, TableError) as exc:
        print(f"config_text: {exc}", file=sys.stderr)
        return 2
    if a.digest:
        for t in a.table:
            print(f"{digest(t)}  {t}")
        return 0
    rels = _targets(a.root, a.paths, a.files_from)
    changed = 0
    for rel in rels:
        full = os.path.join(a.root, rel)
        if not os.path.isfile(full):
            print(f"config_text: no such file: {rel}", file=sys.stderr)
            return 2
        with open(full, "rb") as fh:
            before = fh.read()
        try:
            text = before.decode("utf-8")
        except UnicodeDecodeError:
            continue
        after = text
        for rules in tables:
            after = apply_rules(after, rel, rules)
        if after == text:
            continue
        changed += 1
        if a.apply:
            with open(full, "wb") as fh:
                fh.write(after.encode("utf-8"))
        else:
            print(f"would change: {rel}")
    print(f"config_text: {'rewrote' if a.apply else 'would rewrite'} {changed} of {len(rels)} file(s)")
    return 1 if (a.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
