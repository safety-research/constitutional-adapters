#!/usr/bin/env python3
"""Rename site labels across a git work tree, from a rule table passed on the command line.

The rule table names the old words, so it is kept outside the repository and handed to this tool as an argument. The
tool rewrites every tracked text file, renames every tracked path that carries an old word (``git mv``), stages what it
changed, and proves that a second pass would change nothing. A dry run writes a report and touches nothing.

Usage::

    rename_sites.py --rules TABLE --dry-run [--report FILE] [--json FILE] [--root DIR] [PATHSPEC ...]
    rename_sites.py --rules TABLE --apply   [--report FILE] [--json FILE] [--root DIR] [PATHSPEC ...]
    rename_sites.py --rules TABLE --check   [--root DIR] [PATHSPEC ...]

``--check`` exits 1 when a pass would change any file or path, so running it on the tool's own output is the
idempotence check. ``--apply`` runs that check itself after writing and exits 3 if it fails.

Rule table (tab-separated fields; blank lines and lines starting with ``#`` are ignored)::

    word     OLD   SNAKE  UPPER  CAMEL  PROSE  ARTICLE   one site word and its spellings (ARTICLE is a word or -)
    literal  OLD   NEW                                    an exact, case-sensitive substring, replaced first
    keep     GLOB  REGEX                                  spans that are never rewritten (multi-line regex flags)
    mode     GLOB  full|identifiers|skip                  per-path mode; the first matching glob wins, default full

How one occurrence of a word is spelled after the rename depends on its case and on what touches it:

- lowercase: SNAKE. Inside a long option (a token that starts with ``--``) the underscores become hyphens, so argparse
  and any ``name.replace('_', '-')`` agree with it. A lowercase word that stands alone after a determiner becomes PROSE
  when the row has an ARTICLE, so "copy it to the <word>." reads as prose.
- UPPERCASE: UPPER, with hyphens for underscores when a hyphen touches it. In running text (spaces or sentence
  punctuation on both sides, or a possessive ``'s``) it is emphasis, so it becomes PROSE in capitals, after ARTICLE in
  capitals unless a determiner already precedes it.
- capitalised or camel case: CAMEL when a letter, digit or underscore touches it (an identifier); PROSE with hyphens
  for spaces when a hyphen touches it (a compound adjective); otherwise PROSE, preceded by ARTICLE unless a determiner
  already precedes it, and capitalised when it opens a comment line, a list item or a sentence (a sentence also opens
  mid-line after ``.``, ``!`` or ``?`` and a space, unless the full stop closes an abbreviation or an initial).

Mode ``identifiers`` rewrites only identifier spellings (glued, quoted, option or UPPERCASE forms) and leaves prose
alone. Mode ``skip`` leaves the file untouched. A ``keep`` span is left untouched in every mode.

Every output spelling is checked when the table loads: none may contain an old word or a literal's old text, which is
what makes a second pass a no-op.
"""
from __future__ import annotations

import argparse
import bisect
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

MODES = ("full", "identifiers", "skip")
QUOTES = "\"'`"
BINARY_SUFFIXES = (".gz", ".zst", ".parquet", ".pt", ".pkl", ".safetensors", ".eval", ".png", ".jpg", ".jpeg", ".gif",
                   ".pdf", ".sqlite", ".db", ".zip", ".tar", ".npy", ".npz", ".bin", ".woff", ".woff2", ".ico")

DETERMINERS = frozenset(
    "the a an this that these those each every any no our its your their my one same another either neither whose "
    "which".split())
# Words that end a noun phrase: with one of these just before the site word, an earlier determiner does not reach it.
PHRASE_BREAKS = frozenset(
    "on in at to from of for by with via onto into and or nor than between off under over through per vs versus is are "
    "was were be been being as if when while where whether not but so then also only both reads writes runs uses needs "
    "holds keeps has had have sees calls copies pulls pushes names means makes gets takes lands reaches leaves sends "
    "stays lives".split())

# A full stop after one of these does not end a sentence.
ABBREVIATIONS = frozenset("e.g i.e etc vs cf al approx resp viz ca no nos fig figs sec eq eqs ch vol pp".split())

_SENTENCE_STOP = re.compile(r"([.!?])[)\]\"'`*]*[ \t]+\Z")
_WORD_BEFORE_STOP = re.compile(r"[A-Za-z0-9_.'-]*\Z")
_LEADER = r"(?:#+|//+|--+|\*+|>+|;+|[rRbBuU]?\"\"\"|[rRbBuU]?''')"
_LINE_PREFIX_ONLY = re.compile(rf"[ \t]*(?:{_LEADER}[ \t]*)?(?:(?:[-*+]|\d+[.)])[ \t]+)?\Z")
_LIST_MARKER_END = re.compile(r"(?:^|[ \t])(?:[-*+]|\d+[.)])[ \t]+\Z")
_LEADER_KIND = re.compile(rf"[ \t]*({_LEADER})?")
_STRIP_LEADER = re.compile(rf"^[ \t]*{_LEADER}?[ \t]*")
_NEWLINE_LEADER = re.compile(rf"\n[ \t]*{_LEADER}?[ \t]*")
_GAP_END = re.compile(r"[ \t\r\n`*\"']*\Z")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_TOKEN_CHARS = re.compile(r"[A-Za-z0-9_.$/-]")


class RuleError(ValueError):
    """The rule table is malformed or would not be idempotent."""


@dataclass(frozen=True)
class Word:
    old: str
    snake: str
    upper: str
    camel: str
    prose: str
    article: str  # "" for none

    def spellings(self) -> tuple[str, ...]:
        with_article = f"{self.article} {self.prose}".strip()
        return (self.snake, self.upper, self.camel, self.prose, self.snake.replace("_", "-"),
                self.upper.replace("_", "-"), self.prose.replace(" ", "-"), with_article, self.prose.upper(),
                with_article.upper())


@dataclass
class Hit:
    path: str
    line: int
    kind: str      # literal, snake, option, prose-lower, upper, upper-hyphen, upper-prose, camel, compound, prose
    before: str
    after: str
    col: int = -1  # offset of the occurrence within its line


@dataclass
class FileResult:
    path: str
    old_text: str | None = None
    new_text: str | None = None
    hits: list[Hit] = field(default_factory=list)
    skipped: str = ""          # why the file was not read: binary, undecodable, symlink, mode-skip
    residue: list[tuple[int, str]] = field(default_factory=list)   # (line, text) old words left after the rewrite


class RuleTable:
    """A loaded rule table. ``rewrite_text`` / ``rewrite_bytes`` / ``rewrite_path`` are the whole public interface."""

    def __init__(self, words: list[Word], literals: list[tuple[str, str]], keeps: list[tuple[str, re.Pattern]],
                 modes: list[tuple[str, str]], digest: str = ""):
        self.words = {w.old: w for w in words}
        self.literals = dict(literals)
        self.keeps = keeps
        self.modes = modes
        self.digest = digest
        self._validate()
        alternatives = [f"(?P<lit{i}>{re.escape(old)})" for i, old in enumerate(self.literals)]
        if self.words:
            ordered = sorted(self.words, key=len, reverse=True)
            alternatives.append("(?P<word>(?i:" + "|".join(re.escape(w) for w in ordered) + "))")
        self._literal_by_group = {f"lit{i}": old for i, old in enumerate(self.literals)}
        self._pattern = re.compile("|".join(alternatives)) if alternatives else None
        self._any_old = re.compile("(?i:" + "|".join(re.escape(w) for w in sorted(self.words, key=len, reverse=True))
                                   + ")") if self.words else None

    # ---------------------------------------------------------------- validation

    def _validate(self) -> None:
        olds = list(self.words)
        for w in self.words.values():
            if not re.fullmatch(r"[a-z][a-z0-9]*", w.old):
                raise RuleError(f"word {w.old!r}: the old word must be lowercase letters and digits")
            if not re.fullmatch(r"[a-z][a-z0-9_]*", w.snake):
                raise RuleError(f"word {w.old!r}: snake form {w.snake!r} must be lowercase letters, digits and _")
            if w.upper != w.snake.upper():
                raise RuleError(f"word {w.old!r}: UPPER form {w.upper!r} must be the snake form in capitals")
            if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", w.camel):
                raise RuleError(f"word {w.old!r}: camel form {w.camel!r} must start with a capital and hold no separator")
            if not w.prose or w.prose != w.prose.strip():
                raise RuleError(f"word {w.old!r}: prose form must be nonempty with no outer spaces")
            for spelling in w.spellings():
                for old in olds:
                    if old in spelling.lower():
                        raise RuleError(f"word {w.old!r}: spelling {spelling!r} contains the old word {old!r}")
                for lit in self.literals:
                    if lit in spelling:
                        raise RuleError(f"word {w.old!r}: spelling {spelling!r} contains the literal {lit!r}")
        for old, new in self.literals.items():
            if not old:
                raise RuleError("literal: the old text is empty")
            for w in olds:
                if w in new.lower():
                    raise RuleError(f"literal {old!r}: replacement {new!r} contains the old word {w!r}")
            for lit in self.literals:
                if lit in new:
                    raise RuleError(f"literal {old!r}: replacement {new!r} contains the literal {lit!r}")
        for glob, mode in self.modes:
            if mode not in MODES:
                raise RuleError(f"mode {glob!r}: {mode!r} is not one of {', '.join(MODES)}")

    # ---------------------------------------------------------------- per path

    def mode_of(self, path: str) -> str:
        for glob, mode in self.modes:
            if fnmatch.fnmatchcase(path, glob):
                return mode
        return "full"

    def _keep_spans(self, text: str, path: str) -> list[tuple[int, int]]:
        spans = []
        for glob, rx in self.keeps:
            if fnmatch.fnmatchcase(path, glob):
                spans.extend((m.start(), m.end()) for m in rx.finditer(text) if m.end() > m.start())
        spans.sort()
        return spans

    # ---------------------------------------------------------------- the rewrite

    def rewrite_text(self, text: str, path: str) -> str:
        """The rewritten text of the file at repo-relative ``path`` (its mode and keep spans apply)."""
        return self._rewrite(text, path)[0]

    def rewrite_bytes(self, data: bytes, path: str) -> bytes:
        """Rewritten bytes; binary, undecodable and skipped files come back unchanged."""
        if self.mode_of(path) == "skip" or _looks_binary(data, path):
            return data
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return data
        return self._rewrite(text, path)[0].encode("utf-8")

    def rewrite_path(self, path: str) -> str:
        """The path after the rename: every old word becomes an identifier spelling (never prose)."""
        if self.mode_of(path) == "skip" or self._pattern is None:
            return path
        return self._rewrite(path, path, path_mode=True)[0]

    def _rewrite(self, text: str, path: str, path_mode: bool = False) -> tuple[str, list[tuple[int, str, str, str]]]:
        """Return (new text, [(offset, kind, before, after)]) for one text."""
        if self._pattern is None or not self._may_match(text):
            return text, []
        mode = "full" if path_mode else self.mode_of(path)
        if mode == "skip":
            return text, []
        spans = [] if path_mode else self._keep_spans(text, path)
        out: list[str] = []
        hits: list[tuple[int, str, str, str]] = []
        last = 0
        span_i = 0
        for m in self._pattern.finditer(text):
            s, e = m.start(), m.end()
            while span_i < len(spans) and spans[span_i][1] <= s:
                span_i += 1
            if span_i < len(spans) and spans[span_i][0] < e and s < spans[span_i][1]:
                continue
            group = m.lastgroup
            if group != "word":
                old = self._literal_by_group[group]
                new, kind = self.literals[old], "literal"
                if mode == "identifiers" and (" " in new or old[:1].isupper()):
                    continue
            else:
                new, kind = self._word(text, s, e, path_mode)
                if mode == "identifiers" and not self._identifier_like(text, s, e, kind):
                    continue
            if new == text[s:e]:
                continue
            out.append(text[last:s])
            out.append(new)
            hits.append((s, kind, text[s:e], new))
            last = e
        out.append(text[last:])
        return "".join(out), hits

    def _may_match(self, text: str) -> bool:
        """A fast prefilter: most files hold no old word and no literal."""
        low = text.lower()
        return any(w in low for w in self.words) or any(lit in text for lit in self.literals)

    def _word(self, text: str, s: int, e: int, path_mode: bool) -> tuple[str, str]:
        found = text[s:e]
        w = self.words[found.lower()]
        p = text[s - 1] if s > 0 else ""
        n = text[e] if e < len(text) else ""
        if found.islower():
            if _token_prefix(text, s).startswith("--"):
                return w.snake.replace("_", "-"), "option"
            if (not path_mode and w.article and p in " \t\n" and not _identifier_next(text, e)
                    and _determined(_previous_words(text, s))):
                return w.prose, "prose-lower"
            return w.snake, "snake"
        if found.isupper():
            if p == "-" or n == "-":
                return w.upper.replace("_", "-"), "upper-hyphen"
            if not path_mode and _caps_prose(text, s, e):
                prose = w.prose.upper()
                if w.article and not _determined(_previous_words(text, s)):
                    prose = f"{w.article.upper()} {prose}"
                return prose, "upper-prose"
            return w.upper, "upper"
        # capitalised or camel case
        if path_mode or p.isalnum() or n.isalnum() or p == "_" or n == "_":
            return w.camel, "camel"
        words = _previous_words(text, s)
        opens = _sentence_start(text, s) and not (words and words[-1] in DETERMINERS)
        if p == "-" or n == "-":
            prose = w.prose.replace(" ", "-")
            if p != "-" and opens:
                prose = prose[:1].upper() + prose[1:]
            return prose, "compound"
        prose = w.prose
        if w.article and not _determined(words):
            prose = f"{w.article} {prose}"
        if opens:
            prose = prose[:1].upper() + prose[1:]
        return prose, "prose"

    def _identifier_like(self, text: str, s: int, e: int, kind: str) -> bool:
        """In ``identifiers`` mode: is this occurrence an identifier rather than prose?"""
        if kind in ("upper", "upper-hyphen", "camel", "option"):
            return True
        if kind != "snake":
            return False
        p = text[s - 1] if s > 0 else ""
        n = text[e] if e < len(text) else ""
        return (p.isalnum() or n.isalnum() or p in "_-./$" or n in "_-" or p in QUOTES or n in QUOTES
                or _identifier_next(text, e))

    # ---------------------------------------------------------------- residue

    def residue(self, text: str, path: str) -> list[tuple[int, str]]:
        """Lines of an already rewritten text that still hold an old word outside a keep span (identifier-mode
        prose is expected residue and is reported too, so nothing is hidden)."""
        if self._any_old is None:
            return []
        low = text.lower()
        if not any(w in low for w in self.words):
            return []
        spans = self._keep_spans(text, path)
        rows = []
        newlines = None
        for m in self._any_old.finditer(text):
            if any(a <= m.start() < b for a, b in spans):
                continue
            newlines = newlines if newlines is not None else _newline_offsets(text)
            rows.append((bisect.bisect_right(newlines, m.start()) + 1, _line_at(text, m.start())))
        return rows


# -------------------------------------------------------------------- context helpers

def _token_prefix(text: str, s: int) -> str:
    i = s
    while i > 0 and (text[i - 1].isalnum() or text[i - 1] in "_-"):
        i -= 1
    return text[i:s]


def _identifier_next(text: str, e: int) -> bool:
    """True when what follows the word continues an identifier, path, option value or quoted value."""
    n = text[e] if e < len(text) else ""
    if not n:
        return False
    if n.isalnum() or n in "_-/" or n in QUOTES:
        return True
    if n in ".:":
        n2 = text[e + 1] if e + 1 < len(text) else ""
        return bool(n2) and (n2.isalnum() or n2 in "/_$~")
    return False


def _caps_prose(text: str, s: int, e: int) -> bool:
    """True when an all-caps word is set in running text: a space, line start or ``(`` before it, and after it a space,
    line end, sentence punctuation or a possessive ``'s``. That is emphasis, never a constant or a variable, which always
    touch ``_``, ``$``, ``=``, a quote or a letter."""
    p = text[s - 1] if s > 0 else ""
    if p and p not in " \t\n(":
        return False
    n = text[e] if e < len(text) else ""
    if n == "'":
        after = text[e + 2:e + 3]
        return text[e + 1:e + 2] == "s" and not (after.isalnum() or after == "_")
    if n == ".":
        return not _identifier_next(text, e)
    return n == "" or n in " \t\r\n,;:!?)"


def _previous_words(text: str, s: int, count: int = 2) -> list[str]:
    """The last ``count`` words before ``s`` when only spaces, emphasis or quotes separate them (comment leaders
    after a line break are skipped); empty when punctuation sits in between."""
    before = _NEWLINE_LEADER.sub(" ", text[max(0, s - 240):s])
    gap = _GAP_END.search(before)
    head = before[:gap.start()]
    if not head or not (head[-1].isalpha() or head[-1] == "'"):
        return []
    return [w.lower() for w in _WORD_RE.findall(head[-160:])[-count:]]


def _determined(words: list[str]) -> bool:
    if not words:
        return False
    last = words[-1]
    if last in DETERMINERS or last.endswith("'s") or last.endswith("s'"):
        return True
    return len(words) >= 2 and words[-2] in DETERMINERS and last not in PHRASE_BREAKS


def _sentence_start(text: str, s: int) -> bool:
    """True when the word opens a line's prose (after any comment leader or list marker) and that line starts a new
    sentence: the first line, a list item, a line after a blank one or a line of another kind, or a line after one that
    ends with . ! ? or :."""
    ls = text.rfind("\n", 0, s) + 1
    pre = text[ls:s]
    if not _LINE_PREFIX_ONLY.match(pre):
        return _ends_sentence(pre)
    if _LIST_MARKER_END.search(pre) or ls == 0:
        return True
    pe = ls - 1
    ps = text.rfind("\n", 0, pe) + 1
    prev = text[ps:pe]
    cur_kind = (_LEADER_KIND.match(pre).group(1) or "")[:1]
    if cur_kind and cur_kind != (_LEADER_KIND.match(prev).group(1) or "")[:1]:
        return True   # the first line of a comment block
    body = _STRIP_LEADER.sub("", prev).rstrip()
    return not body or body[-1] in ".!?:"


def _ends_sentence(pre: str) -> bool:
    """True when the text before a word on its line ends a sentence: ``.``, ``!`` or ``?`` (closing brackets, quotes or
    emphasis may follow) and then a space, where a full stop does not close an abbreviation or a one-letter initial."""
    m = _SENTENCE_STOP.search(pre)
    if not m:
        return False
    if m.group(1) != ".":
        return True
    word = _WORD_BEFORE_STOP.search(pre[:m.start()]).group(0)
    if not word:
        return True    # the stop follows a bracket, quote or backtick: "(see below). Next"
    return word.lower() not in ABBREVIATIONS and not (len(word) == 1 and word.isalpha())


def _newline_offsets(text: str) -> list[int]:
    return [m.start() for m in re.finditer("\n", text)]


def _line_at(text: str, offset: int) -> str:
    a = text.rfind("\n", 0, offset) + 1
    b = text.find("\n", offset)
    return text[a:b if b >= 0 else len(text)]


def _looks_binary(data: bytes, path: str) -> bool:
    return path.lower().endswith(BINARY_SUFFIXES) or b"\0" in data[:8192]


# -------------------------------------------------------------------- loading

def load_rules(path: str | Path) -> RuleTable:
    """Load and validate a rule table file."""
    raw = Path(path).read_bytes()
    words: list[Word] = []
    literals: list[tuple[str, str]] = []
    keeps: list[tuple[str, re.Pattern]] = []
    modes: list[tuple[str, str]] = []
    for no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        f = line.split("\t")
        kind = f[0]
        try:
            if kind == "word":
                if len(f) != 7:
                    raise RuleError("a word row has 7 fields")
                old, snake, upper, camel, prose, article = f[1:]
                words.append(Word(old, snake, upper, camel, prose, "" if article == "-" else article))
            elif kind == "literal":
                if len(f) != 3:
                    raise RuleError("a literal row has 3 fields")
                literals.append((f[1], f[2]))
            elif kind == "keep":
                if len(f) != 3:
                    raise RuleError("a keep row has 3 fields")
                keeps.append((f[1], re.compile(f[2], re.M)))
            elif kind == "mode":
                if len(f) != 3:
                    raise RuleError("a mode row has 3 fields")
                modes.append((f[1], f[2]))
            else:
                raise RuleError(f"unknown row kind {kind!r}")
        except (RuleError, re.error) as exc:
            raise RuleError(f"{path}:{no}: {exc}") from None
    if len({w.old for w in words}) != len(words):
        raise RuleError(f"{path}: a word appears twice")
    if len({old for old, _ in literals}) != len(literals):
        raise RuleError(f"{path}: a literal appears twice")
    return RuleTable(words, literals, keeps, modes, digest=hashlib.sha256(raw).hexdigest())


# -------------------------------------------------------------------- the tree

def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def tracked_files(root: Path, pathspecs: list[str]) -> list[tuple[str, str]]:
    """(mode, path) of every tracked file under the pathspecs, sorted by path."""
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-s", "-z", "--", *pathspecs], check=True,
                         capture_output=True).stdout.decode("utf-8")
    rows = []
    for entry in out.split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        rows.append((meta.split()[0], path))
    return sorted(set(rows), key=lambda r: r[1])


def scan(table: RuleTable, root: Path, pathspecs: list[str]) -> tuple[list[FileResult], dict[str, str]]:
    """Rewrite every tracked file in memory. Returns the per-file results and the path renames."""
    results: list[FileResult] = []
    renames: dict[str, str] = {}
    for mode, path in tracked_files(root, pathspecs):
        res = FileResult(path)
        new_path = table.rewrite_path(path)
        if new_path != path:
            renames[path] = new_path
        if mode == "160000":
            res.skipped = "submodule"
        elif mode == "120000":
            res.skipped = "symlink"
            target = (root / path).readlink().as_posix() if (root / path).is_symlink() else ""
            if table._any_old and table._any_old.search(target):
                res.residue.append((0, f"symlink target {target}"))
        elif table.mode_of(path) == "skip":
            res.skipped = "mode-skip"
            data = (root / path).read_bytes()
            text = data.decode("utf-8", "replace")
            res.residue = table.residue(text, path) if table._any_old else []
        elif path.lower().endswith(BINARY_SUFFIXES):
            res.skipped = "binary"
        else:
            data = (root / path).read_bytes()
            if _looks_binary(data, path):
                res.skipped = "binary"
            else:
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    res.skipped = "undecodable"
                    text = data.decode("utf-8", "replace")
                    res.residue = table.residue(text, path)
                else:
                    new_text, raw_hits = table._rewrite(text, path)
                    res.old_text, res.new_text = text, new_text
                    newlines = _newline_offsets(text) if raw_hits else []
                    res.hits = []
                    for off, kind, b, a in raw_hits:
                        line_no = bisect.bisect_right(newlines, off)
                        start = newlines[line_no - 1] + 1 if line_no else 0
                        res.hits.append(Hit(path, line_no + 1, kind, b, a, off - start))
                    res.residue = table.residue(new_text, path)
        results.append(res)
    return results, renames


def apply(table: RuleTable, root: Path, results: list[FileResult], renames: dict[str, str]) -> list[str]:
    """Write the rewritten files, stage them, then ``git mv`` the renamed paths. Returns the changed paths."""
    changed = [r.path for r in results if r.new_text is not None and r.new_text != r.old_text]
    for r in results:
        if r.new_text is not None and r.new_text != r.old_text:
            (root / r.path).write_bytes(r.new_text.encode("utf-8"))
    for i in range(0, len(changed), 500):
        _git(root, "add", "--", *changed[i:i + 500])
    for old, new in sorted(renames.items()):
        (root / new).parent.mkdir(parents=True, exist_ok=True)
        _git(root, "mv", "--", old, new)
    return changed


# -------------------------------------------------------------------- report

def build_summary(table: RuleTable, results: list[FileResult], renames: dict[str, str]) -> dict:
    kinds = Counter(h.kind for r in results for h in r.hits)
    words = Counter(h.before.lower() if h.kind != "literal" else h.before for r in results for h in r.hits)
    changed = [r for r in results if r.new_text is not None and r.new_text != r.old_text]
    lines_changed = 0
    for r in changed:
        lines_changed += sum(1 for a, b in zip(r.old_text.splitlines(), r.new_text.splitlines()) if a != b)
    skipped = Counter(r.skipped for r in results if r.skipped)
    return {
        "rules_sha256": table.digest,
        "files_tracked": len(results),
        "files_changed": len(changed),
        "lines_changed": lines_changed,
        "occurrences": sum(kinds.values()),
        "by_kind": dict(sorted(kinds.items())),
        "by_rule": dict(sorted(words.items())),
        "renames": dict(sorted(renames.items())),
        "skipped": dict(sorted(skipped.items())),
        "residue_lines": sum(len(r.residue) for r in results),
    }


def write_report(out, table: RuleTable, results: list[FileResult], renames: dict[str, str], summary: dict,
                 second_pass: dict | None) -> None:
    w = out.write
    w("# rename_sites report\n\n")
    for k in ("rules_sha256", "files_tracked", "files_changed", "lines_changed", "occurrences", "residue_lines"):
        w(f"{k}: {summary[k]}\n")
    w("\n## occurrences by spelling kind\n")
    for k, v in summary["by_kind"].items():
        w(f"{v:8d}  {k}\n")
    w("\n## occurrences by rule\n")
    for k, v in summary["by_rule"].items():
        w(f"{v:8d}  {k}\n")
    w("\n## path renames\n")
    for a, b in summary["renames"].items():
        w(f"{a} -> {b}\n")
    w("\n## files not read\n")
    for k, v in summary["skipped"].items():
        w(f"{v:8d}  {k}\n")
    for r in results:
        if r.skipped in ("mode-skip", "undecodable", "symlink"):
            w(f"  {r.skipped}: {r.path}\n")
    if second_pass is not None:
        w("\n## idempotence (a second pass over the output)\n")
        w(f"files that would change: {second_pass['files_changed']}\npaths that would move: {len(second_pass['renames'])}\n")
        for p in second_pass["changed_paths"][:50]:
            w(f"  {p}\n")
    w("\n## residue (old words left after the rewrite: keep spans excluded; skipped files and identifier-mode prose "
      "included)\n")
    for r in results:
        for line, text in r.residue:
            w(f"{r.path}:{line}: {text.strip()[:220]}\n")
    w("\n## distinct rewrites (kind, before -> after, count)\n")
    pairs = Counter((h.kind, h.before, h.after) for r in results for h in r.hits)
    for (kind, b, a), c in sorted(pairs.items(), key=lambda t: (-t[1], t[0])):
        w(f"{c:8d}  {kind:12s} {b!r} -> {a!r}\n")
    w("\n## distinct tokens rewritten (token around each occurrence, before -> after)\n")
    tok = Counter()
    for r in results:
        if r.new_text is None:
            continue
        old_lines, new_lines = r.old_text.splitlines(), r.new_text.splitlines()
        for h in r.hits:
            if h.line - 1 < len(old_lines) and h.line - 1 < len(new_lines):
                tok[(_token_with(old_lines[h.line - 1], h.before, h.col), h.kind)] += 1
    for (t, kind), c in sorted(tok.items(), key=lambda x: (-x[1], x[0]))[:400]:
        w(f"{c:8d}  {kind:12s} {t}\n")
    w("\n## external-identity hazards (changed lines outside tests/ that look like hosts, options, defaults, file "
      "names, environment or job names)\n")
    for r in results:
        if r.new_text is None or r.path.startswith("tests/") or not r.hits:
            continue
        old_lines, new_lines = r.old_text.splitlines(), r.new_text.splitlines()
        seen = set()
        for h in r.hits:
            if h.line in seen or h.line - 1 >= len(old_lines):
                continue
            tags = _hazard_tags(old_lines[h.line - 1], h.before)
            if tags:
                seen.add(h.line)
                w(f"{r.path}:{h.line}: [{','.join(tags)}]\n    - {old_lines[h.line - 1].strip()[:200]}\n"
                  f"    + {new_lines[h.line - 1].strip()[:200] if h.line - 1 < len(new_lines) else ''}\n")
    w("\n## per-file occurrence counts\n")
    for r in sorted(results, key=lambda r: (-len(r.hits), r.path)):
        if r.hits:
            w(f"{len(r.hits):8d}  {r.path}\n")


_HAZARDS = [
    ("ssh", re.compile(r"\bssh\b|\bscp\b|\brsync\b")),
    ("host", re.compile(r"(?i)\bhost(name)?\b\s*[=:]|--host\b|\bHOST\b")),
    ("default", re.compile(r":-[^}]*}")),
    ("slurm-cluster", re.compile(r"SLURM_CLUSTER_NAME|--clusters?[= ]|sacct|squeue|sbatch")),
    ("mirror", re.compile(r"mirror-repos")),
    ("url", re.compile(r"://")),
    ("env", re.compile(r"environ|getenv|\bexport\b|\$\{?[A-Z_]{3,}")),
    ("job-name", re.compile(r"--job-name|-J\b")),
    ("state-path", re.compile(r"\.local/state|/registry/|inbox/|/workspace|\$SHARED|~/")),
]


def _hazard_tags(line: str, word: str) -> list[str]:
    tags = [name for name, rx in _HAZARDS if rx.search(line)]
    if re.search(re.escape(word) + r"[A-Za-z0-9_-]*\.(?:json|jsonl|yaml|yml|log|sha256|tsv|txt|sh|py|env|out)\b", line,
                 re.I):
        tags.append("file-name")
    return tags


def _token_with(line: str, word: str, col: int = -1) -> str:
    i = col if 0 <= col and line[col:col + len(word)] == word else line.find(word)
    if i < 0:
        return word
    a, b = i, i + len(word)
    while a > 0 and _TOKEN_CHARS.match(line[a - 1]):
        a -= 1
    while b < len(line) and _TOKEN_CHARS.match(line[b]):
        b += 1
    return line[a:b]


# -------------------------------------------------------------------- CLI

def _second_pass(table: RuleTable, root: Path, pathspecs: list[str]) -> dict:
    results, renames = scan(table, root, pathspecs)
    changed = [r.path for r in results if r.new_text is not None and r.new_text != r.old_text]
    return {"files_changed": len(changed), "changed_paths": changed, "renames": renames}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rules", required=True, help="the rule table (tab-separated; kept outside the repository)")
    ap.add_argument("--root", default=None, help="the git work tree to rewrite (default: the one holding the cwd)")
    act = ap.add_mutually_exclusive_group(required=True)
    act.add_argument("--dry-run", action="store_true", help="report what would change and change nothing")
    act.add_argument("--apply", action="store_true", help="rewrite, stage and git mv, then check idempotence")
    act.add_argument("--check", action="store_true", help="exit 1 if a pass would change any file or path")
    ap.add_argument("--report", default=None, help="write the full report here (default: stdout for --dry-run)")
    ap.add_argument("--json", default=None, help="write a JSON summary here")
    ap.add_argument("pathspecs", nargs="*", help="limit the pass to these git pathspecs")
    a = ap.parse_args(argv)

    try:
        table = load_rules(a.rules)
    except (RuleError, OSError) as exc:
        print(f"rename_sites: refusing: {exc}", file=sys.stderr)
        return 2
    root = Path(a.root) if a.root else Path(_git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    root = root.resolve()

    results, renames = scan(table, root, a.pathspecs)
    summary = build_summary(table, results, renames)

    if a.check:
        changed = summary["files_changed"] + len(renames)
        print(f"rename_sites: check: {summary['files_changed']} file(s) would change, {len(renames)} path(s) would "
              f"move, {summary['residue_lines']} residue line(s); rules sha256 {table.digest}")
        return 1 if changed else 0

    second = None
    if a.apply:
        clashes = [new for new in renames.values() if (root / new).exists()]
        if clashes:
            print(f"rename_sites: refusing: rename target(s) already exist: {', '.join(clashes)}", file=sys.stderr)
            return 2
        apply(table, root, results, renames)
        second = _second_pass(table, root, a.pathspecs)
        summary["second_pass_files_changed"] = second["files_changed"]
        summary["second_pass_renames"] = len(second["renames"])

    if a.report:
        with open(a.report, "w", encoding="utf-8") as fh:
            write_report(fh, table, results, renames, summary, second)
    elif a.dry_run:
        write_report(sys.stdout, table, results, renames, summary, second)
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    verb = "applied" if a.apply else "would change"
    print(f"rename_sites: {verb} {summary['files_changed']} file(s), {summary['lines_changed']} line(s), "
          f"{summary['occurrences']} occurrence(s); {len(renames)} path rename(s); {summary['residue_lines']} residue "
          f"line(s); rules sha256 {table.digest}", file=sys.stderr)
    if second is not None and (second["files_changed"] or second["renames"]):
        print(f"rename_sites: NOT IDEMPOTENT: a second pass would change {second['files_changed']} file(s) and move "
              f"{len(second['renames'])} path(s)", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
