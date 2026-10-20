#!/usr/bin/env python3
"""Apply a comment-and-docstring line table to a tree, and prove that every row changes only comment,
docstring or prose text.

A line table is JSON Lines. Each row replaces one contiguous block of lines in one file::

    {"path": "src/pkg/mod.py", "old": ["    # first line", "    # second line"], "new": ["    # one line"]}

``old`` is the exact text of the block's lines (no line terminators) in the tree the table was written against;
``new`` is what replaces them (zero or more lines). A row is keyed by ``(path, old)``: it applies at every place
in the file where ``old`` occurs as consecutive lines, and nowhere else. A row may also carry ``"occ": [k, n]``:
the block occurred ``n`` times in the file the table was written against and the row is for the ``k``-th of them
(1-based, in file order). Such a row applies at that one occurrence, and only while the file still holds exactly
``n`` occurrences; otherwise it is residue, since a change elsewhere has made the placement ambiguous. A row whose
block is not found applies nowhere and is reported as residue, which is what happens when another change already
rewrote those lines.

The checker allows a row only when the lines it replaces, and the lines it writes, are comment, docstring or prose
lines (a code line may keep its code byte for byte and change or delete only its trailing comment), and the file's
skeleton, meaning everything except comments, docstrings and prose, is identical before and after. Per file type:

* Python: ``COMMENT`` tokens, and ``STRING`` tokens that are the first statement of a module, class or function
  (``ast``). Directive comments (``noqa``, ``type:``, ``pragma``, ``fmt:``, coding cookies, shebangs, inline
  script metadata blocks) are code, with one exception: a comment that opens with a ``noqa``, ``type: ignore``,
  ``pyright: ignore``, ``pragma: no cover``, ``nosec`` or ``pylint: disable`` directive and then a separator
  (whitespace and one of an em or en dash, ``--``, ``(``, ``#`` or ``;``; after a bracketed ``ignore[...]`` also
  whitespace before a letter or digit) may change the text after the separator. The directive and the separator
  stay byte for byte, and the text after them stays nonempty.
* Shell (``.sh``, ``.bash``, ``.sbatch``, ``.bats``, ``.env``, or a shell shebang): ``#`` at the start of a word,
  outside quotes, ``${...}`` and heredoc bodies. ``#SBATCH``-style directives, shebangs and ``shellcheck``
  directives are code, with one exception: on a ``#SBATCH``, ``#PBS`` or ``#BSUB`` line whose options hold no quote,
  backslash or ``#``, the comment after whitespace and a ``#`` may change; everything up to and including that
  ``#`` stays byte for byte, and the comment stays nonempty.
* YAML: ``#`` comments outside scalars; the PyYAML event stream must be equal.
* TOML: ``#`` comments outside strings; the parsed document must be equal.
* SQL: ``--`` comments outside single-quoted literals (the rule ``db.normalize_ddl`` uses); block comments are code.
* Markdown: lines outside fenced code blocks and front matter.

Other file types are not supported, and a row in one is refused.

Subcommands:

* ``scan``: list the comment, docstring and prose lines of a tree that match a rules file's flag patterns, grouped
  into units (a comment block, a docstring paragraph, a prose paragraph) for rewriting.
* ``regex``: apply a rules file's rewrite rules to scanned units; emit rows for units whose every flagged line the
  rules fix cleanly, and list the rest.
* ``check``: validate a table against a tree without writing anything.
* ``apply``: apply a table to a tree (in place), file by file, skipping refused rows and printing the residue.
* ``protect``: list files whose docstrings or source text are read as data, or whose bytes another file pins by
  hash, so a table can leave them alone.

A doc-protect list (``--doc-protect``) names files whose docstrings code reads as data: a line ``PATH`` protects every
docstring of the file, and ``PATH::QUALNAME`` only that class's or function's (``PATH::`` the module's).

The rules file (JSON) is private input passed as an argument; this tool names no token itself::

    {"flags": [{"class": "...", "re": "...", "unless": "..."}],
     "rewrites": [{"name": "...", "re": "...", "sub": "...", "cap": false}],
     "paren_segment": "<regex a parenthetical's every segment must match to be deleted>"}
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import io
import json
import os
import re
import sys
import tokenize
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------------------------------------------
# Line kinds
# ---------------------------------------------------------------------------------------------------------------

COMMENT, DOC, PROSE, BLANK, TRAIL, CODE = "comment", "doc", "prose", "blank", "trail", "code"
EDITABLE = {COMMENT, DOC, PROSE, BLANK, TRAIL}


class Unsupported(Exception):
    """The file type has no checker, or the file does not parse."""


@dataclass
class Analysis:
    """Per-line kinds of one file and the skeleton two versions must share."""

    kinds: List[str]
    comment_col: List[Optional[int]]  # where the comment (or prose) text starts on each line, if any
    skeleton: object
    doc_lines: List[bool] = field(default_factory=list)
    doc_owner: List[Optional[str]] = field(default_factory=list)  # Python: the qualified name whose docstring a line is in


_TERM = re.compile(r"\r\n|\r|\n")
#: characters that some readers treat as line breaks and others do not (str.splitlines, YAML, bash disagree)
_ODD_BREAKS = re.compile("\r(?!\n)|[\x0b\x1c\x1d\x1e\x85\u2028\u2029]")


def split_lines(text: str) -> Tuple[List[str], List[str]]:
    """Lines without terminators, and each line's terminator (the last may be ''). Only ``\\r\\n``, ``\\r`` and
    ``\\n`` end a line, as in Python's tokenizer; ``str.splitlines`` would also split on form feeds and Unicode
    separators, which would put line numbers out of step with the tokens."""
    body, ends = [], []
    pos = 0
    for m in _TERM.finditer(text):
        body.append(text[pos:m.start()])
        ends.append(m.group())
        pos = m.end()
    if pos < len(text):
        body.append(text[pos:])
        ends.append("")
    return body, ends


def join_lines(body: Sequence[str], ends: Sequence[str]) -> str:
    return "".join(b + e for b, e in zip(body, ends))


SHELL_EXT = (".sh", ".bash", ".sbatch", ".bats", ".env")
PY_EXT = (".py", ".pyi")


def lang_of(path: str, text: str) -> Optional[str]:
    base = os.path.basename(path)
    if base.endswith(PY_EXT):
        return "python"
    if base.endswith(SHELL_EXT):
        return "shell"
    if base.endswith((".yaml", ".yml")):
        return "yaml"
    if base.endswith(".toml"):
        return "toml"
    if base.endswith(".sql"):
        return "sql"
    if base.endswith(".md"):
        return "markdown"
    if "." not in base and text.startswith("#!"):
        first = text.split("\n", 1)[0]
        if "python" in first:
            return "python"
        if re.search(r"\b(?:ba|z|da)?sh\b", first):
            return "shell"
    return None


# --- Python ----------------------------------------------------------------------------------------------------

PY_DIRECTIVE = re.compile(
    r"#\s*(?:type:|noqa\b|pragma\b|fmt:|pylint:|mypy:|isort:|nosec\b|pyright:|ruff:|-\*-|vim:|coding[:=])"
    r"|^#!")
PEP723 = re.compile(r"^#\s*///")
#: a directive whose readers stop at a fixed grammar, matched at the start of the comment: the text after it and a
#: separator that ends every reader's parse is free text a row may rewrite (the directive and separator stay exact)
_DIR_HEAD = re.compile(
    r"#[ \t]*(?:"
    r"noqa(?::[ \t]*[A-Z]+[0-9]+(?:[ \t]*,[ \t]*[A-Z]+[0-9]+)*)?"
    r"|(?P<bracketed>(?:type|pyright):[ \t]*ignore\[[^\]#\n]*\])"
    r"|(?:type|pyright):[ \t]*ignore"
    r"|pragma:[ \t]*no[ \t]+(?:cover|branch)"
    r"|nosec(?:[ \t]+B[0-9]+(?:[ \t]*,[ \t]*B[0-9]+)*)?"
    r"|pylint:[ \t]*disable=[\w-]+(?:,[\w-]+)*"
    r")")
_DIR_SEP = re.compile(r"[ \t]+(?:—|–|--|\(|#|;)[ \t]*")
_DIR_SEP_BRACKETED = re.compile(r"[ \t]+(?=[A-Za-z0-9])")


def directive_free_text(comment: str) -> Optional[int]:
    """Where the rewritable free text of a Python directive comment starts (an index into ``comment``), or None when
    the whole comment is the directive's. The directive must open the comment; the separator after it must end every
    reader's parse of the directive; the free text must be nonempty."""
    m = _DIR_HEAD.match(comment)
    if not m:
        return None
    rest = comment[m.end():]
    s = _DIR_SEP.match(rest)
    if s is None and m.group("bracketed"):
        s = _DIR_SEP_BRACKETED.match(rest)
    if s is None or not rest[s.end():].strip():
        return None
    return m.end() + s.end()
_NONCODE = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
            tokenize.ENCODING, tokenize.ENDMARKER}


def _byte_col_to_char(line: str, col: int) -> int:
    return len(line.encode("utf-8")[:col].decode("utf-8", errors="ignore"))


def _docstring_spans(text: str, lines: List[str]) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
    """Each docstring's start, end and owner: "" for the module, else the dotted name of its class or function."""
    tree = ast.parse(text)
    spans = []

    def walk(node, qual):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                v = first.value
                s = (v.lineno, _byte_col_to_char(lines[v.lineno - 1], v.col_offset))
                e = (v.end_lineno, _byte_col_to_char(lines[v.end_lineno - 1], v.end_col_offset))
                spans.append((s, e, qual))
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, f"{qual}.{child.name}" if qual else child.name)
            else:
                walk(child, qual)
    walk(tree, "")
    return spans


_COOKIE = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")


def analyze_python(text: str) -> Analysis:
    lines, _ = split_lines(text)
    n = len(lines)
    for ln in lines[:2]:
        m = _COOKIE.match(ln)
        if m and m.group(1).lower().replace("_", "-") not in ("utf-8", "utf8"):
            raise Unsupported(f"python source declares the encoding {m.group(1)!r}; only UTF-8 source is checked")
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
        spans = _docstring_spans(text, lines)
    except (SyntaxError, tokenize.TokenError, IndentationError, ValueError) as e:
        raise Unsupported(f"python does not parse: {e}")

    def in_doc(t) -> bool:
        return t.type == tokenize.STRING and any(s <= t.start and t.end <= e for s, e, _ in spans)

    owner_of_line: Dict[int, str] = {}
    for (sl_, _), (el_, _), q in spans:
        for ln_ in range(sl_, el_ + 1):
            owner_of_line[ln_] = q

    pep723 = [False] * (n + 2)  # inline script metadata: "# /// <type>" ... "# ///"
    inside = False
    for i, ln in enumerate(lines, 1):
        st = ln.strip()
        if PEP723.match(st):
            pep723[i] = True
            inside = st != "# ///" if not inside else False
            continue
        pep723[i] = inside

    has_code = [False] * (n + 2)
    has_doc = [False] * (n + 2)
    comment_col: List[Optional[int]] = [None] * (n + 2)
    directive = [False] * (n + 2)
    dfree = [False] * (n + 2)  # the line's comment is a directive whose free text a row may rewrite
    skeleton = []
    fstr_open = {getattr(tokenize, n) for n in ("FSTRING_START", "TSTRING_START") if hasattr(tokenize, n)}
    fstr_close = {getattr(tokenize, n) for n in ("FSTRING_END", "TSTRING_END") if hasattr(tokenize, n)}
    fdepth = 0
    for t in toks:
        sl, sc = t.start
        el, _ = t.end
        if t.type in fstr_open:
            fdepth += 1
        elif t.type in fstr_close:
            fdepth -= 1
        if t.type == tokenize.COMMENT and fdepth > 0:
            # a comment inside an f-string's replacement field: the string's value can depend on it (debug "=")
            directive[sl] = True
            skeleton.append(("FSTRING_COMMENT", t.string, sc))
            continue
        if t.type == tokenize.COMMENT:
            free = None if pep723[sl] else directive_free_text(t.string)
            if free is not None:
                # the directive and its separator are code; the text after them is a trailing comment
                skeleton.append(("DIRECTIVE_HEAD", t.string[:free], sc))  # a distinct tag: emptying the free text changes it
                comment_col[sl] = sc + free
                dfree[sl] = True
            elif PY_DIRECTIVE.search(t.string) or pep723[sl]:
                directive[sl] = True
                skeleton.append(("DIRECTIVE", t.string, sc))
            else:
                comment_col[sl] = sc
            continue
        if in_doc(t):
            m = re.match(r"([rRuUbBfF]*)('''|\"\"\"|'|\")", t.string)
            skeleton.append(("DOC", m.group(0) if m else "", sc))
            for ln in range(sl, el + 1):
                has_doc[ln] = True
            continue
        if t.type in (tokenize.NL,):
            continue
        if t.type == tokenize.NEWLINE:
            skeleton.append(("NEWLINE",))
            continue
        if t.type not in _NONCODE:
            for ln in range(sl, el + 1):
                has_code[ln] = True
        skeleton.append((t.type, t.string, sc))
    kinds, ccol, docl, downer = [], [], [], []
    for i, ln in enumerate(lines, 1):
        downer.append(owner_of_line.get(i) if has_doc[i] else None)
        if has_code[i] or directive[i] or dfree[i]:
            if comment_col[i] is not None and (has_code[i] or dfree[i]) and not directive[i]:
                kinds.append(TRAIL)
                ccol.append(comment_col[i])
            else:
                kinds.append(CODE)
                ccol.append(None)
        elif has_doc[i]:
            kinds.append(DOC)
            ccol.append(len(ln) - len(ln.lstrip()))
        elif comment_col[i] is not None:
            kinds.append(COMMENT)
            ccol.append(comment_col[i])
        elif ln.strip() == "":
            kinds.append(BLANK)
            ccol.append(None)
        else:  # a continuation line inside a bracket, etc.
            kinds.append(CODE)
            ccol.append(None)
        docl.append(has_doc[i])
    return Analysis(kinds, ccol, skeleton, docl, downer)


# --- Shell -----------------------------------------------------------------------------------------------------

SH_DIRECTIVE = re.compile(r"^\s*#(?:!|SBATCH\b|PBS\b|BSUB\b|\$ |@ |\s*shellcheck\b)")
SH_TRAIL_DIRECTIVE = re.compile(r"#\s*shellcheck\b")
#: a scheduler directive line with an inline comment: the options hold no quote, backslash or ``#``, so the first
#: whitespace-led ``#`` starts the comment for the scheduler as for the lexer; group 1 (kept exact) ends at that ``#``
_SH_DIR_FREE = re.compile(r"^(#(?:SBATCH|PBS|BSUB)\b[^#'\"\\]*?[ \t]+#[ \t]*)(\S.*)$")
#: ``<<`` or ``<<-`` and its delimiter word: single-quoted, double-quoted, ANSI-C quoted, or a bare word (an optional
#: leading backslash quotes it). A word that mixes quoted and bare parts ends where this stops, which the caller refuses.
_HEREDOC = re.compile(r"<<(-?)[ \t]*(?:'([^']*)'|\"([^\"$`\\]*)\"|\$'([^'\\]*)'|\\?([^\s;&|<>()'\"`\\$]+))(?=[\s;&|<>()]|$)")
_WORD_BREAK = set(" \t;&|()<>")


def analyze_shell(text: str) -> Analysis:
    """A conservative lexer. A ``#`` counts as a comment only at the start of a word outside every quoting
    context; anything it cannot place (an unterminated quote at end of file) refuses the whole file."""
    lines, _ = split_lines(text)
    n = len(lines)
    kinds: List[str] = [CODE] * n
    ccol: List[Optional[int]] = [None] * n
    state = []  # stack of "'", '"', '`', '${', '$(', "$'"
    pending: List[Tuple[str, bool]] = []  # heredoc delimiters opened on the current line
    heredoc: Optional[Tuple[str, bool]] = None
    data_line = [False] * n
    dir_head = [False] * n  # a scheduler directive line whose inline comment a row may rewrite
    cont_prev = False
    for i, ln in enumerate(lines):
        if heredoc is not None:
            data_line[i] = True
            delim, strip_tabs = heredoc
            if (ln.lstrip("\t") if strip_tabs else ln) == delim:
                heredoc = None
            continue
        if state and state[-1] in ("'", '"', "`", "$'"):
            data_line[i] = True  # starts inside a multi-line quoted string
        comment_at = None
        j = 0
        L = len(ln)
        while j < L:
            c = ln[j]
            top = state[-1] if state else None
            if top == "'":
                if c == "'":
                    state.pop()
                j += 1
                continue
            if top == "$'":
                if c == "\\":
                    j += 2
                    continue
                if c == "'":
                    state.pop()
                j += 1
                continue
            if top == '"':
                if c == "\\":
                    j += 2
                    continue
                if c == '"':
                    state.pop()
                elif ln.startswith("$(", j) or ln.startswith("${", j):
                    state.append(ln[j:j + 2])
                    j += 2
                    continue
                elif c == "`":
                    state.append("`")
                j += 1
                continue
            if top == "`":
                if c == "\\":
                    j += 2
                    continue
                if c == "`":
                    state.pop()
                j += 1
                continue
            # unquoted context (top is None, '${' or '$(')
            if c == "\\":
                j += 2
                continue
            if c == "'":
                state.append("'")
                j += 1
                continue
            if ln.startswith("$'", j):
                state.append("$'")
                j += 2
                continue
            if c == '"':
                state.append('"')
                j += 1
                continue
            if c == "`":
                state.append("`")
                j += 1
                continue
            if ln.startswith("${", j):
                state.append("${")
                j += 2
                continue
            if ln.startswith("$(", j):
                state.append("$(")
                j += 2
                continue
            if c == "}" and top == "${":
                state.pop()
                j += 1
                continue
            if c == "(" and top == "$(":
                state.append("$(")  # nested parens inside a substitution count as one level each
                j += 1
                continue
            if c == ")" and top == "$(":
                state.pop()
                j += 1
                continue
            if c == "#" and top != "${" and (j == 0 or ln[j - 1] in _WORD_BREAK) and not (j > 0 and ln[j - 1] == "$"):
                comment_at = j
                break
            if ln.startswith("<<<", j):  # a here-string, not a heredoc
                j += 3
                continue
            if ln.startswith("<<", j) and top in (None, "$("):
                if ln[:j].count("((") > ln[:j].count("))"):  # an arithmetic shift, not a heredoc
                    j += 2
                    continue
                m = _HEREDOC.match(ln, j)
                if not m:
                    raise Unsupported(f"line {i + 1}: a heredoc opener the lexer cannot read")
                delim = next(g for g in m.groups()[1:] if g is not None)
                pending.append((delim, m.group(1) == "-"))
                j = m.end()
                continue
            j += 1
        if data_line[i]:
            kinds[i] = CODE
        elif comment_at is not None:
            dm = _SH_DIR_FREE.match(ln) if comment_at == 0 and not cont_prev else None
            if dm:
                kinds[i] = TRAIL
                ccol[i] = dm.end(1)
                dir_head[i] = True
            elif SH_DIRECTIVE.match(ln) or (i == 0 and ln.startswith("#!")):
                kinds[i] = CODE
            elif ln[:comment_at].strip() == "":
                kinds[i] = CODE if cont_prev else COMMENT
                ccol[i] = None if cont_prev else comment_at
            elif SH_TRAIL_DIRECTIVE.match(ln, comment_at):
                kinds[i] = CODE
            else:
                kinds[i] = TRAIL
                ccol[i] = comment_at
        elif ln.strip() == "":
            kinds[i] = CODE if cont_prev else BLANK
        code_part = ln[:comment_at] if comment_at is not None else ln
        cont_prev = (not data_line[i]) and code_part.rstrip().endswith("\\") and comment_at is None
        if pending:
            heredoc = pending.pop(0)
            if pending:  # several heredocs on one line: too rare to model; refuse
                raise Unsupported("several heredocs opened on one line")
    if state or heredoc is not None:
        raise Unsupported(f"shell lexer ends inside {state[-1] if state else 'a heredoc'}")
    skeleton = []
    for i, ln in enumerate(lines):
        if kinds[i] in (COMMENT, BLANK):
            continue
        if kinds[i] == TRAIL:
            skeleton.append(("DIRECTIVE_HEAD", ln[:ccol[i]]) if dir_head[i] else ln[:ccol[i]].rstrip())
        else:
            skeleton.append(ln)
    return Analysis(kinds, ccol, skeleton)


# --- YAML ------------------------------------------------------------------------------------------------------

def analyze_yaml(text: str) -> Analysis:
    try:
        import yaml  # noqa: F401
    except ImportError:
        raise Unsupported("PyYAML is not installed")
    import yaml
    lines, _ = split_lines(text)
    n = len(lines)
    try:
        toks = list(yaml.scan(text, Loader=yaml.SafeLoader))
        events = [(type(e).__name__, getattr(e, "anchor", None), getattr(e, "tag", None),
                   getattr(e, "implicit", None), getattr(e, "value", None), getattr(e, "style", None),
                   getattr(e, "flow_style", None), getattr(e, "explicit", None))
                  for e in yaml.parse(text, Loader=yaml.SafeLoader)]
    except yaml.YAMLError as e:
        raise Unsupported(f"yaml does not parse: {e}")
    covered = [False] * (n + 1)
    last_end = [-1] * (n + 1)
    for t in toks:
        if type(t).__name__ in ("StreamStartToken", "StreamEndToken"):
            continue
        sl, el = t.start_mark.line, t.end_mark.line
        ec = t.end_mark.column
        if el > sl and ec == 0:
            el -= 1  # a token that ends at column 0 ends on the previous line
            ec = 10 ** 9
        for ln in range(sl, min(el, n - 1) + 1):
            covered[ln] = True
        if el < n:
            last_end[el] = max(last_end[el], ec)
    kinds, ccol = [], []
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if not covered[i]:
            if s.startswith("#"):
                kinds.append(COMMENT)
                ccol.append(len(ln) - len(s))
            elif s == "":
                kinds.append(BLANK)
                ccol.append(None)
            else:
                kinds.append(CODE)
                ccol.append(None)
            continue
        e = last_end[i]
        rest = ln[e:] if 0 <= e <= len(ln) else ""
        k = rest.find("#")
        if e >= 0 and e <= len(ln) and rest.strip().startswith("#") and (k == 0 or rest[k - 1] in " \t"):
            if k == 0 and e > 0 and ln[e - 1] not in " \t":
                kinds.append(CODE)
                ccol.append(None)
            else:
                kinds.append(TRAIL)
                ccol.append(e + k)
        else:
            kinds.append(CODE)
            ccol.append(None)
    return Analysis(kinds, ccol, events)


# --- TOML ------------------------------------------------------------------------------------------------------

def analyze_toml(text: str) -> Analysis:
    import tomllib
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise Unsupported(f"toml does not parse: {e}")
    lines, _ = split_lines(text)
    kinds, ccol = [], []
    multi = None
    for ln in lines:
        if multi is not None:
            kinds.append(CODE)
            ccol.append(None)
            if ln.count(multi) % 2 == 1:
                multi = None
            continue
        j, quote, at = 0, None, None
        while j < len(ln):
            if quote is None:
                if ln.startswith('"""', j) or ln.startswith("'''", j):
                    q = ln[j:j + 3]
                    end = ln.find(q, j + 3)
                    if end < 0:
                        multi = q
                        break
                    j = end + 3
                    continue
                if ln[j] in "\"'":
                    quote = ln[j]
                elif ln[j] == "#":
                    at = j
                    break
            elif ln[j] == "\\" and quote == '"':
                j += 1
            elif ln[j] == quote:
                quote = None
            j += 1
        if multi is not None:
            kinds.append(CODE)
            ccol.append(None)
        elif at is None:
            kinds.append(BLANK if ln.strip() == "" else CODE)
            ccol.append(None)
        elif ln[:at].strip() == "":
            kinds.append(COMMENT)
            ccol.append(at)
        else:
            kinds.append(TRAIL)
            ccol.append(at)
    return Analysis(kinds, ccol, json.dumps(doc, sort_keys=True, default=str))


# --- SQL -------------------------------------------------------------------------------------------------------

_SQL_CLOSE = {"'": "'", '"': '"', "`": "`", "[": "]"}


def analyze_sql(text: str) -> Analysis:
    """``--`` starts a comment only outside single-quoted literals and quoted identifiers (``"x"``, ```x```,
    ``[x]``); a doubled quote inside one is an escaped quote."""
    lines, _ = split_lines(text)
    kinds, ccol, skeleton = [], [], []
    in_str: Optional[str] = None  # the closing character of the literal or identifier we are inside
    for ln in lines:
        j, at, starts_in_str, block = 0, None, in_str is not None, False
        while j < len(ln):
            c = ln[j]
            if in_str is not None:
                if c == in_str:
                    if in_str != "]" and ln.startswith(in_str * 2, j):
                        j += 2
                        continue
                    in_str = None
            elif c in _SQL_CLOSE:
                in_str = _SQL_CLOSE[c]
            elif ln.startswith("--", j):
                at = j
                break
            elif ln.startswith("/*", j) or ln.startswith("*/", j):
                block = True
            j += 1
        if starts_in_str or block:
            kinds.append(CODE)
            ccol.append(None)
            skeleton.append(ln)
        elif at is None:
            kinds.append(BLANK if ln.strip() == "" else CODE)
            ccol.append(None)
            if ln.strip():
                skeleton.append(ln)
        elif ln[:at].strip() == "":
            kinds.append(COMMENT)
            ccol.append(at)
        else:
            kinds.append(TRAIL)
            ccol.append(at)
            skeleton.append(ln[:at].rstrip())
    if in_str is not None:
        raise Unsupported("sql ends inside a string literal or a quoted identifier")
    return Analysis(kinds, ccol, skeleton)


# --- Markdown --------------------------------------------------------------------------------------------------

#: a code fence, also inside block quotes and list items; indentation is not limited, so a deeply nested fence
#: still counts as code (treating a line as code only ever refuses a row)
FENCE = re.compile(r"^(?:[ \t]*>)*[ \t]*(?:(?:[-*+]|\d{1,9}[.)])[ \t]+)?(`{3,}|~{3,})")
_MD_PREFIX = re.compile(r"^(?:[ \t]*>)*[ \t]*")


def analyze_markdown(text: str) -> Analysis:
    lines, _ = split_lines(text)
    kinds, ccol, skeleton = [], [], []
    fence = None
    front = bool(lines) and lines[0].strip() == "---"
    for i, ln in enumerate(lines):
        if front:
            kinds.append(CODE)
            ccol.append(None)
            skeleton.append(ln)
            if i > 0 and ln.strip() in ("---", "..."):
                front = False
            continue
        m = FENCE.match(ln)
        if fence is not None:
            kinds.append(CODE)
            ccol.append(None)
            skeleton.append(ln)
            rest = _MD_PREFIX.sub("", ln).strip()
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and rest == m.group(1):
                fence = None
            continue
        if m:
            fence = m.group(1)
            kinds.append(CODE)
            ccol.append(None)
            skeleton.append(ln)
            continue
        if ln.strip() == "":
            kinds.append(BLANK)
            ccol.append(None)
        else:
            kinds.append(PROSE)
            ccol.append(len(ln) - len(ln.lstrip()))
    return Analysis(kinds, ccol, skeleton)


ANALYZERS = {"python": analyze_python, "shell": analyze_shell, "yaml": analyze_yaml, "toml": analyze_toml,
             "sql": analyze_sql, "markdown": analyze_markdown}


def analyze(path: str, text: str) -> Tuple[str, Analysis]:
    lang = lang_of(path, text)
    if lang is None:
        raise Unsupported("no checker for this file type")
    if _ODD_BREAKS.search(text):
        raise Unsupported("the file holds a bare carriage return or a character some readers treat as a line break")
    return lang, ANALYZERS[lang](text)


# ---------------------------------------------------------------------------------------------------------------
# Rows: finding, checking, applying
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Row:
    path: str
    old: List[str]
    new: List[str]
    raw: dict
    occ: Optional[Tuple[int, int]] = None  # (k, n): the k-th of n occurrences of ``old`` in the origin file

    @property
    def key(self) -> Tuple[str, Tuple[str, ...], Optional[Tuple[int, int]]]:
        return (self.path, tuple(self.old), self.occ)


def load_table(path: str) -> List[Row]:
    rows, seen = [], {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not isinstance(d.get("old"), list) or not isinstance(d.get("new"), list) or not d["old"]:
                raise SystemExit(f"{path}:{n}: a row needs a nonempty list 'old' and a list 'new'")
            if any("\n" in s or "\r" in s for s in d["old"] + d["new"]):
                raise SystemExit(f"{path}:{n}: a line may not hold a line terminator")
            occ = d.get("occ")
            if occ is not None:
                if (not isinstance(occ, list) or len(occ) != 2 or not all(isinstance(x, int) for x in occ)
                        or not 1 <= occ[0] <= occ[1]):
                    raise SystemExit(f"{path}:{n}: 'occ' must be [k, n] with 1 <= k <= n")
                occ = (occ[0], occ[1])
            r = Row(d["path"], list(d["old"]), list(d["new"]), d, occ)
            if r.key in seen:
                if seen[r.key] != r.new:
                    raise SystemExit(f"{path}:{n}: two rows share the key {r.path} {r.old[0]!r} with different text")
                continue
            seen[r.key] = r.new
            rows.append(r)
    return rows


def find_block(lines: Sequence[str], block: Sequence[str]) -> List[int]:
    k = len(block)
    first = block[0]
    return [i for i in range(len(lines) - k + 1) if lines[i] == first and list(lines[i:i + k]) == list(block)]


def _code_prefix(line: str, kind: str, col: Optional[int]) -> Optional[str]:
    if kind == TRAIL and col is not None:
        return line[:col].rstrip()
    return None


def check_edit(path: str, old_text: str, new_text: str,
               spans: Sequence[Tuple[int, int, int]], doc_protected=False) -> List[str]:
    """Errors for replacing old-file lines ``[start, start+old_len)`` by the new-file lines
    ``[new_start, new_start+new_len)`` (``spans`` holds ``(start, old_len, new_len)`` in old-file order).
    Empty means the edit touches comment, docstring or prose text only."""
    errors = []
    try:
        _, a_old = analyze(path, old_text)
        _, a_new = analyze(path, new_text)
    except Unsupported as e:
        return [str(e)]
    old_lines, _ = split_lines(old_text)
    new_lines, _ = split_lines(new_text)
    shift = 0
    for start, old_len, new_len in spans:
        ns = start + shift
        old_k = a_old.kinds[start:start + old_len]
        new_k = a_new.kinds[ns:ns + new_len]
        bad_old = [start + 1 + i for i, k in enumerate(old_k) if k not in EDITABLE]
        old_codes = [p for p in (_code_prefix(old_lines[start + i], k, a_old.comment_col[start + i])
                                 for i, k in enumerate(old_k)) if p is not None]
        # a written line may be pure code only when it is, byte for byte, the code of a replaced line whose trailing
        # comment the row deletes; the order check below keeps those code parts identical and in order
        bad_new, new_codes = [], []
        for i, k in enumerate(new_k):
            line = new_lines[ns + i]
            if k in EDITABLE:
                p = _code_prefix(line, k, a_new.comment_col[ns + i])
                if p is not None:
                    new_codes.append(p)
            elif k == CODE and line == line.rstrip() and line in old_codes:
                new_codes.append(line)
            else:
                bad_new.append(ns + 1 + i)
        if bad_old:
            errors.append(f"replaces non-comment line(s) {bad_old}")
        if doc_protected is True and (DOC in old_k or DOC in new_k):
            errors.append("edits a docstring that code reads as data")
        elif doc_protected and doc_protected is not True:
            owners = {a_old.doc_owner[start + i] for i, k in enumerate(old_k) if k == DOC and a_old.doc_owner}
            owners |= {a_new.doc_owner[ns + i] for i, k in enumerate(new_k) if k == DOC and a_new.doc_owner}
            hit = sorted(o for o in owners if o is not None and o in doc_protected)
            if hit:
                errors.append(f"edits the docstring of {hit[0] or 'the module'}, which code reads as data")
        if bad_new:
            errors.append(f"writes non-comment line(s) {bad_new} (new-file numbering)")
        if all(k == BLANK for k in old_k):
            errors.append("replaces only blank lines")
        if old_codes != new_codes:
            errors.append("changes the code before a trailing comment")
        shift += new_len - old_len
    if a_old.skeleton != a_new.skeleton:
        errors.append("changes the file's code skeleton (something other than comment, docstring or prose text)")
    return errors


@dataclass
class FileResult:
    path: str
    applied: List[Row] = field(default_factory=list)
    residue: List[Tuple[Row, str]] = field(default_factory=list)
    new_text: Optional[str] = None


def _apply_rows(text: str, rows: Sequence[Row]) -> Tuple[Optional[str], List[Tuple[int, int, int]], List[Tuple[Row, str]], List[Row]]:
    lines, ends = split_lines(text)
    term = "\n"
    for e in ends:
        if e:
            term = e
            break
    placements = []  # (start, row)
    missing = []
    for r in rows:
        hits = find_block(lines, r.old)
        if not hits:
            missing.append((r, "not found"))
            continue
        if r.occ is not None:
            k, n = r.occ
            if len(hits) != n:
                missing.append((r, f"the block occurs {len(hits)} time(s); the row was written for occurrence {k} of {n}"))
                continue
            hits = [hits[k - 1]]
        for h in hits:
            placements.append((h, r))
    placements.sort(key=lambda p: p[0])
    taken, last_end, overlap = [], -1, set()
    for i, (h, r) in enumerate(placements):
        if h < last_end:
            overlap.add(id(r))
            overlap.add(id(placements[i - 1][1]))
        last_end = max(last_end, h + len(r.old))
    keep = [(h, r) for h, r in placements if id(r) not in overlap]
    over = []
    seen_over = set()
    for h, r in placements:
        if id(r) in overlap and id(r) not in seen_over:
            seen_over.add(id(r))
            over.append((r, "overlaps another row"))
    if not keep:
        return None, [], missing + over, []
    out_body, out_ends, spans = [], [], []
    cur = 0
    for h, r in keep:
        out_body.extend(lines[cur:h])
        out_ends.extend(ends[cur:h])
        block_end = ends[h + len(r.old) - 1]
        for j, s in enumerate(r.new):
            out_body.append(s)
            out_ends.append(term if (j < len(r.new) - 1 or block_end) else block_end)
        spans.append((h, len(r.old), len(r.new)))
        cur = h + len(r.old)
    out_body.extend(lines[cur:])
    out_ends.extend(ends[cur:])
    used = []
    seen = set()
    for _, r in keep:
        if id(r) not in seen:
            seen.add(id(r))
            used.append(r)
    return join_lines(out_body, out_ends), spans, missing + over, used


def process_file(path: str, text: str, rows: Sequence[Row], doc_protected=False) -> FileResult:
    """Apply every row of one file that the checker allows. Rows are tried together first; when that fails each
    row is checked alone, and the allowed ones are applied together and checked once more."""
    res = FileResult(path)
    new_text, spans, residue, used = _apply_rows(text, rows)
    res.residue.extend(residue)
    if new_text is None:
        return res
    errs = check_edit(path, text, new_text, spans, doc_protected)
    if not errs:
        res.applied, res.new_text = used, new_text
        return res
    good = []
    for r in used:
        t1, sp1, _, u1 = _apply_rows(text, [r])
        e1 = check_edit(path, text, t1, sp1, doc_protected) if t1 is not None else ["not found"]
        if e1:
            res.residue.append((r, "; ".join(e1)))
        else:
            good.append(r)
    if good:
        t2, sp2, _, u2 = _apply_rows(text, good)
        e2 = check_edit(path, text, t2, sp2, doc_protected)
        if e2:  # rows that pass alone but not together: refuse them all rather than guess
            res.residue.extend((r, "allowed alone, refused together: " + "; ".join(e2)) for r in good)
        else:
            res.applied, res.new_text = u2, t2
    return res


def doc_protection(path: str, entries: Sequence[str]):
    """True when every docstring of the file is protected, else the set of protected owners (possibly empty)."""
    owners = set()
    for e in entries:
        g, sep, qual = e.partition("::")
        if fnmatch.fnmatchcase(path, g) or path == g:
            if not sep:
                return True
            owners.add(qual)
    return frozenset(owners)


def path_skipped(path: str, globs: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, g) or path == g for g in globs)


def read_globs(files: Sequence[str], inline: Sequence[str]) -> List[str]:
    out = list(inline)
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.split("\t", 1)[0].strip()
                if line and not line.startswith("#"):
                    out.append(line)
    return out


def run_table(tree: str, table: str, skip: Sequence[str], write: bool, report: Optional[str],
              doc_protect: Sequence[str] = (), applied_out: Optional[str] = None) -> int:
    rows = load_table(table)
    by_path: Dict[str, List[Row]] = {}
    for r in rows:
        by_path.setdefault(r.path, []).append(r)
    n_applied = n_rows_applied = 0
    residue: List[Tuple[Row, str]] = []
    applied_rows: List[Row] = []
    skipped = 0
    root = os.path.realpath(tree)
    for path in sorted(by_path):
        rs = by_path[path]
        if path_skipped(path, skip):
            skipped += len(rs)
            continue
        full = os.path.join(tree, path)
        parts = path.replace("\\", "/").split("/")
        if (os.path.isabs(path) or ".." in parts or "" in parts or "." in parts
                or not os.path.realpath(full).startswith(root + os.sep) or os.path.islink(full)):
            residue.extend((r, "the path is not a plain relative path inside the tree") for r in rs)
            continue
        if not os.path.isfile(full):
            residue.extend((r, "file missing") for r in rs)
            continue
        with open(full, encoding="utf-8", newline="") as f:
            text = f.read()
        res = process_file(path, text, rs, doc_protection(path, doc_protect))
        residue.extend(res.residue)
        if res.applied:
            n_applied += 1
            n_rows_applied += len(res.applied)
            applied_rows.extend(res.applied)
            if write:
                with open(full, "w", encoding="utf-8", newline="") as f:
                    f.write(res.new_text)
    out = {"rows": len(rows), "rows_applied": n_rows_applied, "files_changed": n_applied,
           "rows_skipped_by_owner": skipped, "residue": len(residue)}
    print(json.dumps(out))
    if applied_out:  # exactly the rows written (as given in the table), for tools that re-derive the result
        with open(applied_out, "w", encoding="utf-8") as f:
            for r in applied_rows:
                f.write(json.dumps(r.raw, ensure_ascii=False) + "\n")
    lines = [json.dumps({"path": r.path, "old": r.old, "reason": why}) for r, why in residue]
    if report:
        with open(report, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
    else:
        for ln in lines:
            print("residue " + ln)
    return len(residue)


# ---------------------------------------------------------------------------------------------------------------
# Rules: flags and the regex pass
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Rules:
    flags: List[Tuple[str, re.Pattern, Optional[re.Pattern]]]
    rewrites: List[Tuple[str, re.Pattern, str, bool]]
    paren_segment: Optional[re.Pattern]


def _flags(d: dict) -> int:
    f = 0
    for ch in d.get("flags", ""):
        f |= {"i": re.I, "m": re.M}[ch]
    return f


def load_rules(path: str) -> Rules:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    flags = [(x["class"], re.compile(x["re"], _flags(x)), re.compile(x["unless"], _flags(x)) if x.get("unless") else None)
             for x in d.get("flags", [])]
    rewrites = [(x["name"], re.compile(x["re"], _flags(x)), x["sub"], bool(x.get("cap"))) for x in d.get("rewrites", [])]
    seg = re.compile(d["paren_segment"]) if d.get("paren_segment") else None
    return Rules(flags, rewrites, seg)


def flag_classes(text: str, rules: Rules) -> List[str]:
    out = []
    for cls, rx, unless in rules.flags:
        for m in rx.finditer(text):
            if unless is not None and any(u.start() <= m.start() and m.end() <= u.end() for u in unless.finditer(text)):
                continue
            out.append(cls)
            break
    return sorted(set(out))


def _strip_parens(text: str, seg: re.Pattern) -> str:
    """Delete each parenthetical whose every ``;``/``,`` segment matches ``seg``; from a mixed one, delete only
    the matching segments."""
    def repl(m):
        lead, inner = m.group(1), m.group(2)
        parts = re.split(r"(\s*[;,]\s*)", inner)
        segs, seps = parts[0::2], parts[1::2]
        drop = [bool(seg.fullmatch(x.strip())) for x in segs]
        if not any(drop):
            return m.group(0)
        if all(drop):
            return ""
        kept = []
        for i, x in enumerate(segs):
            if drop[i]:
                continue
            if kept:
                kept.append(seps[i - 1] if i - 1 < len(seps) else ", ")
            kept.append(x)
        return lead + "(" + "".join(kept).strip() + ")"
    return re.sub(r"(\s*)\(([^()]{1,400})\)", repl, text)


def _tidy(s: str) -> str:
    s = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", s)
    s = re.sub(r"\s+([.,;:!?])(?=\s|$)", r"\1", s)
    s = re.sub(r"([,;:])\1+", r"\1", s)
    s = re.sub(r"[,;:]\s*([.!?])(?=\s|$)", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s.rstrip()


def regex_rewrite(text: str, rules: Rules) -> str:
    """Apply the rewrite rules and the parenthetical rule to the comment or prose part of one line."""
    s = text
    if rules.paren_segment is not None:
        s = _strip_parens(s, rules.paren_segment)
    for _, rx, sub, cap in rules.rewrites:
        if cap:
            def repl(m, sub=sub):
                r = m.expand(sub)
                return r
            new = rx.sub(repl, s)
            if new != s:
                # capitalise the first letter that follows the replaced span at a sentence start
                new = re.sub(r"(^|[.!?]\s+|^#+\s*|^\s*)([a-z])", lambda mm: mm.group(1) + mm.group(2).upper(), new, count=1)
            s = new
        else:
            s = rx.sub(sub, s)
    return _tidy(s)


def _sane(before: str, after: str) -> bool:
    """A regex rewrite is kept only if it left no empty or unbalanced brackets, no stray punctuation and no
    broken quoting behind."""
    if re.search(r"\(\s*\)|\[\s*\]", after):
        return False
    for o, c in ("()", "[]"):
        if after.count(o) - after.count(c) != before.count(o) - before.count(c):
            return False
    for q in ('"', "`"):
        if before.count(q) % 2 == 0 and after.count(q) % 2 == 1:
            return False
    stray = re.compile(r"(^|\s)[,;:.](\s|$)|[,;:]\s*$|\(\s*[,;:]")
    if stray.search(after) and not stray.search(before):
        return False
    runs = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")  # column alignment inside the comment (tables, boxes)
    if len(runs.findall(after)) < len(runs.findall(before)):
        return False
    return True


# ---------------------------------------------------------------------------------------------------------------
# scan: units for rewriting
# ---------------------------------------------------------------------------------------------------------------

def iter_files(tree: str, skip: Sequence[str]) -> Iterable[str]:
    for dp, dn, fn in os.walk(tree):
        dn[:] = sorted(d for d in dn if d != ".git")
        for f in sorted(fn):
            rel = os.path.relpath(os.path.join(dp, f), tree)
            if not path_skipped(rel, skip):
                yield rel


def _units(lang: str, a: Analysis, lines: List[str]) -> List[Tuple[int, int]]:
    """Group editable lines into units: a run of comment lines (optionally led by one trailing-comment line), a
    docstring paragraph, or a prose paragraph. Returns half-open ``(start, end)`` line ranges."""
    out = []
    i, n = 0, len(lines)
    while i < n:
        k = a.kinds[i]
        if k in (COMMENT, TRAIL):
            j = i + 1
            col = a.comment_col[i]
            while j < n and a.kinds[j] == COMMENT and (lang != "python" or a.comment_col[j] == col or k == COMMENT):
                j += 1
            out.append((i, j))
            i = j
        elif k in (DOC, PROSE):
            j = i + 1
            while j < n and a.kinds[j] == k and lines[j].strip() not in ("", '"""', "'''"):
                j += 1
            if j < n and a.kinds[j] == k and lines[j].strip() in ('"""', "'''"):
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def comment_text(line: str, kind: str, col: Optional[int]) -> str:
    if kind == TRAIL and col is not None:
        return line[col:]
    return line


def cmd_scan(args) -> int:
    rules = load_rules(args.rules)
    skip = read_globs(args.skip_file or [], args.skip or [])
    doc_protect = read_globs(args.doc_protect or [], [])
    out = open(args.out, "w", encoding="utf-8")
    unsupported = {}
    n_units = n_lines = 0
    for rel in iter_files(args.tree, skip):
        full = os.path.join(args.tree, rel)
        try:
            with open(full, encoding="utf-8", newline="") as f:
                text = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        lang = lang_of(rel, text)
        if lang is None:
            if args.unsupported and any(flag_classes(ln, rules) for ln in text.splitlines()):
                unsupported[rel] = "type"
            continue
        try:
            a = ANALYZERS[lang](text)
        except Unsupported as e:
            if args.unsupported and any(flag_classes(ln, rules) for ln in text.splitlines()):
                unsupported[rel] = str(e)
            continue
        lines, _ = split_lines(text)
        prot = doc_protection(rel, doc_protect)
        for s, e in _units(lang, a, lines):
            if a.kinds[s] == DOC and (prot is True or (prot and a.doc_owner and a.doc_owner[s] in prot)):
                continue
            flagged, classes = [], set()
            for i in range(s, e):
                c = flag_classes(comment_text(lines[i], a.kinds[i], a.comment_col[i]), rules)
                if c:
                    flagged.append(i - s)
                    classes.update(c)
            if not flagged:
                continue
            u = {"path": rel, "lang": lang, "kind": a.kinds[s] if a.kinds[s] != TRAIL else COMMENT,
                 "start": s + 1, "lines": lines[s:e], "flagged": flagged, "classes": sorted(classes),
                 "kinds": a.kinds[s:e], "comment_col": a.comment_col[s:e],
                 "before": lines[max(0, s - args.context):s], "after": lines[e:e + args.context]}
            out.write(json.dumps(u) + "\n")
            n_units += 1
            n_lines += len(flagged)
    out.close()
    if args.unsupported:
        with open(args.unsupported, "w", encoding="utf-8") as f:
            for rel, why in sorted(unsupported.items()):
                f.write(f"{rel}\t{why}\n")
    print(json.dumps({"units": n_units, "flagged_lines": n_lines, "unsupported_files_with_hits": len(unsupported)}))
    return 0


def cmd_regex(args) -> int:
    rules = load_rules(args.rules)
    rows_out = open(args.rows, "w", encoding="utf-8")
    resid_out = open(args.residual, "w", encoding="utf-8")
    n_rows = n_res = 0
    with open(args.units, encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)
            new_lines, ok = [], True
            for i, ln in enumerate(u["lines"]):
                if i not in u["flagged"]:
                    new_lines.append(ln)
                    continue
                kind, col = u["kinds"][i], u["comment_col"][i]
                head, body = (ln[:col], ln[col:]) if (kind == TRAIL and col is not None) else ("", ln)
                nb = regex_rewrite(body, rules)
                if flag_classes(nb, rules) or not _sane(body, nb):
                    ok = False
                    break
                marker = re.match(r"^(\s*)(#+|--)\s*$", nb)
                if kind == TRAIL and (marker or nb.strip() in ("#", "--")):
                    new_lines.append(head.rstrip())
                elif kind in (COMMENT,) and (marker or nb.strip() == ""):
                    continue  # the comment line held nothing but process text: drop it
                elif kind in (DOC, PROSE) and nb.strip() == "":
                    if any(q in ln for q in ('"""', "'''")):
                        ok = False
                        break
                    continue
                else:
                    new_lines.append(head + nb if head else nb)
            if ok:
                old = u["lines"]
                a, b = 0, 0
                while a < min(len(old), len(new_lines)) and old[a] == new_lines[a]:
                    a += 1
                while b < min(len(old), len(new_lines)) - a and old[-1 - b] == new_lines[-1 - b]:
                    b += 1
                o, nw = old[a:len(old) - b], new_lines[a:len(new_lines) - b]
                if not o:
                    ok = False
                else:
                    rows_out.write(json.dumps({"path": u["path"], "old": o, "new": nw, "origin": "regex"}) + "\n")
                    n_rows += 1
                    continue
            u["hint"] = None
            resid_out.write(json.dumps(u) + "\n")
            n_res += 1
    rows_out.close()
    resid_out.close()
    print(json.dumps({"regex_rows": n_rows, "units_for_rewrite": n_res}))
    return 0


# ---------------------------------------------------------------------------------------------------------------
# protect: files a table should leave alone
# ---------------------------------------------------------------------------------------------------------------

HEX12 = re.compile(rb"[0-9a-f]{12,64}")
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _module_index(tree: str, files: Sequence[str]) -> Dict[str, str]:
    idx = {}
    for rel in files:
        if not rel.endswith(".py"):
            continue
        parts = rel[:-3].split("/")
        for root in ("src", "corpus_generator/src", ""):
            rp = root.split("/") if root else []
            if parts[:len(rp)] == rp:
                mod = parts[len(rp):]
                if mod and mod[-1] == "__init__":
                    mod = mod[:-1]
                if mod:
                    idx.setdefault(".".join(mod), rel)
    return idx


def cmd_protect(args) -> int:
    files = list(iter_files(args.tree, []))
    reasons: Dict[str, List[str]] = {}
    # 1. hash pins: a file whose sha256 or git blob id (first 12 hex) appears in another file
    pre: Dict[bytes, set] = {}
    blobs = {}
    for rel in files:
        with open(os.path.join(args.tree, rel), "rb") as f:
            b = f.read()
        blobs[rel] = b
        for m in set(HEX12.findall(b)):
            pre.setdefault(m[:12], set()).add(rel)
    for rel, b in blobs.items():
        s = hashlib.sha256(b).hexdigest()
        if s == EMPTY_SHA256:
            continue
        g = hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()
        for kind, h in (("sha256", s), ("blob", g)):
            where = pre.get(h[:12].encode(), set()) - {rel}
            if where:
                reasons.setdefault(rel, []).append(f"{kind} pinned in {sorted(where)[0]}" + (f" (+{len(where) - 1})" if len(where) > 1 else ""))
    # 2. docstrings read as data, and source read by hashing: resolve module aliases from imports
    idx = _module_index(args.tree, files)
    stems: Dict[str, List[str]] = {}
    for f in files:
        if f.endswith(".py"):
            stems.setdefault(os.path.basename(f)[:-3], []).append(f)
    for rel in files:
        if not rel.endswith(".py"):
            continue
        src = blobs[rel].decode("utf-8", errors="replace")
        if "__doc__" not in src and "getsource" not in src and "getdoc" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        alias: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    alias[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom) and node.module:
                for a in node.names:
                    alias[a.asname or a.name] = f"{node.module}.{a.name}"
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fn = node.value.func
                fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if fname in ("import_module", "_load", "load_module", "_import", "_mod", "_load_script", "load"):
                    if node.value.args and isinstance(node.value.args[0], ast.Constant) and isinstance(node.value.args[0].value, str):
                        for t in node.targets:
                            if isinstance(t, ast.Name):
                                alias[t.id] = node.value.args[0].value
        hashed_source = "getsource" in src and re.search(r"sha256\([^)]*getsource|getsource\([^)]*\)\)?\.encode", src)

        def target_file(expr) -> Optional[str]:
            base = expr
            while isinstance(base, ast.Attribute):
                base = base.value
            if not isinstance(base, ast.Name):
                return None
            mod = alias.get(base.id)
            if mod is None:
                return None
            if mod.endswith(".py") or "/" in mod:
                m = mod.lstrip("./")
                return m if m in blobs else None
            while mod:
                if mod in idx:
                    return idx[mod]
                mod = mod.rsplit(".", 1)[0] if "." in mod else ""
            return None

        def named_files() -> List[str]:
            """Every tree file this reader names: string constants ending in ``.py`` (matched by path suffix)
            and bare imports that map to exactly one file stem."""
            out = set()
            for nd in ast.walk(tree):
                if isinstance(nd, ast.Constant) and isinstance(nd.value, str) and nd.value.endswith(".py"):
                    tail = nd.value.lstrip("./")
                    out.update(f for f in blobs if f == tail or f.endswith("/" + tail))
                elif isinstance(nd, ast.Import):
                    for a in nd.names:
                        hits = stems.get(a.name.split(".")[-1], [])
                        if len(hits) == 1:
                            out.add(hits[0])
            return sorted(out)

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "__doc__" and isinstance(node.ctx, ast.Load):
                tf = target_file(node.value)
                if tf and tf != rel:
                    reasons.setdefault(tf, []).append(f"docstring read as data by {rel}:{node.lineno}")
                elif tf is None and not (isinstance(node.value, ast.Name) and node.value.id in ("self", "cls")):
                    for nf in named_files():
                        if nf != rel:
                            reasons.setdefault(nf, []).append(f"docstring possibly read as data by {rel}:{node.lineno}")
            if isinstance(node, ast.Call):
                fn = node.func
                fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if fname in ("getdoc",) and node.args:
                    tf = target_file(node.args[0])
                    if tf and tf != rel:
                        reasons.setdefault(tf, []).append(f"docstring read by {rel}:{node.lineno}")
                if fname == "getsource" and node.args and hashed_source:
                    tf = target_file(node.args[0]) or rel
                    reasons.setdefault(tf, []).append(f"source hashed by {rel}:{node.lineno}")
    with open(args.out, "w", encoding="utf-8") as f:
        for rel in sorted(reasons):
            f.write(f"{rel}\t{'; '.join(sorted(set(reasons[rel])))}\n")
    print(json.dumps({"protected": len(reasons)}))
    return 0


# ---------------------------------------------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="list flagged comment, docstring and prose units of a tree")
    s.add_argument("--tree", required=True)
    s.add_argument("--rules", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--skip", action="append", help="path glob to leave out (repeatable)")
    s.add_argument("--skip-file", action="append", help="file of path globs, one per line (first tab field)")
    s.add_argument("--context", type=int, default=3)
    s.add_argument("--doc-protect", action="append", help="file of path globs whose docstrings are read as data")
    s.add_argument("--unsupported", help="write files with hits that no checker covers")

    r = sub.add_parser("regex", help="the regex pass over scanned units")
    r.add_argument("--rules", required=True)
    r.add_argument("--units", required=True)
    r.add_argument("--rows", required=True)
    r.add_argument("--residual", required=True)

    for name, hlp in (("check", "validate a table against a tree (writes nothing)"),
                      ("apply", "apply a table to a tree in place")):
        c = sub.add_parser(name, help=hlp)
        c.add_argument("--tree", required=True)
        c.add_argument("--table", required=True)
        c.add_argument("--skip", action="append", help="path glob whose rows are skipped (repeatable)")
        c.add_argument("--skip-file", action="append", help="file of path globs, one per line (first tab field)")
        c.add_argument("--report", help="write the residue as JSON Lines here instead of stdout")
        if name == "check":
            c.add_argument("--allow-residue", action="store_true", help="exit 0 even when rows are residue")
        else:
            c.add_argument("--strict", action="store_true", help="exit 1 if any row is residue")
        c.add_argument("--doc-protect", action="append", help="file of path globs whose docstrings are read as data")
        c.add_argument("--applied", help="write the rows that applied (as JSON Lines, unchanged) here")

    p = sub.add_parser("protect", help="list files a table should leave alone")
    p.add_argument("--tree", required=True)
    p.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "scan":
        return cmd_scan(args)
    if args.cmd == "regex":
        return cmd_regex(args)
    if args.cmd == "protect":
        return cmd_protect(args)
    skip = read_globs(args.skip_file or [], args.skip or [])
    n_residue = run_table(args.tree, args.table, skip, write=(args.cmd == "apply"), report=args.report,
                          doc_protect=read_globs(args.doc_protect or [], []), applied_out=args.applied)
    if args.cmd == "check":
        return 1 if n_residue and not args.allow_residue else 0
    return 1 if n_residue and args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
