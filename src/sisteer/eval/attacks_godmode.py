"""G0DM0D3 jailbreak templates for DEFENSIVE distillation, fetched at run time.

Source: elder-plinius/G0DM0D3, ``src/lib/libertas.ts``
(https://github.com/elder-plinius/G0DM0D3), entries ``grok-420`` and
``hermes-fast``. Author: elder-plinius (Pliny the Liberator), also of the
L1B3RT4S prompt corpus. Upstream licence: **AGPL-3.0**.

The template text is not distributed with this repository. It is fetched at
run time from the pinned upstream commit under upstream's own licence
(AGPL-3.0) and is not covered by this repository's MIT licence.

How the text arrives. The first ``render()`` (or ``load_templates()``) reads
``libertas.ts`` from the third-party cache, downloading it on a miss from the
commit-pinned raw URL (Software Heritage serves the same blob as the second
source). Before the file can be used it must match ``FILE_SHA256``; a
download that does not is never written. The two entries are then parsed out
of the file AS TEXT (a small hand-written reader of the TypeScript literals;
nothing is executed) and each field must match its own sha256 below, so the
attack's identity is pinned independently of the file and of the parser. The
cache is ``$SISTEER_THIRD_PARTY_DIR`` (a cluster's shared root, since compute
nodes have no egress and node-local disk is not shared), else
``$XDG_CACHE_HOME/sisteer/third_party``, else ``~/.cache/sisteer/third_party``.
``SISTEER_THIRD_PARTY_OFFLINE=1`` forbids the download. Every failure (absent
offline, every source failing, a hash mismatch, a parser miss) raises
``GodmodeTemplateError`` naming what failed and how to fix it; there is no
stored text to fall back to. Importing this module reads nothing and touches
no network. Pre-fetch with ``python -m sisteer.eval.attacks_godmode fetch``
(``reproduce/00_setup.sh --what third_party`` installs the verified file on
every cluster).

Why these two and not the rest: measured at n=25 under StrongREJECT,
``grok420`` is the strongest template in the corpus and
``hermes_fast`` the only other nonzero one; they share the SAME mechanism
(invert-the-refusal). The four pure-persona templates and every
Parseltongue obfuscation scored 0/25 -- no training signal, deliberately
not used.

DEPLOYMENT CONVENTION (recorded because it changes what the attack IS):
upstream ships a {system, user} pair, but these are folded into a SINGLE
USER TURN as ``system + "\\n\\n" + user``. The target's
generate path presents one user prompt, and the defended cell folds any
attack-supplied system content into the user turn anyway -- so an
unfolded two-turn rendering would measure a different attack than the one
deployed.

The templates are read from libertas.ts, NOT from an earlier Python port:
that port normalises one of upstream's placeholders, which is functionally
identical under its own injector but is not upstream's text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

UPSTREAM_REPO = "elder-plinius/G0DM0D3"     # AGPL-3.0; NOT distributed with this repository
UPSTREAM_LICENSE = "AGPL-3.0"
UPSTREAM_COMMIT = "f6301765fb90eb7b336bdf365319cd2fe44b1187"
UPSTREAM_PATH = "src/lib/libertas.ts"
PERMALINK = f"https://github.com/{UPSTREAM_REPO}/blob/{UPSTREAM_COMMIT}/{UPSTREAM_PATH}"
#: git blob id of the file (unchanged since it was added upstream); Software Heritage addresses the content by it
FILE_SHA1_GIT = "fa2ab8105db91fe2bd012ffedec6ff2326ffab50"
FILE_SHA256 = "53600aa3d83012759d186b048155f57b96d022e3f835c02a499c9205615792a5"
FILE_BYTES = 9281
#: tried in order; the sha256 check makes the source irrelevant to what is used
SOURCES = (
    f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/{UPSTREAM_PATH}",
    f"https://archive.softwareheritage.org/api/1/content/sha1_git:{FILE_SHA1_GIT}/raw/",
)
#: a redirect or final URL outside these hosts is refused before a byte is read
ALLOWED_HOSTS = frozenset({"raw.githubusercontent.com", "archive.softwareheritage.org"})
#: where the file lives under a third-party root (the same relative path on every host and cluster)
CACHE_RELPATH = PurePosixPath("G0DM0D3") / UPSTREAM_COMMIT / UPSTREAM_PATH
THIRD_PARTY_DIR_ENV = "SISTEER_THIRD_PARTY_DIR"
OFFLINE_ENV = "SISTEER_THIRD_PARTY_OFFLINE"
FETCH_COMMAND = "python -m sisteer.eval.attacks_godmode fetch"
_MAX_DOWNLOAD_BYTES = 4 * FILE_BYTES
_TIMEOUT_S = 30

#: The two templates' IDENTITY -- no text. sha256 of the UTF-8 of each decoded field and of the folded single turn
#: (``system + "\n\n" + user``). Every recorded grok420 run used exactly these (one module blob across the history,
#: equal to upstream), so they are also the values a registry ``prompt_template_sha256`` would carry.
TEMPLATES = {
    "grok420": {
        "ts_id": "grok-420",
        "codename": "GROK 4.20",
        "system_sha256": "70c4fe49b078552957383e658d60cbc1cf31976da3d7373310326955270894f5",
        "user_sha256": "5d08fc004efaf11eb65ef8b0a5aacbdf8153e71246a5a52456e14ac44fefc7a0",
        "folded_sha256": "8e1a7bacec31724bbd945032da3c728c75152ff11fed59a068d172227f5fbd05",
    },
    "hermes_fast": {
        "ts_id": "hermes-fast",
        "codename": "GODMODE FAST",
        "system_sha256": "b33e83ddef622c8931c0aa811558dd6522b9ac1ef4e6a4019ffe9fe03a39d5b2",
        "user_sha256": "aefb9a6a51ae4157d3731b8e1fa7f11467c043e785ef2a364a0b852c682cc6f0",
        "folded_sha256": "bdfc1b2583618a709d3a4bfd2f985ad9ffcb677c73e8e0d73182d19e63bd3046",
    },
}
#: sha256 of the canonical JSON ``{key: folded}`` (sorted keys, ensure_ascii=False, compact separators)
SET_SHA256 = "916168681aeb25c7abbd3de7513b81420e70566ac53be21c7d319d112b513579"
#: the public name callers sort and look keys up in (grok420.py renders every key); identity only, never text
GODMODE_TEMPLATES = TEMPLATES

# Upstream placeholder spellings. grok-420 carries both; hermes-fast only
# the first. Substituting BOTH keeps one code path for the two templates.
_SLOTS = ("{QUERY}", "{Z}")


class GodmodeTemplateError(RuntimeError):
    """The G0DM0D3 templates could not be made available exactly as pinned (refused by name, never degraded)."""


def _sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def folded(system: str, user: str) -> str:
    """The deployed single user turn (the DEPLOYMENT CONVENTION above)."""
    return system + "\n\n" + user


def template_set_sha256(templates: dict[str, dict[str, str]]) -> str:
    """sha256 of the canonical JSON of every template's folded text (``SET_SHA256``'s definition)."""
    doc = {k: folded(v["system"], v["user"]) for k, v in templates.items()}
    return _sha256(json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


# ---------------------------------------------------------------------------------------------------------- the cache

def third_party_root(cache_dir: str | os.PathLike | None = None) -> Path:
    """An explicit ``cache_dir`` > ``$SISTEER_THIRD_PARTY_DIR`` > ``$XDG_CACHE_HOME/sisteer/third_party`` >
    ``~/.cache/sisteer/third_party``."""
    if cache_dir:
        return Path(cache_dir).expanduser()
    env = os.environ.get(THIRD_PARTY_DIR_ENV)
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "sisteer" / "third_party"


def cache_path(cache_dir: str | os.PathLike | None = None) -> Path:
    """Where ``libertas.ts`` at the pinned commit lives under the third-party root."""
    return third_party_root(cache_dir).joinpath(*CACHE_RELPATH.parts)


def offline_mode(offline: bool | None = None) -> bool:
    """An explicit argument wins; else ``SISTEER_THIRD_PARTY_OFFLINE`` set to anything but empty, ``0``, ``false``, ``no``."""
    if offline is not None:
        return bool(offline)
    return os.environ.get(OFFLINE_ENV, "").strip().lower() not in ("", "0", "false", "no")


def _check_bytes(data: bytes, where: str) -> None:
    got = _sha256(data)
    if got != FILE_SHA256:
        raise GodmodeTemplateError(
            f"G0DM0D3 {UPSTREAM_PATH} from {where} is not the pinned file: sha256 {got} ({len(data)} bytes), "
            f"expected {FILE_SHA256} ({FILE_BYTES} bytes) from {UPSTREAM_REPO}@{UPSTREAM_COMMIT}")


def _require_allowed(url: str, *, why: str) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise GodmodeTemplateError(f"{why}: {url!r} is not an https URL on {sorted(ALLOWED_HOSTS)}")


class _RedirectGuard(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to an allowed https host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _require_allowed(newurl, why=f"redirect {code} from {req.full_url}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_url(url: str):
    """The one place a network connection is opened (the tests replace it)."""
    req = urllib.request.Request(url, headers={"User-Agent": "sisteer-third-party-fetch"})
    return urllib.request.build_opener(_RedirectGuard()).open(req, timeout=_TIMEOUT_S)


def _download(url: str) -> bytes:
    _require_allowed(url, why="source")
    with _open_url(url) as resp:
        _require_allowed(resp.geturl(), why="final URL")
        data = resp.read(_MAX_DOWNLOAD_BYTES + 1)
    if len(data) > _MAX_DOWNLOAD_BYTES:
        raise GodmodeTemplateError(f"{url} returned more than {_MAX_DOWNLOAD_BYTES} bytes (the pinned file is {FILE_BYTES})")
    return data


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def ensure_file(cache_dir: str | os.PathLike | None = None, *, offline: bool | None = None) -> tuple[str, Path, bytes, str]:
    """Make the pinned ``libertas.ts`` present and verified. Returns ``(status, path, bytes, source)`` with status
    ``PRESENT`` (the cached copy verified) or ``FETCHED`` (downloaded, verified, then renamed into place). Refuses by
    name: a cached copy that is not the pinned file (left where it is for the operator to inspect), an absent copy in
    offline mode, or every source failing (each source's reason listed). A download is checked BEFORE the atomic
    rename, so a bad or interrupted download never lands in the cache."""
    path = cache_path(cache_dir)
    if path.is_file():
        data = path.read_bytes()
        try:
            _check_bytes(data, str(path))
        except GodmodeTemplateError as exc:
            raise GodmodeTemplateError(f"{exc}. Remove it and run `{FETCH_COMMAND}` again.") from None
        return "PRESENT", path, data, "cache"
    if offline_mode(offline):
        raise GodmodeTemplateError(
            f"G0DM0D3 {UPSTREAM_PATH} is not in the third-party cache ({path}) and fetching is off "
            f"({OFFLINE_ENV}=1 or --offline). Fetch it on a host with network access: `{FETCH_COMMAND}` "
            f"(or `reproduce/00_setup.sh --what third_party` for every cluster), or point {THIRD_PARTY_DIR_ENV} "
            f"at a root that holds {CACHE_RELPATH}.")
    failures = []
    for url in SOURCES:
        try:
            data = _download(url)
            _check_bytes(data, url)
        except (GodmodeTemplateError, urllib.error.URLError, OSError, ValueError) as exc:
            failures.append(f"{url}: {exc}")
            continue
        _write_atomic(path, data)
        return "FETCHED", path, data, url
    raise GodmodeTemplateError(
        f"G0DM0D3 {UPSTREAM_PATH} could not be fetched from any source (nothing was written to {path}):\n  "
        + "\n  ".join(failures)
        + f"\nPlace a copy whose sha256 is {FILE_SHA256} at that path, or run `{FETCH_COMMAND}` where the network allows it.")


# ------------------------------------------------------------------------------------------ the TypeScript reader
# libertas.ts is untrusted text: it is read, never executed, and only after its sha256 matched. The reader accepts the
# subset the file uses -- one ``HALL_OF_FAME ... = [`` array of object literals whose values are quoted strings,
# template literals (with no ``${}``), booleans and numbers, with // and /* */ comments -- and refuses anything else.

_HALL_ANCHOR = re.compile(r"\bHALL_OF_FAME\b[^=\n]*=\s*\[")
_IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_SIMPLE_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v"}
_LINE_TERMINATORS = ("\n", "\r", " ", " ")


class _Reader:
    def __init__(self, text: str, origin: str):
        self.s, self.i, self.origin = text, 0, origin

    def fail(self, what: str, at: int | None = None) -> GodmodeTemplateError:
        at = self.i if at is None else at
        line = self.s.count("\n", 0, at) + 1
        col = at - (self.s.rfind("\n", 0, at) + 1) + 1
        return GodmodeTemplateError(f"{self.origin}: cannot read the HALL_OF_FAME entries: {what} at line {line}, column {col}")

    def skip(self) -> None:
        s, n = self.s, len(self.s)
        while self.i < n:
            if s[self.i].isspace():
                self.i += 1
            elif s.startswith("//", self.i):
                j = s.find("\n", self.i)
                self.i = n if j < 0 else j + 1
            elif s.startswith("/*", self.i):
                j = s.find("*/", self.i + 2)
                if j < 0:
                    raise self.fail("an unterminated /* comment")
                self.i = j + 2
            else:
                return

    def peek(self) -> str:
        self.skip()
        return self.s[self.i] if self.i < len(self.s) else ""

    def expect(self, ch: str) -> None:
        if self.peek() != ch:
            raise self.fail(f"expected {ch!r}, found {self.peek() or 'the end of the file'!r}")
        self.i += 1

    def _escape(self, out: list[str]) -> None:
        """At a backslash: append the cooked character of one JS escape (none for a line continuation) and advance."""
        s, i = self.s, self.i
        if i + 1 >= len(s):
            raise self.fail("a backslash at the end of the file", i)
        c = s[i + 1]
        if c in _LINE_TERMINATORS:                          # line continuation: contributes nothing
            self.i = i + (3 if s.startswith("\r\n", i + 1) else 2)
            return
        if c in _SIMPLE_ESCAPES:
            out.append(_SIMPLE_ESCAPES[c])
            self.i = i + 2
            return
        if c == "0" and not (i + 2 < len(s) and s[i + 2].isdigit()):
            out.append("\0")
            self.i = i + 2
            return
        if c.isdigit():
            raise self.fail(f"a legacy octal escape \\{c}", i)
        if c == "x":
            hexd = s[i + 2:i + 4]
            if not re.fullmatch(r"[0-9A-Fa-f]{2}", hexd):
                raise self.fail("a malformed \\x escape", i)
            out.append(chr(int(hexd, 16)))
            self.i = i + 4
            return
        if c == "u":
            if s.startswith("{", i + 2):
                j = s.find("}", i + 3)
                hexd = s[i + 3:j] if j > 0 else ""
                if not re.fullmatch(r"[0-9A-Fa-f]{1,6}", hexd) or int(hexd, 16) > 0x10FFFF:
                    raise self.fail("a malformed \\u{...} escape", i)
                out.append(chr(int(hexd, 16)))
                self.i = j + 1
                return
            hexd = s[i + 2:i + 6]
            if not re.fullmatch(r"[0-9A-Fa-f]{4}", hexd):
                raise self.fail("a malformed \\u escape", i)
            out.append(chr(int(hexd, 16)))
            self.i = i + 6
            return
        out.append(c)                                       # an identity escape: \\ \` \$ \' \" and the rest
        self.i = i + 2

    def _finish(self, chars: list[str], start: int) -> str:
        """JS strings are UTF-16: pair the surrogates escapes may spell; a lone surrogate is refused."""
        text = "".join(chars)
        try:
            return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le")
        except UnicodeDecodeError:
            raise self.fail("a lone UTF-16 surrogate", start) from None

    def quoted(self) -> str:
        start, quote = self.i, self.s[self.i]
        self.i += 1
        out: list[str] = []
        while True:
            if self.i >= len(self.s):
                raise self.fail("an unterminated string", start)
            c = self.s[self.i]
            if c == quote:
                self.i += 1
                return self._finish(out, start)
            if c == "\\":
                self._escape(out)
            elif c in "\n\r":
                raise self.fail("a line break inside a quoted string", self.i)
            else:
                out.append(c)
                self.i += 1

    def template(self) -> str:
        start = self.i
        self.i += 1
        out: list[str] = []
        while True:
            if self.i >= len(self.s):
                raise self.fail("an unterminated template literal", start)
            c = self.s[self.i]
            if c == "`":
                self.i += 1
                return self._finish(out, start)
            if c == "\\":
                self._escape(out)
            elif c == "$" and self.s.startswith("${", self.i):
                raise self.fail("an interpolation ${...} inside a template literal", self.i)
            elif c == "\r":                                 # JS normalises CR LF and CR to LF in a template's value
                out.append("\n")
                self.i += 2 if self.s.startswith("\r\n", self.i) else 1
            else:
                out.append(c)
                self.i += 1

    def value(self):
        c = self.peek()
        if c in ("'", '"'):
            return self.quoted()
        if c == "`":
            return self.template()
        m = _IDENT.match(self.s, self.i)
        if m and m.group(0) in ("true", "false"):
            self.i = m.end()
            return m.group(0) == "true"
        m = _NUMBER.match(self.s, self.i)
        if m:
            self.i = m.end()
            return float(m.group(0)) if "." in m.group(0) else int(m.group(0))
        raise self.fail("a value that is not a string, template literal, boolean or number")

    def obj(self) -> dict:
        self.expect("{")
        entry: dict = {}
        while True:
            c = self.peek()
            if c == "}":
                self.i += 1
                return entry
            at = self.i
            if c in ("'", '"'):
                key = self.quoted()
            else:
                m = _IDENT.match(self.s, self.i)
                if not m:
                    raise self.fail("expected a property name")
                key, self.i = m.group(0), m.end()
            if key in entry:
                raise self.fail(f"a duplicate property {key!r}", at)
            self.expect(":")
            entry[key] = self.value()
            if self.peek() == ",":
                self.i += 1
            elif self.peek() != "}":
                raise self.fail(f"expected ',' or '}}' after property {key!r}")

    def hall_of_fame(self) -> list[dict]:
        found = list(_HALL_ANCHOR.finditer(self.s))
        if len(found) != 1:
            raise GodmodeTemplateError(f"{self.origin}: expected one `HALL_OF_FAME ... = [` array, found {len(found)}")
        self.i = found[0].end()
        entries = []
        while True:
            if self.peek() == "]":
                return entries
            entries.append(self.obj())
            if self.peek() == ",":
                self.i += 1
            elif self.peek() != "]":
                raise self.fail("expected ',' or ']' after an entry")


def parse_hall_of_fame(text: str, origin: str = UPSTREAM_PATH) -> list[dict]:
    """Every ``HALL_OF_FAME`` entry of a libertas.ts source as a dict of its cooked property values."""
    return _Reader(text, origin).hall_of_fame()


# ------------------------------------------------------------------------------------------------- the templates

_LOADED: dict[tuple, dict[str, dict[str, str]]] = {}


def _pins_key() -> tuple:
    return (FILE_SHA256, SET_SHA256,
            tuple(sorted((k, tuple(sorted(v.items()))) for k, v in TEMPLATES.items())))


def clear_cache() -> None:
    """Forget every loaded template set (the next load reads and verifies the file again)."""
    _LOADED.clear()


def _templates_from(data: bytes, origin: str) -> dict[str, dict[str, str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GodmodeTemplateError(f"{origin}: not UTF-8 ({exc})") from None
    entries = parse_hall_of_fame(text, origin)
    out: dict[str, dict[str, str]] = {}
    for key, ident in TEMPLATES.items():
        hits = [e for e in entries if e.get("id") == ident["ts_id"]]
        if len(hits) != 1:
            raise GodmodeTemplateError(
                f"{origin}: template {key!r} expects exactly one HALL_OF_FAME entry with id {ident['ts_id']!r}, found {len(hits)}")
        entry = hits[0]
        spec = {"ts_id": ident["ts_id"], "codename": entry.get("codename")}
        for field in ("system", "user"):
            if not isinstance(entry.get(field), str):
                raise GodmodeTemplateError(f"{origin}: template {key!r} (id {ident['ts_id']!r}) has no string field {field!r}")
            spec[field] = entry[field]
        if ident.get("codename") is not None and spec["codename"] != ident["codename"]:
            raise GodmodeTemplateError(
                f"{origin}: template {key!r} has codename {spec['codename']!r}, expected {ident['codename']!r}")
        got = {"system": _sha256(spec["system"]), "user": _sha256(spec["user"]),
               "folded": _sha256(folded(spec["system"], spec["user"]))}
        for field, sha in got.items():
            want = ident[f"{field}_sha256"]
            if sha != want:
                raise GodmodeTemplateError(
                    f"{origin}: template {key!r} field {field!r} has sha256 {sha}, expected {want} "
                    f"(the parser or the pin is wrong; the file itself matched its sha256)")
        out[key] = spec
    got_set = template_set_sha256(out)
    if got_set != SET_SHA256:
        raise GodmodeTemplateError(f"{origin}: the template set has sha256 {got_set}, expected {SET_SHA256}")
    return out


def load_templates(cache_dir: str | os.PathLike | None = None, *, offline: bool | None = None) -> dict[str, dict[str, str]]:
    """``{key: {ts_id, codename, system, user}}`` for every key of ``TEMPLATES``, from the verified cached file
    (fetched on a miss unless offline). Loaded once per (file, pins) per process; refuses by name on any failure."""
    key = (str(cache_path(cache_dir)), _pins_key())
    if key not in _LOADED:
        _status, path, data, _source = ensure_file(cache_dir, offline=offline)
        _LOADED[key] = _templates_from(data, str(path))
    return {k: dict(v) for k, v in _LOADED[key].items()}


def render(template: str, goal: str) -> str:
    """One attack prompt: the goal spliced into ``template``'s slot, then
    system and user folded into a single user turn.

    Fails loudly on an unknown template or a template with no slot -- a
    silently un-substituted scaffold would train against the attack's
    boilerplate with no harmful behaviour in it, which looks like a
    successful defense and is not one. An unknown key refuses before
    anything is read.
    """
    if template not in GODMODE_TEMPLATES:
        raise KeyError(
            f"unknown G0DM0D3 template {template!r}; have "
            f"{sorted(GODMODE_TEMPLATES)}")
    spec = load_templates()[template]
    text = folded(spec["system"], spec["user"])
    if not any(s in text for s in _SLOTS):
        raise ValueError(
            f"{template!r} carries no goal slot {_SLOTS} -- it is a fixed "
            f"prompt, not a transform, and cannot take a behaviour")
    for slot in _SLOTS:
        text = text.replace(slot, goal)
    return text


def grok420(goal: str) -> str:
    """The strongest G0DM0D3 template: semantic inversion with dividers."""
    return render("grok420", goal)


def hermes_fast(goal: str) -> str:
    """The other nonzero template; same invert-the-refusal mechanism."""
    return render("hermes_fast", goal)


# ---------------------------------------------------------------------------------------------------------- the CLI

def report(status: str, path: Path, data: bytes, source: str, templates: dict[str, dict[str, str]]) -> dict:
    """The fetch report: where the file is, what it is, and every template identity read from it."""
    return {
        "status": status, "path": str(path), "relpath": str(CACHE_RELPATH), "source": source,
        "upstream": {"repo": UPSTREAM_REPO, "commit": UPSTREAM_COMMIT, "path": UPSTREAM_PATH,
                     "license": UPSTREAM_LICENSE, "permalink": PERMALINK},
        "sha256": _sha256(data), "bytes": len(data),
        "templates": {k: {"ts_id": v["ts_id"], "system_sha256": _sha256(v["system"]), "user_sha256": _sha256(v["user"]),
                          "folded_sha256": _sha256(folded(v["system"], v["user"]))} for k, v in sorted(templates.items())},
        "set_sha256": template_set_sha256(templates),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m sisteer.eval.attacks_godmode",
        description=f"Fetch and verify the G0DM0D3 templates ({UPSTREAM_REPO} {UPSTREAM_PATH} @ {UPSTREAM_COMMIT[:12]}, "
                    f"{UPSTREAM_LICENSE}; not distributed with this repository).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="make the pinned file present in the third-party cache, then verify it and both templates")
    f.add_argument("--cache-dir", default=None,
                   help=f"third-party root (default ${THIRD_PARTY_DIR_ENV}, else $XDG_CACHE_HOME/sisteer/third_party)")
    f.add_argument("--offline", "--check", dest="offline", action="store_true", default=None,
                   help=f"verify the cached copy only: never download, never write (as {OFFLINE_ENV}=1); refuses when it is absent")
    f.add_argument("--json", action="store_true", help="print the report as one JSON object")
    args = ap.parse_args(argv)
    try:
        status, path, data, source = ensure_file(args.cache_dir, offline=args.offline)
        templates = _templates_from(data, str(path))
    except GodmodeTemplateError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    rep = report(status, path, data, source, templates)
    if args.json:
        print(json.dumps(rep, sort_keys=True))
        return 0
    print(f"{status} G0DM0D3 {UPSTREAM_PATH} @ {UPSTREAM_COMMIT[:12]} ({UPSTREAM_LICENSE}) from {source}")
    print(f"  path    {rep['path']}")
    print(f"  sha256  {rep['sha256']} ({rep['bytes']} bytes)")
    for k, t in rep["templates"].items():
        print(f"  {k:<12} folded {t['folded_sha256']}  system {t['system_sha256'][:16]}  user {t['user_sha256'][:16]}")
    print(f"  set     {rep['set_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
