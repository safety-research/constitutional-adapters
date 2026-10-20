"""The runtime reader of ``configs/SCRUB_MAP.tsv``, for verifiers whose recorded sha256 predates the public release.

The release rewrote some files whose sha256 a record had already captured: a config a run ran or was graded with, a
frozen copy, a harness manifest, a registry seed. Their bytes changed only by declared rules (site paths to
placeholders, comment text), and ``scripts/release/scrub_map.py --verify`` re-derived every row byte for byte from its
pre-image before the row was written. So a row certifies that the file whose sha256 is ``post_sha256`` today is the
rule-rewritten form of the bytes a record captured as ``pre_sha256``.

A verifier that finds a sha mismatch asks :func:`check` (or :func:`check_prefix` for a recorded prefix). When a row
certifies the pair, the verifier passes and reports :data:`VIA_MAP` (``verified_via_map``); any other mismatch still
refuses. The map is consulted only on a mismatch, never to bless a file no record names.

Format: UTF-8, tab-separated. Lines that start with ``#`` are comments. The first other line is the header, exactly
the six columns of :data:`COLUMNS` in order; each later line is one row. ``pre_sha256`` and ``post_sha256`` are 64
lowercase hex digits, ``rule_set`` is the sha256 of the private rules manifest the row was derived under, and ``kind``
is one of :data:`KINDS`. One pre-image may have several rows (byte-identical files at two paths). An absent map is an
empty map, so a tree without one refuses every mismatch as before; a malformed map raises :class:`ScrubMapError`.

The map is ``$SISTEER_SCRUB_MAP`` when set, else ``configs/SCRUB_MAP.tsv`` in the checkout this module belongs to.
:func:`verify_frame_config` answers the same question for a paper-frame row (its ``repo:<rel>`` config and recorded
sha256) against a checkout and that checkout's own map.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

COLUMNS: Tuple[str, ...] = ("pre_sha256", "pre_path", "path", "post_sha256", "rule_set", "kind")
KINDS: Tuple[str, ...] = ("config", "harness", "registry_seed", "frozen")
EXACT = "exact"
VIA_MAP = "verified_via_map"
REL_PATH = Path("configs") / "SCRUB_MAP.tsv"
MIN_PREFIX = 8

_SHA = re.compile(r"^[0-9a-f]{64}$")


class ScrubMapError(ValueError):
    """The scrub map exists but cannot be trusted: a bad header, a bad sha, an unknown kind, a short row."""


@dataclass(frozen=True)
class Row:
    pre_sha256: str
    pre_path: str
    path: str
    post_sha256: str
    rule_set: str
    kind: str


class ScrubMap:
    """The rows of one map file, indexed by pre-image sha256."""

    def __init__(self, rows: Iterable[Row], source: Optional[str] = None):
        self.source = source
        self.rows: Tuple[Row, ...] = tuple(rows)
        self._by_pre: Dict[str, List[Row]] = {}
        for r in self.rows:
            self._by_pre.setdefault(r.pre_sha256, []).append(r)

    def __len__(self) -> int:
        return len(self.rows)

    def rows_for(self, pre_sha256: str) -> Tuple[Row, ...]:
        return tuple(self._by_pre.get(str(pre_sha256 or "").lower(), ()))

    def certifies(self, recorded_sha256: str, current_sha256: str, *,
                  path: Optional[Union[str, os.PathLike]] = None) -> Optional[Row]:
        """The row certifying that ``current_sha256`` is the rewritten form of ``recorded_sha256``, else None. With
        ``path``, the row's repo-relative ``path`` must name the same file (equal, or the tail of an absolute path)."""
        cur = str(current_sha256 or "").lower()
        for r in self.rows_for(recorded_sha256):
            if r.post_sha256 == cur and _same_file(r.path, path):
                return r
        return None

    def certifies_prefix(self, recorded_prefix: str, current_sha256: str, *,
                         path: Optional[Union[str, os.PathLike]] = None) -> Optional[Row]:
        """:meth:`certifies` for a record that kept only a prefix of the sha256 (at least :data:`MIN_PREFIX` hex digits;
        a shorter prefix certifies nothing)."""
        pre = str(recorded_prefix or "").lower()
        if len(pre) < MIN_PREFIX:
            return None
        cur = str(current_sha256 or "").lower()
        for r in self.rows:
            if r.pre_sha256.startswith(pre) and r.post_sha256 == cur and _same_file(r.path, path):
                return r
        return None


def _same_file(row_path: str, path: Optional[Union[str, os.PathLike]]) -> bool:
    if path is None:
        return True
    p = os.path.normpath(os.fspath(path)).replace(os.sep, "/")
    rp = os.path.normpath(row_path).replace(os.sep, "/")
    return p == rp or p.endswith("/" + rp)


def default_path() -> Path:
    env = os.environ.get("SISTEER_SCRUB_MAP")
    if env:
        return Path(os.path.expanduser(env))
    return Path(__file__).resolve().parents[2] / REL_PATH


def parse(text: str, source: str = "<scrub map>") -> ScrubMap:
    """Rows of a map's text; raises :class:`ScrubMapError` naming the line on any defect."""
    rows: List[Row] = []
    header_seen = False
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.rstrip("\r").split("\t")
        if not header_seen:
            if tuple(cells) != COLUMNS:
                raise ScrubMapError(f"{source}:{n}: the header must be {' '.join(COLUMNS)} (tab-separated), got {cells}")
            header_seen = True
            continue
        if len(cells) != len(COLUMNS):
            raise ScrubMapError(f"{source}:{n}: {len(cells)} cells, expected {len(COLUMNS)}")
        row = Row(*cells)
        for col in ("pre_sha256", "post_sha256", "rule_set"):
            if not _SHA.match(getattr(row, col)):
                raise ScrubMapError(f"{source}:{n}: {col} {getattr(row, col)!r} is not 64 lowercase hex digits")
        if row.kind not in KINDS:
            raise ScrubMapError(f"{source}:{n}: kind {row.kind!r} is not one of {', '.join(KINDS)}")
        if not row.path or not row.pre_path:
            raise ScrubMapError(f"{source}:{n}: an empty path")
        if row.pre_sha256 == row.post_sha256:
            raise ScrubMapError(f"{source}:{n}: pre_sha256 equals post_sha256 (an unchanged file has no row)")
        rows.append(row)
    if rows and not header_seen:
        raise ScrubMapError(f"{source}: rows without a header")
    return ScrubMap(rows, source)


_CACHE: Dict[Tuple[str, int, int], ScrubMap] = {}


def load(path: Optional[Union[str, os.PathLike]] = None) -> ScrubMap:
    """The map at ``path`` (default :func:`default_path`); an absent file is an empty map."""
    p = Path(path) if path is not None else default_path()
    try:
        st = p.stat()
    except OSError:
        return ScrubMap((), str(p))
    key = (str(p), st.st_mtime_ns, st.st_size)
    if key not in _CACHE:
        _CACHE[key] = parse(p.read_text(encoding="utf-8"), str(p))
    return _CACHE[key]


def check(recorded_sha256: Optional[str], current_sha256: str, *, path: Optional[Union[str, os.PathLike]] = None,
          map_path: Optional[Union[str, os.PathLike]] = None) -> Optional[str]:
    """:data:`EXACT` when the shas are equal, :data:`VIA_MAP` when a map row certifies the pair, else None (the caller
    refuses as it always did). An empty or missing recorded sha is never certified."""
    rec = str(recorded_sha256 or "").lower()
    cur = str(current_sha256 or "").lower()
    if not rec or not cur:
        return None
    if rec == cur:
        return EXACT
    return VIA_MAP if load(map_path).certifies(rec, cur, path=path) else None


def check_prefix(recorded_prefix: Optional[str], current_sha256: str, *,
                 path: Optional[Union[str, os.PathLike]] = None,
                 map_path: Optional[Union[str, os.PathLike]] = None) -> Optional[str]:
    """:func:`check` for a recorded sha256 prefix."""
    pre = str(recorded_prefix or "").lower()
    cur = str(current_sha256 or "").lower()
    if not pre or not cur:
        return None
    if cur.startswith(pre):
        return EXACT
    return VIA_MAP if load(map_path).certifies_prefix(pre, cur, path=path) else None


FRAME_EXACT, FRAME_VIA_MAP, FRAME_MISMATCH, FRAME_ABSENT, FRAME_UNRESOLVABLE = (
    EXACT, VIA_MAP, "mismatch", "absent", "unresolvable")


def verify_frame_config(ref: Optional[str], recorded_sha256: Optional[str], *,
                        repo_root: Optional[Union[str, os.PathLike]] = None) -> Dict[str, Optional[str]]:
    """Whether a paper-frame row's config can be verified byte for byte from a checkout (default: this module's).

    The frame spells a config as ``repo:<rel>`` (or a plain repo-relative path) beside the sha256 its record captured:
    ``run.config_path`` with ``config_sha256``, ``object.fit_config_path`` with ``fit_config_sha256``. ``status`` is
    ``exact`` when today's file hashes to the recorded sha; ``verified_via_map`` when a certified row of the checkout's
    map maps the recorded sha to today's bytes at that path; ``mismatch`` for any other difference; ``absent`` when the
    checkout has no file there; ``unresolvable`` when the row has no reference or sha, or the reference is not a path
    inside the repo (a ``sha256:<hex>`` content reference, an absolute path, a ``..`` escape)."""
    import hashlib

    out: Dict[str, Optional[str]] = {"ref": ref, "path": None, "recorded_sha256": recorded_sha256, "sha256": None,
                                     "status": FRAME_UNRESOLVABLE, "note": None}
    rel = str(ref or "")
    rel = rel[len("repo:"):] if rel.startswith("repo:") else rel
    norm = os.path.normpath(rel) if rel else ""
    if not rel or not recorded_sha256 or ":" in rel or os.path.isabs(rel) or norm == os.pardir or norm.startswith(os.pardir + os.sep):
        out["note"] = "no repo-relative config reference and recorded sha256 on the row"
        return out
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    path = root / norm
    out["path"] = norm.replace(os.sep, "/")
    if not path.is_file():
        out["status"], out["note"] = FRAME_ABSENT, f"no file at {out['path']} in this checkout"
        return out
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    out["sha256"] = got
    map_path = root / REL_PATH
    verdict = check(str(recorded_sha256), got, path=out["path"], map_path=map_path)
    if verdict is None:
        out["status"], out["note"] = FRAME_MISMATCH, mismatch_note(recorded_sha256, got, map_path)
    else:
        out["status"] = verdict
    return out


def mismatch_note(recorded_sha256: Optional[str], current_sha256: str, map_path: Optional[Union[str, os.PathLike]] = None) -> str:
    """A clause a refusal can append: whether the recorded sha has any row at all, so an operator can tell an
    uncertified rewrite from an unrelated edit."""
    m = load(map_path)
    rows = m.rows_for(str(recorded_sha256 or ""))
    if not rows:
        return f"no scrub-map row for the recorded sha ({m.source})"
    return (f"the scrub map maps the recorded sha to {', '.join(r.post_sha256[:12] for r in rows)}, not "
            f"{str(current_sha256)[:12]}")
