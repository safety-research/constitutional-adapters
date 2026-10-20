"""Publish-implies-recorded: anything the registry publishes is an
``artifact`` row FIRST; the publisher refuses a file without a record.

``register_files`` writes ``artifact`` records (kind figure | page | report, sha256 of the bytes,
path, ``extra.notebook_commit``) into the harvester spool and ingests them; ``publish_dir`` copies
ONLY registered files (sha-verified) into the destination and refuses, naming the file, any file in
the source directory that has no artifact row or whose bytes differ from the registered sha, and any
untracked-in-git figure when a git checkout is given (the uncommitted-figures incident). A file is
registered at a path when an artifact row names the path as its primary OR among its locations
(:func:`registered_shas`; a location entry is a bare path or a location object's `uri`): a known
digest's later registration keeps the standing primary and adds its path to the row's locations, so
that is where a local copy filed after a remote declaration lives. A path may carry several
registrations over time (an earlier file at the path standing as its row's primary, a later file's
row naming it as a location), so the bytes at the path pick the registration among every row naming
it, and bytes matching none are refused naming them all. A file ALREADY in the destination is kept
only when the bytes now there have an artifact row (:func:`bytes_recorded`, the one rule
:func:`unregistered_in_dir` applies too): the rows naming its path say nothing about the bytes at it,
so a destination file drifted to a digest nobody holds is refused by name whether or not a row names
the path.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from . import harvest as H
from . import records as R
from . import reducer as RED

#: Every artifact record written here names the ONE location its writer sees: the fold unions `artifact.locations_json` on every record
#: of the table, from every producer (``reducer.UNION_COLUMNS``), so the row remembers every location the log ever named and no record
#: restates the list. The fourth review round of the analysis filing (2026-10-09) made the union a per-record directive carried by these
#: writers (`extra.union_columns`); the fourth closing pass made it the fold's rule, because an adapter's record without the directive
#: still replaced the list and erased a filed location. The writers carry no marker now.

FIGURE_SUFFIXES = (".png", ".pdf", ".svg", ".html", ".ipynb", ".md", ".json", ".csv")


class Unregistered(RuntimeError):
    def __init__(self, files: Sequence[str]):
        super().__init__("refusing to publish unregistered file(s): " + ", ".join(files))
        self.files = list(files)


sha256_of = R.file_sha256


def figure_files(directory: str, suffixes: Sequence[str] = FIGURE_SUFFIXES) -> List[str]:
    """The publishable files of a directory (regular files with a figure / page / report suffix), sorted."""
    return sorted(os.path.join(directory, f) for f in os.listdir(directory)
                  if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(tuple(suffixes)))


def page_files(directory: str, suffixes: Sequence[str] = FIGURE_SUFFIXES) -> List[str]:
    """Every publishable file under a published PAGE directory, its subdirectories included (a page serves its figures from a
    `figures/` directory beside `index.html`), sorted. :func:`figure_files` reads one directory level; a page is a tree."""
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames.sort()
        for f in filenames:
            p = os.path.join(dirpath, f)
            if os.path.isfile(p) and f.lower().endswith(tuple(suffixes)):
                out.append(p)
    return sorted(out)


def kind_of(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".png", ".pdf", ".svg"):
        return "figure"
    if ext in (".html", ".ipynb"):
        return "page"
    return "report"


def artifact_payload(path: str, notebook_commit: Optional[str], now: str, kind: Optional[str] = None,
                     sha: Optional[str] = None, size: Optional[int] = None) -> Dict[str, object]:
    """The `artifact` record of a local file. ``sha`` / ``size`` are a caller's ONE read of the file (`registry analysis file` hashes each
    file once and names the same digest in the artifact record and in the link to it); left out, the file is read here, as every other
    caller always had it."""
    sha = sha if sha is not None else sha256_of(path)
    size = size if size is not None else os.path.getsize(path)
    ex: Dict[str, object] = {}
    if notebook_commit:
        ex["notebook_commit"] = notebook_commit
    return {"artifact_id": sha, "kind": kind or kind_of(path), "uri_primary": os.path.abspath(path),
            "locations_json": [{"uri": os.path.abspath(path), "verified_utc": now, "bytes": size}],
            "bytes": size, "created_utc": now, "verified_utc": now, "extra": ex}


def remote_artifact_payload(uri: str, sha256: str, now: str, kind: Optional[str] = None, substrate: Optional[str] = None,
                            extra: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """An artifact record for a file the registering process CANNOT read -- a report on cluster storage: keyed by the sha the
    filer DECLARED (never a guess: a spec without one is refused by name before this is built), its one location unverified
    (`verified_utc` and `bytes` null in the entry) and `extra.declared_sha256` naming the declaration. The verification columns -- `bytes`,
    `created_utc`, `verified_utc` -- are OMITTED, not stated null: a record that omits a column keeps the folded value, so a fresh row reads
    null (the schema's why-null reasons) and a row another record verified -- a local read of the same bytes filed in the same window --
    is never un-verified by this one (the fourth review round, 2026-10-09). ``now`` is accepted for the callers' symmetry with
    :func:`artifact_payload` and stamps nothing. A later reader on the file's substrate may verify the location."""
    del now
    loc: Dict[str, object] = {"uri": uri, "verified_utc": None, "bytes": None}
    if substrate:
        loc["substrate"] = substrate
    ex: Dict[str, object] = {"declared_sha256": sha256}
    ex.update(dict(extra or {}))
    return {"artifact_id": sha256, "kind": kind or kind_of(uri), "uri_primary": uri, "locations_json": [loc], "extra": ex}


def merged_artifact_payload(existing: Mapping[str, object], incoming: Mapping[str, object]) -> Optional[Dict[str, object]]:
    """The `artifact` record a filing writes for a digest the registry ALREADY holds (`existing`: the standing row), given the record it
    would have written for a fresh one (`incoming`: an :func:`artifact_payload` or :func:`remote_artifact_payload`, one location). The
    record keeps the standing registration's identity -- its `kind` and `uri_primary` (a filing never moves another registration's
    primary) -- names THIS filing's one location (the fold adds it to the standing list by uri on every artifact record,
    ``reducer.union_json_list``: a verified entry replaces the standing entry of its uri, an unverified one never does), states `bytes`
    and `verified_utc` only when the incoming record verified the bytes (a local read) and omits them otherwise so the fold keeps the
    standing values, never states `created_utc` (the standing row's), and carries the incoming record's `extra`. Returns None when the
    filing's location adds nothing to the standing list as it stands (the same uri, unverified, or verified already by this very entry):
    the row stands and the link names it; nothing is written for the artifact.

    Rounds three and four of the review, 2026-10-09: before round three a remote report declaring a known digest nulled the standing row's
    `bytes` / `verified_utc` and replaced its verified local location with the one unverified cluster uri; round three's record restated
    the standing locations plus the filing's, so two filings of one digest before a fold -- each merged against the same database row --
    folded to the second record's list and lost the first's path. The list now lives in the fold alone; this record says what the filing
    itself saw."""
    inc_locs = list(incoming.get("locations_json") or [])
    assert len(inc_locs) == 1, "an incoming artifact record names its one location"
    inc_loc = dict(inc_locs[0])
    verified = incoming.get("verified_utc") is not None
    raw = existing.get("locations_json")
    standing: List[object] = [dict(l) if isinstance(l, dict) else l for l in (json.loads(raw) if isinstance(raw, str) else (raw or []))]
    if RED.union_entries(standing, [inc_loc]) == standing:
        return None
    ex: Dict[str, object] = dict(incoming.get("extra") or {})
    out: Dict[str, object] = {"artifact_id": incoming["artifact_id"], "kind": existing["kind"], "uri_primary": existing["uri_primary"],
                              "locations_json": [inc_loc], "extra": ex}
    if verified:
        out["bytes"] = incoming.get("bytes")
        out["verified_utc"] = incoming.get("verified_utc")
    return out


def untracked_in_git(paths: Sequence[str], checkout: str) -> List[str]:
    out = []
    for p in paths:
        rel = os.path.relpath(p, checkout)
        r = subprocess.run(["git", "-C", checkout, "ls-files", "--error-unmatch", rel], capture_output=True, text=True)
        if r.returncode != 0:
            out.append(p)
    return out


def register_files(root: str, paths: Sequence[str], notebook_commit: Optional[str] = None, now: Optional[str] = None,
                   conn: Optional[sqlite3.Connection] = None, ingest: bool = True,
                   payloads: Optional[Sequence[Mapping[str, object]]] = None,
                   write: Optional[Callable[[str, Sequence[Mapping[str, object]]], Optional[str]]] = None,
                   sha_by_path: Optional[Mapping[str, str]] = None) -> List[str]:
    """Write one artifact record per file into the spool (records/publish/<day>/) and ingest them.

    ``payloads`` replaces the records built from ``paths`` (a caller that mixes local files with :func:`remote_artifact_payload`
    records, or that stamps its own ``extra``); ``write`` replaces the harvester-record writer with the caller's -- ``(record_type,
    payloads) -> path`` writing ONE record file for the batch, an ``emit.RecordWriter.write_many`` -- so the artifact records land in the
    caller's own record directory, ahead of the records that will name them by FK (`registry analysis file`: the links fold after their
    parents by stamp and by path alike); ``sha_by_path`` hands :func:`artifact_payload` a digest already computed for a path. With
    ``ingest=False`` nothing touches the database: the harvester folds the records on its next cycle."""
    now = now or R.utc_now()
    recs = list(payloads) if payloads is not None else [artifact_payload(p, notebook_commit, now, sha=(sha_by_path or {}).get(p)) for p in paths]
    written: List[str] = []
    if write is not None:
        if recs:
            path = write("artifact", recs)
            if path:
                written.append(path)
    else:
        for rec in recs:
            written.append(H.write_harvester_record(root, "artifact", rec, now, subdir="publish"))
    if ingest:
        if conn is None:
            H.ingest_under_lock(root, workers=0, derive_facts=False, min_free=0)
        else:
            H.ingest_once(root, conn, workers=0, derive_facts=False, min_free=0)
    return written


#: Every `artifact` row naming a path (bound twice, once per branch) -- as its `uri_primary` or among its `locations_json` entries --
#: newest registration first by the row's `ingest_seq` (the arrival rank of the record file that produced the row as it stands), the id a
#: tie-break; a row naming the path both ways is one row (UNION). A location entry is a bare path (the older producers' `[path]`: the
#: hooks' completion and rows artifacts) or an object whose `uri` is the path, and the entry's TYPE picks the comparison, so a bare path
#: never reaches `json_extract` -- which raises `malformed JSON` on one, so on any registry holding a bare-path location the scan crashed
#: for every path no primary named (the second closing pass, 2026-10-09); an entry of any other type matches nothing. A row whose
#: `locations_json` is not a JSON text (nothing has set it) reads as the empty list, so the scan never raises on a row's text.
_REGISTERED_SHAS_SQL = (
    "SELECT artifact_id FROM ("
    "SELECT artifact_id, ingest_seq FROM artifact WHERE uri_primary = ? "
    "UNION "
    "SELECT a.artifact_id, a.ingest_seq FROM artifact a, "
    "json_each(CASE WHEN json_valid(a.locations_json) THEN a.locations_json ELSE '[]' END) loc "
    "WHERE (CASE loc.type WHEN 'text' THEN loc.value WHEN 'object' THEN json_extract(loc.value, '$.uri') END) = ?"
    ") ORDER BY ingest_seq DESC, artifact_id")


def registered_shas(conn: sqlite3.Connection, path: str) -> List[str]:
    """Every sha the registry holds for the file at ``path`` (empty when no `artifact` row names the path), newest registration first:
    the rows whose `uri_primary` is the path and the rows naming it among their `locations_json` entries, a bare path or a location
    object's `uri` alike. A filing of a digest the registry already holds never moves the standing primary (the third review round's
    ruling, 2026-10-09), so a local copy read and verified after a remote declaration of the same bytes is registered in the row's
    locations alone -- and :func:`publish_dir` called such a file unregistered and refused it (the closing pass, 2026-10-09). A path may
    be registered under several digests: version A of a report at the path (its row's primary), then version B declared from the cluster
    and its local copy filed at the same path (B's row names the path as a location while its remote primary stands), and reading the
    newest PRIMARY first returned A for a path now holding B -- `publish_dir` refused the correctly registered bytes as drifted (the
    second closing pass, 2026-10-09). So the caller holds every registration and lets the file's bytes pick: a file whose digest is among
    them is registered; bytes matching none are refused naming them all. A location declared without a read (a cluster uri,
    `verified_utc` null) admits a path exactly as an unverified remote primary always has: the bytes must still equal a registered sha.
    One table scan through `json_each` per call; publishing reads a handful of files."""
    abspath = os.path.abspath(path)
    return [str(r[0]) for r in conn.execute(_REGISTERED_SHAS_SQL, (abspath, abspath)).fetchall()]


def registered_sha(conn: sqlite3.Connection, path: str) -> Optional[str]:
    """The NEWEST sha the registry holds for the file at ``path`` (None when no `artifact` row names the path): the first of
    :func:`registered_shas`. A reader that has the file's bytes compares their digest against the whole list instead, as
    :func:`publish_dir` does -- an older registration at the path is not the current one."""
    shas = registered_shas(conn, path)
    return shas[0] if shas else None


def _registered_text(shas: Sequence[str]) -> str:
    """The registrations a drifted file's refusal names: the one sha, or every sha registered at the path, newest first."""
    if len(shas) == 1:
        return f"the registered sha {shas[0][:12]}…"
    return "every sha registered at the path (" + ", ".join(f"{s[:12]}…" for s in shas) + ", newest first)"


def bytes_recorded(conn: sqlite3.Connection, path: str) -> bool:
    """Whether the BYTES now at ``path`` have an artifact row: the digest of the file as it stands is some row's `artifact_id`. The one
    rule for a file already in a published directory -- :func:`publish_dir`'s destination check and :func:`unregistered_in_dir` apply
    it alike, so the two readers agree by construction. A registration at a path is a row whose `artifact_id` is a sha registered
    there, so a file holding the registered bytes of a row naming its path is recorded by this test as a special case, and a file at a
    registered path holding OTHER bytes is not: the rows naming a path say nothing about the bytes at it now. The third closing pass,
    2026-10-09: the destination check took ANY row naming the path (`registered_sha(conn, path) is not None`) as enough, so once a row's
    location entry named a destination path -- the verified local copy of a figure declared from the cluster -- a file there drifted to
    a digest nobody holds passed the guard and stayed, while `unregistered_in_dir` named it; a path named as a row's primary had the same
    gap. Membership of the digest in :func:`registered_shas` is implied by the row's existence (every sha it returns is a row's
    `artifact_id`), so the digest lookup alone is the whole rule."""
    return conn.execute("SELECT 1 FROM artifact WHERE artifact_id = ? LIMIT 1", (sha256_of(path),)).fetchone() is not None


def _destination_refusal(conn: sqlite3.Connection, path: str) -> str:
    """A destination file's refusal: its bytes have no artifact row -- and, when rows name the path, the registrations they differ from."""
    shas = registered_shas(conn, path)
    if shas:
        return f"{path} (present in the destination with bytes that differ from {_registered_text(shas)})"
    return f"{path} (present in the destination with no artifact row)"


@dataclass
class PublishReport:
    published: List[str] = field(default_factory=list)
    refused: List[str] = field(default_factory=list)


def publish_dir(conn: sqlite3.Connection, src_dir: str, dest_dir: str, checkout: Optional[str] = None,
                suffixes: Sequence[str] = FIGURE_SUFFIXES) -> PublishReport:
    """Copy registered, sha-verified files from ``src_dir`` to ``dest_dir``; refuse the rest by name."""
    files = figure_files(src_dir, suffixes)
    problems: List[str] = []
    if checkout:
        for p in untracked_in_git(files, checkout):
            problems.append(f"{p} (untracked in git)")
    ok: List[str] = []
    for p in files:
        shas = registered_shas(conn, p)
        if not shas:
            problems.append(f"{p} (no artifact row)")
            continue
        if sha256_of(p) not in shas:
            problems.append(f"{p} (bytes differ from {_registered_text(shas)})")
            continue
        ok.append(p)
    if problems:
        raise Unregistered(problems)
    os.makedirs(dest_dir, exist_ok=True)
    rep = PublishReport()
    for p in ok:
        shutil.copy2(p, os.path.join(dest_dir, os.path.basename(p)))
        rep.published.append(p)
    # anything already in dest whose BYTES have no record is refused by name (never silently kept or deleted); a row naming the path
    # does not admit the bytes now there (:func:`bytes_recorded`)
    published_names = {os.path.basename(x) for x in ok}
    for f in sorted(os.listdir(dest_dir)):
        dp = os.path.join(dest_dir, f)
        if os.path.isfile(dp) and f.lower().endswith(tuple(suffixes)) and f not in published_names and not bytes_recorded(conn, dp):
            rep.refused.append(dp)
    if rep.refused:
        raise Unregistered([_destination_refusal(conn, p) for p in rep.refused])
    return rep


def unregistered_in_dir(conn: sqlite3.Connection, directory: str, suffixes: Sequence[str] = FIGURE_SUFFIXES) -> List[str]:
    """Files in an already-published directory whose bytes have NO artifact row (:func:`bytes_recorded`): a file registered at its path
    but holding other bytes is among them, exactly as :func:`publish_dir` refuses it from a destination."""
    out: List[str] = []
    if not os.path.isdir(directory):
        return out
    for f in sorted(os.listdir(directory)):
        p = os.path.join(directory, f)
        if os.path.isfile(p) and f.lower().endswith(tuple(suffixes)) and not bytes_recorded(conn, p):
            out.append(p)
    return out
