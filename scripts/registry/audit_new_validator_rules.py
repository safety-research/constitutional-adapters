#!/usr/bin/env python3
"""The whole-log audit of the validator rules the registry record-log branch added, read-only (plan ``recordlog-d635-d636-d639``,
section 5 "Pre-landing audit", commit C7): run it before every landing of a validator change and read its verdict before the window.

The record log is append-only and a rebuild must fold it to the live tables byte for byte, so a NEW validator rule may refuse no
historical record -- or every historical record it would refuse is grandfathered by sha in ``validate.GRANDFATHERED_RECORD_SHAS``, a
list that grows only by this audit. A rule with an unlisted historical hit is a STOP: report it, never
land it. The audit walks every record file under ``<root>/records`` except the item-scale files (a run's items are gigabytes and no rule
here reads them) and the harvester's own fact files, re-applies each rule to the records AS WRITTEN, and compares the hits with the list:

* ``rung_violations`` (``validate.run_cap_ladder_refusal``): a ``run`` record stating ``token_cap_rung_index`` whose index,
  ladder and ``token_cap_tokens`` disagree -- expected 0, nothing grandfathered.
* ``placeholder_run_code_sha`` (``validate.run_code_sha_refusal``): a ``run`` record whose ``code_sha`` is forty identical hex
  digits -- expected exactly the grandfathered records (two synthetic test runs, two files each).
* ``judgment_id_form_misses`` (``identity.judgment_id_form``): a ``judgment`` record whose id no identity form reproduces --
  expected the grandfathered release records; a release record written by the OLD ``pending.release`` after the list was last measured
  is EXPECTED to appear here until the window and is what the window grandfathers (the plan's risk table); the audit also counts the
  records per form, the distinct ids, the ids with several records, and the same-id records that changed ``run_id`` or ``judge_id``
  (expected 0: the immutability the rule makes structural).
* ``run_patch``: patches naming a derived decode column (``records.RUN_DERIVED_FIELDS``; expected 0), naming any ``decode_*`` /
  ``instrument_match`` / ``token_cap_*`` field (a count), carrying a ``fields.extra_json`` that is not an object (the rule in
  ``records.run_patch_extra_json``; expected 0), and carrying a non-empty top-level ``extra`` (the legacy stamps the reconcile
  pass handles: records and runs).
* ``object_lineage``: a ``named_role`` outside ``vocab.OBJECT_NAMED_ROLES`` or a ``parent_object_id`` equal to the object's own id,
  stated as a column or under ``extra`` alike -- expected 0.
* the grandfather list's accounting: every listed sha must be a hit of one of the rules above (``grandfathered_unaccounted`` lists the
  rest; ``--hash-all`` hashes every record read to say whether such a sha is in the log at all).

With the database beside the log (``<root>/registry.sqlite``, opened ``mode=ro``; ``--no-db`` skips it) the audit adds the live
figures the plan's window expects, each a bounded query: the schema version against the code's; ``record_refusal`` rows and, once the
record-refusal-fact migration has landed, the facts and the current refusals without one; the refused holds in ``blocked_records``
(their previous judgments become facts before the first re-attempt) and so the ceiling on ``record_refusal_fact`` after the window's
first cycle; the ``judgment`` statuses, the successor-less superseded judgments, and the identity form of every LIVE judgment row; the
two currency rules' live counts (the (run, judge) pairs carrying both a ``partial`` and a ``complete`` judgment; the judgments
whose binding at claim differs from the run's current identity -- 0 by construction while the two columns do not exist yet); the object
lineage columns against the ``extra_json`` keys they lift; and the reconcile buckets (``reconcile.reconcile_run_patch_extra`` dry run:
missing / equal / conflict per run and per key; ``--skip-reconcile`` omits it). ``--paper-counts`` adds the paper's ``in_current_paper`` /
``selected`` totals, the one query here that folds ``v_paper_current`` (about a minute); off by default.

Nothing is written under the root -- not a ``state/last_run`` stamp, not a file; ``--out`` refuses a path inside it. The one thing a
run can leave is SQLite's ``-wal`` / ``-shm`` side files of a read-only open, which carry no data. Exit 0 when every expectation holds
(no violation, every hit grandfathered, every grandfathered sha accounted for); 1 when one does not, each named on stderr -- the
live counts (the currency rules, the buckets, the refusals) are REPORTED, never a failure, since the plan's stop thresholds on
them are decided outside this audit; 2 on a usage or I/O error. ``--json`` prints the whole report as one JSON object.

    PYTHONPATH=src python scripts/registry/audit_new_validator_rules.py [--root ~/registry] [--json] [--out FILE] [--all]
        [--no-db] [--skip-reconcile] [--paper-counts] [--hash-all] [--progress]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

from sisteer.registry import db as DB  # noqa: E402
from sisteer.registry import identity  # noqa: E402
from sisteer.registry import migrations as MIG  # noqa: E402
from sisteer.registry import records as R  # noqa: E402
from sisteer.registry import schema_def as S  # noqa: E402
from sisteer.registry import validate as VAL  # noqa: E402
from sisteer.registry import vocab as V  # noqa: E402
from sisteer.registry.records import write_json_atomic  # noqa: E402

#: Rows listed per finding unless ``--all``.
DEFAULT_LIMIT = 20
#: The record kinds the audited rules read; by default only their files are walked (every writer names a file by the one kind it
#: holds -- ``emit.RecordWriter.write_many``, the adapters' revision directories -- so the four kinds' files hold the four kinds:
#: 17,8xx files, under a minute). ``--every-file`` walks every non-item-scale record file instead (3.3 M records, about six minutes on
#: the live log) and reports a record of these kinds found in a file named for another kind (``misfiled``: 0 on 2026-10-04).
AUDITED_RECORD_TYPES: Tuple[str, ...] = ("run", "run_patch", "judgment", "object")
#: A record file's type is in its name (``records.RECORD_FILE_RE``: ``<seq>-<type>.jsonl[.gz]``); the harvester's fact files
#: (``file_arrival-<sha12>``, ``record_refusal-<sha12>``) carry no sequence and are not walked.
_TYPED_FILE = re.compile(r"^(?P<seq>\d+)-(?P<type>[a-z_]+)\.jsonl(?P<gz>\.gz)?$")
#: The ``run_patch`` field families the plan's audit counts (section 0: "run_patch naming any decode_* / instrument_match / token_cap_*").
_COUNTED_PATCH_FIELDS = re.compile(r"^(decode_|token_cap_|instrument_match$)")


class AuditError(Exception):
    """A usage or I/O condition the audit refuses by name (exit 2)."""


def _raise_listing_error(exc: OSError) -> None:
    """``os.walk`` skips a directory it cannot list unless told otherwise; an audit over part of the log would report a partial result
    that looks whole, so a listing error is raised by name instead (the reconcile pass's rule)."""
    raise exc


def audited_record_files(records_root: str, every_file: bool = False) -> List[str]:
    """The typed record files under ``records_root`` named for one of :data:`AUDITED_RECORD_TYPES` (every non-item-scale typed file with
    ``every_file``), one per LOGICAL uri (``records.logical_uri``: the path without ``.gz``), sorted (deterministic). The live harvester
    compresses a record file in place once it is old enough (``harvest.gzip_after_ingest``), so a listing taken while it runs can name
    both spellings of one file; the plain one is kept and :func:`existing_spelling` reads whichever exists at read time. An absent root
    is refused."""
    if not os.path.isdir(records_root):
        raise AuditError(f"no record log at {records_root!r}")
    item_scale = set(V.ITEM_SCALE_RECORD_TYPES)
    wanted = None if every_file else set(AUDITED_RECORD_TYPES)
    by_uri: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(records_root, onerror=_raise_listing_error):
        dirnames.sort()
        for fn in filenames:
            m = _TYPED_FILE.match(fn)
            if m and m.group("type") not in item_scale and (wanted is None or m.group("type") in wanted):
                path = os.path.join(dirpath, fn)
                uri = R.logical_uri(path)
                if uri not in by_uri or path == uri:      # the plain spelling wins the listing; the read resolves what exists
                    by_uri[uri] = path
    return sorted(by_uri.values())


def existing_spelling(path: str) -> str:
    """The spelling of a listed record file that exists NOW: the path itself, else its gzip twin (the harvester compressed it since
    the listing), else the plain file (a twin listed as ``.gz`` that was not yet renamed). Raises ``FileNotFoundError`` when neither
    exists: a record file never vanishes from the log, so that is a listing error, not a skip."""
    if os.path.exists(path):
        return path
    plain = R.logical_uri(path)
    for alt in (plain + ".gz", plain):
        if alt != path and os.path.exists(alt):
            return alt
    raise FileNotFoundError(f"{path} (and its gzip / plain twin) vanished between the listing and the read")


# --------------------------------------------------------------------------------------------
# The walk
# --------------------------------------------------------------------------------------------


@dataclass
class Hit:
    """One record a rule refuses (or a figure names), located in the log."""
    sha: str
    uri: str
    written_utc: str
    detail: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {"sha": self.sha, "uri": self.uri, "written_utc": self.written_utc, **self.detail}


@dataclass
class LogAudit:
    records_root: str
    every_file: bool = False
    files_read: int = 0
    records_read: int = 0
    records_by_type: Dict[str, int] = field(default_factory=dict)
    #: a record of an audited kind found in a file named for a kind OUTSIDE the four (``every_file`` only; 0 expected)
    misfiled: Dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    # run: the token-cap ladder rule and the placeholder rule
    run_records: int = 0
    run_index_stated: int = 0
    run_ladder_set_no_index: int = 0
    run_ladder_empty_or_absent: int = 0
    rung_violations: List[Hit] = field(default_factory=list)
    run_code_sha_unverified: int = 0
    placeholder_run_code_sha: List[Hit] = field(default_factory=list)
    # run_patch: the derived pair, the counted families, the extra_json shape, the legacy top-level extra
    run_patch_records: int = 0
    run_patch_derived_pair: List[Hit] = field(default_factory=list)
    run_patch_decode_or_cap_fields_named: int = 0
    run_patch_malformed_extra_json: List[Hit] = field(default_factory=list)
    run_patch_top_level_extra_records: int = 0
    run_patch_top_level_extra_runs: Set[str] = field(default_factory=set)
    # judgment: the identity forms and the immutability of (run_id, judge_id)
    judgment_records: int = 0
    judgment_by_form: Dict[str, int] = field(default_factory=dict)
    judgment_id_form_misses: List[Hit] = field(default_factory=list)
    judgment_identity_by_id: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    judgment_records_per_id: Dict[str, int] = field(default_factory=dict)
    judgment_run_id_changes: List[Hit] = field(default_factory=list)
    judgment_judge_id_changes: List[Hit] = field(default_factory=list)
    # object: the lineage rules
    object_records: int = 0
    object_named_role_stated: int = 0
    object_parent_stated: int = 0
    object_lineage_violations: List[Hit] = field(default_factory=list)
    # the grandfather list seen in the log (``--hash-all`` only)
    hashed_all: bool = False
    grandfathered_seen: Dict[str, str] = field(default_factory=dict)

    def as_dict(self, limit: Optional[int]) -> Dict[str, object]:
        def listing(hits: List[Hit]) -> Dict[str, object]:
            shown = hits if limit is None else hits[:limit]
            return {"count": len(hits), "listed": len(shown), "records": [h.as_dict() for h in shown]}
        multi = {k: n for k, n in self.judgment_records_per_id.items() if n > 1}
        return {
            "records_root": self.records_root, "file_kinds": "every non-item-scale kind" if self.every_file else list(AUDITED_RECORD_TYPES),
            "files_read": self.files_read, "records_read": self.records_read,
            "records_by_type": dict(sorted(self.records_by_type.items())), "misfiled": dict(sorted(self.misfiled.items())),
            "seconds": round(self.seconds, 1),
            "run": {"records": self.run_records, "index_stated": self.run_index_stated, "ladder_set_no_index": self.run_ladder_set_no_index,
                    "ladder_empty_or_absent": self.run_ladder_empty_or_absent, "code_sha_unverified": self.run_code_sha_unverified},
            "rung_violations": listing(self.rung_violations),
            "placeholder_run_code_sha": listing(self.placeholder_run_code_sha),
            "run_patch": {"records": self.run_patch_records, "derived_pair_patches": listing(self.run_patch_derived_pair),
                          "decode_or_cap_fields_named": self.run_patch_decode_or_cap_fields_named,
                          "malformed_extra_json": listing(self.run_patch_malformed_extra_json),
                          "top_level_extra_records": self.run_patch_top_level_extra_records,
                          "top_level_extra_runs": len(self.run_patch_top_level_extra_runs)},
            "judgment": {"records": self.judgment_records, "by_form": dict(sorted(self.judgment_by_form.items())),
                         "distinct_ids": len(self.judgment_identity_by_id), "multi_record_ids": len(multi),
                         "max_records_per_id": max(multi.values()) if multi else (1 if self.judgment_records_per_id else 0),
                         "run_id_changes": listing(self.judgment_run_id_changes), "judge_id_changes": listing(self.judgment_judge_id_changes)},
            "judgment_id_form_misses": listing(self.judgment_id_form_misses),
            "object": {"records": self.object_records, "named_role_stated": self.object_named_role_stated,
                       "parent_object_id_stated": self.object_parent_stated},
            "object_lineage_violations": listing(self.object_lineage_violations),
            "hashed_all": self.hashed_all,
        }


def _uri(records_root: str, path: str, line: int) -> str:
    return f"records/{os.path.relpath(path, records_root)}#{line}"


def _audit_run(a: LogAudit, payload: Mapping[str, object], rec: Mapping[str, object], uri: str) -> None:
    a.run_records += 1
    idx = payload.get("token_cap_rung_index")
    ladder = payload.get("token_cap_ladder_json")
    if idx is not None:
        a.run_index_stated += 1
    elif ladder not in (None, [], "[]", ""):
        a.run_ladder_set_no_index += 1
    else:
        a.run_ladder_empty_or_absent += 1
    bad = VAL.run_cap_ladder_refusal(payload.get("token_cap_tokens"), idx, ladder)
    if bad:
        a.rung_violations.append(Hit(R.record_sha(rec), uri, str(rec.get("written_utc", "")),
                                     {"run_id": payload.get("run_id"), "field": bad[0], "reason": bad[1]}))
    code_sha = payload.get("code_sha")
    if code_sha == "unverified":
        a.run_code_sha_unverified += 1
    if VAL.run_code_sha_refusal(code_sha):
        a.placeholder_run_code_sha.append(Hit(R.record_sha(rec), uri, str(rec.get("written_utc", "")),
                                              {"run_id": payload.get("run_id"), "code_sha": code_sha}))


def _audit_run_patch(a: LogAudit, payload: Mapping[str, object], rec: Mapping[str, object], uri: str) -> None:
    a.run_patch_records += 1
    fields = payload.get("fields")
    if isinstance(fields, Mapping):
        derived = sorted(k for k in fields if k in R.RUN_DERIVED_FIELDS)
        if derived:
            a.run_patch_derived_pair.append(Hit(R.record_sha(rec), uri, str(rec.get("written_utc", "")),
                                                {"run_id": payload.get("run_id"), "fields": derived}))
        if any(_COUNTED_PATCH_FIELDS.match(k) for k in fields):
            a.run_patch_decode_or_cap_fields_named += 1
        if "extra_json" in fields:
            _own, reason = R.run_patch_extra_json(fields["extra_json"])
            if reason:
                a.run_patch_malformed_extra_json.append(Hit(R.record_sha(rec), uri, str(rec.get("written_utc", "")),
                                                            {"run_id": payload.get("run_id"), "reason": reason}))
    extra = payload.get("extra")
    if extra not in (None, {}, "", []):
        a.run_patch_top_level_extra_records += 1
        run_id = payload.get("run_id")
        if isinstance(run_id, str):
            a.run_patch_top_level_extra_runs.add(run_id)


def _audit_judgment(a: LogAudit, payload: Mapping[str, object], rec: Mapping[str, object], uri: str) -> None:
    a.judgment_records += 1
    form = identity.judgment_id_form(payload)
    a.judgment_by_form[form or "none"] = a.judgment_by_form.get(form or "none", 0) + 1
    jid, run, judge = payload.get("judgment_id"), payload.get("run_id"), payload.get("judge_id")
    if form is None:
        a.judgment_id_form_misses.append(Hit(R.record_sha(rec), uri, str(rec.get("written_utc", "")),
                                             {"judgment_id": jid, "run_id": run, "judge_id": judge, "status": payload.get("status"),
                                              "claimed_by_session": payload.get("claimed_by_session"), "claimed_utc": payload.get("claimed_utc"),
                                              "producer_session": payload.get("producer_session"), "script": (rec.get("producer") or {}).get("script")}))
    if isinstance(jid, str):
        a.judgment_records_per_id[jid] = a.judgment_records_per_id.get(jid, 0) + 1
        prior = a.judgment_identity_by_id.get(jid)
        if prior is None:
            a.judgment_identity_by_id[jid] = (str(run), str(judge))
        else:
            if prior[0] != str(run):
                a.judgment_run_id_changes.append(Hit(R.record_sha(rec), uri, str(rec.get("written_utc", "")),
                                                     {"judgment_id": jid, "run_id_first": prior[0], "run_id_here": run}))
            if prior[1] != str(judge):
                a.judgment_judge_id_changes.append(Hit(R.record_sha(rec), uri, str(rec.get("written_utc", "")),
                                                       {"judgment_id": jid, "judge_id_first": prior[1], "judge_id_here": judge}))


def _audit_object(a: LogAudit, payload: Mapping[str, object], rec: Mapping[str, object], uri: str) -> None:
    """The two rules over a ``named_role`` / ``parent_object_id`` stated as a column OR under ``extra`` (the validator checks both
    spellings by the same rules and the fold lifts the extra key when no column is stated)."""
    a.object_records += 1
    extra = payload.get("extra") if isinstance(payload.get("extra"), Mapping) else {}
    role = payload["named_role"] if "named_role" in payload else extra.get("named_role")
    parent = payload["parent_object_id"] if "parent_object_id" in payload else extra.get("parent_object_id")
    if role is not None:
        a.object_named_role_stated += 1
    if parent is not None:
        a.object_parent_stated += 1
    bad: Optional[Tuple[str, str]] = None
    if role is not None and (not isinstance(role, str) or role not in V.OBJECT_NAMED_ROLES):
        bad = ("named_role", f"named_role={role!r} is not in OBJECT_NAMED_ROLES {list(V.OBJECT_NAMED_ROLES)}")
    elif parent is not None and (not isinstance(parent, str) or parent == payload.get("object_id")):
        bad = ("parent_object_id", f"parent_object_id={parent!r} is not a string" if not isinstance(parent, str) else
               f"parent_object_id={parent!r} is the object's own id")
    if bad:
        a.object_lineage_violations.append(Hit(R.record_sha(rec), uri, str(rec.get("written_utc", "")),
                                               {"object_id": payload.get("object_id"), "field": bad[0], "reason": bad[1]}))


def walk_log(records_root: str, hash_all: bool = False, every_file: bool = False, progress=None) -> LogAudit:
    """One pass over the audited files; the rules re-applied to every record as written. ``hash_all`` computes every record's sha
    so the grandfather list can be located in the log; ``every_file`` widens the walk to every non-item-scale file (module
    docstring); ``progress`` is called with ``(files_done, files_total)`` every 1,000 files."""
    files = audited_record_files(records_root, every_file=every_file)
    a = LogAudit(records_root, every_file=every_file, files_read=len(files), hashed_all=hash_all)
    t0 = time.monotonic()
    for n, listed in enumerate(files, start=1):
        path = existing_spelling(listed)
        named_kind = _TYPED_FILE.match(os.path.basename(path)).group("type")  # type: ignore[union-attr]
        for line, rec, sha in R.read_record_file(path, compute_sha=hash_all):
            a.records_read += 1
            rt = rec.get("record_type")
            a.records_by_type[str(rt)] = a.records_by_type.get(str(rt), 0) + 1
            if rt in AUDITED_RECORD_TYPES and named_kind not in AUDITED_RECORD_TYPES:
                # a record the rules read, in a file the default walk never opens (a `judgment` in a `-run_state` file): only
                # --every-file can see it; a run's own file holding its `run_patch` records is walked either way and is not this
                a.misfiled[str(rt)] = a.misfiled.get(str(rt), 0) + 1
            payload = rec.get("payload")
            if not isinstance(payload, Mapping):
                continue
            uri = _uri(records_root, R.logical_uri(path), line)
            if hash_all and sha in VAL.GRANDFATHERED_RECORD_SHAS:
                a.grandfathered_seen[str(sha)] = uri
            if rt == "run":
                _audit_run(a, payload, rec, uri)
            elif rt == "run_patch":
                _audit_run_patch(a, payload, rec, uri)
            elif rt == "judgment":
                _audit_judgment(a, payload, rec, uri)
            elif rt == "object":
                _audit_object(a, payload, rec, uri)
        if progress is not None and n % 1000 == 0:
            progress(n, len(files))
    a.seconds = time.monotonic() - t0
    return a


# --------------------------------------------------------------------------------------------
# The grandfather list against the hits
# --------------------------------------------------------------------------------------------


def grandfather_accounting(a: LogAudit) -> Dict[str, object]:
    """Every listed sha must be a hit of a rule above, and every hit a listed sha. ``not_grandfathered`` per rule is what the window
    must add to the list (or what must not land); ``grandfathered_unaccounted`` is a listed sha no rule needs today -- with
    ``--hash-all`` split into the ones in the log (a stale entry) and the ones not found (a record this log does not hold)."""
    listed = set(VAL.GRANDFATHERED_RECORD_SHAS)
    placeholder = {h.sha for h in a.placeholder_run_code_sha}
    misses = {h.sha for h in a.judgment_id_form_misses}
    ladder_hits = {h.sha for h in a.rung_violations}
    lineage = {h.sha for h in a.object_lineage_violations}
    derived = {h.sha for h in a.run_patch_derived_pair}
    malformed = {h.sha for h in a.run_patch_malformed_extra_json}
    hits = placeholder | misses | ladder_hits | lineage | derived | malformed
    out: Dict[str, object] = {
        "listed": len(listed),
        "placeholder_run_code_sha": {"hits": len(placeholder), "grandfathered": len(placeholder & listed), "not_grandfathered": sorted(placeholder - listed)},
        "judgment_id_form_misses": {"hits": len(misses), "grandfathered": len(misses & listed), "not_grandfathered": sorted(misses - listed)},
        "rung_violations": {"hits": len(ladder_hits), "grandfathered": len(ladder_hits & listed), "not_grandfathered": sorted(ladder_hits - listed)},
        "object_lineage_violations": {"hits": len(lineage), "grandfathered": len(lineage & listed), "not_grandfathered": sorted(lineage - listed)},
        "run_patch_derived_pair": {"hits": len(derived), "grandfathered": len(derived & listed), "not_grandfathered": sorted(derived - listed)},
        "run_patch_malformed_extra_json": {"hits": len(malformed), "grandfathered": len(malformed & listed), "not_grandfathered": sorted(malformed - listed)},
        "grandfathered_unaccounted": sorted(listed - hits),
    }
    if a.hashed_all:
        unaccounted = listed - hits
        out["grandfathered_unaccounted_in_log"] = {s: a.grandfathered_seen[s] for s in sorted(unaccounted & set(a.grandfathered_seen))}
        out["grandfathered_unaccounted_not_in_log"] = sorted(unaccounted - set(a.grandfathered_seen))
    return out


# --------------------------------------------------------------------------------------------
# The database: the window's live figures, bounded queries
# --------------------------------------------------------------------------------------------


def _columns(conn: sqlite3.Connection, table: str) -> Set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone() is not None


def _one(conn: sqlite3.Connection, sql: str, *params) -> object:
    return conn.execute(sql, params).fetchone()[0]


def live_judgment_forms(conn: sqlite3.Connection) -> Dict[str, Dict[str, int]]:
    """The identity form of every live ``judgment`` row by status (``none`` = no form reproduces the row's id: the folded shape of a
    legacy release record, whose ``claimed_utc`` column took the release time). A judgment-scale read (tens of thousands of rows)."""
    out: Dict[str, Dict[str, int]] = {}
    cur = conn.execute("SELECT judgment_id, run_id, judge_id, claimed_by_session, claimed_utc, producer_session, status FROM judgment")
    for jid, run, judge, cbs, cutc, ps, status in cur:
        form = identity.judgment_id_form({"judgment_id": jid, "run_id": run, "judge_id": judge, "claimed_by_session": cbs,
                                          "claimed_utc": cutc, "producer_session": ps}) or "none"
        out.setdefault(form, {})
        out[form][str(status)] = out[form].get(str(status), 0) + 1
    return {f: dict(sorted(v.items())) for f, v in sorted(out.items())}


def currency_rule_counts(conn: sqlite3.Connection) -> Dict[str, object]:
    """The two paper-side currency rules of the record-refusal-fact migration, counted on the live rows.

    Complete over partial per (run, judge): the ``partial`` judgments beside a ``complete`` one of the same (run_id, judge_id),
    and the pairs; the witness is identity-current when the binding columns exist, status-only before the migration. The
    identity bound at claim: the judgments whose stated ``claimed_instrument_id`` / ``claimed_required_judge_id`` differs from the run's
    current instrument / required judge -- 0 by construction while the columns do not exist (every live row is NULL-bound)."""
    cols = _columns(conn, "judgment")
    binding = {"claimed_instrument_id", "claimed_required_judge_id"} <= cols
    if binding:
        witness = ("SELECT 1 FROM judgment jc JOIN run r ON r.run_id = jc.run_id LEFT JOIN instrument i ON i.instrument_id = r.instrument_id "
                   f"WHERE jc.run_id = jp.run_id AND jc.judge_id = jp.judge_id AND jc.status = 'complete' AND {S.judgment_identity_current_sql('jc', 'r', 'i')}")
    else:
        witness = "SELECT 1 FROM judgment jc WHERE jc.run_id = jp.run_id AND jc.judge_id = jp.judge_id AND jc.status = 'complete'"
    d744_partials = int(_one(conn, f"SELECT COUNT(*) FROM judgment jp WHERE jp.status = 'partial' AND EXISTS ({witness})"))
    d744_pairs = int(_one(conn, f"SELECT COUNT(*) FROM (SELECT DISTINCT jp.run_id, jp.judge_id FROM judgment jp WHERE jp.status = 'partial' AND EXISTS ({witness}))"))
    out: Dict[str, object] = {
        "d744_partial_displaced_by_complete": {"partial_judgments": d744_partials, "run_judge_pairs": d744_pairs, "witness_identity_current": binding},
        "d757_identity_bound_at_claim": {"binding_columns_present": binding},
    }
    if binding:
        stated = int(_one(conn, "SELECT COUNT(*) FROM judgment WHERE claimed_instrument_id IS NOT NULL OR claimed_required_judge_id IS NOT NULL"))
        stale = int(_one(conn, "SELECT COUNT(*) FROM judgment j JOIN run r ON r.run_id = j.run_id LEFT JOIN instrument i ON i.instrument_id = r.instrument_id "
                               f"WHERE NOT {S.judgment_identity_current_sql('j', 'r', 'i')}"))
        by_status = {str(r[0]): int(r[1]) for r in conn.execute(
            "SELECT j.status, COUNT(*) FROM judgment j JOIN run r ON r.run_id = j.run_id LEFT JOIN instrument i ON i.instrument_id = r.instrument_id "
            f"WHERE NOT {S.judgment_identity_current_sql('j', 'r', 'i')} GROUP BY j.status ORDER BY j.status").fetchall()}
        out["d757_identity_bound_at_claim"].update(stated_binding=stated, stale_bound=stale, stale_bound_by_status=by_status)
    else:
        out["d757_identity_bound_at_claim"].update(stated_binding=0, stale_bound=0, note="the two columns do not exist yet: every live row is NULL-bound, 0 by construction")
    return out


def audit_database(conn: sqlite3.Connection, paper_counts: bool = False) -> Dict[str, object]:
    """The live figures the window expects (module docstring). Every query bounded; ``paper_counts`` adds the one that folds
    ``v_paper_current``."""
    out: Dict[str, object] = {}
    out["schema_version"] = int(_one(conn, "SELECT MAX(version) FROM schema_migration"))
    out["code_version"] = MIG.code_version()
    # refusals: the current judgments, the facts (once the table exists), the ceiling after the window's first cycle
    refusal_cols = _columns(conn, "record_refusal")
    refusals: Dict[str, object] = {"current_refusals": int(_one(conn, "SELECT COUNT(*) FROM record_refusal")),
                                   "judged_with_column_present": "judged_with" in refusal_cols}
    if "judged_with" in refusal_cols:
        refusals["current_unstamped"] = int(_one(conn, "SELECT COUNT(*) FROM record_refusal WHERE judged_with IS NULL"))
    facts_present = _table_exists(conn, "record_refusal_fact")
    refusals["fact_table_present"] = facts_present
    if facts_present:
        stamp = "COALESCE(r.judged_with, 'unverified')" if "judged_with" in refusal_cols else "'unverified'"
        refusals.update(
            facts=int(_one(conn, "SELECT COUNT(*) FROM record_refusal_fact")),
            fact_records=int(_one(conn, "SELECT COUNT(DISTINCT record_sha256) FROM record_refusal_fact")),
            facts_unverified=int(_one(conn, "SELECT COUNT(*) FROM record_refusal_fact WHERE judged_with = 'unverified'")),
            facts_backfilled=int(_one(conn, "SELECT COUNT(*) FROM record_refusal_fact WHERE json_extract(extra_json, '$.backfilled') = 1")),
            current_without_fact=int(_one(conn, "SELECT COUNT(*) FROM record_refusal r WHERE NOT EXISTS (SELECT 1 FROM record_refusal_fact f "
                                                f"WHERE f.record_sha256 = r.record_sha256 AND f.judged_with = {stamp})")))
    else:
        refusals["current_without_fact"] = refusals["current_refusals"]
    holds = int(_one(conn, "SELECT COUNT(*) FROM blocked_records WHERE json_extract(extra_json, '$.kind') = 'refused'"))
    refusals["blocked_records"] = int(_one(conn, "SELECT COUNT(*) FROM blocked_records"))
    refusals["refused_holds"] = holds
    # the START pass writes one fact per current refusal without one; each refused hold re-attempts once under the moved stamp and
    # the END pass writes at most one new-stamp fact for it
    refusals["fact_rows_ceiling_after_first_cycle"] = int(refusals.get("facts", 0)) + int(refusals["current_without_fact"]) + holds
    out["refusals"] = refusals
    # judgments: statuses, the successor-less superseded ones, the live identity forms, the currency rules
    out["judgment"] = {
        "by_status": {str(r[0]): int(r[1]) for r in conn.execute("SELECT status, COUNT(*) FROM judgment GROUP BY status ORDER BY status").fetchall()},
        "superseded_without_successor": int(_one(conn, "SELECT COUNT(*) FROM judgment WHERE status = 'superseded' AND superseded_by_judgment_id IS NULL")),
        "superseded_without_successor_ids": [str(r[0]) for r in conn.execute(
            "SELECT judgment_id FROM judgment WHERE status = 'superseded' AND superseded_by_judgment_id IS NULL ORDER BY judgment_id LIMIT 10").fetchall()],
        "live_rows_by_identity_form": live_judgment_forms(conn),
    }
    out["currency_rules"] = currency_rule_counts(conn)
    # object lineage: the columns against the extra_json keys the fold lifts
    obj_cols = _columns(conn, "object")
    lineage: Dict[str, object] = {"columns_present": {"parent_object_id", "named_role"} <= obj_cols,
                                  "extra_parent_object_id": int(_one(conn, "SELECT COUNT(*) FROM object WHERE json_extract(extra_json, '$.parent_object_id') IS NOT NULL")),
                                  "extra_named_role": int(_one(conn, "SELECT COUNT(*) FROM object WHERE json_extract(extra_json, '$.named_role') IS NOT NULL"))}
    if lineage["columns_present"]:
        lineage["column_parent_object_id"] = int(_one(conn, "SELECT COUNT(*) FROM object WHERE parent_object_id IS NOT NULL"))
        lineage["column_named_role"] = int(_one(conn, "SELECT COUNT(*) FROM object WHERE named_role IS NOT NULL"))
    out["object_lineage"] = lineage
    if paper_counts:
        row = conn.execute("SELECT SUM(in_current_paper), SUM(selection_state = 'selected'), COUNT(*) FROM v_paper_current").fetchone()
        out["paper_counts"] = {"in_current_paper": int(row[0] or 0), "selected": int(row[1] or 0), "rows": int(row[2] or 0)}
    return out


def c1c_buckets(root: str, conn: sqlite3.Connection) -> Dict[str, object]:
    """The C1c dry run (``reconcile.reconcile_run_patch_extra``, nothing written): runs and keys per bucket, the lost records by what
    the fold did with them."""
    from sisteer.registry import reconcile as RC
    rep = RC.reconcile_run_patch_extra(root, apply=False, conn=conn)
    return {"runs": len(rep.runs), "runs_by_bucket": rep.by_bucket(), "keys_by_bucket": rep.keys_by_bucket(),
            "records_by_status": rep.records_by_status(), "files_read": rep.files_read, "records_read": rep.records_read,
            "conflict_runs": sorted(r.run_id for r in rep.runs if r.bucket == "conflict")[:DEFAULT_LIMIT]}


# --------------------------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------------------------


def expectations(report: Mapping[str, object]) -> List[str]:
    """The plan's pre-landing expectations that are a FAILURE when missed (each a line); the live counts are reported, never failed."""
    log = report["log"]
    gf = report["grandfather"]
    out = []
    if log["rung_violations"]["count"]:
        out.append(f"rung_violations: {log['rung_violations']['count']} run record(s) fail the token-cap ladder rule (expected 0)")
    for rule in ("placeholder_run_code_sha", "judgment_id_form_misses", "rung_violations", "object_lineage_violations",
                 "run_patch_derived_pair", "run_patch_malformed_extra_json"):
        missing = gf[rule]["not_grandfathered"]
        if missing:
            out.append(f"{rule}: {len(missing)} record(s) not in validate.GRANDFATHERED_RECORD_SHAS -- a new rule with an unlisted historical hit is "
                       f"a STOP (grandfather by audit at the window, or do not land): {', '.join(s[:12] for s in missing[:DEFAULT_LIMIT])}"
                       + (" ..." if len(missing) > DEFAULT_LIMIT else ""))
    if log["run_patch"]["derived_pair_patches"]["count"]:
        out.append(f"run_patch derived-pair patches: {log['run_patch']['derived_pair_patches']['count']} (expected 0)")
    if log["run_patch"]["malformed_extra_json"]["count"]:
        out.append(f"run_patch fields.extra_json not an object: {log['run_patch']['malformed_extra_json']['count']} (expected 0)")
    if log["object_lineage_violations"]["count"]:
        out.append(f"object lineage violations: {log['object_lineage_violations']['count']} (expected 0)")
    if log["judgment"]["run_id_changes"]["count"] or log["judgment"]["judge_id_changes"]["count"]:
        out.append(f"judgment same-id records changed run_id ({log['judgment']['run_id_changes']['count']}) or judge_id "
                   f"({log['judgment']['judge_id_changes']['count']}) (expected 0)")
    if gf["grandfathered_unaccounted"]:
        out.append(f"grandfathered_unaccounted: {len(gf['grandfathered_unaccounted'])} listed sha(s) are no rule's hit in this log "
                   f"(a stale entry, or a record this log does not hold; --hash-all locates them): "
                   + ", ".join(s[:12] for s in gf["grandfathered_unaccounted"][:DEFAULT_LIMIT]))
    if log["misfiled"]:
        out.append("misfiled: records of an audited kind sit in files named for another kind (" + ", ".join(f"{k} {v}" for k, v in log["misfiled"].items())
                   + "); the default walk (the four kinds' files) would miss them -- audit with --every-file until the writer is found")
    return out


def run_audit(root: str, with_db: bool = True, reconcile: bool = True, paper_counts: bool = False, hash_all: bool = False,
              every_file: bool = False, limit: Optional[int] = DEFAULT_LIMIT, progress=None) -> Dict[str, object]:
    """The whole report as one JSON-ready object: ``log``, ``grandfather``, ``database`` (or why not), ``c1c`` (or why not),
    ``failures`` (the missed expectations) and ``ok``."""
    records_root = os.path.join(root, "records")
    log = walk_log(records_root, hash_all=hash_all, every_file=every_file, progress=progress)
    report: Dict[str, object] = {"root": root, "audited_utc": R.utc_now(), "log": log.as_dict(limit), "grandfather": grandfather_accounting(log)}
    db_path = DB.db_path(root)
    if with_db and os.path.exists(db_path):
        conn = DB.connect(db_path, readonly=True)
        try:
            report["database"] = audit_database(conn, paper_counts=paper_counts)
            report["c1c"] = c1c_buckets(root, conn) if reconcile else {"skipped": "--skip-reconcile"}
        finally:
            conn.close()
    else:
        why = "--no-db" if not with_db else f"no database at {db_path}"
        report["database"] = {"skipped": why}
        report["c1c"] = {"skipped": why}
    report["failures"] = expectations(report)
    report["ok"] = not report["failures"]
    return report


def format_report(report: Mapping[str, object]) -> str:
    """The text rendering: the log figures, the grandfather accounting, the live figures, the verdict."""
    log, gf = report["log"], report["grandfather"]
    scope = log["file_kinds"] if isinstance(log["file_kinds"], str) else "files of " + ", ".join(log["file_kinds"])
    lines = [f"audit of {report['root']} at {report['audited_utc']}",
             f"log ({scope}): {log['files_read']} files, {log['records_read']} records in {log['seconds']} s; by type: "
             + ", ".join(f"{k} {v}" for k, v in log["records_by_type"].items())
             + ("; MISFILED " + ", ".join(f"{k} {v}" for k, v in log["misfiled"].items()) if log["misfiled"] else ""),
             f"run {log['run']['records']}: index stated {log['run']['index_stated']} (rung_violations {log['rung_violations']['count']}); "
             f"ladder set, no index {log['run']['ladder_set_no_index']}; ladder empty or absent {log['run']['ladder_empty_or_absent']}; "
             f"code_sha unverified {log['run']['code_sha_unverified']}; placeholder_run_code_sha {log['placeholder_run_code_sha']['count']} "
             f"(grandfathered {gf['placeholder_run_code_sha']['grandfathered']}, not {len(gf['placeholder_run_code_sha']['not_grandfathered'])})",
             f"run_patch {log['run_patch']['records']}: derived-pair patches {log['run_patch']['derived_pair_patches']['count']}; decode / cap fields named "
             f"{log['run_patch']['decode_or_cap_fields_named']}; fields.extra_json not an object {log['run_patch']['malformed_extra_json']['count']}; "
             f"top-level extra {log['run_patch']['top_level_extra_records']} records / {log['run_patch']['top_level_extra_runs']} runs",
             f"judgment {log['judgment']['records']}: by form " + ", ".join(f"{k} {v}" for k, v in log["judgment"]["by_form"].items())
             + f"; distinct ids {log['judgment']['distinct_ids']}; multi-record ids {log['judgment']['multi_record_ids']} (max {log['judgment']['max_records_per_id']}); "
             f"run_id changes {log['judgment']['run_id_changes']['count']}; judge_id changes {log['judgment']['judge_id_changes']['count']}; "
             f"judgment_id_form_misses {log['judgment_id_form_misses']['count']} (grandfathered {gf['judgment_id_form_misses']['grandfathered']}, "
             f"not {len(gf['judgment_id_form_misses']['not_grandfathered'])})",
             f"object {log['object']['records']}: named_role stated {log['object']['named_role_stated']}; parent_object_id stated "
             f"{log['object']['parent_object_id_stated']}; object_lineage_violations {log['object_lineage_violations']['count']}",
             f"grandfather list: {gf['listed']} shas; unaccounted {len(gf['grandfathered_unaccounted'])}"]
    for rule in ("judgment_id_form_misses", "placeholder_run_code_sha", "rung_violations", "object_lineage_violations"):
        for h in log[rule]["records"]:
            tag = "grandfathered" if h["sha"] in VAL.GRANDFATHERED_RECORD_SHAS else "NOT GRANDFATHERED"
            detail = {k: v for k, v in h.items() if k not in ("sha", "uri", "written_utc")}
            lines.append(f"  {rule} {h['sha'][:12]} {tag} {h['uri']} {h['written_utc']} {json.dumps(detail, sort_keys=True)}")
        if log[rule]["count"] > log[rule]["listed"]:
            lines.append(f"  ... {log[rule]['count'] - log[rule]['listed']} more (--all)")
    db = report["database"]
    if "skipped" in db:
        lines.append(f"database: skipped ({db['skipped']})")
    else:
        rf, jd, cr, ol = db["refusals"], db["judgment"], db["currency_rules"], db["object_lineage"]
        lines.append(f"database: schema {db['schema_version']} (code {db['code_version']}); record_refusal {rf['current_refusals']} "
                     f"(judged_with column {'present' if rf['judged_with_column_present'] else 'absent'}); facts "
                     + (f"{rf['facts']} over {rf['fact_records']} records, unverified {rf['facts_unverified']}, backfilled {rf['facts_backfilled']}, "
                        f"current without a fact {rf['current_without_fact']}" if rf["fact_table_present"] else "table absent (pre-migration)")
                     + f"; blocked_records {rf['blocked_records']} (refused holds {rf['refused_holds']}); fact rows after the first cycle at most "
                     f"{rf['fact_rows_ceiling_after_first_cycle']}")
        lines.append("judgment live: " + ", ".join(f"{k} {v}" for k, v in jd["by_status"].items())
                     + f"; superseded without successor {jd['superseded_without_successor']}; by identity form "
                     + "; ".join(f"{f}: " + ", ".join(f"{s} {n}" for s, n in v.items()) for f, v in jd["live_rows_by_identity_form"].items()))
        d744, d757 = cr["d744_partial_displaced_by_complete"], cr["d757_identity_bound_at_claim"]
        lines.append(f"currency rules live: D744 partial beside a complete of the same (run, judge): {d744['partial_judgments']} judgments / "
                     f"{d744['run_judge_pairs']} pairs (witness identity-current: {d744['witness_identity_current']}); D757 binding columns "
                     f"{'present' if d757['binding_columns_present'] else 'absent'}, stated bindings {d757['stated_binding']}, stale-bound {d757['stale_bound']}")
        lines.append(f"object lineage live: columns {'present' if ol['columns_present'] else 'absent'}; extra_json parent_object_id {ol['extra_parent_object_id']}, "
                     f"named_role {ol['extra_named_role']}" + (f"; columns parent_object_id {ol['column_parent_object_id']}, named_role {ol['column_named_role']}"
                                                              if ol["columns_present"] else ""))
        if "paper_counts" in db:
            pc = db["paper_counts"]
            lines.append(f"paper: in_current_paper {pc['in_current_paper']} / selected {pc['selected']} of {pc['rows']} rows")
    c1c = report["c1c"]
    if "skipped" in c1c:
        lines.append(f"c1c buckets: skipped ({c1c['skipped']})")
    else:
        b, k, s = c1c["runs_by_bucket"], c1c["keys_by_bucket"], c1c["records_by_status"]
        lines.append(f"c1c buckets (dry run, nothing written): runs {c1c['runs']}: " + ", ".join(f"{n} {v}" for n, v in b.items() if v)
                     + f"; keys: missing {k['missing']} / equal {k['equal']} / conflict {k['conflict']}" + (f" / pending {k['pending']}" if k["pending"] else "")
                     + f"; lost records: accepted {s['accepted']} / refused {s['refused']} / blocked {s['blocked']} / not_ingested {s['not_ingested']}"
                     + f"; {c1c['records_read']} records in {c1c['files_read']} run_patch files")
    if report["failures"]:
        lines.append("FAILED expectations:")
        lines.extend(f"  - {f}" for f in report["failures"])
    else:
        lines.append("every pre-landing expectation holds: no violation, every hit grandfathered, every grandfathered sha accounted for")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.path.expanduser("~/registry"), help="the registry root (records/ and registry.sqlite under it)")
    ap.add_argument("--json", action="store_true", help="print the whole report as one JSON object")
    ap.add_argument("--out", default=None, help="also write the JSON report here (tmp + rename); refused under the root")
    ap.add_argument("--all", action="store_true", help="list every hit, not the first %d" % DEFAULT_LIMIT)
    ap.add_argument("--no-db", action="store_true", help="the log alone: no database read, no C1c buckets")
    ap.add_argument("--skip-reconcile", action="store_true", help="omit the C1c dry run")
    ap.add_argument("--paper-counts", action="store_true", help="add the paper's in_current_paper / selected totals (folds v_paper_current, about a minute)")
    ap.add_argument("--hash-all", action="store_true", help="hash every record read so an unaccounted grandfathered sha is located in the log")
    ap.add_argument("--every-file", action="store_true", help="walk every non-item-scale record file, not only the four audited kinds' files "
                                                               "(reports a record of those kinds filed under another kind's name)")
    ap.add_argument("--progress", action="store_true", help="a stderr line every 1,000 files")
    args = ap.parse_args(argv)
    root = os.path.abspath(os.path.expanduser(args.root))
    try:
        if args.out:
            out = os.path.abspath(os.path.expanduser(args.out))
            if os.path.commonpath([os.path.realpath(root), os.path.realpath(os.path.dirname(out) or ".")]) == os.path.realpath(root):
                raise AuditError(f"--out {args.out!r} is under the registry root {root!r}: the audit writes nothing there")
        progress = (lambda n, total: print(f"  {n}/{total} files", file=sys.stderr)) if args.progress else None
        report = run_audit(root, with_db=not args.no_db, reconcile=not args.skip_reconcile, paper_counts=args.paper_counts,
                           hash_all=args.hash_all, every_file=args.every_file, limit=None if args.all else DEFAULT_LIMIT, progress=progress)
    except (AuditError, OSError, R.RecordLineError, sqlite3.Error) as exc:
        print(f"audit_new_validator_rules: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.out:
        write_json_atomic(out, report, indent=1)
    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True))
    else:
        print(format_report(report))
    for f in report["failures"]:
        print(f"audit_new_validator_rules: FAILED {f}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
