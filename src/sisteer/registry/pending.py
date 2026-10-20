"""The judging work queue and the claim mechanism.

A run is PENDING for judge J iff its lifecycle state admits judging (generated / judging / judged /
ingested) and no COMPLETE judgment by J reaches the run's required coverage (``v_pending_judgment``,
``schema_def.py``). ``claim`` writes an in-progress ``judgment`` record into ``records/claims/<run_dir>/``
and ingests it under the harvester lock, so the next reader sees the claim immediately; a second
claimant is refused (:class:`ClaimHeld`) -- unless the claim it meets is bound to a since-corrected run identity, which holds nothing:
``claim`` closes such a claim ``abandoned`` and binds a new one to the current identity. After its scoped fold and under the
lock, ``claim`` also re-checks that (run, judge) is STILL pending in ``v_pending_judgment`` -- a run another session completed since the
daemon's selection is refused (:class:`ClaimNotPending`, a :class:`ClaimNotFolded`: re-select, nothing written), never given a second
complete judgment. ``worklist_parity`` compares the pending set to a judging
driver's own unit list set-wise (store → object → eval is the run grain).
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import db as DB
from . import harvest as H
from . import identity
from . import records as R
from . import schema_def as S
from . import validate as VAL
from .emit import RecordWriter

DEFAULT_STALE_HOURS = 12.0
#: how many times :func:`claim` re-mints a claim whose binding the fold moved out from under it before giving up (loudly)
REBIND_CAP = 3
log = logging.getLogger(__name__)

#: How long a claim / release / driver-side fold WAITS for the harvester's write lock before refusing (``LockHeld``). Default 30 s;
#: ``SISTEER_REGISTRY_LOCK_TIMEOUT_S`` RAISES it for a caller that must outlast a harvester cycle — the grading daemon sets 300 s for the
#: graders it launches (2026-09-30: every grader it launched died ``LockHeld`` at its first claim while the harvester's back-to-back cycles
#: held the lock 99.7 % of the time). The claim is a real write under the harvester's lock (its record is spooled and folded at once).
LOCK_TIMEOUT_S = float(os.environ.get("SISTEER_REGISTRY_LOCK_TIMEOUT_S") or 30.0)

#: ``v_pending_judgment`` columns a worklist may filter on by equality (a judging driver's query).
FILTER_COLUMNS: Tuple[str, ...] = ("campaign", "eval_id", "eval_category", "harness", "instrument_id", "model_id",
                                   "object_id", "lifecycle_state", "decode_source", "pending_reason", "run_id")
#: the lifecycle states under which a run is pending for any judge at all (``v_pending_judgment``'s WHERE, ``schema_def.py``; read by
#: :func:`not_pending_why` and :func:`_not_pending_reason`)
JUDGEABLE_LIFECYCLE_STATES: Tuple[str, ...] = ("generated", "judging", "judged", "ingested")


class ClaimHeld(RuntimeError):
    """Another session holds an identity-current in-progress claim on (run, judge). Never raised over a claim whose binding differs
    from the run's current identity: that claim holds nothing, and :func:`claim` closes it instead."""

    def __init__(self, run_id: str, session: str, age_s: int):
        super().__init__(f"run {run_id} is claimed by session {session!r} ({age_s} s ago)")
        self.run_id, self.session, self.age_s = run_id, session, age_s


class ClaimPrecheckRefused(ClaimHeld):
    """:func:`claim`'s ``precheck`` hook: the caller's pre-write check refused the
    claim under the write lock. A DISTINCT refusal from :class:`ClaimHeld` proper -- held by another claimant means retry later; this means the
    claim's premise is stale (the worklist's designated-judge comparison, a precheck that attempted a write), so the caller re-selects and never
    retries blindly. Still a :class:`ClaimHeld`, so every driver's existing refusal names it; the CLI exits 1 naming the class; the grade daemon
    reads the class name, or the message's stable opening ``claim refused (precheck)``, as its transient `identity_changed` kind."""

    def __init__(self, run_id: str, reason: str) -> None:
        RuntimeError.__init__(self, f"claim refused (precheck): run {run_id}: {reason}")
        self.run_id, self.session, self.age_s, self.reason = run_id, "", 0, reason


class ClaimNotFolded(RuntimeError):
    """The claim (or release) cannot fold -- the run's record is still in the spool (no run row), or the judge has no row -- so the
    judgment would be dependency-blocked and no reader would see the claim (a second claimant would proceed too, and a later full
    cycle would activate both). Refused by name BEFORE anything is written (:func:`_require_parents`); the caller retries after the
    harvester folds the run; nothing falls back to a full cycle."""

    def __init__(self, run_id: str, what: str):
        super().__init__(f"run {run_id} is not folded yet; retry after the harvester folds it ({what})")
        self.run_id = run_id


class ClaimPendingCorrection(ClaimNotFolded):
    """An identity correction of the run -- a ``run`` record moving its instrument or its fallback judge, a ``run_patch`` moving
    ``judge_id`` where the instrument names no judge -- is PENDING outside the run's own record files, where :func:`claim`'s scoped
    fold-first cannot reach it: a blocked record, a constituent of a held adapter revision, an unprocessed revision directory, an unfolded
    activation bundle of a revision that carried the run, a cluster push still in the inbox (:func:`pending_identity_corrections`, whose
    predicate is literally "would the next cycle fold a binding-moving record of this run?" -- every hit names such a record, never a
    parked one, so a refusal always has a route out within one cycle). A claim made now would bind the identity the
    correction is about to replace and read ``identity_stale`` the moment the harvester's cycle folds it, the grading wasted.
    Refused by name BEFORE anything is written; the contract is :class:`ClaimNotFolded`'s (a subclass: every
    caller that retries on it retries on this; the CLI exits 1): retry after the next harvester cycle."""

    def __init__(self, run_id: str, what: str):
        RuntimeError.__init__(self, f"run {run_id}: an identity correction of this run is pending outside its own record files -- {what}; "
                                    "retry after the next harvester cycle (nothing written)")
        self.run_id = run_id


class ClaimNotPending(ClaimNotFolded):
    """(run, judge) is NOT PENDING any more: after the scoped fold-first,
    ``v_pending_judgment`` -- the selection the judging daemon chose the run from -- no longer lists the pair, because a complete,
    identity-current judgment by this judge now covers the run (another session's completion folded between the daemon's selection and
    this claim) or the run's lifecycle state admits no judging any more. A claim minted now would be a SECOND claim and complete into a
    second complete judgment of the same (run, judge), which no currency rule excludes (they exclude superseded, abandoned and displaced
    partial judgments, never a second complete). Refused by name BEFORE anything is written, under the claim lock; the contract is
    :class:`ClaimNotFolded`'s (a subclass: every caller that retries on it retries on this; the CLI prints ``claim refused: ClaimNotPending: ...`` and exits
    1): re-select from the worklist, never a quarantine. The one pair ``v_pending_judgment`` never lists that is still claimable -- a
    judge-less instrument whose fallback ``run.judge_id`` has filed nothing (the pairing limitation) -- is not refused
    (:func:`not_pending_why`). ``force`` is unaffected unless the pair is genuinely complete: a stale in-progress claim leaves the run
    listed (``claimed``), so the takeover proceeds as before."""

    def __init__(self, run_id: str, judge_id: str, why: str):
        RuntimeError.__init__(self, f"run {run_id}: not pending for judge {judge_id} any more -- {why}; re-select (nothing written)")
        self.run_id, self.judge_id, self.why = run_id, judge_id, why


class UnknownJudge(ValueError):
    """``--judge`` named neither a ``judge.judge_id`` nor a seeded readable id (``extra_json.seed_judge_id``)."""


class UnknownFilter(ValueError):
    """A worklist filter named a column ``v_pending_judgment`` does not expose."""


# --------------------------------------------------------------------------------------------
# the queue
# --------------------------------------------------------------------------------------------


def resolve_judge_id(conn: sqlite3.Connection, ident: str) -> str:
    """The ``judge.judge_id`` (the spec's hash) for ``ident``: the hash itself, or the readable seed id the
    dimension loader parks in ``extra_json.seed_judge_id`` (``configs/registry/judges.yaml``). Refuses loudly
    when neither matches — a missing dimension is reported, never guessed (authoritative-sources rule)."""
    if conn.execute("SELECT 1 FROM judge WHERE judge_id = ?", (ident,)).fetchone():
        return ident
    rows = conn.execute("SELECT judge_id FROM judge WHERE json_extract(extra_json, '$.seed_judge_id') = ?", (ident,)).fetchall()
    if len(rows) == 1:
        return str(rows[0][0])
    if len(rows) > 1:
        raise UnknownJudge(f"judge {ident!r} is the seed id of {len(rows)} judge rows; name the judge_id hash")
    n = conn.execute("SELECT COUNT(*) FROM judge").fetchone()[0]
    raise UnknownJudge(f"judge {ident!r} is neither a judge_id nor a seeded judge id in this registry "
                       f"({n} judge rows; the dimension seed may not be loaded)")


def _filter_sql(campaign: Optional[str], filters: Mapping[str, object]) -> Tuple[str, List[object]]:
    sql, params = "", []
    if campaign:
        sql += " AND campaign = ?"
        params.append(campaign)
    for col, val in filters.items():
        if col not in FILTER_COLUMNS:
            raise UnknownFilter(f"worklist filter {col!r} is not a v_pending_judgment column; one of {FILTER_COLUMNS}")
        if isinstance(val, (list, tuple, set, frozenset)):
            vals = list(val)
            sql += f" AND {col} IN ({','.join('?' * len(vals))})"
            params.extend(vals)
        else:
            sql += f" AND {col} = ?"
            params.append(val)
    return sql, params


def pending(conn: sqlite3.Connection, judge_id: Optional[str] = None, campaign: Optional[str] = None,
            **filters) -> List[Dict[str, object]]:
    """The pending set: for ``judge_id`` (hash or seed id) when given, else for each run's INSTRUMENT judge."""
    if judge_id:
        return pending_for(conn, judge_id, campaign, **filters)
    sql = "SELECT * FROM v_pending_judgment WHERE is_instrument_judge = 1"
    more, params = _filter_sql(campaign, filters)
    return [dict(r) for r in conn.execute(sql + more + " ORDER BY run_id", params).fetchall()]


def pending_for(conn: sqlite3.Connection, judge_id: str, campaign: Optional[str] = None,
                decode_source_not: Optional[str] = None, **filters) -> List[Dict[str, object]]:
    """Runs pending for the requested judge (the view's WHERE parameter), optionally within one campaign and
    narrowed by equality filters on :data:`FILTER_COLUMNS` (a tuple value means ``IN``);
    ``decode_source_not`` excludes one instrument decode source (the sampled static grade's ``≠ greedy``)."""
    jid = resolve_judge_id(conn, judge_id)
    sql = "SELECT * FROM v_pending_judgment WHERE required_judge_id = ?"
    params: List[object] = [jid]
    more, more_params = _filter_sql(campaign, filters)
    sql += more
    params += more_params
    if decode_source_not:
        sql += " AND decode_source <> ?"
        params.append(decode_source_not)
    sql += " ORDER BY run_id"
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def spool_complete_judgments(root: str, campaign: Optional[str], run_id: str) -> List[Dict[str, object]]:
    """Every COMPLETE ``judgment`` record for the run on the registry SPOOL (``<root>/records/<campaign>/<run dir>/*-judgment.jsonl[.gz]``,
    written by a grader and not necessarily folded yet): ``[{judgment_id, judge_id, n_items_judged, claimed_by_session, written_utc}]`` in
    file order. The durable boundary a claim's re-dispatch must read: ``JudgingScope.complete`` writes the
    completion HERE while the claim row in the database stays ``in_progress`` until the harvester folds it, so ``v_pending_judgment`` reads
    ``claimed`` (and a never-ingested run still ``generated``) for a judgment that is already durable. A worklist or a daemon that would
    re-dispatch a claim of its own session checks this first: a complete record under the claim's judgment id means FILED, never dead.
    ``campaign`` None reads the run id's campaign prefix. An unreadable file is skipped, never fatal."""
    from . import identity as I
    out: List[Dict[str, object]] = []
    d = os.path.join(H.records_dir(root), str(campaign or run_id.split(":", 1)[0]), I.run_dir_name(run_id))
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if not (f.endswith(".jsonl") or f.endswith(".jsonl.gz")):
            continue
        opener = gzip.open if f.endswith(".gz") else open
        try:
            with opener(os.path.join(d, f), "rt", encoding="utf-8") as fh:
                for ln in fh:
                    try:
                        rec = json.loads(ln)
                    except ValueError:
                        continue
                    p = rec.get("payload") or {}
                    if rec.get("record_type") == "judgment" and p.get("status") == "complete" and p.get("judgment_id"):
                        out.append({"judgment_id": str(p["judgment_id"]), "judge_id": p.get("judge_id"), "n_items_judged": p.get("n_items_judged"),
                                    "claimed_by_session": p.get("claimed_by_session"), "written_utc": rec.get("written_utc")})
        except (OSError, EOFError):
            continue
    return out


def active_claim(conn: sqlite3.Connection, run_id: str, judge_id: str) -> Optional[Dict[str, object]]:
    """The live in-progress judgment for (run, judge) -- an identity-current one (its binding at claim equal to the run's folded
    identity, :func:`schema_def.judgment_identity_current_sql`, the test ``v_pending_judgment`` applies to its active claim), or None.
    A claim bound to a since-corrected identity is held by nobody and is not live here either, so a worklist that reads
    its claim back through this function is never handed a judgment every reader discards."""
    row = conn.execute("SELECT j.judgment_id, j.claimed_by_session, j.claimed_utc FROM judgment j JOIN run r ON r.run_id = j.run_id "
                       "LEFT JOIN instrument i ON i.instrument_id = r.instrument_id "
                       f"WHERE j.run_id = ? AND j.judge_id = ? AND j.status = 'in_progress' AND {S.judgment_identity_current_sql('j', 'r', 'i')} "
                       "ORDER BY j.claimed_utc DESC LIMIT 1", (run_id, judge_id)).fetchone()
    return None if row is None else {"judgment_id": row[0], "claimed_by_session": row[1], "claimed_utc": row[2]}


# --------------------------------------------------------------------------------------------
# claims (under the harvester lock)
# --------------------------------------------------------------------------------------------


def _claim_payload(run_id: str, judge_id: str, session: str, now: str, status: str = "in_progress",
                   judgment_id: Optional[str] = None, claimed_utc: Optional[str] = None,
                   claimed_instrument_id: Optional[str] = None, claimed_required_judge_id: Optional[str] = None) -> Dict[str, object]:
    """A claim's ``judgment`` payload. A NEW claim is minted over ``now`` (``identity.judgment_id(run, judge, session, now)``) and
    states ``claimed_utc = now``. A record ABOUT an existing claim -- the ``abandoned`` record :func:`release` and a forced
    :func:`claim` write for its id -- passes the claim's own ``claimed_utc``: the id was minted over that time, and
    the validator's identity rule (``identity.judgment_id_form``) refuses a judgment record whose id no form over its own fields
    reproduces, so an abandonment stamped with the release time instead of the claim's would be refused and the stale
    ``in_progress`` row would keep competing with the next claim. Before this the release path stamped ``now`` (the 18 such records in
    the log are grandfathered by sha, ``validate.GRANDFATHERED_RECORD_SHAS``); the forced-takeover path had the same defect and had
    written none. A NEW claim also binds the run's identity AS IT STANDS: ``claimed_instrument_id`` and
    ``claimed_required_judge_id`` -- the run row's ``instrument_id`` and its required judge (the instrument's ``judge_id``, else
    ``run.judge_id``), read by :func:`claim` under the lock (:func:`run_identity_binding`) -- so a completion that folds after a same-id
    run identity correction reads non-current on the paper side (``schema_def.judgment_identity_current_sql``) while the run is listed
    for the new judge. Each binding is stated only when known (an absent one is NULL and reads as current); a record ABOUT an existing
    claim never restates them (the per-column fold keeps the claim's), and the id is minted over run / judge / session / time as before."""
    when = claimed_utc or now
    jid = judgment_id or identity.judgment_id(run_id, judge_id, session, when)
    out: Dict[str, object] = {"judgment_id": jid, "run_id": run_id, "judge_id": judge_id, "status": status,
                              "claimed_by_session": session, "claimed_utc": when}
    if claimed_instrument_id is not None:
        out["claimed_instrument_id"] = claimed_instrument_id
    if claimed_required_judge_id is not None:
        out["claimed_required_judge_id"] = claimed_required_judge_id
    return out


def run_identity_binding(conn: sqlite3.Connection, run_id: str) -> Tuple[Optional[str], Optional[str]]:
    """The run's identity a claim binds: ``(instrument_id, required judge)`` from the live run row -- the required judge being the
    instrument's ``judge_id`` when the instrument names one, else ``run.judge_id`` (exactly the pairing ``v_pending_judgment`` uses) --
    or ``(None, None)`` when the run row is not there (the claim is dependency-blocked on it anyway and binds nothing)."""
    row = conn.execute("SELECT r.instrument_id, COALESCE(i.judge_id, r.judge_id) FROM run r LEFT JOIN instrument i ON i.instrument_id = r.instrument_id "
                       "WHERE r.run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None, None
    return (str(row[0]) if row[0] is not None else None), (str(row[1]) if row[1] is not None else None)


def _write_claims(root: str, run_id: str, session: str, now: str, payloads: Sequence[Mapping[str, object]]) -> str:
    """One judgment record file under ``records/claims/<run_dir>/`` (the producer-side writer, stamped ``now``); returns its path --
    what the caller folds under the lock (only this file, never a harvester cycle)."""
    prod = {"script": "sisteer.registry.pending", "code_sha": "unverified", "session": session,
            "host": os.uname().nodename, "substrate": "grading_host"}
    w = RecordWriter(H.records_dir(root), "claims", run_id, prod, disabled=False)
    path = w.write_many("judgment", payloads, written_utc=now)
    assert path is not None
    return path


def _fold_own(root: str, conn: sqlite3.Connection, path: str) -> H.IngestReport:
    """Fold the claim file just written -- that file ALONE -- so the next reader sees it."""
    return H.ingest_once(root, conn, workers=0, derive_facts=False, min_free=0, only_files=[path])


def run_record_files(root: str, run_id: str) -> List[str]:
    """The run's OWN record files in the log -- every ``records/<campaign>/<run_dir>/<seq>-<kind>.jsonl[.gz]`` of every kind (the
    directory every writer puts a run's records in: ``emit.RecordWriter`` over ``identity.run_dir_name``; the claims directory included),
    delivered or not; the scoped fold skips the delivered ones by sha (:func:`harvest.fold_only_files`). These are the run's own producer's
    statements: a same-id ``run`` record correcting its identity, a ``run_patch`` moving its fallback ``judge_id``, its completions and
    state events, so :func:`claim` folds them before it reads the identity it binds (the ``run`` kind alone left a ``run_patch`` of
    ``judge_id`` unfolded). Revision directories (atomic per revision, the harvester's), the inbox (the
    cycle's to merge) and every other run's files are not listed: a scope, never a spool listing."""
    run_dir = identity.run_dir_name(run_id)
    base = H.records_dir(root)
    out: List[str] = []
    try:
        subs = sorted(os.listdir(base))
    except OSError:
        return out
    for sub in subs:
        d = os.path.join(base, sub, run_dir)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not R.RECORD_FILE_RE.match(fn):
                continue
            p = os.path.join(d, fn)
            if os.path.isfile(p) and not H.revision_of_uri(H.logical_uri(root, p)):
                out.append(p)
    return out


def _fold_run_records(root: str, conn: sqlite3.Connection, run_id: str) -> H.IngestReport:
    """Fold the run's own spooled records (:func:`run_record_files`) -- a same-id identity correction a producer wrote and the harvester
    has not folded yet among them -- and nothing else (a scoped fold, never a cycle), so the identity :func:`claim` binds is the one
    the fold leaves. Nothing on the spool for the run = an empty report, no fold."""
    files = run_record_files(root, run_id)
    if not files:
        return H.IngestReport()
    return H.ingest_once(root, conn, workers=0, derive_facts=False, min_free=0, only_files=files)


#: An adapter revision directory's name (``records/adapters/<adapter>/<rev>/``, ``harvest.REVISION_DIR_RE``'s plain form).
_REVISION_NAME_RE = re.compile(r"^[0-9a-f]{16}$")
#: The record kinds that can move a run's identity binding: a ``run`` record (its ``instrument_id`` and fallback ``judge_id``) and a
#: ``run_patch`` (``fields.judge_id``, patchable; ``instrument_id`` is identity and not).
_BINDING_KINDS: Tuple[str, ...] = ("run", "run_patch")


def _required_judge_for(conn: sqlite3.Connection, instrument_id: Optional[str], run_judge: Optional[str]) -> Tuple[Optional[str], bool]:
    """``(required judge, instrument known)`` for a run on ``instrument_id`` with fallback ``run_judge``: the instrument's ``judge_id`` when
    it names one, else the run's. An instrument with no row yet is unknown (the correction naming it is dependency-blocked until it lands)."""
    if instrument_id is None:
        return run_judge, True
    row = conn.execute("SELECT judge_id FROM instrument WHERE instrument_id = ?", (instrument_id,)).fetchone()
    if row is None:
        return None, False
    return (str(row[0]) if row[0] is not None else run_judge), True


def _standing_stamps(conn: sqlite3.Connection, run_id: str) -> Dict[str, str]:
    """``{column: written_utc}`` of the record that last set each column of the run row (``field_versions_json``): the fold is per-column
    latest statement, so a record stamped BEFORE the standing statement of a column moves it nothing when it folds."""
    row = conn.execute("SELECT field_versions_json FROM run WHERE run_id = ?", (run_id,)).fetchone()
    if row is None or not row[0]:
        return {}
    try:
        fv = json.loads(row[0])
    except ValueError:
        return {}
    return {c: str(v[0]) for c, v in fv.items() if isinstance(v, list) and v and isinstance(v[0], str)} if isinstance(fv, Mapping) else {}


def binding_move_of(conn: sqlite3.Connection, run_id: str, rec: Mapping[str, object], folded: Tuple[Optional[str], Optional[str]],
                    run_judge: Optional[str], stamps: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Why ``rec`` (a record envelope) would move the run's identity binding away from ``folded`` = ``(instrument_id, required judge)``
    when it folds -- or None when it is not about this run, not a binding kind, restates the identity (a held revision re-reading the same
    store restates it; refusing on that would refuse grading for as long as the hold lasts), or is stamped before the standing statement
    of every column it would move (``stamps``, :func:`_standing_stamps`: the fold is per-column latest statement, so such a record folds
    into the history and moves the row nothing). ``run_judge`` is the folded row's own ``judge_id`` (the fallback a ``run_patch`` moves)."""
    kind = rec.get("record_type")
    payload = rec.get("payload") if isinstance(rec.get("payload"), Mapping) else None
    if kind not in _BINDING_KINDS or payload is None or payload.get("run_id") != run_id:
        return None
    when = str(rec.get("written_utc") or "")
    stamps = stamps if stamps is not None else _standing_stamps(conn, run_id)

    def older_than_standing(column: str) -> bool:
        return bool(when) and column in stamps and when < stamps[column]

    if kind == "run":
        # the effective fold, per column (latest statement): a column this record states AFTER its standing statement takes the record's
        # value, one it states before keeps the standing one -- then the binding that fold leaves is compared with the folded binding
        inst = payload.get("instrument_id")
        inst = str(inst) if inst is not None else None
        judge = payload.get("judge_id")
        judge = str(judge) if judge is not None else None
        eff_inst = folded[0] if older_than_standing("instrument_id") else inst
        eff_judge = run_judge if older_than_standing("judge_id") else judge
        required, known = _required_judge_for(conn, eff_inst, eff_judge)
        if not known:
            return f"a run record moves the instrument to {eff_inst} (no instrument row yet)"
        if (eff_inst, required) != folded:
            return f"a run record moves the binding {folded} -> {(eff_inst, required)}"
        return None
    fields = payload.get("fields") if isinstance(payload.get("fields"), Mapping) else {}
    if "judge_id" not in fields or older_than_standing("judge_id"):
        return None
    new_judge = fields.get("judge_id")
    new_judge = str(new_judge) if new_judge is not None else None
    if new_judge == run_judge:
        return None
    required, _known = _required_judge_for(conn, folded[0], new_judge)
    if required != folded[1]:
        return f"a run_patch moves judge_id {run_judge} -> {new_judge} (the required judge {folded[1]} -> {required})"
    return None


def _records_of(path: str) -> Iterable[Tuple[int, Dict[str, object]]]:
    """Every record of a file as ``(line, envelope)``; an unreadable file yields nothing (the harvester refuses it by name)."""
    try:
        for n, rec, _sha in R.read_record_file(path, compute_sha=False):
            yield n, rec
    except (OSError, EOFError, ValueError):
        return


def _physical(root: str, uri: str) -> Optional[str]:
    """The on-disk path of a logical record uri (plain or gzipped), or None."""
    for cand in (os.path.join(root, uri), os.path.join(root, uri + ".gz")):
        if os.path.isfile(cand):
            return cand
    return None


def _is_marker_file(name: str) -> bool:
    """The harvester's marker convention (``harvest.Ingestor._find_marker``, ``adapters.runner._is_marker_file``): the
    ``<seq>-source_revision.jsonl[.gz]`` file of a revision directory or an activation bundle."""
    return "source_revision" in name and (name.endswith(".jsonl") or name.endswith(".jsonl.gz"))


def _marker_complete(d: str) -> bool:
    """Whether the revision directory or bundle ``d`` holds its marker file: published whole by the runner's one rename, so the next cycle
    applies it. A directory without one is a crash between staging and publication (or a publisher mid-rename) -- its files wait, nothing
    applies, and a claim must not wait on it."""
    try:
        return any(_is_marker_file(f) for f in os.listdir(d))
    except OSError:
        return False


def _marker_folded(conn: sqlite3.Connection, root: str, d: str) -> bool:
    """Whether the harvester has processed the marker of ``d`` (``adapters.runner._marker_folded``): a ``record_delivery`` row for its
    logical uri -- written when the revision or bundle activated and when it was held, never before."""
    for f in sorted(os.listdir(d)):
        if _is_marker_file(f):
            return conn.execute("SELECT 1 FROM record_delivery WHERE file_uri = ? LIMIT 1", (H.logical_uri(root, os.path.join(d, f)),)).fetchone() is not None
    return False


def _foldable(rec: Mapping[str, object]) -> bool:
    """A raw record the next cycle would FOLD rather than refuse: the validator admits it (a malformed correction is a refusal fact, not a
    pending identity)."""
    return isinstance(VAL.validate_record(rec), VAL.Validated)


def _run_drop_hold_stands(conn: sqlite3.Connection, root: str, rev: str, marker_path: str) -> bool:
    """Whether a ``kind = run_drop`` hold of ``rev`` trips AGAIN on the next cycle's re-attempt -- the guard's own comparison
    (``harvest.Ingestor._run_set_drop_hold``: the head's run ids from its ``observation`` rows, the revision's from its marker's
    ``observation`` records plus its own rows, the declaration from the marker record's extra; a NET SHRINK with an undeclared drop) made
    with the harvester's functions (:func:`harvest.run_ids_of_observation_keys`, :func:`harvest.declared_run_drops_of`,
    :func:`harvest.run_set_drop_comparison`). False when the re-attempt would pass -- the head moved, or a declaration arrived on this
    marker -- and the next cycle folds the revision."""
    marker_valid = [rec for _n, rec in _records_of(marker_path)]
    mrec = next((rec for rec in marker_valid if rec.get("record_type") == "source_revision" and isinstance(rec.get("payload"), Mapping)
                 and rec["payload"].get("revision_id") == rev), None)
    if mrec is None or not int((mrec["payload"].get("accepted") or 0)):
        return True
    if not H.run_set_drop_rule_applies(str(mrec["payload"].get("observed_utc") or mrec.get("written_utc") or "")):
        return False                   # the versioned rule does not judge a marker observed before it landed: the re-attempt passes
    source_id = str(mrec["payload"].get("source_id") or "")
    head = conn.execute("SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC, revision_id DESC LIMIT 1",
                        (source_id,)).fetchone()
    if head is None or str(head[0]) == rev:
        return False
    prev_ids = H.run_ids_of_observation_keys(k for (k,) in conn.execute("SELECT observation_key FROM observation WHERE source_id = ? AND revision_id = ?",
                                                                        (source_id, str(head[0]))))
    if not prev_ids:
        return False
    new_keys: List[object] = [k for rec in marker_valid if rec.get("record_type") == "observation" and isinstance(rec.get("payload"), Mapping)
                              for k in (rec["payload"].get("observation_keys") or [])]
    new_keys += [k for (k,) in conn.execute("SELECT observation_key FROM observation WHERE source_id = ? AND revision_id = ?", (source_id, rev))]
    new_ids = H.run_ids_of_observation_keys(new_keys)
    if not (prev_ids - new_ids) or len(new_ids) >= len(prev_ids):
        return False
    declaration, _malformed = H.declared_run_drops_of(mrec["payload"].get("extra") if isinstance(mrec["payload"].get("extra"), Mapping) else None)
    return bool(H.run_set_drop_comparison(prev_ids, new_ids, declaration)["tripped"])


#: A dimension seed revision directory's name (``records/dimensions/<rev>/``, ``harvest.DIMENSION_DIR_RE``): applied FIRST in a cycle.
_DIMENSION_NAME_RE = re.compile(r"^[0-9a-f]{16}$")


def _delivered(conn: sqlite3.Connection, root: str, path: str) -> bool:
    return conn.execute("SELECT 1 FROM record_delivery WHERE file_uri = ? LIMIT 1", (H.logical_uri(root, path),)).fetchone() is not None


def _key_column(ptable: str) -> Optional[str]:
    keys = R.NATURAL_KEYS.get(ptable)
    return keys[0] if keys and len(keys) == 1 else None


def _spooled_dimension_parents(root: str, conn: sqlite3.Connection) -> Dict[Tuple[str, str], str]:
    """``{(table, id): bundle revision}`` of every dimension record in an UNDELIVERED ``records/dimensions/<rev>/`` file (21 bundles / 189
    files live; each undelivered file read once per probe): the parents the next cycle folds first."""
    out: Dict[Tuple[str, str], str] = {}
    base = os.path.join(H.records_dir(root), "dimensions")
    try:
        revs = sorted(os.listdir(base))
    except OSError:
        return out
    for rev in revs:
        if not _DIMENSION_NAME_RE.match(rev):
            continue
        ddir = os.path.join(base, rev)
        try:
            names = sorted(os.listdir(ddir))
        except OSError:
            continue
        for fn in names:
            if not R.RECORD_FILE_RE.match(fn):
                continue
            path = os.path.join(ddir, fn)
            if _delivered(conn, root, path):
                continue
            for _n, rec in _records_of(path):
                rt = rec.get("record_type")
                col = _key_column(str(rt)) if rt else None
                payload = rec.get("payload") if isinstance(rec.get("payload"), Mapping) else None
                if col and payload is not None and payload.get(col) is not None:
                    out.setdefault((str(rt), str(payload[col])), rev)
    return out


class _ParentProbe:
    """Per-probe memo of :func:`_parent_pending`: each parent's presence is asked once and the spooled dimension parents are read once."""

    def __init__(self, root: str, conn: sqlite3.Connection):
        self.root, self.conn = root, conn
        self._spooled: Optional[Dict[Tuple[str, str], str]] = None
        self._memo: Dict[Tuple[str, str], Optional[str]] = {}

    def pending(self, ptable: str, pid: str) -> Optional[str]:
        key = (ptable, pid)
        if key not in self._memo:
            self._memo[key] = _parent_pending(self.root, self.conn, ptable, pid, self)
        return self._memo[key]

    def spooled(self) -> Dict[Tuple[str, str], str]:
        if self._spooled is None:
            self._spooled = _spooled_dimension_parents(self.root, self.conn)
        return self._spooled


def _parent_pending(root: str, conn: sqlite3.Connection, ptable: str, pid: str, memo: Optional[_ParentProbe] = None) -> Optional[str]:
    """Why the blocked queue would re-attempt a record waiting on parent ``(ptable, pid)`` in the NEXT cycle -- or None. The queue
    re-attempts once the parent is present (:func:`harvest.parent_absent`), and the cycle folds its other work BEFORE it re-evaluates the
    queue (``harvest.ingest_once``: dimension bundles first, then the plain files and revision directories, then
    ``reevaluate_blocked``), so a parent whose record is already on the spool counts as present for that cycle. Looked for, bounded: a
    dimension bundle ``records/dimensions/<rev>/`` file ``*-<ptable>.jsonl`` with no ``record_delivery`` row (21 bundles / 189 files live)
    naming ``pid``; for a ``run`` parent its own record files (:func:`run_record_files`); for a ``source_revision`` parent its
    marker-complete, unfolded directory. A parent spooled anywhere else (a plain dimension file outside a bundle, an ``object`` under its
    producer's directory) is a KNOWN MISS of this probe (the table in :func:`pending_identity_corrections`)."""
    if not H.parent_absent(conn, ptable, pid):
        return "its parent is present"
    col = _key_column(ptable)
    if ptable == "source_revision":
        for adapter in _adapter_dirs(root):
            rdir = os.path.join(H.records_dir(root), "adapters", adapter, pid)
            if os.path.isdir(rdir) and _marker_complete(rdir) and not _marker_folded(conn, root, rdir):
                return f"its parent revision {pid} is on the spool"
        return None
    if ptable == "run":
        for path in run_record_files(root, pid):
            if _delivered(conn, root, path):
                continue
            if any(rec.get("record_type") == "run" and isinstance(rec.get("payload"), Mapping) and rec["payload"].get("run_id") == pid
                   for _n, rec in _records_of(path)):
                return f"its parent run's record {H.logical_uri(root, path)} is on the spool"
        return None
    if col is None:
        return None
    spooled = memo.spooled() if memo is not None else _spooled_dimension_parents(root, conn)
    rev = spooled.get((ptable, pid))
    return f"its parent {ptable} {pid} is in the spooled dimension bundle {rev}" if rev else None


def _adapter_dirs(root: str) -> List[str]:
    try:
        return sorted(d for d in os.listdir(os.path.join(H.records_dir(root), "adapters")) if not d.startswith("_"))
    except OSError:
        return []


def _adapter_and_revision(uri: str) -> Optional[Tuple[str, str]]:
    """``(adapter, revision)`` of a record uri under ``records/adapters/<adapter>/<rev>/`` or ``…/_activations/<rev>/<stamp>/``, else None."""
    parts = uri.split("/")
    if len(parts) < 4 or parts[0] != "records" or parts[1] != "adapters":
        return None
    adapter = parts[2]
    rev = parts[4] if parts[3] == "_activations" and len(parts) > 4 else parts[3]
    return (adapter, rev) if _REVISION_NAME_RE.match(rev) else None


def pending_identity_corrections(root: str, conn: sqlite3.Connection, run_id: str) -> List[str]:
    """The records outside the run's own files that the NEXT harvester cycle would fold and that would move the run's identity binding,
    each as a short reason naming the record and where it waits -- what :func:`claim` refuses on (:class:`ClaimPendingCorrection`). The
    predicate is literally "would the next cycle fold a binding-moving record of this run?": every hit is compared against the FOLDED
    binding (:func:`binding_move_of`; a record restating the identity is no correction) and each class below follows the harvester's own
    rule for what its next cycle does, taken from ``harvest.py`` rather than re-encoded. The run's
    own files (``records/<campaign>/<run_dir>/``) are not here: :func:`claim` folds them first.

    **Contract.** This probe is a BEST-EFFORT pre-claim check whose purpose is to avoid a WASTED GRADING. The CORRECTNESS belt is the
    identity rule, not the probe: a judgment carries the identity bound at claim (``claimed_instrument_id``,
    ``claimed_required_judge_id``); a correction folded later makes it identity-stale -- excluded from coverage everywhere
    (``schema_def.judgment_identity_current_sql``); ``v_pending_judgment`` then lists the run as pending again for that judge
    (``pending_reason = identity_stale``); and the next :func:`claim` binds the current identity (a stale-bound claim is closed, never
    held). A probe miss therefore costs ONE re-graded run, never a wrong row -- ``tests/test_registry_pending_correction_probe.py``'s
    ``test_a_probe_miss_costs_one_regrade_never_a_wrong_row`` proves the belt end to end for a forced miss, which is what lets a future gap in
    the table below be a cost, not a blocker. The table is kept honest: seen / seen-when / excluded / known misses, each with the harvester
    rule it follows. **Scope of the one-re-grade guarantee.** It holds for INSTRUMENT-JUDGED runs -- the instrument names the judge
    (``is_instrument_judge = 1``), the grade daemon's domain. For a run on a judge-less instrument whose fallback ``run.judge_id`` moves
    J1 -> J2, the J1 judgment becomes identity-stale and the run stays VISIBLE in ``v_pending_judgment`` under J1 with
    ``pending_reason = identity_stale`` and ``is_instrument_judge = 0``, but no daemon re-grade happens (its default selection takes
    ``is_instrument_judge = 1``) and the view does not list J2 until J2 has a judgment on the run -- a known limitation of the pending view
    (its judge join for a judge-less instrument is "a judge with a judgment on the run"; follow-up: the current required judge as a
    candidate row); the PRODUCER re-judges with J2, and that claim binds the new fallback.
    ``test_the_one_regrade_guarantee_is_scoped_to_instrument_judged_runs`` pins exactly this.

    =======================================================  ==========  ==================================================================
    class                                                    probed      the harvester rule it follows (harvest.py)
    =======================================================  ==========  ==================================================================
    marker-complete revision directory with no               seen        ``records/adapters/<adapter>/<rev>/`` published by one rename, no
    ``source_revision`` row                                              ``source_revision`` row yet: the next cycle stages and activates
                                                                         (or holds) it -- its ``run`` / ``run_patch`` files are read whole
    held marker, ``kind = blocked``                          seen only   ``Ingestor.reevaluate_blocked``: re-attempted once the missing
                                                             when the    parent is present (:func:`harvest.parent_absent`) -- and the
                                                             parent is   cycle folds the spool BEFORE it re-evaluates the queue
                                                             present or  (``ingest_once``: dimension bundles, plain files, revision
                                                             spooled     directories, then the queue), so a parent already spooled counts
                                                                         (:func:`_parent_pending`: a dimension bundle file, a run's own
                                                                         files, a revision directory); the revision's staged run-scale
                                                                         rows and its blocked constituents are the candidates
    held marker, ``kind = refused``, exact                   excluded    :func:`harvest.refused_hold_exact`: same bytes, same validator
                                                                         stamp, not superseded -- never re-attempted (only
                                                                         ``ingest --retry-refused``); a changed file or a moved stamp makes
                                                                         it re-attemptable and then SEEN
    held marker, ``kind = run_drop``                         seen only   ``_run_set_drop_hold`` re-compared with the harvester's own
                                                             when the    functions (:func:`_run_drop_hold_stands`): it trips again unless
                                                             guard would the head moved or a declaration arrived on the marker -- a
                                                             now pass    declaration normally arrives as a release bundle, the class below
    held marker of a SUPERSEDED revision, ``kind = refused``  excluded    :func:`harvest.refused_hold_exact`'s superseded branch: never
                                                                         re-attempted (a superseded ``blocked`` hold IS re-attempted once
                                                                         its parent lands, and is seen then)
    a record stamped before the standing statement of every  excluded    the fold is per-column latest statement: it folds into the
    column it would move                                                 history and moves the row nothing (:func:`binding_move_of`)
    ``accepted = 0`` revision with no held marker row        seen        a crash between staging and the marker's delivery: the next cycle
    (partially staged)                                                   re-stages and activates -- its staged rows are the candidates
    unfolded complete activation bundle of a revision that   seen        ``records/adapters/<adapter>/_activations/<rev>/<stamp>/`` with a
    carried this run, carrying ``run`` / ``run_patch`` files             marker and no ``record_delivery`` row for it: the next cycle
                                                                         applies it (``reactivate_committed`` / ``reread_committed``) --
                                                                         found through the revisions of the run's ``row_history`` uris,
                                                                         its staged rows and its blocked rows, newest bundle first,
                                                                         stopping at the first processed one (never a walk over the
                                                                         312,813 historical bundles)
    unfolded MARKER-ONLY bundle of such a revision           seen        ``_apply_revision``: a bundle with nothing to stage folds the
    (``registry revision accept``'s release, a head re-read)             revision's ALREADY STAGED constituents (``_staged_records``,
                                                                         whatever ``accepted`` says) -- those rows are the candidates
    the run's directories in the inbox                       seen        ``scan_inbox`` merges them and the cycle folds them
    plain blocked ``run`` / ``run_patch`` record (no         seen only   ``reevaluate_blocked``, as for a held marker: the parent present
    revision), stale or not                                  when the    or spooled. The ``stale`` bit (``BLOCKED_STALE_S``, seven days)
                                                             parent is   is an age mark for the status page; the queue selects EVERY row,
                                                             present or  so an aged hold is as pending as a fresh one
                                                             spooled
    a superseded hold's constituents, ``kind = refused``     excluded    ``_refused_hold_exact``'s superseded branch
    a parent spooled where this probe does not look (a       KNOWN MISS  the next cycle folds the parent and re-attempts the hold; the
    plain dimension file outside ``records/dimensions/<rev>/``,          identity belt turns the miss into one re-graded run
    an ``object`` parent under its producer's directory)
    a torn directory (no marker); a record the validator     excluded    nothing applies without a marker; a malformed record is refused,
    refuses                                                              not folded (:func:`validate.validate_record`)
    the F12 stat sidecar; ``_activations`` of revisions      not seen    ``state/file_stat.json`` is 123 MB / 479,556 entries live (a 1-2 s
    that never carried the run; files outside run dirs,                  parse per claim); no production writer puts a run's record
    revision dirs, bundles and the inbox                                 anywhere else -- a correction arriving through these is folded
                                                                         by the cycle, the claim reads ``identity_stale`` and the
                                                                         worklist re-claims
    =======================================================  ==========  ==================================================================

    Cost, measured read-only on the live registry (2026-10-04; 4,444 staged run rows of 759 held revisions, 433 held markers of which 156
    stale, 5,432 revision directories under 54 adapters, 21 dimension bundles, 3 inbox hosts; a run carried by 1-2 revisions): median
    170 ms, max 175 ms per claim under the write lock over 8 random runs (warm; up to 3 s once on a cold page cache), plus the read of each
    candidate file. The run's staged and blocked rows are read once and only the held markers of the revisions that mention the run are
    visited; parent presence and the spooled dimension parents are memoized per probe."""
    folded = run_identity_binding(conn, run_id)
    row = conn.execute("SELECT judge_id FROM run WHERE run_id = ?", (run_id,)).fetchone()
    run_judge = str(row[0]) if row is not None and row[0] is not None else None
    stamps = _standing_stamps(conn, run_id)
    out: List[str] = []
    seen: set = set()
    carriers: Dict[str, str] = {}          # revision -> adapter: every revision that carried, stages or blocks a record of this run
    adapters_dir = os.path.join(H.records_dir(root), "adapters")

    def consider(where: str, rec: Mapping[str, object], validated: bool = False) -> None:
        why = binding_move_of(conn, run_id, rec, folded, run_judge, stamps)
        if not why:
            return
        key = (where, rec.get("record_id"))
        if key in seen:
            return
        seen.add(key)
        if validated or _foldable(rec):
            out.append(f"{where}: {why}")

    def note_carrier(uri: str) -> None:
        ar = _adapter_and_revision(uri)
        if ar:
            carriers.setdefault(ar[1], ar[0])

    parents = _ParentProbe(root, conn)

    # --- the run's staged and blocked `run` / `run_patch` rows, read ONCE: by revision ------------------------------------------------
    # every staged row naming this run (one narrowed query over the staging table: by file name, then `instr` on the record text),
    # whatever its marker's accepted flag; every blocked row naming it, by the revision it is tagged with ('' = a plain spool record),
    # stale or not -- the queue re-evaluates every row (`stale` is an age mark for the status page, never a retry predicate)
    staged_by_rev: Dict[str, List[Tuple[str, int, Dict[str, object]]]] = {}
    for rev, uri, line, record_json in conn.execute(
            "SELECT revision_id, file_uri, line, record_json FROM revision_staging_record WHERE (file_uri LIKE '%-run.jsonl' OR "
            "file_uri LIKE '%-run.jsonl.gz' OR file_uri LIKE '%-run_patch.jsonl' OR file_uri LIKE '%-run_patch.jsonl.gz') AND instr(record_json, ?) > 0",
            (run_id,)).fetchall():
        try:
            rec = json.loads(record_json)
        except ValueError:
            continue
        if isinstance(rec, Mapping) and isinstance(rec.get("payload"), Mapping) and rec["payload"].get("run_id") == run_id:
            staged_by_rev.setdefault(str(rev), []).append((str(uri), int(line), rec))
            note_carrier(str(uri))
    blocked_by_rev: Dict[str, List[Tuple[str, str, str, Dict[str, object]]]] = {}
    for uri, ptable, pid, extra_json in conn.execute("SELECT record_uri, missing_parent_table, missing_parent_id, extra_json FROM blocked_records "
                                                     "WHERE record_type IN ('run', 'run_patch')").fetchall():
        try:
            tagged = str((json.loads(extra_json or "{}") or {}).get("revision_id") or "")
        except ValueError:
            tagged = ""
        base, _, line = str(uri).rpartition("#")
        path = _physical(root, base)
        if path is None or not line.isdigit():
            continue
        for n, rec in _records_of(path):
            if n == int(line):
                if isinstance(rec.get("payload"), Mapping) and rec["payload"].get("run_id") == run_id:
                    blocked_by_rev.setdefault(tagged, []).append((str(uri), str(ptable), str(pid), rec))
                    if tagged:
                        note_carrier(base)
                break
    mentioned = (set(staged_by_rev) | set(blocked_by_rev)) - {""}

    # --- revisions that carried this run (the bundle class reads them): its row_history uris and its live carrier --------------------
    key_json = R.canonical_json({"run_id": run_id})
    for (uri,) in conn.execute("SELECT DISTINCT record_uri FROM row_history WHERE table_name = 'run' AND natural_key_json = ?", (key_json,)).fetchall():
        note_carrier(str(uri))
    live = conn.execute("SELECT record_uri FROM run WHERE run_id = ?", (run_id,)).fetchone()
    if live is not None:
        note_carrier(str(live[0]))

    # --- class: marker-complete revision directories no cycle has seen ------------------------------------------------------------
    known_revs = {str(r[0]) for r in conn.execute("SELECT revision_id FROM source_revision").fetchall()}
    for adapter in _adapter_dirs(root):
        adir = os.path.join(adapters_dir, adapter)
        try:
            entries = sorted(os.listdir(adir))
        except OSError:
            continue
        for entry in entries:
            if not _REVISION_NAME_RE.match(entry) or entry in known_revs:
                continue
            rdir = os.path.join(adir, entry)
            if _marker_complete(rdir):
                _consider_dir(root, rdir, run_id, lambda where, rec: consider(where, rec), f"unprocessed revision {entry}")

    # --- class: held markers of the revisions that mention this run, by the queue's own re-attempt rule ---------------------------
    held_revs: set = set()
    for uri, ptable, pid, extra_json in conn.execute("SELECT record_uri, missing_parent_table, missing_parent_id, extra_json FROM blocked_records "
                                                     "WHERE record_type = 'source_revision'").fetchall():
        try:
            ext = json.loads(extra_json or "{}") or {}
        except ValueError:
            continue
        rev = str(ext.get("revision_id") or "")
        if not rev:
            continue
        held_revs.add(rev)
        if rev not in mentioned:
            continue                                                   # no staged or blocked record of this run under it: nothing to move
        base = str(uri).rpartition("#")[0] if "#" in str(uri) else str(uri)      # a marker's row carries the bare file uri
        marker_path = _physical(root, base)
        if marker_path is None or not _marker_complete(os.path.dirname(marker_path)):
            continue                                                   # the directory is gone or torn: nothing to fold
        kind = str(ext.get("kind") or "blocked")
        if kind == "refused" and H.refused_hold_exact(conn, ext, root=root):
            continue                                                   # exact (or superseded): the same bytes under the same code; not re-attempted
        if kind == H.RUN_SET_DROP_KIND and _run_drop_hold_stands(conn, root, rev, marker_path):
            continue                                                   # the guard trips again next cycle: nothing folds until a declaration or a moved head
        if kind == "blocked" and parents.pending(str(ptable), str(pid)) is None:
            continue                                                   # the missing parent is neither present nor on the spool: nothing folds next cycle
        for suri, line, rec in staged_by_rev.get(rev, []):
            consider(f"held revision {rev} ({suri}#{line})", rec, validated=True)
        for buri, _pt, _pid, rec in blocked_by_rev.get(rev, []):
            consider(f"held revision {rev} (blocked {buri})", rec, validated=True)

    # --- class: a pending revision with NO held marker row (a crash between staging and the marker's delivery) -------------------
    for rev in sorted(mentioned - held_revs):
        srow = conn.execute("SELECT accepted, superseded_by_revision_id, refusal FROM source_revision WHERE revision_id = ?", (rev,)).fetchone()
        if srow is None or int(srow[0]) != 0 or srow[1] is not None or srow[2] is not None:
            continue
        for suri, line, rec in staged_by_rev.get(rev, []):
            rdir = os.path.dirname(os.path.join(root, suri))
            if os.path.isdir(rdir) and _marker_complete(rdir) and not _marker_folded(conn, root, rdir):
                consider(f"partially staged revision {rev} ({suri}#{line})", rec, validated=True)

    # --- class: unfolded complete activation bundles of the revisions that carried this run ---------------------------------------
    for rev, adapter in sorted(carriers.items()):
        base = os.path.join(adapters_dir, adapter, "_activations", rev)
        if not os.path.isdir(base):
            continue
        try:
            stamps = sorted(os.listdir(base), reverse=True)
        except OSError:
            continue
        for stamp in stamps:
            bdir = os.path.join(base, stamp)
            if not os.path.isdir(bdir) or not _marker_complete(bdir):
                continue
            if _marker_folded(conn, root, bdir):
                break                                                  # processed; every older bundle of this revision was processed before it
            where = f"unfolded activation bundle {adapter}/_activations/{rev}/{stamp}"
            try:
                names = sorted(os.listdir(bdir))
            except OSError:
                continue
            if any((m := R.RECORD_FILE_RE.match(fn)) and m.group("type") in _BINDING_KINDS for fn in names):
                _consider_dir(root, bdir, run_id, lambda w, rec: consider(w, rec), where)
            else:
                # a MARKER-ONLY bundle (an operator's release, a head re-read): its activation folds the revision's ALREADY STAGED
                # constituents, whatever the marker's accepted flag (the hold of an accepted revision's re-activation keeps them)
                for suri, line, rec in staged_by_rev.get(rev, []):
                    consider(f"release bundle {adapter}/_activations/{rev}/{stamp} folds staged {suri}#{line}", rec, validated=True)

    # --- class: the run's directories in the inbox ----------------------------------------------------------------------------------
    run_dir = identity.run_dir_name(run_id)
    ib = H.inbox_dir(root)
    try:
        hosts = sorted(os.listdir(ib))
    except OSError:
        hosts = []
    for host in hosts:
        hdir = os.path.join(ib, host)
        try:
            campaigns = sorted(os.listdir(hdir))
        except OSError:
            continue
        for campaign in campaigns:
            if campaign.startswith("_"):
                continue
            d = os.path.join(hdir, campaign, run_dir)
            if os.path.isdir(d):
                _consider_dir(root, d, run_id, lambda where, rec: consider(where, rec), f"inbox {host}/{campaign}/{run_dir}")

    # --- class: plain blocked records (no revision), re-attempted once their parent is present or spooled -------------------------
    for buri, ptable, pid, rec in blocked_by_rev.get("", []):
        if parents.pending(ptable, pid) is None:
            continue
        consider(f"blocked record {buri}", rec, validated=True)
    return out


def _consider_dir(root: str, d: str, run_id: str, consider, where: str) -> None:
    """Hand every ``run`` / ``run_patch`` record of the directory's files to ``consider`` (whole files: an adapter batch names many runs)."""
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return
    for fn in names:
        m = R.RECORD_FILE_RE.match(fn)
        if not m or m.group("type") not in _BINDING_KINDS:
            continue
        path = os.path.join(d, fn)
        for n, rec in _records_of(path):
            consider(f"{where} ({H.logical_uri(root, path)}#{n})", rec)


def _judgment_status(conn: sqlite3.Connection, judgment_id: str) -> Optional[str]:
    row = conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (judgment_id,)).fetchone()
    return None if row is None else str(row[0])


def _require_parents(conn: sqlite3.Connection, run_id: str, judge_id: str, what: str) -> None:
    """The rows a judgment record depends on must exist BEFORE it is written (the registry owner, 2026-10-03: a claim that cannot
    fold is never written): the run (its own record may still be in the spool) and the judge. A missing one is
    :class:`ClaimNotFolded`, naming it; nothing has been written."""
    if conn.execute("SELECT 1 FROM run WHERE run_id = ?", (run_id,)).fetchone() is None:
        raise ClaimNotFolded(run_id, f"{what}: no run row yet (the run's record is still in the spool); nothing written")
    if conn.execute("SELECT 1 FROM judge WHERE judge_id = ?", (judge_id,)).fetchone() is None:
        raise ClaimNotFolded(run_id, f"{what}: judge {judge_id} has no row in this registry; nothing written")


def _require_folded(root: str, conn: sqlite3.Connection, run_id: str, judgment_id: str, want: str, rep: H.IngestReport, what: str) -> None:
    """The judgment row must now read ``want`` and the scoped fold must have blocked nothing; otherwise the record is waiting on the
    run's own row (still in the spool) and the caller is refused by name (:class:`ClaimNotFolded`). ``what`` names the operation."""
    got = _judgment_status(conn, judgment_id)
    if got == want and not rep.blocked and not rep.refused:
        return
    raise ClaimNotFolded(run_id, f"{what}: judgment {judgment_id} reads {got!r}, wanted {want!r}; blocked {rep.blocked}, refused {rep.refused}")


def not_pending_why(conn: sqlite3.Connection, run_id: str, judge_id: str) -> Optional[str]:
    """None while (run, judge) is still claimable, else why it is not -- the text :class:`ClaimNotPending` carries. Claimable means
    ``v_pending_judgment`` lists the pair (ONE bounded read on its keys: the same selection the judging daemon chose the run from), or the
    pair is pending in this module's sense but unpaired there -- the known limitation: a judge-less instrument whose fallback judge has
    filed no judgment yet pairs with no judge row, and the producer's direct claim on it stands. Otherwise the run's lifecycle state admits
    no judging (:data:`JUDGEABLE_LIFECYCLE_STATES`), or a complete identity-current judgment by this judge covers the run
    (``schema_def.judgment_covers_sql`` with ``judgment_identity_current_sql`` -- the coverage anti-join of the selection itself, never a
    re-encoding: a stale-bound complete covers nothing and leaves the pair listed as ``identity_stale``, so it is claimed as before); a
    third reason would be a gap in this reasoning and is named as such rather than claimed through."""
    if conn.execute("SELECT 1 FROM v_pending_judgment WHERE run_id = ? AND required_judge_id = ?", (run_id, judge_id)).fetchone():
        return None
    row = conn.execute("SELECT r.lifecycle_state, i.judge_id, r.superseded_by_run_id, " + S.run_superseded_sql("r") + " FROM run r "
                       "LEFT JOIN instrument i ON i.instrument_id = r.instrument_id WHERE r.run_id = ?", (run_id,)).fetchone()
    # a superseded run is never pending; its successor is (the registry ruling of 2026-10-04, migration 0025) -- the selection's own
    # predicate (schema_def.run_superseded_sql), read before the lifecycle: a retired run may well read `judged`
    if row is not None and row[3]:
        return f"superseded by run {row[2]}; admits no judging"
    state = None if row is None else row[0]
    if state not in JUDGEABLE_LIFECYCLE_STATES:
        return f"lifecycle_state={state} admits no judging"
    cover = conn.execute("SELECT j.judgment_id, j.claimed_by_session, j.finished_utc FROM judgment j JOIN run r ON r.run_id = j.run_id "
                         "LEFT JOIN instrument i ON i.instrument_id = r.instrument_id "
                         f"WHERE j.run_id = ? AND j.judge_id = ? AND {S.judgment_covers_sql('j', 'r')} AND {S.judgment_identity_current_sql('j', 'r', 'i')} "
                         "ORDER BY j.finished_utc DESC LIMIT 1", (run_id, judge_id)).fetchone()
    if cover is not None:
        return f"complete judgment {cover[0]} by this judge (session {cover[1]!r}, finished {cover[2] or 'unstated'}) already covers it"
    filed = conn.execute("SELECT 1 FROM judgment WHERE run_id = ? AND judge_id = ?", (run_id, judge_id)).fetchone()
    if row[1] is None and filed is None:
        return None
    return "v_pending_judgment does not list the pair"


def claim(root: str, run_id: str, judge_id: str, session: str, force: bool = False,
          stale_hours: float = DEFAULT_STALE_HOURS, now: Optional[str] = None,
          precheck: Optional[Callable[[sqlite3.Connection], None]] = None) -> str:
    """Register an in-progress judgment for (run, judge) by ``session``; returns the judgment_id.
    Refuses (ClaimHeld) when another session's in-progress claim exists, unless it is older than
    ``stale_hours`` and ``force`` is set (the old claim is then written ``abandoned``).

    A claim that cannot fold is never written: under the write lock, BEFORE any record is written, the judgment's parents are checked -- the
    run row (the run's own record may still be in the spool) and the judge row -- and a missing one refuses by name (:class:`ClaimNotFolded`)
    with NOTHING written; a refused attempt therefore never appears anywhere (an earlier shape wrote the claim and neutralized it with an
    ``abandoned`` record, which ``v_pending_judgment`` then showed as a real abandonment). Every record this function writes rides ONE file
    that is folded at once, and only it (:func:`harvest.fold_only_files`, never a harvester cycle), and the judgment row is CHECKED as the
    belt (:func:`_require_folded`: it must read ``in_progress`` with nothing blocked; it should never trip now and refuses without writing
    anything more if it does), so a second claimant reading ``v_pending_judgment`` or the row right after this returns sees the claim. The
    lock hold is logged at INFO.

    A claim whose binding differs from the run's CURRENT identity (``claimed_instrument_id`` / ``claimed_required_judge_id`` stated
    and not the run's instrument / required judge -- the test is ``schema_def.judgment_identity_current_sql``, the one
    ``v_pending_judgment`` applies) is not held by anyone: the run was corrected after the claim, the only judgment that claim can
    complete into is one the identity rule excludes everywhere, and the worklist no longer presents it as ``claimed``. Under the lock
    such a claim is closed with an ``abandoned`` record that restates its own ``claimed_utc`` (the claim form of the identity rule, the
    forced-takeover shape) and a NEW claim bound to the current identity is minted -- for the same session and for another session
    alike, no ``force`` needed, never ``ClaimHeld`` -- so the same session's retry gets a fresh id bound to the corrected identity and
    another worker can take the run over. Both records ride one file and fold at once.

    The id returned is bound to the identity the fold LEFT, never to one it is about to replace: the
    run's own record files -- a same-id identity correction or a ``run_patch`` of ``judge_id`` a producer wrote and the harvester has
    not folded yet among them, :func:`run_record_files` -- are folded FIRST, under the lock, before the identity is read, as a scoped fold of those files alone
    (the rest of the spool, the inbox and every other run's files are the harvester's cycle's) -- then a correction of the run's
    identity PENDING anywhere the scoped fold must not reach (a blocked record, a constituent of a held adapter revision, an unprocessed
    revision directory, an unfolded activation bundle of a revision that carried the run, a cluster push in the inbox -- each a record
    the next cycle folds; :func:`pending_identity_corrections`) refuses the claim
    by name before anything is written (:class:`ClaimPendingCorrection`, the :class:`ClaimNotFolded` contract: retry after the next
    harvester cycle), since a claim bound now would read ``identity_stale`` the moment the cycle folds it -- and after the claim's own fold the minted claim's binding is RE-CHECKED against the
    folded run: if the identity moved between the two folds, the claim is abandoned (its own ``claimed_utc`` restated) and re-minted one
    millisecond later under the same lock -- each re-mint one file, one scoped fold, one check -- up to :data:`REBIND_CAP` times, else
    the call fails loudly rather than return a stale claim. The re-check is the invariant that survives a fold that processes only the
    claim's own file.

    And the pair must STILL BE PENDING: after the scoped fold-first and before any in-progress claim is read, ``v_pending_judgment`` --
    the selection the judging daemon chose the run from -- is re-read for (run, judge) under the lock (:func:`not_pending_why`, one
    bounded read). A run another session completed between the daemon's selection and this claim (the completion folded by the
    harvester's cycle, or a moment ago by this call's own scoped fold of the run's files) is no longer listed, and a claim minted now
    would be a SECOND claim completing into a second complete judgment of the same pair, which the currency rules do not exclude (they
    exclude superseded, abandoned and displaced partial judgments); so is a run whose lifecycle state left the judgeable set. Refused
    by name before anything is written (:class:`ClaimNotPending`, the :class:`ClaimNotFolded` contract: the daemon re-selects, never a
    quarantine). A completion still on the SPOOL outside the run's own files leaves the completing session's in-progress claim
    standing, so a second claimant meets :class:`ClaimHeld` as before; a stale-bound complete judgment covers nothing and leaves the
    pair listed as ``identity_stale``, so it is claimed as before; and ``force`` is unaffected unless the pair is genuinely complete,
    since a stale in-progress claim leaves the run listed (``claimed``).

    ``precheck``: ONE callable, run UNDER this write lock on THIS connection, right after the parents are checked and before any record
    is written; it refuses by raising :class:`ClaimPrecheckRefused` (or another :class:`ClaimHeld`) with nothing written -- the
    worklist's designated-judge comparison rides here, so a correction folded between the caller's own read and this lock is seen. It
    must be READ-ONLY on the connection: no writes, no commits, no nested locks, no long work (budget about 100 ms -- it sits inside
    the hold the scoped fold just made short), and it logs nothing of its own; a write it attempts is refused (``PRAGMA query_only``
    for the call, the change counter as the belt) as :class:`ClaimPrecheckRefused`. Compose several checks on the caller's side."""
    now = now or R.utc_now()
    folded = 0
    with DB.write_lock(root, timeout_s=LOCK_TIMEOUT_S):
        t_lock = time.monotonic()
        conn = DB.open_or_create(DB.db_path(root))
        try:
            _require_parents(conn, run_id, judge_id, "claim")
            # (1) the run's own spooled `run` records -- a same-id identity correction among them -- are folded FIRST, scoped to those
            # files alone (never a cycle; the run row itself is the harvester's to fold, _require_parents above), so the identity read
            # next is the folded one
            rep0 = _fold_run_records(root, conn, run_id)
            folded += int(rep0.ok) - int(rep0.arrival_records)
            # (1b) a correction of the run's identity pending OUTSIDE its own files -- where the scoped fold must not reach (a held or
            # unseen adapter revision, an unfolded activation bundle, a blocked record, a cluster push in the inbox) -- refuses the claim
            # before anything is written:
            # a claim bound now would be identity_stale the moment the cycle folds it
            pending_fix = pending_identity_corrections(root, conn, run_id)
            if pending_fix:
                raise ClaimPendingCorrection(run_id, "; ".join(pending_fix[:3]) + (f"; +{len(pending_fix) - 3} more" if len(pending_fix) > 3 else ""))
            # (1c) still pending for THIS judge? The selection the daemon chose the run from, re-read under the lock after the scoped
            # fold: a completion by another session that folded between the selection and this claim -- by the harvester's cycle, or a
            # moment ago by the scoped fold-first of the run's own files -- has taken the pair out of v_pending_judgment, and a claim
            # minted now would complete into a SECOND complete judgment of the same (run, judge), which no currency rule excludes.
            # Refused by name, nothing written: re-select. Checked before any in-progress claim is read, so the same session's retry on
            # a run another session has since completed is refused too, and `force` takes over only a run still listed
            why = not_pending_why(conn, run_id, judge_id)
            if why is not None:
                raise ClaimNotPending(run_id, judge_id, why)
            if precheck is not None:
                # the caller's pre-write check, under the lock, nothing written yet -- and READ-ONLY on this connection:
                # `PRAGMA query_only` refuses any write it attempts (surfaced as ClaimPrecheckRefused), the change counter is the belt
                changes_before = conn.total_changes
                conn.execute("PRAGMA query_only = 1")
                try:
                    precheck(conn)
                except sqlite3.OperationalError as exc:
                    if "readonly" not in str(exc).lower():
                        raise
                    raise ClaimPrecheckRefused(run_id, f"the precheck attempted a write under the claim lock ({exc}); a precheck is read-only") from None
                finally:
                    conn.execute("PRAGMA query_only = 0")
                if conn.total_changes != changes_before:
                    raise ClaimPrecheckRefused(run_id, "the precheck changed the database under the claim lock; a precheck is read-only")
            # the latest in-progress claim by this judge, with whether its binding is the run's current identity (a judgment row
            # exists only once its run row does, so the inner join on run loses nothing)
            row = conn.execute(
                "SELECT j.judgment_id, j.claimed_by_session, j.claimed_utc, "
                "CAST((julianday(?) - julianday(j.claimed_utc)) * 86400 AS INTEGER), "
                f"{S.judgment_identity_current_sql('j', 'r', 'i')} "
                "FROM judgment j JOIN run r ON r.run_id = j.run_id LEFT JOIN instrument i ON i.instrument_id = r.instrument_id "
                "WHERE j.run_id = ? AND j.judge_id = ? AND j.status = 'in_progress' ORDER BY j.claimed_utc DESC LIMIT 1",
                (now, run_id, judge_id)).fetchone()
            payloads = []
            if row is not None:
                jid_old, sess, old_claimed_utc, age, bound_current = row
                if bound_current:
                    if sess == session:
                        return str(jid_old)
                    if not (force and age is not None and age > stale_hours * 3600):
                        raise ClaimHeld(run_id, str(sess), int(age or 0))
                # the takeover -- forced over a stale claim, or of a claim bound to a since-corrected identity -- abandons
                # the old claim under ITS claimed_utc (the time its id was minted over), stamped now
                payloads.append(_claim_payload(run_id, judge_id, str(sess), now, status="abandoned", judgment_id=str(jid_old),
                                               claimed_utc=str(old_claimed_utc) if old_claimed_utc else None))
            instrument_id, required_judge = run_identity_binding(conn, run_id)
            claim_payload = _claim_payload(run_id, judge_id, session, now, claimed_instrument_id=instrument_id,
                                           claimed_required_judge_id=required_judge)
            payloads.append(claim_payload)
            jid = str(claim_payload["judgment_id"])
            path = _write_claims(root, run_id, session, now, payloads)
            rep = _fold_own(root, conn, path)
            folded += int(rep.ok) - int(rep.arrival_records)        # the claim's own records; the arrival fact is bookkeeping
            _require_folded(root, conn, run_id, jid, "in_progress", rep, "claim")
            # (2) re-check: the minted claim's binding against the identity the fold left. A correction that landed between the two
            # folds moved it; abandon the claim (under its own claimed_utc) and re-mint one millisecond later, bound to the folded
            # identity, under the same lock -- one file, one scoped fold, one check per re-mint -- never return a claim the worklist
            # already excludes
            stamp = now
            for _ in range(REBIND_CAP):
                left = run_identity_binding(conn, run_id)
                if left == (instrument_id, required_judge):
                    return jid
                stamp = R.epoch_ms_to_utc(R.utc_to_epoch_ms(stamp) + 1)
                instrument_id, required_judge = left
                moved = claim_payload
                claim_payload = _claim_payload(run_id, judge_id, session, stamp, claimed_instrument_id=instrument_id,
                                               claimed_required_judge_id=required_judge)
                jid = str(claim_payload["judgment_id"])
                path = _write_claims(root, run_id, session, stamp,
                                     [_claim_payload(run_id, judge_id, session, stamp, status="abandoned", judgment_id=str(moved["judgment_id"]),
                                                     claimed_utc=str(moved["claimed_utc"])),
                                      claim_payload])
                rep = _fold_own(root, conn, path)
                folded += int(rep.ok) - int(rep.arrival_records)
                _require_folded(root, conn, run_id, jid, "in_progress", rep, "claim")
            raise RuntimeError(f"run {run_id}: its identity moved under the claim lock through {REBIND_CAP} re-mints; no claim returned "
                               f"(the last claim {claim_payload['judgment_id']} stands in_progress; retry, or read the run's record log)")
        finally:
            conn.close()
            log.info("pending: claim %s held the write lock %.2f s, folded %d record(s)", run_id, time.monotonic() - t_lock, folded)


def release(root: str, run_id: str, judge_id: str, session: str, now: Optional[str] = None) -> bool:
    """Write ``abandoned`` for one's own in-progress claim; returns False when none exists. The record carries the claim's own
    ``claimed_utc`` (selected here, never re-stamped), so its id reproduces under the claim form of the identity rule. It is folded
    under the lock the way :func:`claim` folds its own (that file alone), checked to read ``abandoned``
    (:class:`ClaimNotFolded` otherwise), and the hold is logged at INFO."""
    now = now or R.utc_now()
    folded = 0
    with DB.write_lock(root, timeout_s=LOCK_TIMEOUT_S):
        t_lock = time.monotonic()
        conn = DB.open_or_create(DB.db_path(root))
        try:
            row = conn.execute("SELECT judgment_id, claimed_utc FROM judgment WHERE run_id = ? AND judge_id = ? AND status = 'in_progress' "
                               "AND claimed_by_session = ?", (run_id, judge_id, session)).fetchone()
            if row is None:
                return False
            # the abandonment re-states the claim's own claimed_utc (the identity input), stamped now
            path = _write_claims(root, run_id, session, now,
                                 [_claim_payload(run_id, judge_id, session, now, status="abandoned", judgment_id=str(row[0]),
                                                 claimed_utc=str(row[1]) if row[1] else None)])
            rep = _fold_own(root, conn, path)
            folded = int(rep.ok) - int(rep.arrival_records)
            _require_folded(root, conn, run_id, str(row[0]), "abandoned", rep, "release")
            return True
        finally:
            conn.close()
            log.info("pending: release %s held the write lock %.2f s, folded %d record(s)", run_id, time.monotonic() - t_lock, folded)


def ingest_now(root: str) -> H.IngestReport:
    """Fold the producer records now, under the harvester lock -- ONE FULL harvester cycle (every new or changed file, the blocked
    queue, the ranks, the cycle stamp), not a fold of the caller's files alone. Since 2026-09-30 a judging driver's
    :meth:`worklist.RegistryWorklist.finish` calls this only when asked (``finish(fold=True)`` / ``SISTEER_REGISTRY_DRIVER_FOLD=force``):
    by default the completion records stay spooled and the harvester's own cycle folds them within one cadence (a cycle here after
    every driver batch held the lock the timer and the claims needed and rewrote ``state/harvest_cycle.json``, so a dead timer read as
    alive). Only the fold (no harvester facts): the ``ingested`` transition stays the harvester timer's. Raises :class:`db.LockHeld`
    when the lock is still held after the wait. This is the ONE place a judging driver may run a cycle; :func:`claim` and
    :func:`release` fold their own record file alone (``harvest.fold_only_files``) and never rewrite the cycle stamp -- a claim
    is not a harvester cycle, and a stamp it refreshed made a dead timer read as alive (the 2026-09-30 note above) while the full
    cycle it paid, under a spool backlog the refused harvester could not drain, held the lock for minutes."""
    with DB.write_lock(root, timeout_s=LOCK_TIMEOUT_S):
        conn = DB.open_or_create(DB.db_path(root))
        try:
            return H.ingest_once(root, conn, workers=0, derive_facts=False, min_free=0)
        finally:
            conn.close()


# --------------------------------------------------------------------------------------------
# worklist parity (acceptance "Grading queue")
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """One entry of a judging driver's own unit list: a store, its object discriminator and its eval —
    the registry's run grain once the object discriminator is applied."""
    unit: str
    store_path: str
    object_key: str
    eval_id: str
    campaign: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def store_key(self) -> str:
        return os.path.basename(self.store_path.rstrip("/"))


#: How a Sonnet ``units_*.json`` entry (``units[<unit>]``) maps onto :class:`Unit`.
UNITS_FIELDS = {"store_path": "grades_dir", "object_key": "cell_id", "eval_id": "family", "campaign": "source"}


def load_units(paths: Iterable[str]) -> Dict[str, Unit]:
    """The units of one or more ``units_*.json`` files (shape: ``{"units": {<unit>: {grades_dir, cell_id,
    family, source, ...}}}``; a bare ``{<unit>: {...}}`` mapping is accepted too). A unit missing one of the
    grain fields is refused naming the file and the field."""
    out: Dict[str, Unit] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        units = doc.get("units") if isinstance(doc, dict) and isinstance(doc.get("units"), dict) else doc
        if not isinstance(units, dict):
            raise ValueError(f"{path}: not a units file (expected {{'units': {{...}}}} or a mapping of units)")
        for uid, u in units.items():
            if not isinstance(u, dict):
                raise ValueError(f"{path}: unit {uid!r} is not an object")
            missing = [k for k, src in UNITS_FIELDS.items() if k != "campaign" and not u.get(src)]
            if missing:
                raise ValueError(f"{path}: unit {uid!r} lacks {[UNITS_FIELDS[m] for m in missing]}")
            out[str(uid)] = Unit(str(uid), str(u["grades_dir"]), str(u["cell_id"]), str(u["family"]),
                                 str(u["source"]) if u.get("source") else None,
                                 {k: u.get(k) for k in ("kind", "cache_scheme", "trial", "view") if k in u})
    return out


#: A store-root map: ``{substrate: [(remote_root, local_mirror_root), ...]}`` -- the
#: producer recorded the COMPUTE NODE's absolute store path (``run.store_uri``); the grading host holds a mirror of each cluster's
#: stores under its own root. The map is applied at READ time to the selected rows only; the recorded URI stays the provenance
#: and the live database is never rewritten.
StoreRootMap = Dict[str, List[Tuple[str, str]]]


def parse_store_root_map(specs: Optional[Sequence[str]]) -> StoreRootMap:
    """``<substrate>=<remote root>=<local mirror root>`` strings (the drivers' repeatable ``--store-root-map``) → :data:`StoreRootMap`.
    A spec without its three parts, or naming a substrate outside the vocabulary, is refused by name."""
    from .vocab import SUBSTRATES
    out: StoreRootMap = {}
    for spec in specs or ():
        parts = str(spec).split("=")
        if len(parts) != 3 or not all(x.strip() for x in parts):
            raise ValueError(f"--store-root-map {spec!r}: expected <substrate>=<remote root>=<local mirror root>")
        sub, remote, local = (x.strip() for x in parts)
        if sub not in SUBSTRATES:
            raise ValueError(f"--store-root-map {spec!r}: substrate {sub!r} is not one of {'|'.join(SUBSTRATES)}")
        out.setdefault(sub, []).append((remote.rstrip("/"), local.rstrip("/")))
    return out


def map_store_path(path: str, substrate: Optional[str], store_root_map: Optional[StoreRootMap]) -> str:
    """``path`` relocated from the producer's root to the grading host's mirror for the row's ``substrate`` (the longest matching
    remote root wins); unchanged when no map, no substrate or no root matches."""
    if not path or not store_root_map:
        return path
    # a pending row (v_pending_judgment) carries no substrate: without one every mapped root is a candidate, longest match wins
    roots = list(store_root_map.get(str(substrate), ())) if substrate else [rl for pairs in store_root_map.values() for rl in pairs]
    best: Optional[Tuple[str, str]] = None
    for remote, local in roots:
        if path == remote or path.startswith(remote + "/"):
            if best is None or len(remote) > len(best[0]):
                best = (remote, local)
    if best is None:
        return path
    remote, local = best
    return local + path[len(remote):]


def store_path_of(row: Mapping[str, object], store_root_map: Optional[StoreRootMap] = None) -> str:
    """The run's store as a filesystem path (``file://`` stripped); '' when the row carries none. With a ``store_root_map`` the
    compute node's path is relocated onto the grading host's mirror for the row's ``substrate`` (:func:`map_store_path`)."""
    s = str(row.get("store_uri") or "")
    s = s[len("file://"):] if s.startswith("file://") else s
    return map_store_path(s, row.get("substrate"), store_root_map)


def _object_matches(run_object_key: str, unit_object_key: str) -> bool:
    return run_object_key == unit_object_key or run_object_key.split("@", 1)[0] == unit_object_key


def _store_matches(row: Mapping[str, object], unit: Unit) -> bool:
    if str(row.get("producer_run_key_legacy") or "") == unit.store_key:
        return True
    store = store_path_of(row).rstrip("/")
    if not store:
        return False
    return store == unit.store_path.rstrip("/") or store.endswith("/" + unit.store_key)


@dataclass
class ParityReport:
    judge_id: str
    matched: Dict[str, List[str]] = field(default_factory=dict)          # unit -> [run_id]
    unit_without_run: Dict[str, str] = field(default_factory=dict)       # unit -> reason
    unit_not_pending: Dict[str, Dict[str, str]] = field(default_factory=dict)   # unit -> {run_id: reason}
    pending_without_unit: Dict[str, str] = field(default_factory=dict)   # run_id -> reason
    identity_mismatch: Dict[str, str] = field(default_factory=dict)      # run_id -> reason

    @property
    def n_differences(self) -> int:
        return (len(self.unit_without_run) + sum(len(v) for v in self.unit_not_pending.values())
                + len(self.pending_without_unit) + len(self.identity_mismatch))

    def differences(self) -> List[Tuple[str, str, str]]:
        """``(kind, id, reason)`` rows, by run_id where a run exists and by unit id otherwise."""
        out: List[Tuple[str, str, str]] = []
        for u, reason in sorted(self.unit_without_run.items()):
            out.append(("unit_without_run", u, reason))
        for u, runs in sorted(self.unit_not_pending.items()):
            for rid, reason in sorted(runs.items()):
                out.append(("unit_not_pending", rid, f"unit {u}: {reason}"))
        for rid, reason in sorted(self.pending_without_unit.items()):
            out.append(("pending_without_unit", rid, reason))
        for rid, reason in sorted(self.identity_mismatch.items()):
            out.append(("identity_mismatch", rid, reason))
        return out

    def as_dict(self) -> Dict[str, object]:
        return {"judge_id": self.judge_id, "n_matched_units": len(self.matched), "n_differences": self.n_differences,
                "matched": self.matched, "unit_without_run": self.unit_without_run, "unit_not_pending": self.unit_not_pending,
                "pending_without_unit": self.pending_without_unit, "identity_mismatch": self.identity_mismatch}


def _not_pending_reason(conn: sqlite3.Connection, run_id: str, judge_id: str) -> str:
    row = conn.execute("SELECT lifecycle_state, superseded_by_run_id, " + S.run_superseded_sql("run") + " FROM run WHERE run_id = ?", (run_id,)).fetchone()
    state = row[0] if row else None
    if row and row[2]:
        return f"superseded by run {row[1]}; never pending (its successor is)"
    if state not in JUDGEABLE_LIFECYCLE_STATES:
        return f"lifecycle_state={state} never pending"
    # the covering judgment the worklist reads is an identity-current one: a complete judgment bound at claim to a
    # since-corrected run identity covers nothing there, so this diagnostic must not claim it does
    j = conn.execute("SELECT j.status, j.n_items_judged FROM judgment j JOIN run r ON r.run_id = j.run_id "
                     "LEFT JOIN instrument i ON i.instrument_id = r.instrument_id "
                     f"WHERE j.run_id = ? AND j.judge_id = ? AND j.status = 'complete' AND {S.judgment_identity_current_sql('j', 'r', 'i')} "
                     "ORDER BY j.n_items_judged DESC LIMIT 1", (run_id, judge_id)).fetchone()
    if j:
        return f"complete judgment by the requested judge covers it (n_items_judged={j[1]})"
    return "not listed by v_pending_judgment"


def worklist_parity(conn: sqlite3.Connection, judge_id: str, units: Mapping[str, Unit],
                    campaign: Optional[str] = None) -> ParityReport:
    """Compare ``pending_for(judge)`` to a driver's units set-wise. A unit is matched to the run rows with its
    eval and store (``producer_run_key_legacy`` = the store's basename, or ``store_uri`` naming the store) and
    its object discriminator (``producer_object_key_legacy``, ``<object>@<hash>`` compared on the object);
    every matched run's id is re-minted with :func:`identity.run_id` and a disagreement is a difference of its
    own. Differences: a unit with no run row (backfill gap), a unit whose runs are not pending (with the
    reason), a pending run no unit names."""
    jid = resolve_judge_id(conn, judge_id)
    rep = ParityReport(jid)
    pending_rows = pending_for(conn, jid, campaign)
    pending_by_id = {str(r["run_id"]): r for r in pending_rows}
    unmatched_pending = set(pending_by_id)
    sql = ("SELECT run_id, campaign, producer_run_key_legacy, producer_object_key_legacy, eval_id, instrument_id, store_uri "
           "FROM run WHERE eval_id = ?" + (" AND campaign = ?" if campaign else ""))
    for uid, unit in sorted(units.items()):
        params: List[object] = [unit.eval_id] + ([campaign] if campaign else [])
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        if unit.campaign and not campaign:
            rows = [r for r in rows if r["campaign"] == unit.campaign] or rows
        hits = [r for r in rows if _store_matches(r, unit) and _object_matches(str(r["producer_object_key_legacy"]), unit.object_key)]
        if not hits:
            rep.unit_without_run[uid] = (f"no run row for store {unit.store_key!r} object {unit.object_key!r} eval {unit.eval_id!r}"
                                         + (f" campaign {unit.campaign!r}" if unit.campaign else ""))
            continue
        for r in hits:
            rid = str(r["run_id"])
            minted = identity.run_id(str(r["campaign"]), str(r["producer_run_key_legacy"]), str(r["producer_object_key_legacy"]),
                                     str(r["eval_id"]), str(r["instrument_id"]))
            if minted != rid:
                rep.identity_mismatch[rid] = f"identity.run_id over the row's legacy keys mints {minted}"
            if rid in pending_by_id:
                rep.matched.setdefault(uid, []).append(rid)
                unmatched_pending.discard(rid)
            else:
                rep.unit_not_pending.setdefault(uid, {})[rid] = _not_pending_reason(conn, rid, jid)
    for rid in sorted(unmatched_pending):
        r = pending_by_id[rid]
        rep.pending_without_unit[rid] = (f"pending ({r['pending_reason']}) but no unit names store "
                                         f"{r['producer_run_key_legacy']!r} object {r['producer_object_key_legacy']!r} eval {r['eval_id']!r}")
    return rep
