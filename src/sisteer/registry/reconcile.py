"""Record-only reconciliations: a correction filed as a NEW record from what the log already holds, never a database edit.

The one reconciliation here restores the central stamps the fold dropped:
``RecordWriter`` used to park its central stamps (``run_overrides``, ``cluster_label``) under a ``run_patch``'s top-level
``extra``, which ``_validate_run_patch`` never reads -- so every one of those stamps was dropped at the fold (1,634 records on 668
runs by the whole-log audit). History folds exactly as it did and is not edited; the stamps now route through ``fields.extra_json``
for every writer on the new code; THIS module re-files, per run, the keys that are still MISSING from the live ``run.extra_json``, as
one ordinary ``run_patch`` the harvester folds like any other.

The three buckets, per lost key ``k`` of a run (the live row read with ``SELECT run_id, extra_json FROM run WHERE run_id = ?``):

* **missing** -- the live ``extra_json`` lacks ``k``: re-filed (``--apply``);
* **equal** -- the live value is the lost value (a later record stated it the routed way): skipped;
* **conflict** -- the live ``extra_json`` holds ``k`` with a DIFFERENT value: REPORTED for the owner's adjudication, never re-filed.
  The live value came from newer provenance, and a reconciliation record would be the newest statement and would overwrite it
  (``reducer.fold`` merges ``_extra`` in record order, latest wins per key). Measured at the plan (read-only): missing 38 / equal 589 /
  conflict 41 over 668 runs.

Enumeration is from the LOG, the authoritative source: every ``*-run_patch.jsonl`` / ``.jsonl.gz`` under ``<root>/records``
(``RecordWriter`` names a patch file so; the whole-log audit used the same selection), the records of type ``run_patch`` whose payload
carries a non-empty top-level ``extra``, grouped by ``run_id`` and merged in ``(written_utc, record sha)`` order -- the fold's own
merge order -- so the lost value of a key is the one the fold WOULD have kept had the stamps ridden ``fields.extra_json``.

**Only a record the fold ACCEPTED is evidence**. A legacy patch with a valid ``run_id`` and a
well-formed stamp can still have been REFUSED by the harvester -- a ``verify.serving_path`` that disagreed with the row, a column that
failed its check -- and a refused patch is rolled back before its ``row_history`` insert (``harvest.apply_validated``'s savepoint;
``reducer.apply_run_scale`` verifies first), so the run row's existence says nothing about THAT record. Per run, the accepted ``run_patch``
events are read from ``row_history`` by the run's key (a primary-key seek: ``table_name = 'run' AND natural_key_json = ?``); a lost
record whose sha is not among them is **unaccepted**, excluded from the merge, and reported by status -- ``refused: <reason>``
(``record_refusal``), ``blocked: <parent>`` (``blocked_records``) or ``not_ingested`` (in neither: its file has not been folded yet). A
refused record never folds, so the run's accepted evidence is re-filed without it; a blocked or not-ingested record MAY still fold and
change which record is latest, so ``apply`` holds that run (``refused`` names the record; ingest first, then re-run) -- as it holds a run
with a ``pending`` key, whose reconciliation record is in the log and not yet folded, for the same reason.

The reconciliation record: ``{run_id, fields: {extra_json: {<missing keys>, "reconciled_q11": {"from_record_sha256s": [...],
"decision": "Q11"}}}}``, written through ``emit.RecordWriter(spool=<root>/records, campaign=<the run directory's parent>, run_id,
disabled=False)`` into the run's OWN record directory (the directory of its latest lost record), with a new producer id and the
writer's clock -- an ordinary record: validated like any patch, folded by order key, replay-equal by construction (nothing historical
changes; a record is added), and idempotent (after the fold a second pass finds the keys ``equal``). The writer stamps NOTHING of its
own on this record: a central stamp for a key that is not missing (``run_overrides`` from an operator's ``$SISTEER_RUN_OVERRIDES``)
would be a new statement over a live value, exactly what the conflict rule forbids, so ``payload_extra`` is emptied after construction
and a test pins it. The marker is APPEND-ONLY across reconciliations: when the live ``extra_json`` already
carries ``reconciled_q11`` -- an earlier pass re-filed one key, and a legacy patch that arrived later from another machine lost another --
the new marker's ``from_record_sha256s`` is the live marker's list plus the new records not already in it, in that order, so the
reducer's shallow merge of ``extra`` (latest marker wins whole) loses no earlier provenance. Three more buckets keep the pass loud rather
than quietly partial: **pending** -- a reconciliation record for the key is already in the log and not yet folded (a second ``--apply``
before the window's ingest, or a reconciliation record the harvester refused), reported, never written twice; **unaccepted** -- every
lost record of the run is unaccepted (above), so there is no evidence to merge; **no_live_row** -- the run has no row in ``run`` (its own
record refused or still blocked), reported, nothing to compare against. A lost ``extra`` that is not a JSON object carries no keys to
merge and is reported as ``malformed``.

``APPLY_ENABLED`` gates the CLI's ``--apply`` (``registry reconcile run-patch-extra``): this commit ships it ``False`` -- the CLI runs
dry-run only and ``--apply`` refuses with exit 2, the conflicts listed -- per the plan's sequencing (Q12: the dry run always; the
writes at the landing window on the owner's word, after the conflicts have been adjudicated). The library path
(:func:`reconcile_run_patch_extra` with ``apply=True``) is what the window step runs and what the tests fold and rebuild through; lifting
the gate is a one-line change in a reviewed commit, with the test that pins the refusal flipped in the same change. The CLI command
resolves its root without the ``state/last_run`` stamp every other command leaves (``cli._root_unstamped``): a dry run, and a gated
``--apply``, promise to write NOTHING of their own under the root, and a test compares the root's files before and after both -- every
file as it was, the database included, ``state/last_run`` absent. The one allowance is SQLite's own ``-wal`` / ``-shm`` side files of the
WAL-mode database: any read-only (``mode=ro``) connection creates them when no other connection holds them open and cannot remove them
on close (it holds no write lock); they carry no registry data (the WAL stays empty, and the test pins that) and the next writer's close
reclaims both. ``immutable=1`` would avoid them and is not used: it disables locking and WAL reads, so against a live writer it reads a
silently stale snapshot (measured on sqlite 3.45, 2026-10-03).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from . import db as DB
from . import records as R

#: The provenance key the reconciliation record carries beside the re-filed keys.
RECONCILED_KEY = "reconciled_q11"
DECISION = "Q11"
PRODUCER_SCRIPT = "registry reconcile run-patch-extra"
#: The CLI's ``--apply`` gate (module docstring): dry-run only while False.
APPLY_ENABLED = False

_RUN_PATCH_FILE = re.compile(r"-run_patch\.jsonl(\.gz)?$")
#: the files whose records STATE a run's columns: a full ``run`` record's and a ``run_patch``'s (the multi-turn-scope hold reads both)
_RUN_STATEMENT_FILE = re.compile(r"-(?:run|run_patch)\.jsonl(\.gz)?$")
BUCKETS: Tuple[str, ...] = ("missing", "equal", "conflict", "pending", "unaccepted", "no_live_row", "malformed")
#: The statuses of a lost record the fold has not accepted (module docstring): ``refused`` is final; the other two may still fold.
UNACCEPTED_STATUSES: Tuple[str, ...] = ("refused", "blocked", "not_ingested")


@dataclass(frozen=True)
class LostExtra:
    """One ``run_patch`` record whose top-level ``extra`` the fold dropped."""
    run_id: str
    written_utc: str
    sha: str
    path: str
    extra: Dict[str, object]

    @property
    def order_key(self) -> Tuple[str, str]:
        return (self.written_utc, self.sha)


@dataclass
class RunReconciliation:
    """The three buckets (and the two loud ones) of one run's lost keys, plus what ``apply`` wrote."""
    run_id: str
    directory: str
    records: List[LostExtra]
    lost: Dict[str, object] = field(default_factory=dict)
    missing: Dict[str, object] = field(default_factory=dict)
    equal: List[str] = field(default_factory=list)
    conflict: Dict[str, Dict[str, object]] = field(default_factory=dict)    # key -> {"lost": ..., "live": ...}
    pending: Dict[str, str] = field(default_factory=dict)                   # key -> sha of the unfolded reconciliation record
    unaccepted: Dict[str, str] = field(default_factory=dict)                # sha -> "refused: <reason>" | "blocked: <parent>" | "not_ingested"
    live_reconciled: Dict[str, object] = field(default_factory=dict)        # the live row's reconciled_q11 marker, when it carries one
    no_live_row: bool = False
    malformed: List[str] = field(default_factory=list)                      # "<sha>: <reason>" per unmergeable record
    written: Optional[str] = None
    refused: Optional[str] = None                                           # why apply wrote nothing for a run with missing keys

    @property
    def accepted(self) -> List[LostExtra]:
        """The lost records the fold accepted (their sha is a ``row_history`` event of the run), in merge order: the evidence."""
        return [r for r in self.records if r.sha not in self.unaccepted]

    @property
    def unfolded(self) -> Dict[str, str]:
        """The unaccepted records that MAY still fold (blocked, not ingested) -- evidence the fold has not weighed yet, so ``apply``
        holds the run until an ingest has; a ``refused`` record never folds and is simply excluded."""
        return {sha: st for sha, st in self.unaccepted.items() if not st.startswith("refused")}

    def status_of(self, sha: str) -> str:
        """``accepted`` or the unaccepted status word (``refused`` / ``blocked`` / ``not_ingested``) of one lost record."""
        st = self.unaccepted.get(sha)
        return "accepted" if st is None else st.split(":", 1)[0]

    @property
    def bucket(self) -> str:
        """The run's one bucket for the per-run counts: conflict beats missing beats pending beats equal; a run with nothing the
        fold accepted is ``malformed`` or ``unaccepted`` by what it does carry."""
        if self.no_live_row:
            return "no_live_row"
        if not self.lost:
            if self.malformed:
                return "malformed"
            if self.unaccepted:
                return "unaccepted"
        if self.conflict:
            return "conflict"
        if self.missing:
            return "missing"
        if self.pending:
            return "pending"
        return "equal"

    @property
    def from_record_sha256s(self) -> List[str]:
        """The ACCEPTED lost records that carried any key being re-filed, in merge order (the record's provenance); an unaccepted
        record is never named as a source."""
        return [r.sha for r in self.accepted if any(k in r.extra for k in self.missing)]

    def as_dict(self) -> Dict[str, object]:
        return {"run_id": self.run_id, "directory": self.directory, "bucket": self.bucket, "records": len(self.records),
                "accepted": len(self.accepted), "unaccepted": dict(self.unaccepted),
                "lost_keys": sorted(self.lost), "missing": dict(self.missing), "equal": list(self.equal),
                "conflict": {k: dict(v) for k, v in self.conflict.items()}, "pending": dict(self.pending),
                "live_reconciled": dict(self.live_reconciled),
                "no_live_row": self.no_live_row, "malformed": list(self.malformed), "written": self.written, "refused": self.refused}


@dataclass
class Report:
    root: str
    runs: List[RunReconciliation]
    files_read: int
    records_read: int
    applied: bool
    sources: List["OverridesSource"] = field(default_factory=list)      # the overrides files a run_overrides stamp compares by (env-comments' --source)

    def by_bucket(self) -> Dict[str, int]:
        """Runs per bucket (the plan's per-run numbers: missing 38 / equal 589 / conflict 41 over 668)."""
        out = {b: 0 for b in BUCKETS}
        for r in self.runs:
            out[r.bucket] += 1
        return out

    def keys_by_bucket(self) -> Dict[str, int]:
        """Lost (run, key) pairs per bucket -- a run with one missing and one conflicting key counts once in each."""
        return {"missing": sum(len(r.missing) for r in self.runs), "equal": sum(len(r.equal) for r in self.runs),
                "conflict": sum(len(r.conflict) for r in self.runs), "pending": sum(len(r.pending) for r in self.runs)}

    def records_by_status(self) -> Dict[str, int]:
        """Lost records by what the fold did with them: ``accepted`` (evidence) and the three unaccepted statuses."""
        out = {"accepted": 0, **{st: 0 for st in UNACCEPTED_STATUSES}}
        for r in self.runs:
            for rec in r.records:
                out[r.status_of(rec.sha)] += 1
        return out

    @property
    def written(self) -> List[str]:
        return [r.written for r in self.runs if r.written]

    def as_dict(self) -> Dict[str, object]:
        return {"root": self.root, "files_read": self.files_read, "records_read": self.records_read, "applied": self.applied,
                "sources": [{"path": s.path, "sha256": s.sha256, "id": s.id} for s in self.sources],
                "runs": len(self.runs), "runs_by_bucket": self.by_bucket(), "keys_by_bucket": self.keys_by_bucket(),
                "records_by_status": self.records_by_status(), "written": self.written, "per_run": [r.as_dict() for r in self.runs]}


# --------------------------------------------------------------------------------------------
# The walk: lost top-level extras and pending reconciliation records, from the log
# --------------------------------------------------------------------------------------------


def _raise_listing_error(exc: OSError) -> None:
    """``os.walk`` SKIPS a directory it cannot list unless told otherwise; a reconciliation that silently enumerated part of the log
    would file a partial result that looks whole (the first live dry run of this module read 2,722 of 4,076 files once, 2026-10-02),
    so a listing error is raised by name instead."""
    raise exc


def _extra_json_object(value) -> Optional[Dict[str, object]]:
    """A ``fields.extra_json`` as the object it carries -- a dict, or a JSON string that parses to one -- else None. Only the
    reconciliation records' own key is looked for here (a record whose ``fields.extra_json`` is not an object carries none; the
    validator's business, not this pass's), so this reads the two shapes the fold admits and nothing else."""
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return dict(parsed) if isinstance(parsed, dict) else None
    return None


def _record_files(records_root: str, name: "re.Pattern[str]") -> List[str]:
    """Every record file under ``records_root`` whose name matches ``name`` (sorted; the walk is deterministic; a directory that cannot
    be listed raises, never a quietly shorter list). An absent ``records_root`` is refused the same way."""
    if not os.path.isdir(records_root):
        raise FileNotFoundError(f"no record log at {records_root!r}")
    out = []
    for dirpath, dirnames, filenames in os.walk(records_root, onerror=_raise_listing_error):
        dirnames.sort()
        out.extend(os.path.join(dirpath, fn) for fn in filenames if name.search(fn))
    out.sort()
    return out


def run_patch_files(records_root: str) -> List[str]:
    """Every ``*-run_patch.jsonl`` / ``.jsonl.gz`` under ``records_root`` (:func:`_record_files`)."""
    return _record_files(records_root, _RUN_PATCH_FILE)


def run_statement_files(records_root: str) -> List[str]:
    """Every ``*-run.jsonl`` and ``*-run_patch.jsonl`` (``.gz`` too) under ``records_root``: the files whose records can state a run's
    columns -- a full ``run`` record (a registration, a store's opening, an adapter's row, a re-registration) restates every column it
    carries; a ``run_patch`` states the fields it names. Every writer names a record file by its record type (``records.record_file_name``:
    the RecordWriter, the adapters' bundles), so these two names are where the two record types lie."""
    return _record_files(records_root, _RUN_STATEMENT_FILE)


def walk_lost_extras(records_root: str) -> Tuple[Dict[str, List[LostExtra]], Dict[str, Dict[str, str]], Dict[str, List[str]], int, int]:
    """``(lost by run_id, pending reconciliations by run_id {key: sha}, malformed by run_id, files read, records read)``.

    A ``run_patch`` record counts as LOST when its payload carries a non-empty top-level ``extra`` (a dict; anything else is
    ``malformed`` -- it carries no keys the fold could have merged). A record whose ``fields.extra_json`` carries
    :data:`RECONCILED_KEY` is a reconciliation already in the log: every other key under its ``fields.extra_json`` is PENDING for
    that run until the fold makes it ``equal``."""
    lost: Dict[str, List[LostExtra]] = {}
    pending: Dict[str, Dict[str, str]] = {}
    malformed: Dict[str, List[str]] = {}
    files = run_patch_files(records_root)
    n_records = 0
    for path in files:
        for _line, rec, _ in R.read_record_file(path, compute_sha=False):
            n_records += 1
            if rec.get("record_type") != "run_patch":
                continue
            payload = rec.get("payload") or {}
            if not isinstance(payload, Mapping):
                continue
            run_id = payload.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                continue
            fields = payload.get("fields")
            own = _extra_json_object(fields.get("extra_json")) if isinstance(fields, Mapping) else None
            if own and RECONCILED_KEY in own:
                sha = R.record_sha(rec)
                for k in own:
                    if k != RECONCILED_KEY:
                        pending.setdefault(run_id, {})[k] = sha
                continue
            extra = payload.get("extra")
            if extra is None or extra == {} or extra == "" or extra == []:
                continue
            sha = R.record_sha(rec)
            if not isinstance(extra, Mapping):
                malformed.setdefault(run_id, []).append(f"{sha}: top-level extra is a {type(extra).__name__}, not an object")
                continue
            lost.setdefault(run_id, []).append(LostExtra(run_id, str(rec.get("written_utc", "")), sha, path, dict(extra)))
    for recs in lost.values():
        recs.sort(key=lambda r: r.order_key)
    return lost, pending, malformed, len(files), n_records


def merge_lost(records: Sequence[LostExtra]) -> Dict[str, object]:
    """The fold's merge order: ``extra.update(ev_extra)`` per record in ``(written_utc, sha)`` order, latest wins per key."""
    out: Dict[str, object] = {}
    for r in sorted(records, key=lambda r: r.order_key):
        out.update(r.extra)
    return out


def classify_keys(lost: Mapping[str, object], live_extra: Optional[Mapping[str, object]],
                  pending: Optional[Mapping[str, str]] = None,
                  sources: Optional[Mapping[str, "OverridesSource"]] = None) -> Tuple[Dict[str, object], List[str], Dict[str, Dict[str, object]], Dict[str, str]]:
    """``(missing, equal, conflict, pending)`` for one run's merged lost keys against its live ``extra_json`` (the pure rule).

    Values compare by canonical JSON (the fold stores the record's value as canonical JSON, so a lost ``{"a": 1, "b": 2}`` and a
    live ``{"b": 2, "a": 1}`` are equal). A key the live row lacks that a reconciliation record already re-filed is ``pending``. Two
    ``run_overrides`` stamps that differ are still ``equal`` when they are statements of ONE FILE (:func:`same_overrides_stamp`, over
    ``sources``, env-comments' overrides files by digest): the same sha256 and path, the file given, and EACH value of both one of the
    file's two readings -- a legacy stamp still carries the inline comment text the old reader kept, and the live value may already be
    restated from the file (env-comments). A live value that is neither reading is a later statement and stays a ``conflict`` for the
    operator; without the file, only identical stamps are equal. A missing stamp is re-filed as the legacy record stated it (this pass
    re-reads nothing); env-comments corrects it from the file once folded."""
    live = dict(live_extra or {})
    pend = dict(pending or {})
    missing: Dict[str, object] = {}
    equal: List[str] = []
    conflict: Dict[str, Dict[str, object]] = {}
    pending_out: Dict[str, str] = {}
    for k in sorted(lost):
        lost_v = lost[k]
        if k not in live:
            if k in pend:
                pending_out[k] = pend[k]
            else:
                missing[k] = lost_v
        elif R.canonical_json(live[k]) == R.canonical_json(lost_v) or (k == OVERRIDES_EXTRA_KEY and same_overrides_stamp(live[k], lost_v, sources)):
            equal.append(k)
        else:
            conflict[k] = {"lost": lost_v, "live": live[k]}
    return missing, equal, conflict, pending_out


def live_extra_json(conn, run_id: str) -> Tuple[bool, Optional[Dict[str, object]]]:
    """``(row exists, parsed extra_json or None)`` -- the plan's ``SELECT run_id, extra_json FROM run WHERE run_id = ?``."""
    row = conn.execute("SELECT run_id, extra_json FROM run WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return False, None
    raw = row[1]
    if raw is None or raw == "":
        return True, {}
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    return True, (dict(parsed) if isinstance(parsed, Mapping) else {})


def run_history_key(run_id: str) -> str:
    """The run's ``row_history.natural_key_json`` -- ``reducer.Effect.key_json`` of the ``run`` effect a ``run_patch`` has."""
    return R.canonical_json({"run_id": run_id})


def accepted_run_patch_shas(conn, run_id: str) -> Set[str]:
    """The ``run_patch`` events the fold ACCEPTED for a run: ``row_history`` by the run's key (a primary-key seek on
    ``(table_name, natural_key_json)``, never a scan). A patch the harvester refused rolled back before its ``row_history`` insert
    (``harvest.apply_validated``'s savepoint; ``reducer.apply_run_scale`` verifies before it inserts), so it is absent here however
    well-formed its top-level ``extra``: the run row's existence establishes nothing about THAT record (module docstring)."""
    rows = conn.execute("SELECT record_sha256 FROM row_history WHERE table_name = 'run' AND natural_key_json = ? "
                        "AND record_type = 'run_patch'", (run_history_key(run_id),)).fetchall()
    return {str(r[0]) for r in rows}


def accepted_run_statement_shas(conn, run_id: str) -> Set[str]:
    """The ``run`` AND ``run_patch`` events the fold ACCEPTED for a run (the same primary-key seek as :func:`accepted_run_patch_shas`,
    over both record types): what the multi-turn-scope hold weighs a statement of the run's columns against, since a full ``run``
    record restates its columns as a patch states its fields."""
    rows = conn.execute("SELECT record_sha256 FROM row_history WHERE table_name = 'run' AND natural_key_json = ? "
                        "AND record_type IN ('run', 'run_patch')", (run_history_key(run_id),)).fetchall()
    return {str(r[0]) for r in rows}


def unaccepted_status(conn, sha: str) -> str:
    """Why the fold holds no event for a lost record (two primary-key seeks): ``refused: <reason>`` (``record_refusal``),
    ``blocked: <parent table> <parent id>`` (``blocked_records``), else ``not_ingested`` (its file has not been folded yet)."""
    row = conn.execute("SELECT reason FROM record_refusal WHERE record_sha256 = ?", (sha,)).fetchone()
    if row is not None:
        return f"refused: {row[0]}"
    row = conn.execute("SELECT missing_parent_table, missing_parent_id FROM blocked_records WHERE record_sha256 = ?", (sha,)).fetchone()
    if row is not None:
        return f"blocked: {row[0]} {row[1]}"
    return "not_ingested"


# --------------------------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------------------------


def marker_sha256s(live_marker: Mapping[str, object], new: Sequence[str]) -> List[str]:
    """The append-only ``from_record_sha256s`` of the next marker: the live marker's list (an earlier reconciliation's provenance,
    which the reducer's shallow merge of ``extra`` would otherwise lose when the new marker replaces it whole), then the new
    records not already in it, in merge order."""
    prior = live_marker.get("from_record_sha256s") if live_marker else None
    out = [s for s in prior if isinstance(s, str)] if isinstance(prior, list) else []
    out += [s for s in new if s not in out]
    return out


def reconciliation_payload(run: RunReconciliation) -> Dict[str, object]:
    """The plan's record, exactly: the missing keys and the provenance under ``fields.extra_json``; nothing else. The marker's
    sources are :func:`marker_sha256s` -- the live marker's, kept, plus this record's."""
    extra_json: Dict[str, object] = dict(run.missing)
    extra_json[RECONCILED_KEY] = {"from_record_sha256s": marker_sha256s(run.live_reconciled, run.from_record_sha256s), "decision": DECISION}
    return {"run_id": run.run_id, "fields": {"extra_json": extra_json}}


def _writer_for(run: RunReconciliation, now: Optional[str]):
    """A ``RecordWriter`` whose directory IS the run's own (``spool/campaign/run_dir``), refusing by name when the layout
    does not resolve to it; stamps nothing of its own (module docstring)."""
    from . import emit
    directory = run.directory.rstrip(os.sep)
    campaign_dir = os.path.dirname(directory)
    spool = os.path.dirname(campaign_dir)
    producer = emit.producer_block(PRODUCER_SCRIPT, code_sha=emit.git_sha(os.path.dirname(os.path.abspath(__file__))))
    w = emit.RecordWriter(spool, os.path.basename(campaign_dir), run.run_id, producer, disabled=False, payload_extra={})
    w.payload_extra = {}     # no central stamp rides a reconciliation record (a stamp for a non-missing key would overwrite a live value)
    if os.path.abspath(w.dir) != os.path.abspath(directory):
        raise ValueError(f"the run's record directory {directory!r} is not spool/campaign/{{run_dir_name(run_id)}} "
                         f"(the writer would write to {w.dir!r}); refused by name, nothing written")
    return w


def apply_reconciliations(runs: Sequence[RunReconciliation], now: Optional[str] = None,
                          writer_factory: Optional[Callable[[RunReconciliation, Optional[str]], object]] = None) -> None:
    """Write ONE ``run_patch`` per run with missing keys, into the run's own directory -- unless the run is held, with ``refused``
    naming why: a ``malformed`` lost record (adjudicate first); a lost record the fold has not weighed yet (``blocked`` /
    ``not_ingested``: it may fold and change which record is latest); a ``pending`` key (its reconciliation record is in the log and
    not yet folded: a second record now would carry a marker without the first's provenance). Ingest, then re-run."""
    for run in runs:
        if not run.missing:
            continue
        if run.malformed:
            run.refused = "a lost record of this run is not an object; adjudicate before re-filing"
            continue
        unfolded = run.unfolded
        if unfolded:
            run.refused = ("a lost record of this run is not folded yet ("
                           + ", ".join(f"{sha[:12]} {st}" for sha, st in unfolded.items()) + "); ingest first, then re-run")
            continue
        if run.pending:
            run.refused = ("a reconciliation record of this run is not folded yet ("
                           + ", ".join(f"{sha[:12]} for {k}" for k, sha in run.pending.items()) + "); ingest first, then re-run")
            continue
        try:
            w = (writer_factory or _writer_for)(run, now)
            run.written = w.write("run_patch", reconciliation_payload(run))
        except (ValueError, OSError) as exc:
            run.refused = str(exc)


def reconcile_run_patch_extra(root: str, apply: bool = False, conn=None, now: Optional[str] = None,
                              writer_factory: Optional[Callable[[RunReconciliation, Optional[str]], object]] = None,
                              sources: Sequence[str] = ()) -> Report:
    """The pass (module docstring). Reads the log and the database (read-only; every query a primary-key seek); with ``apply``
    writes ONE ``run_patch`` per run that has missing keys and is not held (:func:`apply_reconciliations`), into the run's own
    directory. Never writes the database. ``sources`` are env-comments' overrides files (``--source``; the repository's own are always
    read), by which a lost ``run_overrides`` stamp and the live one compare as readings of one file (:func:`classify_keys`)."""
    records_root = os.path.join(root, "records")
    srcs = overrides_sources(sources)
    lost_by_run, pending_by_run, malformed_by_run, n_files, n_records = walk_lost_extras(records_root)
    own_conn = conn is None
    if own_conn:
        conn = DB.connect(DB.db_path(root), readonly=True)
    runs: List[RunReconciliation] = []
    try:
        for run_id in sorted(set(lost_by_run) | set(malformed_by_run)):
            recs = lost_by_run.get(run_id, [])
            directory = os.path.dirname(recs[-1].path) if recs else ""
            run = RunReconciliation(run_id, directory, recs, malformed=list(malformed_by_run.get(run_id, [])))
            exists, live = live_extra_json(conn, run_id)
            if not exists:
                # nothing of this run folded; the statuses say why each record did not (its run record refused or still blocked)
                run.no_live_row = True
                run.unaccepted = {r.sha: unaccepted_status(conn, r.sha) for r in recs}
                run.lost = merge_lost(recs)          # the log's statement, for the report; there is no fold to compare it against
                runs.append(run)
                continue
            accepted = accepted_run_patch_shas(conn, run_id)
            run.unaccepted = {r.sha: unaccepted_status(conn, r.sha) for r in recs if r.sha not in accepted}
            run.lost = merge_lost(run.accepted)
            marker = live.get(RECONCILED_KEY) if live else None
            run.live_reconciled = dict(marker) if isinstance(marker, Mapping) else {}
            run.missing, run.equal, run.conflict, run.pending = classify_keys(run.lost, live, pending_by_run.get(run_id), srcs)
            runs.append(run)
    finally:
        if own_conn:
            conn.close()
    if apply:
        apply_reconciliations(runs, now, writer_factory)
    return Report(root, runs, n_files, n_records, apply, sources=sorted(srcs.values(), key=lambda s_: s_.path))


def format_report(rep: Report, verbose_equal: bool = False) -> str:
    """The dry run's text: one line per run that is not plainly equal (every run with ``verbose_equal``), then the totals."""
    lines = []
    for r in rep.runs:
        if r.bucket == "equal" and not verbose_equal:
            continue
        parts = [f"{r.bucket:<11} {r.run_id}  ({len(r.records)} lost record(s), {os.path.relpath(r.directory, rep.root) if r.directory else '-'})"]
        if r.missing:
            parts.append("    missing: " + ", ".join(f"{k}={R.canonical_json(v)}" for k, v in r.missing.items()))
        if r.equal:
            parts.append("    equal: " + ", ".join(r.equal))
        for k, v in r.conflict.items():
            parts.append(f"    CONFLICT {k}: lost={R.canonical_json(v['lost'])} live={R.canonical_json(v['live'])}")
        for k, sha in r.pending.items():
            parts.append(f"    pending {k}: reconciliation record {sha[:12]} not yet folded")
        for sha, st in r.unaccepted.items():
            parts.append(f"    unaccepted {sha[:12]}: {st}")
        for m in r.malformed:
            parts.append(f"    malformed: {m}")
        if r.written:
            parts.append(f"    wrote {os.path.relpath(r.written, rep.root)}")
        if r.refused:
            parts.append(f"    refused: {r.refused}")
        lines.append("\n".join(parts))
    b, k, a = rep.by_bucket(), rep.keys_by_bucket(), rep.records_by_status()
    lines.append(f"overrides sources (a run_overrides stamp compares as a reading of its file by these, matched by sha256): {len(rep.sources)} file(s)"
                 + "".join(f"\n  {s.sha256[:8]} {s.id:<36} {s.path}" for s in rep.sources))
    lines.append(f"runs {len(rep.runs)}: missing {b['missing']} / equal {b['equal']} / conflict {b['conflict']}"
                 + (f" / pending {b['pending']}" if b["pending"] else "") + (f" / unaccepted {b['unaccepted']}" if b["unaccepted"] else "")
                 + (f" / no_live_row {b['no_live_row']}" if b["no_live_row"] else "")
                 + (f" / malformed {b['malformed']}" if b["malformed"] else "")
                 + f"   (keys: missing {k['missing']} / equal {k['equal']} / conflict {k['conflict']}"
                 + (f" / pending {k['pending']}" if k["pending"] else "")
                 + f"; lost records: accepted {a['accepted']} / refused {a['refused']} / blocked {a['blocked']} / not_ingested {a['not_ingested']}"
                 + f"; {rep.records_read} records in {rep.files_read} run_patch files)")
    if rep.applied:
        lines.append(f"wrote {len(rep.written)} reconciliation record(s)")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# multi-turn-scope: the steering scope of a run that registered with the job-start default, re-derived from what the row carries
# --------------------------------------------------------------------------------------------
#
# A job-start registration (jobplan._plan) and a store's own manifest registration (hooks.run_payload) used to stamp
# `multi_turn_steering_scope = single_turn` on every steered run, whatever the eval: the scope was never derived there, only in the
# harvester's adapters. Every multi-turn, Petri and agentic run those two paths registered therefore reads `single_turn` -- a wrong
# value, not a missing one. Both paths derive the scope now (derive.multi_turn_steering_scope over the run's applied-positions mode,
# the seeded eval's category and the object's kind); THIS pass corrects the rows already in the database the same way, as records: one
# `run_patch` per run carrying the derived scope, the scope's own `field_confidence_json` entry (the fold reads a patch's confidence JSON
# per column, so the row's other entries stand and the record needs no sight of them), and a marker under `fields.extra_json` naming the
# value it replaced and what the derivation read. The rows are read from the LIVE database (read-only). THE CLASS: `eval.eval_category`
# in vocab.MULTI_TURN_EVAL_CATEGORIES and `object_id != 'none'`, joined to the run's `object` row for its kind. THE PREDICATE: a run of the
# class whose `multi_turn_steering_scope` differs from what the derivation reads from its own (mode, category, kind) -- the job-start
# default `single_turn` is the common case, and an adapter's `unverified` scope stamped while the mode was still unresolved beside a mode
# a later record settled is the other; a row that already carries the derived value is consistent and gets no record, whatever stamped
# it. A system-instruction object steers no positions whatever mode the row carries (a job-start registration copies the selected
# instrument's protocol mode onto a prompted run too), so its scope derives `not_applicable`, never the mode's; a run of the class whose
# object row is missing refuses the whole pass by name (the kind cannot be read, so neither consistency nor the scope can, and the
# schema's reference says the row exists). The dry run counts the selected runs by (model, category) and by object kind; `--apply` writes
# the records into each run's own directory (`<root>/records/<campaign>/<run_dir_name(run_id)>`, the RecordWriter's rule for every run
# record) and never the database; the landing window's reconcile slot runs the dry run, then the apply, so the records fold in the
# window's one ingest. Idempotent: after the fold the predicate selects none of them.
#
# THE VERIFY AND THE FIXED POINT. `--verify` (:func:`verify_multi_turn_scope`) is the predicate alone over the FOLDED database: every run
# of the class whose (mode, scope) is inconsistent with the derivation, listed by name, read-only, no walk of the log and no hold. It
# exists because the hold below cannot close every race: the pass snapshots a run's mode, walks the log, re-reads the run's own
# directory at the write -- and a standup's endpoint patch (served `response` beside the scope that follows it) published into ANOTHER
# record directory after that last read is seen by none of it, and the pass's later-written statement, derived from the snapshot's
# `all`, wins the fold over the endpoint's: (`response`, `every_turn_prefill_and_decode`), a rebuild agreeing. Rather than a lock across
# every producer or a conditional record, the landing window runs the pass to a VERIFIED fixed point: apply -> ingest -> verify; a run
# the verify lists is reconciled ONCE more, its patch derived from the now-folded mode -> ingest -> verify, which must read zero, else
# the window fails by name with the runs (scripts/registry/landing/window.sh, exit 29). Endpoint patches state mode and scope together,
# so the steady state stays consistent and only this one-off pass can race; a second pass converges because the predicate is the
# verify's own.
#
# THE HOLD. The pass derives from the live row and writes a statement of the scope, so any record ALREADY IN THE LOG that the fold has
# not weighed yet and that states the run's mode or scope makes the derivation stale or the statement a duplicate: this pass's own
# marker record from an earlier apply, and equally a standup's endpoint patch (`hooks.serving_patch_payload` stamps the served mode and
# the scope that follows it) pushed into the log and not ingested -- folded in the same ingest as a correction derived from the row's
# old mode, the two would leave the row inconsistent (a served `response` beside a scope derived from `all`). So before anything is
# written, EVERY `run_patch` record under the whole log (every directory, not the run's own alone: a harvester file, a producer's spool
# pushed under another name) whose fields state `applied_positions_mode` or `multi_turn_steering_scope` is read, AND every full `run`
# record there that carries either column -- a re-registration of the identity, an adapter's row, a store's opening (whose copies of the
# plan's mode and scope are marked unstated: whether they apply depends on the row at the fold, which this pass does not predict) -- and a
# selected run with such a record that the fold has not accepted (`row_history`, the run's `run` and `run_patch` events alike) and not
# refused (a refusal is final; a blocked or not-ingested record may still fold) is held (`pending`, with the record's type and the columns
# it states): the dry run reports it, the apply writes nothing for it and the CLI exits 2 -- ingest first, then re-run, and the predicate
# decides again from the folded row. The run's own directory is read once more at the write, so a statement that lands between the walk
# and the write holds the run too; a record not yet pushed from a producer's spool is outside the log and outside this guard, as it is
# outside the fold's. The walk reads the two record types' own file names (`*-run.jsonl`, `*-run_patch.jsonl`, `.gz` too: every writer
# names a record file by its record type), never the item-scale files.

SCOPE_ACTION = "multi-turn-scope"
SCOPE_PRODUCER_SCRIPT = "registry reconcile multi-turn-scope"
SCOPE_RECONCILED_KEY = "reconciled_multi_turn_scope"
#: the job-start default both self-registration paths used to stamp: the common value the predicate finds, not the predicate itself
SCOPE_DEFAULT = "single_turn"
#: the run columns whose statement, unfolded, holds a selected run (the section comment): what the pass derives from, and what it writes --
#: a `run_patch` states them under `fields`, a full `run` record carries them at its top level
SCOPE_STATEMENT_COLUMNS: Tuple[str, ...] = ("applied_positions_mode", "multi_turn_steering_scope")
#: the record types a statement of those columns rides on
SCOPE_STATEMENT_RECORD_TYPES: Tuple[str, ...] = ("run", "run_patch")


@dataclass(frozen=True)
class ScopeStatement:
    """One record in the log that states a run's mode or scope (:data:`SCOPE_STATEMENT_COLUMNS`) -- a ``run_patch`` naming either under its
    ``fields``, a full ``run`` record carrying either -- or a ``run_patch`` carrying this pass's marker: what holds a selected run until the
    fold has weighed it."""
    run_id: str
    sha: str
    path: str
    written_utc: str
    columns: Tuple[str, ...]
    own: bool                                                               # this pass's own correction record (the marker)
    record_type: str = "run_patch"                                          # `run_patch` or `run` (SCOPE_STATEMENT_RECORD_TYPES)

    @property
    def order_key(self) -> Tuple[str, str]:
        return (self.written_utc, self.sha)


@dataclass
class ScopeCorrection:
    """One run the predicate selected: what the row carries, what the derivation reads from it, and what ``apply`` did."""
    run_id: str
    campaign: str
    model_id: str
    eval_id: str
    eval_category: str
    object_kind: str                                                        # the run's object row's kind, what the derivation reads
    applied_positions_mode: str
    live_scope: str
    scope: str                                                              # the derived value (`unverified` when the rule cannot decide)
    confidence: str
    source: str
    pending: Optional[str] = None                                           # sha of an unfolded record stating the mode or scope (the newest)
    pending_columns: Tuple[str, ...] = ()                                   # ... the columns it states
    pending_own: bool = False                                               # ... whether it is this pass's own correction record
    pending_type: str = ""                                                  # ... its record type (`run_patch`, or `run` for a full record)
    written: Optional[str] = None
    refused: Optional[str] = None

    def hold(self, statement: "ScopeStatement") -> None:
        """Record the unfolded statement that holds this run."""
        self.pending, self.pending_columns, self.pending_own = statement.sha, statement.columns, statement.own
        self.pending_type = statement.record_type

    @property
    def hold_reason(self) -> str:
        what = ("this pass's own correction record" if self.pending_own else "a full run record" if self.pending_type == "run" else "a record")
        return (f"{what} of this run stating {', '.join(self.pending_columns) or 'its scope'} is in the log and not folded yet "
                f"({(self.pending or '')[:12]}); ingest first, then re-run")

    @property
    def group(self) -> Tuple[str, str]:
        return (self.model_id, self.eval_category)


@dataclass
class ScopeReport:
    root: str
    runs: List[ScopeCorrection]
    applied: bool
    verify: bool = False                                                    # the predicate alone over the folded database (no hold, no write)

    def by_group(self) -> Dict[Tuple[str, str], int]:
        out: Dict[Tuple[str, str], int] = {}
        for r in self.runs:
            out[r.group] = out.get(r.group, 0) + 1
        return dict(sorted(out.items()))

    def by_scope(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.runs:
            out[r.scope] = out.get(r.scope, 0) + 1
        return dict(sorted(out.items()))

    def by_kind(self) -> Dict[Tuple[str, str], int]:
        """Runs per (object kind, derived scope): what the kind rule decided beside what the mode table decided."""
        out: Dict[Tuple[str, str], int] = {}
        for r in self.runs:
            out[(r.object_kind, r.scope)] = out.get((r.object_kind, r.scope), 0) + 1
        return dict(sorted(out.items()))

    @property
    def written(self) -> List[str]:
        return [r.written for r in self.runs if r.written]

    @property
    def held(self) -> List[ScopeCorrection]:
        return [r for r in self.runs if r.refused]

    def as_dict(self) -> Dict[str, object]:
        return {"root": self.root, "action": SCOPE_ACTION, "applied": self.applied, "verify": self.verify, "runs": len(self.runs),
                "by_model_and_category": [{"model_id": m, "eval_category": c, "runs": n} for (m, c), n in self.by_group().items()],
                "by_derived_scope": self.by_scope(),
                "by_object_kind": [{"object_kind": k, "scope": s, "runs": n} for (k, s), n in self.by_kind().items()],
                "written": len(self.written), "held": len(self.held),
                "corrections": [{"run_id": r.run_id, "campaign": r.campaign, "model_id": r.model_id, "eval_id": r.eval_id,
                                 "eval_category": r.eval_category, "object_kind": r.object_kind,
                                 "applied_positions_mode": r.applied_positions_mode,
                                 "live_scope": r.live_scope, "scope": r.scope, "confidence": r.confidence, "source": r.source,
                                 "pending": r.pending, "pending_columns": list(r.pending_columns), "pending_own": r.pending_own,
                                 "pending_type": r.pending_type or None,
                                 "written": r.written, "refused": r.refused} for r in self.runs]}


def _scope_statements_in(path: str, run_ids: Optional[Set[str]] = None) -> List[ScopeStatement]:
    """The :class:`ScopeStatement` records of one record file: a ``run_patch`` whose ``fields`` state a column of
    :data:`SCOPE_STATEMENT_COLUMNS` or whose ``fields.extra_json`` carries the pass's marker, and a full ``run`` record that carries either
    column (marked unstated or not: the fold decides whether a marked copy applies, this pass only waits for it); every other record type
    in the file is passed over. ``run_ids`` narrows to those runs."""
    out: List[ScopeStatement] = []
    for _i, rec, _sha in R.read_record_file(path, compute_sha=False):
        record_type = rec.get("record_type")
        if record_type not in SCOPE_STATEMENT_RECORD_TYPES:
            continue
        payload = rec.get("payload") if isinstance(rec.get("payload"), Mapping) else None
        run_id = payload.get("run_id") if payload else None
        if not isinstance(run_id, str) or not run_id or (run_ids is not None and run_id not in run_ids):
            continue
        if record_type == "run":
            columns = tuple(c for c in SCOPE_STATEMENT_COLUMNS if c in payload)              # type: ignore[operator]
            own = False
        else:
            fields = payload.get("fields") if isinstance(payload.get("fields"), Mapping) else {}   # type: ignore[union-attr]
            columns = tuple(c for c in SCOPE_STATEMENT_COLUMNS if c in fields)
            extra = _extra_json_object(fields.get("extra_json"))
            own = extra is not None and SCOPE_RECONCILED_KEY in extra
        if columns or own:
            out.append(ScopeStatement(run_id, R.record_sha(rec), path, str(rec.get("written_utc", "")), columns, own, str(record_type)))
    return out


def scope_statements(records_root: str, run_ids: Optional[Set[str]] = None) -> Dict[str, List[ScopeStatement]]:
    """Every statement of a run's mode or scope in the WHOLE log (:func:`run_statement_files` over ``records_root``: every ``run`` and
    ``run_patch`` file in every directory, a listing error raised by name), by run id, each run's in ``(written_utc, sha)`` order;
    ``run_ids`` narrows to the selected runs."""
    out: Dict[str, List[ScopeStatement]] = {}
    for path in run_statement_files(records_root):
        for s in _scope_statements_in(path, run_ids):
            out.setdefault(s.run_id, []).append(s)
    for recs in out.values():
        recs.sort(key=lambda s: s.order_key)
    return out


def scope_statements_in_directory(directory: str, run_id: str) -> List[ScopeStatement]:
    """The statements of one run under one directory (the run's own, read again at the write): its ``run`` and ``run_patch`` files."""
    if not os.path.isdir(directory):
        return []
    out: List[ScopeStatement] = []
    for fn in sorted(os.listdir(directory)):
        if _RUN_STATEMENT_FILE.search(fn):
            out.extend(_scope_statements_in(os.path.join(directory, fn), {run_id}))
    out.sort(key=lambda s: s.order_key)
    return out


def unfolded_statements(conn, statements: Sequence[ScopeStatement], accepted: Set[str]) -> List[ScopeStatement]:
    """The statements the fold has not weighed: not among the run's accepted ``run`` / ``run_patch`` events
    (:func:`accepted_run_statement_shas`) and not refused (a refusal is final and never folds; a blocked or not-ingested record may)."""
    return [s for s in statements if s.sha not in accepted and not unaccepted_status(conn, s.sha).startswith("refused")]


def scope_correction_payload(run: ScopeCorrection) -> Dict[str, object]:
    """The record: the derived scope, the scope's own confidence entry (the fold reads a patch's confidence JSON per column, so the
    row's other entries stand without the record carrying them), and the marker naming the replaced value and the derivation's inputs.
    Nothing else: no identity field, no central stamp."""
    entry: Dict[str, object] = {"confidence": run.confidence, "source": run.source}
    if run.confidence == "unverified":
        entry["reason"] = run.source
    marker = {"replaced": run.live_scope, "derived_from": {"applied_positions_mode": run.applied_positions_mode,
                                                           "eval_category": run.eval_category, "object_kind": run.object_kind},
              "pass": SCOPE_PRODUCER_SCRIPT}
    return {"run_id": run.run_id, "fields": {"multi_turn_steering_scope": run.scope,
                                             "field_confidence_json": {"multi_turn_steering_scope": entry},
                                             "extra_json": {SCOPE_RECONCILED_KEY: marker}}}


def _scope_writer_for(root: str, run: ScopeCorrection, now: Optional[str]):
    """A ``RecordWriter`` for the run's own directory (``<root>/records/<campaign>/<run_dir_name(run_id)>``, the rule every run record
    is written under); stamps nothing of its own."""
    from . import emit
    producer = emit.producer_block(SCOPE_PRODUCER_SCRIPT, code_sha=emit.git_sha(os.path.dirname(os.path.abspath(__file__))))
    w = emit.RecordWriter(os.path.join(root, "records"), run.campaign, run.run_id, producer, disabled=False, payload_extra={})
    w.payload_extra = {}
    return w


def scope_class_rows(conn) -> List[ScopeCorrection]:
    """The predicate (the section comment): every run of the class -- a multi-turn, Petri or agentic eval, ``object_id != 'none'`` -- whose
    live scope differs from what :func:`derive.multi_turn_steering_scope` reads from its own mode, its eval's category and its object's
    kind, in run-id order, each with the derived value. A run of the class whose object row is missing raises ValueError by name: its kind
    cannot be read, so neither its consistency nor its scope can, and a pass that guessed the kind would file a steered scope on what may
    be a prompted run. Reads the database only."""
    from . import derive as D
    from . import vocab as V
    marks = ",".join("?" for _ in V.MULTI_TURN_EVAL_CATEGORIES)
    rows = conn.execute(
        "SELECT r.run_id, r.campaign, r.model_id, r.eval_id, e.eval_category, o.object_kind, r.object_id, r.applied_positions_mode, "
        "r.multi_turn_steering_scope "
        "FROM run r JOIN eval e ON e.eval_id = r.eval_id LEFT JOIN object o ON o.object_id = r.object_id "
        f"WHERE e.eval_category IN ({marks}) AND r.object_id != 'none' ORDER BY r.run_id",
        tuple(V.MULTI_TURN_EVAL_CATEGORIES)).fetchall()
    missing = [(str(run_id), str(object_id)) for run_id, _c, _m, _e, _cat, kind, object_id, _mode, _live in rows if kind is None]
    if missing:
        shown = ", ".join(f"{rid} -> object {oid}" for rid, oid in missing[:5]) + (", ..." if len(missing) > 5 else "")
        raise ValueError(f"{SCOPE_ACTION}: {len(missing)} run(s) of the class name an object with no object row ({shown}); the object's "
                         "kind decides the scope and cannot be read, so nothing is derived -- fold the object records first")
    out: List[ScopeCorrection] = []
    for run_id, campaign, model_id, eval_id, category, kind, _object_id, mode, live_scope in rows:
        scope, confidence, source = D.multi_turn_steering_scope(str(mode), str(category), str(kind))
        if str(scope) == str(live_scope):
            continue                                                        # consistent: no record, whatever stamped the value
        out.append(ScopeCorrection(str(run_id), str(campaign), str(model_id), str(eval_id), str(category), str(kind), str(mode),
                                   str(live_scope), str(scope), str(confidence), str(source)))
    return out


def verify_multi_turn_scope(root: str, conn=None) -> ScopeReport:
    """The verify (the section comment): the predicate alone over the FOLDED database -- every run of the class whose (mode, scope) is
    inconsistent with the derivation, by name, with the value the derivation reads. Read-only; no walk of the log, no hold, nothing
    written. The landing window reads it after its ingest and requires zero (after one more apply and ingest when it lists runs)."""
    own_conn = conn is None
    if own_conn:
        conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        runs = scope_class_rows(conn)
    finally:
        if own_conn:
            conn.close()
    return ScopeReport(root, runs, applied=False, verify=True)


def reconcile_multi_turn_scope(root: str, apply: bool = False, conn=None, now: Optional[str] = None,
                               writer_factory: Optional[Callable[[str, ScopeCorrection, Optional[str]], object]] = None) -> ScopeReport:
    """The pass (the section comment above). Reads the live database read-only; with ``apply`` writes ONE ``run_patch`` per selected
    run (:func:`scope_class_rows`: a run of the class whose scope differs from the derived value) into the run's own directory, holding a
    run with an unfolded record anywhere in the log that states its mode or scope (this pass's own marker record, a standup's endpoint
    patch, a full ``run`` record carrying either column), and reading the run's own directory once more at the write. Never writes the
    database. Raises ValueError when a run of the class names an object with no row: its kind cannot be read, and a pass that guessed it
    would file a steered scope on what may be a prompted run."""
    from . import identity as I
    own_conn = conn is None
    if own_conn:
        conn = DB.connect(DB.db_path(root), readonly=True)
    runs: List[ScopeCorrection] = []
    accepted_by_run: Dict[str, Set[str]] = {}
    try:
        runs = scope_class_rows(conn)
        # the hold (the section comment): every statement of a selected run's mode or scope anywhere in the log, read AFTER the rows so
        # a record that landed since the read is seen; the fold's own verdicts decide which of them it has weighed
        statements = scope_statements(os.path.join(root, "records"), {r.run_id for r in runs}) if runs else {}
        for run in runs:
            accepted_by_run[run.run_id] = accepted_run_statement_shas(conn, run.run_id)
            unfolded = unfolded_statements(conn, statements.get(run.run_id, []), accepted_by_run[run.run_id])
            if unfolded:
                run.hold(unfolded[-1])
        if apply:
            for run in runs:
                if run.pending:
                    run.refused = run.hold_reason
                    continue
                directory = os.path.join(root, "records", run.campaign, I.run_dir_name(run.run_id))
                try:
                    # the run's own directory once more, at the write: a statement that landed between the walk and this moment holds too
                    late = unfolded_statements(conn, scope_statements_in_directory(directory, run.run_id), accepted_by_run[run.run_id])
                    if late:
                        run.hold(late[-1])
                        run.refused = run.hold_reason
                        continue
                    w = (writer_factory or _scope_writer_for)(root, run, now)
                    run.written = w.write("run_patch", scope_correction_payload(run))
                except (ValueError, OSError) as exc:
                    run.refused = str(exc)
    finally:
        if own_conn:
            conn.close()
    return ScopeReport(root, runs, apply)


def format_verify_report(rep: ScopeReport) -> str:
    """The verify's text: one line per inconsistent run -- its id first, then (mode, live scope) -> the derived value and the object's kind
    (the landing window reads the ids off these lines when it fails by name) -- and the count last."""
    lines = ["multi-turn-scope verify (read-only, the folded database): runs of the class -- a multi-turn eval, object_id != none -- whose "
             "(applied_positions_mode, multi_turn_steering_scope) disagrees with the derivation"]
    for r in rep.runs:
        lines.append(f"  {r.run_id} ({r.applied_positions_mode}, {r.live_scope}) -> {r.scope} [{r.object_kind}]")
    lines.append(f"inconsistent {len(rep.runs)} run(s)" + ("; the class is consistent" if not rep.runs else ""))
    return "\n".join(lines)


def format_scope_report(rep: ScopeReport) -> str:
    """The dry run's text: one line per (model, category) with its count, the derived values, the pending / written / held lines, the
    total last (the landing window shows the tail). A verify report renders through :func:`format_verify_report`."""
    if rep.verify:
        return format_verify_report(rep)
    lines = ["multi-turn-scope: runs whose eval category is multi-turn, object_id != none and multi_turn_steering_scope differs from the "
             f"derived value (the job-start default {SCOPE_DEFAULT} the common case)"]
    for (model_id, category), n in rep.by_group().items():
        lines.append(f"  {model_id:<24} {category:<12} {n:>6}")
    lines.append("derived scope: " + (", ".join(f"{s} {n}" for s, n in rep.by_scope().items()) or "none"))
    lines.append("by object kind: " + (", ".join(f"{k} -> {s} {n}" for (k, s), n in rep.by_kind().items()) or "none"))
    pending = [r for r in rep.runs if r.pending]
    if pending:
        lines.append(f"pending (a record stating the run's mode or scope is in the log, not folded yet): {len(pending)} run(s)" + "".join(
            f"\n  {r.run_id} record {r.pending[:12]} states {', '.join(r.pending_columns) or 'its scope'}"
            + (" (this pass's own correction)" if r.pending_own else " (a full run record)" if r.pending_type == "run" else "")
            for r in pending[:20]) + ("\n  ..." if len(pending) > 20 else ""))
    if rep.applied:
        for r in rep.held:
            lines.append(f"  held {r.run_id}: {r.refused}")
        lines.append(f"wrote {len(rep.written)} correction record(s); held {len(rep.held)}")
    lines.append(f"runs {len(rep.runs)} in {len(rep.by_group())} (model, category) group(s)" + ("" if rep.applied else "; nothing written (dry run)"))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# env-comments: the run-overrides stamp restated from its overrides file of record, where the old reader kept inline comment text
# --------------------------------------------------------------------------------------------
#
# `emit._parse_env_file` read a `KEY=value   # comment` line of an overrides file as the whole tail after `=`, so every `run_overrides.keys`
# value on a line with an inline comment folded WITH the comment text (`OVERRIDE_EFFECTIVE_BATCH = "64   # unchanged: the paper's"`), on
# every record the writer stamped: the run, its state events, its judgments, metrics and degeneration rows, the objects it emitted. The
# reader now reads a value by the shell's rule (`emit.env_value`), and THIS pass restates the stamps already folded, as records: one
# `run_patch` per `run` row whose stamp differs from what the reader makes of ITS OWN FILE, with the WHOLE stamp under
# `fields.extra_json.run_overrides` (the fold's shallow merge of `extra` keeps the latest record's value of a key whole: `path` and `sha256`
# -- the file's own digest -- ride along unchanged, `keys` are the file's reading, `id` the file's OVERRIDE_ID), a marker beside it naming
# every replaced key with the value it replaced and the file it was read from, and a `field_confidence_json` entry on the one column the
# record restates (`extra_json`) naming the pass.
#
# THE EVIDENCE IS THE FILE, NEVER THE STORED TEXT. A stored value alone cannot say what its line was: `note #42` is what the reader makes
# of `OVERRIDE_NOTE='note #42'` (a literal `#`, kept by the old reader and the new one alike) and also what the old reader kept of
# `OVERRIDE_NOTE=note #42 ...` had the comment been one word long; read as shell source again, the first would lose its literal content,
# and the "correction" would be a durable record of a value no file ever carried. So the pass takes the overrides files themselves
# (`--source`: a file, or a directory of `.env` files; the repository's own `reproduce/overrides` is always read), matches each to the
# stamps by the file's sha256 -- the digest the writer stamped IS the bytes it read -- and restates a stamp from the matched file's reading
# by the fixed reader. THE CLASS: `run` rows whose `extra_json.run_overrides` is an object with a `keys` object. SELECTED: a row whose file
# is given and whose stored `keys` (or `id`) differ from the file's reading, EVERY differing value being what the OLD reader made of the
# file's line (`historical_env_value`, the reader before the fix replicated here and read over the same file: the whole right-hand side,
# surrounding quotes dropped) -- the artifact this pass exists to correct, established per key, never assumed from the digest. A stored
# value that is neither the file's reading nor the old reader's is a LATER STATEMENT, not an artifact (a later accepted patch restating one
# key for one run while keeping the file's digest as provenance; a hand edit): the file's digest and key set alone cannot tell it from the
# artifact, and restating it would overwrite a clean, deliberate value with the file's. Such a row is UNCLASSIFIED, refused by name with the
# key, the stored value and both readings, and nothing of it is restated. AMBIGUOUS, listed and never corrected: a row whose file is not
# given and whose stored values MAY carry a comment (`looks_polluted`: a value that is not a fixed point of the reader's rule, or that
# begins with `#`) -- undecidable without the file; a row whose file is not given and whose values look clean is nothing to do. UNCLASSIFIED
# too, refused by name: a row whose file is given but whose stored key SET is not the file's (not a reading of that file). Idempotent by
# construction: a restated stamp IS the file's reading, so after the fold the row leaves the class; a stamp whose stored values are the
# file's reading already -- a quoted literal `#`, a value the fixed reader stamped -- is never selected and never touched. The pass reads
# the live database read-only and never writes it.
#
# ONLY `run` is corrected. The stamp also rides the OTHER tables' rows (run_state_event, judgment, run_metric, run_degeneration, object,
# fit_search, fit_trial, item_result), each a FULL statement folded per natural key with no patch kind: restating one would mean re-emitting
# the whole row (for a state event a new event that moves the run's state; for an object a record that re-enters the unstated and origin
# rules), which is the owner's design call, not a correction's. The dry run COUNTS those rows per table (the item-scale table is not
# scanned) and corrects none.
#
# THE HOLD, as multi-turn-scope's: a selected run with a record anywhere in the log that STATES its `run_overrides` and that the fold has not
# weighed -- a full `run` record carrying the stamp under its `extra` (a re-registration, an adapter's row), a `run_patch` carrying it
# under `fields.extra_json` (a writer on the fixed reader stamping the clean value; this pass's own earlier record, by its marker) -- is
# held: the dry run reports it, the apply writes nothing for it (exit 2), and the run's own directory is read once more at the write. A
# legacy `run_patch` whose stamp rides a TOP-LEVEL `extra` never folds the key (run-patch-extra's business) and holds nothing. The verify
# (`--verify`, with the same sources) is the predicate alone over the folded `run` table, read-only, exit 3 when a selected, an
# unclassified or an ambiguous row remains; the landing window runs apply -> ingest -> verify as for multi-turn-scope, passing the sources
# to every invocation.
#
# AN UNSOURCED STAMP IS NOT A CLEAN CLASS. An ambiguous row is listed and never corrected, but it is not nothing: with no file given for
# any stamp (the sources unset, or a directory that holds none of the files the stamps were read from) EVERY polluted row is ambiguous,
# and a pass that counted only what it could decide would write nothing, exit 0, and verify "consistent" over a table it had not touched
# -- a landing window completing with every stored value as it was. So an ambiguous row fails the apply (refused by name: no file among
# the sources has the stamp's sha256 and its values may carry a comment; exit 2, nothing written for it, the selected rows beside it still
# written) and the verify (listed, exit 3), until its file is given -- or the operator LEAVES IT BY NAME: `--leave-unsourced` keeps the
# ambiguous rows listed as stored and lets the apply and the verify complete past them (a file genuinely lost is a decision, recorded in
# the window's log, not a default). `--check-sources` is the coverage alone, read-only over the `run` table without the walk or the other
# tables: every stamped row whose values may carry a comment has its file among the sources (exit 0), or the uncovered stamps are
# listed by (overrides id, sha prefix) with their keys (exit 3); the landing window runs it before anything moves when env-comments is
# listed for apply, so a window never applies its migration on a source list that covers nothing. Rejected: keeping the ambiguous rows
# out of the apply's and the verify's verdict (the first design) -- it made the correction's success independent of whether it had
# corrected anything.
#
# run-patch-extra compares a LEGACY stamp to the live one by the SAME evidence (`same_overrides_stamp`, with the same `--source` files): two
# stamps that are not identical are statements of ONE file -- `equal`, never `conflict` -- only when the file is given and EACH value of
# BOTH is one of the file's two readings (the fixed reader's or the old one's), so a lost stamp with comment text and a live stamp restated
# by this pass read `equal`, while a live value that is neither reading (a later statement) stays a CONFLICT for the operator, and without
# the file only identical stamps are equal. A stamp run-patch-extra re-files is the legacy record's own, which this pass corrects from the
# file once it is folded. Rejected: reading a stored value as shell source again (the first design of this pass, and of run-patch-extra's
# comparison) -- it cannot tell a literal `#` from a comment, and would strip legitimate quoted content into a durable record; and calling
# every two stamps of one digest and key set equal whatever their values (the second design of the comparison) -- it hid a divergent value
# from the operator, as restating every differing key from the file hid it from the correction.

ENV_COMMENTS_ACTION = "env-comments"
ENV_COMMENTS_PRODUCER_SCRIPT = "registry reconcile env-comments"
ENV_COMMENTS_RECONCILED_KEY = "reconciled_run_overrides_comments"
#: the key the stamp rides under (emit.RUN_OVERRIDES_KEY, spelled here so this module's import footprint stays the fold's)
OVERRIDES_EXTRA_KEY = "run_overrides"
#: the run-scale tables whose rows carry the stamp as full statements: counted by the dry run, never corrected here
OVERRIDES_ROW_TABLES: Tuple[str, ...] = ("run_state_event", "judgment", "run_metric", "run_degeneration", "object", "fit_search", "fit_trial")
#: the item-scale table that carries it too: listed as not counted (a whole-table scan of item rows is not a dry run's cost)
OVERRIDES_ITEM_TABLES: Tuple[str, ...] = ("item_result",)
#: the repository's own overrides files (`reproduce/overrides/*.env`), read as a source wherever the package runs from a checkout: a stamp
#: is matched to a file by digest, so a file that is not a stamp's original matches nothing
REPO_OVERRIDES_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "reproduce", "overrides"))
_ENV_FILE = re.compile(r"\.env$")
#: the key of a stamp's `id` in the marker and the flags (never an OVERRIDE_* key, so it collides with no key of `keys`)
_STAMP_ID = "id"


@dataclass(frozen=True)
class OverridesSource:
    """One overrides file offered as evidence: its path, its digest, and the TWO readings of it -- ``id`` and ``keys`` as the fixed reader
    makes them (:func:`emit.run_overrides_stamp`), ``historical_keys`` as the reader before the fix made them (:func:`historical_env_value`
    over the same lines). A stored value is this file's when it is one of the two; the historical reading is what a value the pass
    corrects must equal, and the one thing the digest and key set alone cannot establish."""
    path: str
    sha256: str
    id: str
    keys: Dict[str, str]
    historical_keys: Dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    def readings(self, key: str) -> Tuple[str, ...]:
        """The values a stored ``keys[key]`` may carry as a reading of this file: the fixed reader's, then the old reader's."""
        out = [self.keys[key]] if key in self.keys else []
        if key in self.historical_keys and self.historical_keys[key] not in out:
            out.append(self.historical_keys[key])
        return tuple(out)

    def ids_for(self, stamp: Mapping[str, object]) -> Tuple[str, str]:
        """``(the fixed reader's id, the old reader's id)`` of a stamp of this file, as the writer derives an id: the file's ``OVERRIDE_ID``
        by that reader, else the stem of the STORED ``path`` (the stamp's own name for the file; a renamed copy keeps its stem), else the
        stored ``id``."""
        stored_path = str(stamp.get("path") or "")
        fallback = os.path.splitext(stored_path)[0] if stored_path else str(stamp.get("id") or "")
        return (self.keys.get("OVERRIDE_ID") or fallback, self.historical_keys.get("OVERRIDE_ID") or fallback)


def historical_env_value(raw: str) -> str:
    """What the overrides-file reader BEFORE the fix made of a line's right-hand side: the whole text after ``=``, stripped, the surrounding
    quotes dropped only when one pair wraps it whole -- so ``64   # unchanged`` read ``64   # unchanged``, ``''   # c`` read
    ``''   # c`` (the quotes do not wrap the whole), ``   # c`` read ``# c``, and ``'note #42'`` read ``note #42`` as the fixed reader does.
    Kept as EVIDENCE, never as a reader: a stored value equal to this reading of its file's line is that reader's artifact, which
    ``registry reconcile env-comments`` corrects; a stored value equal to neither reading is a later statement, which it refuses by name."""
    v = raw.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1]
    return v


def default_source_dirs() -> List[str]:
    """The source directories read without being named: the repository's ``reproduce/overrides`` when the package runs from a checkout."""
    return [REPO_OVERRIDES_DIR] if os.path.isdir(REPO_OVERRIDES_DIR) else []


def overrides_sources(paths: Sequence[str] = (), include_defaults: bool = True) -> Dict[str, OverridesSource]:
    """The sources by digest: every file named in ``paths`` (a directory contributes its ``*.env`` files, in name order) after the default
    directories, each read twice (the fixed reader, the historical one). Two files of one digest are one source (the same bytes read the
    same). A path that is neither a file nor a directory is refused (``FileNotFoundError``): a named source that cannot be read is a
    misconfiguration, never a silent no-source."""
    from . import emit
    files: List[str] = []
    for p in list(default_source_dirs() if include_defaults else []) + [os.path.expanduser(str(x)) for x in paths]:
        if os.path.isdir(p):
            files.extend(os.path.join(p, fn) for fn in sorted(os.listdir(p)) if _ENV_FILE.search(fn) and os.path.isfile(os.path.join(p, fn)))
        elif os.path.isfile(p):
            files.append(p)
        else:
            raise FileNotFoundError(f"no overrides file or directory at {p!r}")
    out: Dict[str, OverridesSource] = {}
    for f in files:
        st = emit.run_overrides_stamp(f)
        assert st is not None
        if str(st["sha256"]) in out:
            continue
        old = {str(k): str(v) for k, v in emit._parse_env_file(f, historical_env_value).items() if str(k).startswith("OVERRIDE_")}
        out[str(st["sha256"])] = OverridesSource(os.path.abspath(f), str(st["sha256"]), str(st["id"]),
                                                 {str(k): str(v) for k, v in dict(st["keys"]).items()}, old)   # type: ignore[arg-type]
    return out


def looks_polluted(value: object) -> bool:
    """Whether a stored value MAY carry what the old reader kept: a string that is not a fixed point of the reader's rule, or that begins
    with ``#`` (what the old reader made of ``KEY=   # comment``). A FLAG for the rows whose file is not given (listed, never corrected):
    the stored text alone cannot tell ``note #42`` read from ``'note #42'`` from ``note`` with a comment, which is why the correction reads
    the file."""
    from . import emit
    return isinstance(value, str) and (emit.env_value(value) != value or value.startswith("#"))


def _not_a_reading(stamp: Mapping[str, object], source: OverridesSource) -> Optional[str]:
    """Why ``stamp`` is NOT a reading of ``source`` (None when it is): its key SET is not the file's; or a value under some key, or its
    ``id``, is neither the fixed reader's reading of the file's line nor the old reader's -- named with the stored value and both readings.
    The per-key evidence every use of a source rests on: a value the file's digest and key set alone would pass, a later statement
    restating one key while keeping the file as provenance, is caught here."""
    keys = stamp.get("keys")
    if not isinstance(keys, Mapping):
        return "the stamp's keys is not an object"
    stored = {str(k): v for k, v in keys.items()}
    if set(stored) != set(source.keys):
        gone, extra = sorted(set(source.keys) - set(stored)), sorted(set(stored) - set(source.keys))
        return (f"the stored key set is not the file's (the file has {gone or 'nothing'} more, the stamp {extra or 'nothing'} more): "
                f"not a reading of {source.name} ({source.sha256[:12]}); refused by name, nothing restated")
    odd = [f"{k} is {v!r}, neither the file's reading ({source.keys[k]!r}) nor what the old reader made of its line ({source.historical_keys.get(k, source.keys[k])!r})"
           for k, v in sorted(stored.items()) if v not in source.readings(k)]
    ident, (want, want_old) = str(stamp.get("id") or ""), source.ids_for(stamp)
    if ident not in (want, want_old):
        odd.append(f"id is {ident!r}, neither the file's ({want!r}) nor the old reader's ({want_old!r})")
    if odd:
        return ("; ".join(odd) + f": a later statement, not the reader's artifact; not a reading of {source.name} ({source.sha256[:12]}); "
                "refused by name, nothing restated")
    return None


def same_overrides_stamp(a: object, b: object, sources: Optional[Mapping[str, OverridesSource]] = None) -> bool:
    """Whether two ``run_overrides`` stamps that are not identical are statements of ONE file: the same non-empty ``sha256``, the same
    ``path``, that file among ``sources``, and EACH of the two a reading of it (:func:`_not_a_reading`: the key set the file's, every value
    one of the file's two readings, the ``id`` too). What run-patch-extra compares a legacy stamp by, with env-comments' sources: a lost
    stamp with comment text and a live stamp restated from the file are one file; a live value that is neither reading is not, and stays
    a conflict for the operator; without the file nothing but identity is equality."""
    if not (isinstance(a, Mapping) and isinstance(b, Mapping)):
        return False
    sha = a.get("sha256")
    if not isinstance(sha, str) or not sha or sha != b.get("sha256") or a.get("path") != b.get("path"):
        return False
    source = (sources or {}).get(sha)
    return source is not None and _not_a_reading(a, source) is None and _not_a_reading(b, source) is None


def source_restatement(stamp: Mapping[str, object], source: OverridesSource) -> Tuple[Dict[str, object], Dict[str, object], Optional[str]]:
    """``(the stamp restated from its file, {what changed: the value replaced}, a refusal or None)``. ``keys`` becomes the file's reading
    and ``id`` the file's ``OVERRIDE_ID`` (else the stem of the stored ``path``, as the writer derives it); ``path`` and ``sha256`` stand
    as stored. The replaced map is keyed by the key names, plus :data:`_STAMP_ID` when the ``id`` changes -- and every value it names IS
    the old reader's reading of the file's line, established per key. Refused by name (:func:`_not_a_reading`) when the stored key SET is
    not the file's, or any stored value (or the ``id``) is neither reading: a later statement is not the reader's artifact, and nothing of
    the stamp is restated."""
    refusal = _not_a_reading(stamp, source)
    if refusal:
        return dict(stamp), {}, refusal
    stored = {str(k): v for k, v in dict(stamp["keys"]).items()}       # type: ignore[arg-type]
    replaced: Dict[str, object] = {k: v for k, v in stored.items() if v != source.keys[k]}
    out = dict(stamp)
    out["keys"] = dict(source.keys)
    want_id, _old = source.ids_for(stamp)
    if str(stamp.get("id") or "") != want_id:
        replaced[_STAMP_ID] = stamp.get("id")
        out["id"] = want_id
    return out, replaced, None


@dataclass
class OverridesCorrection:
    """One stamped run row the pass lists: its live stamp, the file it matched (or none), what the file's reading changes, and what
    ``apply`` did. Three faces: SELECTED (a file, a difference), UNCLASSIFIED (a file, but the stamp is no reading of it: refused by name),
    AMBIGUOUS (no file, and values that may carry a comment: listed, never corrected)."""
    run_id: str
    campaign: str
    stamp: Dict[str, object]
    restated: Dict[str, object]                                             # the stamp restated from its file (the live stamp when there is no file)
    replaced: Dict[str, object]                                             # {key, or `id`: the value replaced}
    source: Optional[OverridesSource] = None                                # the digest-matched file; None: unsourced
    suspicious: Tuple[str, ...] = ()                                        # unsourced: the keys (or `id`) whose stored value may carry a comment
    unclassified: Optional[str] = None                                      # sourced, refused by name: the stamp is no reading of its file
    pending: Optional[str] = None                                           # sha of an unfolded record stating the stamp (the newest)
    pending_own: bool = False
    pending_type: str = ""
    written: Optional[str] = None
    refused: Optional[str] = None

    @property
    def selected(self) -> bool:
        return self.source is not None and self.unclassified is None and bool(self.replaced)

    @property
    def ambiguous(self) -> bool:
        return self.source is None and bool(self.suspicious)

    @property
    def unsourced_reason(self) -> str:
        """Why an ambiguous row fails an apply (and is listed by a verify) by name: no file, values that may carry a comment, and what
        would let the pass decide -- the file, or the operator's word."""
        return (f"no overrides file among the sources has this stamp's sha256 ({self.sha256[:12]}; stored path {str(self.stamp.get('path') or '')!r}) "
                f"and its values may carry a comment ({', '.join(self.suspicious)}): the stored text is never evidence, so nothing is restated; "
                "name the file with --source, or leave the stamp as stored by name with --leave-unsourced")

    @property
    def overrides_id(self) -> str:
        return str(self.stamp.get("id") or "")

    @property
    def sha256(self) -> str:
        return str(self.stamp.get("sha256") or "")

    @property
    def group(self) -> Tuple[str, str]:
        return (self.overrides_id, self.sha256[:8])

    def hold(self, statement: "ScopeStatement") -> None:
        self.pending, self.pending_own, self.pending_type = statement.sha, statement.own, statement.record_type

    @property
    def hold_reason(self) -> str:
        what = ("this pass's own correction record" if self.pending_own else "a full run record" if self.pending_type == "run" else "a record")
        return f"{what} of this run stating its run_overrides is in the log and not folded yet ({(self.pending or '')[:12]}); ingest first, then re-run"


@dataclass
class OverridesReport:
    root: str
    runs: List[OverridesCorrection]                                         # every listed row: selected, unclassified, ambiguous
    applied: bool
    verify: bool = False
    row_tables: Dict[str, Optional[int]] = field(default_factory=dict)    # the other tables' rows the pass would list, counted, not corrected (None = not counted)
    sources: List[OverridesSource] = field(default_factory=list)            # the files offered, by digest
    stamped: int = 0                                                        # every run row carrying a stamp
    sourced: int = 0                                                        # ... whose file is among the sources
    leave_unsourced: bool = False                                           # the operator's word: the ambiguous rows are left as stored, not a failure
    check_sources: bool = False                                             # the coverage alone (`--check-sources`): `runs` are the ambiguous rows

    @property
    def selected(self) -> List[OverridesCorrection]:
        """The rows the apply acts on (the shared command reads this on every correction report)."""
        return [r for r in self.runs if r.selected]

    @property
    def unclassified(self) -> List[OverridesCorrection]:
        return [r for r in self.runs if r.unclassified]

    @property
    def ambiguous(self) -> List[OverridesCorrection]:
        return [r for r in self.runs if r.ambiguous]

    @property
    def unsourced_failing(self) -> List[OverridesCorrection]:
        """The ambiguous rows that fail an apply and a verify: every one, unless the operator left them by name (``leave_unsourced``)."""
        return [] if self.leave_unsourced else self.ambiguous

    @property
    def listed(self) -> List[OverridesCorrection]:
        """The rows a verify (or the coverage check) lists, exit 3 when any: the selected rows, the unclassified ones -- a stamp the pass
        refuses is still a row the folded table carries, and the landing window must fail by name rather than read a clean class -- and
        the ambiguous ones, unless left by name: without the file there is nothing to decide, and a class the pass could not decide is
        not a clean class (the section comment). With ``leave_unsourced`` the ambiguous rows are reported beside the list and are not a
        failure."""
        return [r for r in self.runs if r.selected or r.unclassified or (r.ambiguous and not self.leave_unsourced)]

    def by_group(self) -> Dict[Tuple[str, str], int]:
        out: Dict[Tuple[str, str], int] = {}
        for r in self.selected:
            out[r.group] = out.get(r.group, 0) + 1
        return dict(sorted(out.items()))

    def by_key(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.selected:
            for k in r.replaced:
                out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))

    def ambiguous_by_group(self) -> Dict[Tuple[str, str], Tuple[int, Tuple[str, ...]]]:
        """Per (overrides id, sha prefix) of the unsourced rows: (runs, the union of their suspicious keys)."""
        out: Dict[Tuple[str, str], Tuple[int, Tuple[str, ...]]] = {}
        for r in self.ambiguous:
            n, ks = out.get(r.group, (0, ()))
            out[r.group] = (n + 1, tuple(sorted(set(ks) | set(r.suspicious))))
        return dict(sorted(out.items()))

    @property
    def written(self) -> List[str]:
        return [r.written for r in self.runs if r.written]

    @property
    def held(self) -> List[OverridesCorrection]:
        return [r for r in self.runs if r.refused]

    def as_dict(self) -> Dict[str, object]:
        return {"root": self.root, "action": ENV_COMMENTS_ACTION, "applied": self.applied, "verify": self.verify, "check_sources": self.check_sources,
                "leave_unsourced": self.leave_unsourced, "runs": len(self.selected), "listed": len(self.listed),
                "stamped_runs": self.stamped, "sourced_runs": self.sourced, "unclassified": len(self.unclassified), "ambiguous_runs": len(self.ambiguous),
                "unsourced_failing": len(self.unsourced_failing),
                "sources": [{"path": s.path, "sha256": s.sha256, "id": s.id} for s in self.sources],
                "by_overrides_file": [{"id": i, "sha256_prefix": s, "runs": n} for (i, s), n in self.by_group().items()],
                "by_key": self.by_key(),
                "ambiguous": [{"id": i, "sha256_prefix": s, "runs": n, "suspicious_keys": list(ks)} for (i, s), (n, ks) in self.ambiguous_by_group().items()],
                "row_tables_not_corrected": dict(self.row_tables),
                "written": len(self.written), "held": len(self.held),
                "corrections": [{"run_id": r.run_id, "campaign": r.campaign, "overrides_id": r.overrides_id, "sha256": r.sha256,
                                 "source": r.source.path if r.source else None, "selected": r.selected, "unclassified": r.unclassified,
                                 "ambiguous": r.ambiguous, "suspicious_keys": list(r.suspicious),
                                 "replaced": dict(r.replaced), "restated_keys": dict(r.restated.get("keys") or {}),      # type: ignore[arg-type]
                                 "pending": r.pending, "pending_own": r.pending_own, "pending_type": r.pending_type or None,
                                 "written": r.written, "refused": r.refused} for r in self.runs]}


def _overrides_of(extra_json) -> Optional[Dict[str, object]]:
    """The ``run_overrides`` stamp of a row's ``extra_json`` (text or object), when it is an object with a ``keys`` object; else None."""
    extra = _extra_json_object(extra_json)
    stamp = extra.get(OVERRIDES_EXTRA_KEY) if extra else None
    return dict(stamp) if isinstance(stamp, Mapping) and isinstance(stamp.get("keys"), Mapping) else None


def _suspicious_keys(stamp: Mapping[str, object]) -> Tuple[str, ...]:
    keys = stamp.get("keys") if isinstance(stamp.get("keys"), Mapping) else {}
    out = sorted(str(k) for k, v in keys.items() if looks_polluted(v))     # type: ignore[union-attr]
    if looks_polluted(stamp.get("id")):
        out.append(_STAMP_ID)
    return tuple(out)


def classify_stamp(run_id: str, campaign: str, stamp: Dict[str, object], sources: Mapping[str, OverridesSource]) -> Optional[OverridesCorrection]:
    """One stamp against the sources (the pure rule): the row the pass lists, or None when there is nothing to list (the stamp is its
    file's reading already, or no file is given and nothing looks polluted)."""
    source = sources.get(str(stamp.get("sha256") or ""))
    if source is None:
        suspicious = _suspicious_keys(stamp)
        return OverridesCorrection(run_id, campaign, stamp, dict(stamp), {}, None, suspicious) if suspicious else None
    restated, replaced, refusal = source_restatement(stamp, source)
    if refusal:
        return OverridesCorrection(run_id, campaign, stamp, dict(stamp), {}, source, (), refusal)
    return OverridesCorrection(run_id, campaign, stamp, restated, replaced, source) if replaced else None


def overrides_class_rows(conn, sources: Mapping[str, OverridesSource]) -> Tuple[List[OverridesCorrection], int, int]:
    """``(the listed rows in run-id order, every stamped run row, the stamped rows whose file is among the sources)`` over the ``run``
    table, read-only."""
    rows = conn.execute("SELECT run_id, campaign, extra_json FROM run WHERE json_extract(extra_json, '$.run_overrides.keys') IS NOT NULL "
                        "ORDER BY run_id").fetchall()
    out: List[OverridesCorrection] = []
    stamped = sourced = 0
    for run_id, campaign, extra_json in rows:
        stamp = _overrides_of(extra_json)
        if stamp is None:
            continue
        stamped += 1
        if str(stamp.get("sha256") or "") in sources:
            sourced += 1
        row = classify_stamp(str(run_id), str(campaign), stamp, sources)
        if row is not None:
            out.append(row)
    return out, stamped, sourced


def overrides_row_table_counts(conn, sources: Mapping[str, OverridesSource]) -> Dict[str, Optional[int]]:
    """Per other table, how many rows carry a stamp the pass would list (full statements: counted, never corrected by this pass); the
    item-scale tables are not scanned and read None."""
    out: Dict[str, Optional[int]] = {}
    for table in OVERRIDES_ROW_TABLES:
        n = 0
        for (extra_json,) in conn.execute(f"SELECT extra_json FROM {table} WHERE json_extract(extra_json, '$.run_overrides.keys') IS NOT NULL"):
            stamp = _overrides_of(extra_json)
            if stamp is not None and classify_stamp("", "", stamp, sources) is not None:
                n += 1
        out[table] = n
    for table in OVERRIDES_ITEM_TABLES:
        out[table] = None
    return out


def _overrides_statements_in(path: str, run_ids: Optional[Set[str]] = None) -> List[ScopeStatement]:
    """The records of one file that STATE a run's ``run_overrides`` at the fold: a full ``run`` record carrying it under ``extra``, a
    ``run_patch`` carrying it under ``fields.extra_json`` (this pass's own, by its marker); a legacy patch's top-level ``extra`` never folds
    the key and is passed over. ``run_ids`` narrows to those runs."""
    out: List[ScopeStatement] = []
    for _i, rec, _sha in R.read_record_file(path, compute_sha=False):
        record_type = rec.get("record_type")
        if record_type not in SCOPE_STATEMENT_RECORD_TYPES:
            continue
        payload = rec.get("payload") if isinstance(rec.get("payload"), Mapping) else None
        run_id = payload.get("run_id") if payload else None
        if not isinstance(run_id, str) or not run_id or (run_ids is not None and run_id not in run_ids):
            continue
        own = False
        if record_type == "run":
            extra = payload.get("extra") if isinstance(payload.get("extra"), Mapping) else {}       # type: ignore[union-attr]
            states = OVERRIDES_EXTRA_KEY in extra
        else:
            fields = payload.get("fields") if isinstance(payload.get("fields"), Mapping) else {}   # type: ignore[union-attr]
            ej = _extra_json_object(fields.get("extra_json")) or {}
            states = OVERRIDES_EXTRA_KEY in ej
            own = ENV_COMMENTS_RECONCILED_KEY in ej
        if states or own:
            out.append(ScopeStatement(run_id, R.record_sha(rec), path, str(rec.get("written_utc", "")), (OVERRIDES_EXTRA_KEY,) if states else (),
                                      own, str(record_type)))
    return out


def overrides_statements(records_root: str, run_ids: Optional[Set[str]] = None) -> Dict[str, List[ScopeStatement]]:
    """Every statement of a run's stamp in the WHOLE log (:func:`run_statement_files`), by run id, in ``(written_utc, sha)`` order."""
    out: Dict[str, List[ScopeStatement]] = {}
    for path in run_statement_files(records_root):
        for s in _overrides_statements_in(path, run_ids):
            out.setdefault(s.run_id, []).append(s)
    for recs in out.values():
        recs.sort(key=lambda s: s.order_key)
    return out


def overrides_statements_in_directory(directory: str, run_id: str) -> List[ScopeStatement]:
    """The stamp statements of one run under one directory (the run's own, read again at the write)."""
    if not os.path.isdir(directory):
        return []
    out: List[ScopeStatement] = []
    for fn in sorted(os.listdir(directory)):
        if _RUN_STATEMENT_FILE.search(fn):
            out.extend(_overrides_statements_in(os.path.join(directory, fn), {run_id}))
    out.sort(key=lambda s: s.order_key)
    return out


def overrides_correction_payload(run: OverridesCorrection) -> Dict[str, object]:
    """The record: the WHOLE restated stamp (the fold keeps a key's latest value whole), the marker naming every replaced key with the value
    it replaced and the file it was read from (its name and digest), and the confidence entry of the one column the record restates. No
    central stamp."""
    assert run.source is not None
    marker = {"replaced_keys": {k: v for k, v in run.replaced.items() if k != _STAMP_ID}, "source_file": run.source.name,
              "source_sha256": run.sha256, "pass": ENV_COMMENTS_PRODUCER_SCRIPT}
    if _STAMP_ID in run.replaced:
        marker["replaced_id"] = run.replaced[_STAMP_ID]
    entry = {"confidence": "recorded",
             "source": (f"{ENV_COMMENTS_PRODUCER_SCRIPT}: run_overrides restated from the overrides file of record ({run.source.name}, sha256 "
                        f"{run.sha256[:12]}), read as the shell reads it; the file's sha256 and path and every other key of extra_json stand as "
                        "their writers stated them")}
    return {"run_id": run.run_id, "fields": {"extra_json": {OVERRIDES_EXTRA_KEY: dict(run.restated), ENV_COMMENTS_RECONCILED_KEY: marker},
                                             "field_confidence_json": {"extra_json": entry}}}


def _overrides_writer_for(root: str, run: OverridesCorrection, now: Optional[str]):
    """A ``RecordWriter`` for the run's own directory that stamps NOTHING of its own: an operator's ``$SISTEER_RUN_OVERRIDES`` would
    otherwise be stamped over the very value this record restates."""
    from . import emit
    producer = emit.producer_block(ENV_COMMENTS_PRODUCER_SCRIPT, code_sha=emit.git_sha(os.path.dirname(os.path.abspath(__file__))))
    w = emit.RecordWriter(os.path.join(root, "records"), run.campaign, run.run_id, producer, disabled=False, payload_extra={})
    w.payload_extra = {}
    return w


def verify_env_comments(root: str, conn=None, sources: Sequence[str] = (), leave_unsourced: bool = False) -> OverridesReport:
    """The predicate alone over the FOLDED ``run`` table, against the same sources: every row whose stamp still differs from its file's
    reading, by name, the unclassified rows with it, and the ambiguous rows -- a failure too, unless ``leave_unsourced`` leaves them by
    name. Read-only, no walk, no hold."""
    srcs = overrides_sources(sources)
    own_conn = conn is None
    if own_conn:
        conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        runs, stamped, sourced = overrides_class_rows(conn, srcs)
    finally:
        if own_conn:
            conn.close()
    return OverridesReport(root, runs, applied=False, verify=True, sources=sorted(srcs.values(), key=lambda s: s.path), stamped=stamped, sourced=sourced,
                           leave_unsourced=leave_unsourced)


def check_env_comments_sources(root: str, conn=None, sources: Sequence[str] = (), leave_unsourced: bool = False) -> OverridesReport:
    """The coverage of the sources alone (``--check-sources``): over the folded ``run`` table, read-only, without the walk or the other
    tables, every stamped row whose values may carry a comment and whose file is NOT among the sources -- the ambiguous rows -- by name,
    with the counts (stamped, sourced). ``listed`` is those rows (exit 3 when any), or nothing when ``leave_unsourced`` leaves them by
    name. What the landing window runs before anything moves when env-comments is listed for apply."""
    srcs = overrides_sources(sources)
    own_conn = conn is None
    if own_conn:
        conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        runs, stamped, sourced = overrides_class_rows(conn, srcs)
    finally:
        if own_conn:
            conn.close()
    return OverridesReport(root, [r for r in runs if r.ambiguous], applied=False, check_sources=True, sources=sorted(srcs.values(), key=lambda s: s.path),
                           stamped=stamped, sourced=sourced, leave_unsourced=leave_unsourced)


def reconcile_env_comments(root: str, apply: bool = False, conn=None, now: Optional[str] = None,
                           writer_factory: Optional[Callable[[str, OverridesCorrection, Optional[str]], object]] = None,
                           sources: Sequence[str] = (), leave_unsourced: bool = False) -> OverridesReport:
    """The pass (the section comment): the listed ``run`` rows against the sources with the other tables' counts; with ``apply`` ONE
    ``run_patch`` per selected run that is not held, into the run's own directory, the directory read once more at the write; an
    unclassified run is refused by name and never written; an ambiguous run is listed, never written, and REFUSED by name too (the apply
    does not complete past a stamp it could not decide) unless ``leave_unsourced`` leaves it as stored by name. Never writes the
    database."""
    from . import identity as I
    srcs = overrides_sources(sources)
    own_conn = conn is None
    if own_conn:
        conn = DB.connect(DB.db_path(root), readonly=True)
    runs: List[OverridesCorrection] = []
    accepted_by_run: Dict[str, Set[str]] = {}
    try:
        runs, stamped, sourced = overrides_class_rows(conn, srcs)
        row_tables = overrides_row_table_counts(conn, srcs)
        selected = [r for r in runs if r.selected]
        statements = overrides_statements(os.path.join(root, "records"), {r.run_id for r in selected}) if selected else {}
        for run in selected:
            accepted_by_run[run.run_id] = accepted_run_statement_shas(conn, run.run_id)
            unfolded = unfolded_statements(conn, statements.get(run.run_id, []), accepted_by_run[run.run_id])
            if unfolded:
                run.hold(unfolded[-1])
        if apply:
            for run in runs:
                if run.unclassified:
                    run.refused = run.unclassified
                    continue
                if run.ambiguous and not leave_unsourced:
                    run.refused = run.unsourced_reason
                    continue
                if not run.selected:
                    continue
                if run.pending:
                    run.refused = run.hold_reason
                    continue
                directory = os.path.join(root, "records", run.campaign, I.run_dir_name(run.run_id))
                try:
                    late = unfolded_statements(conn, overrides_statements_in_directory(directory, run.run_id), accepted_by_run[run.run_id])
                    if late:
                        run.hold(late[-1])
                        run.refused = run.hold_reason
                        continue
                    w = (writer_factory or _overrides_writer_for)(root, run, now)
                    run.written = w.write("run_patch", overrides_correction_payload(run))
                except (ValueError, OSError) as exc:
                    run.refused = str(exc)
    finally:
        if own_conn:
            conn.close()
    return OverridesReport(root, runs, apply, row_tables=row_tables, sources=sorted(srcs.values(), key=lambda s: s.path), stamped=stamped, sourced=sourced,
                           leave_unsourced=leave_unsourced)


def _replaced_text(run: OverridesCorrection) -> str:
    keys = run.restated.get("keys") if isinstance(run.restated.get("keys"), Mapping) else {}
    parts = [f"{k}={run.replaced[k]!r} -> {(run.restated.get('id') if k == _STAMP_ID else keys.get(k))!r}" for k in sorted(run.replaced)]   # type: ignore[union-attr]
    return ", ".join(parts)


def _unsourced_line(r: OverridesCorrection) -> str:
    """One listed unsourced row in a verify or the coverage check: the id first, as every listed line (the landing window reads the ids
    from these lines), then the stamp and the keys."""
    return f"  {r.run_id} unsourced: no file given for this stamp ({r.overrides_id} {r.sha256[:8]}); values that may carry a comment: {', '.join(r.suspicious)}"


def _left_by_name(rep: OverridesReport) -> str:
    """The clause a count line carries when the operator left the ambiguous rows as stored (``--leave-unsourced``)."""
    return f"; {len(rep.ambiguous)} unsourced run(s) left as stored by name (--leave-unsourced)" if rep.leave_unsourced and rep.ambiguous else ""


def format_overrides_report(rep: OverridesReport) -> str:
    """The dry run's text: the sources, one line per overrides file (id, sha prefix, runs) of the selected class, the keys with their
    counts, the ambiguous groups (no file given), the unclassified runs by name, the other tables' counts (not corrected), the pending /
    written / held lines, the total last (the landing window shows the tail). A verify renders its own shape: one line per listed run (the
    id first, then what changes; an unsourced row with its stamp and keys), the ambiguous groups unindented beside them, the count last --
    an unsourced row counted among the polluted unless left by name. The coverage check (``--check-sources``) renders the sources, the
    counts, the uncovered groups and their rows, and the uncovered count last."""
    if rep.check_sources:
        lines = ["env-comments sources (read-only, the folded run table): every stamped run row whose values may carry a comment has its overrides "
                 "file among the sources"]
        lines.append(f"sources: {len(rep.sources)} file(s) by digest" + "".join(f"\n  {s.sha256[:8]} {s.id:<36} {s.path}" for s in rep.sources))
        lines.append(f"stamped run rows {rep.stamped}; with a source {rep.sourced}")
        for (ident, sha8), (n, ks) in rep.ambiguous_by_group().items():
            lines.append(f"uncovered (no file given for this stamp; values that may carry a comment: {', '.join(ks)}): {ident} {sha8} {n} run(s)")
        for r in rep.listed:
            lines.append(_unsourced_line(r))
        lines.append(f"uncovered {len(rep.listed)} run(s)" + ("; every stamp whose values may carry a comment has its file" if not rep.listed else
                                                              "; name their files with --source, or leave them as stored by name with --leave-unsourced")
                     + _left_by_name(rep))
        return "\n".join(lines)
    if rep.verify:
        lines = ["env-comments verify (read-only, the folded database): run rows whose run_overrides stamp differs from its overrides file's reading"]
        for r in rep.listed:
            lines.append(_unsourced_line(r) if r.ambiguous else f"  {r.run_id} " + (f"unclassified: {r.unclassified}" if r.unclassified else _replaced_text(r)))
        for (ident, sha8), (n, ks) in rep.ambiguous_by_group().items():
            lines.append(f"ambiguous (no file given for this stamp; values that may carry a comment: {', '.join(ks)}): {ident} {sha8} {n} run(s)"
                         + (" -- left as stored by name" if rep.leave_unsourced else " -- listed above: not a clean class until the file is given or --leave-unsourced"))
        lines.append(f"polluted {len(rep.listed)} run(s)" + (f" ({len(rep.unsourced_failing)} unsourced)" if rep.unsourced_failing else "")
                     + ("; the class is consistent" if not rep.listed else "") + _left_by_name(rep))
        return "\n".join(lines)
    lines = ["env-comments: run rows whose run_overrides stamp differs from the reading of its own overrides file (matched by the file's sha256; "
             "each restated from the file)"]
    lines.append(f"sources: {len(rep.sources)} file(s) by digest" + "".join(f"\n  {s.sha256[:8]} {s.id:<36} {s.path}" for s in rep.sources))
    lines.append(f"stamped run rows {rep.stamped}; with a source {rep.sourced}")
    for (ident, sha8), n in rep.by_group().items():
        lines.append(f"  {ident:<36} {sha8:<8} {n:>6}")
    lines.append("by key: " + (", ".join(f"{k} {n}" for k, n in rep.by_key().items()) or "none"))
    amb = rep.ambiguous_by_group()
    lines.append("ambiguous (no file given for the stamp and values that may carry a comment; listed, never corrected; "
                 + ("left as stored by name, --leave-unsourced" if rep.leave_unsourced else
                    "--apply refuses them and --verify lists them until the file is given or --leave-unsourced leaves them by name")
                 + f"): {len(rep.ambiguous)} run(s)"
                 + "".join(f"\n  {ident:<36} {sha8:<8} {n:>6}  keys: {', '.join(ks)}" for (ident, sha8), (n, ks) in amb.items()))
    if rep.unclassified:
        lines.append(f"unclassified (a file given, but the stamp is no reading of it; refused by name, never corrected): {len(rep.unclassified)} run(s)"
                     + "".join(f"\n  {r.run_id} {r.unclassified}" for r in rep.unclassified[:20]) + ("\n  ..." if len(rep.unclassified) > 20 else ""))
    lines.append("other tables (full statements, not corrected by this action): "
                 + ", ".join(f"{t} {'not counted' if n is None else n}" for t, n in rep.row_tables.items()))
    pending = [r for r in rep.selected if r.pending]
    if pending:
        lines.append(f"pending (a record stating the run's run_overrides is in the log, not folded yet): {len(pending)} run(s)" + "".join(
            f"\n  {r.run_id} record {r.pending[:12]}" + (" (this pass's own correction)" if r.pending_own else " (a full run record)" if r.pending_type == "run" else "")
            for r in pending[:20]) + ("\n  ..." if len(pending) > 20 else ""))
    if rep.applied:
        for r in rep.held:
            lines.append(f"  held {r.run_id}: {r.refused}")
        lines.append(f"wrote {len(rep.written)} correction record(s); held {len(rep.held)}")
    lines.append(f"runs {len(rep.selected)} in {len(rep.by_group())} (overrides id, file sha) group(s)" + ("" if rep.applied else "; nothing written (dry run)"))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# fit-effective-batch: an object row's fit_effective_batch restated from its own recorded hyperparameters where a sidecar hook stated the
# micro-batch
# --------------------------------------------------------------------------------------------
#
# The sidecar hook (`hooks.object_from_sidecar`) took `fit_effective_batch` from the trainer's `resolved_batch_size`, the per-step
# MICRO-batch, while copying the same sidecar's `effective_batch` -- the optimizer-step width, `batch_size x accum_steps`, the value every
# other writer of the column states (the fit-sidecars adapter, the trainers' own records, the published finals these rows descend from) --
# into `fit_hyperparameters_json`. Each such row contradicts itself: `fit_effective_batch = 1` beside `fit_hyperparameters_json.effective_batch
# = 64`. The hook is corrected by its own branch; THIS pass corrects the rows already folded, as records: one same-id `object` record per
# row, written by the registry owner into `records/registry_owner/object__<sha16>/`, carrying the row's required identity fields and the
# pairing columns its `fit_target_kind` requires (copied verbatim: the validator admits nothing less), `fit_effective_batch` = the row's own
# recorded `effective_batch`, the row's WHOLE `field_confidence_json` with the column's entry naming the source (the object table has no
# per-column confidence merge: a statement replaces the JSON column wholesale, so the record carries every standing entry), and a marker
# under `extra` (`fit_effective_batch_correction`: the replaced value, what the recorded block says, the pass). `status` is OMITTED (the
# producer's statement; the fold keeps the folded value), `fit_hyperparameters_json` and every other column too (per-column latest
# statement: an omitted column keeps its folded value and stamp). The pass reads the live database read-only, never a sidecar, never a
# parent, and never writes the database.
#
# THE CLASS (clearly wrong): `fit_effective_batch` set and different from the row's own `fit_hyperparameters_json.effective_batch`, AND equal
# to that block's `resolved_batch_size` (the hook's signature), AND the block's `batch_size x accum_steps = effective_batch` (the trainer's
# identity holds in the row's own record). A function of the folded row alone, so the correction is replay-equal and idempotent: after the
# fold the two values agree and the row leaves the class. Rows where the two values disagree but the signature does NOT hold are a new,
# unclassified class: listed by name and REFUSED, never corrected. The signature is read over the row's values IN THEIR OWN TYPE: each of
# the five batch values (the column and the block's effective_batch, resolved_batch_size, batch_size, accum_steps) must be an integer (not
# a boolean) for the row to be clearly wrong; a float, a numeric string, a boolean or an unstated block key makes the row unclassified,
# named with the value and its type, and nothing is coerced. Rejected alternative: coercing with int(): a recorded 64.5 truncated to 64
# beside a micro-batch 1 satisfies the signature and would be WRITTEN as a correction to 64, a value the record never stated; and a
# non-numeric recorded value ("auto") coerced to nothing would drop the row from the report, so --apply and --verify would read a clean
# class past an unclassified row. Every row the SQL selects (the two values disagree) is therefore kept visible, classed or refused by
# name, and the verify exits 3 while one stands. AMBIGUOUS, listed and not corrected: rows whose `fit_effective_batch` is
# NULL while the recorded block carries `effective_batch`. These are dose-scaled copies: the fit-sidecars adapter used to leave every
# fit_* scalar unstated for one ("nothing was fit") while stating the PARENT's hyperparameter block beside it. The rule for them:
# the adapter STATES a dose copy's `fit_effective_batch` as its parent's effective batch, `derived` from ONE
# source -- the parent's fit record copied beside the copy, the copy's own `train_budget_meta.json`, the record the block itself is read
# from (`fit_sidecars._dose_batch_from_parent`, objects_bank EMISSION_VERSION 10; by design the adapter reads
# the copy's own files alone, never another group's directory) -- and the bank's own re-emission restates the rows: NO correction record
# is written for them by this pass, and a dose copy whose stated value equals its recorded block is consistent (neither class). What this
# pass still lists as ambiguous is therefore one of two things, told apart by the row's OWN confidence entry for the column and reported
# beside the group: a row with NO entry (the adapter's earlier emission, awaiting its re-emission at the bumped version) or a row the
# adapter left NULL BY NAME (an `unverified` entry: the copied record names no integer effective batch -- the key absent or null, a float,
# a numeric string, a boolean -- or the copy carries no `train_budget_meta.json`; the reason is in the row's `field_confidence_json` and
# `extra_json.fit_effective_batch_unresolved`). A row left NULL by name is selected by that ENTRY, whatever its block records: when the
# copied record names no `effective_batch` (the key absent or null) or the copy carries no record at all, the folded block has no
# `effective_batch` to read either -- the key absent, the key null, or no `fit_hyperparameters_json` -- and a selection on the recorded
# value alone dropped exactly these rows from the text and the JSON while the row kept the adapter's explicit NULL and its reason.
# So a dose copy (`derivation = 'dose_scale'`) whose column is NULL beside ANY confidence entry for it is in the
# class, its group's recorded value shown as `-`. Outside the class, by design: a dose copy with NULL, no entry and no recorded value
# (nothing stated and nothing recorded, so there is nothing to read; the adapter's re-emission brings it in by its entry), and a row of
# another derivation with an `unverified` entry beside NULL and nothing recorded (a subtraction anchor whose parents' records could not be
# reached: its own writer's reading, not this pass's question). EVERY ambiguous row is listed by OBJECT ID under its group with the
# row's own reason, in the text and the JSON alike (`BatchAmbiguous.members`): a group's count told an operator how many copies of one (writer, recorded value,
# model) bucket were left NULL, never WHICH, so with several copies in one bucket the report could not say which copied record to repair;
# the aggregate stands beside the names. Neither is corrected here. And a disagreeing row whose column's OWN confidence entry is `derived`
# or `unverified` is NOT the hook's statement, so the hook's signature cannot apply: it is unclassified, refused by name, never written
# (`batch_derived_reason`) -- recognised by the ENTRY's confidence, never by its wording or its writer (the sidecar hook's dose path states
# the same entry for the copies it files). `derived` is the adapter's statement from the copied record (it states the value only when it
# is the block's own, so such a row means another writer moved the block after the adapter's statement); `unverified` beside a value is
# the adapter's NULL-by-name statement with a value still standing -- a retained earlier statement (the adapter states the column as null,
# so this shape should not fold; a record that omitted the column instead of stating null left exactly it: a derived 8 kept beside a
# block moved to 64, 8 = the block's resolved_batch_size and 8 x 8 = 64, the hook's signature to the letter), never the hook's reading,
# which carries no `unverified` entry for an integer it wrote. Reported as nothing to compare: rows with the column set and no recorded
# `effective_batch`.
#
# THE PARENT CROSS-CHECK (read-only, the registry's own agreement property): the adapter's one source is the record COPIED beside the
# copy, so whether that copy is a faithful copy of the parent the stamp names is a question the REGISTRY answers, from two rows it
# already holds. The dry run and the verify compare every dose copy (`derivation = 'dose_scale'`) whose `fit_effective_batch` is a
# `derived` statement with its PARENT object row -- the parent by the copy's `parent_object_ids_json` (its one id), else
# `parent_object_id`; the parent's recorded effective batch its own `fit_hyperparameters_json.effective_batch`, else its
# `fit_effective_batch` -- and list by object id every copy whose derived value is not the parent's (`BatchParentCheck`). A disagreement
# is not the hook's signature and not the adapter's to resolve: `--verify` exits 3 while one stands, `--apply` never writes for it (the
# operator reads both records), and the dry run names it beside the counts. A parent row ABSENT from the registry is NOT a disagreement
# (nothing to compare; the report says so by name), nor is a parent present that states no effective batch, nor a copy that names no
# parent. A copy left NULL has no derived value and is in the ambiguous class instead. The two values are read IN THEIR OWN TYPE, as the
# adapter reads the copied record and the hook's signature reads the row: the parent's recorded value (and the copy's derived one) must
# be an integer that is not a boolean for the pair to agree, and a parent recording 8.0, true or "8" beside a copy deriving 8 (or 1) is
# `BATCH_PARENT_NONINTEGRAL` -- listed as a disagreement with the value and its type, exit 3 on --verify, never written for. `==` alone
# read it as agreement (Python's 8.0 == 8 and True == 1 hold), so a parent record whose batch had changed type passed the dry run unnamed
# and --verify exited 0 past it -- the very values the adapter refuses by name. Nothing is coerced.
#
# THE STATEMENT CHECK (read-only, the copy against its OWN row): the adapter's `derived` entry claims the column IS the block's own
# `effective_batch` (one record, read twice), and its `unverified` entry claims the column is NULL. Neither claim is the class SQL's
# question -- that selects a row whose two values DISAGREE, by SQLite's equality, which reads 8 and 8.0 as equal and a missing key as
# nothing to compare -- so a dose copy whose block another object writer moved to 8.0 beside a retained derived 8, or whose block lost the
# key, or whose `unverified` entry stands beside a value its block happens to agree with, was never selected: `batch_derived_reason` never
# saw it, the parent cross-check could still read `agrees` (the parent recording an integer 8), and --verify exited 0 past a copy claiming
# derivation from a record the adapter itself would refuse by name. These are reachable fold states (the
# per-column fold keeps the column's latest statement beside a block a later writer restated without the column), so the pass reads every
# dose copy (`derivation = 'dose_scale'`) whose column carries a `derived` entry, and every one whose `unverified` entry has a value standing
# beside it, in PYTHON and in their own type (`BatchStatementCheck`): a derived copy is consistent only when its column and its block's
# `effective_batch` are one integer that is not a boolean, the same on both sides; otherwise it is named with WHICH side fails -- the column
# NULL beside the derived entry, the column not an integer, the block recording no `effective_batch` (the key absent or null, or no block),
# the block's value not an integer (8.0), or another integer -- and a value standing beside the NULL-by-name entry is named as such, whatever
# its block says. An inconsistent copy is listed by object id in the dry run and the verify (exit 3 while one stands) and never a correction
# candidate (`--apply` writes nothing for it: the pass corrects the hook's reading alone, and the adapter's re-emission is the correction).
# A copy the class SQL also selects (a derived 1 beside a block saying 64) is one row of the table, listed once by the verify. A copy whose
# `unverified` entry stands beside NULL is the adapter's statement to the letter: not read here (it is in the ambiguous class by name), and
# a copy with no entry for the column is not the adapter's statement at all. A derived dose copy whose block records no `effective_batch`
# is therefore NOT counted as nothing to compare: the statement check names it.
#
# THE HOLD, as the other corrections': a selected object with an `object` record anywhere in the log (every `*-object.jsonl` / `.gz`, the
# adapters' bundles included) that the fold has neither accepted (`row_history`, the object's `object` events) nor refused -- this pass's
# own earlier record, a native re-emission, an adapter revision -- is held: the dry run reports it, the apply writes nothing for it (exit
# 2), and the target directory is read once more at the write. The verify (`--verify`) is the predicate alone over the folded `object`
# table, read-only, exit 3 when rows remain; the landing window runs apply -> ingest -> verify (an object has no endpoint-patch race, so
# one verify is expected to read zero; the second pass stands for the case it does not).

BATCH_ACTION = "fit-effective-batch"
BATCH_PRODUCER_SCRIPT = "registry reconcile fit-effective-batch"
BATCH_MARKER_KEY = "fit_effective_batch_correction"
#: the owner's record directory campaign: records/registry_owner/object__<sha16>/ (the precedent of the owner's earlier object corrections)
BATCH_OWNER_CAMPAIGN = "registry_owner"
_OBJECT_FILE = re.compile(r"-object\.jsonl(\.gz)?$")
#: the hyperparameter keys the marker records beside the replaced value
_BATCH_HP_KEYS: Tuple[str, ...] = ("effective_batch", "batch_size", "accum_steps", "resolved_batch_size")
#: the object columns stored as JSON text that the record carries parsed, so the fold canonicalizes them again as the original did
_OBJECT_JSON_COLUMNS: Tuple[str, ...] = ("fit_layers_json", "parent_object_ids_json", "fit_contrast_sets_json", "field_confidence_json")

#: the selection, over the folded object table: every row whose column disagrees with its own recorded effective_batch. The class is then
#: read in Python from the row's parsed block (its values in their own type: SQL's json_extract reads a JSON `true` as 1), never coerced.
BATCH_CLASS_SQL = ("SELECT object_id, object_kind, fit_target_kind, derivation, campaign, status, named_role, base_scale, fit_writer, fit_model_id, "
                   "record_uri, fit_effective_batch, fit_hyperparameters_json, field_confidence_json "
                   "FROM object WHERE fit_effective_batch IS NOT NULL AND json_extract(fit_hyperparameters_json, '$.effective_batch') IS NOT NULL "
                   "AND fit_effective_batch <> json_extract(fit_hyperparameters_json, '$.effective_batch') ORDER BY object_id")
#: the ambiguous rows, ONE PER ROW (the object id and the row's own reason for the NULL -- the column's confidence entry's `reason`, else its
#: `source`; NULL for a row with no entry), in the order of the groups the report shows them under: (writer, recorded value, model) AND what
#: the row's own confidence entry says of the column: NULL (no entry: the adapter's pre-ruling emission, awaiting its re-emission) or the
#: entry's confidence (`unverified`: the adapter left it NULL by name) -- the rule of 2026-10-13 makes the two different things to
#: read. Two ways in: the column NULL beside a recorded `effective_batch` (any row), or a dose copy's column NULL beside a confidence
#: entry FOR it, whatever its block records (the adapter's NULL by name when the copied record names no `effective_batch` or the copy
#: carries none leaves the block without the key, with it null, or absent altogether -- `json_extract` reads all three as NULL -- and the
#: recorded value alone dropped such a row from the report; `recorded` is then NULL and the report shows `-`). Grouped in Python
#: (:func:`batch_ambiguous_groups`) so every row stays named: a GROUP BY count could not say which rows.
BATCH_AMBIGUOUS_SQL = ("SELECT object_id, fit_writer, json_extract(fit_hyperparameters_json, '$.effective_batch') AS recorded, fit_model_id, "
                       "json_extract(field_confidence_json, '$.fit_effective_batch.confidence') AS stated, "
                       "COALESCE(json_extract(field_confidence_json, '$.fit_effective_batch.reason'), "
                       "json_extract(field_confidence_json, '$.fit_effective_batch.source')) AS reason FROM object "
                       "WHERE fit_effective_batch IS NULL AND (json_extract(fit_hyperparameters_json, '$.effective_batch') IS NOT NULL "
                       "OR (derivation = 'dose_scale' AND json_extract(field_confidence_json, '$.fit_effective_batch.confidence') IS NOT NULL)) "
                       "ORDER BY 2, 3, 4, 5, 1")
#: how an ambiguous group's `stated` reads in the report: no confidence entry for the column, or the entry's confidence
BATCH_AMBIGUOUS_UNSTATED = "unstated (the adapter's emission before the ruling of 2026-10-13; its re-emission states the parent's batch)"
BATCH_AMBIGUOUS_BY_NAME = ("left NULL by name (the adapter's unverified entry: the parent's fit record copied beside the copy names no integer "
                           "effective_batch, or the copy carries none; each row's reason below it)")


def batch_ambiguous_statement(stated: Optional[str]) -> str:
    """The reading of an ambiguous group's confidence entry for the column (the SQL's ``stated``): :data:`BATCH_AMBIGUOUS_UNSTATED` for
    none, :data:`BATCH_AMBIGUOUS_BY_NAME` for ``unverified``, and the confidence itself, named, for anything else (a `derived` or
    `recorded` entry beside a NULL column is a statement the fold should not have left: worth reading, never guessed at)."""
    if stated is None:
        return BATCH_AMBIGUOUS_UNSTATED
    if stated == "unverified":
        return BATCH_AMBIGUOUS_BY_NAME
    return f"confidence entry {stated!r} beside a NULL column (read the row)"


@dataclass
class BatchAmbiguous:
    """One group of the ambiguous class -- ``(fit_writer, recorded value as stored, fit_model_id, the row's own confidence for the column;
    None for no entry)`` -- with EVERY row of it by name: ``members`` is ``[(object_id, the row's own reason for the NULL)]`` in object-id
    order, the reason the column's confidence entry's ``reason`` (else its ``source``), None for a row with no entry. ``recorded`` is None
    for a dose copy left NULL by name whose block records no ``effective_batch`` (the key absent or null, or no block at all: the copied
    record named none, or the copy carries none); the report shows it as ``-``. The count alone told
    an operator how many copies of one bucket were left NULL and never which; the ids and the adapter's
    reasons say which parent record to repair. The text and the JSON both list every member; nothing is truncated (a truncated list of ids
    is no list), and the aggregate stands beside them."""
    fit_writer: Optional[str]
    recorded: object
    fit_model_id: Optional[str]
    stated: Optional[str]
    members: List[Tuple[str, Optional[str]]] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return len(self.members)

    @property
    def statement(self) -> str:
        return batch_ambiguous_statement(self.stated)

    @property
    def summary(self) -> Tuple[Optional[str], object, Optional[str], Optional[str], int]:
        """The group's own line: ``(fit_writer, recorded, fit_model_id, stated, rows)``."""
        return (self.fit_writer, self.recorded, self.fit_model_id, self.stated, self.rows)

    @property
    def object_ids(self) -> List[str]:
        return [oid for oid, _why in self.members]

    def as_dict(self) -> Dict[str, object]:
        return {"fit_writer": self.fit_writer, "recorded": self.recorded, "fit_model_id": self.fit_model_id, "stated": self.stated,
                "statement": self.statement, "rows": self.rows, "objects": [{"object_id": oid, "unresolved": why} for oid, why in self.members]}


def batch_ambiguous_groups(conn) -> List[BatchAmbiguous]:
    """The ambiguous class of the folded table as :class:`BatchAmbiguous` groups, in the SQL's order, every row kept by name."""
    out: Dict[Tuple[Optional[str], object, Optional[str], Optional[str]], BatchAmbiguous] = {}
    for oid, w, rec, m, s, why in conn.execute(BATCH_AMBIGUOUS_SQL).fetchall():
        g = out.get((w, rec, m, s))
        if g is None:
            g = out[(w, rec, m, s)] = BatchAmbiguous(w, rec, m, s)
        g.members.append((str(oid), why))
    return list(out.values())


#: the column set, no recorded `effective_batch`: nothing for the class to compare -- a derived dose copy excepted, whose derived value with
#: no block value to be is the statement check's to name (THE STATEMENT CHECK in the section comment), not a count
BATCH_NOTHING_TO_COMPARE_SQL = ("SELECT COUNT(*) FROM object WHERE fit_effective_batch IS NOT NULL "
                                "AND json_extract(fit_hyperparameters_json, '$.effective_batch') IS NULL "
                                "AND NOT (derivation = 'dose_scale' AND json_extract(field_confidence_json, '$.fit_effective_batch.confidence') = 'derived')")
#: the five batch values the class reads, each named as the record states it, in the order the reason names them
_BATCH_COMPONENTS: Tuple[str, ...] = ("fit_effective_batch", "fit_hyperparameters_json.effective_batch", "fit_hyperparameters_json.resolved_batch_size",
                                      "fit_hyperparameters_json.batch_size", "fit_hyperparameters_json.accum_steps")


def integral_batch(v) -> bool:
    """A batch value in the record's own type: an ``int`` that is not a ``bool``. A float (64.5, or an integral 64.0), a numeric string,
    a boolean or an unstated key is NOT one -- the trainers record integers, and a value coerced into an integer would be a reading the
    record never made (a truncated 64.5 would be written as 64)."""
    return isinstance(v, int) and not isinstance(v, bool)


def batch_unclassified_reason(live, recorded, resolved, micro, accum) -> Optional[str]:
    """Why a row whose two values disagree is NOT in the clearly-wrong class, or None when it is. Checked in order and named with the
    value and its type, nothing coerced: a batch value that is not an integer (:func:`integral_batch`), the two values agreeing (never
    in the class), the live value not being the block's micro-batch (``resolved_batch_size``), the block's own ``batch_size x
    accum_steps`` not being its ``effective_batch`` -- the last two are the sidecar hook's signature failing."""
    for name, v in zip(_BATCH_COMPONENTS, (live, recorded, resolved, micro, accum)):
        if not integral_batch(v):
            what = "not stated in the row's record" if v is None else f"{v!r} ({type(v).__name__}), not an integer"
            return f"{name} is {what}; nothing is coerced"
    if live == recorded:
        return "the two values agree"
    if live != resolved:
        return f"fit_effective_batch {live} is not the block's resolved_batch_size {resolved}: the sidecar hook's signature does not hold"
    if micro * accum != recorded:
        return (f"the block's own batch_size {micro} x accum_steps {accum} = {micro * accum} is not its effective_batch {recorded}: the sidecar "
                "hook's signature does not hold (the trainer's identity fails in the row's record)")
    return None


def batch_hook_signature(live, recorded, resolved, micro, accum) -> bool:
    """The pure predicate behind the class: every batch value an integer in its own type, the live value the block's micro-batch
    (``resolved_batch_size``) and the block's own ``batch_size x accum_steps`` its ``effective_batch`` -- so the row's recorded block is
    internally consistent and the column alone carries the hook's reading. ``not batch_unclassified_reason(...)``."""
    return batch_unclassified_reason(live, recorded, resolved, micro, accum) is None


def batch_derived_reason(confidence: Mapping[str, object]) -> Optional[str]:
    """Why a disagreeing row is NOT the hook's statement, or None: the row's own ``field_confidence_json`` marks ``fit_effective_batch``
    ``derived`` -- the fit-sidecars adapter's statement of a dose copy's batch from the parent's record copied beside the copy,
    the sidecar hook's dose path stating the same, or a subtraction anchor's from its parents' -- or ``unverified`` -- the
    adapter's NULL-by-name statement (the copied record names no integer batch, or the copy carries none), beside which no value of the
    adapter's stands, so a value that does stand is an earlier statement the fold retained, never the hook's reading (the hook writes no
    ``unverified`` entry for an integer it wrote) -- so the sidecar hook's micro-batch signature cannot apply to it, whatever the five
    values read. Read from the entry's CONFIDENCE alone, never its wording or its writer. Such a row is unclassified, refused by name and
    never written: the pass corrects the hook's reading alone, never a derived statement (the rule writes no correction record for a
    dose copy; a derived copy's agreement with its parent row is the read-only cross-check, :func:`batch_parent_checks`)."""
    entry = confidence.get("fit_effective_batch") if isinstance(confidence, Mapping) else None
    if isinstance(entry, Mapping) and entry.get("confidence") == "derived":
        return ("fit_effective_batch is a derived statement (field_confidence_json: derived -- "
                f"{str(entry.get('source') or '')[:100]}), not the sidecar hook's reading; the hook's signature does not apply")
    if isinstance(entry, Mapping) and entry.get("confidence") == "unverified":
        return ("fit_effective_batch is the adapter's NULL-by-name statement with a value still standing (field_confidence_json: unverified -- "
                f"{str(entry.get('reason') or entry.get('source') or '')[:100]}): a retained earlier statement, not the sidecar hook's reading; "
                "the hook's signature does not apply")
    return None


#: every dose copy whose `fit_effective_batch` is a `derived` statement, with its parent row's two batch values (NULL for an absent row):
#: the registry-level agreement property the adapter's single source leaves to the registry (the section comment, THE PARENT CROSS-CHECK)
BATCH_PARENT_SQL = ("SELECT o.object_id, o.fit_effective_batch, o.fit_writer, o.parent_object_ids_json, o.parent_object_id "
                    "FROM object o WHERE o.derivation = 'dose_scale' AND o.fit_effective_batch IS NOT NULL "
                    "AND json_extract(o.field_confidence_json, '$.fit_effective_batch.confidence') = 'derived' ORDER BY o.object_id")
BATCH_PARENT_ROW_SQL = "SELECT fit_effective_batch, fit_hyperparameters_json FROM object WHERE object_id = ?"
BATCH_PARENT_AGREES = "agrees"
BATCH_PARENT_DISAGREES = "disagrees"
#: a disagreement of TYPE: the parent's recorded effective batch (or the copy's derived one) is not an integer in its own type -- a float
#: (8.0), a boolean (true), a numeric string ("8") -- so the two values are not one integer whatever `==` says of them (Python's 8.0 == 8
#: and True == 1 hold, and read as agreement before this status existed); listed by name with the value and its type, exit 3 on --verify,
#: never written for; nothing is coerced
BATCH_PARENT_NONINTEGRAL = ("a value not an integer in its own type (the parent's recorded effective batch, or the copy's derived one: a float, a numeric "
                            "string or a boolean; nothing is coerced, so the two are not one integer: listed as a disagreement, never written for)")
BATCH_PARENT_ABSENT = "parent row absent from the registry (not a disagreement: nothing to compare)"
BATCH_PARENT_STATES_NONE = "parent row states no effective batch (not a disagreement: nothing to compare)"
BATCH_PARENT_UNNAMED = "the copy names no parent (not a disagreement: nothing to compare)"


@dataclass
class BatchParentCheck:
    """One dose copy whose ``fit_effective_batch`` is a ``derived`` statement, read against its PARENT object row: ``derived`` the copy's
    column; ``parent_object_id`` the copy's one ``parent_object_ids_json`` id, else its ``parent_object_id``, None when it names none;
    ``parent_present`` whether that row is in the folded table; ``parent_value`` the parent's recorded effective batch (its own
    ``fit_hyperparameters_json.effective_batch``, else its ``fit_effective_batch``; None when it states none) and ``parent_source`` which
    of the two it was read from. ``status`` is one of :data:`BATCH_PARENT_AGREES`, :data:`BATCH_PARENT_DISAGREES` (listed by name, exit 3
    on ``--verify``, never written for), :data:`BATCH_PARENT_NONINTEGRAL` (a value that is not an integer in its own type -- the parent's
    ``8.0``, ``true`` or ``"8"`` beside a copy deriving 8, which ``==`` alone reads as agreement; listed and exited on as a disagreement,
    with the value and its type, nothing coerced), :data:`BATCH_PARENT_ABSENT`, :data:`BATCH_PARENT_STATES_NONE`, :data:`BATCH_PARENT_UNNAMED`."""
    object_id: str
    derived: object
    fit_writer: Optional[str]
    parent_object_id: Optional[str]
    parent_present: bool
    parent_value: object = None
    parent_source: Optional[str] = None

    @property
    def status(self) -> str:
        if self.parent_object_id is None:
            return BATCH_PARENT_UNNAMED
        if not self.parent_present:
            return BATCH_PARENT_ABSENT
        if self.parent_value is None:
            return BATCH_PARENT_STATES_NONE
        if not integral_batch(self.parent_value) or not integral_batch(self.derived):
            return BATCH_PARENT_NONINTEGRAL           # before the value comparison: 8.0 == 8 and True == 1 hold, and are not agreement
        return BATCH_PARENT_AGREES if self.parent_value == self.derived else BATCH_PARENT_DISAGREES

    @property
    def disagrees(self) -> bool:
        """Listed by name and exited on (``--verify`` 3), never written for: a different integer, or a value that is not an integer in its
        own type -- a parent recording ``8.0`` or ``true`` is not recording the copy's 8 or 1, whatever ``==`` says of the pair."""
        return self.status in (BATCH_PARENT_DISAGREES, BATCH_PARENT_NONINTEGRAL)

    @property
    def line(self) -> str:
        """The row's own line of the report: the copy, its derived value, the parent and what the parent says."""
        who = f"{self.object_id} derived {self.derived!r} [{self.fit_writer or '-'}]"
        if self.parent_object_id is None:
            return f"{who}: {BATCH_PARENT_UNNAMED}"
        if not self.parent_present:
            return f"{who}: parent {self.parent_object_id[:16]} {BATCH_PARENT_ABSENT[len('parent row '):]}"
        if self.parent_value is None:
            return f"{who}: parent {self.parent_object_id[:16]} {BATCH_PARENT_STATES_NONE[len('parent row '):]}"
        if self.status == BATCH_PARENT_NONINTEGRAL:
            odd = " and ".join(f"{side} {v!r} ({type(v).__name__})" for side, v in (("the parent's", self.parent_value), ("the copy's derived", self.derived))
                               if not integral_batch(v))
            return (f"{who} vs parent {self.parent_object_id[:16]} {self.parent_value!r} ({self.parent_source}): {odd} not an integer in its own type; "
                    "nothing is coerced, so this is not an agreement")
        verb = "=" if self.status == BATCH_PARENT_AGREES else "!="
        return f"{who} {verb} parent {self.parent_object_id[:16]} {self.parent_value!r} ({self.parent_source})"

    def as_dict(self) -> Dict[str, object]:
        return {"object_id": self.object_id, "derived": self.derived, "fit_writer": self.fit_writer, "parent_object_id": self.parent_object_id,
                "parent_present": self.parent_present, "parent_value": self.parent_value, "parent_source": self.parent_source, "status": self.status}


def batch_parent_checks(conn) -> List[BatchParentCheck]:
    """Every dose copy whose ``fit_effective_batch`` is a ``derived`` statement, read against its parent row (THE PARENT CROSS-CHECK in
    the section comment), in object-id order, from the folded database alone: read-only, nothing coerced -- the parent's value in its own
    type, an integer compared with ``==`` to the copy's integer, and a parent recording ``8.0``, ``true`` or ``"8"`` beside a copy deriving 8
    :data:`BATCH_PARENT_NONINTEGRAL`, listed with the value and its type (``==`` alone read the float and the boolean as agreement)."""
    out: List[BatchParentCheck] = []
    for oid, derived, writer, pids_raw, pid_col in conn.execute(BATCH_PARENT_SQL).fetchall():
        pids = _parsed_json(pids_raw)
        parent = str(pids[0]) if isinstance(pids, list) and len(pids) == 1 and pids[0] else (str(pid_col) if pid_col else None)
        if parent is None:
            out.append(BatchParentCheck(str(oid), derived, writer, None, False))
            continue
        prow = conn.execute(BATCH_PARENT_ROW_SQL, (parent,)).fetchone()
        if prow is None:
            out.append(BatchParentCheck(str(oid), derived, writer, parent, False))
            continue
        pfeb, php = prow
        block = _parsed_json(php)
        recorded = block.get("effective_batch") if isinstance(block, Mapping) else None
        if recorded is not None:
            out.append(BatchParentCheck(str(oid), derived, writer, parent, True, recorded, "the parent's fit_hyperparameters_json.effective_batch"))
        elif pfeb is not None:
            out.append(BatchParentCheck(str(oid), derived, writer, parent, True, pfeb, "the parent's fit_effective_batch"))
        else:
            out.append(BatchParentCheck(str(oid), derived, writer, parent, True))
    return out


#: every dose copy whose column carries the adapter's statement that can be inconsistent with its own row -- a `derived` entry (whatever
#: the column and the block hold), or an `unverified` entry with a value standing beside it -- with the column and the block as stored:
#: THE STATEMENT CHECK in the section comment reads the pair in Python, in its own type, where the class SQL's equality read 8 and 8.0 as
#: one value and a missing key as nothing to compare. An `unverified` entry beside NULL is the adapter's statement to the letter (the
#: ambiguous class names it) and is not selected; a row with no entry for the column is no statement of the adapter's.
BATCH_STATEMENT_SQL = ("SELECT object_id, fit_effective_batch, fit_writer, fit_hyperparameters_json, "
                       "json_extract(field_confidence_json, '$.fit_effective_batch.confidence') AS stated FROM object "
                       "WHERE derivation = 'dose_scale' AND (json_extract(field_confidence_json, '$.fit_effective_batch.confidence') = 'derived' "
                       "OR (json_extract(field_confidence_json, '$.fit_effective_batch.confidence') = 'unverified' AND fit_effective_batch IS NOT NULL)) "
                       "ORDER BY object_id")
BATCH_STATEMENT_CONSISTENT = "consistent"
#: a `derived` entry beside a NULL column: the entry names a value derived and none stands (the adapter writes the two together)
BATCH_STATEMENT_DERIVED_NULL = "a derived entry beside a NULL column"
#: the column itself is not an integer in its own type (a 8.5 the INTEGER affinity keeps as REAL): no derived integer to be the block's
BATCH_STATEMENT_DERIVED_NONINTEGRAL = "the derived value is not an integer in its own type"
#: the block records no `effective_batch` (the key absent or null, or no block at all): nothing for the derived value to be the block's own
BATCH_STATEMENT_BLOCK_NONE = "the block records no effective_batch"
#: the block's `effective_batch` is not an integer in its own type (8.0, true, "8"): the adapter would refuse it by name, and 8 is not 8.0
BATCH_STATEMENT_BLOCK_NONINTEGRAL = "the block's effective_batch is not an integer in its own type"
#: the block records another integer: the class SQL selects this shape too (a disagreement), refused there by the entry
BATCH_STATEMENT_BLOCK_DISAGREES = "the block's effective_batch is another integer"
#: an `unverified` (NULL-by-name) entry with a value standing beside it, whatever the block says: a retained earlier statement
BATCH_STATEMENT_VALUE_BESIDE_NULL_BY_NAME = "a value stands beside the adapter's NULL-by-name entry"
#: the statuses in the order the report counts them: consistent first, then the inconsistent ones
BATCH_STATEMENT_STATUSES: Tuple[str, ...] = (BATCH_STATEMENT_CONSISTENT, BATCH_STATEMENT_DERIVED_NULL, BATCH_STATEMENT_DERIVED_NONINTEGRAL,
                                             BATCH_STATEMENT_BLOCK_NONE, BATCH_STATEMENT_BLOCK_NONINTEGRAL, BATCH_STATEMENT_BLOCK_DISAGREES,
                                             BATCH_STATEMENT_VALUE_BESIDE_NULL_BY_NAME)


@dataclass
class BatchStatementCheck:
    """One dose copy whose ``fit_effective_batch`` carries the adapter's statement, read against its OWN row (THE STATEMENT CHECK in the
    section comment): ``stated`` the entry's confidence (``derived`` or ``unverified``), ``value`` the column as stored, ``block_present``
    whether the row has a ``fit_hyperparameters_json`` block, ``recorded`` the block's ``effective_batch`` as stored (None when the key is
    absent or null, or there is no block). ``status`` is :data:`BATCH_STATEMENT_CONSISTENT` -- a derived column and a block value that are
    one integer (not a boolean) on both sides, or the adapter's NULL beside its ``unverified`` entry -- or names which side fails:
    :data:`BATCH_STATEMENT_DERIVED_NULL`, :data:`BATCH_STATEMENT_DERIVED_NONINTEGRAL`, :data:`BATCH_STATEMENT_BLOCK_NONE`,
    :data:`BATCH_STATEMENT_BLOCK_NONINTEGRAL`, :data:`BATCH_STATEMENT_BLOCK_DISAGREES`, :data:`BATCH_STATEMENT_VALUE_BESIDE_NULL_BY_NAME`.
    Nothing is coerced: ``8.0 == 8`` and ``True == 1`` hold in Python and in SQLite, and neither is an agreement here."""
    object_id: str
    stated: str
    value: object
    fit_writer: Optional[str]
    block_present: bool
    recorded: object = None

    @property
    def status(self) -> str:
        if self.stated == "unverified":
            # the adapter's NULL-by-name statement: consistent only when nothing stands in the column (the SQL selects the other case)
            return BATCH_STATEMENT_CONSISTENT if self.value is None else BATCH_STATEMENT_VALUE_BESIDE_NULL_BY_NAME
        if self.value is None:
            return BATCH_STATEMENT_DERIVED_NULL
        if not integral_batch(self.value):
            return BATCH_STATEMENT_DERIVED_NONINTEGRAL
        if self.recorded is None:
            return BATCH_STATEMENT_BLOCK_NONE
        if not integral_batch(self.recorded):
            return BATCH_STATEMENT_BLOCK_NONINTEGRAL            # before the value comparison: 8.0 == 8 holds, and is not the block's own 8
        return BATCH_STATEMENT_CONSISTENT if self.value == self.recorded else BATCH_STATEMENT_BLOCK_DISAGREES

    @property
    def inconsistent(self) -> bool:
        """Listed by name and exited on (``--verify`` 3), never written for."""
        return self.status != BATCH_STATEMENT_CONSISTENT

    @property
    def block_text(self) -> str:
        """The block's ``effective_batch`` as the report shows it: the value with its type, ``-`` for the key absent or null, ``no block``."""
        if not self.block_present:
            return "no block"
        return "-" if self.recorded is None else f"{self.recorded!r} ({type(self.recorded).__name__})"

    @property
    def line(self) -> str:
        """The row's own line of the report: the copy, its entry, its column with its type, its block's value, and which side fails."""
        value = "NULL" if self.value is None else f"{self.value!r} ({type(self.value).__name__})"
        return (f"{self.object_id} {self.stated} entry, column {value} [{self.fit_writer or '-'}], block effective_batch {self.block_text}: "
                f"{self.status}; nothing is coerced, never written for")

    def as_dict(self) -> Dict[str, object]:
        return {"object_id": self.object_id, "stated": self.stated, "value": self.value, "fit_writer": self.fit_writer,
                "block_present": self.block_present, "recorded": self.recorded, "status": self.status}


def batch_statement_checks(conn) -> List[BatchStatementCheck]:
    """Every dose copy whose ``fit_effective_batch`` carries a ``derived`` entry, and every one whose ``unverified`` entry has a value
    standing beside it, read against its OWN column and block in Python, in their own type (THE STATEMENT CHECK in the section comment),
    in object-id order, from the folded database alone: read-only, nothing coerced."""
    out: List[BatchStatementCheck] = []
    for oid, value, writer, hyper_raw, stated in conn.execute(BATCH_STATEMENT_SQL).fetchall():
        block = _parsed_json(hyper_raw)
        present = isinstance(block, Mapping)
        out.append(BatchStatementCheck(str(oid), str(stated), value, writer, present, block.get("effective_batch") if present else None))
    return out


@dataclass
class BatchCorrection:
    """One object row whose two values disagree: what the row carries, the class it falls in, and what ``apply`` did."""
    object_id: str
    object_kind: str
    fit_target_kind: str
    derivation: str
    campaign: str
    status: str
    named_role: Optional[str]
    base_scale: Optional[float]
    fit_writer: Optional[str]
    fit_model_id: Optional[str]
    record_uri: Optional[str]
    live: object                                                            # the column's value as stored (an int on a clearly-wrong row)
    recorded: object                                                        # the block's effective_batch in its own type (an int on a clearly-wrong row)
    hyper: Dict[str, object]                                                # the row's fit_hyperparameters_json, parsed
    confidence: Dict[str, object]                                           # the row's field_confidence_json, parsed (the record carries it whole)
    row: Dict[str, object]                                                  # the live row's columns the record copies (identity + pairing)
    unclassified: Optional[str]                                             # None: the hook signature holds (clearly wrong); else why not, refused by name
    pending: Optional[str] = None
    pending_own: bool = False
    written: Optional[str] = None
    refused: Optional[str] = None

    @property
    def clearly_wrong(self) -> bool:
        """The hook's signature holds over the row's own values, every one an integer: the apply acts on the row."""
        return self.unclassified is None

    @property
    def carrier(self) -> str:
        """Which writer's record carries the row (the record_uri's prefix): ``adapter`` for an adapters bundle, else ``native``."""
        return "adapter" if str(self.record_uri or "").startswith("records/adapters/") else "native"

    @property
    def group(self) -> Tuple[str, str, str]:
        role = self.named_role if self.named_role is not None else (f"base_scale {self.base_scale:g}" if self.base_scale is not None else "-")
        return (self.object_kind, str(role), self.carrier)

    def hold(self, statement: "ScopeStatement") -> None:
        self.pending, self.pending_own = statement.sha, statement.own

    @property
    def hold_reason(self) -> str:
        what = "this pass's own correction record" if self.pending_own else "an object record"
        return f"{what} of this object is in the log and not folded yet ({(self.pending or '')[:12]}); ingest first, then re-run"


@dataclass
class BatchReport:
    root: str
    rows: List[BatchCorrection]                                             # every row whose two values disagree (clearly wrong and unclassified)
    applied: bool
    verify: bool = False
    #: the ambiguous class by group -- (fit_writer, recorded as stored, fit_model_id, the row's own confidence for the column) -- every row of
    #: each named (:class:`BatchAmbiguous`)
    ambiguous: List[BatchAmbiguous] = field(default_factory=list)
    nothing_to_compare: int = 0
    #: every derived dose copy read against its parent row (:class:`BatchParentCheck`; THE PARENT CROSS-CHECK): read-only, listed, never written for
    parent_checks: List[BatchParentCheck] = field(default_factory=list)
    #: every dose copy carrying the adapter's statement, read against its OWN row (:class:`BatchStatementCheck`; THE STATEMENT CHECK): read-only
    statement_checks: List[BatchStatementCheck] = field(default_factory=list)

    @property
    def selected(self) -> List[BatchCorrection]:
        """The rows the apply acts on: the clearly-wrong class (an unclassified row is listed and refused, never acted on; a statement
        inconsistency or a parent disagreement is no correction candidate at all)."""
        return [r for r in self.rows if r.clearly_wrong]

    @property
    def listed(self) -> List[object]:
        """What a verify lists (exit 3 when any): EVERY disagreement of a row with its own block, the unclassified ones included -- a
        disagreement the pass refuses is still a row the folded table carries, and the landing window must fail by name rather than read
        a clean class -- then every dose copy whose adapter statement is inconsistent with its own row (:attr:`statement_inconsistencies`),
        then every derived dose copy that disagrees with its PARENT row (:attr:`parent_disagreements`); an object once, whichever ways it
        fails (a derived 1 beside a block saying 64 is a disagreement the class SQL selects, a statement inconsistency and, with a parent
        recording 64, a parent disagreement -- and one row of the table)."""
        named = {r.object_id for r in self.rows}
        out: List[object] = list(self.rows)
        for c in list(self.statement_inconsistencies) + list(self.parent_disagreements):
            if c.object_id not in named:
                named.add(c.object_id)
                out.append(c)
        return out

    @property
    def statement_inconsistencies(self) -> List[BatchStatementCheck]:
        """The dose copies whose adapter statement does not hold of their own row -- a derived value that is not the block's own integer
        (the block recording none, a non-integer, another integer; the column NULL or not an integer), or a value standing beside the
        NULL-by-name entry -- listed by name, never written for."""
        return [c for c in self.statement_checks if c.inconsistent]

    def statement_counts(self) -> Dict[str, int]:
        """How many statement-carrying dose copies read each way against their own row, keyed by :attr:`BatchStatementCheck.status`, every
        status present in the order of :data:`BATCH_STATEMENT_STATUSES`."""
        out = {s: 0 for s in BATCH_STATEMENT_STATUSES}
        for c in self.statement_checks:
            out[c.status] += 1
        return out

    @property
    def parent_disagreements(self) -> List[BatchParentCheck]:
        """The derived dose copies whose value is not their parent row's recorded effective batch -- a different integer, or a parent (or
        derived) value that is not an integer in its own type, :data:`BATCH_PARENT_NONINTEGRAL` -- listed by name, never written for."""
        return [c for c in self.parent_checks if c.disagrees]

    @property
    def parent_absent(self) -> List[BatchParentCheck]:
        """The derived dose copies whose parent row is not in the registry: not a disagreement, said so by name."""
        return [c for c in self.parent_checks if c.status == BATCH_PARENT_ABSENT]

    def parent_counts(self) -> Dict[str, int]:
        """How many derived dose copies read each way against their parent row, keyed by :attr:`BatchParentCheck.status`, every status present."""
        out = {s: 0 for s in (BATCH_PARENT_AGREES, BATCH_PARENT_DISAGREES, BATCH_PARENT_NONINTEGRAL, BATCH_PARENT_ABSENT, BATCH_PARENT_STATES_NONE,
                              BATCH_PARENT_UNNAMED)}
        for c in self.parent_checks:
            out[c.status] += 1
        return out

    @property
    def unclassified(self) -> List[BatchCorrection]:
        return [r for r in self.rows if not r.clearly_wrong]

    def by_group(self) -> Dict[Tuple[str, str, str], Tuple[int, int, int]]:
        """Per (object kind, role or base scale, carrier): (rows, the live value, the recorded value) of the clearly-wrong class."""
        out: Dict[Tuple[str, str, str], Tuple[int, int, int]] = {}
        for r in self.selected:
            n, live, rec = out.get(r.group, (0, r.live, r.recorded))
            out[r.group] = (n + 1, live, rec)
        return dict(sorted(out.items()))

    @property
    def written(self) -> List[str]:
        return [r.written for r in self.rows if r.written]

    @property
    def held(self) -> List[BatchCorrection]:
        return [r for r in self.rows if r.refused]

    def as_dict(self) -> Dict[str, object]:
        return {"root": self.root, "action": BATCH_ACTION, "applied": self.applied, "verify": self.verify, "rows": len(self.selected),
                "unclassified": len(self.unclassified), "ambiguous_rows": sum(g.rows for g in self.ambiguous), "nothing_to_compare": self.nothing_to_compare,
                "by_group": [{"object_kind": k, "role": role, "carrier": c, "rows": n, "live": live, "recorded": rec} for (k, role, c), (n, live, rec) in self.by_group().items()],
                "ambiguous": [g.as_dict() for g in self.ambiguous],       # each group with every row by object id and its reason
                "written": len(self.written), "held": len(self.held),
                "parent_check": {"copies": len(self.parent_checks), "disagreements": len(self.parent_disagreements), "by_status": self.parent_counts(),
                                 "copies_by_name": [c.as_dict() for c in self.parent_checks]},     # read-only: every derived copy against its parent row
                "statement_check": {"copies": len(self.statement_checks), "inconsistent": len(self.statement_inconsistencies),
                                    "by_status": self.statement_counts(),
                                    "copies_by_name": [c.as_dict() for c in self.statement_checks]},    # read-only: the adapter's statement against the row's own block
                "corrections": [{"object_id": r.object_id, "object_kind": r.object_kind, "fit_target_kind": r.fit_target_kind, "derivation": r.derivation,
                                 "campaign": r.campaign, "status": r.status, "named_role": r.named_role, "base_scale": r.base_scale, "carrier": r.carrier,
                                 "live": r.live, "recorded": r.recorded, "clearly_wrong": r.clearly_wrong, "unclassified": r.unclassified,
                                 "pending": r.pending, "pending_own": r.pending_own, "written": r.written, "refused": r.refused} for r in self.rows]}


def _parsed_json(value) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def batch_class_rows(conn) -> Tuple[List[BatchCorrection], List[BatchAmbiguous], int]:
    """``(every object row whose fit_effective_batch disagrees with its own recorded effective_batch, classed; the ambiguous rows grouped by
    (fit_writer, recorded value as stored, fit_model_id, the row's own confidence for the column), every row of a group kept by object id with
    its reason; the nothing-to-compare count)``, read from the database only. EVERY selected row
    is returned: clearly wrong when :func:`batch_unclassified_reason` reads None over the row's own values (the column, and the block's
    four keys from the parsed JSON, in their own type), else unclassified with that reason; no value is coerced or dropped. The record's
    copied columns are read from the live row here: the required identity fields but ``status``, and the pairing columns the row's
    ``fit_target_kind`` requires non-null (``vocab.FIT_TARGET_RULES``)."""
    from . import vocab as V
    required = [c for c in R.REQUIRED_FIELDS["object"] if c not in R.OBJECT_OMITTABLE_WITH_ROW]
    cols = {c[1] for c in conn.execute("PRAGMA table_info(object)").fetchall()}
    rows: List[BatchCorrection] = []
    for rec in conn.execute(BATCH_CLASS_SQL).fetchall():
        (object_id, object_kind, kind, derivation, campaign, status, named_role, base_scale, fit_writer, fit_model_id, record_uri, live,
         hyper_raw, conf_raw) = rec
        parsed = _parsed_json(hyper_raw)
        hyper: Dict[str, object] = dict(parsed) if isinstance(parsed, Mapping) else {}
        recorded = hyper.get("effective_batch")
        conf = _parsed_json(conf_raw)
        conf_map: Dict[str, object] = dict(conf) if isinstance(conf, Mapping) else {}
        reason = batch_derived_reason(conf_map) or batch_unclassified_reason(live, recorded, hyper.get("resolved_batch_size"), hyper.get("batch_size"),
                                                                             hyper.get("accum_steps"))
        rule = V.FIT_TARGET_RULES.get(str(kind)) or {}
        carried = list(dict.fromkeys(required + [c for c in rule.get("nonnull", ()) if c in cols]))
        live_row = conn.execute(f"SELECT {', '.join(carried)} FROM object WHERE object_id = ?", (object_id,)).fetchone()
        row = {c: (_parsed_json(v) if c in _OBJECT_JSON_COLUMNS else v) for c, v in zip(carried, live_row)}
        rows.append(BatchCorrection(str(object_id), str(object_kind), str(kind), str(derivation), str(campaign), str(status), named_role,
                                    float(base_scale) if base_scale is not None else None, fit_writer, fit_model_id, record_uri, live, recorded,
                                    hyper, conf_map, row, reason))
    ambiguous = batch_ambiguous_groups(conn)
    nothing = int(conn.execute(BATCH_NOTHING_TO_COMPARE_SQL).fetchone()[0])
    return rows, ambiguous, nothing


def object_history_key(object_id: str) -> str:
    """The object's ``row_history.natural_key_json``."""
    return R.canonical_json({"object_id": object_id})


def accepted_object_statement_shas(conn, object_id: str) -> Set[str]:
    """The ``object`` events the fold ACCEPTED for an object (a primary-key seek on ``row_history``)."""
    rows = conn.execute("SELECT record_sha256 FROM row_history WHERE table_name = 'object' AND natural_key_json = ? AND record_type = 'object'",
                        (object_history_key(object_id),)).fetchall()
    return {str(r[0]) for r in rows}


def _object_statements_in(path: str, object_ids: Optional[Set[str]] = None) -> List[ScopeStatement]:
    """The ``object`` records of one file (narrowed to ``object_ids``), each as a statement of its row; this pass's own by its marker."""
    out: List[ScopeStatement] = []
    for _i, rec, _sha in R.read_record_file(path, compute_sha=False):
        if rec.get("record_type") != "object":
            continue
        payload = rec.get("payload") if isinstance(rec.get("payload"), Mapping) else None
        oid = payload.get("object_id") if payload else None
        if not isinstance(oid, str) or not oid or (object_ids is not None and oid not in object_ids):
            continue
        extra = payload.get("extra") if isinstance(payload.get("extra"), Mapping) else {}       # type: ignore[union-attr]
        out.append(ScopeStatement(oid, R.record_sha(rec), path, str(rec.get("written_utc", "")), ("fit_effective_batch",) if "fit_effective_batch" in payload else (),
                                  BATCH_MARKER_KEY in extra, "object"))
    return out


def object_statement_files(records_root: str) -> List[str]:
    """Every ``*-object.jsonl`` / ``.jsonl.gz`` under ``records_root`` (every writer names a record file by its record type)."""
    return _record_files(records_root, _OBJECT_FILE)


def object_statements(records_root: str, object_ids: Optional[Set[str]] = None) -> Dict[str, List[ScopeStatement]]:
    """Every ``object`` record of the selected objects in the WHOLE log, by object id, in ``(written_utc, sha)`` order."""
    out: Dict[str, List[ScopeStatement]] = {}
    for path in object_statement_files(records_root):
        for s in _object_statements_in(path, object_ids):
            out.setdefault(s.run_id, []).append(s)
    for recs in out.values():
        recs.sort(key=lambda s: s.order_key)
    return out


def object_statements_in_directory(directory: str, object_id: str) -> List[ScopeStatement]:
    """The ``object`` records of one object under one directory (the target directory, read again at the write)."""
    if not os.path.isdir(directory):
        return []
    out: List[ScopeStatement] = []
    for fn in sorted(os.listdir(directory)):
        if _OBJECT_FILE.search(fn):
            out.extend(_object_statements_in(os.path.join(directory, fn), {object_id}))
    out.sort(key=lambda s: s.order_key)
    return out


def batch_correction_payload(row: BatchCorrection) -> Dict[str, object]:
    """The record (the section comment): the copied identity and pairing columns, the recorded batch, the row's whole confidence JSON with the
    column's entry, the marker under ``extra``. No ``status``, no ``fit_hyperparameters_json``, no central stamp."""
    payload: Dict[str, object] = dict(row.row)
    payload["fit_effective_batch"] = row.recorded
    conf = dict(row.confidence)
    conf["fit_effective_batch"] = {"confidence": "recorded",
                                   "source": ("fit_hyperparameters_json.effective_batch (the trainer's effective_batch = batch_size x accum_steps); "
                                              "replaces the resolved_batch_size micro-batch the sidecar hook stated")}
    payload["field_confidence_json"] = conf
    payload["extra"] = {BATCH_MARKER_KEY: {"replaced": row.live, "derived_from": {k: row.hyper.get(k) for k in _BATCH_HP_KEYS}, "pass": BATCH_PRODUCER_SCRIPT}}
    return payload


def batch_target_directory(root: str, object_id: str) -> str:
    from . import identity as I
    return os.path.join(root, "records", BATCH_OWNER_CAMPAIGN, I.run_dir_name(f"object__{object_id[:16]}"))


def _batch_writer_for(root: str, row: BatchCorrection, now: Optional[str]):
    """A ``RecordWriter`` for the owner's directory of the object (:func:`batch_target_directory`) that stamps nothing of its own."""
    from . import emit
    producer = emit.producer_block(BATCH_PRODUCER_SCRIPT, code_sha=emit.git_sha(os.path.dirname(os.path.abspath(__file__))))
    w = emit.RecordWriter(os.path.join(root, "records"), BATCH_OWNER_CAMPAIGN, f"object__{row.object_id[:16]}", producer, disabled=False, payload_extra={})
    w.payload_extra = {}
    return w


def verify_fit_effective_batch(root: str, conn=None) -> BatchReport:
    """The predicate alone over the FOLDED object table: every clearly-wrong row still standing, by name (an unclassified row is listed
    too: it is a disagreement the pass refuses, and the window must see it), every dose copy whose adapter statement does not hold of its
    own row (:func:`batch_statement_checks`: a derived value that is not the block's own integer, in Python and in its own type, or a
    value beside the NULL-by-name entry), and every derived dose copy whose value is not its parent row's (:func:`batch_parent_checks`; a
    parent absent from the registry is said so, not listed). Read-only, no walk, no hold."""
    own_conn = conn is None
    if own_conn:
        conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rows, ambiguous, nothing = batch_class_rows(conn)
        own_row_checks = batch_statement_checks(conn)
        parents = batch_parent_checks(conn)
    finally:
        if own_conn:
            conn.close()
    return BatchReport(root, rows, applied=False, verify=True, ambiguous=ambiguous, nothing_to_compare=nothing, parent_checks=parents,
                       statement_checks=own_row_checks)


def reconcile_fit_effective_batch(root: str, apply: bool = False, conn=None, now: Optional[str] = None,
                                  writer_factory: Optional[Callable[[str, BatchCorrection, Optional[str]], object]] = None) -> BatchReport:
    """The pass (the section comment): the disagreeing rows classed, the ambiguous rows listed, every statement-carrying dose copy read
    against its own row, every derived dose copy read against its parent row; with ``apply`` ONE ``object`` record per clearly-wrong row
    that is not held, into the owner's directory of the object, read once more at the write; an unclassified row is refused by name and
    never written, and nothing is ever written for a statement inconsistency or a parent disagreement (read-only, listed). Never writes
    the database. The record is validated before the write (ValueError, refused by name, when the live row's copied columns do not
    validate as a record)."""
    from . import validate as VAL
    own_conn = conn is None
    if own_conn:
        conn = DB.connect(DB.db_path(root), readonly=True)
    accepted_by: Dict[str, Set[str]] = {}
    try:
        rows, ambiguous, nothing = batch_class_rows(conn)
        own_row_checks = batch_statement_checks(conn)   # read-only; an inconsistent copy is listed, never a candidate for a write (`statements` below is the hold's)
        parents = batch_parent_checks(conn)             # read-only; a disagreement is listed, never a candidate for a write
        selected = [r for r in rows if r.clearly_wrong]
        statements = object_statements(os.path.join(root, "records"), {r.object_id for r in selected}) if selected else {}
        for row in selected:
            accepted_by[row.object_id] = accepted_object_statement_shas(conn, row.object_id)
            unfolded = unfolded_statements(conn, statements.get(row.object_id, []), accepted_by[row.object_id])
            if unfolded:
                row.hold(unfolded[-1])
        if apply:
            for row in rows:
                if not row.clearly_wrong:
                    row.refused = (f"fit_effective_batch {row.live!r} disagrees with the recorded effective_batch {row.recorded!r} and {row.unclassified}: "
                                   "an unclassified disagreement, refused by name, never corrected by this pass")
                    continue
                if row.pending:
                    row.refused = row.hold_reason
                    continue
                directory = batch_target_directory(root, row.object_id)
                try:
                    late = unfolded_statements(conn, object_statements_in_directory(directory, row.object_id), accepted_by[row.object_id])
                    if late:
                        row.hold(late[-1])
                        row.refused = row.hold_reason
                        continue
                    payload = batch_correction_payload(row)
                    from . import emit
                    probe = R.make_record("object", payload, emit.producer_block(BATCH_PRODUCER_SCRIPT, code_sha="unverified"), "0" * 32, R.utc_now())
                    v = VAL.validate_record(probe, "<fit-effective-batch>")
                    if isinstance(v, VAL.Malformed):
                        raise ValueError(f"the correction record for {row.object_id[:16]} does not validate (field {v.field}): {v.reason}; nothing written")
                    w = (writer_factory or _batch_writer_for)(root, row, now)
                    row.written = w.write("object", payload)
                except (ValueError, OSError) as exc:
                    row.refused = str(exc)
    finally:
        if own_conn:
            conn.close()
    return BatchReport(root, rows, apply, ambiguous=ambiguous, nothing_to_compare=nothing, parent_checks=parents, statement_checks=own_row_checks)


def format_batch_report(rep: BatchReport) -> str:
    """The dry run's text: the clearly-wrong class by (kind, role or base scale, carrier) with the two values, the ambiguous class by
    (writer, recorded value, model, the row's own confidence for the column) with EVERY row of a group under it by object id and its reason
    (none truncated: the ids are what an operator repairs by), the nothing-to-compare count, the unclassified rows by name, the pending
    lines, the parent cross-check (:func:`_batch_parent_lines`: the counts, every disagreement and every absent parent by name), the
    written / held lines, the total last (the landing window shows the tail). A verify renders one line per disagreeing row -- with its
    own block, or with its parent row -- the count last."""
    if rep.verify:
        lines = ["fit-effective-batch verify (read-only, the folded database): object rows whose fit_effective_batch disagrees with their own "
                 "recorded effective_batch, dose copies whose adapter statement does not hold of their own row (a derived value that is not the "
                 "block's own integer in its own type, or a value beside the NULL-by-name entry), and derived dose copies whose fit_effective_batch "
                 "disagrees with their PARENT row's recorded effective batch (a different integer, or a value not an integer in its own type: "
                 "nothing is coerced)"]
        for r in rep.rows:
            lines.append(f"  {r.object_id} {r.live!r} -> {r.recorded!r} [{r.object_kind}{'' if r.clearly_wrong else f', unclassified: {r.unclassified}'}]")
        lines.extend(_batch_statement_lines(rep))
        lines.extend(_batch_parent_lines(rep))
        lines.append(f"inconsistent {len(rep.listed)} row(s)" + ("; the class is consistent" if not rep.listed else ""))
        return "\n".join(lines)
    lines = ["fit-effective-batch: object rows whose fit_effective_batch is the micro-batch the sidecar hook read (resolved_batch_size) where the "
             "row's own recorded hyperparameters say the optimizer width (batch_size x accum_steps = effective_batch)"]
    for (kind, role, carrier), (n, live, rec) in rep.by_group().items():
        lines.append(f"  {kind:<16} {role:<16} {carrier:<8} {live} -> {rec} {n:>6}")
    amb = sum(g.rows for g in rep.ambiguous)
    lines.append("ambiguous (fit_effective_batch NULL beside a recorded effective_batch, or a dose copy's NULL stated by the adapter's own entry with "
                 "none recorded: a dose copy the fit-sidecars adapter has not restated yet, or "
                 f"left NULL by name -- its re-emission is the correction, the user 2026-10-13; listed by object id, not corrected): {amb} row(s)")
    for g in rep.ambiguous:
        # recorded `-`: the block records no effective_batch (a copy left NULL by name with the key absent or null, or no block at all)
        lines.append(f"  {str(g.fit_writer or '-'):<40} {('-' if g.recorded is None else str(g.recorded)):<4} {str(g.fit_model_id or '-'):<20} {g.rows:>6}  {g.statement}")
        for oid, why in g.members:              # every row of the group by name, with the adapter's own reason for its NULL
            lines.append(f"    {oid}" + (f"  {why}" if why else ""))
    lines.append(f"nothing to compare (fit_effective_batch set, no recorded effective_batch): {rep.nothing_to_compare} row(s)")
    if rep.unclassified:
        lines.append(f"unclassified (the two values disagree and the hook's signature does not hold, a batch value is not an integer, or the column is "
                     f"the adapter's derived or NULL-by-name statement, not the hook's; refused by name, never corrected): {len(rep.unclassified)} row(s)"
                     + "".join(f"\n  {r.object_id} {r.live!r} -> {r.recorded!r} [{r.object_kind}]: {r.unclassified}" for r in rep.unclassified[:20])
                     + ("\n  ..." if len(rep.unclassified) > 20 else ""))
    pending = [r for r in rep.selected if r.pending]
    if pending:
        lines.append(f"pending (an object record of the row is in the log, not folded yet): {len(pending)} row(s)" + "".join(
            f"\n  {r.object_id} record {r.pending[:12]}" + (" (this pass's own correction)" if r.pending_own else "") for r in pending[:20])
            + ("\n  ..." if len(pending) > 20 else ""))
    lines.extend(_batch_statement_lines(rep))
    lines.extend(_batch_parent_lines(rep))
    if rep.applied:
        for r in rep.held:
            lines.append(f"  held {r.object_id}: {r.refused}")
        lines.append(f"wrote {len(rep.written)} correction record(s); held {len(rep.held)}")
    lines.append(f"rows {len(rep.selected)} in {len(rep.by_group())} (kind, role, carrier) group(s)" + ("" if rep.applied else "; nothing written (dry run)"))
    return "\n".join(lines)


def _batch_parent_lines(rep: BatchReport) -> List[str]:
    """The parent cross-check's lines of either report: the counts by status, then every DISAGREEING copy by name with both values (a
    different integer, or a value not an integer in its own type, named with its type), then every copy whose parent row is absent by
    name (said so: not a disagreement). Agreeing copies and parents stating none are counted, not listed (nothing to read)."""
    counts = rep.parent_counts()
    n = len(rep.parent_checks)
    lines = [f"parent cross-check (read-only: every dose copy whose derived fit_effective_batch is read against its PARENT object row's recorded "
             f"effective batch; a disagreement is listed by name and never written for): {n} derived dose cop{'y' if n == 1 else 'ies'} -- "
             f"agree {counts[BATCH_PARENT_AGREES]}, disagree {counts[BATCH_PARENT_DISAGREES]}, not an integer in its own type "
             f"{counts[BATCH_PARENT_NONINTEGRAL]} (a disagreement: nothing is coerced), parent row absent from the registry "
             f"{counts[BATCH_PARENT_ABSENT]} (not a disagreement), parent states no batch {counts[BATCH_PARENT_STATES_NONE]}, "
             f"no parent named {counts[BATCH_PARENT_UNNAMED]}"]
    for c in rep.parent_disagreements:
        lines.append(f"  {c.line}")
    for c in rep.parent_absent:
        lines.append(f"  {c.line}")
    return lines


def _batch_statement_lines(rep: BatchReport) -> List[str]:
    """The statement check's lines of either report: the counts by status, then every INCONSISTENT copy by name with its column, its
    block's value (each with its type) and which side fails. Consistent copies are counted, not listed (nothing to read)."""
    counts = rep.statement_counts()
    n = len(rep.statement_checks)
    lines = [f"statement check (read-only: every dose copy whose fit_effective_batch carries the adapter's derived entry, or its NULL-by-name entry "
             f"with a value standing beside it, read against the row's OWN column and block in their own type; an inconsistent copy is listed by name "
             f"and never written for): {n} dose cop{'y' if n == 1 else 'ies'} -- consistent {counts[BATCH_STATEMENT_CONSISTENT]}, "
             f"{BATCH_STATEMENT_DERIVED_NULL} {counts[BATCH_STATEMENT_DERIVED_NULL]}, {BATCH_STATEMENT_DERIVED_NONINTEGRAL} "
             f"{counts[BATCH_STATEMENT_DERIVED_NONINTEGRAL]}, {BATCH_STATEMENT_BLOCK_NONE} {counts[BATCH_STATEMENT_BLOCK_NONE]}, "
             f"{BATCH_STATEMENT_BLOCK_NONINTEGRAL} {counts[BATCH_STATEMENT_BLOCK_NONINTEGRAL]}, {BATCH_STATEMENT_BLOCK_DISAGREES} "
             f"{counts[BATCH_STATEMENT_BLOCK_DISAGREES]}, {BATCH_STATEMENT_VALUE_BESIDE_NULL_BY_NAME} {counts[BATCH_STATEMENT_VALUE_BESIDE_NULL_BY_NAME]} "
             "(nothing is coerced: 8 is not 8.0)"]
    for c in rep.statement_inconsistencies:
        lines.append(f"  {c.line}")
    return lines
