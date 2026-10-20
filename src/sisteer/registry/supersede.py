"""``registry judgment show`` / ``registry judgment supersede``: retire a complete or partial judgment BY HAND, as a
record in the log (``records/supersedes/<run_dir>/``, never a database edit) -- snapshot-checked at the CLI, minimal, identity-bound.

The record is minimal: ``records.REQUIRED_FIELDS["judgment"]`` (``judgment_id``, ``run_id``, ``judge_id``, ``status = superseded``) +
``superseded_by_judgment_id`` (the successor, or null with ``--no-successor``) + the judgment's IDENTITY INPUTS when its id was minted
over a claim (``claimed_by_session`` + ``claimed_utc``, or ``producer_session`` + ``claimed_utc``, copied verbatim from the live row --
:func:`identity.judgment_identity_inputs` picks the form that reproduces the id, and a row no form reproduces refuses by name) + ``extra``
holding only this command's keys under ``extra.supersede`` (and ``extra.allow_cross_run`` when asked). No measurement is copied: the
``full`` fold sets only the stated columns (an omitted column keeps its value and its stamp) and ``extra`` merges additively, so
``n_items_judged``, ``store_uri`` and the producer's own extra keys survive, and the only columns the record can move are the two it
states. The assembled record -- exactly as the writer stamps it -- passes :func:`validate.validate_record` (the identity rule included)
BEFORE anything is written, so a record that would not fold never enters the log.

Guard rails at the CLI (each a :class:`SupersedeRefusal` naming its field, exit 1, nothing written): an unknown ``judgment_id``; a status
outside ``complete`` / ``partial`` (a claim is released, a retired judgment is retired); a successor that is unknown, not ``complete`` /
``partial``, the judgment itself, or of ANOTHER run without ``--allow-cross-run``; neither or both of ``--superseded-by`` / ``--no-successor``;
``relayed`` provenance without ``--relayed-by`` + ``--message-utc`` (``adapters.roster.normalise_provenance``); an empty ``--reason``; a
``written_utc`` not strictly above the live ``status`` stamp (the record would not win the fold); and the SNAPSHOT CHECK: the live rows
of the judgment and its successor are re-read under the write lock immediately before the write and the write is refused if anything
moved since the first read. ``--dry-run`` prints the record and the consequences and writes nothing.

What the fold does with it (``reducer``): the successor's EXISTENCE is the harvester's dependency check (the FK on
``superseded_by_judgment_id``: blocked until it folds, never refused for that); the RELATIONSHIP (``successor.run_id == run_id`` unless
``extra.allow_cross_run``) is checked at fold time and refuses by name -- the same on live ingest and on replay because the identity rule
binds every judgment id to its run. The run's lifecycle moves only when the run is ``judging`` AND the superseded judgment was its last
open claim (the fold tracks the run's ``in_progress`` judgment ids in the run key's own event order): another judge's claim still in
progress keeps the run ``judging``. ``v_pending_judgment`` re-queues the run for the judge iff no complete covering judgment by that judge
remains (``pending_reason = superseded``), and the readers hide the superseded judgment's ``run_metric`` / ``item_result`` rows.

The residual race (no fold-time precondition): between the CLI's final re-read and the fold,
(i) a producer re-file of the judgment stamped LATER than this record wins the fold for ``status`` (back to ``complete`` / ``partial``;
the link stays unless the re-file states it) -- the CLI folds its own record under the lock and REPORTS the outcome (``in_force``),
``registry judgment show`` reveals it afterwards, and the operator re-runs; (ii) a concurrent supersede stamped EARLIER loses to this one
(the link is this record's); both retire the judgment. Whichever record wins, the readers are consistent with the final fold.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from . import db as DB
from . import emit
from . import harvest as H
from . import identity
from . import records as R
from . import validate as VAL

#: ``records/<SUBDIR>/<run_dir>/`` -- where the supersede records go (beside ``records/claims/``, the other hand-driven judgment records).
SUBDIR = "supersedes"
PRODUCER_SCRIPT = "sisteer.registry.cli judgment supersede"
DECISION = "D639"
#: The one key this command adds to the judgment's ``extra_json`` (its own provenance), so no producer key is ever clobbered.
EXTRA_KEY = "supersede"
SUPERSEDABLE_STATUSES: Tuple[str, ...] = ("complete", "partial")
PROVENANCE_KINDS: Tuple[str, ...] = ("user_direct", "relayed")
#: The judgment columns ``show`` prints before the derived fields.
SHOW_COLUMNS: Tuple[str, ...] = ("judgment_id", "run_id", "judge_id", "status", "superseded_by_judgment_id", "n_items_judged", "n_judge_refusals",
                                 "n_errors", "claimed_by_session", "claimed_utc", "producer_session", "store_uri", "finished_utc")


class SupersedeRefusal(ValueError):
    """A refusal by name at the CLI: nothing is written. ``field`` names what refused, ``reason`` says why."""

    def __init__(self, field_name: str, reason: str):
        super().__init__(f"{field_name}: {reason}")
        self.field = field_name
        self.reason = reason


# --------------------------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------------------------


def judgment_row(conn: sqlite3.Connection, judgment_id: str) -> Optional[Dict[str, object]]:
    row = conn.execute("SELECT * FROM judgment WHERE judgment_id = ?", (judgment_id,)).fetchone()
    return None if row is None else dict(row)


def stamps(row: Mapping[str, object]) -> Dict[str, list]:
    """The row's ``field_versions_json`` -- per column, the ``[written_utc, sha]`` of the record that last set it."""
    raw = row.get("field_versions_json")
    return json.loads(str(raw)) if raw else {}


def same_judge_judgments(conn: sqlite3.Connection, run_id: str, judge_id: str, exclude: Optional[str] = None) -> List[Dict[str, object]]:
    """Every judgment of (run, judge) but ``exclude``, with what the readers' coverage test reads: the consequence a supersede has for
    the queue is decided by whether one of these is a complete judgment that covers the run."""
    rows = conn.execute("SELECT judgment_id, status, n_items_judged, n_judge_refusals, n_errors, claimed_by_session FROM judgment "
                        "WHERE run_id = ? AND judge_id = ? ORDER BY judgment_id", (run_id, judge_id)).fetchall()
    return [dict(r) for r in rows if r["judgment_id"] != exclude]


def pending_state(conn: sqlite3.Connection, run_id: str, judge_id: str) -> Optional[Dict[str, object]]:
    """The run's ``v_pending_judgment`` row for this judge, or None when the run is not pending for it."""
    row = conn.execute("SELECT pending_reason, items_missing, required_coverage, claim_session, claim_judgment_id FROM v_pending_judgment "
                       "WHERE run_id = ? AND required_judge_id = ?", (run_id, judge_id)).fetchone()
    return None if row is None else dict(row)


def show(conn: sqlite3.Connection, judgment_id: str) -> Dict[str, object]:
    """One judgment as the operator needs it before (and after) a supersede: the live columns, the stamps of ``status`` and the link, the
    identity form that mints its id (None when no form does), its ``extra_json``, its successor's row, the judgments naming it as their
    successor, the other judgments of (run, judge), and the run's pending state for that judge. Refuses an unknown id by name."""
    row = judgment_row(conn, judgment_id)
    if row is None:
        raise SupersedeRefusal("judgment_id", f"no judgment {judgment_id!r} in this registry")
    fv = stamps(row)
    out: Dict[str, object] = {k: row.get(k) for k in SHOW_COLUMNS}
    out["identity_form"] = identity.judgment_id_form(row)
    out["status_stamp"] = fv.get("status")
    out["successor_stamp"] = fv.get("superseded_by_judgment_id")
    out["extra_json"] = json.loads(str(row.get("extra_json") or "{}"))
    out["record_uri"] = row.get("record_uri")
    successor_id = row.get("superseded_by_judgment_id")
    successor = judgment_row(conn, str(successor_id)) if successor_id else None
    out["successor"] = None if successor is None else {k: successor.get(k) for k in ("judgment_id", "run_id", "judge_id", "status", "n_items_judged")}
    out["predecessors"] = [str(r[0]) for r in conn.execute("SELECT judgment_id FROM judgment WHERE superseded_by_judgment_id = ? ORDER BY judgment_id",
                                                           (judgment_id,)).fetchall()]
    out["same_judge_judgments"] = same_judge_judgments(conn, str(row["run_id"]), str(row["judge_id"]), exclude=judgment_id)
    out["pending"] = pending_state(conn, str(row["run_id"]), str(row["judge_id"]))
    return out


# --------------------------------------------------------------------------------------------
# the record
# --------------------------------------------------------------------------------------------


def identity_inputs(row: Mapping[str, object]) -> Dict[str, object]:
    """The identity inputs the record must re-state (:func:`identity.judgment_identity_inputs`): ``{}`` for a plainly minted id,
    the claim's session + time for a claim-backed one. A live id no form reproduces refuses by name: a record about it could state no
    inputs the validator admits (such a row is a grandfathered legacy record; its producer retires it by re-filing)."""
    inputs = identity.judgment_identity_inputs(row)
    if inputs is None:
        raise SupersedeRefusal("judgment_id", f"the live id {row.get('judgment_id')!r} is reproduced by no identity form over the row's own fields "
                               f"(run_id={row.get('run_id')!r}, judge_id={row.get('judge_id')!r}, claimed_by_session={row.get('claimed_by_session')!r}, "
                               f"claimed_utc={row.get('claimed_utc')!r}, producer_session={row.get('producer_session')!r}): a supersede record "
                               "about it could not carry identity inputs the validator admits; a legacy row like this is retired by its "
                               "producer re-filing it")
    return inputs


def assemble(row: Mapping[str, object], successor_id: Optional[str], reason: str, ruled_by: str, provenance: Mapping[str, object],
             allow_cross_run: bool, now: str, producer: Mapping[str, object], record_id: str) -> Dict[str, object]:
    """The supersede record, VALIDATED (``validate.validate_record``, the identity rule included) and refused by name when it would not
    fold. Minimal: the four required fields, the link, the identity inputs, ``extra.supersede`` (this command's keys) and, when asked,
    ``extra.allow_cross_run`` -- the key the fold-time relationship check reads."""
    payload: Dict[str, object] = {"judgment_id": row["judgment_id"], "run_id": row["run_id"], "judge_id": row["judge_id"],
                                  "status": "superseded", "superseded_by_judgment_id": successor_id}
    payload.update(identity_inputs(row))
    extra: Dict[str, object] = {EXTRA_KEY: {"decision": DECISION, "cli": PRODUCER_SCRIPT, "reason": reason, "ruled_by": ruled_by, "ruled_utc": now,
                                            "provenance": dict(provenance), "status_before": row.get("status"),
                                            "successor_before": row.get("superseded_by_judgment_id")}}
    if allow_cross_run:
        extra["allow_cross_run"] = True
    payload["extra"] = extra
    rec = R.make_record("judgment", payload, producer, record_id, now)
    out = VAL.validate_record(rec, PRODUCER_SCRIPT)
    if isinstance(out, VAL.Malformed):
        raise SupersedeRefusal(out.field or "record", f"the assembled record does not validate: {out.reason}")
    return rec


def _provenance(kind: str, relayed_by: Optional[str], message_utc: Optional[str], now: str) -> Dict[str, object]:
    from .adapters import roster as RO
    if kind not in PROVENANCE_KINDS:
        raise SupersedeRefusal("provenance_kind", f"{kind!r} is not one of {PROVENANCE_KINDS}")
    block: Dict[str, object] = {"kind": kind}
    if relayed_by:
        block["relayed_by"] = relayed_by
    if message_utc:
        block["message_utc"] = message_utc
    try:
        out = RO.normalise_provenance(block, now, where="judgment supersede")
    except RO.RosterRefusal as exc:
        raise SupersedeRefusal("provenance", str(exc)) from exc
    return dict(out or block)


def _read(conn: sqlite3.Connection, judgment_id: str, successor_id: Optional[str],
          allow_cross_run: bool) -> Tuple[Dict[str, object], Optional[Dict[str, object]]]:
    """The judgment's live row and its successor's, with the guard rails that read them."""
    row = judgment_row(conn, judgment_id)
    if row is None:
        raise SupersedeRefusal("judgment_id", f"no judgment {judgment_id!r} in this registry")
    if row.get("status") not in SUPERSEDABLE_STATUSES:
        raise SupersedeRefusal("status", f"judgment {judgment_id} is {row.get('status')!r}: only a complete or partial judgment is superseded by "
                               "hand (an in_progress claim is released with pending-judgment --release; an abandoned or superseded one is retired already)")
    successor = None
    if successor_id:
        successor = judgment_row(conn, successor_id)
        if successor is None:
            raise SupersedeRefusal("superseded_by", f"no judgment {successor_id!r} in this registry: the successor must have folded (file it "
                                   "first, or say --no-successor)")
        if successor.get("status") not in SUPERSEDABLE_STATUSES:
            raise SupersedeRefusal("superseded_by", f"successor {successor_id} is {successor.get('status')!r}: a successor is a complete or partial judgment")
        if successor.get("run_id") != row.get("run_id") and not allow_cross_run:
            raise SupersedeRefusal("superseded_by", f"successor {successor_id} judges run {successor.get('run_id')} while {judgment_id} judges run "
                                   f"{row.get('run_id')}: a successor belongs to the same run unless --allow-cross-run says the ruling means it")
    return row, successor


def _moved(before: Optional[Mapping[str, object]], after: Optional[Mapping[str, object]], what: str) -> List[str]:
    """Which columns of ``what`` differ between the first read and the re-read under the lock (every column, the stamps included)."""
    if before is None and after is None:
        return []
    if before is None or after is None:
        return [f"{what} {'appeared' if before is None else 'vanished'}"]
    return [f"{what}.{k}: {before.get(k)!r} -> {after.get(k)!r}" for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)]


# --------------------------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------------------------


@dataclass
class SupersedeReport:
    judgment_id: str
    run_id: str
    judge_id: str
    status_before: str
    successor_id: Optional[str]
    record: Dict[str, object]
    dry_run: bool
    remaining_same_judge: List[Dict[str, object]] = field(default_factory=list)
    path: Optional[str] = None
    folded: bool = False
    status_after: Optional[str] = None
    successor_after: Optional[str] = None
    note: str = ""

    @property
    def written(self) -> bool:
        return self.path is not None

    @property
    def in_force(self) -> Optional[bool]:
        """After a fold: whether the live row reads this record's two columns (None before any fold)."""
        if not self.folded:
            return None
        return self.status_after == "superseded" and (self.successor_after or None) == (self.successor_id or None)

    def as_dict(self) -> Dict[str, object]:
        return {"judgment_id": self.judgment_id, "run_id": self.run_id, "judge_id": self.judge_id, "status_before": self.status_before,
                "superseded_by_judgment_id": self.successor_id, "dry_run": self.dry_run, "written": self.written, "path": self.path,
                "folded": self.folded, "status_after": self.status_after, "successor_after": self.successor_after, "in_force": self.in_force,
                "remaining_same_judge": self.remaining_same_judge, "note": self.note, "record": self.record}

    def lines(self) -> List[str]:
        out = []
        link = self.successor_id or "none (a re-judge is wanted)"
        if self.dry_run:
            out.append(f"DRY RUN: nothing written. judgment {self.judgment_id} ({self.status_before}, run {self.run_id}) -> superseded, successor {link}")
            out.append(json.dumps(self.record, indent=1, sort_keys=True))
        else:
            out.append(f"wrote {self.path}")
            if self.folded:
                out.append(f"folded: status={self.status_after} successor={self.successor_after} ({'in force' if self.in_force else 'NOT in force'})")
            else:
                out.append("not folded here: the harvester's next cycle folds it")
        rest = ", ".join(f"{j['judgment_id']} ({j['status']}, n_items_judged={j['n_items_judged']})" for j in self.remaining_same_judge) or "none"
        out.append(f"other judgments of this run by this judge: {rest} -- the run is pending for the judge again unless one of them is a "
                   "complete judgment that covers it; the superseded judgment's run_metric and item rows leave the current readings")
        return out


def supersede(root: str, judgment_id: str, superseded_by: Optional[str] = None, no_successor: bool = False, reason: str = "",
              ruled_by: Optional[str] = None, provenance_kind: str = "user_direct", relayed_by: Optional[str] = None,
              message_utc: Optional[str] = None, allow_cross_run: bool = False, dry_run: bool = False, fold: bool = True,
              now: Optional[str] = None, session: Optional[str] = None,
              before_write: Optional[Callable[[], None]] = None) -> SupersedeReport:
    """Supersede ``judgment_id`` by ``superseded_by`` (or with a null link, ``no_successor``), as documented in the module docstring.
    ``fold`` folds the record under the write lock right after writing it (ONE fold-only harvester pass, as ``pending.claim`` does) and
    reports the live row after it; a caller that leaves the fold to the harvester passes ``fold=False``. ``now`` is the record's
    ``written_utc`` (tests pass a synthetic clock); ``before_write`` is a test seam called after the first read and before the lock, so the
    snapshot check can be exercised. Raises :class:`SupersedeRefusal` (nothing written) and ``db.LockHeld`` when the lock stays held."""
    from . import worklist as W
    now = now or R.utc_now()
    reason = (reason or "").strip()
    if not reason:
        raise SupersedeRefusal("reason", "--reason is required: why the judgment is retired (free text, kept on the record)")
    if bool(superseded_by) == bool(no_successor):
        raise SupersedeRefusal("superseded_by", "name the successor with --superseded-by JUDGMENT_ID, or say --no-successor for a null link: "
                               "one of the two, never both or neither")
    if superseded_by == judgment_id:
        raise SupersedeRefusal("superseded_by", f"judgment {judgment_id} cannot succeed itself")
    provenance = _provenance(provenance_kind, relayed_by, message_utc, now)
    who = ruled_by or ("the user" if provenance_kind == "user_direct" else f"the user, relayed by {relayed_by}")
    with DB.reading(DB.db_path(root)) as conn:
        row, successor = _read(conn, judgment_id, superseded_by, allow_cross_run)
        remaining = same_judge_judgments(conn, str(row["run_id"]), str(row["judge_id"]), exclude=judgment_id)
    status_stamp = stamps(row).get("status")
    if status_stamp and not now > str(status_stamp[0]):
        raise SupersedeRefusal("written_utc", f"the record's written_utc {now} is not strictly above the live status stamp {status_stamp[0]}: "
                               "it would not win the fold (a producer record stamped in the future, or a clock behind it); nothing written")
    producer = emit.producer_block(PRODUCER_SCRIPT, code_sha=emit.git_sha(os.path.dirname(os.path.abspath(__file__))),
                                   session=session or W.default_session(), substrate="grading_host")
    rec = assemble(row, superseded_by, reason, who, provenance, allow_cross_run, now, producer, identity.producer_record_id())
    report = SupersedeReport(judgment_id, str(row["run_id"]), str(row["judge_id"]), str(row["status"]), superseded_by, rec, dry_run, remaining)
    if dry_run:
        return report
    if before_write is not None:
        before_write()
    with DB.write_lock(root, timeout_s=30.0):
        conn = DB.open_or_create(DB.db_path(root))
        try:
            row2, successor2 = _read(conn, judgment_id, superseded_by, allow_cross_run)
            moved = _moved(row, row2, "judgment") + _moved(successor, successor2, "successor")
            if moved:
                raise SupersedeRefusal("snapshot", "the live rows moved between the first read and the write: " + "; ".join(moved)
                                       + " -- nothing written; read them again with `registry judgment show` and re-run")
            writer = emit.RecordWriter(H.records_dir(root), SUBDIR, str(row["run_id"]), producer, disabled=False)
            # the payload EXACTLY as the writer stamps it (its central stamps ride under `extra` beside this command's keys) must validate
            # too, so nothing that would not fold is written; the record id and the written_utc are the ones write_many will use
            final = R.make_record("judgment", writer._stamped(rec["payload"]), producer, str(rec["record_id"]), now)
            out = VAL.validate_record(final, PRODUCER_SCRIPT)
            if isinstance(out, VAL.Malformed):
                raise SupersedeRefusal(out.field or "record", f"the stamped record does not validate: {out.reason}")
            path = writer.write_many("judgment", [rec["payload"]], written_utc=now)
            assert path is not None
            written = [(r, sha) for _, r, sha in R.read_record_file(path)]
            assert len(written) == 1 and isinstance(VAL.validate_record(written[0][0], path, written[0][1]), VAL.Validated), path
            report.record = written[0][0]
            report.path = path
            if fold:
                H.ingest_once(root, conn, workers=0, derive_facts=False, min_free=0)
                after = judgment_row(conn, judgment_id) or {}
                report.folded = True
                report.status_after = after.get("status")  # type: ignore[assignment]
                report.successor_after = after.get("superseded_by_judgment_id")  # type: ignore[assignment]
                report.remaining_same_judge = same_judge_judgments(conn, str(row["run_id"]), str(row["judge_id"]), exclude=judgment_id)
                if not report.in_force:
                    report.note = (f"written but NOT in force: the fold left status={report.status_after!r} successor={report.successor_after!r} "
                                   f"-- a record stamped later than {now} won the fold (the documented residual race); `registry judgment show "
                                   f"{judgment_id}` shows the row; re-run to file a later supersede")
        finally:
            conn.close()
    return report
