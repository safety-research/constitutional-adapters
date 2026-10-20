"""Registry-driven worklists for the judging drivers (additive, default ``from-disk``).

Every main judging driver gains ``--worklist {registry,from-disk}``:

* ``from-disk`` (still the default) keeps today's disk listing byte for byte and
  prints ONE stderr line, :data:`FROM_DISK_LINE`, so a bypass of the registry is never silent.
* ``registry`` reads the driver's worklist from ``v_pending_judgment`` (:func:`pending.pending_for` with the
  driver's filter), CLAIMS every run before the first judge call (:func:`pending.claim`, refused on a live
  claim by another session — :class:`pending.ClaimHeld` — or on any :class:`pending.ClaimNotFolded` (a run not folded, a correction
  pending, a pair no longer pending), and then nothing is graded and every claim the call took is released), records each run's
  judgment through :func:`hooks.judgment_scope` bound to its claim (``in_progress`` → item rows / metrics →
  ``complete`` + ``run_state judged``), and leaves the completion records SPOOLED under the registry's ``records/``
  for the harvester's next cycle (:meth:`RegistryWorklist.finish`). Until 2026-09-30 ``finish`` ran a FULL harvester
  cycle under ``harvester.lock`` after every driver batch (:func:`pending.ingest_now`): it held the lock the timer and
  the other drivers' claims needed (the harvester lost its own 30-s wait 29 times in a measured hour), re-attempted
  every held revision, and rewrote ``state/harvest_cycle.json`` -- the file the status page's harvester row ages, so a
  dead timer read as alive while drivers ran. The claims are untouched: a claim is written and folded under the lock
  at claim time and is visible to every reader at once. ``finish(fold=True)``, or ``SISTEER_REGISTRY_DRIVER_FOLD=force``
  in the driver's environment, folds under the lock before returning; a held lock is then reported on the returned
  report, never raised -- ``grade``'s closing ``finish`` can never fail a graded batch.

A registry worklist FAILURE — no database, an unknown judge, a run whose store the driver cannot locate —
REFUSES naming the cause (:class:`RegistryUnavailable`); it never degrades to the disk listing.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import db as DB
from . import emit
from . import harvest as H
from . import hooks
from . import identity as I
from . import pending as P
from . import records as R
from . import schema_def as S

#: The claim refusals :meth:`RegistryWorklist.claim_all` releases its just-taken claims on and re-raises -- ``pending.ClaimHeld``
#: (another session holds the run) and ``pending.ClaimNotFolded`` (the run's or the judge's row has not folded yet: nothing written,
#: the caller retries after the harvester folds it). Resolved at import, so this module loads on a tree whose pending module predates
#: ``ClaimNotFolded`` (``ClaimHeld`` then stands in for the missing class and the tuple names one class twice).
#: pending.claim will also refuse with `ClaimPendingCorrection`, a SUBCLASS of ClaimNotFolded, while an identity correction of the run
#: is pending outside its own record files -- refused before anything is written, retry after the next harvester cycle: exactly
#: ClaimNotFolded's semantics. The rule is by TYPE (isinstance / `except`), never an exact class, so every future subclass routes to
#: retry-later by construction: :meth:`RegistryWorklist.claim_all` releases its just-taken claims on and re-raises any ClaimHeld or
#: ClaimNotFolded (or subclass). That class name is kept as ONE token for the readers that name it (the daemon's classifier, this
#: module's tests); nothing here compares a class by its name.
RETRY_LATER_CLAIM_REFUSAL = "ClaimPendingCorrection"
_CLAIM_REFUSALS: Tuple[type, ...] = (P.ClaimHeld, getattr(P, "ClaimNotFolded", P.ClaimHeld))


def is_retry_later_refusal(exc: BaseException) -> bool:
    """Whether ``exc`` means RETRY LATER (nothing written; the caller retries after the next harvester cycle): an instance of ClaimNotFolded or
    any subclass -- the owner's ClaimPendingCorrection included -- never an exact-type test."""
    base = getattr(P, "ClaimNotFolded", None)
    return isinstance(base, type) and isinstance(exc, base)


class StaleRequiredJudge(P.ClaimPrecheckRefused):
    """The worklist row's required judge is no longer the run's designated judge -- a same-id corrective full run record switched the
    run to another registered instrument between the worklist read and the claim. A :class:`pending.ClaimPrecheckRefused` (the
    distinct precheck refusal, itself a :class:`pending.ClaimHeld`), so every driver's existing claim refusal names it,
    :meth:`RegistryWorklist.claim_all` releases what it took, the CLI exits 1 naming the class, and the caller re-selects -- never a
    blind retry. MRO: StaleRequiredJudge -> pending.ClaimPrecheckRefused -> pending.ClaimHeld -> RuntimeError. The message's stable
    leading text ("claim refused: run <id> now designates judge ...") is the grade daemon's `identity_changed` phrase: the refusal
    is transient -- re-selected next tick -- never a quarantine."""

    def __init__(self, run_id: str, current_judge: str, instrument_id: str, row_judge: str) -> None:
        RuntimeError.__init__(self, f"claim refused: run {run_id} now designates judge {current_judge} (instrument {instrument_id}), the worklist "
                                    f"row carried {row_judge}; re-select")
        self.run_id, self.session, self.age_s = run_id, "", 0
        self.reason = f"now designates judge {current_judge} (instrument {instrument_id}), the worklist row carried {row_judge}"
        self.current_judge, self.instrument_id, self.row_judge = current_judge, instrument_id, row_judge


#: What a default-judge row's refusal names when the run's instrument designates NO judge any more (a rule-scored or unregistered
#: instrument): an absent designation is not permission to claim under the judge the row carried
NO_DESIGNATED_JUDGE = "NONE (no designated judge: a rule-scored or unregistered instrument)"


def current_required_judge(conn, run_id: str) -> Tuple[Optional[str], Optional[str]]:
    """``(judge_id, instrument_id)`` of the run's CURRENT instrument -- the designation v_pending_judgment derives (`JOIN instrument i ON
    i.instrument_id = r.instrument_id`; `is_instrument_judge = (i.judge_id = jd.judge_id)`); ``(None, instrument)`` for a rule-scored instrument,
    ``(None, None)`` for an unknown run."""
    row = conn.execute("SELECT i.judge_id, r.instrument_id FROM run r LEFT JOIN instrument i ON i.instrument_id = r.instrument_id WHERE r.run_id = ?",
                       (str(run_id),)).fetchone()
    if row is None:
        return None, None
    return (str(row[0]) if row[0] is not None else None), (str(row[1]) if row[1] is not None else None)


def require_current_judge(conn, row: Mapping[str, object]) -> None:
    """A row selected as the run's INSTRUMENT judge (`is_instrument_judge` 1, or absent: the default worklist's basis) may be claimed
    only while the run's current instrument still designates that judge; a changed designation refuses by name
    (:class:`StaleRequiredJudge`), and so does an ABSENT one (the instrument now names no judge -- rule-scored, or the row /
    instrument is gone -- which is not permission to claim under the judge the row carried). Meant to run INSIDE pending.claim's
    locked pre-write path (its ``precheck``), on the claim's own connection. A row selected for a judge that was never the
    instrument's (a second judge, `is_instrument_judge` 0) is not checked: its judge was the caller's own choice, not the instrument's."""
    if not bool(row.get("is_instrument_judge", 1)):
        return
    rid, want = str(row["run_id"]), str(row["required_judge_id"])
    current, instrument = current_required_judge(conn, rid)
    if current is None:
        raise StaleRequiredJudge(rid, NO_DESIGNATED_JUDGE, str(instrument), want)
    if current != want:
        raise StaleRequiredJudge(rid, current, str(instrument), want)

WORKLIST_CHOICES: Tuple[str, str] = ("registry", "from-disk")
#: Today's behaviour. This flips to ``registry`` after the backfill and a zero-difference
#: ``registry worklist-parity`` run.
DEFAULT_WORKLIST = "from-disk"
FROM_DISK_LINE = "[registry] worklist=from-disk (registry bypassed)"
ROOT_ENV = "REGISTRY_ROOT"
#: :meth:`RegistryWorklist.finish`'s fold switch (2026-09-30): unset / ``skip`` / ``0`` -- the default -- folds nothing (the completion
#: records are spooled and the harvester folds them within one cycle); ``force`` / ``fold`` / ``1`` folds them under the write lock first,
#: the same as ``finish(fold=True)``. A lock still held after the wait is reported on the returned :class:`FinishReport`, never raised.
FOLD_ENV = "SISTEER_REGISTRY_DRIVER_FOLD"
FOLD_VALUES: Tuple[str, ...] = ("force", "fold", "1")
SPOOLED_NOTE = "completion records spooled under the registry's records/; the harvester folds them within one cycle"


class RegistryUnavailable(RuntimeError):
    """``--worklist registry`` cannot be served; the driver refuses (never the disk listing instead)."""


def default_session() -> str:
    return emit.session_from_env() or f"{os.environ.get('USER', 'user')}@{os.uname().nodename}"


def fold_requested() -> bool:
    """Whether :data:`FOLD_ENV` asks :meth:`RegistryWorklist.finish` to fold under the write lock (default: no)."""
    return (os.environ.get(FOLD_ENV) or "").strip().lower() in FOLD_VALUES


@dataclass
class FinishReport(H.IngestReport):
    """What :meth:`RegistryWorklist.finish` returns: the harvester's report when it folded (``folded = True``), else the zero report with
    ``note`` saying why nothing was folded here -- the default (the records are spooled), or a requested fold that met a held lock."""
    folded: bool = False
    note: str = SPOOLED_NOTE

    def line(self) -> str:
        """The one line a driver prints for its registry fold."""
        if self.folded:
            return f"[registry] folded {self.records} record(s); refused={self.refused} blocked={self.blocked}"
        return f"[registry] {self.note}"


def add_worklist_arguments(ap: argparse.ArgumentParser, *, default: str = DEFAULT_WORKLIST, judge: bool = True,
                           campaign: bool = True) -> None:
    """The flags every judging driver shares (a driver that already has ``--campaign`` passes ``campaign=False``)."""
    g = ap.add_argument_group("registry worklist (results registry, plan §8.3)")
    g.add_argument("--worklist", choices=list(WORKLIST_CHOICES), default=default,
                   help=f"where the worklist comes from: 'registry' (v_pending_judgment; claims each run, records the "
                        f"judgment, writes judged) or 'from-disk' (today's listing; logged as a registry bypass). Default {default}")
    g.add_argument("--registry-root", default=None, help=f"the registry root (default ${ROOT_ENV}, else ~/registry)")
    g.add_argument("--session", default=None, help="the claiming session (default $SISTEER_SESSION_NAME, else $USER@host)")
    if judge:
        g.add_argument("--judge", action="append", default=None, metavar="JUDGE",
                       help="judge_id (or seeded readable id) the worklist is pending for; repeatable; default: each run's "
                            "instrument judge")
    if campaign:
        g.add_argument("--campaign", default=None, help="restrict the registry worklist to one campaign")
    # the grading daemon's selectors: exactly the selected runs, and the store map
    # that relocates a compute node's recorded store path onto the grading host's mirror of that cluster
    g.add_argument("--run-id", action="append", default=None, metavar="RUN_ID",
                   help="grade exactly this pending run (repeatable); a pending run not listed is never claimed")
    g.add_argument("--model-id", action="append", default=None, metavar="MODEL_ID",
                   help="restrict the registry worklist to runs of this model_id (repeatable)")
    g.add_argument("--store-root-map", action="append", default=None, metavar="SUBSTRATE=REMOTE_ROOT=LOCAL_ROOT",
                   help="relocate a run's recorded store_uri from the cluster's root to the grading host's mirror (repeatable; "
                        "the recorded URI stays the provenance)")


def log_from_disk() -> None:
    """The one line the from-disk branch prints (stderr): the registry was bypassed, on purpose, visibly."""
    print(FROM_DISK_LINE, file=sys.stderr, flush=True)


def store_path(row: Mapping[str, object], store_root_map: Optional[P.StoreRootMap] = None) -> str:
    """A pending row's store as a filesystem path ('' when it carries none), relocated through ``store_root_map`` when given."""
    return P.store_path_of(row, store_root_map)


def selectors_from_args(args) -> Dict[str, object]:
    """The :class:`RegistryWorklist` keyword arguments the shared flags carry: ``run_ids``, ``model_ids``, ``store_root_map``
    (parsed; a malformed spec refuses by name as :class:`RegistryUnavailable`)."""
    try:
        srm = P.parse_store_root_map(getattr(args, "store_root_map", None))
    except ValueError as exc:
        raise RegistryUnavailable(str(exc)) from None
    return {"run_ids": getattr(args, "run_id", None) or None, "model_ids": getattr(args, "model_id", None) or None,
            "store_root_map": srm or None}


class JudgmentIdentityUnreproducible(ValueError):
    """A same-id judgment record must state the identity inputs OF THE FORM THAT MINTS ITS judgment_id, so the id is reproducible from
    the record's own fields -- the registry refuses a same-id record whose fields reproduce it under none of its three forms. The
    forms (:data:`IDENTITY_FORM_FIELDS`): PLAIN, identity.judgment_id(run, judge) with no session or time (zero fields to state: the
    2026-09 static-attack imports); CLAIMED, uuid5 over run|judge|claimed_by_session|claimed_utc (what pending.claim mints);
    PRODUCER, over run|judge|producer_session|claimed_utc. A judgment row reproducible under NO form cannot be re-filed by id at all:
    the WHOLE batch is refused by this name before anything is written -- a superseded record for a partial in that state would be
    refused at the fold and the retired partial would STAND beside the complete (the duplicate-row bug), so neither half of the
    completion batch is written alone."""

    def __init__(self, judgment_id: str, run_id: str, judge_id: str, forms_tried: Sequence[str]) -> None:
        super().__init__(f"judgment identity unreproducible: judgment {judgment_id} of judge {judge_id} on run {run_id} -- its judgment_id is "
                         f"reproducible from none of the registry's identity forms ({'; '.join(forms_tried)}), so a same-id record for it would "
                         "be refused by the registry (D639) -- nothing written for this (run, judge)")
        self.judgment_id, self.run_id, self.judge_id, self.forms_tried = str(judgment_id), str(run_id), str(judge_id), [str(f) for f in forms_tried]


#: The registry's three judgment-identity forms, in the order they are tried: the name -> the row fields a same-id record states for it
#: (identity.judgment_id's (session, claimed_utc) slots; PLAIN states none).
IDENTITY_FORM_FIELDS: Dict[str, Tuple[str, ...]] = {"plain": (), "claimed": ("claimed_by_session", "claimed_utc"), "producer": ("producer_session", "claimed_utc")}


def judgment_identity_form(row: Mapping[str, object], *, run_id: str, judge_id: str) -> Tuple[str, Dict[str, object]]:
    """WHICH identity form mints ``row``'s judgment_id, and the fields a same-id record states for it -- ``("plain", {})`` when
    identity.judgment_id(run, judge) reproduces it; ``("claimed", {claimed_by_session, claimed_utc})`` / ``("producer", {producer_session,
    claimed_utc})`` when that pair does (a form whose field the row does not state is skipped by name). Refuses by name
    (:class:`JudgmentIdentityUnreproducible`, naming what each form lacked) only when no form reproduces the id."""
    jid = str(row.get("judgment_id") or "")
    tried: List[str] = []
    for form, fields in IDENTITY_FORM_FIELDS.items():
        vals = [row.get(f) for f in fields]
        missing = [f for f, v in zip(fields, vals) if v is None or str(v) == ""]
        if missing:
            tried.append(f"{form}: {', '.join(missing)} not stated")
            continue
        if I.judgment_id(run_id, judge_id, *[str(v) for v in vals]) == jid:
            return form, {f: str(v) for f, v in zip(fields, vals)}
        tried.append(f"{form}: does not reproduce the id")
    raise JudgmentIdentityUnreproducible(jid, run_id, judge_id, tried)


def identity_inputs_of(row: Mapping[str, object], *, run_id: str, judge_id: str) -> Dict[str, object]:
    """The identity inputs a same-id judgment record restates -- exactly the fields of the form that mints the id
    (:func:`judgment_identity_form`; none for the plain form). Stated by ``identity.judgment_identity_inputs(row)`` when the registry's
    identity module has it (it gives the same answer: ``{}`` for plain, the pair otherwise; ``row`` is handed over with ``run_id`` /
    ``judge_id`` added), else by the form found here. The refusal is this module's in either case: a row no form
    reproduces raises :class:`JudgmentIdentityUnreproducible` before any write."""
    _form, mine = judgment_identity_form(row, run_id=run_id, judge_id=judge_id)
    helper = getattr(I, "judgment_identity_inputs", None)
    if callable(helper):
        return dict(helper(dict(row, run_id=run_id, judge_id=judge_id)))
    return mine


def own_partial_judgments(conn, run_id: str, judge_id: str) -> List[Dict[str, object]]:
    """The PARTIAL judgments of (run, judge) -- what a complete judgment by the SAME judge displaces.
    ``[{judgment_id, n_items_judged, finished_utc, claimed_by_session, claimed_utc, producer_session}]`` in finished order; never
    another judge's partial, never an ``in_progress`` claim (a claim is the live grader's, released or completed by it), never a
    complete. The identity fields every form may need ride along: the superseded record restates the ones of the form that mints the
    partial's id, so the id is reproducible from its own fields."""
    rows = conn.execute("SELECT judgment_id, n_items_judged, finished_utc, claimed_by_session, claimed_utc, producer_session FROM judgment "
                        "WHERE run_id = ? AND judge_id = ? AND status = 'partial' ORDER BY finished_utc, judgment_id", (run_id, judge_id)).fetchall()
    return [{"judgment_id": str(r[0]), "n_items_judged": r[1], "finished_utc": r[2], "claimed_by_session": r[3], "claimed_utc": r[4], "producer_session": r[5]}
            for r in rows]


def superseded_judgment_payloads(run_id: str, judge_id: str, complete_id: str, partials: Sequence[Mapping[str, object]],
                                 now: Optional[str] = None, via: str = "D1675 g39-a") -> List[Dict[str, object]]:
    """The ONE shape of a superseded judgment record: per partial, the SAME judgment_id, status ``superseded``,
    ``superseded_by_judgment_id`` = the complete, the identity inputs OF THE FORM THAT MINTS THE PARTIAL'S ID (none for a plain-form
    id, ``claimed_by_session`` + ``claimed_utc`` for a claim-minted one, ``producer_session`` + ``claimed_utc`` for the producer form
    -- the registry refuses a same-id record whose id is not reproducible from its own fields), ``extra.superseded_reason``
    naming the complete and ``extra.superseded_utc`` -- and nothing else, so the partial's other provenance stands under the per-column fold.
    Built here for the grader's scope, the daemon's reconciliation and the one-off correction tool alike (one builder, never a
    re-implementation); ``via`` names the path in the reason. A partial no form reproduces refuses the WHOLE batch by name
    (:class:`JudgmentIdentityUnreproducible`): the list is built in full before any caller writes, so nothing is written."""
    now = now or hooks.utc_now()
    out: List[Dict[str, object]] = []
    for p in partials:
        pid = str(p["judgment_id"])
        if not pid or pid == complete_id:
            continue
        inputs = identity_inputs_of(p, run_id=run_id, judge_id=judge_id)
        reason = (f"superseded by complete judgment {complete_id} of the same judge {judge_id} on run {run_id} at {now} "
                  f"(the grader's own earlier partial, n_items_judged={p.get('n_items_judged')}; {via})")
        out.append({"judgment_id": pid, "run_id": run_id, "judge_id": judge_id, "status": "superseded", "superseded_by_judgment_id": complete_id,
                    **inputs, "extra": {"superseded_reason": reason, "superseded_utc": now}})
    return out


class AtomicBatchWriter(emit.RecordWriter):
    """A :class:`emit.RecordWriter` that can write SEVERAL records of one type as ONE record file, atomically -- the bytes go to a
    temp name in the run directory and are LINKED into place under the next sequence number (the exclusive create every writer relies
    on; a taken number takes the next), so the harvester folds the file whole or never sees it, and a grader dying mid-write leaves a
    `.tmp` nothing reads. Each record carries its own strictly monotone stamp, so their order in the file is their order in the fold
    (the complete judgment before the records it supersedes). :meth:`attach_to_next` parks payloads that ride the NEXT write of their
    record type -- how the superseding scope puts its supersessions into the completion record's own file -- and
    :meth:`write_batch` writes a batch directly (the daemon's reconciliation, the correction tool)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._attached: Tuple[Optional[str], List[Mapping[str, object]]] = (None, [])

    @classmethod
    def from_writer(cls, w: emit.RecordWriter) -> "AtomicBatchWriter":
        return cls(w.spool, w.campaign, w.run_id, w.producer, push_script=w.push_script, disabled=w.disabled, payload_extra=w.payload_extra)

    def attach_to_next(self, record_type: Optional[str], payloads: Sequence[Mapping[str, object]]) -> None:
        """Park ``payloads`` to be written in the SAME file as the next ``write_many`` of ``record_type`` (after its records); ``(None, [])``
        clears. Nothing is written here."""
        self._attached = (record_type, [dict(p) for p in payloads]) if payloads else (None, [])

    def write_many(self, record_type: str, payloads: Sequence[Mapping[str, object]], gz: bool = False,
                   written_utc: Optional[str] = None) -> Optional[str]:
        atype, attached = self._attached
        if atype == record_type and attached:
            self._attached = (None, [])
            return self.write_batch(record_type, list(payloads) + attached, gz=gz)
        return super().write_many(record_type, payloads, gz=gz, written_utc=written_utc)

    def write_batch(self, record_type: str, payloads: Sequence[Mapping[str, object]], gz: bool = False) -> Optional[str]:
        """ONE record file holding every payload (each with its own monotone stamp, in order), written atomically: temp file, fsync, hard link
        under the next free sequence number, temp unlinked; returns the path (None when disabled). A failure anywhere leaves no record file."""
        if self.disabled or not payloads:
            return None
        recs = [R.make_record(record_type, self._stamped(p), self.producer, I.producer_record_id(), self._stamp()) for p in payloads]
        os.makedirs(self.dir, exist_ok=True)
        tmp = os.path.join(self.dir, f".batch-{os.getpid()}-{recs[0]['record_id']}.tmp")
        try:
            R.write_records(tmp, recs, gz=gz, exclusive=True)
            fd = os.open(tmp, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            for _ in range(1000):
                path = os.path.join(self.dir, R.record_file_name(self._next_seq(), record_type, gz=gz))
                try:
                    os.link(tmp, path)
                except FileExistsError:
                    continue
                self.written.append(path)
                return path
            raise RuntimeError(f"AtomicBatchWriter: could not find a free sequence number under {self.dir}")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def spool_judgment_records(root: str, campaign: str, run_id: str) -> List[Dict[str, object]]:
    """Every judgment record payload on the run's spool directory (any status, file order; a torn or unreadable file is skipped) -- the
    durable boundary a supersession's idempotence reads: a record not yet folded still counts as written."""
    d = os.path.join(H.records_dir(root), str(campaign), I.run_dir_name(run_id))
    out: List[Dict[str, object]] = []
    if not os.path.isdir(d):
        return out
    for p in R.iter_record_files(d):
        try:
            for _, rec, _sha in R.read_record_file(p, compute_sha=False):
                if rec.get("record_type") == "judgment" and isinstance(rec.get("payload"), Mapping):
                    out.append(dict(rec["payload"]))
        except Exception:  # noqa: BLE001 -- a torn file (a writer mid-append) is not this reader's refusal
            continue
    return out


def write_supersessions(root: str, *, campaign: str, run_id: str, judge_id: str, complete_id: str, partials: Sequence[Mapping[str, object]],
                        script: str, session: Optional[str], disabled: bool = False, now: Optional[str] = None,
                        via: str = "D1675 g39-a") -> Tuple[Optional[str], List[Dict[str, object]]]:
    """The superseded records of (run, judge) for an EXISTING complete judgment, as ONE atomic record file through :class:`AtomicBatchWriter`
    -- the writer the grader's scope uses -- under the registry's own records directory; ``(path, payloads)``, path None when ``disabled``
    (a dry run: the payloads are built, nothing is written). A partial no identity form reproduces raises
    :class:`JudgmentIdentityUnreproducible` before the writer is built: nothing written for the whole batch."""
    payloads = superseded_judgment_payloads(run_id, judge_id, complete_id, partials, now=now, via=via)
    if not payloads:
        return None, []
    w = AtomicBatchWriter(H.records_dir(root), str(campaign), str(run_id), emit.producer_block(script, session=session), disabled=disabled)
    return w.write_batch("judgment", payloads), payloads


def supersession_candidates(conn, model_ids: Optional[Sequence[str]] = None, campaign: Optional[str] = None) -> List[Dict[str, object]]:
    """Every (run, judge) with a COMPLETE judgment and standing PARTIALS of the same judge -- the legacy half-written state a grader
    dying between its completion and its supersessions left (or any older re-judge). Per pair: the LATEST complete and the
    partials (:func:`own_partial_judgments`), with the run's campaign."""
    sql = ("SELECT c.run_id, c.judge_id, r.campaign, c.judgment_id, c.finished_utc FROM judgment c JOIN run r ON r.run_id = c.run_id "
           "WHERE c.status = 'complete' AND EXISTS (SELECT 1 FROM judgment p WHERE p.run_id = c.run_id AND p.judge_id = c.judge_id AND p.status = 'partial')")
    params: List[object] = []
    if model_ids:
        sql += " AND r.model_id IN (" + ",".join("?" for _ in model_ids) + ")"; params += [str(m) for m in model_ids]
    if campaign:
        sql += " AND r.campaign = ?"; params.append(str(campaign))
    sql += " ORDER BY c.run_id, c.judge_id, c.finished_utc, c.judgment_id"
    latest: Dict[Tuple[str, str], Dict[str, object]] = {}
    for rid, jid, camp, cid, fin in conn.execute(sql, params).fetchall():
        latest[(str(rid), str(jid))] = {"run_id": str(rid), "judge_id": str(jid), "campaign": str(camp), "complete_id": str(cid), "finished_utc": fin}
    out: List[Dict[str, object]] = []
    for (rid, jid), c in sorted(latest.items()):
        c["partials"] = own_partial_judgments(conn, rid, jid)
        out.append(c)
    return out


def reconcile_supersessions(root: str, *, model_ids: Optional[Sequence[str]] = None, campaign: Optional[str] = None, session: Optional[str] = None,
                            script: str = "sisteer.registry.worklist.reconcile_supersessions", dry_run: bool = False) -> List[Dict[str, object]]:
    """Write the superseded records every :func:`supersession_candidates` pair still lacks -- a partial whose superseded record is
    already on the spool (unfolded) or whose status already folded is skipped, so a pass is idempotent and a crash between a grader's
    completion and its supersessions is repaired by the next pass. Returns one entry per (run, judge) written:
    ``{run_id, judge_id, complete_id, partial_ids, path}``; a dry run returns the entries with ``path`` None and writes nothing. A
    pair whose standing partial's judgment_id no identity form reproduces is REFUSED by name
    (:class:`JudgmentIdentityUnreproducible`) -- nothing written for it, the pass goes on to the next pair -- and comes back as an
    entry with ``refused`` = the exception's name and text, ``path`` None, ``partial_ids`` [] and ``refused_partial_ids`` naming the
    partials; it is refused again every pass until the partial is retired by another path (a reader keys on ``refused``, never on
    ``path`` alone)."""
    with DB.reading(DB.db_path(root)) as conn:
        cands = supersession_candidates(conn, model_ids, campaign)
    out: List[Dict[str, object]] = []
    for c in cands:
        spooled = {str(p.get("judgment_id")) for p in spool_judgment_records(root, c["campaign"], c["run_id"]) if p.get("status") == "superseded"}
        todo = [p for p in c["partials"] if str(p["judgment_id"]) not in spooled]
        if not todo:
            continue
        try:
            path, payloads = write_supersessions(root, campaign=c["campaign"], run_id=c["run_id"], judge_id=c["judge_id"], complete_id=c["complete_id"],
                                                 partials=todo, script=script, session=session, disabled=dry_run, via="D1675 g39-a, reconciled by the grade daemon (g39-fix)")
        except JudgmentIdentityUnreproducible as exc:             # nothing written for this (run, judge); said by name, the pass goes on
            print(f"[registry] reconcile_supersessions: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            out.append({"run_id": c["run_id"], "judge_id": c["judge_id"], "complete_id": c["complete_id"], "partial_ids": [], "path": None, "dry_run": dry_run,
                        "refused": f"{type(exc).__name__}: {exc}", "refused_partial_ids": [str(p["judgment_id"]) for p in todo]})
            continue
        out.append({"run_id": c["run_id"], "judge_id": c["judge_id"], "complete_id": c["complete_id"], "partial_ids": [p["judgment_id"] for p in payloads],
                    "path": path, "dry_run": dry_run})
    return out


class SupersedingJudgingScope(hooks.JudgingScope):
    """The :class:`hooks.JudgingScope` the registry worklist hands its drivers, which on ``complete("complete")`` marks the same
    judge's earlier PARTIAL judgments of the run ``superseded``. A re-judge appends under a NEW judgment_id (pending.claim mints one per
    claim) and no path wrote `superseded`, so the registry's currency predicate (schema_def.NON_CURRENT_JUDGMENT_STATUSES) left the
    partial current beside the complete and v_paper_long carried duplicate metric rows (three misaligned_rate rows on one run,
    2026-10-02). ``supersedes`` is the (run, judge)'s partial list read when the scope was built (:func:`own_partial_judgments`; the claim
    is held, so it cannot change under the grader). The completion record and its superseded
    records are ONE atomic record file -- the scope's writer is an :class:`AtomicBatchWriter` and the supersessions ride the completion's
    own write (:meth:`AtomicBatchWriter.attach_to_next`), so the registry folds a complete WITH its supersessions or neither; a grader dying
    between the two, a full spool or a failed second write can no longer leave a covering complete beside still-current partials (the
    daemon's :func:`reconcile_supersessions` repairs the legacy half-written state). Each record is the SAME judgment_id, status
    ``superseded``, ``superseded_by_judgment_id`` = this judgment, ``extra.superseded_reason`` naming it -- and nothing else (the
    partial's own provenance stands under the per-column fold; :func:`superseded_judgment_payloads`) plus the identity inputs of the form that
    mints the partial's id (none for a plain-form id, the claim pair for a claim-minted one -- the registry refuses a same-id record
    whose id is not reproducible from its own fields; a standing partial NO form reproduces refuses the scope by name
    at construction, :class:`JudgmentIdentityUnreproducible`, before any grading or write). A partial filing, a failure, another judge's
    partial and any in_progress claim leave everything as it was."""

    def __init__(self, *args, supersedes: Sequence[Mapping[str, object]] = (), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.writer = AtomicBatchWriter.from_writer(self.writer)        # the same spool, producer and stamps; batches written atomically
        self.supersedes: List[Dict[str, object]] = [dict(p) for p in supersedes
                                                    if str(p.get("judgment_id") or "") not in ("", self.judgment_id)]
        for p in self.supersedes:            # a partial no identity form reproduces refuses the scope by name NOW, before a judge pass is spent
            judgment_identity_form(p, run_id=self.run_id, judge_id=self.judge_id)
        self.superseded: List[str] = []

    def complete(self, status: str = "complete", n_items_judged: Optional[int] = None, n_judge_refusals: Optional[int] = None,
                 n_errors: Optional[int] = None, **more) -> None:
        if self.closed:
            return
        payloads = superseded_judgment_payloads(self.run_id, self.judge_id, self.judgment_id, self.supersedes) if (status == "complete" and self.supersedes) else []
        if payloads:
            self.writer.attach_to_next("judgment", payloads)              # ride the completion record's own file: one atomic write
        try:
            super().complete(status, n_items_judged=n_items_judged, n_judge_refusals=n_judge_refusals, n_errors=n_errors, **more)
        finally:
            self.writer.attach_to_next(None, [])                           # never left for a later write
        if not payloads or not self.completion_emitted:
            return
        for p, rec in zip(self.supersedes, payloads):
            self.superseded.append(str(rec["judgment_id"]))
            print(f"[registry] {self.writer.producer.get('script')}: partial judgment {rec['judgment_id']} (n_items_judged={p.get('n_items_judged')}, "
                  f"session {p.get('claimed_by_session')}) superseded by complete judgment {self.judgment_id} in the completion's own record file "
                  "(D1675 g39-a; g39-fix)", file=sys.stderr, flush=True)


class RegistryWorklist:
    """One driver invocation's registry worklist: judge(s), session, filters, claims, scopes."""

    def __init__(self, root: Optional[str], judges: Optional[Sequence[str]] = None, session: Optional[str] = None,
                 campaign: Optional[str] = None, *, driver: str = "judging", api_pool: Optional[str] = None,
                 stale_hours: float = P.DEFAULT_STALE_HOURS, force: bool = False,
                 run_ids: Optional[Sequence[str]] = None, model_ids: Optional[Sequence[str]] = None,
                 store_root_map: Optional[P.StoreRootMap] = None) -> None:
        self.root = os.path.abspath(root or os.environ.get(ROOT_ENV) or DB.DEFAULT_ROOT)
        self.driver = driver
        self.session = session or default_session()
        self.campaign = campaign
        self.api_pool = api_pool
        self.stale_hours = stale_hours
        self.force = force
        # the selectors: a listed run id set closes the worklist to exactly those runs; a model id set to those models
        self.run_ids: Optional[List[str]] = [str(r) for r in run_ids] if run_ids else None
        self.model_ids: Optional[List[str]] = [str(m) for m in model_ids] if model_ids else None
        self.store_root_map: Optional[P.StoreRootMap] = dict(store_root_map) if store_root_map else None
        self.claims: Dict[Tuple[str, str], Dict[str, object]] = {}
        path = DB.db_path(self.root)
        if not os.path.exists(path):
            raise RegistryUnavailable(f"{driver}: --worklist registry but there is no registry database at {path}; "
                                      "refusing (a registry worklist never falls back to the disk listing)")
        try:
            with DB.reading(path) as conn:
                self.judges: Optional[List[str]] = [P.resolve_judge_id(conn, j) for j in judges] if judges else None
        except P.UnknownJudge as exc:
            raise RegistryUnavailable(f"{driver}: {exc}") from None

    # -- the worklist -------------------------------------------------------------------------
    def pending(self, **filters) -> List[Dict[str, object]]:
        """Pending rows for the requested judge(s) (default: each run's instrument judge) under ``filters``
        (:data:`pending.FILTER_COLUMNS`; ``decode_source_not=`` excludes one decode source), narrowed to this worklist's
        ``run_ids`` / ``model_ids`` when set (a run outside the selection is never returned, so never claimed). Every row gains
        ``substrate`` and ``view_legacy`` (from its ``run`` row), ``padding_tokens`` (from its instrument) and ``store_path`` -- the
        recorded path relocated through the store map when there is one (:meth:`_located`)."""
        if self.run_ids is not None:
            filters.setdefault("run_id", tuple(self.run_ids))
        if self.model_ids is not None:
            filters.setdefault("model_id", tuple(self.model_ids))
        try:
            with DB.reading(DB.db_path(self.root)) as conn:
                if self.judges is None:
                    dsn = filters.pop("decode_source_not", None)
                    rows = P.pending(conn, None, self.campaign, **filters)
                    if dsn is not None:
                        rows = [r for r in rows if r.get("decode_source") != dsn]
                else:
                    rows = []
                    for j in self.judges:
                        rows += P.pending_for(conn, j, self.campaign, **filters)
                return self._without_superseded_siblings(conn, [self._located(conn, r) for r in rows])
        except P.UnknownFilter as exc:
            raise RegistryUnavailable(f"{self.driver}: {exc}") from None

    def _without_superseded_siblings(self, conn, rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """A pending run whose STORE already carries a COMPLETE judgment by its instrument judge on a SUPERSEDED sibling -- another
        run id over the same ``store_uri`` and ``eval_id`` whose ``run.superseded_by_run_id`` is set (the static correction
        re-registers a unit under a rung-keyed run id and the supersede retires the old one) -- is never judged again: the judgment
        re-filed under the new id is what fills it. Skipped by name, one ``[registry]`` line naming the sibling and its judgment;
        every other row stands. A sibling NOT yet superseded skips nothing (that window is the operator's dispatch hold,
        grade_daemon's dispatch.hold); a different eval over the same store (the Petri per-behaviour runs share a cell store) is
        never a sibling here."""
        out: List[Dict[str, object]] = []
        for r in rows:
            store, judge = str(r.get("store_uri") or ""), str(r.get("required_judge_id") or "")
            hit = None
            if store and judge:
                # bound to the DESIGNATED successor -- the old run's superseded_by_run_id IS this pending run -- over the same store,
                # eval, campaign and model; any other pending run over the same store (another correction, rerun or rung
                # registration) is not hidden by the old judgment
                # the sibling's judgment must be IDENTITY-CURRENT under the OLD run (schema_def.judgment_identity_current_sql over
                # the sibling and ITS instrument): a complete judgment claimed under instrument X on a run since corrected to Y is
                # stale-bound -- it covers nothing anywhere and suppresses nothing here either, so the successor stays in the
                # worklist instead of being hidden with no refusal
                hit = conn.execute("SELECT s.run_id, s.superseded_by_run_id, j.judgment_id FROM run s JOIN judgment j ON j.run_id = s.run_id "
                                   "LEFT JOIN instrument si ON si.instrument_id = s.instrument_id "
                                   "WHERE s.store_uri = ? AND s.eval_id = ? AND s.run_id <> ? AND s.superseded_by_run_id = ? "
                                   "AND s.campaign = ? AND s.model_id = ? AND j.judge_id = ? AND j.status = 'complete' "
                                   f"AND {S.judgment_identity_current_sql('j', 's', 'si')} ORDER BY j.finished_utc DESC LIMIT 1",
                                   (store, str(r.get("eval_id") or ""), str(r["run_id"]), str(r["run_id"]), str(r.get("campaign") or ""),
                                    str(r.get("model_id") or ""), judge)).fetchone()
            if hit is not None:
                print(f"[registry] {self.driver}: run {r['run_id']} skipped -- its store {store} already carries complete judgment {hit[2]} by "
                      f"{judge} on superseded sibling {hit[0]} (superseded_by {hit[1]}); the judgment re-filed under the new run id fills it, "
                      "never a re-judge (D1254)", file=sys.stderr, flush=True)
                continue
            out.append(r)
        return out

    def _located(self, conn, row: Mapping[str, object]) -> Dict[str, object]:
        """The row located on THIS host: the run's ``substrate`` and ``view_legacy``, its instrument's ``padding_tokens`` (one lookup) and
        ``store_path`` (mapped through :attr:`store_root_map`) -- the ONE enrichment of a pending row, read here by the drivers and by
        the grading daemon (which groups by ``view_legacy`` and routes a padded instrument to the padding family)."""
        out = dict(row)
        hit = conn.execute("SELECT r.substrate, r.view_legacy, i.padding_tokens FROM run r LEFT JOIN instrument i ON i.instrument_id = r.instrument_id "
                           "WHERE r.run_id = ?", (str(row["run_id"]),)).fetchone()
        out["substrate"], out["view_legacy"], out["padding_tokens"] = tuple(hit) if hit else (None, None, None)
        out["store_path"] = P.store_path_of(out, self.store_root_map)
        return out

    def store_path(self, row: Mapping[str, object]) -> str:
        """The row's store on THIS host: the recorded ``store_uri`` relocated through the store map (``''`` when it names none) -- what
        :meth:`pending` already parked on the row as ``store_path``, recomputed so a row from any source reads the same."""
        return P.store_path_of(row, self.store_root_map)

    # -- claims -------------------------------------------------------------------------------
    def claim(self, row: Mapping[str, object]) -> Dict[str, object]:
        """Claim one pending row for its required judge; returns ``{judgment_id, claimed_utc, judge_id}``. A cached claim is returned
        only while it is still THIS session's live claim under the run's CURRENT folded identity (:meth:`_cached_claim_live`): a run
        identity correction folding between :meth:`claim_all` and :meth:`scope` (or across a reused instance), a release or another
        session's takeover evicts it, and the row is claimed again through :func:`pending.claim`, which mints a current-bound claim
        (handing the judging driver the old-bound id spent its work on a judgment every reader discards while the current work stayed
        pending). The claim read back is the identity-current one (:func:`pending.active_claim`) and must be the
        id ``pending.claim`` returned."""
        rid, jid = str(row["run_id"]), str(row["required_judge_id"])
        key = (rid, jid)
        cached = self.claims.get(key)
        if cached is not None:
            if self._cached_claim_live(rid, cached):
                return cached
            self.claims.pop(key, None)
        # the designated-judge comparison runs INSIDE pending.claim's locked pre-write path (its `precheck`: after the parents are
        # checked and the run's own records folded, before any record is written), never as a read closed before the lock --
        # a correction folded in that gap would otherwise yield a durable claim for the old judge
        got = P.claim(self.root, rid, jid, self.session, force=self.force, stale_hours=self.stale_hours,
                      precheck=lambda conn: require_current_judge(conn, row))
        with DB.reading(DB.db_path(self.root)) as conn:
            info = P.active_claim(conn, rid, jid)
        assert info is not None and info["judgment_id"] == got and info["claimed_by_session"] == self.session, (rid, jid, got, info)
        self.claims[key] = {"judgment_id": info["judgment_id"], "claimed_utc": info["claimed_utc"], "judge_id": jid}
        return self.claims[key]

    def _cached_claim_live(self, rid: str, cached: Mapping[str, object]) -> bool:
        """Whether a cached claim is still this session's in-progress judgment bound to the run's current folded identity
        (``schema_def.judgment_identity_current_sql`` over the run and its instrument -- the worklist's own test)."""
        with DB.reading(DB.db_path(self.root)) as conn:
            row = conn.execute("SELECT j.status, j.claimed_by_session, " + S.judgment_identity_current_sql("j", "r", "i") + " FROM judgment j "
                               "JOIN run r ON r.run_id = j.run_id LEFT JOIN instrument i ON i.instrument_id = r.instrument_id "
                               "WHERE j.judgment_id = ? AND j.run_id = ?", (str(cached["judgment_id"]), rid)).fetchone()
        return row is not None and row[0] == "in_progress" and row[1] == self.session and bool(row[2])

    def claim_all(self, rows: Sequence[Mapping[str, object]]) -> Dict[Tuple[str, str], Dict[str, object]]:
        """Claim every row BEFORE any judging (the claim precedes the first API call). ANY claim refusal on a later row --
        a claim held by another session (:class:`pending.ClaimHeld`), a run not folded yet, an identity correction pending, a pair no
        longer pending because another session's completion folded since the selection (every :class:`pending.ClaimNotFolded`,
        :class:`pending.ClaimNotPending` included) -- releases the claims this call just took and re-raises, so a second driver on the
        same run refuses whole, never half-grades a worklist, and never leaves an earlier row's claim orphaned ``in_progress`` with no
        judging behind it (before this only ``ClaimHeld`` released; a ``ClaimNotPending`` on the second row left the first row's claim
        standing, and every later session met it as a held claim). A claim this call took is one whose judgment id was not cached
        before it (a cached claim :meth:`claim` evicted and re-claimed counts as taken). The tuple caught is ``_CLAIM_REFUSALS``
        (ClaimHeld and ClaimNotFolded, by isinstance -- resolved at import -- so every ClaimNotFolded subclass, ClaimPendingCorrection
        and ClaimNotPending alike, routes here by construction)."""
        taken: List[Tuple[str, str]] = []
        try:
            for row in rows:
                key = (str(row["run_id"]), str(row["required_judge_id"]))
                before = self.claims.get(key)
                info = self.claim(row)
                if before is None or before["judgment_id"] != info["judgment_id"]:
                    taken.append(key)
        except (P.ClaimHeld, P.ClaimNotFolded):   # = _CLAIM_REFUSALS: by isinstance, so every ClaimNotFolded subclass (ClaimPendingCorrection, ClaimNotPending) too; the owner's literal, pinned by test_registry_pending_claim_recheck
            for rid, jid in taken:
                P.release(self.root, rid, jid, self.session)
                self.claims.pop((rid, jid), None)
            raise
        return dict(self.claims)

    def release(self, row: Mapping[str, object]) -> bool:
        rid, jid = str(row["run_id"]), str(row["required_judge_id"])
        self.claims.pop((rid, jid), None)
        return P.release(self.root, rid, jid, self.session)

    # -- recording ----------------------------------------------------------------------------
    def scope(self, row: Mapping[str, object], *, store_uri: Optional[str] = None, rubric_sha256: Optional[str] = None,
              declared_degeneration: Sequence[Tuple[str, str]] = hooks.DEFAULT_DECLARED_DEGENERATION) -> "SupersedingJudgingScope":
        """The :class:`SupersedingJudgingScope` that completes this row's CLAIMED judgment (claims it first if needed; a cached claim
        is re-validated against the run's current folded identity by :meth:`claim` and re-claimed when it moved, so the scope never
        completes an old-bound id) and, on a complete,
        supersedes the same judge's earlier partials of the run. Emission is forced on and the spool is the registry's own records
        directory: a registry worklist that cannot record its judgment refuses rather than grading unrecorded (the DISABLED switch wins
        over the forced-on emission, exactly as in :func:`hooks.judgment_scope`, whose resolution this mirrors with the worklist's own
        spool and campaign)."""
        info = self.claim(row)
        if not hooks.emission_enabled(True):
            raise RegistryUnavailable(f"{self.driver}: emission is disabled ({hooks.emit.DISABLED_ENV}); a registry worklist "
                                      "cannot record its judgments — refusing")
        rid, jid = str(row["run_id"]), str(info["judge_id"])
        campaign = str(row.get("campaign") or I.parse_run_id(rid)["campaign"])
        with DB.reading(DB.db_path(self.root)) as conn:
            partials = own_partial_judgments(conn, rid, jid)
        return SupersedingJudgingScope(rid, campaign, jid, api_pool=self.api_pool, session=self.session, spool=H.records_dir(self.root),
                                       script=self.driver, declared_degeneration=declared_degeneration, rubric_sha256=rubric_sha256 or None,
                                       store_uri=store_uri or (str(row.get("store_uri")) if row.get("store_uri") else None),
                                       claim={"judgment_id": info["judgment_id"], "claimed_utc": info["claimed_utc"]}, supersedes=partials)

    def finish(self, fold: Optional[bool] = None) -> FinishReport:
        """The driver's completion records are on the spool; by default NOTHING is folded here (2026-09-30) -- the harvester's next cycle
        folds them within one cadence, and the claims (folded at claim time) stay visible meanwhile. ``fold=True`` (or, when ``fold`` is
        None, :data:`FOLD_ENV` = ``force``) folds them under the write lock first (:func:`pending.ingest_now`); a lock still held after
        the wait is reported on the returned report and named on stderr, never raised: :meth:`grade` calls this from a ``finally``, and a
        lock failure must neither mask the driver's own exception nor fail a fully graded batch (the daemon logged such batches failed with
        every record already spooled)."""
        if not (fold_requested() if fold is None else bool(fold)):
            return FinishReport(folded=False, note=SPOOLED_NOTE)
        try:
            rep = P.ingest_now(self.root)
        except DB.LockHeld as exc:
            note = f"fold requested ({FOLD_ENV}) but the registry's write lock is held ({exc}); the {SPOOLED_NOTE}"
            print(f"[registry] {note}", file=sys.stderr, flush=True)
            return FinishReport(folded=False, note=note)
        return FinishReport(**rep.__dict__, folded=True, note="folded under the write lock")

    def grade(self, rows: Sequence[Mapping[str, object]],
              grade_one: Callable[[Mapping[str, object], hooks.JudgingScope], None], *,
              store_uri_of: Optional[Callable[[Mapping[str, object]], Optional[str]]] = None) -> List[str]:
        """The generic loop: claim all → for each row a scope → ``grade_one(row, scope)`` (its item rows / metrics
        through the scope; an exception files ``partial`` and propagates) → complete → :meth:`finish` (the records stay
        spooled for the harvester unless a fold is asked for; it never raises). Returns the run ids graded."""
        self.claim_all(rows)
        done: List[str] = []
        try:
            for row in rows:
                scope = self.scope(row, store_uri=store_uri_of(row) if store_uri_of else None)
                with scope:
                    grade_one(row, scope)
                done.append(str(row["run_id"]))
        finally:
            self.finish()
        return done


def gated_defense_metric_rows(m: Mapping[str, object], thresholds: Sequence[float]) -> List[Dict[str, object]]:
    """``run_metric`` rows from the jailbreak harness's per-(attack, object) metrics block (``jailbreak_run.
    _gated_defense_metrics``): ``asr`` at every pre-registered threshold (n = goals graded; k = jailbroken at the
    canonical one) and ``mean_best_score``. The producer spelling is kept beside each registry metric id."""
    asr = m.get("asr") if isinstance(m.get("asr"), Mapping) else {}
    n = m.get("n_goals")
    canonical = thresholds[len(thresholds) // 2] if thresholds else None
    rows: List[Dict[str, object]] = []
    for t in thresholds:
        v = asr.get(str(t), asr.get(t))
        if v is None:
            continue
        row: Dict[str, object] = {"metric_id": "asr", "threshold": float(t), "value": float(v), "n": n,
                                  "producer_metric_spelling": f"asr[{t}]"}
        if canonical is not None and float(t) == float(canonical) and m.get("n_jailbroken") is not None:
            row["k"] = int(m["n_jailbroken"])
        rows.append(row)
    if m.get("mean_best_score") is not None:
        rows.append({"metric_id": "mean_best_score", "value": float(m["mean_best_score"]), "n": n,
                     "producer_metric_spelling": "mean_best_score"})
    return rows


def refuse_unlocated(driver: str, rows: Sequence[Mapping[str, object]], located: Mapping[str, object]) -> None:
    """A pending row whose store the driver could not locate is a refusal naming run and store (never dropped)."""
    missing = [r for r in rows if str(r["run_id"]) not in located]
    if missing:
        names = "; ".join(f"{r['run_id']} (store_uri={r.get('store_uri')!r})" for r in missing[:8])
        raise RegistryUnavailable(f"{driver}: {len(missing)} pending run(s) name a store this driver cannot locate under its "
                                  f"inputs — {names}{' …' if len(missing) > 8 else ''}")
