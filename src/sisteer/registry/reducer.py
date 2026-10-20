"""The replay-equivalent fold.

A run-scale row is ``fold(events of its natural key ordered by (written_utc, tie_rank, record_sha256))`` -- the ORDER KEY
(:data:`ORDER_SQL`, :func:`order_key`), where ``tie_rank`` is 1 for a ``run_state`` event to a lateral or terminal state
(:data:`LATERAL_STATES`: failed, cancelled, superseded, withdrawn) and 0 for every other event, so that at one ``written_utc`` the
lateral event folds AFTER the progression event (see below).
The reducer stores every event in ``row_history`` (``payload_json`` = the event's effect,
``folded_state_json`` = the state after it) and RECOMPUTES the live row from the full ordered
event list on every apply, so:

* live ingest equals chronological replay by construction (the fold never depends on arrival order);
* a late earlier event re-derives the historical SUFFIX (every ``row_history`` row at or after its
  position is rewritten) — ``recompute_history_suffix`` is the same code path;
* per-field ``field_versions_json`` stamps (the FULL order key of the record that last set each
  column) are an OUTPUT of the fold; a duplicate sha is a no-op;
* the lifecycle is MONOTONIC along ``vocab.LIFECYCLE_PROGRESSION`` (:func:`lifecycle_regresses`): a ``state`` event
  never moves a run backwards; a backwards event is folded — its history row stands — and moves nothing;
* two run-scale events of one key at the SAME ``written_utc`` fold in a DETERMINISTIC, content-derived order: a ``run_state``
  to a lateral/terminal state after a progression-state event, then by ``record_sha256``. A declared-store reader emits a
  run's ``judged`` and its ``superseded`` in one batch with one stamp (``declared_common``, 2026-09-29: four already-superseded
  rows flipped ``lifecycle_state`` at every re-fold because the pair's order was the sha draw), and a reactivation re-stamps
  EVERY committed measurement record of a revision to one ``written_utc`` (``runner._reemit_committed``), so an emitter-side
  stamp offset could not have fixed it. The rank is a function of the record's own bytes (``record_type`` and the state it
  names): no log edit, no identity change -- the ``field_versions_json`` stamps stay ``[written_utc, record_sha256]``, which
  still names the record uniquely.

Item-scale tables (``item_result``, ``item_flags``, ``run_degeneration``) take the degenerate case:
whole-row records, greatest order key wins, two stamp columns, no history rows (their history is
the record log; ``fold_item_events`` replays it on demand).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from . import db as DB
from . import records as R
from . import schema_def as S
from . import observations as OBS
from . import site_aliases as SA
from . import vocab as V
from .identity import canonical_json
from .validate import Validated

#: The order key of a run-scale event: ``(written_utc, tie_rank, record_sha256)`` (:func:`order_key`; :data:`ORDER_SQL` is the
#: same order in SQL). The two stamps a row carries (``field_versions_json``) stay ``[written_utc, record_sha256]`` -- the rank is
#: derived from the record's content, so the pair identifies the record and its slot in the order alike.
OrderKey = Tuple[str, int, str]

#: The rank of a lifecycle state along ``V.LIFECYCLE_PROGRESSION`` (queued → running → generated → judged → ingested).
#: ``judging`` is a transient beside ``generated`` (a judgment in progress; the judgment_state branch of :func:`fold`
#: moves generated → judging on a claim and back when the run's LAST open claim ends) and ranks with it. Terminal and
#: lateral states (failed, cancelled, superseded, withdrawn) carry no rank: a state event to or from one always applies.
_LIFECYCLE_RANK: Dict[str, int] = {s: i for i, s in enumerate(V.LIFECYCLE_PROGRESSION)}
_LIFECYCLE_RANK["judging"] = _LIFECYCLE_RANK["generated"]

#: The lifecycle states that carry no rank -- terminal (``failed``, ``cancelled``) and lateral (``superseded``, ``withdrawn``):
#: a ``run_state`` event to one of them takes ``tie_rank`` 1 and folds AFTER a same-stamp progression-state event. ONE definition
#: (``vocab.LIFECYCLE_LATERAL``, shared with ``validate.Validated.order_key``), pinned here to be exactly the unranked states.
LATERAL_STATES: Tuple[str, ...] = V.LIFECYCLE_LATERAL
assert LATERAL_STATES == tuple(s for s in V.LIFECYCLE_STATES if s not in _LIFECYCLE_RANK), (LATERAL_STATES, _LIFECYCLE_RANK)


def tie_rank(record_type: str, payload: Mapping[str, object]) -> int:
    """1 for a ``run_state`` record whose ``state`` is one of :data:`LATERAL_STATES`, else 0 -- from the record's own bytes, the way
    :data:`TIE_RANK_SQL` reads them off ``row_history`` (``record_type`` and ``json_extract(payload_json, '$.state')``). A
    ``run_ingest`` stamp (``ingested``), a ``run_state`` to a progression state, a patch or any other record is 0."""
    return 1 if record_type == "run_state" and str(payload.get("state")) in LATERAL_STATES else 0


#: ``tie_rank`` as a SQL expression over a ``row_history`` row.
TIE_RANK_SQL = ("CASE WHEN record_type = 'run_state' AND json_extract(payload_json, '$.state') IN ("
                + ", ".join(f"'{s}'" for s in LATERAL_STATES) + ") THEN 1 ELSE 0 END")
#: The ORDER BY of every query that orders or compares run-scale events (ascending, and its descending twin): the ONE spelling
#: of the order. Measured on sqlite 3.45 (synthetic ``row_history``, 2026-10-04): the ``LIMIT 2`` seek stays O(1) in the key's
#: history (block sort over the leading ``written_utc`` run, ~10 us against ~6 us; identical at 300 and 100,000 events), the
#: whole-history read ~17 % slower at 300 events (680 us against 580 us).
ORDER_SQL = f"written_utc, {TIE_RANK_SQL}, record_sha256"
ORDER_DESC_SQL = f"written_utc DESC, {TIE_RANK_SQL} DESC, record_sha256 DESC"

#: The fold order's VERSION: the rule's number and its spelling, recorded in the database by the first writer that folds into it
#: (``db.stamp_fold_order``: the ``schema_migration`` row ``version = 0``, ``description`` = this string, ``applied_utc`` = the moment of
#: the stamp) and checked by every writer before it folds (``db.require_fold_order``: ``db.open_or_create``, ``harvest.fold_only_files``, a
#: passed-in connection of ``harvest.ingest_once``, the replica's refold writer ``replica._refold_rows``, and so a claim's fold): a database
#: recording ANOTHER order refuses by name (``db.FoldOrderDrift``, a ``SchemaDrift``), a database recording none is stamped with this one and
#: folded (the bootstrap -- every database that predates the stamp, the live one included), and readers never check. Why: the order is a
#: property of the ROWS, not of the DDL, and two orders in one database cannot be told apart afterwards -- the live database carried rows
#: folded under :data:`FOLD_ORDERS` ``[1]`` by a harvester whose checkout took the tie-break commit of 2026-10-04 only on 2026-10-06, with a
#: judgment daemon on an older checkout folding beside it until 2026-10-08, and nothing in the database said so until a landing's replica
#: check met same-stamp movers (``replica.residual_same_stamp_class`` lists what that left). The string IS the version: a new rule takes
#: a new number and spelling together, and the spelling is never edited alone.
FOLD_ORDER_VERSION = "fold_order/2: written_utc, tie_rank, record_sha256"
#: Every order the registry has folded run-scale rows under, by number: 1 the pre-tie-break order (``(written_utc, record_sha256)``: the
#: sha draw decided a same-stamp pair), 2 the current one (:data:`FOLD_ORDER_VERSION`). A refusal names both sides by these strings.
FOLD_ORDERS: Dict[int, str] = {1: "fold_order/1: written_utc, record_sha256", 2: FOLD_ORDER_VERSION}


def lifecycle_regresses(current: Optional[object], target: str) -> bool:
    """True when a ``state`` event would move a run BACKWARDS along the progression: the lifecycle is monotonic in the
    fold. A legacy-store reader re-emitting its rows at a new EMISSION_VERSION re-sends the run's ``judged`` (a rule-scored
    run's ``generated``); ordered by ``written_utc`` that event folds AFTER the harvester's ``run_ingest`` stamp and, applied,
    left the row at ``judged`` with an ``ingested_utc`` (2,929 live runs on 2026-09-21, 14 of them rule-scored runs the paper
    admits only at ``ingested``). :func:`fold` records such an event (its history row stands, provenance intact) and moves
    nothing; :func:`lifecycle_regressions_held` counts them for the cycle's report."""
    a = _LIFECYCLE_RANK.get(str(current)) if current is not None else None
    b = _LIFECYCLE_RANK.get(target)
    return a is not None and b is not None and b < a


@dataclass(frozen=True)
class LedgerOutcome:
    """What :meth:`Reducer.apply_ledger_fact` did with one ledger-fact record: ``inserted`` (a new row), ``duplicate`` (the row stands;
    for an ordered table an identical payload under an earlier key moved the key columns alone), ``replaced`` (an ordered table: the
    incoming record sorts earlier than the standing row and took its place; ``displaced`` names the filing it displaced, (sha, uri), for the
    harvester to refuse by name), ``conflict`` (an ordered table: the incoming record sorts later and its payload differs in
    ``conflict_fields``; nothing written, the harvester refuses it by name). ``kept_sha`` is the record sha the row now carries."""
    status: str
    displaced: Optional[Tuple[str, str]] = None
    conflict_fields: Tuple[str, ...] = ()
    kept_sha: Optional[str] = None
    kept_changed: bool = False            # the row now carries another filing's key than before (inserted, replaced, or the key moved): the
                                          # harvester judges every other indexed record of the key again against the new kept filing

    @property
    def applied(self) -> bool:
        return self.status in ("inserted", "replaced")


class VerifyMismatch(ValueError):
    """A run_patch's ``verify`` value differs from the run row (Malformed refusal)."""

    def __init__(self, field_name: str, expected, actual, run_id: str):
        super().__init__(f"run_patch.verify[{field_name!r}] = {expected!r} but run {run_id} has {actual!r}")
        self.field = field_name


class LifecycleEvidenceMissing(Exception):
    """A ``run`` lifecycle event (a ``run_state``, or the harvester's ``run_ingest``) whose REFOLD reaches a state without the ``*_utc``
    of an earlier state in the progression -- the evidence ``trg_run_state_utc`` requires has not arrived IN THE FOLD'S ORDER, so the
    harvester blocks the record on it (parent ``run_state_event <run_id>:<state>``, :meth:`parent`) exactly as
    ``harvest.Ingestor._missing_lifecycle_evidence`` blocks a state whose evidence is absent from the ROW, and never refuses it.

    The two checks differ when the row carries the evidence but the history does not carry it BEFORE the new event: a ``judged``
    folded from events stamped after an ``ingested`` regresses in the fold and is held (:func:`lifecycle_regresses`), so the refold
    leaves ``judged_utc`` unset while the row still shows it, and the live-row upsert refuses -- SQLite fires a BEFORE INSERT trigger
    on the INSERT half of an upsert over the values being inserted, a column the existing row carries notwithstanding. The R3
    bootstrap (2026-10-05) refused 128 ``run_ingest`` records this way: their first carrier revision was held in the replay's main
    pass and activated only in the closing blocked-queue pass, after the retry of the stamp had refolded the key over the later
    carriers alone; live had never met that order (the stamp is derived after the run's own records folded), so live refused
    0. Blocking instead of refusing makes the outcome independent of the application order: the record is re-attempted when the
    missing evidence lands (the queue never declares a ``run_state_event`` parent absent), and a replay folds the row live folded.
    Deliberately not a ``ValueError``: the harvester's refusal clause must not catch it."""

    def __init__(self, run_id: str, state: str, missing_columns: Sequence[str]):
        self.run_id, self.state, self.missing_columns = run_id, state, list(missing_columns)
        super().__init__(f"run {run_id}: the fold reaches {state!r} without " + ", ".join(self.missing_columns)
                         + " -- the evidence of an earlier lifecycle state has not arrived in fold order (blocked on it, not refused)")

    def parent(self) -> Tuple[str, str]:
        """The pseudo-parent the harvester blocks on: ``("run_state_event", "<run_id>:<state whose utc is missing>")``, named the way
        ``_missing_lifecycle_evidence`` names it (the first missing column, in progression order)."""
        by_col = {col: st for st, col in V.LIFECYCLE_UTC_COLUMN.items()}
        col = self.missing_columns[0]
        return "run_state_event", f"{self.run_id}:{by_col.get(col, col)}"


def lifecycle_evidence_missing(state: Mapping[str, object]) -> List[str]:
    """The ``run.*_utc`` columns a folded state lacks for its ``lifecycle_state`` (``vocab.LIFECYCLE_REQUIRED_UTC``, the trigger's own
    requirement): ``[]`` when the row the fold describes would pass ``trg_run_state_utc``."""
    required = V.LIFECYCLE_REQUIRED_UTC.get(str(state.get("lifecycle_state")), ())
    return [c for c in required if state.get(c) is None]


class SupersedeRelationshipMismatch(ValueError):
    """A ``judgment`` record folding as ``superseded`` names a successor of ANOTHER run without ``extra.allow_cross_run`` (Malformed
    refusal naming ``superseded_by_judgment_id``)."""

    def __init__(self, judgment_id: str, run_id: str, successor_id: str, successor_run_id: str):
        super().__init__(f"judgment {judgment_id} of run {run_id} names successor {successor_id} of run {successor_run_id}: a successor "
                         "belongs to the same run unless the record carries extra.allow_cross_run = true (registry judgment supersede "
                         "--allow-cross-run)")
        self.field = "superseded_by_judgment_id"


@dataclass
class Effect:
    table: str
    key: Dict[str, object]           # natural key columns
    payload: Dict[str, object]       # effect payload (columns to set / state info)
    kind: str                        # full | patch | state | judgment_state | ruling_eval
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def key_json(self) -> str:
        return canonical_json(self.key)

    def history_payload(self) -> Dict[str, object]:
        p = dict(self.payload)
        p["_kind"] = self.kind
        if self.extra:
            p["_extra"] = self.extra
        return p


# --------------------------------------------------------------------------------------------
# record -> effects
# --------------------------------------------------------------------------------------------


def _key_of(table: str, payload: Mapping[str, object]) -> Dict[str, object]:
    """The effect's natural key. A site label in it (``mirror_ack.destination``, the mirror's key) is the label whichever spelling the
    record carries (``site_aliases.canonical_row``), so one row's events share one ``row_history`` key, the key its live row has."""
    key = {}
    for col in R.NATURAL_KEYS[table]:
        if col in payload:
            key[col] = payload[col]
        elif col in S.SENTINELS:
            key[col] = S.SENTINELS[col]
        else:
            raise KeyError(f"{table}: natural key column {col!r} absent from payload")
    return SA.canonical_row(table, key)


def effects_of(v: Validated) -> List[Effect]:
    rt = v.record_type
    p = v.payload
    if rt in R.ITEM_SCALE_RECORD_TYPES:
        table = R.RECORD_TYPE_TABLE[rt]
        return [Effect(table, _key_of(table, p), dict(p), "full", dict(v.extra))]
    if rt == "run_state":
        ev = dict(p)
        ev.setdefault("evidence_uri", "")
        effs = [Effect("run_state_event", _key_of("run_state_event", ev), ev, "full", dict(v.extra))]
        effs.append(Effect("run", {"run_id": p["run_id"]},
                           {"state": p["state"], "event_utc": p["event_utc"]}, "state"))
        return effs
    if rt == "run_patch":
        fields = dict(p.get("fields") or {})
        extra = dict(v.extra)
        if p.get("verify"):
            extra["verified_fields"] = dict(p["verify"])
        return [Effect("run", {"run_id": p["run_id"]}, fields, "patch", extra)]
    if rt == "run_ingest":
        fields: Dict[str, object] = {"state": "ingested", "event_utc": p["ingested_utc"]}
        if p.get("judged_utc"):
            fields["judged_utc"] = p["judged_utc"]      # a rule-scored run's scoring time (records.EFFECTS)
        return [Effect("run", {"run_id": p["run_id"]}, fields, "state")]
    if rt == "observation":
        effs = []
        for k in p["observation_keys"]:
            key_json = k if isinstance(k, str) else canonical_json(k)
            row = {"source_id": p["source_id"], "revision_id": p["revision_id"], "observation_key": key_json,
                   "table_name": p["table_name"], "current": 1}
            effs.append(Effect("observation", _key_of("observation", row), row, "full"))
        return effs
    if rt == "item_withdrawal":
        # one row per named key: every event carries the same column set (the optional columns as None), so the key's
        # fold is the latest record naming it and the appended-fold shortcut applies
        effs = []
        for k in p["items"]:
            row = {"run_id": p["run_id"], **{f: k[f] for f in R.ITEM_WITHDRAWAL_KEY_FIELDS}, "reason": p["reason"],
                   "evidence_uri": p.get("evidence_uri"), "belongs_to_run_id": p.get("belongs_to_run_id")}
            effs.append(Effect("item_withdrawal", _key_of("item_withdrawal", row), row, "full", dict(v.extra)))
        return effs
    table = R.RECORD_TYPE_TABLE[rt]
    effs = [Effect(table, _key_of(table, p), dict(p), "full", dict(v.extra))]
    if rt == "judgment":
        effs.append(Effect("run", {"run_id": p["run_id"]},
                           {"judgment_id": p["judgment_id"], "status": p["status"]}, "judgment_state"))
    if rt == "ruling_state":
        effs.append(Effect("ruling", {"ruling_id": p["ruling_id"]},
                           {"evaluated_utc": p["evaluated_utc"], "evidence_state": p["evidence_state"],
                            "expired": int(p["expired"])}, "ruling_eval"))
    return effs


# --------------------------------------------------------------------------------------------
# the fold
# --------------------------------------------------------------------------------------------

@dataclass
class HistoryEvent:
    written_utc: str
    sha: str
    record_type: str
    record_uri: str
    payload: Dict[str, object]     # history payload (with _kind / _extra)

    @property
    def kind(self) -> str:
        return str(self.payload.get("_kind", "full"))

    @property
    def tie_rank(self) -> int:
        return tie_rank(self.record_type, self.payload)


def order_key(ev: HistoryEvent) -> OrderKey:
    """The event's slot in the fold: ``(written_utc, tie_rank, record_sha256)`` -- ``sorted(events, key=order_key)`` is the order
    :meth:`Reducer._load_events` reads from SQL (:data:`ORDER_SQL`), pinned equal by test."""
    return (ev.written_utc, ev.tie_rank, ev.sha)


def lift_object_fields(table: str, payload: Dict[str, object], ev_extra: object) -> Dict[str, object]:
    """The lift, applied to ONE event's payload before it folds: for the ``object`` table, every column of
    ``records.OBJECT_LIFTED_FROM_EXTRA`` the event states no column for but carries (non-null) under its OWN parked extra is set from that
    key, with this event's stamp -- an old record that wrote `extra.named_role` / `extra.parent_object_id` folds the column exactly as a new
    record stating it would. Per-event, never from the accumulated extra: an explicit null in a later record resets the column, and a still
    later record that is silent on it leaves the reset standing. The key stays in `extra_json` (the extra accumulation is untouched) and
    `row_history.payload_json` keeps the record as it was validated, so replay is byte-equal. Any other table: the payload unchanged."""
    if table != "object" or not isinstance(ev_extra, dict):
        return payload
    for col in R.OBJECT_LIFTED_FROM_EXTRA:
        if col not in payload and ev_extra.get(col) is not None:
            payload[col] = ev_extra[col]
    return payload


#: The record-level marker a non-statement rides on: ``extra.unstated_columns = [<column>, …]`` on an ``object`` record
#: names columns whose values the record CARRIES (they are NOT NULL / required, so it cannot omit them) but does not STATE. The
#: fold applies a listed column only when no row exists yet for the key -- the record supplies the initial value -- and skips it
#: when one does: the column keeps its folded value and stamp, whatever the stamps' order -- and so does its PROVENANCE: for each
#: skipped column ``c`` the event's ``field_confidence_json[c]`` entry and its ``extra.<c>_origin`` key are not applied either (the
#: standing statement's confidence and origin stay), so a row never claims evidence for a value it does not carry; and the converse
#: holds: an event that EXPLICITLY states ``c`` replaces the accumulated ``extra.<c>_origin`` with its own ``<c>_origin`` when it
#: carries one and REMOVES it otherwise (:func:`_origin_follows_statement`), so the origin always describes the standing statement
#: (a bank-created row's ``bank_default`` used to outlive the producer's later ``reference``). The marker is
#: a fold directive, not a park: it never lands in ``extra_json`` (a later record without it would otherwise inherit a stale
#: "unstated" from the fold's ``extra.update``). The validator (``validate._check_unstated_columns``) admits it on the tables in
#: :data:`UNSTATED_TABLES`, every name a column of that table other than the key and present in the payload. The objects_bank adapter
#: marked ``status`` unstated whenever it did not read one from the bank's own evidence (it now omits the column instead): a bank
#: record is a MIRROR of the object, and whether an object is ``approved`` or the paper's ``reference`` is the producer's or a ruling's
#: statement -- without the marker a bank record stamped AFTER a producer's but folded BEFORE it (a spooled producer record, a cluster
#: mirror) kept the demoted status until a correction.
#:
#: ``run`` (2026-10-07): the stats readers of a store (``petri_stats``, and the same shape is open to
#: ``multiturn_grades`` / ``bon_stores``) mint the export's run id and file per-run statistics onto it; the run ROW's provenance --
#: ``lane_source_legacy``, the clock columns, ``store_uri``, ``serving_path``, the counts, ``field_confidence_json`` -- belongs to the row's
#: owner: the registration owns a run's identity / provenance columns, and the export owns the paper rows. Two full
#: records restating one row per-column-latest flipped the live row's ``lane_source_legacy`` / ``generated_utc`` / ``store_uri`` /
#: ``serving_path`` (546 of 749 shared Petri runs) on every re-read of either adapter, by nothing but the write order -- the same
#: ``written_utc`` resolved by record sha. A stats reader therefore marks every non-identity column it carries
#: (``lineage_common.stats_reader_run_record``), the ones it carries as null included (a withdrawal, below). A marked column over an EXISTING row applies
#: only where the standing row has NO value for it -- the reader's sidecar evidence (``serving_identity``, ``n_items_planned``: columns
#: the export omits) fills a hole the owner left, the way ``ownership.CARRY_COLUMNS`` let a secondary contribute where the owner has none --
#: and never overwrites a value. For ``object.status`` (NOT NULL) the fill case cannot arise, so the object rule is unchanged.
UNSTATED_COLUMNS_KEY = "unstated_columns"
UNSTATED_TABLES: Tuple[str, ...] = ("object", "run")

#: The UNION columns: the JSON-LIST columns, per table, whose entries EVERY ``full`` / ``patch`` record of the table ADDS to the standing
#: list rather than restating it whole -- `artifact.locations_json`, the places a file's bytes are known to be. A full record overwrites every
#: column it states, so two records of one `artifact` that each named "the standing locations plus mine" -- two analyses filing one known
#: digest from two cluster paths before a fold, each merged by the filer against the same database row -- folded to the second record's list
#: alone: the first path was lost, and a rebuild replayed the same payloads to the same loss (the fourth review round of the analysis filing,
#: 2026-10-09). That round made the union a per-record DIRECTIVE, ``extra.union_columns = ["locations_json"]``, carried by the publish
#: writers alone; a record without it still replaced the list, so an adapter revision re-registering an unchanged input -- the adapter's
#: record names its own locations and nothing more -- erased the location an analysis had filed for the same bytes, and the filed copy
#: then refused to publish as unregistered (the fourth closing pass, 2026-10-09). The union is now the fold's rule for the column, from
#: every producer: adapter, hook, status page, publish, a record written by hand. The fold of a union column is :func:`union_json_list`
#: over the state before the event and the event's list: an entry whose key (``uri`` of a location, the string itself of a bare path, the
#: canonical JSON of anything else) is new is appended, a VERIFIED entry (a location with a non-null `verified_utc`) replaces the standing
#: entry of its key, an unverified one never does -- a pure function of the record and the prior state (:func:`union_payload`), so a replay
#: gives the same row whatever the arrival order, and the row's list is the union of every record's in stamp order. A record carrying the
#: column as NULL over a standing list says nothing about the list (the column leaves the payload: the list and its stamp stand); over no
#: standing value it folds null, as a first record always did. No producer narrows a list on purpose: a location that no longer holds the
#: bytes is the readers' concern (`publish.registered_shas` lets the bytes at a path pick the registration), so there is no replacement
#: directive. The marker ``extra.union_columns`` stays ADMITTED by the validator (``validate._check_union_columns``: on the tables below
#: alone, naming their union columns, each carried as a list) and names what the fold does anyway; it is a directive, never a park, and
#: never lands in ``extra_json``. Landing: a live `artifact` row whose later record dropped a location an earlier record named folds
#: differently under this rule than it stood -- such rows are movers of the landing's refold-compare, and the row after the refold holds
#: every location the log ever named.
UNION_COLUMNS_KEY = "union_columns"
UNION_COLUMNS: Dict[str, Tuple[str, ...]] = {"artifact": ("locations_json",)}


def union_payload(table: str, payload: Mapping[str, object], state: Mapping[str, object]) -> Dict[str, object]:
    """The ``full`` / ``patch`` payload with every union column of ``table`` it carries (:data:`UNION_COLUMNS`) folded over the standing
    value: the column becomes :func:`union_json_list` of the state before the event and the event's list (canonical JSON text). A column
    carried as NULL over a standing value leaves the payload -- a null adds nothing, so the list and its stamp stand -- and over no standing
    value stays null. A table with no union column returns the payload unchanged. Shared by :func:`fold` and the appended-fold shortcut
    (``Reducer._appended_fold``), so the two paths agree byte for byte."""
    cols = UNION_COLUMNS.get(table)
    if not cols:
        return dict(payload)
    out = dict(payload)
    for c in cols:
        if c not in out:
            continue
        if out[c] is None:
            if state.get(c) is not None:
                out.pop(c)
            continue
        out[c] = union_json_list(state.get(c), out[c])
    return out


def location_key(entry: object) -> str:
    """The identity of one list entry under the union: a location's ``uri``, a bare string itself, the canonical JSON of anything else."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and isinstance(entry.get("uri"), str):
        return str(entry["uri"])
    return canonical_json(entry)


def _verified_entry(entry: object) -> bool:
    return isinstance(entry, dict) and entry.get("verified_utc") is not None


def union_entries(standing: Sequence[object], incoming: Sequence[object]) -> List[object]:
    """The standing list with the incoming entries folded in by :func:`location_key`: a new key is appended in the incoming order, a
    verified entry replaces the standing entry of its key in place, an unverified one leaves it. Idempotent (folding a list into itself
    gives the list); the caller's lists are not modified."""
    out: List[object] = list(standing)
    index: Dict[str, int] = {location_key(e): i for i, e in enumerate(out)}
    for e in incoming:
        k = location_key(e)
        at = index.get(k)
        if at is None:
            index[k] = len(out)
            out.append(e)
        elif _verified_entry(e):
            out[at] = e
    return out


def union_json_list(standing: object, incoming: object) -> str:
    """:func:`union_entries` over two JSON-list column values as the fold holds them (canonical JSON text, or None / a non-list for a
    column nothing has set, read as the empty list); returns canonical JSON text, byte-equal to the standing text when nothing was added.
    A non-list INCOMING value never reaches the fold from a validated record: the validator refuses a union column carried in any shape
    but a JSON list by name, marker or none (``validate._check_union_column_shapes``; the seventh closing pass, 2026-10-09 -- before it a
    hand-written artifact record carrying its location as an object passed the validator and folded here as no entries, the location lost
    without a refusal). The empty-list reading below is for a STANDING value nothing has set, never a silent correction of a record."""
    def parse(v: object) -> List[object]:
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                return []
        return list(v) if isinstance(v, list) else []
    return canonical_json(union_entries(parse(standing), parse(incoming)))
#: The tables whose marked columns may REFRESH a value an earlier MARKED statement supplied. Without this, "skip over any existing value"
#: froze a stats reader's own counts at their first value -- `n_items_generated` 4 then 12 stayed 4 -- on the runs no owner ever states, and
#: nothing reported it. The fold records, per row and column, that the standing value came from a marked statement: the flag
#: ``"unstated": true`` inside the row's ``field_confidence_json[c]`` entry (:data:`UNSTATED_FLAG`), written when a marked column applies and
#: shed when an unmarked (owner) statement states the column. The rule: a marked column applies where the standing value is NULL, or where
#: the standing value itself came from a marked statement; never over an unmarked value; a later unmarked statement still overwrites, as
#: every statement does, and an owner's statement that OMITS a flagged column carries the flagged entry forward (the owner did not state
#: it). Derived purely from the ordered events -- the flag is the fold's own rendering of them -- so a rebuild replays identically.
#: Transition: no ``run`` record in the live log carries the marker (the first writer is petri_stats EMISSION_VERSION 4, on this branch), so
#: no value was ever filled before the flag existed and nothing needs re-deriving. ``object`` keeps its rule exactly (skip over any standing
#: value): its only marker writer was the bank's EMISSION_VERSION 5 (2026-10-03), whose records are in the log, and the object rows they
#: folded must rebuild byte for byte.
#:
#: A marked NULL on these tables is a WITHDRAWAL: it applies -- the column becomes null, with the event's stamp and its (flagged)
#: confidence entry -- only where the standing value came from a marked statement, the one case a null has something to withdraw; over an
#: owner's unmarked value it is skipped like any marked column, and over a standing null (or no row) it moves nothing: no value, no stamp,
#: no flag entry. Before it the reader's record omitted a null, so a reader could never retract a value it had filed: a
#: ``serving_identity`` the corrected sidecar no longer attested stood as current. The validator admits a null marked column on these
#: tables only (``validate.UNSTATED_REFRESH_TABLES`` mirrors this tuple); on ``object`` it is refused as before.
UNSTATED_REFRESH_TABLES: Tuple[str, ...] = ("run",)
UNSTATED_FLAG = "unstated"
#: The tables carrying the ``extra.<column>_origin`` convention (:func:`_origin_follows_statement`): a column this event states sheds the
#: earlier record's origin. ``object`` only -- the convention was written for the bank's ``status_origin``; a ``run`` record's extra
#: carries producer keys of its own that happen to end in ``_origin`` nowhere today, and widening the sweep to ``run`` would be a fold
#: rule change over every live run event (a rebuild would have to agree), for a convention no run writer uses.
ORIGIN_TABLES: Tuple[str, ...] = ("object",)
#: The provenance a skipped unstated column keeps: its entry in this JSON column, and this ``extra`` key pattern.
CONFIDENCE_COLUMN = "field_confidence_json"
ORIGIN_KEY_SUFFIX = "_origin"


def _origin_follows_statement(extra: Dict[str, object], stated: Sequence[str]) -> None:
    """Drop the accumulated ``<c>_origin`` of every column this event explicitly states, in place, BEFORE the event's own extra is
    folded in: an origin the event carries then sets itself, one it does not carry is gone -- the origin describes the standing
    statement, never an earlier record's. Only for the tables carrying the convention (:data:`UNSTATED_TABLES`)."""
    for c in stated:
        extra.pop(c + ORIGIN_KEY_SUFFIX, None)


def _skip_provenance(payload: Dict[str, object], ev_extra: Dict[str, object], state: Mapping[str, object],
                     skipped: Sequence[str]) -> Tuple[Dict[str, object], Dict[str, object]]:
    """The event's payload and extra with the provenance of ``skipped`` columns replaced by the standing row's: the event's
    ``field_confidence_json[c]`` gives way to the state's entry (or to none), and ``extra.<c>_origin`` is not applied."""
    if not skipped:
        return payload, ev_extra
    ev_extra = {k: v for k, v in ev_extra.items() if not any(k == c + ORIGIN_KEY_SUFFIX for c in skipped)}
    raw = payload.get(CONFIDENCE_COLUMN)
    if raw is None:
        return payload, ev_extra
    try:
        conf = json.loads(raw) if isinstance(raw, str) else dict(raw)
        prior_raw = state.get(CONFIDENCE_COLUMN)
        prior = (json.loads(prior_raw) if isinstance(prior_raw, str) else dict(prior_raw or {})) if prior_raw is not None else {}
    except (TypeError, ValueError):
        return payload, ev_extra
    if not isinstance(conf, dict):
        return payload, ev_extra
    for c in skipped:
        if c in prior:
            conf[c] = prior[c]
        else:
            conf.pop(c, None)
    payload = dict(payload)
    payload[CONFIDENCE_COLUMN] = canonical_json(conf) if isinstance(raw, str) else conf
    return payload, ev_extra


def _conf_of(raw: object) -> Dict[str, object]:
    """A ``field_confidence_json`` value (a canonical string from the log, a dict from a caller) as a dict; ``{}`` when absent or malformed."""
    if raw is None:
        return {}
    try:
        conf = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return {}
    return conf if isinstance(conf, dict) else {}


def unstated_flagged_columns(state: Mapping[str, object]) -> set:
    """The columns of a standing row whose value came from a MARKED statement: the entries of its ``field_confidence_json`` carrying
    :data:`UNSTATED_FLAG` (:data:`UNSTATED_REFRESH_TABLES`)."""
    return {c for c, ent in _conf_of(state.get(CONFIDENCE_COLUMN)).items() if isinstance(ent, dict) and ent.get(UNSTATED_FLAG) is True}


def _refresh_provenance(payload: Dict[str, object], state: Mapping[str, object], unstated: Sequence[str], skipped: Sequence[str],
                        applied_marked: Sequence[str], kind: str = "full") -> Dict[str, object]:
    """The ``field_confidence_json`` a :data:`UNSTATED_REFRESH_TABLES` row carries after this event (the carrier of the per-column
    ``unstated`` flag; never itself a marked column). A MARKED statement merges INTO the standing entries, touching only the columns it
    applies: each applied column takes the event's entry (if it carries one) with the flag; a skipped column keeps the standing entry; the
    rest stand -- so a stats reader's re-read moves no provenance it has no statement about. An UNMARKED FULL statement is the event's JSON
    wholesale, as every statement is, plus the flagged entries of the columns it does not state (carried: the owner did not state them);
    an unmarked statement of a flagged column sheds the flag. An unmarked PATCH that carries the JSON is read per column, as the patch's
    columns are: each entry it carries replaces the standing entry of its column (flag shed), an entry of null withdraws the standing
    entry, the entries of the columns it names nothing about stand, and a column it states outright without an entry keeps the standing
    entry with the flag shed -- so a hook that cannot read the row (the standup's endpoint record stamping the served positions mode)
    states the provenance of what it patches and nothing else, and an ``unverified`` value on a patch carries its reason in the patch
    itself, as the validator requires. A patch that carries no JSON keeps the standing entries, shedding the flag of every column it
    states. Whole-log audit at the per-column rule's landing (read-only): no ``run_patch`` record in the log carried the column, so
    nothing already folded moves under it. Returns the payload with the column set when it changed or the event carried it."""
    prev = _conf_of(state.get(CONFIDENCE_COLUMN))
    raw = payload.get(CONFIDENCE_COLUMN) if CONFIDENCE_COLUMN in payload else None
    ev_conf = _conf_of(raw) if CONFIDENCE_COLUMN in payload else None
    if unstated:
        new: Dict[str, object] = dict(prev)
        for c in applied_marked:
            ent = dict(ev_conf[c]) if ev_conf and isinstance(ev_conf.get(c), dict) else {}
            ent[UNSTATED_FLAG] = True
            new[c] = ent
        for c in payload:        # a column the marked event states OUTRIGHT (its identity): the event's entry when it carries one, flag shed
            if c in unstated or c in skipped or c == CONFIDENCE_COLUMN:
                continue
            if ev_conf and isinstance(ev_conf.get(c), dict):
                new[c] = {k: v for k, v in ev_conf[c].items() if k != UNSTATED_FLAG}
            elif isinstance(new.get(c), dict) and new[c].get(UNSTATED_FLAG):
                ent = {k: v for k, v in new[c].items() if k != UNSTATED_FLAG}
                if ent:
                    new[c] = ent
                else:
                    new.pop(c)
    elif ev_conf is None or kind == "patch":
        # no JSON on the event: the standing entries stand, the flag shed where the event states the column; a PATCH's JSON: per column,
        # the entries it carries over the standing ones (null withdraws), the rest standing, the flag shed where the patch states the column
        new = dict(prev)
        for c, ent in (ev_conf or {}).items():
            if ent is None:
                new.pop(c, None)
            else:
                new[c] = {k: v for k, v in ent.items() if k != UNSTATED_FLAG} if isinstance(ent, dict) else ent
        for c in payload:
            if c == CONFIDENCE_COLUMN or c in (ev_conf or {}):
                continue
            if isinstance(new.get(c), dict) and new[c].get(UNSTATED_FLAG):
                ent = {k: v for k, v in new[c].items() if k != UNSTATED_FLAG}
                if ent:
                    new[c] = ent
                else:
                    new.pop(c)
    else:
        new = {c: ({k: v for k, v in ent.items() if k != UNSTATED_FLAG} if isinstance(ent, dict) else ent) for c, ent in ev_conf.items()}
        for c, ent in prev.items():
            if isinstance(ent, dict) and ent.get(UNSTATED_FLAG) is True and c not in payload and c not in new:
                new[c] = ent
    if new == prev and CONFIDENCE_COLUMN not in payload:
        return payload
    out = dict(payload)
    out[CONFIDENCE_COLUMN] = new if (raw is not None and not isinstance(raw, str)) else canonical_json(new)
    return out


def unstated_columns_of(ev_extra: Mapping[str, object]) -> Tuple[str, ...]:
    """The columns an event's ``_extra`` marks unstated (``()`` when none; a malformed marker never reaches the fold -- the validator
    refused the record -- so anything that is not a list of strings reads as none)."""
    cols = ev_extra.get(UNSTATED_COLUMNS_KEY)
    if not isinstance(cols, list):
        return ()
    return tuple(str(c) for c in cols if isinstance(c, str))


def fold(table: str, events: Sequence[HistoryEvent]) -> List[Tuple[Dict[str, object], Dict[str, list]]]:
    """Fold ordered events; returns, per event, ``(state_after, field_versions_after)`` (deep copies). A ``full`` / ``patch`` event
    on a table in :data:`UNSTATED_TABLES` whose ``_extra`` carries :data:`UNSTATED_COLUMNS_KEY` applies those columns only when
    the key has no state yet (no row exists before this event) or where the standing row has no value for the column (a hole the
    owner left, filled, never a value overwritten) -- and, on :data:`UNSTATED_REFRESH_TABLES`, where the standing value itself came from a
    marked statement (refreshed, or WITHDRAWN by a marked null); a pure function of the event and the prior state, so a replay gives the same row."""
    state: Dict[str, object] = {}
    fv: Dict[str, list] = {}
    extra: Dict[str, object] = {}
    event_owned: set = set()
    open_claims: set = set()       # run: the judgment ids whose latest judgment_state status so far is `in_progress`
    evaluations: List[Tuple[str, str, int]] = []
    out = []
    for ev in events:
        stamp = [ev.written_utc, ev.sha]
        kind = ev.kind
        payload = {k: v for k, v in ev.payload.items() if not k.startswith("_")}
        ev_extra = ev.payload.get("_extra") or {}
        if kind in ("full", "patch"):
            payload = lift_object_fields(table, payload, ev_extra)          # The lineage columns an old record parked under extra
        unstated: Tuple[str, ...] = ()
        row_exists = bool(state)
        # a marked column is SKIPPED over a standing value an UNMARKED statement supplied; it applies where the row holds none (null /
        # never stated) and, on the refresh tables, where the standing value itself came from a marked statement (the flag in the row's
        # confidence entry) -- a stats reader may refresh its own earlier value, never an owner's (:data:`UNSTATED_REFRESH_TABLES`)
        skipped: set = set()
        applied_marked: List[str] = []
        if kind in ("full", "patch") and table in UNION_COLUMNS:
            # a union column ADDS its entries to the standing list, on every record of the table (UNION_COLUMNS): the event's list becomes
            # the union of the state before it and its own, then folds as a statement
            payload = union_payload(table, payload, state)
        if isinstance(ev_extra, dict):
            if UNION_COLUMNS_KEY in ev_extra:
                # the marker names what the fold does anyway; a directive, never a park: it never lands in the extra
                ev_extra = {k: v for k, v in ev_extra.items() if k != UNION_COLUMNS_KEY}
            if table in UNSTATED_TABLES and UNSTATED_COLUMNS_KEY in ev_extra:
                unstated = unstated_columns_of(ev_extra)
                ev_extra = {k: v for k, v in ev_extra.items() if k != UNSTATED_COLUMNS_KEY}
                if kind in ("full", "patch"):
                    if table in UNSTATED_REFRESH_TABLES:
                        # the carrier of the flag is never itself marked: the validator refuses a marker that names it
                        # (`validate.UNSTATED_CARRIER_COLUMN`) and this drop is the belt behind that refusal
                        unstated = tuple(c for c in unstated if c != CONFIDENCE_COLUMN)
                        flagged = unstated_flagged_columns(state) if row_exists else set()
                        skipped = {c for c in unstated if c in payload and row_exists and state.get(c) is not None and c not in flagged}
                        # a marked NULL is a withdrawal: it clears a flagged value (the reader's own) and nothing else -- over a standing
                        # null, or no row, there is nothing to withdraw, so it moves no value, stamp or flag entry
                        skipped |= {c for c in unstated if c in payload and payload[c] is None and state.get(c) is None}
                        applied_marked = [c for c in unstated if c in payload and c not in skipped]
                    elif row_exists:
                        skipped = {c for c in unstated if c in payload and state.get(c) is not None}
                        # the skipped columns keep their provenance too: the standing confidence entry and origin, never the event's
                        payload, ev_extra = _skip_provenance(payload, ev_extra, state, sorted(skipped))
            if table in ORIGIN_TABLES and kind in ("full", "patch"):
                # the origin describes the standing statement: a column this event states sheds the earlier record's origin
                _origin_follows_statement(extra, [c for c in payload if c not in skipped])
            extra.update(ev_extra)
        if kind in ("full", "patch") and table in UNSTATED_REFRESH_TABLES:
            payload = _refresh_provenance(payload, state, unstated, sorted(skipped), applied_marked, kind)
        if kind in ("full", "patch"):
            for k, val in payload.items():
                if table == "run" and kind == "full" and k in event_owned:
                    continue
                if k in skipped:
                    continue            # a non-statement over an existing value: the folded value and its stamp stand
                state[k] = val
                fv[k] = stamp
        elif kind == "state" and lifecycle_regresses(state.get("lifecycle_state"), str(payload["state"])):
            # the lifecycle is monotonic: a backwards state event (a reader's re-emitted `judged` folding after the
            # harvester's `ingested` stamp) is folded — its history row stands, for provenance — and moves nothing,
            # neither lifecycle_state nor a *_utc; the stamps therefore stay the earlier records' and
            # lifecycle_regressions_held counts the event for the cycle's report
            pass
        elif kind == "state":
            st = str(payload["state"])
            state["lifecycle_state"] = st
            fv["lifecycle_state"] = stamp
            event_owned.add("lifecycle_state")
            col = V.LIFECYCLE_UTC_COLUMN.get(st)
            if col:
                state[col] = payload["event_utc"]
                fv[col] = stamp
                event_owned.add(col)
            if st == "ingested" and payload.get("judged_utc") and not state.get("judged_utc"):
                # a rule-scored run reaches ingested from generated: the harvester's stamp carries the scoring time
                state["judged_utc"] = payload["judged_utc"]
                fv["judged_utc"] = stamp
                event_owned.add("judged_utc")
            if st in V.LIFECYCLE_TERMINAL_FINISHED:
                state["finished_utc"] = payload["event_utc"]
                fv["finished_utc"] = stamp
                event_owned.add("finished_utc")
        elif kind == "judgment_state":
            # a judgment record's second effect, the run's transient `judging`. The fold tracks the run's OPEN CLAIMS -- the
            # judgment ids whose latest status in THIS key's event order is `in_progress` -- so the move back is a function
            # of the run key's ordered events alone, the same on live ingest and on replay (an accumulator like
            # `event_owned`; the run key always refolds): a claim moves a `generated` run to `judging`, and an `abandoned` /
            # `superseded` judgment moves a `judging` run back only when it ends the run's last open claim. Before
            # 2026-10-03 the move back was unconditional, so superseding one judge's complete judgment while another
            # judge's claim was still in progress read the run as `generated`. The whole-log audit
            # found no event where the two rules differ (0 of the 74 abandoned / superseded judgment records in the log
            # had another claim open), so historical replay is unchanged.
            status = payload.get("status")
            jid = str(payload.get("judgment_id"))
            cur = state.get("lifecycle_state")
            if status == "in_progress":
                open_claims.add(jid)
            else:
                open_claims.discard(jid)
            if status == "in_progress" and cur == "generated":
                state["lifecycle_state"] = "judging"
                fv["lifecycle_state"] = stamp
                event_owned.add("lifecycle_state")
            elif status in ("abandoned", "superseded") and cur == "judging" and not open_claims:
                state["lifecycle_state"] = "judged" if state.get("judged_utc") else "generated"
                fv["lifecycle_state"] = stamp
        elif kind == "ruling_eval":
            evaluations.append((str(payload["evaluated_utc"]), str(payload["evidence_state"]), int(payload["expired"])))
            latest = max(evaluations)
            state["evidence_state"] = latest[1]
            state["evidence_evaluated_utc"] = latest[0]
            fv["evidence_state"] = stamp
            fv["evidence_evaluated_utc"] = stamp
            expired = [e[0] for e in evaluations if e[2]]
            state["expired_utc"] = min(expired) if expired else None
            fv["expired_utc"] = stamp
        snapshot = dict(state)
        if extra:
            snapshot["extra_json"] = canonical_json(extra)
        out.append((snapshot, {k: list(val) for k, val in fv.items()}))
    return out


def lifecycle_regressions_held(events: Sequence[HistoryEvent], folded: Sequence[Tuple[Dict[str, object], Dict[str, list]]],
                               start: int = 0) -> int:
    """How many ``state`` events from position ``start`` on the fold held (:func:`lifecycle_regresses`). A state event that
    applies always stamps ``lifecycle_state`` with its own order key, so one whose stamp after it is another record's was
    held. ``start`` is the refolded suffix (``Reducer.apply_run_scale``): the events this apply folded again."""
    n = 0
    for ev, (_, fv) in zip(events[start:], folded[start:]):
        if ev.kind == "state" and fv.get("lifecycle_state") != [ev.written_utc, ev.sha]:
            n += 1
    return n


# --------------------------------------------------------------------------------------------
# the reducer
# --------------------------------------------------------------------------------------------

#: An appended ``full`` / ``patch`` event on a table without a hidden fold accumulator folds from the previous event's stored
#: snapshot instead of re-reading the key's whole history (``Reducer._appended_fold``). Tests flip this off to compare the two
#: paths byte for byte; production leaves it on.
INCREMENTAL_FOLD = True
#: Tables whose fold keeps state the snapshot does not carry, so they always refold: ``run`` (``event_owned`` -- the columns
#: a lifecycle event set, which a later full record may not overwrite -- and ``open_claims``, the judgment ids still
#: ``in_progress``, which decide whether an ended claim moves a ``judging`` run back) and ``ruling`` (the evaluations list
#: behind ``ruling_eval``).
_REFOLD_ALWAYS: Tuple[str, ...] = ("run", "ruling")



def _ledger_seq_of(conn: sqlite3.Connection) -> Callable[[str], Optional[int]]:
    def seq_of(file_uri: str) -> Optional[int]:
        row = conn.execute("SELECT ingest_seq FROM record_ingest WHERE file_uri = ?", (file_uri,)).fetchone()
        return int(row[0]) if row else None
    return seq_of


class Reducer:
    """``seq_of(file_uri)`` is the ledger accessor (``harvest.Ranker.seq_of``): a live row carries the rank of
    the file holding its LAST event, which may be a file ranked in an earlier cycle."""

    def __init__(self, conn: sqlite3.Connection, seq_of: Optional[Callable[[str], Optional[int]]] = None):
        self.conn = conn
        self.seq_of = seq_of or _ledger_seq_of(conn)
        self._cols: Dict[str, List[str]] = {t.name: t.column_names for t in S.TABLES}
        self._payload_cols: Dict[str, List[str]] = {t.name: S.payload_columns(t.name) for t in S.TABLES}
        self._item_sql: Dict[str, str] = {}
        self._live_sql: Dict[Tuple[str, Tuple[str, ...]], str] = {}
        self._item_plan: Dict[str, ItemPlan] = {t: item_plan(t) for t in S.ITEM_SCALE_TABLES}
        self.deferred_current: Optional[Set[str]] = None   # an activation collects the sources whose `current` it recomputes ONCE
        self.lifecycle_regressions_held = 0   # backwards state events the fold held (lifecycle_regresses); the cycle's report takes the delta per record

    # -- run-scale ------------------------------------------------------------------------------
    def _load_events(self, table: str, key_json: str) -> List[HistoryEvent]:
        rows = self.conn.execute(
            "SELECT written_utc, record_sha256, record_type, record_uri, payload_json FROM row_history "
            f"WHERE table_name = ? AND natural_key_json = ? ORDER BY {ORDER_SQL}",
            (table, key_json)).fetchall()
        return [HistoryEvent(r[0], r[1], r[2], r[3], json.loads(r[4])) for r in rows]

    def apply_run_scale(self, eff: Effect, v: Validated, record_uri: str, ingest_seq: int) -> bool:
        """Insert the event, refold the key, rewrite the live row and the history suffix. Returns
        False when the event was already present (duplicate sha).

        The refold reads the key's WHOLE history (``_load_events``) unless the event is an appended ``full`` / ``patch``
        event on a table whose fold carries no hidden accumulator, in which case :meth:`_appended_fold` derives the same
        result from the previous event's stored snapshot alone (2026-09-24: a re-emitted ``source_watermark`` or refused
        ``source_revision`` marker re-read ~70-300 history rows, one random page each, on every adapter pass -- 0.45 s per
        record, 22-31k s of write-lock time a day on the live database -- and the history grows by one event per pass)."""
        key_json = eff.key_json
        hist_payload = eff.history_payload()
        if eff.kind == "patch" and eff.extra.get("verified_fields"):
            self._verify(eff, v)
        if eff.table == "judgment" and eff.kind == "full":
            self._check_successor(eff)
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO row_history(table_name, natural_key_json, written_utc, record_sha256, record_type, "
            "record_uri, payload_json, folded_state_json) VALUES (?,?,?,?,?,?,?,?)",
            (eff.table, key_json, v.written_utc, v.sha, v.record_type, record_uri, canonical_json(hist_payload), "{}"))
        if cur.rowcount == 0:
            self._adopt_canonical_carrier(eff, v, key_json, record_uri, ingest_seq)
            return False
        appended = self._appended_fold(eff, v, key_json, hist_payload) if INCREMENTAL_FOLD else None
        if appended is not None:
            state, fv = appended
            self.conn.execute(
                "UPDATE row_history SET folded_state_json = ? WHERE table_name = ? AND natural_key_json = ? "
                "AND written_utc = ? AND record_sha256 = ?",
                (canonical_json(state), eff.table, key_json, v.written_utc, v.sha))
            last_seq = self.seq_of(record_uri)
            self._write_live_row(eff.table, eff.key, state, fv, record_uri, last_seq if last_seq is not None else ingest_seq)
            self._post_apply(eff, state, fv)
            return True
        events = self._load_events(eff.table, key_json)
        folded = fold(eff.table, events)
        pos = next(i for i, e in enumerate(events) if e.sha == v.sha and e.written_utc == v.written_utc)
        if eff.table == "run" and eff.kind == "state":
            # the evidence rule as the FOLD sees it (LifecycleEvidenceMissing): a lifecycle event whose refold reaches a state without an
            # earlier state's utc is blocked by the harvester on that evidence, never refused by the trigger the upsert below would fire;
            # raised before the history suffix and the counters move, inside the caller's savepoint, so the event's history row rolls back
            missing = lifecycle_evidence_missing(folded[-1][0])
            if missing:
                raise LifecycleEvidenceMissing(str(eff.key.get("run_id")), str(folded[-1][0].get("lifecycle_state")), missing)
        self.lifecycle_regressions_held += lifecycle_regressions_held(events, folded, pos)
        self.rewrite_history(eff.table, eff.key, key_json, events, folded, start=pos)
        state, fv = folded[-1]
        last = events[-1]
        last_seq = self.seq_of(last.record_uri)
        seq = last_seq if last_seq is not None else ingest_seq
        self._write_live_row(eff.table, eff.key, state, fv, last.record_uri, seq)
        self._post_apply(eff, state, fv)
        return True

    def rewrite_history(self, table: str, key: Mapping[str, object], key_json: str, events: Sequence[HistoryEvent],
                        folded: Sequence[Tuple[Dict[str, object], Dict[str, list]]], start: int = 0) -> int:
        """THE ONE WRITER of ``row_history.folded_state_json``: the snapshot after every event from ``start`` on, as ``fold`` computed it
        over the key's whole ordered history, plus the fold observations for an observed table (observations.py: a detector over the
        fold's output, never a fold rule -- the key's rows replaced from the same ``(events, folded)``, so live equals replay). The live
        fold calls it from the position of the event it inserted (:meth:`apply_run_scale`); a refold of a whole key under a changed order
        (``replica._refold_rows``, the restamp of the C1 movers) calls it from 0, so the stored prefixes read as the new order's -- a
        rewrite of the live row alone left ``registry history`` and ``harvest.verify_equal`` showing the old order's snapshots. Returns the
        rows rewritten."""
        n = 0
        for i in range(start, len(events)):
            e = events[i]
            self.conn.execute(
                "UPDATE row_history SET folded_state_json = ? WHERE table_name = ? AND natural_key_json = ? "
                "AND written_utc = ? AND record_sha256 = ?",
                (canonical_json(folded[i][0]), table, key_json, e.written_utc, e.sha))
            n += 1
        if table in OBS.OBSERVED_TABLES:
            OBS.observe_key(self.conn, key, events, folded)
        return n

    def _appended_fold(self, eff: Effect, v: Validated, key_json: str, hist_payload: Mapping[str, object]):
        """The fold of the key after ``v`` WITHOUT re-reading its history -- ``(snapshot, field_versions)`` exactly as
        :func:`fold` would return them for the key's last event -- or None when the shortcut does not apply and the caller
        must refold the whole history. It applies when the four facts below hold, each checked here, never assumed:

        * the table's fold has no accumulator outside the stored snapshot: ``run`` keeps ``event_owned`` (the columns a
          lifecycle event set, which a later full record may not overwrite) and ``open_claims`` (the judgment ids still
          ``in_progress``), ``ruling`` keeps its evaluations list, so both always refold; every other table folds a
          ``full`` / ``patch`` event as ``state[k] = payload[k]``, ``field_versions[k] = stamp`` and ``extra.update(_extra)``
          -- a function of the previous snapshot and the event;
        * the event is the key's LAST by order key ``(written_utc, tie_rank, record_sha256)`` (:data:`ORDER_DESC_SQL`): an
          event arriving late that sorts before one already folded changes the suffix after it, which only the refold rewrites;
        * every column the previous snapshot set is set again by this event, so no ``field_versions`` entry survives from
          an older event (the snapshot stores the state, not the stamps) -- true of every re-emitted record, which carries
          its whole row; a record that omits a column the key already has refolds;
        * the previous snapshot's ``extra_json`` is the accumulated ``_extra`` of the events before this one: the state never
          holds a column of that name (``validate._normalise_payload`` parks it), so the key is unambiguous.

        The event's values take the same JSON round trip ``_load_events`` gives them, so the snapshot's bytes are the
        refold's bytes. A key with no earlier event (a first record) refolds: its history is one row. The ``object`` lift
        (:func:`lift_object_fields`) is applied to the event's payload before the subset test and the state, exactly as
        :func:`fold` applies it, so both paths agree byte for byte."""
        if eff.table in _REFOLD_ALWAYS or eff.kind not in ("full", "patch"):
            return None
        if eff.table in UNSTATED_TABLES and isinstance(eff.extra, dict) and UNSTATED_COLUMNS_KEY in eff.extra:
            return None            # an unstated column keeps the previous event's value and stamp: only the refold knows them
        rows = self.conn.execute(
            "SELECT written_utc, record_sha256, folded_state_json FROM row_history WHERE table_name = ? AND natural_key_json = ? "
            f"ORDER BY {ORDER_DESC_SQL} LIMIT 2", (eff.table, key_json)).fetchall()
        if len(rows) < 2 or (str(rows[0][0]), str(rows[0][1])) != (v.written_utc, v.sha):
            return None            # a first event, or one that sorts before an event already folded
        prev = json.loads(rows[1][2])
        extra_json = prev.pop("extra_json", None)
        extra: Dict[str, object] = json.loads(extra_json) if extra_json is not None else {}
        event = json.loads(canonical_json(hist_payload))          # the round trip _load_events gives every stored event
        payload = {k: val for k, val in event.items() if not k.startswith("_")}
        ev_extra = event.get("_extra") or {}
        payload = lift_object_fields(eff.table, payload, ev_extra)
        # a union column folds over the previous snapshot's list exactly as fold() folds it over the state before the event
        # (union_payload, UNION_COLUMNS); a null carried over a standing list leaves the payload, so the subset test below refolds
        payload = union_payload(eff.table, payload, prev)
        if not set(prev) <= set(payload):
            return None
        if eff.table in ORIGIN_TABLES:
            _origin_follows_statement(extra, list(payload))      # every column of this (unmarked) event is an explicit statement
        if isinstance(ev_extra, dict):
            ev_extra = {k: val for k, val in ev_extra.items() if k != UNION_COLUMNS_KEY}     # the marker never lands in the extra
            extra.update(ev_extra)
        stamp = [v.written_utc, v.sha]
        state: Dict[str, object] = dict(payload)
        fv = {k: list(stamp) for k in payload}
        snapshot = dict(state)
        if extra:
            snapshot["extra_json"] = canonical_json(extra)
        return snapshot, fv

    def _adopt_canonical_carrier(self, eff: Effect, v: Validated, key_json: str, record_uri: str, ingest_seq: int) -> None:
        """The same record delivered under two file names: the event's carrier is the lexicographically smallest
        logical URI, whatever arrived first — provenance metadata (uri, rank) is updated, the semantic
        effect is NOT re-applied. When the event is the key's LAST, the live row's pointer follows."""
        row = self.conn.execute(
            "SELECT record_uri FROM row_history WHERE table_name = ? AND natural_key_json = ? AND written_utc = ? AND record_sha256 = ?",
            (eff.table, key_json, v.written_utc, v.sha)).fetchone()
        if row is None or not (record_uri < str(row[0])):
            return
        self.conn.execute(
            "UPDATE row_history SET record_uri = ? WHERE table_name = ? AND natural_key_json = ? AND written_utc = ? AND record_sha256 = ?",
            (record_uri, eff.table, key_json, v.written_utc, v.sha))
        if eff.table in OBS.OBSERVED_TABLES:
            OBS.adopt_carrier(self.conn, eff.key, v.sha, record_uri)      # an observation's uri is the event's carrier, live and on replay
        last = self.conn.execute(
            "SELECT written_utc, record_sha256 FROM row_history WHERE table_name = ? AND natural_key_json = ? "
            f"ORDER BY {ORDER_DESC_SQL} LIMIT 1", (eff.table, key_json)).fetchone()
        if last is not None and (last[0], last[1]) == (v.written_utc, v.sha):
            seq = self.seq_of(record_uri)
            where = " AND ".join(f"{k} = ?" for k in eff.key)
            self.conn.execute(f"UPDATE {eff.table} SET record_uri = ?, ingest_seq = ? WHERE {where}",
                              (record_uri, seq if seq is not None else ingest_seq, *eff.key.values()))

    # -- ledger-scale facts ------------------------------------------------------------------
    def apply_ledger_fact(self, v: Validated) -> "LedgerOutcome":
        """Fold a record whose table is a LEDGER (``records.RECORD_TYPE_TABLE`` -> a table in ``S.LEDGER_TABLES``: the `record_refusal`
        kind -> `record_refusal_fact`, the analysis kinds -> their tables): the payload columns, the parked extra as `extra_json` -- no
        `row_history` event, no field stamps, no refold (a ledger-scale table has no natural-key fold: the row IS the fact). Two rules, by
        table:

        * a table WITHOUT an ordering key (``records.LEDGER_ORDERED_TABLES`` names none): ONE ``INSERT OR IGNORE`` -- the key names the fact
          whole and a second record of it is a duplicate (`inserted` / `duplicate`).
        * a table WITH one (the three analysis tables): the row kept for a natural key is the record with the smallest ``(written_utc,
          record_sha256)`` -- the LOG's order, which a rebuild reproduces whatever the arrival order (an unconditional INSERT OR IGNORE keeps
          the first ARRIVAL instead, so a late-delivered earlier filing gives live and rebuild different headers; ordering the header alone
          is not enough either, since a link or an input filed twice with another payload would still keep the first arrival). The three
          derived columns the mapping names carry the kept record's stamp, sha and uri. An incoming record that sorts EARLIER than the
          standing row replaces it (`replaced`, the displaced record named in the outcome for the harvester to refuse by name); one that
          sorts later and carries the same payload is a `duplicate`; one that sorts later with a different payload is a `conflict` (the
          harvester refuses it by name, naming the differing columns); the same record again is a `duplicate` (its uri adopted when it is
          the smaller). An identical payload under an earlier key makes the row that filing's (its key columns and its event columns, as a
          rebuild writes it) and reads `duplicate`. The columns that describe the filing EVENT
          (``records.LEDGER_ORDERED_EVENT_COLUMNS``: `filed_utc`) are not compared -- a re-run on the same spec is a duplicate. EVERY record
          of an ordered table, whatever its outcome, is indexed by its key in ``records.LEDGER_ORDERED_RECORD_INDEX`` (`ordered_fact_record`:
          sha, uri, stamp -- carriers, never payloads), so the harvester can judge every record of a key again when the kept one moves.

        A pure function of the records: a rebuild replays the same files to the same rows, the same index and the same refusals."""
        table = v.table
        assert table is not None and table in S.LEDGER_TABLES, table
        row: Dict[str, object] = {c: v.payload[c] for c in self._payload_cols[table] if c in v.payload}
        row = SA.canonical_row(table, row)         # a site label written with a legacy spelling lands as its label
        if "extra_json" in self._cols[table]:
            row["extra_json"] = canonical_json(v.extra) if v.extra else row.get("extra_json", "{}") or "{}"
        order = R.LEDGER_ORDERED_TABLES.get(table)
        if order is None:
            names = tuple(row)
            sql = f"INSERT OR IGNORE INTO {table} ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})"
            inserted = self.conn.execute(sql, tuple(row[c] for c in names)).rowcount == 1
            return LedgerOutcome("inserted" if inserted else "duplicate")
        utc_col, sha_col, uri_col = order
        pk = R.NATURAL_KEYS[table]
        key = tuple(v.payload[c] for c in pk)
        where = " AND ".join(f"{c} = ?" for c in pk)
        self._index_ordered_record(table, dict(zip(pk, key)), v)        # every record of the key, whatever the outcome below
        existing = self.conn.execute(f"SELECT * FROM {table} WHERE {where}", key).fetchone()
        # the full row as an INSERT would write it: a payload column the record omits takes the column's DEFAULT (never "differs" from it)
        full: Dict[str, object] = {c: (row[c] if c in row else self._default_value(table, c)) for c in self._payload_cols[table]}
        full[utc_col], full[sha_col], full[uri_col] = v.written_utc, v.sha, v.uri
        if existing is None:
            names = tuple(full)
            self.conn.execute(f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})", tuple(full[c] for c in names))
            return LedgerOutcome("inserted", kept_sha=v.sha, kept_changed=True)
        kept_key = (str(existing[utc_col]), str(existing[sha_col]))
        incoming_key = (v.written_utc, v.sha)
        if incoming_key == kept_key:
            # the same record again: under a smaller logical uri it is the row's carrier (the canonical carrier, whatever arrived first)
            if v.uri < str(existing[uri_col]):
                self.conn.execute(f"UPDATE {table} SET {uri_col} = ? WHERE {where}", (v.uri,) + key)
            return LedgerOutcome("duplicate", kept_sha=v.sha)
        event_cols = R.LEDGER_ORDERED_EVENT_COLUMNS.get(table, ())
        differing = tuple(c for c in self._payload_cols[table] if c not in event_cols and full[c] != existing[c])
        if not differing:
            if incoming_key < kept_key:
                # the same header under an earlier key: the whole row becomes the earlier filing's -- its key columns AND its event columns
                # (`filed_utc`), exactly the row a rebuild writes when it meets the earlier filing first -- and reads `duplicate`
                sets = ", ".join(f"{c} = ?" for c in full)
                self.conn.execute(f"UPDATE {table} SET {sets} WHERE {where}", tuple(full.values()) + key)
                return LedgerOutcome("duplicate", kept_sha=v.sha, kept_changed=True)
            return LedgerOutcome("duplicate", kept_sha=str(existing[sha_col]))
        if incoming_key < kept_key:
            sets = ", ".join(f"{c} = ?" for c in full)
            self.conn.execute(f"UPDATE {table} SET {sets} WHERE {where}", tuple(full.values()) + key)
            return LedgerOutcome("replaced", displaced=(str(existing[sha_col]), str(existing[uri_col])), conflict_fields=differing, kept_sha=v.sha, kept_changed=True)
        return LedgerOutcome("conflict", conflict_fields=differing, kept_sha=str(existing[sha_col]))

    def _index_ordered_record(self, table: str, key: Mapping[str, object], v: Validated) -> None:
        """One ``ordered_fact_record`` row per record of an ordered table's key (``records.LEDGER_ORDERED_RECORD_INDEX``): the carriers the
        harvester judges again when the key's kept record moves. The same record under two logical uris keeps the smaller, so the
        index is a pure function of the records, not of their arrival."""
        idx = R.LEDGER_ORDERED_RECORD_INDEX
        self.conn.execute(
            f"INSERT INTO {idx} (table_name, natural_key_json, record_sha256, record_type, record_uri, written_utc) VALUES (?, ?, ?, ?, ?, ?) "
            f"ON CONFLICT(table_name, natural_key_json, record_sha256) DO UPDATE SET record_uri = MIN(record_uri, excluded.record_uri)",
            (table, canonical_json(dict(key)), v.sha, v.record_type, v.uri, v.written_utc))

    def ordered_records_of(self, table: str, key: Mapping[str, object]) -> List[Tuple[str, str, str, str]]:
        """Every indexed record of an ordered table's key: ``[(record_sha256, record_uri, record_type, written_utc)]`` by uri."""
        return [(str(r[0]), str(r[1]), str(r[2]), str(r[3])) for r in self.conn.execute(
            f"SELECT record_sha256, record_uri, record_type, written_utc FROM {R.LEDGER_ORDERED_RECORD_INDEX} "
            "WHERE table_name = ? AND natural_key_json = ? ORDER BY record_uri", (table, canonical_json(dict(key)))).fetchall()]

    def ordered_carrier(self, table: str, key: Mapping[str, object], sha: str) -> Optional[str]:
        """The CANONICAL carrier of one indexed record of an ordered table's key: the smallest logical uri the index holds for the sha
        (``_index_ordered_record`` keeps the MIN over every arrival), or None when the record is not indexed. The harvester's
        refusal row of an ordered-fact record names this carrier, never the arrival's (the fourth review round, 2026-10-09: a conflicting
        filing copied to a smaller uri left the live `record_refusal.record_uri` at whichever copy arrived last while a rebuild, meeting
        the smaller copy first, wrote the smaller one)."""
        row = self.conn.execute(f"SELECT record_uri FROM {R.LEDGER_ORDERED_RECORD_INDEX} WHERE table_name = ? AND natural_key_json = ? "
                                "AND record_sha256 = ?", (table, canonical_json(dict(key)), sha)).fetchone()
        return str(row[0]) if row is not None else None

    def _default_value(self, table: str, column: str):
        """The value an INSERT gives a payload column the record omits: its DEFAULT literal (quoted text or a number), else NULL."""
        d = S.TABLE_BY_NAME[table].column(column).default
        if d is None:
            return None
        if d.startswith("'") and d.endswith("'"):
            return d[1:-1]
        try:
            return int(d)
        except ValueError:
            return float(d)

    def _check_successor(self, eff: Effect) -> None:
        """A judgment folding as ``superseded`` with a successor names one of ITS OWN run unless the record carries
        ``extra.allow_cross_run``. The successor's EXISTENCE is the harvester's dependency check (the FK on
        ``superseded_by_judgment_id``, ``validate.parent_references``: blocked until the successor folds, never refused for that
        alone); its run is read from the live row here -- the same on live ingest and on replay, because the identity rule
        (``validate._judgment_id_refusal``) binds every judgment id to its ``run_id``, so no later record can move it. A self-link and
        a successor whose row is not there (the FK then refuses at apply) are left to the existing checks."""
        p = eff.payload
        if p.get("status") != "superseded":
            return
        successor = p.get("superseded_by_judgment_id")
        if not successor or str(successor) == str(p.get("judgment_id")) or eff.extra.get("allow_cross_run"):
            return
        row = self.conn.execute("SELECT run_id FROM judgment WHERE judgment_id = ?", (str(successor),)).fetchone()
        if row is not None and str(row[0]) != str(p.get("run_id")):
            raise SupersedeRelationshipMismatch(str(p.get("judgment_id")), str(p.get("run_id")), str(successor), str(row[0]))

    def _verify(self, eff: Effect, v: Validated) -> None:
        row = self.conn.execute("SELECT * FROM run WHERE run_id = ?", (eff.key["run_id"],)).fetchone()
        if row is None:
            return  # dependency check already ensures presence; nothing to compare against
        for k, expected in eff.extra["verified_fields"].items():
            actual = row[k]
            if actual is not None and actual != expected:
                raise VerifyMismatch(k, expected, actual, str(eff.key["run_id"]))

    def _write_live_row(self, table: str, key: Mapping[str, object], state: Mapping[str, object],
                        fv: Mapping[str, list], record_uri: str, ingest_seq: int) -> None:
        """The folded state as ONE upsert: the payload columns the fold set, the key, and the derived tail."""
        row: Dict[str, object] = {c: state[c] for c in self._payload_cols[table] if c in state}
        row.update(key)
        row = SA.canonical_row(table, row)         # a site label written with a legacy spelling lands as its label
        row.setdefault("extra_json", "{}")
        row["field_versions_json"] = canonical_json(fv)
        row["record_uri"] = record_uri
        row["ingest_seq"] = ingest_seq
        names = tuple(row)
        sql = self._live_sql.get((table, names))
        if sql is None:
            sql = self._live_sql[(table, names)] = DB.upsert_sql(table, names, R.NATURAL_KEYS[table])
        self.conn.execute(sql, tuple(row[c] for c in names))

    def _post_apply(self, eff: Effect, state: Mapping[str, object], fv: Mapping[str, list]) -> None:
        if eff.table == "source_revision":
            self._recompute_revision_chain(str(state.get("source_id") or eff.payload.get("source_id")))
        elif eff.table == "withdrawal":
            self._apply_withdrawal(eff.payload)
        elif eff.table == "observation":
            self._recompute_current(str(eff.payload["source_id"]))
            self._apply_pending_withdrawals_to_observation(eff.payload)
        elif eff.table == "item_withdrawal":
            self._apply_item_withdrawal(eff.key, state, fv)

    # -- producer item withdrawals (migration 0020) -------------------------------------------
    def _apply_item_withdrawal(self, key: Mapping[str, object], state: Mapping[str, object], fv: Mapping[str, list]) -> None:
        """The folded ``item_withdrawal`` row of one item key is the LATEST record naming it (by order key; the fold's last event,
        whose stamp every column carries). Its mark lands on the natively emitted ``item_result`` row of the key iff the row's
        order key is below the record's; otherwise the row stands and the mark is cleared -- the same function
        :func:`remark_item_withdrawals` applies after an item flush, so the marks are a pure function of the records."""
        written_utc, sha = fv["reason"]
        mark_item_withdrawal(self.conn, key, str(state["reason"]), str(written_utc), str(sha))

    def _recompute_revision_chain(self, source_id: str) -> None:
        rows = self.conn.execute(
            "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc, revision_id",
            (source_id,)).fetchall()
        ids = [r[0] for r in rows]
        for i, rid in enumerate(ids):
            nxt = ids[i + 1] if i + 1 < len(ids) else None
            # a revision whose successor changes has its withdrawal marks derived again
            self.conn.execute("UPDATE source_revision SET reclaimed = CASE WHEN superseded_by_revision_id IS ? THEN reclaimed ELSE 0 END, "
                              "superseded_by_revision_id = ? WHERE revision_id = ?", (nxt, nxt, rid))
        self._recompute_current(source_id, head=ids[-1] if ids else None)

    def _recompute_current(self, source_id: str, head: Optional[str] = "?") -> None:
        """``observation.current`` is DERIVED: a key is current iff it belongs to the source's LATEST accepted
        revision. Recomputed whenever a revision or an observation of the source folds, so an older revision's row
        never keeps a withdrawn key alive and the result is identical live and on replay. Item rows need no such
        pass: their visibility is derived at read time from the head revision; the marks an outgoing
        revision's withdrawals left on a re-activated head's rows are cleared afterwards in bounded batches
        (``Ingestor.reclaim``). During an activation the recomputation is collected and run once per source."""
        if self.deferred_current is not None:
            self.deferred_current.add(source_id)
            return
        if head == "?":
            row = self.conn.execute("SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 "
                                    "ORDER BY observed_utc DESC, revision_id DESC LIMIT 1", (source_id,)).fetchone()
            head = row[0] if row else None
        self.conn.execute("UPDATE observation SET current = CASE WHEN revision_id IS ? THEN 1 ELSE 0 END WHERE source_id = ? "
                          "AND current <> CASE WHEN revision_id IS ? THEN 1 ELSE 0 END", (head, source_id, head))

    def _head_of(self, source_id: str) -> Optional[str]:
        row = self.conn.execute("SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 "
                                "ORDER BY observed_utc DESC, revision_id DESC LIMIT 1", (source_id,)).fetchone()
        return str(row[0]) if row else None

    def _withdrawal_stands(self, source_id: str, from_rev: str) -> bool:
        """A withdrawal only ever marks an OUTGOING revision's rows: when its source revision is the source's current
        (re-activated) revision, the withdrawal is history and must not be applied."""
        return self._head_of(source_id) != from_rev

    def _apply_withdrawal(self, w: Mapping[str, object]) -> None:
        if not self._withdrawal_stands(str(w["source_id"]), str(w["from_revision_id"])):
            return
        self.conn.execute(
            "UPDATE observation SET current = 0, withdrawn_by_revision_id = ? WHERE source_id = ? AND revision_id = ? "
            "AND observation_key = ?", (w["to_revision_id"], w["source_id"], w["from_revision_id"], w["observation_key"]))
        # the item keys a withdrawal lists are bookkeeping: their `withdrawn_by_revision_id` marks are derived from
        # the rows by Ingestor.reclaim in bounded batches after the activation — never written by this fold

    def _apply_pending_withdrawals_to_observation(self, ob: Mapping[str, object]) -> None:
        if not self._withdrawal_stands(str(ob["source_id"]), str(ob["revision_id"])):
            return
        rows = self.conn.execute(
            "SELECT to_revision_id FROM withdrawal WHERE source_id = ? AND from_revision_id = ? AND observation_key = ?",
            (ob["source_id"], ob["revision_id"], ob["observation_key"])).fetchall()
        for (to_rev,) in rows:
            self.conn.execute(
                "UPDATE observation SET current = 0, withdrawn_by_revision_id = ? WHERE source_id = ? AND revision_id = ? "
                "AND observation_key = ?", (to_rev, ob["source_id"], ob["revision_id"], ob["observation_key"]))

    # -- item-scale -----------------------------------------------------------------------------
    @staticmethod
    def item_guard(table: str) -> str:
        """The item fold's update guard: the greatest (written_utc, record_sha256) wins; the SAME record from a
        second file (equal key, different file) adopts the smaller logical URI as its carrier."""
        return (f"(excluded.written_utc, excluded.record_sha256) > ({table}.written_utc, {table}.record_sha256) "
                f"OR (excluded.written_utc = {table}.written_utc AND excluded.record_sha256 = {table}.record_sha256 "
                f"AND excluded.record_file_id <> {table}.record_file_id "
                f"AND (SELECT file_uri FROM record_ingest WHERE record_file_id = excluded.record_file_id) "
                f"< (SELECT file_uri FROM record_ingest WHERE record_file_id = {table}.record_file_id))")

    def item_upsert_sql(self, table: str) -> str:
        """Whole-row upsert of one item row, guarded by :meth:`item_guard`."""
        if table not in self._item_sql:
            self._item_sql[table] = DB.upsert_sql(table, self._cols[table], R.NATURAL_KEYS[table], guard=self.item_guard(table))
        return self._item_sql[table]


    def item_row(self, table: str, v: Validated, record_file_id: int, ingest_seq: int) -> Tuple:
        """The full row tuple (the parsed prefix + the file's rank + the trailing marks, NULL at every fold)."""
        plan = self._item_plan[table]
        return item_row_prefix(plan, v, record_file_id) + plan.tail(ingest_seq)


# --------------------------------------------------------------------------------------------
# item rows are built in the parse workers (F3): a per-table plan and a pure function
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemPlan:
    table: str
    columns: Tuple[str, ...]                      # every column before ingest_seq, in table order
    sources: Tuple[str, ...]                      # per column: payload | written_utc | record_sha256 | record_file_id
    sentinels: Tuple[Tuple[str, object], ...]     # (column, sentinel) for the key columns a payload may omit
    pk_positions: Tuple[int, ...]                 # positions of the natural-key columns in the row tuple (F2 sort key)
    trailing: Tuple[str, ...] = ()                # the derived nullable columns AFTER ingest_seq (item_result's producer-withdrawal
                                                  # marks): written NULL by every fold, set by a later effect

    def tail(self, ingest_seq: int) -> Tuple:
        """What the parent appends to the parsed prefix: the file's rank, then NULL for every trailing mark."""
        return (ingest_seq,) + (None,) * len(self.trailing)


def item_plan(table: str) -> ItemPlan:
    t = S.TABLE_BY_NAME[table]
    cols = t.column_names
    seq = cols.index("ingest_seq")                # the parent appends the file's rank after the parsed prefix
    head = t.columns[:seq]
    for c in t.columns[seq + 1:]:
        # a column after the rank is a trailing mark: derived (no producer carries it) and nullable (every fold writes NULL)
        assert c.derived and not c.nn, (table, c.name)
    sources = tuple(c.name if c.derived else "payload" for c in head)
    assert set(sources) <= {"payload", "written_utc", "record_sha256", "record_file_id"}, (table, sources)
    return ItemPlan(table, tuple(cols[:seq]), sources,
                    tuple((c.name, S.SENTINELS[c.name]) for c in head if not c.derived and c.name in S.SENTINELS),
                    tuple(cols.index(c) for c in R.NATURAL_KEYS[table]), tuple(cols[seq + 1:]))


ITEM_PLANS: Dict[str, ItemPlan] = {t: item_plan(t) for t in S.ITEM_SCALE_TABLES}


def item_row_prefix(plan: ItemPlan, v: Validated, record_file_id: int) -> Tuple:
    """Every column of an item row EXCEPT the file's ingest_seq (the parent appends it once the file is ranked):
    payload columns (sentinels for absent key columns, canonical extra_json) and the derived stamps / file id."""
    p = v.payload
    for c, sentinel in plan.sentinels:
        if c not in p:
            p[c] = sentinel
    p["extra_json"] = canonical_json(v.extra) if v.extra else "{}"
    derived = {"written_utc": R.utc_to_epoch_ms(v.written_utc), "record_sha256": bytes.fromhex(v.sha),
               "record_file_id": record_file_id}
    get = p.get
    return tuple(get(c) if src == "payload" else derived[src] for c, src in zip(plan.columns, plan.sources))


# --------------------------------------------------------------------------------------------
# producer item withdrawals (migration 0020): the mark as a pure function of the records
# --------------------------------------------------------------------------------------------

#: The natively emitted item_result row of one item_withdrawal key: the mark lands iff the row's order key (written_utc as
#: epoch ms, record_sha256 as bytes) is BELOW the withdrawing record's; a row with a greater order key stands as a new measurement
#: and reads NULL. Rows of an adapter revision (source_revision_id <> '') are never targeted: their membership is the revision's.
_ITEM_WITHDRAWAL_MARK_SQL = (
    "UPDATE item_result SET "
    "withdrawn_by_record_sha256 = CASE WHEN written_utc < :ms OR (written_utc = :ms AND record_sha256 < :sha_bytes) THEN :sha ELSE NULL END, "
    "withdrawn_reason = CASE WHEN written_utc < :ms OR (written_utc = :ms AND record_sha256 < :sha_bytes) THEN :reason ELSE NULL END "
    "WHERE " + " AND ".join(f"{c} = :{c}" for c in R.NATURAL_KEYS["item_withdrawal"]) + " AND source_revision_id = ''")


def mark_item_withdrawal(conn: sqlite3.Connection, key: Mapping[str, object], reason: str, written_utc: str, sha: str) -> int:
    """Apply one folded ``item_withdrawal`` row (its key, reason and the stamp of the record that stands for it) to the native
    ``item_result`` row of the key; returns the rows touched (0 when the row is not there yet). Idempotent and order-independent:
    the same call after a re-filed row re-derives the mark from the two order keys alone."""
    params = {"ms": R.utc_to_epoch_ms(written_utc), "sha_bytes": bytes.fromhex(sha), "sha": sha, "reason": reason}
    params.update({c: key[c] for c in R.NATURAL_KEYS["item_withdrawal"]})
    return conn.execute(_ITEM_WITHDRAWAL_MARK_SQL, params).rowcount


def remark_item_withdrawals(conn: sqlite3.Connection, run_ids: Iterable[str]) -> int:
    """Re-derive the producer-withdrawal marks of every native item_result row of ``run_ids`` that an ``item_withdrawal`` row names
    -- the item side of the pure function: a flushed item batch wrote NULL marks (a re-filed row, whole-row upsert), and a
    row whose order key is still below a standing withdrawal's takes the mark back. Runs with no withdrawal row cost one
    indexed probe each (the table's key starts with run_id). Returns the rows touched."""
    n = 0
    for rid in sorted(set(run_ids)):
        rows = conn.execute(
            f"SELECT {', '.join(R.NATURAL_KEYS['item_withdrawal'])}, reason, field_versions_json FROM item_withdrawal WHERE run_id = ?",
            (rid,)).fetchall()
        for r in rows:
            key = dict(zip(R.NATURAL_KEYS["item_withdrawal"], r))
            written_utc, sha = json.loads(r[-1])["reason"]
            n += mark_item_withdrawal(conn, key, str(r[-2]), str(written_utc), str(sha))
    return n


# --------------------------------------------------------------------------------------------
# on-demand history for item-scale keys
# --------------------------------------------------------------------------------------------


def fold_item_events(table: str, records: Iterable[Tuple[Dict[str, object], str, str]]) -> List[Dict[str, object]]:
    """``records`` = (record, sha, uri) of one item key; returns every state in order (whole-row fold:
    each event IS the state)."""
    evs = []
    for rec, sha, uri in records:
        evs.append(((str(rec["written_utc"]), sha), rec, uri))
    evs.sort(key=lambda e: e[0])
    out = []
    for (w, sha), rec, uri in evs:
        state = dict(rec["payload"])
        state["_written_utc"] = w
        state["_record_sha256"] = sha
        state["_record_uri"] = uri
        out.append(state)
    return out


def matches_key(table: str, payload: Mapping[str, object], key: Mapping[str, object]) -> bool:
    for c in R.NATURAL_KEYS[table]:
        want = key.get(c, S.SENTINELS.get(c))
        have = payload.get(c, S.SENTINELS.get(c))
        if str(want) != str(have):
            return False
    return True
