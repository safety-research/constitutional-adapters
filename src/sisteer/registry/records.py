"""The record contract (``docs/registry/records.md``) — STDLIB ONLY.

One JSON object per line, append-only files, never rewritten; a correction is a new record::

    {"record_type": "...", "schema_version": 1, "record_id": "<uuid>", "written_utc": "2026-09-16T18:04:11.312Z",
     "producer": {"script": "...", "code_sha": "<40-hex>|unverified", "session": "...|null", "host": "...", "substrate": "..."},
     "payload": { ...table columns..., "extra": {unknown fields} }}

This module names every record type and the table it feeds, the natural key the reducer upserts on,
the required fields per type (derived from the schema definition: NOT NULL columns without a
default), the ``run_patch`` field sets (S7), the scale classification, the record sha and the
order key, and reads record files (plain or gzip).
"""
from __future__ import annotations

import functools
import gzip
import hashlib
import io
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from . import schema_def as S
from . import vocab as V
from .identity import canonical_json  # re-exported

SCHEMA_VERSION = 1
RECORD_TYPES: Tuple[str, ...] = V.RECORD_TYPES

#: record type -> the table its payload is a row of. Types that are not 1:1 with a table are
#: listed in EFFECT_TYPES.
RECORD_TYPE_TABLE: Dict[str, str] = {
    "run": "run", "run_metric": "run_metric", "run_contrast": "run_contrast", "item_result": "item_result",
    "item_flags": "item_flags", "run_degeneration": "run_degeneration", "judgment": "judgment", "object": "object",
    "artifact": "artifact", "model": "model", "eval": "eval", "item_set": "item_set", "instrument": "instrument",
    "judge": "judge", "si_text": "si_text", "fit_search": "fit_search", "fit_trial": "fit_trial",
    "paper_round": "paper_round", "roster_entry": "roster_entry", "ruling": "ruling", "decode_policy": "decode_policy",
    "sampling_mode_policy": "sampling_mode_policy", "computation": "computation", "disposition": "disposition",
    "snapshot": "snapshot", "parity_gate": "parity_gate", "ruling_state": "ruling_state", "source_watermark": "source_watermark",
    "source_state": "source_state", "source_revision": "source_revision", "withdrawal": "withdrawal",
    "source_status": "source_status",
    "mirror_ack": "mirror_ack",
    "file_arrival": "file_arrival",
    # the harvester's durable fact of one refusal judgment: a LEDGER-scale table, folded by its own apply path
    # (reducer.apply_ledger_fact: INSERT OR IGNORE, no row_history), never through the run-scale fold
    "record_refusal": "record_refusal_fact",
    # a filed analysis: three ledger-fact kinds, one row of one table each, the same apply path (INSERT OR IGNORE, no row_history)
    "analysis": "analysis", "analysis_input": "analysis_input", "analysis_artifact": "analysis_artifact",
}

#: LEDGER-FACT tables whose row is kept by an ORDERING KEY rather than by first arrival:
#: table -> (written_utc column, record_sha256 column, record_uri column), three derived columns the apply writes from the record itself.
#: Among the records of one natural key the row kept is the one with the smallest ``(written_utc, record_sha256)`` -- the log's order, which
#: a rebuild reproduces whatever the arrival order -- so the live apply REPLACES a standing row when the incoming record sorts earlier and
#: leaves it when the incoming sorts later; a record whose PAYLOAD differs from the kept one is a refusal by name (`conflicting re-filing`),
#: an identical one a duplicate (``reducer.Reducer.apply_ledger_fact``). All three analysis tables are ordered (the second review round,
#: 2026-10-09: the header alone was, so an `analysis_artifact` link filed twice with another role or page url kept whichever arrived first,
#: and a rebuild kept the other; an `analysis_input`'s `extra_json` is substance of the same kind). `record_refusal_fact` keeps its
#: first-arrival INSERT OR IGNORE: its key names the fact whole.
LEDGER_ORDERED_TABLES: Dict[str, Tuple[str, str, str]] = {
    t: ("record_written_utc", "record_sha256", "record_uri") for t in ("analysis", "analysis_input", "analysis_artifact")}
#: Per ordered table, the payload columns that describe the FILING EVENT rather than the fact's substance: never a conflict when they
#: differ (a re-run of `registry analysis file` on the same spec stamps a new `filed_utc` and is a duplicate, not a conflicting
#: re-filing); the kept row keeps the kept filing's values of them.
LEDGER_ORDERED_EVENT_COLUMNS: Dict[str, Tuple[str, ...]] = {"analysis": ("filed_utc",), "analysis_input": ("filing_sha256",),
                                                            "analysis_artifact": ("filing_sha256",)}
#: Ordered ledger-fact tables whose rows are the CHILDREN of a filing (the fifth closing pass, 2026-10-09): child table -> (the parent
#: table, the child's column naming the parent RECORD's sha -- the `analysis` header that wrote the child). A child folds only while the
#: filing it names is the parent key's KEPT record or a duplicate of it (indexed in ``LEDGER_ORDERED_RECORD_INDEX`` with no standing
#: refusal); a child of a filing that stands refused -- a conflicting or displaced re-filing -- is refused by name with it
#: (``harvest.Ingestor._filing_verdict``), one whose filing has not folded yet is a dependency refusal the retry pass re-selects every cycle
#: (``harvest.FILING_PARENT``), and when the kept filing of a key MOVES every child of the key is judged again from its immutable file
#: (``harvest.Ingestor._rejudge_filing_children``): the displaced filing's children withdrawn and refused, the new kept filing's accepted.
#: Before it, a second filing of an analysis with another report had its header refused while its link to the new report folded on its
#: own -- the kept analysis_id being a row -- so readers attributed a rejected report to the accepted analysis. The binding column is an
#: EVENT column of the ordering rule (above): two filings of one substance write identical children under two header shas, and the later
#: one's children are duplicates, not conflicts.
LEDGER_FILING_CHILDREN: Dict[str, Tuple[str, str]] = {"analysis_input": ("analysis", "filing_sha256"), "analysis_artifact": ("analysis", "filing_sha256")}
#: The index of EVERY record of an ordered table the fold has applied, by the record's table and natural key -- kept, duplicate, displaced
#: and conflicting alike (one row per record sha; the smallest logical uri when the same record arrives under two names). Written by
#: ``reducer.Reducer.apply_ledger_fact`` on every outcome, so it is a pure function of the records; read by the harvester when the kept
#: record of a key MOVES (an earlier-sorting filing arrived late), to judge every other record of the key again against the new kept one
#: (``harvest.Ingestor._rejudge_refilings``). Before it (the second review round, 2026-10-09) only the standing conflict refusals were judged
#: again, so a filing accepted as a DUPLICATE of the displaced one -- which left no row to find -- stayed unrefused while a rebuild, meeting
#: the earliest filing first, refused it: the same records, two refusal tables.
LEDGER_ORDERED_RECORD_INDEX = "ordered_fact_record"
#: Every ARRIVAL of a record of an ordered table, whatever its verdict -- one row per (record sha, logical uri), where the index above keeps
#: one carrier per record -- with the side of the carrying file's delivery audit (`record_delivery`) the record is counted on
#: (``vocab.DELIVERY_SIDES``: `accepted` = `n_ok` or `n_duplicate`, `refused` = `n_refused`). Written by ``harvest.Ingestor.apply_validated``
#: as the fold counts the record; read and moved by ``harvest.Ingestor._move_record_delivery`` when a record is judged again after its file's
#: fold, so every file that carried it follows the verdict. The ninth closing pass, 2026-10-10: the mover followed the index's one
#: canonical carrier, so a displaced filing's copy under a second uri kept `ok` and the status page listed it as ingested.
LEDGER_ORDERED_CARRIERS = "ordered_fact_carrier"

#: Object columns the fold LIFTS from the record's parked extra when the record states no column of that name: the
#: epoch-checkpoint producers wrote `extra_json.parent_object_id` / `extra_json.named_role` before the columns existed, and keep writing
#: them until the window-done mail. The lift is a fold rule over the EVENT's own extra (``reducer.lift_object_fields``), so an old record
#: folds the column exactly as a new one stating it would, `row_history.payload_json` keeps the record as it was validated, and the key
#: stays in `extra_json` too. ``validate.parent_references`` reads the FK among them from the extra as well, so a child whose parent has
#: not folded is dependency-blocked (never refused for that alone).
OBJECT_LIFTED_FROM_EXTRA: Tuple[str, ...] = ("parent_object_id", "named_role")
#: Object columns a record may OMIT when a row for its id exists: `status` is the PRODUCER's statement; a mirror
#: adapter (the objects bank, the staged-object readers) that restated it overwrote a producer statement stamped before the pass but folded
#: after it, and a bank default demoted reference objects. The validator does not require it; the harvester dependency-blocks an object
#: record that omits it while no row exists for the id (``harvest.Ingestor._missing_parents``: missing parent `object` = its own id, never a
#: refusal), and the fold keeps the folded value on omission (per-column latest statement). An explicit null is refused by name.
OBJECT_OMITTABLE_WITH_ROW: Tuple[str, ...] = ("status",)


@dataclass(frozen=True)
class EffectSpec:
    """A record type whose effect is not "a row of one table" (run_state: a run_state_event row + the run's
    state; run_patch: a run field patch; run_ingest: the harvester's ingested stamp; observation: rows keyed by
    a list of keys). ``columns_table`` names whose columns the payload fields are vocabulary-checked against;
    ``required`` the payload fields; ``parents`` the (parent_table, parent_column, payload_key) rows that must
    exist before the record can fold (dependency-blocked otherwise); ``strict_keys`` keeps ONLY the required
    keys and the ``optional`` keys the payload carries (a harvester-only type parks nothing)."""
    columns_table: str
    required: Tuple[str, ...]
    parents: Tuple[Tuple[str, str, str], ...]
    strict_keys: bool = False
    optional: Tuple[str, ...] = ()


EFFECTS: Dict[str, EffectSpec] = {
    "run_state": EffectSpec("run_state_event", ("run_id", "state", "event_utc", "evidence_kind"), (("run", "run_id", "run_id"),)),
    "run_patch": EffectSpec("run", ("run_id",), (("run", "run_id", "run_id"),)),
    # ``judged_utc`` rides only the run_ingest of a RULE-SCORED run (an instrument without a judge): its scoring happened
    # at generation, so the harvester's stamp carries the scoring time (the headline metric's ``computed_utc``, else the
    # run's ``generated_utc``) and the fold sets ``judged_utc`` from it — the row reaches ``ingested`` with every reached
    # state's utc, as ``trg_run_state_utc`` requires, without a judgment row the schema says such a run never has
    "run_ingest": EffectSpec("run", ("run_id", "ingested_utc"), (("run", "run_id", "run_id"),), strict_keys=True,
                             optional=("judged_utc",)),
    "observation": EffectSpec("observation", ("source_id", "revision_id", "table_name", "observation_keys"),
                              (("source_revision", "revision_id", "revision_id"),)),
    # a producer retracts item_result rows it filed under the wrong run: `items` is a list of exact item keys, one
    # `item_withdrawal` row per key (the observation pattern); the named ROWS are parents too, checked by the harvester
    # (`Ingestor._missing_parents`: a composite key is not a single-column FK) -- blocked until they exist, never refused for it
    "item_withdrawal": EffectSpec("item_withdrawal", ("run_id", "items", "reason"), (("run", "run_id", "run_id"),),
                                  optional=("evidence_uri", "belongs_to_run_id")),
}
EFFECT_TYPES: Tuple[str, ...] = tuple(EFFECTS)
assert set(RECORD_TYPE_TABLE) | set(EFFECT_TYPES) == set(RECORD_TYPES), (
    set(RECORD_TYPES) ^ (set(RECORD_TYPE_TABLE) | set(EFFECT_TYPES)))

#: Scale classification.
RUN_SCALE_TABLES: Tuple[str, ...] = S.RUN_SCALE_TABLES
ITEM_SCALE_TABLES: Tuple[str, ...] = S.ITEM_SCALE_TABLES
ITEM_SCALE_RECORD_TYPES: Tuple[str, ...] = V.ITEM_SCALE_RECORD_TYPES
LEDGER_TABLES: Tuple[str, ...] = S.LEDGER_TABLES
ALL_TABLES: Tuple[str, ...] = S.ALL_TABLES

#: The natural key the reducer upserts on, per table (== the table's primary key).
NATURAL_KEYS: Dict[str, Tuple[str, ...]] = {t.name: tuple(t.pk) for t in S.TABLES}
#: What a revision CARRIES for an item: the item's natural key without the revision tag that keys the row.
ITEM_MEMBERSHIP_KEY: Tuple[str, ...] = tuple(c for c in NATURAL_KEYS["item_result"] if c != "source_revision_id")
#: The fields of ONE item key in an ``item_withdrawal`` record's ``items`` list: the membership key without the record's own
#: ``run_id`` -- ``item_id`` and ``metric_id`` are required, the four indices take the item_result defaults / sentinels when absent
#: (``trial_index`` 0, ``sample_index`` 0, ``turn_index`` -1, ``judgment_id`` ''), exactly as ``reducer._key_of`` fills them.
ITEM_WITHDRAWAL_KEY_FIELDS: Tuple[str, ...] = tuple(c for c in ITEM_MEMBERSHIP_KEY if c != "run_id")
ITEM_WITHDRAWAL_KEY_REQUIRED: Tuple[str, ...] = ("item_id", "metric_id")
assert set(ITEM_WITHDRAWAL_KEY_FIELDS) == set(NATURAL_KEYS["item_withdrawal"]) - {"run_id"}, (ITEM_WITHDRAWAL_KEY_FIELDS, NATURAL_KEYS["item_withdrawal"])

# --------------------------------------------------------------------------------------------
# run_patch field sets (S7, Revision 4)
# --------------------------------------------------------------------------------------------
#: The OBSERVED fields a native hook may patch after ``write_cell_manifest``.
RUN_PATCHABLE_FIELDS: Tuple[str, ...] = (
    # job start
    "node_hostname", "cluster_job_id", "job_name", "gpu_type", "gpu_count", "started_utc", "modal_call_id", "modal_app_id",
    # serving (write_endpoint)
    "served_max_num_seqs", "served_max_num_seqs_source", "context_window_tokens", "serving_engine_version",
    "serving_identity", "serving_config_sha256", "adapter_fingerprint", "applied_positions_mode", "applied_layers_json",
    "prefix_cache_policy", "attacker_identity_attested",
    # completion (write_results / write_generation_complete)
    "n_items_generated", "generated_utc", "walltime_s", "walltime_source", "store_completion_artifact_uri",
    "store_completion_sha256", "store_uri",
    # the plan's item count, patchable after the fact: a producer that registered a run with the WHOLE item set's
    # count where the documented per-run count is smaller corrects it by patch; the completion predicate reads
    # COALESCE(n_items_generated, n_items_planned), so the patch can decide whether a run is complete -- and it is an ordinary fold
    # by order key: a later full `run` record carrying the old count re-sets it, so the patch is written after the producer's last
    # `run` record, or the producer stops filing the old count
    "n_items_planned",
    # judging
    "n_items_judged", "judge_id", "judged_utc",
    # decode observed at request time: what the client SENT and the engine's effective sampler are measurements
    # a pipeline re-read may correct after registration; `decode_source` is the declared serving fact the by-source match rule
    # reads, not part of `identity.run_id`, and a full `run` re-emission could already change it -- so it is patchable too.
    # The derived pair (`decode_policy_id`, `decode_policy_match`) is NOT here: RUN_DERIVED_FIELDS, refused by name.
    "decode_sent_json", "decode_resolved_json", "decode_temperature_sent", "decode_top_p_sent", "decode_top_k_sent",
    "decode_source", "instrument_match",
    # strength facts filed after the operating-point ruling
    "strength_nominal", "strength_nominal_source", "is_operating_point",
    # the multi-turn steering scope is DERIVED from the applied-positions mode (patchable above: the standup's endpoint record stamps
    # the served mode after registration) and the eval's category, so it follows the mode: a job-start registration that defaulted it
    # is corrected by `registry reconcile multi-turn-scope`, one run_patch per run, never by hand; it is not part of `identity.run_id`
    "multi_turn_steering_scope",
    # always
    "wandb_run_id", "field_confidence_json", "extra_json", "notes",
)
#: The IMMUTABLE fields a later hook re-observes: sent under ``verify`` and COMPARED, never patched.
RUN_VERIFIED_FIELDS: Tuple[str, ...] = ("serving_path", "intervention_hash", "harness_config_hash", "model_revision", "env_digest")
#: Identity / lifecycle fields no patch may touch (state moves only through run_state records).
RUN_IDENTITY_FIELDS: Tuple[str, ...] = (
    "run_id", "campaign", "producer_run_key_legacy", "producer_object_key_legacy", "model_id", "object_id", "eval_id",
    "item_set_id", "instrument_id", "lifecycle_state", "code_sha", "substrate", "substrate_source", "queued_utc",
    "ingested_utc", "finished_utc", "transfer_kind", "harness", "harness_version", "scoring_rule", "token_cap_tokens",
    "thinking_mode", "decode_seed_rule", "prompt_template_sha256",
)
#: DERIVED run columns no record may set by patch: ``decode_match.stamp_run_rows`` rewrites both from the
#: run's recorded decode fields at the end of every ingest cycle and of a rebuild, so a patched value would be overwritten within
#: the cycle that folded it -- a patch naming one is refused by name, pointing at the fields the pair is derived from. A full
#: ``run`` record may still carry a producer's placeholder (``no_policy``): the same restamp replaces it, as it always has.
RUN_DERIVED_FIELDS: Tuple[str, ...] = ("decode_policy_id", "decode_policy_match")
assert not set(RUN_PATCHABLE_FIELDS) & set(RUN_VERIFIED_FIELDS)
assert not set(RUN_PATCHABLE_FIELDS) & set(RUN_IDENTITY_FIELDS)
assert not set(RUN_DERIVED_FIELDS) & (set(RUN_PATCHABLE_FIELDS) | set(RUN_VERIFIED_FIELDS) | set(RUN_IDENTITY_FIELDS))
_run_cols = set(S.TABLE_BY_NAME["run"].column_names)
assert set(RUN_PATCHABLE_FIELDS) <= _run_cols, set(RUN_PATCHABLE_FIELDS) - _run_cols
assert set(RUN_VERIFIED_FIELDS) <= _run_cols, set(RUN_VERIFIED_FIELDS) - _run_cols
assert set(RUN_DERIVED_FIELDS) <= _run_cols, set(RUN_DERIVED_FIELDS) - _run_cols


def run_patch_extra_json(value) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    """A ``run_patch``'s ``fields.extra_json`` as the object it carries, or the reason it cannot fold: ``(object, None)`` for a
    JSON object or its serialization, ``(None, None)`` for an explicit null (the fold's per-column statement), ``(None, reason)``
    for anything else. ``fields.extra_json`` is the ONE place a patch's stamps and provenance ride, so a value that carries no keys
    -- a serialized array or scalar, text that is not JSON, a non-mapping -- is refused BY NAME at both ends of the rule:
    ``emit.RecordWriter`` before anything is written and ``validate._validate_run_patch`` before the fold. Never dropped: the
    payload normalizer keeps only an object's keys, and before this rule a malformed value was dropped silently there while the
    patch's typed columns folded without the writer's ``run_overrides`` / ``cluster_label`` stamps. The reason names the shape,
    never the value (a malformed value may be large, or someone's data). Whole-log audit at the rule's landing (2026-10-02,
    read-only): 0 of the ``run_patch`` records in the log carry such a value (``fields.extra_json`` is absent or an object on every
    one), so historical replay is unchanged and nothing is grandfathered."""
    if value is None:
        return None, None
    if isinstance(value, Mapping):
        return dict(value), None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as e:
            return None, (f"run_patch fields.extra_json is not valid JSON ({e.msg} at char {e.pos}): a patch's stamps and provenance "
                          "ride there, so it must be a JSON object")
        if isinstance(parsed, dict):
            return parsed, None
        return None, (f"run_patch fields.extra_json must be a JSON object, got a serialized {type(parsed).__name__}: it carries no keys "
                      "for the patch's stamps and provenance to ride under")
    return None, (f"run_patch fields.extra_json must be an object, got {type(value).__name__}: it carries no keys for the patch's "
                  "stamps and provenance to ride under")

# --------------------------------------------------------------------------------------------
# Required fields per record type
# --------------------------------------------------------------------------------------------


def required_fields(record_type: str) -> Tuple[str, ...]:
    """The payload fields a record of this type must carry: the table's NOT NULL columns without a default
    (the derived tail excluded), or the effect type's declared fields."""
    if record_type in RECORD_TYPE_TABLE:
        return tuple(S.required_fields(RECORD_TYPE_TABLE[record_type]))
    return EFFECTS[record_type].required


REQUIRED_FIELDS: Dict[str, Tuple[str, ...]] = {t: required_fields(t) for t in RECORD_TYPES}

#: Item-scale record types must carry EVERY column of the row (whole-row); a partial record
#: is a Malformed refusal naming the missing columns.
#: The columns an item-scale record must carry (the WHOLE row) — all but `extra_json` and the revision tag: the tag
#: is the adapter runner's to set, natively emitted rows leave it at its '' sentinel.
ITEM_WHOLE_ROW_COLUMNS: Dict[str, Tuple[str, ...]] = {
    t: tuple(c for c in S.payload_columns(t) if c not in ("extra_json", "source_revision_id")) for t in ITEM_SCALE_RECORD_TYPES}

# --------------------------------------------------------------------------------------------
# Envelope helpers
# --------------------------------------------------------------------------------------------

UTC_MS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
_UTC_MS_PARTS = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d{3})Z$")
_EPOCH_DATE = date(1970, 1, 1)
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


def _fmt_utc(dt: datetime) -> str:
    """THE spelling of a registry timestamp: ISO-8601 UTC with milliseconds, ``2026-09-16T18:04:11.312Z``."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def utc_now() -> str:
    return _fmt_utc(datetime.now(timezone.utc))


@functools.lru_cache(maxsize=4096)
def utc_to_epoch_ms(utc: str) -> int:
    """Exact integer epoch milliseconds from the registry spelling — field parsing and integer arithmetic only (F6:
    ``strptime`` + ``int(dt.timestamp() * 1000)`` returned ms − 1 for 0.65 % of stamps from ~2038 on, so the integer
    stamp and the string order key could disagree)."""
    m = _UTC_MS_PARTS.match(utc)
    if not m:
        raise ValueError(f"not an ISO-8601 UTC-with-ms timestamp: {utc!r}")
    y, mo, d, h, mi, sec, ms = (int(g) for g in m.groups())
    if h > 23 or mi > 59 or sec > 59:
        raise ValueError(f"not a valid UTC time of day: {utc!r}")
    days = (date(y, mo, d) - _EPOCH_DATE).days   # date() refuses an impossible calendar day
    return ((days * 86400 + h * 3600 + mi * 60 + sec) * 1000) + ms


def epoch_ms_to_utc(ms: int) -> str:
    """The exact inverse of :func:`utc_to_epoch_ms` (integer arithmetic; no float instant)."""
    secs, rem = divmod(int(ms), 1000)
    days, sod = divmod(secs, 86400)
    d = _EPOCH_DATE + timedelta(days=days)
    h, r = divmod(sod, 3600)
    mi, sec = divmod(r, 60)
    return f"{d.year:04d}-{d.month:02d}-{d.day:02d}T{h:02d}:{mi:02d}:{sec:02d}.{rem:03d}Z"


def record_sha(record: Mapping[str, object]) -> str:
    """sha256 over the record's canonical JSON — the record's identity for dedupe and ordering."""
    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def first_nonfinite(obj, path: str = "") -> Optional[str]:
    """The dotted path (``extra.stats.ratios[1]``) of the first non-finite float in a nested value, or None. A record
    carries finite numbers only: ``NaN`` / ``Infinity`` / ``-Infinity`` are what Python's ``json`` writes AND reads for
    them, but RFC 8259 has no such literals and SQLite's ``json_valid`` refuses them — so a record holding one is not
    JSON, and every ``*_json`` column it would reach (``revision_staging_record.record_json``, ``row_history.payload_json``,
    ``extra_json``) fails its CHECK. The validator refuses the record by this path; a producer that has a
    ratio over a zero denominator records a STATE (``not_measured`` / ``refused``), never a number."""
    if isinstance(obj, float):
        return None if math.isfinite(obj) else path
    if isinstance(obj, dict):
        for k, v in obj.items():
            hit = first_nonfinite(v, f"{path}.{k}" if path else str(k))
            if hit is not None:
                return hit
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hit = first_nonfinite(v, f"{path}[{i}]")
            if hit is not None:
                return hit
    return None


def make_record(record_type: str, payload: Mapping[str, object], producer: Mapping[str, object],
                record_id: str, written_utc: Optional[str] = None) -> Dict[str, object]:
    return {"record_type": record_type, "schema_version": SCHEMA_VERSION, "record_id": record_id,
            "written_utc": written_utc or utc_now(), "producer": dict(producer), "payload": dict(payload)}


# --------------------------------------------------------------------------------------------
# Record files
# --------------------------------------------------------------------------------------------

RECORD_FILE_RE = re.compile(r"^(?P<seq>\d+)-(?P<type>[a-z_]+)\.jsonl(?P<gz>\.gz)?$")


def record_file_name(seq: int, record_type: str, gz: bool = False) -> str:
    return f"{seq:06d}-{record_type}.jsonl" + (".gz" if gz else "")


def logical_uri(path: str) -> str:
    """The record_uri of a file: its path without a trailing ``.gz`` (compression never changes identity)."""
    return path[:-3] if path.endswith(".gz") else path


def open_record_file(path: str):
    """Open a plain or gzip record file for text reading."""
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _sha256_stream(fh) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
    return h.hexdigest()


def file_sha256(path: str) -> str:
    """sha256 of a file's bytes, streamed (artifacts, snapshot files, manifests)."""
    with open(path, "rb") as fh:
        return _sha256_stream(fh)


def file_logical_sha(path: str) -> str:
    """sha256 of the DECOMPRESSED bytes of a record file — what every mirror manifest hashes."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            return _sha256_stream(fh)
    return file_sha256(path)


def record_file_id_from_sha(sha_hex: str) -> int:
    """First 63 bits of a sha256 hex digest as a positive integer."""
    return int(sha_hex[:16], 16) >> 1


def record_file_id_for_uri(file_uri: str) -> int:
    """The integer id item rows point at: first 63 bits of sha256(logical file uri) — arrival-independent, one
    per file (two files with identical bytes under different names are two files)."""
    return record_file_id_from_sha(hashlib.sha256(file_uri.encode("utf-8")).hexdigest())


def peek_written_utc(path: str) -> str:
    """The ``written_utc`` of a record file's FIRST record, or '' when unreadable — a cheap ordering hint (the
    harvester parses a cycle's files in this order so ranks are assigned in log order; the ledger itself uses
    the exact first key, so a hint that is wrong costs a re-rank, never correctness)."""
    try:
        with open_record_file(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    w = json.loads(line).get("written_utc")
                    return w if isinstance(w, str) else ""
    except (OSError, EOFError, gzip.BadGzipFile, ValueError, AttributeError):
        pass
    return ""


def open_record_file_bytes(path: str):
    """Open a plain or gzip record file for BINARY reading (the logical bytes)."""
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


class RecordLineError(ValueError):
    """A line of a record file that is not a JSON object. The message names the path and line (``<path>:<line>: <detail>``);
    ``line`` and ``detail`` let the harvester name the LOGICAL uri instead (``<uri>#<line>: <detail>``), so the recorded
    reason is the same on every machine that replays the log (an absolute path in a refusal broke live == replay)."""

    def __init__(self, path: str, line: int, detail: str):
        super().__init__(f"{path}:{line}: {detail}")
        self.path, self.line, self.detail = path, line, detail


def read_record_file(path: str, hasher=None, compute_sha: bool = True) -> Iterator[Tuple[int, Dict[str, object], Optional[str]]]:
    """Yield ``(line_no, record, record_sha256)`` for every non-blank line; a line that is not JSON raises
    :class:`RecordLineError` (a ``ValueError``) naming the path and line. A ``hasher`` (``hashlib.sha256()``) is fed every
    logical byte read, so the caller learns the sha of exactly the bytes it parsed without a second pass. Python's
    ``json`` accepts the non-JSON literals ``NaN`` / ``Infinity``; a record carrying one is read here and refused by the
    validator (:func:`first_nonfinite`), so the refusal names the field rather than the file. ``compute_sha=False`` yields
    ``None`` for the sha (a selective reader that will discard most records hashes only the ones it keeps: the canonical
    re-serialisation and digest cost about as much as the parse)."""
    with open_record_file_bytes(path) as fh:
        for i, raw in enumerate(fh, start=1):
            if hasher is not None:
                hasher.update(raw)
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise RecordLineError(path, i, f"not JSON ({getattr(exc, 'msg', exc)})") from exc
            if not isinstance(rec, dict):
                raise RecordLineError(path, i, "record is not a JSON object")
            yield i, rec, (record_sha(rec) if compute_sha else None)


def write_records(path: str, records: Sequence[Mapping[str, object]], gz: bool = False, exclusive: bool = False) -> str:
    """Append records to a file (creating it); returns the path written. ``exclusive=True`` CREATES the
    file (``O_EXCL``) and raises ``FileExistsError`` when it already exists: a producer's record file is
    immutable once written (the push and pull transports copy with ignore-existing semantics, so a file
    appended after its first copy would never travel again), so a writer that finds its sequence number
    taken by another writer of the same run directory takes the next one instead of appending."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = "".join(canonical_json(r) + "\n" for r in records).encode("utf-8")
    if gz or path.endswith(".gz"):
        if not path.endswith(".gz"):
            path += ".gz"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | (os.O_EXCL if exclusive else 0)
    fd = os.open(path, flags, 0o644)
    if path.endswith(".gz"):
        # gzip append: each write is its own member; readers concatenate members transparently
        with os.fdopen(fd, "ab") as raw, gzip.GzipFile(fileobj=raw, mode="ab", mtime=0) as fh:
            fh.write(data)
    else:
        with os.fdopen(fd, "ab") as fh:
            fh.write(data)
    return path


def iter_record_files(root: str) -> Iterator[str]:
    """Every record file under ``root`` (plain and .gz), sorted for determinism."""
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in filenames:
            if fn.endswith(".jsonl") or fn.endswith(".jsonl.gz"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return iter(out)



# --------------------------------------------------------------------------------------------
# Pointer files (snapshot CURRENT, mirror CHECKSUMS.CURRENT): one line, written .tmp + fsync + rename
# --------------------------------------------------------------------------------------------


def read_pointer(path: str) -> Optional[str]:
    """The one line a pointer file names, or None when the file is absent or empty."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().strip() or None


def write_pointer(path: str, value: str) -> None:
    """Atomically point ``path`` at ``value``: ``.tmp`` + fsync + rename (atomic on every filesystem here)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(value + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------
# JSON state files (sidecars under state/): read leniently, written whole through a unique temp name + fsync + rename
# --------------------------------------------------------------------------------------------


def read_json_or_none(path: str) -> Optional[object]:
    """The parsed JSON of ``path``, or None when the file is absent, unreadable or not JSON (a torn write): the caller treats
    None as "no state" and rebuilds — a sidecar is a cache of what the records and the disk already say."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_json_atomic(path: str, obj: object, indent: Optional[int] = 0, **dump_kwargs) -> None:
    """Write ``path`` whole: a PER-PROCESS temp name (a fixed ``<path>.tmp`` let two writers interleave into one file — a valid
    document followed by another document's tail, the mirror ledger 2026-09-19), fsync, then ``os.replace``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, **dump_kwargs)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
