#!/usr/bin/env python3
"""qwen-seedrep-native-battery — the request-evidence audit.

    request_evidence_audit.py --stores RS/stores --out RS/audits [--policy-file configs/registry/decode_policy.json]
        [--roots <attempt root> ...] [--legs pasttense,bon,...] [--petri-behaviours b1,b2,...]
        [--thinking-expect gsm8k=true --thinking-expect mmlu=none ...] [--model-key qwen35_4b]

For EVERY pulled accepted attempt root (`RS/stores/<leg>/<view>/<shard>/`) and every store of every
table leg it holds, ONE verdict `AGREES` | `REFUSED(<why>)` written to `audits/request_log/<leg>/<arm>__s<s>.json`,
plus `audits/request_log_summary.csv` (405 rows at the full wave). The verdict engine is
`contracts.request_evidence_reconciliation` — imported, never re-implemented — run TWICE per store:

1. **Engine-wide identity, per (task, vmodel) partition** (rev 3 / rev 5): the attempt root's `request_evidence.jsonl`
   and `usage_ledger.jsonl` are partitioned on the labels BOTH sides carry (`task`, `vmodel`; the probes as their own
   partition), and every partition must be ONE inventory (request ids unique on each side, every ledger attempt has
   its record and every record its attempt, digests and labels agree). A torn label lands in two partitions and refuses
   on both. No ledger row and no record is left outside every partition.
2. **The store's verdict**: the partition of the store's canonical task(s) × the arm's vmodel (`endpoint.json.vmodels`, the arm
   resolved in EITHER spelling through the standup's records — `audit_common.standup_vmodel_of`, F-AN)
   through the FULL reconciliation with `filed` = the store's own item pairs (:mod:`audit_common`'s per-leg readers —
   the store's own row keys), `expected_sampling` = the leg's frozen row, `expected_seed` = the leg's seed rule
   closed over the store's recorded base, `target_vmodel` = the arm's vmodel, `expected_enable_thinking` per leg
   (GSM8K True; every other leg the NO-SWITCH reading, overridable). The `X-Tinycap-Task` must be the store's row task
   — a lane label (`gsm8k/none`) or the Inspect variant of a sampled control (`inspect/xstest`) returns
   "filed item(s) have NO recorded attempt" from the coverage clause.
3. **Every stored row binds to its attempt AND to the engine** (rev 3): each draw's `request_id` names a
   2xx ledger row and a TASK record whose task / item / sample / vmodel are the draw's; its `completion_id` equals the
   AUTHORITATIVE completion of that attempt — the usage ledger's additive `completion_id` field or a `response_evidence.jsonl`
   row, REQUIRED for every filed successful draw (an attempt with neither is a binding failure, never a skipped check) —
   and matches EXACTLY ONE `Received request <id>` line of the engine log by `contracts.is_engine_id_of` (the external id
   plus one `-<8 hex>` suffix). A row with no attempt, an attempt with no authoritative completion or no engine line, or
   two engine lines for one completion is a binding failure named per row.

A store whose seed rule has not landed on the tree is REFUSED by name — the audit never invents a
seed rule. `resolved_sampler.json` (`record_resolved_sampler.py --strict`, the spec's secondary interface) is read and
its verdict recorded beside the primary one when the root holds it (`ABSENT` otherwise); it never establishes AGREES.

Exit codes: 0 every store AGREES; 1 the table was written and at least one store is REFUSED; 2 an input is missing or
malformed (nothing partial is written).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit_common as ac  # noqa: E402
from audit_common import Standup, C, AttemptRoot, StoreRead, refuse  # noqa: E402

VERDICT_AGREES = "AGREES"
VERDICT_REFUSED = "REFUSED"
SUMMARY_NAME = "request_log_summary.csv"
#: F-AO (fp3): the WRITER's index of every verdict this run filed — `<out>/request_log_index.json`: per verdict its rel under `<out>`
#: (`request_log/<table leg>/<file>`), leg / s / stage / shard / attempt id, the STORE's arm (the filename spelling, preserved), and its
#: `arm_binding` (harness id, vmodel, spelling) — so a consumer that knows a plan child's CELL (the app's `producer_evidence`,
#: `seedrep_served.bind_audit_verdicts`) resolves the ONE file the audit wrote through the audit's own binding, never by composing a name.
#: Held equal to `seedrep_served.AUDIT_INDEX_NAME` by tests/test_seedrep_fp3_verdict_binding.py.
INDEX_NAME = "request_log_index.json"
INDEX_SCHEMA = "request_log_index/1"
SUMMARY_COLUMNS: Tuple[str, ...] = (
    "leg", "arm", "s", "view", "stage", "shard", "attempt_id", "verdict", "problem?", "n_records_task", "n_ledger_rows_partition?",
    "n_filed", "n_draws", "n_draws_bound", "n_engine_received", "seed_rule", "seed_base?", "expected_sampling_source",
    # the frozen row the wire was HELD TO (on AGREES = the decode AS SENT, every term equal by the sampling gate)
    "temperature", "top_p?", "top_k?", "min_p?", "presence_penalty?", "repetition_penalty?", "target_vmodel",
    "thinking_expect", "resolved_sampler_verdict", "client_task_names", "store_dir",
    # the standup record's provenance (F-AA): `standup_jobs.jsonl`, or the derivation the vmodel binding was read under
    "standup_record",
    # the usage ledger's place in this row's request path (`declared` | `present_undeclared` | `not_on_path`), the attempt
    # inventory the identity clause ran over, and the contract version the verdict was issued under — never AGREES silently
    "ledger", "attempt_source", "reconciliation_contract")
RESOLVED_SAMPLER_FILE = "resolved_sampler.json"
DEFAULT_SHARD = "shard00"


def verdict_filename(arm: str, s: int, *, stage: str, shard: str) -> str:
    """`<arm>__s<s>[__cap<N>][__<shard>].json` — the base root's `shard00` verdict is the plan's `<arm>__s<s>.json`; a rung
    root appends its cap and a non-default shard its name, so no two stores share a verdict path."""
    name = f"{arm}__s{s}"
    if stage != ac.STAGE_BASE:
        name += f"__cap{stage}"
    if shard != DEFAULT_SHARD:
        name += f"__{shard}"
    return name + ".json"


# --------------------------------------------------------------------------- #
# Partitions                                                                    #
# --------------------------------------------------------------------------- #

def _thinking_label(v: Optional[bool]) -> str:
    return "none" if v is None else ("true" if v else "false")


def partition_key(task: Any, vmodel: Any) -> Tuple[str, str]:
    """`("probe", "*")` for a probe-labelled call (the probes render either way and are one inventory across
    routes), else `(task, vmodel)`; a missing label is spelled `"<none>"` so it forms its own — refusing —
    partition rather than joining another."""
    if C.is_probe_task(task):
        return ("probe", "*")
    return (str(task) if task is not None else "<none>", str(vmodel) if vmodel is not None else "<none>")


def partitions(records: Sequence[Mapping[str, Any]], ledger_rows: Sequence[Mapping[str, Any]]
               ) -> Dict[Tuple[str, str], Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]]:
    """The engine-wide (task, vmodel) partitions of the records and the attempt rows. An attempt row is attributed through its
    REQUEST RECORD — the record is the labelled side of the pair — and only a row naming NO record falls to its own labels, where
    the identity clause refuses it as an orphan (fz4: the response hook's record of a FAILED attempt carries
    `vmodel=None` — no body to read `model` off — so partitioning it by its own labels put a 503-then-retry's failed attempt in a
    spurious `<none>` partition and reported it missing from the real model's inventory, forcing every store verdict REFUSED over a
    retry that produced a valid, fully bound completion). Contradictory labels between a row and its record are the reconciliation's
    label clause to refuse, inside the record's partition — never split across two."""
    out: Dict[Tuple[str, str], Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]] = {}
    key_of_rid: Dict[str, Tuple[str, str]] = {}
    for r in records:
        key = partition_key(r.get("task"), r.get("vmodel"))
        out.setdefault(key, ([], []))[0].append(r)
        if r.get("request_id"):
            key_of_rid[str(r["request_id"])] = key
    for row in ledger_rows:
        rid = row.get(C.LEDGER_REQUEST_ID_FIELD)
        key = key_of_rid.get(str(rid)) if rid else None
        if key is None:
            key = partition_key(row.get("task"), row.get("vmodel"))  # an orphan: refused by the identity clause where its labels put it
        out.setdefault(key, ([], []))[1].append(row)
    return out


def engine_wide_problems(records: Sequence[Mapping[str, Any]], ledger_rows: Sequence[Mapping[str, Any]], *,
                         thinking_expect_of_task: Mapping[str, Optional[bool]], default_thinking: Optional[bool],
                         attempt_source: str = "ledger") -> Dict[str, Optional[str]]:
    """The identity reconciliation per (task, vmodel) partition — `{partition label: None | refusal}`. The
    probe partition runs without a sampling row; a task partition runs with the task's thinking expectation
    (falling back to `default_thinking`) and no sampling row (the store verdict adds those)."""
    out: Dict[str, Optional[str]] = {}
    for (task, vmodel), (recs, rows) in sorted(partitions(records, ledger_rows).items()):
        label = f"{task} @ {vmodel}"
        if task == "probe":
            why = (C.request_evidence_reconciliation(recs, rows, expected_enable_thinking=None, attempt_source=attempt_source)
                   if recs or rows else None)
            if why is not None and "did not render" in why:
                why = None  # probes may render either way; identity problems keep their text
            out[label] = why
            continue
        expect = thinking_expect_of_task.get(task, default_thinking)
        out[label] = C.request_evidence_reconciliation(recs, rows, expected_enable_thinking=expect, attempt_source=attempt_source)
    return out


# --------------------------------------------------------------------------- #
# Row → attempt → engine binding                                                #
# --------------------------------------------------------------------------- #

def _ok_status(row: Mapping[str, Any]) -> bool:
    st = row.get("status")
    return (not row.get("rejected")) and isinstance(st, int) and 200 <= st < 300


def completion_ids_by_request(root: AttemptRoot, ledger_rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """The AUTHORITATIVE `(request_id → completion_id)` association of the attempt root: the usage ledger's additive
    `completion_id` field and/or `response_evidence.jsonl`'s `{request_id, completion_id}` rows
    (the Inspect-family response hook, step 3b). Both are records the CLIENT SIDE did not write into the store row, which is
    what makes them evidence; when both name one request they must agree (a disagreement refuses the root — two records
    of one attempt naming two completions). A request present in neither has NO authoritative completion and every row
    claiming it fails to bind (:func:`bind_rows`) — the check is never skipped for want of the field."""
    out: Dict[str, str] = {}
    for row in ledger_rows:
        rid, cid = row.get(C.LEDGER_REQUEST_ID_FIELD), row.get("completion_id")
        if rid and cid:
            out[str(rid)] = str(cid)
    for r in ac.completed_response_rows(root.path):  # COMPLETIONS only: a failed retry attempt (completion_id None) is never one (fz2 HOLD A)
        rid, cid = r["request_id"], r["completion_id"]
        prior = out.get(str(rid))
        if prior is not None and prior != str(cid):
            refuse(f"{root.label}: the usage ledger records completion_id {prior!r} for request {rid!r} but "
                   f"{ac.RESPONSE_EVIDENCE_FILE} says {cid!r} — two records of one attempt name two completions")
        out[str(rid)] = str(cid)
    return out


@dataclass
class Evidence:
    """ONE attempt root's request pipeline as the binding reads it (or the MERGE of several accepted roots of one
    (leg, view, stage) — the shards of a static ancestor stage — request ids are attempt-unique, so the union is one
    inventory): the request-evidence records and 2xx ledger rows by request id, every engine-log `Received request`
    id, the authoritative completion per attempt (:func:`completion_ids_by_request`) and the arm → vmodel map(s)."""

    label: str
    records: List[Mapping[str, Any]]
    ledger_rows: List[Mapping[str, Any]]
    records_by_id: Dict[str, Mapping[str, Any]]
    ledger_by_id: Dict[str, Mapping[str, Any]]
    engine_ids: List[str]
    #: external id → the engine-log ids that are it, or it + exactly one `-<8 hex>` suffix (`contracts.is_engine_id_of`),
    #: in log order with duplicates kept — the per-draw engine binding reads this instead of scanning the whole log
    engine_by_external: Dict[str, List[str]]
    #: `Received request` ids the log(s) carry more than once — an engine that logged one completion twice
    dup_engine_ids: List[str]
    completion_by_rid: Dict[str, str]
    #: (task, sample header) → the completed attempts `response_evidence.jsonl` recorded for it (the Inspect family's
    #: binding) — the association an id-less filed row resolves against, on THIS evidence
    #: (a copied-forward controls row resolves on its ANCESTOR's, chosen before any id is looked up)
    completed_by_sample: ac.ResponseIndex
    #: request_id → the response record's additive `text_sha256` (lane-on) — the stored text is held to the response the
    #: selected attempt actually produced (without it a row can be bound to an unrelated completed attempt)
    response_sha_by_rid: Dict[str, str]
    standups: List[Standup]
    attempt_ids: List[str]
    engine_logs: List[Path]
    #: the usage ledger's place in this root's request path (`audit_common.LEDGER_STATUSES`) and its source
    ledger: str = ac.LEDGER_DECLARED
    ledger_source: str = ""
    #: the ATTEMPT inventory the identity clause and the row binding run over: the ledger's rows (`"ledger"`) or, on a `not_on_path`
    #: row, the response hook's rows (`"response_evidence"`, one per attempt that returned or died — `contracts.request_evidence_
    #: reconciliation(attempt_source=…)`); `attempts_by_id` is that inventory by request id
    attempt_source: str = "ledger"
    attempt_rows: List[Mapping[str, Any]] = field(default_factory=list)
    attempts_by_id: Dict[str, Mapping[str, Any]] = field(default_factory=dict)
    #: the controls-events contract (fc3): every engine `Received request` line's PARAMS by engine id (`audit_common.engine_request_params`:
    #: seed, max_tokens, the sampler keys, lora_name) -- filled only when the root reconciles on its controls event streams; the roots the
    #: evidence was loaded from (an ancestor's controls stores are located through them)
    engine_params_by_id: Dict[str, dict] = field(default_factory=dict)
    roots: List[AttemptRoot] = field(default_factory=list)

    def __post_init__(self) -> None:
        # an Evidence built the older way (ledger fields only — the unit tests' and any older caller's construction) IS a
        # ledger-backed inventory: derive the attempt view from the ledger view rather than reading an empty inventory as "no attempt"
        if self.attempt_source == "ledger" and not self.attempt_rows and not self.attempts_by_id and (self.ledger_rows or self.ledger_by_id):
            self.attempt_rows = list(self.ledger_rows)
            self.attempts_by_id = dict(self.ledger_by_id) or {str(r.get(C.LEDGER_REQUEST_ID_FIELD)): r for r in self.ledger_rows
                                                              if r.get(C.LEDGER_REQUEST_ID_FIELD)}

    @property
    def contract(self) -> str:
        """The reconciliation contract this root's evidence shape was reconciled under (`audit_common.CONTRACT_OF_ATTEMPT_SOURCE`); an unknown
        attempt source refuses rather than defaulting to the ledger-backed contract."""
        try:
            return ac.CONTRACT_OF_ATTEMPT_SOURCE[self.attempt_source]
        except KeyError:
            refuse(f"{self.label}: attempt_source {self.attempt_source!r} names no reconciliation contract ({sorted(ac.CONTRACT_OF_ATTEMPT_SOURCE)})")

    def binding_of(self, arm: str) -> ac.VmodelBinding:
        """The arm's binding on this evidence — the standup's own (`Standup.binding_of`: the arm in EITHER spelling, resolved through the
        standup's records, an unserved arm refused by name — F-AN); merged ancestor roots must agree on the harness cell and the vmodel."""
        if not self.standups:
            refuse(f"{self.label}: no standup record on this evidence — arm {arm!r} cannot be bound to a vmodel")
        bindings = [st.binding_of(arm) for st in self.standups]
        vals = {(b.harness_id, b.vmodel) for b in bindings}
        if len(vals) != 1:
            refuse(f"{self.label}: arm {arm!r} is served as {sorted(vals)} across its roots — one vmodel is required")
        return bindings[0]

    def vmodel_of(self, arm: str) -> str:
        return self.binding_of(arm).vmodel

    def lora_expectation(self, arm: str) -> Tuple[Optional[str], str]:
        """fc6: the LoRA name the engine must state for `arm`'s requests, or None for an arm the standup records as served on the base weights
        (`Standup.engine_lora_expectation`); merged ancestor roots must agree."""
        if not self.standups:
            refuse(f"{self.label}: no standup record on this evidence -- arm {arm!r}'s engine identity cannot be established")
        got = [st.engine_lora_expectation(arm) for st in self.standups]
        if len({g[0] for g in got}) != 1:
            refuse(f"{self.label}: arm {arm!r}'s LoRA identity differs across its roots ({sorted({str(g[0]) for g in got})}) -- one identity is required")
        return got[0]

    @staticmethod
    def merge(parts: Sequence["Evidence"], label: str) -> "Evidence":
        out = Evidence(label=label, records=[], ledger_rows=[], records_by_id={}, ledger_by_id={}, engine_ids=[], engine_by_external={},
                       dup_engine_ids=[], completion_by_rid={}, completed_by_sample={}, response_sha_by_rid={}, standups=[], attempt_ids=[],
                       engine_logs=[])
        srcs = {e.attempt_source for e in parts}
        if len(srcs) > 1:
            refuse(f"{label}: the roots merged here reconcile on different attempt inventories ({sorted(srcs)}) — one stage, one contract")
        if parts:
            out.attempt_source, out.ledger = parts[0].attempt_source, parts[0].ledger
            out.ledger_source = "; ".join(sorted({e.ledger_source for e in parts}))
        for e in parts:
            dup = set(out.records_by_id) & set(e.records_by_id)
            if dup:
                refuse(f"{label}: request id(s) {sorted(dup)[:3]} appear on two accepted roots of one stage — the pull mixed attempts")
            out.attempt_rows.extend(e.attempt_rows)
            out.attempts_by_id.update(e.attempts_by_id)
            out.records.extend(e.records)
            out.ledger_rows.extend(e.ledger_rows)
            out.records_by_id.update(e.records_by_id)
            out.ledger_by_id.update(e.ledger_by_id)
            out.engine_ids.extend(e.engine_ids)
            for k, v in e.engine_by_external.items():
                out.engine_by_external.setdefault(k, []).extend(v)
            out.dup_engine_ids.extend(e.dup_engine_ids)
            out.completion_by_rid.update(e.completion_by_rid)
            for k, v in e.completed_by_sample.items():
                out.completed_by_sample.setdefault(k, []).extend(v)
            out.response_sha_by_rid.update(e.response_sha_by_rid)
            out.standups.extend(e.standups)
            out.attempt_ids.extend(e.attempt_ids)
            out.engine_logs.extend(e.engine_logs)
            out.engine_params_by_id.update(e.engine_params_by_id)
            out.roots.extend(e.roots)
        return out


_ENGINE_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$")  # the ONE internal-id suffix `contracts.is_engine_id_of` admits


def engine_id_index(engine_ids: Sequence[str]) -> Dict[str, List[str]]:
    """external id → the engine-log ids `contracts.is_engine_id_of` matches for it (the id itself, or the id plus exactly
    one `-<8 lowercase hex>` suffix), in log order with duplicates kept — one pass over the log instead of a scan per draw."""
    out: Dict[str, List[str]] = {}
    for e in engine_ids:
        out.setdefault(e, []).append(e)
        if len(e) > 9 and _ENGINE_SUFFIX_RE.search(e):
            out.setdefault(e[:-9], []).append(e)
    return out


def load_evidence(root: AttemptRoot, *, reads: Optional[ac.RootReads] = None, cache: Optional[Dict[str, Evidence]] = None,
                  plan: Optional[ac.LaunchPlan] = None) -> Evidence:
    """ONE attempt root's request pipeline — standup, records, attempt inventory, engine log — as :class:`Evidence`. An engine log
    without `Received request` lines refuses (rows cannot be bound). `cache` (per run) hands back the same object for a
    root loaded before, so the ancestor every rung root binds on is loaded once. The usage ledger's place comes from
    the BOUND PLAN (`ac.ledger_status`: the row's children's `ledger.ledger_path_rel`; the plan builders' leg contract when no
    plan is bound) — `declared` / `present_undeclared` read the ledger (a declared one missing refuses), `not_on_path` reconciles
    on the response hook's attempt rows (`attempt_source="response_evidence"`), recorded on the evidence and every verdict."""
    key = str(root.path)
    if cache is not None and key in cache:
        return cache[key]
    reads = reads or ac.RootReads()
    standup = ac.read_standup(root)
    controls_root = root.leg == ac.CONTROLS_CONTAINER_LEG
    records = reads.records(root, required=not controls_root)
    ledger, ledger_source = ac.ledger_status(root, plan)
    controls_events_mode = False
    if ledger == ac.LEDGER_NOT_ON_PATH:
        ledger_rows: List[dict] = []
        attempt_rows: List[Mapping[str, Any]] = list(ac.response_evidence_rows(root.path))  # EVERY attempt, failed ones included
        attempt_source = ac.ATTEMPT_SOURCE_RESPONSE_EVIDENCE
        if not attempt_rows and controls_root and not records:
            # fc3: the controls driver wires no request-evidence client -- no request_evidence.jsonl, no response_evidence.jsonl anywhere
            # under the root (the controls-arditi shape) -- so the reconciliation runs PER STORE over the driver's own controls event
            # stream + the engine log (`audit_controls_store_events`), the MMLU readout at the count level (`audit_mmlu_store_count`); the
            # contract each verdict is issued under says so by name. A controls root that DOES carry request evidence keeps the ledger path.
            controls_events_mode, attempt_source = True, ac.ATTEMPT_SOURCE_CONTROLS_EVENTS
            ledger_source += f"; D133: no {ac.EVIDENCE_FILE} / {ac.RESPONSE_EVIDENCE_FILE} on a controls root -> reconciled on controls_events.jsonl + the engine log"
        elif not attempt_rows:
            refuse(f"{root.path}: the plan declares no usage ledger for this row and {ac.RESPONSE_EVIDENCE_FILE} holds no rows — no attempt "
                   "inventory to reconcile the request records against (D106 needs the response hook's rows)")
    else:
        ledger_rows = reads.ledger_rows(root)
        attempt_rows, attempt_source = list(ledger_rows), ac.ATTEMPT_SOURCE_LEDGER
    log_p = ac.find_engine_log(root)
    engine_ids = ac.engine_request_ids(log_p)
    if not engine_ids:
        refuse(f"{log_p}: no `Received request` line — the engine did not log requests (was --enable-log-requests set?); "
               "rows cannot be bound to the engine")
    ev = Evidence(label=root.label, records=list(records), ledger_rows=list(ledger_rows),
                  records_by_id={str(r.get("request_id")): r for r in records if r.get("request_id")},
                  ledger_by_id={str(r.get(C.LEDGER_REQUEST_ID_FIELD)): r for r in ledger_rows if r.get(C.LEDGER_REQUEST_ID_FIELD)},
                  engine_ids=list(engine_ids), engine_by_external=engine_id_index(engine_ids), dup_engine_ids=ac.duplicates(engine_ids),
                  completion_by_rid=completion_ids_by_request(root, ledger_rows), completed_by_sample=ac.response_evidence_index(root.path),
                  response_sha_by_rid=ac.response_text_shas_by_rid(root.path), standups=[standup], attempt_ids=[root.attempt_id],
                  engine_logs=[log_p], ledger=ledger, ledger_source=ledger_source, attempt_source=attempt_source, attempt_rows=attempt_rows,
                  attempts_by_id={str(r.get(C.LEDGER_REQUEST_ID_FIELD)): r for r in attempt_rows if r.get(C.LEDGER_REQUEST_ID_FIELD)},
                  engine_params_by_id=(ac.engine_request_params(log_p) if controls_events_mode else {}), roots=[root])
    if cache is not None:
        cache[key] = ev
    return ev


def ancestor_evidence(root: AttemptRoot, siblings: Sequence[AttemptRoot], *, leg: str, cap: int, petri_behaviours: Sequence[str],
                      reads: Optional[ac.RootReads] = None, cache: Optional[Dict[str, Evidence]] = None,
                      plan: Optional[ac.LaunchPlan] = None) -> Evidence:
    """The MERGED evidence of the accepted root(s) of this root's (leg, view) whose store of `leg` was drawn at `cap` —
    the rung root at stage `cap`, or the base root(s) whose store's own cap is `cap` (a copied-forward row's
    request is an attempt on the rung that produced its text). None pulled → refuse by name (the copied rows' evidence is
    absent, so they cannot bind — never assumed)."""
    reads = reads or ac.RootReads()
    same = [r for r in siblings if (r.leg, r.view) == (root.leg, root.view) and r.attempt_id != root.attempt_id]
    hits = [r for r in same if r.stage == str(cap)]
    if not hits:
        for r in same:
            if r.stage != ac.STAGE_BASE:
                continue
            # a root that holds no store of the leg contributes nothing (an empty list); a MALFORMED store refuses by its
            # own name — never swallowed into "no ancestor"
            if any(st.cap == cap for st in reads.stores(r, leg, petri_behaviours)):
                hits.append(r)
    if not hits:
        refuse(f"{root.label}: copied-forward rows were drawn at cap {cap} but no accepted ancestor root of ({root.leg}, {root.view}) "
               f"at that cap was pulled (siblings: {[r.label for r in same]}) — their attempts cannot be bound")
    return Evidence.merge([load_evidence(r, reads=reads, cache=cache, plan=plan) for r in hits],
                          label=f"{root.leg}/{root.view}/cap{cap} (ancestor of {root.label})")


def bind_rows(store: StoreRead, *, own: Evidence, target_vmodel: str,
              ancestors: Optional[Mapping[int, Evidence]] = None) -> Tuple[int, List[str]]:
    """Every draw of the store bound to ONE accepted attempt and ONE engine-log request. Returns
    `(n_bound, problems)`; `problems` names each failing draw (first 20 kept, the count is exact).
    `own.completion_by_rid` (:func:`completion_ids_by_request`) — a filed successful draw whose attempt has no authoritative
    completion id there is a binding failure (never a bypass: without it the row's completion_id is a claim nothing
    corroborates, and a row naming request A + completion B would pass whenever B is anywhere in the engine log).
    A draw whose `source_cap` is not the store's own cap (a copied-forward static or controls rung row) binds against
    `ancestors[source_cap]` — the accepted root(s) of the rung that produced its text — and that root's vmodel for the arm;
    the evidence is chosen FIRST, and an id-less producer row (the Inspect family) then resolves exactly one completed
    attempt for its (task, sample) from THAT evidence's `response_evidence.jsonl`."""
    ancestors = ancestors or {}
    problems: List[str] = []
    n_bound = 0
    for d in store.draws:
        where = f"{d.task} {d.item_id}/{d.sample_idx}"
        ev, vmodel = own, target_vmodel
        copied_cap = store.copied_cap(d)
        if copied_cap is not None:
            ev = ancestors.get(copied_cap)
            if ev is None:
                problems.append(f"{where}: copied forward from cap {copied_cap} but no ancestor evidence for that cap was resolved")
                continue
            vmodel = ev.vmodel_of(store.arm)
            where += f" (copied forward from cap {copied_cap}; bound on {ev.label})"
        rid, cid = d.request_id, d.completion_id
        if (not rid or not cid) and not store.binds_by_response_evidence:
            problems.append(f"{where}: the stored row carries no request_id / completion_id (request_id {d.request_id!r}, "
                            f"completion_id {d.completion_id!r}) — a row that binds to no attempt")
            continue
        if not rid or not cid:
            # an Inspect-family row (no ids on the row by the writer's design) resolves EXACTLY ONE completed attempt for its
            # (task, sample, vmodel) from the evidence chosen above — the ancestor's for a copied-forward draw
            sample = C.sample_header_value(d.item_id, d.sample_idx)
            cands = ev.completed_by_sample.get((d.task, sample, vmodel), [])
            if len(cands) == 1:
                rid, cid = cands[0]
            elif not cands:
                problems.append(f"{where}: the stored row carries no request_id / completion_id and {ev.label}'s "
                                f"{ac.RESPONSE_EVIDENCE_FILE} records no completed attempt for ({d.task}, {sample}, {vmodel}) — a row "
                                "that binds to no attempt")
                continue
            else:
                problems.append(f"{where}: the stored row carries no request_id / completion_id and {ev.label}'s "
                                f"{ac.RESPONSE_EVIDENCE_FILE} records {len(cands)} completed attempts for ({d.task}, {sample}, {vmodel}) "
                                "— the binding is ambiguous, never resolved by guessing")
                continue
        src = "usage ledger" if ev.attempt_source == "ledger" else f"{ac.RESPONSE_EVIDENCE_FILE} (D106: the ledger is not on this row's path)"
        row = ev.attempts_by_id.get(rid)
        if row is None:
            problems.append(f"{where}: request_id {rid!r} is not in the {src} — a row with no attempt")
            continue
        if not _ok_status(row):
            problems.append(f"{where}: request_id {rid!r} is a REJECTED {'ledger' if ev.attempt_source == 'ledger' else 'response-evidence'} "
                            f"attempt (status {row.get('status')!r}, error_class {row.get('error_class', row.get('error'))!r}) — a filed row "
                            "cannot come from a rejected call")
            continue
        auth_cid = ev.completion_by_rid.get(rid)
        if not auth_cid:
            problems.append(f"{where}: no authoritative completion_id for attempt {rid!r} — neither the usage ledger's "
                            f"completion_id field nor a {ac.RESPONSE_EVIDENCE_FILE} row records which completion this attempt "
                            f"produced, so the row's completion_id {cid!r} is an unverifiable claim")
            continue
        if auth_cid != cid:
            problems.append(f"{where}: the attempt evidence records completion_id {auth_cid!r} for request {rid!r} but "
                            f"the row says {cid!r} — torn binding")
            continue
        if d.text_sha256 is not None:
            # the stored TEXT is held to the response the selected attempt produced (the response record's additive text_sha256):
            # an Inspect-family row bound through response evidence needs it (refused when absent — never assumed); a row-id writer's
            # row is checked when the record carries it
            rsha = ev.response_sha_by_rid.get(rid)
            if rsha is None and store.binds_by_response_evidence:
                problems.append(f"{where}: attempt {rid!r}'s response record carries no {ac.RESPONSE_TEXT_SHA_FIELD} (WS1-D′'s additive "
                                "response hash) — the stored text cannot be bound to the response that produced it; refused, never assumed")
                continue
            if rsha is not None and rsha != d.text_sha256:
                problems.append(f"{where}: attempt {rid!r}'s response text hashes to {rsha} but the stored text hashes to {d.text_sha256} "
                                "— the row's text is not this attempt's response")
                continue
        rec = ev.records_by_id.get(rid)
        if rec is None:
            problems.append(f"{where}: request_id {rid!r} has no request-evidence record")
            continue
        if rec.get("kind") != C.REQUEST_EVIDENCE_KIND_TASK:
            problems.append(f"{where}: request_id {rid!r} is a {rec.get('kind')!r} record, not a task call")
            continue
        if rec.get("task") != d.task or rec.get("item_id") != d.item_id or rec.get("sample_idx") != d.sample_idx:
            problems.append(f"{where}: the request-evidence record labels attempt {rid!r} task {rec.get('task')!r} "
                            f"item {rec.get('item_id')!r}/{rec.get('sample_idx')!r} — not this row's")
            continue
        if rec.get("vmodel") != vmodel:
            problems.append(f"{where}: attempt {rid!r} was sent to vmodel {rec.get('vmodel')!r}, not the arm's "
                            f"{vmodel!r}")
            continue
        hits = [e for e in ev.engine_by_external.get(cid, []) if C.is_engine_id_of(e, cid)]
        if len(hits) != 1:
            problems.append(f"{where}: completion_id {cid!r} matches {len(hits)} engine-log `Received request` "
                            f"line(s) ({hits[:3]}) — exactly one is required")
            continue
        n_bound += 1
    return n_bound, problems


# --------------------------------------------------------------------------- #
# fc3: the controls evidence shape -- per-draw events + the engine line; the MMLU readout at the count level          #
# --------------------------------------------------------------------------- #

MMLU_ENGINE_SHAPE = {"max_tokens": 1, "temperature": 0.0, "seed": 0}   # the letter readout's request (`served_capability_run._mmlu_scoring_payload`)


def _sampler_divergences(params: Mapping[str, Any], expected_row: Any) -> List[str]:
    """The sampler keys of ONE engine line that resolved to something other than the frozen row (only the row's stated keys are compared)."""
    out = []
    for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"):
        want = getattr(expected_row, k, None)
        if want is None:
            continue
        got = params.get(k)
        if got is None or float(got) != float(want):
            out.append(f"{k}={got!r} (frozen row {want!r})")
    return out


def _ancestor_controls_events(anc: Evidence, *, leg: str, arm: str, s: int, reads: ac.RootReads) -> Tuple[Optional[Path], Optional[str]]:
    """The controls event stream of the ANCESTOR store of (leg, arm, s) on the ancestor evidence's root(s) -- a rung store's copied-forward draws
    were drawn there; none / several is a named problem, never a guess."""
    hits: List[Path] = []
    for r in anc.roots:
        for st in reads.stores(r, leg):
            if st.arm == arm and ac.seed_of_store(r, st) == s:
                hits.append(st.store_dir / ac.CONTROLS_EVENTS_FILENAME)
    if len(hits) != 1:
        return None, f"{len(hits)} ancestor store(s) of ({leg}, {arm}, s{s}) on {anc.label}"
    return hits[0], None


def audit_controls_store_events(store: StoreRead, *, s: int, binding: ac.VmodelBinding, own: Evidence, expected_row: Any, expected_source: str,
                                ancestors: Optional[Mapping[int, Evidence]] = None, reads: Optional[ac.RootReads] = None) -> Dict[str, Any]:
    """The verdict of ONE SAMPLED controls store under `RECONCILIATION_CONTRACT_CONTROLS_EVENTS`: the store's own per-draw
    event stream (`controls_events.jsonl`, named by its `gen_stats[control].controls_events`; the LAST event per item) is the attempt record --
    every FILED draw must have an event with no error, the store's cap, the seed rule's seed (== the meta row's), the spooled text's hash, the
    meta's finish_reason, the served cell id -- and each event's `completion_id` binds to EXACTLY ONE engine `Received request` line whose
    seed, max_tokens and sampler are RE-READ from the engine and held to the draw / the frozen row, and whose LoRA name (when the engine states
    one) is the arm's harness cell; every draw's LoRA identity must be one value. The SENT side (`gen_stats[control].decode`) is held to the
    frozen row too. A copied-forward draw (a rung store) binds on the ANCESTOR root's store events and engine. What this contract cannot
    state -- a body digest, a pre-engine attempt record -- is recorded on the verdict, never assumed."""
    from sisteer.eval import controls_events as cev
    reads = reads or ac.RootReads()
    ancestors = ancestors or {}
    problems: List[str] = []
    control = store.leg
    gs_p = store.store_dir / "gen_stats.json"
    gs = ac.read_json(gs_p) if gs_p.is_file() else None
    blk = gs.get(control) if isinstance(gs, dict) else None
    if not isinstance(blk, dict):
        problems.append(f"{gs_p}: no gen_stats block for control {control!r}")
        blk = {}
    ev_p = store.store_dir / ac.CONTROLS_EVENTS_FILENAME
    named = blk.get(cev.GEN_STATS_EVENTS_KEY)
    events: List[dict] = []
    if not ev_p.is_file():
        problems.append(f"{store.store_dir}: no {ac.CONTROLS_EVENTS_FILENAME} beside gen_stats.json -- the driver ran without {cev.CONTROLS_EVENTS_FLAG} (D102 / F-AF), so "
                        "there is no per-draw attempt record to bind on")
    else:
        if not named or Path(str(named)).name != ev_p.name:
            problems.append(f"{gs_p}: gen_stats[{control!r}].{cev.GEN_STATS_EVENTS_KEY} = {named!r} does not name the stream beside it ({ev_p.name}) -- the writer's own record of the stream is missing")
        try:
            events = list(cev.read_controls_events(ev_p))
        except cev.ControlsEventsError as e:
            problems.append(f"{ev_p}: {e}")
    latest = cev.latest_by_item(events) if events else {}
    n_failed = sum(1 for e in events if e.get("error") is not None)
    # the SENT sampler (the store's decode block) against the frozen row
    dec = blk.get("decode") if isinstance(blk.get("decode"), dict) else None
    if dec is None:
        problems.append(f"{gs_p}: gen_stats[{control!r}] carries no decode block -- a sampled controls store records the sampler it sent")
    else:
        sent_div = _sampler_divergences(dec, expected_row)
        if sent_div:
            problems.append(f"sent sampler (gen_stats decode) differs from the frozen row ({expected_source}): {sent_div}")
    n_bound = 0
    bind_problems: List[str] = []
    seen_cids: Dict[str, str] = {}
    lora_names: Counter = Counter()
    engine_div: Counter = Counter()
    anc_events_cache: Dict[int, Tuple[Optional[Dict], Optional[str], Optional[Evidence]]] = {}
    # fc6: the engine identity every bound line must state -- from the standup's adapter / routing records, per evidence (own or ancestor)
    want_lora, lora_source = own.lora_expectation(store.arm)
    lora_expect_of: Dict[str, Tuple[Optional[str], str]] = {own.label: (want_lora, lora_source)}
    for d in store.draws:
        where = f"{d.task} {d.item_id}/{d.sample_idx}"
        ev, lat = own, latest
        copied_cap = store.copied_cap(d)
        if copied_cap is not None:
            if copied_cap not in anc_events_cache:
                anc = ancestors.get(copied_cap)
                if anc is None:
                    anc_events_cache[copied_cap] = (None, f"copied forward from cap {copied_cap} but no ancestor evidence for that cap was resolved", None)
                else:
                    ap, why = _ancestor_controls_events(anc, leg=control, arm=store.arm, s=s, reads=reads)
                    if ap is None or not ap.is_file():
                        anc_events_cache[copied_cap] = (None, why or f"the ancestor store has no {ac.CONTROLS_EVENTS_FILENAME}", anc)
                    else:
                        try:
                            anc_events_cache[copied_cap] = (cev.latest_by_item(cev.read_controls_events(ap)), None, anc)
                        except cev.ControlsEventsError as e:
                            anc_events_cache[copied_cap] = (None, str(e), anc)
            lat, why, anc = anc_events_cache[copied_cap]
            if lat is None:
                bind_problems.append(f"{where}: {why}")
                continue
            ev = anc  # type: ignore[assignment]
            where += f" (copied forward from cap {copied_cap}; bound on {ev.label})"
        e = lat.get((control, d.item_id))
        if e is None:
            bind_problems.append(f"{where}: no controls event for this filed item -- a scored item with no recorded attempt")
            continue
        if e.get("error") is not None:
            bind_problems.append(f"{where}: the item's LAST event records a FAILED attempt ({str(e.get('error'))[:80]!r}) yet the item is spooled -- torn store / stream")
            continue
        if str(e.get("cell_id")) != str(store.arm):
            bind_problems.append(f"{where}: the event names cell {e.get('cell_id')!r}, not this store's arm {store.arm!r}")
            continue
        cap_want = d.source_cap if d.source_cap is not None else store.cap
        if cap_want is not None and int(e.get("cap", -1)) != int(cap_want):
            bind_problems.append(f"{where}: the event's cap {e.get('cap')!r} is not the draw's {cap_want}")
            continue
        seed_want = store.expected_seed(d.task, d.item_id, d.sample_idx) if store.expected_seed is not None else None
        if seed_want is not None and e.get("sampling_seed") != seed_want:
            bind_problems.append(f"{where}: the event's sampling_seed {e.get('sampling_seed')!r} is not the seed rule's {seed_want} ({store.seed_rule_note})")
            continue
        if d.text_sha256 is not None and e.get("text_sha256") != d.text_sha256:
            bind_problems.append(f"{where}: the event's text_sha256 {str(e.get('text_sha256'))[:12]} is not the spooled text's {d.text_sha256[:12]} -- the stored text is not this attempt's response")
            continue
        if d.finish_reason is not None and e.get("finish_reason") != d.finish_reason:
            bind_problems.append(f"{where}: the event's finish_reason {e.get('finish_reason')!r} is not the meta row's {d.finish_reason!r}")
            continue
        cid = e.get("completion_id")
        if not cid:
            bind_problems.append(f"{where}: the event records no completion_id -- the attempt cannot be bound to the engine")
            continue
        cid = str(cid)
        prior = seen_cids.get(cid)
        if prior is not None:
            bind_problems.append(f"{where}: completion_id {cid!r} is also {prior}'s -- one completion bound to two draws")
            continue
        seen_cids[cid] = where
        hits = [x for x in ev.engine_by_external.get(cid, []) if C.is_engine_id_of(x, cid)]
        if len(hits) != 1:
            bind_problems.append(f"{where}: completion_id {cid!r} matches {len(hits)} engine-log `Received request` line(s) ({hits[:3]}) — exactly one is required")
            continue
        params = ev.engine_params_by_id.get(hits[0])
        if params is None:
            bind_problems.append(f"{where}: engine line {hits[0]!r} carries no readable SamplingParams -- the engine's seed / cap / sampler cannot be re-read")
            continue
        if params.get("seed") != e.get("sampling_seed"):
            bind_problems.append(f"{where}: the ENGINE received seed {params.get('seed')!r} for completion {cid!r}, the client sent {e.get('sampling_seed')!r}")
            continue
        if params.get("max_tokens") != int(e.get("cap")):
            bind_problems.append(f"{where}: the ENGINE received max_tokens {params.get('max_tokens')!r} for completion {cid!r}, the client's cap is {e.get('cap')!r}")
            continue
        div = _sampler_divergences(params, expected_row)
        if div:
            engine_div[tuple(div)] += 1
            bind_problems.append(f"{where}: the ENGINE resolved {div} for completion {cid!r} (frozen row {expected_source})")
            continue
        if ev.label not in lora_expect_of:
            lora_expect_of[ev.label] = ev.lora_expectation(store.arm)
        want, _src = lora_expect_of[ev.label]
        if not params.get("lora_request_stated"):
            bind_problems.append(f"{where}: engine line {hits[0]!r} states no `lora_request` clause -- the identity the engine served completion {cid!r} under cannot be read")
            continue
        lname = params.get("lora_name")
        lora_names[str(lname)] += 1
        if want is None and lname is not None:
            bind_problems.append(f"{where}: the ENGINE served completion {cid!r} under LoRA {lname!r}, but the standup records arm {store.arm!r} (cell {binding.harness_id!r}) "
                                 f"as served on the base weights ({_src})")
            continue
        if want is not None and lname is None:
            bind_problems.append(f"{where}: the ENGINE served completion {cid!r} on the BASE weights (lora_request None), but the standup attests arm {store.arm!r} "
                                 f"(cell {binding.harness_id!r}) as LoRA-served under {want!r} ({_src}) -- a base-model draw under a fitted arm's identity")
            continue
        if want is not None and str(lname) != str(want):
            bind_problems.append(f"{where}: the ENGINE served completion {cid!r} under LoRA {lname!r}, not the arm's attested adapter {want!r} ({binding.vmodel}; {_src})")
            continue
        n_bound += 1
    if len(lora_names) > 1:
        problems.append(f"the store's bound draws were served under {len(lora_names)} LoRA identities {dict(lora_names)} -- one arm is one identity")
    if bind_problems:
        problems.append(f"row binding: {len(bind_problems)} of {len(store.draws)} draw(s) do not bind (first: {bind_problems[0]})")
    if not store.complete:
        problems.append(f"store incomplete: {store.completeness_note}")
    verdict = VERDICT_AGREES if not problems else VERDICT_REFUSED
    return {"leg": store.leg, "arm": store.arm, "s": s, "verdict": verdict, "problem": None if not problems else " | ".join(problems), "problems": problems,
            "checks": {"reconciliation": None if not problems else problems[0],
                       "controls_events": {"path": str(ev_p), "n_events": len(events), "n_items_latest": len(latest), "n_failed_attempts": n_failed,
                                           "named_by_gen_stats": bool(named), "engine_lora_names": dict(lora_names),
                                           "engine_identity_expected": {"lora_name": want_lora, "source": lora_source},
                                           "n_engine_sampler_divergent_draws": sum(engine_div.values())},
                       "row_binding": {"n_draws": len(store.draws), "n_bound": n_bound, "n_problems": len(bind_problems), "problems": bind_problems[:20],
                                       "n_copied_forward": len(store.copied_draws), "ancestor_caps": sorted({d.source_cap for d in store.copied_draws}),
                                       "ancestor_attempt_ids": {str(c): a.attempt_ids for c, a in sorted((ancestors or {}).items())}}},
            "n_records_task": 0, "n_ledger_rows_partition": None, "n_attempt_rows_partition": len(latest), "attempt_source": ac.ATTEMPT_SOURCE_CONTROLS_EVENTS,
            "n_filed": len(store.filed), "n_filed_here": len(store.filed_here), "n_draws": len(store.draws), "n_draws_bound": n_bound,
            "n_engine_received": len(own.engine_ids), "seed_rule": store.seed_rule_note, "seed_base": store.seed_base, "expected_sampling": expected_row._asdict(),
            **{k: getattr(expected_row, k, None) for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")},
            "expected_sampling_source": expected_source, "target_vmodel": binding.vmodel, "thinking_expect": _thinking_label(None), "tasks": list(store.tasks),
            "store_dir": str(store.store_dir), "store_complete": store.complete, "store_completeness_note": store.completeness_note, "n_rows": store.n_rows, "store_notes": list(store.notes),
            "_contract": ac.RECONCILIATION_CONTRACT_CONTROLS_EVENTS}


def audit_mmlu_store_count(store: StoreRead, *, s: int, binding: ac.VmodelBinding, own: Evidence, expected_row: Any, expected_source: str,
                           n_sibling_stores: int) -> Dict[str, Any]:
    """The verdict of ONE MMLU letter-readout store under `RECONCILIATION_CONTRACT_MMLU_COUNT`: NO per-draw binding exists (the parity
    emitter records no request id per item; the readout fires no controls event), so identity is held at the COUNT level -- the engine log must
    carry at least `n_filed x n_sibling_stores` MMLU-shaped requests (`MMLU_ENGINE_SHAPE`: max_tokens 1, temperature 0.0, seed 0) under this arm's
    weights (the LoRA name RE-READ from the engine == the harness cell, or the base weights when the arm is served without a LoRA), every one
    of them resolving the frozen row's stated keys; `n_sibling_stores` = the stores on this root that share these weights (the Arditi root's
    three seed namespaces of one cell), because their requests cannot be told apart. Weaker than a per-draw binding BY DESIGN and said so on
    the verdict."""
    problems: List[str] = []
    n_items = len(store.filed)
    if n_items == 0:
        problems.append("the store files no MMLU item")
    shaped = {eid: p for eid, p in own.engine_params_by_id.items()
              if p.get("max_tokens") == MMLU_ENGINE_SHAPE["max_tokens"] and p.get("temperature") is not None and float(p["temperature"]) == MMLU_ENGINE_SHAPE["temperature"]
              and p.get("seed") == MMLU_ENGINE_SHAPE["seed"]}
    by_lora: Counter = Counter(str(p.get("lora_name")) for p in shaped.values())
    # fc6: the identity the standup ATTESTS for this arm decides which lines count -- a fitted arm's LoRA name, or the base weights only for an
    # arm the records say is served without an adapter; base-weight requests are NEVER substituted for a fitted arm's
    want_lora, lora_source = own.lora_expectation(store.arm)
    under_arm = {eid: p for eid, p in shaped.items() if p.get("lora_request_stated") and p.get("lora_name") == want_lora}
    weights = f"LoRA {want_lora!r}" if want_lora is not None else "the base weights (the standup attests no adapter for this arm)"
    n_engine = len(under_arm)
    if want_lora is not None and not under_arm:
        problems.append(f"no MMLU-shaped engine request names the arm's attested LoRA {want_lora!r} ({lora_source}); the {by_lora.get('None', 0)} base-weight "
                        "request(s) are NOT substituted -- a base-model readout is never certified under a fitted arm's identity (fc6)")
    div = Counter()
    for p in under_arm.values():
        for d in _sampler_divergences(p, expected_row):
            div[d] += 1
    if div:
        problems.append(f"{sum(div.values())} MMLU-shaped engine request(s) under {weights} resolved off the frozen row ({expected_source}): {dict(div)}")
    want = n_items * max(1, int(n_sibling_stores))
    if n_engine < want:
        problems.append(f"the engine log carries {n_engine} MMLU-shaped request(s) under {weights} for {n_items} filed item(s) x {n_sibling_stores} store(s) sharing these weights "
                        f"(>= {want} required) -- the readout did not reach the engine for every item")
    if not store.complete:
        problems.append(f"store incomplete: {store.completeness_note}")
    verdict = VERDICT_AGREES if not problems else VERDICT_REFUSED
    return {"leg": store.leg, "arm": store.arm, "s": s, "verdict": verdict, "problem": None if not problems else " | ".join(problems), "problems": problems,
            "checks": {"reconciliation": None if not problems else problems[0],
                       "engine_count": {"n_filed_items": n_items, "n_sibling_stores_same_weights": n_sibling_stores, "n_required": want, "n_engine_mmlu_shaped_under_arm": n_engine,
                                        "weights": weights, "shape": dict(MMLU_ENGINE_SHAPE), "n_engine_mmlu_shaped_by_lora": dict(by_lora),
                                        "engine_identity_expected": {"lora_name": want_lora, "source": lora_source}},
                       "row_binding": {"n_draws": len(store.draws), "n_bound": 0, "n_problems": 0, "problems": [],
                                       "note": "no per-draw binding exists for the MMLU readout (no request id on the row, no controls event) -- count-level identity only (D133)"}},
            "n_records_task": 0, "n_ledger_rows_partition": None, "n_attempt_rows_partition": n_engine, "attempt_source": ac.ATTEMPT_SOURCE_ENGINE_COUNT,
            "n_filed": n_items, "n_filed_here": len(store.filed_here), "n_draws": len(store.draws), "n_draws_bound": 0, "n_engine_received": len(own.engine_ids),
            "seed_rule": store.seed_rule_note, "seed_base": store.seed_base, "expected_sampling": expected_row._asdict(),
            **{k: getattr(expected_row, k, None) for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")},
            "expected_sampling_source": expected_source, "target_vmodel": binding.vmodel, "thinking_expect": _thinking_label(None), "tasks": list(store.tasks),
            "store_dir": str(store.store_dir), "store_complete": store.complete, "store_completeness_note": store.completeness_note, "n_rows": store.n_rows, "store_notes": list(store.notes),
            "_contract": ac.RECONCILIATION_CONTRACT_MMLU_COUNT}


def qualification_of(contract: str, own: Evidence) -> Dict[str, Any]:
    """The qualification every verdict carries (`seedrep_launch_plan.AUDIT_QUALIFICATION_KEYS`), per reconciliation contract: the
    evidence files the verdict rests on, whether a body digest was checked, and what the contract cannot state."""
    if contract == ac.RECONCILIATION_CONTRACT_LEDGER:
        return {"reconciliation_contract": contract, "evidence_basis": [ac.EVIDENCE_FILE, ac.LEDGER_FILE, ac.RESPONSE_EVIDENCE_FILE, "engine log (Received request)"],
                "body_digest_check": "usage ledger body_sha256 == client record body_sha256", "independence_note": None}
    if contract == ac.RECONCILIATION_CONTRACT_LEDGERLESS:
        return {"reconciliation_contract": contract, "evidence_basis": [ac.EVIDENCE_FILE, ac.RESPONSE_EVIDENCE_FILE, "engine log (Received request)"],
                "body_digest_check": ("not_available: the ledger is not on this row's path, so the client's body digest has no independent counterpart "
                                      "(D106 — a revised, qualified contract, not equivalent evidence)"),
                "independence_note": ("D106: the engine log is the only server-side record; rejected / pre-engine attempts are corroborated by the client's "
                                      "response hook alone, never independently")}
    if contract == ac.RECONCILIATION_CONTRACT_CONTROLS_EVENTS:
        return {"reconciliation_contract": contract,
                "evidence_basis": [ac.CONTROLS_EVENTS_FILENAME, "gen_stats.json (decode, controls_events)", "private/gens_<control>.jsonl + .meta.jsonl",
                                   f"{ac.ENDPOINT_FILE} (arm_binding)", "engine log (Received request: id, seed, max_tokens, sampler, lora_request)"],
                "body_digest_check": ("not_available: the controls driver wires no request-evidence client (no request_evidence.jsonl / response_evidence.jsonl), so the "
                                      "wire body has no client-side digest (D133 — a revised, qualified contract, not equivalent evidence)"),
                "independence_note": ("D133: every draw binds through the client's completion_id to exactly one engine `Received request` line whose seed, max_tokens, "
                                      "sampler and LoRA name are re-read from the engine; failed attempts are known from the client's own event stream alone; no body digest")}
    if contract == ac.RECONCILIATION_CONTRACT_MMLU_COUNT:
        return {"reconciliation_contract": contract,
                "evidence_basis": ["rows.jsonl", f"{ac.ENDPOINT_FILE} (arm_binding)", "engine log (Received request: max_tokens 1, temperature 0.0, seed 0, lora_request)"],
                "body_digest_check": ("not_available: the MMLU letter readout records no request id per item and fires no controls event (D133 — a revised, qualified "
                                      "contract, not equivalent evidence)"),
                "independence_note": ("D133: NO per-draw binding -- identity is held at the COUNT level only (at least one MMLU-shaped engine request per filed item under "
                                      "this arm's weights); the readout is a teacher-forced greedy argmax the sampler cannot move (LADDER_RULES.md)")}
    refuse(f"{own.label}: reconciliation contract {contract!r} has no qualification (known: {list(ac.RECONCILIATION_CONTRACTS)})")


# --------------------------------------------------------------------------- #
# One store's verdict                                                           #
# --------------------------------------------------------------------------- #

record_extra_s = ac.record_extra_s  # the record-side seed attribution lives beside the row-side ids


def _sample_item(sample: Any) -> Optional[str]:
    """The coverage clause's item: the sample header's FIRST slash component (`contracts.request_evidence_reconciliation`)."""
    return str(sample).split("/", 1)[0] if isinstance(sample, str) and sample else None


def store_partition(records: Sequence[Mapping[str, Any]], ledger_rows: Sequence[Mapping[str, Any]], *,
                    tasks: Sequence[str], target_vmodel: str, known_tasks: Sequence[str], filed_items: Sequence[str] = (),
                    extra_s: Optional[int] = None) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """The records and ledger rows a store is judged against: the arm's vmodel AND (the store's own row task(s) OR a
    task label that is NO leg's canonical task on this engine AND names one of the store's FILED items). The second
    clause is what makes a mislabelled header (`gsm8k/none`, `inspect/xstest`) fail the store's COVERAGE clause — "filed
    item(s) have NO recorded attempt" — instead of vanishing into an empty partition: a label nobody's store files under
    belongs to the store whose arm AND item it names, and to no sibling store on the
    same engine. Probe-labelled calls have their own partition.

    `extra_s` (the ARDITI engine): three seed bases draw the SAME task on the same vmodel, so a record is
    admitted only when its `extra.s` is the store's seed; the ledger rows admitted are those whose request id names an
    admitted record, plus every ledger row that names NO record at all (an orphan refuses in every partition, never
    hides in none)."""
    items = set(filed_items)

    def admit(task: Any, vmodel: Any, sample: Any) -> bool:
        if vmodel != target_vmodel or C.is_probe_task(task):
            return False
        if task in tasks:
            return True
        return task not in known_tasks and _sample_item(sample) in items
    recs = [r for r in records if admit(r.get("task"), r.get("vmodel"), r.get("sample"))]
    all_ids = {str(r.get("request_id")) for r in records if r.get("request_id")}
    admitted_ids = {str(r.get("request_id")) for r in recs if r.get("request_id")}

    def admit_attempt(r: Mapping[str, Any]) -> bool:
        # an attempt row carrying its own labels is admitted by them (a ledger row always does); a row WITHOUT a vmodel — the
        # response hook's record of a FAILED attempt has no body to read `model` off (a retry's 503 is an attempt) — is
        # admitted through the REQUEST RECORD it names, and one naming no record at all is admitted to every partition so it
        # refuses by name there rather than hiding in none
        if r.get("vmodel") is not None:
            return admit(r.get("task"), r.get("vmodel"), r.get("sample"))
        rid = str(r.get(C.LEDGER_REQUEST_ID_FIELD))
        return rid in admitted_ids or rid not in all_ids
    if extra_s is None:
        rows = [r for r in ledger_rows if admit_attempt(r)]
        return recs, rows
    recs = [r for r in recs if record_extra_s(r) == extra_s]
    admitted_ids = {str(r.get("request_id")) for r in recs}
    rows = [r for r in ledger_rows if admit_attempt(r)
            and (str(r.get(C.LEDGER_REQUEST_ID_FIELD)) in admitted_ids or str(r.get(C.LEDGER_REQUEST_ID_FIELD)) not in all_ids)]
    return recs, rows


def audit_store(store: StoreRead, *, s: int, target_vmodel: str, records: Sequence[Mapping[str, Any]],
                ledger_rows: Sequence[Mapping[str, Any]], own: Evidence, expected_row: "C.SamplingRow",
                expected_source: str, thinking_expect: Optional[bool], known_tasks: Sequence[str] = (),
                partition_by_extra_s: bool = False, ancestors: Optional[Mapping[int, Evidence]] = None,
                attempt_source: str = "ledger") -> Dict[str, Any]:
    """The verdict of ONE store — pure over its inputs (a test drives it on synthetic
    stores; the CLI on pulled roots). `known_tasks` = every canonical row task of the legs served on this engine
    (default: the store's own); `partition_by_extra_s` = the Arditi engine's three-bases-one-task case, see
    :func:`store_partition`; `ancestors` = the evidence of the rungs a static rung store copied rows from, by cap.
    The reconciliation's `filed` is `store.filed_here` — the items with a draw on THIS root's evidence (a copied-forward
    item's request is on its ancestor's); the store's completeness still covers every declared item."""
    filed_here = store.filed_here
    recs, rows = store_partition(records, ledger_rows, tasks=store.tasks, target_vmodel=target_vmodel,
                                 known_tasks=list(known_tasks) or list(store.tasks), filed_items=[i for _, i in filed_here],
                                 extra_s=s if partition_by_extra_s else None)
    checks: Dict[str, Any] = {}
    problems: List[str] = []
    if store.vmodel is not None and store.vmodel != target_vmodel:
        problems.append(f"the store records virtual_model {store.vmodel!r} but the standup serves arm {store.arm!r} as "
                        f"{target_vmodel!r}")
    if store.expected_seed is None:
        problems.append(f"seed rule unavailable: {store.seed_rule_note}")
        checks["reconciliation"] = "not run (seed rule unavailable)"
    else:
        why = C.request_evidence_reconciliation(recs, rows, filed=filed_here, expected_enable_thinking=thinking_expect,
                                                expected_sampling=expected_row, expected_seed=store.expected_seed,
                                                target_vmodel=target_vmodel, attempt_source=attempt_source)
        checks["reconciliation"] = why
        if why is not None:
            problems.append(f"reconciliation: {why}")
    n_bound, bind_problems = bind_rows(store, own=own, target_vmodel=target_vmodel, ancestors=ancestors)
    copied = store.copied_draws
    checks["row_binding"] = {"n_draws": len(store.draws), "n_bound": n_bound, "n_problems": len(bind_problems),
                             "problems": bind_problems[:20],
                             "n_authoritative_completion_ids": sum(1 for d in store.draws if d.request_id in own.completion_by_rid),
                             "n_copied_forward": len(copied), "ancestor_caps": sorted({d.source_cap for d in copied}),
                             "ancestor_attempt_ids": {str(c): e.attempt_ids for c, e in sorted((ancestors or {}).items())}}
    if bind_problems:
        problems.append(f"row binding: {len(bind_problems)} of {len(store.draws)} draw(s) do not bind (first: {bind_problems[0]})")
    if not store.complete:
        problems.append(f"store incomplete: {store.completeness_note}")
    verdict = VERDICT_AGREES if not problems else VERDICT_REFUSED
    return {"leg": store.leg, "arm": store.arm, "s": s, "verdict": verdict,
            "problem": None if not problems else " | ".join(problems),
            "problems": problems, "checks": checks,
            "n_records_task": sum(1 for r in recs if r.get("kind") == C.REQUEST_EVIDENCE_KIND_TASK),
            # on a ledger-less row `n_ledger_rows_partition` is None BY NAME (never 0 read as "an empty ledger"); the attempt
            # inventory's count is `n_attempt_rows_partition`, its source `attempt_source`
            "n_ledger_rows_partition": len(rows) if attempt_source == "ledger" else None, "n_attempt_rows_partition": len(rows),
            "attempt_source": attempt_source, "n_filed": len(store.filed), "n_filed_here": len(filed_here), "n_draws": len(store.draws),
            "n_draws_bound": n_bound, "n_engine_received": len(own.engine_ids),
            "seed_rule": store.seed_rule_note, "seed_base": store.seed_base, "expected_sampling": expected_row._asdict(),
            **{k: getattr(expected_row, k) for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")},
            "expected_sampling_source": expected_source, "target_vmodel": target_vmodel,
            "thinking_expect": _thinking_label(thinking_expect), "tasks": list(store.tasks), "store_dir": str(store.store_dir),
            "store_complete": store.complete, "store_completeness_note": store.completeness_note, "n_rows": store.n_rows, "store_notes": list(store.notes)}


# --------------------------------------------------------------------------- #
# One attempt root                                                              #
# --------------------------------------------------------------------------- #

def _thinking_by_task(thinking_expect: Mapping[str, Optional[bool]], petri_behaviours: Sequence[str]) -> Dict[str, Optional[bool]]:
    out: Dict[str, Optional[bool]] = {}
    for leg in ac.LEGS:
        for t in ac.tasks_of_leg(leg, petri_behaviours):
            out[t] = thinking_expect[leg]
    return out


def audit_root(root: AttemptRoot, *, out_dir: Path, legs: Sequence[str], policy_file: Optional[Path],
               thinking_expect: Mapping[str, Optional[bool]], petri_behaviours: Sequence[str], model_key: str,
               client_status: str, siblings: Sequence[AttemptRoot] = (), reads: Optional[ac.RootReads] = None,
               evidence_cache: Optional[Dict[str, Evidence]] = None,
               ancestor_cache: Optional[Dict[Tuple[str, str, str, int], Evidence]] = None,
               plan: Optional[ac.LaunchPlan] = None, audited_at: Optional[str] = None,
               resolved_sampler: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Every store verdict of ONE accepted attempt root. `siblings` = every pulled root (the CLI passes them all): a static
    RUNG store's copied-forward rows bind on the ancestor root(s) of its (leg, view) at the cap that produced their text
    (:func:`ancestor_evidence`); audited alone, such a store refuses by name. `reads` / `evidence_cache` / `ancestor_cache`
    (per run — the CLI passes one of each) load every root's pipeline and every (container leg, view, leg, cap) ancestor ONCE
    across the roots audited."""
    reads = reads or ac.RootReads()
    own = load_evidence(root, reads=reads, cache=evidence_cache, plan=plan)
    # the ATTEMPT inventory the identity clause runs over: the ledger's rows, or (`not_on_path`) the response hook's
    records, ledger_rows, log_p, dup_engine = own.records, own.attempt_rows, own.engine_logs[0], own.dup_engine_ids
    table_legs = [l for l in legs if ac.CONTAINER_LEG_OF[l] == root.leg]
    if not table_legs:
        return []
    rs_p = ac.find_one(root.path, RESOLVED_SAMPLER_FILE, required=False)
    rs_verdict = "ABSENT"
    if rs_p is not None:
        rs = ac.read_json(rs_p)
        rs_verdict = str(rs.get("verdict")) if isinstance(rs, dict) and rs.get("verdict") is not None else "UNREADABLE"
    # fc11d: a RE-AUDIT of a carried / prior root may hand the audit a sampler verdict RE-DERIVED by the auditing tree (`--resolved-sampler`, written by
    # that tree's recorder into the re-audit's evidence dir -- the historical root's record, the F-AR-era UNCOMPARED, is never overwritten). The
    # verdict every store verdict carries is then the fresh one, with its provenance (`resolved_sampler_source`: path, sha256 of the bytes read,
    # the re-deriving tree) and the historical verdict kept alongside (`historical_resolved_sampler_verdict`), so a reader can hold the audit to the
    # exact bytes the recorder wrote and never mistake the stored verdict for the fresh one.
    historical_verdict, historical_path = rs_verdict, (str(rs_p) if rs_p else None)
    rs_source: Optional[Dict[str, Any]] = None
    if resolved_sampler is not None:
        if audited_at is None:
            ac.refuse("--resolved-sampler is another tree's re-derivation of the sampler verdict: it needs --audited-at <that tree> (the tree whose recorder wrote it)")
        if not Path(resolved_sampler).is_file():
            ac.refuse(f"--resolved-sampler {resolved_sampler}: not a file")
        fresh = ac.read_json(Path(resolved_sampler))
        if not isinstance(fresh, dict) or fresh.get("verdict") is None:
            ac.refuse(f"--resolved-sampler {resolved_sampler}: not the recorder's record (no verdict)")
        rs_verdict = str(fresh["verdict"])
        rs_p = Path(resolved_sampler)
        rs_source = {"path": str(resolved_sampler), "sha256": ac.sha256_file(Path(resolved_sampler)), "rederived_by": audited_at}
    engine_wide = engine_wide_problems(records, ledger_rows,
                                       thinking_expect_of_task=_thinking_by_task(thinking_expect, petri_behaviours),
                                       default_thinking=thinking_expect.get(table_legs[0]), attempt_source=own.attempt_source)
    ew_problems = {k: v for k, v in engine_wide.items() if v is not None}
    verdicts: List[Dict[str, Any]] = []
    # every canonical task of EVERY table leg served on this engine (not only the legs requested): a label that is some
    # other leg's store task is that leg's business, never an "unknown label" judged against this store
    known_tasks = [t for leg in ac.table_legs_of(root.leg) for t in ac.tasks_of_leg(leg, petri_behaviours)]
    arditi = root.view == ac.ARDITI_VIEW
    if ancestor_cache is None:
        ancestor_cache = {}
    if arditi:
        # the ONE Arditi engine serves three seed bases on one task: every task record must say which (extra.s)
        unstamped = [r for r in records if r.get("kind") == C.REQUEST_EVIDENCE_KIND_TASK and record_extra_s(r) is None]
        if unstamped:
            refuse(f"{root.label}: {len(unstamped)} task record(s) on the Arditi engine carry no extra.s (first: task "
                   f"{unstamped[0].get('task')!r} sample {unstamped[0].get('sample')!r}) — three seed bases share one task on this "
                   "engine and a record without its base cannot be attributed to a store")
    events_mode = own.attempt_source == ac.ATTEMPT_SOURCE_CONTROLS_EVENTS
    for leg in table_legs:
        expected_row, expected_source = ac.expected_sampling(leg, policy_file=policy_file, model_key=model_key)
        stores = ac.discover_stores(root, leg, petri_behaviours=petri_behaviours)
        # mmlu-count: the MMLU stores on this root that share one set of weights (the Arditi root: three seed namespaces of one cell) --
        # their engine requests cannot be told apart, so the count is held for all of them together
        siblings_by_weights: Counter = Counter(str(own.lora_expectation(st.arm)[0]) for st in stores) if (events_mode and leg == ac.MMLU_LEG) else Counter()
        for store in stores:
            s = ac.seed_of_store(root, store)
            # F-AN: the store's arm (the DIR name the driver wrote — the TrainTest cell id for static / BoN / multi-turn, the harness id for
            # misalign / CellStore cells) resolves through the standup's records in either spelling; the binding rides the verdict
            binding = own.binding_of(store.arm)
            vmodel = binding.vmodel
            ancestors: Dict[int, Evidence] = {}
            for cap in sorted({int(store.copied_cap(d) or 0) for d in store.copied_draws}):
                akey = (root.leg, root.view, leg, cap)
                if akey not in ancestor_cache:
                    ancestor_cache[akey] = ancestor_evidence(root, siblings, leg=leg, cap=cap, petri_behaviours=petri_behaviours,
                                                             reads=reads, cache=evidence_cache, plan=plan)
                ancestors[cap] = ancestor_cache[akey]
            if events_mode and leg in ac.CONTROLS:
                v = audit_controls_store_events(store, s=s, binding=binding, own=own, expected_row=expected_row, expected_source=expected_source,
                                                ancestors=ancestors, reads=reads)
            elif events_mode and leg == ac.MMLU_LEG:
                v = audit_mmlu_store_count(store, s=s, binding=binding, own=own, expected_row=expected_row, expected_source=expected_source,
                                           n_sibling_stores=siblings_by_weights.get(str(own.lora_expectation(store.arm)[0]), 1))
            elif events_mode:
                refuse(f"{root.label}: table leg {leg!r} on a controls root reconciled on controls events -- no D133 contract covers it")
            else:
                v = audit_store(store, s=s, target_vmodel=vmodel, records=records, ledger_rows=ledger_rows, own=own,
                                expected_row=expected_row, expected_source=expected_source, thinking_expect=thinking_expect[leg],
                                known_tasks=known_tasks, partition_by_extra_s=arditi, ancestors=ancestors, attempt_source=own.attempt_source)
            contract = v.pop("_contract", own.contract)
            qual = qualification_of(contract, own)
            # fj7 (re-audit): `W` is the tree the audit RAN FROM -- the root's own unless `audited_at` names a later tree re-auditing a
            # CARRIED root (its verdicts are then that tree's readers' work; the store's own tree stays under `store_W`, never rewritten)
            v.update(view=root.view, stage=root.stage, shard=root.shard, attempt_id=root.attempt_id, W=(audited_at or root.W), store_W=root.W,
                     audited_at=audited_at,
                     # F-AN: how the store's arm was bound to its vmodel (spelling, harness id, the records read)
                     arm_binding=binding.as_dict(),
                     # the qualification — carried on every verdict, the summary CSV, and whatever reads them (the certificate, the analysis):
                     ledger=own.ledger, ledger_source=own.ledger_source, **qual,
                     engine_log=str(log_p), engine_wide_identity=engine_wide, resolved_sampler_verdict=rs_verdict,
                     resolved_sampler_path=(str(rs_p) if rs_p else None), client_task_names=client_status,
                     # fc11d: the fresh verdict's provenance (None when the root's own record was read) and the historical record's verdict / path
                     resolved_sampler_source=rs_source, historical_resolved_sampler_verdict=historical_verdict, historical_resolved_sampler_path=historical_path,
                     standup_record=own.standups[0].record_source if own.standups else None,
                     standup_provenance=dict(own.standups[0].provenance) if own.standups else None,
                     # the files the rows came from (reader provenance `_source`): a multi-child root's per-child evidence and ledgers
                     evidence_sources=sorted({str(r.get(ac.ROW_SOURCE_FIELD)) for r in records}),
                     ledger_sources=sorted({str(r.get(ac.ROW_SOURCE_FIELD)) for r in ledger_rows}))
            extra: List[str] = []
            if ew_problems:
                extra.append("engine-wide identity: " + "; ".join(f"[{k}] {w}" for k, w in list(ew_problems.items())[:3]))
            if dup_engine:
                extra.append(f"engine log: {len(dup_engine)} duplicate `Received request` id(s) (first: {dup_engine[:3]})")
            if extra:
                v["problems"] = list(v["problems"]) + extra
                v["problem"] = " | ".join(v["problems"])
                v["verdict"] = VERDICT_REFUSED
            path = out_dir / "request_log" / leg / verdict_filename(store.arm, s, stage=root.stage, shard=root.shard)
            if path.exists():
                refuse(f"{path}: a verdict for ({leg}, {store.arm}, s{s}, {root.stage}, {root.shard}) already exists — two stores "
                       "claim one arm-run")
            # F-AO: the verdict names its own file (rel under the audit's out dir — the store's spelling, preserved), so the record and
            # the index the run writes (`write_verdict_index`) agree on where it lives
            v["verdict_file"] = f"request_log/{leg}/{path.name}"
            ac.write_json(path, v)
            v["path"] = str(path)
            verdicts.append(v)
    return verdicts


def index_entry(v: Mapping[str, Any]) -> Dict[str, Any]:
    """ONE verdict's index row (F-AO): where the audit filed it and the arm binding it filed it under — the store's arm (the filename's
    spelling) beside the harness id / TrainTest id / vmodel `audit_common.standup_vmodel_of` resolved (`arm_binding`), plus the file's sha."""
    ab = v.get("arm_binding") if isinstance(v.get("arm_binding"), Mapping) else {}
    p = Path(str(v["path"]))
    return {"rel": str(v["verdict_file"]), "leg": v.get("leg"), "arm": v.get("arm"), "s": v.get("s"), "stage": v.get("stage"), "shard": v.get("shard"),
            "attempt_id": v.get("attempt_id"), "verdict": v.get("verdict"), "harness_id": ab.get("harness_id"), "traintest_id": ab.get("traintest_id"),
            "vmodel": ab.get("vmodel"), "spelling": ab.get("spelling"), "sha256": ac.sha256_file(p)}


def write_verdict_index(out_dir: Path, verdicts: Sequence[Mapping[str, Any]]) -> Path:
    """`<out>/request_log_index.json` — every verdict this run filed (F-AO), the writer's own record of its outputs."""
    p = Path(out_dir) / INDEX_NAME
    ac.write_json(p, {"schema": INDEX_SCHEMA, "writer": "scripts/sdfbase/seedrep/request_evidence_audit.py", "n_verdicts": len(verdicts),
                      "verdicts": [index_entry(v) for v in verdicts]})
    return p


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def _parse_thinking(specs: Sequence[str]) -> Dict[str, Optional[bool]]:
    out = dict(ac.THINKING_EXPECT_DEFAULT)
    for spec in specs:
        if "=" not in spec:
            refuse(f"--thinking-expect {spec!r}: expected <leg>=true|false|none")
        leg, val = spec.split("=", 1)
        if leg not in ac.LEGS:
            refuse(f"--thinking-expect: unknown leg {leg!r}")
        val = val.strip().lower()
        if val not in ("true", "false", "none"):
            refuse(f"--thinking-expect {spec!r}: value must be true|false|none")
        out[leg] = None if val == "none" else (val == "true")
    return out


def client_task_name_status() -> str:
    """`agree` when `request_evidence_client.store_task_name` is importable and agrees with
    :func:`audit_common.canonical_task` on every leg; `absent` when the module is not on the tree; a
    disagreement REFUSES (the header side and the filed-pairs side would drift)."""
    try:
        from sisteer.eval import request_evidence_client as rec_client  # type: ignore[import-not-found]
    except ImportError:
        return "absent"
    bad = ac.check_client_task_names(rec_client)
    if bad:
        refuse("request_evidence_client.store_task_name disagrees with audit_common.canonical_task: " + "; ".join(bad[:5]))
    return "agree"


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stores", type=Path, help="RS/stores — every accepted attempt root under it")
    ap.add_argument("--roots", type=Path, nargs="*", default=[], help="explicit attempt roots (instead of --stores)")
    ap.add_argument("--out", type=Path, required=True, help="RS/audits")
    ap.add_argument("--policy-file", type=Path, default=None,
                    help="configs/registry/decode_policy.json — resolve the BoN / multi-turn / Petri vendor rows by exact ref")
    ap.add_argument("--legs", default=",".join(ac.LEGS), help="comma-separated table legs (default: all 15)")
    ap.add_argument("--petri-behaviours", default="", help="comma-separated Petri behaviours (the petri/<b> row tasks)")
    ap.add_argument("--thinking-expect", action="append", default=[], help="<leg>=true|false|none (default: gsm8k=true, others none)")
    ap.add_argument("--model-key", default=ac.MODEL_KEY)
    ap.add_argument("--plan", type=Path, default=None,
                    help="RS/launch_plan.json — the BOUND plan whose children declare each row's usage ledger (D106); without it the plan "
                         "builders' leg contract decides, and the verdict says so")
    ap.add_argument("--audited-at", default=None,
                    help="fj7 (D112): the 40-hex tree this audit runs FROM when it re-audits a CARRIED root of an earlier tree -- stamped as the "
                         "verdicts' W (the root's own tree stays under store_W); never given for an attempt audited on its own tree")
    ap.add_argument("--resolved-sampler", type=Path, default=None,
                    help="fc11d: the sampler verdict RE-DERIVED by the --audited-at tree's recorder over the root's engine log (its resolved_sampler.json in the "
                         "re-audit's evidence dir); every verdict then carries it as resolved_sampler_verdict with resolved_sampler_source {path, sha256, "
                         "rederived_by}, the root's own record under historical_resolved_sampler_verdict; requires --audited-at")
    args = ap.parse_args(argv)
    if args.audited_at is not None and not re.fullmatch(r"[0-9a-f]{40}", args.audited_at):
        refuse(f"--audited-at {args.audited_at!r} is not a 40-hex tree sha")
    if args.resolved_sampler is not None and args.audited_at is None:
        refuse("--resolved-sampler needs --audited-at (the tree whose recorder re-derived the verdict)")
    legs = [l.strip() for l in args.legs.split(",") if l.strip()]
    bad = [l for l in legs if l not in ac.LEGS]
    if bad:
        refuse(f"--legs: unknown leg(s) {bad}")
    petri_behaviours = [b.strip() for b in args.petri_behaviours.split(",") if b.strip()]
    if "petri" in legs and not petri_behaviours:
        refuse("--petri-behaviours is required when the petri leg is audited (its row tasks are petri/<behaviour>)")
    thinking = _parse_thinking(args.thinking_expect)
    if args.roots:
        roots = [ac.read_attempt_root(p) for p in args.roots]
    elif args.stores is not None:
        roots = ac.discover_attempt_roots(args.stores)
    else:
        refuse("one of --stores / --roots is required")
    out_dir = Path(args.out)
    summary_p = out_dir / SUMMARY_NAME
    if (out_dir / "request_log").exists():
        refuse(f"{out_dir / 'request_log'} exists — the audit writes a fresh tree; remove it deliberately to re-run")
    client_status = client_task_name_status()
    plan = ac.read_launch_plan(args.plan) if args.plan else None
    reads, ev_cache, anc_cache = ac.RootReads(), {}, {}  # one run: every root's pipeline and every ancestor loaded once
    verdicts: List[Dict[str, Any]] = []
    for root in roots:  # rung roots hold re-draws of the base's items under the same rule and are audited the same way
        verdicts.extend(audit_root(root, out_dir=out_dir, legs=legs, policy_file=args.policy_file, thinking_expect=thinking,
                                   petri_behaviours=petri_behaviours, model_key=args.model_key, client_status=client_status,
                                   siblings=roots, reads=reads, evidence_cache=ev_cache, ancestor_cache=anc_cache, plan=plan,
                                   audited_at=args.audited_at, resolved_sampler=args.resolved_sampler))
    if not verdicts:
        refuse("no store of the requested legs under the given roots — nothing to audit")
    rows = [{c.rstrip("?"): v.get(c.rstrip("?")) for c in SUMMARY_COLUMNS} for v in verdicts]
    ac.write_csv(summary_p, rows, SUMMARY_COLUMNS)
    write_verdict_index(out_dir, verdicts)     # F-AO: the writer's index of the files it filed, for the app's / certificate's binding
    n_ref = sum(1 for v in verdicts if v["verdict"] != VERDICT_AGREES)
    print(f"[request-evidence-audit] {len(verdicts)} store verdict(s): {len(verdicts) - n_ref} AGREES, {n_ref} REFUSED → {summary_p}")
    for v in verdicts:
        if v["verdict"] != VERDICT_AGREES:
            print(f"  REFUSED {v['leg']}/{v['arm']}__s{v['s']}: {v['problem']}")
    return ac.EXIT_OK if n_ref == 0 else ac.EXIT_FAILED


if __name__ == "__main__":
    sys.exit(ac.run_main(main))
