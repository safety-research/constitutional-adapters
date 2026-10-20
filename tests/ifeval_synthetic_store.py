"""SYNTHETIC IFEval stores for the ifeval-canonical-battery tests (the draw-subset contract, the k = 2 ladder adapter, the
analysis). Placeholder shas, ``TESTHOST``, invented item ids (``ife_0000`` …) and prompt texts — never real recorded data.

Every store is written THROUGH THE REAL WRITERS AND RECORD BUILDERS of this tree, so the reader and the producer cannot
drift apart without a red test: ``inspect_served.outcome_record`` over ``ItemResult`` objects (k = 2 draws per item, the
IFEval row facts riding ``ItemResult.row_facts`` — the shape ``ifeval_row_facts`` produces),
``completion.finish_task`` (summary, the outcome record, the DONE marker — with ``n_draws`` / ``n_expected_draws`` per the
draw-subset contract), ``contracts.sampling_identity`` / ``sampling_ref``, ``usage_ledger.ledger_row``,
``ifeval_contracts.request_evidence_record_v3`` over the wire bytes (a pure delegate to ``contracts.request_evidence_record``
at version 3 — the production writer), ``inspect_ai.log.write_eval_log`` for both samples' ``.eval`` logs (dict-valued ``instruction_following``
scores, the stock task's grammar). THE LADDER RECORD AND THE RUNG SUBSET ARE THE ADAPTER'S OWN: ``inspect_rung.
classify_store(task=inspect/ifeval)`` over the finished store with the fixture's draw texts, filed at
``ifeval_paths(...).cap_ladder``; its ``rung_subset/2`` body written VERBATIM (validated by
``ifeval_contracts.validate_item_subset_v2``); a rung store's ``cap_rung`` identity and ``item_subset`` block are
``cap_rung_identity_v2`` / ``item_subset_block_v2`` of that subset.

The matrix is the FROZEN production matrix (``configs/sdfbase/ifeval/matrix.json``, 40 rows: 30 launch + 2 held + 8
deferred AlphaSteer / Arditi slots).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import heapq
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sisteer.capeval.serving import usage_ledger as ul
from sisteer.capeval.serving.usage_ledger import ledger_row
from sisteer.capeval.tiers import cap_ladder as CL
from sisteer.capeval.tiers import inspect_rung as IR
from sisteer.capeval.tiers import inspect_served as isv
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import ifeval_contracts as IC
from sisteer.capeval.serving import geometry as SG

REPO = Path(__file__).resolve().parents[1]
# The planner is a scripts/ module, importable only with scripts/ on sys.path. Every importer of this helper used to rely on
# ANOTHER test module in the same pytest process having done the insert (test_ifeval_analysis_review_r2.py inserts it AFTER
# importing this helper; test_ifeval_ladder_plan.py / test_tinycap_launcher_port.py before) — so tests/test_ifeval_analysis_review_r{2,3}.py
# collected only when they shared a shard with one of those, and failed alone (the Slurm gate's r2/r3 "collection flip",
# 2026-09-29). The helper now puts scripts/ on the path itself, once, so it imports in any order.
import sys as _sys
if str(REPO / "scripts") not in _sys.path:
    _sys.path.insert(0, str(REPO / "scripts"))
import tinycap_ifeval_ladder_plan as LP        # the REAL planner files every ladder record here (THE decision, C1)

FULL_SHA = "7777777788888888999999990000000011111111"
SHA7 = FULL_SHA[:7]
TASK = IC.IFEVAL_TASK
TASK_NAME = IC.IFEVAL_TASK_NAME
N = IC.IFEVAL_N
K = IC.IFEVAL_SAMPLES_PER_ITEM
CAPS = IC.IFEVAL_CAP_LADDER
CAP1 = IC.IFEVAL_PASS1_CAP
MODE = IC.SAMPLING_VENDOR_PRESET
SCORER_NAME = IC.PRIMARY_SCORER_IFEVAL
#: the synthetic item ids (never the vendor's keys) and prompts
IDS: list[str] = [f"ife_{i:04d}" for i in range(N)]
PROMPTS: list[str] = [f"Synthetic instruction prompt {i}: write about topic {i} following {1 + i % 3} rule(s)." for i in range(N)]
N_INST: list[int] = [1 + (i % 3) for i in range(N)]
INSTRUCTION_IDS = ("length_constraints:number_words", "detectable_format:json_format", "punctuation:no_comma")
PROXY_URL, LEDGER_URL = "http://127.0.0.1:19000", "http://127.0.0.1:19001"
UPSTREAM_TIMEOUT_S = 7200.0
UPSTREAM_TIMEOUT_REFUSING_S = 600.0
PROMPT_BASE = 60
DECODE_TOK_S = 40.0
TS0 = "2026-01-01T00:00:00Z"
CHANNELS_NAME = "synthetic_channels.json"
RESULT_NAME = "RESULT"
#: a loop by Evals' rule (whitespace 4-grams repeat) and a non-loop long text
LOOP_TEXT = ("the rule is the rule is the rule is " * 120).strip()
TRUNC_TEXT = " ".join(f"section{i} explains point{i} then continues{i}" for i in range(300))
#: the served window per cap: the G-LENGTHS entries (max prompt 468 + cap, block 16) for the short prompt class, and the
#: constitution-prompted cells' own entries (max prompt 6,385 + cap) — keyed by the row's geometry entry
MAX_MODEL_LEN_BY_CAP = {2048: 2528, 4096: 4576, 8192: 8672}
MAX_MODEL_LEN_BY_ENTRY = {**{f"capability:ifeval_{c}": L for c, L in MAX_MODEL_LEN_BY_CAP.items()},
                          "capability:ifeval_longsi_2048": 8448, "capability:ifeval_longsi_4096": 10496,
                          "capability:ifeval_longsi_8192": 14592}
#: per cell the PROMPT-STRICT pass rate the fixture draws with (deterministic per (item, trial, cell))
PASS_RATE_BY_CELL: Mapping[str, float] = {
    IC.CELL_NONE: 0.72, IC.CELL_PROMPTED: 0.70,
    "constv2-mix-full__psr_ce_all": 0.60, "constv2-mix-full__lora_r4_ce": 0.62,
    "constv2ctl2-full__psr_ce_all": 0.71, "constv2ctl2-full__lora_r4_ce": 0.715,
    "safety_refusal__psr_kl_all": 0.66, "safety_refusal__lora_r4": 0.67,
    "constv2-mix-full__psr_ce_all_minus_ctl_distilled": 0.69, "constv2-mix-full__lora_r4_ce_minus_ctl": 0.695,
    IC.CELL_PROMPTED_CONSTITUTION: 0.68, "alphasteer": 0.65, "arditi": 0.64,
}
MANIFEST_KEY_ACCEPTED_ATTEMPT = C.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT
ATTEMPT_ID = "a1"
#: a retry (C4): the INTERRUPTED first attempt ``a1`` sent some task requests (served, ledgered, evidenced) and died; the
#: accepted attempt is then ``a2``, complete over every draw — every historical row stays on disk, as the driver leaves it
ATTEMPT_ID_RETRY = "a2"
TS_ABANDONED = "2025-12-31T23:00:00Z"


def _hash(s: str, n: int = 64) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def kv_shrink(configured: int, served: int, max_model_len: int) -> dict[str, Any]:
    """A KV-derived shrink record in the PRODUCER's shape (``sisteer.capeval.serving.geometry.KV_SHRINK_FIELDS``) that HOLDS
    the rule for these numbers: the served S is ``floor`` of the reported concurrency, the capacity divides to it at this
    window, and it is below the configured S — what a Modal standup records when the allocator's KV cannot back the
    configured S (the Nemotron 4,096 rung: 171 -> 168)."""
    assert served < configured
    return {"configured_max_num_seqs": int(configured), "served_max_num_seqs": int(served), "max_model_len": int(max_model_len),
            "kv_backed_concurrency": served + 0.09, "kv_capacity_tokens": served * max_model_len + max_model_len // 2,
            "basis": "hybrid: the allocator's kv_max_concurrency", "rule": SG.KV_SHRINK_RULE, "reported_by": "synthetic standup"}


#: the endpoint's thinking-header probe per family (the real records: gemma / Qwen / Nemotron templates carry the switch, the
#: wrapper composes the OFF pin on each — gemma's does not MOVE the header; llama's template has no switch, nothing is pinned)
THINKING_HEADER_BY_FAMILY: Mapping[str, dict[str, Any]] = {
    "gemma4": {"switch": True, "moves": False, "off_ids": [105, 4368, 107], "on_ids": [105, 4368, 107]},
    "qwen3_5": {"switch": True, "moves": True, "off_ids": [1, 2, 3, 4, 5, 6, 7], "on_ids": [1, 2, 3, 4, 5]},
    "nemotron_h": {"switch": True, "moves": None, "off_ids": [10, 11, 12, 13, 14, 15], "on_ids": None},
    "llama": {"switch": False, "moves": None, "off_ids": [20, 21, 22, 23], "on_ids": None},
}


def served_template_kwargs(family: str) -> dict[str, Any]:
    return json.loads(IC.THINKING_OFF_TEMPLATE_KWARGS) if THINKING_HEADER_BY_FAMILY[family]["switch"] else {}


def mamba_attestation(*, ok: bool = True, n_mamba_layers: int = 21) -> dict[str, Any]:
    """The standup's own fused-Mamba attestation in the producer's shape (``mamba_backend.live_attest``'s record)."""
    return {"ok": ok, "problems": [] if ok else ["synthetic: a layer on the fallback"], "rpc_method": "sisteer_mamba_backend_report",
            "contract_sha256": _hash("mamba contract"), "expected_n_mamba_layers": n_mamba_layers, "asked_at": TS0,
            "reports": [{"rpc_method": "sisteer_mamba_backend_report", "n_layers": 2 * n_mamba_layers, "n_mamba_layers": n_mamba_layers}]}


# --------------------------------------------------------------------------- the matrix
def matrix_rows() -> list[dict[str, Any]]:
    return json.loads((REPO / IC.IFEVAL_MATRIX_REL).read_text(encoding="utf-8"))["rows"]


def row_for(rows: Sequence[Mapping[str, Any]], key: str, view: str, cell: str) -> dict[str, Any]:
    for r in rows:
        if (r["model_key"], r["view"], r["cell"]) == (key, view, cell):
            return dict(r)
    raise KeyError((key, view, cell))


def launch_rows(rows: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = matrix_rows() if rows is None else rows
    return [dict(r) for r in rows if r["status"] == IC.STATUS_LAUNCH]


def pins_doc(rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    rows = matrix_rows() if rows is None else rows
    out: dict[str, Any] = {}
    for r in rows:
        if r["fitted"]:
            out.setdefault(r["model_key"], {})[r["staged_id"]] = {"object_sha256": r["object_sha256"]}
    return out


# --------------------------------------------------------------------------- the draws
def draw_facts(cell: str, i: int, trial: int, *, at_cap: bool = False) -> dict[str, Any]:
    """The four IFEval values + the instruction count of ONE draw, deterministic in (cell, item, trial): the strict
    prompt-level pass drawn against the cell's rate; loose >= strict; instruction counts consistent. An AT-CAP draw fails
    what it did not complete: one instruction fewer than it would have (scored exactly as IFEval scores it — never
    excluded)."""
    n_inst = N_INST[i]
    u = int(_hash(f"{cell}|{i}|{trial}", 8), 16) / 16 ** 8
    rate = PASS_RATE_BY_CELL.get(cell, 0.7)
    inst_strict = n_inst if u < rate else max(0, int(u * 100) % n_inst)
    if at_cap:
        inst_strict = max(0, inst_strict - 1)
    inst_loose = min(n_inst, inst_strict + (1 if (int(u * 1000) % 5 == 0 and inst_strict < n_inst) else 0))
    return {"prompt_level_strict": inst_strict == n_inst, "prompt_level_loose": inst_loose == n_inst,
            "inst_level_strict": inst_strict, "inst_level_loose": inst_loose, "num_instructions": n_inst,
            "instruction_id_list": list(INSTRUCTION_IDS[:n_inst])}


#: the weights each fitted object was FIT on, in the PRODUCER's `fit_provenance` shape: the `main` bank view
#: holds base-weight document fits served on the instruct checkpoint (a transfer, `transfer: true`), the `it` view
#: instruct-weight document fits served on their own weights (`transfer: false`); Nemotron has one checkpoint. Synthetic
#: fit revisions (placeholders), the served pins the frozen matrix's.
BASE_FIT_PINS: Mapping[str, tuple[str, str]] = {"gemma4e4b": ("google/gemma-4-E4B", "1111111111111111111111111111111111111111"),
                                                "qwen35_4b": ("Qwen/Qwen3.5-4B-Base", "2222222222222222222222222222222222222222"),
                                                "llama8b": ("NousResearch/Meta-Llama-3.1-8B", "3333333333333333333333333333333333333333")}


def fit_provenance_block(row: Mapping[str, Any], *, hf: str, rev: str) -> dict[str, Any]:
    """The manifest's `fit_provenance` block as the served-capability producer writes it (its field names)."""
    if not row["fitted"]:
        return {"chat_template_bypassed": None, "fit_kind": None, "model_id": None, "object_sha": None, "override_evidence": None,
                "positions_at_fit": None, "revision": None, "served_model_id": hf, "served_revision": rev, "served_template_hash": None,
                "source": None, "status": None, "template_check": "baseline", "template_check_reason": "nothing was fit (baseline)",
                "template_hash": None, "training_budget_kind": None, "transfer": False}
    base = BASE_FIT_PINS.get(str(row["model_key"])) if row["view"] == IC.VIEW_MAIN else None
    fit_hf, fit_rev = base if base is not None else (hf, rev)
    return {"chat_template_bypassed": True, "fit_kind": "document", "model_id": fit_hf, "object_sha": str(row["object_sha256"])[:16],
            "override_evidence": None, "positions_at_fit": "all", "revision": fit_rev, "served_model_id": hf, "served_revision": rev,
            "served_template_hash": None, "source": "train_budget_meta", "status": "document_fit", "template_check": "inapplicable",
            "template_check_reason": "the fit bypassed the chat template (document fit)", "template_hash": "e3b0c44298fc1c14",
            "training_budget_kind": "document_tokens", "transfer": base is not None}


def sampling_block(key: str) -> dict[str, Any]:
    row = IC.vendor_preset_row(key)
    block = {"mode": MODE, "model_key": key, **C.sampling_identity(row, {TASK: K}), "source": row.source}
    block[C.IDENTITY_KEY_SEED_RULE] = IC.seed_rules_for((TASK,))
    return block


def task_plan(key: str, ids: Sequence[str], *, cap: int, draws: Sequence[tuple[str, int]] | None = None,
              thinking: str = C.THINKING_OFF) -> isv.TaskPlan:
    row = IC.vendor_preset_row(key)
    return isv.TaskPlan(row_task=TASK, name=TASK_NAME, task=None, sample_ids=list(ids), samples_per_item=K, max_tokens=cap,
                        primary_scorer=SCORER_NAME, expected_n=len(ids), n_run=len(ids), temperature=row.temperature,
                        top_p=row.top_p, top_k=row.top_k, item_set="ifeval541", thinking=thinking,
                        cap_source=("off_table" if cap == CAP1 else IC.CAP_SOURCE_OFF_CAP_EXPLICIT),
                        min_p=row.min_p, presence_penalty=row.presence_penalty, repetition_penalty=row.repetition_penalty,
                        note="synthetic ifeval plan")


def task_body(row: Mapping[str, Any], srow: C.SamplingRow, *, item_id: str, seed: int, cap: int) -> dict[str, Any]:
    """The provider's wire body for one draw — the bytes AS THEY LEAVE THE CLIENT: ``inspect_ai`` hands ``top_k`` /
    ``min_p`` / ``repetition_penalty`` to the OpenAI client as ``extra_body``, and the client MERGES ``extra_body`` into
    the top level of the JSON it sends, so on the wire every sampling term is a top-level key beside ``temperature`` /
    ``top_p`` / ``presence_penalty`` / ``seed`` (the production writer ``contracts.request_evidence_record`` reads the
    wire body's top level, and nothing else); the thinking-OFF kwarg on every request of a family WITH a switch."""
    body: dict[str, Any] = {"model": C.cell_spellings(str(row["cell"]), declared=False).vmodel,
                            "messages": [{"role": "user", "content": PROMPTS[IDS.index(item_id)]}],
                            "max_tokens": int(cap), "temperature": float(srow.temperature), "seed": int(seed)}
    if row["cell"] == IC.CELL_PROMPTED:
        body["messages"].insert(0, {"role": "system", "content": "SYNTHETIC SAFETY SI"})
    if srow.top_p is not None:
        body["top_p"] = float(srow.top_p)
    if srow.presence_penalty is not None:
        body["presence_penalty"] = float(srow.presence_penalty)
    # the client-side ``extra_body`` terms, merged into the top level as the OpenAI client sends them
    if srow.top_k is not None:
        body["top_k"] = int(srow.top_k)
    for f in ("min_p", "repetition_penalty"):
        if getattr(srow, f) is not None:
            body[f] = float(getattr(srow, f))
    # the no-switch policy: a family with no thinking channel (llama) runs its OFF arm as `not_available` and
    # the provider sends NO chat_template_kwargs — the evidence record then reads enable_thinking_present False, as produced
    if C.sends_thinking_kwargs(IC.ifeval_effective_arm(row["family"])):
        body[C.CHAT_TEMPLATE_KWARGS_KEY] = {C.ENABLE_THINKING_KEY: False}
    return body


def wire_bytes(body: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(body), separators=(",", ":")).encode("utf-8")


def _request_id(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:32]


def task_attempt(row: Mapping[str, Any], srow: C.SamplingRow, item: Mapping[str, Any], *, cap: int, ts: str, salt: str,
                 body_edit: Callable[[dict[str, Any]], None] | None = None, attempt_id: str = ATTEMPT_ID) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(request-evidence v3 record, ledger row)`` of one draw's attempt over the SAME wire bytes, both stamped ``attempt_id``."""
    sample = C.sample_header_value(item["item_id"], item["sample_idx"])
    body = task_body(row, srow, item_id=str(item["item_id"]), seed=int(item["seed"]), cap=cap)
    if body_edit:
        body_edit(body)
    raw = wire_bytes(body)
    rid = _request_id(salt, ts, TASK, sample)
    headers = {C.HEADER_TASK: TASK, C.HEADER_SAMPLE: sample, C.HEADER_REQUEST_ID: rid}
    rec = IC.request_evidence_record_v3(raw, headers=headers, ts=ts)
    lrow = dict.fromkeys(C.LEDGER_ROW_FIELDS)
    lrow.update(ts=ts, task=TASK, sample=sample, has_tools=False, n_tools=0, rejected=False, status=200, error_code=None,
                prompt_tokens_pre=int(item["prompt_tokens"]), usage_prompt_tokens=int(item["prompt_tokens"]),
                usage_completion_tokens=int(item["completion_tokens"]),
                finish_reason=CL.LEDGER_FINISH_AT_CAP if item["at_cap"] else "stop",
                latency_s=round(int(item["completion_tokens"]) / DECODE_TOK_S, 3), tokenize_mismatch=False,
                error_class=C.LEDGER_ERROR_CLASS_NONE, request_id=rid, body_sha256=C.wire_body_sha256(raw))
    lrow.update(ul.body_facts(body))
    lrow["thinking"] = False
    return {**rec, C.REQUEST_EVIDENCE_ATTEMPT_FIELD: attempt_id}, {**ledger_row(**lrow), C.LEDGER_ATTEMPT_ID_FIELD: attempt_id}


def request_starts(completion_tokens: Sequence[int], served_s: int, *, t0: str = TS0) -> list[str]:
    """The accepted attempt's request arrival stamps (the ledger's ``ts``, ISO-8601 with milliseconds and an offset, as the
    ledger writes them), scheduled the way a served engine runs them: ``served_s`` concurrent slots, each request taking
    the slot that frees first and holding it for ``completion_tokens / DECODE_TOK_S`` seconds (the fixture's per-sequence
    decode rate, the same figure ``task_attempt`` writes as ``latency_s``). So the rows OVERLAP in time and the aggregate
    rate over the generation wall is ~``served_s x DECODE_TOK_S`` (less the last wave's tail), while ``sum(tokens) /
    sum(latency_s)`` is exactly ``DECODE_TOK_S`` — the two readings C5 tells apart."""
    base = _dt.datetime.fromisoformat(t0.replace("Z", "+00:00"))
    slots = [0.0] * max(1, int(served_s))
    heapq.heapify(slots)
    out: list[str] = []
    for n in completion_tokens:
        start = heapq.heappop(slots)
        heapq.heappush(slots, start + round(int(n) / DECODE_TOK_S, 3))
        out.append((base + _dt.timedelta(seconds=start)).isoformat(timespec="milliseconds"))
    return out


# --------------------------------------------------------------------------- the store
def build_store(gen_root: Path, row: Mapping[str, Any], *, at_cap: Iterable[tuple[int, int]] = (), loops: Iterable[tuple[int, int]] = (),
                cap: int = CAP1, draws: Sequence[tuple[str, int]] | None = None,
                rung_source: Mapping[str, Any] | None = None, subset_doc: Mapping[str, Any] | None = None,
                subset_path: str | None = None, sha: str = SHA7, tree_sha: str = FULL_SHA,
                completion_tokens: Mapping[tuple[int, int], int] | None = None,
                texts: Mapping[tuple[str, int], str] | None = None,
                kv_shrink: Mapping[str, Any] | None = None, transport_ok: bool = True,
                summary_edit: Callable[[dict[str, Any]], None] | None = None,
                header_edit: Callable[[dict[str, Any]], None] | None = None,
                manifest_edit: Callable[[dict[str, Any]], None] | None = None,
                ledger_edit: Callable[[list[dict[str, Any]]], None] | None = None,
                request_evidence_edit: Callable[[list[dict[str, Any]]], None] | None = None,
                body_edit: Callable[[dict[str, Any]], None] | None = None,
                rows_edit: Callable[[list[dict[str, Any]]], None] | None = None,
                endpoint_edit: Callable[[dict[str, Any]], None] | None = None,
                geometry_edit: Callable[[dict[str, Any]], None] | None = None,
                evidence_edit: Callable[[dict[str, Any]], None] | None = None,
                write_recorder: bool = True,
                ladder: bool = True, write_ledger: bool = True, write_request_evidence: bool = True,
                n_expected_draws: int | None = "auto", identity_edit: Callable[[dict[str, Any]], None] | None = None,
                drop_draws: Iterable[tuple[str, int]] = (), abandoned_attempt: int | None = None) -> Path:
    """One leg's store under ``<gen_root>/<ifeval_paths(...).out_root>`` (pass 1) or the rung's root. Draw coordinates
    are ``(item index, trial)`` pairs: ``at_cap`` the draws stopped by the cap, ``loops`` the at-cap draws whose text is
    degenerate. A RUNG store passes ``draws`` (the subset's ``(item_id, sample_idx)`` list), ``rung_source`` (the SOURCE
    store's manifest), ``subset_doc`` (the adapter's validated ``rung_subset/2`` body) and ``subset_path``.
    ``abandoned_attempt=n`` (C4) prepends an INTERRUPTED attempt ``a1`` that sent the first ``n`` draws' requests before
    dying; the accepted, complete attempt is then ``a2`` and the manifest records both. Returns the cell dir."""
    key, cell, view = str(row["model_key"]), str(row["cell"]), str(row["view"])
    fitted = bool(row["fitted"])
    srow = IC.vendor_preset_row(key)
    # the arm the task RUNS under (the family policy): `off`, or `not_available` on a no-switch family
    # (llama) — hashed, stamped on every row / the summary / the outcome header, the manifest's switch null, the proxy's
    # render regime `unspecified`; the MODE stays `off` (the frozen matrix's `thinking`, the launcher line)
    arm = IC.ifeval_effective_arm(row["family"])
    is_rung = draws is not None
    paths = IC.ifeval_paths(key, view, cell, cap)
    root = Path(gen_root) / paths.out_root
    at_cap_set, loop_set = set(at_cap), set(loops)
    length_of = dict(completion_tokens or {})
    dropped = set(drop_draws)          # a TORN pass-1 store: these (item_id, trial) draws were never filed
    pairs: list[tuple[str, int]] = (sorted(draws) if is_rung else [(iid, t) for iid in IDS for t in range(K) if (iid, t) not in dropped])
    ids = sorted({i for i, _ in pairs}, key=IDS.index)
    spell = C.cell_spellings(cell, declared=False)
    vmodel = spell.vmodel
    hf, rev = IC.IFEVAL_MODEL_PINS[key]
    ihash = _hash(f"{key}|{view}|{cell}|ihash", 8)
    chash = _hash(f"{key}|{view}|{cell}|{MODE}|{cap}|ifeval|chash", 8)
    cell_key = f"{paths.cell_slug}__controls__t0__a0__{ihash}__{chash}"
    cell_dir = root / "cells" / cell_key
    cell_dir.mkdir(parents=True, exist_ok=True)
    block = sampling_block(key)
    sampling_ref = C.sampling_ref(block)
    results: list[isv.ItemResult] = []
    channels: dict[tuple[str, int], str] = dict(texts or {})
    for item_id, trial in pairs:
        i = IDS.index(item_id)
        is_at = (i, trial) in at_cap_set
        is_loop = (i, trial) in loop_set
        facts = draw_facts(cell, i, trial, at_cap=is_at)
        default_text = LOOP_TEXT if is_loop else (TRUNC_TEXT if is_at else f"Answer {i}/{trial}: a short compliant response.")
        content = channels.setdefault((item_id, trial), default_text)
        results.append(isv.ItemResult(
            item_id=item_id, value=1.0 if facts["prompt_level_strict"] else 0.0, sample_idx=trial,
            seed=IC.ifeval_sample_seed(item_id, trial), answer=content, prompt_tokens=PROMPT_BASE + (i % 400),
            completion_tokens=cap if is_at else length_of.get((i, trial), 150 + (i * 7 + trial * 13) % 600),
            paired_id=None, scores={SCORER_NAME: 1.0 if facts["prompt_level_strict"] else 0.0},
            content=content, reasoning=None, content_empty_with_reasoning=False,
            stop_reason=isv.STOP_REASON_AT_CAP if is_at else "stop", unparseable=False, attempt=1, max_tokens_used=cap,
            think_block_present=False, think_block_unterminated=False, stray_open_tag=False, row_facts=facts))
    plan = task_plan(key, ids, cap=cap, draws=pairs if is_rung else None, thinking=arm)
    counts = isv.attempted_counts(results)
    split_counts = isv.think_split_counts(results, plan=plan)
    n_scored = counts["n_scored"]
    stamps = IC.ifeval_expected_summary_stamps(row)      # the render regime family-resolved (`unspecified` on llama)
    by_trial = {t: [r for r in results if r.sample_idx == t] for t in range(K)}
    summary: dict[str, Any] = {
        C.METRIC_ACC: (sum(r.value for r in results) / len(results)) if results else None, C.METRIC_N: counts["n"],
        "n_requests": len(results), "n_rejected": 0, "n_http_rejected_retried": 0, "samples_per_item": K, "task": TASK,
        "item_set": "ifeval541", "n_attempted_samples": len(results),
        "framework": C.INSPECT_TASK_VERSION, "engine": "vllm", "route": vmodel, "thinking": arm,
        "render": C.THINKING_RENDER_NATIVE, "wall_clock_s": 600.0, "logs_reused": 0, "reasoning_parser": None,
        # the producer's operational record (`inspect_served.client_record`): the SUMMARY carries the client timeout
        "max_retries": 2, isv.CLIENT_TIMEOUT_FIELD: 7200.0,
        "n_content_empty_with_reasoning": 0, "n_scored": n_scored, "n_unscored": counts["n_unscored"],
        "n_scored_samples": counts["n_scored_samples"], "n_unscored_samples": counts["n_unscored_samples"],
        **isv.task_quality(results), **split_counts,
        "max_tokens": cap, "temperature": srow.temperature, "top_p": srow.top_p, "top_k": srow.top_k, "grader_max_tokens": None,
        "ifeval_task_version": IC.IFEVAL_TASK_VERSION, "langdetect_seed": IC.LANGDETECT_SEED,
        **{f"{IC.METRIC_BY_FIELD[f]}_mean_by_trial": {str(t): (sum(float(r.row_facts[f]) for r in by_trial[t]) / len(by_trial[t]) if by_trial[t] else None)
                                                      for t in range(K)} for f in ("prompt_level_strict", "prompt_level_loose")},
        C.SAMPLING_REF_FIELD: sampling_ref,
    }
    for f in C.EXPECTED_STAMP_TRIPLE_FIELDS:
        summary[f] = stamps[f]
        summary[C.STAMP_REGIME_COUNT_FIELD[f]] = 1
    if summary_edit:
        summary_edit(summary)
    header = isv.outcome_record(results, plan=plan, task=TASK, thinking=arm, summary=summary,
                                split_counts={k: summary[k] for k in split_counts}, producer_tree_sha=tree_sha,
                                sampling_ref=sampling_ref)
    # the IFEval row facts on every outcome item (lane 1's outcome_record copies them for this task; the same six keys)
    facts_by = {(r.item_id, r.sample_idx): r.row_facts for r in results}
    header["items"] = [{**it, **facts_by[(it["item_id"], it["sample_idx"])]} for it in header["items"]]
    if header_edit:
        header_edit(header)

    max_model_len = MAX_MODEL_LEN_BY_ENTRY[row["geometry_eval_by_cap"][str(cap)]]
    seqs = {"gemma4e4b": 233, "qwen35_4b": 142, "nemotron3nano4b": 256, "llama8b": 187}[key]
    served_s = int(kv_shrink["served_max_num_seqs"]) if kv_shrink else seqs
    configured_s = int(kv_shrink["configured_max_num_seqs"]) if kv_shrink else seqs
    tinycap: dict[str, Any] = {
        "tasks": [TASK], "pass": None, "samples": K,
        "served": {"reasoning_parser": None, "tool_calls": False, "max_model_len": max_model_len, "max_num_seqs": served_s,
                   "concurrency": served_s, C.AGENT_LOOP_ENTRY_KEY: True, C.REDUCTION_MODE_FIELD: "stacked"},
        "thinking": {"mode": C.THINKING_OFF, "by_task": {TASK: arm}, "render": "flag_only"},
        C.IDENTITY_KEY_THINKING_BY_TASK: {TASK: arm},
        "versions": {"tinycap": C.TINYCAP_VERSION}, "limits": {}, "frameworks": {"inspect_ai": "0.3.260"},
        C.IDENTITY_KEY_IDENTITY_VERSION: C.TINYCAP_IDENTITY_VERSION_2,
        C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: {TASK: cap},
        **{k: block[k] for k in C.SAMPLING_IDENTITY_FIELDS},
        **{f: block[f] for f in C.SAMPLING_OPTIONAL_IDENTITY_FIELDS if f in block},
    }
    item_subset_block = None
    if is_rung:
        assert rung_source is not None and subset_doc is not None
        doc = IC.validate_item_subset_v2(dict(subset_doc), where="synthetic rung subset")
        tinycap[C.IDENTITY_KEY_CAP_RUNG] = IC.cap_rung_identity_v2(doc)
        item_subset_block = IC.item_subset_block_v2(doc, path=str(subset_path or (Path(gen_root) / paths.subset)))
    if identity_edit:
        identity_edit(tinycap)
    pin = str(row["object_sha256"]) if fitted else None
    accepted_id = ATTEMPT_ID if abandoned_attempt is None else ATTEMPT_ID_RETRY
    attempts = [{"ts": TS0, "tree_sha": tree_sha, "attempt": 1, "completed": True, "attempt_id": ATTEMPT_ID,
                 "kind": "generation", "rc": 0, "terminal": True, "started": TS0, "ended": "2026-01-01T00:30:00Z"}]
    if abandoned_attempt is not None:
        attempts = [{"ts": TS_ABANDONED, "tree_sha": tree_sha, "attempt": 1, "completed": False, "attempt_id": ATTEMPT_ID,
                     "kind": C.ATTEMPT_KIND_GENERATION, "generation_calls": int(abandoned_attempt), "rc": None,
                     "terminal": C.ATTEMPT_TERMINAL_INTERRUPTED, "started": TS_ABANDONED, "ended": None},
                    {"ts": TS0, "tree_sha": tree_sha, "attempt": 2, "completed": True, "attempt_id": ATTEMPT_ID_RETRY,
                     "kind": C.ATTEMPT_KIND_GENERATION, "generation_calls": len(pairs), "rc": 0,
                     "terminal": C.ATTEMPT_TERMINAL_COMPLETED, "started": TS0, "ended": "2026-01-01T00:30:00Z"}]
    manifest: dict[str, Any] = {
        "cell_id": cell, "harness_cell_id": spell.harness, "virtual_model": vmodel,
        "intervention_hash": ihash + "0" * 8, "harness_config_hash": chash + "0" * 8,
        "model_id": hf, "model_revision": rev,
        "fit_provenance": fit_provenance_block(row, hf=hf, rev=rev),
        C.MANIFEST_KEY_PRODUCER: dict(zip(C.PRODUCER_BLOCK_FIELDS, (tree_sha, C.PRODUCER_DRIVER_SERVED))),
        C.MANIFEST_KEY_CONTRACT_VERSION: C.CONTRACT_VERSION,
        C.MANIFEST_KEY_PROVENANCE: {"cell_id": cell, "class": "A" if fitted else "baseline", "artifact_sha256": pin,
                                    "intervention_hash": ihash + "0" * 8, "model": hf, "model_revision": rev},
        C.MANIFEST_KEY_TINYCAP: tinycap, "resolved_config": {"tinycap": tinycap},
        C.MANIFEST_KEY_ROUTE: {"virtual_model": vmodel, "grader_virtual_model": None, "tool_calls": False, C.AGENT_LOOP_ENTRY_KEY: True,
                               "serving_identity": "proxy_vllm+lens+interleaved", C.REDUCTION_MODE_FIELD: "stacked",
                               C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: pin,
                               C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {spell.harness: pin} if fitted else {},
                               C.ENDPOINT_KEY_LENS_SALT_POLICY: C.LENS_SALT_POLICY_ATTESTED,
                               C.ROUTE_KEY_RESUMED_PREFILL_POLICY: stamps["resumed_policy"],
                               C.LENS_SERVED_POSITIONS_KEY: ([str(row["served_positions_expected"])] if fitted else None),
                               "view": view, "route_kind": row["route_kind"], "bank_view": view},
        C.MANIFEST_KEY_THINKING: {"enable_thinking": (False if C.sends_thinking_kwargs(arm) else None), "render": C.THINKING_RENDER_NATIVE,
                                  "reasoning_parser": None, "mode": C.THINKING_OFF, "by_task": {TASK: arm}},
        # the driver's copy of the standup's geometry — the Modal launcher hands the standup EXPLICIT dims, so the copy reads
        # `eval None`, `window_source` / `seqs_source` 'override' (the real stores, V20 / V36 (ii)); the served record's own
        # block (`served_geometry.json`) names the entry and `table`
        C.MANIFEST_KEY_GEOMETRY: {"max_model_len": max_model_len, "max_num_seqs": served_s, C.GEOMETRY_PASS_FIELD: 1, "eval": None,
                                  "window_source": "override", "seqs_source": "override", "identity": True, "model_key": key,
                                  "source": "endpoint.json+served_geometry.json"},
        C.MANIFEST_KEY_CONCURRENCY: served_s, C.MANIFEST_KEY_SAMPLING: block,
        C.MANIFEST_KEY_SEEDS: IC.seed_rules_for((TASK,)),
        C.MANIFEST_KEY_USAGE_LEDGER: str(root / IC.LEDGER_FILENAME), "ledger_url": LEDGER_URL,
        C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S: UPSTREAM_TIMEOUT_S if transport_ok else UPSTREAM_TIMEOUT_REFUSING_S,
        "proxy_url": PROXY_URL,
        "binding": {"proxy_url": PROXY_URL, "ledger_url": LEDGER_URL, "ledger_upstream": PROXY_URL},
        "support": C.support_label(fitted=fitted, out_of_support=False), "header_located": bool(row["header_located"]) if fitted else None,
        C.LENS_SERVED_POSITIONS_KEY: ([str(row["served_positions_expected"])] if fitted else None),
        C.MANIFEST_KEY_RESUME_ATTEMPTS: attempts,
        MANIFEST_KEY_ACCEPTED_ATTEMPT: accepted_id,
        C.MANIFEST_KEY_REQUEST_EVIDENCE: str(root / C.REQUEST_EVIDENCE_FILENAME), "host": "TESTHOST",
    }
    if item_subset_block is not None:
        manifest[C.MANIFEST_KEY_ITEM_SUBSET] = item_subset_block
    if cap != CAP1:
        # the OFF-cap setter's statement, UNHASHED at the manifest TOP LEVEL — the producer's site (`served_capability_run`
        # writes `manifest[MANIFEST_KEY_OFF_CAP_BY_TASK]`); the fixture once put it under `resolved_config.tinycap`, where
        # the producer never writes it, and the reader that read it there refused every real rung leg (V36 (i))
        manifest[IC.MANIFEST_KEY_OFF_CAP_BY_TASK] = {TASK_NAME: cap}
    if manifest_edit:
        manifest_edit(manifest)
    (cell_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (cell_dir / "gen_stats.json").write_text(json.dumps({TASK: summary}, indent=2, default=str), encoding="utf-8")
    (cell_dir / "private").mkdir(exist_ok=True)
    (cell_dir / "private" / CHANNELS_NAME).write_text(json.dumps({f"{i}/{t}": c for (i, t), c in channels.items()}), encoding="utf-8")

    # rows.jsonl: per-sample acc@<idx> rows for every scored draw (+ the per-item mean rows at pass 1, none on a rung),
    # every row carrying the per-row identity (P42) in extra_json
    identity = {"object_artifact_sha256": pin, "route_kind": row["route_kind"], "view": view, "producer_tree_sha": tree_sha,
                "thinking": arm, "sampling_mode": MODE, C.SAMPLING_REF_FIELD: sampling_ref}
    rows_out = []
    for r in results:
        extra = r.extra(max_tokens=cap, **identity)
        rows_out.append({"task": TASK, "item_id": r.item_id, "metric": C.sample_metric(C.METRIC_ACC, r.sample_idx), "value": r.value,
                         "engine": "vllm", "kind": "cell", "domain": "inspect_gen", "extra_json": json.dumps(extra),
                         "harness_config_hash": chash + "0" * 8, "intervention_hash": ihash + "0" * 8, "seed": r.seed,
                         "trial_id": r.sample_idx})
    if not is_rung:
        for item_id in ids:
            rs = [r for r in results if r.item_id == item_id]
            extra = rs[0].extra(max_tokens=cap, samples_per_item=K, seeds=[r.seed for r in rs],
                                values=[r.value for r in rs], n_unscored_samples=0, provenance_sample_index=0,
                                per_sample=[r.sample_provenance() for r in rs], unscored=False, **identity)
            rows_out.append({"task": TASK, "item_id": item_id, "metric": C.METRIC_ACC, "value": sum(r.value for r in rs) / len(rs),
                             "engine": "vllm", "kind": "cell", "domain": "inspect_gen", "extra_json": json.dumps(extra),
                             "harness_config_hash": chash + "0" * 8, "intervention_hash": ihash + "0" * 8})
    if rows_edit:
        rows_edit(rows_out)
    lines = [json.dumps(r) for r in rows_out]
    lines.append(json.dumps({"task": TASK, "item_id": C.AGG_ITEM_ID, "metric": C.METRIC_ACC_AGGREGATE, "value": summary[C.METRIC_ACC],
                             "engine": "vllm", "kind": "cell", "extra_json": json.dumps({**identity, "sampling": block})}))
    lines.append(json.dumps({"task": TASK, "item_id": C.AGG_ITEM_ID, "metric": C.METRIC_N, "value": float(len(ids)), "engine": "vllm",
                             "kind": "cell", "extra_json": None}))
    (cell_dir / "rows.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # THE COMPLETION CONTRACT: the draw counts beside the item counts
    n_exp_draws = len(pairs) if n_expected_draws == "auto" else n_expected_draws
    completion.finish_task(cell_dir, task=TASK, n_items=len(ids), n_expected_full=len(ids), limits=None, n_rows=len(rows_out),
                           n_requests=len(results), n_rejected=0, wall_clock_s=600.0, summary=summary, outcomes=header,
                           n_draws=len(results), n_expected_draws=n_exp_draws)

    # the ledger and the v3 request evidence: one attempt per draw, the same wire bytes — preceded, on a retry, by the
    # interrupted attempt's rows (the driver appends across attempts; the accepted attempt is the manifest's)
    items_by = {(it["item_id"], it["sample_idx"]): it for it in header["items"]}
    recs, lrows = [], []
    if abandoned_attempt is not None:
        for item_id, trial in pairs[:int(abandoned_attempt)]:
            rec, lrow = task_attempt(row, srow, items_by[(item_id, trial)], cap=cap, ts=TS_ABANDONED, salt=f"{cell_key}|{ATTEMPT_ID}",
                                     body_edit=body_edit, attempt_id=ATTEMPT_ID)
            recs.append(rec)
            lrows.append(lrow)
    for (item_id, trial), ts in zip(pairs, request_starts([int(items_by[d]["completion_tokens"]) for d in pairs], served_s)):
        rec, lrow = task_attempt(row, srow, items_by[(item_id, trial)], cap=cap, ts=ts, salt=cell_key, body_edit=body_edit, attempt_id=accepted_id)
        recs.append(rec)
        lrows.append(lrow)
    if write_request_evidence:
        if request_evidence_edit:
            request_evidence_edit(recs)
        (root / C.REQUEST_EVIDENCE_FILENAME).write_text("\n".join(json.dumps(r, sort_keys=True) for r in recs) + "\n", encoding="utf-8")
    if write_ledger:
        if ledger_edit:
            ledger_edit(lrows)
        (root / IC.LEDGER_FILENAME).write_text("\n".join(json.dumps(r) for r in lrows) + "\n", encoding="utf-8")
    family = row["family"]
    entry = row["geometry_eval_by_cap"][str(cap)]
    template_hash = _hash(f"template|{family}", 16)
    endpoint: dict[str, Any] = {
        "upstream_timeout_s": UPSTREAM_TIMEOUT_S if transport_ok else UPSTREAM_TIMEOUT_REFUSING_S,
        C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S: UPSTREAM_TIMEOUT_S if transport_ok else UPSTREAM_TIMEOUT_REFUSING_S,
        "max_model_len": max_model_len, "max_num_seqs": served_s, "model": vmodel, "node": "TESTHOST",
        C.ENDPOINT_KEY_CHAT_TEMPLATE_HASH: template_hash,
        C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: pin,
        # the launcher's endpoint copy of the geometry: explicit dims (eval None, 'override'), the kv_shrink beside it
        "geometry": {"eval": None, "max_model_len": max_model_len, "max_num_seqs": served_s, "window_source": "override", "seqs_source": "override",
                     "identity": True, "model_key": key},
        SG.KV_SHRINK_KEY: dict(kv_shrink) if kv_shrink else None,
        "lens_served_positions": ([str(row["served_positions_expected"])] if fitted else None),
        "thinking_header": dict(THINKING_HEADER_BY_FAMILY[family]),
        "mamba_backend_attestation": (mamba_attestation() if family == "nemotron_h" else None),
    }
    if endpoint_edit:
        endpoint_edit(endpoint)
    served_geometry: dict[str, Any] = {
        "eval": entry, "max_model_len": max_model_len, "max_num_seqs": served_s, "batched_tokens": max_model_len * served_s,
        "budget": max_model_len * configured_s, "budget_source": "measured", "model_key": key, "over_budget": False, "identity": True,
        "window_source": "table", "seqs_source": (SG.KV_DERIVED_SOURCE if kv_shrink else "model_budget"), "provisional": False,
        "measurement_pass": False,
    }
    if kv_shrink:
        served_geometry["configured_max_num_seqs"] = configured_s
        served_geometry[SG.KV_SHRINK_KEY] = dict(kv_shrink)
    if geometry_edit:
        geometry_edit(served_geometry)
    # the Modal launcher's served-geometry record beside the store: geometry + the standup's endpoint + argv (no endpoint.json)
    (root / "served_geometry.json").write_text(json.dumps({"geometry": served_geometry, "standup_argv": ["capctrl_serving_up.py", "--cell", cell],
                                                           "endpoint": endpoint, "reduction_mode_requested": "stacked"}, indent=1), encoding="utf-8")
    evidence: dict[str, Any] = {"driver": "served_capability_run_v2", "cell_id": cell, "virtual_model": vmodel, "thinking": C.THINKING_OFF,
                                # the driver's serve-header evidence: the template kwargs the SERVER rendered with (the thinking pin as served)
                                "serve_header": {"served_template_hash": template_hash, "serve_header_ids": THINKING_HEADER_BY_FAMILY[family]["off_ids"],
                                                 "serve_header_len": len(THINKING_HEADER_BY_FAMILY[family]["off_ids"]),
                                                 "template_kwargs": served_template_kwargs(family)}}
    if evidence_edit:
        evidence_edit(evidence)
    (root / "evidence.json").write_text(json.dumps(evidence, indent=1), encoding="utf-8")
    if write_recorder:
        # the resolved-sampler recorder's record (the per-kind mode): the client class matched the pinned row
        (root / C.RESOLVED_SAMPLER_FILENAME).write_text(json.dumps({
            "verdict": "AGREES", "probe_verdict_by_kind": {"render": "EMPTY", "bites": "EMPTY", "lens_on_bite": "EMPTY", "load_check": "EMPTY", "positive_control": "EMPTY"},
            "n_client_requests": len(results), "n_unclassified_requests": 0, "n_probe_unclassified_requests": 0, "client_caps": [str(cap)],
            "probe_caps": ["48", "256", "512"], "reference_source": "--expect", "why": f"client {len(results)} request(s) = --expect", "error": None}, indent=1),
            encoding="utf-8")
    (root / RESULT_NAME).write_text(f"tinycap-modal name={row['job_id']} rc=0 result=COMPLETE status=OK sha={tree_sha} at={TS0}\n", encoding="utf-8")
    if ladder:
        file_decided_record(Path(gen_root), row, cap=cap, sha=sha)
    return cell_dir


# --------------------------------------------------------------------------- the adapter, the texts, the ladder
def store_texts(cell_dir: Path) -> dict[tuple[str, int], str | None]:
    """The fixture's draw texts (the adapter's text feed; the real store's come from the .eval logs)."""
    side = json.loads((Path(cell_dir) / "private" / CHANNELS_NAME).read_text(encoding="utf-8"))
    out = {}
    for k, v in side.items():
        i, _, t = k.rpartition("/")
        out[(i, int(t))] = v
    return out


def classify(root: Path, *, source_out_root: Path | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The planner's ``classify`` callable over the fixture's stores: the adapter with the fixture's draw texts; for a RUNG
    store the SOURCE's texts come from ``source_out_root`` when the planner resolved one (a relocated pull; seam 4, M11),
    else from the subset's recorded absolute root (the in-place workflow)."""
    cell_dir = IR.cell_dir_of(Path(root))
    facts = IR.load_store(Path(root), task=TASK)
    source_texts = None
    if facts.is_rung:
        src_root = Path(source_out_root) if source_out_root is not None else Path(str(facts.item_subset["source_out_root"]))
        source_texts = store_texts(IR.cell_dir_of(src_root))
    return IR.classify_store(Path(root), task=TASK, texts=store_texts(cell_dir), source_texts=source_texts,
                             source_out_root=source_out_root)


def _atomic_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def file_ladder_record(root: Path, record: Mapping[str, Any]) -> Path:
    p = Path(root) / IC.CAP_LADDER_FILENAME
    _atomic_json(p, record)
    return p


def file_decided_record(gen_root: Path, row: Mapping[str, Any], *, cap: int, sha: str = SHA7) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """The record the PLANNER files at ``ifeval_paths(...).cap_ladder`` for the store at ``cap``: the adapter's classification
    of the store (``classify``) plus THE decision over the merged reading (``tinycap_ifeval_ladder_plan.decided_record``, C1)
    — the fixtures write exactly what production writes. Returns ``(record, decision, subset | None)``."""
    root = Path(gen_root) / IC.ifeval_paths(row["model_key"], row["view"], row["cell"], cap).out_root
    record, adapter_subset = classify(root)
    rec, decision, subset = LP.decided_record(Path(gen_root), row, from_cap=cap, record=record, adapter_subset=adapter_subset, sha=sha)
    file_ladder_record(root, rec)
    return rec, decision, subset


def write_rung_subset(gen_root: Path, row: Mapping[str, Any], subset: Mapping[str, Any], *, rung: int) -> Path:
    p = Path(gen_root) / IC.ifeval_paths(row["model_key"], row["view"], row["cell"], rung).subset
    _atomic_json(p, IC.validate_item_subset_v2(dict(subset), where="synthetic rung subset"))
    return p


def build_escalated_cell(gen_root: Path, row: Mapping[str, Any], *, at_cap: Iterable[tuple[int, int]], loops: Iterable[tuple[int, int]] = (),
                         rung_still_at_cap: Iterable[tuple[int, int]] = (), rung_boundary: Iterable[tuple[int, int]] = (),
                         rung8192: bool = False, rung8192_still_at_cap: Iterable[tuple[int, int]] = (), **common: Any) -> dict[str, Any]:
    """A pass-1 store at 2,048 whose at-cap share of DRAWS triggers, its rung at 4,096 over exactly the adapter's draw
    subset (``rung_still_at_cap`` draws at cap again, ``rung_boundary`` landing below the source cap, the rest
    extended), and — when ``rung8192`` — the 8,192 rung over the merged reading's residual."""
    at_cap, loops = set(at_cap), set(loops)
    src_cd = build_store(gen_root, row, at_cap=at_cap, loops=loops, cap=CAP1, **common)
    src_root = src_cd.parents[1]
    # the planner's record (the adapter's classification + THE decision over the merged reading = the store at pass 1)
    record, decision, subset = file_decided_record(Path(gen_root), row, cap=CAP1, sha=common.get("sha", SHA7))
    out: dict[str, Any] = {"source_cell_dir": src_cd, "source_root": src_root, "record": record, "decision": decision, "rung_cell_dir": None,
                           "subset": None, "subset_path": None, "rung8192_cell_dir": None}
    if decision["exit"] != CL.ESCALATE:
        return out
    assert subset is not None and subset["schema"] == IC.ITEM_SUBSET_SCHEMA_V2
    out["subset"] = subset
    out["subset_path"] = write_rung_subset(gen_root, row, subset, rung=4096)
    draws = IC.draws_of_subset(subset)
    still, boundary = set(rung_still_at_cap), set(rung_boundary)
    lengths = {(IDS.index(i), t): (900 if (IDS.index(i), t) in boundary else 3000) for i, t in draws}
    src_manifest = completion.read_store_manifest(src_cd)
    rung_cd = build_store(gen_root, row, at_cap=still, cap=4096, draws=draws, rung_source=src_manifest, subset_doc=subset,
                          subset_path=str(out["subset_path"]), completion_tokens=lengths, **common)
    out["rung_cell_dir"] = rung_cd
    if rung8192:
        # the rung store's record carries THE decision over the MERGED reading (pass 1 with the rung's replacements)
        rec2, decision2, subset2 = file_decided_record(Path(gen_root), row, cap=4096, sha=common.get("sha", SHA7))
        out["rung_record"], out["rung_decision"] = rec2, decision2
        if decision2["exit"] == CL.ESCALATE and subset2 is not None:
            out["subset8192"] = subset2
            out["subset8192_path"] = write_rung_subset(gen_root, row, subset2, rung=8192)
            draws2 = IC.draws_of_subset(subset2)
            still2 = set(rung8192_still_at_cap)
            out["rung8192_cell_dir"] = build_store(gen_root, row, at_cap=still2, cap=8192, draws=draws2,
                                                   rung_source=completion.read_store_manifest(rung_cd), subset_doc=subset2,
                                                   subset_path=str(out["subset8192_path"]),
                                                   completion_tokens={(IDS.index(i), t): 5000 for i, t in draws2}, **common)
    return out


def build_all(gen_root: Path, *, rows: Sequence[Mapping[str, Any]] | None = None, skip: Iterable[tuple[str, str, str]] = (),
              **common: Any) -> dict[tuple[str, str, str], Path]:
    """Every LAUNCH row's pass-1 store (held rows are never built — a hard exclusion)."""
    rows = launch_rows(rows)
    skip = set(skip)
    out = {}
    for r in rows:
        key = (r["model_key"], r["view"], r["cell"])
        if key in skip:
            continue
        out[key] = build_store(gen_root, r, **common)
    return out


# --------------------------------------------------------------------------- the .eval logs (needs inspect_ai)
def write_inspect_log(cell_dir: Path, *, name: str, sample_idx: int, max_tokens: int | str = "store",
                      texts: Mapping[str, str] | None = None, value_edit: Callable[[str, dict], dict] | None = None,
                      created: str | None = None) -> Path:
    """A REAL Inspect ``.eval`` log of ONE sample index of the store's IFEval pass through ``inspect_ai.log.write_eval_log``:
    the stock scorer's name with its DICT-valued score (the five keys), the completion as the answer, the prompt as the
    input, the content channel only (thinking OFF), ``stop_reason`` from the outcome item."""
    from inspect_ai.log import (EvalConfig, EvalDataset, EvalLog, EvalMetric, EvalPlan, EvalPlanStep, EvalResults, EvalSample, EvalScore,
                                EvalSpec, EvalStats, write_eval_log)
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, GenerateConfig, ModelOutput, ModelUsage
    from inspect_ai.scorer import Score

    cell_dir = Path(cell_dir)
    manifest = completion.read_store_manifest(cell_dir)
    summary = completion.read_task_summary(cell_dir, TASK)
    items = [it for it in completion.read_task_outcomes(cell_dir, TASK)["items"] if int(it["sample_idx"]) == sample_idx]
    vmodel = str(manifest[C.MANIFEST_KEY_ROUTE]["virtual_model"])
    model = C.inspect_model_name(vmodel)
    gen: dict[str, Any] = {}
    if max_tokens == "store":
        max_tokens = int(summary["max_tokens"])
    if max_tokens is not None:
        gen["max_tokens"] = int(max_tokens)
    for f, cast in (("temperature", float), ("top_p", float), ("top_k", int)):
        if summary.get(f) is not None:
            gen[f] = cast(summary[f])
    penalty = (manifest.get(C.MANIFEST_KEY_SAMPLING) or {}).get(C.IDENTITY_KEY_PRESENCE_PENALTY)
    if penalty is not None:
        gen["presence_penalty"] = float(penalty)
    plan = EvalPlan(name="plan", steps=[EvalPlanStep(solver="generate", params={})], config=GenerateConfig(**gen))
    channels = store_texts(cell_dir)
    samples = []
    for it in items:
        iid = str(it["item_id"])
        content = (texts or {}).get(iid, channels.get((iid, sample_idx), ""))
        usage = ModelUsage(input_tokens=int(it["prompt_tokens"]), output_tokens=int(it["completion_tokens"]))
        output = ModelOutput(model=model, usage=usage,
                             choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=content or ""), stop_reason=it["stop_reason"])])
        value = {f: it[f] for f in ("prompt_level_strict", "inst_level_strict", "prompt_level_loose", "inst_level_loose", "num_instructions")}
        if value_edit:
            value = value_edit(iid, value)
        samples.append(EvalSample(id=iid, epoch=1, input=PROMPTS[IDS.index(iid)], target="", output=output,
                                  scores={SCORER_NAME: Score(value=value, answer=content, explanation=" ".join(it["instruction_id_list"]))},
                                  model_usage={model: usage}, metadata={"prompt": PROMPTS[IDS.index(iid)],
                                                                        "instruction_id_list": it["instruction_id_list"]}))
    if created is None:
        created = TS0.replace("Z", "+00:00")
    spec = EvalSpec(created=created, task=TASK_NAME, model=model, config=EvalConfig(), dataset=EvalDataset(name=TASK_NAME, samples=len(items)))
    # inspect_evals' OWN aggregate metrics of this trial's log (the stock scorer's `prompt_strict_acc` / `_stderr` and siblings), as
    # the real task files them on `results.scores` — the analysis reads them for comparability, never to decide
    metrics: dict[str, Any] = {}
    n_s = max(len(samples), 1)
    for field, stem in (("prompt_level_strict", "prompt_strict"), ("prompt_level_loose", "prompt_loose"), ("inst_level_strict", "inst_strict"),
                        ("inst_level_loose", "inst_loose")):
        if field.startswith("prompt"):
            vals = [1.0 if s_.scores[SCORER_NAME].value[field] else 0.0 for s_ in samples]
        else:
            vals = [s_.scores[SCORER_NAME].value[field] / max(int(s_.scores[SCORER_NAME].value["num_instructions"]), 1) for s_ in samples]
        mean = sum(vals) / n_s
        var = sum((v - mean) ** 2 for v in vals) / max(n_s - 1, 1)
        metrics[f"{stem}_acc"] = EvalMetric(name=f"{stem}_acc", value=mean)
        metrics[f"{stem}_stderr"] = EvalMetric(name=f"{stem}_stderr", value=(var / n_s) ** 0.5)
    metrics["final_acc"] = EvalMetric(name="final_acc", value=sum(m.value for k, m in metrics.items() if k.endswith("_acc")) / 4)
    metrics["final_stderr"] = EvalMetric(name="final_stderr", value=sum(m.value for k, m in metrics.items() if k.endswith("_stderr")) / 4)
    log = EvalLog(eval=spec, plan=plan, status="success", stats=EvalStats(), samples=samples,
                  results=EvalResults(total_samples=len(items), completed_samples=len(items),
                                      scores=[EvalScore(name=SCORER_NAME, scorer=SCORER_NAME, metrics=metrics)]))
    log_dir = C.inspect_log_dir(cell_dir, TASK, sample_idx)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{name}_{TASK_NAME}_{_hash(name + str(sample_idx), 8)}.eval"
    write_eval_log(log, path)
    return path


def write_inspect_logs(cell_dir: Path, *, name: str = "2026-01-01T00-00-00") -> list[Path]:
    """Both samples' logs (a rung over a draw subset may have no draw at an index — then no log for it)."""
    items = completion.read_task_outcomes(Path(cell_dir), TASK)["items"]
    return [write_inspect_log(cell_dir, name=name, sample_idx=t) for t in range(K) if any(int(it["sample_idx"]) == t for it in items)]
