"""SYNTHETIC stores for the fit-support-transfer analysis tests and for a
manual figure render. Placeholder sha, ``TESTHOST``, invented item
ids — never real recorded data. Every store is written through the REAL
writers of the A1 tree: ``completion.finish_task`` files the summary, the
``outcomes/<task>.json`` record (``completion.write_task_outcomes``) and the
done marker; the manifest, rows, evidence and usage ledger are written in the
shapes the driver / ledger write them (field names read off the contracts).

A helper module rather than a test (the ``attacks_static_common.py``
precedent): the figure smoke imports it from a venv without pytest."""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C

FULL_SHA = "1111111122222222333333334444444455555555"
SHA7 = FULL_SHA[:7]
#: the thinking-OFF comparator's OWN store pin (a separate
#: standup at a separate pin, out-suffix ``offcap-<sha7>``) — placeholder too
OFFCAP_FULL_SHA = "aaaaaaaabbbbbbbbccccccccddddddddeeeeeeee"
OFFCAP_SHA7 = OFFCAP_FULL_SHA[:7]
OFFCAP_OUT_SUFFIX_PREFIX = "offcap-"
TASK = C.GSM8K_PAIRED_TASK
MODEL = C.AUTHORISED_OOS_THINKING.model
NONE = C.ROUTE_NONE_CELL
PSR = C.CELL_PSRKL_SAFETY
N = C.GSM8K_PAIRED_N
IDS = [f"gsm8k_{i:08x}" for i in range(N)]
CAP_ON = C.OOS_THINKING_MAX_TOKENS
CAP_OFF = 512
#: the comparator's OFF cap: the ON cap under the OFF render — the contract's own
#: constant, pinned to ``OOS_THINKING_MAX_TOKENS`` by reference
CAP_OFF_OVERRIDE = C.OFF_CAP_OVERRIDE_MAX_TOKENS
#: the driver's evidence block on the comparator's ``none`` leg: the base
#: route's live refusal of the thinking-ON declaration key, under the
#: contract's ONE spelling (the key the driver writes)
BASE_ROUTE_PROBE_BLOCK = C.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED
BASE_ROUTE_PROBE_REFUSAL = {"status": 400, "code": "thinking_on_not_lens", "ok": True}
SERVED = {"reasoning_parser": None, "tool_calls": False, "max_model_len": 16384,
          "max_num_seqs": 22, "concurrency": 22, C.AGENT_LOOP_ENTRY_KEY: False,
          C.REDUCTION_MODE_FIELD: C.REDUCTION_MODE_PER_ROW}
PROMPT_BASE = 1000
#: the ON render's max(prompt_tokens_pre) — the CPU artifact's ``max`` and the
#: ON legs' live / ledger max
PROMPT_MAX = PROMPT_BASE + N - 1
#: the OFF render is LONGER on the same messages (two tokens on this family:
#: 35 vs 33 for one measured item), so an OFF leg's prompts, ledger and live
#: stamp read ``PROMPT_MAX + OFF_PROMPT_EXTRA`` — the artifact must never be
#: compared against them (plan S5.6: an ON-leg check)
OFF_PROMPT_EXTRA = 2
PROMPT_MAX_OFF = PROMPT_MAX + OFF_PROMPT_EXTRA
#: the paired-200 prompt lengths under the ON render, in item order
PROMPT_LENGTHS_ON = [PROMPT_BASE + j for j in range(N)]
#: ... and under the OFF render (the comparator's artifact and its OFF/8192 legs)
PROMPT_LENGTHS_OFF = [x + OFF_PROMPT_EXTRA for x in PROMPT_LENGTHS_ON]


def _vmodel(cell: str, arm: str) -> str:
    """The virtual model the PRODUCTION route wears on this leg, through the
    contracts' ONE derivation (``cell_spellings`` at the leg's door -> the
    registry's mapping): the base route bare on either arm (``none`` ->
    ``petri__base``), a lens cell bare on the OFF arm and under the DECLARED
    ``@thinking`` spelling on the ON arm (the out-of-support standup's cell
    spelling IS the declaration, contract §2.5.2; B0 filled the declared
    spelling's constants, so the producer stand-in and the analysis read the
    same derivation)."""
    declared = C.is_declared_configuration(cell, arm=arm, out_of_support=True)
    return C.cell_spellings(cell, declared=declared).vmodel


def _hash8(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def object_artifact_sha256(cell: str) -> str:
    """The synthetic fitted object's artifact digest for a lens cell: a 64-hex
    sha256 of the cell's BASE id (a variant serves its base's object), the
    same under every arm and cap — the identity the analysis compares across
    a cell's configurations (M3)."""
    import hashlib
    return hashlib.sha256(f"{cell.split('@')[0]}|object.pt".encode()).hexdigest()


def build_store(gen_root: Path, *, cell: str, arm: str, correct: Iterable[int],
                unscored: Iterable[int] = (), not_at_cap: Iterable[int] = (),
                stray: Iterable[int] = (), capped_scored: Iterable[int] = (),
                closed_think: Iterable[int] = (),
                sha: str = SHA7, ids: list[str] = IDS,
                write_outcomes: bool = True, write_rows: bool = True,
                evidence: bool = True, stamps: bool = True, regimes: Mapping[str, Any] | None = None,
                ledger: bool = True, ledger_edit: Callable[[list[dict[str, Any]]], None] | None = None,
                summary_edit: Callable[[dict[str, Any]], None] | None = None,
                header_edit: Callable[[dict[str, Any]], None] | None = None,
                manifest_edit: Callable[[dict[str, Any]], None] | None = None,
                served: Mapping[str, Any] | None = None, oos_marker: bool | None = None,
                out_suffix: str | None = None,
                limits: Mapping[str, int] | None = None, n_expected_full: int | None = None,
                cap: int | None = None, cap_override: bool = False, oos_keys: bool = False,
                tree_sha: str = FULL_SHA, completion_tokens: Mapping[int, int] | None = None,
                probe: bool | None = None,
                evidence_edit: Callable[[dict[str, Any]], None] | None = None) -> Path:
    """One leg's store. ``correct`` / ``unscored`` / ``not_at_cap`` / ``stray``
    are item INDICES into ``ids``: an unscored item's think block never closed
    (at the cap unless in ``not_at_cap``); a ``stray`` item carries a literal
    ``<think>`` inside its completion (``stray_open_tag``; scored unless also
    ``unscored``); a ``capped_scored`` item ran to the cap yet IS scored (its
    value per ``correct``) — on an ON leg its think block CLOSED before the cap
    (a truncated answer after completed reasoning), on an OFF leg a truncated
    answer with no think block; a ``closed_think`` item on an OFF leg emitted a
    CLOSED ``<think>…</think>`` block inside a completion no prompt opened — the
    ``record_only`` splitter's fact for it (``think_split.split_metadata``) is
    ``think_block_present True, think_block_unterminated False, stray_open_tag
    False``, and the item is scored unless also ``unscored`` (the review's A1: the
    shape the legacy anomaly predicate never counts). Returns the cell dir.
    ``regimes`` = the §3.4 triple to stamp
    into the summary (``{"salt_regime": ..., ...}`` plus optional ``n_*``
    counts, default 1). ``stamps`` = write the S3 certification flags and
    ``prompt_budget_live`` into evidence.json. ``limits`` / ``n_expected_full``
    go to ``completion.finish_task`` — a limited run (e.g. ``limits={"inspect":
    5}`` over 5 ids with ``n_expected_full=200``) gets the PARTIAL marker the
    real driver writes, never a done one. OFF prompts are ``OFF_PROMPT_EXTRA``
    tokens longer than ON prompts on the same items.

    The thinking-OFF comparator: ``cap`` is no longer
    hard-wired to the arm — a comparator store is ``arm="off", cap=CAP_OFF_OVERRIDE,
    cap_override=True``: it writes ``max_tokens 8192`` into the summary, ONE
    optional identity key (``tinycap.max_tokens_by_task = {task: 8192}``) and no
    OOS key, ``finish_reason`` / ``at_cap`` consistent with that cap, the OFF
    render's (+2) prompt lengths in the ledger and the live stamp, and — on the
    base route (``none``) — the driver's negative probe
    ``evidence.base_route_declaration_refused`` (``probe``; default: exactly
    when the store is a comparator ``none`` store; ``False`` omits it).
    ``oos_keys`` makes an ON store carry the REAL out-of-support store's three
    optional keys (``out_of_support_thinking``, ``max_tokens_by_task``,
    ``serve_out_of_support`` — the historical 5f5fd3d shape) instead of the
    marker alone. ``tree_sha`` is the producer pin stamped into the manifest
    and the outcome header. ``completion_tokens`` overrides the completion
    length of NOT-capped items by index (a capped item's length is the cap)."""
    on = arm == C.THINKING_ON
    closed_think = set(closed_think)
    correct, unscored, not_at_cap, stray = set(correct), set(unscored), set(not_at_cap), set(stray)
    capped_scored = set(capped_scored) - unscored
    cap = (CAP_ON if on else CAP_OFF) if cap is None else int(cap)
    length_of = dict(completion_tokens or {})
    prompt_extra = 0 if on else OFF_PROMPT_EXTRA
    served = dict(SERVED if served is None else served)
    out_root = Path(C.cell_out_dir(str(gen_root), cell, arm, model_slug=MODEL)) / (out_suffix or f"oos-{sha}")
    base = cell.split("@")[0]
    ihash = _hash8(f"{cell}|{arm}|ihash")
    chash = _hash8(f"{cell}|{arm}|{cap}|chash")
    key = f"{base.replace('__', '--')}__controls__t0__a0__{ihash}__{chash}"
    cell_dir = out_root / "cells" / key
    cell_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for j, item_id in enumerate(ids):
        is_unscored = j in unscored
        at_cap = (is_unscored and j not in not_at_cap) or j in capped_scored
        items.append({
            "item_id": item_id, "sample_idx": 0, "attempt": 1, "scored": not is_unscored,
            "value": None if is_unscored else (1.0 if j in correct else 0.0),
            "unparseable": None if is_unscored else False,
            "stop_reason": "max_tokens" if at_cap else "stop", "at_cap": at_cap,
            "think_block_present": on or j in closed_think, "think_block_unterminated": is_unscored,
            "stray_open_tag": j in stray,
            "content_empty_with_reasoning": False,
            "prompt_tokens": PROMPT_BASE + j + prompt_extra,
            "completion_tokens": cap if at_cap else length_of.get(j, 300 if on else 120),
            "seed": 1000 + j})
    prompt_max_leg = max(r["prompt_tokens"] for r in items)
    n_unscored = len(unscored)
    n_scored = len(ids) - n_unscored
    n_correct = sum(1 for r in items if r["scored"] and r["value"] == 1.0)
    n_at_cap = sum(1 for r in items if r["at_cap"])
    tokens = sorted(r["completion_tokens"] for r in items)
    summary: dict[str, Any] = {
        C.METRIC_ACC: n_correct / n_scored, C.METRIC_N: len(ids), "n_requests": len(ids),
        # the ledger-derived rejected count this tree's Inspect tier files
        "n_rejected": 0, "n_http_rejected_retried": 0,
        "samples_per_item": 1, "task": TASK, "item_set": C.GSM8K_ITEM_SET_PAIRED,
        "framework": C.INSPECT_TASK_VERSION, "engine": "vllm", "route": _vmodel(cell, arm),
        "thinking": arm, "render": C.THINKING_RENDER_NATIVE, "wall_clock_s": 100.0,
        "logs_reused": 0, "n_paired": len(ids), "reasoning_parser": None,
        "n_content_empty_with_reasoning": 0,
        "n_scored": n_scored, "n_unscored": n_unscored,
        "n_scored_samples": n_scored, "n_unscored_samples": n_unscored,
        "unparseable_rate": 0.0, "n_unparseable": 0, "n_answered": n_scored,
        "at_cap_rate": n_at_cap / len(ids), "n_at_cap": n_at_cap, "n_stop_reason": len(ids),
        "n_think_block_present": len(ids) if on else len(closed_think),
        "n_think_block_unterminated": n_unscored,
        "n_unterminated_not_at_cap": len(unscored & not_at_cap),
        "think_split_mode": C.THINK_SPLIT_IMPLICIT_OPEN if on else C.THINK_SPLIT_RECORD_ONLY,
        "unterminated_policy": C.UNTERMINATED_UNSCORED,
        "mean_completion_tokens": sum(tokens) / len(tokens),
        # the producer's nearest-rank convention (sorted[ceil(0.99 n) - 1]; sorted[-3] at n = 200)
        "p99_completion_tokens": C.nearest_rank_percentile(tokens, 0.99),
        "n_completion_tokens": len(ids),
        "max_tokens": cap, "temperature": 0.7, "top_p": 0.95, "top_k": 20,
        "grader_max_tokens": None,
    }
    if regimes:
        for f in C.EXPECTED_STAMP_TRIPLE_FIELDS:
            if f in regimes:
                summary[f] = regimes[f]
        for f in C.STAMP_REGIME_COUNT_FIELD.values():
            summary[f] = regimes.get(f, 1)
    if summary_edit:
        summary_edit(summary)
    header = {**{f: summary[f] for f in C.TASK_OUTCOMES_SUMMARY_FIELDS},
        "n_attempted_items": len(ids), "n_attempted_samples": len(items),
        "producer_tree_sha": tree_sha, "items": items}
    if header_edit:
        header_edit(header)

    # -- manifest (the driver's shape: tinycap identity block, route, producer, geometry)
    tinycap = {"tasks": [TASK], "pass": 1, "samples": 1, "served": served,
               "thinking": {"mode": arm, "by_task": {TASK: arm}},
               "versions": {"tinycap": C.TINYCAP_VERSION}, "limits": {}}
    if oos_marker is None:
        oos_marker = on
    if oos_marker:
        tinycap[C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] = True
        if oos_keys:
            # the historical out-of-support store's THREE optional keys (its identity
            # carries the executed cap and the contract-derived serve expectation)
            tinycap[C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] = {TASK: cap}
            tinycap[C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT] = C.expected_serve_out_of_support(
                lens=C.is_lens_cell(cell), arm=arm, out_of_support=True)
    if cap_override:
        # the comparator's ONE optional key — the executed cap, hashed
        tinycap[C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] = {TASK: cap}
    row = C.model_row(MODEL)
    lens_cell = C.is_lens_cell(cell)
    manifest = {
        "cell_id": cell, "harness_cell_id": _vmodel(cell, arm).removeprefix("petri__"),
        "virtual_model": _vmodel(cell, arm), "intervention_hash": ihash + "0" * 56,
        # the driver's model identity (the serving-parity comparison reads both)
        "model_id": row.hf, "model_revision": row.rev,
        "harness_config_hash": chash + "0" * 56,
        C.MANIFEST_KEY_PRODUCER: {"driver": "served_capability_run", "tree_sha": tree_sha},
        # the driver's provenance block (cell_controls' shape): the fitted object's artifact
        # digest on a lens cell — a function of the CELL alone, so every configuration of the
        # cell serves the same object (the analysis review's M3 compares it) — None on the base
        # route; the intervention hash differs per arm (the declared ON route), as the real one does
        C.MANIFEST_KEY_PROVENANCE: {"cell_id": cell, "class": "A" if lens_cell else "baseline",
                                    "artifact_sha256": object_artifact_sha256(cell) if lens_cell else None,
                                    "intervention_hash": ihash + "0" * 56, "model": row.hf, "model_revision": row.rev},
        C.MANIFEST_KEY_TINYCAP: tinycap, "resolved_config": {"tinycap": tinycap},
        C.MANIFEST_KEY_ROUTE: {"virtual_model": _vmodel(cell, arm), "tool_calls": False,
                               C.REDUCTION_MODE_FIELD: served[C.REDUCTION_MODE_FIELD]},
        C.MANIFEST_KEY_GEOMETRY: {"max_model_len": served["max_model_len"],
                                  "max_num_seqs": served["max_num_seqs"], C.GEOMETRY_PASS_FIELD: 1},
        C.MANIFEST_KEY_CONCURRENCY: served["concurrency"],
        C.MANIFEST_KEY_USAGE_LEDGER: str(out_root / C.LEDGER_FILENAME) if ledger else None,
        "host": "TESTHOST",
    }
    if manifest_edit:
        manifest_edit(manifest)
    (cell_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # -- rows.jsonl: the SCORED subset plus the two aggregate rows
    if write_rows:
        lines = []
        for r in items:
            if not r["scored"]:
                continue
            extra = {"sample_idx": 0, "seed": r["seed"], "answer": "7", "prompt_tokens": r["prompt_tokens"],
                     "completion_tokens": r["completion_tokens"], "paired_id": r["item_id"],
                     "stop_reason": r["stop_reason"], "unparseable": False, "unscored": False,
                     "attempt": 1, "max_tokens_used": cap, "think_block_present": on,
                     "think_block_unterminated": False}
            lines.append(json.dumps({"task": TASK, "item_id": r["item_id"], "metric": C.METRIC_ACC,
                                     "value": r["value"], "engine": "vllm", "extra_json": json.dumps(extra),
                                     "harness_config_hash": chash + "0" * 56,
                                     "intervention_hash": ihash + "0" * 56}))
        lines.append(json.dumps({"task": TASK, "item_id": C.AGG_ITEM_ID, "metric": C.METRIC_ACC_AGGREGATE,
                                 "value": summary[C.METRIC_ACC], "engine": "vllm", "extra_json": "{}"}))
        lines.append(json.dumps({"task": TASK, "item_id": C.AGG_ITEM_ID, "metric": C.METRIC_N,
                                 "value": float(len(ids)), "engine": "vllm", "extra_json": None}))
        (cell_dir / "rows.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # -- summary + outcomes + marker through the real writer: DONE for an
    #    unlimited full run, PARTIAL under `limits` / short of `n_expected_full`
    completion.finish_task(cell_dir, task=TASK, n_items=len(ids),
                           n_expected_full=(len(ids) if n_expected_full is None else n_expected_full),
                           limits=limits, n_rows=n_scored, n_requests=len(ids), n_rejected=0,
                           wall_clock_s=100.0, summary=summary,
                           outcomes=header if write_outcomes else None)

    # -- evidence.json (the driver's blocks under contracts.OOS_EVIDENCE_*; the
    #    S3 stamps when `stamps`)
    if evidence:
        lens = C.is_lens_cell(cell)
        hdr: dict[str, Any] = {
            C.OOS_EVIDENCE_HEADER_SOURCE: "proxy:/v1/tokenize", C.OOS_EVIDENCE_HEADER_VERIFIED: False,
            "header_len_by_arm": {"off": 7, "on": 5},
            C.OOS_EVIDENCE_HEADER_IDS_BY_ARM: {"off": list(C.QWEN35_HEADER_IDS_OFF),
                                               "on": list(C.QWEN35_HEADER_IDS_ON)},
            C.OOS_EVIDENCE_HEADER_MOVES: True, "out_of_support_thinking": True}
        bites: dict[str, Any] = (
            {C.OOS_EVIDENCE_BITES_APPLICABLE: True, C.OOS_EVIDENCE_BITES_VERIFIED: True,
             C.OOS_EVIDENCE_BITES_TWIN: C.TWIN_CELL_ID, "arm": "psr_kl_all", "n_probes": 5,
             C.OOS_EVIDENCE_BITES_N_DIFFERING: 3}
            if lens else {C.OOS_EVIDENCE_BITES_APPLICABLE: False, "reason": "base route"})
        ev: dict[str, Any] = {"cell_id": cell, "virtual_model": _vmodel(cell, arm), "engine": "vllm",
                              C.OOS_EVIDENCE_HEADER_BLOCK: hdr, C.OOS_EVIDENCE_BITES_BLOCK: bites,
                              "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(0))}
        if stamps:
            hdr.update({C.OOS_EVIDENCE_HEADER_CERTIFIED: True, C.OOS_EVIDENCE_HEADER_ATTESTED_AGREES: True,
                        C.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED: True,
                        C.OOS_EVIDENCE_ATTESTED_SWITCH: True, C.OOS_EVIDENCE_ATTESTED_MOVES: True})
            if lens:
                bites[C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] = arm
            # the live pass measures THIS leg's render: the OFF legs read the longer OFF prompts
            ev[C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK] = {
                C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_MAX: prompt_max_leg,
                C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_P99: prompt_max_leg - 2,
                "arm": arm, "cap": cap}
        if probe is None:
            probe = cap_override and not lens
        if probe:
            ev[BASE_ROUTE_PROBE_BLOCK] = dict(BASE_ROUTE_PROBE_REFUSAL)
        if evidence_edit:
            evidence_edit(ev)
        (cell_dir / "evidence.json").write_text(json.dumps(ev, indent=1), encoding="utf-8")
        (out_root / "evidence.json").write_text(json.dumps(ev, indent=1), encoding="utf-8")

    # -- the usage ledger at the OUT-ROOT (one row per attempted call + foreign-task rows)
    if ledger:
        rows = []
        for r in items:
            rows.append({"ts": "2026-01-01T00:00:00Z", "task": TASK, "sample": f"{r['item_id']}/0",
                         "vmodel": _vmodel(cell, arm), "thinking": on, "prompt_tokens_pre": r["prompt_tokens"],
                         "max_tokens_requested": cap, "has_tools": False, "n_tools": 0, "status": 200,
                         "error_code": None, "rejected": False, "usage_prompt_tokens": r["prompt_tokens"],
                         "usage_completion_tokens": r["completion_tokens"],
                         "finish_reason": "length" if r["at_cap"] else "stop", "latency_s": 1.0,
                         "tokenize_mismatch": False, "error_class": C.LEDGER_ERROR_CLASS_NONE,
                         # the per-attempt identity columns — a
                         # legacy-shaped synthetic row carries them null
                         "request_id": None, "body_sha256": None})
        rows.append({**rows[0], "task": "evalplus/humaneval_plus", "sample": "0#steered/0",
                     "max_tokens_requested": 1, "thinking": False})
        if ledger_edit:
            ledger_edit(rows)
        assert all(set(r) == set(C.LEDGER_ROW_FIELDS) for r in rows)
        (out_root / C.LEDGER_FILENAME).write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                                                  encoding="utf-8")
    return cell_dir


def build_2x2(gen_root: Path, *, none_off: Mapping[str, Any], psr_off: Mapping[str, Any],
              none_on: Mapping[str, Any], psr_on: Mapping[str, Any], **common: Any) -> dict[str, Path]:
    """The four stores; each mapping = ``build_store`` kwargs for that leg
    (``correct`` required; ``unscored`` etc. optional)."""
    out = {}
    for cell, arm, kw in ((NONE, C.THINKING_OFF, none_off), (PSR, C.THINKING_OFF, psr_off),
                          (NONE, C.THINKING_ON, none_on), (PSR, C.THINKING_ON, psr_on)):
        out[f"{cell}|{arm}"] = build_store(gen_root, cell=cell, arm=arm, **{**common, **kw})
    return out


def build_2x3(gen_root: Path, *, none_off: Mapping[str, Any], psr_off: Mapping[str, Any],
              none_on: Mapping[str, Any], psr_on: Mapping[str, Any],
              none_off8192: Mapping[str, Any], psr_off8192: Mapping[str, Any],
              offcap_gen_root: Path | None = None, offcap_sha: str = OFFCAP_SHA7,
              **common: Any) -> dict[str, Path]:
    """The four 2x2 stores under ``oos-<sha>`` plus the two thinking-OFF
    comparator stores under ``offcap-<offcap_sha>`` (their own pin and,
    optionally, their own gen root — the comparator pull lives in its own
    directory). The comparator stores are ``arm off``,
    ``cap CAP_OFF_OVERRIDE``, ``cap_override True``, producer ``OFFCAP_FULL_SHA``."""
    out = build_2x2(gen_root, none_off=none_off, psr_off=psr_off, none_on=none_on, psr_on=psr_on, **common)
    root = gen_root if offcap_gen_root is None else offcap_gen_root
    for cell, kw in ((NONE, none_off8192), (PSR, psr_off8192)):
        out[f"{cell}|{C.THINKING_OFF}|cap{CAP_OFF_OVERRIDE}"] = build_store(
            root, cell=cell, arm=C.THINKING_OFF,
            **{"cap": CAP_OFF_OVERRIDE, "cap_override": True, "oos_marker": False, "sha": offcap_sha,
               "out_suffix": f"{OFFCAP_OUT_SUFFIX_PREFIX}{offcap_sha}", "tree_sha": OFFCAP_FULL_SHA,
               **common, **kw})
    return out


# -- the fixtures the tests share -------------------------------------------------

#: OFF legs: none 120/200, psr 150/200 with 43 psr-only and 13 none-only correct
#: items -> effect_off +15 pp, seed-0 95 % CI [+8.0, +22.0] pp (B = 10,000).
OFF_NONE_CORRECT = set(range(107)) | set(range(150, 163))
OFF_PSR_CORRECT = set(range(150))
LAST4 = {196, 197, 198, 199}


def off_legs() -> dict[str, Any]:
    return {"none_off": {"correct": OFF_NONE_CORRECT}, "psr_off": {"correct": OFF_PSR_CORRECT}}


#: the comparator fixture, in miniature: under OFF/512
#: 10 ``none`` items and 6 ``psr`` items ran to the cap (8 / 4 of them incorrect);
#: under OFF/8192 the same items stop short of the cap and 6 / 3 of the formerly
#: capped-incorrect ones are correct -> cap_relief_none = +6/200, cap_relief_psr = +3/200
OFF_NONE_CAPPED = {0, 1} | set(range(120, 128))
OFF_PSR_CAPPED = {2, 3} | set(range(150, 154))
OFF8192_NONE_CORRECT = OFF_NONE_CORRECT | set(range(120, 126))
OFF8192_PSR_CORRECT = OFF_PSR_CORRECT | set(range(150, 153))
#: the formerly capped items' OFF/8192 completion lengths (index -> tokens), per cell
OFF8192_LENGTHS_NONE = {j: 900 + 10 * j for j in OFF_NONE_CAPPED}
OFF8192_LENGTHS_PSR = {j: 700 + 10 * j for j in OFF_PSR_CAPPED}


def off_legs_capped() -> dict[str, Any]:
    """The OFF/512 legs WITH the capped items marked (the comparator fixture's
    first draws); the accuracies are :func:`off_legs`' (120 / 150 correct)."""
    return {"none_off": {"correct": OFF_NONE_CORRECT, "capped_scored": OFF_NONE_CAPPED},
            "psr_off": {"correct": OFF_PSR_CORRECT, "capped_scored": OFF_PSR_CAPPED}}


def offcap_legs() -> dict[str, Any]:
    """The two OFF/8192 comparator legs of the fixture: no item at the cap."""
    return {"none_off8192": {"correct": OFF8192_NONE_CORRECT, "completion_tokens": OFF8192_LENGTHS_NONE},
            "psr_off8192": {"correct": OFF8192_PSR_CORRECT, "completion_tokens": OFF8192_LENGTHS_PSR}}


def _prompt_budget_doc(*, arm: str, cap: int, tree_sha: str | None, where: str) -> dict[str, Any]:
    """A CPU prompt-budget artifact in the CONTRACT's shape — exactly
    ``contracts.OOS_PROMPT_BUDGET_FIELDS`` over the fixture's prompt lengths under
    ``arm``'s render, validated through the reader's own rule so the analysis
    reads it the way it reads the real writer's document. Same template digest
    under either arm: the digest hashes the template, not the render."""
    row = C.model_row(MODEL)
    lengths = list(PROMPT_LENGTHS_ON if arm == C.THINKING_ON else PROMPT_LENGTHS_OFF)
    doc = {
        "n": len(lengths), "max": max(lengths),
        "p99": C.nearest_rank_percentile(lengths, 0.99), "p50": C.nearest_rank_percentile(lengths, 0.50),
        "mean": round(sum(lengths) / len(lengths), 2),
        "arm": arm, "model": row.hf, "model_slug": row.slug, "revision": row.rev,
        "chat_template_hash": "0123456789abcdef",
        "cap": cap, "cap_task": TASK, "max_plus_cap": max(lengths) + cap,
        "lengths": lengths,
        "render": {"add_generation_prompt": True,
                   C.CHAT_TEMPLATE_KWARGS_KEY: dict(C.chat_template_kwargs(arm)[C.CHAT_TEMPLATE_KWARGS_KEY])},
        "messages_source": C.OOS_PROMPT_BUDGET_MESSAGES_SOURCE,
        "tree_sha": tree_sha,
        "produced_by": {"script": C.OOS_PROMPT_BUDGET_SCRIPT, "tokenizer_loader": "sisteer.models.load_tokenizer",
                        "transformers": None},
    }
    return C.validate_oos_prompt_budget(doc, where=where)


def prompt_budget_artifact() -> dict[str, Any]:
    """The CPU prompt-budget artifact (plan S4.6, the ON arm's): the fixture's ON
    prompt lengths at the ON cap (:func:`_prompt_budget_doc`)."""
    return _prompt_budget_doc(arm=C.THINKING_ON, cap=CAP_ON, tree_sha=FULL_SHA,
                              where="tests/oos_synthetic_store.prompt_budget_artifact")


def prompt_budget_artifact_off(*, cap: int = CAP_OFF_OVERRIDE, tree_sha: str | None = OFFCAP_FULL_SHA
                               ) -> dict[str, Any]:
    """The comparator's CPU prompt-budget artifact: the OFF
    render's lengths (two tokens longer than the ON render's on this family),
    ``arm off`` and the OVERRIDE cap (8192) — the artifact ``prompt_budget_fit``
    holds against the OFF/8192 legs' cap; ``cap=512`` writes the table-cap
    artifact the analysis must refuse (:func:`_prompt_budget_doc`)."""
    return _prompt_budget_doc(arm=C.THINKING_OFF, cap=cap, tree_sha=tree_sha,
                              where="tests/oos_synthetic_store.prompt_budget_artifact_off")
