#!/usr/bin/env python3
"""IFEval canonical battery — the per-leg VALIDATORS of the IFEval analysis (the GSM8K validators' shape,
``scripts/tinycap_gsm8k_validators.py``), every one PARAMETERIZED by the matrix row (model, view, cell, cap, pin, the
served-positions expectation) so one function reads a pass-1 leg, a 4,096 rung and an 8,192 rung alike.

The producer's spellings BY IMPORT, never re-typed: the instrument's rule / threshold / provenance and the record fields
from ``inspect_rung`` (the IFEval instrument), the draw-subset contract from ``ifeval_contracts``, the ledger's at-cap
finish reason from ``cap_ladder``, the marker / outcome names from ``completion``.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from sisteer.capeval.tiers import cap_ladder as CL
from sisteer.capeval.tiers import inspect_rung as IR
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import ifeval_contracts as IC

TASK = IC.IFEVAL_TASK
TASK_NAME = IC.IFEVAL_TASK_NAME
K = IC.IFEVAL_SAMPLES_PER_ITEM
N = IC.IFEVAL_N
CAPS = IC.IFEVAL_CAP_LADDER
CAP_BASE = IC.IFEVAL_PASS1_CAP
CEILING = IC.IFEVAL_CAP_CEILING
INSTRUMENT = IR.IFEVAL_INSTRUMENT
TRIGGER_THRESHOLD = INSTRUMENT.threshold
TRIGGER_RULE = INSTRUMENT.rule
TRIGGER_PROVENANCE = IC.IFEVAL_TRIGGER_PROVENANCE
LEDGER_FINISH_AT_CAP = CL.LEDGER_FINISH_AT_CAP
CLIENT_TIMEOUT_MIN_S = float(C.PROXY_UPSTREAM_TIMEOUT_S)
#: the per-REQUEST client timeout the target's OpenAI client was built with, as the PRODUCER records it: in the task
#: SUMMARY (``contracts.SUMMARY_PAYLOAD_FIELDS``; the runner's ``inspect_served.CLIENT_TIMEOUT_FIELD``, written by
#: ``client_record`` off the provider), never in the manifest — a real-producer regression found I1 reading
#: ``manifest.client_timeout_s``, a key the driver never writes, so every real store would have refused here while the
#: synthetic fixture (which had put the key on the manifest) passed
CLIENT_TIMEOUT_FIELD = "client_timeout_s"
assert CLIENT_TIMEOUT_FIELD in C.SUMMARY_PAYLOAD_FIELDS


class AnalysisError(RuntimeError):
    """A store, a matrix or a report the analysis cannot read as a production result."""


def leg_key(model_key: str, view: str, cell: str, cap: int) -> str:
    return f"{model_key}/{view}/{cell}@{int(cap)}"


def cell_key(model_key: str, view: str, cell: str) -> str:
    return f"{model_key}/{view}/{cell}"


def same_store_tail(path: Any, tail: str) -> bool:
    """A relocation-aware root match (a pull relocates every root): the RECORDED root (absolute, wherever the planner or
    the driver ran — the volume's ``/vol/.../controls/<TAG>/<slug>/vendor_preset`` or a local pull's) names the same store
    as the reader's WAVE-RELATIVE layout (``ifeval_paths(...).out_root``) iff its trailing components ARE that layout.
    ``tail`` must be that relative layout: an absolute tail is the caller's mistake (C2 — the analysis once passed its own
    absolute local root here, so no recorded root could ever match after a pull) and is refused by name, never matched."""
    if PurePosixPath(str(tail)).is_absolute():
        raise ValueError(f"same_store_tail: the tail must be a wave-relative store layout (ifeval_paths().out_root), not the "
                         f"absolute path {tail!r}")
    want = PurePosixPath(str(tail).strip("/")).parts
    have = PurePosixPath(str(path).rstrip("/")).parts
    return bool(want) and len(have) >= len(want) and tuple(have[-len(want):]) == tuple(want)


def read_json_if_file(p: Path) -> Any:
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def iter_jsonl(p: Path) -> list[dict[str, Any]]:
    out = []
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- the attempt scope (C4)
def accepted_attempt_id(manifest: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """``(accepted attempt id, None)`` — the ONE generation attempt whose ledger rows and request-evidence records are the
    store's evidence (the attempt contract: the driver records every attempt under ``resume_attempts`` and names the
    accepted one; the IFEval line always runs under ``--attempt-id``, F8) — or ``(None, why)`` when the manifest names none
    or names one it did not record. Every historical attempt's rows stay on disk (the driver appends evidence across
    attempts and certifies the accepted one), so a reader that counts EVERY record against the draw count refuses a
    successful retry after an interrupted attempt — C4; the counts are scoped here and the other attempts' rows reported."""
    accepted = manifest.get(C.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT)
    recorded = sorted({str(a[C.ATTEMPT_ID_FIELD]) for a in manifest.get(C.MANIFEST_KEY_RESUME_ATTEMPTS) or []
                       if isinstance(a, Mapping) and a.get(C.ATTEMPT_ID_FIELD) is not None})
    if not isinstance(accepted, str) or not accepted:
        return None, f"the manifest names no {C.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT} — the evidence cannot be scoped to an attempt"
    if accepted not in recorded:
        return None, f"{C.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT} {accepted!r} is not a recorded attempt ({recorded})"
    return accepted, None


def evidence_index(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]], str | None]:
    """``contracts.evidence_by_request_id`` over the store's request-evidence records (the identity join a ledger row
    without its own attempt stamp is attributed through), or ``({}, why)`` when the join is not one-to-one."""
    try:
        return C.evidence_by_request_id(records), None
    except C.AttemptScopeError as e:
        return {}, str(e)


def rows_by_attempt(rows: Sequence[Mapping[str, Any]], evidence_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Diagnostic: how many of ``rows`` each attempt owns (``contracts.ledger_row_attempt``: the row's own stamp, else its
    request-evidence record's); a row nobody can attribute is counted under ``unattributed``, never dropped."""
    out: Counter[str] = Counter()
    for r in rows:
        try:
            out[C.ledger_row_attempt(r, evidence_by_id)] += 1
        except C.AttemptScopeError:
            out["unattributed"] += 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- the matrix
def load_matrix(source: Path | str | Sequence[Any]) -> dict[str, Any]:
    """The frozen matrix's rows (a path, or the rows themselves), validated: schema, index order, the IFEval grammar."""
    if isinstance(source, (str, Path)):
        doc = json.loads(Path(source).read_text(encoding="utf-8"))
        if doc.get("schema") != IC.MATRIX_SCHEMA:
            raise AnalysisError(f"{source}: not a {IC.MATRIX_SCHEMA} matrix")
        rows = doc["rows"]
    else:
        rows = list(source)
    for i, r in enumerate(rows):
        p = row_problem(r, i)
        if p:
            raise AnalysisError(f"matrix row {i}: {p}")
    return {"rows": rows, "counts": matrix_counts(rows)}


def row_problem(row: Mapping[str, Any], i: int) -> str | None:
    need = ("model_key", "view", "cell", "tag", "slug", "cell_slug", "fitted", "object_sha256", "route_kind", "header_located",
            "served_positions_expected", "declared", "thinking", "sampling_mode", "sampling", "samples_per_item", "expected_stamps",
            "caps", "out_root", "rung_out_roots", "subset_paths", "cap_ladder_paths", "status", "held_reason", "hf", "rev", "family",
            "geometry_eval_by_cap", "trigger_threshold", "trigger_provenance")
    missing = [k for k in need if k not in row]
    if missing:
        return f"missing {missing}"
    if row.get("index") != i:
        return f"index {row.get('index')!r} != position {i}"
    if row["status"] not in IC.ROW_STATUSES:
        return f"status {row['status']!r}"
    if row["status"] == IC.STATUS_HELD and not row["held_reason"]:
        return "held without a reason"
    deferred = row["status"] == IC.STATUS_DEFERRED
    if deferred and (row["object_sha256"] is not None or not row["fitted"] or not isinstance(row.get("deferred"), Mapping)
                     or not row["deferred"].get("record")):
        return "a deferred slot is a fitted arm with NO object sha and a per-model record (the follow-up, D23)"
    if row["declared"] is not False or row["thinking"] != C.THINKING_OFF or row["sampling_mode"] != IC.SAMPLING_VENDOR_PRESET:
        return "not an OFF / undeclared / vendor_preset row"
    if row["samples_per_item"] != K or list(row["caps"]) != list(CAPS):
        return f"samples_per_item {row['samples_per_item']} / caps {row['caps']}"
    if not deferred and (row["fitted"] != (row["object_sha256"] is not None)
                         or (row["fitted"] and not C.is_sha256_hex(row["object_sha256"]))):
        return "fitted / object_sha256 disagree"
    if row["route_kind"] != IC.ifeval_route_kind(row["cell"]) or row["route_kind"] not in IC.ROUTE_KINDS:
        return f"route_kind {row['route_kind']!r}"
    if row["view"] not in IC.BANK_VIEWS or row["model_key"] not in IC.IFEVAL_MODELS:
        return f"view {row['view']!r} / model {row['model_key']!r}"
    paths = IC.ifeval_paths(row["model_key"], row["view"], row["cell"], CAP_BASE)
    if row["out_root"] != paths.out_root or row["tag"] != paths.tag or row["cell_slug"] != paths.cell_slug:
        return "paths are not ifeval_paths'"
    want = list(IC.ifeval_expected_stamps(fitted=row["fitted"], header_located=row["header_located"]))
    if list(row["expected_stamps"]) != want:
        return f"expected_stamps {row['expected_stamps']} != the contract's {want}"
    if row["trigger_threshold"] != TRIGGER_THRESHOLD or row["trigger_provenance"] != TRIGGER_PROVENANCE:
        return "trigger threshold / provenance are not the instrument's"
    return None


def matrix_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    launch = [r for r in rows if r["status"] == IC.STATUS_LAUNCH]
    return {"rows": len(rows), "launched": len(launch),
            "held": sum(1 for r in rows if r["status"] == IC.STATUS_HELD),
            "deferred": sum(1 for r in rows if r["status"] == IC.STATUS_DEFERRED),
            "fitted_launched": sum(1 for r in launch if r["fitted"]),
            "contrasts_designed": sum(1 for r in launch if r["cell"] != IC.CELL_NONE)}


# --------------------------------------------------------------------------- the store
def read_store(out_root: Path) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    root = Path(out_root)
    if not root.is_dir():
        return None, None, f"no out-root at {root}"
    cells_root = root / "cells"
    cells = sorted(p for p in cells_root.glob("*") if (p / "manifest.json").is_file()) if cells_root.is_dir() else []
    if not cells:
        return None, None, f"no cells/*/manifest.json under {root}"
    if len(cells) > 1:
        return None, None, f"{len(cells)} stores under {root} — two identities in one out-root"
    manifest = completion.read_store_manifest(cells[0])
    if manifest is None:
        return cells[0], None, f"{cells[0] / 'manifest.json'} is not a JSON object"
    return cells[0], manifest, None


def identity_problem(manifest: Mapping[str, Any], *, row: Mapping[str, Any], cap: int, srow: C.SamplingRow) -> str | None:
    """The hashed identity + the manifest's sampling / seeds / route blocks against the ROW: identity version 2, the executed
    cap == ``cap``, the family's OFF arm for the task (``not_available`` on a no-switch family), k = 2, the vendor row's numbers (and its optional terms — Qwen's
    presence_penalty 1.5), the IFEval seed rule, ``sampling.mode vendor_preset`` with a matching ``sampling_ref``, the model
    pins, the object digest (a stated null on bare / prompted), the bank view."""
    tinycap = (manifest.get("resolved_config") or {}).get(C.MANIFEST_KEY_TINYCAP) or {}
    if tinycap.get(C.IDENTITY_KEY_IDENTITY_VERSION) != C.TINYCAP_IDENTITY_VERSION_2:
        return f"identity version {tinycap.get(C.IDENTITY_KEY_IDENTITY_VERSION)!r} != 2"
    caps = tinycap.get(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK) or {}
    if caps.get(TASK) != int(cap):
        return f"max_tokens_by_task[{TASK}] {caps.get(TASK)!r} != {cap}"
    # The OFF arm a no-switch family (llama) runs under is hashed as `not_available`, so the
    # expectation is FAMILY-resolved from the row — the frozen matrix's `thinking: off` is the MODE, never the arm
    arm = IC.ifeval_effective_arm(row["family"])
    if (tinycap.get(C.IDENTITY_KEY_THINKING_BY_TASK) or {}).get(TASK) != arm:
        return f"thinking_by_task[{TASK}] {(tinycap.get(C.IDENTITY_KEY_THINKING_BY_TASK) or {}).get(TASK)!r} is not the family's OFF arm {arm!r}"
    if (tinycap.get(C.IDENTITY_KEY_SAMPLES_BY_TASK) or {}).get(TASK) != K:
        return f"samples_by_task[{TASK}] {(tinycap.get(C.IDENTITY_KEY_SAMPLES_BY_TASK) or {}).get(TASK)!r} != {K}"
    terms = [(C.IDENTITY_KEY_TEMPERATURE, srow.temperature), (C.IDENTITY_KEY_TOP_P, srow.top_p), (C.IDENTITY_KEY_TOP_K, srow.top_k),
             *((f, getattr(srow, f, None)) for f in C.SAMPLING_OPTIONAL_IDENTITY_FIELDS)]
    for f, want in terms:       # a term the row leaves unset must be absent from the identity, a set one equal
        got = tinycap.get(f)
        if (want is None) != (got is None) or (want is not None and float(got) != float(want)):
            return f"identity {f} {got!r} != the vendor row's {want!r}"
    rules = tinycap.get(C.IDENTITY_KEY_SEED_RULE) or {}
    if rules.get(TASK) != IC.SEED_RULE_IFEVAL:
        return f"seed_rule[{TASK}] is not SEED_RULE_IFEVAL"
    block = manifest.get(C.MANIFEST_KEY_SAMPLING)
    if not isinstance(block, Mapping) or block.get("mode") != IC.SAMPLING_VENDOR_PRESET:
        return f"manifest.sampling.mode {block.get('mode') if isinstance(block, Mapping) else None!r} != vendor_preset"
    if block.get("model_key") != row["model_key"]:
        return f"manifest.sampling.model_key {block.get('model_key')!r} != {row['model_key']}"
    if manifest.get("model_id") != row["hf"] or manifest.get("model_revision") != row["rev"]:
        return f"model pin {manifest.get('model_id')}@{str(manifest.get('model_revision'))[:8]} != the row's {row['hf']}@{row['rev'][:8]}"
    route = manifest.get(C.MANIFEST_KEY_ROUTE) or {}
    pin = row["object_sha256"]
    if route.get(C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256) != pin:
        return f"route.object_artifact_sha256 {str(route.get(C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256))[:8]!r} != the row's pin {str(pin)[:8]!r}"
    if route.get("view", route.get("bank_view")) != row["view"]:
        return f"route.view {route.get('view')!r} != the row's {row['view']}"
    # the OFF-cap setter's STATEMENT is recorded UNHASHED at the manifest TOP LEVEL (`served_capability_run`
    # writes `manifest[MANIFEST_KEY_OFF_CAP_BY_TASK]`; the executed cap hashes through `max_tokens_by_task` above) —
    # the rung wave's reader item (V36 (i) / V41): this read it under `resolved_config.tinycap`, where the producer never
    # writes it, so every rung leg read "does not state the rung cap" while its hashed cap was right
    if cap != CAP_BASE and off_cap_statement(manifest).get(TASK_NAME) != int(cap):
        return (f"{IC.MANIFEST_KEY_OFF_CAP_BY_TASK}[{TASK_NAME}] {off_cap_statement(manifest).get(TASK_NAME)!r} does not state the "
                f"rung cap {cap} (the manifest's top-level statement of {IC.FLAG_OFF_CAP_BY_TASK})")
    return None


def off_cap_statement(manifest: Mapping[str, Any]) -> dict[str, int]:
    """The manifest's top-level ``off_cap_by_task`` statement (``{task name: cap}``; empty at pass 1, where no setter ran)."""
    block = manifest.get(IC.MANIFEST_KEY_OFF_CAP_BY_TASK)
    return {str(k): int(v) for k, v in block.items()} if isinstance(block, Mapping) else {}


def rows_identity_check(rows: Sequence[Mapping[str, Any]], *, row: Mapping[str, Any], manifest: Mapping[str, Any],
                        cap: int) -> dict[str, Any]:
    """Gate I2's per-row half (P42): every emitted per-item row carries the six guard keys with the cell's values, and
    ``sampling_ref`` equal to the manifest block's digest; the per-sample metric names are ``acc@<idx>``."""
    sha = manifest.get(C.MANIFEST_KEY_PRODUCER, {}).get("tree_sha")
    want = {"object_artifact_sha256": row["object_sha256"], "route_kind": row["route_kind"], "view": row["view"],
            "producer_tree_sha": sha, "thinking": IC.ifeval_effective_arm(row["family"]), "sampling_mode": IC.SAMPLING_VENDOR_PRESET,
            C.SAMPLING_REF_FIELD: C.sampling_ref(manifest.get(C.MANIFEST_KEY_SAMPLING) or {})}
    failures: list[str] = []
    n_sample_rows = n_mean_rows = 0
    for r in rows:
        if r.get("item_id") == C.AGG_ITEM_ID:
            continue
        extra = r.get("extra_json")
        extra = json.loads(extra) if isinstance(extra, str) else (extra or {})
        metric = str(r.get("metric"))
        if metric == C.METRIC_ACC:
            n_mean_rows += 1
        elif metric.startswith(f"{C.METRIC_ACC}{C.SAMPLE_METRIC_SEP}"):
            n_sample_rows += 1
        else:
            failures.append(f"row {r.get('item_id')}: metric {metric!r}")
            continue
        for k, v in want.items():
            if k not in extra:
                failures.append(f"row {r.get('item_id')} {metric}: no {k!r} in extra_json")
                break
            if extra[k] != v:
                failures.append(f"row {r.get('item_id')} {metric}: {k} {extra[k]!r} != {v!r}")
                break
        if extra.get("max_tokens") not in (None, int(cap)):
            failures.append(f"row {r.get('item_id')}: max_tokens {extra.get('max_tokens')!r} != {cap}")
        if len(failures) > 20:
            break
    return {"n_sample_rows": n_sample_rows, "n_mean_rows": n_mean_rows, "failures": failures[:20], "passed": not failures}


def ledger_check(path: Path, *, header: Mapping[str, Any], summary: Mapping[str, Any], cap: int, max_model_len: Any,
                 manifest: Mapping[str, Any], evidence_by_id: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Gate I1 per leg — the reconciliation that runs BEFORE any share is read: the forwarded task rows OF THE ACCEPTED
    ATTEMPT (:func:`accepted_attempt_id`; rows attributed by ``contracts.rows_of_attempts`` — the row's own stamp, else the
    request-evidence join) are exactly the attempted draws (one per (item, sample)), every ``max_tokens_requested == cap``,
    ``finish_reason == length`` count == the summary's ``n_at_cap`` == the outcome record's at-cap draws (ledger-at-cap ==
    recorded-at-cap), no rejected / errored row, ``prompt_tokens_pre + cap <= max_model_len``, the client timeout >= the
    proxy's 7,200 s. Every other attempt's rows are counted per attempt (``rows_by_attempt``), never held against the draws.
    Also the accepted attempt's completion-token quantiles and its AGGREGATE decode rate (``ifeval_contracts.
    aggregate_decode_tok_s``: completions / the generation wall — the per-request ``sum / sum`` is labelled beside it, C5)."""
    failures: list[str] = []
    if not Path(path).is_file():
        return {"passed": False, "failures": [f"no {IC.LEDGER_FILENAME} at {path}"], "n_length": None}
    rows = iter_jsonl(path)
    all_task_rows = [r for r in rows if r.get("task") == TASK]
    by_attempt = rows_by_attempt(all_task_rows, evidence_by_id or {})
    accepted, why = accepted_attempt_id(manifest)
    if why:
        return {"passed": False, "failures": [why], "n_length": None, "accepted_attempt": None, "rows_by_attempt": by_attempt,
                "n_task_rows_all_attempts": len(all_task_rows)}
    try:
        task_rows = C.rows_of_attempts(all_task_rows, [accepted], evidence_by_id=evidence_by_id or {})
    except C.AttemptScopeError as e:
        return {"passed": False, "failures": [str(e)], "n_length": None, "accepted_attempt": accepted, "rows_by_attempt": by_attempt,
                "n_task_rows_all_attempts": len(all_task_rows)}
    forwarded = [r for r in task_rows if r.get("rejected") is not True]
    draws = {(str(it["item_id"]), int(it["sample_idx"])) for it in header["items"]}
    samples = [str(r.get("sample")) for r in forwarded]
    if len(samples) != len(set(samples)):
        failures.append("the ledger forwards a (item, sample) twice")
    seen = set()
    for s in samples:
        iid, _, idx = s.rpartition("/")
        seen.add((iid, int(idx) if idx.isdigit() else -1))
    if seen != draws:
        failures.append(f"ledger draws {len(seen)} != attempted draws {len(draws)} (missing {sorted(draws - seen)[:3]}, extra {sorted(seen - draws)[:3]})")
    n_rejected = sum(1 for r in task_rows if r.get("rejected") is True)
    if n_rejected:
        failures.append(f"{n_rejected} rejected task rows")
    bad_cap = [r for r in forwarded if r.get("max_tokens_requested") != int(cap)]
    if bad_cap:
        failures.append(f"{len(bad_cap)} rows with max_tokens_requested != {cap}")
    bad_err = [r for r in forwarded if r.get("error_class") not in (C.LEDGER_ERROR_CLASS_NONE, None) or r.get("error_code")]
    if bad_err:
        failures.append(f"{len(bad_err)} rows with an error class / code")
    n_length = sum(1 for r in forwarded if r.get("finish_reason") == LEDGER_FINISH_AT_CAP)
    n_at_cap_summary = summary.get("n_at_cap")
    n_at_cap_recorded = sum(1 for it in header["items"] if it.get("at_cap"))
    if n_at_cap_summary is None:
        failures.append(f"summary n_at_cap is null while the ledger counts {n_length} finish_reason == length")
    elif n_length != n_at_cap_summary:
        failures.append(f"ledger finish_reason == length {n_length} != summary n_at_cap {n_at_cap_summary}")
    if n_length != n_at_cap_recorded:
        failures.append(f"ledger at-cap {n_length} != recorded at-cap {n_at_cap_recorded} (the outcome record)")
    if isinstance(max_model_len, int):
        over = [r for r in forwarded if isinstance(r.get("prompt_tokens_pre"), int) and r["prompt_tokens_pre"] + int(cap) > max_model_len]
        if over:
            failures.append(f"{len(over)} rows with prompt_tokens_pre + cap > max_model_len {max_model_len}")
    ct = float(summary.get(CLIENT_TIMEOUT_FIELD) or 0)
    if ct < CLIENT_TIMEOUT_MIN_S:
        failures.append(f"summary {CLIENT_TIMEOUT_FIELD} {summary.get(CLIENT_TIMEOUT_FIELD)!r} < the proxy's {CLIENT_TIMEOUT_MIN_S:.0f} s")
    comp = sorted(int(r["usage_completion_tokens"]) for r in forwarded if isinstance(r.get("usage_completion_tokens"), int))
    quant = ({q: C.nearest_rank_percentile(comp, p) for q, p in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99))} | {"max": comp[-1], "mean": sum(comp) / len(comp)}) if comp else None
    # C5: the decode rate is the AGGREGATE (the accepted attempt's completions / the wall they spanned); the per-request
    # reading rides beside it, labelled — a ledger whose rows the rate cannot be read from is torn, and refused by name
    rate: Mapping[str, Any] | None = None
    if forwarded:
        try:
            rate = IC.aggregate_decode_tok_s(forwarded, min_rows=1)
        except ValueError as e:
            failures.append(f"decode rate unreadable: {e}")
    return {"passed": not failures, "failures": failures, "n_forwarded": len(forwarded), "n_draws": len(draws), "n_rejected": n_rejected,
            "n_length": n_length, "n_at_cap_summary": n_at_cap_summary, "n_at_cap_recorded": n_at_cap_recorded,
            "completion_tokens": quant, "decode_tok_s_aggregate": (rate["tok_s"] if rate else None),
            "decode_tok_s_per_request": (rate["per_request_tok_s"] if rate else None),
            "generation_span": ({k: rate[k] for k in ("first_start", "last_end", "elapsed_s")} if rate else None),
            # EVERY recorded attempt's own span (the billed cost includes the abandoned attempts' rows)
            "generation_span_by_attempt": generation_span_by_attempt(all_task_rows, evidence_by_id or {}),
            "client_timeout_s": summary.get(CLIENT_TIMEOUT_FIELD),
            "window": max_model_len, "accepted_attempt": accepted, "rows_by_attempt": by_attempt,
            "n_task_rows_all_attempts": len(all_task_rows)}


def generation_span_by_attempt(rows: Sequence[Mapping[str, Any]], evidence_by_id: Mapping[str, Mapping[str, Any]]
                               ) -> dict[str, dict[str, Any] | None]:
    """Per attempt (``contracts.ledger_row_attempt``; an unattributable row under ``unattributed``), the wall its
    task rows spanned (``ifeval_contracts.generation_span``) — the basis of the BILLED cost beside the accepted attempt's
    actual cost. A span the rows cannot yield (a naive stamp, a non-positive latency) is ``None`` for that attempt,
    never a guess."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        try:
            aid = C.ledger_row_attempt(r, evidence_by_id)
        except C.AttemptScopeError:
            aid = "unattributed"
        grouped.setdefault(aid, []).append(r)
    out: dict[str, dict[str, Any] | None] = {}
    for aid in sorted(grouped):
        try:
            span = IC.generation_span(grouped[aid])
            out[aid] = {k: span[k] for k in ("first_start", "last_end", "elapsed_s", "n_rows")}
        except ValueError:
            out[aid] = None
    return out


def request_evidence_check(path: Path, *, ledger_rows: Sequence[Mapping[str, Any]], row: Mapping[str, Any], srow: C.SamplingRow,
                           n_draws: int, manifest: Mapping[str, Any], evidence_by_id: Mapping[str, Mapping[str, Any]] | None = None,
                           evidence_problem: str | None = None) -> dict[str, Any]:
    """Gate I2's wire half (P29 / P42): every TASK record OF THE ACCEPTED ATTEMPT (:func:`accepted_attempt_id`;
    ``contracts.records_of_attempts`` — a record without its stamp refuses by name) is version 3, joins the accepted
    attempt's ledger rows by request id + body digest, and proves the row on the wire
    (``ifeval_contracts.request_evidence_sampling_problem``: the sampling numbers, presence_penalty present iff the row sets
    it, the IFEval seed for (item, trial), the family's OFF arm: the switch False, or ABSENT on a no-switch family). The
    accepted attempt's records are exactly the draws; every
    other attempt's records are counted per attempt (``records_by_attempt``), never held against the draws (C4). A v2
    record is NOT RECORDED — never a pass."""
    failures: list[str] = []
    if not Path(path).is_file():
        return {"passed": False, "failures": [f"no {C.REQUEST_EVIDENCE_FILENAME} at {path}"], "n_task_records": 0}
    if evidence_problem:
        failures.append(evidence_problem)
    all_recs = [r for r in iter_jsonl(path) if r.get("kind") == C.REQUEST_EVIDENCE_KIND_TASK and r.get("task") == TASK]
    by_attempt = dict(sorted(Counter(str(r.get(C.REQUEST_EVIDENCE_ATTEMPT_FIELD)) for r in all_recs).items()))
    accepted, why = accepted_attempt_id(manifest)
    recs: list[Mapping[str, Any]] = []
    acc_ledger: list[Mapping[str, Any]] = []
    if why:
        failures.append(why)
    else:
        try:
            recs = C.records_of_attempts(all_recs, [accepted])
            acc_ledger = C.rows_of_attempts([r for r in ledger_rows if r.get("task") == TASK], [accepted], evidence_by_id=evidence_by_id or {})
        except C.AttemptScopeError as e:
            failures.append(str(e))
    if not why and len(recs) != n_draws:
        failures.append(f"{len(recs)} task records of the accepted attempt {accepted!r} != {n_draws} draws")
    by_id = {r.get("request_id"): r for r in acc_ledger}
    n_v3 = n_pp = 0
    for rec in recs:
        lrow = by_id.get(rec.get("request_id"))
        if lrow is None:
            failures.append(f"record {rec.get('request_id')} has no ledger row")
        elif lrow.get("body_sha256") != rec.get("body_sha256"):
            failures.append(f"record {rec.get('request_id')}: body digest != the ledger's")
        if int(rec.get("request_evidence_version") or 0) >= IC.REQUEST_EVIDENCE_VERSION_V3:
            n_v3 += 1
        n_pp += 1 if rec.get("presence_penalty_present") else 0
        why = IC.request_evidence_sampling_problem(rec, row=srow, arm=IC.ifeval_effective_arm(row["family"]))
        if why:
            failures.append(f"record {rec.get('sample')}: {why}")
        if len(failures) > 20:
            break
    want_pp = srow.presence_penalty is not None
    return {"passed": not failures, "failures": failures[:20], "n_task_records": len(recs), "n_v3": n_v3,
            "n_presence_penalty_present": n_pp, "presence_penalty_expected": want_pp, "accepted_attempt": accepted,
            "n_task_records_all_attempts": len(all_recs), "records_by_attempt": by_attempt}


def wire_digest(evidence_recs: Sequence[Mapping[str, Any]], ledger_rows: Sequence[Mapping[str, Any]], *, manifest: Mapping[str, Any],
                evidence_by_id: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The ACCEPTED attempt's wire facts per draw (keyed by the sample header value ``<item_id>/<sample_idx>``) — what the
    noise-floor attestation compares draw for draw across the two legs of a pair: the request body's
    sha256 and the seed the body carried (the v3 evidence record), the prompt tokens the server counted before generation
    and the cap requested (the forwarded ledger row). Scoped to the accepted attempt exactly as I1 / I2 scope their counts;
    a store whose attempt cannot be resolved files ``problem`` and empty maps (the attestation then reads it as not
    attestable, never as equal)."""
    out: dict[str, Any] = {"accepted_attempt": None, "problem": None, "body_sha256_by_draw": {}, "seed_by_draw": {},
                           "prompt_tokens_by_draw": {}, "max_tokens_by_draw": {}}
    accepted, why = accepted_attempt_id(manifest)
    out["accepted_attempt"] = accepted
    if why:
        out["problem"] = why
        return out
    try:
        recs = C.records_of_attempts([r for r in evidence_recs if r.get("kind") == C.REQUEST_EVIDENCE_KIND_TASK and r.get("task") == TASK],
                                     [accepted])
        rows = C.rows_of_attempts([r for r in ledger_rows if r.get("task") == TASK and r.get("rejected") is not True], [accepted],
                                  evidence_by_id=evidence_by_id or {})
    except C.AttemptScopeError as e:
        out["problem"] = str(e)
        return out
    for r in recs:
        s = str(r.get("sample"))
        out["body_sha256_by_draw"][s] = r.get("body_sha256")
        out["seed_by_draw"][s] = r.get("seed")
    for r in rows:
        s = str(r.get("sample"))
        out["prompt_tokens_by_draw"][s] = r.get("prompt_tokens_pre")
        out["max_tokens_by_draw"][s] = r.get("max_tokens_requested")
    return out


SERVED_GEOMETRY_FILENAME = "served_geometry.json"
EVIDENCE_FILENAME = "evidence.json"
#: the endpoint's thinking-header probe (`{"switch", "moves", "off_ids", "on_ids"}`; `contracts.attested_thinking_header`)
THINKING_HEADER_KEY = "thinking_header"
#: the standup's fused-Mamba attestation ON the endpoint (`sisteer.capeval.serving.mamba_backend.live_attest`'s record:
#: `{ok, problems, reports, rpc_method, contract_sha256, expected_n_mamba_layers, asked_at}`)
MAMBA_ATTESTATION_KEY = "mamba_backend_attestation"


def served_record(out_root: Path) -> dict[str, Any] | None:
    """The Modal launcher's ``served_geometry.json`` beside the store — ``{geometry, endpoint, standup_argv,
    reduction_mode_requested}``: the geometry the standup SERVED (``eval``, ``max_model_len``, the served ``max_num_seqs``,
    ``configured_max_num_seqs`` + ``kv_shrink`` when the KV-derived shrink fired, ``window_source``, ``seqs_source``) and
    the standup's endpoint record (``chat_template_hash``, ``thinking_header``, ``object_artifact_sha256``, the Mamba
    attestation). ``None`` when absent (the reader then refuses by name)."""
    doc = read_json_if_file(Path(out_root) / SERVED_GEOMETRY_FILENAME)
    return dict(doc) if isinstance(doc, Mapping) else None


def evidence_record(out_root: Path) -> dict[str, Any] | None:
    """The driver's ``evidence.json`` beside the store (``serve_header.template_kwargs`` = the chat-template kwargs the
    SERVER rendered with — the thinking-OFF pin as served; ``served_template_hash``)."""
    doc = read_json_if_file(Path(out_root) / EVIDENCE_FILENAME)
    return dict(doc) if isinstance(doc, Mapping) else None


def thinking_pin_problem(evidence: Mapping[str, Any] | None, endpoint: Mapping[str, Any] | None, *, family: str) -> tuple[str | None, dict[str, Any]]:
    """The thinking-OFF pin AS SERVED, read from the producer's records: ``evidence.serve_header.template_kwargs`` (what the
    server's chat template was rendered with) judged against the endpoint's ``thinking_header`` probe — a template WITH a
    switch (``switch`` true: gemma / Qwen / Nemotron; the wrapper composes the pin on every such family, not only the two
    that also move the header) must carry exactly the OFF pin; a template with NO switch (llama, ``switch`` false, the
    ``not_available`` arm) carries no kwargs (``{}``); ``enable_thinking: true`` anywhere, a missing pin on a switch
    family, a missing probe, or a served template hash that is not the endpoint's are contract violations. Returns
    ``(problem or None, the facts read)``. The old reader looked for ``endpoint.default_chat_template_kwargs`` — a key
    the Modal launcher never writes — and refused every store (V20 / V41)."""
    header = (endpoint or {}).get(THINKING_HEADER_KEY) if isinstance(endpoint, Mapping) else None
    serve = (evidence or {}).get("serve_header") if isinstance(evidence, Mapping) else None
    kwargs = serve.get("template_kwargs") if isinstance(serve, Mapping) else None
    kwargs = _norm_json(kwargs)
    switch = header.get("switch") if isinstance(header, Mapping) else None
    facts = {"template_kwargs": kwargs, "thinking_switch": switch, "header_moves": (header or {}).get("moves") if isinstance(header, Mapping) else None,
             "served_template_hash": serve.get("served_template_hash") if isinstance(serve, Mapping) else None,
             "endpoint_chat_template_hash": (endpoint or {}).get(C.ENDPOINT_KEY_CHAT_TEMPLATE_HASH) if isinstance(endpoint, Mapping) else None,
             "arm": IC.ifeval_effective_arm(family)}
    if not isinstance(serve, Mapping):
        return f"no {EVIDENCE_FILENAME} serve_header beside the store — the served template kwargs are not recorded", facts
    if not isinstance(header, Mapping) or not isinstance(switch, bool):
        return f"the endpoint records no {THINKING_HEADER_KEY} probe (switch) — whether the template has a thinking switch is not recorded", facts
    want = _norm_json(IC.THINKING_OFF_TEMPLATE_KWARGS)
    if isinstance(kwargs, Mapping) and kwargs.get("enable_thinking") is True:
        return f"served template_kwargs {kwargs!r} ENABLE thinking (family {family})", facts
    if switch and kwargs != want:
        return f"served template_kwargs {kwargs!r} != the thinking-OFF pin {want!r} on a template with a switch (family {family})", facts
    if not switch and kwargs not in (None, {}):
        return f"served template_kwargs {kwargs!r} on a template with NO switch (family {family}, arm {facts['arm']}) — nothing to pin", facts
    if facts["served_template_hash"] is not None and facts["endpoint_chat_template_hash"] is not None \
            and facts["served_template_hash"] != facts["endpoint_chat_template_hash"]:
        return f"served_template_hash {facts['served_template_hash']!r} != the endpoint's chat_template_hash {facts['endpoint_chat_template_hash']!r}", facts
    return None, facts


def served_geometry_problem(served: Mapping[str, Any] | None, manifest: Mapping[str, Any], *, want_entry: str) -> tuple[str | None, dict[str, Any]]:
    """The geometry the standup SERVED, read from ``served_geometry.geometry`` (the launcher's record; the ``endpoint``
    block's own ``geometry`` carries ``eval None`` / ``window_source 'override'`` because the launcher passes the standup
    explicit dims, and the manifest's copy says the same — those are the launcher's spellings of the SAME numbers, reconciled
    here, never the contract). The contract: the served entry is the row's for this cap; the served window is the table
    entry's (``window_source table``); the served S is the model budget's (``seqs_source model_budget``) or the KV-derived one
    under a ``kv_shrink`` record that HOLDS (``sisteer.capeval.serving.geometry.kv_shrink_problem``, ``seqs_source
    kv_derived``); the endpoint's ``max_model_len`` / ``max_num_seqs`` equal the served L / S; the manifest's ``concurrency``
    (the client's in-flight width) and its ``geometry`` copy equal the served S / L. Returns ``(problem, facts)``."""
    from sisteer.capeval.serving import geometry as sg
    g = served.get("geometry") if isinstance(served, Mapping) else None
    ep = served.get("endpoint") if isinstance(served, Mapping) else None
    facts: dict[str, Any] = {"geometry_eval": None, "max_model_len": None, "served_seqs": None, "configured_seqs": None, "kv_shrink": None,
                             "window_source": None, "seqs_source": None, "manifest_geometry_source": None}
    if not isinstance(g, Mapping):
        return f"no {SERVED_GEOMETRY_FILENAME} geometry block beside the store — the served geometry is not recorded", facts
    L, S = g.get("max_model_len"), g.get("max_num_seqs")
    shrink = g.get(sg.KV_SHRINK_KEY)
    if shrink is None and isinstance(ep, Mapping):
        shrink = ep.get(sg.KV_SHRINK_KEY)
    configured = g.get("configured_max_num_seqs", S if shrink is None else None)
    mgeom = manifest.get(C.MANIFEST_KEY_GEOMETRY) if isinstance(manifest.get(C.MANIFEST_KEY_GEOMETRY), Mapping) else {}
    facts.update(geometry_eval=g.get("eval"), max_model_len=L, served_seqs=S, configured_seqs=configured,
                 kv_shrink=(dict(shrink) if isinstance(shrink, Mapping) else None), window_source=g.get("window_source"), seqs_source=g.get("seqs_source"),
                 manifest_geometry_source={"window_source": mgeom.get("window_source"), "seqs_source": mgeom.get("seqs_source"),
                                           "max_model_len": mgeom.get("max_model_len"), "max_num_seqs": mgeom.get("max_num_seqs")})
    if g.get("eval") != want_entry:
        return f"served geometry.eval {g.get('eval')!r} != the row's entry {want_entry} for this cap", facts
    if not isinstance(L, int) or not isinstance(S, int) or isinstance(L, bool) or isinstance(S, bool) or L <= 0 or S <= 0:
        return f"served max_model_len / max_num_seqs {L!r} / {S!r} are not the standup's integer resolution", facts
    if g.get("window_source") != "table":
        return f"served geometry.window_source {g.get('window_source')!r} != 'table' — the served window was not the table entry's", facts
    if shrink is not None:
        if g.get("seqs_source") != sg.KV_DERIVED_SOURCE:
            return f"a {sg.KV_SHRINK_KEY} record is present but seqs_source is {g.get('seqs_source')!r}, not {sg.KV_DERIVED_SOURCE!r}", facts
        if not isinstance(configured, int):
            return f"a {sg.KV_SHRINK_KEY} record is present but configured_max_num_seqs {configured!r} is not recorded", facts
        why = sg.kv_shrink_problem(shrink, configured_max_num_seqs=int(configured), max_model_len=int(L), served_max_num_seqs=S)
        if why:
            return f"the standup's {sg.KV_SHRINK_KEY} record does not hold: {why}", facts
    elif g.get("seqs_source") != "model_budget":
        return f"served geometry.seqs_source {g.get('seqs_source')!r} != 'model_budget' with no {sg.KV_SHRINK_KEY} record", facts
    if isinstance(ep, Mapping) and (ep.get("max_model_len") != L or ep.get("max_num_seqs") != S):
        return f"the endpoint serves {ep.get('max_model_len')!r} x {ep.get('max_num_seqs')!r}, not the served-geometry record's {L} x {S}", facts
    if manifest.get(C.MANIFEST_KEY_CONCURRENCY) not in (None, S):
        return f"manifest concurrency {manifest.get(C.MANIFEST_KEY_CONCURRENCY)!r} != the served S {S}", facts
    if mgeom and (mgeom.get("max_model_len") != L or mgeom.get("max_num_seqs") != S):
        return (f"the manifest's geometry copy {mgeom.get('max_model_len')!r} x {mgeom.get('max_num_seqs')!r} != the served {L} x {S} "
                "(the driver was handed other dims than the standup served)"), facts
    return None, facts


def mamba_attestation_problem(endpoint: Mapping[str, Any] | None) -> tuple[str | None, dict[str, Any]]:
    """The standup's OWN fused-Mamba attestation on the endpoint (``mamba_backend_attestation``: ``ok``, no ``problems``,
    every report's ``n_mamba_layers == expected_n_mamba_layers``) — the evidence that THIS standup ran every Mamba layer on
    the fused kernels. The (tree, geometry entry) certification that ADMITTED the configuration is a separate record on
    the volume (``nemo_backend/<attestation sha256>/``), cited from the certification doc (:func:`nemotron_certification_problem`)."""
    att = (endpoint or {}).get(MAMBA_ATTESTATION_KEY) if isinstance(endpoint, Mapping) else None
    facts = {"ok": None, "n_reports": None, "expected_n_mamba_layers": None, "contract_sha256": None, "problems": None, "asked_at": None}
    if not isinstance(att, Mapping):
        return f"no {MAMBA_ATTESTATION_KEY} on the endpoint — the fused backend of this standup is not attested", facts
    reports = att.get("reports") if isinstance(att.get("reports"), list) else []
    facts.update(ok=att.get("ok"), n_reports=len(reports), expected_n_mamba_layers=att.get("expected_n_mamba_layers"),
                 contract_sha256=att.get("contract_sha256"), problems=list(att.get("problems") or []), asked_at=att.get("asked_at"))
    if att.get("ok") is not True or facts["problems"]:
        return f"the standup's {MAMBA_ATTESTATION_KEY} is not ok: {facts['problems'] or att.get('ok')!r}", facts
    if not reports:
        return f"the standup's {MAMBA_ATTESTATION_KEY} carries no report", facts
    want_n = att.get("expected_n_mamba_layers")
    bad = [r.get("n_mamba_layers") for r in reports if not isinstance(r, Mapping) or r.get("n_mamba_layers") != want_n]
    if bad:
        return f"a report's n_mamba_layers {bad} != expected_n_mamba_layers {want_n!r}", facts
    return None, facts


def nemotron_certification_problem(certifications: Mapping[str, Any] | None, *, tree_sha: str | None, entry: str) -> tuple[str | None, dict[str, Any] | None]:
    """The launch-time certification of a Nemotron configuration — ONE admitted record per (producer tree, geometry entry)
    (the preflight's ``nemotron_backend_attested``) — looked up in the certification doc keyed by tree sha (full
    or 7+ hex prefix) then by entry: ``{"<tree>": {"<entry>": {"admitted": true, "attestation_sha256": ...}}}``. ``None`` doc =
    not asked (the record says so); a doc without the leg's (tree, entry) or with ``admitted`` false is a refusal."""
    if certifications is None:
        return None, None
    tree = str(tree_sha or "")
    by_tree = next((v for k, v in certifications.items() if tree and (tree.startswith(str(k)) or str(k).startswith(tree))), None)
    rec = (by_tree or {}).get(entry) if isinstance(by_tree, Mapping) else None
    if not isinstance(rec, Mapping):
        return f"no Nemotron certification for tree {tree[:8]!r} at entry {entry!r} in the certification doc", None
    if rec.get("admitted") is not True:
        return f"the Nemotron certification for tree {tree[:8]!r} at {entry!r} was not admitted", dict(rec)
    return None, dict(rec)


def certification_check(manifest: Mapping[str, Any], served: Mapping[str, Any] | None, *, row: Mapping[str, Any], cap: int,
                        evidence: Mapping[str, Any] | None = None, certifications: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Gate I3 per leg, read from the PRODUCER's records (the wave's reader items V20 / V36 (ii) / V41 (1)): the thinking-OFF
    pin as served (:func:`thinking_pin_problem`: ``evidence.serve_header.template_kwargs`` judged by the endpoint's
    ``thinking_header`` switch); the object digest == the row's pin (endpoint + manifest route); served positions ``all`` /
    ``response`` per the row, UNDECLARED; the served geometry (:func:`served_geometry_problem`: the row's entry per cap,
    the table window, the model-budget or KV-derived S under a holding ``kv_shrink``, the endpoint / manifest agreeing);
    on Nemotron the standup's own fused-backend attestation (:func:`mamba_attestation_problem`) and, when a certification
    doc is given, the admitted (tree, entry) certification (:func:`nemotron_certification_problem`)."""
    failures: list[str] = []
    endpoint = served.get("endpoint") if isinstance(served, Mapping) and isinstance(served.get("endpoint"), Mapping) else None
    out: dict[str, Any] = {"passed": False, "failures": failures, "served_seqs": None, "kv_shrink": None, "max_model_len": None, "geometry_eval": None}
    if not isinstance(served, Mapping):
        failures.append(f"no {SERVED_GEOMETRY_FILENAME} beside the store (the launcher's served-geometry record)")
        return out
    if endpoint is None:
        failures.append(f"{SERVED_GEOMETRY_FILENAME} carries no endpoint record")
    family = row["family"]
    pin_problem, pin_facts = thinking_pin_problem(evidence, endpoint, family=family)
    if pin_problem:
        failures.append(pin_problem)
    if endpoint is not None and endpoint.get(C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256) != row["object_sha256"]:
        failures.append("endpoint object digest != the row's pin")
    route = manifest.get(C.MANIFEST_KEY_ROUTE) or {}
    positions = route.get(C.LENS_SERVED_POSITIONS_KEY)
    if row["fitted"]:
        if positions != [row["served_positions_expected"]]:
            failures.append(f"served positions {positions!r} != {[row['served_positions_expected']]}")
    elif positions not in (None, []):
        failures.append(f"a bare / prompted route serves positions {positions!r}")
    if manifest.get("declared") or route.get(C.ROUTE_KEY_SERVE_DECLARATIONS):
        failures.append("the cell is DECLARED — IFEval serves every cell undeclared (P15)")
    want_entry = row["geometry_eval_by_cap"][str(cap)]
    geo_problem, geo = served_geometry_problem(served, manifest, want_entry=want_entry)
    if geo_problem:
        failures.append(geo_problem)
    att_facts: dict[str, Any] | None = None
    cert_rec: dict[str, Any] | None = None
    if family == "nemotron_h":
        att_problem, att_facts = mamba_attestation_problem(endpoint)
        if att_problem:
            failures.append(att_problem)
        tree = (manifest.get(C.MANIFEST_KEY_PRODUCER) or {}).get("tree_sha")
        cert_problem, cert_rec = nemotron_certification_problem(certifications, tree_sha=tree, entry=want_entry)
        if cert_problem:
            failures.append(cert_problem)
    out.update({"passed": not failures, "served_seqs": geo["served_seqs"], "configured_seqs": geo["configured_seqs"], "kv_shrink": geo["kv_shrink"],
                "max_model_len": geo["max_model_len"], "geometry_eval": geo["geometry_eval"], "window_source": geo["window_source"],
                "seqs_source": geo["seqs_source"], "manifest_geometry_source": geo["manifest_geometry_source"],
                "template_kwargs": pin_facts["template_kwargs"], "thinking_switch": pin_facts["thinking_switch"], "header_moves": pin_facts["header_moves"],
                "served_template_hash": pin_facts["served_template_hash"], "arm": pin_facts["arm"],
                "object_sha256": endpoint.get(C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256) if endpoint is not None else None,
                "nemotron_attestation": att_facts, "nemotron_certification": cert_rec,
                # the old key, kept for readers of earlier reports: the pin AS SERVED (the evidence's template kwargs)
                "default_chat_template_kwargs": pin_facts["template_kwargs"]})
    return out


def recorder_record(out_root: Path) -> dict[str, Any] | None:
    """The resolved-sampler RECORDER's verdict beside the store (``contracts.RESOLVED_SAMPLER_FILENAME``; the per-kind
    mode): the top-level ``verdict`` (``AGREES`` = the client class matched the pinned row and every probe kind agreed),
    the per-kind probe verdicts, the client / probe caps and the unclassified counts. ``None`` when the launcher wrote no
    record (never a pass — the wire evidence is the gate; the recorder is reported beside it)."""
    doc = read_json_if_file(Path(out_root) / C.RESOLVED_SAMPLER_FILENAME)
    if not isinstance(doc, Mapping):
        return None
    return {"verdict": doc.get("verdict"), "probe_verdict_by_kind": doc.get("probe_verdict_by_kind"), "n_client_requests": doc.get("n_client_requests"),
            "n_unclassified_requests": doc.get("n_unclassified_requests"), "n_probe_unclassified_requests": doc.get("n_probe_unclassified_requests"),
            "client_caps": doc.get("client_caps"), "probe_caps": doc.get("probe_caps"), "reference_source": doc.get("reference_source"),
            "why": doc.get("why"), "error": doc.get("error")}


RECORDER_AGREES = "AGREES"


def recorder_problem(rec: Mapping[str, Any] | None) -> str | None:
    """A recorder record that is present but does NOT agree (or leaves requests unclassified) is a finding on the leg."""
    if rec is None:
        return None
    if rec.get("verdict") != RECORDER_AGREES:
        return f"recorder verdict {rec.get('verdict')!r} (why: {rec.get('why')!r}; error: {rec.get('error')!r})"
    if (rec.get("n_unclassified_requests") or 0) or (rec.get("n_probe_unclassified_requests") or 0):
        return f"recorder left {rec.get('n_unclassified_requests')} / {rec.get('n_probe_unclassified_requests')} requests unclassified"
    return None


def result_stamp(out_root: Path) -> dict[str, Any] | None:
    """The launcher's ``RESULT`` stamp beside the store (its last line: ``... rc=<n> result=<COMPLETE|INCOMPLETE> status=<OK|...>
    sha=<tree> at=<ts>``), parsed — the operator's completion fact per store; None when absent."""
    p = Path(out_root) / "RESULT"
    if not p.is_file():
        return None
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return {"raw": ""}
    last = lines[-1]
    out: dict[str, Any] = {"raw": last[:240]}
    for tok in last.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            if k in ("rc", "result", "status", "sha", "at", "name"):
                out[k] = v
    return out


def _norm_json(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def stamps_problem(summary: Mapping[str, Any], *, row: Mapping[str, Any]) -> str | None:
    """The summary's stamp triple against the row's — the render regime FAMILY-resolved (``ifeval_contracts.
    ifeval_expected_summary_stamps``: the frozen ``expected_stamps`` spell ``thinking_off`` on every row; a no-switch
    family's store reads ``unspecified`` because its requests carry no ``enable_thinking``, lane X's seam 2)."""
    want = IC.ifeval_expected_summary_stamps(row)
    got = {f: summary.get(f) for f in C.EXPECTED_STAMP_TRIPLE_FIELDS}
    if got != want:
        return f"stamps {got} != the row's expected {want}"
    return None


# --------------------------------------------------------------------------- the ladder
def ladder_decision_of(record: Any) -> tuple[dict[str, Any] | None, str | None]:
    """THE decision of a filed ladder record (``ifeval_contracts.ladder_decision``: the planner's block over the MERGED
    reading, C1) or ``(None, why)`` — a record without it is the store's classification only, never a decision."""
    try:
        return IC.ladder_decision(record), None
    except IC.LadderDecisionError as e:
        return None, str(e)


def ladder_decision_problem(record: Mapping[str, Any], *, cap: int, n_store_at_cap: int) -> str | None:
    """Gate I5's decision half (C1): the record carries THE decision (``ifeval_contracts.LADDER_DECISION_KEY``), which is the
    instrument's (threshold / rule / provenance), over the FULL population (``n_draws == 541 x 2``: the merged reading has
    every draw), composed of the ladder up to ``cap`` (``chain``; ``r2048`` at pass 1, ``merged`` above), with an exit in the
    adapter's vocabulary consistent with its OWN share against the threshold (a ceiling exit at the top of the ladder), and —
    at pass 1, where the merged reading IS the store — equal to the store's own exit and at-cap count; at a rung the merged
    at-cap count is at least the rung store's (the store's still-at-cap draws plus the source's loops never re-drawn), and
    an escalating decision re-draws exactly the store's eligible draws."""
    dec, why = ladder_decision_of(record)
    if why:
        return why
    if dec["threshold"] != TRIGGER_THRESHOLD or dec["rule"] != TRIGGER_RULE or dec["trigger_provenance"] != TRIGGER_PROVENANCE:
        return "the decision's rule / threshold / trigger_provenance are not the instrument's"
    if dec["n_draws"] != N * K:
        return f"the decision's population {dec['n_draws']} != the merged reading's {N * K} draws"
    if dec["chain"] != [c for c in CAPS if c <= int(cap)]:
        return f"the decision's chain {dec['chain']} is not the ladder up to {int(cap)}"
    if dec["exit"] not in IR.EXITS:
        return f"the decision's exit {dec['exit']!r} is not one of the adapter's {IR.EXITS}"
    if int(cap) == CAP_BASE:
        if dec["n_at_cap"] != n_store_at_cap or dec["exit"] != record.get("exit") or dec["n_rung"] != 0:
            return (f"at pass 1 the decision is over the store itself, yet it says {dec['exit']!r} over {dec['n_at_cap']} at cap with "
                    f"{dec['n_rung']} rung draws while the store says {record.get('exit')!r} over {n_store_at_cap}")
    elif dec["n_at_cap"] < n_store_at_cap:
        return f"the merged reading's at-cap count {dec['n_at_cap']} < the rung store's own {n_store_at_cap}"
    triggered = dec["n_at_cap"] / dec["n_draws"] > TRIGGER_THRESHOLD
    if not triggered and dec["exit"] not in IR.NOT_TRIGGERED_EXITS:
        return f"decision exit {dec['exit']!r} though its share {dec['share_at_cap']} does not exceed {TRIGGER_THRESHOLD}"
    if triggered and dec["exit"] in IR.NOT_TRIGGERED_EXITS:
        return f"decision exit {dec['exit']!r} though its share {dec['share_at_cap']} exceeds {TRIGGER_THRESHOLD}"
    if triggered and int(cap) >= CEILING and dec["exit"] != CL.EXIT_CEILING:
        return f"decision exit {dec['exit']!r} at the ceiling"
    if dec["exit"] == CL.ESCALATE:
        if dec["next_cap"] != CL.next_rung(int(cap), CAPS, CEILING):
            return f"decision next_cap {dec['next_cap']} is not the rung above {int(cap)}"
        if [tuple(d) for d in dec["eligible_draws"]] != [tuple(d) for d in record.get("eligible_draws") or []]:
            return "the decision's eligible draws are not the store's eligible (truncated, non-loop) draws"
    return None


def ladder_record_problem(record: Any, *, items: Sequence[Mapping[str, Any]], cap: int, source_manifest: Mapping[str, Any]) -> str | None:
    """Gate I5 / I6: the adapter's record is THIS instrument's (schema, instrument, rule, threshold, provenance), its
    population is the store's draws, its at-cap draws are the outcome record's, ``n_at_cap == n_truncated + n_loop`` (the
    INTEGER identity; each share equal to the producer's rounding of its own count over ``n_draws``), the eligible ∪ loop
    draws == the at-cap draws, boundary ⊆ eligible, the store's own exit is consistent with the threshold over its own
    population (the adapter's record is internally consistent — a DIAGNOSTIC at a rung), it is bound to the store's identity,
    and it carries THE decision (:func:`ladder_decision_problem`, C1)."""
    if not isinstance(record, Mapping):
        return "no ladder record"
    if record.get("schema") != CL.RECORD_SCHEMA or record.get("instrument") != INSTRUMENT.name:
        return f"schema / instrument {record.get('schema')!r} / {record.get('instrument')!r}"
    if record.get("rule") != TRIGGER_RULE or record.get("threshold") != TRIGGER_THRESHOLD or record.get("trigger_provenance") != TRIGGER_PROVENANCE:
        return "rule / threshold / trigger_provenance are not the instrument's"
    if record.get("cap") != int(cap) or record.get("ceiling") != CEILING:
        return f"cap {record.get('cap')!r} / ceiling {record.get('ceiling')!r}"
    draws = {(str(it["item_id"]), int(it["sample_idx"])) for it in items}
    at = {(str(it["item_id"]), int(it["sample_idx"])) for it in items if it.get("at_cap")}
    if record.get("n_draws") != len(draws) or record.get("n_at_cap") != len(at):
        return f"n_draws {record.get('n_draws')!r} / n_at_cap {record.get('n_at_cap')!r} != the store's {len(draws)} / {len(at)}"
    elig = {(i, s) for i, s in record.get("eligible_draws") or []}
    loops = {(i, s) for i, s in record.get("loop_draws") or []}
    bound = {(i, s) for i, s in record.get("boundary_draws") or []}
    if elig | loops != at or elig & loops:
        return "eligible_draws ∪ loop_draws != the at-cap draws (or they overlap)"
    if not bound <= elig:
        return "boundary_draws are not a subset of eligible_draws"
    if record.get("n_truncated") != len(elig) or record.get("n_loop") != len(loops) or record.get("n_boundary") != len(bound):
        return "n_truncated / n_loop / n_boundary disagree with the draw lists"
    # C3: the at-cap identity is held on the INTEGER counts, and every share to the producer's OWN rounding of its own
    # count over its own denominator — never "the rounded shares sum within a tolerance" (4 / 1,082 rounds to 0.003697,
    # 2 / 1,082 + 2 / 1,082 to 0.003696: a correct record whose rounded shares differ by 1e-6 + epsilon)
    if record["n_at_cap"] != record["n_truncated"] + record["n_loop"]:
        return f"n_at_cap {record['n_at_cap']} != n_truncated {record['n_truncated']} + n_loop {record['n_loop']}"
    for share, count in (("share_at_cap", "n_at_cap"), ("share_truncated", "n_truncated"), ("share_loop", "n_loop")):
        want = round(record[count] / record["n_draws"], IR.SHARE_DECIMALS)
        if record.get(share) != want:
            return f"{share} {record.get(share)!r} != round({count} / n_draws, {IR.SHARE_DECIMALS}) = {want}"
    share = len(at) / len(draws) if draws else 0.0
    triggered = share > TRIGGER_THRESHOLD
    exit_ = record.get("exit")
    if not triggered and exit_ not in IR.NOT_TRIGGERED_EXITS:
        return f"exit {exit_!r} though the share {share:.4f} does not exceed {TRIGGER_THRESHOLD}"
    if triggered and exit_ in IR.NOT_TRIGGERED_EXITS:
        return f"exit {exit_!r} though the share {share:.4f} exceeds {TRIGGER_THRESHOLD}"
    if triggered and int(cap) >= CEILING and exit_ != CL.EXIT_CEILING:
        return f"exit {exit_!r} at the ceiling"
    src = record.get("source") or {}
    if src.get("identity_hash") != source_manifest.get("harness_config_hash"):
        return "the record's source identity_hash is not the store's harness_config_hash"
    return ladder_decision_problem(record, cap=cap, n_store_at_cap=len(at))


def rung_subset_problem(subset: Any, *, record: Mapping[str, Any], decision: Mapping[str, Any], source_manifest: Mapping[str, Any],
                        source_out_root: str | None) -> str | None:
    """The rung subset beside its SOURCE leg's record: a valid ``rung_subset/2`` body whose draws are the record's eligible
    draws, at THE decision's next cap under its transport verdict (C1: the decision, never the store's own ``next_cap``),
    bound to the source store's identity hash and to its cap, and whose recorded ``source_out_root`` names the source leg's
    WAVE-RELATIVE layout (``source_out_root`` = ``ifeval_paths(model, view, cell, source cap).out_root``; the recorded root
    is absolute and relocates with every pull — C2)."""
    try:
        IC.validate_item_subset_v2(subset, where=IC.RUNG_SUBSET_FILENAME)
    except IC.ItemSubsetError as e:
        return str(e)
    if [tuple(d) for d in subset["draws"]] != [tuple(d) for d in record.get("eligible_draws") or []]:
        return f"{IC.RUNG_SUBSET_FILENAME}.draws != the record's eligible_draws"
    if subset["source_identity_hash"] != source_manifest.get("harness_config_hash"):
        return f"{IC.RUNG_SUBSET_FILENAME}.source_identity_hash != the source store's"
    if subset["cap"] != decision.get("next_cap"):
        return f"{IC.RUNG_SUBSET_FILENAME}.cap {subset['cap']} != the decision's next_cap {decision.get('next_cap')}"
    if subset["source_cap"] != record.get("cap"):
        return f"{IC.RUNG_SUBSET_FILENAME}.source_cap {subset['source_cap']} != the source store's cap {record.get('cap')}"
    if dict(subset["transport_check"]) != dict(decision.get("transport") or {}):
        return f"{IC.RUNG_SUBSET_FILENAME}.transport_check is not the decision's transport verdict"
    if source_out_root is not None and not same_store_tail(subset.get("source_out_root"), source_out_root):
        return f"{IC.RUNG_SUBSET_FILENAME}.source_out_root {subset.get('source_out_root')!r} does not name the source leg {source_out_root!r}"
    return None


def population_problem(items: Sequence[Mapping[str, Any]], *, canonical_ids: Sequence[str] | None,
                       subset_draws: Sequence[Sequence[Any]] | None) -> str | None:
    """Gate I7: a pass-1 / merged reading has EXACTLY the 541 canonical ids x 2 trials; a rung leg has EXACTLY its subset's draws."""
    got = sorted((str(it["item_id"]), int(it["sample_idx"])) for it in items)
    if subset_draws is not None:
        want = sorted((str(i), int(s)) for i, s in subset_draws)
        if got != want:
            return f"rung draws ({len(got)}) != the subset's ({len(want)})"
        return None
    if canonical_ids is None:
        return None
    want = sorted((str(i), t) for i in canonical_ids for t in range(K))
    if got != want:
        missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
        return f"draws != the canonical {N} ids x {K} trials (missing {missing[:3]}, extra {extra[:3]})"
    return None


def by_trial_value(by_trial: Any, idx: int) -> Any:
    """One trial's value of a ``*_mean_by_trial`` summary field: the PRODUCER writes a list indexed by trial
    (``inspect_served``'s summary; the wave shakedown's real store, 2026-09-13), the synthetic store a dict keyed by
    the trial's string — both read here, anything else is "not recorded" (None)."""
    if isinstance(by_trial, (list, tuple)):
        return by_trial[idx] if 0 <= idx < len(by_trial) else None
    if isinstance(by_trial, Mapping):
        return by_trial.get(str(idx), by_trial.get(idx))
    return None


def eval_cross_check(cell_dir: Path, items: Sequence[Mapping[str, Any]], *, summary: Mapping[str, Any], cap: int) -> dict[str, Any]:
    """Gate I4 (needs inspect_ai): both samples' retained ``.eval`` logs, read through the producer's log selector, carry
    the stock scorer's dict-valued score per draw whose five keys equal the outcome item's, and the per-trial means of the
    two prompt-level metrics equal the summary's ``*_mean_by_trial`` to 1e-9; the texts of every at-cap draw are on record."""
    from sisteer.capeval.tiers import inspect_served as isv
    from inspect_ai.log import read_eval_log

    failures: list[str] = []
    by_draw = {(str(it["item_id"]), int(it["sample_idx"])): it for it in items}
    seen = 0
    means: dict[str, dict[str, float]] = {}
    stock: dict[str, Any] = {}
    for idx in range(K):
        ids = sorted({i for (i, s) in by_draw if s == idx})
        if not ids:
            continue
        d = C.inspect_log_dir(cell_dir, TASK, idx)
        log = isv.reusable_log(d, expected_n=len(ids), expected_ids=ids, expected_max_tokens=int(cap))
        if log is None:
            failures.append(f"no complete .eval for sample {idx} at cap {cap} over {len(ids)} ids")
            continue
        full = read_eval_log(str(getattr(log, "location", "") or "")) if not getattr(log, "samples", None) else log
        vals: dict[str, list[float]] = {f: [] for f in IC.IFEVAL_METRIC_FIELDS}
        for s in full.samples or []:
            it = by_draw.get((str(s.id), idx))
            if it is None:
                failures.append(f"sample {s.id}/{idx} is not an attempted draw")
                continue
            score = (s.scores or {}).get(IC.PRIMARY_SCORER_IFEVAL)
            value = getattr(score, "value", None)
            if not isinstance(value, Mapping) or set(value) != {*IC.IFEVAL_METRIC_FIELDS, "num_instructions"}:
                failures.append(f"sample {s.id}/{idx}: the score is not the stock scorer's five-key dict")
                continue
            for f in (*IC.IFEVAL_METRIC_FIELDS, "num_instructions"):
                # EXACT integer equality on every one of the five stock-score fields: the earlier reader
                # required an integer inequality AND a truthiness difference, so a positive-to-positive move (1 vs 2
                # followed instructions) passed; a field the outcome item does not file is a mismatch, never a zero
                filed = it.get(f)
                if filed is None or int(value[f]) != int(filed):
                    failures.append(f"sample {s.id}/{idx}: {f} {value[f]!r} != the outcome item's {filed!r}")
                    break
            for f in IC.IFEVAL_METRIC_FIELDS:
                vals[f].append(float(value[f]))
            stop = getattr(getattr(s, "output", None), "stop_reason", None)
            if stop != it.get("stop_reason"):
                failures.append(f"sample {s.id}/{idx}: stop_reason {stop!r} != the outcome item's {it.get('stop_reason')!r}")
            seen += 1
        for f in ("prompt_level_strict", "prompt_level_loose"):
            key = f"{IC.METRIC_BY_FIELD[f]}_mean_by_trial"
            want = by_trial_value(summary.get(key), idx)
            if vals[f] and want is not None and abs(sum(vals[f]) / len(vals[f]) - float(want)) > 1e-9:
                failures.append(f"sample {idx}: .eval {f} mean {sum(vals[f]) / len(vals[f]):.6f} != the summary's {want}")
            means.setdefault(IC.METRIC_BY_FIELD[f], {})[str(idx)] = (sum(vals[f]) / len(vals[f])) if vals[f] else None
        # inspect_evals' OWN aggregate metrics of this trial's log (`prompt_strict_acc` / `_stderr`, ...): the naive per-trial
        # stderr published beside IFEval tables — read for comparability, labelled, never deciding
        stock[str(idx)] = stock_metrics_of_log(full)
        if len(failures) > 20:
            break
    if seen != len(by_draw) and not failures:
        failures.append(f"{seen} log samples != {len(by_draw)} draws")
    return {"passed": not failures, "failures": failures[:20], "n_log_samples": seen, "means_by_trial": means,
            "stock_metrics_by_trial": stock, "stock_metrics_note": STOCK_METRICS_NOTE}


#: the stock scorer's aggregate metric names inspect_evals' `ifeval` task files per log (one trial = one log here)
STOCK_METRIC_NAMES: tuple[str, ...] = ("prompt_strict_acc", "prompt_strict_stderr", "prompt_loose_acc", "prompt_loose_stderr",
                                       "inst_strict_acc", "inst_strict_stderr", "inst_loose_acc", "inst_loose_stderr", "final_acc", "final_stderr")
STOCK_METRICS_NOTE = ("inspect_evals' own aggregate metrics of each trial's retained .eval log (one trial = 541 prompts, one seeded draw "
                      "each): the naive stderr published beside IFEval tables, for comparability only — the analysis's intervals are the "
                      "item-paired bootstrap over both trials; the stock numbers never decide anything")


def stock_metrics_of_log(log: Any) -> dict[str, Any] | None:
    """``{metric name: value}`` of the stock scorer's aggregate metrics filed on one ``.eval`` log's ``results`` (the
    ``instruction_following`` scorer's ``prompt_strict_acc`` / ``prompt_strict_stderr`` and their siblings), or ``None``
    when the log carries no results block for that scorer (not recorded, never zero)."""
    results = getattr(log, "results", None)
    for sc in (getattr(results, "scores", None) or []):
        if getattr(sc, "name", None) != IC.PRIMARY_SCORER_IFEVAL and getattr(sc, "scorer", None) != IC.PRIMARY_SCORER_IFEVAL:
            continue
        metrics = getattr(sc, "metrics", None) or {}
        out = {}
        for name in STOCK_METRIC_NAMES:
            m = metrics.get(name) if isinstance(metrics, Mapping) else None
            out[name] = (float(m.value) if m is not None and getattr(m, "value", None) is not None else None)
        return out
    return None


# --------------------------------------------------------------------------- the FIT PROVENANCE of the served object
FIT_PROVENANCE_KEY = "fit_provenance"
FIT_PROVENANCE_FIELDS: tuple[str, ...] = ("model_id", "revision", "served_model_id", "served_revision", "fit_kind", "status", "transfer",
                                          "chat_template_bypassed", "object_sha", "source", "template_check")
FIT_CONDITION_NO_OBJECT = "no object (bare or prompted route)"
FIT_CONDITION_SAME_WEIGHTS = "document fit on the served checkpoint's own weights, served there (no transfer)"
FIT_CONDITION_TRANSFER = "document fit on OTHER weights, served on this checkpoint (a transfer)"
FIT_CONDITION_UNRECORDED = "fit provenance not recorded on the manifest"
FIT_CONDITION_RULE = ("the condition is read from the manifest's fit_provenance block — fit model_id / revision against served_model_id / "
                      "served_revision, with the producer's own `transfer` flag beside it — never inferred from the bank-view name "
                      "(the `it` view of a fitted row names the object bank, not the weights the object was fit on)")


def fit_provenance_record(manifest: Mapping[str, Any], *, row: Mapping[str, Any]) -> dict[str, Any]:
    """WHAT the served object was fit on, per the PRODUCER's ``fit_provenance`` block (the served-capability manifest writes
    the fit's ``model_id`` / ``revision``, the ``served_model_id`` / ``served_revision``, ``fit_kind``, ``transfer`` and
    ``chat_template_bypassed``): the condition word is derived from fit-vs-served identity, the manifest's ``transfer``
    flag is compared with it (a disagreement is a ``problem`` the certification gate reads), and the bank-view name is never
    consulted — the r1 record called the llama ``it`` LoRAs "base-fit transfer objects" from their view name while both
    manifests file an INSTRUCT-weight document fit served on instruct."""
    fp = manifest.get(FIT_PROVENANCE_KEY)
    recorded = isinstance(fp, Mapping)
    out: dict[str, Any] = {f: (fp.get(f) if recorded else None) for f in FIT_PROVENANCE_FIELDS}
    out.update(recorded=recorded, fit_weights=None, condition=None, problem=None, rule=FIT_CONDITION_RULE)
    if not row.get("fitted"):
        out["condition"] = FIT_CONDITION_NO_OBJECT
        return out
    if not recorded or fp.get("model_id") is None:
        out["condition"] = FIT_CONDITION_UNRECORDED
        out["problem"] = "a fitted cell whose manifest records no fit_provenance.model_id (what the object was fit on is not on record)"
        return out
    same = fp.get("model_id") == fp.get("served_model_id") and fp.get("revision") == fp.get("served_revision")
    out["fit_weights"] = "served_checkpoint" if same else "other_checkpoint"
    out["condition"] = FIT_CONDITION_SAME_WEIGHTS if same else FIT_CONDITION_TRANSFER
    flag = fp.get("transfer")
    if flag is not None and bool(flag) != (not same):
        out["problem"] = (f"the manifest's transfer flag {flag!r} disagrees with its fit model_id {fp.get('model_id')!r} @ {str(fp.get('revision'))[:8]} "
                          f"vs served {fp.get('served_model_id')!r} @ {str(fp.get('served_revision'))[:8]}")
    return out


# --------------------------------------------------------------------------- gate I5's TEXT half: the loop rule re-derived
def at_cap_text_check(out_root: Path, items: Sequence[Mapping[str, Any]], *, record: Any) -> dict[str, Any]:
    """Gate I5's text half — the eligible set re-derived from the texts equals the planner's (needs inspect_ai):
    every at-cap draw's emitted text is read from the retained ``.eval`` logs through the producer's own reader
    (``inspect_rung.store_texts``), the cap ladder's loop rule (``cap_ladder.degeneracy``) is applied to it, and the derived
    loop / eligible draw SETS are compared EXACTLY with the record's ``loop_draws`` / ``eligible_draws``; every at-cap text's
    sha256 must equal the record's ``item_facts[*].text_sha256``. The r1 gate trusted the recorded lists (a draw moved from
    ``loop_draws`` to ``eligible_draws`` with the counts kept consistent was accepted although its own ``item_facts`` said
    ``degenerate`` with a tail repetition of 0.868). An at-cap draw with no text on record is a failure (a loop is never
    assumed absent)."""
    from sisteer.capeval.tiers import inspect_rung as IR

    out: dict[str, Any] = {"passed": False, "failures": [], "n_at_cap": 0, "n_loop_derived": None, "n_eligible_derived": None,
                           "n_loop_recorded": None, "n_eligible_recorded": None, "n_text_sha256_verified": 0, "n_text_sha256_mismatch": 0,
                           "n_no_text": 0, "rule": CL.DEGENERATE_RULE}
    failures: list[str] = out["failures"]
    if not isinstance(record, Mapping):
        failures.append("no ladder record to re-derive against")
        return out
    try:
        facts = IR.load_store(Path(out_root), task=TASK)
        texts = IR.store_texts(facts)
    except IR.InspectRungError as e:
        failures.append(f"the retained texts cannot be read: {e}")
        return out
    rec_loops = {(str(i), int(s)) for i, s in record.get("loop_draws") or []}
    rec_elig = {(str(i), int(s)) for i, s in record.get("eligible_draws") or []}
    sha_by = {(str(f.get("item_id")), int(f.get("sample_idx"))): f.get("text_sha256") for f in record.get("item_facts") or []
              if isinstance(f, Mapping) and f.get("sample_idx") is not None}
    derived_loops: set[tuple[str, int]] = set()
    derived_elig: set[tuple[str, int]] = set()
    for it in items:
        if not it.get("at_cap"):
            continue
        out["n_at_cap"] += 1
        key = (str(it["item_id"]), int(it["sample_idx"]))
        text = texts.get(key)
        if text is None:
            out["n_no_text"] += 1
            failures.append(f"at-cap draw {key} has no retained text (a loop is never assumed absent)")
            continue
        (derived_loops if CL.degeneracy(text)["degenerate"] else derived_elig).add(key)
        want = sha_by.get(key)
        got = CL.text_sha256(text)
        if want is None:
            failures.append(f"at-cap draw {key}: the record's item_facts file no text_sha256 for it")
        elif got != want:
            out["n_text_sha256_mismatch"] += 1
            failures.append(f"at-cap draw {key}: the retained text's sha256 {got[:12]} != the record's {str(want)[:12]}")
        else:
            out["n_text_sha256_verified"] += 1
    out.update(n_loop_derived=len(derived_loops), n_eligible_derived=len(derived_elig), n_loop_recorded=len(rec_loops), n_eligible_recorded=len(rec_elig))
    if derived_loops != rec_loops:
        moved = sorted(derived_loops ^ rec_loops)
        failures.append(f"loop draws re-derived from the retained texts ({len(derived_loops)}) != the record's loop_draws ({len(rec_loops)}); "
                        f"differing draws {moved[:5]}")
    if derived_elig != rec_elig:
        moved = sorted(derived_elig ^ rec_elig)
        failures.append(f"eligible (truncated) draws re-derived from the retained texts ({len(derived_elig)}) != the record's eligible_draws "
                        f"({len(rec_elig)}); differing draws {moved[:5]}")
    out["passed"] = not failures
    out["failures"] = failures[:20]
    return out
