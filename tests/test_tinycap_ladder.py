"""The truncation ladder (``scripts/tinycap_ladder.py``) on
PRODUCTION-SHAPED stores: rows built from ``inspect_served.ItemResult.extra()``
(the contract's ``INSPECT_EXTRA_FIELDS`` + the two legs' ``code_source`` /
``sandbox_status`` / ``verifier_explanation``), summaries carrying
``task_quality``'s counts, outcome records with exactly
``TASK_OUTCOMES_HEADER_FIELDS`` / ``TASK_OUTCOME_ITEM_FIELDS`` and verified
against the summary by the store side's own verifier, markers through the
store side's writers, manifests through ``CellStore.write_cell_manifest`` with
the hashed ``tinycap`` identity block. The lane's REAL pins (slice B; verified
by ``contracts.load_bcb_pin`` / ``load_bfcl_pin``) are copied under a temp repo
root, so a test may mutate them.

Covers: the detector fires at 3/200 and not at 2/200, refuses a None
measurement, a short ``n_stop_reason`` and a row/summary disagreement, and
reports ``model_length`` apart; ``select`` records per-source provenance and
the pin-ordered union with ``n_selected``, ``cap_from`` / ``cap_to`` and the
prompt bound, and its digest binds the FULL block (a tampered prompt bound no
longer recomputes); ``merge`` recomputes and refuses every adversarial input —
an omitted id, a substituted id, a foreign-store id, edited rows, a mismatched
rung-2 digest, a missing rung-2 item, a smoke PARTIAL, rows off the cap,
differing ``served`` blocks, an identity change outside the admitted keys —
and, on the happy path, gives rung-2 rows precedence with ``attempt 2`` /
``cap_rung 2``, keeps exactly 200 items per (cell, task) with identical sets
across cells, and files ``n_at_cap_rung1`` / ``n_at_cap_residual``.

The fixture builders here (``make_pins``, ``make_rung1``, ``make_rung2``,
``make_certificate``) are shared with ``tests/test_bcb_bfcl_200_analysis.py``.
Pure: no Modal, no network."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pytest

from sisteer.capeval import store as S
from sisteer.capeval.manifest import canonical_json
from sisteer.capeval.tiers import inspect_served as I
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import rung_probe as RP

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "tinycap_ladder.py"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


L = _load_script(SCRIPT, "tinycap_ladder")

# --------------------------------------------------------------------------- #
# the synthetic world (the doc screenshots' neutral domain: two cells, two legs) #
# --------------------------------------------------------------------------- #

CELL_BASE = "none"
CELL_STEERED = "constv2-mix-full__lora_r4_ce_minus_ctl"
CELLS = (CELL_BASE, CELL_STEERED)
BCB, BFCL = L.BCB_TASK, L.BFCL_TASK
#: row task → control atom, the inverse of the contracts' canonical mapping
CONTROL_OF_TASK = {t: atom for atom, rows in C.TASKS_OF_CONTROL.items() for t in rows}
HF = "google/gemma-4-E4B-it"
REV = "ee0ef6023621cff504d758262d4e04895a5af4a2"
TREE = "5039ce98" + "0" * 32
RUN_ID = "11111111-2222-3333-4444-555555555555"
OBJECT_SHA = "be93b469a647248c" + "0" * 48          # the derived object's sha256 (its prefix real; synthetic tail)
CAP_OFF = {BCB: 4096, BFCL: 1024}
SERVED_RUNG1 = (16_384, 36)
SERVED_RUNG2 = (32_768, 16)
#: The lane's REAL pins (slice B) — the fixtures' id vocabulary, in pin order.
BCB_PIN = C.load_bcb_pin(REPO)
BFCL_PIN = C.load_bfcl_pin(REPO)
BCB_IDS: List[str] = list(BCB_PIN["ids"])                       # pinned vocabulary ("BigCodeBench/17")
BCB_LOG: List[str] = [C.bcb_log_id(i) for i in BCB_IDS]         # the rows' / log vocabulary ("17")
BFCL_IDS: List[str] = list(BFCL_PIN["ids"])                     # one vocabulary
#: A BigCodeBench log id the pin does not carry — a foreign store's item.
BCB_LOG_FOREIGN = next(str(k) for k in range(C.BCB_FULL_N) if f"BigCodeBench/{k}" not in set(BCB_IDS))
#: The rung-1 fixture's capped items: BCB pin indices 1..3 (rung-1 prompt_tokens 184 / 188 / 192, so the
#: selection's prompt bound is 192), the steered cell all three and the base cell the first; BFCL two per cell.
CAPPED_BCB: List[str] = BCB_LOG[1:4]
DEFAULT_CAPPED = {(CELL_STEERED, BCB): set(CAPPED_BCB), (CELL_BASE, BCB): {CAPPED_BCB[0]},
                  (CELL_BASE, BFCL): set(BFCL_IDS[0:2]), (CELL_STEERED, BFCL): set(BFCL_IDS[2:4])}
#: One base-cell BFCL item that overflowed the context on rung 1 (`model_length`, not at-cap).
MODEL_LENGTH_BFCL = BFCL_IDS[10]


def write_sampling_validation_marker(root: Path, *, cell: str, run_id: str = RUN_ID, fcid: str = "fc-test", tree_sha: str = TREE,
                                     status: str = C.REC_CLAMP_OVERRIDE_INEFFECTIVE) -> Path:
    """The worker's DURABLE sampling-validation failure at a configuration's out root, in the exact JSON shape
    `tinycap_served.write_sampling_validation_failure` files (status, error, run_id, function_call_id, cell, tree_sha,
    clamp_warning_lines, at, why): the clamp override did not hold, so nothing under ``root`` is protocol evidence."""
    doc = {"status": status, "error": "vLLM's temperature-clamp warning appeared in the server log behind the D13 override",
           "run_id": run_id, "function_call_id": fcid, "cell": cell, "tree_sha": tree_sha,
           "clamp_warning_lines": ["WARNING ... temperature 0.001 is less than 0.01, which may cause numerical errors; clamping"],
           "at": "2026-09-11T20:30:00+00:00",
           "why": ("vLLM's temperature-clamp warning appeared in this server's log behind the D13 override: the BFCL rows executed at 0.01, "
                   "not the protocol's 0.001; every store under this root is refused by the worker's resume (untrusted_store_reason) and "
                   "by the analysis; relaunch into a NEW --out-suffix")}
    p = Path(root) / C.SAMPLING_VALIDATION_FAILED_FILENAME
    p.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    return p


def make_pins(repo_root: Path) -> Path:
    """The lane's REAL pins + library-family map copied under ``repo_root``
    (a temp root the tests may mutate; the contracts' loaders verify them there)."""
    for rel in (C.BCB_PIN_PATH, C.BFCL_PIN_PATH, C.BCB_LIB_FAMILIES_PATH):
        dst = repo_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, dst)
    return repo_root


# --------------------------------------------------------------------------- #
# production-shaped stores                                                     #
# --------------------------------------------------------------------------- #


def _hash8(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def cell_key(cell: str, task: str, ihash: str, chash: str) -> str:
    return f"{cell.replace('__', '--')}__controls__t0__a0__{ihash[:8]}__{chash[:8]}"


def _result(task: str, log_id: str, *, value: float, stop_reason: str, prompt_tokens: int,
            completion_tokens: int, cap: int, answer: Optional[str] = None,
            content: Optional[str] = None) -> I.ItemResult:
    scorer = "verify" if task == BCB else "bfcl_scorer"
    if answer is None:
        answer = ("def task_func():\n    return 1\n" if task == BCB else '[{"name": "f", "arguments": {}}]')
    if content is None:
        content = f"```python\n{answer}```" if task == BCB else answer
    return I.ItemResult(item_id=log_id, value=value, sample_idx=0, seed=I.sample_seed(log_id, 0), answer=answer,
                        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, paired_id=None,
                        scores={scorer: value}, content=content, reasoning=None, stop_reason=stop_reason,
                        unparseable=I.unparseable_from_answer(answer), attempt=1, max_tokens_used=cap)


def _identity(tasks: Any, *, limits: Mapping[str, int], served: Tuple[int, int],
              extra: Optional[Mapping[str, Any]] = None, concurrency: Optional[int] = None,
              serving_engine: Optional[str] = None) -> Dict[str, Any]:
    """The hashed `tinycap` block in the launcher's REAL shape: the driver's
    `concurrency` IS the resolved S (`tinycap_served.expected_identity`), the leg's
    hashed keys are `contracts.item_set_identity(pin)` / `bcb_sandbox_identity()`,
    and the key set is EXACTLY `contracts.tinycap_identity_fields(...)` for the
    legs filed (plus the ladder tuple when `extra` carries it). ``serving_engine``
    ``lora_native`` gives the `served` block the native driver's two facts —
    `serving_engine` (`contracts.SERVED_IDENTITY_NATIVE_FIELDS`, present iff native)
    and the `not_applicable` reduction mode; None / `lens` is byte-identical."""
    tasks = [tasks] if isinstance(tasks, str) else list(tasks)
    has_bfcl = BFCL in tasks
    native = serving_engine == C.SERVING_ENGINE_LORA_NATIVE
    # the producer's vocabularies EXACTLY (`served_capability_run`): `tasks` = sorted CONTROL ATOMS
    # (`sorted(atoms)`), the per-task maps keyed by ROW task (`_resolve_thinking_by_task`, `sampling_identity`)
    ident = {"tasks": sorted(CONTROL_OF_TASK[t] for t in tasks), "thinking": "off",
             "versions": {"tinycap": C.TINYCAP_VERSION},
             "served": {"reasoning_parser": None, "tool_calls": has_bfcl, "max_model_len": served[0],
                        "max_num_seqs": served[1], "concurrency": (served[1] if concurrency is None else concurrency),
                        C.AGENT_LOOP_ENTRY_KEY: has_bfcl,
                        C.REDUCTION_MODE_FIELD: (C.REDUCTION_MODE_NOT_APPLICABLE if native else "stacked"),
                        **({C.SERVED_ENGINE_FIELD: C.SERVING_ENGINE_LORA_NATIVE} if native else {})},
             "limits": dict(limits), "samples": 1,
             "frameworks": {"inspect_ai": "0.3.260", "inspect_evals": C.INSPECT_EVALS_PIN, "client_lock_sha256": "2" * 64},
             C.GEOMETRY_PASS_FIELD: 1, C.IDENTITY_KEY_THINKING_BY_TASK: {t: "off" for t in tasks},
             C.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS: 0, C.IDENTITY_KEY_OR_BENCH_ITEM_SET: None,
             C.IDENTITY_KEY_TEMPERATURE: 1.0, C.IDENTITY_KEY_TOP_P: 0.95, C.IDENTITY_KEY_TOP_K: 64,
             C.IDENTITY_KEY_SAMPLES_BY_TASK: {t: 1 for t in tasks}, C.IDENTITY_KEY_SEED_RULE: C.SEED_RULE_INSPECT}
    if BCB in tasks:
        ident[C.IDENTITY_KEY_BCB_ITEM_SET] = C.item_set_identity(BCB_PIN)
        ident[C.IDENTITY_KEY_SANDBOX] = C.bcb_sandbox_identity()
    if has_bfcl:
        ident[C.IDENTITY_KEY_BFCL_ITEM_SET] = C.item_set_identity(BFCL_PIN)
    if extra:
        ident.update(extra)
    ladder = all(k in ident for k in C.TINYCAP_IDENTITY_LADDER_FIELDS)
    want = set(C.tinycap_identity_fields(files_bigcodebench_200=(BCB in tasks), files_bfcl_200=has_bfcl, ladder=ladder))
    assert set(ident) == want, (sorted(set(ident) ^ want))
    assert tuple(ident["served"]) == C.served_identity_fields(native=native)      # the driver's exact served key set
    return ident


def _bcb_counts(facts: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    """`inspect_served.bcb_summary_counts`'s fields (`contracts.BCB_SUMMARY_COUNT_FIELDS`) over the per-item facts."""
    out = {"n_sandbox_ok": 0, "n_verify_timeout": 0, "n_code_source_python_fence": 0, "n_code_source_bare_fence": 0,
           "n_code_source_whole_completion": 0}
    for f in facts.values():
        if f["sandbox_status"] == "ok":
            out["n_sandbox_ok"] += 1
        elif f["sandbox_status"] == "verify_timeout":
            out["n_verify_timeout"] += 1
        out[f"n_code_source_{f['code_source']}"] += 1
    return out


def _summary(results: Sequence[I.ItemResult], *, task: str, cap: int, wall: float,
             bcb_facts: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    quality = I.task_quality(results)
    tokens = [r.completion_tokens for r in results if r.completion_tokens is not None]
    scored = [r for r in results if not r.unscored]
    return {C.METRIC_ACC: sum(r.value for r in scored) / len(scored), C.METRIC_N: len({r.item_id for r in results}),
            "n_requests": len(results), "n_rejected": 0, "samples_per_item": 1, "task": task,
            "item_set": ("bcb_200" if task == BCB else "bfcl_200"), "framework": C.INSPECT_TASK_VERSION, "engine": "vllm",
            "route": "tinycap-cell", "thinking": "off", "render": C.THINKING_RENDER_NATIVE, "wall_clock_s": round(wall, 3),
            "logs_reused": False, "n_paired": 0, "reasoning_parser": None, "n_content_empty_with_reasoning": 0,
            "n_scored": len({r.item_id for r in scored}), "n_unscored": 0, "n_scored_samples": len(scored),
            "n_unscored_samples": 0, "max_tokens": cap, "temperature": 1.0, "top_p": 0.95, "top_k": 64, **quality,
            "n_think_block_present": 0, "n_think_block_unterminated": 0, "n_unterminated_not_at_cap": 0,
            "think_split_mode": "off", "unterminated_policy": "score_as_is",
            "mean_completion_tokens": (sum(tokens) / len(tokens)) if tokens else None,
            "p99_completion_tokens": (C.nearest_rank_percentile(tokens, 0.99) if tokens else None),
            "n_completion_tokens": len(tokens), **(_bcb_counts(bcb_facts) if bcb_facts is not None else {})}


def _outcomes(results: Sequence[I.ItemResult], *, task: str, summary: Mapping[str, Any]) -> Dict[str, Any]:
    items = [{"item_id": r.item_id, "sample_idx": r.sample_idx, "attempt": r.attempt, "scored": not r.unscored,
              "value": None if r.unscored else float(r.value), "unparseable": r.unparseable, "stop_reason": r.stop_reason,
              "at_cap": bool(r.at_cap), "think_block_present": False, "think_block_unterminated": False,
              "stray_open_tag": False, "content_empty_with_reasoning": False, "prompt_tokens": r.prompt_tokens,
              "completion_tokens": r.completion_tokens, "seed": r.seed} for r in results]
    assert all(set(it) == set(C.TASK_OUTCOME_ITEM_FIELDS) for it in items)
    header = {"task": task, "item_set": summary["item_set"], "thinking": "off", "think_split_mode": summary["think_split_mode"],
              "unterminated_policy": summary["unterminated_policy"], "max_tokens": summary["max_tokens"], "samples_per_item": 1,
              "n_attempted_items": len({r.item_id for r in results}), "n_attempted_samples": len(items),
              "n_scored": summary["n_scored"], "n_unscored": summary["n_unscored"],
              "n_scored_samples": summary["n_scored_samples"], "n_unscored_samples": summary["n_unscored_samples"],
              "n_stop_reason": summary["n_stop_reason"], "n_at_cap": summary["n_at_cap"],
              "n_think_block_present": 0, "n_think_block_unterminated": 0, "n_unterminated_not_at_cap": 0,
              "n_completion_tokens": summary["n_completion_tokens"], "producer_tree_sha": TREE, "items": items}
    assert set(header) == set(C.TASK_OUTCOMES_HEADER_FIELDS)
    return header


def make_multi_store(rung_root: Path, cell: str, results_by_task: Mapping[str, Sequence[I.ItemResult]], *,
                     caps: Mapping[str, int], served: Tuple[int, int], limits: Mapping[str, int], n_expected_full: int = 200,
                     identity_extra: Optional[Mapping[str, Any]] = None,
                     bcb_extras: Optional[Callable[[I.ItemResult], Dict[str, Any]]] = None,
                     manifest_extra: Optional[Mapping[str, Any]] = None, wall: float = 900.0,
                     concurrency: Optional[int] = None, model_revision: str = REV, tree_sha: str = TREE,
                     intervention_hash: Optional[str] = None, object_sha256: Optional[str] = None,
                     route_extra: Optional[Mapping[str, Any]] = None, serving_engine: Optional[str] = None) -> Path:
    """One CELL store under ``<rung_root>/<cell>/cells/<key>/`` filing every task
    of ``results_by_task`` (the normal launch files both legs in ONE configuration
    per cell; a fired-leg rerun files one) through the store side's own writers:
    rows via ``CellStore.append_row`` with ``extra_json = canonical_json(ItemResult.extra(...))``,
    the manifest via ``write_cell_manifest``, per task a summary / outcomes / marker
    via ``completion``'s writers (the marker rule mirrors ``finish_task``: DONE iff
    unlimited and ``n_items == n_expected_full``, else PARTIAL naming the limits).

    ``serving_engine`` (the native-merge regression's stores were LENS-shaped —
    a fixture that did not match the producer): ``lora_native`` writes the
    manifest AS ``served_capability_run`` writes a native store's — the engine
    stamped in every spelling the reader consults (top-level, the ``route``
    block, the hashed ``resolved_config.serving`` and ``tinycap.served``), the
    top-level ``serving_identity`` / parsed ``serving_path``, the steered route
    ``proxy_vllm+lora+…`` (the base cell passthrough on both engines), both reduction
    modes ``not_applicable`` and the route's ``reduction_mode_expectation: not_consulted``.
    None / ``lens`` is the historical lens manifest, byte for byte."""
    tasks = list(results_by_task)
    if serving_engine not in (None, C.SERVING_ENGINE_LENS, C.SERVING_ENGINE_LORA_NATIVE):
        raise ValueError(f"serving_engine {serving_engine!r} is not one of {list(C.SERVING_ENGINES)}")
    native = serving_engine == C.SERVING_ENGINE_LORA_NATIVE
    ident = _identity(tasks, limits=limits, served=served, extra=identity_extra, concurrency=concurrency,
                      serving_engine=serving_engine)
    ihash = intervention_hash or _hash8("intervention", cell)
    chash = _hash8("harness", "+".join(tasks), canonical_json(ident))
    key = cell_key(cell, "controls", ihash, chash)
    store = S.CellStore(rung_root / cell, key)
    obj_sha = object_sha256 if object_sha256 is not None else (None if cell == CELL_BASE else OBJECT_SHA)
    has_bfcl = BFCL in tasks
    template = {"sweep": "tinycap", "run_id": RUN_ID, "model_id": HF, "model_revision": model_revision, "arm_name": cell,
                "kind": ("none" if cell == CELL_BASE else "lora_subtraction"),
                "coeff": (None if cell == CELL_BASE else 1.0), "intervention_hash": ihash, "engine": "vllm",
                "trial_id": 0, "seed": 0, "attempt": 0, "harness_config_hash": chash, "env_digest": "3" * 16,
                "slurm_job_id": None}
    # the producer's `<backend>+<path>+<mode>`: passthrough on the base cell (both engines), lens / lora on a steered one
    serving_identity = ("proxy_vllm+passthrough+interleaved" if cell == CELL_BASE
                        else f"proxy_vllm+{'lora' if native else 'lens'}+interleaved")
    engine_stamp = {C.SERVED_ENGINE_FIELD: C.SERVING_ENGINE_LORA_NATIVE} if native else {}
    manifest = {"intervention_hash": ihash, "harness_config_hash": chash, "model_revision": model_revision, "engine": "vllm",
                "model_id": HF, "arm_name": cell, "kind": template["kind"], "row_constants": template,
                # the native driver's top-level facts (`served_capability_run`: serving_path = the parsed identity's path,
                # serving_identity, the UNHASHED engine stamp present iff native; the hashed copy in resolved_config.serving)
                **({"serving_path": "passthrough" if cell == CELL_BASE else "lora", "serving_identity": serving_identity,
                    **engine_stamp,
                    "resolved_config": {"serving": {"virtual_model": f"tinycap-{cell}", "serving_identity": serving_identity,
                                                    **engine_stamp}}} if native else {}),
                C.MANIFEST_KEY_TINYCAP: ident,
                C.MANIFEST_KEY_GEOMETRY: {"max_model_len": served[0], "max_num_seqs": served[1], C.GEOMETRY_PASS_FIELD: 1,
                                          "source": "endpoint.json"},
                C.MANIFEST_KEY_ROUTE: {"virtual_model": f"tinycap-{cell}", "grader_virtual_model": None,
                                       "serving_identity": serving_identity,
                                       "tool_calls": has_bfcl,
                                       C.AGENT_LOOP_ENTRY_KEY: has_bfcl,
                                       C.REDUCTION_MODE_FIELD: (C.REDUCTION_MODE_NOT_APPLICABLE if native else "stacked"),
                                       # the native route block's two extra keys (the engine, and the lens expectation not consulted)
                                       **({**engine_stamp, "reduction_mode_expectation": "not_consulted"} if native else {}),
                                       C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: obj_sha,
                                       C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: ({cell: obj_sha} if obj_sha else None),
                                       **(dict(route_extra) if route_extra else {})},
                C.MANIFEST_KEY_PRODUCER: {"tree_sha": tree_sha, "driver": "served_capability_run"},
                C.MANIFEST_KEY_SAMPLING: {"mode": "card", "model_key": "gemma4e4b", "temperature": 1.0, "top_p": 0.95,
                                          "top_k": 64, "samples_by_task": {t: 1 for t in tasks}, "seed_rule": C.SEED_RULE_INSPECT,
                                          "source": "model card"},
                C.MANIFEST_KEY_THINKING: {"enable_thinking": False, "render": C.THINKING_RENDER_NATIVE,
                                          "reasoning_parser": None, "mode": "off", "by_task": {t: "off" for t in tasks}},
                # EvalPlus's sandbox block: the producer writes None on an Inspect-only run (`_sandbox_block(families)`)
                C.MANIFEST_KEY_SANDBOX: None}
    if manifest_extra:
        manifest.update(manifest_extra)
    store.write_cell_manifest(manifest)
    t0 = time.time()
    for task, results in results_by_task.items():
        cap = caps[task]
        facts: Dict[str, Dict[str, Any]] = {}
        if task == BCB:
            for r in results:
                facts[r.item_id] = (bcb_extras(r) if bcb_extras else
                                    {"code_source": "python_fence", "sandbox_status": "ok", "verifier_explanation": None})
        for k, r in enumerate(results):
            more: Dict[str, Any] = {"thinking": "off", "max_tokens": cap}
            if task == BCB:
                more.update(facts[r.item_id])
            store.append_row(S.ResultRow(**template, tier=C.ROW_TIER, task=task, task_version=C.INSPECT_TASK_VERSION,
                                         domain=C.DOMAIN_INSPECT_GEN, item_id=r.item_id, position_order=k, metric=C.METRIC_ACC,
                                         value=float(r.value), extra_json=canonical_json(r.extra(**more)),
                                         created_at=t0 + k, wall_clock_s=None))
        summary = _summary(results, task=task, cap=cap, wall=wall, bcb_facts=(facts if task == BCB else None))
        outcomes = _outcomes(results, task=task, summary=summary)
        for metric, value, extra in ((C.METRIC_ACC_AGGREGATE, summary[C.METRIC_ACC], {"n": summary[C.METRIC_N], **I.task_quality(results)}),
                                     (C.METRIC_N, float(summary[C.METRIC_N]), None)):
            store.append_row(S.ResultRow(**template, tier=C.ROW_TIER, task=task, task_version=C.INSPECT_TASK_VERSION,
                                         domain=C.DOMAIN_INSPECT_GEN, item_id=C.AGG_ITEM_ID, position_order=None, metric=metric,
                                         value=value, extra_json=(canonical_json(extra) if extra else None),
                                         created_at=t0 + len(results), wall_clock_s=wall))
        completion.verify_outcomes_against_summary(outcomes, summary)
        completion.write_task_summary(store.cell_dir, task, summary)
        completion.write_task_outcomes(store.cell_dir, task, outcomes)
        n_items = len({r.item_id for r in results})
        base = {"task": task, "n_items": n_items, "n_rows": len(results) + 2, "n_requests": len(results), "n_rejected": 0,
                "wall_clock_s": wall, "finished_at": t0 + wall}
        if not limits and n_items == n_expected_full:
            completion.write_done_marker(store.cell_dir, task, base)
        else:
            completion.write_partial_marker(store.cell_dir, task, {**base, "n_expected_full": n_expected_full, "limits": dict(limits)})
    return store.cell_dir


def make_store(rung_root: Path, cell: str, task: str, *, results: Sequence[I.ItemResult], cap: int, **kw: Any) -> Path:
    """A ONE-task store (a fired-leg rerun's shape): :func:`make_multi_store` over ``{task: results}``."""
    return make_multi_store(rung_root, cell, {task: results}, caps={task: cap}, **kw)


def default_value(cell: str, task: str, i: int) -> float:
    """A deterministic paired pattern: the base passes 2 of 3; the steered cell
    flips a few passes to fails and a few fails to passes."""
    base = i % 3 != 0
    if cell == CELL_BASE:
        return float(base)
    if base and i % 7 == 1:
        return 0.0
    if not base and i % 11 == 5:
        return 1.0
    return float(base)


def rung1_results(cell: str, task: str, *, capped: Iterable[str] = (), model_length: Iterable[str] = (),
                  values: Callable[[str, str, int], float] = default_value,
                  cap: Optional[int] = None) -> List[I.ItemResult]:
    cap = cap or CAP_OFF[task]
    pin = BCB_IDS if task == BCB else BFCL_IDS
    capped, ml = set(capped), set(model_length)
    out = []
    for i, pid in enumerate(pin):
        lid = L.to_log_id(task, pid)
        if lid in capped:
            reason, ctoks, v = L.STOP_REASON_AT_CAP, cap, 0.0
        elif lid in ml:
            reason, ctoks, v = L.STOP_REASON_MODEL_LENGTH, 7, 0.0
        else:
            reason, ctoks, v = "stop", 40 + (i % 50), values(cell, task, i)
        out.append(_result(task, lid, value=v, stop_reason=reason, prompt_tokens=180 + 4 * i, completion_tokens=ctoks, cap=cap))
    return out


def make_rung1(root: Path, *, capped: Optional[Mapping[Tuple[str, str], Iterable[str]]] = None,
               model_length: Optional[Mapping[Tuple[str, str], Iterable[str]]] = None,
               served: Tuple[int, int] = SERVED_RUNG1, tasks: Sequence[str] = (BCB, BFCL),
               values: Callable[[str, str, int], float] = default_value, combined: bool = False,
               store_kwargs_by_cell: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Path:
    """A pulled rung-1 dir: both cells × both legs, DONE markers, no limits —
    ``combined`` files both legs in ONE store per cell (the normal launch's
    shape), else one store per (cell, task)."""
    capped, model_length = dict(capped or {}), dict(model_length or {})
    for cell in CELLS:
        by_task = {task: rung1_results(cell, task, capped=capped.get((cell, task), ()),
                                       model_length=model_length.get((cell, task), ()), values=values) for task in tasks}
        kw = dict((store_kwargs_by_cell or {}).get(cell, {}))
        if combined:
            make_multi_store(root, cell, by_task, caps={t: CAP_OFF[t] for t in tasks}, served=served, limits={}, **kw)
        else:
            for task, res in by_task.items():
                make_store(root, cell, task, results=res, cap=CAP_OFF[task], served=served, limits={}, **kw)
    return root


def make_rung2(root: Path, selection: Mapping[str, Any], *, served: Tuple[int, int] = SERVED_RUNG2,
               residual: Optional[Mapping[Tuple[str, str], Iterable[str]]] = None,
               values: Callable[[str, str, int], float] = default_value, cells: Sequence[str] = CELLS,
               limits: Mapping[str, int] = L.LIMITS_LADDER, identity_override: Optional[Mapping[str, Any]] = None,
               extra_ids: Optional[Mapping[Tuple[str, str], Iterable[str]]] = None,
               row_cap: Optional[int] = None, served_by_cell: Optional[Mapping[str, Tuple[int, int]]] = None,
               store_kwargs: Optional[Mapping[str, Any]] = None,
               store_kwargs_by_cell: Optional[Mapping[str, Mapping[str, Any]]] = None, combined: bool = False) -> Path:
    """A pulled rung-2 dir: the fired tasks' selection PARTIALs per cell at
    ``cap_to`` carrying the ladder identity (``cap_overrides_by_task`` +
    ``item_selection``) — ``combined`` files every fired task in ONE store per
    cell (a `--controls <both fired>` rerun), else one store per (cell, task);
    ``residual`` names items still at the cap on rung 2."""
    residual, extra_ids = dict(residual or {}), dict(extra_ids or {})
    ident_extra = {"cap_overrides_by_task": {t: b["cap_to"] for t, b in selection["selection"].items()},
                   "item_selection": {t: {"n": b["n"], "ids_sha256": b["ids_sha256"]} for t, b in selection["selection"].items()}}
    if identity_override:
        ident_extra.update(identity_override)
    for cell in cells:
        by_task: Dict[str, List[I.ItemResult]] = {}
        caps: Dict[str, int] = {}
        for task, block in selection["selection"].items():
            cap_to = block["cap_to"]
            ids = [L.to_log_id(task, i) for i in block["ids"]] + list(extra_ids.get((cell, task), ()))
            res = []
            for k, lid in enumerate(ids):
                still = lid in set(residual.get((cell, task), ()))
                res.append(_result(task, lid, value=(0.0 if still else values(cell, task, k + 1)),
                                   stop_reason=(L.STOP_REASON_AT_CAP if still else "stop"), prompt_tokens=200 + 4 * k,
                                   completion_tokens=(cap_to if still else 900 + k), cap=(row_cap or cap_to)))
            by_task[task], caps[task] = res, cap_to
        kw = dict(served=(served_by_cell or {}).get(cell, served), limits=limits, identity_extra=ident_extra,
                  **dict(store_kwargs or {}), **dict((store_kwargs_by_cell or {}).get(cell, {})))
        if combined:
            make_multi_store(root, cell, by_task, caps=caps, **kw)
        else:
            for task, res in by_task.items():
                make_store(root, cell, task, results=res, cap=caps[task], **kw)
    return root


def make_certificate(path: Path, selection: Mapping[str, Any], *, accepted: Tuple[int, int] = SERVED_RUNG2,
                     rung_source: str = "probe", seqs_source: str = "override", suffix: str = "bcbbfcl200-5039ce9-cap16k",
                     selection_sha256: Optional[str] = None, caps: Optional[Mapping[str, int]] = None,
                     walked: Optional[List[Dict[str, Any]]] = None, rev: str = REV, tree_sha: str = TREE,
                     object_digests: Optional[Mapping[str, str]] = None) -> Path:
    cert = {"schema": RP.CERTIFICATE_SCHEMA, "suffix": suffix,
            "accepted": {"max_model_len": accepted[0], "max_num_seqs": accepted[1], "seqs_source": seqs_source,
                         "rung_source": rung_source},
            "walked": walked if walked is not None else ([] if rung_source == "operator" else [
                {"S": 18, "outcome": "refused", "kv_tokens": 528_000,
                 "refusal_tail": "KV cache holds 528000 tokens < the 589824-token budget (= 18 x 32768)"},
                {"S": 16, "outcome": "accepted", "kv_tokens": None, "refusal_tail": None}]),
            "tree_sha": tree_sha, "model_slug": "gemma4e4b-sdfbase", "hf": HF, "rev": rev,
            "cfg_path": "configs/two_si_gemma4e4b_sdfbase.yaml", "cfg_sha256": "4" * 64,
            "object_digests": dict(object_digests) if object_digests is not None else {CELL_STEERED: OBJECT_SHA},
            "selection_sha256": selection_sha256 or selection["selection_sha256"],
            "cap_overrides_by_task": dict(caps) if caps is not None else {t: b["cap_to"] for t, b in selection["selection"].items()},
            "budget": {"batched_token_budget": 589_824, "budget_source": "measured"},
            # bcb-bfcl-200-widen S6 (schema 2): the ladder walked (the row's model) and the re-probe origin
            "ladder": RP.ladder_spec("gemma4e4b").block(),
            "probed_cell": CELL_STEERED, "probe_run_id": RUN_ID, "standup_s": 331.4, "reprobed_from": None,
            "at": "2026-09-10T18:00:00+00:00"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cert, indent=1, sort_keys=True), encoding="utf-8")
    return path


def retarget_selection(sel: Mapping[str, Any], task: str, ids: List[str]) -> Dict[str, Any]:
    """A selection whose ``ids`` of ``task`` were replaced, with every digest
    recomputed CONSISTENTLY — the adversary who edits the file properly."""
    doc = copy.deepcopy(dict(sel))
    block = doc["selection"][task]
    block["ids"] = list(ids)
    block["n"] = block["n_selected"] = len(ids)
    block["ids_sha256"] = C.ids_sha256(ids)
    doc["selection_sha256"] = L.selection_digest(doc["selection"])
    return doc


@pytest.fixture()
def world(tmp_path):
    repo = make_pins(tmp_path / "repo")
    rung1 = make_rung1(tmp_path / "rung1", capped=DEFAULT_CAPPED, model_length={(CELL_BASE, BFCL): {MODEL_LENGTH_BFCL}})
    return {"repo": repo, "rung1": rung1, "tmp": tmp_path}


# --------------------------------------------------------------------------- #
# vocabulary pins                                                              #
# --------------------------------------------------------------------------- #


def test_task_names_and_marker_rule_are_the_contracts():
    assert L.BCB_TASK == C.inspect_task("bigcodebench_200") and L.BFCL_TASK == C.inspect_task("bfcl_200")
    assert L.marker_filename(BCB) == C.marker_filename(BCB) == "inspect__bigcodebench_200.json"


def test_bcb_id_round_trip_and_refusals():
    assert L.to_pinned_id(BCB, "17") == "BigCodeBench/17" and L.to_log_id(BCB, "BigCodeBench/17") == "17"
    for bad in ("", "017", "BigCodeBench/", "BigCodeBench/17", "HumanEval/3", "17a"):
        with pytest.raises(L.LadderError):
            L.to_pinned_id(BCB, bad)
    for bad in ("17", "BigCodeBench/017", "BigCodeBench/", "HumanEval/3"):
        with pytest.raises(L.LadderError):
            L.to_log_id(BCB, bad)
    assert L.to_pinned_id(BFCL, "simple_python_0") == "simple_python_0" == L.to_log_id(BFCL, "simple_python_0")


def test_pins_load_and_refuse_drift(world):
    bcb, bfcl = L.load_pin(world["repo"], BCB), L.load_pin(world["repo"], BFCL)
    assert bcb["ids"] == BCB_IDS and bfcl["ids"] == BFCL_IDS and len(bcb["libs_by_id"]) == 200
    p = L.pin_path(world["repo"], BFCL)
    doc = json.loads(p.read_text())
    doc["population"]["sql"].append(BFCL_IDS[0])            # an id in two categories: the population no longer regenerates
    p.write_text(json.dumps(doc))
    with pytest.raises(L.LadderError, match="population|regenerate"):
        L.load_pin(world["repo"], BFCL)
    p = L.pin_path(world["repo"], BCB)
    doc = json.loads(p.read_text())
    doc["ids"][0] = f"BigCodeBench/{BCB_LOG_FOREIGN}"          # a moved id is another instrument
    p.write_text(json.dumps(doc))
    with pytest.raises(L.LadderError, match="regenerate"):
        L.load_pin(world["repo"], BCB)


# --------------------------------------------------------------------------- #
# the detector                                                                 #
# --------------------------------------------------------------------------- #


def test_detector_fires_at_3_of_200_and_not_at_2(world):
    det = L.detect(world["rung1"])
    bcb, bfcl = det["tasks"][BCB], det["tasks"][BFCL]
    assert bcb["fired"] is True and bcb["cells"][CELL_STEERED]["n_at_cap"] == 3 and bcb["cells"][CELL_BASE]["n_at_cap"] == 1
    assert bfcl["fired"] is False and {r["n_at_cap"] for r in bfcl["cells"].values()} == {2}
    assert all(r["n_stop_reason"] == 200 and r["n"] == 200 for t in det["tasks"].values() for r in t["cells"].values())
    assert det["fired_any"] is True and det["cells"] == sorted(CELLS)
    # model_length is reported beside, never counted as at-cap
    assert bfcl["cells"][CELL_BASE]["n_model_length"] == 1 and bfcl["cells"][CELL_STEERED]["n_model_length"] == 0


def test_leg_fires_is_strictly_greater_than_one_percent():
    r = {"a": {"n_at_cap": 2, "n_stop_reason": 200}, "b": {"n_at_cap": 0, "n_stop_reason": 200}}
    assert L.leg_fires(r) is False
    r["b"]["n_at_cap"] = 3
    assert L.leg_fires(r) is True


def test_detector_refuses_a_none_measurement(world):
    d = L.store_dir_for(world["rung1"], CELL_BASE, BCB)
    p = d / C.SUMMARY_DIR / L.marker_filename(BCB)
    s = json.loads(p.read_text())
    s["n_at_cap"] = None
    p.write_text(json.dumps(s))
    op = d / C.TASK_OUTCOMES_DIR / L.marker_filename(BCB)
    o = json.loads(op.read_text())
    o["n_at_cap"] = None                                     # the producer measured nothing, consistently
    op.write_text(json.dumps(o))
    with pytest.raises(L.LadderError, match="did not measure"):
        L.detect(world["rung1"])


def test_detector_refuses_a_short_stop_reason_count(world):
    d = L.store_dir_for(world["rung1"], CELL_STEERED, BFCL)
    p = d / C.SUMMARY_DIR / L.marker_filename(BFCL)
    s = json.loads(p.read_text())
    s["n_stop_reason"] = 199
    p.write_text(json.dumps(s))
    op = d / C.TASK_OUTCOMES_DIR / L.marker_filename(BFCL)
    o = json.loads(op.read_text())
    o["n_stop_reason"] = 199                                 # consistently short: one response without a stop reason
    op.write_text(json.dumps(o))
    with pytest.raises(L.LadderError, match="n_stop_reason 199 != 200"):
        L.detect(world["rung1"])


def test_detector_refuses_a_row_summary_disagreement(world):
    d = L.store_dir_for(world["rung1"], CELL_BASE, BCB)
    rp = d / "rows.jsonl"
    lines = rp.read_text().splitlines()
    for i, line in enumerate(lines):
        row = json.loads(line)
        if row["metric"] == "acc" and row["item_id"] == BCB_LOG[4]:
            extra = json.loads(row["extra_json"])
            extra["stop_reason"] = "max_tokens"                 # a row that says at-cap; the summary does not
            row["extra_json"] = canonical_json(extra)
            lines[i] = json.dumps(row, sort_keys=True)
            break
    rp.write_text("\n".join(lines) + "\n")
    with pytest.raises(L.LadderError, match="at-cap disagreement"):
        L.detect(world["rung1"])


def test_detector_refuses_a_store_without_outcomes(world):
    d = L.store_dir_for(world["rung1"], CELL_BASE, BFCL)
    (d / C.TASK_OUTCOMES_DIR / L.marker_filename(BFCL)).unlink()
    with pytest.raises(L.LadderError, match="all-attempted record"):
        L.detect(world["rung1"])


def test_store_dir_refuses_zero_or_two_stores(world, tmp_path):
    with pytest.raises(L.LadderError, match="no store carries"):
        L.store_dir_for(world["rung1"], CELL_BASE, "inspect/math")
    other = make_store(tmp_path / "dup", CELL_BASE, BCB, results=rung1_results(CELL_BASE, BCB), cap=4096,
                       served=SERVED_RUNG1, limits={}, identity_extra={"samples": 2})
    shutil.copytree(other, world["rung1"] / CELL_BASE / "cells" / other.name)
    with pytest.raises(L.LadderError, match="2 stores carry"):
        L.store_dir_for(world["rung1"], CELL_BASE, BCB)


# --------------------------------------------------------------------------- #
# select                                                                       #
# --------------------------------------------------------------------------- #


def test_select_records_provenance_and_the_pin_ordered_union(world):
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    assert set(doc["selection"]) == {BCB}                     # BFCL did not fire
    assert doc["fired_any"] and doc["detector"][BFCL]["fired"] is False
    block = doc["selection"][BCB]
    # the union of the steered cell's three and the base cell's one, canonical PINNED ids, pin order
    assert block["ids"] == BCB_IDS[1:4]
    assert block["n"] == block["n_selected"] == 3
    assert block["ids_sha256"] == C.ids_sha256(block["ids"])
    assert block["cap_from"] == 4096 and block["cap_to"] == 16_384
    # the prompt bound: the largest rung-1 prompt among the selected items (pin index 1, 2, 3 → 180 + 4·3 = 192)
    assert block["max_prompt_tokens_rung1"] == 192
    assert doc["selection_sha256"] == L.selection_digest(doc["selection"])
    srcs = {(s["cell"], s["task"]): s for s in doc["sources"]}
    assert set(srcs) == {(CELL_BASE, BCB), (CELL_STEERED, BCB)}
    st = srcs[(CELL_STEERED, BCB)]
    assert st["capped_ids"] == BCB_IDS[1:4] and st["n_at_cap"] == 3
    assert st["cap"] == 4096 and st["capped_ids_sha256"] == C.ids_sha256(st["capped_ids"])
    assert st["identity"]["served"] == {"reasoning_parser": None, "tool_calls": False, "max_model_len": 16_384,
                                        "max_num_seqs": 36, "concurrency": 36, C.AGENT_LOOP_ENTRY_KEY: False,
                                        C.REDUCTION_MODE_FIELD: "stacked"}
    assert st["identity"][C.IDENTITY_KEY_BCB_ITEM_SET] == C.item_set_identity(BCB_PIN)
    assert st["identity"][C.IDENTITY_KEY_SANDBOX] == C.bcb_sandbox_identity()
    for k in L.SHA_KEYS:
        assert len(st[k]) == 64
    d = L.store_dir_for(world["rung1"], CELL_STEERED, BCB)
    assert st["store_key"] == d.name and st["rows_sha256"] == L.sha256_file(d / "rows.jsonl")
    out = L.write_selection(doc, world["tmp"] / "ladder_selection.json")
    assert L.load_selection(out)["selection_sha256"] == doc["selection_sha256"]


def test_selection_digest_binds_the_prompt_bound(world):
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    p = L.write_selection(doc, world["tmp"] / "sel.json")
    tampered = json.loads(p.read_text())
    tampered["selection"][BCB]["max_prompt_tokens_rung1"] = 100          # a bound that would pass a stricter fit guard
    p.write_text(json.dumps(tampered))
    with pytest.raises(L.LadderError, match="selection_sha256 does not recompute"):
        L.load_selection(p)
    assert L.selection_digest(tampered["selection"]) != doc["selection_sha256"]


def test_select_refuses_a_capped_item_outside_the_pin(world):
    d = L.store_dir_for(world["rung1"], CELL_STEERED, BCB)
    op = d / C.TASK_OUTCOMES_DIR / L.marker_filename(BCB)
    o = json.loads(op.read_text())
    for it in o["items"]:
        if it["item_id"] == CAPPED_BCB[0]:
            it["item_id"] = BCB_LOG_FOREIGN                     # a BigCodeBench id the pin does not carry
    op.write_text(json.dumps(o))
    rp = d / "rows.jsonl"
    lines = []
    for line in rp.read_text().splitlines():
        row = json.loads(line)
        if row["metric"] == "acc" and row["item_id"] == CAPPED_BCB[0]:
            row["item_id"] = BCB_LOG_FOREIGN
        lines.append(json.dumps(row, sort_keys=True))
    rp.write_text("\n".join(lines) + "\n")
    with pytest.raises(L.LadderError, match="foreign store"):
        L.build_selection(world["rung1"], repo_root=world["repo"])


def test_select_with_no_fired_leg_writes_an_empty_selection(tmp_path):
    repo = make_pins(tmp_path / "repo")
    rung1 = make_rung1(tmp_path / "r1", capped={(CELL_BASE, BCB): set(BCB_LOG[0:2])})
    doc = L.build_selection(rung1, repo_root=repo)
    assert doc["selection"] == {} and doc["fired_any"] is False and doc["sources"] == []


def test_select_follows_the_waves_runnable_legs_a_bcb_only_wave_selects_its_fired_bcb_items(tmp_path):
    """`select --rung1 …` on the BCB-only nemotron wave (BFCL blocked until the nemotron_h live parser PASS of
    2026-09-23; since then the shipped roster's BCB-only waves are the llama ones — skipped for Llama with no unblock
    path — so this case reads `llama8b_main`) REFUSED "no store carries summary/inspect__bfcl_200.json"
    — `detect` walked LADDER_TASKS unconditionally and `build_selection` called it before looking at `fired`. The selector's leg
    set is now the wave's RUNNABLE legs: from the roster (`--roster/--wave`), else from the rung-1 analysis.json beside the
    stores, else both legs (the two-leg default, unchanged). A runnable leg without a store still refuses (naming the leg set and
    the way out); a store carrying a leg the roster does not run refuses as a mismatch; rung 2 carries only the fired leg and merges."""
    repo = make_pins(tmp_path / "repo")
    capped = {k: v for k, v in DEFAULT_CAPPED.items() if k[1] == BCB}
    bcb_only = make_rung1(tmp_path / "bcbw-llama8b_main-0f5414a", capped=capped, tasks=(BCB,))
    roster = L.REPO / "configs" / "capeval" / "bcb_bfcl_200_widen_roster.yaml"
    # fails before: both legs assumed → the BFCL store is demanded; the refusal now names the leg set and the way out
    with pytest.raises(L.LadderError, match="no store carries summary/inspect__bfcl_200.json .*--roster/--wave"):
        L.detect(bcb_only)
    # (1) the roster: the llama wave's bfcl_200 leg is blocked (instrument failure; skipped for Llama, no unblock
    # path) → the leg set is BCB alone
    tasks, source = L.ladder_tasks(bcb_only, roster=roster, wave="llama8b_main")
    assert tasks == [BCB] and source.startswith("roster ") and "runnable legs ['bigcodebench_200']" in source
    det = L.detect(bcb_only, tasks=tasks, tasks_source=source)
    assert list(det["tasks"]) == [BCB] and det["legs"] == [BCB] and det["legs_not_run"] == [BFCL] and det["fired_any"] and det["tasks"][BCB]["fired"]
    doc = L.build_selection(bcb_only, repo_root=repo, tasks=tasks, tasks_source=source)
    assert list(doc["selection"]) == [BCB] and doc["selection"][BCB]["n_selected"] == 3 and doc["legs"] == [BCB] and doc["legs_source"] == source
    assert {s["task"] for s in doc["sources"]} == {BCB} and doc["legs_not_run"] == [BFCL]
    # the selection round-trips through the contracts' reader; rung 2 carries only the fired leg's items and merges
    sel_path = L.write_selection(doc, tmp_path / "ladder_selection.json")
    assert L.load_selection(sel_path)["legs"] == [BCB]
    rung2 = make_rung2(tmp_path / "rung2", doc)
    cert = make_certificate(tmp_path / "rung2" / "_ladder" / "rung_probe.json", doc)
    summ = L.merge(bcb_only, rung2, sel_path, cert, tmp_path / "merged", repo_root=repo)
    assert summ["tasks"] == [BCB] and set(summ["per_cell"]) == set(CELLS) and all(list(v) == [BCB] for v in summ["per_cell"].values())
    # (2) the analysis.json beside the stores names the runnable legs (what --analyze wrote); (3) neither → both legs, and it says so
    (bcb_only / L.ANALYSIS_FILENAME).write_text(json.dumps({"schema": 2, "lane": "bcb-bfcl-200-widen", "tasks": [BCB],
                                                            "blocked_legs": {BFCL: {"status": "blocked", "reason": "instrument failure (D79): the pinned Llama template renders no tool definitions"}}}))
    tasks2, source2 = L.ladder_tasks(bcb_only)
    assert tasks2 == [BCB] and source2.startswith(L.ANALYSIS_FILENAME)
    assert L.build_selection(bcb_only, repo_root=repo, tasks=tasks2, tasks_source=source2)["selection"][BCB]["ids"] == doc["selection"][BCB]["ids"]
    (bcb_only / L.ANALYSIS_FILENAME).write_text(json.dumps({"schema": 2, "tasks": ["inspect/mmlu"]}))
    with pytest.raises(L.LadderError, match="not a non-empty list of ladder legs"):
        L.ladder_tasks(bcb_only)
    (bcb_only / L.ANALYSIS_FILENAME).unlink()
    tasks3, source3 = L.ladder_tasks(bcb_only)
    assert tasks3 == list(L.LADDER_TASKS) and source3 == L.DEFAULT_LEGS_SOURCE
    # the CLI: --roster/--wave select the fired BCB items; without them the refusal names the way out; --roster alone refuses
    rc = L.main(["select", "--rung1", str(bcb_only), "--out", str(tmp_path / "s.json"), "--repo-root", str(repo),
                 "--roster", str(roster), "--wave", "llama8b_main"])
    assert rc == 0 and json.loads((tmp_path / "s.json").read_text())["legs"] == [BCB]
    with pytest.raises(L.LadderError, match="no store carries summary/inspect__bfcl_200.json .*--roster/--wave"):
        L.main(["select", "--rung1", str(bcb_only), "--out", str(tmp_path / "s2.json"), "--repo-root", str(repo)])
    with pytest.raises(L.LadderError, match="--roster and --wave come together"):
        L.main(["detect", "--rung1", str(bcb_only), "--roster", str(roster)])
    with pytest.raises(L.LadderError, match="has no wave 'no_such_wave'"):
        L.ladder_tasks(bcb_only, roster=roster, wave="no_such_wave")
    # a runnable leg without a store still refuses; a store carrying a leg the roster does not run is a mismatch
    with pytest.raises(L.LadderError, match="no store carries summary/inspect__bfcl_200.json for cell .*\\(leg set .* from test: both runnable\\)$"):
        L.detect(bcb_only, tasks=[BCB, BFCL], tasks_source="test: both runnable")
    both = make_rung1(tmp_path / "both", capped=DEFAULT_CAPPED)
    with pytest.raises(L.LadderError, match="carries a store for inspect/bfcl_200 .*the roster does not run that leg"):
        L.detect(both, tasks=[BCB], tasks_source="roster test")
    with pytest.raises(L.LadderError, match="not a non-empty subset of the ladder legs"):
        L.detect(both, tasks=["inspect/mmlu"], tasks_source="test")
    # the two-leg waves are unchanged: the default leg set is both legs, and the roster's gemma wave (both runnable) agrees
    assert L.ladder_tasks(both, roster=roster, wave="gemma4e4b_it")[0] == list(L.LADDER_TASKS)
    det_both = L.detect(both)
    assert list(det_both["tasks"]) == list(L.LADDER_TASKS) and det_both["legs_not_run"] == [] and det_both["legs_source"] == L.DEFAULT_LEGS_SOURCE


# --------------------------------------------------------------------------- #
# merge — the happy path                                                       #
# --------------------------------------------------------------------------- #


def _select_and_rung2(world, **rung2_kwargs):
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    sel_path = L.write_selection(doc, world["tmp"] / "ladder_selection.json")
    rung2 = make_rung2(world["tmp"] / "rung2", doc, **rung2_kwargs)
    cert = make_certificate(world["tmp"] / "rung2" / "_ladder" / "rung_probe.json", doc)
    return doc, sel_path, rung2, cert


def test_merge_gives_rung2_precedence_and_files_both_readings(world):
    doc, sel_path, rung2, cert = _select_and_rung2(world, residual={(CELL_STEERED, BCB): {CAPPED_BCB[2]}})
    out = world["tmp"] / "merged"
    summ = L.merge(world["rung1"], rung2, sel_path, cert, out, repo_root=world["repo"])
    assert summ["selection_sha256"] == doc["selection_sha256"] and summ["tasks"] == [BCB]
    assert summ["certificate"]["accepted"] == {"max_model_len": 32_768, "max_num_seqs": 16, "seqs_source": "override",
                                               "rung_source": "probe"}
    for cell in CELLS:
        r = summ["per_cell"][cell][BCB]
        assert r["n"] == 200 and r["n_selected"] == 3 and r["cap_from"] == 4096 and r["cap_to"] == 16_384
        rows = [json.loads(l) for l in Path(r["merged_rows"]).read_text().splitlines()]
        assert len(rows) == 200 and len({x["item_id"] for x in rows}) == 200
        extras = {x["item_id"]: json.loads(x["extra_json"]) for x in rows}
        for lid in CAPPED_BCB:
            assert extras[lid]["attempt"] == 2 and extras[lid]["cap_rung"] == 2 and extras[lid]["max_tokens_used"] == 16_384
        unselected = extras[BCB_LOG[0]]
        assert unselected["cap_rung"] == 1 and unselected["attempt"] == 1 and unselected["max_tokens_used"] == 4096
        # rung-1 rows keep their producer facts (the same INSPECT_EXTRA_FIELDS + the BCB facts)
        assert {"stop_reason", "unparseable", "code_source", "sandbox_status", "verifier_explanation"} <= set(unselected)
        assert r["k_rung1"] == sum(1 for x in rung1_results(cell, BCB, capped=DEFAULT_CAPPED[(cell, BCB)]) if x.value == 1.0)
        assert r["acc_merged"] == r["k_merged"] / 200
    steered, base = summ["per_cell"][CELL_STEERED][BCB], summ["per_cell"][CELL_BASE][BCB]
    assert steered["n_at_cap_rung1"] == 3 and steered["n_at_cap_residual"] == 1
    assert base["n_at_cap_rung1"] == 1 and base["n_at_cap_residual"] == 0
    # the capped items were 0 on rung 1 and (mostly) score on rung 2
    assert steered["k_merged"] >= steered["k_rung1"]
    written = json.loads((out / L.MERGED_SUMMARY_FILENAME).read_text())
    assert written["per_cell"] == summ["per_cell"]


def test_merge_cli_round_trip(world, capsys):
    doc, sel_path, rung2, cert = _select_and_rung2(world)
    rc = L.main(["merge", "--rung1", str(world["rung1"]), "--rung2", str(rung2), "--selection", str(sel_path),
                 "--certificate", str(cert), "--out", str(world["tmp"] / "m"), "--repo-root", str(world["repo"])])
    assert rc == 0 and "merged summary" in capsys.readouterr().out
    rc = L.main(["detect", "--rung1", str(world["rung1"])])
    assert rc == 0 and "FIRED" in capsys.readouterr().out
    rc = L.main(["select", "--rung1", str(world["rung1"]), "--out", str(world["tmp"] / "s.json"), "--repo-root", str(world["repo"])])
    assert rc == 0 and "n_selected=3" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# merge — adversarial (recompute, never trust)                                 #
# --------------------------------------------------------------------------- #


def test_merge_refuses_an_omitted_id(world):
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    tampered = retarget_selection(doc, BCB, BCB_IDS[1:3])                        # the third omitted, digests consistent
    sel_path = L.write_selection(tampered, world["tmp"] / "sel.json")
    rung2 = make_rung2(world["tmp"] / "rung2", tampered)
    cert = make_certificate(world["tmp"] / "cert.json", tampered)
    with pytest.raises(L.LadderError, match="recomputed union does not equal"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


def test_merge_refuses_a_substituted_id(world):
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    tampered = retarget_selection(doc, BCB, [BCB_IDS[1], BCB_IDS[2], BCB_IDS[4]])   # the third → a pinned, uncapped item
    sel_path = L.write_selection(tampered, world["tmp"] / "sel.json")
    rung2 = make_rung2(world["tmp"] / "rung2", tampered)
    cert = make_certificate(world["tmp"] / "cert.json", tampered)
    with pytest.raises(L.LadderError, match="recomputed union does not equal"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


def test_merge_and_read_store_refuse_a_root_with_a_sampling_validation_failure(world):
    """A ladder rerun whose driver finished and whose clamp scan then filed
    `sampling_validation_failed.json` leaves a completed PARTIAL store; the ONE root-admissibility rule
    (`root_admissibility_reason`, run by `read_store` for every store the ladder or the analysis loads)
    refuses it — `merge` never accepts a poisoned rung-2 (or rung-1) root, whatever the markers say."""
    doc, sel_path, rung2, cert = _select_and_rung2(world)
    out = world["tmp"] / "merged"
    assert L.root_admissibility_reason(rung2 / CELL_STEERED) is None
    summ = L.merge(world["rung1"], rung2, sel_path, cert, out, repo_root=world["repo"])
    assert summ["per_cell"][CELL_STEERED][BCB]["rung2_root"] == str(rung2 / CELL_STEERED)
    assert summ["per_cell"][CELL_STEERED][BCB]["rung1_root"] == str(world["rung1"] / CELL_STEERED)
    marker = write_sampling_validation_marker(rung2 / CELL_STEERED, cell=CELL_STEERED)
    reason = L.root_admissibility_reason(rung2 / CELL_STEERED)
    assert reason and "not protocol evidence" in reason and C.REC_CLAMP_OVERRIDE_INEFFECTIVE in reason and "attempt 'fc-test'" in reason
    with pytest.raises(L.LadderError, match="not protocol evidence"):
        L.read_store(L.store_dir_for(rung2, CELL_STEERED, BCB), BCB, cell=CELL_STEERED)
    with pytest.raises(L.LadderError, match="not protocol evidence"):
        L.load_store(rung2, CELL_STEERED, BCB)
    with pytest.raises(L.LadderError, match=f"{C.SAMPLING_VALIDATION_FAILED_FILENAME}.*not protocol evidence"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "merged2", repo_root=world["repo"])
    marker.unlink()
    # an unreadable marker is a failure too, never "absent"; a rung-1 root is held to the same rule
    (world["rung1"] / CELL_BASE / C.SAMPLING_VALIDATION_FAILED_FILENAME).write_text("{not json")
    with pytest.raises(L.LadderError, match=f"{C.REC_SAMPLING_VALIDATION_FAILED}: unreadable"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "merged3", repo_root=world["repo"])
    (world["rung1"] / CELL_BASE / C.SAMPLING_VALIDATION_FAILED_FILENAME).unlink()
    st = L.load_store(rung2, CELL_STEERED, BCB)
    assert st.root == rung2 / CELL_STEERED and L.store_root_of(st.cell_dir) == st.root


def test_merge_refuses_a_foreign_store_id_on_rung_2(world):
    # one item too many: the marker's n_items disagrees with the selection first
    doc, sel_path, rung2, cert = _select_and_rung2(world, extra_ids={(CELL_STEERED, BCB): [BCB_LOG[5]]})
    with pytest.raises(L.LadderError, match="n_items 4 .* != 3"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


def test_merge_refuses_a_substituted_id_on_rung_2(world):
    # the right COUNT and the right identity digest, but one selected item swapped for a foreign one
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    sel_path = L.write_selection(doc, world["tmp"] / "sel.json")
    swapped = retarget_selection(doc, BCB, [BCB_IDS[1], BCB_IDS[2], BCB_IDS[5]])
    rung2 = make_rung2(world["tmp"] / "rung2", swapped,
                       identity_override={"item_selection": {BCB: {"n": 3, "ids_sha256": doc["selection"][BCB]["ids_sha256"]}}})
    cert = make_certificate(world["tmp"] / "cert.json", doc)
    with pytest.raises(L.LadderError, match="rung-2 outcomes .* != the selection"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


def test_merge_refuses_edited_rung1_rows(world):
    doc, sel_path, rung2, cert = _select_and_rung2(world)
    rp = L.store_dir_for(world["rung1"], CELL_BASE, BCB) / "rows.jsonl"
    lines = rp.read_text().splitlines()
    row = json.loads(lines[0])
    row["created_at"] = row["created_at"] + 1              # innocuous to every reading; the bytes moved
    lines[0] = json.dumps(row, sort_keys=True)
    rp.write_text("\n".join(lines) + "\n")
    with pytest.raises(L.LadderError, match="rows_sha256 .* drifted"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


def test_merge_refuses_a_mismatched_rung2_certificate_digest(world):
    doc, sel_path, rung2, _ = _select_and_rung2(world)
    cert = make_certificate(world["tmp"] / "cert.json", doc, selection_sha256="f" * 64)
    with pytest.raises(L.LadderError, match="certified on another selection"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


def test_merge_refuses_a_rung2_store_generated_on_another_selection(world):
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    sel_path = L.write_selection(doc, world["tmp"] / "sel.json")
    other = retarget_selection(doc, BCB, [BCB_IDS[1], BCB_IDS[2], BCB_IDS[4]])   # same n, other ids
    rung2 = make_rung2(world["tmp"] / "rung2", other)          # its identity item_selection digest is another selection's
    cert = make_certificate(world["tmp"] / "cert.json", doc)
    with pytest.raises(L.LadderError, match="item_selection"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


def test_merge_refuses_a_missing_rung2_item(world):
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    sel_path = L.write_selection(doc, world["tmp"] / "sel.json")
    short = retarget_selection(doc, BCB, BCB_IDS[1:3])
    rung2 = make_rung2(world["tmp"] / "rung2", short,
                       identity_override={"item_selection": {BCB: {"n": 3, "ids_sha256": doc["selection"][BCB]["ids_sha256"]}}})
    # the marker says n_items 2 while the selection has 3
    cert = make_certificate(world["tmp"] / "cert.json", doc)
    with pytest.raises(L.LadderError, match="n_items"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


def test_merge_refuses_a_smoke_partial_as_a_ladder_store(world):
    doc, sel_path, rung2, cert = _select_and_rung2(world, limits={"inspect_limit": 3})
    with pytest.raises(L.LadderError, match="a smoke PARTIAL"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


def test_merge_refuses_rung2_rows_not_generated_at_cap_to(world):
    doc, sel_path, rung2, cert = _select_and_rung2(world, row_cap=8192)
    with pytest.raises(L.LadderError, match="not generated at cap_to"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


def test_merge_refuses_differing_served_blocks_across_rung2_cells(world):
    doc, sel_path, rung2, cert = _select_and_rung2(world, served_by_cell={CELL_BASE: (32_768, 16), CELL_STEERED: (32_768, 14)})
    with pytest.raises(L.LadderError, match="accepted|different `served`"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


def test_merge_refuses_an_identity_change_outside_the_admitted_keys(world):
    doc, sel_path, rung2, cert = _select_and_rung2(world, identity_override={"samples": 2})
    with pytest.raises(L.LadderError, match=r"differs from rung 1 in \['samples'\]"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


def test_merge_refuses_a_certificate_cap_map_that_disagrees(world):
    doc, sel_path, rung2, _ = _select_and_rung2(world)
    cert = make_certificate(world["tmp"] / "cert.json", doc, caps={BCB: 8192})
    with pytest.raises(L.LadderError, match="cap_overrides_by_task|cap"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


def test_certificate_gate_runs_the_launchers_rules(world):
    """`load_certificate(selection=)` re-runs `rung_probe.certificate_is_valid`: a
    valid probe / operator certificate passes; window, S, source, walk, budget, cap map and
    the fit guard each refuse — before any merge or forecast reads the accepted rung."""
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    sel = doc["selection"]
    good = json.loads(make_certificate(world["tmp"] / "c.json", doc).read_text())
    assert RP.certificate_is_valid is not None
    L.validate_certificate(good, sel)
    op = json.loads(make_certificate(world["tmp"] / "op.json", doc, rung_source="operator").read_text())
    assert op["walked"] == []
    L.validate_certificate(op, sel)
    tail = "KV cache holds 528000 tokens < the 589824-token budget (= 18 x 32768)"
    for mutate, why in (
            (lambda c: c["accepted"].update(max_model_len=16_384), "max_model_len"),
            (lambda c: c["accepted"].update(max_num_seqs=17), "not on the ladder"),
            (lambda c: c["accepted"].update(seqs_source="model_budget"), "seqs_source|source"),
            (lambda c: c.update(walked=[{"S": 18, "outcome": "refused", "kv_tokens": 528_000, "refusal_tail": tail},
                                        {"S": 14, "outcome": "refused", "kv_tokens": 528_000, "refusal_tail": tail},
                                        {"S": 16, "outcome": "accepted", "kv_tokens": None, "refusal_tail": None}]), "walk|walked|descending|next"),
            (lambda c: c["budget"].update(batched_token_budget=500_000), "budget"),
            (lambda c: c.update(cap_overrides_by_task={BCB: 8192}), "cap"),
            (lambda c: c.update(selection_sha256="f" * 64), "selection"),
            (lambda c: c.update(walked=[]), "walk|accepted")):
        c = copy.deepcopy(good)
        mutate(c)
        with pytest.raises(L.LadderError, match=f"certificate refused: .*({why})"):
            L.validate_certificate(c, sel)
    # the fit guard: a selection whose longest prompt no longer fits the window with cap_to
    wide = copy.deepcopy(doc)
    wide["selection"][BCB]["max_prompt_tokens_rung1"] = 20_000
    wide["selection_sha256"] = L.selection_digest(wide["selection"])
    cert_wide = json.loads(make_certificate(world["tmp"] / "w.json", wide).read_text())
    with pytest.raises(L.LadderError, match="certificate refused: .*(fit|window|prompt)"):
        L.validate_certificate(cert_wide, wide["selection"])


def test_selection_loads_only_through_the_contracts_reader(world):
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    p = L.write_selection(doc, world["tmp"] / "sel.json")
    assert L.load_selection(p)["selection_sha256"] == doc["selection_sha256"]
    for mutate, why in (
            (lambda d: d["selection"][BCB].update(n=2), "n / n_selected / len"),
            (lambda d: d["selection"][BCB]["ids"].__setitem__(0, "BigCodeBench/999999"), "ids|vocabulary|pin"),
            (lambda d: d["selection"][BCB].update(cap_to=8192), "cap_to"),
            (lambda d: d["selection"].__setitem__("inspect/math", d["selection"].pop(BCB)), "ladder row task"),
            (lambda d: d.update(selection={}), "non-empty|nothing fired"),
            (lambda d: d.pop("cells"), "cells")):
        d = json.loads(p.read_text())
        mutate(d)
        bad = world["tmp"] / "bad.json"
        bad.write_text(json.dumps(d))
        with pytest.raises(L.LadderError, match=why):
            L.load_selection(bad)


def test_certificate_shape_refusals(world):
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    p = make_certificate(world["tmp"] / "c.json", doc)
    good = json.loads(p.read_text())
    assert L.load_certificate(p)["accepted"]["rung_source"] == "probe"
    for mutate, why in ((lambda c: c.pop("budget"), "missing"),
                        (lambda c: c.update(extra=1), "extra"),
                        (lambda c: c["accepted"].update(max_num_seqs="16"), "positive ints"),
                        (lambda c: c["accepted"].update(rung_source="guess"), "rung_source"),
                        (lambda c: c["accepted"].update(seqs_source="probe_override"), "seqs_source")):
        c = copy.deepcopy(good)
        mutate(c)
        p.write_text(json.dumps(c))
        with pytest.raises(L.LadderError, match=why):
            L.load_certificate(p)


LADDER_EXTRA = {"cap_overrides_by_task": {BCB: 16_384}, "item_selection": {BCB: {"n": 3, "ids_sha256": "0" * 64}}}
ACCEPTED_16 = {"max_model_len": 32_768, "max_num_seqs": 16, "seqs_source": "override", "rung_source": "probe"}


def test_identity_contract_admits_only_the_ladder_keys_and_a_concurrency_that_follows_s():
    # the production transition: rung 1 (16,384 × 36, width 36) → rung 2 (32,768 × 16, width 16)
    a = _identity(BCB, limits={}, served=SERVED_RUNG1)
    b = _identity(BCB, limits=L.LIMITS_LADDER, served=SERVED_RUNG2, extra=LADDER_EXTRA)
    assert (a["served"]["concurrency"], b["served"]["concurrency"]) == (36, 16)
    L.check_rung_identity_contract(a, b, where="x", accepted=ACCEPTED_16, merged_tasks=[BCB])
    # a width that did NOT follow S is another regime
    b_stuck = _identity(BCB, limits=L.LIMITS_LADDER, served=SERVED_RUNG2, extra=LADDER_EXTRA, concurrency=36)
    with pytest.raises(L.LadderError, match="concurrency 36 != its max_num_seqs 16"):
        L.check_rung_identity_contract(a, b_stuck, where="x", accepted=ACCEPTED_16, merged_tasks=[BCB])
    b_odd = _identity(BCB, limits=L.LIMITS_LADDER, served=SERVED_RUNG2, extra=LADDER_EXTRA, concurrency=4)
    with pytest.raises(L.LadderError, match="concurrency 4 != its max_num_seqs 16"):
        L.check_rung_identity_contract(a, b_odd, where="x", accepted=ACCEPTED_16, merged_tasks=[BCB])
    # rung 1 itself off the launcher's rule
    a_odd = _identity(BCB, limits={}, served=SERVED_RUNG1, concurrency=8)
    with pytest.raises(L.LadderError, match="rung 1 served concurrency 8"):
        L.check_rung_identity_contract(a_odd, b, where="x", accepted=ACCEPTED_16, merged_tasks=[BCB])
    # a rung-2 width that follows its S but not the certificate's accepted S
    with pytest.raises(L.LadderError, match="!= the certificate's accepted S 14"):
        L.check_rung_identity_contract(a, b, where="x", accepted={**ACCEPTED_16, "max_num_seqs": 14}, merged_tasks=[BCB])
    b3 = dict(b, thinking="on")
    with pytest.raises(L.LadderError, match="thinking"):
        L.check_rung_identity_contract(a, b3, where="x", accepted=ACCEPTED_16, merged_tasks=[BCB])


# --------------------------------------------------------------------------- #
# merge — the experiment identity and the certificate's provenance #
# --------------------------------------------------------------------------- #


def test_merge_happy_path_carries_the_real_concurrency_transition(world):
    doc, sel_path, rung2, cert = _select_and_rung2(world)
    st1, st2 = L.load_store(world["rung1"], CELL_STEERED, BCB), L.load_store(rung2, CELL_STEERED, BCB)
    assert (st1.identity["served"]["concurrency"], st2.identity["served"]["concurrency"]) == (36, 16)
    summ = L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])
    assert summ["experiment"][CELL_STEERED]["object_sha256"] == OBJECT_SHA and summ["experiment"][CELL_BASE]["object_sha256"] is None
    assert summ["certificate"]["rev"] == REV and summ["certificate"]["tree_sha"] == TREE and summ["certificate"]["cfg_sha256"] == "4" * 64
    r = summ["per_cell"][CELL_BASE][BCB]
    assert r["merged_rows_sha256"] == L.sha256_file(r["merged_rows"])


@pytest.mark.parametrize("kw,needle", [
    ({"object_sha256": "0" * 64}, "object_artifact_sha256"),
    ({"model_revision": "f" * 40}, "model_revision"),
    ({"tree_sha": "e" * 40}, "producer.tree_sha"),
    ({"intervention_hash": "d" * 64}, "intervention_hash"),
])
def test_merge_refuses_a_rung2_store_from_another_build(world, kw, needle):
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    sel_path = L.write_selection(doc, world["tmp"] / "sel.json")
    rung2 = make_rung2(world["tmp"] / "rung2", doc, store_kwargs_by_cell={CELL_STEERED: kw})
    cert = make_certificate(world["tmp"] / "cert.json", doc)
    with pytest.raises(L.LadderError, match=f"experiment identity differs from rung 1 in .*{needle}"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


def test_merge_refuses_a_rung2_concurrency_that_did_not_follow_s(world):
    doc, sel_path, rung2, cert = _select_and_rung2(world, store_kwargs={"concurrency": 36})
    with pytest.raises(L.LadderError, match="concurrency 36 != its max_num_seqs 16"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


@pytest.mark.parametrize("kw,needle", [
    ({"rev": "f" * 40}, "certificate rev"),
    ({"tree_sha": "e" * 40}, "certificate tree_sha"),
    ({"object_digests": {CELL_STEERED: "0" * 64}}, "certificate object digest for cell"),
    ({"object_digests": {}}, "certificate object digest for cell"),
])
def test_merge_binds_the_certificate_provenance_to_the_rung2_stores(world, kw, needle):
    doc, sel_path, rung2, _ = _select_and_rung2(world)
    cert = make_certificate(world["tmp"] / "cert.json", doc, **kw)
    with pytest.raises(L.LadderError, match=needle):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


# --------------------------------------------------------------------------- #
# the detector — rows and outcomes joined per item                             #
# --------------------------------------------------------------------------- #


def _swap_row_stop_reasons(store_dir: Path, task: str, *, uncap: str, cap: str, cap_tokens: int) -> None:
    """Rows only: item ``uncap`` no longer at the cap, item ``cap`` now at the cap
    — the SAME count, so the summary and the outcome record (untouched) still agree on 3."""
    rp = store_dir / "rows.jsonl"
    lines = []
    for line in rp.read_text().splitlines():
        row = json.loads(line)
        if row["metric"] == "acc" and row["item_id"] in (uncap, cap):
            extra = json.loads(row["extra_json"])
            if row["item_id"] == uncap:
                extra["stop_reason"], extra["completion_tokens"] = "stop", 77
            else:
                extra["stop_reason"], extra["completion_tokens"] = "max_tokens", cap_tokens
            row["extra_json"] = canonical_json(extra)
        lines.append(json.dumps(row, sort_keys=True))
    rp.write_text("\n".join(lines) + "\n")


def test_detector_refuses_a_same_count_id_swap_between_rows_and_outcomes(world):
    d = L.store_dir_for(world["rung1"], CELL_STEERED, BCB)
    _swap_row_stop_reasons(d, BCB, uncap=CAPPED_BCB[2], cap=BCB_LOG[4], cap_tokens=4096)     # outcomes {1,2,3}; rows {1,2,4}
    st = L.load_store(world["rung1"], CELL_STEERED, BCB)
    assert st.summary["n_at_cap"] == 3 == sum(1 for r in st.acc_rows().values() if r["extra"]["stop_reason"] == "max_tokens")
    with pytest.raises(L.LadderError, match="at-cap disagreement between rows and outcomes on 2 item"):
        L.at_cap_reading(st)
    with pytest.raises(L.LadderError, match="at-cap disagreement between rows and outcomes"):
        L.detect(world["rung1"])
    with pytest.raises(L.LadderError, match="at-cap disagreement between rows and outcomes"):
        L.build_selection(world["rung1"], repo_root=world["repo"])


def test_selection_is_recomputed_from_the_validated_rows(world):
    st = L.load_store(world["rung1"], CELL_STEERED, BCB)
    rows, by_item = L.validated_items(st)
    assert set(rows) == set(by_item) and len(rows) == 200
    from_rows = sorted(L.to_pinned_id(BCB, i) for i, r in rows.items() if r["extra"]["stop_reason"] == "max_tokens")
    assert sorted(L.capped_pinned_ids(st, L.load_pin(world["repo"], BCB))) == from_rows == sorted(BCB_IDS[1:4])
    # an outcome record whose value disagrees with its row is refused too
    op = st.cell_dir / C.TASK_OUTCOMES_DIR / L.marker_filename(BCB)
    o = json.loads(op.read_text())
    o["items"][7]["value"] = 1.0 - o["items"][7]["value"]
    op.write_text(json.dumps(o))
    with pytest.raises(L.LadderError, match="at-cap disagreement between rows and outcomes"):
        L.validated_items(L.load_store(world["rung1"], CELL_STEERED, BCB))


# --------------------------------------------------------------------------- #
# rung-2 rows validated; a single fired leg from a combined store              #
# --------------------------------------------------------------------------- #


def test_merge_refuses_rung2_rows_edited_without_the_outcomes(world):
    # rung 2 leaves all three selected items at the cap (failures)
    doc, sel_path, rung2, cert = _select_and_rung2(world, residual={(CELL_STEERED, BCB): set(CAPPED_BCB)})
    summ = L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m0", repo_root=world["repo"])
    st = summ["per_cell"][CELL_STEERED][BCB]
    assert st["n_at_cap_residual"] == 3 and st["k_merged"] == st["k_rung1"]
    # the adversary edits the steered rung-2 ROWS to normally-stopped passes; outcomes and summary untouched
    rp = L.store_dir_for(rung2, CELL_STEERED, BCB) / "rows.jsonl"
    lines = []
    for line in rp.read_text().splitlines():
        row = json.loads(line)
        if row["metric"] == "acc" and row["item_id"] in CAPPED_BCB:
            extra = json.loads(row["extra_json"])
            extra["stop_reason"], extra["completion_tokens"] = "stop", 900
            row["extra_json"], row["value"] = canonical_json(extra), 1.0
        lines.append(json.dumps(row, sort_keys=True))
    rp.write_text("\n".join(lines) + "\n")
    with pytest.raises(L.LadderError, match="at-cap disagreement between rows and outcomes on 3 item"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m1", repo_root=world["repo"])


#: BFCL fires (3 capped in the steered cell), BCB does not (2).
BFCL_FIRES = {(CELL_STEERED, BFCL): set(BFCL_IDS[0:3]), (CELL_BASE, BFCL): {BFCL_IDS[0]},
              (CELL_STEERED, BCB): set(CAPPED_BCB[0:2]), (CELL_BASE, BCB): {CAPPED_BCB[0]}}
#: Both legs fire.
BOTH_FIRE = {(CELL_STEERED, BCB): set(CAPPED_BCB), (CELL_BASE, BCB): {CAPPED_BCB[0]},
             (CELL_STEERED, BFCL): set(BFCL_IDS[0:3]), (CELL_BASE, BFCL): set(BFCL_IDS[3:5])}


@pytest.mark.parametrize("capped,fired,combined_rung2", [
    (DEFAULT_CAPPED, [BCB], False), (BFCL_FIRES, [BFCL], False), (BOTH_FIRE, [BCB, BFCL], True), (BOTH_FIRE, [BCB, BFCL], False)])
def test_merge_from_a_combined_rung1_store(tmp_path, capped, fired, combined_rung2):
    # the normal launch: ONE store per cell filing both legs; the rerun files only the fired leg(s)
    repo = make_pins(tmp_path / "repo")
    rung1 = make_rung1(tmp_path / "rung1", capped=capped, combined=True)
    assert L.store_dir_for(rung1, CELL_BASE, BCB) == L.store_dir_for(rung1, CELL_BASE, BFCL)
    st1 = L.load_store(rung1, CELL_BASE, BCB)
    # the producer's shapes: control atoms in `tasks`, row tasks in the per-task maps
    assert st1.identity["tasks"] == sorted([C.CONTROL_BFCL_200, C.CONTROL_BIGCODEBENCH_200]) == ["bfcl_200", "bigcodebench_200"]
    assert set(st1.identity[C.IDENTITY_KEY_THINKING_BY_TASK]) == set(st1.identity[C.IDENTITY_KEY_SAMPLES_BY_TASK]) == {BCB, BFCL}
    assert L.row_tasks_of(st1.identity) == [BFCL, BCB]
    doc = L.build_selection(rung1, repo_root=repo)
    assert sorted(doc["selection"]) == sorted(fired)
    sel_path = L.write_selection(doc, tmp_path / "sel.json")
    rung2 = make_rung2(tmp_path / "rung2", doc, combined=combined_rung2)
    cert = make_certificate(tmp_path / "cert.json", doc)
    st2 = L.load_store(rung2, CELL_STEERED, fired[0])
    assert set(L.row_tasks_of(st2.identity)) == (set(fired) if combined_rung2 else {fired[0]})
    assert all(atom in C.TASKS_OF_CONTROL for atom in st2.identity["tasks"])
    # a BCB-only rerun serves no tool calls while the combined rung 1 did — admitted, the fact follows the legs
    if fired == [BCB]:
        assert st2.identity["served"]["tool_calls"] is False and L.load_store(rung1, CELL_STEERED, BCB).identity["served"]["tool_calls"] is True
    summ = L.merge(rung1, rung2, sel_path, cert, tmp_path / "m", repo_root=repo)
    assert summ["tasks"] == sorted(fired)
    for cell in CELLS:
        for task in fired:
            r = summ["per_cell"][cell][task]
            assert r["n"] == 200 and r["n_selected"] == doc["selection"][task]["n"]
            rows = [json.loads(l) for l in Path(r["merged_rows"]).read_text().splitlines()]
            assert len({x["item_id"] for x in rows}) == 200


@pytest.mark.parametrize("override,needle", [
    ({C.IDENTITY_KEY_THINKING_BY_TASK: {BCB: "on"}}, "thinking_by_task differs from rung 1 on the merged task"),
    ({C.IDENTITY_KEY_SAMPLES_BY_TASK: {BCB: 2}}, "samples_by_task differs from rung 1 on the merged task"),
    ({"tasks": [C.CONTROL_BFCL_200]}, "filing the merged task"),
    ({"tasks": [BCB]}, "not a control atom"),                       # a ROW name where the producer writes atoms
])
def test_merge_refuses_a_projected_map_that_moved_on_the_merged_task(tmp_path, override, needle):
    repo = make_pins(tmp_path / "repo")
    rung1 = make_rung1(tmp_path / "rung1", capped=DEFAULT_CAPPED, combined=True)
    doc = L.build_selection(rung1, repo_root=repo)
    sel_path = L.write_selection(doc, tmp_path / "sel.json")
    rung2 = make_rung2(tmp_path / "rung2", doc, identity_override=override)
    cert = make_certificate(tmp_path / "cert.json", doc)
    with pytest.raises(L.LadderError, match=needle):
        L.merge(rung1, rung2, sel_path, cert, tmp_path / "m", repo_root=repo)


def test_merge_refuses_a_manifest_block_that_moved_on_the_merged_task(tmp_path):
    repo = make_pins(tmp_path / "repo")
    rung1 = make_rung1(tmp_path / "rung1", capped=DEFAULT_CAPPED, combined=True)
    doc = L.build_selection(rung1, repo_root=repo)
    sel_path = L.write_selection(doc, tmp_path / "sel.json")
    rung2 = make_rung2(tmp_path / "rung2", doc)
    cert = make_certificate(tmp_path / "cert.json", doc)
    d = L.store_dir_for(rung2, CELL_STEERED, BCB)
    m = json.loads((d / "manifest.json").read_text())
    m[C.MANIFEST_KEY_SAMPLING]["temperature"] = 0.7                       # a scalar of the block moved
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(L.LadderError, match="experiment identity differs from rung 1 in .*sampling"):
        L.merge(rung1, rung2, sel_path, cert, tmp_path / "m", repo_root=repo)


def test_row_tasks_of_normalises_the_producers_control_atoms():
    assert L.row_tasks_of({"tasks": ["bfcl_200", "bigcodebench_200"]}) == [BFCL, BCB]
    assert L.row_tasks_of({"tasks": ["bigcodebench_200"]}) == [BCB]
    for bad in ({"tasks": [BCB]}, {"tasks": []}, {"tasks": "bfcl_200"}, {}):
        with pytest.raises(L.LadderError, match="control atom"):
            L.row_tasks_of(bad)


def _null_one_outcome_value(store_dir: Path, task: str, item: str) -> None:
    """One outcome record's `value` → null (scored stays as written); the numeric acc row, its stop
    reason, attempt, cap and tokens all untouched — a record that must NOT validate the row."""
    op = store_dir / C.TASK_OUTCOMES_DIR / L.marker_filename(task)
    o = json.loads(op.read_text())
    for it in o["items"]:
        if it["item_id"] == item:
            it["value"] = None
    op.write_text(json.dumps(o))


def test_a_null_outcome_value_never_validates_a_numeric_row(world):
    # rung 1: an otherwise consistent store — refused by the detector, the selection, and the join itself
    d = L.store_dir_for(world["rung1"], CELL_STEERED, BCB)
    _null_one_outcome_value(d, BCB, BCB_LOG[7])
    with pytest.raises(L.LadderError, match="at-cap disagreement between rows and outcomes on 1 item.*record value None"):
        L.validated_items(L.load_store(world["rung1"], CELL_STEERED, BCB))
    with pytest.raises(L.LadderError, match="record value None"):
        L.detect(world["rung1"])
    with pytest.raises(L.LadderError, match="record value None"):
        L.build_selection(world["rung1"], repo_root=world["repo"])


def test_a_null_rung2_outcome_value_is_refused_by_the_merge(world):
    doc, sel_path, rung2, cert = _select_and_rung2(world)
    _null_one_outcome_value(L.store_dir_for(rung2, CELL_STEERED, BCB), BCB, CAPPED_BCB[1])
    with pytest.raises(L.LadderError, match="record value None"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


# --------------------------------------------------------------------------- #
# the ONE grouped projection rule set — scalars stripped of every task map /   #
# provenance field, task maps on the retained tasks, per-task provenance,      #
# conditional optional keys — exercised on full -> BCB-only and full ->        #
# BFCL-only under card AND protocol                                            #
# --------------------------------------------------------------------------- #

PROTOCOL_BY_TASK = {
    BCB: {"executed": {"temperature": 0.0, "top_p": 1.0, "top_k": 0}, "protocol_temperature": 0.0,
          "executed_reason": "equal", "source": "BigCodeBench's reference protocol: greedy pass@1"},
    BFCL: {"executed": {"temperature": 0.001, "top_p": 1.0, "top_k": 0}, "protocol_temperature": 0.001,
           "executed_reason": "equal (clamp override applied)", "source": "BFCL's generation script default 0.001"},
}


def _sampling_block(tasks, *, mode="card"):
    tasks = list(tasks)
    if mode == "card":
        return {"mode": "card", "model_key": "gemma4e4b", "temperature": 1.0, "top_p": 0.95, "top_k": 64,
                "samples_by_task": {t: 1 for t in tasks}, "seed_rule": C.SEED_RULE_INSPECT, "source": "model card"}
    return {"mode": L.SAMPLING_MODE_PROTOCOL, "model_key": "gemma4e4b", "temperature": None, "top_p": None, "top_k": None,
            "samples_by_task": {t: 1 for t in tasks}, "seed_rule": C.SEED_RULE_INSPECT,
            "by_task": {t: dict(PROTOCOL_BY_TASK[t]) for t in tasks},
            # the run-level source is a DERIVED JOIN over the filed tasks: it legitimately changes on a single-leg rerun
            "source": " | ".join(PROTOCOL_BY_TASK[t]["source"] for t in tasks)}


def _protocol_identity(tasks, *, limits, served, extra=None):
    ident = _identity(tasks, limits=limits, served=served, extra=extra)
    ident.update({C.IDENTITY_KEY_TEMPERATURE: None, C.IDENTITY_KEY_TOP_P: None, C.IDENTITY_KEY_TOP_K: None,
                  L.IDENTITY_KEY_SAMPLING_BY_TASK: {t: dict(PROTOCOL_BY_TASK[t]["executed"]) for t in tasks}})
    return ident


def test_project_identity_groups_the_rules_and_names_the_kinds():
    assert L.PROJECTED_IDENTITY_KEYS == ("tasks", C.IDENTITY_KEY_THINKING_BY_TASK, C.IDENTITY_KEY_SAMPLES_BY_TASK,
                                         L.IDENTITY_KEY_SAMPLING_BY_TASK)
    assert L.PROJECTED_MANIFEST_BLOCKS == ((C.MANIFEST_KEY_SAMPLING, "samples_by_task"), (C.MANIFEST_KEY_SAMPLING, "by_task"),
                                           (C.MANIFEST_KEY_THINKING, "by_task"))
    assert L.IDENTITY_KEY_SAMPLING_BY_TASK == "sampling_by_task" and L.SAMPLING_MODE_PROTOCOL == "protocol"
    if hasattr(C, "IDENTITY_KEY_SAMPLING_BY_TASK"):                       # Slice A's name, once it lands
        assert C.IDENTITY_KEY_SAMPLING_BY_TASK == L.IDENTITY_KEY_SAMPLING_BY_TASK
    if hasattr(C, "SAMPLING_PROTOCOL"):
        assert C.SAMPLING_PROTOCOL == L.SAMPLING_MODE_PROTOCOL
    both = _sampling_block([BCB, BFCL], mode="protocol")
    p = L.project_identity(both, [BCB], kind=C.MANIFEST_KEY_SAMPLING, mode="protocol")
    # (2) scalars: every task map AND the derived run-level `source` stripped at once
    assert set(p["scalars"]) == {"mode", "model_key", "temperature", "top_p", "top_k", "seed_rule"}
    # (3) maps restricted to the retained task; the cover reported
    assert set(p["maps"]) == {"samples_by_task", "by_task"} and list(p["maps"]["by_task"]) == [BCB]
    assert p["maps_cover"]["by_task"] == sorted([BCB, BFCL])
    # (4) per-task provenance split out of the map entries
    assert p["maps"]["by_task"][BCB] == {k: v for k, v in PROTOCOL_BY_TASK[BCB].items() if k != "source"}
    assert p["provenance"]["by_task"][BCB] == {"source": PROTOCOL_BY_TASK[BCB]["source"]}
    # (5) the conditional map's presence
    assert p["optional"] == {"by_task": True}
    card = L.project_identity(_sampling_block([BCB, BFCL]), [BCB], kind=C.MANIFEST_KEY_SAMPLING, mode="card")
    assert card["optional"] == {"by_task": False} and "by_task" not in card["maps"]
    # a protocol block missing by_task, or a card block carrying it, refuses by name
    with pytest.raises(L.LadderError, match="present iff the mode is 'protocol'"):
        L.project_identity(_sampling_block([BCB]), [BCB], kind=C.MANIFEST_KEY_SAMPLING, mode="protocol")
    with pytest.raises(L.LadderError, match="present iff the mode is 'protocol'"):
        L.project_identity(both, [BCB], kind=C.MANIFEST_KEY_SAMPLING, mode="card")
    # a retained task the map lacks refuses (a rerun that dropped a task it files)
    with pytest.raises(L.LadderError, match="lacks the retained task"):
        L.project_identity(_sampling_block([BCB]), [BCB, BFCL], kind=C.MANIFEST_KEY_SAMPLING)
    with pytest.raises(L.LadderError, match="unknown block kind"):
        L.project_identity({}, [BCB], kind="nope")
    # code review: an ABSENT non-conditional per-task map refuses — it used to be skipped, so a
    # rung-2 identity wholly missing thinking_by_task / samples_by_task compared as equal
    for missing in (C.IDENTITY_KEY_THINKING_BY_TASK, C.IDENTITY_KEY_SAMPLES_BY_TASK):
        hole = {k: v for k, v in _protocol_identity([BCB, BFCL], limits={}, served=SERVED_RUNG1).items() if k != missing}
        with pytest.raises(L.LadderError, match=f"{missing!r} is absent"):
            L.project_identity(hole, [BCB], kind="identity")
    no_samples = {k: v for k, v in _sampling_block([BCB]).items() if k != "samples_by_task"}
    with pytest.raises(L.LadderError, match="'samples_by_task' is absent"):
        L.project_identity(no_samples, [BCB], kind=C.MANIFEST_KEY_SAMPLING, mode="card")
    with pytest.raises(L.LadderError, match="'by_task' is absent"):
        L.project_identity({"mode": "off"}, [BCB], kind=C.MANIFEST_KEY_THINKING)
    with pytest.raises(L.LadderError, match="not an object"):
        L.project_identity([], [BCB], kind="identity")
    # the identity kind: `tasks` is not a map; the three maps + the conditional sampling_by_task
    ident = _protocol_identity([BCB, BFCL], limits={}, served=SERVED_RUNG1)
    q = L.project_identity(ident, [BFCL], kind="identity")
    assert set(q["maps"]) == {C.IDENTITY_KEY_THINKING_BY_TASK, C.IDENTITY_KEY_SAMPLES_BY_TASK, L.IDENTITY_KEY_SAMPLING_BY_TASK}
    assert q["maps"][L.IDENTITY_KEY_SAMPLING_BY_TASK] == {BFCL: PROTOCOL_BY_TASK[BFCL]["executed"]} and "tasks" in q["scalars"]
    assert q["optional"] == {L.IDENTITY_KEY_SAMPLING_BY_TASK: True}
    # projection_differences: scalars, optional set, maps per retained task, provenance per task
    other = _sampling_block([BCB], mode="protocol")
    other["by_task"][BCB] = {**other["by_task"][BCB], "executed": {"temperature": 0.01, "top_p": 1.0, "top_k": 0}}
    d = L.projection_differences(p, L.project_identity(other, [BCB], kind=C.MANIFEST_KEY_SAMPLING, mode="protocol"))
    assert set(d) == {f"by_task[{BCB}]"}
    other2 = _sampling_block([BCB], mode="protocol")
    other2["by_task"][BCB]["source"] = "another provenance"
    d2 = L.projection_differences(p, L.project_identity(other2, [BCB], kind=C.MANIFEST_KEY_SAMPLING, mode="protocol"))
    assert set(d2) == {f"by_task[{BCB}].provenance"}
    with pytest.raises(L.LadderError, match="incomparable"):
        L.projection_differences(p, q)


@pytest.mark.parametrize("mode", ["card", "protocol"])
@pytest.mark.parametrize("retained", [BCB, BFCL])
def test_both_consumers_admit_a_full_to_single_leg_rerun_under_card_and_protocol(mode, retained):
    """The normal launch files BOTH legs; a rerun of ONE fired leg files
    one — the identity contract and the experiment identity both project onto the retained
    leg, under `card` (no optional map) and `protocol` (the hashed `sampling_by_task` and the
    manifest's `sampling.by_task` present on both, the joined run-level `source` excluded)."""
    build = _identity if mode == "card" else _protocol_identity
    full = build([BCB, BFCL], limits={}, served=SERVED_RUNG1)
    extra = {"cap_overrides_by_task": {retained: 16_384}, "item_selection": {retained: {"n": 2, "ids_sha256": "0" * 64}}}
    rerun = build([retained], limits=L.LIMITS_LADDER, served=SERVED_RUNG2, extra=extra)
    L.check_rung_identity_contract(full, rerun, where="x", accepted=ACCEPTED_16, merged_tasks=[retained])
    m1 = {"model_id": HF, "model_revision": REV, "engine": "vllm", "intervention_hash": "i" * 16, "arm_name": CELL_STEERED,
          "kind": "lora_subtraction", C.MANIFEST_KEY_PRODUCER: {"tree_sha": TREE},
          C.MANIFEST_KEY_ROUTE: {"serving_identity": "proxy_vllm+lens+interleaved", "tool_calls": True,
                                 C.AGENT_LOOP_ENTRY_KEY: True, C.REDUCTION_MODE_FIELD: "stacked",
                                 C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: OBJECT_SHA,
                                 C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {CELL_STEERED: OBJECT_SHA}},
          C.MANIFEST_KEY_SANDBOX: None,
          C.MANIFEST_KEY_SAMPLING: _sampling_block([BCB, BFCL], mode=mode),
          C.MANIFEST_KEY_THINKING: {"enable_thinking": False, "render": C.THINKING_RENDER_NATIVE, "reasoning_parser": None,
                                    "mode": "off", "by_task": {BCB: "off", BFCL: "off"}}}
    m2 = copy.deepcopy(m1)
    m2[C.MANIFEST_KEY_SAMPLING] = _sampling_block([retained], mode=mode)
    m2[C.MANIFEST_KEY_THINKING]["by_task"] = {retained: "off"}
    if retained == BCB:
        m2[C.MANIFEST_KEY_ROUTE].update(tool_calls=False, **{C.AGENT_LOOP_ENTRY_KEY: False})   # the leg-derived facts follow BCB-only
    L.check_rung_experiment_identity(m1, m2, where="x", merged_tasks=[retained], tasks2=[retained])
    if mode == "protocol":
        assert m1[C.MANIFEST_KEY_SAMPLING]["source"] != m2[C.MANIFEST_KEY_SAMPLING]["source"]   # the joined source moved; admitted
        # NEGATIVE: a changed EXECUTED temperature on the retained task refuses (identity AND manifest)
        bad = copy.deepcopy(rerun)
        bad[L.IDENTITY_KEY_SAMPLING_BY_TASK][retained] = {"temperature": 0.01, "top_p": 1.0, "top_k": 0}
        with pytest.raises(L.LadderError, match=f"{L.IDENTITY_KEY_SAMPLING_BY_TASK} differs from rung 1 on the merged task"):
            L.check_rung_identity_contract(full, bad, where="x", accepted=ACCEPTED_16, merged_tasks=[retained])
        m3 = copy.deepcopy(m2)
        m3[C.MANIFEST_KEY_SAMPLING]["by_task"][retained]["executed"] = {"temperature": 0.01, "top_p": 1.0, "top_k": 0}
        with pytest.raises(L.LadderError, match=f"sampling.by_task\\[{retained}\\]"):
            L.check_rung_experiment_identity(m1, m3, where="x", merged_tasks=[retained], tasks2=[retained])
        # a changed per-task PROVENANCE refuses too (compared per task)
        m4 = copy.deepcopy(m2)
        m4[C.MANIFEST_KEY_SAMPLING]["by_task"][retained]["source"] = "someone else's protocol"
        with pytest.raises(L.LadderError, match="provenance"):
            L.check_rung_experiment_identity(m1, m4, where="x", merged_tasks=[retained], tasks2=[retained])
        # NEGATIVE: a protocol / card MIX refuses at both consumers
        card_rerun = _identity([retained], limits=L.LIMITS_LADDER, served=SERVED_RUNG2, extra=extra)
        # (refused on the null scalars first — temperature / top_p / top_k differ — or on the optional key set)
        with pytest.raises(L.LadderError, match="protocol store never merges with a card store|optional|may differ"):
            L.check_rung_identity_contract(full, card_rerun, where="x", accepted=ACCEPTED_16, merged_tasks=[retained])
        m5 = copy.deepcopy(m2)
        m5[C.MANIFEST_KEY_SAMPLING] = _sampling_block([retained], mode="card")
        with pytest.raises(L.LadderError, match="sampling"):
            L.check_rung_experiment_identity(m1, m5, where="x", merged_tasks=[retained], tasks2=[retained])
    else:
        # a card store pair stays valid with NO optional key on either side
        assert L.project_identity(full, [retained], kind="identity")["optional"] == {L.IDENTITY_KEY_SAMPLING_BY_TASK: False}
        # a legacy card store carrying the protocol-only key against one without refuses
        mixed = dict(rerun, **{L.IDENTITY_KEY_SAMPLING_BY_TASK: {retained: {"temperature": 0.0, "top_p": 1.0, "top_k": 0}}})
        with pytest.raises(L.LadderError, match="optional identity key|only \\["):
            L.check_rung_identity_contract(full, mixed, where="x", accepted=ACCEPTED_16, merged_tasks=[retained])


# --------------------------------------------------------------------------- #
# the served object is read where the producer recorded it — the               #
# first lane's REAL manifests carry the route digests null and the provenance  #
# sha filled, so the certificate binds through `provenance.artifact_sha256`    #
# --------------------------------------------------------------------------- #

ROUTE_DIGESTS_NULL = {C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: None, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: None}


def _provenance_shaped(prov_sha_steered):
    """Store kwargs per cell shaped like a real first-lane manifest: route digests null on
    both cells, `provenance.artifact_sha256` filled on the fitted cell, null on the base."""
    return {CELL_STEERED: dict(route_extra=ROUTE_DIGESTS_NULL,
                               manifest_extra={C.MANIFEST_KEY_PROVENANCE: {"artifact_sha256": prov_sha_steered}}),
            CELL_BASE: dict(route_extra=ROUTE_DIGESTS_NULL,
                            manifest_extra={C.MANIFEST_KEY_PROVENANCE: {"artifact_sha256": None}})}


def test_merge_binds_the_certificate_through_the_provenance_when_the_route_digests_are_null(tmp_path):
    repo = make_pins(tmp_path / "repo")
    rung1 = make_rung1(tmp_path / "rung1", capped=DEFAULT_CAPPED, model_length={(CELL_BASE, BFCL): {MODEL_LENGTH_BFCL}},
                       store_kwargs_by_cell=_provenance_shaped(OBJECT_SHA))
    doc = L.build_selection(rung1, repo_root=repo)
    sel_path = L.write_selection(doc, tmp_path / "sel.json")
    rung2 = make_rung2(tmp_path / "rung2", doc, store_kwargs_by_cell=_provenance_shaped(OBJECT_SHA))
    cert = make_certificate(tmp_path / "cert.json", doc)              # object_digests {steered: OBJECT_SHA}
    st2 = L.load_store(rung2, CELL_STEERED, BCB)
    assert L._dig(st2.manifest, (C.MANIFEST_KEY_ROUTE, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256)) is None       # the real shape
    assert L.served_object_sha256(st2.manifest, cell=CELL_STEERED) == (OBJECT_SHA, L.OBJECT_SHA_SOURCES[2])
    assert L.served_object_sha256(L.load_store(rung2, CELL_BASE, BCB).manifest, cell=CELL_BASE) == (None, None)
    summ = L.merge(rung1, rung2, sel_path, cert, tmp_path / "m", repo_root=repo)
    exp = summ["experiment"]
    assert exp[CELL_STEERED]["object_sha256"] == OBJECT_SHA and exp[CELL_STEERED]["object_sha256_source"] == "provenance.artifact_sha256"
    assert exp[CELL_BASE]["object_sha256"] is None and exp[CELL_BASE]["object_sha256_source"] is None
    # the SAME shape with another object in the provenance: the certificate no longer binds
    rung2b = make_rung2(tmp_path / "rung2b", doc, store_kwargs_by_cell=_provenance_shaped("9" * 64))
    with pytest.raises(L.LadderError, match="certificate object digest for cell .*provenance.artifact_sha256"):
        L.merge(rung1, rung2b, sel_path, cert, tmp_path / "m2", repo_root=repo)
    # a fitted store that recorded the object NOWHERE cannot vouch for the certificate's digest
    rung2c = make_rung2(tmp_path / "rung2c", doc, store_kwargs_by_cell=_provenance_shaped(None))
    with pytest.raises(L.LadderError, match="certificate object digest for cell .*all null"):
        L.merge(rung1, rung2c, sel_path, cert, tmp_path / "m3", repo_root=repo)


def test_served_object_sha256_reads_the_sources_in_order_and_refuses_a_disagreement():
    route_only = {C.MANIFEST_KEY_ROUTE: {C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: "a" * 64}}
    assert L.served_object_sha256(route_only, cell="x") == ("a" * 64, L.OBJECT_SHA_SOURCES[0])
    by_cell = {C.MANIFEST_KEY_ROUTE: {C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: None,
                                      C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {"x": "b" * 64, "y": "c" * 64}}}
    assert L.served_object_sha256(by_cell, cell="x") == ("b" * 64, f"{L.OBJECT_SHA_SOURCES[1]}[x]")
    assert L.served_object_sha256(by_cell, cell="z") == (None, None)            # another cell's digest never leaks
    agree = {C.MANIFEST_KEY_ROUTE: {C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: "a" * 64},
             C.MANIFEST_KEY_PROVENANCE: {"artifact_sha256": "a" * 64}}
    assert L.served_object_sha256(agree, cell="x")[0] == "a" * 64
    disagree = {C.MANIFEST_KEY_ROUTE: {C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: "a" * 64},
                C.MANIFEST_KEY_PROVENANCE: {"artifact_sha256": "b" * 64}}
    with pytest.raises(L.LadderError, match="disagree"):
        L.served_object_sha256(disagree, cell="x")
    assert L.served_object_sha256({}, cell="x") == (None, None)


# --------------------------------------------------------------------------- #
# merge — the explicit tree admission                                          #
# --------------------------------------------------------------------------- #

TREE_FIXED = "d73d73d7" + "1" * 32          # the tree carrying the verifier-transport fix; rung 1 stays at TREE
WHY_FIXED = "D73: rung 2 re-scored at the tree carrying the BCB verifier transport fix (Modal ARG_MAX)"


def _rung2_at(world, tree, **kw):
    """rung 2 and its certificate produced at ``tree``; rung 1 stays at TREE."""
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    sel_path = L.write_selection(doc, world["tmp"] / "sel.json")
    rung2 = make_rung2(world["tmp"] / "rung2", doc, store_kwargs={"tree_sha": tree}, **kw)
    cert = make_certificate(world["tmp"] / "cert.json", doc, tree_sha=tree)
    return doc, sel_path, rung2, cert


def _merge_argv(world, rung2, sel_path, cert, out, *extra):
    return ["merge", "--rung1", str(world["rung1"]), "--rung2", str(rung2), "--selection", str(sel_path), "--certificate", str(cert),
            "--out", str(out), "--repo-root", str(world["repo"]), *extra]


def test_merge_refuses_a_moved_rung2_tree_without_the_admission(world):
    """Today's behaviour stays the default: a rung-2 store from another tree refuses by name."""
    _, sel_path, rung2, cert = _rung2_at(world, TREE_FIXED)
    with pytest.raises(L.LadderError, match="experiment identity differs from rung 1 in .*producer.tree_sha"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])


def test_merge_admits_exactly_the_named_tree_and_records_it(world, capsys):
    doc, sel_path, rung2, cert = _rung2_at(world, TREE_FIXED)
    out = world["tmp"] / "m"
    summ = L.merge(world["rung1"], rung2, sel_path, cert, out, repo_root=world["repo"], admit_tree=TREE_FIXED, admit_tree_why=WHY_FIXED)
    assert summ["tree_admission"] == {"rung1_tree_sha": TREE, "rung2_tree_sha": TREE_FIXED, "why": WHY_FIXED, "flag": "--admit-tree"}
    assert set(summ["tree_admission"]) == set(L.TREE_ADMISSION_FIELDS) and L.ADMIT_TREE_FLAG == "--admit-tree"
    for cell in CELLS:
        assert summ["experiment"][cell]["tree_sha"] == TREE and summ["experiment"][cell]["rung2_tree_sha"] == TREE_FIXED
    assert summ["certificate"]["tree_sha"] == TREE_FIXED                 # the certificate still binds to the rung-2 store's tree
    assert summ["per_cell"][CELL_STEERED][BCB]["n_selected"] == 3         # the admission changes no reading
    written = json.loads((out / L.MERGED_SUMMARY_FILENAME).read_text())
    assert written["tree_admission"] == summ["tree_admission"] and written["experiment"] == summ["experiment"]
    # the CLI: both flags admit and announce it
    rc = L.main(_merge_argv(world, rung2, sel_path, cert, world["tmp"] / "m_cli", "--admit-tree", TREE_FIXED, "--admit-tree-why", WHY_FIXED))
    assert rc == 0 and f"tree admission (--admit-tree): rung 2 produced at {TREE_FIXED[:8]} (rung 1 at {TREE[:8]}) — {WHY_FIXED}" in capsys.readouterr().out
    assert json.loads((world["tmp"] / "m_cli" / L.MERGED_SUMMARY_FILENAME).read_text())["tree_admission"] == summ["tree_admission"]
    # one flag without the other is an argparse error (exit 2), before anything is read
    for extra in (("--admit-tree", TREE_FIXED), ("--admit-tree-why", WHY_FIXED)):
        with pytest.raises(SystemExit) as ex:
            L.main(_merge_argv(world, rung2, sel_path, cert, world["tmp"] / "m_half", *extra))
        assert ex.value.code == 2 and not (world["tmp"] / "m_half").exists()


def test_merge_without_the_flag_records_no_admission(world, capsys):
    doc, sel_path, rung2, cert = _select_and_rung2(world)
    summ = L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"])
    assert summ["tree_admission"] is None
    assert all(summ["experiment"][c]["tree_sha"] == TREE == summ["experiment"][c]["rung2_tree_sha"] for c in CELLS)
    assert json.loads((world["tmp"] / "m" / L.MERGED_SUMMARY_FILENAME).read_text())["tree_admission"] is None
    assert L.main(_merge_argv(world, rung2, sel_path, cert, world["tmp"] / "m_cli")) == 0
    assert "tree admission" not in capsys.readouterr().out


def test_merge_refuses_an_admitted_sha_that_is_not_rung_2s_tree(world):
    # rung 2 at the fixed tree, the flag naming a THIRD sha: the identity check refuses as before
    _, sel_path, rung2, cert = _rung2_at(world, TREE_FIXED)
    with pytest.raises(L.LadderError, match="experiment identity differs from rung 1 in .*producer.tree_sha"):
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m1", repo_root=world["repo"], admit_tree="e" * 40, admit_tree_why=WHY_FIXED)
    # rung 2 at TREE like rung 1 (its own dir: a second producer in the first rung-2 dir's stores would trip the
    # store's resume gate first), the flag naming another sha: rung 2 was not produced there
    doc2 = L.build_selection(world["rung1"], repo_root=world["repo"])
    sel_path2 = L.write_selection(doc2, world["tmp"] / "sel2.json")
    rung2b = make_rung2(world["tmp"] / "rung2b", doc2)
    cert2 = make_certificate(world["tmp"] / "cert2.json", doc2)
    with pytest.raises(L.LadderError, match="is not the rung-2 stores' tree"):
        L.merge(world["rung1"], rung2b, sel_path2, cert2, world["tmp"] / "m2", repo_root=world["repo"], admit_tree=TREE_FIXED, admit_tree_why=WHY_FIXED)
    # the flag naming rung 1's own sha: the trees agree, nothing to admit
    with pytest.raises(L.LadderError, match="equals rung 1's tree"):
        L.merge(world["rung1"], rung2b, sel_path2, cert2, world["tmp"] / "m3", repo_root=world["repo"], admit_tree=TREE, admit_tree_why=WHY_FIXED)


def test_merge_refuses_another_differing_path_even_with_the_admission(world):
    _, sel_path, rung2, cert = _rung2_at(world, TREE_FIXED, store_kwargs_by_cell={CELL_STEERED: {"model_revision": "f" * 40}})
    with pytest.raises(L.LadderError, match="experiment identity differs from rung 1 in .*model_revision") as ex:
        L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"], admit_tree=TREE_FIXED, admit_tree_why=WHY_FIXED)
    assert "producer.tree_sha" not in str(ex.value)                      # the admitted difference is not what refused


def test_merge_admission_flag_grammar(world):
    _, sel_path, rung2, cert = _rung2_at(world, TREE_FIXED)
    for kw, needle in (({"admit_tree": TREE_FIXED}, "required together"), ({"admit_tree_why": WHY_FIXED}, "required together"),
                       ({"admit_tree": TREE_FIXED[:39], "admit_tree_why": WHY_FIXED}, "not a 40-hex"),
                       ({"admit_tree": TREE_FIXED.upper(), "admit_tree_why": WHY_FIXED}, "not a 40-hex"),
                       ({"admit_tree": TREE_FIXED, "admit_tree_why": "  "}, "is empty")):
        with pytest.raises(L.LadderError, match=needle):
            L.merge(world["rung1"], rung2, sel_path, cert, world["tmp"] / "m", repo_root=world["repo"], **kw)


def test_check_rung_experiment_identity_admits_only_the_named_tree():
    tasks = [BCB]
    m1 = {"model_id": HF, "model_revision": REV, "engine": "vllm", "intervention_hash": "i", "arm_name": "a", "kind": "none",
          C.MANIFEST_KEY_PRODUCER: {"tree_sha": TREE}, C.MANIFEST_KEY_ROUTE: {"serving_identity": "x"}, C.MANIFEST_KEY_SANDBOX: None,
          C.MANIFEST_KEY_SAMPLING: _sampling_block(tasks),
          C.MANIFEST_KEY_THINKING: {"enable_thinking": False, "render": C.THINKING_RENDER_NATIVE, "reasoning_parser": None,
                                    "mode": "off", "by_task": {BCB: "off"}}}
    m2 = {**m1, C.MANIFEST_KEY_PRODUCER: {"tree_sha": TREE_FIXED}}
    with pytest.raises(L.LadderError, match="producer.tree_sha"):
        L.check_rung_experiment_identity(m1, m2, where="x", merged_tasks=tasks, tasks2=tasks)
    L.check_rung_experiment_identity(m1, m2, where="x", merged_tasks=tasks, tasks2=tasks, admit_tree_sha=TREE_FIXED)
    with pytest.raises(L.LadderError, match="producer.tree_sha"):
        L.check_rung_experiment_identity(m1, m2, where="x", merged_tasks=tasks, tasks2=tasks, admit_tree_sha="e" * 40)
    m3 = {**m2, "model_revision": "f" * 40}
    with pytest.raises(L.LadderError, match="model_revision") as ex:
        L.check_rung_experiment_identity(m1, m3, where="x", merged_tasks=tasks, tasks2=tasks, admit_tree_sha=TREE_FIXED)
    assert "producer.tree_sha" not in str(ex.value)


def test_certificate_gate_judges_by_the_recorded_ladder_when_the_shared_table_refuses(world, monkeypatch):
    """The landed tree keeps main's geometry rows, under which a model's row may be
    provisional and `rung_probe.ladder_spec` refuses — a finished ladder's certificates must still be READABLE
    (the merged llama waves are otherwise unreproducible from the landed code). `spec_for_certificate` then
    holds the certificate to the ladder and budget IT recorded at launch; every other rule of the gate runs
    unchanged, and a certificate without a recorded ladder, or naming another model, refuses."""
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    sel = doc["selection"]
    good = json.loads(make_certificate(world["tmp"] / "c2.json", doc).read_text())
    key = C.model_row(str(good["model_slug"])).model_key

    def refuse(model_key, *, window=RP.LADDER_WINDOW):
        raise RP.LadderError(f"the ladder needs the MEASURED budget row of {model_key!r}; the geometry table holds 'provisional-conservative'")
    monkeypatch.setattr(RP, "ladder_spec", refuse)
    spec, tokens, source = L.spec_for_certificate(key, good)
    assert (spec.window, tuple(spec.seqs), spec.identity_S) == (int(good["ladder"]["window"]), tuple(good["ladder"]["seqs"]), int(good["ladder"]["identity_S"]))
    assert (tokens, source) == (int(good["budget"]["batched_token_budget"]), str(good["budget"]["budget_source"]))
    L.validate_certificate(good, sel)                                            # the recorded ladder admits its own certificate
    bad = copy.deepcopy(good); bad["accepted"].update(max_num_seqs=17)
    with pytest.raises(L.LadderError, match="certificate refused: .*not on the ladder"):
        L.validate_certificate(bad, sel)                                         # the other rules still bite
    with pytest.raises(L.LadderError, match="records no `ladder`"):
        L.spec_for_certificate(key, {k: v for k, v in good.items() if k != "ladder"})
    with pytest.raises(L.LadderError, match="names model"):
        L.spec_for_certificate("qwen35_4b" if key != "qwen35_4b" else "gemma4e4b", good)


def test_the_certificate_gate_and_the_merge_take_a_native_certificate_on_the_engine_served_ladder(world, monkeypatch):
    """``spec_for_certificate`` always reconstructed the LENS ladder for a table-sized model and consulted the
    certificate's recorded block only on a table refusal, so a valid native gemma certificate — its `ladder` block
    naming the engine, the requested -> served mapping and the served rungs (15, 14, 12) — failed the gate for its
    extra keys and `merge` could not merge native rung-2 stores. Now the engine is read off the block and the
    certificate is held to the ENGINE-SERVED ladder recomputed from the table (`rung_probe.native_ladder_spec`): the
    gate admits it at the served 14 and at the served identity 15, refuses a lens rung (16, served at 15), a forged
    mapping and an unknown engine by name; the merge of native rung-2 stores at the served 14 goes through; the lens
    certificate and its gate are byte-identical; with the table refusing (a provisional row) the native block is
    RE-DERIVED from its requested rungs rather than restated, so a forged block still refuses."""
    key = "gemma4e4b"
    native, lens = RP.native_ladder_spec(RP.ladder_spec(key)), RP.ladder_spec(key)
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    sel_path = L.write_selection(doc, world["tmp"] / "ladder_selection.json")
    sel = doc["selection"]
    walked = [{"S": 15, "outcome": "refused", "kv_tokens": 480_000,
               "refusal_tail": "KV cache holds 480000 tokens < the 491520-token budget (= 15 x 32768)"},
              {"S": 14, "outcome": "accepted", "kv_tokens": None, "refusal_tail": None}]
    cert_path = make_certificate(world["tmp"] / "rung2" / "_ladder" / "rung_probe.json", doc, accepted=(32_768, 14), walked=walked)
    cert = json.loads(cert_path.read_text())
    cert["ladder"] = native.block()
    cert_path.write_text(json.dumps(cert, indent=1, sort_keys=True), encoding="utf-8")
    lens_cert = json.loads(make_certificate(world["tmp"] / "lens.json", doc).read_text())
    assert L.certificate_serving_engine(cert) == RP.SERVING_ENGINE_NATIVE and L.certificate_serving_engine(lens_cert) == RP.SERVING_ENGINE_LENS
    # the gate: the ENGINE-SERVED ladder from the table, the certificate admitted (this line refused before: extra native keys)
    spec, tokens, source = L.spec_for_certificate(key, cert)
    assert spec == native and spec.native and (tokens, source) == (589_824, "measured")
    L.validate_certificate(cert, sel)
    assert L.load_certificate(cert_path, selection=sel) == cert
    L.validate_certificate({**cert, "accepted": {"max_model_len": 32_768, "max_num_seqs": 15, "seqs_source": "model_budget", "rung_source": "identity"},
                            "walked": [{"S": 15, "outcome": "accepted", "kv_tokens": None, "refusal_tail": None}]}, sel)
    with pytest.raises(L.LadderError, match=r"not on the ladder \[15, 14, 12\]"):
        L.validate_certificate({**cert, "accepted": {**cert["accepted"], "max_num_seqs": 16}}, sel)
    forged = json.loads(json.dumps(cert))
    forged["ladder"]["served_rungs"][0]["requested"] = [18]
    with pytest.raises(L.LadderError, match="!= this launch's"):
        L.validate_certificate(forged, sel)
    with pytest.raises(L.LadderError, match="serving_engine 'hf'"):
        L.spec_for_certificate(key, {**cert, "ladder": {**cert["ladder"], "serving_engine": "hf"}})
    # the lens certificate: the lens ladder, byte-identical
    assert L.spec_for_certificate(key, lens_cert) == (lens, 589_824, "measured")
    L.validate_certificate(lens_cert, sel)
    # the MERGE of native rung-2 stores at the served 14 against the native certificate. This regression's stores were
    # LENS-shaped (the shared fixtures' `+lens+` route, no engine stamp — a fixture that did not match the producer),
    # and the merge took the native certificate for them all the same. The stores of BOTH rungs are now written as the
    # native driver writes them (`serving_engine=lora_native` in every spelling, the steered route `+lora+`,
    # `not_applicable`), the selection is drawn from the native rung 1, and the canonical reader identifies every
    # store as native before the merge is asked
    NATIVE = C.SERVING_ENGINE_LORA_NATIVE
    native_kw = {c: {"serving_engine": NATIVE} for c in CELLS}
    rung1 = make_rung1(world["tmp"] / "rung1-native", capped=DEFAULT_CAPPED, model_length={(CELL_BASE, BFCL): {MODEL_LENGTH_BFCL}},
                       store_kwargs_by_cell=native_kw)
    doc = L.build_selection(rung1, repo_root=world["repo"])
    sel_path = L.write_selection(doc, world["tmp"] / "native_selection.json")
    cert["selection_sha256"] = doc["selection_sha256"]
    cert_path.write_text(json.dumps(cert, indent=1, sort_keys=True), encoding="utf-8")
    rung2 = make_rung2(world["tmp"] / "rung2", doc, served=(32_768, 14), residual={(CELL_STEERED, BCB): {CAPPED_BCB[2]}},
                       store_kwargs={"serving_engine": NATIVE})
    assert sorted(doc["selection"]) == [BCB]                               # the fixture's fired leg; rung 2 files it alone
    for rung, tasks in ((rung1, (BCB, BFCL)), (rung2, (BCB,))):
        for cell in CELLS:
            for task in tasks:
                st = L.load_store(rung, cell, task)
                reading = C.read_store_serving_engine(st.manifest, where="m")
                assert (reading.engine, reading.stated, reading.refusal) == (NATIVE, NATIVE, None), (rung, cell, task, reading)
                assert L.store_serving_engine(st) == NATIVE
                m = st.manifest
                assert m[C.SERVED_ENGINE_FIELD] == m[C.MANIFEST_KEY_ROUTE][C.SERVED_ENGINE_FIELD] == NATIVE
                assert m["resolved_config"]["serving"][C.SERVED_ENGINE_FIELD] == st.identity["served"][C.SERVED_ENGINE_FIELD] == NATIVE
                assert m[C.MANIFEST_KEY_ROUTE]["serving_identity"] == m["serving_identity"] == (
                    "proxy_vllm+passthrough+interleaved" if cell == CELL_BASE else "proxy_vllm+lora+interleaved")
                assert m["serving_path"] == ("passthrough" if cell == CELL_BASE else "lora")
                mode = C.read_store_reduction_mode(m, where="m")
                assert (mode.mode, mode.refusal, mode.serving_engine) == (C.REDUCTION_MODE_NOT_APPLICABLE, None, NATIVE)
    summ = L.merge(rung1, rung2, sel_path, cert_path, world["tmp"] / "merged", repo_root=world["repo"])
    assert summ["certificate"]["accepted"] == cert["accepted"] == {"max_model_len": 32_768, "max_num_seqs": 14, "seqs_source": "override",
                                                                   "rung_source": "probe"}
    assert summ["per_cell"][CELL_STEERED][BCB]["n_at_cap_residual"] == 1
    # rung-2 stores at another S than the native certificate's served rung do not merge
    other = make_rung2(world["tmp"] / "rung2b", doc, served=(32_768, 16), store_kwargs={"serving_engine": NATIVE})
    with pytest.raises(L.LadderError, match=r"rung-2 served \(32768, 16\) != the certificate's accepted \(32768, 14\)"):
        L.merge(rung1, other, sel_path, cert_path, world["tmp"] / "merged-b", repo_root=world["repo"])
    # the lens fixtures are untouched: lens by era on positive evidence, nothing stated, the historical manifest key set
    lens_store = L.load_store(world["rung1"], CELL_STEERED, BCB)
    assert C.read_store_serving_engine(lens_store.manifest, where="m")[:3] == (RP.SERVING_ENGINE_LENS, None, None)
    assert not ({C.SERVED_ENGINE_FIELD, "serving_path", "serving_identity", "resolved_config"} & set(lens_store.manifest))
    assert lens_store.manifest[C.MANIFEST_KEY_ROUTE]["serving_identity"] == "proxy_vllm+lens+interleaved"
    # the table refuses (a provisional row): the native block RE-DERIVES from its requested rungs through the one reader;
    # a forged block refuses by name instead of being restated as its own claim
    def refuse(model_key, *, window=RP.LADDER_WINDOW):
        raise RP.LadderError(f"the ladder needs the MEASURED budget row of {model_key!r}; the geometry table holds 'provisional-conservative'")
    monkeypatch.setattr(RP, "ladder_spec", refuse)
    spec, tokens, source = L.spec_for_certificate(key, cert)
    assert spec == native and (tokens, source) == (589_824, "measured")
    L.validate_certificate(cert, sel)
    with pytest.raises(L.LadderError, match="recorded ladder block cannot be read"):
        L.spec_for_certificate(key, forged)
    assert L.spec_for_certificate(key, lens_cert)[0].block() == lens.block()


def test_the_merge_binds_the_certificate_to_the_stores_serving_engine(world):
    """Certificates were not bound to the stores' serving ENGINE. A valid NATIVE gemma certificate at (32,768, 14) —
    its `ladder` block naming `lora_native` — merged LENS rung-2 stores at that rung silently (the fitted `+lens+`
    route included): `validate_certificate` judged it on its own engine's ladder, `bind_certificate_to_store` checked
    tree / model / object only, rung-to-rung agreement holds when BOTH rungs sit on the other engine, and the analysis
    then labelled the ladder's engine from the certificate. Now `bind_certificate_to_store` requires the certificate's
    engine (`certificate_serving_engine`) to equal the store's canonical reading (`store_serving_engine` ->
    `contracts.read_store_serving_engine`) and `merge` holds every rung-1 source to the certificate's engine too: a
    native certificate over lens stores and a lens certificate over native stores refuse by name (both through `merge`
    and at the binding boundary directly); a store whose engine cannot be read — torn of its stamps, or not a served
    store's manifest at all — binds nothing; same-engine merges on either engine are unchanged."""
    NATIVE, LENS = C.SERVING_ENGINE_LORA_NATIVE, RP.SERVING_ENGINE_LENS
    native_spec = RP.native_ladder_spec(RP.ladder_spec("gemma4e4b"))
    native_kw = {c: {"serving_engine": NATIVE} for c in CELLS}
    # two worlds: the fixture's LENS rung 1 and a NATIVE rung 1, each with its own selection
    lens_doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    lens_sel = L.write_selection(lens_doc, world["tmp"] / "lens_sel.json")
    rung1_native = make_rung1(world["tmp"] / "rung1n", capped=DEFAULT_CAPPED, model_length={(CELL_BASE, BFCL): {MODEL_LENGTH_BFCL}},
                              store_kwargs_by_cell=native_kw)
    native_doc = L.build_selection(rung1_native, repo_root=world["repo"])
    native_sel = L.write_selection(native_doc, world["tmp"] / "native_sel.json")

    def native_cert(doc, path):
        walked = [{"S": 15, "outcome": "refused", "kv_tokens": 480_000,
                   "refusal_tail": "KV cache holds 480000 tokens < the 491520-token budget (= 15 x 32768)"},
                  {"S": 14, "outcome": "accepted", "kv_tokens": None, "refusal_tail": None}]
        p = make_certificate(path, doc, accepted=(32_768, 14), walked=walked)
        cert = json.loads(p.read_text())
        cert["ladder"] = native_spec.block()
        p.write_text(json.dumps(cert, indent=1, sort_keys=True), encoding="utf-8")
        return p, cert

    # THE finding: a valid native certificate at the served 14, LENS stores at (32,768, 14) on both rungs — merged before
    n_cert_path, n_cert = native_cert(lens_doc, world["tmp"] / "n_over_lens" / "rung_probe.json")
    assert L.certificate_serving_engine(n_cert) == NATIVE and L.validate_certificate(n_cert, lens_doc["selection"]) is None
    lens_rung2 = make_rung2(world["tmp"] / "lens_rung2", lens_doc, served=(32_768, 14))
    st2 = L.load_store(lens_rung2, CELL_STEERED, BCB)
    assert st2.manifest[C.MANIFEST_KEY_ROUTE]["serving_identity"] == "proxy_vllm+lens+interleaved" and L.store_serving_engine(st2) == LENS
    with pytest.raises(L.LadderError, match=r"certificate serving engine 'lora_native' != the rung-2 store's 'lens' — .*cross-engine certificate binds nothing"):
        L.bind_certificate_to_store(n_cert, st2)                                          # the binding boundary itself
    with pytest.raises(L.LadderError, match=r"rung-1 source's serving engine 'lens' != the certificate's 'lora_native' — the ladder is single-engine"):
        L.merge(world["rung1"], lens_rung2, lens_sel, n_cert_path, world["tmp"] / "m1", repo_root=world["repo"])
    # ... and the RECIPROCAL: a valid lens certificate (the fixture's, at 16) over NATIVE stores at (32,768, 16) on both rungs
    l_cert_path = make_certificate(world["tmp"] / "lens_over_n" / "rung_probe.json", native_doc)
    l_cert = json.loads(l_cert_path.read_text())
    assert L.certificate_serving_engine(l_cert) == LENS and L.validate_certificate(l_cert, native_doc["selection"]) is None
    native_rung2 = make_rung2(world["tmp"] / "native_rung2", native_doc, served=(32_768, 16), store_kwargs={"serving_engine": NATIVE})
    nst2 = L.load_store(native_rung2, CELL_STEERED, BCB)
    assert L.store_serving_engine(nst2) == NATIVE
    with pytest.raises(L.LadderError, match=r"certificate serving engine 'lens' != the rung-2 store's 'lora_native'"):
        L.bind_certificate_to_store(l_cert, nst2)
    with pytest.raises(L.LadderError, match=r"rung-1 source's serving engine 'lora_native' != the certificate's 'lens'"):
        L.merge(rung1_native, native_rung2, native_sel, l_cert_path, world["tmp"] / "m2", repo_root=world["repo"])
    # a store whose engine cannot be read binds nothing: torn of its stamps (a `+lora+` identity without them), or not a
    # served store's manifest at all (an HF reference's shape: no engine stamp, no serving_path, no route)
    torn = copy.deepcopy(nst2.manifest)
    for node in (torn, torn[C.MANIFEST_KEY_ROUTE], torn["resolved_config"]["serving"], torn[C.MANIFEST_KEY_TINYCAP]["served"]):
        node.pop(C.SERVED_ENGINE_FIELD)
    with pytest.raises(L.LadderError, match=r"serving engine cannot be read — .*a torn native record"):
        L.store_serving_engine(dataclasses.replace(nst2, manifest=torn))
    with pytest.raises(L.LadderError, match=r"serving engine cannot be read"):
        L.bind_certificate_to_store(l_cert, dataclasses.replace(nst2, manifest=torn))
    hf_shaped = {"cell_id": CELL_STEERED, "engine": "hf", "model_id": HF, "model_revision": REV, "row_constants": {"engine": "hf"},
                 C.MANIFEST_KEY_PRODUCER: dict(nst2.manifest[C.MANIFEST_KEY_PRODUCER])}          # cell_controls_run's reference shape
    assert C.read_store_serving_engine(hf_shaped, where="m").engine is None
    with pytest.raises(L.LadderError, match=r"states no serving engine .* a certificate binds to no store of unknown engine"):
        L.bind_certificate_to_store(l_cert, dataclasses.replace(nst2, manifest=hf_shaped))
    # the same-engine merges stand: lens over lens (the historical path, byte for byte) and native over native
    lens_ok = L.merge(world["rung1"], make_rung2(world["tmp"] / "lens_ok", lens_doc), lens_sel,
                      make_certificate(world["tmp"] / "lens_ok.json", lens_doc), world["tmp"] / "m3", repo_root=world["repo"])
    assert "serving_engine" not in lens_ok and "serving_engine" not in lens_ok["certificate"]      # the lens summary's key set
    n_cert_path2, _ = native_cert(native_doc, world["tmp"] / "n_over_n" / "rung_probe.json")
    native_ok = L.merge(rung1_native, make_rung2(world["tmp"] / "native_ok", native_doc, served=(32_768, 14),
                                                 store_kwargs={"serving_engine": NATIVE}),
                        native_sel, n_cert_path2, world["tmp"] / "m4", repo_root=world["repo"])
    assert native_ok["certificate"]["accepted"]["max_num_seqs"] == 14
