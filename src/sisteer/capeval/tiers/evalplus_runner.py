"""HumanEval+ generation seam — generation local, execution sandboxed
(`capeval_exec_wave.py` owns the execution
side; this module produces/consumes the fixture-pinned wire formats).

Base-lane importable: evalplus is a heavy-lane dependency, imported
lazily inside the functions that need it (§3.2/§13). The pure helpers
(samples.jsonl writing, eval-results→row conversion) run under base deps
only and are pinned by tests/test_capeval_protocol.py's frozen EvalPlus
fixture pair.

Pinned facts, verified against the evalplus 0.3.1 wheel (2026-08-15):

* The chat prompt helper IS importable at the 0.3.1 pin — no vendoring
  needed: ``evalplus.provider.utility.make_raw_chat_prompt(task_prompt,
  instruction_prefix, response_prefix, tokenizer)``
  (provider/utility.py:25). It renders a user turn (instruction prefix +
  the problem in a fenced block) plus an assistant turn opened with the
  response prefix and a ```` ```python ```` fence, applies
  ``tokenizer.apply_chat_template(..., tokenize=False)`` and truncates
  at evalplus's magic splitter — so the prompt ends INSIDE the assistant
  code fence and the model continues with code. For a tokenizer without
  a chat template it returns the raw task prompt (base-model
  completion convention) — that path is refused here: v0's anchor is an
  instruct model and a silent convention flip would change the metric.
* The instruction/response prefixes are the evalplus CLI defaults,
  codegen.py:176-177 (non-perf variants) — copied verbatim below with
  provenance, since codegen.py hardcodes them inside its CLI function
  (not importable as constants).
* ``evalplus.data.get_human_eval_plus()`` -> {task_id: problem} with
  "prompt" and "entry_point" keys (data/humaneval.py:42).
* ``evalplus.sanitize.sanitize(code, entrypoint=None)``
  (sanitize.py:168) — tree-sitter-based extraction with a code-block
  fallback.
* Problem data cache: ``appdirs.user_cache_dir("evalplus")``
  (data/utils.py:11) — i.e. ``$XDG_CACHE_HOME/evalplus`` when
  XDG_CACHE_HOME is set, else ``~/.cache/evalplus``; full-file override
  via the ``HUMANEVAL_OVERRIDE_PATH`` env var (data/humaneval.py:15).
  Version constant ``HUMANEVAL_PLUS_VERSION = "v0.1.10"`` at 0.3.1.
  The fetch script pre-seeds that cache (pins.yaml `evalplus:` block);
  test_capeval_offline.py verifies a no-network read of it.

Wire formats (frozen by tests/test_capeval_protocol.py):
* samples.jsonl — one ``{"task_id": ..., "solution": ...}`` per line
  (solution = sanitized self-contained code), written ATOMICALLY and
  validated against the pinned 164-task ID set (review tier1 #3/#4).
* generation marker — the cell's atomic ``generation_complete.json`` is
  the DRIVER's to write (capeval_run.py), only after every local phase
  is done and results_pending.json is committed (review tier1 #3);
  ``results.json`` stays the only cell-done marker and is written by the
  execution wave, never here.
* eval results — the 0.3.1-shaped ``*_eval_results.json``
  (``{"date", "hash", "eval": {task_id: [{task_id, base_status,
  plus_status, ...}]}}``) converts to per-item rows via
  ``eval_results_to_rows`` (pass@1_base / pass@1_plus display rows plus
  the normalized ``acc`` gating row = pass@1_plus, simplify-pass #7).

Served route (everything below the "SERVED ROUTE" banner; the HF path
above it is untouched):

* ``dataset`` switch — ``humaneval`` (164 ``HumanEval/<n>``) and ``mbpp``
  (MBPP+ v0.2.0: 378 ``Mbpp/<n>``, the sanitized-MBPP ids ``Mbpp/2`` ..
  ``Mbpp/809``, NOT a contiguous range). Both id sets and the release
  archives' hashes are vendored in ``configs/capeval/evalplus/datasets.json``
  (generated from the downloaded files, never typed) and refused on drift.
* generation goes through :class:`~sisteer.capeval.serving.proxy_engine.
  ProxyGenerateEngine` as CHAT ITEMS: user = evalplus's instruction block
  (``make_raw_chat_prompt``'s ``task_prompt`` f-string), assistant prefill
  = its ``response`` f-string cut at ``_MAGIC_SPLITTER_`` (``response_prefix
  + "\n```python\n"``) — the proxy renders the template server-side. One
  render difference vs the HF path is structural and recorded: HF's
  ``make_raw_chat_prompt`` renders the whole turn and cuts at the
  splitter, so its prompt ends ``"```python\n"``; a continued final message
  is cut at the STRIPPED assistant content (HF/vLLM ``continue_final_message``
  semantics), so the served prompt ends ``"```python"`` and the model emits
  the newline itself. ``sanitize`` is indifferent to it.
* thinking-ON: the prefilled instrument is flag-only and runs in
  both modes (paired); the ``:open`` variant (no prefill, ``max_tokens``
  4096, no stop string, first fenced python block extracted from
  ``content``) is the true thinking-ON readout, a DISTINCT task name.
* every persisted generation was ATTESTED first: the engine
  verifies the proxy's response stamps (``model``, ``petri_intervention_hash``,
  ``petri.cell_id``, ``petri.serving_identity``) against the route before a
  result exists, and the sidecar row records the verified
  ``intervention_hash``; a mismatch is a per-item typed error, never a row.
  A context-length 400 is clamped ONCE (``served_window``) —
  a clamped row records ``max_tokens_clamped_from`` / ``prompt_tokens_source``;
  a prompt that cannot fit is the typed ``prompt_over_window`` error, which
  ``gen_meta.json`` names under ``errors`` so a task blocked on it says why.
  ``n_requests`` counts HTTP attempts (a clamped retry is a second attempt)
  and ``n_rejected`` the non-2xx responses among them (the ledger's meaning;
  the geometry basis counts rejections the same way); ``n_errors`` is the
  number of items the last invocation left ungenerated, ``window_clamp`` /
  ``n_stamp_mismatch`` the cumulative typed counters.
* BOTH channels are recorded: under a composed reasoning parser vLLM
  splits the assistant message into ``reasoning_content`` and ``content``;
  the sidecar row keeps ``text`` (= ``content``, ``""`` when null — the
  only thing sanitize scores) AND ``reasoning_content`` (None when the
  response carried none). A row whose content is empty while reasoning
  is present is FLAGGED ``content_empty_with_reasoning: true`` and still
  scores as-is (no answer within ``max_tokens`` IS the measurement);
  ``gen_meta.json`` counts them (``n_content_empty_with_reasoning``) and
  names them (``content_empty_with_reasoning_item_ids``), and conversion
  stamps the flag into every per-item row's ``extra_json``. A response
  with NEITHER channel under a composed parser is the typed
  ``empty_response`` error (``proxy_engine.EmptyResponse``; cumulative
  ``n_empty_response``) — never a row;
  ``gen_meta.thinking.reasoning_parser`` records the parser the engine
  was told the standup composed (None = none, no refusal arm).
* resume: a per-item ``generations.jsonl`` sidecar (fsync per line) is
  the resume state; ``samples.jsonl`` is written ATOMICALLY only when the
  set is complete. A ``limit`` (smoke) pads the remaining problems with
  empty solutions — evalplus 0.3.1 refuses a samples file missing any
  problem — and ``gen_meta.json`` names the padded ids so conversion
  emits rows for the GENERATED items only and never writes the
  completion marker for a padded run.
* execution is the sandbox orchestrator's (``scripts/modal/
  tinycap_sandbox.py``): it validates ``samples_eval_results.json`` with
  :func:`validate_eval_results` BEFORE anything touches the volume and
  converts with :func:`convert_eval_results_to_store` (rows ``evalplus/
  <dataset>_plus[:open]``, domain ``evalplus_served``, framework
  ``evalplus-0.3.1`` — the row ``task_version`` names the pass@1_plus
  definition beside it, ``evalplus-0.3.1+base_and_plus``; engine
  from the cell manifest's ``row_constants``). A store whose existing rows
  for the task were written under ANOTHER definition is refused rather
  than resumed (:func:`_check_existing_rows_definition`).
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Dict, FrozenSet, Iterable, List,
                    Mapping, Optional, Sequence, Tuple)

from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from sisteer.capeval.serving.proxy_engine import ChatItem, ProxyGenerateEngine
    from sisteer.capeval.store import CellStore

# evalplus 0.3.1 codegen.py:176-177 — the CLI defaults for instruct-model
# generation (the non-perf variants). Copied with provenance because
# codegen.py defines them inside its CLI function body, not as module
# constants; make_raw_chat_prompt itself IS imported, not vendored.
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)

TASK_NAME = "humaneval_plus"  # the domains.yaml `code` task (§2.2)
# Frozen row conversion metrics (tests/test_capeval_protocol.py
# EVALPLUS_METRICS): per task_id, one greedy sample, two per-item display
# rows PLUS the normalized gating row metric="acc" (value = pass@1_plus)
# that the code domain's §5.2 pairing reads (simplify-pass #7).
METRICS = ("pass@1_base", "pass@1_plus")
# Gating convention for the `code` domain (recorded for the integrator /
# stats): HumanEval+ IS the plus variant — pass@1_plus gates;
# pass@1_base is display.
GATING_METRIC = "pass@1_plus"
GATING_ROW_METRIC = "acc"  # the normalized per-doc gating row (§5.2)
#: PW12 (E19): the pass@1_plus DEFINITION every converter of this tree applies
#: — evalplus 0.3.1's own headline, base AND plus (``evaluate.py``
#: ``new_correct``) — named in ``extra_json`` of every ``pass@1_plus`` and
#: gating ``acc`` row and in the summary payload, so a row or summary written
#: under the earlier plus-only reading (no field) is never mistaken for one.
PLUS_METRIC_DEF = C.EVALPLUS_PLUS_DEF_BASE_AND_PLUS

EVALPLUS_PIN = "0.3.1"
# The exact pinned HumanEval+ v0.1.10 task-ID set (164 problems;
# evalplus 0.3.1 REQUIRES a sample per problem). Validated
# before generation is accepted, when samples.jsonl is consumed, and
# before execution results convert (review tier1-runtime #4).
EXPECTED_TASK_IDS = frozenset(f"HumanEval/{i}" for i in range(164))


def validate_task_ids(ids: Iterable[str], *, where: str,
                      expected: frozenset = EXPECTED_TASK_IDS,
                      samples_per_item: int = 1) -> None:
    """Refuse anything but exactly the pinned task-ID set — a truncated
    file, a stale override, or a partial eval must never convert into a
    completed-looking (inflated) pass@1 (tier1 #4). With ``samples_per_item``
    k > 1 (U8 / PW-sampling: k seeded samples per problem) every id must
    appear EXACTLY k times — a task with fewer samples than its siblings
    would get a pass@1 over a different denominator and look like any other."""
    ids = list(ids)
    k = int(samples_per_item)
    if k < 1:
        raise EvalPlusSeamError(f"{where}: samples_per_item must be >= 1, got {samples_per_item!r}")
    counts: Dict[str, int] = {}
    for i in ids:
        counts[i] = counts.get(i, 0) + 1
    if k == 1:
        if len(ids) != len(set(ids)):
            dups = sorted({i for i in ids if counts[i] > 1})[:5]
            raise EvalPlusSeamError(f"{where}: duplicate task_ids (e.g. {dups})")
    else:
        off = sorted(i for i, c in counts.items() if c != k)[:5]
        if off:
            raise EvalPlusSeamError(
                f"{where}: every task_id must carry exactly {k} samples (samples_per_item), but "
                f"e.g. {[(i, counts[i]) for i in off]} do not — a partial or foreign sample set")
    got = set(ids)
    if got != expected:
        missing = sorted(expected - got)[:5]
        extra = sorted(got - expected)[:5]
        raise EvalPlusSeamError(
            f"{where}: task-ID set != the pinned {len(expected)} HumanEval+ "
            f"problems ({len(got)} present; missing e.g. {missing}, "
            f"unexpected e.g. {extra}) — refusing a partial/foreign set "
            "(plan §5.5, review tier1 #4)")


class EvalPlusSeamError(RuntimeError):
    """The installed evalplus does not match the 0.3.1 seams pinned here."""


def _require_pinned_evalplus() -> None:
    """Enforce the evalplus pin at runner entry, before any scored work
    (review tier1-runtime #10): sanitize/prompt behavior is pinned
    against 0.3.1 and a drifted install would silently change the metric."""
    import evalplus  # heavy-lane import

    if getattr(evalplus, "__version__", None) != EVALPLUS_PIN:
        raise EvalPlusSeamError(
            f"installed evalplus {getattr(evalplus, '__version__', '?')} != "
            f"pinned {EVALPLUS_PIN}: prompt convention + sanitize are pinned "
            "against 0.3.1 — re-verify before generating (plan §5.5).")


@dataclass(frozen=True)
class RenderedPrompt:
    task_id: str
    prompt: str
    entry_point: str


# ---------------------------------------------------------------------------
# Generation side (heavy lane: evalplus import + an Engine)
# ---------------------------------------------------------------------------


def load_problems(dataset: str = "humaneval") -> Dict[str, Dict[str, Any]]:
    """EvalPlus problems, {task_id: problem} in evalplus's own order —
    HumanEval+ by default (the HF path), MBPP+ for ``dataset="mbpp"`` (the
    served route's dataset switch). Requires the evalplus data cache
    (pre-seeded by the fetch script; downloads from GitHub
    releases on first use otherwise). The loaded id set is checked against
    the vendored pin: a drifted release never renders."""
    if dataset == "humaneval":
        from evalplus.data import get_human_eval_plus  # heavy-lane import

        problems = get_human_eval_plus()
    elif dataset == "mbpp":
        from evalplus.data import get_mbpp_plus  # heavy-lane import

        problems = get_mbpp_plus()
    else:
        raise EvalPlusSeamError(
            f"unknown evalplus dataset {dataset!r}; datasets: {list(C.EVALPLUS_DATASETS)}")
    validate_task_ids(problems.keys(), where=f"load_problems({dataset!r})",
                      expected=expected_task_ids(dataset))
    return problems


def render_prompts(
    tokenizer: Any,
    problems: Optional[Mapping[str, Mapping[str, Any]]] = None,
    *,
    limit: Optional[int] = None,
) -> List[RenderedPrompt]:
    """Chat prompts per evalplus's own convention at the 0.3.1 pin
    (make_raw_chat_prompt — §5.5)."""
    from evalplus.provider.utility import make_raw_chat_prompt  # heavy-lane

    if getattr(tokenizer, "chat_template", None) is None:
        raise EvalPlusSeamError(
            "tokenizer has no chat_template: make_raw_chat_prompt would fall "
            "back to the base-model completion convention, silently changing "
            "the metric — v0's anchor is an instruct model (plan §5.5)."
        )
    items = list((problems or load_problems()).items())
    if limit is not None:
        items = items[:limit]
    return [
        RenderedPrompt(
            task_id=tid,
            prompt=make_raw_chat_prompt(
                p["prompt"], INSTRUCTION_PREFIX, RESPONSE_PREFIX, tokenizer
            ),
            entry_point=p["entry_point"],
        )
        for tid, p in items
    ]


def sanitize_solution(generation: str, entry_point: str) -> str:
    """evalplus.sanitize at the 0.3.1 pin (§5.5)."""
    from evalplus.sanitize import sanitize  # heavy-lane import

    return sanitize(generation, entrypoint=entry_point)


def write_samples_jsonl(path: str | Path, samples: Iterable[Mapping[str, Any]]) -> Path:
    """The fixture-pinned samples.jsonl: {"task_id", "solution"} per line.

    Written ATOMICALLY (tmp + fsync + replace): the file's existence is
    the generation-resume signal, so a torn partial write must never be
    observable (review tier1-runtime #3)."""
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for s in samples:
            if set(s) != {"task_id", "solution"}:
                raise ValueError(
                    f"samples.jsonl rows carry exactly task_id+solution "
                    f"(protocol fixture); got keys {sorted(s)}"
                )
            f.write(json.dumps(s, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def generate_samples(
    *,
    engine: Any,
    arm: Any,
    tokenizer: Any,
    gen_cfg: Mapping[str, Any],
    store: Any,
    problems: Optional[Mapping[str, Mapping[str, Any]]] = None,
    expected_task_ids: frozenset = EXPECTED_TASK_IDS,
) -> List[Dict[str, str]]:
    """The §5.5 generation step for one cell, any engine/arm: render with
    evalplus's chat convention, generate greedily through the engine seam
    (steering ctx active inside engine.generate), sanitize, validate the
    exact pinned task-ID set (tier1 #4 — a partial set never persists),
    and write samples.jsonl ATOMICALLY.

    ``generation_complete.json`` is NOT written here (review tier1 #3):
    the DRIVER writes it only after every local phase is complete and
    ``results_pending.json`` is committed, so a crash between generation
    and the parity probes can never let the execution wave finalize a
    cell whose local rows are missing. ``store`` supplies ``cell_dir``
    only; results.json belongs to the execution wave.
    """
    if problems is None:  # the production path — the pin binds it
        _require_pinned_evalplus()
    prompts = render_prompts(tokenizer, problems)
    validate_task_ids([p.task_id for p in prompts],
                      where="generate_samples: rendered problems",
                      expected=expected_task_ids)
    cfg = dict(gen_cfg)
    # Bound the generation batch (same guard as the tier-0 canary path,
    # tier0.py): the engine's no-batch_size default is ONE chunk of ALL
    # prompts, and one 164-problem chunk at max_new=768 retains ~65 GB
    # of fp32 per-step scores (output_scores, |V|=128k) plus ~25 GB of
    # KV cache — which OOM'd an H200 next to control:weight_noise's
    # ~48 GB of on-GPU weight caches (job 115651_14). A configured
    # batch_size still wins; the config hash is untouched (F8).
    cfg.setdefault("batch_size", 16)
    gens = engine.generate([p.prompt for p in prompts], arm, cfg)
    if len(gens) != len(prompts):
        raise EvalPlusSeamError(
            f"engine.generate returned {len(gens)} outputs for "
            f"{len(prompts)} prompts"
        )
    samples = [
        {"task_id": p.task_id, "solution": sanitize_solution(g.text, p.entry_point)}
        for p, g in zip(prompts, gens)
    ]
    write_samples_jsonl(Path(store.cell_dir) / "samples.jsonl", samples)
    return samples


# ---------------------------------------------------------------------------
# Execution-results side (pure — the interface the execution wave consumes;
# pinned by the frozen fixture pair)
# ---------------------------------------------------------------------------


def eval_results_to_rows(
    results: Mapping[str, Any],
    expected_task_ids: Optional[frozenset] = None,
    *,
    task: str = TASK_NAME,
    domain: str = "code",
    task_version: str = "evalplus-0.3.1",
    samples_per_item: int = 1,
    per_sample_rows: bool = False,
) -> List[Dict[str, Any]]:
    """Convert an evalplus ``*_eval_results.json`` dict into per-item row
    dicts (task="humaneval_plus", item_id=task_id): the two display rows
    pass@1_base/pass@1_plus PLUS the normalized gating row metric="acc"
    with value = pass@1_plus — the registered gating metric the code
    domain pairs on (simplify-pass #7; without it the code domain never
    gated).

    ``pass@1_plus`` is evalplus's OWN headline definition (its
    ``evaluate.py``: a sample counts for the plus set only when it passes
    the BASE tests AND the plus tests — ``new_correct`` there), so the
    store's plus number is the number evalplus prints. Until PW10 this
    scored ``plus_status`` alone, which counts a base-fail / plus-pass
    sample as a plus PASS: +2 items on every MBPP+ run (Mbpp/631, Mbpp/635
    on all three 618f9dd conversions), +0.5 pp over evalplus's headline.
    ``pass@1_base`` is ``base_status`` alone, as before.

    ``samples_per_item`` (U8 / PW-sampling; 1 = the HF path and every
    pre-U8 served store, whose rows are byte-identical to before) is the
    task's k: every task_id must carry EXACTLY k samples (fewer or more is a
    refusal, never an average over whatever came back); a missing status
    field is a refusal, never a fail default. Each sample files a
    per-sample ``acc@<i>`` row (``contracts.sample_metric``; the plus
    definition, ``sample_idx`` and the sample's base value in its
    ``extra_json``) whenever the task drew more than one sample OR
    ``per_sample_rows`` is set — R16-A (E45 M-EvalPlus): a k = 1 draw is
    still A DRAW, and a store that recorded the sampling contract must be
    able to say which sample each number came from, so its lone sample gets
    an ``acc@0`` row like any other. ``per_sample_rows`` defaults to False so
    that the HF path and every pre-sampling served store keep the exact k = 1
    rows they were written under; :func:`convert_eval_results_to_store` and
    :func:`rederive_evalplus` pass True iff the task's ``gen_meta.json``
    carries the ``sampling`` block (i.e. was generated under the contract),
    and :func:`_join_sample_provenance` then puts the draw's ``seed`` /
    ``temperature`` / ``top_p`` / ``top_k`` on every one of those rows.
    With k > 1 the per-item ``pass@1_base`` / ``pass@1_plus`` / ``acc``
    rows carry the UNBIASED pass@1 estimator over the k samples
    (:func:`pass_at_k_unbiased` at k = 1 — exactly c / n; named as
    :data:`PASS_AT_1_ESTIMATOR` in their ``extra_json`` with
    ``samples_per_item``, the per-sample ``values`` and ``n_correct``) — the
    per-item value the pairing tool reads and reduces on, the per-sample rows
    beside it. ``expected_task_ids`` (pass EXPECTED_TASK_IDS on the real
    conversion path) additionally enforces the pinned 164-ID set
    (tier1 #4).

    ``task`` / ``domain`` / ``task_version`` default to the HF path's frozen
    names (tests/test_capeval_protocol.py); the served route passes
    ``contracts.evalplus_task(...)`` / ``DOMAIN_EVALPLUS_SERVED`` /
    ``EVALPLUS_TASK_VERSION`` (P31: same framework pin string, the task
    NAME and the engine column keep the paths apart). ``task_version`` is
    the FRAMEWORK pin; the rows carry
    ``contracts.evalplus_row_task_version(PLUS_METRIC_DEF, pin=task_version)``
    — ``evalplus-0.3.1+base_and_plus`` — so the definition is identity on
    the row itself (PW13 / E21 M1), and the bare pin stays the literal of
    the retained plus-only rows only."""
    if expected_task_ids is not None:
        validate_task_ids(results["eval"].keys(),
                          where="eval_results_to_rows",
                          expected=expected_task_ids)
    # the definition rides in the row's identity, never only in extra_json (PW13)
    row_version = C.evalplus_row_task_version(PLUS_METRIC_DEF, pin=task_version)
    n_samples = int(samples_per_item)
    if n_samples < 1:
        raise ValueError(f"samples_per_item must be >= 1, got {samples_per_item!r}")
    emit_per_sample = bool(per_sample_rows) or n_samples > 1
    rows: List[Dict[str, Any]] = []
    for task_id, items in results["eval"].items():
        if len(items) != n_samples:
            raise ValueError(
                f"{task_id}: expected exactly {n_samples} sample(s) per task "
                f"(samples_per_item{'' if n_samples > 1 else ': one greedy sample in v0'}), "
                f"got {len(items)} (plan §5.5)"
            )
        base_values: List[float] = []
        plus_values: List[float] = []
        for item in items:
            for key in ("base_status", "plus_status"):
                if key not in item:
                    raise ValueError(
                        f"{task_id}: eval result item lacks {key!r} — evalplus "
                        "output shape drifted; refusing to convert (plan §5.5)")
            b = 1.0 if item["base_status"] == "pass" else 0.0
            # evalplus's headline: plus passes only if base does too (PW10); the
            # definition rides in extra_json so no other reading shares its identity (PW12)
            base_values.append(b)
            plus_values.append(1.0 if (b and item["plus_status"] == "pass") else 0.0)
        base = {
            "tier": 1,
            "task": task,
            "task_version": row_version,
            "domain": domain,
            "item_id": task_id,
        }
        plus_def = {C.EVALPLUS_METRIC_DEF_FIELD: PLUS_METRIC_DEF}
        # U8: one row per SAMPLE — since R16-A also for a lone one, whenever the
        # caller asks (a store written under the sampling contract does)
        if emit_per_sample:
            for idx, (b, pv) in enumerate(zip(base_values, plus_values)):
                rows.append({
                    **base,
                    "metric": C.sample_metric(GATING_ROW_METRIC, idx),
                    "value": pv,
                    "extra_json": json.dumps(
                        {"source_metric": GATING_METRIC, **plus_def, "sample_idx": idx,
                         "value_base": b}, sort_keys=True),
                })
        if n_samples == 1:
            # the pre-U8 per-item rows, byte for byte (the frozen HF fixture pins them)
            base_value, plus_value = base_values[0], plus_values[0]
            per_item_extra_base: Optional[str] = None
            per_item_extra_plus = json.dumps(plus_def, sort_keys=True)
            per_item_extra_acc = json.dumps({"source_metric": GATING_METRIC, **plus_def}, sort_keys=True)
        else:
            # ... then the unbiased pass@1 per item (= c / n)
            c_base, c_plus = int(sum(base_values)), int(sum(plus_values))
            base_value = pass_at_k_unbiased(n_samples, c_base, 1)
            plus_value = pass_at_k_unbiased(n_samples, c_plus, 1)
            common = {"samples_per_item": n_samples, "estimator": PASS_AT_1_ESTIMATOR}
            per_item_extra_base = json.dumps(
                {**common, "values": base_values, "n_correct": c_base}, sort_keys=True)
            per_item_extra_plus = json.dumps(
                {**plus_def, **common, "values": plus_values, "n_correct": c_plus}, sort_keys=True)
            per_item_extra_acc = json.dumps(
                {"source_metric": GATING_METRIC, **plus_def, **common, "values": plus_values,
                 "n_correct": c_plus}, sort_keys=True)
        for metric, value, extra in (
            ("pass@1_base", base_value, per_item_extra_base),
            ("pass@1_plus", plus_value, per_item_extra_plus),
        ):
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "value": value,
                    "extra_json": extra,
                }
            )
        rows.append(
            {
                **base,
                "metric": GATING_ROW_METRIC,
                "value": plus_value,
                "extra_json": per_item_extra_acc,
            }
        )
    return rows


def read_samples_jsonl(
    path: str | Path,
    expected_task_ids: Optional[frozenset] = None,
    *,
    samples_per_item: int = 1,
) -> List[Dict[str, Any]]:
    """Read samples.jsonl; with ``expected_task_ids`` the pinned ID set
    is re-validated at consumption (tier1 #4 — a truncated or foreign
    file must refuse, never quietly execute), at ``samples_per_item``
    lines per id (the task's k, read off its ``gen_meta.json``; U8)."""
    samples = [json.loads(line)
               for line in Path(path).read_text().splitlines() if line]
    if expected_task_ids is not None:
        validate_task_ids([s["task_id"] for s in samples],
                          where=f"read_samples_jsonl({path})",
                          expected=expected_task_ids, samples_per_item=samples_per_item)
    return samples


# ---------------------------------------------------------------------------
# Goldens: the exact rendered chat prompt for 2 fixed items
# ---------------------------------------------------------------------------

GOLDEN_TASK_IDS = ("HumanEval/0", "HumanEval/1")


def render_prompt_goldens(
    tokenizer: Any,
    *,
    task_ids: Sequence[str] = GOLDEN_TASK_IDS,
    tokenizer_meta: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    problems = load_problems()
    missing = [t for t in task_ids if t not in problems]
    if missing:
        raise EvalPlusSeamError(f"golden task ids missing from HumanEval+: {missing}")
    rendered = render_prompts(tokenizer, {t: problems[t] for t in task_ids})
    return {
        "evalplus_version": "0.3.1",
        "task": TASK_NAME,
        "prompt_convention": "make_raw_chat_prompt (evalplus 0.3.1, §5.5)",
        "instruction_prefix": INSTRUCTION_PREFIX,
        "response_prefix": RESPONSE_PREFIX,
        "tokenizer": dict(tokenizer_meta or {}),
        "items": [
            {
                "task_id": r.task_id,
                "entry_point": r.entry_point,
                "prompt": r.prompt,
            }
            for r in rendered
        ],
    }


# =========================================================================== #
# SERVED ROUTE — everything above is the HF path, untouched. Names come from  #
# sisteer.capeval.tinycap.contracts.                                          #
# =========================================================================== #

#: Vendored release facts + exact task-id sets (module docstring). Generated
#: from the downloaded archives; ``SISTEER_EVALPLUS_DATASETS`` overrides the
#: repo-relative location (the Modal orchestrator image mounts the file
#: outside a checkout).
DATASETS_JSON_RELPATH = "configs/capeval/evalplus/datasets.json"
DATASETS_JSON_ENV = "SISTEER_EVALPLUS_DATASETS"

#: evalplus 0.3.1 ``provider/utility.py`` — the splitter ``make_raw_chat_prompt``
#: cuts the rendered assistant turn at. Copied with provenance (it is a
#: module-private name upstream); the heavy-lane test pins equality.
EVALPLUS_MAGIC_SPLITTER = "-[[]]-this-is-really-our-highest-priority-[[]]-"
#: The assistant prefill the served instrument continues: evalplus's
#: ``response`` f-string up to the splitter (``response_prefix`` + an open
#: python fence). The proxy/vLLM cut a continued final message at its
#: STRIPPED content, so the served prompt ends at ``"```python"`` (PC2).
EVALPLUS_ASSISTANT_PREFILL = f"{RESPONSE_PREFIX}\n```python\n"

#: Generation configs. The stock instrument (the HF tier-1 generation and
#: the served one alike) is greedy, 768, stop at the closing fence;
#: the ``:open`` variant (P12) has no prefill and NO stop string — a fence
#: inside the thought channel must not truncate the answer — and a 4096 cap.
DEFAULT_GEN_CFG: Mapping[str, Any] = {"do_sample": False, "max_new_tokens": 768,
                                      "until": ["```"]}
OPEN_VARIANT_GEN_CFG: Mapping[str, Any] = {"do_sample": False, "max_new_tokens": 4096,
                                           "until": []}
#: The per-(task, item, sample) request seed rule the engine applies
#: (``proxy_engine.stable_seed``), recorded in ``gen_meta.json`` and the cell
#: manifest's ``seeds`` block — the contract's :data:`contracts.SEED_RULE_EVALPLUS`
#: (hashed identity since U8; the derivation stays in the engine).
SEED_RULE = C.SEED_RULE_EVALPLUS

#: With k samples per problem (U8 / PW-sampling), the per-item ``pass@1_base``
#: / ``pass@1_plus`` / ``acc`` rows carry the UNBIASED pass@1 estimator over
#: the n = k samples (Chen et al. 2021, ``1 - C(n-c, k)/C(n, k)`` at k = 1 =
#: c / n) beside the per-sample ``acc@<i>`` rows; the summary's
#: ``pass_at_1_*`` are the means of those per-item estimates. Named on every
#: multi-sample per-item row's ``extra_json`` so a reader never has to guess
#: which reduction produced the number.
PASS_AT_1_ESTIMATOR = "unbiased pass@1 over n samples: 1 - C(n-c,1)/C(n,1) = c/n (Chen et al. 2021)"


def pass_at_k_unbiased(n: int, c: int, k: int = 1) -> float:
    """The Chen et al. (2021) unbiased pass@k over ``n`` samples of which
    ``c`` passed: ``1 - C(n-c, k) / C(n, k)``, computed as the stable product
    ``1 - prod_{i=n-c+1}^{n} (1 - k/i)``; 1.0 when ``n - c < k``. At k = 1 it
    is exactly ``c / n``. Refuses an impossible triple."""
    n, c, k = int(n), int(c), int(k)
    if n < 1 or k < 1 or c < 0 or c > n or k > n:
        raise ValueError(f"pass@k needs 1 <= k <= n and 0 <= c <= n, got n={n} c={c} k={k}")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod

#: Per-task files under ``served_task_dir``.
GENERATIONS_FILENAME = "generations.jsonl"   # resume sidecar, one raw generation per line
SAMPLES_FILENAME = "samples.jsonl"           # the sandbox INPUT (fixture-pinned shape)
GEN_META_FILENAME = "gen_meta.json"          # what was generated, how, and what was padded
EVAL_RESULTS_FILENAME = "eval_results.json"  # the validated sandbox OUTPUT (orchestrator-written)
EXECUTION_FILENAME = "execution.json"        # sandbox provenance (orchestrator-written)
SUMMARY_FILENAME = "summary.json"            # counts-only payload (SUMMARY_PAYLOAD_FIELDS)

#: The 0.3.1 ``samples_eval_results.json`` item shape (evaluate.py:287-295):
#: the two status axes named by the contract (``SANDBOX_RESULTS_KEYS`` ->
#: ``<axis>_status``, PC1), the echoed task_id, and the optional
#: solution / failed-test lists. Anything else is drift and refuses.
RESULT_STATUS_KEYS: Tuple[str, ...] = tuple(f"{k}_status" for k in C.SANDBOX_RESULTS_KEYS)
RESULT_ITEM_REQUIRED_KEYS: FrozenSet[str] = frozenset(("task_id",) + RESULT_STATUS_KEYS)
RESULT_ITEM_ALLOWED_KEYS: FrozenSet[str] = RESULT_ITEM_REQUIRED_KEYS | frozenset(
    ("solution", "base_fail_tests", "plus_fail_tests"))
RESULT_TOP_KEYS: FrozenSet[str] = frozenset(("date", "hash", "eval"))
#: evalplus.eval PASS / FAIL / TIMEOUT; ``None`` is the ``base_only`` run we never make.
RESULT_STATUSES: Tuple[str, ...] = ("pass", "fail", "timeout")


# --------------------------------------------------------------------------- #
# dataset pins                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DatasetPin:
    dataset: str
    name: str            # HumanEvalPlus / MbppPlus (the release repo's name)
    version: str
    url: str
    gz_filename: str
    gz_bytes: int
    gz_sha256: str
    gz_md5: str          # == evalplus's dataset hash when the .gz is the override file
    override_env: str    # HUMANEVAL_OVERRIDE_PATH / MBPP_OVERRIDE_PATH
    task_ids: FrozenSet[str]

    @property
    def n_tasks(self) -> int:
        return len(self.task_ids)


def datasets_json_path() -> Path:
    env = os.environ.get(DATASETS_JSON_ENV)
    if env:
        return Path(env)
    # src/sisteer/capeval/tiers/evalplus_runner.py -> parents[4] == repo root
    return Path(__file__).resolve().parents[4] / DATASETS_JSON_RELPATH


@functools.lru_cache(maxsize=None)
def _dataset_pins() -> Dict[str, DatasetPin]:
    path = datasets_json_path()
    if not path.is_file():
        raise EvalPlusSeamError(
            f"vendored evalplus dataset pins not found at {path} (set ${DATASETS_JSON_ENV} "
            "to point at configs/capeval/evalplus/datasets.json)")
    doc = json.loads(path.read_text())
    pins: Dict[str, DatasetPin] = {}
    for dataset, e in doc["datasets"].items():
        if dataset not in C.EVALPLUS_DATASETS:
            raise EvalPlusSeamError(f"{path}: unknown dataset {dataset!r}")
        ids = list(e["task_ids"])
        if len(ids) != len(set(ids)) or len(ids) != int(e["n_tasks"]):
            raise EvalPlusSeamError(f"{path}: {dataset} task_ids are not {e['n_tasks']} unique ids")
        pins[dataset] = DatasetPin(
            dataset=dataset, name=e["name"], version=e["version"], url=e["url"],
            gz_filename=e["gz_filename"], gz_bytes=int(e["gz_bytes"]),
            gz_sha256=e["gz_sha256"], gz_md5=e["gz_md5"], override_env=e["override_env"],
            task_ids=frozenset(ids))
    missing = sorted(set(C.EVALPLUS_DATASETS) - set(pins))
    if missing:
        raise EvalPlusSeamError(f"{path}: no pin for dataset(s) {missing}")
    if pins["humaneval"].task_ids != EXPECTED_TASK_IDS:
        raise EvalPlusSeamError(
            f"{path}: the vendored HumanEval+ id set differs from EXPECTED_TASK_IDS")
    if pins["humaneval"].n_tasks != C.EXPECTED_HUMANEVAL_PLUS_N or \
            pins["mbpp"].n_tasks != C.EXPECTED_MBPP_PLUS_N:
        raise EvalPlusSeamError(
            f"{path}: task counts {pins['humaneval'].n_tasks}/{pins['mbpp'].n_tasks} != the "
            f"contract's {C.EXPECTED_HUMANEVAL_PLUS_N}/{C.EXPECTED_MBPP_PLUS_N}")
    return pins


def dataset_pin(dataset: str) -> DatasetPin:
    if dataset not in C.EVALPLUS_DATASETS:
        raise EvalPlusSeamError(
            f"unknown evalplus dataset {dataset!r}; datasets: {list(C.EVALPLUS_DATASETS)}")
    return _dataset_pins()[dataset]


def expected_task_ids(dataset: str) -> FrozenSet[str]:
    """The exact pinned id set per dataset: the frozen 164 for HumanEval+,
    the vendored 378 sanitized-MBPP ids for MBPP+."""
    return dataset_pin(dataset).task_ids


# --------------------------------------------------------------------------- #
# chat items — evalplus's own prompt convention, split at the splitter        #
# --------------------------------------------------------------------------- #


def evalplus_user_content(task_prompt: str) -> str:
    """``make_raw_chat_prompt``'s ``task_prompt`` f-string (evalplus 0.3.1
    provider/utility.py:36-41): the instruction prefix + the stripped
    problem in a bare fence. codegen.py:80 feeds ``prompt.strip() + "\\n"``,
    which the inner strip makes idempotent — same bytes for both datasets."""
    return f"{INSTRUCTION_PREFIX}\n```\n{task_prompt.strip()}\n```\n"


def evalplus_response_content(splitter: str = EVALPLUS_MAGIC_SPLITTER) -> str:
    """``make_raw_chat_prompt``'s ``response`` f-string (utility.py:42-46);
    the served prefill is everything before the splitter."""
    return f"{RESPONSE_PREFIX}\n```python\n{splitter}\n```\n"


def render_chat_items(
    problems: Mapping[str, Mapping[str, Any]],
    *,
    open_variant: bool = False,
    limit: Optional[int] = None,
    samples_per_item: int = 1,
) -> List["ChatItem"]:
    """``samples_per_item`` :class:`ChatItem`s per problem in evalplus order
    (problem-major, then ``sample_idx`` 0..k-1 — distinct seeds through the
    engine's rule; U8 / PW-sampling): user = the instruction block, assistant
    prefill = ``EVALPLUS_ASSISTANT_PREFILL`` (None for the ``:open``
    variant). ``limit`` is the smoke head cap on PROBLEMS."""
    from sisteer.capeval.serving.proxy_engine import ChatItem

    k = int(samples_per_item)
    if k < 1:
        raise EvalPlusSeamError(f"samples_per_item must be >= 1, got {samples_per_item!r}")
    items = list(problems.items())
    if limit is not None:
        items = items[: int(limit)]
    return [ChatItem(item_id=tid, user=evalplus_user_content(p["prompt"]),
                     assistant_prefill=None if open_variant else EVALPLUS_ASSISTANT_PREFILL,
                     sample_idx=idx)
            for tid, p in items for idx in range(k)]


_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)(?:\r?\n)?```", re.DOTALL)


def extract_first_python_block(text: str) -> Tuple[str, str]:
    """The ``:open`` variant's extraction (P12): the FIRST fenced block
    tagged ``python``/``py``, else the first fenced block of any tag, else
    the whole text (sanitize's own code_extract fallback handles it).
    Returns ``(code, mode)`` with mode in ``fenced_python`` / ``fenced`` /
    ``unfenced`` so the run records how often the model skipped the fence."""
    blocks = _FENCE_RE.findall(text or "")
    for tag, body in blocks:
        if tag.lower() in ("python", "py", "python3"):
            return body, "fenced_python"
    if blocks:
        return blocks[0][1], "fenced"
    return text or "", "unfenced"


def sanitize_open(generation: str, entry_point: str) -> Tuple[str, str]:
    code, mode = extract_first_python_block(generation)
    return sanitize_solution(code, entry_point), mode


# --------------------------------------------------------------------------- #
# served layout + generation                                                   #
# --------------------------------------------------------------------------- #


def served_task_dir(cell_dir: str | Path, dataset: str, open_variant: bool = False) -> Path:
    """``<cell_dir>/evalplus/<dataset>[_open]/`` — samples, sidecar, meta,
    and (orchestrator-written) eval results + execution provenance."""
    if dataset not in C.EVALPLUS_DATASETS:
        raise EvalPlusSeamError(f"unknown evalplus dataset {dataset!r}")
    return Path(cell_dir) / "evalplus" / (dataset + ("_open" if open_variant else ""))


def done_marker_path(cell_dir: str | Path, task: str) -> Path:
    """``<cell_dir>/done/<task with "/" -> "__">.json`` (contracts DONE_DIR)."""
    return completion.done_marker_path(cell_dir, task)


def served_gen_cfg(gen_cfg: Optional[Mapping[str, Any]], open_variant: bool) -> Dict[str, Any]:
    if open_variant:
        return dict(OPEN_VARIANT_GEN_CFG)
    return dict(gen_cfg if gen_cfg is not None else DEFAULT_GEN_CFG)


def thinking_render(thinking: str, *, open_variant: bool) -> Optional[str]:
    """P25's label: under thinking-ON the prefilled instrument is
    ``flag_only`` (the flag renders, thoughts are impossible after a
    prefill) and the ``:open`` variant thinks ``native``; under OFF there
    is no render to label (None)."""
    if not C.enable_thinking(thinking):
        return None
    return C.THINKING_RENDER_NATIVE if open_variant else C.THINKING_RENDER_FLAG_ONLY


#: The shared tmp + fsync + replace writer (``gen_meta.json``, ``summary.json``,
#: and — from the sandbox orchestrator — ``eval_results.json`` / ``execution.json``);
#: kept under this name because the orchestrator calls ``er._atomic_write_json``.
_atomic_write_json = completion.atomic_write_json


def _read_json(path: Path, *, what: str) -> Dict[str, Any]:
    if not path.is_file():
        raise EvalPlusSeamError(f"{what} missing at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_generations(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """The resume sidecar: ``{generation key: record}`` — the key is
    ``proxy_engine.generation_key(item_id, sample_idx)`` =
    ``<item_id>/<sample_idx>`` (the ledger's ``X-Tinycap-Sample`` value; k
    samples of one item are k records — U8 / PW-sampling; a record written
    before the switch carries no ``sample_idx`` and reads as sample 0). A torn
    FINAL line (an append cut by a kill) is dropped with a notice so that
    generation reruns; a corrupt non-final line is file damage and refuses
    (the store's rule)."""
    from sisteer.capeval.serving.proxy_engine import generation_key
    path = Path(path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out: Dict[str, Dict[str, Any]] = {}
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not text.endswith("\n"):
                print(f"[evalplus-served] {path}: dropped a torn final line; that item reruns")
                break
            raise EvalPlusSeamError(f"{path}: corrupt line {i + 1} (not a torn append) — refusing")
        key = generation_key(rec["item_id"], int(rec.get("sample_idx", 0)))
        if key in out:
            raise EvalPlusSeamError(f"{path}: duplicate generation for {key!r}")
        out[key] = rec
    return out


def _append_generation(path: Path, rec: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def generate_samples_served(
    *,
    engine: "ProxyGenerateEngine",
    dataset: str,
    cell_dir: str | Path,
    open_variant: bool = False,
    gen_cfg: Optional[Mapping[str, Any]] = None,
    problems: Optional[Mapping[str, Mapping[str, Any]]] = None,
    limit: Optional[int] = None,
    expected: Optional[FrozenSet[str]] = None,
    limits: Optional[Mapping[str, int]] = None,
    samples_per_item: int = 1,
) -> Dict[str, Any]:
    """The served generation step for one (dataset, variant): render chat
    items, fan out through ``engine``, keep every raw generation in the
    resume sidecar, and — only on a COMPLETE set — sanitize and write
    ``samples.jsonl`` atomically plus ``gen_meta.json``. Any failed request
    raises after the successful ones are on disk; a re-invoke fills the
    gaps (the served driver's discipline). Returns the counts-only summary
    of this call. ``problems`` / ``expected`` are the test seam (synthetic
    problem sets); the production path loads the pinned dataset. ``limits``
    is the driver's active limit set (R2), recorded as ``driver_limits`` in
    ``gen_meta.json`` so conversion writes a PARTIAL marker for the task
    even when the item set itself was not padded.

    ``samples_per_item`` (U8 / PW-sampling) is THIS task's k: k
    :class:`ChatItem`s per problem with distinct ``sample_idx`` (distinct
    seeds), each a SEPARATE request (never ``n``), each its own sidecar
    record keyed ``<item_id>/<sample_idx>``, and ``samples.jsonl`` carries k
    lines per task_id (evalplus's own multi-sample shape — its ``evaluate``
    files every sample under the task id); a padded (smoke) id gets k empty
    solutions so the file is uniform. ``gen_meta`` records
    ``samples_per_item`` and the engine's sampling row."""
    from sisteer.capeval.serving import proxy_engine as pe
    from sisteer.capeval.serving import served_window as sw

    t0 = time.time()
    n_samples = int(samples_per_item)
    if n_samples < 1:
        raise EvalPlusSeamError(f"samples_per_item must be >= 1, got {samples_per_item!r}")
    task = C.evalplus_task(dataset, open_variant=open_variant)
    tdir = served_task_dir(cell_dir, dataset, open_variant)
    tdir.mkdir(parents=True, exist_ok=True)
    expected = expected if expected is not None else expected_task_ids(dataset)
    if problems is None:  # the production path — the pin binds it
        _require_pinned_evalplus()
        problems = load_problems(dataset)
    validate_task_ids(problems.keys(), where=f"generate_samples_served({dataset})",
                      expected=expected)
    if limit is not None and int(limit) <= 0:
        limit = None
    items = render_chat_items(problems, open_variant=open_variant, limit=limit, samples_per_item=n_samples)
    cfg = served_gen_cfg(gen_cfg, open_variant)
    thinking = engine.thinking
    render = thinking_render(thinking, open_variant=open_variant)

    sidecar = tdir / GENERATIONS_FILENAME
    meta_path = tdir / GEN_META_FILENAME
    prior = _read_json(meta_path, what="gen_meta") if meta_path.exists() else {}
    if prior and prior.get("task") != task:
        raise EvalPlusSeamError(f"{meta_path} records task {prior.get('task')!r}, expected {task!r}")
    if prior and int(prior.get("samples_per_item", 1)) != n_samples:
        raise EvalPlusSeamError(
            f"{meta_path} records samples_per_item={prior.get('samples_per_item', 1)}, this run asks "
            f"{n_samples} — a task's k is identity (U8); resume it at its own k or generate into a fresh store")
    done = read_generations(sidecar)
    todo = [it for it in items if pe.generation_key(it.item_id, it.sample_idx) not in done]
    n_requests = int(prior.get("n_requests", 0))
    n_rejected = int(prior.get("n_rejected", 0))
    prior_clamp = prior.get("window_clamp") if isinstance(prior.get("window_clamp"), dict) else {}
    clamp: Dict[str, Any] = {
        "max_model_len": engine.max_model_len,
        "min_clamped_completion": sw.MIN_CLAMPED_COMPLETION,
        "n_clamped": int(prior_clamp.get("n_clamped", 0)),
        "n_prompt_over_window": int(prior_clamp.get("n_prompt_over_window", 0)),
        "n_tokenize_resolved": int(prior_clamp.get("n_tokenize_resolved", 0)),
        "n_floor_only_unresolvable": int(prior_clamp.get("n_floor_only_unresolvable", 0)),
        "window_mismatches": list(prior_clamp.get("window_mismatches") or []),
    }
    n_stamp_mismatch = int(prior.get("n_stamp_mismatch", 0))
    n_empty_response = int(prior.get("n_empty_response", 0))
    errors: List[Any] = []
    if todo:
        results, errors = engine.generate_items(todo, cfg)
        st = engine.stats
        n_requests += st.n_attempts
        n_rejected += st.n_rejected
        n_stamp_mismatch += st.n_stamp_mismatch
        n_empty_response += st.n_empty_response
        for k in ("n_clamped", "n_prompt_over_window", "n_tokenize_resolved",
                  "n_floor_only_unresolvable"):
            clamp[k] += getattr(st, k)
        clamp["window_mismatches"].extend(
            m for m in st.window_mismatches if m not in clamp["window_mismatches"])
        for it in todo:
            key = pe.generation_key(it.item_id, it.sample_idx)
            r = results.get(key)
            if r is None:
                continue  # its error is in `errors`; the generation reruns on resume
            # both channels, verbatim: `text` is content ("" when null) and is
            # what sanitize scores; `reasoning_content` is the thought (None
            # when the response carried none) and is never scored
            rec = {"item_id": it.item_id, "sample_idx": it.sample_idx, "seed": r.seed,
                   "text": r.text, "finish_reason": r.finish_reason,
                   "prompt_tokens": r.prompt_tokens,
                   "completion_tokens": r.completion_tokens,
                   "reasoning_content": r.reasoning,
                   "reasoning_chars": (len(r.reasoning) if r.reasoning is not None else None),
                   "content_empty_with_reasoning": bool(r.content_empty_with_reasoning),
                   "latency_s": round(r.latency_s, 3),
                   "intervention_hash": r.intervention_hash}
            if r.max_tokens_clamped_from is not None:
                rec["max_tokens_clamped_from"] = r.max_tokens_clamped_from
                rec["prompt_tokens_source"] = r.prompt_tokens_source
            _append_generation(sidecar, rec)
            done[key] = rec

    generated_ids = [it.item_id for it in items if it.sample_idx == 0]   # the PROBLEMS, once each
    gen_keys = [pe.generation_key(it.item_id, it.sample_idx) for it in items]
    # the flagged GENERATIONS, read off the persisted sidecar (exact across
    # resumes): per generation key, and the distinct items any sample of
    # which was flagged (== the keys' items at k = 1, the pre-U8 record)
    flagged_keys = [g for g in gen_keys if g in done and done[g].get("content_empty_with_reasoning")]
    flagged = [i for i in generated_ids
               if any(done.get(pe.generation_key(i, idx), {}).get("content_empty_with_reasoning")
                      for idx in range(n_samples))]
    row = engine.sampling
    meta: Dict[str, Any] = {
        "task": task, "dataset": dataset, "open_variant": bool(open_variant),
        "framework": C.EVALPLUS_TASK_VERSION,
        "thinking": {"mode": thinking, "enable_thinking": C.enable_thinking(thinking),
                     "render": render, "reasoning_parser": engine.reasoning_parser},
        # U8 / PW-sampling: this task's k and the row every generation was drawn under
        "samples_per_item": n_samples,
        "sampling": {"temperature": float(row.temperature),
                     "top_p": (None if row.top_p is None else float(row.top_p)),
                     "top_k": (None if row.top_k is None else int(row.top_k)),
                     "source": row.source},
        "n_generations": len(items),
        "route": {"virtual_model": engine.route.virtual_model,
                  "cell_id": engine.route.cell_id,
                  "serving_identity": engine.route.serving_identity,
                  "intervention_hash": engine.route.intervention_hash,
                  "declared_response_prefill": bool(engine.declare and not open_variant)},
        "base_url": engine.base_url, "concurrency": engine.concurrency,
        "gen_cfg": cfg, "seed_rule": SEED_RULE,
        "limit": int(limit) if limit else None,
        "driver_limits": dict(limits or {}),
        "n_items": len(generated_ids), "item_ids": generated_ids,
        "n_requests": n_requests, "n_rejected": n_rejected,
        "n_errors": len(errors), "n_stamp_mismatch": n_stamp_mismatch,
        "n_empty_response": n_empty_response,
        "n_content_empty_with_reasoning": len(flagged_keys),
        "content_empty_with_reasoning_item_ids": flagged,
        "content_empty_with_reasoning_keys": flagged_keys,
        "window_clamp": clamp,
        "wall_clock_s": float(prior.get("wall_clock_s", 0.0)) + (time.time() - t0),
    }
    if errors:
        kinds: Dict[str, int] = {}
        for e in errors:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        meta["status"] = "incomplete"
        meta["n_generated"] = sum(1 for g in gen_keys if g in done)
        meta["errors"] = [{"item_id": e.item_id, "sample_idx": int(e.sample_idx), "kind": e.kind,
                           "message": e.message[:600], "record": e.record} for e in errors]
        over = [e.item_id for e in errors if e.kind == pe.ERR_PROMPT_OVER_WINDOW]
        hint = ("; re-run to fill the gaps rather than executing a partial set"
                if not over else
                f"; {len(over)} item(s) do NOT fit the serving window even clamped "
                f"({over[:5]}{'…' if len(over) > 5 else ''}) — a re-run will fail them "
                "again; this task cannot complete at this window")
        _atomic_write_json(meta_path, meta)
        raise EvalPlusSeamError(
            f"[evalplus-served] {task}: {len(errors)} of {len(todo)} items FAILED "
            f"(by kind: {kinds}; first: {[(e.item_id, e.message[:200]) for e in errors[:3]]}) "
            f"— {meta['n_generated']} generations are in {sidecar} and resume{hint}")

    samples: List[Dict[str, str]] = []
    modes: Dict[str, int] = {}
    finish: Dict[str, int] = {}
    for it in items:                       # problem-major, sample_idx 0..k-1 within
        rec = done[pe.generation_key(it.item_id, it.sample_idx)]
        ep = problems[it.item_id]["entry_point"]
        if open_variant:
            sol, mode = sanitize_open(rec["text"], ep)
            modes[mode] = modes.get(mode, 0) + 1
        else:
            sol = sanitize_solution(rec["text"], ep)
        samples.append({"task_id": it.item_id, "solution": sol})
        fr = str(rec.get("finish_reason"))
        finish[fr] = finish.get(fr, 0) + 1
    padded = sorted(expected - set(generated_ids))
    samples.extend({"task_id": t, "solution": ""} for t in padded for _ in range(n_samples))
    validate_task_ids([s["task_id"] for s in samples],
                      where=f"generate_samples_served({task}): samples", expected=expected,
                      samples_per_item=n_samples)
    samples_path = write_samples_jsonl(tdir / SAMPLES_FILENAME, samples)
    meta.update({
        # E55 #2: the GENERATION writer names the exact bytes it wrote. This is
        # where the durable binding starts: the conversion carries this digest
        # into the task's summary and done marker (and requires the sandbox's
        # execution.json to agree), so the number stays bound to its samples
        # even when the evalplus/<dataset>/ directory is later lost.
        SAMPLES_DIGEST_FIELD: hashlib.sha256(samples_path.read_bytes()).hexdigest(),
        "status": "complete", "n_generated": len(gen_keys),
        "padded_item_ids": padded, "finish_reasons": finish,
        "n_truncated": finish.get("length", 0),
        "open_extraction": modes if open_variant else None,
    })
    _atomic_write_json(meta_path, meta)
    print(f"[evalplus-served] {task}: {len(generated_ids)} problems x {n_samples} sample(s) generated "
          f"({len(padded)} padded), finish={finish}, requests={n_requests} "
          f"rejected={n_rejected} clamped={clamp['n_clamped']}, "
          f"{meta['wall_clock_s']:.1f}s -> {tdir}")
    return {"task": task, "n": len(generated_ids), "n_padded": len(padded),
            "samples_per_item": n_samples, "n_generations": len(gen_keys),
            "n_requests": n_requests, "n_rejected": n_rejected,
            "n_truncated": finish.get("length", 0),
            "n_content_empty_with_reasoning": len(flagged_keys),
            "wall_clock_s": meta["wall_clock_s"]}


def run_served(
    *,
    base_url: str,
    route: Any,
    system: Optional[str],
    thinking: str,
    concurrency: int,
    cell_dir: str | Path,
    datasets: Sequence[str] = C.EVALPLUS_DATASETS,
    limit: int = 0,
    api_key: Optional[str] = None,
    gen_cfg: Optional[Mapping[str, Any]] = None,
    transport: Any = None,
    max_model_len: Optional[int] = None,
    reasoning_parser: Optional[str] = None,
    limits: Optional[Mapping[str, int]] = None,
    sampling: Optional[C.SamplingRow] = None,
    samples_by_task: Optional[Mapping[str, int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """The served driver's entry point for the ``humaneval_plus`` /
    ``mbpp_plus`` atoms (``DISPATCH_MODULE_OF_CONTROL``): per dataset the
    stock prefilled instrument in the requested thinking mode and, under
    thinking-ON only, the ``:open`` variant too (P12). Tasks run one after
    another ON PURPOSE — each fan-out saturates the server at the resolved
    concurrency and the report needs a per-task wall clock. Generation
    only: execution is the sandbox orchestrator's, rows land after it.
    ``max_model_len`` (optional) is the launcher's resolved window, threaded
    to the engine as provenance for the once-only clamp (the server's
    stated window governs either way). ``reasoning_parser`` is the parser
    the standup COMPOSED (the driver reads it off ``endpoint.json``; None
    when none was) — it arms the engine's neither-channel refusal and is
    recorded in ``gen_meta.thinking``. ``limits`` is the driver's active
    limit set (R2; ``limit`` here is one of them), recorded in ``gen_meta``
    so the orchestrator's conversion ends the task PARTIAL rather than done.
    ``sampling`` / ``samples_by_task`` (U8 / PW-sampling) are the run's
    :class:`contracts.SamplingRow` (the engine draws every generation under
    it; None = greedy, the pre-U8 draw) and its ``{row task: k}`` map — every
    task takes ITS k, a task the map does not name refuses, and without the
    map every task is one sample (the pre-U8 instrument).
    Returns ``{row task: summary}``."""
    from sisteer.capeval.serving.proxy_engine import ProxyGenerateEngine

    on = C.enable_thinking(thinking)  # refuses anything but on/off
    lim = dict(limits or {})
    if limit:
        lim.setdefault("limit", int(limit))
    out: Dict[str, Dict[str, Any]] = {}
    variants = (False, True) if on else (False,)
    for dataset in datasets:
        for open_variant in variants:
            task = C.evalplus_task(dataset, open_variant=open_variant)
            if samples_by_task is not None and task not in samples_by_task:
                raise EvalPlusSeamError(
                    f"{task}: no samples-per-item entry in this run's map {sorted(samples_by_task)} "
                    "— refusing to generate at a default k (U8)")
            k = int(samples_by_task[task]) if samples_by_task is not None else 1
            engine = ProxyGenerateEngine(
                base_url, route=route, system=system, thinking=thinking,
                concurrency=concurrency, task=task, api_key=api_key, transport=transport,
                max_model_len=max_model_len, reasoning_parser=reasoning_parser,
                sampling=sampling)
            out[task] = generate_samples_served(
                engine=engine, dataset=dataset, open_variant=open_variant,
                cell_dir=cell_dir, gen_cfg=gen_cfg, limit=limit or None, limits=lim,
                samples_per_item=k)
    return out


# --------------------------------------------------------------------------- #
# execution results: validation BEFORE the volume, conversion into the store  #
# --------------------------------------------------------------------------- #


def validate_eval_results(
    results: Any,
    *,
    expected_task_ids: FrozenSet[str],
    where: str = "samples_eval_results.json",
    expected_hash: Optional[str] = None,
    samples_per_item: int = 1,
) -> None:
    """Refuse anything but the 0.3.1 ``samples_eval_results.json`` shape
    over EXACTLY the expected id set: top-level ``{date, hash, eval}``,
    exactly ``samples_per_item`` items per task (the task's k, read off its
    ``gen_meta.json``; 1 before U8), each item ``{task_id, base_status, plus_status}``
    (+ optional ``solution`` / ``*_fail_tests`` lists), statuses in
    ``pass``/``fail``/``timeout`` (a ``None`` plus_status is a base-only
    run we never make), the echoed task_id equal to its key. With
    ``expected_hash`` the file's dataset hash must equal it — the
    orchestrator passes the md5 of the archive it uploaded, so the sandbox
    provably evaluated against OUR dataset bytes."""
    if not isinstance(results, Mapping):
        raise EvalPlusSeamError(f"{where}: not a JSON object")
    extra_top = sorted(set(results) - RESULT_TOP_KEYS)
    missing_top = sorted(RESULT_TOP_KEYS - set(results))
    if extra_top or missing_top:
        raise EvalPlusSeamError(
            f"{where}: top-level keys != {sorted(RESULT_TOP_KEYS)} (extra {extra_top}, "
            f"missing {missing_top}) — evalplus output shape drifted")
    if expected_hash is not None and results.get("hash") != expected_hash:
        raise EvalPlusSeamError(
            f"{where}: dataset hash {results.get('hash')!r} != the uploaded archive's "
            f"{expected_hash!r} — the sandbox did not evaluate against our dataset bytes")
    ev = results["eval"]
    if not isinstance(ev, Mapping):
        raise EvalPlusSeamError(f"{where}: 'eval' is not an object")
    validate_task_ids(ev.keys(), where=where, expected=expected_task_ids)
    n_samples = int(samples_per_item)
    if n_samples < 1:
        raise EvalPlusSeamError(f"{where}: samples_per_item must be >= 1, got {samples_per_item!r}")
    for task_id, items in ev.items():
        if not isinstance(items, list) or len(items) != n_samples:
            raise EvalPlusSeamError(
                f"{where}: {task_id} carries {len(items) if isinstance(items, list) else items!r} "
                f"samples — exactly {n_samples} sample(s) per task (samples_per_item"
                + ("" if n_samples > 1 else ": one greedy sample") + ")")
        for item in items:
            if not isinstance(item, Mapping):
                raise EvalPlusSeamError(f"{where}: {task_id} item is not an object")
            missing = sorted(RESULT_ITEM_REQUIRED_KEYS - set(item))
            extra = sorted(set(item) - RESULT_ITEM_ALLOWED_KEYS)
            if missing or extra:
                raise EvalPlusSeamError(
                    f"{where}: {task_id} item keys drifted (missing {missing}, unexpected {extra})")
            if item["task_id"] != task_id:
                raise EvalPlusSeamError(
                    f"{where}: item under {task_id} echoes task_id {item['task_id']!r}")
            for k in RESULT_STATUS_KEYS:
                if item[k] not in RESULT_STATUSES:
                    raise EvalPlusSeamError(
                        f"{where}: {task_id}.{k} = {item[k]!r} not in {RESULT_STATUSES}")
            for k in ("base_fail_tests", "plus_fail_tests"):
                if k in item and not isinstance(item[k], list):
                    raise EvalPlusSeamError(f"{where}: {task_id}.{k} is not a list")
            if "solution" in item and not isinstance(item["solution"], str):
                raise EvalPlusSeamError(f"{where}: {task_id}.solution is not a string")


def _flag_rows(rows: List[Dict[str, Any]], meta: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The flagged-row fact travels into every row's ``extra_json``:
    True/False from gen_meta's item list; None when the meta predates the
    record (never a silent False). A per-sample ``acc@<i>`` row (U8 at k > 1,
    R16-A also at k = 1 under the sampling contract)
    reads its OWN generation's flag off gen_meta's per-generation keys
    (``content_empty_with_reasoning_keys``; None for a meta without them) and
    a per-item row is flagged when ANY of its samples was."""
    from sisteer.capeval.serving.proxy_engine import generation_key

    flagged_ids = meta.get("content_empty_with_reasoning_item_ids")
    flagged = set(flagged_ids) if isinstance(flagged_ids, list) else None
    flagged_keys = meta.get("content_empty_with_reasoning_keys")
    keys = set(flagged_keys) if isinstance(flagged_keys, list) else None
    for r in rows:
        extra = json.loads(r["extra_json"]) if r["extra_json"] else {}
        if C.SAMPLE_METRIC_SEP in r["metric"] and "sample_idx" in extra:
            extra["content_empty_with_reasoning"] = (
                (generation_key(r["item_id"], int(extra["sample_idx"])) in keys)
                if keys is not None else None)
        else:
            extra["content_empty_with_reasoning"] = (
                (r["item_id"] in flagged) if flagged is not None else None)
        r["extra_json"] = json.dumps(extra, sort_keys=True)
    return rows


#: The per-sample provenance fields the sampling contract records ONCE per
#: task (``gen_meta.json``'s ``sampling`` block) — the row every generation
#: of that task was drawn under. ``seed`` is per GENERATION and comes from
#: the ``generations.jsonl`` sidecar instead.
_SAMPLING_ROW_FIELDS = ("temperature", "top_p", "top_k")
#: The ``gen_meta.json`` key whose PRESENCE says a task was generated under the
#: sampling contract (``contracts.MANIFEST_KEY_SAMPLING``) — the discriminator
#: :func:`_records_sampling` reads — against the store manifest's own
#: ``sampling`` block, E54 #3 — and the one E50 #8's refusal names.
_SAMPLING_BLOCK_KEY = C.MANIFEST_KEY_SAMPLING


def _records_sampling(meta: Mapping[str, Any], manifest: Mapping[str, Any], *,
                      where: str) -> bool:
    """Was this task generated under the sampling contract (U8 /
    PW-sampling)? Its ``gen_meta.json`` carries the ``sampling`` block iff it
    was — :class:`proxy_engine.ProxyGenerateEngine` always has a row (greedy by
    default) and always records it. Only a task under the contract gets the
    per-sample ``acc@<i>`` row at k = 1 (R16-A); a pre-sampling task's k = 1
    rows stay exactly what was written, so the retained 618f9dd stores and
    their off-volume re-derivations do not move.

    E54 #3: the TASK-LOCAL block alone is not that discriminator. The store's
    own ``manifest.json`` — which both conversion paths already read, for
    ``row_constants`` — carries the run's ``sampling`` block too, and the two
    are written by ONE driver run from ONE :class:`contracts.SamplingRow`. So a
    parent that names the contract while the task-local block is missing is a
    store contradicting itself, not a pre-sampling task: before this it read as
    pre-sampling and, at k = 1, SILENTLY DROPPED the ``acc@0`` row — which also
    disarmed E50 #8's seedless refusal, since that only fires on per-sample
    rows. Three rules, and only the first is about age:

    * parent NAMES the contract, task-local block ABSENT — refuse, unless the
      store positively dates itself before the contract
      (:func:`contracts.store_era`), in which case today's reading stands;
    * both present and their drawn row DISAGREES on any of
      :data:`_SAMPLING_ROW_FIELDS` — refuse naming both, for any era. Two
      present statements contradicting each other is not an absence, so age
      cannot excuse it;
    * parent says NOTHING — the task-local block governs, as before. This is
      not the missing-key inference: nothing is being waived on the strength of
      an absence, the local block is a positive statement either way, and a
      store whose manifest cannot vouch for how its rows were drawn is already
      refused at resume by the driver's own ``_check_manifest_sampling``.

    ``source`` is never compared: it is provenance prose and
    :func:`contracts.sampling_identity` does not hash it either."""
    local = meta.get(_SAMPLING_BLOCK_KEY)
    parent = manifest.get(C.MANIFEST_KEY_SAMPLING)
    local_row = ({f: local.get(f) for f in _SAMPLING_ROW_FIELDS}
                 if isinstance(local, Mapping) else None)
    parent_row = ({f: parent.get(f) for f in _SAMPLING_ROW_FIELDS}
                  if isinstance(parent, Mapping) else None)
    if parent_row is None:
        return local_row is not None
    if local_row is None:
        era, why_era = C.store_era(manifest, contract=C.CONTRACT_SAMPLING)
        if era == C.STORE_ERA_HISTORICAL:
            return False
        raise EvalPlusSeamError(
            f"{where}: the store's manifest declares the sampling contract "
            f"({C.MANIFEST_KEY_SAMPLING} block: {parent_row}) but this task's "
            f"{GEN_META_FILENAME} carries no {_SAMPLING_BLOCK_KEY} block — the two are written "
            "by one driver run from one sampling row, so the task cannot predate a contract its "
            f"own store ran under. Nothing on that store says otherwise: {why_era}. Reading it as "
            "a pre-sampling task would silently drop the per-sample acc@<i> row at k = 1 (and "
            "with it the seed check that row exists for), so it refuses. Re-generate this task, "
            f"or repair its {GEN_META_FILENAME} (E54 #3)")
    if local_row != parent_row:
        moved = [f"{f}: {GEN_META_FILENAME} says {local_row[f]!r}, the manifest's "
                 f"{C.MANIFEST_KEY_SAMPLING} says {parent_row[f]!r}"
                 for f in _SAMPLING_ROW_FIELDS if local_row[f] != parent_row[f]]
        raise EvalPlusSeamError(
            f"{where}: this task's recorded sampling row disagrees with the store's — "
            + "; ".join(moved) + ". Both are written by ONE driver run from ONE sampling row, so "
            "a disagreement is a torn or hand-edited store, and the rows' recorded draw would be "
            "attributed to a draw the run did not make. Re-generate this task (E54 #3)")
    return True


def _join_sample_provenance(rows: List[Dict[str, Any]], meta: Mapping[str, Any],
                            generations: Mapping[str, Mapping[str, Any]], *,
                            records_sampling: bool) -> List[Dict[str, Any]]:
    """R16-A (E45 M-EvalPlus): every per-sample ``acc@<i>`` row says HOW its
    generation was drawn — ``sample_idx`` (already on the row), the ``seed``
    the request carried (joined per generation off the ``generations.jsonl``
    sidecar, keyed ``proxy_engine.generation_key``) and the ``temperature`` /
    ``top_p`` / ``top_k`` of gen_meta's ``sampling`` row. Before this the five
    sat only in the sidecars and never reached a row, so a row could not say
    which draw produced it — while ``docs/tinycap_interfaces.md`` §3's
    ``seeds`` key already promised the per-(item, sample) values ride in every
    row's ``extra_json``.

    A fact no sidecar records is ``None``, never a default: a store written
    before the sampling contract has no ``sampling`` block and cannot vouch
    for the three — the ``content_empty_with_reasoning`` precedent ("nobody
    said" and "0.0" are different facts). ``top_p`` / ``top_k`` are
    legitimately ``None`` in a recorded row too (a card that pins neither);
    the block's presence, not a field's value, is what says the store can
    vouch.

    E50 #8: the SEED is where that permissiveness stops. A store that
    :func:`_records_sampling` accepts declares it was generated under the
    sampling contract, and that contract's own producer
    (``proxy_engine.ProxyGenerateEngine``) writes one ``generations.jsonl``
    record per (item, sample) carrying the seed the request went out with —
    so under a ``sampling`` block a missing record, or a record with no seed,
    is not "nobody said", it is a store contradicting its own declaration
    (a torn or half-rewritten sidecar). Filing ``acc@<i>`` with ``seed: None``
    there publishes an unreproducible draw as a reproducible one, which is
    what the whole per-sample provenance exists to prevent, so it REFUSES
    naming the rows. Permissive only for a PRE-SAMPLING store (no ``sampling``
    block), whose sidecars predate the rule and whose k = 1 rows must not
    move."""
    from sisteer.capeval.serving.proxy_engine import generation_key

    sampling_row = meta.get(_SAMPLING_BLOCK_KEY)
    drawn = ({f: sampling_row.get(f) for f in _SAMPLING_ROW_FIELDS}
             if isinstance(sampling_row, Mapping) else dict.fromkeys(_SAMPLING_ROW_FIELDS))
    # E54 #3: the ONE decision, made by the caller against the store's manifest
    # as well as this task's gen_meta — never re-derived from gen_meta alone
    seedless: List[str] = []
    for r in rows:
        if C.SAMPLE_METRIC_SEP not in r["metric"]:
            continue
        extra = json.loads(r["extra_json"]) if r["extra_json"] else {}
        if "sample_idx" not in extra:
            continue
        idx = int(extra["sample_idx"])
        rec = generations.get(generation_key(r["item_id"], idx)) or {}
        extra["seed"] = rec.get("seed")
        if records_sampling and extra["seed"] is None:
            seedless.append(generation_key(r["item_id"], idx))
        extra.update(drawn)
        r["extra_json"] = json.dumps(extra, sort_keys=True)
    if seedless:
        raise EvalPlusSeamError(
            f"{meta.get('task')!r}: {len(seedless)} per-sample row(s) have no recorded "
            f"seed (first: {seedless[:3]}) — but this task's gen_meta declares the "
            f"sampling contract ({_SAMPLING_BLOCK_KEY} block: {drawn}), whose generator "
            f"writes one {GENERATIONS_FILENAME} record per (item, sample) carrying the "
            "seed its request went out with. A missing record is the store contradicting "
            "its own declaration, not an unrecorded fact, and filing acc@<i> with "
            "seed: null would publish an unreproducible draw as a reproducible one. "
            f"Re-generate this task, or repair {GENERATIONS_FILENAME}. (A PRE-SAMPLING "
            f"store — no {_SAMPLING_BLOCK_KEY} block — is unaffected and still files a "
            "null seed.)")
    return rows


def _wall_clock(tdir: Path, meta: Mapping[str, Any]) -> float:
    exec_meta = (_read_json(tdir / EXECUTION_FILENAME, what="execution")
                 if (tdir / EXECUTION_FILENAME).exists() else {})
    return float(meta.get("wall_clock_s", 0.0)) + float(exec_meta.get("wall_clock_s", 0.0))


def _summary_payload(rows: Sequence[Mapping[str, Any]], *, n: int, meta: Mapping[str, Any],
                     engine: str, wall: float,
                     samples_sha256: Optional[str] = None) -> Dict[str, Any]:
    """The counts-only ``summary.json`` payload of one served EvalPlus task
    from its rows (``SUMMARY_PAYLOAD_FIELDS``): ``acc`` = ``pass_at_1_plus``
    under :data:`PLUS_METRIC_DEF`, named in the payload (PW12). ONE recipe for
    the conversion and the off-volume re-derivation, so the two can never
    disagree on how a number is computed.

    E55 #2: ``samples_sha256`` is the DURABLE binding — the digest of the exact
    ``samples.jsonl`` these numbers were scored on, carried into the payload
    that becomes the ``controls_summary.json`` entry. Omitted (never null) when
    the store positively lacks the digest contract, so every payload already
    on disk stays byte-identical and the standing-digest discipline sees no
    change."""
    by_metric: Dict[str, float] = {}
    for r in rows:
        by_metric[r["metric"]] = by_metric.get(r["metric"], 0.0) + float(r["value"])
    plus = by_metric.get("pass@1_plus", 0.0) / n if n else None
    summary: Dict[str, Any] = {
        "acc": plus,
        "pass_at_1_base": (by_metric.get("pass@1_base", 0.0) / n if n else None),
        "pass_at_1_plus": plus,
        C.EVALPLUS_METRIC_DEF_FIELD: PLUS_METRIC_DEF,
        "n": n, "n_requests": int(meta.get("n_requests", 0)),
        "n_rejected": int(meta.get("n_rejected", 0)),
        "samples_per_item": int(meta.get("samples_per_item", 1)),
        "framework": C.EVALPLUS_TASK_VERSION, "engine": engine,
        "route": meta["route"]["virtual_model"], "thinking": meta["thinking"]["mode"],
        "wall_clock_s": wall,
    }
    if samples_sha256 is not None:
        summary[SAMPLES_DIGEST_FIELD] = samples_sha256
    bad = sorted(set(summary) - set(C.SUMMARY_PAYLOAD_FIELDS))
    assert not bad, f"summary payload keys outside SUMMARY_PAYLOAD_FIELDS: {bad}"
    return summary


#: ResultRow columns that vary per row; everything else must come from the
#: cell manifest's ``row_constants`` (the served driver's template).
_PER_ROW_COLUMNS = frozenset(("tier", "task", "task_version", "domain", "item_id",
                              "position_order", "metric", "value", "extra_json",
                              "created_at", "wall_clock_s"))

#: The rows whose ``extra_json`` names the pass@1_plus definition
#: (``pass@1_base`` carries none — its definition never changed).
_DEFINED_METRICS = frozenset(C.EVALPLUS_DEFINED_METRICS)   # == {GATING_METRIC, GATING_ROW_METRIC} (PW14: one tuple, the stats readers scan the same)


def _check_existing_rows_definition(store: CellStore, *, task: str, cell_dir: Path) -> None:
    """PW13 (E21 M2): the rows ``rows.jsonl`` ALREADY holds for ``task`` must
    have been written under THIS converter's definition — ``task_version`` =
    :func:`contracts.evalplus_row_task_version` (:data:`PLUS_METRIC_DEF`) and,
    on every ``pass@1_plus`` / ``acc`` row, ``extra_json.metric_def`` =
    :data:`PLUS_METRIC_DEF` — before any of them is skipped as complete.
    The resume dedupes on ``(task, item_id, metric)``, which does not see
    the definition: a store where an OLD converter appended plus-only rows
    and died before the summary and the done marker would otherwise have
    its rows skipped and a fresh base-AND-plus summary written over them —
    ``rows.jsonl`` saying one thing, ``summary.json`` another, both labelled
    current. Legacy, mixed and contradictory rows (a definition named in
    ``extra_json`` that disagrees with the version) all REFUSE, naming the
    counts and the remedy: the store stands as written, and its numbers are
    brought onto the current definition OFF-VOLUME
    (``scripts/tinycap_rederive_evalplus.py``) into a new identity — never
    by re-converting into this store, whatever markers it holds."""
    expected_version = C.evalplus_row_task_version(PLUS_METRIC_DEF)
    existing = store.rows_of_task(task)
    if not existing:
        return
    bad: List[str] = []
    for r in existing:
        version = r.get("task_version")
        try:
            version_def = C.evalplus_row_metric_def(version)
        except ValueError:
            version_def = None
        named = None
        if r.get("metric") in _DEFINED_METRICS:
            try:
                extra = json.loads(r["extra_json"]) if r.get("extra_json") else {}
            except (TypeError, ValueError):
                extra = {}
            named = extra.get(C.EVALPLUS_METRIC_DEF_FIELD) if isinstance(extra, Mapping) else None
        if version != expected_version or (named is not None and named != PLUS_METRIC_DEF):
            bad.append(f"task_version={version!r} ({version_def or 'unreadable'})"
                       + (f", {C.EVALPLUS_METRIC_DEF_FIELD}={named!r}" if named is not None else ""))
    if not bad:
        return
    kinds = sorted(set(bad))
    raise EvalPlusSeamError(
        f"{cell_dir}: {len(bad)} of {len(existing)} existing {task} rows in {store.rows_path} were "
        f"written under another pass@1_plus definition ({'; '.join(kinds)}); this converter writes "
        f"task_version={expected_version!r} ({PLUS_METRIC_DEF}). Refusing to resume: skipping them by "
        "(task, item_id, metric) would leave rows.jsonl under the old definition and summary.json under "
        "the new one, both labelled current. The store stands as written — bring its numbers onto the "
        "current definition OFF-VOLUME, into a new identity (scripts/tinycap_rederive_evalplus.py "
        f"{cell_dir} --out <dir>), never by re-converting into this store (PW13 / E21 M2)")


def convert_eval_results_to_store(
    cell_dir: str | Path,
    *,
    dataset: str,
    open_variant: bool = False,
    results: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Append the served EvalPlus rows for one (dataset, variant) to the
    cell's ``rows.jsonl`` from its validated ``eval_results.json`` + the
    manifest's ``row_constants`` (the exec wave's discipline), idempotent
    on ``(task, item_id, metric)`` — AFTER the existing rows of the task
    are checked to be this converter's own definition
    (:func:`_check_existing_rows_definition`, PW13 / E21 M2: rows under
    another ``pass@1_plus`` definition refuse, whatever markers the store
    holds). Rows are emitted for the GENERATED items only (a smoke's padded
    ids never become numbers). Writes the counts-only ``summary.json`` and
    — on a complete, unpadded set — the ``done/<task>.json`` marker.
    Returns the summary payload."""
    from sisteer.capeval.store import CellStore, ResultRow, StoreError

    cell_dir = Path(cell_dir)
    task = C.evalplus_task(dataset, open_variant=open_variant)
    tdir = served_task_dir(cell_dir, dataset, open_variant)
    manifest = _read_json(cell_dir / "manifest.json", what="cell manifest")
    rc = manifest.get("row_constants")
    if not isinstance(rc, Mapping) or not rc:
        raise EvalPlusSeamError(f"{cell_dir}/manifest.json has no row_constants — cannot build rows")
    if rc.get("engine") != "vllm":
        raise EvalPlusSeamError(
            f"{cell_dir}: row_constants.engine = {rc.get('engine')!r}; the served EvalPlus "
            "rows are vllm rows only (an HF store never converts through this path)")
    meta = _read_json(tdir / GEN_META_FILENAME, what="gen_meta")
    if meta.get("task") != task or meta.get("status") != "complete":
        raise EvalPlusSeamError(
            f"{tdir}/{GEN_META_FILENAME}: task {meta.get('task')!r} status "
            f"{meta.get('status')!r} — expected a complete {task!r} generation")
    if results is None:
        results = _read_json(tdir / EVAL_RESULTS_FILENAME, what="eval results")
    expected = expected_task_ids(dataset)
    k = int(meta.get("samples_per_item", 1))          # the task's k (1 before U8)
    validate_eval_results(results, expected_task_ids=expected, where=str(tdir / EVAL_RESULTS_FILENAME),
                          samples_per_item=k)
    records_sampling = _records_sampling(meta, manifest, where=str(tdir))
    # E55 #2: the number about to be written is bound to the sample bytes it
    # was scored on BEFORE it is written — the generation's own digest, the
    # sandbox's, and the bytes on disk must agree — and the agreed digest goes
    # into the summary and the done marker, where it outlives the task dir.
    samples_digest, _binding = samples_binding_of_task(
        tdir, manifest, task=task, declared=meta.get(SAMPLES_DIGEST_FIELD))
    rows = eval_results_to_rows(results, expected, task=task,
                                domain=C.DOMAIN_EVALPLUS_SERVED,
                                task_version=C.EVALPLUS_TASK_VERSION, samples_per_item=k,
                                per_sample_rows=records_sampling)
    generated = set(meta["item_ids"])
    padded = set(meta.get("padded_item_ids") or [])
    if generated & padded or (generated | padded) != expected:
        raise EvalPlusSeamError(f"{tdir}: gen_meta item_ids/padded_item_ids do not partition the pinned set")
    rows = _join_sample_provenance(
        _flag_rows([r for r in rows if r["item_id"] in generated], meta), meta,
        read_generations(tdir / GENERATIONS_FILENAME), records_sampling=records_sampling)

    const_cols = [f.name for f in fields(ResultRow) if f.name not in _PER_ROW_COLUMNS]
    missing_rc = [c for c in const_cols if c not in rc]
    if missing_rc:
        raise EvalPlusSeamError(f"{cell_dir}: row_constants lack {missing_rc}")
    constants = {c: rc[c] for c in const_cols}
    store = CellStore(cell_dir.parent.parent, cell_dir.name, row_constants=constants)
    if store.cell_dir.resolve() != cell_dir.resolve():
        raise EvalPlusSeamError(
            f"{cell_dir} is not a <sweep>/cells/<key> cell directory (store would write to {store.cell_dir})")
    # the existing rows' definition is checked BEFORE any is skipped as complete (PW13 / E21 M2)
    _check_existing_rows_definition(store, task=task, cell_dir=cell_dir)
    have = store.completed_keys()
    now = time.time()
    n_new = 0
    for r in rows:
        if (r["task"], r["item_id"], r["metric"]) in have:
            continue
        try:
            store.append_row(ResultRow(
                **constants, tier=r["tier"], task=r["task"], task_version=r["task_version"],
                domain=r["domain"], item_id=r["item_id"], position_order=None,
                metric=r["metric"], value=float(r["value"]), extra_json=r["extra_json"],
                created_at=now, wall_clock_s=None))
        except StoreError as e:
            raise EvalPlusSeamError(f"{cell_dir}: {e}") from e
        n_new += 1

    n = len(generated)
    summary = _summary_payload(rows, n=n, meta=meta, engine=rc["engine"],
                               wall=_wall_clock(tdir, meta), samples_sha256=samples_digest)
    wall = summary["wall_clock_s"]
    _atomic_write_json(tdir / SUMMARY_FILENAME, summary)
    # the ONE completion decision (tinycap.completion): done iff the driver
    # ran unlimited AND the set is the full pinned one; a padded smoke or a
    # limited driver run ends PARTIAL, naming the limits (R2)
    driver_limits = dict(meta.get("driver_limits") or {})
    if meta.get("limit"):
        driver_limits.setdefault("limit", int(meta["limit"]))
    fin = completion.finish_task(
        cell_dir, task=task, n_items=n, n_expected_full=len(expected), limits=driver_limits,
        n_rows=len(rows), n_requests=summary["n_requests"], n_rejected=summary["n_rejected"],
        wall_clock_s=wall, summary=summary)
    print(f"[evalplus-served] {task}: {n_new} new rows ({len(rows)} total) -> {store.rows_path}; "
          f"pass@1_base={summary['pass_at_1_base']} pass@1_plus={summary['pass_at_1_plus']} n={n}"
          + ("" if fin["complete"] else
             f" (PARTIAL: {len(padded)} ids padded, limits {driver_limits} — no done marker)"))
    return summary


# --------------------------------------------------------------------------- #
# PW12 (E19): off-volume re-derivation under the current pass@1_plus definition #
# --------------------------------------------------------------------------- #

REDERIVATION_FILENAME = "rederivation.json"


#: E54 #1: the sha256 of the exact ``samples.jsonl`` bytes a sandbox evaluated,
#: recorded in ``execution.json`` (E50 #4). Named here as well as in
#: ``scripts/modal/tinycap_sandbox.py`` and ``eval.cell_controls`` because the
#: sandbox's orchestrator image cannot import either of the other two; the
#: three are pinned equal by test.
SAMPLES_DIGEST_FIELD = "samples_sha256"
_SAMPLES_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def execution_record_of_task(tdir: str | Path) -> Tuple[Optional[Mapping[str, Any]], str]:
    """``(record, why)`` — the task's ``execution.json`` parsed, or ``None``
    with the reason. Separated out because that record is a SECOND PRODUCER's
    statement about the same store (E55 #2c) and is dated by its own marker,
    never by the generation's."""
    path = Path(tdir) / EXECUTION_FILENAME
    if not path.is_file():
        return None, f"there is no {EXECUTION_FILENAME} in {tdir}"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, f"{path} did not parse ({type(e).__name__})"
    if not isinstance(doc, Mapping):
        return None, f"{path} is a {type(doc).__name__}, not an execution record"
    return doc, f"{path}"


def _stated_digest(doc: Optional[Mapping[str, Any]], where: str
                   ) -> Tuple[Optional[str], Optional[str]]:
    """``(digest, gap)`` — the sha256 a record states under
    :data:`SAMPLES_DIGEST_FIELD`, or the reason it states none."""
    if doc is None:
        return None, None
    value = doc.get(SAMPLES_DIGEST_FIELD)
    if isinstance(value, str) and _SAMPLES_SHA256_RE.fullmatch(value):
        return value, None
    if SAMPLES_DIGEST_FIELD in doc:
        return None, f"{where} records {SAMPLES_DIGEST_FIELD}={value!r}, which is not a sha256"
    return None, f"{where} records no {SAMPLES_DIGEST_FIELD}"


def samples_binding_of_task(tdir: str | Path, manifest: Mapping[str, Any], *, task: str,
                            declared: Optional[str] = None) -> Tuple[Optional[str], str]:
    """E54 #1, widened by E55 #2: ``(samples_sha256, binding)`` for one task
    dir — WHICH sample bytes this task's numbers were produced from, from up to
    THREE witnesses that must all agree, judged against what each of the
    store's two producers positively says about itself.

    The witnesses: ``declared`` (the digest the task's own record already
    carries — the ``gen_meta.json`` the GENERATION writer stamped, or the
    summary payload it was carried into), the ``execution.json`` the SANDBOX
    wrote, and the ``samples.jsonl`` bytes on disk. Any two that disagree
    REFUSE on any era: two present statements contradicting each other is not
    an absence, so age cannot excuse it.

    ``binding`` is ``contracts.SAMPLES_BINDING_BOUND`` with the agreed digest,
    or ``contracts.SAMPLES_BINDING_HISTORICAL_UNBOUND`` with ``None`` when NO
    witness names the bytes and the store positively lacks the digest
    contract. Otherwise it RAISES.

    TWO PRODUCERS, judged separately (E55 #2c). The exemption for a missing
    digest comes only from the GENERATION producer — the cell manifest's tree,
    which is the store's own age. The EXECUTION producer, read off the
    execution record's own marker (:func:`contracts.store_era` reads the same
    two keys there), can only REVOKE that exemption: a sandbox that positively
    says it wrote under the digest contract and recorded no digest is damage,
    whatever the generation's age — an old generation re-executed by a CURRENT
    sandbox must carry the digest. It can never GRANT one, so an execution
    record that names no writer — every record on the volume today — changes
    nothing. Using the generation's age to excuse the sandbox's silence is the
    conflation the finding names; using the sandbox's silence to excuse the
    generation is the missing-key inference this round removes."""
    tdir = Path(tdir)
    era, why_era = C.store_era(manifest, contract=C.CONTRACT_SAMPLES_DIGEST)
    samples = tdir / SAMPLES_FILENAME
    execution, why_exec = execution_record_of_task(tdir)
    recorded, exec_gap = _stated_digest(execution, why_exec)
    if execution is None:
        exec_gap = why_exec
    if declared is not None and not (isinstance(declared, str)
                                     and _SAMPLES_SHA256_RE.fullmatch(declared)):
        raise EvalPlusSeamError(
            f"{task}: this task's own record states {SAMPLES_DIGEST_FIELD}={declared!r}, which is "
            "not a sha256 — the sample bytes a number was scored on are named by their full "
            "digest or not at all (E55 #2)")
    if declared is not None and recorded is not None and declared != recorded:
        raise EvalPlusSeamError(
            f"{task}: the generation recorded samples sha256 {declared} but {tdir / EXECUTION_FILENAME} "
            f"says the sandbox scored sha256 {recorded} — the sandbox did not evaluate the bytes "
            "this task's own record names, so its results belong to a different generation. "
            f"Re-run the sandbox stage for that task (remove its {EVAL_RESULTS_FILENAME} and "
            f"{EXECUTION_FILENAME}) and re-convert (E55 #2)")
    claim = declared if declared is not None else recorded
    if claim is None:
        gap = (f"there is no {SAMPLES_FILENAME} in {tdir}" if not samples.is_file()
               else exec_gap or f"nothing in {tdir} names the sample bytes")
        # the EXECUTION producer can revoke the generation's exemption
        if execution is not None:
            exec_era, why_exec_era = C.store_era(execution,
                                                 contract=C.CONTRACT_SAMPLES_DIGEST)
            if exec_era == C.STORE_ERA_CURRENT:
                raise EvalPlusSeamError(
                    f"{task}: {gap} — and the execution record itself says it was written under "
                    f"the samples-digest contract ({why_exec_era}), so the digest it did not "
                    "record is damage rather than age. The store's own age cannot excuse the "
                    "sandbox's silence: an old generation re-executed by a current sandbox must "
                    f"carry the digest. Re-run the sandbox stage for that task (remove its "
                    f"{EVAL_RESULTS_FILENAME} and {EXECUTION_FILENAME}) (E55 #2c)")
        if era == C.STORE_ERA_HISTORICAL:
            return None, C.SAMPLES_BINDING_HISTORICAL_UNBOUND
        raise EvalPlusSeamError(
            f"{task}: {gap} — and nothing on that store says its producer tree lacks the "
            f"samples digest: {why_era}. A missing sidecar is not evidence (a torn write, a "
            "partial copy and a deleted file all look exactly like a store written without the "
            "digest), so these results are not used as they stand: nothing would bind the number "
            "to the bytes it was scored on. Re-run the sandbox stage for that task, or, if this "
            "store really is one this campaign wrote from a tree that did not contain the digest, "
            "name its tree in contracts.KNOWN_PRODUCER_TREES with the contracts it does not "
            "contain (E54 #1 / E55 #1)")
    if samples.is_file():
        current = hashlib.sha256(samples.read_bytes()).hexdigest()
        if current != claim:
            raise EvalPlusSeamError(
                f"{task}: this task's numbers were produced from samples sha256 {claim}, but "
                f"{SAMPLES_FILENAME} beside them now hashes to sha256 {current} — "
                "the samples were regenerated after those results, so using them would "
                "attribute one generation's score to another's samples (E50 #4)")
    return claim, C.SAMPLES_BINDING_BOUND


def rederive_evalplus(cell_dir: str | Path, *, dataset: str, open_variant: bool = False
                      ) -> Dict[str, Any]:
    """PW12 (E19, NEW): recompute one served EvalPlus task's rows and summary
    from the store's RETAINED ``eval_results.json`` + ``gen_meta.json`` under
    the CURRENT ``pass@1_plus`` definition (:data:`PLUS_METRIC_DEF`, base AND
    plus), touching NOTHING in the store — a pure read. The retained 618f9dd
    stores were converted under the earlier plus-only reading (their summary
    names no ``metric_def``); this is how their numbers are brought onto the
    current definition without re-converting the stores on the volume (a
    re-conversion would regenerate rows under the same ``harness_config_hash``
    — PW10-8). Returns ``{task, cell_dir, cell_key, harness_config_hash,
    producer_tree_sha, eval_results_sha256, samples_sha256, samples_binding,
    rows, summary, retained_summary,
    retained_metric_def, flipped_items, delta_pass_at_1_plus}``: ``rows`` /
    ``summary`` are exactly what :func:`convert_eval_results_to_store` would
    write today (the same ``eval_results_to_rows`` / ``_summary_payload``),
    ``flipped_items`` the ids whose ``pass@1_plus`` differs between the two
    definitions (base-fail / plus-pass), and ``delta_pass_at_1_plus`` the
    re-derived minus the retained headline (None without a retained summary)."""
    cell_dir = Path(cell_dir)
    task = C.evalplus_task(dataset, open_variant=open_variant)
    tdir = served_task_dir(cell_dir, dataset, open_variant)
    manifest = _read_json(cell_dir / "manifest.json", what="cell manifest")
    rc = manifest.get("row_constants")
    if not isinstance(rc, Mapping) or not rc:
        raise EvalPlusSeamError(f"{cell_dir}/manifest.json has no row_constants — cannot build rows")
    meta = _read_json(tdir / GEN_META_FILENAME, what="gen_meta")
    if meta.get("task") != task or meta.get("status") != "complete":
        raise EvalPlusSeamError(
            f"{tdir}/{GEN_META_FILENAME}: task {meta.get('task')!r} status "
            f"{meta.get('status')!r} — expected a complete {task!r} generation")
    results_path = tdir / EVAL_RESULTS_FILENAME
    results = _read_json(results_path, what="eval results")
    # E54 #1: the results are bound to the sample bytes they were produced from
    # BEFORE anything is re-derived from them — a store that cannot name them
    # and cannot date itself before the digest refuses, named
    samples_digest, samples_binding = samples_binding_of_task(
        tdir, manifest, task=task, declared=meta.get(SAMPLES_DIGEST_FIELD))
    expected = expected_task_ids(dataset)
    k = int(meta.get("samples_per_item", 1))
    validate_eval_results(results, expected_task_ids=expected, where=str(results_path),
                          samples_per_item=k)
    records_sampling = _records_sampling(meta, manifest, where=str(tdir))
    rows = eval_results_to_rows(results, expected, task=task, domain=C.DOMAIN_EVALPLUS_SERVED,
                                task_version=C.EVALPLUS_TASK_VERSION, samples_per_item=k,
                                per_sample_rows=records_sampling)
    generated = set(meta["item_ids"])
    rows = _join_sample_provenance(
        _flag_rows([r for r in rows if r["item_id"] in generated], meta), meta,
        read_generations(tdir / GENERATIONS_FILENAME), records_sampling=records_sampling)
    summary = _summary_payload(rows, n=len(generated), meta=meta, engine=rc.get("engine"),
                               wall=_wall_clock(tdir, meta), samples_sha256=samples_digest)
    flipped = sorted(
        tid for tid, items in results["eval"].items()
        if tid in generated and any(it["base_status"] != "pass" and it["plus_status"] == "pass"
                                    for it in items))
    retained = completion.read_task_summary(cell_dir, task)
    retained_def = ((retained or {}).get(C.EVALPLUS_METRIC_DEF_FIELD)
                    or C.EVALPLUS_PLUS_DEF_PLUS_ONLY) if retained is not None else None
    delta = (None if not retained or retained.get("pass_at_1_plus") is None
             else summary["pass_at_1_plus"] - float(retained["pass_at_1_plus"]))
    producer = manifest.get(C.MANIFEST_KEY_PRODUCER)
    return {
        "task": task, "cell_dir": str(cell_dir), "cell_key": cell_dir.name,
        "harness_config_hash": manifest.get("harness_config_hash"),
        "producer_tree_sha": (producer.get("tree_sha") if isinstance(producer, Mapping) else None),
        "eval_results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        # E54 #1: and WHICH SAMPLE BYTES those results were produced from —
        # verified above, retained here, so the sidecar binds its numbers to a
        # generation the way the store's own execution.json does
        SAMPLES_DIGEST_FIELD: samples_digest,
        "samples_binding": samples_binding,
        "rows": rows, "summary": summary,
        "retained_summary": retained, "retained_metric_def": retained_def,
        "flipped_items": flipped, "delta_pass_at_1_plus": delta,
    }


def write_rederivation(out_dir: str | Path, derived: Mapping[str, Any]) -> Path:
    """Write one :func:`rederive_evalplus` result as a SIDECAR under
    ``out_dir/<cell_key>/`` — ``rows.jsonl`` (appended per task, deduped on
    ``(task, item_id, metric)``), ``summary/<task marker>.json`` (the
    re-derived payload, ``metric_def`` named) and ``rederivation.json`` (the
    provenance: source store, producer sha, identity hash, the eval results'
    sha256, the retained vs re-derived headline, the flipped items) — and
    REFUSES an ``out_dir`` inside the source store or its out_root, so the
    volume's stores are never written. Never a ``done/`` marker and never a
    ``controls_summary.json``: the sidecar is a derived reading for the
    results operator, not a store the merge discipline would source."""
    out_dir = Path(out_dir).resolve()
    src = Path(derived["cell_dir"]).resolve()
    if out_dir == src or out_dir.is_relative_to(src) or out_dir.is_relative_to(src.parent.parent):
        raise EvalPlusSeamError(
            f"--out {out_dir} lies inside the source store {src} or its out_root — the "
            "re-derivation is a sidecar OFF the volume, never a write into a store (PW12)")
    cell_out = out_dir / derived["cell_key"]
    cell_out.mkdir(parents=True, exist_ok=True)
    rows_path = cell_out / "rows.jsonl"
    seen = set()
    if rows_path.is_file():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r0 = json.loads(line)
                seen.add((r0.get("task"), r0.get("item_id"), r0.get("metric")))
    with rows_path.open("a", encoding="utf-8") as f:
        for r in derived["rows"]:
            if (r["task"], r["item_id"], r["metric"]) in seen:
                continue
            f.write(json.dumps(r, sort_keys=True) + "\n")
    _atomic_write_json(cell_out / C.SUMMARY_DIR / C.marker_filename(derived["task"]), derived["summary"])
    rec_path = cell_out / REDERIVATION_FILENAME
    rec = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.is_file() else {"tasks": {}}
    rec["tasks"][derived["task"]] = {k: v for k, v in derived.items() if k != "rows"}
    rec["source_cell_dir"] = derived["cell_dir"]
    rec["metric_def"] = PLUS_METRIC_DEF
    rec["note"] = ("re-derived from the store's retained eval_results.json under the current "
                   "pass@1_plus definition (base AND plus); the store itself is untouched and its "
                   f"canonical summary loads as {C.EVALPLUS_PLUS_DEF_PLUS_ONLY} (PW12)")
    _atomic_write_json(rec_path, rec)
    return cell_out

