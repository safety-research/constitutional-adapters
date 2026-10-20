#!/usr/bin/env python
"""The thinking-ON PROMPT BUDGET of the paired-200 GSM8K instrument, on CPU (plan S4.6).

Renders the 200 prompts exactly as the served task does
(``inspect_served.gsm8k_paired200_messages``), tokenizes each under the declared
render of the PINNED tokenizer — ``apply_chat_template(messages,
add_generation_prompt=True, **contracts.chat_template_kwargs(arm))``, the render
``object_identity.header_ids_under_arm`` uses, loaded through
``sisteer.models.load_tokenizer`` at the registry row's revision — and writes
``oos_prompt_budget.json`` (``contracts.OOS_PROMPT_BUDGET_FIELDS``): nearest-rank
statistics, the cap the gate adds, every item's length and the
``chat_template_hash`` of the template it measured (``sisteer.models.chat_template_hash``,
the digest ``endpoint.json`` carries). A NECESSARY pre-check the preflight and the
dry run read; the served render's fit is proven post-standup (``endpoint.json``
digest equality, the live tokenize pass).

Usage:
  PYTHONPATH=src python scripts/tinycap_oos_prompt_budget.py --model qwen35_4b \\
      --arm on --out oos_prompt_budget.json [--offline] [--tree-sha <40hex>]

MATH-500 generalisation:
  PYTHONPATH=src python scripts/tinycap_oos_prompt_budget.py --task math500 --arm on \\
      --config configs/sdfbase/math500/sdfbase_full22_<key>.math500.yaml --model-key <key> \\
      [--system-prompt-cell prompted__safety_refusal] [--geometry-eval capability:gen_long_math] \\
      --out <wave root>/budgets/<key>/<cell class>/oos_prompt_budget.json
renders the 500 MATH-500 problems through ``inspect_evals.math``'s own solver chain (``math500_messages``:
the same construction as ``inspect_served.gsm8k_paired200_messages``), the cell's system turn prepended for
the prompted cell, tokenizes them under the ON render of the tokenizer pinned by the CONFIG's ``model`` /
``model_revision`` (``--model-key`` names the geometry key: the launcher's ``contracts.MODELS`` lacks glm /
nemotron), and writes the artifact with EXACTLY the contract's keys (``cap`` = ``contracts.MATH500_MAX_TOKENS_DEFAULT``,
16,384 — one pass at the standing truncation rule's ceiling —
``cap_task`` ``inspect/math``, ``messages_source`` naming the MATH renderer) plus a SIDECAR ``<out>.fit.json``
with the GEOMETRY fit (``contracts.prompt_budget_fit`` against the named geometry entry's window: ``task``,
``cap``, ``window``, ``fits``, ``margin``, the system-prompt cell and its sha16). Exit 2 on ``fits: false``.
The artifact itself stays contract-exact so both consumers (the driver's ``--oos-prompt-budget`` reader and
``check_serve_attestation``'s template compare) read it unchanged.

GSM8K paired-200, the GENERAL shape:
  PYTHONPATH=src python scripts/tinycap_oos_prompt_budget.py --task gsm8k_paired200 --arm on|off \\
      --config <battery config> --model-key <key> [--system-prompt-cell prompted__safety_refusal] \\
      --cap <8192|16384|512> [--geometry-eval capability:gen_long_gsm8k] --out <...>/oos_prompt_budget.json
the same construction as the MATH-500 shape over the served task's OWN 200 renders
(``inspect_served.gsm8k_paired200_messages``: the stock task's 10-shot system turn + the templated user turn), the
prompted cell's SI folded into the leading system turn the way the provider collapses it
(``inspect_served.collapse_system_messages_dicts`` — ONE system turn, the render the prompted route serves), both
arms (the thinking-OFF overlap leg budgets at 512), the cap EXPLICIT (``--cap``; default = the arm's table / tuple
cap as before). ``--geometry-eval`` is OPTIONAL here: WITHOUT it the artifact carries the lengths and the cap and NO
sidecar (G-LENGTHS — the measurement a geometry entry is registered FROM; nothing to fit against yet); WITH it the
``.fit.json`` sidecar is written against the named entry's window and a non-fit exits 2 (G-FIT, a second, separable
pass against the registered entry at the pinned tree). Selected by ``--config`` + ``--model-key``; the default
invocation (``--model <slug>``, no config) is byte-identical to before.

IFEval:
  PYTHONPATH=src python scripts/tinycap_oos_prompt_budget.py --task ifeval --arm off --model-key <key> \\
      [--config <ifeval overlay>] [--system-prompt-cell prompted__safety_refusal] --cap <2048|4096|8192> \\
      [--geometry-eval capability:ifeval_<cap>] --out <...>/oos_prompt_budget.json
renders the 541 IFEval prompts through the stock task's own solver chain (``sisteer.capeval.tiers.ifeval_messages``:
the vendor's ``prompt`` as ONE user turn, the prompted cell's SI folded into ONE leading system turn), tokenizes them
under the thinking-OFF render of the tokenizer pinned by ``ifeval_contracts.IFEVAL_MODEL_PINS[--model-key]`` (all four:
gemma4e4b, qwen35_4b, nemotron3nano4b, llama8b; a ``--config`` whose model / model_revision disagree with the pin
refuses), ``cap_task`` ``inspect/ifeval``, the cap EXPLICIT (default 2,048, the OFF table's); the artifact contract-exact;
the ``.fit.json`` sidecar ONLY with ``--geometry-eval`` (exit 2 on a non-fit). ``--arm on`` refuses (IFEval is OFF).
A request that would route an IFEval budget to the GSM8K renderer (``main_gsm8k_general`` with ``--task ifeval``)
REFUSES BY NAME — the budget is measured on THIS task's prompts, never another's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap.completion import atomic_write_json

#: Which task's cap the artifact carries: the standard GSM8K (the ONE control
#: the authorised experiment names — ``contracts.AUTHORISED_OOS_THINKING``).
CAP_TASK_NAME = C.GSM8K_PAIRED_TASK_NAME
#: The MATH-500 task (math500-thinking-on-full22): ``--task math500`` budgets the 500 MATH-500 problems at the
#: experiment's ONE-PASS cap — ``contracts.MATH500_MAX_TOKENS_DEFAULT`` (16,384; the retry pass retired), read
#: from the contracts, so the budget, the wrapper and the driver cannot disagree; the sidecar records which one
#: the artifact used.
TASK_GSM8K = C.GSM8K_PAIRED_TASK_NAME          # "gsm8k_paired200"
TASK_MATH500 = C.CONTROL_MATH500               # "math500"
#: IFEval (ifeval-canonical-battery): the control atom / task name, from the contracts like the other two
#: (``ifeval_contracts`` below carries the matrix's ladder / pins vocabulary, resolved through the contracts)
from sisteer.capeval.tinycap import ifeval_contracts as IC  # noqa: E402

TASK_IFEVAL = C.IFEVAL_TASK_NAME               # "ifeval"
TASKS = (TASK_GSM8K, TASK_MATH500, TASK_IFEVAL)
MATH500_INSPECT_TASK = "math"
MATH500_THINKING_CAP = C.MATH500_MAX_TOKENS_DEFAULT
MATH500_MESSAGES_SOURCE = "scripts/tinycap_oos_prompt_budget.py:math500_messages (inspect_evals.math's solver chain, generate a no-op)"
#: the default geometry entry the MATH-500 fit is checked against: MATH-500's OWN entry, resolved through the same
#: class map the launcher and the standup read (``capability:gen_long_math``, L 17,664 = prompt max 1,041 + 16,384
#: rounded), never a typed key
MATH500_GEOMETRY_EVAL = C.geometry_subentry(C.GEOMETRY_CLASS_OF_CONTROL[C.CONTROL_MATH500], prompted=False, thinking=True)
#: the GSM8K paired-200 GENERAL shape (lane B): the prompted class's messages_source names the fold (the bare class
#: renders exactly what the default branch renders and keeps the contract's source string)
GSM8K_INSPECT_TASK = C.GSM8K_PAIRED_TASK_NAME
GSM8K_PROMPTED_MESSAGES_SOURCE = ("sisteer.capeval.tiers.inspect_served.gsm8k_paired200_messages + the cell's system "
                                  "turn folded into the leading system message (collapse_system_messages_dicts)")


def percentile(values: Sequence[int], q: float) -> int:
    """``contracts.nearest_rank_percentile`` — the geometry measurement's
    convention, so a prompt p99 here is comparable to a ledger p99 there."""
    return C.nearest_rank_percentile(values, q)


def render_kwargs(arm: str) -> Dict[str, bool]:
    """The ``chat_template_kwargs`` fragment for ``arm`` — ``contracts``' own
    mapping, never a second copy; EMPTY under the no-switch arm
    (:data:`contracts.THINKING_NOT_AVAILABLE`: no switch is sent, P7)."""
    return dict(C.chat_template_kwargs(arm).get(C.CHAT_TEMPLATE_KWARGS_KEY) or {})


def content_in_format(messages: Sequence[Mapping[str, object]], content_format: str) -> List[Dict[str, object]]:
    """The messages with each string ``content`` in ``content_format`` (``contracts.PROMPT_BUDGET_CONTENT_FORMAT_*``):
    the OpenAI text-parts form ``[{"type": "text", "text": …}]`` — what vLLM's chat layer hands a multimodal template
    — or the string as given; a content that is already a list is left alone."""
    if content_format == C.PROMPT_BUDGET_CONTENT_FORMAT_STRING:
        return [dict(m) for m in messages]
    if content_format != C.PROMPT_BUDGET_CONTENT_FORMAT_OPENAI:
        raise ValueError(f"content_format must be {C.PROMPT_BUDGET_CONTENT_FORMAT_STRING!r} or "
                         f"{C.PROMPT_BUDGET_CONTENT_FORMAT_OPENAI!r}, got {content_format!r}")
    out: List[Dict[str, object]] = []
    for m in messages:
        d = dict(m)
        if isinstance(d.get("content"), str):
            d["content"] = [{"type": "text", "text": d["content"]}]
        out.append(d)
    return out


def prompt_token_count(tok: Any, messages: Sequence[Mapping[str, str]], arm: str, *,
                       content_format: str = C.PROMPT_BUDGET_CONTENT_FORMAT_STRING) -> int:
    """The prompt's token count under ``arm``: the FULL render with the
    generation prompt appended (what the server tokenizes before it generates),
    ``chat_template_kwargs`` passed explicitly — the twin of
    ``object_identity.header_ids_under_arm``'s render call. ``content_format``
    (``contracts.prompt_budget_content_format(model_key)``): the served path's
    chat-content form for the family — the wave shakedown found gemma's
    template one token longer for a system message given as parts."""
    out = tok.apply_chat_template(content_in_format(messages, content_format), add_generation_prompt=True,
                                  tokenize=True, **render_kwargs(arm))
    raw = out["input_ids"] if hasattr(out, "keys") else out       # transformers 5.x: BatchEncoding
    return len(raw)


def cap_for(arm: str, task_name: str = CAP_TASK_NAME, *, off_cap_override: bool = False) -> Tuple[int, str]:
    """``(cap, row task)``: the ON arm's cap is the authorised experiment's
    raised one (``contracts.oos_max_tokens_on``); the OFF arm's is the table's
    (``inspect_served.MAX_TOKENS_OFF``) — or, under ``off_cap_override``,
    the authorised comparator's raised OFF cap
    (``contracts.off_cap_override_max_tokens``: 8192), so the artifact the
    comparator's preflight writes carries the cap the launch SENDS and
    ``contracts.prompt_budget_fit``'s cap equality holds. The override is an
    OFF fact: naming it under the ON arm refuses."""
    task = C.inspect_task(task_name)
    if arm == C.THINKING_NOT_AVAILABLE:
        # llama-paper-rows (P7): a NO-SWITCH family has no cap table — the cap is the explicit `--cap`
        # (the GSM8K launcher's 8,192 / 16,384), never a table's or a tuple's
        raise C.OutOfSupportPromptBudgetError(
            f"{C.OOS_PROMPT_BUDGET_ARG_ARM} {arm}: a no-thinking-channel family has no cap table (the OFF table is the "
            "retired instrument's, the ON table a thinking budget) — pass --cap explicitly (the launcher's setter value)")
    if C.enable_thinking(arm):
        if off_cap_override:
            raise C.OutOfSupportPromptBudgetError(
                f"{C.OOS_PROMPT_BUDGET_ARG_OFF_CAP_OVERRIDE} is the comparator's thinking-OFF cap "
                f"fact (contracts.AUTHORISED_OFF_CAP_COMPARATOR) and was named under "
                f"{C.OOS_PROMPT_BUDGET_ARG_ARM} {arm}: the override raises the OFF cap only")
        cap = C.oos_max_tokens_on(task_name)
        if cap is None:
            raise C.OutOfSupportPromptBudgetError(
                f"contracts.AUTHORISED_OOS_THINKING.max_tokens_on names no cap for {task_name!r}")
        return int(cap), task
    if off_cap_override:
        cap = C.off_cap_override_max_tokens(task_name)
        if cap is None:
            raise C.OutOfSupportPromptBudgetError(
                f"contracts.AUTHORISED_OFF_CAP_COMPARATOR.max_tokens_off names no cap for {task_name!r}")
        return int(cap), task
    from sisteer.capeval.tiers import inspect_served as isv

    cap = isv.MAX_TOKENS_OFF.get(task_name)
    if cap is None:
        raise C.OutOfSupportPromptBudgetError(f"inspect_served.MAX_TOKENS_OFF names no cap for {task_name!r}")
    return int(cap), task


def build_prompt_budget(messages: Sequence[Sequence[Mapping[str, str]]], tok: Any, *, arm: str,
                        model: str, model_slug: str, revision: str, cap: int, cap_task: str,
                        tree_sha: Optional[str] = None,
                        messages_source: str = C.OOS_PROMPT_BUDGET_MESSAGES_SOURCE,
                        content_format: Optional[str] = None) -> Dict[str, object]:
    """The artifact (``contracts.OOS_PROMPT_BUDGET_FIELDS``) for ``messages``
    tokenized by ``tok`` under ``arm`` — validated through the same rule the
    readers apply, so nothing is written that the gate would refuse.
    ``content_format``: the chat-content form the messages are rendered in
    (``contracts.prompt_budget_content_format(model_key)`` on the IFEval path —
    the served path's form per family; the legacy GSM8K / MATH paths keep the
    string form their artifacts were pinned with)."""
    from sisteer.models import chat_template_hash

    C.enable_thinking(arm)                                   # refuses a mode outside the two arms
    # the IFEval path (and the geometry measurement) STATE the family's form and the artifact records it; a legacy
    # caller passes none — the string form its pinned artifacts were written with, unrecorded, byte-identical
    fmt = content_format if content_format is not None else C.PROMPT_BUDGET_CONTENT_FORMAT_STRING
    lengths = [prompt_token_count(tok, m, arm, content_format=fmt) for m in messages]
    if not lengths:
        raise C.OutOfSupportPromptBudgetError("no messages to budget")
    try:
        import transformers
        tf_version = transformers.__version__
    except ImportError:                                      # a fake tokenizer in tests
        tf_version = None
    doc: Dict[str, object] = {
        "n": len(lengths), "max": max(lengths),
        "p99": percentile(lengths, 0.99), "p50": percentile(lengths, 0.50),
        "mean": round(sum(lengths) / len(lengths), 2),
        "arm": arm, "model": model, "model_slug": model_slug, "revision": revision,
        "chat_template_hash": chat_template_hash(tok),
        "cap": int(cap), "cap_task": cap_task, "max_plus_cap": max(lengths) + int(cap),
        "lengths": lengths,
        "render": {"add_generation_prompt": True,
                   C.CHAT_TEMPLATE_KWARGS_KEY: render_kwargs(arm),
                   **({"content_format": content_format} if content_format is not None else {})},
        "messages_source": messages_source,
        "tree_sha": tree_sha,
        "produced_by": {"script": C.OOS_PROMPT_BUDGET_SCRIPT,
                        "tokenizer_loader": "sisteer.models.load_tokenizer",
                        "transformers": tf_version},
    }
    return C.validate_oos_prompt_budget(doc, where="the artifact about to be written")


def _messages(repo_root: Path) -> List[List[Dict[str, str]]]:
    """The 200 message lists — ONE source, the served task's own rendering."""
    from sisteer.capeval.tiers import inspect_served as isv

    return isv.gsm8k_paired200_messages(repo_root)


def math500_messages(repo_root: Path, *, system_prompt: Optional[str] = None,
                     verify_math500: bool = False) -> List[List[Dict[str, str]]]:
    """The exact chat messages the MATH-500 plan sends for each of its 500 items,
    in plan (dataset) order — one OpenAI-shaped list per item — rendered through
    ``inspect_evals.math``'s OWN solver chain with ``generate`` a no-op (the
    construction of ``inspect_served.gsm8k_paired200_messages``, applied to the
    ``math`` task plan), with the cell's ``system_prompt`` (the prompted cell's
    SI text) prepended as the ONE system turn when given. Refuses a plan whose
    item count is not 500, a sample the dataset lacks, or non-text content."""
    import asyncio

    from inspect_ai._eval.task.util import sample_messages
    from inspect_ai.model import ModelName
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState
    from sisteer.capeval.tiers import inspect_served as isv

    plan = isv.plan_task(MATH500_INSPECT_TASK, grader_model=None, thinking=C.THINKING_OFF,
                         repo_root=repo_root, verify_math500=verify_math500)
    ids = list(plan.sample_ids or [])
    if len(ids) != 500:
        raise C.OutOfSupportPromptBudgetError(f"math500 planned {len(ids)} items, not 500")
    by_id = {str(s.id): s for s in plan.task.dataset}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise C.OutOfSupportPromptBudgetError(
            f"math500: {len(missing)} planned ids are not in the task's dataset (first {missing[:3]})")

    async def _noop_generate(state: TaskState, *a: Any, **kw: Any) -> TaskState:
        return state

    async def _render(sample: Any) -> List[Dict[str, str]]:
        state = TaskState(model=ModelName("tinycap/prompt-budget"), sample_id=sample.id,
                          epoch=1, input=sample.input, target=Target(sample.target),
                          choices=sample.choices, messages=sample_messages(sample),
                          metadata=sample.metadata if sample.metadata else {})
        state = await plan.task.solver(state, _noop_generate)
        out: List[Dict[str, str]] = []
        for m in state.messages:
            content = getattr(m, "content", None)
            if not isinstance(content, str):
                raise C.OutOfSupportPromptBudgetError(
                    f"math500 {sample.id}: a {m.role} message carries non-text content "
                    f"({type(content).__name__}) — the budget tokenizes text only")
            out.append({"role": str(m.role), "content": content})
        if system_prompt:
            out = [{"role": "system", "content": system_prompt}] + out
        return out

    async def _all() -> List[List[Dict[str, str]]]:
        return [await _render(by_id[i]) for i in ids]

    return asyncio.run(_all())


def gsm8k_paired200_class_messages(repo_root: Path, *, system_prompt: Optional[str] = None
                                   ) -> List[List[Dict[str, str]]]:
    """The 200 message lists of ONE prompt class of the paired-200 instrument: the served task's own renders
    (``inspect_served.gsm8k_paired200_messages`` — the stock 10-shot system turn + the user turn) with the prompted
    cell's SI folded into the leading system turn as the provider collapses it (``collapse_system_messages_dicts``:
    ONE system turn, the render the prompted route serves); ``system_prompt`` None = the bare / object class."""
    from sisteer.capeval.tiers import inspect_served as isv

    return isv.gsm8k_paired200_messages(repo_root, system=system_prompt)


def system_prompt_for_cell(cell: Optional[str]) -> Optional[str]:
    """The system text a cell's requests carry — the prompted cell's SI text
    (``sisteer.eval.jailbreak.resolve_instruction``, the driver's own resolver:
    ``cell_controls._cell_system``); ``None`` for ``none`` / an object cell
    (they carry no system turn)."""
    if not cell or cell == "none":
        return None
    if not cell.startswith("prompted__"):
        return None
    from sisteer.eval.jailbreak import resolve_instruction

    return str(resolve_instruction(cell[len("prompted__"):]).text)


def geometry_window(eval_name: str) -> int:
    """The registered geometry entry's ``max_model_len`` (the window the fit is
    checked against — MATH-500's own ``capability:gen_long_math`` = 17,664 by default)."""
    from sisteer.capeval.serving import geometry as sg

    try:
        return int(sg.geometry_for(eval_name).max_model_len)
    except sg.UnknownEvalGeometry as e:
        raise C.OutOfSupportPromptBudgetError(f"no geometry entry named {eval_name!r}") from e


class SidecarCurrencyUndecidable(RuntimeError):
    """The site geometry record is REVOKED, unusable or older than the probe record it names: the sidecar's currency cannot be decided
    (neither current nor stale) — a named refusal, never a silent 'no record' that would accept the sidecar on its tree alone."""


def site_geometry_generation(model_key: Optional[str]) -> Optional[str]:
    """The `result_generation` stamp of the model's site geometry record: None ONLY when no overlay is configured or no record and no
    REVOKED marker exist (the built-in tables stand — geometry.load_site_geometry's None); a :class:`SidecarCurrencyUndecidable` when the
    loader refuses (the marker stands where the record stood during a re-probe, an unreadable file, a stale generation) — the same
    refusal every other resolver makes by name, never swallowed into "no record"."""
    if not model_key:
        return None
    from sisteer.capeval.serving import geometry as G
    try:
        site = G.load_site_geometry(model_key)
    except G.SiteGeometryError as exc:
        raise SidecarCurrencyUndecidable(f"site record revoked/unverified — sidecar currency undecidable: {exc}") from None
    if site is None:
        return None
    gen = (getattr(site, "provenance", None) or {}).get("result_generation")
    return str(gen) if gen else None


def sidecar_currency(side: Mapping[str, object], *, wave_sha: Optional[str], current_generation: Optional[str]) -> Tuple[bool, str]:
    """ONE predicate for every reader of a fit sidecar (the in-job make-if-absent and the artifact check): is this sidecar THIS
    wave's? Stale by name when its `tree_sha` is not the wave sha (a budget pinned to another tree — pass 2's sidecars refused pass 3's
    rows on 2026-09-30) or when it names a `geometry_generation` that is not the current record's (fitted under a geometry since
    re-measured). A sidecar with no generation against a known current one is current on the tree alone (legacy). Returns (current, why)."""
    ts = side.get("tree_sha")
    if wave_sha and ts != wave_sha:
        return False, f"tree_sha {ts!r} != the wave sha {wave_sha} (a budget pinned to another tree)"
    gen = side.get("geometry_generation")
    if gen and current_generation and str(gen) != str(current_generation):
        return False, f"geometry_generation {gen!r} != the current site record's {current_generation!r} (fitted under a geometry since re-measured)"
    return True, "current"


def fit_sidecar(doc: Mapping[str, object], *, task: str, control: str, geometry_eval: str, max_model_len: int,
                system_prompt_cell: Optional[str], system_prompt: Optional[str], geometry_generation: Optional[str] = None) -> Dict[str, object]:
    """The SIDECAR beside an artifact: the geometry fit through the contract's
    own rule (``prompt_budget_fit``: arm and cap must equal the artifact's;
    ``max + cap <= window``) plus the facts the contract's key set has no room
    for — the task and control, the entry, the system-prompt cell and its sha16,
    and the artifact's ``tree_sha`` (the PIN every fit sidecar names).
    ``fits`` False never raises here — the caller refuses; the sidecar
    records the verdict either way."""
    import hashlib

    try:
        rec = C.prompt_budget_fit(doc, arm=str(doc["arm"]), cap=int(doc["cap"]), max_model_len=max_model_len,
                                  where=f"{control} prompt budget")
        fits, why = True, None
    except C.OutOfSupportPromptBudgetError as e:
        rec = {"prompt_max": doc["max"], "cap": doc["cap"], "max_model_len": max_model_len,
               "total": int(doc["max"]) + int(doc["cap"]), "margin": max_model_len - int(doc["max"]) - int(doc["cap"])}
        fits, why = False, str(e)
    return {"task": task, "control": control, "cap": int(doc["cap"]), "window": int(max_model_len),
            "geometry_eval": geometry_eval, "fits": fits, "why_not": why, **{k: rec.get(k) for k in ("prompt_max", "total", "margin", "p99")},
            "n": doc["n"], "arm": doc["arm"], "model": doc["model"], "revision": doc["revision"],
            "chat_template_hash": doc["chat_template_hash"], "tree_sha": doc.get("tree_sha"),
            "geometry_generation": geometry_generation,        # the site record's result_generation the window was fitted under (None: no record)
            "system_prompt_cell": system_prompt_cell,
            "system_prompt_sha16": (hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16] if system_prompt else None)}


def math500_fit_sidecar(doc: Mapping[str, object], *, geometry_eval: str, max_model_len: int,
                        system_prompt_cell: Optional[str], system_prompt: Optional[str]) -> Dict[str, object]:
    """The MATH-500 sidecar (:func:`fit_sidecar` under the MATH task / control)."""
    return fit_sidecar(doc, task=MATH500_INSPECT_TASK, control=TASK_MATH500, geometry_eval=geometry_eval,
                       max_model_len=max_model_len, system_prompt_cell=system_prompt_cell, system_prompt=system_prompt)


def _load_config_tokenizer(config_path: Path, model_key: str) -> Tuple[str, str, str, Any]:
    """``(hf_id, revision, model_key, tokenizer)`` from a battery config's
    ``model`` / ``model_revision`` (the campaign's pins), through
    ``sisteer.models.load_tokenizer`` at that explicit revision — for the four
    battery models, two of which the launcher's ``contracts.MODELS`` lacks."""
    import yaml

    from sisteer.models import load_tokenizer

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    hf, rev = cfg["model"], cfg["model_revision"]
    from sisteer.capeval.serving.geometry import MODEL_HF_IDS, ONBOARDED_MODEL_HF_IDS

    # the paper's eight AND the front door's onboarded models (geometry keeps them in their own table by ruling; the standups read
    # the same merged view: an onboarded model's first unit once died on `MODEL_HF_IDS` alone)
    hf_id_by_key = {**MODEL_HF_IDS, **ONBOARDED_MODEL_HF_IDS}
    if hf_id_by_key.get(model_key) != hf:
        raise C.OutOfSupportPromptBudgetError(
            f"--model-key {model_key!r} names {hf_id_by_key.get(model_key)!r} but the config serves {hf!r} "
            f"(known keys: {sorted(hf_id_by_key)})")
    return hf, rev, model_key, load_tokenizer(hf, revision=rev)


def _load_pinned_tokenizer(model_slug: str) -> Tuple[C.ModelRow, Any]:
    """The registry row and its tokenizer at the PINNED revision, through the
    loader the object-identity derivation uses."""
    from sisteer.models import load_tokenizer

    row = C.model_row(model_slug)
    return row, load_tokenizer(row.hf, revision=row.rev)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(C.OOS_PROMPT_BUDGET_ARG_MODEL, default=C.AUTHORISED_OOS_THINKING.model,
                    help=f"registry model slug ({', '.join(C.MODEL_SLUGS)}); the HF id and the "
                         "pinned revision come from contracts.MODELS")
    ap.add_argument(C.OOS_PROMPT_BUDGET_ARG_ARM, default=C.THINKING_ON, choices=list(C.THINKING_MODES),
                    help="the render to tokenize under (the gate is the ON arm's, or the OFF arm's "
                         f"under {C.OOS_PROMPT_BUDGET_ARG_OFF_CAP_OVERRIDE})")
    ap.add_argument(C.OOS_PROMPT_BUDGET_ARG_OFF_CAP_OVERRIDE, dest="off_cap_override", action="store_true",
                    help="offcap plan §5.2: the artifact's cap is the authorised comparator's raised "
                         f"thinking-OFF cap (contracts.AUTHORISED_OFF_CAP_COMPARATOR.max_tokens_off) instead "
                         f"of the table's; only with {C.OOS_PROMPT_BUDGET_ARG_ARM} {C.THINKING_OFF}")
    ap.add_argument(C.OOS_PROMPT_BUDGET_ARG_OUT, default=C.OOS_PROMPT_BUDGET_FILENAME,
                    help="where to write the artifact")
    ap.add_argument(C.OOS_PROMPT_BUDGET_ARG_REPO_ROOT, default=str(REPO),
                    help="the tree whose pins.yaml / task plan render the messages (default: "
                         "this script's own tree)")
    ap.add_argument(C.OOS_PROMPT_BUDGET_ARG_TREE_SHA, default="",
                    help="the staged tree's full sha, recorded in the artifact (the preflight "
                         "passes it; empty off-volume)")
    ap.add_argument(C.OOS_PROMPT_BUDGET_ARG_OFFLINE, action="store_true",
                    help="HF_HUB_OFFLINE=1 + HF_DATASETS_OFFLINE=1: resolve the tokenizer and the "
                         "datasets from the local cache only (the preflight's mode — it can only "
                         "pass from what is already on the volume)")
    # --- MATH-500 --------------------------------------------------------------------
    ap.add_argument("--task", default=TASK_GSM8K, choices=list(TASKS),
                    help="which task's prompts to budget: gsm8k_paired200 (default, byte-identical to before), "
                         f"math500 (the 500 MATH-500 problems through inspect_evals.math's solver, cap {MATH500_THINKING_CAP}) "
                         f"or ifeval (the 541 IFEval prompts through the stock task's solver, thinking OFF, cap {IC.IFEVAL_MAX_TOKENS_OFF} "
                         f"/ the rungs {IC.IFEVAL_RUNG_CAPS})")
    ap.add_argument("--config", default=None,
                    help="math500 / the general gsm8k_paired200 shape: a battery config whose model / model_revision "
                         "pin the tokenizer (configs/sdfbase/<lane>/sdfbase_full22_<TAG>.<task>.yaml); with --model-key")
    ap.add_argument("--model-key", default=None,
                    help="the geometry model key (gemma4e4b / qwen35_4b / glm46vflash / nemotron3nano4b; + llama8b under "
                         "--task ifeval) the config serves; recorded as the artifact's model_slug (selects the general shape with --config)")
    ap.add_argument("--system-prompt-cell", default=None,
                    help="the cell whose system turn the prompts carry (prompted__safety_refusal); "
                         "none / an object cell = no system turn (they share one artifact class)")
    ap.add_argument("--geometry-eval", default=None,
                    help="the geometry entry whose max_model_len the fit is checked against -> the .fit.json sidecar; "
                         f"math500 defaults to {MATH500_GEOMETRY_EVAL}; the general gsm8k_paired200 shape writes NO "
                         "sidecar without it (G-LENGTHS: the measurement an entry is registered from)")
    ap.add_argument("--cap", type=int, default=None,
                    help=f"the completion cap the gate adds (math500 default {MATH500_THINKING_CAP}, the contracts' "
                         "MATH500_MAX_TOKENS_DEFAULT; gsm8k_paired200 default = the arm's table / tuple cap, "
                         "8192 ON / 512 OFF — pass it EXPLICITLY for the 16,384 rung)")
    return ap


def main_math500(a) -> int:
    """``--task math500``: the artifact (contract-exact) + the ``.fit.json`` sidecar; exit 2 on ``fits: false``."""
    if not (a.config and a.model_key):
        raise SystemExit("--task math500 needs --config <battery config> and --model-key <geometry key>")
    if a.arm == C.THINKING_NOT_AVAILABLE:
        # math-eight-arms-extension: a NO-SWITCH family's MATH-500 rows (llama8b: `thinking_mode: not_available` on the
        # matrix, the launcher's sidecar check demands `arm == not_available`) render with no chat_template_kwargs
        # (render_kwargs: EMPTY under the no-switch arm) and have no cap table, so the cap is the EXPLICIT --cap (the
        # row's 16,384) — the same rule the GSM8K general shape applies; never a default. The arm is admitted for any
        # model key (the tool measures a render; which family RUNS the arm is the matrix's fact, held by the launcher's
        # sidecar check against the row) — the launch's contract, test-pinned on a switch tokenizer
        if a.cap is None:
            raise SystemExit(f"--task math500 --arm {a.arm}: a no-thinking-channel family has no cap table — pass --cap explicitly "
                             "(the launcher's row cap, 16384)")
    elif not C.enable_thinking(a.arm):
        raise SystemExit("--task math500 is the thinking-ON experiment's budget: pass --arm on (or --arm not_available with --cap for a no-switch family, P7)")
    if a.off_cap_override:
        raise SystemExit("--off-cap-override is the GSM8K comparator's fact; not with --task math500")
    hf, rev, key, tok = _load_config_tokenizer(Path(a.config), a.model_key)
    system_prompt = system_prompt_for_cell(a.system_prompt_cell)
    messages = math500_messages(Path(a.repo_root), system_prompt=system_prompt)
    cap = int(a.cap) if a.cap is not None else MATH500_THINKING_CAP
    doc = build_prompt_budget(messages, tok, arm=a.arm, model=hf, model_slug=key, revision=rev, cap=cap,
                              cap_task=C.inspect_task(MATH500_INSPECT_TASK), tree_sha=a.tree_sha or None,
                              messages_source=MATH500_MESSAGES_SOURCE)
    geometry_eval = a.geometry_eval or MATH500_GEOMETRY_EVAL
    window = geometry_window(geometry_eval)
    side = math500_fit_sidecar(doc, geometry_eval=geometry_eval, max_model_len=window,
                               system_prompt_cell=a.system_prompt_cell, system_prompt=system_prompt)
    out = Path(a.out)
    atomic_write_json(out, doc)
    atomic_write_json(Path(str(out) + ".fit.json"), side)
    print(json.dumps({k: v for k, v in doc.items() if k != "lengths"}, sort_keys=True))
    print(json.dumps(side, sort_keys=True))
    print(f"[prompt-budget] wrote {out} (+ .fit.json sidecar)")
    if not side["fits"]:
        print(f"[prompt-budget] REFUSED: {side['why_not']}")
        return 2
    return 0


def main_gsm8k_general(a) -> int:
    """``--task gsm8k_paired200 --config --model-key``: the general shape — both arms, the cap explicit (default = the
    arm's table / tuple cap), the prompted cell's SI folded; the artifact contract-exact; the ``.fit.json`` sidecar ONLY
    with ``--geometry-eval`` (exit 2 on ``fits: false``), no sidecar without it (the lengths-only G-LENGTHS pass)."""
    if a.task != TASK_GSM8K:
        # ifeval-canonical-battery: the general shape renders GSM8K prompts and NOTHING else; an IFEval
        # (or any other) task routed here would budget the WRONG prompts — refuse BY NAME, never render another task's
        raise SystemExit(f"--task {a.task} was routed to the GSM8K renderer (main_gsm8k_general renders "
                         f"{GSM8K_INSPECT_TASK}'s prompts only) — a prompt budget is measured on ITS task's prompts; "
                         f"the {TASK_IFEVAL} budget is main_ifeval's")
    if not (a.config and a.model_key):
        raise SystemExit("the general gsm8k_paired200 shape needs --config <battery config> and --model-key <geometry key>")
    if a.off_cap_override:
        raise SystemExit("--off-cap-override is the comparator's fact; not with --config / --model-key (pass --cap)")
    hf, rev, key, tok = _load_config_tokenizer(Path(a.config), a.model_key)
    system_prompt = system_prompt_for_cell(a.system_prompt_cell)
    messages = gsm8k_paired200_class_messages(Path(a.repo_root), system_prompt=system_prompt)
    if a.cap is not None:
        cap, cap_source = int(a.cap), "--cap"
    else:
        cap, _ = cap_for(a.arm)
        cap_source = "the arm's table / tuple cap (cap_for)"
    doc = build_prompt_budget(messages, tok, arm=a.arm, model=hf, model_slug=key, revision=rev, cap=cap,
                              cap_task=C.inspect_task(GSM8K_INSPECT_TASK), tree_sha=a.tree_sha or None,
                              messages_source=(GSM8K_PROMPTED_MESSAGES_SOURCE if system_prompt
                                               else C.OOS_PROMPT_BUDGET_MESSAGES_SOURCE))
    out = Path(a.out)
    atomic_write_json(out, doc)
    print(json.dumps({k: v for k, v in doc.items() if k != "lengths"}, sort_keys=True))
    if not a.geometry_eval:
        print(f"[prompt-budget] wrote {out} (lengths + cap {cap} from {cap_source}; NO fit sidecar: no --geometry-eval)")
        return 0
    window = geometry_window(a.geometry_eval)
    side = fit_sidecar(doc, task=GSM8K_INSPECT_TASK, control=C.CONTROL_GSM8K, geometry_eval=a.geometry_eval, geometry_generation=site_geometry_generation(a.model_key),
                       max_model_len=window, system_prompt_cell=a.system_prompt_cell, system_prompt=system_prompt)
    atomic_write_json(Path(str(out) + ".fit.json"), side)
    print(json.dumps(side, sort_keys=True))
    print(f"[prompt-budget] wrote {out} (+ .fit.json sidecar against {a.geometry_eval} = {window})")
    if not side["fits"]:
        print(f"[prompt-budget] REFUSED: {side['why_not']}")
        return 2
    return 0


# --------------------------------------------------------------------------- IFEval (ifeval-canonical-battery, lane 2)
IFEVAL_INSPECT_TASK = IC.IFEVAL_TASK_NAME


def ifeval_class_messages(repo_root: Path, *, system_prompt: Optional[str] = None) -> List[List[Dict[str, str]]]:
    """The 541 message lists of ONE prompt class of the IFEval instrument: the stock task's own renders
    (``ifeval_messages.ifeval_messages`` — the vendor's prompt as ONE user turn) with the prompted cell's SI folded into
    the leading system turn as the provider collapses it; ``system_prompt`` None = the bare / object class."""
    from sisteer.capeval.tiers import ifeval_messages as IFM

    return IFM.ifeval_messages(repo_root, system=system_prompt)


def _load_ifeval_tokenizer(model_key: str, config_path: Optional[Path]) -> Tuple[str, str, str, Any]:
    """``(hf_id, revision, model_key, tokenizer)`` at the IFEval PIN (``ifeval_contracts.IFEVAL_MODEL_PINS`` — the launcher
    slug's ModelRow pins) through ``sisteer.models.load_tokenizer``; a ``--config`` whose ``model`` / ``model_revision``
    disagree with the pin refuses by name (one pin per model, never two)."""
    from sisteer.models import load_tokenizer

    pins = IC.IFEVAL_MODEL_PINS
    if model_key not in pins:
        raise C.OutOfSupportPromptBudgetError(f"--task {TASK_IFEVAL}: --model-key {model_key!r} is not an IFEval model "
                                              f"({sorted(pins)})")
    hf, rev = pins[model_key]
    if config_path is not None:
        import yaml

        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        if (cfg.get("model"), cfg.get("model_revision")) != (hf, rev):
            raise C.OutOfSupportPromptBudgetError(
                f"--config {config_path} serves {cfg.get('model')}@{str(cfg.get('model_revision'))[:8]} but the IFEval pin for "
                f"{model_key} is {hf}@{rev[:8]} — the two pins disagree")
    return hf, rev, model_key, load_tokenizer(hf, revision=rev)


def main_ifeval(a) -> int:
    """``--task ifeval --arm off --model-key <key> [--config] [--system-prompt-cell] --cap <rung>``: the IFEval branch
    (P40) — the stock task's own 541 renders through ``sisteer.capeval.tiers.ifeval_messages``, the prompted cell's
    SI folded into ONE leading system turn, tokenized under the OFF render of the PINNED tokenizer, the cap EXPLICIT
    (default the OFF table's 2,048; a rung passes 4,096 / 8,192); the artifact contract-exact with ``cap_task
    inspect/ifeval``; the ``.fit.json`` sidecar ONLY with ``--geometry-eval`` (exit 2 on ``fits: false``), no sidecar
    without it (the G-LENGTHS pass a geometry entry is REGISTERED from). ``--arm on`` refuses: IFEval runs thinking OFF."""
    if not a.model_key:
        raise SystemExit(f"--task {TASK_IFEVAL} needs --model-key <one of {sorted(IC.IFEVAL_MODEL_PINS)}>")
    if C.enable_thinking(a.arm):
        raise SystemExit(f"--task {TASK_IFEVAL} is the thinking-OFF instrument's budget: pass --arm {C.THINKING_OFF}")
    if a.off_cap_override:
        raise SystemExit("--off-cap-override is the GSM8K comparator's fact; not with --task ifeval (pass --cap)")
    cap = int(a.cap) if a.cap is not None else int(IC.IFEVAL_MAX_TOKENS_OFF)
    if cap not in IC.IFEVAL_CAP_LADDER:
        raise SystemExit(f"--task {TASK_IFEVAL}: --cap {cap} is not a rung of the IFEval ladder {IC.IFEVAL_CAP_LADDER}")
    hf, rev, key, tok = _load_ifeval_tokenizer(a.model_key, Path(a.config) if a.config else None)
    system_prompt = system_prompt_for_cell(a.system_prompt_cell)
    from sisteer.capeval.tiers import ifeval_messages as IFM

    messages = ifeval_class_messages(Path(a.repo_root), system_prompt=system_prompt)
    if len(messages) != IC.IFEVAL_N:
        raise SystemExit(f"--task {TASK_IFEVAL}: rendered {len(messages)} prompts, not the frozen {IC.IFEVAL_N}")
    doc = build_prompt_budget(messages, tok, arm=a.arm, model=hf, model_slug=key, revision=rev, cap=cap,
                              cap_task=IC.IFEVAL_TASK, tree_sha=a.tree_sha or None,
                              messages_source=IFM.messages_source(prompted=bool(system_prompt)),
                              # the served path's chat-content form for the family (the wave shakedown, 2026-09-13)
                              content_format=C.prompt_budget_content_format(key))
    out = Path(a.out)
    atomic_write_json(out, doc)
    print(json.dumps({k: v for k, v in doc.items() if k != "lengths"}, sort_keys=True))
    if not a.geometry_eval:
        print(f"[prompt-budget] wrote {out} (lengths + cap {cap}; NO fit sidecar: no --geometry-eval — the G-LENGTHS pass)")
        return 0
    window = geometry_window(a.geometry_eval)
    side = fit_sidecar(doc, task=IFEVAL_INSPECT_TASK, control=IC.CONTROL_IFEVAL, geometry_eval=a.geometry_eval,
                       max_model_len=window, system_prompt_cell=a.system_prompt_cell, system_prompt=system_prompt)
    atomic_write_json(Path(str(out) + ".fit.json"), side)
    print(json.dumps(side, sort_keys=True))
    print(f"[prompt-budget] wrote {out} (+ .fit.json sidecar against {a.geometry_eval} = {window})")
    if not side["fits"]:
        print(f"[prompt-budget] REFUSED: {side['why_not']}")
        return 2
    return 0


_OFFLINE_ENV = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if not a.offline:
        return _main(a)
    # --offline holds ONLY for this call: an in-process caller (the tests, a
    # wrapper) must not inherit HF offline flags that silently break every later
    # dataset load with a different cache fingerprint (a process-wide env write
    # is exactly the pollution the single-process suite exists to catch)
    saved = {k: os.environ.get(k) for k in _OFFLINE_ENV}
    try:
        for k in _OFFLINE_ENV:
            os.environ[k] = "1"
        return _main(a)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _assert_tree_sha(repo_root: Path, tree_sha: Optional[str]) -> None:
    """``--tree-sha`` is the PIN every fit sidecar names; it must be the sha of the tree that produced the artifact — this checkout's
    HEAD — never a value the caller merely asserts (it used to be recorded unchecked)."""
    if not tree_sha:
        return
    import subprocess

    try:
        head = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        raise SystemExit(f"REFUSED: --tree-sha {tree_sha} cannot be checked against {repo_root} (git rev-parse failed: {e})")
    if head != tree_sha:
        raise SystemExit(f"REFUSED: --tree-sha {tree_sha} names a tree this checkout is not at (HEAD {head or 'unknown'}, {repo_root}) — the "
                         "sidecar's pin must be the producer's actual checkout; run the producer from the checkout at that sha")


def _main(a) -> int:
    _assert_tree_sha(Path(a.repo_root), a.tree_sha)
    if a.task == TASK_MATH500:
        return main_math500(a)
    if a.task == TASK_IFEVAL:
        # BEFORE the config / model-key route: an IFEval request must never reach the GSM8K renderer (P40)
        return main_ifeval(a)
    if a.config or a.model_key:
        return main_gsm8k_general(a)
    row, tok = _load_pinned_tokenizer(a.model)
    messages = _messages(Path(a.repo_root))
    cap, cap_task = cap_for(a.arm, off_cap_override=bool(a.off_cap_override))
    doc = build_prompt_budget(messages, tok, arm=a.arm, model=row.hf, model_slug=row.slug,
                              revision=row.rev, cap=cap, cap_task=cap_task,
                              tree_sha=a.tree_sha or None)
    out = Path(a.out)
    atomic_write_json(out, doc)               # the store side's one JSON writer: tmp + fsync + replace
    print(json.dumps({k: v for k, v in doc.items() if k != "lengths"}, sort_keys=True))
    print(f"[prompt-budget] wrote {out}")
    return 0


if __name__ == "__main__":                                      # pragma: no cover
    raise SystemExit(main())
