"""S2-parity probes.

Two probes reusing the repo's OWN S2 instruments, engine-generic through
the §2.4 Engine/ArmHandle protocol, run in every calibration trial and
every rediscovery cell (runtime.run_tier1 dispatches them; gating there):

* ``s2parity/gsm8k_0shot`` (domain math_s2parity — RETIRED 2026-09-03:
  the standard GSM8K is inspect_evals' stock ``gsm8k`` on the served
  path, ``served_capability_run.py generate --controls gsm8k --thinking
  off``; this leg's rows stay READABLE in every existing store,
  ``gsm8k_parity_rows`` still loads the pinned row set for the paired-200
  comparison bridge, and ``_run_gsm8k`` refuses to GENERATE unless
  ``SISTEER_LEGACY_GSM8K=1`` — the frozen historical analyses only;
  ``run`` skips the leg when ``spec.gsm8k_rows`` is empty): bespoke 0-shot
  GSM8K on madrylab/gsm8k-platinum, n=200 seed-0 — capability_ext's exact
  prompt convention (build_chat_prompt + _GSM8K_INSTRUCTION) and answer
  extractor (extract_gsm8k_answer, extraction-failure tracked), with
  generation routed through ``engine.generate`` (the engine enters the
  arm ctx per call; token kinds get arm.transform_prompt — this
  path renders its own prompts, §5.1).
* ``s2parity/mmlu_letter`` (domain knowledge_s2parity): the S2 panel's
  A–D letter-logprob MMLU — cais/mmlu at the pins.yaml revision,
  validation split, n=200 seed-0, 0-shot chat, capability's exact prompt
  ("Answer with a single letter." + "Answer:") and letter tokenization
  (_letter_token_ids, single-token asserted; encode_kwargs so no second
  BOS) — scored through ``engine.score_corpus`` as four
  teacher-forced one-position windows per question. argmin NLL over the
  four letters == argmax over the letter logits of one forward (same
  softmax denominator), i.e. EXACTLY the S2 panel's decision rule
  (pinned by test against a direct forward).

Row selection is the INSTRUMENT'S OWN (datasets.shuffle(seed=0).select),
at the pins.yaml revision — bit-identical rows to the S2 campaign's
loaders, so probe deltas are directly comparable to the stored S2 panel
numbers. item_ids are index+content-hash and therefore stable
across trials (pairing key); each trial processes items in its own
seed-derived order, recorded in position_order (§7.1).

capability_ext itself is NOT modified: its pure helpers are imported;
its model-bound loops are re-expressed here against the engine
protocol, emitting per-item rows.
"""

from __future__ import annotations

import contextlib
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from sisteer.capeval.manifest import sha256_hex16
from sisteer.capeval.tinycap import contracts as _tinycap_contracts
from sisteer.capeval.serving import (GENERATED_POSITIONS, RESPONSE_POSITIONS,
                                     arm_position_policy)
from sisteer.capeval.tiers.tier0 import (
    ScoredItem,
    Tier0Error,
    WindowedCorpus,
    _completed_items,
    _Emitter,
)
from sisteer.eval.capability import _mmlu_prompt as _capability_mmlu_prompt
from sisteer.eval.capability_ext import (
    _GSM8K_INSTRUCTION,
    LegacyGSM8KRefused,
    _letter_token_ids,
    _letters,
    extract_gsm8k_answer,
    parse_gsm8k_gold,
    require_legacy_gsm8k,
)
from sisteer.generate import encode_kwargs
from sisteer.models import build_chat_prompt

PARITY_TASK_GSM8K = "s2parity/gsm8k_0shot"
PARITY_TASK_MMLU = "s2parity/mmlu_letter"
#: The instrument's row count — single-sourced in the tinycap contracts
#: (PW10): the Inspect plan builds ``mmlu_letter_valid`` with it and the
#: launcher checks that task's done marker against the same number.
PARITY_N = _tinycap_contracts.PARITY_N
# The INSTRUMENT's selection seed (capability_ext convention) — never the
# trial seed: the row set must be identical across trials/arms so
# per-item pairing vs trial 0 works. The trial seed only permutes
# processing ORDER (§7.1). Single-sourced in the tinycap contracts since
# mmlu-full-1531 (the served driver hashes the full-pool item set with it).
PARITY_SLICE_SEED = _tinycap_contracts.PARITY_SLICE_SEED
TASK_VERSION = "s2parity_v1"
# The per-(task, trial) summary row's metric name. It was "acc" — the SAME
# name as the per-item rows — until 2026-08-17, when the __aggregate__ row
# rode into the paired stats as a 201st item (see noninferiority._metric_map,
# which now excludes aggregate pseudo-items on the pairing side; that
# exclusion, not this rename, is what repairs already-stored campaign rows).
PARITY_AGG_METRIC = "acc_aggregate"


class ParityError(Tier0Error):
    """A parity-probe refusal."""


def _content_id(prefix: str, i: int, text: str) -> str:
    return f"{prefix}{i:04d}-{sha256_hex16(text.encode('utf-8'))[:8]}"


def gsm8k_parity_rows(pins: Mapping[str, Any],
                      n: Optional[int] = PARITY_N) -> List[dict]:
    """The instrument's own selection (capability_ext._gsm8k_rows:
    load_dataset + shuffle(seed=0) + select), pinned to pins.yaml's
    gsm8k_platinum revision. Refuses without the pin (never floats).

    ``n=None`` selects the FULL pinned pool (the controls-expansion seam,
    2026-08-29 — 1,209 test rows at the pinned revision). The seed-0
    shuffle permutes the whole dataset and ``select`` takes its head, so
    any smaller ``n`` is a strict PREFIX of any larger one — item_ids
    included (the enumerate index is subset-relative but prefix-stable).
    Widening the probe therefore strictly extends the item set, and every
    narrower run stays interpretable as a subsample of a wider one
    (pinned on the real pinned corpus by tests/test_served_capability.py
    ``test_mmlu_full_width_is_the_pinned_pool_and_nests_the_default``)."""
    from sisteer.capeval.corpora import pin_entry

    entry = pin_entry(pins, "gsm8k_platinum")  # refusal when absent
    from datasets import load_dataset

    ds = load_dataset("madrylab/gsm8k-platinum", "main", split="test",
                      revision=entry["revision"])
    take = len(ds) if n is None else min(n, len(ds))
    ds = ds.shuffle(seed=PARITY_SLICE_SEED).select(range(take))
    return [{"item_id": _content_id("g", i, r["question"]),
             "question": r["question"], "answer": r["answer"]}
            for i, r in enumerate(ds)]


def mmlu_parity_rows(pins: Mapping[str, Any],
                     n: Optional[int] = PARITY_N) -> List[dict]:
    """The S2 panel's selection (capability_ext._mmlu_rows: cais/mmlu
    'all' validation, shuffle(seed=0) + select), pinned to pins.yaml's
    mmlu revision. ``n=None`` = the full pinned pool (1,531 validation
    rows, ``contracts.MMLU_FULL_N``); smaller widths are strict prefixes
    of larger ones — see :func:`gsm8k_parity_rows`. The served driver
    reaches ``n=None`` through ``contracts.mmlu_parity_n(full=True)``
    (its ``--mmlu-full``), which also stamps the width into the store
    identity; the 200-item default is the head of the same permutation."""
    from sisteer.capeval.corpora import pin_entry

    entry = pin_entry(pins, "mmlu")  # refusal when absent
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="validation",
                      revision=entry["revision"])
    take = len(ds) if n is None else min(n, len(ds))
    ds = ds.shuffle(seed=PARITY_SLICE_SEED).select(range(take))
    return [{"item_id": _content_id("m", i, r["question"]),
             "question": r["question"], "choices": list(r["choices"]),
             "answer": int(r["answer"])}
            for i, r in enumerate(ds)]


@dataclass
class ParitySpec:
    """Everything one parity.run(...) needs beyond engine/arm/writer.
    ``template``/``domain_of`` follow the tier0.Tier0Spec convention."""

    template: Dict[str, Any]
    domain_of: Mapping[str, str]
    gsm8k_rows: List[dict]
    mmlu_rows: List[dict]
    order_seed: int  # per-trial processing order (§7.1); NOT the slice seed
    gsm8k_task: str = PARITY_TASK_GSM8K
    mmlu_task: str = PARITY_TASK_MMLU
    # Additive (capeval-controls §3.2): a prompted defense cell's SI
    # rendered as the system message, so the seam can run MMLU/GSM8K under
    # the cell's instruction. None (the default) preserves today's
    # prompts byte-for-byte.
    system: Optional[str] = None
    gen: Dict[str, Any] = field(default_factory=lambda: {
        "max_new_tokens": 512, "do_sample": False, "batch_size": 8})


def _permuted(rows: List[dict], seed: int) -> List[dict]:
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [rows[i] for i in order]


def _mmlu_prompt(tok, row: Mapping[str, Any],
                 system: Optional[str] = None) -> str:
    """The S2 panel's exact prompt convention — the instrument's OWN
    capability._mmlu_prompt (its A-D letter zip truncates to 4 choices,
    matching the scoring side's _letters(min(k, 4))). ``system`` (the
    seam's prompted-cell SI) reproduces the same user text with the SI as
    the system message; the None path stays the instrument's own render,
    pinned equal by test."""
    if system is None:
        return _capability_mmlu_prompt(tok, row["question"], row["choices"])
    lines = [row["question"].strip()]
    for letter, choice in zip(_letters(4), row["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer with a single letter.")
    return build_chat_prompt(tok, "\n".join(lines), system=system) + "Answer:"


def _transform(arm: Optional[Any], prompt: str) -> str:
    return arm.transform_prompt(prompt) if arm is not None else prompt


def _done_acc(writer, task: str) -> Dict[str, Dict[str, float]]:
    """Per-item resume state for one probe (metric "acc"); the aggregate
    summary row is not an item and is excluded. The pop covers LEGACY
    stores only — rows filed before 2026-08-17 carry the aggregate row
    under metric "acc" (it now files under PARITY_AGG_METRIC)."""
    done = _completed_items(writer, task, {"acc"})
    done.pop("__aggregate__", None)
    return done


def _run_gsm8k(engine, arm, spec: ParitySpec, em: _Emitter, writer) -> Dict[str, Any]:
    # the RETIRED instrument's generation entry point (PW3): refuses with the
    # shared pointer unless the legacy switch is set — before any generation
    try:
        require_legacy_gsm8k(f"tiers/parity._run_gsm8k, row task {spec.gsm8k_task}")
    except LegacyGSM8KRefused as e:
        raise ParityError(f"gsm8k parity: {e}") from None
    t0 = time.time()
    tok = engine.tokenizer
    done = _done_acc(writer, spec.gsm8k_task)
    todo = [(k, r) for k, r in enumerate(_permuted(spec.gsm8k_rows, spec.order_seed))
            if r["item_id"] not in done]
    prompts = [_transform(arm, build_chat_prompt(
        tok, f"{r['question']}\n\n{_GSM8K_INSTRUCTION}",
        system=spec.system)) for _, r in todo]
    # spec.gen carries batch_size through to the ENGINE, which chunks
    # the prompts and enters a fresh arm ctx per chunk — never one
    # 200-prompt batch with 512 retained score tensors.
    outs = engine.generate(prompts, arm, dict(spec.gen)) if prompts else []
    if len(outs) != len(todo):
        raise ParityError(f"gsm8k parity: {len(outs)} generations for "
                          f"{len(todo)} prompts")

    acc_by_item = {iid: m["acc"] for iid, m in done.items()}
    failures = fallbacks = 0
    for (order_idx, row), out in zip(todo, outs):
        gold = row["answer"]
        gold_val = gold if isinstance(gold, (int, float)) else parse_gsm8k_gold(gold)
        pred, used_fallback = extract_gsm8k_answer(out.text)
        failed = pred is None
        acc = float((not failed) and abs(pred - float(gold_val)) < 1e-6)
        failures += failed
        fallbacks += (not failed) and used_fallback
        em.emit(task=spec.gsm8k_task, item_id=row["item_id"], metric="acc",
                value=acc, tier=1, position_order=order_idx,
                extra={"pred": pred, "used_fallback": bool(used_fallback)})
        em.emit(task=spec.gsm8k_task, item_id=row["item_id"],
                metric="extraction_failed", value=float(failed), tier=1,
                position_order=order_idx)
        acc_by_item[row["item_id"]] = acc
    n = len(acc_by_item)
    if n != len(spec.gsm8k_rows):
        raise ParityError(f"gsm8k parity: {n} scored items != "
                          f"{len(spec.gsm8k_rows)} rows")
    summary = {"acc": sum(acc_by_item.values()) / n, "n": n,
               "extraction_failures_new": failures,
               "extraction_fallbacks_new": fallbacks,
               "wall_clock_s": time.time() - t0}
    em.emit(task=spec.gsm8k_task, item_id="__aggregate__",
            metric=PARITY_AGG_METRIC,
            value=summary["acc"], tier=1,
            extra={k: v for k, v in summary.items() if k != "acc"},
            wall_clock_s=summary["wall_clock_s"])
    return summary


def _rfind_subseq(ids, sub) -> int:
    """Index of the LAST occurrence of ``sub`` in ``ids``, or -1."""
    n, m = len(ids), len(sub)
    for start in range(n - m, -1, -1):
        if list(ids[start:start + m]) == list(sub):
            return start
    return -1


@contextlib.contextmanager
def _response_boundary(obj, eff_h: int):
    """Temporarily point an armed response mask's boundary at a LOCATED
    per-request header start, through the object's own mask seams (the
    same two ``_set_response_positions`` arms), restoring the armed value
    after. Needed because the mmlu prompt carries an assistant PREFILL
    ("Answer:") after the header, so the armed width-derived H would mask
    the prefill text instead of header+prefill — the located boundary
    (header start onward) is the response side by construction, matching
    the lens-path semantics (assistant-prefill content is response
    content). Refuses when the object carries no armed response mask."""
    saved = []
    if getattr(obj, "fit_positions", None) == "response":
        saved.append(("fit", obj.fit_response_header_len))
        obj.set_fit_positions("response", int(eff_h))
    if (getattr(obj, "positions", None) == "response"
            and hasattr(obj, "response_header_len")):
        saved.append(("deploy", obj.response_header_len))
        obj.response_header_len = int(eff_h)
    if not saved:
        raise ParityError(
            "mmlu parity: the arm's object carries no armed response mask "
            "(neither fit_positions='response' nor the vector deployment "
            "attrs) — the response scoring path needs the mask the engine "
            "armed at arm build; refusing")
    try:
        yield
    finally:
        for kind, val in saved:
            if kind == "fit":
                obj.set_fit_positions("response", val)
            else:
                obj.response_header_len = val


def _response_letter_nlls(engine, arm, prompts_ids, letter_ids_by_row):
    """DEPLOYMENT-EXACT letter NLLs under the response mask: ONE prefill
    of the PROMPT per item inside the arm ctx — the exact forward the
    deployed serving runs to emit token 1 (the header tail of the prompt
    is inside the mask, so the final row's logits are steered) — reading
    the candidate letters' logprobs from the final position's
    distribution. argmin NLL over the letters == argmax over their logits
    (one softmax denominator): the S2 panel's decision rule, now measured
    under the serving instead of arm-free.

    This cannot ride ``score_corpus``: the teacher-forced windows forward
    prompt+letter, so the mask's width-derived boundary would land one
    token past the deployed one and the appended letter — never a real
    input position in deployment — would sit inside the mask. The
    boundary here is LOCATED per request
    (``chat_markers.generation_header_ids`` — the fit-time width's
    id-form, so the mask stays inside the fit support — matched by exact
    id subsequence; loud refusal when the header ids are not in the
    render), never computed from a width alone."""
    import torch

    from sisteer.chat_markers import ChatMarkerError, generation_header_ids

    model = getattr(engine, "model", None)
    tok = getattr(engine, "tokenizer", None)
    if model is None or tok is None:
        raise ParityError(
            "mmlu parity: response-mask scoring runs a direct prefill and "
            "needs the engine's loaded model/tokenizer (the HF engine "
            "exposes both); this engine does not — refusing rather than "
            "approximating the deployed serving")
    obj = getattr(arm, "_obj", None)
    if obj is None:
        raise ParityError(
            "mmlu parity: the arm handle exposes no intervention object "
            "(_obj) — the response scoring path cannot re-point the mask "
            "boundary; refusing")
    try:
        header = list(generation_header_ids(tok))
    except ChatMarkerError as e:
        raise ParityError(
            f"mmlu parity: cannot derive the generation-prompt assistant "
            f"header — {e}") from e
    device = next(model.parameters()).device
    out = []
    for ids, letter_ids in zip(prompts_ids, letter_ids_by_row):
        start = _rfind_subseq(ids, header)
        if start < 0:
            raise ParityError(
                "mmlu parity: the assistant header ids "
                f"{header!r} do not occur in the rendered prompt — the "
                "boundary must be located in the request's own tokens, "
                "never assumed; refusing")
        eff_h = len(ids) - start
        t = torch.tensor([list(ids)], dtype=torch.long, device=device)
        with _response_boundary(obj, eff_h):
            with torch.no_grad(), arm.ctx():
                logits = model(input_ids=t,
                               attention_mask=torch.ones_like(t)).logits
        lp = torch.log_softmax(logits[0, -1, :].double(), dim=-1)
        out.append([-float(lp[int(l)]) for l in letter_ids])
    return out


def _run_mmlu(engine, arm, spec: ParitySpec, em: _Emitter, writer) -> Dict[str, Any]:
    t0 = time.time()
    policy = arm_position_policy(arm)
    score_arm = arm
    response_mode = False
    if policy == RESPONSE_POSITIONS:
        # Response mask (2026-08-28): the deployed serving steers the
        # assistant-header tail of the prompt, so the forward that emits
        # token 1 — the predictor of the letter — is STEERED, and the
        # letter readout is a real capability measurement of the served
        # defense (the generated-only branch below is arm-free precisely
        # because ITS deployed serving leaves that forward unsteered).
        # Scored deployment-exact via _response_letter_nlls.
        response_mode = True
    elif policy == GENERATED_POSITIONS:
        # Generated-only serving (2026-08-20): this probe's single scored
        # position is the FIRST GENERATED position, and its predictor is
        # the prompt's LAST token — a prompt position, which the policy
        # serves with the plain base model (bit-equal, pinned by
        # tests/test_steer_generated_only.py). The instrument's EXACT
        # value under this serving is therefore the ARM-FREE score:
        # entering the ctx for this teacher-forced forward would steer
        # the very prompt positions the deployed serving leaves
        # unsteered. The letter instrument is base-by-construction under
        # decode-only steering — score it that way and report it, don't
        # refuse it. (Prompt text still goes through arm.transform_prompt
        # below: token-kind injection is prompt CONTENT, not steering.)
        score_arm = None
    elif policy != "all":
        raise ParityError(
            f"mmlu parity: arm resolves position policy {policy!r} — the "
            "letter-logprob probe appends the candidate letter, so a "
            "selector policy like 'last' would steer the LETTER token "
            "instead of the prompt's last token (the S2 instrument's "
            "position); the selector-aware teacher-forced scorer is a v1 "
            "item (r1 finding 1) — refusing")
    tok = engine.tokenizer
    done = _done_acc(writer, spec.mmlu_task)
    todo = [(k, r) for k, r in enumerate(_permuted(spec.mmlu_rows, spec.order_seed))
            if r["item_id"] not in done]

    items: List[ScoredItem] = []
    meta: List[dict] = []
    prompts_ids: List[List[int]] = []
    letter_ids_by_row: List[List[int]] = []
    for order_idx, row in todo:
        prompt = _transform(arm, _mmlu_prompt(tok, row, system=spec.system))
        ids = list(tok(prompt, **encode_kwargs(tok, prompt))["input_ids"])
        letters = _letters(min(len(row["choices"]), 4))
        letter_ids = _letter_token_ids(tok, letters)
        prompts_ids.append(ids)
        letter_ids_by_row.append([int(l) for l in letter_ids])
        for letter, lid in zip(letters, letter_ids):
            items.append(ScoredItem(
                item_id=f"{row['item_id']}::{letter}",
                input_ids=tuple(ids) + (int(lid),),
                score_positions=(len(ids),)))
        meta.append({"order_idx": order_idx, "row": row, "k": len(letters)})

    if response_mode:
        nlls_by_row = _response_letter_nlls(engine, arm, prompts_ids,
                                            letter_ids_by_row)
        if len(nlls_by_row) != len(meta):
            raise ParityError(
                f"mmlu parity: {len(nlls_by_row)} response-scored rows for "
                f"{len(meta)} prompts")
    else:
        scores = list(engine.score_corpus(
            WindowedCorpus(name="s2parity_mmlu", items=tuple(items)),
            score_arm))
        if len(scores) != len(items):
            raise ParityError(f"mmlu parity: {len(scores)} position scores "
                              f"for {len(items)} letter windows")
        nlls_by_row = []
        pos = 0
        for m in meta:
            nlls_by_row.append(
                [scores[pos + j].token_nll for j in range(m["k"])])
            pos += m["k"]

    acc_by_item = {iid: m["acc"] for iid, m in done.items()}
    for m, nlls in zip(meta, nlls_by_row):
        pick = min(range(m["k"]), key=nlls.__getitem__)  # argmin NLL ==
        # argmax letter logit of one forward — the S2 panel's rule
        acc = float(pick == m["row"]["answer"])
        em.emit(task=spec.mmlu_task, item_id=m["row"]["item_id"], metric="acc",
                value=acc, tier=1, position_order=m["order_idx"],
                extra={"pick": _letters(m["k"])[pick],
                       "gold": _letters(m["k"])[m["row"]["answer"]]})
        acc_by_item[m["row"]["item_id"]] = acc
    n = len(acc_by_item)
    if n != len(spec.mmlu_rows):
        raise ParityError(f"mmlu parity: {n} scored items != "
                          f"{len(spec.mmlu_rows)} rows")
    summary = {"acc": sum(acc_by_item.values()) / n, "n": n,
               "wall_clock_s": time.time() - t0}
    em.emit(task=spec.mmlu_task, item_id="__aggregate__",
            metric=PARITY_AGG_METRIC,
            value=summary["acc"], tier=1,
            extra={k: v for k, v in summary.items() if k != "acc"},
            wall_clock_s=summary["wall_clock_s"])
    return summary


def run(engine, arm, spec: ParitySpec, writer) -> Dict[str, Any]:
    """Run the probes whose row sets are non-empty, emitting per-item rows
    (metric="acc", domains per spec.domain_of) through the idempotent tier0
    emitter — resume-safe per item, exactly one aggregate row per (task,
    trial). The gsm8k leg is the RETIRED instrument (PW3): a caller that
    wants only the mmlu probe passes ``gsm8k_rows=[]`` and gets a summary
    without a ``gsm8k`` key; a caller that passes gsm8k rows without the
    legacy switch is refused by ``_run_gsm8k`` before anything runs."""
    if not spec.gsm8k_rows and not spec.mmlu_rows:
        raise ParityError("parity: empty row sets — the pinned loaders "
                          "must supply at least one probe's rows")
    em = _Emitter(writer, dict(spec.template), spec.domain_of,
                  task_version=TASK_VERSION)
    out: Dict[str, Any] = {}
    if spec.gsm8k_rows:
        out["gsm8k"] = _run_gsm8k(engine, arm, spec, em, writer)
    if spec.mmlu_rows:
        out["mmlu"] = _run_mmlu(engine, arm, spec, em, writer)
    return out
