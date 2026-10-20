"""Contrastive pair generation and conversion to per-method training inputs.

A "pair file" is JSONL rows:
  {"prompt": ..., "pos": ..., "neg": ..., "instruction_id": ...}
where pos = completion generated WITH the instruction as system prompt
(the teacher / on-policy positive) and neg = completion generated with
the anti-instruction (neg_mode="anti") or with no system prompt
(neg_mode="none"). On-policy generation follows the persona-vector /
BiPO finding that completions the model actually produced under the
instruction beat hand-written ones.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..generate import generate_batch
from ..models import build_chat_prompt
from .instructions import Instruction


@dataclass
class Pair:
    prompt: str
    pos: str
    neg: str
    instruction_id: str
    # The requirement-(a) label, carried from the matched-pool manifest
    # through emit-pairs into the trainer: "strict" (the undefended model
    # COMPLIED, so ``neg`` is a genuine harmful compliance), "relaxed"
    # (undefended already refused, ``neg`` is a refusal) or "benign".
    # bidpo — a DPO preference arm — trains on "strict" alone so its
    # rejected side is a real compliance, not a refusal. None for a pool
    # emitted before the label existed (e.g. the native regime).
    requirement_a: str | None = None
    # The ANTI-SAFETY-elicited harmful-COMPLIANCE negative (bidpo amendment
    # 2026-08-20): a completion produced under ``safety_refusal.anti_text`` and
    # accepted only when the production grader scored it a genuine compliance
    # (StrongREJECT > 0.5). Present on a HARMFUL pair only, and only when the
    # anti-neg source is enabled + generated + graded; it is bidpo's dispreferred
    # side for EVERY harmful prompt (not just the ~6% the base model complied
    # with), lifting the DPO preference off the undefended refusal. Read ONLY by
    # bidpo (train_two_si_objects.pairs_for_arm, anti mode); every other arm
    # ignores it and keeps consuming ``neg`` (the matched-pool identity invariant
    # is untouched). None when the source is off or this prompt yielded no
    # accepted compliance at K.
    anti_neg: str | None = None


def generate_pairs(
    model,
    tok,
    instruction: Instruction,
    prompts: list[str],
    neg_mode: str = "anti",
    batch_size: int = 8,
    max_new_tokens: int = 200,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> list[Pair]:
    if neg_mode == "anti" and not instruction.anti_text:
        neg_mode = "none"
    pos_rendered = [
        build_chat_prompt(tok, p, system=instruction.text) for p in prompts
    ]
    neg_system = instruction.anti_text if neg_mode == "anti" else None
    neg_rendered = [build_chat_prompt(tok, p, system=neg_system) for p in prompts]

    pos_out = generate_batch(
        model, tok, pos_rendered, batch_size=batch_size,
        max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature,
    )
    neg_out = generate_batch(
        model, tok, neg_rendered, batch_size=batch_size,
        max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature,
    )
    return [
        Pair(prompt=p, pos=o_p, neg=o_n, instruction_id=instruction.id)
        for p, o_p, o_n in zip(prompts, pos_out, neg_out)
    ]


def generate_compliant_pairs(
    model,
    tok,
    instruction: Instruction,
    prompts: list[str],
    *,
    k: int = 16,
    temperature: float = 1.0,
    max_new_tokens: int = 200,
    batch_size: int = 8,
    seed: int = 0,
) -> tuple[list[Pair], dict]:
    """Rejection-sampling teacher: compliance-filtered pos.

    For each prompt, sample up to k SI-prompted completions at
    T=temperature and keep the FIRST one that passes the instruction's
    OFFICIAL checker — the exact `checkers.check` strict dispatch the eval
    scores with. A prompt with no
    compliant sample in k tries yields NO pair and is recorded as
    unyielded in the stats. neg is generated exactly as generate_pairs
    does (greedy, anti/none system) so the ablation isolates the
    pos-filter effect.

    Sampling is batched by ROUNDS: each round draws one sample for every
    still-unyielded prompt through generate_batch at batch_size, so VRAM
    stays at the ordinary generation footprint and easy prompts stop
    paying after their first compliant sample.

    Returns (pairs, yield_stats). yield_stats records n_prompts,
    n_yielded, yield_rate, mean_samples_used (over ALL prompts; an
    unyielded prompt used k), mean_samples_to_yield (yielded prompts
    only), the realized strict compliance of the kept pos set
    (pos_compliance, ~1.0 by construction — computed, not assumed), and
    the unyielded prompts themselves.
    """
    if instruction.checker == "external":
        raise ValueError(
            f"instruction {instruction.id!r} has checker='external' (scored "
            "by suite-specific machinery, such as an LLM judge), so the "
            "rejection-sampling teacher cannot score its samples; RS is "
            "programmatic-checker only"
        )
    # Lazy import: the eval layer imports this data module, so a module-
    # level import here would be circular; checkers itself is stdlib-only.
    from ..eval.checkers import check

    def _passes(completion: str) -> bool:
        return bool(check(instruction.checker, completion,
                          **instruction.checker_kwargs))

    import torch

    torch.manual_seed(seed)

    rendered = [build_chat_prompt(tok, p, system=instruction.text)
                for p in prompts]
    pos_by_idx: dict[int, str] = {}
    samples_used = [0] * len(prompts)
    pending = list(range(len(prompts)))
    for _ in range(k):
        if not pending:
            break
        outs = generate_batch(
            model, tok, [rendered[i] for i in pending],
            batch_size=batch_size, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature,
        )
        still: list[int] = []
        for i, out in zip(pending, outs):
            samples_used[i] += 1
            if _passes(out):
                pos_by_idx[i] = out
            else:
                still.append(i)
        pending = still

    yielded = sorted(pos_by_idx)
    # neg exactly as generate_pairs (greedy, anti-instruction or none) —
    # unchanged by design, so filtered-vs-unfiltered differs ONLY in pos.
    neg_system = instruction.anti_text or None
    neg_out: list[str] = []
    if yielded:
        neg_rendered = [build_chat_prompt(tok, prompts[i], system=neg_system)
                        for i in yielded]
        neg_out = generate_batch(
            model, tok, neg_rendered, batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
    pairs = [
        Pair(prompt=prompts[i], pos=pos_by_idx[i], neg=neg,
             instruction_id=instruction.id)
        for i, neg in zip(yielded, neg_out)
    ]
    n = len(prompts)
    kept_pass = [_passes(p.pos) for p in pairs]
    yield_stats = {
        "instruction_id": instruction.id,
        "k": k,
        "temperature": temperature,
        "seed": seed,
        "n_prompts": n,
        "n_yielded": len(pairs),
        "yield_rate": (len(pairs) / n) if n else 0.0,
        "mean_samples_used": (sum(samples_used) / n) if n else 0.0,
        "mean_samples_to_yield": (
            sum(samples_used[i] for i in yielded) / len(yielded)
            if yielded else None
        ),
        "pos_compliance": (
            sum(kept_pass) / len(kept_pass) if kept_pass else None
        ),
        "unyielded": [
            {"idx": i, "prompt": prompts[i]}
            for i in range(n) if i not in pos_by_idx
        ],
    }
    return pairs, yield_stats


def save_pairs(pairs: list[Pair], path: str | Path) -> None:
    """Atomic publish (tmp + fsync + os.replace, the stage2_run._write_json
    pattern): pairs.jsonl is a SHARED artifact that concurrent arm cells
    resume from by existence, so a preemption mid-write must never leave a
    truncated file that exists()-based resume then trusts — the file either
    doesn't exist or is complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w") as f:
            for pair in pairs:
                f.write(json.dumps(pair.__dict__) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_pairs(path: str | Path) -> list[Pair]:
    """The pairs in ``path``. An evaluation-only corpus (the ReNeLLM paper corpus, shipped under
    terms that forbid training on it) never reaches a method that trains on these pairs: it is
    refused by path, file hash and content hash, and by each pair's prompt, before any pair is
    returned (:mod:`sisteer.eval.attacks.eval_only`; counts and hashes only)."""
    from ..eval.attacks import eval_only

    raw = Path(path).read_bytes()
    consumer = f"the pairs loader ({path})"
    eval_only.refuse_corpus(consumer=consumer, path=path, raw=raw)
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    eval_only.refuse_texts((r.get("prompt") for r in rows), consumer=consumer)
    return [Pair(**r) for r in rows]


# -- conversions to per-method training inputs --------------------------------


def pairs_to_caa_inputs(tok, pairs: list[Pair]) -> tuple[list[str], list[str], list[str]]:
    """(prompts_rendered_without_si, pos_completions, neg_completions)."""
    prompts = [build_chat_prompt(tok, p.prompt, system=None) for p in pairs]
    return prompts, [p.pos for p in pairs], [p.neg for p in pairs]


def pairs_to_pref_examples(tok, pairs: list[Pair]):
    from ..extraction.bidpo import PrefExample

    return [
        PrefExample(
            prompt=build_chat_prompt(tok, p.prompt, system=None),
            chosen=p.pos,
            rejected=p.neg,
        )
        for p in pairs
    ]


def pairs_to_distill_examples(tok, pairs: list[Pair], instruction: Instruction,
                              instruction_idx: int = 0):
    from ..extraction.distill import DistillExample

    return [
        DistillExample(
            prompt_with_si=build_chat_prompt(tok, p.prompt, system=instruction.text),
            prompt_without=build_chat_prompt(tok, p.prompt, system=None),
            completion=p.pos,
            instruction_idx=instruction_idx,
        )
        for p in pairs
    ]


def docs_to_ce_examples(tok, pairs: list[Pair]):
    """Document-mode example builder (sdf-base-steering §4): VERBATIM.

    ``prompt_with_si == prompt_without == p.prompt`` — no
    ``build_chat_prompt``, no template, no strip, no separator edits — and
    ``completion = p.pos``. The doc pools carry the loss-masked prefix in
    the ``prompt`` slot (advice ``system + "\\n\\n"``, or ``""``) and the
    trained text in ``pos``; chat-rendering either would train a different
    corpus than the one the sampler budgeted, and
    :func:`~sisteer.models.build_chat_prompt` mutates text even on
    template-less tokenizers (strip + ``"\\n"``). The teacher leg is never
    forwarded under CE; making the two prompt fields identical keeps
    DistillExample / ``_encode_aligned`` / tokcache reuse for free
    (identical strings share one cache key, and
    ``max(len(t_prompt), len(s_prompt))`` budgets off the one real
    prompt). ``tok`` is accepted for signature parity with
    :func:`pairs_to_distill_examples` and deliberately unused — a builder
    that consulted the tokenizer could start rendering."""
    del tok  # verbatim by construction — see the docstring
    from ..extraction.distill import DistillExample

    return [
        DistillExample(prompt_with_si=p.prompt, prompt_without=p.prompt,
                       completion=p.pos, instruction_idx=0)
        for p in pairs
    ]


def instruction_diff_inputs(
    tok, instruction: Instruction, prompts: list[str]
) -> tuple[list[str], list[str]]:
    """Stolfo-style: same prompt rendered with vs without the instruction
    (no completions needed)."""
    with_si = [build_chat_prompt(tok, p, system=instruction.text) for p in prompts]
    without = [build_chat_prompt(tok, p, system=None) for p in prompts]
    return with_si, without


# -- StrongREJECT-rejection-sampled safety teacher ----------------------------
# The S2 distillation teacher is the base model WITH a system instruction,
# teacher-forced on its own completions. For a "refuse harmful requests"
# safety SI, some teacher rollouts on harmful prompts COMPLY (the base model's
# safety under a bare system nudge is imperfect), and those compliant rollouts
# would become distillation targets — the compiled vector then inherits the
# very compliance the defense exists to fix.
#
# The fix is rejection SAMPLING, not one-shot filtering: draw MULTIPLE
# teacher candidates per prompt, grade ALL of them, keep ONE verified-good
# candidate per prompt, with a fixed retry cap; a prompt still unresolved at
# the cap is REPORTED (loud unresolved-prompt rate), never silently dropped.
# Filtering one completion would make the post-filter refusal rate
# tautologically ~100% AND drop exactly the hard prompts, reproducing the
# failure. Grading is injected (grade_round) so the pure selection logic is
# testable and the Batch API plumbing stays in the grading host's driver.


@dataclass
class RejectionReport:
    """The loud rejection-gate outputs — reported, never the tautological
    post-filter rate. ``unresolved_ids`` are ids only (no harmful text)."""

    side: str                       # "harmful" | "benign"
    n_prompts: int
    n_resolved: int
    n_unresolved: int
    unresolved_ids: list[str] = field(default_factory=list)
    acceptance_prob: float = 0.0    # good candidates / all graded candidates
    mean_retries: float = 0.0       # mean resample rounds among resolved prompts
    rounds_used: int = 0
    n_candidates_graded: int = 0
    K: int = 0
    max_rounds: int = 0


def _rejection_seed(prompt: str, round_idx: int, k: int, seed: int) -> int:
    """Stable per-(prompt, round, candidate) sampling seed so a resumed run
    re-draws the same candidates."""
    h = hashlib.sha256(f"{seed}|{prompt}|{round_idx}|{k}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def _sample_teacher_candidates(
    model, tok, instruction: Instruction, prompt: str, K: int, *,
    round_idx: int, seed: int, temperature: float, max_new_tokens: int,
) -> list[str]:
    """K sampled teacher completions for one prompt under the safety SI as
    system (each seeded per (prompt, round, k) — reproducible)."""
    import torch

    rendered = build_chat_prompt(tok, prompt, system=instruction.text)
    outs: list[str] = []
    for k in range(K):
        torch.manual_seed(_rejection_seed(prompt, round_idx, k, seed))
        out = generate_batch(
            model, tok, [rendered], batch_size=1, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature,
        )
        outs.append(out[0])
    return outs


def generate_pairs_rejection(
    model,
    tok,
    instruction: Instruction,
    prompts: list[str],
    *,
    grade_round,
    K: int = 8,
    max_rounds: int = 4,
    temperature: float = 0.8,
    seed: int = 0,
    max_new_tokens: int = 512,
    side: str = "harmful",
    prompt_ids: list[str] | None = None,
    sample_fn=None,
) -> tuple[list[Pair], RejectionReport]:
    """Rejection-sample a verified teacher (JB2). For each prompt draw ``K``
    candidate completions under ``instruction`` as system, grade them all via
    ``grade_round`` and keep the FIRST verified-good one; resample up to
    ``max_rounds`` for prompts with no good candidate yet.

    ``grade_round(pairs) -> {(prompt, completion): bool}`` is injected — for
    harmful prompts a good candidate is a StrongREJECT ``refused == 1`` (a
    genuine refusal); for benign prompts a good candidate is an XSTest
    full-compliance (``C``). This keeps the Batch grading in the driver and
    makes the selection loop pure/testable. ``sample_fn(prompt, round_idx) ->
    list[str]`` is the generation test seam.

    Returns (kept_pairs, report). Pairs feed ``pairs_to_distill_examples`` ->
    ``train_distill(loss='soft')`` unchanged (``neg`` is unused by soft
    distillation and left empty). Unresolved prompts are EXCLUDED from
    distillation and COUNTED in the report — never silently dropped."""
    prompts = list(prompts)
    ids = list(prompt_ids) if prompt_ids is not None else [
        str(i) for i in range(len(prompts))
    ]
    if len(ids) != len(prompts):
        raise ValueError("prompt_ids must match prompts in length")

    # MED: key ALL selection state by prompt INDEX, not prompt TEXT. Two prompts
    # with identical text but distinct ids must NOT collapse into one slot — a
    # text-keyed `kept`/`unresolved`/`id_of` would drop one of them (and
    # `id_of` would even mis-report the surviving id). Index keying keeps each
    # prompt slot independent; the injected grader still keys verdicts by
    # (text, completion) (identical text -> identical candidates + verdict).
    n = len(prompts)
    kept: dict[int, Pair] = {}
    retries: dict[int, int] = {}
    graded_total = 0
    graded_good = 0
    unresolved = list(range(n))
    rounds_used = 0

    for rnd in range(max_rounds):
        if not unresolved:
            break
        rounds_used = rnd + 1
        cand_map: dict[int, list[str]] = {}
        for i in unresolved:
            p = prompts[i]
            cand_map[i] = (
                sample_fn(p, rnd) if sample_fn is not None
                else _sample_teacher_candidates(
                    model, tok, instruction, p, K, round_idx=rnd, seed=seed,
                    temperature=temperature, max_new_tokens=max_new_tokens,
                )
            )
        to_grade = [(prompts[i], c) for i in unresolved for c in cand_map[i]]
        verdicts = grade_round(to_grade)  # grade ALL candidates
        new_unresolved: list[int] = []
        for i in unresolved:
            p = prompts[i]
            good = None
            for c in cand_map[i]:
                graded_total += 1
                v = bool(verdicts[(p, c)])
                graded_good += int(v)
                if v and good is None:
                    good = c
            if good is not None:
                kept[i] = Pair(prompt=p, pos=good, neg="",
                               instruction_id=instruction.id)
                retries[i] = rnd
            else:
                new_unresolved.append(i)
        unresolved = new_unresolved

    report = RejectionReport(
        side=side,
        n_prompts=n,
        n_resolved=len(kept),
        n_unresolved=len(unresolved),
        unresolved_ids=[ids[i] for i in unresolved],
        acceptance_prob=(graded_good / graded_total) if graded_total else 0.0,
        mean_retries=(sum(retries.values()) / len(retries)) if retries else 0.0,
        rounds_used=rounds_used,
        n_candidates_graded=graded_total,
        K=K,
        max_rounds=max_rounds,
    )
    kept_pairs = [kept[i] for i in range(n) if i in kept]
    return kept_pairs, report


def replay_rejection_selection(
    prompt_order: list[str],
    text_of: dict[str, str],
    cand_by: dict[str, dict[tuple[int, int], str]],
    good_by: dict[str, dict[tuple[int, int], bool]],
    *,
    K: int,
    max_rounds: int,
    side: str,
    instruction_id: str,
) -> tuple[list[Pair], "RejectionReport"]:
    """Replay the rejection-sampling SELECTION over an already-generated candidate
    pool + its grades (B1 node/the grading host split): candidate generation (GPU node) and
    Batch grading (the grading host) are now separate steps, so selection is a PURE replay
    over ``cand_by[prompt_id][(round, k)] -> completion`` and
    ``good_by[prompt_id][(round, k)] -> bool``.

    Walks rounds 0..``max_rounds``-1 per prompt; within a round grades ALL K
    candidates (honest acceptance denominator) and keeps the FIRST good one, then
    stops (later rounds are not counted for a resolved prompt — the same adaptive
    accounting :func:`generate_pairs_rejection` reports). Unresolved prompts are
    COUNTED, never silently dropped. Returns (kept_pairs in ``prompt_order``,
    report)."""
    kept: dict[str, Pair] = {}
    retries: dict[str, int] = {}
    graded_total = 0
    graded_good = 0
    unresolved: list[str] = []

    for pid in prompt_order:
        good_comp = None
        resolved_round = None
        cands = cand_by.get(pid, {})
        goods = good_by.get(pid, {})
        for rnd in range(max_rounds):
            round_has_candidate = False
            for k in range(K):
                comp = cands.get((rnd, k))
                if comp is None:
                    continue
                round_has_candidate = True
                graded_total += 1
                g = bool(goods.get((rnd, k), False))
                graded_good += int(g)
                if g and good_comp is None:
                    good_comp = comp
            if good_comp is not None:
                resolved_round = rnd
                break
            if not round_has_candidate:
                break  # pool exhausted for this prompt
        if good_comp is not None:
            kept[pid] = Pair(prompt=text_of[pid], pos=good_comp, neg="",
                             instruction_id=instruction_id)
            retries[pid] = resolved_round
        else:
            unresolved.append(pid)

    walked = [retries[pid] + 1 for pid in kept]
    if unresolved:
        walked.append(max_rounds)
    report = RejectionReport(
        side=side,
        n_prompts=len(prompt_order),
        n_resolved=len(kept),
        n_unresolved=len(unresolved),
        unresolved_ids=list(unresolved),
        acceptance_prob=(graded_good / graded_total) if graded_total else 0.0,
        mean_retries=(sum(retries.values()) / len(retries)) if retries else 0.0,
        rounds_used=max(walked) if walked else 0,
        n_candidates_graded=graded_total,
        K=K,
        max_rounds=max_rounds,
    )
    kept_pairs = [kept[pid] for pid in prompt_order if pid in kept]
    return kept_pairs, report
