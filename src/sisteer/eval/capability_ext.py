"""Extended capability panel under intervention objects (stage 2).

Four metrics, one intervention context each — the object-generic
generalization of scripts/object_ablation.mmlu_under:

- mmlu_accuracy      (existing eval/capability.py; A-D letter-logprob)
- gsm8k_accuracy     gsm8k-platinum, 0-shot chat, exact match on the ####
                     convention + fallback regex; extraction failure rate
                     reported (spec open question: default accepted).
                     RETIRED as a forward instrument (2026-09-03): the
                     standard GSM8K is inspect_evals' stock ``gsm8k`` on
                     the served path (``served_capability_run.py generate
                     --controls gsm8k --thinking off``). The helpers
                     (extractor, gold parser, row loader) stay importable
                     for READING existing stores; the generation entry
                     points here (``gsm8k_accuracy``, ``capability_panel``)
                     refuse unless ``SISTEER_LEGACY_GSM8K=1`` — for the
                     frozen historical analyses only (see
                     ``require_legacy_gsm8k``).
- gpqa_diamond_accuracy  198 questions, A-D letter-logprob, choices
                     shuffled deterministically per question.
- mmlu_pro_accuracy  A-J letter-logprob subset.

Every function takes `obj` (an InterventionObject with apply(model, scale),
inject(prompt), kind) or None for the unsteered baseline; token-kind
objects get their sentinel injected into every capability prompt.
GSM8K/GPQA/MMLU-Pro are REPORTED at the chosen scales but do NOT enter the
stage-1 within-budget filter, which stays MMLU+distinct3.

Every prompt-rendering metric also takes ``system`` (default ``None`` = the
historical no-system render, byte-identical): a system prompt installed via
the SAME ``build_chat_prompt`` the jailbreak leg's prompted serving uses, so
obj+system together express a STACKED defense. To measure a defense CELL, do
not assemble the halves by hand — hand the cell to
``eval.capability_cell.capability_for_cell`` (or run the full pipeline,
``scripts/cell_controls_run.py``), which serves the WHOLE defense the cell
names or refuses loudly. A hand-assembled half-defense here is
indistinguishable downstream from the full one, mislabelled.
"""

from __future__ import annotations

import contextlib
import re

import torch

from ..generate import encode_kwargs
from ..models import build_chat_prompt

_GSM8K_INSTRUCTION = (
    "Solve the problem step by step. After your reasoning, write the final "
    "numeric answer on its own line in the form '#### <answer>'."
)
# Their #### convention, tolerant of $ and thousands separators.
_GSM8K_ANS_RE = re.compile(r"####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


class LegacyGSM8KRefused(RuntimeError):
    """The retired bespoke GSM8K was asked to GENERATE without the legacy
    switch (``contracts.GSM8K_LEGACY_ENV``); the message names the standard."""


def require_legacy_gsm8k(what: str) -> None:
    """Guard at the retired instrument's GENERATION entry points (never at
    import — reading rows, extracting answers and loading the pinned row set
    stay unconditional). Refuses with the one shared pointer text unless
    ``SISTEER_LEGACY_GSM8K=1`` is set (the frozen historical analyses:
    stage2_frontier / stage2_dose1 / realign / agentic bands)."""
    # lazy: the contracts module is torch-free but lives under capeval;
    # importing it at call time keeps this module's import graph unchanged
    from ..capeval.tinycap import contracts as tc

    if not tc.legacy_gsm8k_enabled():
        raise LegacyGSM8KRefused(tc.gsm8k_retired_message(what))


def _obj_ctx(model, obj, scale: float):
    if obj is None:
        return contextlib.nullcontext()
    return obj.apply(model, scale=scale)


def _maybe_inject(obj, prompt: str) -> str:
    # startswith covers BOTH token kinds ("token", "token_appended");
    # `== "token"` silently evaluated token_faithful UNSTEERED.
    if obj is not None and (getattr(obj, "kind", "") or "").startswith("token"):
        return obj.inject(prompt)
    return prompt


def _parse_number(text: str) -> float | None:
    text = text.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(text)
    except ValueError:
        return None


def extract_gsm8k_answer(response: str) -> tuple[float | None, bool]:
    """(value, used_fallback). value None = extraction failure.

    Takes the LAST '#### N' in the response: chain-of-thought text can
    legitimately contain intermediate '####' lines, and the answer
    convention puts the final one last."""
    matches = _GSM8K_ANS_RE.findall(response)
    for cand in reversed(matches):
        val = _parse_number(cand)
        if val is not None:
            return val, False
    numbers = _NUMBER_RE.findall(response)
    for cand in reversed(numbers):
        val = _parse_number(cand)
        if val is not None:
            return val, True
    return None, True


def parse_gsm8k_gold(answer_field: str) -> float:
    """Gold answers carry '#### N' at the end (both gsm8k and -platinum)."""
    m = _GSM8K_ANS_RE.search(answer_field)
    if m is None:
        raise ValueError(f"gold answer without #### marker: {answer_field!r}")
    val = _parse_number(m.group(1))
    if val is None:
        raise ValueError(f"unparseable gold answer: {answer_field!r}")
    return val


def _gsm8k_rows(n: int = 200, seed: int = 0) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [{"question": r["question"], "answer": r["answer"]} for r in ds]


@torch.no_grad()
def gsm8k_accuracy(
    model,
    tok,
    obj=None,
    scale: float = 1.0,
    n: int = 200,
    max_new_tokens: int = 512,
    seed: int = 0,
    rows: list[dict] | None = None,
    batch_size: int = 8,
    generate_fn=None,
    system: str | None = None,
) -> dict:
    """rows: optional [{question, answer(float|str-with-####)}] (tests);
    otherwise loads madrylab/gsm8k-platinum (cleaner labels, spec).
    ``system``: optional system prompt (None = historical render, see module
    docstring)."""
    from ..generate import generate_with_object

    require_legacy_gsm8k("capability_ext.gsm8k_accuracy")
    if generate_fn is not None and obj is not None:
        raise ValueError("pass obj or generate_fn, not both (generate_fn "
                         "replaces generation, so obj would be silently ignored)")
    if rows is None:
        rows = _gsm8k_rows(n=n, seed=seed)

    rendered = [
        build_chat_prompt(tok, f"{r['question']}\n\n{_GSM8K_INSTRUCTION}",
                          system=system)
        for r in rows
    ]
    if generate_fn is not None:
        responses = generate_fn(rendered)
    else:
        # generate_with_object injects token sentinels and enters obj.apply
        # PER CHUNK: one context around all chunks left the skip_bos
        # position tracker counting across generate calls.
        responses = generate_with_object(
            model, tok, rendered, obj=obj, scale=scale,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
        )

    correct = failures = fallbacks = 0
    for row, response in zip(rows, responses):
        gold = row["answer"]
        gold_val = gold if isinstance(gold, (int, float)) else parse_gsm8k_gold(gold)
        pred, used_fallback = extract_gsm8k_answer(response)
        if pred is None:
            failures += 1
            continue
        fallbacks += used_fallback
        correct += abs(pred - float(gold_val)) < 1e-6
    n_rows = max(len(rows), 1)
    return {
        "acc": correct / n_rows,
        "extraction_failure_rate": failures / n_rows,
        "extraction_fallback_rate": fallbacks / n_rows,
        "n": len(rows),
    }


def _letters(k: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(k)]


def _letter_token_ids(tok, letters: list[str]) -> list[int]:
    """Single token id per answer letter, asserted: silently taking [-1] of
    a multi-token encoding would compare logits of the wrong tokens and
    the letter-logprob method would be quietly invalid."""
    ids = []
    for letter in letters:
        enc = tok(f" {letter}", add_special_tokens=False)["input_ids"]
        if len(enc) != 1:
            raise ValueError(
                f"answer letter {letter!r} does not encode to a single token "
                f"({enc}); letter-logprob scoring is invalid for this tokenizer"
            )
        ids.append(enc[0])
    return ids


@torch.no_grad()
def _letter_logprob_accuracy(
    model, tok, obj, scale: float, rows: list[dict], max_letters: int,
    system: str | None = None,
) -> float:
    """rows: [{question, choices(list[str]), answer(int)}]. One forward per
    question; pick = argmax over answer-letter next-token logits.
    ``system``: optional system prompt (None = historical render, see module
    docstring).

    obj.apply is entered PER QUESTION, not once around the loop: a single
    long-lived context leaves the skip_bos position tracker counting
    across forwards, so only the first question's BOS was masked (the
    per-row BOS masking in hooks._position_mask is per forward)."""
    device = next(model.parameters()).device
    correct = 0
    for row in rows:
        choices = row["choices"][:max_letters]
        letters = _letters(len(choices))
        lines = [row["question"].strip()]
        lines += [f"{letter}. {c}" for letter, c in zip(letters, choices)]
        lines.append("Answer with a single letter.")
        prompt = build_chat_prompt(tok, "\n".join(lines),
                                   system=system) + "Answer:"
        prompt = _maybe_inject(obj, prompt)
        letter_ids = _letter_token_ids(tok, letters)
        # encode_kwargs: the prompt is already chat-templated (BOS rendered
        # as text), so default special-token handling would prepend a SECOND
        # BOS — which skip_bos does NOT mask, so the steering addition lands
        # on its massive activation and distorts every letter-logprob metric
        # under steering.
        enc = tok(prompt, return_tensors="pt",
                  **encode_kwargs(tok, prompt)).to(device)
        with _obj_ctx(model, obj, scale):
            logits = model(**enc).logits[0, -1]
        pick = int(logits[letter_ids].argmax())
        correct += pick == row["answer"]
    return correct / max(len(rows), 1)


def gpqa_diamond_rows(seed: int = 0) -> list[dict]:
    """All 198 GPQA-diamond questions, choices shuffled per-question
    deterministically (the dataset stores correct + 3 incorrect)."""
    import random

    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rows = []
    for i, r in enumerate(ds):
        choices = [
            r["Correct Answer"].strip(),
            r["Incorrect Answer 1"].strip(),
            r["Incorrect Answer 2"].strip(),
            r["Incorrect Answer 3"].strip(),
        ]
        order = list(range(4))
        random.Random(seed * 10007 + i).shuffle(order)
        rows.append(
            {
                "question": r["Question"],
                "choices": [choices[j] for j in order],
                "answer": order.index(0),
            }
        )
    return rows


def gpqa_diamond_accuracy(
    model, tok, obj=None, scale: float = 1.0, seed: int = 0,
    rows: list[dict] | None = None, system: str | None = None,
) -> float:
    if rows is None:
        rows = gpqa_diamond_rows(seed=seed)
    return _letter_logprob_accuracy(model, tok, obj, scale, rows,
                                    max_letters=4, system=system)


def mmlu_pro_rows(n: int = 300, seed: int = 0) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [
        {"question": r["question"], "choices": list(r["options"]),
         "answer": int(r["answer_index"])}
        for r in ds
    ]


def mmlu_pro_accuracy(
    model, tok, obj=None, scale: float = 1.0, n: int = 300, seed: int = 0,
    rows: list[dict] | None = None, system: str | None = None,
) -> float:
    if rows is None:
        rows = mmlu_pro_rows(n=n, seed=seed)
    return _letter_logprob_accuracy(model, tok, obj, scale, rows,
                                    max_letters=10, system=system)


@torch.no_grad()
def mmlu_under_obj(
    model, tok, obj=None, scale: float = 1.0, rows: list[dict] | None = None,
    n_questions: int = 200, seed: int = 0, system: str | None = None,
) -> float:
    """MMLU A-D letter-logprob with an object intervention (object-generic
    twin of capability.mmlu_accuracy / object_ablation.mmlu_under)."""
    if rows is None:
        rows = _mmlu_rows(n_questions=n_questions, seed=seed)
    return _letter_logprob_accuracy(model, tok, obj, scale, rows,
                                    max_letters=4, system=system)


def _mmlu_rows(n_questions: int = 200, seed: int = 0) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(n_questions, len(ds))))
    return [
        {"question": r["question"], "choices": r["choices"], "answer": r["answer"]}
        for r in ds
    ]


def capability_panel(
    model,
    tok,
    obj,
    scales: list[float],
    *,
    mmlu_rows: list[dict] | None = None,
    gsm8k_rows: list[dict] | None = None,
    gpqa_rows: list[dict] | None = None,
    mmlu_pro_rows_: list[dict] | None = None,
    gsm8k_n: int = 200,
    mmlu_pro_n: int = 300,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    include_unsteered: bool = True,
) -> dict:
    """The 4-metric panel at each requested scale (+ scale 0 = unsteered
    reference unless disabled). Returns {scale: {metric: value}}.

    The panel's gsm8k branch is the RETIRED bespoke instrument: the whole
    panel refuses unless ``SISTEER_LEGACY_GSM8K=1`` (the stage-2 analyses
    that read it are frozen and re-run only under that switch)."""
    require_legacy_gsm8k("capability_ext.capability_panel's gsm8k branch")
    # Load each question set once — the per-metric defaults are seed-fixed,
    # so reloading them inside every scale iteration is pure waste.
    if gsm8k_rows is None:
        gsm8k_rows = _gsm8k_rows(n=gsm8k_n)
    if mmlu_rows is None:
        mmlu_rows = _mmlu_rows()
    if gpqa_rows is None:
        gpqa_rows = gpqa_diamond_rows()
    if mmlu_pro_rows_ is None:
        mmlu_pro_rows_ = mmlu_pro_rows(n=mmlu_pro_n)
    panel: dict = {}
    scale_list = list(scales)
    if include_unsteered and 0.0 not in scale_list:
        scale_list = [0.0] + scale_list
    for s in scale_list:
        use_obj = obj if s != 0.0 else None
        gsm8k = gsm8k_accuracy(
            model, tok, use_obj, scale=s, n=gsm8k_n, rows=gsm8k_rows,
            max_new_tokens=max_new_tokens, batch_size=batch_size,
        )
        panel[s] = {
            "mmlu": mmlu_under_obj(model, tok, use_obj, scale=s, rows=mmlu_rows),
            "gsm8k": gsm8k["acc"],
            "gsm8k_extraction_failure_rate": gsm8k["extraction_failure_rate"],
            "gpqa_diamond": gpqa_diamond_accuracy(
                model, tok, use_obj, scale=s, rows=gpqa_rows
            ),
            "mmlu_pro": mmlu_pro_accuracy(
                model, tok, use_obj, scale=s, n=mmlu_pro_n, rows=mmlu_pro_rows_
            ),
        }
    return panel
