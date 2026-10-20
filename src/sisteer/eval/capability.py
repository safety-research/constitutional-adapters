"""Off-target / capability checks under steering.

- mmlu_accuracy: 0-shot multiple-choice scored by answer-letter logprob
  (cheap: one forward per question), on a deterministic subset.
- perplexity: mean NLL on neutral text (wikitext slice or provided
  texts) — the coherence canary; degeneration shows up here before it
  shows up in compliance numbers.

Both take an optional steering vector so the same metric runs at every
point of the scale sweep.
"""

from __future__ import annotations

import torch

from ..generate import encode_kwargs
from ..hooks import steering
from ..models import build_chat_prompt
from ..vectors import SteeringVector

_LETTERS = ["A", "B", "C", "D"]


def distinct_ngram_ratio(texts: list[str], n: int = 3) -> float:
    """Fraction of distinct word n-grams across texts. Degenerate repetition
    ('e e e ...') drives this toward 0; healthy prose sits well above 0.5.
    Preferred degeneration signal over neutral-corpus perplexity, which is
    confounded for instructions that legitimately shift the output
    distribution (e.g. a French vector raises English-text NLL)."""
    grams = []
    for t in texts:
        words = t.split()
        grams.extend(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def _mmlu_prompt(tok, question: str, choices: list[str],
                 system: str | None = None) -> str:
    lines = [question.strip()]
    for letter, choice in zip(_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer with a single letter.")
    user = "\n".join(lines)
    # system=None renders byte-identically to the historical no-system call;
    # a system prompt goes through the SAME build_chat_prompt the jailbreak
    # leg's prompted serving uses (LLMHFSteered.generate), so the system
    # message a capability run sees is byte-identical to the one the JB leg
    # serves for the same SI.
    return build_chat_prompt(tok, user, system=system) + "Answer:"


@torch.no_grad()
def mmlu_accuracy(
    model,
    tok,
    sv: SteeringVector | None = None,
    scale: float = 1.0,
    positions: str = "all",
    n_questions: int = 200,
    seed: int = 0,
    rows: list[dict] | None = None,
    system: str | None = None,
) -> float:
    """rows: optional pre-loaded [{question, choices, answer(int)}] (tests);
    otherwise loads cais/mmlu 'all' validation split.

    ``system`` (default ``None`` = the historical no-system render, byte-
    identical) installs a system prompt in every question's render — the
    prompt half of a prompted or stacked defense. To measure a defense CELL,
    do not assemble ``sv``/``system`` halves by hand: hand the cell to
    ``eval.capability_cell.capability_for_cell`` (or run the full pipeline,
    ``scripts/cell_controls_run.py``), which serves the WHOLE defense the
    cell names or refuses loudly — a hand-assembled half-defense here is
    indistinguishable downstream from the full one, mislabelled."""
    device = next(model.parameters()).device
    if rows is None:
        from datasets import load_dataset

        ds = load_dataset("cais/mmlu", "all", split="validation")
        ds = ds.shuffle(seed=seed).select(range(min(n_questions, len(ds))))
        rows = [
            {"question": r["question"], "choices": r["choices"], "answer": r["answer"]}
            for r in ds
        ]

    letter_ids = [
        tok(f" {letter}", add_special_tokens=False)["input_ids"][-1]
        for letter in _LETTERS
    ]
    correct = 0
    with steering(model, sv, scale=scale, positions=positions):
        for row in rows:
            prompt = _mmlu_prompt(tok, row["question"], row["choices"],
                                  system=system)
            # encode_kwargs: the prompt is chat-templated (BOS rendered as
            # text); the tokenizer default would prepend a SECOND BOS whose
            # massive activation receives the steering addition — same
            # convention as the stage-2 panel in capability_ext.
            enc = tok(prompt, return_tensors="pt",
                      **encode_kwargs(tok, prompt)).to(device)
            logits = model(**enc).logits[0, -1]
            pick = int(torch.tensor([logits[i] for i in letter_ids]).argmax())
            correct += pick == row["answer"]
    return correct / max(len(rows), 1)


@torch.no_grad()
def perplexity(
    model,
    tok,
    texts: list[str] | None = None,
    sv: SteeringVector | None = None,
    scale: float = 1.0,
    positions: str = "all",
    n_texts: int = 50,
    max_len: int = 512,
) -> float:
    """Mean per-token perplexity over neutral texts."""
    device = next(model.parameters()).device
    if texts is None:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        texts = [r["text"] for r in ds if len(r["text"].split()) > 50][:n_texts]

    total_nll = 0.0
    total_tokens = 0
    with steering(model, sv, scale=scale, positions=positions):
        for text in texts:
            # encode_kwargs for panel-internal convention consistency:
            # neutral wikitext is a plain string, so this is a no-op today —
            # but a caller passing pre-templated texts would otherwise get a
            # double BOS here and nowhere else.
            enc = tok(
                text, return_tensors="pt", truncation=True, max_length=max_len,
                **encode_kwargs(tok, text),
            ).to(device)
            ids = enc["input_ids"]
            if ids.shape[1] < 2:
                continue
            logits = model(**enc).logits
            logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            tok_logp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            total_nll += float(-tok_logp.sum())
            total_tokens += ids.shape[1] - 1
    return float(torch.tensor(total_nll / max(total_tokens, 1)).exp())
