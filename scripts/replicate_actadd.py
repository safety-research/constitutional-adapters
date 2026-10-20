"""Replicate ActAdd's (Turner et al. 2308.10248) qualitative main finding
on GPT-2-XL: the "weddings"-vs-whitespace contrast pair, front-injected,
(a) makes completions wedding-related at a large multiple of the baseline
rate, while (b) leaving perplexity on UNRELATED text nearly untouched and
LOWERING it on wedding-related text — the on-target/off-target split that
is the paper's central evidence (Table 9: related ratio 0.875, unrelated
0.994 at layer 16, c=1).

Faithfulness notes: per-position [P,d] tensor at resid_pre (paper layer l
= harness layer l-1), space-token pair padding, prefill-only front
injection. GPT-2 gets no BOS here, which is equivalent to the paper's
BOS-anchored alignment (the BOS diff is identically zero). Scored NLL
skips the steered front positions, as the paper does. OpenAI-style
frequency penalty has no HF equivalent; plain nucleus sampling is used
(noted deviation). The paper's own recipe is a grid search over
(layer, c) — first run probed a single point and found nothing; local
gpt2-124M sweeps show the effect concentrates at early layers and
c >= ~10, so the grid here covers layers {2..24} x c {3..20}.

Usage: uv run python scripts/replicate_actadd.py [--model gpt2-xl] [--quick]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sisteer.extraction.actadd import extract_actadd
from sisteer.hooks import steering
from sisteer.models import load_model_and_tokenizer

WEDDING_WORDS = ["wedding", "weddings", "wed", "marry", "married", "marriage",
                 "bride", "groom", "honeymoon"]

RELATED = [
    "The wedding was held in a small chapel and the bride wore her mother's dress.",
    "After the ceremony, the groom thanked everyone who had traveled for the wedding.",
    "They planned the wedding for months, choosing flowers, a cake, and the venue.",
    "The honeymoon began the morning after the marriage was celebrated.",
    "Guests threw rice as the newly married couple left the church.",
    "Her sister was the maid of honor and gave a toast at the reception.",
    "The couple exchanged rings and vows before two hundred guests.",
    "Choosing a wedding venue in June proved harder than the bride expected.",
    "The groom's family hosted the rehearsal dinner the night before the wedding.",
    "Their marriage ceremony blended two family traditions beautifully.",
]
UNRELATED = [
    "The committee reviewed the quarterly budget and approved new spending.",
    "Rainfall in the region has declined steadily over the past decade.",
    "The software update fixes several bugs in the payment system.",
    "He tightened the last bolt and lowered the car from the jack.",
    "The museum's new exhibit features fossils from the late Cretaceous.",
    "Interest rates were left unchanged after the central bank meeting.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Migrating birds use the stars and magnetic fields to navigate.",
    "The startup raised a funding round to expand its logistics network.",
    "A new bus route will connect the harbor with the industrial district.",
]


@torch.no_grad()
def nll_skip_front(model, tok, texts, sv=None, scale=1.0, skip: int = 0) -> float:
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for text in texts:
        enc = tok(" " + text, return_tensors="pt").to(device)  # paper prepends a space
        ids = enc["input_ids"]
        with steering(model, sv, scale=scale, positions="front") as h:
            if h is not None:
                h.reset()
            logits = model(**enc).logits
        logp = torch.log_softmax(logits[:, :-1].float(), -1)
        tok_lp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
        tok_lp = tok_lp[skip:]  # exclude predictions made from steered positions
        total += float(-tok_lp.sum())
        count += tok_lp.numel()
    return float(torch.tensor(total / max(count, 1)).exp())


@torch.no_grad()
def wedding_fraction(model, tok, sv, scale, n, quick=False) -> float:
    device = next(model.parameters()).device
    prompt = "I went up to my friend and said"
    enc = tok(prompt, return_tensors="pt").to(device)
    hits = 0
    bs = 10
    n = 20 if quick else n
    for i in range(0, n, bs):
        with steering(model, sv, scale=scale, positions="front") as h:
            if h is not None:
                h.reset()
            out = model.generate(
                **enc, max_new_tokens=40, do_sample=True, temperature=1.0,
                top_p=0.3,
                num_return_sequences=min(bs, n - i),
                pad_token_id=tok.eos_token_id,
            )
        for row in out[:, enc["input_ids"].shape[1]:]:
            text = tok.decode(row, skip_special_tokens=True).lower()
            hits += any(w in text for w in WEDDING_WORDS)
    return hits / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-xl")
    ap.add_argument("--out", default="out/replicate_actadd")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    model, tok = load_model_and_tokenizer(args.model)
    results = {}

    grid_layers = [2, 6, 10, 16, 20, 24]
    grid_cs = [3.0, 6.0, 10.0, 15.0, 20.0]
    n_per_point = 20 if args.quick else 50
    vectors = {
        pl: extract_actadd(model, tok, " weddings", " ", paper_layers=[pl],
                           instruction_id="weddings")
        for pl in grid_layers + [16]
    }
    P = vectors[grid_layers[0]].metadata["n_positions"]

    base_frac = wedding_fraction(model, tok, None, 0.0, n_per_point, args.quick)
    results["baseline_wedding_fraction"] = base_frac
    best = (0.0, None, None)  # frac, layer, c
    grid = {}
    for pl in grid_layers:
        for c in grid_cs:
            frac = wedding_fraction(model, tok, vectors[pl], c, n_per_point,
                                    args.quick)
            grid[f"l{pl}_c{c:g}"] = frac
            if frac > best[0]:
                best = (frac, pl, c)
        row = "  ".join(f"c={c:g}:{grid[f'l{pl}_c{c:g}']:.2f}" for c in grid_cs)
        print(f"wedding fraction layer {pl:2d}: {row} (baseline {base_frac:.2f})",
              flush=True)
    results["grid"] = grid
    results["best"] = {"frac": best[0], "layer": best[1], "c": best[2]}

    if best[1] is not None:
        from sisteer.generate import generate_batch  # samples at the best point

        samples = generate_batch(
            model, tok, ["I went up to my friend and said"] * 3,
            sv=vectors[best[1]], scale=best[2], positions="front",
            max_new_tokens=40, do_sample=True, temperature=1.0,
        )
        for s in samples:
            print(f"  best-point sample: {s[:110]!r}", flush=True)

    # Perplexity split at the paper's exact operating point (layer 16, c=1)
    # and at the grid-best steering point.
    base_rel = nll_skip_front(model, tok, RELATED, skip=P)
    base_unrel = nll_skip_front(model, tok, UNRELATED, skip=P)
    for tag, sv, c in [("paper_l16_c1", vectors[16], 1.0),
                       ("best", vectors.get(best[1]), best[2] or 1.0)]:
        if sv is None:
            continue
        rel = nll_skip_front(model, tok, RELATED, sv=sv, scale=c, skip=P)
        unrel = nll_skip_front(model, tok, UNRELATED, sv=sv, scale=c, skip=P)
        results[f"ppl_ratio_related_{tag}"] = rel / base_rel
        results[f"ppl_ratio_unrelated_{tag}"] = unrel / base_unrel
        print(f"ppl ratio [{tag}] related {rel / base_rel:.3f} "
              f"unrelated {unrel / base_unrel:.3f}", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=1))

    print("\n===== QUALITATIVE VERDICTS =====")
    checks = {
        "grid-best wedding fraction >= 3x baseline":
            best[0] >= max(3 * base_frac, base_frac + 0.15),
        "related-text ppl ratio < 1.0 (paper point)":
            results["ppl_ratio_related_paper_l16_c1"] < 1.0,
        "unrelated-text ppl ratio within 5% of 1 (paper point)":
            abs(results["ppl_ratio_unrelated_paper_l16_c1"] - 1.0) < 0.05,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


if __name__ == "__main__":
    main()
