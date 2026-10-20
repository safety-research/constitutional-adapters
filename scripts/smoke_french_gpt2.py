"""End-to-end CPU smoke: does a CAA vector extracted from French/English
contrast pairs actually shift gpt2 toward French?

Mechanism check, not a science result: extract the freeform-CAA vector
from 20 fixture pairs, then verify dose-response on 5 held-out pairs —
NLL of French text should fall as the steering scale grows, NLL of
English text should rise. Prints a table plus sample generations, exits
nonzero if the direction is wrong.

Run: uv run python scripts/smoke_french_gpt2.py
"""

import json
import sys
from pathlib import Path

import torch

from sisteer.eval.capability import perplexity
from sisteer.extraction.diffmean import extract_caa
from sisteer.generate import generate_batch
from sisteer.models import load_model_and_tokenizer

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures" / "french_pairs.json"
LAYER = 6  # mid-depth for gpt2's 12 blocks
# The raw diff-in-means norm is large relative to gpt2's residual stream;
# the usable window sits below 1x (first run: scale 2 already degenerate,
# 'é é é' spam — the narrow-window failure the literature documents).
SCALES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
N_HELDOUT = 5


def main() -> int:
    pairs = json.loads(FIXTURES.read_text())
    fit, held = pairs[:-N_HELDOUT], pairs[-N_HELDOUT:]

    model, tok = load_model_and_tokenizer("gpt2", device="cpu")
    sv = extract_caa(
        model,
        tok,
        prompts=[p["prompt"] + "\n" for p in fit],
        pos_completions=[p["fr"] for p in fit],
        neg_completions=[p["en"] for p in fit],
        layers=[LAYER],
        instruction_id="french_smoke",
    )
    diag = sv.metadata["diagnostics"][str(LAYER)]
    print(f"vector norm={sv.norms()[LAYER]:.2f}  "
          f"d'={diag['d_prime']:.2f}  "
          f"agreement={diag['directional_agreement']:.2f}")

    fr_texts = [p["fr"] for p in held]
    en_texts = [p["en"] for p in held]
    rows = []
    for scale in SCALES:
        use = sv if scale else None
        fr_ppl = perplexity(model, tok, texts=fr_texts, sv=use, scale=scale)
        en_ppl = perplexity(model, tok, texts=en_texts, sv=use, scale=scale)
        rows.append((scale, fr_ppl, en_ppl))
        print(f"scale={scale:4.1f}   ppl(French)={fr_ppl:9.2f}   "
              f"ppl(English)={en_ppl:9.2f}")

    print("\nsample generations ('The weather'):")
    for scale in SCALES:
        out = generate_batch(
            model, tok, ["The weather"], sv=sv if scale else None,
            scale=scale, max_new_tokens=25,
        )[0].replace("\n", " ")
        print(f"  scale={scale:4.1f}  {out[:110]}")

    base_fr, best_fr = rows[0][1], min(r[1] for r in rows[1:])
    base_en = rows[0][2]
    en_at_best = [r[2] for r in rows[1:] if r[1] == best_fr][0]
    ok_fr = best_fr < base_fr
    ok_en = en_at_best > base_en
    print(f"\nFrench ppl {base_fr:.1f} -> {best_fr:.1f} "
          f"({'DOWN ok' if ok_fr else 'NOT down — FAIL'})")
    print(f"English ppl {base_en:.1f} -> {en_at_best:.1f} "
          f"({'UP ok' if ok_en else 'not up — WARN'})")
    return 0 if ok_fr else 1


if __name__ == "__main__":
    torch.manual_seed(0)
    sys.exit(main())
