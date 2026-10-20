#!/usr/bin/env python
"""Timing-only reference for the HF (in-process, hooked) jailbreak target.

The multi-turn wave's ~19-27 s per target forward bundled attacker
generation + in-loop scoring + the HF target forward, so it is a CEILING on
what serving the target on vLLM can save, not a measurement of the target's
share. This probe measures the target leg ALONE: it builds the SAME two-SI
cells the wave served (``none`` + a compiled cell) through
``build_defense_cell`` on HF transformers with the residual-stream hook, and
times ``defense.chat`` on a fixed prompt set (greedy, the wave's
``target_max_new_tokens``). The served-target smoke times the identical
prompts through the RoutingProxy, so the two numbers decompose the bundled
figure: "HF target forward = X s; vLLM target forward = Y s".

Writes one JSON with per-call latency, generated-token count and tokens/s.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: Single-turn prompts spanning short-refusal-shaped to long-answer-shaped
#: replies (the wave's replies range from refusals to 2k-token essays), plus
#: one three-turn conversation (the multi-turn attacks always send history).
TIMING_PROMPTS: list[list[dict]] = [
    [{"role": "user", "content": "Explain how vaccines train the immune system, in about 300 words."}],
    [{"role": "user", "content": "Write a short story (roughly 500 words) about a lighthouse keeper who finds a message in a bottle."}],
    [{"role": "user", "content": "Give a step-by-step recipe for a vegetarian lasagna, with quantities."}],
    [{"role": "user", "content": "Summarize the causes of the First World War in five bullet points."}],
    [{"role": "user", "content": "What is the capital of Australia?"}],
    [{"role": "user", "content": "Describe, in detail, how a refrigerator keeps food cold."}],
    [{"role": "user", "content": "I'm researching the history of cryptography for a class."},
     {"role": "assistant", "content": "Great topic. Where would you like to start — classical ciphers, the Enigma era, or modern public-key systems?"},
     {"role": "user", "content": "Start with classical ciphers: explain the Caesar and Vigenere ciphers and how each was broken."}],
]


def _count_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cells", default="none,si_constitution__psr_kl_all")
    ap.add_argument("--positions", default="response",
                    help="serve leg for compiled cells (the wave passed --positions response)")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="default cfg attacks.target_max_new_tokens")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import yaml
    from sisteer.eval.defense_cell import by_id, registry_from_cfg
    from sisteer.eval.jailbreak import build_defense_cell
    from sisteer.models import load_model_and_tokenizer

    cfg = yaml.safe_load(Path(args.config).read_text())
    max_new = args.max_new_tokens or int(cfg["attacks"]["target_max_new_tokens"])
    t0 = time.time()
    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    load_s = time.time() - t0
    reg = by_id(registry_from_cfg(cfg, repo_root=REPO))
    out = {"engine": "hf", "model": cfg["model"], "revision": cfg.get("model_revision"),
           "dtype": cfg["dtype"], "attn": cfg.get("attn_implementation", "sdpa"),
           "max_new_tokens": max_new, "load_s": round(load_s, 1), "cells": {}}
    for cell_id in args.cells.split(","):
        cell = reg[cell_id]
        positions = "all" if cell.arm is None else args.positions
        defense = build_defense_cell(model, tok, cell, positions=positions)
        # warm-up (CUDA graphs / allocator) — not recorded
        defense.chat([{"role": "user", "content": "Say hello."}], max_new_tokens=16)
        rows = []
        for rep in range(args.repeats):
            for i, msgs in enumerate(TIMING_PROMPTS):
                t1 = time.time()
                text = defense.chat(msgs, max_new_tokens=max_new)
                dt = time.time() - t1
                n_out = _count_tokens(tok, text)
                rows.append({"prompt_idx": i, "repeat": rep, "latency_s": round(dt, 3),
                             "completion_tokens": n_out,
                             "tok_per_s": round(n_out / dt, 2) if dt > 0 else None,
                             "n_messages": len(msgs), "reply_head": text[:120]})
                print(f"[hf-probe] {cell_id} p{i} r{rep}: {dt:.1f}s {n_out} tok "
                      f"({n_out / dt:.1f} tok/s)", flush=True)
        lat = [r["latency_s"] for r in rows]
        toks = [r["completion_tokens"] for r in rows]
        out["cells"][cell_id] = {
            "positions_served": positions, "n_calls": len(rows),
            "mean_latency_s": round(sum(lat) / len(lat), 3),
            "total_latency_s": round(sum(lat), 3),
            "total_completion_tokens": sum(toks),
            "tok_per_s_overall": round(sum(toks) / sum(lat), 2),
            "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[hf-probe] wrote {args.out}")


if __name__ == "__main__":
    main()
