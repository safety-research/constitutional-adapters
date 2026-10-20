#!/usr/bin/env python
"""Settle CI-ambiguous floor verdicts with a PAIRED per-batch test.

The re-score against job 187947's measured floors left three diffmean
cells straddling the <1.0 fit criterion inside the floor's CI (sr A 1.02,
sr C 1.07, con C 1.10). A ratio of two independently-measured means cannot
settle a marginal case; a paired difference on the SAME prompts can. For
each cell this measures, per batch of the SAME examples:

    d_i = KL(teacher || student_ACTIVE)_i - KL(teacher || student_NOOP)_i

and reports mean(d) with a 95% t-CI. Verdict: FIT if the CI is entirely
below 0, NOT-FIT if entirely above, still-AMBIGUOUS if it spans 0.

Measurement conventions match the wave probe and measure_teacher_signal.py:
pairs_to_distill_examples -> _encode_aligned -> completion-position gather
-> soft_divergence("soft"). The object is applied through its OWN restored
semantics: load_object re-hydrates the persisted fit leg for torch-blob
objects, and a positions="response" object gets its per-row boundaries via
handle.set_prompt_lens(steer_prompt_boundaries(batch, H)) — the same
single mask home the trainer uses, never a reimplementation.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd() / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--cells", required=True,
                    help="json: [{name, si, object}] — object = dir holding "
                         "object.pt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--select", default="prefix",
                    choices=("prefix", "hashrank"),
                    help="prefix = pool[:n] (DEFECTIVE on a pool sorted by "
                         "(family, goal_id, variant_key): it samples one "
                         "family); hashrank = deterministic sha256 rank over "
                         "the whole pool")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import yaml

    from sisteer.data.pairs import Pair, pairs_to_distill_examples
    from sisteer.eval.jailbreak import resolve_instruction
    from sisteer.extraction.distill import (soft_divergence,
                                            steer_prompt_boundaries)
    from sisteer.extraction.lockstep import (_encode_aligned,
                                             _gather_completion_logits)
    from sisteer.extraction.traindata import collate
    from sisteer.methods import load_object
    from sisteer.models import load_model_and_tokenizer

    cfg = yaml.safe_load(open(args.config))
    cells = json.loads(Path(args.cells).read_text())
    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    model.eval()
    device = next(model.parameters()).device

    ex_cache: dict[str, list] = {}
    pop_meta: dict[str, dict] = {}
    results: dict[str, dict] = {}
    for cell in cells:
        name, si = cell["name"], cell["si"]
        t0 = time.time()
        obj = load_object(Path(cell["object"]) / "object.pt",
                          model=model, tok=tok)
        # The FIT leg, from whichever attribute this object family uses:
        # PositionMaskedFit objects (diffmean/psr/lora/loreft/lofit) carry
        # fit_positions (restored by load_blob); SteeringVector-path
        # objects carry a deploy-time .positions. Reading only .positions
        # here is exactly the bug that made run 188041 apply two
        # response-fit diffmean objects at ALL positions (the handle's
        # scalar prompt_len default of 0 makes `idx >= 0` = everything).
        positions = getattr(obj, "fit_positions",
                            getattr(obj, "positions", "all"))
        header_len = getattr(obj, "fit_response_header_len",
                             getattr(obj, "response_header_len", None))
        print(f"[{name}] applied leg: positions={positions!r} "
              f"H={header_len!r}", flush=True)
        if si not in ex_cache:
            all_rows = [json.loads(l) for l in
                        (Path(args.pairs_dir) / f"pairs_{si}.jsonl")
                        .read_text().splitlines() if l.strip()]
            if args.select == "hashrank":
                import hashlib

                def _rank(r):
                    k = (r.get("family", "?"), r.get("goal_id", ""),
                         r.get("variant_key", ""))
                    return hashlib.sha256("|".join(map(str, k))
                                          .encode("utf-8")).hexdigest()
                rows = sorted(all_rows, key=_rank)[:args.n]
            else:
                rows = all_rows[:args.n]
            # A floor/verdict reported without its POPULATION is the root
            # cause of the aim-family contamination: record what was
            # measured, not just the number.
            import collections as _c
            comp = _c.Counter(r.get("family", "?") for r in rows)
            print(f"[{si}] population: select={args.select} n={len(rows)} "
                  f"families={dict(comp)}", flush=True)
            pop_meta[si] = {"select": args.select, "n": len(rows),
                            "families": dict(comp),
                            "keys": [[r.get("family", "?"),
                                      r.get("goal_id", ""),
                                      r.get("variant_key", "")]
                                     for r in rows]}
            pairs = [Pair(prompt=r["prompt"], pos=r["pos"],
                          neg=r.get("neg", ""),
                          instruction_id=r.get("instruction_id", ""),
                          requirement_a=r.get("requirement_a"),
                          anti_neg=r.get("anti_neg")) for r in rows]
            ex_cache[si] = pairs_to_distill_examples(
                tok, pairs, resolve_instruction(si))
        examples = ex_cache[si]

        deltas, act_l, noop_l = [], [], []
        bs = max(1, args.batch_size)
        for i in range(0, len(examples), bs):
            chunk = examples[i:i + bs]
            enc = [_encode_aligned(tok, ex, args.max_len) for ex in chunk]
            tb = collate([a for a, _ in enc], tok.pad_token_id).to(device)
            sb = collate([b for _, b in enc], tok.pad_token_id).to(device)
            with torch.no_grad():
                t_logits = _gather_completion_logits(
                    model(input_ids=tb.input_ids,
                          attention_mask=tb.attention_mask).logits, tb)
                t_logp = torch.log_softmax(t_logits.float(), -1)

                def student_kl() -> float:
                    s_logits = _gather_completion_logits(
                        model(input_ids=sb.input_ids,
                              attention_mask=sb.attention_mask).logits, sb)
                    s_logp = torch.log_softmax(s_logits.float(), -1)
                    return float(soft_divergence("soft", t_logp, s_logp))

                kl_noop = student_kl()
                with obj.apply(model) as handle:
                    if positions == "response":
                        if handle is None or not hasattr(handle,
                                                         "set_prompt_lens"):
                            raise RuntimeError(
                                f"{name}: response-positions object yielded "
                                "no boundary-capable handle — refusing an "
                                "all-position measurement of a masked fit")
                        bounds = steer_prompt_boundaries(sb, header_len)
                        handle.set_prompt_lens(bounds)
                        if i == 0:
                            steered = float((sb.attention_mask.sum(1).cpu()
                                             - bounds.cpu().float()).mean())
                            print(f"[{name}] batch0 audit: mean boundary "
                                  f"{float(bounds.float().mean()):.1f}, mean "
                                  f"~{steered:.1f} steered positions/row",
                                  flush=True)
                    kl_act = student_kl()
            if math.isfinite(kl_act) and math.isfinite(kl_noop):
                deltas.append(kl_act - kl_noop)
                act_l.append(kl_act)
                noop_l.append(kl_noop)
            if len(deltas) % 32 == 0 and deltas:
                print(f"[{name}] {len(deltas)} batches, running mean "
                      f"delta={statistics.mean(deltas):+.5f}", flush=True)

        n = len(deltas)
        md = statistics.mean(deltas)
        sem = (statistics.stdev(deltas) / math.sqrt(n)) if n > 1 else float("nan")
        lo, hi = md - 1.96 * sem, md + 1.96 * sem
        verdict = ("FIT" if hi < 0 else
                   "NOT-FIT" if lo > 0 else "AMBIGUOUS")
        results[name] = {
            "si": si, "positions": positions, "n_batches": n,
            "mean_active": statistics.mean(act_l),
            "mean_noop": statistics.mean(noop_l),
            "ratio": statistics.mean(act_l) / statistics.mean(noop_l),
            "mean_delta": md, "delta_ci95": [lo, hi], "verdict": verdict,
            "secs": round(time.time() - t0, 1),
        }
        print(f"=== {name}: delta={md:+.5f} [{lo:+.5f}, {hi:+.5f}] "
              f"ratio={results[name]['ratio']:.3f} -> {verdict} ===",
              flush=True)

    print("\n================ PAIRED SETTLEMENT ================")
    for name, r in results.items():
        print(f"{name:24s} ratio={r['ratio']:.3f}  "
              f"delta={r['mean_delta']:+.5f} [{r['delta_ci95'][0]:+.5f}, "
              f"{r['delta_ci95'][1]:+.5f}]  {r['verdict']}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"results": results, "population": pop_meta}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
