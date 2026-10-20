#!/usr/bin/env python
"""Per-SI no-op floors, sampled STRATIFIED BY FAMILY, reported per family.

Supersedes the prefix-sampled floor measurement. That one took pool[:n],
and the pool is sorted by (family, goal_id, variant_key) — matched_pool's
`order()` — so "256 paired prompts" was in fact 235 aim + 21
amplegcg_plus, i.e. one family that is 1% of the pool standing in for all
eleven. The two families that are 78% of the data contributed nothing.

Three defects fixed here, two of them from the adversarial review of the
guard rework:

1. SELECTION. Deterministic hash-rank within each family stratum
   (sha256 of the pool key + seed), so the sample spans every family,
   is reproducible from the seed alone, and — because the per-SI pools
   are matched on (family, goal_id, variant_key) — selects the SAME
   prompt identities for all three SIs, keeping the cross-SI comparison
   paired rather than merely equal-sized. Selected keys are recorded in
   the output.

2. CHUNK INVARIANCE. The estimand is the PER-EXAMPLE KL, accumulated as
   sufficient statistics, never a mean of per-batch scalars over ragged
   token counts (which makes the estimate depend on batch_size; the
   guard still has this and it is not fixed here). Batching is used for
   speed only: each batch's gathered rows are split back per example by
   that row's completion-token count, so batch_size affects wall-clock
   and nothing else.

3. FAIL BEFORE THE MODEL LOAD. Inputs are validated — files exist, n > 0,
   families present, enough rows per stratum — before paying the 8B load.

Reported per SI: a per-family table (mean KL, 95% CI, n) and TWO pooled
numbers, which answer different questions and are NOT interchangeable:
  - unweighted-over-families: what an equal-family design measures;
  - POOL-WEIGHTED: each family's mean weighted by its true share of the
    pool. This is the pool-representative floor a fit criterion should
    divide by, and it is the one to compare against the old prefix number.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd() / "src"))


from sisteer.eval.pool_sampling import (pool_key,  # noqa: E402
                                        select_stratified)


def ci95(vals: list[float]) -> tuple[float, float, float]:
    m = statistics.mean(vals)
    if len(vals) < 2:
        return m, float("nan"), float("nan")
    sem = statistics.stdev(vals) / math.sqrt(len(vals))
    return m, m - 1.96 * sem, m + 1.96 * sem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--per-family", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(open(args.config))
    si_ids = list(cfg["si_ids"])

    # ---- validate EVERYTHING before the 8B load ----
    if args.per_family < 1:
        raise SystemExit("--per-family must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    pools: dict[str, list[dict]] = {}
    for si in si_ids:
        p = Path(args.pairs_dir) / f"pairs_{si}.jsonl"
        if not p.exists():
            raise SystemExit(f"missing pool file: {p}")
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not rows:
            raise SystemExit(f"empty pool file: {p}")
        pools[si] = rows
    sel = {si: select_stratified(pools[si], args.per_family, args.seed)
           for si in si_ids}
    keysets = {si: {pool_key(r) for r in sel[si]} for si in si_ids}
    shared = set.intersection(*keysets.values())
    for si in si_ids:
        n_fam = len({r.get("family") for r in sel[si]})
        print(f"[select] {si}: {len(sel[si])} rows over {n_fam} families "
              f"({len(keysets[si] - shared)} not shared with all SIs)")
    if not shared:
        raise SystemExit("no prompt identities shared across SIs — the "
                         "pools are not matched; cross-SI comparison would "
                         "not be paired")
    # Keep ONLY shared identities, in one canonical order, so every SI is
    # measured on exactly the same prompts.
    order = sorted(shared)
    for si in si_ids:
        by_key = {pool_key(r): r for r in sel[si]}
        sel[si] = [by_key[k] for k in order]
    print(f"[select] measuring {len(order)} shared identities per SI, "
          f"seed={args.seed}")

    from sisteer.data.pairs import Pair, pairs_to_distill_examples
    from sisteer.eval.jailbreak import resolve_instruction
    from sisteer.extraction.distill import soft_divergence
    from sisteer.extraction.lockstep import (_encode_aligned,
                                             _gather_completion_logits)
    from sisteer.extraction.traindata import collate
    from sisteer.models import load_model_and_tokenizer

    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    model.eval()
    device = next(model.parameters()).device

    results: dict[str, dict] = {}
    for si in si_ids:
        t0 = time.time()
        rows = sel[si]
        fams = [r.get("family", "?") for r in rows]
        pairs = [Pair(prompt=r["prompt"], pos=r["pos"], neg=r.get("neg", ""),
                      instruction_id=r.get("instruction_id", ""),
                      requirement_a=r.get("requirement_a"),
                      anti_neg=r.get("anti_neg")) for r in rows]
        examples = pairs_to_distill_examples(
            tok, pairs, resolve_instruction(si))

        per_example: list[float] = []
        ex_fams: list[str] = []
        bs = args.batch_size
        for i in range(0, len(examples), bs):
            chunk = examples[i:i + bs]
            cfams = fams[i:i + bs]
            enc = [_encode_aligned(tok, ex, args.max_len) for ex in chunk]
            tb = collate([a for a, _ in enc], tok.pad_token_id).to(device)
            sb = collate([b for _, b in enc], tok.pad_token_id).to(device)
            with torch.no_grad():
                t_all = model(input_ids=tb.input_ids,
                              attention_mask=tb.attention_mask).logits
                s_all = model(input_ids=sb.input_ids,
                              attention_mask=sb.attention_mask).logits
                # Per-example split: the gather flattens completion
                # positions row-major, so each row's slice is its own
                # completion-token count. Computing KL per slice makes the
                # estimate independent of how examples were chunked.
                t_flat = _gather_completion_logits(t_all, tb)
                s_flat = _gather_completion_logits(s_all, sb)
                counts = (tb.labels[:, 1:] != -100).sum(1).tolist()
                off = 0
                for j, k in enumerate(counts):
                    if k <= 0:
                        off += k
                        continue
                    t_logp = torch.log_softmax(
                        t_flat[off:off + k].float(), -1)
                    s_logp = torch.log_softmax(
                        s_flat[off:off + k].float(), -1)
                    kl = float(soft_divergence("soft", t_logp, s_logp))
                    off += k
                    if math.isfinite(kl):
                        per_example.append(kl)
                        ex_fams.append(cfams[j])
            if (i // bs) % 40 == 0 and per_example:
                print(f"[{si}] {len(per_example)} examples, running mean "
                      f"{statistics.mean(per_example):.5f}", flush=True)

        if not per_example:
            raise SystemExit(f"{si}: no finite KL values — refusing to "
                             "report a floor")
        by_fam: dict[str, list[float]] = collections.defaultdict(list)
        for f, v in zip(ex_fams, per_example):
            by_fam[f].append(v)
        pool_counts = collections.Counter(
            r.get("family", "?") for r in pools[si])
        total_pool = sum(pool_counts.values())

        fam_rows = {}
        for f in sorted(by_fam):
            m, lo, hi = ci95(by_fam[f])
            fam_rows[f] = {"mean_kl": m, "ci95_lo": lo, "ci95_hi": hi,
                           "n": len(by_fam[f]),
                           "pool_share": pool_counts[f] / total_pool}
        m_un, lo_un, hi_un = ci95(per_example)
        weighted = sum(fam_rows[f]["mean_kl"] * fam_rows[f]["pool_share"]
                       for f in fam_rows)
        wnorm = sum(fam_rows[f]["pool_share"] for f in fam_rows)
        results[si] = {
            "per_family": fam_rows,
            "unweighted_mean_kl": m_un,
            "unweighted_ci95": [lo_un, hi_un],
            "pool_weighted_mean_kl": weighted / wnorm if wnorm else float("nan"),
            "families_covered": len(fam_rows),
            "n_examples": len(per_example),
            "seed": args.seed, "per_family_target": args.per_family,
            "batch_size": bs, "secs": round(time.time() - t0, 1),
        }
        print(f"=== {si}: pool-weighted {results[si]['pool_weighted_mean_kl']:.5f} "
              f"| unweighted {m_un:.5f} [{lo_un:.5f}, {hi_un:.5f}] "
              f"over {len(fam_rows)} families, n={len(per_example)} ===",
              flush=True)

    print("\n============ FLOORS, STRATIFIED BY FAMILY ============")
    for si, r in results.items():
        print(f"\n{si}  pool-weighted={r['pool_weighted_mean_kl']:.5f}  "
              f"unweighted={r['unweighted_mean_kl']:.5f}")
        print(f"  {'family':<24}{'mean KL':>10}{'95% CI':>24}{'n':>6}"
              f"{'pool%':>8}")
        for f, fr in sorted(r["per_family"].items(),
                            key=lambda kv: -kv[1]["pool_share"]):
            print(f"  {f:<24}{fr['mean_kl']:>10.5f}"
                  f"   [{fr['ci95_lo']:.5f}, {fr['ci95_hi']:.5f}]"
                  f"{fr['n']:>6}{100 * fr['pool_share']:>7.1f}%")
    print("\nPrefix-sampled floors being replaced (235 aim + 21 amplegcg_plus):")
    print("  safety_refusal 0.10516 | si_constitution 0.07129 | "
          "si_irrelevant 0.06898")
    if args.out:
        payload = {"results": results, "selected_keys": [list(k) for k in order],
                   "seed": args.seed}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
