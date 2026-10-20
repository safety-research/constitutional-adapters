#!/usr/bin/env python
"""Per-(cell, family) FIT RATIOS: the criterion axis of the cross-tab.

Evals owns the outcome axis (per-family ASR, its baseline, and absolute +
relative reduction); this produces the criterion axis they pair it against.

WHY THE SIDECAR NUMBERS CANNOT BE REUSED. A fit ratio is
final_probe_loss / no-op floor, and the recorded numerator was measured on
the arm's BATCH 0 — the same prefix-sorted slice that contaminated the
denominator (235 aim + 21 amplegcg_plus of an eleven-family pool). Fixing
only the floor would leave a per-family ratio with an aim-family numerator
over a per-family denominator: worse than either error alone, because it
looks corrected. Both legs are therefore re-measured here, on the SAME
stratified prompts, paired per example.

WHAT IT COMPUTES, per cell and per family:
  active   = KL(teacher || student WITH the object applied)
  noop     = KL(teacher || unsteered student)      <- the floor, same prompts
  ratio    = mean(active) / mean(noop)             <- fit criterion: < 1.0
  delta    = mean(active - noop), paired, with a 95% CI
  bootstrap CI on the ratio (paired resampling over examples in that family)
with n reported for every stratum, so a thin family is visibly thin rather
than silently dropped.

EFFICIENCY. Teacher and no-op forwards depend only on the SI, so they are
computed ONCE per SI and reused across every object of that SI — the
expensive long-prefix teacher pass is not repeated per cell.

CORRECTNESS GUARDS carried from the settlement runs:
  - the applied leg is read from fit_positions (the FIT attribute), not the
    deploy attr, and PRINTED per cell; a response-fit object gets explicit
    per-row boundaries via set_prompt_lens, and the script REFUSES rather
    than measuring a masked fit at all positions;
  - per-example sufficient statistics, so batch_size affects speed only;
  - the selection's composition is recorded in the output next to every
    number derived from it.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd() / "src"))


def ci95(vals: list[float]) -> tuple[float, float, float]:
    m = statistics.mean(vals)
    if len(vals) < 2:
        return m, float("nan"), float("nan")
    sem = statistics.stdev(vals) / math.sqrt(len(vals))
    return m, m - 1.96 * sem, m + 1.96 * sem


def boot_ratio_ci(active: list[float], noop: list[float], n_boot: int,
                  seed: int) -> tuple[float, float]:
    """Paired bootstrap on the RATIO of means (a ratio's CI is not the
    ratio of the CIs; pairing preserves the per-example correlation between
    the two legs, which is large here)."""
    if len(active) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    idx = range(len(active))
    out = []
    for _ in range(n_boot):
        s = [rng.choice(idx) for _ in idx]
        a = statistics.mean(active[i] for i in s)
        d = statistics.mean(noop[i] for i in s)
        if d > 0:
            out.append(a / d)
    if not out:
        return float("nan"), float("nan")
    out.sort()
    lo = out[max(0, int(0.025 * len(out)) - 1)]
    hi = out[min(len(out) - 1, int(0.975 * len(out)))]
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--cells", required=True,
                    help="json list of {name, si, object}")
    ap.add_argument("--per-family", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import yaml

    from sisteer.eval.pool_sampling import (composition, pool_key,
                                            select_stratified,
                                            shared_identities)

    cfg = yaml.safe_load(open(args.config))
    cells = json.loads(Path(args.cells).read_text())

    # ---- validate before the 8B load ----
    if args.per_family < 1 or args.batch_size < 1:
        raise SystemExit("--per-family and --batch-size must be >= 1")
    sis = sorted({c["si"] for c in cells})
    pools: dict[str, list[dict]] = {}
    for si in sis:
        p = Path(args.pairs_dir) / f"pairs_{si}.jsonl"
        if not p.exists():
            raise SystemExit(f"missing pool file: {p}")
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not rows:
            raise SystemExit(f"empty pool file: {p}")
        pools[si] = rows
    def artifact_of(d: Path) -> Path:
        for n in ("object.pt", "object.safetensors"):
            if (d / n).exists():
                return d / n
        raise SystemExit(f"no object.pt / object.safetensors in {d}")

    for c in cells:
        artifact_of(Path(c["object"]))
        sc = Path(c["object"]) / "train_budget_meta.json"
        if not sc.exists():
            raise SystemExit(f"missing provenance sidecar: {sc} — refusing "
                             "to measure an artifact whose fit leg is "
                             "unknown")

    sel = {si: select_stratified(pools[si], args.per_family, args.seed)
           for si in sis}
    order = shared_identities(sel)
    if not order:
        raise SystemExit("no shared prompt identities across SIs")
    for si in sis:
        by_key = {pool_key(r): r for r in sel[si]}
        sel[si] = [by_key[k] for k in order]
    comp = composition(sel[sis[0]], pools[sis[0]])
    print(f"[select] {len(order)} shared identities, "
          f"{len(comp['families'])} families, seed={args.seed}")
    for f, s in sorted(comp["shares"].items(),
                       key=lambda kv: -kv[1]["pool_pct"]):
        print(f"  {f:<24} sel {s['sel_pct']:>5.1f}%   pool {s['pool_pct']:>5.1f}%")

    from sisteer.data.pairs import Pair, pairs_to_distill_examples
    from sisteer.eval.jailbreak import resolve_instruction
    from sisteer.extraction.distill import (soft_divergence,
                                            steer_prompt_boundaries)
    from sisteer.extraction.lockstep import (_encode_aligned,
                                             _gather_completion_logits)
    from sisteer.extraction.traindata import collate
    from sisteer.methods import load_object
    from sisteer.models import load_model_and_tokenizer

    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    model.eval()
    device = next(model.parameters()).device

    def per_example_kl(t_flat, s_flat, counts) -> list[float]:
        out, off = [], 0
        for k in counts:
            if k <= 0:
                out.append(float("nan"))
                continue
            t_logp = torch.log_softmax(t_flat[off:off + k].float(), -1)
            s_logp = torch.log_softmax(s_flat[off:off + k].float(), -1)
            out.append(float(soft_divergence("soft", t_logp, s_logp)))
            off += k
        return out

    results: dict[str, dict] = {}
    for si in sis:
        rows = sel[si]
        fams = [pool_key(r)[0] for r in rows]
        pairs = [Pair(prompt=r["prompt"], pos=r["pos"], neg=r.get("neg", ""),
                      instruction_id=r.get("instruction_id", ""),
                      requirement_a=r.get("requirement_a"),
                      anti_neg=r.get("anti_neg")) for r in rows]
        examples = pairs_to_distill_examples(
            tok, pairs, resolve_instruction(si))
        si_cells = [c for c in cells if c["si"] == si]
        objs = {}
        for c in si_cells:
            d = Path(c["object"])
            o = load_object(artifact_of(d), model=model, tok=tok)
            # The SIDECAR is authoritative for what leg this object was FIT
            # on; hydration is not, and the two artifact families disagree
            # by design. A torch-blob object (diffmean/lora/psr/loreft/
            # lofit) restores its fit leg via load_blob. A SteeringVector
            # artifact (vector/bidpo) deliberately does NOT — its fit leg
            # travels in the sidecar as provenance and the deploy leg is
            # the caller's explicit choice. Measuring a fit ratio means
            # applying the FIT leg, so reconcile loudly either way.
            want = json.loads((d / "train_budget_meta.json").read_text())
            leg = want.get("positions_at_fit", "all")
            H = want.get("response_header_len")
            if hasattr(o, "fit_positions"):
                if o.fit_positions != leg:
                    raise SystemExit(
                        f"{c['name']}: hydrated fit leg {o.fit_positions!r} "
                        f"disagrees with the sidecar's {leg!r} — refusing "
                        "to guess which is the fit")
                src = "restored by load_blob"
            else:
                # vector-family: set the fit leg explicitly from provenance.
                o.positions = leg
                o.response_header_len = H
                src = "set from sidecar (vector-family hydrates unmasked)"
            # HALF-APPLIED OBJECT GUARD. Three arms (token_bos,
            # token_faithful, composition) carry half their effect in a
            # STRING transform — inject() appends/inserts a sentinel that
            # the embedding hook then substitutes. An apply()-only probe
            # leaves no sentinel in the prompt, the hook has no position to
            # act on, and the object is INERT: ratio exactly 1.000 on every
            # stratum, which reads as a balanced no-op rather than as a
            # measurement failure. sr_token_faithful_A did exactly that
            # (1.000/1.000/1.000, 2026-08-24). Detect by PROBE rather than
            # by an arm allowlist, so a new injecting arm cannot slip past.
            probe = "\u2205probe\u2205"
            injects = False
            try:
                injects = o.inject(probe) != probe
            except Exception as exc:                       # noqa: BLE001
                raise SystemExit(
                    f"{c['name']}: inject() raised on a probe string "
                    f"({exc!r}) — refusing to measure an object whose "
                    "string leg cannot be exercised")
            print(f"[{c['name']}] applied leg: positions={leg!r} H={H!r} "
                  f"({src}); injects={injects}; "
                  f"scale={float(c.get('scale', 1.0))}", flush=True)
            # Optional scale override. diffmean ships RAW and unnormalised
            # — scale 1.0 adds exactly the mean SI-presence displacement, so
            # ||r|| IS the intervention magnitude. Comparing a depth-extracted
            # object (||r|| 0.20) against its shallow twin (0.63) at natural
            # scale therefore confounds MAGNITUDE with DIRECTION: a weaker
            # result would be equally consistent with the rotation ruining
            # the direction and with nothing wrong but a 3x smaller shift.
            # A norm-matched run isolates the rotation.
            objs[c["name"]] = (o, leg, H, injects, float(c.get("scale", 1.0)))

        # Injecting objects need their OWN student encoding (the sentinel
        # changes the prompt string, hence its token ids and prompt_len).
        # The teacher and the no-op leg keep the UN-injected prompt: the
        # floor is what the unsteered model does on the real prompt, and
        # the object's injection is part of the object's effect, exactly as
        # the trainer treats it (objects.py applies inject() to
        # prompt_without before encoding).
        inj_examples = {}
        for c in si_cells:
            if objs[c["name"]][3]:
                o = objs[c["name"]][0]
                inj_examples[c["name"]] = pairs_to_distill_examples(
                    tok, [Pair(prompt=r["prompt"], pos=r["pos"],
                               neg=r.get("neg", ""),
                               instruction_id=r.get("instruction_id", ""),
                               requirement_a=r.get("requirement_a"),
                               anti_neg=r.get("anti_neg")) for r in rows],
                    resolve_instruction(si))
                inj_examples[c["name"]] = [
                    type(ex)(prompt_with_si=ex.prompt_with_si,
                             prompt_without=o.inject(ex.prompt_without),
                             completion=ex.completion,
                             instruction_idx=ex.instruction_idx)
                    for ex in inj_examples[c["name"]]]
                print(f"[{c['name']}] re-encoding student prompts through "
                      f"inject() ({len(inj_examples[c['name']])} examples)",
                      flush=True)

        noop_vals: list[float] = []
        act_vals: dict[str, list[float]] = {c["name"]: [] for c in si_cells}
        t0 = time.time()
        bs = args.batch_size
        for i in range(0, len(examples), bs):
            chunk = examples[i:i + bs]
            enc = [_encode_aligned(tok, ex, args.max_len) for ex in chunk]
            tb = collate([a for a, _ in enc], tok.pad_token_id).to(device)
            sb = collate([b for _, b in enc], tok.pad_token_id).to(device)
            counts = (tb.labels[:, 1:] != -100).sum(1).tolist()
            with torch.no_grad():
                # teacher + no-op: SI-dependent only, computed once and
                # reused by every object of this SI.
                t_flat = _gather_completion_logits(
                    model(input_ids=tb.input_ids,
                          attention_mask=tb.attention_mask).logits, tb)
                s_flat = _gather_completion_logits(
                    model(input_ids=sb.input_ids,
                          attention_mask=sb.attention_mask).logits, sb)
                noop_vals.extend(per_example_kl(t_flat, s_flat, counts))
                for name, (o, leg, H, injects, _sc) in objs.items():
                    asb = sb
                    if injects:
                        ienc = [_encode_aligned(tok, ex, args.max_len)
                                for ex in inj_examples[name][i:i + bs]]
                        asb = collate([b for _, b in ienc],
                                      tok.pad_token_id).to(device)
                    with o.apply(model, scale=objs[name][4]) as handle:
                        if leg == "response":
                            if handle is None or not hasattr(
                                    handle, "set_prompt_lens"):
                                raise RuntimeError(
                                    f"{name}: response-fit object yielded no "
                                    "boundary-capable handle — refusing to "
                                    "measure a masked fit at all positions")
                            handle.set_prompt_lens(
                                steer_prompt_boundaries(asb, H))
                        a_flat = _gather_completion_logits(
                            model(input_ids=asb.input_ids,
                                  attention_mask=asb.attention_mask).logits,
                            asb)
                    acounts = (asb.labels[:, 1:] != -100).sum(1).tolist()
                    act_vals[name].extend(
                        per_example_kl(t_flat, a_flat, acounts))
            if (i // bs) % 40 == 0:
                print(f"[{si}] {len(noop_vals)}/{len(examples)} examples "
                      f"({time.time() - t0:.0f}s)", flush=True)

        for c in si_cells:
            name = c["name"]
            per_fam: dict[str, dict] = {}
            A, N, F = act_vals[name], noop_vals, fams
            keep = [j for j in range(len(N))
                    if math.isfinite(N[j]) and math.isfinite(A[j])]
            by_fam: dict[str, list[int]] = collections.defaultdict(list)
            for j in keep:
                by_fam[F[j]].append(j)
            for f, idxs in sorted(by_fam.items()):
                a = [A[j] for j in idxs]
                d = [N[j] for j in idxs]
                deltas = [A[j] - N[j] for j in idxs]
                ma, mn = statistics.mean(a), statistics.mean(d)
                md, dlo, dhi = ci95(deltas)
                rlo, rhi = boot_ratio_ci(a, d, args.n_boot, args.seed)
                per_fam[f] = {
                    "n": len(idxs), "mean_active": ma, "mean_noop": mn,
                    "ratio": ma / mn if mn > 0 else float("nan"),
                    "ratio_ci95": [rlo, rhi],
                    "mean_delta": md, "delta_ci95": [dlo, dhi],
                    "verdict": ("FIT" if dhi < 0 else
                                "NOT-FIT" if dlo > 0 else "AMBIGUOUS"),
                }
            a_all = [A[j] for j in keep]
            n_all = [N[j] for j in keep]
            md, dlo, dhi = ci95([A[j] - N[j] for j in keep])
            rlo, rhi = boot_ratio_ci(a_all, n_all, args.n_boot, args.seed)
            results[name] = {
                "si": si, "positions": objs[name][1],
                "injects": objs[name][3], "scale": objs[name][4],
                # PER-EXAMPLE values, persisted deliberately: every derived
                # statistic anyone asks for later (a tax CI over a family
                # SUBSET, a reweighting to another family mix, a paired
                # test between two cells) needs the raw paired values, and
                # re-running costs a GPU while re-deriving from these costs
                # nothing. Aggregates cannot be un-aggregated.
                "examples": {
                    "family": [F[j] for j in keep],
                    "active": [A[j] for j in keep],
                    "noop": [N[j] for j in keep],
                },
                "per_family": per_fam,
                "overall": {
                    "n": len(keep),
                    "ratio": (statistics.mean(a_all) / statistics.mean(n_all)
                              if statistics.mean(n_all) > 0 else float("nan")),
                    "ratio_ci95": [rlo, rhi],
                    "mean_delta": md, "delta_ci95": [dlo, dhi],
                    "verdict": ("FIT" if dhi < 0 else
                                "NOT-FIT" if dlo > 0 else "AMBIGUOUS"),
                },
            }
            ov = results[name]["overall"]
            print(f"=== {name}: overall ratio {ov['ratio']:.3f} "
                  f"[{ov['ratio_ci95'][0]:.3f}, {ov['ratio_ci95'][1]:.3f}] "
                  f"{ov['verdict']}, {len(per_fam)} families ===", flush=True)

    print("\n========== PER-(CELL, FAMILY) FIT RATIOS ==========")
    for name, r in results.items():
        ov = r["overall"]
        print(f"\n{name}  [{r['si']}, positions={r['positions']}]  "
              f"overall {ov['ratio']:.3f} {ov['verdict']}")
        print(f"  {'family':<24}{'ratio':>8}{'95% CI':>20}{'n':>6}  verdict")
        for f, fr in sorted(r["per_family"].items(),
                            key=lambda kv: kv[1]["ratio"]):
            print(f"  {f:<24}{fr['ratio']:>8.3f}"
                  f"   [{fr['ratio_ci95'][0]:.3f}, {fr['ratio_ci95'][1]:.3f}]"
                  f"{fr['n']:>6}  {fr['verdict']}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"cells": results, "population": comp,
             "selected_keys": [list(k) for k in order],
             "seed": args.seed, "model": cfg["model"],
             "model_revision": cfg.get("model_revision")}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
