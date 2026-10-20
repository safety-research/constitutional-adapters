"""Real-model verification of the offline top-K teacher (task VERIFY, 3a-3c).

The CPU tests (tests/test_teacher_topk.py) pin the MECHANISM — gather
parity, binding, the K=|V| loss identity. This script measures what only a
real model can answer, because the HPO tuned every LR on the FULL-VOCAB
online-teacher loss and the top-50 loss may not silently replace it:

  (a) determinism + exactness: rebuild-vs-rebuild bitwise equality; the
      canonical bs=1 store vs the fit's BATCHED online teacher (top-50
      index agreement + logit deltas — the accepted fp class, measured);
  (b) THE NUMBER THE SHIP DECISION RIDES ON: full-vocab forward KL vs the
      renormalized top-50 (and top-100) KL, teacher vs the unprompted
      student, per training batch over a sha-ranked pool subset, at T=1
      (the production arms) and T=10 (token_faithful's regime, expected
      to collapse — measured, not asserted), plus the head-mass tail
      that explains the gap;
  (c) a short REAL fit (train_lockstep, lora_r1_all) online vs
      topk-consuming: per-step loss deltas + trained-object deltas;
  (d) accumulation parity at bf16 production scale: accumulated
      micro-batches vs one true batch at the same effective width, under
      an ACTIVE clip applied once per optimizer step on the accumulated
      global norm (the exact-math half is pinned on CPU fp32 in
      tests/test_accum.py; this measures the bf16 kernel-drift class).

Writes a JSON report per SI and prints a human summary. Run on one GPU:

  python scripts/verify_teacher_topk.py --config configs/two_si_llama_fits.yaml \
      --pairs <pool>/pairs_<si>.jsonl --si safety_refusal --n 64 \
      --out /path/verify_<si>.json [--fit-steps 12]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def _mean_max(vals):
    return {"mean": float(sum(vals) / max(len(vals), 1)),
            "max": float(max(vals)) if vals else None,
            "n": len(vals)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--si", required=True)
    ap.add_argument("--n", type=int, default=64,
                    help="sha-ranked pool subset size for (a)/(b)")
    ap.add_argument("--fit-n", type=int, default=24,
                    help="subset size for the short-fit check (c)")
    ap.add_argument("--fit-steps", type=int, default=12,
                    help="~optimizer steps for (c) (epochs=1 at bs=2)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-fit", action="store_true")
    args = ap.parse_args()

    import yaml

    from sisteer.data.pairs import pairs_to_distill_examples
    from sisteer.eval.jailbreak import resolve_instruction
    from sisteer.eval.train_budget import resolve_max_len
    from sisteer.extraction.distill import (
        _gather_completion_logits, soft_divergence, soft_divergence_topk)
    from sisteer.extraction.fitmon import pin_attention_backends
    from sisteer.extraction.lockstep import (
        LockstepArm, _encode_arm_batches, build_lockstep_object,
        default_lockstep_params, train_lockstep)
    from sisteer.extraction.subsample import sha_rank_indices
    from sisteer.extraction.teacher_topk import (
        compute_teacher_topk, head_mass_stats)
    from sisteer.models import load_model_and_tokenizer
    import train_two_si_objects as t2

    cfg = yaml.safe_load(open(args.config))
    max_len = resolve_max_len(cfg, args.si)
    instruction = resolve_instruction(args.si)
    pairs = t2._load_pairs(args.pairs)
    keys = [p.prompt for p in pairs]
    sub = [pairs[i] for i in sha_rank_indices(keys, args.n)]
    print(f"[verify] {args.si}: {len(sub)}/{len(pairs)} sha-ranked pairs, "
          f"max_len={max_len}", flush=True)

    pin_attention_backends()
    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    device = next(model.parameters()).device
    examples = pairs_to_distill_examples(tok, sub, instruction)

    report: dict = {"si": args.si, "model": cfg["model"],
                    "model_revision": cfg.get("model_revision"),
                    "dtype": cfg.get("dtype"), "max_len": max_len,
                    "n_pairs": len(sub), "batch_size": args.batch_size,
                    "gpu": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else "cpu")}

    # ---- build the store at K=200 (the production width); its first
    # 50/100 columns ARE the top-50/100 (torch.topk returns values sorted
    # descending), so one build serves every reported width.
    t0 = time.time()
    store100 = compute_teacher_topk(model, tok, examples, max_len=max_len,
                                    top_k=200, progress=True)
    report["build_seconds_n"] = round(time.time() - t0, 1)
    # Head mass at every audited grid temperature (T=1 the fidelity
    # question; T=2/4 the re-tune's swept objectives, audited not gated).
    for t in (1.0, 2.0, 4.0):
        col = store100._t_col(t)
        for k in (50, 100, 200):
            zk = torch.exp(store100.val[:, :k].float() / t
                           - store100.lse_t[:, col].unsqueeze(1)).sum(1)
            report[f"head_mass_T{t:g}_top{k}"] = {
                "mean": float(zk.mean()), "min": float(zk.min()),
                "p1": float(torch.quantile(zk, 0.01)),
                "p50": float(torch.quantile(zk, 0.5))}

    # ---- (a) determinism: rebuild a slice; on any inequality, CLASSIFY
    # it — torch.topk's CUDA selection is not order-stable on TIES
    # (bitwise-equal bf16 logits), which permutes tied boundary entries
    # run-to-run without changing a single stored VALUE; that class is
    # loss-neutral up to which of two equal-logit tokens the head names.
    # Anything beyond it (val/lse drift) would be real forward
    # nondeterminism and must be visible here.
    again = compute_teacher_topk(model, tok, examples[:8], max_len=max_len,
                                 top_k=200, progress=False)
    hi = int(store100.offsets[8])
    idx_eq = bool(torch.equal(again.idx, store100.idx[:hi]))
    val_eq = bool(torch.equal(again.val, store100.val[:hi]))
    lse_eq = bool(torch.equal(again.lse_t, store100.lse_t[:hi]))
    rebuild = {"bitwise": idx_eq and val_eq and lse_eq,
               "idx_equal": idx_eq, "val_equal": val_eq,
               "lse_equal": lse_eq}
    if not rebuild["bitwise"]:
        m = again.idx != store100.idx[:hi]
        rebuild["n_positions"] = int(hi)
        rebuild["idx_diff_entries"] = int(m.sum())
        rebuild["positions_with_idx_diff"] = int(m.any(dim=1).sum())
        rebuild["val_max_abs_delta"] = float(
            (again.val.float() - store100.val[:hi].float()).abs().max())
        rebuild["val_delta_at_idx_diff"] = (
            float((again.val[m].float()
                   - store100.val[:hi][m].float()).abs().max())
            if bool(m.any()) else 0.0)
        rebuild["lse_max_abs_delta"] = float(
            (again.lse_t - store100.lse_t[:hi]).abs().max())
    report["a_rebuild"] = rebuild
    print(f"[verify] (a) rebuild: {rebuild}", flush=True)

    # ---- (a') the batched online teacher (the fit's alternative): index
    # agreement + value deltas at top-50 — the accepted fp class, measured.
    class _Id:
        def inject(self, s):
            return s

    t_batches, s_batches = _encode_arm_batches(
        tok, _Id(), examples, max_len=max_len, batch_size=args.batch_size)
    agree, val_dm = [], []
    # Per-batch online gathered teacher logits, reused for (b) — parked on
    # CPU: at 16384 x 128k a GPU-resident list is tens of GB and would
    # crowd out the (c) fit.
    b_rows = []
    with torch.no_grad():
        for tb in t_batches:
            tb_d = tb.to(device)
            logits = model(input_ids=tb_d.input_ids,
                           attention_mask=tb_d.attention_mask).logits
            g = _gather_completion_logits(logits, tb_d)
            b_rows.append(g.cpu())
            bval, bidx = torch.topk(g, 50, dim=-1)
            i, v = store100.gather_for_batch(tb)
            i50, v50 = i[:, :50].to(device), v[:, :50].to(device)
            same = (bidx.to(torch.int32) == i50).float().mean()
            agree.append(float(same))
            val_dm.append(float((bval.float() - v50.float()).abs().max()))
    report["a_batched_vs_store_top50"] = {
        "index_agreement_mean": float(sum(agree) / len(agree)),
        "index_agreement_min": float(min(agree)),
        "max_abs_logit_delta": float(max(val_dm))}
    print(f"[verify] (a') batched-vs-store top-50: index agreement "
          f"mean={report['a_batched_vs_store_top50']['index_agreement_mean']:.6f} "
          f"min={min(agree):.6f}, max|dlogit|={max(val_dm):.4f}", flush=True)

    # ---- (b) the loss gap: full-vocab KL vs top-50/top-100, teacher vs
    # the UNPROMPTED student (the step-0 / noop-floor regime the HPO's
    # losses lived in), per training batch.
    def batch_losses(temperature: float) -> dict:
        rows = {"full": [], "top50_store": [], "top100_store": [],
                "top200_store": [], "top50_online": []}
        with torch.no_grad():
            for bi, (tb, sb) in enumerate(zip(t_batches, s_batches)):
                sb_d = sb.to(device)
                s_logits = _gather_completion_logits(
                    model(input_ids=sb_d.input_ids,
                          attention_mask=sb_d.attention_mask).logits, sb_d)
                t_full = b_rows[bi].to(device)
                t_logp = torch.log_softmax(t_full.float() / temperature, -1)
                s_logp = torch.log_softmax(s_logits.float() / temperature,
                                           -1)
                rows["full"].append(float(soft_divergence(
                    "soft", t_logp, s_logp)))
                i, v = store100.gather_for_batch(t_batches[bi])
                for name, k in (("top50_store", 50), ("top100_store", 100),
                                ("top200_store", 200)):
                    rows[name].append(float(soft_divergence_topk(
                        i[:, :k].to(device), v[:, :k].to(device), s_logits,
                        temperature=temperature)))
                # Truncation error ISOLATED from bs1-vs-batched fp drift:
                # the head taken from the SAME batched forward.
                bval, bidx = torch.topk(t_full, 50, dim=-1)
                rows["top50_online"].append(float(soft_divergence_topk(
                    bidx.to(torch.int32), bval, s_logits,
                    temperature=temperature)))
        full = rows["full"]
        out = {"full_kl": _mean_max(full)}
        for name in ("top50_store", "top100_store", "top200_store",
                     "top50_online"):
            d = [abs(a - b) for a, b in zip(rows[name], full)]
            r = [dd / max(abs(f), 1e-12) for dd, f in zip(d, full)]
            out[name] = {"loss": _mean_max(rows[name]),
                         "abs_delta": _mean_max(d),
                         "rel_delta": _mean_max(r)}
        return out

    report["b_T1"] = batch_losses(1.0)
    report["b_T2"] = batch_losses(2.0)
    report["b_T4"] = batch_losses(4.0)
    report["b_T10_token_faithful_regime"] = batch_losses(10.0)
    b1 = report["b_T1"]
    print(f"[verify] (b) T=1: full KL mean={b1['full_kl']['mean']:.6f}; "
          f"top50 rel delta mean={b1['top50_store']['rel_delta']['mean']:.3e} "
          f"max={b1['top50_store']['rel_delta']['max']:.3e}; "
          f"top100 rel delta mean="
          f"{b1['top100_store']['rel_delta']['mean']:.3e}; "
          f"top200 rel delta mean="
          f"{b1['top200_store']['rel_delta']['mean']:.3e} "
          f"max={b1['top200_store']['rel_delta']['max']:.3e}", flush=True)
    for tname in ("b_T2", "b_T4"):
        bt = report[tname]
        print(f"[verify] (b) {tname}: full KL mean="
              f"{bt['full_kl']['mean']:.6f}; top50 rel delta mean="
              f"{bt['top50_store']['rel_delta']['mean']:.3e} "
              f"(audited swept-T regime: the head IS the tuned objective; "
              "reported for the record)", flush=True)
    b10 = report["b_T10_token_faithful_regime"]
    print(f"[verify] (b) T=10: full KL mean={b10['full_kl']['mean']:.6f}; "
          f"top50 rel delta mean={b10['top50_store']['rel_delta']['mean']:.3e}"
          " (token_faithful's regime — expected to be material)", flush=True)

    # ---- (c) short fit parity: online vs topk-consuming lockstep fit.
    del b_rows  # free the retained teacher logits before the fit
    if not args.skip_fit:
        fit_pairs = [pairs[i] for i in sha_rank_indices(keys, args.fit_n)]
        fit_examples = pairs_to_distill_examples(tok, fit_pairs, instruction)
        store50 = compute_teacher_topk(model, tok, fit_examples,
                                       max_len=max_len, top_k=200,
                                       progress=False)

        from sisteer.hooks import response_header_len as _derive_h

        rhl = _derive_h(tok) or None  # production's header-steered leg

        def short_fit(topk):
            obj = build_lockstep_object("lora_r1_all", model, tok,
                                        instruction, fit_pairs, layer=int(
                                            cfg.get("layer", 15)),
                                        positions="response",
                                        response_header_len=rhl)
            params = default_lockstep_params("lora_r1_all")
            hist = train_lockstep(
                model, tok, [LockstepArm("lora_r1_all", obj, params)],
                fit_examples, epochs=1, batch_size=args.batch_size,
                max_len=max_len, seed=0, n_probe=8, teacher_topk=topk)
            return hist["lora_r1_all"], [p.detach().float().cpu()
                                         for p in obj.trainable]

        t0 = time.time()
        h_on, p_on = short_fit(None)
        h_tk, p_tk = short_fit(store50)
        d_loss = [abs(a - b) for a, b in zip(h_on, h_tk)]
        r_loss = [d / max(abs(a), 1e-12) for d, a in zip(d_loss, h_on)]
        p_delta = [float((a - b).abs().max()) for a, b in zip(p_on, p_tk)]
        p_scale = [float(a.abs().max()) for a in p_on]
        report["c_short_fit"] = {
            "arm": "lora_r1_all", "steps": len(h_on),
            "seconds": round(time.time() - t0, 1),
            "loss_abs_delta": _mean_max(d_loss),
            "loss_rel_delta": _mean_max(r_loss),
            "first_last_loss_online": [h_on[0], h_on[-1]],
            "first_last_loss_topk": [h_tk[0], h_tk[-1]],
            "param_max_abs_delta": max(p_delta),
            "param_max_abs_value": max(p_scale),
            "param_rel_delta": max(p_delta) / max(max(p_scale), 1e-12),
        }
        c = report["c_short_fit"]
        print(f"[verify] (c) {c['steps']}-step fit: loss rel delta "
              f"mean={c['loss_rel_delta']['mean']:.3e} "
              f"max={c['loss_rel_delta']['max']:.3e}; param rel delta "
              f"{c['param_rel_delta']:.3e}", flush=True)

    # ---- (d) accumulation parity ON THE REAL MODEL (bf16): accumulated
    # micro-batches vs one true batch at the same effective width, under an
    # ACTIVE clip applied ONCE per optimizer step. The exact-math version
    # is pinned on CPU fp32 (tests/test_accum.py, ragged groups); this leg
    # measures the bf16/batched-kernel drift class at production scale and
    # proves the clip fired on the ACCUMULATED global norm.
    if not args.skip_fit:
        from sisteer.hooks import response_header_len as _dh

        rhl_d = _dh(tok) or None
        # Production's header-steered response leg where the tokenizer
        # derives a header (real chat models); all-token otherwise (a
        # template-less test tokenizer's H=0 would be a plain decode-only
        # leg, whose keep-subselect the accum weights deliberately refuse).
        pos_d = "response" if rhl_d else "all"
        acc_pairs = [pairs[i] for i in sha_rank_indices(keys, 4)]
        acc_ex = pairs_to_distill_examples(tok, acc_pairs, instruction)

        def accum_fit(batch_size, accum, clip):
            obj = build_lockstep_object(
                "lora_r1_all", model, tok, instruction, acc_pairs,
                layer=int(cfg.get("layer", 15)), positions=pos_d,
                response_header_len=rhl_d)
            params = replace(default_lockstep_params("lora_r1_all"),
                             accum_steps=accum, max_grad_norm=clip)
            grabbed: dict = {}

            def cb(step, name, o, loss):
                if "grads" not in grabbed:
                    grabbed["grads"] = [p.grad.detach().float().cpu().clone()
                                        for p in o.trainable]

            hist = train_lockstep(
                model, tok, [LockstepArm("lora_r1_all", obj, params)],
                acc_ex, epochs=3, batch_size=batch_size, max_len=max_len,
                seed=0, n_probe=4, probe_cb=cb)
            return (hist["lora_r1_all"],
                    [p.detach().float().cpu() for p in obj.trainable],
                    grabbed["grads"])

        def gnorm(grads):
            return float(torch.norm(torch.stack([g.norm() for g in grads])))

        # Unclipped accumulated norm first, so the parity runs clip BELOW
        # it — an inactive clip cannot catch per-micro-batch clipping.
        _h, _p, g_raw = accum_fit(2, 2, None)
        n0 = gnorm(g_raw)
        clip = n0 / 2.0
        h_big, p_big, g_big = accum_fit(4, 1, clip)
        h_acc, p_acc, g_acc = accum_fit(2, 2, clip)
        d_loss = [abs(a - b) for a, b in zip(h_big, h_acc)]
        r_loss = [d / max(abs(a), 1e-12) for d, a in zip(d_loss, h_big)]
        gd = [float((a - b).abs().max()) for a, b in zip(g_big, g_acc)]
        gs = [float(a.abs().max()) for a in g_big]
        pd = [float((a - b).abs().max()) for a, b in zip(p_big, p_acc)]
        ps = [float(a.abs().max()) for a in p_big]
        report["d_accum_parity"] = {
            "arm": "lora_r1_all", "effective_batch": 4,
            "sides": "bs4 x accum1 vs bs2 x accum2",
            "optimizer_steps": len(h_big),
            "unclipped_grad_norm": n0, "clip": clip,
            "postclip_norm_big": gnorm(g_big),
            "postclip_norm_accum": gnorm(g_acc),
            "clip_active": bool(gnorm(g_acc) < 0.75 * n0),
            "step1_grad_max_abs_delta": max(gd),
            "step1_grad_rel_delta": max(gd) / max(max(gs), 1e-12),
            "loss_abs_delta": _mean_max(d_loss),
            "loss_rel_delta": _mean_max(r_loss),
            "param_max_abs_delta": max(pd),
            "param_rel_delta": max(pd) / max(max(ps), 1e-12),
        }
        d = report["d_accum_parity"]
        print(f"[verify] (d) accum parity (eff 4, clip ACTIVE at "
              f"{clip:.4g} vs raw {n0:.4g}): step-1 grad rel delta "
              f"{d['step1_grad_rel_delta']:.3e}; loss rel delta "
              f"mean={d['loss_rel_delta']['mean']:.3e} "
              f"max={d['loss_rel_delta']['max']:.3e}; param rel delta "
              f"{d['param_rel_delta']:.3e}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"[verify] report -> {out}", flush=True)


if __name__ == "__main__":
    main()
