#!/usr/bin/env python
"""ddp_parity_smoke.py — the REAL 2-GPU-vs-1-GPU parity smoke of the data-parallel fit.

``run`` trains ONE arm of the production document-CE trainer (``sisteer.extraction.objects.train_object_ce``) for a
bounded number of optimizer steps on the first rows of a REAL pool (the model, revision, micro-batch, sequence cap and
LoRA architecture read from the fits config), at whatever width the launcher gives it: plain ``python`` = width 1,
``torchrun --standalone --nproc_per_node=N`` = an N-rank group-sharded fit (``ddp.setup_from_env`` reads torchrun's env and
sets the ambient context the trainer shards on). Rank 0 writes ``<out>/result.json`` (per-step loss / tokens / seconds, the
pace, the peak memory) and ``<out>/params.pt`` (the final adapter parameters).

``compare <w1 dir> <wN dir>`` reads two such results — the same arm, pool rows, steps and accumulation — and writes
``<out>/parity.json`` + ``parity.md``. The single-GPU run is the reference; the verdict is PASS only when EVERY gate holds
(a 1 % loss band over 20 steps cannot see a gradient-scaling or reduction error inside a base-dominated CE):

* the per-step LOSS trajectory within ``--loss-rtol`` (default 1e-2, bf16);
* the FINAL ADAPTER PARAMETERS within ``--params-rtol`` of the reference's largest weight (default 5e-3 — bf16 carries
  ~3 significant digits, 2^-8 = 3.9e-3 relative; the CPU fp32 parity test uses atol 1e-6 for the same comparison);
* the per-step TOKEN histories EQUAL (``loss_tokens`` and ``real_tokens`` per optimizer step — the row union: every rank's
  share of a step adds up to exactly the single-process step);
* the per-step LR history EQUAL (the schedule is a pure function of the step count);
* the step-0 GRAD NORM within ``--gradnorm-rtol`` (default 1e-2): the first step's accumulated gradient norm is the direct
  reading of the reduction's scale before any optimizer step can diverge the replicas (a summed-twice or averaged
  reduction reads 2x / 0.5x here, which Adam's scale invariance would otherwise hide from the loss and the params); and
  the MEDIAN per-step grad-norm relative difference within ``--gradnorm-median-rtol`` (default 5e-2; the median, because
  a zero-initialised LoRA B under Adam amplifies bf16 sign flips into one large early outlier);
* the same step count; the steady step time at least ``--min-speedup`` (default 1.6) faster.

Bounded runs never publish (``max_steps`` refuses checkpoints / early stopping): this is a measurement, not a fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (HERE, REPO / "scripts", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from memprobe_fit import (  # noqa: E402
    build_probe_object, enable_gradient_checkpointing, experts_implementation_of, load_config)

TAG = "ddpsmoke"


def _load_rows(pairs: Path, n: int):
    """The first ``n`` rows of the pool, as the trainer's Pair objects (the production loader's fields)."""
    from sisteer.data.pairs import Pair

    rows = []
    with pairs.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(Pair(prompt=r["prompt"], pos=r["pos"], neg=r.get("neg", ""),
                             instruction_id=r.get("instruction_id", ""),
                             requirement_a=r.get("requirement_a"), anti_neg=r.get("anti_neg")))
            if len(rows) >= n:
                break
    if len(rows) < n:
        raise SystemExit(f"[{TAG}] {pairs} holds {len(rows)} rows < the {n} the run needs (steps x accum x micro-batch)")
    return rows


def cmd_run(a) -> int:
    import torch

    from sisteer.extraction import ddp
    from sisteer.extraction.objects import train_object_ce
    from sisteer.methods import docs_to_ce_examples
    from sisteer.models import load_model_and_tokenizer

    cfg = load_config(Path(a.config))
    arm = a.arm
    bs = int(a.micro_batch or (cfg.get("batch_size_by_arm") or {}).get(arm) or 1)
    seq_len = int(cfg.get("max_len", 8192))
    sel = ((cfg.get("train_kwargs") or {}).get(arm) or {}).get("fit_selection") or {}
    lora_rank = int(sel.get("lora_rank", 4)) if arm == "lora_r4_ce" else None
    lora_modules = list(sel.get("lora_modules") or ["o_proj"]) if arm == "lora_r4_ce" else None
    init_scheme = sel.get("init_scheme") or "zero"
    n_rows = a.steps * a.accum * bs

    par = ddp.setup_from_env()
    world = par.world_size if par is not None else 1
    rank = par.rank if par is not None else 0
    is_main = rank == 0
    if a.expected_width and world != a.expected_width:
        raise SystemExit(f"[{TAG}] WORLD_SIZE={world} but --expected-width {a.expected_width}")
    if a.accum % world:
        raise SystemExit(f"[{TAG}] width {world} does not divide accum {a.accum}")
    out = Path(a.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
    print(f"[{TAG}] rank {rank}/{world} arm={arm} micro_batch={bs} accum={a.accum} steps={a.steps} rows={n_rows} "
          f"seq_len={seq_len} lr={a.lr} init={init_scheme} lora_rank={lora_rank} lora_modules={lora_modules}", flush=True)

    t0 = time.perf_counter()
    model, tok = load_model_and_tokenizer(cfg["model"], revision=cfg.get("model_revision"), dtype=cfg.get("dtype", "bfloat16"),
                                          attn_implementation=cfg.get("attn_implementation", "sdpa"))
    load_s = time.perf_counter() - t0
    if a.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    device = next(model.parameters()).device
    rows = _load_rows(Path(a.pairs), n_rows)
    exs = docs_to_ce_examples(tok, rows)
    obj, builder = build_probe_object(model, arm=arm, lora_rank=lora_rank, lora_modules=lora_modules,
                                      init_scheme=init_scheme, seed=a.seed)
    per_step: list[dict] = []

    def log_fn(rec, step=None, **_):
        if f"{TAG}/ce_loss" in rec:
            per_step.append({"step": int(step), **{k.split("/", 1)[1]: float(v) for k, v in rec.items()
                                                    if isinstance(v, (int, float)) and k.startswith(f"{TAG}/")}})

    diag: dict = {}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t1 = time.perf_counter()
    history = train_object_ce(model, tok, obj, exs, lr=a.lr, weight_decay=0.01, epochs=1, batch_size=bs, max_len=seq_len,
                              seed=a.seed, log_fn=log_fn, log_tag=TAG, accum_steps=a.accum, lr_schedule="warmup_cosine",
                              warmup_steps=2, max_grad_norm=1.0, reg_coeff=0.1 if arm == "psr_ce_all" else 0.0,
                              diagnostics=diag, max_steps=a.steps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_s = time.perf_counter() - t1
    params = [p.detach().float().cpu().clone() for p in obj.trainable]
    h = hashlib.sha256()
    for p in params:
        h.update(p.numpy().tobytes())
    steps_done = len(history)
    tokens = sum(r.get("real_tokens", 0.0) for r in per_step)
    rec = {"schema": 1, "kind": "ddp_parity_smoke", "arm": arm, "world_size": world, "fit_width": diag.get("fit_width"),
           "micro_batch": bs, "accum_steps": a.accum, "per_rank_accum": a.accum // world, "effective_batch_rows": bs * a.accum,
           "steps_requested": a.steps, "steps_done": steps_done, "rows": n_rows, "seq_len": seq_len, "lr": a.lr, "seed": a.seed,
           "init_scheme": init_scheme, "lora_rank": lora_rank, "lora_modules": lora_modules, "object_builder": builder,
           "model": {"model": cfg["model"], "revision": cfg.get("model_revision"), "config": str(a.config),
                     "model_class": type(model).__name__, "experts_implementation": experts_implementation_of(model)},
           "pairs": str(a.pairs), "gradient_checkpointing": bool(a.gradient_checkpointing),
           "history": [float(x) for x in history], "per_step": per_step,
           "model_load_s": load_s, "train_s": train_s,
           "s_per_step_mean": (sum(r["sec_per_step"] for r in per_step) / max(len(per_step), 1)) if per_step else None,
           "s_per_step_steady": (sum(r["sec_per_step"] for r in per_step[2:]) / max(len(per_step) - 2, 1)) if len(per_step) > 2 else None,
           "tokens_real_total": tokens, "tokens_per_s_wall": (tokens / train_s) if train_s > 0 else None,
           "peak_alloc_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
           "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
           "torch": torch.__version__, "hostname": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "params_sha256": h.hexdigest(), "trainable_params": int(sum(p.numel() for p in params))}
    if is_main:
        (out / "result.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
        torch.save({"params": params}, out / "params.pt")
        print(f"[{TAG}] rank 0 wrote {out}/result.json: {steps_done} steps, {rec['s_per_step_mean']:.2f} s/step mean "
              f"({rec['s_per_step_steady']} steady), {rec['tokens_per_s_wall']:.0f} tok/s wall, "
              f"peak {rec['peak_alloc_bytes'] / 2**30:.1f} GiB" if rec['peak_alloc_bytes'] else "", flush=True)
    if par is not None:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
    return 0


def cmd_compare(a) -> int:
    import torch

    ref = json.loads((Path(a.ref) / "result.json").read_text())
    wide = json.loads((Path(a.wide) / "result.json").read_text())
    problems = []
    for k in ("arm", "micro_batch", "accum_steps", "steps_done", "rows", "seq_len", "lr", "seed", "init_scheme"):
        if ref.get(k) != wide.get(k):
            problems.append(f"{k}: reference {ref.get(k)!r} vs wide {wide.get(k)!r}")
    if ref["world_size"] != 1:
        problems.append(f"the reference ran at width {ref['world_size']}, not 1")
    if len(ref["history"]) != len(wide["history"]):
        problems.append(f"step count: reference {len(ref['history'])} vs wide {len(wide['history'])}")
    n = min(len(ref["history"]), len(wide["history"]))
    diffs = [abs(r - w) for r, w in zip(ref["history"][:n], wide["history"][:n])]
    rels = [d / max(abs(r), 1e-12) for d, r in zip(diffs, ref["history"][:n])]
    pr = torch.load(Path(a.ref) / "params.pt", weights_only=False)["params"]
    pw = torch.load(Path(a.wide) / "params.pt", weights_only=False)["params"]
    if len(pr) != len(pw) or any(x.shape != y.shape for x, y in zip(pr, pw)):
        problems.append("adapter parameter shapes differ between the two runs")
        pdiff, pscale = float("inf"), 0.0
    else:
        pdiff = max((float((x.double() - y.double()).abs().max()) for x, y in zip(pr, pw)), default=0.0)
        pscale = max((float(x.abs().max()) for x in pr), default=0.0)
    params_rel = (pdiff / pscale) if pscale > 0 else (0.0 if pdiff == 0 else float("inf"))
    params_ok = params_rel <= a.params_rtol
    # the per-step token histories (the row union) and the LR schedule: EQUAL, not close
    def _hist(res, key):
        return [res["per_step"][i].get(key) for i in range(n)] if len(res.get("per_step") or []) >= n else None
    hist_checks = {}
    for key in ("loss_tokens", "real_tokens", "lr"):
        hr, hw = _hist(ref, key), _hist(wide, key)
        if hr is None or hw is None or any(v is None for v in hr + hw):
            hist_checks[key] = {"ok": False, "reason": "history not recorded"}
        else:
            bad = [i for i, (x, y) in enumerate(zip(hr, hw)) if x != y]
            hist_checks[key] = {"ok": not bad, "first_mismatch_step": (bad[0] if bad else None),
                                "n_mismatch": len(bad), "reference": hr, "wide": hw}
    tokens_ok = hist_checks["loss_tokens"]["ok"] and hist_checks["real_tokens"]["ok"]
    lr_ok = hist_checks["lr"]["ok"]
    # the gradient norm: step 0 reads the reduction's scale; the median over steps reads its consistency
    gr, gw = _hist(ref, "grad_norm"), _hist(wide, "grad_norm")
    if gr is None or gw is None or any(v is None for v in gr + gw) or not n:
        gn0_rel, gn_med_rel, gn_ok0, gn_ok_med = None, None, False, False
    else:
        gn_rels = [abs(x - y) / max(abs(x), 1e-12) for x, y in zip(gr, gw)]
        gn0_rel = gn_rels[0]
        gn_med_rel = sorted(gn_rels)[len(gn_rels) // 2]
        gn_ok0 = gn0_rel <= a.gradnorm_rtol
        gn_ok_med = gn_med_rel <= a.gradnorm_median_rtol
    s_ref = ref.get("s_per_step_steady") or ref.get("s_per_step_mean")
    s_wide = wide.get("s_per_step_steady") or wide.get("s_per_step_mean")
    speedup = (s_ref / s_wide) if (s_ref and s_wide) else None
    tok_ratio = (wide["tokens_per_s_wall"] / ref["tokens_per_s_wall"]) if (ref.get("tokens_per_s_wall") and wide.get("tokens_per_s_wall")) else None
    loss_ok = bool(n) and max(rels) <= a.loss_rtol
    speed_ok = speedup is not None and speedup >= a.min_speedup
    gates = {"loss": loss_ok, "params": params_ok, "tokens": tokens_ok, "lr": lr_ok, "gradnorm_step0": gn_ok0,
             "gradnorm_median": gn_ok_med, "speed": speed_ok, "same_run_shape": not problems}
    verdict = "PASS" if all(gates.values()) else "FAIL"
    doc = {"schema": 2, "kind": "ddp_parity", "verdict": verdict, "gates": gates, "problems": problems,
           "widths": {"reference": ref["world_size"], "wide": wide["world_size"]},
           "steps_compared": n, "loss_max_abs_diff": max(diffs) if diffs else None, "loss_max_rel_diff": max(rels) if rels else None,
           "loss_rtol": a.loss_rtol, "loss_ok": loss_ok,
           "params_max_abs_diff": pdiff, "params_max_abs": pscale, "params_rel_diff": params_rel, "params_rtol": a.params_rtol,
           "params_ok": params_ok, "params_sha_equal": ref["params_sha256"] == wide["params_sha256"],
           "token_histories_equal": tokens_ok, "lr_history_equal": lr_ok,
           "histories": {k: {kk: vv for kk, vv in v.items() if kk not in ("reference", "wide")} for k, v in hist_checks.items()},
           "gradnorm_step0_rel_diff": gn0_rel, "gradnorm_rtol": a.gradnorm_rtol, "gradnorm_median_rel_diff": gn_med_rel,
           "gradnorm_median_rtol": a.gradnorm_median_rtol,
           "gradnorm_reference": gr, "gradnorm_wide": gw,
           "s_per_step": {"reference": s_ref, "wide": s_wide}, "step_speedup": speedup, "min_speedup": a.min_speedup, "speed_ok": speed_ok,
           "tokens_per_s_wall": {"reference": ref.get("tokens_per_s_wall"), "wide": wide.get("tokens_per_s_wall")}, "tokens_per_s_ratio": tok_ratio,
           "peak_alloc_gib": {"reference": (ref.get("peak_alloc_bytes") or 0) / 2**30, "wide_rank0": (wide.get("peak_alloc_bytes") or 0) / 2**30},
           "model_load_s": {"reference": ref.get("model_load_s"), "wide_rank0": wide.get("model_load_s")},
           "reference_history": ref["history"][:n], "wide_history": wide["history"][:n],
           "reference": str(a.ref), "wide": str(a.wide)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "parity.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    lines = [f"# DDP parity smoke — width {wide['world_size']} vs 1: **{verdict}**", "",
             f"arm {ref['arm']}, micro-batch {ref['micro_batch']}, accum {ref['accum_steps']} (per rank {wide['per_rank_accum']} at width "
             f"{wide['world_size']}), {n} optimizer steps on {ref['rows']} pool rows at seq_len {ref['seq_len']}, lr {ref['lr']}.", "",
             f"- loss trajectory: max |Δ| = {doc['loss_max_abs_diff']:.3e}, max rel = {doc['loss_max_rel_diff']:.3e} (rtol {a.loss_rtol:g}) -> {'ok' if loss_ok else 'FAIL'}",
             f"- final adapter params: max |Δ| = {pdiff:.3e} (max |p| = {pscale:.3e}, rel {params_rel:.3e}; rtol {a.params_rtol:g}) -> {'ok' if params_ok else 'FAIL'}; sha equal: {doc['params_sha_equal']}",
             f"- per-step token histories (loss_tokens / real_tokens) equal: {'ok' if tokens_ok else 'FAIL'}; lr history equal: {'ok' if lr_ok else 'FAIL'}",
             (f"- grad norm: step 0 rel diff {gn0_rel:.3e} (rtol {a.gradnorm_rtol:g}) -> {'ok' if gn_ok0 else 'FAIL'}; median rel diff "
              f"{gn_med_rel:.3e} (rtol {a.gradnorm_median_rtol:g}) -> {'ok' if gn_ok_med else 'FAIL'}") if gn0_rel is not None
             else "- grad norm: history not recorded -> FAIL",
             f"- step time (steady, after 2 warm steps): {s_ref:.2f} s -> {s_wide:.2f} s = {speedup:.2f}x (min {a.min_speedup:g}) -> {'ok' if speed_ok else 'FAIL'}"
             if speedup else "- step time: unavailable",
             f"- tokens/s over the training wall: {ref.get('tokens_per_s_wall', 0):.0f} -> {wide.get('tokens_per_s_wall', 0):.0f} ({tok_ratio:.2f}x)" if tok_ratio else "",
             f"- peak allocated: {doc['peak_alloc_gib']['reference']:.1f} GiB (1 GPU) vs {doc['peak_alloc_gib']['wide_rank0']:.1f} GiB (rank 0 of {wide['world_size']})",
             f"- model load: {ref.get('model_load_s', 0):.0f} s vs {wide.get('model_load_s', 0):.0f} s (rank 0)", ""]
    if problems:
        lines += ["problems:"] + [f"- {p}" for p in problems] + [""]
    lines += ["| step | loss (1 GPU) | loss (wide) | abs diff | grad norm (1 GPU) | grad norm (wide) | loss tokens | lr |",
              "|---|---|---|---|---|---|---|---|"]
    for i, (r, w) in enumerate(zip(ref["history"][:n], wide["history"][:n])):
        g1 = (gr[i] if gr else None); g2 = (gw[i] if gw else None)
        tk = hist_checks["loss_tokens"].get("reference", [None] * n)[i] if hist_checks["loss_tokens"].get("reference") else None
        lr_i = hist_checks["lr"].get("reference", [None] * n)[i] if hist_checks["lr"].get("reference") else None
        lines.append(f"| {i} | {r:.6f} | {w:.6f} | {abs(r - w):.2e} | {g1 if g1 is None else f'{g1:.5f}'} | {g2 if g2 is None else f'{g2:.5f}'} | {tk} | {lr_i} |")
    (out / "parity.md").write_text("\n".join(lines) + "\n")
    print(f"[{TAG}] {verdict}: gates {gates}; loss max rel {doc['loss_max_rel_diff']}, params rel {params_rel:.3e}, "
          f"grad-norm step0 rel {gn0_rel}, speedup {speedup} -> {out}/parity.md", flush=True)
    return 0 if verdict == "PASS" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True); r.add_argument("--arm", required=True, choices=("psr_ce_all", "lora_r4_ce"))
    r.add_argument("--pairs", required=True); r.add_argument("--out", required=True)
    r.add_argument("--steps", type=int, default=20); r.add_argument("--accum", type=int, default=4)
    r.add_argument("--micro-batch", type=int, default=None, help="default: the config's batch_size_by_arm[arm]")
    r.add_argument("--lr", type=float, default=1e-5); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--expected-width", type=int, default=0, help="refuse unless WORLD_SIZE equals this (0 = any)")
    r.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    c = sub.add_parser("compare")
    c.add_argument("ref"); c.add_argument("wide"); c.add_argument("--out", required=True)
    c.add_argument("--loss-rtol", type=float, default=1e-2); c.add_argument("--min-speedup", type=float, default=1.6)
    c.add_argument("--params-rtol", type=float, default=5e-3, help="final adapter params: max |Δ| / max |p_ref| (bf16: ~2^-8)")
    c.add_argument("--gradnorm-rtol", type=float, default=1e-2, help="step-0 accumulated grad norm, relative")
    c.add_argument("--gradnorm-median-rtol", type=float, default=5e-2, help="median per-step grad-norm relative difference")
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
