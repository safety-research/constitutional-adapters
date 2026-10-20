#!/usr/bin/env python
"""Prove the fused Mamba kernels are USED on a Mamba-hybrid target -- not
merely importable -- and measure the campaign capture rate against the
pure-torch fallback.

    python scripts/mamba_kernels_check.py --config configs/alphasteer_f22_nemotron3nano4b.yaml \
        --rows 128 [--batch-size 8] [--fallback] [--json OUT]

Three readings, in order of authority:
  1. the RESOLVED implementation module of every kernel wrapper transformers'
     modeling module defines (sisteer.models.resolved_mamba_kernels -- the
     live closures, bound at import time);
  2. the CUDA kernel names one capture batch launches (torch.profiler): the
     fused path shows mamba_ssm's triton SSD kernels (chunk_scan / chunk_state
     / state_passing / bmm_chunk / chunk_cumsum) and causal_conv1d's kernel;
     the fallback shows none of them;
  3. rows/s through the campaign's OWN capture seam
     (alphasteer_screen.capture_all_layers) over the first --rows prompts of
     the config's benign list, after one warm-up batch (triton JIT).

--fallback blocks the mamba_ssm / causal_conv1d imports BEFORE transformers
loads, so the identical forward runs on the pure-torch path -- job 470950's
path -- for the baseline number. It loads through transformers directly,
because the repo loader REFUSES that path by design
(models.assert_fused_mamba_kernels).

Exit 0 in --fallback mode (a baseline is a measurement). In fused mode, exit 1
unless every wrapper resolved to its package AND a fused kernel name appeared
in the profile -- installed-but-unused is the failure this exists to catch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

FUSED_KERNEL_MARKS = ("chunk_scan", "chunk_state", "chunk_cumsum",
                      "state_passing", "bmm_chunk", "causal_conv1d")


def _block_fused_imports() -> None:
    # importlib.import_module(name) raises ImportError when sys.modules[name]
    # is None -- exactly the path use_kernel_func_from_hub_with_fallback
    # takes to its torch fallback. Must run before transformers imports.
    for name in ("mamba_ssm", "causal_conv1d"):
        sys.modules[name] = None


def _load(cfg: dict, *, fallback: bool):
    import torch

    from sisteer.models import (load_tokenizer, auto_model_class,
                                pin_kv_cache_off_if_unallocatable)
    import fit_class_b as fcb

    if not fallback:
        return fcb.load_class_b_model(cfg)            # the drivers' load, guard included
    name, rev = cfg["model"], cfg.get("model_revision")
    tok = load_tokenizer(name, revision=rev)
    cls = auto_model_class(name, revision=rev)
    model = cls.from_pretrained(name, revision=rev, dtype=getattr(torch, cfg["dtype"]),
                                attn_implementation=cfg.get("attn_implementation", "sdpa"))
    model.to("cuda").eval()
    pin_kv_cache_off_if_unallocatable(model)
    return model, tok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="default: the config's class_b.alphasteer capture width (the rule's)")
    ap.add_argument("--fallback", action="store_true",
                    help="block mamba_ssm/causal_conv1d so the pure-torch path runs (baseline)")
    ap.add_argument("--json", default=None, help="write the summary dict here too")
    args = ap.parse_args(argv)
    if args.fallback:
        _block_fused_imports()

    import torch
    import yaml

    import fit_class_b as fcb
    from sisteer.extraction import alphasteer_screen as scr
    from sisteer.models import resolved_mamba_kernels, MAMBA_KERNEL_WRAPPERS

    cfg = yaml.safe_load(Path(args.config).read_text())
    mode = "fallback" if args.fallback else "fused"
    model, tok = _load(cfg, fallback=args.fallback)
    kern = resolved_mamba_kernels(model)
    print(f"[mamba_check] mode={mode} model={cfg['model']} kernels resolved:")
    if kern is None:
        print("[mamba_check]   (this model instantiates no Mamba mixer -- no module class of "
              "it calls a mamba_ssm wrapper; nothing to check)")
        return 0
    for n, m in kern.items():
        print(f"[mamba_check]   {n:36s} -> {m}")
    all_fused = all(m == MAMBA_KERNEL_WRAPPERS[n] or m.startswith(MAMBA_KERNEL_WRAPPERS[n] + ".")
                    for n, m in kern.items())

    rule = fcb.alphasteer_layer_rule(cfg)
    bs = int(args.batch_size or rule["capture_batch_size"])
    site = rule["site"]
    data = fcb.build_class_b_data(cfg, tok)
    prompts = data["benign_prompts"][:args.rows]
    texts = [scr.build_chat_prompt(tok, p, system=None) for p in prompts]
    print(f"[mamba_check] {len(texts)} rows (first of the benign list), batch {bs}, site={site}")

    # warm-up: the first fused call JIT-compiles the triton kernels
    t0 = time.time()
    scr.capture_all_layers(model, tok, texts[:bs], site=site, batch_size=bs)
    torch.cuda.synchronize()
    print(f"[mamba_check] warm-up batch {time.time() - t0:.1f}s")

    # profile ONE batch: which CUDA kernels actually launch
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        scr.capture_all_layers(model, tok, texts[:bs], site=site, batch_size=bs)
        torch.cuda.synchronize()
    agg: dict[str, float] = {}
    for e in prof.key_averages():              # CUDA-only activity: these are the kernels
        t = float(getattr(e, "self_device_time_total", 0.0)
                  or getattr(e, "self_cuda_time_total", 0.0) or 0.0)
        if t > 0:
            agg[e.key] = agg.get(e.key, 0.0) + t
    top = sorted(agg.items(), key=lambda kv: -kv[1])[:12]
    fused_seen = sorted({k for k in agg if any(mark in k.lower() for mark in FUSED_KERNEL_MARKS)})
    print("[mamba_check] top CUDA kernels of one batch (us):")
    for k, us in top:
        print(f"[mamba_check]   {us:12.0f}  {k[:110]}")
    print(f"[mamba_check] fused-kernel names seen: {len(fused_seen)} "
          f"{[k[:60] for k in fused_seen[:8]]}")

    # the rate, through the campaign's capture seam
    torch.cuda.synchronize()
    t0 = time.time()
    acts, info = scr.capture_all_layers(model, tok, texts, site=site, batch_size=bs)
    torch.cuda.synchronize()
    secs = time.time() - t0
    n_layers = len(acts)
    rate = len(texts) / secs
    print(f"[mamba_check] capture {len(texts)} rows x {n_layers} layers in {secs:.1f}s "
          f"= {rate:.1f} rows/s ({mode})")

    summary = {"mode": mode, "model": cfg["model"], "kernels": kern, "all_fused": all_fused,
               "fused_kernel_names_seen": fused_seen, "top_kernels": top,
               "rows": len(texts), "batch_size": bs, "site": site, "n_layers": n_layers,
               "seconds": secs, "rows_per_s": rate,
               "projected_14000_rows_s": 14000 / rate}
    print("[mamba_check] SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "top_kernels"}))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1))
    if args.fallback:
        return 0
    if not all_fused:
        print("[mamba_check] FAIL: a wrapper resolved to the pure-torch fallback", file=sys.stderr)
        return 1
    if not fused_seen:
        print("[mamba_check] FAIL: no fused kernel launched in the profiled batch "
              "(resolved to the package, yet the forward did not run it)", file=sys.stderr)
        return 1
    print("[mamba_check] OK: fused kernels resolved AND launched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
