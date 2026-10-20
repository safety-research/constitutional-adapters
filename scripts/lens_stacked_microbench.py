#!/usr/bin/env python
"""lens_stacked_microbench — the lens hook's per-layer apply cost and the
stacked-vs-per-row deviation, per op kind, at B in {8, 32, 64, 128, 256},
on one GPU, without a model (branch lens-grouped-scope, 2026-09-04).

WHAT IT MEASURES. ``lens_adapters._row_reduction`` issues every reduction
inside a vectorized decode group either as ONE stacked ``[B, K]`` call (the
``stacked`` mode — the served default since the 2026-09-04 decision) or PER
ROW (``SISTEER_LENS_STACKED_REDUCTIONS=0`` — bit-identical to the per-request
call by construction, the reference). Job array 382416 found the two not
bit-identical at B > 1; this bench measured how much faster and how far
apart (docs/lens_stacked_reductions_scoping.md). It times the three arms of
ONE hooked layer's apply on synthetic bf16 activations of each served
family's shape —

  legacy   B separate single-row applies (the pre-vectorize per-slot loop's
           arithmetic; its per-slot Python bookkeeping is NOT included, so
           this is a lower bound on the legacy path's real cost),
  per_row  one grouped apply, reductions issued per row (the opt-out),
  stacked  one grouped apply, one stacked reduction (the default),

— and, on the SAME inputs, the elementwise deviation of ``stacked`` from
``per_row``: how many elements differ, the largest |diff|, and the
difference in ulps of the per-row output's own dtype at that element's
magnitude (``lens_adapters.ulp_distance``; 1.0 = adjacent representable
values), binned <= 1 / <= 2 / > 2. For the psr gate it also reports the
fp32 gate scalar's own deviation (the reduction that differs), in fp32 ulps.
Every op kind the worker vectorizes in decode is covered: the psr ReLU-gated
gemv, the lowrank_gated block low-rank pair (rank 4), the LoReFT affine
subspace (rank 4), the elementwise add_vector control (expected 0
deviation), and the o_proj-site LoRA GEMM pair at rank 1 and rank 4 for
every o_proj shape the family carries.

The mode is flipped PROCESS-LOCALLY here (``la._STACKED_REDUCTIONS``) —
exactly the attribute the env var sets at import — so both arms run in one
process on identical tensors, and restored to what it was after every
apply. Activation profiles: ``gaussian`` (unit-RMS
normal), ``heavy`` (the same plus 8 sink-like dims at 64x — the massive-
activation shape of real residual streams; ulps are scale-free, so this
only moves the absolute numbers) and ``ints`` (integer-valued fp32, exact
arithmetic: the CPU test's zero-deviation control).

USAGE (one GPU; a few minutes):
  python scripts/lens_stacked_microbench.py --device cuda --out <dir>/microbench.json \
      [--figs <dir>] [--models llama8b,gemma4e4b,...] [--batches 8,32,64,128,256]
Shapes come from the pinned snapshots' configs (SHAPES below, with the
per-layer o_proj widths the gate's synthetic mint reads off the module
tree): hidden size, decoder layers, o_proj (in, out, n_layers).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from sisteer.capeval.serving import lens_adapters as la  # noqa: E402

#: hidden size / decoder layers / o_proj (in_features, out_features, layers
#: carrying it) per served family — the pinned snapshots' text configs and
#: the module-tree o_proj widths (gate: oproj_shapes_from_skeleton). gemma-4
#: E4B: 35 sliding layers at 8 x 256 = 2048 and 7 global at 8 x 512 = 4096;
#: Qwen3.5-4B: o_proj on its 8 full-attention layers only (16 x 256 = 4096);
#: Nemotron-3-Nano-4B: 4 attention blocks of 42 (40 x 128 = 5120).
SHAPES: Dict[str, Dict[str, Any]] = {
    "llama8b": {"hidden": 4096, "n_layers": 32, "oproj": {"attn": (4096, 4096, 32)}},
    "gemma4e4b": {"hidden": 2560, "n_layers": 42,
                  "oproj": {"sliding": (2048, 2560, 35), "global": (4096, 2560, 7)}},
    "qwen35_4b": {"hidden": 2560, "n_layers": 32, "oproj": {"full_attn": (4096, 2560, 8)}},
    "glm46vflash": {"hidden": 4096, "n_layers": 40, "oproj": {"attn": (4096, 4096, 40)}},
    "nemotron3nano4b": {"hidden": 3136, "n_layers": 42, "oproj": {"attn": (5120, 3136, 4)}},
    #: CPU test shapes only
    "tiny": {"hidden": 16, "n_layers": 2, "oproj": {"attn": (12, 16, 2)}},
}
MODEL_ORDER = ("llama8b", "gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b")
#: (kind, site): the block-site kinds carry the residual width; the o_proj
#: kinds are instantiated once per o_proj shape of the family
KINDS: Tuple[Tuple[str, str], ...] = (
    ("psr_gemv", la.HOOK_SITE_BLOCK), ("lowrank_r4", la.HOOK_SITE_BLOCK),
    ("loreft_r4", la.HOOK_SITE_BLOCK), ("add_vector", la.HOOK_SITE_BLOCK),
    ("lora_oproj_r1", la.HOOK_SITE_OPROJ), ("lora_oproj_r4", la.HOOK_SITE_OPROJ))
ARMS = ("legacy", "per_row", "stacked")
DEFAULT_BATCHES = (8, 32, 64, 128, 256)
PROFILES = ("gaussian", "heavy", "ints")
#: kinds whose only work is elementwise — expected bit-identical at any B
ELEMENTWISE_KINDS = ("add_vector",)


def _gen(seed: int, device: str) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def make_rows(profile: str, B: int, d: int, *, device: str, dtype: torch.dtype,
              seed: int) -> torch.Tensor:
    """``[B, d]`` activations of one profile (module docstring)."""
    g = _gen(seed, device)
    if profile == "ints":
        return torch.randint(-3, 4, (B, d), generator=g).to(torch.float32).to(device)
    x = torch.randn(B, d, generator=g)
    if profile == "heavy":
        dims = torch.randperm(d, generator=g)[: min(8, d)]
        x[:, dims] *= 64.0
    elif profile != "gaussian":
        raise ValueError(f"unknown profile {profile!r} (one of {PROFILES})")
    return x.to(dtype).to(device)


def make_ops(kind: str, *, d: int, d_in: int, d_out: int, device: str,
             dtype: torch.dtype, seed: int) -> Tuple[la.HookOp, ...]:
    """One materialized op of ``kind`` at layer 0 (the fp32 fields fp32, the
    model-dtype fields ``dtype``, all on ``device`` — materialize_ops'
    policy), positions=all so no context is needed."""
    g = _gen(seed, device)
    ints = dtype == torch.float32   # the exact-arithmetic control uses integer weights

    def w(*shape, scale=1.0):
        if ints:
            return torch.randint(-3, 4, shape, generator=g).to(torch.float32)
        return torch.randn(*shape, generator=g) * scale
    if kind == "psr_gemv":
        op = la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0, vector=w(d, scale=0.05),
                       gate_w=w(d, scale=1.0 / d ** 0.5), gate_b=torch.tensor([0.0 if ints else 0.1]))
    elif kind == "lowrank_r4":
        op = la.HookOp(op=la.OP_LOWRANK_RELU_GATED, layer=0,
                       lowrank_a=w(4, d, scale=1.0 / d ** 0.5), lowrank_b=w(d, 4, scale=0.05))
    elif kind == "loreft_r4":
        rot = w(4, d)
        if not ints:
            rot, _ = torch.linalg.qr(rot.t())
            rot = rot.t().contiguous()
        op = la.HookOp(op=la.OP_AFFINE_SUBSPACE, layer=0, rotate=rot,
                       affine_w=w(4, d, scale=1.0 / d ** 0.5), affine_b=w(4, scale=0.1))
    elif kind == "add_vector":
        op = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=w(d, scale=0.05))
    elif kind in ("lora_oproj_r1", "lora_oproj_r4"):
        k = 1 if kind.endswith("r1") else 4
        op = la.HookOp(op=la.OP_LORA_OPROJ, layer=0,
                       lowrank_a=w(k, d_in, scale=1.0 / d_in ** 0.5), lowrank_b=w(d_out, k, scale=0.05))
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return la.materialize_ops((op,), device=device, dtype=dtype)


def _set_stacked(flag: bool) -> None:
    la._STACKED_REDUCTIONS = bool(flag)


def apply_arm(arm: str, site: str, ops: Sequence[la.HookOp], rows: torch.Tensor,
              oproj_in: Optional[torch.Tensor]) -> torch.Tensor:
    """One hooked layer's apply of ``ops`` to ``rows`` (block: the residual
    rows; oproj: ``oproj_in`` -> ``rows`` as the o_proj output) on ``arm``.
    legacy = B single-row applies written into a buffer (the per-slot loop's
    arithmetic); per_row / stacked = one grouped apply (independent_rows)
    with the flag off / on."""
    B = rows.shape[0]
    if arm == "legacy":
        out = rows.clone()
        for i in range(B):
            if site == la.HOOK_SITE_BLOCK:
                out[i:i + 1] = la.apply_hook_ops(rows[i:i + 1], ops, 1.0, layer=0)
            else:
                out[i:i + 1] = la.apply_oproj_ops(oproj_in[i:i + 1], rows[i:i + 1], ops, 1.0, layer=0)
        return out
    prev = bool(la._STACKED_REDUCTIONS)
    _set_stacked(arm == "stacked")
    try:
        if site == la.HOOK_SITE_BLOCK:
            return la.apply_hook_ops(rows, ops, 1.0, layer=0, independent_rows=True)
        return la.apply_oproj_ops(oproj_in, rows, ops, 1.0, layer=0, independent_rows=True)
    finally:
        _set_stacked(prev)          # the process mode is left exactly as it was found


def time_call(fn: Callable[[], Any], *, iters: int, warmup: int, device: str) -> Dict[str, float]:
    """Median / min / max wall time of ``fn`` in microseconds (CUDA events
    on a GPU, perf_counter on the CPU), after ``warmup`` calls."""
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    for _ in range(max(0, warmup)):
        fn()
    if cuda:
        torch.cuda.synchronize()
    samples: List[float] = []
    for _ in range(max(1, iters)):
        if cuda:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            samples.append(s.elapsed_time(e) * 1000.0)
        else:
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e6)
    return {"median_us": statistics.median(samples), "min_us": min(samples), "max_us": max(samples),
            "n": len(samples)}


def deviation(stacked: torch.Tensor, per_row: torch.Tensor) -> Dict[str, Any]:
    """The elementwise deviation of ``stacked`` from ``per_row`` (module
    docstring): counts, the largest |diff|, and the ulp histogram over
    the differing elements whose per-row value is nonzero."""
    a, r = stacked.detach(), per_row.detach()
    diff = (a.float() - r.float()).abs()
    ne = diff > 0
    n_ne = int(ne.sum().item())
    out: Dict[str, Any] = {
        "n_elems": int(diff.numel()), "n_unequal": n_ne,
        "frac_unequal": (n_ne / diff.numel()) if diff.numel() else 0.0,
        "n_rows": int(r.shape[0]),
        "n_rows_unequal": int(ne.any(dim=-1).sum().item()) if r.dim() == 2 else None,
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "max_abs_ref": float(r.float().abs().max().item()) if r.numel() else 0.0,
        "ulps_hist": {"<=1": 0, "<=2": 0, ">2": 0}, "max_ulps": 0.0, "n_unequal_ref_zero": 0}
    if n_ne == 0:
        return out
    nz = ne & (r != 0)
    out["n_unequal_ref_zero"] = int((ne & ~nz).sum().item())
    if bool(nz.any()):
        u = la.ulp_distance(a, r)[nz]
        out["max_ulps"] = float(u.max().item())
        out["ulps_hist"] = {"<=1": int((u <= 1.0).sum().item()),
                            "<=2": int(((u > 1.0) & (u <= 2.0)).sum().item()),
                            ">2": int((u > 2.0).sum().item())}
    return out


def gate_scalar_deviation(rows: torch.Tensor, op: la.HookOp) -> Dict[str, Any]:
    """The psr gate's fp32 scalar ``h.float() @ gate_w`` — the ONE reduction
    that differs — stacked vs per row, in fp32 ulps of the per-row value."""
    st = la._matmul(rows.float(), op.gate_w)
    pr = torch.cat([la._matmul(rows[i:i + 1].float(), op.gate_w) for i in range(rows.shape[0])], dim=0)
    ne = st != pr
    d = {"n_rows": int(rows.shape[0]), "n_rows_gate_unequal": int(ne.sum().item()),
         "max_abs_diff": float((st - pr).abs().max().item()), "max_fp32_ulps": 0.0,
         "max_abs_gate": float(pr.abs().max().item())}
    nz = ne & (pr != 0)
    if bool(nz.any()):
        d["max_fp32_ulps"] = float(la.ulp_distance(st, pr)[nz].max().item())
    return d


def bmm_gate(rows: torch.Tensor, op: la.HookOp) -> torch.Tensor:
    """EXPLORATORY third form of the psr gate: one strided-batched GEMM with
    M=1 per batch item (``bmm([B, 1, K], [B, K, 1])``) — a single launch
    that may or may not reproduce the single-row gemv's reduction order.
    Not an adapter path; measured here so the option can be judged."""
    B, K = rows.shape
    return torch.bmm(rows.float().unsqueeze(1), op.gate_w.view(1, K, 1).expand(B, K, 1)).view(B)


def probe_bmm_gate(rows: torch.Tensor, op: la.HookOp, *, iters: int, warmup: int, device: str) -> Dict[str, Any]:
    pr = torch.cat([la._matmul(rows[i:i + 1].float(), op.gate_w) for i in range(rows.shape[0])], dim=0)
    st = la._matmul(rows.float(), op.gate_w)
    bm = bmm_gate(rows, op)
    t = {arm: time_call(fn, iters=iters, warmup=warmup, device=device)["median_us"] for arm, fn in (
        ("per_row_gemv", lambda: torch.cat([la._matmul(rows[i:i + 1].float(), op.gate_w) for i in range(rows.shape[0])], dim=0)),
        ("stacked_gemv", lambda: la._matmul(rows.float(), op.gate_w)),
        ("bmm_gemv", lambda: bmm_gate(rows, op)))}
    return {"bmm_equals_per_row": bool(torch.equal(bm, pr)), "stacked_equals_per_row": bool(torch.equal(st, pr)),
            "bmm_equals_stacked": bool(torch.equal(bm, st)), "median_us": t}


def run_microbench(*, models: Sequence[str] = MODEL_ORDER, batches: Sequence[int] = DEFAULT_BATCHES,
                   kinds: Sequence[Tuple[str, str]] = KINDS, profiles: Sequence[str] = ("gaussian", "heavy"),
                   iters: int = 50, warmup: int = 10, trials: int = 5, device: str = "cuda",
                   dtype: torch.dtype = torch.bfloat16, seed: int = 0,
                   say: Callable[[str], None] = lambda s: None) -> List[Dict[str, Any]]:
    """One record per (model, kind[, o_proj shape], profile, B): the three
    arms' timing (µs per layer-call, and x n_layers -> ms per decode step
    of hook work) and the stacked-vs-per-row deviation over ``trials``
    random inputs (timing on the first trial's inputs)."""
    records: List[Dict[str, Any]] = []
    if not str(device).startswith("cuda"):
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    with torch.no_grad():
        for m in models:
            shp = SHAPES[m]
            d = int(shp["hidden"])
            for kind, site in kinds:
                shapes = ([(None, d, d, int(shp["n_layers"]))] if site == la.HOOK_SITE_BLOCK
                          else [(nm, din, dout, nl) for nm, (din, dout, nl) in shp["oproj"].items()])
                for oname, d_in, d_out, n_layers in shapes:
                    for profile in profiles:
                        pdtype = torch.float32 if profile == "ints" else dtype
                        ops = make_ops(kind, d=d, d_in=d_in, d_out=d_out, device=device, dtype=pdtype,
                                       seed=seed + 7)
                        for B in batches:
                            rec: Dict[str, Any] = {
                                "model": m, "kind": kind, "site": site, "oproj_shape": oname,
                                "d": d, "d_in": d_in, "d_out": d_out, "n_layers": n_layers,
                                "profile": profile, "dtype": str(pdtype).replace("torch.", ""),
                                "B": int(B), "timing": {}, "deviation": [], "elementwise": kind in ELEMENTWISE_KINDS}
                            devs = []
                            t_first = None
                            for trial in range(max(1, trials)):
                                s = seed + 1000 * trial + B
                                if site == la.HOOK_SITE_BLOCK:
                                    rows, oin = make_rows(profile, B, d, device=device, dtype=pdtype, seed=s), None
                                else:
                                    rows = make_rows(profile, B, d_out, device=device, dtype=pdtype, seed=s)
                                    oin = make_rows(profile, B, d_in, device=device, dtype=pdtype, seed=s + 1)
                                out_pr = apply_arm("per_row", site, ops, rows, oin)
                                out_st = apply_arm("stacked", site, ops, rows, oin)
                                out_lg = apply_arm("legacy", site, ops, rows, oin)
                                dv = deviation(out_st, out_pr)
                                dv["legacy_equals_per_row"] = bool(torch.equal(out_lg, out_pr))
                                if kind == "psr_gemv":
                                    dv["gate_scalar"] = gate_scalar_deviation(rows, ops[0])
                                devs.append(dv)
                                if trial == 0:
                                    t_first = (rows, oin)
                            rows, oin = t_first
                            for arm in ARMS:
                                t = time_call(lambda: apply_arm(arm, site, ops, rows, oin),
                                              iters=iters, warmup=warmup, device=device)
                                t["ms_per_step_all_layers"] = t["median_us"] * n_layers / 1000.0
                                rec["timing"][arm] = t
                            rec["deviation"] = devs
                            rec["deviation_summary"] = summarize_deviation(devs)
                            if kind == "psr_gemv":
                                rec["bmm_gate_probe"] = probe_bmm_gate(rows, ops[0], iters=iters, warmup=warmup,
                                                                       device=device)
                            records.append(rec)
                            tm = rec["timing"]
                            say(f"{m:16s} {kind:14s} {oname or '-':9s} {profile:8s} B={B:4d}  "
                                f"legacy {tm['legacy']['median_us']:9.1f}us  per_row {tm['per_row']['median_us']:9.1f}us  "
                                f"stacked {tm['stacked']['median_us']:9.1f}us  | unequal "
                                f"{rec['deviation_summary']['frac_unequal']:.2e} max_ulps {rec['deviation_summary']['max_ulps']:.1f}"
                                + (f"  bmm_gate==per_row {rec['bmm_gate_probe']['bmm_equals_per_row']}"
                                   if "bmm_gate_probe" in rec else ""))
    return records


def summarize_deviation(devs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n_el = sum(int(d["n_elems"]) for d in devs)
    n_ne = sum(int(d["n_unequal"]) for d in devs)
    rows = sum(int(d["n_rows"]) for d in devs)
    rows_ne = sum(int(d["n_rows_unequal"] or 0) for d in devs)
    hist = {"<=1": 0, "<=2": 0, ">2": 0}
    for d in devs:
        for k in hist:
            hist[k] += int(d["ulps_hist"][k])
    out = {"trials": len(devs), "n_elems": n_el, "n_unequal": n_ne,
           "frac_unequal": (n_ne / n_el) if n_el else 0.0,
           "frac_rows_unequal": (rows_ne / rows) if rows else 0.0,
           "max_abs_diff": max((float(d["max_abs_diff"]) for d in devs), default=0.0),
           "max_ulps": max((float(d["max_ulps"]) for d in devs), default=0.0),
           "ulps_hist": hist, "legacy_equals_per_row_all_trials": all(d.get("legacy_equals_per_row") for d in devs)}
    gs = [d["gate_scalar"] for d in devs if "gate_scalar" in d]
    if gs:
        out["gate_scalar"] = {"frac_rows_gate_unequal": sum(g["n_rows_gate_unequal"] for g in gs) / max(1, sum(g["n_rows"] for g in gs)),
                              "max_fp32_ulps": max(g["max_fp32_ulps"] for g in gs)}
    return out


def render_markdown(records: Sequence[Mapping[str, Any]]) -> str:
    lines = ["| model | kind | o_proj | profile | B | legacy µs | per_row µs | stacked µs | per_row/stacked | "
             "hook ms/step per_row -> stacked (x layers) | elems unequal | rows unequal | max ulps | ulps <=1 / <=2 / >2 | max abs diff |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|"]
    for r in records:
        t, s = r["timing"], r["deviation_summary"]
        ratio = t["per_row"]["median_us"] / t["stacked"]["median_us"] if t["stacked"]["median_us"] else float("nan")
        h = s["ulps_hist"]
        lines.append(
            f"| {r['model']} | {r['kind']} | {r['oproj_shape'] or '-'} | {r['profile']} | {r['B']} | "
            f"{t['legacy']['median_us']:.1f} | {t['per_row']['median_us']:.1f} | {t['stacked']['median_us']:.1f} | {ratio:.2f} | "
            f"{t['per_row']['ms_per_step_all_layers']:.2f} -> {t['stacked']['ms_per_step_all_layers']:.2f} | "
            f"{s['frac_unequal']:.2e} | {s['frac_rows_unequal']:.3f} | {s['max_ulps']:.1f} | "
            f"{h['<=1']} / {h['<=2']} / {h['>2']} | {s['max_abs_diff']:.3g} |")
    return "\n".join(lines)


def make_figures(records: Sequence[Mapping[str, Any]], figs_dir: Path) -> List[str]:
    """One PNG per model: time per layer-call by arm (left) and the fraction
    of differing elements (right) against B, one line per kind; best-effort
    (no matplotlib -> nothing written)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 — figures are a convenience
        return []
    figs_dir.mkdir(parents=True, exist_ok=True)
    written = []
    models = sorted({r["model"] for r in records}, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
    for m in models:
        rs = [r for r in records if r["model"] == m and r["profile"] != "ints"]
        if not rs:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        labels = sorted({(r["kind"], r["oproj_shape"] or "") for r in rs})
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, (kind, oname) in enumerate(labels):
            sub = sorted([r for r in rs if r["kind"] == kind and (r["oproj_shape"] or "") == oname
                          and r["profile"] == rs[0]["profile"]], key=lambda r: r["B"])
            Bs = [r["B"] for r in sub]
            lab = kind + (f"[{oname}]" if oname else "")
            color = palette[i % len(palette)]          # one colour per kind, one line style per arm
            for arm, ls in (("legacy", ":"), ("per_row", "--"), ("stacked", "-")):
                axes[0].plot(Bs, [r["timing"][arm]["median_us"] for r in sub], ls, color=color, marker="o",
                             ms=3, label=(lab if arm == "stacked" else None))
            axes[1].plot(Bs, [max(r["deviation_summary"]["frac_unequal"], 1e-9) for r in sub], "-", color=color,
                         marker="o", ms=3, label=lab)
        axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
        axes[0].set_xlabel("B (rows in the decode group)"); axes[0].set_ylabel("µs per hooked layer (median)")
        axes[0].set_title(f"{m}: apply cost per layer — legacy (:) / per_row (--) / stacked (-)")
        axes[0].legend(fontsize=7, title="colour = kind", title_fontsize=7)
        axes[1].set_xscale("log", base=2); axes[1].set_yscale("log")
        axes[1].set_xlabel("B"); axes[1].set_ylabel("fraction of output elements differing (stacked vs per_row)")
        axes[1].set_title(f"{m}: deviation ({rs[0]['profile']} activations; 1e-9 = none)")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        p = figs_dir / f"microbench_{m}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(str(p))
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", default=",".join(MODEL_ORDER))
    ap.add_argument("--batches", default=",".join(str(b) for b in DEFAULT_BATCHES))
    ap.add_argument("--profiles", default="gaussian,heavy")
    ap.add_argument("--kinds", default=",".join(k for k, _ in KINDS))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON path (a sibling .md table is written next to it)")
    ap.add_argument("--figs", default=None, help="directory for the per-model PNGs")
    args = ap.parse_args(argv)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("FATAL: --device cuda but torch.cuda.is_available() is False", file=sys.stderr)
        return 1
    kinds = [(k, s) for k, s in KINDS if k in {x.strip() for x in args.kinds.split(",") if x.strip()}]
    records = run_microbench(
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        batches=[int(b) for b in args.batches.split(",") if b.strip()],
        kinds=kinds, profiles=[p.strip() for p in args.profiles.split(",") if p.strip()],
        iters=args.iters, warmup=args.warmup, trials=args.trials, device=args.device, seed=args.seed,
        say=lambda s: print(s, flush=True))
    doc = {"device": args.device, "torch": torch.__version__,
           "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() and str(args.device).startswith("cuda") else None),
           "iters": args.iters, "warmup": args.warmup, "trials": args.trials, "seed": args.seed,
           "records": records}
    md = render_markdown(records)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1))
        out.with_suffix(".md").write_text(md + "\n")
        print(f"wrote {out} and {out.with_suffix('.md')}")
    if args.figs:
        for p in make_figures(records, Path(args.figs)):
            print(f"wrote {p}")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
