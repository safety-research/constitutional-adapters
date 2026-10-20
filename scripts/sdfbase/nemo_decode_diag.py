#!/usr/bin/env python
"""Decode-path diagnosis for ``base_gsm8k.py`` on a Mamba2 hybrid:
WHY did Nemotron-3-Nano-4B decode at ~1.75 s/step (batch 128, the same
~2 h per 500-question cell at batch 32) when GLM-4.6V-Flash took ~13 min
per cell on the same evaluator?

One container, one model load, ONE length-bucketed chunk of the real
500-question draw (the evaluator's own render, bucketing, StopRecorder and
generate kwargs), a fixed number of new tokens per arm:

  cache_free           use_cache False on the model + generation configs —
                       the path ``models.pin_kv_cache_off_if_unallocatable``
                       put the published r2 / r2_x2 cells on (transformers
                       5.14.1 cannot build a DynamicCache for NemotronH's
                       "mlp" blocks, so the cache was pinned OFF);
  cached               use_cache True after ``models.prepare_kv_cache``
                       registered the "mlp" placeholder cache layer (the
                       fix; the entry transformers 5.15.0 ships);
  cached_no_criteria   ``cached`` without the StopRecorder / stop-string
                       criterion (candidate 3: per-step host syncs);
  cached_torch_scan    ``cached`` with the modeling module's fast-path
                       latch forced off, so every Mamba2 mixer runs the
                       pure-torch SSD scan (candidate 2: what an image
                       without the fused kernels would measure).

``cache_free`` and ``cached`` run at every ``--batch-sizes`` value
(candidate 4: padding geometry / batch dependence). Per arm: mean s/step
over the decode steps (the prefill step excluded), GPU SM duty sampled by
``nvidia-smi dmon`` for the arm's duration, peak allocated memory. Then a
token-level parity read of cache_free vs cached on the first ``--steps``
tokens of every row (identical rows, first divergence step), and a
torch.profiler table of a few cached decode steps (top CUDA kernels + CPU
ops). Everything lands in ``--out`` as JSON and is printed as a table.
Diagnosis only: nothing is published to any label."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import base_gsm8k as bg  # noqa: E402


class _Dmon:
    """``nvidia-smi dmon -s u -d 1`` for the life of a ``with`` block; SM%
    samples (one per second) in ``.sm``."""

    def __init__(self, gpu: int = 0) -> None:
        self.gpu = gpu
        self.sm: list[int] = []
        self.mem: list[int] = []
        self._p = None
        self._t = None

    def _pump(self) -> None:
        assert self._p is not None and self._p.stdout is not None
        for line in self._p.stdout:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    self.sm.append(int(parts[1]))
                    self.mem.append(int(parts[2]))
                except ValueError:
                    pass

    def __enter__(self) -> "_Dmon":
        try:
            self._p = subprocess.Popen(
                ["nvidia-smi", "dmon", "-i", str(self.gpu), "-s", "u", "-d", "1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self._t = threading.Thread(target=self._pump, daemon=True)
            self._t.start()
        except (OSError, FileNotFoundError):
            self._p = None
        return self

    def __exit__(self, *exc) -> None:
        if self._p is not None:
            self._p.terminate()
            try:
                self._p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._p.kill()
        if self._t is not None:
            self._t.join(timeout=5)

    def summary(self) -> dict:
        sm = self.sm
        return {"n_samples": len(sm),
                "sm_mean": round(sum(sm) / len(sm), 1) if sm else None,
                "sm_min": min(sm) if sm else None,
                "sm_max": max(sm) if sm else None,
                "mem_bw_mean": (round(sum(self.mem) / len(self.mem), 1)
                                if self.mem else None)}


def _step_timer_cls():
    """A StoppingCriteria that records a wall-clock stamp after every
    generated token (with a CUDA sync, so a stamp means the step's kernels
    finished) and never stops anything."""
    import torch
    from transformers.generation import StoppingCriteria

    class StepTimer(StoppingCriteria):
        def __init__(self) -> None:
            self.stamps: list[float] = []

        def __call__(self, input_ids, scores=None, **kwargs):
            if input_ids.is_cuda:
                torch.cuda.synchronize(input_ids.device)
            self.stamps.append(time.perf_counter())
            return torch.zeros(input_ids.shape[0], dtype=torch.bool,
                               device=input_ids.device)

    return StepTimer


def _set_cache(model, on: bool) -> None:
    model.config.use_cache = bool(on)
    tc = getattr(model.config, "text_config", None)
    if tc is not None:
        tc.use_cache = bool(on)
    model.generation_config.use_cache = bool(on)


def run_arm(model, tok, enc, *, name: str, steps: int, cache: bool,
            criteria: bool, torch_scan: bool, eos_ids, stop_strings,
            pad_id: int) -> tuple[dict, "object"]:
    """One timed generate on ``enc`` (an already-built left-padded batch).
    Returns (record, generated ids)."""
    import torch
    from transformers.generation import StoppingCriteriaList

    _set_cache(model, cache)
    mixers = [m for m in model.modules()
              if type(m).__name__.endswith("Mamba2Mixer")]
    modeling = sys.modules[type(mixers[0]).__module__] if mixers else None
    prev_fast = getattr(modeling, "is_fast_path_available", None)
    if torch_scan and modeling is not None:
        modeling.is_fast_path_available = False
    timer = _step_timer_cls()()
    crits = []
    if criteria:
        rec = bg.make_stop_recorder(tok, eos_ids, stop_strings)
        rec.reset(int(enc["input_ids"].shape[0]),
                  int(enc["input_ids"].shape[1]))
        crits.append(rec)
    crits.append(timer)
    gen_kwargs = dict(max_new_tokens=int(steps), do_sample=False,
                      pad_token_id=pad_id,
                      stopping_criteria=StoppingCriteriaList(crits),
                      use_cache=bool(cache))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        with _Dmon() as dm, torch.no_grad():
            gen = model.generate(**enc, **gen_kwargs)
            torch.cuda.synchronize()
    finally:
        if torch_scan and modeling is not None:
            modeling.is_fast_path_available = prev_fast
    wall = time.perf_counter() - t0
    st = timer.stamps
    first = (st[0] - t0) if st else None          # prefill + first token
    decode = [b - a for a, b in zip(st, st[1:])]  # pure decode steps
    n_dec = len(decode)
    rec_out = {
        "arm": name, "batch": int(enc["input_ids"].shape[0]),
        "width": int(enc["input_ids"].shape[1]),
        "steps_generated": int(gen.shape[1] - enc["input_ids"].shape[1]),
        "wall_s": round(wall, 3),
        "prefill_plus_first_token_s": round(first, 3) if first else None,
        "decode_s_per_step_mean": (round(sum(decode) / n_dec, 4)
                                   if n_dec else None),
        "decode_s_per_step_median": (round(sorted(decode)[n_dec // 2], 4)
                                     if n_dec else None),
        "decode_s_per_step_max": round(max(decode), 4) if n_dec else None,
        "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "gpu_duty": dm.summary(),
        "kv_cache": bool(cache), "stop_criteria": bool(criteria),
        "mamba_path": ("torch" if torch_scan else
                       ("fused" if prev_fast else "torch")),
    }
    return rec_out, gen


def parity(gen_a, gen_b, width: int, steps: int) -> dict:
    """Row-level agreement of two greedy generations over the first
    ``steps`` new tokens."""
    a = gen_a[:, width:width + steps].cpu()
    b = gen_b[:, width:width + steps].cpu()
    n = min(a.shape[1], b.shape[1])
    a, b = a[:, :n], b[:, :n]
    same = (a == b).all(dim=1)
    first_div = []
    for r in range(a.shape[0]):
        if not bool(same[r]):
            d = (a[r] != b[r]).nonzero()
            first_div.append(int(d[0]) if len(d) else None)
    return {"rows": int(a.shape[0]), "tokens_compared": int(n),
            "rows_identical": int(same.sum()),
            "rows_differing": int((~same).sum()),
            "first_divergence_steps": sorted(x for x in first_div
                                             if x is not None)}


def reference_mode(args, model, tok, rows, prompts, pad_id, eos_ids) -> dict:
    """Which bf16 decode path is FAITHFUL? On the rows named by --ref-qidx
    (the published cell's rows whose stop reason changed between the
    cache-free and the cached decode), run the full evaluator decode (512
    new tokens, StopRecorder, greedy) under four arms — bf16 / fp32 x
    cache-free / cached — as ONE left-padded batch, plus bf16 cache-free vs
    cached at batch 1 (no padding at all) on the first --ref-bs1 rows. fp32
    weights are the bf16 weights widened (lossless), so the two fp32 arms
    are the model's own answer up to fp32 rounding; whichever bf16 arm
    tracks them on the row OUTCOMES (extracted answer, stop reason) is the
    faithful one. Everything per row per arm lands in the JSON."""
    import torch
    from sisteer.generate import assert_left_padded, encode_kwargs

    qidx = [int(x) for x in args.ref_qidx.split(",") if x]
    sel = [prompts[i] for i in qidx]
    device = next(model.parameters()).device
    prev_side = tok.padding_side
    tok.padding_side = "left"

    def build(chunk):
        enc = tok(chunk, return_tensors="pt", padding=True,
                  **encode_kwargs(tok, chunk[0])).to(device)
        assert_left_padded(enc.get("attention_mask"), what="ref batch")
        return enc

    def decode(enc, gen, width, steps_cap):
        rec_rows = []
        for k in range(gen.shape[0]):
            new = gen[k, width:].tolist()
            rec_rows.append(new)
        return rec_rows

    def run(name, enc, *, cache, dtype):
        from transformers.generation import StoppingCriteriaList

        if dtype == "fp32":
            model.float()
        else:
            model.to(torch.bfloat16)
        _set_cache(model, cache)
        rec = bg.make_stop_recorder(tok, eos_ids, bg.STOP_STRINGS)
        width = int(enc["input_ids"].shape[1])
        rec.reset(int(enc["input_ids"].shape[0]), width)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=int(args.steps),
                                     do_sample=False, pad_token_id=pad_id,
                                     stopping_criteria=StoppingCriteriaList(
                                         [rec]),
                                     use_cache=bool(cache))
                torch.cuda.synchronize()
        except (torch.OutOfMemoryError, RuntimeError) as e:
            torch.cuda.empty_cache()
            r = {"arm": name, "error": str(e).splitlines()[0][:300]}
            print(f"[ref] {json.dumps(r)}", flush=True)
            return r, None
        wall = time.perf_counter() - t0
        accounted = rec.finish(int(gen.shape[1]) - width, int(args.steps))
        per_row = []
        for k in range(gen.shape[0]):
            n_new, why = accounted[k]
            text = tok.decode(gen[k, width:width + n_new].tolist(),
                              skip_special_tokens=True)
            block, _stopped = bg.truncate_at_stop(text, bg.STOP_STRINGS)
            pred, rule = bg.extract_answer(block)
            per_row.append({"stop_reason": why, "n_new": int(n_new),
                            "extracted": pred, "rule": rule,
                            "tail": block[-80:]})
        r = {"arm": name, "dtype": dtype, "kv_cache": bool(cache),
             "batch": int(gen.shape[0]), "width": width,
             "wall_s": round(wall, 1),
             "stop_reasons": {k: sum(p["stop_reason"] == k for p in per_row)
                              for k in ("eos", "stop_string", "cap",
                                        "early_unknown")},
             "rows": per_row}
        print(f"[ref] {json.dumps({k: v for k, v in r.items() if k != 'rows'})}",
              flush=True)
        return r, gen[:, width:].cpu()

    out = {"qidx": qidx, "arms": {}, "tokens_identical": {}}
    gens: dict[str, object] = {}
    try:
        enc = build(sel)
        for name, cache, dtype in (("bf16_cache_free", False, "bf16"),
                                   ("bf16_cached", True, "bf16"),
                                   ("fp32_cached", True, "fp32"),
                                   ("fp32_cache_free", False, "fp32")):
            r, g = run(name, enc, cache=cache, dtype=dtype)
            out["arms"][name] = r
            if g is not None:
                gens[name] = g
        # batch-1 (unpadded) bf16 pair on the first --ref-bs1 rows
        bs1 = {"bf16_cache_free_bs1": [], "bf16_cached_bs1": []}
        for i in qidx[:int(args.ref_bs1)]:
            enc1 = build([prompts[i]])
            for name, cache in (("bf16_cache_free_bs1", False),
                                ("bf16_cached_bs1", True)):
                r, g = run(name, enc1, cache=cache, dtype="bf16")
                if g is not None:
                    bs1[name].append((i, r["rows"][0], g[0].tolist()))
        out["bs1"] = {
            "qidx": qidx[:int(args.ref_bs1)],
            "rows": [{"qidx": a[0], "cache_free": a[1], "cached": b[1],
                      "tokens_identical": a[2] == b[2]}
                     for a, b in zip(bs1["bf16_cache_free_bs1"],
                                     bs1["bf16_cached_bs1"])]}
        names = list(gens)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ga, gb = gens[a], gens[b]
                n = min(ga.shape[1], gb.shape[1])
                same = int((ga[:, :n] == gb[:, :n]).all(dim=1).sum())
                out["tokens_identical"][f"{a}|{b}"] = same
        # outcome agreement per pair (extracted + stop_reason)
        out["outcomes_identical"] = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ra, rb = out["arms"][a]["rows"], out["arms"][b]["rows"]
                out["outcomes_identical"][f"{a}|{b}"] = sum(
                    x["extracted"] == y["extracted"]
                    and x["stop_reason"] == y["stop_reason"]
                    for x, y in zip(ra, rb))
        print("[ref] tokens identical:", json.dumps(out["tokens_identical"]),
              flush=True)
        print("[ref] outcomes identical:",
              json.dumps(out["outcomes_identical"]), flush=True)
    finally:
        tok.padding_side = prev_side
        model.to(torch.bfloat16)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-qidx", default="",
                    help="REFERENCE mode instead of the timing arms: comma "
                         "list of draw indices (qidx) to decode under bf16/"
                         "fp32 x cache-free/cached (see reference_mode)")
    ap.add_argument("--ref-bs1", type=int, default=3,
                    help="reference mode: how many of the rows also run at "
                         "batch 1 (unpadded) under both bf16 paths")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=bg.DEFAULT_SHOTS)
    ap.add_argument("--chunk", type=int, default=1,
                    help="which length-bucketed chunk of the draw (at the "
                         "largest --batch-sizes value) to measure on")
    ap.add_argument("--batch-sizes", default="128,32")
    ap.add_argument("--steps", type=int, default=32,
                    help="new tokens for the cache-free arms and the parity")
    ap.add_argument("--steps-cached", type=int, default=256,
                    help="new tokens for the cached arms (long enough for "
                         "dmon to sample the steady state)")
    ap.add_argument("--profile-steps", type=int, default=8)
    ap.add_argument("--no-torch-scan", action="store_true")
    ap.add_argument("--noise", action="store_true",
                    help="also repeat cache_free and cached once at the big "
                         "batch and report each path's run-to-run token "
                         "agreement (the floor the cross-path parity is read "
                         "against)")
    args = ap.parse_args(argv)

    import torch
    import yaml

    from sisteer.eval.capability_ext import _maybe_inject
    from sisteer.extraction.fitmon import pin_attention_backends
    from sisteer.generate import assert_left_padded, encode_kwargs
    from sisteer.models import load_model_and_tokenizer

    pin_attention_backends()
    cfg = yaml.safe_load(Path(args.config).read_text())
    rows, dataset_revision = bg.load_rows(args.n, args.seed)
    prompts = [bg.render_prompt(r["question"], args.shots) for r in rows]

    model, tok = load_model_and_tokenizer(
        cfg["model"], dtype=cfg.get("dtype", "bfloat16"),
        revision=cfg.get("model_revision"),
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    model.eval()
    runtime = bg.decode_runtime(model)
    from transformers.cache_utils import DYNAMIC_LAYER_TYPE_MAPPING
    runtime["mlp_in_cache_mapping"] = "mlp" in DYNAMIC_LAYER_TYPE_MAPPING
    try:
        import mamba_ssm  # noqa: F401
        runtime["mamba_ssm"] = getattr(mamba_ssm, "__version__", "present")
    except ImportError:
        runtime["mamba_ssm"] = None
    try:
        import causal_conv1d  # noqa: F401
        runtime["causal_conv1d"] = getattr(causal_conv1d, "__version__",
                                           "present")
    except ImportError:
        runtime["causal_conv1d"] = None
    runtime["gpu"] = torch.cuda.get_device_name(0)
    print(f"[diag] runtime after load: {json.dumps(runtime)}", flush=True)

    pad_id, eos_ids = bg._end_ids(model, tok)
    prompts = [_maybe_inject(None, p) for p in prompts]
    if args.ref_qidx:
        ref = reference_mode(args, model, tok, rows, prompts, pad_id, eos_ids)
        ref["runtime"] = runtime
        ref["dataset_revision"] = dataset_revision
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(ref, indent=1))
        print(f"[diag] wrote {args.out}", flush=True)
        return 0
    lengths = [len(tok(p, **encode_kwargs(tok, p))["input_ids"])
               for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: (lengths[i], i))
    sizes = [int(b) for b in args.batch_sizes.split(",") if b]
    big = max(sizes)
    idx_big = order[args.chunk * big:(args.chunk + 1) * big]
    device = next(model.parameters()).device
    prev_side = tok.padding_side
    tok.padding_side = "left"

    def build(idx):
        chunk = [prompts[j] for j in idx]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  **encode_kwargs(tok, chunk[0])).to(device)
        assert_left_padded(enc.get("attention_mask"), what="diag batch")
        return enc

    arms: list[dict] = []
    gens: dict[tuple, object] = {}

    def timed(name, enc, **kw):
        """One arm; an OOM / kernel failure is RECORDED, not fatal (the
        pure-torch SSD scan at batch 128 asked for 1152 GiB on the first
        run of this diagnosis — that number is the finding)."""
        try:
            r, g = run_arm(model, tok, enc, name=name, eos_ids=eos_ids,
                           stop_strings=bg.STOP_STRINGS, pad_id=pad_id, **kw)
        except (torch.OutOfMemoryError, RuntimeError) as e:  # noqa: PERF203
            torch.cuda.empty_cache()
            r, g = {"arm": name, "batch": int(enc["input_ids"].shape[0]),
                    "width": int(enc["input_ids"].shape[1]),
                    "error": str(e).splitlines()[0][:300], **kw}, None
        arms.append(r)
        print(f"[diag] {json.dumps(r, default=str)}", flush=True)
        return g

    try:
        # warm-up (triton autotune / kernel compile for the cached path)
        enc_w = build(idx_big[:2])
        run_arm(model, tok, enc_w, name="warmup", steps=4, cache=True,
                criteria=True, torch_scan=False, eos_ids=eos_ids,
                stop_strings=bg.STOP_STRINGS, pad_id=pad_id)
        for bs in sizes:
            idx = idx_big[:bs]
            enc = build(idx)
            gens[("cache_free", bs)] = timed(
                "cache_free", enc, steps=args.steps, cache=False,
                criteria=True, torch_scan=False)
            gens[("cached", bs)] = timed(
                "cached", enc, steps=args.steps_cached, cache=True,
                criteria=True, torch_scan=False)
            if bs == big:
                timed("cached_no_criteria", enc, steps=args.steps_cached,
                      cache=True, criteria=False, torch_scan=False)
        if not args.no_torch_scan:
            # the pure-torch SSD scan materialises (B, heads, L, L) chunk
            # products: batch 128 x width ~760 asked for 1152 GiB — so the
            # what-if is measured on a small sub-batch, and both paths on
            # it, so the ratio is like for like
            small = idx_big[:min(8, big)]
            enc_s = build(small)
            timed("cached_fused_bs8", enc_s, steps=args.steps, cache=True,
                  criteria=True, torch_scan=False)
            timed("cached_torch_scan_bs8", enc_s, steps=args.steps,
                  cache=True, criteria=True, torch_scan=True)
        # parity cache_free vs cached on the first --steps tokens
        par = {}
        for bs in sizes:
            ga, gb = gens.get(("cache_free", bs)), gens.get(("cached", bs))
            if ga is None or gb is None:
                continue
            width = int(build(idx_big[:bs])["input_ids"].shape[1])
            par[str(bs)] = parity(ga, gb, width, args.steps)
            print(f"[diag] parity bs={bs}: {json.dumps(par[str(bs)])}",
                  flush=True)
        if args.noise:
            # the noise FLOOR each path has against itself: a second run of
            # cache_free and of cached at the big batch, same rows, so the
            # cross-path divergence above can be read against run-to-run
            # bf16 nondeterminism (text diff is a hypersensitive readout)
            enc = build(idx_big)
            width = int(enc["input_ids"].shape[1])
            g_free2 = timed("cache_free_repeat", enc, steps=args.steps,
                            cache=False, criteria=True, torch_scan=False)
            g_cached2 = timed("cached_repeat", enc, steps=args.steps,
                              cache=True, criteria=True, torch_scan=False)
            ga, gb = gens.get(("cache_free", big)), gens.get(("cached", big))
            if ga is not None and g_free2 is not None:
                par["noise_cache_free_vs_repeat"] = parity(ga, g_free2, width,
                                                           args.steps)
            if gb is not None and g_cached2 is not None:
                par["noise_cached_vs_repeat"] = parity(gb, g_cached2, width,
                                                       args.steps)
            if g_free2 is not None and g_cached2 is not None:
                par["cache_free_repeat_vs_cached_repeat"] = parity(
                    g_free2, g_cached2, width, args.steps)
            for k in ("noise_cache_free_vs_repeat", "noise_cached_vs_repeat",
                      "cache_free_repeat_vs_cached_repeat"):
                if k in par:
                    print(f"[diag] parity {k}: {json.dumps(par[k])}",
                          flush=True)
        # profile a few cached decode steps at the big batch
        prof_txt = {}
        if args.profile_steps > 0:
            from torch.profiler import ProfilerActivity, profile

            _set_cache(model, True)
            enc = build(idx_big)
            with torch.no_grad(), profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                model.generate(**enc, max_new_tokens=int(args.profile_steps),
                               do_sample=False, pad_token_id=pad_id,
                               use_cache=True)
                torch.cuda.synchronize()
            ka = prof.key_averages()
            prof_txt["by_cuda_time"] = ka.table(sort_by="cuda_time_total",
                                                row_limit=25)
            prof_txt["by_cpu_time"] = ka.table(sort_by="cpu_time_total",
                                               row_limit=25)
            print("[diag] profile (cached, top CUDA time):\n"
                  + prof_txt["by_cuda_time"], flush=True)
    finally:
        tok.padding_side = prev_side

    out = {"runtime": runtime, "dataset_revision": dataset_revision,
           "chunk": args.chunk, "chunk_rows": len(idx_big),
           "prompt_width_tokens": {"min": min(lengths[j] for j in idx_big),
                                   "max": max(lengths[j] for j in idx_big)},
           "arms": arms, "parity": par, "profile": prof_txt}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("\n| arm | batch | width | kv_cache | mamba | criteria | s/step "
          "(mean) | s/step (median) | prefill+1st (s) | SM% mean | peak GiB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in arms:
        if "error" in r:
            print(f"| {r['arm']} | {r['batch']} | {r['width']} | "
                  f"{r.get('cache')} | {'torch' if r.get('torch_scan') else 'fused'}"
                  f" | {r.get('criteria')} | ERROR: {r['error']} | | | | |")
            continue
        print(f"| {r['arm']} | {r['batch']} | {r['width']} | {r['kv_cache']} "
              f"| {r['mamba_path']} | {r['stop_criteria']} | "
              f"{r['decode_s_per_step_mean']} | "
              f"{r['decode_s_per_step_median']} | "
              f"{r['prefill_plus_first_token_s']} | "
              f"{r['gpu_duty']['sm_mean']} | {r['peak_mem_gib']} |")
    print(f"[diag] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
