"""Scoped microbenchmark: does per-forward wall time stay FLAT in batch size
(memory-bound) or scale ~LINEARLY (compute-bound)?

PHASE MATTERS: the original grid timed FULL-SEQUENCE forwards — PREFILL-shaped,
compute-bound by nature, so linear scaling there is EXPECTED and non-diagnostic
(the mirror image of the original prefill-logic projection error). The
workload's cost is 256 sequential DECODE steps (one new token against a KV
cache), which theory says is memory-bound. So the decisive phase is
--phase decode: time a SINGLE-TOKEN forward against a pre-built KV cache of
length S, across batch sizes. Also instruments a real model.generate for
per-forward input shapes (is the KV cache actually used on THIS stack? — the
CPU code path is cache-on, verified) and times a fixed-length generate
(min_new_tokens == max_new_tokens, no early-stop variance) as the end-to-end
corroboration.

DELIBERATELY NARROW: one forward pass at a FIXED sequence length across batch
sizes — NO padding (every row full length, attention_mask all ones), NO
generation-length variance, NO steering (a separate small ctx-on pass isolates
hook cost). No smoke, no probe, no held-out leg. Random token ids (content-free).

DECISION RULE (report the verdict against it):
  - per-forward time ~FLAT in batch (ratio t(B=8)/t(B=1) small) -> memory-bound;
    the whole batched-generation loss is PADDING; keep+tune batching via the
    term-2 levers (tighter cap, early retirement, smaller chunks).
  - per-forward time ~LINEAR in batch (ratio ~= B) -> batching cannot win on this
    stack regardless of padding; stop tuning batching, put effort into
    generation length alone.
Two sequence lengths bracket our geometry: ~300 (short attack prompt) and ~6700
(long-instruction / constitution prefix), so we also see whether any batch
penalty scales with sequence length. Both in ONE job (free).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path


def _summarise_steps(tag, w):
    """Per-call step-trace summary + THE SHAPE VERDICT: fit per-step wall vs
    STEP INDEX over the decode steps (drop step 0, the prefill — a
    different-shape forward). A least-squares slope + its R^2 separate the
    hypotheses:
      FLAT            -> a fixed per-step overhead (kernel/config)
      FRONT_LOADED    -> warmup/compile amortised (first steps dear, then cheap)
      LINEARLY_RISING -> per-step work grows with the prefix (DynamicCache
                         torch.cat copying the whole accumulated cache — the
                         signature step_trace REFUTED, kept for reuse)
    Shared by the step_trace and warmup_scope phases."""
    if len(w) < 12:
        return {"tag": tag, "n_steps": len(w),
                "steps_ms": [round(x * 1e3, 2) for x in w]}
    ms = [x * 1e3 for x in w]
    dec = ms[1:]
    n = len(dec)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(dec) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, dec))
    slope = (sxy / sxx) if sxx else 0.0            # ms per step
    ss_tot = sum((y - my) ** 2 for y in dec)
    ss_res = sum((y - (my + slope * (x - mx))) ** 2
                 for x, y in zip(xs, dec))
    r2 = (1 - ss_res / ss_tot) if ss_tot else 0.0
    f10 = sum(dec[:10]) / 10
    l10 = sum(dec[-10:]) / 10
    if l10 > 1.3 * f10 and slope > 0:
        shape = "LINEARLY_RISING__cache_growth_consistent"
    elif f10 > 1.3 * l10:
        shape = "FRONT_LOADED__warmup_consistent"
    else:
        shape = "FLAT__fixed_per_step_overhead"
    return {"tag": tag, "n_steps": len(w),
            "prefill_ms": round(ms[0], 1),
            "first10_mean_ms": round(f10, 2),
            "mid10_mean_ms": round(
                sum(ms[len(ms)//2:len(ms)//2+10]) / 10, 2),
            "last10_mean_ms": round(l10, 2),
            "decode_slope_ms_per_step": round(slope, 4),
            "decode_slope_r2": round(r2, 3),
            "last10_over_first10": round(l10 / f10, 2) if f10 else None,
            "shape_verdict": shape,
            "steps_ms": [round(x, 2) for x in ms]}


def _time_forward(model, B, S, reps, device, warmup, obj=None, scale=1.0):
    import torch
    ids = torch.randint(0, int(model.config.vocab_size), (B, S), device=device)
    am = torch.ones(B, S, dtype=torch.long, device=device)

    def one():
        with torch.no_grad():
            model(input_ids=ids, attention_mask=am, use_cache=False)

    ctx = obj.apply(model, scale=scale) if obj is not None else nullcontext()
    with ctx:
        for _ in range(warmup):
            one()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            one()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
    del ids, am
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dt


def _grid(model, device, batches, seqs, reps, warmup, obj=None, tag=""):
    rows = []
    for S in seqs:
        for B in batches:
            try:
                dt = _time_forward(model, B, S, reps, device, warmup, obj=obj)
                rows.append({"tag": tag, "seq_len": S, "batch": B,
                             "per_forward_ms": round(dt * 1e3, 2),
                             "per_seq_ms": round(dt * 1e3 / B, 3)})
                print(f"[bench] {tag} S={S} B={B}: "
                      f"{dt*1e3:.1f} ms/forward  {dt*1e3/B:.2f} ms/seq")
            except Exception as exc:  # noqa: BLE001 - OOM etc: record, keep going
                rows.append({"tag": tag, "seq_len": S, "batch": B,
                             "error": f"{type(exc).__name__}"})
                print(f"[bench] {tag} S={S} B={B}: ERROR {type(exc).__name__}")
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return rows


def _verdict(rows, seq_len):
    """t(B=8)/t(B=1) per-forward at seq_len: <2 -> flat/memory-bound; ~8 ->
    linear/compute-bound; between -> partial. Decision rule verdict."""
    at = {r["batch"]: r for r in rows
          if r.get("seq_len") == seq_len and "per_forward_ms" in r and r["tag"] == "no_ctx"}
    if 1 not in at or 8 not in at:
        return {"seq_len": seq_len, "verdict": "insufficient_data"}
    ratio = at[8]["per_forward_ms"] / at[1]["per_forward_ms"]
    if ratio < 2.0:
        v = "FLAT_memory_bound__loss_is_padding__keep_and_tune_batching"
    elif ratio > 5.0:
        v = "LINEAR_compute_bound__batching_cannot_win__tune_generation_length_only"
    else:
        v = "PARTIAL__batching_helps_sublinearly"
    return {"seq_len": seq_len, "ratio_B8_over_B1": round(ratio, 2), "verdict": v}


def _time_decode_step(model, B, S, reps, device, warmup, obj=None, scale=1.0):
    """Per-DECODE-STEP wall: one new token against a pre-built KV cache of length
    ~S. The cache grows by (warmup+reps) tokens across the timed loop — negligible
    vs S. THE decision-rule quantity: decode is the phase the workload pays for."""
    import torch
    from contextlib import nullcontext
    ids = torch.randint(0, int(model.config.vocab_size), (B, S), device=device)
    step = torch.randint(0, int(model.config.vocab_size), (B, 1), device=device)
    ctx = obj.apply(model, scale=scale) if obj is not None else nullcontext()
    with ctx, torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        for _ in range(warmup):
            out = model(input_ids=step, past_key_values=past, use_cache=True)
            past = out.past_key_values
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            out = model(input_ids=step, past_key_values=past, use_cache=True)
            past = out.past_key_values
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
    del ids, step, out, past
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dt


def _decode_grid(model, device, batches, seqs, reps, warmup, obj=None, tag="decode"):
    rows = []
    for S in seqs:
        for B in batches:
            try:
                dt = _time_decode_step(model, B, S, reps, device, warmup, obj=obj)
                rows.append({"tag": tag, "seq_len": S, "batch": B,
                             "per_forward_ms": round(dt * 1e3, 3),
                             "per_seq_ms": round(dt * 1e3 / B, 4)})
                print(f"[bench] {tag} S={S} B={B}: {dt*1e3:.2f} ms/step  "
                      f"{dt*1e3/B:.3f} ms/seq-step")
            except Exception as exc:  # noqa: BLE001
                rows.append({"tag": tag, "seq_len": S, "batch": B,
                             "error": type(exc).__name__})
                print(f"[bench] {tag} S={S} B={B}: ERROR {type(exc).__name__}")
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return rows


def _instrument_cache_use(model, tok) -> dict:
    """Is the KV cache ACTUALLY used by the real serving path on THIS stack?
    Records every model forward's input shape + incoming cache length during one
    real defense.generate (2 prompts, 5 new tokens). Cache-on = one [B,S] prefill
    then [B,1] steps; cache-off = growing full-width recomputes (the would-be
    finding of the day)."""
    from sisteer.eval.jailbreak import build_defense
    calls = []

    def pre(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        pkv = kwargs.get("past_key_values")
        n = int(pkv.get_seq_length()) if (pkv is not None and
                                          hasattr(pkv, "get_seq_length")) else None
        calls.append({"shape": list(ids.shape) if ids is not None else None,
                      "cache_len_in": n, "use_cache": kwargs.get("use_cache")})
        return None

    h = model.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        d = build_defense(model, tok, mode="none", positions="all", batch_size=2)
        d.generate(["short probe one", "a somewhat longer second probe here"],
                   max_new_tokens=5)
    finally:
        h.remove()
    steps = calls[1:]
    cache_used = bool(steps) and all(c["shape"][1] == 1 for c in steps)
    return {"calls": calls, "kv_cache_used": cache_used}


def _time_fixed_generate(model, tok, B, S, new_tokens, device, obj=None,
                         scale: float = 1.0):
    """End-to-end model.generate with min_new_tokens == max_new_tokens (every
    sequence forced to the same length: no early-stop, no padding variance) —
    the real serving primitive with zero length confound. ``obj`` gives the
    STEERED twin, mirroring generate_with_object's serving (the ctx entered
    around the generate, handle reset(prompt_len) like generate_batch does)."""
    import torch
    from contextlib import nullcontext
    ids = torch.randint(0, int(model.config.vocab_size), (B, S), device=device)
    am = torch.ones(B, S, dtype=torch.long, device=device)
    kw = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens,
              do_sample=False, pad_token_id=tok.pad_token_id)

    def one():
        ctx = obj.apply(model, scale=scale) if obj is not None else nullcontext()
        with torch.no_grad(), ctx as h:
            if h is not None and hasattr(h, "reset"):
                h.reset(prompt_len=S)
            model.generate(input_ids=ids, attention_mask=am, **kw)

    one()  # warmup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    one()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"tag": ("fixed_generate_with_ctx" if obj is not None
                    else "fixed_generate"), "seq_len": S, "batch": B,
            "new_tokens": new_tokens,
            "per_step_ms": round(dt * 1e3 / new_tokens, 3),
            "per_seq_token_ms": round(dt * 1e3 / new_tokens / B, 4),
            "wall_s": round(dt, 2)}


def _time_fixed_generate_padded(model, tok, B, S_long, S_short, new_tokens,
                                device):
    """The PADDING cell (bare): half the rows are short prompts LEFT-padded to
    the long width — fixed output length (min == max), so this isolates the
    non-trivial-attention-mask/padding term alone. The real smoke's batches are
    left-padded; the clean fixed_generate cell is not. If this cell's per-step
    cost is many-fold the uniform cell's, the padded attention path (e.g. sdpa
    dropping to a slow kernel under a custom mask) is the bare-arm culprit."""
    import torch
    pad_id = tok.pad_token_id or 0
    half = max(B // 2, 1)
    ids = torch.full((B, S_long), int(pad_id), dtype=torch.long, device=device)
    attn = torch.zeros(B, S_long, dtype=torch.long, device=device)
    for r in range(B):
        L = S_long if r >= half else S_short
        row = torch.randint(0, int(model.config.vocab_size), (L,), device=device)
        ids[r, S_long - L:] = row
        attn[r, S_long - L:] = 1
    kw = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens,
              do_sample=False, pad_token_id=pad_id)
    with torch.no_grad():
        model.generate(input_ids=ids, attention_mask=attn, **kw)  # warmup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(input_ids=ids, attention_mask=attn, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return {"tag": "fixed_generate_padded", "seq_len": S_long,
            "seq_short": S_short, "batch": B, "new_tokens": new_tokens,
            "per_step_ms": round(dt * 1e3 / new_tokens, 3),
            "per_seq_token_ms": round(dt * 1e3 / new_tokens / B, 4),
            "wall_s": round(dt, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--revision", default="")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--phase",
                    choices=["prefill", "decode", "decode_companion",
                             "generate_long", "smoke_profile", "step_trace",
                             "warmup_scope", "backend_equiv", "asr_noise"],
                    default="decode")
    ap.add_argument("--n-goals", type=int, default=40,
                    help="asr_noise: number of real HarmBench goals to sweep")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="asr_noise generation cap (matches the smoke)")
    ap.add_argument("--family", default="artprompt",
                    choices=["artprompt", "renellm"],
                    help="asr_noise attack family (renellm = strong, mid-range)")
    ap.add_argument("--sdpa-backend", default="",
                    choices=["", "flash", "mem_efficient", "cudnn", "math"],
                    help="pin torch sdpa to ONE backend for the warmup_scope "
                         "plan (mechanism probe: 173917 measured a ~295 ms "
                         "cold tax per NEW (B, kv_len) attention shape — the "
                         "signature of cuDNN SDPA plan construction, which "
                         "torch enables by default on sm90 and caches per "
                         "problem geometry. If pinning flash erases the cold "
                         "plateau in a fresh process, the mechanism is NAMED "
                         "and the fix is a backend pin, not warmup "
                         "choreography). Empty = torch's default dispatch.")
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    import torch
    from sisteer.models import load_model_and_tokenizer

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(
        args.model, revision=(args.revision or None), dtype="bfloat16",
        attn_implementation="sdpa")
    model.eval()
    device = next(model.parameters()).device
    print(f"[bench] model loaded in {time.time()-t0:.1f}s on {device}")

    rows = []
    if args.phase == "step_trace":
        # MEASUREMENT (a): per-STEP walls inside real calls. The resolved
        # generation config is processor-free (read from config.json: no
        # no_repeat_ngram/repetition_penalty/stop_strings/warpers), so the
        # curve is the discriminator between the remaining candidates:
        # UNIFORMLY slow -> kernel/config difference; FIRST-STEPS-SLOW then
        # fast -> per-shape warmup/compile (every real call is a fresh (B, W);
        # the clean cell reused one warmed tensor); MONOTONICALLY DEGRADING ->
        # prefix-proportional per-step work. Each forward is
        # synchronize-bracketed (adds ~us/step; shapes the curve, not the
        # story). The SAME real chunk runs twice (cold then warm — a warm
        # repeat separates warmup from everything) plus a clean random-ids
        # control at the same (B, W, steps).
        import importlib.util as _ilu
        _vpp_path = Path(__file__).resolve().parent / "validate_phase_probe.py"
        _spec = _ilu.spec_from_file_location("validate_phase_probe", _vpp_path)
        _vpp = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_vpp)
        from sisteer.eval.attacks import get_attack
        from sisteer.eval.jailbreak import build_defense

        goals = _vpp._synthetic_goals(4)          # 8 prompts = one B=8 chunk
        run_fam = get_attack("artprompt")
        walls: list = []
        orig_fwd = model.forward

        def fwd(*a, **kw):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig_fwd(*a, **kw)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            walls.append(time.perf_counter() - t0)
            return out

        _summarise = _summarise_steps  # hoisted to module scope (warmup_scope shares it)

        model.forward = fwd
        try:
            defense = build_defense(model, tok, mode="none", positions="all",
                                    batch_size=8)
            for rep in ("cold", "warm"):
                walls.clear()
                t0 = time.perf_counter()
                run_fam(defense, goals,
                        out_dir=out_root / f"step_trace_{rep}" / "run",
                        max_new_tokens=256, seed=0, resume=False)
                row = _summarise(f"real_{rep}", list(walls))
                row["call_wall_s"] = round(time.perf_counter() - t0, 2)
                rows.append(row)
                print(f"[bench] real_{rep}: n_steps={row['n_steps']} "
                      f"prefill={row.get('prefill_ms')}ms "
                      f"first10={row.get('first10_mean_ms')}ms "
                      f"mid10={row.get('mid10_mean_ms')}ms "
                      f"last10={row.get('last10_mean_ms')}ms "
                      f"wall={row['call_wall_s']}s")
            # clean control at the same geometry: random ids, one warmed shape.
            ids = torch.randint(0, int(model.config.vocab_size), (8, 289),
                                device=next(model.parameters()).device)
            am = torch.ones_like(ids)
            kw = dict(max_new_tokens=256, min_new_tokens=256, do_sample=False,
                      pad_token_id=tok.pad_token_id)
            for rep in ("cold", "warm"):
                walls.clear()
                t0 = time.perf_counter()
                with torch.no_grad():
                    model.generate(input_ids=ids, attention_mask=am, **kw)
                row = _summarise(f"clean_{rep}", list(walls))
                row["call_wall_s"] = round(time.perf_counter() - t0, 2)
                rows.append(row)
                print(f"[bench] clean_{rep}: n_steps={row['n_steps']} "
                      f"prefill={row.get('prefill_ms')}ms "
                      f"first10={row.get('first10_mean_ms')}ms "
                      f"mid10={row.get('mid10_mean_ms')}ms "
                      f"last10={row.get('last10_mean_ms')}ms "
                      f"wall={row['call_wall_s']}s")

            # ---- STATIC-CACHE A/B: the fix test, same real inputs ----------
            # If the curve is LINEARLY_RISING, HF's default DynamicCache is the
            # cause: it torch.cats the whole accumulated K/V every decode step
            # (O(prefix)/step, O(n^2) total). A pre-allocated StaticCache writes
            # in place (O(1)/step). Run the SAME real rendered chunk both ways at
            # a FORCED 256 steps (min==max: identical step counts, worst-case
            # growth fully exercised, directly comparable curves) and (1) time
            # each per-step curve, (2) verify OUTPUT EQUIVALENCE token-for-token
            # (same greedy ids). A faster WRONG answer is worthless — the smoke
            # is the 'none' defense, so this path has no steering hooks; static
            # cache's interaction with hooks/variable widths is out of scope for
            # the reproduced 'none' gap and flagged in the report. B=8 mirrors
            # the real chunk; B=1 isolates the padding-free / batch-free 5.2x
            # floor and lets the slope be compared across B.
            from sisteer.eval.attacks.artprompt import artprompt_variants
            from sisteer.eval.jailbreak import build_chat_prompt
            from sisteer.generate import encode_kwargs
            raw = [p for g in goals for _, p in artprompt_variants(g, seed=0)]
            rendered = [build_chat_prompt(tok, r, system=None) for r in raw]

            def _ab_run(prompts, cache_impl):
                prev = tok.padding_side
                tok.padding_side = "left"
                try:
                    enc = tok(prompts, return_tensors="pt", padding=True,
                              **encode_kwargs(tok, prompts[0])).to(device)
                finally:
                    tok.padding_side = prev
                pw = int(enc["input_ids"].shape[1])
                gkw = dict(max_new_tokens=256, min_new_tokens=256,
                           do_sample=False, pad_token_id=tok.pad_token_id)
                if cache_impl is not None:
                    gkw["cache_implementation"] = cache_impl
                walls.clear()
                t_ab = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(**enc, **gkw)
                wall = time.perf_counter() - t_ab
                return out[:, pw:].detach().to("cpu"), wall, pw

            def _equiv(a, b):
                same_shape = list(a.shape) == list(b.shape)
                m = min(a.shape[1], b.shape[1])
                match = (a[:, :m] == b[:, :m])
                # first per-row step where dynamic and static diverge (256 = none)
                first_div = []
                for r in range(match.shape[0]):
                    ne = (~match[r]).nonzero()
                    first_div.append(int(ne[0]) if ne.numel() else m)
                return {"token_ids_identical":
                            bool(same_shape and match.all().item()),
                        "same_shape": same_shape,
                        "shape_dynamic": list(a.shape),
                        "shape_static": list(b.shape),
                        "frac_tokens_match": round(float(match.float().mean()), 4),
                        "min_first_divergence_step": min(first_div),
                        "first_divergence_per_row": first_div}

            for tagB, prompts in (("b8", rendered), ("b1", rendered[:1])):
                dyn_ids, dyn_wall, pw = _ab_run(prompts, None)
                row_d = _summarise(f"ab_dynamic_{tagB}", list(walls))
                row_d.update(call_wall_s=round(dyn_wall, 2), prompt_width=pw,
                             cache="dynamic")
                rows.append(row_d)
                stat_ids, stat_wall, pw2 = _ab_run(prompts, "static")
                row_s = _summarise(f"ab_static_{tagB}", list(walls))
                row_s.update(call_wall_s=round(stat_wall, 2), prompt_width=pw2,
                             cache="static")
                rows.append(row_s)
                eq = _equiv(dyn_ids, stat_ids)
                eq_row = {"tag": f"ab_equivalence_{tagB}",
                          "speedup_dynamic_over_static":
                              round(dyn_wall / max(stat_wall, 1e-9), 2),
                          "dynamic_wall_s": round(dyn_wall, 2),
                          "static_wall_s": round(stat_wall, 2), **eq}
                rows.append(eq_row)
                print(f"[bench] ab_{tagB}: dyn={dyn_wall:.1f}s "
                      f"(slope={row_d['decode_slope_ms_per_step']}ms/step "
                      f"{row_d['shape_verdict']}) "
                      f"static={stat_wall:.1f}s "
                      f"(slope={row_s['decode_slope_ms_per_step']}ms/step "
                      f"{row_s['shape_verdict']}) "
                      f"speedup={eq_row['speedup_dynamic_over_static']}x "
                      f"ids_identical={eq['token_ids_identical']} "
                      f"frac_match={eq['frac_tokens_match']}")
        finally:
            model.forward = orig_fwd
        verdict_key = "decode"  # no decode rows; verdicts read insufficient
    elif args.phase == "warmup_scope":
        # THE GLOBAL-vs-PER-SHAPE DISCRIMINATOR: step_trace identified
        # per-shape cold-start warmup as the whole real-vs-clean gap (36x
        # cold->warm on the SAME shape) but not its SCOPE. Its own data hints:
        # the two cold cells at matched (B, W) cost 133 s (ran FIRST) vs 85 s
        # (ran second) — ~36% cheaper, so part of the cost is GLOBAL and
        # amortized by any prior call. And cold calls stayed slow THROUGHOUT
        # (real_cold last10 = 380 ms), which fits a PER-KV-LENGTH key: every
        # decode step is a fresh kv length, so a cold call pays at every step
        # and the warm repeat reuses all 256 of them. Mechanisms imply
        # different fixes:
        #   GLOBAL (allocator arena under expandable_segments / CUDA context /
        #     cuBLAS workspace): ONE max-shape warmup generation at server start
        #     fixes every smaller shape — a few lines.
        #   PER-SHAPE / PER-KV-LENGTH (cuBLAS heuristic or kernel selection
        #     cached per exact shape): a warm pass per shape, so prompt-width
        #     bucketing becomes load-bearing scheduling work.
        # Protocol — ONE process, calls in ORDER (the order IS the experiment),
        # random ids, greedy, min==max=256 steps:
        #   A_cold_large   (8, 512): the LARGEST shape, deliberately first
        #                  (coldest; its decode covers kv lengths 512..768).
        #   A2_warm_large  (8, 512): warm repeat -> the warm floor.
        #   B_fresh_smaller(8, 289): FRESH strictly-smaller shape, never run.
        #                  Fast throughout => GLOBAL. Cold throughout =>
        #                  per-shape. Cold then dropping to the floor mid-call
        #                  => PER-KV-LENGTH (its kv lengths cross into A's
        #                  warmed 512..768 range at step 512-289 = 223).
        #   C_fresh_smallest(8, 160): second fresh shape — does cold cost keep
        #                  DECAYING across distinct shapes (more global
        #                  amortization) or hold constant (per-shape)?
        #   D_fresh_batch4 (4, 200): fresh shape with a DIFFERENT batch — is
        #                  the batch dim part of the cold key?
        # Memory is instrumented DIRECTLY (torch.cuda.memory_stats) around
        # every call — reserved-bytes growth concentrated in call A = allocator
        # arena (global, named not inferred); flat reserved with cold B/C =
        # kernel selection (per-shape). The sbatch runs this phase TWICE in
        # separate processes — PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        # (production setting, the arena-growth prime suspect) vs unset
        # (native segments) — since that env var changes exactly the suspected
        # mechanism and, if confirmed, is directly tunable.
        def _mem():
            if not torch.cuda.is_available():
                return {}
            s = torch.cuda.memory_stats()
            return {"reserved_mb": round(torch.cuda.memory_reserved() / 2**20),
                    "allocated_mb": round(torch.cuda.memory_allocated() / 2**20),
                    "num_alloc_retries": s.get("num_alloc_retries", 0),
                    "segments_current": s.get("segment.all.current", 0),
                    "segments_allocated": s.get("segment.all.allocated", 0)}

        walls: list = []
        orig_fwd = model.forward

        def fwd(*a, **kw):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig_fwd(*a, **kw)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            walls.append(time.perf_counter() - t0)
            return out

        steps = 256
        plan = [("A_cold_large", 8, 512), ("A2_warm_large", 8, 512),
                ("B_fresh_smaller", 8, 289), ("C_fresh_smallest", 8, 160),
                ("D_fresh_batch4", 4, 200)]
        sdpa_ctx = nullcontext()
        if args.sdpa_backend:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            _bk = {"flash": SDPBackend.FLASH_ATTENTION,
                   "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
                   "cudnn": SDPBackend.CUDNN_ATTENTION,
                   "math": SDPBackend.MATH}[args.sdpa_backend]
            sdpa_ctx = sdpa_kernel([_bk])
            print(f"[bench] sdpa pinned to {args.sdpa_backend} "
                  f"(cudnn {torch.backends.cudnn.version()})", flush=True)
        model.forward = fwd
        try:
            with sdpa_ctx:
                for tag, B, W in plan:
                    g = torch.Generator().manual_seed((B * 10007 + W) % 2**31)
                    ids = torch.randint(0, int(model.config.vocab_size), (B, W),
                                        generator=g).to(device)
                    am = torch.ones_like(ids)
                    kw = dict(max_new_tokens=steps, min_new_tokens=steps,
                              do_sample=False, pad_token_id=tok.pad_token_id)
                    m0 = _mem()
                    walls.clear()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        model.generate(input_ids=ids, attention_mask=am, **kw)
                    wall = time.perf_counter() - t0
                    m1 = _mem()
                    row = _summarise_steps(tag, list(walls))
                    row.update(
                        batch=B, prompt_width=W, call_wall_s=round(wall, 2),
                        alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
                        sdpa_backend=args.sdpa_backend or "default",
                        mem_before=m0, mem_after=m1,
                        reserved_delta_mb=(m1.get("reserved_mb", 0)
                                           - m0.get("reserved_mb", 0)),
                        segments_delta=(m1.get("segments_allocated", 0)
                                        - m0.get("segments_allocated", 0)),
                        alloc_retries_delta=(m1.get("num_alloc_retries", 0)
                                             - m0.get("num_alloc_retries", 0)))
                    rows.append(row)
                    print(f"[bench] {tag} B={B} W={W}: wall={wall:.1f}s "
                          f"first10={row.get('first10_mean_ms')}ms "
                          f"mid10={row.get('mid10_mean_ms')}ms "
                          f"last10={row.get('last10_mean_ms')}ms "
                          f"reserved {m0.get('reserved_mb')}->"
                          f"{m1.get('reserved_mb')}MB (+{row['reserved_delta_mb']}) "
                          f"segs+{row['segments_delta']} "
                          f"retries+{row['alloc_retries_delta']}", flush=True)
        finally:
            model.forward = orig_fwd
        verdict_key = "decode"  # no decode rows; verdicts read insufficient
    elif args.phase == "backend_equiv":
        # THE DISQUALIFIER CHECK for the flash-attention pin. warmup_scope
        # named the cold tax as cuDNN SDPA plan construction and 173943 showed
        # pinning flash erases it — but a backend pin must clear the SAME
        # output-equivalence bar that just killed static cache. flash and cuDNN
        # attention have different numerics, so switching WILL change some
        # greedy token ids. Generate the SAME real rendered artprompt chunk
        # under the DEFAULT dispatch (cuDNN on sm90 — what every existing result
        # in this lane ran on) and under a flash pin, greedy, min==max=256, and
        # compare token-for-token. Any divergence is DISQUALIFYING (a
        # faster-but-different fix, exactly as static cache was), independent of
        # the separate — and backend-agnostic — comparability cost (existing
        # results are byte-locked to the current backend).
        #
        # THE FIRST RUN (173977) RETURNED A FINDING, NOT A NUMBER: on the real
        # LEFT-PADDED B=8 chunk, the flash pin CRASHES — RuntimeError: No
        # available kernel — because a padded batch forces an explicit attention
        # mask and torch's flash SDPA kernel accepts none (warmup_scope's flash
        # arm ran only because its all-ones masks reduce to the is_causal fast
        # path). So the pin is INFEASIBLE for the production padded-batch path
        # via sdpa_kernel, which out-disqualifies mere divergence. This run
        # records that infeasibility as a RESULT (try/except), and measures id
        # equivalence where flash CAN run: B=1 (single row, all-ones mask ->
        # is_causal) and a UNIFORM B=8 (8 copies of one prompt, no padding).
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from sisteer.eval.attacks.artprompt import artprompt_variants
        from sisteer.eval.jailbreak import build_chat_prompt
        from sisteer.generate import encode_kwargs
        import importlib.util as _ilu
        _vpp_path = Path(__file__).resolve().parent / "validate_phase_probe.py"
        _spec = _ilu.spec_from_file_location("validate_phase_probe", _vpp_path)
        _vpp = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_vpp)
        goals = _vpp._synthetic_goals(4)
        raw = [p for g in goals for _, p in artprompt_variants(g, seed=0)]
        rendered = [build_chat_prompt(tok, r, system=None) for r in raw]

        def _gen(prompts, ctx):
            prev = tok.padding_side
            tok.padding_side = "left"
            try:
                enc = tok(prompts, return_tensors="pt", padding=True,
                          **encode_kwargs(tok, prompts[0])).to(device)
            finally:
                tok.padding_side = prev
            pw = int(enc["input_ids"].shape[1])
            gkw = dict(max_new_tokens=256, min_new_tokens=256, do_sample=False,
                       pad_token_id=tok.pad_token_id)
            t0 = time.perf_counter()
            with torch.no_grad(), ctx:
                out = model.generate(**enc, **gkw)
            return out[:, pw:].detach().to("cpu"), round(time.perf_counter()-t0, 2)

        cells = (("b8_padded", rendered),          # real chunk: flash expected INFEASIBLE
                 ("b8_uniform", [rendered[0]] * 8),  # no padding: flash can run
                 ("b1", rendered[:1]))
        for tagB, prompts in cells:
            # warm both paths once (equivalence is about ids, not timing, but a
            # warm pass keeps the compare honest and quick).
            _gen(prompts, nullcontext())
            def_ids, def_wall = _gen(prompts, nullcontext())          # default = cuDNN on sm90
            try:
                fl_ids, fl_wall = _gen(prompts, sdpa_kernel([SDPBackend.FLASH_ATTENTION]))
            except RuntimeError as exc:
                row = {"tag": f"backend_equiv_{tagB}",
                       "flash_infeasible": True,
                       "flash_error": str(exc).splitlines()[0][:120],
                       "default_wall_s": def_wall,
                       "cudnn_version": torch.backends.cudnn.version()}
                rows.append(row)
                print(f"[bench] backend_equiv_{tagB}: FLASH INFEASIBLE "
                      f"({row['flash_error']})", flush=True)
                continue
            m = min(def_ids.shape[1], fl_ids.shape[1])
            match = (def_ids[:, :m] == fl_ids[:, :m])
            first_div = []
            for r in range(match.shape[0]):
                ne = (~match[r]).nonzero()
                first_div.append(int(ne[0]) if ne.numel() else m)
            row = {"tag": f"backend_equiv_{tagB}",
                   "flash_infeasible": False,
                   "token_ids_identical": bool(match.all().item()
                                               and def_ids.shape == fl_ids.shape),
                   "frac_tokens_match": round(float(match.float().mean()), 4),
                   "min_first_divergence_step": min(first_div),
                   "first_divergence_per_row": first_div,
                   "default_wall_s": def_wall, "flash_wall_s": fl_wall,
                   "cudnn_version": torch.backends.cudnn.version()}
            rows.append(row)
            print(f"[bench] backend_equiv_{tagB}: ids_identical="
                  f"{row['token_ids_identical']} frac_match={row['frac_tokens_match']} "
                  f"first_div={row['min_first_divergence_step']} "
                  f"(default={def_wall}s flash={fl_wall}s)", flush=True)
        verdict_key = "decode"  # no decode rows; verdicts read insufficient
    elif args.phase == "asr_noise":
        # ASR NOISE FLOOR under a SEMANTICALLY-NULL numerical perturbation.
        # backend_equiv showed this pipeline's greedy generations are extremely
        # numerically sensitive (frac_match 0.037, diverge @ step 7 for flash vs
        # cuDNN). Our OWN production path carries the same live variable: batch
        # composition. B=1 vs B=N genuinely differs numerically (batched-GEMM
        # reductions + left padding), and run_static's chunk composition depends
        # on goal count and on where a resume picks up — so two legitimate runs
        # of the same condition that chunk differently produce different
        # generations by exactly this mechanism. QUESTION: does the AGGREGATE
        # ASR (and each goal's jailbreak verdict) move under B=1 vs B=8? This
        # phase does the GPU half: ONE family (artprompt, deterministic, no
        # attacker) x ONE condition (none, baseline — no compiled object) on N
        # REAL HarmBench goals, greedy, generated TWICE (B=1 and B=8), capturing
        # token ids + text per (goal, variant) and the token-level agreement
        # (the POSITIVE CONTROL: expected LOW, confirming the perturbation
        # actually bit). StrongREJECT grading + the ASR/per-goal-flip comparison
        # is the grading-host post-step (scripts/asr_noise_grade.py, Low-Prio
        # judge) — real harmful completions never leave the 0700 private dir
        # except to the grader.
        from sisteer.eval.jailbreak_data import load_harmbench, HARMBENCH_REVISION
        from sisteer.eval.jailbreak import build_chat_prompt
        from sisteer.generate import encode_kwargs, assert_left_padded
        from sisteer.eval.jailbreak_artifacts import private_umask

        # --family selects the attack. artprompt (weak here, floors ASR) vs
        # renellm (a strong frozen-corpus attack, for a mid-range operating point
        # — reads its corpus via SISTEER_HELDOUT_CORPUS_DIR; pure replay).
        if args.family == "renellm":
            from sisteer.eval.attacks.renellm import (renellm_variants as
                                                      _variants_fn, _corpus)
            if _corpus() is None:
                raise SystemExit("[asr_noise] renellm frozen corpus not found — "
                                 "set SISTEER_HELDOUT_CORPUS_DIR to its dir")
        else:
            from sisteer.eval.attacks.artprompt import (artprompt_variants as
                                                        _variants_fn)
        # take the first n_goals HarmBench goals the family actually COVERS
        # (renellm returns [] for an uncovered goal; artprompt covers all).
        all_goals = load_harmbench(split="test", functional="standard")
        goals, work = [], []  # work: (goal_id, category, variant_idx, goal_text, rendered)
        for g in all_goals:
            vs = list(_variants_fn(g, seed=0))
            if not vs:
                continue
            goals.append(g)
            for i, (_name, prompt) in enumerate(vs):
                work.append((g.goal_id, g.category, i, g.text,
                             build_chat_prompt(tok, prompt, system=None)))
            if len(goals) >= args.n_goals:
                break
        rendered = [w[4] for w in work]
        print(f"[bench] asr_noise family={args.family}: {len(goals)} goals, "
              f"{len(work)} prompts ({len(work)/max(len(goals),1):.1f}/goal)",
              flush=True)

        def _gen(prompts, bs):
            """Generate at batch size bs (left-padded chunks, greedy); return
            (generated_ids_list, text) per prompt in order — the production
            generate_batch path, but keeping the ids for the token compare."""
            outs = []
            prev = tok.padding_side
            tok.padding_side = "left"
            try:
                for i in range(0, len(prompts), bs):
                    chunk = prompts[i:i + bs]
                    enc = tok(chunk, return_tensors="pt", padding=True,
                              **encode_kwargs(tok, chunk[0])).to(device)
                    assert_left_padded(enc.get("attention_mask"))
                    pw = int(enc["input_ids"].shape[1])
                    gkw = dict(max_new_tokens=args.max_new_tokens,
                               do_sample=False, pad_token_id=tok.pad_token_id)
                    with torch.no_grad():
                        out = model.generate(**enc, **gkw)
                    for row_ids in out[:, pw:]:
                        gen = row_ids.tolist()
                        text = tok.decode(row_ids, skip_special_tokens=True).strip()
                        outs.append((gen, text))
            finally:
                tok.padding_side = prev
            return outs

        t_b1 = time.time()
        b1 = _gen(rendered, 1)
        print(f"[bench] asr_noise B=1: {len(b1)} gens in {time.time()-t_b1:.1f}s",
              flush=True)
        t_b8 = time.time()
        b8 = _gen(rendered, 8)
        print(f"[bench] asr_noise B=8: {len(b8)} gens in {time.time()-t_b8:.1f}s",
              flush=True)

        def _tok_agree(a, b):
            # drop trailing eos/pad-equal tails by comparing over min real len
            m = min(len(a), len(b))
            if m == 0:
                return {"len_b1": len(a), "len_b8": len(b),
                        "frac_match": 0.0, "first_div": 0}
            match = [a[i] == b[i] for i in range(m)]
            first = next((i for i, ok in enumerate(match) if not ok), m)
            return {"len_b1": len(a), "len_b8": len(b),
                    "frac_match": round(sum(match) / m, 4), "first_div": first}

        # persist the per-(goal,variant) responses to a 0700 private dir for the
        # grading host grader; the sanitised aggregate goes in the summary rows.
        priv = out_root / "asr_noise_private"
        with private_umask():
            priv.mkdir(parents=True, exist_ok=True)
        try:
            priv.chmod(0o700)
        except OSError:
            pass
        resp_path = priv / "responses.jsonl"
        agrees = []
        with resp_path.open("w") as f:
            for (gid, cat, vidx, gtext, _r), (i1, t1), (i8, t8) in zip(work, b1, b8):
                ag = _tok_agree(i1, i8)
                agrees.append(ag["frac_match"])
                f.write(json.dumps({
                    "goal_id": gid, "category": cat, "variant_idx": vidx,
                    "goal_text": gtext, "dataset": "harmbench",
                    "revision": HARMBENCH_REVISION,
                    "response_b1": t1, "response_b8": t8,
                    "tok_frac_match": ag["frac_match"],
                    "tok_first_div": ag["first_div"],
                    "len_b1": ag["len_b1"], "len_b8": ag["len_b8"]}) + "\n")
        mean_ag = round(sum(agrees) / len(agrees), 4) if agrees else None
        rows.append({"tag": "asr_noise_generation",
                     "n_goals": len(goals), "n_prompts": len(work),
                     "variants_per_goal": round(len(work) / max(len(goals), 1), 2),
                     "family": args.family,
                     "condition": "none", "max_new_tokens": args.max_new_tokens,
                     "mean_token_agreement_b1_vs_b8": mean_ag,
                     "responses_path": str(resp_path),
                     "note": ("token agreement is the POSITIVE CONTROL (low => "
                              "perturbation bit); ASR + per-goal flips are the "
                              "grading host StrongREJECT post-step, asr_noise_grade.py")})
        print(f"[bench] asr_noise: {len(work)} prompts, mean token agreement "
              f"B1-vs-B8 = {mean_ag}; responses -> {resp_path}", flush=True)
        verdict_key = "decode"  # no decode rows; verdicts read insufficient
    elif args.phase == "smoke_profile":
        # ATTRIBUTION PROFILE of the REAL smoke path: every synthetic candidate
        # is measured and dead (model flat, steering +1-2%, padding <=+20%/72%
        # at 50% upper-bound, context 2.35x, resume re-scan structurally absent,
        # NFS fsync 5 ms/row) yet the real arm pays ~2.2 s/prompt at B=1 over
        # clean compute. Replay the real artprompt smoke (synthetic benign
        # goals, real 256-token cap, real run_static/defense.generate/
        # generate_batch/chat-template path) with model.generate and
        # defense.generate BOTH wrapped, so the first split — inside HF generate
        # vs everything around it — is decisive per call.
        import importlib.util as _ilu
        import sys as _sys
        _vpp_path = Path(__file__).resolve().parent / "validate_phase_probe.py"
        _spec = _ilu.spec_from_file_location("validate_phase_probe", _vpp_path)
        _vpp = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_vpp)
        from sisteer.eval.attacks import get_attack
        from sisteer.eval.jailbreak import build_defense

        goals = _vpp._synthetic_goals(10)
        run_fam = get_attack("artprompt")
        mg_calls = []
        orig_generate = model.generate

        def mg_shim(*a, **kw):
            ids = kw.get("input_ids", a[0] if a else None)
            t0 = time.perf_counter()
            out = orig_generate(*a, **kw)
            wall = time.perf_counter() - t0
            seqs = out.sequences if hasattr(out, "sequences") else out
            mg_calls.append({
                "B": int(ids.shape[0]), "prompt_width": int(ids.shape[1]),
                "new_tokens": int(seqs.shape[1] - ids.shape[1]),
                "model_generate_s": round(wall, 3)})
            return out

        for bs in (1, 8):
            model.generate = mg_shim
            try:
                defense = build_defense(model, tok, mode="none",
                                        positions="all", batch_size=bs)
                dg_calls = []
                orig_dg = defense.generate

                def dg_shim(prompts, **kw):
                    t0 = time.perf_counter()
                    out2 = orig_dg(prompts, **kw)
                    dg_calls.append({"n_prompts": len(prompts),
                                     "defense_generate_s":
                                         round(time.perf_counter() - t0, 3)})
                    return out2

                defense.generate = dg_shim
                mg_calls.clear()
                t_arm = time.perf_counter()
                run_fam(defense, goals,
                        out_dir=out_root / f"smoke_profile_b{bs}" / "run",
                        max_new_tokens=256, seed=0)
                arm_wall = time.perf_counter() - t_arm
            finally:
                model.generate = orig_generate
            mg_total = sum(c["model_generate_s"] for c in mg_calls)
            dg_total = sum(c["defense_generate_s"] for c in dg_calls)
            gen_tokens = sum(c["new_tokens"] * c["B"] for c in mg_calls)
            row = {"tag": "smoke_profile", "batch": bs,
                   "arm_wall_s": round(arm_wall, 2),
                   "model_generate_total_s": round(mg_total, 2),
                   "defense_generate_total_s": round(dg_total, 2),
                   "wrapper_s (defense - model)": round(dg_total - mg_total, 2),
                   "outside_defense_s (arm - defense)":
                       round(arm_wall - dg_total, 2),
                   "n_model_generate_calls": len(mg_calls),
                   "total_generated_tokens": gen_tokens,
                   "ms_per_generated_token_inside_model_generate":
                       round(mg_total * 1e3 / max(gen_tokens, 1), 2),
                   # COPIES, not references: mg_calls is cleared and refilled
                   # per arm, so storing the list object aliased every arm's
                   # detail rows to the LAST arm's contents (caught in the
                   # 173765 artifact — aggregates were computed before the
                   # clear and were unaffected).
                   "model_generate_calls": list(mg_calls),
                   "defense_generate_calls": list(dg_calls)}
            rows.append(row)
            print(f"[bench] smoke_profile B={bs}: arm={arm_wall:.1f}s "
                  f"model.generate={mg_total:.1f}s "
                  f"wrapper={dg_total - mg_total:.1f}s "
                  f"outside={arm_wall - dg_total:.1f}s "
                  f"gen_tokens={gen_tokens} "
                  f"({mg_total * 1e3 / max(gen_tokens, 1):.1f} ms/tok inside)")
        verdict_key = "decode"  # no decode rows; verdicts read insufficient
    elif args.phase == "generate_long":
        # The missing cell: the REAL generate path has never been timed at the
        # long-instruction geometry — the S=6700 decode grid's absolutes are
        # inflated ~24x by the manual past_key_values drive and only their
        # scaling is diagnostic. Fixed-length generate at S=6700, bare + steered
        # twins, B in {1,8}; plus ONE padded long cell for symmetry (half the
        # rows 3350 -> padded to 6700; ~50% padding, an aggressive DETECTOR — a
        # positive here is an UPPER BOUND on the real ~17% effect, not an
        # estimate of it). Predicted ~1.36x per-step / ~5.9x throughput at B=8;
        # ~1.4x KILLS the sequence-length candidate cleanly and leaves padding
        # + steering to fight over the whole real-smoke gap.
        from sisteer.methods import VectorObject
        from sisteer.vectors import SteeringVector
        g = torch.Generator().manual_seed(0)
        v = torch.randn(model.config.hidden_size, generator=g)
        v = v / v.norm()
        sv = SteeringVector(vectors={args.layer: v}, method="bench",
                            model_name="", instruction_id="")
        obj = VectorObject(sv, positions="all", skip_bos=True)
        for B in (1, 8):
            r = _time_fixed_generate(model, tok, B, 6700, 32, device)
            rows.append(r)
            print(f"[bench] fixed_generate S=6700 B={B}: {r['per_step_ms']} "
                  f"ms/step  {r['per_seq_token_ms']} ms/seq-token")
            r2 = _time_fixed_generate(model, tok, B, 6700, 32, device, obj=obj)
            rows.append(r2)
            print(f"[bench] fixed_generate_with_ctx S=6700 B={B}: "
                  f"{r2['per_step_ms']} ms/step  "
                  f"{r2['per_seq_token_ms']} ms/seq-token")
        r3 = _time_fixed_generate_padded(model, tok, 8, 6700, 3350, 32, device)
        rows.append(r3)
        print(f"[bench] fixed_generate_padded S=6700/3350 B=8: "
              f"{r3['per_step_ms']} ms/step  {r3['per_seq_token_ms']} "
              f"ms/seq-token")
        verdict_key = "decode"  # no decode rows here; verdicts read insufficient
    elif args.phase == "prefill":
        rows += _grid(model, device, [1, 2, 4, 8, 16], [300], args.reps,
                      args.warmup, tag="no_ctx")
        rows += _grid(model, device, [1, 2, 4, 8], [6700], args.reps,
                      args.warmup, tag="no_ctx")
        from sisteer.methods import VectorObject
        from sisteer.vectors import SteeringVector
        g = torch.Generator().manual_seed(0)
        v = torch.randn(model.config.hidden_size, generator=g); v = v / v.norm()
        sv = SteeringVector(vectors={args.layer: v}, method="bench",
                            model_name="", instruction_id="")
        obj = VectorObject(sv, positions="all", skip_bos=True)
        rows += _grid(model, device, [1, 8], [300], args.reps, args.warmup,
                      obj=obj, tag="with_ctx")
        verdict_key = "no_ctx"
    else:
        # THE decision phase: decode steps against a real KV cache. More timed
        # reps — a step is milliseconds.
        d_reps, d_warm = max(args.reps, 20), max(args.warmup, 3)
        rows += _decode_grid(model, device, [1, 2, 8], [300, 6700], d_reps,
                             d_warm, tag="decode")
        if args.phase == "decode_companion":
            # THE FOUR-CELL COMPARISON: the same decode cells WITH the steering
            # ctx installed — bare and steered differing in exactly one thing
            # (the SAME VectorObject construction as the prefill branch). The
            # discriminator is the DIFFERENCE of the two ratios, not either
            # verdict alone: bare-flat + ctx-linear -> the per-row steering code
            # is the whole cost (vectorise it); both-linear -> decode genuinely
            # does not batch here; both-flat -> the cost is somewhere neither
            # bench looked (per-step Python in generate_batch, logits
            # processing, detokenisation) — to be said plainly.
            from sisteer.methods import VectorObject
            from sisteer.vectors import SteeringVector
            g = torch.Generator().manual_seed(0)
            v = torch.randn(model.config.hidden_size, generator=g)
            v = v / v.norm()
            sv = SteeringVector(vectors={args.layer: v}, method="bench",
                                model_name="", instruction_id="")
            obj = VectorObject(sv, positions="all", skip_bos=True)
            rows += _decode_grid(model, device, [1, 8], [300, 6700], d_reps,
                                 d_warm, obj=obj, tag="decode_with_ctx")
        # is the cache actually used by the REAL serving path on this stack?
        instr = _instrument_cache_use(model, tok)
        rows.append({"tag": "cache_instrumentation", **instr})
        print(f"[bench] kv_cache_used={instr['kv_cache_used']} "
              f"calls={[c['shape'] for c in instr['calls']]}")
        # end-to-end corroboration: fixed-length generate, zero length variance
        # (+ the steered twin in companion mode).
        for B in (1, 8):
            r = _time_fixed_generate(model, tok, B, 300, 32, device)
            rows.append(r)
            print(f"[bench] fixed_generate S=300 B={B}: {r['per_step_ms']} "
                  f"ms/step  {r['per_seq_token_ms']} ms/seq-token")
            if args.phase == "decode_companion":
                r2 = _time_fixed_generate(model, tok, B, 300, 32, device,
                                          obj=obj)
                rows.append(r2)
                print(f"[bench] fixed_generate_with_ctx S=300 B={B}: "
                      f"{r2['per_step_ms']} ms/step  "
                      f"{r2['per_seq_token_ms']} ms/seq-token")
        if args.phase == "decode_companion":
            # the padding cell (bare, B=8): the one thing the clean cell lacks
            # vs the real smoke's batches. Uniform-vs-padded at fixed output.
            r3 = _time_fixed_generate_padded(model, tok, 8, 300, 150, 32, device)
            rows.append(r3)
            print(f"[bench] fixed_generate_padded S=300/150 B=8: "
                  f"{r3['per_step_ms']} ms/step  "
                  f"{r3['per_seq_token_ms']} ms/seq-token")
        verdict_key = "decode"

    def _v(seq_len):
        at = {r["batch"]: r for r in rows
              if r.get("seq_len") == seq_len and "per_forward_ms" in r
              and r["tag"] == verdict_key}
        if 1 not in at or 8 not in at:
            return {"seq_len": seq_len, "verdict": "insufficient_data"}
        ratio = at[8]["per_forward_ms"] / at[1]["per_forward_ms"]
        if ratio < 2.0:
            v = "FLAT_memory_bound__loss_is_padding__keep_and_tune_batching"
        elif ratio > 5.0:
            v = "LINEAR_compute_bound__batching_cannot_win__tune_generation_length_only"
        else:
            v = "PARTIAL__batching_helps_sublinearly"
        return {"seq_len": seq_len, "ratio_B8_over_B1": round(ratio, 2),
                "verdict": v}

    summary = {
        "model": args.model, "attn": "sdpa", "dtype": "bfloat16",
        "phase": args.phase, "reps": args.reps, "warmup": args.warmup,
        "rows": rows,
        "verdict_short_S300": _v(300),
        "verdict_long_S6700": _v(6700),
    }
    if args.phase == "decode_companion":
        verdict_key = "decode_with_ctx"
        summary["verdict_with_ctx_S300"] = _v(300)
        summary["verdict_with_ctx_S6700"] = _v(6700)
    out = out_root / f"bench_summary_{args.phase}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[bench] wrote {out}")
    print(f"[bench] VERDICT short: {summary['verdict_short_S300']}")
    print(f"[bench] VERDICT long:  {summary['verdict_long_S6700']}")


if __name__ == "__main__":
    main()
