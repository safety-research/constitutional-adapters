"""Measure the teacher prefix KV cache on a real si_constitution batch.

The acceptance measurement for the 2026-08-24 teacher-prefix-cache change:
one MEASURED before/after tokens-per-second comparison of the TEACHER LEG
on the production shape — the pinned Llama-3.1-8B mirror, the real
si_constitution system text (5,872 SI tokens rendered into every teacher
prompt), bf16, sdpa, the wave's max_len and a production batch width — not
an estimate. Prompts and completions are SYNTHETIC (invented benchmark
text; never real store data): the throughput of the teacher forward
depends on token counts, not token content.

Both paths run the identical workload — the same collated teacher batches
— so "tokens/s" is EFFECTIVE throughput on the same work: total real
(attention-mask) tokens the teacher leg must produce logits for, divided
by wall clock. The cached path's clock includes its one prefix build.

Also prints the bf16 parity numbers (max/mean abs gathered-logit diff and
the KL between the two teacher distributions) so the fp-tolerance claim is
measured on the real model, not only on the CPU fp32 fixture.

MEMORY INSTRUMENTATION POST-MORTEM (2026-08-24, job 190490 VOID): the
first version of this bench kept every batch's gathered logits in a list
for the parity comparison, and ran the cached pass while the uncached
pass's list was STILL LIVE — ``reset_peak_memory_stats`` resets the
statistic, not the allocations, so the cached peak included the entire
retained uncached corpus, and the "delta" scaled with n = batches x
batch_size rather than with the prefix KV. The memory numbers it printed
were invalid (its throughput and parity numbers were not: retention
costs memory, not time, and parity read the same tensors either way).
The fix, and the standing rules for any memory pass in this file:

- MEASURE and PARITY are separate passes. The measured passes consume
  and discard each batch's output (scalar checksums only); parity runs
  its own streaming pass afterwards, retaining scalars only.
- Between paths: free everything, ``empty_cache``, synchronize, and
  ASSERT ``memory_allocated`` is back at the pre-path baseline — a
  memory bench that can be contaminated by its own harness must detect
  that condition itself, loudly.
- VALIDATE THE INSTRUMENT against a known quantity before believing it:
  the uncached path is measured TWICE and the two peaks must agree —
  the check that would have caught the retained-corpus bug immediately
  (a lint cannot: nothing was undefined; it was a live-object lifetime
  bug).

Run (cluster, one GPU):
    python scripts/bench_prefix_cache.py --config configs/two_si.yaml \
        --si si_constitution --batches 32 --batch-size 4
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sisteer.eval.jailbreak import resolve_instruction  # noqa: E402
from sisteer.eval.train_budget import resolve_max_len  # noqa: E402
from sisteer.extraction.distill import (  # noqa: E402
    DistillExample, _encode_aligned, _gather_completion_logits)
from sisteer.extraction.prefix_cache import (  # noqa: E402
    prefix_cache_for_batches)
from sisteer.extraction.traindata import collate  # noqa: E402
from sisteer.models import build_chat_prompt, load_model_and_tokenizer  # noqa: E402

# Invented benchmark text only. The suffix filler puts the attack-prompt
# allowance in a realistic band (~100-200 tokens); the completion filler
# fills the shared 512-token completion cap (truncated by _encode_aligned's
# avail arithmetic exactly as production truncates).
_SUFFIX_FILLER = (
    "considering every stated obligation carefully, including edge cases, "
    "counterexamples, and the reasoning behind each numbered rule, ")
_COMPLETION_FILLER = (
    "I need to decline that part of the request, but here is the safe and "
    "careful context I can offer instead, step by step. ")


def _examples(tok, instruction, n: int) -> list[DistillExample]:
    exs = []
    for i in range(n):
        prompt = (f"Benchmark suffix {i}: please restate your obligations, "
                  f"{_SUFFIX_FILLER * 2} and then answer question {i}.")
        exs.append(DistillExample(
            prompt_with_si=build_chat_prompt(tok, prompt,
                                             system=instruction.text),
            prompt_without=build_chat_prompt(tok, prompt, system=None),
            completion=_COMPLETION_FILLER * 16,
        ))
    return exs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/two_si.yaml")
    ap.add_argument("--si", default="si_constitution")
    ap.add_argument("--batches", type=int, default=32)
    # Comma list: peak-memory must be re-measured at every production width
    # the wave configures (the batch expansion is materialised as a
    # per-batch COPY by transformers 5.14.1, ~B x P x 128KiB transient, so
    # memory scales with B and the widest width is the one that matters).
    ap.add_argument("--batch-sizes", default="4",
                    help="comma-separated batch widths to measure")
    ap.add_argument("--warmup-batches", type=int, default=2)
    # dtype override: an fp32 control run separates bf16 accumulation noise
    # (collapses by orders of magnitude at fp32) from a real misalignment
    # (dtype-independent) — the decisive read on the raw-logit diff tail.
    ap.add_argument("--dtype", default=None,
                    help="override the config dtype (e.g. float32)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype or cfg.get("dtype", "bfloat16"))
    print(f"[bench] loading {cfg['model']} @ {cfg.get('model_revision')} "
          f"({dtype}, {cfg.get('attn_implementation', 'sdpa')}, {device})",
          flush=True)
    model, tok = load_model_and_tokenizer(
        cfg["model"], device=device, dtype=dtype,
        revision=cfg.get("model_revision"),
        attn_implementation=cfg.get("attn_implementation", "sdpa"))

    instruction = resolve_instruction(args.si)
    max_len = resolve_max_len(cfg, args.si)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]

    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()

    for bs in batch_sizes:
        _run_width(model, tok, instruction, max_len, args, bs, device,
                   _sync)


def _run_width(model, tok, instruction, max_len, args, batch_size, device,
               _sync) -> None:
    # The parity print names the dtype the model REALLY runs in — derived
    # from the loaded weights, not re-plumbed from main (whose local went
    # out of scope in the width-loop refactor; job 190473's NameError).
    dtype = next(model.parameters()).dtype
    n = args.batches * batch_size
    exs = _examples(tok, instruction, n)
    encoded = [_encode_aligned(tok, ex, max_len)[0] for ex in exs]
    batches = [
        collate(encoded[i:i + batch_size], tok.pad_token_id).to(device)
        for i in range(0, len(encoded), batch_size)
    ]
    workload_tokens = int(sum(int(b.attention_mask.sum()) for b in batches))
    widths = [int(b.input_ids.shape[1]) for b in batches]
    print(f"[bench] si={args.si} max_len={max_len} "
          f"{len(batches)} batches x bs={batch_size}, "
          f"seq width {min(widths)}..{max(widths)}, "
          f"workload {workload_tokens} real tokens", flush=True)

    # -- measured passes: consume-and-discard, scalar checksums only ------
    # (the module docstring's post-mortem: retaining outputs contaminated
    # the NEXT path's peak — job 190490's memory numbers were void)

    def run_uncached_mem(bs) -> float:
        s = 0.0
        with torch.no_grad():
            for tb in bs:
                out = _gather_completion_logits(
                    model(input_ids=tb.input_ids,
                          attention_mask=tb.attention_mask).logits, tb)
                # fp32-accumulated reduction, no materialised cast copy.
                s += float(out.sum(dtype=torch.float32))
                del out
        return s

    def run_cached_mem(bs):
        # The prefix build is INSIDE the timed region on purpose — the
        # honest cost of the cached path is build + replays. The returned
        # tpc is the caller's to free before the next path's baseline
        # assertion.
        tpc = prefix_cache_for_batches(model, bs)
        assert tpc is not None, "no shared prefix — cache not engaged"
        s = 0.0
        for tb in bs:
            out = tpc.completion_logits(model, tb)
            s += float(out.sum(dtype=torch.float32))
            del out
        return s, tpc

    def _reset_peak():
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def _peak_gib() -> float:
        if device != "cuda":
            return float("nan")
        return torch.cuda.max_memory_allocated() / 2**30

    def _free_and_assert_baseline(tag: str, baseline: int | None) -> None:
        """Free dead objects and REFUSE to proceed if live memory has not
        returned to the pre-path baseline — the self-check that makes
        harness contamination a loud failure instead of a wrong number."""
        gc.collect()
        if device != "cuda":
            return
        torch.cuda.empty_cache()
        _sync()
        if baseline is None:
            return
        live = torch.cuda.memory_allocated()
        if live > baseline + 256 * 2**20:
            raise RuntimeError(
                f"[bench] {tag}: {live / 2**30:.2f} GiB live vs baseline "
                f"{baseline / 2**30:.2f} GiB — a previous path's outputs "
                "are still held and would contaminate the next peak")

    # Warmup both paths (kernel selection/autotune), outside the clocks,
    # then establish the model+staging baseline every pass must return to.
    warm = batches[: args.warmup_batches]
    run_uncached_mem(warm)
    _ck, _tpc = run_cached_mem(warm)
    del _tpc
    _free_and_assert_baseline("post-warmup", None)
    baseline = (torch.cuda.memory_allocated() if device == "cuda" else None)

    # Pass 1: uncached (timed + peak).
    _reset_peak()
    t0 = time.perf_counter()
    ck_plain = run_uncached_mem(batches)
    _sync()
    t_plain = time.perf_counter() - t0
    peak_plain = _peak_gib()
    _free_and_assert_baseline("after uncached pass", baseline)

    # Pass 2: uncached AGAIN — the instrument validation. The same path
    # measured twice must read the same peak (and checksum); if it does
    # not, the harness is measuring itself and no reading is believed.
    _reset_peak()
    ck_valid = run_uncached_mem(batches)
    _sync()
    peak_valid = _peak_gib()
    _free_and_assert_baseline("after validation pass", baseline)
    if device == "cuda":
        tol = max(0.02 * peak_plain, 0.25)
        if abs(peak_valid - peak_plain) > tol:
            raise RuntimeError(
                f"[bench] instrument check FAILED: uncached peaks "
                f"{peak_plain:.2f} vs {peak_valid:.2f} GiB differ by more "
                f"than {tol:.2f} GiB — the peak reading is not "
                "reproducible; no memory number from this run is valid")
        if abs(ck_valid - ck_plain) > 1e-3 * max(abs(ck_plain), 1.0):
            raise RuntimeError(
                f"[bench] instrument check FAILED: uncached checksums "
                f"{ck_plain!r} vs {ck_valid!r} differ — the two passes did "
                "not run the same computation")
        print(f"[bench] instrument check (bs={batch_size}): uncached peak "
              f"{peak_plain:.2f} GiB reproduced at {peak_valid:.2f} GiB, "
              f"checksums agree — readings believed", flush=True)

    # Pass 3: cached (timed + peak).
    _reset_peak()
    t0 = time.perf_counter()
    ck_cached, tpc = run_cached_mem(batches)
    _sync()
    t_cached = time.perf_counter() - t0
    peak_cached = _peak_gib()
    p = len(tpc)
    del tpc
    _free_and_assert_baseline("after cached pass", baseline)

    # -- parity: its own STREAMING pass, scalars retained only ------------
    tpc = prefix_cache_for_batches(model, batches)
    max_abs = mean_abs = kl = max_tv = 0.0
    n_comp = 0
    q999 = 0.0
    n_rows = 0
    with torch.no_grad():
        for tb in batches:
            plain = _gather_completion_logits(
                model(input_ids=tb.input_ids,
                      attention_mask=tb.attention_mask).logits, tb)
            cached = tpc.completion_logits(model, tb)
            d = (cached.float() - plain.float()).abs()
            max_abs = max(max_abs, float(d.max()))
            mean_abs += float(d.sum())
            n_comp += d.numel()
            q999 = max(q999, float(torch.quantile(
                d.flatten()[torch.randperm(d.numel(),
                                           device=d.device)[:2_000_000]],
                0.999)))
            pa = torch.softmax(cached.float(), -1)
            pb = torch.softmax(plain.float(), -1)
            # Per-position total variation between the two teacher
            # DISTRIBUTIONS — the semantically scaled parity number (a raw
            # bf16 logit diff on a -30 tail token moves no probability
            # mass).
            max_tv = max(max_tv, float((pa - pb).abs().sum(-1).max()) / 2)
            kl += float(F.kl_div(torch.log_softmax(cached.float(), -1),
                                 torch.log_softmax(plain.float(), -1),
                                 log_target=True, reduction="batchmean")
                        ) * cached.shape[0]
            n_rows += cached.shape[0]
            del plain, cached, d, pa, pb
    mean_abs /= max(n_comp, 1)
    kl /= max(n_rows, 1)
    del tpc
    _free_and_assert_baseline("after parity pass", baseline)

    tps_plain = workload_tokens / t_plain
    tps_cached = workload_tokens / t_cached
    print(f"[bench] prefix length P={p} tokens "
          f"({p / (workload_tokens / n):.0%} of the mean row)", flush=True)
    print(f"[bench] UNCACHED: {t_plain:8.2f}s  "
          f"{tps_plain:12.0f} tokens/s", flush=True)
    print(f"[bench] CACHED:   {t_cached:8.2f}s  "
          f"{tps_cached:12.0f} tokens/s (effective, same workload; "
          f"includes one prefix build)", flush=True)
    print(f"[bench] SPEEDUP:  {t_plain / t_cached:.2f}x", flush=True)
    print(f"[bench] PEAK MEM (bs={batch_size}): "
          f"uncached {peak_plain:.2f} GiB, cached {peak_cached:.2f} GiB, "
          f"delta {peak_cached - peak_plain:+.2f} GiB "
          f"(consume-and-discard passes over a "
          f"{baseline / 2**30 if baseline else float('nan'):.2f} GiB "
          f"model+staging baseline; instrument-validated)", flush=True)
    print(f"[bench] parity (gathered completion logits, {dtype}): "
          f"max|diff|={max_abs:.4g} p99.9|diff|={q999:.4g} "
          f"mean|diff|={mean_abs:.4g}", flush=True)
    print(f"[bench] parity (distribution level): "
          f"KL(cached||uncached)={kl:.4g} "
          f"max per-position TV={max_tv:.4g}", flush=True)


if __name__ == "__main__":
    main()
