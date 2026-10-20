"""Real-trainer-path peak-memory probe (accel-fit 2026-08-21).

Supersedes the synthetic shape in gpu_memprobe.py, which put the WITH-GRAD
student forward at the full max_len — but the real trainer's student is
SI-ABSENT (a short attack prompt + completion); the long ~5.9k-token
sequence is the TEACHER (SI present), which runs under no_grad and frees
its activations. So the synthetic probe overstated the with-grad cost
several-fold. This runs the ACTUAL production path (train_si_objects, real
SI via resolve_instruction, real per-arm batch size from the config) on a
tiny synthetic pool and reports peak allocated AND reserved — the number
that reconciles against the config's measured batch table.

Synthetic pool only (invented prompt/pos/neg text, never store data): the
memory shape is set by TOKEN COUNTS, not content, and the SI (the long,
memory-relevant side) is the real one. --prompt-tokens / --completion-tokens
size the student side conservatively toward the config's attack+completion
caps (constitution: up to 1024 attack + 512 completion).

Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in the environment
before running (the wave's sbatch bodies do) — it changes peak RESERVED on
long sequences materially.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling import


def _synth_pairs(tok, n: int, prompt_tokens: int, completion_tokens: int,
                 si_id: str):
    from sisteer.data.pairs import Pair

    # A filler string whose token count ≈ the target — attack prompts and
    # completions are what the student (with-grad) sequence length is made
    # of, so we size them toward the config's caps to be conservative.
    def filler(n_tok: int) -> str:
        words = ("the model should consider the request carefully and "
                 "respond in accordance with its guidelines ").split()
        out = []
        while len(tok(" ".join(out), add_special_tokens=False)["input_ids"]) < n_tok:
            out.extend(words)
        return " ".join(out)

    prompt = filler(prompt_tokens)
    pos = filler(completion_tokens)
    neg = filler(completion_tokens)
    return [Pair(prompt=prompt, pos=pos, neg=neg, instruction_id=si_id)
            for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/two_si.yaml")
    ap.add_argument("--si", default="si_constitution")
    ap.add_argument("--arms", default="psr_kl_all",
                    help="comma-separated; each measured in its own fresh "
                         "process-memory window")
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=800,
                    help="student attack-prompt length (constitution cap ~1024)")
    ap.add_argument("--completion-tokens", type=int, default=400,
                    help="completion length entering the loss (cap 512)")
    ap.add_argument("--sweep", action="store_true",
                    help="for each arm, walk batch size UP a ladder and "
                         "report the LARGEST that still clears the 0.85 "
                         "reserved guard (TrainTest 2026-08-21: the "
                         "wider-batch option, recorded not committed). "
                         "Uses exactly one full batch per candidate.")
    ap.add_argument("--sweep-max", type=int, default=64,
                    help="upper bound of the sweep ladder")
    ap.add_argument("--ceiling", action="store_true",
                    help="with --sweep: DOUBLE until the guard is exceeded, "
                         "then binary-search the gap, so the reported width "
                         "is the true ceiling rather than a ladder rung")
    ap.add_argument("--class-b", action="store_true",
                    help="measure CLASS B arms (caa_static_r, antidote, "
                         "alphasteer): SI-agnostic content fits driven "
                         "directly, not through train_si_objects. Implies "
                         "the ceiling search.")
    args = ap.parse_args()

    import train_two_si_objects as tt
    from sisteer.models import load_model_and_tokenizer

    cfg = yaml.safe_load(open(args.config))
    # Test the CONFIG's claim as measured pre-parallel: width 1 (whole batch
    # on one GPU), the arm's own configured batch size. Row-sharding only
    # ever REDUCES this (bs/width rows per rank).
    cfg["data_parallel"] = {"gpus": 1}
    cfg.setdefault("train_kwargs", {}).setdefault("default", {})
    cfg["train_kwargs"]["default"].update(
        {"epochs": 1, "holdout_frac": 0.0})
    cfg["wandb_fits"] = False

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    name = torch.cuda.get_device_name(0)
    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    from sisteer.eval.train_budget import resolve_batch_size, resolve_max_len

    max_len = resolve_max_len(cfg, args.si)
    guard = 0.85 * total

    def measure(arm: str, bs: int, n_pairs: int):
        """One (arm, batch size) measurement. Returns (alloc, reserved) or
        None on OOM. The batch size is injected through the config's own
        per-arm override, so resolve_batch_size's precedence — including
        any arm-intrinsic cap — still applies exactly as in production."""
        cfg.setdefault("batch_size_by_arm", {})[arm] = bs
        eff = resolve_batch_size(cfg, args.si, arm)
        pairs = _synth_pairs(tok, n_pairs, args.prompt_tokens,
                             args.completion_tokens, args.si)
        # The DPO arms take their OWN two-sided pool; without it the
        # driver refuses before training and the "measurement" is just a
        # model load (which is exactly what an earlier run of this probe
        # reported for bidpo at 15.1 GiB).
        pool_kw = {}
        if arm in tt.DPO_PREF_ARMS:
            h, b = _synth_bidpo_pools(max(n_pairs, bs))
            pool_kw = {"bidpo_harmful_pairs": h, "bidpo_benign_pairs": b}
        with tempfile.TemporaryDirectory() as td:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                outcome = tt.train_si_objects(
                    model, tok, args.si, pairs, [arm], cfg=cfg,
                    save_dir=Path(td), log_fn=None, overwrite=True,
                    **pool_kw)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return None, eff
            # The arm's GUARD outcome, surfaced: train_si_objects isolates a
            # failing arm (DeadGateError, floor-ratio) into the fit report
            # rather than raising, so a memory-only probe would silently
            # report a healthy-looking number for an arm that refused to
            # save. Print it next to the memory (psr_mse_all's guard
            # outcome in particular).
            res = outcome.arms.get(arm)
            if res is not None and res.status != "trained":
                print(f"MEMPROBE-GUARD si={args.si} arm={arm} bs={eff}: "
                      f"status={res.status} {res.error_type or ''} "
                      f"{(res.error or '')[:200]}")
            else:
                print(f"MEMPROBE-GUARD si={args.si} arm={arm} bs={eff}: "
                      "status=trained (all guards passed)")
            if res is not None and res.error_type == "OutOfMemoryError":
                # train_si_objects CATCHES a per-arm OOM (its isolation
                # contract), so it never reaches this frame's except clause.
                # An aborted run's memory stats are LOWER than a completed
                # one's and would score as a fit — the sweep once reported
                # diffmean's ceiling as 29 when every width above 16 was in
                # fact OOMing. Report it as the OOM it is.
                torch.cuda.empty_cache()
                return None, eff
            return (torch.cuda.max_memory_allocated() / 2**30,
                    torch.cuda.max_memory_reserved() / 2**30), eff

    def _synth_bidpo_pools(n: int):
        """Synthetic two-sided bidpo pool, correctly STAMPED so the driver's
        validators accept it (harmful rows requirement_a="strict"; benign
        rows the reversed-polarity BIDPO_BENIGN_REQ_A stamp), at the wave's
        worst-case lengths.

        Content is invented — for a MEMORY measurement only token counts and
        batch shape matter, and bidpo renders its prompts WITHOUT the SI, so
        its sequences are attack-prompt + completion regardless of which SI
        is nominally being fit. The real harmful side (1,360 accepted
        anti-safety compliances) is not yet emitted as a pairs file — it
        lands there only after manifest + emit-pairs, which wait on the
        benign side — so this measures the SHAPE the real pool will have."""
        import build_two_si_pool as bp
        from dataclasses import replace

        base = _synth_pairs(tok, n, args.prompt_tokens,
                            args.completion_tokens, args.si)
        harmful = [replace(p, requirement_a="strict", anti_neg=p.neg)
                   for p in base]
        benign = [replace(p, requirement_a=bp.BIDPO_BENIGN_REQ_A)
                  for p in base]
        return harmful, benign

    def measure_class_b(arm: str, bs: int):
        """One (Class B arm, batch size) measurement. Class B arms are
        SI-agnostic content fits driven by fit_class_b, not
        train_si_objects, so they need their own path: resolve the config's
        pinned prompt lists onto this host (the pod mounts its volume at
        /workspace, which the existing sibling-root remap already handles),
        take one batch's worth from each side — peak is per-batch, so one
        batch measures it — and call the arm's content trainer directly.

        DELIBERATELY NOT a real fit: the lists are truncated to bs prompts,
        so the resulting object is meaningless and is thrown away. Only the
        memory is being measured."""
        import json as _json

        from fit_class_b import resolve_cluster_path  # sibling (path at top)
        from sisteer.extraction.caa_static_r import (
            PAPER_LLAMA31_LAYERS, train_caa_static_r_content)

        cb = cfg["class_b"]
        benign = _json.loads(Path(resolve_cluster_path(
            cb["benign_prompts_path"])).read_text())
        harmful = _json.loads(Path(resolve_cluster_path(
            cb["harmful_prompts_path"])).read_text())
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            train_caa_static_r_content(
                model, tok, benign[:bs], harmful[:bs],
                layers=PAPER_LLAMA31_LAYERS, batch_size=bs, log_fn=None)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None, bs
        return (torch.cuda.max_memory_allocated() / 2**30,
                torch.cuda.max_memory_reserved() / 2**30), bs

    def ceiling_search(arm: str, configured: int, run):
        """The largest batch that clears the guard: DOUBLE until it fails,
        then binary-search the gap so the reported ceiling is the real one
        rather than the last rung of a coarse ladder."""
        lo_ok = None
        bs = max(1, configured)
        first_bad = None
        while bs <= args.sweep_max:
            got, eff = run(arm, bs)
            ok = got is not None and got[1] <= guard
            print(f"MEMPROBE-SWEEP gpu={name!r} si={args.si} arm={arm} "
                  f"bs={eff}: "
                  + ("OOM" if got is None else
                     f"alloc={got[0]:.1f}GiB reserved={got[1]:.1f}GiB "
                     f"{'FITS' if ok else 'OVER'}@guard={guard:.1f}GiB"))
            if not ok:
                first_bad = bs
                break
            lo_ok = (eff, got[0], got[1])
            bs *= 2
        if first_bad is not None and lo_ok is not None:
            lo, hi = lo_ok[0], first_bad
            while hi - lo > 1:
                mid = (lo + hi) // 2
                got, eff = run(arm, mid)
                ok = got is not None and got[1] <= guard
                print(f"MEMPROBE-SWEEP gpu={name!r} si={args.si} arm={arm} "
                      f"bs={eff} (refine): "
                      + ("OOM" if got is None else
                         f"reserved={got[1]:.1f}GiB "
                         f"{'FITS' if ok else 'OVER'}"))
                if ok:
                    lo, lo_ok = mid, (eff, got[0], got[1])
                else:
                    hi = mid
        if lo_ok:
            print(f"MEMPROBE-SWEEP-MAX si={args.si} arm={arm} "
                  f"configured={configured} largest_fitting={lo_ok[0]} "
                  f"(alloc={lo_ok[1]:.1f}GiB reserved={lo_ok[2]:.1f}GiB)")
        else:
            print(f"MEMPROBE-SWEEP-MAX si={args.si} arm={arm} "
                  f"configured={configured} largest_fitting=NONE "
                  "(does not fit even at the configured width)")

    if args.class_b:
        for arm in args.arms.split(","):
            ceiling_search(arm, int(cfg["class_b"].get("batch_size", 8)),
                           measure_class_b)
        sys.exit(0)

    failures = 0
    for arm in args.arms.split(","):
        configured = resolve_batch_size(cfg, args.si, arm)
        if args.sweep and args.ceiling:
            ceiling_search(arm, configured,
                           lambda a, b: measure(a, b, b))
            continue
        if not args.sweep:
            got, eff = measure(arm, configured, args.n_pairs)
            if got is None:
                failures += 1
                print(f"MEMPROBE-REAL gpu={name!r} total={total:.0f}GiB "
                      f"si={args.si} arm={arm} bs={eff}: OOM")
                continue
            peak, resv = got
            verdict = "FITS@0.85" if resv <= guard else "OVER-0.85-GUARD"
            print(f"MEMPROBE-REAL gpu={name!r} total={total:.0f}GiB "
                  f"si={args.si} arm={arm} bs={eff} max_len={max_len} "
                  f"student_prompt~{args.prompt_tokens} "
                  f"completion~{args.completion_tokens}: "
                  f"alloc={peak:.1f}GiB reserved={resv:.1f}GiB "
                  f"(0.85-guard={guard:.1f}GiB) {verdict}")
            continue
        # Sweep: walk UP from the configured width, one full batch per
        # candidate, and report the largest that clears the guard.
        ladder = [b for b in (configured, configured * 2, configured * 3,
                              configured * 4, configured * 6, configured * 8)
                  if b <= args.sweep_max]
        best = None
        for bs in ladder:
            got, eff = measure(arm, bs, bs)  # exactly one full batch
            if got is None:
                print(f"MEMPROBE-SWEEP si={args.si} arm={arm} bs={eff}: OOM "
                      "— ladder stops here")
                break
            peak, resv = got
            ok = resv <= guard
            print(f"MEMPROBE-SWEEP gpu={name!r} si={args.si} arm={arm} "
                  f"bs={eff}: alloc={peak:.1f}GiB reserved={resv:.1f}GiB "
                  f"{'FITS' if ok else 'OVER'}@0.85-guard={guard:.1f}GiB")
            if not ok:
                break
            best = (eff, peak, resv)
        if best:
            print(f"MEMPROBE-SWEEP-MAX si={args.si} arm={arm} "
                  f"configured={configured} largest_fitting={best[0]} "
                  f"(alloc={best[1]:.1f}GiB reserved={best[2]:.1f}GiB)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
