#!/usr/bin/env python
"""Hyperparameter search + divergence-monitored fits for lora_r1_all.

The all-layer rank-1 LoRA arm (MultiLayerLoRAObject: 32 jointly-trained
rank-1 o_proj edits) has never produced a usable fit: stage 2 trained it at
the single-edit defaults (fixed lr 5e-3, no warmup, no clipping) and it was
the capability-destroying worst arm of the matrix; the audit found NO
implementation bug and named optimisation difficulty at matched budget
as the leading candidate. The sibling all-layer arm psr_kl_all diverged the
same way and was fixed by lr 5e-4 + warmup_cosine + clip 1.0 — with the
BASE RATE the load-bearing change (configs/two_si.yaml train_kwargs notes).

This script runs the tuning-playbook protocol to locate lora_r1_all's own
stable recipe on safety_refusal, without touching any file the two-SI
pipeline owns: every trial trains through train_lockstep with a SINGLETON
group — the exact production path (probes, early stopping, best-epoch
restore, the mask->loss wiring) — with the trial's (lr, warmup_frac,
max_grad_norm) carried by LockstepArmParams, the same per-arm
optimiser-schedule seam the production config uses for psr_kl_all. A winning
recipe therefore translates verbatim into a train_kwargs.lora_r1_all block.

Protocol (the tuning-playbook house heuristics):
  - base-2 Sobol grid, 2^m trials, jointly over log(lr) x log(warmup_frac)
    x log(max_grad_norm); schedule fixed at warmup_cosine (linear ramp,
    cosine decay to 0 over the full step budget);
  - after a round, fit ONE joint quadratic in the log-hyperparameters over
    the trial losses and take the fitted minimum (see fit_quadratic below;
    re-center and re-run if it lands on the box edge);
  - per-step monitoring: step loss, LR, pre-clip AND post-clip global grad
    norm, per-layer |A_l| / |B_l| (the object-norm analogue of the
    "monitor steering-vector norms" heuristic), throughput;
  - divergence monitoring on a fixed held-out probe: BOTH KL ratios —
      residual      KL(prompted || steered) / KL(prompted || unsteered)
                    (the lane's floor_ratio: 0 = reproduces prompting,
                    1 = no better than doing nothing)
      displacement  KL(steered || unsteered) / KL(prompted || unsteered)
                    (how far the object has moved the model, in units of
                    how far prompting moves it — a dead arm and a
                    wrong-direction arm separate only when both ratios
                    are read together)
    measured densely early (the playbook: early instability hides from
    sparse evals), then every --probe-every steps;
  - a NaN step loss aborts the trial and records it as diverged.

Subcommands:
  emit-trials  write the deterministic Sobol trial list for a round
  probe-mem    peak-memory probe at a given batch size (playbook: batch
               size is a throughput knob — largest that fits)
  trial        run one trial end to end (one GPU)

Data: --pairs must point at the blessed pool copy WITH its .tokcache.npz
sidecar next to it (the sidecar convention); the run refuses to fall back
to a full tokenize unless --allow-tokenize is given.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Sobol (base-2, dims 1-3, Joe-Kuo direction numbers) — no scipy dependency.
# --------------------------------------------------------------------------

# (s, a, m_init) per dimension >= 2; dim 1 is the Van der Corput sequence.
_SOBOL_POLY = {
    2: (1, 0, (1,)),
    3: (2, 1, (1, 3)),
}
_SOBOL_BITS = 30


def _direction_ints(dim: int) -> list[int]:
    if dim == 1:
        m = [1] * _SOBOL_BITS
    else:
        s, a, m_init = _SOBOL_POLY[dim]
        m = list(m_init)
        while len(m) < _SOBOL_BITS:
            k = len(m)  # 0-based index of the value being computed
            new = m[k - s] ^ (m[k - s] << s)
            for j in range(1, s):
                if (a >> (s - 1 - j)) & 1:
                    new ^= m[k - j] << j
            m.append(new)
    return [m[k] << (_SOBOL_BITS - 1 - k) for k in range(_SOBOL_BITS)]


def sobol_points(m_log2: int, d: int = 3) -> list[tuple[float, ...]]:
    """First 2^m_log2 points of the d-dimensional Sobol sequence (gray-code
    order, point 0 at the origin). Coverage guarantees hold exactly at 2^m
    points, which is why the trial count is a power of two."""
    if d > 3:
        raise ValueError("direction numbers included for d <= 3 only")
    n = 1 << m_log2
    vs = [_direction_ints(dim + 1) for dim in range(d)]
    x = [0] * d
    out = [tuple(0.0 for _ in range(d))]
    for i in range(1, n):
        k = i - 1
        c = ((k + 1) & ~k).bit_length() - 1  # lowest zero bit of k
        for dim in range(d):
            x[dim] ^= vs[dim][c]
        out.append(tuple(x[dim] / (1 << _SOBOL_BITS) for dim in range(d)))
    return out


def map_log(u: float, lo: float, hi: float) -> float:
    return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def emit_trials(args) -> None:
    box = {
        "lr": (args.lr_min, args.lr_max),
        "warmup_frac": (args.warm_min, args.warm_max),
        "max_grad_norm": (args.clip_min, args.clip_max),
    }
    m_log2 = int(round(math.log2(args.n)))
    if (1 << m_log2) != args.n:
        raise SystemExit(f"--n must be a power of 2, got {args.n}")
    pts = sobol_points(m_log2, d=3)
    trials = []
    for i, (u_lr, u_w, u_c) in enumerate(pts):
        trials.append({
            "index": i,
            "lr": map_log(u_lr, *box["lr"]),
            "warmup_frac": map_log(u_w, *box["warmup_frac"]),
            "max_grad_norm": map_log(u_c, *box["max_grad_norm"]),
        })
    payload = {"round": args.round, "box": {k: list(v) for k, v in box.items()},
               "sobol_m": m_log2, "trials": trials}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[emit-trials] wrote {len(trials)} trials -> {args.out}")
    for t in trials:
        print(f"  t{t['index']:02d}: lr={t['lr']:.3e} "
              f"warmup_frac={t['warmup_frac']:.3f} "
              f"clip={t['max_grad_norm']:.3f}")


# --------------------------------------------------------------------------
# Joint quadratic fit over the log-hyperparameters (house heuristic 3).
# --------------------------------------------------------------------------

def fit_quadratic(rows: list[dict], keys: tuple[str, ...],
                  y_key: str = "best_heldout_loss") -> dict:
    """Least-squares full quadratic (cross terms included) of log(y) over
    the log-hyperparameters; returns the stationary point, whether the
    Hessian is positive definite, and the fitted minimum. Pure torch — no
    scipy. Diverged/NaN trials must be filtered (or given a penalty y) by
    the caller before this."""
    X = torch.tensor([[math.log(r[k]) for k in keys] for r in rows],
                     dtype=torch.float64)
    y = torch.tensor([math.log(max(r[y_key], 1e-12)) for r in rows],
                     dtype=torch.float64)
    n, d = X.shape
    cols = [torch.ones(n, dtype=torch.float64)]
    names = ["1"]
    for i in range(d):
        cols.append(X[:, i]); names.append(keys[i])
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j]); names.append(f"{keys[i]}*{keys[j]}")
    A = torch.stack(cols, dim=1)
    coef = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
    # Assemble gradient system: grad = b + H z = 0.
    b = coef[1:1 + d].clone()
    H = torch.zeros(d, d, dtype=torch.float64)
    idx = 1 + d
    for i in range(d):
        for j in range(i, d):
            c = coef[idx]; idx += 1
            if i == j:
                H[i, i] = 2 * c
            else:
                H[i, j] = c; H[j, i] = c
    eigs = torch.linalg.eigvalsh(H)
    pd = bool((eigs > 0).all())
    z_star = torch.linalg.solve(H, -b) if pd else None
    out = {"coef": coef.tolist(), "names": names, "hessian_eigs": eigs.tolist(),
           "positive_definite": pd}
    if z_star is not None:
        out["argmin"] = {k: math.exp(float(z_star[i]))
                         for i, k in enumerate(keys)}
        pred = float(coef[0] + (b * z_star).sum()
                     + 0.5 * (z_star @ H @ z_star))
        out["fitted_min_log_y"] = pred
    return out


# --------------------------------------------------------------------------
# Shared setup for probe-mem / trial.
# --------------------------------------------------------------------------

def _load_driver():
    """Import scripts/train_two_si_objects.py as a module so the holdout
    split, pool loader and sidecar conventions are the driver's OWN code,
    never a reimplementation."""
    import importlib.util
    import sys

    p = Path(__file__).resolve().parent / "train_two_si_objects.py"
    spec = importlib.util.spec_from_file_location("_two_si_driver", p)
    mod = importlib.util.module_from_spec(spec)
    # Must be registered BEFORE exec: the driver defines @dataclass classes,
    # and dataclasses resolves cls.__module__ through sys.modules.
    sys.modules["_two_si_driver"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _warm_tokcache(cfg: dict, pairs_path: str, tok, allow_tokenize: bool):
    from sisteer.extraction import tokcache

    sidecar = Path(str(pairs_path) + ".tokcache.npz")
    if sidecar.exists():
        cache = tokcache.warm_from_sidecar(sidecar, tok,
                                           expect_model_id=cfg["model"])
        tokcache.set_active(cache)
        print(f"[tokcache] warmed {len(cache)} entries from {sidecar}")
    elif allow_tokenize:
        tokcache.set_active(tokcache.TokCache())
        print(f"[tokcache] NO sidecar at {sidecar}; tokenizing in-process")
    else:
        raise SystemExit(
            f"[tune-lora-all] no token sidecar at {sidecar} — the sweep is "
            "meant to run pre-tokenized (point --pairs at the pool copy "
            "whose .tokcache.npz sits next to it, or pass --allow-tokenize)")


def _setup(args, need_model: bool = True):
    """cfg, driver, tok/model, fit+heldout DistillExamples for the SI."""
    from sisteer.data.pairs import pairs_to_distill_examples
    from sisteer.eval.jailbreak import resolve_instruction
    from sisteer.eval.train_budget import resolve_max_len
    from sisteer.models import (configure_fast_sdpa, load_model_and_tokenizer,
                                load_tokenizer)

    configure_fast_sdpa()  # cuDNN-SDPA footgun — the one shared enforcement

    cfg = _load_cfg(args.config)
    driver = _load_driver()
    instr = None
    if need_model:
        model, tok = load_model_and_tokenizer(
            cfg["model"], dtype=cfg.get("dtype", "bfloat16"),
            revision=cfg.get("model_revision"),
            attn_implementation=cfg.get("attn_implementation", "sdpa"))
    else:
        model, tok = None, load_tokenizer(cfg["model"],
                                          revision=cfg.get("model_revision"))
    _warm_tokcache(cfg, args.pairs, tok, args.allow_tokenize)
    pairs = driver._load_pairs(args.pairs)
    tkd = driver._train_kwargs_default(cfg)
    fit_pairs, held_pairs = driver.split_pairs(
        pairs, float(tkd.get("holdout_frac", 0.1)),
        str(tkd.get("holdout_salt", "two_si_holdout")))
    instr = resolve_instruction(args.si)
    fit_ex = pairs_to_distill_examples(tok, fit_pairs, instr)
    held_ex = pairs_to_distill_examples(tok, held_pairs, instr)
    max_len = resolve_max_len(cfg, args.si)
    print(f"[setup] si={args.si} pairs={len(pairs)} fit={len(fit_ex)} "
          f"heldout={len(held_ex)} max_len={max_len}")
    return cfg, driver, model, tok, instr, fit_ex, held_ex, max_len


# --------------------------------------------------------------------------
# The two KL ratios, on a fixed held-out probe.
# --------------------------------------------------------------------------

class KLRatioProbe:
    """Fixed sha-rank-selected held-out probe measuring, per call:

      residual_kl      KL(prompted || steered)        (the floor numerator)
      displacement_kl  KL(steered || unsteered)       (the displacement one)

    against a once-computed denominator KL(prompted || unsteered) — the
    no-op floor on THESE examples. All three are per-example completion-
    token means averaged over examples (the chunking-invariant estimator
    the lane's probes use), teacher rendered WITH the SI, student without,
    completion positions index-aligned by _encode_aligned's contract."""

    def __init__(self, tok, examples, *, max_len: int, batch_size: int,
                 n: int = 64):
        from sisteer.extraction.distill import _encode_aligned
        from sisteer.extraction.subsample import sha_rank_indices
        from sisteer.extraction.traindata import collate

        idx = sha_rank_indices([ex.prompt_without for ex in examples], n)
        sel = [examples[i] for i in idx]
        enc = [_encode_aligned(tok, ex, max_len) for ex in sel]
        self.batches = []
        for i in range(0, len(enc), batch_size):
            chunk = enc[i:i + batch_size]
            self.batches.append(
                (collate([t for t, _ in chunk], tok.pad_token_id),
                 collate([s for _, s in chunk], tok.pad_token_id)))
        self.n = len(sel)
        self.denom: float | None = None

    @staticmethod
    def _counts(batch) -> list[int]:
        return [int(c) for c in (batch.labels[:, 1:] != -100).sum(-1)]

    @torch.no_grad()
    def measure(self, model, obj=None) -> dict:
        from sisteer.extraction.distill import (_gather_completion_logits,
                                                per_example_means,
                                                soft_divergence_per_token)

        device = next(model.parameters()).device
        res_vals, disp_vals, floor_vals = [], [], []
        for teacher_b, student_b in self.batches:
            teacher_b = teacher_b.to(device)
            student_b = student_b.to(device)
            counts = self._counts(student_b)
            t_logits = _gather_completion_logits(
                model(input_ids=teacher_b.input_ids,
                      attention_mask=teacher_b.attention_mask).logits,
                teacher_b)
            u_logits = _gather_completion_logits(
                model(input_ids=student_b.input_ids,
                      attention_mask=student_b.attention_mask).logits,
                student_b)
            t_logp = torch.log_softmax(t_logits.float(), -1)
            u_logp = torch.log_softmax(u_logits.float(), -1)
            if self.denom is None:
                floor_vals.extend(per_example_means(
                    soft_divergence_per_token("soft", t_logp, u_logp),
                    counts))
            if obj is not None:
                with obj.apply(model, scale=1.0) as handle:
                    if (handle is not None
                            and getattr(handle, "positions", "all")
                            in ("response", "prompt")):
                        # Masked-fit objects need per-row boundaries before
                        # the forward (the _batch0_loss convention). The
                        # probe KLs still cover ALL completion positions:
                        # masked positions are bit-identical to unsteered,
                        # so they contribute 0 displacement by construction.
                        from sisteer.extraction.distill import (
                            steer_prompt_boundaries)
                        handle.set_prompt_lens(steer_prompt_boundaries(
                            student_b, handle.response_header_len))
                    s_logits = _gather_completion_logits(
                        model(input_ids=student_b.input_ids,
                              attention_mask=student_b.attention_mask).logits,
                        student_b)
                s_logp = torch.log_softmax(s_logits.float(), -1)
                # KL(prompted || steered): forward KL, target = teacher.
                res_vals.extend(per_example_means(
                    soft_divergence_per_token("soft", t_logp, s_logp),
                    counts))
                # KL(steered || unsteered): forward KL, target = steered.
                disp_vals.extend(per_example_means(
                    soft_divergence_per_token("soft", s_logp, u_logp),
                    counts))
        out: dict = {}
        if floor_vals:
            self.denom = sum(floor_vals) / len(floor_vals)
        if self.denom is not None:
            out["probe/noop_floor"] = self.denom
        if res_vals:
            res = sum(res_vals) / len(res_vals)
            disp = sum(disp_vals) / len(disp_vals)
            out.update({
                "probe/residual_kl": res,
                "probe/displacement_kl": disp,
                "probe/residual_ratio": res / max(self.denom, 1e-6),
                "probe/displacement_ratio": disp / max(self.denom, 1e-6),
            })
        return out


def length_sorted(tok, examples):
    """Length-bucket batching, without touching the engine: train_lockstep
    chunks examples IN THE ORDER GIVEN (and shuffles only batch ORDER per
    epoch), so sorting by rendered token length makes every batch pad to a
    near-uniform width. Two wins, both measured on the r5 confirm wave:
    ragged batches wasted ~30-40% of token compute on padding, and every
    distinct padded shape made cuDNN SDPA re-plan on CPU (the 33%-GPU-duty
    footgun; also fixed by disabling that backend). Teacher order follows
    student order by construction (one example = one aligned pair).
    NOTE: batch COMPOSITION changes vs the unsorted production driver —
    same objective, different gradient grouping; recorded in the summary."""
    from sisteer.extraction.tokcache import cached_ids

    def keylen(ex):
        return (len(cached_ids(tok, ex.prompt_without, True))
                + len(cached_ids(tok, ex.completion, False)))

    return sorted(examples, key=keylen)


class GradNormTap:
    """Pre-clip global gradient norm via per-parameter backward hooks —
    clipping mutates .grad in place before opt.step(), so the post-step
    .grad tensors are the CLIPPED gradients; the hooks see the raw ones."""

    def __init__(self, params):
        self._sq = None
        self._handles = [p.register_hook(self._add) for p in params]

    def _add(self, g):
        v = g.detach().float().pow(2).sum()
        self._sq = v if self._sq is None else self._sq + v

    def flush(self) -> float:
        if self._sq is None:
            return float("nan")
        v = float(self._sq.sqrt())
        self._sq = None
        return v

    def remove(self):
        for h in self._handles:
            h.remove()


class TrialDiverged(RuntimeError):
    pass


class Heartbeat:
    """Continuous liveness logging — training logs continuously so nobody
    has to infer a trial's phase from nvidia-smi. A daemon thread prints one
    line every ``interval`` seconds: the named pre-training phase before
    step 0, then step/total, loss, LR, s/step EMA and ETA — with an explicit
    note when the trial is inside a long no-step pass (dense KL probe,
    per-epoch held-out eval), so a quiet GPU never reads as a stall. log_fn
    feeds it; phase() marks stages."""

    def __init__(self, total_steps: int, interval: float = 30.0):
        import threading

        self.t0 = time.time()
        self.total = total_steps
        self.state = {"phase": "init", "phase_t": self.t0, "step": None,
                      "loss": float("nan"), "lr": None, "spe": None,
                      "last_step_t": self.t0}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def phase(self, msg: str) -> None:
        self.state.update(phase=msg, phase_t=time.time())
        print(f"[{time.strftime('%H:%M:%S')}] [phase +{time.time() - self.t0:.0f}s] {msg}",
              flush=True)

    def step(self, step: int, loss: float, lr) -> None:
        now = time.time()
        dt = now - self.state["last_step_t"]
        spe = self.state["spe"]
        self.state.update(step=step, loss=loss, lr=lr, last_step_t=now,
                          spe=dt if spe is None else 0.9 * spe + 0.1 * dt)
        if step % 50 == 0:
            self._line()

    def _line(self) -> None:
        s = self.state
        if s["step"] is None:
            print(f"[{time.strftime('%H:%M:%S')}] [hb] pre-training: "
                  f"{s['phase']} ({time.time() - s['phase_t']:.0f}s in phase, "
                  f"{time.time() - self.t0:.0f}s total)", flush=True)
            return
        gap = time.time() - s["last_step_t"]
        spe = s["spe"] or 0.0
        eta = (self.total - s["step"]) * spe / 60 if spe else float("nan")
        note = (f" | {gap:.0f}s since last step (probe/held-out pass)"
                if gap > max(15.0, 5 * spe) else "")
        lr = f" lr={s['lr']:.2e}" if s["lr"] is not None else ""
        print(f"[{time.strftime('%H:%M:%S')}] [hb] step {s['step']}/{self.total} "
              f"loss={s['loss']:.4f}{lr} {spe:.2f}s/step eta={eta:.0f}min{note}",
              flush=True)

    def _run(self) -> None:
        while not self._stop.wait(30.0):
            self._line()

    def close(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------
# probe-mem
# --------------------------------------------------------------------------

def probe_mem(args) -> None:
    from sisteer.extraction.lockstep import (LockstepArm, LockstepArmParams,
                                             train_lockstep)
    from sisteer.extraction.multilayer import MultiLayerLoRAObject
    from sisteer.extraction.subsample import sha_rank_indices

    cfg, _driver, model, tok, _instr, fit_ex, _held, max_len = _setup(args)
    n = max(args.batch_size * args.steps, args.batch_size)
    if args.worst_case:
        # The LONGEST n examples: with length-bucketed batches the densest
        # batch is all-long rows, which is the true peak-memory shape.
        sub = length_sorted(tok, fit_ex)[-n:]
    else:
        idx = sha_rank_indices([ex.prompt_without for ex in fit_ex], n)
        sub = [fit_ex[i] for i in idx]
    obj = MultiLayerLoRAObject(model)
    arm = LockstepArm("lora_r1_all", obj,
                      LockstepArmParams(lr=1e-5,
                                        lr_schedule="warmup_cosine",
                                        warmup_steps=10, max_grad_norm=1.0))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_lockstep(model, tok, [arm], sub, epochs=1,
                   batch_size=args.batch_size, max_len=max_len, seed=0,
                   n_probe=args.batch_size)
    dt = time.time() - t0
    alloc = torch.cuda.max_memory_allocated() / 2**30
    reserv = torch.cuda.max_memory_reserved() / 2**30
    steps = math.ceil(len(sub) / args.batch_size)
    print(f"[probe-mem] bs={args.batch_size} max_len={max_len} "
          f"steps={steps} peak_alloc={alloc:.1f}GiB "
          f"peak_reserved={reserv:.1f}GiB wall={dt:.0f}s "
          f"({dt / steps:.2f}s/step)")


# --------------------------------------------------------------------------
# trial
# --------------------------------------------------------------------------

def run_trial(args) -> None:
    from sisteer.extraction.distill import EarlyStopSpec, floor_ratio_check
    from sisteer.extraction.distill import FloorRatioError
    from sisteer.extraction.lockstep import (LockstepArm, LockstepArmParams,
                                             train_lockstep)

    spec = json.loads(Path(args.trials).read_text())
    trial = spec["trials"][args.index]
    out = Path(args.out) / f"r{spec['round']}_t{trial['index']:02d}"
    out.mkdir(parents=True, exist_ok=True)

    hb = Heartbeat(total_steps=0)
    hb.phase("loading model+tokenizer, warming token cache, loading+"
             "splitting pool")
    cfg, _driver, model, tok, instr, fit_ex, held_ex, max_len = _setup(args)

    # The fit's position leg (Class A "all" vs Class C decode-only
    # "response"), mirroring the driver: H derived from the tokenizer when
    # not pinned, refusing a collapsed header (a zero H silently turns
    # header-steered into plain decode-only).
    response_header_len = None
    if args.positions != "all":
        rhl_cfg = _driver._train_kwargs_default(cfg).get(
            "response_header_len")
        if rhl_cfg is not None:
            response_header_len = int(rhl_cfg)
        else:
            from sisteer.hooks import response_header_len as _derive_h
            response_header_len = _derive_h(tok)
            if response_header_len == 0:
                raise SystemExit(
                    f"positions={args.positions!r} but H derived as 0 from "
                    "this tokenizer — refusing (the driver's convention; "
                    "pin train_kwargs.default.response_header_len to "
                    "override)")

    if args.fit_subsample and args.fit_subsample < len(fit_ex):
        # Exploration rounds train on a deterministic sha-rank subpool (the
        # SAME subpool for every trial — ranking configs needs a shared
        # horizon, not full-length runs; measured 3.0 s/step at bs 8 makes
        # a full-pool trial ~6.5 h). The held-out slice stays FULL, so
        # heldout/probe numbers remain comparable to a full-pool fit's.
        # The winning config's confirm run passes 0 (= full pool).
        from sisteer.extraction.subsample import sha_rank_indices
        idx = sha_rank_indices([ex.prompt_without for ex in fit_ex],
                               args.fit_subsample)
        fit_ex = [fit_ex[i] for i in idx]
        print(f"[trial] fit subsampled to {len(fit_ex)} examples (sha-rank)")

    if not args.no_length_sort:
        fit_ex = length_sorted(tok, fit_ex)
        held_ex = length_sorted(tok, held_ex)  # eval-only: speed, not semantics
        print("[trial] length-bucketed batches (sorted by rendered length)")

    steps_per_epoch = math.ceil(len(fit_ex) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    hb.total = total_steps
    warmup_steps = max(1, int(round(trial["warmup_frac"] * total_steps)))
    params = LockstepArmParams(
        lr=float(trial["lr"]), lr_schedule="warmup_cosine",
        warmup_steps=warmup_steps,
        max_grad_norm=float(trial["max_grad_norm"]))
    from sisteer.extraction.lockstep import build_lockstep_object
    obj = build_lockstep_object(
        "lora_r1_all", model, tok, instr, [],
        layer=int(cfg.get("layer", 15)), positions=args.positions,
        response_header_len=response_header_len)
    arm = LockstepArm("lora_r1_all", obj, params)

    # Supervised-token throughput denominator (per-step average).
    from sisteer.extraction.tokcache import cached_ids
    sup_tokens = sum(len(cached_ids(tok, ex.completion, False))
                     for ex in fit_ex)
    sup_per_step = sup_tokens / max(steps_per_epoch, 1)

    hb.phase(f"encoding KL probe (n={args.probe_n}) + measuring the "
             "KL(prompted||unsteered) denominator")
    probe = KLRatioProbe(tok, held_ex, max_len=max_len,
                         batch_size=args.batch_size, n=args.probe_n)
    base = probe.measure(model)  # denominator only (object not yet applied)
    print(f"[trial t{trial['index']:02d}] lr={params.lr:.3e} "
          f"warmup={warmup_steps}/{total_steps} steps "
          f"clip={params.max_grad_norm:.3f} bs={args.batch_size} "
          f"noop_floor={base.get('probe/noop_floor', float('nan')):.5f}")

    run = None
    if not args.no_wandb:
        try:
            # the config's wandb_project (the configs write $WANDB_PROJECT), else $WANDB_PROJECT itself; one with no
            # value on this host disables W&B for the trial below, the message naming the variable to set
            from sisteer import site as _site
            project = _site.expand(cfg.get("wandb_project") or "$WANDB_PROJECT", key="wandb_project")
            import wandb
            run = wandb.init(
                project=str(project),
                entity=cfg.get("wandb_entity"),
                name=(f"hp-lora1all-{args.si}"
                      + ("-resp" if args.positions == "response" else "")
                      + f"-r{spec['round']}-t{trial['index']:02d}"),
                group=f"hp-lora1all-{args.si}-r{spec['round']}",
                config={**trial, "warmup_steps": warmup_steps,
                        "total_steps": total_steps, "epochs": args.epochs,
                        "batch_size": args.batch_size, "si": args.si,
                        "max_len": max_len, "arm": "lora_r1_all",
                        "positions": args.positions,
                        "response_header_len": response_header_len,
                        "teacher_prefix_cache": args.teacher_prefix_cache,
                        "round": spec["round"]},
                settings=wandb.Settings(console="off", init_timeout=60))
        except Exception as e:  # W&B must never kill a trial
            print(f"[wandb] disabled: {e}")
            run = None

    jsonl = (out / "metrics.jsonl").open("a")
    tap = GradNormTap(obj.trainable)
    layer_ids = list(obj.layers)
    loss_key = "lockstep/lora_r1_all/soft_loss"
    state = {"step": 0, "t": time.time(), "disp_curve": []}

    prof = None
    if args.profile:
        tb_dir = out / "tb"
        tb_dir.mkdir(exist_ok=True)
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=20, warmup=3, active=12,
                                             repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(tb_dir)))
        prof.__enter__()

    def _emit(payload: dict, step: int):
        rec = {k: v for k, v in payload.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
        rec["step"] = step
        jsonl.write(json.dumps(rec) + "\n")
        jsonl.flush()
        if run is not None:
            run.log(rec)

    def log_fn(payload, step=None):
        step = state["step"] if step is None else step
        if loss_key not in payload:
            if any("heldout" in k for k in payload):
                hb.phase(f"held-out eval done (epoch boundary, step {step}) "
                         "— training resumes")
            _emit(payload, step)
            return
        state["step"] = step
        loss_val = payload[loss_key]
        hb.step(step, loss_val, payload.get(f"lockstep/lora_r1_all/lr"))
        now = time.time()
        dt = now - state["t"]
        state["t"] = now
        extra = {
            "train/grad_norm_preclip": tap.flush(),
            "train/sec_per_step": dt,
            "train/supervised_tokens_per_s": sup_per_step / max(dt, 1e-9),
        }
        pc = [p.grad for p in obj.trainable if p.grad is not None]
        if pc:
            extra["train/grad_norm_postclip"] = float(torch.norm(
                torch.stack([g.detach().float().norm() for g in pc])))
        if step % args.norms_every == 0 or step <= 3:
            b_norms = {l: float(obj.B[l].detach().norm()) for l in layer_ids}
            a_norms = {l: float(obj.A[l].detach().norm()) for l in layer_ids}
            extra["norms/B_total"] = math.fsum(v * v for v in
                                               b_norms.values()) ** 0.5
            extra["norms/B_max"] = max(b_norms.values())
            extra["norms/A_total"] = math.fsum(v * v for v in
                                               a_norms.values()) ** 0.5
            for l in layer_ids:
                extra[f"norms/B_{l:02d}"] = b_norms[l]
        dense = step <= args.dense_until and step % args.dense_every == 0
        if dense or step % args.probe_every == 0:
            kl = probe.measure(model, obj)
            extra.update(kl)
            state["disp_curve"].append(
                (step, kl.get("probe/displacement_ratio"),
                 kl.get("probe/residual_ratio")))
            dr = kl.get("probe/displacement_ratio")
            rr = kl.get("probe/residual_ratio")
            print(f"  step {step}: loss={loss_val:.4f} "
                  f"disp_ratio={dr:.3f} resid_ratio={rr:.3f}")
        _emit({**payload, **extra}, step)
        if prof is not None and step <= 40:
            prof.step()
        if math.isnan(loss_val) or math.isinf(loss_val):
            raise TrialDiverged(f"step {step}: loss={loss_val}")

    diag: dict = {}
    status, err = "ok", ""
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    hb.phase(f"train_lockstep: encode {len(fit_ex)} fit + {len(held_ex)} "
             f"held-out examples (narrated below), then startup probe "
             f"(n=256) + held-out baseline BEFORE step 0 — first step "
             f"typically lands a few minutes in; {steps_per_epoch} "
             f"steps/epoch x {args.epochs} epochs planned")
    try:
        train_lockstep(
            model, tok, [arm], fit_ex,
            epochs=args.epochs, batch_size=args.batch_size, max_len=max_len,
            seed=0, log_fn=log_fn, heldout=held_ex,
            early_stop=EarlyStopSpec(patience=2, min_epochs=1,
                                     min_delta=0.0),
            diagnostics=diag, n_probe=256,
            teacher_prefix_cache=args.teacher_prefix_cache)
    except TrialDiverged as e:
        status, err = "nan_diverged", str(e)
    except FloorRatioError as e:  # not raised by train_lockstep, but safe
        status, err = "floor_failed", str(e)
    finally:
        tap.remove()
        if prof is not None:
            prof.__exit__(None, None, None)
    wall = time.time() - t0

    hb.phase(f"training done ({status}); final KL probe + saving artifacts")
    final_kl = probe.measure(model, obj) if status == "ok" else {}
    d = diag.get("lora_r1_all", {})
    floor_ok = None
    if status == "ok" and d.get("final_probe_loss") is not None:
        try:
            floor_ratio_check(d["final_probe_loss"], d["noop_floor"],
                              ratio_max=1.0, tag="lora_r1_all")
            floor_ok = True
        except FloorRatioError:
            floor_ok = False

    summary = {
        "trial": trial, "round": spec["round"], "status": status,
        "error": err, "si": args.si, "epochs": args.epochs,
        "fit_subsample": args.fit_subsample, "n_fit": len(fit_ex),
        "length_sorted": not args.no_length_sort,
        "positions": args.positions,
        "response_header_len": response_header_len,
        "teacher_prefix_cache": args.teacher_prefix_cache,
        "batch_size": args.batch_size, "max_len": max_len,
        "warmup_steps": warmup_steps, "total_steps": total_steps,
        "steps_run": state["step"], "wall_s": wall,
        "peak_alloc_gib": (torch.cuda.max_memory_allocated() / 2**30
                           if cuda else None),
        "best_heldout_loss": d.get("best_heldout_loss"),
        "best_epoch": d.get("best_epoch"),
        "heldout_curve": d.get("heldout_curve"),
        "stopped_early": d.get("stopped_early"),
        "noop_floor": d.get("noop_floor"),
        "final_probe_loss": d.get("final_probe_loss"),
        "floor_ratio": d.get("floor_ratio"),
        "floor_ratio_sem": d.get("floor_ratio_sem"),
        "floor_ok_at_1.0": floor_ok,
        "displacement_curve": state["disp_curve"],
        **{k.replace("probe/", "final_"): v for k, v in final_kl.items()},
        "wandb_run": getattr(run, "id", None),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if status == "ok":
        obj.save(out / "object.pt")
        # The lane's per-object provenance sidecar (train_budget_meta.json)
        # — measure_cell_ratios.py refuses an artifact whose fit leg is
        # unrecorded and cross-checks positions_at_fit vs the hydrated
        # object; minimal-but-faithful field set.
        (out / "train_budget_meta.json").write_text(json.dumps({
            "arm": "lora_r1_all", "si_id": args.si,
            "positions_at_fit": args.positions,
            "response_header_len": response_header_len,
            "batch_size": args.batch_size, "max_len": max_len,
            "epochs": args.epochs, "seed": 0,
            "lr": params.lr, "lr_schedule": params.lr_schedule,
            "warmup_steps": warmup_steps,
            "max_grad_norm": params.max_grad_norm,
            "length_sorted": not args.no_length_sort,
            "teacher_prefix_cache": args.teacher_prefix_cache,
            "trainer": "tune_lora_all.py trial (train_lockstep singleton)",
        }, indent=2) + "\n")
        _registry_object(out, args.si)
    if run is not None:
        run.summary.update({k: v for k, v in summary.items()
                            if isinstance(v, (int, float, bool, str))
                            and v is not None})
        run.finish()
    hb.close()
    print(f"[trial t{trial['index']:02d}] {status} "
          f"best_heldout={summary['best_heldout_loss']} "
          f"floor_ratio={summary['floor_ratio']} "
          f"disp_ratio={summary.get('final_displacement_ratio')} "
          f"wall={wall:.0f}s -> {out}", flush=True)


def _registry_object(out: Path, si_id: str) -> None:
    """Results-registry native emission (additive; a no-op when emission is off): the
    trial's ``object`` + ``artifact`` records from the sidecar just written. The writer
    states what it fitted (a LoRA distilled from the SI teacher by KL); an SI whose role is
    not declared in the registry's table is not recorded, never guessed."""
    try:
        from sisteer.registry import hooks as _rh
        role = _rh.object_role_for_si(si_id)
        if role is None:
            _rh.notice(f"tune_lora_all: si_id {si_id!r} has no declared object role; trial object not recorded")
            return
        _rh.emit_object_from_sidecar(
            str(out), campaign=os.environ.get("REGISTRY_CAMPAIGN", "lora_hpo"), object_kind="lora",
            paper_label="other", object_role=role, fit_target_kind="system_instruction_distillation",
            fit_objective="kl", fit_thinking_render="off", status="reference", fit_teacher_si_id=si_id,
            script="scripts/tune_lora_all.py trial")
    except Exception as e:  # noqa: BLE001 — a lost record never costs a trial
        print(f"[registry] tune_lora_all: object not recorded ({type(e).__name__}: {e})", flush=True)


# --------------------------------------------------------------------------
# analyze: joint quadratic over a finished round's summaries
# --------------------------------------------------------------------------

def analyze(args) -> None:
    keys = tuple(k.strip() for k in args.keys.split(",") if k.strip())
    rows, skipped = [], []
    for p in sorted(Path(args.out).glob(f"r{args.round}_t*/summary.json")):
        s = json.loads(p.read_text())
        if s["status"] == "ok" and s.get("best_heldout_loss") is not None:
            rows.append({**s["trial"],
                         "best_heldout_loss": s["best_heldout_loss"]})
        else:
            skipped.append((p.parent.name, s["status"]))
    print(f"[analyze] {len(rows)} usable trials, {len(skipped)} skipped: "
          f"{skipped}")
    n_coef = (len(keys) + 1) * (len(keys) + 2) // 2
    if len(rows) < n_coef:
        raise SystemExit(f"too few usable trials ({len(rows)}) for a "
                         f"{len(keys)}-dim quadratic ({n_coef} coefficients)"
                         " — investigate the failures")
    fit = fit_quadratic(rows, keys)
    print(json.dumps(fit, indent=2))
    best = min(rows, key=lambda r: r["best_heldout_loss"])
    print("[analyze] best raw trial: "
          + " ".join(f"{k}={best[k]:.4g}" for k in keys)
          + f" loss={best['best_heldout_loss']:.5f}")
    Path(args.out, f"r{args.round}_fit.json").write_text(
        json.dumps({"fit": fit, "rows": rows, "skipped": skipped}, indent=2)
        + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("emit-trials")
    e.add_argument("--out", required=True)
    e.add_argument("--n", type=int, default=16)
    e.add_argument("--round", type=int, default=1)
    e.add_argument("--lr-min", type=float, default=3e-5)
    e.add_argument("--lr-max", type=float, default=3e-3)
    e.add_argument("--warm-min", type=float, default=0.01)
    e.add_argument("--warm-max", type=float, default=0.30)
    e.add_argument("--clip-min", type=float, default=0.1)
    e.add_argument("--clip-max", type=float, default=10.0)
    e.set_defaults(fn=emit_trials)

    def _common(p):
        p.add_argument("--config", default=str(REPO / "configs/two_si.yaml"))
        p.add_argument("--pairs", required=True)
        p.add_argument("--si", default="safety_refusal")
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--allow-tokenize", action="store_true")

    m = sub.add_parser("probe-mem")
    _common(m)
    m.add_argument("--steps", type=int, default=30)
    m.add_argument("--worst-case", action="store_true",
                   help="probe on the LONGEST examples (the densest batch "
                        "under length-bucketed batching) instead of a "
                        "sha-rank sample")
    m.set_defaults(fn=probe_mem)

    t = sub.add_parser("trial")
    _common(t)
    t.add_argument("--trials", required=True)
    t.add_argument("--index", type=int, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--positions", choices=("all", "response"), default="all",
                   help="the fit's mask leg: all-token (Class A) or "
                        "decode-only header-steered (Class C)")
    t.add_argument("--teacher-prefix-cache", action="store_true",
                   help="replay the shared SI prefix KV for gradient-"
                        "training teacher forwards (production default for "
                        "the long SIs; probes stay uncached by design)")
    t.add_argument("--no-length-sort", action="store_true",
                   help="keep the driver's unsorted batch composition "
                        "(slower: ragged pads + per-shape attention cost)")
    t.add_argument("--fit-subsample", type=int, default=0,
                   help="train on a deterministic sha-rank subpool of this "
                        "many fit examples (0 = full pool); heldout stays "
                        "full either way")
    t.add_argument("--probe-n", type=int, default=64)
    t.add_argument("--probe-every", type=int, default=250)
    t.add_argument("--dense-every", type=int, default=25)
    t.add_argument("--dense-until", type=int, default=250)
    t.add_argument("--norms-every", type=int, default=25)
    t.add_argument("--profile", action="store_true")
    t.add_argument("--no-wandb", action="store_true")
    t.set_defaults(fn=run_trial)

    a = sub.add_parser("analyze")
    a.add_argument("--out", required=True)
    a.add_argument("--round", type=int, default=1)
    a.add_argument("--keys", default="lr,warmup_frac,max_grad_norm",
                   help="comma-separated hyperparameters to fit jointly "
                        "(drop any held fixed in the round)")
    a.set_defaults(fn=analyze)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
