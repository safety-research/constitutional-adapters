"""bidpo optimizer plumbing for the two-model sweep: Adam betas/eps,
row-weighted gradient accumulation, the step budget, warmup_cosine +
warmup_frac, gradient clipping, the probe-callback seam, and the
non-finite abort (ported from the bidpo-adam branch @ ffe1167 and
extended; synthetic placeholder data ONLY — invented prompts/completions,
never recorded session or pool text).

Pins, in order:

1. BETAS/EPS REACH THE OPTIMIZER. train_bidpo used to call AdamW without
   them, so a config that swept Adam's conditioning knobs would have run
   16 identical trials at the torch defaults — a sweep whose knobs don't
   reach the optimizer is worse than no sweep. Asserted at the AdamW call
   itself (a spy), through the registry (the driver's path), and in the
   artifact metadata (the provenance a harvester reads).

2. ACCUMULATED == TRUE-BATCH GRADIENT. accum_steps micro-batches, each
   mean loss weighted by its ROW SHARE of the group, must equal one true
   batch of those rows under mean reduction — including a RAGGED tail,
   where a flat 1/accum_steps weighting silently biases the mean (the
   equivalence is the whole point of the knob: DPO's effective-batch
   floor on hardware that cannot hold the true batch's four graphs).

3. max_steps IS A HARD OPTIMIZER-STEP BUDGET, stopping mid-epoch and
   recording what really ran — the fair-comparison unit for a sweep whose
   trials share a step budget rather than an epoch count.

4. THE SCHEDULE IS LOCKSTEP'S, formula for formula (linear ramp
   lr*(i+1)/warmup then cosine to 0 over the planned budget), logged live
   as bidpo/lr; warmup_steps without a schedule is refused, not ignored.
   NEW: warmup_frac resolves against the run's OWN planned optimizer-step
   horizon (max_steps-capped), is mutually exclusive with warmup_steps,
   and is refused without a schedule too.

5. max_grad_norm CLIPS WHAT THE OPTIMIZER CONSUMES (new — the clip axis
   of the lr x clip sweep): the telemetry norm stays PRE-clip, the
   post-clip norm is logged beside it, and the gradient AdamW actually
   steps on is bounded by the setting.

6. GRAD-NORM TELEMETRY fires once per OPTIMIZER step (never per
   micro-batch) and lands final/max in the artifact metadata.

7. THE PROBE SEAM: probe_cb(step, obj) fires at the probe_every cadence
   with a transient VectorObject wrapping a detached CLONE of the current
   vec — usable for a mid-fit KL probe, never aliasing the live parameter.

8. NON-FINITE ABORT: a NaN/inf step loss raises BidpoNonFiniteError
   before the step is applied — a diverged fit never returns a vector.

9. SYNC DISCIPLINE (adversarial-review finding, 2026-08-25): telemetry
   must not force a CPU<->GPU sync per micro-batch — at accum_steps=32
   under direction="both" the historical per-scalar float()s cost ~96+
   cudaStreamSynchronize per optimizer step and destroyed the ~100% GPU
   duty the 27B fits depend on. Per-step scalars stay 0-dim DEVICE
   tensors, buffered, and materialise in ONE stacked D2H transfer at the
   logging cadence (log_every / probe cadence / end of fit / the abort);
   the ONLY per-step sync is the non-finite guard's single pre-reduced
   bool, read before opt.step(). Pinned by counting the sync-inducing
   tensor calls bidpo.py itself makes: the count per optimizer step is
   O(1) — constant in accum_steps — and the emitted telemetry is
   unchanged, just batched.
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from sisteer.data.instructions import Instruction
from sisteer.data.pairs import Pair
from sisteer.extraction.bidpo_lora import train_bidpo_lora
from sisteer.extraction.bidpo import (
    BidpoNonFiniteError,
    PrefExample,
    train_bidpo,
)
from sisteer.methods import ARMS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_two_si_objects as tso  # noqa: E402

LAYER = 1

INSTRUCTION = Instruction(
    id="testsi", category="style", text="Always answer in French.",
    anti_text=None, checker="contains")


def _prefs(n: int = 6) -> list[PrefExample]:
    texts = [
        ("Say something nice.", "Bonjour le monde", "Hello world"),
        ("Describe a tree.", "C'est un grand arbre", "It is a tall tree"),
        ("Name a color.", "Le ciel est bleu", "The sky is blue"),
        ("Greet a friend.", "Salut mon ami", "Hi my friend"),
        ("Count to two.", "Un puis deux", "One then two"),
        ("Describe rain.", "La pluie tombe doucement", "The rain falls"),
        ("Name an animal.", "Le chat dort", "The cat sleeps"),
    ]
    return [PrefExample(prompt=f"{p}\n", chosen=c, rejected=r)
            for p, c, r in texts[:n]]


def _pairs(n: int = 3) -> list[Pair]:
    prefs = _prefs(n)
    return [Pair(prompt=e.prompt.strip(), pos=e.chosen, neg=e.rejected,
                 instruction_id="testsi") for e in prefs]


# -- 1. betas/eps reach the optimizer, the registry, and the metadata -----------


def _spy_adamw(monkeypatch):
    """Record every AdamW construction's betas/eps, then build the real one."""
    real = torch.optim.AdamW
    calls: list[dict] = []

    def spy(params, **kw):
        calls.append(dict(kw))
        return real(params, **kw)

    monkeypatch.setattr(torch.optim, "AdamW", spy)
    return calls


def test_betas_eps_reach_adamw_and_metadata(tiny_llama, monkeypatch):
    model, tok = tiny_llama
    calls = _spy_adamw(monkeypatch)
    sv = train_bidpo(model, tok, _prefs(2), LAYER, epochs=1, batch_size=2,
                     betas=(0.8, 0.9999), eps=1e-4)
    assert calls and calls[0]["betas"] == (0.8, 0.9999)
    assert calls[0]["eps"] == 1e-4
    assert sv.metadata["betas"] == [0.8, 0.9999]
    assert sv.metadata["eps"] == 1e-4


def test_betas_eps_flow_through_the_registry(tiny_llama, monkeypatch):
    """The driver's exact path: train_kwargs.<arm> keys ride candidate_kw
    into ARMS['bidpo'].train as PROTECTED kwargs — they must reach the
    trainer (and its artifact metadata), never be TypeError-dropped."""
    model, tok = tiny_llama
    calls = _spy_adamw(monkeypatch)
    obj, used = tso._safe_registry_train(
        ARMS["bidpo"].train, model, tok, INSTRUCTION, _pairs(2),
        layer=LAYER, log_fn=None,
        candidate_kw={"epochs": 1, "batch_size": 2, "max_len": 64,
                      "seed": 0, "betas": [0.7, 0.95], "eps": 1e-6,
                      "accum_steps": 1},
        protected=frozenset({"betas", "eps", "accum_steps"}))
    assert used["betas"] == [0.7, 0.95] and used["eps"] == 1e-6
    assert calls and calls[0]["betas"] == (0.7, 0.95)
    assert obj.sv.metadata["betas"] == [0.7, 0.95]
    assert obj.sv.metadata["eps"] == 1e-6


def test_bad_betas_eps_refused(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="betas"):
        train_bidpo(model, tok, _prefs(2), LAYER, epochs=1,
                    betas=(0.9, 1.0))
    with pytest.raises(ValueError, match="eps"):
        train_bidpo(model, tok, _prefs(2), LAYER, epochs=1, eps=0.0)


# -- 2. accumulated gradient == true-batch gradient ------------------------------


def _fit_vec(model, tok, prefs, **kw):
    sv = train_bidpo(model, tok, prefs, LAYER, epochs=1, max_len=64,
                     lr=5e-4, seed=0, **kw)
    return sv.vectors[LAYER]


def test_accumulation_matches_true_batch(tiny_llama):
    """One optimizer step over 6 rows: 3 micro-batches of 2 accumulated
    must equal the true batch of 6 (same rows, mean reduction) — the
    returned vectors after that single Adam step agree to float noise."""
    model, tok = tiny_llama
    prefs = _prefs(6)
    accum = _fit_vec(model, tok, prefs, batch_size=2, accum_steps=3)
    true = _fit_vec(model, tok, prefs, batch_size=6, accum_steps=1)
    assert torch.allclose(accum, true, rtol=1e-4, atol=1e-6), (
        f"max |diff| = {(accum - true).abs().max():.3e}")
    assert not torch.equal(true, torch.zeros_like(true))


def test_accumulation_weights_the_ragged_tail_by_rows(tiny_llama):
    """5 rows as micro-batches of 2+2+1: row-share weights (2/5, 2/5, 1/5)
    reproduce the true 5-row batch. A flat 1/accum_steps weighting fails
    exactly here — the tail's rows would count 1.67x their share."""
    model, tok = tiny_llama
    prefs = _prefs(5)
    accum = _fit_vec(model, tok, prefs, batch_size=2, accum_steps=3)
    true = _fit_vec(model, tok, prefs, batch_size=5, accum_steps=1)
    assert torch.allclose(accum, true, rtol=1e-4, atol=1e-6), (
        f"max |diff| = {(accum - true).abs().max():.3e}")


def test_accumulation_recorded_in_metadata(tiny_llama):
    model, tok = tiny_llama
    sv = train_bidpo(model, tok, _prefs(4), LAYER, epochs=1, batch_size=2,
                     accum_steps=2, max_len=64)
    assert sv.metadata["accum_steps"] == 2
    assert sv.metadata["effective_batch"] == 4
    with pytest.raises(ValueError, match="accum_steps"):
        train_bidpo(model, tok, _prefs(2), LAYER, epochs=1, accum_steps=0)


# -- 3. max_steps is a hard optimizer-step budget --------------------------------


def test_max_steps_caps_the_budget_mid_epoch(tiny_llama):
    model, tok = tiny_llama
    steps_seen = []
    sv = train_bidpo(model, tok, _prefs(4), LAYER, epochs=3, batch_size=1,
                     max_len=64, max_steps=5,
                     log_fn=lambda m, s=None, **kw: steps_seen.append(s))
    assert sv.metadata["steps_run"] == 5
    assert sv.metadata["max_steps"] == 5
    assert sv.metadata["total_steps_planned"] == 5
    # 4 steps/epoch at batch 1: budget 5 ends inside epoch 2 — one full
    # epoch completed, and no step ran past the cap.
    assert sv.metadata["epochs_run"] == 1
    assert max(s for s in steps_seen if s is not None) == 5
    with pytest.raises(ValueError, match="max_steps"):
        train_bidpo(model, tok, _prefs(2), LAYER, epochs=1, max_steps=0)


# -- 4. the schedule is lockstep's, logged live ----------------------------------


def test_warmup_cosine_matches_lockstep_formula(tiny_llama):
    model, tok = tiny_llama
    lr = 1e-3
    logged = []

    def log_fn(m, step=None, **kw):
        if isinstance(m, dict) and "bidpo/lr" in m:
            logged.append(m["bidpo/lr"])

    train_bidpo(model, tok, _prefs(4), LAYER, epochs=1, batch_size=1,
                max_len=64, lr=lr, lr_schedule="warmup_cosine",
                warmup_steps=2, log_fn=log_fn)
    # total=4, warmup=2: ramp lr/2, lr; then cosine over the remaining 2.
    expect = [lr * 0.5, lr,
              lr * 0.5 * (1 + math.cos(math.pi * 0 / 2)),
              lr * 0.5 * (1 + math.cos(math.pi * 1 / 2))]
    assert logged == pytest.approx(expect)


def test_schedule_refusals_match_lockstep(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="lr_schedule"):
        train_bidpo(model, tok, _prefs(2), LAYER, epochs=1,
                    lr_schedule="linear")
    with pytest.raises(ValueError, match="warmup_steps"):
        train_bidpo(model, tok, _prefs(2), LAYER, epochs=1, warmup_steps=10)


def test_no_schedule_stays_fixed_lr_and_unlogged(tiny_llama):
    model, tok = tiny_llama
    payloads = []
    sv = train_bidpo(model, tok, _prefs(2), LAYER, epochs=1, batch_size=2,
                     max_len=64,
                     log_fn=lambda m, s=None, **kw: payloads.append(m))
    assert all("bidpo/lr" not in p for p in payloads if isinstance(p, dict))
    assert sv.metadata["lr_schedule"] is None
    assert sv.metadata["warmup_steps"] is None


# -- 4b. warmup_frac: the run's OWN budget, exclusive with warmup_steps ----------


def test_warmup_frac_resolves_against_the_runs_own_budget(tiny_llama):
    """epochs=2 x 4 steps/epoch = 8 planned, but max_steps=4 caps the
    horizon at 4 — warmup_frac 0.5 must resolve to 2 warmup steps (the
    run's OWN budget), not 4 (the uncapped epoch count), and the lr trace
    must be exactly lockstep's formula at warmup=2 / total=4."""
    model, tok = tiny_llama
    lr = 1e-3
    logged = []

    def log_fn(m, step=None, **kw):
        if isinstance(m, dict) and "bidpo/lr" in m:
            logged.append(m["bidpo/lr"])

    sv = train_bidpo(model, tok, _prefs(4), LAYER, epochs=2, batch_size=1,
                     max_len=64, lr=lr, lr_schedule="warmup_cosine",
                     warmup_frac=0.5, max_steps=4, log_fn=log_fn)
    assert sv.metadata["warmup_steps"] == 2
    assert sv.metadata["warmup_frac"] == 0.5
    assert sv.metadata["total_steps_planned"] == 4
    expect = [lr * 0.5, lr,
              lr * 0.5 * (1 + math.cos(math.pi * 0 / 2)),
              lr * 0.5 * (1 + math.cos(math.pi * 1 / 2))]
    assert logged == pytest.approx(expect)


def test_warmup_frac_floors_at_one_step(tiny_llama):
    """A tiny fraction of a tiny budget still warms up for >= 1 step —
    lockstep's _lr_at_factory divides by warmup_steps, so 0 would be a
    silent ZeroDivisionError waiting on the smallest trial."""
    model, tok = tiny_llama
    sv = train_bidpo(model, tok, _prefs(2), LAYER, epochs=1, batch_size=1,
                     max_len=64, lr_schedule="warmup_cosine",
                     warmup_frac=0.05)
    assert sv.metadata["warmup_steps"] == 1  # round(0.05 * 2) == 0 -> floor 1


def test_warmup_frac_exclusive_with_warmup_steps(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_bidpo(model, tok, _prefs(2), LAYER, epochs=1,
                    lr_schedule="warmup_cosine", warmup_steps=2,
                    warmup_frac=0.1)


def test_warmup_frac_refused_without_schedule_or_out_of_range(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="warmup_frac"):
        train_bidpo(model, tok, _prefs(2), LAYER, epochs=1, warmup_frac=0.1)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="warmup_frac"):
            train_bidpo(model, tok, _prefs(2), LAYER, epochs=1,
                        lr_schedule="warmup_cosine", warmup_frac=bad)


# -- 5. max_grad_norm clips what the optimizer consumes --------------------------


def test_max_grad_norm_bounds_the_step(tiny_llama, monkeypatch):
    """Clip far below the raw norms so every step clips: the logged
    pre-clip norm must exceed the logged post-clip norm, the post-clip
    norm must respect the bound, and — the part that matters — the
    gradient AdamW actually steps on is the CLIPPED one (spied at
    opt.step() time). Telemetry metadata stays pre-clip."""
    model, tok = tiny_llama
    clip = 1e-6
    consumed = []
    real_step = torch.optim.AdamW.step

    def spy(self, *a, **kw):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    consumed.append(float(p.grad.norm()))
        return real_step(self, *a, **kw)

    monkeypatch.setattr(torch.optim.AdamW, "step", spy)
    payloads = []
    sv = train_bidpo(model, tok, _prefs(4), LAYER, epochs=1, batch_size=2,
                     max_len=64, max_grad_norm=clip,
                     log_fn=lambda m, s=None, **kw: payloads.append(m))
    pre = [m["bidpo/grad_norm"] for m in payloads
           if isinstance(m, dict) and "bidpo/grad_norm" in m]
    post = [m["bidpo/grad_norm_clipped"] for m in payloads
            if isinstance(m, dict) and "bidpo/grad_norm_clipped" in m]
    assert len(pre) == len(post) == 2
    assert all(p > q for p, q in zip(pre, post)), (pre, post)
    assert all(q <= clip * 1.001 for q in post)
    assert consumed and all(n <= clip * 1.001 for n in consumed)
    assert sv.metadata["max_grad_norm"] == clip
    # final/max grad-norm metadata are the PRE-clip series — comparable
    # across clip settings, the divergence signal a monitor watches.
    assert sv.metadata["final_grad_norm"] == pre[-1]
    assert sv.metadata["max_grad_norm_seen"] == max(pre)


def test_no_clip_by_default_and_nonpositive_refused(tiny_llama):
    model, tok = tiny_llama
    payloads = []
    sv = train_bidpo(model, tok, _prefs(2), LAYER, epochs=1, batch_size=2,
                     max_len=64,
                     log_fn=lambda m, s=None, **kw: payloads.append(m))
    assert all("bidpo/grad_norm_clipped" not in p for p in payloads
               if isinstance(p, dict))
    assert sv.metadata["max_grad_norm"] is None
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="max_grad_norm"):
            train_bidpo(model, tok, _prefs(2), LAYER, epochs=1,
                        max_grad_norm=bad)


# -- 6. grad-norm telemetry: once per OPTIMIZER step, into the metadata ----------


def test_grad_norm_logged_once_per_optimizer_step(tiny_llama):
    """4 rows at micro-batch 1 x accum 2 = 2 optimizer steps: exactly TWO
    grad_norm entries (one per step, never per micro-batch — a micro-batch
    norm is a different quantity and incomparable across accum settings),
    and the artifact metadata carries final/max for the runs table."""
    model, tok = tiny_llama
    payloads = []
    sv = train_bidpo(model, tok, _prefs(4), LAYER, epochs=1, batch_size=1,
                     accum_steps=2, max_len=64,
                     log_fn=lambda m, s=None, **kw: payloads.append(m))
    norms = [m["bidpo/grad_norm"] for m in payloads
             if isinstance(m, dict) and "bidpo/grad_norm" in m]
    assert len(norms) == 2
    assert all(n > 0 for n in norms)
    assert sv.metadata["final_grad_norm"] == norms[-1]
    assert sv.metadata["max_grad_norm_seen"] == max(norms)


# -- 7. the probe seam: cadence, snapshot semantics, protocol usability ----------


def test_probe_cb_fires_at_cadence_with_a_usable_object(tiny_llama):
    """4 optimizer steps at probe_every=2 fire at steps 2 and 4, each with
    a transient VectorObject: a detached CLONE (the step-2 capture keeps
    its step-2 value while training marches on; the step-4 capture equals
    the returned artifact), servable through the object protocol."""
    model, tok = tiny_llama
    seen = []
    sv = train_bidpo(model, tok, _prefs(4), LAYER, epochs=1, batch_size=1,
                     max_len=64,
                     probe_cb=lambda step, obj: seen.append((step, obj)),
                     probe_every=2)
    assert [s for s, _ in seen] == [2, 4]
    first, last = seen[0][1], seen[1][1]
    assert first.kind == "vector"
    v2 = first.sv.vectors[LAYER]
    v4 = last.sv.vectors[LAYER]
    assert not v2.requires_grad
    # Snapshots, never aliases of the live parameter: had the object held
    # the live vec, both captures would have marched to the final value.
    assert not torch.allclose(v2, v4)
    assert torch.allclose(v4.float().cpu(), sv.vectors[LAYER])
    # The fit's deploy convention travels with the transient object.
    assert first.skip_bos is True and first.positions == "all"
    # And it is servable: a steered forward under apply() works.
    batch = tok(["Say something nice."], return_tensors="pt")
    with first.apply(model):
        out = model(**batch)
    assert torch.isfinite(out.logits).all()


def test_probe_cb_and_probe_every_come_together(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="probe"):
        train_bidpo(model, tok, _prefs(2), LAYER, epochs=1,
                    probe_cb=lambda s, o: None)
    with pytest.raises(ValueError, match="probe"):
        train_bidpo(model, tok, _prefs(2), LAYER, epochs=1, probe_every=5)
    with pytest.raises(ValueError, match="probe_every"):
        train_bidpo(model, tok, _prefs(2), LAYER, epochs=1,
                    probe_cb=lambda s, o: None, probe_every=0)


# -- 8. non-finite abort: a diverged fit never returns a vector ------------------


def test_non_finite_loss_aborts_never_returns_a_vector(tiny_llama,
                                                       monkeypatch):
    model, tok = tiny_llama
    # NaN loss (a poisoned logsigmoid keeps the graph, so backward runs
    # and the guard sees both a NaN loss and a NaN grad norm).
    monkeypatch.setattr(F, "logsigmoid", lambda x: x * float("nan"))
    with pytest.raises(BidpoNonFiniteError, match="non-finite"):
        train_bidpo(model, tok, _prefs(2), LAYER, epochs=1, batch_size=2,
                    max_len=64)
    # inf, not just NaN.
    monkeypatch.setattr(F, "logsigmoid", lambda x: x - float("inf"))
    with pytest.raises(BidpoNonFiniteError):
        train_bidpo(model, tok, _prefs(2), LAYER, epochs=1, batch_size=2,
                    max_len=64)
    # A distinct, catchable class: config errors stay ValueError, the
    # divergence abort is its own type (the sweep files it as `failed`).
    assert issubclass(BidpoNonFiniteError, RuntimeError)
    assert not issubclass(BidpoNonFiniteError, ValueError)


def test_abort_flushes_the_clean_steps_telemetry(tiny_llama, monkeypatch):
    """The abort must not eat the buffered telemetry: poison the loss from
    optimizer step 3 (direction='both' calls logsigmoid twice per step at
    accum_steps=1) — the fit dies with BidpoNonFiniteError naming step 3,
    AND steps 1-2's buffered records are emitted first (flush-before-
    raise). The poisoned step itself is never logged, exactly as the
    historical per-step emission never reached it."""
    model, tok = tiny_llama
    real = F.logsigmoid
    calls = {"n": 0}

    def poisoned(x):
        calls["n"] += 1
        return real(x) if calls["n"] <= 4 else x * float("nan")

    monkeypatch.setattr(F, "logsigmoid", poisoned)
    payloads: list[tuple] = []
    with pytest.raises(BidpoNonFiniteError, match="step 3"):
        train_bidpo(model, tok, _prefs(4), LAYER, epochs=1, batch_size=1,
                    max_len=64,
                    log_fn=lambda m, s=None, **kw: payloads.append((s, m)))
    assert [s for s, m in payloads if "bidpo/loss" in m] == [1, 2]


# -- 9. sync discipline: D2H materialisations per optimizer step are O(1) --------
#
# On CUDA every .item()/.tolist()/.cpu()/float()/bool() on a device tensor
# forces a cudaStreamSynchronize; CPU CI can't observe the stall, so the
# pin counts the sync-INDUCING tensor calls instead, attributed by the
# IMMEDIATE caller frame so only bidpo.py's own materialisations count
# (the model's internals are not this discipline's to spend).

_SYNC_CALLS = ("item", "tolist", "cpu", "numpy",
               "__float__", "__bool__", "__int__")


def _count_bidpo_materialisations(monkeypatch) -> list[str]:
    counted: list[str] = []

    def wrap(real, name):
        def wrapper(t, *a, **kw):
            if sys._getframe(1).f_code.co_filename.endswith("bidpo.py"):
                counted.append(name)
            return real(t, *a, **kw)
        return wrapper

    for name in _SYNC_CALLS:
        monkeypatch.setattr(torch.Tensor, name,
                            wrap(getattr(torch.Tensor, name), name))
    return counted


def _prefs_n(n: int) -> list[PrefExample]:
    base = _prefs(7)
    return [base[i % len(base)] for i in range(n)]


def test_d2h_syncs_per_optimizer_step_constant_in_accum_steps(tiny_llama,
                                                              monkeypatch):
    """The F6 regression pin: 16 rows at micro-batch 1 under
    direction='both', swept over accum_steps in {1, 4, 16} (16, 4 and 1
    optimizer steps over the SAME 16 micro-batches). The historical code
    materialised ~3 scalars per MICRO-batch (loss + both margins) plus 2
    per step — ~51 calls even at accum_steps=16 — so the per-step budget
    below (1 finite-check bool per optimizer step + a bounded number of
    stacked flush/end transfers) fails it at every accum setting. The new
    code's counts must track optimizer steps, never micro-batches, and the
    batched emission must still deliver one record per optimizer step."""
    model, tok = tiny_llama
    prefs = _prefs_n(16)
    counted = _count_bidpo_materialisations(monkeypatch)
    counts: dict[int, int] = {}
    records: dict[int, list] = {}
    for accum in (1, 4, 16):
        payloads: list[dict] = []
        counted.clear()
        train_bidpo(model, tok, prefs, LAYER, epochs=1, batch_size=1,
                    accum_steps=accum, max_len=64, direction="both",
                    log_fn=lambda m, s=None, _p=payloads, **kw: _p.append(m))
        counts[accum] = len(counted)
        records[accum] = [m for m in payloads if "bidpo/loss" in m]
    for accum, c in counts.items():
        steps = 16 // accum
        # 1 pre-reduced finite bool per optimizer step, plus a bounded
        # constant for the stacked flush transfer(s) and the final
        # vectors-to-CPU move. O(1) per step; NEVER a term in accum_steps.
        assert c <= 2 * steps + 6, (
            f"accum_steps={accum}: {c} D2H materialisations for {steps} "
            f"optimizer steps — the per-micro-batch sync leak is back "
            f"(all counts: {counts})")
    # Fixed micro-batch count, fewer optimizer steps -> never more syncs.
    assert counts[16] <= counts[4] <= counts[1], counts
    # Batching the emission dropped nothing: one record per optimizer
    # step, margins and norms still present in each.
    for accum, recs in records.items():
        assert len(recs) == 16 // accum
        assert all("bidpo/fwd_margin" in m and "bidpo/rev_margin" in m
                   and "bidpo/vec_norm" in m for m in recs)


def test_sync_budget_holds_with_every_telemetry_arm_on(tiny_llama,
                                                       monkeypatch):
    """The fullest record — clipping (clip_grad_norm_'s RETURNED norm is
    the pre-clip telemetry, never a recomputed float()), the lr schedule,
    the stop rule (one reduced bool per EPOCH), and the probe seam (a
    flush point, no sync of its own: the transient object is a device
    clone) — stays inside the same O(steps) budget."""
    model, tok = tiny_llama
    counted = _count_bidpo_materialisations(monkeypatch)
    payloads: list[dict] = []
    probed: list[int] = []
    sv = train_bidpo(
        model, tok, _prefs_n(8), LAYER, epochs=2, batch_size=1,
        accum_steps=4, max_len=64, direction="both",
        max_grad_norm=1e-6, lr_schedule="warmup_cosine", warmup_frac=0.25,
        stop_pref_prob=0.51,
        probe_cb=lambda s, obj: probed.append(s), probe_every=2,
        log_fn=lambda m, s=None, **kw: payloads.append(m))
    steps = sv.metadata["steps_run"]
    epochs = sv.metadata["epochs_run"]
    assert counted, "the counter armed nothing — the pin is vacuous"
    assert len(counted) <= 2 * steps + epochs + 6, (
        f"{len(counted)} D2H materialisations for {steps} optimizer steps "
        f"/ {epochs} epochs: {counted}")
    assert probed == [s for s in range(2, steps + 1, 2)]
    recs = [m for m in payloads if "bidpo/loss" in m]
    assert len(recs) == steps
    assert all("bidpo/grad_norm" in m and "bidpo/grad_norm_clipped" in m
               and "bidpo/lr" in m for m in recs)


# -- 9b. log_every: batched emission, unchanged content --------------------------


def test_log_every_batches_emission_without_changing_content(tiny_llama):
    """log_every only defers the log_fn calls — flushing every step and
    flushing once at the end must emit byte-identical record sequences
    (same steps, same keys, same float values), and the metadata loss/
    grad series must agree with the emitted one."""
    model, tok = tiny_llama

    def run(log_every):
        payloads: list[tuple] = []
        sv = train_bidpo(
            model, tok, _prefs(4), LAYER, epochs=1, batch_size=1,
            max_len=64, seed=0, log_every=log_every,
            log_fn=lambda m, s=None, **kw: payloads.append((s, dict(m))))
        return payloads, sv

    every_step, sv1 = run(1)
    end_only, sv2 = run(100)
    assert every_step == end_only
    losses = [m["bidpo/loss"] for _s, m in every_step if "bidpo/loss" in m]
    assert sv1.metadata["final_loss"] == losses[-1]
    assert sv1.metadata["final_loss"] == sv2.metadata["final_loss"]


def test_log_every_validated(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="log_every"):
        train_bidpo(model, tok, _prefs(2), LAYER, epochs=1, log_every=0)


# -- ported with the bidpo-adam-sweep merge: the LoRA-substrate engine pins,
# and the W&B seams the sweep harvests through ------------------------------


def test_lora_engine_accumulation_matches_true_batch(tiny_llama):
    """The same equivalence on the LoRA substrate (train_bidpo_object via
    train_bidpo_lora): the engines are held knob-for-knob identical."""
    model, tok = tiny_llama
    prefs = _prefs(5)

    def fit(bs, accum):
        obj = train_bidpo_lora(model, tok, prefs, LAYER, epochs=1,
                               batch_size=bs, accum_steps=accum, max_len=64,
                               lr=5e-4, seed=0)
        a, b = obj.trainable
        return torch.cat([a.detach().flatten(), b.detach().flatten()])

    accum = fit(2, 3)
    true = fit(5, 1)
    assert torch.allclose(accum, true, rtol=1e-4, atol=1e-6), (
        f"max |diff| = {(accum - true).abs().max():.3e}")


def test_lora_engine_grad_norm_logged_per_step(tiny_llama):
    model, tok = tiny_llama
    payloads = []
    train_bidpo_lora(model, tok, _prefs(4), LAYER, epochs=1, batch_size=1,
                     accum_steps=2, max_len=64,
                     log_fn=lambda m, s=None, **kw: payloads.append(m))
    norms = [m["bidpo_lora/grad_norm"] for m in payloads
             if isinstance(m, dict) and "bidpo_lora/grad_norm" in m]
    assert len(norms) == 2
    assert all(n > 0 for n in norms)


# -- 5. the W&B seams the sweep harvests through ----------------------------------


def _fake_wandb(monkeypatch):
    captured = {}
    mod = types.ModuleType("wandb")

    def init(**kw):
        captured.update(kw)
        return types.SimpleNamespace(summary={}, finish=lambda **_: None)

    mod.init = init
    mod.Settings = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "wandb", mod)
    # The test env disables W&B (conftest WANDB_MODE=disabled) — this test
    # is ABOUT the init path, so re-enable it for the call.
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    # a config with no wandb_project logs to $WANDB_PROJECT, which must resolve (sisteer.site)
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    return captured


def test_tt_wandb_fit_shapes_the_run_at_init(monkeypatch):
    captured = _fake_wandb(monkeypatch)
    blob = {"name": "trial-03", "group": "bidpo-adam-sweep",
            "job_type": "sweep-trial", "tags": ["bidpo-adam-sweep"],
            "config": {"trial_idx": 3, "lr": 1e-3, "eps": 1e-7}}
    monkeypatch.setenv("TT_WANDB_FIT", json.dumps(blob))
    run = tso.wandb_fit_run({}, "testsi", ["bidpo"])
    assert run is not None
    assert captured["name"] == "trial-03"
    assert captured["group"] == "bidpo-adam-sweep"
    assert captured["job_type"] == "sweep-trial"
    assert captured["tags"] == ["bidpo-adam-sweep"]
    # Sweep config merges OVER the computed identifiers, which survive.
    assert captured["config"]["trial_idx"] == 3
    assert captured["config"]["eps"] == 1e-7
    assert captured["config"]["si_id"] == "testsi"


def test_tt_wandb_fit_unparseable_falls_back_loudly(monkeypatch, capsys):
    captured = _fake_wandb(monkeypatch)
    monkeypatch.setenv("TT_WANDB_FIT", "{not json")
    run = tso.wandb_fit_run({}, "testsi", ["bidpo"])
    assert run is not None  # never a lost run
    assert captured["name"].startswith("fit-testsi")
    assert "TT_WANDB_FIT ignored" in capsys.readouterr().err


def test_fit_summary_lands_floor_ratio_pass_or_fail():
    """The single most load-bearing seam of the sweep: a FloorRatioError
    trial (the EXPECTED outcome for most exploration trials) still gets a
    summary row with the ratio, status floor_refused, and the trainer's
    numeric self-report — flat keys for the single-arm shape."""
    outcome = tso.FitOutcome(si_id="testsi")
    outcome.arms["bidpo"] = tso.ArmOutcome(
        status="failed", error_type="FloorRatioError", error="5.28x > 3.0",
        detail={"floor_ratio": 5.28, "final_probe_loss": 0.3317,
                "noop_floor": 0.0628, "floor_ratio_max": 3.0,
                "trainer_metadata": {"min_loss": 0.011, "final_loss": 0.02,
                                     "steps_run": 1000, "epochs_run": 23,
                                     "effective_batch": 32,
                                     "final_grad_norm": 0.4,
                                     "max_grad_norm_seen": 3.7,
                                     "final_vec_norm": 2.03}})
    run = types.SimpleNamespace(summary={})
    tso._write_fit_summary(run, outcome, ["bidpo"])
    s = run.summary
    assert s["status"] == "floor_refused"
    assert s["floor_ratio"] == 5.28
    assert s["final_probe_loss"] == 0.3317
    assert s["noop_floor"] == 0.0628
    assert s["final_vec_norm"] == 2.03
    assert s["min_loss"] == 0.011
    assert s["final_grad_norm"] == 0.4
    assert s["max_grad_norm_seen"] == 3.7
    assert s["steps_completed"] == 1000
    assert s["effective_batch"] == 32
    assert s["killed_early"] is False


def test_floor_refused_solo_arm_keeps_its_fit_record(tiny_llama, tmp_path):
    """END TO END: a solo arm the floor guard REFUSES still lands its
    floor_ratio in the outcome detail (the report) and from there in
    run.summary. Before 2026-08-25 the solo except recorded status/error
    only — the wd2 baseline's report carried NO ratio for its failed arm,
    which is precisely the number a sweep exists to harvest. An impossible
    bar (floor_ratio_max 1e-6) forces the refusal on the loreft solo arm;
    the fixture never touches pool data."""
    model, tok = tiny_llama
    si = "safety_refusal"  # resolvable id; pairs stay synthetic
    pairs = [Pair(prompt=e.prompt.strip(), pos=e.chosen, neg=e.rejected,
                  instruction_id=si) for e in _prefs(3)]
    cfg = {"layer": 1, "completion_tokens": 16, "max_len": 256,
           "max_len_by_si": {si: 256},
           "train_kwargs": {"default": {"epochs": 1, "seed": 0},
                            "loreft": {"floor_ratio_max": 1e-6}},
           "si_ids": [si]}
    outcome = tso.train_si_objects(
        model, tok, si, pairs, ["loreft"], cfg=cfg, save_dir=tmp_path)
    res = outcome.arms["loreft"]
    assert res.status == "failed"
    assert res.error_type == "FloorRatioError"
    assert res.detail is not None, "the refusal dropped its fit record"
    assert res.detail["floor_ratio"] > 0
    assert res.detail["floor_ratio_max"] == 1e-6
    run = types.SimpleNamespace(summary={})
    tso._write_fit_summary(run, outcome, ["loreft"])
    assert run.summary["status"] == "floor_refused"
    assert run.summary["floor_ratio"] == res.detail["floor_ratio"]


def test_fit_summary_prefixes_multi_arm_and_keeps_trained_status():
    outcome = tso.FitOutcome(si_id="testsi")
    outcome.arms["bidpo"] = tso.ArmOutcome(
        status="trained", detail={"floor_ratio": 1.2})
    outcome.arms["vector"] = tso.ArmOutcome(
        status="failed", error_type="RuntimeError", error="boom",
        detail={})
    run = types.SimpleNamespace(summary={})
    tso._write_fit_summary(run, outcome, ["bidpo", "vector"])
    assert run.summary["bidpo/status"] == "trained"
    assert run.summary["bidpo/floor_ratio"] == 1.2
    assert run.summary["vector/status"] == "failed"
