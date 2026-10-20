"""Row-sharded fit parallelism (accel-fit): the exact-semantics contract.

What must hold (the load-bearing invariants):

- ROW-SHARDING NEVER CHANGES THE OPTIMIZATION — for the trainers that KEEP
  a rowshard path on this trunk (the capture collection pass, lofit,
  loreft): a k-rank fit trains on the same batches, in the same order, at
  the same effective batch size and step count as the single-process fit —
  parameters match to fp reduction order, held-out curves and stopping
  epochs match exactly.
- bidpo, lockstep and psr deliberately have NO rowshard path here: this
  trunk packs arms 1-per-GPU (extraction.ddp not ported into them —
  row-sharding measured counterproductive at these widths, all_reduce
  dominates; scripts/train_two_si_objects.py:3258-3260, and arm_width's
  arm_gpus>1 SystemExit at :3307-3319). Their tests pin that ABSENCE — an
  ambient FitParallelCtx must be inert and the modules reference none of
  the fit-parallel runtime — and FLIP on a re-port, pointing at accel-fit
  @ 1ae8662 for the real 2-rank equivalence tests to restore.
- A lockstep-compatible arm fanned out as a SINGLETON group is bitwise the
  arm's in-group result, and ``group_batch_size`` over a one-arm list is
  PINNED to that arm's own resolved batch size (asked for explicitly as a
  test, not an emergent property).
- The world size is pinned by the config: a pool that disagrees REFUSES.
- Sharded collection passes return the same activations as unsharded, on
  every rank.
- Per-arm config kwargs still reach the trainer through the parallel
  driver's path (the alphakl_soft beta regression), and every sidecar
  records the parallel block with an EXPLICIT effective batch.

CPU-only: tiny_llama + gloo process groups over synthetic pairs (invented
text, never real store data).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction import ddp
from sisteer.extraction.distill import EarlyStopSpec
from sisteer.extraction.lockstep import (
    LockstepArm,
    build_lockstep_object,
    default_lockstep_params,
    train_lockstep,
)

import train_two_si_objects as tt

INSTRUCTION = BY_ID["french"]
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _pairs(n=5):
    data = [
        ("Say something nice.", "Bonjour le monde", "Hello world"),
        ("Describe a tree.", "C'est un grand arbre", "It is a tall tree"),
        ("Greet the reader.", "Salut a tous", "Hi everyone"),
        ("Name a color.", "Le bleu est joli", "Blue is nice"),
        ("Count to two.", "Un et deux", "One and two"),
        ("Name an animal.", "Le chat dort", "The cat sleeps"),
    ]
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id="french")
            for p, pos, neg in data[:n]]


TRAIN_KW = dict(epochs=2, batch_size=2, max_len=96, seed=0)


# --------------------------------------------------------------------------- #
# Pure helpers                                                                  #
# --------------------------------------------------------------------------- #


def test_rows_for_rank_strided_and_exhaustive():
    ctx0 = ddp.FitParallelCtx(rank=0, world_size=2,
                              device=torch.device("cpu"))
    ctx1 = ddp.FitParallelCtx(rank=1, world_size=2,
                              device=torch.device("cpu"))
    assert ddp.rows_for_rank(5, None) == [0, 1, 2, 3, 4]
    r0, r1 = ddp.rows_for_rank(5, ctx0), ddp.rows_for_rank(5, ctx1)
    assert sorted(r0 + r1) == [0, 1, 2, 3, 4]
    assert not set(r0) & set(r1)
    # Ragged: a 1-row batch leaves rank 1 with nothing — by design (it
    # still joins every collective).
    assert ddp.rows_for_rank(1, ctx1) == []


def test_scale_loss_parts_recombines_means():
    # Two ranks holding 3 and 1 of a 4-element mean: the shares must sum to
    # the global mean exactly.
    vals = torch.tensor([1.0, 2.0, 3.0, 10.0])
    global_mean = vals.mean()
    a = ddp.scale_loss_parts([(vals[:3].mean(), 3, 4)], vals.device)
    b = ddp.scale_loss_parts([(vals[3:].mean(), 1, 4)], vals.device)
    assert torch.allclose(a + b, global_mean)
    # An empty rank contributes exactly zero.
    z = ddp.scale_loss_parts([(None, 0, 4)], vals.device)
    assert float(z) == 0.0


def test_rowshard_width_caps_at_batch_size():
    assert ddp.rowshard_width(batch_size=4, requested=8) == 4
    assert ddp.rowshard_width(batch_size=2, requested=8) == 2
    assert ddp.rowshard_width(batch_size=8, requested=2) == 2
    assert ddp.rowshard_width(batch_size=4, requested=1) == 1


# --------------------------------------------------------------------------- #
# The singleton pin                                                             #
# --------------------------------------------------------------------------- #


def test_singleton_group_batch_size_pins_to_arm():
    """group_batch_size over a ONE-arm list must resolve to exactly that
    arm's own batch size — the property that makes fanning a lockstep arm
    out as a singleton group carry the group==solo bitwise contract (and
    the current wave's shared width) over unchanged."""
    from sisteer.eval.train_budget import resolve_batch_size
    from sisteer.extraction.lockstep import _LOCKSTEP_COMPATIBLE

    cfg = {"train_kwargs": {"default": {"batch_size": 4}},
           "batch_size_by_arm": {"lora_r1": 3}}
    for arm in sorted(_LOCKSTEP_COMPATIBLE):
        assert tt.group_batch_size(cfg, "safety_refusal", [arm]) == \
            resolve_batch_size(cfg, "safety_refusal", arm)


def test_singleton_lockstep_bitwise_equals_group_member(tiny_llama):
    """An arm trained in a singleton lockstep group is BITWISE the same arm
    trained inside a multi-arm group (same seed/bs/data): fan-out mode's
    semantics == the current driver's, arm for arm."""
    model, tok = tiny_llama
    pairs = _pairs(4)
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)

    def build(arm):
        obj = build_lockstep_object(arm, model, tok, INSTRUCTION, pairs,
                                    layer=1)
        return LockstepArm(arm, obj, default_lockstep_params(arm))

    grouped = [build("vector_all"), build("lora_r1")]
    train_lockstep(model, tok, grouped, examples, **TRAIN_KW)
    for arm_name in ("vector_all", "lora_r1"):
        solo = [build(arm_name)]
        train_lockstep(model, tok, solo, examples, **TRAIN_KW)
        grouped_arm = next(a for a in grouped if a.name == arm_name)
        for p_solo, p_group in zip(solo[0].obj.trainable,
                                   grouped_arm.obj.trainable):
            assert torch.equal(p_solo.detach(), p_group.detach()), \
                f"{arm_name}: singleton != in-group (bitwise)"


# --------------------------------------------------------------------------- #
# Row-sharded == single-process (2-rank gloo)                                   #
# --------------------------------------------------------------------------- #

_WORLD = 2


def _init_worker(rank: int, tmpdir: str):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{tmpdir}/pg_init", rank=rank,
        world_size=_WORLD)
    ctx = ddp.FitParallelCtx(rank=rank, world_size=_WORLD,
                             device=torch.device("cpu"))
    ddp.set_active(ctx)
    return ctx


def _solo_trainer_worker(rank: int, tmpdir: str, which: str):
    from sisteer.methods import ARMS
    from sisteer.models import load_model_and_tokenizer

    ctx = _init_worker(rank, tmpdir)
    model, tok = load_model_and_tokenizer(TINY, device="cpu",
                                          dtype=torch.float32)
    obj = ARMS[which].train(model, tok, INSTRUCTION, _pairs(5), layer=1,
                            log_fn=None, **TRAIN_KW)
    if ctx.is_main:
        params = getattr(obj, "trainable", None)
        if not params:  # SteeringVector-returning arms (bidpo-family)
            params = [v for v in obj.vectors.values()]
        torch.save({"params": [torch.as_tensor(p.detach() if hasattr(p, "detach") else p)
                               for p in params]}, f"{tmpdir}/sharded.pt")


def _capture_worker(rank: int, tmpdir: str):
    from sisteer.extraction.capture import collect_pooled_activations
    from sisteer.models import load_model_and_tokenizer

    ctx = _init_worker(rank, tmpdir)
    model, tok = load_model_and_tokenizer(TINY, device="cpu",
                                          dtype=torch.float32)
    texts = [p.prompt for p in _pairs(5)]
    acts = collect_pooled_activations(model, tok, texts, [0, 1],
                                      position="last", batch_size=1)
    torch.save({l: v for l, v in acts.acts.items()},
               f"{tmpdir}/capture.rank{rank}.pt")


def _spawn(fn, tmpdir: str, *extra):
    import torch.multiprocessing as mp

    mp.start_processes(fn, args=(tmpdir, *extra), nprocs=_WORLD,
                       join=True, start_method="spawn")


@pytest.mark.parametrize("arm", ["loreft", "lofit"])
def test_rowshard_solo_trainer_matches_single_process(tmp_path, arm):
    """2-rank row-sharded registry trainers == single-process, for one
    representative of each edited loss family that KEEPS a rowshard path
    on this trunk: CE (loreft), two-phase CE+L1 (lofit). (psr_mse_all left
    this list with the deliberate rowshard drop — psr trains
    single-process here, so a 2-rank run of it was vacuously equal; its
    absence pin is test_psr_has_no_rowshard_path below.)"""
    from sisteer.methods import ARMS
    from sisteer.models import load_model_and_tokenizer

    _spawn(_solo_trainer_worker, str(tmp_path), arm)
    sharded = torch.load(tmp_path / "sharded.pt", weights_only=False)

    model, tok = load_model_and_tokenizer(TINY, device="cpu",
                                          dtype=torch.float32)
    obj = ARMS[arm].train(model, tok, INSTRUCTION, _pairs(5), layer=1,
                          log_fn=None, **TRAIN_KW)
    ref_params = getattr(obj, "trainable", None)
    if not ref_params:
        ref_params = [v for v in obj.vectors.values()]
    assert len(sharded["params"]) == len(ref_params)
    for p_sh, p_ref in zip(sharded["params"], ref_params):
        p_ref = p_ref.detach() if hasattr(p_ref, "detach") else p_ref
        assert torch.allclose(p_sh, torch.as_tensor(p_ref), atol=1e-5), \
            f"{arm}: row-sharded params drifted from single-process"


def test_sharded_capture_matches_unsharded(tmp_path):
    """collect_pooled_activations under a 2-rank context returns the same
    full, ordered activations as the unsharded call — on EVERY rank (the
    closed-form solves downstream run replicated on identical inputs)."""
    from sisteer.extraction.capture import collect_pooled_activations
    from sisteer.models import load_model_and_tokenizer

    _spawn(_capture_worker, str(tmp_path))
    model, tok = load_model_and_tokenizer(TINY, device="cpu",
                                          dtype=torch.float32)
    texts = [p.prompt for p in _pairs(5)]
    ref = collect_pooled_activations(model, tok, texts, [0, 1],
                                     position="last", batch_size=1)
    for rank in range(_WORLD):
        got = torch.load(tmp_path / f"capture.rank{rank}.pt",
                         weights_only=False)
        for l in (0, 1):
            # batch_size=1 -> identical batch composition -> exact equality
            assert torch.equal(got[l], ref.acts[l]), \
                f"rank {rank} layer {l}: sharded capture != unsharded"


# --------------------------------------------------------------------------- #
# Rowshard deliberately ABSENT: bidpo / lockstep / psr (this trunk)             #
# --------------------------------------------------------------------------- #
#
# This trunk packs arms 1-per-GPU: row-sharding measured counterproductive
# at these widths (all_reduce dominates) and extraction.ddp is deliberately
# NOT ported into bidpo, lockstep, or psr — the in-job scheduler header
# states the decision (scripts/train_two_si_objects.py:3258-3260) and
# arm_width refuses data_parallel.arm_gpus > 1 with a loud SystemExit
# (:3307-3319). The three tests below pin that ABSENCE instead of faking
# the old 2-rank equivalence (which passed vacuously once the trainers
# stopped reading the context: every rank did full-batch work). Each
# asserts two things:
#
#   1. the module references none of the fit-parallel runtime's entry
#      points (any re-port must import extraction.ddp in some form);
#   2. an ambient FitParallelCtx is INERT — training under a fake NON-MAIN
#      2-rank context (rank 1 of 2, with NO process group behind it) is
#      bitwise the no-context fit. Under any real rowshard path rank 1
#      holds different rows (rows_for_rank strides rank::world) and its
#      first collective crashes on the missing process group, so a trainer
#      that starts consuming the context cannot slip through.
#
# If one of these fails, rowshard came back: RESTORE the real 2-rank gloo
# equivalence tests from accel-fit @ 1ae8662 (this file there:
# test_rowshard_lockstep_matches_single_process, the psr_mse_all case of
# test_rowshard_solo_trainer_matches_single_process, and
# test_bidpo_sampled_rng_stream_survives_rowless_ranks — the shared-rng
# d-draw on rowless ranks), adapting the hunks, rather than weakening
# these pins.

_DDP_RUNTIME_TOKENS = ("import ddp", "from .ddp", "extraction.ddp",
                       "rows_for_rank", "slice_batch", "allreduce",
                       "shard_items", "gather_sharded")


def _assert_no_rowshard_references(module) -> None:
    import inspect

    src = inspect.getsource(module)
    hits = [t for t in _DDP_RUNTIME_TOKENS if t in src]
    assert not hits, (
        f"{module.__name__} now references the fit-parallel runtime "
        f"({hits}): rowshard has been (re-)ported into a trainer this "
        "trunk runs single-process only. Restore its real 2-rank "
        "equivalence test (accel-fit @ 1ae8662, see the section comment "
        "above) instead of relying on this absence pin.")


def _fit_with_and_without_ctx(fit_fn):
    """``fit_fn()`` once with no ambient context, once under the fake
    non-main 2-rank context. Returns (with_ctx, without_ctx)."""
    ref = fit_fn()
    ddp.set_active(ddp.FitParallelCtx(rank=1, world_size=2,
                                      device=torch.device("cpu")))
    try:
        got = fit_fn()
    finally:
        ddp.set_active(None)
    return got, ref


def _pref_examples(n=5):
    """Synthetic DPO preference rows (invented text). FIVE at batch_size 2
    keeps the ragged final batch the dropped rng-stream test relied on, so
    a restored 2-rank test can reuse this data unchanged."""
    from sisteer.extraction.bidpo import PrefExample

    rows = [
        ("Say something nice.", "Bonjour le monde", "Hello world"),
        ("Describe a tree.", "C'est un grand arbre", "It is a tall tree"),
        ("Greet the reader.", "Salut a tous", "Hi everyone"),
        ("Name a color.", "Le bleu est joli", "Blue is nice"),
        ("Count to two.", "Un et deux", "One and two"),
    ]
    return [PrefExample(prompt=p, chosen=c, rejected=r)
            for p, c, r in rows[:n]]


BIDPO_KW = dict(epochs=2, batch_size=2, max_len=96, seed=0,
                direction="sampled")


def test_bidpo_has_no_rowshard_path():
    """train_bidpo ignores an ambient fit-parallel context — the DPO
    trainer runs single-process on this trunk by design (its docstring
    says so too). direction='sampled' makes this the sharpest of the
    three: the per-step d draw rides the shared cpu rng, exactly the
    stream a re-ported rowshard path must keep advancing on rowless
    ranks."""
    from sisteer.extraction import bidpo
    from sisteer.extraction.bidpo import train_bidpo
    from sisteer.models import load_model_and_tokenizer

    _assert_no_rowshard_references(bidpo)

    def fit():
        model, tok = load_model_and_tokenizer(TINY, device="cpu",
                                              dtype=torch.float32)
        return train_bidpo(model, tok, _pref_examples(), layer=1,
                           **BIDPO_KW)

    got, ref = _fit_with_and_without_ctx(fit)
    assert torch.equal(got.vectors[1], ref.vectors[1]), (
        "an ambient FitParallelCtx changed train_bidpo's result — the "
        "trainer has started consuming the fit-parallel context; restore "
        "the real 2-rank rng-stream test (see the section comment)")


def test_lockstep_has_no_rowshard_path():
    """train_lockstep ignores an ambient fit-parallel context. Compares
    params bitwise AND the held-out early-stop diagnostics — the same two
    things the dropped 2-rank equivalence test compared against
    single-process, so a restored rowshard path re-inherits the full
    contract."""
    from sisteer.extraction import lockstep as lockstep_mod
    from sisteer.models import load_model_and_tokenizer

    _assert_no_rowshard_references(lockstep_mod)

    def fit():
        model, tok = load_model_and_tokenizer(TINY, device="cpu",
                                              dtype=torch.float32)
        pairs = _pairs(5)  # ragged final batch, as the dropped test used
        fitp, heldp = pairs[:4], pairs[4:]
        examples = pairs_to_distill_examples(tok, fitp, INSTRUCTION)
        heldout = pairs_to_distill_examples(tok, heldp, INSTRUCTION)
        obj = build_lockstep_object("lora_r1", model, tok, INSTRUCTION,
                                    fitp, layer=1)
        diag: dict = {}
        train_lockstep(model, tok,
                       [LockstepArm("lora_r1", obj,
                                    default_lockstep_params("lora_r1"))],
                       examples, heldout=heldout,
                       early_stop=EarlyStopSpec(patience=1, min_epochs=1),
                       diagnostics=diag, **TRAIN_KW)
        return obj, diag["lora_r1"]

    (got_obj, got_diag), (ref_obj, ref_diag) = _fit_with_and_without_ctx(fit)
    for k in ("epochs_run", "stopped_early", "best_epoch", "heldout_curve"):
        assert got_diag[k] == ref_diag[k], (
            f"an ambient FitParallelCtx changed lockstep diagnostics[{k!r}]"
            " — the trainer has started consuming the fit-parallel context")
    for p_got, p_ref in zip(got_obj.trainable, ref_obj.trainable):
        assert torch.equal(p_got.detach(), p_ref.detach()), (
            "an ambient FitParallelCtx changed train_lockstep's params — "
            "the trainer has started consuming the fit-parallel context; "
            "restore the real 2-rank test (see the section comment)")


def test_psr_has_no_rowshard_path():
    """The psr trainer ignores an ambient fit-parallel context (via the
    registry entry the dropped test used, psr_mse_all — all-layer, so the
    zero-init path: no capture-based CAA init, which DOES legitimately
    read the context and is covered by
    test_sharded_capture_matches_unsharded)."""
    from sisteer.extraction import psr as psr_mod
    from sisteer.methods import ARMS
    from sisteer.models import load_model_and_tokenizer

    _assert_no_rowshard_references(psr_mod)

    def fit():
        model, tok = load_model_and_tokenizer(TINY, device="cpu",
                                              dtype=torch.float32)
        return ARMS["psr_mse_all"].train(model, tok, INSTRUCTION, _pairs(5),
                                         layer=1, log_fn=None, **TRAIN_KW)

    got, ref = _fit_with_and_without_ctx(fit)
    assert len(got.trainable) == len(ref.trainable)
    for p_got, p_ref in zip(got.trainable, ref.trainable):
        assert torch.equal(p_got.detach(), p_ref.detach()), (
            "an ambient FitParallelCtx changed the psr fit — the trainer "
            "has started consuming the fit-parallel context; restore the "
            "real 2-rank test (see the section comment)")


# --------------------------------------------------------------------------- #
# Driver: widths, placement, pinning, kwargs, sidecars                          #
# --------------------------------------------------------------------------- #


# (accel-fit's test_arm_width_policy left with the merge into twosweep:
# it asserted the ORIGINAL row-sharded widths — batch-size caps, the
# capture table — for the same arm_width this trunk pins to 1-per-GPU.
# The port's policy, including the loud refusal of arm_gpus > 1, is
# already pinned by test_twosi_fit_wave_plan.py::
# test_arm_width_pins_one_gpu_and_refuses_wider.)


def test_plan_parallel_deterministic_and_fits_pool():
    cfg = {"data_parallel": {"gpus": 8},
           "train_kwargs": {"default": {"batch_size": 4}},
           "batch_size_by_arm": {"loreft": 8, "bidpo": 2,
                                 "alphakl_null": 2, "alphakl_soft": 2,
                                 "alphakl_fisher": 2}}
    arms = ["psr_kl_all", "psr_mse_all", "lora_r1", "loreft", "lofit",
            "token_bos", "alphakl_null", "alphakl_soft", "alphakl_fisher",
            "diffmean", "bidpo"]
    plan1 = tt.plan_parallel(cfg, "s", arms)
    plan2 = tt.plan_parallel(cfg, "s", list(arms))
    assert plan1 == plan2, "placement must be deterministic"
    assert {t["arm"] for t in plan1["tasks"]} == set(arms)
    for t in plan1["tasks"]:
        assert 1 <= t["width"] <= 8
    # No instant of the simulated schedule exceeds the pool.
    events = sorted({t["start"] for t in plan1["tasks"]})
    for e in events:
        in_flight = sum(t["width"] for t in plan1["tasks"]
                        if t["start"] <= e < t["start"] + t["est_wall"])
        assert in_flight <= plan1["gpus"]


def test_scheduler_refuses_mismatched_pool(tmp_path, monkeypatch):
    """The world-size pin: data_parallel.gpus is asserted against the
    visible pool — an SI can never silently fit at a different width."""
    monkeypatch.setenv("SISTEER_FIT_GPU_POOL", "0,1")
    cfg = {"data_parallel": {"gpus": 3}}
    args = type("A", (), {"si": "safety_refusal", "overwrite": False,
                          "config": "x", "pairs": "y", "out": "",
                          "arms": ""})()
    with pytest.raises(SystemExit) as ei:
        tt.run_parallel_fit(args, cfg, ["diffmean"], tmp_path)
    assert "PINNED" in str(ei.value)


def test_train_refuses_distributed_env(monkeypatch):
    """`train` under a launcher whose WORLD_SIZE differs from the fit width
    (config fit_width / --expected-width; absent = 1) refuses by name — the
    width gate (_enter_fit_process_group). A matching torchrun launch of a CE
    arm is the data-parallel path, pinned in test_ddp_group_sharded_ce.py."""
    monkeypatch.setenv("WORLD_SIZE", "4")
    args = type("A", (), {"arms": "lora_r4_ce", "si": "s", "pairs": "p",
                          "out": "", "overwrite": False})()
    with pytest.raises(SystemExit) as ei:
        tt.cmd_train(args, {})
    assert "WORLD_SIZE=4" in str(ei.value) and "fit width is 1" in str(ei.value)


def test_arm_kwargs_and_parallel_sidecar(tiny_llama, tmp_path):
    """Per-arm train_kwargs still reach the trainer (the alphakl_soft beta
    regression: its silent loss once cost real object cells), and the
    sidecar records the width the arm REALLY captured at. (accel-fit's
    original also asserted a `parallel` rowshard block + effective_batch;
    this trunk packs arms 1-per-GPU — extraction.ddp deliberately not
    ported into the driver — so the sidecar's width record is
    `batch_size`, and lockstep arms' effective_batch lives on their own
    meta path.)"""
    model, tok = tiny_llama
    seen: dict = {}

    from sisteer.methods import ARMS

    real = ARMS["diffmean"]

    def spy_train(model_, tok_, instruction, pairs, **kw):
        seen.update(kw)
        return real.train(model_, tok_, instruction, pairs,
                          layer=kw.get("layer", 1), log_fn=None,
                          batch_size=2)

    class SpySpec:
        train = staticmethod(spy_train)

    cfg = {"layer": 1, "completion_tokens": 16, "max_len": 96,
           "train_kwargs": {"default": {"epochs": 1, "batch_size": 2,
                                        "max_len": 96, "holdout_frac": 0.0,
                                        "floor_ratio_max": None},
                            "diffmean": {"beta": 1.0}}}
    try:
        ARMS["diffmean"] = SpySpec
        outcome = tt.train_si_objects(
            model, tok, "french", _pairs(4), ["diffmean"], cfg=cfg,
            save_dir=tmp_path)
    finally:
        ARMS["diffmean"] = real
    assert outcome.ok, outcome.arms
    assert seen.get("beta") == 1.0, \
        "per-arm train_kwargs no longer reach the trainer"
    sidecars = list(tmp_path.glob("**/train_budget_meta.json"))
    assert sidecars, "no sidecar written"
    meta = json.loads(sidecars[0].read_text())
    assert meta["batch_size"] == 2, \
        "sidecar must record the width the arm really captured at"


# (accel-fit's test_end_to_end_parallel_scheduler left with the merge into
# twosweep: it drove one arm as a 2-rank gloo row-sharded arm task —
# arm_gpus.diffmean=2 — which this trunk's driver REFUSES loudly by
# design (extraction.ddp row-sharding deliberately not ported; it
# measured counterproductive). The refusal is pinned by
# test_twosi_fit_wave_plan.py::test_arm_width_pins_one_gpu_and_refuses_wider,
# and the width-1 scheduler end-to-end — launch, object at its cell path,
# fragment merge, exit code — by test_train_two_si_objects.py's
# run_parallel_fit tests.)


# --------------------------------------------------------------------------- #
# Capture pass: asking for one logit position must not change what it returns   #
# --------------------------------------------------------------------------- #


def test_capture_logits_to_keep_is_value_identical(tiny_llama):
    """collect_pooled_activations discards the model output and reads only
    hooked hidden states, so restricting the forward to one logit position
    (capture._last_logit_kwargs — what lifts the capture batch ceiling by
    ~an order of magnitude at Llama-3's 128k vocab) must return BITWISE the
    same pooled activations.

    Pinned because the kwarg lives in shared code every capture arm rides
    (diffmean, alphakl stages 1-2, antidote, alphasteer, caa_static_r): if
    it ever stopped being value-identical, those fits would change silently
    with no shape error."""
    from sisteer.extraction import capture as cap

    model, tok = tiny_llama
    texts = ["Say something nice.", "Describe a very tall tree in detail.",
             "Hi.", "Name a colour you find pleasant to look at."]
    layers = [0, 1]

    got = cap.collect_pooled_activations(model, tok, texts, layers,
                                         position="last", batch_size=2)
    real = cap._last_logit_kwargs
    try:  # force the plain full-logits forward
        cap._last_logit_kwargs = lambda _m: {}
        ref = cap.collect_pooled_activations(model, tok, texts, layers,
                                             position="last", batch_size=2)
    finally:
        cap._last_logit_kwargs = real

    assert cap._last_logit_kwargs(model), \
        "this transformers exposes neither logits_to_keep nor its old name"
    for l in layers:
        assert torch.equal(got.acts[l], ref.acts[l]), (
            f"layer {l}: restricting the capture forward to one logit "
            "position changed the pooled activations")


def test_class_b_batch_size_per_arm_override():
    """Class B's three arms capture very differently (caa_static_r harvests
    TEN layers, antidote/alphasteer one), so each takes its own width via
    class_b.batch_size_by_arm, falling back to the block's shared
    batch_size. Pinned so raising one arm to its ceiling cannot silently
    move the other two's fits — they are closed-form and batch-invariant in
    VALUE, but not bitwise across widths."""
    from fit_class_b import class_b_batch_size

    cb = {"batch_size": 8, "batch_size_by_arm": {"caa_static_r": 64}}
    assert class_b_batch_size(cb, "caa_static_r") == 64
    assert class_b_batch_size(cb, "antidote") == 8
    assert class_b_batch_size(cb, "alphasteer") == 8
    # No override block at all -> everyone keeps the shared width.
    assert class_b_batch_size({"batch_size": 8}, "caa_static_r") == 8


def test_arm_task_inherits_positions_and_bidpo_pool():
    """The parallel driver's per-arm entrypoint is a SECOND path into
    train_si_objects, and it re-reads the config FILE — so anything
    cmd_train resolves in memory (the fit's mask leg) or loads from disk
    (bidpo's two-sided pool) must be forwarded to it explicitly.

    Without this, a decode-only (Class C) wave scheduled in parallel would
    fit ALL-TOKEN while the wave labelled it decode-only — a silently wrong
    artifact with no error — and the DPO arms would refuse before training.
    A guard covering one of two paths is worse than none."""
    import inspect

    src = inspect.getsource(tt.cmd_arm_task)
    assert "_cfg_with_positions" in src, \
        "arm-task does not apply the wave's fit positions"
    assert "_load_bidpo_pool" in src and "bidpo_harmful_pairs=" in src, \
        "arm-task does not load/forward bidpo's own pool"
    spawn = inspect.getsource(tt._spawn_arm_task)
    for flag in ("--positions", "--bidpo-harmful-pairs",
                 "--bidpo-benign-pairs"):
        assert flag in spawn, f"scheduler does not forward {flag}"
