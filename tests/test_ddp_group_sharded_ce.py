"""Data-parallel (group-sharded DDP) document-CE fits.

CPU-only: tiny_llama in fp32 + 2-rank gloo process groups over INVENTED documents. What is pinned:

- `train_object_ce` under a 2-rank context == the single-process fit: the per-step loss trajectory (fp tolerance), the
  held-out curve, the final adapter parameters, and the ROW UNION per optimizer step (each rank forwards a disjoint strided
  share of every accumulation group's micro-batches; together they are exactly the single-process group) — the effective
  batch and the step count unchanged;
- checkpoint RESUME under DDP: a 2-rank run checkpointing every step, then a 2-rank run over the longer cap resuming it,
  lands on the uninterrupted single-process fit;
- the divisibility refusal (`accum_steps % fit_width != 0`) BY NAME, before any collective;
- the entrypoint gates (`cmd_train` / `cmd_arm_task`): a wide config outside torchrun, a launcher whose WORLD_SIZE differs
  from the width, a non-CE arm under a width > 1 — each refused by name;
- the width on the record: `fit_selection.json`'s `fit_width` stamp (and the registry hook carrying it on `extra`), the
  `train_budget_meta.json` sidecar's `fit_width` / `fit_parallel_mode`, the trainer's `diagnostics["fit_width"]`.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for _sub in ("scripts", "src"):
    _p = str(REPO / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_two_si_objects as tt  # noqa: E402
from sisteer.extraction import ddp  # noqa: E402
from sisteer.extraction.distill import DistillExample, EarlyStopSpec  # noqa: E402

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TAG = "ce-test"
_WORLD = 2

# invented documents of varying length (the trainer length-buckets them; batch_size 1 -> 12 micro-batches)
_DOCS = [
    "The river bends twice before the mill.",
    "A quiet lamp on the desk.",
    "Three sparrows argued over one crumb and then flew off together.",
    "Bread rises slowly in a cold kitchen.",
    "The map showed a road that no longer exists, so we walked the field instead.",
    "Rain on tin.",
    "Every ledger in the archive was bound in green cloth.",
    "The clockmaker kept his own time.",
    "Two chairs, one table, and a window that faced the orchard in autumn.",
    "Salt for the road.",
    "She copied the recipe onto the back of an envelope.",
    "The ferry left at dawn and again at dusk, never at noon.",
]
_HELDOUT = [
    "The gate was painted the colour of moss.",
    "A letter arrived with no stamp.",
    "Nine lanterns lined the path to the boathouse.",
]


def _examples(texts):
    return [DistillExample(prompt_with_si="", prompt_without="", completion=t) for t in texts]


def _model():
    from sisteer.models import load_model_and_tokenizer

    return load_model_and_tokenizer(TINY, device="cpu", dtype=torch.float32)


def _object(model):
    from sisteer.extraction.object_factory import build_object

    return build_object(model, object_kind="lora", lora_rank=2, lora_modules=["o_proj"],
                        init_scheme="random_lecun_1e-3", seed=0)


def _kw(epochs: int, accum_steps: int = 4, **over):
    kw = dict(lr=1e-2, weight_decay=0.0, epochs=epochs, batch_size=1, max_len=48, seed=0,
              accum_steps=accum_steps, lr_schedule="constant", max_grad_norm=1.0,
              heldout=_examples(_HELDOUT), early_stop=EarlyStopSpec(patience=10, min_epochs=1),
              log_tag=TAG)
    kw.update(over)
    return kw


def _instrument(model, rows: list, counter: list):
    """Record (optimizer-step index, sha of input_ids) for every TRAINING forward (grad enabled; the held-out and
    startup passes run under no_grad and are excluded)."""
    orig = model.forward

    def fwd(*a, **k):
        if torch.is_grad_enabled():
            ids = k["input_ids"] if "input_ids" in k else a[0]
            rows.append((counter[0], hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest()))
        return orig(*a, **k)

    model.forward = fwd


def _fit(model, tok, **kw):
    """One fit; returns (params, history, heldout_curve, rows-by-step, losses-by-step, diagnostics)."""
    from sisteer.extraction.objects import train_object_ce

    obj = _object(model)
    rows: list = []
    counter = [0]
    losses: list = []
    _instrument(model, rows, counter)

    def log_fn(rec, step=None, **_):
        if f"{TAG}/ce_loss" in rec:
            losses.append(float(rec[f"{TAG}/ce_loss"]))
            counter[0] += 1

    diag: dict = {}
    history = train_object_ce(model, tok, obj, _examples(_DOCS), log_fn=log_fn, diagnostics=diag, **kw)
    params = [p.detach().cpu().clone() for p in obj.trainable]
    return params, list(history), list(diag.get("heldout_curve") or []), rows, losses, diag


# ----------------------------------------------------------------------------- the 2-rank workers (spawned)
def _init_worker(rank: int, tmpdir: str, tag: str):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{tmpdir}/pg_init_{tag}", rank=rank, world_size=_WORLD)
    ctx = ddp.FitParallelCtx(rank=rank, world_size=_WORLD, device=torch.device("cpu"))
    ddp.set_active(ctx)
    return ctx


def _parity_worker(rank: int, tmpdir: str):
    _init_worker(rank, tmpdir, "parity")
    model, tok = _model()
    params, history, curve, rows, losses, diag = _fit(model, tok, **_kw(epochs=2))
    torch.save({"params": params, "history": history, "curve": curve, "rows": rows, "losses": losses,
                "fit_width": diag.get("fit_width")}, f"{tmpdir}/parity.rank{rank}.pt")


def _ckpt_worker(rank: int, tmpdir: str, phase: str):
    _init_worker(rank, tmpdir, f"ckpt_{phase}")
    model, tok = _model()
    ck = dict(checkpoint_every=1, checkpoint_dir=f"{tmpdir}/ckpt", checkpoint_key="ddp-resume-test")
    epochs = 1 if phase == "A" else 2
    params, history, curve, rows, losses, diag = _fit(model, tok, **_kw(epochs=epochs, **ck))
    torch.save({"params": params, "history": history, "curve": curve, "resumed_from_step": diag.get("resumed_from_step"),
                "epochs_run": diag.get("epochs_run"), "fit_width": diag.get("fit_width")},
               f"{tmpdir}/ckpt_{phase}.rank{rank}.pt")


def _spawn(fn, tmpdir: str, *extra):
    import torch.multiprocessing as mp

    mp.start_processes(fn, args=(tmpdir, *extra), nprocs=_WORLD, join=True, start_method="spawn")


def _by_step(rows):
    out: dict = {}
    for step, h in rows:
        out.setdefault(step, []).append(h)
    return out


# ----------------------------------------------------------------------------- parity
def test_two_rank_group_sharded_fit_equals_single_process(tmp_path):
    _spawn(_parity_worker, str(tmp_path))
    ranks = [torch.load(tmp_path / f"parity.rank{r}.pt", weights_only=False) for r in range(_WORLD)]
    model, tok = _model()
    ref_params, ref_history, ref_curve, ref_rows, ref_losses, ref_diag = _fit(model, tok, **_kw(epochs=2))

    assert ref_diag["fit_width"] == 1 and all(r["fit_width"] == _WORLD for r in ranks)
    # same number of optimizer steps, same loss trajectory (fp reduction order only), same held-out curve
    assert len(ref_history) == 6 and all(len(r["history"]) == 6 for r in ranks)
    for r in ranks:
        assert r["history"] == pytest.approx(ref_history, rel=1e-5, abs=1e-6), "loss trajectory drifted"
        assert r["losses"] == pytest.approx(ref_losses, rel=1e-5, abs=1e-6)
        assert r["curve"] == pytest.approx(ref_curve, rel=1e-6, abs=1e-7), "held-out curve drifted"
        assert len(r["params"]) == len(ref_params)
        for p_r, p_ref in zip(r["params"], ref_params):
            assert torch.allclose(p_r, p_ref, atol=1e-6, rtol=1e-5), "adapter params drifted from single-process"
    # replicas bit-equal across the two ranks (the trainer's own assert ran too)
    for p0, p1 in zip(ranks[0]["params"], ranks[1]["params"]):
        assert torch.equal(p0, p1)
    # the row union: per optimizer step, the two ranks' forwards are DISJOINT and together exactly the single-process step's
    ref_steps = _by_step(ref_rows)
    r0, r1 = _by_step(ranks[0]["rows"]), _by_step(ranks[1]["rows"])
    assert set(ref_steps) == set(r0) | set(r1) == set(range(6))
    for s in ref_steps:
        a, b = r0.get(s, []), r1.get(s, [])
        assert len(a) == len(b) == 2, (s, a, b)          # accum 4 / world 2 = 2 micro-batches per rank per step
        assert not set(a) & set(b), f"step {s}: a micro-batch reached both ranks"
        assert sorted(a + b) == sorted(ref_steps[s]), f"step {s}: the row union differs from the single-process step"


def test_checkpoint_resume_under_ddp_lands_on_the_uninterrupted_fit(tmp_path):
    _spawn(_ckpt_worker, str(tmp_path), "A")     # epochs=1, a checkpoint every step (rank 0 writes)
    a0 = torch.load(tmp_path / "ckpt_A.rank0.pt", weights_only=False)
    assert a0["resumed_from_step"] is None and a0["epochs_run"] == 1 and len(a0["history"]) == 3
    assert (tmp_path / "ckpt" / "ckpt.pt").exists() or any((tmp_path / "ckpt").glob("*.pt"))
    _spawn(_ckpt_worker, str(tmp_path), "B")     # epochs=2: resumes epoch 0's last checkpoint, trains epoch 1
    b = [torch.load(tmp_path / f"ckpt_B.rank{r}.pt", weights_only=False) for r in range(_WORLD)]
    assert all(x["resumed_from_step"] == 3 for x in b), [x["resumed_from_step"] for x in b]
    assert all(x["epochs_run"] == 2 and len(x["history"]) == 6 for x in b)
    model, tok = _model()
    ref_params, ref_history, ref_curve, *_ = _fit(model, tok, **_kw(epochs=2))
    for x in b:
        assert x["history"] == pytest.approx(ref_history, rel=1e-5, abs=1e-6)
        assert x["curve"] == pytest.approx(ref_curve, rel=1e-6, abs=1e-7)
        for p_x, p_ref in zip(x["params"], ref_params):
            assert torch.allclose(p_x, p_ref, atol=1e-6, rtol=1e-5)


# ----------------------------------------------------------------------------- refusals
def test_width_must_divide_accum_steps_refused_by_name_before_any_collective():
    from sisteer.extraction.objects import train_object_ce

    model, tok = _model()
    obj = _object(model)
    ddp.set_active(ddp.FitParallelCtx(rank=0, world_size=2, device=torch.device("cpu")))   # no process group behind it
    try:
        with pytest.raises(ValueError, match=r"fit_width=2 does not divide accum_steps=3"):
            train_object_ce(model, tok, obj, _examples(_DOCS), **_kw(epochs=1, accum_steps=3))
    finally:
        ddp.set_active(None)


def _args(**over):
    base = {"arms": "lora_r4_ce", "si": "s", "pairs": "p", "out": "", "overwrite": False,
            "positions": "", "expected_width": "", "config": "", "expect_config_sha": ""}
    base.update(over)
    return type("A", (), base)()


def test_train_refuses_a_launcher_whose_world_differs_from_the_width(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(SystemExit) as ei:
        tt.cmd_train(_args(), {})                       # no fit_width -> width 1 expected
    assert "WORLD_SIZE=4" in str(ei.value) and "fit width is 1" in str(ei.value)
    with pytest.raises(SystemExit) as ei:
        tt.cmd_train(_args(), {"fit_width": 2})
    assert "mis-sized" in str(ei.value)


def test_train_refuses_a_wide_config_outside_torchrun(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(SystemExit) as ei:
        tt.cmd_train(_args(), {"fit_width": 4})
    msg = str(ei.value)
    assert "torchrun --standalone --nproc_per_node=4" in msg and "--expected-width 1" in msg
    # the explicit --expected-width wins over the config (a deliberate one-rank run of a wide config passes the gate)
    assert tt._fit_width_expected(_args(expected_width="1"), {"fit_width": 4}) == 1
    assert tt._fit_width_expected(_args(), {"fit_width": 4}) == 4
    assert tt._fit_width_expected(_args(), {}) == 1


def test_wide_fits_are_wired_for_the_ce_arms_only_and_never_with_the_scheduler(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(SystemExit, match=r"document-CE arms"):
        tt._enter_fit_process_group(_args(), {"fit_width": 2}, ["diffmean"], entry="train")
    with pytest.raises(SystemExit, match=r"data_parallel.gpus=2"):
        tt._enter_fit_process_group(_args(), {"fit_width": 2, "data_parallel": {"gpus": 2}}, ["lora_r4_ce"], entry="train")
    with pytest.raises(SystemExit, match=r"does not divide accum_steps=3"):
        tt._enter_fit_process_group(_args(), {"fit_width": 2, "train_kwargs": {"lora_r4_ce": {"accum_steps": 3}}},
                                    ["lora_r4_ce"], entry="train")
    # arm-task: the scheduler's --expected-width against the launcher
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(SystemExit, match=r"nproc_per_node=2"):
        tt._enter_fit_process_group(_args(expected_width="2"), {}, ["lora_r4_ce"], entry="arm-task")
    # width 1 outside any launcher: the single-process path, no context
    assert tt._enter_fit_process_group(_args(), {}, ["lora_r4_ce"], entry="train") is None
    assert ddp.active() is None


def test_scheduler_arm_width_still_refuses_and_points_at_fit_width():
    with pytest.raises(SystemExit) as ei:
        tt.arm_width({"data_parallel": {"arm_gpus": {"lora_r4_ce": 2}}}, "s", "lora_r4_ce", 4)
    assert "fit_width" in str(ei.value) and "torchrun" in str(ei.value)


# ----------------------------------------------------------------------------- the width on the record
SEL = {"fit_search_id": "search-test-1", "fit_lr_selected_by": "search", "lora_rank": 4, "lora_modules": ["o_proj"],
       "per_expert_rank": None, "init_scheme": "zero"}
CFG = {"model": "fake/model", "model_revision": "0" * 40,
       "train_kwargs": {"default": {"positions": "all"}, "lora_r4_ce": {"lr": 1e-3, "lora_rank": 4, "fit_selection": SEL}}}


def test_fit_selection_carries_the_width_and_the_hook_rides_it_on_extra(tmp_path):
    legacy = {"kind": "lora_all", "layers": [0], "rank": 4}          # the artefact the selection is held against: o_proj / zero
    (tmp_path / "w1").mkdir(); torch.save(legacy, tmp_path / "w1" / "object.pt")
    rec1 = json.loads(tt.write_fit_selection(tmp_path / "w1", CFG, "s", "lora_r4_ce").read_text())
    assert rec1["fit_width"] == 1
    ddp.set_active(ddp.FitParallelCtx(rank=0, world_size=2, device=torch.device("cpu")))
    try:
        (tmp_path / "w2").mkdir(); torch.save(legacy, tmp_path / "w2" / "object.pt")
        rec2 = json.loads(tt.write_fit_selection(tmp_path / "w2", CFG, "s", "lora_r4_ce").read_text())
    finally:
        ddp.set_active(None)
    assert rec2["fit_width"] == 2 and {k: rec2[k] for k in SEL} == SEL
    # a config block may not claim the stamp
    clash = {"train_kwargs": {"lora_r4_ce": {"fit_selection": {**SEL, "fit_width": 8}}}}
    with pytest.raises(tt.CellAuditError, match="identity stamps"):
        tt.write_fit_selection(tmp_path / "w2", clash, "s", "lora_r4_ce")
    # the registry hook: fit_width is not a selection COLUMN, so it rides extra
    from sisteer.registry import hooks

    cell = tmp_path / "cell"
    cell.mkdir()
    blob = {"kind": "lora_all", "layers": [0], "rank": 4, "lora_modules": ["o_proj"],
            "A": {"0.o_proj": torch.zeros(8, 4)}, "B": {"0.o_proj": torch.zeros(8, 4)}, "positions": "all",
            "response_header_len": None}
    torch.save(blob, cell / "object.pt")
    (cell / "train_budget_meta.json").write_text(json.dumps({"cell_id": "s__lora_r4_ce", "si_id": "s", "arm": "lora_r4_ce",
                                                              "positions_at_fit": "all", "fit_width": 2}))
    (cell / "fit_selection.json").write_text(json.dumps(rec2))
    payload, _art = hooks.object_from_sidecar(str(cell), campaign="two_si", object_kind="lora", paper_label="constitution_lora",
                                              object_role="constitution", fit_target_kind="corpus_document_fit", fit_objective="ce")
    assert payload["extra"]["fit_width"] == 2
    assert payload["fit_search_id"] == SEL["fit_search_id"]


def test_sidecar_records_the_width_the_fit_ran_at(tiny_llama, tmp_path, monkeypatch):
    """train_budget_meta.json carries fit_width / fit_parallel_mode from the driver's ONE width reader (the diffmean spy
    keeps the fit itself collective-free; the reader is what a data-parallel rank 0 would see)."""
    from sisteer.data.pairs import Pair
    from sisteer.methods import ARMS

    model, tok = tiny_llama
    real = ARMS["diffmean"]

    class SpySpec:
        train = staticmethod(lambda m, t, ins, pairs, **kw: real.train(m, t, ins, pairs, layer=kw.get("layer", 1),
                                                                        log_fn=None, batch_size=2))

    pairs = [Pair(prompt=f"Question {i}.", pos="Oui.", neg="Yes.", instruction_id="french") for i in range(4)]
    cfg = {"layer": 1, "completion_tokens": 16, "max_len": 96,
           "train_kwargs": {"default": {"epochs": 1, "batch_size": 2, "max_len": 96, "holdout_frac": 0.0,
                                        "floor_ratio_max": None}}}
    monkeypatch.setattr(tt, "fit_width_active", lambda: 4)
    try:
        ARMS["diffmean"] = SpySpec
        outcome = tt.train_si_objects(model, tok, "french", pairs, ["diffmean"], cfg=cfg, save_dir=tmp_path)
    finally:
        ARMS["diffmean"] = real
    assert outcome.ok, outcome.arms
    meta = json.loads(next(tmp_path.glob("**/train_budget_meta.json")).read_text())
    assert meta["fit_width"] == 4 and meta["fit_parallel_mode"] == "ddp_group_sharded"
    assert meta["effective_batch"] == 2      # the width never changes the effective batch


def test_replica_rank_writes_nothing_and_records_locally(tiny_llama, tmp_path, monkeypatch):
    """A non-main rank of train_si_objects: no sidecar, no report, no temp cell dir, no store sentinel — the arm is
    recorded `trained` locally (rank 0 owns the publication)."""
    from sisteer.data.pairs import Pair
    from sisteer.methods import ARMS

    model, tok = tiny_llama
    real = ARMS["diffmean"]

    class SpySpec:
        train = staticmethod(lambda m, t, ins, pairs, **kw: real.train(m, t, ins, pairs, layer=kw.get("layer", 1),
                                                                        log_fn=None, batch_size=2))

    pairs = [Pair(prompt=f"Question {i}.", pos="Oui.", neg="Yes.", instruction_id="french") for i in range(4)]
    cfg = {"layer": 1, "completion_tokens": 16, "max_len": 96,
           "train_kwargs": {"default": {"epochs": 1, "batch_size": 2, "max_len": 96, "holdout_frac": 0.0,
                                        "floor_ratio_max": None}}}
    monkeypatch.setattr(tt, "_is_main_rank", lambda: False)
    monkeypatch.setattr(tt, "fit_width_active", lambda: 2)
    try:
        ARMS["diffmean"] = SpySpec
        outcome = tt.train_si_objects(model, tok, "french", pairs, ["diffmean"], cfg=cfg, save_dir=tmp_path)
    finally:
        ARMS["diffmean"] = real
    assert outcome.ok and outcome.arms["diffmean"].status == "trained"
    assert outcome.arms["diffmean"].detail.get("replica_rank") is True
    assert not list(tmp_path.rglob("*")), sorted(str(p) for p in tmp_path.rglob("*"))
