"""Per-arm early stopping on a held-out pair split + the no-op-floor ratio
guard.

`epochs` used to be the only budget knob: no held-out split, no stopping,
and a degenerate arm could plateau ~30x above the no-op floor for 966
silent steps and still ship an artifact. Pinned here, on CPU fixtures with
synthetic pairs only:

- the held-out carve is DETERMINISTIC (data/prompts.split_prompts' exact
  hash convention, keyed on the pair's prompt), order-independent, and the
  held-out pairs never reach any trainer — lockstep or solo;
- stopping is PER ARM: the stopper stops on `patience` non-improving
  held-out epochs and keeps the BEST epoch's weights, proven bitwise by
  re-running to exactly that epoch;
- the lockstep group's per-arm stopping matches the solo engine exactly
  (same curve, same stop, bitwise-identical params) — an arm stopping
  cannot perturb its group partners;
- the end-of-training floor guard RAISES (FloorRatioError) and the failed
  arm writes NO artifact while the rest of the run continues — extending
  distill.startup_loss_check's warn-only concept to a refusal;
- the chosen stopping epoch and held-out curve are recorded per arm in the
  fit report and the train_budget_meta.json sidecar, so every arm's
  stopping point is justified by held-out loss rather than a fixed count.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_two_si_objects as t2

import sisteer.extraction.lockstep as lockstep_mod
from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.eval.defense_cell import object_path_for
from sisteer.extraction.distill import (
    EarlyStopSpec,
    EarlyStopper,
    FloorRatioError,
    floor_ratio_check,
    train_steered_lm,
)
from sisteer.extraction.lockstep import (
    LockstepArm,
    build_lockstep_object,
    default_lockstep_params,
    train_lockstep,
)
from sisteer.models import hidden_size

INSTRUCTION = BY_ID["french"]
SI_ID = "safety_refusal"
MAX_LEN = 96

# Ten prompts whose sha256(salt + prompt) first byte is KNOWN under the
# driver's default salt ("two_si_holdout"): at holdout_frac=0.5 the first
# five land HELD-OUT (hash < 0.5) and the last five FIT.
HELDOUT_PROMPTS = ["Say something nice.", "Describe a tree.",
                   "Name a color.", "Le ciel est bleu.", "Offer some tea."]
FIT_PROMPTS = ["Greet the reader.", "Count to two.", "What causes rain?",
               "Name a planet.", "Describe the sea."]


def _pairs(prompts=None, instruction_id="french"):
    prompts = prompts if prompts is not None else HELDOUT_PROMPTS + FIT_PROMPTS
    return [Pair(prompt=p, pos=f"Reponse {i} en francais",
                 neg=f"Reply {i} in English", instruction_id=instruction_id)
            for i, p in enumerate(prompts)]


# ---------------------------------------------------------------------------
# the deterministic carve
# ---------------------------------------------------------------------------


def test_split_pairs_is_deterministic_and_order_independent():
    pairs = _pairs()
    fit, heldout = t2.split_pairs(pairs, 0.5, "two_si_holdout")
    assert sorted(p.prompt for p in heldout) == sorted(HELDOUT_PROMPTS)
    assert sorted(p.prompt for p in fit) == sorted(FIT_PROMPTS)
    # order-independent: a reversed pool carves the SAME sides
    fit2, heldout2 = t2.split_pairs(list(reversed(pairs)), 0.5,
                                    "two_si_holdout")
    assert {p.prompt for p in heldout2} == {p.prompt for p in heldout}
    # duplicates of one prompt can never straddle the split (hash on prompt)
    dup = _pairs(["Say something nice."] * 3)
    f3, h3 = t2.split_pairs(dup, 0.5, "two_si_holdout")
    assert len(f3) == 0 and len(h3) == 3
    # frac 0 disables the carve; out-of-range fracs refuse
    f4, h4 = t2.split_pairs(pairs, 0.0, "two_si_holdout")
    assert len(f4) == len(pairs) and h4 == []
    with pytest.raises(ValueError, match="holdout_frac"):
        t2.split_pairs(pairs, 1.0, "two_si_holdout")


# ---------------------------------------------------------------------------
# the stopper itself
# ---------------------------------------------------------------------------


def test_early_stopper_stops_on_patience_and_keeps_the_best_epoch():
    stopper = EarlyStopper(EarlyStopSpec(patience=2, min_epochs=1))
    p = nn.Parameter(torch.zeros(3))
    curve = [5.0, 4.0, 4.2, 4.1]  # best at epoch 2, then two bad epochs
    stops = []
    for e, loss in enumerate(curve, start=1):
        with torch.no_grad():
            p.fill_(float(e))  # epoch e's "trained" params
        stops.append(stopper.update(loss, [p]))
    assert stops == [False, False, False, True]
    assert stopper.best_epoch == 2 and stopper.best == 4.0
    stopper.restore_best([p])
    assert torch.equal(p.detach(), torch.full((3,), 2.0))


def test_early_stopper_min_epochs_defers_the_stop():
    stopper = EarlyStopper(EarlyStopSpec(patience=1, min_epochs=3))
    p = nn.Parameter(torch.zeros(1))
    assert stopper.update(1.0, [p]) is False
    assert stopper.update(2.0, [p]) is False  # patience hit, min_epochs not
    assert stopper.update(2.0, [p]) is True


def test_floor_ratio_check_raises_above_and_passes_below():
    logged = []
    ratio = floor_ratio_check(0.30, 0.148, ratio_max=3.0, tag="t",
                              log_fn=lambda d, *a, **k: logged.append(d))
    assert ratio == pytest.approx(0.30 / 0.148)
    assert logged and logged[0]["t/floor_ratio"] == ratio
    # the real calibration case: 4.82 against a 0.148 floor (~30x) raises
    with pytest.raises(FloorRatioError, match="no-op floor"):
        floor_ratio_check(4.82, 0.148, ratio_max=3.0, tag="t")
    # materiality clamp: a near-zero floor with an excellent absolute loss
    # is NOT degeneration (ratio noise around zero must not fail fits)
    assert floor_ratio_check(2e-4, 3e-5, ratio_max=3.0, tag="t") < 3.0


def test_floor_ratio_check_refuses_non_finite_probes():
    """NaN comparisons are all False, so a NaN'd fit would sail through
    "ratio > ratio_max" as if it passed — and for arms without a dead-gate
    guard (vector, lora_r1) nothing downstream stops the save. Observed
    live 2026-08-25: psr_kl_all's response-leg fit went NaN in epoch 1;
    only PSR's gate guard stood between that object and disk. The guard
    must refuse non-finite inputs on EITHER side, loudly."""
    for bad_final, floor in ((float("nan"), 0.148),
                             (float("inf"), 0.148),
                             (0.30, float("nan"))):
        with pytest.raises(FloorRatioError, match="non-finite"):
            floor_ratio_check(bad_final, floor, ratio_max=3.0, tag="t")


# ---------------------------------------------------------------------------
# engine-level stopping: restore-best is exact, lockstep matches solo
# ---------------------------------------------------------------------------


def _solo_vector(model, tok, examples, *, epochs, heldout=None, spec=None,
                 lr=5e-3):
    vec = nn.Parameter(torch.zeros(hidden_size(model)))
    diag: dict = {}
    train_steered_lm(
        model, tok, examples, 1,
        vector_provider=lambda idxs: vec, trainable_params=[vec],
        loss="soft", lr=lr, epochs=epochs, batch_size=2, max_len=MAX_LEN,
        seed=0, heldout_examples=heldout, early_stop=spec, diagnostics=diag)
    return vec.detach().clone(), diag


def test_restored_params_are_bitwise_the_best_epochs(tiny_llama):
    """Train with early stopping, read the chosen epoch b, then re-run the
    SAME schedule for exactly b epochs with no stopping: the kept params
    must be bitwise identical — the restore really is 'the best epoch',
    not an approximation of it."""
    model, tok = tiny_llama
    examples = pairs_to_distill_examples(tok, _pairs(FIT_PROMPTS),
                                         INSTRUCTION)
    heldout = pairs_to_distill_examples(tok, _pairs(HELDOUT_PROMPTS),
                                        INSTRUCTION)
    v_a, diag = _solo_vector(model, tok, examples, epochs=5, heldout=heldout,
                             spec=EarlyStopSpec(patience=1))
    b = diag["best_epoch"]
    assert 1 <= b <= diag["epochs_run"] <= 5
    assert len(diag["heldout_curve"]) == diag["epochs_run"] + 1  # + baseline
    v_b, _ = _solo_vector(model, tok, examples, epochs=b)
    assert torch.equal(v_a, v_b)


def test_flat_heldout_curve_stops_at_patience(tiny_llama):
    """lr=0 freezes the params, so the held-out loss is exactly constant:
    epoch 1 is the (only) best and patience=1 stops training at epoch 2 of
    a 6-epoch budget — the budget is a CAP, not the stopping rule."""
    model, tok = tiny_llama
    examples = pairs_to_distill_examples(tok, _pairs(FIT_PROMPTS),
                                         INSTRUCTION)
    heldout = pairs_to_distill_examples(tok, _pairs(HELDOUT_PROMPTS),
                                        INSTRUCTION)
    _, diag = _solo_vector(model, tok, examples, epochs=6, heldout=heldout,
                           spec=EarlyStopSpec(patience=1), lr=0.0)
    assert diag["stopped_early"] is True
    assert diag["epochs_run"] == 2
    assert diag["best_epoch"] == 1


def test_lockstep_per_arm_stopping_matches_solo_and_isolates_arms(
        tiny_llama):
    """The lockstep group's per-arm stopping is the SAME rule the solo
    engine runs: identical held-out curve, identical chosen epoch, bitwise
    identical restored params for the vector arm — while lora_r1 rides the
    same group with its own independent stopper and diagnostics."""
    model, tok = tiny_llama
    pairs = _pairs(FIT_PROMPTS)
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)
    heldout = pairs_to_distill_examples(tok, _pairs(HELDOUT_PROMPTS),
                                        INSTRUCTION)
    spec = EarlyStopSpec(patience=1)
    v_solo, d_solo = _solo_vector(model, tok, examples, epochs=5,
                                  heldout=heldout, spec=spec)

    diag: dict = {}
    group = [
        LockstepArm("vector",
                    build_lockstep_object("vector", model, tok, INSTRUCTION,
                                          pairs, layer=1),
                    default_lockstep_params("vector")),
        LockstepArm("lora_r1",
                    build_lockstep_object("lora_r1", model, tok, INSTRUCTION,
                                          pairs, layer=1),
                    default_lockstep_params("lora_r1")),
    ]
    train_lockstep(model, tok, group, examples, epochs=5, batch_size=2,
                   max_len=MAX_LEN, seed=0, heldout=heldout, early_stop=spec,
                   diagnostics=diag)
    dv = diag["vector"]
    assert dv["heldout_curve"] == d_solo["heldout_curve"]
    assert dv["best_epoch"] == d_solo["best_epoch"]
    assert dv["epochs_run"] == d_solo["epochs_run"]
    assert dv["stopped_early"] == d_solo["stopped_early"]
    assert torch.equal(group[0].obj.trainable[0].detach(), v_solo)
    # the partner arm carries its OWN record — per-arm stopping, not global
    dl = diag["lora_r1"]
    assert len(dl["heldout_curve"]) == dl["epochs_run"] + 1
    assert dl["best_epoch"] >= 1 and "floor_ratio" in dl


# ---------------------------------------------------------------------------
# train_psr's mse loop (psr_mse_all): the last arm to get the seam
# ---------------------------------------------------------------------------


def _psr_mse_fit(model, tok, examples, *, epochs, heldout=None, spec=None,
                 lr=5e-3):
    """One train_psr mse fit under a CONSTANT lr schedule (a cosine's
    shape depends on the epoch budget, so only the constant schedule makes
    're-run for exactly b epochs' the same trajectory prefix)."""
    from sisteer.extraction.psr import PSRObject, train_psr

    obj = PSRObject(model, [1], train_direction=True)
    diag: dict = {}
    hist = train_psr(model, tok, obj, examples, loss="mse", lr=lr,
                     epochs=epochs, batch_size=2, max_len=MAX_LEN, seed=0,
                     lr_schedule="constant", heldout=heldout,
                     early_stop=spec, diagnostics=diag)
    return hist, [p.detach().clone() for p in obj.trainable], diag


def test_psr_mse_flat_heldout_curve_stops_at_patience(tiny_llama):
    """lr=0 freezes the object, so the held-out masked MSE is exactly
    constant: epoch 1 is the (only) best and patience=1 stops training at
    epoch 2 of a 6-epoch budget — the budget is a CAP for the psr mse
    loop now too (it was the one arm still running the fixed budget to
    the cap while every other arm early-stopped)."""
    model, tok = tiny_llama
    examples = pairs_to_distill_examples(tok, _pairs(FIT_PROMPTS),
                                         INSTRUCTION)
    heldout = pairs_to_distill_examples(tok, _pairs(HELDOUT_PROMPTS),
                                        INSTRUCTION)
    hist, _params, diag = _psr_mse_fit(model, tok, examples, epochs=6,
                                       heldout=heldout,
                                       spec=EarlyStopSpec(patience=1),
                                       lr=0.0)
    assert diag["stopped_early"] is True
    assert diag["epochs_run"] == 2
    assert diag["best_epoch"] == 1
    # baseline + one point per completed epoch; history stops with epoch 2
    assert len(diag["heldout_curve"]) == diag["epochs_run"] + 1
    assert len(hist) == 2 * 3  # 3 batches/epoch, 2 of the 6 budgeted epochs
    # lr=0: the curve really is flat — the stop was the patience rule
    assert diag["heldout_curve"][1] == diag["heldout_curve"][2]


def test_psr_mse_restored_params_are_bitwise_the_best_epochs(tiny_llama):
    """Train with early stopping, read the chosen epoch b, then re-run the
    SAME constant-lr schedule for exactly b epochs with no stopping: the
    kept params must be bitwise identical — the restore really is 'the
    best epoch', not an approximation of it (the solo-vector engine's
    exact pin, on the psr mse loop)."""
    model, tok = tiny_llama
    examples = pairs_to_distill_examples(tok, _pairs(FIT_PROMPTS),
                                         INSTRUCTION)
    heldout = pairs_to_distill_examples(tok, _pairs(HELDOUT_PROMPTS),
                                        INSTRUCTION)
    _h, p_a, diag = _psr_mse_fit(model, tok, examples, epochs=5,
                                 heldout=heldout,
                                 spec=EarlyStopSpec(patience=1))
    b = diag["best_epoch"]
    assert 1 <= b <= diag["epochs_run"] <= 5
    assert len(diag["heldout_curve"]) == diag["epochs_run"] + 1  # + baseline
    _h2, p_b, _d2 = _psr_mse_fit(model, tok, examples, epochs=b)
    assert len(p_a) == len(p_b) and len(p_a) > 0
    for a, bb in zip(p_a, p_b):
        assert torch.equal(a, bb)


def test_psr_mse_refuses_early_stop_without_heldout():
    """A spec with no held-out curve to rule on would be silently inert —
    the fit would run the fixed budget while wearing an early-stop
    request. Refused before any model work, so dummies suffice."""
    from sisteer.extraction.psr import train_psr

    with pytest.raises(ValueError, match="early_stop without heldout"):
        train_psr(None, None, None, [], loss="mse",
                  early_stop=EarlyStopSpec(patience=1))


# ---------------------------------------------------------------------------
# driver level: the carve is enforced, the guard refuses, the record lands
# ---------------------------------------------------------------------------

CFG = {
    "layer": 1,
    "completion_tokens": 16,
    "max_len": 256,
    "max_len_by_si": {SI_ID: 256},
    "si_ids": [SI_ID, "si_constitution"],
}


def _cfg(**default_overrides):
    cfg = dict(CFG)
    cfg["train_kwargs"] = {"default": {"epochs": 1, "seed": 0,
                                       **default_overrides}}
    return cfg


class _DummyObj:
    kind = "dummy"
    # Not a registry class, so the driver's _FIT_SITES table (rightly) does
    # not know it — self-report through the fit_site seam instead.
    fit_site = "test_double"
    trainable: list = []

    def inject(self, rendered_prompt):
        return rendered_prompt

    def save(self, path):
        Path(path).write_bytes(b"dummy")


def test_heldout_pairs_never_reach_any_trainer(tiny_llama, tmp_path,
                                               monkeypatch):
    """The carve is enforced at the driver: both the lockstep group and the
    solo registry path train on the FIT side only, and the held-out pairs
    flow exclusively into train_lockstep's ``heldout``."""
    model, tok = tiny_llama
    pairs = _pairs(instruction_id=SI_ID)
    seen: dict = {}

    def fake_lockstep(model_, tok_, group, examples, **kw):
        seen["lockstep_prompts"] = [ex.prompt_without for ex in examples]
        seen["heldout_prompts"] = [ex.prompt_without
                                   for ex in (kw.get("heldout") or [])]
        # the engine's epoch contract: the driver's epoch_cb runs at every epoch end and the publish REQUIRES the
        # epoch index it writes (a wired hook that saved nothing is refused by name); one epoch here, as a real engine would
        if kw.get("epoch_cb") is not None:
            kw["epoch_cb"](0, 0, [{"arm": a, "history": [], "heldout_curve": [], "stopped": False} for a in group])
        return {a.name: [] for a in group}

    monkeypatch.setattr(lockstep_mod, "train_lockstep", fake_lockstep)

    def fake_solo(train_fn, model_, tok_, instruction, pairs_, **kw):
        seen["solo_prompts"] = [p.prompt for p in pairs_]
        # the real contract returns (object, surviving_kwargs)
        return _DummyObj(), dict(kw.get("candidate_kw") or {})

    monkeypatch.setattr(t2, "_safe_registry_train", fake_solo)

    outcome = t2.train_si_objects(
        model, tok, SI_ID, pairs, ["lora_r1", "loreft"],
        cfg=_cfg(holdout_frac=0.5, floor_ratio_max=None),
        save_dir=tmp_path / "defenses")
    assert outcome.ok, {a: o.error for a, o in outcome.failed.items()}
    # solo trainer saw exactly the fit prompts
    assert sorted(seen["solo_prompts"]) == sorted(FIT_PROMPTS)
    # no held-out prompt appears in any lockstep TRAINING example, and all
    # of them appear in the held-out set handed to the engine
    for h in HELDOUT_PROMPTS:
        assert not any(h in p for p in seen["lockstep_prompts"])
        assert any(h in p for p in seen["heldout_prompts"])
    for f in FIT_PROMPTS:
        assert any(f in p for p in seen["lockstep_prompts"])


def test_floor_guard_refuses_arms_and_writes_no_artifact(tiny_llama,
                                                         tmp_path):
    """An impossible ratio ceiling makes every arm 'degenerate': both the
    lockstep arm and the solo arm must record FloorRatioError, write NO
    artifact, and the run must keep going (per-arm containment)."""
    model, tok = tiny_llama
    out_dir = tmp_path / "defenses"
    outcome = t2.train_si_objects(
        model, tok, SI_ID, _pairs(instruction_id=SI_ID),
        ["lora_r1", "loreft"], cfg=_cfg(floor_ratio_max=1e-12),
        save_dir=out_dir)
    assert set(outcome.failed) == {"lora_r1", "loreft"}
    for arm in ("lora_r1", "loreft"):
        assert outcome.arms[arm].error_type == "FloorRatioError"
        assert not object_path_for(out_dir, f"{SI_ID}__{arm}").exists()
    report = json.loads(t2.report_path_for(out_dir, SI_ID).read_text())
    assert sorted(report["failed"]) == ["lora_r1", "loreft"]


def test_stopping_epoch_and_curve_land_in_sidecar_and_report(tiny_llama,
                                                             tmp_path):
    """Early stopping's paper trail: the chosen stopping epoch and the
    held-out curve are recorded per arm — in the fit report AND the
    train_budget_meta.json sidecar — so the stopping point can be reported
    without re-running anything."""
    model, tok = tiny_llama
    out_dir = tmp_path / "defenses"
    outcome = t2.train_si_objects(
        model, tok, SI_ID, _pairs(instruction_id=SI_ID), ["lora_r1"],
        cfg=_cfg(epochs=3, holdout_frac=0.5, early_stop_patience=1),
        save_dir=out_dir)
    assert outcome.ok, {a: o.error for a, o in outcome.failed.items()}
    detail = outcome.arms["lora_r1"].detail
    assert detail["best_epoch"] >= 1
    assert len(detail["heldout_curve"]) == detail["epochs_run"] + 1
    assert {"noop_floor", "final_probe_loss", "floor_ratio"} <= set(detail)

    meta = json.loads(
        (object_path_for(out_dir, f"{SI_ID}__lora_r1").parent
         / "train_budget_meta.json").read_text())
    assert meta["fit_record"]["best_epoch"] == detail["best_epoch"]
    assert meta["fit_record"]["heldout_curve"] == detail["heldout_curve"]

    report = json.loads(t2.report_path_for(out_dir, SI_ID).read_text())
    assert report["arms"]["lora_r1"]["detail"]["best_epoch"] == \
        detail["best_epoch"]


def test_psr_mse_solo_arm_early_stops_and_records(tiny_llama, tmp_path):
    """The driver wires the SAME held-out carve + config spec into the psr
    mse SOLO path (train_psr's heldout/early_stop seam — psr_mse_all was
    the one arm still running the fixed budget to the cap): lr=0 makes
    the held-out curve exactly flat, so patience=1 stops the fit at epoch
    2 of a 4-epoch budget, and the stopping record lands at the SAME
    top-level detail keys the lockstep arms publish, in the report AND
    the train_budget_meta.json sidecar."""
    model, tok = tiny_llama
    out_dir = tmp_path / "defenses"
    cfg = dict(CFG)
    cfg["batch_size_by_arm"] = {"psr_mse_all": 2}
    cfg["train_kwargs"] = {
        "default": {"epochs": 4, "seed": 0, "holdout_frac": 0.5,
                    "early_stop_patience": 1, "floor_ratio_max": None,
                    "monitors": False},
        # lr=0 freezes the object -> a deterministically flat curve.
        "psr_mse_all": {"lr": 0.0},
    }
    outcome = t2.train_si_objects(
        model, tok, SI_ID, _pairs(instruction_id=SI_ID), ["psr_mse_all"],
        cfg=cfg, save_dir=out_dir)
    assert outcome.ok, {a: o.error for a, o in outcome.failed.items()}
    detail = outcome.arms["psr_mse_all"].detail
    assert detail["early_stop"]["supported"] is True
    assert detail["early_stop"]["patience"] == 1
    assert detail["early_stop"]["heldout_n"] == len(HELDOUT_PROMPTS)
    assert detail["stopped_early"] is True
    assert detail["best_epoch"] == 1
    # epochs_run is the count the fit REALLY ran, not the 4-epoch cap
    assert detail["epochs_run"] == 2
    assert len(detail["heldout_curve"]) == detail["epochs_run"] + 1

    meta = json.loads(
        (object_path_for(out_dir, f"{SI_ID}__psr_mse_all").parent
         / "train_budget_meta.json").read_text())
    assert meta["fit_record"]["best_epoch"] == detail["best_epoch"]
    assert meta["fit_record"]["heldout_curve"] == detail["heldout_curve"]
    assert meta["fit_record"]["epochs_run"] == 2

    report = json.loads(t2.report_path_for(out_dir, SI_ID).read_text())
    assert report["arms"]["psr_mse_all"]["detail"]["stopped_early"] is True


def test_thin_pool_disables_stopping_loudly_not_silently(tiny_llama,
                                                         tmp_path, capsys):
    """A pool too small to carve trains at the fixed budget — with a LOUD
    stderr note, never a silent behavior change."""
    model, tok = tiny_llama
    # 'Greet the reader.' hashes to 0.839 — a 0.1 frac carves nothing
    pairs = [Pair(prompt="Greet the reader.", pos="Salut", neg="Hi",
                  instruction_id=SI_ID)] * 3
    outcome = t2.train_si_objects(
        model, tok, SI_ID, pairs, ["lora_r1"],
        cfg=_cfg(holdout_frac=0.1, floor_ratio_max=None),
        save_dir=tmp_path / "defenses")
    assert outcome.ok
    err = capsys.readouterr().err
    assert "early stopping is DISABLED" in err
