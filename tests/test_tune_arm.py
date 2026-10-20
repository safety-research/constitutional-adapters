"""T1: the generalized HPO harness.

Covers: the per-warmup-level Sobol designs (independent, digitally
shifted, stratum-covering — and the parity-aliasing failure mode the
assert exists to kill); the 9-term quadratic recovering a planted optimum
including the warmup x clip interaction; the analyzer excluding
floor-failing and non-finite rows; edge -> recenter; fitted-argmin (never
best-raw) selection; the design rank check; the bidpo winner carrying the
realized accum_steps (consistency enforced); the adapter registry covering
all five arms including the solo paths (trainer spies — CPU synthetic);
bidpo's production pool routing (pairs_for_arm, reversed benign polarity)
and full-optimizer-group trim; the psr_mse trainer probe seam; the
canonical model-identity check (payload vs --config) BEFORE skip or load;
and fingerprint-gated trial idempotence (torn/stale/non-terminal
summaries re-run).

CPU-only: tiny_llama fixture + synthetic French pairs, never real store
data.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

TUNE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "tune_arm.py"
_spec = importlib.util.spec_from_file_location("_tune_arm", TUNE_PATH)
tune_arm = importlib.util.module_from_spec(_spec)
sys.modules["_tune_arm"] = tune_arm
_spec.loader.exec_module(tune_arm)

BOX = {"lr": (3e-5, 3e-3), "max_grad_norm": (0.03, 3.0)}
LEVELS = (0.05, 0.15)


def _units(trials, level, box):
    """Recover each trial's unit-cube coordinates from its mapped values,
    snapped back onto the Sobol integer lattice (exact by construction —
    the snap only removes exp/log float noise)."""
    out = []
    for t in trials:
        if t["warmup_frac"] != level:
            continue
        u = []
        for k in ("lr", "max_grad_norm"):
            lo, hi = box[k]
            est = ((math.log(t[k]) - math.log(lo))
                   / (math.log(hi) - math.log(lo)))
            u.append(round(est * (1 << 30)) / (1 << 30))
        out.append(tuple(u))
    return out


# --------------------------------------------------------------------------
# Sobol design
# --------------------------------------------------------------------------

class TestSobolDesign:
    def test_per_level_designs_cover_every_stratum(self):
        payload = tune_arm.build_trials(arm="token_faithful", model="m",
                                        position="all", round_no=1, n=16,
                                        box=BOX)
        counts = collections.Counter(t["warmup_frac"]
                                     for t in payload["trials"])
        assert counts == {0.05: 8, 0.15: 8}
        for lv in LEVELS:
            pts = _units(payload["trials"], lv, BOX)
            assert len(pts) == 8
            # Per-axis stratification: each of the 8 width-1/8 bins holds
            # exactly one point, on BOTH continuous axes...
            for axis in (0, 1):
                bins = {int(p[axis] * 8) for p in pts}
                assert bins == set(range(8)), (lv, axis, sorted(pts))
            # ...and every mixed dyadic grid too (the full (0,3,2)-net
            # property; kills correlated-axes designs that pass the
            # marginal check).
            tune_arm.assert_net_coverage(pts, 3, tag=f"w{lv}")

    def test_levels_are_independent_digitally_shifted_designs(self):
        payload = tune_arm.build_trials(arm="token_faithful", model="m",
                                        position="all", round_no=1, n=16,
                                        box=BOX)
        lo, hi = (_units(payload["trials"], lv, BOX) for lv in LEVELS)
        assert set(lo) != set(hi), "levels re-used one design"
        shifts = payload["design_shift"]
        assert set(shifts) == {"0.05", "0.15"}
        assert shifts["0.05"] != shifts["0.15"]
        # Nonzero shift: neither level is the bare base sequence (whose
        # point 0 is the origin — a corner trial at exactly the box min).
        base = {(x / (1 << 30), y / (1 << 30))
                for x, y in tune_arm.sobol_ints(3)}
        assert set(lo) != base and set(hi) != base
        # And rounds draw fresh shifts.
        r2 = tune_arm.build_trials(arm="token_faithful", model="m",
                                   position="all", round_no=2, n=16, box=BOX)
        assert r2["design_shift"] != shifts

    def test_parity_split_is_caught_by_the_coverage_assert(self):
        # The review-P1-2 bug class: split ONE 16-point 2-D sequence by
        # index parity. The even half's 2nd coordinate lands in
        # [0, 0.4375] — warmup aliased with clip. The emit-time assert
        # must refuse exactly this.
        pts16 = [(x / (1 << 30), y / (1 << 30))
                 for x, y in tune_arm.sobol_ints(4)]
        even = pts16[0::2]
        assert max(p[1] for p in even) < 0.5  # the aliasing, demonstrated
        with pytest.raises(SystemExit, match="stratum coverage FAILED"):
            tune_arm.assert_net_coverage(even, 3, tag="parity-even")

    def test_trials_interleave_levels_for_pod_slicing(self):
        payload = tune_arm.build_trials(arm="bidpo", model="m",
                                        position="all", round_no=1, n=16,
                                        box=BOX)
        # Any contiguous START/COUNT slice a pod takes spans both levels.
        warms = [t["warmup_frac"] for t in payload["trials"]]
        assert warms[0::2] == [0.05] * 8 and warms[1::2] == [0.15] * 8

    def test_refine_round_keeps_both_levels_at_m2(self):
        payload = tune_arm.build_trials(arm="token_faithful", model="m",
                                        position="all", round_no=2, n=8,
                                        box=BOX)
        counts = collections.Counter(t["warmup_frac"]
                                     for t in payload["trials"])
        assert counts == {0.05: 4, 0.15: 4}
        for lv in LEVELS:
            tune_arm.assert_net_coverage(_units(payload["trials"], lv, BOX),
                                         2, tag=f"refine-w{lv}")

    def test_token_faithful_response_cell_refused_at_emit(self):
        ns = argparse.Namespace(arm="token_faithful", position="response",
                                model="m", out="/dev/null", n=None, round=1,
                                box_from=None, lr_min=None, lr_max=None,
                                clip_min=None, clip_max=None,
                                warmup_levels="0.05,0.15")
        with pytest.raises(SystemExit, match="permanent hole"):
            tune_arm.emit_trials(ns)


# --------------------------------------------------------------------------
# The 9-term quadratic + analyzer verdicts
# --------------------------------------------------------------------------

# A planted objective the 9-term model represents EXACTLY: per-level
# parabolas whose CLIP optimum moves with the warmup level (a genuine
# warmup x clip interaction) and whose high-warmup level is globally
# better (warmup main effect). W is the level indicator, so W^2 == W and
# the expansion stays inside the model's 9 terms.
_L0 = math.log(3e-4)
_C0_LO = math.log(0.1)
_C0_HI = math.log(1.0)


def _planted_log_y(lr: float, clip: float, warmup: float) -> float:
    w = 0.0 if warmup == 0.05 else 1.0
    c0 = _C0_LO + w * (_C0_HI - _C0_LO)
    return (2.0 * (math.log(lr) - _L0) ** 2
            + 1.5 * (math.log(clip) - c0) ** 2
            - 0.5 * w)


def _write_summaries(out: Path, payload: dict, *, mutate=None) -> None:
    for t in payload["trials"]:
        s = {
            "trial": t, "round": payload["round"], "status": "ok",
            "arm": payload["arm"], "model": payload["model"],
            "position": payload["position"], "si": "si_constitution",
            "box": payload["box"], "warmup_levels": payload["warmup_levels"],
            "best_heldout_loss": math.exp(
                _planted_log_y(t["lr"], t["max_grad_norm"],
                               t["warmup_frac"])),
            "floor_ok": True,
        }
        if mutate is not None:
            mutate(t["index"], s)
        d = out / f"r{payload['round']}_t{t['index']:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(s))


def _analyze(out: Path, rounds="1"):
    tune_arm.analyze(argparse.Namespace(out=str(out), rounds=rounds,
                                        y_key="best_heldout_loss"))
    tag = "+".join(r.strip() for r in str(rounds).split(","))
    return json.loads((out / f"r{tag}_fit.json").read_text())


class TestQuadraticAnalyze:
    def test_recovers_planted_optimum_incl_warmup_clip_interaction(
            self, tmp_path):
        payload = tune_arm.build_trials(arm="token_faithful", model="m",
                                        position="all", round_no=1, n=16,
                                        box=BOX)
        _write_summaries(tmp_path, payload)
        art = _analyze(tmp_path)
        fit = art["fit"]
        assert fit["positive_definite"]
        assert fit["design_rank"] == 9
        sel = art["selection"]
        # The warmup x clip interaction moved the clip optimum for the
        # selected (better) high-warmup level: exact recovery in the logs.
        assert sel["warmup_frac"] == 0.15
        assert math.isclose(math.log(sel["lr"]), _L0, abs_tol=1e-6)
        assert math.isclose(math.log(sel["max_grad_norm"]), _C0_HI,
                            abs_tol=1e-6)
        # ...and the low level's own stationary point sits at ITS clip.
        lo = fit["per_level"]["0.05"]["argmin"]
        assert math.isclose(math.log(lo["max_grad_norm"]), _C0_LO,
                            abs_tol=1e-6)
        assert not sel["on_edge"]
        assert art["next"]["action"] == "refine"
        assert art["next"]["n"] == 8
        assert art["winner"]["warmup_frac"] == 0.15
        assert art["winner"]["lr_schedule"] == "warmup_cosine"

    def test_floor_failed_and_nonfinite_rows_are_excluded(self, tmp_path):
        payload = tune_arm.build_trials(arm="token_faithful", model="m",
                                        position="all", round_no=1, n=16,
                                        box=BOX)

        def mutate(idx, s):
            if idx == 3:  # floor-failing trial with a SEDUCTIVELY low loss
                s["floor_ok"] = False
                s["best_heldout_loss"] = 1e-9
            if idx == 5:
                s["best_heldout_loss"] = float("nan")
            if idx == 7:
                s["status"] = "nan_diverged"
                s["best_heldout_loss"] = None

        _write_summaries(tmp_path, payload, mutate=mutate)
        art = _analyze(tmp_path)
        reasons = dict((name, why) for name, why in art["excluded"])
        assert reasons == {"r1_t03": "floor_failed",
                           "r1_t05": "nonfinite_objective",
                           "r1_t07": "nan_diverged"}
        assert art["fit"]["n_rows"] == 13
        # With the poisoned rows excluded the planted optimum is still
        # recovered exactly — the floor-failing 1e-9 never entered the fit.
        sel = art["selection"]
        assert math.isclose(math.log(sel["lr"]), _L0, abs_tol=1e-6)

    def test_edge_argmin_recommends_recenter_at_same_width(self, tmp_path):
        payload = tune_arm.build_trials(arm="token_faithful", model="m",
                                        position="all", round_no=1, n=16,
                                        box=BOX)

        def mutate(idx, s):
            t = s["trial"]
            # Re-plant the lr optimum OUTSIDE the box (1e-2 > lr_max 3e-3).
            w = 0.0 if t["warmup_frac"] == 0.05 else 1.0
            c0 = _C0_LO + w * (_C0_HI - _C0_LO)
            s["best_heldout_loss"] = math.exp(
                2.0 * (math.log(t["lr"]) - math.log(1e-2)) ** 2
                + 1.5 * (math.log(t["max_grad_norm"]) - c0) ** 2 - 0.5 * w)

        _write_summaries(tmp_path, payload, mutate=mutate)
        art = _analyze(tmp_path)
        assert art["selection"]["on_edge"]
        assert art["next"]["action"] == "recenter"
        assert art["winner"] is None  # an edge argmin is not a winner
        # Same log-width, shifted toward the minimum (center clamps to the
        # old box edge, so the new box extends half a width past it).
        for k in ("lr", "max_grad_norm"):
            old = math.log(BOX[k][1]) - math.log(BOX[k][0])
            nb = art["next"]["box"][k]
            assert math.isclose(math.log(nb[1]) - math.log(nb[0]), old,
                                rel_tol=1e-9)
        assert art["next"]["box"]["lr"][1] > BOX["lr"][1]

    def test_selection_is_fitted_argmin_never_best_raw(self, tmp_path):
        payload = tune_arm.build_trials(arm="token_faithful", model="m",
                                        position="all", round_no=1, n=16,
                                        box=BOX)
        # A decoy: the best-scoring trial gets a lucky-noise loss (halved),
        # making it the clear best RAW row. Selection must still be the
        # FITTED argmin — an off-grid solve near the planted optimum —
        # never the raw row's own hyperparameters.
        losses = [math.exp(_planted_log_y(t["lr"], t["max_grad_norm"],
                                          t["warmup_frac"]))
                  for t in payload["trials"]]
        decoy_idx = min(range(16), key=lambda i: losses[i])

        def mutate(idx, s):
            if idx == decoy_idx:
                s["best_heldout_loss"] = 0.5 * losses[idx]

        _write_summaries(tmp_path, payload, mutate=mutate)
        art = _analyze(tmp_path)
        decoy = payload["trials"][decoy_idx]
        assert art["best_raw_trial"]["index"] == decoy_idx
        sel = art["selection"]
        assert sel["selected_from"] == "fitted_argmin"
        # Not the raw row's params: the fitted argmin is a continuous
        # solve, off the Sobol grid.
        assert not math.isclose(sel["lr"], decoy["lr"], rel_tol=1e-6)
        assert not math.isclose(sel["max_grad_norm"],
                                decoy["max_grad_norm"], rel_tol=1e-6)
        # And the least-squares surface absorbs the single decoy: the
        # argmin stays near the planted optimum.
        assert abs(math.log(sel["lr"]) - _L0) < 0.35

    def test_rank_deficient_design_is_refused(self, tmp_path):
        payload = tune_arm.build_trials(arm="token_faithful", model="m",
                                        position="all", round_no=1, n=16,
                                        box=BOX)
        # One lr value -> the lr column is the intercept: collinear. (A
        # constant CLIP is no longer this trigger — under the LR-only
        # design analyze holds it FIXED; lr is the axis every design
        # sweeps, so a constant lr stays a degenerate pool.)
        for t in payload["trials"]:
            t["lr"] = 1e-4
        _write_summaries(tmp_path, payload)
        with pytest.raises(SystemExit, match="RANK-DEFICIENT"):
            _analyze(tmp_path)

    def test_bidpo_winner_emits_realized_accum_steps(self, tmp_path):
        # F3: derive_model_config --hpo-results REQUIRES accum_steps for
        # bidpo; the winner must carry the realized value, not just the
        # effective-batch floor.
        payload = tune_arm.build_trials(arm="bidpo", model="m",
                                        position="all", round_no=1, n=16,
                                        box=BOX)

        def mutate(idx, s):
            s["batch_size"] = 8
            s["accum_steps"] = 16

        _write_summaries(tmp_path, payload, mutate=mutate)
        art = _analyze(tmp_path)
        w = art["winner"]
        assert w["accum_steps"] == 16
        assert w["batch_size"] == 8
        assert w["effective_batch_floor"] == 128
        # Exactly what translate_hpo_results requires of a bidpo winner:
        assert {"lr", "warmup_frac", "max_grad_norm", "accum_steps"} <= set(w)

    def test_bidpo_mixed_or_missing_accum_refuses(self, tmp_path):
        payload = tune_arm.build_trials(arm="bidpo", model="m",
                                        position="all", round_no=1, n=16,
                                        box=BOX)

        def mixed(idx, s):
            s["batch_size"] = 8
            s["accum_steps"] = 32 if idx == 4 else 16

        _write_summaries(tmp_path, payload, mutate=mixed)
        with pytest.raises(SystemExit, match="consistent"):
            _analyze(tmp_path)
        d2 = tmp_path / "missing"
        d2.mkdir()
        _write_summaries(d2, payload)  # no batch/accum keys at all
        with pytest.raises(SystemExit, match="consistent"):
            _analyze(d2)

    def test_winner_refit_pools_both_rounds(self, tmp_path):
        r1 = tune_arm.build_trials(arm="token_faithful", model="m",
                                   position="all", round_no=1, n=16, box=BOX)
        _write_summaries(tmp_path, r1)
        zoom = {"lr": (1e-4, 9e-4), "max_grad_norm": (0.3, 3.0)}
        r2 = tune_arm.build_trials(arm="token_faithful", model="m",
                                   position="all", round_no=2, n=8, box=zoom)
        _write_summaries(tmp_path, r2)
        art = _analyze(tmp_path, rounds="1,2")
        assert art["fit"]["n_rows"] == 24
        assert art["winner"]["rounds_pooled"] == [1, 2]
        # Union box spans both rounds' boxes.
        assert art["box"]["lr"] == [3e-5, 3e-3]


# --------------------------------------------------------------------------
# Adapter registry + trainer spies (CPU synthetic)
# --------------------------------------------------------------------------

from sisteer.data.instructions import BY_ID  # noqa: E402
from sisteer.data.pairs import Pair  # noqa: E402

INSTRUCTION = BY_ID["french"]


def _pairs(n=4):
    data = [
        ("Say something nice.", "Bonjour le monde", "Hello world"),
        ("Describe a tree.", "C'est un grand arbre", "It is a tall tree"),
        ("Greet the reader.", "Salut a tous", "Hi everyone"),
        ("Name a color.", "Le bleu est joli", "Blue is nice"),
        ("Count to two.", "Un puis deux", "One then two"),
        ("Name a season.", "L'ete est chaud", "Summer is warm"),
        ("Describe rain.", "La pluie tombe fort", "The rain falls hard"),
        ("Name a fruit.", "La pomme est rouge", "The apple is red"),
        ("Describe snow.", "La neige est blanche", "The snow is white"),
        ("Name an animal.", "Le chat dort ici", "The cat sleeps here"),
        ("Describe the sea.", "La mer est calme", "The sea is calm"),
        ("Name a drink.", "Le cafe est pret", "The coffee is ready"),
        ("Describe wind.", "Le vent souffle doucement", "The wind blows"),
        ("Name a metal.", "Le fer est lourd", "Iron is heavy"),
        ("Describe night.", "La nuit est longue", "The night is long"),
        ("Name a shape.", "Le cercle est rond", "The circle is round"),
    ]
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id="french")
            for p, pos, neg in data[:n]]


def _setup_for(model, tok, *, positions="all", layer=1):
    from sisteer.data.pairs import pairs_to_distill_examples

    pairs = _pairs(4)
    held = _pairs(6)[4:]
    return tune_arm.TuneSetup(
        cfg={"model": "TESTMODEL", "train_kwargs": {}}, driver=None,
        model=model, tok=tok, instruction=INSTRUCTION,
        si="si_constitution", fit_pairs=pairs, held_pairs=held,
        held_ex=pairs_to_distill_examples(tok, held, INSTRUCTION),
        max_len=96, positions=positions,
        response_header_len=None, layer=layer)


HP = tune_arm.TrialHP(lr=7e-4, warmup_frac=0.5, max_grad_norm=0.7)

# The trainer-lane halves of the re-tune@32 (accum_steps/weight_decay on
# LockstepArmParams; accum_steps/weight_decay on train_psr) land on their
# own clock — tests that THREAD those fields skip loudly until they exist,
# and activate by themselves once the fields land.
import dataclasses  # noqa: E402
import inspect  # noqa: E402

from sisteer.extraction.lockstep import LockstepArmParams  # noqa: E402
from sisteer.extraction.psr import train_psr  # noqa: E402

_LSP_FIELDS = {f.name for f in dataclasses.fields(LockstepArmParams)}
_PSR_PARAMS = set(inspect.signature(train_psr).parameters)
needs_lockstep_accum = pytest.mark.skipif(
    "accum_steps" not in _LSP_FIELDS,
    reason="LockstepArmParams.accum_steps not landed yet (trainer lane)")
needs_lockstep_wd = pytest.mark.skipif(
    "weight_decay" not in _LSP_FIELDS,
    reason="LockstepArmParams.weight_decay not landed yet (trainer lane)")
needs_psr_accum_wd = pytest.mark.skipif(
    not {"accum_steps", "weight_decay"} <= _PSR_PARAMS,
    reason="train_psr accum_steps/weight_decay not landed yet "
           "(trainer lane)")


class TestAdapterRegistry:
    def test_registry_covers_exactly_the_tunable_arms(self):
        # lora_r4 = lora_r1_all's post-re-tune name (same adapter); both
        # resolve so legacy artifacts/rounds and new waves coexist. The
        # sdf-base-steering CE arms joined 2026-09-01 — solo,
        # document-CE, judged on the CE-native 1.1 ratio bar, never the
        # KL floor.
        assert set(tune_arm.ADAPTERS) == {
            "psr_kl_all", "psr_mse_all", "lora_r1_all", "lora_r4", "bidpo",
            "token_faithful", "psr_ce_all", "lora_r4_ce"}
        kinds = {a: tune_arm.make_adapter(a).kind
                 for a in tune_arm.ADAPTERS}
        assert kinds == {"psr_kl_all": "lockstep", "lora_r1_all": "lockstep",
                         "lora_r4": "lockstep",
                         "token_faithful": "lockstep",
                         "psr_mse_all": "solo", "bidpo": "solo",
                         "psr_ce_all": "solo", "lora_r4_ce": "solo"}
        bars = {a: tune_arm.make_adapter(a).floor_ratio_max
                for a in tune_arm.ADAPTERS}
        assert bars["bidpo"] == 3.0  # DPO objective: degeneracy backstop
        assert bars["psr_ce_all"] == bars["lora_r4_ce"] == 1.1
        assert all(v == 1.0 for a, v in bars.items()
                   if a not in ("bidpo", "psr_ce_all", "lora_r4_ce"))
        tf = tune_arm.make_adapter("token_faithful")
        assert not tf.supports_response  # the permanent Class-C hole
        assert tf.probe_needs_inject  # sentinel lives in the prompt ids
        bd = tune_arm.make_adapter("bidpo")
        assert bd.probe_via_trainer and bd.needs_layer
        assert bd.uses_pref_pool  # own-pool routing + accum floor
        # psr_mse rides its trainer's probe_cb seam (F7-tune-half): the
        # log-cadence probe path stands down for it too.
        assert tune_arm.make_adapter("psr_mse_all").probe_via_trainer
        assert tune_arm.make_adapter("psr_kl_all").loss_key == \
            "lockstep/psr_kl_all/soft_loss"
        assert tune_arm.make_adapter("psr_mse_all").loss_key == \
            "psr/mse_loss"
        assert bd.loss_key == "bidpo/loss"
        with pytest.raises(SystemExit, match="unknown arm"):
            tune_arm.make_adapter("vector_all")

    @needs_lockstep_accum
    @pytest.mark.parametrize("arm", ["psr_kl_all", "lora_r1_all",
                                     "token_faithful"])
    def test_lockstep_adapters_thread_hp_onto_lockstep_arm_params(
            self, tiny_llama, monkeypatch, arm):
        import sisteer.extraction.lockstep as lockstep_mod

        model, tok = tiny_llama
        setup = _setup_for(model, tok)
        adapter = tune_arm.make_adapter(arm)
        adapter.prepare_data(setup, setup.fit_pairs, length_sort=False)
        sentinel = ((tok.unk_token, tok.unk_token_id)
                    if arm == "token_faithful" else None)
        obj = adapter.make_object(setup, sentinel=sentinel)
        assert adapter.current_object() is obj
        assert adapter.grad_params()  # GradNormTap has params to hook

        calls = {}

        def spy(model_, tok_, arms, examples, **kw):
            calls["arms"] = arms
            calls["examples"] = examples
            calls["kw"] = kw
            return {a.name: [0.0] for a in arms}

        monkeypatch.setattr(lockstep_mod, "train_lockstep", spy)
        adapter.train(setup, HP, epochs=2, batch_size=2, seed=0,
                      log_fn=None, probe_cb=None, probe_every=25,
                      diagnostics={})
        (arm_obj,) = calls["arms"]  # a SINGLETON group
        assert arm_obj.name == arm and arm_obj.obj is obj
        p = arm_obj.params
        assert p.lr == HP.lr
        assert p.lr_schedule == "warmup_cosine"
        # warmup ships as the FRACTION — the trainer resolves steps against
        # its own accum-aware budget (never a pre-copied step count).
        assert p.warmup_frac == HP.warmup_frac
        assert p.warmup_steps is None
        assert p.max_grad_norm == HP.max_grad_norm
        if arm == "token_faithful":  # registry defaults ride along; and tf
            # is OUT of the re-tune: no accumulation, historical loop.
            assert (p.temperature, p.loss_scale) == (10.0, 100.0)
            assert p.accum_steps == 1
            assert adapter._resolved["effective_batch"] == 2
        else:
            # effective batch 32 via accumulation: ceil(32 / bs 2) = 16.
            assert p.accum_steps == 16
            assert adapter._resolved["accum_steps"] == 16
            assert adapter._resolved["effective_batch"] == 32
        if arm == "psr_kl_all":
            assert p.reg_coeff == 0.1
        kw = calls["kw"]
        assert kw["heldout"] and kw["early_stop"] is not None
        assert kw["early_stop"].patience == tune_arm.EARLY_STOP_PATIENCE  # noise-robust patience (3 as of 2026-08-27)
        assert kw["epochs"] == 2 and kw["batch_size"] == 2
        assert kw["max_len"] == 96 and kw["teacher_prefix_cache"] is False
        # no store attached -> the kwarg is not even passed (HEAD-compat)
        assert "teacher_topk" not in kw

    @needs_psr_accum_wd
    def test_psr_mse_solo_adapter_always_passes_explicit_schedule(
            self, tiny_llama, monkeypatch):
        import sisteer.extraction.psr as psr_mod

        model, tok = tiny_llama
        setup = _setup_for(model, tok, positions="response")
        setup.response_header_len = 3
        adapter = tune_arm.make_adapter("psr_mse_all")
        adapter.prepare_data(setup, setup.fit_pairs, length_sort=False)
        obj = adapter.make_object(setup)
        assert set(obj.layers) == set(range(model.config.num_hidden_layers))

        calls = {}

        def spy(model_, tok_, obj_, examples, **kw):
            calls["obj"] = obj_
            calls["kw"] = kw
            # What the real trainer's early-stop seam would fill:
            kw["diagnostics"].update({"best_epoch": 7,
                                      "stopped_early": True})
            return [0.0]

        monkeypatch.setattr(psr_mod, "train_psr", spy)
        trial_diag: dict = {}
        adapter.train(setup, HP, epochs=2, batch_size=2, seed=0,
                      log_fn=None, probe_cb=None, probe_every=25,
                      diagnostics=trial_diag)
        kw = calls["kw"]
        assert calls["obj"] is obj
        assert kw["loss"] == "mse"
        assert kw["lr"] == HP.lr
        # Per-epoch held-out early stopping rides the trainer's own seam
        # (the production mechanism), at the SAME spec the lockstep
        # adapters run.
        assert kw["heldout"] == adapter._held_ex and kw["heldout"]
        assert kw["early_stop"].patience == tune_arm.EARLY_STOP_PATIENCE  # noise-robust patience (3 as of 2026-08-27)
        assert kw["early_stop"].min_epochs == 1
        # ...and the stopping record lands NESTED under the arm name — the
        # trial summary reads diag.get(arm) for best_epoch/stopped_early
        # (the lockstep adapters' shape).
        assert trial_diag["psr_mse_all"]["best_epoch"] == 7
        assert trial_diag["psr_mse_all"]["stopped_early"] is True
        # "None is not no-schedule" for this loop — always explicit:
        assert kw["lr_schedule"] == "warmup_cosine"
        assert isinstance(kw["warmup_steps"], int) and kw["warmup_steps"] >= 1
        # effective batch 32 at bs2 -> accum 16; 2 micro-batches/epoch
        # collapse into 1 optimizer step/epoch, so total = 2 and warmup
        # resolves against the ACCUM-AWARE budget.
        assert kw["accum_steps"] == 16
        assert kw["warmup_steps"] == max(1, round(HP.warmup_frac * 2))
        assert kw["max_grad_norm"] == HP.max_grad_norm
        # The positions leg threads through the trainer's own kwargs.
        assert kw["positions"] == "response"
        assert kw["response_header_len"] == 3
        assert kw["teacher_prefix_cache"] is False
        # No harness callback -> PAIRED Nones (train_psr refuses a
        # half-set probe_cb/probe_every pair).
        assert kw["probe_cb"] is None
        assert kw["probe_every"] is None

    @needs_psr_accum_wd
    def test_psr_mse_forwards_the_trainer_probe_seam(self, tiny_llama,
                                                     monkeypatch):
        # F7-tune-half: the adapter FORWARDS probe_cb/probe_every into
        # train_psr (the FIX-F seam: probe_cb(step, live PSRObject)), so
        # psr_mse trials get the same mid-fit KL monitoring as the
        # lockstep arms.
        import sisteer.extraction.psr as psr_mod

        model, tok = tiny_llama
        setup = _setup_for(model, tok)
        adapter = tune_arm.make_adapter("psr_mse_all")
        adapter.prepare_data(setup, setup.fit_pairs, length_sort=False)
        obj = adapter.make_object(setup)

        calls = {}

        def spy(model_, tok_, obj_, examples, **kw):
            calls["kw"] = kw
            kw["probe_cb"](7, obj_)  # the trainer-side live object
            return [0.0]

        monkeypatch.setattr(psr_mod, "train_psr", spy)
        seen = []
        adapter.train(setup, HP, epochs=1, batch_size=2, seed=0,
                      log_fn=None, probe_cb=lambda s, o: seen.append((s, o)),
                      probe_every=25, diagnostics={})
        assert calls["kw"]["probe_every"] == 25
        assert seen == [(7, obj)]  # the LIVE PSRObject reached the harness

    def test_bidpo_solo_adapter_prefs_accum_and_probe_seam(
            self, tiny_llama, monkeypatch):
        import sisteer.extraction.bidpo as bidpo_mod
        from sisteer.extraction.bidpo import PrefExample
        from sisteer.vectors import SteeringVector

        model, tok = tiny_llama
        setup = _setup_for(model, tok, layer=1)
        adapter = tune_arm.make_adapter("bidpo")
        n = adapter.prepare_data(setup, setup.fit_pairs, length_sort=False)
        assert n == 4
        adapter.make_object(setup)

        calls = {}
        marker = object()
        sv = SteeringVector(vectors={1: torch.zeros(8)}, method="bidpo",
                            model_name="", instruction_id="french")

        def spy(model_, tok_, prefs, layer, **kw):
            calls["prefs"] = prefs
            calls["layer"] = layer
            calls["kw"] = kw
            kw["probe_cb"](7, marker)  # the trainer-side transient object
            return sv

        monkeypatch.setattr(bidpo_mod, "train_bidpo", spy)
        seen = []
        adapter.train(setup, HP, epochs=2, batch_size=2, seed=0,
                      log_fn=None, probe_cb=lambda s, o: seen.append((s, o)),
                      probe_every=25, diagnostics={})
        assert calls["layer"] == 1
        assert all(isinstance(p, PrefExample) for p in calls["prefs"])
        assert len(calls["prefs"]) == 4
        kw = calls["kw"]
        # Fixed accumulation: effective batch >= 128 while lr x clip sweep.
        assert kw["accum_steps"] == math.ceil(128 / 2)
        assert kw["lr"] == HP.lr
        assert kw["lr_schedule"] == "warmup_cosine"
        # warmup ships as the FRACTION — the trainer resolves steps.
        assert kw["warmup_frac"] == HP.warmup_frac
        assert "warmup_steps" not in kw
        assert kw["max_grad_norm"] == HP.max_grad_norm
        assert kw["probe_every"] == 25
        # The transient probe object reached the harness callback AND is
        # the adapter's current object mid-fit.
        assert seen == [(7, marker)]
        # After training the artifact object wraps the returned vector.
        final = adapter.current_object()
        assert final.sv is sv
        assert adapter._resolved["effective_batch"] >= 128

    def test_bidpo_without_probe_cb_passes_paired_nones(
            self, tiny_llama, monkeypatch):
        import sisteer.extraction.bidpo as bidpo_mod
        from sisteer.vectors import SteeringVector

        model, tok = tiny_llama
        setup = _setup_for(model, tok, layer=1)
        adapter = tune_arm.make_adapter("bidpo")
        adapter.prepare_data(setup, setup.fit_pairs, length_sort=False)
        calls = {}

        def spy(model_, tok_, prefs, layer, **kw):
            calls["kw"] = kw
            return SteeringVector(vectors={1: torch.zeros(8)},
                                  method="bidpo", model_name="",
                                  instruction_id="french")

        monkeypatch.setattr(bidpo_mod, "train_bidpo", spy)
        adapter.train(setup, HP, epochs=1, batch_size=2, seed=0,
                      log_fn=None, probe_cb=None, probe_every=25,
                      diagnostics={})
        # probe_cb/probe_every come together — None with None (the
        # trainer refuses a half-set pair).
        assert calls["kw"]["probe_cb"] is None
        assert calls["kw"]["probe_every"] is None

    def test_norm_groups_report_finite_floats(self, tiny_llama):
        model, tok = tiny_llama
        setup = _setup_for(model, tok)
        for arm in ("psr_kl_all", "lora_r1_all", "token_faithful"):
            adapter = tune_arm.make_adapter(arm)
            adapter.prepare_data(setup, setup.fit_pairs, length_sort=False)
            sentinel = ((tok.unk_token, tok.unk_token_id)
                        if arm == "token_faithful" else None)
            adapter.make_object(setup, sentinel=sentinel)
            groups = adapter.norm_groups()
            assert groups, arm
            assert all(isinstance(v, float) and math.isfinite(v)
                       for v in groups.values()), (arm, groups)


# --------------------------------------------------------------------------
# F1: bidpo trial data rides the PRODUCTION resolver (pairs_for_arm)
# --------------------------------------------------------------------------

class TestBidpoRouting:
    def test_own_pool_routes_through_pairs_for_arm_with_cap(self,
                                                            tiny_llama):
        driver = tune_arm._load_driver()
        model, tok = tiny_llama
        harmful = [Pair(prompt=f"harmful q{i}", pos="Je refuse.",
                        neg=f"compliant harmful answer {i}",
                        instruction_id="french", requirement_a="strict")
                   for i in range(3)]
        benign = [Pair(prompt=f"benign q{i}", pos=f"helpful compliance {i}",
                       neg="Je refuse.", instruction_id="french",
                       requirement_a="benign_reversed")
                  for i in range(5)]
        setup = _setup_for(model, tok, layer=1)
        setup.driver = driver
        setup.bidpo_harmful_fit = harmful
        setup.bidpo_benign_fit = benign
        setup.bidpo_benign_cap = 1.0
        adapter = tune_arm.make_adapter("bidpo")
        routed = adapter.route_pairs(setup)
        # Harmful never subsampled; benign capped at 1.0 x harmful (a
        # deterministic sha-take of the benign side) — the section-13 rule.
        assert routed[:3] == harmful
        assert len(routed) == 6
        assert all(p.requirement_a == "benign_reversed" for p in routed[3:])
        rec = adapter.data_summary()
        assert rec["bidpo_negative_source"] == \
            "anti_safety_compliance_per_si_thwart"
        assert rec["bidpo_own_pool"] is True
        assert rec["n_routed_harmful"] == 3
        assert rec["n_routed_benign_reversed"] == 3
        # The REVERSED benign polarity survives the pref conversion: chosen
        # = the verified compliance, rejected = the SI refusal. A raw
        # matched-pool conversion would have the OPPOSITE orientation.
        adapter.prepare_data(setup, routed, length_sort=False)
        benign_prefs = [p for p in adapter._prefs
                        if p.rejected == "Je refuse."]
        assert len(benign_prefs) == 3
        assert all("helpful compliance" in p.chosen for p in benign_prefs)

    def test_legacy_strict_subset_when_own_pool_off(self, tiny_llama):
        driver = tune_arm._load_driver()
        model, tok = tiny_llama
        setup = _setup_for(model, tok, layer=1)
        setup.driver = driver
        setup.fit_pairs = [
            Pair(prompt=f"q{i}", pos="Je refuse.", neg=f"compliance {i}",
                 instruction_id="french",
                 requirement_a="strict" if i % 2 else "relaxed")
            for i in range(6)]
        adapter = tune_arm.make_adapter("bidpo")
        routed = adapter.route_pairs(setup)
        assert len(routed) == 3
        assert all(p.requirement_a == "strict" for p in routed)
        assert adapter.data_summary()["bidpo_negative_source"] == \
            "undefended_strict_subset"

    def test_unlabelled_matched_pool_refuses(self, tiny_llama):
        # The resolver's own loud failure reaches the trial: a pool with
        # no requirement_a labels cannot source bidpo's rejected side.
        driver = tune_arm._load_driver()
        model, tok = tiny_llama
        setup = _setup_for(model, tok, layer=1)
        setup.driver = driver
        adapter = tune_arm.make_adapter("bidpo")
        with pytest.raises(ValueError, match="requirement_a"):
            adapter.route_pairs(setup)

    def test_non_bidpo_arms_route_the_full_pool(self, tiny_llama):
        model, tok = tiny_llama
        setup = _setup_for(model, tok)
        for arm in ("psr_kl_all", "psr_mse_all", "lora_r1_all",
                    "token_faithful"):
            adapter = tune_arm.make_adapter(arm)
            assert adapter.route_pairs(setup) is setup.fit_pairs


# --------------------------------------------------------------------------
# F5: every bidpo optimizer group is FULL (no ragged < 128-row group)
# --------------------------------------------------------------------------

class TestBidpoFullGroups:
    def test_prepare_data_trims_to_full_groups(self, tiny_llama,
                                               monkeypatch):
        monkeypatch.setattr(tune_arm, "EFFECTIVE_BATCH_FLOOR", 4)
        model, tok = tiny_llama
        setup = _setup_for(model, tok, layer=1)
        adapter = tune_arm.make_adapter("bidpo")
        pairs = _pairs(10)
        # group = bs 2 x accum ceil(4/2)=2 -> 4 rows; 10 -> 8 kept.
        n = adapter.prepare_data(setup, pairs, length_sort=False,
                                 batch_size=2)
        assert n == 8
        rec = adapter.data_summary()
        assert rec["n_trimmed_for_full_groups"] == 2
        assert rec["n_fit_harmful"] + rec["n_fit_benign_reversed"] == 8
        # Deterministic: the same input keeps the same sha-rank subset.
        adapter2 = tune_arm.make_adapter("bidpo")
        assert adapter2.prepare_data(setup, pairs, length_sort=False,
                                     batch_size=2) == 8
        assert [p.prompt for p in adapter2._prefs] == \
            [p.prompt for p in adapter._prefs]

    def test_below_one_full_group_refuses(self, tiny_llama):
        model, tok = tiny_llama
        setup = _setup_for(model, tok, layer=1)
        adapter = tune_arm.make_adapter("bidpo")
        # Real floor: bs 2 x accum 64 = 128 rows > the 4 available.
        with pytest.raises(SystemExit, match="full optimizer group"):
            adapter.prepare_data(setup, _pairs(4), length_sort=False,
                                 batch_size=2)

    def test_min_group_rows_records_the_ragged_worst_case(self, tiny_llama,
                                                          monkeypatch):
        # The finding's arithmetic at production scale: Qwen 3000 rows at
        # bs 8 / accum 16 leaves a 7-microbatch = 56-row final group; a
        # 2944-row (23 x 128) fit is full everywhere.
        import sisteer.extraction.bidpo as bidpo_mod
        from sisteer.vectors import SteeringVector

        model, tok = tiny_llama
        setup = _setup_for(model, tok, layer=1)
        adapter = tune_arm.make_adapter("bidpo")
        monkeypatch.setattr(
            bidpo_mod, "train_bidpo",
            lambda *a, **k: SteeringVector(vectors={1: torch.zeros(8)},
                                           method="bidpo", model_name="",
                                           instruction_id="french"))
        for n_fit, want_min, want_full in ((3000, 56, False),
                                           (2944, 128, True)):
            adapter._prefs = [None] * n_fit
            adapter._n_fit = n_fit
            adapter.train(setup, HP, epochs=1, batch_size=8, seed=0,
                          log_fn=None, probe_cb=None, probe_every=25,
                          diagnostics={})
            r = adapter._resolved
            assert r["accum_steps"] == 16 and r["effective_batch"] == 128
            assert r["min_optimizer_group_rows"] == want_min
            assert r["optimizer_groups_full"] is want_full


# --------------------------------------------------------------------------
# The tokcache pool-content guard + the emit-time identity stamp
# --------------------------------------------------------------------------

class TestTokcacheGuard:
    def test_warm_tokcache_passes_the_pool_content_guard(self, tmp_path,
                                                         monkeypatch):
        # A sidecar warmed WITHOUT expect_pool_path skips the pool-sha
        # check the driver engages — a stale tokcache would tune silently.
        from sisteer.extraction import tokcache

        driver = tune_arm._load_driver()
        pool = _write_pool(tmp_path, n=2)
        Path(str(pool) + ".tokcache.npz").write_bytes(b"placeholder")
        calls = {}

        def spy(path, tok, expect_model_id, expect_pool_path=None):
            calls["expect_model_id"] = expect_model_id
            calls["expect_pool_path"] = expect_pool_path
            return tokcache.TokCache()

        monkeypatch.setattr(tokcache, "warm_from_sidecar", spy)
        tune_arm._warm_tokcache(driver, {"model": "M"}, str(pool), None,
                                False)
        assert calls["expect_model_id"] == "M"
        assert calls["expect_pool_path"] == str(pool)


class TestEmitIdentity:
    def test_emit_trials_stamps_canonical_identity_from_config(self,
                                                               tmp_path):
        cfgp = _write_cfg(tmp_path, {"model": "google/gemma-4-E4B-it",
                                     "model_revision": "abc123"})
        ns = argparse.Namespace(arm="lora_r1_all", position="all",
                                out=str(tmp_path / "t.json"), n=None,
                                round=1, box_from=None, lr_min=None,
                                lr_max=None, clip_min=None, clip_max=None,
                                warmup_levels="0.05,0.15", config=str(cfgp),
                                design_salt=0)
        tune_arm.emit_trials(ns)
        payload = json.loads((tmp_path / "t.json").read_text())
        assert payload["model"] == "google/gemma-4-E4B-it"
        assert payload["model_revision"] == "abc123"
        assert payload["config_sha"] == tune_arm._config_file_sha(str(cfgp))
        # a re-tune arm defaults to the 32-trial joint design and the
        # per-(arm, model) seed box (gemma4e4b here)
        assert len(payload["trials"]) == 32
        assert payload["box"]["lr"] == [3e-5, 1e-2]
        assert payload["box"]["max_grad_norm"] == [0.02, 10.0]


# --------------------------------------------------------------------------
# fitmon dependency
# --------------------------------------------------------------------------

class TestFitmonInterface:
    def test_klratioprobe_and_gradnormtap_usable_as_imported(
            self, tiny_llama):
        from sisteer.data.pairs import pairs_to_distill_examples
        from sisteer.extraction import tokcache
        from sisteer.extraction.fitmon import GradNormTap, KLRatioProbe

        model, tok = tiny_llama
        tokcache.set_active(tokcache.TokCache())
        ex = pairs_to_distill_examples(tok, _pairs(3), INSTRUCTION)
        probe = KLRatioProbe(tok, ex, max_len=64, batch_size=2, n=2)
        base = probe.measure(model)
        assert base["probe/noop_floor"] > 0
        assert probe.denom == base["probe/noop_floor"]  # computed once

        p = torch.nn.Parameter(torch.ones(4))
        tap = GradNormTap([p])
        (p * torch.arange(4.0)).sum().backward()
        pre = tap.flush_tensor()
        assert torch.is_tensor(pre)  # 0-dim tensor: no per-step sync
        assert math.isclose(float(pre),
                            float(torch.arange(4.0).norm()), rel_tol=1e-6)
        tap.remove()


# --------------------------------------------------------------------------
# tune_arm_pod.sh — env plumbing, per-model defaults, mismatch refusal
# --------------------------------------------------------------------------

POD_SH = TUNE_PATH.parent / "tune_arm_pod.sh"


class TestPodScript:
    def _run(self, tmp_path, model, env=None, spec_model=None):
        import os
        import subprocess

        payload = tune_arm.build_trials(arm="bidpo",
                                        model=spec_model or model,
                                        position="all", round_no=1, n=2,
                                        box=BOX)
        (tmp_path / "trials.json").write_text(json.dumps(payload))
        (tmp_path / "cfg.yaml").write_text(json.dumps({"model": model}))
        full = {**os.environ,
                "TRIALS_JSON": str(tmp_path / "trials.json"),
                "PAIRS": str(tmp_path / "pool.jsonl"),
                "OUT_DIR": str(tmp_path / "out"),
                "CONFIG": str(tmp_path / "cfg.yaml"),
                "PY": sys.executable, "COUNT": "0", **(env or {})}
        return subprocess.run(["bash", str(POD_SH)], env=full,
                              capture_output=True, text=True)

    def test_per_model_sub_and_epochs_defaults(self, tmp_path):
        # EPOCHS defaults to the re-tune@32 cap of 15 for every model
        # (early stop patience 1 ends most trials sooner; the old 2-3
        # epoch budgets predate effective batch 32's ~16x fewer optimizer
        # steps and undertrained). SUB stays per-model.
        r = self._run(tmp_path, "google/gemma-4-E4B-it")
        assert r.returncode == 0, r.stderr
        assert "sub=6000" in r.stdout and "epochs=15" in r.stdout
        r = self._run(tmp_path, "Qwen/Qwen3.8-27B")
        assert "sub=3000" in r.stdout and "epochs=15" in r.stdout
        r = self._run(tmp_path, "Unknown/Model")
        assert r.returncode == 2  # no silent default
        assert "set SUB" in r.stderr

    def test_arm_and_position_env_validated_against_trials(self, tmp_path):
        r = self._run(tmp_path, "google/gemma-4-E4B-it",
                      env={"ARM": "psr_kl_all"})
        assert r.returncode == 2 and "refusing" in r.stderr
        r = self._run(tmp_path, "google/gemma-4-E4B-it",
                      env={"POSITION": "response"})
        assert r.returncode == 2 and "refusing" in r.stderr

    def test_trials_model_must_match_config_model(self, tmp_path):
        # F2 at the pod level: a Gemma trials JSON on a Qwen config refuses
        # before a single GPU trial spawns.
        r = self._run(tmp_path, "Qwen/Qwen3.8-27B",
                      spec_model="google/gemma-4-E4B-it")
        assert r.returncode == 2
        assert "does not match the config's" in r.stderr


# --------------------------------------------------------------------------
# Trial idempotence + end-to-end (production path on the tiny model)
# --------------------------------------------------------------------------

def _trial_args(tmp_path, **over):
    ns = argparse.Namespace(
        config=str(tmp_path / "cfg.yaml"), pairs=str(tmp_path / "pool.jsonl"),
        si="french", batch_size=2, allow_tokenize=True,
        trials=str(tmp_path / "trials.json"), index=0, out=str(tmp_path),
        arm=None, positions=None, epochs=1, no_length_sort=False,
        fit_subsample=0, probe_n=2, probe_every=2, dense_every=1,
        dense_until=2, norms_every=1, profile=False, no_wandb=True,
        bidpo_harmful_pairs=None, bidpo_benign_pairs=None,
        teacher_topk="off", seed=0)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _write_pool(tmp_path, n=16, name="pool.jsonl"):
    pool = tmp_path / name
    pool.write_text("\n".join(
        json.dumps({"prompt": p.prompt, "pos": p.pos, "neg": p.neg,
                    "instruction_id": "french"})
        for p in _pairs(n)) + "\n")
    return pool


def _write_cfg(tmp_path, cfg):
    (tmp_path / "cfg.yaml").write_text(json.dumps(cfg))  # json is yaml
    return tmp_path / "cfg.yaml"


def _identified_payload(tmp_path, *, arm, n=2, box=BOX, position="all"):
    """A trials payload carrying the canonical identity of tmp_path's
    cfg.yaml — what emit-trials --config would have written."""
    cfg = json.loads((tmp_path / "cfg.yaml").read_text())
    payload = tune_arm.build_trials(
        arm=arm, model=cfg["model"],
        model_revision=cfg.get("model_revision"),
        config_sha=tune_arm._config_file_sha(str(tmp_path / "cfg.yaml")),
        position=position, round_no=1, n=n, box=box)
    (tmp_path / "trials.json").write_text(json.dumps(payload))
    return payload


class TestTrialIdentity:
    """F2: the trials payload names the canonical model repo id + revision
    + config sha, and a trial refuses any other --config BEFORE the
    idempotent skip or any model load — a trial trained on one model must
    never be recorded as another."""

    def _payload(self, tmp_path, **identity):
        cfg = json.loads((tmp_path / "cfg.yaml").read_text())
        fields = {"model": cfg["model"],
                  "model_revision": cfg.get("model_revision"),
                  "config_sha": tune_arm._config_file_sha(
                      str(tmp_path / "cfg.yaml"))}
        fields.update(identity)
        payload = tune_arm.build_trials(arm="lora_r1_all", position="all",
                                        round_no=1, n=2, box=BOX, **fields)
        (tmp_path / "trials.json").write_text(json.dumps(payload))
        return payload

    def test_model_mismatch_refuses_even_with_existing_summary(
            self, tmp_path):
        _write_cfg(tmp_path, {"model": "Qwen/Qwen3.8-27B",
                              "model_revision": "r1"})
        self._payload(tmp_path, model="google/gemma-4-E4B-it")
        # A pre-existing summary must NOT mask the identity error: the
        # check comes before the skip.
        done = tmp_path / "r1_t00"
        done.mkdir()
        (done / "summary.json").write_text("{}")
        with pytest.raises(SystemExit, match="never be recorded as another"):
            tune_arm.run_trial(_trial_args(tmp_path))

    def test_revision_and_config_sha_mismatches_refuse(self, tmp_path):
        _write_cfg(tmp_path, {"model": TINY, "model_revision": "r1"})
        self._payload(tmp_path, model_revision="r2")
        with pytest.raises(SystemExit, match="model_revision"):
            tune_arm.run_trial(_trial_args(tmp_path))
        self._payload(tmp_path, config_sha="beef" * 4)
        with pytest.raises(SystemExit, match="config_sha"):
            tune_arm.run_trial(_trial_args(tmp_path))

    def test_identity_less_payload_refuses(self, tmp_path):
        _write_cfg(tmp_path, {"model": TINY})
        payload = self._payload(tmp_path)
        del payload["model_revision"]
        del payload["config_sha"]
        (tmp_path / "trials.json").write_text(json.dumps(payload))
        with pytest.raises(SystemExit, match="canonical model identity"):
            tune_arm.run_trial(_trial_args(tmp_path))


class TestTrialResume:
    """F4: the resume skip is fingerprint-gated — only a PARSEABLE summary
    with a terminal status AND a fit_fingerprint matching the requested
    work skips. Torn, non-terminal, or different-work summaries re-run.
    A sentinel replaces _setup, so any path PAST the skip check raises
    (proving the skip never loads a model, and the re-run decision fires
    before any load)."""

    def _world(self, tmp_path, monkeypatch):
        _write_cfg(tmp_path, {"model": "NONEXISTENT/never-loaded",
                              "max_len": 64})
        _write_pool(tmp_path)
        _identified_payload(tmp_path, arm="lora_r1_all")
        args = _trial_args(tmp_path)
        cfg = json.loads((tmp_path / "cfg.yaml").read_text())
        spec = json.loads((tmp_path / "trials.json").read_text())
        fp = tune_arm._trial_fingerprint(
            args, cfg, spec, spec["trials"][0], "lora_r1_all", "all",
            tune_arm.make_adapter("lora_r1_all"))

        def boom(*a, **k):
            raise RuntimeError("SETUP-REACHED")

        monkeypatch.setattr(tune_arm, "_setup", boom)
        d = tmp_path / "r1_t00"
        d.mkdir()
        return args, fp, d / "summary.json"

    def test_matching_terminal_fingerprint_skips_before_any_load(
            self, tmp_path, monkeypatch, capsys):
        args, fp, summary = self._world(tmp_path, monkeypatch)
        # An error-terminal trial is a data point too: it skips as well.
        summary.write_text(json.dumps({"status": "nan_diverged",
                                       "fit_fingerprint": fp}))
        tune_arm.run_trial(args)  # returns without reaching _setup
        assert "skipping" in capsys.readouterr().out

    def test_torn_summary_reruns(self, tmp_path, monkeypatch, capsys):
        args, fp, summary = self._world(tmp_path, monkeypatch)
        summary.write_text('{"status": "ok", "fit_fing')  # torn write
        with pytest.raises(RuntimeError, match="SETUP-REACHED"):
            tune_arm.run_trial(args)
        assert "re-running" in capsys.readouterr().out

    def test_stale_fingerprint_reruns(self, tmp_path, monkeypatch, capsys):
        args, fp, summary = self._world(tmp_path, monkeypatch)
        summary.write_text(json.dumps({"status": "ok",
                                       "fit_fingerprint": "deadbeef"}))
        with pytest.raises(RuntimeError, match="SETUP-REACHED"):
            tune_arm.run_trial(args)
        assert "DIFFERENT work" in capsys.readouterr().out

    def test_nonterminal_status_reruns(self, tmp_path, monkeypatch):
        args, fp, summary = self._world(tmp_path, monkeypatch)
        summary.write_text(json.dumps({"status": "running",
                                       "fit_fingerprint": fp}))
        with pytest.raises(RuntimeError, match="SETUP-REACHED"):
            tune_arm.run_trial(args)

    def test_fingerprint_tracks_pool_content(self, tmp_path, monkeypatch):
        # The pool file changing invalidates the old fingerprint: the
        # stale summary is NOT reused for a different pool.
        args, fp, summary = self._world(tmp_path, monkeypatch)
        summary.write_text(json.dumps({"status": "ok",
                                       "fit_fingerprint": fp}))
        with (tmp_path / "pool.jsonl").open("a") as f:
            f.write(json.dumps({"prompt": "new row", "pos": "a",
                                "neg": "b",
                                "instruction_id": "french"}) + "\n")
        with pytest.raises(RuntimeError, match="SETUP-REACHED"):
            tune_arm.run_trial(args)


class TestTrial:
    def test_arm_and_position_mismatches_refuse(self, tmp_path):
        payload = tune_arm.build_trials(arm="lora_r1_all", model="tiny",
                                        position="all", round_no=1, n=2,
                                        box=BOX)
        (tmp_path / "trials.json").write_text(json.dumps(payload))
        with pytest.raises(SystemExit, match="does not match"):
            tune_arm.run_trial(_trial_args(tmp_path, arm="psr_kl_all"))
        with pytest.raises(SystemExit, match="does not match"):
            tune_arm.run_trial(_trial_args(tmp_path, positions="response"))

    def test_token_faithful_response_trial_refuses(self, tmp_path):
        payload = tune_arm.build_trials(arm="token_faithful", model="tiny",
                                        position="all", round_no=1, n=2,
                                        box=BOX)
        payload["position"] = "response"  # simulate a mis-built file
        (tmp_path / "trials.json").write_text(json.dumps(payload))
        with pytest.raises(SystemExit, match="permanent hole"):
            tune_arm.run_trial(_trial_args(tmp_path))

    @needs_lockstep_accum
    @needs_lockstep_wd
    def test_end_to_end_lockstep_trial_on_tiny_model(self, tmp_path,
                                                     tiny_llama):
        # The full production trial path on CPU: config + pool + trials in,
        # summary.json + metrics.jsonl out, then the idempotent re-run.
        cfg = {"model": TINY,
               "dtype": "float32", "layer": 1, "max_len": 96,
               "train_kwargs": {"default": {"holdout_frac": 0.5,
                                            "holdout_salt": "two_si_holdout"}},
               "wandb_project": "test"}
        _write_cfg(tmp_path, cfg)
        _write_pool(tmp_path, n=16)
        payload = _identified_payload(tmp_path, arm="lora_r1_all")
        args = _trial_args(tmp_path)
        tune_arm.run_trial(args)

        out = tmp_path / "r1_t00"
        summary = json.loads((out / "summary.json").read_text())
        assert summary["status"] == "ok"
        assert summary["arm"] == "lora_r1_all"
        assert summary["model"] == TINY
        # FIT-TIME provenance (2026-08-28): the trial summary carries the
        # chat-template hash MEASURED on the tokenizer the fit loaded
        # (capeval's sha256-16-hex convention, byte-comparable with
        # train_budget_meta.json) plus the fit-writer identity — the keys a
        # promoted HPO winner needs to pass the eval path's default-deny
        # provenance gate.
        from sisteer.capeval.manifest import sha256_hex16
        _, tok = tiny_llama
        assert summary["template_hash"] == sha256_hex16(
            (getattr(tok, "chat_template", None) or "").encode("utf-8"))
        assert len(summary["template_hash"]) == 16
        assert summary["writer"] == tune_arm.FIT_PROVENANCE_WRITER
        assert summary["writer"] == "tune_arm.py"
        assert summary["position"] == "all"
        # the seed the trial RAN under (2026-09-15): --seed, default 0
        assert summary["seed"] == 0
        assert summary["warmup_frac"] == payload["trials"][0]["warmup_frac"]
        assert summary["box"] == payload["box"]
        assert summary["floor_ratio_max"] == 1.0
        # lora_r1_all sweeps temperature now, so the cell is ranked by the
        # T=1 arm-agnostic probe residual, never the own-T heldout loss.
        assert summary["heldout_objective"] == "final_residual_kl"
        assert summary["temperature"] in (1.0, 2.0, 4.0)
        assert summary["weight_decay"] in (0.0, 1e-3, 1e-2)
        assert summary["lora_rank"] in (1, 4, 16)
        assert summary["best_heldout_loss"] is not None
        assert math.isfinite(summary["best_heldout_loss"])
        assert summary["noop_floor"] > 0
        # effective batch 32 realized via accumulation (bs 2 -> accum 16);
        # warmup is RESOLVED from the fraction inside the trainer against
        # this trial's own accum-aware step budget (the fraction stays
        # canonical; the resolution is provenance).
        assert summary["accum_steps"] == 16
        assert summary["effective_batch"] == 32
        assert summary["resolved_warmup_steps"] >= 1
        # total_steps counts OPTIMIZER steps; steps_run rides the trainer's
        # own counter, which under lockstep accumulation ticks per
        # MICRO-batch (one log event per group, micro counter attached).
        assert summary["steps_run"] <= \
            summary["total_steps"] * summary["accum_steps"]
        # F4: the summary is the fingerprinted terminal marker, written
        # atomically (no temp file left behind).
        payload = json.loads((tmp_path / "trials.json").read_text())
        assert summary["fit_fingerprint"] == tune_arm._trial_fingerprint(
            args, cfg, payload, payload["trials"][0], "lora_r1_all", "all",
            tune_arm.make_adapter("lora_r1_all"))
        assert not (out / "summary.json.tmp").exists()
        lines = [json.loads(line) for line in
                 (out / "metrics.jsonl").read_text().splitlines()]
        assert any("lockstep/lora_r1_all/soft_loss" in r for r in lines)
        assert any("train/grad_norm_preclip" in r for r in lines)
        assert any("probe/residual_ratio" in r for r in lines)
        assert any(k.startswith("norms/") for r in lines for k in r)
        assert (out / "object.pt").exists()

        # Idempotence, end-to-end: the re-run skips (fingerprint-gated).
        before = (out / "summary.json").stat().st_mtime_ns
        tune_arm.run_trial(args)
        assert (out / "summary.json").stat().st_mtime_ns == before

        # A torn/foreign summary re-runs to a fresh terminal marker.
        (out / "summary.json").write_text("{}")
        tune_arm.run_trial(args)
        fresh = json.loads((out / "summary.json").read_text())
        assert fresh["status"] == "ok"
        assert fresh["fit_fingerprint"] == summary["fit_fingerprint"]

    def test_bidpo_trial_routes_production_pool_and_full_groups(
            self, tmp_path, tiny_llama, monkeypatch, capsys):
        """F1+F4+F5 end-to-end (trainer spied): the bidpo trial loads the
        own-pool files through the driver's sha-verified loader, routes
        through pairs_for_arm (reversed benign polarity preserved), trims
        to full optimizer groups, records counts + polarity + fingerprint,
        and skips only on a matching fingerprint."""
        import sisteer.extraction.bidpo as bidpo_mod
        from sisteer.vectors import SteeringVector

        # A small floor keeps the synthetic pool small; every formula
        # under test reads the module constant.
        monkeypatch.setattr(tune_arm, "EFFECTIVE_BATCH_FLOOR", 8)
        bp = tune_arm._pool_builder()
        cfg = {"model": TINY, "dtype": "float32", "layer": 1, "max_len": 96,
               "train_kwargs": {"default": {"holdout_frac": 0.5,
                                            "holdout_salt": "two_si_holdout"}},
               "pool": {"bidpo_benign": {"enabled": True,
                                         "refusal_labels": ["I"],
                                         "floor": 0,
                                         "max_benign_per_harmful": 1.0}},
               "wandb_project": "test"}
        _write_cfg(tmp_path, cfg)
        _write_pool(tmp_path, n=16)  # matched pool: probes/holdout only
        hpath = tmp_path / bp.bidpo_harmful_filename("french")
        bpath = tmp_path / bp.bidpo_benign_filename("french")
        hpath.write_text("\n".join(
            json.dumps({"prompt": f"harmful question {i}",
                        "pos": "Je refuse.",
                        "neg": f"compliant harmful answer {i}",
                        "instruction_id": "french",
                        "requirement_a": "strict"})
            for i in range(24)) + "\n")
        bpath.write_text("\n".join(
            json.dumps({"prompt": f"benign question {i}",
                        "pos": f"helpful compliance {i}",
                        "neg": "Je refuse.",
                        "instruction_id": "french",
                        "requirement_a": "benign_reversed"})
            for i in range(24)) + "\n")
        import hashlib as _hl
        (tmp_path / bp.PAIRS_META_NAME).write_text(json.dumps(
            {"bidpo_benign": {
                "refusal_labels": ["I"],
                "per_si": {"french": {"n_harmful": 24, "n_benign": 24}},
                "files": {"french": {
                    "harmful": {"file_sha256":
                                _hl.sha256(hpath.read_bytes()).hexdigest()},
                    "benign": {"file_sha256":
                               _hl.sha256(bpath.read_bytes()).hexdigest()},
                }}}}))
        _identified_payload(tmp_path, arm="bidpo")

        calls = {}

        def spy(model_, tok_, prefs, layer, **kw):
            calls["prefs"] = prefs
            calls["kw"] = kw
            return SteeringVector(
                vectors={1: torch.zeros(model_.config.hidden_size)},
                method="bidpo", model_name="", instruction_id="french")

        monkeypatch.setattr(bidpo_mod, "train_bidpo", spy)
        args = _trial_args(tmp_path)
        tune_arm.run_trial(args)

        out = tmp_path / "r1_t00"
        summary = json.loads((out / "summary.json").read_text())
        assert summary["status"] == "ok"
        # F1: routed through the production resolver, own pool, cap held.
        assert summary["bidpo_negative_source"] == \
            "anti_safety_compliance_per_si_thwart"
        assert summary["bidpo_own_pool"] is True
        assert summary["n_routed_harmful"] > 0
        assert 0 < summary["n_routed_benign_reversed"] <= \
            summary["n_routed_harmful"]
        assert (summary["n_fit_harmful"]
                + summary["n_fit_benign_reversed"]) == summary["n_fit"]
        # F5: every optimizer group full (bs 2 x accum ceil(8/2)=4 -> 8).
        assert summary["accum_steps"] == 4
        assert summary["effective_batch"] == 8
        assert summary["n_fit"] % 8 == 0
        assert summary["min_optimizer_group_rows"] == 8
        assert summary["optimizer_groups_full"] is True
        assert summary["n_trimmed_for_full_groups"] >= 0
        assert len(calls["prefs"]) == summary["n_fit"]
        assert calls["kw"]["accum_steps"] == 4
        # The REVERSED benign preference polarity survives conversion:
        # chosen = the verified compliance, rejected = the SI refusal.
        benign_prefs = [p for p in calls["prefs"]
                        if p.rejected == "Je refuse."]
        assert benign_prefs
        assert all("helpful compliance" in p.chosen for p in benign_prefs)
        assert summary["fit_fingerprint"]

        # F4: the re-run skips on the matching fingerprint...
        capsys.readouterr()
        tune_arm.run_trial(args)
        assert "skipping" in capsys.readouterr().out
        # ...and a swapped bidpo pool file changes the fingerprint, so the
        # trial re-runs — straight into the driver's sidecar sha refusal.
        with bpath.open("a") as f:
            f.write(json.dumps({"prompt": "late row", "pos": "x", "neg": "y",
                                "instruction_id": "french",
                                "requirement_a": "benign_reversed"}) + "\n")
        with pytest.raises(SystemExit, match="does not match the sha256"):
            tune_arm.run_trial(args)


class TestTrialSeed:
    """`trial --seed` (2026-09-15): the trainer seed a trial runs under,
    default 0 = every trial before it (the Modal driver passes the config's
    train_kwargs.default.seed only when one is declared). Recorded in
    summary.json as `seed`; part of the fit_fingerprint ONLY when non-zero,
    so every recorded fingerprint (all at seed 0) is unchanged while a
    seed-1 trial can never be skipped as the MATCHING fit of a seed-0
    summary of the same design."""

    def test_trial_seed_defaults_through_getattr(self):
        # older Namespace builders carry no `seed` at all: the historical 0
        assert tune_arm.trial_seed(argparse.Namespace()) == 0
        assert tune_arm.trial_seed(argparse.Namespace(seed=None)) == 0
        assert tune_arm.trial_seed(argparse.Namespace(seed=0)) == 0
        assert tune_arm.trial_seed(argparse.Namespace(seed=2)) == 2
        assert tune_arm.trial_seed(argparse.Namespace(seed="4")) == 4

    def test_cli_default_is_zero_and_the_flag_parses(self, monkeypatch):
        seen = []
        monkeypatch.setattr(tune_arm, "run_trial", seen.append)
        base = ["tune_arm.py", "trial", "--pairs", "p", "--trials", "t",
                "--index", "0", "--out", "o"]
        monkeypatch.setattr(sys, "argv", base)
        tune_arm.main()
        monkeypatch.setattr(sys, "argv", base + ["--seed", "3"])
        tune_arm.main()
        assert [a.seed for a in seen] == [0, 3]
        assert [tune_arm.trial_seed(a) for a in seen] == [0, 3]

    def test_run_trial_threads_the_seed_into_the_trainer_and_the_summary(self):
        src = TUNE_PATH.read_text()
        body = src.split("def run_trial(args)", 1)[1].split("\n# ---", 1)[0]
        assert "seed=trial_seed(args)" in body
        assert "seed=0, log_fn" not in body          # the old hardcoded 0
        assert '"seed": trial_seed(args)' in body     # summary.json

    def test_fingerprint_is_unchanged_at_seed_zero_and_binds_a_nonzero_seed(
            self, tmp_path):
        _write_cfg(tmp_path, {"model": TINY, "model_revision": "r1"})
        _write_pool(tmp_path)
        spec = _identified_payload(tmp_path, arm="lora_r1_all")
        cfg = json.loads((tmp_path / "cfg.yaml").read_text())
        ad = tune_arm.make_adapter("lora_r1_all")

        def fp(args):
            return tune_arm._trial_fingerprint(
                args, cfg, spec, spec["trials"][0], "lora_r1_all", "all", ad)

        legacy = _trial_args(tmp_path)
        del legacy.seed                                 # a pre-flag Namespace
        assert fp(legacy) == fp(_trial_args(tmp_path)) == \
            fp(_trial_args(tmp_path, seed=0))
        assert fp(_trial_args(tmp_path, seed=1)) != fp(_trial_args(tmp_path))
        assert fp(_trial_args(tmp_path, seed=1)) != \
            fp(_trial_args(tmp_path, seed=2))
        assert fp(_trial_args(tmp_path, seed=1)) == \
            fp(_trial_args(tmp_path, seed=1))

    def test_ce_adapter_seeds_its_lora_init_and_leaves_psr_alone(
            self, tiny_llama):
        """The CE LoRA adapter's make_object honours the trial seed exactly
        as the production driver does (methods._train_lora_ce): seed 1 vs 2
        -> different A on every layer, zero B; the default (and an explicit
        0) is the pre-flag object bit for bit; the PSR adapter's zero-init
        object is the same under any seed."""
        import types

        from sisteer.extraction.multilayer import MultiLayerLoRAObject

        model, _tok = tiny_llama
        setup = types.SimpleNamespace(
            model=model, cfg={"train_kwargs": {"lora_r4_ce": {"lora_rank": 4}}})

        def lora(**kw):
            return tune_arm.make_adapter("lora_r4_ce").make_object(setup, **kw)

        o1, o2, o0, od = lora(seed=1), lora(seed=2), lora(seed=0), lora()
        frozen = MultiLayerLoRAObject(model, rank=4)
        assert o1.layers == o2.layers == frozen.layers and o1.rank == 4
        for l in o1.layers:
            assert not torch.equal(o1.A[l], o2.A[l]), l
            assert torch.equal(o0.A[l], frozen.A[l]) and torch.equal(od.A[l], frozen.A[l])
            assert torch.equal(lora(seed=1).A[l], o1.A[l])
            assert not o1.B[l].any() and not o2.B[l].any()
        p1 = tune_arm.make_adapter("psr_ce_all").make_object(setup, seed=1)
        p2 = tune_arm.make_adapter("psr_ce_all").make_object(setup, seed=2)
        for l in p1.layers:
            assert torch.equal(p1.directions[l], p2.directions[l])
            assert torch.equal(p1.gate_w[l], p2.gate_w[l])
            assert torch.equal(p1.gate_b[l], p2.gate_b[l])

    def test_every_adapter_accepts_the_seed_and_run_trial_passes_it(self):
        import inspect

        for name in sorted(tune_arm.ADAPTERS):
            sig = inspect.signature(type(tune_arm.make_adapter(name)).make_object)
            assert "seed" in sig.parameters, name
            assert sig.parameters["seed"].default == 0, name
        src = TUNE_PATH.read_text()
        body = src.split("def run_trial(args)", 1)[1].split("\n# ---", 1)[0]
        assert "adapter.make_object(setup, lora_rank=hp.lora_rank, seed=trial_seed(args))" in body
