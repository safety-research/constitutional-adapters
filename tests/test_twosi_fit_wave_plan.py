"""The two-SI FIT WAVE, verified submittable without a GPU or a pool.

Pins everything the cluster B fit wave resolves BEFORE any job runs, so a
typo dies here and not 8 seconds into a cluster allocation:

- ``plan_si`` (pure CPU) resolves the shipped config's arm partition,
  per-SI max_len, and per-cell batch sizes for ALL THREE SIs — including
  stage 3's si_irrelevant, which the measured-budget table does not cover;
- the alphakl_soft manual-beta pin is present (its absence silently lost
  both alphakl_soft cells on the n3 run);
- every SI's instruction resolves (si_irrelevant is the newest and easiest
  to leave unregistered);
- the orchestrator's rendered cluster B scripts consume exactly the pairs
  filenames emit-pairs writes (one spelling, both read from
  build_two_si_pool);
- the checkout staging script keeps its safety properties: additive-only
  (no mirror-repos.sh, no push to the branch mirror — the EXPECTED_SHA gate
  killed jobs 23789/23790 when that mirror moved), refuses a dirty tree,
  and verifies the staged HEAD round-trip.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_two_si_pool as bp  # noqa: E402
import train_two_si_objects as tt  # noqa: E402


def _cfg() -> dict:
    return yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())


def test_plan_resolves_arms_and_batch_sizes_for_every_si():
    cfg = _cfg()
    arms = tt.default_class_a_arms(cfg)
    # the stage-3 Class A matrix: 14 arms, every one registered (2026-08-21:
    # alphakl_fisher pulled from the wave; vector + bidpo_lora, then
    # token_faithful, appended; 2026-08-28: the all-layer LoRA arm appended
    # as lora_r1_all and renamed lora_r4 the same day)
    assert sorted(arms) == sorted([
        "psr_kl_all", "psr_mse_all", "lora_r1", "loreft", "lofit",
        "token_bos", "alphakl_null", "alphakl_soft",
        "diffmean", "bidpo", "vector", "bidpo_lora", "token_faithful",
        "lora_r4"])
    assert cfg["si_ids"] == [
        "safety_refusal", "si_constitution", "si_irrelevant"]
    for si in cfg["si_ids"]:
        plan = tt.plan_si(cfg, si, arms)
        # per-SI max_len is the memory knob — exactly the config table
        assert plan["max_len"] == cfg["max_len_by_si"][si]
        # every arm lands somewhere, nowhere twice
        planned = list(plan["lockstep_group"]) + list(plan["solo"])
        assert sorted(planned) == sorted(arms)
        # every resolved width is a positive int (a None/0 would OOM-guard
        # or crash the trainer immediately)
        if plan["lockstep_group"]:
            assert plan["lockstep_batch_size"] >= 1
        for arm, bs in plan["solo"].items():
            assert bs >= 1, (si, arm)


def test_alphakl_soft_beta_stays_unpinned():
    """INVERTED on 2026-08-24 (this test used to pin beta == 1.0): the
    manual-beta raise this pin worked around is gone — train_alphakl now
    DERIVES the stage-2 beta against the eps benign budget when the key is
    absent, and the 1.0 pin was the uncalibrated step size the arm
    diverged under (wave-record floor_ratio 2.63/3.14 vs a no-op's 1.0 —
    against since-re-measured floors, so population-scoped). An absent
    beta is now the deliberate configuration; one reappearing here would
    silently re-pin the diverged value."""
    cfg = _cfg()
    arm_kw = cfg["train_kwargs"]["alphakl_soft"]
    assert "beta" not in arm_kw
    # alphakl faces the DEFAULT 1.0 bar and carries no per-arm exemption:
    # it minimises a KL to the same teacher, so the no-op floor means what
    # it means everywhere else. Pinned as an ABSENCE, because the exemption
    # it briefly had would have spared the very arm whose divergence
    # prompted the 1.0 ruling.
    assert "floor_ratio_max" not in arm_kw


def test_every_si_instruction_resolves():
    from sisteer.eval.jailbreak import resolve_instruction

    for si in _cfg()["si_ids"]:
        assert resolve_instruction(si)


def test_fit_scripts_consume_exactly_the_emitted_pairs_filenames():
    """emit-pairs writes pairs_<si>.jsonl (build_two_si_pool.pairs_filename);
    the orchestrator's rendered srun payload must name the same file — one
    spelling on both sides, asserted for every SI."""
    import two_si_overnight as ov

    cfg = _cfg()
    for si in cfg["si_ids"]:
        fname = bp.pairs_filename(si)
        assert fname == f"pairs_{si}.jsonl"
        text = ov.fit_sbatch_script(
            name=f"tt-twosi-fit-{si}-t1", si=si,
            pairs_remote=f"/workspace/TESTUSER/exp/jailbreak/two_si_pairs/{fname}",
            config="configs/two_si.yaml", time_limit="3-12:00:00",
            logdir="/workspace/TESTUSER/exp/jailbreak/_logs",
            store_dir="/workspace/TESTUSER/exp/jailbreak/two_si_store",
            objects_dir="/workspace/TESTUSER/exp/jailbreak/two_si_defenses_stage3/cells",
            sha="1" * 40, shared_root="/workspace/TESTUSER", qos="high-b")
        assert f"--pairs /workspace/TESTUSER/exp/jailbreak/two_si_pairs/{fname}" in text
        assert "#SBATCH --qos=high-b" in text
        assert f"--si {si}" in text


def test_staging_script_is_additive_and_guarded():
    text = (REPO / "scripts" / "stage_twosi_cluster_b_checkout.sh").read_text()
    # executable lines only — the comments deliberately NAME the tools the
    # script must never run, so the negative asserts skip them
    code = "\n".join(l for l in text.splitlines()
                     if l.strip() and not l.strip().startswith("#"))
    assert "set -euo pipefail" in code
    # additive only: never the mirror sync tool, never an --export, and the
    # one push goes to the NEW clone's staging ref
    assert "mirror-repos" not in code
    assert "--export" not in code
    assert '"$SHA:refs/heads/twosi-staged"' in code
    assert 'push -q "$HOST:$DEST"' in code
    # never a push whose target is the branch mirror
    assert 'push -q "$HOST:$MIRROR"' not in code
    assert "$MIRROR" in code          # the mirror is only ever a clone SOURCE
    # dirty-tree refusal + staged-HEAD round-trip verification
    assert "diff --quiet" in code
    assert 'if [ "$GOT" != "$SHA" ]' in code


def test_both_psr_arms_pin_the_measured_5e4_rate():
    """Neither psr arm may fall back to ``train_psr``'s 5e-3 default.

    Both diverged there and both were only rescued at 5e-4 -- psr_kl_all
    outright (job 195459), psr_mse_all in its RESPONSE cell alone (job
    193660 FAILED while the all-positions cell survived, which is why the
    rate looked like arm flakiness). An absent entry is the bug: the fix
    lived in a side config while the shipped one still reproduced the
    runaway.

    The asymmetry is deliberate and pinned as such. psr_kl_all carries
    schedule + clip; psr_mse_all's usable cells (197065_0/_1) recorded
    ``lr_schedule=None`` and ``max_grad_norm=None``, so a bare loop at 5e-4
    is what is measured for it. Copying the sibling's extras across would
    ship an unrun configuration.
    """
    tk = _cfg()["train_kwargs"]
    for arm in ("psr_kl_all", "psr_mse_all"):
        assert tk[arm]["lr"] == 5.0e-4, arm
    # ...and the RESOLVED rate, not just the file. psr_kl_all is a LOCKSTEP
    # arm, and lockstep params come from the registry with only an allow-list
    # of train_kwargs keys merged in. `lr` was missing from that list, so this
    # config read 5e-4 while the arm trained at the registry's 5e-3 -- the very
    # rate the pin exists to avoid, with nothing failing and no warning. A test
    # that reads the config file cannot see that; this one resolves the arm.
    assert tt.lockstep_arm_params(_cfg(), "psr_kl_all").lr == 5.0e-4
    assert tk["psr_kl_all"]["lr_schedule"] == "warmup_cosine"
    assert tk["psr_kl_all"]["max_grad_norm"] == 1.0
    # pinned as ABSENCES: the measured psr_mse recipe has neither
    assert "lr_schedule" not in tk["psr_mse_all"]
    assert "max_grad_norm" not in tk["psr_mse_all"]


def test_psr_arms_resolve_for_the_two_stage3_sis():
    """Both psr arms must plan on con and irr: psr_mse_all is
    lockstep-incompatible and has to land in ``solo`` with a real width, or
    the cell silently never runs."""
    cfg = _cfg()
    for si in ("si_irrelevant", "si_constitution"):
        plan = tt.plan_si(cfg, si, ["psr_kl_all", "psr_mse_all"])
        assert plan["max_len"] == 8192
        assert "psr_mse_all" in plan["solo"]
        assert plan["solo"]["psr_mse_all"] >= 1
        assert "psr_kl_all" in plan["lockstep_group"]
        assert plan["lockstep_batch_size"] >= 1


def test_wandb_run_name_identifies_the_cell_not_just_the_si(monkeypatch):
    """A wave runs several arms x both position modes against ONE SI, so a
    run named only for the SI leaves those runs distinguishable in the
    browser by job id alone -- which defeats the monitoring the run exists
    for. Name and config must both carry arm and positions."""
    captured = {}

    class _FakeWandb:
        class Settings:
            def __init__(self, **kw): pass

        @staticmethod
        def init(**kw):
            captured.update(kw)
            return object()

    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb)
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_PROJECT", "test-project")   # these configs name no wandb_project: $WANDB_PROJECT must resolve
    monkeypatch.setenv("SLURM_JOB_ID", "31822")

    assert tt.wandb_fit_run({}, "si_irrelevant", ["psr_kl_all"], "response")
    assert captured["name"] == "fit-si_irrelevant-psr_kl_all-response-31822"
    assert captured["config"]["positions"] == "response"
    assert captured["config"]["arms"] == ["psr_kl_all"]

    captured.clear()
    assert tt.wandb_fit_run({}, "si_constitution", ["psr_mse_all"], "all")
    assert captured["name"] == "fit-si_constitution-psr_mse_all-all-31822"
    assert captured["config"]["positions"] == "all"


def test_wandb_stays_opt_outable(monkeypatch):
    """The disable guards must survive the naming change -- a fit must never
    depend on W&B being reachable."""
    monkeypatch.setenv("WANDB_DISABLED", "true")
    assert tt.wandb_fit_run({}, "si_x", ["psr_kl_all"], "all") is None
    monkeypatch.setenv("WANDB_DISABLED", "")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert tt.wandb_fit_run({}, "si_x", ["psr_kl_all"], "all") is None
    monkeypatch.setenv("WANDB_MODE", "")
    assert tt.wandb_fit_run({"wandb_fits": False}, "si_x", ["a"], "all") is None


def test_lockstep_optimizer_keys_carry_the_rate():
    """`lr` must be mergeable into a lockstep arm's params. It governs the
    same private optimiser the schedule keys do, so it carries the same
    no-partner-perturbation argument -- and while it was excluded, a pinned
    rate was dropped in silence."""
    assert "lr" in tt._LOCKSTEP_OPTIMIZER_KEYS
    cfg = {"train_kwargs": {"psr_kl_all": {"lr": 1.25e-4,
                                           "lr_schedule": "constant"}}}
    p = tt.lockstep_arm_params(cfg, "psr_kl_all")
    assert p.lr == 1.25e-4
    assert p.lr_schedule == "constant"
    # absent key -> registry default, unchanged
    from sisteer.extraction.lockstep import default_lockstep_params
    assert (tt.lockstep_arm_params({}, "psr_kl_all").lr
            == default_lockstep_params("psr_kl_all").lr)


def test_banner_states_the_effective_rate_and_its_source():
    """The launch banner must name the rate an arm will TRAIN under and
    where it came from. A config echo would have read 5e-4 throughout the
    wave that trained at 5e-3; only a resolved value with a source shows
    the difference."""
    cfg = _cfg()
    lines = tt.optimizer_banner(cfg, "si_irrelevant",
                                ["psr_kl_all", "psr_mse_all"])
    kl = next(l for l in lines if "psr_kl_all" in l)
    assert "[lockstep]" in kl and "lr=0.0005 (config)" in kl
    # a rate NOT in the config must be reported as coming from the registry,
    # never silently printed as if configured
    bare = tt.effective_optimizer_settings({}, "psr_kl_all", lockstep=True)
    assert bare["lr"][1] == "registry default"
    assert bare["lr"][0] != 5.0e-4


def test_a_key_a_lockstep_arm_would_ignore_is_refused():
    """Configuration that reads as set and behaves as unset must fail at
    launch, not train silently."""
    import pytest

    cfg = {"train_kwargs": {"psr_kl_all": {"lr": 5e-4, "epochs": 9}}}
    with pytest.raises(SystemExit) as e:
        tt.refuse_silently_dropped_kwargs(cfg, ["psr_kl_all"])
    assert "epochs" in str(e.value)
    # the legitimate keys do not trip it
    ok = {"train_kwargs": {"psr_kl_all": {"lr": 5e-4, "max_grad_norm": 1.0,
                                          "floor_ratio_max": 1.0}}}
    tt.refuse_silently_dropped_kwargs(ok, ["psr_kl_all"])
    # and the shipped config must be clean
    tt.refuse_silently_dropped_kwargs(_cfg(), tt.default_class_a_arms(_cfg()))


def test_tokenize_progress_reports_and_stays_cheap(capsys):
    """The tokenize pass is the minutes-to-half-an-hour phase between model
    load and step 1, and it used to emit nothing — eight cells looked hung
    at once and were read as normal startup three times. It must narrate,
    and it must not narrate so much that the reporting becomes the cost."""
    from sisteer.extraction.distill import TokenizeProgress

    p = TokenizeProgress(20_000, label="probe", max_len=8192)
    for i in range(1, 20_001):
        p.tick(i)
    p.done()
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert lines[0].startswith("[tokenize] probe: 20000 examples at max_len 8192")
    assert "no token cache" in lines[0]
    # ~10 progress lines + header + footer, never one per example
    assert 5 <= len(lines) <= 15, len(lines)
    assert any("%" in l and "left" in l for l in lines)
    assert lines[-1].startswith("[tokenize] probe: 20000 examples in")


def test_tokenize_progress_is_silent_when_unlabelled():
    """`_encode_arm_batches` must stay quiet for tiny/unit-test encodes —
    the label is the opt-in."""
    import torch

    from sisteer.extraction.lockstep import _encode_arm_batches
    import inspect
    sig = inspect.signature(_encode_arm_batches)
    assert sig.parameters["label"].default is None


def test_fit_report_merges_arms_across_runs(tmp_path):
    """The per-SI report is an INVENTORY, not a transcript: one wave trains
    one arm per invocation, so a replacing writer makes every run erase the
    previous arm's entry. Job 31850 did exactly that to si_constitution's
    lora_r1 record, with no backup."""
    class _Arm:
        def __init__(self, r): self._r = r
        def as_dict(self): return {"status": "trained", "ratio": self._r}

    class _Outcome:
        def __init__(self, arms):
            self.si_id = "si_x"
            self.arms = arms
            self.failed = []
            self.skipped = []
            self.condition_sources = {}
            self.report_path = None

    tt._write_report(_Outcome({"lora_r1": _Arm(0.44)}), tmp_path)
    tt._write_report(_Outcome({"psr_mse_all": _Arm(0.10)}), tmp_path)

    rec = json.loads(tt.report_path_for(tmp_path, "si_x").read_text())
    assert sorted(rec["arms"]) == ["lora_r1", "psr_mse_all"], rec["arms"]
    assert rec["arms"]["lora_r1"]["ratio"] == 0.44      # survived run 2
    assert rec["arms"]["psr_mse_all"]["ratio"] == 0.10
    # a reader must be able to tell which entries this run did not write
    assert rec["arms_from_earlier_runs"] == ["lora_r1"]
    # re-running an arm overwrites its own entry rather than duplicating
    tt._write_report(_Outcome({"lora_r1": _Arm(0.41)}), tmp_path)
    rec = json.loads(tt.report_path_for(tmp_path, "si_x").read_text())
    assert rec["arms"]["lora_r1"]["ratio"] == 0.41
    assert sorted(rec["arms"]) == ["lora_r1", "psr_mse_all"]


def test_corrupt_prior_report_does_not_cost_this_runs_record(tmp_path, capsys):
    """An unreadable prior file must be reported and stepped over, never
    silently swallow the arms this run just trained."""
    class _Arm:
        def as_dict(self): return {"status": "trained"}

    class _Outcome:
        si_id = "si_x"; failed = []; skipped = []
        condition_sources = {}; report_path = None
        arms = {"psr_kl_all": _Arm()}

    tt.report_path_for(tmp_path, "si_x").write_text("{not json")
    tt._write_report(_Outcome(), tmp_path)
    rec = json.loads(tt.report_path_for(tmp_path, "si_x").read_text())
    assert sorted(rec["arms"]) == ["psr_kl_all"]
    assert "could not merge prior report" in capsys.readouterr().err


def test_goal_level_holdout_regime_is_configured_and_guarded():
    """Family-level holdout is OFF by user directive: the five headroom
    families are trained on AND evaluated on, held out at the goal level.
    Pinned because the flag silently changes what 'held out' means."""
    pool = _cfg()["pool"]
    assert pool["family_disjointness"] is False
    five = {"bon", "pasttense", "codeattack", "renellm", "grok420"}
    assert five <= set(pool["train_families"]), five - set(pool["train_families"])
    assert five <= set(pool["eval_families"]), five - set(pool["eval_families"])
    # a family can only enter the pool at all if static_families gates it in
    assert five <= set(pool["static_families"])


def test_threshold_is_the_preregistered_strict_half():
    """0.5, pinned explicitly rather than inherited. The 0.75 directive it
    supersedes made pasttense and renellm look infeasible (churn-over-target
    1.40 / 1.12) purely by shrinking the baseline the relative target scales
    with."""
    assert _cfg()["pool"]["threshold"] == 0.5
    assert _cfg()["prereg"]["primary_statistic"].startswith(
        "goal_any_strongreject_gt_0p5")


def test_empty_goal_split_is_refused_when_family_holdout_is_off():
    """With families on both sides the goal split is the ONLY holdout, so an
    empty eval goal set is not a thin holdout -- it is none, and every
    'held-out' number downstream would be in-sample while still labelled
    held out."""
    import pytest

    from sisteer.eval.matched_pool import (MatchedPoolError, PoolSpec,
                                            build_matched_pool)

    spec = PoolSpec(
        si_ids=("si_a", "si_b"), train_families=("bon",), eval_families=("bon",),
        train_goal_ids=("g1",), eval_goal_ids=(),
        static_families=frozenset({"bon"}))
    with pytest.raises(MatchedPoolError, match="nothing is held out"):
        build_matched_pool([], spec)
    # (the strict family-disjointness regime is RETIRED — user ruling
    # 2026-08-28, applied at the phase-C join: shared families never refuse;
    # the goal split is the only holdout, and the refusal above is what
    # keeps it load-bearing)


def test_a_silently_thin_family_is_refused_not_merely_counted():
    """"Every family present with a non-zero count" passes identically for a
    family covering 3 goals and one covering 300. A thin family LOOKS LIKE
    DATA; an absent one is visible. renellm is the live case -- frozen-corpus,
    so it reaches only the goal_ids its pinned corpus contains."""
    # Assert the MECHANISM, not the shipped constant. An earlier version of
    # this test pinned the config to 0.5 -- which is the weak form: it stays
    # green whether or not anything reads the value, and it turns a
    # deliberate, documented change into a test failure that says nothing
    # about behaviour. What must hold is that a thin family is REFUSED.
    cfg_floor = _cfg()["pool"]["min_family_goal_coverage"]
    assert 0.0 < cfg_floor <= 0.5, (
        f"floor {cfg_floor} is off or looser than the original 0.5; if that "
        f"is deliberate the config must say why (grok420 at 11.3% is the "
        f"2026-08-26 case), but it may never be 0 or absent")

    from sisteer.eval.matched_pool import PoolSpec

    spec = PoolSpec(
        si_ids=("si_a", "si_b"), train_families=("bon", "renellm"),
        eval_families=("bon",), train_goal_ids=("g1",), eval_goal_ids=("g2",),
        static_families=frozenset({"bon", "renellm"}),
        min_family_goal_coverage=0.5)
    assert spec.min_family_goal_coverage == 0.5
    # the knob defaults OFF, so callers that never set it keep the old
    # presence-only behaviour rather than silently gaining a new refusal
    bare = PoolSpec(
        si_ids=("si_a", "si_b"), train_families=("bon",),
        eval_families=("pair",), train_goal_ids=("g1",), eval_goal_ids=("g2",),
        static_families=frozenset({"bon"}))
    assert bare.min_family_goal_coverage == 0.0


# --------------------------------------------------------------------------- #
# T10 (twosi-model-sweeps): the ported in-job scheduler — packing,             #
# cheap-first barrier, width-1 env stripping — on THIS branch (arms pack       #
# 1-per-GPU; extraction.ddp row-sharding deliberately not ported).             #
# --------------------------------------------------------------------------- #


def _dp_cfg(gpus: int, **extra) -> dict:
    return {"data_parallel": {"gpus": gpus, **extra},
            "train_kwargs": {"default": {}}}


def test_plan_parallel_packs_cheap_first_then_longest():
    cfg = _dp_cfg(4)
    arms = ["psr_mse_all", "diffmean", "bidpo", "alphakl_null", "psr_kl_all"]
    plan = tt.plan_parallel(cfg, "si_constitution", arms)
    assert plan["gpus"] == 4
    by_arm = {t["arm"]: t for t in plan["tasks"]}
    # every width is 1 on this branch
    assert {t["width"] for t in plan["tasks"]} == {1}
    # the closed-form arms are the cheap-first batch...
    assert by_arm["diffmean"]["phase"] == "cheap-first"
    assert by_arm["alphakl_null"]["phase"] == "cheap-first"
    assert by_arm["psr_mse_all"]["phase"] == "main"
    # ...and start at t=0, ahead of everything
    assert by_arm["diffmean"]["start"] == 0.0
    assert by_arm["alphakl_null"]["start"] == 0.0
    # the main batch is longest-first (psr_mse 9 > bidpo 8 > psr_kl 7)
    main_starts = [(t["arm"], t["start"]) for t in plan["tasks"]
                   if t["phase"] == "main"]
    order = [a for a, _s in sorted(main_starts, key=lambda x: x[1])]
    assert order.index("psr_mse_all") <= order.index("bidpo") \
        <= order.index("psr_kl_all")


def test_cheap_first_barrier_holds_on_a_narrow_pool():
    """No main task may start while ANY cheap task is pending — a GPU
    idling for the minutes a cheap arm needs beats a cut wave missing its
    cheapest rows (the 2026-08-22 diffmean loss)."""
    cfg = _dp_cfg(1)
    plan = tt.plan_parallel(cfg, "si_constitution",
                            ["psr_mse_all", "diffmean", "alphakl_null"])
    last_cheap = max(t["start"] for t in plan["tasks"]
                     if t["phase"] == "cheap-first")
    first_main = min(t["start"] for t in plan["tasks"]
                     if t["phase"] == "main")
    assert first_main >= last_cheap
    # 0 disables the phase entirely (pure longest-first)
    off = tt.plan_parallel(_dp_cfg(1, cheap_first_below=0),
                           "si_constitution", ["psr_mse_all", "diffmean"])
    assert {t["phase"] for t in off["tasks"]} == {"main"}


def test_arm_width_pins_one_gpu_and_refuses_wider():
    import pytest

    assert tt.arm_width(_dp_cfg(8), "si_constitution", "psr_kl_all", 8) == 1
    assert tt.arm_width(_dp_cfg(8, arm_gpus={"bidpo": 1}),
                        "si_constitution", "bidpo", 8) == 1
    with pytest.raises(SystemExit, match="1-per-GPU"):
        tt.arm_width(_dp_cfg(8, arm_gpus={"psr_kl_all": 4}),
                     "si_constitution", "psr_kl_all", 8)


def test_spawn_arm_task_strips_distributed_env_and_subsets_gpus(
        tmp_path, monkeypatch):
    """Width 1 (this branch's only width) must run the trainers'
    historical single-process path byte-identically: EVERY distributed
    env variable stripped, CUDA_VISIBLE_DEVICES a strict SUBSET of the
    job's pool (cluster hard rule 3)."""
    import argparse
    import subprocess

    captured = {}

    class _P:
        def __init__(self, cmd, stdout=None, stderr=None, env=None, cwd=None):
            captured["cmd"], captured["env"] = cmd, env

        def poll(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", _P)
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR",
                "MASTER_PORT"):
        monkeypatch.setenv(var, "9")  # a stale launcher env must not leak
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    args = argparse.Namespace(
        config="configs/two_si.yaml", si="si_constitution",
        pairs="pool.jsonl", out=str(tmp_path / "out"), overwrite=False,
        positions="", token_cache="", bidpo_harmful_pairs="",
        bidpo_benign_pairs="", expect_config_sha="")
    ap = tt._spawn_arm_task(args, "psr_kl_all", 1, ["5"], tmp_path)
    env = captured["env"]
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR",
                "MASTER_PORT"):
        assert var not in env, f"{var} leaked into a width-1 arm task"
    assert env["CUDA_VISIBLE_DEVICES"] == "5"  # strict subset, never widened
    assert "arm-task" in captured["cmd"]
    assert "--expected-width" in captured["cmd"]
    # the config byte-identity pin is ALWAYS forwarded (stale-config guard)
    assert "--expect-config-sha" in captured["cmd"]
    assert ap.poll() == 0


def test_wandb_run_name_carries_the_model_slug(monkeypatch):
    """twosi-model-sweeps: two models fit concurrently, so the run name is
    fit-{model_slug}-{si}-{arms}[-{positions}][-{job}] and the run config
    carries model + model_revision (the cross-model grouping fields)."""
    captured = {}

    class _FakeWandb:
        class Settings:
            def __init__(self, **kw): pass

        @staticmethod
        def init(**kw):
            captured.update(kw)
            return object()

    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb)
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_PROJECT", "test-project")   # these configs name no wandb_project: $WANDB_PROJECT must resolve
    monkeypatch.setenv("SLURM_JOB_ID", "31822")
    cfg = {"model": "google/gemma-4-E4B-it",
           "model_revision": "ee0ef6023621cff5"}
    assert tt.wandb_fit_run(cfg, "si_constitution", ["psr_kl_all"],
                            "response")
    assert captured["name"] == \
        "fit-gemma-4-E4B-it-si_constitution-psr_kl_all-response-31822"
    assert captured["config"]["model"] == "google/gemma-4-E4B-it"
    assert captured["config"]["model_revision"] == "ee0ef6023621cff5"
    # a config without a model keeps the historical name shape (old tests,
    # the legacy Llama config): the slug segment is simply absent
    captured.clear()
    assert tt.wandb_fit_run({}, "si_x", ["bidpo"], "all")
    assert captured["name"] == "fit-si_x-bidpo-all-31822"
