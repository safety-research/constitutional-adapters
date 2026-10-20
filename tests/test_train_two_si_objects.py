"""One-model-load-per-(SI) training via lockstep — the
``scripts/train_two_si_objects.py`` entry point.

CPU-only: the tiny_llama fixture and synthetic French pairs. Verifies the
partition (lockstep group vs solo), that a batched multi-arm train produces the
SAME objects as training each arm on its own path (so the one-load driver is a
pure efficiency change), and that objects are saved to the cell-id paths
``two_si_run generate`` loads.

Also the two defects of fit jobs 148862/148863 (both SIs died on the
condition-prompt guard with ZERO objects written): a condition-taking arm
must receive the category condition prompts from the config-resolved content
contrast, each object must land on disk the instant its arm completes, one
arm's failure must be recorded and contained (never fatal to the run), a
re-run must skip already-saved arms (with a partially-done lockstep group
dropping its finished members at the FULL group's batch size,
bitwise-identically), and the over-refusal leakage guard must still raise
through the driver's config path.

cast itself left methods.ARMS ENTIRELY on 2026-08-20, so
the condition-seam tests inject a SYNTHETIC condition-taking arm
(``condarm``, whose trainer IS extraction.cast.train_cast — the historical
consumer's exact signature and metadata) into the registry per test. The
seam is generic and signature-introspected; these tests keep it honest for
the next consumer.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_two_si_objects as t2

from sisteer.data.pairs import Pair
from sisteer.eval.defense_cell import object_path_for, sha256_file
from sisteer.eval.jailbreak import resolve_instruction
from sisteer.extraction.cast import train_cast
from sisteer.extraction.distill import PROBE_ESTIMATE_RECORD_KEYS
from sisteer.methods import ARMS, ArmSpec, S_TWO_SI

SI_ID = "safety_refusal"

# A config slice sized for the tiny model + synthetic pairs (no store data).
CFG = {
    "layer": 1,
    "completion_tokens": 16,
    "max_len": 256,
    "max_len_by_si": {"safety_refusal": 256},
    "train_kwargs": {"default": {"epochs": 2, "seed": 0}},
    "si_ids": ["safety_refusal", "si_constitution"],
}


def _pairs():
    data = [
        ("Say something nice.", "Bonjour le monde", "Hello world"),
        ("Describe a tree.", "C'est un grand arbre", "It is a tall tree"),
        ("Greet the reader.", "Salut a tous", "Hi everyone"),
    ]
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id="safety_refusal")
            for p, pos, neg in data]


def _sentinel(tok):
    return (tok.unk_token, tok.unk_token_id)


def _fp(arm, cfg=None, pairs=None, positions="all"):
    """The fit fingerprint a train_si_objects call with these inputs both
    STORES (COMPLETE marker) and REQUIRES at resume (F15) — the
    programmatic-caller derivation: pool content hash + cfg-dict hash."""
    cfg = CFG if cfg is None else cfg
    pairs = _pairs() if pairs is None else pairs
    return t2.cell_fit_fingerprint(
        cfg, SI_ID, arm, positions=positions,
        pool_sha256=t2.pool_content_sha256(pairs),
        config_sha=t2._cfg_dict_sha(cfg))


def _inject_condarm(monkeypatch):
    """Register the synthetic condition-taking arm for one test: train_cast
    behind a fresh name, S_TWO_SI-suite so it can never leak into a core
    suite. monkeypatch.setitem restores the registry afterwards."""
    monkeypatch.setitem(
        ARMS, "condarm",
        ArmSpec("condarm", lambda model, tok, layer, **kw: None,
                train_cast, frozenset({S_TWO_SI})))


def test_partition_and_plan():
    arms = ["psr_kl_all", "psr_mse_all", "lora_r1", "loreft", "token_bos"]
    compatible, solo = t2.partition_arms(arms)
    assert compatible == ["psr_kl_all", "lora_r1", "token_bos"]
    assert solo == ["psr_mse_all", "loreft"]

    plan = t2.plan_si(CFG, SI_ID, arms)
    assert plan["si_id"] == SI_ID
    assert plan["max_len"] == 256
    assert plan["lockstep_group"] == compatible
    assert plan["lockstep_batch_size"] >= 1
    assert set(plan["solo"]) == set(solo)
    assert plan["condition_arms"] == []  # none of these take condition prompts


def test_plan_names_condition_arms_up_front(monkeypatch):
    # a condition-taking arm declares condition_pos/neg_prompts, so the plan
    # names it up front (signature-introspected, never a name list)
    _inject_condarm(monkeypatch)
    arms = ["psr_kl_all", "condarm"]
    plan = t2.plan_si(CFG, SI_ID, arms)
    assert plan["condition_arms"] == ["condarm"]


def test_default_class_a_arms_are_registered():
    arms = t2.default_class_a_arms(CFG | {"class_a_arms": None})
    # falls back to the module CLASS_A_ARMS, filtered to the registry
    from sisteer.eval.defense_cell import CLASS_A_ARMS
    assert arms == [a for a in CLASS_A_ARMS if a in ARMS]
    assert arms  # non-empty


def test_train_si_objects_matches_per_arm_paths_bitwise(tiny_llama):
    """Batched multi-arm training (lockstep group + solo) yields bitwise the
    same objects as training each arm individually — the one-load driver is a
    pure efficiency change, not a behaviour change.

    Compatible arms are covered against the registry trainer by test_lockstep;
    here we additionally confirm the SOLO arm (loreft) is identical whether
    trained through this driver or its registry trainer directly, at the per-SI
    max_len the driver resolves.

    The driver LENGTH-BUCKETS the pool by default (the production
    duty-cycle fix, 2026-08-25 — recorded per cell as ``length_bucketed``),
    so the reference call receives the SAME bucketed order: the equality
    under test is driver-vs-registry, not sorted-vs-unsorted (batch
    composition is an input, and the sidecar records which one ran)."""
    from sisteer.data.pairs import pairs_to_distill_examples
    from sisteer.extraction.fitmon import length_sorted

    model, tok = tiny_llama
    pairs = _pairs()
    arms = ["lora_r1", "psr_kl_all", "loreft"]

    outcome = t2.train_si_objects(model, tok, SI_ID, pairs, arms, cfg=CFG,
                                  sentinel=_sentinel(tok))
    assert outcome.ok
    objs = outcome.objects
    assert set(objs) == set(arms)

    # loreft is the solo arm: reproduce its registry call with the driver's
    # resolved kwargs — and the driver's bucketed pair order — and assert
    # bitwise identical trained params.
    instr = resolve_instruction(SI_ID)
    exs = pairs_to_distill_examples(tok, pairs, instr)
    pos = {id(e): i for i, e in enumerate(exs)}
    bucketed = [pairs[pos[id(e)]] for e in length_sorted(tok, exs)]
    from sisteer.eval.train_budget import resolve_batch_size, resolve_max_len
    bs = resolve_batch_size(CFG, SI_ID, "loreft")
    ml = resolve_max_len(CFG, SI_ID)
    ref = ARMS["loreft"].train(model, tok, instr, bucketed,
                               layer=1, epochs=2, batch_size=bs, max_len=ml,
                               seed=0)
    driver = objs["loreft"]
    for i, (a, b) in enumerate(zip(driver.trainable, ref.trainable)):
        assert torch.equal(a.detach(), b.detach()), f"loreft param {i} differs"


def test_train_si_objects_saves_to_cell_id_paths(tiny_llama, tmp_path):
    model, tok = tiny_llama
    arms = ["lora_r1", "loreft"]
    outcome = t2.train_si_objects(model, tok, SI_ID, _pairs(), arms, cfg=CFG,
                                  save_dir=tmp_path, sentinel=_sentinel(tok))
    assert outcome.ok

    for arm in arms:
        cell_id = f"{SI_ID}__{arm}"
        path = object_path_for(tmp_path, cell_id)
        assert path.exists(), f"{arm}: object not saved at {path}"
        meta = json.loads((path.parent / "train_budget_meta.json").read_text())
        assert meta["cell_id"] == cell_id
        assert meta["si_id"] == SI_ID
        assert meta["completion_tokens"] == 16  # the shared cap, recorded
        assert meta["max_len"] == 256
    # the compatible arm records a shared teacher forward, the solo one does not
    ls = json.loads((object_path_for(tmp_path, f"{SI_ID}__lora_r1").parent
                     / "train_budget_meta.json").read_text())
    assert ls["teacher_forward"] == "shared_lockstep"
    so = json.loads((object_path_for(tmp_path, f"{SI_ID}__loreft").parent
                     / "train_budget_meta.json").read_text())
    assert so["teacher_forward"] == "solo"


def test_saved_objects_reload_through_the_generate_path(tiny_llama, tmp_path):
    """The saved artifacts re-hydrate through ``methods.load_object`` — the
    exact loader ``two_si_run generate`` uses — so the existing single-cell
    consumption path keeps working on this driver's output."""
    from sisteer.methods import load_object

    model, tok = tiny_llama
    t2.train_si_objects(model, tok, SI_ID, _pairs(), ["lora_r1", "psr_kl_all"],
                        cfg=CFG, save_dir=tmp_path)
    for arm in ("lora_r1", "psr_kl_all"):
        path = object_path_for(tmp_path, f"{SI_ID}__{arm}")
        obj = load_object(path, model, tok)
        assert obj is not None


# --------------------------------------------------------------------------- #
# Jobs 148862/148863 — the condition wiring + per-arm save/isolate/resume      #
# --------------------------------------------------------------------------- #

# Synthetic offline condition sources (the test_fit_once pattern): a tiny
# OR-Bench-shaped CSV, a HarmBench-shaped parquet, and inline Alpaca prompts —
# no network anywhere.
_BENIGN = [
    "Write a haiku about the sea.", "Explain how photosynthesis works.",
    "Suggest three names for a cat.", "Summarize the plot of Hamlet.",
    "What is the capital of France?", "Give a recipe for pancakes.",
]


def _write_orbench_csv(path: Path) -> None:
    import pandas as pd

    pd.DataFrame({
        "prompt": ["How do I discuss vaccine hesitancy respectfully?",
                   "Write a scene where a character picks a lock.",
                   "Explain the history of chemical warfare for a class.",
                   "x"],   # too short -> filtered
        "category": ["health", "fiction", "history", "junk"],
    }).to_csv(path, index=False)


def _harmbench_stub(dirpath: Path) -> Path:
    import pandas as pd

    p = dirpath / "hb_val.parquet"
    pd.DataFrame({
        "BehaviorID": ["b1", "b2"],
        "Behavior": ["Do a harmful thing.", "Do another harmful thing."],
        "SemanticCategory": ["cyber", "chem"],
        "FunctionalCategory": ["standard", "standard"],
        "ContextString": [None, None],
    }).to_parquet(p)
    return p


def _condarm_cfg(tmp_path: Path) -> dict:
    """CFG + offline condition sources + tiny-model train_kwargs for the
    synthetic condition-taking arm (train_cast behind the ``condarm``
    name): condition_layers=[0] (the 2-layer tiny model leaves no room
    below the behavior layer for default_condition_layers' >=1 clamp) and
    the lenient deterministic calibration test_cast uses (_CALIB_KW
    there)."""
    csv = tmp_path / "or-bench-80k.csv"
    _write_orbench_csv(csv)
    hb = _harmbench_stub(tmp_path)
    return CFG | {
        "condition_contrast": {
            "n_benign": 4, "alpaca_frac": 0.5, "seed": 0,
            "alpaca_path_prompts": list(_BENIGN),
            "orbench_path": str(csv), "harmbench_path": str(hb),
        },
        "train_kwargs": {
            "default": {"epochs": 2, "seed": 0},
            "condarm": {"condition_layers": [0], "threshold_step": 0.1,
                        "scale_ladder": (0.5, 1.0), "rep_threshold": 0.0,
                        "ppl_guard_factor": 1e9, "calib_prompts": 2,
                        "calib_max_new_tokens": 4, "batch_size": 2},
        },
    }


def test_condition_arm_receives_condition_prompts_and_trains(
        tiny_llama, tmp_path, monkeypatch):
    """Defect 1 of jobs 148862/148863, kept honest through the generic seam:
    a condition-taking Class A arm must be handed the category-based
    condition (harmful = should-fire pos, benign = should-not-fire neg)
    resolved from the config's condition_contrast block, introspected from
    the trainer's signature (cast — the historical consumer — left the
    registry 2026-08-20; ``condarm`` is its trainer behind a test name)."""
    model, tok = tiny_llama
    _inject_condarm(monkeypatch)
    cfg = _condarm_cfg(tmp_path)
    out_dir = tmp_path / "defenses"
    outcome = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["condarm"],
                                  cfg=cfg, save_dir=out_dir)
    assert outcome.ok and set(outcome.objects) == {"condarm"}
    obj = outcome.objects["condarm"]
    assert obj.metadata["fit_path"] == "explicit_kwargs"  # never the fallback
    # the pairwise condition PCA needs equal sides: 2 harmful (harmbench
    # stub) caps the 4 benign to 2 — deterministically, heads of sha-sorted
    # lists — and the equalization is recorded in the sidecar.
    assert obj.metadata["n_condition_pos"] == 2
    assert obj.metadata["n_condition_neg"] == 2

    # saved at its cell-id path, with the sources recorded so a reader can tell
    path = object_path_for(out_dir, f"{SI_ID}__condarm")
    assert path.exists()
    meta = json.loads((path.parent / "train_budget_meta.json").read_text())
    assert meta["condition_sources"]["harmful_source"] == "harmbench:val"
    assert meta["condition_sources"]["benign_sources"] == {
        "alpaca": 2, "or-bench-80k": 2}
    assert meta["condition_sources"]["n_condition_per_side"] == 2
    assert meta["condition_sources"]["equalized_from"] == {
        "harmful": 2, "benign": 4}
    report = json.loads(t2.report_path_for(out_dir, SI_ID).read_text())
    assert report["arms"]["condarm"]["status"] == "trained"
    assert report["condition_sources"]["n_benign"] == 4


def test_condition_contrast_resolved_once_per_process(tmp_path):
    """The contrast is shared across arms and SIs: same config -> the SAME
    cached object, never rebuilt per arm."""
    cfg = _condarm_cfg(tmp_path)
    c1 = t2.resolve_condition_contrast(cfg)
    c2 = t2.resolve_condition_contrast(cfg)
    assert c1 is c2
    assert len(c1.harmful) == 2 and len(c1.benign) == 4


def test_leakage_guard_reaches_through_the_driver(tiny_llama, tmp_path,
                                                  monkeypatch):
    """No bypass: naming an over-refusal EVAL CONTROL as the benign source
    still raises out of build_content_contrast — and through the driver it
    fails ONLY the condition-taking arm, never the run."""
    model, tok = tiny_llama
    _inject_condarm(monkeypatch)
    cfg = _condarm_cfg(tmp_path)
    cfg["condition_contrast"] = dict(cfg["condition_contrast"],
                                     orbench_config="or-bench-hard-1k")
    with pytest.raises(ValueError, match="EVAL"):
        t2.resolve_condition_contrast(cfg)

    out_dir = tmp_path / "defenses"
    outcome = t2.train_si_objects(model, tok, SI_ID, _pairs(),
                                  ["lora_r1", "condarm"], cfg=cfg,
                                  save_dir=out_dir)
    assert set(outcome.failed) == {"condarm"}
    assert "EVAL" in outcome.failed["condarm"].error
    # the other arm's work is on disk regardless
    assert object_path_for(out_dir, f"{SI_ID}__lora_r1").exists()


def test_jobs_148862_148863_one_arm_refusal_never_zeroes_the_run(
        tiny_llama, tmp_path, monkeypatch):
    """Regression for fit jobs 148862 (safety_refusal) / 148863
    (si_constitution): cast's condition-guard ValueError propagated out of
    _safe_registry_train (which swallowed only TypeError) and killed the
    process AFTER the lockstep group had trained but BEFORE anything was
    saved — 15:00 / 34:45 of training, zero objects, defenses_dir never
    created. Reproduce the exact failure (the condition-taking trainer run
    WITHOUT condition prompts -> the guard raises) by forcing the
    introspection off; the refusal must be recorded as a failed arm, the
    run must continue, and every other arm's object must already be on
    disk. (cast itself left the registry 2026-08-20; ``condarm`` carries
    its exact trainer and guard.)"""
    model, tok = tiny_llama
    _inject_condarm(monkeypatch)
    monkeypatch.setattr(t2, "_accepts_condition_prompts", lambda fn: False)
    cfg = _condarm_cfg(tmp_path)   # sources present but unused once forced off
    out_dir = tmp_path / "defenses"
    outcome = t2.train_si_objects(
        model, tok, SI_ID, _pairs(), ["lora_r1", "psr_kl_all", "condarm"],
        cfg=cfg, save_dir=out_dir, sentinel=_sentinel(tok))
    # the guard fired exactly as in the jobs...
    assert set(outcome.failed) == {"condarm"}
    assert outcome.failed["condarm"].error_type == "ValueError"
    assert "condition_pos_prompts" in outcome.failed["condarm"].error
    # ...but the lockstep arms' work survived, on disk
    for arm in ("lora_r1", "psr_kl_all"):
        assert object_path_for(out_dir, f"{SI_ID}__{arm}").exists()
    report = json.loads(t2.report_path_for(out_dir, SI_ID).read_text())
    assert report["failed"] == ["condarm"]
    assert report["arms"]["lora_r1"]["status"] == "trained"


def test_objects_land_on_disk_before_the_next_arm_starts(
        tiny_llama, tmp_path, monkeypatch):
    """Defect 2's save-at-end: each object must be on disk the instant its
    arm completes, BEFORE the next arm even starts — proven by having the
    next arm's trainer look for it (and then blow up, which must also be
    contained)."""
    model, tok = tiny_llama
    out_dir = tmp_path / "defenses"
    seen: list[bool] = []

    def probe_train(model_, tok_, instruction, pairs, *, layer,
                    log_fn=None, **kw):
        seen.append(object_path_for(out_dir, f"{SI_ID}__loreft").exists())
        raise RuntimeError("boom after probing")

    monkeypatch.setitem(ARMS, "psr_mse_all",
                        replace(ARMS["psr_mse_all"], train=probe_train))
    outcome = t2.train_si_objects(model, tok, SI_ID, _pairs(),
                                  ["loreft", "psr_mse_all"], cfg=CFG,
                                  save_dir=out_dir)
    assert seen == [True]   # loreft was ALREADY saved when psr_mse_all started
    assert set(outcome.failed) == {"psr_mse_all"}
    assert outcome.failed["psr_mse_all"].error_type == "RuntimeError"
    assert set(outcome.objects) == {"loreft"}


def test_cli_train_exits_nonzero_after_saving_good_arms(
        tiny_llama, tmp_path, monkeypatch):
    """A failed arm exits the JOB non-zero (so the orchestrator sees it) —
    but only after every other arm was attempted and saved."""
    import sisteer.models

    model, tok = tiny_llama

    def boom(*a, **kw):
        raise RuntimeError("deliberate")

    monkeypatch.setitem(ARMS, "loreft", replace(ARMS["loreft"], train=boom))
    monkeypatch.setattr(sisteer.models, "load_model_and_tokenizer",
                        lambda *a, **kw: (model, tok))
    pairs_file = tmp_path / "pairs.jsonl"
    pairs_file.write_text("\n".join(json.dumps(
        {"prompt": p.prompt, "pos": p.pos, "neg": p.neg,
         "instruction_id": p.instruction_id}) for p in _pairs()))
    out_dir = tmp_path / "defenses"
    args = argparse.Namespace(si=SI_ID, arms="lora_r1,loreft",
                              pairs=str(pairs_file), out=str(out_dir),
                              overwrite=False)
    with pytest.raises(SystemExit) as ei:
        t2.cmd_train(args, CFG | {"model": "unused", "dtype": "float32"})
    assert ei.value.code == 1
    assert object_path_for(out_dir, f"{SI_ID}__lora_r1").exists()
    report = json.loads(t2.report_path_for(out_dir, SI_ID).read_text())
    assert report["failed"] == ["loreft"]


def test_rerun_skips_already_saved_arms(tiny_llama, tmp_path, monkeypatch):
    """Resume: a second run over the same save_dir touches NO trainer — every
    finished arm is skipped from its on-disk object."""
    import sisteer.extraction.lockstep as lockstep_mod

    model, tok = tiny_llama
    out_dir = tmp_path / "defenses"
    arms = ["lora_r1", "loreft"]
    first = t2.train_si_objects(model, tok, SI_ID, _pairs(), arms, cfg=CFG,
                                save_dir=out_dir)
    assert first.ok and set(first.objects) == set(arms)

    def no_train(*a, **kw):
        raise AssertionError("resume must not retrain a finished arm")

    monkeypatch.setattr(lockstep_mod, "train_lockstep", no_train)
    monkeypatch.setitem(ARMS, "loreft", replace(ARMS["loreft"], train=no_train))
    second = t2.train_si_objects(model, tok, SI_ID, _pairs(), arms, cfg=CFG,
                                 save_dir=out_dir)
    assert second.ok
    assert second.objects == {}
    assert sorted(second.skipped) == sorted(arms)
    report = json.loads(t2.report_path_for(out_dir, SI_ID).read_text())
    assert report["skipped_existing"] and not report["failed"]


def test_lockstep_resume_drops_finished_members_and_stays_bitwise(
        tiny_llama, tmp_path):
    """The chosen resume semantics for a partially-done lockstep group: DROP
    the finished members and train only the remainder — at the batch size of
    the FULL requested group, never recomputed over the survivors. Verified
    bitwise: psr_kl_all resumed next to an already-done lora_r1 equals
    psr_kl_all from a from-scratch full-group run. (batch_size_by_arm is
    asymmetric on purpose: recomputing the min over the survivors would give
    2, not the full group's 1, and the objects would differ.)"""
    model, tok = tiny_llama
    cfg = CFG | {"batch_size_by_arm": {"lora_r1": 1, "psr_kl_all": 2}}
    arms = ["lora_r1", "psr_kl_all"]

    ref = t2.train_si_objects(model, tok, SI_ID, _pairs(), arms, cfg=cfg,
                              save_dir=tmp_path / "ref")
    assert ref.ok

    res_dir = tmp_path / "resume"
    partial = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["lora_r1"],
                                  cfg=cfg, save_dir=res_dir)
    assert partial.ok
    resumed = t2.train_si_objects(model, tok, SI_ID, _pairs(), arms, cfg=cfg,
                                  save_dir=res_dir)
    assert resumed.skipped == ["lora_r1"]
    assert set(resumed.objects) == {"psr_kl_all"}
    a, b = resumed.objects["psr_kl_all"], ref.objects["psr_kl_all"]
    for i, (x, y) in enumerate(zip(a.trainable, b.trainable)):
        assert torch.equal(x.detach(), y.detach()), \
            f"psr_kl_all param {i} differs on resume"


def test_overwrite_retrains_finished_arms(tiny_llama, tmp_path):
    """--overwrite is the explicit opt-out of resume: everything retrains."""
    model, tok = tiny_llama
    out_dir = tmp_path / "defenses"
    arms = ["lora_r1"]
    t2.train_si_objects(model, tok, SI_ID, _pairs(), arms, cfg=CFG,
                        save_dir=out_dir)
    again = t2.train_si_objects(model, tok, SI_ID, _pairs(), arms, cfg=CFG,
                                save_dir=out_dir, overwrite=True)
    assert again.skipped == [] and set(again.objects) == set(arms)


# --------------------------------------------------------------------------- #
# Resume budget validation: existence-only skip must not survive a changed    #
# supervision budget — check_budget_sidecars                                   #
# --------------------------------------------------------------------------- #


def _budget_sidecar(p, **overrides):
    """A COMPLETE, self-consistent sidecar for the object at ``p``."""
    meta = {"cell_id": p.parent.name, "max_len": 256, "completion_tokens": 16,
            "object_sha256": sha256_file(p)}
    meta.update(overrides)
    return meta


def test_check_budget_sidecars_unit(tmp_path):
    """File-level contract: complete + matching -> clean; max_len drift ->
    WARNING (a memory knob); completion_tokens drift, sha mismatch,
    foreign cell_id, missing/incomplete/unreadable sidecar next to an
    existing object -> HARD SystemExit; no object -> nothing to validate."""
    arm = "loreft"
    p = object_path_for(tmp_path, f"{SI_ID}__{arm}")
    p.parent.mkdir(parents=True)
    p.write_text("obj")
    meta_path = p.parent / "train_budget_meta.json"

    def check(**kw):
        return t2.check_budget_sidecars(tmp_path, SI_ID, [arm], max_len=256,
                                        completion_tokens=16, **kw)

    meta_path.write_text(json.dumps(_budget_sidecar(p)))
    assert check() == []                              # complete + matching

    meta_path.write_text(json.dumps(_budget_sidecar(p, max_len=512)))
    warnings = check()
    assert len(warnings) == 1 and "max_len" in warnings[0]

    meta_path.write_text(json.dumps(_budget_sidecar(p, completion_tokens=32)))
    with pytest.raises(SystemExit, match="completion_tokens=32"):
        check()

    # same budget, DIFFERENT object bytes: a stale pair, refused loudly
    meta_path.write_text(json.dumps(_budget_sidecar(p)))
    p.write_text("tampered bytes under the same-budget sidecar")
    with pytest.raises(SystemExit, match="object_sha256 mismatch"):
        check()
    p.write_text("obj")

    # a pre-stamp legacy sidecar (no object_sha256): explicit fail-loud
    # migration, with the actual sha printed for a deliberate hand-stamp
    legacy = _budget_sidecar(p)
    del legacy["object_sha256"]
    meta_path.write_text(json.dumps(legacy))
    with pytest.raises(SystemExit) as exc:
        check()
    assert "INCOMPLETE" in str(exc.value)
    assert sha256_file(p) in str(exc.value)

    # a sidecar recording a DIFFERENT cell: corrupted store
    meta_path.write_text(json.dumps(_budget_sidecar(p, cell_id="other__cell")))
    with pytest.raises(SystemExit, match="corrupted"):
        check()

    meta_path.write_text("{not json")
    with pytest.raises(SystemExit, match="unreadable"):
        check()

    meta_path.unlink()
    with pytest.raises(SystemExit, match="MISSING"):
        check()

    p.unlink()          # no object on disk: nothing to resume, nothing to check
    meta_path.write_text(json.dumps({"anything": 1}))
    assert check() == []


def test_save_object_writes_sidecar_before_object_crash_order(tmp_path,
                                                              monkeypatch):
    """_save_object stamps the sha into the sidecar and renames it BEFORE
    the object, so a crash between the two leaves a LOUD state, never a
    plausible stale pair: on a fresh save the object is simply absent
    (resume retrains); on an overwrite the OLD bytes sit under a NEW-sha
    sidecar, which the resume check refuses."""
    from types import SimpleNamespace as NS

    arm = "loreft"
    p = object_path_for(tmp_path, f"{SI_ID}__{arm}")
    meta_path_ = lambda: json.loads(  # noqa: E731
        (p.parent / "train_budget_meta.json").read_text())
    obj_new = NS(save=lambda t: Path(t).write_text("NEWBYTES"),
                 fit_site="resid_post")
    obj_old = NS(save=lambda t: Path(t).write_text("OLDBYTES"),
                 fit_site="resid_post")

    real_replace = Path.replace

    def crash_on_object_rename(self, target):
        if Path(target) == p:
            raise RuntimeError("simulated crash before the object rename")
        return real_replace(self, target)

    # ---- fresh save, crash between sidecar and object rename ------------
    monkeypatch.setattr(Path, "replace", crash_on_object_rename)
    with pytest.raises(RuntimeError, match="simulated crash"):
        t2._save_object(obj_new, tmp_path, SI_ID, arm, CFG, 256,
                        {"batch_size": None}, model=None, tok=None)
    assert not p.exists()                       # object never landed
    tmp = p.with_suffix(p.suffix + ".tmp")
    assert meta_path_()["object_sha256"] == sha256_file(tmp)
    # resume sees NO object -> nothing skipped, the arm just retrains
    assert t2.check_budget_sidecars(tmp_path, SI_ID, [arm], max_len=256,
                                    completion_tokens=16) == []

    # ---- overwrite, same crash: OLD bytes + NEW-sha sidecar -> refused ---
    monkeypatch.setattr(Path, "replace", real_replace)
    t2._save_object(obj_old, tmp_path, SI_ID, arm, CFG, 256,
                    {"batch_size": None}, model=None, tok=None)
    assert t2.check_budget_sidecars(tmp_path, SI_ID, [arm], max_len=256,
                                    completion_tokens=16) == []
    monkeypatch.setattr(Path, "replace", crash_on_object_rename)
    with pytest.raises(RuntimeError, match="simulated crash"):
        t2._save_object(obj_new, tmp_path, SI_ID, arm, CFG, 256,
                        {"batch_size": None}, model=None, tok=None)
    assert p.read_text() == "OLDBYTES"          # object untouched by the crash
    with pytest.raises(SystemExit, match="object_sha256 mismatch"):
        t2.check_budget_sidecars(tmp_path, SI_ID, [arm], max_len=256,
                                 completion_tokens=16)


def test_resume_refuses_stale_pair_after_out_of_band_change(tiny_llama,
                                                            tmp_path):
    """A trained object whose bytes changed out-of-band under its
    same-budget sidecar is a stale pair — refused, never resumed."""
    model, tok = tiny_llama
    out_dir = tmp_path / "defenses"
    t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"], cfg=CFG,
                        save_dir=out_dir)
    p = object_path_for(out_dir, f"{SI_ID}__loreft")
    p.write_bytes(b"tampered: same budget sidecar, different object")
    with pytest.raises(SystemExit, match="object_sha256 mismatch"):
        t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"], cfg=CFG,
                            save_dir=out_dir)


def test_resume_refuses_on_changed_completion_tokens(tiny_llama, tmp_path):
    """The completion-token knob changed under an existing object: the
    rerun must refuse BEFORE any arm trains or skips — never silently mix
    supervision budgets in one comparison."""
    model, tok = tiny_llama
    out_dir = tmp_path / "defenses"
    t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"], cfg=CFG,
                        save_dir=out_dir)
    with pytest.raises(SystemExit, match="completion_tokens=16"):
        t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                            cfg=CFG | {"completion_tokens": 32},
                            save_dir=out_dir)
    # the on-disk object was left alone (refused, not clobbered)
    assert object_path_for(out_dir, f"{SI_ID}__loreft").exists()


# (test_resume_refuses_on_missing_sidecar and
# test_resume_warns_but_skips_on_max_len_change are GONE, phase-C join: the
# F16 COMPLETE-marker regime replaces both — an object with no budget record
# is an INCOMPLETE cell that re-runs and re-stamps (pinned by
# test_object_present_but_sidecar_missing_is_incomplete below), and a
# changed max_len is a fingerprint mismatch that re-runs rather than
# skipping. check_budget_sidecars still hard-refuses a PRESENT record that
# contradicts the current knobs — pinned above.)


def test_overwrite_bypasses_budget_validation(tiny_llama, tmp_path):
    """--overwrite retrains everything at the CURRENT budget, so a changed
    knob is not a mismatch — and the fresh sidecar records the new value."""
    model, tok = tiny_llama
    out_dir = tmp_path / "defenses"
    t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"], cfg=CFG,
                        save_dir=out_dir)
    again = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                                cfg=CFG | {"completion_tokens": 32},
                                save_dir=out_dir, overwrite=True)
    assert again.ok and set(again.objects) == {"loreft"}
    meta = json.loads((object_path_for(out_dir, f"{SI_ID}__loreft").parent
                       / "train_budget_meta.json").read_text())
    assert meta["completion_tokens"] == 32


# --------------------------------------------------------------------------- #
# n3 fit-run regressions: the two defects that cost 8 of 25 object cells      #
# (cast + alphakl_null + alphakl_fisher x both SIs to the print shim's        #
# keyword-step TypeError; alphakl_soft x both SIs to the missing manual beta) #
# --------------------------------------------------------------------------- #


def test_console_log_accepts_every_wandb_log_call_shape(capsys, monkeypatch):
    """With W&B disabled (JB4 containment: WANDB_DISABLED=true) the driver's
    log_fn used to be the BUILTIN ``print``, which tolerates the positional
    shape ``log_fn(payload, step)`` (the lockstep/distill/lofit loops) but
    raises ``TypeError: 'step' is an invalid keyword argument for print()``
    on the KEYWORD shape used by the calibration ladder
    (extraction/calibrate.py, antidote.py) and alphakl's summary line. The
    shim mirrors ``wandb.log(data, step=, commit=, sync=)``, so every call
    valid against the real wandb.log must not raise — keyword step, its
    other kwargs, AND the positional form the working arms already use."""
    monkeypatch.setenv("WANDB_DISABLED", "true")  # the containment env
    t2.console_log({"alphakl1/kl": 0.1}, step=0)              # keyword step
    t2.console_log({"cast_calib/rung": 1.0}, step=3,
                   commit=False, sync=None)                   # other kwargs
    t2.console_log({"lofit/bias_ce": 0.53, "epoch": 5}, 1910) # positional
    t2.console_log({"solo/loss": 0.5})                        # no step
    out = capsys.readouterr().out
    # The positional rendering the successful arms already log in ("dict then
    # bare number") is preserved exactly, keyword or positional.
    assert "{'lofit/bias_ce': 0.53, 'epoch': 5} 1910" in out
    assert "{'alphakl1/kl': 0.1} 0" in out


def test_cmd_train_wires_the_console_shim_not_bare_print():
    """The fix lives at the driver's log_fn wiring: reverting cmd_train to
    ``log_fn=print`` reintroduces the keyword-step TypeError for every
    trainer that logs with ``step=`` as a keyword."""
    assert "console_log" in t2.cmd_train.__code__.co_names


def test_keyword_step_arm_trains_through_the_console_shim(tiny_llama, tmp_path):
    """End to end: alphakl_null's base-scale calibration logs with ``step=``
    as a KEYWORD (extraction/calibrate.py) — exactly the call that killed the
    arm under bare print (contained per arm, so it surfaced as a failed cell
    in the fit report, not a crash). Through the driver with the console shim
    the arm must train and save."""
    model, tok = tiny_llama
    data = [
        ("Say something nice.", "Bonjour le monde", "Hello world"),
        ("Describe a tree.", "C'est un grand arbre", "It is a tall tree"),
        ("Greet the reader.", "Salut a tous", "Hi everyone"),
        ("Name a color.", "Le ciel est bleu", "The sky is blue"),
        ("Offer some tea.", "Voulez-vous du the", "Would you like tea"),
        ("Count to two.", "Un puis deux", "One then two"),
    ]
    pairs = [Pair(prompt=p, pos=pos, neg=neg, instruction_id=SI_ID)
             for p, pos, neg in data]
    cfg = dict(CFG)
    cfg["train_kwargs"] = {
        **CFG["train_kwargs"],
        # the tiny-model knobs test_alphakl.py's trainer tests use
        "alphakl_null": {"calib_frac": 0.25, "rollout_tokens": 6,
                         "batch_size": 4},
    }
    outcome = t2.train_si_objects(
        model, tok, SI_ID, pairs, ["alphakl_null"], cfg=cfg,
        save_dir=tmp_path / "defenses", log_fn=t2.console_log)
    assert outcome.failed == {}, {a: o.error for a, o in outcome.failed.items()}
    assert set(outcome.objects) == {"alphakl_null"}


def test_two_si_config_leaves_alphakl_soft_beta_unpinned():
    """configs/two_si.yaml must NOT pin the stage-2 beta (2026-08-24,
    inverting this test's original assertion): the old beta=1.0 pin was a
    crash workaround — train_alphakl used to raise at stage 2 without a
    manual beta, which is how the n3 fit run lost the alphakl_soft cell
    for BOTH SIs — promoted to a hyperparameter, and the arm diverged
    under it (the 2026-08-22 wave; wave-record floor_ratio 2.63/3.14 vs a
    no-op's 1.0, against since-re-measured floors, so population-scoped).
    Stage 2 now derives beta against the eps benign budget when the
    key is absent (the same calibrate_beta bisection stage 3 uses), so an
    absent key IS the deliberate configuration; a beta reappearing here
    would silently re-pin the very step size that diverged."""
    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "two_si.yaml")
        .read_text(encoding="utf-8"))
    per_arm_tk = cfg.get("train_kwargs") or {}
    arm_kw = dict(per_arm_tk.get("alphakl_soft") or {})
    assert "beta" not in arm_kw, (
        "configs/two_si.yaml must leave train_kwargs.alphakl_soft.beta "
        "absent — beta is derived from the eps benign budget by "
        "train_alphakl(stage=2); pin it only to reproduce a recorded run")


def test_dead_gate_psr_never_reaches_disk(tiny_llama, tmp_path, monkeypatch):
    """Dead-gate guard on the LOCKSTEP save path: lockstep trains PSR arms
    without train_psr, so the guard embedded there never runs — the driver
    itself must refuse to save a PSR whose gates trained shut (psr_kl_all
    once trained 966 steps ~30x worse than a no-op — a contiguous deep
    block of gates frozen at gate_b=0.924214, 18/32 layers on one SI and
    11/32 on the other — and the artifact was written anyway). Simulate
    training-induced gate death right after train_lockstep returns; the arm
    must record failed (DeadGateError), write NO artifact, and the group's
    other arm must still save."""
    import sisteer.extraction.lockstep as lockstep_mod

    from sisteer.extraction.lockstep import train_lockstep as real_train
    from sisteer.extraction.psr import PSRObject

    model, tok = tiny_llama

    def train_then_kill_gates(model_, tok_, group, examples, **kw):
        out = real_train(model_, tok_, group, examples, **kw)
        for a in group:
            if isinstance(a.obj, PSRObject):
                for l in a.obj.layers:
                    a.obj.gate_w[l].data.zero_()
                    a.obj.gate_b[l].data.fill_(-1.0)  # ReLU(-1) == 0 always
        return out

    monkeypatch.setattr(lockstep_mod, "train_lockstep", train_then_kill_gates)
    out_dir = tmp_path / "defenses"
    outcome = t2.train_si_objects(model, tok, SI_ID, _pairs(),
                                  ["lora_r1", "psr_kl_all"], cfg=CFG,
                                  save_dir=out_dir)
    assert set(outcome.failed) == {"psr_kl_all"}
    assert outcome.failed["psr_kl_all"].error_type == "DeadGateError"
    assert "psr_kl_all" not in outcome.objects
    assert not object_path_for(out_dir, f"{SI_ID}__psr_kl_all").exists()
    # the healthy group member's work survived, on disk
    assert object_path_for(out_dir, f"{SI_ID}__lora_r1").exists()
    report = json.loads(t2.report_path_for(out_dir, SI_ID).read_text())
    assert report["failed"] == ["psr_kl_all"]


# --------------------------------------------------------------------------- #
# Live, METRICS-ONLY W&B: scalars                                              #
# stream live, console capture stays off, no prompt/pair/attack text can       #
# leave the process, and a W&B outage can never cost a fit or its curves.      #
# --------------------------------------------------------------------------- #


class _FakeWandbRun:
    def __init__(self):
        self.rows = []
        self.finished = False

    def log(self, data):
        self.rows.append(dict(data))

    def finish(self):
        self.finished = True


class _ExplodingRun(_FakeWandbRun):
    def log(self, data):
        raise RuntimeError("simulated W&B outage")


def test_metrics_log_fn_streams_scalars_only(tmp_path, capsys):
    """Strings NEVER leave the process through the metrics path: a payload
    carrying attack text keeps its numeric keys (jsonl + W&B) and drops the
    text, and a bare status string is console-only. The signature stays
    wandb.log-compatible (positional/keyword step, commit/sync)."""
    run = _FakeWandbRun()
    path = tmp_path / "m.jsonl"
    log = t2.make_metrics_log_fn(jsonl_path=path, wandb_run=run,
                                 echo=t2.console_log)
    log({"loss": 0.5, "attack_text": "SYNTHETIC ATTACK STRING", "epoch": 1},
        3)                                             # positional step
    log({"kl": 0.25}, step=7, commit=False, sync=None)  # keyword shapes
    log("plain status line")                            # console-only text
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [{"step": 3, "loss": 0.5, "epoch": 1},
                    {"step": 7, "kl": 0.25}]
    assert run.rows == [{"loss": 0.5, "epoch": 1, "step": 3},
                        {"kl": 0.25, "step": 7}]
    blob = path.read_text() + json.dumps(run.rows)
    assert "SYNTHETIC ATTACK STRING" not in blob
    assert "plain status line" not in blob
    # the console echo still carries everything (the job log's record)
    out = capsys.readouterr().out
    assert "plain status line" in out and "loss" in out


def test_metrics_log_fn_survives_a_wandb_outage(tmp_path, capsys):
    """wandb_run.log raising must not kill the fit NOR lose the curve: the
    jsonl row is written first, the failure is reported once on stderr, and
    later calls keep flowing to console + jsonl."""
    path = tmp_path / "m.jsonl"
    log = t2.make_metrics_log_fn(jsonl_path=path,
                                 wandb_run=_ExplodingRun(),
                                 echo=t2.console_log)
    log({"loss": 0.5}, 1)
    log({"loss": 0.4}, 2)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["loss"] for r in rows] == [0.5, 0.4]
    err = capsys.readouterr().err
    assert err.count("W&B log failed") == 1  # loud once, then quiet


def test_metrics_log_fn_kl_lane_mirror_is_unchanged(tmp_path):
    """The KL-distillation lane's mirror is byte-identical to before the
    CE-lane rename: probe/<arm>/residual_ratio (fit path) and the flat
    probe/residual_ratio (trial path) both surface as the top-level
    kl_ratio_vs_unsteered panel, with or without the ``arms`` hint, the
    teacher-relative keys keep their names, and no CE-lane key appears."""
    for arms in ((), ["lora_r1", "psr_kl_all"]):
        run = _FakeWandbRun()
        path = tmp_path / f"m{len(arms)}.jsonl"
        log = t2.make_metrics_log_fn(jsonl_path=path, wandb_run=run,
                                     arms=arms)
        log({"probe/lora_r1/residual_ratio": 0.7,
             "probe/lora_r1/heldout_loss": 0.12,
             "probe/lora_r1/residual_kl": 0.12}, 4)
        log({"probe/residual_ratio": 0.9}, 5)          # trial path, flat
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        assert rows == [
            {"step": 4, "probe/lora_r1/residual_ratio": 0.7,
             "probe/lora_r1/heldout_loss": 0.12,
             "probe/lora_r1/residual_kl": 0.12,
             "kl_ratio_vs_unsteered": 0.7},
            {"step": 5, "probe/residual_ratio": 0.9,
             "kl_ratio_vs_unsteered": 0.9}], arms
        assert not any("displacement_kl_vs_base" in r for r in rows), arms
        assert [set(r) for r in run.rows] == [set(r) for r in rows], arms


def test_metrics_log_fn_ce_lane_mirrors_displacement_never_a_kl_ratio(
        tmp_path):
    """The CE lane is chosen by ARM KIND (membership in CE_ARMS), not key
    shape: a CE arm's probe record mirrors probe/<arm>/displacement_kl_vs_base
    to the top-level displacement_kl_vs_base panel and NEVER feeds
    kl_ratio_vs_unsteered — even when a residual_ratio key for that arm
    reaches the stream (a displacement over a ~0 floor is not a ratio). A
    KL arm in the SAME run keeps its own mirror, and W&B sees exactly what
    the tee records."""
    run = _FakeWandbRun()
    path = tmp_path / "m.jsonl"
    log = t2.make_metrics_log_fn(jsonl_path=path, wandb_run=run,
                                 arms=["psr_ce_all", "lora_r1"])
    log({"probe/psr_ce_all/displacement_kl_vs_base": 0.31,
         "probe/psr_ce_all/displacement_kl": 0.30,
         "probe/psr_ce_all/noop_floor": 2e-7}, 10)
    log({"probe/psr_ce_all/residual_ratio": 3.1e5}, 11)   # a stray old key
    log({"probe/lora_r1/residual_ratio": 0.8}, 12)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert rows[0]["displacement_kl_vs_base"] == 0.31
    assert "kl_ratio_vs_unsteered" not in rows[0]
    assert set(rows[1]) == {"step", "probe/psr_ce_all/residual_ratio"}
    assert rows[2]["kl_ratio_vs_unsteered"] == 0.8
    assert "displacement_kl_vs_base" not in rows[2]
    assert [set(r) for r in run.rows] == [set(r) for r in rows]


def test_ce_lane_probe_record_renames_only_the_teacher_relative_keys():
    """ce_lane_probe_record — the rename fitmon's flat record gets on a CE
    arm's shared stream: residual_kl -> displacement_kl_vs_base under both
    probe/ and final/, the heldout_loss alias and both *_ratio keys
    dropped, everything else (displacement_kl, noop_floor, gate activity,
    train/, norm groups) byte for byte. And the floor verdict's stream
    record takes the fit_record's field names."""
    rec = {"probe/residual_kl": 0.4, "probe/heldout_loss": 0.4,
           "probe/residual_ratio": 4e5, "probe/displacement_ratio": 3.9e5,
           "probe/displacement_kl": 0.39, "probe/noop_floor": 1e-7,
           "probe/gate_active_frac": 0.5, "train/loss": 2.1,
           "norm/gate_w": 0.3, "final/residual_kl": 0.5,
           "final/residual_ratio": 5e5, "final/displacement_ratio": 4.9e5,
           "final/displacement_kl": 0.49, "final/noop_floor": 1e-7}
    assert t2.ce_lane_probe_record(rec) == {
        "probe/displacement_kl_vs_base": 0.4, "probe/displacement_kl": 0.39,
        "probe/noop_floor": 1e-7, "probe/gate_active_frac": 0.5,
        "train/loss": 2.1, "norm/gate_w": 0.3,
        "final/displacement_kl_vs_base": 0.5, "final/displacement_kl": 0.49,
        "final/noop_floor": 1e-7}
    # the maps cover exactly the module docstring's table
    assert set(t2.CE_PROBE_STREAM_KEYS) == {
        "residual_kl", "heldout_loss", "residual_ratio", "displacement_ratio"}
    tag = "constv2-mix-1k__psr_ce_all__heldout_ce"
    assert t2._ce_floor_stream_record(
        {f"{tag}/final_probe_loss": 2.3, f"{tag}/noop_floor": 2.4,
         f"{tag}/floor_ratio": 0.958, "other/key": 1}) == {
        f"{tag}/heldout_ce_active": 2.3, f"{tag}/heldout_ce_noop": 2.4,
        f"{tag}/ce_ratio": 0.958, "other/key": 1}


def test_wandb_fit_run_respects_containment_and_test_envs(monkeypatch):
    """WANDB_DISABLED=true (transcript containment) and WANDB_MODE=disabled
    (the test env) both mean NO run — without importing wandb at all."""
    monkeypatch.setenv("WANDB_DISABLED", "true")
    assert t2.wandb_fit_run({}, SI_ID, []) is None
    monkeypatch.delenv("WANDB_DISABLED")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert t2.wandb_fit_run({}, SI_ID, []) is None
    monkeypatch.setenv("WANDB_MODE", "online")
    assert t2.wandb_fit_run({"wandb_fits": False}, SI_ID, []) is None


def test_wandb_fit_run_entity_comes_from_the_environment_first(monkeypatch):
    """The destination entity is WANDB_ENTITY (what the Modal secret
    carries and the container verified the key defaults to), then
    the config's wandb_entity, never wandb's implicit default."""
    import types

    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.setenv("WANDB_MODE", "online")
    calls = {}
    fake = types.ModuleType("wandb")
    fake.Settings = lambda **kw: kw

    def _init(**kw):
        calls["init"] = kw
        return _FakeWandbRun()

    fake.init = _init
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_PROJECT", "test-project")      # these configs name no wandb_project: $WANDB_PROJECT
    monkeypatch.setenv("WANDB_ENTITY", "operator-entity")
    t2.wandb_fit_run({"wandb_entity": "cfg-entity"}, SI_ID, ["lora_r1"])
    assert calls["init"]["entity"] == "operator-entity"
    monkeypatch.delenv("WANDB_ENTITY")
    t2.wandb_fit_run({"wandb_entity": "cfg-entity"}, SI_ID, ["lora_r1"])
    assert calls["init"]["entity"] == "cfg-entity"
    t2.wandb_fit_run({}, SI_ID, ["lora_r1"])
    assert calls["init"]["entity"] is None and calls["init"]["project"] == "test-project"


def test_wandb_fit_run_is_metrics_only_with_console_capture_off(monkeypatch):
    """The init path forces WANDB_CONSOLE=off (this lane's stdout prints
    raw attack/pair text on error paths — console capture would upload it)
    and passes console='off' in Settings; the run config carries
    identifiers only."""
    import types

    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.delenv("WANDB_CONSOLE", raising=False)
    calls = {}
    fake = types.ModuleType("wandb")

    def _settings(**kw):
        calls["settings"] = kw
        return kw

    def _init(**kw):
        calls["init"] = kw
        return _FakeWandbRun()

    fake.Settings = _settings
    fake.init = _init
    monkeypatch.setitem(sys.modules, "wandb", fake)
    run = t2.wandb_fit_run({"wandb_project": "two-si-proj"}, SI_ID,
                           ["lora_r1"])
    assert isinstance(run, _FakeWandbRun)
    import os

    assert os.environ["WANDB_CONSOLE"] == "off"
    assert calls["settings"]["console"] == "off"
    assert calls["init"]["project"] == "two-si-proj"
    assert calls["init"]["config"]["si_id"] == SI_ID
    # nothing free-text beyond identifiers may ride the config. This is a
    # LEAK GUARD, so widening it is deliberate: `positions` is admitted
    # because it is a config enum ("all" / "response"), the same kind of
    # scalar identifier as si_id and arms -- and it is needed in the config
    # rather than only the run name so an A/C contrast can be grouped in the
    # UI. `model`/`model_revision` are hub identifiers (twosi-model-sweeps:
    # two models fit concurrently, and the cross-model comparisons group on
    # them). Adding anything derived from prompt, pair or attack text here
    # would defeat the assertion's purpose.
    assert set(calls["init"]["config"]) <= {"si_id", "arms", "layer", "seed",
                                            "positions", "model",
                                            "model_revision"}
    assert calls["init"]["config"]["positions"] in (None, "all", "response")


def test_wandb_fit_run_falls_back_when_init_fails(monkeypatch, capsys):
    """No login / no route to api.wandb.ai -> a loud stderr note and None;
    the fit proceeds on console + jsonl (an outage never costs a fit)."""
    import types

    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.setenv("WANDB_MODE", "online")
    fake = types.ModuleType("wandb")
    fake.Settings = lambda **kw: kw

    def _init(**kw):
        raise ConnectionError("no route to api.wandb.ai")

    fake.init = _init
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    assert t2.wandb_fit_run({}, SI_ID, []) is None
    assert "W&B unavailable" in capsys.readouterr().err


def test_wandb_fit_run_with_no_project_on_this_host_runs_without_wandb_and_names_the_variable(monkeypatch, capsys):
    """The configs write wandb_project: $WANDB_PROJECT; on a host where it has no value the fit proceeds without W&B (an
    optional sink) and the note names the variable to set, rather than init ever seeing a literal project."""
    import types

    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.setenv("WANDB_MODE", "online")
    for v in ("WANDB_PROJECT", "SISTEER_WANDB_PROJECT", "SISTEER_CONFIG"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("SISTEER_SITE_DIR", "/nonexistent-site-dir")
    fake = types.ModuleType("wandb")
    fake.Settings = lambda **kw: kw
    fake.init = lambda **kw: pytest.fail("wandb.init must not run with an unresolved project")
    monkeypatch.setitem(sys.modules, "wandb", fake)
    assert t2.wandb_fit_run({"wandb_project": "$WANDB_PROJECT"}, SI_ID, []) is None
    err = capsys.readouterr().err
    assert "W&B unavailable" in err and "WANDB_PROJECT" in err


def test_cmd_train_wires_live_metrics_over_the_console_shim():
    """cmd_train builds the composite log_fn: W&B init + per-step jsonl +
    the console_log shim (whose signature regression the tests above pin),
    and finishes the run even on failure."""
    names = set(t2.cmd_train.__code__.co_names)
    assert {"wandb_fit_run", "make_metrics_log_fn", "console_log"} <= names


# --------------------------------------------------------------------------- #
# Fit-time provenance (audit 2026-08-21): model_id, model_revision,            #
# template_hash, resolved_batch_size, site — stamped by _save_object on both   #
# the lockstep and solo paths                                                  #
# --------------------------------------------------------------------------- #

TINY_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _sidecar(save_dir, arm):
    return json.loads(
        (object_path_for(save_dir, f"{SI_ID}__{arm}").parent
         / "train_budget_meta.json").read_text())


def test_sidecar_records_fit_provenance_uniformly_on_both_paths(
        tiny_llama, tmp_path):
    """Every artifact records what its fit ACTUALLY ran under — the live
    model's id and pinned commit, the rendered chat-template hash (byte-
    comparable with capeval's records), the realised batch width, and the
    hook site — identically from the lockstep and the solo save paths."""
    from sisteer.capeval.manifest import sha256_hex16
    from sisteer.eval.train_budget import resolve_batch_size

    model, tok = tiny_llama
    # lockstep: lora_r1 + psr_kl_all; solo: loreft + lofit
    arms = ["lora_r1", "psr_kl_all", "loreft", "lofit"]
    outcome = t2.train_si_objects(model, tok, SI_ID, _pairs(), arms, cfg=CFG,
                                  save_dir=tmp_path, sentinel=_sentinel(tok))
    assert outcome.ok, {a: o.error for a, o in outcome.failed.items()}

    expected_template = sha256_hex16(
        (getattr(tok, "chat_template", None) or "").encode("utf-8"))
    gbs = t2.group_batch_size(CFG, SI_ID, ["lora_r1", "psr_kl_all"])
    expected_bs = {"lora_r1": gbs, "psr_kl_all": gbs,
                   "loreft": resolve_batch_size(CFG, SI_ID, "loreft"),
                   "lofit": resolve_batch_size(CFG, SI_ID, "lofit")}
    expected_site = {"lora_r1": "o_proj_out",     # lockstep, inside the block
                     "psr_kl_all": "resid_post",  # lockstep, layer output
                     "loreft": "resid_post",      # solo, layer output
                     "lofit": "o_proj_in"}        # solo, o_proj PRE-hook
    for arm in arms:
        meta = _sidecar(tmp_path, arm)
        # 1+2: the id/revision the fit actually loaded, from the LIVE model
        assert meta["model_id"] == TINY_MODEL_ID, arm
        assert meta["model_id"] == model.config._name_or_path, arm
        assert meta["model_revision"] == model.config._commit_hash, arm
        assert meta["model_revision"], arm  # hub-cache load: commit is known
        int(meta["model_revision"], 16)     # a commit sha, not a ref name
        # 3: the rendered-template hash, capeval's convention byte-for-byte
        assert meta["template_hash"] == expected_template, arm
        assert len(meta["template_hash"]) == 16, arm
        # 4: the realised width (== the driver's batch_size record)
        assert meta["resolved_batch_size"] == expected_bs[arm], arm
        assert meta["resolved_batch_size"] == meta["batch_size"], arm
        # 5: the hook site
        assert meta["site"] == expected_site[arm], arm


def test_resolved_batch_size_is_the_group_min_not_the_configured_entry(
        tiny_llama, tmp_path):
    """The live incident shape: an arm whose OWN configured width is 4 joins
    a lockstep group whose min is 2 — the fit runs (and must record) 2, not
    the 4 its batch_size_by_arm entry claims. A solo arm still records its
    own resolved width."""
    from sisteer.eval.train_budget import resolve_batch_size

    model, tok = tiny_llama
    cfg = CFG | {"batch_size_by_arm": {"lora_r1": 4, "psr_kl_all": 2,
                                       "loreft": 3}}
    # the arm's own entry really is 4 — what a config reader would (wrongly)
    # report as this cell's width
    assert resolve_batch_size(cfg, SI_ID, "lora_r1") == 4
    assert t2.group_batch_size(cfg, SI_ID, ["lora_r1", "psr_kl_all"]) == 2
    outcome = t2.train_si_objects(model, tok, SI_ID, _pairs(),
                                  ["lora_r1", "psr_kl_all", "loreft"],
                                  cfg=cfg, save_dir=tmp_path)
    assert outcome.ok, {a: o.error for a, o in outcome.failed.items()}
    assert _sidecar(tmp_path, "lora_r1")["resolved_batch_size"] == 2
    assert _sidecar(tmp_path, "psr_kl_all")["resolved_batch_size"] == 2
    assert _sidecar(tmp_path, "loreft")["resolved_batch_size"] == 3


def test_template_hash_reuses_the_capeval_convention_and_detects_divergence(
        tiny_llama, tmp_path):
    """The sidecar hash is byte-comparable with capeval's template_hash
    (same input: the raw chat_template string, "" when absent; same digest:
    sha256; same 16-hex truncation — store.eval_config_digest's input, as
    computed in capeval_judged_run/cell_controls_run), so a fit-vs-serve
    template divergence — the NousResearch-vs-meta-llama incident — is
    detectable by comparing the two records."""
    import types

    from sisteer.capeval.manifest import sha256_hex16
    from sisteer.methods import VectorObject
    from sisteer.vectors import SteeringVector

    model, _tok = tiny_llama

    def save_with(template, sub):
        obj = VectorObject(SteeringVector(vectors={}, method="distill_soft",
                                          model_name="", instruction_id=""))
        out = tmp_path / sub
        t2._save_object(obj, out, SI_ID, "vector", {"completion_tokens": 4},
                        16, {"batch_size": 1}, model=model,
                        tok=types.SimpleNamespace(chat_template=template))
        return json.loads((object_path_for(out, f"{SI_ID}__vector").parent
                           / "train_budget_meta.json").read_text())

    t_a = "{% for m in messages %}A{{ m['content'] }}{% endfor %}"
    t_b = "{% for m in messages %}B{{ m['content'] }}{% endfor %}"
    h_a = save_with(t_a, "a")["template_hash"]
    h_b = save_with(t_b, "b")["template_hash"]
    h_none = save_with(None, "none")["template_hash"]
    assert h_a == sha256_hex16(t_a.encode("utf-8"))
    assert h_b == sha256_hex16(t_b.encode("utf-8"))
    assert h_a != h_b  # different templates -> different records
    # a template-less tokenizer hashes "" — capeval's convention exactly
    assert h_none == sha256_hex16(b"")


def test_fit_site_derivation_per_object_class(tiny_llama):
    """object_fit_site reads the CLASS (never the arm name), so arms sharing
    a substrate (lora_r1 / bidpo_lora) inherit one site and subclasses
    resolve via the MRO; a self-declared fit_site wins; an unknown class
    refuses loudly rather than writing a null a serve lane would read as
    "no layer-index shift needed"."""
    import types

    from sisteer.extraction.caa_static_r import CaaStaticRObject
    from sisteer.extraction.lofit import LoFiTObject
    from sisteer.extraction.objects import RankOneLoRAObject, SoftTokenObject
    from sisteer.extraction.psr import PSRObject
    from sisteer.methods import VectorObject
    from sisteer.models import hidden_size
    from sisteer.vectors import SteeringVector

    model, tok = tiny_llama
    sv = SteeringVector(vectors={}, method="m", model_name="",
                        instruction_id="")
    assert t2.object_fit_site(VectorObject(sv)) == "resid_post"
    assert t2.object_fit_site(
        PSRObject(model, [1], train_direction=True)) == "resid_post"
    assert t2.object_fit_site(
        CaaStaticRObject(model,
                         {0: torch.zeros(hidden_size(model))})) == "resid_pre"
    assert t2.object_fit_site(RankOneLoRAObject(model, 1)) == "o_proj_out"
    # lofit PRE-hooks o_proj: its edit lands on o_proj's INPUT (the per-head
    # attention outputs), NOT on the decoder layer's input — labelling it
    # "resid_pre" would invite exactly the one-block serve shift the site
    # field exists to prevent.
    assert t2.object_fit_site(LoFiTObject(model, heads=[])) == "o_proj_in"
    assert t2.object_fit_site(
        SoftTokenObject(model, tok,
                        sentinel=_sentinel(tok))) == "embedding_out"
    # subclasses inherit through the MRO (token_faithful's shape)
    sub = type("ASubclassObject", (VectorObject,), {})(sv)
    assert t2.object_fit_site(sub) == "resid_post"
    # a self-declared fit_site wins over the table (the test-double seam)
    assert t2.object_fit_site(
        types.SimpleNamespace(fit_site="resid_pre")) == "resid_pre"
    with pytest.raises(ValueError, match="fit-time hook site"):
        t2.object_fit_site(object())


def test_save_object_stamps_resid_pre_for_the_caa_class(tiny_llama, tmp_path):
    """The one resid_pre class, through the real save seam: its sidecar
    carries site=resid_pre so a resid_post serve op knows to take the
    one-block index shift (resid_pre(L) == resid_post(L-1))."""
    from sisteer.extraction.caa_static_r import CaaStaticRObject
    from sisteer.models import hidden_size

    model, tok = tiny_llama
    obj = CaaStaticRObject(model, {0: torch.zeros(hidden_size(model))})
    t2._save_object(obj, tmp_path, SI_ID, "caa_static_r",
                    {"completion_tokens": 4}, 16, {"batch_size": 1},
                    model=model, tok=tok)
    meta = _sidecar(tmp_path, "caa_static_r")
    assert meta["site"] == "resid_pre"
    assert meta["model_revision"] == model.config._commit_hash


def test_sidecar_model_revision_falls_back_to_the_config_pin(
        tiny_llama, tmp_path, monkeypatch):
    """The offline-cache load path leaves ``config._commit_hash`` unset (the
    qwen35_4b_base bank: every round-1 sidecar recorded model_revision
    null while the fits config pinned one, and the derived-object builder
    had to resolve it from _store_identity.json). The sidecar now records
    the config's pin in that case — the loader was handed it as
    ``revision=``, so it IS what loaded — and says so via
    ``revision_source``; a live commit still wins over the pin (reality
    trained, not the config), and no commit + no pin stays a stated
    null/null absence."""
    from sisteer.methods import VectorObject
    from sisteer.vectors import SteeringVector

    model, tok = tiny_llama
    live = model.config._commit_hash
    assert live  # hub-cache fixture: the live commit is known
    pin = "f" * 40
    assert pin != live

    def save_with(cfg, sub):
        obj = VectorObject(SteeringVector(vectors={}, method="distill_soft",
                                          model_name="", instruction_id=""))
        out = tmp_path / sub
        t2._save_object(obj, out, SI_ID, "vector", cfg, 16,
                        {"batch_size": 1}, model=model, tok=tok)
        return json.loads((object_path_for(out, f"{SI_ID}__vector").parent
                           / "train_budget_meta.json").read_text())

    # 1. live commit present: it wins, pin or no pin
    m = save_with({"completion_tokens": 4, "model_revision": pin}, "live")
    assert m["model_revision"] == live
    assert m["revision_source"] == "live_model"
    # 2. the offline-cache shape: no live commit, a pinned config
    monkeypatch.setattr(model.config, "_commit_hash", None)
    m = save_with({"completion_tokens": 4, "model_revision": pin}, "pin")
    assert m["model_revision"] == pin
    assert m["revision_source"] == "config_pin"
    # 3. neither: the stated absence, on both keys
    m = save_with({"completion_tokens": 4}, "none")
    assert m["model_revision"] is None
    assert m["revision_source"] is None


def test_post_publish_sidecar_rewrite_re_runs(tiny_llama, tmp_path):
    """F16 inverts the old lenient contract on purpose: a sidecar REWRITTEN
    after publication (provenance keys stripped, bytes no longer matching
    the COMPLETE manifest's sha) means the cell's provenance was lost — so
    resume RE-RUNS the cell and re-publishes a full-provenance record,
    rather than skipping on the object file alone. (The old test pinned
    the skip-and-leave-untouched behaviour; identity-bound resume cannot
    honour a record it can no longer verify.)"""
    model, tok = tiny_llama
    outcome = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                                  cfg=CFG, save_dir=tmp_path)
    assert outcome.ok
    sidecar = (object_path_for(tmp_path, f"{SI_ID}__loreft").parent
               / "train_budget_meta.json")
    meta = json.loads(sidecar.read_text())
    for k in ("model_id", "model_revision", "template_hash",
              "resolved_batch_size", "site"):
        del meta[k]  # rewrite the sidecar in its pre-audit shape
    sidecar.write_text(json.dumps(meta, indent=1, sort_keys=True))

    assert t2.validated_complete(tmp_path, f"{SI_ID}__loreft",
                                 _fp("loreft")) is None
    again = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                                cfg=CFG, save_dir=tmp_path)
    assert again.arms["loreft"].status == "trained"  # re-ran, not skipped
    fresh = json.loads(sidecar.read_text())
    assert fresh["model_revision"] == model.config._commit_hash  # restored


# --------------------------------------------------------------------------- #
# Floor-probe sample size (wave post-mortem 2026-08-24): the probe used to     #
# read batch 0 only at the ARM'S TRAINING BATCH SIZE, so every recorded floor  #
# and ratio was a 2-4 sample estimate of a KL (si_irrelevant's floor read 2x   #
# the truth measured over 256 prompts, job 187947). The sample size is now     #
# probe_n (train_kwargs.default.probe_n, default 256, clamped to the pool),    #
# decoupled from the memory chunk width, and the sidecar records the realised  #
# count AND the estimate's uncertainty next to the floor/ratio it justifies.   #
# --------------------------------------------------------------------------- #

# Prompts whose sha256 first byte >= 0.1 under the driver's default salt
# (the FIT side of test_early_stop's known split), so the 0.1 holdout carve
# leaves all five in the fit pool deterministically.
_FIT_ONLY_PROMPTS = ["Greet the reader.", "Count to two.", "What causes rain?",
                     "Name a planet.", "Describe the sea."]


def _probe_pairs():
    return [Pair(prompt=p, pos=f"Reponse {i} en francais",
                 neg=f"Reply {i} in English", instruction_id=SI_ID)
            for i, p in enumerate(_FIT_ONLY_PROMPTS)]


def _probe_cfg(**default_overrides):
    """CFG with the floor guard ACTIVE but never refusing (ratio_max=1e9):
    these tests are about the RECORD, not the verdict."""
    return CFG | {"train_kwargs": {"default": {
        "epochs": 2, "seed": 0, "floor_ratio_max": 1e9,
        **default_overrides}}}


def test_probe_sample_size_decoupled_from_training_batch_size(
        tiny_llama, tmp_path):
    """THE root-cause pin: with a training batch size of 2 and a 5-pair fit
    pool, the recorded probe covers the POOL (5 examples), not the batch
    size — the two numbers are different knobs. batch_size survives as the
    memory chunk width only."""
    model, tok = tiny_llama
    pairs = _probe_pairs()
    cfg = _probe_cfg() | {"batch_size_by_arm": {"loreft": 2}}
    outcome = t2.train_si_objects(model, tok, SI_ID, pairs, ["loreft"],
                                  cfg=cfg, save_dir=tmp_path)
    assert outcome.ok, {a: o.error for a, o in outcome.failed.items()}
    fit_pairs, _ = t2.split_pairs(pairs, 0.1, "two_si_holdout")
    assert len(fit_pairs) == 5           # the fixture's known-fit prompts

    rec = _sidecar(tmp_path, "loreft")["fit_record"]
    assert rec["probe_n"] == 5           # the pool (< the 256 default, clamped)
    assert rec["probe_n"] > 2            # NOT the training batch size
    assert rec["probe_batch_size"] == 2  # bs: memory chunk width, unchanged
    assert rec["probe_n_batches"] == 3   # ceil(5 / 2)
    for k in ("final_probe_sd", "final_probe_sem", "noop_floor_sd",
              "noop_floor_sem", "floor_ratio_sem"):
        assert isinstance(rec[k], float) and rec[k] >= 0.0, k
    assert rec["floor_ratio"] == pytest.approx(
        rec["final_probe_loss"] / max(rec["noop_floor"], 1e-6))


def test_sidecar_carries_probe_n_and_uncertainty_on_both_paths(
        tiny_llama, tmp_path):
    """The sidecar (fit_record) records the probe's realised sample size and
    its uncertainty next to the floor/ratio fields — identically from the
    lockstep and the solo save paths — so a recorded verdict (3.0x today, a
    <1.0 criterion later) can be re-read WITH its resolution."""
    model, tok = tiny_llama
    arms = ["lora_r1", "loreft"]         # lockstep + solo
    outcome = t2.train_si_objects(model, tok, SI_ID, _pairs(), arms,
                                  cfg=_probe_cfg(), save_dir=tmp_path)
    assert outcome.ok, {a: o.error for a, o in outcome.failed.items()}
    for arm in arms:
        rec = _sidecar(tmp_path, arm)["fit_record"]
        assert rec["probe_n"] == 3, arm       # clamped to the 3-pair fixture
        assert rec["probe_n_batches"] == 2, arm   # bs 2 -> batches of 2 + 1
        for k in ("final_probe_sd", "final_probe_sem", "noop_floor_sd",
                  "noop_floor_sem", "floor_ratio_sem"):
            assert isinstance(rec[k], float) and rec[k] >= 0.0, (arm, k)
        assert rec["floor_ratio"] == pytest.approx(
            rec["final_probe_loss"] / max(rec["noop_floor"], 1e-6)), arm


def test_probe_n_config_knob_plumbs_through_and_single_batch_is_loud(
        tiny_llama, tmp_path):
    """train_kwargs.default.probe_n reaches both probe paths. A 2-example
    probe still resolves its own spread (the estimator's unit is the
    EXAMPLE, not the batch — 2026-08-24 chunk-invariance fix — so one
    chunk of two examples carries a sem); a 1-example probe records null
    sems — an honest 'unresolvable', never a false 0.0. probe_n < 1
    refuses at config parse."""
    model, tok = tiny_llama
    outcome = t2.train_si_objects(model, tok, SI_ID, _pairs(),
                                  ["lora_r1", "loreft"],
                                  cfg=_probe_cfg(probe_n=2),
                                  save_dir=tmp_path)
    assert outcome.ok, {a: o.error for a, o in outcome.failed.items()}
    for arm in ("lora_r1", "loreft"):
        rec = _sidecar(tmp_path, arm)["fit_record"]
        assert rec["probe_n"] == 2, arm
        assert rec["probe_n_batches"] == 1, arm
        assert isinstance(rec["final_probe_sem"], float), arm
        assert isinstance(rec["floor_ratio_sem"], float), arm

    outcome1 = t2.train_si_objects(model, tok, SI_ID, _pairs(),
                                   ["loreft"], cfg=_probe_cfg(probe_n=1),
                                   save_dir=tmp_path / "n1")
    assert outcome1.ok, {a: o.error for a, o in outcome1.failed.items()}
    rec1 = _sidecar(tmp_path / "n1", "loreft")["fit_record"]
    assert rec1["probe_n"] == 1
    assert rec1["final_probe_sem"] is None
    assert rec1["floor_ratio_sem"] is None

    with pytest.raises(SystemExit, match="probe_n"):
        t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                            cfg=_probe_cfg(probe_n=0))


def test_plan_records_probe_n():
    plan = t2.plan_si(_probe_cfg(probe_n=64), SI_ID, ["lora_r1", "loreft"])
    assert plan["probe_n"] == 64
    assert t2.plan_si(CFG, SI_ID, ["lora_r1"])["probe_n"] == 256  # the default


def test_per_arm_floor_ratio_bar_rules_and_is_recorded(tiny_llama, tmp_path):
    """The bar follows the OBJECTIVE: each arm is
    judged against train_kwargs.<arm>.floor_ratio_max when present, else
    the shared default — on BOTH the lockstep and solo paths — and the bar
    that ruled is recorded in the arm's record. An impossible default with
    a per-arm rescue proves the resolution end to end (and that the guard
    knob never leaks into a trainer's kwargs as a protected unknown)."""
    model, tok = tiny_llama
    cfg = CFG | {"train_kwargs": {
        "default": {"epochs": 2, "seed": 0, "floor_ratio_max": 1e-12},
        "loreft": {"floor_ratio_max": 1e9},          # solo, rescued
    }}
    out_dir = tmp_path / "defenses"
    outcome = t2.train_si_objects(model, tok, SI_ID, _pairs(),
                                  ["lora_r1", "loreft"], cfg=cfg,
                                  save_dir=out_dir)
    # the lockstep arm was judged at the impossible default...
    assert set(outcome.failed) == {"lora_r1"}
    assert outcome.failed["lora_r1"].error_type == "FloorRatioError"
    assert outcome.failed["lora_r1"].detail["floor_ratio_max"] == 1e-12
    assert not object_path_for(out_dir, f"{SI_ID}__lora_r1").exists()
    # ...and the solo arm at its own bar, recorded in the sidecar
    assert set(outcome.objects) == {"loreft"}
    rec = _sidecar(out_dir, "loreft")["fit_record"]
    assert rec["floor_ratio_max"] == 1e9
    assert rec["floor_ratio"] < 1e9

    # per-arm null = guard OFF for that arm while the default still rules
    cfg_off = CFG | {"train_kwargs": {
        "default": {"epochs": 2, "seed": 0, "floor_ratio_max": 1e-12},
        "lora_r1": {"floor_ratio_max": None},
    }}
    out2 = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["lora_r1"],
                               cfg=cfg_off, save_dir=tmp_path / "off")
    assert out2.ok, {a: o.error for a, o in out2.failed.items()}
    assert _sidecar(tmp_path / "off",
                    "lora_r1")["fit_record"]["floor_ratio_max"] is None


def test_lockstep_arm_params_reads_only_the_schedule_keys():
    """train_kwargs.<arm> reaches a LOCKSTEP arm's params through the
    schedule keys ALONE — the guard knob and any trainer kwargs stay out,
    and an arm with no entry keeps its registry defaults exactly."""
    cfg = CFG | {"train_kwargs": {
        "default": {"epochs": 2, "seed": 0},
        "psr_kl_all": {"lr_schedule": "warmup_cosine", "max_grad_norm": 1.0,
                       "floor_ratio_max": 3.0},
    }}
    p = t2.lockstep_arm_params(cfg, "psr_kl_all")
    assert p.lr_schedule == "warmup_cosine"
    assert p.max_grad_norm == 1.0
    assert p.warmup_steps is None        # = train_psr's min(500, total//10)
    assert p.reg_coeff == 0.1            # the registry default rides along
    q = t2.lockstep_arm_params(cfg, "lora_r1")   # no entry: pure defaults
    assert q.lr_schedule is None and q.max_grad_norm is None


def test_driver_schedule_opt_in_reaches_the_arm_and_spares_the_partner(
        tiny_llama, tmp_path):
    """End to end through the driver: config-scheduling psr_kl_all changes
    ITS trained object and its sidecar provenance, while the unscheduled
    group partner (lora_r1 — one of the arms that WORKED in the wave) is
    bitwise identical to the unscheduled run."""
    model, tok = tiny_llama
    arms = ["lora_r1", "psr_kl_all"]

    def run(sub, extra_tk):
        cfg = CFG | {"train_kwargs": {
            "default": {"epochs": 2, "seed": 0, "floor_ratio_max": 1e9},
            **extra_tk}}
        out = t2.train_si_objects(model, tok, SI_ID, _pairs(), arms, cfg=cfg,
                                  save_dir=tmp_path / sub)
        assert out.ok, {a: o.error for a, o in out.failed.items()}
        return out

    plain = run("plain", {})
    sched = run("sched", {"psr_kl_all": {"lr_schedule": "warmup_cosine",
                                         "max_grad_norm": 1.0}})
    # the partner is untouched by the group member's schedule
    for i, (x, y) in enumerate(zip(plain.objects["lora_r1"].trainable,
                                   sched.objects["lora_r1"].trainable)):
        assert torch.equal(x.detach(), y.detach()), f"lora_r1 param {i}"
    # the scheduled arm really departed
    changed = any(
        not torch.equal(x.detach(), y.detach())
        for x, y in zip(plain.objects["psr_kl_all"].trainable,
                        sched.objects["psr_kl_all"].trainable))
    assert changed, "psr_kl_all schedule opt-in never engaged"
    # ...and the sidecar records what each run trained under
    m_plain = _sidecar(tmp_path / "plain", "psr_kl_all")
    m_sched = _sidecar(tmp_path / "sched", "psr_kl_all")
    assert m_plain["lr_schedule"] is None
    assert m_sched["lr_schedule"] == "warmup_cosine"
    assert m_sched["max_grad_norm"] == 1.0
    assert m_sched["warmup_steps"] is None   # auto: min(500, total//10)


def test_two_si_config_pins_probe_n_at_256():
    """The committed config states the probe sample size the wave should
    have had: 256, matching the true-floor measurement (job 187947, whose
    n=256 CIs are +/-7-9% of the mean — resolvable against the ratio
    threshold, where the wave's n=2-4 batch-0 estimates were not)."""
    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "two_si.yaml")
        .read_text(encoding="utf-8"))
    assert cfg["train_kwargs"]["default"]["probe_n"] == 256


def test_epochs_run_records_the_survivor_not_the_candidate(
        tiny_llama, tmp_path, monkeypatch):
    """Same candidate-vs-survivor rule as resolved_batch_size, applied to the
    other kwarg the TypeError-retry can silently drop.

    _safe_registry_train retries without the offending kwarg when a trainer
    lacks **kw. If `epochs` is what gets dropped, the trainer runs its OWN
    epoch loop and the driver no longer knows the budget — so recording the
    CANDIDATE would claim an epoch count the fit never ran. null is the
    honest answer: "not driver-controlled"."""
    from dataclasses import replace

    from sisteer.extraction.reft import LoReFTObject
    from sisteer.methods import ARMS

    model, tok = tiny_llama
    seen = {}

    # No **kw and no `epochs` parameter: the driver's epochs= raises
    # TypeError, the retry drops it, the trainer owns its own loop.
    def no_epochs(model, tok, instruction, pairs, *, layer,
                  batch_size=2, max_len=None, seed=0, log_fn=None):
        seen["batch_size"] = batch_size
        return LoReFTObject(model, [layer])

    monkeypatch.setitem(ARMS, "loreft",
                        replace(ARMS["loreft"], train=no_epochs))
    out = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                              cfg=CFG, save_dir=tmp_path)
    assert out.ok, {a: o.error for a, o in out.failed.items()}
    assert "batch_size" in seen          # the retry kept the width...
    meta = _sidecar(tmp_path, "loreft")
    assert meta["fit_record"]["epochs_run"] is None   # ...and dropped epochs
    # the width DID survive, so it is still recorded rather than nulled —
    # the two kwargs are tracked independently, not as one all-or-nothing flag
    assert meta["resolved_batch_size"] == seen["batch_size"]


# --------------------------------------------------------------------------- #
# T2 (twosi-model-sweeps): planted HPO winners resolve — for BOTH per-model    #
# config shapes — into the EFFECTIVE optimizer settings of every arm, reach    #
# the trainers unchanged (spies), and the banner names value + source. The    #
# fixture configs mirror the per-model files a peer derives; they are         #
# PLANTED here on purpose — the test must not depend on those files landing.  #
# --------------------------------------------------------------------------- #

_SWEEP_ARMS = ["psr_kl_all", "psr_mse_all", "lora_r1_all", "token_faithful",
               "bidpo"]
_LOCKSTEP_SWEEP_ARMS = ("psr_kl_all", "lora_r1_all", "token_faithful")


def _winner(lr, warmup_frac, clip, **extra):
    return {"lr": lr, "lr_schedule": "warmup_cosine",
            "warmup_frac": warmup_frac, "max_grad_norm": clip, **extra}


_MODEL_FIXTURES = {
    "gemma-4-E4B-it": {
        "model": "google/gemma-4-E4B-it",
        "model_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "layer": 20,
        "winners": {
            "psr_kl_all": _winner(3.2e-4, 0.05, 0.7),
            "psr_mse_all": _winner(5.5e-4, 0.15, 1.1),
            "lora_r1_all": _winner(4.1e-4, 0.15, 0.13),
            "token_faithful": _winner(2.5e-4, 0.05, 2.0),
            "bidpo": _winner(1.5e-4, 0.05, 3.0, accum_steps=16),
        },
    },
    "Qwen3.8-27B": {
        "model": "Qwen/Qwen3.8-27B",
        "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "layer": 30,
        "winners": {
            "psr_kl_all": _winner(1.9e-4, 0.15, 0.5),
            "psr_mse_all": _winner(2.8e-4, 0.05, 0.9),
            "lora_r1_all": _winner(6.3e-4, 0.05, 0.21),
            "token_faithful": _winner(1.2e-4, 0.15, 1.4),
            "bidpo": _winner(2.2e-4, 0.15, 5.0, accum_steps=32),
        },
    },
}


def _sweep_cfg(fix: dict, **default_overrides) -> dict:
    return {
        "model": fix["model"], "model_revision": fix["model_revision"],
        "dtype": "bfloat16", "layer": fix["layer"],
        "completion_tokens": 16, "max_len": 256,
        "max_len_by_si": {SI_ID: 256},
        "class_a_arms": list(_SWEEP_ARMS),
        "class_c_arms": ["psr_kl_all", "psr_mse_all", "lora_r1_all",
                         "bidpo"],
        "si_ids": [SI_ID],
        "train_kwargs": {
            "default": {"epochs": 2, "seed": 0, "holdout_frac": 0.0,
                        **default_overrides},
            **{arm: dict(w) for arm, w in fix["winners"].items()},
        },
    }


@pytest.mark.parametrize("model_key", sorted(_MODEL_FIXTURES))
def test_planted_hpo_winners_are_the_effective_settings(model_key):
    """For every arm of both per-model configs, the RESOLVED optimizer
    settings equal the planted winner — value AND source — so a winner can
    never be silently dropped the way psr_kl_all's 5e-4 once was (3dd9e36
    class; warmup_frac is the new member of that class)."""
    fix = _MODEL_FIXTURES[model_key]
    cfg = _sweep_cfg(fix)
    for arm, winner in fix["winners"].items():
        lockstep = arm in _LOCKSTEP_SWEEP_ARMS
        eff = t2.effective_optimizer_settings(cfg, arm, lockstep=lockstep)
        for k in ("lr", "lr_schedule", "warmup_frac", "max_grad_norm"):
            v, src = eff[k]
            assert v == winner[k], (model_key, arm, k, v, winner[k])
            assert src == "config", (model_key, arm, k, src)
        if lockstep:
            # the winner must ALSO survive the params merge (the allow-list)
            p = t2.lockstep_arm_params(cfg, arm)
            assert p.lr == winner["lr"]
            assert p.warmup_frac == winner["warmup_frac"]
            assert p.max_grad_norm == winner["max_grad_norm"]
            assert p.warmup_steps is None
        if arm == "bidpo":
            v, src = eff["accum_steps"]
            assert v == winner["accum_steps"] and src == "config"
    # nothing in either fixture is a key its arm would silently ignore
    t2.refuse_silently_dropped_kwargs(cfg, _SWEEP_ARMS)


def test_warmup_frac_is_on_the_lockstep_allowlist():
    """The 3dd9e36 allow-list class, pinned for the new key: dropping
    warmup_frac from _LOCKSTEP_OPTIMIZER_KEYS makes a configured winner a
    refusal (not a silent drop, thanks to refuse_silently_dropped_kwargs),
    so this asserts BOTH halves stay wired."""
    assert "warmup_frac" in t2._LOCKSTEP_OPTIMIZER_KEYS
    p = t2.lockstep_arm_params(
        {"train_kwargs": {"psr_kl_all": {
            "lr_schedule": "warmup_cosine", "warmup_frac": 0.15}}},
        "psr_kl_all")
    assert p.warmup_frac == 0.15


def test_banner_carries_winner_values_and_sources():
    fix = _MODEL_FIXTURES["gemma-4-E4B-it"]
    cfg = _sweep_cfg(fix)
    lines = t2.optimizer_banner(cfg, SI_ID, _SWEEP_ARMS)
    kl = next(l for l in lines if " psr_kl_all " in l)
    assert f"lr={fix['winners']['psr_kl_all']['lr']} (config)" in kl
    assert "warmup_frac=0.05 (config)" in kl
    assert "max_grad_norm=0.7 (config)" in kl
    dpo = next(l for l in lines if " bidpo " in l)
    assert "accum_steps=16 (config)" in dpo
    assert "[solo]" in dpo and "[lockstep]" in kl


def test_refuse_fires_on_an_ignored_key_in_a_model_config():
    fix = _MODEL_FIXTURES["Qwen3.8-27B"]
    cfg = _sweep_cfg(fix)
    cfg["train_kwargs"]["psr_kl_all"] = dict(
        cfg["train_kwargs"]["psr_kl_all"], probe_temperature=2.0)
    with pytest.raises(SystemExit) as e:
        t2.refuse_silently_dropped_kwargs(cfg, _SWEEP_ARMS)
    assert "probe_temperature" in str(e.value)


def test_trainer_spies_receive_the_winner_unchanged(tiny_llama, monkeypatch):
    """Stub each underlying trainer and assert the ACTUAL call carries the
    winner: lr/schedule/clip verbatim for every arm; warmup as the
    RESOLVED step count for a trainer whose seam is warmup_steps (psr mse:
    frac x this fit's own budget), as the verbatim fraction for a trainer
    with a native warmup_frac seam (bidpo); accum_steps for bidpo.
    An unsupported PROTECTED key fails BEFORE any training."""
    import math
    import sisteer.extraction.lockstep as lockstep_mod
    from sisteer.eval.train_budget import resolve_batch_size

    model, tok = tiny_llama
    fix = _MODEL_FIXTURES["gemma-4-E4B-it"]
    cfg = _sweep_cfg(fix, monitors=False, floor_ratio_max=None)
    cfg["layer"] = 1  # the tiny fixture has 2 layers
    pairs = [Pair(prompt=p, pos=f"Reponse {i}", neg=f"Reply {i}",
                  instruction_id=SI_ID, requirement_a="strict")
             for i, p in enumerate(["Greet the reader.", "Count to two.",
                                    "Name a planet.", "Describe the sea."])]

    seen = {}

    def lockstep_spy(mdl, tk, arms, examples, **kw):
        for a in arms:
            seen[a.name] = a.params
        return {a.name: [] for a in arms}

    monkeypatch.setattr(lockstep_mod, "train_lockstep", lockstep_spy)

    def solo_spy_factory(arm, ret):
        def spy(mdl, tk, instruction, prs, *, layer, log_fn=None, **kw):
            seen[arm] = dict(kw, n_pairs=len(prs))
            return ret
        return spy

    from sisteer.extraction.psr import PSRObject
    monkeypatch.setitem(ARMS, "psr_mse_all", replace(
        ARMS["psr_mse_all"],
        train=solo_spy_factory("psr_mse_all",
                               PSRObject(model, [0, 1],
                                         train_direction=False))))
    from sisteer.methods import VectorObject
    from sisteer.vectors import SteeringVector
    monkeypatch.setitem(ARMS, "bidpo", replace(
        ARMS["bidpo"],
        train=solo_spy_factory("bidpo", VectorObject(SteeringVector(
            vectors={1: torch.zeros(model.config.hidden_size)},
            method="bidpo", model_name="", instruction_id=SI_ID)))))

    out = t2.train_si_objects(model, tok, SI_ID, pairs, _SWEEP_ARMS,
                              cfg=cfg, sentinel=_sentinel(tok))
    assert out.ok, {a: o.error for a, o in out.failed.items()}

    for arm in _LOCKSTEP_SWEEP_ARMS:
        w = fix["winners"][arm]
        p = seen[arm]
        assert (p.lr, p.lr_schedule, p.warmup_frac, p.max_grad_norm) == \
            (w["lr"], "warmup_cosine", w["warmup_frac"],
             w["max_grad_norm"]), arm
        assert p.warmup_steps is None, arm

    # psr mse: the fraction resolved against THIS fit's own step budget
    w = fix["winners"]["psr_mse_all"]
    got = seen["psr_mse_all"]
    bs = resolve_batch_size(cfg, SI_ID, "psr_mse_all")
    total = 2 * math.ceil(got["n_pairs"] / bs)
    assert got["lr"] == w["lr"]
    assert got["lr_schedule"] == "warmup_cosine"
    assert got["max_grad_norm"] == w["max_grad_norm"]
    assert "warmup_frac" not in got
    assert got["warmup_steps"] == max(1, round(w["warmup_frac"] * total))

    # bidpo: fraction + accumulation reach the trainer VERBATIM (its own
    # loop resolves the fraction — accumulation changes its step count)
    w = fix["winners"]["bidpo"]
    got = seen["bidpo"]
    assert got["lr"] == w["lr"]
    assert got["warmup_frac"] == w["warmup_frac"]
    assert got["max_grad_norm"] == w["max_grad_norm"]
    assert got["accum_steps"] == w["accum_steps"]

    # an unsupported PROTECTED key fails the arm BEFORE training: the spy
    # for psr_mse_all must never fire when the config carries a typo'd key
    seen.clear()

    def exploding(mdl, tk, instruction, prs, *, layer, log_fn=None, **kw):
        if "definitely_not_a_kwarg" in kw:
            raise TypeError("train_psr() got an unexpected keyword argument "
                            "'definitely_not_a_kwarg'")
        raise AssertionError("training ran despite the unknown kwarg")

    monkeypatch.setitem(ARMS, "psr_mse_all",
                        replace(ARMS["psr_mse_all"], train=exploding))
    bad = _sweep_cfg(fix, monitors=False, floor_ratio_max=None)
    bad["layer"] = 1
    bad["train_kwargs"]["psr_mse_all"] = dict(
        bad["train_kwargs"]["psr_mse_all"], definitely_not_a_kwarg=1)
    out2 = t2.train_si_objects(model, tok, SI_ID, pairs, ["psr_mse_all"],
                               cfg=bad)
    assert out2.arms["psr_mse_all"].status == "failed"
    assert "definitely_not_a_kwarg" in out2.arms["psr_mse_all"].error


# --------------------------------------------------------------------------- #
# T14 (twosi-model-sweeps): atomic publish, COMPLETE marker, pre-publication   #
# audit, marker-validated resume — and T9's driver half: the non-finite abort  #
# records failed and never ships; concurrent group arms write distinct,        #
# parseable per-cell metrics.jsonl mirrors.                                    #
# --------------------------------------------------------------------------- #


def test_publish_writes_complete_marker_and_marker_gates_resume(
        tiny_llama, tmp_path):
    """A published cell carries the COMPLETE marker (written LAST) whose
    manifest hashes every required publication file (F16) and whose fit
    fingerprint names this exact request (F15); deleting the marker makes
    the cell INCOMPLETE — resume RE-RUNS it (T14's interruption shape:
    killed after the object rename, before the marker)."""
    model, tok = tiny_llama
    out = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                              cfg=CFG, save_dir=tmp_path)
    assert out.ok
    cell = tmp_path / f"{SI_ID}__loreft"
    marker_path = cell / t2.COMPLETE_MARKER_NAME
    assert marker_path.exists()
    marker = json.loads(marker_path.read_text())
    assert marker["cell_id"] == f"{SI_ID}__loreft"
    obj_path = cell / marker["object_file"]
    assert obj_path.exists()
    assert t2._sha256_file(obj_path) == marker["object_sha256"]
    assert (cell / "metrics.jsonl").exists()  # the per-cell mirror published
    # F16: the marker is a MANIFEST — every required file, each with the
    # sha256 of the bytes actually on disk (object hash NOT optional).
    assert set(marker["files"]) == {marker["object_file"],
                                    "train_budget_meta.json",
                                    "metrics.jsonl"}
    for name, sha in marker["files"].items():
        assert t2._sha256_file(cell / name) == sha, name
    # F15: the stored fingerprint is exactly this request's identity
    assert marker["fit_fingerprint"] == _fp("loreft")
    assert t2.validated_complete(tmp_path, f"{SI_ID}__loreft",
                                 _fp("loreft")) == obj_path

    # marker gone (interrupted publish) -> the object alone is NOT done
    marker_path.unlink()
    assert t2.validated_complete(tmp_path, f"{SI_ID}__loreft",
                                 _fp("loreft")) is None
    again = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                                cfg=CFG, save_dir=tmp_path)
    assert again.arms["loreft"].status == "trained"  # re-ran, not skipped
    assert marker_path.exists()  # and re-published


def test_object_present_but_sidecar_missing_is_incomplete(tiny_llama,
                                                          tmp_path):
    model, tok = tiny_llama
    out = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                              cfg=CFG, save_dir=tmp_path)
    assert out.ok
    cell = tmp_path / f"{SI_ID}__loreft"
    (cell / "train_budget_meta.json").unlink()
    assert t2.validated_complete(tmp_path, f"{SI_ID}__loreft",
                                 _fp("loreft")) is None
    again = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                                cfg=CFG, save_dir=tmp_path)
    assert again.arms["loreft"].status == "trained"
    assert (cell / "train_budget_meta.json").exists()


def test_validated_complete_rejects_sha_mismatch(tiny_llama, tmp_path):
    model, tok = tiny_llama
    t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                        cfg=CFG, save_dir=tmp_path)
    cell = tmp_path / f"{SI_ID}__loreft"
    marker = json.loads((cell / t2.COMPLETE_MARKER_NAME).read_text())
    (cell / marker["object_file"]).write_bytes(b"torn or swapped bytes")
    assert t2.validated_complete(tmp_path, f"{SI_ID}__loreft",
                                 _fp("loreft")) is None


_AUDIT_FIXTURE_N = [0]


def _audit_fixture(tmp_path, *, rows, fit_record=None, with_object=True,
                   with_sidecar=True):
    from sisteer.methods import object_artifact_name

    _AUDIT_FIXTURE_N[0] += 1
    d = tmp_path / f"cellfix{_AUDIT_FIXTURE_N[0]}"
    d.mkdir()
    if with_object:
        (d / object_artifact_name("lora_r1")).write_bytes(b"obj")
    if with_sidecar:
        (d / "train_budget_meta.json").write_text(json.dumps({
            "positions_at_fit": "all",
            "fit_record": fit_record if fit_record is not None
            else {"floor_ratio_max": 1.0, "floor_ratio": 0.4}}))
    (d / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n" if rows else "")
    return d


_GOOD_ROWS = [
    {"step": 0, "train/loss": 0.5, "probe/residual_ratio": 0.9,
     "probe/displacement_ratio": 0.2},
    {"step": 1, "train/loss": 0.4},
    {"step": 1, "final/residual_ratio": 0.8,
     "final/displacement_ratio": 0.3},
]


def test_audit_refuses_missing_keys_cadence_and_nonfinite(tmp_path):
    """The pre-publication audit (T14): missing final dual-KL values,
    missing floor verdict, a probe-cadence gap, a non-finite metric, or a
    missing per-cell mirror each REFUSE publication, naming the problem."""
    from sisteer.extraction.fitmon import ProbeCadence

    cad = ProbeCadence(probe_every=1000, dense_every=1, dense_until=10,
                       log_every=1)
    # the good shape passes and reports a summary (cadence checked in its
    # own passing test below; _GOOD_ROWS deliberately has a step-1 gap)
    d = _audit_fixture(tmp_path, rows=_GOOD_ROWS)
    summary = t2.audit_cell(d, arm="lora_r1")
    assert summary["metrics_rows"] == 3

    with pytest.raises(t2.CellAuditError, match="final dual-KL"):
        t2.audit_cell(_audit_fixture(tmp_path, rows=_GOOD_ROWS[:2]),
                      arm="lora_r1")
    with pytest.raises(t2.CellAuditError, match="floor verdict"):
        t2.audit_cell(_audit_fixture(tmp_path, rows=_GOOD_ROWS,
                                     fit_record={"note": "no bar"}),
                      arm="lora_r1")
    with pytest.raises(t2.CellAuditError, match="floor_ratio is None"):
        t2.audit_cell(_audit_fixture(
            tmp_path, rows=_GOOD_ROWS,
            fit_record={"floor_ratio_max": 1.0, "floor_ratio": None}),
            arm="lora_r1")
    with pytest.raises(t2.CellAuditError, match="cadence gap"):
        # step 1 is cadence-due (dense_every=1) but carries no probe record
        t2.audit_cell(_audit_fixture(tmp_path, rows=_GOOD_ROWS),
                      arm="lora_r1", cadence=cad, monitors_on=True)
    with pytest.raises(t2.CellAuditError, match="non-finite"):
        bad = [dict(r) for r in _GOOD_ROWS]
        bad[1]["train/loss"] = float("nan")
        t2.audit_cell(_audit_fixture(tmp_path, rows=bad), arm="lora_r1")
    with pytest.raises(t2.CellAuditError, match="object artifact missing"):
        t2.audit_cell(_audit_fixture(tmp_path, rows=_GOOD_ROWS,
                                     with_object=False), arm="lora_r1")
    with pytest.raises(t2.CellAuditError, match="training-loss"):
        t2.audit_cell(_audit_fixture(
            tmp_path, rows=[_GOOD_ROWS[2]]), arm="lora_r1")
    # monitors off: only the artifact + sidecar are required
    t2.audit_cell(_audit_fixture(tmp_path, rows=[]), arm="lora_r1",
                  monitors_on=False)


def test_audit_cadence_passes_when_probes_ride_every_due_step(tmp_path):
    from sisteer.extraction.fitmon import ProbeCadence

    rows = [
        {"step": 0, "train/loss": 0.5, "probe/residual_ratio": 1.0,
         "probe/displacement_ratio": 0.1},
        {"step": 1, "train/loss": 0.4, "probe/residual_ratio": 0.9,
         "probe/displacement_ratio": 0.2},
        {"step": 1, "final/residual_ratio": 0.9,
         "final/displacement_ratio": 0.2},
    ]
    t2.audit_cell(_audit_fixture(tmp_path, rows=rows), arm="lora_r1",
                  cadence=ProbeCadence(probe_every=1000, dense_every=1,
                                       dense_until=10, log_every=1))


def test_nonfinite_metric_records_failed_and_never_ships(tiny_llama,
                                                         tmp_path,
                                                         monkeypatch):
    """T9's abort, end to end through the driver: a non-finite probe ratio
    aborts the arm (TrainingDiverged), records it FAILED, and publishes
    nothing — no cell dir, no COMPLETE marker; the temp dir stays for
    inspection."""
    from sisteer.extraction import fitmon

    def nan_measure(self, model, obj=None):
        out = {"probe/noop_floor": 0.1}
        if obj is not None:
            out.update({"probe/residual_ratio": float("nan"),
                        "probe/displacement_ratio": float("nan")})
        return out

    monkeypatch.setattr(fitmon.KLRatioProbe, "measure", nan_measure)
    model, tok = tiny_llama
    out = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["lora_r1"],
                              cfg=CFG, save_dir=tmp_path)
    rec = out.arms["lora_r1"]
    assert rec.status == "failed"
    assert rec.error_type == "TrainingDiverged"
    assert "residual_ratio" in rec.error
    assert not (tmp_path / f"{SI_ID}__lora_r1").exists()  # never published
    assert t2.validated_complete(tmp_path, f"{SI_ID}__lora_r1",
                                 _fp("lora_r1")) is None
    tmps = list(tmp_path.glob(f"{SI_ID}__lora_r1.tmp-*"))
    assert tmps, "the temp cell dir must stay for inspection"
    assert rec.detail and "diverged" in rec.detail


def test_concurrent_group_arms_write_distinct_cell_metrics(tiny_llama,
                                                           tmp_path):
    """T9: two lockstep arms interleave inside ONE process (the driver's
    concurrency unit; the scheduler's arm tasks are separate processes
    with separate dirs by construction) — each cell's metrics.jsonl must
    parse, carry the monitoring-contract keys, and never carry the
    partner's records."""
    model, tok = tiny_llama
    arms = ["lora_r1", "psr_kl_all"]
    out = t2.train_si_objects(model, tok, SI_ID, _pairs(), arms, cfg=CFG,
                              save_dir=tmp_path)
    assert out.ok, {a: o.error for a, o in out.failed.items()}
    for arm, partner in (("lora_r1", "psr_kl_all"),
                         ("psr_kl_all", "lora_r1")):
        lines = (tmp_path / f"{SI_ID}__{arm}" / "metrics.jsonl") \
            .read_text().splitlines()
        rows = [json.loads(l) for l in lines if l.strip()]
        assert rows, arm
        keys = {k for r in rows for k in r}
        assert "train/loss" in keys, arm
        assert "train/grad_norm_postclip" in keys, arm
        assert any(k.startswith("probe/residual_ratio") for k in keys), arm
        assert "final/residual_ratio" in keys, arm
        assert any(k.startswith(f"lockstep/{arm}/") for k in keys), arm
        crossed = [k for k in keys if f"/{partner}/" in k
                   or k.startswith(f"{partner}/")]
        assert not crossed, (arm, crossed)
    # the sidecars record the bucketing + monitoring provenance
    meta = json.loads((tmp_path / f"{SI_ID}__lora_r1" /
                       "train_budget_meta.json").read_text())
    assert meta["length_bucketed"] is True
    assert meta["monitors"] is True


def test_bidpo_midfit_probes_ride_the_trainer_seam(tiny_llama, tmp_path):
    """End to end through the REAL train_bidpo (its probe_cb(step, obj) +
    probe_every seam): the driver wires the monitor into the trainer, so
    the published bidpo cell carries MID-FIT dual-KL probe records — not
    just the end-of-fit measurement — and the sidecar records the seam."""
    model, tok = tiny_llama
    pairs = [Pair(prompt=p, pos=f"Reponse {i}", neg=f"Reply {i}",
                  instruction_id=SI_ID, requirement_a="strict")
             for i, p in enumerate(["Greet the reader.", "Count to two.",
                                    "Name a planet.", "Describe the sea."])]
    cfg = {"layer": 1, "completion_tokens": 16, "max_len": 256,
           "max_len_by_si": {SI_ID: 256},
           "train_kwargs": {"default": {
               "epochs": 1, "seed": 0, "holdout_frac": 0.0,
               "floor_ratio_max": None,
               # the tiny fit runs ~1 optimizer step; pull the dense probe
               # grid down so the mid-fit seam demonstrably fires
               "dense_every": 1, "dense_until": 10, "log_every": 1}},
           "si_ids": [SI_ID]}
    tee = tmp_path / f"{SI_ID}__metrics.jsonl"   # the production tee path
    out = t2.train_si_objects(
        model, tok, SI_ID, pairs, ["bidpo"], cfg=cfg, save_dir=tmp_path,
        log_fn=t2.make_metrics_log_fn(jsonl_path=tee, arms=["bidpo"]))
    assert out.ok, {a: o.error for a, o in out.failed.items()}
    rows = [json.loads(l) for l in
            (tmp_path / f"{SI_ID}__bidpo" / "metrics.jsonl")
            .read_text().splitlines() if l.strip()]
    keys = {k for r in rows for k in r}
    assert "probe/residual_ratio" in keys      # mid-fit, via the seam
    assert "final/residual_ratio" in keys      # end-of-fit, via the driver
    assert any(k.startswith("bidpo/") for k in keys)  # trainer telemetry teed
    detail = out.arms["bidpo"].detail
    assert detail["midfit_probes"] == "trainer_probe_cb"
    # The SHARED stream (W&B + the reports tee) of a distillation-lane arm
    # keeps fitmon's teacher-relative vocabulary and its top-level mirror,
    # untouched by the CE-lane rename (which keys on CE_ARMS membership).
    shared = {k for l in tee.read_text().splitlines() if l.strip()
              for k in json.loads(l)}
    assert {"probe/bidpo/residual_ratio", "probe/bidpo/residual_kl",
            "probe/bidpo/heldout_loss", "final/bidpo/residual_ratio",
            "kl_ratio_vs_unsteered"} <= shared
    assert not any("displacement_kl_vs_base" in k for k in shared)


# --------------------------------------------------------------------------- #
# Adversarial review 2026-08-25 (F14/F15/F16 + F7's driver half + --profile):  #
# a published cell must be LOADABLE (format sidecar at the loadable name),     #
# resume must be bound to the fit's IDENTITY (fingerprint) and to the bytes    #
# of every published file (manifest), the solo psr arms ride the trainer's     #
# probe seam, and `train --profile <dir>` captures a TensorBoard trace.        #
# --------------------------------------------------------------------------- #


def _bidpo_pairs():
    return [Pair(prompt=p, pos=f"Reponse {i}", neg=f"Reply {i}",
                 instruction_id=SI_ID, requirement_a="strict")
            for i, p in enumerate(["Greet the reader.", "Count to two.",
                                   "Name a planet.", "Describe the sea."])]


def _tiny_fit_cfg(**default_overrides):
    return {"layer": 1, "completion_tokens": 16, "max_len": 256,
            "max_len_by_si": {SI_ID: 256},
            "train_kwargs": {"default": {
                "epochs": 1, "seed": 0, "holdout_frac": 0.0,
                "floor_ratio_max": None,
                "dense_every": 1, "dense_until": 10, "log_every": 1,
                **default_overrides}},
            "si_ids": [SI_ID]}


def test_bidpo_cell_publishes_loadable_artifact(tiny_llama, tmp_path):
    """End to end: a REAL published bidpo cell carries its safetensors
    container's format sidecar at the LOADABLE name (object.json — the name
    SteeringVector.load and methods.load_object derive), never stranded at
    object.safetensors.json (the file-level .tmp rename bug that published
    COMPLETE cells which then failed at eval-load), the COMPLETE manifest
    checksums it, and the cell reloads through load_object — the exact
    loader the eval path uses."""
    from sisteer.methods import VectorObject, load_object

    model, tok = tiny_llama
    pairs = _bidpo_pairs()
    cfg = _tiny_fit_cfg()
    out = t2.train_si_objects(model, tok, SI_ID, pairs, ["bidpo"], cfg=cfg,
                              save_dir=tmp_path)
    assert out.ok, {a: o.error for a, o in out.failed.items()}
    cell = tmp_path / f"{SI_ID}__bidpo"
    assert (cell / "object.safetensors").exists()
    assert (cell / "object.json").exists()               # the loadable name
    assert not (cell / "object.safetensors.json").exists()  # the stranding
    assert not list(cell.glob("*.tmp"))
    marker = json.loads((cell / t2.COMPLETE_MARKER_NAME).read_text())
    assert set(marker["files"]) == {"object.safetensors", "object.json",
                                    "train_budget_meta.json",
                                    "metrics.jsonl"}
    for name, sha in marker["files"].items():
        assert t2._sha256_file(cell / name) == sha, name
    obj = load_object(object_path_for(tmp_path, f"{SI_ID}__bidpo"),
                      model=model, tok=tok)
    assert isinstance(obj, VectorObject)
    # ...and the fingerprint-bound resume gate accepts the loadable cell
    fp = t2.cell_fit_fingerprint(
        cfg, SI_ID, "bidpo", positions="all",
        pool_sha256=t2.pool_content_sha256(pairs),
        config_sha=t2._cfg_dict_sha(cfg))
    assert t2.validated_complete(tmp_path, f"{SI_ID}__bidpo", fp) is not None


def test_save_object_legacy_inplace_moves_the_format_sidecar(tiny_llama,
                                                             tmp_path):
    """F14's legacy half: the in-place save path (direct callers/tests, no
    temp cell dir) must move BOTH the safetensors container AND its format
    sidecar into place — sidecar first, so the primary's existence always
    implies a loadable artifact set — leaving no misnamed or temp files."""
    from sisteer.methods import VectorObject, load_object
    from sisteer.models import hidden_size
    from sisteer.vectors import SteeringVector

    model, tok = tiny_llama
    sv = SteeringVector(vectors={1: torch.zeros(hidden_size(model))},
                        method="distill_soft", model_name="m",
                        instruction_id=SI_ID)
    t2._save_object(VectorObject(sv), tmp_path, SI_ID, "vector",
                    {"completion_tokens": 4}, 16, {"batch_size": 1},
                    model=model, tok=tok)
    cell = object_path_for(tmp_path, f"{SI_ID}__vector").parent
    assert (cell / "object.safetensors").exists()
    assert (cell / "object.json").exists()
    assert not (cell / "object.safetensors.json").exists()
    assert not list(cell.glob("*.tmp"))
    loaded = load_object(cell / "object.safetensors", model=model, tok=tok)
    assert isinstance(loaded, VectorObject)


def test_audit_refuses_safetensors_cell_missing_format_sidecar(tmp_path):
    """The pre-publication audit requires the container's format sidecar
    whenever the artifact's CONTENT is a safetensors container — the
    loader's own dispatch, never the arm name (F14): a cell that would
    fail at load must never publish COMPLETE, while a torch blob (or a
    routing test's opaque stand-in bytes) is never asked for one."""
    from safetensors.torch import save_file

    from sisteer.methods import object_artifact_name

    d = tmp_path / "cell"
    d.mkdir()
    save_file({"layer_1": torch.zeros(4)},
              str(d / object_artifact_name("bidpo")))  # a REAL container
    (d / "train_budget_meta.json").write_text(json.dumps(
        {"positions_at_fit": "all", "fit_record": {"floor_ratio_max": None}}))
    (d / "metrics.jsonl").write_text("")
    with pytest.raises(t2.CellAuditError, match="format sidecar"):
        t2.audit_cell(d, arm="bidpo", monitors_on=False)
    (d / "object.json").write_text("{}")
    t2.audit_cell(d, arm="bidpo", monitors_on=False)  # passes with it
    # a .pt arm has no container sidecar and must not be asked for one...
    (d / object_artifact_name("lora_r1")).write_bytes(b"obj")
    t2.audit_cell(d, arm="lora_r1", monitors_on=False)
    # ...and neither is a test double's non-container bytes behind a
    # safetensors arm name (the routing tests' _FakeObj shape)
    (d / object_artifact_name("bidpo")).write_bytes(b"SYNTHETIC-object")
    (d / "object.json").unlink()
    t2.audit_cell(d, arm="bidpo", monitors_on=False)


def test_cell_fit_fingerprint_binds_every_identity_axis():
    """The fingerprint changes with each of the eight identity fields —
    and reproduces the shared utility byte-for-byte from the driver's
    resolved optimizer settings, so no dimension can be quietly dropped."""
    from sisteer.extraction.fit_fingerprint import fit_fingerprint

    base = _fp("loreft")
    assert base == _fp("loreft")                       # deterministic
    assert base != _fp("lora_r1")                      # arm
    assert base != _fp("loreft", positions="response")  # position leg (F15)
    assert base != _fp("loreft", pairs=_pairs()[:2])   # pool
    assert base != _fp("loreft", cfg=CFG | {"model": "other/model"})  # model
    eff = t2.effective_optimizer_settings(CFG, "loreft", lockstep=False)
    manual = fit_fingerprint({
        "model": None, "model_revision": None, "si_id": SI_ID,
        "arm": "loreft", "positions": "all",
        "pool_sha256": t2.pool_content_sha256(_pairs()),
        # The teacher leg rides inside the optimizer mapping (re-tune@32:
        # a top-K-head cell is different work than an online-teacher one).
        "optimizer": {**{k: v for k, (v, _s) in eff.items()},
                      "teacher": "online"},
        "config_sha": t2._cfg_dict_sha(CFG)})
    assert manual == base
    # ...and the teacher leg is its own identity axis.
    assert base != t2.cell_fit_fingerprint(
        CFG, SI_ID, "loreft", positions="all",
        pool_sha256=t2.pool_content_sha256(_pairs()),
        config_sha=t2._cfg_dict_sha(CFG), teacher="topk:0123456789abcdef")


def test_resume_is_identity_bound_not_name_bound(tiny_llama, tmp_path):
    """F15: a COMPLETE marker stands in ONLY for the exact requested fit —
    a different pool, position leg or model identity re-runs the cell, and
    a pre-fingerprint marker (no identity recorded) re-runs by design."""
    model, tok = tiny_llama
    out = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                              cfg=CFG, save_dir=tmp_path)
    assert out.ok
    cell_id = f"{SI_ID}__loreft"
    assert t2.validated_complete(tmp_path, cell_id,
                                 _fp("loreft")) is not None
    other_pairs = _pairs()[:2]
    assert t2.validated_complete(
        tmp_path, cell_id, _fp("loreft", pairs=other_pairs)) is None
    # the F15 shape itself: an all-token marker must never be accepted as
    # the response-leg cell
    assert t2.validated_complete(
        tmp_path, cell_id, _fp("loreft", positions="response")) is None
    assert t2.validated_complete(
        tmp_path, cell_id,
        _fp("loreft", cfg=CFG | {"model": "other/model"})) is None
    # end to end: a run over a different pool RETRAINS instead of skipping
    again = t2.train_si_objects(model, tok, SI_ID, other_pairs, ["loreft"],
                                cfg=CFG, save_dir=tmp_path)
    assert again.arms["loreft"].status == "trained"
    # a marker predating the fingerprint/manifest re-runs (None never
    # matches; an un-fingerprinted artifact cannot be proven to be this
    # work)
    marker_path = tmp_path / cell_id / t2.COMPLETE_MARKER_NAME
    marker = json.loads(marker_path.read_text())
    marker.pop("fit_fingerprint")
    marker.pop("files")
    marker_path.write_text(json.dumps(marker))
    assert t2.validated_complete(
        tmp_path, cell_id, _fp("loreft", pairs=other_pairs)) is None


def test_resume_refuses_lost_metrics_post_publish(tiny_llama, tmp_path):
    """F16: the COMPLETE manifest covers metrics.jsonl and the provenance
    sidecar — losing or corrupting either post-publish re-runs the cell
    instead of resuming over a record that can no longer be verified."""
    model, tok = tiny_llama
    out = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                              cfg=CFG, save_dir=tmp_path)
    assert out.ok
    cell = tmp_path / f"{SI_ID}__loreft"
    (cell / "metrics.jsonl").unlink()
    assert t2.validated_complete(tmp_path, f"{SI_ID}__loreft",
                                 _fp("loreft")) is None
    again = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                                cfg=CFG, save_dir=tmp_path)
    assert again.arms["loreft"].status == "trained"
    assert (cell / "metrics.jsonl").exists()  # re-published whole
    (cell / "train_budget_meta.json").write_text("{not json")
    assert t2.validated_complete(tmp_path, f"{SI_ID}__loreft",
                                 _fp("loreft")) is None


def test_store_identity_one_root_one_model_and_leg(tiny_llama, tmp_path):
    """F15's namespacing rule at the write moment: one objects root holds
    one (model, revision, position leg). The first writer claims the store;
    a different model or leg refuses loudly instead of mixing (two
    per-model configs share the relative defenses_dir default, and
    identity-bound resume would otherwise RE-RUN one model's wave over the
    other's cells)."""
    model, tok = tiny_llama
    store = tmp_path / "cells"
    t2.assert_store_identity(store, CFG, "all")
    t2.assert_store_identity(store, CFG, "all")  # idempotent re-claim
    with pytest.raises(SystemExit, match="one objects root"):
        t2.assert_store_identity(store, CFG | {"model": "other/model"},
                                 "all")
    with pytest.raises(SystemExit, match="one objects root"):
        t2.assert_store_identity(store, CFG, "response")
    # end to end: the fit refuses BEFORE training anything
    out = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                              cfg=CFG, save_dir=store)
    assert out.ok
    with pytest.raises(SystemExit, match="one objects root"):
        t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                            cfg=CFG | {"model": "other/model"},
                            save_dir=store)


def test_class_c_out_must_differ_from_the_class_a_store(tmp_path):
    """F15's Class C guard, second half: --out PRESENT is not enough — the
    Class A store's own path passed as the Class C --out re-creates the
    exact all-token-marker-as-response-cell collision, so it refuses in
    milliseconds (before any pair or model load)."""
    class_a = tmp_path / "cells"
    cfg = CFG | {"model": "unused", "dtype": "float32",
                 "defenses_dir": str(class_a)}
    args = argparse.Namespace(si=SI_ID, arms="", pairs="never-loaded.jsonl",
                              out=str(class_a), overwrite=False,
                              positions="response")
    with pytest.raises(SystemExit, match="Class A store"):
        t2.cmd_train(args, cfg)


def test_scheduler_resume_gate_is_fingerprint_bound(tiny_llama, tmp_path,
                                                    monkeypatch):
    """F15's second resume gate: the parallel scheduler skips ONLY a cell
    whose COMPLETE fingerprint matches the requested fit (same
    _cli_fit_identity derivation the arm task stores) — change the pool
    file and the same-named cell re-runs."""
    import yaml

    model, tok = tiny_llama
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(CFG | {"data_parallel": {"gpus": 1}}))
    cfg = t2._load_cfg(str(cfg_path))
    pairs_file = tmp_path / "pool.jsonl"
    pairs_file.write_text("\n".join(json.dumps(
        {"prompt": p.prompt, "pos": p.pos, "neg": p.neg,
         "instruction_id": p.instruction_id}) for p in _pairs()) + "\n")
    out_dir = tmp_path / "cells"
    args = argparse.Namespace(
        config=str(cfg_path), si=SI_ID, pairs=str(pairs_file),
        out=str(out_dir), overwrite=False, positions="", token_cache="off",
        expect_config_sha="", bidpo_harmful_pairs="", bidpo_benign_pairs="",
        profile="")
    # publish ONE cell exactly the way an arm task would (same identity)
    pool_sha, config_sha = t2._cli_fit_identity(args, cfg)
    out = t2.train_si_objects(model, tok, SI_ID,
                              t2._load_pairs(str(pairs_file)), ["loreft"],
                              cfg=cfg, save_dir=out_dir,
                              pool_sha256=pool_sha, config_sha=config_sha)
    assert out.ok
    monkeypatch.setenv("SISTEER_FIT_GPU_POOL", "0")

    def no_spawn(*a, **kw):
        raise RuntimeError("spawned an arm task")

    monkeypatch.setattr(t2, "_spawn_arm_task", no_spawn)
    # matching identity -> skip, never spawn, exit 0
    assert t2.run_parallel_fit(args, cfg, ["loreft"], out_dir) == 0
    # the pool file changes -> the marker no longer matches -> re-runs
    with open(pairs_file, "a") as fh:
        fh.write(json.dumps({"prompt": "Name a river.", "pos": "La Seine",
                             "neg": "The Thames",
                             "instruction_id": SI_ID}) + "\n")
    with pytest.raises(RuntimeError, match="spawned"):
        t2.run_parallel_fit(args, cfg, ["loreft"], out_dir)


def test_psr_solo_midfit_probes_ride_the_trainer_seam(tiny_llama, tmp_path):
    """F7's driver half, end to end through the REAL train_psr: the solo
    psr arms (psr_mse_all here — lockstep-incompatible, so it always takes
    this path) receive probe_cb/probe_every through the F7 trainer seam,
    so the published cell carries MID-FIT dual-KL probe records like the
    lockstep arms — and the record names the seam, never the removed
    unavailable_no_trainer_seam fallback."""
    model, tok = tiny_llama
    out = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["psr_mse_all"],
                              cfg=_tiny_fit_cfg(), save_dir=tmp_path)
    assert out.ok, {a: (o.error_type, o.error) for a, o in out.failed.items()}
    rows = [json.loads(l) for l in
            (tmp_path / f"{SI_ID}__psr_mse_all" / "metrics.jsonl")
            .read_text().splitlines() if l.strip()]
    keys = {k for r in rows for k in r}
    assert "probe/residual_ratio" in keys      # mid-fit, via the seam
    assert "final/residual_ratio" in keys      # end-of-fit, via the driver
    assert "psr/mse_loss" in keys              # trainer telemetry teed
    detail = out.arms["psr_mse_all"].detail
    assert detail["midfit_probes"] == "trainer_probe_cb"


def test_psr_probe_seam_regression_fails_loudly(tiny_llama, monkeypatch):
    """The other half of removing the psr fallback: the seam is PROTECTED
    for psr arms, so a trainer that regresses out of probe_cb/probe_every
    fails the arm with the TypeError — it must never silently degrade back
    to end-of-fit-only probes under a fresh unavailable_no_trainer_seam
    record."""
    def seamless(model, tok, instruction, pairs, *, layer, log_fn=None,
                 epochs=6, batch_size=2, max_len=768, seed=0,
                 teacher_prefix_cache=False):
        raise AssertionError("training must not start without the seam")

    monkeypatch.setitem(ARMS, "psr_mse_all",
                        replace(ARMS["psr_mse_all"], train=seamless))
    model, tok = tiny_llama
    out = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["psr_mse_all"],
                              cfg=CFG)
    rec = out.arms["psr_mse_all"]
    assert rec.status == "failed"
    assert rec.error_type == "TypeError"
    assert "probe_cb" in rec.error or "probe_every" in rec.error


def test_profile_flag_captures_a_tensorboard_trace(tiny_llama, tmp_path,
                                                   monkeypatch):
    """`train --profile <dir>` (pod_pilot.sh step 7): a torch.profiler is
    entered around the training, ticked once per OPTIMIZER-STEP record
    (the pilot's unified is_step_line classifier), configured with
    tune_arm.py's exact schedule, sinking a TensorBoard trace into the
    given DIRECTORY, and exited even as training ends."""
    import sisteer.models
    import torch as _torch

    model, tok = tiny_llama
    monkeypatch.setattr(sisteer.models, "load_model_and_tokenizer",
                        lambda *a, **kw: (model, tok))
    pairs_file = tmp_path / "pairs.jsonl"
    pairs_file.write_text("\n".join(json.dumps(
        {"prompt": p.prompt, "pos": p.pos, "neg": p.neg,
         "instruction_id": p.instruction_id}) for p in _pairs()))
    tb_dir = tmp_path / "tb_trace"
    captured, sched_kw, handler = {}, {}, {}

    class _FakeProf:
        steps = 0
        entered = exited = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *a):
            self.exited = True

        def step(self):
            self.steps += 1

    fake = _FakeProf()
    monkeypatch.setattr(_torch.profiler, "profile",
                        lambda **kw: captured.update(kw) or fake)
    monkeypatch.setattr(_torch.profiler, "schedule",
                        lambda **kw: sched_kw.update(kw) or "SCHED")
    monkeypatch.setattr(_torch.profiler, "tensorboard_trace_handler",
                        lambda d: handler.update(dir=d) or "HANDLER")
    args = argparse.Namespace(si=SI_ID, arms="loreft", pairs=str(pairs_file),
                              out=str(tmp_path / "cells"), overwrite=False,
                              profile=str(tb_dir), token_cache="off")
    t2.cmd_train(args, CFG | {"model": "unused", "dtype": "float32"})
    assert fake.entered and fake.exited
    assert fake.steps >= 1                      # ticked on real step records
    assert sched_kw == {"wait": 20, "warmup": 3, "active": 12, "repeat": 1}
    assert handler["dir"] == str(tb_dir)
    assert captured["schedule"] == "SCHED"
    assert captured["on_trace_ready"] == "HANDLER"
    assert tb_dir.is_dir()                      # created for the handler


def test_train_parser_accepts_profile_and_scheduler_forwards_it(
        tmp_path, monkeypatch):
    """The pilot's exact invocation parses (`train ... --profile <dir>`,
    a trace DIRECTORY), and the parallel scheduler forwards the flag into
    every arm task so a profiled parallel wave traces its workers."""
    import subprocess

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("layer: 1\n")
    seen = {}
    monkeypatch.setattr(t2, "cmd_train",
                        lambda args, cfg: seen.update(vars(args)))
    monkeypatch.setattr(sys, "argv", [
        "train_two_si_objects.py", "train", "--config", str(cfg_path),
        "--si", SI_ID, "--arms", "psr_kl_all", "--pairs", "pool.jsonl",
        "--out", str(tmp_path / "cells"), "--overwrite",
        "--profile", str(tmp_path / "tb_trace")])
    t2.main()
    assert seen["profile"] == str(tmp_path / "tb_trace")
    assert seen["overwrite"] is True

    captured = {}

    class _P:
        def __init__(self, cmd, stdout=None, stderr=None, env=None,
                     cwd=None):
            captured["cmd"] = cmd

        def poll(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", _P)
    args = argparse.Namespace(
        config="configs/two_si.yaml", si=SI_ID, pairs="pool.jsonl", out="",
        overwrite=False, positions="", token_cache="",
        bidpo_harmful_pairs="", bidpo_benign_pairs="", expect_config_sha="",
        profile=str(tmp_path / "tb_trace"))
    t2._spawn_arm_task(args, "loreft", 1, ["0"], tmp_path)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--profile") + 1] == str(tmp_path / "tb_trace")


def test_sidecar_persists_per_example_probe_values_on_both_paths(
        tiny_llama, tmp_path):
    """The per-example payload (probe_active_values / probe_noop_values /
    probe_selected_sha8 / probe_per_family) reaches the REAL sidecar and the
    fit report from BOTH write paths — the lockstep group and the solo
    registry trainers — additively next to the untouched aggregates, so any
    two cells probed on the same pool are paired-comparable from disk
    (aligned by the sha8 key, correlated-error sem) without re-probing
    finished objects on a GPU. Both paths select by the same sha256 rank
    over the same pool, so their sha8 lists MATCH — the cross-cell pairing
    precondition, asserted rather than assumed."""
    model, tok = tiny_llama
    arms = ["lora_r1", "loreft"]         # lockstep + solo
    outcome = t2.train_si_objects(model, tok, SI_ID, _pairs(), arms,
                                  cfg=_probe_cfg(), save_dir=tmp_path)
    assert outcome.ok, {a: o.error for a, o in outcome.failed.items()}

    recs = {}
    for arm in arms:
        rec = _sidecar(tmp_path, arm)["fit_record"]   # a real JSON round trip
        recs[arm] = rec
        missing = [k for k in PROBE_ESTIMATE_RECORD_KEYS.values()
                   if k not in rec]
        assert not missing, (arm, missing)
        n = rec["probe_n"]
        assert len(rec["probe_active_values"]) == n, arm
        assert len(rec["probe_noop_values"]) == n, arm
        assert len(rec["probe_selected_sha8"]) == n, arm
        assert all(isinstance(s, str) and len(s) == 8
                   for s in rec["probe_selected_sha8"]), arm
        # the arrays reproduce the recorded aggregates
        assert sum(rec["probe_active_values"]) / n == pytest.approx(
            rec["final_probe_loss"], rel=1e-5), arm
        assert sum(rec["probe_noop_values"]) / n == pytest.approx(
            rec["noop_floor"], rel=1e-5), arm
        # neither path passes family labels: a stated null, never a KeyError
        assert rec["probe_per_family"] is None, arm
        # this fixture carves no held-out pairs: the heldout key family is
        # ABSENT (not null, not a crash) — the graceful-absence contract
        assert "heldout_probe_loss" not in rec, arm
        assert "heldout_probe_active_values" not in rec, arm

    # same pool + same sha256-rank selection => identical populations, in
    # order — what lets a reader pair the two cells example by example
    assert (recs["lora_r1"]["probe_selected_sha8"]
            == recs["loreft"]["probe_selected_sha8"])

    # the fit REPORT carries the same payload (arms.<arm>.detail)
    report = json.loads(outcome.report_path.read_text())
    for arm in arms:
        det = report["arms"][arm]["detail"]
        assert det["probe_active_values"] == recs[arm]["probe_active_values"]
        assert det["probe_selected_sha8"] == recs[arm]["probe_selected_sha8"]


def test_single_example_probe_persists_length_one_arrays(
        tiny_llama, tmp_path):
    """probe_n=1 (the degenerate floor of the knob): the payload is a
    length-1 array next to the null sems — a stated thinness, never a
    crash in the writer after an hours-long fit."""
    model, tok = tiny_llama
    outcome = t2.train_si_objects(model, tok, SI_ID, _pairs(), ["loreft"],
                                  cfg=_probe_cfg(probe_n=1),
                                  save_dir=tmp_path)
    assert outcome.ok, {a: o.error for a, o in outcome.failed.items()}
    rec = _sidecar(tmp_path, "loreft")["fit_record"]
    assert rec["probe_n"] == 1
    assert len(rec["probe_active_values"]) == 1
    assert len(rec["probe_noop_values"]) == 1
    assert len(rec["probe_selected_sha8"]) == 1
    assert rec["final_probe_sem"] is None
