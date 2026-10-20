"""Verification of the five-family TRAINING registration at base e2d7fea.

e2d7fea ("Register codeattack, renellm and grok420 for TRAINING; drop the
leakage gate") is claimed to register all five training families
(bon, pasttense, codeattack, renellm, grok420) plus wildjailbreak_harmful at
StrongREJECT strictly > 0.5 with GOAL-level holdout. This file verifies
that claim and closes the coverage gaps the tip left.

WHAT IS ALREADY COVERED ELSEWHERE (this file deliberately does NOT duplicate):
  * five family keys resolve in JB_ATTACKS / STATIC_FAMILIES / get_attack, and
    train_families/static_families are a subset of the registry
    -> tests/test_attacks_registered_for_training.py
  * grok420 TEMPLATE render/fold/slot-refusal/byte-exactness + the two variants
    on a STRING goal -> tests/test_attacks_godmode.py,
    tests/test_attacks_registered_for_training.py
  * the leakage gate is gone, not dormant
    -> tests/test_attacks_registered_for_training.py
  * goal-level held_in/held_out split derived from the store's category_group
    (seed 0 / frac 0.33) -> tests/test_store_goal_split.py

WHAT THIS FILE ADDS (genuine gaps):
  * the goal-level-holdout REGIME flags in configs/two_si.yaml
    (family_disjointness false, threshold 0.5) and the train<->eval family
    relationship they imply (the five headroom families on BOTH sides).
  * the replay/persist/resume static contract exercised END-TO-END for
    codeattack and grok420 (only the string-goal / template path was covered).
  * whether prompt construction splices the harmful goal TEXT rather than the
    Goal dataclass repr -- codeattack does (passes); grok420 does NOT (a live
    src bug, pinned as a strict xfail below, NOT fixed here).
  * renellm's ASSET DEPENDENCY: it is NOT self-serviceable from a goal list --
    it reads a frozen corpus built by an LLM rewriter, and loud-skips without
    one. The three-scenario stage-2 nesting is the only deterministic half.

CPU-only, offline, synthetic placeholder text only (privacy rule).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from attacks_static_common import StubDefense, assert_static_contract, goals
from godmode_standin import godmode_standin  # noqa: F401  (grok420 renders the invented G0DM0D3 stand-in)
from sisteer.eval.attacks import JB_ATTACKS
from sisteer.eval.jailbreak_data import Goal

REPO = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
POOL = CFG["pool"]

# The five families the spec (Fixed 11) and the e2d7fea commit call the
# "headroom" strategies -- each trained on AND evaluated on under goal-level
# holdout. wildjailbreak_harmful is the undefended harmful CORPUS, not a
# transform (no JB_ATTACKS entry by design).
FIVE_HEADROOM = ["bon", "pasttense", "codeattack", "renellm", "grok420"]
CORPUS = "wildjailbreak_harmful"


# -- the goal-level-holdout regime in the config -----------------------------


def test_goal_level_holdout_regime_flags():
    """Goal-level holdout is in effect: family disjointness OFF, and the
    acceptance threshold is the preregistered strict > 0.5 (NOT the later
    0.75 directive)."""
    assert POOL["family_disjointness"] is False, \
        "family-level holdout must be OFF under the goal-level regime"
    assert POOL["threshold"] == 0.5


def test_the_five_headroom_families_plus_corpus_are_in_train_families():
    """The claim's five + wildjailbreak_harmful must all be trainable. NOTE
    train_families is deliberately a SUPERSET (all twelve static families) --
    this pins the five-headroom+corpus SUBSET, not equality."""
    train = set(POOL["train_families"])
    for fam in FIVE_HEADROOM:
        assert fam in train, f"{fam!r} missing from train_families"
    assert CORPUS in train


def test_five_headroom_families_appear_on_both_train_and_eval_sides():
    """The point of dropping family-level holdout: a family may -- and the
    five informative ones must -- appear on BOTH sides, evaluated against
    goals the fit never saw."""
    train, ev = set(POOL["train_families"]), set(POOL["eval_families"])
    for fam in FIVE_HEADROOM:
        assert fam in train and fam in ev, \
            f"{fam!r} must be on both train and eval sides (goal-level regime)"


def test_wildjailbreak_harmful_is_a_corpus_not_a_transform():
    """It ships already-adversarial prompts replayed directly, so it has no
    JB_ATTACKS entry -- being in train_families without one is BY DESIGN, and
    is exactly why test_attacks_registered excludes it from the resolve check."""
    assert CORPUS in POOL["train_families"]
    assert CORPUS not in JB_ATTACKS


# -- the static replay/persist/resume contract, end-to-end -------------------


def test_codeattack_replay_persist_resume_contract():
    """codeattack replays through run_static like the five existing static
    transforms: ordered variants, private 0700 dir, per-goal progress, resume
    without dupes, and the pool-builder variant_key convention."""
    from sisteer.eval.attacks.codeattack import run_codeattack

    assert_static_contract(
        run_codeattack, "codeattack",
        expected_variant_names=["docstring_completion", "list_fill",
                                "stack_simulation"])


def test_grok420_replay_persist_resume_contract(godmode_standin):
    """grok420 replays through run_static with both G0DM0D3 variants in a
    fixed order. (This exercises the RUN path; the goal-text SPLICE is checked
    separately below.)"""
    from sisteer.eval.attacks.grok420 import run_grok420

    assert_static_contract(
        run_grok420, "grok420",
        expected_variant_names=["grok420", "hermes_fast"])


# -- prompt construction splices the goal TEXT, not the dataclass repr --------

_SENTINEL = "SENTINEL_HARMFUL_BEHAVIOUR"


def _repr_leaked(prompt: str) -> bool:
    """A Goal dataclass repr splice leaves these tokens in the prompt; a clean
    goal.text splice does not."""
    return "Goal(goal_id=" in prompt or "goal_id='" in prompt


def test_codeattack_splices_goal_text_not_the_dataclass_repr():
    """CONTROL: codeattack uses goal.text, so a real Goal object yields a
    prompt carrying only the behaviour text -- no dataclass boilerplate."""
    from sisteer.eval.attacks.codeattack import codeattack_variants

    g = Goal("g0", _SENTINEL, "illegal", "harmbench", "rev")
    for _name, prompt in codeattack_variants(g):
        assert _SENTINEL in prompt
        assert not _repr_leaked(prompt), "codeattack leaked the Goal repr"


def test_grok420_splices_goal_text_not_the_dataclass_repr(godmode_standin):
    """grok420 must splice ONLY the harmful behaviour text -- the same
    discipline codeattack keeps. It currently splices the Goal repr instead
    (goal_id/category/dataset/revision boilerplate around the text), a silent
    degradation the string-only template tests never caught."""
    from sisteer.eval.attacks.grok420 import grok420_variants

    g = Goal("g0", _SENTINEL, "illegal", "harmbench", "rev")
    for _name, prompt in grok420_variants(g):
        assert _SENTINEL in prompt
        assert not _repr_leaked(prompt), \
            "grok420 spliced the Goal dataclass repr, not goal.text"


# -- renellm asset dependency: a frozen corpus, not a goal-list transform -----


def test_renellm_loud_skips_without_a_frozen_corpus(monkeypatch, tmp_path):
    """renellm is NOT self-serviceable from a goal list: run_renellm needs a
    pre-built frozen corpus and LOUD-SKIPS (never silently degrades) when the
    corpus is absent -- documenting that our per-model pools cannot build
    renellm prompts without first running an LLM-rewriter build. The repository
    ships the paper's corpus, so "absent" here means no site copy AND no
    shipped copy (a tree that lost data/heldout_corpora/renellm.jsonl)."""
    from sisteer.eval.attacks import renellm

    monkeypatch.setenv("SISTEER_HELDOUT_CORPUS_DIR", str(tmp_path))
    monkeypatch.setattr(renellm, "SHIPPED_CORPUS", "data/heldout_corpora/absent-for-this-test.jsonl")
    renellm._CORPUS_CACHE.clear()
    res = renellm.run_renellm(StubDefense(), goals(2), out_dir=tmp_path / "run")
    assert res.get("skipped") is True
    assert res["family"] == "renellm"
    assert res["n_goals_run"] == 0


def test_renellm_variants_empty_without_a_frozen_corpus(monkeypatch, tmp_path):
    """The variants function is a pure corpus LOOKUP: with no corpus (no site
    copy, no shipped copy) it returns an empty list rather than constructing
    anything from the goal alone."""
    from sisteer.eval.attacks import renellm

    monkeypatch.setenv("SISTEER_HELDOUT_CORPUS_DIR", str(tmp_path))
    monkeypatch.setattr(renellm, "SHIPPED_CORPUS", "data/heldout_corpora/absent-for-this-test.jsonl")
    renellm._CORPUS_CACHE.clear()
    g = Goal("g0", _SENTINEL, "illegal", "harmbench", "rev")
    assert renellm.renellm_variants(g) == []


def test_renellm_stage2_nesting_is_deterministic_over_three_scenarios():
    """Only ReNeLLM stage 2 (scenario nesting) is a deterministic template;
    stage 1 (the rewrite) needs an LLM at build time. Pin the three scenarios
    and the loud failure on an unknown one."""
    from sisteer.eval.attacks.renellm import _SCENARIOS, _nest

    assert _SCENARIOS == ("code_completion", "table_filling",
                          "text_continuation")
    rewritten = "REWRITTEN_REQUEST_SENTINEL"
    for scenario in _SCENARIOS:
        out1 = _nest(scenario, rewritten)
        out2 = _nest(scenario, rewritten)
        assert out1 == out2, f"{scenario} nesting is not deterministic"
        assert rewritten in out1
    with pytest.raises(ValueError, match="unknown ReNeLLM scenario"):
        _nest("not_a_scenario", rewritten)
