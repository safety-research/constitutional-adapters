"""The sweep's HELD-OUT leg: registry wiring + Condition/TSV selection.

Covers requirement 5 (a HELD_OUT_FAMILIES frozenset distinct from the
STATIC_FAMILIES as a DISPATCH classification — one name resolves to exactly
one scaffolding — and the sweep driver able to select held-out-only).
NOTE (2026-08-28): family-level holdout is retired; the goal split is the
holdout, and a family here may also appear in a config's train_families.
"""

import tempfile
from pathlib import Path

import pytest

import sisteer.eval.attacks as A
from sisteer.eval.sweep import (
    HELD_OUT_FAMILIES,
    Condition,
    load_conditions,
)


# --------------------------------------------------------------------------- #
# registry                                                                     #
# --------------------------------------------------------------------------- #

def test_one_name_one_scaffolding():
    # a REGISTRY-consistency guard (one name must resolve to exactly one
    # scaffolding), NOT the retired train/eval family-disjointness rule
    dispatch = A.STATIC_FAMILIES | A.SUBENV_ATTACKS | {"bon", "gcg"}
    assert not (A.HELD_OUT_FAMILIES & dispatch), \
        "a family name is registered under two scaffoldings"


def test_held_out_families_are_registered_and_dispatchable():
    for name in A.HELD_OUT_FAMILIES:
        assert name in A.JB_ATTACKS, f"{name} not in JB_ATTACKS"
        assert callable(A.get_attack(name))
        spec = A.get_heldout_spec(name)
        assert spec.name == name and spec.build_kind in ("deterministic", "rewriter", "ingested")


def test_heldout_specs_covers_every_held_out_family():
    assert set(A.heldout_specs()) == set(A.HELD_OUT_FAMILIES)


def test_config_eval_families_are_the_implemented_held_out_set():
    # the six static families in the sweep's held-out leg. codeattack and
    # renellm moved to the static TRAINING dispatch at the phase-C join
    # (goal-level holdout) while REMAINING eval families, so the sweep-leg
    # set now splits across the two dispatch tables.
    expected = {"artprompt", "cipherchat", "codeattack", "drattack",
                "multilingual", "renellm"}
    assert expected <= (A.HELD_OUT_FAMILIES | A.STATIC_FAMILIES)
    assert set(HELD_OUT_FAMILIES) == expected  # the sweep leg = the config six


# --------------------------------------------------------------------------- #
# Condition / TSV                                                              #
# --------------------------------------------------------------------------- #

def test_condition_default_held_out_and_stamp():
    from sisteer.eval.sweep import EVAL_DEFAULT_HELD_OUT_FAMILIES

    c = Condition(label="base", cell="none")
    # default = the wave set (2026-08-28: codeattack + renellm — the two
    # held-out-leg families with real headroom); the full universe stays
    # selectable (test_tsv_can_select_a_single_held_out_family).
    assert c.held_out == EVAL_DEFAULT_HELD_OUT_FAMILIES
    # held-out families use NO attacker (batched generation), like the static leg
    assert c.held_out_stamp(object_sha="x")["attacker_engine"] == "none"


def test_condition_rejects_unknown_held_out_family():
    with pytest.raises(ValueError, match="held_out leg has unknown"):
        Condition(label="bad", cell="none", held_out=("not_a_family",))


def test_tsv_backcompat_and_held_out_only_selection():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cond.tsv"
        p.write_text(
            # an existing 11-column row (no held_out col) still parses; held_out
            # defaults to the full set
            "legacy\tnone\t\tall\t*\t-\t*\thf\tnone\thf\t\n"
            # held-out ONLY: static/adaptive/benign skipped, held_out=*
            "heldonly\tnone__si\t\tall\t-\t-\t-\thf\tnone\thf\t\t*\n"
        )
        conds = {c.label: c for c in load_conditions(p)}
    from sisteer.eval.sweep import EVAL_DEFAULT_HELD_OUT_FAMILIES

    assert conds["legacy"].held_out == EVAL_DEFAULT_HELD_OUT_FAMILIES
    ho = conds["heldonly"]
    assert ho.static == () and ho.adaptive == () and ho.benign == ()
    assert ho.held_out == EVAL_DEFAULT_HELD_OUT_FAMILIES


def test_tsv_can_select_a_single_held_out_family():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cond.tsv"
        p.write_text(
            "one\tnone\t\tall\t-\t-\t-\thf\tnone\thf\t\tartprompt,cipherchat\n")
        c = load_conditions(p)[0]
    assert c.held_out == ("artprompt", "cipherchat")
