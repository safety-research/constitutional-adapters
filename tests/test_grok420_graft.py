"""grok420 graft onto the matrixfill lineage (held-out machinery preserved).

grok420 is a STATIC, seed-independent semantic-inversion family (G0DM0D3).
This asserts it is wired for generation AND that grafting it did not disturb
the held-out families (codeattack/renellm) or their leakage gate.
"""
from godmode_standin import godmode_standin  # noqa: F401  (grok420 renders the invented G0DM0D3 stand-in)
from sisteer.eval.attacks import (
    JB_ATTACKS, STATIC_FAMILIES, HELD_OUT_FAMILIES, get_attack,
)


def test_grok420_registered_as_static_training_family():
    assert "grok420" in JB_ATTACKS
    assert JB_ATTACKS["grok420"] == (
        "sisteer.eval.attacks.grok420", "run_grok420")
    assert "grok420" in STATIC_FAMILIES
    assert callable(get_attack("grok420"))


def test_grok420_variants_are_deterministic_and_seed_independent(godmode_standin):
    from sisteer.eval.attacks.grok420 import grok420_variants
    a = grok420_variants("SENTINEL")
    b = grok420_variants("SENTINEL", seed=7)
    assert a == b, "static transforms must not depend on the seed"
    assert [k for k, _ in a] == ["grok420", "hermes_fast"]
    for _key, prompt in a:
        assert "SENTINEL" in prompt
        assert "{QUERY}" not in prompt and "{Z}" not in prompt


def test_graft_did_not_break_the_heldout_families():
    # codeattack/renellm moved to the static TRAINING dispatch at the
    # phase-C join (goal-level holdout); the one-name-one-scaffolding
    # dispatch guard (import-time in attacks/__init__) still holds —
    # grok420 must not have leaked into the held-out scaffolding.
    assert {"codeattack", "renellm"} <= STATIC_FAMILIES
    assert "grok420" not in HELD_OUT_FAMILIES
    assert not (HELD_OUT_FAMILIES & STATIC_FAMILIES)
