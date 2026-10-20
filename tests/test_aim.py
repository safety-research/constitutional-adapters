"""AIM-style roleplay — template persona wrapper. CPU only, model-free. Covers
registry + non-gated, deterministic persona prompts, resume, record fields
incl. variant_key, and the 0700 dir."""

from attacks_static_common import assert_static_contract


def test_aim_registered_and_not_gated():
    from sisteer.eval.attacks import (GATED_ATTACKS, JB_ATTACKS,
                                      SUBENV_ATTACKS, get_attack)

    assert "aim" in JB_ATTACKS
    assert get_attack("aim").__name__ == "run_aim"
    assert "aim" not in GATED_ATTACKS
    assert "aim" not in SUBENV_ATTACKS


def test_aim_variants_deterministic_persona_wrapper():
    from sisteer.eval.attacks.aim import aim_variants
    from sisteer.eval.jailbreak_data import Goal

    g = Goal("g0", "the request text", "illegal", "harmbench", "rev")
    v1 = aim_variants(g, seed=0)
    v2 = aim_variants(g, seed=0)
    assert v1 == v2                                  # deterministic
    assert [n for n, _ in v1] == ["aim", "unbound_persona"]
    for _, prompt in v1:
        assert "the request text" in prompt          # goal spliced into persona


def test_aim_contract():
    from sisteer.eval.attacks.aim import run_aim

    assert_static_contract(run_aim, "aim",
                           expected_variant_names=["aim", "unbound_persona"])
