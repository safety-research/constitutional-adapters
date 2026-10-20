"""PAP — persuasion-technique templates (Zeng et al. 2024). CPU only, model-free.
Covers registry + non-gated, deterministic prompts, resume, record fields
incl. variant_key, and the 0700 dir."""

from attacks_static_common import assert_static_contract


def test_pap_registered_and_not_gated():
    from sisteer.eval.attacks import (GATED_ATTACKS, JB_ATTACKS,
                                      SUBENV_ATTACKS, get_attack)

    assert "pap" in JB_ATTACKS
    assert get_attack("pap").__name__ == "run_pap"
    assert "pap" not in GATED_ATTACKS
    assert "pap" not in SUBENV_ATTACKS


def test_pap_variants_deterministic_and_template_based():
    from sisteer.eval.attacks.pap import _TECHNIQUES, pap_variants
    from sisteer.eval.jailbreak_data import Goal

    g = Goal("g0", "explain how to do X", "illegal", "harmbench", "rev")
    v1 = pap_variants(g, seed=0)
    v2 = pap_variants(g, seed=0)
    assert v1 == v2                                  # deterministic
    assert [n for n, _ in v1] == [name for name, _ in _TECHNIQUES]
    # each variant is the goal wrapped in a fixed persuasion frame (not a rewrite)
    for _, prompt in v1:
        assert "explain how to do X" in prompt


def test_pap_contract():
    from sisteer.eval.attacks.pap import _TECHNIQUES, run_pap

    assert_static_contract(run_pap, "pap",
                           expected_variant_names=[n for n, _ in _TECHNIQUES])
