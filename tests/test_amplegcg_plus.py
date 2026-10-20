"""AmpleGCG-plus — learned suffix generator in a transformers<4.56 sub-env,
suffixes replayed statically. CPU only, model-free: the sub-env generation is a
monkeypatched seam. Covers registry (non-gated, sub-env axis), the available
path's static contract, the skip-with-loud-reason path, and the own cost class +
forward_cap exemption."""

import sys

from attacks_static_common import (StubDefense, assert_static_contract, goals,
                                    read_jsonl)


def _fake_gen(goal, seed):
    """Deterministic stand-in for the sub-env AmpleGCG suffix generator."""
    variants = [("amplegcg0", f"{goal.text} SUF0"),
                ("amplegcg1", f"{goal.text} SUF1")]
    cost = {"n_generator_forwards": 128, "n_prompts_generated": 2}
    return variants, cost


def test_amplegcg_registered_not_gated_but_subenv():
    from sisteer.eval.attacks import (GATED_ATTACKS, JB_ATTACKS,
                                      SUBENV_ATTACKS, get_attack)

    assert "amplegcg_plus" in JB_ATTACKS
    assert get_attack("amplegcg_plus").__name__ == "run_amplegcg"
    assert "amplegcg_plus" not in GATED_ATTACKS
    assert "amplegcg_plus" in SUBENV_ATTACKS


def test_amplegcg_available_contract():
    from sisteer.eval.attacks.amplegcg import run_amplegcg

    assert_static_contract(
        run_amplegcg, "amplegcg_plus",
        expected_variant_names=["amplegcg0", "amplegcg1"],
        run_kwargs={"_available": lambda: (True, "/fake/py"),
                    "_generate": _fake_gen})


def test_amplegcg_skips_loudly_when_subenv_absent(tmp_path, capsys):
    from sisteer.eval.attacks.amplegcg import run_amplegcg

    out = tmp_path / "run"
    result = run_amplegcg(StubDefense(), goals(2), out_dir=out,
                          _available=lambda: (False, "generator checkpoint not configured"))
    assert result["skipped"] is True
    assert result["family"] == "amplegcg_plus"
    assert result["n_goals_run"] == 0
    assert not (out / "responses.jsonl").exists()
    assert "SKIPPED" in capsys.readouterr().err


def test_amplegcg_own_cost_class_and_exempt(tmp_path):
    from sisteer.eval.attacks.amplegcg import run_amplegcg

    result = run_amplegcg(StubDefense(), goals(1), out_dir=tmp_path / "run",
                          _available=lambda: (True, "/fake/py"),
                          _generate=_fake_gen)
    rec = result["per_goal"][0]
    assert rec["cost_class"] == "amplegcg_optimizer_cost"
    assert rec["n_generator_forwards"] == 128
    assert rec["n_prompts_generated"] == 2
    assert rec["capped"] is False
    assert rec["n_target_forwards"] == 2
    assert rec["n_generator_forwards"] != rec["n_target_forwards"]
    assert result["amplegcg_optimizer_cost"] == 128
    rows = read_jsonl((tmp_path / "run") / "responses.jsonl")
    assert [r["variant_name"] for r in rows] == ["amplegcg0", "amplegcg1"]


def test_amplegcg_availability_check_real():
    from sisteer.eval.attacks.amplegcg import MODEL_ENV_VAR, SUBENV_ENV_VAR
    from sisteer.eval.attacks.subenv_common import subenv_available

    ok, reason = subenv_available(SUBENV_ENV_VAR, "/nonexistent/python",
                                  model_env=MODEL_ENV_VAR)
    assert ok is False and SUBENV_ENV_VAR in reason
    ok2, reason2 = subenv_available(SUBENV_ENV_VAR, sys.executable,
                                    model_env=MODEL_ENV_VAR, model_path=None)
    assert ok2 is False and "checkpoint" in reason2
