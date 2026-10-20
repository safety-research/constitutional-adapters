"""GCG-by-replay — replay banked suffixes without re-running the optimiser.
CPU only, model-free (stub defense — replay never touches the model). Covers
registry + non-gated, deterministic prompts, resume, record fields incl.
variant_key, 0700 dir, and the GCG cost-class reuse (zero marginal optimiser
cost, forward_cap-exempt)."""

from attacks_static_common import (StubDefense, assert_static_contract, goals,
                                    read_jsonl)


def test_gcg_replay_registered_and_not_gated():
    from sisteer.eval.attacks import (GATED_ATTACKS, JB_ATTACKS,
                                      SUBENV_ATTACKS, get_attack)

    assert "gcg_replay" in JB_ATTACKS
    assert get_attack("gcg_replay").__name__ == "run_gcg_replay"
    # replay runs no attacker model, no PyRIT, no sub-env.
    assert "gcg_replay" not in GATED_ATTACKS
    assert "gcg_replay" not in SUBENV_ATTACKS


def test_gcg_replay_variants_deterministic_and_splice():
    from sisteer.eval.attacks.gcg_replay import gcg_replay_variants
    from sisteer.eval.jailbreak_data import Goal

    g = Goal("g0", "the goal", "illegal", "harmbench", "rev")
    suffixes = ("AAA", "BBB")
    v1 = gcg_replay_variants(g, seed=0, suffixes=suffixes)
    v2 = gcg_replay_variants(g, seed=0, suffixes=suffixes)
    assert v1 == v2                                  # deterministic
    assert [n for n, _ in v1] == ["bank0", "bank1"]  # distinct positional keys
    assert v1[0][1] == "the goal AAA" and v1[1][1] == "the goal BBB"


def test_gcg_replay_contract_default_bank():
    from sisteer.eval.attacks.gcg_replay import DEFAULT_SUFFIX_BANK, run_gcg_replay

    names = [f"bank{i}" for i in range(len(DEFAULT_SUFFIX_BANK))]
    assert_static_contract(run_gcg_replay, "gcg_replay",
                           expected_variant_names=names)


def test_gcg_replay_cost_class_zero_and_exempt(tmp_path):
    from sisteer.eval.attacks.gcg_replay import run_gcg_replay

    result = run_gcg_replay(StubDefense(), goals(1), out_dir=tmp_path / "run",
                            suffixes=("XXX", "YYY", "ZZZ"))
    # replay runs NO optimiser -> zero marginal optimiser cost, reusing the GCG
    # cost-class field names so gcg_optimizer_cost reads it uniformly.
    assert result["n_optimizer_forwards_total"] == 0
    rec = result["per_goal"][0]
    assert rec["n_optimizer_steps"] == 0
    assert rec["n_optimizer_forwards"] == 0
    assert rec["replayed"] is True
    assert rec["n_suffixes"] == 3
    assert rec["capped"] is False                    # forward_cap-exempt
    # three real replays, counted at the defense boundary (not the optimiser).
    assert rec["n_target_forwards"] == 3
    rows = read_jsonl((tmp_path / "run") / "responses.jsonl")
    assert [r["variant_name"] for r in rows] == ["bank0", "bank1", "bank2"]
