"""The canonical IFEval battery — the IFEval seed rule.

The static / over-refusal sampled re-run's derivation is:
``seed = sha256(f"{seed_base}|{trial}|{readout}|{item_key}")[:4]`` big-endian,
``seed_base`` 20260909, range [0, 2^32), NO modulus — which differs from the
Inspect rule's ``mod 2^31-1``, so IFEval carries its OWN rule string
(``SEED_RULE_IFEVAL``) hashed ONLY by runs that file ``inspect/ifeval``.
"""

from __future__ import annotations

import hashlib

import pytest

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tiers import inspect_served as isv

IFEVAL = C.inspect_task(C.IFEVAL_TASK_NAME)


def _rerun_derivation(seed_base: int, trial: int, readout: str, item_key: str) -> int:
    """The sampled re-run's ``sampling_seed`` (branch sampled-rerun,
    ``served_capability_run.py``), re-derived inline."""
    key = f"{seed_base}|{trial}|{readout}|{item_key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def test_the_derivation_is_the_sampled_reruns_and_reproduces_its_literal():
    # the re-run's own test literal (sampled-rerun tests/test_sampled_controls.py)
    assert _rerun_derivation(20260909, 0, "xstest", "xst0000-aaaaaaaa") == 3515763156
    # the plan's IFEval example values (P1)
    assert isv.ifeval_sample_seed("1000", 0) == 3028447421 == _rerun_derivation(20260909, 0, "ifeval", "1000")
    assert isv.ifeval_sample_seed("1000", 1) == 3178680996 == _rerun_derivation(20260909, 1, "ifeval", "1000")
    assert C.IFEVAL_SEED_BASE == 20260909 and C.IFEVAL_SEED_READOUT == "ifeval" == C.CONTROL_IFEVAL
    for item in ("1000", "1001", "2314", "3891"):
        for trial in (0, 1):
            assert isv.ifeval_sample_seed(item, trial) == _rerun_derivation(
                C.IFEVAL_SEED_BASE, trial, C.IFEVAL_SEED_READOUT, item)


def test_the_range_is_2_to_the_32_with_no_modulus_and_seeds_are_distinct():
    seeds = {(i, t): isv.ifeval_sample_seed(i, t) for i in ("1000", "1001", "1005") for t in (0, 1)}
    assert len(set(seeds.values())) == len(seeds)                         # distinct across item AND trial
    for s in seeds.values():
        assert 0 <= s < 2 ** 32
    # no modulus: an item whose top bit is set stays above 2^31-1 (the Inspect
    # rule could never produce such a value)
    big = [isv.ifeval_sample_seed(str(i), 0) for i in range(1000, 1400)]
    assert any(s >= 2 ** 31 - 1 for s in big)
    assert all(isv.sample_seed(str(i), 0) < 2 ** 31 - 1 for i in range(1000, 1400))
    # the two rules differ on the same (item, trial)
    assert isv.ifeval_sample_seed("1000", 0) != isv.sample_seed("1000", 0)
    assert isv.ifeval_sample_seed("1000") == isv.ifeval_sample_seed("1000", 0)   # trial defaults to 0
    for bad in (-1, True, 1.0, "0"):
        with pytest.raises(ValueError, match="non-negative int"):
            isv.ifeval_sample_seed("1000", bad)
    for bad_key in ("", None, 1000):
        with pytest.raises(ValueError, match="non-empty str"):
            isv.ifeval_sample_seed(bad_key, 0)


def test_the_seed_is_independent_of_cell_cap_and_model_by_construction():
    """The key names no cell, cap or model: the same (item, trial) draws the
    same seed in every lane and cell, and a rung re-draw reuses it."""
    import inspect
    src = inspect.getsource(isv.ifeval_sample_seed)
    assert "IFEVAL_SEED_BASE" in src and "IFEVAL_SEED_READOUT" in src
    for name in ("cell", "cap", "model", "max_tokens"):
        assert name not in inspect.signature(isv.ifeval_sample_seed).parameters
    assert inspect.signature(isv.ifeval_sample_seed).parameters.keys() == {"item_key", "trial"}


def test_seed_for_task_dispatches_ifeval_to_its_rule_and_everything_else_to_inspects():
    assert isv.seed_for_task(IFEVAL, "1000", 1) == isv.ifeval_sample_seed("1000", 1)
    for task in ("inspect/math", "inspect/gsm8k_paired200", "inspect/xstest", None):
        assert isv.seed_for_task(task, "1000", 1) == isv.sample_seed("1000", 1)


def test_the_rule_string_is_hashed_only_by_runs_that_file_ifeval():
    assert C.SEED_RULE_IFEVAL == ("sha256(f'{20260909}|{trial}|ifeval|{item_key}')[:4] big-endian (no modulus); "
                                  "trial = sample_idx; item_key = the Inspect sample id")
    assert C.SEED_RULES == {"inspect": C.SEED_RULE_INSPECT, "evalplus": C.SEED_RULE_EVALPLUS}   # unchanged
    assert C.seed_rules_for(["inspect/math", "inspect/gsm8k_paired200"]) == dict(C.SEED_RULES)
    assert C.seed_rules_for([]) == dict(C.SEED_RULES)
    assert C.seed_rules_for([IFEVAL]) == {**C.SEED_RULES, IFEVAL: C.SEED_RULE_IFEVAL}
    assert C.seed_rules_for({IFEVAL: 2, "inspect/math": 1}) == {**C.SEED_RULES, IFEVAL: C.SEED_RULE_IFEVAL}
    row = C.THINKING_SAMPLING_BY_MODEL_KEY["gemma4e4b"]
    legacy = C.sampling_identity(row, {"inspect/gsm8k_paired200": 1})
    assert legacy[C.IDENTITY_KEY_SEED_RULE] == dict(C.SEED_RULES)                # byte-identical to before
    with_if = C.sampling_identity(row, {IFEVAL: 2})
    assert with_if[C.IDENTITY_KEY_SEED_RULE] == {**C.SEED_RULES, IFEVAL: C.SEED_RULE_IFEVAL}
    assert with_if[C.IDENTITY_KEY_SAMPLES_BY_TASK] == {IFEVAL: 2}


def test_the_request_fragment_carries_the_ifeval_seed_for_the_ifeval_task(monkeypatch):
    """The provider's ``tinycap_config`` seeds a request through ``seed_for_task``
    with the task it is building for — an IFEval request's ``seed`` is the
    IFEval rule's value for (item, trial)."""
    import inspect
    import sisteer.capeval.tiers.inspect_served as mod
    src = inspect.getsource(mod)
    i = src.index("def tinycap_config")
    body = src[i:src.index("def sampling_params", i)]
    assert "seed_for_task(task, item, idx)" in body
    assert "= sample_seed(item, idx)" not in body


def test_results_from_log_recomputes_the_seed_by_the_tasks_rule():
    from test_inspect_served import _Log, _Sample, _Score

    log = _Log([_Sample("1000", {"s": _Score("C")})])
    plain = isv.results_from_log(log, sample_idx=1)
    assert plain[0].seed == isv.sample_seed("1000", 1)
    tagged = isv.results_from_log(log, sample_idx=1, row_task=IFEVAL)
    assert tagged[0].seed == isv.ifeval_sample_seed("1000", 1) == 3178680996
    other = isv.results_from_log(log, sample_idx=1, row_task="inspect/math")
    assert other[0].seed == isv.sample_seed("1000", 1)


def test_a_rung_redraw_reuses_the_pass_seed():
    """A rung re-draws (item, trial) at a higher cap under the SAME seed: the
    derivation reads nothing that changes between pass 1 and a rung."""
    for cap in (2048, 4096, 8192):     # noqa: B007 — the cap is deliberately unused
        assert isv.seed_for_task(IFEVAL, "1234", 1) == isv.ifeval_sample_seed("1234", 1)
