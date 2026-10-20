"""qwen-base-seed-replicates -> qwen-seedrep-native-battery: the battery's eval-seed rule, pinned.

The replicate base IS the literal training seed — seed_base(s) = s (1 / 2 / 3); the hashed rule (sha256 of the campaign base
and the seed, s1 4057844608 / s2 2226795507 / s3 2554970601) is RETIRED and pinned here as the KNOWN-BAD every consumer must no
longer produce. The values below are the literals every downstream reader compares against (the per-view sampled configs, the BoN
and GSM8K opt-in policies, the seed ledger). Every leg of the battery carries exactly one of the three row markers and its
mechanism is named; an unknown leg refuses rather than defaulting."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "seedrep"))
import eval_seed as es  # noqa: E402

PINNED = {1: 1, 2: 2, 3: 3}
#: the RETIRED hashed rule's values (qwen-base-seed-replicates): int(sha256(f"20260909|seedrep|{s}").hexdigest()[:8], 16) — the
#: known-bad no consumer may produce or declare any more
RETIRED_HASHED = {1: 4057844608, 2: 2226795507, 3: 2554970601}


def test_the_three_bases_are_the_literal_training_seeds_and_the_hashed_rule_is_retired():
    assert es.SEED_BASE_CAMPAIGN == 20260909 and es.SEEDS == (1, 2, 3) and es.TARGET == "qwen35_4b"
    for s, want in PINNED.items():
        assert es.seed_base(s) == want == s
        assert 0 < want < 2 ** 32 and want != es.SEED_BASE_CAMPAIGN
        # the KNOWN-BAD: the retired rule's value is what its statement still derives — and is NOT what the module returns
        assert int(hashlib.sha256(f"20260909|seedrep|{s}".encode()).hexdigest()[:8], 16) == RETIRED_HASHED[s] != es.seed_base(s)
    assert es.SEED_BASE_BY_SEED == PINNED
    assert es.SEED_BASE_BY_VIEW == {"s1": PINNED[1], "s2": PINNED[2], "s3": PINNED[3]} == {"s1": 1, "s2": 2, "s3": 3}
    assert es.SEED_BASE_BY_TAG == {"qwen35_4b_s1": 1, "qwen35_4b_s2": 2, "qwen35_4b_s3": 3}
    assert len(set(PINNED.values())) == 3          # three distinct bases
    assert not set(RETIRED_HASHED.values()) & set(es.SEED_BASE_BY_SEED.values())
    # a base outside the triplet is derivable (the rule is total over positive ints) but never a lane view
    assert es.seed_base(4) == 4 != es.seed_base(1)
    for bad in (0, -1, True, 1.0, "1"):
        with pytest.raises(ValueError):
            es.seed_base(bad)


def test_view_and_tag_grammar_round_trips_and_refuses_every_other_view():
    for s in (1, 2, 3):
        assert es.view_of(s) == f"s{s}" and es.seed_of_view(f"s{s}") == s and es.tag_of(s) == f"qwen35_4b_s{s}"
        assert es.is_seedrep_view(f"s{s}")
    for bad in ("main", "it", "s0", "s4", "it_d1", "S1", "s01", "", "1"):
        assert not es.is_seedrep_view(bad)
        with pytest.raises(ValueError, match="not a seed-replicate view"):
            es.seed_of_view(bad)
    with pytest.raises(ValueError):
        es.view_of(4)


def test_every_leg_carries_exactly_one_of_the_three_rules_per_section_13():
    assert es.SEED_RULES == ("replicate_seed_base", "instrument_frozen", "none_sent")
    want = {
        "pasttense": "replicate_seed_base", "codeattack": "replicate_seed_base", "renellm": "replicate_seed_base", "grok420": "replicate_seed_base",
        "xstest": "replicate_seed_base", "or_bench_hard_1k": "replicate_seed_base",
        "mmlu": "instrument_frozen",
        "gsm8k": "replicate_seed_base", "bon": "replicate_seed_base",
        "crescendo": "none_sent", "pair": "none_sent", "actorattack": "none_sent",
        "petri": "none_sent", "misalign_leaking": "none_sent", "misalign_murder": "none_sent",
    }
    assert {k: v.seed_rule for k, v in es.LEG_SEED_RULES.items()} == want and set(es.EVAL_IDS) == set(want)
    for eval_id, rule in want.items():
        leg = es.leg_seed_rule(eval_id)
        assert leg.eval_id == eval_id and leg.seed_rule in es.SEED_RULES
        if rule == "replicate_seed_base":
            assert leg.seed_field in ("sampling_seed", "seed") and leg.reaches_via
            assert all(es.leg_seed_base(eval_id, s) == PINNED[s] for s in (1, 2, 3))
        else:
            assert all(es.leg_seed_base(eval_id, s) is None for s in (1, 2, 3))
        if rule == "none_sent":
            assert leg.seed_field is None and leg.decode_seed_policy is None
    # the two opt-in policies are named on exactly the two instruments that implement them
    assert es.leg_seed_rule("bon").decode_seed_policy == es.BON_DECODE_SEED_POLICY == "replicate_v1"
    assert es.BON_SEED_RULE_RECORDED == "bon_seed_v1+decode:replicate_v1"
    assert es.leg_seed_rule("gsm8k").decode_seed_policy == es.GSM8K_INSPECT_SEED_POLICY == "replicate"
    assert es.leg_seed_rule("mmlu").seed_field == "seed" and es.MMLU_FROZEN_SEED == 0
    with pytest.raises(ValueError, match="not a leg of the lane"):
        es.leg_seed_rule("wmdp")


def test_the_cli_prints_the_bases_and_refuses_a_foreign_seed():
    py = sys.executable
    script = REPO / "scripts/sdfbase/seedrep/eval_seed.py"
    r = subprocess.run([py, str(script)], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and r.stdout.splitlines() == [f"s{s} {PINNED[s]}" for s in (1, 2, 3)]
    for args, want in ((["--seed", "2"], PINNED[2]), (["--view", "s3"], PINNED[3])):
        r = subprocess.run([py, str(script), *args], capture_output=True, text=True, check=False)
        assert r.returncode == 0 and r.stdout.strip() == str(want), args
    for args in (["--seed", "4"], ["--view", "main"], ["--view", "s0"]):
        r = subprocess.run([py, str(script), *args], capture_output=True, text=True, check=False)
        assert r.returncode == 2 and r.stdout == "" and r.stderr.strip(), args
    r = subprocess.run([py, str(script), "--legs"], capture_output=True, text=True, check=False)
    legs = json.loads(r.stdout)
    assert r.returncode == 0 and set(legs) == set(es.EVAL_IDS) and legs["bon"]["decode_seed_policy"] == "replicate_v1"
    # stdlib only: importable by the shell gates and the matrix generator without the venv
    r = subprocess.run(["python3", "-c", "import sys; sys.path.insert(0, sys.argv[1]); import eval_seed; print(eval_seed.seed_base(1))",
                        str(script.parent)], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and r.stdout.strip() == str(PINNED[1])


def test_the_paired_view_of_a_replicate_carries_its_seed_base_and_nothing_else_is_admitted():
    """b64-capability-agentic-queue: `s<s>p` is a SECOND view of replicate s (the composed difference_relu arm on its
    own additive view), never a fourth seed: same seed base, same tag family; `view_of` still names the seed view."""
    for s in es.SEEDS:
        assert es.paired_view_of(s) == f"s{s}p" and es.seed_of_view(f"s{s}p") == s == es.seed_of_view(f"s{s}")
        assert es.is_seedrep_view(f"s{s}p") and es.is_paired_view(f"s{s}p") and not es.is_paired_view(f"s{s}")
        assert es.seed_base(es.seed_of_view(f"s{s}p")) == es.seed_base(s) == s
        assert es.view_of(s) == f"s{s}"
    for bad in ("s4p", "s0p", "sp", "s1pp", "p1", "s1 p", "S1p"):
        assert not es.is_seedrep_view(bad) and not es.is_paired_view(bad)
        with pytest.raises(ValueError):
            es.seed_of_view(bad)
    with pytest.raises(ValueError):
        es.paired_view_of(4)


def test_the_paired_seed_bases_are_kept_apart_from_the_seed_views_table():
    assert es.PAIRED_SEED_BASE_BY_VIEW == {"s1p": 1, "s2p": 2, "s3p": 3} and set(es.SEED_BASE_BY_VIEW) == {"s1", "s2", "s3"}
    assert es.SEED_BASE_BY_ANY_VIEW == {**es.SEED_BASE_BY_VIEW, **es.PAIRED_SEED_BASE_BY_VIEW}
