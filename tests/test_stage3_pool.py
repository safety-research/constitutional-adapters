"""Stage-3 pool construction:
the N-SI matched intersection, the >=200 hard floor, the benign 50/50 half,
the composition assertion, train/scoring disjointness, and the irrelevant SI's
rejection-sampled ``pos`` — plus the build_two_si_pool CLI plumbing for all of
it (generate-pos --si/--candidates, generate-benign, grade-benign, and the
stage-3 end-to-end manifest + emit-pairs).

CPU-only, offline, no GPU, no harmful text: every prompt/response here is an
INVENTED synthetic string (privacy rule — placeholder ids, no real data).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_two_si_pool as bp  # noqa: E402
from sisteer.eval.jailbreak_data import normalized_text_sha  # noqa: E402
from sisteer.eval.matched_pool import (  # noqa: E402
    GradedRow,
    MatchedPoolError,
    PairRecord,
    PoolAssertionError,
    PoolCompositionError,
    PoolDimensionError,
    PoolFloorError,
    PoolSpec,
    assert_pools_identical_except_pos,
    build_matched_pool,
    dumps_manifest,
    text_sha,
)

REPO = Path(__file__).resolve().parents[1]
SI3 = ("safety_refusal", "si_constitution", "si_irrelevant")


# --------------------------------------------------------------------------- #
# synthetic worlds                                                              #
# --------------------------------------------------------------------------- #


def _gr(fam, gid, vk, cond, si, prompt, resp, *, score=None, label=None,
        cidx=0):
    return GradedRow(family=fam, goal_id=gid, variant_key=vk, condition=cond,
                     si_id=si, prompt=prompt, response=resp, score=score,
                     label=label, candidate_idx=cidx)


def _hprompt(i):
    return f"SYNTHETIC attack prompt number {i} benign stand-in text"


def _hneg(i):
    return f"SYNTHETIC undefended completion {i} that complied"


def _hpos(si, i):
    return f"SYNTHETIC refusal from {si} for prompt {i}"


def _bprompt(i):
    return f"SYNTHETIC benign question {i} about a safe topic"


def _bneg(i):
    return f"SYNTHETIC undefended benign answer {i}"


def _bpos(si, i):
    return f"SYNTHETIC compliant answer from {si} to question {i}"


def _harmful(i, *, fam="bon", thwarted=SI3, si_rows=SI3):
    """One harmful prompt: none-successful, thwarted by ``thwarted``,
    jailbroken under the rest of ``si_rows``."""
    gid = f"g{i}"
    rows = [_gr(fam, gid, "0:0", "none", None, _hprompt(i), _hneg(i),
                score=0.9)]
    for si in si_rows:
        thw = si in thwarted
        resp = _hpos(si, i) if thw else f"SYNTHETIC jailbroken reply {si} {i}"
        rows.append(_gr(fam, gid, "0:0", "si", si, _hprompt(i), resp,
                        score=0.0 if thw else 0.9))
    return rows


def _benign(i, *, labels=None, si_rows=SI3):
    """One benign prompt: an undefended none row (the shared neg) + one
    labelled SI row per ``si_rows`` (label defaults to full compliance C)."""
    labels = labels or {}
    gid = f"phtest_{i}"
    rows = [_gr("phtest", gid, "0", "none", None, _bprompt(i), _bneg(i))]
    for si in si_rows:
        lab = labels.get(si, "C")
        resp = (_bpos(si, i) if lab == "C"
                else f"SYNTHETIC hedged half answer from {si} {i}")
        rows.append(_gr("phtest", gid, "0", "si", si, _bprompt(i), resp,
                        label=lab))
    return rows


def _spec3(**over):
    base = dict(
        si_ids=SI3,
        train_families=("bon", "pasttense"),
        eval_families=("pair", "tap"),
        train_goal_ids=tuple(f"g{i}" for i in range(64)),
        eval_goal_ids=("g_eval",),
        static_families=frozenset({"bon", "pasttense"}),
        threshold=0.5, min_pool_size=1,
        benign_source="phtest", benign_fraction=0.5,
        hard_floor_pairs=2)
    base.update(over)
    return PoolSpec(**base)


def _world(n_h=4, n_b=4):
    """The happy stage-3 world: n_h matched harmful prompts alternating over
    the two train families, n_b all-comply benign prompts."""
    rows = []
    for i in range(n_h):
        rows += _harmful(i, fam="bon" if i % 2 == 0 else "pasttense")
    for j in range(n_b):
        rows += _benign(100 + j)
    return rows


# --------------------------------------------------------------------------- #
# 1. the THREE-way intersection                                                  #
# --------------------------------------------------------------------------- #


def test_three_way_intersection_requires_all_three_sis():
    rows = _world(4, 4)
    rows += _harmful(10, thwarted=("safety_refusal", "si_constitution"))
    rows += _harmful(11, thwarted=("safety_refusal", "si_irrelevant"))
    rows += _harmful(12, thwarted=("si_constitution", "si_irrelevant"))
    m = build_matched_pool(rows, _spec3())

    assert m["hard_floor"]["n_harmful_matched"] == 4      # g10-12 excluded
    goals = {p["goal_id"] for p in m["matched_intersection"]["ordered_pairs"]}
    assert {"g10", "g11", "g12"}.isdisjoint(goals)
    # each two-of-three prompt still shows up in ITS thwarting SIs' native pools
    assert m["native_per_si"]["safety_refusal"]["n"] == 6      # 4 + g10 + g11
    assert m["native_per_si"]["si_constitution"]["n"] == 6     # 4 + g10 + g12
    assert m["native_per_si"]["si_irrelevant"]["n"] == 6       # 4 + g11 + g12
    # the balanced pool: 4 harmful + 4 benign, identical for every SI
    assert m["pool_sizes"] == {si: 8 for si in SI3}
    for p in m["matched_intersection"]["ordered_pairs"]:
        assert set(p["pos_sha"]) == set(SI3)                   # one pos per SI


def test_three_si_pools_differ_only_in_pos_including_benign_pairs():
    m = build_matched_pool(_world(2, 2), _spec3(hard_floor_pairs=1))
    pairs = m["matched_intersection"]["ordered_pairs"]
    benign = [p for p in pairs if p["family"] == "phtest"]
    assert benign and len(pairs) == 4
    for p in pairs:
        # the pos completions genuinely differ per SI; prompt/neg are shared
        assert len(set(p["pos_sha"].values())) == 3
    b = benign[0]
    i = int(b["goal_id"].split("_")[1])
    assert b["neg_sha"] == text_sha(_bneg(i))
    assert b["pos_sha"]["si_irrelevant"] == text_sha(_bpos("si_irrelevant", i))


# --------------------------------------------------------------------------- #
# 2. the five identity invariants, one negative test each, at THREE SIs         #
# --------------------------------------------------------------------------- #


def _base_pools3():
    def rec(g, pos):
        return PairRecord("bon", g, "0:0", text_sha(f"prompt-{g}"),
                          text_sha(f"neg-{g}"), text_sha(pos))
    return {si: [rec("g0", f"pos-{si}-0"), rec("g1", f"pos-{si}-1")]
            for si in SI3}


def test_identity_holds_for_three_pools_differing_only_in_pos():
    assert_pools_identical_except_pos(_base_pools3())  # no raise


def test_identity3_fails_on_reordered_prompt_ids():
    pools = _base_pools3()
    pools[SI3[2]] = list(reversed(pools[SI3[2]]))
    with pytest.raises(PoolAssertionError, match="ordered_prompt_ids"):
        assert_pools_identical_except_pos(pools)


def test_identity3_fails_on_per_family_counts():
    pools = _base_pools3()
    p = pools[SI3[2]][1]
    pools[SI3[2]][1] = PairRecord("pasttense", p.goal_id, p.variant_key,
                                  p.prompt_sha, p.neg_sha, p.pos_sha)
    with pytest.raises(PoolAssertionError, match="per_family_counts"):
        assert_pools_identical_except_pos(pools)


def test_identity3_fails_on_per_goal_counts():
    pools = _base_pools3()
    p = pools[SI3[2]][1]
    pools[SI3[2]][1] = PairRecord(p.family, "g_other", p.variant_key,
                                  p.prompt_sha, p.neg_sha, p.pos_sha)
    with pytest.raises(PoolAssertionError, match="per_goal_counts"):
        assert_pools_identical_except_pos(pools)


def test_identity3_fails_on_differing_neg_responses():
    pools = _base_pools3()
    p = pools[SI3[2]][0]
    pools[SI3[2]][0] = PairRecord(p.family, p.goal_id, p.variant_key,
                                  p.prompt_sha, text_sha("DIFFERENT-neg"),
                                  p.pos_sha)
    with pytest.raises(PoolAssertionError, match="neg_responses"):
        assert_pools_identical_except_pos(pools)


def test_identity3_fails_when_prompts_differ_not_only_pos():
    pools = _base_pools3()
    p = pools[SI3[2]][0]
    pools[SI3[2]][0] = PairRecord(p.family, p.goal_id, p.variant_key,
                                  text_sha("DIFFERENT-prompt"), p.neg_sha,
                                  p.pos_sha)
    with pytest.raises(PoolAssertionError, match="differ_only_in_pos"):
        assert_pools_identical_except_pos(pools)


def test_identity3_names_the_violating_si():
    pools = _base_pools3()
    pools[SI3[1]] = list(reversed(pools[SI3[1]]))
    with pytest.raises(PoolAssertionError, match=SI3[1]):
        assert_pools_identical_except_pos(pools)


# --------------------------------------------------------------------------- #
# 3. rejection sampling                                                       #
# --------------------------------------------------------------------------- #


def _irr_candidates(i, cand_scores):
    """Rejection-sampling candidate rows for si_irrelevant on prompt i, one
    row per (candidate_idx, score); distinct completions per candidate."""
    return [_gr("bon", f"g{i}", "0:0", "si", "si_irrelevant", _hprompt(i),
                f"SYNTHETIC irrelevant candidate {c} reply for {i}",
                score=s, cidx=c)
            for c, s in cand_scores]


def _rs_world():
    """g0: irrelevant accepts on candidate 1 of 3; g1: all 2 candidates
    jailbroken -> excluded. Plus 2 all-comply benign prompts."""
    rows = []
    for i in (0, 1):
        rows += _harmful(i, thwarted=("safety_refusal", "si_constitution"),
                         si_rows=("safety_refusal", "si_constitution"))
    rows += _irr_candidates(0, [(0, 0.9), (1, 0.0), (2, 0.0)])
    rows += _irr_candidates(1, [(0, 0.9), (1, 0.9)])
    rows += _benign(100) + _benign(101)
    return rows


def test_rejection_sampling_accepts_only_thwarting_candidates():
    m = build_matched_pool(
        _rs_world(), _spec3(train_families=("bon",), hard_floor_pairs=1))
    pairs = m["matched_intersection"]["ordered_pairs"]
    goals = {p["goal_id"] for p in pairs}
    assert "g0" in goals and "g1" not in goals
    g0 = next(p for p in pairs if p["goal_id"] == "g0")
    # the frozen pos is the FIRST accepted candidate (idx 1), never the
    # jailbroken idx-0 draw — that is what keeps harmful-compliance targets out
    assert g0["pos_sha"]["si_irrelevant"] == text_sha(
        "SYNTHETIC irrelevant candidate 1 reply for 0")


def test_rejection_sampling_reports_realised_k_and_per_prompt_yield():
    m = build_matched_pool(
        _rs_world(), _spec3(train_families=("bon",), hard_floor_pairs=1))
    rs = m["rejection_sampling"]["si_irrelevant"]
    assert rs["n_prompts_with_candidates"] == 2
    assert rs["n_candidates"] == 5
    assert rs["k_max"] == 3 and rs["k_mean"] == 2.5
    assert rs["n_prompts_accepted"] == 1
    assert rs["per_prompt_yield"] == 0.5
    assert rs["per_candidate_accept_rate"] == 0.4
    # the safety SIs sampled once per prompt, full yield
    assert m["rejection_sampling"]["safety_refusal"]["k_max"] == 1
    assert m["rejection_sampling"]["safety_refusal"]["per_prompt_yield"] == 1.0
    # coverage names the all-candidates-rejected prompt a raised K could save
    assert m["coverage"]["si_irrelevant"]["n_all_candidates_rejected"] == 1


def test_conflicting_verdicts_within_one_candidate_still_raise():
    rows = _rs_world()
    # duplicate of g0's candidate 1 with the OPPOSITE verdict — ambiguous
    rows += _irr_candidates(0, [(1, 0.9)])
    with pytest.raises(MatchedPoolError, match="conflicting graded verdicts"):
        build_matched_pool(rows, _spec3(train_families=("bon",),
                                        hard_floor_pairs=1))


# --------------------------------------------------------------------------- #
# 4. the hard floor                                                           #
# --------------------------------------------------------------------------- #


def _floor_world():
    """2 fully-matched harmful prompts + 1 two-of-three (safety+const only);
    3 all-comply benign + 1 rejected by si_irrelevant (P). So: harmful matched
    2, thwarted per SI 3/3/2; benign matched 3, complied per SI 4/4/3."""
    rows = _world(2, 0)
    rows += _harmful(10, thwarted=("safety_refusal", "si_constitution"))
    for j in range(3):
        rows += _benign(100 + j)
    rows += _benign(103, labels={"si_irrelevant": "P"})
    return rows


def test_hard_floor_raises_with_realised_counts_per_si_and_side():
    with pytest.raises(PoolFloorError) as e:
        build_matched_pool(_floor_world(), _spec3(hard_floor_pairs=3))
    msg = str(e.value)
    # the failing and passing side, each with its realised count and verdict
    assert "harmful-refusal matched pairs = 2 (FAIL, floor 3)" in msg
    assert "benign-compliance matched pairs = 3 (ok, floor 3)" in msg
    # realised counts per SI, both sides
    assert ("harmful thwarted: safety_refusal=3, si_constitution=3, "
            "si_irrelevant=2") in msg
    assert ("benign complied: safety_refusal=4, si_constitution=4, "
            "si_irrelevant=3") in msg
    assert "never falls back to native_per_si" in msg


def test_hard_floor_passes_at_exactly_the_floor():
    m = build_matched_pool(_floor_world(), _spec3(hard_floor_pairs=2))
    assert m["hard_floor"] == {"floor_pairs": 2, "enforced": True,
                               "n_harmful_matched": 2,
                               "n_strict_harmful_matched": 2,
                               "n_benign_matched": 3}
    assert m["regime"] == "matched_intersection"


def test_hard_floor_gates_the_benign_side_too():
    rows = _world(4, 0) + _benign(100)          # 4 harmful, only 1 benign
    with pytest.raises(PoolFloorError) as e:
        build_matched_pool(rows, _spec3(hard_floor_pairs=2))
    msg = str(e.value)
    assert "benign-compliance matched pairs = 1 (FAIL, floor 2)" in msg
    assert "harmful-refusal matched pairs = 4 (ok, floor 2)" in msg


def test_floor_disables_the_native_fallback_entirely():
    # legacy behavior (no stage-3 gates): a thin intersection FALLS BACK
    rows = _world(1, 0)
    legacy = _spec3(benign_source=None, benign_fraction=None,
                    hard_floor_pairs=0, min_pool_size=5)
    assert build_matched_pool(rows, legacy)["regime"] == "native_per_si"
    # the floor gate: same world RAISES instead — never a confounded pool
    with pytest.raises(PoolFloorError):
        build_matched_pool(rows, _spec3(benign_source=None,
                                        benign_fraction=None,
                                        hard_floor_pairs=5, min_pool_size=5))
    # and even a PASSING floor never re-enables the fallback: min_pool_size
    # above the matched size raises rather than shipping native
    with pytest.raises(PoolFloorError, match="native fallback is DISABLED"):
        build_matched_pool(_world(2, 2), _spec3(hard_floor_pairs=1,
                                                min_pool_size=50))


# --------------------------------------------------------------------------- #
# 5. the benign half: all-comply intersection + 50/50 balance                 #
# --------------------------------------------------------------------------- #


def test_benign_admission_needs_all_three_sis_to_comply():
    rows = _world(1, 0)
    rows += _benign(100)                                        # all C -> in
    rows += _benign(101, labels={"si_constitution": "P"})       # P -> out
    rows += _benign(102, labels={"si_irrelevant": "I"})         # I -> out
    rows += _benign(103, si_rows=("safety_refusal",             # missing row
                                  "si_constitution"))           #   -> out
    m = build_matched_pool(rows, _spec3(train_families=("bon",),
                                        hard_floor_pairs=1))
    b = m["benign"]
    assert b["n_candidates"] == 4
    assert b["n_matched_all_si_comply"] == 1
    # the C/P/I tallies come from select_xstest_complied, per SI, over the
    # prompts that HAVE a row for that SI
    assert b["per_si_xstest_counts"]["si_constitution"] == {"C": 3, "P": 1,
                                                            "I": 0}
    assert b["per_si_xstest_counts"]["si_irrelevant"] == {"C": 2, "P": 0,
                                                          "I": 1}
    assert b["coverage"]["si_irrelevant"]["n_missing"] == 1
    goals = {p["goal_id"] for p in m["matched_intersection"]["ordered_pairs"]}
    assert "phtest_100" in goals
    assert goals.isdisjoint({"phtest_101", "phtest_102", "phtest_103"})


def test_benign_fraction_balances_the_larger_side_deterministically():
    m = build_matched_pool(_world(2, 5), _spec3(hard_floor_pairs=1))
    comp = m["composition"]
    assert comp["pre_balance"] == {"n_harmful_matched": 2,
                                   "n_benign_matched": 5,
                                   "harmful_taken": 2, "benign_taken": 2}
    assert comp["realised"]["benign_fraction"] == 0.5
    assert comp["realised"]["n_pairs"] == 4
    assert m["pool_sizes"] == {si: 4 for si in SI3}
    # byte-reproducible: the sha-ordered subsample is seedless + deterministic
    m2 = build_matched_pool(_world(2, 5), _spec3(hard_floor_pairs=1))
    assert dumps_manifest(m) == dumps_manifest(m2)
    assert m["manifest_sha256"] == m2["manifest_sha256"]


# ---------------------------------------------------------------------------- #
# 6. the composition assertion                                                  #
# ---------------------------------------------------------------------------- #


def test_composition_missing_family_fails_the_build():
    # declared bon+pasttense, realised bon only — the exact stage-2 regression
    rows = []
    for i in range(2):
        rows += _harmful(i, fam="bon")
    rows += _benign(100) + _benign(101)
    with pytest.raises(PoolCompositionError, match="pasttense"):
        build_matched_pool(rows, _spec3(hard_floor_pairs=1))


def test_composition_off_target_benign_fraction_fails_the_build():
    # declared 0.25 +/- 0.02 but integer rounding starves the benign side to
    # zero -> realised 0.0, outside tolerance
    rows = _world(1, 0) + _benign(100)
    with pytest.raises(PoolCompositionError, match="benign fraction"):
        build_matched_pool(rows, _spec3(train_families=("bon",),
                                        benign_fraction=0.25,
                                        hard_floor_pairs=0))


def test_composition_cannot_pass_vacuously_on_an_empty_pool():
    with pytest.raises(PoolCompositionError, match="vacuously"):
        build_matched_pool([], _spec3(hard_floor_pairs=0))


def test_apply_families_override_narrows_and_records_skipped():
    """Decision (a): --families narrows the committed pool to the families
    actually generated, listing the dropped ones as skipped — the composition
    assertion then passes truthfully on the K present, without editing the
    committed eight-family decision."""
    cfg = {"pool": {
        "static_families": ["bon", "pasttense", "pap", "encoding", "aim",
                            "gcg_replay", "amplegcg_plus"],
        "train_families": ["bon", "pasttense", "pap", "encoding", "aim",
                           "gcg_replay", "amplegcg_plus"]}}
    six = ["bon", "pasttense", "pap", "encoding", "aim", "gcg_replay"]
    cfg2, skipped = bp._apply_families_override(cfg, six)
    assert cfg2["pool"]["static_families"] == six
    assert cfg2["pool"]["train_families"] == six
    assert skipped == ["amplegcg_plus"]
    assert cfg["pool"]["static_families"][-1] == "amplegcg_plus"   # cfg untouched


def test_apply_families_override_refuses_non_committed_family():
    """--families may only NARROW: adding a family the config never declared is
    a scope change and is refused (the fixed eight-family decision stands)."""
    cfg = {"pool": {"static_families": ["bon", "pasttense"],
                    "train_families": ["bon", "pasttense"]}}
    with pytest.raises(SystemExit, match="scope change"):
        bp._apply_families_override(cfg, ["bon", "renellm"])


def test_composition_block_records_declared_vs_realised():
    m = build_matched_pool(_world(4, 4), _spec3())
    comp = m["composition"]
    assert comp["declared"] == {
        "train_families": ["bon", "pasttense"],
        "benign_source": "phtest",
        "benign_fraction": 0.5,
        "benign_fraction_tol": 0.02,
    }
    assert comp["realised"]["per_family_counts"] == {"bon": 2, "pasttense": 2,
                                                     "phtest": 4}
    assert comp["asserted"] is True


# --------------------------------------------------------------------------- #
# 7. train/scoring disjointness                                               #
# --------------------------------------------------------------------------- #


def test_scoring_prompts_are_removed_from_both_sides_by_normalised_match():
    # a REFORMATTED copy of each scoring prompt (case + whitespace mangled)
    # must still be caught — the filter is normalised, not raw-bytes
    scoring = frozenset({
        normalized_text_sha("  " + _hprompt(0).upper() + "   "),
        normalized_text_sha(_bprompt(100).replace(" ", "  ")),
    })
    m = build_matched_pool(
        _world(3, 3), _spec3(hard_floor_pairs=1, scoring_prompt_shas=scoring))
    sd = m["scoring_disjointness"]
    assert sd == {"n_scoring_prompt_shas": 2, "n_removed_harmful": 1,
                  "n_removed_benign": 1}
    goals = {p["goal_id"] for p in m["matched_intersection"]["ordered_pairs"]}
    assert "g0" not in goals and "phtest_100" not in goals
    assert m["hard_floor"]["n_harmful_matched"] == 2      # counted AFTER removal


def test_floor_is_counted_after_the_scoring_subtraction():
    scoring = frozenset({normalized_text_sha(_hprompt(0))})
    with pytest.raises(PoolFloorError, match="harmful-refusal matched pairs = 2"):
        build_matched_pool(
            _world(3, 3),
            _spec3(hard_floor_pairs=3, scoring_prompt_shas=scoring))


# --------------------------------------------------------------------------- #
# 8. spec-level declaration guards                                              #
# --------------------------------------------------------------------------- #


def test_benign_source_and_fraction_must_be_declared_together():
    with pytest.raises(MatchedPoolError, match="declared together"):
        _spec3(benign_fraction=None)
    with pytest.raises(MatchedPoolError, match="declared together"):
        _spec3(benign_source=None)


def test_benign_source_may_not_collide_with_an_attack_family():
    with pytest.raises(MatchedPoolError, match="collides"):
        _spec3(benign_source="bon")


# --------------------------------------------------------------------------- #
# the CLI: config wiring, per-SI pos runs, benign runs, end to end              #
# --------------------------------------------------------------------------- #


def _store_row(fam, gid, sample, trial, si_condition, prompt, resp, score,
               *, group="held_in", si_id=None, cidx=None, xstest_label=None):
    row = {"attack": fam, "goal_id": gid, "si_condition": si_condition,
           "category": "illegal", "category_group": group,
           "attack_prompt": prompt, "response": resp, "score": score}
    if sample is not None:
        row["sample_idx"] = sample
        row["trial"] = trial
    if si_id is not None:
        row["si_id"] = si_id
    if cidx is not None:
        row["candidate_idx"] = cidx
    if xstest_label is not None:
        row["xstest_label"] = xstest_label
    return row


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _stage3_cfg(tmp_path, *, n_prompts=2, floor=1):
    """The REAL shipped config, shrunk to a synthetic bon-only world small
    enough for a test: 3 SIs, benign half on, floor on (tiny), no hub access
    (scoring_disjointness emptied; the loader has its own test below)."""
    cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    store = tmp_path / "labeled"
    rows = []
    for i in range(n_prompts):
        rows += [
            _store_row("bon", f"g{i}", i, 0, "none", _hprompt(i), _hneg(i),
                       0.9),
            _store_row("bon", f"g{i}", i, 0, "prompted", _hprompt(i),
                       _hpos("safety_refusal", i), 0.0),
        ]
    # One held-out goal. Under the shipped config's goal-level holdout
    # regime (pool.family_disjointness: false) the goal split is the ONLY
    # thing held out, and build_matched_pool refuses a pool with an empty
    # eval goal set -- correctly, since every "held-out" number from such a
    # pool would be in-sample. A synthetic world therefore needs a split
    # too, or it is not exercising the shipped regime.
    rows += [
        _store_row("bon", "g_heldout", n_prompts, 0, "none",
                   _hprompt(n_prompts), _hneg(n_prompts), 0.9,
                   group="held_out"),
        _store_row("bon", "g_heldout", n_prompts, 0, "prompted",
                   _hprompt(n_prompts), _hpos("safety_refusal", n_prompts),
                   0.0, group="held_out"),
    ]
    _write_jsonl(store / "bon.jsonl", rows)
    cfg["pool"] = {**cfg["pool"], "store_dir": str(store),
                   "static_families": ["bon"], "train_families": ["bon"],
                   "min_pool_size": 1, "hard_floor_pairs": floor,
                   "scoring_disjointness": {},
                   # the scale-up amendment keys are neutralised for these
                   # tiny synthetic worlds: the dimension floor (alphakl d=4096)
                   # is meaningless at n=2, and the extra benign corpora are
                   # cluster-only (not loadable offline). Each has its own
                   # dedicated test below.
                   "fit_floor": {}, "benign_sources": [],
                   "relaxed_stratum": {"enabled": False},
                   "bidpo_benign": {"enabled": False}}
    # The shipped config declares both generation halves vllm
    # (pos/neg_generation_engine, the ed01306 engine-symmetry keys). These
    # synthetic worlds default to the LEGACY hf route (fast, fully stubbed);
    # the vllm route's tests opt in explicitly (--engine / pool.engine / a
    # declaration they set themselves — see test_pool_vllm_generation.py),
    # so the declaration must not ride in and silently select vllm here.
    for k in ("pos_generation_engine", "neg_generation_engine"):
        cfg["pool"].pop(k, None)
    cfg["pool_manifest"] = str(tmp_path / "pool_manifest.json")
    return cfg


def _seed_pos_runs(priv: Path, cfg, n_prompts=2, *, irr_scores=((0.9, 0.0),)):
    """Graded pos rows for every GENERATED SI: a safety_refusal run row per
    prompt (the shipped config flips the store SI to a generated run, so the
    store's own prompted rows are EXCLUDED and the run must carry the pos), a
    constitution row per prompt, and irrelevant candidate rows (default: a
    jailbroken candidate 0 and a thwarting candidate 1 per prompt)."""
    runs = bp._pos_runs(cfg)
    srows, crows, irows = [], [], []
    for i in range(n_prompts):
        srows.append(_store_row("bon", f"g{i}", i, 0, "prompted", _hprompt(i),
                                _hpos("safety_refusal", i), 0.0,
                                si_id="safety_refusal"))
        crows.append(_store_row("bon", f"g{i}", i, 0, "prompted", _hprompt(i),
                                _hpos("si_constitution", i), 0.0,
                                si_id="si_constitution"))
        for scores in irr_scores:
            for c, s in enumerate(scores):
                irows.append(_store_row(
                    "bon", f"g{i}", i, 0, "prompted", _hprompt(i),
                    f"SYNTHETIC irrelevant candidate {c} reply for {i}", s,
                    si_id="si_irrelevant", cidx=c))
    _write_jsonl(priv / runs["safety_refusal"] / "bon.jsonl", srows)
    _write_jsonl(priv / runs["si_constitution"] / "bon.jsonl", crows)
    _write_jsonl(priv / runs["si_irrelevant"] / "bon.jsonl", irows)


def _seed_benign_run(priv: Path, cfg, n=2, *, labels=None):
    labels = labels or {}
    brows = []
    for j in range(n):
        gid = f"phtest_{j}"
        brows.append({"attack": "phtest", "goal_id": gid, "order": 0,
                      "si_condition": "none", "attack_prompt": _bprompt(j),
                      "response": _bneg(j), "score": None})
        for si in SI3:
            brows.append({"attack": "phtest", "goal_id": gid, "order": 0,
                          "si_condition": "prompted", "si_id": si,
                          "attack_prompt": _bprompt(j),
                          "response": _bpos(si, j), "score": None,
                          "xstest_label": labels.get((j, si), "C"),
                          "candidate_idx": 0})
    _write_jsonl(priv / bp._benign_cfg(cfg)[1] / "phtest.jsonl", brows)


def test_cli_stage3_manifest_end_to_end(tmp_path):
    cfg = _stage3_cfg(tmp_path)
    priv = tmp_path / "priv"
    _seed_pos_runs(priv, cfg)
    _seed_benign_run(priv, cfg)

    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None),
                    cfg)
    m = json.loads(Path(cfg["pool_manifest"]).read_text())
    assert m["regime"] == "matched_intersection"
    assert m["si_ids"] == list(SI3)
    assert m["pool_sizes"] == {si: 4 for si in SI3}          # 2 harmful + 2 benign
    assert m["composition"]["realised"]["per_family_counts"] == {
        "bon": 2, "phtest": 2}
    assert m["composition"]["realised"]["benign_fraction"] == 0.5
    rs = m["rejection_sampling"]["si_irrelevant"]
    assert rs["k_max"] == 2 and rs["per_prompt_yield"] == 1.0
    # the frozen irrelevant pos is candidate 1 (the thwarting draw)
    for p in m["matched_intersection"]["ordered_pairs"]:
        if p["family"] == "bon":
            i = int(p["goal_id"][1:])
            assert p["pos_sha"]["si_irrelevant"] == text_sha(
                f"SYNTHETIC irrelevant candidate 1 reply for {i}")
    # nothing free-text leaked
    blob = Path(cfg["pool_manifest"]).read_text()
    for needle in (_hprompt(0), _hneg(0), _bprompt(0), _bneg(0),
                   _bpos("si_irrelevant", 0)):
        assert needle not in blob


def test_cli_stage3_floor_violation_raises_and_freezes_nothing(tmp_path):
    cfg = _stage3_cfg(tmp_path, floor=3)                    # only 2 per side
    priv = tmp_path / "priv"
    _seed_pos_runs(priv, cfg)
    _seed_benign_run(priv, cfg)
    with pytest.raises(PoolFloorError, match="matched pairs = 2"):
        bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv),
                                              out=None), cfg)
    assert not Path(cfg["pool_manifest"]).exists()          # nothing shipped


def test_cli_stage3_emit_pairs_three_files_identical_except_pos(tmp_path):
    cfg = _stage3_cfg(tmp_path)
    priv = tmp_path / "priv"
    _seed_pos_runs(priv, cfg)
    _seed_benign_run(priv, cfg)
    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None),
                    cfg)
    out = tmp_path / "pairs_out"
    bp.cmd_emit_pairs(types.SimpleNamespace(
        manifest=None, private_root=str(priv), out_dir=str(out),
        dry_run=False), cfg)

    files = {si: [json.loads(l) for l in
                  (out / bp.pairs_filename(si)).read_text().splitlines()]
             for si in SI3}
    ref = files[SI3[0]]
    assert len(ref) == 4
    for si in SI3[1:]:
        assert [r["prompt"] for r in files[si]] == [r["prompt"] for r in ref]
        assert [r["neg"] for r in files[si]] == [r["neg"] for r in ref]
        assert [r["pos"] for r in files[si]] != [r["pos"] for r in ref]
    # benign pairs emitted with the SI's complying answer as pos
    b = next(r for r in files["si_irrelevant"] if r["family"] == "phtest")
    j = int(b["goal_id"].split("_")[1])
    assert b["pos"] == _bpos("si_irrelevant", j) and b["neg"] == _bneg(j)
    meta = json.loads((out / bp.PAIRS_META_NAME).read_text())
    assert meta["pool_sizes"] == {si: 4 for si in SI3}


def test_cli_benign_p_labelled_item_is_excluded_end_to_end(tmp_path):
    cfg = _stage3_cfg(tmp_path, n_prompts=2, floor=1)
    priv = tmp_path / "priv"
    _seed_pos_runs(priv, cfg)
    # 3 benign items; one hedged (P) under si_irrelevant -> rejected
    _seed_benign_run(priv, cfg, n=3, labels={(2, "si_irrelevant"): "P"})
    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None),
                    cfg)
    m = json.loads(Path(cfg["pool_manifest"]).read_text())
    assert m["benign"]["n_matched_all_si_comply"] == 2
    assert m["benign"]["per_si_xstest_counts"]["si_irrelevant"]["P"] == 1
    goals = {p["goal_id"] for p in m["matched_intersection"]["ordered_pairs"]}
    assert "phtest_2" not in goals


# --------------------------------------------------------------------------- #
# generate-pos --si/--candidates: the rejection-sampling delta workflow          #
# --------------------------------------------------------------------------- #


def _stub_generation(monkeypatch):
    calls = {"rendered": []}

    def _gen(model, tok, rendered, **k):
        calls["rendered"].append(list(rendered))
        return [f"RESP::{r}" for r in rendered]

    monkeypatch.setattr("sisteer.models.load_model_and_tokenizer",
                        lambda *a, **k: (object(), object()))
    # render carries the system prefix so responses DIFFER per condition —
    # otherwise the graders' legitimate dedupe would fold the SIs together
    monkeypatch.setattr(
        "sisteer.models.build_chat_prompt",
        lambda tok, user, system=None: f"{(system or 'nosys')[:12]}|{user}")
    monkeypatch.setattr("sisteer.generate.generate_batch", _gen)
    return calls


# The vLLM side of generate-pos (engine selection, HF-parity sampling params,
# per-candidate seeds, done-mask resume, canonical F8 render, F9 exact-set
# verification, provenance) is covered end-to-end in
# tests/test_pool_vllm_generation.py against this branch's dual-engine
# cmd_generate_pos. The tests below exercise the RETAINED hf contingency
# route (`--engine`/`pool.engine` default to hf), which must stay
# byte-for-byte the pre-ruling behaviour.


def _irr_rows(priv, cfg):
    run = bp._pos_runs(cfg)["si_irrelevant"]
    path = priv / run / "bon.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_generate_pos_si_candidates_writes_k_rows_then_only_the_delta(
        tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    priv = tmp_path / "priv"
    _stub_generation(monkeypatch)

    args = types.SimpleNamespace(private_root=str(priv), dry_run=False,
                                 batch_size=4, si="si_irrelevant",
                                 candidates=2)
    bp.cmd_generate_pos(args, cfg)
    rows = _irr_rows(priv, cfg)
    assert len(rows) == 2
    assert {r["candidate_idx"] for r in rows} == {0, 1}
    assert all(r["si_id"] == "si_irrelevant" and r["score"] is None
               for r in rows)
    # pos rows keep the store schema whichever engine wrote them: the store's
    # category_group rides through, si_condition is the "prompted" mapping.
    assert all(r["category_group"] == "held_in"
               and r["si_condition"] == "prompted" for r in rows)

    # raising K generates ONLY the delta (candidate 2), never a regeneration
    args.candidates = 3
    bp.cmd_generate_pos(args, cfg)
    rows = _irr_rows(priv, cfg)
    assert len(rows) == 3
    assert {r["candidate_idx"] for r in rows} == {0, 1, 2}

    # once any candidate is graded-THWARTING the prompt is covered: no more
    run = bp._pos_runs(cfg)["si_irrelevant"]
    rows[1]["score"], rows[1]["graded"] = 0.0, True
    _write_jsonl(priv / run / "bon.jsonl", rows)
    args.candidates = 8
    bp.cmd_generate_pos(args, cfg)
    assert len(_irr_rows(priv, cfg)) == 3                    # unchanged


def test_generate_pos_refuses_unknown_sis(tmp_path):
    cfg = _stage3_cfg(tmp_path)
    with pytest.raises(SystemExit):
        bp.cmd_generate_pos(types.SimpleNamespace(
            private_root=str(tmp_path / "priv"), dry_run=True,
            batch_size=2, si="si_bogus", candidates=1), cfg)


def test_generate_pos_store_si_gated_on_declared_pos_run(tmp_path, capsys):
    """The store SI is generable ONLY once pool.pos_runs declares its run
    (the 2026-08-24 flip). Undeclared -> the historical refusal, with the
    enabling knob named; declared (the shipped config) -> a normal dry-run."""
    cfg = _stage3_cfg(tmp_path)
    legacy = dict(cfg, pool={
        **cfg["pool"],
        "pos_runs": {k: v for k, v in (cfg["pool"].get("pos_runs") or {}).items()
                     if k != "safety_refusal"}})
    with pytest.raises(SystemExit, match=r"pool\.pos_runs\.safety_refusal"):
        bp.cmd_generate_pos(types.SimpleNamespace(
            private_root=str(tmp_path / "priv"), dry_run=True,
            batch_size=2, si="safety_refusal", candidates=1), legacy)

    bp.cmd_generate_pos(types.SimpleNamespace(
        private_root=str(tmp_path / "priv"), dry_run=True,
        batch_size=2, si="safety_refusal", candidates=None), cfg)
    out = capsys.readouterr().out
    assert "run two_si_pos_safety_refusal" in out and "K=8" in out


def test_generate_pos_default_k_comes_from_pool_pos_candidates(tmp_path,
                                                               capsys):
    cfg = _stage3_cfg(tmp_path)
    cfg["pool"]["pos_candidates"] = {"si_irrelevant": 5}
    bp.cmd_generate_pos(types.SimpleNamespace(
        private_root=str(tmp_path / "priv"), dry_run=True, batch_size=2,
        si="si_irrelevant", candidates=None), cfg)
    assert "K=5" in capsys.readouterr().out


def test_grade_pos_grades_every_pos_run(tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    priv = tmp_path / "priv"
    runs = bp._pos_runs(cfg)
    assert "safety_refusal" in runs          # the flip: the store SI has a run
    _write_jsonl(priv / runs["safety_refusal"] / "bon.jsonl", [
        _store_row("bon", "g0", 0, 0, "prompted", _hprompt(0),
                   _hpos("safety_refusal", 0), None,
                   si_id="safety_refusal")])
    _write_jsonl(priv / runs["si_constitution"] / "bon.jsonl", [
        _store_row("bon", "g0", 0, 0, "prompted", _hprompt(0),
                   _hpos("si_constitution", 0), None,
                   si_id="si_constitution")])
    _write_jsonl(priv / runs["si_irrelevant"] / "bon.jsonl", [
        _store_row("bon", "g0", 0, 0, "prompted", _hprompt(0),
                   "SYNTHETIC irrelevant candidate 0 reply for 0", None,
                   si_id="si_irrelevant", cidx=0)])
    monkeypatch.setattr(bp, "_build_grader",
                        lambda a, c, d: (lambda items: {it: 0.0
                                                        for it in items}))
    bp.cmd_grade_pos(types.SimpleNamespace(private_root=str(priv),
                                           dry_run=False, concurrency=4,
                                           si=None), cfg)
    for run in runs.values():
        rows = [json.loads(l) for l in
                (priv / run / "bon.jsonl").read_text().splitlines()]
        assert all(r["score"] == 0.0 and r["graded"] for r in rows)


def test_grade_pos_cli_exposes_allow_unscored():
    """grade-pos grades a 22k-row pass in ONE shot (no lazy rounds), so a
    handful of permanently un-gradeable completions would otherwise raise JB1
    and block the whole grade. The same --allow-unscored knob grade-store /
    grade-anti-neg expose must be on grade-pos too — landing it on the wrong
    subparser fails only at launch, hours in."""
    import subprocess
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_two_si_pool.py"),
         "grade-pos", "--help"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "--allow-unscored" in out.stdout


def test_grade_pos_threads_allow_unscored_to_the_grader(tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    priv = tmp_path / "priv"
    _write_jsonl(priv / bp._pos_runs(cfg)["si_irrelevant"] / "bon.jsonl", [
        _store_row("bon", "g0", 0, 0, "prompted", _hprompt(0),
                   "SYNTHETIC irrelevant candidate 0 reply for 0", None,
                   si_id="si_irrelevant", cidx=0)])
    seen = {}

    def _build(a, c, d):
        seen["allow_unscored"] = int(getattr(a, "allow_unscored", 0) or 0)
        return lambda items: {it: 0.0 for it in items}

    monkeypatch.setattr(bp, "_build_grader", _build)
    bp.cmd_grade_pos(types.SimpleNamespace(
        private_root=str(priv), dry_run=False, concurrency=4,
        si="si_irrelevant", allow_unscored=7), cfg)
    assert seen["allow_unscored"] == 7


# --------------------------------------------------------------------------- #
# the store SI flip (2026-08-24): safety_refusal as a GENERATED pos SI          #
# --------------------------------------------------------------------------- #


def test_pos_runs_includes_store_si_only_when_declared(tmp_path):
    cfg = _stage3_cfg(tmp_path)
    runs = bp._pos_runs(cfg)                       # shipped config: declared
    assert runs["safety_refusal"] == "two_si_pos_safety_refusal"
    assert bp._store_pos_si(cfg) is None           # store rows excluded

    legacy = dict(cfg, pool={
        **cfg["pool"],
        "pos_runs": {k: v for k, v in cfg["pool"]["pos_runs"].items()
                     if k != "safety_refusal"}})
    lruns = bp._pos_runs(legacy)                   # undeclared: absent, as ever
    assert "safety_refusal" not in lruns
    assert lruns["si_constitution"] == "two_si_pos_constitution"
    assert bp._store_pos_si(legacy) == "safety_refusal"


def test_generated_pos_wins_over_conflicting_store_rows(tmp_path):
    """THE two-source precedence pin. The store holds safety_refusal rows of
    BOTH encodings — an explicit si_id row and a legacy bare-`prompted` row —
    each at candidate_idx 0 with a COMPLYING verdict (score 0.9), while the
    generated run's candidate 0 THWARTS (score 0.0). Without the exclusion
    both sources land in one candidate set and matched_pool's `place_cand`
    raises `conflicting graded verdicts` on the idx-0 collision; with it the
    manifest builds cleanly, the frozen pos is the RUN row, and the store
    drafts' shas appear nowhere in the artifact."""
    cfg = _stage3_cfg(tmp_path)
    store = Path(cfg["pool"]["store_dir"])
    store_draft = "SYNTHETIC store jailbroken sr draw for {}"
    rows = [json.loads(l) for l in
            (store / "bon.jsonl").read_text().splitlines()]
    for i in range(2):
        # the explicit-si encoding (what the pool_sample store lane writes)
        rows.append(_store_row("bon", f"g{i}", i, 0, "prompted", _hprompt(i),
                               store_draft.format(i), 0.9,
                               si_id="safety_refusal", cidx=0))
    _write_jsonl(store / "bon.jsonl", rows)
    # sanity: the store rows CONFLICT with the run rows at candidate_idx 0
    # (0.9 complies, the run's 0.0 thwarts) — the crash the exclusion prevents.

    priv = tmp_path / "priv"
    _seed_pos_runs(priv, cfg)
    _seed_benign_run(priv, cfg)
    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None),
                    cfg)
    m = json.loads(Path(cfg["pool_manifest"]).read_text())
    assert m["regime"] == "matched_intersection"
    assert m["pool_sizes"] == {si: 4 for si in SI3}
    blob = Path(cfg["pool_manifest"]).read_text()
    for i in range(2):
        run_sha = text_sha(_hpos("safety_refusal", i))
        for p in m["matched_intersection"]["ordered_pairs"]:
            if p["family"] == "bon" and p["goal_id"] == f"g{i}":
                assert p["pos_sha"]["safety_refusal"] == run_sha
        assert text_sha(store_draft.format(i)) not in blob


def test_store_si_legacy_behaviour_without_declared_run(tmp_path):
    """With pool.pos_runs.safety_refusal UNDECLARED nothing changes: the
    store's bare-`prompted` rows still serve safety_refusal's pos, no
    run dir is needed, and the provenance block says so."""
    cfg = _stage3_cfg(tmp_path)
    priv = tmp_path / "priv"
    _seed_pos_runs(priv, cfg)                       # writes an sr run too --
    import shutil                                   # remove it: legacy world
    shutil.rmtree(priv / "two_si_pos_safety_refusal")
    cfg["pool"] = {**cfg["pool"],
                   "pos_runs": {k: v for k, v in cfg["pool"]["pos_runs"].items()
                                if k != "safety_refusal"}}
    _seed_benign_run(priv, cfg)
    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None),
                    cfg)
    m = json.loads(Path(cfg["pool_manifest"]).read_text())
    assert m["pool_sizes"] == {si: 4 for si in SI3}
    for p in m["matched_intersection"]["ordered_pairs"]:
        if p["family"] == "bon":
            i = int(p["goal_id"][1:])
            assert p["pos_sha"]["safety_refusal"] == text_sha(
                _hpos("safety_refusal", i))         # the store `prompted` row
    prov = m["pos_provenance"]["safety_refusal"]
    assert prov["store_rows"] == "admitted"
    assert "no generated pos run is declared" in prov["note"]


def test_manifest_pos_provenance_block(tmp_path):
    cfg = _stage3_cfg(tmp_path)
    priv = tmp_path / "priv"
    _seed_pos_runs(priv, cfg)
    _seed_benign_run(priv, cfg)
    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None),
                    cfg)
    m = json.loads(Path(cfg["pool_manifest"]).read_text())
    prov = m["pos_provenance"]
    assert set(prov) == set(SI3)
    sr = prov["safety_refusal"]
    assert sr["store_rows"] == "excluded"
    assert sr["generated_run"] == "two_si_pos_safety_refusal"
    assert sr["configured_k"] == 8                  # the shipped config's K
    assert "MIGRATION" in sr["note"]                # the comparability warning
    for si in ("si_constitution", "si_irrelevant"):
        assert prov[si]["store_rows"] == "admitted"
    # realised stats mirror the manifest's own rejection_sampling block
    assert (prov["si_irrelevant"]["realised_k_max"]
            == m["rejection_sampling"]["si_irrelevant"]["k_max"])
    # the hash was re-embedded after the block was added (write_manifest
    # verified it on freeze; verify once more against the file content)
    from sisteer.eval.matched_pool import manifest_hash
    assert m["manifest_sha256"] == manifest_hash(m)


def test_pos_plan_relaxed_base_and_store_coverage(tmp_path):
    """Two _pos_plan properties the flip depends on: (1) the plan covers the
    DECISION-1 relaxed harmful base (every GRADED none train prompt), not just
    the none-successful subset; (2) the STORE's explicit-si rows count as
    coverage for a store-sourced SI (the store draw is candidate 0, a raised K
    tops it up from candidate 1) but NOT for the flipped store SI (its run
    must stand alone)."""
    cfg = _stage3_cfg(tmp_path)
    store = Path(cfg["pool"]["store_dir"])
    rows = [
        _store_row("bon", "g0", 0, 0, "none", _hprompt(0), _hneg(0), 0.9),
        _store_row("bon", "g1", 1, 0, "none", _hprompt(1), _hneg(1), 0.2),
        _store_row("bon", "g2", 2, 0, "none", _hprompt(2), _hneg(2), None),
        # constitution store rows: g0 accepted at idx 0, g1 rejected at idx 0
        _store_row("bon", "g0", 0, 0, "prompted", _hprompt(0),
                   _hpos("si_constitution", 0), 0.0,
                   si_id="si_constitution", cidx=0),
        _store_row("bon", "g1", 1, 0, "prompted", _hprompt(1),
                   "SYNTHETIC jailbroken constitution draw 1", 0.9,
                   si_id="si_constitution", cidx=0),
        # safety_refusal store rows exist and even THWART — still not coverage
        _store_row("bon", "g0", 0, 0, "prompted", _hprompt(0),
                   _hpos("safety_refusal", 0), 0.0,
                   si_id="safety_refusal", cidx=0),
        _store_row("bon", "g1", 1, 0, "prompted", _hprompt(1),
                   _hpos("safety_refusal", 1), 0.0,
                   si_id="safety_refusal", cidx=0),
    ]
    _write_jsonl(store / "bon.jsonl", rows)
    train_goals, eval_goals = bp._goal_split_from_store(cfg, ("bon",))
    spec = bp._pool_spec(cfg, train_goals, eval_goals)

    # (1) + (2a): constitution — base holds g0 (complied) AND g1 (refused,
    # relaxed); the UNGRADED g2 is out. g0 is covered by its store draw; g1
    # needs K-1 more starting at candidate 1 (no idx-0 collision).
    none_prompt, base, covered, plan = bp._pos_plan(
        cfg, spec, ("bon",), si="si_constitution",
        run_name="two_si_pos_constitution", k=4, private_root=str(tmp_path))
    pid = lambda g: ("bon", g, f"{int(g[1:])}:0")   # noqa: E731
    assert base == {pid("g0"), pid("g1")}
    assert covered == [pid("g0")]
    # F9 plan shape: the exact missing indices + the exact set on disk — the
    # store draw owns candidate 0, so g1's top-up starts at candidate 1.
    assert plan == [(pid("g1"), (1, 2, 3), frozenset({0}))]

    # (2b): the flipped store SI — the same store rows are NOT coverage; the
    # plan is a fresh K from candidate 0 for the whole base.
    _np, base_sr, covered_sr, plan_sr = bp._pos_plan(
        cfg, spec, ("bon",), si="safety_refusal",
        run_name="two_si_pos_safety_refusal", k=4,
        private_root=str(tmp_path))
    assert base_sr == base and covered_sr == []
    assert plan_sr == [(pid("g0"), (0, 1, 2, 3), frozenset()),
                       (pid("g1"), (0, 1, 2, 3), frozenset())]


def test_store_row_grade_kind_respects_flip():
    explicit = {"attack": "bon", "goal_id": "g0", "sample_idx": 0, "trial": 0,
                "si_condition": "prompted", "si_id": "safety_refusal",
                "attack_prompt": "SYNTHETIC p", "response": "SYNTHETIC r"}
    bare = {k: v for k, v in explicit.items() if k != "si_id"}
    const = {**explicit, "si_id": "si_constitution"}
    # store still serves the SI (default) — all three are pool rows
    assert bp._store_row_grade_kind(explicit, SI3, None) == "si"
    assert bp._store_row_grade_kind(bare, SI3, None) == "si"
    assert bp._store_row_grade_kind(const, SI3, None) == "si"
    # flipped (store_pos_si=None): only the store SI's rows stop being pool rows
    assert bp._store_row_grade_kind(explicit, SI3, None,
                                    store_pos_si=None) is None
    assert bp._store_row_grade_kind(bare, SI3, None, store_pos_si=None) is None
    assert bp._store_row_grade_kind(const, SI3, None,
                                    store_pos_si=None) == "si"


# The configured-K>1-forces-a-sampled-candidate-0 behaviour (GenSpec.sample /
# the staged-round draw) is covered against this branch's dual-engine
# cmd_generate_pos by test_pool_vllm_generation.py::
# test_configured_k_forces_a_sampled_candidate_zero and
# ::test_greedy_single_draw_zeroes_the_sampler.


def test_generate_pos_masks_store_candidate0_for_a_store_served_si(
        tmp_path, monkeypatch):
    """For an SI the store still serves (si_constitution: store draw IS
    candidate 0), generate-pos must top up from candidate 1 — never regenerate
    candidate 0, which would collide with the store draw at candidate_idx 0.
    The plan folds the store coverage into each prompt's `existing` set, so
    its `missing` tuple starts at candidate 1 — the same plan both engines
    consume (the run-file resume alone can't see the store row, which lives
    outside the run file)."""
    priv = tmp_path / "priv"
    cfg = _stage3_cfg(tmp_path, n_prompts=1)
    # add the store's si_constitution candidate-0 draw that COMPLIED (score
    # 0.9 > threshold): not accepted, so the prompt is residue needing more.
    store = tmp_path / "labeled" / "bon.jsonl"
    rows = [json.loads(l) for l in store.read_text().splitlines()]
    rows.append(_store_row("bon", "g0", 0, 0, "prompted", _hprompt(0),
                           _hpos("si_constitution", 0), 0.9,
                           si_id="si_constitution", cidx=0))
    _write_jsonl(store, rows)
    _stub_generation(monkeypatch)

    bp.cmd_generate_pos(types.SimpleNamespace(
        private_root=str(priv), dry_run=False, batch_size=4,
        si="si_constitution", candidates=3), cfg)
    run = bp._pos_runs(cfg)["si_constitution"]
    out = [json.loads(l) for l in
           (priv / run / "bon.jsonl").read_text().splitlines() if l.strip()]
    # candidate 0 was masked (store owns it); the run tops up 1..2 only
    assert {r["candidate_idx"] for r in out} == {1, 2}


def test_candidate_seed_is_distinct_reproducible_and_overridable():
    """The rejection-sampling determinism guarantee, unit-tested
    without importing vLLM: distinct draws per candidate, reproducible from
    the base seed + candidate index alone, and the sample override decoupled
    from k."""
    import sisteer.sampling.engine as E

    k8 = E.GenSpec(si_id="si_x", system_text="sys", k=8, temperature=1.0)
    seeds = [E.candidate_seed(0, k8, ci) for ci in range(8)]
    assert seeds == [1000 * ci for ci in range(8)]          # distinct per ci
    assert len(set(seeds)) == 8
    # reproducible from (base_seed, candidate_idx) alone, base offset honoured
    assert E.candidate_seed(0, k8, 3) == E.candidate_seed(0, k8, 3) == 3000
    assert E.candidate_seed(7, k8, 2) == 7 + 2000

    # k == 1 with no override is the greedy teacher pass: no RNG seed
    k1 = E.GenSpec(si_id="si_x", system_text="sys", k=1)
    assert E._greedy(k1) is True and E.candidate_seed(0, k1, 0) is None

    # sample=True forces a real draw even at k==1 (the staged-round case)
    k1s = E.GenSpec(si_id="si_x", system_text="sys", k=1, sample=True)
    assert E._greedy(k1s) is False and E.candidate_seed(0, k1s, 0) == 0
    # sample=False forces greedy even at k>1 (symmetry; not used by pos)
    k8g = E.GenSpec(si_id="si_x", system_text="sys", k=8, sample=False)
    assert E._greedy(k8g) is True and E.candidate_seed(0, k8g, 5) is None


# --------------------------------------------------------------------------- #
# generate-benign / grade-benign                                                #
# --------------------------------------------------------------------------- #


def _phtest_csv(tmp_path):
    import pandas as pd

    prompts = [_bprompt(0), _bprompt(1)]
    pd.DataFrame([{"ID": i, "Request": p, "Harmfulness": "harmless"}
                  for i, p in enumerate(prompts)]).to_csv(
        tmp_path / "PHTest.csv", index=False)
    return str(tmp_path / "PHTest.csv"), prompts


def test_generate_benign_writes_none_plus_per_si_rows_and_resumes(
        tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path)
    csv, prompts = _phtest_csv(tmp_path)
    priv = tmp_path / "priv"
    calls = _stub_generation(monkeypatch)

    args = types.SimpleNamespace(private_root=str(priv), dry_run=False,
                                 batch_size=4, phtest_path=csv)
    bp.cmd_generate_benign(args, cfg)
    path = priv / bp._benign_cfg(cfg)[1] / "phtest.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    # 2 prompts x (1 none + 3 SIs) = 8 rows, all ungraded/unlabelled
    assert len(rows) == 8
    kinds = {(r["goal_id"], r.get("si_id") or r["si_condition"])
             for r in rows}
    assert len(kinds) == 8
    assert all(r["xstest_label"] is None for r in rows)
    assert {r.get("si_id") for r in rows if r.get("si_id")} == set(SI3)

    # resume: a second run regenerates nothing
    n_calls = len(calls["rendered"])
    bp.cmd_generate_benign(args, cfg)
    assert len(calls["rendered"]) == n_calls
    assert len([json.loads(l) for l in path.read_text().splitlines()]) == 8


def test_grade_benign_labels_si_rows_only_and_manifest_consumes_them(
        tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path, n_prompts=2)
    csv, prompts = _phtest_csv(tmp_path)
    priv = tmp_path / "priv"
    _stub_generation(monkeypatch)
    bp.cmd_generate_benign(types.SimpleNamespace(
        private_root=str(priv), dry_run=False, batch_size=4,
        phtest_path=csv), cfg)

    labelled = {}

    def _fake_xs(a, c, d):
        def grade(items):
            labelled["items"] = list(items)
            return {it: "C" for it in items}
        return grade

    monkeypatch.setattr(bp, "_build_xstest_grader", _fake_xs)
    bp.cmd_grade_benign(types.SimpleNamespace(
        private_root=str(priv), dry_run=False, concurrency=4), cfg)
    # only the 6 SI rows were graded — never the 2 none rows
    assert len(labelled["items"]) == 6
    path = priv / bp._benign_cfg(cfg)[1] / "phtest.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert all(r["xstest_label"] == "C" for r in rows if r.get("si_id"))
    assert all(r["xstest_label"] is None for r in rows if not r.get("si_id"))

    # the manifest folds the labelled benign half in (harmful side seeded too)
    _seed_pos_runs(priv, cfg)
    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None),
                    cfg)
    m = json.loads(Path(cfg["pool_manifest"]).read_text())
    assert m["benign"]["n_matched_all_si_comply"] == 2
    assert m["composition"]["realised"]["benign_fraction"] == 0.5


def test_grade_benign_failure_writes_nothing(tmp_path, monkeypatch):
    cfg = _stage3_cfg(tmp_path)
    csv, _ = _phtest_csv(tmp_path)
    priv = tmp_path / "priv"
    _stub_generation(monkeypatch)
    bp.cmd_generate_benign(types.SimpleNamespace(
        private_root=str(priv), dry_run=False, batch_size=4,
        phtest_path=csv), cfg)

    def _boom(a, c, d):
        def grade(items):
            raise RuntimeError("judge item(s) still UNSCORED (simulated)")
        return grade

    monkeypatch.setattr(bp, "_build_xstest_grader", _boom)
    with pytest.raises(RuntimeError, match="UNSCORED"):
        bp.cmd_grade_benign(types.SimpleNamespace(
            private_root=str(priv), dry_run=False, concurrency=4), cfg)
    path = priv / bp._benign_cfg(cfg)[1] / "phtest.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert all(r["xstest_label"] is None for r in rows)      # untouched


# --------------------------------------------------------------------------- #
# the scoring-disjointness loader + the shipped config's declared intent        #
# --------------------------------------------------------------------------- #


def test_scoring_prompt_shas_loads_declared_controls_from_paths(tmp_path):
    import pandas as pd

    hard = ["SYNTHETIC hard benign eval prompt one for holdout",
            "SYNTHETIC hard benign eval prompt two for holdout"]
    pd.DataFrame({"prompt": hard, "category": ["x", "y"]}).to_csv(
        tmp_path / "hard.csv", index=False)
    pd.DataFrame([
        {"id": 1, "prompt": "SYNTHETIC xstest safe prompt here",
         "type": "homonyms", "label": "safe", "focus": "f", "note": ""},
        {"id": 2, "prompt": "SYNTHETIC xstest unsafe contrast prompt",
         "type": "homonyms", "label": "unsafe", "focus": "f", "note": ""},
    ]).to_csv(tmp_path / "xstest.csv", index=False)

    cfg = {"pool": {"scoring_disjointness": {
        "or_bench_hard_1k": {"path": str(tmp_path / "hard.csv")},
        "xstest": {"path": str(tmp_path / "xstest.csv")},
    }}, "datasets": {"xstest": {"revision": "rev"}}}
    shas = bp._scoring_prompt_shas(cfg)
    expected = {normalized_text_sha(t) for t in
                hard + ["SYNTHETIC xstest safe prompt here"]}
    assert shas == expected                       # unsafe rows are NOT scoring

    with pytest.raises(SystemExit, match="unknown scoring_disjointness"):
        bp._scoring_prompt_shas({"pool": {"scoring_disjointness":
                                          {"bogus": {}}}})


def test_manifest_refuses_an_eval_control_as_benign_source(tmp_path):
    cfg = _stage3_cfg(tmp_path)
    cfg["pool"]["benign_source"] = "xstest"
    with pytest.raises(ValueError, match="EVAL"):
        bp.cmd_manifest(types.SimpleNamespace(
            private_root=str(tmp_path / "priv"), out=None), cfg)


def test_shipped_config_declares_the_stage3_pool_intent():
    cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    pool = cfg["pool"]
    # all eight static families, declared for the
    # composition assertion to check against
    # eight attack transforms + the standalone wildjailbreak_harmful coverage
    # family
    # FIVE headroom families + the wildjailbreak filler, cutting from twelve.
    # The dropped families -- pap, encoding, aim, gcg_replay, amplegcg_plus --
    # all sit at near-floor undefended baselines (0-2.7% ASR), so no defense
    # can move them and generating them buys rows that cannot discriminate.
    harmful = ["bon", "pasttense", "codeattack", "renellm", "grok420",
               "wildjailbreak_harmful"]
    assert pool["static_families"] == harmful
    assert pool["train_families"] == harmful
    assert pool["family_disjointness"] is False
    # the five headroom families appear on BOTH sides, by design
    for fam in ("bon", "pasttense", "codeattack", "renellm", "grok420"):
        assert fam in pool["train_families"] and fam in pool["eval_families"]
    # wildjailbreak_harmful capped before the 50/50 balance
    assert pool["family_caps"] == {"wildjailbreak_harmful": 5000}
    # the benign half and the per-side floor. RAISED 2026-08-20
    # (well-determined probe fits): the per-side guard went from 200 to
    # 2300 (~half the ~4,600 dimension target at 50/50).
    assert pool["benign_source"] == "phtest"
    assert pool["benign_fraction"] == 0.5
    assert pool["benign_fraction_tol"] == 0.02
    assert pool["hard_floor_pairs"] == 2300
    # the DIMENSION-AWARE floor (the authoritative n >= d gate): the alphakl
    # probes at d = 4096, with the alphakl 64-row calibration carve declared.
    ff = pool["fit_floor"]
    assert ff["dims"] == {"alphakl_null": 4096, "alphakl_soft": 4096,
                          "alphakl_fisher": 4096}
    assert ff["calibration_rows"] == {"alphakl_null": 64, "alphakl_soft": 64,
                                      "alphakl_fisher": 64}
    # the broadened benign stratum: PHTest + the two cluster-only corpora,
    # each its own labelled family/source
    # PHTest harmless + wildjailbreak adversarial_benign only
    # (alphasteer_benign dropped; "explicitly enough")
    assert [s["name"] for s in pool["benign_sources"]] == [
        "phtest", "wildjailbreak_benign"]
    assert pool["benign_sources"][0]["split"] == "harmless"
    # the relaxed requirement-(a) stratum is declared but OFF by default
    assert pool["relaxed_stratum"]["enabled"] is False
    # the irrelevant SI's pos run + raised K
    assert pool["pos_runs"]["si_irrelevant"] == "two_si_pos_irrelevant"
    assert pool["pos_candidates"]["si_irrelevant"] >= 2
    # train/scoring disjointness names both over-refusal eval controls
    assert set(pool["scoring_disjointness"]) == {"or_bench_hard_1k", "xstest"}
    # the fit-side stage-3 keys ride in the same config
    assert cfg["wandb_fits"] is True
    assert cfg["wandb_project"] == "$WANDB_PROJECT"          # the site's W&B project (src/sisteer/site.py)
    assert cfg["wandb_entity"] is None
    tk = cfg["train_kwargs"]["default"]
    assert tk["holdout_frac"] == 0.1
    assert tk["holdout_salt"] == "two_si_holdout"
    # user ruling 2026-08-24: a distillation fit must BEAT doing nothing
    # (ratio < 1.0, each object against its own SI's no-op floor); the 3.0
    # bar survives only as the per-arm backstop of the NON-distillation
    # objectives below, and the probe's sample size is pinned at the
    # measured-floor n (job 187947).
    assert tk["floor_ratio_max"] == 1.0
    assert tk["probe_n"] == 256
    assert tk["positions"] == "all" and tk["response_header_len"] is None
    # alphakl carries NO per-arm bar: it minimises a KL to the same teacher,
    # so it faces the default 1.0 with the rest of the distillation group.
    # Pinned as an absence — the 3.0 exemption it briefly held would have
    # spared the very arm whose divergence prompted the 1.0 ruling. Its
    # beta is ALSO pinned as an absence (2026-08-24): the old beta: 1.0
    # was a crash workaround promoted to a hyperparameter and the arm
    # diverged under it; stage 2 now derives beta against the eps benign
    # budget when the key is absent.
    assert cfg["train_kwargs"]["alphakl_soft"] == {}
    for arm in ("alphakl_soft", "alphakl_null"):
        assert "floor_ratio_max" not in (cfg["train_kwargs"].get(arm) or {}), arm
    # the surviving 3.0 backstops are the arms that never optimise a KL to
    # the teacher: the two preference objectives and the closed-form baseline
    for arm in ("bidpo", "bidpo_lora", "diffmean"):
        assert cfg["train_kwargs"][arm]["floor_ratio_max"] == 3.0, arm
    # psr_kl_all's per-arm opt-in (the runaway KL arm). warmup_steps
    # deliberately absent = train_psr's min(500, total//10).
    #
    # The lr is pinned and LOAD-BEARING, not decoration: train_psr defaults
    # to 5e-3, and at 5e-3 this arm diverges WITH the schedule already on
    # (soft_loss 6.797, gate_reg 12.07; its C partner NaN'd). At 5e-4 with
    # the identical schedule it is stable at gate_reg 0.0. So the schedule
    # alone does not hold this arm and an unset lr silently reinstates the
    # divergence. Pinned as a PRESENCE for that reason -- a future edit
    # dropping it would reintroduce a measured failure.
    assert cfg["train_kwargs"]["psr_kl_all"] == {
        "lr": 5.0e-4, "lr_schedule": "warmup_cosine", "max_grad_norm": 1.0}


# --------------------------------------------------------------------------- #
# the DIMENSION-AWARE floor                                                     #
# --------------------------------------------------------------------------- #


def _dim_spec(**over):
    # balanced _world(6, 6) -> 12 pairs; n_fit = floor(0.9*12) - calib
    base = dict(fit_dims={"alphakl_null": 8}, fit_holdout_frac=0.1,
                fit_calibration_rows={"alphakl_null": 2}, hard_floor_pairs=1)
    base.update(over)
    return _spec3(**base)


def test_dimension_floor_fires_with_realised_n_and_d_in_the_message():
    # 12 balanced pairs -> n_fit = floor(0.9*12) - 2 = 8; d = 9 -> FAIL
    with pytest.raises(PoolDimensionError) as e:
        build_matched_pool(_world(6, 6), _dim_spec(fit_dims={"alphakl_null": 9}))
    msg = str(e.value)
    assert "alphakl_null: n = 8 vs d = 9 (FAIL)" in msg
    assert "rank-deficient" in msg
    assert "12 pairs" in msg                       # the balanced pool size


def test_dimension_floor_passes_when_n_meets_d_and_records_the_block():
    m = build_matched_pool(_world(6, 6), _dim_spec(fit_dims={"alphakl_null": 8}))
    df = m["dimension_floor"]
    assert df["enabled"] is True and df["all_arms_meet_dimension"] is True
    assert df["n_matched"] == 12 and df["holdout_frac"] == 0.1
    assert df["per_arm"] == [{"arm": "alphakl_null", "d": 8, "n_fit": 8,
                              "calibration_rows": 2, "ok": True}]
    assert df["binding_arm"] == "alphakl_null"


def test_dimension_floor_is_a_hard_floor_error_subclass():
    # so any existing `except PoolFloorError` still catches the dimension gate
    assert issubclass(PoolDimensionError, PoolFloorError)
    with pytest.raises(PoolFloorError):
        build_matched_pool(_world(6, 6), _dim_spec(fit_dims={"alphakl_null": 99}))


def test_dimension_floor_names_the_binding_arm_across_arms():
    # every arm passes; the binding arm is the largest d + calibration carve
    m = build_matched_pool(
        _world(6, 6),
        _dim_spec(fit_dims={"small": 2, "alphakl_null": 6},
                  fit_calibration_rows={"small": 0, "alphakl_null": 2}))
    assert m["dimension_floor"]["binding_arm"] == "alphakl_null"
    assert m["dimension_floor"]["all_arms_meet_dimension"] is True


def test_dimension_floor_off_when_no_fit_dims():
    m = build_matched_pool(_world(6, 6), _spec3(hard_floor_pairs=1))
    assert "dimension_floor" not in m


# --------------------------------------------------------------------------- #
# multi-source benign stratum (scale-up amendment 2026-08-20)                    #
# --------------------------------------------------------------------------- #


def _benign_fam(i, fam="phtest", *, si_rows=SI3, labels=None):
    """One benign prompt tagged with corpus family ``fam`` (each corpus is its
    own labelled family, mirroring the attack families)."""
    labels = labels or {}
    gid = f"{fam}_{i}"
    rows = [_gr(fam, gid, "0", "none", None, _bprompt(i), _bneg(i))]
    for si in si_rows:
        lab = labels.get(si, "C")
        resp = (_bpos(si, i) if lab == "C"
                else f"SYNTHETIC hedged half answer from {si} {i}")
        rows.append(_gr(fam, gid, "0", "si", si, _bprompt(i), resp, label=lab))
    return rows


def test_multi_source_benign_counts_each_corpus_as_its_own_family():
    rows = [*_harmful(0, fam="bon"), *_harmful(1, fam="pasttense")]
    rows += _benign_fam(100, "phtest")
    rows += _benign_fam(200, "wildjailbreak_benign")
    spec = _spec3(benign_source=("phtest", "wildjailbreak_benign"),
                  hard_floor_pairs=1)
    m = build_matched_pool(rows, spec)
    assert m["composition"]["realised"]["per_family_counts"] == {
        "bon": 1, "pasttense": 1, "phtest": 1, "wildjailbreak_benign": 1}
    assert m["composition"]["realised"]["benign_fraction"] == 0.5
    # declared benign_source serialises as a sorted list (JSON-safe), never a
    # frozenset/tuple
    assert m["composition"]["declared"]["benign_source"] == [
        "phtest", "wildjailbreak_benign"]
    b = m["benign"]
    assert b["sources"] == ["phtest", "wildjailbreak_benign"]
    assert b["per_source_yields"]["phtest"]["n_matched_all_si_comply"] == 1
    assert (b["per_source_yields"]["wildjailbreak_benign"]
            ["n_matched_all_si_comply"] == 1)
    assert b["all_sources_meet_floor"] is True


def test_benign_source_floor_flags_a_declared_but_empty_corpus():
    # phtest supplies the benign half; wildjailbreak_benign is declared with a
    # floor but has NO rows -> it must still appear at n=0 and trip its floor
    rows = _world(2, 0)
    rows += _benign_fam(100, "phtest") + _benign_fam(101, "phtest")
    spec = _spec3(benign_source=("phtest", "wildjailbreak_benign"),
                  hard_floor_pairs=1,
                  source_floors={"wildjailbreak_benign": 1})
    m = build_matched_pool(rows, spec)
    ps = m["benign"]["per_source_yields"]
    assert ps["wildjailbreak_benign"] == {
        "n_candidates": 0, "n_matched_all_si_comply": 0,
        "floor": 1, "floor_ok": False}
    assert m["benign"]["all_sources_meet_floor"] is False


def test_single_string_benign_source_still_serialises_as_a_string():
    m = build_matched_pool(_world(2, 2), _spec3(hard_floor_pairs=1))
    assert m["composition"]["declared"]["benign_source"] == "phtest"
    assert m["benign"]["source"] == "phtest"
    assert m["benign"]["sources"] == ["phtest"]


# --------------------------------------------------------------------------- #
# the RELAXED requirement-(a) stratum                                           #
# --------------------------------------------------------------------------- #


def _harmful_refused(i, *, fam="bon", thwarted=SI3):
    """A harmful prompt the UNDEFENDED model already REFUSED (none score below
    threshold), thwarted by ``thwarted`` under each SI."""
    gid = f"g{i}"
    rows = [_gr(fam, gid, "0:0", "none", None, _hprompt(i), _hneg(i),
                score=0.1)]
    for si in SI3:
        thw = si in thwarted
        resp = _hpos(si, i) if thw else f"SYNTHETIC jailbroken reply {si} {i}"
        rows.append(_gr(fam, gid, "0:0", "si", si, _hprompt(i), resp,
                        score=0.0 if thw else 0.9))
    return rows


def _relaxed_world():
    rows = _world(2, 2)                       # 2 complied-harmful + 2 benign
    rows += _harmful_refused(50) + _harmful_refused(51)   # 2 already-refused
    return rows


def test_relaxed_is_the_default_and_folds_refused_prompts_into_the_primary():
    # requirement (a) is DROPPED by
    # default, so a harmful prompt the UNDEFENDED model already REFUSED but
    # every SI thwarts is now admitted to the PRIMARY pool, labelled
    # requirement_a="relaxed"; the complied-with ones are "strict". There is NO
    # separate relaxed_requirement_a block in the default regime (folded in).
    m = build_matched_pool(
        _relaxed_world(),
        _spec3(train_families=("bon", "pasttense"), benign_source=None,
               benign_fraction=None, hard_floor_pairs=1))
    assert "relaxed_requirement_a" not in m
    pairs = m["matched_intersection"]["ordered_pairs"]
    by_goal = {p["goal_id"]: p["requirement_a"] for p in pairs}
    # the two already-refused prompts ARE in the primary now, labelled relaxed
    assert by_goal.get("g50") == "relaxed" and by_goal.get("g51") == "relaxed"
    # the complied-with harmful prompts stay identifiable as the strict subset
    assert by_goal.get("g0") == "strict" and by_goal.get("g1") == "strict"
    mi = m["matched_intersection"]
    assert mi["n_relaxed"] == 2 and mi["n_strict"] == 2
    assert mi["requirement_a_required"] is False
    assert m["hard_floor"]["n_strict_harmful_matched"] == 2


def test_relaxed_stratum_is_labelled_and_separate_from_the_primary_pool():
    m = build_matched_pool(
        _relaxed_world(), _spec3(hard_floor_pairs=1,
                                 relaxed_stratum_enabled=True,
                                 require_undefended_compliance=True))
    rel = m["relaxed_requirement_a"]
    assert rel["enabled"] is True and rel["requirement_a"] == "relaxed"
    assert rel["n"] == 2
    relaxed_goals = {p["goal_id"] for p in rel["ordered_pairs"]}
    assert relaxed_goals == {"g50", "g51"}
    # NEVER mixed into the primary matched intersection
    primary_goals = {p["goal_id"] for p in m["matched_intersection"]["ordered_pairs"]}
    assert primary_goals.isdisjoint(relaxed_goals)
    # each relaxed pair carries one pos_sha per SI (identity invariant held)
    assert all(set(p["pos_sha"]) == set(SI3) for p in rel["ordered_pairs"])
    assert rel["per_si_thwarted"] == {si: 2 for si in SI3}


def test_relaxed_stratum_needs_all_sis_to_thwart_the_refused_prompt():
    rows = _world(1, 1)
    rows += _harmful_refused(50)                                    # all 3 -> in
    rows += _harmful_refused(51, thwarted=("safety_refusal",))      # 1 of 3 -> out
    m = build_matched_pool(rows, _spec3(train_families=("bon",),
                                        hard_floor_pairs=1,
                                        relaxed_stratum_enabled=True,
                                        require_undefended_compliance=True))
    relaxed_goals = {p["goal_id"] for p in
                     m["relaxed_requirement_a"]["ordered_pairs"]}
    assert relaxed_goals == {"g50"}


# --------------------------------------------------------------------------- #
# the matched invariant still holds at the raised scale                          #
# --------------------------------------------------------------------------- #


def test_matched_invariant_holds_at_larger_scale():
    m = build_matched_pool(_world(50, 50), _spec3(hard_floor_pairs=10))
    assert m["matched_intersection"]["n"] == 100         # 50 harmful + 50 benign
    assert m["composition"]["realised"]["benign_fraction"] == 0.5
    # the five identity invariants ran on 100 pairs and held (else it raised)
    assert m["assertions"]["matched_pools_identical_except_pos"] is True
    # dimension floor at this scale: d must be <= n_fit = floor(0.9*100)=90
    m2 = build_matched_pool(
        _world(50, 50),
        _spec3(hard_floor_pairs=10, fit_dims={"alphakl_null": 90},
               fit_holdout_frac=0.1))
    assert m2["dimension_floor"]["all_arms_meet_dimension"] is True
    with pytest.raises(PoolDimensionError):
        build_matched_pool(
            _world(50, 50),
            _spec3(hard_floor_pairs=10, fit_dims={"alphakl_null": 91},
                   fit_holdout_frac=0.1))


def test_new_benign_corpora_stay_disjoint_from_eval_controls_at_pool_level():
    # a wildjailbreak_benign prompt that is ALSO an eval-control (scoring)
    # prompt must be stripped before the intersection — enforced by
    # _strip_scoring, regardless of which benign corpus supplied it.
    rows = [*_harmful(0, fam="bon"), *_harmful(1, fam="pasttense")]
    rows += _benign_fam(100, "phtest") + _benign_fam(101, "phtest")
    rows += _benign_fam(200, "wildjailbreak_benign")          # this is the leak
    scoring = frozenset({normalized_text_sha(_bprompt(200))})
    spec = _spec3(benign_source=("phtest", "wildjailbreak_benign"),
                  hard_floor_pairs=1, scoring_prompt_shas=scoring)
    m = build_matched_pool(rows, spec)
    assert m["scoring_disjointness"]["n_removed_benign"] == 1
    goals = {p["goal_id"] for p in m["matched_intersection"]["ordered_pairs"]}
    assert "wildjailbreak_benign_200" not in goals


def test_all_amendment_features_together_freeze_reproducibly():
    # dimension floor + multi-source benign + relaxed stratum all ON, and the
    # frozen manifest is byte-reproducible and passes the containment walk.
    rows = _relaxed_world()
    rows += _benign_fam(300, "wildjailbreak_benign")
    rows += _benign_fam(301, "wildjailbreak_benign")
    spec = _spec3(benign_source=("phtest", "wildjailbreak_benign"),
                  hard_floor_pairs=1, relaxed_stratum_enabled=True,
                  require_undefended_compliance=True,
                  fit_dims={"alphakl_null": 2}, fit_holdout_frac=0.1,
                  fit_calibration_rows={"alphakl_null": 0})
    m1 = build_matched_pool(rows, spec)
    m2 = build_matched_pool(rows, spec)
    assert dumps_manifest(m1) == dumps_manifest(m2)
    assert m1["manifest_sha256"] == m2["manifest_sha256"]
    assert m1["dimension_floor"]["all_arms_meet_dimension"] is True
    assert m1["relaxed_requirement_a"]["n"] == 2
    assert set(m1["benign"]["sources"]) == {"phtest", "wildjailbreak_benign"}


def test_family_cap_inverts_the_balance_binding_side():
    """The family cap applied BEFORE the 50/50 balance can
    flip which side is binding: trimming the dominant harmful family below the
    benign matched count makes BENIGN the side the balance cuts."""
    rows = []
    for i in range(10):
        rows += _harmful(i, fam="bon")
    for j in range(6):
        rows += _benign(100 + j)

    def spec(**o):
        return _spec3(train_families=("bon",),
                      static_families=frozenset({"bon"}),
                      hard_floor_pairs=1, **o)

    # uncapped: harmful (10) is the larger side, so the balance cuts HARMFUL
    m0 = build_matched_pool(rows, spec())
    pb0 = m0["composition"]["pre_balance"]
    assert (pb0["n_harmful_matched"], pb0["n_benign_matched"]) == (10, 6)
    assert (pb0["harmful_taken"], pb0["benign_taken"]) == (6, 6)  # benign binding

    # capped bon->3: harmful (3) is now SMALLER, so the balance cuts BENIGN
    m1 = build_matched_pool(rows, spec(family_caps={"bon": 3}))
    pb1 = m1["composition"]["pre_balance"]
    assert pb1["n_harmful_matched"] == 3          # capped before the balance
    assert pb1["n_benign_matched"] == 6           # benign untouched pre-balance
    assert (pb1["harmful_taken"], pb1["benign_taken"]) == (3, 3)  # benign cut now
    assert m1["family_caps"]["applied"]["bon"]["n_dropped"] == 7
