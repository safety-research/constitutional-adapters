"""Matched training-pool builder (src/sisteer/eval/matched_pool.py) + its CLI
(scripts/build_two_si_pool.py) — the frozen pool artifact.

CPU-only, offline, no GPU, no harmful text: every prompt/response here is an
INVENTED synthetic string (privacy rule — placeholder uuids, TESTHOST). The
suite proves the correctness properties the experiment's headline rests on:

- the intersection (thwarted_by_safety ∩ thwarted_by_constitution) is computed
  correctly, with the native per-SI pools recorded as a labelled secondary.
- each of the FIVE identity invariants FAILS when violated (five separate
  negative tests — an assertion nobody has seen fail is not known to work).
- an adaptive-attack row can NEVER enter the primary pool, only the
  labelled replay_transfer stratum.
- the manifest is byte-stable across two runs on identical input, and the
  hash covers everything load-bearing.
- the regime field is present in BOTH regimes, and the too-small
  intersection falls back to native pools — never a subsample.
- the no-harmful-text invariant holds (counts/ids/hashes only).
- two-dimensional (family AND goal) train/eval disjointness is asserted.
- the frozen manifest is exactly what two_si_run.aggregate reads back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_two_si_pool as bp  # noqa: E402
from sisteer.eval.matched_pool import (  # noqa: E402
    FORBIDDEN_REGIME,
    GradedRow,
    MatchedPoolError,
    PairRecord,
    PoolAssertionError,
    PoolSpec,
    assert_no_store_text,
    assert_pools_identical_except_pos,
    build_matched_pool,
    dumps_manifest,
    manifest_hash,
    text_sha,
    write_manifest,
)

REPO = Path(__file__).resolve().parents[1]
SI = ("safety_refusal", "si_constitution")

# distinctive synthetic prompt/response strings (>= 12 chars) — grep needles the
# manifest must never contain (no-harmful-text invariant).
PROMPTS = {
    "pA": "SYNTHETIC prompt alpha please do the benign thing 11111111",
    "pB": "SYNTHETIC prompt bravo please do the benign thing 22222222",
    "pC": "SYNTHETIC prompt charlie please do the benign thing 33333333",
    "pD": "SYNTHETIC prompt delta please do the benign thing 44444444",
    "pE": "SYNTHETIC prompt echo please do the benign thing 55555555",
    "adv": "SYNTHETIC adaptive prompt foxtrot benign 66666666",
}
RESP = {
    "none_succ": "SYNTHETIC undefended completion that complied aaaaaaaa",
    "refusal": "SYNTHETIC refusal completion I cannot help bbbbbbbb",
    "jb": "SYNTHETIC jailbroken completion complied cccccccc",
}


def _row(fam, gid, vk, cond, si, prompt, resp, score):
    return GradedRow(family=fam, goal_id=gid, variant_key=vk, condition=cond,
                     si_id=si, prompt=prompt, response=resp, score=score)


def _fixture_rows():
    """A small synthetic world with every relevant case (all thresholds at 0.5).

    The primary intersection does not require the undefended model to have
    COMPLIED, so the base is every prompt with a none row and requirement (a)
    survives as a per-pair label:

      pA  none COMPLIED, safety-thwarted, constitution-thwarted -> MATCHED (strict)
      pB  none COMPLIED, safety-thwarted, constitution JAILBROKEN -> native(safety) only
      pC  none COMPLIED, safety JAILBROKEN, constitution-thwarted -> native(const) only
      pD  none REFUSED, safety-thwarted, constitution-thwarted  -> MATCHED (relaxed:
          undefended already refused so its neg is a refusal — folded into the
          primary pool, labelled requirement_a="relaxed")
      pE  none COMPLIED, safety-thwarted, constitution MISSING   -> coverage gap
      adv adaptive family 'pair': none + both thwarted           -> replay_transfer only
    """
    hi = "held_in"; ho = "held_out"  # noqa: E702 (compact fixture)
    r = []
    # -- pA (bon, g0, 0:0): matched --
    r += [_row("bon", "g0", "0:0", "none", None, PROMPTS["pA"], RESP["none_succ"], 0.9),
          _row("bon", "g0", "0:0", "si", SI[0], PROMPTS["pA"], RESP["refusal"], 0.0),
          _row("bon", "g0", "0:0", "si", SI[1], PROMPTS["pA"], RESP["refusal"], 0.0)]
    # -- pB (bon, g0, 1:0): safety thwarts, constitution does not --
    r += [_row("bon", "g0", "1:0", "none", None, PROMPTS["pB"], RESP["none_succ"], 0.8),
          _row("bon", "g0", "1:0", "si", SI[0], PROMPTS["pB"], RESP["refusal"], 0.1),
          _row("bon", "g0", "1:0", "si", SI[1], PROMPTS["pB"], RESP["jb"], 0.9)]
    # -- pC (bon, g1, 0:0): constitution thwarts, safety does not --
    r += [_row("bon", "g1", "0:0", "none", None, PROMPTS["pC"], RESP["none_succ"], 0.7),
          _row("bon", "g1", "0:0", "si", SI[0], PROMPTS["pC"], RESP["jb"], 0.9),
          _row("bon", "g1", "0:0", "si", SI[1], PROMPTS["pC"], RESP["refusal"], 0.0)]
    # -- pD (bon, g1, 1:0): none failed -> not even a candidate --
    r += [_row("bon", "g1", "1:0", "none", None, PROMPTS["pD"], RESP["refusal"], 0.1),
          _row("bon", "g1", "1:0", "si", SI[0], PROMPTS["pD"], RESP["refusal"], 0.0),
          _row("bon", "g1", "1:0", "si", SI[1], PROMPTS["pD"], RESP["refusal"], 0.0)]
    # -- pE (bon, g2, 0:0): none-success, safety thwarts, NO constitution row --
    r += [_row("bon", "g2", "0:0", "none", None, PROMPTS["pE"], RESP["none_succ"], 0.9),
          _row("bon", "g2", "0:0", "si", SI[0], PROMPTS["pE"], RESP["refusal"], 0.0)]
    # -- adv (adaptive 'pair'): would be matched but the family is not static --
    r += [_row("pair", "g0", "0", "none", None, PROMPTS["adv"], RESP["none_succ"], 0.9),
          _row("pair", "g0", "0", "si", SI[0], PROMPTS["adv"], RESP["refusal"], 0.0),
          _row("pair", "g0", "0", "si", SI[1], PROMPTS["adv"], RESP["refusal"], 0.0)]
    return r


def _spec(**over):
    base = dict(
        si_ids=SI, train_families=("bon",),
        eval_families=("pair", "tap"), train_goal_ids=("g0", "g1", "g2"),
        eval_goal_ids=("g9",), static_families=frozenset({"bon"}),
        threshold=0.5, min_pool_size=1)
    base.update(over)
    return PoolSpec(**base)


# --------------------------------------------------------------------------- #
# 1. the intersection is computed correctly                                     #
# --------------------------------------------------------------------------- #


def test_intersection_and_native_pools_computed_correctly():
    m = build_matched_pool(_fixture_rows(), _spec())
    mi = m["matched_intersection"]
    # both SIs thwart pA (strict, undefended complied) AND pD
    # (relaxed, undefended refused) — both are in the primary pool now.
    assert mi["n"] == 2
    assert mi["n_strict"] == 1 and mi["n_relaxed"] == 1
    assert mi["requirement_a_required"] is False
    ids = [(p["family"], p["goal_id"], p["variant_key"]) for p in mi["ordered_pairs"]]
    assert ids == [("bon", "g0", "0:0"), ("bon", "g1", "1:0")]
    # the per-pair requirement_a label keeps the strict subset identifiable
    assert [p["requirement_a"] for p in mi["ordered_pairs"]] == ["strict", "relaxed"]
    # both SIs supply a pos on the SAME matched prompt; only pos differs
    pos = mi["ordered_pairs"][0]["pos_sha"]
    assert set(pos) == set(SI)
    # pA (strict) neg is the undefended COMPLIANCE; pD (relaxed) neg is a refusal
    assert mi["ordered_pairs"][0]["neg_sha"] == text_sha(RESP["none_succ"])
    assert mi["ordered_pairs"][1]["neg_sha"] == text_sha(RESP["refusal"])

    nat = m["native_per_si"]
    # native = harmful_base ∩ SI-thwarted (no undefended-complied gate).
    # safety thwarts {pA, pB, pD, pE}; constitution {pA, pC, pD}.
    assert nat[SI[0]]["n"] == 4
    assert nat[SI[1]]["n"] == 3
    # per-source (bon): 5 candidates, 4 none-success, 2 matched (1 strict)
    src = m["per_source_yields"]["bon"]
    assert src["n_candidates"] == 5 and src["n_none_success"] == 4
    assert src["n_matched"] == 2 and src["n_strict_matched"] == 1
    # thwart counted over the whole base now, not gated on undefended compliance
    assert src["n_thwarted_safety_refusal"] == 4
    assert src["n_thwarted_si_constitution"] == 3
    # coverage: constitution is missing exactly pE among the harmful base
    cov = m["coverage"][SI[1]]
    assert cov["n_harmful_base"] == 5 and cov["n_none_success"] == 4
    assert cov["n_missing"] == 1
    assert cov["missing_prompt_ids"] == ["bon/g2/0:0"]
    assert m["coverage"][SI[0]]["n_missing"] == 0


# --------------------------------------------------------------------------- #
# 2. the FIVE identity invariants each FAIL when violated                      #
# --------------------------------------------------------------------------- #


def _base_pools():
    def rec(g, pos):
        return PairRecord("bon", g, "0:0", text_sha(f"prompt-{g}"),
                          text_sha(f"neg-{g}"), text_sha(pos))
    a = [rec("g0", "posA0"), rec("g1", "posA1")]
    b = [rec("g0", "posB0"), rec("g1", "posB1")]  # differ ONLY in pos
    return {SI[0]: a, SI[1]: b}


def test_identity_holds_for_pools_that_differ_only_in_pos():
    assert_pools_identical_except_pos(_base_pools())  # no raise


def test_identity_fails_on_reordered_prompt_ids():
    pools = _base_pools()
    pools[SI[1]] = list(reversed(pools[SI[1]]))
    with pytest.raises(PoolAssertionError, match="ordered_prompt_ids"):
        assert_pools_identical_except_pos(pools)


def test_identity_fails_on_per_family_counts():
    pools = _base_pools()
    p = pools[SI[1]][1]
    pools[SI[1]][1] = PairRecord("pair", p.goal_id, p.variant_key,
                                 p.prompt_sha, p.neg_sha, p.pos_sha)
    with pytest.raises(PoolAssertionError, match="per_family_counts"):
        assert_pools_identical_except_pos(pools)


def test_identity_fails_on_per_goal_counts():
    pools = _base_pools()
    p = pools[SI[1]][1]
    pools[SI[1]][1] = PairRecord(p.family, "g_other", p.variant_key,
                                 p.prompt_sha, p.neg_sha, p.pos_sha)
    with pytest.raises(PoolAssertionError, match="per_goal_counts"):
        assert_pools_identical_except_pos(pools)


def test_identity_fails_on_differing_neg_responses():
    pools = _base_pools()
    p = pools[SI[1]][0]
    pools[SI[1]][0] = PairRecord(p.family, p.goal_id, p.variant_key,
                                 p.prompt_sha, text_sha("DIFFERENT-neg"), p.pos_sha)
    with pytest.raises(PoolAssertionError, match="neg_responses"):
        assert_pools_identical_except_pos(pools)


def test_identity_fails_when_prompts_differ_not_only_pos():
    pools = _base_pools()
    p = pools[SI[1]][0]
    pools[SI[1]][0] = PairRecord(p.family, p.goal_id, p.variant_key,
                                 text_sha("DIFFERENT-prompt"), p.neg_sha, p.pos_sha)
    with pytest.raises(PoolAssertionError, match="differ_only_in_pos"):
        assert_pools_identical_except_pos(pools)


# --------------------------------------------------------------------------- #
# 3. an adaptive-attack row can never enter the primary pool                    #
# --------------------------------------------------------------------------- #


def test_adaptive_row_never_enters_primary_pool():
    m = build_matched_pool(_fixture_rows(), _spec())
    adv_prompt_sha = text_sha(PROMPTS["adv"])
    # not in the matched intersection nor either native pool
    all_pair_shas = [p["prompt_sha"] for p in m["matched_intersection"]["ordered_pairs"]]
    for si in SI:
        all_pair_shas += [p["prompt_sha"] for p in m["native_per_si"][si]["ordered_pairs"]]
    assert adv_prompt_sha not in all_pair_shas
    # it IS accounted for, labelled, in the replay_transfer stratum
    rt = m["replay_transfer"]
    assert rt["families"] == ["pair"]
    assert rt["per_family_counts"]["pair"] == 3
    assert rt["n_rows"] == 3


def test_train_families_must_be_static():
    # a train family absent from static_families is rejected at spec time
    with pytest.raises(MatchedPoolError, match="static"):
        _spec(train_families=("bon", "pair"))


# --------------------------------------------------------------------------- #
# 4. byte-stability + hashing                                                   #
# --------------------------------------------------------------------------- #


def test_manifest_byte_stable_across_two_runs(tmp_path):
    rows = _fixture_rows()
    m1 = build_matched_pool(rows, _spec())
    m2 = build_matched_pool(list(rows), _spec())
    assert dumps_manifest(m1) == dumps_manifest(m2)
    assert m1["manifest_sha256"] == m2["manifest_sha256"]
    # and the frozen files are byte-identical
    p1 = write_manifest(m1, tmp_path / "a.json")
    p2 = write_manifest(m2, tmp_path / "b.json")
    assert p1.read_bytes() == p2.read_bytes()


def test_manifest_hash_covers_content_and_write_guards_tamper(tmp_path):
    m = build_matched_pool(_fixture_rows(), _spec())
    assert m["manifest_sha256"] == manifest_hash(m)
    # tamper with a load-bearing field after hashing -> write refuses
    m["matched_intersection"]["n"] = 999
    with pytest.raises(MatchedPoolError, match="manifest_sha256 mismatch"):
        write_manifest(m, tmp_path / "tampered.json")


# --------------------------------------------------------------------------- #
# 5. regime present in BOTH cases; 6. small intersection -> native, no subsample #
# --------------------------------------------------------------------------- #


def test_regime_field_present_in_matched_regime():
    m = build_matched_pool(_fixture_rows(), _spec(min_pool_size=1))
    assert m["regime"] == "matched_intersection"
    assert m["headline_confounded_by_pool_composition"] is False
    # pA (strict) + pD (relaxed) both matched -> identical size 2
    assert m["pool_sizes"] == {SI[0]: 2, SI[1]: 2}          # identical
    assert m["fallback_reason"] is None
    assert "regime_note" in m and m["regime_note"]


def test_small_intersection_falls_back_to_native_never_subsamples():
    # the matched intersection now has 2 pairs (pA strict + pD relaxed);
    # require 3 -> fall back to native per-SI pools.
    m = build_matched_pool(_fixture_rows(), _spec(min_pool_size=3))
    assert m["regime"] == "native_per_si"
    assert m["headline_confounded_by_pool_composition"] is True
    assert m["fallback_reason"] == "intersection_below_min_pool_size"
    # native pools are UNEQUAL and carried as-is — NOT trimmed to a common size
    # (native = harmful_base ∩ SI-thwarted: safety 4, constitution 3)
    assert m["pool_sizes"] == {SI[0]: 4, SI[1]: 3}
    # the forbidden path never appears anywhere in the frozen bytes
    assert FORBIDDEN_REGIME not in dumps_manifest(m)
    assert "subsample" not in dumps_manifest(m)
    # the matched intersection is still recorded (assertions still ran on it)
    assert m["matched_intersection"]["n"] == 2
    assert m["assertions"]["matched_pools_identical_except_pos"] is True


# --------------------------------------------------------------------------- #
# 7. no-harmful-text invariant                                                  #
# --------------------------------------------------------------------------- #


def test_manifest_contains_no_free_text_from_the_store():
    m = build_matched_pool(_fixture_rows(), _spec())
    blob = dumps_manifest(m)
    for t in {**PROMPTS, **RESP}.values():
        assert t not in blob                # no prompt/response text ever
    # and the structural self-check passes (it ran inside build already)
    labels = ({"bon", "pair"} | set(SI) | {"g0", "g1", "g2", "g9"}
              | {"0:0", "1:0", "0"} | {"tap"})
    assert_no_store_text(m, allowed_labels=labels)


def test_assert_no_store_text_rejects_injected_free_text():
    # a crafted manifest carrying a spaced free-text leaf must fail loudly
    bad = {"regime": "matched_intersection",
           "leak": "this is obviously free harmful text with spaces"}
    with pytest.raises(MatchedPoolError, match="not a safe id"):
        assert_no_store_text(bad)


# --------------------------------------------------------------------------- #
# two-dimensional disjointness + conflicting verdicts                           #
# --------------------------------------------------------------------------- #


def test_shared_families_are_legal_and_recorded():
    """Family-level holdout is RETIRED (2026-08-28): the goal split is the
    holdout, so a family on both sides builds fine and the manifest
    records the sharing as design rather than refusing."""
    m = build_matched_pool(_fixture_rows(),
                           _spec(eval_families=("bon", "pair")))
    fam = m["disjointness"]["family"]
    assert fam["disjoint"] is False and fam["overlap"] == ["bon"]


def test_shared_families_with_no_goal_split_refuse():
    """With families shared, an empty eval goal set is NO holdout at all —
    every 'held-out' number would be in-sample; refuse."""
    with pytest.raises(MatchedPoolError, match="nothing is held out"):
        build_matched_pool(_fixture_rows(),
                           _spec(eval_families=("bon", "pair"),
                                 eval_goal_ids=()))


def test_goal_disjointness_asserted():
    with pytest.raises(MatchedPoolError, match="goals overlap"):
        build_matched_pool(_fixture_rows(),
                           _spec(eval_goal_ids=("g0", "g9")))


def test_disjointness_counts_recorded():
    m = build_matched_pool(_fixture_rows(), _spec())
    dj = m["disjointness"]
    assert dj["family"]["disjoint"] is True and dj["goal"]["disjoint"] is True
    assert "GOAL split" in dj["family"]["note"]
    assert dj["family"]["n_train"] == 1 and dj["goal"]["n_train"] == 3
    assert dj["goal"]["n_eval"] == 1


def test_conflicting_graded_verdicts_raise():
    rows = _fixture_rows()
    # a second 'none' row for pA whose score is on the OTHER side of threshold
    rows.append(_row("bon", "g0", "0:0", "none", None, PROMPTS["pA"],
                     RESP["refusal"], 0.1))
    with pytest.raises(MatchedPoolError, match="conflicting graded verdicts"):
        build_matched_pool(rows, _spec())


def test_per_source_floor_is_flagged():
    m = build_matched_pool(_fixture_rows(),
                           _spec(source_floors={"bon": 5}))  # matched=1 < 5
    assert m["all_sources_meet_floor"] is False
    assert m["per_source_yields"]["bon"]["floor_ok"] is False


# --------------------------------------------------------------------------- #
# the frozen manifest is exactly what two_si_run.aggregate reads back           #
# --------------------------------------------------------------------------- #


def test_two_si_run_loads_our_frozen_manifest(tmp_path):
    import two_si_run

    m = build_matched_pool(_fixture_rows(), _spec())
    path = write_manifest(m, tmp_path / "pool_manifest.json")
    cfg = {"pool_manifest": str(path), "si_ids": list(SI)}
    block = two_si_run._load_pool_regime(cfg)
    assert block["regime"] == "matched_intersection"
    # pA (strict) + pD (relaxed) -> matched size 2 per SI
    assert block["pool_sizes"] == {SI[0]: 2, SI[1]: 2}
    assert block["headline_confounded_by_pool_composition"] is False


# --------------------------------------------------------------------------- #
# the CLI store reader (piece A) + end-to-end manifest freeze                    #
# --------------------------------------------------------------------------- #


def _store_row(fam, gid, vk_sample, vk_trial, si_condition, prompt, resp, score,
               *, group, si_id=None):
    row = {"attack": fam, "goal_id": gid, "si_condition": si_condition,
           "category": "illegal", "category_group": group,
           "attack_prompt": prompt, "response": resp, "score": score,
           "jailbroken": (score is not None and score > 0.5)}
    if vk_sample is not None:
        row["sample_idx"] = vk_sample
        row["trial"] = vk_trial
    if si_id is not None:
        row["si_id"] = si_id
    return row


def test_row_to_graded_maps_conditions_and_skips_compiled():
    thr = 0.5
    # none + safety(prompted) + explicit-constitution map; compiled -> None
    none = bp._row_to_graded(_store_row("bon", "g0", 0, 0, "none",
                                        PROMPTS["pA"], RESP["none_succ"], 0.9,
                                        group="held_in"), si_ids=SI, threshold=thr)
    assert none.condition == "none" and none.si_id is None
    assert none.variant_key == "0:0"
    safety = bp._row_to_graded(_store_row("bon", "g0", 0, 0, "prompted",
                                          PROMPTS["pA"], RESP["refusal"], 0.0,
                                          group="held_in"), si_ids=SI, threshold=thr)
    assert safety.condition == "si" and safety.si_id == "safety_refusal"
    const = bp._row_to_graded(_store_row("bon", "g0", 0, 0, "prompted",
                                         PROMPTS["pA"], RESP["refusal"], 0.0,
                                         group="held_in", si_id="si_constitution"),
                              si_ids=SI, threshold=thr)
    assert const.si_id == "si_constitution"
    compiled = bp._row_to_graded(_store_row("bon", "g0", 0, 0, "compiled",
                                            PROMPTS["pA"], RESP["jb"], 0.9,
                                            group="held_in"), si_ids=SI, threshold=thr)
    assert compiled is None                              # object transcript, skipped
    ungraded = bp._row_to_graded(_store_row("bon", "g0", 0, 0, "none",
                                            PROMPTS["pA"], RESP["none_succ"], None,
                                            group="held_in"), si_ids=SI, threshold=thr)
    assert ungraded is None                              # no score -> no verdict


def _write_store(store_dir: Path):
    """A synthetic labelled store: bon.jsonl with none + safety(prompted) +
    constitution rows for held-in (train) goals and a held-out (eval) goal, plus
    an adaptive pair.jsonl. All benign, invented text."""
    store_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    # held-in train goal g0: pA matched (none-succ, both SIs refuse)
    rows += [_store_row("bon", "g0", 0, 0, "none", PROMPTS["pA"], RESP["none_succ"], 0.9, group="held_in"),
             _store_row("bon", "g0", 0, 0, "prompted", PROMPTS["pA"], RESP["refusal"], 0.0, group="held_in"),
             _store_row("bon", "g0", 0, 0, "prompted", PROMPTS["pA"], RESP["refusal"], 0.0, group="held_in", si_id="si_constitution")]
    # held-out eval goal g9: excluded from the pool (train scope is held_in only)
    rows += [_store_row("bon", "g9", 0, 0, "none", PROMPTS["pB"], RESP["none_succ"], 0.9, group="held_out"),
             _store_row("bon", "g9", 0, 0, "prompted", PROMPTS["pB"], RESP["refusal"], 0.0, group="held_out")]
    # a compiled object transcript that must be ignored
    rows += [_store_row("bon", "g0", 1, 0, "compiled", PROMPTS["pC"], RESP["jb"], 0.9, group="held_in")]
    with (store_dir / "bon.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # adaptive family -> replay_transfer only
    with (store_dir / "pair.jsonl").open("w") as f:
        f.write(json.dumps(_store_row("pair", "g0", None, None, "none",
                                      PROMPTS["adv"], RESP["none_succ"], 0.9,
                                      group="held_in")) + "\n")


def _legacy_two_si(cfg):
    """Rewind the shipped (stage-3) config to the LEGACY two-SI pool world
    these fixtures were written for: two SIs, bon-only, no benign half, no
    hard floor, no scoring subtraction — and the store still serving
    safety_refusal's pos rows (the 2026-08-24 flip is rewound; it has its own
    tests in tests/test_stage3_pool.py / tests/test_grade_store.py). The
    stage-3 path has its own suite (tests/test_stage3_pool.py)."""
    cfg["si_ids"] = ["safety_refusal", "si_constitution"]
    cfg["pool"] = {**cfg["pool"], "static_families": ["bon"],
                   "train_families": ["bon"], "hard_floor_pairs": 0,
                   "benign_source": None, "benign_fraction": None,
                   "benign_sources": [], "fit_floor": {},
                   "pos_runs": {k: v
                                for k, v in (cfg["pool"].get("pos_runs")
                                             or {}).items()
                                if k != "safety_refusal"},
                   # ... and the K=8 default (these fixtures predate raised K;
                   # the rejection-sampling delta workflow has its own tests)
                   "pos_candidates": {},
                   "relaxed_stratum": {"enabled": False},
                   "bidpo_benign": {"enabled": False},
                   "scoring_disjointness": {}}
    # ...which also predates the shipped config's vllm engine declarations
    # (pos/neg_generation_engine, ed01306): leaving them in would silently
    # select the vllm route for every fixture built here. Tests that want
    # the vllm route pass --engine vllm explicitly.
    for k in ("pos_generation_engine", "neg_generation_engine"):
        cfg["pool"].pop(k, None)
    return cfg


def _cli_cfg(tmp_path):
    cfg = _legacy_two_si(
        yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text()))
    store = tmp_path / "labeled"
    _write_store(store)
    cfg["pool"] = {**cfg["pool"], "store_dir": str(store)}
    cfg["pool_manifest"] = str(tmp_path / "pool_manifest.json")
    return cfg


def test_cli_manifest_end_to_end_freezes_and_is_loadable(tmp_path):
    import types

    import two_si_run

    cfg = _cli_cfg(tmp_path)
    args = types.SimpleNamespace(private_root=str(tmp_path / "priv"), out=None)
    bp.cmd_manifest(args, cfg)

    path = Path(cfg["pool_manifest"])
    assert path.exists()
    m = json.loads(path.read_text())
    assert m["regime"] in ("matched_intersection", "native_per_si")
    # g9 (held_out) is NOT in the pool; g0 (held_in) matched pair present
    ids = [(p["family"], p["goal_id"]) for p in m["matched_intersection"]["ordered_pairs"]]
    assert ("bon", "g0") in ids
    assert all(g != "g9" for _f, g in ids)
    # the adaptive pair row landed in replay_transfer, never the pool
    assert m["replay_transfer"]["per_family_counts"].get("pair") == 1
    # no free text leaked into the frozen artifact
    blob = path.read_text()
    for t in {**PROMPTS, **RESP}.values():
        assert t not in blob
    # and two_si_run reads it back through the SAME contract it enforces
    block = two_si_run._load_pool_regime({"pool_manifest": str(path),
                                          "si_ids": cfg["si_ids"]})
    assert block["pool_sizes"] == m["pool_sizes"]


def test_cli_generate_pos_dry_run_reports_gaps_without_model(tmp_path, capsys):
    import types

    cfg = _cli_cfg(tmp_path)
    args = types.SimpleNamespace(private_root=str(tmp_path / "priv"),
                                 dry_run=True, batch_size=2, concurrency=1)
    bp.cmd_generate_pos(args, cfg)          # must NOT import torch / load a model
    out = capsys.readouterr().out
    assert "harmful-base" in out and "need candidates" in out


# --------------------------------------------------------------------------- #
# generate-pos is GENERATION-ONLY (GPU node, no API key); a separate grade-pos  #
# step grades on the grading host. These prove the split — that the generate    #
# path never touches the grader (the crash this guards against), that ungraded  #
# rows are a manifest gap until grade-pos scores them, and that a grader        #
# failure never lands a fabricated refusal 0.                                   #
# --------------------------------------------------------------------------- #


def _pool_only_store(store_dir: Path, prompts):
    """A synthetic labelled store with ONLY none + safety(prompted) rows — no
    constitution rows (those arrive from the constitution-run dir, the way
    generate-pos/grade-pos actually produce them). Every prompt is held_in,
    none-successful and safety-thwarted, so each becomes a matched candidate the
    moment its constitution row is graded-thwarted. ``prompts`` = ``[(sample_idx,
    prompt_text), ...]`` under one held_in goal ``g0``."""
    store_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, text in prompts:
        rows += [_store_row("bon", "g0", idx, 0, "none", text,
                            RESP["none_succ"], 0.9, group="held_in"),
                 _store_row("bon", "g0", idx, 0, "prompted", text,
                            RESP["refusal"], 0.0, group="held_in")]
    # ...plus one HELD-OUT goal. The shipped config holds out at the goal
    # level (pool.family_disjointness: false), so an all-held_in store means
    # an empty eval goal set -- nothing held out in either dimension, which
    # build_matched_pool refuses. The docstring above says "every prompt is
    # held_in"; that is still true of ``prompts``, this is the split beside it.
    _first = prompts[0][1]
    rows += [_store_row("bon", "g_held", 90, 0, "none", _first,
                        RESP["none_succ"], 0.9, group="held_out"),
             _store_row("bon", "g_held", 90, 0, "prompted", _first,
                        RESP["refusal"], 0.0, group="held_out")]
    with (store_dir / "bon.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _pool_only_cfg(tmp_path, prompts):
    cfg = _legacy_two_si(
        yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text()))
    store = tmp_path / "labeled"
    _pool_only_store(store, prompts)
    # min_pool_size 1 so a single matched pair yields the matched regime.
    # pos_candidates pinned at 1: the accel merge raised the shipped
    # config's K to 8 for every SI (a deliberate training-target change),
    # but these tests exercise generate/skip/grader-isolation MECHANICS,
    # which are K-independent — one candidate per prompt keeps the row
    # arithmetic they assert readable.
    cfg["pool"] = {**cfg["pool"], "store_dir": str(store), "min_pool_size": 1,
                   "pos_candidates": {"si_constitution": 1}}
    cfg["pool_manifest"] = str(tmp_path / "pool_manifest.json")
    return cfg


def _const_row(sample_idx, prompt, *, score, graded, goal="g0"):
    """A constitution-run store row. ``score=None, graded=False`` is the
    generated-UNGRADED state generate-pos writes; a float score + ``graded=True``
    is what grade-pos fills in."""
    return {"attack": "bon", "goal_id": goal, "sample_idx": sample_idx,
            "trial": 0, "si_condition": "prompted", "si_id": "si_constitution",
            "category": "illegal", "category_group": "held_in",
            "attack_prompt": prompt, "response": RESP["refusal"],
            "score": score, "graded": graded}


def _write_const_run(priv: Path, cfg, rows):
    cdir = priv / cfg["pool"]["constitution_run"]
    cdir.mkdir(parents=True, exist_ok=True)
    with (cdir / "bon.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return cdir


def _read_const_run(priv: Path, cfg):
    cdir = priv / cfg["pool"]["constitution_run"]
    return [json.loads(l) for l in (cdir / "bon.jsonl").read_text().splitlines()
            if l.strip()]


def _boom_grader_module():
    """A stand-in ``grading_live`` module that raises on ANY attribute access —
    so if the generate path so much as imports the grader, the test fails."""
    import types
    mod = types.ModuleType("sisteer.eval.attacks.grading_live")

    def _raise(name):
        # torch's lazy custom-op registration (pulled in transitively when
        # generate-pos imports resolve_instruction -> sisteer.generate) walks
        # sys.modules and probes dunders like __file__ via inspect; let those
        # miss NORMALLY (AttributeError) so hasattr()/getmodule skip the fake,
        # and boom only on a REAL grader symbol (LiveStrongREJECTGrader, ...).
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise AssertionError(f"generate-pos touched the grader: {name}")

    mod.__getattr__ = _raise                                 # PEP 562 hook
    return mod


def _stub_pos_vllm(monkeypatch):
    """Route generate-pos's vLLM leg through the SHARED fake-engine pattern
    (fake ``vllm`` module + fake AutoTokenizer + fake VllmSampler whose
    ``.llm.generate`` records every request) — the same seam
    test_pool_vllm_generation drives the real ``_generate_pos_vllm`` through,
    so these tests exercise the branch's actual engine path rather than a
    stand-in run() contract it no longer has. Returns the request log; each
    entry is ``(rendered_prompts, sampling_kwargs)`` and every response is
    ``VRESP::<rendered>``."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_pool_vllm_generation import _install_fake_vllm

    log: list = []
    _install_fake_vllm(monkeypatch, log)
    return log


def _pos_args(priv, **kw):
    import types

    base = dict(private_root=str(priv), dry_run=False, batch_size=2, seed=0,
                engine="vllm", max_model_len=16384,
                gpu_memory_utilization=0.9, enforce_eager=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_generate_pos_generates_ungraded_without_touching_grader(tmp_path,
                                                                  monkeypatch):
    cfg = _pool_only_cfg(tmp_path, [(0, PROMPTS["pA"])])
    priv = tmp_path / "priv"

    # route generation to the vLLM engine stub (no GPU, no vLLM import)
    _stub_pos_vllm(monkeypatch)
    # the grader (and its ~/.config Low-Prio key) must be UNREACHABLE here —
    # importing it is exactly the cluster crash this guards against.
    monkeypatch.setitem(sys.modules, "sisteer.eval.attacks.grading_live",
                        _boom_grader_module())

    bp.cmd_generate_pos(_pos_args(priv), cfg)  # must COMPLETE with no grader

    rows = _read_const_run(priv, cfg)
    assert len(rows) == 1
    r = rows[0]
    assert r["si_id"] == "si_constitution" and r["goal_id"] == "g0"
    assert r["score"] is None and r["graded"] is False      # explicitly ungraded
    assert r["response"].startswith("VRESP::")              # our fake engine wrote it
    assert r["attack_prompt"] == PROMPTS["pA"]              # prompt identity carried


def test_generate_pos_skips_already_generated_ungraded_rows(tmp_path,
                                                            monkeypatch):
    cfg = _pool_only_cfg(tmp_path, [(0, PROMPTS["pA"]), (1, PROMPTS["pB"])])
    priv = tmp_path / "priv"
    # pA already generated but still UNGRADED -> must NOT be regenerated
    _write_const_run(priv, cfg, [_const_row(0, PROMPTS["pA"], score=None,
                                            graded=False)])

    log = _stub_pos_vllm(monkeypatch)
    monkeypatch.setitem(sys.modules, "sisteer.eval.attacks.grading_live",
                        _boom_grader_module())

    bp.cmd_generate_pos(_pos_args(priv, batch_size=8), cfg)

    # only pB was regenerated: exactly one prompt reached the engine, and it
    # renders to <system-prefix>|<user> under the fake tokenizer
    rendered = [p for prompts, _params in log for p in prompts]
    assert len(rendered) == 1 and rendered[0].endswith("|" + PROMPTS["pB"])
    rows = _read_const_run(priv, cfg)
    assert len(rows) == 2                                    # pA (kept) + pB (new)
    assert {r["attack_prompt"] for r in rows} == {PROMPTS["pA"], PROMPTS["pB"]}


def test_grade_pos_grades_ungraded_then_manifest_counts_it(tmp_path, monkeypatch):
    import types

    cfg = _pool_only_cfg(tmp_path, [(0, PROMPTS["pA"])])
    priv = tmp_path / "priv"
    _write_const_run(priv, cfg, [_const_row(0, PROMPTS["pA"], score=None,
                                            graded=False)])

    # BEFORE grading: the manifest treats the generated-ungraded row as a GAP
    margs = types.SimpleNamespace(private_root=str(priv), out=None)
    bp.cmd_manifest(margs, cfg)
    m0 = json.loads(Path(cfg["pool_manifest"]).read_text())
    assert m0["coverage"][SI[1]]["n_missing"] == 1          # constitution missing
    assert m0["matched_intersection"]["n"] == 0

    # a fake grader: a validly-parsed refusal scores 0.0 (thwarted). Never raises.
    monkeypatch.setattr(bp, "_build_grader",
                        lambda a, c, d: (lambda items: {it: 0.0 for it in items}))
    gargs = types.SimpleNamespace(private_root=str(priv), dry_run=False,
                                  concurrency=4)
    bp.cmd_grade_pos(gargs, cfg)

    rows = _read_const_run(priv, cfg)
    assert len(rows) == 1                                    # rewritten, not dup'd
    assert rows[0]["score"] == 0.0 and rows[0]["graded"] is True

    # AFTER grading: the manifest counts it as a usable constitution pos
    bp.cmd_manifest(margs, cfg)
    m1 = json.loads(Path(cfg["pool_manifest"]).read_text())
    assert m1["coverage"][SI[1]]["n_missing"] == 0
    assert m1["matched_intersection"]["n"] == 1


def test_grade_pos_never_writes_zero_on_grader_failure(tmp_path, monkeypatch):
    import types

    cfg = _pool_only_cfg(tmp_path, [(0, PROMPTS["pA"])])
    priv = tmp_path / "priv"
    _write_const_run(priv, cfg, [_const_row(0, PROMPTS["pA"], score=None,
                                            graded=False)])

    # mirror LiveStrongREJECTGrader raising when an item stays UNSCORED after
    # retries: grade-pos must propagate and write NOTHING — never fabricate a 0.
    def _boom(a, c, d):
        def grade(items):
            raise RuntimeError("judge item(s) still UNSCORED (simulated)")
        return grade

    monkeypatch.setattr(bp, "_build_grader", _boom)
    gargs = types.SimpleNamespace(private_root=str(priv), dry_run=False,
                                  concurrency=4)
    with pytest.raises(RuntimeError, match="UNSCORED"):
        bp.cmd_grade_pos(gargs, cfg)

    rows = _read_const_run(priv, cfg)
    assert len(rows) == 1
    assert rows[0]["score"] is None and rows[0]["graded"] is False   # untouched


def test_grade_pos_is_resumable_and_skips_already_graded(tmp_path, monkeypatch):
    import types

    cfg = _pool_only_cfg(tmp_path, [(0, PROMPTS["pA"]), (1, PROMPTS["pB"])])
    priv = tmp_path / "priv"
    # pA already graded, pB still ungraded — grade-pos must grade ONLY pB
    _write_const_run(priv, cfg, [
        _const_row(0, PROMPTS["pA"], score=0.0, graded=True),
        _const_row(1, PROMPTS["pB"], score=None, graded=False)])

    graded_items = {}

    def _fake(a, c, d):
        def grade(items):
            graded_items["items"] = list(items)
            return {it: 0.0 for it in items}
        return grade

    monkeypatch.setattr(bp, "_build_grader", _fake)
    gargs = types.SimpleNamespace(private_root=str(priv), dry_run=False,
                                  concurrency=4)
    bp.cmd_grade_pos(gargs, cfg)

    # only pB was sent to the grader (pA already had a score)
    assert [it[2] for it in graded_items["items"]] == [PROMPTS["pB"]]
    rows = {r["attack_prompt"]: r for r in _read_const_run(priv, cfg)}
    assert rows[PROMPTS["pA"]]["graded"] is True
    assert rows[PROMPTS["pB"]]["graded"] is True and rows[PROMPTS["pB"]]["score"] == 0.0


def test_grade_pos_default_concurrency_is_wide():
    """The launcher default is 200 (the repo's live-grader norm) — the old
    4 quietly treated the coordination post as a gate, which the standing
    rule forbids (print-and-proceed)."""
    args = bp.build_parser().parse_args(["grade-pos"])
    assert args.concurrency == 200


def test_grade_pos_prints_coordination_line_and_proceeds(tmp_path,
                                                         monkeypatch, capsys):
    """The courtesy line is PRINTED and grading runs in the SAME call —
    the post never gates the launch."""
    import types

    cfg = _pool_only_cfg(tmp_path, [(0, PROMPTS["pA"])])
    priv = tmp_path / "priv"
    _write_const_run(priv, cfg, [_const_row(0, PROMPTS["pA"], score=None,
                                            graded=False)])
    monkeypatch.setattr(bp, "_build_grader",
                        lambda a, c, d: (lambda items: {it: 0.0
                                                        for it in items}))
    gargs = types.SimpleNamespace(private_root=str(priv), dry_run=False,
                                  concurrency=200)
    monkeypatch.setenv("SISTEER_COORDINATION_CHANNEL", "#example-api-coordination")
    bp.cmd_grade_pos(gargs, cfg)

    out = capsys.readouterr().out
    assert "[coordination] ready to post to #example-api-coordination" in out and "lowprio" in out
    assert "proceeds" in out                        # print-and-proceed, stated
    (row,) = _read_const_run(priv, cfg)
    assert row["graded"] is True                    # grading ran, ungated


def test_grade_pos_without_a_coordination_channel_prints_no_line_and_grades(tmp_path, monkeypatch, capsys):
    """The channel is site config: a site that names none gets no coordination line, and grading still runs."""
    import types

    cfg = _pool_only_cfg(tmp_path, [(0, PROMPTS["pA"])])
    priv = tmp_path / "priv"
    _write_const_run(priv, cfg, [_const_row(0, PROMPTS["pA"], score=None,
                                            graded=False)])
    monkeypatch.setattr(bp, "_build_grader",
                        lambda a, c, d: (lambda items: {it: 0.0
                                                        for it in items}))
    monkeypatch.delenv("SISTEER_COORDINATION_CHANNEL", raising=False)
    bp.cmd_grade_pos(types.SimpleNamespace(private_root=str(priv), dry_run=False, concurrency=200), cfg)
    assert "[coordination]" not in capsys.readouterr().out
    (row,) = _read_const_run(priv, cfg)
    assert row["graded"] is True


def test_grade_pos_dry_run_prints_no_coordination_line(tmp_path, capsys):
    """--dry-run makes no API calls, so there is nothing to coordinate."""
    import types

    cfg = _pool_only_cfg(tmp_path, [(0, PROMPTS["pA"])])
    priv = tmp_path / "priv"
    _write_const_run(priv, cfg, [_const_row(0, PROMPTS["pA"], score=None,
                                            graded=False)])
    gargs = types.SimpleNamespace(private_root=str(priv), dry_run=True,
                                  concurrency=200)
    bp.cmd_grade_pos(gargs, cfg)
    assert "[coordination]" not in capsys.readouterr().out


def test_manifest_three_state_missing_ungraded_graded(tmp_path):
    import types

    # three train prompts, all none-success + safety-thwarted
    prompts = [(0, PROMPTS["pA"]), (1, PROMPTS["pB"]), (2, PROMPTS["pC"])]
    cfg = _pool_only_cfg(tmp_path, prompts)
    priv = tmp_path / "priv"
    # constitution rows in three distinct states:
    #   pA (0): GRADED-thwarted    -> covered (a usable pos)
    #   pB (1): generated-UNGRADED -> STILL a gap (not usable until graded)
    #   pC (2): NO row             -> missing
    _write_const_run(priv, cfg, [
        _const_row(0, PROMPTS["pA"], score=0.0, graded=True),
        _const_row(1, PROMPTS["pB"], score=None, graded=False)])

    margs = types.SimpleNamespace(private_root=str(priv), out=None)
    bp.cmd_manifest(margs, cfg)
    m = json.loads(Path(cfg["pool_manifest"]).read_text())

    cov = m["coverage"][SI[1]]
    assert cov["n_none_success"] == 3
    # ONLY the graded row counts as present; the ungraded one is still missing
    assert cov["n_si_rows_present"] == 1
    assert cov["n_missing"] == 2
    assert set(cov["missing_prompt_ids"]) == {"bon/g0/1:0", "bon/g0/2:0"}
    # the matched pool contains ONLY the graded-thwarted prompt
    ids = [(p["goal_id"], p["variant_key"])
           for p in m["matched_intersection"]["ordered_pairs"]]
    assert ids == [("g0", "0:0")]


# --------------------------------------------------------------------------- #
# per-family caps                                                               #
# --------------------------------------------------------------------------- #


def _matched_rows(fam, n):
    """`n` distinct prompts for `fam` that all MATCH (none complied, both SIs
    thwart) — a controllable population for the cap tests."""
    r = []
    for i in range(n):
        vk = f"{i}:0"
        p = f"SYNTHETIC cap prompt {i} 7777{i:04d}"
        r += [_row(fam, "g0", vk, "none", None, p, RESP["none_succ"], 0.9),
              _row(fam, "g0", vk, "si", SI[0], p, RESP["refusal"], 0.0),
              _row(fam, "g0", vk, "si", SI[1], p, RESP["refusal"], 0.0)]
    return r


def test_family_cap_deterministic_and_recorded():
    from sisteer.eval.matched_pool import _det_take
    rows = _matched_rows("bon", 5)
    # cap to 2
    m1 = build_matched_pool(rows, _spec(family_caps={"bon": 2}))
    assert m1["matched_intersection"]["n"] == 2
    fc = m1["family_caps"]
    assert fc["configured"] == {"bon": 2}
    assert fc["applied"]["bon"] == {"cap": 2, "n_before": 5, "n_after": 2,
                                    "n_dropped": 3}
    assert fc["n_dropped_total"] == 3
    ids1 = {(p["family"], p["goal_id"], p["variant_key"])
            for p in m1["matched_intersection"]["ordered_pairs"]}
    # determinism: a re-run keeps EXACTLY the same pairs
    m2 = build_matched_pool(rows, _spec(family_caps={"bon": 2}))
    ids2 = {(p["family"], p["goal_id"], p["variant_key"])
            for p in m2["matched_intersection"]["ordered_pairs"]}
    assert ids1 == ids2
    # and they are the sha-ordered _det_take subset (same convention as balance)
    all_pids = {("bon", "g0", f"{i}:0") for i in range(5)}
    assert ids1 == _det_take(all_pids, 2)


def test_family_cap_noop_when_absent_or_slack():
    rows = _matched_rows("bon", 5)
    # default (no family_caps): all 5 matched, block present but empty
    m_default = build_matched_pool(rows, _spec())
    assert m_default["matched_intersection"]["n"] == 5
    assert m_default["family_caps"]["configured"] == {}
    assert m_default["family_caps"]["n_dropped_total"] == 0
    # explicit empty dict is identical
    m_empty = build_matched_pool(rows, _spec(family_caps={}))
    assert m_empty["matched_intersection"]["n"] == 5
    # a cap >= the present count drops nothing (no-op)
    m_high = build_matched_pool(rows, _spec(family_caps={"bon": 99}))
    assert m_high["matched_intersection"]["n"] == 5
    assert m_high["family_caps"]["applied"]["bon"]["n_dropped"] == 0
