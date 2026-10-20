"""bidpo's OWN training pool.

CPU-only, offline, zero API calls, NO real store data: every prompt/response
here is an INVENTED synthetic string (privacy rule — placeholder text only).

bidpo used to train on harmful pairs alone — every preference it saw said
"prefer refusing", pushing exactly the over-refusal the campaign measures.
Section 13 gives bidpo its own per-SI pool: HARMFUL = that SI's OWN thwart
set (not the three-way joint, not the 50/50 balance subsample) with the
accepted anti-safety compliance as the rejected side; BENIGN, polarity
REVERSED = benign prompts that SI refused while the undefended model
VERIFIABLY complied (chosen = the undefended compliance, rejected = the SI's
refusal); mix = benign capped at 1x harmful by default (min rule), harmful
never subsampled. What this suite proves:

- THE polarity assertion (the single most important one): an emitted
  benign-reversed pair reaches ``pairs_to_pref_examples`` with chosen == the
  compliance and rejected == the refusal;
- the harmful side is the PER-SI thwart set and escapes the matched
  intersection and its balance subsample;
- only bidpo ever sees its own-pool pairs; every other arm's pair list is
  the SAME OBJECT with the feature on or off, and the matched-pool pairs
  files are byte-identical with the feature toggled;
- the block qualifies a benign prompt iff the SI's REDUCED label is in the
  refusal-label setting (I-only default; I+P opt-in; P excluded from BOTH
  sides under the default) AND the undefended row is a VERIFIED ``C`` —
  ungraded undefended rows RAISE (never assumed C), undefended refusals are
  dropped AND counted;
- per-SI UNEQUAL counts AND the projected harmful:benign ratios are recorded
  everywhere (manifest, emit sidecar, each object's train_budget_meta.json,
  the fit report) — the section-13 confound record;
- the section-5 >1000 floor is superseded: a thin side passes and is
  recorded; an EMPTY side (benign or harmful) still raises;
- grade-store labels benign ``none`` rows when (and only when) the feature
  is enabled;
- loud failures: one-sided supply, empty sides, mislabelled (wrong-polarity)
  files, missing files / sidecar / sha mismatch at train time, store
  diverged from the frozen block.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_two_si_pool as bp  # noqa: E402
import train_two_si_objects as tt  # noqa: E402
from sisteer.data.pairs import Pair, pairs_to_pref_examples  # noqa: E402
from sisteer.eval.matched_pool import (  # noqa: E402
    GradedRow,
    MatchedPoolError,
    PoolFloorError,
    PoolSpec,
    build_matched_pool,
    dumps_manifest,
    manifest_hash,
    text_sha,
)

REPO = Path(__file__).resolve().parents[1]
SI3 = ("safety_refusal", "si_constitution", "si_irrelevant")
ANTI = "anti_safety_refusal"

# A tokenizer stand-in with no chat template: build_chat_prompt falls back to
# plain concatenation, so pairs_to_pref_examples runs with no model download.
FAKE_TOK = types.SimpleNamespace(chat_template=None)


def _gr(fam, gid, vk, cond, si, prompt, resp, *, score=None, label=None,
        cidx=0):
    return GradedRow(family=fam, goal_id=gid, variant_key=vk, condition=cond,
                     si_id=si, prompt=prompt, response=resp, score=score,
                     label=label, candidate_idx=cidx)


def _hprompt(i):
    return f"SYNTHETIC attack prompt number {i} harmful stand-in text"


def _hrefusal(si, i):
    return f"SYNTHETIC refusal from {si} for prompt {i}"


def _anti_comply(i):
    return f"SYNTHETIC anti-safety compliance for prompt {i} qqqq"


def _bprompt(i):
    return f"SYNTHETIC benign question {i} about a safe topic"


def _bneg(i):
    return f"SYNTHETIC undefended benign answer {i} fully helpful"


def _brefusal(si, i):
    return f"SYNTHETIC over-refusal from {si} on benign question {i}"


def _bcomply(si, i):
    return f"SYNTHETIC compliant answer from {si} to question {i}"


def _harmful(i, *, thwarted=SI3, none_score=0.9, anti_score=0.9):
    """One harmful prompt: a none row, one SI row per SI (thwarted or
    jailbroken), and one anti-safety candidate (``anti_score`` None = no anti
    row at all)."""
    gid = f"g{i}"
    rows = [_gr("bon", gid, "0:0", "none", None, _hprompt(i),
                f"SYNTHETIC undefended completion {i}", score=none_score)]
    for si in SI3:
        thw = si in thwarted
        resp = (_hrefusal(si, i) if thw
                else f"SYNTHETIC jailbroken reply {si} {i}")
        rows.append(_gr("bon", gid, "0:0", "si", si, _hprompt(i), resp,
                        score=0.0 if thw else 0.9))
    if anti_score is not None:
        rows.append(_gr("bon", gid, "0:0", "si", ANTI, _hprompt(i),
                        _anti_comply(i), score=anti_score))
    return rows


def _benign(i, *, si_labels=None, none_label="C"):
    """One benign prompt: an undefended none row (label ``none_label``; None =
    ungraded) + one labelled SI row per SI (default all-C).

    ``si_labels[si]`` may be a single label or a LIST of labels (one per
    rejection-sampling candidate, candidate_idx order)."""
    si_labels = si_labels or {}
    gid = f"phtest_{i}"
    rows = [_gr("phtest", gid, "0", "none", None, _bprompt(i), _bneg(i),
                label=none_label)]
    for si in SI3:
        labels = si_labels.get(si, "C")
        if isinstance(labels, str):
            labels = [labels]
        for c, lab in enumerate(labels):
            resp = (_bcomply(si, i) if lab == "C"
                    else _brefusal(si, i) + f" candidate {c} label {lab}")
            rows.append(_gr("phtest", gid, "0", "si", si, _bprompt(i), resp,
                            label=lab, cidx=c))
    return rows


def _spec(**over):
    base = dict(
        si_ids=SI3,
        train_families=("bon",),
        eval_families=("pair",),
        train_goal_ids=tuple(f"g{i}" for i in range(16)),
        eval_goal_ids=("g_eval",),
        static_families=frozenset({"bon"}),
        threshold=0.5, min_pool_size=1,
        benign_source="phtest", benign_fraction=0.5,
        anti_si_id=ANTI,
        bidpo_benign_enabled=True, bidpo_benign_floor=0)
    base.update(over)
    return PoolSpec(**base)


def _world():
    """Harmful: g0 (thwarted by all, strict), g1 (thwarted by all, relaxed —
    the undefended model already refused), g2 + g4 (thwarted by SAFETY ONLY —
    in safety's bidpo pool but NEVER in the joint intersection), g3 (thwarted
    by all, anti candidate REJECTED -> missing its anti_neg). Benign:

      b100, b101: all-comply (the intersection's benign half)
      b102: safety refused (I),           undefended C -> safety qualifies
      b103: safety I + constitution I,    undefended C -> both qualify
      b104: safety refused (I),           undefended I -> DROPPED + counted
      b105: constitution partial (P),     undefended C -> excluded under
            I-only; qualifies for constitution under I+P
      b106: irrelevant candidates [I, C]  -> any-C = complied, never qualifies
      b107: irrelevant candidates [P, I]  -> reduced label = P (smallest
            candidate), excluded under I-only
      b109: irrelevant refused (I),       undefended C -> irrelevant qualifies
    """
    rows = _harmful(0) + _harmful(1, none_score=0.1)
    rows += _harmful(2, thwarted=("safety_refusal",))
    rows += _harmful(3, anti_score=0.2)
    rows += _harmful(4, thwarted=("safety_refusal",))
    rows += _benign(100) + _benign(101)
    rows += _benign(102, si_labels={"safety_refusal": "I"})
    rows += _benign(103, si_labels={"safety_refusal": "I",
                                    "si_constitution": "I"})
    rows += _benign(104, si_labels={"safety_refusal": "I"}, none_label="I")
    rows += _benign(105, si_labels={"si_constitution": "P"})
    rows += _benign(106, si_labels={"si_irrelevant": ["I", "C"]})
    rows += _benign(107, si_labels={"si_irrelevant": ["P", "I"]})
    rows += _benign(109, si_labels={"si_irrelevant": "I"})
    return rows


# --------------------------------------------------------------------------- #
# 1. the manifest block: per-SI thwart harmful side + reversed benign side      #
# --------------------------------------------------------------------------- #


def test_harmful_side_is_the_per_si_thwart_not_the_joint():
    m = build_matched_pool(_world(), _spec())
    per = m["bidpo_benign"]["per_si"]
    # safety thwarted g0..g4; g3's anti candidate failed acceptance.
    assert per["safety_refusal"]["harmful"]["n_thwarted"] == 5
    assert per["safety_refusal"]["harmful"]["n_with_anti_neg"] == 4
    assert per["safety_refusal"]["harmful"]["n_missing_anti_neg"] == 1
    # constitution/irrelevant never thwarted g2 (safety-only) — per-SI sets.
    assert per["si_constitution"]["harmful"]["n_with_anti_neg"] == 2
    assert per["si_irrelevant"]["harmful"]["n_with_anti_neg"] == 2
    # g2 escapes the JOINT intersection but sits in safety's bidpo pool:
    joint_gids = {e["goal_id"]
                  for e in m["matched_intersection"]["ordered_pairs"]}
    assert "g2" not in joint_gids
    safety_gids = {e["goal_id"]
                   for e in per["safety_refusal"]["harmful"]["ordered_pairs"]}
    assert "g2" in safety_gids
    # ...and safety's bidpo harmful side (4) EXCEEDS the balanced pool's
    # harmful half (the 50/50 balance caps it at the benign count, 3):
    assert (per["safety_refusal"]["harmful"]["n_with_anti_neg"]
            > m["composition"]["pre_balance"]["harmful_taken"])


def test_harmful_polarity_and_strict_relaxed_labels():
    m = build_matched_pool(_world(), _spec())
    entries = {e["goal_id"]: e for e in m["bidpo_benign"]["per_si"][
        "safety_refusal"]["harmful"]["ordered_pairs"]}
    # chosen = THIS SI's refusal; rejected = the accepted anti compliance.
    assert entries["g0"]["chosen_sha"] == text_sha(
        _hrefusal("safety_refusal", 0))
    assert entries["g0"]["rejected_sha"] == text_sha(_anti_comply(0))
    assert entries["g0"]["requirement_a"] == "strict"
    assert entries["g1"]["requirement_a"] == "relaxed"  # undefended refused


def test_reversed_benign_polarity_and_per_si_unequal_counts():
    m = build_matched_pool(_world(), _spec())
    blk = m["bidpo_benign"]
    assert blk["refusal_labels"] == ["I"]
    assert blk["undefended_required_label"] == "C"
    per = blk["per_si"]
    # per-SI, UNEQUAL — each SI keeps its own refusals.
    assert per["safety_refusal"]["benign"]["n_qualified"] == 2    # b102, b103
    assert per["si_constitution"]["benign"]["n_qualified"] == 1   # b103
    assert per["si_irrelevant"]["benign"]["n_qualified"] == 1     # b109
    # THE polarity: chosen = the undefended compliance, rejected = the refusal.
    e102 = next(e for e in per["safety_refusal"]["benign"]["ordered_pairs"]
                if e["goal_id"] == "phtest_102")
    assert e102["chosen_sha"] == text_sha(_bneg(102))
    assert e102["rejected_sha"] == text_sha(
        _brefusal("safety_refusal", 102) + " candidate 0 label I")
    assert e102["rejected_label"] == "I"
    assert e102["prompt_sha"] == text_sha(_bprompt(102))


def test_projected_mix_records_counts_and_ratio_per_si():
    """Section 13's confound record: the three objects train at DIFFERENT
    harmful:benign ratios; the block freezes each SI's counts + ratio."""
    m = build_matched_pool(_world(), _spec())
    per = m["bidpo_benign"]["per_si"]
    assert per["safety_refusal"]["projected_mix_at_default_1to1"] == {
        "n_harmful": 4, "n_benign": 2, "ratio_harmful_to_benign": 2.0}
    assert per["si_constitution"]["projected_mix_at_default_1to1"] == {
        "n_harmful": 2, "n_benign": 1, "ratio_harmful_to_benign": 2.0}
    assert per["si_irrelevant"]["projected_mix_at_default_1to1"] == {
        "n_harmful": 2, "n_benign": 1, "ratio_harmful_to_benign": 2.0}


def test_undefended_refusal_is_dropped_and_counted():
    m = build_matched_pool(_world(), _spec())
    safety = m["bidpo_benign"]["per_si"]["safety_refusal"]["benign"]
    # b104: safety refused but the undefended model ALSO refused — no
    # compliance to prefer, dropped and counted per label, never emitted.
    assert safety["n_dropped_undefended_not_complied"] == {"P": 0, "I": 1}
    assert safety["n_si_refused"] == 3          # b102, b103, b104
    assert all(e["goal_id"] != "phtest_104"
               for e in safety["ordered_pairs"])


def test_ungraded_undefended_row_raises_never_assumed_C():
    rows = _harmful(0) + _harmful(1) + _benign(100) + _benign(101)
    rows += _benign(102, si_labels={"safety_refusal": "I"}, none_label=None)
    rows += _benign(109, si_labels={"si_irrelevant": "I"})
    rows += _benign(110, si_labels={"si_constitution": "I"})
    with pytest.raises(MatchedPoolError, match="no XSTest label"):
        build_matched_pool(rows, _spec())


def test_refusal_label_setting_I_only_vs_I_plus_P():
    rows = _world()
    m_i = build_matched_pool(rows, _spec())
    per_i = m_i["bidpo_benign"]["per_si"]
    # I-only (default): b105's P and b107's reduced-P are excluded — P is
    # excluded from BOTH sides, so each side is unambiguous.
    assert per_i["si_constitution"]["benign"]["n_qualified"] == 1
    assert per_i["si_irrelevant"]["benign"]["n_qualified"] == 1
    m_ip = build_matched_pool(
        rows, _spec(bidpo_benign_refusal_labels=("I", "P")))
    per_ip = m_ip["bidpo_benign"]["per_si"]
    # I+P: b105 (constitution P) and b107 (irrelevant reduced P) now qualify.
    assert per_ip["si_constitution"]["benign"]["n_qualified"] == 2
    assert per_ip["si_irrelevant"]["benign"]["n_qualified"] == 2
    assert m_ip["bidpo_benign"]["refusal_labels"] == ["I", "P"]


def test_multi_candidate_reduction_matches_the_intersection_rule():
    m = build_matched_pool(_world(), _spec())
    irr = m["bidpo_benign"]["per_si"]["si_irrelevant"]["benign"]
    gids = {e["goal_id"] for e in irr["ordered_pairs"]}
    # b106: any candidate C -> the SI complied, never a refusal.
    assert "phtest_106" not in gids
    # b107: no C anywhere but the REDUCED label is the smallest candidate's
    # (P), so I-only excludes it — the same reduction the intersection uses.
    assert "phtest_107" not in gids
    assert gids == {"phtest_109"}


def test_thin_benign_side_passes_and_is_recorded_zero_raises():
    """Section 13 supersedes the >1000 floor: a THIN side (si_constitution =
    1 here) freezes fine with its count recorded; an EMPTY side raises."""
    m = build_matched_pool(_world(), _spec())     # thin sides pass
    assert m["bidpo_benign"]["floor"] == 0
    rows = _harmful(0) + _harmful(1) + _benign(100) + _benign(101)
    with pytest.raises(PoolFloorError, match="EMPTY or below"):
        build_matched_pool(rows, _spec())         # zero benign anywhere
    # a configured floor still binds when explicitly requested
    with pytest.raises(PoolFloorError, match="si_constitution=1"):
        build_matched_pool(_world(), _spec(bidpo_benign_floor=1))


def test_empty_harmful_side_raises():
    rows = (_harmful(0, anti_score=0.2) + _harmful(1, anti_score=None)
            + _benign(100) + _benign(101)
            + _benign(102, si_labels={"safety_refusal": "I",
                                      "si_constitution": "I",
                                      "si_irrelevant": "I"}))
    with pytest.raises(PoolFloorError, match="harmful side is EMPTY"):
        build_matched_pool(rows, _spec())


def test_everything_else_byte_identical_with_feature_on_vs_off():
    rows = _world()
    m_on = build_matched_pool(rows, _spec())
    m_off = build_matched_pool(rows, _spec(bidpo_benign_enabled=False))
    assert "bidpo_benign" in m_on and "bidpo_benign" not in m_off
    for key in m_off:
        if key == "manifest_sha256":
            continue
        assert m_on[key] == m_off[key], key


def test_manifest_carries_no_text_and_hash_covers_block():
    rows = _world()
    m = build_matched_pool(rows, _spec())
    blob = dumps_manifest(m)
    for needle in (_bneg(102), _bprompt(102),
                   _brefusal("safety_refusal", 102),
                   _bcomply("si_constitution", 102),
                   _anti_comply(0), _hrefusal("safety_refusal", 2)):
        assert needle not in blob
    # perturb the frozen rejected side -> a different manifest hash.
    rows2 = [r if not (r.si_id == "safety_refusal"
                       and r.goal_id == "phtest_102" and r.condition == "si")
             else _gr(r.family, r.goal_id, r.variant_key, "si", r.si_id,
                      r.prompt, "SYNTHETIC EDITED refusal text zzz",
                      label=r.label, cidx=r.candidate_idx)
             for r in rows]
    assert manifest_hash(m) != manifest_hash(
        build_matched_pool(rows2, _spec()))


def test_poolspec_validation():
    with pytest.raises(MatchedPoolError, match="benign stratum"):
        _spec(benign_source=None, benign_fraction=None)
    with pytest.raises(MatchedPoolError, match="anti_si_id"):
        _spec(anti_si_id=None)
    with pytest.raises(MatchedPoolError, match="non-empty subset"):
        _spec(bidpo_benign_refusal_labels=("C",))
    with pytest.raises(MatchedPoolError, match="non-empty subset"):
        _spec(bidpo_benign_refusal_labels=())
    with pytest.raises(MatchedPoolError, match=">= 0"):
        _spec(bidpo_benign_floor=-1)


# --------------------------------------------------------------------------- #
# 2. pairs_for_arm: the own pool reaches bidpo and ONLY bidpo                   #
# --------------------------------------------------------------------------- #

B_COMPLY = "SYNTHETIC undefended benign compliance text {i} aaaa"
B_REFUSE = "SYNTHETIC SI benign over-refusal text {i} bbbb"
H_REFUSE = "SYNTHETIC SI harmful refusal text {i} cccc"
H_ANTI = "SYNTHETIC anti compliance text {i} dddd"


def _matched_pair(i, req="strict"):
    return Pair(prompt=f"SYNTHETIC matched prompt {i} 55555555",
                pos=f"SYNTHETIC SI refusal styled {i} 66666666",
                neg=f"SYNTHETIC undefended completion {i} 77777777",
                instruction_id="safety_refusal", requirement_a=req)


def _hpair(i, req="strict"):
    return Pair(prompt=f"SYNTHETIC bidpo harmful prompt {i} 88880000",
                pos=H_REFUSE.format(i=i), neg=H_ANTI.format(i=i),
                instruction_id="safety_refusal", requirement_a=req)


def _bpair(i):
    return Pair(prompt=f"SYNTHETIC benign prompt {i} 88888888",
                pos=B_COMPLY.format(i=i), neg=B_REFUSE.format(i=i),
                instruction_id="safety_refusal",
                requirement_a=bp.BIDPO_BENIGN_REQ_A)


def test_polarity_through_pairs_to_pref_examples():
    """THE assertion: pairs reach the (untouched, measurement-frozen)
    pairs_to_pref_examples with the right polarity on BOTH sides — benign
    chosen == the undefended COMPLIANCE and rejected == the SI's REFUSAL;
    harmful chosen == the SI's refusal and rejected == the anti compliance."""
    got = tt.pairs_for_arm("bidpo", [],
                           bidpo_harmful_pairs=[_hpair(0), _hpair(1)],
                           bidpo_benign_pairs=[_bpair(0), _bpair(1)])
    benign = [p for p in got if p.requirement_a == bp.BIDPO_BENIGN_REQ_A]
    assert len(benign) == 2
    prefs = pairs_to_pref_examples(FAKE_TOK, benign)
    assert {e.chosen for e in prefs} == {B_COMPLY.format(i=0),
                                         B_COMPLY.format(i=1)}
    assert {e.rejected for e in prefs} == {B_REFUSE.format(i=0),
                                           B_REFUSE.format(i=1)}
    harm = [p for p in got if p.requirement_a != bp.BIDPO_BENIGN_REQ_A]
    hp = pairs_to_pref_examples(FAKE_TOK, harm)
    assert {e.chosen for e in hp} == {H_REFUSE.format(i=0),
                                      H_REFUSE.format(i=1)}
    assert {e.rejected for e in hp} == {H_ANTI.format(i=0),
                                        H_ANTI.format(i=1)}


def test_only_dpo_pref_arms_see_the_own_pool():
    pool = [_matched_pair(0), _matched_pair(1)]
    from sisteer.methods import ARMS

    for arm in ARMS:
        if arm in tt.DPO_PREF_ARMS:
            continue
        assert tt.pairs_for_arm(arm, pool,
                                bidpo_harmful_pairs=[_hpair(0)],
                                bidpo_benign_pairs=[_bpair(0)]) is pool
        assert tt.pairs_for_arm(arm, pool) is pool


def test_every_dpo_pref_arm_shares_bidpos_filtered_pairs():
    """The routing gates on SET membership, never the "bidpo" string: a
    sibling DPO arm (bidpo_lora) must receive EXACTLY the same filtered
    pairs, or it silently trains a preference objective on the full matched
    pool's refusal-vs-refusal pairs — the degenerate signal the filter
    exists to prevent."""
    assert "bidpo_lora" in tt.DPO_PREF_ARMS
    harmful = [_hpair(0), _hpair(1)]
    benign = [_bpair(0), _bpair(1), _bpair(2)]
    ref = tt.pairs_for_arm("bidpo", [], bidpo_harmful_pairs=harmful,
                           bidpo_benign_pairs=benign)
    for arm in sorted(tt.DPO_PREF_ARMS):
        assert tt.pairs_for_arm(arm, [], bidpo_harmful_pairs=harmful,
                                bidpo_benign_pairs=benign) == ref
    # the legacy (feature-off) routes are shared too...
    pool = [_matched_pair(0), _matched_pair(1, req="relaxed"),
            _matched_pair(2, req="benign")]
    for arm in sorted(tt.DPO_PREF_ARMS):
        assert tt.pairs_for_arm(arm, pool) == tt.pairs_for_arm("bidpo", pool)
    # ...raises included: no DPO arm can silently take the full pool.
    unlabelled = [Pair(prompt="SYNTHETIC u 1", pos="SYNTHETIC p",
                       neg="SYNTHETIC n", instruction_id="safety_refusal")]
    for arm in sorted(tt.DPO_PREF_ARMS):
        with pytest.raises(ValueError, match="requirement_a"):
            tt.pairs_for_arm(arm, unlabelled)


def test_own_pool_replaces_the_matched_pool_for_bidpo():
    """With the own pool supplied, the matched pairs are IGNORED for bidpo —
    its harmful side is the per-SI file, not the strict/anti subset."""
    pool = [_matched_pair(0), _matched_pair(1, req="benign")]
    got = tt.pairs_for_arm("bidpo", pool,
                           bidpo_harmful_pairs=[_hpair(0)],
                           bidpo_benign_pairs=[_bpair(0)])
    assert all("matched prompt" not in p.prompt for p in got)
    assert len(got) == 2


def test_cap_subsamples_the_benign_side_only_and_deterministically():
    harmful = [_hpair(i) for i in range(2)]
    benign = [_bpair(i) for i in range(10)]
    got = tt.pairs_for_arm("bidpo", [], bidpo_harmful_pairs=harmful,
                           bidpo_benign_pairs=benign, bidpo_benign_cap=1.0)
    n_b = sum(1 for p in got if p.requirement_a == bp.BIDPO_BENIGN_REQ_A)
    n_h = len(got) - n_b
    assert (n_h, n_b) == (2, 2)          # the min rule: benign = min(h, avail)
    # harmful is NEVER subsampled, whatever the cap
    got3 = tt.pairs_for_arm("bidpo", [], bidpo_harmful_pairs=harmful,
                            bidpo_benign_pairs=benign, bidpo_benign_cap=3.0)
    assert sum(1 for p in got3
               if p.requirement_a != bp.BIDPO_BENIGN_REQ_A) == 2
    assert sum(1 for p in got3
               if p.requirement_a == bp.BIDPO_BENIGN_REQ_A) == 6
    # uncapped (None) takes every qualified benign pair
    got_all = tt.pairs_for_arm("bidpo", [], bidpo_harmful_pairs=harmful,
                               bidpo_benign_pairs=benign,
                               bidpo_benign_cap=None)
    assert sum(1 for p in got_all
               if p.requirement_a == bp.BIDPO_BENIGN_REQ_A) == 10
    # benign short of the cap: take everything (min rule's other branch)
    got_short = tt.pairs_for_arm("bidpo", [], bidpo_harmful_pairs=harmful,
                                 bidpo_benign_pairs=[_bpair(0)],
                                 bidpo_benign_cap=1.0)
    assert sum(1 for p in got_short
               if p.requirement_a == bp.BIDPO_BENIGN_REQ_A) == 1
    # deterministic: same call -> same pairs; reversed input order -> the
    # SAME SUBSET is selected (sha-ranked, order-independent)
    again = tt.pairs_for_arm("bidpo", [], bidpo_harmful_pairs=harmful,
                             bidpo_benign_pairs=benign, bidpo_benign_cap=1.0)
    assert [p.prompt for p in again] == [p.prompt for p in got]
    flipped = tt.pairs_for_arm("bidpo", [], bidpo_harmful_pairs=harmful,
                               bidpo_benign_pairs=list(reversed(benign)),
                               bidpo_benign_cap=1.0)
    assert ({p.prompt for p in flipped
             if p.requirement_a == bp.BIDPO_BENIGN_REQ_A}
            == {p.prompt for p in got
                if p.requirement_a == bp.BIDPO_BENIGN_REQ_A})


def test_one_sided_supply_raises():
    with pytest.raises(ValueError, match="BOTH sides"):
        tt.pairs_for_arm("bidpo", [], bidpo_harmful_pairs=[_hpair(0)])
    with pytest.raises(ValueError, match="BOTH sides"):
        tt.pairs_for_arm("bidpo", [], bidpo_benign_pairs=[_bpair(0)])


def test_empty_sides_raise_loudly():
    with pytest.raises(ValueError, match="ZERO harmful"):
        tt.pairs_for_arm("bidpo", [], bidpo_harmful_pairs=[],
                         bidpo_benign_pairs=[_bpair(0)])
    with pytest.raises(ValueError, match="ZERO benign-reversed"):
        tt.pairs_for_arm("bidpo", [], bidpo_harmful_pairs=[_hpair(0)],
                         bidpo_benign_pairs=[])


def test_wrong_polarity_file_is_refused():
    """A matched-pool pairs file passed as the benign-reversed pool would
    train INVERTED preferences — refused by the requirement_a stamp."""
    with pytest.raises(ValueError, match="OPPOSITE polarity"):
        tt.pairs_for_arm("bidpo", [], bidpo_harmful_pairs=[_hpair(0)],
                         bidpo_benign_pairs=[_matched_pair(1, req="benign")])
    # ...and a benign-reversed file passed as the harmful side is refused too.
    with pytest.raises(ValueError, match="harmful file"):
        tt.pairs_for_arm("bidpo", [], bidpo_harmful_pairs=[_bpair(0)],
                         bidpo_benign_pairs=[_bpair(1)])


def test_degenerate_pair_is_refused():
    bad = Pair(prompt="SYNTHETIC benign prompt x 99999999", pos="",
               neg=B_REFUSE.format(i=9), instruction_id="safety_refusal",
               requirement_a=bp.BIDPO_BENIGN_REQ_A)
    with pytest.raises(ValueError, match="empty"):
        tt.pairs_for_arm("bidpo", [], bidpo_harmful_pairs=[_hpair(0)],
                         bidpo_benign_pairs=[bad])


def test_nonpositive_cap_is_refused():
    with pytest.raises(ValueError, match="must be > 0"):
        tt.pairs_for_arm("bidpo", [], bidpo_harmful_pairs=[_hpair(0)],
                         bidpo_benign_pairs=[_bpair(0)], bidpo_benign_cap=0)


def test_feature_off_is_the_legacy_behavior():
    pool = [_matched_pair(0), _matched_pair(1, req="relaxed"),
            _matched_pair(2, req="benign")]
    got = tt.pairs_for_arm("bidpo", pool)      # no own pool supplied
    assert [p.requirement_a for p in got] == ["strict"]


# --------------------------------------------------------------------------- #
# 3. emit-pairs end to end: manifest block -> per-SI files + sidecar            #
# --------------------------------------------------------------------------- #

SI2 = ("safety_refusal", "si_constitution")
E_HPROMPT = {0: "SYNTHETIC emit harmful prompt zero aaaa0000",
             1: "SYNTHETIC emit harmful prompt one aaaa1111"}
E_HNEG = "SYNTHETIC emit undefended harmful completion bbbb0000"
E_HPOS = {"safety_refusal": "SYNTHETIC emit safety refusal cccc0000",
          "si_constitution": "SYNTHETIC emit constitution refusal dddd0000"}
E_ANTI = {0: "SYNTHETIC emit anti compliance zero pppp0000",
          1: "SYNTHETIC emit anti compliance one pppp1111"}
E_BPROMPT = {0: "SYNTHETIC emit benign question zero eeee0000",
             1: "SYNTHETIC emit benign question one eeee1111",
             2: "SYNTHETIC emit benign question two eeee2222"}
E_BNEG = {i: f"SYNTHETIC emit undefended benign answer {i} ffff" for i in
          range(3)}
E_BREFUSE = {(si, i): f"SYNTHETIC emit over-refusal {si} q{i} gggg"
             for si in SI2 for i in range(3)}
E_BCOMPLY = {(si, i): f"SYNTHETIC emit compliance {si} q{i} hhhh"
             for si in SI2 for i in range(3)}


def _store_row(fam, gid, sample, si_condition, prompt, resp, *, score=None,
               si_id=None, cidx=None, xstest_label=None, order=None,
               group="held_in"):
    row = {"attack": fam, "goal_id": gid, "si_condition": si_condition,
           "category": "illegal", "category_group": group,
           "attack_prompt": prompt, "response": resp,
           "score": score, "graded": score is not None}
    if sample is not None:
        row["sample_idx"], row["trial"] = sample, 0
    if order is not None:
        row["order"] = order
    if si_id is not None:
        row["si_id"] = si_id
    if cidx is not None:
        row["candidate_idx"] = cidx
    if xstest_label is not None:
        row["xstest_label"] = xstest_label
    return row


def _emit_world(tmp_path, *, enabled=True, none_labelled=True):
    """A two-SI world: 2 matched harmful prompts (both SIs thwart both, both
    with an accepted anti compliance), benign q0 all-comply, q1 refused by
    safety only, q2 refused by BOTH SIs — undefended labelled C (unless
    ``none_labelled`` is False, the awaiting-grading state)."""
    cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    cfg["si_ids"] = list(SI2)
    cfg["pool"] = {**cfg["pool"], "store_dir": str(tmp_path / "labeled"),
                   "min_pool_size": 1, "static_families": ["bon"],
                   "train_families": ["bon"], "hard_floor_pairs": 0,
                   "benign_source": "phtest", "benign_sources": [],
                   "benign_fraction": 0.5, "scoring_disjointness": {},
                   "fit_floor": {}, "relaxed_stratum": {"enabled": False},
                   # pre-flip world: the store serves safety_refusal's pos
                   "pos_runs": {k: v
                                for k, v in (cfg["pool"].get("pos_runs")
                                             or {}).items()
                                if k != "safety_refusal"},
                   "anti_neg": {"enabled": True, "anti_si_id": ANTI,
                                "run": "two_si_anti_neg", "candidates": 8,
                                "temperature": 1.0},
                   "bidpo_benign": {"enabled": enabled,
                                    "refusal_labels": ["I"], "floor": 0,
                                    "max_benign_per_harmful": 1.0}}
    cfg["pool_manifest"] = str(tmp_path / "pool_manifest.json")

    store = tmp_path / "labeled"
    store.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in (0, 1):
        rows += [_store_row("bon", f"g{idx}", idx, "none", E_HPROMPT[idx],
                            E_HNEG, score=0.9),
                 _store_row("bon", f"g{idx}", idx, "prompted", E_HPROMPT[idx],
                            E_HPOS["safety_refusal"], score=0.0)]
    # A held-out goal: the shipped config holds out at the GOAL level
    # (pool.family_disjointness: false), so an all-held_in store leaves
    # nothing held out in either dimension and the builder refuses.
    rows += [_store_row("bon", "g_held", 2, "none", E_HPROMPT[0], E_HNEG,
                        score=0.9, group="held_out"),
             _store_row("bon", "g_held", 2, "prompted", E_HPROMPT[0],
                        E_HPOS["safety_refusal"], score=0.0,
                        group="held_out")]
    (store / "bon.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))

    refused_by = {0: (), 1: ("safety_refusal",), 2: SI2}
    brows = []
    for i in range(3):
        gid = f"phtest_{i}"
        brows.append(_store_row(
            "phtest", gid, None, "none", E_BPROMPT[i], E_BNEG[i], order=0,
            xstest_label="C" if none_labelled else None))
        for si in SI2:
            refused = si in refused_by[i]
            brows.append(_store_row(
                "phtest", gid, None, "prompted", E_BPROMPT[i],
                E_BREFUSE[(si, i)] if refused else E_BCOMPLY[(si, i)],
                order=0, si_id=si, cidx=0,
                xstest_label="I" if refused else "C"))
    (store / "phtest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in brows))

    priv = tmp_path / "priv"
    cdir = priv / cfg["pool"]["constitution_run"]
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "bon.jsonl").write_text("".join(json.dumps(
        _store_row("bon", f"g{idx}", idx, "prompted", E_HPROMPT[idx],
                   E_HPOS["si_constitution"], score=0.0,
                   si_id="si_constitution", cidx=0)) + "\n"
        for idx in (0, 1)))
    adir = priv / "two_si_anti_neg"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "anti_neg.jsonl").write_text("".join(json.dumps(
        _store_row("bon", f"g{idx}", idx, "prompted", E_HPROMPT[idx],
                   E_ANTI[idx], score=0.9, si_id=ANTI, cidx=0)) + "\n"
        for idx in (0, 1)))
    return cfg, priv


def _emit(cfg, priv, out):
    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None,
                                          families=None), cfg)
    bp.cmd_emit_pairs(types.SimpleNamespace(
        private_root=str(priv), out_dir=str(out), dry_run=False,
        manifest=None), cfg)


def test_emit_writes_bidpo_files_with_the_right_polarity(tmp_path):
    cfg, priv = _emit_world(tmp_path)
    out = tmp_path / "pairs"
    _emit(cfg, priv, out)

    m = json.loads(Path(cfg["pool_manifest"]).read_text())
    per = m["bidpo_benign"]["per_si"]
    assert per["safety_refusal"]["benign"]["n_qualified"] == 2     # q1, q2
    assert per["si_constitution"]["benign"]["n_qualified"] == 1    # q2
    assert per["safety_refusal"]["harmful"]["n_with_anti_neg"] == 2

    for si in SI2:
        harm = tt._load_pairs(str(out / bp.bidpo_harmful_filename(si)))
        assert {p.prompt for p in harm} == set(E_HPROMPT.values())
        for p in harm:
            assert p.pos == E_HPOS[si]           # CHOSEN: this SI's refusal
            assert p.neg in set(E_ANTI.values())  # REJECTED: anti compliance
            assert p.requirement_a == "strict"

    safety = tt._load_pairs(str(out / bp.bidpo_benign_filename(SI2[0])))
    assert {p.prompt for p in safety} == {E_BPROMPT[1], E_BPROMPT[2]}
    for p in safety:
        i = 1 if p.prompt == E_BPROMPT[1] else 2
        assert p.pos == E_BNEG[i]                          # CHOSEN: compliance
        assert p.neg == E_BREFUSE[("safety_refusal", i)]   # REJECTED: refusal
        assert p.requirement_a == bp.BIDPO_BENIGN_REQ_A
    const = tt._load_pairs(str(out / bp.bidpo_benign_filename(SI2[1])))
    assert [p.prompt for p in const] == [E_BPROMPT[2]]
    assert const[0].neg == E_BREFUSE[("si_constitution", 2)]

    sidecar = json.loads((out / bp.PAIRS_META_NAME).read_text())
    blk = sidecar["bidpo_benign"]
    assert blk["per_si"]["safety_refusal"]["n_harmful"] == 2
    assert blk["per_si"]["safety_refusal"]["n_benign_qualified"] == 2
    assert (blk["per_si"]["si_constitution"]
            ["projected_mix_at_default_1to1"]["ratio_harmful_to_benign"]
            == 2.0)
    assert blk["refusal_labels"] == ["I"]
    assert "identity_exception" in blk
    assert blk["files"][SI2[0]]["benign"]["n_pairs"] == 2
    assert blk["files"][SI2[0]]["harmful"]["n_pairs"] == 2


def test_matched_pairs_files_byte_identical_with_feature_toggled(tmp_path):
    cfg_on, priv = _emit_world(tmp_path)
    out_on = tmp_path / "pairs_on"
    _emit(cfg_on, priv, out_on)
    cfg_off, _ = _emit_world(tmp_path, enabled=False)
    out_off = tmp_path / "pairs_off"
    _emit(cfg_off, priv, out_off)
    for si in SI2:
        on = (out_on / bp.pairs_filename(si)).read_bytes()
        off = (out_off / bp.pairs_filename(si)).read_bytes()
        assert on == off                       # other arms' pool: untouched
        assert not (out_off / bp.bidpo_benign_filename(si)).exists()
        assert not (out_off / bp.bidpo_harmful_filename(si)).exists()


def test_manifest_raises_on_unlabelled_undefended_rows(tmp_path):
    cfg, priv = _emit_world(tmp_path, none_labelled=False)
    with pytest.raises(MatchedPoolError, match="grade-store"):
        bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv),
                                              out=None, families=None), cfg)


def test_emit_raises_when_store_diverged_from_frozen_block(tmp_path):
    cfg, priv = _emit_world(tmp_path)
    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None,
                                          families=None), cfg)
    # regenerate a frozen REJECTED side in the store after the freeze
    store = Path(cfg["pool"]["store_dir"])
    rows = [json.loads(l) for l in
            (store / "phtest.jsonl").read_text().splitlines()]
    for r in rows:
        if (r.get("si_id") == "safety_refusal"
                and r["goal_id"] == "phtest_1"):
            r["response"] = "SYNTHETIC regenerated refusal text zzzz"
    (store / "phtest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(MatchedPoolError, match="rejected_sha"):
        bp.cmd_emit_pairs(types.SimpleNamespace(
            private_root=str(priv), out_dir=str(tmp_path / "p"),
            dry_run=False, manifest=None), cfg)


# --------------------------------------------------------------------------- #
# 4. grade-store: benign none rows are labelled iff the feature is enabled      #
# --------------------------------------------------------------------------- #


def _grade_world(tmp_path, *, enabled):
    cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    cfg["si_ids"] = list(SI2)
    cfg["pool"] = {**cfg["pool"], "store_dir": str(tmp_path / "labeled"),
                   "static_families": ["bon"], "train_families": ["bon"],
                   "replay_families": [],
                   "benign_source": "phtest", "benign_sources": [],
                   "anti_neg": {"enabled": False},
                   "bidpo_benign": {"enabled": enabled}}
    store = tmp_path / "labeled"
    store.mkdir(parents=True, exist_ok=True)
    brows = [
        _store_row("phtest", "b0", None, "none",
                   "SYNTHETIC benign store q0 iiii0000",
                   "SYNTHETIC undefended answer q0 jjjj0000", order=0),
        _store_row("phtest", "b1", None, "none",
                   "SYNTHETIC benign store q1 iiii1111",
                   "SYNTHETIC undefended answer q1 jjjj1111", order=0,
                   xstest_label="C"),           # already labelled — resume
        _store_row("phtest", "b0", None, "prompted",
                   "SYNTHETIC benign store q0 iiii0000",
                   "SYNTHETIC si answer q0 kkkk0000", order=0,
                   si_id=SI2[0], cidx=0),
    ]
    p = store / "phtest.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in brows))
    p.chmod(0o600)
    return cfg, store


def _fake_xs(monkeypatch, label="C"):
    seen = {"xs": []}

    def _build_xs(args, cfg, cache_dir):
        def grade_xs(items):
            seen["xs"].extend(items)
            return {it: label for it in items}
        return grade_xs
    monkeypatch.setattr(bp, "_build_xstest_grader", _build_xs)
    monkeypatch.setattr(
        bp, "_build_grader",
        lambda *a, **k: (lambda items: {it: 0.0 for it in items}))
    return seen


def test_grade_store_labels_none_rows_when_enabled(tmp_path, monkeypatch,
                                                   capsys):
    cfg, store = _grade_world(tmp_path, enabled=True)
    seen = _fake_xs(monkeypatch)
    bp.cmd_grade_store(types.SimpleNamespace(store_dir=None, dry_run=False,
                                             concurrency=4), cfg)
    rows = [json.loads(l) for l in
            (store / "phtest.jsonl").read_text().splitlines()]
    by_gid_cond = {(r["goal_id"], r["si_condition"]): r for r in rows}
    # the unlabelled none row got its label; the labelled one was not re-sent
    assert by_gid_cond[("b0", "none")]["xstest_label"] == "C"
    assert by_gid_cond[("b0", "none")]["graded"] is True
    sent = {it[1] for it in seen["xs"]}
    assert "b1" not in sent
    # the SI row is labelled through the same pass
    assert by_gid_cond[("b0", "prompted")]["xstest_label"] == "C"


def test_grade_store_dry_run_counts_none_rows_only_when_enabled(
        tmp_path, monkeypatch, capsys):
    for enabled, expect_key, n_calls in ((True, "unlabelled_none_rows", 2),
                                         (False, "none_rows_needing_no_label",
                                          1)):
        cfg, _store = _grade_world(tmp_path / str(enabled), enabled=enabled)
        bp.cmd_grade_store(types.SimpleNamespace(store_dir=None, dry_run=True,
                                                 concurrency=4), cfg)
        report = json.loads(capsys.readouterr().out)
        assert expect_key in report["benign"]["phtest"]
        assert report["totals"]["xstest_round0_calls"] == n_calls


# --------------------------------------------------------------------------- #
# 5. the fit driver: routing, meta.json + fit-report counts, loud failures      #
# --------------------------------------------------------------------------- #

DRIVER_CFG = {
    "layer": 1,
    "completion_tokens": 16,
    "max_len": 256,
    "max_len_by_si": {"safety_refusal": 256},
    "train_kwargs": {"default": {"epochs": 1, "seed": 0,
                                 "holdout_frac": 0.0,
                                 "floor_ratio_max": None}},
    "si_ids": ["safety_refusal", "si_constitution"],
    "pool": {"bidpo_benign": {"enabled": True, "refusal_labels": ["I"],
                              "floor": 0, "max_benign_per_harmful": 1.0}},
}

PER_SI_FROZEN = {
    "safety_refusal": {
        "n_harmful": 3000, "n_benign_qualified": 3000,
        "projected_mix_at_default_1to1": {
            "n_harmful": 3000, "n_benign": 3000,
            "ratio_harmful_to_benign": 1.0}},
    "si_irrelevant": {
        "n_harmful": 3000, "n_benign_qualified": 660,
        "projected_mix_at_default_1to1": {
            "n_harmful": 3000, "n_benign": 660,
            "ratio_harmful_to_benign": 4.5455}},
}
PER_SI_META = {"refusal_labels": ["I"], "per_si_frozen": PER_SI_FROZEN}


class _FakeObj:
    # Not a registry class, so the driver's _FIT_SITES table (rightly) does
    # not know it — self-report through the fit_site seam instead.
    fit_site = "test_double"

    def save(self, path):
        Path(path).write_text("SYNTHETIC-object")


def _fake_arms(monkeypatch, captured):
    from sisteer.methods import ARMS, S_TWO_SI, ArmSpec

    def _train(name):
        def train(model, tok, instruction, pairs, *, layer, log_fn=None,
                  **kw):
            captured[name] = list(pairs)
            return _FakeObj()
        return train

    for arm in ("bidpo", "bidpo_lora", "diffmean"):
        monkeypatch.setitem(ARMS, arm, ArmSpec(
            arm, lambda *a, **k: _FakeObj(), _train(arm),
            frozenset({S_TWO_SI})))


def test_driver_routes_the_own_pool_to_bidpo_only_and_records_the_confound(
        tmp_path, monkeypatch):
    captured = {}
    _fake_arms(monkeypatch, captured)
    pool = [_matched_pair(0), _matched_pair(1),
            _matched_pair(2, req="benign")]
    harmful = [_hpair(0), _hpair(1)]
    benign = [_bpair(0), _bpair(1), _bpair(2)]
    out = tt.train_si_objects(
        None, None, "safety_refusal", pool,
        ["bidpo", "bidpo_lora", "diffmean"],
        cfg=DRIVER_CFG, save_dir=tmp_path,
        bidpo_harmful_pairs=harmful, bidpo_benign_pairs=benign,
        bidpo_benign_meta=dict(PER_SI_META))
    assert out.ok
    # the sibling DPO arm consumed EXACTLY the same pairs as bidpo, and got
    # its own confound record.
    assert captured["bidpo_lora"] == captured["bidpo"]
    lora_meta = json.loads(
        (tmp_path / "safety_refusal__bidpo_lora" / "train_budget_meta.json")
        .read_text())
    assert lora_meta["bidpo_benign"]["realised_ratio_harmful_to_benign"] == 1.0
    # diffmean saw the untouched pool — nothing from bidpo's own files
    assert [p.prompt for p in captured["diffmean"]] == [p.prompt for p in pool]
    assert all(p.requirement_a != bp.BIDPO_BENIGN_REQ_A
               for p in captured["diffmean"])
    # bidpo: 2 harmful (never subsampled) + min(2, 3) = 2 benign-reversed,
    # and NONE of the matched pairs.
    n_b = sum(1 for p in captured["bidpo"]
              if p.requirement_a == bp.BIDPO_BENIGN_REQ_A)
    assert (len(captured["bidpo"]) - n_b, n_b) == (2, 2)
    assert all("matched prompt" not in p.prompt for p in captured["bidpo"])

    # the section-13 confound record: counts AND ratios, this object's
    # realised + every SI's frozen, in BOTH meta.json and the fit report.
    meta = json.loads(
        (tmp_path / "safety_refusal__bidpo" / "train_budget_meta.json")
        .read_text())
    bb = meta["bidpo_benign"]
    assert bb["n_harmful_pairs"] == 2 and bb["n_benign_pairs"] == 2
    assert bb["realised_ratio_harmful_to_benign"] == 1.0
    assert bb["realised_benign_fraction"] == 0.5
    assert bb["parity"] == "per_si_unequal_no_parity_subsample"
    assert bb["harmful_selection"] == "per_si_thwart_with_anti_neg"
    assert (bb["per_si_frozen"]["si_irrelevant"]
            ["projected_mix_at_default_1to1"]["ratio_harmful_to_benign"]
            == 4.5455)
    assert bb["refusal_labels"] == ["I"]
    report = json.loads(
        (tmp_path / "safety_refusal__fit_report.json").read_text())
    detail = report["arms"]["bidpo"]["detail"]
    assert detail["bidpo_benign"]["realised_ratio_harmful_to_benign"] == 1.0
    assert (detail["bidpo_benign"]["per_si_frozen"]["safety_refusal"]
            ["n_harmful"] == 3000)
    assert detail["negative_source"]["source"] == (
        "anti_safety_compliance_per_si_thwart")
    assert detail["negative_source"]["n_pairs"] == 2   # harmful side only


def test_driver_fails_bidpo_arm_when_enabled_but_pairs_missing(
        tmp_path, monkeypatch):
    captured = {}
    _fake_arms(monkeypatch, captured)
    pool = [_matched_pair(0), _matched_pair(1)]
    out = tt.train_si_objects(
        None, None, "safety_refusal", pool, ["bidpo", "diffmean"],
        cfg=DRIVER_CFG, save_dir=tmp_path)     # no own pool supplied
    # isolation: bidpo fails LOUDLY, the other arm still trains and saves
    assert out.arms["bidpo"].status == "failed"
    assert "bidpo" in out.arms["bidpo"].error
    assert out.arms["diffmean"].status == "trained"
    assert "bidpo" not in captured


def test_driver_refuses_own_pool_when_feature_disabled(tmp_path):
    cfg = {**DRIVER_CFG,
           "pool": {"bidpo_benign": {"enabled": False}}}
    with pytest.raises(SystemExit, match="contradictory"):
        tt.train_si_objects(None, None, "safety_refusal", [_matched_pair(0)],
                            ["bidpo"], cfg=cfg, save_dir=tmp_path,
                            bidpo_harmful_pairs=[_hpair(0)],
                            bidpo_benign_pairs=[_bpair(0)])


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _train_args(tmp_path, **over):
    ppath = tmp_path / "pairs_safety_refusal.jsonl"
    _write_jsonl(ppath, [
        {"prompt": "SYNTHETIC p 11112222", "pos": "SYNTHETIC pos",
         "neg": "SYNTHETIC neg", "requirement_a": "strict"}])
    base = dict(config="unused", si="safety_refusal", arms="bidpo",
                pairs=str(ppath), out=str(tmp_path / "cells"),
                overwrite=False, bidpo_harmful_pairs="",
                bidpo_benign_pairs="")
    base.update(over)
    return types.SimpleNamespace(**base)


def _write_bidpo_files(tmp_path, *, sidecar=True, corrupt=False):
    hfile = tmp_path / bp.bidpo_harmful_filename("safety_refusal")
    _write_jsonl(hfile, [{
        "prompt": "SYNTHETIC hq 33330000", "pos": "SYNTHETIC refusal",
        "neg": "SYNTHETIC anti compliance", "requirement_a": "strict"}])
    bfile = tmp_path / bp.bidpo_benign_filename("safety_refusal")
    _write_jsonl(bfile, [{
        "prompt": "SYNTHETIC bq 33334444", "pos": "SYNTHETIC compliance",
        "neg": "SYNTHETIC refusal",
        "requirement_a": bp.BIDPO_BENIGN_REQ_A}])
    if sidecar:
        shas = {side: hashlib.sha256(f.read_bytes()).hexdigest()
                for side, f in (("harmful", hfile), ("benign", bfile))}
        if corrupt:
            shas["benign"] = "0" * 64
        (tmp_path / bp.PAIRS_META_NAME).write_text(json.dumps({
            "bidpo_benign": {
                "refusal_labels": ["I"], "per_si": PER_SI_FROZEN,
                "files": {"safety_refusal": {
                    "harmful": {"file": hfile.name, "n_pairs": 1,
                                "file_sha256": shas["harmful"]},
                    "benign": {"file": bfile.name, "n_pairs": 1,
                               "file_sha256": shas["benign"]}}}}}))


def test_cmd_train_fails_fast_when_bidpo_files_missing(tmp_path):
    args = _train_args(tmp_path)
    with pytest.raises(SystemExit, match="missing"):
        tt.cmd_train(args, DRIVER_CFG)


def test_cmd_train_fails_fast_without_the_sidecar_counts(tmp_path):
    args = _train_args(tmp_path)
    _write_bidpo_files(tmp_path, sidecar=False)
    with pytest.raises(SystemExit, match="sidecar"):
        tt.cmd_train(args, DRIVER_CFG)


def test_cmd_train_refuses_a_sha_mismatched_pool_file(tmp_path):
    args = _train_args(tmp_path)
    _write_bidpo_files(tmp_path, corrupt=True)
    with pytest.raises(SystemExit, match="sha256"):
        tt.cmd_train(args, DRIVER_CFG)


def test_load_bidpo_pool_returns_verified_pairs_and_meta(tmp_path):
    args = _train_args(tmp_path)
    _write_bidpo_files(tmp_path)
    harmful, benign, meta = tt._load_bidpo_pool(args, DRIVER_CFG, ["bidpo"])
    assert len(harmful) == 1 and harmful[0].requirement_a == "strict"
    assert len(benign) == 1
    assert benign[0].requirement_a == bp.BIDPO_BENIGN_REQ_A
    assert meta["per_si_frozen"]["si_irrelevant"]["n_benign_qualified"] == 660
    # feature off, or bidpo not requested -> nothing loaded
    off_cfg = {**DRIVER_CFG, "pool": {"bidpo_benign": {"enabled": False}}}
    assert tt._load_bidpo_pool(args, off_cfg, ["bidpo"]) == (None, None, None)
    assert tt._load_bidpo_pool(args, DRIVER_CFG, ["diffmean"]) == (
        None, None, None)
