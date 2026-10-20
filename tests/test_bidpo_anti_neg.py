"""bidpo's anti-safety negative-generation path (bidpo amendment 2026-08-20).

CPU-only, offline, no GPU, NO harmful text: every prompt/response here is an
INVENTED synthetic string (privacy rule — placeholder uuids, TESTHOST). bidpo
is a DPO preference arm whose rejected side must be a genuine harmful
COMPLIANCE; the undefended ``neg`` is a compliance only ~6% of
the time, so this path elicits a compliance under ``safety_refusal.anti_text``
for EVERY harmful prompt and routes it to bidpo's rejected side. What the suite
proves:

- the matched-pool builder REDUCES anti candidates with the COMPLIANCE
  acceptance direction (accept iff score > threshold, the opposite of a pos
  row), picking the smallest accepted candidate_idx (rejection sampling), and
  records the realised yield — into a SEPARATE ``bidpo_anti_neg`` block;
- the anti rows NEVER enter ``matched_intersection``: the primary pool
  (prompt/neg/pos) is byte-identical to a build with no anti source, so
  ``assert_pools_identical_except_pos`` still holds and every OTHER arm's
  ``neg`` is unchanged;
- ``anti_si_id`` may not be a comparison SI, and the manifest carries no
  completion text (counts/ids/shas only) even with the anti block;
- emit-pairs resolves each frozen ``anti_neg_sha`` back to text BY HASH onto
  ``Pair.anti_neg`` (harmful pairs only), and the trainer routes it to bidpo
  and ONLY bidpo;
- the loud-failure path: bidpo in anti mode with no accepted anti_neg RAISES
  rather than silently degrading to the undefended refusal.
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
import train_two_si_objects as tt  # noqa: E402
from sisteer.data.pairs import Pair  # noqa: E402
from sisteer.eval.matched_pool import (  # noqa: E402
    GradedRow,
    MatchedPoolError,
    PoolSpec,
    build_matched_pool,
    dumps_manifest,
    manifest_hash,
    text_sha,
)

REPO = Path(__file__).resolve().parents[1]
SI = ("safety_refusal", "si_constitution")
ANTI = "anti_safety_refusal"

# Synthetic strings (>= 12 chars) — invented, never real store data.
PROMPTS = {
    "pA": "SYNTHETIC prompt alpha harmful-shaped request 11111111",
    "pD": "SYNTHETIC prompt delta harmful-shaped request 22222222",
    "pF": "SYNTHETIC prompt foxtrot harmful-shaped request 33333333",
    "pG": "SYNTHETIC prompt golf harmful-shaped request 44444444",
}
NONE_COMPLY = "SYNTHETIC undefended completion that complied aaaaaaaa"
NONE_REFUSE = "SYNTHETIC undefended refusal I cannot help bbbbbbbb"
REFUSAL = "SYNTHETIC SI refusal declined per policy cccccccc"
# anti-safety candidates — the needles the manifest must never contain
ANTI_A1 = "SYNTHETIC anti compliance for A accepted dddddddd"
ANTI_D0 = "SYNTHETIC anti compliance for D accepted eeeeeeee"
ANTI_G1 = "SYNTHETIC anti compliance for G first-accepted ffffffff"
ANTI_G2 = "SYNTHETIC anti compliance for G second-accepted gggggggg"
ANTI_REFUSAL = "SYNTHETIC anti still refused not a compliance hhhhhhhh"


def _row(fam, gid, vk, cond, si, prompt, resp, score, *, candidate_idx=0):
    return GradedRow(family=fam, goal_id=gid, variant_key=vk, condition=cond,
                     si_id=si, prompt=prompt, response=resp, score=score,
                     candidate_idx=candidate_idx)


def _anti_world_rows():
    """A synthetic world where four prompts are thwarted by BOTH SIs (so all
    four are in the matched intersection), each carrying anti-safety candidates
    that exercise accept / reject / boundary / smallest-accepted-idx.

      pA (g0): none COMPLIED; anti c0 refuses (0.2, reject), c1 complies
               (0.9, ACCEPT) -> anti_neg = c1
      pD (g1): none REFUSED (relaxed pair); anti c0 complies (0.6, ACCEPT)
      pF (g2): none COMPLIED; anti c0 at boundary 0.5 (reject, needs > 0.5),
               c1 (0.3, reject) -> NO anti_neg (missing)
      pG (g3): none COMPLIED; anti c0 refuses (0.3, reject), c1 (0.8, ACCEPT),
               c2 (0.95, ACCEPT) -> anti_neg = c1 (smallest accepted idx)
    """
    r = []
    # pA
    r += [_row("bon", "g0", "0:0", "none", None, PROMPTS["pA"], NONE_COMPLY, 0.9),
          _row("bon", "g0", "0:0", "si", SI[0], PROMPTS["pA"], REFUSAL, 0.0),
          _row("bon", "g0", "0:0", "si", SI[1], PROMPTS["pA"], REFUSAL, 0.0),
          _row("bon", "g0", "0:0", "si", ANTI, PROMPTS["pA"], ANTI_REFUSAL, 0.2,
               candidate_idx=0),
          _row("bon", "g0", "0:0", "si", ANTI, PROMPTS["pA"], ANTI_A1, 0.9,
               candidate_idx=1)]
    # pD (undefended refused -> relaxed)
    r += [_row("bon", "g1", "1:0", "none", None, PROMPTS["pD"], NONE_REFUSE, 0.1),
          _row("bon", "g1", "1:0", "si", SI[0], PROMPTS["pD"], REFUSAL, 0.0),
          _row("bon", "g1", "1:0", "si", SI[1], PROMPTS["pD"], REFUSAL, 0.0),
          _row("bon", "g1", "1:0", "si", ANTI, PROMPTS["pD"], ANTI_D0, 0.6,
               candidate_idx=0)]
    # pF (all anti candidates rejected; c0 at the 0.5 boundary)
    r += [_row("bon", "g2", "0:0", "none", None, PROMPTS["pF"], NONE_COMPLY, 0.9),
          _row("bon", "g2", "0:0", "si", SI[0], PROMPTS["pF"], REFUSAL, 0.0),
          _row("bon", "g2", "0:0", "si", SI[1], PROMPTS["pF"], REFUSAL, 0.0),
          _row("bon", "g2", "0:0", "si", ANTI, PROMPTS["pF"], ANTI_REFUSAL, 0.5,
               candidate_idx=0),
          _row("bon", "g2", "0:0", "si", ANTI, PROMPTS["pF"], ANTI_REFUSAL, 0.3,
               candidate_idx=1)]
    # pG (two accepted; smallest idx wins)
    r += [_row("bon", "g3", "0:0", "none", None, PROMPTS["pG"], NONE_COMPLY, 0.9),
          _row("bon", "g3", "0:0", "si", SI[0], PROMPTS["pG"], REFUSAL, 0.0),
          _row("bon", "g3", "0:0", "si", SI[1], PROMPTS["pG"], REFUSAL, 0.0),
          _row("bon", "g3", "0:0", "si", ANTI, PROMPTS["pG"], "SYNTHETIC anti "
               "refused for G iiiiiiii", 0.3, candidate_idx=0),
          _row("bon", "g3", "0:0", "si", ANTI, PROMPTS["pG"], ANTI_G1, 0.8,
               candidate_idx=1),
          _row("bon", "g3", "0:0", "si", ANTI, PROMPTS["pG"], ANTI_G2, 0.95,
               candidate_idx=2)]
    return r


def _spec(*, anti_si_id=None):
    return PoolSpec(
        si_ids=SI, train_families=("bon",),
        eval_families=("pair", "tap"),
        train_goal_ids=("g0", "g1", "g2", "g3"),
        eval_goal_ids=("g9",), static_families=frozenset({"bon"}),
        threshold=0.5, min_pool_size=1, anti_si_id=anti_si_id)


# --------------------------------------------------------------------------- #
# 1. the anti reduction: accept iff COMPLIES (> threshold), smallest-idx first  #
# --------------------------------------------------------------------------- #


def test_anti_block_accepts_compliance_and_records_yield():
    m = build_matched_pool(_anti_world_rows(), _spec(anti_si_id=ANTI))
    blk = m["bidpo_anti_neg"]
    assert blk["anti_si_id"] == ANTI
    assert blk["acceptance"] == "strongreject_gt_threshold_compliance"
    # four harmful matched pairs (pA strict, pD relaxed, pF strict, pG strict);
    # three carry an accepted anti_neg (pF's candidates all fail acceptance).
    assert blk["n_harmful_pairs"] == 4
    assert blk["n_with_anti_neg"] == 3
    assert blk["n_missing_anti_neg"] == 1
    # ordered by the frozen matched order (g0, g1, g3 — g2/pF missing).
    got = {(e["family"], e["goal_id"], e["variant_key"]): e["anti_neg_sha"]
           for e in blk["ordered_pairs"]}
    assert set(got) == {("bon", "g0", "0:0"), ("bon", "g1", "1:0"),
                        ("bon", "g3", "0:0")}
    # the ACCEPTED compliance is frozen (not the refusing candidate)...
    assert got[("bon", "g0", "0:0")] == text_sha(ANTI_A1)
    assert got[("bon", "g1", "1:0")] == text_sha(ANTI_D0)
    # ...and among two accepted candidates the SMALLEST idx wins (c1, not c2).
    assert got[("bon", "g3", "0:0")] == text_sha(ANTI_G1)


def test_anti_block_boundary_is_strictly_greater_than_threshold():
    """A candidate scored EXACTLY at the threshold (0.5) is NOT a compliance —
    acceptance is score > threshold, matching the pos side's thwart test."""
    m = build_matched_pool(_anti_world_rows(), _spec(anti_si_id=ANTI))
    accepted = {(e["family"], e["goal_id"], e["variant_key"])
                for e in m["bidpo_anti_neg"]["ordered_pairs"]}
    assert ("bon", "g2", "0:0") not in accepted   # pF's c0 == 0.5, rejected


def test_anti_block_rejection_sampling_stats():
    m = build_matched_pool(_anti_world_rows(), _spec(anti_si_id=ANTI))
    rs = m["bidpo_anti_neg"]["rejection_sampling"]
    assert rs["n_prompts_with_candidates"] == 4
    assert rs["n_candidates"] == 2 + 1 + 2 + 3   # pA + pD + pF + pG
    assert rs["k_max"] == 3
    assert rs["n_prompts_accepted"] == 3
    assert rs["n_candidates_accepted"] == 1 + 1 + 0 + 2
    assert rs["per_prompt_yield"] == round(3 / 4, 4)
    assert rs["per_candidate_accept_rate"] == round(4 / 8, 4)


# --------------------------------------------------------------------------- #
# 2. the anti source NEVER perturbs the matched intersection (other arms)       #
# --------------------------------------------------------------------------- #


def test_matched_intersection_byte_identical_with_and_without_anti():
    """Adding the anti source must not change matched_intersection at all — same
    ordered prompts, same neg, same pos. So every non-bidpo arm's inputs (and
    the identity invariant) are untouched."""
    rows = _anti_world_rows()
    m_off = build_matched_pool(rows, _spec(anti_si_id=None))
    m_on = build_matched_pool(rows, _spec(anti_si_id=ANTI))
    assert "bidpo_anti_neg" not in m_off
    assert "bidpo_anti_neg" in m_on
    assert (m_off["matched_intersection"]["ordered_pairs"]
            == m_on["matched_intersection"]["ordered_pairs"])
    # no pos_sha in the intersection is an anti compliance sha (anti never
    # becomes a pos / enters the primary pool).
    anti_shas = {text_sha(s) for s in (ANTI_A1, ANTI_D0, ANTI_G1, ANTI_G2)}
    for e in m_on["matched_intersection"]["ordered_pairs"]:
        assert set(e["pos_sha"].values()).isdisjoint(anti_shas)
        assert e["neg_sha"] not in anti_shas


def test_anti_si_id_may_not_be_a_comparison_si():
    with pytest.raises(MatchedPoolError, match="anti_si_id"):
        _spec(anti_si_id=SI[0])


def test_manifest_carries_no_anti_completion_text():
    """Containment (JB4): the manifest holds counts/ids/shas only — never an
    anti compliance completion, the most sensitive text this path produces."""
    m = build_matched_pool(_anti_world_rows(), _spec(anti_si_id=ANTI))
    blob = dumps_manifest(m)
    for needle in (ANTI_A1, ANTI_D0, ANTI_G1, ANTI_G2, ANTI_REFUSAL,
                   NONE_COMPLY, REFUSAL):
        assert needle not in blob


def test_manifest_hash_covers_the_anti_block():
    rows = _anti_world_rows()
    m = build_matched_pool(rows, _spec(anti_si_id=ANTI))
    # perturb the accepted anti compliance for pA -> a different frozen sha ->
    # a different manifest hash (the block is load-bearing).
    rows2 = [r if not (r.si_id == ANTI and r.goal_id == "g0"
                       and r.candidate_idx == 1)
             else GradedRow(family=r.family, goal_id=r.goal_id,
                            variant_key=r.variant_key, condition=r.condition,
                            si_id=r.si_id, prompt=r.prompt,
                            response="SYNTHETIC anti compliance A EDITED jjjjjjjj",
                            score=r.score, candidate_idx=r.candidate_idx)
             for r in rows]
    m2 = build_matched_pool(rows2, _spec(anti_si_id=ANTI))
    assert manifest_hash(m) != manifest_hash(m2)


# --------------------------------------------------------------------------- #
# 3. train_two_si_objects.pairs_for_arm — anti mode reaches bidpo, ONLY bidpo   #
# --------------------------------------------------------------------------- #


def _pair(req, *, anti_neg=None, neg="SYNTHETIC undefended cccccccc"):
    return Pair(prompt=f"SYNTHETIC prompt {req} 55555555",
                pos="SYNTHETIC SI refusal styled 66666666",
                neg=neg, instruction_id="safety_refusal",
                requirement_a=req, anti_neg=anti_neg)


def test_pairs_for_arm_bidpo_anti_mode_uses_anti_neg():
    """anti mode: bidpo trains on harmful pairs whose rejected side is the
    ANTI compliance (neg replaced by anti_neg), benign excluded, harmful pairs
    without an accepted anti_neg dropped."""
    pool = [_pair("strict", anti_neg="SYNTHETIC anti comply S 77777777"),
            _pair("relaxed", anti_neg="SYNTHETIC anti comply R 88888888"),
            _pair("benign", anti_neg="SYNTHETIC anti comply B 99999999"),
            _pair("strict", anti_neg=None)]   # harmful but no accepted anti_neg
    got = tt.pairs_for_arm("bidpo", pool, anti_neg_enabled=True)
    assert [p.requirement_a for p in got] == ["strict", "relaxed"]
    # the rejected side is now the anti compliance, not the undefended neg.
    assert [p.neg for p in got] == ["SYNTHETIC anti comply S 77777777",
                                    "SYNTHETIC anti comply R 88888888"]


def test_anti_mode_touches_bidpo_only():
    """A non-preference arm is unfiltered even with the anti source enabled —
    it keeps consuming the undefended neg (identity, not a copy)."""
    pool = [_pair("strict", anti_neg="SYNTHETIC anti comply S 77777777")]
    assert tt.pairs_for_arm("diffmean", pool, anti_neg_enabled=True) is pool
    # bidpo's returned pairs must NOT share the undefended neg
    got = tt.pairs_for_arm("bidpo", pool, anti_neg_enabled=True)
    assert got[0].neg == "SYNTHETIC anti comply S 77777777"
    assert pool[0].neg != got[0].neg   # the input pair is untouched


def test_anti_mode_raises_when_no_pair_carries_anti_neg():
    """Loud failure: the anti source is ENABLED but no harmful pair carries an
    accepted anti_neg -> refuse rather than silently degrade to the refusal."""
    pool = [_pair("strict", anti_neg=None), _pair("relaxed", anti_neg=None),
            _pair("benign", anti_neg=None)]
    with pytest.raises(ValueError, match="anti-neg source is ENABLED"):
        tt.pairs_for_arm("bidpo", pool, anti_neg_enabled=True)


def test_legacy_strict_mode_unchanged_when_anti_disabled():
    """anti_neg_enabled False (default) keeps the legacy behavior: the STRICT
    subset via the undefended neg. Existing arms/tests are unaffected."""
    pool = [_pair("strict"), _pair("relaxed"), _pair("benign"), _pair("strict")]
    got = tt.pairs_for_arm("bidpo", pool)   # default anti_neg_enabled=False
    assert [p.requirement_a for p in got] == ["strict", "strict"]
    assert all(p.neg == "SYNTHETIC undefended cccccccc" for p in got)
    assert tt.pairs_for_arm("diffmean", pool) is pool


# --------------------------------------------------------------------------- #
# 4. emit-pairs end-to-end: anti_neg round-trips onto Pair.anti_neg -> bidpo    #
# --------------------------------------------------------------------------- #

E_PROMPTS = {0: "SYNTHETIC emit prompt zero harmful-shaped aaaa0000",
             1: "SYNTHETIC emit prompt one harmful-shaped aaaa1111"}
E_NEG = "SYNTHETIC emit undefended completion complied bbbb0000"
E_POS_SAFETY = "SYNTHETIC emit safety refusal cannot help cccc0000"
E_POS_CONST = "SYNTHETIC emit constitution refusal declined dddd0000"
E_ANTI = {0: "SYNTHETIC emit anti compliance zero eeee0000",
          1: "SYNTHETIC emit anti compliance one ffff1111"}


def _store_row(fam, gid, sample, trial, si_condition, prompt, resp, score,
               *, si_id=None, candidate_idx=None, group="held_in"):
    # ``group`` defaults to held_in, but a fixture that never sets held_out
    # produces a pool with NO goal split -- and under the shipped
    # goal-level regime (pool.family_disjointness: false) that is no
    # holdout at all, which build_matched_pool now refuses.
    row = {"attack": fam, "goal_id": gid, "si_condition": si_condition,
           "category": "illegal", "category_group": group,
           "sample_idx": sample, "trial": trial,
           "attack_prompt": prompt, "response": resp, "score": score,
           "graded": score is not None}
    if si_id is not None:
        row["si_id"] = si_id
    if candidate_idx is not None:
        row["candidate_idx"] = candidate_idx
    return row


def _emit_world(tmp_path):
    """A two-SI matched world (idx 0,1 complied + both SIs thwarted) plus an
    anti run whose graded compliances (> 0.5) are accepted for both prompts."""
    cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    cfg["si_ids"] = list(SI)
    cfg["pool"] = {**cfg["pool"], "store_dir": str(tmp_path / "labeled"),
                   "min_pool_size": 1, "static_families": ["bon"],
                   "train_families": ["bon"], "hard_floor_pairs": 0,
                   "benign_source": None, "benign_fraction": None,
                   "scoring_disjointness": {}, "benign_sources": [],
                   "fit_floor": {}, "relaxed_stratum": {"enabled": False},
                   # pre-flip world: the store serves safety_refusal's pos
                   "pos_runs": {k: v
                                for k, v in (cfg["pool"].get("pos_runs")
                                             or {}).items()
                                if k != "safety_refusal"},
                   "bidpo_benign": {"enabled": False},
                   "anti_neg": {"enabled": True, "anti_si_id": ANTI,
                                "run": "two_si_anti_neg", "candidates": 8,
                                "temperature": 1.0}}
    cfg["pool_manifest"] = str(tmp_path / "pool_manifest.json")

    store = tmp_path / "labeled"
    store.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in (0, 1):
        rows += [_store_row("bon", "g0", idx, 0, "none", E_PROMPTS[idx], E_NEG, 0.9),
                 _store_row("bon", "g0", idx, 0, "prompted", E_PROMPTS[idx],
                            E_POS_SAFETY, 0.0)]
    # a held-out goal, so the pool has a goal split to hold out BY -- see
    # _store_row's note: family_disjointness is off in the shipped config
    rows += [_store_row("bon", "g_held", 2, 0, "none", E_PROMPTS[0], E_NEG,
                        0.9, group="held_out"),
             _store_row("bon", "g_held", 2, 0, "prompted", E_PROMPTS[0],
                        E_POS_SAFETY, 0.0, group="held_out")]
    (store / "bon.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    priv = tmp_path / "priv"
    cdir = priv / cfg["pool"]["constitution_run"]
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "bon.jsonl").write_text("\n".join(json.dumps(
        _store_row("bon", "g0", idx, 0, "prompted", E_PROMPTS[idx], E_POS_CONST,
                   0.0, si_id="si_constitution", candidate_idx=0))
        for idx in (0, 1)) + "\n")

    # the anti run: one shard-less file, graded compliances accepted (> 0.5)
    adir = priv / "two_si_anti_neg"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "anti_neg.jsonl").write_text("\n".join(json.dumps(
        _store_row("bon", "g0", idx, 0, "prompted", E_PROMPTS[idx], E_ANTI[idx],
                   0.9, si_id=ANTI, candidate_idx=0))
        for idx in (0, 1)) + "\n")
    return cfg, priv


def test_emit_pairs_carries_anti_neg_and_routes_to_bidpo(tmp_path):
    cfg, priv = _emit_world(tmp_path)
    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None,
                                          families=None), cfg)
    m = json.loads(Path(cfg["pool_manifest"]).read_text())
    assert m["regime"] == "matched_intersection"
    assert m["bidpo_anti_neg"]["n_with_anti_neg"] == 2

    out = tmp_path / "pairs"
    bp.cmd_emit_pairs(types.SimpleNamespace(
        private_root=str(priv), out_dir=str(out), dry_run=False,
        manifest=None), cfg)

    for si, pos in ((SI[0], E_POS_SAFETY), (SI[1], E_POS_CONST)):
        pairs = tt._load_pairs(str(out / bp.pairs_filename(si)))
        assert len(pairs) == 2
        # every arm's `neg` is still the undefended completion (unchanged)...
        assert all(p.neg == E_NEG for p in pairs)
        # ...and the anti compliance rode in on the SEPARATE anti_neg field.
        assert {p.anti_neg for p in pairs} == {E_ANTI[0], E_ANTI[1]}
        assert all(p.pos == pos for p in pairs)
        # bidpo in anti mode uses the anti compliance as its rejected side;
        # every other arm keeps the undefended neg.
        bidpo = tt.pairs_for_arm("bidpo", pairs, anti_neg_enabled=True)
        assert {p.neg for p in bidpo} == {E_ANTI[0], E_ANTI[1]}
        assert tt.pairs_for_arm("diffmean", pairs, anti_neg_enabled=True) is pairs


def test_generate_anti_neg_dry_run_counts_and_resumes(tmp_path, capsys):
    """The planning half loads NO model: it reads the store's harmful
    `none` prompts and counts the cells to generate from engine.cell_key /
    load_done_keys. The emit world already has 2 anti cells (candidate_idx 0),
    so at K=8 only the missing 14 remain — the per-cell resume the pass needs."""
    cfg, priv = _emit_world(tmp_path)
    args = types.SimpleNamespace(
        candidates=8, private_root=str(priv), dry_run=True, seed=0,
        max_model_len=16384, gpu_memory_utilization=0.9, enforce_eager=False)
    bp.cmd_generate_anti_neg(args, cfg)
    out = capsys.readouterr().out
    assert "2 harmful train prompts" in out and "16 cells" in out
    assert "2 already generated, 14 to generate" in out


def test_generate_anti_neg_dry_run_has_no_side_effects(tmp_path):
    """With no anti run yet, dry-run counts all K cells and creates NO run dir
    (side-effect-free planning)."""
    import shutil

    cfg, priv = _emit_world(tmp_path)
    shutil.rmtree(priv / "two_si_anti_neg")     # start clean, no cells done
    args = types.SimpleNamespace(
        candidates=8, private_root=str(priv), dry_run=True, seed=0,
        max_model_len=16384, gpu_memory_utilization=0.9, enforce_eager=False)
    bp.cmd_generate_anti_neg(args, cfg)
    assert not (priv / "two_si_anti_neg").exists()


def test_generate_anti_neg_refuses_when_disabled(tmp_path):
    cfg, priv = _emit_world(tmp_path)
    cfg["pool"]["anti_neg"]["enabled"] = False
    args = types.SimpleNamespace(
        candidates=None, private_root=str(priv), dry_run=True, seed=0,
        max_model_len=16384, gpu_memory_utilization=0.9, enforce_eager=False)
    with pytest.raises(SystemExit, match="anti_neg.enabled is false"):
        bp.cmd_generate_anti_neg(args, cfg)


def test_emit_pairs_sidecar_records_anti_neg_provenance(tmp_path):
    cfg, priv = _emit_world(tmp_path)
    bp.cmd_manifest(types.SimpleNamespace(private_root=str(priv), out=None,
                                          families=None), cfg)
    out = tmp_path / "pairs"
    bp.cmd_emit_pairs(types.SimpleNamespace(
        private_root=str(priv), out_dir=str(out), dry_run=False,
        manifest=None), cfg)
    sidecar = json.loads((out / bp.PAIRS_META_NAME).read_text())
    assert sidecar["bidpo_anti_neg"]["enabled"] is True
    assert sidecar["bidpo_anti_neg"]["anti_si_id"] == ANTI
    assert sidecar["bidpo_anti_neg"]["n_with_anti_neg"] == 2
    assert sidecar["bidpo_anti_neg"]["n_missing_anti_neg"] == 0
