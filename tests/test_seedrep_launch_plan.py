"""scripts/modal/seedrep_launch_plan.py — the ONE generator of DRAFT / FINAL `launch_plan.json` + `capacity_ledger.json`
(the two materialisations). Every check asserts CONTENT and each rule has a KNOWN-BAD input:
  * same inputs -> byte-identical draft; a draft with non-default S4 fields REFUSES.
  * a GSM8K standup with an explicit --max-model-len / --max-num-seqs REFUSES; a standup serving MMLU without
    --max-logprobs 64 REFUSES; a table window with factors, a non-table window without the identity pair, a view engine
    missing a prompted route, an Arditi engine with two objects — each REFUSES by name.
  * the capacity ledger: 74 target containers, 4 pools at ONE H200 each (4 GPUs), 56 declared rung containers -> unstaged peak
    134 containers / 134 GPUs at the draft defaults (56 executable rung functions, the 4 GSM8K rungs included) — containers and GPUs counted separately.
  * planned standups - S5 references = empty with the ten windows referenced, the missing one named otherwise.
  * the derived adapter set is 17 (15 fitted + none__lora0 + the Arditi carrier), hashes filled from the smokes.
  * draft == final except the S4-decided fields (a differing pin anywhere else REFUSES).
  * the driver children carry the agreed flag names and one registry identity file per (leg, arm, seed); 27 arm-runs per leg.
  * the CLI mints RS/wave.json once, writes the draft + ledger, refuses a final without decisions, holds a final to its draft.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts" / "modal"):
    sys.path.insert(0, str(p))

import seedrep_launch_plan as lp  # noqa: E402

W = "2" * 40
WAVE = "22222222-3333-4444-8444-555555555555"


@pytest.fixture(scope="module")
def inputs():
    return lp.load_inputs(REPO)


def _plan(inputs, **kw):
    base = dict(W=W, wave_uuid=WAVE, reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"], pins=inputs["pins"],
                arditi_pins=inputs["arditi_pins"], arms=inputs["arms"], materialisation="draft", generated_at="2026-09-18T00:00:00Z")
    base.update(kw)
    return lp.build_plan(**base)


@pytest.fixture(scope="module")
def draft(inputs):
    return _plan(inputs)


def test_draft_is_byte_identical_for_the_same_inputs(inputs, draft):
    again = _plan(inputs)
    assert json.dumps(again, sort_keys=True) == json.dumps(draft, sort_keys=True)
    assert draft["launch_plan_sha"] == lp.plan_sha(draft) and len(draft["launch_plan_sha"]) == 16


def test_shape_74_targets_56_rungs_4_pools_27_arm_runs(draft):
    # the four GSM8K rung rows are EXECUTABLE (the queue-derived Inspect assembly); nothing is deferred -- 56 declared = 56 executable
    assert len(draft["standups"]) == 74 and len(draft["rungs"]) == 56 and len(draft["pools"]) == 4
    assert draft["deferred_rungs"] == [] and lp.GSM8K_RUNG_DEFERRED_WHY is None
    g8 = [r for r in draft["rungs"] if r["leg"] == "gsm8k"]
    assert [r["standup_id"] for r in g8] == [f"gsm8k__{e}__16384__shard00" for e in lp.ENGINES]
    assert all(r["rung_mode"] == lp.RUNG_MODE_GSM8K_SUBSET and r["rung_consumer"] is None and r["rung_assembly"]["kind"] == "gsm8k" for r in g8)
    assert all((r["geometry"]["L"], r["geometry"]["S"], r["geometry"]["thinking"]) == (24832, 21, True) for r in g8)
    assert len(lp.required_accepted_keys(draft, "gsm8k", "rungs")) == 4
    # a GSM8K rung child without its queue-derived --item-subset, or a base child WITH one, is refused by name
    bad = copy.deepcopy(draft)
    kid = [r for r in bad["rungs"] if r["leg"] == "gsm8k" and r["engine"] == "s1"][0]["children"][0]
    i = kid["argv"].index("--item-subset")
    del kid["argv"][i:i + 2]
    assert any("must carry --item-subset" in p for p in lp.validate_plan(bad))
    bad = copy.deepcopy(draft)
    kid = [r for r in bad["standups"] if r["leg"] == "gsm8k" and r["engine"] == "s1"][0]["children"][0]
    kid["argv"] += ["--item-subset", "/x/rung_subset.json"]
    assert any("BASE GSM8K child carries --item-subset" in p for p in lp.validate_plan(bad))
    by_leg = {}
    for r in draft["standups"]:
        by_leg[r["leg"]] = by_leg.get(r["leg"], 0) + 1
    assert by_leg == {"pasttense": 4, "codeattack": 4, "renellm": 4, "grok420": 4, "bon": 4, "controls": 4, "gsm8k": 4, "petri": 4,
                      "misalign": 4, "pair": 7, "crescendo": 14, "actorattack": 17}
    for leg in lp.STANDUP_LEGS:
        runs = [a for r in draft["standups"] if r["leg"] == leg and r["shard"] == 0 for a in r["arm_runs"]]
        assert len(runs) == 27, leg
        arms = {(a["arm"], a["s"]) for a in runs}
        assert {("none", s) for s in (1, 2, 3)} <= arms and {("ar_c0p3__arditi_allpos", s) for s in (1, 2, 3)} <= arms
        assert {("prompted__si_constitution", 1), ("prompted__safety_refusal", 3)} <= arms
    assert draft["eval_legs"] == 15 and draft["arm_runs_per_leg"] == 27


def test_validate_plan_admits_the_draft(draft):
    assert lp.validate_plan(draft) == []


def _mutate(draft, pick, fn):
    p = copy.deepcopy(draft)
    rows = [r for r in p["standups"] + p["rungs"] if pick(r)]
    assert rows
    fn(rows[0])
    return p


def test_known_bad_gsm8k_with_explicit_window_is_refused(draft):
    p = _mutate(draft, lambda r: r["leg"] == "gsm8k" and r["stage"] == "base", lambda r: r["argv"].extend(["--max-model-len", "16640"]))
    probs = lp.validate_plan(p)
    assert any("GSM8K standup passes --eval ONLY" in x and "--max-model-len '16640'" in x for x in probs), probs
    p = _mutate(draft, lambda r: r["leg"] == "gsm8k" and r["stage"] == "base", lambda r: r["argv"].extend(["--max-num-seqs", "31"]))
    assert any("GSM8K standup passes --eval ONLY" in x for x in lp.validate_plan(p))
    p = _mutate(draft, lambda r: r["leg"] == "gsm8k" and r["stage"] == "base", lambda r: r["argv"].remove("--thinking-on"))
    assert any("GSM8K needs --thinking-on" in x for x in lp.validate_plan(p))


def test_known_bad_mmlu_standup_without_max_logprobs_is_refused(draft):
    def strip(r):
        i = r["argv"].index("--max-logprobs")
        del r["argv"][i:i + 2]
    p = _mutate(draft, lambda r: r["leg"] == "controls" and r["stage"] == "base" and r["engine"] == "s1", strip)
    probs = lp.validate_plan(p)
    assert any("serves MMLU but carries no --max-logprobs 64" in x and "controls__s1__base__shard00" in x for x in probs), probs
    # a controls RUNG re-draws the ladder controls alone -- its engine serves no MMLU, so the flag is ABSENT by design
    # and a rung that carries it (or the `mmlu` eval leg) is refused by name; the opposite ("every controls rung serves MMLU too") does not hold.
    rungs = [r for r in draft["rungs"] if r["leg"] == "controls"]
    assert rungs and all("--max-logprobs" not in r["argv"] and "mmlu" not in r["eval_legs"] for r in rungs), \
        [(r["standup_id"], r["eval_legs"]) for r in rungs]
    p = _mutate(draft, lambda r: r["leg"] == "controls" and r["stage"] == "8192", lambda r: r["argv"].extend(["--max-logprobs", "64"]))
    assert any("a controls RUNG serves no MMLU" in x and "controls__s1__8192__shard00" in x for x in lp.validate_plan(p))
    p = _mutate(draft, lambda r: r["leg"] == "controls" and r["stage"] == "8192", lambda r: r["eval_legs"].append("mmlu"))
    assert any("a controls RUNG serves no MMLU" in x for x in lp.validate_plan(p))


def test_known_bad_table_window_with_factors_and_non_table_without_pair(draft):
    p = _mutate(draft, lambda r: r["leg"] == "pasttense" and r["stage"] == "base", lambda r: r["argv"].extend(["--max-num-seqs", "45"]))
    assert any("TABLE window passes no" in x for x in lp.validate_plan(p))
    def drop_pair(r):
        i = r["argv"].index("--max-model-len"); del r["argv"][i:i + 4]
    p = _mutate(draft, lambda r: r["leg"] == "pair" and r["stage"] == "base", drop_pair)
    assert any("NON-table window passes the identity PAIR" in x for x in lp.validate_plan(p))
    def bare_s(r):
        i = r["argv"].index("--max-num-seqs"); r["argv"][i + 1] = "21"      # the KERNEL S typed instead of the identity S
    p = _mutate(draft, lambda r: r["leg"] == "pair" and r["stage"] == "base", bare_s)
    assert any("!= identity pair" in x for x in lp.validate_plan(p))


def test_known_bad_route_set_and_arditi_single_object(draft):
    def drop_prompted(r):
        i = r["argv"].index("prompted__safety_refusal"); del r["argv"][i - 1:i + 1]
    p = _mutate(draft, lambda r: r["leg"] == "bon" and r["engine"] == "s2", drop_prompted)
    probs = lp.validate_plan(p)
    assert any("does not select the expected route set" in x for x in probs) and any("requests both prompted routes" in x for x in probs)
    def second_object(r):
        r["argv"].extend(["--cell", "constv2-mix-full__psr_ce_all"])
    p = _mutate(draft, lambda r: r["leg"] == "bon" and r["engine"] == "arditi", second_object)
    assert any("the Arditi engine serves ONE object" in x for x in lp.validate_plan(p))
    def wrong_pin(r):
        i = [k for k, a in enumerate(r["argv"]) if a.startswith("constv2-mix-full__psr_ce_all=")][0]
        r["argv"][i] = "constv2-mix-full__psr_ce_all=" + "f" * 64
    p = _mutate(draft, lambda r: r["leg"] == "bon" and r["engine"] == "s1", wrong_pin)
    assert any("pin for constv2-mix-full__psr_ce_all disagrees" in x for x in lp.validate_plan(p))


def test_capacity_ledger_counts_containers_and_gpus_separately(draft):
    led = draft["capacity_ledger"]
    assert led["targets"] == {"containers": 74, "gpus": 74}
    assert led["pools"] == {"containers": 4, "gpus": 4}                      # one H200 per pool
    assert all((p["gpu"], p["gpus"]) == ("H200", 1) for p in draft["pools"])
    assert led["rungs_declared"] == {"functions": 56, "containers": 56, "gpus": 56}
    assert led["rungs_deferred"] == {"functions": 0, "rows": [], "why": None} and led["gsm8k_rungs"]["functions"] == 4 and "executable" in led["gsm8k_rungs"]["rule"]
    assert led["targets_plus_pools"] == {"containers": 78, "gpus": 78}
    assert led["unstaged_peak"] == {"containers": 134, "gpus": 134}
    assert led["by_leg"]["gsm8k"] == {"target_containers": 4, "rung_containers": 4}
    assert led["by_leg"]["pasttense"] == {"target_containers": 4, "rung_containers": 8}
    assert led["by_leg"]["controls"] == {"target_containers": 4, "rung_containers": 12}
    assert led["by_leg"]["crescendo"] == {"target_containers": 14, "rung_containers": 0}


def test_gate8_s5_roster_and_gap(draft):
    refs = [{"L": r["L"], "S": r["S"], "thinking": r["thinking"], "tok_s": 1000.0} for r in draft["s5_roster"]]
    assert len(refs) == 10 and lp.s5_gap(draft, refs) == []                     # the GSM8K rung's 24,832-ON window is the tenth
    missing = [r for r in refs if not r["thinking"] or r["L"] != 16640]
    gap = lp.s5_gap(draft, missing)
    assert [(g["L"], g["S"], g["thinking"]) for g in gap] == [(16640, 31, True)]
    assert "gsm8k:base" in gap[0]["used_by"]


def test_gate10_adapter_set_is_derived_17_and_hashes_fill_from_the_smokes(inputs, draft):
    ids = [a["staged_id"] for a in draft["adapter_hash_set"]]
    assert len(ids) == 17 and "none__lora0" in ids and "ar_c0p3__arditi_allpos" in ids
    assert sum(1 for i in ids if i.startswith("base64s")) == 12 and sum(1 for i in ids if i.startswith("derived_base64s")) == 3
    assert all(a["intervention_hash"] is None for a in draft["adapter_hash_set"])
    gap = lp.adapter_gap(draft, [])
    assert gap["ok"] is False and len(gap["missing_hash"]) == 17
    hashes = {sid: f"ih_{sid[:8]}" for sid in ids}
    final = _plan(inputs, materialisation="final", s4=lp.S4_DEFAULTS, adapter_hashes=hashes)
    assert all(a["intervention_hash"] == f"ih_{a['staged_id'][:8]}" for a in final["adapter_hash_set"])
    assert lp.adapter_gap(final, hashes.values())["ok"] is True
    partial = lp.adapter_gap(final, list(hashes.values())[:16])
    assert partial["ok"] is False and len(partial["not_banked"]) == 1
    for r in final["standups"]:
        for cell, a in r["adapters"].items():
            assert a["expected_reused"] is True and a["intervention_hash"]


def test_draft_equals_final_except_the_s4_fields(inputs, draft):
    final = _plan(inputs, materialisation="final",
                  s4={"multiturn_window": {"crescendo": 16384, "actorattack": 24576},
                      "pools": {"count": 5, "assignment": ["crescendo", "crescendo", "actorattack", "actorattack", "pair"], "s_att": 12},
                      "actorattack": {"forwards_per_goal": 26, "containers_per_view": 3, "containers_arditi": 1}})
    assert lp.compare_draft_final(draft, final) == []
    cr = [r for r in final["standups"] if r["leg"] == "crescendo" and r["engine"] == "s1"][0]
    assert cr["geometry"]["L"] == 16384 and cr["geometry"]["S"] == 31
    aa = [r for r in final["standups"] if r["leg"] == "actorattack"]
    assert len(aa) == 3 * 3 + 1 and aa[0]["geometry"]["S"] == 21
    assert len(final["pools"]) == 5 and final["capacity_ledger"]["pools"] == {"containers": 5, "gpus": 5}
    # a NON-S4 diff is a generator bug: a moved pin on a static row
    bad = copy.deepcopy(final)
    row = [r for r in bad["standups"] if r["leg"] == "pasttense" and r["engine"] == "s1"][0]
    row["objects"]["constv2-mix-full__psr_ce_all"]["object_sha256"] = "e" * 64
    diffs = lp.compare_draft_final(draft, bad)
    assert diffs and any("object_sha256" in d for d in diffs)


def test_draft_with_non_default_s4_fields_refuses(inputs):
    with pytest.raises(ValueError, match="DRAFT plan carries the S4 fields at their spec defaults"):
        _plan(inputs, s4={"multiturn_window": {"crescendo": 16384}})
    with pytest.raises(ValueError, match="unknown S4 field"):
        _plan(inputs, materialisation="final", s4={"made_up": 1})
    with pytest.raises(ValueError, match="never fewer than one per attack"):
        _plan(inputs, materialisation="final", s4={"pools": {"count": 2, "assignment": ["crescendo", "crescendo"]}})


def test_children_carry_ws1_flags_and_one_identity_file_each(draft):
    def row(leg, engine="s2", stage="base"):
        return [r for r in draft["standups"] + draft["rungs"] if r["leg"] == leg and r["engine"] == engine and r["stage"] == stage][0]
    bon = row("bon")
    assert len(bon["children"]) == 8
    k = [c for c in bon["children"] if c["arm"] == "none"][0]
    a = k["argv"]
    assert a[a.index("--bon-decode-preset") + 1] == "vendor" and a[a.index("--model-key") + 1] == "qwen35_4b"
    assert a[a.index("--decode-seed-policy") + 1] == "replicate_v1" and a[a.index("--decode-seed-base") + 1] == "2"
    assert a[a.index("--require-engine") + 1] == "lora_native" and k["registry_identity_rel"] == "registry/plans/bon/none__s2.json"
    ar = row("bon", "arditi")
    assert [c["s"] for c in ar["children"]] == [1, 2, 3] and all(c["arm"] == "ar_c0p3__arditi_allpos" for c in ar["children"])
    assert ar["children"][2]["argv"][ar["children"][2]["argv"].index("--decode-seed-base") + 1] == "3"
    assert ar["children"][2]["argv"][3].endswith("configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep_s3.yaml"), "the per-seed WAVE config (WS5-I review round 3, finding 3), never the _sampled_t0 static run"
    mt = row("crescendo")
    a = mt["children"][0]["argv"]
    assert a[a.index("--multiturn-target-seed-base") + 1] == "2" and a[a.index("--multiturn-target-decode") + 1] == "vendor"
    assert a[a.index("--target-max-model-len") + 1] == "49152" and a[a.index("--target-max-new-tokens") + 1] == "4096"
    assert a[a.index("--attacker-endpoint") + 1].startswith("{pool_url:seedrep-attacker-")
    mis = row("misalign")
    assert len(mis["children"]) == 16                                     # 8 arms x 2 behaviours, one child per cell
    a = mis["children"][0]["argv"]
    assert a[a.index("--seed-policy") + 1] == "per_response" and a[a.index("--seed-base") + 1] == "2" and a[a.index("--sampler") + 1] == "vendor"
    # cells.json speaks HARNESS ids (the standup's enumeration), so the selection is spelled in that namespace;
    # one out root per (seed base, behaviour) with the cell dir retained
    assert a[a.index("--only-cells") + 1] == lp.harness_id(mis["children"][0]["cell"]) and mis["children"][0]["env"] == {"CUDA_VISIBLE_DEVICES": ""}
    assert a[a.index("--out") + 1] == "{attempt_root}/exp/misalign/agentic__s2__leaking"
    g8 = row("gsm8k")
    a = g8["children"][3]["argv"]
    assert a[a.index("--thinking") + 1] == "on" and a[a.index("--sampling") + 1] == "thinking_card"
    assert a[a.index("--inspect-seed-policy") + 1] == "replicate" and a[a.index("--inspect-seed-base") + 1] == "2"
    # the g8k_driver_argv shape -- identity v2, the cap stated AND set, one cell per out root, the budget artifact; the pre-fix
    # --gsm8k-arm / --standup-endpoint / --controls-cap (which the driver's parser does not know) are gone
    assert a[a.index("--tinycap-identity-version") + 1] == "2" and a[a.index("--max-tokens-by-task") + 1] == a[a.index("--on-cap-by-task") + 1] == "gsm8k_paired200=8192"
    assert a[a.index("--samples") + 1] == "1" and a[a.index("--out-root") + 1] == "{attempt_root}/exp/sampled/gsm8k_thinking_on_cap8192__s2/" + g8["children"][3]["cell"]
    assert not {"--gsm8k-arm", "--standup-endpoint", "--controls-cap", "--item-subset"} & set(a) and a[a.index("--oos-prompt-budget") + 1].endswith("/oos_prompt_budget.json")
    g8r = row("gsm8k", "s2", "16384")
    a = g8r["children"][3]["argv"]
    assert a[a.index("--item-subset") + 1] == "{attempt_root}/exp/rung_subsets/gsm8k_cap16384__s2/" + g8r["children"][3]["cell"] + "/rung_subset.json"
    assert a[a.index("--on-cap-by-task") + 1] == "gsm8k_paired200=16384" and g8r["children"][3]["source_out_rel"] == "exp/sampled/gsm8k_thinking_on_cap8192__s2/" + g8r["children"][3]["cell"]
    ctl = row("controls")
    a = ctl["children"][0]["argv"]
    assert a[a.index("--top-logprobs") + 1] == "64" and a[a.index("--controls") + 1] == "xstest,or_bench_hard_1k,mmlu"
    assert ctl["children"][0]["task"] == {"xstest": "judged/xstest_v2", "or_bench_hard_1k": "judged/or_bench_hard_1k", "mmlu": "s2parity/mmlu_letter"}
    st = row("renellm", "s3")
    a = st["children"][0]["argv"]
    assert a[a.index("--attack") + 1] == "renellm" and a[a.index("--target-engine") + 1] == "served" and "--confirmatory" in a
    assert "sdfbase_full22_qwen35_4b_s3_sampled_t0.yaml" in a[a.index("--config") + 1]
    # static fan-out = S ONE-REQUEST-EACH drivers: 8 arms x ceil(45 / 8) = 6 goal shards -> 48 children at S 45 (the sbatch's
    # per-(family, cell, shard) shape); the Arditi engine 3 seed-runs x 15 = 45; the 8,192 rung at S 31 -> 4 shards per arm
    assert len(st["children"]) == 48 and lp.static_nshards(45, 8) == 6 and lp.static_nshards(45, 3) == 15
    assert {(c["arm"], c["goal_shard"]) for c in st["children"]} == {(arm, k) for arm in {c["arm"] for c in st["children"]} for k in range(6)}
    assert all(c["argv"][c["argv"].index("--nshards") + 1] == "6" for c in st["children"])
    assert len(row("renellm", "arditi")["children"]) == 45
    esc = row("renellm", "s3", "8192")
    assert "sdfbase_full22_qwen35_4b_s3_sampled_t0_cap8192.yaml" in esc["children"][0]["argv"][esc["children"][0]["argv"].index("--config") + 1]
    assert len(esc["children"]) == 8 * 4 and esc["children"][0]["argv"][esc["children"][0]["argv"].index("--nshards") + 1] == "4"
    petri = row("petri")
    assert all(c["argv"] is None and c["grading_host_drive"]["target_decode"] == "vendor" for c in petri["children"])
    assert petri["children"][0]["grading_host_drive"]["petri_eval_max_connections"] == 3
    assert row("petri", "arditi")["children"][0]["grading_host_drive"]["petri_eval_max_connections"] == 4
    rels = {c["registry_identity_rel"] for r in draft["standups"] for c in r["children"]}
    assert all(x.startswith("registry/plans/") and x.endswith(".json") for x in rels)


def test_queues_declare_producers_and_immediate_ancestry(draft):
    q = {(x["leg"], x["engine"], x["cap_next"]): x for x in draft["queues"]}
    assert set(q) >= {("pasttense", "s1", "8192"), ("pasttense", "s1", "16384"), ("controls", "arditi", "2048"),
                      ("controls", "arditi", "8192"), ("controls", "arditi", "16384"), ("gsm8k", "s2", "16384"),
                      ("misalign", "s3", "16384"), ("petri", "s1", "petri-rerun")}
    assert q[("controls", "s1", "2048")]["feeder_stage"] == "base" and q[("controls", "s1", "8192")]["feeder_stage"] == "2048"
    assert q[("controls", "s1", "16384")]["feeder_stage"] == "8192" and q[("controls", "s1", "16384")]["ceiling"] is True
    assert [p["standup_id"] for p in q[("pasttense", "s1", "8192")]["producers"]] == ["pasttense__s1__base__shard00"]
    assert [p["standup_id"] for p in q[("pasttense", "s1", "16384")]["producers"]] == ["pasttense__s1__8192__shard00"]
    assert q[("pasttense", "s1", "8192")]["consumer"] == "pasttense__s1__8192__shard00"
    assert not any(x["leg"] in lp.MULTITURN_LEGS or x["leg"] == "bon" for x in draft["queues"]), "multi-turn / BoN have no ladder"


def test_geometry_numbers_per_leg(draft):
    g = {(r["leg"], r["stage"]): r["geometry"] for r in draft["standups"] + draft["rungs"] if r["engine"] == "s1"}
    assert (g[("pasttense", "base")]["L"], g[("pasttense", "base")]["S"], g[("pasttense", "base")]["identity_S"]) == (11520, 45, 64)
    assert (g[("controls", "base")]["L"], g[("controls", "base")]["S"]) == (7424, 70)
    assert (g[("controls", "2048")]["L"], g[("controls", "2048")]["S"]) == (10240, 51)
    assert (g[("gsm8k", "base")]["L"], g[("gsm8k", "base")]["S"], g[("gsm8k", "base")]["identity_S"]) == (16640, 31, 44)
    assert (g[("gsm8k", "16384")]["L"], g[("gsm8k", "16384")]["S"], g[("gsm8k", "16384")]["thinking"]) == (24832, 21, True)   # executable
    assert (g[("petri", "base")]["L"], g[("petri", "base")]["S"]) == (32768, 15)
    assert (g[("petri", "petri-rerun")]["L"], g[("petri", "petri-rerun")]["S"]) == (49152, 10)
    assert (g[("misalign", "base")]["L"], g[("misalign", "base")]["S"]) == (24576, 21)
    assert (g[("misalign", "16384")]["L"], g[("misalign", "16384")]["S"]) == (40960, 12)
    assert (g[("pair", "base")]["L"], g[("pair", "base")]["S"]) == (24576, 21)
    assert (g[("crescendo", "base")]["L"], g[("crescendo", "base")]["S"]) == (49152, 10)
    for key, gg in g.items():
        assert gg["expected_window_source"] == ("table" if gg["table"] else "override")
        assert (gg["expected_adjustment"] is None) == (gg["identity_S"] * gg["L"] <= 524_287)
    assert g[("gsm8k", "base")]["expected_adjustment"] == {"kind": "lora_kernel_index_ceiling", "from_S": 44, "to_S": 31, "D": 4096, "ceiling": 524287}


def test_cli_mints_wave_writes_draft_and_holds_final_to_draft(tmp_path, inputs):
    rs = tmp_path / "rs"
    wave = rs / "wave.json"
    rc = lp.main(["--repo", str(REPO), "--w", W, "--wave-json", str(wave), "--mint-wave", "--materialisation", "draft",
                  "--out", str(rs / "launch_plan_draft.json"), "--ledger", str(rs / "capacity_ledger.json")])
    assert rc == 0 and wave.is_file() and (rs / "launch_plan_draft.json").is_file()
    w = json.loads(wave.read_text())
    assert lp.check_wave_uuid(w["wave_uuid"]) == w["wave_uuid"] and len(w["wave_uuid"]) == 36 and w["W"] == W
    draft = json.loads((rs / "launch_plan_draft.json").read_text())
    assert draft["wave_uuid"] == w["wave_uuid"] and json.loads((rs / "capacity_ledger.json").read_text())["unstaged_peak"]["containers"] == 134
    # a final without decisions refuses
    with pytest.raises(SystemExit, match="needs --decisions"):
        lp.main(["--repo", str(REPO), "--w", W, "--wave-json", str(wave), "--materialisation", "final",
                 "--out", str(rs / "launch_plan.json"), "--ledger", str(rs / "capacity_ledger.json")])
    dec = tmp_path / "decisions.json"
    dec.write_text(json.dumps({"multiturn_window": {"crescendo": 16384, "actorattack": 16384}}))
    # a final without S5 references refuses (the multi-turn windows must be justified by measurements)
    with pytest.raises(SystemExit, match="needs --s5-references"):
        lp.main(["--repo", str(REPO), "--w", W, "--wave-json", str(wave), "--materialisation", "final", "--decisions", str(dec),
                 "--out", str(rs / "launch_plan.json"), "--ledger", str(rs / "capacity_ledger.json")])
    wu = w["wave_uuid"]
    smoke_wave = "11111111-2222-4333-8444-555555555555"                 # the smoke's EXECUTION wave: the records were measured there

    def s5_block(refs, mt_):
        return {"S5": {"W": W, "execution_wave_uuid": smoke_wave, "production_wave_uuid": wu, "launch_plan_sha": draft["launch_plan_sha"], "references": refs, "multiturn": mt_}}

    def m(leg, arm, kind, ctx, objs, **over):
        rec = {"schema": lp.S5_WINDOW_MEASUREMENT_SCHEMA, "window_source": lp.S5_WINDOW_SOURCE, "leg": leg, "arm": arm, "arm_kind": kind, "model_key": "qwen35_4b",
               "decode": "vendor", "decode_policy_ref": lp.ARDITI_DECODE_POLICY_REF, "objects_measured": objs, "store_path": f"/vol/x/{leg}/{arm}", "n_goals_completed": 20,
               "n_goals_expected": 20, "store_complete": True, "context_max": ctx, "clamped_calls": 0, "cap": 4096, "measured_utc": "2026-09-18T00:00:00Z",
               "wave_uuid": smoke_wave, "attempt_id": "ta-1"}
        rec.update(over)
        return rec
    probe = [c["staged_id"] for c in lp.S5_PROBE_CELLS]
    mt = []
    for leg in ("crescendo", "actorattack"):
        mt += [m(leg, "none", "parent", 8597, ["none"]), m(leg, probe[0], "dosed", 12000, [probe[0]]), m(leg, probe[1], "dosed", 7000, [probe[1]])]
    smokes = tmp_path / "smokes.json"
    refs_all = None
    # first pass: the S5 roster of THIS final (16,384 windows) -- every window referenced, the measurements justify 16,384
    tmp_final = _plan(inputs, materialisation="final", s4={"multiturn_window": {"crescendo": 16384, "actorattack": 16384}})
    refs_all = [{"L": r["L"], "S": r["S"], "thinking": r["thinking"]} for r in tmp_final["s5_roster"]]
    # the block WITHOUT its wrapper (the pre-fix smokes.json) refuses by name; a wrapper bound to another production wave refuses
    smokes.write_text(json.dumps({"S5": {"references": refs_all, "multiturn": mt}}))
    assert lp.main(["--repo", str(REPO), "--w", W, "--wave-json", str(wave), "--materialisation", "final", "--decisions", str(dec), "--out", str(rs / "n1.json"),
                    "--ledger", str(rs / "n1l.json"), "--compare-draft", str(rs / "launch_plan_draft.json"), "--s5-references", str(smokes)]) == 2
    foreign = s5_block(refs_all, mt); foreign["S5"]["production_wave_uuid"] = "33333333-4444-4555-8666-777777777777"
    smokes.write_text(json.dumps(foreign))
    assert lp.main(["--repo", str(REPO), "--w", W, "--wave-json", str(wave), "--materialisation", "final", "--decisions", str(dec), "--out", str(rs / "n2.json"),
                    "--ledger", str(rs / "n2l.json"), "--compare-draft", str(rs / "launch_plan_draft.json"), "--s5-references", str(smokes)]) == 2
    smokes.write_text(json.dumps(s5_block(refs_all, mt)))
    rc = lp.main(["--repo", str(REPO), "--w", W, "--wave-json", str(wave), "--materialisation", "final", "--decisions", str(dec),
                  "--out", str(rs / "launch_plan.json"), "--ledger", str(rs / "capacity_ledger.json"),
                  "--compare-draft", str(rs / "launch_plan_draft.json"), "--s5-references", str(smokes)])
    assert rc == 0
    final = json.loads((rs / "launch_plan.json").read_text())
    assert final["wave_uuid"] == draft["wave_uuid"] and final["materialisation"] == "final"
    # the serialised sha is the sha OF the serialised plan (the S5 records were inserted after build_plan hashed it)
    assert final["launch_plan_sha"] == lp.plan_sha(final) and final["multiturn_window_provenance"]["crescendo"]["s5_measurements"]
    unjustified = lp.build_plan(W=W, wave_uuid=wu, reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"], pins=inputs["pins"], arditi_pins=inputs["arditi_pins"],
                                arms=inputs["arms"], materialisation="final", s4=json.loads(dec.read_text()), generated_at=final["generated_at"])
    assert unjustified["launch_plan_sha"] != final["launch_plan_sha"], "the repro: build_plan's sha predates the justification"
    prov = final["multiturn_window_provenance"]["actorattack"]
    assert prov["gate"]["ok"] and len(prov["s5_measurements"]) == 3 and prov["s5_measurements"][0]["window_source"] == "measured_max_complete_store"
    just = json.loads((rs / "window_justification.json").read_text())
    assert just["legs"]["crescendo"]["gate"]["rule_would_select"]["L"] == 16384 and just["problems"] == []
    assert just["execution_wave_uuid"] == smoke_wave and just["production_wave_uuid"] == wu
    # gate 8 through the CLI: missing 11,520 refuses
    refs = [r for r in refs_all if r["L"] != 11520]
    smokes.write_text(json.dumps(s5_block(refs, mt)))
    rc = lp.main(["--repo", str(REPO), "--w", W, "--wave-json", str(wave), "--materialisation", "final", "--decisions", str(dec),
                  "--out", str(rs / "x.json"), "--ledger", str(rs / "y.json"), "--s5-references", str(smokes)])
    assert rc == 2
    # a window below a reading, a clamped call, an incomplete store, a missing probe cell, a percentile source -- each refuses the FINAL
    for bad_mt in ([m("crescendo", "none", "parent", 20000, ["none"]), m("crescendo", probe[0], "dosed", 12000, [probe[0]]), m("crescendo", probe[1], "dosed", 7000, [probe[1]])] + mt[3:],
                   [m("crescendo", "none", "parent", 8597, ["none"], clamped_calls=2)] + mt[1:],
                   [m("crescendo", "none", "parent", 8597, ["none"], n_goals_completed=19, store_complete=False)] + mt[1:],
                   mt[:2] + mt[3:],
                   [m("crescendo", "none", "parent", 8597, ["none"], window_source="p99")] + mt[1:]):
        smokes.write_text(json.dumps({"S5": {"references": refs_all, "multiturn": bad_mt}}))
        rc = lp.main(["--repo", str(REPO), "--w", W, "--wave-json", str(wave), "--materialisation", "final", "--decisions", str(dec),
                      "--out", str(rs / "x.json"), "--ledger", str(rs / "y.json"), "--s5-references", str(smokes)])
        assert rc == 2
    # a record that says complete while a goal is missing is a contradiction the validator names
    probs = lp.window_measurement_problems(m("crescendo", "none", "parent", 8597, ["none"], n_goals_completed=19))
    assert any("FLOOR" in p for p in probs)
    assert lp.window_measurement_problems(mt[0]) == []
    assert lp.window_measurement_problems({k: v for k, v in mt[0].items() if k != "attempt_id"}) == ["missing attempt_id"]
