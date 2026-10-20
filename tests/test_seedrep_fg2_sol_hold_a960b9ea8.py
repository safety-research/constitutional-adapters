"""qwen-seedrep-native-battery fg2 (two P2s).

P2-1 -- `audit_common._tinycap_filed_vs_declared` (fg1) held the filed rows to the outcomes record but no longer held the SAMPLE AXIS: with
samples_per_item = 2 a store whose writer attempted only sample 0 reads COMPLETE when sample 1 is absent from BOTH the rows and the outcomes
(the header reports one attempted sample, the marker n_items 1 / n_planned 1, the filed-row count matches). Fix: the outcome keys must equal
declared_item_ids x range(samples_per_item) independently of scored / unscored filing -- a missing OUTCOME refuses by (item, sample); the
historical exemption stays for missing ROWS backed by unscored outcomes only.

P2-2 -- `seedrep_launch_plan._determinism_source` (fg1) handed the SMOKE-WIDE `checks["resolved_sampler"]` to `verified_resolved_decode`, a
helper written for ONE row's check, which reads every source without ownership: a same-tree GSM8K at 0.6 with a 2/20 replay qualified alone but
REFUSED once a greedy row joined the smoke (the combined temperatures held a 0.0), and a prior-tree GSM8K at 0.6 qualified alone but refused
beside another sampled row at 0.7 (the combined fresh [0.6, 0.7] against GSM8K's historical [0.6]). Fix: `_row_sampler_check` selects the CURRENT
row's exact sampler sources -- under its re-audit evidence root (fresh, the authority) on the prior path, under its attempt root (historical) on
the same-tree path -- on both derivation paths and both certify-time paths; each row is qualified on its own sources.

Synthetic throughout: the certificate suite's fake smoke world, placeholder shas, no recorded data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import seedrep_carry_over as co  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
import test_seedrep_review_fixes as RF  # noqa: E402
from sisteer.capeval.tiers import inspect_served as isv  # noqa: E402
from sisteer.capeval.tinycap import completion  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from test_seedrep_review_fixes import draft, inputs, srv  # noqa: E402,F401  (module fixtures register by import)
from test_seedrep_fc11_prior_smoke_pass import W25, _smoke_spell_cells, _stub_reaudit, world  # noqa: E402,F401  (module fixture)
from test_seedrep_fg1_fau_store_completeness import GSM8K_TASK, S, _result  # noqa: E402
from test_seedrep_fg1_faw_d180_smoke_pass import MISSES, NA, THINKING_CARD, _rebind  # noqa: E402


# =============================================================================================== P2-1: the sample axis is held whole
def _k2_cell(tmp_path: Path, *, items=("gsm8k_0001",), samples=(0,), k=2) -> Path:
    """A gsm8k cell in the writer's real shape at samples_per_item = k whose writer attempted only ``samples`` of every item: the outcomes header,
    the rows (acc@<i> per scored sample + the per-item acc mean), and the marker all agree WITH THEMSELVES (n_items == n_planned == the items)."""
    d = tmp_path / "cells" / "none"
    d.mkdir(parents=True, exist_ok=True)
    results = [_result(it, s_, scored=True, value=1.0) for it in items for s_ in samples]
    plan = isv.TaskPlan(row_task=GSM8K_TASK, name=C.GSM8K_PAIRED_TASK_NAME, task=object(), expected_n=len(items), samples_per_item=k, max_tokens=8192,
                        item_set=C.GSM8K_ITEM_SET_PAIRED, think_split_mode=C.THINK_SPLIT_IMPLICIT_OPEN)
    n = len(results)
    summary = {"n_scored": len(items), "n_unscored": 0, "n_scored_samples": n, "n_unscored_samples": 0, "n_stop_reason": n, "n_at_cap": 0}
    rec = isv.outcome_record(results, plan=plan, task=GSM8K_TASK, thinking=C.THINKING_ON, summary=summary, split_counts=isv.think_split_counts(results, plan=plan),
                             producer_tree_sha="0" * 40)
    completion.write_task_outcomes(d, GSM8K_TASK, rec)
    rows = [{"task": GSM8K_TASK, "item_id": r.item_id, "metric": C.sample_metric(C.METRIC_ACC, r.sample_idx), "value": 1.0} for r in results]
    rows += [{"task": GSM8K_TASK, "item_id": it, "metric": C.METRIC_ACC, "value": 1.0} for it in items]
    rows += [{"task": GSM8K_TASK, "item_id": C.AGG_ITEM_ID, "metric": C.METRIC_ACC_AGGREGATE, "value": 1.0}, {"task": GSM8K_TASK, "item_id": C.AGG_ITEM_ID, "metric": C.METRIC_N, "value": float(len(items))}]
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "manifest.json").write_text(json.dumps({"cell_id": "none", "seeds": {"inspect": {"replicate": {"policy": C.INSPECT_SEED_POLICY_REPLICATE, "seed_base": S}}}}))
    completion.write_done_marker(d, GSM8K_TASK, {"task": GSM8K_TASK, "n_items": len(items), "n_rows": n + len(items), "n_requests": n, "n_rejected": 0,
                                                 "wall_clock_s": 1.0, "finished_at": 1.0, C.DONE_MARKER_N_PLANNED_FIELD: len(items)})
    return d


def test_p2_1_a_k2_store_missing_sample_1_from_rows_and_outcomes_refuses_by_item_and_sample(tmp_path):
    # the reproduction: one item, samples_per_item 2, only sample 0 anywhere -- header n_attempted_samples 1, marker n_items 1 == n_planned 1, rows match
    out = completion.read_task_outcomes(_k2_cell(tmp_path / "one"), GSM8K_TASK)
    assert (out["samples_per_item"], out["n_attempted_samples"], out["n_attempted_items"]) == (2, 1, 1), "the record agrees with itself: nothing but the axis rule sees the hole"
    sr = ac.read_tinycap_store(tmp_path / "one" / "cells" / "none", task=GSM8K_TASK, leg="gsm8k", arm="none")
    assert not sr.complete, sr.completeness_note
    assert "1 declared (item_id, sample_idx) outcome record(s) absent from the outcomes record" in sr.completeness_note
    assert "1 item(s) x samples_per_item 2" in sr.completeness_note and "(first: [('gsm8k_0001', 1)])" in sr.completeness_note
    # two items, one missing sample each -> both named; the whole axis present -> complete
    sr2 = ac.read_tinycap_store(_k2_cell(tmp_path / "two", items=("gsm8k_0001", "gsm8k_0002")), task=GSM8K_TASK, leg="gsm8k", arm="none")
    assert not sr2.complete and "2 declared (item_id, sample_idx) outcome record(s) absent" in sr2.completeness_note
    assert "[('gsm8k_0001', 1), ('gsm8k_0002', 1)]" in sr2.completeness_note
    sr3 = ac.read_tinycap_store(_k2_cell(tmp_path / "whole", samples=(0, 1)), task=GSM8K_TASK, leg="gsm8k", arm="none")
    assert sr3.complete, sr3.completeness_note
    assert len(sr3.draws) == 2 and sr3.filed == {(GSM8K_TASK, "gsm8k_0001")}


def test_p2_1_the_historical_exemption_covers_missing_rows_behind_unscored_outcomes_only(tmp_path):
    """The pre-F-AU writer's shape at k = 2: sample 1 of the item is UNSCORED and present in the outcomes record but files no row -- exempt (a missing
    ROW backed by an unscored outcome). The same store with that outcome record DELETED is a missing OUTCOME -- refused by (item, sample), whatever the
    header then says about itself."""
    d = _k2_cell(tmp_path / "unscored_row_missing", samples=(0, 1))
    out = json.loads(completion.task_outcomes_path(d, GSM8K_TASK).read_text())
    for it in out["items"]:
        if it["sample_idx"] == 1:
            it.update(scored=False, value=None, at_cap=True, stop_reason=isv.STOP_REASON_AT_CAP)
    out.update(n_scored_samples=1, n_unscored_samples=1, n_at_cap=1)
    completion.task_outcomes_path(d, GSM8K_TASK).write_text(json.dumps(out))
    rows = [json.loads(x) for x in (d / "rows.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["metric"] != C.sample_metric(C.METRIC_ACC, 1)]                       # the pre-F-AU writer files no acc@1 for the unscored draw
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    dm = json.loads(completion.done_marker_path(d, GSM8K_TASK).read_text()); dm["n_rows"] = 2
    completion.done_marker_path(d, GSM8K_TASK).write_text(json.dumps(dm))
    sr = ac.read_tinycap_store(d, task=GSM8K_TASK, leg="gsm8k", arm="none")
    assert sr.complete, sr.completeness_note
    assert "1 unscored outcome(s) without a row" in " | ".join(sr.notes) and len(sr.draws) == 2
    # now the unscored OUTCOME itself goes missing (the header re-agreeing with itself): a hole in the axis, refused by (item, sample)
    out["items"] = [it for it in out["items"] if it["sample_idx"] == 0]
    out.update(n_attempted_samples=1, n_unscored_samples=0, n_at_cap=0, n_stop_reason=1)
    completion.task_outcomes_path(d, GSM8K_TASK).write_text(json.dumps(out))
    sr2 = ac.read_tinycap_store(d, task=GSM8K_TASK, leg="gsm8k", arm="none")
    assert not sr2.complete and "(first: [('gsm8k_0001', 1)])" in sr2.completeness_note and "absent from the outcomes record" in sr2.completeness_note


# =============================================================================================== P2-2: each row qualified on its OWN sampler sources
def _world_rows(draft, tmp_path, legs):
    """A same-tree S3 world with one accepted row per leg (the certificate suite's `_smoke_world`), each row's determinism record and
    resolved-sampler record addressable per leg."""
    T = RF.TestSolR3_2_launch_certificate()
    rs = tmp_path / "RS"
    wave = RF._uuid4(7)
    wp = lp.smoke_wave_path(rs, "S3")
    wp.parent.mkdir(parents=True, exist_ok=True)
    wp.write_text(json.dumps({"smoke_id": "S3", "wave_uuid": wave}))
    rids = {leg: RF._row(draft, leg)["standup_id"] for leg in legs}
    accepted, texts = T._smoke_world(draft, draft, rs, {"S3": wave}, {"S3": [rids[leg] for leg in legs]})
    getter = lp.RecordingGetter(lp.smoke_identity(draft, "S3", wave)["dicts"]["accepted"], accepted["S3"].get)
    rows = {}
    for leg in legs:
        det_path = [p for p in texts if p.endswith(f"/audits/determinism/{leg}.json")][0]
        rows[leg] = dict(rid=rids[leg], det_path=det_path, root=det_path[: -len(f"/audits/determinism/{leg}.json")], leg=leg)
    return dict(rs=rs, texts=texts, getter=getter, wave=wave, rows=rows, read_text=lp.volume_only(lambda p_: texts.get(str(p_))))


def _diverge_row(texts, row, *, identical=2, n_requests=20):
    doc = {**json.loads(texts[row["det_path"]]), "n_items": 20, "n_requests": n_requests, "n": n_requests, "identical": identical, "reproduces": identical == n_requests,
           "misses": MISSES[:1] if identical < n_requests else []}
    _rebind(texts, row["root"], lp.DETERMINISM_EVIDENCE_KEY, f"audits/determinism/{row['leg']}.json", json.dumps(doc))


def _sampler_row(texts, root, temperatures):
    doc = {**json.loads(texts[f"{root}/{lp.RESOLVED_SAMPLER_REL}"]), "distinct_resolved": [{"sampler": {**THINKING_CARD, "temperature": t}, "n": 400} for t in temperatures]}
    _rebind(texts, root, lp.PRODUCER_EVIDENCE_KEY, lp.RESOLVED_SAMPLER_REL, json.dumps(doc))


def _derive_same(w, draft, legs):
    return lp.derive_smoke_pass(rs=w["rs"], plan=draft, smoke_id="S3", standup_ids=[w["rows"][leg]["rid"] for leg in legs], accepted_get=w["getter"],
                                read_text=w["read_text"], derived_utc="2026-09-20T00:00:00Z")


def _problems_same(w, draft, rec, *, read_text=True):
    return lp.smoke_pass_problems(rec, smoke_id="S3", plan=draft, execution_wave_uuid=w["wave"], accepted_get=w["getter"], read_text=(w["read_text"] if read_text else None))


def test_p2_2_same_tree_a_sampled_gsm8k_row_qualifies_beside_a_greedy_row_and_each_row_is_judged_on_its_own_decode(draft, tmp_path):
    legs = ["gsm8k", "bon"]
    w = _world_rows(draft, tmp_path, legs)
    g, b = w["rows"]["gsm8k"], w["rows"]["bon"]
    _diverge_row(w["texts"], g)                          # gsm8k: 2 of 20 under the thinking card
    _sampler_row(w["texts"], g["root"], [0.6])
    _sampler_row(w["texts"], b["root"], [0.0])           # bon: a GREEDY sibling in the same smoke, its replay reproducing
    rec = _derive_same(w, draft, legs)                   # a combined decode would hold bon's 0.0 here
    det = rec["checks"]["determinism"]
    by_path = {s_["path"]: s_ for s_ in det["sources"]}
    sg, sb = by_path[g["det_path"]], by_path[b["det_path"]]
    assert sg["criterion3"] == NA and sg["criterion3_decode"]["temperatures"] == [0.6] and sg["criterion3_decode"]["source"][0]["path"] == f"{g['root']}/{lp.RESOLVED_SAMPLER_REL}"
    assert "criterion3" not in sb and sb["reproduces"] is True
    assert det["criterion3"]["records"] == [g["det_path"]] and det["criterion3"]["decode"]["temperatures"] == [0.6]
    assert _problems_same(w, draft, rec) == [] and _problems_same(w, draft, rec, read_text=False) == []
    # the greedy sibling's OWN replay diverging still refuses, by ITS name -- the sampled row's decode never covers it
    _diverge_row(w["texts"], b, identical=3, n_requests=40)
    with pytest.raises(ValueError, match=r"determinism.*3 of 40 re-sent request\(s\) reproduced byte-identically .*criterion 3 requires every one") as ei:
        _derive_same(w, draft, legs)
    assert b["det_path"] in str(ei.value) and g["det_path"] not in str(ei.value)
    # two SAMPLED rows at different temperatures, both diverged: each qualified on its own record, the block sums them
    _sampler_row(w["texts"], b["root"], [0.7])
    rec2 = _derive_same(w, draft, legs)
    by_path = {s_["path"]: s_ for s_ in rec2["checks"]["determinism"]["sources"]}
    assert by_path[g["det_path"]]["criterion3_decode"]["temperatures"] == [0.6] and by_path[b["det_path"]]["criterion3_decode"]["temperatures"] == [0.7]
    c3 = rec2["checks"]["determinism"]["criterion3"]
    assert (c3["reproduced"], c3["resent"], sorted(c3["records"])) == (5, 60, sorted([g["det_path"], b["det_path"]]))
    assert _problems_same(w, draft, rec2) == [] and _problems_same(w, draft, rec2, read_text=False) == []
    # a row whose own sampler source is missing from the recorded check cannot borrow a sibling's decode at certify: refused by name
    bad = json.loads(json.dumps(rec2))
    bad["checks"]["resolved_sampler"]["sources"] = [s_ for s_ in bad["checks"]["resolved_sampler"]["sources"] if not s_["path"].startswith(g["root"] + "/")]
    probs = _problems_same(w, draft, bad)
    assert any("the D180 exemption cannot be judged" in p and "no verified resolved-sampler source" in p for p in probs), probs


@pytest.fixture
def prior2(world):
    """fc11's prior-tree world with a TWO-row S3 under the prior tree (bon + gsm8k), the prior smoke Dict naming both accepted attempts."""
    w = world
    c = w["c"]
    prior_plan = c["draft"]
    T = RF.TestSolR3_2_launch_certificate()
    wave = c["waves"]["S3"]
    rids = {leg: RF._row(prior_plan, leg)["standup_id"] for leg in ("bon", "gsm8k")}
    accepted2, texts2 = T._smoke_world(prior_plan, prior_plan, c["rs"], {"S3": wave}, {"S3": [rids["bon"], rids["gsm8k"]]})
    texts = dict(w["texts"])
    texts.update(texts2)
    ident = lp.smoke_identity(prior_plan, "S3", wave)
    roots = {}
    for leg, rid in rids.items():
        row = lp.plan_row_by_id(prior_plan, rid)
        akey = lp.accepted_key(ident["W7"], wave, row["leg"], row["engine"], row["stage"], row["shard"])
        roots[leg] = lp.attempt_root_for(ident["results_root"], leg=row["leg"], view=row["engine"], stage=row["stage"], shard=row["shard"], attempt_id=accepted2["S3"][akey], kind=row.get("kind"))
    _smoke_spell_cells(texts, "S3", {rid: {"attempt_root": roots[leg]} for leg, rid in rids.items()})
    by_dict = dict(w["by_dict"])
    by_dict[ident["dicts"]["accepted"]] = accepted2["S3"]
    return dict(w=w, texts=texts, rids=rids, roots=roots, read_text=lp.volume_only(lambda p_: texts.get(str(p_))), for_dict=lambda name: by_dict.get(name, {}).get)


def _derive_prior(p2):
    w = p2["w"]
    return lp.derive_smoke_pass_from_priors(rs=w["rs25"], plan=w["draft25"], smoke_id="S3", standup_ids=list(p2["rids"].values()), priors=[w["prior"]],
                                            accepted_get_for_dict=p2["for_dict"], read_text=p2["read_text"], reaudit=_stub_reaudit(p2["texts"], W25),
                                            derived_utc="2026-09-20T00:00:00Z")


def _problems_prior(p2, rec, *, read_text=True):
    w = p2["w"]
    return lp.smoke_pass_problems(rec, smoke_id="S3", plan=w["final25"], execution_wave_uuid=lp.smoke_execution_wave(w["rs25"], "S3"), draft_launch_plan_sha=w["draft25"]["launch_plan_sha"],
                                  read_text=(p2["read_text"] if read_text else None), accepted_get_for_dict=p2["for_dict"])


def test_p2_2_prior_tree_two_sampled_rows_at_different_temperatures_are_each_qualified_on_their_own_rederived_decode(prior2):
    p2 = prior2
    texts, roots = p2["texts"], p2["roots"]
    for leg, t, ident, n in (("gsm8k", 0.6, 2, 20), ("bon", 0.7, 3, 40)):
        det_rel = f"audits/determinism/{leg}.json"
        doc = {**json.loads(texts[f"{roots[leg]}/{det_rel}"]), "n_items": 20, "n_requests": n, "n": n, "identical": ident, "reproduces": False, "misses": MISSES[:1]}
        _rebind(texts, roots[leg], lp.DETERMINISM_EVIDENCE_KEY, det_rel, json.dumps(doc))
        _sampler_row(texts, roots[leg], [t])
    rec = _derive_prior(p2)                              # fg1 REFUSED here: the combined fresh [0.6, 0.7] disagreed with each row's historical [t]
    det = rec["checks"]["determinism"]
    by_path = {s_["path"]: s_ for s_ in det["sources"]}
    for leg, t, ident, n in (("gsm8k", 0.6, 2, 20), ("bon", 0.7, 3, 40)):
        s_ = by_path[f"{roots[leg]}/audits/determinism/{leg}.json"]
        ev_root = rec["rows"][p2["rids"][leg]]["reaudit"]["evidence_root"]
        assert (s_["criterion3"], s_["identical"], s_["n_requests"]) == (NA, ident, n)
        assert s_["criterion3_decode"]["temperatures"] == [t] and s_["criterion3_decode"]["authority"] == co.DECODE_AUTHORITY_REAUDIT
        assert s_["criterion3_decode"]["source"] == [{"path": f"{ev_root}/{lp.RESOLVED_SAMPLER_REL}", "sha256": lp._sha_text(texts[f"{ev_root}/{lp.RESOLVED_SAMPLER_REL}"]),
                                                     "authority": co.DECODE_AUTHORITY_REAUDIT}], "THIS row's fresh record, never the sibling's"
    c3 = det["criterion3"]
    assert (c3["reproduced"], c3["resent"], sorted(c3["records"])) == (5, 60, sorted(f"{roots[leg]}/audits/determinism/{leg}.json" for leg in roots))
    assert sorted(c3["decode"]["temperatures"]) == [0.6, 0.7] and len(c3["decode"]["cross_check"]["historical"]) == 2
    assert _problems_prior(p2, rec) == []
    assert _problems_prior(p2, rec, read_text=None) == []
    # a greedy prior sibling (0.0, diverged) still refuses by ITS name; the sampled row's decode never covers it
    _sampler_row(texts, roots["bon"], [0.0])
    with pytest.raises(ValueError, match=r"3 of 40 re-sent request\(s\) reproduced byte-identically .*criterion 3 requires every one") as ei:
        _derive_prior(p2)
    assert f"{roots['bon']}/audits/determinism/bon.json" in str(ei.value)


def test_p2_2_row_sampler_check_selects_by_root_and_carries_the_authority():
    chk = {"ok": True, "sources": [{"path": "/vol/x/a/resolved_sampler.json", "sha256": "1" * 64}, {"path": "/vol/x/ab/resolved_sampler.json", "sha256": "2" * 64},
                                   {"path": "/vol/x/b/resolved_sampler.json", "sha256": "3" * 64}]}
    own = lp._row_sampler_check(chk, root="/vol/x/a", reaudit=False)
    assert [s_["path"] for s_ in own["sources"]] == ["/vol/x/a/resolved_sampler.json"] and "reaudit" not in own and own["row_root"] == "/vol/x/a", "a prefix is not a root"
    re_ = lp._row_sampler_check(chk, root="/vol/x/b/", reaudit=True)
    assert [s_["path"] for s_ in re_["sources"]] == ["/vol/x/b/resolved_sampler.json"] and re_["reaudit"]
    none = lp._row_sampler_check(chk, root="/vol/y", reaudit=False)
    assert none["sources"] == [] and co.verified_resolved_decode(none, root="/vol/y", prev_row={}, evidence={}, read_text=lambda p_: None)["ok"] is False
    assert lp._row_sampler_check(None, root="/vol/y", reaudit=False)["sources"] == []
