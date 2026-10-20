"""A declared reader END TO END through the runner and the harvester on a temp registry under the shared emitter's anchor hook
(`strength_ladder` on its synthetic fixture): the first pass over a fresh registry parks every
comparison (the own unsteered runs are emitted by that very read and the frame does not know them -- no provisional anchors), the fold makes
them current and the anchor decision input moves, so the second pass re-reads the group under a new revision id and files every contrast
`own_current` with `extra.anchor_selection`; a third pass is quiet; an `exclude` ruling on one standup's unsteered runs retires them and the
next pass finds the other standup's unsteered runs only under ANOTHER instrument (the record serves the unsteered reading on the passthrough
path and the steered one through the lens: two instruments -- the strict identity, the T2 case) -- so those comparisons park
by name (`no_current_anchor`, `own_reason excluded`), the fold withdraws their old contrast keys, and the other standup's contrasts stand.
Fixtures are invented (tests/registry_declared_fixtures.py); nothing here reads a real store."""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import anchor_current as AC
from sisteer.registry.adapters import declared_common as DC
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import runner as RUN
from sisteer.registry.adapters import strength_ladder as SL

import _b1_fixtures as B1  # noqa: E402
import registry_declared_fixtures as F  # noqa: E402
from _reanchor_fixtures import head_of, q  # noqa: E402

T = ["2026-10-03T0{}:00:00.000Z".format(i) for i in range(1, 8)]
SOURCE = "strength_lane"


def _world(tmp_path, cls=SL.StrengthLadderAdapter):
    root = B1.registry_root(tmp_path, LC.Seeds.locate())
    # the fixture's two fitted objects: the strength reader mints no `object` row (the objects bank does, live), and a run names its object
    R.write_records(os.path.join(root, "records", "standins", "000002-objects.jsonl"),
                    [synth.rec("object", dict(synth.object_lora(F.SHA_A, "constitution_lora", "constitution"), fit_model_id="gemma4e4b_base"), synth.t(30)),
                     synth.rec("object", dict(synth.object_lora(F.SHA_B, "constitution_lora", "constitution"), fit_model_id="gemma4e4b_base"), synth.t(31))])
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refusals, ing.blocked)
    fx = F.build_strength(str(tmp_path / "lora"))
    decl = SL.StrengthRecordDecl(fx["path"], SOURCE, ("const_lora", "diff_lora"), None, {"llama8b": "llama8b"})
    return root, fx, cls(records=(decl,))


def _exclude(root: str, ruling_id: str, run_id: str, i: int) -> None:
    """One `exclude` ruling on one run, filed and folded -- under the stand-in dimensions' paper round (`LC.ROUND_ID`; a ruling names its round)."""
    payload = {"ruling_id": ruling_id, "round_id": LC.ROUND_ID, "action": "exclude", "scope_kind": "run", "scope_json": {"run_id": run_id},
               "target_json": {}, "reason": f"synthetic: exclude {run_id}", "ruled_utc": synth.t(i), "ruled_utc_precision": "datetime",
               "ruled_by": "test", "source_ref": "synthetic"}
    R.write_records(os.path.join(root, "records", "rulings", f"{i:06d}-{ruling_id}.jsonl"), [synth.rec("ruling", payload, synth.t(i))])
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refused, ing.blocked, ing.refusals)


def _pass(root, ad, now):
    rep, ing, ing2 = B1.run_and_ingest(root, ad, now)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refused, ing.blocked, ing.refusals)
    return rep


#: the dose standup of a strength run: the record's two views share `view_legacy` (the model's tag), the producer key names the view
STANDUP_SQL = "CASE WHEN r.producer_run_key_legacy LIKE '%|it_d0p5:%' THEN 'it_d0p5' WHEN r.producer_run_key_legacy LIKE '%|it_d1:%' THEN 'it_d1' END"


def _contrasts(root):
    return q(root, f"SELECT rc.run_id, rc.anchor_run_id, rc.anchor_kind, rc.metric_id, json_extract(rc.extra_json, '$.anchor_selection.state'), {STANDUP_SQL} "
                   "FROM run_contrast rc JOIN run r ON r.run_id = rc.run_id ORDER BY rc.run_id, rc.metric_id, rc.anchor_run_id")


def _current_contrast_keys(root, source_id):
    return {r[0] for r in q(root, "SELECT observation_key FROM observation WHERE source_id = ? AND table_name = 'run_contrast' AND current = 1", source_id)}


def _parks(root):
    """(run_id, metric_id, delta_block.anchor.state, own_reason) of every steered metric row carrying a park."""
    return q(root, "SELECT rm.run_id, rm.metric_id, json_extract(rm.extra_json, '$.delta_block.anchor.state'), json_extract(rm.extra_json, '$.delta_block.anchor.own_reason') "
                   "FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.object_id <> 'none' AND json_extract(rm.extra_json, '$.delta_block') IS NOT NULL "
                   "ORDER BY rm.run_id, rm.metric_id")


def test_fresh_registry_parks_then_files_own_current_then_a_retired_standup_re_picks_and_the_ladder_refuses_by_name(tmp_path):
    root, fx, ad = _world(tmp_path)
    source_id = RUN.source_id_for(ad.name, SOURCE)
    # ---- pass 1: nothing folded -> every comparison parks (own_reason not_in_frame), no contrast, the decision set recorded on the marker
    _pass(root, ad, T[0])
    assert _contrasts(root) == []
    parks1 = _parks(root)
    assert parks1 and {(s, o) for _r, _m, s, o in parks1} == {("no_current_anchor", "not_in_frame")}
    rev1, extra1 = head_of(root, source_id)
    block1 = extra1["anchor_decisions"]
    assert block1["n_comparisons"] == block1["n_parked"] == len(parks1) > 0 and block1["pre_read_fingerprint"] is None
    assert extra1["emission_version"] == f"5+emitter{DC.EMITTER_VERSION}"
    # ---- pass 2: the unsteered runs are current now -> the decision input moved -> a NEW revision, every contrast own_current
    rep2 = _pass(root, ad, T[1])
    assert rep2.anchor_decisions_changed == 1
    rev2, extra2 = head_of(root, source_id)
    assert rev2 != rev1 and extra2["anchor_decisions"]["n_parked"] == 0 and extra2["anchor_decisions"]["n_comparisons"] == block1["n_comparisons"]
    assert extra2["anchor_decisions"]["pre_read_fingerprint"] == extra2["anchor_decisions"]["fingerprint"]        # no drift: the same frame decided both
    cs2 = _contrasts(root)
    assert len(cs2) == block1["n_comparisons"] and {c[4] for c in cs2} == {"own_current"} and {c[2] for c in cs2} == {"same_standup_none"}
    assert _parks(root) == []                                                                          # the parks of pass 1 were per-column replaced
    views = {c[5] for c in cs2}
    assert views == {"it_d0p5", "it_d1"}
    none_by_view = {v: [r[0] for r in q(root, f"SELECT run_id FROM run r WHERE object_id = 'none' AND lane_source_legacy = ? AND {STANDUP_SQL} = ?", SOURCE, v)] for v in views}
    assert all(none_by_view.values())
    for run_id, anchor, _k, _m, _s, view in cs2:
        assert anchor in none_by_view[view], (run_id, anchor, view)
    keys2 = _current_contrast_keys(root, source_id)
    assert len(keys2) == len(cs2)
    # ---- pass 3: nothing moved -> quiet (the head stands, no new revision)
    rep3 = _pass(root, ad, T[2])
    assert rep3.anchor_decisions_changed == 0 and head_of(root, source_id)[0] == rev2
    # ---- an `exclude` ruling on the it_d0p5 unsteered runs: the next pass finds the it_d1 unsteered runs only under the passthrough instrument
    # (T2: never an anchor for a reading under the lens instrument) and parks those comparisons by name, own_reason excluded
    for i, rid in enumerate(none_by_view["it_d0p5"]):
        _exclude(root, f"r_synth_exclude_d0p5_{i}", rid, 10 + i)
    rep4 = _pass(root, ad, T[3])
    assert rep4.anchor_decisions_changed == 1
    rev4, extra4 = head_of(root, source_id)
    assert rev4 != rev2
    cs4 = _contrasts(root)
    current4 = _current_contrast_keys(root, source_id)
    assert {c[5] for c in cs4} == {"it_d0p5", "it_d1"}                       # the old it_d0p5 rows stay in the table (no run-scale row is deleted) ...
    withdrawn = {k for k in keys2 if k not in current4}
    d0p5_run_ids = {c[0] for c in cs2 if c[5] == "it_d0p5"}
    assert withdrawn and all(json.loads(k)["run_id"] in d0p5_run_ids for k in withdrawn)      # ... but their observation keys are withdrawn
    assert len(withdrawn) == sum(1 for c in cs2 if c[5] == "it_d0p5")
    assert all(json.loads(k)["run_id"] not in d0p5_run_ids for k in current4) and len(current4) == sum(1 for c in cs2 if c[5] == "it_d1")
    parks4 = _parks(root)
    assert parks4 and {(s, o) for _r, _m, s, o in parks4} == {("no_current_anchor", "excluded")} and {r for r, *_ in parks4} == d0p5_run_ids
    blk = json.loads(q(root, "SELECT extra_json FROM run_metric WHERE run_id = ? AND json_extract(extra_json, '$.delta_block') IS NOT NULL LIMIT 1", sorted(d0p5_run_ids)[0])[0][0])["delta_block"]
    assert "only under instrument" in blk["why_not_a_contrast_row"] and "selection_state excluded (ruling" in blk["why_not_a_contrast_row"] and blk["anchor"]["anchor_run_id"] is None
    assert blk["anchor"]["own_anchor_run_id"] in none_by_view["it_d0p5"] and blk["anchor_selection"]["rule"] == AC.RULE and blk["delta"] is not None
    assert blk["anchor_selection"]["rejected"] == [] and blk["anchor_selection"]["n_candidates"] == 0        # nothing at the exact key: the other standup's run is under another instrument
    assert extra4["anchor_decisions"]["n_parked"] == len(parks4)
    # the status row names each parked comparison with the own anchor's reason
    refusals = q(root, "SELECT json_extract(extra_json, '$.row_refusals') FROM source_status WHERE source_id = ? ORDER BY observed_utc DESC LIMIT 1", source_id)[0][0]
    assert sum(1 for line in json.loads(refusals) if "contrast has no current anchor at key" in line and ": excluded]" in line) == len(parks4)
    # the paper frame: the excluded standup's steered rows anchor on nothing; the other standup's keep their anchor
    for rid in sorted(d0p5_run_ids):
        rows = q(root, "SELECT anchor_run_id FROM v_paper_long WHERE run_id = ?", rid)
        assert rows and all(r[0] is None for r in rows), (rid, rows)
    d1_run_ids = {c[0] for c in cs2 if c[5] == "it_d1"}
    for rid in sorted(d1_run_ids):
        rows = q(root, "SELECT anchor_run_id FROM v_paper_long WHERE run_id = ? AND anchor_run_id IS NOT NULL", rid)
        assert rows and all(r[0] in none_by_view["it_d1"] for r in rows), (rid, rows)
    # ---- pass 5: quiet again
    rep5 = _pass(root, ad, T[4])
    assert rep5.anchor_decisions_changed == 0 and head_of(root, source_id)[0] == rev4


# ============================================================================================
# a RE-PICKED rate contrast through the runner, the fold and the paper frame (plan commit 3 review: the binding)
# ============================================================================================

class RateLadderAdapter(SL.StrengthLadderAdapter):
    """The ladder under `unpaired_rates` (the family rule `em_misalign` names), so the shared emitter's re-pick path -- the registry's
    unpaired rate difference against a sibling unsteered run -- runs end to end on the ladder's synthetic fixture. Same `name`: the heads,
    the source ids and the fixture are the production adapter's."""
    REANCHOR_STATISTIC = "unpaired_rates"


def _sibling_none(root: str, steered_run: dict, judge_id: str, metric: dict, *, k: int, n: int, i0: int = 200, view_legacy: str = "gemma4e4b_other_standup") -> str:
    """A complete, judged unsteered run at the steered run's EXACT key (model, eval, instrument, cap, trial, pass, decode) from the same
    source under ANOTHER standup (`view_legacy`), carrying one reading at the steered row's (metric, threshold) under the steered row's
    judge -- k of n -- filed as records and folded. The exact-key pool's one eligible candidate: tier 1b, `sibling_none`."""
    run_key = f"ladder/{view_legacy}/none/1.0"
    rid = I.run_id(str(steered_run["campaign"]), run_key, "none", str(steered_run["eval_id"]), str(steered_run["instrument_id"]))
    over = {c: steered_run[c] for c in ("campaign", "model_id", "eval_id", "item_set_id", "instrument_id", "token_cap_tokens", "trial_index", "pass_label", "decode_policy_id")}
    over.update(run_id=rid, lane_source_legacy=SOURCE, view_legacy=view_legacy)
    _rid, recs = synth.run_records(run_key, i0, "none", "none", n_items=n, judge=judge_id, prod=B1.producer(), **over)
    assert _rid == rid
    recs = [r for r in recs if r["record_type"] != "run_metric"]        # the default `asr 0.25` reading is replaced by the steered row's metric at k / n
    recs.append(synth.rec("run_metric", {"run_id": rid, "judgment_id": I.judgment_id(rid, judge_id), "metric_id": metric["metric_id"], "threshold": metric["threshold"],
                                         "value": k / n, "k": k, "n": n, "computed_by": "producer"}, synth.t(i0 + 9), B1.producer()))
    R.write_records(os.path.join(root, "records", "sibling", f"{i0:06d}-sibling-none.jsonl"), recs)
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refused, ing.blocked, ing.refusals)
    cur = q(root, "SELECT is_current, why FROM v_run_current WHERE run_id = ?", rid)
    assert cur == [(1, None)], cur
    return rid


def test_a_re_picked_rate_contrast_is_keyed_on_the_steered_readings_computation_and_is_the_paper_frames_row(tmp_path):
    """The emitted contrast's `computation_id` is the steered metric row's, so `v_paper_long` (which joins `current_contrast` to the
    metric row on the full identity) shows the registry's statistic against the re-picked anchor -- `anchor_run_id`, `anchor_kind`,
    `delta`, `n_current_contrasts 1` -- and the parked producer delta against the RETIRED anchor stays on `metric_extra_json.delta_block`,
    never the frame's `delta`. Under the statistic's own computation id (the shape before the review) the row was accepted into
    `run_contrast` and invisible in the frame (anchor_run_id NULL there)."""
    root, fx, ad = _world(tmp_path, cls=RateLadderAdapter)
    source_id = RUN.source_id_for(ad.name, SOURCE)
    _pass(root, ad, T[0])                                   # park (nothing folded)
    _pass(root, ad, T[1])                                   # own_current everywhere
    cs2 = _contrasts(root)
    assert cs2 and {c[4] for c in cs2} == {"own_current"}
    # the two steered static readings of one key (the two fitted objects at pasttense, standup it_d0p5) and their reading
    steered = [dict(zip(("run_id", "campaign", "model_id", "eval_id", "item_set_id", "instrument_id", "token_cap_tokens", "trial_index", "pass_label",
                         "decode_policy_id", "view_legacy", "judge_id"), r))
               for r in q(root, "SELECT r.run_id, r.campaign, r.model_id, r.eval_id, r.item_set_id, r.instrument_id, r.token_cap_tokens, r.trial_index, r.pass_label, "
                                f"r.decode_policy_id, r.view_legacy, r.judge_id FROM run r WHERE r.object_id <> 'none' AND r.eval_id = 'pasttense' AND {STANDUP_SQL} = 'it_d0p5' "
                                "ORDER BY r.run_id")]
    assert len(steered) == 2 and len({(s["model_id"], s["instrument_id"], s["token_cap_tokens"], s["trial_index"], s["pass_label"], s["judge_id"]) for s in steered}) == 1
    metrics = {s["run_id"]: dict(zip(("judgment_id", "metric_id", "threshold", "computation_id", "k", "n", "value"),
                                    q(root, "SELECT judgment_id, metric_id, threshold, computation_id, k, n, value FROM run_metric WHERE run_id = ? AND metric_id = 'asr'", s["run_id"])[0]))
               for s in steered}
    assert all(m["k"] is not None and m["n"] and m["computation_id"] for m in metrics.values()), metrics
    own_anchor = {c[0]: c[1] for c in cs2 if c[3] == "asr"}
    producer_delta = {s["run_id"]: q(root, "SELECT delta FROM run_contrast WHERE run_id = ? AND anchor_run_id = ? AND metric_id = 'asr'", s["run_id"], own_anchor[s["run_id"]])[0][0]
                      for s in steered}
    # a sibling unsteered run at the exact key (k 5 of 12) and an `exclude` ruling on the own anchors of the it_d0p5 standup
    k_a, n_a = 5, 12
    sib = _sibling_none(root, steered[0], str(steered[0]["judge_id"]), metrics[steered[0]["run_id"]], k=k_a, n=n_a)
    d0p5_nones = [r[0] for r in q(root, f"SELECT run_id FROM run r WHERE object_id = 'none' AND lane_source_legacy = ? AND {STANDUP_SQL} = 'it_d0p5'", SOURCE)]
    assert d0p5_nones and sib not in d0p5_nones
    for i, rid in enumerate(d0p5_nones):
        _exclude(root, f"r_synth_exclude_d0p5_{i}", rid, 10 + i)
    rep = _pass(root, ad, T[3])
    assert rep.anchor_decisions_changed == 1 and ad.last_report.contrasts_reanchored == 2
    # ---- the registry rows: the two re-picked contrasts bind to the STEERED reading's computation; the statistic's row is named in extra
    repicked = q(root, "SELECT rc.run_id, rc.computation_id, rc.anchor_kind, rc.delta, rc.delta_method, rc.computed_by, rc.extra_json FROM run_contrast rc WHERE rc.anchor_run_id = ? ORDER BY rc.run_id", sib)
    assert [r[0] for r in repicked] == [s["run_id"] for s in steered]
    for run_id, comp, kind, delta, method, by, extra_json in repicked:
        m = metrics[run_id]
        ex = json.loads(extra_json)
        assert comp == m["computation_id"] and kind == "sibling_none" and by == "registry" and method == DC.UNPAIRED_RATE_METHOD
        assert delta == pytest.approx(100 * (m["k"] / m["n"] - k_a / n_a)) and delta != pytest.approx(producer_delta[run_id])
        assert ex["anchor_selection"]["state"] == "sibling_same_source" and ex["anchor_selection"]["own_reason"] == "excluded"
        assert ex["statistic_computation_id"] != comp and ex["anchor_standup"]["view_legacy"] == "gemma4e4b_other_standup" and ex["cross_source"] is False
        assert q(root, "SELECT algorithm, version FROM computation WHERE computation_id = ?", ex["statistic_computation_id"]) == [("registry.unpaired_rate_contrast", AC.RULE)]
        assert q(root, "SELECT algorithm FROM computation WHERE computation_id = ?", comp)[0][0].startswith("paperlib.")
    # the own-anchor keys of those two comparisons are withdrawn; the re-picked keys are the source's current ones for them
    current = _current_contrast_keys(root, source_id)
    for s in steered:
        keys = {(json.loads(k)["anchor_run_id"], json.loads(k)["computation_id"]) for k in current if json.loads(k)["run_id"] == s["run_id"] and json.loads(k)["metric_id"] == "asr"}
        assert keys == {(sib, metrics[s["run_id"]]["computation_id"])}, keys
    # ---- the paper frame: ONE current contrast per row, the re-picked anchor, the registry's delta; the producer's delta parked on the metric row
    for s in steered:
        rows = q(root, "SELECT anchor_run_id, anchor_kind, delta, delta_method, n_current_contrasts, contrast_extra_json, metric_extra_json, anchor_token_cap_tokens, cap_matched "
                       "FROM v_paper_long WHERE run_id = ? AND metric_id = 'asr'", s["run_id"])
        assert len(rows) == 1, rows
        anchor, kind, delta, method, n_cur, cx, mx, anchor_cap, cap_matched = rows[0]
        m = metrics[s["run_id"]]
        assert anchor == sib and kind == "sibling_none" and n_cur == 1 and method == DC.UNPAIRED_RATE_METHOD
        assert delta == pytest.approx(100 * (m["k"] / m["n"] - k_a / n_a)) and anchor_cap == s["token_cap_tokens"] and cap_matched == 1
        assert json.loads(cx)["anchor_selection"]["state"] == "sibling_same_source" and json.loads(cx)["statistic_computation_id"]
        blk = json.loads(mx)["delta_block"]
        assert blk["delta"] == pytest.approx(producer_delta[s["run_id"]]) and blk["computed_by"] == "producer" and blk["anchor"]["registry_contrast_filed_against"] == sib
        assert blk["anchor"]["own_anchor_run_id"] == own_anchor[s["run_id"]] and "stands" in blk["why_not_a_contrast_row"]
        assert blk["delta"] != pytest.approx(delta)
    # the two re-picked rows carry the producer's PARKED delta beside their filed contrast (state sibling_same_source); every other it_d0p5
    # comparison (other evals: no exact-key sibling) parks by name with no contrast, and the it_d1 standup's contrasts stand
    parks = _parks(root)
    repicked_ids = {s["run_id"] for s in steered}
    assert {(st, o) for r, m, st, o in parks if r in repicked_ids and m == "asr"} == {("sibling_same_source", "excluded")}
    others = [(r, m, st, o) for r, m, st, o in parks if not (r in repicked_ids and m == "asr")]
    assert others and {(st, o) for _r, _m, st, o in others} == {("no_current_anchor", "excluded")}
    d1_contrasts = [c for c in _contrasts(root) if c[5] == "it_d1"]
    assert d1_contrasts and {c[4] for c in d1_contrasts} == {"own_current"}
    # ---- quiet afterwards
    rep5 = _pass(root, ad, T[4])
    assert rep5.anchor_decisions_changed == 0
