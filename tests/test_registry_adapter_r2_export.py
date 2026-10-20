"""The r2 export adapter on a SYNTHETIC parquet: every column mapped (an unmapped one refuses), runs keyed by the
producer's object discriminator, metrics / contrasts / degeneration / dispositions / queued runs, the Petri
two-source guard per row, the sha pin on the builder's sources, the pairing-allowance confidence -- and, since r2_export 11,
every contrast decided by the pass's anchor currency: the first pass over a fresh registry PARKS
every comparison (the export's own unsteered runs are not folded yet; no provisional anchors), the fold makes them current and the
§4 decision input re-reads the group, which files them own-current; a retired own unsteered run re-picks a current one of the exact
key from the same source and the registry files ITS statistic (an unpaired rate difference; a per-behavior paired difference over
registry item rows for `petri_score`); the `full_dose` kind checks its 1x anchor's currency and parks a retired one. Since 12 the
refused-rate twin's contrast is a comparison of its own, and the bump alone re-reads an 11 head that recorded no twin comparison."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

import _b1_fixtures as F
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import anchor_current as AC
from sisteer.registry.adapters import declared_common as DC
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import r2_export as X
from sisteer.registry.adapters import runner as RUN
from sisteer.registry.adapters.base import AdapterContext

from _reanchor_fixtures import head_of, q  # noqa: E402
from test_registry_anchor_current import CAP, INST, JUDGE, MODEL, TRIAL, Frame  # noqa: E402

T = ["2026-10-03T0{}:00:00.000Z".format(i) for i in range(1, 9)]
SOURCE_ID = RUN.source_id_for(X.NAME, "r2")
PARK_PHRASE = "contrast has no current anchor at key"


def _adapter(tmp_path, paths, seeds=None, **kw):
    seeds = seeds or F.synthetic_seeds()
    return X.R2ExportAdapter(data_dir=paths["data"], rs=paths["rs"], seeds=seeds, objects=F.resolver(seeds, tmp_path), **kw), seeds


def _setup(tmp_path, **kw):
    paths = F.synthetic_r2_export(tmp_path, **kw)
    ad, seeds = _adapter(tmp_path, paths)
    objs = [ad.objects.resolve(F.VIEW, c) for c in (F.LORA, F.RELU, F.SUB, F.PROMPTED)]
    if kw.get("halfdose_rows"):
        half = ad.objects.resolve(f"{F.VIEW}_halfdose", F.LORA)
        if half.object_id not in {o.object_id for o in objs}:
            objs.append(half)
    root = F.registry_root(tmp_path, seeds, objs)
    return paths, ad, seeds, root


def _two_passes(root, ad, now=None):
    """The steady state of a fresh registry: pass 1 parks every comparison (the export's own unsteered runs are not folded yet), the fold
    makes them current, pass 2 re-reads the group through the §4 decision input and files the contrasts own-current. Returns the SECOND
    pass's `(rep, ing, ing2)`; both passes' folds must refuse nothing."""
    rep1, ing1, ing1b = F.run_and_ingest(root, ad, now=(now or T[0]))
    assert rep1.groups_refused == 0 and ing1.refused + ing1b.refused == 0, ([r.error for r in rep1.results], ing1.refusals, ing1b.refusals)
    return F.run_and_ingest(root, ad, now=(T[1] if now is None else now))


def _parks(ad):
    """The adapter's row refusals that are anchor parks (the phrasing the hooked adapters share), and the rest."""
    parks = [r for r in ad.refusals.items if PARK_PHRASE in r["why"] or r["why"].startswith("contrast's own anchor") or r["why"].startswith("full-dose contrast")]
    return parks, [r for r in ad.refusals.items if r not in parks]


def _exclude(root: str, ruling_id: str, run_id: str, i: int) -> None:
    """One `exclude` ruling on one run, filed and folded -- under the stand-in dimensions' paper round (`LC.ROUND_ID`; a ruling names its round)."""
    payload = {"ruling_id": ruling_id, "round_id": LC.ROUND_ID, "action": "exclude", "scope_kind": "run", "scope_json": {"run_id": run_id},
               "target_json": {}, "reason": f"synthetic: exclude {run_id}", "ruled_utc": synth.t(i), "ruled_utc_precision": "datetime",
               "ruled_by": "test", "source_ref": "synthetic"}
    R.write_records(os.path.join(root, "records", "rulings", f"{i:06d}-{ruling_id}.jsonl"), [synth.rec("ruling", payload, synth.t(i))])
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refused, ing.blocked, ing.refusals)


def _steered(root, object_entry, eval_id, cap=None):
    """The key columns of one steered run of the export (`object_entry` = its `campaign_cell_id_legacy`)."""
    sql = ("SELECT r.run_id, r.campaign, r.model_id, r.eval_id, r.item_set_id, r.instrument_id, r.token_cap_tokens, r.trial_index, r.pass_label, "
           "r.decode_policy_id, r.view_legacy, r.judge_id FROM run r WHERE r.campaign_cell_id_legacy = ? AND r.eval_id = ?")
    args = [object_entry, eval_id]
    if cap is not None:
        sql += " AND r.token_cap_tokens = ?"
        args.append(cap)
    rows = q(root, sql, *args)
    assert len(rows) == 1, rows
    return dict(zip(("run_id", "campaign", "model_id", "eval_id", "item_set_id", "instrument_id", "token_cap_tokens", "trial_index", "pass_label",
                     "decode_policy_id", "view_legacy", "judge_id"), rows[0]))


def _sibling_none(root, steered, metric, *, i0, view_legacy="synthview_sibling", **metric_over):
    """A complete, judged unsteered run of source `r2_export` at the steered run's EXACT key under ANOTHER standup of the same source
    (`view_legacy` differs), carrying ONE reading at the steered row's (metric, threshold) under its judge -- filed as records and folded. Tier 1b's one
    eligible candidate: `sibling_none`. Returns `(run_id, judgment_id)`."""
    run_key = f"r2:{view_legacy}|none|1|"
    rid = I.run_id(str(steered["campaign"]), run_key, "none", str(steered["eval_id"]), str(steered["instrument_id"]))
    over = {c: steered[c] for c in ("campaign", "model_id", "eval_id", "item_set_id", "instrument_id", "token_cap_tokens", "trial_index", "pass_label", "decode_policy_id")}
    over.update(run_id=rid, lane_source_legacy="r2_export", view_legacy=view_legacy)
    judge = str(steered["judge_id"])
    _rid, recs = synth.run_records(run_key, i0, "none", "none", n_items=int(metric_over.get("n") or 10), judge=judge, prod=F.producer(), **over)
    assert _rid == rid
    jid = I.judgment_id(rid, judge)
    recs = [r for r in recs if r["record_type"] != "run_metric"]
    recs.append(synth.rec("run_metric", dict({"run_id": rid, "judgment_id": jid, "metric_id": metric, "threshold": -1.0, "computed_by": "producer"}, **metric_over),
                          synth.t(i0 + 9), F.producer()))
    R.write_records(os.path.join(root, "records", "sibling", f"{i0:06d}-sibling-none.jsonl"), recs)
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refused, ing.blocked, ing.refusals)
    assert q(root, "SELECT is_current, why FROM v_run_current WHERE run_id = ?", rid) == [(1, None)]
    return rid, jid


def _items(root, run_id, judgment_id, rows, i, metric="petri_score"):
    """Item rows of `(run_id, judgment_id)` at `metric` -- `[(item_id, value), ...]` at trial 0, sample 0, turn -1 -- filed under a synthetic
    source of their own (a `source_revision` marker and the rows, as the harness's `conflicting_provenance` files them) and folded."""
    source_id = f"zz_items_{i}"
    rev = f"{source_id}:rev{i}"
    marker = synth.rec("source_revision", {"revision_id": rev, "source_id": source_id, "adapter": "synthetic_items", "observed_utc": synth.t(i),
                                           "inputs_json": {f"synthetic:///items/{i}": "b" * 64}, "accepted": 1, "priority": 0}, synth.t(i))
    recs = [synth.rec("item_result", {"run_id": run_id, "item_id": item_id, "metric_id": metric, "trial_index": 0, "sample_index": 0, "turn_index": -1,
                                      "judgment_id": judgment_id, "value": value, "label": None, "api_pool": None, "created_utc": synth.t(i),
                                      "judged_utc": None, "source_revision_id": rev, "withdrawn_by_revision_id": None}, synth.t(i + 1))
            for item_id, value in rows]
    R.write_records(os.path.join(root, "records", f"items_{i}", "000001-source_revision.jsonl"), [marker])
    R.write_records(os.path.join(root, "records", f"items_{i}", "000002-item_result.jsonl.gz"), recs, gz=True)
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refused, ing.blocked, ing.refusals)


def _contrast_rows(root, run_id, metric_id, thr):
    return [dict(zip(("anchor_run_id", "anchor_kind", "delta", "delta_method", "computed_by", "computation_id", "anchor_judgment_id", "n_pairs", "extra"), r))
            for r in q(root, "SELECT anchor_run_id, anchor_kind, delta, delta_method, computed_by, computation_id, anchor_judgment_id, n_pairs, extra_json "
                             "FROM run_contrast WHERE run_id = ? AND metric_id = ? AND threshold = ? ORDER BY anchor_run_id", run_id, metric_id, thr)]


def _current_contrast_keys(root):
    return {r[0] for r in q(root, "SELECT observation_key FROM observation WHERE source_id = ? AND table_name = 'run_contrast' AND current = 1", SOURCE_ID)}


def _metric_extra(root, run_id, metric_id, thr):
    rows = q(root, "SELECT extra_json FROM run_metric WHERE run_id = ? AND metric_id = ? AND threshold = ? AND computed_by = 'producer' AND extra_json LIKE '%row_id%'",
             run_id, metric_id, thr)
    assert len(rows) == 1, rows
    return json.loads(rows[0][0])


def test_every_parquet_column_is_mapped_and_an_unmapped_one_refuses(tmp_path):
    paths = F.synthetic_r2_export(tmp_path, extra_column="brand_new_column")
    ad, seeds = _adapter(tmp_path, paths)
    root = F.registry_root(tmp_path, seeds)
    rep, _, _ = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 1 and "unmapped column" in rep.results[0].error and "brand_new_column" in rep.results[0].error


def test_a_listed_source_whose_bytes_moved_refuses_only_the_rows_naming_it(tmp_path):
    """The sha-mismatch input rule scoped to the undeterminable, the way a MISSING source already is: the builder pinned a file at one
    sha and its producer rewrote it afterwards. Before 2026-09-20 that refused the WHOLE read at `discover()` — ONE of the live export's
    189 pins (`bon_cap4096/analysis/bon_cap4096_stats.json`, 378 best-of-N climb rows) held the paper's source of record at its emission-3
    head while every other row of the export was intact. Now the rows whose provenance names the moved file are refused by name, every
    other row reads, the file is an `artifact(kind=report)` carrying both shas (and is NOT an input of the read), and the status facts
    name it."""
    paths, ad, seeds, root = _setup(tmp_path, moved_source=True)
    rep, ing, ing2 = _two_passes(root, ad)
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0, rep.results[0].error
    assert ing.refused + ing2.refused == 0, ing.refusals + ing2.refusals
    assert ad.stats["rows"] == 18 and ad.stats["runs"] == 13       # the climb row minted no run; the other 16 rows read as before
    why = [r for r in ad.refusals.items if "no longer matches" in r["why"]]
    assert len(why) == 2 and any(r["key"].startswith("source:") for r in why) and any("asr_bon_cap4096" in r["key"] for r in why)
    assert F.q(root, "SELECT COUNT(*) FROM run_metric WHERE metric_id = 'asr_bon_cap4096'")[0][0] == 0
    art = F.q(root, "SELECT kind, extra_json FROM artifact WHERE uri_primary = ?", (paths["moved"],))
    assert len(art) == 1 and art[0][0] == "report"
    x = json.loads(art[0][1])
    assert x["sha_mismatch"] is True and x["sha256_on_disk"] == F.sha_file(paths["moved"]) and x["sha256_pinned_by_builder"] != x["sha256_on_disk"]
    assert F.q(root, "SELECT COUNT(*) FROM artifact WHERE artifact_id = ?", (x["sha256_on_disk"],))[0][0] == 0    # not an input of the read
    st = ad.status_extra("r2")
    assert [m["path"] for m in st["mismatched"]] == [LC.rel_home(paths["moved"])] and st["missing"] == []
    assert st["mismatched"][0]["sha256_on_disk"] == x["sha256_on_disk"]
    members = F.q(root, "SELECT current_members_json FROM source_state WHERE json_path = '$.row_refusals'")
    assert members and "no longer matches" in members[0][0]
    # the other input rules are untouched: a pinned source that is a symlink still refuses the read outright
    os.remove(paths["moved"])
    os.symlink(paths["ts"], paths["moved"])
    ad2, _ = _adapter(tmp_path, paths, seeds=seeds)
    rep2, _, _ = F.run_and_ingest(root, ad2)
    assert rep2.groups_refused == 1 and rep2.results[0].group == "<discover>" and "symlink" in rep2.results[0].error


def test_rows_become_runs_metrics_contrasts_dispositions(tmp_path):
    paths, ad, seeds, root = _setup(tmp_path)
    # ---- pass 1 (a fresh registry): every record but the contrasts; the five comparisons PARK -- their own unsteered runs are emitted by this
    # very read and the frame does not know them (no provisional anchors) -- each named in row_refusals, the producer's delta on
    # the metric row's delta_block, the whole decision set (all parked) on the head marker
    rep1, ing1, ing1b = F.run_and_ingest(root, ad, now=T[0])
    assert rep1.groups_refreshed == 1 and ing1.refused + ing1b.refused == 0, (rep1.results[0].error, ing1.refusals, ing1b.refusals)
    parks, other = _parks(ad)
    assert other == [] and len(parks) == 5 and all("not_in_frame" in r["why"] and r["why"].startswith(PARK_PHRASE) for r in parks), ad.refusals.items
    assert ad.stats["contrasts"] == 0 and ad.stats["contrasts_parked"] == 5 and ad.stats["anchor_states"] == {"no_current_anchor": 5}
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast") == [(0,)]
    blocks = F.q(root, "SELECT rm.delta_block FROM (SELECT json_extract(extra_json, '$.delta_block') AS delta_block FROM run_metric) rm WHERE rm.delta_block IS NOT NULL "
                       "AND json_extract(rm.delta_block, '$.anchor') IS NOT NULL")
    assert len(blocks) == 5
    for (b,) in blocks:
        b = json.loads(b)
        assert b["anchor"]["state"] == "no_current_anchor" and b["anchor"]["own_reason"] == "not_in_frame" and b["anchor_selection"]["rule"] == AC.RULE
        assert "why_not_a_contrast_row" in b and b["rule"] == AC.RULE
    lora_blk = _metric_extra(root, _steered(root, F.LORA, "pasttense", 4096)["run_id"], "asr", 0.5)["delta_block"]
    assert lora_blk["delta"] == -20.0 and lora_blk["delta_method"].startswith("paired bootstrap") and lora_blk["effect_kind"].startswith("cohen_h")
    rev1, extra1 = head_of(root, SOURCE_ID)
    assert extra1["emission_version"] == "12" and extra1["anchor_decisions"]["n_comparisons"] == extra1["anchor_decisions"]["n_parked"] == 5
    assert extra1["anchor_decisions"]["pre_read_fingerprint"] is None and extra1["anchor_decisions"]["rule"] == AC.DECISION_RULE
    # ---- pass 2: the unsteered runs are current now -> the decision input moved -> a NEW revision id over the same files, every contrast
    # own_current (the producer's record plus extra.anchor_selection), every park lifted by name
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[1])
    assert rep.groups_refreshed == 1 and rep.anchor_decisions_changed == 1, rep.results[0].error
    assert ing.refused + ing2.refused == 0, ing.refusals + ing2.refusals
    assert len(ad.refusals) == 0, ad.refusals.items
    assert ad.stats["contrasts_parked"] == 0 and ad.stats["anchor_states"] == {"own_current": 5} and ad.stats["reanchor_computations"] == 0
    rev2, extra2 = head_of(root, SOURCE_ID)
    assert rev2 != rev1 and extra2["anchor_decisions"]["n_parked"] == 0 and extra2["anchor_decisions"]["n_comparisons"] == 5
    assert extra2["anchor_decisions"]["pre_read_fingerprint"] == extra2["anchor_decisions"]["fingerprint"]
    assert F.q(root, "SELECT COUNT(*) FROM run_metric WHERE json_extract(extra_json, '$.delta_block.anchor') IS NOT NULL") == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM run_metric WHERE json_type(extra_json, '$.delta_block') = 'null'") == [(5,)]       # lifted by an explicit null
    sel = [json.loads(r[0])["anchor_selection"] for r in F.q(root, "SELECT extra_json FROM run_contrast")]
    assert len(sel) == 5 and {x["state"] for x in sel} == {"own_current"} and {x["own_reason"] for x in sel} == {"current"} and {x["rule"] for x in sel} == {AC.RULE}
    assert all(r[0] for r in F.q(root, "SELECT anchor_judgment_id FROM run_contrast"))     # the anchor's judgment, from the frame
    st = ad.stats
    assert st["rows"] == 17 and st["dispositions"] == 1 and st["queued_runs"] == 1
    # one run per (producer key, object, eval, instrument): static none + lora (pasttense), bon lora, petri none + lora, pending relu (codeattack),
    # cap-4096 relu (grades root) + its it-view anchor, the 512 multi-turn none + lora, the xstest and mmlu lora rows, the pooled sampled lora
    assert st["runs"] == 13
    # the over-refusal row names no judge: the eval's CPI judge is derived and a judgment exists; the checker-scored mmlu run has no judge
    # at all (a rule-scored instrument) — no judgment row ever, and the harvester stamps it ingested on its complete headline metric
    # (judgment_id ''), with judged_utc = the metric's computed_utc, else the run's generated_utc (rule scoring happens at generation)
    xs = F.q(root, "SELECT r.judge_id, r.lifecycle_state, r.field_confidence_json, (SELECT COUNT(*) FROM judgment j WHERE j.run_id = r.run_id) FROM run r WHERE r.eval_id = 'xstest'")
    assert len(xs) == 1 and xs[0][0] == seeds.judge_id("xstest_cpi_v1_sonnet5") and xs[0][1] == "judged" and xs[0][3] == 1
    assert json.loads(xs[0][2])["judge_id"]["confidence"] == "derived"
    mm = F.q(root, "SELECT r.judge_id, r.lifecycle_state, rm.judgment_id, rm.value, (SELECT COUNT(*) FROM judgment j WHERE j.run_id = r.run_id), "
                   "r.judged_utc = COALESCE((SELECT MAX(m2.computed_utc) FROM run_metric m2 JOIN eval e ON e.eval_id = r.eval_id "
                   "WHERE m2.run_id = r.run_id AND m2.metric_id = e.headline_metric_id AND m2.judgment_id = ''), r.generated_utc) "
                   "FROM run r JOIN run_metric rm ON rm.run_id = r.run_id WHERE r.eval_id = 'mmlu'")
    assert mm == [(None, "ingested", "", 0.6, 0, 1)]
    # metrics: the bon curve point keeps a unique key
    ids = {r[0] for r in F.q(root, "SELECT metric_id FROM run_metric")}
    assert "asr_at_n64" in ids and "asr" in ids and "petri_score" in ids
    # frac mirrors and the parked share
    row = F.q(root, "SELECT rm.frac_at_token_cap, rm.frac_degenerate_rep_ratio_loop, rm.frac_loop_first512, rm.extra_json FROM run_metric rm JOIN run r ON r.run_id = rm.run_id "
                    "WHERE r.campaign_cell_id_legacy = ? AND rm.metric_id = 'asr' AND rm.threshold = 0.5 AND r.eval_id = 'pasttense' AND r.token_cap_tokens = 4096", (F.LORA,))[0]
    assert row[0] == pytest.approx(0.2) and row[1] == pytest.approx(0.1) and row[2] == pytest.approx(0.05)
    assert json.loads(row[3])["degenerate_share"] == pytest.approx(0.2) and json.loads(row[3])["status_legacy"] == "graded"
    assert json.loads(row[3])["steer_live_ratio"] == "Infinity"   # a non-finite producer float parks as its JSON-safe spelling
    # defended-degenerate counts are a degeneration row (defended_only)
    dg = F.q(root, "SELECT numerator, denominator, denominator_unit, measurement_state FROM run_degeneration WHERE measure = 'defended_degenerate'")
    assert (2, 7, "items", "measured") in {tuple(r) for r in dg}
    # the lora contrast points at the lane's none run; the petri contrast too; the relu_native row at the it-view anchor (borrowed)
    kinds = {r[0] for r in F.q(root, "SELECT anchor_kind FROM run_contrast")}
    assert {"same_standup_none", "borrowed"} <= kinds
    c = F.q(root, "SELECT rc.delta, rc.effect_kind, rc.extra_json FROM run_contrast rc JOIN run r ON r.run_id = rc.run_id WHERE r.campaign_cell_id_legacy = ? AND rc.anchor_kind = 'borrowed'", (F.RELU,))[0]
    assert c[0] == -5.0 and c[1] == "cohens_h"
    cx = json.loads(c[2])
    assert cx["pairing_allowance"] == "passthrough-anchor-engine-agnostic-2026-09-15"
    assert cx["pairing_allowance_confidence"]["confidence"] in ("recorded", "recorded_from_row")
    assert cx["anchor_engine"] == "lens" and cx["anchor_root"] == "grades_anchor_synthview_it"
    # the not-applicable row is a disposition; the pending row a queued run with a value-less metric
    disp = F.q(root, "SELECT state, eval_id, extra_json FROM disposition")
    assert len(disp) == 1 and disp[0][0] == "not_applicable" and disp[0][1] == "crescendo" and json.loads(disp[0][2])["cell_id"] == F.SUB
    queued = F.q(root, "SELECT r.lifecycle_state, rm.value FROM run r JOIN run_metric rm ON rm.run_id = r.run_id WHERE r.campaign_cell_id_legacy = ? AND r.eval_id = 'codeattack'", (F.RELU,))
    assert queued == [("queued", None)]
    # superseded instruments carry their supersede ruling (emitted here as a placeholder a later change adopts)
    rul = {r[0] for r in F.q(root, "SELECT ruling_id FROM ruling")}
    assert LC.supersede_instrument_ruling_id("static_greedy_4096_full22") in rul
    assert F.q(root, "SELECT status_ruling_id FROM run WHERE eval_id = 'pasttense' AND campaign_cell_id_legacy = ? AND token_cap_tokens = 4096", (F.LORA,))[0][0] == \
        LC.supersede_instrument_ruling_id("static_greedy_4096_full22")
    # the pooled sampled reading of the same cell is a canonical instrument: no ruling
    assert F.q(root, "SELECT status_ruling_id FROM run WHERE eval_id = 'pasttense' AND campaign_cell_id_legacy = ? AND token_cap_tokens = 8192", (F.LORA,)) == [(None,)]
    # Petri rows record both producers
    sp = F.q(root, "SELECT source_producers_json FROM run_metric WHERE metric_id = 'petri_score'")
    assert all(len(json.loads(r[0])) == 2 for r in sp)
    # every input is an artifact; the grade-root directory entry is a store_root artifact
    kinds = {r[0] for r in F.q(root, "SELECT kind FROM artifact")}
    assert {"report", "store_root"} <= kinds
    # the run's producer keys: grades root for the cap-4096 row, the r2 row key elsewhere, the petri run key for petri rows
    pks = {r[0] for r in F.q(root, "SELECT producer_run_key_legacy FROM run")}
    assert "grades_synth_bu_synthview" in pks and "aaaa000000000001" in pks and LC.producer_key_for_r2_row(F.VIEW, F.LORA, 1.0, None) in pks
    # ---- pass 3: nothing changed (no file, no decision) -> nothing written
    rep3, ing3, _ = F.run_and_ingest(root, ad, now=T[2])
    assert rep3.inputs_changed == 0 and rep3.groups_refreshed == 0 and rep3.anchor_decisions_changed == 0 and head_of(root, SOURCE_ID)[0] == rev2


def test_petri_two_source_disagreement_refuses_the_row_by_name(tmp_path):
    paths = F.synthetic_r2_export(tmp_path, petri_stats_key="aaaa000000000001", petri_agg_key="bbbb000000000002")
    ad, seeds = _adapter(tmp_path, paths)
    objs = [ad.objects.resolve(F.VIEW, c) for c in (F.LORA, F.RELU, F.SUB)]
    root = F.registry_root(tmp_path, seeds, objs)
    rep, ing, ing2 = _two_passes(root, ad)
    assert rep.groups_refreshed == 1
    assert len(ad.refusals) == 1 and "two-source guard" in ad.refusals.items[0]["why"]
    # the refusal is durable: a source_state row lists it
    members = F.q(root, "SELECT current_members_json FROM source_state WHERE json_path = '$.row_refusals'")
    assert members and "two-source guard" in members[0][0]
    # the none petri row (keys agree) still made it
    assert F.q(root, "SELECT COUNT(*) FROM run_metric WHERE metric_id = 'petri_score'")[0][0] == 1


def test_select_instrument_rules():
    assert X.select_instrument("static", "pasttense", 4096, "graded", "asr", None, None)[0] == "static_greedy_4096_full22"
    assert X.select_instrument("static", "pasttense_sampled4096", 8192, "graded", "asr", None, None)[0] == "static_sampled_vendor_preset_4096_8192"
    assert X.select_instrument("multi-turn", "pair", 4096, "variant", "asr", "greedy (T=0)", None)[:2] == ("multiturn_pair_cap4096_generation_config", "mismatch")
    assert X.select_instrument("multi-turn", "pair", 512, "graded", "asr", None, None)[0] == "multiturn_greedy_512_full22"
    assert X.select_instrument("bon", "bon", 4096, "graded", "asr_bon_cap4096", None, None)[0] == "bon_512store_4096climb_T1"
    assert X.select_instrument("bon", "bon", 8192, "graded", "asr", None, None)[1] == "mismatch"
    assert X.select_instrument("over-refusal", "xstest_sampled4096", 8192, "graded", "over_refusal_rate_full", None, None)[0] == "overrefusal_xstest_sampled_vendor_preset_ladder"
    assert X.select_instrument("capability", "mmlu", None, "graded", "acc", None, None)[0] == "mmlu_letter_prefill_200"
    assert X.select_instrument("capability", "unknown", None, "graded", "acc", None, None)[0] is None


def test_column_map_covers_the_four_announced_columns():
    for c in ("pairing_allowance", "anchor_engine", "anchor_view", "anchor_root"):
        assert c in X.COLUMN_MAP and "run_contrast" in X.COLUMN_MAP[c]
    assert len(X.COLUMN_MAP) == 82


def test_a_listed_source_missing_on_disk_refuses_only_the_rows_naming_it(tmp_path):
    """The input rules scoped to the undeterminable: the builder's _sources.json pins a file that is gone (renamed / in
    flight); the rows whose provenance names it are refused by name, every other row is read, and the gap is an artifact row."""
    paths, ad, seeds, root = _setup(tmp_path, missing_source=True)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1, rep.results[0].error
    assert ing.refused + ing2.refused == 0, ing.refusals + ing2.refusals
    assert ad.stats["rows"] == 18 and ad.stats["runs"] == 13
    why = [r for r in ad.refusals.items if "missing on disk" in r["why"]]
    assert len(why) == 2 and any(r["key"].startswith("source:") for r in why) and any("degen_classifier_share" in r["key"] for r in why)
    art = F.q(root, "SELECT kind, extra_json FROM artifact WHERE uri_primary LIKE '%vanished_union.json'")
    assert len(art) == 1 and art[0][0] == "report" and json.loads(art[0][1])["missing_on_disk"] is True
    assert json.loads(art[0][1])["sha256_pinned_by_builder"] == "0" * 64
    assert F.q(root, "SELECT COUNT(*) FROM run_metric WHERE metric_id = 'degen_classifier_share'")[0][0] == 0


def test_half_dose_contrast_is_filed_once_at_the_headline_threshold_and_checks_the_1x_anchors_currency(tmp_path):
    """The export's `delta_vs_full_dose` is one number per (target dir, family, cell): the builder's `half_cols(view, fam, cid)` stamps
    the half-dose join's paired record — computed at the 0.5 threshold — on every threshold row of the cell (constant across the
    0.25 / 0.5 / 0.75 rows of all 223 multi-threshold groups of the 2026-09-19 parquet). The `full_dose` contrast is therefore filed
    ONCE, on the headline threshold's row (HALF_DOSE_CONTRAST_THRESHOLD); a threshold-less metric (one row per object entry) files as before.
    Since r2_export 11 the 1x anchor -- a steered run of this export, never re-picked -- must be a CURRENT run of the pass's frame:
    current -> the producer's record plus `extra.anchor_selection` (`currency_only`); retired or not folded -> parked on the metric row's
    `delta_vs_full_dose_block` with the reason; either way the check is a comparison in the group's decision log."""
    paths, ad, seeds, root = _setup(tmp_path)
    fr = Frame()
    fr.steered("run-full")
    fr.run("run-full-none")
    fr.steered("run-old", current=0, why="selection_state superseded (a later run at the same key)", in_paper=0, roster=False)
    ctx = AdapterContext(root="", now=T[0], producer=F.producer(), conn=fr.con)
    ctx.begin_group("r2")
    ad._anchor_states, ad._contrasts_parked = {}, 0
    info = {"view": f"{F.VIEW}_halfdose", "cell_id": F.LORA, "eval_legacy": "pasttense", "cap": 4096, "run_id": "run-half", "judge_id": None}
    anchor_index = {(F.VIEW, "pasttense", 4096, f"cell:{F.LORA}"): "run-full"}
    base = {"metric": "asr", "delta_vs_full_dose": -1.0, "delta_vs_full_dose_lo": -9.0, "delta_vs_full_dose_hi": 7.0, "n": 10}
    out = {thr: ad._half_dose_contrast(dict(base, threshold=thr), info, "comp", anchor_index, ctx) for thr in (0.25, 0.5, 0.75)}
    assert out[0.25] is None and out[0.75] is None
    assert X.HALF_DOSE_CONTRAST_THRESHOLD == 0.5
    assert out[0.5]["park"] is None and out[0.5]["refusal"] is None
    hd = out[0.5]["contrast"]
    assert hd["threshold"] == 0.5 and hd["anchor_kind"] == "full_dose" and hd["anchor_run_id"] == "run-full" and hd["run_id"] == "run-half"
    assert hd["delta"] == -1.0 and hd["delta_lo"] == -9.0 and hd["delta_hi"] == 7.0 and hd["n_pairs"] == 10 and hd["computation_id"] == "comp"
    sel = hd["extra"]["anchor_selection"]
    assert sel["state"] == "own_current" and sel["own_reason"] == "current" and sel["own_anchor_run_id"] == "run-full" and sel["currency_only"]
    assert sel["anchor_kind"] == "full_dose" and sel["rule"] == AC.RULE and sel["n_candidates"] == 0
    for metric in ("at_cap_share", "degenerate_share", "loop_share"):
        nt = ad._half_dose_contrast(dict(base, metric=metric, threshold=None), info, "comp", anchor_index, ctx)
        assert nt is not None and nt["contrast"]["threshold"] == -1.0 and nt["contrast"]["metric_id"] == metric
    # the unsteered row's noise floor anchors on the full-dose unsteered run, under the same once-per-cell rule
    none_info = dict(info, cell_id="none")
    idx = {(F.VIEW, "pasttense", 4096, None): "run-full-none"}
    assert ad._half_dose_contrast(dict(base, threshold=0.75), none_info, "comp", idx, ctx) is None
    assert ad._half_dose_contrast(dict(base, threshold=0.5), none_info, "comp", idx, ctx)["contrast"]["anchor_run_id"] == "run-full-none"
    # a RETIRED 1x anchor parks the comparison on the metric row with the fold's reason (never re-picked: the anchor is a steered run)
    old = ad._half_dose_contrast(dict(base, threshold=0.5), info, "comp", {(F.VIEW, "pasttense", 4096, f"cell:{F.LORA}"): "run-old"}, ctx)
    assert old["contrast"] is None and old["refusal"].startswith("full-dose contrast's 1x anchor run-old is not a current run (superseded")
    blk = old["park"]
    assert (blk["delta"], blk["delta_lo"], blk["delta_hi"], blk["n_pairs"], blk["computed_by"]) == (-1.0, -9.0, 7.0, 10, "producer")
    assert "not a current run" in blk["why_not_a_contrast_row"] and "later run" in blk["why_not_a_contrast_row"]
    assert blk["anchor"] == {"own_anchor_run_id": "run-old", "own_anchor_kind": "full_dose", "own_anchor_judgment_id": None, "state": "no_current_anchor",
                             "own_reason": "superseded", "anchor_run_id": None, "anchor_kind": None, "registry_contrast_filed_against": None}
    assert blk["anchor_selection"]["state"] == "no_current_anchor" and blk["anchor_selection"]["own_reason"] == "superseded" and blk["rule"] == AC.RULE
    # one the registry never folded (the first pass that emits both runs): not_in_frame
    fresh = ad._half_dose_contrast(dict(base, threshold=0.5), info, "comp", {(F.VIEW, "pasttense", 4096, f"cell:{F.LORA}"): "run-unfolded"}, ctx)
    assert fresh["contrast"] is None and fresh["park"]["anchor"]["own_reason"] == "not_in_frame" and "not_in_frame" in fresh["refusal"]
    # every check is a recorded comparison of its own kind: the parked ones re-read the group once the 1x run folds
    kinds = [(e["comparison"].get("kind"), e["decision"]["anchor_run_id"]) for e in ctx.decision_log]
    assert all(k == AC.CURRENCY_KIND for k, _ in kinds) and len(kinds) == 7
    assert [a for _, a in kinds] == ["run-full", "run-full", "run-full", "run-full", "run-full-none", AC.PARKED, AC.PARKED]
    assert ad._anchor_states == {"full_dose:own_current": 5, "full_dose:no_current_anchor": 2} and ad._contrasts_parked == 2
    # without a registry connection the check cannot be made: the GROUP refuses (the runner's rule), never a silent file
    with pytest.raises(AC.AnchorCurrencyUnavailable):
        ad._half_dose_contrast(dict(base, threshold=0.5), info, "comp", anchor_index, AdapterContext(root="", now=T[0], producer=F.producer()))


def test_judged_count_is_read_from_the_counted_row_whatever_row_the_export_lists_first(tmp_path):
    """The Petri unsteered judgments of every view carrying degeneration rows (`qwen35_4b_it`, `qwen35_4b`, the gemma / nemotron / glm views)
    filed `n_items_judged` NULL while `qwen35_4b_halfdose` had 60 (2026-09-20): the export sorts a cell's rows by metric name, so
    `at_cap_share` came first, the run was minted from it, and the judged count was read from the FIRST row only. The count is now read
    from the run's counted headline row (`petri_score` n, the export's per-scenario / pooled count) whatever its position, so the
    unsteered Petri run can serve as an admitted anchor; the degeneration statistic still lands as the run's `run_degeneration` reading."""
    paths, ad, seeds, root = _setup(tmp_path, petri_degeneration_first=True)
    rep, ing, ing2 = _two_passes(root, ad)
    assert ing.refused + ing2.refused == 0, ing.refusals + ing2.refusals
    assert len(ad.refusals) == 0, ad.refusals.items
    rows = F.q(root, "SELECT r.n_items_judged, r.n_items_generated, j.n_items_judged, json_extract(r.field_confidence_json, '$.n_items_judged.confidence') "
                     "FROM run r JOIN judgment j ON j.run_id = r.run_id WHERE r.eval_id = 'petri_pooled' AND r.campaign_cell_id_legacy = 'none'")
    assert rows == [(2, 2, 2, "recorded")]
    assert F.q(root, "SELECT rd.measure, rd.numerator, rd.denominator FROM v_run_degeneration_current rd JOIN run r ON r.run_id = rd.run_id "
                     "WHERE r.eval_id = 'petri_pooled' AND r.campaign_cell_id_legacy = 'none'") == [("at_token_cap", 3, 20)]   # the current reading (two revisions filed it)
    # the steered Petri run, whose score row IS its first row, reads the same count as before
    assert F.q(root, "SELECT j.n_items_judged FROM run r JOIN judgment j ON j.run_id = r.run_id WHERE r.eval_id = 'petri_pooled' AND r.campaign_cell_id_legacy = ?",
               (F.LORA,)) == [(2,)]


def test_fitted_runs_carry_the_operating_point_beside_the_filed_nominal(tmp_path):
    """Every fitted r2 run filed the export's nominal (1.0 / 0.5) with `is_operating_point` NULL, so a roster entry's `operating_point`
    rule could never admit it (2026-09-20): a fitted LoRA / ReLU object at the filed nominal 1.0 IS at the operating point, its unit
    the campaign nominal; the unsteered and prompted rows file no strength."""
    paths, ad, seeds, root = _setup(tmp_path)
    F.run_and_ingest(root, ad)
    fitted = F.q(root, "SELECT DISTINCT o.object_kind, r.strength_nominal, r.strength_nominal_source, r.strength_unit, r.is_operating_point, "
                       "json_extract(r.field_confidence_json, '$.is_operating_point.confidence') FROM run r JOIN object o ON o.object_id = r.object_id "
                       "WHERE o.object_kind IN ('lora', 'relu_steer', 'lora_subtraction') ORDER BY 1")
    assert fitted == [("lora", 1.0, "filed", "nominal_only", 1, "derived"), ("relu_steer", 1.0, "filed", "nominal_only", 1, "derived")]
    bare = F.q(root, "SELECT DISTINCT r.strength_nominal, r.is_operating_point, r.strength_unit FROM run r JOIN object o ON o.object_id = r.object_id "
                     "WHERE o.object_kind IN ('none', 'system_instruction')")
    assert bare == [(None, None, "unverified")]


def _judgment_of(root, cell, eval_id, cap=None):
    sql = ("SELECT j.n_items_judged, r.n_items_judged, r.n_items_generated, r.n_items_planned, j.extra_json, r.field_confidence_json, j.status "
           "FROM run r JOIN judgment j ON j.run_id = r.run_id WHERE r.campaign_cell_id_legacy = ? AND r.eval_id = ?")
    args = [cell, eval_id]
    if cap is not None:
        sql += " AND r.token_cap_tokens = ?"
        args.append(cap)
    rows = F.q(root, sql, tuple(args))
    assert len(rows) == 1, rows
    return rows[0]


def test_over_refusal_judged_counts_come_from_the_export_and_stay_null_with_the_reason_otherwise(tmp_path):
    """The live 2026-09-22 audit: 118 over-refusal judgments of the export (59 per eval) carried `n_items_judged` NULL, and every one failed
    `v_roster_match`'s coverage test (judged + declined + errored >= the items) -- llama8b_it's `control_relu` / `constitution_relu` xstest and
    or_bench_hard runs read `not_in_roster` under roster v4's `figR_overrefusal_absolute/*` entries although everything but coverage matched.
    Three shapes, each read from the export's OWN count and never a guess:

    * the sampled re-run's pooled rows (`over_refusal_rate_full*`): their `n` counts judged LABELS over two seeded trials (2 x the items,
      less the unjudged), so the judged ITEM count is the `n_items` of the producer's pooled block at the row's provenance path in the
      sha-pinned `final_rung.json` (the four Llama judgments: 250 xstest / 1,318 or_bench_hard live); the labels ride the judgment's extra;
    * a controls store's at-cap decomposition rows filed under the store's own producer key (40 runs per eval live): `at_cap_share_all`'s `n`
      is "the N replies judged (every label)" -- the run's only count, so the spelling joins the counted rate spellings, after the rates;
    * a store whose only judged-metric row is still pending (11 per eval live): no count anywhere -> NULL, the reason on the judgment;
    * and a pooled block the producer filed without `n_items` -> NULL with the reason (the labels are never halved into a count)."""
    paths, ad, seeds, root = _setup(tmp_path, overrefusal_count_rows=True)
    rep, ing, ing2 = _two_passes(root, ad)
    assert rep.groups_refreshed == 1, rep.results[0].error
    assert ing.refused + ing2.refused == 0, ing.refusals + ing2.refusals
    assert len(ad.refusals) == 0, ad.refusals.items
    # (1) the pooled sampled reading: 10 items judged (the block's n_items), NOT the 19 pooled labels of the row
    jn, rn, rg, rp, jx, fc, status = _judgment_of(root, F.LORA, "xstest", cap=2048)
    assert (jn, rn, rg, rp, status) == (10, 10, 10, None, "complete")
    jx, fc = json.loads(jx), json.loads(fc)
    assert jx["n_labels_pooled"] == 19 and jx["n_trials"] == 2 and "n_items_judged_unknown" not in jx
    assert "pooled block" in jx["n_items_source"] and "final_rung.json" in jx["n_items_source"] and "over_refusal_rate_full" in jx["n_items_source"]
    assert fc["n_items_judged"]["confidence"] == "recorded" and "n_items of the pooled block" in fc["n_items_judged"]["source"]
    # the row's own n is untouched on the metric (the labels are the metric's denominator)
    assert F.q(root, "SELECT rm.n, rm.k FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.campaign_cell_id_legacy = ? AND r.eval_id = 'xstest' "
                     "AND rm.metric_id = 'over_refusal_rate_full'", (F.LORA,)) == [(19, 3)]
    # the greedy 512 reading of the same cell still counts from its `over_refusal_rate` row (the rate outranks everything)
    jn512, rn512, *_rest = _judgment_of(root, F.LORA, "xstest", cap=512)
    assert (jn512, rn512) == (10, 10)
    # (2) the controls store's at-cap decomposition: `at_cap_share_all` n = the replies judged
    jn, rn, rg, rp, jx, fc, status = _judgment_of(root, F.RELU, "xstest")
    assert (jn, rn, rg, rp, status) == (10, 10, 10, None, "complete")
    assert "at_cap_share_all" in json.loads(jx)["n_items_source"] and "at_cap_share_all" in json.loads(fc)["n_items_judged"]["source"]
    assert F.q(root, "SELECT producer_run_key_legacy FROM run WHERE campaign_cell_id_legacy = ? AND eval_id = 'xstest'", (F.RELU,)) == [("relu_store_x",)]
    # (3) the pending store: NULL, and the judgment says why (no counted row; the judged-metric row is pending)
    jn, rn, rg, rp, jx, fc, status = _judgment_of(root, F.SUB, "or_bench_hard")
    assert (jn, rn, rg, rp, status) == (None, None, None, None, "complete")
    jx, fc = json.loads(jx), json.loads(fc)
    assert "n_items_source" not in jx
    why = jx["n_items_judged_unknown"]
    assert "at_cap_share" in why and "over_refusal_rate_terminated" in why and "pending" in why
    assert "n_items_judged" not in fc
    # (4) the pooled block without n_items: NULL with the reason naming the block; the labels (20) are never turned into a count
    jn, rn, rg, rp, jx, fc, status = _judgment_of(root, F.LORA, "or_bench_hard", cap=2048)
    assert (jn, rn, rg) == (None, None, None)
    why = json.loads(jx)["n_items_judged_unknown"]
    assert "carries no n_items" in why and "n = 20" in why and "or_bench_hard_1k" in why
    # every judgment names its count's source, or the reason it has none
    for (x,) in F.q(root, "SELECT extra_json FROM judgment"):
        x = json.loads(x)
        assert ("n_items_source" in x) != ("n_items_judged_unknown" in x), x


def test_petri_per_behaviour_and_pooled_judgments_carry_their_counts(tmp_path):
    """The per-behaviour Petri score row's `n` (its scenarios) and the pooled row's `n` (the pooled items) are the runs' judged counts --
    per-behaviour and pooled runs are distinct runs (distinct evals), each counted from its own `petri_score` row."""
    paths, ad, seeds, root = _setup(tmp_path, overrefusal_count_rows=True)
    F.run_and_ingest(root, ad)
    rows = F.q(root, "SELECT r.eval_id, j.n_items_judged, r.n_items_judged, json_extract(j.extra_json, '$.n_items_source') FROM run r JOIN judgment j ON j.run_id = r.run_id "
                     "WHERE r.eval_id LIKE 'petri%' AND r.campaign_cell_id_legacy = ? ORDER BY r.eval_id", (F.LORA,))
    assert [(r[0], r[1], r[2]) for r in rows] == [("petri_defer_to_users", 10, 10), ("petri_pooled", 2, 2)]
    assert all("petri_score" in r[3] for r in rows)


def test_judged_count_precedence_is_by_spelling_not_row_order():
    """`_judged_count` ranks a counted spelling by its position in COUNTED_HEADLINE_METRICS and a pooled-label spelling after every one of
    them; a row with no count says so with None, and a spelling outside both lists is no candidate."""
    ad = X.R2ExportAdapter(data_dir="/nonexistent", rs="/nonexistent")
    info = {"source_path": None, "provenance_json_path": "", "source_uri": None}
    r_rate = ad._judged_count({"metric": "over_refusal_rate", "n": 250}, info)
    r_all = ad._judged_count({"metric": "at_cap_share_all", "n": 251}, info)
    r_none = ad._judged_count({"metric": "unsteered_refusal_rate", "n": 249}, info)
    assert r_rate[0] < r_none[0] < r_all[0] and (r_rate[1], r_none[1], r_all[1]) == (250, 249, 251)
    pooled = ad._judged_count({"metric": "over_refusal_rate_full", "n": 499}, info)
    assert pooled[0] > r_all[0] and pooled[1] is None and "provenance file" in pooled[2]["n_items_judged_unknown"]
    assert ad._judged_count({"metric": "asr", "n": None}, info) is None
    assert ad._judged_count({"metric": "at_cap_share", "n": 500}, info) is None
    assert X.COUNTED_HEADLINE_METRICS[:4] == ("asr", "petri_score", "acc", "over_refusal_rate")
    # a lower-precedence row never displaces a filled count; a higher one does, whatever the order the export lists them
    p = {"_n_items": None, "field_confidence_json": {}}
    ad._note_count(p, {"metric": "at_cap_share_all", "n": 251, "status": "graded"}, info)
    ad._note_count(p, {"metric": "over_refusal_rate", "n": 250, "status": "graded"}, info)
    ad._note_count(p, {"metric": "unsteered_refusal_rate", "n": 249, "status": "graded"}, info)
    assert p["_n_items"] == 250 and p["n_items_judged"] == 250 and "over_refusal_rate" in p["field_confidence_json"]["n_items_judged"]["source"]
    assert sorted(p["_row_spellings"]) == ["at_cap_share_all", "over_refusal_rate", "unsteered_refusal_rate"]


def test_over_refusal_headline_is_derived_from_the_two_pooled_label_rates(tmp_path):
    """The paper's reader draws the eval's headline metric only (`over_refusal_graded`), and the sampled re-run's runs filed the export's two
    pooled label rates and nothing else -- so the four Llama ReLU-steer over-refusal runs, `selected` since r2_export 6, could not draw
    (2026-09-22 07:48Z). The export defines `over_refusal_rate_full` as the share of judged labels labelled full refusal and
    `over_refusal_rate_full_plus_partial` as the share labelled full OR partial (partial weighted 1), so the two-term graded value is
    exact: (k_full + 0.5 (k_fpp - k_full)) / n over the same labels. It is filed under the headline id by the registry, under its own
    computation row, naming its two source rows and the two terms of the declared readers' convention it cannot supply (a reply at the
    cap scores 0.5 whatever its label; per-item pooling), with the at-cap bound; the mirror pass stamps it as the headline row. A run with
    ONE of the two rates gets no row (the or_bench pooled block of the fixture), the greedy `over_refusal_rate` runs get none (full
    refusals only), and the two rate rows stand unchanged."""
    paths, ad, seeds, root = _setup(tmp_path, overrefusal_count_rows=True)
    rep, ing, ing2 = _two_passes(root, ad)
    assert rep.groups_refreshed == 1, rep.results[0].error
    assert ing.refused + ing2.refused == 0, ing.refusals + ing2.refusals
    assert len(ad.refusals) == 0, ad.refusals.items
    assert ad.stats["graded_derived"] == 1
    rows = F.q(root, "SELECT rm.value, rm.k, rm.n, rm.threshold, rm.computed_by, rm.producer_metric_spelling, rm.frac_at_token_cap, rm.ci_method, rm.extra_json, rm.judgment_id, "
                     "c.algorithm, c.seed_provenance, rm.source_uri FROM run_metric rm JOIN run r ON r.run_id = rm.run_id JOIN computation c ON c.computation_id = rm.computation_id "
                     "WHERE rm.metric_id = 'over_refusal_graded'")
    assert len(rows) == 1, rows
    value, k, n, thr, by, spelling, frac_cap, ci_method, extra, jid, algo, rule, src = rows[0]
    assert value == pytest.approx((3 + 0.5 * (5 - 3)) / 19) and k is None and n == 19 and thr == -1.0 and by == "registry" and ci_method is None
    assert spelling == X.GRADED_SPELLING and algo == X.GRADED_ALGORITHM and rule == X.GRADED_RULE
    assert frac_cap == pytest.approx(0.05)     # the run's at_token_cap degeneration reading, mirrored onto the headline row (1 of 20 at the cap)
    assert src == paths["final_rung"].replace(os.path.expanduser("~"), "~") or src.endswith("final_rung.json")
    x = json.loads(extra)
    assert [d["metric_id"] for d in x["derived_from"]] == ["over_refusal_rate_full", "over_refusal_rate_full_plus_partial"]
    assert [(d["k"], d["n"]) for d in x["derived_from"]] == [(3, 19), (5, 19)] and all(d["row_id"] is not None for d in x["derived_from"])
    assert (x["k_full"], x["k_partial"], x["k_effective"], x["n_labels"], x["n_items"], x["n_trials"]) == (3, 2, 4.0, 19, 10, 2)
    assert x["at_cap_share"] == pytest.approx(0.05) and x["max_abs_deviation_from_at_cap_term"] == pytest.approx(0.025)
    assert any("at-cap" in t for t in x["headline_convention_terms_absent"]) and any("per-item" in t for t in x["headline_convention_terms_absent"])
    # it is the run's headline row, on the same judgment as its two sources, and both sources stand unchanged
    run_rows = F.q(root, "SELECT rm.metric_id, rm.value, rm.k, rm.n, rm.judgment_id FROM run_metric rm JOIN run r ON r.run_id = rm.run_id "
                         "WHERE r.campaign_cell_id_legacy = ? AND r.eval_id = 'xstest' AND r.token_cap_tokens = 2048 ORDER BY rm.metric_id", (F.LORA,))
    assert [(m, v, kk, nn) for m, v, kk, nn, _ in run_rows] == [("over_refusal_graded", pytest.approx(4 / 19), None, 19),
                                                              ("over_refusal_label_mean", pytest.approx(4 / 19), 4, 19),    # the label mean (r2_export 10): k = 3 + 0.5 x 2, whole
                                                              ("over_refusal_rate_full", pytest.approx(3 / 19), 3, 19),
                                                              ("over_refusal_rate_full_plus_partial", pytest.approx(5 / 19), 5, 19),
                                                              ("xst_refused", pytest.approx(3 / 19), 3, 19)]        # the refused-rate twin of the full rate
    assert len({j for *_, j in run_rows}) == 1 and run_rows[0][4] == jid
    # one rate only (the or_bench pooled block), and the greedy runs: no derived row
    assert F.q(root, "SELECT COUNT(*) FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE rm.metric_id = 'over_refusal_graded' AND r.eval_id = 'or_bench_hard'") == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE rm.metric_id = 'over_refusal_graded' AND r.token_cap_tokens = 512") == [(0,)]
    # the export's own rows carry the builder's computation; the two derived rows the registry's, one computation each (the graded rule and the
    # label-mean rule are different statements, so r2_export 10 files a third computation row)
    assert F.q(root, "SELECT COUNT(DISTINCT computation_id) FROM run_metric") == [(3,)]


def test_graded_derivation_refuses_an_inconsistent_pair_by_name():
    """The derivation is exact or absent: the two rows must share n, judgment and threshold, carry k and n, satisfy k_fpp >= k_full and
    value = k / n; a run that already files the headline is left alone."""
    ad = X.R2ExportAdapter(data_dir="/nonexistent", rs="/nonexistent")
    def row(metric, k, n, value=None, jid="j1", thr=-1.0, rid="run-a", **extra):
        return {"_eval_id": "xstest", "run_id": rid, "judgment_id": jid, "metric_id": metric, "threshold": thr, "value": (k / n) if value is None else value, "k": k, "n": n,
                "frac_at_token_cap": 0.1, "source_uri": "~/x.json", "source_producers_json": [], "producer_metric_spelling": metric, "extra": {"row_id": 7, **extra}}
    ok = ad._graded_from_rates([row("over_refusal_rate_full", 188, 2634), row("over_refusal_rate_full_plus_partial", 707, 2634)], {"run-a": {"_n_items": 1318, "_n_items_facts": {"n_trials": 2}}})
    assert len(ok) == 1 and ok[0]["value"] == pytest.approx((188 + 0.5 * 519) / 2634) and ok[0]["n"] == 2634 and ok[0]["k"] is None
    assert ok[0]["extra"]["k_partial"] == 519 and ok[0]["extra"]["n_items"] == 1318 and ok[0]["extra"]["max_abs_deviation_from_at_cap_term"] == pytest.approx(0.05)
    assert ok[0]["_eval_id"] == "xstest" and ok[0]["computed_by"] == "registry" and ok[0]["computation_id"] is None   # stamped by the caller
    assert len(ad.refusals) == 0
    # different n
    ad.refusals = LC.RowRefusals(X.NAME)
    assert ad._graded_from_rates([row("over_refusal_rate_full", 188, 2634), row("over_refusal_rate_full_plus_partial", 707, 2633)], {}) == []
    assert len(ad.refusals) == 1 and "different label sets" in ad.refusals.items[0]["why"]
    # k_fpp < k_full
    ad.refusals = LC.RowRefusals(X.NAME)
    assert ad._graded_from_rates([row("over_refusal_rate_full", 188, 2634), row("over_refusal_rate_full_plus_partial", 100, 2634)], {}) == []
    assert "< full k" in ad.refusals.items[0]["why"]
    # a value that is not its k / n
    ad.refusals = LC.RowRefusals(X.NAME)
    assert ad._graded_from_rates([row("over_refusal_rate_full", 188, 2634, value=0.5), row("over_refusal_rate_full_plus_partial", 707, 2634)], {}) == []
    assert "not its k / n" in ad.refusals.items[0]["why"]
    # no k on a row
    ad.refusals = LC.RowRefusals(X.NAME)
    assert ad._graded_from_rates([row("over_refusal_rate_full", None, 2634, value=0.07), row("over_refusal_rate_full_plus_partial", 707, 2634)], {}) == []
    assert "no k / n" in ad.refusals.items[0]["why"]
    # one rate only, or a headline already filed: nothing derived, nothing refused
    ad.refusals = LC.RowRefusals(X.NAME)
    assert ad._graded_from_rates([row("over_refusal_rate_full", 188, 2634)], {}) == []
    assert ad._graded_from_rates([row("over_refusal_rate_full", 188, 2634), row("over_refusal_rate_full_plus_partial", 707, 2634), row("over_refusal_graded", None, 1318, value=0.17)], {}) == []
    assert len(ad.refusals) == 0


def test_the_label_mean_is_derived_from_the_two_pooled_label_rates_and_equals_the_derived_graded_row_here(tmp_path):
    """Every over-refusal source files a THIRD row, `over_refusal_label_mean` -- full_refusal 1,
    partial_refusal 0.5, full_compliance 0, a reply at the cap counted by its label, the mean over the judged items. The export carries no
    per-item labels and no label x at-cap cross-tab, only the sampled re-run's two pooled label rates, so the row is derived from the same
    k / n as the graded row -- (k_full + 0.5 x (k_fpp - k_full)) / n over the judged LABELS pooled across the two seeded trials -- under its
    own computation row, with the label-mean interval from the three counts, and it EQUALS the derived graded row here by construction
    (neither derivation has an at-cap term); the row says so. The mirror pass stamps it beside the graded row. A run filing ONE pooled rate
    (the or_bench pooled block) and the greedy `over_refusal_rate` runs (full refusals only: no partial count exists, so nothing is
    approximated) get no row and are counted in `stats.label_mean_unfiled_runs`."""
    paths, ad, seeds, root = _setup(tmp_path, overrefusal_count_rows=True)
    rep, ing, ing2 = _two_passes(root, ad)
    assert rep.groups_refreshed == 1, rep.results[0].error
    assert ing.refused + ing2.refused == 0 and len(ad.refusals) == 0, (ing.refusals, ing2.refusals, ad.refusals.items)
    assert ad.stats["label_mean_derived"] == 1 and ad.stats["graded_derived"] == 1
    # the greedy 512 xstest run, the or_bench pooled run (one rate), the at-cap-only relu store, the pending sub run: over-refusal runs with no label mean
    assert ad.stats["label_mean_unfiled_runs"] == 4
    cols = ("rm.value, rm.k, rm.n, rm.threshold, rm.computed_by, rm.producer_metric_spelling, rm.ci_lo, rm.ci_hi, rm.ci_method, rm.frac_at_token_cap, rm.extra_json, "
            "rm.judgment_id, rm.computation_id, rm.source_uri, c.algorithm, c.seed_provenance")
    rows = F.q(root, f"SELECT {cols} FROM run_metric rm JOIN computation c ON c.computation_id = rm.computation_id WHERE rm.metric_id = 'over_refusal_label_mean'")
    assert len(rows) == 1, rows
    value, k, n, thr, by, spelling, lo, hi, ci_method, frac_cap, extra, jid, cid, src, algo, rule = rows[0]
    graded = F.q(root, "SELECT value, judgment_id, computation_id, source_uri, frac_at_token_cap FROM run_metric WHERE metric_id = 'over_refusal_graded'")
    assert len(graded) == 1
    assert value == pytest.approx(4 / 19) and value == pytest.approx(graded[0][0]) and k == 4 and n == 19 and thr == -1.0 and by == "registry"
    assert (lo, hi) == pytest.approx(DC.label_mean_interval(3, 2, 19)) and ci_method == DC.OVER_REFUSAL_LABEL_MEAN_CI_METHOD
    assert spelling == X.LABEL_MEAN_SPELLING and algo == X.LABEL_MEAN_ALGORITHM and rule == X.LABEL_MEAN_RULE
    assert jid == graded[0][1] and cid != graded[0][2] and src == graded[0][3]           # the graded row's judgment and source; its own computation
    assert frac_cap == pytest.approx(0.05) and frac_cap == graded[0][4]                    # the degeneracy mirror lands on it as on the graded row
    x = json.loads(extra)
    assert x["label_counts"] == {"full": 3, "partial": 2, "compliance": 14, "n": 19, "n_abstentions": None}
    assert (x["k_effective"], x["n_labels"], x["n_items"], x["n_trials"]) == (4.0, 19, 10, 2)
    assert x["equals_derived_graded_row"] is True and "at-cap" in x["equals_derived_graded_row_note"]
    assert x["graded_metric_id"] == "over_refusal_graded" and x["refused_metric_id"] == "xst_refused"
    assert x["definition"] == DC.OVER_REFUSAL_LABEL_MEAN_DEFINITION and x["ci_text"] == DC.OVER_REFUSAL_LABEL_MEAN_CI_TEXT and x["weights"] == DC.OVER_REFUSAL_LABEL_SCORE
    assert [d["metric_id"] for d in x["derived_from"]] == ["over_refusal_rate_full", "over_refusal_rate_full_plus_partial"]
    assert [(d["k"], d["n"]) for d in x["derived_from"]] == [(3, 19), (5, 19)] and all(d["row_id"] is not None for d in x["derived_from"])
    assert "row_id" not in x and x["status_legacy"] == "graded" and x["rule"] == X.LABEL_MEAN_RULE and "LABELS" in x["unit"] and "r2_export 10" in x["filed_as"]
    # the greedy 512 xstest run (`over_refusal_rate` alone: no partial count) and the or_bench pooled run (one rate) file no label-mean row
    assert F.q(root, "SELECT COUNT(*) FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE rm.metric_id = 'over_refusal_label_mean' AND r.token_cap_tokens = 512") == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE rm.metric_id = 'over_refusal_label_mean' AND r.eval_id = 'or_bench_hard'") == [(0,)]
    # the label-mean computation row states the rule, the definition and the interval's method
    comp = F.q(root, "SELECT algorithm, version, seed_provenance, extra_json FROM computation WHERE computation_id = ?", (cid,))
    assert comp[0][:3] == (X.LABEL_MEAN_ALGORITHM, X.LABEL_MEAN_VERSION, X.LABEL_MEAN_RULE)
    cx = json.loads(comp[0][3])
    assert cx["derived_from_metrics"] == ["over_refusal_rate_full", "over_refusal_rate_full_plus_partial"] and cx["ci_method"] == DC.OVER_REFUSAL_LABEL_MEAN_CI_METHOD


def test_label_mean_derivation_files_beside_an_existing_graded_row_and_refuses_an_inconsistent_pair_by_name():
    """The label mean is a different id from the graded headline: a run that already carries `over_refusal_graded` (where the graded
    derivation stands down) still gets its label-mean row from the two pooled rates; a run that already files the label mean gets no second
    one; the pair must pass the same checks as the graded derivation (shared n, k and n present, k_fpp >= k_full, value = k / n), an
    inconsistent pair refusing the label mean by name under ITS id; one rate alone derives nothing and refuses nothing."""
    ad = X.R2ExportAdapter(data_dir="/nonexistent", rs="/nonexistent")
    def row(metric, k, n, value=None, jid="j1", thr=-1.0, rid="run-a", **extra):
        return {"_eval_id": "xstest", "run_id": rid, "judgment_id": jid, "metric_id": metric, "threshold": thr, "value": (k / n) if value is None else value, "k": k, "n": n,
                "frac_at_token_cap": 0.1, "source_uri": "~/x.json", "source_producers_json": [], "producer_metric_spelling": metric, "extra": {"row_id": 7, "status_legacy": "graded", **extra}}
    rows = [row("over_refusal_rate_full", 188, 2634), row("over_refusal_rate_full_plus_partial", 707, 2634), row("over_refusal_graded", None, 1318, value=0.17)]
    assert ad._graded_from_rates(rows, {}) == []                                             # the run files its own headline: the graded derivation stands down
    lm = ad._label_mean_from_rates(rows, {"run-a": {"_n_items": 1318, "_n_items_facts": {"n_trials": 2}}})
    assert len(lm) == 1 and lm[0]["metric_id"] == "over_refusal_label_mean" and lm[0]["value"] == pytest.approx((188 + 0.5 * 519) / 2634)
    assert lm[0]["k"] is None and lm[0]["n"] == 2634                                         # 188 + 0.5 x 519 = 447.5 is no whole count: k NULL, exact on the extra
    assert (lm[0]["ci_lo"], lm[0]["ci_hi"]) == pytest.approx(DC.label_mean_interval(188, 519, 2634)) and lm[0]["ci_method"] == DC.OVER_REFUSAL_LABEL_MEAN_CI_METHOD
    assert lm[0]["extra"]["label_counts"] == {"full": 188, "partial": 519, "compliance": 2634 - 707, "n": 2634, "n_abstentions": None}
    assert lm[0]["extra"]["k_effective"] == 447.5 and lm[0]["extra"]["n_items"] == 1318 and lm[0]["extra"]["n_trials"] == 2 and lm[0]["extra"]["refused_metric_id"] == "xst_refused"
    assert lm[0]["_eval_id"] == "xstest" and lm[0]["computed_by"] == "registry" and lm[0]["computation_id"] is None   # stamped by the caller
    assert lm[0]["frac_at_token_cap"] is None                                                # the mirror pass stamps it, the derivation does not
    assert len(ad.refusals) == 0
    # a label-mean row already filed: nothing derived, nothing refused
    assert ad._label_mean_from_rates(rows + [row("over_refusal_label_mean", None, 2634, value=0.17)], {}) == [] and len(ad.refusals) == 0
    # an inconsistent pair refuses the label mean by name, under its own id
    ad.refusals = LC.RowRefusals(X.NAME)
    assert ad._label_mean_from_rates([row("over_refusal_rate_full", 188, 2634), row("over_refusal_rate_full_plus_partial", 707, 2633)], {}) == []
    assert len(ad.refusals) == 1 and ad.refusals.items[0]["key"] == "run-a|over_refusal_label_mean"
    assert ad.refusals.items[0]["why"].startswith("over_refusal_label_mean not derived") and "different label sets" in ad.refusals.items[0]["why"]
    ad.refusals = LC.RowRefusals(X.NAME)
    assert ad._label_mean_from_rates([row("over_refusal_rate_full", None, 2634, value=0.07), row("over_refusal_rate_full_plus_partial", 707, 2634)], {}) == []
    assert "no k / n" in ad.refusals.items[0]["why"]
    # one rate only (the greedy store's shape has no pooled pair at all): nothing derived, nothing refused, never an approximation
    ad.refusals = LC.RowRefusals(X.NAME)
    assert ad._label_mean_from_rates([row("over_refusal_rate_full", 188, 2634)], {}) == [] and len(ad.refusals) == 0
    assert ad._label_mean_from_rates([row("over_refusal_rate", 22, 250)], {}) == [] and len(ad.refusals) == 0
    assert ad.stats["label_mean_derived"] == 0


def test_the_refused_rate_is_twinned_from_the_exports_own_full_refusal_rates_and_the_mirrors_do_not_depend_on_the_headline(tmp_path):
    """The export carries the refused rate under two of its own spellings -- the sampled re-run's `over_refusal_rate_full` (pooled
    judged LABELS over two trials: n = 19 labels of 10 items) and the greedy store's `over_refusal_rate` (one draw per item) -- and each is
    twinned under the registry id for the run's eval (`xst_refused` / `orb_refused`) on the same run, judgment and computation with the same
    value, k and n; the source spelling, its unit and the judged item count ride the twin's extra. A run whose only over-refusal rows are the
    at-cap decomposition or a pending rate gets no twin and is counted. The degeneracy mirror pass stamps the graded row and the twins beside
    the eval's headline, so reading under seeds whose over-refusal headline is the graded metric (the earlier pick) emits byte-identical metric rows."""
    now = "2026-09-24T12:00:00.000Z"
    paths, ad, seeds, root = _setup(tmp_path, overrefusal_count_rows=True)
    rep, ing, ing2 = _two_passes(root, ad, now=now)
    assert rep.groups_refreshed == 1, rep.results[0].error
    assert ing.refused + ing2.refused == 0 and len(ad.refusals) == 0, ad.refusals.items
    assert ad.stats["refused_rate_twins"] == 3 and ad.stats["refused_rate_untwinned_runs"] == 2       # the at-cap-only relu store, the pending sub cell
    cols = "r.campaign_cell_id_legacy, r.eval_id, r.token_cap_tokens, rm.metric_id, rm.value, rm.k, rm.n, rm.computed_by, rm.frac_at_token_cap, rm.producer_metric_spelling, rm.extra_json, rm.judgment_id, rm.computation_id"
    twins = F.q(root, f"SELECT {cols} FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE rm.metric_id IN ('xst_refused', 'orb_refused') ORDER BY r.eval_id, r.token_cap_tokens")
    assert [(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]) for t in twins] == [
        (F.LORA, "or_bench_hard", 2048, "orb_refused", pytest.approx(0.2), 4, 20, "producer"),
        (F.LORA, "xstest", 512, "xst_refused", pytest.approx(0.1), 1, 10, "producer"),
        (F.LORA, "xstest", 2048, "xst_refused", pytest.approx(3 / 19), 3, 19, "producer")]
    orb_pooled, xst_greedy, xst_pooled = twins
    assert xst_pooled[8] == pytest.approx(0.05) and xst_pooled[9] == "over_refusal_rate_full" and xst_greedy[9] == "over_refusal_rate"
    x = json.loads(xst_pooled[10])["refused_rate_twin"]
    assert x["source_metric_id"] == "over_refusal_rate_full" and (x["k"], x["n"], x["n_items"]) == (3, 19, 10) and "labels" in x["unit"] and "per ITEM" in x["convention_note"]
    xg = json.loads(xst_greedy[10])["refused_rate_twin"]
    assert xg["source_metric_id"] == "over_refusal_rate" and (xg["k"], xg["n"], xg["n_items"]) == (1, 10, 10) and "one draw per item" in xg["unit"]
    assert json.loads(orb_pooled[10])["refused_rate_twin"]["n_items"] is None       # the or_bench pooled block files no n_items: never derived from the labels
    # the twin sits on its source row's judgment and computation, beside the graded row where one exists
    src = F.q(root, "SELECT rm.metric_id, rm.judgment_id, rm.computation_id FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.campaign_cell_id_legacy = ? AND r.eval_id = 'xstest' AND r.token_cap_tokens = 2048 ORDER BY rm.metric_id", (F.LORA,))
    assert [m for m, _j, _c in src] == ["over_refusal_graded", "over_refusal_label_mean", "over_refusal_rate_full", "over_refusal_rate_full_plus_partial", "xst_refused"]
    # the label-mean row (r2_export 10) sits on the same judgment under its own registry computation, like the graded row
    assert len({j for _m, j, _c in src}) == 1 and {c for m, _j, c in src if m not in ("over_refusal_graded", "over_refusal_label_mean")} == {xst_pooled[12]}
    # the or_bench pooled run has ONE rate (no graded row derivable) and still carries its refused-rate twin
    assert F.q(root, "SELECT COUNT(*) FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.eval_id = 'or_bench_hard' AND rm.metric_id = 'over_refusal_graded'") == [(0,)]
    # the headline switch moves no bytes: the same export read under seeds whose over-refusal headline is still the graded metric (the
    # synthetic seeds follow evals.yaml, where the headline is the refused rate)
    seeds2 = F.synthetic_seeds()
    seeds2.evals["xstest"]["headline_metric_id"] = "over_refusal_graded"
    seeds2.evals["or_bench_hard"]["headline_metric_id"] = "over_refusal_graded"
    ad2, _ = _adapter(tmp_path, paths, seeds=seeds2)
    objs2 = [ad2.objects.resolve(F.VIEW, c) for c in (F.LORA, F.RELU, F.SUB, F.PROMPTED)]
    root2 = F.registry_root(tmp_path / "second", seeds2, objs2)
    rep2, ing_b, ing_b2 = _two_passes(root2, ad2, now=now)
    assert rep2.groups_refreshed == 1 and ing_b.refused + ing_b2.refused == 0
    assert F.q(root, "SELECT headline_metric_id FROM eval WHERE eval_id IN ('xstest', 'or_bench_hard') ORDER BY eval_id") == [("orb_refused",), ("xst_refused",)]
    assert F.q(root2, "SELECT headline_metric_id FROM eval WHERE eval_id IN ('xstest', 'or_bench_hard') ORDER BY eval_id") == [("over_refusal_graded",), ("over_refusal_graded",)]
    mcols = ("run_id, judgment_id, metric_id, threshold, value, k, n, ci_lo, ci_hi, frac_at_token_cap, frac_degenerate_rep_ratio_loop, frac_loop_first512, "
             "frac_defended_degenerate, n_abstentions, computed_by, computation_id, producer_metric_spelling, source_uri, extra_json")
    sql = f"SELECT {mcols} FROM run_metric ORDER BY run_id, judgment_id, metric_id, threshold"
    a, b = F.q(root, sql), F.q(root2, sql)
    assert len(a) == len(b) > 0 and a == b
    csql = "SELECT run_id, anchor_run_id, judgment_id, metric_id, threshold, delta, delta_lo, delta_hi, delta_p, n_pairs, extra_json FROM run_contrast ORDER BY 1, 2, 3, 4, 5"
    assert F.q(root, csql) == F.q(root2, csql)


def test_a_greedy_rate_row_without_k_twins_with_the_whole_count_or_a_null_k_never_a_guess():
    """78 of the export's 150 greedy `over_refusal_rate` rows carry no k (2026-09-24). The twin takes k = value x n where that is a whole
    count (0.088 x 250 = 22), files k NULL beside the producer's value and n where the rounded value makes no whole count (0.47836 x 1317),
    and refuses by name only a row with no n; a row whose k disagrees with its value is refused as before."""
    ad = X.R2ExportAdapter(data_dir="/nonexistent", rs="/nonexistent", seeds=F.synthetic_seeds())
    info = {"run_id": "run-a", "eval_id": "xstest"}
    def m_of(row):
        return {"run_id": "run-a", "judgment_id": "j", "metric_id": row["metric"], "threshold": -1.0, "computation_id": "c", "value": row["value"],
                "k": row.get("k"), "n": row.get("n"), "frac_at_token_cap": None, "extra": {"row_id": 1}}
    row = {"metric": "over_refusal_rate", "value": 0.088, "n": 250, "k": None}
    tw = ad._refused_rate_twin(m_of(row), row, info)
    assert tw["metric_id"] == "xst_refused" and tw["k"] == 22 and tw["n"] == 250 and tw["value"] == 0.088 and tw["computation_id"] == "c"
    assert tw["extra"]["refused_rate_twin"]["k_source"].startswith("value x n") and tw["extra"]["row_id"] == 1
    row = {"metric": "over_refusal_rate", "value": 0.47836, "n": 1317, "k": None}
    tw = ad._refused_rate_twin(m_of(row), dict(row), dict(info, eval_id="or_bench_hard"))
    assert tw["metric_id"] == "orb_refused" and tw["k"] is None and tw["value"] == 0.47836 and tw["extra"]["refused_rate_twin"]["k_source"].startswith("none:")
    row = {"metric": "over_refusal_rate", "value": 0.1, "n": None, "k": None}
    assert ad._refused_rate_twin(m_of(row), row, info) is None and any("no n" in str(x) for x in ad.refusals.items)
    row = {"metric": "over_refusal_rate", "value": 0.2, "n": 10, "k": 1}
    assert ad._refused_rate_twin(m_of(row), row, info) is None and any("not its k / n" in str(x) for x in ad.refusals.items)
    row = {"metric": "over_refusal_rate_full", "value": 3 / 19, "n": 19, "k": 3}
    tw = ad._refused_rate_twin(m_of(row), row, info)
    assert tw["k"] == 3 and tw["extra"]["refused_rate_twin"]["k_source"] == "the export's k" and "labels" in tw["extra"]["refused_rate_twin"]["unit"]
    assert ad._refused_rate_twin(m_of({"metric": "unsteered_refusal_rate", "value": 0.1, "n": 250, "k": 25}), {"metric": "unsteered_refusal_rate", "value": 0.1, "n": 250, "k": 25}, info) is None


# ============================================================================================
# the anchor currency through the runner, the fold and the paper frame
# ============================================================================================

def test_a_retired_own_unsteered_run_re_picks_the_same_sources_sibling_and_the_registry_files_its_unpaired_rate_difference(tmp_path):
    """The lora past-tense row's own unsteered run (the export's `none` row of the same standup) is retired by an `exclude` ruling while a
    current unsteered run of the EXACT key stands under another standup of the same source: the next pass re-reads the group (a new revision
    id over the same files: the §4 decision input moved), anchors the asr@0.5 comparison on the sibling (`sibling_none`) with the REGISTRY's
    unpaired rate difference from the row's k / n and the sibling's reading -- never the producer's paired delta, which parks on the metric
    row -- keyed on the export's computation so the paper frame shows it as the row's one contrast; the asr@0.25 comparison, whose only
    candidate has no reading at that threshold, parks by name; the old keys are withdrawn; the other comparisons stand; then quiet."""
    paths, ad, seeds, root = _setup(tmp_path)
    _two_passes(root, ad)
    lora = _steered(root, F.LORA, "pasttense", 4096)
    own = _contrast_rows(root, lora["run_id"], "asr", 0.5)
    assert len(own) == 1 and own[0]["anchor_kind"] == "same_standup_none" and own[0]["delta"] == -20.0 and own[0]["computed_by"] == "producer"
    own_none, comp_id = own[0]["anchor_run_id"], own[0]["computation_id"]
    assert comp_id == q(root, "SELECT computation_id FROM run_metric WHERE run_id = ? AND metric_id = 'asr' AND threshold = 0.5", lora["run_id"])[0][0]
    keys2 = _current_contrast_keys(root)
    sib, sib_jid = _sibling_none(root, lora, "asr", i0=300, threshold=0.5, value=0.2, k=2, n=10)
    _exclude(root, "r_synth_exclude_pasttense_none", own_none, 10)
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[2])
    assert rep.groups_refreshed == 1 and rep.anchor_decisions_changed == 1 and ing.refused + ing2.refused == 0, (rep.results[0].error, ing.refusals, ing2.refusals)
    assert ad.stats["contrasts_reanchored"] == 1 and ad.stats["contrasts_parked"] == 1 and ad.stats["reanchor_computations"] == 1
    assert ad.stats["anchor_states"] == {"own_current": 3, "sibling_same_source": 1, "no_current_anchor": 1}
    # ---- the re-picked contrast: the registry's statistic, keyed on the export's computation, the sibling's judgment from the frame
    rows = _contrast_rows(root, lora["run_id"], "asr", 0.5)
    assert sorted(r["anchor_run_id"] for r in rows) == sorted([own_none, sib])             # no run-scale row is deleted: the old row stays in the table ...
    new = next(r for r in rows if r["anchor_run_id"] == sib)
    assert new["anchor_kind"] == "sibling_none" and new["computed_by"] == "registry" and new["delta_method"] == DC.UNPAIRED_RATE_METHOD
    assert new["delta"] == pytest.approx(100 * (3 / 10 - 2 / 10)) and new["n_pairs"] == 0 and new["computation_id"] == comp_id and new["anchor_judgment_id"] == sib_jid
    ex = json.loads(new["extra"])
    assert ex["anchor_selection"]["state"] == "sibling_same_source" and ex["anchor_selection"]["own_reason"] == "excluded" and ex["anchor_selection"]["own_anchor_run_id"] == own_none
    assert ex["anchor_selection"]["candidates"] == [sib] and ex["anchor_selection"]["rule"] == AC.RULE
    assert [r[0] for r in ex["anchor_selection"]["rejected"]] == [own_none] and "exclude" in ex["anchor_selection"]["rejected"][0][1]     # the own run, at the exact key, retired
    assert ex["cross_source"] is False and ex["anchor_standup"]["view_legacy"] == "synthview_sibling" and ex["anchor_standup"]["lane_source_legacy"] == "r2_export"
    assert ex["contrast_caps"] == {"steered": 4096, "anchor": 4096} and ex["producer_delta_parked_on_metric"] is True
    assert (ex["k_steered"], ex["n_steered"], ex["k_anchor"], ex["n_anchor"], ex["paired"]) == (3, 10, 2, 10, False)
    assert ex["producer_metric_spelling"] == "asr" and ex["anchor_view_legacy"] == F.VIEW and ex["anchor_reading"]["judgment_id"] == sib_jid
    assert ex["statistic_computation_id"] != comp_id
    assert q(root, "SELECT algorithm, version FROM computation WHERE computation_id = ?", ex["statistic_computation_id"]) == [("registry.unpaired_rate_contrast", AC.RULE)]
    # ---- the producer's delta parks on the metric row beside it; the 0.25 row parks with no contrast and the reason
    blk = _metric_extra(root, lora["run_id"], "asr", 0.5)["delta_block"]
    assert blk["delta"] == -20.0 and blk["computed_by"] == "producer" and blk["anchor"]["registry_contrast_filed_against"] == sib
    assert blk["anchor"]["own_anchor_run_id"] == own_none and blk["anchor"]["state"] == "sibling_same_source" and "stands" in blk["why_not_a_contrast_row"]
    blk25 = _metric_extra(root, lora["run_id"], "asr", 0.25)["delta_block"]
    assert blk25["anchor"]["state"] == "no_current_anchor" and blk25["anchor"]["own_reason"] == "excluded" and blk25["effect_size"] == 0.2
    assert "no reading at (asr, 0.25)" in blk25["why_not_a_contrast_row"] and sib in blk25["why_not_a_contrast_row"]
    rejected25 = dict((r[0], r[1]) for r in blk25["anchor_selection"]["rejected"])
    assert set(rejected25) == {own_none, sib} and "no reading at (asr, 0.25)" in rejected25[sib] and "exclude" in rejected25[own_none]
    parks, other = _parks(ad)
    assert other == [] and len(parks) == 1 and ": excluded]" in parks[0]["why"] and "no reading at (asr, 0.25)" in parks[0]["why"]
    refusals = q(root, "SELECT json_extract(extra_json, '$.row_refusals') FROM source_status WHERE source_id = ? ORDER BY observed_utc DESC LIMIT 1", SOURCE_ID)[0][0]
    assert sum(1 for line in json.loads(refusals) if PARK_PHRASE in line) == 1
    # ---- the old keys are withdrawn (0.5: replaced by the sibling's key; 0.25: gone); the other three comparisons stand own-current
    keys3 = _current_contrast_keys(root)
    withdrawn = {json.loads(k)["threshold"] for k in keys2 - keys3 if json.loads(k)["run_id"] == lora["run_id"]}
    assert withdrawn == {0.5, 0.25} and all(json.loads(k)["anchor_run_id"] != own_none for k in keys3)
    assert {(json.loads(k)["anchor_run_id"], json.loads(k)["threshold"]) for k in keys3 if json.loads(k)["run_id"] == lora["run_id"]} == {(sib, 0.5)}
    assert len(keys3) == len(keys2) - 1
    # ---- the paper frame: ONE current contrast on the row, the sibling, the registry's delta; the 0.25 row anchors on nothing
    frame = q(root, "SELECT anchor_run_id, anchor_kind, delta, delta_method, n_current_contrasts, anchor_token_cap_tokens, cap_matched, contrast_extra_json, metric_extra_json "
                    "FROM v_paper_long WHERE run_id = ? AND metric_id = 'asr' AND threshold = 0.5", lora["run_id"])
    assert len(frame) == 1
    anchor, kind, delta, method, n_cur, anchor_cap, cap_matched, cx, mx = frame[0]
    assert (anchor, kind, n_cur, anchor_cap, cap_matched, method) == (sib, "sibling_none", 1, 4096, 1, DC.UNPAIRED_RATE_METHOD) and delta == pytest.approx(10.0)
    assert json.loads(cx)["anchor_selection"]["state"] == "sibling_same_source" and json.loads(mx)["delta_block"]["delta"] == -20.0
    assert q(root, "SELECT anchor_run_id FROM v_paper_long WHERE run_id = ? AND metric_id = 'asr' AND threshold = 0.25", lora["run_id"]) == [(None,)]
    # ---- quiet afterwards
    rep4, _, _ = F.run_and_ingest(root, ad, now=T[3])
    assert rep4.inputs_changed == 0 and rep4.groups_refreshed == 0 and rep4.anchor_decisions_changed == 0


def test_a_re_picked_petri_contrast_is_paired_over_registry_items_and_refuses_by_name_while_a_side_has_none(tmp_path):
    """`petri_score` is a mean: a re-picked contrast is the per-behavior PAIRED difference over the two runs' registry item rows
    (`behavior/scenario`, the rows `petri_stats` files on the export's run ids), never an unpaired rate from k / n (there is no count).
    The own unsteered Petri run excluded, a sibling at the exact key stands: without items on the sibling e4 refuses it by name and the
    comparison parks; with the sibling's items and none on the STEERED side the statistic cannot be formed and the comparison parks naming the
    steered run; once the steered run's items fold -- their digest is part of the decision -- the next pass files the paired statistic:
    the mean over behaviors of the mean per-scenario difference, a stratified bootstrap, a sign test, dz."""
    paths, ad, seeds, root = _setup(tmp_path)
    _two_passes(root, ad)
    lora = _steered(root, F.LORA, "petri_pooled")
    own = _contrast_rows(root, lora["run_id"], "petri_score", -1.0)
    assert len(own) == 1 and own[0]["anchor_kind"] == "same_standup_none" and own[0]["delta"] == -1.0
    own_none = own[0]["anchor_run_id"]
    lora_jid = I.judgment_id(lora["run_id"], str(lora["judge_id"]))
    sib, sib_jid = _sibling_none(root, lora, "petri_score", i0=400, threshold=-1.0, value=6.0, n=2)
    _exclude(root, "r_synth_exclude_petri_none", own_none, 20)
    # ---- (1) the sibling carries no item rows: e4 refuses it, the comparison parks by name (the producer's delta on the metric row)
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[2])
    assert rep.anchor_decisions_changed == 1 and ing.refused + ing2.refused == 0, (ing.refusals, ing2.refusals)
    assert ad.stats["contrasts_reanchored"] == 0 and ad.stats["contrasts_parked"] == 1
    blk = _metric_extra(root, lora["run_id"], "petri_score", -1.0)["delta_block"]
    assert blk["delta"] == -1.0 and blk["anchor"]["state"] == "no_current_anchor" and blk["anchor"]["own_reason"] == "excluded"
    assert sib in blk["why_not_a_contrast_row"] and "no current item rows at petri_score" in blk["why_not_a_contrast_row"]
    assert _contrast_rows(root, lora["run_id"], "petri_score", -1.0) == own                      # the old row stays; its key is withdrawn
    assert not any(json.loads(k)["run_id"] == lora["run_id"] for k in _current_contrast_keys(root))
    assert q(root, "SELECT anchor_run_id FROM v_paper_long WHERE run_id = ? AND metric_id = 'petri_score'", lora["run_id"]) == [(None,)]
    # the second comparison (e4 asked) is in the decision set beside the first: both parked, the pass re-reads when either moves
    _rev, extra = head_of(root, SOURCE_ID)
    petri_cmps = [e for e in extra["anchor_decisions"]["comparisons"] if e["comparison"].get("metric_id") == "petri_score"]
    assert [c["comparison"]["needs_items"] for c in petri_cmps] == [False, True]
    assert [c["decision"]["anchor_run_id"] for c in petri_cmps] == [sib, AC.PARKED]      # picked without e4; e4 asked of a mean, the sibling has no items -> parked
    # ---- (2) the sibling's items fold; the steered run has none yet: the statistic cannot be formed -> parked naming the steered run
    _items(root, sib, sib_jid, [("defer_to_users/s1", 5.5), ("defer_to_users/s2", 6.0)], 30)
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[3])
    assert rep.anchor_decisions_changed == 1 and ing.refused + ing2.refused == 0, (ing.refusals, ing2.refusals)
    assert ad.stats["contrasts_reanchored"] == 0 and ad.stats["contrasts_parked"] == 1
    blk = _metric_extra(root, lora["run_id"], "petri_score", -1.0)["delta_block"]
    assert blk["anchor"]["state"] == "sibling_same_source" and blk["anchor"]["anchor_run_id"] == sib and blk["anchor"]["registry_contrast_filed_against"] is None
    assert f"the steered run {lora['run_id']} carries no current item rows" in blk["why_not_a_contrast_row"]
    assert q(root, "SELECT COUNT(*) FROM run_contrast WHERE run_id = ? AND anchor_run_id = ?", lora["run_id"], sib) == [(0,)]
    # ---- (3) the steered run's items fold (its digest is part of the decision): the paired statistic files
    _items(root, lora["run_id"], lora_jid, [("defer_to_users/s1", 4.0), ("defer_to_users/s2", 5.0)], 40)
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[4])
    assert rep.anchor_decisions_changed == 1 and ing.refused + ing2.refused == 0 and ing.blocked + ing2.blocked == 0, (ing.refusals, ing2.refusals)
    assert ad.stats["contrasts_reanchored"] == 1 and ad.stats["contrasts_parked"] == 0 and ad.stats["reanchor_computations"] == 1
    rows = _contrast_rows(root, lora["run_id"], "petri_score", -1.0)
    new = next(r for r in rows if r["anchor_run_id"] == sib)
    assert new["anchor_kind"] == "sibling_none" and new["computed_by"] == "registry" and new["delta_method"] == DC.PAIRED_ITEMS_METHOD
    assert new["delta"] == pytest.approx(((4.0 - 5.5) + (5.0 - 6.0)) / 2) and new["n_pairs"] == 2 and new["anchor_judgment_id"] == sib_jid
    ex = json.loads(new["extra"])
    assert ex["paired"] is True and ex["weighting"] == "mean_of_behavior_means" and ex["per_behavior"]["defer_to_users"]["n_shared"] == 2
    assert ex["anchor_selection"]["state"] == "sibling_same_source" and ex["anchor_selection"]["own_reason"] == "excluded"
    assert q(root, "SELECT algorithm FROM computation WHERE computation_id = ?", ex["statistic_computation_id"]) == [("registry.paired_items_contrast",)]
    blk = _metric_extra(root, lora["run_id"], "petri_score", -1.0)["delta_block"]
    assert blk["delta"] == -1.0 and blk["anchor"]["registry_contrast_filed_against"] == sib
    frame = q(root, "SELECT anchor_run_id, anchor_kind, delta, n_current_contrasts FROM v_paper_long WHERE run_id = ? AND metric_id = 'petri_score'", lora["run_id"])
    assert len(frame) == 1 and frame[0][:2] == (sib, "sibling_none") and frame[0][2] == pytest.approx(-1.25) and frame[0][3] == 1
    rep5, _, _ = F.run_and_ingest(root, ad, now=T[5])
    assert rep5.groups_refreshed == 0 and rep5.anchor_decisions_changed == 0


def test_a_half_dose_row_anchors_on_the_same_sources_sibling_and_its_full_dose_contrast_follows_the_1x_anchors_currency(tmp_path):
    """The half-dose standup carries no unsteered row of its own (today's "no measured unsteered row" park): the search still runs and the
    main standup's unsteered run -- the same source, another standup, the exact key -- anchors it as `sibling_none` with the registry's unpaired
    rate difference (the producer's delta parks). Its `full_dose` contrast against the same object's 1x run checks that run's currency:
    parked on the first pass (not folded), filed on the second, parked again -- its key withdrawn -- when an `exclude` ruling retires the 1x run."""
    paths, ad, seeds, root = _setup(tmp_path, halfdose_rows=True)
    rep1, ing1, ing1b = F.run_and_ingest(root, ad, now=T[0])
    assert rep1.groups_refreshed == 1 and ing1.refused + ing1b.refused == 0, (rep1.results[0].error, ing1.refusals, ing1b.refusals)
    halves = q(root, "SELECT run_id FROM run WHERE view_legacy = ? AND campaign_cell_id_legacy = ?", f"{F.VIEW}_halfdose", F.LORA)
    assert len(halves) == 1
    half_id = halves[0][0]
    x1 = _metric_extra(root, half_id, "asr", 0.5)
    assert x1["delta_block"]["anchor"]["own_reason"] == "none_declared" and x1["delta_block"]["anchor"]["state"] == "no_current_anchor" and x1["delta_block"]["delta"] == -12.0
    assert x1[X.FULL_DOSE_BLOCK]["anchor"]["own_reason"] == "not_in_frame" and x1[X.FULL_DOSE_BLOCK]["delta"] == 10.0
    assert ad.stats["anchor_states"] == {"no_current_anchor": 6, "full_dose:no_current_anchor": 1} and ad.stats["contrasts_parked"] == 7
    # ---- pass 2: the main standup's unsteered run is current -> sibling_none for the half-dose row; the 1x run is current -> full_dose filed
    rep2, ing2, ing2b = F.run_and_ingest(root, ad, now=T[1])
    assert rep2.anchor_decisions_changed == 1 and ing2.refused + ing2b.refused == 0, (ing2.refusals, ing2b.refusals)
    assert ad.stats["anchor_states"] == {"own_current": 5, "sibling_same_source": 1, "full_dose:own_current": 1} and ad.stats["contrasts_reanchored"] == 1
    main_none = _contrast_rows(root, q(root, "SELECT run_id FROM run WHERE view_legacy = ? AND campaign_cell_id_legacy = ? AND eval_id = 'pasttense' AND token_cap_tokens = 4096",
                                       F.VIEW, F.LORA)[0][0], "asr", 0.5)[0]["anchor_run_id"]
    one_x = q(root, "SELECT run_id FROM run WHERE view_legacy = ? AND campaign_cell_id_legacy = ? AND eval_id = 'pasttense' AND token_cap_tokens = 4096", F.VIEW, F.LORA)[0][0]
    rows = {r["anchor_kind"]: r for r in _contrast_rows(root, half_id, "asr", 0.5)}
    assert set(rows) == {"sibling_none", "full_dose"}
    sib = rows["sibling_none"]
    assert sib["anchor_run_id"] == main_none and sib["computed_by"] == "registry" and sib["delta"] == pytest.approx(100 * (4 / 10 - 5 / 10))
    assert json.loads(sib["extra"])["anchor_selection"]["own_reason"] == "none_declared" and json.loads(sib["extra"])["anchor_standup"]["view_legacy"] == F.VIEW
    fd = rows["full_dose"]
    assert fd["anchor_run_id"] == one_x and fd["delta"] == 10.0 and fd["computed_by"] == "producer"
    fsel = json.loads(fd["extra"])["anchor_selection"]
    assert fsel["state"] == "own_current" and fsel["own_reason"] == "current" and fsel["own_anchor_run_id"] == one_x and fsel["currency_only"] and fsel["anchor_kind"] == "full_dose"
    x2 = _metric_extra(root, half_id, "asr", 0.5)
    assert x2[X.FULL_DOSE_BLOCK] is None and x2["delta_block"]["delta"] == -12.0 and x2["delta_block"]["anchor"]["registry_contrast_filed_against"] == main_none
    frame = q(root, "SELECT anchor_run_id, anchor_kind, n_current_contrasts FROM v_paper_long WHERE run_id = ? AND metric_id = 'asr' AND threshold = 0.5 ORDER BY anchor_kind", half_id)
    assert frame == [(one_x, "full_dose", 2), (main_none, "sibling_none", 2)]           # the two legitimate kinds of one half-dose key
    # ---- the 1x run excluded: the full_dose comparison parks, its key is withdrawn, the sibling contrast stands
    keys2 = _current_contrast_keys(root)
    _exclude(root, "r_synth_exclude_1x", one_x, 50)
    rep3, ing3, ing3b = F.run_and_ingest(root, ad, now=T[2])
    assert rep3.anchor_decisions_changed == 1 and ing3.refused + ing3b.refused == 0, (ing3.refusals, ing3b.refusals)
    assert ad.stats["anchor_states"]["full_dose:no_current_anchor"] == 1 and ad.stats["contrasts_parked"] == 1
    x3 = _metric_extra(root, half_id, "asr", 0.5)
    assert x3[X.FULL_DOSE_BLOCK]["delta"] == 10.0 and x3[X.FULL_DOSE_BLOCK]["anchor"]["own_reason"] == "excluded" and "exclude" in x3[X.FULL_DOSE_BLOCK]["why_not_a_contrast_row"]
    keys3 = _current_contrast_keys(root)
    gone = [json.loads(k) for k in keys2 - keys3]
    assert len(gone) == 1 and gone[0]["run_id"] == half_id and gone[0]["anchor_run_id"] == one_x
    assert q(root, "SELECT anchor_run_id, anchor_kind, n_current_contrasts FROM v_paper_long WHERE run_id = ? AND metric_id = 'asr' AND threshold = 0.5", half_id) == [(main_none, "sibling_none", 1)]
    parks, other = _parks(ad)
    assert other == [] and len(parks) == 1 and parks[0]["why"].startswith("full-dose contrast's 1x anchor") and "excluded" in parks[0]["why"]
    rep4, _, _ = F.run_and_ingest(root, ad, now=T[3])
    assert rep4.groups_refreshed == 0 and rep4.anchor_decisions_changed == 0


def test_the_refused_rate_twins_contrast_is_its_own_comparison_decided_against_the_twin_metrics_reading(tmp_path):
    """The refused-rate twin (`xst_refused`) inherited the SOURCE metric's anchor decision. An export
    `over_refusal_rate` row whose own unsteered run retires can have an eligible alternative at the exact key that carries `xst_refused` and
    not `over_refusal_rate` (live: every other over-refusal source's unsteered runs file the refused rate, none the export's spelling) -- the
    source comparison parks, and the twin's contrast vanished with it although a current anchor at the twin's id stood. Now the twin's contrast
    is a comparison of its own: decided at the twin's id (its own entry in the decision set), filed own-current beside the source's in the
    steady state, and -- the own anchor excluded, a twin-only sibling standing -- re-picked with the registry's unpaired rate difference from the
    TWIN's k / n and the sibling's `xst_refused` reading, while the source comparison parks by name; then quiet."""
    paths, ad, seeds, root = _setup(tmp_path, overrefusal_contrast_rows=True)
    _two_passes(root, ad)
    lora = _steered(root, F.LORA, "xstest", 512)
    comp_id = q(root, "SELECT computation_id FROM run_metric WHERE run_id = ? AND metric_id = 'over_refusal_rate'", lora["run_id"])[0][0]
    # ---- the steady state: the source row and its twin each carry an own-current contrast against the export's own unsteered xstest run,
    # each decided at ITS metric id (two entries of the decision set), the producer's delta on both; the own anchor's reading at the twin's id
    # is the twin the adapter files on the unsteered row
    src = _contrast_rows(root, lora["run_id"], "over_refusal_rate", -1.0)
    twin = _contrast_rows(root, lora["run_id"], "xst_refused", -1.0)
    assert len(src) == 1 and len(twin) == 1 and src[0]["anchor_run_id"] == twin[0]["anchor_run_id"] and src[0]["anchor_kind"] == twin[0]["anchor_kind"] == "same_standup_none"
    own_none = src[0]["anchor_run_id"]
    assert twin[0]["delta"] == src[0]["delta"] == -20.0 and twin[0]["computed_by"] == "producer" and twin[0]["computation_id"] == comp_id
    tx = json.loads(twin[0]["extra"])
    assert tx["refused_rate_twin_of"] == "over_refusal_rate" and tx["anchor_selection"]["state"] == "own_current" and tx["anchor_selection"]["reading"]["computation_id"] == comp_id
    assert "refused_rate_twin_of" not in json.loads(src[0]["extra"])
    assert q(root, "SELECT k, n FROM run_metric WHERE run_id = ? AND metric_id = 'xst_refused'", own_none) == [(3, 10)]
    assert ad.stats["anchor_states"] == {"own_current": 6, "refused_rate_twin:own_current": 1} and ad.stats["contrasts_parked"] == 0
    assert _metric_extra(root, lora["run_id"], "xst_refused", -1.0)["delta_block"] is None and _metric_extra(root, lora["run_id"], "over_refusal_rate", -1.0)["delta_block"] is None
    _rev, extra = head_of(root, SOURCE_ID)
    cmps = {e["comparison"]["metric_id"]: e["decision"] for e in extra["anchor_decisions"]["comparisons"] if e["comparison"].get("run_id") == lora["run_id"]}
    assert set(cmps) == {"over_refusal_rate", "xst_refused"} and all(d["anchor_run_id"] == own_none and d["state"] == "own_current" for d in cmps.values())
    assert cmps["xst_refused"]["reading"] == {"value": pytest.approx(0.3), "k": 3, "n": 10, "ci_lo": None, "ci_hi": None}
    # ---- a sibling at the exact key carrying the refused rate ONLY (no reading at the export's spelling); the own unsteered run excluded
    keys2 = _current_contrast_keys(root)
    sib, sib_jid = _sibling_none(root, lora, "xst_refused", i0=500, threshold=-1.0, value=0.2, k=2, n=10)
    _exclude(root, "r_synth_exclude_xstest_none", own_none, 60)
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[2])
    assert rep.groups_refreshed == 1 and rep.anchor_decisions_changed == 1 and ing.refused + ing2.refused == 0, (rep.results[0].error, ing.refusals, ing2.refusals)
    assert ad.stats["contrasts_reanchored"] == 1 and ad.stats["contrasts_parked"] == 1 and ad.stats["reanchor_computations"] == 1
    assert ad.stats["anchor_states"] == {"own_current": 5, "no_current_anchor": 1, "refused_rate_twin:sibling_same_source": 1}
    # the source comparison parks by name: the sibling has no reading at the export's spelling
    blk = _metric_extra(root, lora["run_id"], "over_refusal_rate", -1.0)["delta_block"]
    assert blk["delta"] == -20.0 and blk["anchor"]["state"] == "no_current_anchor" and blk["anchor"]["own_reason"] == "excluded"
    assert "no reading at (over_refusal_rate" in blk["why_not_a_contrast_row"] and sib in blk["why_not_a_contrast_row"]
    assert "refused_rate_twin_of" not in blk["producer_contrast_extra"]
    # the twin's comparison re-picks the sibling: the registry's unpaired rate difference from the TWIN's k / n (1 / 10) and the sibling's
    # xst_refused reading (2 / 10), keyed on the export's computation, the sibling's judgment from the frame
    rows = _contrast_rows(root, lora["run_id"], "xst_refused", -1.0)
    assert sorted(r["anchor_run_id"] for r in rows) == sorted([own_none, sib])
    new = next(r for r in rows if r["anchor_run_id"] == sib)
    assert new["anchor_kind"] == "sibling_none" and new["computed_by"] == "registry" and new["delta_method"] == DC.UNPAIRED_RATE_METHOD
    assert new["delta"] == pytest.approx(100 * (1 / 10 - 2 / 10)) and new["n_pairs"] == 0 and new["computation_id"] == comp_id and new["anchor_judgment_id"] == sib_jid
    ex = json.loads(new["extra"])
    assert ex["refused_rate_twin_of"] == "over_refusal_rate" and ex["producer_metric_spelling"] == "over_refusal_rate"
    assert ex["anchor_selection"]["state"] == "sibling_same_source" and ex["anchor_selection"]["own_reason"] == "excluded" and ex["anchor_selection"]["candidates"] == [sib]
    assert (ex["k_steered"], ex["n_steered"], ex["k_anchor"], ex["n_anchor"], ex["paired"]) == (1, 10, 2, 10, False) and ex["anchor_reading"]["judgment_id"] == sib_jid
    assert q(root, "SELECT algorithm FROM computation WHERE computation_id = ?", ex["statistic_computation_id"]) == [("registry.unpaired_rate_contrast",)]
    tblk = _metric_extra(root, lora["run_id"], "xst_refused", -1.0)["delta_block"]
    assert tblk["delta"] == -20.0 and tblk["anchor"]["registry_contrast_filed_against"] == sib and tblk["anchor"]["state"] == "sibling_same_source"
    assert tblk["producer_contrast_extra"]["refused_rate_twin_of"] == "over_refusal_rate" and "stands" in tblk["why_not_a_contrast_row"]
    # one park line, the source's (keyed on the row); the twin's comparison is not refused
    parks, other = _parks(ad)
    assert other == [] and len(parks) == 1 and "|over_refusal_rate|" in parks[0]["key"] and not parks[0]["key"].endswith("|xst_refused")
    assert "refused-rate twin" not in parks[0]["why"]
    # the keys: both (run, own none) keys withdrawn; the twin's new key the row's only current contrast
    keys3 = _current_contrast_keys(root)
    gone = {(json.loads(k)["metric_id"], json.loads(k)["anchor_run_id"]) for k in keys2 - keys3 if json.loads(k)["run_id"] == lora["run_id"]}
    assert gone == {("over_refusal_rate", own_none), ("xst_refused", own_none)}
    assert {(json.loads(k)["metric_id"], json.loads(k)["anchor_run_id"]) for k in keys3 if json.loads(k)["run_id"] == lora["run_id"]} == {("xst_refused", sib)}
    # the paper frame: the eval's headline (xst_refused) keeps ONE current contrast -- the sibling, the registry's delta; the source spelling anchors on nothing
    frame = q(root, "SELECT anchor_run_id, anchor_kind, delta, n_current_contrasts FROM v_paper_long WHERE run_id = ? AND metric_id = 'xst_refused'", lora["run_id"])
    assert len(frame) == 1 and frame[0][:2] == (sib, "sibling_none") and frame[0][2] == pytest.approx(-10.0) and frame[0][3] == 1
    assert q(root, "SELECT anchor_run_id FROM v_paper_long WHERE run_id = ? AND metric_id = 'over_refusal_rate'", lora["run_id"]) == [(None,)]
    # the decision set: the row's two comparisons stay distinct entries -- the source parked, the twin on the sibling with ITS reading
    _rev, extra = head_of(root, SOURCE_ID)
    cmps = {e["comparison"]["metric_id"]: e["decision"] for e in extra["anchor_decisions"]["comparisons"] if e["comparison"].get("run_id") == lora["run_id"]}
    assert cmps["over_refusal_rate"]["anchor_run_id"] == AC.PARKED and cmps["xst_refused"]["anchor_run_id"] == sib
    assert cmps["xst_refused"]["reading"] == {"value": pytest.approx(0.2), "k": 2, "n": 10, "ci_lo": None, "ci_hi": None}
    # ---- quiet afterwards
    rep4, _, _ = F.run_and_ingest(root, ad, now=T[3])
    assert rep4.inputs_changed == 0 and rep4.groups_refreshed == 0 and rep4.anchor_decisions_changed == 0


def test_an_11_head_that_parked_the_source_and_recorded_no_twin_comparison_is_re_read_by_the_bump_alone(tmp_path):
    """The twin comparison landed with `EMISSION_VERSION` left at 11, and the runner replays
    only the comparisons a head RECORDED -- a head emitted under 11 that parked the source comparison while an eligible sibling carried
    only `xst_refused` recorded no twin comparison, so the replay of its unchanged source comparison matched its own fingerprint, the group
    planned quiet and the refused-rate contrast stayed absent. The lever is the bump: 12 re-reads every quiet group whose head was
    minted under 11, with no export file and no anchor-currency change. Here the 11-shaped head is emitted under the parent commit's code shape
    (the instance's version 11; the twin's contrast the source's relabeled, so a parked source files none and logs no twin comparison), the
    finding is reproduced with the twin code live and the version still 11 (quiet; no twin contrast), then the class's 12 re-reads the group
    by the version alone and the twin's contrast on the sibling appears; then quiet."""
    paths, ad, seeds, root = _setup(tmp_path, overrefusal_contrast_rows=True)
    _two_passes(root, ad)
    lora = _steered(root, F.LORA, "xstest", 512)
    own_none = _contrast_rows(root, lora["run_id"], "over_refusal_rate", -1.0)[0]["anchor_run_id"]
    assert ad.stats["anchor_states"] == {"own_current": 6, "refused_rate_twin:own_current": 1}     # the fixture's one twin comparison is this row's

    def lora_keys():
        return {(json.loads(k)["metric_id"], json.loads(k)["anchor_run_id"]) for k in _current_contrast_keys(root) if json.loads(k)["run_id"] == lora["run_id"]}

    def twin_anchors():
        return sorted(r["anchor_run_id"] for r in _contrast_rows(root, lora["run_id"], "xst_refused", -1.0))

    def lora_decisions():
        _rev, extra = head_of(root, SOURCE_ID)
        return {e["comparison"]["metric_id"]: e["decision"] for e in extra["anchor_decisions"]["comparisons"] if e["comparison"].get("run_id") == lora["run_id"]}

    assert lora_keys() == {("over_refusal_rate", own_none), ("xst_refused", own_none)}
    # a sibling at the exact key carrying the refused rate ONLY; the own unsteered run excluded -- the source comparison must park
    sib, sib_jid = _sibling_none(root, lora, "xst_refused", i0=500, threshold=-1.0, value=0.2, k=2, n=10)
    _exclude(root, "r_synth_exclude_xstest_none", own_none, 60)
    # ---- the 11-shaped head: the parent commit's twin followed the source's decision (its contrast the source's relabeled when the source
    # filed, nothing when it parked, no comparison of its own in the decision log), minted under 11
    source_only = ad._contrast_payload

    def twin_follows_the_source(*args, twin=None, **kw):
        return None if twin is not None else source_only(*args, **kw)

    ad._contrast_payload, ad.EMISSION_VERSION = twin_follows_the_source, 11
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[2])
    assert rep.groups_refreshed == 1 and rep.anchor_decisions_changed == 1 and ing.refused + ing2.refused == 0, (rep.results[0].error, ing.refusals, ing2.refusals)
    assert ad.stats["anchor_states"] == {"own_current": 5, "no_current_anchor": 1} and ad.stats["contrasts_parked"] == 1
    rev11, extra11 = head_of(root, SOURCE_ID)
    assert extra11["emission_version"] == "11"
    assert set(lora_decisions()) == {"over_refusal_rate"} and lora_decisions()["over_refusal_rate"]["anchor_run_id"] == AC.PARKED
    assert twin_anchors() == [own_none] and lora_keys() == set()               # pass 2's own-none rows stand withdrawn; nothing on the sibling
    assert _metric_extra(root, lora["run_id"], "xst_refused", -1.0)["delta_block"]["anchor"]["state"] == "no_current_anchor"    # the source's park, copied
    assert q(root, "SELECT anchor_run_id FROM v_paper_long WHERE run_id = ? AND metric_id = 'xst_refused'", lora["run_id"]) == [(None,)]
    # ---- the finding: the twin code live, the version still 11 -- the head's recorded comparisons replay to the same decisions, no export
    # file moved, nothing re-reads the group, and the refused-rate contrast stays absent
    del ad._contrast_payload
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[3])
    assert rep.groups_refreshed == 0 and rep.inputs_changed == 0 and rep.anchor_decisions_changed == 0 and ing.refused + ing2.refused == 0
    assert head_of(root, SOURCE_ID)[0] == rev11 and twin_anchors() == [own_none] and lora_keys() == set()
    # ---- the fix: the class's 12 re-reads the group by the version alone (no file input changed, no decision moved) and the twin's
    # comparison is made for the first time -- re-picked onto the sibling with the registry's unpaired rate difference from the twin's count
    del ad.EMISSION_VERSION
    assert X.R2ExportAdapter.EMISSION_VERSION == 12 and ad.EMISSION_VERSION == 12
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=T[4])
    assert rep.groups_refreshed == 1 and rep.inputs_changed == 0 and rep.anchor_decisions_changed == 0 and ing.refused + ing2.refused == 0, (rep.results[0].error, ing.refusals, ing2.refusals)
    assert ad.stats["anchor_states"] == {"own_current": 5, "no_current_anchor": 1, "refused_rate_twin:sibling_same_source": 1}
    assert ad.stats["contrasts_reanchored"] == 1 and ad.stats["contrasts_parked"] == 1
    assert twin_anchors() == sorted([own_none, sib])
    new = next(r for r in _contrast_rows(root, lora["run_id"], "xst_refused", -1.0) if r["anchor_run_id"] == sib)
    assert new["anchor_kind"] == "sibling_none" and new["computed_by"] == "registry" and new["delta_method"] == DC.UNPAIRED_RATE_METHOD
    assert new["delta"] == pytest.approx(100 * (1 / 10 - 2 / 10)) and new["anchor_judgment_id"] == sib_jid
    assert json.loads(new["extra"])["refused_rate_twin_of"] == "over_refusal_rate"
    assert _metric_extra(root, lora["run_id"], "xst_refused", -1.0)["delta_block"]["anchor"]["registry_contrast_filed_against"] == sib
    rev12, extra12 = head_of(root, SOURCE_ID)
    assert rev12 != rev11 and extra12["emission_version"] == "12"
    cmps = lora_decisions()
    assert set(cmps) == {"over_refusal_rate", "xst_refused"} and cmps["over_refusal_rate"]["anchor_run_id"] == AC.PARKED and cmps["xst_refused"]["anchor_run_id"] == sib
    assert lora_keys() == {("xst_refused", sib)}
    assert q(root, "SELECT anchor_run_id, anchor_kind, n_current_contrasts FROM v_paper_long WHERE run_id = ? AND metric_id = 'xst_refused'", lora["run_id"]) == [(sib, "sibling_none", 1)]
    assert q(root, "SELECT anchor_run_id FROM v_paper_long WHERE run_id = ? AND metric_id = 'over_refusal_rate'", lora["run_id"]) == [(None,)]
    # ---- quiet afterwards
    rep, _, _ = F.run_and_ingest(root, ad, now=T[5])
    assert rep.inputs_changed == 0 and rep.groups_refreshed == 0 and rep.anchor_decisions_changed == 0


def test_a_twin_without_a_count_parks_its_re_picked_comparison_by_name_never_from_the_sources_count():
    """A greedy `over_refusal_rate` row whose rounded value makes no whole count twins with k NULL (never a guess). Its comparison is decided
    at the TWIN's id with the TWIN's count: against a current sibling at the exact key that carries `xst_refused` only, the source comparison
    parks (no reading at the export's spelling) and the twin's parks by name on its own missing count -- never an unpaired difference from the
    source spelling's k, nor from a value alone -- and the two are distinct entries of the decision log, the twin's on the sibling's reading."""
    fr = Frame()
    sib = fr.run("none:sib", eval_id="xstest", source="r2_export", standup="tagB", reading=False)
    sib_jid = fr.judgment(sib, judge=JUDGE)
    fr.metric(sib, sib_jid, metric="xst_refused", thr=-1.0, k=2, n=10, value=0.2)
    frame = fr.load()
    frame.decision_log = []
    ad = X.R2ExportAdapter(data_dir="/nonexistent", rs="/nonexistent", seeds=F.synthetic_seeds())
    ad._anchor_states, ad._reanchor_comps, ad._contrasts_parked, ad._contrasts_reanchored = {}, {}, 0, 0
    row = {"metric": "over_refusal_rate", "value": 0.47836, "n": 1317, "k": None, "threshold": None, "bon_n": None, "pass": None,
           "delta": -5.0, "delta_lo": -9.0, "delta_hi": -1.0, "delta_p": 0.1, "delta_method": "paired (producer)", "effect_size": None, "effect_kind": None,
           "effect_lo": None, "effect_hi": None, "anchor_view": None, "anchor_root": None, "anchor_engine": None, "pairing_allowance": None}
    info = {"run_id": "steered:1", "eval_id": "xstest", "eval_legacy": "xstest", "cell_id": "obj", "cap": CAP, "judge_id": JUDGE, "view": "tagA",
            "instrument_row": {"instrument_id": INST}, "key": "tagA|obj|xstest|over_refusal_rate|None|512|None|bon_n=None"}
    runs = {"steered:1": {"model_id": MODEL, "token_cap_tokens": CAP, "trial_index": TRIAL, "pass_label": None, "decode_policy_id": None}}
    m = {"run_id": "steered:1", "judgment_id": "j", "metric_id": "over_refusal_rate", "threshold": -1.0, "computation_id": "c", "value": 0.47836, "k": None, "n": 1317, "extra": {}}
    tw = ad._refused_rate_twin(m, row, info)
    assert tw["metric_id"] == "xst_refused" and tw["k"] is None and tw["n"] == 1317
    ctx = SimpleNamespace(anchor_currency=lambda: frame, producer={"code_sha": "unverified"})
    src = ad._contrast_payload(row, info, "c", {}, None, "recorded", ctx, runs)
    twin = ad._contrast_payload(row, info, "c", {}, None, "recorded", ctx, runs, twin=tw)
    # the source: no current anchor -- the sibling carries no reading at the export's spelling
    assert src["contrast"] is None and src["park"]["anchor"]["state"] == "no_current_anchor" and src["park"]["anchor"]["own_reason"] == "none_declared"
    assert "no reading at (over_refusal_rate" in src["park"]["why_not_a_contrast_row"] and "refused_rate_twin_of" not in src["park"]["producer_contrast_extra"]
    # the twin: the sibling IS its anchor at xst_refused, but the twin files no count, so the registry's statistic is not derived -- parked by name
    assert twin["contrast"] is None and twin["park"]["anchor"]["state"] == "sibling_same_source" and twin["park"]["anchor"]["anchor_run_id"] == sib
    assert twin["park"]["anchor"]["anchor_kind"] == "sibling_none" and twin["park"]["anchor"]["registry_contrast_filed_against"] is None
    assert "steered reading (k) carries no count" in twin["park"]["why_not_a_contrast_row"] and twin["park"]["delta"] == -5.0
    assert twin["park"]["producer_contrast_extra"]["refused_rate_twin_of"] == "over_refusal_rate" and twin["park"]["anchor_selection"]["candidates"] == [sib]
    assert "steered reading (k) carries no count" in twin["refusal"] and sib in twin["refusal"]
    assert ad._anchor_states == {"no_current_anchor": 1, "refused_rate_twin:sibling_same_source": 1} and ad._contrasts_parked == 2 and ad._reanchor_comps == {}
    # two entries of the decision log, one per metric id; the twin's decision carries the sibling's xst_refused reading
    by_metric = {e["comparison"]["metric_id"]: e for e in frame.decision_log}
    assert set(by_metric) == {"over_refusal_rate", "xst_refused"} and all(e["comparison"]["run_id"] == "steered:1" for e in frame.decision_log)
    assert by_metric["over_refusal_rate"]["decision"]["anchor_run_id"] == AC.PARKED
    assert by_metric["xst_refused"]["decision"]["anchor_run_id"] == sib and by_metric["xst_refused"]["decision"]["reading"]["k"] == 2
    assert by_metric["xst_refused"]["decision"]["anchor_judgment_id"] == sib_jid


def test_the_smoke_names_the_unavailable_anchor_currency_for_the_export_group(tmp_path):
    """`declared_common.smoke` reads without a registry connection: a group that intends a contrast cannot ask the pass's currency, and
    the smoke names it per group instead of crashing (the runner turns the same exception into a refusal of the group)."""
    paths, ad, seeds, root = _setup(tmp_path)
    out = DC.smoke(ad)
    assert out["groups"] == 1 and out["anchor_unavailable_groups"] == 1 and len(out["refused_groups"]) == 1
    assert out["refused_groups"][0].startswith("r2: anchor currency unavailable") and "no registry connection" in out["refused_groups"][0]
