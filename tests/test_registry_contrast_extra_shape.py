"""The shared emitter's anchor hook (`declared_common.RowEmitter._anchor_contrast`) and the
`run_contrast.extra` conventions it files (`docs/registry/records.md`): `extra.anchor_selection` on every declared contrast (deterministic,
no timestamp, the closed states), the park on the metric row's `extra.delta_block` when no current anchor exists at the exact key or when the
family files no statistic against a re-picked one, the registry-derived unpaired rate difference (`REANCHOR_STATISTIC = unpaired_rates`)
with `contrast_caps` / `anchor_standup` / `cross_source`, keyed on the steered reading's `computation_id` with its own statistic `computation`
row named by `extra.statistic_computation_id`, the refusals by name (a side with no count; a
current unsteered run only under another instrument), the own anchor's DECLARED judgment (a second judgment under the same judge is
the adapter's pairing, not an ambiguity), the two statistic helpers, the smoke's named refusal without a registry connection, and the spelling
`cross_source` (never `cross_server`) under `src/` and `docs/`. Rows are synthetic (`declared_common.blank_row`); frames are the sidecar-shaped
in-memory `Frame`. Nothing here reads a real store.
"""
from __future__ import annotations

import json
import math
import os
import re

import pytest

from sisteer.registry import identity as I
from sisteer.registry import validate as VAL
from sisteer.registry import vocab as V
from sisteer.registry.adapters import anchor_current as AC
from sisteer.registry.adapters import declared_common as DC
from sisteer.registry.adapters.base import AdapterContext, RecordBatch

import _anchor_frame_fixtures as AF  # noqa: E402
import registry_declared_fixtures as F  # noqa: E402
from test_registry_anchor_current import Frame  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NOW = "2026-10-03T00:00:00.000Z"
PRODUCER = {"script": "tests.contrast_extra", "code_sha": "a" * 40, "session": None, "host": "TESTHOST", "substrate": "grading_host"}
INST = {"token_cap_tokens": 4096, "serving_path": "lens", "decode_source": "greedy"}
JUDGE = DC.judge_fields(**DC.STRONGREJECT_JUDGE)
STEERED_SHA = "a1" * 32
#: the emitter's keys for the two rows (`_key`); the steered row names the anchor's as its `_anchor_key`
ANCHOR_KEY, STEERED_KEY = "s|gemma4e4b|gemma4e4b_it|none|pasttense", "s|gemma4e4b|gemma4e4b_it|lora|pasttense"


def _ctx(conn=None) -> AdapterContext:
    return AdapterContext(root="", now=NOW, producer=dict(PRODUCER), conn=conn)


def _rows(*, k_s: int = 2, n: int = 10, k_a: int = 3, delta: float = -10.0, judge=JUDGE, view: str = "gemma4e4b_it"):
    """A steered static reading (k_s / n) with a producer-filed delta against the unsteered reading (k_a / n) of the same standup. The two rows
    mint DIFFERENT instruments even from one `_instrument` dict: the applied-positions mode is instrument identity (`none` for an unsteered run,
    `all` for a lens-served LoRA), so the own unsteered run is never in the steered row's exact-key pool -- the own pairing stands through tier 0
    whatever the instruments, and a re-pick needs an unsteered run under the STEERED row's instrument (`_at_key`), the strict identity
    that the end-to-end over the strength fixture exercises as a T2 refusal."""
    anchor = DC.blank_row(source="s", model="gemma4e4b", lane_kind="instruct-direct", view=view, dose=1.0, cell_id="none", arm="none", component="static",
                          eval_id="pasttense", metric="asr", threshold=0.5, value=k_a / n, k=k_a, n=n, status="graded", _instrument=dict(INST),
                          _producer_key=f"ladder/{view}/none/1.0", _key=ANCHOR_KEY, _judge=judge)
    steered = DC.blank_row(source="s", model="gemma4e4b", lane_kind="instruct-direct", view=view, dose=1.0, cell_id="constv2-mix-full__lora_r4_ce", arm="const_lora",
                           component="static", eval_id="pasttense", metric="asr", threshold=0.5, value=k_s / n, k=k_s, n=n, status="graded", _instrument=dict(INST),
                           _object_id=STEERED_SHA, _producer_key=f"ladder/{view}/const_lora/1.0", _key=STEERED_KEY, _judge=judge,
                           delta=delta, delta_lo=delta - 5, delta_hi=delta + 5, delta_p=0.4, effect_size=-0.2, effect_kind="cohens_h of defense success",
                           delta_method="producer: exact McNemar (synthetic)", _anchor_key=ANCHOR_KEY, _anchor_kind="same_standup_none", _n_pairs=n,
                           _contrast_extra={"producer_note": "synthetic"})
    return anchor, steered


def _emit(rows, ctx, **kw):
    batch = RecordBatch("t", "g")
    rep = DC.emit_rows(batch, ctx, [dict(r) for r in rows], adapter="t", campaign="c", reader="x", seed_provenance="none", **kw)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in batch.records) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    by = {}
    for r in batch.records:
        by.setdefault(r["record_type"], []).append(r["payload"])
    return by, rep


def _primed(rows, **kw) -> AdapterContext:
    return AF.primed_ctx_for_rows([dict(r) for r in rows], _ctx(), adapter="t", campaign="c", reader="x", seed_provenance="none", **kw)


def _frame_of(ctx: AdapterContext) -> Frame:
    fr = Frame.__new__(Frame)
    fr.con = ctx.conn
    return fr


def _steered_metric(by):
    return next(m for m in by["run_metric"] if m["k"] == 2 or m["run_id"] == next(r["run_id"] for r in by["run"] if r["object_id"] != "none"))


def assert_selection_shape(sel, *, state=None):
    """The `extra.anchor_selection` block as records.md declares it: the closed state and own-reason vocabularies, the key axes, the reading, no timestamp."""
    assert set(sel) == {"state", "own_reason", "own_anchor_run_id", "own_anchor_why", "why", "n_candidates", "candidates", "rejected", "key", "reading", "order", "rule"}, sorted(sel)
    assert sel["state"] in AC.STATES and sel["own_reason"] in AC.OWN_REASONS and sel["rule"] == AC.RULE and sel["order"] == AC.ORDER
    assert set(sel["key"]) == {"token_cap_tokens", "trial_index", "pass_label", "decode_policy_id"}
    assert sel["n_candidates"] == len(sel["candidates"]) and all(isinstance(r, list) and len(r) == 2 for r in sel["rejected"])
    if sel["state"] == "no_current_anchor":
        assert sel["reading"] is None
    else:
        assert set(sel["reading"]) == {"judgment_id", "computation_id", "computed_by", "source_uri"}
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", json.dumps(sel)), "no timestamp inside anchor_selection (the record id is a uuid5 over the payload)"
    if state is not None:
        assert sel["state"] == state


# ============================================================================================
# the hook: own current, parked, re-picked under each family rule
# ============================================================================================

def test_without_a_registry_connection_a_contrast_filing_read_refuses_by_name_and_a_contrast_less_one_does_not():
    anchor, steered = _rows()
    with pytest.raises(AC.AnchorCurrencyUnavailable):
        _emit([anchor, steered], _ctx())
    by, rep = _emit([anchor], _ctx())                      # an unsteered reading intends no contrast: no frame asked
    assert rep.runs == 1 and rep.contrasts == 0 and "run_contrast" not in by


def test_an_own_current_anchor_keeps_the_producers_record_and_gains_anchor_selection():
    anchor, steered = _rows()
    by, rep = _emit([anchor, steered], _primed([anchor, steered]))
    assert rep.contrasts == 1 and rep.contrasts_parked == 0 and rep.contrasts_reanchored == 0 and rep.refusals == []
    (c,) = by["run_contrast"]
    none_run = next(r for r in by["run"] if r["object_id"] == "none")
    assert c["anchor_run_id"] == none_run["run_id"] and c["anchor_kind"] == "same_standup_none" and c["computed_by"] == "producer"
    assert c["delta"] == -10.0 and c["n_pairs"] == 10 and c["effect_kind"] == "cohens_h" and c["delta_method"].startswith("producer:")
    assert c["anchor_judgment_id"] == next(j["judgment_id"] for j in by["judgment"] if j["run_id"] == none_run["run_id"])
    assert c["extra"]["producer_note"] == "synthetic"                       # the producer's contrast extra rides as before
    sel = c["extra"]["anchor_selection"]
    assert_selection_shape(sel, state="own_current")
    assert sel["own_reason"] == "current" and sel["own_anchor_run_id"] == none_run["run_id"]
    assert sel["candidates"] == [] and sel["rejected"] == []            # the own run is under its own instrument: the pool at the steered row's key is empty
    assert sel["reading"]["judgment_id"] == c["anchor_judgment_id"] and sel["key"] == {"token_cap_tokens": 4096, "trial_index": 0, "pass_label": None, "decode_policy_id": None}
    # the metric row carries no park
    assert _steered_metric(by)["extra"].get("delta_block") is None


def test_the_selection_block_is_byte_stable_across_two_reads_of_one_frame():
    anchor, steered = _rows()
    ctx = _primed([anchor, steered])
    a, _ = _emit([anchor, steered], ctx)
    b, _ = _emit([anchor, steered], ctx)
    assert I.canonical_json(a["run_contrast"][0]) == I.canonical_json(b["run_contrast"][0])


def test_an_own_anchor_the_registry_never_folded_parks_the_comparison_on_the_metric_row_and_names_it():
    """The first pass over a fresh registry (no provisional anchors): the own none is emitted by this very read, so the frame does
    not know it; the comparison parks (`own_reason not_in_frame`), its producer delta on the metric's `delta_block`, one row_refusals line."""
    anchor, steered = _rows()
    by, rep = _emit([anchor, steered], _ctx(Frame().con))
    assert rep.contrasts == 0 and rep.contrasts_parked == 1 and "run_contrast" not in by
    assert len(rep.refusals) == 1 and "contrast has no current anchor at key" in rep.refusals[0] and ": not_in_frame]" in rep.refusals[0]
    assert "not a run the registry has folded" in rep.refusals[0]
    blk = _steered_metric(by)["extra"]["delta_block"]
    assert blk["delta"] == -10.0 and blk["delta_method"].startswith("producer:") and blk["computed_by"] == "producer" and blk["n_pairs"] == 10
    assert blk["anchor"]["state"] == "no_current_anchor" and blk["anchor"]["own_reason"] == "not_in_frame" and blk["anchor"]["anchor_run_id"] is None
    assert blk["anchor"]["own_anchor_key"] == ANCHOR_KEY and blk["anchor"]["own_anchor_kind"] == "same_standup_none" and blk["anchor"]["own_anchor_run_id"]
    assert "not a run the registry has folded" in blk["why_not_a_contrast_row"] and blk["rule"] == AC.RULE
    assert blk["producer_contrast_extra"] == {"producer_note": "synthetic"}
    assert_selection_shape(blk["anchor_selection"], state="no_current_anchor")
    assert blk["anchor_selection"]["own_reason"] == "not_in_frame"


def test_a_retired_own_anchor_with_no_eligible_candidate_parks_with_the_fold_reason():
    anchor, steered = _rows()
    ctx = _primed([anchor, steered])
    none_id = next(r for r in _emit([anchor], ctx)[0]["run"])["run_id"]
    AF.hold_out(_frame_of(ctx), none_id)
    by, rep = _emit([anchor, steered], ctx)
    blk = _steered_metric(by)["extra"]["delta_block"]
    assert "run_contrast" not in by and rep.contrasts_parked == 1
    assert blk["anchor"]["own_reason"] == "held_out" and "held_out (ruling r_synth_hold_out)" in blk["why_not_a_contrast_row"]
    assert blk["anchor_selection"]["own_anchor_why"] == "held_out (ruling r_synth_hold_out)" and blk["anchor_selection"]["own_anchor_run_id"] == none_id
    assert "no current unsteered run of" in blk["why_not_a_contrast_row"]   # the nearest miss walks CURRENT runs only: the held-out own run is not one


def _at_key(ctx: AdapterContext, *, retire_own: str | None = None, instrument: str | None = None, k: int | None = 5, source: str = "s",
            view: str = "gemma4e4b_other") -> str:
    """A current unsteered run at the STEERED row's exact key -- its instrument (or `instrument`, for a T2 miss), cap, trial and pass -- as
    another standup (`view`) of `source`; `retire_own` retires the own unsteered run first."""
    fr = _frame_of(ctx)
    row = fr.con.execute("SELECT model_id, eval_id, instrument_id, token_cap_tokens, trial_index, pass_label FROM run WHERE object_id <> 'none'").fetchone()
    if retire_own is not None:
        AF.retire(fr, retire_own)
    rid = f"none:at_key:{view}:{instrument or 'steered-instrument'}"
    fr.run(rid, model=row[0], eval_id=row[1], inst=instrument or row[2], cap=row[3], trial=row[4], pass_label=row[5], source=source, standup=view,
           judge=JUDGE["judge_id"], k=k, n=10, value=(k / 10 if k is not None else 0.5), in_paper=0, roster=False, utc="2026-09-02T00:00:00Z")
    fr.con.commit()
    return rid


def test_the_default_family_rule_refuses_a_re_picked_anchor_by_name_and_parks_the_producers_delta():
    anchor, steered = _rows()
    ctx = _primed([anchor, steered])
    own = next(r for r in _emit([anchor], ctx)[0]["run"])["run_id"]
    sib = _at_key(ctx, retire_own=own)
    by, rep = _emit([anchor, steered], ctx)
    assert "run_contrast" not in by and rep.contrasts_parked == 1 and rep.contrasts_reanchored == 0
    assert "files no registry-derived statistic against the re-picked" in rep.refusals[0] and sib in rep.refusals[0]
    blk = _steered_metric(by)["extra"]["delta_block"]
    assert blk["anchor"]["state"] == "sibling_same_source" and blk["anchor"]["own_reason"] == "superseded" and blk["anchor"]["anchor_run_id"] == sib
    assert blk["anchor"]["anchor_kind"] == "sibling_none" and blk["anchor"]["registry_contrast_filed_against"] is None
    assert "files no registry-derived statistic" in blk["why_not_a_contrast_row"] and blk["delta"] == -10.0
    assert_selection_shape(blk["anchor_selection"], state="sibling_same_source")


def test_unpaired_rates_files_the_registrys_contrast_against_the_re_picked_anchor_and_parks_the_producers_delta():
    anchor, steered = _rows(k_s=2, n=10)
    ctx = _primed([anchor, steered])
    own = next(r for r in _emit([anchor], ctx)[0]["run"])["run_id"]
    sib = _at_key(ctx, retire_own=own, k=5)
    by, rep = _emit([anchor, steered], ctx, reanchor_statistic="unpaired_rates")
    assert rep.contrasts == 1 and rep.contrasts_reanchored == 1 and rep.contrasts_parked == 0 and rep.refusals == []
    (c,) = by["run_contrast"]
    assert c["anchor_run_id"] == sib and c["anchor_kind"] == "sibling_none" and c["computed_by"] == "registry" and c["n_pairs"] == 0
    assert c["delta_method"] == DC.UNPAIRED_RATE_METHOD and c["delta"] == pytest.approx(100 * (0.2 - 0.5)) and c["delta_lo"] < c["delta"] < c["delta_hi"]
    assert c["effect_kind"] == "cohens_h" and c["effect_size"] < 0 and 0 < c["delta_p"] < 1
    assert c["anchor_judgment_id"] == f"j:{sib}:{JUDGE['judge_id']}:1" and c["computed_utc"] == NOW
    ex = c["extra"]
    assert ex["cross_source"] is False and ex["contrast_caps"] == {"steered": 4096, "anchor": 4096} and ex["paired"] is False
    assert ex["anchor_standup"] == {"lane_source_legacy": "s", "view_legacy": "gemma4e4b_other", "cluster_job_id": None}
    assert (ex["n_steered"], ex["k_steered"], ex["n_anchor"], ex["k_anchor"]) == (10, 2, 10, 5) and ex["producer_delta_parked_on_metric"] is True
    assert ex["anchor_reading"]["judgment_id"] == c["anchor_judgment_id"]
    assert_selection_shape(ex["anchor_selection"], state="sibling_same_source")
    # the contrast's `computation_id` is the STEERED READING's (the identity the paper frame joins a contrast to its metric row on)
    # -- never the statistic's own row, which is filed once and named by `extra.statistic_computation_id`
    steered_metric = _steered_metric(by)
    assert c["computation_id"] == steered_metric["computation_id"] == next(p for p in by["computation"] if p["algorithm"].startswith("paperlib."))["computation_id"]
    assert ex["statistic_computation_id"] != c["computation_id"]
    comps = [p for p in by["computation"] if p["computation_id"] == ex["statistic_computation_id"]]
    assert len(comps) == 1 and comps[0]["algorithm"] == "registry.unpaired_rate_contrast" and comps[0]["version"] == AC.RULE and comps[0]["code_sha"] == "a" * 40
    # the producer's delta parks on the metric row, naming the contrast that stands
    blk = _steered_metric(by)["extra"]["delta_block"]
    assert blk["delta"] == -10.0 and blk["computed_by"] == "producer" and blk["anchor"]["registry_contrast_filed_against"] == sib
    assert "the registry-derived contrast against" in blk["why_not_a_contrast_row"] and "stands" in blk["why_not_a_contrast_row"]


def test_unpaired_rates_refuses_by_name_when_a_side_carries_no_count():
    anchor, steered = _rows()
    ctx = _primed([anchor, steered])
    own = next(r for r in _emit([anchor], ctx)[0]["run"])["run_id"]
    sib = _at_key(ctx, retire_own=own, k=None)            # a pooled reading: k NULL, a value alone
    by, rep = _emit([anchor, steered], ctx, reanchor_statistic="unpaired_rates")
    assert "run_contrast" not in by and rep.contrasts_parked == 1 and rep.contrasts_reanchored == 0
    assert "carries no count" in rep.refusals[0] and sib in rep.refusals[0]
    blk = _steered_metric(by)["extra"]["delta_block"]
    assert "carries no count" in blk["why_not_a_contrast_row"] and blk["anchor"]["anchor_run_id"] == sib


def test_a_current_unsteered_run_only_under_another_instrument_is_no_anchor_t2():
    anchor, steered = _rows()
    ctx = _primed([anchor, steered])
    own = next(r for r in _emit([anchor], ctx)[0]["run"])["run_id"]
    _at_key(ctx, retire_own=own, instrument="inst:other")
    by, rep = _emit([anchor, steered], ctx, reanchor_statistic="unpaired_rates")
    assert "run_contrast" not in by and rep.contrasts_parked == 1
    blk = _steered_metric(by)["extra"]["delta_block"]
    assert blk["anchor"]["state"] == "no_current_anchor" and "only under instrument inst:other" in blk["why_not_a_contrast_row"]


def test_a_steered_row_whose_anchor_key_names_no_run_of_the_read_still_searches_the_frame():
    """`none_declared`: the search runs -- here it finds the same-standup unsteered run the frame holds from an earlier read.
    The producer's delta was computed against SOME reading the emitter cannot name, so the family rule decides: the default parks it by name,
    `unpaired_rates` files the registry's statistic against the one same-standup current run (`same_standup_none`)."""
    anchor, steered = _rows()
    ctx = _primed([anchor, steered])
    _at_key(ctx, view="gemma4e4b_it", k=3)               # the one same-standup current unsteered run at the steered row's key (k 3 of 10, as the own one)
    by, rep = _emit([steered], ctx)                      # the anchor row is not in THIS read: run_by_key has no anchor -> none declared
    assert "run_contrast" not in by and rep.contrasts_parked == 1 and "none declared" in rep.refusals[0]
    blk = _steered_metric(by)["extra"]["delta_block"]
    assert blk["anchor"]["state"] == "sibling_same_source" and blk["anchor"]["own_reason"] == "none_declared" and blk["anchor"]["own_anchor_run_id"] is None
    assert blk["anchor"]["anchor_kind"] == "same_standup_none" and blk["anchor_selection"]["n_candidates"] == 1
    by2, rep2 = _emit([steered], ctx, reanchor_statistic="unpaired_rates")
    (c,) = by2["run_contrast"]
    assert c["anchor_kind"] == "same_standup_none" and c["computed_by"] == "registry" and c["extra"]["anchor_selection"]["own_reason"] == "none_declared"
    assert rep2.contrasts_reanchored == 1 and c["delta"] == pytest.approx(100 * (0.2 - 0.3))


def test_the_own_anchors_declared_judgment_is_honored_where_the_run_carries_two_complete_judgments_under_one_judge():
    """The registered plain none carries the store's own judgment AND ReLU's re-judge under the SAME judge id (told apart by session).
    The emitter pairs against the judgment it names (`judgment_by_run`); the helper reads the reading under it instead of refusing the run as
    ambiguous (e3 without a declaration)."""
    anchor, steered = _rows()
    ctx = _primed([anchor, steered])
    by0, _ = _emit([anchor], ctx)
    own = by0["run"][0]["run_id"]
    first_jid = by0["judgment"][0]["judgment_id"]
    fr = _frame_of(ctx)
    second_jid = fr.judgment(own, "j:second", judge=JUDGE["judge_id"])
    fr.metric(own, second_jid, metric="asr", thr=0.5, k=4, n=10, value=0.4)
    fr.con.commit()
    frame = fr.load()
    base = dict(model_id="gemma4e4b_it", eval_id="pasttense", instrument_id=fr.con.execute("SELECT instrument_id FROM run WHERE run_id = ?", (own,)).fetchone()[0],
                token_cap_tokens=4096, trial_index=0, pass_label=None, decode_policy_id=None, lane_source_legacy="s",
                standup={"view_legacy": "gemma4e4b_it", "lane_source_legacy": "s"}, own_anchor_run_id=own, own_anchor_kind="same_standup_none")
    undeclared = frame.choose(AC.AnchorKey(**base), "asr", 0.5, JUDGE["judge_id"])
    assert undeclared.state == "no_current_anchor" and undeclared.own_reason == "ambiguous" and "2 complete judgments" in undeclared.own_why
    declared = frame.choose(AC.AnchorKey(**base, own_anchor_judgment_id=second_jid), "asr", 0.5, JUDGE["judge_id"])
    assert declared.state == "own_current" and declared.anchor_judgment_id == second_jid and declared.measurement.k == 4
    wrong = frame.choose(AC.AnchorKey(**base, own_anchor_judgment_id="j:never"), "asr", 0.5, JUDGE["judge_id"])
    assert wrong.own_reason == "no_reading" and "not a complete judgment" in wrong.own_why
    # through the emitter: the anchor emitted with the first judgment pairs under the first (its `judgment_by_run`)
    by, rep = _emit([anchor, steered], ctx)
    (c,) = by["run_contrast"]
    assert c["anchor_judgment_id"] == first_jid and c["extra"]["anchor_selection"]["reading"]["judgment_id"] == first_jid and c["extra"]["anchor_selection"]["state"] == "own_current"
    # the declared judgment is part of the recorded comparison (the decision set re-decides it on another frame)
    rec = AC.comparison_record(AC.AnchorKey(**base, own_anchor_judgment_id=second_jid), "asr", 0.5, JUDGE["judge_id"])
    assert rec["key"]["own_anchor_judgment_id"] == second_jid and AC.key_of_comparison(rec).own_anchor_judgment_id == second_jid
    assert AC.comparison_record(AC.AnchorKey(**base), "asr", 0.5, JUDGE["judge_id"])["key"]["own_anchor_judgment_id"] is None
    assert AC.DECISION_RULE == "anchor_decision/7"      # 5: the currency-only comparison kind (r2_export's full_dose) joins the recorded set;
    # 6: an item-paired comparison may name the metric its item rows are filed at (`items_metric_id`, the store readers' label rows);
    # 7: it may also ask the per-item cap evidence (`needs_flags`: both sides' `item_flags.at_token_cap` digests, the graded mean's)


def test_the_decision_log_carries_every_comparison_the_emitter_made_parked_ones_included():
    anchor, steered = _rows()
    ctx = _ctx(Frame().con)
    ctx.begin_group("g")
    _emit([anchor, steered], ctx)
    assert len(ctx.decision_log) == 1 and ctx.decision_log[0]["decision"]["anchor_run_id"] == AC.PARKED
    cmp = ctx.decision_log[0]["comparison"]
    assert cmp["metric_id"] == "asr" and cmp["threshold"] == 0.5 and cmp["judge_id"] == JUDGE["judge_id"] and cmp["key"]["own_anchor_kind"] == "same_standup_none"
    assert cmp["key"]["own_anchor_judgment_id"] and cmp["computation_id"] and cmp["needs_items"] is False


# ============================================================================================
# the statistic helpers
# ============================================================================================

def test_unpaired_rate_contrast_delta_interval_test_and_effect():
    st = DC.unpaired_rate_contrast(20, 100, 10, 100)
    assert st["delta"] == pytest.approx(10.0) and st["delta_lo"] < 10.0 < st["delta_hi"] and st["n_pairs"] == 0
    assert st["delta_method"] == DC.UNPAIRED_RATE_METHOD and st["computed_by"] == "registry" and st["effect_kind"] == "cohens_h"
    assert st["effect_size"] == pytest.approx(DC.cohen_h(0.2, 0.1)) and st["effect_size"] > 0 and 0 < st["delta_p"] < 0.1
    assert st["extra"] == dict(st["extra"], n_steered=100, k_steered=20, n_anchor=100, k_anchor=10, paired=False, unit="pp")
    # the Newcombe interval is the hybrid of the two Wilson intervals
    lo, hi = DC.newcombe_interval(20, 100, 10, 100)
    assert (st["delta_lo"], st["delta_hi"]) == (pytest.approx(100 * lo), pytest.approx(100 * hi))
    p1, p0 = 0.2, 0.1
    l1, u1 = DC.wilson(20, 100)
    l0, u0 = DC.wilson(10, 100)
    assert lo == pytest.approx((p1 - p0) - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)) and hi == pytest.approx((p1 - p0) + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2))
    # the mirror image flips the sign of delta AND the effect
    back = DC.unpaired_rate_contrast(10, 100, 20, 100)
    assert back["delta"] == pytest.approx(-10.0) and back["effect_size"] == pytest.approx(-st["effect_size"])
    # equal rates: delta 0, p undefined when both are 0 / n (no variance) -> None, else 1
    assert DC.unpaired_rate_contrast(0, 10, 0, 10)["delta_p"] is None and DC.unpaired_rate_contrast(3, 10, 3, 10)["delta_p"] == pytest.approx(1.0)


@pytest.mark.parametrize("args,needle", [((None, 100, 10, 100), "steered reading (k)"), ((20, 100, None, 100), "anchor (k)"), ((20, None, 10, 100), "steered reading (n)"),
                                         ((20, 100, 10, 0), "no items"), ((120, 100, 10, 100), "exceeds its denominator"), ((float("nan"), 100, 10, 100), "steered reading (k)")])
def test_unpaired_rate_contrast_refuses_by_name(args, needle):
    with pytest.raises(DC.ContrastRefusal) as exc:
        DC.unpaired_rate_contrast(*args)
    assert needle in exc.value.why


def _items(n_per: dict, base: float = 2.0, bump: float = 0.0):
    """Synthetic per-item scores per stratum; `bump` shifts one side, with a per-item wobble so the paired differences are not all equal."""
    return {f"{b}/s{i}": base + bump * (1.0 + 0.1 * ((i % 3) - 1)) + i * 0.1 + (0.5 if b == "b2" else 0.0) for b, n in n_per.items() for i in range(n)}


def test_paired_items_contrast_preserves_the_stratum_weighted_estimand():
    s = _items({"b1": 10, "b2": 10}, bump=0.3)
    a = _items({"b1": 10, "b2": 10})
    st = DC.paired_items_contrast(s, a, seed=("run:s", "run:a"), n_boot=400)
    expected = sum(0.3 * (1.0 + 0.1 * ((i % 3) - 1)) for i in range(10)) / 10
    assert st["delta"] == pytest.approx(expected) and st["n_pairs"] == 20 and st["computed_by"] == "registry" and st["delta_method"] == DC.PAIRED_ITEMS_METHOD
    assert st["delta_lo"] <= st["delta"] <= st["delta_hi"] and st["effect_kind"] == "dz" and st["extra"]["paired"] is True
    # every item shared: each side's helper mean IS its filed mean_of_behavior_means
    def mobm(items):
        by = {}
        for k, v in items.items():
            by.setdefault(k.split("/")[0], []).append(v)
        return sum(sum(v) / len(v) for v in by.values()) / len(by)
    assert st["extra"]["means"]["steered"] == pytest.approx(mobm(s)) and st["extra"]["means"]["anchor"] == pytest.approx(mobm(a))
    assert set(st["extra"]["per_behavior"]) == {"b1", "b2"} and st["extra"]["per_behavior"]["b1"]["n_shared"] == 10
    assert st["extra"]["sign_test"] == {"positive": 20, "negative": 0, "ties": 0} and st["delta_p"] < 0.001
    # deterministic in the seed; another seed moves only the draws
    again = DC.paired_items_contrast(s, a, seed=("run:s", "run:a"), n_boot=400)
    assert (again["delta_lo"], again["delta_hi"]) == (st["delta_lo"], st["delta_hi"])
    other = DC.paired_items_contrast(s, a, seed=("run:s", "run:z"), n_boot=400)
    assert other["delta"] == st["delta"]


def test_paired_items_contrast_weights_strata_equally_where_the_item_weighted_mean_would_not():
    # b1: 10 shared items at +0.1; b2: 2 shared items at +1.0 -> stratum-weighted 0.55, item-weighted 0.25
    s = {**{f"b1/s{i}": 1.1 for i in range(10)}, **{f"b2/s{i}": 2.0 for i in range(2)}}
    a = {**{f"b1/s{i}": 1.0 for i in range(10)}, **{f"b2/s{i}": 1.0 for i in range(2)}}
    st = DC.paired_items_contrast(s, a, seed=("x", "y"), n_boot=200)
    assert st["delta"] == pytest.approx(0.55) and st["n_pairs"] == 12
    assert st["extra"]["per_behavior"]["b2"]["delta"] == pytest.approx(1.0) and st["extra"]["per_behavior"]["b1"]["delta"] == pytest.approx(0.1)
    # unshared items (judged on one side only, or no finite value) are not paired
    s2 = dict(s, **{"b1/s99": 3.0, "b2/s1": None})
    st2 = DC.paired_items_contrast(s2, a, seed=("x", "y"), n_boot=200)
    assert st2["n_pairs"] == 11 and st2["extra"]["per_behavior"]["b2"]["n_shared"] == 1


def test_paired_items_contrast_refuses_a_stratum_with_no_shared_item_by_name():
    s = {**{f"b1/s{i}": 1.0 for i in range(3)}, "b3/s0": 2.0}
    a = {**{f"b1/s{i}": 1.0 for i in range(3)}, "b2/s0": 1.0}
    with pytest.raises(DC.ContrastRefusal) as exc:
        DC.paired_items_contrast(s, a, seed=("x", "y"), n_boot=50)
    assert "'b2'" in exc.value.why and "'b3'" in exc.value.why and "no item judged on both sides" in exc.value.why
    with pytest.raises(DC.ContrastRefusal):
        DC.paired_items_contrast({}, {}, seed=("x", "y"))


def test_reanchor_computation_rows_are_stable_identities_with_a_sha_or_unverified_code_sha():
    c1 = DC.reanchor_computation("unpaired_rates", "a" * 40)
    c2 = DC.reanchor_computation("unpaired_rates", "unverified")
    assert c1["computation_id"] == c2["computation_id"] == I.computation_id(c1["algorithm"], AC.RULE, c1["seed_provenance"])
    assert c1["code_sha"] == "a" * 40 and c2["code_sha"] == "unverified" and DC.reanchor_computation("unpaired_rates", "not a sha")["code_sha"] == "unverified"
    assert DC.reanchor_computation("paired_items", "unverified")["algorithm"] == "registry.paired_items_contrast"
    with pytest.raises(ValueError):
        DC.reanchor_computation("refuse", "unverified")
    with pytest.raises(ValueError):
        DC.RowEmitter(RecordBatch("t", "g"), _ctx(), DC.load_seeds(), adapter="t", campaign="c", reader="x", seed_provenance="none", reanchor_statistic="paired_items")


# ============================================================================================
# the smoke, the versions, the spelling
# ============================================================================================

def test_the_smoke_names_the_unavailable_anchor_currency_per_group_instead_of_crashing(tmp_path):
    from sisteer.registry.adapters import strength_ladder as SL
    fx = F.build_strength(str(tmp_path / "lora"))
    decl = SL.StrengthRecordDecl(fx["path"], "strength_lane", ("const_lora", "diff_lora"), None, {"llama8b": "llama8b"})
    out = DC.smoke(SL.StrengthLadderAdapter(records=(decl,)))
    assert out["groups"] == 1 and out["anchor_unavailable_groups"] == 1
    assert len(out["refused_groups"]) == 1 and out["refused_groups"][0].startswith("strength_lane: anchor currency unavailable") and "no registry connection" in out["refused_groups"][0]


def test_every_emitter_adapter_declares_a_known_reanchor_statistic_and_the_versions_moved():
    import importlib
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import base as AB
    for mod_name in ADP.adapter_modules():
        mod = importlib.import_module(f"sisteer.registry.adapters.{mod_name}")
        for cls in ADP.adapter_classes(mod):
            stat = getattr(cls, "REANCHOR_STATISTIC", DC.REANCHOR_STATISTIC_DEFAULT)
            assert stat in DC.REANCHOR_STATISTICS, (cls.name, stat)
    from sisteer.registry.adapters import em_misalign as EM, relu_paired as RP
    assert EM.EmMisalignAdapter.REANCHOR_STATISTIC == "unpaired_rates" and getattr(RP.ReluPairedAdapter, "REANCHOR_STATISTIC", "refuse") == "refuse"
    assert DC.EMITTER_VERSION == 4 and AB.emission_version_of(RP.ReluPairedAdapter) == "16+emitter4" and AB.emission_version_of(EM.EmMisalignAdapter) == "12+emitter4"


def test_cross_source_is_the_one_spelling_under_src_and_docs():
    hits = []
    for base in ("src", "docs"):
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, base)):
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
            for fn in filenames:
                if fn.endswith((".py", ".md", ".sql", ".yaml", ".j2")):
                    with open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace") as fh:
                        if "cross_server" in fh.read():
                            hits.append(os.path.relpath(os.path.join(dirpath, fn), ROOT))
    assert hits == [], hits
    assert "cross_source" in open(os.path.join(ROOT, "docs", "registry", "records.md"), encoding="utf-8").read()
    # migration 0027 added ONE kind: `solo_stand_in`, the shared readers'
    # stand-in comparison, so a shared key's stand-in and its re-anchored steering contrast (`borrowed`) tell apart by anchor_kind
    assert set(V.ANCHOR_KINDS) == {"same_standup_none", "sibling_none", "borrowed", "full_dose", "solo_stand_in"}
