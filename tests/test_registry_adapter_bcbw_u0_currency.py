"""The anchor currency on main's unsteered ROW (the u0 row of the pipeline test's Qwen3.5-35B-A3B; `_bcbw_common.UNSTEERED_ROW_RULE` /
`WINDOW_ANCHOR_RULE`), the one piece of coverage commit 7's u0-standup tests carried alone, re-expressed on the landed row model: the
currency decides every contrast, but it never crosses serving windows. A u0 `none` that the frame has RETIRED leaves every row served
at its window PARKED BY NAME -- the three s1 LoRA rows and the u0 safety row at 8,192 -- even though the family's other `none`, the
14,336 re-serve, is current at the very same exact key (that key has no window term, so the currency offers it); the constitution row
at 14,336 keeps its own window-matched contrast. The pipeline's `-w<N>` contract: the baseline must not differ by serving window."""
from __future__ import annotations

import pytest

import _anchor_frame_fixtures as AF
import _bcbw_full_fixtures as Z
import _bcbw_seedrep_fixtures as X
import test_registry_adapter_bcbw_qwen35_35b_a3b as T35
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import lineage_common as LC

MK35, TAG35, TAG35_U0, NONE, LORA35 = T35.MK35, T35.TAG35, T35.TAG35_U0, T35.NONE, T35.LORA35


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


@pytest.mark.parametrize("eval_id", sorted(T35.MODULES))
def test_a_retired_window_matched_u0_none_parks_every_row_at_its_window_by_name_and_the_currency_never_re_anchors_across_windows(tmp_path, seeds, eval_id):
    mod, cls = T35.MODULES[eval_id]
    fx = Z.mirror(tmp_path, seeds, eval_id=eval_id, qwen35b_u0=True)
    ad = Z.adapter(cls, fx, seeds)
    frame = X.primed_frame(ad)                                     # the steady state after the fold: every run of the mirror current
    bt0, facts0, _ = Z.read_group(ad, TAG35, conn=frame.con)
    runs = {(r["view_legacy"], r["campaign_cell_id_legacy"], r["context_window_tokens"]): r for r in bt0["run"]}
    none_a, none_w = runs[(TAG35_U0, NONE, Z.WINDOW35)], runs[(TAG35_U0, NONE, Z.WINDOW35_CONST)]
    assert len(bt0["run_contrast"]) == 5 and facts0["contrasts_parked"] == 0 and facts0["contrasts_filed"] == 5, facts0["contrasts"]
    # the 8,192 baseline RETIRED in the frame (a later run at the same key superseded it); the 14,336 re-serve stays current at the same exact key
    AF.retire(frame, none_a["run_id"])
    bt, facts, _ = Z.read_group(ad, TAG35, conn=frame.con)
    at_8k = [(TAG35, Z.CTL35), (TAG35, LORA35), (TAG35, Z.DIFF35), (TAG35_U0, Z.PROMPTED)]
    parked_ids = set()
    for tag, okey in at_8k:
        r0 = runs[(tag, okey, Z.WINDOW35)]
        r = next(x for x in bt["run"] if x["run_id"] == r0["run_id"])             # the same stores: the same run ids
        parked_ids.add(r["run_id"])
        assert T35._contrast_of(bt, r["run_id"]) is None, (tag, okey)
        blk = r["extra"]["anchor"]
        assert blk["state"] == "no_current_anchor" and blk["anchor_run_id"] is None and blk["currency_own_reason"] == "superseded", blk
        assert "the window rule admits no baseline but the u0 `none` served at the row's window" in blk["reason"] and none_w["run_id"] in blk["reason"], blk["reason"]
        assert blk["window_matched"]["row_window"] == Z.WINDOW35 and blk["anchor_selection"]["state"] == "sibling_same_source"   # the currency's offer: the re-serve, a sibling at the key
        f = facts["contrasts"][f"{tag}|{okey}"]
        assert f["state"] == "no_current_anchor" and f["own_anchor_run_id"] == none_a["run_id"] and f["window_matched"]["row_window"] == Z.WINDOW35
        vr = next(m for m in bt["run_metric"] if m["run_id"] == r["run_id"] and m["computation_id"] == mod.recount_computation_id())
        assert vr["extra"]["delta_block"]["why_not_a_contrast_row"].startswith("the window rule admits no baseline") and vr["extra"]["delta_block"]["anchor"]["anchor_run_id"] == none_w["run_id"]
    # nothing filed onto the re-serve from the 8,192 rows: the currency never crosses serving windows
    assert not any(cn["run_id"] in parked_ids for cn in bt["run_contrast"])
    # the constitution row at 14,336 keeps its window-matched contrast against its own, current none
    r_c = next(x for x in bt["run"] if x["run_id"] == runs[(TAG35_U0, Z.CONST35, Z.WINDOW35_CONST)]["run_id"])
    cn = T35._contrast_of(bt, r_c["run_id"])
    assert cn is not None and cn["anchor_run_id"] == none_w["run_id"] and cn["extra"]["anchor_rule"] == B.WINDOW_ANCHOR_RULE and cn["extra"]["anchor_selection"]["state"] == "own_current"
    assert len(bt["run_contrast"]) == 1 and facts["contrasts_filed"] == 1 and facts["contrasts_parked"] == 4 and facts["contrasts_reanchored"] == 0
    assert sum(1 for x in facts["refusals"] if "window-matched anchor" in x) == 4, facts["refusals"]
    assert facts["anchor_states"] == {"own_current": 1, "sibling_same_source": 4}       # the currency offered the re-serve to the four 8,192 rows; the window rule declined it
