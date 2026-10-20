"""The IFEval canonical battery — ONE end-to-end DRAW-subset contract, through the REAL
writers and validators: ``rung_subset/2`` with sample 1 only for an item, both trials of one item, a mixed subset;
``validate_item_subset_v2`` refuses a draw / item inconsistency; ``item_subset_block_v2`` round-trips ``draws``;
``completion.finish_task`` writes DONE iff ``n_items == n_expected_full`` AND ``n_draws == n_expected_draws`` (both counts
on the marker), PARTIAL otherwise; ``verify_outcomes_against_summary`` checks the draw count; every existing caller
(``n_expected_draws=None``) writes a byte-identical marker; the outcome shape admits the IFEval items for the IFEval task
ONLY; ``rung_probe.draw_subset_task_reason`` admits a complete rung store (the runner's zero-request resume) and refuses
one missing a draw, one with an extra draw, a PARTIAL one, one at another cap, one whose manifest block is a v1 block;
the GSM8K v1 path is untouched (``ladder_task_partial_reason`` still PARTIAL-by-contract).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import ifeval_contracts as IC
from sisteer.capeval.tinycap import rung_probe as RP

import ifeval_synthetic_store as G

ROWS = G.matrix_rows()
NONE_MAIN = G.row_for(ROWS, "gemma4e4b", "main", "none")
QWEN_PSR = G.row_for(ROWS, "qwen35_4b", "main", "constv2-mix-full__lora_r4_ce")   # the LoRA object: PSR left the matrix; the name stays the fixture's


def _summary(n_items, n_draws, *, cap=2048, task=IC.IFEVAL_TASK):
    return {"task": task, C.METRIC_ACC: 0.5, C.METRIC_N: n_items, "n_requests": n_draws, "item_set": "ifeval541",
            "thinking": "off", "think_split_mode": C.THINK_SPLIT_RECORD_ONLY, "unterminated_policy": "score",
            "max_tokens": cap, "samples_per_item": 2, "n_scored": n_items, "n_unscored": 0, "n_scored_samples": n_draws,
            "n_unscored_samples": 0, "n_attempted_samples": n_draws, "n_stop_reason": n_draws, "n_at_cap": 0,
            "n_think_block_present": 0, "n_think_block_unterminated": 0, "n_unterminated_not_at_cap": 0, "n_completion_tokens": n_draws}


def _item(iid, idx, *, ifeval=True, **over):
    it = {"item_id": iid, "sample_idx": idx, "attempt": 1, "scored": True, "value": 1.0, "unparseable": False, "stop_reason": "stop",
          "at_cap": False, "think_block_present": False, "think_block_unterminated": False, "stray_open_tag": False,
          "content_empty_with_reasoning": False, "prompt_tokens": 80, "completion_tokens": 20, "seed": IC.ifeval_sample_seed(iid, idx),
          "sampling_ref": "abcdef012345"}
    if ifeval:
        it.update({"prompt_level_strict": True, "prompt_level_loose": True, "inst_level_strict": 1, "inst_level_loose": 1,
                   "num_instructions": 1, "instruction_id_list": ["punctuation:no_comma"]})
    it.update(over)
    return it


def _outcomes(draws, *, cap=2048, task=IC.IFEVAL_TASK, ifeval=True):
    items = [_item(i, s, ifeval=ifeval) for i, s in draws]
    summary = _summary(len({i for i, _ in draws}), len(draws), cap=cap, task=task)
    header = {**{f: summary[f] for f in C.TASK_OUTCOMES_SUMMARY_FIELDS}, "n_attempted_items": len({i for i, _ in draws}),
              "n_attempted_samples": len(draws), "producer_tree_sha": "f" * 40, "items": items}
    return summary, header


# --------------------------------------------------------------------------- the completion contract
def test_finish_task_writes_done_iff_both_item_and_draw_counts_match_and_the_marker_carries_both(tmp_path):
    draws = [("a", 0), ("a", 1), ("b", 1)]
    summary, header = _outcomes(draws)
    res = completion.finish_task(tmp_path / "d", task=IC.IFEVAL_TASK, n_items=2, n_expected_full=2, limits=None, n_rows=3, n_requests=3,
                                 n_rejected=0, wall_clock_s=1.0, summary=summary, outcomes=header, n_draws=3, n_expected_draws=3)
    assert res["complete"] is True
    marker = json.loads(Path(res["marker"]).read_text())
    assert set(marker) == set(C.DONE_MARKER_FIELDS) | set(IC.DONE_MARKER_DRAW_FIELDS)
    assert (marker["n_items"], marker["n_draws"], marker["n_expected_draws"]) == (2, 3, 3)
    # the same items, one draw short of the expected -> PARTIAL, both counts on the marker
    res2 = completion.finish_task(tmp_path / "p", task=IC.IFEVAL_TASK, n_items=2, n_expected_full=2, limits=None, n_rows=3, n_requests=3,
                                  n_rejected=0, wall_clock_s=1.0, summary=summary, outcomes=header, n_draws=3, n_expected_draws=4)
    assert res2["complete"] is False
    partial = json.loads(Path(res2["marker"]).read_text())
    assert set(partial) == set(C.PARTIAL_MARKER_FIELDS) | set(IC.DONE_MARKER_DRAW_FIELDS)
    assert (partial["n_draws"], partial["n_expected_draws"], partial["n_expected_full"]) == (3, 4, 2)
    # n_draws must agree with the outcome record
    with pytest.raises(completion.CompletionError, match="n_draws"):
        completion.finish_task(tmp_path / "x", task=IC.IFEVAL_TASK, n_items=2, n_expected_full=2, limits=None, n_rows=3, n_requests=3,
                               n_rejected=0, wall_clock_s=1.0, summary=summary, outcomes=header, n_draws=4, n_expected_draws=4)
    with pytest.raises(completion.CompletionError, match="without n_draws"):
        completion.finish_task(tmp_path / "y", task=IC.IFEVAL_TASK, n_items=2, n_expected_full=2, limits=None, n_rows=3, n_requests=3,
                               n_rejected=0, wall_clock_s=1.0, summary=summary, outcomes=header, n_expected_draws=3)


def test_a_no_switch_familys_ifeval_record_is_admitted_for_the_ifeval_task_only_and_the_switched_shape_still_is(tmp_path):
    """The llama probe (2026-09-13, the wave's second batch): a llama IFEval store's items carry the six row facts AND
    the `none` split's close-tag stray — the producer wrote that record and `_check_outcomes_shape` refused it, the
    contract's fifth shape missing. Admitted now for the IFEval task; a GSM8K record with it still refused; the
    switched families' IFEval shape (the 18 done stores, no stray) still admitted; a record MIXING the two refused."""
    draws = [("a", 0), ("a", 1), ("b", 0)]
    s_ns, h_ns = _outcomes(draws, ifeval=True)
    for it in h_ns["items"]:
        it["stray_close_tag"] = False
    assert all(set(it) == set(IC.IFEVAL_OUTCOME_ITEM_FIELDS_NO_SWITCH) for it in h_ns["items"])
    completion.write_task_outcomes(tmp_path / "ns", IC.IFEVAL_TASK, h_ns)
    completion.verify_outcomes_against_summary(h_ns, s_ns, n_draws=3)
    assert completion.read_task_outcomes(tmp_path / "ns", IC.IFEVAL_TASK)["items"][0]["stray_close_tag"] is False
    _, h_bad = _outcomes(draws, task=C.GSM8K_STANDARD_TASK, ifeval=True)          # a GSM8K record carrying the union: refused
    for it in h_bad["items"]:
        it["stray_close_tag"] = False
    with pytest.raises(completion.CompletionError, match="not the contract"):
        completion.write_task_outcomes(tmp_path / "bad", C.GSM8K_STANDARD_TASK, h_bad)
    s_sw, h_sw = _outcomes(draws, ifeval=True)                                     # the switched families' shape
    completion.write_task_outcomes(tmp_path / "sw", IC.IFEVAL_TASK, h_sw)
    completion.verify_outcomes_against_summary(h_sw, s_sw, n_draws=3)
    mixed = {**h_ns, "items": [h_ns["items"][0], h_sw["items"][1], h_ns["items"][2]]}
    with pytest.raises(completion.CompletionError, match="record 1 carries .* one shape"):
        completion.verify_outcomes_against_summary(mixed, s_ns, n_draws=3)


def test_every_existing_caller_writes_a_byte_identical_marker_and_the_gsm8k_shape_is_untouched(tmp_path, monkeypatch):
    task = C.GSM8K_STANDARD_TASK
    draws = [("g1", 0), ("g2", 0)]
    summary, header = _outcomes(draws, task=task, ifeval=False)
    summary.pop("n_attempted_samples")
    for it in header["items"]:
        it.pop("sampling_ref")
    monkeypatch.setattr("time.time", lambda: 1.0)
    res = completion.finish_task(tmp_path / "g", task=task, n_items=2, n_expected_full=2, limits=None, n_rows=2, n_requests=2,
                                 n_rejected=0, wall_clock_s=1.0, summary=summary, outcomes=header)
    marker = json.loads(Path(res["marker"]).read_text())
    assert set(marker) == set(C.DONE_MARKER_FIELDS)
    assert marker == {"task": task, "n_items": 2, "n_rows": 2, "n_requests": 2, "n_rejected": 0, "wall_clock_s": 1.0, "finished_at": 1.0}
    # a GSM8K record with IFEval item facts is REFUSED: the extra shape is admitted for the IFEval task only
    _, bad = _outcomes(draws, task=task, ifeval=True)
    with pytest.raises(completion.CompletionError, match="not the contract"):
        completion.write_task_outcomes(tmp_path / "bad", task, bad)
    # and an IFEval record with the plain v2 items is admitted too (the pre-facts shape)
    s2, h2 = _outcomes([("a", 0)], ifeval=False)
    completion.write_task_outcomes(tmp_path / "ok", IC.IFEVAL_TASK, h2)
    completion.verify_outcomes_against_summary(h2, s2)
    with pytest.raises(completion.CompletionError, match="n_draws"):
        completion.verify_outcomes_against_summary(h2, s2, n_draws=2)
    with pytest.raises(completion.CompletionError, match="n_attempted_samples"):
        completion.verify_outcomes_against_summary(h2, {**s2, "n_attempted_samples": 5})


# --------------------------------------------------------------------------- the rung store through the real producers
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("ife_draw_subset")
    at = {(i, 1) for i in range(500, 520)} | {(i, 0) for i in range(510, 530)}       # 20 sample-1-only, 10 both, 10 sample-0-only
    esc = G.build_escalated_cell(root, QWEN_PSR, at_cap=at, loops={(500, 1), (501, 1)}, rung_still_at_cap={(515, 0)},
                                 rung_boundary={(512, 1)})
    assert esc["rung_cell_dir"] is not None
    return root, esc


def test_the_adapters_v2_subset_names_sample_1_only_both_trials_and_mixed_items_and_round_trips(world):
    root, esc = world
    subset = esc["subset"]
    assert subset["schema"] == IC.ITEM_SUBSET_SCHEMA_V2 and subset["task"] == IC.IFEVAL_TASK
    draws = IC.draws_of_subset(subset)
    by_item = {}
    for i, s in draws:
        by_item.setdefault(i, set()).add(s)
    assert {1} in by_item.values() and {0, 1} in by_item.values() and {0} in by_item.values()   # sample-1-only, both, sample-0-only
    assert subset["n_at_cap"] == 40 and subset["n_loop"] == 2 and subset["n_truncated"] == 38 == subset["n_draws"] == len(draws)
    # 30 items were at cap; the two loop draws were items 500 / 501's ONLY at-cap draws, so 28 items remain in the subset
    assert subset["n_items"] == len(subset["ids"]) == 28 and subset["source_cap"] == 2048 and subset["cap"] == 4096
    assert (G.IDS[500], 1) not in draws and (G.IDS[501], 1) not in draws        # the loops are excluded
    # boundary (P11): the 8 sample-1-only items (502..509) + the 10 sample-0-only items (520..529) have a sibling that
    # terminated below the cap -> 18 boundary draws, IN the redraw set (a sub-class of truncated)
    assert subset["n_boundary"] == 18 and subset["n_truncated"] == 38
    on_disk = json.loads(Path(esc["subset_path"]).read_text())
    assert on_disk == IC.validate_item_subset_v2(on_disk) and on_disk == subset
    block = IC.item_subset_block_v2(on_disk, path=str(esc["subset_path"]))
    assert block["draws"] == [[i, s] for i, s in draws] and block["n_draws"] == 38 and block["n_items"] == 28
    rung_manifest = completion.read_store_manifest(esc["rung_cell_dir"])
    assert rung_manifest[C.MANIFEST_KEY_ITEM_SUBSET] == block
    assert rung_manifest["resolved_config"]["tinycap"][C.IDENTITY_KEY_CAP_RUNG] == IC.cap_rung_identity_v2(on_disk)
    assert rung_manifest["resolved_config"]["tinycap"][C.IDENTITY_KEY_CAP_RUNG]["subset_sha16"] == IC.item_subset_sha16_v2(on_disk["draws"])


def test_the_rung_store_files_per_sample_rows_and_outcomes_no_mean_rows_and_a_done_marker_with_both_counts(world):
    root, esc = world
    cd = esc["rung_cell_dir"]
    marker = json.loads((cd / C.DONE_DIR / completion.marker_filename(IC.IFEVAL_TASK)).read_text())
    assert (marker["n_items"], marker["n_draws"], marker["n_expected_draws"]) == (28, 38, 38)
    outcomes = completion.read_task_outcomes(cd, IC.IFEVAL_TASK)
    assert outcomes["samples_per_item"] == 2 and outcomes["n_attempted_items"] == 28 and outcomes["n_attempted_samples"] == 38
    assert {(it["item_id"], it["sample_idx"]) for it in outcomes["items"]} == set(IC.draws_of_subset(esc["subset"]))
    assert all(set(it) == set(IC.IFEVAL_OUTCOME_ITEM_FIELDS) for it in outcomes["items"])
    rows = [json.loads(l) for l in (cd / "rows.jsonl").read_text().splitlines()]
    metrics = {r["metric"] for r in rows if r["item_id"] != C.AGG_ITEM_ID}
    assert metrics <= {"acc@0", "acc@1"} and C.METRIC_ACC not in metrics                  # per-sample rows only, no mean rows
    assert sum(1 for r in rows if r["item_id"] != C.AGG_ITEM_ID) == 38
    for r in rows:
        if r["item_id"] != C.AGG_ITEM_ID:
            extra = json.loads(r["extra_json"])
            assert all(k in extra for k in IC.ROW_GUARD_FIELDS) and extra["view"] == "main" and extra["route_kind"] == "lens"
            assert extra["object_artifact_sha256"] == QWEN_PSR["object_sha256"] and extra["sampling_mode"] == "vendor_preset"
    summary = completion.read_task_summary(cd, IC.IFEVAL_TASK)
    assert summary["max_tokens"] == 4096 and summary[C.METRIC_N] == 28 and summary["n_attempted_samples"] == 38


def test_draw_subset_task_reason_admits_the_complete_rung_store_and_refuses_every_deviation(world, tmp_path):
    root, esc = world
    cd = esc["rung_cell_dir"]
    draws = esc["subset"]["draws"]
    src_hash = esc["subset"]["source_identity_hash"]
    assert RP.draw_subset_task_reason(cd, IC.IFEVAL_TASK, draws=draws, cap=4096, source_identity_hash=src_hash) is None
    assert "cap" in RP.draw_subset_task_reason(cd, IC.IFEVAL_TASK, draws=draws, cap=8192)
    more = RP.draw_subset_task_reason(cd, IC.IFEVAL_TASK, draws=draws + [[G.IDS[0], 0]], cap=4096)
    assert more and "n_items" in more and "29" in more                           # the marker's counts refuse first, by name
    fewer = RP.draw_subset_task_reason(cd, IC.IFEVAL_TASK, draws=draws[:-1], cap=4096)
    assert fewer and "!=" in fewer
    # a TORN store: the marker's counts agree but the outcomes name another draw -> refused on the draw set, by name
    outcomes = completion.read_task_outcomes(cd, IC.IFEVAL_TASK)
    torn = dict(outcomes)
    torn["items"] = [({**it, "item_id": G.IDS[0]} if (it["item_id"], it["sample_idx"]) == tuple(draws[0]) else it) for it in outcomes["items"]]
    completion.write_task_outcomes(cd, IC.IFEVAL_TASK, torn)
    why = RP.draw_subset_task_reason(cd, IC.IFEVAL_TASK, draws=draws, cap=4096)
    assert why and "missing" in why and "extra" in why and "never widen" in why
    completion.write_task_outcomes(cd, IC.IFEVAL_TASK, outcomes)                  # restore the fixture
    assert "source_identity_hash" in RP.draw_subset_task_reason(cd, IC.IFEVAL_TASK, draws=draws, cap=4096, source_identity_hash="0" * 16)
    assert RP.draw_subset_task_reason(tmp_path, IC.IFEVAL_TASK, draws=draws, cap=4096) == "no done marker"
    assert "empty" in RP.draw_subset_task_reason(cd, IC.IFEVAL_TASK, draws=[], cap=4096)
    assert "repeats" in RP.draw_subset_task_reason(cd, IC.IFEVAL_TASK, draws=draws + [draws[0]], cap=4096)
    # a pass-1 store is NOT a rung store: its marker carries the draw counts but the manifest has no subset block
    src = esc["source_cell_dir"]
    src_draws = [[i, t] for i in G.IDS for t in range(2)]
    why = RP.draw_subset_task_reason(src, IC.IFEVAL_TASK, draws=src_draws, cap=2048)
    assert why and C.MANIFEST_KEY_ITEM_SUBSET in why
    # a PARTIAL store (one draw short) refuses by name — repair = delete and re-run, never widen
    partial_row = G.row_for(ROWS, "gemma4e4b", "main", "none")
    short = list(IC.draws_of_subset(esc["subset"]))[:5]
    cd2 = G.build_store(tmp_path / "short", partial_row, at_cap=(), cap=4096, draws=short,
                        rung_source=completion.read_store_manifest(src), subset_doc=esc["subset"], subset_path="x", n_expected_draws=6, ladder=False)
    why = RP.draw_subset_task_reason(cd2, IC.IFEVAL_TASK, draws=[[i, s] for i, s in short], cap=4096)
    assert why and "partial marker" in why


def test_the_gsm8k_ladder_validator_is_untouched_and_a_v1_block_is_refused_by_the_draw_validator(tmp_path):
    src = (Path(__file__).resolve().parent.parent / "src/sisteer/capeval/tinycap/rung_probe.py").read_text()
    assert 'return "a done marker (a rung-2 selection run ends PARTIAL by contract)"' in src
    # the runner's dispatch still names the k = 1 validator, and the new one is exported beside it
    assert "draw_subset_task_reason" in RP.__all__ and "ladder_task_partial_reason" in RP.__all__
