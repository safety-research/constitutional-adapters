"""qwen-seedrep-native-battery fg1 / F-AU: the gsm8k store's completeness accounting.

THE FINDING, re-validated as the first assertions: the gsm8k `S3` smoke (5 cells, thinking ON, cap 8192) filed 196 `acc` rows for 200
attempted items -- the scorer files NO acc row for a draw it returned no verdict for (4 draws stopped at the token cap) -- while
its `outcomes/<task>.json` declared all 200 (each of the 4 typed `scored: false`, `at_cap: true`, bound to its request) and its done
marker said `n_items: 200` (the ATTEMPTED count, `finish_task(n_items=n)`), `n_rows: 196`. The reader (`audit_common.read_tinycap_store`)
took the acc rows as the writer's declaration and refused every such store: "4 stored (item_id, sample_idx) outcome record(s) the writer
declares nowhere | done marker n_items=200 != 196 filed item(s)" -- an AUDIT MISCOUNT, not a hole in the store.

THE FIX, both sides: the WRITER files every attempted (item, sample) exactly once -- an unscored draw gets a TYPED OUTCOME row
(`contracts.METRIC_UNSCORED`, value 1.0 = an indicator, `extra.outcome` naming why: `unscored_at_cap` / `unscored_unparseable` /
`unscored`) -- and the done marker records `n_planned` beside `n_items` (which now counts the filed == attempted items). The AUDIT holds
the rows to the outcomes record (the writer's ALL-ATTEMPTED declaration): every scored outcome has its acc row, every filed row its
outcome, the marker's `n_items` is the record's `n_attempted_items`, its `n_rows` the filed row keys, `n_planned` (when present) equals
`n_items`; an unscored outcome WITHOUT a row is the pre-F-AU writer's documented shape and is recorded, never refused. The S3-shaped store
therefore reads COMPLETE under the fixed audit, and every GENUINE gap still refuses by name.

Synthetic throughout: placeholder item ids, no recorded data."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
from sisteer.capeval.tiers import inspect_served as isv  # noqa: E402
from sisteer.capeval.tinycap import completion  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from test_inspect_served import VMODEL as HARNESS_VMODEL  # noqa: E402  (the fake sample records its usage under the harness route)
from test_inspect_served import _Emitter, _Log, _Sample, _Score, _Store, _fake_eval, _model_output, _plan  # noqa: E402

GSM8K_TASK = C.GSM8K_STANDARD_TASK
VMODEL = "petri__base"
S = 2
CAP = 8192


# --------------------------------------------------------------------------------------------- the contract: one typed-outcome rule
def test_outcome_type_and_the_sample_metric_parser_are_one_rule_the_writer_and_the_audit_share():
    assert C.METRIC_UNSCORED == "unscored" and C.OUTCOME_TYPE_FIELD == "outcome"
    assert C.OUTCOME_TYPES == ("scored", "unscored_at_cap", "unscored_unparseable", "unscored")
    assert C.outcome_type(scored=True, at_cap=True) == "scored"
    assert C.outcome_type(scored=False, at_cap=True, unparseable=True) == "unscored_at_cap"     # the cap wins: the draw never finished
    assert C.outcome_type(scored=False, at_cap=False, unparseable=True) == "unscored_unparseable"
    assert C.outcome_type(scored=False, at_cap=None, unparseable=None) == "unscored"
    assert C.split_sample_metric(C.sample_metric(C.METRIC_ACC, 3)) == ("acc", 3)
    assert C.split_sample_metric(C.sample_metric(C.METRIC_UNSCORED, 0)) == ("unscored", 0)
    assert C.split_sample_metric("acc") == ("acc", None) and C.split_sample_metric(C.METRIC_PASS1_BASE) == ("pass@1_base", None)
    assert C.DONE_MARKER_N_PLANNED_FIELD == "n_planned" and "n_planned" in C.DONE_MARKER_OPTIONAL_FIELDS
    assert "n_planned" not in C.DONE_MARKER_FIELDS, "additive: a marker from before F-AU stays a valid marker"
    for name in ("METRIC_UNSCORED", "outcome_type", "split_sample_metric", "OUTCOME_TYPES", "OUTCOME_TYPE_FIELD", "DONE_MARKER_N_PLANNED_FIELD"):
        assert name in C.__all__, name


def test_finish_task_writes_n_planned_on_the_done_marker_only_when_given(tmp_path):
    summary = {"n": 2, "n_scored": 2, "n_unscored": 0, "n_scored_samples": 2, "n_unscored_samples": 0, "acc": 1.0}
    fin = completion.finish_task(tmp_path / "a", task=GSM8K_TASK, n_items=2, n_expected_full=2, limits=None, n_rows=2, n_requests=2, n_rejected=0,
                                 wall_clock_s=1.0, summary=summary, n_planned=2)
    dm = json.loads(Path(fin["marker"]).read_text())
    assert fin["complete"] and dm["n_planned"] == 2 and dm["n_items"] == 2 and completion.is_done(tmp_path / "a", GSM8K_TASK)
    fin0 = completion.finish_task(tmp_path / "b", task=GSM8K_TASK, n_items=2, n_expected_full=2, limits=None, n_rows=2, n_requests=2, n_rejected=0,
                                  wall_clock_s=1.0, summary=summary)
    assert "n_planned" not in json.loads(Path(fin0["marker"]).read_text()), "a caller that passes None writes a byte-identical marker"
    # a LIMITED run keeps the partial marker byte-identical: its plan count is `n_expected_full`
    fin1 = completion.finish_task(tmp_path / "c", task=GSM8K_TASK, n_items=1, n_expected_full=2, limits={"limit": 1}, n_rows=1, n_requests=1,
                                  n_rejected=0, wall_clock_s=1.0, summary={**summary, "n": 1}, n_planned=2)
    pm = json.loads(Path(fin1["marker"]).read_text())
    assert not fin1["complete"] and "n_planned" not in pm and pm["n_expected_full"] == 2


# --------------------------------------------------------------------------------------------- the WRITER, end to end
def test_the_writer_files_a_typed_outcome_row_for_an_unscored_draw_and_the_marker_records_n_planned(tmp_path, monkeypatch):
    """Three items: `a` scored 1, `b` UNSCORED at the token cap (the smoke's shape), `c` scored 0. Previously `b` filed no row at all
    (n_rows 2 of n_items 3, no n_planned); now it files ONE typed `unscored` row and the marker says n_items 3 == n_rows 3 == n_planned 3."""
    log = _Log([_Sample("a", {"m": _Score("C", "1")}, output=_model_output("1")),
                _Sample("b", {"m": _Score(float("nan"), None)}, output=_model_output("<think>still reasoning", stop_reason=isv.STOP_REASON_AT_CAP)),
                _Sample("c", {"m": _Score("I", "3")}, output=_model_output("3"))])
    monkeypatch.setattr("inspect_ai.eval_async", _fake_eval({0: log}))
    em, store = _Emitter(), _Store(tmp_path / "c")
    summary = asyncio.run(isv.run_one_task(_plan(row_task=GSM8K_TASK, name="gsm8k_paired200", expected_n=3, max_tokens=512),
                                           model=None, store=store, template={}, concurrency=4, route_virtual_model=HARNESS_VMODEL, emitter=em))
    assert (summary["n"], summary["n_scored"], summary["n_unscored"]) == (3, 2, 1) and summary["acc"] == 0.5
    assert [r["item_id"] for r in em.by(C.METRIC_ACC)] == ["a", "c"], "§PW-nanfix-1 stands: no acc row for the unscored draw"
    (u,) = em.by(C.METRIC_UNSCORED)
    assert u["item_id"] == "b" and u["value"] == 1.0 and u["tier"] == C.ROW_TIER
    assert u["extra"][C.OUTCOME_TYPE_FIELD] == C.OUTCOME_TYPE_UNSCORED_AT_CAP and u["extra"]["unscored"] is True
    assert u["extra"]["stop_reason"] == isv.STOP_REASON_AT_CAP and u["extra"]["scores"] == {"m": None}
    # the item is FILED (the producer's own definition: any metric counts) -- the audit's inventory sees three items
    keys = {(r["task"], r["item_id"], r["metric"]) for r in em.rows}
    assert C.filed_item_pairs(keys) == {(GSM8K_TASK, "a"), (GSM8K_TASK, "b"), (GSM8K_TASK, "c")}
    dm = json.loads(completion.done_marker_path(store.cell_dir, GSM8K_TASK).read_text())
    assert (dm["n_items"], dm["n_rows"], dm["n_planned"]) == (3, 3, 3), dm
    # the outcomes record (unchanged shape) types the same draw the same way the row does
    out = completion.read_task_outcomes(store.cell_dir, GSM8K_TASK)
    b = [it for it in out["items"] if it["item_id"] == "b"][0]
    assert b["scored"] is False and b["at_cap"] is True and C.outcome_type(scored=b["scored"], at_cap=b["at_cap"], unparseable=b["unparseable"]) == u["extra"]["outcome"]
    # ... and the reader, over the rows this run emitted, reads the store COMPLETE
    (store.cell_dir / "rows.jsonl").write_text("".join(json.dumps({"task": r["task"], "item_id": r["item_id"], "metric": r["metric"], "value": r["value"]}) + "\n" for r in em.rows))
    (store.cell_dir / "manifest.json").write_text(json.dumps({"cell_id": "none", "seeds": {"inspect": {"replicate": {"policy": C.INSPECT_SEED_POLICY_REPLICATE, "seed_base": S}}}}))
    sr = ac.read_tinycap_store(store.cell_dir, task=GSM8K_TASK, leg="gsm8k", arm="none")
    assert sr.complete, sr.completeness_note
    assert sr.filed == {(GSM8K_TASK, "a"), (GSM8K_TASK, "b"), (GSM8K_TASK, "c")} and len(sr.draws) == 3


# --------------------------------------------------------------------------------------------- the AUDIT, over the S3-shaped store
def _result(item: str, idx: int = 0, *, scored: bool = True, at_cap: bool = False, value: float = 1.0) -> isv.ItemResult:
    return isv.ItemResult(item_id=item, value=(float("nan") if not scored else value), sample_idx=idx, seed=int(C.inspect_replicate_seed(item, idx, S)),
                          answer=None if not scored else "42", prompt_tokens=10, completion_tokens=5, paired_id=None, scores={"acc": (float("nan") if not scored else value)},
                          content="42", reasoning=None, stop_reason=(isv.STOP_REASON_AT_CAP if at_cap else "stop"), unparseable=False,
                          request_id=f"rid-{item}-{idx}", completion_id=f"cmpl-{item}-{idx}")


def _cell(tmp_path: Path, *, n: int = 200, unscored_at_cap: int = 4, k: int = 1, file_unscored_rows: bool = False, marker: dict | None = None,
          rows_override=None) -> Path:
    """A gsm8k cell dir in the writer's real shape: `n` items, the last `unscored_at_cap` of them UNSCORED at the cap; the pre-F-AU writer files
    NO row for those (`file_unscored_rows=False`), the F-AU writer their typed `unscored` rows. `marker` overrides the done marker's counts."""
    d = tmp_path / "cells" / "none"
    d.mkdir(parents=True, exist_ok=True)
    items = [f"gsm8k_{i:04d}" for i in range(n)]
    results = [_result(it, idx, scored=(i < n - unscored_at_cap), at_cap=(i >= n - unscored_at_cap), value=float(i % 2)) for i, it in enumerate(items) for idx in range(k)]
    plan = isv.TaskPlan(row_task=GSM8K_TASK, name=C.GSM8K_PAIRED_TASK_NAME, task=object(), expected_n=n, samples_per_item=k, max_tokens=CAP,
                        item_set=C.GSM8K_ITEM_SET_PAIRED, think_split_mode=C.THINK_SPLIT_IMPLICIT_OPEN)
    n_s, n_u = (n - unscored_at_cap), unscored_at_cap
    summary = {"n_scored": n_s, "n_unscored": n_u, "n_scored_samples": n_s * k, "n_unscored_samples": n_u * k, "n_stop_reason": n * k, "n_at_cap": n_u * k}
    rec = isv.outcome_record(results, plan=plan, task=GSM8K_TASK, thinking=C.THINKING_ON, summary=summary, split_counts=isv.think_split_counts(results, plan=plan),
                             producer_tree_sha="0" * 40)
    completion.write_task_outcomes(d, GSM8K_TASK, rec)
    rows = []
    for r in results:
        if not r.unscored:
            rows.append({"task": GSM8K_TASK, "item_id": r.item_id, "metric": (C.METRIC_ACC if k == 1 else C.sample_metric(C.METRIC_ACC, r.sample_idx)), "value": r.value})
        elif file_unscored_rows:
            rows.append({"task": GSM8K_TASK, "item_id": r.item_id, "metric": (C.METRIC_UNSCORED if k == 1 else C.sample_metric(C.METRIC_UNSCORED, r.sample_idx)), "value": 1.0,
                         "extra_json": {C.OUTCOME_TYPE_FIELD: C.OUTCOME_TYPE_UNSCORED_AT_CAP}})
    if k > 1:
        rows += [{"task": GSM8K_TASK, "item_id": it, "metric": C.METRIC_ACC, "value": 0.5} for it in items[:n_s]]
        if file_unscored_rows:
            rows += [{"task": GSM8K_TASK, "item_id": it, "metric": C.METRIC_UNSCORED, "value": 1.0} for it in items[n_s:]]
    rows += [{"task": GSM8K_TASK, "item_id": C.AGG_ITEM_ID, "metric": C.METRIC_ACC_AGGREGATE, "value": 0.5}, {"task": GSM8K_TASK, "item_id": C.AGG_ITEM_ID, "metric": C.METRIC_N, "value": float(n)}]
    if rows_override is not None:
        rows = rows_override(rows)
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "manifest.json").write_text(json.dumps({"cell_id": "none", "seeds": {"inspect": {"replicate": {"policy": C.INSPECT_SEED_POLICY_REPLICATE, "seed_base": S}}}}))
    n_rows = sum(1 for r in rows if not C.is_aggregate_item(r["item_id"]))
    dm = {"task": GSM8K_TASK, "n_items": n, "n_rows": n_rows, "n_requests": n * k, "n_rejected": 0, "wall_clock_s": 1.0, "finished_at": 1.0, **(marker or {})}
    completion.write_done_marker(d, GSM8K_TASK, dm)
    return d


def _read(d: Path) -> ac.StoreRead:
    return ac.read_tinycap_store(d, task=GSM8K_TASK, leg="gsm8k", arm="none")


def test_the_s3_shaped_pre_fix_store_reads_complete_under_the_fixed_audit(tmp_path):
    """The real `S3` smoke's numbers (200 attempted, 196 acc rows, 4 at-cap unscored draws in the outcomes record, marker n_items 200 / n_rows 196,
    no n_planned): the refusal was the audit's miscount; the fixed reader reads the store COMPLETE and names the four unfiled unscored draws."""
    sr = _read(_cell(tmp_path))
    assert sr.complete, sr.completeness_note
    assert "declares nowhere" not in sr.completeness_note and "!= 196" not in sr.completeness_note
    assert len(sr.filed) == 196 and len(sr.draws) == 200, "filed = the rows (the producer's definition); the draws = every attempted outcome, bound"
    assert all(d.request_id and d.completion_id for d in sr.draws)
    at_cap = [d for d in sr.draws if d.at_cap]
    assert len(at_cap) == 4 and all(d.finish_reason == isv.STOP_REASON_AT_CAP for d in at_cap)
    notes = " | ".join(sr.notes)
    assert "4 unscored outcome(s) without a row" in notes and "pre-F-AU" in notes and "'unscored_at_cap': 4" in notes and "'scored': 196" in notes
    assert "n_planned None (absent: a pre-F-AU marker)" in notes


def test_the_f_au_writers_store_reads_complete_with_its_typed_rows_and_n_planned(tmp_path):
    sr = _read(_cell(tmp_path, file_unscored_rows=True, marker={"n_planned": 200}))
    assert sr.complete, sr.completeness_note
    assert len(sr.filed) == 200 and len(sr.draws) == 200
    assert "200 attempted draw(s) over 200 item(s)" in " | ".join(sr.notes) and "4 typed unscored row(s) filed, 0 unscored outcome(s) without a row" in " | ".join(sr.notes)
    # avg@k: per-sample rows, the mean row per scored item, the typed row per unscored sample / all-unscored item
    sr2 = _read(_cell(tmp_path / "k2", n=6, unscored_at_cap=2, k=2, file_unscored_rows=True, marker={"n_planned": 6}))
    assert sr2.complete, sr2.completeness_note
    assert len(sr2.draws) == 12 and len(sr2.filed) == 6


@pytest.mark.parametrize("mutate, needle", [
    # a score the writer dropped: a scored outcome without its acc row
    (dict(rows_override=lambda rows: [r for r in rows if r["item_id"] != "gsm8k_0000"]), "1 scored outcome record(s) without a filed acc row -- a score the writer dropped"),
    # an acc row for a draw the record declares UNSCORED (the grader's silence written down as a number)
    (dict(rows_override=lambda rows: rows + [{"task": GSM8K_TASK, "item_id": "gsm8k_0199", "metric": "acc", "value": 0.0}]), "1 filed acc row(s) the outcomes record does not declare scored"),
    # a typed unscored row for a draw the record declares SCORED
    (dict(rows_override=lambda rows: rows + [{"task": GSM8K_TASK, "item_id": "gsm8k_0001", "metric": "unscored", "value": 1.0}]), "1 filed unscored row(s) the outcomes record does not declare unscored"),
    # the marker's counts are not the record's / the rows'
    (dict(marker={"n_items": 199}), "done marker n_items=199 != the outcomes record's n_attempted_items=200"),
    (dict(marker={"n_rows": 195}), "done marker n_rows=195 != 196 filed row key(s)"),
    (dict(marker={"n_planned": 201}), "done marker n_planned=201 != n_items=200"),
    # a row under a metric this store's writer never files
    (dict(rows_override=lambda rows: rows + [{"task": GSM8K_TASK, "item_id": "gsm8k_0001", "metric": "acc_norm", "value": 1.0}]), "1 filed row(s) under metric(s) this reader does not know"),
])
def test_every_genuine_gap_still_refuses_by_name(tmp_path, mutate, needle):
    kw = dict(mutate)
    rows_override = kw.pop("rows_override", None)
    marker = kw.pop("marker", None)
    if rows_override is not None:
        # the marker's n_rows is the WRITER's count; a row we add or drop by hand must not be silently re-counted into it
        base_rows = {"n_rows": 196}
        marker = {**base_rows, **(marker or {})}
    sr = _read(_cell(tmp_path, file_unscored_rows=False, marker=marker, rows_override=rows_override))
    assert not sr.complete and needle in sr.completeness_note, sr.completeness_note


def test_a_duplicated_outcome_record_and_a_duplicated_row_are_genuine_gaps(tmp_path):
    d = _cell(tmp_path)
    out = json.loads(completion.task_outcomes_path(d, GSM8K_TASK).read_text())
    out["items"].append(dict(out["items"][0]))
    out["n_attempted_samples"] += 1
    completion.task_outcomes_path(d, GSM8K_TASK).write_text(json.dumps(out))
    sr = _read(d)
    assert not sr.complete and "1 (item_id, sample_idx) outcome record(s) stored twice" in sr.completeness_note, sr.completeness_note
    d2 = _cell(tmp_path / "dup_row", rows_override=lambda rows: rows + [dict(rows[0])])
    sr2 = _read(d2)
    assert not sr2.complete and "1 filed acc row key(s) stored twice" in sr2.completeness_note, sr2.completeness_note
