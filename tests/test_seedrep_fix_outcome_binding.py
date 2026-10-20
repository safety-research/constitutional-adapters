"""The GSM8K outcome record carries each draw's attempt binding.

The finding, RE-VALIDATED as the first assertions: `ItemResult.extra()` / `sample_provenance()` carried the pair the
provider's response hook recorded, but `outcome_record()` built its item dicts by hand and dropped both — and that
`outcomes/<task>.json` is what the GSM8K reader (`read_tinycap_store`) binds draws off. Its fallback recovers a
binding from `response_evidence.jsonl` keyed on `(task, sample)` alone; on the SHARED engine every arm (and Arditi's
three seed bases) draws the same (item, sample), so the lookup finds several completed attempts and refuses the store as
ambiguous — although the correct binding existed on the `ItemResult`.

The fix: `outcome_record` stamps `request_id` / `completion_id` on every item ONLY when recorded (a default-off run's
record is byte-identical); the completion validator (writer AND reader) admits every shape with or without the pair,
exact either way (`contracts.TASK_OUTCOME_ITEM_BINDING_FIELDS`); the rung merge writers copy items whole. Proven end to
end against the REAL reader and `bind_rows` (loaded by file) with two arms sharing an (item, sample)."""
from __future__ import annotations

import json
import shutil

import pytest

from seedrep_fixes_common import (ATTEMPT, GSM8K_TASK, S, VMODEL, body, evidence, hook_off, ledger_row,  # noqa: F401
                                  store_read, wire, ws3g_modules)
from sisteer.capeval.tiers import inspect_rung
from sisteer.capeval.tiers import inspect_served as isv
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.eval import request_evidence_client as rev

CAP = 8192
ARDITI_VMODEL = "petri__arditi_allpos"


def _plan(**kw):
    base = dict(row_task=GSM8K_TASK, name=C.GSM8K_PAIRED_TASK_NAME, task=object(), expected_n=1, samples_per_item=1,
                max_tokens=CAP, item_set=C.GSM8K_ITEM_SET_PAIRED, think_split_mode=C.THINK_SPLIT_IMPLICIT_OPEN)
    base.update(kw)
    return isv.TaskPlan(**base)


def _result(item: str, idx: int, *, rid=None, cid=None, value: float = 1.0) -> isv.ItemResult:
    return isv.ItemResult(item_id=item, value=value, sample_idx=idx, seed=int(C.inspect_replicate_seed(item, idx, S)),
                          answer="42", prompt_tokens=10, completion_tokens=5, paired_id=None, scores={"acc": value},
                          content="42", reasoning=None, stop_reason="stop", unparseable=False, request_id=rid, completion_id=cid)


def _record(results, *, sampling_ref=None):
    plan = _plan(expected_n=len({r.item_id for r in results}), samples_per_item=max(r.sample_idx for r in results) + 1)
    n = len(results)
    summary = {"n_scored": n, "n_unscored": 0, "n_scored_samples": n, "n_unscored_samples": 0, "n_stop_reason": n, "n_at_cap": 0}
    counts = isv.think_split_counts(results, plan=plan)
    return isv.outcome_record(results, plan=plan, task=GSM8K_TASK, thinking=C.THINKING_OFF, summary=summary,
                              split_counts=counts, producer_tree_sha="0" * 40, sampling_ref=sampling_ref)


def test_outcome_items_carry_the_binding_only_when_recorded_and_the_validator_admits_each_shape_bound_or_not(tmp_path):
    # RE-VALIDATION: the ItemResult carries the pair on the row and the per_sample entry ...
    r = _result("q1", 0, rid="rid-A", cid="cmpl-A")
    assert r.extra()["request_id"] == "rid-A" and r.sample_provenance()["completion_id"] == "cmpl-A"
    # ... and the outcome item does too now (pre-fix: the hand-built dict dropped both)
    bound = _record([r])
    item = bound["items"][0]
    assert item["request_id"] == "rid-A" and item["completion_id"] == "cmpl-A"
    assert set(item) == set(C.outcome_item_shape_with_binding(C.TASK_OUTCOME_ITEM_FIELDS))
    assert set(_record([r], sampling_ref="0" * 12)["items"][0]) == set(C.outcome_item_shape_with_binding(C.TASK_OUTCOME_ITEM_FIELDS_V2))
    # a default-off draw: the item is exactly the pre-fix shape (v1; v2 under a sampling_ref) — byte-identical records
    plain = _record([_result("q1", 0)])
    assert set(plain["items"][0]) == set(C.TASK_OUTCOME_ITEM_FIELDS)
    assert set(_record([_result("q1", 0)], sampling_ref="0" * 12)["items"][0]) == set(C.TASK_OUTCOME_ITEM_FIELDS_V2)
    assert {k: v for k, v in item.items() if k not in C.TASK_OUTCOME_ITEM_BINDING_FIELDS} == plain["items"][0]
    # the validator the writer and the reader share admits both shapes, exact
    completion.write_task_outcomes(tmp_path / "bound", GSM8K_TASK, bound)
    completion.write_task_outcomes(tmp_path / "plain", GSM8K_TASK, plain)
    assert completion.read_task_outcomes(tmp_path / "bound", GSM8K_TASK) == bound
    assert completion.read_task_outcomes(tmp_path / "plain", GSM8K_TASK) == plain
    # known-bad: a record mixing a bound and an unbound item; an item with HALF the pair; the pair on a task-scoped
    # (IFEval) shape for a GSM8K task — each refused by name
    mixed = _record([_result("q1", 0, rid="rid-A", cid="cmpl-A"), _result("q2", 0)])
    with pytest.raises(completion.CompletionError, match="shares one shape"):
        completion.write_task_outcomes(tmp_path / "mixed", GSM8K_TASK, mixed)
    half = json.loads(json.dumps(bound))
    del half["items"][0]["completion_id"]
    with pytest.raises(completion.CompletionError, match="not the contract"):
        completion.write_task_outcomes(tmp_path / "half", GSM8K_TASK, half)
    ifeval = json.loads(json.dumps(bound))
    ifeval["items"][0].update({k: 1 for k in C.IFEVAL_ROW_FACT_FIELDS}, sampling_ref="0" * 12)
    with pytest.raises(completion.CompletionError, match="not the contract"):
        completion.write_task_outcomes(tmp_path / "ifeval", GSM8K_TASK, ifeval)
    assert not (tmp_path / "mixed").exists() and not (tmp_path / "half").exists()
    # one authority spells the pair: the contract's fields ARE the row writers' fields
    assert C.TASK_OUTCOME_ITEM_BINDING_FIELDS == (rev.ROW_REQUEST_ID_FIELD, rev.ROW_COMPLETION_ID_FIELD) == ("request_id", "completion_id")


def test_the_rung_merge_writers_copy_the_binding_with_the_item():
    src = [{"item_id": "q1", "sample_idx": 0, "request_id": "rid-src", "completion_id": "cmpl-src", "value": 0.0}]
    rung = [{"item_id": "q1", "sample_idx": 0, "request_id": "rid-rung", "completion_id": "cmpl-rung", "value": 1.0}]
    merged = inspect_rung.merge_items(src, rung, ["q1"])
    assert merged == [{**rung[0], "draw": "rung"}]
    assert inspect_rung.merge_draws(src, rung, [("q1", 0)]) == [{**rung[0], "draw": "rung"}]
    assert inspect_rung.merge_items(src, [], []) == [{**src[0], "draw": "source"}]


def test_ws3g_binds_each_arms_draw_off_its_outcome_record_where_the_response_evidence_lookup_is_ambiguous(tmp_path):
    ac, rea = ws3g_modules()
    root = tmp_path / "attempt"
    root.mkdir()
    # two arms of the SHARED engine draw the same (item, sample): two attempts, the same (task, sample) label
    arms = {"none": ("rid-A", "cmpl-A", VMODEL), "arditi_allpos": ("rid-B", "cmpl-B", ARDITI_VMODEL)}
    records, responses, ledger = [], [], []
    for arm, (rid, cid, vm) in arms.items():
        raw, headers = wire(GSM8K_TASK, "q1", 0, payload=body(model=vm), request_id=rid)
        records.append({**C.request_evidence_record(raw, headers=headers), rev.EXTRA_FIELD: {"s": S, "attempt_id": ATTEMPT},
                        C.REQUEST_EVIDENCE_ATTEMPT_FIELD: ATTEMPT})
        responses.append({"request_id": rid, "completion_id": cid, "status": 200, "task": GSM8K_TASK, "sample": "q1/0"})
        ledger.append(ledger_row(rid, status=200, completion_id=cid))
    (root / C.REQUEST_EVIDENCE_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in records))
    (root / rev.RESPONSE_EVIDENCE_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in responses))

    def cell(arm: str, *, bound: bool):
        rid, cid, _vm = arms[arm]
        d = root / "cells" / arm
        d.mkdir(parents=True)
        (d / "rows.jsonl").write_text(json.dumps({"task": GSM8K_TASK, "item_id": "q1", "metric": "acc", "value": 1.0}) + "\n")
        (d / "manifest.json").write_text(json.dumps({"cell_id": arm, "seeds": {"inspect": {"replicate": {
            "policy": C.INSPECT_SEED_POLICY_REPLICATE, "seed_base": S}}}}))
        completion.write_task_outcomes(d, GSM8K_TASK, _record([_result("q1", 0, rid=rid if bound else None, cid=cid if bound else None)]))
        return d

    # RE-VALIDATION (the pre-fix record: no pair on the items): an id-less draw resolves through
    # `_bind_from_response_evidence` only when the response index holds exactly ONE completed attempt for its
    # (task, sample[, vmodel]) — at 71d4bae2 the key had no vmodel and two ARMS on `q1/0` were ambiguous; since
    # 8ecdd9a03 the vmodel axis separates arms, but Arditi's three SEED BASES share one vmodel and one (item, sample),
    # so the same refusal stands there. Either way the record's own pair short-circuits the lookup entirely.
    unbound = ac.Draw(task=GSM8K_TASK, item_id="q1", sample_idx=0, request_id=None, completion_id=None)
    one_vmodel = {(GSM8K_TASK, "q1/0", ARDITI_VMODEL): [("rid-s1", "cmpl-s1"), ("rid-s2", "cmpl-s2")]}
    with pytest.raises(ac.AuditRefusal, match="ambiguous"):
        ac._bind_from_response_evidence(unbound, one_vmodel, root, vmodel=ARDITI_VMODEL)
    bound_draw = ac.Draw(task=GSM8K_TASK, item_id="q1", sample_idx=0, request_id="rid-s1", completion_id="cmpl-s1")
    assert ac._bind_from_response_evidence(bound_draw, one_vmodel, root, vmodel=ARDITI_VMODEL) is bound_draw
    # ... and the pre-fix STORE (no pair on its items) yields id-less draws the audit then refuses by name
    plain_store = ac.read_tinycap_store(cell("none", bound=False), task=GSM8K_TASK, leg="gsm8k", arm="none")
    assert [(d.request_id, d.completion_id) for d in plain_store.draws] == [(None, None)]
    shutil.rmtree(root / "cells")
    # the fix: each arm's record carries its own draw's binding; the reader takes it off the record — no lookup needed
    stores = {arm: ac.read_tinycap_store(cell(arm, bound=True), task=GSM8K_TASK, leg="gsm8k", arm=arm) for arm in arms}
    for arm, (rid, cid, _vm) in arms.items():
        d = stores[arm].draws
        assert len(d) == 1 and (d[0].task, d[0].item_id, d[0].sample_idx) == (GSM8K_TASK, "q1", 0)
        assert (d[0].request_id, d[0].completion_id) == (rid, cid)
    # ... and each binds to its OWN attempt and exactly one engine-log request through the real bind_rows
    vm = {arm: v for arm, (_r, _c, v) in arms.items()}
    own = evidence(rea, records=records, ledger_rows=ledger, response_rows=responses, vmodels=vm)
    for arm in arms:
        assert rea.bind_rows(stores[arm], own=own, target_vmodel=vm[arm]) == (1, [])
    # known-bad: arm A's row claiming arm B's completion (torn), and A's draw judged against B's vmodel — each named
    torn = store_read(ac, draws=[ac.Draw(task=GSM8K_TASK, item_id="q1", sample_idx=0, request_id="rid-A", completion_id="cmpl-B")],
                      arm="none", leg="gsm8k", tasks=(GSM8K_TASK,))
    n, problems = rea.bind_rows(torn, own=own, target_vmodel=VMODEL)
    assert n == 0 and len(problems) == 1 and "torn binding" in problems[0]
    n, problems = rea.bind_rows(stores["none"], own=own, target_vmodel=ARDITI_VMODEL)
    assert n == 0 and "not the arm's" in problems[0]
