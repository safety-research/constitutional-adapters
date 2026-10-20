"""The Petri TARGET turns' attempt bindings reach the FILED store rows.

The finding, RE-VALIDATED as the first assertions: the hooks `maybe_install_petri_target_hooks` installs recorded each
attempt's `(request_id, completion_id)` in the two files and set `_LAST_BINDING`, but nothing carried the pairs into
sample metadata or `convert_eval_log`'s rows — `read_petri_store` made unbound draws and `bind_rows` rejected
every Petri draw ("the stored row carries no request_id / completion_id").

The fix: the response hook also files each Petri target attempt on the ACTIVE Inspect sample's store
(`rev.petri_sample_turn_sink`, `TaskState.store[rev.PETRI_TARGET_TURNS_STORE_KEY]` — inspect persists the store into
the eval log's `sample.store`), one entry per attempt in wire order at the TURN the request's sample label named
(retries and rollbacks are distinct entries); `convert_eval_log` copies the validated list onto every row of the
scenario under `extra_json.target_turns`; `rev.petri_target_turn_draws` turns rows into the audit's per-turn draws
(one per successful attempt) — the schema the reader adopts. Tested with
REAL httpx hooks on a loopback transport under a real inspect `TaskState`, the REAL converter over a synthetic eval log,
and the REAL `bind_rows` (loaded by file)."""
from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pytest

from seedrep_fixes_common import (ATTEMPT, S, VMODEL, body, evidence, hook_off, lane_on, ledger_row,  # noqa: F401
                                  read_jsonl, store_read, ws3g_modules)
from sisteer.capeval.tiers import petri_runner as pr
from sisteer.capeval.tinycap import contracts as C
from sisteer.eval import request_evidence_client as rev
from test_petri_store_shape import SIDS, _build_world, full_dims, make_log, sample

B = "self_preservation"
TASK = pr.PETRI_TASK_PREFIX + B
SID, OTHER_SID = SIDS[0], SIDS[1]
KEY = rev.PETRI_TARGET_TURNS_STORE_KEY


# --------------------------------------------------------------------------- #
# an ACTIVE Inspect sample (the real TaskState the eval log persists)          #
# --------------------------------------------------------------------------- #

@contextmanager
def active_sample(sid: str):
    from inspect_ai.model import ModelName
    from inspect_ai.solver._task_state import TaskState, _sample_state

    state = TaskState(model=ModelName("openai/" + VMODEL), sample_id=sid, epoch=1, input="hello", messages=[])
    token = _sample_state.set(state)
    try:
        yield state
    finally:
        _sample_state.reset(token)


def _row(task: str, sample_label: str, *, rid: str, cid: Optional[str], status: int) -> Dict[str, Any]:
    """A response row as `RequestEvidenceClient.record_response` writes it (the fields the sink reads)."""
    return {"ts": "2026-09-18T00:00:00Z", "request_id": rid, "completion_id": cid, "status": status, "vmodel": VMODEL,
            "finish_reason": "stop" if cid else None, "task": task, "sample": sample_label, "error": None if cid else "boom"}


def test_the_sink_files_every_target_attempt_on_the_active_sample_in_wire_order_and_refuses_a_homeless_one():
    pytest.importorskip("inspect_ai")   # Modal CPU image: the tinycap client venv is not installed there (main's test_petri_auditor_retry_bounds idiom)
    with active_sample(SID) as state:
        assert state.store.get(KEY) is None
        # turn 1 ok; turn 2 fails (500) then its retry succeeds; turn 3; the auditor rolls back and turn 2 is served again
        rev.petri_sample_turn_sink(_row(TASK, f"{SID}/1", rid="r1", cid="cmpl-1", status=200))
        rev.petri_sample_turn_sink(_row(TASK, f"{SID}/2", rid="r2", cid=None, status=500))
        rev.petri_sample_turn_sink(_row(TASK, f"{SID}/2", rid="r3", cid="cmpl-2", status=200))
        rev.petri_sample_turn_sink(_row(TASK, f"{SID}/3", rid="r4", cid="cmpl-3", status=200))
        rev.petri_sample_turn_sink(_row(TASK, f"{SID}/2", rid="r5", cid="cmpl-4", status=200))
        assert state.store.get(KEY) == [
            {"turn": 1, "request_id": "r1", "completion_id": "cmpl-1", "status": 200, "task": TASK},
            {"turn": 2, "request_id": "r2", "completion_id": None, "status": 500, "task": TASK},
            {"turn": 2, "request_id": "r3", "completion_id": "cmpl-2", "status": 200, "task": TASK},
            {"turn": 3, "request_id": "r4", "completion_id": "cmpl-3", "status": 200, "task": TASK},
            {"turn": 2, "request_id": "r5", "completion_id": "cmpl-4", "status": 200, "task": TASK},
        ]
        # a row of another task is not a Petri target attempt: a no-op, the store untouched
        rev.petri_sample_turn_sink(_row("inspect/gsm8k_paired200", "q1/0", rid="r9", cid="cmpl-9", status=200))
        assert len(state.store.get(KEY)) == 5
        # known-bad: the label names ANOTHER scenario than the active sample — never filed on this record
        with pytest.raises(rev.RequestEvidenceError, match="active Inspect sample"):
            rev.petri_sample_turn_sink(_row(TASK, f"{OTHER_SID}/1", rid="r6", cid="cmpl-5", status=200))
        with pytest.raises(rev.RequestEvidenceError, match="<scenario_id>/<turn>"):
            rev.petri_sample_turn_sink(_row(TASK, "malformed", rid="r7", cid="cmpl-6", status=200))
        assert len(state.store.get(KEY)) == 5
    # known-bad: outside any sample the binding has no durable home — refused, never dropped
    with pytest.raises(rev.RequestEvidenceError, match="outside an Inspect sample"):
        rev.petri_sample_turn_sink(_row(TASK, f"{SID}/1", rid="r8", cid="cmpl-7", status=200))


# --------------------------------------------------------------------------- #
# the REAL hooks on a loopback transport, under the sample                     #
# --------------------------------------------------------------------------- #

PLAN: List[Tuple[int, bool]] = [(1, True), (2, False), (2, True), (3, True), (2, True)]   # (turn, succeeds)


def drive_petri_turns(client: rev.RequestEvidenceClient, sid: str = SID) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Five TARGET attempts of one scenario through `install_httpx(..., on_binding=petri_sample_turn_sink)` — the
    two labels on each request (the `TargetTurn` through `TARGET_TURN_HEADERS_HOOK`), the id minted by the hook — against
    a loopback engine that answers `cmpl-<n>` or 500s the flagged attempt. Returns (the sample state, requests seen,
    responses sent)."""
    n = {"ok": 0}
    seen: List[Dict[str, Any]] = []
    sent: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        seen.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        if request.headers.get("x-test-fail") == "1":
            sent.append({"status": 500})
            return httpx.Response(500, json={"error": {"message": "engine hiccup"}})
        n["ok"] += 1
        doc = {"id": f"cmpl-{n['ok']}", "object": "chat.completion", "model": VMODEL,
               "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": f"reply {n['ok']}"}}]}
        sent.append({"status": 200, **doc})
        return httpx.Response(200, json=doc)

    async def go(state):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://loopback") as hc:
            client.install_httpx(hc, on_binding=rev.petri_sample_turn_sink)
            for turn, ok in PLAN:
                headers = {**rev.label_headers(TASK, sid, turn), "content-type": "application/json"}
                if not ok:
                    headers["x-test-fail"] = "1"
                await hc.post("/v1/chat/completions", content=json.dumps(body(turn_messages=turn)).encode(), headers=headers)
        return state

    with active_sample(sid) as state:
        asyncio.run(go(state))
    return state, seen, sent


def test_real_httpx_hooks_file_the_turn_binding_on_the_sample_and_the_two_files(tmp_path, monkeypatch):
    pytest.importorskip("inspect_ai")   # Modal CPU image: the tinycap client venv is not installed there (main's test_petri_auditor_retry_bounds idiom)
    client = lane_on(monkeypatch, tmp_path / "attempt", extra={"leg": "petri", "arm": "none", "s": S, "attempt_id": ATTEMPT})
    state, seen, sent = drive_petri_turns(client)
    turns = state.store.get(KEY)
    assert [(t["turn"], t["status"], t["completion_id"]) for t in turns] == [
        (1, 200, "cmpl-1"), (2, 500, None), (2, 200, "cmpl-2"), (3, 200, "cmpl-3"), (2, 200, "cmpl-4")]
    rids = [t["request_id"] for t in turns]
    assert len(set(rids)) == 5 and all(t["task"] == TASK for t in turns)
    # the ids on the store ARE the ids on the wire (the hook minted them onto the requests) ...
    assert rids == [s["headers"][C.HEADER_REQUEST_ID.lower()] for s in seen]
    # ... and the two files hold the same five attempts: every record stamped extra.s + attempt_id, every response bound
    records = read_jsonl(client.request_path)
    responses = read_jsonl(client.response_path)
    assert [r["request_id"] for r in records] == rids == [r["request_id"] for r in responses]
    assert all(r["extra"]["s"] == S and r[C.REQUEST_EVIDENCE_ATTEMPT_FIELD] == ATTEMPT for r in records)
    assert [(r["task"], r["item_id"], r["sample_idx"]) for r in records] == [(TASK, SID, t) for t, _ in PLAN]
    assert [r["completion_id"] for r in responses] == [t["completion_id"] for t in turns]


# --------------------------------------------------------------------------- #
# the REAL converter: the list rides every row of the scenario, none without   #
# --------------------------------------------------------------------------- #

def _convert(world, cell, samples):
    log = make_log(world, cell, B, samples)
    rows, ledger = pr.convert_eval_log(log, spec=world.spec, cell=cell, dir_key=B, canonical_ids=SIDS, env_digest="ee" * 8)
    assert ledger == []
    return rows


def _turns_fixture() -> List[Dict[str, Any]]:
    return [{"turn": 1, "request_id": "r1", "completion_id": "cmpl-1", "status": 200, "task": TASK},
            {"turn": 2, "request_id": "r2", "completion_id": None, "status": 500, "task": TASK},
            {"turn": 2, "request_id": "r3", "completion_id": "cmpl-2", "status": 200, "task": TASK}]


def test_convert_eval_log_carries_the_turn_bindings_on_every_row_of_the_scenario_and_no_key_without_them(tmp_path_factory):
    world = _build_world(tmp_path_factory.mktemp("petri-turns-world"))
    cell = world.cells["none"]
    turns = _turns_fixture()
    with_store = [{**sample(SID, full_dims(B)), "store": {KEY: turns}}, sample(OTHER_SID, full_dims(B))]
    rows = _convert(world, cell, with_store)
    plain = _convert(world, cell, [sample(SID, full_dims(B)), sample(OTHER_SID, full_dims(B))])
    assert len(rows) == len(plain) == 2 * len(pr.required_metrics(B))
    by_item = lambda rs: {(r.item_id, r.metric): r for r in rs}  # noqa: E731
    got, ref = by_item(rows), by_item(plain)
    for (item, metric), r in got.items():
        extra = json.loads(r.extra_json)
        if item == SID:
            # every dimension row of the scenario carries the SAME list, verbatim, beside the unchanged keys
            assert extra[rev.PETRI_TARGET_TURNS_ROW_KEY] == turns
            extra.pop(rev.PETRI_TARGET_TURNS_ROW_KEY)
            assert extra == json.loads(ref[(item, metric)].extra_json)
        else:
            # a sample without the store key: byte-identical to the pre-fix row
            assert rev.PETRI_TARGET_TURNS_ROW_KEY not in extra and r.extra_json == ref[(item, metric)].extra_json
    # the audit's per-turn draws: one per SUCCESSFUL attempt, de-duplicated across the dimension rows
    result = rev.petri_target_turn_draws([{"task": r.task, "item_id": r.item_id, "extra_json": r.extra_json} for r in rows if r.item_id == SID])
    assert result.draws == [{"task": TASK, "item_id": SID, "sample_idx": 1, "request_id": "r1", "completion_id": "cmpl-1"},
                            {"task": TASK, "item_id": SID, "sample_idx": 2, "request_id": "r3", "completion_id": "cmpl-2"}]
    assert (result.n_successful, result.n_failed, result.n_scenarios) == (2, 1, 1) and len(result.draws) == result.n_successful
    # fail closed: the other scenario's rows carry NO list — over ALL rows the helper refuses, never fewer draws
    with pytest.raises(rev.RequestEvidenceError, match="carries no target_turns"):
        rev.petri_target_turn_draws([{"task": r.task, "item_id": r.item_id, "extra_json": r.extra_json} for r in rows])
    # known-bad: a binding filed under ANOTHER behaviour's task, or a store value that is not a list — refused by name
    foreign = [{**turns[0], "task": pr.PETRI_TASK_PREFIX + "delusion_sycophancy"}]
    with pytest.raises(pr.PetriRunnerError, match="another behaviour's binding"):
        _convert(world, cell, [{**sample(SID, full_dims(B)), "store": {KEY: foreign}}])
    with pytest.raises(pr.PetriRunnerError, match="not a list"):
        _convert(world, cell, [{**sample(SID, full_dims(B)), "store": {KEY: {"turn": 1}}}])
    with pytest.raises(pr.PetriRunnerError, match="turn 0"):
        _convert(world, cell, [{**sample(SID, full_dims(B)), "store": {KEY: [{**turns[0], "turn": 0}]}}])


# --------------------------------------------------------------------------- #
# The binding audit over a REAL converted scenario                             #
# --------------------------------------------------------------------------- #

def test_ws3g_bind_rows_binds_every_served_turn_of_a_converted_scenario_and_names_a_torn_or_rejected_one(
        tmp_path, tmp_path_factory, monkeypatch):
    pytest.importorskip("inspect_ai")   # Modal CPU image: the tinycap client venv is not installed there (main's test_petri_auditor_retry_bounds idiom)
    ac, rea = ws3g_modules()
    client = lane_on(monkeypatch, tmp_path / "attempt", extra={"leg": "petri", "arm": "none", "s": S, "attempt_id": ATTEMPT})
    state, _seen, _sent = drive_petri_turns(client)
    records, responses = read_jsonl(client.request_path), read_jsonl(client.response_path)
    ledger = [ledger_row(r["request_id"], status=r["status"], completion_id=r["completion_id"]) for r in responses]
    own = evidence(rea, records=records, ledger_rows=ledger, response_rows=responses, vmodels={"none": VMODEL})
    # the eval log persists the sample store; the converter puts the list on the rows
    world = _build_world(tmp_path_factory.mktemp("petri-bind-world"))
    rows = _convert(world, world.cells["none"], [{**sample(SID, full_dims(B)), "store": {KEY: list(state.store.get(KEY))}}])
    row_dicts = [{"task": r.task, "item_id": r.item_id, "extra_json": r.extra_json} for r in rows]
    # RE-VALIDATION: the CURRENT single-binding reader sees no pair on these rows (the list is the schema it adopts)
    assert ac._binding(row_dicts[0]) == (None, None)
    # the fix's contract: one draw per served turn — every one binds to its attempt and to exactly one engine request
    result = rev.petri_target_turn_draws(row_dicts)
    assert (result.n_successful, result.n_failed, result.n_scenarios) == (4, 1, 1) and len(result.draws) == 4
    draws = [ac.Draw(**d) for d in result.draws]
    assert [(d.sample_idx, d.completion_id) for d in draws] == [(1, "cmpl-1"), (2, "cmpl-2"), (3, "cmpl-3"), (2, "cmpl-4")]
    n_bound, problems = rea.bind_rows(store_read(ac, draws=draws, arm="none", leg="petri", tasks=(TASK,)), own=own, target_vmodel=VMODEL)
    assert (n_bound, problems) == (4, [])
    # known-bad: a torn binding (turn 1's request with turn 3's completion) and the REJECTED attempt forced in as a draw
    torn = [ac.Draw(task=TASK, item_id=SID, sample_idx=1, request_id=draws[0].request_id, completion_id="cmpl-3")]
    n_bound, problems = rea.bind_rows(store_read(ac, draws=torn, arm="none", leg="petri", tasks=(TASK,)), own=own, target_vmodel=VMODEL)
    assert n_bound == 0 and len(problems) == 1 and "torn binding" in problems[0]
    failed = next(r for r in responses if r["status"] == 500)
    rejected = [ac.Draw(task=TASK, item_id=SID, sample_idx=2, request_id=failed["request_id"], completion_id="cmpl-2")]
    n_bound, problems = rea.bind_rows(store_read(ac, draws=rejected, arm="none", leg="petri", tasks=(TASK,)), own=own, target_vmodel=VMODEL)
    assert n_bound == 0 and "REJECTED ledger attempt" in problems[0]
    # and the pre-fix reality, stated: a Petri draw read the old way (no pair) is exactly what bind_rows refuses
    unbound = [ac.Draw(task=TASK, item_id=SID, sample_idx=0, request_id=None, completion_id=None)]
    assert "carries no request_id / completion_id" in rea.bind_rows(store_read(ac, draws=unbound, arm="none", leg="petri", tasks=(TASK,)),
                                                                     own=own, target_vmodel=VMODEL)[1][0]
