"""The Inspect provider's REQUEST sink under the lane switch.

The finding, RE-VALIDATED here as the first assertion: with `SISTEER_REQUEST_EVIDENCE_DIR` set, `run_inspect_tasks`
selected the lane client's RESPONSE sink but passed the caller's REQUEST sink through unchanged — production's
`_RequestEvidence.record_wire` (`scripts/served_capability_run.py`), which writes the v3 record + `attempt_id` and never
`extra.s`. `audit_root` refuses every Arditi task record without `extra.s` (three seed bases share one task on
that engine), so an Arditi GSM8K run produced unattributable request records; requiring `seed_base` at the lane client's
construction protected nothing written by the OTHER writer.

The fix (`inspect_served.request_evidence_sinks`, called by `run_inspect_tasks`): OFF = the caller's sinks, untouched;
ON = BOTH sides are the lane client's, whatever the caller passed — its request sink is not called for the provider's
requests (no double-recording), must append to the SAME file (`find_one` refuses two request_evidence.jsonl
under one root) and run as the same attempt (the lane client stamps production's top-level `attempt_id` so
`--attempt-id` scoping — `records_of_attempts` — still admits every Inspect record). Driven with PRODUCTION's real writer."""
from __future__ import annotations

import inspect
import json

import pytest

from seedrep_fixes_common import (ATTEMPT, GSM8K_TASK, S, hook_off, lane_on, load_served_capability_run,  # noqa: F401
                                  read_jsonl, wire, ws3g_modules)
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tiers import inspect_served as isv
from sisteer.eval import request_evidence_client as rev


def test_off_the_callers_sinks_pass_through_untouched_and_production_writes_what_it_wrote(tmp_path):
    srun = load_served_capability_run()
    prod = srun._RequestEvidence(tmp_path / "out" / C.REQUEST_EVIDENCE_FILENAME, attempt_id=ATTEMPT)
    # the identity: production's bound method comes back as itself; no sink stays no sink
    req, resp = isv.request_evidence_sinks(prod.record_wire, None)
    assert req == prod.record_wire and resp is None
    assert isv.request_evidence_sinks(None, None) == (None, None)
    # ... and production's writer, driven as the provider's request hook drives it, writes exactly its record: the v3
    # fields + its attempt stamp, no `extra` (the default-off bytes this fix must preserve)
    raw, headers = wire(GSM8K_TASK, "q1", 0)
    rec = prod.record_wire(raw, headers=headers)
    assert rec[C.REQUEST_EVIDENCE_ATTEMPT_FIELD] == ATTEMPT and rev.EXTRA_FIELD not in rec
    assert [json.loads(l) for l in prod.path.read_text().splitlines()] == [rec] and prod.n == 1
    assert rec["body_sha256"] == C.wire_body_sha256(raw) and rec["task"] == GSM8K_TASK and rec["item_id"] == "q1"


def test_on_the_production_request_sink_is_routed_through_the_lane_recorder(tmp_path, monkeypatch):
    ac, rea = ws3g_modules()
    srun = load_served_capability_run()
    root = tmp_path / "attempt"
    root.mkdir()
    prod = srun._RequestEvidence(root / C.REQUEST_EVIDENCE_FILENAME, attempt_id=ATTEMPT)
    raw, headers = wire(GSM8K_TASK, "q1", 0)
    # RE-VALIDATION of the finding: production's record alone carries no extra.s — exactly what audit_root refuses on an
    # Arditi root (in memory; the file stays untouched for the fix's half below)
    bare = {**C.request_evidence_record(raw, headers=headers), C.REQUEST_EVIDENCE_ATTEMPT_FIELD: ATTEMPT}
    assert bare["kind"] == C.REQUEST_EVIDENCE_KIND_TASK and rea.record_extra_s(bare) is None

    # the fix: lane ON (the env on the driver child) → the provider records BOTH sides through the lane client
    client = lane_on(monkeypatch, root, extra={"leg": "gsm8k", "arm": "arditi_allpos", "s": S, "attempt_id": ATTEMPT})
    req, resp = isv.request_evidence_sinks(prod.record_wire, None)
    assert req == client.record_request and resp == client.response_sink
    # driven exactly as `_wire_request_hook` drives its sink (the exact bytes + the headers with the minted id)
    rec = req(raw, headers=headers)
    lines = read_jsonl(root / C.REQUEST_EVIDENCE_FILENAME)
    assert lines == [rec]                                            # ONE record, in ONE file
    assert rea.record_extra_s(rec) == S and rec["extra"]["s"] == S    # the seed-base discriminator the audit partitions by
    assert rec[C.REQUEST_EVIDENCE_ATTEMPT_FIELD] == ATTEMPT           # production's stamp, kept
    assert rec["task"] == GSM8K_TASK and rec["item_id"] == "q1" and rec["sample_idx"] == 0
    assert rec["body_sha256"] == C.wire_body_sha256(raw) and rec["request_id"] == headers[C.HEADER_REQUEST_ID]
    assert rec["vmodel"] == json.loads(raw)["model"] and rec["presence_penalty_present"] is True and rec["presence_penalty_value"] == 1.5
    # no double-recording: production's writer never ran for the provider's request (its own counter is 0) ...
    assert prod.n == 0
    # ... yet production reads the shared file and its --attempt-id scoping admits the record (the stamp it requires)
    assert prod.records() == [rec] and C.records_of_attempts(prod.records(), {ATTEMPT}) == [rec]
    # the probes production still records itself land in the SAME inventory, unstamped by the lane (not task records)
    raw_p, headers_p = wire(C.probe_task_label(C.PROBE_KIND_BITES), "petri__base", 0)
    probe = prod.record_wire(raw_p, headers=headers_p)
    assert read_jsonl(root / C.REQUEST_EVIDENCE_FILENAME) == [rec, probe] and rev.EXTRA_FIELD not in probe
    # a caller-given response sink is kept; a driver with NO request sink still records both sides under the lane
    marker = object()
    assert isv.request_evidence_sinks(prod.record_wire, marker)[1] is marker
    assert isv.request_evidence_sinks(None, None) == (client.record_request, client.response_sink)


def test_on_a_production_writer_at_another_path_or_another_attempt_is_refused_by_name(tmp_path, monkeypatch):
    srun = load_served_capability_run()
    root, other = tmp_path / "attempt", tmp_path / "elsewhere"
    root.mkdir()
    other.mkdir()
    lane_on(monkeypatch, root, extra={"s": S, "attempt_id": ATTEMPT})
    with pytest.raises(isv.InspectServedError, match="two request_evidence.jsonl"):
        isv.request_evidence_sinks(srun._RequestEvidence(other / C.REQUEST_EVIDENCE_FILENAME, attempt_id=ATTEMPT).record_wire, None)
    with pytest.raises(isv.InspectServedError, match="one run has one attempt id"):
        isv.request_evidence_sinks(srun._RequestEvidence(root / C.REQUEST_EVIDENCE_FILENAME, attempt_id="ta-other").record_wire, None)
    assert not (root / C.REQUEST_EVIDENCE_FILENAME).exists() and not (other / C.REQUEST_EVIDENCE_FILENAME).exists()


def test_on_the_lane_client_adopts_the_drivers_attempt_id_when_its_extra_names_none(tmp_path, monkeypatch):
    srun = load_served_capability_run()
    root = tmp_path / "attempt"
    root.mkdir()
    client = lane_on(monkeypatch, root, extra={"leg": "gsm8k", "s": S})          # the driver's extra without attempt_id
    assert client.attempt_id is None
    prod = srun._RequestEvidence(root / C.REQUEST_EVIDENCE_FILENAME, attempt_id="ta-99999999-0000")
    req, _ = isv.request_evidence_sinks(prod.record_wire, None)
    rec = req(*wire(GSM8K_TASK, "q7", 1))
    assert rec[C.REQUEST_EVIDENCE_ATTEMPT_FIELD] == "ta-99999999-0000" == rec["extra"]["attempt_id"] and rec["extra"]["s"] == S
    assert C.records_of_attempts([rec], {"ta-99999999-0000"}) == [rec]
    # a run with no attempt id anywhere (a driver without --attempt-id) writes no stamp — the pre-fix record shape
    client2 = lane_on(monkeypatch, tmp_path / "b", extra={"s": S})
    rec2 = client2.record_request(*wire(GSM8K_TASK, "q7", 1))
    assert C.REQUEST_EVIDENCE_ATTEMPT_FIELD not in rec2 and rec2["extra"] == {"s": S}
    with pytest.raises(C.AttemptScopeError):
        C.records_of_attempts([rec2], {"x"})


def test_run_inspect_tasks_selects_its_sinks_through_the_one_function():
    """The call site: `run_inspect_tasks` hands BOTH caller sinks to `request_evidence_sinks` and uses what comes back
    (the pre-fix branch — the response side only, `from_env` inline — is gone)."""
    src = inspect.getsource(isv.run_inspect_tasks)
    assert "request_evidence_sink, response_evidence_sink = request_evidence_sinks(request_evidence_sink, response_evidence_sink)" in src
    assert "_rev_client.response_sink" not in src
    assert src.index("request_evidence_sinks(") < src.index("target = build_model(")
