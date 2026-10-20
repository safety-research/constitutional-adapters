"""Engine mechanics: chunking, submit-all-before-poll, keyed harvest,
bounded retry rounds, stamp counts, max_retries=0 + nonce carriage."""

from __future__ import annotations

import json

import pytest

from conftest import make_engine
from corpus_generator.batch import (
    RETRYABLE_STATUSES,
    ChunkTooLarge,
    WaveHalt,
    WaveInvalidRequest,
    WaveRequest,
    outcome_status,
    plan_chunks,
    wire_tag,
)
from corpus_generator.schema import CustomIdError
from fakes import FakeBatchClient, canceled_result, error_result, expired_result, text_result


def reqs(n: int, prefix: str = "req", pad: int = 0) -> list[WaveRequest]:
    return [
        WaveRequest(custom_id=f"{prefix}-{i:03d}", params={"messages": "x" * pad})
        for i in range(n)
    ]


def echo_responder(cid, params, attempt):
    return text_result(cid, f"echo {cid}")


def test_plan_chunks_by_bytes_and_count():
    rs = reqs(10, pad=100)
    one = plan_chunks(rs, pack_target_bytes=10**6, hard_cap_bytes=10**6, max_requests=100)
    assert len(one) == 1 and len(one["c000"]) == 10

    size = rs[0].encoded_size()
    many = plan_chunks(rs, pack_target_bytes=size * 3, hard_cap_bytes=10**6, max_requests=100)
    assert len(many) == 4  # 3+3+3+1
    assert all(len(c) <= 3 for c in many.values())

    by_count = plan_chunks(rs, pack_target_bytes=10**6, hard_cap_bytes=10**6, max_requests=4)
    assert len(by_count) == 3

    with pytest.raises(ChunkTooLarge):
        plan_chunks(reqs(1, pad=1000), pack_target_bytes=100, hard_cap_bytes=200, max_requests=10)


def test_submit_all_before_poll_multi_chunk(tmp_path):
    client = FakeBatchClient(echo_responder)
    rs = reqs(9, pad=50)
    size = rs[0].encoded_size()
    engine = make_engine(client, pack_target_bytes=size * 3, hard_cap_bytes=10**6)
    out = engine.run_wave(tmp_path / "w", rs, wave_name="w")
    ops = client.op_sequence()
    assert ops.index("retrieve") > len([o for o in ops if o == "create"]) - 1
    last_create = max(i for i, o in enumerate(ops) if o == "create")
    first_retrieve = min(i for i, o in enumerate(ops) if o == "retrieve")
    assert last_create < first_retrieve, "every chunk must be created before any poll"
    assert len(out) == 9
    # per-chunk results files exist
    files = sorted(p.name for p in (tmp_path / "w").glob("results-*.jsonl"))
    assert len(files) == 3


def test_harvest_keyed_by_custom_id_shuffled(tmp_path):
    # responder order is store order; shuffle by reversing request insertion
    client = FakeBatchClient(echo_responder)
    orig_results = client.messages.batches.results

    def reversed_results(bid):
        return reversed(list(orig_results(bid)))

    client.messages.batches.results = reversed_results
    engine = make_engine(client)
    rs = reqs(5)
    out = engine.run_wave(tmp_path / "w", rs, wave_name="w")
    assert set(out) == {r.custom_id for r in rs}
    assert out["req-003"]["text"] == "echo req-003"


def test_retry_rounds_for_server_errors(tmp_path):
    fail_ids = {"req-001", "req-003"}

    def flaky(cid, params, attempt):
        if cid in fail_ids and attempt == 1:
            return error_result(cid) if cid == "req-001" else expired_result(cid)
        return text_result(cid, f"ok {cid}")

    client = FakeBatchClient(flaky)
    engine = make_engine(client)
    out = engine.run_wave(tmp_path / "w", reqs(5), wave_name="w")
    assert all(outcome_status(o) == "succeeded" for o in out.values())
    # exactly one follow-up sub-batch containing ONLY the failed ids
    creates = [d for op, d in client.calls if op == "create"]
    assert [c["n"] for c in creates] == [5, 2]
    stamp = json.loads((tmp_path / "w" / "STAMP.json").read_text())
    assert stamp["counts"] == {"succeeded": 5}
    assert len(stamp["batch_ids"]) == 2


def test_retry_rounds_bounded(tmp_path):
    def always_fails(cid, params, attempt):
        if cid == "req-000":
            return error_result(cid)
        return text_result(cid, "ok")

    client = FakeBatchClient(always_fails)
    engine = make_engine(client, retry_rounds=2)
    out = engine.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert outcome_status(out["req-000"]) == "errored"
    creates = [d["n"] for op, d in client.calls if op == "create"]
    assert creates == [3, 1, 1]  # base + exactly 2 bounded rounds
    stamp = json.loads((tmp_path / "w" / "STAMP.json").read_text())
    assert stamp["counts"] == {"succeeded": 2, "errored": 1}


def test_invalid_request_surfaces_and_stops(tmp_path):
    def invalid(cid, params, attempt):
        if cid == "req-001":
            return error_result(cid, error_type="invalid_request_error")
        return text_result(cid, "ok")

    client = FakeBatchClient(invalid)
    engine = make_engine(client)
    with pytest.raises(WaveInvalidRequest):
        engine.run_wave(tmp_path / "w", reqs(3), wave_name="w")


CONTENT_FILTER_MSG = "Output blocked by content filtering policy"


def test_content_filtered_output_tolerated_like_refusal(tmp_path):
    """Production 2026-09-01: the API's OUTPUT content filter surfaces as an
    invalid_request_error whose message names content filtering. That is a
    lost sample, not our bug: the wave completes, counts it in the stamp, and
    never engine-retries it (stage-level repair owns the second chance)."""

    def filtered(cid, params, attempt):
        if cid == "req-001":
            return error_result(
                cid, error_type="invalid_request_error", message=CONTENT_FILTER_MSG
            )
        return text_result(cid, "ok")

    client = FakeBatchClient(filtered)
    engine = make_engine(client)
    out = engine.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert outcome_status(out["req-001"]) == "content_filtered"
    creates = [d["n"] for op, d in client.calls if op == "create"]
    assert creates == [3], "content_filtered must not enter engine retry rounds"
    stamp = json.loads((tmp_path / "w" / "STAMP.json").read_text())
    assert stamp["counts"] == {"succeeded": 2, "content_filtered": 1}


def test_genuine_invalid_request_still_raises(tmp_path):
    """A recognized our-bug message (deterministic malformed-request shape)
    fails loudly on the FIRST occurrence — even under a maximally generous
    rejection cap."""

    def invalid(cid, params, attempt):
        if cid == "req-001":
            return error_result(
                cid, error_type="invalid_request_error",
                message="prompt is too long: 250000 tokens > 200000 maximum",
            )
        return text_result(cid, "ok")

    client = FakeBatchClient(invalid)
    engine = make_engine(client, max_rejected_frac=1.0)
    with pytest.raises(WaveInvalidRequest, match="our bug"):
        engine.run_wave(tmp_path / "w", reqs(3), wave_name="w")


REJECTED_MSG = "Invalid request data"  # the s5 production signature (generic)


def test_request_rejected_tolerated_under_cap(tmp_path):
    """Production s5 (2026-09-01): 2/17,846 well-formed score requests came
    back invalid_request_error / "Invalid request data" — content-dependent
    server-side rejection. Under max_rejected_frac it is per-request
    attrition: the wave completes, counts it in the stamp, and never
    engine-retries it (stage repair owns the fresh-id second chance)."""

    def rejected(cid, params, attempt):
        if cid == "req-001":
            return error_result(
                cid, error_type="invalid_request_error", message=REJECTED_MSG
            )
        return text_result(cid, "ok")

    client = FakeBatchClient(rejected)
    engine = make_engine(client, max_rejected_frac=0.5)  # 1/3 <= 50%
    out = engine.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert outcome_status(out["req-001"]) == "request_rejected"
    creates = [d["n"] for op, d in client.calls if op == "create"]
    assert creates == [3], "request_rejected must not enter engine retry rounds"
    stamp = json.loads((tmp_path / "w" / "STAMP.json").read_text())
    assert stamp["counts"] == {"succeeded": 2, "request_rejected": 1}


def test_request_rejected_over_cap_is_systematic_and_raises(tmp_path):
    """Above max_rejected_frac the rejection is systematic — our bug — and the
    wave fails loudly exactly as before. The default cap (0.1%) means any
    rejection in a small wave still halts, so fail-loud is preserved
    everywhere the long-tail argument doesn't hold."""

    def rejected(cid, params, attempt):
        if cid == "req-001":
            return error_result(
                cid, error_type="invalid_request_error", message=REJECTED_MSG
            )
        return text_result(cid, "ok")

    client = FakeBatchClient(rejected)
    engine = make_engine(client)  # default max_rejected_frac=0.001; 1/3 >> cap
    with pytest.raises(WaveInvalidRequest, match="max_rejected_frac"):
        engine.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert not (tmp_path / "w" / "STAMP.json").exists()


def test_create_uses_max_retries_zero_and_carries_nonce(tmp_path):
    client = FakeBatchClient(echo_responder)
    engine = make_engine(client, nonce_metadata=True)
    engine.run_wave(tmp_path / "w", reqs(2), wave_name="w", description="desc")
    assert client.last_create_max_retries == 0
    creates = [d for op, d in client.calls if op == "create"]
    assert creates[0]["metadata"] and "corpusgen_nonce" in creates[0]["metadata"]
    # the journaled nonce is the one carried on the wire
    ledger = [
        json.loads(line)
        for line in (tmp_path / "w" / "ledger.jsonl").read_text().splitlines()
    ]
    journaled_nonce = next(e["nonce"] for e in ledger if e["state"] == "journaled")
    assert creates[0]["metadata"]["corpusgen_nonce"] == journaled_nonce


def test_no_nonce_mode_sends_no_metadata(tmp_path):
    client = FakeBatchClient(echo_responder, metadata_supported=False)
    engine = make_engine(client, nonce_metadata=False)
    engine.run_wave(tmp_path / "w", reqs(2), wave_name="w")
    creates = [d for op, d in client.calls if op == "create"]
    assert creates[0]["metadata"] is None


def test_refusal_and_max_tokens_counted_not_retried(tmp_path):
    def statuses(cid, params, attempt):
        if cid == "req-000":
            return text_result(cid, "", stop_reason="refusal")
        if cid == "req-001":
            return text_result(cid, "truncated…", stop_reason="max_tokens")
        return text_result(cid, "ok")

    client = FakeBatchClient(statuses)
    engine = make_engine(client)
    out = engine.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert outcome_status(out["req-000"]) == "refusal"
    assert outcome_status(out["req-001"]) == "max_tokens"
    creates = [d["n"] for op, d in client.calls if op == "create"]
    assert creates == [3]  # never carried into a retry round
    stamp = json.loads((tmp_path / "w" / "STAMP.json").read_text())
    assert stamp["counts"] == {"succeeded": 1, "refusal": 1, "max_tokens": 1}


def test_canceled_is_not_retryable_and_halts(tmp_path):
    """A batch only cancels through an explicit batches.cancel — a deliberate
    operator action. The wave must neither resubmit those requests nor stamp;
    it halts for manual resolution."""
    assert "canceled" not in RETRYABLE_STATUSES

    def canceled(cid, params, attempt):
        if cid == "req-001":
            return canceled_result(cid)
        return text_result(cid, "ok")

    client = FakeBatchClient(canceled)
    engine = make_engine(client)
    with pytest.raises(WaveHalt, match="canceled"):
        engine.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    creates = [d["n"] for op, d in client.calls if op == "create"]
    assert creates == [3], "a canceled result must never be auto-resubmitted"
    assert not (tmp_path / "w" / "STAMP.json").exists()


def test_wire_tag_stable_and_charset_legal():
    tag = wire_tag("production-20260831")
    assert tag == wire_tag("production-20260831")
    assert len(tag) == 8
    assert tag != wire_tag("pilot-20260831")


def test_run_tagged_wire_id_must_fit_charset_length(tmp_path):
    """The 64-char custom_id ceiling applies to the TAGGED wire id."""
    client = FakeBatchClient(echo_responder)
    engine = make_engine(client, run_tag=wire_tag("x"))
    ok = WaveRequest(custom_id="a" * 55, params={})
    too_long = WaveRequest(custom_id="a" * 60, params={})  # 60 + 9 > 64
    engine.run_wave(tmp_path / "w1", [ok], wave_name="w1")
    with pytest.raises(CustomIdError):
        engine.run_wave(tmp_path / "w2", [too_long], wave_name="w2")


def test_duplicate_custom_ids_rejected(tmp_path):
    client = FakeBatchClient(echo_responder)
    engine = make_engine(client)
    rs = reqs(2) + [WaveRequest(custom_id="req-000", params={})]
    with pytest.raises(Exception, match="duplicate custom_ids"):
        engine.run_wave(tmp_path / "w", rs, wave_name="w")
