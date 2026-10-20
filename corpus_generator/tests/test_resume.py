"""Resume logic: stamped waves skip, submitted-unharvested waves re-attach,
harvested-unstamped waves re-derive outcomes from STORED results, hash
mismatch within a run+wave fails loudly."""

from __future__ import annotations

import json

import pytest

import corpus_generator.batch as batch_mod
from conftest import make_engine
from corpus_generator.batch import WaveHashMismatch, WaveRequest, outcome_status
from fakes import FakeBatchClient, error_result, text_result


def reqs(n: int, salt: str = "") -> list[WaveRequest]:
    return [
        WaveRequest(custom_id=f"req-{i:03d}", params={"i": i, "salt": salt})
        for i in range(n)
    ]


def echo(cid, params, attempt):
    return text_result(cid, f"echo {cid}")


def test_stamped_wave_skips_all_api_calls(tmp_path):
    client = FakeBatchClient(echo)
    engine = make_engine(client)
    out1 = engine.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    n_calls = len(client.calls)
    out2 = engine.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert out1 == out2
    assert len(client.calls) == n_calls, "a stamped wave must not touch the API"


def test_submitted_unharvested_resumes_by_polling(tmp_path):
    client = FakeBatchClient(echo)
    client.retrieve_script = ["raise"]  # first poll dies after submit
    engine1 = make_engine(client)
    with pytest.raises(Exception):
        engine1.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert client.create_count() == 1

    engine2 = make_engine(client)
    out = engine2.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert client.create_count() == 1, "resume polls the exact submitted batch id"
    assert engine2.reconcile_count == 0
    assert len(out) == 3


class Crash(BaseException):
    """Not an Exception so nothing in the engine can swallow it."""


def test_harvested_unstamped_wave_reclassifies_stored_results_on_resume(
    tmp_path, monkeypatch
):
    """The production 2026-09-01 halt states (s3 and s5 alike): every chunk
    harvested (per-chunk results + raw_results.jsonl on disk), no STAMP — the
    engine died between harvest and stamp. Resume must re-derive outcomes from
    the STORED results under the current classification (content_filtered AND
    request_rejected tolerated) and stamp, with ZERO new creates and ZERO
    re-harvests for the ended chunks."""

    def responder(cid, params, attempt):
        if cid == "req-001":  # the s3 halt's signature
            return error_result(
                cid, error_type="invalid_request_error",
                message="Output blocked by content filtering policy",
            )
        if cid == "req-004":  # the s5 halt's signature
            return error_result(
                cid, error_type="invalid_request_error",
                message="Invalid request data",
            )
        return text_result(cid, f"ok {cid}")

    client = FakeBatchClient(responder)
    rs = reqs(6)
    size = rs[0].encoded_size()
    engine_kw = dict(
        pack_target_bytes=size * 2, hard_cap_bytes=10**6, max_rejected_frac=0.5
    )

    # die exactly at the stamp write -> harvested chunks, raw_results, no STAMP
    orig_write = batch_mod.atomic_write

    def crash_on_stamp(path, text):
        if path.name == "STAMP.json":
            raise Crash("pre-stamp")
        orig_write(path, text)

    monkeypatch.setattr(batch_mod, "atomic_write", crash_on_stamp)
    engine1 = make_engine(client, **engine_kw)
    with pytest.raises(Crash):
        engine1.run_wave(tmp_path / "w", rs, wave_name="w")
    monkeypatch.undo()

    assert not (tmp_path / "w" / "STAMP.json").exists()
    assert (tmp_path / "w" / "raw_results.jsonl").exists()
    n_creates = client.create_count()
    n_harvests = sum(1 for op, _ in client.calls if op == "results")
    assert n_creates >= 2  # multi-chunk wave, all ended

    engine2 = make_engine(client, **engine_kw)
    out = engine2.run_wave(tmp_path / "w", rs, wave_name="w")
    assert client.create_count() == n_creates, "ended chunks must not be resubmitted"
    assert sum(1 for op, _ in client.calls if op == "results") == n_harvests, (
        "outcomes must come from the STORED results, not a re-harvest"
    )
    assert engine2.reconcile_count == 0
    assert outcome_status(out["req-001"]) == "content_filtered"
    assert outcome_status(out["req-004"]) == "request_rejected"
    stamp = json.loads((tmp_path / "w" / "STAMP.json").read_text())
    assert stamp["counts"] == {
        "succeeded": 4, "content_filtered": 1, "request_rejected": 1,
    }


def test_stamped_hash_mismatch_fails_loudly(tmp_path):
    client = FakeBatchClient(echo)
    engine = make_engine(client)
    engine.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    with pytest.raises(WaveHashMismatch, match="immutable"):
        engine.run_wave(tmp_path / "w", reqs(3, salt="drifted"), wave_name="w")


def test_inflight_hash_mismatch_fails_loudly(tmp_path):
    client = FakeBatchClient(echo)
    client.retrieve_script = ["raise"]
    engine1 = make_engine(client)
    with pytest.raises(Exception):
        engine1.run_wave(tmp_path / "w", reqs(3), wave_name="w")

    engine2 = make_engine(client)
    with pytest.raises(WaveHashMismatch):
        engine2.run_wave(tmp_path / "w", reqs(3, salt="drifted"), wave_name="w")


def test_template_hash_participates_in_wave_hash(tmp_path):
    client = FakeBatchClient(echo)
    engine = make_engine(client)
    engine.run_wave(
        tmp_path / "w", reqs(2), wave_name="w", template_hashes={"t.md": "aaa"}
    )
    with pytest.raises(WaveHashMismatch):
        engine.run_wave(
            tmp_path / "w", reqs(2), wave_name="w", template_hashes={"t.md": "bbb"}
        )
