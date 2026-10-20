"""P1-4 crash-idempotence. Core property: crash-before-create, transport-error
-during-create, and crash-after-create-before-`submitted` leave the SAME
durable state (journaled-terminal) and take the SAME conservative resume path
(reconciliation — never straight to re-create)."""

from __future__ import annotations

import json

import pytest

from conftest import make_engine
from corpus_generator.batch import Ledger, WaveError, WaveHalt, WaveRequest
from fakes import FakeBatchClient, text_result


def reqs(n: int) -> list[WaveRequest]:
    return [WaveRequest(custom_id=f"req-{i:03d}", params={"i": i}) for i in range(n)]


def echo(cid, params, attempt):
    return text_result(cid, f"echo {cid}")


class Crash(BaseException):
    """Not an Exception so nothing in the engine can swallow it."""


def crash_at(point: str):
    def hooks(p, info):
        if p == point:
            raise Crash(p)

    return hooks


INJECTION_POINTS = [
    ("pre_create", "crash before create (server has NO batch)", False),
    ("transport_no_create", "transport error, not created", False),
    ("transport_after_create", "transport error, created server-side", True),
    ("post_create", "crash after create before `submitted` append", True),
]


@pytest.mark.parametrize("point,desc,server_has_batch", INJECTION_POINTS)
def test_all_submit_crashes_leave_journaled_and_reconcile(
    tmp_path, point, desc, server_has_batch
):
    client = FakeBatchClient(echo)
    if point in ("transport_no_create", "transport_after_create"):
        client.create_script = [point]
        engine1 = make_engine(client)
        # transport failures surface as APIConnectionError -> reconcile happens
        # in-process on the same run; to test the RESUME path, make the engine
        # die before reconciliation by crashing its visibility sleep
        engine1.sleep = _raise_crash
    else:
        engine1 = make_engine(client, hooks=crash_at(point))

    with pytest.raises(Crash):
        engine1.run_wave(tmp_path / "w", reqs(3), wave_name="w")

    # identical durable state at every injection point: journaled-terminal
    ledger = Ledger(tmp_path / "w" / "ledger.jsonl")
    st = ledger.chunk_state("c000")
    assert st.state == "journaled", desc
    assert st.nonce

    # resume: reconciliation ALWAYS runs; re-create only on proven absence
    engine2 = make_engine(client)
    out = engine2.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert engine2.reconcile_count == 1, f"{desc}: resume must reconcile"
    assert set(out) == {"req-000", "req-001", "req-002"}
    if server_has_batch:
        assert client.create_count() == 1, f"{desc}: adopted, no second create"
        adopted = [
            e for e in ledger.entries() if e["state"] == "submitted" and e.get("adopted")
        ]
        assert adopted, "adoption must be recorded in the ledger"
    else:
        assert client.create_count() == 1, f"{desc}: proven absence -> exactly one create"
        # fresh nonce on the re-create attempt
        journaled = [e for e in ledger.entries() if e["state"] == "journaled"]
        assert len(journaled) == 2
        assert journaled[0]["nonce"] != journaled[1]["nonce"]


def _raise_crash(_s):
    raise Crash("visibility-sleep")


def test_no_nonce_zero_candidates_halts(tmp_path):
    client = FakeBatchClient(echo, metadata_supported=False)
    engine1 = make_engine(client, nonce_metadata=False, hooks=crash_at("pre_create"))
    with pytest.raises(Crash):
        engine1.run_wave(tmp_path / "w", reqs(3), wave_name="w")

    engine2 = make_engine(client, nonce_metadata=False)
    with pytest.raises(WaveHalt, match="0 candidate"):
        engine2.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    # NEVER auto-resubmits on the no-nonce branch
    assert client.create_count() == 0


def test_no_nonce_unique_candidate_adopted(tmp_path):
    client = FakeBatchClient(echo, metadata_supported=False)
    client.create_script = ["transport_after_create"]
    engine1 = make_engine(client, nonce_metadata=False)
    engine1.sleep = _raise_crash
    with pytest.raises(Crash):
        engine1.run_wave(tmp_path / "w", reqs(3), wave_name="w")

    engine2 = make_engine(client, nonce_metadata=False)
    out = engine2.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert client.create_count() == 1
    assert len(out) == 3


def test_no_nonce_multiple_candidates_halt(tmp_path):
    client = FakeBatchClient(echo, metadata_supported=False)
    client.create_script = ["transport_after_create"]
    engine1 = make_engine(client, nonce_metadata=False)
    engine1.sleep = _raise_crash
    with pytest.raises(Crash):
        engine1.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    # a second unledgered batch with the same request count in the window
    client.register_foreign_batch(3)

    engine2 = make_engine(client, nonce_metadata=False)
    with pytest.raises(WaveHalt, match="2 candidate"):
        engine2.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert client.create_count() == 1  # nothing new created


def test_wrong_adoption_rolled_back_at_harvest(tmp_path):
    """no-nonce; the only window candidate is a DECOY with matching count but
    different custom_ids -> adopted, then harvest set-equality rolls it back
    and re-reconciliation (decoy excluded) halts."""
    client = FakeBatchClient(echo, metadata_supported=False)
    engine1 = make_engine(client, nonce_metadata=False, hooks=crash_at("pre_create"))
    with pytest.raises(Crash):
        engine1.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    client.register_foreign_batch(3, custom_ids=["other-0", "other-1", "other-2"])

    engine2 = make_engine(client, nonce_metadata=False)
    with pytest.raises(WaveHalt):
        engine2.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    entries = Ledger(tmp_path / "w" / "ledger.jsonl").entries()
    assert any(e["state"] == "rolled_back" for e in entries)
    assert client.create_count() == 0


def test_nonce_branch_adopts_even_with_decoys(tmp_path):
    """With metadata round-trip, decoys are irrelevant: identity is the nonce."""
    client = FakeBatchClient(echo, metadata_supported=True)
    client.create_script = ["transport_after_create"]
    engine1 = make_engine(client, nonce_metadata=True)
    engine1.sleep = _raise_crash
    with pytest.raises(Crash):
        engine1.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    client.register_foreign_batch(3)

    engine2 = make_engine(client, nonce_metadata=True)
    out = engine2.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert len(out) == 3
    assert client.create_count() == 1
    harvested_bid = next(
        e["batch_id"]
        for e in Ledger(tmp_path / "w" / "ledger.jsonl").entries()
        if e["state"] == "harvested"
    )
    assert not harvested_bid.startswith("batch_foreign")


@pytest.mark.parametrize(
    "foreign_ids,desc",
    [
        (["req-000", "req-001", "req-002"],
         "untagged foreign batch carrying OUR logical ids"),
        ([f"bbbb1111-req-{i:03d}" for i in range(3)],
         "another run's batch (different run tag), identical logical ids"),
    ],
)
def test_foreign_batch_with_identical_ids_never_stamps(tmp_path, foreign_ids, desc):
    """E1: no-nonce reconciliation sees ONE count-matching candidate that is a
    foreign batch whose custom_ids mirror ours. Run identity in the wire ids
    must make harvest verification reject it: rollback, re-reconcile, HALT —
    never a stamp over foreign results."""
    client = FakeBatchClient(echo, metadata_supported=False)
    engine1 = make_engine(
        client, nonce_metadata=False, run_tag="aaaa0000", hooks=crash_at("pre_create")
    )
    with pytest.raises(Crash):
        engine1.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    client.register_foreign_batch(3, custom_ids=foreign_ids)

    engine2 = make_engine(client, nonce_metadata=False, run_tag="aaaa0000")
    with pytest.raises(WaveHalt):
        engine2.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert not (tmp_path / "w" / "STAMP.json").exists(), desc
    assert not (tmp_path / "w" / "results-c000.jsonl").exists(), desc
    assert client.create_count() == 0, f"{desc}: nothing may be auto-created"
    entries = Ledger(tmp_path / "w" / "ledger.jsonl").entries()
    assert any(e["state"] == "rolled_back" for e in entries), desc


def test_own_tagged_batch_adopted_and_results_keep_logical_ids(tmp_path):
    """E1 positive control: the run's OWN batch (wire ids run-tagged) is
    adopted after an interrupted submit and harvests cleanly; artifacts store
    the logical (untagged) ids."""
    client = FakeBatchClient(echo, metadata_supported=False)
    client.create_script = ["transport_after_create"]
    engine1 = make_engine(client, nonce_metadata=False, run_tag="aaaa0000")
    engine1.sleep = _raise_crash
    with pytest.raises(Crash):
        engine1.run_wave(tmp_path / "w", reqs(3), wave_name="w")

    engine2 = make_engine(client, nonce_metadata=False, run_tag="aaaa0000")
    out = engine2.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert client.create_count() == 1
    assert set(out) == {"req-000", "req-001", "req-002"}
    # the wire carried tagged ids; artifacts carry logical ids
    (bid,) = [b for b in client.batches if not b.startswith("batch_foreign")]
    wire_ids = {r["custom_id"] for r in client.batches[bid]["requests"]}
    assert wire_ids == {f"aaaa0000-req-{i:03d}" for i in range(3)}
    lines = (tmp_path / "w" / "results-c000.jsonl").read_text().splitlines()
    assert {json.loads(x)["custom_id"] for x in lines} == set(out)


def test_flipped_reconciliation_mode_on_resume_hard_errors(tmp_path):
    """E2 (engine backstop): a chunk journaled under one reconciliation mode
    must never be reconciled under the other — a no-nonce journal resumed by a
    nonce-branch engine would search for metadata that was never sent, 'prove'
    absence, and double-submit. Hard error, zero creates."""
    client = FakeBatchClient(echo, metadata_supported=False)
    engine1 = make_engine(client, nonce_metadata=False, hooks=crash_at("pre_create"))
    with pytest.raises(Crash):
        engine1.run_wave(tmp_path / "w", reqs(3), wave_name="w")

    flipped = make_engine(client, nonce_metadata=True)
    with pytest.raises(WaveError, match="nonce_metadata"):
        flipped.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert client.create_count() == 0, "a flipped verdict must create NOTHING"

    # and the reverse flip (nonce journal, no-nonce resume)
    client2 = FakeBatchClient(echo)
    engine3 = make_engine(client2, nonce_metadata=True, hooks=crash_at("pre_create"))
    with pytest.raises(Crash):
        engine3.run_wave(tmp_path / "w2", reqs(3), wave_name="w2")
    flipped2 = make_engine(client2, nonce_metadata=False)
    with pytest.raises(WaveError, match="nonce_metadata"):
        flipped2.run_wave(tmp_path / "w2", reqs(3), wave_name="w2")
    assert client2.create_count() == 0


def test_crash_mid_harvest_no_duplicates(tmp_path):
    client = FakeBatchClient(echo)
    engine1 = make_engine(client, hooks=crash_at("pre_harvest_rename"))
    with pytest.raises(Crash):
        engine1.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert (tmp_path / "w" / "results-c000.jsonl.tmp").exists()
    assert not (tmp_path / "w" / "results-c000.jsonl").exists()

    engine2 = make_engine(client)
    out = engine2.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert client.create_count() == 1  # submitted state survived; no re-create
    lines = (tmp_path / "w" / "results-c000.jsonl").read_text().splitlines()
    ids = [json.loads(line)["custom_id"] for line in lines if line.strip()]
    assert sorted(ids) == ["req-000", "req-001", "req-002"]  # exactly once each
    assert len(out) == 3
    raw = (tmp_path / "w" / "raw_results.jsonl").read_text().splitlines()
    assert len(raw) == 3
