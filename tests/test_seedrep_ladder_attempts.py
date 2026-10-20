"""Fault injection over the attempt / acceptance / closure state machine. A container is simulated as one snapshot of
the two-snapshot fake volume plus the protocol's own steps in order — (i) store COMPLETE, (ii) queue_inventory.json,
(iii) the acceptance put, (iv) the CLOSED marker — and killed at each boundary: after COMPLETE, after the inventory,
after the acceptance, during the queue commit, before / after CLOSED, and "the accepted producer dies after
acceptance, a superseded re-execution publishes its own queue files afterwards, the reconciler still synthesises
CLOSED from the accepted inventory and the rung reads only the accepted projection". Each case must end with exactly
one accepted attempt and a closable queue; every known-bad artifact (a tampered inventory file, a forged marker)
fails CLOSED."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seedrep_testlib as T  # noqa: E402

L = T.load_ladder()
RR = T.load_rung_redraw()

ROWS = T.rows3()


def _run(shared, spec, shard, aid, kv, *, rows=ROWS, crash_after=None, commit_each=False):
    """One container's life: enqueue -> COMPLETE -> inventory -> acceptance -> marker, killed at ``crash_after`` ∈
    {queue_commit, complete, inventory, acceptance, None}. Returns the container (its snapshot is crashed when a
    kill happened)."""
    c = T.Container(L, shared, spec, shard, aid)
    for i, r in enumerate(rows):
        c.enqueue_row(r, commit=commit_each)
        if crash_after == "queue_commit" and i == 0:
            c.vol.commit()                       # the first file lands …
    if crash_after == "queue_commit":
        c.vol.crash()                            # … the rest and everything after are lost
        return c
    c.complete()
    if crash_after == "complete":
        c.vol.crash()
        return c
    try:
        out = c.finish(kv["accepted"], crash_after=crash_after)
    except L._Crash:
        c.vol.crash()
        return c
    c.outcome = out
    return c


def _closed(shared, kv, spec):
    return L.reconcile_queue(shared.read_only_view(), kv["accepted"], spec)


def _exactly_one_accepted(kv, spec, aid):
    acc = [v for _, v in kv["accepted"].items()]
    assert acc == [aid]
    return L.accepted_producer_attempts(kv["accepted"], spec)


@pytest.mark.parametrize("crash_after", ["queue_commit", "complete", "inventory"])
def test_a_crash_before_acceptance_leaves_no_acceptance_and_the_reexecution_takes_it(crash_after):
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    a = _run(shared, spec, "0", "a1", kv, crash_after=crash_after)
    st = _closed(shared, kv, spec)
    assert st["producers"]["0"]["state"] == "PRODUCER_MISSING" and not st["queue_closed"]
    assert len(kv["accepted"]) == 0
    before = dict(shared.files)                                   # a's partial root as it stands
    b = _run(shared, spec, "0", "a2", kv)
    assert b.outcome["state"] == "accepted"
    st = _closed(shared, kv, spec)
    assert st["queue_closed"] and st["closed"]["producers"] == [["0", "a2"]] and st["closed"]["n_items"] == 3
    _exactly_one_accepted(kv, spec, "a2")
    # a's partial root is untouched (never resumed into, never read), its queue files inert
    for k, v in before.items():
        assert shared.files[k] == v
    assert all(k.startswith(a.root) or "/a1.json" in k for k in before)
    proj = L.accepted_queue_entries(shared.read_only_view(), kv["accepted"], spec)
    assert {r["producer"]["attempt_id"] for r in proj.values()} == {"a2"} and len(proj) == 3


def test_a_crash_after_acceptance_before_its_marker_is_synthesised_by_the_reconciler():
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    a = _run(shared, spec, "0", "a1", kv, crash_after="acceptance")
    assert L.accepted_attempt(kv["accepted"], a.ref.key) == "a1"
    view = shared.read_only_view()
    assert not view.exists(a.producer.marker_path)
    st = _closed(shared, kv, spec)
    assert st["producers"]["0"]["state"] == "CLOSED_SYNTHESISED"
    marker = json.loads(shared.files[a.producer.marker_path].decode())
    assert marker["synthesised_by"] == "reconciler" and marker["attempt_id"] == "a1" and marker["n_enqueued"] == 3
    assert marker["item_keys_sha256"] == json.loads(shared.files[a.producer.inventory_path].decode())["item_keys_sha256"]
    assert st["queue_closed"] and st["closed"]["producers"] == [["0", "a1"]]
    # idempotent: the next pass reads the marker (state CLOSED) and writes nothing
    n = len(shared.commit_log)
    st2 = _closed(shared, kv, spec)
    assert st2["producers"]["0"]["state"] == "CLOSED" and st2["closed"] == st["closed"] and len(shared.commit_log) == n
    # the re-executed attempt completes and is superseded; it publishes no marker
    b = _run(shared, spec, "0", "a2", kv)
    assert b.outcome == {"state": "superseded", "attempt_id": "a2", "holder": "a1"}
    assert not view.exists(L.producer_marker_rel(spec.ladder_root, "0", "a2"))
    _exactly_one_accepted(kv, spec, "a1")


def test_accepted_producer_dies_after_acceptance_then_a_superseded_reexecution_publishes_afterwards():
    """rev 4: the superseded attempt's files can never replace the accepted attempt's bytes — no two
    attempts share a path — so the reconciler synthesises CLOSED from the accepted inventory and the rung reads
    only the accepted projection, even where the re-execution enqueued the SAME items with DIFFERENT content."""
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    a = _run(shared, spec, "0", "a1", kv, crash_after="acceptance")
    a_files = {k: v for k, v in shared.files.items() if "/a1.json" in k}
    assert len(a_files) == 3
    # the re-execution AFTER the acceptance: the same items plus one more, other bytes (a fresh enqueued_utc, a
    # different text in the base row it read), a superseded outcome
    other_rows = [T.static_row(r["goal_id"], r["sample_idx"], seed=r["sampling_seed"], text=f"other {r['goal_id']}") for r in ROWS]
    other_rows.append(T.static_row("g7", 0, seed=70))
    b = _run(shared, spec, "0", "a2", kv, rows=other_rows)
    assert b.outcome["state"] == "superseded"
    assert all(shared.files[k] == v for k, v in a_files.items())            # a's bytes untouched
    assert len([k for k in shared.files if "/a2.json" in k]) == 4             # b's files exist …
    st = _closed(shared, kv, spec)
    assert st["producers"]["0"]["state"] == "CLOSED_SYNTHESISED" and st["queue_closed"]
    assert st["closed"]["producers"] == [["0", "a1"]] and st["closed"]["n_items"] == 3
    proj = L.accepted_queue_entries(shared.read_only_view(), kv["accepted"], spec)
    assert len(proj) == 3 and {r["producer"]["attempt_id"] for r in proj.values()} == {"a1"}   # … and inert
    assert all("g7" not in slot for slot in proj)
    for r in proj.values():                                                   # the accepted content, not b's
        assert r["base_row_sha256"] == L.base_row_identity_sha256(text=f"resp {r['identity']['goal_id']} {r['identity']['sample_idx']}",
                                                                  sampling_seed=r["sampling_seed"], finish_reason="length", cap=4096)
    # the rung consumes exactly the accepted projection
    rv = shared.snapshot("rung-1")
    seen = []
    t = RR.consume(rv, accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=spec, attempt_id="rung-1",
                   call_id="fc", sleep=lambda s: None, log=lambda m: None,
                   draw=lambda r: (seen.append(r["producer"]["attempt_id"]) or RR.Draw(
                       record={"text": "t", "finish_reason": "stop", "completion_tokens": 1, "prompt_tokens": 1, "max_tokens": 8192,
                               "sampling_seed": r["sampling_seed"]}, text="t", response=None, request_id=None, completion_id=None)))
    assert t["drawn"] == 3 and set(seen) == {"a1"} and t["closed"] and t["covered"]
    _exactly_one_accepted(kv, spec, "a1")


def test_two_successive_reexecutions_after_partial_writes_get_two_roots_and_neither_is_resumed():
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    a = _run(shared, spec, "0", "a1", kv, crash_after="queue_commit")
    b = _run(shared, spec, "0", "a2", kv, crash_after="complete")
    roots = sorted({k.split("/attempts/")[1].split("/")[0] for k in shared.files if "/attempts/" in k})
    assert roots == ["a2"]                                                 # a1 committed no root file (its one queue file only)
    assert [k for k in shared.files if "/a1.json" in k] and a.root != b.root
    partial = dict(shared.files)
    c = _run(shared, spec, "0", "a3", kv)
    assert c.outcome["state"] == "accepted" and c.root not in (a.root, b.root)
    assert all(shared.files[k] == v for k, v in partial.items())         # the partial roots stand as evidence, untouched
    st = _closed(shared, kv, spec)
    assert st["queue_closed"] and st["closed"]["producers"] == [["0", "a3"]]
    assert L.accepted_attempt_root_or_refuse(kv["accepted"], c.ref.key) == c.root


def test_the_pull_refuses_a_shard_with_no_accepted_attempt_and_a_later_complete_attempt_is_superseded():
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0", "1"))
    with pytest.raises(L.LadderRefusal, match="no accepted attempt"):
        L.accepted_attempt_root_or_refuse(kv["accepted"], spec.producers[0].key)
    a = _run(shared, spec, "0", "a1", kv)
    assert L.accepted_attempt_root_or_refuse(kv["accepted"], spec.producers[0].key) == a.root
    marks = []
    b = T.Container(L, shared, spec, "0", "a2")
    b.enqueue_row(ROWS[0]); b.complete()
    out = b.finish(kv["accepted"], mark_superseded=marks.append)
    assert out["state"] == "superseded" and marks == ["a1"]
    with pytest.raises(L.LadderRefusal, match="no accepted attempt"):
        L.accepted_attempt_root_or_refuse(kv["accepted"], spec.producers[1].key)      # shard 1 never ran
    st = _closed(shared, kv, spec)
    assert st["producers"]["1"]["state"] == "PRODUCER_MISSING" and not st["queue_closed"]


def test_an_accepted_attempt_whose_inventory_files_are_missing_or_altered_is_not_closable():
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    a = _run(shared, spec, "0", "a1", kv, crash_after="acceptance")
    inv = json.loads(shared.files[a.producer.inventory_path].decode())
    target = inv["entries"][1]["path"]
    original = shared.files[target]
    shared.files[target] = original + b" "                                 # a corrupted queue file
    st = _closed(shared, kv, spec)
    assert st["producers"]["0"]["state"] == "ACCEPTED_UNVERIFIED" and not st["closable"] and not st["queue_closed"]
    assert any("sha differs" in r for r in st["producers"]["0"]["reasons"])
    del shared.files[target]                                                # a lost queue file
    st = _closed(shared, kv, spec)
    assert any("missing on the volume" in r for r in st["producers"]["0"]["reasons"]) and not st["queue_closed"]
    shared.files[target] = original                                         # restored: closable again
    st = _closed(shared, kv, spec)
    assert st["producers"]["0"]["state"] == "CLOSED_SYNTHESISED" and st["queue_closed"]


def test_a_forged_marker_that_disagrees_with_the_accepted_inventory_is_refused():
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    a = _run(shared, spec, "0", "a1", kv, crash_after="acceptance")
    forged = {"schema": L.SCHEMA_MARKER, "key": L.key_str(a.ref.key), "queue_root": spec.ladder_root, "shard": "0",
              "attempt_id": "a1", "n_enqueued": 1, "item_keys_sha256": "0" * 64, "synthesised_by": None}
    shared.files[a.producer.marker_path] = (json.dumps(forged) + "\n").encode()
    with pytest.raises(L.LadderRefusal, match="does not describe its inventory"):
        _closed(shared, kv, spec)
    # a marker of another attempt at the accepted attempt's path is refused too
    forged["attempt_id"] = "a9"
    shared.files[a.producer.marker_path] = (json.dumps(forged) + "\n").encode()
    with pytest.raises(L.LadderRefusal, match="not the accepted attempt"):
        _closed(shared, kv, spec)


def test_an_accepted_attempt_without_complete_json_is_not_closable():
    """Acceptance follows COMPLETE by construction; a torn commit that left the acceptance without its COMPLETE.json
    is still never closed on the Dict's word alone."""
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    c = T.Container(L, shared, spec, "0", "a1")
    c.enqueue_row(ROWS[0])
    c.producer.write_inventory()                                          # (ii) without (i)
    assert L.accept_attempt(kv["accepted"], c.ref.key, "a1")
    st = _closed(shared, kv, spec)
    assert st["producers"]["0"]["state"] == "ACCEPTED_UNVERIFIED" and any("COMPLETE.json missing" in r for r in st["producers"]["0"]["reasons"])
    c.complete()
    assert _closed(shared, kv, spec)["queue_closed"]
