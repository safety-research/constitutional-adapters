"""The streaming ladder's queue protocol against the TWO-SNAPSHOT Modal simulation: a fake Volume where each
container has its own snapshot and commits are last-writer-wins, and a fake Dict offering ONLY
``put(skip_if_exists)`` / ``get`` / ``pop`` / ``contains`` / ``items`` with modal 1.5.4's semantics (cited from
``modal/dict.py`` in the modal venv when present).

Cases: two containers both succeed at a local ``O_EXCL`` on one path -> exactly one elected
result and the loser recorded; out-of-order shard completion; a queue entry from a non-accepted attempt refused;
a lost claimant reclaimed only after its call is terminal; conflicting duplicates (first put kept, second recorded
per attempt); the 16,384 queue cannot close before the 8,192 rung; the controls ancestry 512 -> 2,048 -> 8,192 ->
16,384 with the immediate predecessor only; the ceiling never queues; no two attempts share a queue or materialised
path; the consumer loop end to end. Every known-bad input fails closed."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seedrep_testlib as T  # noqa: E402

L = T.load_ladder()
RR = T.load_rung_redraw()
REPO = T.REPO


# --------------------------------------------------------------------------- #
# the double vs the cited Modal semantics                                      #
# --------------------------------------------------------------------------- #


def test_memory_kv_matches_the_cited_modal_dict_semantics():
    kv = L.MemoryKV("d")
    assert kv.put("k", {"a": 1}, skip_if_exists=True) is True          # added
    assert kv.put("k", {"a": 2}, skip_if_exists=True) is False         # existed: NOT overwritten
    assert kv.get("k") == {"a": 1}
    assert kv.put("k", {"a": 3}) is True and kv.get("k") == {"a": 3}   # a plain put overwrites
    assert kv.get("absent") is None and kv.get("absent", 0) == 0
    assert kv.contains("k") and not kv.contains("absent")
    with pytest.raises(KeyError):
        kv.pop("absent")
    assert kv.pop("absent", "dflt") == "dflt"
    assert kv.pop("k") == {"a": 3} and not kv.contains("k")
    # values are serialised on the wire: a stored value is a copy, a returned value is a copy
    v = {"n": [1]}
    kv.put("v", v)
    v["n"].append(2)
    got = kv.get("v")
    assert got == {"n": [1]}
    got["n"].append(9)
    assert kv.get("v") == {"n": [1]}
    # items(): unordered — a consumer may not rely on insertion order
    for i in range(20):
        kv.put(f"i{i}", i)
    keys = [k for k, _ in kv.items()]
    assert set(keys) == {"v", *[f"i{i}" for i in range(20)]} and keys != ["v", *[f"i{i}" for i in range(20)]]
    with pytest.raises(L.LadderRefusal):
        kv.put(("tuple", "key"), 1)                                     # the protocol builds string keys only


def test_the_cited_modal_source_says_what_the_double_implements():
    """The semantics the double reproduces are read off modal's own client source (1.5.4) when the modal venv is
    present; skipped otherwise — the assertion is about the CITATION, not a deployment."""
    hits = glob.glob(str(Path.home() / "venvs" / "modal" / "lib" / "python*" / "site-packages" / "modal" / "dict.py"))
    if not hits:
        pytest.skip("no modal venv on this machine; the citation cannot be re-read")
    src = Path(hits[0]).read_text(encoding="utf-8")
    assert "async def put(self, key: Any, value: Any, *, skip_if_exists: bool = False) -> bool:" in src
    assert ("Returns True if the key-value pair was added and False if it wasn't because the key already existed and"
            in src)
    assert "if_not_exists=skip_if_exists" in src and "return resp.created" in src
    assert "async def pop(self, key: Any, default: Any = _NO_DEFAULT) -> Any:" in src
    assert 'raise KeyError(f"{key} not in dict {self.object_id}")' in src
    assert "Returns `default` if key does not exist." in src
    assert "results are unordered" in src
    # and the module docstring cites exactly that
    assert "skip_if_exists=True) -> bool" in L.__doc__ and "modal 1.5.4" in L.__doc__


# --------------------------------------------------------------------------- #
# the two-snapshot volume                                                      #
# --------------------------------------------------------------------------- #


def test_two_containers_both_succeed_at_local_O_EXCL_on_one_path_and_commits_are_last_writer_wins():
    shared = L.SharedVolume()
    a, b = shared.snapshot("a"), shared.snapshot("b")
    a.write_bytes("x/same.json", b"from a", exclusive=True)
    b.write_bytes("x/same.json", b"from b", exclusive=True)           # NO FileExistsError: separate snapshots
    assert not shared.files                                             # nothing committed yet
    assert not b.exists("y") and a.read_bytes("x/same.json") == b"from a"
    a.commit()
    assert shared.files["x/same.json"] == b"from a"
    b.commit()
    assert shared.files["x/same.json"] == b"from b"                     # last writer wins
    a.reload()
    assert a.read_bytes("x/same.json") == b"from b"
    # a local O_EXCL refuses only against the SAME snapshot
    with pytest.raises(FileExistsError):
        a.write_bytes("x/same.json", b"again", exclusive=True)
    # a crashed container loses its uncommitted writes and writes nothing more
    c = shared.snapshot("c")
    c.write_bytes("z/lost.json", b"never committed")
    c.crash()
    assert "z/lost.json" not in shared.files
    with pytest.raises(L.LadderRefusal):
        c.write_bytes("z/more.json", b"")
    assert [name for name, _ in shared.commit_log] == ["a", "b"]


# --------------------------------------------------------------------------- #
# election: exactly one result, the loser recorded per attempt                 #
# --------------------------------------------------------------------------- #


def _accepted_producer(shared, spec, shard, aid, rows, kv):
    c = T.Container(L, shared, spec, shard, aid)
    recs = [c.enqueue_row(r) for r in rows]
    c.complete()
    assert c.finish(kv["accepted"])["state"] == "accepted"
    return c, recs


def _draw_record(text, *, seed, cap=8192, finish="stop"):
    return {"text": text, "finish_reason": finish, "completion_tokens": cap if finish == "length" else 3,
            "prompt_tokens": 5, "max_tokens": cap, "sampling_seed": seed}


def test_two_claimants_of_one_item_yield_exactly_one_elected_result_and_the_loser_records_itself():
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    _, (rec,) = _accepted_producer(shared, spec, "0", "a1", [T.static_row("g0", 1, seed=7)], kv)
    ra, rb = shared.snapshot("r-a"), shared.snapshot("r-b")
    pa, da = L.write_materialised(ra, spec, queue_record=rec, attempt_id="r-a", call_id="fc-a",
                                  record=_draw_record("text A", seed=7), text="text A", request_id="rq-a", completion_id="cm-a")
    pb, db = L.write_materialised(rb, spec, queue_record=rec, attempt_id="r-b", call_id="fc-b",
                                  record=_draw_record("text B", seed=7), text="text B", request_id="rq-b", completion_id="cm-b")
    assert pa != pb and pa.endswith("/r-a.json") and pb.endswith("/r-b.json")      # attempt-unique paths
    ra.commit(); rb.commit()
    assert L.elect_result(kv["results"], spec, db, pb) is True                       # B's put lands first
    assert L.elect_result(kv["results"], spec, da, pa) is False                      # A loses: first put kept
    winner = kv["results"].get(db["key"])
    assert winner == {"attempt_id": "r-b", "path": pb, "text_sha256": T.sha("text B"), "request_id": "rq-b",
                      "completion_id": "cm-b", "call_id": "fc-b"}         # + the claimant's call (the adopter reads it off the Dict)
    dup = L.record_duplicate(ra, spec, "r-a", da, pa, winner)
    ra.commit()
    assert dup == f"{spec.ladder_root}/duplicates/r-a.jsonl"
    line = json.loads(shared.files[dup].decode().splitlines()[0])
    assert line["text_sha256"] == T.sha("text A") and line["winner"]["attempt_id"] == "r-b"
    assert f"{spec.ladder_root}/duplicates/r-b.jsonl" not in shared.files            # never a shared duplicates file
    assert len(kv["results"]) == 1
    elected = L.load_elected(shared.read_only_view(), kv["results"], spec, "none__s1")
    assert set(elected) == {rec["item_key"]} and elected[rec["item_key"]].doc["record"]["text"] == "text B"
    # a torn / forged file fails closed: the Dict names a sha the file's text does not carry
    tampered = json.loads(shared.files[pb].decode())
    tampered["record"]["text"] = "text B (edited)"
    shared.files[pb] = (json.dumps(tampered) + "\n").encode()
    with pytest.raises(L.LadderRefusal, match="text_sha256"):
        L.load_elected(shared.read_only_view(), kv["results"], spec, "none__s1")


def test_write_materialised_by_one_attempt_twice_is_refused_and_no_two_attempts_share_a_queue_path():
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    row = T.static_row("g0", 1, seed=7)
    a, b = T.Container(L, shared, spec, "0", "a1"), T.Container(L, shared, spec, "0", "a2")
    ra, rb = a.enqueue_row(row, commit=True), b.enqueue_row(row, commit=True)
    paths = sorted(k for k in shared.files if "/queue/" in k)
    assert paths == [L.queue_entry_rel(spec.ladder_root, "none__s1", ra["item_key"], "a1"),
                     L.queue_entry_rel(spec.ladder_root, "none__s1", ra["item_key"], "a2")]
    with pytest.raises(L.LadderRefusal, match="enqueued twice"):
        a.enqueue_row(row)
    r = shared.snapshot("r1")
    L.write_materialised(r, spec, queue_record=ra, attempt_id="r1", call_id="fc", record=_draw_record("t", seed=7),
                         text="t", request_id=None, completion_id=None)
    with pytest.raises(FileExistsError):
        L.write_materialised(r, spec, queue_record=ra, attempt_id="r1", call_id="fc", record=_draw_record("t", seed=7),
                             text="t", request_id=None, completion_id=None)


# --------------------------------------------------------------------------- #
# claims and the reconciler's ONE authorised reclaim                            #
# --------------------------------------------------------------------------- #


def test_a_held_claim_is_skipped_and_reclaimed_only_after_the_claimants_call_is_terminal():
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    _, (rec,) = _accepted_producer(shared, spec, "0", "a1", [T.static_row("g0", 1, seed=7)], kv)
    assert L.claim_item(kv["claims"], spec, rec["arm_s"], rec["item_key"], attempt_id="r-a", call_id="fc-a") is True
    assert L.claim_item(kv["claims"], spec, rec["arm_s"], rec["item_key"], attempt_id="r-b", call_id="fc-b") is False
    view = shared.read_only_view()
    states = {"fc-a": "running"}
    st = L.reconcile_queue(view, kv["accepted"], spec, claims_kv=kv["claims"], results_kv=kv["results"],
                           lease_state=lambda cid: states.get(cid, "unknown"))
    assert st["reclaimed"] == [] and len(st["claims_left"]) == 1                    # running: left alone
    states["fc-a"] = "unknown"
    st = L.reconcile_queue(view, kv["accepted"], spec, claims_kv=kv["claims"], results_kv=kv["results"],
                           lease_state=lambda cid: states.get(cid, "unknown"))
    assert st["reclaimed"] == []                                                     # uncertainty fails closed
    # without a reconciler nothing ever pops it, however long B waits
    for _ in range(3):
        assert L.claim_item(kv["claims"], spec, rec["arm_s"], rec["item_key"], attempt_id="r-b", call_id="fc-b") is False
    states["fc-a"] = "failed"
    st = L.reconcile_queue(view, kv["accepted"], spec, claims_kv=kv["claims"], results_kv=kv["results"],
                           lease_state=lambda cid: states.get(cid, "unknown"))
    assert len(st["reclaimed"]) == 1 and not kv["claims"].contains(st["reclaimed"][0])
    assert L.claim_item(kv["claims"], spec, rec["arm_s"], rec["item_key"], attempt_id="r-b", call_id="fc-b") is True
    # an elected item's claim is never popped, terminal or not (the claim did its job)
    rb = shared.snapshot("r-b")
    p, d = L.write_materialised(rb, spec, queue_record=rec, attempt_id="r-b", call_id="fc-b",
                                record=_draw_record("t", seed=7), text="t", request_id=None, completion_id=None)
    rb.commit()
    assert L.elect_result(kv["results"], spec, d, p)
    states["fc-b"] = "finished"
    st = L.reconcile_queue(view, kv["accepted"], spec, claims_kv=kv["claims"], results_kv=kv["results"],
                           lease_state=lambda cid: states.get(cid, "unknown"))
    assert st["reclaimed"] == [] and kv["claims"].contains(d["key"])
    assert L.TERMINAL_STATES == {"finished", "failed", "cancelled"}


# --------------------------------------------------------------------------- #
# producers, projections, closure                                              #
# --------------------------------------------------------------------------- #


def test_out_of_order_shard_completion_closes_only_when_every_declared_producer_is_accepted():
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0", "1", "2"))
    view = shared.read_only_view()
    c2, r2 = _accepted_producer(shared, spec, "2", "a2", [T.static_row("g9", 0, seed=90)], kv)        # shard 2 first
    st = L.reconcile_queue(view, kv["accepted"], spec)
    assert {s: p["state"] for s, p in st["producers"].items()} == {"0": "PRODUCER_MISSING", "1": "PRODUCER_MISSING", "2": "CLOSED"}
    assert not st["closable"] and st["closed"] is None and not view.exists(L.closed_rel(spec.ladder_root))
    c0, r0 = _accepted_producer(shared, spec, "0", "a0", [T.static_row("g0", 1, seed=7), T.static_row("g1", 1, seed=8)], kv)
    assert not L.reconcile_queue(view, kv["accepted"], spec)["closable"]
    c1, r1 = _accepted_producer(shared, spec, "1", "a1", [], kv)                                       # an empty producer
    st = L.reconcile_queue(view, kv["accepted"], spec)
    assert st["closable"] and st["queue_closed"]
    closed = st["closed"]
    assert closed["producers"] == [["0", "a0"], ["1", "a1"], ["2", "a2"]] and closed["n_items"] == 3
    keys = [r["item_key"] for r in r0 + r1 + r2]
    assert closed["item_keys_sha256"] == L.item_keys_sha256(L.slot("none__s1", k) for k in keys)
    proj = L.accepted_queue_entries(view, kv["accepted"], spec)
    assert set(proj) == {f"none__s1/{k}" for k in keys}
    # idempotent: a second pass re-reads the same CLOSED.json and writes nothing new
    n = len(shared.commit_log)
    again = L.reconcile_queue(view, kv["accepted"], spec)
    assert again["closed"] == closed and len(shared.commit_log) == n
    # a CLOSED.json naming another producer set is refused, never overwritten
    forged = dict(closed); forged["producers"] = [["0", "zz"], ["1", "a1"], ["2", "a2"]]
    shared.files[L.closed_rel(spec.ladder_root)] = (json.dumps(forged) + "\n").encode()
    with pytest.raises(L.LadderRefusal, match="CLOSED.json"):
        L.reconcile_queue(view, kv["accepted"], spec)


def test_a_queue_entry_from_a_non_accepted_attempt_is_inert_and_refused_by_the_assembly():
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    row = T.static_row("g0", 1, seed=7)
    a, b = T.Container(L, shared, spec, "0", "a1"), T.Container(L, shared, spec, "0", "a2")
    ra, rb = a.enqueue_row(row), b.enqueue_row(row)
    a.complete(); b.complete()
    assert a.finish(kv["accepted"])["state"] == "accepted"
    out = b.finish(kv["accepted"])
    assert out["state"] == "superseded" and out["holder"] == "a1"
    view = shared.read_only_view()
    assert view.exists(L.queue_entry_rel(spec.ladder_root, "none__s1", rb["item_key"], "a2"))   # on the volume …
    proj = L.accepted_queue_entries(view, kv["accepted"], spec)
    assert [r["producer"]["attempt_id"] for r in proj.values()] == ["a1"]                        # … and inert
    live = L.live_queue_entries(view, spec)
    assert sorted(r["producer"]["attempt_id"] for r in live[f"none__s1/{ra['item_key']}"]) == ["a1", "a2"]
    # a superseded attempt writes no CLOSED marker
    with pytest.raises(L.LadderRefusal, match="superseded"):
        b.producer.write_closed_marker(kv["accepted"])
    # the assembly refuses an elected row whose queue record names the superseded producer
    r = shared.snapshot("r1")
    p, d = L.write_materialised(r, spec, queue_record=rb, attempt_id="r1", call_id="fc", record=_draw_record("t", seed=7),
                                text="t", request_id=None, completion_id=None)
    r.commit()
    assert L.elect_result(kv["results"], spec, d, p)
    el = L.load_elected(view, kv["results"], spec, "none__s1")[rb["item_key"]]
    ident = L.base_row_identity_sha256(text=row["response"], sampling_seed=7, finish_reason="length", cap=4096)
    with pytest.raises(L.ElectedMismatch, match="not an accepted producer"):
        L.verify_elected(el, base_row_sha256=ident, sampling_seed=7, cap=8192, preset=T.PRESET, accepted_producers=["a1"])
    # the same row from the accepted producer verifies (another rung attempt: one attempt writes one file per item)
    r2 = shared.snapshot("r2")
    p2, d2 = L.write_materialised(r2, spec, queue_record=ra, attempt_id="r2", call_id="fc", record=_draw_record("t", seed=7),
                                  text="t", request_id=None, completion_id=None)
    L.verify_elected(L.Elected(ra["item_key"], "none__s1", {"attempt_id": "r2"}, d2, p2), base_row_sha256=ident,
                     sampling_seed=7, cap=8192, preset=T.PRESET, accepted_producers=["a1"])
    # and every other mismatch is named: seed, cap, preset, base row
    with pytest.raises(L.ElectedMismatch, match="seed"):
        L.verify_elected(el, base_row_sha256=ident, sampling_seed=8, cap=8192, preset=T.PRESET, accepted_producers=["a2"])
    with pytest.raises(L.ElectedMismatch, match="cap"):
        L.verify_elected(el, base_row_sha256=ident, sampling_seed=7, cap=16384, preset=T.PRESET, accepted_producers=["a2"])
    with pytest.raises(L.ElectedMismatch, match="preset"):
        L.verify_elected(el, base_row_sha256=ident, sampling_seed=7, cap=8192, preset={**T.PRESET, "top_k": 40},
                         accepted_producers=["a2"])
    with pytest.raises(L.ElectedMismatch, match="base row"):
        L.verify_elected(el, base_row_sha256="0" * 64, sampling_seed=7, cap=8192, preset=T.PRESET, accepted_producers=["a2"])


def test_a_late_item_after_the_inventory_is_a_producer_bug_and_is_refused():
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    c = T.Container(L, shared, spec, "0", "a1")
    c.enqueue_row(T.static_row("g0", 1, seed=7))
    c.complete()
    c.producer.write_inventory()
    with pytest.raises(L.ProducerSealed, match="late item"):
        c.enqueue_row(T.static_row("g1", 1, seed=8))
    # a record of the wrong queue / producer / cap refuses before it is written
    other = T.static_spec(L, shards=("0",), wave=T.WAVE2)
    with pytest.raises(L.LadderRefusal):
        L.static_queue_record(other, arm="none", s=1, family="pasttense", row=T.static_row("g2", 0, seed=1), base_store="x",
                              producer=c.ref, producer_attempt_id="a1", vmodel="v", sampling=T.PRESET)
    with pytest.raises(L.LadderRefusal, match="not at cap"):
        L.static_queue_record(spec, arm="none", s=1, family="pasttense", row=T.static_row("g2", 0, seed=1, finish="stop"),
                              base_store="x", producer=c.ref, producer_attempt_id="a1", vmodel="v", sampling=T.PRESET)
    with pytest.raises(L.LadderRefusal, match="predecessor cap"):
        L.static_queue_record(spec, arm="none", s=1, family="pasttense", row=T.static_row("g2", 0, seed=1, cap=2048),
                              base_store="x", producer=c.ref, producer_attempt_id="a1", vmodel="v", sampling=T.PRESET)


# --------------------------------------------------------------------------- #
# closure propagates rung by rung; the ladders; the ceiling                     #
# --------------------------------------------------------------------------- #


def test_the_16384_queue_cannot_close_before_the_8192_rung():
    shared, kv = L.SharedVolume(), T.kvs(L)
    q8 = T.static_spec(L, cap_next=8192, shards=("0",))
    q16 = q8.next_spec(["0"])
    assert q16.cap_next == 16384 and [p.key.stage for p in q16.producers] == ["8192"]
    view = shared.read_only_view()
    _accepted_producer(shared, q8, "0", "a1", [T.static_row("g0", 1, seed=7)], kv)
    assert L.reconcile_queue(view, kv["accepted"], q8)["queue_closed"]
    st = L.reconcile_queue(view, kv["accepted"], q16)
    assert st["producers"]["0"]["state"] == "PRODUCER_MISSING" and not st["queue_closed"]   # 8192 closed, rung not accepted
    rung = T.Container(L, shared, q16, "0", "rung-1")
    rung.complete()
    assert rung.finish(kv["accepted"])["state"] == "accepted"
    st = L.reconcile_queue(view, kv["accepted"], q16)
    assert st["queue_closed"] and st["closed"]["producers"] == [["0", "rung-1"]] and st["closed"]["n_items"] == 0
    # known-bad: a 16384 queue whose producers are base-stage keys is refused at construction
    with pytest.raises(L.LadderRefusal, match="immediate predecessor"):
        L.QueueSpec(T.W7, T.WAVE, "pasttense", "s1", "static", 16384,
                    (L.ProducerRef("0", L.SixAxisKey(T.W7, T.WAVE, "pasttense", "s1", "base", "0")),))


def test_controls_ancestry_is_the_three_transition_ladder_with_the_immediate_predecessor_only():
    assert L.LADDERS["controls"] == (512, 2048, 8192, 16384) and L.LADDERS["static"] == (4096, 8192, 16384)
    assert L.LADDERS["misalign"] == (3000, 16384) and L.LADDERS["gsm8k"] == (8192, 16384)
    assert [L.predecessor_cap("controls", c) for c in (2048, 8192, 16384)] == [512, 2048, 8192]
    assert [L.producer_stage("controls", c) for c in (2048, 8192, 16384)] == ["base", "2048", "8192"]
    assert L.next_cap("controls", 512) == 2048 and L.next_cap("controls", 16384) is None
    with pytest.raises(L.LadderRefusal, match="base"):
        L.predecessor_cap("controls", 512)
    with pytest.raises(L.LadderRefusal, match="not a rung"):
        L.predecessor_cap("controls", 4096)
    chain = [L.QueueSpec.build(T.W7, T.WAVE, "xstest", "s1", "controls", 2048, ["0"])]
    while chain[-1].next_spec(["r"]) is not None:
        chain.append(chain[-1].next_spec(["r"]))
    assert [(q.base_cap, q.cap_next) for q in chain] == [(512, 2048), (2048, 8192), (8192, 16384)]
    assert [p.key.stage for q in chain for p in q.producers] == ["base", "2048", "8192"]
    # a record that skips a rung (a 512 row into the 8192 queue) refuses
    q8 = chain[1]
    ev = {"item_id": "xst0001-bbbbbbbb", "at_cap": True, "finish_reason": "length", "sampling_seed": 5, "text_sha256": "a" * 64}
    rec = L.controls_queue_record(q8, arm="none", s=1, control="xstest", event=ev, prompt="p", system=None, base_store="s",
                                  producer=q8.producers[0], producer_attempt_id="a", vmodel="v", sampling=T.PRESET)
    assert rec["base_cap"] == 2048 and rec["item_key"] == "controls__xstest__xst0001-bbbbbbbb__0"
    with pytest.raises(L.LadderRefusal, match="predecessor cap"):
        L.make_queue_record(q8, arm="none", s=1, kind="controls", identity={}, item_key_="controls__x", sampling_seed=1,
                            base_cap=512, base_row_sha256="0" * 64, base_store="s", producer=q8.producers[0],
                            producer_attempt_id="a")
    with pytest.raises(L.LadderRefusal, match="not at cap"):
        L.controls_queue_record(q8, arm="none", s=1, control="xstest", event={**ev, "at_cap": False}, prompt="p", system=None,
                                base_store="s", producer=q8.producers[0], producer_attempt_id="a", vmodel="v", sampling=T.PRESET)


def test_the_ladders_equal_the_drivers_own_constants():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ladder_decide_pin", REPO / "scripts/sdfbase/full22/sampled/ladder_decide.py")
    ld = importlib.util.module_from_spec(spec); sys.modules["ladder_decide_pin"] = ld; spec.loader.exec_module(ld)
    assert L.LADDERS["static"] == tuple(ld.LADDERS["static"]) and L.LADDERS["controls"] == tuple(ld.LADDERS["controls"])
    assert L.CEILING == ld.CEILING == 16384
    from sisteer.capeval.misalign.runner import DEFAULT_MAX_TOKENS
    from sisteer.capeval.tinycap import contracts as C
    from sisteer.eval.sampled_controls import CONTROLS_CAP_LADDER
    assert L.LADDERS["controls"] == tuple(CONTROLS_CAP_LADDER)
    assert L.LADDERS["misalign"] == (DEFAULT_MAX_TOKENS["agentic"], L.CEILING)
    assert L.LADDERS["gsm8k"] == (C.G8K_CAP, C.G8K_RUNG_CAP)


def test_the_ceiling_never_queues():
    shared, kv = L.SharedVolume(), T.kvs(L)
    q16 = L.QueueSpec.build(T.W7, T.WAVE, "pasttense", "s1", "static", 16384, ["0"])
    assert q16.next_spec(["x"]) is None and L.next_cap("static", 16384) is None
    with pytest.raises(L.LadderRefusal):
        L.QueueSpec.build(T.W7, T.WAVE, "pasttense", "s1", "static", 32768, ["0"])
    # a rung consumer at the ceiling records an at-cap draw and enqueues nothing
    feeder = T.Container(L, shared, q16, "0", "rung8k-1")
    row = T.static_row("g0", 1, seed=7, cap=8192)
    rec = feeder.enqueue_row(row)
    feeder.complete()
    assert feeder.finish(kv["accepted"])["state"] == "accepted"
    rv = shared.snapshot("rung16k-1")
    tally = RR.consume(rv, accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=q16,
                       attempt_id="rung16k-1", call_id="fc-16", sleep=lambda s: None, log=lambda m: None, max_rounds=1,
                       draw=lambda r: RR.Draw(record=_draw_record("still long", seed=7, cap=16384, finish="length"),
                                              text="still long", response=None, request_id="rq", completion_id="cm"))
    assert tally["drawn"] == 1 and tally["won"] == 1 and tally["at_ceiling"] == 1 and tally["enqueued_next"] == 0
    el = L.load_elected(shared.read_only_view(), kv["results"], q16, "none__s1")[rec["item_key"]]
    assert el.doc["record"]["at_ceiling"] is True and el.doc["record"]["finish_reason"] == "length"
    assert not [k for k in shared.files if "/ladder/pasttense/s1/32768/" in k]


# --------------------------------------------------------------------------- #
# the consumer loop end to end                                                  #
# --------------------------------------------------------------------------- #


def test_the_consumer_claims_draws_elects_and_feeds_the_next_queue():
    shared, kv = L.SharedVolume(), T.kvs(L)
    q8 = T.static_spec(L, shards=("0",))
    q16 = q8.next_spec(["0"])
    rows = [T.static_row("g0", 1, seed=7), T.static_row("g1", 1, seed=8), T.static_row("g3", 0, seed=9)]
    _, recs = _accepted_producer(shared, q8, "0", "a1", rows, kv)
    view = shared.read_only_view()
    assert L.reconcile_queue(view, kv["accepted"], q8)["queue_closed"]
    rv = shared.snapshot("rung-1")
    nxt = L.Producer(rv, q16, q16.producers[0], "rung-1", L.producer_attempt_root(q16.producers[0], "rung-1"))
    drawn = []

    def draw(rec):
        drawn.append(rec["item_key"])
        at_cap = rec["identity"]["goal_id"] == "g0"
        text = f"redraw {rec['identity']['goal_id']}"
        return RR.Draw(record=_draw_record(text, seed=rec["sampling_seed"], finish="length" if at_cap else "stop"), text=text,
                       response={"id": "cmpl-x", "choices": []}, request_id=f"rq-{rec['item_key']}", completion_id="cmpl-x")

    tally = RR.consume(rv, accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=q8,
                       attempt_id="rung-1", call_id="fc-1", draw=draw, next_producer=nxt, sleep=lambda s: None, log=lambda m: None)
    assert tally["drawn"] == 3 and tally["won"] == 3 and tally["lost"] == 0 and tally["enqueued_next"] == 1
    assert tally["closed"] and tally["covered"] and sorted(drawn) == sorted(r["item_key"] for r in recs)
    # every elected result binds request_id / completion_id and verifies against its accepted base row
    elected = L.load_elected(view, kv["results"], q8, "none__s1")
    assert set(elected) == {r["item_key"] for r in recs}
    for row, rec in zip(rows, recs):
        el = elected[rec["item_key"]]
        assert el.value["request_id"] == f"rq-{rec['item_key']}" and el.value["completion_id"] == "cmpl-x"
        L.verify_elected(el, base_row_sha256=L.base_row_identity_sha256(text=row["response"], sampling_seed=row["sampling_seed"],
                                                                        finish_reason="length", cap=4096),
                         sampling_seed=row["sampling_seed"], cap=8192, preset=T.PRESET, accepted_producers=["a1"])
    # the next queue holds exactly the at-cap re-draw, produced by THIS rung attempt, bound to the materialised record
    live = L.live_queue_entries(view, q16)
    assert list(live) == [f"none__s1/{recs[0]['item_key']}"]
    nq = live[f"none__s1/{recs[0]['item_key']}"][0]
    assert nq["producer"] == {"shard": "0", "attempt_id": "rung-1"} and nq["base_cap"] == 8192 and nq["cap_next"] == 16384
    assert nq["base_store"] == elected[recs[0]["item_key"]].path
    assert nq["base_row_sha256"] == L.base_row_identity_sha256(text="redraw g0", sampling_seed=7, finish_reason="length", cap=8192)
    assert nq["redraw"] == recs[0]["redraw"] and nq["sampling_seed"] == 7
    # a second consumer sees every item held / elected and draws nothing
    rv2 = shared.snapshot("rung-2")
    t2 = RR.consume(rv2, accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=q8,
                    attempt_id="rung-2", call_id="fc-2", draw=lambda r: pytest.fail("drew a held item"), sleep=lambda s: None,
                    log=lambda m: None)
    assert t2["drawn"] == 0 and t2["already_elected"] == 3 and t2["closed"]
    # the rung finishes its producer side and the 16384 queue closes with its one item
    rv.write_bytes(L.complete_rel(nxt.attempt_root), b"{}")
    assert L.finish_producer(nxt, kv["accepted"])["state"] == "accepted"
    st = L.reconcile_queue(view, kv["accepted"], q16)
    assert st["queue_closed"] and st["closed"]["n_items"] == 1 and st["closed"]["producers"] == [["0", "rung-1"]]
    # GSM8K never pre-draws
    with pytest.raises(L.LadderRefusal, match="Inspect driver"):
        RR.consume(rv, accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"],
                   spec=L.QueueSpec.build(T.W7, T.WAVE, "gsm8k", "s1", "gsm8k", 16384, ["0"]), attempt_id="x", call_id="y",
                   draw=lambda r: None)


def test_speculative_draws_are_tentative_until_their_producer_is_accepted_and_a_losers_draw_is_never_elected():
    """P1-5: a speculative draw (a live, not-yet-accepted producer's entry) is written to speculative/ and NOT elected;
    when a re-execution wins the acceptance instead, the tentative draw stays inert and the consumer — re-entering with
    the claim it already holds — draws the ACCEPTED record, which is what the assembly verifies."""
    shared, kv = T.load_ladder().SharedVolume(), T.kvs(L)
    q8 = T.static_spec(L, shards=("0",))
    row = T.static_row("g0", 1, seed=7)
    a, b = T.Container(L, shared, q8, "0", "a1"), T.Container(L, shared, q8, "0", "a2")
    ra = a.enqueue_row(row, commit=True)                           # a's entry is live, a not yet accepted
    rv = shared.snapshot("rung-1")
    draws = []

    def draw(r):
        draws.append(r["producer"]["attempt_id"])
        return RR.Draw(record=_draw_record(f"spec text {r['producer']['attempt_id']}", seed=7), text=f"spec text {r['producer']['attempt_id']}",
                       response=None, request_id=None, completion_id=None)

    t = RR.consume(rv, accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=q8, attempt_id="rung-1",
                   call_id="fc", draw=draw, speculative=True, sleep=lambda s: None, log=lambda m: None, max_rounds=1)
    assert t["drawn"] == 1 and t["speculative_drawn"] == 1 and t["speculative_pending"] == 1 and not t["closed"]
    assert len(kv["results"]) == 0                                     # tentative: nothing elected
    spec_path = L.speculative_rel(q8.ladder_root, "none__s1", ra["item_key"], "rung-1")
    assert shared.read_only_view().exists(spec_path) and json.loads(shared.files[spec_path])["speculative"] is True
    # the accepted projection is still empty; the non-speculative consumer sees nothing
    assert L.accepted_queue_entries(shared.read_only_view(), kv["accepted"], q8) == {}
    # b (a re-execution) wins the acceptance instead of a: a's speculative draw is a loser's
    other = T.static_row("g0", 1, seed=7, text="other base text")
    rb = b.enqueue_row(other, commit=True)
    b.complete()
    assert b.finish(kv["accepted"])["state"] == "accepted"
    assert L.reconcile_queue(shared.read_only_view(), kv["accepted"], q8)["queue_closed"]
    # the same rung attempt re-enters (a new call, e.g. after a deadline): its own claim is honoured, the ACCEPTED record is drawn
    rv2 = shared.snapshot("rung-1b")
    t2 = RR.consume(rv2, accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=q8, attempt_id="rung-1",
                    call_id="fc", draw=draw, speculative=True, sleep=lambda s: None, log=lambda m: None)
    assert t2["drawn"] == 1 and t2["won"] == 1 and t2["skipped_held"] == 0 and t2["closed"] and t2["covered"] and draws == ["a1", "a2"]
    el = L.load_elected(shared.read_only_view(), kv["results"], q8, "none__s1")[rb["item_key"]]
    assert el.path == L.materialised_rel(q8.ladder_root, "none__s1", rb["item_key"], "rung-1") and el.doc["record"]["text"] == "spec text a2"
    assert el.doc["queue_record"]["producer"]["attempt_id"] == "a2"
    L.verify_elected(el, base_row_sha256=L.base_row_identity_sha256(text="other base text", sampling_seed=7, finish_reason="length", cap=4096),
                     sampling_seed=7, cap=8192, preset=T.PRESET, accepted_producers=["a2"])
    # KNOWN-BAD: a's row would not verify against the elected result — and a's tentative file was never named by the Dict
    with pytest.raises(L.ElectedMismatch, match="base row"):
        L.verify_elected(el, base_row_sha256=L.base_row_identity_sha256(text=row["response"], sampling_seed=7, finish_reason="length", cap=4096),
                         sampling_seed=7, cap=8192, preset=T.PRESET, accepted_producers=["a2"])
    assert rb["item_key"] == ra["item_key"] and shared.read_only_view().exists(spec_path)
