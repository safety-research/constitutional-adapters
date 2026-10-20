"""Each fixed defect in the ladder's code, re-validated with a KNOWN-BAD case beside the good one.
Everything synthetic: placeholder W7 / wave uuid, invented goal ids, ``TESTHOST``-free."""
from __future__ import annotations

import copy
import json
import sys
import itertools
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seedrep_testlib as T  # noqa: E402

from sisteer.eval.attacks import bon_served as bs  # noqa: E402
from sisteer.eval.attacks import served_target as st  # noqa: E402

L = T.load_ladder()
RR = T.load_rung_redraw()

SI = "synthetic constitution instruction: many principles (test bytes)"
PROMPTED = "prompted__si_constitution"
VM_PROMPTED = f"petri__{PROMPTED}"
#: the SECOND prompted arm of the battery (`prompted__si_constitution` + `prompted__safety_refusal`), with ITS OWN SI —
#: no longer than the constitution's in bytes, so the constitution's SI on this route is a hash MISMATCH (not a short prefix)
SI2 = "synthetic safety-refusal instruction: refuse harm (test bytes)"
SAFETY = "prompted__safety_refusal"
VM_SAFETY = f"petri__{SAFETY}"


def _static_rec(spec, row, *, arm, vmodel, system, s=1, attempt="a1"):
    return L.static_queue_record(spec, arm=arm, s=s, family="pasttense", row=row, base_store="exp/base/shard00/responses.jsonl",
                                 producer=spec.producers[0], producer_attempt_id=attempt, vmodel=vmodel, sampling=T.PRESET, system=system)


# --------------------------------------------------------------------------- #
# P1-2 — static rung requests carry the prompted arm's authoritative SI         #
# --------------------------------------------------------------------------- #


def test_p1_2_static_record_messages_are_the_targets_own_and_pass_the_real_prompted_prefix_validator():
    spec = T.static_spec(L, shards=("0",))
    row = T.static_row("g0", 1, seed=7)
    target = st.ServedChatTarget(lambda p: pytest.fail("never called"), virtual_model=VM_PROMPTED, cell_id=PROMPTED, system=SI,
                                 route_info={"virtual_model": VM_PROMPTED})
    payload = target._payload([{"role": "user", "content": row["attack_prompt"]}], max_new_tokens=8192, do_sample=True,
                              temperature=0.7, sampling_seed=7, decode=T.PRESET)
    rec = _static_rec(spec, row, arm=PROMPTED, vmodel=VM_PROMPTED, system=target.system)
    body = RR.request_body(rec)
    # the record's request IS the authoritative builder's: messages (SI as the sole system turn), the six vendor keys, seed, cap
    assert body["messages"] == payload["messages"] == [{"role": "system", "content": SI}, {"role": "user", "content": "g0/1"}]
    assert {k: body[k] for k in bs.SAMPLER_KEYS} == {k: payload[k] for k in bs.SAMPLER_KEYS}
    assert (body["model"], body["seed"], body["max_tokens"]) == (payload["model"], payload["seed"], payload["max_tokens"]) == (VM_PROMPTED, 7, 8192)
    verifiers = (T.prompted_prefix_verifier(VM_PROMPTED, SI), T.prompted_prefix_verifier(VM_SAFETY, SI2))   # one route per prompted arm
    proxy = T.StubProxy(verify=lambda body: [v(body) for v in verifiers])
    try:
        draw = RR.http_draw_fn(proxy.url)
        d = draw(rec)                                                     # the real H8 check admits it
        assert d.record["finish_reason"] == "stop" and d.text == "reply:g0/1" and d.completion_id == "cmpl-1"
        assert proxy.bodies[-1]["body"]["messages"][0] == {"role": "system", "content": SI}
        # KNOWN-BAD: the same prompted arm queued WITHOUT its system (the pre-fix record) is refused si_prefix_missing
        bad = _static_rec(spec, row, arm=PROMPTED, vmodel=VM_PROMPTED, system=None)
        assert bad["redraw"]["messages"] == [{"role": "user", "content": "g0/1"}]
        with pytest.raises(L.LadderRefusal, match="si_prefix_missing"):
            draw(bad)
        # a byte changed INSIDE the prefix hashes differently: si_prefix_mismatch (text appended after it is admitted, as the proxy does)
        with pytest.raises(L.LadderRefusal, match="si_prefix_mismatch"):
            draw(_static_rec(spec, row, arm=PROMPTED, vmodel=VM_PROMPTED, system="X" + SI[1:]))
        assert draw(_static_rec(spec, row, arm=PROMPTED, vmodel=VM_PROMPTED, system=SI + " appended")).text == "reply:g0/1"
        # `none` carries no system and passes as before
        none = _static_rec(spec, row, arm="none", vmodel="petri__base", system=None)
        assert none["redraw"]["messages"] == [{"role": "user", "content": "g0/1"}] and draw(none).text == "reply:g0/1"
        # the SECOND prompted arm: its record carries ITS OWN target.system and is admitted under its route
        t2 = st.ServedChatTarget(lambda p: pytest.fail("never called"), virtual_model=VM_SAFETY, cell_id=SAFETY, system=SI2,
                                 route_info={"virtual_model": VM_SAFETY})
        safe = _static_rec(spec, row, arm=SAFETY, vmodel=VM_SAFETY, system=t2.system)
        assert safe["redraw"]["messages"] == [{"role": "system", "content": SI2}, {"role": "user", "content": "g0/1"}]
        assert draw(safe).text == "reply:g0/1" and proxy.bodies[-1]["body"]["model"] == VM_SAFETY
        # KNOWN-BAD: the OTHER prompted arm's SI on this route (a builder that took the wrong arm's system) is refused — a hash
        # mismatch when it is at least as long as the expected prefix, a short prefix (missing) when it is shorter; and no system at all
        assert len(SI.encode()) >= len(SI2.encode())
        with pytest.raises(L.LadderRefusal, match="si_prefix_mismatch"):
            draw(_static_rec(spec, row, arm=SAFETY, vmodel=VM_SAFETY, system=target.system))
        with pytest.raises(L.LadderRefusal, match="si_prefix_missing"):
            draw(_static_rec(spec, row, arm=PROMPTED, vmodel=VM_PROMPTED, system=t2.system))
        with pytest.raises(L.LadderRefusal, match="si_prefix_missing"):
            draw(_static_rec(spec, row, arm=SAFETY, vmodel=VM_SAFETY, system=None))
    finally:
        proxy.close()
    # the controls builder composes through the same function
    ev = {"item_id": "x1", "at_cap": True, "finish_reason": "length", "sampling_seed": 5, "text_sha256": "0" * 64}
    cspec = L.QueueSpec.build(T.W7, T.WAVE, "controls", "s1", "controls", 2048, ["0"])
    crec = L.controls_queue_record(cspec, arm=PROMPTED, s=1, control="xstest", event=ev, prompt="hi", system=SI, base_store="b",
                                   producer=cspec.producers[0], producer_attempt_id="a1", vmodel=VM_PROMPTED, sampling=T.PRESET)
    assert crec["redraw"]["messages"] == L.redraw_messages("hi", SI) == [{"role": "system", "content": SI}, {"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------- #
# P1-6 — a producer marker never bypasses inventory integrity                   #
# --------------------------------------------------------------------------- #

ROWS = T.rows3()


def _finished_container(shared, spec, kv, aid="a1", rows=ROWS):
    """A producer that ran to its OWN marker: enqueue -> COMPLETE -> inventory -> acceptance -> marker."""
    c = T.Container(L, shared, spec, "0", aid)
    for r in rows:
        c.enqueue_row(r)
    c.complete()
    assert c.finish(kv["accepted"])["state"] == "accepted"
    assert shared.read_only_view().exists(c.producer.marker_path)
    return c


def test_p1_6_with_the_producers_own_marker_present_a_tampered_queue_file_or_missing_complete_is_not_closable():
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    c = _finished_container(shared, spec, kv)
    inv = json.loads(shared.files[c.producer.inventory_path].decode())
    target = inv["entries"][1]["path"]
    original = shared.files[target]
    # KNOWN-BAD 1: the marker stands, one inventoried queue file is modified (sha mismatch) -> ACCEPTED_UNVERIFIED, open
    shared.files[target] = original.replace(b'"sampling_seed":8', b'"sampling_seed":9')
    assert shared.files[target] != original
    st = L.reconcile_queue(shared.read_only_view(), kv["accepted"], spec)
    assert st["producers"]["0"]["state"] == "ACCEPTED_UNVERIFIED" and not st["closable"] and not st["queue_closed"]
    assert any("sha differs" in r for r in st["producers"]["0"]["reasons"]) and "marker" not in st["producers"]["0"]
    assert not shared.read_only_view().exists(L.closed_rel(spec.ladder_root))
    # KNOWN-BAD 2: the marker stands, COMPLETE.json is gone -> not closable either
    shared.files[target] = original
    complete = shared.files.pop(L.complete_rel(c.root))
    st = L.reconcile_queue(shared.read_only_view(), kv["accepted"], spec)
    assert st["producers"]["0"]["state"] == "ACCEPTED_UNVERIFIED" and any("COMPLETE.json missing" in r for r in st["producers"]["0"]["reasons"])
    # KNOWN-BAD 3: a committed queue file of the accepted attempt OUTSIDE its inventory (written after sealing)
    shared.files[L.complete_rel(c.root)] = complete
    stray_rec = L.static_queue_record(spec, arm="none", s=1, family="pasttense", row=T.static_row("g9", 0, seed=90), base_store="b",
                                      producer=c.ref, producer_attempt_id="a1", vmodel="petri__base", sampling=T.PRESET)
    stray = L.queue_entry_rel(spec.ladder_root, "none__s1", stray_rec["item_key"], "a1")
    shared.files[stray] = (L.canonical_json(stray_rec) + "\n").encode()
    st = L.reconcile_queue(shared.read_only_view(), kv["accepted"], spec)
    assert st["producers"]["0"]["state"] == "ACCEPTED_UNVERIFIED" and any("not in its inventory" in r for r in st["producers"]["0"]["reasons"])
    # GOOD: evidence restored -> CLOSED on the producer's own marker, CLOSED.json names exactly the inventory's slots
    del shared.files[stray]
    st = L.reconcile_queue(shared.read_only_view(), kv["accepted"], spec)
    assert st["producers"]["0"]["state"] == "CLOSED" and st["producers"]["0"]["marker"]["synthesised_by"] is None
    assert st["queue_closed"] and st["closed"]["n_items"] == 3 and st["closed"]["item_keys_sha256"] == inv["item_keys_sha256"]


def test_p1_6_and_p2_9_closure_refuses_when_the_projection_is_not_the_inventories_union(monkeypatch):
    """The closure-time guard itself, against a KNOWN-BAD input: with the per-producer file check bypassed, a committed
    file of the accepted attempt outside its inventory would enter the projection — the guard refuses to close."""
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    c = _finished_container(shared, spec, kv)
    stray_rec = L.static_queue_record(spec, arm="none", s=1, family="pasttense", row=T.static_row("g9", 0, seed=90), base_store="b",
                                      producer=c.ref, producer_attempt_id="a1", vmodel="petri__base", sampling=T.PRESET)
    shared.files[L.queue_entry_rel(spec.ladder_root, "none__s1", stray_rec["item_key"], "a1")] = (L.canonical_json(stray_rec) + "\n").encode()
    monkeypatch.setattr(L, "_verify_inventory_files", lambda *a, **k: [])
    with pytest.raises(L.LadderRefusal, match="not the accepted inventories' union"):
        L.reconcile_queue(shared.read_only_view(), kv["accepted"], spec)
    assert not shared.read_only_view().exists(L.closed_rel(spec.ladder_root))


# --------------------------------------------------------------------------- #
# P2-9 — inventory sealing does not race the tailer                              #
# --------------------------------------------------------------------------- #


class _SlowCommitSnapshot(L.SnapshotVolume):
    """A container view whose commit is slow (the Volume.commit of a real mount) — the window the race lived in."""

    def commit(self):
        time.sleep(0.05)
        super().commit()


def test_p2_9_a_tailer_racing_the_seal_lands_in_the_inventory_or_is_refused_never_between():
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    vol = _SlowCommitSnapshot(shared, "a1")
    ref = spec.producers[0]
    prod = L.Producer(vol, spec, ref, "a1", L.attempt_root_rel(spec.W7, spec.leg, spec.view, "0", "a1"))
    outcomes = {"enqueued": [], "sealed_refusals": 0, "other": []}
    go, mid = threading.Event(), threading.Event()
    PRE = 40                                           # enqueues that land BEFORE the seal is even asked for

    def tailer():
        """Enqueue without a fixed count or a timer: after PRE items it announces it is mid-stream (the seal waits for
        that, never for a clock), keeps enqueuing WHILE the seal runs (its enqueues queue on the producer's lock behind
        the seal's slow commits), and stops only once the sealed producer has refused it three times — so the
        interleaving is explicit and the outcome cannot depend on scheduling (the earlier 20 ms sleep + 400 fixed
        enqueues let a fast container finish every enqueue before the seal and see no refusal). The bound is
        a liveness backstop that FAILS the test (an `other` outcome), never a pass."""
        go.wait()
        for i in itertools.count():
            if i >= 200_000:
                outcomes["other"].append("the seal never refused an enqueue: the producer did not seal")
                return
            row = T.static_row(f"g{i}", 0, seed=1000 + i)
            rec = L.static_queue_record(spec, arm="none", s=1, family="pasttense", row=row, base_store="b", producer=ref,
                                        producer_attempt_id="a1", vmodel="petri__base", sampling=T.PRESET)
            try:
                prod.enqueue(rec)
                outcomes["enqueued"].append(rec["item_key"])
                if len(outcomes["enqueued"]) == PRE:
                    mid.set()
            except L.ProducerSealed:
                outcomes["sealed_refusals"] += 1
                if outcomes["sealed_refusals"] >= 3:
                    return
            except Exception as e:  # noqa: BLE001
                outcomes["other"].append(repr(e))
                return

    t = threading.Thread(target=tailer)
    t.start()
    go.set()
    assert mid.wait(timeout=30), "the tailer never reached its PRE enqueues"   # the tailer IS mid-stream when the finish seals
    vol.write_bytes(L.complete_rel(prod.attempt_root), b"{}")
    inv = prod.write_inventory()                       # seals under the lock; the tailer's next enqueues are refused
    t.join(timeout=60)
    assert not t.is_alive() and outcomes["other"] == []
    assert outcomes["sealed_refusals"] >= 1 and len(outcomes["enqueued"]) >= PRE   # the race happened: late enqueues were refused
    # the invariant: the inventory IS the set of this attempt's committed queue files — no file outside it, none missing
    committed = sorted(k for k in shared.files if k.startswith(f"{spec.ladder_root}/queue/") and k.endswith("/a1.json"))
    assert sorted(e["path"] for e in inv["entries"]) == committed and inv["n"] == len(committed) == len(outcomes["enqueued"])
    assert L.item_keys_sha256(L.slot("none__s1", k) for k in outcomes["enqueued"]) == inv["item_keys_sha256"]
    # a refused late item left no file anywhere
    assert not any(k.endswith("/a1.json") and k not in committed for k in vol.local)
    # and the reconciler, given the acceptance, closes on exactly that set
    assert L.accept_attempt(kv["accepted"], ref.key, "a1")
    st = L.reconcile_queue(shared.read_only_view(), kv["accepted"], spec)
    assert st["queue_closed"] and st["closed"]["n_items"] == inv["n"] and st["closed"]["item_keys_sha256"] == inv["item_keys_sha256"]


def test_p2_9_finish_producer_drains_the_tailer_before_sealing():
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    c = T.Container(L, shared, spec, "0", "a1")
    c.enqueue_row(ROWS[0])
    c.complete()
    order = []

    def drain():                                       # the tailer's join: its last item lands BEFORE the seal
        order.append(("drain", c.producer.sealed))
        c.enqueue_row(ROWS[1])

    out = c.finish(kv["accepted"], drain=drain)
    assert out["state"] == "accepted" and order == [("drain", False)]
    inv = json.loads(shared.files[c.producer.inventory_path].decode())
    assert inv["n"] == 2 and sorted(e["item_key"] for e in inv["entries"]) == ["static__pasttense__g0__1", "static__pasttense__g1__1"]
    # KNOWN-BAD: an enqueue after the finish is a producer bug
    with pytest.raises(L.ProducerSealed):
        c.enqueue_row(ROWS[2])


# --------------------------------------------------------------------------- #
# P1-3 — the seams take the source store's explicit seed base (three-seed Arditi view)
# --------------------------------------------------------------------------- #

ARDITI = "arditi_allpos"


class _FakeTarget:
    """A served target whose generate_detailed is the ENGINE: every call is recorded (the seam must not reach it)."""

    def __init__(self):
        self.evidence = {}
        self.calls = []

    def generate_detailed(self, prompts, *, max_new_tokens=512, decode=None, seeds=None, labels=None):
        self.calls.append(list(prompts))
        return [{"text": f"engine {p}", "finish_reason": "stop", "completion_tokens": 1, "prompt_tokens": 1,
                 "max_tokens": int(max_new_tokens), "sampling_seed": s} for p, s in zip(prompts, seeds or [None] * len(prompts))]


def _closed_static_queue(vol, spec, records, acc):
    T.close_queue(L, vol, spec, records, acc)


def _elect(vol, spec, rec, record, results, attempt="rung-1"):
    return T.materialise_elect(L, vol, spec, rec, record, results, attempt=attempt)


def _arditi_three_seed_ladder(tmp_path):
    """One static ladder root of view `arditi` holding the SAME item under arditi_allpos__s1, __s2, __s3 (three client-side
    seed bases on one engine), each with its own base row, seed and elected 8,192 draw."""
    vol = L.LocalVolume(tmp_path / "vol")
    spec = L.QueueSpec.build(T.W7, T.WAVE, "pasttense", "arditi", "static", 8192, ["0"])
    acc, res = L.MemoryKV("acc"), L.MemoryKV("res")
    rows, recs = {}, {}
    for s_ in (1, 2, 3):
        rows[s_] = T.static_row("g0", 1, seed=70 + s_, seed_base=s_, text=f"base text of s{s_}")
        recs[s_] = L.static_queue_record(spec, arm=ARDITI, s=s_, family="pasttense", row=rows[s_], base_store=f"exp/s{s_}/responses.jsonl",
                                         producer=spec.producers[0], producer_attempt_id="a1", vmodel="petri__arditi", sampling=T.PRESET)
    _closed_static_queue(vol, spec, list(recs.values()), acc)
    for s_ in (1, 2, 3):
        _elect(vol, spec, recs[s_], {"text": f"redraw of s{s_}", "finish_reason": "stop", "completion_tokens": 3, "prompt_tokens": 1,
                                     "max_tokens": 8192, "sampling_seed": 70 + s_}, res)
    snap = tmp_path / "results.json"
    L.snapshot_kv(res, snap)
    return tmp_path / "vol" / spec.ladder_root, snap, rows


def test_p1_3_static_assembly_of_the_three_seed_arditi_view_reads_exactly_its_seed_slot(tmp_path):
    ladder_root, snap, rows = _arditi_three_seed_ladder(tmp_path)
    asm = RR.load_assembly(ladder_root, snap)
    assert asm.arm_slots() == [f"{ARDITI}__s1", f"{ARDITI}__s2", f"{ARDITI}__s3"]          # three seed slots, one arm, one root
    assert asm.slot_for(ARDITI, 2) == f"{ARDITI}__s2" and asm.slot_for(ARDITI, 4) is None and asm.slot_for("none", 2) is None
    for s_ in (1, 2, 3):
        target = _FakeTarget()
        wrapped = RR.wrap_static_target(target, [(0, rows[s_])], ladder_root=ladder_root, results=snap, family="pasttense",
                                        defense=ARDITI, seed_base=s_, cap_to=8192, preset=T.PRESET, log=lambda m: None)
        (rec,) = wrapped.generate_detailed([rows[s_]["attack_prompt"]], max_new_tokens=8192, decode=T.PRESET, seeds=[rows[s_]["sampling_seed"]])
        assert rec["text"] == f"redraw of s{s_}" and rec["sampling_seed"] == 70 + s_ and target.calls == []   # its OWN seed's draw, engine untouched
        summ = target.evidence["seedrep_assembly"]
        assert (summ["arm_s"], summ["seed_base"], summ["n_elected"], summ["n_verified"], summ["n_materialised"]) == (f"{ARDITI}__s{s_}", s_, 1, 1, 1)
    # KNOWN-BAD: the assembly given seed base 3 over rows whose own decode.seed_base is 2 refuses (never another seed's slot)
    with pytest.raises(L.LadderRefusal, match="seed_base 2, this assembly was given 3"):
        RR.wrap_static_target(_FakeTarget(), [(0, rows[2])], ladder_root=ladder_root, results=snap, family="pasttense", defense=ARDITI,
                              seed_base=3, cap_to=8192, preset=T.PRESET, log=lambda m: None)
    # a seed base with no slot in the root: nothing materialised, everything drawn by the engine (noted, not refused)
    target = _FakeTarget()
    row4 = T.static_row("g0", 1, seed=74, seed_base=4)
    wrapped = RR.wrap_static_target(target, [(0, row4)], ladder_root=ladder_root, results=snap, family="pasttense", defense=ARDITI,
                                    seed_base=4, cap_to=8192, preset=T.PRESET, log=lambda m: None)
    assert wrapped is target and target.evidence["seedrep_assembly"]["arm_s"] is None


def test_p1_3_controls_and_misalign_seams_select_their_own_seed_slot(tmp_path):
    import types
    # controls: two seed slots of one cell in one 2048 queue; the seam at seed base 2 answers from s2's draw only
    vol = L.LocalVolume(tmp_path / "cvol")
    spec = L.QueueSpec.build(T.W7, T.WAVE, "controls", "arditi", "controls", 2048, ["0"])
    acc, res = L.MemoryKV("acc"), L.MemoryKV("res")
    recs = {}
    for s_ in (1, 2):
        ev = {"item_id": "x1", "at_cap": True, "finish_reason": "length", "sampling_seed": 500 + s_, "text_sha256": T.sha(f"base s{s_}")}
        recs[s_] = L.controls_queue_record(spec, arm=ARDITI, s=s_, control="xstest", event=ev, prompt="P1", system=None, base_store="b",
                                           producer=spec.producers[0], producer_attempt_id="a1", vmodel="petri__arditi", sampling=T.PRESET)
    _closed_static_queue(vol, spec, list(recs.values()), acc)
    for s_ in (1, 2):
        _elect(vol, spec, recs[s_], {"text": f"controls redraw s{s_}", "finish_reason": "stop", "completion_tokens": 3, "prompt_tokens": 17,
                                     "max_tokens": 2048, "sampling_seed": 500 + s_}, res)
    snap = tmp_path / "cres.json"
    L.snapshot_kv(res, snap)
    specs = {"xstest": types.SimpleNamespace(rows=[{"item_id": "x1", "prompt": "P1"}])}

    def sources_for(s_):
        meta = {"x1": {"text_sha256": T.sha(f"base s{s_}"), "sampling_seed": 500 + s_, "finish_reason": "length", "max_tokens": 512}}
        return {"xstest": types.SimpleNamespace(meta=meta, regen_ids={"x1"})}

    import asyncio
    for s_ in (1, 2):
        engine = []

        async def post(req, headers=None):
            engine.append(req)
            return {"choices": [{"message": {"content": "ENGINE"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

        wrapped = RR.wrap_controls_post(post, ladder_root=tmp_path / "cvol" / spec.ladder_root, results=snap, cell_id=ARDITI, seed_base=s_,
                                        cap=2048, decode=T.PRESET, specs=specs, sources=sources_for(s_), trial=0, log=lambda m: None)
        out = asyncio.run(wrapped({"messages": [{"role": "user", "content": "P1"}], "seed": 500 + s_, "max_tokens": 2048, **T.PRESET}))
        assert out["choices"][0]["message"]["content"] == f"controls redraw s{s_}" and engine == []
        summ = RR.wrap_controls_post.last_summary
        assert (summ["arm_s"], summ["seed_base"], summ["n_verified"]) == (f"{ARDITI}__s{s_}", s_, 1)
    # misalign: the seed base comes from the SOURCE cell's meta.json when the runner gives none; a source naming none refuses
    from sisteer.capeval.misalign import runner as R
    mvol = L.LocalVolume(tmp_path / "mvol")
    mspec = L.QueueSpec.build(T.W7, T.WAVE, "misalign", "arditi", "misalign", 16384, ["0"])
    macc, mres = L.MemoryKV("acc"), L.MemoryKV("res")
    base = {"family": "agentic", "cell_id": ARDITI, "virtual_model": "petri__arditi", "item_id": "leaking_explicit-none", "sample_idx": 0,
            "behavior": "leaking", "prompt_sha": "f" * 64, "response": "base " * 50, "finish_reason": "length",
            "usage": {"prompt_tokens": 3, "completion_tokens": 5}, "model_reported": "petri__arditi"}
    mrecs = {}
    for s_ in (1, 2):
        row = dict(base, seed=900 + s_)
        mrecs[s_] = L.misalign_queue_record(mspec, arm=ARDITI, s=s_, row=row, messages=[{"role": "user", "content": "m"}], base_store="b",
                                            producer=mspec.producers[0], producer_attempt_id="a1", sampling=T.PRESET)
    _closed_static_queue(mvol, mspec, list(mrecs.values()), macc)
    for s_ in (1, 2):
        rung_row = dict(base, seed=900 + s_, response=f"misalign redraw s{s_}", finish_reason="stop")
        rung_row["text"] = rung_row["response"]
        _elect(mvol, mspec, mrecs[s_], rung_row, mres)
    msnap = tmp_path / "mres.json"
    L.snapshot_kv(mres, msnap)
    src = tmp_path / "msrc"
    (src / ARDITI).mkdir(parents=True)
    (src / ARDITI / "responses.jsonl").write_text(json.dumps(dict(base, seed=902)) + "\n")
    (src / ARDITI / "meta.json").write_text(json.dumps({"cell_id": ARDITI, "seed_base": 2, "params": {"max_tokens": 3000}}))
    only = {ARDITI: [("leaking_explicit-none", 0)]}
    out = tmp_path / "mout"
    summ = RR.seed_misalign_root(out, ladder_root=tmp_path / "mvol" / mspec.ladder_root, results=msnap, src_root=src, cap=16384, only=only,
                                 cell_dirname=R.cell_dirname, read_done=R.read_done, log=lambda m: None)
    cs = summ["cells"][ARDITI]
    assert (cs["arm_s"], cs["seed_base"], cs["n_verified"], cs["n_seeded"]) == (f"{ARDITI}__s2", 2, 1, 1)
    seeded = json.loads((out / ARDITI / "responses.jsonl").read_text().splitlines()[0])
    assert seeded["response"] == "misalign redraw s2" and seeded["seed"] == 902
    # KNOWN-BAD: an explicit seed base disagreeing with the source meta refuses; a source with no seed base anywhere refuses
    with pytest.raises(L.LadderRefusal, match="seed_base 2, this assembly was given 1"):
        RR.seed_misalign_root(tmp_path / "m2", ladder_root=tmp_path / "mvol" / mspec.ladder_root, results=msnap, src_root=src, cap=16384,
                              only=only, cell_dirname=R.cell_dirname, read_done=R.read_done, seed_base=1, log=lambda m: None)
    (src / ARDITI / "meta.json").write_text(json.dumps({"cell_id": ARDITI, "params": {"max_tokens": 3000}}))
    with pytest.raises(L.LadderRefusal, match="no seed_base"):
        RR.seed_misalign_root(tmp_path / "m3", ladder_root=tmp_path / "mvol" / mspec.ladder_root, results=msnap, src_root=src, cap=16384,
                              only=only, cell_dirname=R.cell_dirname, read_done=R.read_done, log=lambda m: None)


# --------------------------------------------------------------------------- #
# P1-4 — GSM8K filters the queue to the source's exact (arm, seed) slot          #
# --------------------------------------------------------------------------- #


def test_p1_4_gsm8k_compares_only_the_source_arms_slot_against_its_store(tmp_path, monkeypatch):
    import test_inspect_rung as tir
    from sisteer.capeval.tiers import inspect_rung as IR

    tir._forbid_peer_exit_rule(monkeypatch)
    ids = tir._ids()
    none_store = tir._store(tmp_path / "none_src", cap=tir.CAP, at_cap={i: tir.CAP for i in ids[:8]})     # none: 8 at cap
    texts = tir._texts(ids, loops=ids[:2])
    vol = L.LocalVolume(tmp_path / "vol")
    spec = L.QueueSpec.build(T.W7, T.WAVE, "gsm8k", "s1", "gsm8k", 16384, ["0"])
    acc, res = L.MemoryKV("acc"), L.MemoryKV("res")

    def item(i, n):
        return {"item_id": i, "sample_idx": 0, "stop_reason": "max_tokens", "at_cap": True, "seed": n}

    recs = []
    for n, i in enumerate(ids[:8]):                                                              # the none arm's at-cap set
        recs.append(L.gsm8k_queue_record(spec, arm="none", s=1, item=item(i, n), base_store="none/outcomes.json", producer=spec.producers[0],
                                         producer_attempt_id="a1"))
    for n, i in enumerate(ids[:10]):                                                             # LoRA truncates two more: ids[8], ids[9]
        recs.append(L.gsm8k_queue_record(spec, arm="lora_x", s=1, item=item(i, 100 + n), base_store="lora/outcomes.json",
                                         producer=spec.producers[0], producer_attempt_id="a1"))
    _closed_static_queue(vol, spec, recs, acc)
    snap = tmp_path / "results.json"
    L.snapshot_kv(res, snap)
    ladder_root = tmp_path / "vol" / spec.ladder_root
    # assembling `none`: only none__s1's 8 entries are compared — ids[8], ids[9] (LoRA-only truncations) are not "foreign"
    record, subset, verdict = IR.subset_from_materialised_queue(ladder_root, snap, source_out_root=none_store, arm="none", seed_base=1, texts=texts)
    assert record["exit"] == "escalate" and subset["ids"] == sorted(ids[2:8])
    assert (verdict["arm_s"], verdict["n_queued"], verdict["missing_from_queue"], verdict["extra_in_queue"]) == ("none__s1", 8, [], sorted(ids[:2]))
    # KNOWN-BAD 1: the LoRA arm's slot against the none store — its two extra items were never filed at cap there
    with pytest.raises(L.LadderRefusal, match="2 queued item\\(s\\) the accepted source store never filed at cap"):
        IR.subset_from_materialised_queue(ladder_root, snap, source_out_root=none_store, arm="lora_x", seed_base=1, texts=texts)
    # KNOWN-BAD 2: a foreign item INSIDE the right slot still refuses
    vol2 = L.LocalVolume(tmp_path / "vol2")
    acc2 = L.MemoryKV("acc2")
    bad = recs[:8] + [L.gsm8k_queue_record(spec, arm="none", s=1, item=item(ids[50], 50), base_store="none/outcomes.json",
                                            producer=spec.producers[0], producer_attempt_id="a1")]
    _closed_static_queue(vol2, spec, bad, acc2)
    with pytest.raises(L.LadderRefusal, match="never filed at cap"):
        IR.subset_from_materialised_queue(tmp_path / "vol2" / spec.ladder_root, snap, source_out_root=none_store, arm="none", seed_base=1, texts=texts)
    # another seed's slot of the same arm is simply empty here: everything is "missing from the queue" (noted), nothing foreign
    _, _, v3 = IR.subset_from_materialised_queue(ladder_root, snap, source_out_root=none_store, arm="none", seed_base=2, texts=texts)
    assert v3["n_queued"] == 0 and v3["missing_from_queue"] == sorted(ids[2:8]) and v3["arm_s"] == "none__s2"


# --------------------------------------------------------------------------- #
# P2-10 — the misalign response adapter: queue -> HTTP -> consumer -> assembly    #
# --------------------------------------------------------------------------- #


def test_p2_10_misalign_draw_materialises_the_runners_row_and_the_assembly_seeds_it_through_the_full_chain(tmp_path):
    from sisteer.capeval.misalign import runner as R

    messages = [{"role": "system", "content": "scenario system"}, {"role": "user", "content": "the user prompt"}]
    base = {"family": "agentic", "cell_id": "none", "virtual_model": "petri__base", "item_id": "leaking_explicit-none", "sample_idx": 1,
            "behavior": "leaking", "prompt_sha": R.messages_sha(messages), "response": "truncated " * 40, "finish_reason": "length",
            "usage": {"prompt_tokens": 9, "completion_tokens": 3000}, "model_reported": "petri__base", "seed": 4242}
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = L.QueueSpec.build(T.W7, T.WAVE, "misalign", "s1", "misalign", 16384, ["0"])
    c = T.Container(L, shared, spec, "0", "a1")
    rec = L.misalign_queue_record(spec, arm="none", s=1, row=base, messages=messages, base_store="exp/agentic/none/responses.jsonl",
                                  producer=c.ref, producer_attempt_id="a1", sampling=T.PRESET)
    c.producer.enqueue(rec)
    c.complete()
    assert c.finish(kv["accepted"])["state"] == "accepted"
    assert L.reconcile_queue(shared.read_only_view(), kv["accepted"], spec)["queue_closed"]
    proxy = T.StubProxy(decide=lambda b: ("a full sixteen-k answer", "stop", 120))
    try:
        rv = shared.snapshot("rung-1")
        tally = RR.consume(rv, accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=spec, attempt_id="rung-1",
                           call_id="fc-1", draw=RR.http_draw_fn(proxy.url), sleep=lambda s: None, log=lambda m: None)
    finally:
        proxy.close()
    assert tally["drawn"] == 1 and tally["won"] == 1 and tally["closed"] and tally["covered"]
    sent = proxy.bodies[0]["body"]
    assert sent["messages"] == messages and sent["max_tokens"] == 16384 and sent["seed"] == 4242 and sent["model"] == "petri__base"
    rid = proxy.bodies[0]["headers"]["X-Tinycap-Request-Id"]
    # the materialised record IS the runner's row (+ seed, + request provenance), not the generic turn record
    (el,) = L.load_elected(shared.read_only_view(), kv["results"], spec, "none__s1").values()
    row = el.doc["record"]
    assert {k: row[k] for k in RR.MISALIGN_ROW_KEYS} == {
        "family": "agentic", "cell_id": "none", "virtual_model": "petri__base", "item_id": "leaking_explicit-none", "sample_idx": 1,
        "behavior": "leaking", "prompt_sha": R.messages_sha(messages), "response": "a full sixteen-k answer", "finish_reason": "stop",
        "usage": {"prompt_tokens": 11, "completion_tokens": 120, "total_tokens": 131}, "model_reported": "petri__base"}
    assert row["seed"] == 4242 and row["request_id"] == rid == el.value["request_id"] and row["completion_id"] == "cmpl-1" == el.value["completion_id"]
    assert "text" not in row and el.value["text_sha256"] == T.sha("a full sixteen-k answer")
    # the assembly: the runner's --rerun-truncated-from selection over the source root + this CLOSED queue seeds the row as-is
    src = tmp_path / "src"
    (src / "none").mkdir(parents=True)
    (src / "none" / "responses.jsonl").write_text(json.dumps(base) + "\n")
    (src / "none" / "meta.json").write_text(json.dumps({"cell_id": "none", "seed_base": 1, "params": {"max_tokens": 3000}}))
    pull = tmp_path / "pull"
    for k, v in shared.files.items():                                  # the volume pulled with its relative prefix intact
        (pull / k).parent.mkdir(parents=True, exist_ok=True)
        (pull / k).write_bytes(v)
    snap = tmp_path / "results.json"
    L.snapshot_kv(kv["results"], snap)
    out = tmp_path / "out16k"
    summ = RR.seed_misalign_root(out, ladder_root=pull / spec.ladder_root, results=snap, src_root=src, cap=16384,
                                 only={"none": [("leaking_explicit-none", 1)]}, cell_dirname=R.cell_dirname, read_done=R.read_done,
                                 log=lambda m: None)
    cs = summ["cells"]["none"]
    assert (cs["n_elected"], cs["n_verified"], cs["n_seeded"], cs["refused"]) == (1, 1, 1, [])
    seeded = [json.loads(l) for l in (out / "none" / "responses.jsonl").read_text().splitlines()]
    assert seeded == [row] and R.read_done(out / "none" / "responses.jsonl") == {("leaking_explicit-none", 1)}
    prov = json.loads((out / "none" / "materialised_seeded.jsonl").read_text().splitlines()[0])
    assert prov["request_id"] == rid and prov["completion_id"] == "cmpl-1" and prov["attempt_id"] == "rung-1"
    # KNOWN-BAD: a record whose identity prompt_sha is not the re-issued messages' refuses at the draw (nothing materialised);
    # a completion without usage refuses too
    tampered = copy.deepcopy(rec)
    tampered["identity"]["prompt_sha"] = "0" * 64
    proxy = T.StubProxy(decide=lambda b: ("x", "stop", 1))
    try:
        with pytest.raises(L.LadderRefusal, match="a rung re-draws the SAME prompt"):
            RR.http_draw_fn(proxy.url)(tampered)
    finally:
        proxy.close()
    with pytest.raises(L.LadderRefusal, match="no finish_reason / usage"):
        RR.record_from_response(rec, {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]})
    with pytest.raises(L.LadderRefusal, match="expected exactly one choice"):
        RR.record_from_response(rec, {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}] * 2, "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


# --------------------------------------------------------------------------- #
# P1-8 — draws dispatch concurrently at the attested rung width                  #
# --------------------------------------------------------------------------- #


def _blocking_first_draw(first_key, release, order, wait_s):
    olock = threading.Lock()

    def draw(rec):
        if rec["item_key"] == first_key:
            assert release.wait(timeout=wait_s), "the first request was never released: no later request started and elected"
        with olock:
            order.append(rec["item_key"])
        return RR.Draw(record={"text": f"t {rec['item_key']}", "finish_reason": "stop", "completion_tokens": 1, "prompt_tokens": 1,
                               "max_tokens": 8192, "sampling_seed": rec["sampling_seed"]}, text=f"t {rec['item_key']}", response=None,
                       request_id=None, completion_id=None)

    return draw


def _release_when_others_elected(results, first_key, release, n_others):
    def watcher():
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if sum(1 for k, _ in results.items() if not k.endswith("|" + first_key)) >= n_others:
                release.set()
                return
            time.sleep(0.005)

    threading.Thread(target=watcher, daemon=True).start()


def test_p1_8_a_blocked_first_request_does_not_hold_the_later_ones_at_width_3_and_does_at_width_1():
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    _finished_container(shared, spec, kv)
    view = shared.read_only_view()
    assert L.reconcile_queue(view, kv["accepted"], spec)["queue_closed"]
    keys = [sl.split("/", 1)[1] for sl in sorted(L.accepted_queue_entries(view, kv["accepted"], spec))]
    first, others = keys[0], set(keys[1:])
    release, order = threading.Event(), []
    _release_when_others_elected(kv["results"], first, release, len(others))
    tally = RR.consume(shared.snapshot("rung-1"), accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=spec,
                       attempt_id="rung-1", call_id="fc-1", draw=_blocking_first_draw(first, release, order, wait_s=30), width=3,
                       sleep=lambda s: None, log=lambda m: None)
    assert release.is_set()                                                   # the later two were drawn AND elected while the first blocked
    assert (tally["drawn"], tally["won"], tally["width"], tally["closed"], tally["covered"]) == (3, 3, 3, True, True)
    assert set(order[:2]) == others and order[2] == first
    elected = L.load_elected(view, kv["results"], spec, "none__s1")
    assert {k: e.value["attempt_id"] for k, e in elected.items()} == {k: "rung-1" for k in keys}
    for k, e in elected.items():
        assert e.doc["record"]["text"] == f"t {k}" and e.doc["claimant"]["call_id"] == "fc-1"
    # KNOWN-BAD: at width 1 the blocked first request is the whole rung — nothing later starts, nothing is elected
    shared2, kv2 = L.SharedVolume(), T.kvs(L)
    _finished_container(shared2, spec, kv2)
    release2, order2 = threading.Event(), []
    _release_when_others_elected(kv2["results"], first, release2, len(others))
    with pytest.raises(AssertionError, match="never released"):
        RR.consume(shared2.snapshot("rung-1"), accepted_kv=kv2["accepted"], claims_kv=kv2["claims"], results_kv=kv2["results"], spec=spec,
                   attempt_id="rung-1", call_id="fc-1", draw=_blocking_first_draw(first, release2, order2, wait_s=0.5), width=1,
                   sleep=lambda s: None, log=lambda m: None)
    assert order2 == [] and len(kv2["results"]) == 0
    # width below 1 refuses
    with pytest.raises(L.LadderRefusal, match="width 0"):
        RR.consume(shared2.snapshot("rung-2"), accepted_kv=kv2["accepted"], claims_kv=kv2["claims"], results_kv=kv2["results"], spec=spec,
                   attempt_id="rung-2", call_id="fc-2", draw=lambda r: None, width=0)


def test_p1_8_a_failing_draw_stops_the_rung_and_leaves_nothing_half_published():
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    _finished_container(shared, spec, kv)
    keys = [sl.split("/", 1)[1] for sl in sorted(L.accepted_queue_entries(shared.read_only_view(), kv["accepted"], spec))]

    def draw(rec):
        if rec["item_key"] == keys[1]:
            raise L.LadderRefusal("proxy returned 500")
        time.sleep(0.05)
        return RR.Draw(record={"text": "ok", "finish_reason": "stop", "completion_tokens": 1, "prompt_tokens": 1, "max_tokens": 8192,
                               "sampling_seed": rec["sampling_seed"]}, text="ok", response=None, request_id=None, completion_id=None)

    with pytest.raises(L.LadderRefusal, match="proxy returned 500"):
        RR.consume(shared.snapshot("rung-1"), accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=spec,
                   attempt_id="rung-1", call_id="fc-1", draw=draw, width=3, sleep=lambda s: None, log=lambda m: None)
    # every elected result has its committed file with a matching sha; the failed item has no result and no file
    elected = L.load_elected(shared.read_only_view(), kv["results"], spec, "none__s1")
    assert keys[1] not in elected and all(e.doc["record"]["text"] == "ok" for e in elected.values())
    assert not any(k.endswith(f"/{keys[1]}/rung-1.json") for k in shared.files)


# --------------------------------------------------------------------------- #
# P1-7 — next-queue publication is recoverable from validated elected results    #
# --------------------------------------------------------------------------- #


def _rung_world():
    """A CLOSED 8192 static queue of three accepted items; g0's re-draw is at cap (feeds the 16384 queue)."""
    shared, kv = L.SharedVolume(), T.kvs(L)
    q8 = T.static_spec(L, shards=("0",))
    _finished_container(shared, q8, kv)
    assert L.reconcile_queue(shared.read_only_view(), kv["accepted"], q8)["queue_closed"]
    return shared, kv, q8


def _at_cap_g0_draw(rec):
    at_cap = rec["identity"]["goal_id"] == "g0"
    text = f"redraw {rec['identity']['goal_id']}"
    return RR.Draw(record={"text": text, "finish_reason": "length" if at_cap else "stop", "completion_tokens": 8192 if at_cap else 3,
                           "prompt_tokens": 1, "max_tokens": 8192, "sampling_seed": rec["sampling_seed"]}, text=text, response=None,
                   request_id=f"rq-{rec['item_key']}", completion_id="cmpl-x")


class _DiesOnEnqueue(L.Producer):
    """A rung attempt's next-queue producer that dies (the container is lost) at its first enqueue — after the election."""

    def enqueue(self, record):
        raise L._Crash("died before publishing the next-queue entry")


def _rung(shared, kv, q8, q16, attempt, *, producer_cls=L.Producer, draw=_at_cap_g0_draw, **kw):
    rv = shared.snapshot(attempt)
    nxt = producer_cls(rv, q16, q16.producers[0], attempt, L.producer_attempt_root(q16.producers[0], attempt))
    tally = RR.consume(rv, accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=q8, attempt_id=attempt,
                       call_id=f"fc-{attempt}", draw=draw, next_producer=nxt, sleep=lambda s: None, log=lambda m: None, **kw)
    return rv, nxt, tally


def test_p1_7_a_replacement_rung_publishes_the_elected_at_cap_result_its_predecessor_died_before_forwarding():
    shared, kv, q8 = _rung_world()
    q16 = q8.next_spec(["0"])
    # attempt A elects g0's at-cap draw, then dies before the next-queue enqueue
    with pytest.raises(L._Crash):
        _rung(shared, kv, q8, q16, "rung-1", producer_cls=_DiesOnEnqueue)
    g0 = L.item_key("static", "pasttense", "g0", 1)
    el = L.load_elected_one(shared.read_only_view(), kv["results"], q8, "none__s1", g0)
    assert el is not None and el.value["attempt_id"] == "rung-1" and el.doc["record"]["finish_reason"] == "length"
    assert L.live_queue_entries(shared.read_only_view(), q16) == {}                    # nothing reached the 16384 queue
    assert len(kv["claims"]) == 1                                                        # A claimed only the item it drew
    # the reconciler (A's call is terminal) clears A's undrawn claims — g0's stands: it has its result
    st = L.reconcile_queue(shared.read_only_view(), kv["accepted"], q8, claims_kv=kv["claims"], results_kv=kv["results"],
                           lease_state=lambda cid: "failed")
    assert st["reclaimed"] == [] and len(kv["claims"]) == 1
    # the replacement B (same shard r0): finds g0's result elected, publishes the 16384 entry under ITS OWN attempt id, draws the rest
    rv, nxt, t = _rung(shared, kv, q8, q16, "rung-2",
                       draw=lambda r: _at_cap_g0_draw(r) if r["identity"]["goal_id"] != "g0" else pytest.fail("B re-drew the elected item"))
    assert (t["drawn"], t["already_elected"], t["enqueued_next"], t["forwarded_found"], t["closed"], t["covered"]) == (2, 1, 1, 1, True, True)
    rv.write_bytes(L.complete_rel(nxt.attempt_root), b"{}")
    assert L.finish_producer(nxt, kv["accepted"])["state"] == "accepted"
    st = L.reconcile_queue(shared.read_only_view(), kv["accepted"], q16)
    assert st["queue_closed"] and st["closed"]["n_items"] == 1 and st["closed"]["producers"] == [["0", "rung-2"]]
    (nq,) = L.accepted_queue_entries(shared.read_only_view(), kv["accepted"], q16).values()
    assert nq["producer"] == {"shard": "0", "attempt_id": "rung-2"} and nq["base_store"] == el.path and nq["item_key"] == g0
    assert nq["base_row_sha256"] == L.base_row_identity_sha256(text="redraw g0", sampling_seed=7, finish_reason="length", cap=8192)
    # a third pass (idempotence): nothing new is enqueued by an attempt that already forwarded, nothing drawn
    rv3, nxt3, t3 = _rung(shared, kv, q8, q16, "rung-3", draw=lambda r: pytest.fail("drew"), max_rounds=2)
    assert t3["enqueued_next"] == 1 and len([k for k in shared.files if k.endswith("/rung-3.json") and "/16384/queue/" in k]) == 1


def test_p1_7_a_predecessor_that_forwarded_then_died_before_acceptance_leaves_an_inert_file_the_replacement_supersedes():
    shared, kv, q8 = _rung_world()
    q16 = q8.next_spec(["0"])
    rv1, nxt1, t1 = _rung(shared, kv, q8, q16, "rung-1")                                # A draws, elects, forwards …
    assert t1["enqueued_next"] == 1 and t1["won"] == 3
    rv1.crash()                                                                          # … and dies before finish_producer
    assert len(L.live_queue_entries(shared.read_only_view(), q16)) == 1                  # A's 16384 file is committed but A is never accepted
    rv2, nxt2, t2 = _rung(shared, kv, q8, q16, "rung-2", draw=lambda r: pytest.fail("drew"))
    assert t2["forwarded_found"] == 1 and t2["already_elected"] == 3
    rv2.write_bytes(L.complete_rel(nxt2.attempt_root), b"{}")
    assert L.finish_producer(nxt2, kv["accepted"])["state"] == "accepted"
    st = L.reconcile_queue(shared.read_only_view(), kv["accepted"], q16)
    assert st["queue_closed"] and st["closed"]["n_items"] == 1 and st["closed"]["producers"] == [["0", "rung-2"]]
    g0 = L.item_key("static", "pasttense", "g0", 1)
    live = L.live_queue_entries(shared.read_only_view(), q16)[f"none__s1/{g0}"]
    assert sorted(r["producer"]["attempt_id"] for r in live) == ["rung-1", "rung-2"]     # both files exist …
    (acc_rec,) = L.accepted_queue_entries(shared.read_only_view(), kv["accepted"], q16).values()
    assert acc_rec["producer"]["attempt_id"] == "rung-2"                                   # … only rung-2's is read


def test_p1_7_two_live_rung_shards_forwarding_one_elected_result_are_one_entry_and_a_differing_pair_refuses():
    shared, kv, q8 = _rung_world()
    q16 = q8.next_spec(["0", "1"])                                                     # two rung containers over one queue (static burst)
    rv0 = shared.snapshot("rung-a")
    p0 = L.Producer(rv0, q16, q16.producers[0], "rung-a", L.producer_attempt_root(q16.producers[0], "rung-a"))
    ta = RR.consume(rv0, accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=q8, attempt_id="rung-a",
                    call_id="fc-a", draw=_at_cap_g0_draw, next_producer=p0, sleep=lambda s: None, log=lambda m: None)
    rv1 = shared.snapshot("rung-b")
    p1 = L.Producer(rv1, q16, q16.producers[1], "rung-b", L.producer_attempt_root(q16.producers[1], "rung-b"))
    tb = RR.consume(rv1, accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=q8, attempt_id="rung-b",
                    call_id="fc-b", draw=lambda r: pytest.fail("drew"), next_producer=p1, sleep=lambda s: None, log=lambda m: None)
    assert ta["enqueued_next"] == 1 and tb["enqueued_next"] == 1 and tb["forwarded_found"] == 1
    for rv, p in ((rv0, p0), (rv1, p1)):
        rv.write_bytes(L.complete_rel(p.attempt_root), b"{}")
        assert L.finish_producer(p, kv["accepted"])["state"] == "accepted"
    st = L.reconcile_queue(shared.read_only_view(), kv["accepted"], q16)
    assert st["queue_closed"] and st["closed"]["n_items"] == 1 and st["closed"]["producers"] == [["0", "rung-a"], ["1", "rung-b"]]
    (rec,) = L.accepted_queue_entries(shared.read_only_view(), kv["accepted"], q16).values()
    assert rec["producer"]["attempt_id"] == "rung-a"                                       # the first by declared shard order
    # KNOWN-BAD: the two accepted producers' entries DIFFER in binding -> refused
    g0 = L.item_key("static", "pasttense", "g0", 1)
    rel_b = L.queue_entry_rel(q16.ladder_root, "none__s1", g0, "rung-b")
    forged = json.loads(shared.files[rel_b].decode())
    forged["base_row_sha256"] = "f" * 64
    shared.files[rel_b] = (L.canonical_json(forged) + "\n").encode()
    with pytest.raises(L.LadderRefusal, match="DIFFERENT binding"):
        L.accepted_queue_entries(shared.read_only_view(), kv["accepted"], q16)


def test_p1_7_a_found_result_bound_to_another_base_row_is_not_forwarded():
    shared, kv, q8 = _rung_world()
    q16 = q8.next_spec(["0"])
    # an elected result whose queue record binds to ANOTHER base row (a different producer attempt's world)
    g0 = L.item_key("static", "pasttense", "g0", 1)
    (acc_rec,) = [r for r in L.accepted_queue_entries(shared.read_only_view(), kv["accepted"], q8).values() if r["item_key"] == g0]
    other = copy.deepcopy(acc_rec)
    other["base_row_sha256"], other["producer"] = "e" * 64, {"shard": "0", "attempt_id": "a0"}
    rv0 = shared.snapshot("rung-0")
    path, doc = L.write_materialised(rv0, q8, queue_record=other, attempt_id="rung-0", call_id="fc-0",
                                     record={"text": "stale", "finish_reason": "length", "completion_tokens": 8192, "prompt_tokens": 1,
                                             "max_tokens": 8192, "sampling_seed": 7}, text="stale", request_id=None, completion_id=None)
    rv0.commit()
    assert L.elect_result(kv["results"], q8, doc, path)
    rv, nxt, t = _rung(shared, kv, q8, q16, "rung-1", draw=lambda r: _at_cap_g0_draw(r) if r["identity"]["goal_id"] != "g0" else pytest.fail("g0 is elected"))
    assert (t["already_elected"], t["unforwardable"], t["enqueued_next"], t["forwarded_found"]) == (1, 1, 0, 0)
    assert L.live_queue_entries(shared.read_only_view(), q16) == {}


# --------------------------------------------------------------------------- #
# P1-5 — --speculative keeps draws tentative until their producer is accepted     #
# --------------------------------------------------------------------------- #


def _spec_draw(rec):
    at_cap = rec["identity"]["goal_id"] == "g0"
    text = f"spec {rec['identity']['goal_id']} by {rec['producer']['attempt_id']}"
    return RR.Draw(record={"text": text, "finish_reason": "length" if at_cap else "stop", "completion_tokens": 8192 if at_cap else 3,
                           "prompt_tokens": 1, "max_tokens": 8192, "sampling_seed": rec["sampling_seed"]}, text=text, response=None,
                   request_id=None, completion_id=None)


def test_p1_5_a_speculative_draw_is_promoted_elected_and_forwarded_only_after_its_producer_is_accepted():
    shared, kv = L.SharedVolume(), T.kvs(L)
    q8 = T.static_spec(L, shards=("0",))
    q16 = q8.next_spec(["0"])
    a = T.Container(L, shared, q8, "0", "a1")
    ra = a.enqueue_row(T.static_row("g0", 1, seed=7), commit=True)          # live, not accepted: the base leg is still running
    rv = shared.snapshot("rung-1")
    nxt = L.Producer(rv, q16, q16.producers[0], "rung-1", L.producer_attempt_root(q16.producers[0], "rung-1"))
    seen = []

    def on_sleep(_s):                                                        # between round 1 and round 2 the base leg completes
        seen.append(("results_at_sleep", len(kv["results"]), len(L.live_queue_entries(shared.read_only_view(), q16))))
        a.complete()
        assert a.finish(kv["accepted"])["state"] == "accepted"
        assert L.reconcile_queue(shared.read_only_view(), kv["accepted"], q8)["queue_closed"]

    t = RR.consume(rv, accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=q8, attempt_id="rung-1", call_id="fc",
                   draw=_spec_draw, next_producer=nxt, speculative=True, sleep=on_sleep, log=lambda m: None)
    assert seen == [("results_at_sleep", 0, 0)]                              # round 1: drawn, NOT elected, NOT forwarded
    assert (t["rounds"], t["drawn"], t["speculative_drawn"], t["speculative_promoted"], t["won"], t["enqueued_next"]) == (2, 1, 1, 1, 1, 1)
    assert t["speculative_pending"] == 0 and t["closed"] and t["covered"]
    el = L.load_elected(shared.read_only_view(), kv["results"], q8, "none__s1")[ra["item_key"]]
    assert el.path == L.materialised_rel(q8.ladder_root, "none__s1", ra["item_key"], "rung-1")
    assert el.doc["promoted_from"] == L.speculative_rel(q8.ladder_root, "none__s1", ra["item_key"], "rung-1") and "speculative" not in el.doc
    assert el.doc["record"]["text"] == "spec g0 by a1" and shared.read_only_view().exists(el.doc["promoted_from"])
    L.verify_elected(el, base_row_sha256=ra["base_row_sha256"], sampling_seed=7, cap=8192, preset=T.PRESET, accepted_producers=["a1"])
    (nq,) = L.live_queue_entries(shared.read_only_view(), q16)[f"none__s1/{ra['item_key']}"]
    assert nq["producer"]["attempt_id"] == "rung-1" and nq["base_store"] == el.path              # forwarded from the PROMOTED file


def test_p1_5_a_speculative_draw_of_a_losing_producer_is_dropped_and_the_accepted_record_is_drawn_instead():
    shared, kv = L.SharedVolume(), T.kvs(L)
    q8 = T.static_spec(L, shards=("0",))
    a, b = T.Container(L, shared, q8, "0", "a1"), T.Container(L, shared, q8, "0", "a2")
    ra = a.enqueue_row(T.static_row("g0", 1, seed=7), commit=True)
    rv = shared.snapshot("rung-1")

    def on_sleep(_s):                                                        # a re-execution with ANOTHER base text wins the acceptance
        rb = b.enqueue_row(T.static_row("g0", 1, seed=7, text="other base"), commit=True)
        assert rb["item_key"] == ra["item_key"] and rb["base_row_sha256"] != ra["base_row_sha256"]
        b.complete()
        assert b.finish(kv["accepted"])["state"] == "accepted"
        assert L.reconcile_queue(shared.read_only_view(), kv["accepted"], q8)["queue_closed"]

    t = RR.consume(rv, accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=q8, attempt_id="rung-1", call_id="fc",
                   draw=_spec_draw, speculative=True, sleep=on_sleep, log=lambda m: None)
    assert (t["drawn"], t["speculative_drawn"], t["speculative_dropped"], t["speculative_promoted"], t["won"], t["skipped_held"]) == (2, 1, 1, 0, 1, 0)
    el = L.load_elected(shared.read_only_view(), kv["results"], q8, "none__s1")[ra["item_key"]]
    assert el.doc["record"]["text"] == "spec g0 by a2" and el.doc["queue_record"]["producer"]["attempt_id"] == "a2" and "promoted_from" not in el.doc
    L.verify_elected(el, base_row_sha256=L.base_row_identity_sha256(text="other base", sampling_seed=7, finish_reason="length", cap=4096),
                     sampling_seed=7, cap=8192, preset=T.PRESET, accepted_producers=["a2"])
    # the loser's tentative file is evidence only: never at a materialised path, never in the Dict
    spec_path = L.speculative_rel(q8.ladder_root, "none__s1", ra["item_key"], "rung-1")
    assert shared.read_only_view().exists(spec_path) and json.loads(shared.files[spec_path])["record"]["text"] == "spec g0 by a1"
    assert [k for k in shared.files if "/materialised/" in k] == [el.path] and len(kv["results"]) == 1


def test_p1_5_a_known_losers_live_entry_is_never_drawn():
    shared, kv = L.SharedVolume(), T.kvs(L)
    q8 = T.static_spec(L, shards=("0",))
    a, b = T.Container(L, shared, q8, "0", "a1"), T.Container(L, shared, q8, "0", "a2")
    a.enqueue_row(T.static_row("g0", 1, seed=7), commit=True)               # a1: live entry for g0, a1 never finishes
    b.enqueue_row(T.static_row("g1", 1, seed=8), commit=True)               # a2: g1 only, ACCEPTED before the rung starts
    b.complete()
    assert b.finish(kv["accepted"])["state"] == "accepted"
    assert L.reconcile_queue(shared.read_only_view(), kv["accepted"], q8)["queue_closed"]
    drawn = []

    def draw(rec):
        assert rec["producer"]["attempt_id"] == "a2", "a known loser's entry was drawn"
        drawn.append(rec["item_key"])
        return _spec_draw(rec)

    t = RR.consume(shared.snapshot("rung-1"), accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"], spec=q8,
                   attempt_id="rung-1", call_id="fc", draw=draw, speculative=True, sleep=lambda s: None, log=lambda m: None)
    assert drawn == [L.item_key("static", "pasttense", "g1", 1)]
    assert (t["drawn"], t["speculative_drawn"], t["speculative_losers_skipped"], t["closed"], t["covered"]) == (1, 0, 1, True, True)
    assert not any("/speculative/" in k for k in shared.files)


# --------------------------------------------------------------------------- #
# P1-1 — the production CLI binds the mounted Volume's real commit/reload, uses the Volume API remotely,
#        and exits nonzero (no producer finish) unless the queue is CLOSED and covered
# --------------------------------------------------------------------------- #


class _FakeVolume:
    """A stand-in for a modal.Volume handle: the mount's commit/reload (counted) AND the Volume API the grading host
    RemoteVolume adapter uses (listdir / read_file / batch_upload), all over one local directory."""

    def __init__(self, root: Path):
        self.root, self.commits, self.reloads, self.uploads = Path(root), 0, 0, []

    def commit(self):
        self.commits += 1

    def reload(self):
        self.reloads += 1

    def listdir(self, path, *, recursive=False):
        import types
        p = self.root / path
        if not p.is_dir():
            raise FileNotFoundError(path)
        it = p.rglob("*") if recursive else p.iterdir()
        return [types.SimpleNamespace(path="/" + q.relative_to(self.root).as_posix(),
                                      type=types.SimpleNamespace(name="DIRECTORY" if q.is_dir() else "FILE")) for q in sorted(it)]

    def read_file(self, path):
        data = (self.root / path).read_bytes()
        yield data[:7]
        yield data[7:]

    def batch_upload(self, force=False):
        vol = self

        class _Batch:
            def __enter__(self):
                return self

            def put_file(self, fileobj, remote):
                dest = vol.root / remote.lstrip("/")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(fileobj.read())
                vol.uploads.append(remote)

            def __exit__(self, *a):
                return False
        return _Batch()


class _FakeDict(L.MemoryKV):
    """modal.Dict's surface as the adapter calls it (put(skip_if_exists) / get / pop / contains / items)."""


def _fake_modal(monkeypatch, volumes: dict, dicts: dict):
    import types
    mod = types.ModuleType("modal")
    mod.Volume = types.SimpleNamespace(from_name=lambda name, **k: volumes[name])
    mod.Dict = types.SimpleNamespace(from_name=lambda name, create_if_missing=True: dicts.setdefault(name, _FakeDict(name)))
    monkeypatch.setitem(sys.modules, "modal", mod)
    return mod


def _tally(out) -> dict:
    """The CLI's tally JSON (indent=1, so it starts a line with '{'); the consumer's log lines precede it on stdout."""
    return json.loads(out.out[out.out.index("{\n"):])


def _axes(spec):
    return ["--W7", spec.W7, "--wave-uuid", spec.wave_uuid, "--leg", spec.leg, "--view", spec.view, "--kind", spec.kind,
            "--cap-next", str(spec.cap_next), "--producers", ",".join(p.shard for p in spec.producers)]


def _tally_of(stdout: str) -> dict:
    """The CLI's final tally JSON (indent=1: it starts on a line of its own after the consumer's log lines)."""
    return json.loads(stdout[stdout.rindex("\n{\n") + 1:] if "\n{\n" in stdout else stdout)


def test_p1_1_consume_cli_binds_the_volumes_commit_and_reload_uses_the_modal_dicts_and_finishes_only_when_covered(tmp_path, monkeypatch, capsys):
    vol_root = tmp_path / "vol"
    vol = L.LocalVolume(vol_root)
    spec = T.static_spec(L, shards=("0",))
    dicts: dict = {}
    fake_vol = _FakeVolume(vol_root)
    _fake_modal(monkeypatch, {"twosi-fits": fake_vol}, dicts)
    acc = _FakeDict(L.dict_name(T.W7, T.WAVE, "accepted"))
    dicts[acc.name] = acc
    recs = {}
    for r in ROWS:
        rec = L.static_queue_record(spec, arm="none", s=1, family="pasttense", row=r, base_store="b", producer=spec.producers[0],
                                    producer_attempt_id="a1", vmodel="petri__base", sampling=T.PRESET)
        recs[rec["item_key"]] = rec
    _closed_static_queue(vol, spec, list(recs.values()), acc)
    proxy = T.StubProxy()
    try:
        rc = RR.main(["consume", "--vol-root", str(vol_root), "--volume", "twosi-fits", *_axes(spec), "--attempt-id", "rung-1", "--call-id", "fc-1",
                      "--no-request-evidence",
                      "--proxy-url", proxy.url, "--concurrency", "2", "--rung-shard", "0", "--rung-shards", "0", "--finish-producer"])
    finally:
        proxy.close()
    out = capsys.readouterr()
    assert rc == 0
    tally = _tally(out)
    assert (tally["drawn"], tally["won"], tally["width"], tally["closed"], tally["covered"]) == (3, 3, 2, True, True)
    assert fake_vol.reloads >= 1 and fake_vol.commits >= 3                      # the mount's REAL reload / commit were bound and used
    assert set(dicts) == {L.dict_name(T.W7, T.WAVE, p) for p in ("accepted", "claims", "results")}   # the wave's Dicts, by their names
    assert len(dicts[L.dict_name(T.W7, T.WAVE, "results")]) == 3 and len(dicts[L.dict_name(T.W7, T.WAVE, "claims")]) == 3
    assert len(proxy.bodies) == 3 and all(b["headers"].get("X-Tinycap-Request-Id") for b in proxy.bodies)
    # the rung finished its producer side for the 16384 queue (CLOSED + covered): inventory written, acceptance taken
    q16 = spec.next_spec(["0"])
    assert tally["producer"]["state"] == "accepted" and L.accepted_attempt(acc, q16.producers[0].key) == "rung-1"
    inv = json.loads((vol_root / L.inventory_rel(L.attempt_root_for(T.W7, leg="pasttense", view="s1", stage="8192", shard="0", attempt_id="rung-1"))).read_text())
    assert inv["n"] == 0                                                           # every re-draw terminated: nothing for 16384
    # KNOWN-BAD: an OPEN queue (a live producer, nothing accepted) by the deadline -> exit 3, nothing finished
    vol2_root = tmp_path / "vol2"
    fake2 = _FakeVolume(vol2_root)
    dicts2: dict = {}
    _fake_modal(monkeypatch, {"twosi-fits": fake2}, dicts2)
    prod = L.Producer(L.LocalVolume(vol2_root), spec, spec.producers[0], "a1", L.attempt_root_rel(T.W7, "pasttense", "s1", "0", "a1"))
    prod.enqueue(next(iter(recs.values())))
    proxy = T.StubProxy()
    try:
        rc = RR.main(["consume", "--vol-root", str(vol2_root), "--volume", "twosi-fits", *_axes(spec), "--attempt-id", "rung-9", "--call-id", "fc-9",
                      "--proxy-url", proxy.url, "--width", "4", "--rung-shard", "0", "--rung-shards", "0", "--finish-producer", "--deadline-s", "0",
                      "--no-request-evidence"])
    finally:
        proxy.close()
    out = capsys.readouterr()
    assert rc == RR.EXIT_INCOMPLETE == 3 and "INCOMPLETE" in out.err and "NOT finished" in out.err
    tally = _tally(out)
    assert tally["closed"] is False and "producer" not in tally and proxy.bodies == []
    assert not (vol2_root / L.inventory_rel(L.attempt_root_for(T.W7, leg="pasttense", view="s1", stage="8192", shard="0", attempt_id="rung-9"))).exists()
    assert len(dicts2.get(L.dict_name(T.W7, T.WAVE, "accepted"), {})) == 0 and fake2.reloads >= 1


def test_p1_1_reconcile_cli_reads_and_writes_through_the_volume_api_and_the_parser_refuses_ambiguous_forms(tmp_path, monkeypatch, capsys):
    vol_root = tmp_path / "vol"
    vol = L.LocalVolume(vol_root)
    spec = T.static_spec(L, shards=("0",))
    fake_vol = _FakeVolume(vol_root)
    dicts: dict = {}
    _fake_modal(monkeypatch, {"twosi-fits": fake_vol}, dicts)
    acc = _FakeDict(L.dict_name(T.W7, T.WAVE, "accepted"))
    dicts[acc.name] = acc
    # an accepted producer that died after its acceptance (no marker): the remote reconciler must synthesise + close
    root = L.attempt_root_rel(T.W7, "pasttense", "s1", "0", "a1")
    prod = L.Producer(vol, spec, spec.producers[0], "a1", root)
    for r in ROWS:
        prod.enqueue(L.static_queue_record(spec, arm="none", s=1, family="pasttense", row=r, base_store="b", producer=spec.producers[0],
                                           producer_attempt_id="a1", vmodel="petri__base", sampling=T.PRESET))
    vol.write_bytes(L.complete_rel(root), b"{}")
    prod.write_inventory()
    assert L.accept_attempt(acc, spec.producers[0].key, "a1")
    rc = RR.main(["reconcile", "--volume", "twosi-fits", *_axes(spec), "--out", str(tmp_path / "status.json")])
    assert rc == 0
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["producers"]["0"]["state"] == "CLOSED_SYNTHESISED" and status["queue_closed"] and status["closed"]["n_items"] == 3
    # both writes went through the Volume API (batch_upload), read back through listdir/read_file
    assert sorted(fake_vol.uploads) == sorted(["/" + prod.marker_path, "/" + L.closed_rel(spec.ladder_root)])
    assert json.loads((vol_root / L.closed_rel(spec.ladder_root)).read_text())["n_items"] == 3
    # KNOWN-BAD parser forms: reconcile with both or neither of --volume / --vol-root; consume without --volume (no smoke dir); no width
    for argv in (["reconcile", "--volume", "twosi-fits", "--vol-root", str(vol_root), *_axes(spec)],
                 ["reconcile", *_axes(spec)],
                 ["consume", "--vol-root", str(vol_root), *_axes(spec), "--attempt-id", "x", "--call-id", "y", "--proxy-url", "http://h/v1", "--width", "2"],
                 ["consume", "--vol-root", str(vol_root), "--volume", "twosi-fits", *_axes(spec), "--attempt-id", "x", "--call-id", "y",
                  "--proxy-url", "http://h/v1"]):
        with pytest.raises(SystemExit) as ei:
            RR.main(argv)
        assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "exactly one of --volume" in err and "needs --volume" in err and "--width/--concurrency" in err


# --------------------------------------------------------------------------- #
# /simplify follow-ups pinned: duplicate tolerance is rung-only; the lifted lease rule fails closed
# --------------------------------------------------------------------------- #


def test_simplify_two_base_shards_naming_one_item_refuse_even_when_their_entries_agree():
    """accepted_queue_entries tolerates an agreeing duplicate ONLY on a rung-produced queue (two rung shards forwarding
    one elected result, P1-7); base shards are disjoint by construction, so an agreeing duplicate there is a
    partition bug and still refuses (the fail-loud the pre-P1-7 code had, kept)."""
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0", "1"))
    row = T.static_row("g0", 1, seed=7)
    for shard, aid in (("0", "a1"), ("1", "b1")):
        c = T.Container(L, shared, spec, shard, aid)
        c.enqueue_row(row, base_store="shared/base/responses.jsonl")     # the SAME base row, the SAME store, from both shards
        c.complete()
        assert c.finish(kv["accepted"])["state"] == "accepted"
    recs = L.live_queue_entries(shared.read_only_view(), spec)[f"none__s1/{L.item_key('static', 'pasttense', 'g0', 1)}"]
    assert len(recs) == 2 and L.same_world(*recs)                      # agreeing entries …
    with pytest.raises(L.LadderRefusal, match="base shards must be disjoint"):
        L.accepted_queue_entries(shared.read_only_view(), kv["accepted"], spec)   # … still refused on a base-produced queue


def test_simplify_modal_lease_state_is_the_registry_rule_and_fails_closed_on_client_errors(monkeypatch):
    import types

    class NotFoundError(Exception):
        pass

    class AuthError(Exception):
        pass

    class ConnectionError_(Exception):
        pass

    outcomes = {}

    class FC:
        def __init__(self, cid):
            self.cid = cid

        def get(self, timeout=0):
            o = outcomes[self.cid]
            if isinstance(o, Exception):
                raise o
            return o

    def from_id(cid):
        if cid == "missing-at-lookup":
            raise RuntimeError("no such call")
        return FC(cid)

    mod = types.ModuleType("modal")
    mod.FunctionCall = types.SimpleNamespace(from_id=from_id)
    mod.exception = types.SimpleNamespace(NotFoundError=NotFoundError, AuthError=AuthError, ConnectionError=ConnectionError_)
    monkeypatch.setitem(sys.modules, "modal", mod)
    outcomes.update({"done": {"ok": 1}, "live": TimeoutError(), "gone": NotFoundError(), "auth": AuthError(),
                     "net": ConnectionError_(), "raised": ValueError("the call raised")})
    got = {cid: L.modal_lease_state(cid) for cid in [*outcomes, "missing-at-lookup"]}
    assert got == {"done": "finished", "live": "running", "gone": "unknown", "auth": "unknown", "net": "unknown",
                   "raised": "failed", "missing-at-lookup": "unknown"}
    # only finished / failed / cancelled authorise a reclaim: an auth or network error never does
    assert {s_ for s_ in got.values() if s_ in L.TERMINAL_STATES} == {"finished", "failed"}
