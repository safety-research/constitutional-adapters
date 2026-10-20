"""The reconciler's DEFAULT producer-root resolution spelled the shard bare
(`<view>/0/attempts/`) while the Modal app writes `<view>/shard<NN>/attempts/` (and `rungs/<cap>/` for a rung attempt), so from the
`reconcile` CLI's default no queue would ever close. ONE encoder now (`seedrep_ladder.attempt_root_for`), pinned equal to
`lp.attempt_root_for` / `lp.producer_attempt_root_of` read by path; a root found under the bare spelling is REFUSED by name, two
spellings for one shard refuse, a `shard00` root closes. Synthetic W7 / uuid throughout."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seedrep_testlib as T  # noqa: E402

L = T.load_ladder()
RR = T.load_rung_redraw()


def test_the_one_encoder_spells_the_apps_roots_and_refuses_a_label_the_app_cannot_spell():
    root = f"seedrep/{T.W7}/exp/seedrep_battery_modal"
    assert L.attempt_root_for(T.W7, leg="pasttense", view="s1", stage="base", shard="0", attempt_id="a1") == f"{root}/pasttense/s1/shard00/attempts/a1"
    assert L.attempt_root_for(T.W7, leg="pasttense", view="s1", stage="8192", shard="1", attempt_id="r") == f"{root}/pasttense/s1/rungs/8192/shard01/attempts/r"
    assert L.attempt_root_for(T.W7, leg="controls", view="arditi", stage=2048, shard=12, attempt_id="x") == f"{root}/controls/arditi/rungs/2048/shard12/attempts/x"
    assert L.attempt_root_rel(T.W7, "pasttense", "s1", "0", "a1") == L.attempt_root_for(T.W7, leg="pasttense", view="s1", stage="base", shard="0", attempt_id="a1")
    assert L.shard_dir("7") == L.shard_dir(7) == "shard07" and L.shard_dir("00") == "shard00"
    for bad in ("r0", "shard00", "", "-1", True):                       # KNOWN-BAD: no root exists for a label the app cannot spell
        with pytest.raises(L.LadderRefusal, match="not numeric"):
            L.shard_dir(bad)
    with pytest.raises(L.LadderRefusal, match="not numeric"):
        L.attempt_root_rel(T.W7, "pasttense", "s1", "r0", "a1")
    # a declared producer's root follows ITS key's stage: base producers of the 8192 queue, the 8192 rung's shards for the 16384 queue
    q8 = T.static_spec(L, shards=("0", "1"))
    q16 = q8.next_spec(["0", "1"])
    assert L.producer_attempt_root(q8.producers[1], "a1") == f"{root}/pasttense/s1/shard01/attempts/a1"
    assert L.producer_attempt_root(q16.producers[0], "rung-1") == f"{root}/pasttense/s1/rungs/8192/shard00/attempts/rung-1"
    # the pull rule spells the same
    kv = T.kvs(L)
    assert L.accept_attempt(kv["accepted"], q16.producers[1].key, "rung-b")
    assert L.accepted_attempt_root_or_refuse(kv["accepted"], q16.producers[1].key) == f"{root}/pasttense/s1/rungs/8192/shard01/attempts/rung-b"


def test_the_encoder_equals_ws2ds_function_read_by_path():
    lp = T.load_ws2d_launch_plan()
    assert lp.RESULTS_ROOT_NAME == L.RESULTS_ROOT_NAME == "seedrep_battery_modal" and lp.STAGE_BASE == L.STAGE_BASE == "base"
    results_root = f"{lp.VOL_PATH}/{L.results_root_rel(T.W7)}"
    for stage in ("base", "2048", "8192", "16384"):
        for shard in (0, 1, 12):
            for leg, view in (("pasttense", "s1"), ("controls", "arditi"), ("misalign", "s3")):
                theirs = lp.attempt_root_for(results_root, leg=leg, view=view, stage=stage, shard=shard, attempt_id="ta-1",
                                             kind=("base" if stage == "base" else "rung"))
                assert theirs.startswith(lp.VOL_PATH + "/")
                assert theirs[len(lp.VOL_PATH) + 1:] == L.attempt_root_for(T.W7, leg=leg, view=view, stage=stage, shard=str(shard), attempt_id="ta-1")
    # and their reconciler injection over OUR ProducerRefs equals our default (so the CLI's default now closes what the app writes)
    theirs_of = lp.producer_attempt_root_of(results_root)
    for spec in (T.static_spec(L, shards=("0", "1")), T.static_spec(L, shards=("0",)).next_spec(["0", "1"]),
                 L.QueueSpec.build(T.W7, T.WAVE, "controls", "s2", "controls", 2048, ["0"]).next_spec(["0"])):
        for p in spec.producers:
            assert theirs_of(p, "att-x") == L.producer_attempt_root(p, "att-x"), (spec.kind, spec.cap_next, p.shard)


def _bare_producer(shared, spec, shard, aid, kv):
    """A producer whose attempt evidence sits at the PRE-FIX bare-label root (what the old default resolved; the app never writes it)."""
    ref = next(p for p in spec.producers if p.shard == shard)
    vol = shared.snapshot(aid)
    root = L._bare_attempt_root_rel(spec.W7, spec.leg, spec.view, shard, aid)
    prod = L.Producer(vol, spec, ref, aid, root)
    for row in T.rows3():
        prod.enqueue(L.static_queue_record(spec, arm="none", s=1, family="pasttense", row=row, base_store="b", producer=ref,
                                           producer_attempt_id=aid, vmodel="petri__none", sampling=T.PRESET))
    vol.write_bytes(L.complete_rel(root), b"{}")
    assert L.finish_producer(prod, kv["accepted"])["state"] == "accepted"
    return vol, root


def test_the_reconciler_closes_from_a_shard00_root_and_refuses_the_bare_spelling_or_two_spellings():
    # the app's spelling: a producer at <view>/shard00/attempts/ closes the queue through the DEFAULT resolution (the CLI's path)
    shared, kv = L.SharedVolume(), T.kvs(L)
    spec = T.static_spec(L, shards=("0",))
    c = T.Container(L, shared, spec, "0", "a1")
    assert "/pasttense/s1/shard00/attempts/a1" in c.root
    for row in T.rows3():
        c.enqueue_row(row)
    c.complete()
    assert c.finish(kv["accepted"])["state"] == "accepted"
    st = L.reconcile_queue(shared.read_only_view(), kv["accepted"], spec)
    assert st["queue_closed"] and st["producers"]["0"]["state"] == "CLOSED" and st["closed"]["n_items"] == 3
    # KNOWN-BAD 1: the evidence ONLY at the bare root -> refused BY NAME (never silently unverified, never closed)
    shared2, kv2 = L.SharedVolume(), T.kvs(L)
    vol2, bare_root = _bare_producer(shared2, spec, "0", "a1", kv2)
    assert "/pasttense/s1/0/attempts/a1" in bare_root
    with pytest.raises(L.LadderRefusal, match="only under .*/pasttense/s1/0/attempts/a1 — a bare shard-label root the Modal app never writes"):
        L.reconcile_queue(shared2.read_only_view(), kv2["accepted"], spec)
    # an injected resolver is used verbatim: pointing it at the bare root closes the queue (the operator's explicit choice)
    st2 = L.reconcile_queue(shared2.read_only_view(), kv2["accepted"], spec,
                            attempt_root_of=lambda p, aid: L._bare_attempt_root_rel(spec.W7, spec.leg, spec.view, p.shard, aid))
    assert st2["queue_closed"] and st2["closed"]["n_items"] == 3
    # KNOWN-BAD 2: evidence under BOTH spellings for one shard -> the divergence is named, neither resolved
    shared3, kv3 = L.SharedVolume(), T.kvs(L)
    vol3, bare3 = _bare_producer(shared3, spec, "0", "a1", kv3)
    canon = L.producer_attempt_root(spec.producers[0], "a1")
    vol3.write_bytes(L.complete_rel(canon), b"{}")
    vol3.commit()
    with pytest.raises(L.LadderRefusal, match="TWO spellings"):
        L.reconcile_queue(shared3.read_only_view(), kv3["accepted"], spec)
    # a producer with NO evidence anywhere stays ACCEPTED_UNVERIFIED at the app's root (the missing files are named there)
    shared4, kv4 = L.SharedVolume(), T.kvs(L)
    assert L.accept_attempt(kv4["accepted"], spec.producers[0].key, "ghost")
    st4 = L.reconcile_queue(shared4.read_only_view(), kv4["accepted"], spec)
    assert st4["producers"]["0"]["state"] == "ACCEPTED_UNVERIFIED" and any("/shard00/attempts/ghost/COMPLETE.json missing" in r for r in st4["producers"]["0"]["reasons"])


def test_the_consumer_cli_defaults_its_own_and_its_producers_roots_to_the_apps_spelling(tmp_path, monkeypatch, capsys):
    """`consume --rung-shard 0` without `--attempt-root`: the next queue's inventory lands at `rungs/<this cap>/shard00/attempts/<id>`."""
    import types
    spec = T.static_spec(L, shards=("0",))
    vol_root = tmp_path / "vol"
    vol = L.LocalVolume(vol_root)
    kv = T.kvs(L)
    T.close_queue(L, vol, spec, [L.static_queue_record(spec, arm="none", s=1, family="pasttense", row=T.static_row("g0", 1, seed=7), base_store="b",
                                                        producer=spec.producers[0], producer_attempt_id="a1", vmodel="petri__none", sampling=T.PRESET)],
                  kv["accepted"])
    kvdir = tmp_path / "kv"
    kvdir.mkdir()
    for purpose in ("accepted", "claims", "results"):
        (kvdir / f"{L.dict_name(T.W7, T.WAVE, purpose)}.json").write_text(json.dumps(dict(kv[purpose].items())))
    proxy = T.StubProxy()
    try:
        rc = RR.main(["consume", "--vol-root", str(vol_root), "--memory-kv-dir", str(kvdir), "--W7", spec.W7, "--wave-uuid", spec.wave_uuid, "--leg", spec.leg,
                      "--view", spec.view, "--kind", spec.kind, "--cap-next", str(spec.cap_next), "--producers", "0", "--attempt-id", "rung-1", "--call-id", "fc",
                      "--proxy-url", proxy.url, "--width", "2", "--rung-shard", "0", "--rung-shards", "0", "--finish-producer"])
    finally:
        proxy.close()
    assert rc == 0
    expected = L.attempt_root_for(T.W7, leg="pasttense", view="s1", stage="8192", shard="0", attempt_id="rung-1")
    assert "/pasttense/s1/rungs/8192/shard00/attempts/rung-1" in expected and (vol_root / L.inventory_rel(expected)).is_file()
    assert json.loads((vol_root / L.inventory_rel(expected)).read_text())["shard"] == "0"
    capsys.readouterr()
