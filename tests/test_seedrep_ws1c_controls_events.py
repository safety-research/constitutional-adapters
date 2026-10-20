"""The sampled controls' durable per-draw stream (``sisteer.eval.controls_events``) and the
controls ladder's ancestry + floor rule, end to end through the real driver on the offline
controls world.

Unit: the writer's schema and fsync append, seq continuation across a resume, a torn tail
truncated and a corrupt earlier line refused, a sink bound to the wrong cap refused, the strict
reader, latest-per-item and at-cap selection; ``ladder_transitions`` / ``predecessor_cap`` /
``check_escalated_from`` on the real ladder (512 -> 2048 -> 8192 -> 16384) and on a synthetic one,
with the known-bad inputs (the base cap, an off-ladder cap, a rung fed by an EARLIER rung).

End to end (the ``ladder`` fixture of tests/test_sampled_controls.py: a base store at 64 and rungs
at 128 / 256 against a canned poster): ``--controls-events`` writes one event per DRAWN item in
completion order with exactly the contract's fields, the store's own files are byte-identical to
a run without the flag, a rung's events carry its immediate predecessor; the ancestry chain
64 -> 128 -> 256 holds on every rung store; the floor sample of the 256 rung is computed from the
COMPLETED 128 rung's below-cap population (not the 64 base's); a rung fed by the base two rungs
down is refused; a predecessor with no completion stamp (no gen_stats.json) is refused; the flag
on a greedy run is refused before any store exists."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22" / "sampled"))

import served_capability_run as srun  # noqa: E402
from sisteer.eval import controls_events as cev  # noqa: E402
from sisteer.eval import sampled_controls as scp  # noqa: E402
from test_sampled_controls import (  # noqa: E402,F401  (the offline controls world + the ladder)
    AT_CAP_64, ITEM_OF_PROMPT, META, ORB2, PRESET, SEED_BASE, SPOOL, TT_CELL, XST4,
    _cell_dir, _meta_rows, _rung_argv, _snapshot, _spool_lines, ladder, sampled, world,
)

EV = cev.CONTROLS_EVENT_FIELDS
ORB2_IDS = [r["item_id"] for r in ORB2]


def _resp(text, fr="stop", ct=5, cid=None):
    r = {"choices": [{"message": {"content": text}, "finish_reason": fr}],
         "usage": {"prompt_tokens": 2, "completion_tokens": ct}}
    if cid:
        r["id"] = cid
    return r


# --------------------------------------------------------------------------- the writer / sink / reader
def test_writer_appends_exactly_the_contract_fsynced_and_continues_seq_across_a_resume(tmp_path, monkeypatch):
    fsyncs = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (fsyncs.append(fd), real_fsync(fd)))
    path = tmp_path / cev.CONTROLS_EVENTS_FILENAME
    w = cev.ControlsEventWriter(path, cell_id="si_safety__psr_kl_all")
    sink = w.sink(control="xstest", cap=512)
    ev0 = sink.publish("xst0000", {"max_tokens": 512, "seed": 4242}, _resp("hello", cid="cmpl-9"))
    assert tuple(ev0) == EV and ev0["v"] == 1 and ev0["kind"] == "draw" and ev0["seq"] == 0
    assert ev0["cell_id"] == "si_safety__psr_kl_all" and ev0["control"] == "xstest" and ev0["cap"] == 512
    assert ev0["escalated_from_cap"] is None and ev0["item_id"] == "xst0000" and ev0["sampling_seed"] == 4242
    assert ev0["finish_reason"] == "stop" and ev0["at_cap"] is False and ev0["completion_tokens"] == 5
    assert ev0["prompt_tokens"] == 2 and ev0["text_sha256"] == scp._text_sha("hello")
    assert ev0["request_id"] is None and ev0["completion_id"] == "cmpl-9" and ev0["error"] is None
    assert ev0["ts_utc"].endswith("Z")
    assert len(fsyncs) == 1                                          # one fsync per event
    ev1 = sink.publish("xst0001", {"max_tokens": 512, "seed": 7}, _resp("runs on", fr="length", ct=512))
    assert ev1["seq"] == 1 and ev1["at_cap"] is True and len(fsyncs) == 2
    lines = path.read_text().splitlines()
    assert [json.loads(l) for l in lines] == [ev0, ev1] and path.read_text().endswith("\n")
    # a resume continues the numbering after the last complete event
    w2 = cev.ControlsEventWriter(path, cell_id="si_safety__psr_kl_all")
    ev2 = w2.sink(control="xstest", cap=512).publish("xst0002", {"max_tokens": 512, "seed": 1}, _resp("z"))
    assert ev2["seq"] == 2 and [e["seq"] for e in cev.read_controls_events(path)] == [0, 1, 2]
    # the request-evidence hook point: the id the client minted lands on the event
    ev3 = w2.sink(control="xstest", cap=512, request_id_of=lambda payload, outcome: f"req-{payload['seed']}").publish(
        "xst0003", {"max_tokens": 512, "seed": 99}, _resp("q"))
    assert ev3["request_id"] == "req-99"
    # a rung's sink stamps its immediate predecessor
    ev4 = w2.sink(control="xstest", cap=2048, escalated_from_cap=512).publish(
        "xst0001", {"max_tokens": 2048, "seed": 7}, _resp("done now"))
    assert ev4["escalated_from_cap"] == 512 and ev4["cap"] == 2048
    latest = cev.latest_by_item(cev.read_controls_events(path))
    assert latest[("xstest", "xst0001")] == ev4                      # the later event supersedes
    assert cev.at_cap_items(cev.read_controls_events(path)) == []    # xst0001 left the cap at 2048


def test_writer_refusals_torn_tail_corrupt_line_wrong_cap_bad_fields(tmp_path):
    path = tmp_path / cev.CONTROLS_EVENTS_FILENAME
    w = cev.ControlsEventWriter(path, cell_id="none")
    sink = w.sink(control="or_bench_hard_1k", cap=512)
    sink.publish("orb0", {"max_tokens": 512, "seed": 1}, _resp("a"))
    # a payload at another cap: the sink is bound to the wrong control -> refused, nothing written
    with pytest.raises(cev.ControlsEventsError, match="bound to or_bench_hard_1k @ cap 512"):
        sink.publish("orb1", {"max_tokens": 2048, "seed": 2}, _resp("b"))
    with pytest.raises(cev.ControlsEventsError, match="no request payload"):
        sink.publish("orb1", None, _resp("b"))
    assert len(cev.read_controls_events(path)) == 1
    # a malformed response is an error event, not a crash
    ev = sink.publish("orb2", {"max_tokens": 512, "seed": 3}, {"choices": []})
    assert ev["error"].startswith("malformed response") and ev["at_cap"] is None
    # a torn final line (crash mid-append) is truncated away by the next writer; the reader refuses it
    good = path.read_bytes()
    path.write_bytes(good + b'{"v": 1, "kind": "draw", "seq": 2, "ts_u')
    with pytest.raises(cev.ControlsEventsError, match="corrupt or torn event at line 3"):
        cev.read_controls_events(path)
    w2 = cev.ControlsEventWriter(path, cell_id="none")
    assert path.read_bytes() == good and w2._seq == 2
    # a corrupt EARLIER line is file damage: both the writer and the reader refuse
    lines = good.decode().splitlines()
    path.write_text(lines[0][:-3] + "\n" + lines[1] + "\n")
    with pytest.raises(cev.ControlsEventsError, match="corrupt event at line 1 \\(not a torn tail\\)"):
        cev.ControlsEventWriter(path, cell_id="none")
    with pytest.raises(cev.ControlsEventsError, match="corrupt or torn event at line 1"):
        cev.read_controls_events(path)
    # the reader's field / version / seq gates
    path.write_bytes(good)
    rows = [json.loads(l) for l in good.decode().splitlines()]
    bad = dict(rows[1]); bad["seq"] = 0
    path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(bad) + "\n")
    with pytest.raises(cev.ControlsEventsError, match="seq 0 does not increase past 0"):
        cev.read_controls_events(path)
    bad = dict(rows[0]); bad["v"] = 2
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(cev.ControlsEventsError, match="v=2"):
        cev.read_controls_events(path)
    bad = dict(rows[0]); bad.pop("text_sha256")
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(cev.ControlsEventsError, match="does not carry exactly"):
        cev.read_controls_events(path)
    with pytest.raises(cev.ControlsEventsError, match="no controls events stream"):
        cev.read_controls_events(tmp_path / "absent.jsonl")
    # the writer's own field check
    with pytest.raises(cev.ControlsEventsError, match="event fields"):
        cev.ControlsEventWriter(tmp_path / "x.jsonl", cell_id="none").append({"item_id": "only"})


# --------------------------------------------------------------------------- the ancestry rule
def test_ladder_transitions_and_predecessor_on_the_real_ladder_and_the_known_bad_caps():
    assert scp.CONTROLS_CAP_LADDER == (512, 2048, 8192, 16384)
    assert cev.ladder_transitions() == ((512, 2048), (2048, 8192), (8192, 16384))
    assert [cev.predecessor_cap(c) for c in (2048, 8192, 16384)] == [512, 2048, 8192]
    with pytest.raises(cev.ControlsEventsError, match="base .* has no source rung"):
        cev.predecessor_cap(512)
    with pytest.raises(cev.ControlsEventsError, match="not a rung of the ladder"):
        cev.predecessor_cap(4096)
    with pytest.raises(cev.ControlsEventsError, match="strictly increasing"):
        cev.ladder_transitions((512, 512, 2048))
    with pytest.raises(cev.ControlsEventsError, match="at least two caps"):
        cev.ladder_transitions((512,))
    # check_escalated_from: a rung fed by its immediate predecessor passes; every other shape refuses
    ok = {"max_new_tokens": 8192, "escalated_from": {"cap": 2048, "root": "/r/2048"}}
    assert cev.check_escalated_from(ok, cap=8192) == {"cap": 2048, "root": "/r/2048"}
    assert cev.check_escalated_from({"max_new_tokens": 512}, cap=512) == {}
    with pytest.raises(cev.ControlsEventsError, match="immediate predecessor .* is 2048"):
        cev.check_escalated_from({"max_new_tokens": 8192, "escalated_from": {"cap": 512, "root": "/r"}}, cap=8192)
    with pytest.raises(cev.ControlsEventsError, match="immediate predecessor .* is 8192"):
        cev.check_escalated_from({"max_new_tokens": 16384, "escalated_from": {"cap": 2048, "root": "/r"}}, cap=16384)
    with pytest.raises(cev.ControlsEventsError, match="no escalated_from.cap"):
        cev.check_escalated_from({"max_new_tokens": 2048}, cap=2048)
    with pytest.raises(cev.ControlsEventsError, match="a base has no source"):
        cev.check_escalated_from({"max_new_tokens": 512, "escalated_from": {"cap": 512}}, cap=512)
    with pytest.raises(cev.ControlsEventsError, match="store's cap is 2048, not the rung 8192"):
        cev.check_escalated_from({"max_new_tokens": 2048, "escalated_from": {"cap": 512}}, cap=8192)


# --------------------------------------------------------------------------- end to end, real driver
def _events(cell_dir: Path):
    return cev.read_controls_events(cell_dir / cev.CONTROLS_EVENTS_FILENAME)


def _gen(cell_dir: Path) -> dict:
    return json.loads((cell_dir / "manifest.json").read_text())["resolved_config"]["gen"]


def test_base_run_publishes_one_event_per_draw_in_completion_order_and_leaves_the_store_bytes_alone(sampled, tmp_path):
    # the same base run WITHOUT the flag, for the byte comparison
    plain_root = tmp_path / "plain_cap64"
    srun.main(sampled.argv(plain_root))
    plain = _snapshot(plain_root)
    del sampled.posted[:]
    srun.main(sampled.argv(sampled.base_root, cev.CONTROLS_EVENTS_FLAG))
    cell_dir = _cell_dir(sampled.base_root)
    got = _snapshot(sampled.base_root)
    rel = "cells/" + cell_dir.name + "/"
    assert set(got) == set(plain) | {rel + cev.CONTROLS_EVENTS_FILENAME}
    # evidence.json (root + cell: written_at) and gen_stats.json (per-invocation timings + the new key) are volatile;
    # cap_ladder.json / the manifest name their own root, so the plain root's path is rewritten to this one's
    def volatile(k):
        return k.endswith("evidence.json") or k.endswith("gen_stats.json") or k.endswith(cev.CONTROLS_EVENTS_FILENAME)
    def rerooted(v):
        return v.replace(str(plain_root).encode(), str(sampled.base_root).encode())
    assert {k: v for k, v in got.items() if not volatile(k)} == {k: rerooted(v) for k, v in plain.items() if not volatile(k)}
    assert sum(1 for k in got if k.endswith("gen_stats.json")) == 1 and sum(1 for k in got if k.endswith("evidence.json")) == 2
    gs = json.loads((cell_dir / "gen_stats.json").read_text())
    gs_plain = json.loads((_cell_dir(plain_root) / "gen_stats.json").read_text())
    for control in ("xstest", "or_bench_hard_1k"):
        assert gs[control].pop(cev.GEN_STATS_EVENTS_KEY) == str(cell_dir / cev.CONTROLS_EVENTS_FILENAME)
        assert set(gs[control]) == set(gs_plain[control])
    assert cev.GEN_STATS_EVENTS_KEY not in json.dumps(gs_plain)      # the flagless store carries no such key
    events = _events(cell_dir)
    assert len(events) == len(XST4) + len(ORB2) == len(sampled.posted)
    assert [tuple(e) for e in events] == [EV] * len(events)
    assert [e["seq"] for e in events] == list(range(len(events)))
    # every drawn item has its event with the seed AS SENT, the cap, the base marker, the text's sha
    posted = {ITEM_OF_PROMPT[p["messages"][-1]["content"]]: p for p in sampled.posted}
    texts = {c: {l["item_id"]: l["text"] for l in _spool_lines(cell_dir, SPOOL[c])[1:]} for c in SPOOL}
    meta = {c: {m["item_id"]: m for m in _meta_rows(cell_dir, META[c])} for c in SPOOL}
    for e in events:
        c = e["control"]
        assert e["cell_id"] == TT_CELL == "safety_refusal__psr_kl_all"   # the TRUE cell id (the dir name carries the token)
        assert e["cap"] == 64 and e["escalated_from_cap"] is None and e["error"] is None
        assert e["sampling_seed"] == posted[e["item_id"]]["seed"] == scp.sampling_seed(SEED_BASE, 0, c, e["item_id"])
        assert e["text_sha256"] == scp._text_sha(texts[c][e["item_id"]]) == meta[c][e["item_id"]]["text_sha256"]
        assert e["at_cap"] == meta[c][e["item_id"]]["at_cap"] == (e["item_id"] in AT_CAP_64)
        assert e["completion_id"] is None                            # the canned poster sends no id
    assert {e["item_id"] for e in cev.at_cap_items(events)} == AT_CAP_64
    # a resume (everything spooled) publishes nothing new and keeps the file
    before = (cell_dir / cev.CONTROLS_EVENTS_FILENAME).read_bytes()
    srun.main(sampled.argv(sampled.base_root, cev.CONTROLS_EVENTS_FLAG))
    assert (cell_dir / cev.CONTROLS_EVENTS_FILENAME).read_bytes() == before


def test_events_land_in_completion_order_not_submission_order(sampled, monkeypatch):
    """The canned poster returns instantly, so the base test cannot see ordering; here the first
    submitted item is delayed and must be published LAST."""
    import asyncio
    import contextlib
    real = srun._proxy_poster
    slow_item = {"box": None}

    @contextlib.asynccontextmanager
    async def poster(proxy_url):
        async with real(proxy_url) as post:
            async def delayed(req):
                iid = ITEM_OF_PROMPT[req["messages"][-1]["content"]]
                if slow_item["box"] is None:
                    slow_item["box"] = iid                           # the first request seen is held back
                if iid == slow_item["box"]:
                    await asyncio.sleep(0.15)
                return await post(req)
            yield delayed
    monkeypatch.setattr(srun, "_proxy_poster", poster)
    srun.main(sampled.argv(sampled.base_root, cev.CONTROLS_EVENTS_FLAG))
    events = _events(_cell_dir(sampled.base_root))
    first_control_events = [e for e in events if e["control"] == events[0]["control"]]
    assert first_control_events[-1]["item_id"] == slow_item["box"]   # published last, submitted first
    assert first_control_events[0]["item_id"] != slow_item["box"]


def test_the_ancestry_chain_holds_on_every_rung_and_the_floor_comes_from_the_completed_predecessor(ladder):
    l = ladder
    base_gen = _gen(l.base_cell)
    assert cev.check_escalated_from(base_gen, cap=64, ladder=scp.CONTROLS_CAP_LADDER) == {}
    # rung 128 <- 64, with events stamped escalated_from_cap 64 for exactly the re-drawn items
    srun.main(_rung_argv(l, l.rung128, 128, l.base_root, "--floor-sample", "1", cev.CONTROLS_EVENTS_FLAG))
    r128 = _cell_dir(l.rung128)
    g128 = _gen(r128)
    assert cev.check_escalated_from(g128, cap=128, ladder=scp.CONTROLS_CAP_LADDER) == \
        {"cap": 64, "root": str(l.base_root)} == g128["escalated_from"]
    ev128 = _events(r128)
    drawn128 = {(ITEM_OF_PROMPT[p["messages"][-1]["content"]]) for p in l.posted}
    assert {e["item_id"] for e in ev128} == drawn128 and len(ev128) == len(l.posted)
    assert all(e["cap"] == 128 and e["escalated_from_cap"] == 64 for e in ev128)
    # the 128 rung's population as the assembly reads it: at cap = orb0000 only (xst0001 landed below)
    m128 = {c: {m["item_id"]: m for m in _meta_rows(r128, META[c])} for c in SPOOL}
    at_cap_128 = {c: sorted(i for i, m in m128[c].items() if m["at_cap"]) for c in SPOOL}
    assert at_cap_128 == {"xstest": [], "or_bench_hard_1k": ["orb0000-cccccccc"]}
    assert {e["item_id"] for e in cev.at_cap_items(ev128)} == {"orb0000-cccccccc"}
    below_128 = {c: sorted(i for i, m in m128[c].items() if not m["at_cap"]) for c in SPOOL}
    below_64 = {c: sorted(r["item_id"] for r in rows if r["item_id"] not in AT_CAP_64)
                for c, rows in (("xstest", XST4), ("or_bench_hard_1k", ORB2))}
    assert below_128["xstest"] != below_64["xstest"]                 # xst0001 joined the below-cap set at 128
    # rung 256 <- 128: the floor is the COMPLETED 128 rung's below-cap population under the rule — the
    # default k (200) takes ALL of it, so the floor IS that population, and it contains xst0001 (below cap
    # at 128, AT CAP at 64): a floor computed from the base's population could never hold it
    del l.posted[:]
    srun.main(_rung_argv(l, l.rung256, 256, l.rung128, cev.CONTROLS_EVENTS_FLAG))
    r256 = _cell_dir(l.rung256)
    g256 = _gen(r256)
    assert cev.check_escalated_from(g256, cap=256, ladder=scp.CONTROLS_CAP_LADDER) == {"cap": 128, "root": str(l.rung128)}
    with pytest.raises(cev.ControlsEventsError, match="immediate predecessor"):
        cev.check_escalated_from({**g256, "escalated_from": {"cap": 64, "root": str(l.base_root)}},
                                 cap=256, ladder=scp.CONTROLS_CAP_LADDER)
    assert _gen(r256)["floor_sample"]["k"] == scp.FLOOR_SAMPLE_DEFAULT == 200
    m256 = {c: {m["item_id"]: m for m in _meta_rows(r256, META[c])} for c in SPOOL}
    floor_drawn = {c: sorted(i for i, m in m256[c].items() if m["floor_sample"]) for c in SPOOL}
    want_floor = {c: sorted(scp.floor_sample_ids(below_128[c], seed_base=SEED_BASE, control=c, k=200)) for c in SPOOL}
    assert floor_drawn == want_floor == below_128                    # the whole completed-predecessor population
    assert "xst0001-bbbbbbbb" in floor_drawn["xstest"] and "xst0001-bbbbbbbb" not in below_64["xstest"]
    assert floor_drawn["or_bench_hard_1k"] == ["orb0001-eeeeeeee"]  # orb0000 was at cap at 128: re-drawn, not floor
    regen256 = {c: sorted(i for i, m in m256[c].items() if m["regenerated"]) for c in SPOOL}
    assert regen256 == {"xstest": below_128["xstest"], "or_bench_hard_1k": sorted(ORB2_IDS)}
    gs256 = json.loads((r256 / "gen_stats.json").read_text())
    assert gs256["xstest"]["n_floor_sample"] == len(XST4) and gs256["or_bench_hard_1k"]["n_floor_sample"] == 1
    ev256 = _events(r256)
    assert {e["item_id"] for e in ev256} == {ITEM_OF_PROMPT[p["messages"][-1]["content"]] for p in l.posted}
    assert all(e["escalated_from_cap"] == 128 and e["cap"] == 256 for e in ev256)
    assert cev.at_cap_items(ev256) == []                             # orb0000 terminated at 256


def test_known_bad_a_rung_fed_two_rungs_down_and_a_predecessor_that_never_completed_are_refused(ladder):
    l = ladder
    # the 256 rung fed by the 64 base (skipping 128): refused before any store exists
    with pytest.raises(SystemExit, match="is not the rung below 256"):
        srun.main(_rung_argv(l, l.rung256, 256, l.base_root, cev.CONTROLS_EVENTS_FLAG))
    assert not l.rung256.exists() and l.posted == []
    # the 128 rung's population, then its completion stamp removed: the 256 rung's floor cannot be
    # computed from a population that never completed -> refused, nothing written
    srun.main(_rung_argv(l, l.rung128, 128, l.base_root, "--floor-sample", "1"))
    r128 = _cell_dir(l.rung128)
    gs = r128 / "gen_stats.json"
    saved = gs.read_bytes()
    gs.unlink()
    del l.posted[:]
    with pytest.raises(SystemExit, match="no gen_stats.json .* never completed"):
        srun.main(_rung_argv(l, l.rung256, 256, l.rung128, cev.CONTROLS_EVENTS_FLAG))
    assert not l.rung256.exists() and l.posted == []
    # a completion stamp that names no at-cap population for a control is not a completed sampled store
    d = json.loads(saved)
    d["xstest"].pop("n_at_cap")
    gs.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="not a completed sampled store"):
        srun.main(_rung_argv(l, l.rung256, 256, l.rung128))
    gs.write_bytes(saved)
    assert not l.rung256.exists()


def test_controls_events_on_a_greedy_run_is_refused_before_any_store(world):
    argv = [a if a != "gsm8k,xstest,or_bench_hard_1k" else "xstest,or_bench_hard_1k" for a in world.argv]
    argv += ["--greedy", cev.CONTROLS_EVENTS_FLAG]
    with pytest.raises(SystemExit, match="rides a SAMPLED judged-controls run"):
        srun.main(argv)
    assert not world.out_root.exists() or not any(world.out_root.rglob(cev.CONTROLS_EVENTS_FILENAME))


def test_generate_help_names_the_flag(capsys):
    with pytest.raises(SystemExit) as ei:
        srun.main(["generate", "--help"])
    assert ei.value.code == 0
    assert cev.CONTROLS_EVENTS_FLAG in capsys.readouterr().out
