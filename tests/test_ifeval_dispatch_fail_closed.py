"""ifeval-canonical-battery (P1): a restart fails CLOSED under an ambiguous reservation, and every
reservation counts against the concurrency ceiling by the calls its journal names.

Until this change an ambiguous ``launching`` reservation (its ``spawned.jsonl`` key on no line, or on lines naming two
calls) was never re-spawned — but it consumed no slot: ``_fill_slots`` compared ``len(state["active"])`` alone against the
ceiling, so a restart after a crash with 3 active + 1 ambiguous filled FIVE more slots (nine calls outstanding at Modal
against a ceiling of eight, the ambiguous call possibly among them), and the ambiguity failure fired only after the
queue drained. Now: ``run`` exits FAILED immediately after the reconciliation when any reservation is ambiguous — nothing
adopted, nothing spawned, the remedies named — and ``main`` asks the same read-only question BEFORE the preflights and
before entering the detached app; ``outstanding()`` (what ``_fill_slots`` holds under the ceiling) is the active entries
plus every reservation weighted by the DISTINCT call ids its journal lines carry, at least one, so a duplicated journal
line is one call and two ids are two calls, never "one reservation = one call".
"""
from __future__ import annotations

import json
from typing import List

import pytest

from test_ifeval_dispatch import CID, D, LAUNCH, MATRIX, FakeRegistry, Wave, make_launcher
from test_ifeval_dispatch_reservation import _crashing_wave


def _append_journal(w, cid: str, fc_id: str) -> None:
    """A hand-written journal line for the crashed wave's reservation of ``cid`` (what the launcher would have fsync'd)."""
    key = w.state()[D.LAUNCHING][cid]["launch_key"]
    with open(w.local_root / "spawned.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"name": "n", "model": "m", "cell": "c", "fc_id": fc_id, "t": 1.0, "configuration_id": cid,
                             "launch_key": key}) + "\n")


def test_a_restart_with_an_ambiguous_reservation_spawns_nothing_and_exits_failed_before_adopting(tmp_path):
    w1 = _crashing_wave(tmp_path, before_journal=True)
    with pytest.raises(RuntimeError, match="before the journal line"):
        w1.run()
    st1 = w1.state()
    (cid,) = list(st1[D.LAUNCHING])
    assert len(st1["active"]) == 3 and len(w1.registry.order) == 4 and st1["queued"]     # 4 calls at Modal, 26 still queued
    # the restart: reconciliation finds no line -> FAILED at once; NO spawn, NO adoption, nothing else moves
    w2 = Wave(tmp_path, registry=w1.registry, adopt=lambda fc_id: pytest.fail(f"adopted {fc_id} under an ambiguous reservation"))
    assert w2.d.ambiguous_reservations() == {cid: []}                                   # the read-only question
    rc = w2.run()
    assert rc == D.EXIT_FAILED
    assert len(w1.registry.order) == 4 and w2.launcher.launch_calls == [] and w2.d.handles == {}
    assert w1.registry.outstanding == 4 and w1.registry.peak == 4                         # never more than at the crash
    st2 = w2.state()
    assert st2["queued"] == st1["queued"] and set(st2["active"]) == set(st1["active"]) and st2["done"] == {}
    assert st2[D.LAUNCHING][cid]["journal_fc_ids"] == [] and st2.get("apps_stopped") is None
    assert not any("->" in s and "active" in s for s in w2.log), "no transition happened on the restart"
    assert any("AMBIGUOUS reservation" in s and f"--adopt {cid}=<fc_id>" in s for s in w2.log)
    assert any("nothing adopted, nothing spawned" in s and "exits FAILED" in s for s in w2.log)
    # main: the SAME verdict before any preflight subprocess and before the app context — the preflight is never run
    mpath = tmp_path / "matrix.json"
    mpath.write_text(json.dumps(MATRIX), encoding="utf-8")
    logs: List[str] = []
    rc = D.main(["--matrix", str(mpath), "--sha", w1.d.sha, "--state", str(w1.state_path), "--local-root", str(w1.local_root),
                 "--app-name", w1.d.app_name, "--wave-root", w1.d.wave_root],
                launcher=w1.launcher, run_subprocess=lambda *a, **k: pytest.fail("a preflight ran under an ambiguous reservation"),
                adopt=lambda fc_id: pytest.fail("adopted under an ambiguous reservation"), log=logs.append)
    assert rc == D.EXIT_FAILED and len(w1.registry.order) == 4
    assert any("nothing preflighted, nothing adopted, nothing spawned" in s for s in logs)
    assert any("AMBIGUOUS reservation" in s and f"--release {cid}" in s for s in logs)
    # settled by the operator (the call found), the next start adopts it and completes the wave with ONE spawn per row
    accepted_fc = w1.registry.order[3]
    assert D.reconcile_reservation(w1.state_path, cid, fc_id=accepted_fc, log=lambda s: None) == D.EXIT_OK
    w3 = Wave(tmp_path, registry=w1.registry, adopt=lambda fc_id: w1.registry.calls[fc_id])
    assert w3.d.ambiguous_reservations() == {} and w3.run() == D.EXIT_OK
    assert set(w1.registry.spawns_by_cid.values()) == {1} and len(w3.state()["done"]) == len(LAUNCH)
    assert w1.registry.peak <= 8


def test_a_restart_with_resolved_reservations_never_exceeds_eight_outstanding_at_any_tick(tmp_path):
    # the crash AFTER the journal line: 3 active + 1 reserved-and-journaled = 4 calls at Modal, all resolvable
    w1 = _crashing_wave(tmp_path, before_journal=False)
    with pytest.raises(RuntimeError, match="before the state transition"):
        w1.run()
    assert w1.registry.outstanding == 4
    seen: List[int] = []

    def on_tick(w, st, n):
        seen.append(w.registry.outstanding)
        assert w.registry.outstanding <= 8, f"tick {n}: {w.registry.outstanding} calls outstanding at Modal"
        assert len(st["active"]) + len(st[D.LAUNCHING]) <= 8 and st[D.LAUNCHING] == {}
        assert w.d.outstanding() == len(st["active"])

    w2 = Wave(tmp_path, registry=w1.registry, adopt=lambda fc_id: w1.registry.calls[fc_id], complete_per_tick=1, on_tick=on_tick)
    assert w2.d.ambiguous_reservations() == {}
    assert w2.run() == D.EXIT_OK
    assert w1.registry.peak == 8 and max(seen) == 8 and set(w1.registry.spawns_by_cid.values()) == {1}
    assert len(w1.registry.order) == len(LAUNCH) and len(w2.state()["done"]) == len(LAUNCH)
    # the reconciled entry carries the one call the journal named
    (cid,) = [c for c, e in w2.state()["reconciled"].items() if e["how"] == "spawned.jsonl"]
    assert w2.state()["done"][cid]["journal_fc_ids"] == [w1.registry.order[3]]


def test_reservations_count_against_the_ceiling_by_the_calls_their_journal_names(tmp_path):
    """The counting rule on its own (the fail-closed exit never lets an ambiguous reservation reach `_fill_slots` through
    `run`; this pins the invariant the loop holds regardless): a reservation naming two calls takes two slots, one
    naming none takes one, and the fill stops at 8 - their weight."""
    for weight, ids in ((2, ["fc-x", "fc-y"]), (1, []), (1, ["fc-z"])):
        w = Wave(tmp_path / f"w{weight}{len(ids)}")
        w.d.state[D.LAUNCHING]["synthetic"] = {"launch_key": "k", "journal_fc_ids": ids, "fc_id": None}
        assert w.d.outstanding() == weight
        w.d._fill_slots()
        assert len(w.d.state["active"]) == 8 - weight and w.d.outstanding() == 8 and w.registry.outstanding == 8 - weight
        assert len(w.d.state["queued"]) == len(LAUNCH) - (8 - weight)
    # and the plain wave (no reservation) still peaks at exactly eight
    w = Wave(tmp_path / "plain")
    assert w.run() == D.EXIT_OK and w.registry.peak == 8


def test_duplicate_journal_lines_for_one_call_reconcile_and_two_calls_block(tmp_path):
    # (a) one call written twice: ONE call — reconciled, adopted, the wave completes with one spawn for the row
    w1 = _crashing_wave(tmp_path / "a", before_journal=True)
    with pytest.raises(RuntimeError):
        w1.run()
    (cid,) = list(w1.state()[D.LAUNCHING])
    accepted = w1.registry.order[3]
    _append_journal(w1, cid, accepted)
    _append_journal(w1, cid, accepted)
    key = w1.state()[D.LAUNCHING][cid]["launch_key"]
    assert len(D.journal_lines_for(w1.local_root / "spawned.jsonl", key)) == 2
    assert D.journal_calls_for(w1.local_root / "spawned.jsonl", key) == [accepted]
    w2 = Wave(tmp_path / "a", registry=w1.registry, adopt=lambda fc_id: w1.registry.calls[fc_id])
    assert w2.d.ambiguous_reservations() == {}
    assert w2.run() == D.EXIT_OK and w1.registry.spawns_by_cid[cid] == 1 and set(w1.registry.spawns_by_cid.values()) == {1}
    st = w2.state()
    assert st["reconciled"][cid] == {**st["reconciled"][cid], "fc_id": accepted, "how": "spawned.jsonl"}
    assert st["done"][cid]["fc_id"] == accepted and st["done"][cid]["journal_fc_ids"] == [accepted]
    assert any("2 journal lines, one call" in s for s in w2.log)
    # (b) two DIFFERENT calls for one key: two spawns of one reservation — ambiguous, blocks, weighs two, names both
    w3 = _crashing_wave(tmp_path / "b", before_journal=True)
    with pytest.raises(RuntimeError):
        w3.run()
    (cid3,) = list(w3.state()[D.LAUNCHING])
    first = w3.registry.order[3]
    _append_journal(w3, cid3, first)
    _append_journal(w3, cid3, "fc-second-spawn")
    key3 = w3.state()[D.LAUNCHING][cid3]["launch_key"]
    assert D.journal_calls_for(w3.local_root / "spawned.jsonl", key3) == [first, "fc-second-spawn"]
    w4 = Wave(tmp_path / "b", registry=w3.registry, adopt=lambda fc_id: pytest.fail("adopted under two calls"))
    assert w4.d.ambiguous_reservations() == {cid3: [first, "fc-second-spawn"]}
    assert w4.run() == D.EXIT_FAILED and len(w3.registry.order) == 4 and w4.launcher.launch_calls == []
    st4 = w4.state()
    assert st4[D.LAUNCHING][cid3]["journal_fc_ids"] == [first, "fc-second-spawn"] and w4.d.outstanding() == 3 + 2
    assert any("2 calls carry its launch key" in s and "stop all but one" in s and first in s and "fc-second-spawn" in s
               for s in w4.log)
    # a line without a call id records no call
    assert D.journal_calls_for(w3.local_root / "spawned.jsonl", "no-such-key") == []
    _append_journal(w3, cid3, "")
    assert D.journal_calls_for(w3.local_root / "spawned.jsonl", key3) == [first, "fc-second-spawn"]


def test_source_order_reconcile_then_fail_closed_then_adopt_and_main_asks_before_the_preflights():
    src = open(D.__file__, encoding="utf-8").read()
    run = src[src.index("    def run(self) -> int:"):src.index("    # -- reporting")]
    i_rec = run.index("ambiguous = self._reconcile_reservations()")
    i_fail = run.index("return self._fail_closed(ambiguous)")
    i_adopt = run.index("self._adopt_active()")
    assert i_rec < i_fail < i_adopt < run.index("while True:") < run.index("self._fill_slots()")
    fill = src[src.index("    def _fill_slots(self) -> None:"):src.index("    def _stop(")]
    assert "self.outstanding() < self.max_outstanding" in fill and "len(self.state[\"active\"]) <" not in fill
    main = src[src.index("def main("):]
    assert main.index("ambiguous = probe.ambiguous_reservations()") < main.index("return EXIT_FAILED") \
        < main.index("run_preflights(") < main.index("_app_context(launcher)")
    assert "journal_fc_ids" in D.ENTRY_KEYS
    assert str(CID) and MATRIX and make_launcher and FakeRegistry                      # (fixtures untouched)
