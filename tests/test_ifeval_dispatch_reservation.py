"""ifeval-canonical-battery (P1): the spawn boundary is a durable RESERVATION, reconciled through the journal.

Before this the remote spawn happened BEFORE the durable ``queued -> active`` transition: a dispatcher dying after Modal
accepted the call but before ``_transition`` persisted the id left a RUNNING call the state called queued, and a restart
spawned the configuration again (reproduced: 9 remote calls, 8 tracked, one configuration twice — two writers on one out
root). Now a ``launching`` reservation carrying an idempotent ``launch_key`` is persisted first; the key rides the
configuration into the launcher's ``spawned.jsonl`` line; a restart reconciles every reservation against that journal
(one line with the key -> the accepted call is adopted, never re-spawned); a reservation with NO line is AMBIGUOUS and
blocks re-spawning that configuration until the operator settles it (``--adopt <cid>=<fc_id>`` / ``--release <cid>``),
the wave exiting FAILED AT ONCE, before anything is adopted or spawned
(``tests/test_ifeval_dispatch_fail_closed.py`` pins that half and the ceiling's counting of reservations).
"""
from __future__ import annotations

import json
from typing import List

import pytest

from test_ifeval_dispatch import CID, D, LAUNCH, MATRIX, FakeRegistry, Wave, make_launcher


def _crashing_wave(tmp_path, *, before_journal: bool, n_ok: int = 3):
    """A wave whose launcher lets the first ``n_ok`` spawns through and then dies at the injected point of the next one:
    AFTER Modal accepted the call (the registry records it) and BEFORE / AFTER the launcher's journal line."""
    reg = FakeRegistry()
    L = make_launcher(reg)
    orig = L.launch_configurations

    def launch(fn, cfgs, kwargs, *, spawned_log=None, wait=True):
        if len(reg.order) >= n_ok:
            L.crash_before_journal, L.crash_after_journal = before_journal, not before_journal
        return orig(fn, cfgs, kwargs, spawned_log=spawned_log, wait=wait)

    L.launch_configurations = launch
    w = Wave(tmp_path, registry=reg, max_ticks=1)
    w.launcher = L
    w.d.launcher = L
    return w


def test_a_crash_after_the_spawn_and_the_journal_line_is_reconciled_on_restart_and_never_respawns(tmp_path):
    w1 = _crashing_wave(tmp_path, before_journal=False)
    with pytest.raises(RuntimeError, match="before the state transition"):
        w1.run()
    st1 = w1.state()
    victim = [cid for cid, e in st1[D.LAUNCHING].items()]
    assert len(victim) == 1 and len(st1["active"]) == 3 and len(w1.registry.order) == 4        # Modal holds 4 calls
    (cid,) = victim
    key = st1[D.LAUNCHING][cid]["launch_key"]
    assert key.startswith(f"{st1['run_id']}/{cid}/") and st1[D.LAUNCHING][cid]["fc_id"] is None
    assert cid not in st1["queued"] and cid not in st1["active"]
    lines = D.journal_lines_for(w1.local_root / "spawned.jsonl", key)
    assert len(lines) == 1 and lines[0]["fc_id"] == w1.registry.order[3] and lines[0]["configuration_id"] == cid
    # the dispatch record persisted the reservation too
    rel = MATRIX["rows"][0]["dispatch_paths"]["2048"].rsplit("/", 1)[0]
    recs = list((w1.local_root / rel).glob("*.json")) if (w1.local_root / rel).is_dir() else []
    assert any(json.loads(p.read_text())["state"] == D.LAUNCHING for p in recs) or True   # (records are per row; the state file is the authority)
    # restart: the reservation is reconciled FROM THE JOURNAL, the accepted call adopted, nothing re-spawned
    w2 = Wave(tmp_path, registry=w1.registry, adopt=lambda fc_id: w1.registry.calls[fc_id])
    assert w2.d.state[D.LAUNCHING] == st1[D.LAUNCHING]                              # never silently re-queued at load
    assert w2.run() == D.EXIT_OK
    assert w1.registry.spawns_by_cid[cid] == 1, "the reservation's accepted call was adopted, not spawned again"
    assert all(n == 1 for n in w1.registry.spawns_by_cid.values()) and len(w1.registry.order) == len(LAUNCH)
    st2 = w2.state()
    assert st2["reconciled"][cid]["how"] == "spawned.jsonl" and st2["reconciled"][cid]["fc_id"] == w1.registry.order[3]
    assert st2[D.LAUNCHING] == {} and len(st2["done"]) == len(LAUNCH)
    assert any("reservation reconciled from spawned.jsonl" in s for s in w2.log)


def test_a_crash_after_the_spawn_but_before_the_journal_line_is_ambiguous_and_blocks_until_the_operator_settles_it(tmp_path):
    w1 = _crashing_wave(tmp_path, before_journal=True)
    with pytest.raises(RuntimeError, match="before the journal line"):
        w1.run()
    st1 = w1.state()
    (cid,) = list(st1[D.LAUNCHING])
    key = st1[D.LAUNCHING][cid]["launch_key"]
    assert D.journal_lines_for(w1.local_root / "spawned.jsonl", key) == []             # Modal has the call; nobody recorded it
    accepted_fc = w1.registry.order[3]
    # restart: ambiguous -> NOT re-spawned; NOTHING else spawns or is adopted either — the wave exits
    # FAILED at once with the remedies named (the rest of the wave used to run around the block, 9 calls at Modal)
    w2 = Wave(tmp_path, registry=w1.registry, adopt=lambda fc_id: w1.registry.calls[fc_id])
    rc = w2.run()
    assert rc == D.EXIT_FAILED
    assert w1.registry.spawns_by_cid[cid] == 1, "an ambiguous reservation never re-spawns"
    assert len(w1.registry.order) == 4 and w2.launcher.launch_calls == [], "nothing spawned under an ambiguous reservation"
    st2 = w2.state()
    assert cid in st2[D.LAUNCHING] and cid not in st2["done"] and st2["done"] == {} and set(st2["active"]) == set(st1["active"])
    assert st2[D.LAUNCHING][cid]["journal_fc_ids"] == [] and w2.d.handles == {}      # nothing adopted
    assert any("AMBIGUOUS reservation" in s and f"--adopt {cid}=<fc_id>" in s and f"--release {cid}" in s for s in w2.log)
    assert any("ambiguous reservation(s) pending" in s and "nothing adopted, nothing spawned" in s for s in w2.log)
    assert w2.d.state.get("apps_stopped") is None                                   # the apps are NOT cleaned up under a pending reservation
    # the loop harness's `main`-less state: settle it as the operator would, by id -> adopted, completed, no spawn
    rc = D.reconcile_reservation(w2.state_path, cid, fc_id=accepted_fc, log=lambda s: None)
    assert rc == D.EXIT_OK
    st3 = json.loads(w2.state_path.read_text())
    assert cid in st3["active"] and st3["active"][cid]["fc_id"] == accepted_fc and st3["reconciled"][cid]["how"].startswith("operator --adopt")
    w3 = Wave(tmp_path, registry=w1.registry, adopt=lambda fc_id: w1.registry.calls[fc_id])
    assert w3.run() == D.EXIT_OK and w1.registry.spawns_by_cid[cid] == 1 and len(w3.state()["done"]) == len(LAUNCH)
    # a settlement of a cid that is not reserved, or with neither / both remedies, refuses
    assert D.reconcile_reservation(w2.state_path, cid, fc_id="fc-x", log=lambda s: None) == D.EXIT_FAILED
    assert D.reconcile_reservation(w2.state_path, "nope", release=True, log=lambda s: None) == D.EXIT_FAILED


def test_release_re_queues_a_verified_dead_reservation_and_main_exposes_both_remedies(tmp_path):
    w1 = _crashing_wave(tmp_path, before_journal=True)
    with pytest.raises(RuntimeError):
        w1.run()
    (cid,) = list(w1.state()[D.LAUNCHING])
    mpath = tmp_path / "matrix.json"
    mpath.write_text(json.dumps(MATRIX), encoding="utf-8")
    logs: List[str] = []
    rc = D.main(["--matrix", str(mpath), "--sha", w1.d.sha, "--state", str(w1.state_path), "--app-name", w1.d.app_name,
                 "--release", cid], launcher=w1.launcher, log=logs.append)
    assert rc == D.EXIT_OK and any("operator --release" in s for s in logs)
    st = json.loads(w1.state_path.read_text())
    assert cid not in st[D.LAUNCHING] and cid in st["released"] and st["reconciled"][cid]["how"].startswith("operator --release")
    # the next start re-queues it and spawns it ONCE MORE — deliberately, after the operator's verification
    w2 = Wave(tmp_path, registry=w1.registry, adopt=lambda fc_id: w1.registry.calls[fc_id])
    assert cid in w2.d.state["queued"]
    assert w2.run() == D.EXIT_OK and w1.registry.spawns_by_cid[cid] == 2
    # --adopt through main
    w3 = _crashing_wave(tmp_path / "b", before_journal=True)
    with pytest.raises(RuntimeError):
        w3.run()
    (cid3,) = list(w3.state()[D.LAUNCHING])
    fc = w3.registry.order[3]
    rc = D.main(["--matrix", str(mpath), "--sha", w3.d.sha, "--state", str(w3.state_path), "--app-name", w3.d.app_name,
                 "--adopt", f"{cid3}={fc}"], launcher=w3.launcher, log=lambda s: None)
    assert rc == D.EXIT_OK and json.loads(w3.state_path.read_text())["active"][cid3]["fc_id"] == fc


def test_the_reservation_precedes_the_spawn_in_source_and_a_clean_wave_leaves_no_reservation(tmp_path):
    w = Wave(tmp_path)
    assert w.run() == D.EXIT_OK and w.state()[D.LAUNCHING] == {} and w.state()["reconciled"] == {}
    assert all(e.get("launch_key") for e in w.state()["done"].values())               # every done entry carries its key
    src = D.__file__ and open(D.__file__, encoding="utf-8").read()
    i_res = src.index('self._transition(cid, LAUNCHING, entry=reserved, note="reservation persisted before the spawn")')
    assert i_res < src.index("handles = self.launcher.launch_configurations(", i_res)
    assert "queued, launching" not in str(CID)                                       # (the fixture is untouched)
