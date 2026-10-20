"""The fixed defects in scripts/modal/seedrep_petri_drive.py -- one test block per defect, each with the
RE-VALIDATED repro as its known-bad twin:

- the schedule counted served rows blind and refused the native engine's REAL enumeration (9 per view, 3 for
  Arditi: ``enumerate_petri_native`` derives ``none__lora0``); now the arms are SELECTED by the rows' own
  fields with their ``--cell-index`` preserved, and every index resolves through petri_run's own resolver.
- local state / out roots carried the window + view only, so a new wave at the same window adopted a prior
  wave's completed stores; now every writable root carries (window, wave_uuid8, attempt_id), a skip needs the
  Petri driver's OWN store reconciliation, and continuing prior stores is an explicit, provenance-recorded
  ``--resume-from`` import.
- the attempt root fell back to ``serving_dir_on_volume.split("/exp/")``, which lands on
  ``/vol/seedrep/<window>``; now ``attempt_root_on_volume`` + ``attempt_id`` are REQUIRED fields of the tunnel
  record and the only source.
- every leg after the first waited up to 900 s for ITS OWN root's bundle (0 / 902 / 1,804 s on the three fresh
  Arditi roots) with the heartbeat armed after; now the first leg per root launches at once, only a same-root
  leg waits, and the heartbeat is armed before the launches.
Plus the landed contract: every leg argv carries ``run ... --target-decode vendor --target-seed-base <its base>``
or is refused before launch. Fixtures: the REAL ``enumerate_petri_native`` (test_seedrep_petri_drive.native_cells_json)
and, for the reconciliation, a store the runner itself wrote (test_petri_store_shape's world). Synthetic throughout."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import json

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "scripts/modal/seedrep_petri_drive.py"
sys.path.insert(0, str(REPO / "scripts"))

import petri_run as prun  # noqa: E402
import test_petri_store_shape as shape  # noqa: E402
from test_petri_store_shape import world  # noqa: E402,F401
from test_seedrep_petri_drive import ARDITI_CELLS, VIEW_CELLS, WAVE, _load, _served, arditi_cells, native_cells_json, view_cells  # noqa: E402,F401

from sisteer.capeval import store as cs  # noqa: E402
from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
from sisteer.capeval.tiers import petri_runner as pr  # noqa: E402

W = "0c1925110000000000000000000000000000abcd"
W7 = "0c19251"


@pytest.fixture(scope="module")
def d():
    return _load()


# --------------------------------------------------------------------------- P1-1: arm SELECTION from the complete enumeration

def test_p1_1_lane_arms_are_selected_from_the_real_native_enumeration_and_every_index_resolves_in_petri_run(d, view_cells, arditi_cells):
    for cells, view, n_rows in ((view_cells, "s1", 9), (arditi_cells, "arditi", 3)):
        vp = d.view_params(view)
        served = _served(cells)
        rows = d.leg_rows(cells)
        # RE-VALIDATED repro: the complete native enumeration serves n_rows (the derived control included) != the lane's arm count --
        # the blind count the old rule refused on ("serves 9 cells, the lane expects 8" / "serves 3 cells, the lane expects 1")
        assert len(rows) == n_rows and len(d.served_cells(cells, served)[0]) == n_rows != vp.expected_cells
        assert dc.NOOP_LORA_CELL_ID in [r["cell_id"] for r in rows] and rows[-1]["cell_class"] == "control"
        runs = d.petri_schedule(cells, served, vp)
        assert len(runs) == vp.expected_runs
        # every scheduled --cell-index resolves, through petri_run's OWN resolver over the frozen record's canonical+controls
        # order (what `run --cell-index i` does), to exactly the run's cell, run key and vmodel
        snapshot = [dc.PetriCell(**c) for c in list(cells["canonical"]) + list(cells["controls"])]
        for run in runs:
            cell = prun._resolve_cell(snapshot, SimpleNamespace(cell_index=run.cell_index, cell_id=None))
            assert (cell.cell_id, cell.petri_run_key, cell.virtual_model_name) == (run.cell_id, run.run_key, run.vmodel)
            argv = d.leg_argv("/py", [], run, Path("/o"))
            assert argv[argv.index("--cell-index") + 1] == str(run.cell_index)
        # the derived no-op control is never scheduled on either proxy, and it is the one row excluded from a view
        assert dc.NOOP_LORA_CELL_ID not in {r.cell_id for r in runs}
    assert [c for _i, c, _w in d.lane_arm_indices(view_cells, d.view_params("s1"))[1]] == [dc.NOOP_LORA_CELL_ID]
    # Arditi: `none` and the control are EXCLUDED (logged), never refused; the arm's index is its enumeration index (1), not 0
    ar = d.petri_schedule(arditi_cells, _served(arditi_cells), d.view_params("arditi"))
    assert {r.cell_index for r in ar} == {1} and {r.cell_id for r in ar} == {d.ARDITI_ARM} and [r.seed_base for r in ar] == [1, 2, 3]
    assert [(i, c) for i, c, _w in d.lane_arm_indices(arditi_cells, d.view_params("arditi"))[1]] == [(0, "none"), (2, dc.NOOP_LORA_CELL_ID)]
    # `--legs` addresses the enumeration's own indices: the control's index is not a lane arm
    with pytest.raises(SystemExit, match="requested legs \\[8\\] are not served lane arms"):
        d.petri_schedule(view_cells, _served(view_cells), d.view_params("s1"), requested=[8])
    with pytest.raises(SystemExit, match="requested legs \\[0\\] are not served lane arms"):
        d.petri_schedule(arditi_cells, _served(arditi_cells), d.view_params("arditi"), requested=[0])
    # the drive's expected counts are the plan's (rev 3: 8 per view = 5 fitted + none + 2 prompted; 1 Arditi cell), unchanged by the fix
    assert (d.EXPECTED_VIEW_ARM_RUNS, d.EXPECTED_ARDITI_CELLS, d.total_auditor_width()) == (8, 1, 84)


# --------------------------------------------------------------------------- the lane profile on every leg argv

def test_every_leg_argv_carries_target_decode_vendor_and_its_literal_seed_base_after_run_or_is_refused(d):
    run = d.LegRun(view="arditi", cell_index=1, cell_id="arditi_allpos", vmodel="petri__arditi_allpos", run_key="rk", seed_base=2, width=4,
                   tag="qwen35_4b_arditi_s2")
    opts = ["--manifest", "M.yaml", "--serving-path", "lora_native"]
    argv = d.leg_argv("/py", opts, run, Path("/out/petri_seedrep_x_qwen35_4b_arditi_s2"))
    assert argv[argv.index("run") + 1:] == ["--out-root", "/out/petri_seedrep_x_qwen35_4b_arditi_s2", "--cell-index", "1",
                                            "--target-decode", "vendor", "--target-seed-base", "2"]
    d.assert_lane_leg_argv(argv, run)                                        # the drive's own argv passes
    for s in (1, 2, 3):                                                      # Arditi: the LITERAL base 1 / 2 / 3 per seed-run, distinct roots
        r = d.LegRun(**{**run.__dict__, "seed_base": s, "tag": f"qwen35_4b_arditi_s{s}"})
        a = d.leg_argv("/py", opts, r, Path(f"/out/root_s{s}"))
        assert a[a.index("--target-seed-base") + 1] == str(s) and a[a.index("--out-root") + 1] == f"/out/root_s{s}"
        d.assert_lane_leg_argv(a, r)
    # KNOWN-BAD 1: no --target-decode at all (the leg would audit at HARNESS decode under the lane's name)
    bare = [x for x in argv if x not in ("--target-decode", "vendor")]
    with pytest.raises(SystemExit, match="does not carry `--target-decode vendor` after `run`.*got None"):
        d.assert_lane_leg_argv(bare, run)
    # KNOWN-BAD 2: --target-decode harness
    with pytest.raises(SystemExit, match="got 'harness'"):
        d.assert_lane_leg_argv([("harness" if x == "vendor" else x) for x in argv], run)
    # KNOWN-BAD 3: another leg's seed base on this run's argv; KNOWN-BAD 4: the flags spelled TOP-LEVEL (before `run`); 5: no run; 6: no out root
    with pytest.raises(SystemExit, match="carries `--target-seed-base '3'`, this run's base is 2"):
        d.assert_lane_leg_argv([("3" if (i > 0 and argv[i - 1] == "--target-seed-base") else x) for i, x in enumerate(argv)], run)
    top_level = ["/py", "scripts/petri_run.py", *opts, "--target-decode", "vendor", "--target-seed-base", "2", "run", "--out-root", "/o", "--cell-index", "1"]
    with pytest.raises(SystemExit, match="does not carry `--target-decode vendor` after `run`"):
        d.assert_lane_leg_argv(top_level, run)
    with pytest.raises(SystemExit, match="no `run` subcommand"):
        d.assert_lane_leg_argv(["/py", "scripts/petri_run.py", "print-cells"], run)
    with pytest.raises(SystemExit, match="names no --out-root"):
        d.assert_lane_leg_argv(["/py", "scripts/petri_run.py", "run", "--cell-index", "1", "--target-decode", "vendor", "--target-seed-base", "2"], run)
    # the drive runs the guard on the argv it is about to launch, before the subprocess, and asserts the vendor intent on print-cells too
    body = SRC.read_text().split("def run_leg(")[1]
    assert body.index("assert_lane_leg_argv(argv, run)") < body.index("subprocess.run(argv")
    assert '"print-cells", "--out-root", str(out), "--target-decode", TARGET_DECODE' in SRC.read_text()


# --------------------------------------------------------------------------- P2-3: the attempt root is an EXPLICIT field of the tunnel record

ATTEMPT = "ta-01HXYZ-3f2a1c9e-8b7d-4e6f-a5c4-1d2e3f4a5b6c"
ATTEMPT_ROOT = f"/vol/seedrep/{W7}/exp/seedrep_battery_modal/petri/s1/0/attempts/{ATTEMPT}"
#: the record's own six-axis key -- the Dict key it sits under, minus the name; WAVE is the wave whose sentinels hold it
TUNNEL_KEY = f"{W7}/{WAVE}/petri/s1/base/0"
MINIMUM_TUNNEL = {"url": "https://x.modal.host", "function_call_id": "fc-1", "proxy_port": 8984, "view": "s1", "stage": "base", "control_key": TUNNEL_KEY,
                  "serving_dir_on_volume": f"{ATTEMPT_ROOT}/exp/seedrep_battery_modal/petri/s1/serve",
                  "attempt_root_on_volume": ATTEMPT_ROOT, "attempt_id": ATTEMPT}
ENDPOINT = {"mode": "petri", "engine": "lora_native", "routing_table": "/workspace/seedrep/exp/seedrep_battery_modal/petri/s1/serve/routing_table.json",
            "cells_served": "/workspace/seedrep/exp/seedrep_battery_modal/petri/s1/serve/cells_served.json"}


def test_p2_3_the_minimum_tunnel_record_resolves_the_routing_table_under_its_own_attempt_root(d):
    assert d.tunnel_record_problems(MINIMUM_TUNNEL, view="s1", stage="base", expected_control_key=TUNNEL_KEY, w7=W7) == []
    paths = d.attempt_paths(MINIMUM_TUNNEL, ENDPOINT, "/workspace/seedrep/exp")
    assert paths == {"attempt_root": ATTEMPT_ROOT, "attempt_exp": f"{ATTEMPT_ROOT}/exp",
                     "routing_table": f"{ATTEMPT_ROOT}/exp/seedrep_battery_modal/petri/s1/serve/routing_table.json",
                     "cells_served": f"{ATTEMPT_ROOT}/exp/seedrep_battery_modal/petri/s1/serve/cells_served.json"}
    # RE-VALIDATED repro: the volume layout carries /exp/ TWICE, so the old fallback (split at the FIRST /exp/) landed on
    # /vol/seedrep/<W7> and read a routing table OUTSIDE the attempt; the explicit field never does
    old_root = MINIMUM_TUNNEL["serving_dir_on_volume"].split("/exp/", 1)[0]
    assert old_root == f"/vol/seedrep/{W7}" and old_root != paths["attempt_root"]
    assert paths["routing_table"].startswith(ATTEMPT_ROOT + "/exp/") and not paths["routing_table"].startswith(old_root + "/exp/serve")
    assert d.volume_client_path(paths["routing_table"]) == paths["routing_table"][len("/vol"):]
    # the drive's own resolution goes through attempt_paths; the split fallback is gone from the source
    src = SRC.read_text()
    assert 'split("/exp/"' not in src.split("def attempt_paths")[1].split("def launch_legs")[0].split('"""')[2]  # not in the code body
    assert "paths = attempt_paths(tunnel, ep, exp_root)" in src and 'serving_vol.split("/exp/", 1)[0]' not in src
    # KNOWN-BAD 1: the record without the attempt fields (yesterday's minimum) is refused BY NAME, and attempt_paths refuses to derive
    legacy = {k: v for k, v in MINIMUM_TUNNEL.items() if k not in ("attempt_root_on_volume", "attempt_id")}
    assert d.tunnel_record_problems(legacy, view="s1", stage="base", expected_control_key=TUNNEL_KEY, w7=W7) == ["missing attempt_root_on_volume", "missing attempt_id"]
    with pytest.raises(SystemExit, match="names no attempt_root_on_volume -- refusing to derive"):
        d.attempt_paths(legacy, ENDPOINT, "/workspace/seedrep/exp")
    # KNOWN-BAD 2: a serving dir outside the attempt's exp tree; 3: an attempt id that is not the root's last component; 4: a path-unsafe id;
    # 5: a root outside /vol/seedrep; 6: a trailing slash
    assert d.tunnel_record_problems({**MINIMUM_TUNNEL, "serving_dir_on_volume": f"/vol/seedrep/{W7}/exp/serve"}, view="s1", stage="base", expected_control_key=TUNNEL_KEY, w7=W7) == \
        [f"serving_dir_on_volume '/vol/seedrep/{W7}/exp/serve' is not under the attempt's exp tree {ATTEMPT_ROOT}/exp/"]
    assert d.tunnel_record_problems({**MINIMUM_TUNNEL, "attempt_id": "ta-other"}, view="s1", stage="base", expected_control_key=TUNNEL_KEY, w7=W7) == \
        [f"attempt_id 'ta-other' != the attempt root's last component {ATTEMPT!r}"]
    assert d.tunnel_record_problems({**MINIMUM_TUNNEL, "attempt_id": "../x", "attempt_root_on_volume": f"/vol/seedrep/{W7}/attempts/../x"},
                                    view="s1", stage="base", expected_control_key=TUNNEL_KEY, w7=W7)[-1] == "attempt_id '../x' is not a path-safe attempt id"
    assert d.tunnel_record_problems({**MINIMUM_TUNNEL, "attempt_root_on_volume": f"/vol/arditi/{ATTEMPT}"}, view="s1", stage="base", expected_control_key=TUNNEL_KEY, w7=W7)[0] == \
        f"attempt_root_on_volume '/vol/arditi/{ATTEMPT}' is not under /vol/seedrep/{W7}/"
    assert f"attempt_root_on_volume {ATTEMPT_ROOT + '/'!r} ends with '/'" in d.tunnel_record_problems({**MINIMUM_TUNNEL, "attempt_root_on_volume": ATTEMPT_ROOT + "/"}, view="s1", stage="base", expected_control_key=TUNNEL_KEY, w7=W7)
    # the attempt id as a path component
    assert d.check_attempt_id(ATTEMPT) == ATTEMPT and d.check_attempt_id("ta-1.uuid") == "ta-1.uuid"
    for bad in ("", "a/b", "../x", ".hidden", "x" * 129, None, 7):
        with pytest.raises(SystemExit, match="not a path-safe attempt id"):
            d.check_attempt_id(bad)
    # an endpoint path outside the container's exp root still refuses (the map is the only route onto the volume)
    with pytest.raises(ValueError):
        d.attempt_paths(MINIMUM_TUNNEL, {**ENDPOINT, "routing_table": "/workspace/other/rt.json"}, "/workspace/seedrep/exp")


# --------------------------------------------------------------------------- P2-4: launch order on fresh roots, with a virtual clock

class _Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class _Handle:
    def __init__(self, name, alive=True):
        self.name, self.alive = name, alive

    def is_alive(self):
        return self.alive


def _runs(d, view, n=None):
    vp = d.view_params(view)
    if view == "arditi":
        return [d.LegRun(view=view, cell_index=1, cell_id="arditi_allpos", vmodel="petri__arditi_allpos", run_key="rk", seed_base=s, width=4,
                         tag=d.run_tag(view, s)) for s in vp.seed_bases]
    s = vp.seed_bases[0]
    return [d.LegRun(view=view, cell_index=i, cell_id=f"c{i}", vmodel=f"petri__c{i}", run_key=f"rk{i}", seed_base=s, width=3, tag=d.run_tag(view, s))
            for i in range(n or vp.expected_cells)]


def _launch(d, runs, *, ready_at=None, dead=(), cap=900):
    clock, ready_at, log = _Clock(), dict(ready_at or {}), []
    handles = {}

    def start(run):
        h = _Handle(run.record_name, alive=run.record_name not in dead)
        handles[run.record_name] = h
        return h

    events = d.launch_legs(runs, {r.tag: Path("/roots") / r.tag for r in runs}, start=start,
                           bundle_ready=lambda tag: clock.t >= ready_at.get(tag, float("inf")),
                           now=clock.now, sleep=clock.sleep, log=log.append, wait_cap_s=cap)
    return events, log, clock


def test_p2_4_fresh_roots_launch_immediately_and_only_a_same_root_leg_waits_for_its_root_s_writer(d):
    # the RE-VALIDATED repro: the three Arditi seed-runs live in THREE fresh roots (no bundle anywhere, every writer alive)
    ar = _runs(d, "arditi")
    assert len({r.tag for r in ar}) == 3
    events, log, clock = _launch(d, ar)
    assert [(e["record_name"], e["t"], e["waited_s"], e["waited_on"]) for e in events] == [
        ("leg_1_s1", 0.0, 0.0, None), ("leg_1_s2", 2.0, 0.0, None), ("leg_1_s3", 4.0, 0.0, None)]   # the old loop: 0 / 902 / 1,804 s
    assert clock.t == 6.0 and log == []                                    # nothing waited, nothing logged as waited-on
    # a VIEW: eight legs on ONE root -- the first launches at once, the second waits for THAT root's bundle (its writer is alive),
    # every later leg finds the bundle and launches on the stagger alone
    view = _runs(d, "s2")
    assert len({r.tag for r in view}) == 1
    events, log, clock = _launch(d, view, ready_at={"qwen35_4b_s2": 30.0})
    # (the wait polls every 5 s from t=2, so the bundle ready at 30 is seen at 32)
    assert [(e["record_name"], e["t"], e["waited_s"], e["waited_on"]) for e in events[:3]] == [
        ("leg_0_s2", 0.0, 0.0, None), ("leg_1_s2", 32.0, 30.0, "leg_0_s2"), ("leg_2_s2", 34.0, 0.0, None)]
    assert [e["t"] for e in events[3:]] == [36.0, 38.0, 40.0, 42.0, 44.0] and all(e["waited_s"] == 0.0 for e in events[2:])
    assert len(log) == 1 and log[0].startswith("leg_1_s2: bundle under /roots/qwen35_4b_s2 materialised=True (waited 30 s on leg_0_s2)")
    # KNOWN-BAD twin (bounded): the bundle never appears while the writer stays alive -> the same-root leg waits the CAP, not forever
    events, log, clock = _launch(d, view[:2], cap=900)
    assert (events[1]["t"], events[1]["waited_s"], events[1]["waited_on"]) == (902.0, 900.0, "leg_0_s2")
    assert log[0].startswith("leg_1_s2: bundle under /roots/qwen35_4b_s2 materialised=False (waited 900 s on leg_0_s2)")
    # a root whose first writer DIED without materialising is claimed by the next leg at once, and the third leg waits on THAT one
    events, log, clock = _launch(d, view[:3], dead=("leg_0_s2",), ready_at={"qwen35_4b_s2": 60.0})
    assert [(e["record_name"], e["t"], e["waited_s"], e["waited_on"]) for e in events] == [
        ("leg_0_s2", 0.0, 0.0, None), ("leg_1_s2", 2.0, 0.0, None), ("leg_2_s2", 64.0, 60.0, "leg_1_s2")]
    # mixed: two roots interleaved -- each root's first leg is immediate, each root's second waits on its OWN root's writer only
    s1, s3 = _runs(d, "s1", 2), _runs(d, "s3", 2)
    events, log, clock = _launch(d, [s1[0], s3[0], s1[1], s3[1]], ready_at={"qwen35_4b_s1": 10.0, "qwen35_4b_s3": 100.0})
    assert [(e["record_name"], e["t"], e["waited_on"]) for e in events] == [
        ("leg_0_s1", 0.0, None), ("leg_0_s3", 2.0, None), ("leg_1_s1", 14.0, "leg_0_s1"), ("leg_1_s3", 101.0, "leg_0_s3")]
    # the drive arms the heartbeat BEFORE launch_legs and keys its loop on the stop event (not on "any thread alive", which was False
    # before the first launch); the constants are the plan's
    main = SRC.read_text().split("def main(")[1]
    assert main.index("hb_t.start()") < main.index("launches = launch_legs(") < main.index("for t in threads:\n        t.join()")
    assert "while any(t.is_alive() for t in threads):" not in main and "if hb_stop.wait(HEARTBEAT_PERIOD_S):" in main
    assert (d.BUNDLE_WAIT_CAP_S, d.LAUNCH_STAGGER_S, d.HEARTBEAT_PERIOD_S) == (900, 2, 300)


# --------------------------------------------------------------------------- P1-2: wave/attempt-scoped roots + the store reconciliation

WAVE_A, WAVE_B = "8f14e45f-ceea-467a-9575-6e8e7c2f5b31", "c0ffee11-2222-4333-8444-555566667777"
ATT_A, ATT_B = "ta-A1-0f0f0f0f-1111-4222-8333-444455556666", "ta-B2-0e0e0e0e-1111-4222-8333-444455556666"


def _complete_store(world, out_root: Path, scratch: Path):
    """A genuinely COMPLETE cell written by the Petri runner itself into ``<out_root>/cells/<run_key>/`` (manifest, rows,
    an identity-bearing settled ledger, a footed results.json) -- the fixture test_petri_already_done uses."""
    cell = world.cells["si_safety__psr_kl_all"]
    store = cs.CellStore(out_root, cell.petri_run_key)
    result = pr._run(object(), world.spec, cell, store, seed=world.spec.bloom_options.seed, driver=shape._driver(world, cell),
                     config_root=world.config_root, bundle_dir=world.bundle.bundle_dir, work_root=scratch / "work", log_root=scratch / "logs",
                     prompt_files=shape.PROMPT_FILES)
    assert store.is_done() and result["incomplete"] is False
    return cell, store


def _legs(d, cell):
    """The Arditi proxy's three seed-runs, keyed to the runner-written cell (one manifest-derived run key, three out roots)."""
    return [d.LegRun(view="arditi", cell_index=1, cell_id=cell.cell_id, vmodel=cell.virtual_model_name, run_key=cell.petri_run_key, seed_base=s,
                     width=4, tag=d.run_tag("arditi", s)) for s in (1, 2, 3)]


def _real_reconcile(d, epochs):
    """The drive's reconciliation, run through the LEG python -- here the test interpreter -- exactly as main() wires it."""
    return lambda run, cell_dir: d.reconcile_cell_store(sys.executable, REPO, cell_dir, run.cell_id, run.run_key, epochs)


def test_p1_2_a_new_wave_at_the_same_w_starts_from_empty_roots_and_launches_every_leg(d, world, tmp_path):
    home = tmp_path / "home"
    epochs = world.spec.bloom_options.epochs
    cell, _ = _complete_store(world, tmp_path / "seed-store", tmp_path / "scratch0")
    runs = _legs(d, cell)
    # WAVE A at W7: three complete Arditi seed-runs (stores the runner wrote), return records in A's attempt state dir
    roots_a = {r.tag: d.out_root_for(home, W7, d.wave_uuid8(WAVE_A), ATT_A, r.tag) for r in runs}
    state_a = d.attempt_state_dir(d.state_dir_for(home, W7, d.wave_uuid8(WAVE_A), "arditi", ""), ATT_A)
    state_a.mkdir(parents=True)
    for i, r in enumerate(runs):
        _complete_store(world, roots_a[r.tag], tmp_path / f"scratchA{i}")
        (state_a / f"{r.record_name}.rc").write_text(d.leg_status_line(0, "DA", roots_a[r.tag] / "cells" / r.run_key / "results.json") + "\n")
    va = d.completion_verdict(runs, roots_a, state_a, _real_reconcile(d, epochs))
    assert va["counts"] == {"complete": 3, "incomplete": 0, "failed": 0} and d.legs_to_run(runs, va) == []      # A itself resumes to nothing
    # WAVE B at the SAME W7 (a relaunch mints a new wave uuid; its proxy is a new attempt): every root is DISTINCT and EMPTY
    roots_b = {r.tag: d.out_root_for(home, W7, d.wave_uuid8(WAVE_B), ATT_B, r.tag) for r in runs}
    state_b = d.attempt_state_dir(d.state_dir_for(home, W7, d.wave_uuid8(WAVE_B), "arditi", ""), ATT_B)
    state_b.mkdir(parents=True)
    assert set(roots_b.values()).isdisjoint(roots_a.values()) and not any(p.exists() for p in roots_b.values())
    # RE-VALIDATED repro: under the old spelling (W7 + tag only) wave B's roots WERE wave A's -- three "complete" runs and zero launches
    old_spelling = {r.tag: home / ".local/state/petri_native" / f"petri_seedrep_{W7}_{r.tag}" for r in runs}
    assert len({*old_spelling.values()}) == 3 and all(str(p).split(f"petri_seedrep_{W7}_")[1] in str(roots_a[t]) for t, p in old_spelling.items())
    assert all(d.wave_uuid8(WAVE_A) in str(p) and ATT_A in str(p) for p in roots_a.values())
    assert all(d.wave_uuid8(WAVE_B) in str(p) and ATT_B in str(p) for p in roots_b.values())
    vb = d.completion_verdict(runs, roots_b, state_b, _real_reconcile(d, epochs))
    assert vb["counts"] == {"complete": 0, "incomplete": 0, "failed": 3} and {w for _n, _k, w in vb["detail"]} == {"failed_no_results"}
    assert [r.record_name for r in d.legs_to_run(runs, vb)] == [r.record_name for r in runs]                # wave B launches ALL its legs
    # the same-W re-execution of the PROXY (new attempt id, same wave) is likewise a fresh set of roots
    roots_a2 = {r.tag: d.out_root_for(home, W7, d.wave_uuid8(WAVE_A), ATT_B, r.tag) for r in runs}
    assert set(roots_a2.values()).isdisjoint(roots_a.values()) and not any(p.exists() for p in roots_a2.values())


def test_p1_2_the_skip_needs_the_petri_driver_s_own_store_reconciliation_never_a_results_file(d, world, tmp_path):
    epochs = world.spec.bloom_options.epochs
    cell, store = _complete_store(world, tmp_path / "root_s1", tmp_path / "scratch")
    runs = _legs(d, cell)
    roots = {r.tag: tmp_path / f"root_{r.tag[-2:]}" for r in runs}
    assert roots["qwen35_4b_arditi_s1"] == tmp_path / "root_s1"
    state = tmp_path / "state"
    state.mkdir()
    real = _real_reconcile(d, epochs)
    # s1: the runner-written store -> the driver's own `_already_done` bears it out (in-process and through the leg-python subprocess agree)
    assert prun._already_done(store, cell, epochs) is True
    assert d.reconcile_cell_store(sys.executable, REPO, store.cell_dir, cell.cell_id, cell.petri_run_key, epochs) == (True, "")
    # s2: a BARE results.json (what a prior wave's adopted marker, a copied file, or a hand-made done marker looks like) -> NOT complete
    bare = roots["qwen35_4b_arditi_s2"] / "cells" / cell.petri_run_key
    bare.mkdir(parents=True)
    (bare / "results.json").write_text(json.dumps({"unscored": 0}))
    ok, why = d.reconcile_cell_store(sys.executable, REPO, bare, cell.cell_id, cell.petri_run_key, epochs)
    assert ok is False and why.startswith("DriverError: ") and "no manifest.json" in why
    # s3: a runner-written store whose LEDGER was removed -> the marker is no longer borne out (the driver's own refusal, quoted)
    _complete_store(world, roots["qwen35_4b_arditi_s3"], tmp_path / "scratch3")
    pr.ledger_path(cs.CellStore(roots["qwen35_4b_arditi_s3"], cell.petri_run_key)).unlink()
    ok3, why3 = real(runs[2], roots["qwen35_4b_arditi_s3"] / "cells" / cell.petri_run_key)
    assert ok3 is False and f"no {prun.FAILURE_LEDGER}" in why3
    # the verdict: ONLY s1 is complete (skipped); s2 and s3 are `unreconciled` -> failed, re-driven, the proxy held -- despite three results.json files
    assert all((roots[r.tag] / "cells" / r.run_key / "results.json").is_file() for r in runs)
    v = d.completion_verdict(runs, roots, state, real)
    assert v["counts"] == {"complete": 1, "incomplete": 0, "failed": 2} and v["release"] is False
    assert v["detail"][0][2] == "complete_by_cell_dir_alone" and v["detail"][1][2].startswith("unreconciled: DriverError: ") and "no manifest.json" in v["detail"][1][2]
    assert v["detail"][2][2].startswith("unreconciled: DriverError: ") and prun.FAILURE_LEDGER in v["detail"][2][2]
    assert [r.record_name for r in d.legs_to_run(runs, v)] == ["leg_1_s2", "leg_1_s3"]
    # another leg's directory under this run key (the driver's "another leg's" refusal) and a directory not named by its run key
    other = d.LegRun(**{**runs[0].__dict__, "cell_id": "some_other_cell"})
    ok_o, why_o = real(other, store.cell_dir)
    assert ok_o is False and "another leg's" in why_o
    assert d.reconcile_cell_store(sys.executable, REPO, tmp_path / "root_s1" / "cells" / "not-the-key", cell.cell_id, cell.petri_run_key, epochs)[1].startswith("DriverError: ")
    # no results.json at all -> (False, "no results.json"), never an exception
    assert real(runs[0], tmp_path / "nowhere" / "cells" / cell.petri_run_key) == (False, "no results.json")
    # the drive's epochs come from the leg python through the driver's own loader; the real s1 manifest says 1
    assert d.manifest_epochs(sys.executable, REPO, d.view_params("s1").manifest_rel) == 1
    with pytest.raises(SystemExit, match="cannot load"):
        d.manifest_epochs(sys.executable, REPO, "configs/does-not-exist.yaml")
    # the source: no results-file-existence skip anywhere, the reconciliation is required (no default), the drive wires the leg python into it
    src = SRC.read_text()
    assert "reconcile: ReconcileFn) -> dict" in src and "reconcile_cell_store(py, wt, cell_dir, run.cell_id, run.run_key, epochs)" in src
    assert "petri_run._already_done(store, SimpleNamespace(cell_id=cell_id, petri_run_key=run_key), int(epochs))" in d.RECONCILE_SNIPPET


def test_p1_2_resume_from_imports_a_prior_wave_s_stores_explicitly_with_provenance_and_refuses_the_unsafe_cases(d, world, tmp_path):
    home = tmp_path / "home"
    epochs = world.spec.bloom_options.epochs
    cell, _ = _complete_store(world, tmp_path / "seed", tmp_path / "scratch0")
    runs = _legs(d, cell)
    # the PRIOR wave A: complete stores + its drive record (what main() writes) in its attempt state dir
    roots_a = {r.tag: d.out_root_for(home, W7, d.wave_uuid8(WAVE_A), ATT_A, r.tag) for r in runs}
    state_a = d.attempt_state_dir(d.state_dir_for(home, W7, d.wave_uuid8(WAVE_A), "arditi", ""), ATT_A)
    state_a.mkdir(parents=True)
    for i, r in enumerate(runs):
        _complete_store(world, roots_a[r.tag], tmp_path / f"scratchA{i}")
        (roots_a[r.tag] / "cells.json").write_text("{}")
        (roots_a[r.tag] / "modal_drive.json").write_text("{}")
    rec_a = {"schema": "seedrep_petri_drive/1", "drive_id": "DA", "view": "arditi", "W": W, "W7": W7, "wave_uuid": WAVE_A, "attempt_id": ATT_A,
             "out_roots": {t: str(p) for t, p in roots_a.items()}, "started_at": "2026-09-18T00:00:00Z"}
    (state_a / "drive_DA.json").write_text(json.dumps(rec_a))
    (state_a / "drive_D9.json").write_text(json.dumps({**rec_a, "drive_id": "D9", "started_at": "2026-09-17T00:00:00Z", "out_roots": {}}))  # older, ignored
    # wave B imports EXPLICITLY
    roots_b = {r.tag: d.out_root_for(home, W7, d.wave_uuid8(WAVE_B), ATT_B, r.tag) for r in runs}
    rec = d.import_prior_stores(state_a, roots_b, w7=W7, wave_uuid=WAVE_B, attempt_id=ATT_B, view="arditi", drive_id="DB")
    assert rec["schema"] == "seedrep_petri_drive_resume/1" and rec["drive_id"] == "DB"
    assert rec["into"] == {"W7": W7, "wave_uuid": WAVE_B, "attempt_id": ATT_B, "view": "arditi"}
    assert rec["from"]["drive_record"] == str(state_a / "drive_DA.json") and rec["from"]["drive_id"] == "DA"        # the NEWEST record, by started_at
    assert (rec["from"]["W"], rec["from"]["W7"], rec["from"]["wave_uuid"], rec["from"]["attempt_id"]) == (W, W7, WAVE_A, ATT_A)
    assert rec["from"]["out_roots"] == {t: str(p) for t, p in roots_a.items()}
    for r in runs:
        assert rec["copied"][r.tag] == {"from": str(roots_a[r.tag]), "to": str(roots_b[r.tag]), "cell_dirs": [cell.petri_run_key]}
        assert (roots_b[r.tag] / "cells" / cell.petri_run_key / "results.json").read_bytes() == (roots_a[r.tag] / "cells" / cell.petri_run_key / "results.json").read_bytes()
        assert json.loads((roots_b[r.tag] / d.RESUMED_FROM_FILE).read_text()) == rec                  # provenance beside every root
        assert not (roots_b[r.tag] / "cells.json").exists() and not (roots_b[r.tag] / "modal_drive.json").exists()  # B freezes its own enumeration
    # the imported stores pass the driver's own reconciliation -> wave B skips them (explicitly resumed, not silently adopted)
    state_b = d.attempt_state_dir(d.state_dir_for(home, W7, d.wave_uuid8(WAVE_B), "arditi", ""), ATT_B)
    state_b.mkdir(parents=True)
    vb = d.completion_verdict(runs, roots_b, state_b, _real_reconcile(d, epochs))
    assert vb["counts"] == {"complete": 3, "incomplete": 0, "failed": 0} and d.legs_to_run(runs, vb) == []
    # KNOWN-BAD 1: an import never writes over an existing cells/ (a second import, or a root the legs already wrote)
    with pytest.raises(SystemExit, match="already exists -- an import never writes over a store"):
        d.import_prior_stores(state_a, roots_b, w7=W7, wave_uuid=WAVE_B, attempt_id=ATT_B, view="arditi", drive_id="DB2")
    # KNOWN-BAD 2: another W; 3: another view; 4: itself; 5: a record without the tag; 6: not an attempt state dir
    fresh = {r.tag: tmp_path / "fresh" / r.tag for r in runs}
    with pytest.raises(SystemExit, match="W7 '0c19251' != this drive's 'abcdef0'"):
        d.import_prior_stores(state_a, fresh, w7="abcdef0", wave_uuid=WAVE_B, attempt_id=ATT_B, view="arditi", drive_id="DB")
    with pytest.raises(SystemExit, match="view 'arditi' != 's1'"):
        d.import_prior_stores(state_a, fresh, w7=W7, wave_uuid=WAVE_B, attempt_id=ATT_B, view="s1", drive_id="DB")
    with pytest.raises(SystemExit, match="this very \\(wave, attempt\\)"):
        d.import_prior_stores(state_a, fresh, w7=W7, wave_uuid=WAVE_A, attempt_id=ATT_A, view="arditi", drive_id="DB")
    with pytest.raises(SystemExit, match="no out root recorded for qwen35_4b_s1"):
        d.import_prior_stores(state_a, {"qwen35_4b_s1": tmp_path / "x"}, w7=W7, wave_uuid=WAVE_B, attempt_id=ATT_B, view="arditi", drive_id="DB")
    with pytest.raises(SystemExit, match="no drive_<id>.json under"):
        d.import_prior_stores(tmp_path / "empty", fresh, w7=W7, wave_uuid=WAVE_B, attempt_id=ATT_B, view="arditi", drive_id="DB")
    assert not any(p.exists() for p in fresh.values())                    # a refused import leaves no root behind
    # the CLI flag exists and is the ONLY import path; main() wires it before the enumeration freeze and records it
    main = SRC.read_text().split("def main(")[1]
    # main resolves it through resolve_resume -- the flag imports, or a COMPLETED recovery is adopted as the import; main never
    # calls import_prior_stores directly any more
    assert main.index("recovery, resumed_from, resume_how = recover_and_resolve_resume(out_roots, resume_from=Path(args.resume_from).expanduser() "
                      "if args.resume_from else None,") < main.index('"print-cells"') and '"resumed_from": resumed_from' in main
    assert "import_prior_stores(" not in main


def test_integration_the_drive_reads_the_engine_first_through_mains_reader_and_never_a_lens_fact(d):
    """Integration fix (W gate 2, main's `tests/test_tinycap_launcher_native.py::test_every_reduction_mode_reader_under_scripts_modal_
    handles_the_engine_or_names_the_lens`): the inventory admits exactly two engine-aware readers of the lens reduction-mode fact under
    scripts/modal (tinycap_served, capctrl_served); every other launcher either names `--serving-path lens` or reads no lens fact. The drive
    is native-only, so it reads the ENGINE through main's one contract reader (`contracts.endpoint_is_native`) and spells no lens key. KNOWN-BAD
    first: a lens record (engine absent = lens BY ERA, or `lens`) refuses by name; a torn record (two engine spellings that disagree) refuses
    through the reader; the native record passes."""
    src = Path(d.__file__).read_text()
    assert "lens_reduction_mode" not in src and "PROXY_IDENTITY_REDUCTION_MODE" not in src
    assert d.endpoint_engine_problem(ENDPOINT) is None
    for lens in ({k: v for k, v in ENDPOINT.items() if k != "engine"}, {**ENDPOINT, "engine": "lens"}):
        p = d.endpoint_engine_problem(lens)
        assert p is not None and "the lane requires lora_native" in p, p
    from sisteer.capeval.tinycap import contracts as C
    torn = d.endpoint_engine_problem({**ENDPOINT, C.ENDPOINT_KEY_ENGINE: "lora_native", C.SERVED_ENGINE_FIELD: "lens"})   # two spellings that disagree
    assert torn is not None and "torn record" in torn, torn
    with pytest.raises(d.DriveRefused, match="the lane requires lora_native"):
        raise d.DriveRefused(d.endpoint_engine_problem({**ENDPOINT, "engine": "lens"}))


@pytest.mark.parametrize("stated", ["per_row", "per_batch", "unknown_mode"], ids=["per_row", "per_batch", "not_a_mode"])
def test_integration_a_native_endpoint_that_also_states_a_lens_reduction_mode_is_refused_through_mains_whole_record_reader(d, stated):
    """The refusal that predates the reader refactor -- `engine=lora_native` PLUS a non-null `lens_reduction_mode` is two engines on
    one record -- is back, read through main's engine-aware `contracts.served_reduction_mode` over the WHOLE record (the reader refuses
    that combination itself), so the drive still spells no lens key of its own. KNOWN-BAD: an `endpoint_is_native`-only check passed
    exactly this record."""
    from sisteer.capeval.tinycap import contracts as C
    src = Path(d.__file__).read_text()
    assert "lens_reduction_mode" not in src and "PROXY_IDENTITY_REDUCTION_MODE" not in src        # the roster test's rule still holds
    assert "served_reduction_mode(ep" in src
    torn = {**ENDPOINT, C.PROXY_IDENTITY_REDUCTION_MODE: stated}
    assert C.endpoint_is_native(torn) is True                                                    # the engine check alone admits it
    with pytest.raises(ValueError, match="no lens hook"):
        C.served_reduction_mode(torn, source="endpoint.json")                                    # main's reader refuses it
    p = d.endpoint_engine_problem(torn)
    assert p is not None and "two engines on one record" in p and "no lens hook" in p and f"{stated!r}" in p, p
    with pytest.raises(d.DriveRefused, match="two engines on one record"):
        raise d.DriveRefused(p)
    # the same record with the lens fact NULL is the native record the lane serves: no problem (a null mode states nothing)
    assert d.endpoint_engine_problem({**ENDPOINT, C.PROXY_IDENTITY_REDUCTION_MODE: None}) is None
    assert d.endpoint_engine_problem(ENDPOINT) is None


def test_d77_the_drive_reads_the_pool_bearer_only_from_under_the_runtime_dir(d, monkeypatch, tmp_path):
    """`read_private_key_file` -- what `--qwen-auditor-pool`'s record `api_key_file` goes through -- reads a 0600 file under
    `$XDG_RUNTIME_DIR` and refuses, unread, a file anywhere else (the earlier `~/.local/state` location included, a symlink
    resolving outside included), a runtime dir unset, a non-private mode, an empty file."""
    rd = tmp_path / "run"
    rd.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(rd))
    kf = rd / "seedrep-battery" / "attacker-pool" / "seedrep" / "api_key"
    kf.parent.mkdir(parents=True)
    kf.write_text("bearer-x\n")
    kf.chmod(0o600)
    assert d.read_private_key_file(kf) == "bearer-x"
    # KNOWN-BAD: the earlier location, 0600 and non-empty, is refused unread
    legacy = tmp_path / "home" / ".local" / "state" / "sisteer" / "attacker-pool-modal" / "seedrep" / "api_key"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("bearer-legacy\n")
    legacy.chmod(0o600)
    with pytest.raises(d.DriveRefused, match="not under the tmpfs runtime dir .*SEEDREP_ATTACKER_ROTATE_KEY=1"):
        d.read_private_key_file(legacy)
    link = rd / "sneaky_api_key"
    link.symlink_to(legacy)
    with pytest.raises(d.DriveRefused, match="not under the tmpfs runtime dir"):
        d.read_private_key_file(link)
    kf.chmod(0o640)
    with pytest.raises(d.DriveRefused, match="is not private"):
        d.read_private_key_file(kf)
    kf.chmod(0o600)
    kf.write_text("")
    with pytest.raises(d.DriveRefused, match="refusing empty bearer file"):
        d.read_private_key_file(kf)
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    kf.write_text("bearer-x\n")
    with pytest.raises(d.DriveRefused, match="XDG_RUNTIME_DIR is unset"):
        d.read_private_key_file(kf)
    # the drive reads NO pool bearer any more -- the abliterated-Qwen third auditor leg is retired and the chain is API-only
    # (sonnet -> grok); the old call site is gone and the retired flags refuse by name.
    src = Path(d.__file__).read_text()
    assert 'bearer = read_private_key_file(Path(kf)) if kf else os.environ.get("VLLM_API_KEY", "")' not in src
    assert "REMOVED_QWEN_LEG_MSG" in src and "raise DriveRefused(f\"{REMOVED_QWEN_LEG_MSG}\")" in src
    assert "sonnet -> grok" in d.REMOVED_QWEN_LEG_MSG and "--qwen-auditor-pool" in d.REMOVED_QWEN_LEG_MSG
