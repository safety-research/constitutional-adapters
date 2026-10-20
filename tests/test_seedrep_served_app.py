"""scripts/modal/seedrep_served.py under a stub `modal` (nothing launched): the registry identity env wiring per CHILD,
the container env, the per-child concurrency from the
attested S, the readiness assertions against a row (good endpoint admitted; wrong S / missing adjustment / `override` on
a table window / max_logprobs 20 on the controls standup / an exported adapter / a wrong pin each NAMED), the app-name
rule, the spawn context, and the watcher's pure classification (STARTED / PROGRESS / STALL / FAILED / PRODUCER_MISSING /
DONE / WATCH-TIMEOUT) over a fake backend."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal"):
    sys.path.insert(0, str(p))
import seedrep_launch_plan as lp  # noqa: E402
import seedrep_watch as sw  # noqa: E402

DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"


class _Chain:
    def __getattr__(self, _name):
        return lambda *a, **k: self

    def __call__(self, *a, **k):
        return self


class _App:
    def __init__(self, name=None, **k):
        self.description = None

    def function(self, **kw):
        return lambda f: f

    def local_entrypoint(self, **kw):
        return lambda f: f


def _stub_modal():
    m = types.ModuleType("modal")
    m.App = _App
    m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(commit=lambda: None, reload=lambda: None))
    m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain(), debian_slim=lambda *a, **k: _Chain())
    m.Dict = types.SimpleNamespace(from_name=lambda *a, **k: {})
    m.is_local = lambda: False
    m.current_function_call_id = lambda: "fc-test"
    return m


_ABSENT = object()


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_app_under_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal", _ABSENT)
    sys.modules["modal"] = _stub_modal()
    try:
        spec.loader.exec_module(mod)
    finally:
        if prior is _ABSENT:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    return mod


@pytest.fixture(scope="module")
def plan():
    inputs = lp.load_inputs(REPO)
    return lp.build_plan(W="4" * 40, wave_uuid="44444444-5555-4666-8444-777777777777", reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"],
                         pins=inputs["pins"], arditi_pins=inputs["arditi_pins"], arms=inputs["arms"], generated_at="2026-09-18T00:00:00Z")


def _row(plan, leg, engine="s1", stage="base"):
    return [r for r in plan["standups"] + plan["rungs"] if r["leg"] == leg and r["engine"] == engine and r["stage"] == stage][0]


def test_child_env_wires_one_identity_per_child_and_an_attempt_unique_spool(srv, plan):
    row = _row(plan, "pasttense")
    child = [c for c in row["children"] if c["arm"] == "prompted__si_constitution"][0]
    base = srv.container_env(bank_view="/bank/s1", tree="/tree", thinking=False, base={"PATH": "/usr/bin", "HF_HOME": "/stale", "PYTHONPATH": "/stale"})
    argv, env = srv.child_argv_env(child, attempt_root="/workspace/attempt", plans_root="/vol/seedrep/4444444/registry/plans",
                                   proxy_url="http://127.0.0.1:9000/v1", routing_table="/workspace/attempt/serve/routing_table.json",
                                   driver_python="/home/seedrep/venvs/f22-444444444444/bin/python", S=45, ncells=8, bank_view="/bank/s1", base_env=base)
    assert env["SISTEER_REGISTRY_RUN_IDENTITY"] == "/vol/seedrep/4444444/registry/plans/pasttense/prompted__si_constitution__s1.json"
    assert env["SISTEER_REGISTRY_SPOOL"] == "/workspace/attempt/registry/records"
    assert argv[0] == "/home/seedrep/venvs/f22-444444444444/bin/python" and argv[argv.index("--target-proxy-url") + 1] == "http://127.0.0.1:9000/v1"
    assert argv[argv.index("--private-root") + 1] == "/workspace/attempt/exp/transcripts_results"
    assert "{" not in " ".join(argv)
    # the container env: scrubbed then set
    assert env["HF_HOME"] == "/vol/hf" and env["HF_HUB_OFFLINE"] == "1" and env["PYTHONPATH"] == "/tree/src:/tree/scripts"
    assert env["SISTEER_ADAPTER_BANK_ROOT"] == "/vol/seedrep/multilora_adapters" and env["TT_TWO_SI_DEFENSES_DIR"] == "/bank/s1"
    assert env["F22_LOG_REQUESTS"] == "1" and env["F22_SERVER_DEFAULT_POLICY"] == "contract_override"
    assert env["CUDA_VISIBLE_DEVICES"] == "0" and env["USER"] == "seedrep" and "F22_THINKING_ON" not in env
    g8 = srv.container_env(bank_view="/b", tree="/t", thinking=True)
    assert g8["F22_THINKING_ON"] == "1"
    pet = srv.container_env(bank_view="/b", tree="/t", thinking=False, petri=True)
    assert pet["PETRI_EVAL_MAX_CONNECTIONS"] == "3"
    # two children of one engine: distinct identity files, the SAME attempt spool; two attempts: distinct spools
    other = [c for c in row["children"] if c["arm"] == "none"][0]
    _, env2 = srv.child_argv_env(other, attempt_root="/workspace/attempt", plans_root="/p", proxy_url="u", routing_table="t",
                                 driver_python="py", S=45, ncells=8, bank_view="/b", base_env=base)
    assert env2["SISTEER_REGISTRY_RUN_IDENTITY"] == "/p/pasttense/none__s1.json" and env2["SISTEER_REGISTRY_SPOOL"] == env["SISTEER_REGISTRY_SPOOL"]
    _, env3 = srv.child_argv_env(other, attempt_root="/workspace/other", plans_root="/p", proxy_url="u", routing_table="t",
                                 driver_python="py", S=45, ncells=8, bank_view="/b", base_env=base)
    assert env3["SISTEER_REGISTRY_SPOOL"] == "/workspace/other/registry/records"
    # the controls child names ONE file that is a per-task map (the writer produces it); misalign children carry the runner's env
    ctl = _row(plan, "controls", "arditi")["children"][1]
    _, envc = srv.child_argv_env(ctl, attempt_root="/a", plans_root="/p", proxy_url="u", routing_table="t", driver_python="py", S=70, ncells=2,
                                 bank_view="/b", base_env=base)
    assert envc["SISTEER_REGISTRY_RUN_IDENTITY"] == "/p/_stores/controls/arditi_allpos__s2.json"   # the producer's STORE map under ITS arm spelling
    mis = _row(plan, "misalign", "s3")["children"][0]
    argv_m, envm = srv.child_argv_env(mis, attempt_root="/a", plans_root="/p", proxy_url="u", routing_table="t", driver_python="py", S=21, ncells=8,
                                      bank_view="/b", base_env=base)
    assert envm["CUDA_VISIBLE_DEVICES"] == "" and envm["SISTEER_REGISTRY_RUN_IDENTITY"] == "/p/misalign_leaking/none__s3.json"
    assert argv_m[argv_m.index("--concurrency") + 1] == "2"


def test_concurrency_rules_from_the_attested_s(srv):
    assert srv.concurrency_for("one_request_each_S_drivers", 45, 8) == 45
    assert srv.concurrency_for("S", 31, 8) == 31
    assert srv.concurrency_for("S_over_ncells_floor_4", 45, 8) == 5 and srv.concurrency_for("S_over_ncells_floor_4", 21, 8) == 4
    assert srv.concurrency_for("S_over_ncells_floor_8", 70, 8) == 8 and srv.concurrency_for("S_over_ncells_floor_8", 70, 2) == 35
    assert srv.concurrency_for("S_over_ncells", 21, 8) == 2
    assert srv.concurrency_for("grading_host_drive", 15, 8) == 0
    with pytest.raises(ValueError):
        srv.concurrency_for("made_up", 1, 1)


def test_multiturn_child_needs_its_pool_url(srv, plan):
    child = _row(plan, "crescendo")["children"][0]
    base = srv.container_env(bank_view="/b", tree="/t", thinking=False, base={})
    with pytest.raises(ValueError, match="needs pool"):
        srv.child_argv_env(child, attempt_root="/a", plans_root="/p", proxy_url="u", routing_table="t", driver_python="py", S=10, ncells=8,
                           bank_view="/b", base_env=base)
    argv, _ = srv.child_argv_env(child, attempt_root="/a", plans_root="/p", proxy_url="u", routing_table="t", driver_python="py", S=10, ncells=8,
                                 bank_view="/b", base_env=base, pool_urls={"seedrep-attacker-0": "https://pool0.modal.run"})
    assert argv[argv.index("--attacker-endpoint") + 1] == "https://pool0.modal.run"


def _h(cell):
    return cell if cell == lp.NONE_LORA0 else lp.harness_id(cell)


FIXTURE_S1 = Path(__file__).resolve().parent / "fixtures" / "seedrep_f_l" / "S1"


def _endpoint(row, **over):
    """The NATIVE records' shape FROM THE REAL BYTES (the failed S1's endpoint.json + cells_served.json, tests/fixtures/seedrep_f_l),
    with the ROW's own values laid over the fields the plan asserts -- HARNESS ids everywhere, the derived `none__lora0` control beside
    the route set. Every key the writer produces is present; a test changes one and names the refusal."""
    import copy
    g = row["geometry"]
    ep = copy.deepcopy(json.loads((FIXTURE_S1 / "endpoint.json").read_text()))
    cells = copy.deepcopy(json.loads((FIXTURE_S1 / "cells_served.json").read_text()))
    S = g["S"]
    digests = {_h(c): o["object_sha256"] for c, o in row["objects"].items()}
    ep.update({"attested_max_model_len": g["L"], "attested_max_num_seqs": S, "max_model_len": g["L"], "max_num_seqs": S,
               "geometry": {**ep["geometry"], "window_source": g["expected_window_source"], "adjustments": [g["expected_adjustment"]] if g["expected_adjustment"] else []},
               "attested_cudagraph_full_capture_max": S, "attested_cudagraph_full_capture_sizes": sorted({x for x in (1, 2, 4, 8, 16, 24, 32, 40) if x < S} | {S}),   # vLLM: FULL sizes <= S only
               "lora_config": {**ep["lora_config"], "max_lora_rank": row["readiness"]["max_lora_rank"], "max_loras": row["readiness"]["max_loras"]},
               "max_logprobs": row["readiness"]["max_logprobs"], "thinking_mode": row["readiness"]["thinking_mode"],
               "adapters_reused": {_h(c): True for c in row["adapters"]},
               "object_artifact_sha256_by_cell": digests,
               "residual_slot": {"armed": True, "cells": {_h(c): {"kind": "psr", "position_mode": "all", "residual_fingerprint": "ab" * 32}
                                                          for c in row["readiness"]["residual_slot_armed"]}}})
    ep.update(over)
    cells.update({"attested_max_model_len": g["L"], "attested_max_num_seqs": S, "expected_object_sha256_by_cell": digests,
                  "canonical": [{"cell_id": _h(c), "cell_class": ("base" if c == "none" else "prompted" if c.startswith("prompted__") else "class_a"),
                                 "virtual_model_name": f"petri__{_h(c)}"} for c in row["route_set"]],
                  "controls": [{"cell_id": lp.NONE_LORA0, "cell_class": "control", "control_of": "none", "virtual_model_name": "petri__none__lora0"}]})
    return ep, cells


def test_readiness_admits_the_plan_and_names_every_mismatch(srv, plan):
    row = _row(plan, "controls")
    ep, cells = _endpoint(row)
    assert srv.readiness_problems(ep, cells, row) == []
    ep2, _ = _endpoint(row, attested_max_num_seqs=99)
    assert any("attested_max_num_seqs 99 != 70" in p for p in srv.readiness_problems(ep2, cells, row))
    ep3, _ = _endpoint(row, geometry={"window_source": "table", "adjustments": []})
    assert any("kernel adjustment" in p for p in srv.readiness_problems(ep3, cells, row))
    ep4, _ = _endpoint(row, geometry={"window_source": "override", "adjustments": [row["geometry"]["expected_adjustment"]]})
    assert any("window_source 'override' != 'table'" in p and "#200" in p for p in srv.readiness_problems(ep4, cells, row))
    ep5, _ = _endpoint(row, max_logprobs=20)
    assert any("max_logprobs 20 != 64" in p for p in srv.readiness_problems(ep5, cells, row))
    ep6, _ = _endpoint(row, adapters_reused={**{_h(c): True for c in row["adapters"]}, "si_constv2_mix_full__psr_ce_all": False})
    assert any("was EXPORTED" in p and "gate 10" in p for p in srv.readiness_problems(ep6, cells, row))
    ep7, _ = _endpoint(row, attested_cudagraph_mode="PIECEWISE")
    assert any("has no FULL capture surface" in p for p in srv.readiness_problems(ep7, cells, row))   # the real property, not the literal FULL
    # the pins live in cells_served.json; a tampered pin there is named, and an artifact map that disagrees with it is named too
    cells8 = {**cells, "expected_object_sha256_by_cell": {**cells["expected_object_sha256_by_cell"], "si_constv2_mix_full__lora_r4_ce": "0" * 64}}
    probs8 = srv.readiness_problems(ep, cells8, row)
    assert any("!= pin" in p for p in probs8) and any("!= cells_served.expected_object_sha256_by_cell" in p for p in probs8)
    assert any("cells_served.expected_object_sha256_by_cell ABSENT" in p for p in srv.readiness_problems(ep, {k: v for k, v in cells.items() if k != "expected_object_sha256_by_cell"}, row))
    bad_cells = {"canonical": [c for c in cells["canonical"] if c["cell_id"] != "prompted__safety"], "controls": cells["controls"]}
    assert any("cells_served.canonical" in p for p in srv.readiness_problems(ep, bad_cells, row))
    no_control = {"canonical": cells["canonical"], "controls": []}
    assert any("cells_served.controls [] != ['none__lora0']" in p for p in srv.readiness_problems(ep, no_control, row))
    # a non-table row (PAIR) reads `override` and no adjustment when the identity S already fits... PAIR lowers 30 -> 21
    pair = _row(plan, "pair")
    epp, cp = _endpoint(pair)
    assert epp["geometry"]["window_source"] == "override" and srv.readiness_problems(epp, cp, pair) == []
    # a static engine at 11,520: identity 64 -> 45 under the kernel adjustment, table source kept
    st = _row(plan, "grok420", "s3")
    eps, cs = _endpoint(st)
    assert {k: eps["geometry"][k] for k in ("window_source", "adjustments")} == \
        {"window_source": "table", "adjustments": [{"kind": "lora_kernel_index_ceiling", "from_S": 64, "to_S": 45, "D": 4096, "ceiling": 524287}]}   # the real block carries more keys
    assert srv.readiness_problems(eps, cs, st) == []


def test_app_name_rule_and_spawn_ctx(srv, plan):
    assert srv.expected_app_name("4" * 40) == "seedrep-served-4444444"
    ctx = srv.spawn_ctx(plan, "seedrep-served-4444444", {"seedrep-attacker-0": "u"})
    assert ctx["W"] == "4" * 40 and ctx["dicts"]["accepted"] == "seedrep-4444444-44444444-accepted"
    assert ctx["plans_root"] == "/vol/seedrep/4444444/registry/plans" and ctx["ladder_root"].endswith("/ladder")
    assert ctx["pool_urls"] == {"seedrep-attacker-0": "u"}
    assert srv.volume_client_path("/vol/seedrep/x") == "/seedrep/x"
    with pytest.raises(ValueError):
        srv.volume_client_path("/workspace/x")


def test_verify_staged_objects_reads_view_manifests_and_pins(srv, tmp_path):
    import hashlib
    pins = json.loads((REPO / lp.PINS_REL).read_text())
    apins = json.loads((REPO / lp.ARDITI_PINS_REL).read_text())
    bank = tmp_path / "bank" / "qwen35_4b"
    ar = tmp_path / "arditi"
    roster = __import__("yaml").safe_load((REPO / lp.ROSTER_REL).read_text())
    for v in lp.VIEWS:
        view = bank / "cells_view" / v
        view.mkdir(parents=True)
        entries = []
        for cell, sid in lp.roster_view_cells(roster, v).items():
            d = bank / "cells" / sid
            d.mkdir(parents=True, exist_ok=True)
            (d / "object.pt").write_bytes(sid.encode())
            (view / cell).symlink_to(f"../../cells/{sid}")
            entries.append({"cell_id": cell, "staged_id": sid, "object_sha256": hashlib.sha256(sid.encode()).hexdigest()})
        (view / "VIEW_MANIFEST.json").write_text(json.dumps({"target": "qwen35_4b", "view": v, "entries": entries}))
    (ar / "cells" / "ar_c0p3__arditi_allpos").mkdir(parents=True)
    (ar / "cells" / "ar_c0p3__arditi_allpos" / "object.pt").write_bytes(b"arditi")
    (ar / "cells_view" / "ar_c0p3").mkdir(parents=True)
    (ar / "cells_view" / "ar_c0p3" / "arditi_allpos").symlink_to("../../cells/ar_c0p3__arditi_allpos")
    (ar / "cells_view" / "ar_c0p3" / "VIEW_MANIFEST.json").write_text(json.dumps({"target": "qwen35_4b", "view": "ar_c0p3", "entries": [
        {"cell_id": "arditi_allpos", "staged_id": "ar_c0p3__arditi_allpos", "object_sha256": hashlib.sha256(b"arditi").hexdigest()}]}))
    # synthetic bytes never equal the real pins -> every entry is a NAMED problem (the pin is the authority)
    rep = srv.verify_staged_objects(str(tmp_path / "bank"), str(ar), pins, apins)
    assert rep["ok"] is False and len(rep["problems"]) >= 16
    # with pins rewritten to the synthetic shas the 16 verify
    fake_pins = {"qwen35_4b": {sid: {"object_sha256": hashlib.sha256(sid.encode()).hexdigest()}
                               for v in lp.VIEWS for sid in lp.roster_view_cells(roster, v).values()}}
    fake_ar = {"qwen35_4b": {"ar_c0p3__arditi_allpos": {"object_sha256": hashlib.sha256(b"arditi").hexdigest()}}}
    rep = srv.verify_staged_objects(str(tmp_path / "bank"), str(ar), fake_pins, fake_ar)
    assert rep["ok"] is True and len(rep["shas"]) == 16 and rep["n_entries"] == 16


# --- the watcher's pure classification -----------------------------------------------------------------------------

def _calls(n=2):
    return [{"call_id": f"fc-{i}", "standup_id": f"pasttense__s{i + 1}__base__shard00", "leg": "pasttense", "engine": f"s{i + 1}",
             "stage": "base", "shard": 0, "control_key": f"W/wave/pasttense/s{i + 1}/base/0", "accepted_key": f"W|wave|pasttense|s{i + 1}|base|0",
             "wave_uuid": "wave"} for i in range(n)]


def _required(calls):
    return {c["accepted_key"]: c["standup_id"] for c in calls}


def _hb(t, rows, running=45, attempt="a-1"):
    return {"t": t, "rows_written": rows, "running": running, "phase": "drivers", "attempt_id": attempt}


def test_watcher_classifies_started_progress_stall_failed_done():
    import time as _t
    st = sw.WatchState()
    calls = _calls(2)
    t0 = 1_800_000_000.0
    iso = lambda t: _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(t))  # noqa: E731
    snap = {"fc-0": {"done": False, "heartbeat": _hb(iso(t0), 10)}, "fc-1": {"done": False, "heartbeat": None}}
    req = _required(calls)
    ev = sw.classify(now=t0, leg="pasttense", calls=calls, snapshot=snap, state=st, stall_min=20, rows_flat_min=40, progress_min=30, required_keys=req)
    assert ev[0].startswith("WAVE_LAUNCHED {")                       # the summary line leads the first poll with calls, once
    per_call = lambda evs: [e for e in evs if not e.startswith(("WAVE_LAUNCHED ", "WAVE_DONE ", "HOLD "))]  # noqa: E731
    assert per_call(ev) == ["STARTED pasttense__s1__base__shard00 attempt=a-1 call=fc-0", "PROGRESS pasttense__s1__base__shard00 rows=10 rows/min=10.0 running=45 phase=drivers"]
    # 25 min later with no fresh heartbeat -> STALL (no heartbeat); the second call still has none (never started -> no stall)
    snap = {"fc-0": {"done": False, "heartbeat": _hb(iso(t0), 10)}, "fc-1": {"done": False, "heartbeat": None}}
    ev = sw.classify(now=t0 + 25 * 60, leg="pasttense", calls=calls, snapshot=snap, state=st, stall_min=20, rows_flat_min=40, progress_min=30, required_keys=req)
    assert ev == ["STALL pasttense__s1__base__shard00 call=fc-0 no heartbeat for 25 min"]
    # a failed call: the error + the log tail ride the event; a DONE summary counts as complete
    snap = {"fc-0": {"done": True, "error": "RemoteError: OOM", "log_tail": "line1\nline2", "summary": None, "accepted": None},
            "fc-1": {"done": True, "summary": {"status": "DONE"}, "accepted": "b-2", "heartbeat": _hb(iso(t0), 500, attempt="b-2")}}
    ev = sw.classify(now=t0 + 30 * 60, leg="pasttense", calls=calls, snapshot=snap, state=st, stall_min=20, rows_flat_min=40, progress_min=30, required_keys=req)
    assert "STARTED pasttense__s2__base__shard00 attempt=b-2 call=fc-1" in ev
    assert any(e.startswith("FAILED pasttense__s1__base__shard00 call=fc-0 RemoteError: OOM") and "line2" in e for e in ev)
    assert any(e.startswith("PRODUCER_MISSING pasttense/s1/base") for e in ev), "every attempt of s1 terminal, none accepted"
    assert not any(e.startswith("DONE") for e in ev)
    # both accepted -> DONE
    snap = {"fc-0": {"done": True, "summary": {"status": "DONE"}, "accepted": "c-3"}, "fc-1": {"done": True, "summary": {"status": "DONE"}, "accepted": "b-2"}}
    ev = sw.classify(now=t0 + 31 * 60, leg="pasttense", calls=calls, snapshot=snap, state=st, stall_min=20, rows_flat_min=40, progress_min=30, required_keys=req)
    assert per_call(ev) == ["DONE pasttense scope=base required=2 accepted=2"] and any(e.startswith("WAVE_DONE {") for e in ev)


def test_watcher_rows_flat_rule_applies_to_multiturn_and_petri_only():
    import time as _t
    iso = lambda t: _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(t))  # noqa: E731
    t0 = 1_800_000_000.0
    for leg, expect_stall in (("crescendo", True), ("pasttense", False)):
        st = sw.WatchState()
        calls = [{**_calls(1)[0], "leg": leg}]
        sw.classify(now=t0, leg=leg, calls=calls, snapshot={"fc-0": {"done": False, "heartbeat": _hb(iso(t0), 7)}}, state=st,
                    stall_min=20, rows_flat_min=40, progress_min=30, required_keys=_required(calls))
        ev = sw.classify(now=t0 + 45 * 60, leg=leg, calls=calls, snapshot={"fc-0": {"done": False, "heartbeat": _hb(iso(t0 + 45 * 60), 7)}},
                         state=st, stall_min=20, rows_flat_min=40, progress_min=30, required_keys=_required(calls))
        assert any("rows flat for 45 min" in e for e in ev) is expect_stall, (leg, ev)


def test_watch_leg_exits_on_done_and_on_timeout(tmp_path, plan):
    rs = tmp_path
    bon_row = [r for r in plan["standups"] if r["leg"] == "bon" and r["engine"] == "s1"][0]
    calls = [{"call_id": "fc-9", "standup_id": "bon__s1__base__shard00", "leg": "bon", "engine": "s1", "stage": "base", "shard": 0,
              "control_key": bon_row["control_key"], "accepted_key": bon_row["accepted_key"], "wave_uuid": plan["wave_uuid"]}]
    (rs / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n")

    class Backend:
        def __init__(self, done):
            self.done = done

        def snapshot(self, calls_):
            return {c["call_id"]: ({"done": True, "summary": {"status": "DONE"}, "accepted": "a"} if self.done else {"done": False, "heartbeat": None})
                    for c in calls_}
    clock = {"t": 0.0}
    plan_bon = {**plan, "standups": [r for r in plan["standups"] if r["leg"] == "bon" and r["engine"] == "s1"]}
    rc = sw.watch_leg(rs, plan_bon, "bon", poll_s=120, stall_min=20, rows_flat_min=40, progress_min=30, max_hours=24,
                      backend=Backend(True), clock=lambda: clock["t"], sleep=lambda s: clock.__setitem__("t", clock["t"] + s))
    assert rc == 0
    rc = sw.watch_leg(rs, plan_bon, "bon", poll_s=120, stall_min=20, rows_flat_min=40, progress_min=30, max_hours=1,
                      backend=Backend(False), clock=lambda: clock["t"], sleep=lambda s: clock.__setitem__("t", clock["t"] + s))
    assert rc == 3
    with pytest.raises(SystemExit, match=r"\[30, 300\]"):
        sw.main(["--rs", str(rs), "--leg", "bon", "--poll", "10"])
