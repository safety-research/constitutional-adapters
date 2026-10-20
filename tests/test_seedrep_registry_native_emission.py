"""The qwen-seedrep-native-battery's rows record themselves in the results registry NATIVELY.

The design was in place before the records were: `resolve_spawn` composed every child's `SISTEER_REGISTRY_RUN_IDENTITY` (the staged
identity file's in-container path) and `SISTEER_REGISTRY_SPOOL` (`<attempt root>/registry/records`) correctly -- the earlier
pasttense children's `.argv.json` records carry both -- yet the attempt root on the volume has no `registry/` directory and the
registry on the grading host holds zero seedrep runs. The silences these tests pin, each on the REAL hooks (never a double):

* container: a static / BoN / multi-turn driver builds no CellStore and calls no hook, so the identity env reached a process
  that never read it. Now: `registry_row_preflight` asserts BEFORE the standup that the hooks under a composed child env are ON
  with the ATTEMPT spool (refused by name `REGISTRY_EMISSION_OFF` otherwise), and `child_registry_completion` records every
  exited child's generation through `hooks.emit_generation_complete` under the child's own env.
* The grading host: `_registry_queued`'s `from sisteer.registry import hooks` failed under `~/venvs/modal` (no `sisteer`; the app put only
  `scripts/modal` on `sys.path`) and the ImportError was returned on silently. Now: `grading_host_repo_src` puts `<repo>/src` on the
  path for the `<repo>/scripts/modal/` layout, the failure is SAID, and one run is queued per identity (six shard children of a
  static arm share one file) with the run id the container's store mints for the same identity.
* standup: `write_endpoint`'s hook found no `registry_runs.json` beside `endpoint.json` and would have resolved the SHARED
  `/vol/registry/records`. Now: `write_registry_runs` writes the served runs keyed by served key (a LIST of runs per key for the
  combined controls container -- an ADDITIVE acceptance in `hooks.emit_serving_patches`), and the standup gets the attempt spool.

The identity files are the REAL producer's (`registry_plans.build` over the committed decode-policy export); the plan is the REAL
generator's (`seedrep_launch_plan.build_plan`); the app module loads under a stub `modal` exactly as tests/test_seedrep_served_app.py
loads it. Synthetic wave ids / W only; no recorded data.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import emit_serving_facts_fanout as FAN  # noqa: E402
import registry_plans as RP  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
from sisteer.capeval import store as cs  # noqa: E402
from sisteer.registry import emit, hooks, identity, synth  # noqa: E402
from sisteer.registry import records as R  # noqa: E402
from sisteer.registry import validate as VAL  # noqa: E402

DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"
W = synth.code_sha("4")
WAVE = "44444444-5555-4666-8444-777777777777"
CAMPAIGN = "seedrep_battery_test"
REAL_POLICY = REPO / "configs" / "registry" / "decode_policy.json"
REV = "0123456789abcdef0123456789abcdef01234567"
KEY = "none__tier1__t0__a0__none__0f1e2d3c"      # a synthetic store key, not a credential -- gitleaks:allow
RESOLVED = {"sweep": "synthsweep", "run_id": "synthsweep.0.t0.a0", "model_id": synth.MODEL_ID, "model_revision": REV, "arm_name": "none",
            "kind": "none", "coeff": 1.0, "intervention_hash": None, "engine": "lora_native", "trial_id": 0, "seed": 1, "attempt": 0,
            "harness_config_hash": "0f1e2d3c4b5a6978", "env_digest": "99aabbccddeeff00", "tiers": [1], "cell_index": 0}


# --- the app under a stub modal (tests/test_seedrep_served_app.py's loader) -------------------------------------------------------

class _Chain:
    def __getattr__(self, name):
        return lambda *a, **k: self


class _App:
    def __init__(self, *a, **k):
        pass

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
    spec = importlib.util.spec_from_file_location("seedrep_served_registry_under_test", DRIVER)
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
    return lp.build_plan(W=W, wave_uuid=WAVE, reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"], pins=inputs["pins"],
                         arditi_pins=inputs["arditi_pins"], arms=inputs["arms"], generated_at="2026-09-19T00:00:00Z")


@pytest.fixture(scope="module")
def rs(tmp_path_factory):
    """A run-state root whose `registry/plans` holds the REAL producer's identity files for pasttense + the three controls legs
    (the `_stores/controls/<arm>__s<s>.json` per-task maps included), as `registry_plans.py --out RS/registry/plans` writes them."""
    root = tmp_path_factory.mktemp("RS")
    inp = RP.load_inputs(campaign=CAMPAIGN, w_sha=W, policy=REAL_POLICY)
    RP.write_plan_set(RP.build(inp, ["pasttense", "bon", "crescendo", "xstest", "or_bench", "mmlu"]), root / lp.PLANS_REL)
    return root


@pytest.fixture
def clean_switch(monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.delenv(hooks.EMIT_ENV, raising=False)
    monkeypatch.delenv(hooks.RUN_IDENTITY_ENV, raising=False)
    monkeypatch.delenv(emit.SPOOL_ENV, raising=False)
    monkeypatch.setattr(emit.socket, "gethostname", lambda: synth.TESTHOST)
    hooks._DEFAULT_NOTICED.clear()


def _row(plan, leg, engine="s1", stage="base", shard=0):
    return [r for r in plan["standups"] + plan["rungs"] if r["leg"] == leg and r["engine"] == engine and r["stage"] == stage and r["shard"] == shard][0]


def _child_env(srv, child, *, attempt_root: Path, plans_root: Path):
    base = srv.container_env(bank_view="/bank/s1", tree="/tree", thinking=False, base={"PATH": "/usr/bin"})
    return srv.child_argv_env(child, attempt_root=str(attempt_root), plans_root=str(plans_root), proxy_url="http://127.0.0.1:9000/v1",
                              routing_table=str(attempt_root / "exp/serve/routing_table.json"), driver_python="/venv/bin/python", S=45, ncells=8,
                              bank_view="/bank/s1", base_env=base)


def _records(spool: Path):
    return [(p, rec, sha) for p in R.iter_record_files(str(spool)) for _i, rec, sha in R.read_record_file(p)]


def _validate_all(spool: Path):
    for p, rec, sha in _records(spool):
        v = VAL.validate_record(rec, p, sha)
        assert isinstance(v, VAL.Validated), (p, v)


def _identity_of(path) -> dict:
    return hooks.normalise_identities(json.loads(Path(path).read_text()))["*"]


def _inprocess_probe(srv):
    """The REAL hooks under the composed env, in this process: every SISTEER_REGISTRY_* / REGISTRY_EMIT key of the env overlaid."""
    def probe(env):
        keys = {k: v for k, v in env.items() if k.startswith("SISTEER_REGISTRY") or k == hooks.EMIT_ENV}
        for k in (emit.DISABLED_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV, emit.SPOOL_ENV):
            keys.setdefault(k, None)
        with srv._env_overlay(keys):
            ident = hooks.identity_from_env()
            return {"enabled": hooks.emission_enabled(), "spool": hooks.resolve_spool(), "identity_routes": sorted(hooks.normalise_identities(ident)) if ident else []}
    return probe


# --- deliverable 1: the child env the PRODUCTION function composes turns the real hooks on; a CellStore under it records ----------

def test_child_env_from_the_production_function_turns_the_real_hooks_on_and_a_cellstore_writes_the_run(srv, plan, rs, tmp_path, monkeypatch, clean_switch):
    row = _row(plan, "pasttense")
    child = [c for c in row["children"] if c["arm"] == "none"][0]
    attempt = tmp_path / "attempt"
    _argv, env = _child_env(srv, child, attempt_root=attempt, plans_root=rs / lp.PLANS_REL)
    assert env[hooks.RUN_IDENTITY_ENV] == f"{rs / lp.PLANS_REL}/pasttense/none__s1.json" and Path(env[hooks.RUN_IDENTITY_ENV]).is_file()
    assert env[emit.SPOOL_ENV] == f"{attempt}/registry/records"
    # the switch, read exactly as the child process reads it
    monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, env[hooks.RUN_IDENTITY_ENV])
    monkeypatch.setenv(emit.SPOOL_ENV, env[emit.SPOOL_ENV])
    assert hooks.emission_enabled() is True
    assert hooks.resolve_spool() == str(attempt / "registry" / "records")
    ident = hooks.identity_from_env()
    assert ident["run_id"] == _identity_of(env[hooks.RUN_IDENTITY_ENV])["run_id"]
    # a CellStore built with NO recorder (the drivers' default since 2026-09-17) resolves the recorder from THAT env and records the run
    store = cs.CellStore(tmp_path / "store", KEY, row_constants={k: v for k, v in RESOLVED.items() if k not in ("tiers", "cell_index")})
    assert isinstance(store.recorder, hooks.StoreRecorder)
    store.validate_resume(RESOLVED)
    store.write_cell_manifest(RESOLVED)
    spool = attempt / "registry" / "records"
    runs = [rec for _p, rec, _s in _records(spool) if rec["record_type"] == "run"]
    assert len(runs) == 1 and runs[0]["payload"]["run_id"] == ident["run_id"], runs
    assert runs[0]["payload"]["campaign"] == CAMPAIGN and runs[0]["payload"]["eval_id"] == ident["eval_id"]
    _validate_all(spool)
    assert srv.count_spool_records(spool) >= 2          # run + run_state running


def test_registry_row_preflight_passes_on_a_composed_env_and_refuses_by_name_when_off_or_unstaged(srv, plan, rs, tmp_path, clean_switch):
    row = _row(plan, "pasttense")
    base = srv.container_env(bank_view="/b", tree="/t", thinking=False, base={"PATH": "/usr/bin"})
    attempt = tmp_path / "attempt"
    probe = _inprocess_probe(srv)
    block, problems = srv.registry_row_preflight(row, plans_root=str(rs / lp.PLANS_REL), attempt_root=str(attempt), base_env=base, probe=probe)
    assert problems == []
    assert block["enabled"] is True and block["spool"] == f"{attempt}/registry/records"
    assert block["probe"] == {"enabled": True, "spool": f"{attempt}/registry/records", "identity_routes": ["*"]}
    assert block["n_children_with_identity"] == len(row["children"]) == 48
    assert set(block["identity_paths"]) == {srv.child_log_name(k) for k in row["children"]}
    assert all(p.startswith(f"{rs / lp.PLANS_REL}/pasttense/") and Path(p).is_file() for p in block["identity_paths"].values())
    assert block["probe_env"] == {hooks.RUN_IDENTITY_ENV: f"{rs / lp.PLANS_REL}/pasttense/constv2-mix-full__lora_r4_ce__s1.json",
                                  emit.SPOOL_ENV: f"{attempt}/registry/records"} or block["probe_env"][emit.SPOOL_ENV] == f"{attempt}/registry/records"
    # KNOWN-BAD 1: the switch forced off in the container env -> refused, named
    off = dict(base, **{emit.DISABLED_ENV: "1"})
    block_off, problems_off = srv.registry_row_preflight(row, plans_root=str(rs / lp.PLANS_REL), attempt_root=str(attempt), base_env=off, probe=probe)
    assert block_off["enabled"] is False and any("emission is OFF" in p for p in problems_off), problems_off
    # KNOWN-BAD 2: the identity files not staged under the plans root -> every child named absent
    _b, problems_unstaged = srv.registry_row_preflight(row, plans_root=str(tmp_path / "nowhere"), attempt_root=str(attempt), base_env=base, probe=probe)
    assert len([p for p in problems_unstaged if "identity file absent" in p]) == 48
    # a row with no identity-bearing child (a rung consumer) is probed under the spool alone: ON, no problems
    rung = {"standup_id": "x", "children": [{"eval_leg": "pasttense", "arm": "consumer", "s": 0, "argv": ["c"]}]}
    block_r, problems_r = srv.registry_row_preflight(rung, plans_root=str(rs / lp.PLANS_REL), attempt_root=str(attempt), base_env=base, probe=probe)
    assert problems_r == [] and block_r["enabled"] is True and block_r["identity_paths"] == {}
    # the overlay leaves this process's environment as it found it
    assert hooks.RUN_IDENTITY_ENV not in os.environ and emit.SPOOL_ENV not in os.environ and emit.DISABLED_ENV not in os.environ


def test_registry_probe_runs_the_real_hooks_in_a_subprocess_under_the_child_env(srv, plan, rs, tmp_path, clean_switch):
    """The production probe: `python -c` under the child's env (the driver venv in-container; this interpreter here)."""
    child = [c for c in _row(plan, "pasttense")["children"] if c["arm"] == "none"][0]
    attempt = tmp_path / "attempt"
    _argv, env = _child_env(srv, child, attempt_root=attempt, plans_root=rs / lp.PLANS_REL)
    env = dict(env, PYTHONPATH=str(REPO / "src"), PATH=os.environ.get("PATH", "/usr/bin"))
    res = srv.registry_probe(sys.executable, env, tree=REPO)
    assert res == {"enabled": True, "spool": f"{attempt}/registry/records", "identity_routes": ["*"]}
    off = srv.registry_probe(sys.executable, dict(env, **{emit.DISABLED_ENV: "1"}), tree=REPO)
    assert off["enabled"] is False


# --- deliverable 2: the served runs beside endpoint.json, in the shape write_endpoint's hook reads -----------------------------

def _endpoint(runs: dict) -> dict:
    fitted = [k for k in runs if k not in ("none",) and not k.startswith("prompted__")]
    return {"engine": "lora_native", "attested_max_num_seqs": 45, "attested_max_model_len": 8192, "model_revision": REV, "attempt_id": "att-1",
            "adapters": {k: {"layers": [4, 8], "intervention_hash": "2ff9cdff091aef6f"} for k in fitted},
            "served_positions": {k: {"mode": "all"} for k in fitted}, "prefix_caching": "off"}


def test_registry_runs_json_is_keyed_by_served_key_and_feeds_emit_serving_patches(srv, plan, rs, tmp_path, clean_switch):
    row = _row(plan, "pasttense")
    serve_dir = tmp_path / "attempt" / lp.SERVE_DIR_REL
    serve_dir.mkdir(parents=True)
    summary = srv.write_registry_runs(row, plans_root=str(rs / lp.PLANS_REL), serve_dir=serve_dir)
    assert summary == {"path": str(serve_dir / hooks.RUNS_FILE), "n_served_keys": 8, "n_runs": 8}
    runs = hooks.load_runs_file(str(serve_dir))                   # THE consumer: write_endpoint's hook reads this file by this name
    # keyed by the endpoint's harness spelling of each arm's route (lp.harness_id), never the plan's own spelling
    assert runs is not None and set(runs) == {lp.harness_id(k["arm"]) for k in row["children"]}
    assert "prompted__constitution" in runs and "si_constv2_mix_full__lora_r4_ce" in runs and "constv2-mix-full__lora_r4_ce" not in runs
    FAN.check_keyed_by_served_key(runs)                            # never a run id as key (the hook reads keys as served keys)
    expected_ids = {_identity_of(p)["run_id"] for p in srv.registry_identity_files(row, plans_root=str(rs / lp.PLANS_REL)).values()}
    assert {r["run_id"] for r in runs.values()} == expected_ids and len(expected_ids) == 8       # 48 shard children -> 8 runs, each once
    assert all(r["campaign"] == CAMPAIGN and r["expected"] == {"serving_path": "lora_native"} for r in runs.values())
    # the hook, off the file, records one run_patch per served run into the given spool (the attempt spool in-container)
    spool = tmp_path / "attempt" / lp.SPOOL_REL
    paths = hooks.emit_serving_patches(_endpoint(runs), out_dir=str(serve_dir), spool=str(spool), registry_emit=True)
    assert len(paths) == 8
    patches = [rec for _p, rec, _s in _records(spool) if rec["record_type"] == "run_patch"]
    assert {p["payload"]["run_id"] for p in patches} == expected_ids
    fitted = [p for p in patches if p["payload"]["fields"].get("adapter_fingerprint")]
    assert len(fitted) == 5 and all(p["payload"]["verify"]["intervention_hash"] == "2ff9cdff091aef6f" for p in fitted)
    assert all(p["payload"]["fields"]["served_max_num_seqs"] == 45 and p["payload"]["verify"]["serving_path"] == "lora_native" for p in patches)
    _validate_all(spool)


def test_a_combined_controls_row_lists_its_three_runs_per_served_key_and_the_hook_records_each(srv, plan, rs, tmp_path, clean_switch):
    row = _row(plan, "controls")
    runs = srv.registry_runs_for_row(row, plans_root=str(rs / lp.PLANS_REL))
    assert len(runs) == 8 and all(isinstance(v, list) and len(v) == 3 for v in runs.values()), {k: type(v).__name__ for k, v in runs.items()}
    evals = {k: sorted(identity.parse_run_id(r["run_id"])["eval_id"] for r in v) for k, v in runs.items()}
    assert set(map(tuple, evals.values())) == {("mmlu", "or_bench_hard", "xstest")} or all(len(set(e)) == 3 for e in evals.values())
    FAN.check_keyed_by_served_key(runs)
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    srv.write_registry_runs(row, plans_root=str(rs / lp.PLANS_REL), serve_dir=serve_dir)
    spool = tmp_path / "spool"
    paths = hooks.emit_serving_patches(_endpoint(runs), out_dir=str(serve_dir), spool=str(spool), registry_emit=True)
    assert len(paths) == 24
    _validate_all(spool)


def test_hooks_emit_serving_patches_accepts_a_list_of_runs_per_served_key_additively(tmp_path, clean_switch):
    """The additive shape: a served key's value may be ONE run (the original) or a LIST of runs; each gets its patch."""
    def run(i):
        return {"run_id": identity.run_id(synth.CAMPAIGN, f"k{i}", "none", f"eval{i}", synth.INSTRUMENT_ID), "campaign": synth.CAMPAIGN,
                "expected": {"serving_path": "lora_native"}}
    ep = {"engine": "lora_native", "attested_max_num_seqs": 7, "adapters": {}}
    paths = hooks.emit_serving_patches(ep, {"cellA": [run(1), run(2)], "none": run(3)}, spool=str(tmp_path / "s"), registry_emit=True)
    assert len(paths) == 3
    ids = {rec["payload"]["run_id"] for _p, rec, _s in _records(tmp_path / "s")}
    assert ids == {run(1)["run_id"], run(2)["run_id"], run(3)["run_id"]}
    assert hooks.emit_serving_patches(ep, {"cellA": [{"campaign": "x"}]}, spool=str(tmp_path / "s2"), registry_emit=True) == []   # no run_id: skipped, named


# --- deliverable 3: the grading host spawn queues ONE run per identity, minted as the container's store mints it -------------------------

def test_registry_queued_writes_one_queued_run_per_identity_with_the_id_the_container_store_mints(srv, plan, rs, tmp_path, monkeypatch, clean_switch, capsys):
    row = _row(plan, "pasttense")
    spool = tmp_path / "grading_host-spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))                # the grading host resolves ~/registry/records; here a temp spool
    summary = srv._registry_queued(row, plan, "fc-0001", f"seedrep-served-{W[:7]}", rs)
    assert summary["queued"] == 8 and summary["records"] == 24 and summary["refused"] == [] and summary["skipped"] == [], summary
    assert summary["spool"] == str(spool)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("[registry] pasttense__s1__base__shard00: queued 8 run(s) (24 record(s)) -> ") and str(spool) in out[0]
    recs = _records(spool)
    runs = [rec["payload"] for _p, rec, _s in recs if rec["record_type"] == "run"]
    states = [rec["payload"] for _p, rec, _s in recs if rec["record_type"] == "run_state"]
    patches = [rec["payload"] for _p, rec, _s in recs if rec["record_type"] == "run_patch"]
    assert len(runs) == len(states) == len(patches) == 8                       # ONE per identity, not one per shard child
    # every queued run id == the identity file's run_id == what the container's store mints for it (identity.run_id over the same fields)
    by_child = srv.registry_identity_files(row, plans_root=str(rs / lp.PLANS_REL))
    expected = {}
    for path in set(by_child.values()):
        ident = _identity_of(path)
        minted = identity.run_id(ident["campaign"], ident["producer_run_key_legacy"], ident["producer_object_key_legacy"], ident["eval_id"], ident["instrument_id"])
        assert minted == ident["run_id"], path
        expected[ident["run_id"]] = ident
    assert {r["run_id"] for r in runs} == set(expected)
    for r in runs:
        assert r["lifecycle_state"] == "queued" and r["substrate"] == "modal" and r["eval_id"] == expected[r["run_id"]]["eval_id"]
    assert all(s["state"] == "queued" for s in states)
    assert all(p["fields"] == {"modal_app_id": f"seedrep-served-{W[:7]}", "modal_call_id": "fc-0001"} for p in patches)
    # and the SAME identity read through the child's env door mints the SAME id in the store recorder (the two sides name one run)
    one = next(iter(set(by_child.values())))
    monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, one)
    rec = hooks.store_recorder(KEY, RESOLVED, None, spool=str(tmp_path / "s2"), registry_emit=True)
    assert rec is not None and rec._run_id(rec.identities["*"]) == _identity_of(one)["run_id"]
    _validate_all(spool)


def test_registry_queued_says_when_the_hooks_are_unimportable_or_a_file_is_missing_instead_of_swallowing(srv, plan, rs, tmp_path, monkeypatch, clean_switch, capsys):
    row = _row(plan, "pasttense")

    def boom():
        raise ImportError("No module named 'sisteer'")
    monkeypatch.setattr(srv, "_hooks", boom)
    summary = srv._registry_queued(row, plan, "fc-1", "app", rs)
    assert summary["queued"] == 0 and summary["refused"] == ["sisteer.registry.hooks unimportable: ImportError: No module named 'sisteer'"]
    line = capsys.readouterr().out.strip()
    assert line.startswith("[registry] pasttense__s1__base__shard00: spawn NOT registered -- sisteer.registry.hooks unimportable") and "sys.path" in line
    monkeypatch.undo()
    monkeypatch.setenv(emit.SPOOL_ENV, str(tmp_path / "spool"))
    hooks._DEFAULT_NOTICED.clear()
    # a run-state root with no identity files: every child named, nothing queued, one line
    summary2 = srv._registry_queued(row, plan, "fc-2", "app", tmp_path / "emptyRS")
    assert summary2["queued"] == 0 and len(summary2["refused"]) == 48 and all("identity file absent under RS" in r for r in summary2["refused"])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and "queued 0 run(s)" in out[0] and "48 NOT registered" in out[0]


def test_grading_host_repo_src_is_detected_for_the_scripts_modal_layout_and_not_for_a_container_copy(srv):
    assert srv.grading_host_repo_src(REPO / "scripts" / "modal") == REPO / "src"
    assert srv.grading_host_repo_src(Path("/root")) is None                      # the container's copy beside the plan module: no repo
    assert srv.grading_host_repo_src(REPO / "scripts" / "sdfbase") is None
    assert str(REPO / "src") in sys.path


# --- the container side: a store-less child's generation is recorded by the container --------------------------------------------

def test_child_registry_completion_records_a_generation_patch_for_an_exited_child_and_nothing_for_a_failed_one(srv, plan, rs, tmp_path, clean_switch):
    row = _row(plan, "pasttense")
    child = [c for c in row["children"] if c["arm"] == "none" and c.get("goal_shard") == 2][0]
    attempt = tmp_path / "attempt"
    argv, env = _child_env(srv, child, attempt_root=attempt, plans_root=rs / lp.PLANS_REL)
    store_root = lp.child_store_root(child, argv)
    assert store_root == f"{attempt}/exp/transcripts_results"
    name = srv.child_log_name(child)
    reg_env = {e: env.get(e, "") for e in (lp.RUN_IDENTITY_ENV, lp.SPOOL_ENV, lp.SUBSTRATE_ENV)}
    paths = srv.child_registry_completion(child, reg_env, rc=0, store_root=store_root, attempt_id="att-1", log_name=name, code_sha=W)
    assert len(paths) == 1 and paths[0].startswith(str(attempt / "registry" / "records"))
    recs = _records(attempt / "registry" / "records")
    assert len(recs) == 1 and recs[0][1]["record_type"] == "run_patch"
    assert recs[0][1]["producer"]["code_sha"] == W and recs[0][1]["producer"]["substrate"] == "modal"      # the tree sha, never "unverified"
    payload = recs[0][1]["payload"]
    assert payload["run_id"] == _identity_of(env[hooks.RUN_IDENTITY_ENV])["run_id"]
    extra = payload["fields"]["extra_json"]
    assert extra["generation_complete_uri"] == store_root and extra["container_child"] == name and extra["attempt_id"] == "att-1"
    assert extra["goal_shard"] == 2 and extra["goal_nshards"] == child["goal_nshards"] and extra["eval_leg"] == "pasttense" and extra["rc"] == 0
    assert recs[0][1]["producer"]["script"] == "scripts/modal/seedrep_served.py child"
    _validate_all(attempt / "registry" / "records")
    # a failed child records nothing; a child with no identity records nothing; the process env is left untouched
    assert srv.child_registry_completion(child, reg_env, rc=1, store_root=store_root, attempt_id="att-1", log_name=name, code_sha=W) == []
    assert srv.child_registry_completion(child, {}, rc=0, store_root=store_root, attempt_id="att-1", log_name=name, code_sha=W) == []
    # the hook itself: code_sha threads to the producer; without it the git probe decides (this checkout's HEAD or "unverified")
    monkeypatch_free_spool = tmp_path / "hook-spool"
    with srv._env_overlay({lp.RUN_IDENTITY_ENV: reg_env[lp.RUN_IDENTITY_ENV], lp.SPOOL_ENV: str(monkeypatch_free_spool), lp.SUBSTRATE_ENV: "modal"}):
        hooks.emit_generation_complete("none", ["static/pasttense"], "/m", script="t", code_sha=synth.code_sha("c"))
    assert [rec["producer"]["code_sha"] for _p, rec, _s in _records(monkeypatch_free_spool)] == [synth.code_sha("c")]
    assert hooks.RUN_IDENTITY_ENV not in os.environ and emit.SPOOL_ENV not in os.environ
    assert srv.count_spool_records(attempt / "registry" / "records") == 1


def test_child_registry_completion_routes_a_controls_child_through_its_per_task_map(srv, plan, rs, tmp_path, clean_switch):
    row = _row(plan, "controls")
    child = [c for c in row["children"] if c["arm"] == "none"][0]
    # the BASE row's `none` child runs the three controls in ONE store: its task is the {control: store-writer task} map, three routes
    assert isinstance(child["task"], dict) and sorted(child["task"]) == ["mmlu", "or_bench_hard_1k", "xstest"]
    assert srv.child_tasks(child) == list(child["task"].values())
    attempt = tmp_path / "attempt"
    argv, env = _child_env(srv, child, attempt_root=attempt, plans_root=rs / lp.PLANS_REL)
    assert env[hooks.RUN_IDENTITY_ENV].endswith("/_stores/controls/none__s1.json")
    routes = hooks.normalise_identities(json.loads(Path(env[hooks.RUN_IDENTITY_ENV]).read_text()))
    assert set(routes) == {"judged/xstest_v2", "judged/or_bench_hard_1k", "s2parity/mmlu_letter"}
    reg_env = {e: env.get(e, "") for e in (lp.RUN_IDENTITY_ENV, lp.SPOOL_ENV)}
    paths = srv.child_registry_completion(child, reg_env, rc=0, store_root=lp.child_store_root(child, argv), attempt_id="a", log_name=srv.child_log_name(child))
    assert len(paths) == 3                                                              # one patch per registered run the child's tasks claim
    rids = {rec["payload"]["run_id"] for _p, rec, _s in _records(attempt / "registry" / "records")}
    assert rids == {routes[t]["run_id"] for t in srv.child_tasks(child)} and len(rids) == 3   # the runs of THIS child's controls, each once
    # a child carrying ONE control (a ladder rung's single-control store) claims exactly that route's run
    one = [c for c in row["children"] if isinstance(c.get("task"), dict) and len(c["task"]) == 1]
    if one:
        _a1, env1 = _child_env(srv, one[0], attempt_root=tmp_path / "attempt1", plans_root=rs / lp.PLANS_REL)
        p1 = srv.child_registry_completion(one[0], {e: env1.get(e, "") for e in (lp.RUN_IDENTITY_ENV, lp.SPOOL_ENV)}, rc=0, store_root="/s",
                                           attempt_id="a", log_name=srv.child_log_name(one[0]))
        assert len(p1) == 1


# --- a judged route's GENERATED items are filed as null-valued item_result rows at generation -----------------------------------------

CTL_RESOLVED = dict(RESOLVED, arm_name="arditi_allpos", kind="vector_all", intervention_hash="2ff9cdff091aef6f")


def _controls_store(srv, plan, rs, tmp_path, monkeypatch, *, identity_rel="registry/plans/_stores/controls/arditi_allpos__s1.json", engine="arditi"):
    """A controls CellStore built with NO recorder under the PRODUCTION-composed child env of the controls row's child naming
    `identity_rel` (the per-task identity map): the default recorder resolves from that env, exactly as the driver's store does."""
    row = _row(plan, "controls", engine=engine)
    child = [c for c in row["children"] if c.get("registry_identity_rel") == identity_rel][0]
    attempt = tmp_path / "attempt"
    _argv, env = _child_env(srv, child, attempt_root=attempt, plans_root=rs / lp.PLANS_REL)
    for k in (hooks.RUN_IDENTITY_ENV, emit.SPOOL_ENV, emit.SUBSTRATE_ENV):
        if env.get(k):
            monkeypatch.setenv(k, env[k])
    hooks._DEFAULT_NOTICED.clear()
    store = cs.CellStore(attempt / "exp" / "controls", KEY, row_constants={k: v for k, v in CTL_RESOLVED.items() if k not in ("tiers", "cell_index")})
    assert isinstance(store.recorder, hooks.StoreRecorder)
    store.validate_resume(CTL_RESOLVED)
    store.write_cell_manifest(CTL_RESOLVED)
    routes = hooks.normalise_identities(json.loads(Path(env[hooks.RUN_IDENTITY_ENV]).read_text()))
    return store, routes, attempt / "registry" / "records", env


def _run_records(spool: Path, run_id: str):
    return [rec for _p, rec, _s in _records(spool) if rec["payload"].get("run_id") == run_id]


def test_a_judged_routes_generated_items_are_filed_as_null_valued_item_results_at_generation(srv, plan, rs, tmp_path, monkeypatch, clean_switch):
    store, routes, spool, _env = _controls_store(srv, plan, rs, tmp_path, monkeypatch)
    xst = routes["judged/xstest_v2"]
    n = int(xst["n_items_planned"])
    ids = [f"xst_{i:04d}" for i in range(n)]
    before = store.rows_path.read_bytes() if store.rows_path.exists() else b""
    store.note_generated("judged/xstest_v2", ids, str(store.cell_dir / "private" / "gens_xstest_v2.jsonl"))
    assert (store.rows_path.read_bytes() if store.rows_path.exists() else b"") == before          # NO store byte: the recorder's business only
    recs = _run_records(spool, xst["run_id"])
    items = [r["payload"] for r in recs if r["record_type"] == "item_result"]
    assert len(items) == n and {i["item_id"] for i in items} == set(ids)                          # ONE per generated item, the plan's count
    assert all(i["value"] is None and i["judgment_id"] == "" and i["metric_id"] == hooks.GENERATED_METRIC_ID for i in items)
    assert all(i["extra"]["task"] == "judged/xstest_v2" and i["extra"]["stage"] == "generated" for i in items)
    assert {r["record_type"] for r in recs} >= {"run", "run_state", "item_result"}                 # the run OPENED at its generation
    assert [r["payload"]["state"] for r in recs if r["record_type"] == "run_state"] == ["running"]
    # flushed at once (the controls store never reaches on_results in-container), idempotent on a replay, silent on an unknown task
    store.note_generated("judged/xstest_v2", ids[:10], "again")
    assert len([r for r in _run_records(spool, xst["run_id"]) if r["record_type"] == "item_result"]) == n
    orb = routes["judged/or_bench_hard_1k"]
    assert _run_records(spool, orb["run_id"]) == []                                                # the sibling route: untouched
    store.note_generated("judged/nobody_registered", ["q1"], "x")
    assert len(_records(spool)) == len(recs)
    _validate_all(spool)


# --- ONE `run` payload per logical run per spool, however many recorders open it -----------------------------------------------------

def test_a_re_opened_store_files_no_second_run_payload_for_a_run_its_spool_already_holds(srv, plan, rs, tmp_path, monkeypatch, clean_switch):
    """Before this, on_generated opened the judged runs in the driver's recorder, then the judge's grading-time re-open (a fresh
    CellStore over the same cell dir) minted `run` again for xstest / or_bench -- 5 payloads for 3 runs: the SAME run_ids, a second
    `run` per re-opening instance. Now a recorder that finds the run's `run` record in the spool files a `run_state running` only:
    three routes, three `run` payloads, the generated item_results and the aggregate present."""
    store, routes, spool, env = _controls_store(srv, plan, rs, tmp_path, monkeypatch)
    n_x = int(routes["judged/xstest_v2"]["n_items_planned"]); n_o = int(routes["judged/or_bench_hard_1k"]["n_items_planned"]); n_m = int(routes["s2parity/mmlu_letter"]["n_items_planned"])
    # the DRIVER's recorder: the two judged routes' generated items (2a) -- their runs open here, once
    store.note_generated("judged/xstest_v2", [f"x{i:04d}" for i in range(n_x)], "gens_x")
    store.note_generated("judged/or_bench_hard_1k", [f"o{i:04d}" for i in range(n_o)], "gens_o")
    runs_gen = [r["payload"] for r in [rec for _p, rec, _s in _records(spool)] if r["record_type"] == "run"]
    assert sorted(r["extra"]["route"] for r in runs_gen) == ["judged/or_bench_hard_1k", "judged/xstest_v2"]
    # the GRADING-TIME re-open: a SECOND CellStore over the same cell dir (the judge's _file_first_rows), rows for all three routes
    store2 = cs.CellStore(store.sweep_dir, store.key, row_constants={k: v for k, v in CTL_RESOLVED.items() if k not in ("tiers", "cell_index")})
    assert isinstance(store2.recorder, hooks.StoreRecorder) and store2.recorder is not store.recorder
    store2.write_cell_manifest(CTL_RESOLVED)
    for i, task in enumerate(sorted(routes)):
        store2.append_row(_ctl_row(task, f"probe-{i:03d}", "over_refusal", 0.0, position_order=i))
    for i in range(n_m):
        store2.append_row(_ctl_row("s2parity/mmlu_letter", f"mmlu_{i:04d}", "acc", float(i % 2), position_order=i))
    store2.append_row(_ctl_row("s2parity/mmlu_letter", "__aggregate__", "acc_aggregate", 0.5, extra_json=json.dumps({"n": n_m})))
    store2.recorder.flush()
    recs = [rec for _p, rec, _s in _records(spool)]
    runs = [r["payload"] for r in recs if r["record_type"] == "run"]
    assert len(runs) == 3 and {r["run_id"] for r in runs} == {i["run_id"] for i in routes.values()}          # ONE `run` per logical run
    assert sorted(r["extra"]["route"] for r in runs) == sorted(routes)
    states = [r["payload"] for r in recs if r["record_type"] == "run_state"]
    assert sorted(s["run_id"] for s in states) == sorted([routes["judged/xstest_v2"]["run_id"]] * 2 + [routes["judged/or_bench_hard_1k"]["run_id"]] * 2
                                                          + [routes["s2parity/mmlu_letter"]["run_id"]])                      # the re-open says `running` again, and only that
    assert all(s["state"] == "running" for s in states)
    items = [r["payload"] for r in recs if r["record_type"] == "item_result"]
    by_run = {}
    for it in items:
        by_run.setdefault(it["run_id"], set()).add(it["item_id"])
    assert len(by_run[routes["judged/xstest_v2"]["run_id"]]) == n_x + 1 and len(by_run[routes["judged/or_bench_hard_1k"]["run_id"]]) == n_o + 1   # generated + the probe row
    assert len(by_run[routes["s2parity/mmlu_letter"]["run_id"]]) == n_m + 1 and "__aggregate__" not in by_run[routes["s2parity/mmlu_letter"]["run_id"]]
    metrics = [r["payload"] for r in recs if r["record_type"] == "run_metric"]
    assert len(metrics) == 1 and metrics[0]["metric_id"] == "acc" and metrics[0]["run_id"] == routes["s2parity/mmlu_letter"]["run_id"]
    # a THIRD opener over the same spool (another resume) still adds no `run`; a fresh spool for the same identity mints one
    store3 = cs.CellStore(store.sweep_dir, store.key, row_constants={k: v for k, v in CTL_RESOLVED.items() if k not in ("tiers", "cell_index")})
    store3.write_cell_manifest(CTL_RESOLVED)
    store3.append_row(_ctl_row("judged/xstest_v2", "probe-zzz", "over_refusal", 1.0))
    store3.recorder.flush()
    assert len([r for _p, r, _s in _records(spool) if r["record_type"] == "run"]) == 3
    monkeypatch.setenv(emit.SPOOL_ENV, str(tmp_path / "other-attempt" / "registry" / "records"))
    hooks._DEFAULT_NOTICED.clear()
    store4 = cs.CellStore(tmp_path / "other-store", KEY, row_constants={k: v for k, v in CTL_RESOLVED.items() if k not in ("tiers", "cell_index")})
    store4.write_cell_manifest(CTL_RESOLVED)
    store4.note_generated("judged/xstest_v2", ["x0000"], "g")
    assert len([r for _p, r, _s in _records(tmp_path / "other-attempt" / "registry" / "records") if r["record_type"] == "run"]) == 1
    _validate_all(spool)
    # the cluster path: the submit WRAPPER's `queued` run record already sits in the SAME spool -- the store still mints ITS `run` (the
    # hashes, the model revision, the store uri the wrapper cannot know); only a prior STORE opening (`running`) suppresses a second
    wrapper_spool = tmp_path / "wrapper-spool"
    x_ident = {k: v for k, v in routes["judged/xstest_v2"].items() if k != "run_id"}
    assert emit.write_planned_run(str(wrapper_spool), x_ident, script="sbatch_registered.sh")
    monkeypatch.setenv(emit.SPOOL_ENV, str(wrapper_spool))
    hooks._DEFAULT_NOTICED.clear()
    store5 = cs.CellStore(tmp_path / "wrapper-store", KEY, row_constants={k: v for k, v in CTL_RESOLVED.items() if k not in ("tiers", "cell_index")})
    store5.write_cell_manifest(CTL_RESOLVED)
    store5.note_generated("judged/xstest_v2", ["x0000"], "g")
    w_runs = [r["payload"] for _p, r, _s in _records(wrapper_spool) if r["record_type"] == "run"]
    assert [r["lifecycle_state"] for r in w_runs] == ["queued", "running"] and w_runs[1]["intervention_hash"] == CTL_RESOLVED["intervention_hash"]
    # the job-start REGISTRATION (emit_cli register, the Petri / misalign pending-file path) writes a `running` run with no store_key into
    # the same spool -- a store under that identity still mints ITS `run` (the mark of a prior STORE opening is extra.store_key, never a state)
    reg_spool = tmp_path / "reg-spool"
    reg_w = emit.RecordWriter(str(reg_spool), str(x_ident["campaign"]), routes["judged/xstest_v2"]["run_id"], emit.producer_block("emit_cli register"))
    assert reg_w.emit_run(dict(x_ident, run_id=routes["judged/xstest_v2"]["run_id"], lifecycle_state="running", queued_utc=emit.utc_now(), started_utc=emit.utc_now(),
                              extra=dict(x_ident.get("extra") or {}, registered_at="job_start")))
    monkeypatch.setenv(emit.SPOOL_ENV, str(reg_spool))
    hooks._DEFAULT_NOTICED.clear()
    store6 = cs.CellStore(tmp_path / "reg-store", KEY, row_constants={k: v for k, v in CTL_RESOLVED.items() if k not in ("tiers", "cell_index")})
    store6.write_cell_manifest(CTL_RESOLVED)
    store6.note_generated("judged/xstest_v2", ["x0000"], "g")
    r_runs = [r["payload"] for _p, r, _s in _records(reg_spool) if r["record_type"] == "run"]
    assert len(r_runs) == 2 and "store_key" not in (r_runs[0].get("extra") or {}) and r_runs[1]["extra"]["store_key"] == KEY
    store7 = cs.CellStore(tmp_path / "reg-store2", KEY, row_constants={k: v for k, v in CTL_RESOLVED.items() if k not in ("tiers", "cell_index")})
    store7.write_cell_manifest(CTL_RESOLVED)
    store7.note_generated("judged/xstest_v2", ["x0001"], "g")
    assert len([r for _p, r, _s in _records(reg_spool) if r["record_type"] == "run"]) == 2                    # the store's opening is the one that counts


# --- the producer's aggregate row is a run_metric, never an item -------------------------------------------------------------------

def _ctl_row(task, item, metric, value, **kw):
    base = dict(sweep="synthsweep", run_id="synthsweep.0.t0.a0", model_id=synth.MODEL_ID, model_revision=REV, arm_name="arditi_allpos",
                kind="vector_all", coeff=0.3, intervention_hash="2ff9cdff091aef6f", engine="lora_native", tier=1, task=task, task_version="s2parity_v1",
                domain="knowledge_s2parity", item_id=item, trial_id=0, seed=1, attempt=0, position_order=None, metric=metric, value=value,
                extra_json=None, harness_config_hash="0f1e2d3c4b5a6978", env_digest="99aabbccddeeff00", slurm_job_id=None,
                created_at=1755000000.0, wall_clock_s=None)
    base.update(kw)
    return cs.ResultRow(**base)


def test_the_parity_aggregate_row_lands_as_a_run_metric_and_the_item_count_equals_the_plan(srv, plan, rs, tmp_path, monkeypatch, clean_switch):
    """parity.py's `__aggregate__` row (metric acc_aggregate) was filed as an item: the mmlu union held 201 against the plan's 200 and every
    mmlu run refused admission. Filed as a run_metric in the cellstore adapter's exact shape, the item count is the plan's."""
    from sisteer.capeval.tiers import parity
    store, routes, spool, _env = _controls_store(srv, plan, rs, tmp_path, monkeypatch)
    mmlu = routes["s2parity/mmlu_letter"]
    n = int(mmlu["n_items_planned"])
    for i in range(n):
        store.append_row(_ctl_row("s2parity/mmlu_letter", f"mmlu_{i:04d}", "acc", float(i % 2), position_order=i))
    store.append_row(_ctl_row("s2parity/mmlu_letter", "__aggregate__", parity.PARITY_AGG_METRIC, 0.5,
                              extra_json=json.dumps({"n": n, "wall_clock_s": 2.86, "top_logprobs": 64}), wall_clock_s=2.86))
    store.recorder.flush()
    recs = _run_records(spool, mmlu["run_id"])
    items = [r["payload"] for r in recs if r["record_type"] == "item_result"]
    metrics = [r["payload"] for r in recs if r["record_type"] == "run_metric"]
    assert len(items) == n and {i["item_id"] for i in items} == {f"mmlu_{i:04d}" for i in range(n)}     # the plan's count, no __aggregate__
    assert len(metrics) == 1
    m = metrics[0]
    assert m["metric_id"] == "acc" and m["producer_metric_spelling"] == "acc_aggregate" and m["value"] == 0.5 and m["n"] == n
    assert m["computed_by"] == "producer" and m["judgment_id"] == "" and m["threshold"] == -1.0 and m["computation_id"] == ""
    assert m["extra"]["wall_clock_s"] == 2.86 and m["extra"]["top_logprobs"] == 64 and "n" not in m["extra"] and m["extra"]["task"] == "s2parity/mmlu_letter"
    # a judged `*_aggregate` spelling with no __aggregate__ id takes the same road; a plain item never does
    store.append_row(_ctl_row("s2parity/mmlu_letter", "__aggregate__", "xst_refused_bounds_aggregate", 0.1))
    store.append_row(_ctl_row("s2parity/mmlu_letter", "mmlu_9999", "acc", 1.0))
    store.recorder.flush()
    recs2 = _run_records(spool, mmlu["run_id"])
    assert len([r for r in recs2 if r["record_type"] == "run_metric"]) == 2 and len([r for r in recs2 if r["record_type"] == "item_result"]) == n + 1
    _validate_all(spool)


# --- the identity fields the fold cannot patch later -- substrate and the Arditi strength unit ------------------------------------

def test_every_in_container_registry_writer_declares_the_modal_substrate(srv, plan, rs, tmp_path, monkeypatch, clean_switch):
    """emit.substrate_from_env probes the site's shared roots BEFORE Modal; container `run` records once said a cluster (derived) and
    overwrote the queued record's modal. The composed child env, the rung consumer env and the standup env now DECLARE modal."""
    row = _row(plan, "pasttense")
    child = [c for c in row["children"] if c["arm"] == "none"][0]
    attempt = tmp_path / "attempt"
    _argv, env = _child_env(srv, child, attempt_root=attempt, plans_root=rs / lp.PLANS_REL)
    assert env[emit.SUBSTRATE_ENV] == "modal" and lp.SUBSTRATE_ENV == emit.SUBSTRATE_ENV
    assert lp.rung_child_env(attempt_root=str(attempt), attempt_id="a", standup_id="s", leg="pasttense")[emit.SUBSTRATE_ENV] == "modal"
    consumer = {"eval_leg": "pasttense", "arm": "consumer", "s": 0, "argv": ["x"], "concurrency_rule": "one_request_each_S_drivers"}
    ctx = lp.SpawnContext(attempt_root=str(attempt), attempt_id="a", standup_id="s", plans_root="/p", proxy_url="u", routing_table="t",
                          driver_python="py", S=4, ncells=1, bank_view="/b", base_env={})
    assert lp.resolve_spawn(consumer, ctx)[1][emit.SUBSTRATE_ENV] == "modal"                        # a child with no identity too
    assert "lp.SUBSTRATE_ENV: lp.SUBSTRATE" in DRIVER.read_text().split("standup_env = dict(env")[1].split("\n")[0]   # the standup's env
    # a host whose probes would name a cluster (a shared root the site map names, under the probe root) still records modal under the
    # child env
    probe_root = tmp_path / "host"
    (probe_root / "srv" / "shared" / os.environ.get("USER", "u")).mkdir(parents=True)
    monkeypatch.setenv(emit.PROBE_ROOT_ENV, str(probe_root))
    monkeypatch.setenv(emit.SITE_MAP_ENV, "cluster-two=cluster_b@/srv/shared/{user}")
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)                                     # a container names no Slurm cluster
    monkeypatch.setenv("USER", os.environ.get("USER", "u"))
    monkeypatch.delenv(emit.SUBSTRATE_ENV, raising=False)
    assert emit.substrate_from_env() == "cluster_b"                                                 # the probe's reading, reproduced
    for k in (hooks.RUN_IDENTITY_ENV, emit.SPOOL_ENV, emit.SUBSTRATE_ENV):
        monkeypatch.setenv(k, env[k])
    assert emit.substrate_from_env() == "modal"
    hooks._DEFAULT_NOTICED.clear()
    store = cs.CellStore(attempt / "exp" / "store", KEY, row_constants={k: v for k, v in RESOLVED.items() if k not in ("tiers", "cell_index")})
    store.validate_resume(RESOLVED)
    store.write_cell_manifest(RESOLVED)
    runs = [rec for _p, rec, _s in _records(attempt / "registry" / "records") if rec["record_type"] == "run"]
    assert len(runs) == 1 and runs[0]["payload"]["substrate"] == "modal" and runs[0]["producer"]["substrate"] == "modal"
    _validate_all(attempt / "registry" / "records")


def test_the_arditi_planned_record_carries_its_strength_unit_from_the_pins_dose_unit(rs, tmp_path, monkeypatch, clean_switch):
    for s in (1, 2, 3):
        ident = _identity_of(f"{rs / lp.PLANS_REL}/pasttense/arditi_allpos__s{s}.json")
        assert ident["extra"]["dose"]["unit"] == "raw_coefficient" and ident["strength_unit"] == "raw_coefficient"
        assert ident["strength_nominal"] == 0.3 and ident["strength_nominal_source"] == "filed" and ident["is_operating_point"] == 1
        assert emit.check_planned_run({k: v for k, v in ident.items() if k != "run_id"}) == ident["run_id"]     # strict, and the id is unchanged
    fitted = _identity_of(f"{rs / lp.PLANS_REL}/pasttense/constv2-mix-full__lora_r4_ce__s1.json")
    assert fitted["strength_unit"] == "nominal_only" and "strength_unit" not in _identity_of(f"{rs / lp.PLANS_REL}/pasttense/none__s1.json")
    # the queued record (the grading host spawn) carries it
    spool = tmp_path / "spool"
    ident = _identity_of(f"{rs / lp.PLANS_REL}/pasttense/arditi_allpos__s1.json")
    paths = hooks.queued_modal({k: v for k, v in ident.items() if k != "run_id"}, modal_app_id="a", modal_call_id="c", script="t", spool=str(spool), registry_emit=True)
    assert paths
    run = [rec["payload"] for _p, rec, _s in _records(spool) if rec["record_type"] == "run"][0]
    assert run["strength_unit"] == "raw_coefficient" and run["substrate"] == "modal" and run["run_id"] == ident["run_id"]
    # an Arditi pin whose dose unit is not a registry strength unit refuses the plan set by name (never a silent `unverified`)
    import registry_plans as RP2
    inp = RP2.load_inputs(campaign=CAMPAIGN, w_sha=W, policy=REAL_POLICY)
    bad = dict(inp.arditi_pins)
    bad[RP2.MODEL_KEY] = {**bad[RP2.MODEL_KEY], RP2.ARDITI_STAGED_ID: {**bad[RP2.MODEL_KEY][RP2.ARDITI_STAGED_ID], "dose_unit": "furlongs"}}
    import dataclasses as _dc
    inp_bad = _dc.replace(inp, arditi_pins=bad)                                                     # Inputs is a dataclass
    with pytest.raises(RP2.RegistryPlanRefused, match="furlongs"):
        RP2.build(inp_bad, ["pasttense"])


# --- a row that is not DONE ends every child run in the registry ------------------------------------------------------------------

def test_an_attempt_that_is_not_done_records_recovery_evidence_for_every_child_run_and_never_a_logical_state(srv, plan, rs, tmp_path, clean_switch):
    """The LOGICAL terminal state is the grading host HOLD's; an attempt records ONE run_patch of
    attempt_terminal evidence per run -- and no run_state, whatever its refusal, so registry_publish can never ship a false failure."""
    row = _row(plan, "pasttense")
    attempt = tmp_path / "attempt"
    uri = f"/vol/seedrep/{W[:7]}/exp/seedrep_battery_modal/pasttense/s1/shard00/attempts/att-1/RESULT"
    res = srv.registry_attempt_terminal(row, plans_root=str(rs / lp.PLANS_REL), attempt_root=str(attempt), status="DRIVERS_FAILED", attempt_id="att-1",
                                        evidence_uri=uri, code_sha=W, standup_id=row["standup_id"], accepted_holder=None, note="row DRIVERS_FAILED: 3 of 48 children failed")
    assert res["runs"] == 8 and len(res["records"]) == 8 and res["problems"] == [] and res["state_hint"] == "failed" and res["closure_trouble"] is False
    recs = _records(attempt / "registry" / "records")
    assert len(recs) == 8 and all(rec["record_type"] == "run_patch" for _p, rec, _s in recs)       # ONE per run -- and NEVER a run_state
    ids = {_identity_of(p)["run_id"] for p in set(srv.registry_identity_files(row, plans_root=str(rs / lp.PLANS_REL)).values())}
    assert {rec["payload"]["run_id"] for _p, rec, _s in recs} == ids
    for _p, rec, _s in recs:
        at = rec["payload"]["fields"]["extra_json"]["attempt_terminal"]
        assert at["attempt_id"] == "att-1" and at["status"] == "DRIVERS_FAILED" and at["state_hint"] == "failed" and at["evidence_uri"] == uri
        assert at["accepted_holder"] is None and at["closure_trouble"] is False and at["note"].startswith("row DRIVERS_FAILED")
        assert rec["producer"]["code_sha"] == W and rec["producer"]["substrate"] == "modal" and rec["producer"]["script"] == "scripts/modal/seedrep_served.py attempt-terminal"
    _validate_all(attempt / "registry" / "records")
    # a stop-Dict drain is hinted cancelled; a missing identity file is NAMED while the rest are filed; the controls-arditi container's 9 runs
    res2 = srv.registry_attempt_terminal(row, plans_root=str(rs / lp.PLANS_REL), attempt_root=str(tmp_path / "a2"), status="DRIVERS_FAILED", attempt_id="att-2",
                                         evidence_uri=uri, code_sha=W, standup_id=row["standup_id"], accepted_holder=None, cancelled=True)
    assert res2["state_hint"] == "cancelled" and all(rec["payload"]["fields"]["extra_json"]["attempt_terminal"]["state_hint"] == "cancelled"
                                                     for _p, rec, _s in _records(tmp_path / "a2" / "registry" / "records"))
    ghost = {"eval_leg": "pasttense", "arm": "ghost", "s": 1, "registry_identity_rel": "registry/plans/pasttense/ghost__s1.json"}
    part = {"standup_id": "x", "children": [c for c in row["children"] if c["arm"] == "none"] + [ghost]}
    res3 = srv.registry_attempt_terminal(part, plans_root=str(rs / lp.PLANS_REL), attempt_root=str(tmp_path / "a4"), status="X", attempt_id="a", evidence_uri=uri,
                                         code_sha=W, standup_id="x", accepted_holder=None)
    assert res3["runs"] == 1 and len(res3["problems"]) == 1 and "ghost" in res3["problems"][0]
    ctl = _row(plan, "controls", engine="arditi")
    res4 = srv.registry_attempt_terminal(ctl, plans_root=str(rs / lp.PLANS_REL), attempt_root=str(tmp_path / "a5"), status="X", attempt_id="a", evidence_uri=uri,
                                         code_sha=W, standup_id=ctl["standup_id"], accepted_holder=None)
    assert res4["runs"] == 9 and res4["problems"] == []


def test_a_publication_failure_after_acceptance_is_closure_trouble_never_a_logical_failure(srv, tmp_path, clean_switch):
    """finish_attempt() elected the attempt (accepted.put won), then the closing publish raised PublicationFailed ->
    fail("PUBLICATION_FAILED"), which filed run_state failed; the accepted, COMPLETE attempt's spool is what registry_publish ships,
    and the reducer kept the false finished_utc past the later `generated`. Now: the accepted holder is this attempt ->
    closure_trouble evidence, no run_state anywhere -- and the same spool, folded, leaves the run exactly where the queue put it."""
    rs, row, plan, ident, reg = _hold_world(tmp_path)
    attempt = tmp_path / "attempt"
    spool = attempt / lp.SPOOL_REL
    assert hooks.queued_modal({k: v for k, v in ident.items() if k != "run_id"}, modal_app_id="app", modal_call_id="fc-1", script="t", spool=str(spool), registry_emit=True)
    res = srv.registry_attempt_terminal(row, plans_root=str(rs / lp.PLANS_REL), attempt_root=str(attempt), status="PUBLICATION_FAILED", attempt_id="att-1",
                                        evidence_uri=f"{plan['results_root']}/pasttense/s1/shard00/attempts/att-1/RESULT", code_sha=W, standup_id=row["standup_id"],
                                        accepted_holder="att-1", note="row refused: PUBLICATION_FAILED")
    assert res["runs"] == 1 and res["closure_trouble"] is True and res["accepted_holder"] == "att-1"
    recs = _records(spool)
    assert [rec["record_type"] for _p, rec, _s in recs if rec["payload"].get("run_id") == ident["run_id"] and rec["record_type"] == "run_state"] == [] or \
        all(rec["payload"]["state"] == "queued" for _p, rec, _s in recs if rec["record_type"] == "run_state")     # the queue's own state only; NO failed
    patches = [rec for _p, rec, _s in recs if rec["record_type"] == "run_patch" and "attempt_terminal" in (rec["payload"].get("fields") or {}).get("extra_json", {})]
    assert len(patches) == 1 and patches[0]["payload"]["fields"]["extra_json"]["attempt_terminal"]["closure_trouble"] is True
    _validate_all(spool)
    # the accepted attempt's spool IS what publication ships: fold it -- the run stays as queued, never failed, finished_utc unset
    import shutil
    shutil.copytree(spool / synth.CAMPAIGN, reg / "records" / synth.CAMPAIGN)
    runs, events, _n = _fold(reg)
    assert len(runs) == 1 and runs[0][0] == "queued" and runs[0][1] is None and {e[0] for e in events} == {"queued"}
    # and the grading host's own view never HOLDS an accepted key (apply_reattempt_policy skips a call whose key is accepted): no logical state either
    ledger = _LedgerDict()
    calls = [{"call_id": "fc-1", "standup_id": row["standup_id"], "accepted_key": row["accepted_key"], "control_key": row["control_key"], "wave_uuid": WAVE,
              "launch_plan_sha": plan["launch_plan_sha"]}]
    snap = {"fc-1": {"done": True, "summary": {"status": "PUBLICATION_FAILED", "attempt_id": "att-1", "attempt_root": "/vol/x"}, "error": "RESULT rc=1 status=PUBLICATION_FAILED",
                     "accepted": "att-1"}}
    out = srv.apply_reattempt_policy(plan, rs, calls=calls, snapshot=snap, rows_by_sid={row["standup_id"]: row}, ledger=ledger, spawn=lambda r, i: "never",
                                     log=lambda *a: None, registry_spool=str(reg / "records"))
    assert out == [] and srv.reattempt_ledger_get(ledger, row["standup_id"])["held"] is False
    assert _fold(reg)[0][0][0] == "queued"


def test_the_row_records_attempt_evidence_at_result_and_in_every_refusal_but_a_duplicate_execution_reading_the_holder_first(srv):
    src = DRIVER.read_text()
    i_result = src.index("(root / RESULT_FILE).write_text(f\"{utc()} status={status} rc={0 if not failed else 1}")
    i_term = src.index("registry_attempt_terminal(\n                row, plans_root=ctx[\"plans_root\"]", i_result)
    i_sync = src.index("_sync_all(syncers, lock, \"final\")", i_result)
    assert i_result < i_term < i_sync                                                        # filed BEFORE the final sync ships the spool
    assert "cancelled=drained_by_stop" in src[i_term:i_sync] and "drained_by_stop = True" in src.split("STOP sentinel present")[1][:200]
    i_fail = src.index("    def fail(status: str, **more) -> dict:")
    i_holder = src.index("holder = accepted.get(akey)", i_fail)
    i_fail_term = src.index("registry_attempt_terminal(row, plans_root=ctx[\"plans_root\"], attempt_root=str(root), status=status", i_fail)
    i_publish = src.index("final_status = terminal_publish(root, status=status", i_fail)
    assert i_fail < i_holder < i_fail_term < i_publish and "if status not in NO_TERMINAL_STATUSES:" in src[i_fail:i_holder]
    assert "accepted_holder=holder" in src[i_fail_term:i_publish]
    assert set(srv.NO_TERMINAL_STATUSES) == {srv.REATTEMPT_DUPLICATE, lp.FANOUT_DUPLICATE} and srv.TERMINAL_STATES == ("failed", "cancelled")
    # the attempt-side helper emits patches only: the logical run_state is the grading host HOLD emitter's alone
    body = src[src.index("def registry_attempt_terminal("):src.index("def count_spool_records(")]
    assert "emit_patch(" in body and "emit_state(" not in body
    hold = src[src.index("def registry_hold_terminal_states("):src.index("def count_spool_records(")]
    assert "emit_state(" in hold


# --- the completion path over a synthetic COMPLETED controls-arditi attempt -------------------------------------------------------

import logical_run_complete as LRC  # noqa: E402
import registry_publish as PUB  # noqa: E402

ATT = "01TESTTASKID0000000000000-abcdef012345"


def test_the_completion_path_admits_and_publishes_every_run_of_a_completed_controls_arditi_attempt(srv, plan, rs, tmp_path, monkeypatch, clean_switch):
    """Before this, all 9 runs of the completed controls-arditi attempt REFUSED (six judged runs with no item rows and no run state;
    three mmlu runs at 201 items against 200). Built through the REAL emitters -- the CellStore under the production-composed arditi
    child env, the judged routes' generated items, mmlu's rows and its aggregate -- logical_run_complete ADMITS every route at the
    plan's count and registry_publish publishes the attempt; the CLI re-run is a clean no-op."""
    attempts = tmp_path / "attempts"
    a = attempts / "controls" / "arditi" / "0" / "attempts" / ATT
    a.mkdir(parents=True)
    (a / PUB.ATTEMPT_FILE).write_text(json.dumps({"attempt_id": ATT, "leg": "controls", "view": "arditi", "shard": 0, "stage": "base", "W": W,
                                                  "wave_uuid": WAVE, "superseded_by": None, "call_id": f"fc-{ATT}", "task_id": "ta-1"}))
    row = _row(plan, "controls", engine="arditi")
    child = [c for c in row["children"] if c.get("registry_identity_rel") == "registry/plans/_stores/controls/arditi_allpos__s1.json"][0]
    _argv, env = _child_env(srv, child, attempt_root=a, plans_root=rs / lp.PLANS_REL)
    for k in (hooks.RUN_IDENTITY_ENV, emit.SPOOL_ENV, emit.SUBSTRATE_ENV):
        monkeypatch.setenv(k, env[k])
    hooks._DEFAULT_NOTICED.clear()
    routes = hooks.normalise_identities(json.loads(Path(env[hooks.RUN_IDENTITY_ENV]).read_text()))
    store = cs.CellStore(a / "exp" / "controls", KEY, row_constants={k: v for k, v in CTL_RESOLVED.items() if k not in ("tiers", "cell_index")})
    store.validate_resume(CTL_RESOLVED)
    store.write_cell_manifest(CTL_RESOLVED)
    for task in ("judged/xstest_v2", "judged/or_bench_hard_1k"):                                    # 2a: the judged routes' generated items
        n = int(routes[task]["n_items_planned"])
        store.note_generated(task, [f"{task.rsplit('/', 1)[1]}_{i:04d}" for i in range(n)], str(store.cell_dir / "private" / "gens.jsonl"))
    n_mmlu = int(routes["s2parity/mmlu_letter"]["n_items_planned"])
    for i in range(n_mmlu):                                                                          # 2b: mmlu's rows + the aggregate
        store.append_row(_ctl_row("s2parity/mmlu_letter", f"mmlu_{i:04d}", "acc", float(i % 2), position_order=i))
    store.append_row(_ctl_row("s2parity/mmlu_letter", "__aggregate__", "acc_aggregate", 0.5, extra_json=json.dumps({"n": n_mmlu})))
    store.recorder.flush()
    (a / PUB.COMPLETE_FILE).write_text(json.dumps({"attempt_id": ATT, "utc": "2026-01-01T00:00:00.000Z"}))
    spool_recs = _records(a / lp.SPOOL_REL)
    assert all(rec["payload"].get("substrate", "modal") == "modal" for _p, rec, _s in spool_recs if rec["record_type"] == "run")   # 3
    _validate_all(a / lp.SPOOL_REL)
    accepted = tmp_path / "accepted.json"
    accepted.write_text(json.dumps({f"{W[:7]}/{WAVE}/controls/arditi/base/0": ATT}))
    grading_host, union = tmp_path / "grading_host-spool", tmp_path / "union"
    plan_file = rs / lp.PLANS_REL / "_stores" / "controls" / "arditi_allpos__s1.json"
    out = LRC.complete_plan_file(plan_file, accepted_path=accepted, attempt_roots=[attempts], spool=grading_host, verdict_dir=union, items_source="spool")
    assert len(out) == 3 and [v["verdict"] for v in out] == [LRC.VERDICT_ADMITTED] * 3, [(v["route"], v["verdict"], v.get("reasons")) for v in out]
    assert all(v["n_union"] == v["expected_n"] == int(routes[v["route"]]["n_items_planned"]) for v in out), [(v["route"], v["n_union"], v["expected_n"]) for v in out]
    assert {v["route"] for v in out} == set(routes) and all(v["view"] == "arditi" and v["container"] == "controls" for v in out)
    gen = [rec for _p, rec, _s in _records(grading_host) if rec["record_type"] == "run_state" and rec["payload"]["state"] == "generated"]
    assert {g["payload"]["run_id"] for g in gen} == {i["run_id"] for i in routes.values()}          # generated ONCE per logical run
    rep = PUB.publish(PUB.discover_attempts([attempts]), PUB.load_accepted(accepted), inbox=tmp_path / "inbox", host="seedrep-modal", union_dir=union)
    assert rep.published_attempt_ids() == {ATT} and len(rep.published) == len(list(R.iter_record_files(str(a / lp.SPOOL_REL)))) > 0 and rep.duplicates == []
    # the CLI, as the operator steps run it: a second pass is a clean no-op, exit 0
    rc = LRC.main(["--plan", str(plan_file), "--accepted", str(accepted), "--attempts", str(attempts), "--spool", str(grading_host), "--verdict-dir", str(union),
                   "--items-source", "spool"])
    assert rc == 0
    assert len([rec for _p, rec, _s in _records(grading_host) if rec["record_type"] == "run_state" and rec["payload"]["state"] == "generated"]) == 3
    _validate_all(grading_host)


# --- a HELD standup's LOGICAL terminal state reaches the registry -- through publication and folding -------------------------------

from sisteer.registry import db as DB  # noqa: E402
from sisteer.registry import harvest as H  # noqa: E402


class _LedgerDict:
    """modal.Dict's atomic primitive, in memory (the policy test's FakeDict, private to this module)."""

    def __init__(self) -> None:
        self.d: dict = {}

    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self.d:
            return False
        self.d[key] = value
        return True

    def get(self, key, default=None):
        return self.d.get(key, default)

    def pop(self, key, default=None):
        return self.d.pop(key, default)

    def items(self):
        return list(self.d.items())


def _synth_identity(run_key: str, leg: str = "pasttense") -> dict:
    """A planned-run identity over synth's SEEDED dimensions (so the fold accepts it), in the producer's file shape (run_id + extra.container)."""
    ident = {k: v for k, v in synth.run_payload(run_key, i=1).items() if k not in ("lifecycle_state", "queued_utc", "store_uri")}
    ident.update(substrate="modal", substrate_source="recorded", seed_base=1)
    ident["extra"] = {"leg": leg, "container": leg, "arm": "none", "arm_kind": "none", "s": 1, "view": "s1", "served_key": "none", "seed_rule": "literal_training_seed"}
    return ident


def _hold_world(tmp_path):
    """A grading host run-state root with ONE synthetic identity file two shard children share, a plan-shaped row, and a registry root whose
    `records/` is the grading host spool the harvester ingests."""
    rs = tmp_path / "RS"
    ident = _synth_identity(f"seedrep_battery:{W[:7]}:pasttense:s1")
    rel = "registry/plans/pasttense/none__s1.json"
    (rs / rel).parent.mkdir(parents=True)
    (rs / rel).write_text(json.dumps(ident, indent=1, sort_keys=True))
    sid = "pasttense__s1__base__shard00"
    row = {"standup_id": sid, "leg": "pasttense", "engine": "s1", "stage": "base", "shard": 0, "kind": "base",
           "control_key": f"{W[:7]}/{WAVE}/pasttense/s1/base/0", "accepted_key": f"{W[:7]}|{WAVE}|pasttense|s1|base|0",
           "children": [{"eval_leg": "pasttense", "arm": "none", "s": 1, "goal_shard": k, "goal_nshards": 2, "registry_identity_rel": rel} for k in range(2)]}
    plan = {"W": W, "W7": W[:7], "wave_uuid": WAVE, "launch_plan_sha": "p" * 64, "results_root": f"/vol/seedrep/{W[:7]}/exp/seedrep_battery_modal",
            "standups": [row], "rungs": []}
    reg = tmp_path / "reg"
    H.ensure_layout(str(reg))
    R.write_records(str(reg / "records" / "dims" / "000001-dims.jsonl"), synth.seed_dimensions())
    return rs, row, plan, ident, reg


def _fold(reg: Path):
    rep = H.ingest_once(str(reg), workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    conn = DB.connect(DB.db_path(str(reg)), readonly=True)
    try:
        run = conn.execute("SELECT lifecycle_state, finished_utc, queued_utc FROM run").fetchall()
        events = conn.execute("SELECT state, event_utc, evidence_kind, evidence_uri FROM run_state_event ORDER BY event_utc").fetchall()
        n_items = conn.execute("SELECT COUNT(*) FROM item_result").fetchone()[0]
    finally:
        conn.close()
    return [tuple(r) for r in run], [tuple(e) for e in events], n_items


def test_a_held_standups_runs_end_failed_in_the_registry_through_the_hold_decision_not_the_attempts_spool(srv, tmp_path, clean_switch):
    """registry_publish copies ACCEPTED, COMPLETE attempts only, so a failed attempt's private-spool run_state never reached the
    registry and its runs stayed `queued`. The HOLD -- the durable ledger's decision after the one re-attempt -- now mints the
    LOGICAL run_state on the grading host, metadata only, folded here through the real harvester into a real registry DB."""
    rs, row, plan, ident, reg = _hold_world(tmp_path)
    spool = str(reg / "records")
    # the grading host spawn queued the run: the row the fold will end
    assert hooks.queued_modal({k: v for k, v in ident.items() if k != "run_id"}, modal_app_id="app", modal_call_id="fc-1", script="t", spool=spool, registry_emit=True)
    sid = row["standup_id"]
    rows = {sid: row}
    ledger = _LedgerDict()
    spawned: list = []
    calls = [{"call_id": "fc-1", "standup_id": sid, "accepted_key": row["accepted_key"], "control_key": row["control_key"], "wave_uuid": WAVE, "launch_plan_sha": plan["launch_plan_sha"]}]
    root1 = f"{plan['results_root']}/pasttense/s1/shard00/attempts/att-1"
    snap = {"fc-1": {"done": True, "summary": {"status": "LOST_CONTAINER", "attempt_id": "att-1", "attempt_root": root1}, "error": None, "accepted": None}}
    out = srv.apply_reattempt_policy(plan, rs, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=lambda r, i: (spawned.append(r["standup_id"]), "fc-re1")[1],
                                     log=lambda *a: None, registry_spool=spool)
    assert [d["action"] for d in out] == ["reattempt"] and spawned == [sid]
    runs0 = _fold(reg)[0]
    assert len(runs0) == 1 and runs0[0][:2] == ("queued", None) and runs0[0][2]                  # ONE re-attempt pending: nothing terminal yet
    # the re-attempt fails too -> HOLD -> the logical runs end, with job_log evidence at the LAST attempt's RESULT
    calls.append({**calls[0], "call_id": "fc-re1"})
    root2 = f"{plan['results_root']}/pasttense/s1/shard00/attempts/att-2"
    snap["fc-re1"] = {"done": True, "summary": {"status": "DRIVERS_FAILED", "attempt_id": "att-2", "attempt_root": root2}, "error": "RESULT rc=1 status=DRIVERS_FAILED", "accepted": None}
    lines: list = []
    out = srv.apply_reattempt_policy(plan, rs, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=lambda r, i: "never", log=lines.append, registry_spool=spool)
    assert [d["action"] for d in out] == ["hold"] and spawned == [sid]
    entry = srv.reattempt_ledger_get(ledger, sid)
    ts = entry[srv.LEDGER_TERMINAL_KEY]
    held_ms = srv.registry_utc_ms(entry["held_at"])
    assert held_ms == entry["held_at"][:-1] + ".000Z" and srv.registry_utc_ms(held_ms) == held_ms and srv.registry_utc_ms("garbage") is None
    assert entry["held"] and ts["state"] == "failed" and ts["runs"] == 1 and ts["problems"] == [] and ts["event_utc"] == held_ms     # ONE run: the two shard children share it
    assert ts["evidence_uri"] == f"{root2}/RESULT" and out[0][srv.LEDGER_TERMINAL_KEY] == ts
    assert any(l.startswith(f"[registry] HOLD {sid}: 1 run(s) -> run_state failed") for l in lines)
    recs = _records(reg / "records" / synth.CAMPAIGN)
    states = [rec for _p, rec, _s in recs if rec["record_type"] == "run_state" and rec["payload"]["state"] == "failed"]
    assert len(states) == 1 and states[0]["payload"]["evidence_kind"] == "job_log" and states[0]["payload"]["evidence_uri"] == f"{root2}/RESULT"
    assert states[0]["payload"]["event_utc"] == held_ms and states[0]["producer"]["script"] == srv.HOLD_TERMINAL_SCRIPT and states[0]["producer"]["code_sha"] == W
    assert "D73 HOLD after call fc-re1" in states[0]["payload"]["note"] and {rec["record_type"] for _p, rec, _s in recs} <= {"run", "run_state", "run_patch"}   # metadata only
    _validate_all(reg / "records" / synth.CAMPAIGN)
    runs, events, n_items = _fold(reg)
    assert len(runs) == 1 and runs[0][:2] == ("failed", held_ms) and runs[0][2] == runs0[0][2] and n_items == 0   # FOLDED: failed, finished_utc stamped, queued kept
    assert {(e[0], e[2]) for e in events} == {("queued", "submit_wrapper"), ("failed", "job_log")} and len(events) == 2
    assert [e for e in events if e[0] == "failed"][0][3] == f"{root2}/RESULT"
    # idempotent: the same pass again decides and emits nothing; a HOLD that lost its emission (a pass that died between the two ledger
    # writes) recovers onto the SAME held_at, and the fold still holds ONE terminal event
    assert srv.apply_reattempt_policy(plan, rs, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=lambda r, i: "never", log=lambda *a: None, registry_spool=spool) == []
    entry.pop(srv.LEDGER_TERMINAL_KEY)
    srv.reattempt_ledger_put(ledger, entry)
    assert srv.apply_reattempt_policy(plan, rs, calls=calls, snapshot=snap, rows_by_sid=rows, ledger=ledger, spawn=lambda r, i: "never", log=lambda *a: None, registry_spool=spool) == []
    entry2 = srv.reattempt_ledger_get(ledger, sid)
    assert entry2[srv.LEDGER_TERMINAL_KEY]["runs"] == 1 and entry2[srv.LEDGER_TERMINAL_KEY]["event_utc"] == held_ms
    runs2, events2, _ = _fold(reg)
    assert runs2 == runs and len([e for e in events2 if e[0] == "failed"]) == 1
    # a cancelled wave ends its runs `cancelled`; an attempt that left no verdict has no RESULT to point at, and the note says so
    assert srv.hold_terminal_state("CANCELLED_BY_STOP") == "cancelled" and srv.hold_terminal_state("DRIVERS_FAILED") == "failed" and srv.hold_terminal_state(None) == "failed"
    res = srv.registry_hold_terminal_states(row, rs=rs, plan=plan, entry={"held_at": "2026-01-02T00:00:00.000Z", "hold_reason": "r"}, call_id="fc-x",
                                            status=None, error="OutputExpiredError: the attempt left no verdict", summary=None, spool=str(tmp_path / "s2"))
    assert res["runs"] == 1 and res["evidence_uri"] == "" and "left no RESULT" in _records(tmp_path / "s2")[0][1]["payload"]["note"]
    # no plans under RS: nothing filed, said by name; no spool: said
    none = srv.registry_hold_terminal_states(row, rs=tmp_path / "noRS", plan=plan, entry={"held_at": "2026-01-02T00:00:00.000Z"}, call_id="c", status="X", error=None, summary=None, spool=str(tmp_path / "s3"))
    assert none["runs"] == 0 and len(none["problems"]) == 2 and not (tmp_path / "s3").exists()


# --- base and every rung share ONE ladder instrument ------------------------------------------------------------------------------

def test_every_ladder_legs_base_identity_carries_the_shared_ladder_instrument_never_a_per_cap_one(tmp_path):
    """The registry must never show a ladder as a single-cap instrument. `instrument_fields` mints ONE instrument per (leg, eval) --
    token_cap_ladder_json = the declared ladder, token_cap_tokens = the ladder's CEILING -- so a base run at the first cap and a rung
    at 8192 share the instrument id and differ on the RUN's token_cap_tokens / token_cap_rung_index. Pinned for every ladder leg the
    producer builds. (The rung RUN identity itself, its escalated_from / escalation record and the queue-sized expected_n are not
    covered here: the rung consumer writes no store today.)"""
    inp = RP.load_inputs(campaign=CAMPAIGN, w_sha=W, policy=REAL_POLICY)
    ladder_legs = [name for name in RP.LEG_NAMES if len(RP.LEGS[name].token_cap_ladder) > 1 and RP.LEGS[name].store in ("static", "controls", "gsm8k")]
    assert {"pasttense", "codeattack", "renellm", "grok420", "xstest", "or_bench", "gsm8k"} <= set(ladder_legs), ladder_legs
    ps = RP.build(inp, sorted(set(ladder_legs) | {"mmlu"}))                  # the controls MAP needs all three controls legs; mmlu's ladder is one rung
    out = tmp_path / "plans"
    RP.write_plan_set(ps, out)
    seen_instruments = set()
    for name in ladder_legs:
        leg = RP.LEGS[name]
        ladder = list(leg.token_cap_ladder)
        files = [rel for rel in ps.files if rel.startswith(f"{name}/")]
        assert files, name
        idents = [i for rel in files for i in RP.identities_of(ps.files[rel]).values()]
        for ident in idents:
            assert ident["token_cap_ladder_json"] == ladder and ident["token_cap_tokens"] == int(leg.token_cap_tokens) == ladder[0]
            assert ident["token_cap_rung_index"] == 0                                                   # the BASE run: first rung
            inst = json.loads((out / RP.INSTRUMENTS_DIR / f"{ident['instrument_id']}.json").read_text())
            assert inst["fields"]["token_cap_ladder_json"] == ladder and inst["fields"]["token_cap_tokens"] == ladder[-1]   # the LADDER instrument: ceiling + ladder
            assert identity.instrument_id(inst["fields"]) == ident["instrument_id"]
            seen_instruments.add((name, ident["eval_id"], ident["instrument_id"]))
        # ONE instrument per (leg, eval) across every arm and seed -- a rung at any cap shares it and differs on the run's cap fields only
        per_eval = {}
        for _leg, ev, iid in seen_instruments:
            if _leg == name:
                per_eval.setdefault(ev, set()).add(iid)
        assert all(len(v) == 1 for v in per_eval.values()), per_eval
        # a rung's identity would differ from the base's ONLY on token_cap_tokens / token_cap_rung_index (and its run key): the instrument id is
        # a pure function of the instrument fields, which name the whole ladder, not a cap
        rung_fields = dict(inst["fields"], token_cap_tokens=ladder[-1])
        assert identity.instrument_id(rung_fields) == ident["instrument_id"]


# --- the served-runs keys are the ENDPOINT's harness keys, against a REAL endpoint ---------------------------------------------------

S1_ENDPOINT = REPO / "tests" / "fixtures" / "seedrep_f_l" / "S1" / "endpoint.json"      # the real S1 standup's READY record, harness-keyed


def test_served_runs_keys_are_the_real_endpoints_harness_keys_so_every_fitted_runs_adapter_facts_are_recorded(srv, plan, rs, tmp_path, clean_switch):
    """The plan-spelled `extra.served_key` never matched `endpoint["adapters"]`, so the hook emitted engine facts and silently omitted
    the adapter fingerprint / intervention-hash check for all five fitted routes. Against the REAL S1 endpoint (independent of the
    runs map): the fitted keys ARE its adapter keys, every fitted patch carries the fingerprint and the hash, and no run id changed."""
    endpoint = json.loads(S1_ENDPOINT.read_text())
    assert set(endpoint["adapters"]) == set(endpoint["served_positions"]) and len(endpoint["adapters"]) == 5
    row = _row(plan, "pasttense")
    runs = srv.registry_runs_for_row(row, plans_root=str(rs / lp.PLANS_REL))
    fitted_keys = {k for k in runs if k not in ("none",) and not k.startswith("prompted__")}
    assert fitted_keys == set(endpoint["adapters"]), (sorted(fitted_keys), sorted(endpoint["adapters"]))
    assert set(runs) - fitted_keys == {"none", "prompted__constitution", "prompted__safety"}
    # the run ids are the identity files' own, untouched by the key translation
    idents = {p: _identity_of(p) for p in set(srv.registry_identity_files(row, plans_root=str(rs / lp.PLANS_REL)).values())}
    assert {r["run_id"] for r in runs.values()} == {i["run_id"] for i in idents.values()}
    by_plan_key = {i["extra"]["served_key"]: i["run_id"] for i in idents.values()}
    assert all(runs[lp.harness_id(pk)]["run_id"] == rid for pk, rid in by_plan_key.items())
    # the hook off the file + the REAL endpoint: one patch per run; the five fitted carry the adapter fingerprint AND the hash to verify
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    srv.write_registry_runs(row, plans_root=str(rs / lp.PLANS_REL), serve_dir=serve_dir)
    spool = tmp_path / "spool"
    paths = hooks.emit_serving_patches(endpoint, out_dir=str(serve_dir), spool=str(spool), registry_emit=True)
    assert len(paths) == 8
    patches = {rec["payload"]["run_id"]: rec["payload"] for _p, rec, _s in _records(spool) if rec["record_type"] == "run_patch"}
    for key in fitted_keys:
        pl = patches[runs[key]["run_id"]]
        assert pl["fields"]["adapter_fingerprint"], key
        assert pl["verify"]["intervention_hash"] == endpoint["adapters"][key]["petri_intervention_hash"], key
    for key in ("none", "prompted__constitution", "prompted__safety"):
        pl = patches[runs[key]["run_id"]]
        assert "adapter_fingerprint" not in pl["fields"] and "intervention_hash" not in pl["verify"]
    assert all(pl["fields"]["served_max_num_seqs"] == endpoint["attested_max_num_seqs"] and pl["verify"]["serving_path"] == "lora_native"
               for pl in patches.values())
    _validate_all(spool)


def test_the_arditi_engines_served_key_is_its_arm_label_and_an_unmapped_key_refuses_by_name(srv, plan, rs, tmp_path, clean_switch):
    row = _row(plan, "pasttense", engine="arditi")
    runs = srv.registry_runs_for_row(row, plans_root=str(rs / lp.PLANS_REL))
    ident = _identity_of(f"{rs / lp.PLANS_REL}/pasttense/arditi_allpos__s1.json")
    assert ident["producer_object_key_legacy"] == "ar_c0p3__arditi_allpos" and ident["extra"]["served_key"] == "arditi_allpos"
    assert "arditi_allpos" in runs and "ar_c0p3__arditi_allpos" not in runs           # the ARM LABEL the engine serves under, never the staged id
    assert lp.harness_id("arditi_allpos") == "arditi_allpos"
    arditi = runs["arditi_allpos"]
    arditi = arditi if isinstance(arditi, list) else [arditi]
    assert {r["run_id"] for r in arditi} == {_identity_of(f"{rs / lp.PLANS_REL}/pasttense/arditi_allpos__s{s}.json")["run_id"] for s in (1, 2, 3)}
    # KNOWN-BAD: an identity whose served key is outside the lane's table refuses the row by name -- never a silent rename
    bad_root = tmp_path / "plans"
    src = Path(f"{rs / lp.PLANS_REL}/pasttense/none__s1.json")
    d = json.loads(src.read_text())
    d["extra"]["served_key"] = "not_a_lane_cell"
    (bad_root / "pasttense").mkdir(parents=True)
    (bad_root / "pasttense" / "none__s1.json").write_text(json.dumps(d))
    one = {"standup_id": "x", "children": [{"eval_leg": "pasttense", "arm": "none", "s": 1, "registry_identity_rel": "registry/plans/pasttense/none__s1.json"}]}
    with pytest.raises(ValueError, match="not_a_lane_cell.*no harness translation"):
        srv.registry_runs_for_row(one, plans_root=str(bad_root))


# --- a COLD app interpreter on a non-ladder row ----------------------------------------------------------------------------------

class _ColdAppInterpreter(importlib.abc.MetaPathFinder):
    """The Modal app interpreter's world: `sisteer` is importable IFF the staged tree's `src` is on `sys.path` (the image's system
    python carries no sisteer; a driver child's PYTHONPATH never reaches the parent). Installed at the head of `sys.meta_path`."""

    def __init__(self, src: str) -> None:
        self.src = src

    def find_spec(self, name, path=None, target=None):
        if (name == "sisteer" or name.startswith("sisteer.")) and self.src not in sys.path:
            raise ModuleNotFoundError("No module named 'sisteer'")
        return None


@pytest.mark.parametrize("leg", ["bon", "crescendo"])
def test_a_cold_app_interpreter_on_a_non_ladder_row_passes_the_preflight_only_after_the_tree_src_is_on_its_path(srv, plan, rs, tmp_path, leg, clean_switch):
    """BoN / multi-turn / Petri rows are neither producer nor rung, so the staged tree's src reached the app interpreter only for the
    ladder rows; the preflight imported sisteer.registry.hooks in the app process and refused REGISTRY_EMISSION_OFF with
    ModuleNotFoundError while the driver probe (its own venv) said ON. The row now puts the src on sys.path for EVERY row first."""
    row = _row(plan, leg)
    # the app's own predicates (`_run_standup_row`): neither a rung nor a producer row -- the rows the old conditional skipped
    is_rung = row["kind"] == "rung"
    producer_row = bool(row.get("ladder")) and (row.get("ladder") or {}).get("cap_next") is not None
    assert not is_rung and not producer_row and row["stage"] == lp.STAGE_BASE
    src = str(REPO / "src")
    base = {"PATH": os.environ.get("PATH", "/usr/bin")}
    driver_probe = lambda env: srv.registry_probe(sys.executable, dict(env, PYTHONPATH=src, PATH=base["PATH"]), tree=REPO)   # the DRIVER venv: its own path, ON
    saved_mods = {k: v for k, v in sys.modules.items() if k == "sisteer" or k.startswith("sisteer.")}
    saved_path = list(sys.path)
    finder = _ColdAppInterpreter(src)
    try:
        for k in saved_mods:
            sys.modules.pop(k)
        sys.path[:] = [p for p in sys.path if p != src]
        sys.meta_path.insert(0, finder)
        with pytest.raises(ModuleNotFoundError):
            import sisteer.registry.hooks  # noqa: F401 -- the cold interpreter, proven cold
        # BEFORE: the real preflight on the cold interpreter -- every child refused with ModuleNotFoundError, the driver probe ON
        block, problems = srv.registry_row_preflight(row, plans_root=str(rs / lp.PLANS_REL), attempt_root=str(tmp_path / "a"), base_env=base, probe=driver_probe)
        assert problems and len(problems) == len(row["children"]) and all("ModuleNotFoundError: No module named 'sisteer'" in p for p in problems), problems[:2]
        assert block["probe"]["enabled"] is True and block["enabled"] is True
        # AFTER: the row's FIRST act with the built tree -- the staged tree's src on THIS interpreter's path -- then the same preflight passes
        app_src = srv.ensure_tree_src_importable(REPO)
        assert app_src == src and sys.path[0] == src
        block2, problems2 = srv.registry_row_preflight(row, plans_root=str(rs / lp.PLANS_REL), attempt_root=str(tmp_path / "a"), base_env=base, probe=driver_probe)
        assert problems2 == [] and block2["enabled"] is True and len(block2["identity_paths"]) == len(row["children"])
    finally:
        sys.meta_path.remove(finder)
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if k == "sisteer" or k.startswith("sisteer.")]:
            sys.modules.pop(k)
        sys.modules.update(saved_mods)


def test_the_row_puts_the_tree_src_on_the_app_path_unconditionally_before_the_registry_preflight(srv):
    src = DRIVER.read_text()
    i_src = src.index("evidence[\"app_sys_path_src\"] = ensure_tree_src_importable(tree)")
    i_cond = src.index("if producer_row or is_rung:", i_src)                      # the ladder LIBRARY stays conditional, AFTER the path
    i_pre = src.index("reg_block, reg_problems = registry_row_preflight(row, plans_root=ctx[\"plans_root\"]")
    assert i_src < i_cond < i_pre
    assert src.count("ensure_tree_src_importable(tree)") == 1                      # one unconditional call; none left inside the conditional
    body = src[i_src:i_pre]
    assert "return fail(\"LADDER_LIBRARY_MISSING\"" in body                          # the conditional block sits between them, still refusing by name


# --- the row wiring + the operator steps (deliverable 4) -----------------------------------------------------------------------

def test_the_row_asserts_emission_before_its_standup_and_records_the_spool_count_at_its_end(srv):
    src = DRIVER.read_text()
    assert "log_name=child_log_name(k),\n                                                      code_sha=str(ctx[\"W\"]))" in src   # the row passes its tree sha
    i_pre = src.index("reg_block, reg_problems = registry_row_preflight(row, plans_root=ctx[\"plans_root\"]")
    i_refuse = src.index("return fail(REGISTRY_EMISSION_OFF, problems=reg_problems)")
    i_runs = src.index("write_registry_runs(row, plans_root=ctx[\"plans_root\"], serve_dir=serve_dir)")
    i_popen = src.index("standup = subprocess.Popen(argv, env=venv_env(standup_env, argv[0])")
    i_done = src.index("child_registry_completion(k, reg_env, rc=rc, store_root=reg_store")
    i_count = src.index("[\"records_written\"] = count_spool_records(root / lp.SPOOL_REL)")
    assert i_pre < i_refuse < i_runs < i_popen < i_done < i_count
    standup_line = src[src.index("standup_env = dict(env"):].split("\n")[0]
    assert "lp.SPOOL_ENV: f\"{root}/{lp.SPOOL_REL}\"" in standup_line and "lp.SUBSTRATE_ENV: lp.SUBSTRATE" in standup_line   # the attempt spool + modal
    assert srv.REGISTRY_EMISSION_OFF == "REGISTRY_EMISSION_OFF" and srv.REGISTRY_RUNS_FILE == hooks.RUNS_FILE
    assert f"{srv.SERVE_DIR}/registry_runs.json" in srv.ATTEMPT_ROOT_LAYOUT and f"{lp.SPOOL_REL}/" in srv.ATTEMPT_ROOT_LAYOUT


def test_operator_steps_name_completion_and_publication_after_the_pull(srv):
    doc = srv.__doc__
    i_pull = doc.index("--pull pasttense --pull-stages all --pull-to $RS/stores")
    i_acc = doc.index("$RS/accepted.json")
    i_complete = doc.index("logical_run_complete.py --plan $f")
    i_publish = doc.index("registry_publish.py --accepted $RS/accepted.json --attempts $RS/stores --union-dir $RS/registry/union")
    assert i_pull < i_acc < i_complete < i_publish
    assert "--spool ~/registry/records --verdict-dir $RS/registry/union" in doc
