"""backfill-fix-1: the SERVINGJOBS backfill and the degap reader resolve a run's funnel-ledger unit from the run's
OWN identity — the ledger's record of its job, else the producer key's shape — never from registry trial_index (0 for both static trials,
which handed the t1 run the t0 unit's attempts) or a family prefix (one `gsm8k/` for every object). The resolved unit must be the run's own
(a mis-filed job, a key the ledger contradicts, an unresolvable unit -> refused by name); the census cross-check flags additions as well as
missing jobs; the sacct cache keeps terminal states only, expunges the rest on load and before save, and a dry run never writes it.

Through the REAL registry (synth records + the harvester), synthetic ledgers and a stubbed sacct query — the live registry is never read."""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
SCRIPT = REPO / "scripts" / "sdfbase" / "full22" / "registry_serving_jobs_backfill.py"
MODEL = "synthmodel_4b_it"            # synth.MODEL_ID (asserted in _world)


@pytest.fixture(scope="module")
def bf():
    spec = importlib.util.spec_from_file_location("serving_jobs_backfill_fix1_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def walker(bf):
    return bf._walker()


def _world(tmp_path, runs):
    """A registry of CURRENT runs of the synthetic model: `runs` = [(producer key, own job id, view, token cap)] — distinct token caps keep
    the runs distinct under the supersession key while trial_index stays 0 on every one (the registry's trial_index is NOT the trial)."""
    from sisteer.registry import db as DB, harvest as H, identity, records as R, synth
    root = str(tmp_path / "reg"); H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    model = synth.MODEL_ID; rids = {}
    assert model == MODEL
    for i, (key, job, view, cap) in enumerate(runs):
        rid, recs = synth.run_records(key, 100 + 20 * i, through="judged", view_legacy=f"{model}_{view}", token_cap_tokens=cap)
        recs.append(synth.rec("run_patch", {"run_id": rid, "fields": {"cluster_job_id": job}}, synth.t(100 + 20 * i + 9)))
        R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid), "000001-life.jsonl"), recs)
        rids[key] = rid
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0, rep.as_dict()
    return root, Path(DB.db_path(root)), model, rids, synth.CAMPAIGN


def _ledger(tmp_path, model, rows):
    state = tmp_path / "state"; (state / model).mkdir(parents=True, exist_ok=True)
    (state / model / "attempts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(state)


def _attempt(lwid, job, cluster="cluster_a", n=1, kind="target"):
    return {"event": "attempt", "logical_work_id": lwid, "attempt_id": f"{lwid}@{cluster}@{n}", "cluster": cluster, "job_id": job, "kind": kind}


def _args(root, db, model, state, **kw):
    a = ["--registry", str(db), "--model", model, "--spool", os.path.join(root, "records"), "--ledger-state", state, "--code-sha", "c" * 40]
    for k, v in kw.items():
        if v is True:
            a.append(f"--{k.replace('_', '-')}")
        elif v not in (None, False):
            a += [f"--{k.replace('_', '-')}", str(v)]
    return a


def _emission_on(monkeypatch, tmp_path):
    from sisteer.registry import emit
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv("SISTEER_REGISTRY_SPOOL", str(tmp_path / "pinned-spool"))


def _plan(bf, root, db, model, state, tmp_path, sacct, **kw):
    rc = bf.main(_args(root, db, model, state, report=tmp_path / "plan.json", **kw), sacct=sacct)
    return rc, {r["run_id"]: r for r in json.loads((tmp_path / "plan.json").read_text())["plan"]}


def _trial_index(db, rid):
    import sqlite3
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True); t = conn.execute("SELECT trial_index FROM run WHERE run_id = ?", (rid,)).fetchone()[0]; conn.close()
    return t


# ------------------------------------------------------------------------------------------------- (a) the unit from the producer identity

def test_the_producer_identity_names_the_unit_for_every_family_shape_and_never_the_registry_trial(walker):
    m = walker; q = "qwen35_35b_a3b"
    def run(key, view, trial_index=0):
        return {"model_id": q, "view": f"{q}_{view}", "producer_run_key": key, "trial_index": trial_index, "cluster_job_id": None}
    # the two static trials: registry trial_index is 0 on both; the key's segment decides
    assert m.unit_from_producer_identity(run(f"f22s-{q}-{q}_s2r-static_base_t0@833933", "s2r")) == f"{q}/s2r/static/t0"
    assert m.unit_from_producer_identity(run(f"f22s-{q}-{q}_s2r-static_base_t1@833934", "s2r", trial_index=0)) == f"{q}/s2r/static/t1"
    assert m.unit_from_producer_identity(run(f"sdfbase_full22_{q}_s2rl_sampled_t1", "s2rl")) == f"{q}/s2rl/static/t1"
    assert m.unit_from_producer_identity(run(f"sdfbase_full22_{q}_s2r_sampled_t1_cap8192", "s2r")) == f"{q}/s2r/static/t1"
    assert m.unit_from_producer_identity(run(f"f22s-{q}-{q}_s2r-controls_t1@833936", "s2r")) == f"{q}/s2r/controls/t1"
    assert m.unit_from_producer_identity(run(f"padstat-{q}-p16384@807666", "u0")) == f"{q}/u0/padding/p16384"
    # gsm8k: the OBJECT is part of the unit (one `gsm8k/` prefix covered every object)
    assert m.unit_from_producer_identity(run(f"g8k-{q}-s2r-control_relu@834529", "s2r")) == f"{q}/s2r/gsm8k/control_relu"
    assert m.unit_from_producer_identity(run(f"g8k-{q}-d0p75_relu-difference_relu@874351", "d0p75_relu")) == f"{q}/d0p75_relu/gsm8k/difference_relu"
    assert m.unit_from_producer_identity(run(f"mmluF-{q}-s1@882571", "s1")) == f"{q}/s1/mmlu/mmlu"
    # multi-turn: the attack + the shard group, in both key spellings; a key naming another view is not this run's identity
    assert m.unit_from_producer_identity(run(f"f22-att-{q}_s2r-g01234-actorattack@831868", "s2r")) == f"{q}/s2r/multiturn/actorattack/g01234"
    assert m.unit_from_producer_identity(run(f"sdfbase_full22-att-crescendo-{q}_s2rl-sh5_6_7_8of9", "s2rl")) == f"{q}/s2rl/multiturn/crescendo/g5678"
    assert m.unit_from_producer_identity(run(f"sdfbase_full22-att-pair-{q}_s2rl-sh0_1_2_3_4of9", "s2rl")) == f"{q}/s2rl/multiturn/pair/g01234"
    assert m.unit_from_producer_identity(run(f"f22-att-{q}_s1-g01234-crescendo@872043", "s2r")) is None
    assert m.unit_from_producer_identity(run(f"f22-petri-{q}-s2r-rerun@868755", "s2r")) == f"{q}/s2r/petri/proxy"
    assert m.unit_from_producer_identity(run(f"sdfbase_full22-petri-{q}_s2rl-pass-first", "s2rl")) == f"{q}/s2rl/petri/proxy"
    assert m.unit_from_producer_identity(run(f"f22-misalign-{q}_s2r@833967", "s2r")) == f"{q}/s2r/agentic/proxy"
    assert m.unit_from_producer_identity(run(f"sdfbase_full22-misalign-agentic-{q}_s2rl-target-{q}_s2rl-mode-full", "s2rl")) == f"{q}/s2rl/agentic/proxy"
    assert m.unit_from_producer_identity(run(f"f22-bonclimb-{q}_u0@805407", "u0")) == f"{q}/u0/bon/climb"
    assert m.unit_from_producer_identity(run(f"sdfbase_full22-bonclimb-{q}_u0", "u0")) == f"{q}/u0/bon/climb"          # the job-independent climb key
    assert m.unit_from_producer_identity(run(f"f22-stat-{q}_u0-3@805406", "u0")) is None                 # no known shape: the ledger's record decides
    assert m.unit_from_producer_identity({"model_id": q, "view": "other_s1", "producer_run_key": f"f22s-{q}-{q}_s1-static_base_t0@1"}) is None
    # the members of a unit: a multi-turn attack admits every shard group; everything else admits itself only
    mt = m.unit_members(f"{q}/s2r/multiturn/pair/g01234")
    assert mt(f"{q}/s2r/multiturn/pair/g5678") and mt(f"{q}/s2r/multiturn/pair/g01234") and not mt(f"{q}/s2r/multiturn/crescendo/g01234") and not mt(f"{q}/s1/multiturn/pair/g01234")
    st = m.unit_members(f"{q}/s2r/static/t1")
    assert st(f"{q}/s2r/static/t1") and not st(f"{q}/s2r/static/t0") and not st(f"{q}/s2r/static/t10")


# ------------------------------------------------------------------------------------------------- (e) the populations

def test_a_two_trial_static_unit_gives_each_trial_only_its_own_attempts_and_a_retried_trial_both_of_its_own(bf, tmp_path, capsys):
    model = MODEL
    k0, k1, kc = (f"f22s-{model}-{model}_s2r-static_base_t0@7001", f"f22s-{model}-{model}_s2r-static_base_t1@7003", f"f22s-{model}-{model}_s2r-controls_t1@7006")
    root, db, model, rids, campaign = _world(tmp_path, [(k0, "7001", "s2r", 512), (k1, "7003", "s2r", 1024), (kc, "7006", "s2r", 2048)])
    t0, t1, c1 = rids[k0], rids[k1], rids[kc]
    assert _trial_index(db, t0) == 0 and _trial_index(db, t1) == 0                      # the registry says trial 0 for BOTH (the trap)
    state = _ledger(tmp_path, model, [_attempt(f"{model}/s2r/static/t0", "7001"),
                                      _attempt(f"{model}/s2r/static/t1", "7002"), _attempt(f"{model}/s2r/static/t1", "7003", n=2),
                                      _attempt(f"{model}/s2r/controls/t0", "7005"), _attempt(f"{model}/s2r/controls/t1", "7006")])
    asked = []
    def sacct(cl, ids, cache=None):
        asked.extend(ids); return {j: ("FAILED" if j == "7002" else "COMPLETED") for j in ids}
    rc, plan = _plan(bf, root, db, model, state, tmp_path, sacct)
    out = capsys.readouterr().out
    assert rc == 0 and "1 to patch, 2 skipped, 0 not yet (live siblings), 0 refused" in out, out
    assert plan[t0]["action"] == "skip" and plan[t0]["declared_jobs"] == ["7001"] and plan[t0]["unit"] == f"{model}/s2r/static/t0" and plan[t0]["unit_how"] == "ledger"
    assert plan[t1]["action"] == "patch" and plan[t1]["payload"]["serving_jobs"] == ["7002", "7003"] and plan[t1]["unit"] == f"{model}/s2r/static/t1"
    assert plan[t1]["sources"]["7002"] == [f"ledger:{model}/s2r/static/t1@cluster_a@1@cluster_a:FAILED"]
    assert plan[c1]["action"] == "skip" and plan[c1]["declared_jobs"] == ["7006"] and plan[c1]["unit"] == f"{model}/s2r/controls/t1"
    assert "7001" not in plan[t1]["declared_jobs"] and "7005" not in plan[c1]["declared_jobs"]      # the sibling trial's job never crosses over
    assert sorted(set(asked)) == ["7001", "7002", "7003", "7006"]                                      # sacct is asked about the units' own jobs only
    assert f"by family" in out and f"by view   {model}_s2r" in out, out


def test_three_gsm8k_objects_each_read_only_their_own_objects_attempts(bf, tmp_path):
    model = MODEL
    keys = [f"g8k-{model}-s2r-{o}@{j}" for o, j in (("control_relu", "8001"), ("constitution_relu", "8002"), ("difference_relu", "8003"))]
    root, db, model, rids, campaign = _world(tmp_path, [(keys[0], "8001", "s2r", 512), (keys[1], "8002", "s2r", 1024), (keys[2], "8003", "s2r", 2048)])
    state = _ledger(tmp_path, model, [_attempt(f"{model}/s2r/gsm8k/control_relu", "8000"), _attempt(f"{model}/s2r/gsm8k/control_relu", "8001", n=2),
                                      _attempt(f"{model}/s2r/gsm8k/constitution_relu", "8002"), _attempt(f"{model}/s2r/gsm8k/difference_relu", "8003"),
                                      _attempt(f"{model}/s1/gsm8k/control_relu", "8100")])
    rc, plan = _plan(bf, root, db, model, state, tmp_path, lambda cl, ids, cache=None: {j: ("FAILED" if j == "8000" else "COMPLETED") for j in ids})
    assert rc == 0
    assert plan[rids[keys[0]]]["action"] == "patch" and plan[rids[keys[0]]]["payload"]["serving_jobs"] == ["8000", "8001"] and plan[rids[keys[0]]]["unit"] == f"{model}/s2r/gsm8k/control_relu"
    assert plan[rids[keys[1]]]["action"] == "skip" and plan[rids[keys[1]]]["declared_jobs"] == ["8002"]
    assert plan[rids[keys[2]]]["action"] == "skip" and plan[rids[keys[2]]]["declared_jobs"] == ["8003"]


# ------------------------------------------------------------------------------------------------- (b) the unit must be this run's own

def test_a_unit_that_is_not_this_runs_own_refuses_by_name_and_a_derivable_unit_without_ledger_attempts_skips(bf, tmp_path, capsys):
    model = MODEL
    k_misfiled = f"f22s-{model}-{model}_s2r-static_base_t0@7003"                                              # the key says t0; the ledger files 7003 under t1
    k_unknown_job = f"f22s-{model}-{model}_s2r-static_base_t1@7009"                                           # 7009 is no ledger attempt; t1 has others
    k_no_shape = "k_unknown"                                                                                   # no key shape, job 7010 not in the ledger
    k_fresh = f"f22s-{model}-{model}_s2r-controls_t0@7011"                                                    # derivable; the ledger has nothing under it
    root, db, model, rids, campaign = _world(tmp_path, [(k_misfiled, "7003", "s2r", 512), (k_unknown_job, "7009", "s2r", 1024),
                                                        (k_no_shape, "7010", "s2r", 2048), (k_fresh, "7011", "s2r", 4096)])
    state = _ledger(tmp_path, model, [_attempt(f"{model}/s2r/static/t1", "7002"), _attempt(f"{model}/s2r/static/t1", "7003", n=2)])
    rc, plan = _plan(bf, root, db, model, state, tmp_path, lambda cl, ids, cache=None: {j: "COMPLETED" for j in ids})
    out = capsys.readouterr().out
    assert rc == 3 and "0 to patch, 1 skipped, 0 not yet (live siblings), 3 refused" in out, out
    r = plan[rids[k_misfiled]]
    assert r["action"] == "refuse" and f"the ledger files this run's job(s) ['7003'] under {model}/s2r/static/t1 but the producer key" in r["reason"] and f"names {model}/s2r/static/t0" in r["reason"], r["reason"]
    assert r["declared_jobs"] == ["7003"] and not any(s.startswith("ledger:") for s in r["sources"]["7003"])     # nothing declared from the ledger on a unit problem
    r = plan[rids[k_unknown_job]]
    assert r["action"] == "refuse" and f"the ledger's attempts under {model}/s2r/static/t1 (['7002', '7003']) do not include this run's job(s) ['7009']" in r["reason"], r["reason"]
    r = plan[rids[k_no_shape]]
    assert r["action"] == "refuse" and "the run's unit could not be resolved: its job(s) ['7010'] are not ledger attempts and the producer key 'k_unknown' has no known shape" in r["reason"], r["reason"]
    r = plan[rids[k_fresh]]
    assert r["action"] == "skip" and r["unit"] == f"{model}/s2r/controls/t0" and r["unit_how"] == "producer_key" and r["declared_jobs"] == ["7011"]
    assert "REFUSED" in out and "0 record(s) written" in out
    # without a ledger the registry's job is the whole declared set and no unit is required (the fix6 contract)
    rc, plan = _plan(bf, root, db, model, "", tmp_path, lambda cl, ids, cache=None: {j: "COMPLETED" for j in ids})
    assert rc == 0 and all(p["action"] == "skip" and p["unit"] is None for p in plan.values())


# ------------------------------------------------------------------------------------------------- (c) the census flags additions too

def test_the_census_cross_check_flags_additions_as_well_as_missing_jobs(bf, walker, tmp_path):
    from sisteer.registry import identity
    model = MODEL
    key = f"f22s-{model}-{model}_s2r-static_base_t1@7003"
    root, db, model, rids, campaign = _world(tmp_path, [(key, "7003", "s2r", 512)])
    rid = rids[key]
    state = _ledger(tmp_path, model, [_attempt(f"{model}/s2r/static/t1", "7002"), _attempt(f"{model}/s2r/static/t1", "7003", n=2)])
    sacct = lambda cl, ids, cache=None: {j: ("FAILED" if j == "7002" else "COMPLETED") for j in ids}
    # a FAILED attempt of the run's own unit that wrote no shard is NOT an addition: the store names 7003 only and the set still reads patch
    census = tmp_path / "results.jsonl"
    census.write_text(json.dumps({"run_id": rid, "status": "measured", "store": {"identity": {"store_uri:transcripts_results": {"checks": {"job": {"registry": "7003", "store": ["7003"]}}}}}}) + "\n")
    rc, plan = _plan(bf, root, db, model, state, tmp_path, sacct, census=census)
    assert rc == 0 and plan[rid]["action"] == "patch" and plan[rid]["payload"]["serving_jobs"] == ["7002", "7003"], plan[rid]
    # a missing job (the store names one the declared set does not cover) refuses, as before
    census.write_text(json.dumps({"run_id": rid, "status": "measured", "store": {"identity": {"store_uri:transcripts_results": {"checks": {"job": {"registry": "7003", "store": ["7003", "9999"]}}}}}}) + "\n")
    rc, plan = _plan(bf, root, db, model, state, tmp_path, sacct, census=census)
    assert rc == 3 and "the store names serving job(s) ['9999'] the ledger + registry do not declare" in plan[rid]["reason"]
    # an ADDITION: a declared job no shard names that is not an attempt of this run's unit — reachable only through a resolver that hands a
    # run another unit's attempts, so the planner is driven with such a resolver and must refuse by name
    census.write_text(json.dumps({"run_id": rid, "status": "measured", "store": {"identity": {"store_uri:transcripts_results": {"checks": {"job": {"registry": "7003", "store": ["7003"]}}}}}}) + "\n")
    runs = bf.current_runs(db, model)
    def defective(run, *, extra, ledger_state, sacct, sacct_cache):
        d = walker.declared_jobs_for(run, extra=extra, ledger_state=ledger_state, sacct=sacct, sacct_cache=sacct_cache)
        d["declared_jobs"] = sorted(set(d["declared_jobs"]) | {"7001"}); d["declared_jobs_sources"]["7001"] = [f"ledger:{model}/s2r/static/t0@cluster_a@1@cluster_a:COMPLETED"]
        return d
    stub = SimpleNamespace(declared_jobs_for=defective, unit_members=walker.unit_members, own_jobs=walker.own_jobs)
    plan = bf.plan_runs(runs, ledger_state=state, sacct=sacct, sacct_cache={}, census=bf.census_store_jobs(census), spool=Path(root) / "records", walker=stub, identity=identity)
    r = next(p for p in plan if p["run_id"] == rid)
    assert r["action"] == "refuse" and f"declared job(s) ['7001'] that no shard of this store names and that are not attempts of this run's unit {model}/s2r/static/t1 (census cross-check: additions)" in r["reason"], r["reason"]
    # the same defective set with no census attesting the store is caught by nothing here — the census is the cross-check (and the resolver the fix)
    plan = bf.plan_runs(runs, ledger_state=state, sacct=sacct, sacct_cache={}, census={}, spool=Path(root) / "records", walker=stub, identity=identity)
    assert next(p for p in plan if p["run_id"] == rid)["action"] == "patch"


# ------------------------------------------------------------------------------------------------- (d) the sacct cache

def _site_login(tmp_path, monkeypatch, login="login-a"):
    """A site config whose one cluster block (a site label of its own, the registry label the records carry) names the login
    host sacct is read through: the host comes from the site, never from the registry label."""
    site = tmp_path / "site"
    (site / "clusters").mkdir(parents=True)
    (site / "config.env").write_text('SISTEER_CLUSTERS="site_one"\nSISTEER_PRIMARY_CLUSTER=site_one\n')
    (site / "clusters" / "site_one.env").write_text(f"CLUSTER_SSH={login}\nCLUSTER_SHARED_ROOT={tmp_path}\nCLUSTER_PARTITION=p\n"
                                                    "CLUSTER_QOS=q\nCLUSTER_SUBSTRATE=cluster_a\nCLUSTER_LABEL=site_one\n")
    monkeypatch.setenv("SISTEER_CONFIG", str(site / "config.env"))
    return login


def test_the_sacct_cache_keeps_terminal_states_only_and_a_live_sibling_completes_across_two_passes_on_one_cache_file(bf, walker, tmp_path, capsys, monkeypatch):
    _emission_on(monkeypatch, tmp_path)
    login = _site_login(tmp_path, monkeypatch)
    model = MODEL
    key = f"f22s-{model}-{model}_s2r-static_base_t1@7003"
    root, db, model, rids, campaign = _world(tmp_path, [(key, "7003", "s2r", 512)])
    rid = rids[key]
    state = _ledger(tmp_path, model, [_attempt(f"{model}/s2r/static/t1", "7002"), _attempt(f"{model}/s2r/static/t1", "7003", n=2)])
    states = {"7002": "RUNNING", "7003": "COMPLETED"}; asked = []
    def query(host, ids):
        asked.append((host, tuple(sorted(ids)))); return {j: states[j] for j in ids if j in states}
    sacct = lambda cl, ids, cache=None: walker.sacct_states(cl, ids, cache=cache, query=query)
    cache = tmp_path / "sacct.json"
    # pass 1 (write mode): the sibling runs — not yet; the cache file holds the COMPLETED job only, never the RUNNING one
    rc = bf.main(_args(root, db, model, state, write=True, sacct_cache=cache, report=tmp_path / "p1.json"), sacct=sacct)
    out = capsys.readouterr().out
    assert rc == 0 and "0 to patch, 0 skipped, 1 not yet (live siblings), 0 refused" in out and "1 terminal state(s) written" in out, out
    assert json.loads(cache.read_text()) == {"cluster_a:7003": "COMPLETED"} and asked == [(login, ("7002", "7003"))]
    # pass 2: the sibling completed — only it is asked about (7003 is cached terminal); the set is patched; the cache holds both
    states["7002"] = "COMPLETED"; asked.clear()
    rc = bf.main(_args(root, db, model, state, write=True, sacct_cache=cache, report=tmp_path / "p2.json"), sacct=sacct)
    out = capsys.readouterr().out
    assert rc == 0 and "1 to patch" in out and "1 record(s) written" in out and asked == [(login, ("7002",))], (out, asked)
    assert json.loads(cache.read_text()) == {"cluster_a:7002": "COMPLETED", "cluster_a:7003": "COMPLETED"}
    # a cache seeded with live / unknown entries is expunged on load (the jobs are asked about again) and a DRY run leaves the file untouched
    seeded = {"cluster_a:7002": "RUNNING", "cluster_a:7003": None, "cluster_a:7777": "PENDING", "cluster_a:7778": "FAILED"}
    cache.write_text(json.dumps(seeded)); before = cache.read_text(); asked.clear()
    rc = bf.main(_args(root, db, model, state, sacct_cache=cache, report=tmp_path / "p3.json"), sacct=sacct)
    out = capsys.readouterr().out
    assert rc == 0 and asked == [(login, ("7002", "7003"))] and "3 non-terminal entries expunged on load" in out and "NOT written (dry run)" in out, (out, asked)
    assert cache.read_text() == before
    # the module-level helpers: terminal-only caching, 'CANCELLED by 1' terminal by its first word, None / live never cached
    c = {}
    got = walker.sacct_states("cluster_a", ["1", "2", "3"], cache=c, query=lambda host, ids: {"1": "CANCELLED by 1", "2": "RUNNING"})
    assert got == {"1": "CANCELLED by 1", "2": "RUNNING", "3": None} and c == {"cluster_a:1": "CANCELLED by 1"}
    c = {"cluster_a:9": "RUNNING", "cluster_b:8": None, "cluster_a:7": "COMPLETED"}
    assert walker.expunge_nonterminal_sacct(c) == ["cluster_a:9", "cluster_b:8"] and c == {"cluster_a:7": "COMPLETED"}


def test_the_degap_manifest_command_expunges_live_states_from_its_cache_on_load_and_before_save(walker, tmp_path, monkeypatch):
    """The reader shares the resolver and the cache rule: cmd_manifest keeps terminal states only across runs."""
    m = walker
    cache = tmp_path / "c.json"; cache.write_text(json.dumps({"cluster_a:1": "RUNNING", "cluster_a:2": "COMPLETED", "cluster_b:3": None}))
    monkeypatch.setattr(m, "manifest_from_registry", lambda *a, **k: (k["sacct_cache"].__setitem__("cluster_a:4", "PENDING"), [])[1])
    out = tmp_path / "m.jsonl"
    a = SimpleNamespace(sacct_cache=str(cache), registry=str(tmp_path / "none.sqlite"), model=None, all_current=False, ledger_state="", out=str(out))
    m.cmd_manifest(a)
    assert json.loads(cache.read_text()) == {"cluster_a:2": "COMPLETED"}
