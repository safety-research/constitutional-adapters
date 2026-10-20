"""Integration F1 (judge path ↔ registry identity): the controls JUDGE unit
carries `SISTEER_REGISTRY_RUN_IDENTITY` = the ACCEPTED attempt's own per-child identity file (the `children/controls__<arm>__s<s>.argv.json`
`env_registry` → the SAME plans file under `RS/registry/plans`, the controls per-task map), so that the FIRST judged
row opens the controls `run` records with the identity the generate child carried — the generate-only controls child files no controls rows.

Known-bad, proven here: without the env, one judged row through the real `CellStore` opens NOTHING (no `run` record); with the env,
exactly one `run` record with the map's run id for the judged task. And the runbook's plan step for the judge carries the env, resolved
from the attempt's record, never from a hand-supplied string — an attempt without the record, or a record naming a file the producer
never wrote (launch_report F-C), is REFUSED by name.

The fixture world is the grading runbook's own (`tests/test_seedrep_grading_e2e.py`: World / attempt / write_controls_store), imported by
path and not edited; the identities are the PRODUCER's (`registry_plans.build` over the controls legs — never hand-written).
"""
from __future__ import annotations

import functools
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SEEDREP = REPO / "scripts" / "sdfbase" / "seedrep"
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", SEEDREP, REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import judge_run_identity as jri  # noqa: E402
import registry_plans as RP  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
from sisteer.capeval import store as cs  # noqa: E402
from sisteer.registry import emit, hooks, records as R  # noqa: E402
from sisteer.registry import synth  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E2E = _load("seedrep_grading_e2e_world", REPO / "tests" / "test_seedrep_grading_e2e.py")   # the runbook's own fixture world (read-only reuse)

W = synth.code_sha("f")
W7 = W[:7]
CAMPAIGN = "seedrep_battery"
VIEW, TAG, CELL = E2E.VIEW, E2E.TAG, E2E.CELL_NONE            # `none`: an arm AND a cell of the runbook's world
CAP = 512
XSTEST_TASK = "judged/xstest_v2"                               # judged.TASK_XSTEST — the store writer's task name, the map's route key


@functools.lru_cache(maxsize=None)
def _plan_set():
    """The producer's OWN controls identities (registry_plans.build over the three controls legs), built once per session."""
    inp = RP.load_inputs(campaign=CAMPAIGN, w_sha=W)
    return RP.build(inp, ["xstest", "or_bench", "mmlu"])


def _write_plans(rs: Path) -> Path:
    out = rs / jri.PLANS_REL
    RP.write_plan_set(_plan_set(), out)
    return out


def _child_record(root: Path, *, cell: str, s: int, cap: int, identity: str, out_root_name: str) -> Path:
    """The per-child record as the spawn loop writes it (`seedrep_served.py`: `{"argv": k_argv, "env_registry": {...}}` under
    children/<child_log_name>.argv.json) for the controls child of `cell` at seed `s`; the argv shape is `_controls_children`'s."""
    argv = ["/venv/bin/python", "/tree/scripts/served_capability_run.py", "generate", "--proxy-url", "http://127.0.0.1:9000/v1",
            "--routing-table", "/workspace/attempt/exp/serve/routing_table.json", "--cell", cell, "--config", "configs/x.yaml",
            "--defenses-dir", "/bank/s1", "--out-root", f"/workspace/attempt/exp/transcripts_results/{out_root_name}",
            "--controls", ",".join(lp.CONTROLS_ALL), "--concurrency", "8", "--top-logprobs", "64", "--trial", "0", "--controls-cap", str(cap)]
    p = root / jri.CHILDREN_DIR / f"controls__{cell}__s{s}.argv.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"argv": argv, "env_registry": {lp.RUN_IDENTITY_ENV: identity, lp.SPOOL_ENV: f"/workspace/attempt/{lp.SPOOL_REL}"}}, indent=1))
    return p


def _world(tmp_path: Path, *, identity_rel: str | None = RP.store_relpath("controls", "none", 1), with_record: bool = True):
    """A controls attempt of view s1 holding the `none` store at cap 512, its pull record, the plans under RS, and the child's record."""
    w = E2E.World(tmp_path, with_arditi=False, with_rungs=False)
    root = w.rs / "stores" / "controls" / VIEW / "shard00"
    c_exp = E2E.attempt(root, "C-s1", leg="controls", view=VIEW)
    store = E2E.write_controls_store(c_exp / f"controls_cap{CAP}", CELL, CAP)
    plans = _write_plans(w.rs)
    if with_record:
        _child_record(root, cell=CELL, s=1, cap=CAP, identity=f"{lp.VOL_PATH}/{lp.LANE}/{W7}/{lp.PLANS_REL}/{identity_rel}", out_root_name=f"controls_cap{CAP}")
    return w, root, store, plans


# --------------------------------------------------------------------------- the resolver (the attempt's record -> the plans file)
def test_resolves_the_identity_from_the_attempts_own_child_record(tmp_path):
    w, root, store, plans = _world(tmp_path)
    ident = jri.resolve_controls_identity(root, store, CELL, plans)
    expect = plans / RP.store_relpath("controls", "none", 1)
    assert ident["identity_path"] == str(expect) and expect.is_file()
    assert ident["child_record"] == str(root / "children" / "controls__none__s1.argv.json")
    m = json.loads(expect.read_text())
    assert ident["routes"] == {t: i["run_id"] for t, i in m.items()} and XSTEST_TASK in ident["routes"]
    assert all(i["extra"]["container"] == "controls" and i["seed_base"] == 1 and i["extra"]["arm"] == "none" for i in m.values())


def test_refuses_by_name_without_the_record_or_when_the_named_file_was_never_written(tmp_path):
    # (a) no children/ record at all -> refused (nothing guessed from the leg)
    w, root, store, plans = _world(tmp_path / "a", with_record=False)
    with pytest.raises(jri.IdentityRefused, match="no child record"):
        jri.resolve_controls_identity(root, store, CELL, plans)
    # (b) the record names launch_report F-C's spelling (`registry/plans/controls/none__s1.json`), which the producer never wrote -> refused by name
    w, root, store, plans = _world(tmp_path / "b", identity_rel="controls/none__s1.json")
    with pytest.raises(jri.IdentityRefused, match="F-C"):
        jri.resolve_controls_identity(root, store, CELL, plans)
    # (c) a record for another cell only -> exactly-one binding fails
    w, root, store, plans = _world(tmp_path / "c", with_record=False)
    _child_record(root, cell="cellA__psr_ce_all", s=1, cap=CAP, identity=f"/vol/x/{lp.PLANS_REL}/{RP.store_relpath('controls', 'none', 1)}", out_root_name=f"controls_cap{CAP}")
    with pytest.raises(jri.IdentityRefused, match="0 controls child record"):
        jri.resolve_controls_identity(root, store, CELL, plans)
    # (d) an identity outside registry/plans/, or a single identity where a per-task map is due, or a route with no run_id -> refused
    w, root, store, plans = _world(tmp_path / "d", with_record=False)
    _child_record(root, cell=CELL, s=1, cap=CAP, identity="/vol/x/pending/123.json", out_root_name=f"controls_cap{CAP}")
    with pytest.raises(jri.IdentityRefused, match="not under registry/plans"):
        jri.resolve_controls_identity(root, store, CELL, plans)
    w, root, store, plans = _world(tmp_path / "e", identity_rel="xstest/none__s1.json")      # the single-eval file, not the store's map
    with pytest.raises(jri.IdentityRefused, match="per-task MAP"):
        jri.resolve_controls_identity(root, store, CELL, plans)
    w, root, store, plans = _world(tmp_path / "f")
    p = plans / RP.store_relpath("controls", "none", 1)
    m = json.loads(p.read_text()); del m[XSTEST_TASK]["run_id"]; p.write_text(json.dumps(m))
    with pytest.raises(jri.IdentityRefused, match="no run_id"):
        jri.resolve_controls_identity(root, store, CELL, plans)


# --------------------------------------------------------------------------- the runbook: the judge step carries the env
def test_the_controls_judge_step_exports_the_attempts_identity_and_refuses_without_it(tmp_path):
    w, root, store, plans = _world(tmp_path)
    prep = w.bat("prepare")                                          # the runbook copies the controls stores (the judge writes into a copy)
    assert prep.returncode == 0, prep.stdout + prep.stderr
    r, job = w.plan_job("controls", f"controls/{TAG}", plan_out=tmp_path / "plan")
    assert r.returncode == 0, r.stdout + r.stderr
    assert job["ready"], job["problems"]
    judge = [s for s in job["steps"] if s["name"] == f"judge_cap{CAP}_{CELL}"]
    assert len(judge) == 1, [s["name"] for s in job["steps"]]
    expect = str(plans / RP.store_relpath("controls", "none", 1))
    names = [s["name"] for s in job["steps"]]
    ident_step = [s for s in job["steps"] if s["name"] == f"identity_cap{CAP}_{CELL}"][0]         # the grading-time record, BEFORE the judge
    assert names.index(f"identity_cap{CAP}_{CELL}") < names.index(f"own_cache_cap{CAP}_{CELL}") < names.index(f"judge_cap{CAP}_{CELL}")
    assert ident_step["argv"][1:5] == ["scripts/sdfbase/seedrep/judge_run_identity.py", "record", str(judge[0]["argv"][judge[0]["argv"].index("--controls-root") + 1]) + f"/{TAG}_t0/cells/" + judge[0]["argv"][judge[0]["argv"].index("--match") + 1], expect] and ident_step["argv"][5] == "bound"
    assert judge[0]["env"][jri.RUN_IDENTITY_ENV] == expect and judge[0]["env"]["CAPEVAL_JUDGE_CACHE"]
    assert judge[0]["registry_identity"]["identity_path"] == expect and XSTEST_TASK in judge[0]["registry_identity"]["routes"]
    assert judge[0]["identity_unresolved"] is None and judge[0]["provisional"] is False
    assert f"{jri.RUN_IDENTITY_ENV}={expect}" in job["script"]          # the unit script renders it as the judge step's env
    assert any(jri.RUN_IDENTITY_ENV in n for n in job["notes"])
    # KNOWN-BAD: the store's record gone -> REFUSED by name (never READY, never graded with no identity), with or without children/
    (root / "children" / "controls__none__s1.argv.json").unlink()
    r2, job2 = w.plan_job("controls", f"controls/{TAG}", plan_out=tmp_path / "plan2")
    assert r2.returncode == 2 and not job2["ready"] and any("REFUSED — registry run identity" in p and "child record" in p and CELL in p for p in job2["problems"]), job2["problems"]
    assert not [s for s in job2["steps"] if s["name"].startswith("judge_")]
    (root / "children").rmdir()                                       # a root with NO children/ records at all: the same refusal
    r2b, job2b = w.plan_job("controls", f"controls/{TAG}", plan_out=tmp_path / "plan2b")
    assert r2b.returncode == 2 and not job2b["ready"] and any("no child record" in p and "--allow-partial" in p for p in job2b["problems"]), job2b["problems"]
    # --allow-partial ONLY: graded as an explicitly PROVISIONAL diagnostic -- no identity env, the gap on the step and in the notes
    r3, job3 = w.plan_job("controls", f"controls/{TAG}", "--allow-partial", plan_out=tmp_path / "plan3")
    assert r3.returncode == 0 and job3["ready"], (r3.stdout, job3["problems"])
    j3 = [s for s in job3["steps"] if s["name"] == f"judge_cap{CAP}_{CELL}"][0]
    assert jri.RUN_IDENTITY_ENV not in j3["env"] and j3["provisional"] is True and "no child record" in j3["identity_unresolved"]
    i3 = [s for s in job3["steps"] if s["name"] == f"identity_cap{CAP}_{CELL}"][0]
    assert i3["argv"][4] == "-" and i3["argv"][5] == "provisional" and "no child record" in i3["argv"][6]
    assert any("PROVISIONAL (--allow-partial)" in n and CELL in n for n in job3["notes"]) and jri.RUN_IDENTITY_ENV not in job3["script"]
    # KNOWN-BAD: F-C's path (a file the producer never wrote) -> refused by name too
    _child_record(root, cell=CELL, s=1, cap=CAP, identity=f"{lp.VOL_PATH}/{lp.LANE}/{W7}/{lp.PLANS_REL}/controls/none__s1.json", out_root_name=f"controls_cap{CAP}")
    r4, job4 = w.plan_job("controls", f"controls/{TAG}", plan_out=tmp_path / "plan4")
    assert not job4["ready"] and any("F-C" in p for p in job4["problems"]), job4["problems"]


def _stamp_judged(store: Path) -> None:
    """A complete xstest judge pass on the store (the e2e world's own judge_summary shape) so `post` has an `ok` block to publish."""
    E2E.write_judge_summary(store, {"xstest": E2E.judge_block("xstest", n=250, n_judged=250, refusals=2)})


def _run_step(job: dict, name: str) -> subprocess.CompletedProcess:
    """Run one planned step's argv exactly as the unit would (cwd = the worktree)."""
    st = [x for x in job["steps"] if x["name"] == name][0]
    return subprocess.run(st["argv"], capture_output=True, text=True, cwd=str(REPO))


def _unjudge(store: Path) -> None:
    for n in jri.JUDGE_ARTIFACTS:
        (store / n).unlink(missing_ok=True)


def test_post_publishes_the_grading_time_identity_and_the_table_reader_turns_a_gap_into_a_partial_table(tmp_path):
    """The full `controls -> post -> table reader` path (3d). The judge unit's identity_* step records what it exported (or that it
    exported nothing, provisionally) in the store's REGISTRY_IDENTITY.json BEFORE the judge; `post` publishes THAT as `identity_unresolved`
    (never a re-resolution); the REAL reader (`battery_tables.read_index_controls`) carries it, the row's `registry_identity_status` is MISSING
    and the run's table_status is PARTIAL naming the gap. Regression (3d): a store graded provisionally whose identity record is RECOVERED
    afterwards stays PARTIAL (no registry run was opened; the judge re-emits from cache), the bound identity step over the standing judgments
    REFUSES naming the re-grade, and only a re-grade from scratch (judgment artifacts removed) with the identity bound reads `ok`."""
    import battery_tables as bt
    # (a) the bound world: the unit's identity step (bound) then the judgment -> null on the entry, `ok` registry provenance
    w, root, store, plans = _world(tmp_path / "bound")
    assert w.bat("prepare").returncode == 0
    r, job = w.plan_job("controls", f"controls/{TAG}", plan_out=tmp_path / "bound" / "plan")
    assert job["ready"], job["problems"]
    st = _run_step(job, f"identity_cap{CAP}_{CELL}"); assert st.returncode == 0 and "[judge-identity] recorded" in st.stdout and "bound to" in st.stdout, st.stdout + st.stderr
    copies = {json.loads((d / "manifest.json").read_text())["cell_id"]: d for d in (w.dest / "controls_sampled" / f"controls_cap{CAP}" / f"{TAG}_t0" / "cells").iterdir()}
    rec = json.loads((copies[CELL] / jri.GRADING_IDENTITY_FILE).read_text())
    assert rec["bound"] is True and rec["identity_path"] == str(plans / RP.store_relpath("controls", "none", 1)) and rec["provisional"] is False
    _stamp_judged(copies[CELL])
    assert w.bat("post").returncode == 0
    ent = w.index("controls")["tags"][TAG][f"xstest|{CELL}"]
    assert ent["status"] == "ok" and ent["identity_unresolved"] is None and ent["identity_provisional"] is False and ent["grading_identity"].endswith(jri.GRADING_IDENTITY_FILE)
    rd = bt.read_index_controls(w.grades, TAG, "xstest", CELL, "pp")
    assert rd.eligible and rd.extra["identity_unresolved"] is None and rd.extra["identity_checked"] is True
    cols = bt._reading_columns(rd, None, None, [])
    assert cols["registry_identity_status"] == bt.OK and bt.registry_identity_gaps([{**cols, "leg": "xstest", "arm": CELL, "seed": 1}]) == []
    # (b) the record-less world graded PROVISIONALLY under --allow-partial: the entry stays `ok` (the judge's verdict) but carries WHY + the re-grade hint
    w2, root2, store2, plans2 = _world(tmp_path / "recordless", with_record=False)
    assert w2.bat("prepare").returncode == 0
    r2, job2 = w2.plan_job("controls", f"controls/{TAG}", "--allow-partial", plan_out=tmp_path / "recordless" / "plan")
    assert job2["ready"], job2["problems"]
    st2 = _run_step(job2, f"identity_cap{CAP}_{CELL}"); assert st2.returncode == 0 and "PROVISIONAL" in st2.stdout, st2.stdout + st2.stderr
    copies2 = {json.loads((d / "manifest.json").read_text())["cell_id"]: d for d in (w2.dest / "controls_sampled" / f"controls_cap{CAP}" / f"{TAG}_t0" / "cells").iterdir()}
    _stamp_judged(copies2[CELL])
    assert w2.bat("post").returncode == 0
    ent2 = w2.index("controls")["tags"][TAG][f"xstest|{CELL}"]
    assert ent2["status"] == "ok" and ent2["identity_provisional"] is True and "graded PROVISIONALLY" in ent2["identity_unresolved"] and "re-grade from scratch" in ent2["identity_unresolved"]
    rd2 = bt.read_index_controls(w2.grades, TAG, "xstest", CELL, "pp")
    cols2 = bt._reading_columns(rd2, None, None, [])
    assert rd2.eligible and bt.is_missing(cols2["registry_identity_status"]) and bt.registry_identity_gaps([{**cols2, "leg": "xstest", "arm": CELL, "seed": 1}]) == [f"xstest/{CELL}/s1"]
    # REGRESSION (3d): the identity record / file is RECOVERED after the provisional grade -> post keeps the gap (the judgment stands under NO identity)
    _child_record(root2, cell=CELL, s=1, cap=CAP, identity=f"{lp.VOL_PATH}/{lp.LANE}/{W7}/{lp.PLANS_REL}/{RP.store_relpath('controls', 'none', 1)}", out_root_name=f"controls_cap{CAP}")
    assert w2.bat("prepare").returncode == 0 and w2.bat("post").returncode == 0
    ent2b = w2.index("controls")["tags"][TAG][f"xstest|{CELL}"]
    assert ent2b["status"] == "ok" and ent2b["identity_provisional"] is True and "graded PROVISIONALLY" in ent2b["identity_unresolved"]
    # the unit re-planned with the identity now bound: its identity step REFUSES over the standing provisional judgments (re-grade from scratch)
    r3, job3 = w2.plan_job("controls", f"controls/{TAG}", plan_out=tmp_path / "recordless" / "plan3")
    assert job3["ready"] and [x for x in job3["steps"] if x["name"] == f"identity_cap{CAP}_{CELL}"][0]["argv"][5] == "bound", job3["problems"]
    st3 = _run_step(job3, f"identity_cap{CAP}_{CELL}")
    assert st3.returncode != 0 and "graded PROVISIONALLY" in st3.stderr and "re-grade from scratch" in st3.stderr, st3.stdout + st3.stderr
    assert json.loads((copies2[CELL] / jri.GRADING_IDENTITY_FILE).read_text())["provisional"] is True          # the record is untouched
    assert w2.bat("post").returncode == 0 and w2.index("controls")["tags"][TAG][f"xstest|{CELL}"]["identity_provisional"] is True
    # the re-grade from scratch: judgment artifacts removed -> the bound identity step records, the judgment is re-made, post reads `ok`
    _unjudge(copies2[CELL])
    st4 = _run_step(job3, f"identity_cap{CAP}_{CELL}"); assert st4.returncode == 0 and "bound to" in st4.stdout, st4.stdout + st4.stderr
    _stamp_judged(copies2[CELL])
    assert w2.bat("post").returncode == 0
    ent4 = w2.index("controls")["tags"][TAG][f"xstest|{CELL}"]
    assert ent4["status"] == "ok" and ent4["identity_unresolved"] is None and ent4["identity_provisional"] is False
    assert bt._reading_columns(bt.read_index_controls(w2.grades, TAG, "xstest", CELL, "pp"), None, None, [])["registry_identity_status"] == bt.OK
    # (c) a store judged with NO grading-time record at all (graded outside the unit / before the record existed) -> unresolved, re-grade hint
    (copies2[CELL] / jri.GRADING_IDENTITY_FILE).unlink()
    assert w2.bat("post").returncode == 0
    ent5 = w2.index("controls")["tags"][TAG][f"xstest|{CELL}"]
    assert "no grading-time identity record" in ent5["identity_unresolved"] and ent5["identity_provisional"] is None
    # REGRESSION (3e): standing judgments + ABSENT marker + the identity now recoverable -> the bound identity step REFUSES (nothing recorded),
    # a provisional invocation refuses too (the state the judgments were made under is unknown), post stays unresolved -- never a fresh `bound`
    st5 = _run_step(job3, f"identity_cap{CAP}_{CELL}")
    assert st5.returncode != 0 and "stand with no REGISTRY_IDENTITY.json" in st5.stderr and "re-grade from scratch" in st5.stderr, st5.stdout + st5.stderr
    assert not (copies2[CELL] / jri.GRADING_IDENTITY_FILE).exists()
    with pytest.raises(jri.IdentityRefused, match="stand with no REGISTRY_IDENTITY.json"):
        jri.record_grading_identity(copies2[CELL], identity_path=None, provisional=True, unresolved="x", stamp="S", unit="u")
    assert w2.bat("post").returncode == 0
    ent6 = w2.index("controls")["tags"][TAG][f"xstest|{CELL}"]
    assert "no grading-time identity record" in ent6["identity_unresolved"] and bt.is_missing(bt._reading_columns(bt.read_index_controls(w2.grades, TAG, "xstest", CELL, "pp"), None, None, [])["registry_identity_status"])
    # an unreadable / malformed record beside standing judgments refuses the same way
    (copies2[CELL] / jri.GRADING_IDENTITY_FILE).write_text("{not json")
    with pytest.raises(jri.IdentityRefused, match="unreadable grading-time record"):
        jri.record_grading_identity(copies2[CELL], identity_path=str(plans2 / RP.store_relpath("controls", "none", 1)), provisional=False, unresolved=None, stamp="S", unit="u")
    (copies2[CELL] / jri.GRADING_IDENTITY_FILE).write_text(json.dumps({"something": "else"}))
    with pytest.raises(jri.IdentityRefused, match="not a grading-time identity record"):
        jri.record_grading_identity(copies2[CELL], identity_path=str(plans2 / RP.store_relpath("controls", "none", 1)), provisional=False, unresolved=None, stamp="S", unit="u")
    # with the judgments removed the same store records freely again (a re-grade from scratch)
    (copies2[CELL] / jri.GRADING_IDENTITY_FILE).unlink(); _unjudge(copies2[CELL])
    assert "recorded" in jri.record_grading_identity(copies2[CELL], identity_path=str(plans2 / RP.store_relpath("controls", "none", 1)), provisional=False, unresolved=None, stamp="S", unit="u")
    # (d) the REAL table run on the fixture world: stamping the runbook's field on the xstest entries makes the table PARTIAL by name
    fxb = _load("seedrep_battery_fixture_f1", REPO / "tests" / "fixtures" / "seedrep_battery" / "build_fixture.py")
    world = tmp_path / "tables"; fxb.build(world)
    base = bt.run(bt.load_inputs(world), world / "out0", legs=(bt.LEG_BY_ID["xstest"],))
    assert not any("registry run identity unresolved" in r_ for r_ in base["provisional_reasons"]) and base["registry_identity_gaps"] == []
    idx_p = world / "grades" / bt.INDEX_CONTROLS
    idx = json.loads(idx_p.read_text())
    n_stamped = 0
    for tag_, ents in idx["tags"].items():
        for key_, e_ in ents.items():
            if key_.startswith("xstest|"):
                e_["identity_unresolved"] = "graded PROVISIONALLY with no registry run identity — synthetic gap"; n_stamped += 1
    assert n_stamped
    idx_p.write_text(json.dumps(idx, indent=1))
    part = bt.run(bt.load_inputs(world), world / "out1", legs=(bt.LEG_BY_ID["xstest"],))
    assert part["partial"] is True and part["table_status"].startswith("PARTIAL (") and "registry run identity unresolved" in part["table_status"]
    assert len(part["registry_identity_gaps"]) == sum(1 for r_ in part["legs"]["xstest"]["rows"] if r_["reading_status"] == bt.OK) > 0
    assert all(bt.is_missing(r_["registry_identity_status"]) for r_ in part["legs"]["xstest"]["rows"] if r_["reading_status"] == bt.OK)
    assert "registry_identity_status" in bt.ROW_COLUMNS


# --------------------------------------------------------------------------- the mechanism: one judged row opens exactly one `run`
def _run_records(spool: Path):
    return [rec for p in R.iter_record_files(str(spool)) for _i, rec, _s in R.read_record_file(p) if rec["record_type"] == "run"]


def _judged_row(store_dir: Path) -> cs.ResultRow:
    return cs.ResultRow(sweep="controls_sampled", run_id="r", model_id="test/model", model_revision="r" * 40, arm_name="none", kind="none", coeff=None,
                        intervention_hash="i" * 16, engine="vllm", tier=1, task=XSTEST_TASK, task_version="v2", domain="over_refusal", item_id="xst0000",
                        trial_id=0, seed=1, attempt=0, position_order=0, metric="label", value=0.0, extra_json=None, harness_config_hash="h" * 16,
                        env_digest="e" * 16, slurm_job_id=None, created_at=time.time(), wall_clock_s=None)


def test_one_judged_row_opens_exactly_one_run_record_with_the_maps_identity_and_none_without_the_env(tmp_path, monkeypatch):
    """The real CellStore the judge opens (`cs.CellStore(cells_dir.parent, name, row_constants=manifest[row_constants])`) filing ONE judged row:
    with the judge step's env -> exactly one `run` record, run_id = the map's route for the judged task, campaign the lane's; the SAME store
    with the env unset -> no record at all (the F1 failure: the controls runs never open)."""
    w, root, store, plans = _world(tmp_path)
    ident = jri.resolve_controls_identity(root, store, CELL, plans)
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.delenv(hooks.EMIT_ENV, raising=False)
    hooks._DEFAULT_NOTICED.clear()
    # without the identity (the pre-F1 judge unit): nothing opens
    spool0 = tmp_path / "spool0"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool0))
    monkeypatch.delenv(jri.RUN_IDENTITY_ENV, raising=False)
    # the judge hands the store its manifest's row_constants; the lane's native stores carry `seed_base = s` on every row
    # (results-registry, hooks._seed_base's fallback carrier) -- the run's seed_base is read from THERE, never lifted from the identity
    constants = {"seed_base": 1}
    st0 = cs.CellStore(store.parent.parent, store.name, row_constants=constants)
    assert st0.recorder is None
    st0.append_row(_judged_row(store))
    assert not spool0.exists() or _run_records(spool0) == []
    # with the judge step's env: the first routed row opens the run
    spool1 = tmp_path / "spool1"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool1))
    monkeypatch.setenv(jri.RUN_IDENTITY_ENV, ident["identity_path"])
    st1 = cs.CellStore(store.parent.parent, store.name, row_constants=constants)
    assert st1.recorder is not None
    st1.append_row(_judged_row(store))
    runs = _run_records(spool1)
    assert len(runs) == 1, runs
    payload = runs[0]["payload"]
    assert payload["run_id"] == ident["routes"][XSTEST_TASK] and payload["campaign"] == CAMPAIGN and payload["lifecycle_state"] == "running"
    m = json.loads(Path(ident["identity_path"]).read_text())
    assert payload["run_id"] == m[XSTEST_TASK]["run_id"] and payload["eval_id"] == m[XSTEST_TASK]["eval_id"] == "xstest"
    assert payload["extra"]["route"] == XSTEST_TASK and payload["seed_base"] == 1
    # a second row of the same task opens nothing more (one run per route, opened once)
    st1.append_row(_judged_row(store))
    assert len(_run_records(spool1)) == 1
