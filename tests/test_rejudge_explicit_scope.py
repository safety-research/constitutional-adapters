"""The explicit-run-list scopes of the MiniMax-M3 re-judge (scripts/rejudge/strongreject/scope_runs.py, scripts/rejudge/terra/
scope_cpi_runs.py): runs with no Claude judgment get their units from their own stores, in exactly the shapes the enumeration, the
manifest build and the filing read; a listed store must be the one the registry ties to the run (a valid store of ANOTHER run is
refused), and a StrongREJECT run's rubric is the one its instrument's judge names. Offline: synthetic stores, a synthetic registry file,
synthetic pinned prompts."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pathlib
import sqlite3
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from rejudge.common import runlist as RL  # noqa: E402
from rejudge.strongreject import enumerate_units as E  # noqa: E402
from rejudge.strongreject import scope_runs as SR  # noqa: E402
from rejudge.terra import cpi as C  # noqa: E402
from rejudge.terra import scope_cpi_runs as SC  # noqa: E402

HAIKU = SR.HAIKU_V1
HAIKU2 = SR.HAIKU_V2
GOALS = {"goals": {f"g{i}": {"dataset": "synthetic/goals", "revision": "0" * 40, "text": f"Synthetic goal text {i}"} for i in range(6)},
         "goal_sets": {"static": [f"g{i}" for i in range(6)], "mt": [f"g{i}" for i in range(6)], "bon": [f"g{i}" for i in range(3)]}}


def jl(path: pathlib.Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def static_cell(root: pathlib.Path, name="static_cell"):
    cell = root / name
    for s in range(2):
        jl(cell / f"shard{s:02d}" / "responses.jsonl",
           [{"goal_id": f"g{g}", "trial": 0, "sample_idx": v, "response": f"static reply g{g} v{v}"} for g in range(s * 3, s * 3 + 3) for v in range(3)])
    return cell


def mt_cell(root: pathlib.Path, name="mt_cell"):
    """Two shards: g0 errors in shard00 and is re-run fine in shard01 (latest outcome ok: kept); g4 is refused in shard01 (dropped);
    g5 errors (dropped)."""
    cell = root / name
    jl(cell / "shard00" / "responses.jsonl", [{"goal_id": f"g{g}", "order": o, "response": f"turn {o} of g{g} (a)"} for g in (0, 1, 2) for o in range(3)])
    jl(cell / "shard00" / "progress.jsonl", [{"goal_id": "g0", "outcome": "error: timeout"}, {"goal_id": "g1", "outcome": "success"},
                                             {"goal_id": "g2", "outcome": "fail"}])
    jl(cell / "shard01" / "responses.jsonl", [{"goal_id": f"g{g}", "order": o, "response": f"turn {o} of g{g} (b)"} for g in (0, 3, 4, 5) for o in range(2)])
    jl(cell / "shard01" / "progress.jsonl", [{"goal_id": "g0", "outcome": "success"}, {"goal_id": "g3", "outcome": "fail"},
                                             {"goal_id": "g4", "outcome": "refused: target"}, {"goal_id": "g5", "outcome": "error: oom"}])
    return cell


def bon_store(root: pathlib.Path, name="bon_cell"):
    cell = root / name
    jl(cell / "bon_responses.jsonl", [{"goal_id": f"g{g}", "trial": 0, "sample_idx": s, "response": f"draw {s} of g{g}"} for g in range(3) for s in range(4)])
    return cell


def runs_tsv(path: pathlib.Path, rows, cols=("run_id", "eval_id", "store", "campaign")):
    path.write_text("\t".join(cols) + "\n" + "".join("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n" for r in rows))
    return str(path)


def registry(path: pathlib.Path, runs, judgments=(), judges=()):
    """A synthetic registry file. A run row: ``run_id``, ``eval_id``, and optionally ``campaign``, ``store_uri`` (the store its
    generation closed on) and ``instrument_judge`` (the judge its instrument names; default the Haiku v2 judge for strongreject_bon and
    the v1 judge otherwise)."""
    con = sqlite3.connect(str(path))
    con.executescript("""
      CREATE TABLE run (run_id TEXT PRIMARY KEY, campaign TEXT, eval_id TEXT, model_id TEXT, object_id TEXT, lane_source_legacy TEXT,
                        pass_label TEXT, trial_index INTEGER, token_cap_tokens INTEGER, store_uri TEXT, instrument_id TEXT);
      CREATE TABLE instrument (instrument_id TEXT PRIMARY KEY, eval_id TEXT, judge_id TEXT);
      CREATE TABLE judge (judge_id TEXT PRIMARY KEY, judge_model TEXT, rubric_name TEXT);
      CREATE TABLE judgment (judgment_id TEXT PRIMARY KEY, run_id TEXT, judge_id TEXT, status TEXT, finished_utc TEXT, n_items_judged INTEGER,
                             n_judge_refusals INTEGER, store_uri TEXT);
    """)
    for r in runs:
        ij = r.get("instrument_judge", HAIKU2 if r["eval_id"] == "strongreject_bon" else HAIKU)
        inst = f"inst-{r['eval_id']}-{str(ij)[:8]}"
        con.execute("INSERT OR IGNORE INTO instrument VALUES (?,?,?)", (inst, r["eval_id"], ij))
        con.execute("INSERT INTO run VALUES (?,?,?,?,?,?,?,?,?,?,?)", (r["run_id"], r.get("campaign", "synthcamp"), r["eval_id"], "qwen_synth",
                                                                       "o" * 64, None, None, 0, 4096,
                                                                       None if r.get("store_uri") is None else str(r["store_uri"]), inst))
    con.executemany("INSERT INTO judge VALUES (?,?,?)", [(HAIKU, "claude-haiku-4-5", "strongreject"), *judges])
    con.executemany("INSERT INTO judgment VALUES (?,?,?,?,?,?,?,?)", list(judgments))
    con.commit()
    con.close()
    return str(path)


@pytest.fixture
def world(tmp_path):
    goals = tmp_path / "goals_src.json"
    goals.write_text(json.dumps(GOALS))
    # the site the stores lie on: the always-declared local site when the temp dir is under the home directory, else the one declared
    site = "local" if str(tmp_path).startswith(SR.HOME + "/") else "synth"
    return {"tmp": tmp_path, "goals": str(goals), "static": static_cell(tmp_path), "mt": mt_cell(tmp_path), "bon": bon_store(tmp_path),
            "site": site, "n": [0]}


def build(world, rows, *extra, cols=("run_id", "eval_id", "store", "campaign"), reg=None):
    world["n"][0] += 1
    out = world["tmp"] / f"scope{world['n'][0]}"
    world["out"] = out
    argv = ["--runs", runs_tsv(world["tmp"] / f"runs{world['n'][0]}.tsv", rows, cols), "--goals", world["goals"], "--out", str(out),
            "--site", f"synth={world['tmp']}"]
    argv += ["--registry", reg] if reg else ["--no-registry"]
    code = SR.main(argv + list(extra))
    jobs = [json.loads(l) for l in open(out / "jobs.jsonl")] if (out / "jobs.jsonl").exists() else []
    return code, {j["run_id"]: j for j in jobs}, json.load(open(out / "scope_summary.json"))


def test_units_follow_the_store_layout_and_count_every_store_row(world, tmp_path):
    rows = [{"run_id": "synth:codeattack:1", "eval_id": "codeattack", "store": world["static"], "campaign": "synthcamp"},
            {"run_id": "synth:pair:1", "eval_id": "pair", "store": world["mt"], "campaign": "synthcamp"},
            {"run_id": "synth:bon:1", "eval_id": "strongreject_bon", "store": world["bon"], "campaign": "synthcamp"}]
    code, jobs, summ = build(world, rows)
    assert code == 0 and summ["jobs"] == 3 and not summ["problems"]
    st, mt, bon = jobs["synth:codeattack:1"], jobs["synth:pair:1"], jobs["synth:bon:1"]
    assert st["units"] == [{"type": "static", "path": str(world["static"])}] and st["kind"] == "sr" and st["category"] == "static"
    assert mt["category"] == "mt" and mt["kind"] == "sr" and mt["units"][0]["type"] == "mt"
    assert mt["units"][0]["files"] == sorted(str(p) for p in world["mt"].glob("shard*/responses.jsonl"))
    # the latest outcome decides: g0 errored then succeeded (kept), g4 refused and g5 errored (dropped)
    assert mt["units"][0]["drop_goals"] == ["g4", "g5"]
    assert bon["kind"] == "sr2" and bon["units"] == [{"type": "bon", "path": str(world["bon"] / "bon_responses.jsonl"),
                                                      "files": [str(world["bon"] / "bon_responses.jsonl")], "drop_goals": []}]
    for j in jobs.values():
        assert j["mapping_method"] == "explicit_run_list" and j["fold"] is None and j["sites"] == [world["site"]]
        assert j["haiku_judge_id"] is None and j["haiku_judgment_id"] is None and j["scope"] == "explicit run list (campaign synthcamp)"
    assert (world["out"] / "goals.json").read_bytes() == pathlib.Path(world["goals"]).read_bytes()
    # a scope is built once: the same directory again is refused
    with pytest.raises(SystemExit, match="already holds a jobs.jsonl"):
        SR.main(["--runs", str(world["tmp"] / "runs1.tsv"), "--goals", world["goals"], "--out", str(world["out"]), "--no-registry"])
    # the enumeration of these jobs counts every store row (minus the dropped goals' rows)
    E._init(str(world["out"] / "goals.json"))
    reps = {j["run_id"]: E.process(j)[0] for j in jobs.values()}
    assert reps["synth:codeattack:1"]["units"] == 18 and reps["synth:codeattack:1"]["unknown_goal"] == 0
    assert reps["synth:bon:1"]["units"] == 12
    # pair: shard00 3 goals x 3 turns, shard01 4 goals x 2 turns; g0 appears in both (the last shard wins: 2 turns), g4 and g5 dropped
    assert reps["synth:pair:1"]["units"] == 3 * 2 + 2 + 2 and reps["synth:pair:1"]["dropped_goal_rows"] == 4
    assert reps["synth:pair:1"]["last_shard_dropped"] == 3


def test_refusals_write_no_jobs(world, tmp_path):
    (world["bon"] / "bon_responses_final_rung.jsonl").write_text("")
    bad = [{"run_id": "synth:a", "eval_id": "xstest", "store": world["static"], "campaign": "c"},              # not a StrongREJECT eval
           {"run_id": "synth:b", "eval_id": "codeattack", "store": str(tmp_path / "nope"), "campaign": "c"},   # no store there
           {"run_id": "synth:c", "eval_id": "crescendo", "store": world["static"].parent / "empty", "campaign": "c"},
           {"run_id": "synth:d", "eval_id": "strongreject_bon", "store": world["bon"], "campaign": "c"},       # a climb beside the store
           {"run_id": "synth:e", "eval_id": "codeattack", "store": world["static"], "campaign": None}]         # no campaign, no registry
    (world["static"].parent / "empty").mkdir()
    code, jobs, summ = build(world, bad)
    assert code == 2 and not jobs and not (world["out"] / "jobs.jsonl").exists() and len(summ["problems"]) == 5
    assert not (world["out"] / "goals.json").exists()
    probs = " | ".join(summ["problems"])
    assert "not one of the StrongREJECT evals" in probs and "holds no shard" in probs and "folded" in probs and "campaign" in probs
    # a store under no declared site is refused (it would never be enumerated)
    if world["site"] == "local":
        pytest.skip("the temp dir lies under the home directory, the always-declared local site")
    out = tmp_path / "scope_nosite"
    code = SR.main(["--runs", runs_tsv(tmp_path / "r2.tsv", [{"run_id": "synth:f", "eval_id": "codeattack", "store": world["static"],
                                                              "campaign": "c"}]),
                    "--goals", world["goals"], "--out", str(out), "--no-registry"])
    assert code == 2 and "under no declared site" in " ".join(json.load(open(out / "scope_summary.json"))["problems"])


def test_the_registry_fills_the_run_and_its_haiku_judgment_and_refuses_disagreements(world, tmp_path):
    rid, rid2, rid3 = "synth:actorattack:1", "synth:crescendo:2", "synth:pasttense:3"
    reg = registry(tmp_path / "reg.sqlite",
                   [{"run_id": rid, "eval_id": "actorattack"}, {"run_id": rid2, "eval_id": "crescendo"},
                    {"run_id": rid3, "eval_id": "pasttense", "store_uri": world["static"]}],
                   judgments=[("hj-1", rid, HAIKU, "complete", "2026-01-01T00:00:00Z", 6, 0, str(world["mt"])),
                              ("hj-2", rid2, HAIKU, "complete", "2026-01-01T00:00:00Z", 6, 0, str(world["mt"]))])
    rows = [{"run_id": rid, "eval_id": "actorattack", "store": "@judgment"}, {"run_id": rid2, "eval_id": "crescendo", "store": world["mt"]},
            {"run_id": rid3, "eval_id": "pasttense", "store": world["static"]}]
    code, jobs, summ = build(world, rows, "--scope-note", "synthetic note", cols=("run_id", "eval_id", "store"), reg=reg)
    assert code == 0, summ["problems"]
    assert jobs[rid]["haiku_judge_id"] == HAIKU and jobs[rid]["haiku_judgment_id"] == "hj-1" and jobs[rid]["units"][0]["path"] == str(world["mt"])
    assert jobs[rid2]["haiku_judgment_id"] == "hj-2" and jobs[rid3]["haiku_judgment_id"] is None
    assert jobs[rid]["campaign"] == "synthcamp" and jobs[rid]["model_id"] == "qwen_synth" and jobs[rid]["scope"] == "synthetic note"
    assert summ["with_haiku_judgment"] == 2 and summ["stores_bound_by"] == {"judgment": 2, "run": 1}
    # a store other than the one the Haiku judgment graded, a wrong eval, an unknown run, a second filing: refused
    reg2 = registry(tmp_path / "reg2.sqlite", [{"run_id": rid, "eval_id": "actorattack"},
                                               {"run_id": rid3, "eval_id": "pasttense", "store_uri": world["static"]}],
                    judgments=[("hj-1", rid, HAIKU, "complete", None, 6, 0, str(world["mt"])),
                               ("mm-3", rid3, "f" * 64, "complete", None, 18, 0, None)],
                    judges=[("f" * 64, "minimax-m3", "strongreject")])
    rows = [{"run_id": rid, "eval_id": "actorattack", "store": world["static"]}, {"run_id": rid3, "eval_id": "codeattack", "store": world["static"]},
            {"run_id": "synth:none", "eval_id": "pasttense", "store": world["static"]}]
    code, jobs, summ = build(world, rows, cols=("run_id", "eval_id", "store"), reg=reg2)
    probs = " | ".join(summ["problems"])
    assert code == 2 and not jobs
    assert "not the one its Haiku judgment" in probs and "the registry's eval is 'pasttense'" in probs and "not a run in the registry" in probs
    rows = [{"run_id": rid3, "eval_id": "pasttense", "store": world["static"]}]
    code, jobs, summ = build(world, rows, cols=("run_id", "eval_id", "store"), reg=reg2)
    assert code == 2 and "already judged by minimax-m3" in summ["problems"][0]
    code, jobs, summ = build(world, rows, "--allow-follow-up", cols=("run_id", "eval_id", "store"), reg=reg2)
    assert code == 0 and rid3 in jobs


def test_verify_jobs_is_the_registry_half_of_a_cluster_side_build(world, tmp_path):
    rid = "synth:codeattack:9"
    code, jobs, _ = build(world, [{"run_id": rid, "eval_id": "codeattack", "store": world["static"], "campaign": "synthcamp"}])
    assert code == 0
    reg = registry(tmp_path / "reg.sqlite", [{"run_id": rid, "eval_id": "codeattack", "store_uri": world["static"]}])
    # the job built without the registry lacks model / object: the exact check names them
    assert SR.main(["--verify-jobs", str(world["out"] / "jobs.jsonl"), "--registry", reg]) == 2
    cols = ("run_id", "eval_id", "store", "campaign", "model_id", "object_id")
    code, jobs, _ = build(world, [{"run_id": rid, "eval_id": "codeattack", "store": world["static"], "campaign": "synthcamp",
                                   "model_id": "qwen_synth", "object_id": "o" * 64}], cols=cols)
    assert code == 0 and SR.main(["--verify-jobs", str(world["out"] / "jobs.jsonl"), "--registry", reg]) == 0
    assert SR.verify_jobs(str(world["out"] / "jobs.jsonl"), registry(tmp_path / "r3.sqlite", [{"run_id": rid, "eval_id": "codeattack",
                                                                                                   "campaign": "other"}]), False)


# ---------------------------------------------------------------------------------------------------------------------------------------
# the store is the run's: a listed store must be the one the registry ties to the run
# ---------------------------------------------------------------------------------------------------------------------------------------
COLS = ("run_id", "eval_id", "store")
#: the columns a cluster-side build (--no-registry) states so that the exact registry check passes
COLS_NOREG = ("run_id", "eval_id", "store", "campaign", "model_id", "object_id", "kind")
NOREG = {"campaign": "synthcamp", "model_id": "qwen_synth", "object_id": "o" * 64}
CLUSTER = "/synth-cluster/exp/"


def test_a_valid_store_of_another_run_is_refused_and_the_run_token_takes_the_registrys(world, tmp_path):
    """Runs with no Claude judgment, each with a valid store of its own on record: a run listed with the OTHER's store is refused,
    however valid that store's layout."""
    a, b, unclosed = "synth:codeattack:a", "synth:codeattack:b", "synth:codeattack:open"
    other = static_cell(tmp_path, "static_cell_b")
    reg = registry(tmp_path / "reg.sqlite", [{"run_id": a, "eval_id": "codeattack", "store_uri": world["static"]},
                                             {"run_id": b, "eval_id": "codeattack", "store_uri": other},
                                             {"run_id": unclosed, "eval_id": "codeattack"}])
    code, jobs, summ = build(world, [{"run_id": a, "eval_id": "codeattack", "store": other}], cols=COLS, reg=reg)
    assert code == 2 and not jobs and not (world["out"] / "jobs.jsonl").exists()
    assert "are not the one the registry records for the run" in summ["problems"][0] and str(world["static"]) in summ["problems"][0]
    # each with its own store, by path or by the @run token
    code, jobs, summ = build(world, [{"run_id": a, "eval_id": "codeattack", "store": world["static"]},
                                     {"run_id": b, "eval_id": "codeattack", "store": "@run"}], cols=COLS, reg=reg)
    assert code == 0, summ["problems"]
    assert jobs[a]["units"] == [{"type": "static", "path": str(world["static"])}] and jobs[b]["units"] == [{"type": "static", "path": str(other)}]
    assert summ["stores_bound_by"] == {"run": 2}
    # two stores for one run: the registry ties it to one
    code, jobs, summ = build(world, [{"run_id": a, "eval_id": "codeattack", "store": world["static"]},
                                     {"run_id": a, "eval_id": "codeattack", "store": other}], cols=COLS, reg=reg)
    assert code == 2 and "are not the one the registry records for the run" in summ["problems"][0]
    # a run the registry records no store for (its generation has not closed): nothing says which texts are its own
    code, jobs, summ = build(world, [{"run_id": unclosed, "eval_id": "codeattack", "store": world["static"]}], cols=COLS, reg=reg)
    assert code == 2 and "records no store" in summ["problems"][0]
    code, jobs, summ = build(world, [{"run_id": unclosed, "eval_id": "codeattack", "store": "@run"}], cols=COLS, reg=reg)
    assert code == 2 and "@run needs the registry and a store_uri on the run" in summ["problems"][0]
    # the tokens are the registry's: without it they name nothing
    code, jobs, summ = build(world, [dict(NOREG, run_id=a, eval_id="codeattack", store="@run")], cols=COLS_NOREG)
    assert code == 2 and "@run needs the registry" in summ["problems"][0]


def test_the_registrys_store_is_read_through_a_mirror_and_a_best_of_n_store_by_its_directory(world, tmp_path):
    """The registry records a store where it was generated (a path this host does not hold): --path-rewrite maps it to the mirror read
    here. A best-of-N run records its store's DIRECTORY; the reply file in it is the same store."""
    rid, rb = "synth:pair:m", "synth:bon:m"
    reg = registry(tmp_path / "reg.sqlite", [{"run_id": rid, "eval_id": "pair", "store_uri": CLUSTER + "mt_cell"},
                                             {"run_id": rb, "eval_id": "strongreject_bon", "store_uri": CLUSTER + "bon_cell/"}])
    rows = [{"run_id": rid, "eval_id": "pair", "store": "@run"},
            {"run_id": rb, "eval_id": "strongreject_bon", "store": world["bon"] / "bon_responses.jsonl"}]
    code, jobs, summ = build(world, rows, cols=COLS, reg=reg)          # no rewrite: the registry's path is neither here nor the listed one
    assert code == 2 and len(summ["problems"]) == 2
    mirror = ("--path-rewrite", f"{CLUSTER}={tmp_path}/")
    code, jobs, summ = build(world, rows, *mirror, cols=COLS, reg=reg)
    assert code == 0, summ["problems"]
    assert jobs[rid]["units"][0]["path"] == str(world["mt"]) and jobs[rid]["units"][0]["drop_goals"] == ["g4", "g5"]
    assert jobs[rb]["units"][0]["path"] == str(world["bon"] / "bon_responses.jsonl") and jobs[rb]["kind"] == "sr2"
    # the list may name the store where it was generated: the same rewrite reads it through the mirror
    code, jobs, summ = build(world, [{"run_id": rid, "eval_id": "pair", "store": CLUSTER + "mt_cell"}], *mirror, cols=COLS, reg=reg)
    assert code == 0 and jobs[rid]["units"][0]["path"] == str(world["mt"])
    # the reply file of ANOTHER best-of-N store (a valid one) is refused
    other = bon_store(tmp_path, "bon_cell_b")
    for store in (other, other / "bon_responses.jsonl"):
        code, jobs, summ = build(world, [{"run_id": rb, "eval_id": "strongreject_bon", "store": store}], *mirror, cols=COLS, reg=reg)
        assert code == 2 and "are not the one the registry records for the run" in summ["problems"][0]


def test_verify_jobs_holds_unit_paths_to_the_registrys_store_as_strings(world, tmp_path):
    a, rb = "synth:codeattack:a", "synth:bon:a"
    other = static_cell(tmp_path, "static_cell_b")
    row = dict(NOREG, run_id=a, eval_id="codeattack")
    reg = registry(tmp_path / "reg.sqlite", [{"run_id": a, "eval_id": "codeattack", "store_uri": world["static"]},
                                             {"run_id": rb, "eval_id": "strongreject_bon", "store_uri": world["bon"]}])
    # built where the registry is not, from a valid store that is ANOTHER run's: the build cannot know, the verify refuses
    code, jobs, summ = build(world, [dict(row, store=other)], cols=COLS_NOREG)
    assert code == 0 and jobs[a]["units"][0]["path"] == str(other) and summ["stores_bound_by"] == {"list": 1}
    wrong = str(world["out"] / "jobs.jsonl")
    probs = SR.verify_jobs(wrong, reg, False)
    assert len(probs) == 1 and "are not the one the registry records for the run" in probs[0]
    assert SR.main(["--verify-jobs", wrong, "--registry", reg]) == 2
    # the run's own store verifies; a best-of-N job's reply file is the store whose directory the registry records
    code, jobs, _ = build(world, [dict(row, store=world["static"]), dict(NOREG, run_id=rb, eval_id="strongreject_bon", store=world["bon"])],
                          cols=COLS_NOREG)
    right = str(world["out"] / "jobs.jsonl")
    assert code == 0 and SR.verify_jobs(right, reg, False) == []
    # strings, not files: a job built where the store is (a path this host does not hold) verifies against the registry's path ...
    job = dict(jobs[a], units=[{"type": "static", "path": CLUSTER + "static_cell"}])
    cluster_jobs = tmp_path / "jobs_cluster.jsonl"
    cluster_jobs.write_text(json.dumps(job) + "\n")
    reg_c = registry(tmp_path / "reg_c.sqlite", [{"run_id": a, "eval_id": "codeattack", "store_uri": CLUSTER + "static_cell/"},
                                                 {"run_id": rb, "eval_id": "strongreject_bon", "store_uri": CLUSTER + "bon_cell"}])
    assert SR.verify_jobs(str(cluster_jobs), reg_c, False) == []
    # ... and a job built from a mirror verifies through --path-rewrite, not without it
    assert len(SR.verify_jobs(right, reg_c, False)) == 2
    assert SR.verify_jobs(right, reg_c, False, [(CLUSTER, f"{tmp_path}/")]) == []
    assert SR.main(["--verify-jobs", right, "--registry", reg_c, "--path-rewrite", f"{CLUSTER}={tmp_path}/"]) == 0
    # a run the registry records no store for cannot be verified
    reg_n = registry(tmp_path / "reg_n.sqlite", [{"run_id": a, "eval_id": "codeattack"}])
    assert any("records no store" in p for p in SR.verify_jobs(str(cluster_jobs), reg_n, False))
    # a job whose category is not its eval's (it would be enumerated as another layout) is not an explicit-run-list job of that eval
    odd = tmp_path / "jobs_odd.jsonl"
    odd.write_text(json.dumps(dict(job, category="mt")) + "\n")
    assert any("not an explicit-run-list job of category static" in p for p in SR.verify_jobs(str(odd), reg_c, False))


def test_verify_jobs_holds_the_files_the_enumeration_reads_inside_the_unit_path(world, tmp_path):
    """The enumeration reads a multi-turn or best-of-N unit by its ``files``, never by its ``path``: a job whose path is the run's own
    store but whose files are ANOTHER run's reads the other run's texts. The verify holds every file inside the path, as strings."""
    rm, rb = "synth:pair:f", "synth:bon:f"
    other_mt, other_bon = mt_cell(tmp_path, "mt_cell_b"), bon_store(tmp_path, "bon_cell_b")
    reg = registry(tmp_path / "reg.sqlite", [{"run_id": rm, "eval_id": "pair", "store_uri": world["mt"]},
                                             {"run_id": rb, "eval_id": "strongreject_bon", "store_uri": world["bon"]}])
    code, jobs, _ = build(world, [dict(NOREG, run_id=rm, eval_id="pair", store=world["mt"]),
                                  dict(NOREG, run_id=rb, eval_id="strongreject_bon", store=world["bon"])], cols=COLS_NOREG)
    assert code == 0 and SR.verify_jobs(str(world["out"] / "jobs.jsonl"), reg, False) == []

    def verify(*edited):
        path = tmp_path / f"jobs_edit{len(list(tmp_path.glob('jobs_edit*')))}.jsonl"
        path.write_text("".join(json.dumps(j) + "\n" for j in edited))
        return SR.verify_jobs(str(path), reg, False)

    def with_unit(job, **unit):
        return dict(job, units=[dict(job["units"][0], **unit)])

    theirs = sorted(str(p) for p in other_mt.glob("shard*/responses.jsonl"))
    # the right path, another run's files (a valid store): the enumeration would read the other run
    probs = verify(with_unit(jobs[rm], files=theirs))
    assert len(probs) == 1 and "reads 2 file(s) that are not under its path" in probs[0] and str(world["mt"]) in probs[0]
    E._init(str(world["out"] / "goals.json"))
    assert E.process(with_unit(jobs[rm], files=theirs))[0]["units"] > 0          # ... and it does read them: the path is never opened
    # one file of its own and one of another run; a path that climbs out of the store; no file at all
    mine = jobs[rm]["units"][0]["files"]
    assert len(verify(with_unit(jobs[rm], files=mine + theirs[:1]))) == 1
    assert "not under its path" in verify(with_unit(jobs[rm], files=[str(world["mt"] / ".." / "mt_cell_b" / "shard00" / "responses.jsonl")]))[0]
    assert "names no file" in verify(with_unit(jobs[rm], files=[]))[0]
    # a sibling directory whose name merely starts with the store's is not under it
    assert "not under its path" in verify(with_unit(jobs[rm], files=[str(world["mt"]) + "_b/shard00/responses.jsonl"]))[0]
    # best-of-N: the unit's path IS the reply file, and the one file read must be it
    probs = verify(with_unit(jobs[rb], files=[str(other_bon / "bon_responses.jsonl")]))
    assert len(probs) == 1 and "that are not its path" in probs[0]
    assert "names no file" in verify(with_unit(jobs[rb], files=[]))[0]
    # a fold is read INSTEAD of the units: an explicit-run-list job has none
    fold = {"store": [{"type": "bon", "path": str(other_bon / "bon_responses.jsonl"), "files": [str(other_bon / "bon_responses.jsonl")]}],
            "climb": []}
    assert any("carries a fold" in p for p in verify(dict(jobs[rb], fold=fold)))
    assert verify(jobs[rm], jobs[rb]) == []


def test_the_rubric_kind_is_the_run_instruments_judge_not_the_eval_name(world, tmp_path):
    v1, v2, odd = "synth:bon:v1", "synth:bon:v2", "synth:codeattack:odd"
    bon_b = bon_store(tmp_path, "bon_cell_b")
    reg = registry(tmp_path / "reg.sqlite", [
        {"run_id": v1, "eval_id": "strongreject_bon", "store_uri": world["bon"], "instrument_judge": HAIKU},     # a best-of-N instrument on rubric v1
        {"run_id": v2, "eval_id": "strongreject_bon", "store_uri": bon_b},                                        # ... and one on v2
        {"run_id": odd, "eval_id": "codeattack", "store_uri": world["static"], "instrument_judge": "9" * 64}])
    code, jobs, summ = build(world, [{"run_id": v1, "eval_id": "strongreject_bon", "store": "@run"},
                                     {"run_id": v2, "eval_id": "strongreject_bon", "store": "@run"}], cols=COLS, reg=reg)
    assert code == 0, summ["problems"]
    assert jobs[v1]["kind"] == "sr" and jobs[v2]["kind"] == "sr2" and summ["by_kind"] == {"sr": 1, "sr2": 1}
    # the enumeration keys the v1 run's units under the v1 rubric
    E._init(str(world["out"] / "goals.json"))
    assert E.process(jobs[v1])[0]["units"] == 12 and {u[6] for u in E.process(jobs[v1])[1]} == {"sr"}
    # an instrument whose judge is neither Haiku StrongREJECT judge: the rubric cannot be derived
    code, jobs, summ = build(world, [{"run_id": odd, "eval_id": "codeattack", "store": "@run"}], cols=COLS, reg=reg)
    assert code == 2 and "rubric cannot be derived" in summ["problems"][0]
    # a kind the list states must be the instrument's
    code, jobs, summ = build(world, [{"run_id": v1, "eval_id": "strongreject_bon", "store": "@run", "kind": "sr2"}], cols=COLS + ("kind",), reg=reg)
    assert code == 2 and "the run's instrument is rubric sr, the list says kind 'sr2'" in summ["problems"][0]
    # without the registry: the eval name's default, or the list's kind; the verify holds the job to the instrument
    base = dict(NOREG, run_id=v1, eval_id="strongreject_bon", store=world["bon"])
    code, jobs, _ = build(world, [base], cols=COLS_NOREG)
    assert code == 0 and jobs[v1]["kind"] == "sr2"
    probs = SR.verify_jobs(str(world["out"] / "jobs.jsonl"), reg, False)
    assert len(probs) == 1 and "the run's instrument is rubric sr, the list says kind 'sr2'" in probs[0]
    code, jobs, _ = build(world, [dict(base, kind="sr")], cols=COLS_NOREG)
    assert code == 0 and jobs[v1]["kind"] == "sr" and SR.verify_jobs(str(world["out"] / "jobs.jsonl"), reg, False) == []
    with pytest.raises(SystemExit, match="kind 'v3' is not one of"):
        build(world, [dict(base, kind="v3")], cols=COLS_NOREG)


def test_the_run_list_helpers_are_string_rules(tmp_path):
    assert RL.rewrite("/a/b/c", [("/x/", "/y/"), ("/a/", "/m/")]) == "/m/b/c" and RL.rewrite("/a/b", []) == "/a/b"
    assert RL.recorded_store("/a//b/../c/#shard3", [("/a/", "/m/")]) == "/m/c" and RL.recorded_store(None) is None and RL.recorded_store("") is None
    assert RL.recorded_store("~/x") == os.path.join(os.path.expanduser("~"), "x")
    assert RL.parse_rules(["/a/=/b/"]) == [("/a/", "/b/")]
    with pytest.raises(SystemExit, match="expected FROM=TO"):
        RL.parse_rules(["/a/"])
    tsv = tmp_path / "t.tsv"
    tsv.write_text("# a comment\nrun_id\teval_id\n\nr1\t\n")
    assert list(RL.read_tsv(str(tsv), ("run_id",), ("eval_id",))) == [(2, {"run_id": "r1", "eval_id": None})]
    with pytest.raises(RL.Refused, match="unknown columns"):
        list(RL.read_tsv(str(tsv), ("run_id",), ()))
    assert SR.Refused is RL.Refused and SC.Refused is RL.Refused


# ---------------------------------------------------------------------------------------------------------------------------------------
# C/P/I
# ---------------------------------------------------------------------------------------------------------------------------------------
def pinned():
    xs = {f"xst{i:03d}": {"item_id": f"xst{i:03d}", "prompt": f"Synthetic xstest prompt {i}", "subset": "safe" if i < 5 else "unsafe"}
          for i in range(9)}
    ob = {f"orb{i:03d}": {"item_id": f"orb{i:03d}", "prompt": f"Synthetic or-bench prompt {i}"} for i in range(4)}
    return {"or_bench_hard": ob, "xstest": xs}


def cpi_cell(root: pathlib.Path, name: str, task: str, items, cap: int):
    cell = root / name
    (cell / "private").mkdir(parents=True)
    hdr = {"__spool_header__": {"task": f"judged/{task}", "gen_cfg": {"max_new_tokens": cap}, "task_version": "judged_v1"}}
    rows = [hdr] + [{"item_id": i, "text": f"synthetic reply to {i}"} for i in items]
    (cell / "private" / f"gens_{task}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return cell


def cpi_build(tmp_path, rows, *extra, name="cpi_scope", reg=None):
    tsv = tmp_path / f"{name}.tsv"
    cols = ("run_id", "eval_id", "cell", "cap", "campaign")
    tsv.write_text("\t".join(cols) + "\n" + "".join("\t".join(str(r.get(c, "")) for c in cols) + "\n" for r in rows))
    out = tmp_path / name
    code = SC.main(["--runs", str(tsv), "--out", str(out), *(["--registry", reg] if reg else ["--no-registry"]), *extra], pinned_fn=pinned)
    return code, out


def controls_cell(root: pathlib.Path, target: str, obj: str, cap: int, task: str, items, tail="t0__a0__aaaa1111__bbbb2222", base="wave/sampled"):
    """A cell in the controls store layout: ``<root>/<base>/controls_cap<cap>/<target>/cells/<obj>__controls__<tail>``."""
    return cpi_cell(root, f"{base}/controls_cap{cap}/{target}/cells/{obj}__controls__{tail}", task, items, cap)


def cpi_runs(out: pathlib.Path):
    return {r["run_id"]: r for r in (json.loads(l) for l in open(out / "runs.jsonl"))}


def spool_sha(cell: pathlib.Path, task: str) -> str:
    return hashlib.sha256((cell / "private" / f"gens_{task}.jsonl").read_bytes()).hexdigest()


def test_cpi_scope_takes_every_pinned_prompt_of_both_xstest_halves_at_the_named_rung(tmp_path):
    P = pinned()
    xc = cpi_cell(tmp_path, "xs2048", "xstest_v2", list(P["xstest"]), 2048)
    oc = cpi_cell(tmp_path, "orb8192", "or_bench_hard_1k", list(P["or_bench_hard"]), 8192)
    code, out = cpi_build(tmp_path, [{"run_id": "synth:xstest:1", "eval_id": "xstest", "cell": xc, "cap": 2048, "campaign": "synthcamp"},
                                     {"run_id": "synth:orb:1", "eval_id": "or_bench_hard", "cell": oc, "cap": 8192, "campaign": "synthcamp"}],
                          "--scope-note", "synthetic note")
    assert code == 0
    with gzip.open(out / "units.tsv.gz", "rt") as fh:
        hdr = next(fh).rstrip("\n").split("\t")
        units = [dict(zip(hdr, l.rstrip("\n").split("\t"))) for l in fh]
    # the scope_cpi.py header, unchanged
    assert hdr == ["run_id", "item_id", "trial", "sample", "turn", "custom_id", "rubric", "eval_id", "orig_label", "orig_status", "note"]
    xs = [u for u in units if u["run_id"] == "synth:xstest:1"]
    assert len(xs) == 9 and sorted(json.loads(u["note"])["subset"] for u in xs).count("unsafe") == 4
    assert all(u["custom_id"] == C.request_key("xstest_cpi", P["xstest"][u["item_id"]]["prompt"], f"synthetic reply to {u['item_id']}")
               for u in xs)
    assert len([u for u in units if u["rubric"] == "or_bench_cpi"]) == 4 and all((u["sample"], u["turn"]) == ("0", "-1") for u in units)
    runs = {r["run_id"]: r for r in (json.loads(l) for l in open(out / "runs.jsonl"))}
    x = runs["synth:xstest:1"]
    assert x["sonnet_judge_id"] is None and x["sonnet_judgment_id"] is None and x["rubric"] == "xstest_cpi"
    assert x["judged_store"] == str(xc) and x["judged_cap"] == 2048 and x["sources"][0]["cap"] == 2048 and x["scope"] == "synthetic note"
    # without the registry nothing vouches for the cell but the list: its origin is the cell as listed, and the spool's sha256 is recorded
    assert x["judged_store_origin"] == str(xc) and x["judged_spool_sha256"] == spool_sha(xc, "xstest_v2") == x["sources"][0]["spool_sha256"]
    with gzip.open(out / "texts.jsonl.gz", "rt") as fh:
        texts = [json.loads(l) for l in fh]
    assert len(texts) == 13 and set(texts[0]) == {"k", "rubric", "item_id", "prompt", "response"}
    rep = json.load(open(out / "report.json"))
    assert rep["units"] == 13 and rep["xstest_subset"] == "all" and not rep["problems"] and rep["stores_bound_by"] == {"list": 2}
    with pytest.raises(SystemExit, match="already holds a scope"):
        SC.main(["--runs", str(tmp_path / "cpi_scope.tsv"), "--out", str(out), "--no-registry"], pinned_fn=pinned)


def test_cpi_scope_refuses_a_missing_generation_an_unpinned_item_and_the_wrong_rung(tmp_path):
    P = pinned()
    miss = cpi_cell(tmp_path, "miss", "xstest_v2", list(P["xstest"])[:-1], 2048)
    extra = cpi_cell(tmp_path, "extra", "or_bench_hard_1k", list(P["or_bench_hard"]) + ["orb999"], 8192)
    wrong = cpi_cell(tmp_path, "wrong", "xstest_v2", list(P["xstest"]), 8192)
    code, out = cpi_build(tmp_path, [{"run_id": "synth:x1", "eval_id": "xstest", "cell": miss, "cap": 2048, "campaign": "c"},
                                     {"run_id": "synth:o1", "eval_id": "or_bench_hard", "cell": extra, "cap": 8192, "campaign": "c"},
                                     {"run_id": "synth:x2", "eval_id": "xstest", "cell": wrong, "cap": 2048, "campaign": "c"}])
    rep = json.load(open(out / "report.json"))
    probs = " | ".join(p["problem"] for p in rep["problems"])
    assert code == 2 and not (out / "units.tsv.gz").exists() and not (out / "runs.jsonl").exists()
    assert "have no generation" in probs and "are not pinned" in probs and "max_new_tokens is 8192" in probs


def test_cpi_cell_must_be_a_rung_of_the_runs_own_store(tmp_path):
    """The registry records a run's cell at the rung its generation closed on. The judged rung may be another rung of the same ladder
    (another rung directory, another cell-name tail); the root, the target directory and the object must be the run's. A VALID cell of
    another run is refused."""
    P = pinned()
    xs, ob = list(P["xstest"]), list(P["or_bench_hard"])
    T, A, B = "qwen_synth_s1_t0", "objA--lora", "objB--relu"
    x8192 = controls_cell(tmp_path, T, A, 8192, "xstest_v2", xs, tail="t0__a0__aaaa1111__cccc8192")     # the registry's store of run x
    x2048 = controls_cell(tmp_path, T, A, 2048, "xstest_v2", xs, tail="t0__a0__aaaa1111__dddd2048")     # the rung judged
    o8192 = controls_cell(tmp_path, T, B, 8192, "or_bench_hard_1k", ob)
    reg = registry(tmp_path / "reg.sqlite", [{"run_id": "synth:xstest:1", "eval_id": "xstest", "store_uri": x8192},
                                             {"run_id": "synth:orb:1", "eval_id": "or_bench_hard", "store_uri": o8192},
                                             {"run_id": "synth:xstest:open", "eval_id": "xstest"},
                                             {"run_id": "synth:xstest:flat", "eval_id": "xstest", "store_uri": tmp_path / "flat_store"}])
    xrow = {"run_id": "synth:xstest:1", "eval_id": "xstest", "cap": 2048, "campaign": "synthcamp"}
    code, out = cpi_build(tmp_path, [dict(xrow, cell=x2048), {"run_id": "synth:orb:1", "eval_id": "or_bench_hard", "cell": o8192, "cap": 8192,
                                                              "campaign": "synthcamp"}], reg=reg, name="own")
    assert code == 0, json.load(open(out / "report.json"))["problems"]
    x = cpi_runs(out)["synth:xstest:1"]
    assert x["judged_store"] == str(x2048) == x["judged_store_origin"] and x["judged_cap"] == 2048 and x["store_bound_by"] == "run"
    assert x["judged_spool_sha256"] == spool_sha(x2048, "xstest_v2") == x["sources"][0]["spool_sha256"]
    assert json.load(open(out / "report.json"))["stores_bound_by"] == {"run": 2}
    # valid cells of OTHER runs, each holding a full xstest spool at the judged cap: another object, another target, another root
    others = {"object": controls_cell(tmp_path, T, B, 2048, "xstest_v2", xs),
              "target": controls_cell(tmp_path, "qwen_synth_s2_t0", A, 2048, "xstest_v2", xs),
              "root": controls_cell(tmp_path, T, A, 2048, "xstest_v2", xs, base="smoke/sampled")}
    for part, cell in others.items():
        code, out = cpi_build(tmp_path, [dict(xrow, cell=cell)], reg=reg, name=f"other_{part}")
        problem = json.load(open(out / "report.json"))["problems"][0]["problem"]
        assert code == 2 and not (out / "runs.jsonl").exists() and not (out / "units.tsv.gz").exists()
        assert "is not a rung of the run's own store" in problem and f"its {part} " in problem
        # the same cell without the registry is taken on the list's word: the registry is what refuses it
        assert cpi_build(tmp_path, [dict(xrow, cell=cell)], name=f"other_{part}_noreg")[0] == 0
    # a run the registry records no store for; a registry store or a cell that is not a controls cell
    flat = cpi_cell(tmp_path, "flat_cell", "xstest_v2", xs, 2048)
    for name, row, why in (("open", dict(xrow, run_id="synth:xstest:open", cell=x2048), "records no store"),
                           ("flat_reg", dict(xrow, run_id="synth:xstest:flat", cell=x2048), "is not a controls cell"),
                           ("flat_cell", dict(xrow, cell=flat), "is not a controls cell")):
        code, out = cpi_build(tmp_path, [row], reg=reg, name=name)
        assert code == 2 and why in json.load(open(out / "report.json"))["problems"][0]["problem"]


def test_cpi_cell_read_through_a_mirror_records_where_it_was_generated(tmp_path):
    P = pinned()
    xs = list(P["xstest"])
    T, A = "qwen_synth_s1_t0", "objA--lora"
    x2048 = controls_cell(tmp_path, T, A, 2048, "xstest_v2", xs, tail="t0__a0__aaaa1111__dddd2048")
    cluster = "/synth-cluster/exp"
    origin = f"{cluster}/wave/sampled/controls_cap2048/{T}/cells/{A}__controls__t0__a0__aaaa1111__dddd2048"
    reg = registry(tmp_path / "reg.sqlite", [{"run_id": "synth:xstest:1", "eval_id": "xstest",
                                              "store_uri": f"{cluster}/wave/sampled/controls_cap8192/{T}/cells/{A}__controls__t0__a0__aaaa1111__cccc8192"}])
    xrow = {"run_id": "synth:xstest:1", "eval_id": "xstest", "cap": 2048, "campaign": "synthcamp"}
    # the registry's root is the cluster's: the mirror cell is refused until --path-rewrite says the mirror is that root
    code, out = cpi_build(tmp_path, [dict(xrow, cell=x2048)], reg=reg, name="no_rewrite")
    problem = json.load(open(out / "report.json"))["problems"][0]["problem"]
    assert code == 2 and "its root " in problem and "--path-rewrite" in problem
    mirror = ("--path-rewrite", f"{cluster}/={tmp_path}/")
    for name, listed in (("mirror_listed", x2048), ("origin_listed", origin)):       # the list may name the mirror or the cluster path
        code, out = cpi_build(tmp_path, [dict(xrow, cell=listed)], *mirror, reg=reg, name=name)
        assert code == 0, json.load(open(out / "report.json"))["problems"]
        x = cpi_runs(out)["synth:xstest:1"]
        assert x["judged_store"] == str(x2048) and x["judged_store_origin"] == origin == x["sources"][0]["origin"]
        assert x["judged_spool_sha256"] == spool_sha(x2048, "xstest_v2")
    assert SC.cell_parts(origin) == {"root": f"{cluster}/wave/sampled", "rung": "2048", "target": T, "object": A,
                                     "key": "t0__a0__aaaa1111", "rung_hash": "dddd2048"}
    assert SC.cell_parts(f"{cluster}/wave/sampled/{T}/cells/{A}__controls__x") is None


def test_cpi_cell_name_matches_in_full_and_keeps_the_runs_trial_key(tmp_path):
    """Along one run's ladder only the rung directory and the LAST hash of the cell's name vary. A moved-aside sibling of the run's own
    cell (``<cell>.failed_<job>``), a cell of another trial and a cell with another first hash each hold a full, valid spool at the
    judged cap under the run's own root, target and object: each is refused."""
    P = pinned()
    xs = list(P["xstest"])
    T, A = "qwen_synth_s1_t0", "objA--lora"
    x8192 = controls_cell(tmp_path, T, A, 8192, "xstest_v2", xs, tail="t0__a0__aaaa1111__cccc8192")     # the registry's store
    own = controls_cell(tmp_path, T, A, 2048, "xstest_v2", xs, tail="t0__a0__aaaa1111__dddd2048")
    reg = registry(tmp_path / "reg.sqlite", [{"run_id": "synth:xstest:1", "eval_id": "xstest", "store_uri": x8192},
                                             {"run_id": "synth:xstest:aside", "eval_id": "xstest", "store_uri": str(x8192) + ".failed_4242"}])
    xrow = {"run_id": "synth:xstest:1", "eval_id": "xstest", "cap": 2048, "campaign": "synthcamp"}
    assert cpi_build(tmp_path, [dict(xrow, cell=own)], reg=reg, name="own")[0] == 0
    refused = {"aside": (controls_cell(tmp_path, T, A, 2048, "xstest_v2", xs, tail="t0__a0__aaaa1111__dddd2048.failed_4242"),
                         "is not a controls cell"),
               "trial": (controls_cell(tmp_path, T, A, 2048, "xstest_v2", xs, tail="t1__a0__aaaa1111__dddd2048"),
                         "its trial key 't1__a0__aaaa1111' (the registry's: 't0__a0__aaaa1111')"),
               "hash": (controls_cell(tmp_path, T, A, 2048, "xstest_v2", xs, tail="t0__a0__eeee5555__dddd2048"),
                        "its trial key 't0__a0__eeee5555'"),
               "index": (controls_cell(tmp_path, T, A, 2048, "xstest_v2", xs, tail="t0__a1__aaaa1111__dddd2048"), "its trial key "),
               "short": (controls_cell(tmp_path, T, A, 2048, "xstest_v2", xs, tail="t0__a0__aaaa1111"), "is not a controls cell")}
    for name, (cell, why) in refused.items():
        code, out = cpi_build(tmp_path, [dict(xrow, cell=cell)], reg=reg, name=f"refused_{name}")
        problem = json.load(open(out / "report.json"))["problems"][0]["problem"]
        assert code == 2 and why in problem and not (out / "runs.jsonl").exists(), (name, problem)
        assert cpi_build(tmp_path, [dict(xrow, cell=cell)], name=f"refused_{name}_noreg")[0] == 0      # a valid spool: the registry refuses it
    # a registry store that is itself a moved-aside name is no controls cell either
    code, out = cpi_build(tmp_path, [dict(xrow, run_id="synth:xstest:aside", cell=own)], reg=reg, name="aside_reg")
    assert code == 2 and "the registry's store" in json.load(open(out / "report.json"))["problems"][0]["problem"]
    assert SC.cell_parts(str(own)) and SC.cell_parts(str(own) + ".failed_4242") is None and SC.cell_parts(str(own) + "\n") is None
