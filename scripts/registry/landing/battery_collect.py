#!/usr/bin/env python3
"""The per-commit test BATTERY's plan and verdict (stdlib only; lives in scripts/registry/landing/ beside the gate code and
is shipped to the cluster by battery_slurm.sh as $SHARED/gates/battery/bin/battery_collect.py).

A battery is a targeted pytest run — the test files a change touches, found by grep — on cluster A CPU array tasks from an
exported tree, never on the grading host. ``gate_task.py --files … --rerun-label L`` runs exactly the named files and records under
``<results>/<sha>/reruns/L/``; given the SAME list and label to every task of an array it runs the whole list K times and races on
one record, so a battery is planned first:

  plan        allocate the requested files into K DISJOINT lists by predicted duration (LPT: longest file first, into the least
              loaded task, ties to the task with the fewest files then the lowest index; a file unknown to the duration table
              takes the median of the known ones; every prediction has a positive floor, PREDICTED_FLOOR_S, so a table of
              observed 0.0-second files -- fast probes rounding to nothing -- never collapses K tasks into one) with a per-task budget
              (default 8 min of predicted test time; K grows past the default only while a task over the budget holds more than one
              file -- a single file longer than the budget runs alone, named in over_budget_tasks, and K stops there) and a default K of
              ceil(total predicted / 6 min); the plan names every task's
              files, the pytest bound and the array's walltime, and is the ONE input battery.sbatch reads — task i runs
              ``plan.tasks[i].files`` under ``--rerun-label <label>/t<i>`` → ``<results>/<sha>/reruns/<label>/t<i>/`` (gate_task
              nests the label as a path), so no two tasks share a list or a record path
  durations   per-file test seconds from a finished gate's shard JUnit reports (``<results>/<sha>/shards/*/junit.xml``: the
              baseline summary's per-file durations) — the plan's duration table
  collect     the verdict: EXACTLY K records at the planned paths, each carrying the exported tree's sha, its own index, count
              and the plan's list for that index, the ARRAY that wrote it (``slurm_array_job_id`` == the attempt's recorded
              ``--array-job-id``, and ``slurm_array_task_id`` == its index when the record carries it; gate_task.py writes both
              from Slurm's environment), a terminal PASS/FAIL status and at least one test; the union of the lists is the
              requested set with no duplicate; a missing, duplicate, wrong-sha, wrong-array, extra, non-terminal or empty record
              REFUSES the verdict (``battery incomplete: …``, exit 4, status INCOMPLETE) — a verdict is PASS or FAIL and nothing
              else, and records of two arrays (a double submission of one label) can never mix into one: a record from another
              array is an identity refusal, as is a collection with no array id to bind the records to.
              Writes <prefix>.verdict.json, <prefix>.junit.xml (every task's report under one <testsuites>) and <prefix>.log
              (every task's pytest.log), and records each task's elapsed and per-file observed seconds (the next plan's table).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

PLAN_SCHEMA = "battery_plan/1"
VERDICT_SCHEMA = "battery_verdict/1"
DURATIONS_SCHEMA = "battery_durations/1"
BUDGET_S_DEFAULT = 480          # predicted test seconds per task, the plan's cap (8 min)
TARGET_S_DEFAULT = 360          # predicted test seconds per task the default K aims at (6 min)
UNKNOWN_S_DEFAULT = 60.0        # a file's predicted seconds when NO duration is known at all (no table, or an empty one)
#: the floor on every file's predicted seconds (known, observed or filled): a duration table can carry 0.0 for a file whose tests
#: ran in under a millisecond (the landing-chain tests' mock probes on a fast node), and an LPT fed only zeros never changes a task's
#: load, so every file landed in task 0 and check_plan refused the K-task re-plan with "task 2 has an empty list"
PREDICTED_FLOOR_S = 1.0
TIMEOUT_FLOOR_S = 600           # pytest's bound is never below this …
TIMEOUT_CAP_S = 3600            # … nor above this (gate_array's re-run bound)
#: seconds the array's walltime carries beyond pytest's bound: the task copies the tree to node-local scratch (~40 s measured) and,
#: with --local-venv, the venv (169–261 s measured on gate 876172's tasks), verifies both and writes the record afterwards
OVERHEAD_S = 600
TERMINAL = ("PASS", "FAIL")
TASK_FLAGS_DEFAULT = ["--local-venv", "--hide-slurm"]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def hms(seconds: int) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


# ---- durations ---------------------------------------------------------------------------------------------------------------
def file_of_classname(classname: str) -> str:
    """``tests.test_x.TestK`` -> ``tests/test_x.py``: the module is the classname up to its last ``test_*`` component (pytest
    names test modules so; measure/durations.py applies the same rule)."""
    parts = classname.split(".") if classname else []
    if not parts:
        return ""
    k = max((i for i, p in enumerate(parts) if p.startswith("test_")), default=len(parts) - 1) + 1
    return "/".join(parts[:k]) + ".py"


def junit_file_seconds(path: Path) -> dict:
    """Per-file wall seconds from one JUnit report (the sum of its ``<testcase time>`` attributes)."""
    out: dict = {}
    root = ET.parse(str(path)).getroot()
    for tc in root.iter("testcase"):
        try:
            t = float(tc.get("time") or 0.0)
        except ValueError:
            t = 0.0
        f = file_of_classname(tc.get("classname") or "")
        if f:
            out[f] = round(out.get(f, 0.0) + t, 3)
    return out


def durations_from_reports(reports: list) -> dict:
    files: dict = {}
    for p in reports:
        for f, t in junit_file_seconds(Path(p)).items():
            files[f] = round(files.get(f, 0.0) + t, 3)
    vals = sorted(files.values())
    return {"schema": DURATIONS_SCHEMA, "files": files, "n_files": len(files),
            "median_s": round(statistics.median(vals), 3) if vals else None, "total_s": round(sum(vals), 3),
            "reports": [str(p) for p in reports], "rendered_utc": utc_now()}


def cmd_durations(a) -> int:
    reports = sorted(Path(a.shards_dir).glob("*/junit.xml")) if a.shards_dir else []
    reports += [Path(p) for p in (a.junit or [])]
    if not reports:
        print(f"battery durations: no junit.xml under {a.shards_dir or '(no --shards-dir)'} and no --junit", file=sys.stderr)
        return 2
    rep = durations_from_reports(reports)
    if a.out:
        write_json_atomic(Path(a.out), rep)
        print(f"battery durations: {rep['n_files']} files from {len(reports)} reports, median {rep['median_s']} s -> {a.out}")
    else:
        print(json.dumps(rep, indent=1, sort_keys=True))
    return 0


def load_durations(paths: list) -> tuple[dict, list]:
    """The merged duration table of ``paths`` (later files override earlier: pass the baseline first, observed seconds last) and
    the sources read; a missing or unreadable file is skipped and named."""
    table: dict = {}
    sources = []
    for p in paths or []:
        p = Path(p)
        if not p.is_file():
            sources.append({"path": str(p), "state": "missing"})
            continue
        try:
            d = json.loads(p.read_text())
            files = d.get("files") or {}
        except (ValueError, AttributeError):
            sources.append({"path": str(p), "state": "unreadable"})
            continue
        table.update({k: float(v) for k, v in files.items()})
        sources.append({"path": str(p), "state": "read", "n_files": len(files)})
    return table, sources


# ---- plan --------------------------------------------------------------------------------------------------------------------
def lpt(files: list, predicted: dict, count: int) -> list:
    """Longest-processing-time-first: files by predicted seconds descending (ties by name) into the least loaded of ``count`` tasks
    (ties to the task with the FEWEST files, then the lowest index -- so equal loads, zeros included, still spread the files and K
    tasks are never left empty while files remain); each task's list is then sorted by name. Deterministic: the same files and
    predictions give the same allocation. Returns ``[(files, predicted_s), …]``."""
    loads = [0.0] * count
    bins: list = [[] for _ in range(count)]
    for f in sorted(files, key=lambda x: (-predicted[x], x)):
        i = min(range(count), key=lambda j: (loads[j], len(bins[j]), j))
        bins[i].append(f)
        loads[i] += predicted[f]
    return [(sorted(b), round(load, 3)) for b, load in zip(bins, loads)]


def make_plan(sha: str, label: str, files: list, table: dict, *, tasks: int | None = None, budget_s: int = BUDGET_S_DEFAULT,
              target_s: int = TARGET_S_DEFAULT, task_flags: list | None = None, sources: list | None = None, branch: str | None = None) -> dict:
    requested = sorted(set(files))
    if not requested:
        raise ValueError("battery plan: the file list is empty")
    if len(requested) != len(files):
        dup = sorted({f for f in files if files.count(f) > 1})
        raise ValueError(f"battery plan: duplicate file(s) in the request: {dup}")
    known = {f: max(float(table[f]), PREDICTED_FLOOR_S) for f in requested if f in table}
    unknown = [f for f in requested if f not in table]
    fill = max(round(statistics.median(table.values()), 3), PREDICTED_FLOOR_S) if table else UNKNOWN_S_DEFAULT
    predicted = {**known, **{f: fill for f in unknown}}
    floored = sorted(f for f in known if float(table[f]) < PREDICTED_FLOOR_S)
    total = round(sum(predicted.values()), 3)
    n = len(requested)
    forced = tasks is not None
    k = min(max(1, int(tasks)), n) if forced else max(1, math.ceil(total / float(target_s)))
    k = min(k, n)
    # The budget grows K only while it can help: a task over the budget that holds ONE file cannot get shorter by adding tasks, so K stops
    # growing once every over-budget task is a single file (the file runs alone and is named in over_budget_tasks; the pytest bound is sized
    # from it). Before this, one file predicted past the budget drove K to n -- one battery ran 414 one-file tasks for 5,810 s of predicted
    # tests, ~17 tasks' worth (2026-10-04). --tasks caps K exactly, over budget or not.
    while True:
        alloc = lpt(requested, predicted, k)
        max_task = max(load for _, load in alloc)
        over = [i for i, (_, load) in enumerate(alloc) if load > budget_s]
        if forced or not over or k >= n or all(len(alloc[i][0]) == 1 for i in over):
            break
        k += 1
    timeout_s = int(min(TIMEOUT_CAP_S, max(TIMEOUT_FLOOR_S, math.ceil(2 * max_task + 300))))
    walltime_s = timeout_s + OVERHEAD_S
    return {"schema": PLAN_SCHEMA, "sha": sha, "label": label, "branch": branch, "requested_files": requested, "count": k,
            "tasks": [{"index": i, "files": fs, "predicted_s": load} for i, (fs, load) in enumerate(alloc)],
            "predicted": {"total_s": total, "max_task_s": round(max_task, 3), "unknown_fill_s": fill, "unknown_files": unknown,
                          "known_files": len(known), "over_budget_tasks": over, "floor_s": PREDICTED_FLOOR_S, "floored_files": floored},
            "budget_s": int(budget_s), "target_s": int(target_s), "tasks_forced": forced, "timeout_s": timeout_s,
            "walltime_s": walltime_s, "walltime": hms(walltime_s),
            "task_flags": list(TASK_FLAGS_DEFAULT if task_flags is None else task_flags),
            "durations_sources": sources or [], "planned_utc": utc_now()}


def check_plan(plan: dict) -> list:
    """The plan's own invariants, re-checked by ``collect`` so a doctored or hand-edited plan cannot pass a record through:
    K lists, indices 0..K-1 in order, every list non-empty, the union == requested_files, no file in two tasks."""
    problems = []
    tasks = plan.get("tasks") or []
    k = plan.get("count")
    if k != len(tasks) or not tasks:
        problems.append(f"count {k} != {len(tasks)} task lists")
    if [t.get("index") for t in tasks] != list(range(len(tasks))):
        problems.append("task indices are not 0..K-1 in order")
    seen: dict = {}
    for t in tasks:
        fs = t.get("files") or []
        if not fs:
            problems.append(f"task {t.get('index')} has an empty list")
        for f in fs:
            if f in seen:
                problems.append(f"file {f} is claimed by tasks {seen[f]} and {t.get('index')}")
            seen[f] = t.get("index")
    requested = sorted(plan.get("requested_files") or [])
    if sorted(seen) != requested:
        problems.append(f"the union of the task lists is not the requested set: missing {sorted(set(requested) - set(seen))}, "
                        f"extra {sorted(set(seen) - set(requested))}")
    return problems


def cmd_plan(a) -> int:
    table, sources = load_durations(a.durations)
    flags = None if a.task_flags is None else a.task_flags.split()
    try:
        plan = make_plan(a.sha, a.label, list(a.files), table, tasks=a.tasks, budget_s=a.budget_s, target_s=a.target_s,
                         task_flags=flags, sources=sources, branch=a.branch)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    problems = check_plan(plan)
    if problems:
        print("battery plan: invariant violated: " + "; ".join(problems), file=sys.stderr)
        return 2
    write_json_atomic(Path(a.out), plan)
    p = plan["predicted"]
    print(f"battery plan: label={plan['label']} sha={plan['sha'][:12]} files={len(plan['requested_files'])} tasks={plan['count']} "
          f"predicted total={p['total_s']}s max_task={p['max_task_s']}s known={p['known_files']} unknown={len(p['unknown_files'])} "
          f"(fill {p['unknown_fill_s']}s) budget={plan['budget_s']}s timeout_s={plan['timeout_s']} walltime={plan['walltime']} "
          f"over_budget_tasks={p['over_budget_tasks']} -> {a.out}")
    for t in plan["tasks"]:
        print(f"  t{t['index']}: {t['predicted_s']}s {len(t['files'])} files: " + " ".join(t["files"]))
    return 0


# ---- collect -----------------------------------------------------------------------------------------------------------------
class Incomplete(Exception):
    """A battery whose records do not add up to a verdict; ``kind`` classifies it for the driver (``timeout`` re-plans with K+1)."""

    def __init__(self, kind: str, why: str):
        super().__init__(why)
        self.kind = kind
        self.why = why


def _read_record(p: Path) -> dict | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except ValueError:
        return None


def verify_records(plan: dict, records: Path, sacct: dict | None = None, array_job_id: str | None = None) -> list:
    """Every task's record, verified against the plan AND bound to the attempt's array (``array_job_id``, the id the submission
    recorded): raises Incomplete at the first defect (named by task and kind). A record whose ``slurm_array_job_id`` is not that
    array was written by another submission of the same label — identity, never a verdict."""
    problems = check_plan(plan)
    if problems:
        raise Incomplete("plan", "the plan violates its invariants: " + "; ".join(problems))
    array_job_id = str(array_job_id).strip() if array_job_id is not None else ""
    if not array_job_id:
        raise Incomplete("identity", "no array job id was recorded for this attempt: its records cannot be bound to the submission that wrote them")
    k, sha, label = plan["count"], plan["sha"], plan["label"]
    expected = [f"t{i}" for i in range(k)]
    if not records.is_dir():
        raise Incomplete("missing", f"no records directory at {records}")
    present = sorted(os.listdir(records))
    extra = sorted(set(present) - set(expected))
    if extra:
        raise Incomplete("extra", f"extra record(s) beside the {k} planned tasks: {extra}")
    missing = [t for t in expected if t not in present]
    if missing:
        states = {t: (sacct or {}).get(t[1:]) for t in missing} if sacct else {}
        raise Incomplete("timeout" if states and all(str(s or "").startswith("TIMEOUT") for s in states.values()) else "missing",
                         f"missing task record(s): {missing}" + (f" (slurm states {states})" if states else ""))
    out = []
    for i in range(k):
        d = records / f"t{i}"
        rec = _read_record(d / "summary.json")
        if rec is None:
            state = (sacct or {}).get(str(i))
            raise Incomplete("timeout" if str(state or "").startswith("TIMEOUT") else "missing",
                             f"task {i}: no readable summary.json at {d}" + (f" (slurm state {state})" if state else ""))
        if rec.get("sha") != sha:
            raise Incomplete("identity", f"task {i}: record sha {rec.get('sha')} is not the battery's tree {sha}")
        got_array = rec.get("slurm_array_job_id")
        if str(got_array if got_array is not None else "").strip() != array_job_id:
            raise Incomplete("identity", f"task {i}: record was written by array {got_array!r} (slurm_array_job_id), not this attempt's "
                                         f"array {array_job_id} — another submission of label {label}?")
        got_task = rec.get("slurm_array_task_id")
        if got_task is not None and str(got_task).strip() != str(i):
            raise Incomplete("identity", f"task {i}: record's slurm_array_task_id {got_task!r} is not its task index {i}")
        shard = rec.get("shard") or {}
        if shard.get("index") != i or shard.get("count") != k:
            raise Incomplete("identity", f"task {i}: record claims shard {shard.get('index')}/{shard.get('count')}, planned {i}/{k}")
        want = plan["tasks"][i]["files"]
        if list(shard.get("files") or []) != list(want):
            raise Incomplete("identity", f"task {i}: record ran {shard.get('files')}, planned {want}")
        rd = str(rec.get("results_dir") or "")
        if not rd.replace(os.sep, "/").endswith(f"/{sha}/reruns/{label}/t{i}"):
            raise Incomplete("identity", f"task {i}: record's results_dir {rd!r} is not …/{sha}/reruns/{label}/t{i}")
        status = rec.get("status")
        if status not in TERMINAL:
            raise Incomplete("timeout" if status == "TIMEOUT" else "refused",
                             f"task {i}: status {status} is not a result" + (f" ({rec.get('error')})" if rec.get("error") else ""))
        counts = rec.get("counts")
        if not isinstance(counts, dict) or int(counts.get("tests") or 0) < 1:
            raise Incomplete("empty", f"task {i}: ran no tests (counts {counts})")
        if not (d / "junit.xml").is_file():
            raise Incomplete("missing", f"task {i}: no junit.xml beside its record")
        out.append(rec)
    return out


def merge_junit(reports: list, dest: Path) -> None:
    root = ET.Element("testsuites")
    for p in reports:
        r = ET.parse(str(p)).getroot()
        for s in ([r] if r.tag == "testsuite" else list(r.iter("testsuite"))):
            root.append(s)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    ET.ElementTree(root).write(str(tmp), encoding="unicode", xml_declaration=True)
    os.replace(tmp, dest)


def collect(plan: dict, records: Path, prefix: Path, *, sacct: dict | None = None, array_job_id: str | None = None,
            venv_job_id: str | None = None, extra: dict | None = None) -> dict:
    base = {"schema": VERDICT_SCHEMA, "label": plan.get("label"), "sha": plan.get("sha"), "branch": plan.get("branch"),
            "tasks": plan.get("count"), "requested_files": plan.get("requested_files"), "records": str(records),
            "array_job_id": array_job_id, "venv_job_id": venv_job_id, "collected_utc": utc_now(), **(extra or {})}
    try:
        recs = verify_records(plan, records, sacct, array_job_id)
    except Incomplete as e:
        verdict = {**base, "status": "INCOMPLETE", "refusal": f"battery incomplete: {e.why}", "refusal_kind": e.kind}
        write_json_atomic(prefix.with_name(prefix.name + ".verdict.json"), verdict)
        return verdict
    counts = {k: sum(int((r.get("counts") or {}).get(k) or 0) for r in recs) for k in ("tests", "failures", "errors", "skipped")}
    failures = sorted((f for r in recs for f in (r.get("failures") or [])), key=lambda f: str(f.get("node")))
    observed: dict = {}
    for i in range(plan["count"]):
        observed.update(junit_file_seconds(records / f"t{i}" / "junit.xml"))
    tasks = [{"index": i, "status": r.get("status"), "pytest_rc": r.get("pytest_rc"), "elapsed_s": r.get("elapsed_s"),
              "n_files": len(plan["tasks"][i]["files"]), "predicted_s": plan["tasks"][i].get("predicted_s"),
              "observed_test_s": round(sum(observed.get(f, 0.0) for f in plan["tasks"][i]["files"]), 3),
              "counts": r.get("counts"), "node": r.get("node"), "slurm_job_id": r.get("slurm_job_id"),
              "slurm_array_job_id": r.get("slurm_array_job_id"), "slurm_array_task_id": r.get("slurm_array_task_id"),
              "results_dir": r.get("results_dir"), "venv": r.get("venv"), "finished_utc": r.get("finished_utc")} for i, r in enumerate(recs)]
    status = "FAIL" if any(r.get("status") == "FAIL" for r in recs) else "PASS"
    verdict = {**base, "status": status, "counts": counts, "failures": failures, "task_records": tasks,
               "observed_durations": observed, "elapsed_s_max": max(float(t["elapsed_s"] or 0) for t in tasks),
               "tree": recs[0].get("tree"), "dependency_hashes": recs[0].get("dependency_hashes"), "versions": recs[0].get("versions"),
               "env_overrides": recs[0].get("env_overrides")}
    merge_junit([records / f"t{i}" / "junit.xml" for i in range(plan["count"])], prefix.with_name(prefix.name + ".junit.xml"))
    log_p = prefix.with_name(prefix.name + ".log")
    with log_p.open("w") as fh:
        for i, t in enumerate(tasks):
            fh.write(f"==== task t{i} status={t['status']} rc={t['pytest_rc']} elapsed_s={t['elapsed_s']} node={t['node']} "
                     f"files={' '.join(plan['tasks'][i]['files'])}\n")
            lp = records / f"t{i}" / "pytest.log"
            fh.write(lp.read_text(errors="replace") if lp.is_file() else "(no pytest.log)\n")
            fh.write("\n")
    write_json_atomic(prefix.with_name(prefix.name + ".verdict.json"), verdict)
    return verdict


def update_observed(path: Path, observed: dict, *, sha: str | None, label: str | None) -> None:
    """Merge a verdict's per-file observed seconds into a duration table (latest wins), for the next plan."""
    cur: dict = {}
    if path.is_file():
        try:
            cur = json.loads(path.read_text())
        except ValueError:
            cur = {}
    files = dict(cur.get("files") or {})
    files.update(observed)
    hist = list(cur.get("history") or [])[-199:] + [{"sha": sha, "label": label, "n_files": len(observed), "utc": utc_now()}]
    write_json_atomic(path, {"schema": DURATIONS_SCHEMA, "files": files, "n_files": len(files), "history": hist, "rendered_utc": utc_now()})


def cmd_collect(a) -> int:
    plan = json.loads(Path(a.plan).read_text())
    sacct = None
    if a.sacct and Path(a.sacct).is_file():
        try:
            sacct = json.loads(Path(a.sacct).read_text())
        except ValueError:
            sacct = None
    prefix = Path(a.out_prefix)
    v = collect(plan, Path(a.records), prefix, sacct=sacct, array_job_id=a.array_job_id, venv_job_id=a.venv_job_id)
    if v["status"] == "INCOMPLETE":
        print(f"{v['refusal']} [kind={v['refusal_kind']}] label={v['label']} sha={str(v['sha'])[:12]}", flush=True)
        return 4
    if a.observed:
        update_observed(Path(a.observed), v["observed_durations"], sha=v["sha"], label=v["label"])
    c = v["counts"]
    print(f"battery {v['status']} label={v['label']} sha={str(v['sha'])[:12]} tasks={v['tasks']} files={len(v['requested_files'])} "
          f"tests={c['tests']} failures={c['failures']} errors={c['errors']} skipped={c['skipped']} elapsed_s_max={v['elapsed_s_max']}", flush=True)
    for t in v["task_records"]:
        print(f"  t{t['index']}: {t['status']} rc={t['pytest_rc']} elapsed_s={t['elapsed_s']} test_s={t['observed_test_s']} "
              f"(predicted {t['predicted_s']}) {t['n_files']} files node={t['node']}", flush=True)
    print(f"battery FAILURE NODES ({len(v['failures'])}):", flush=True)
    for f in v["failures"]:
        print(f"  {f.get('node')}  [{f.get('kind')}] {str(f.get('message') or '')[:160]}", flush=True)
    print(f"battery wrote {prefix}.verdict.json {prefix}.junit.xml {prefix}.log", flush=True)
    return 0 if v["status"] == "PASS" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("durations")
    d.add_argument("--shards-dir", default=None, help="a finished gate's <results>/<sha>/shards/ (every */junit.xml is read)")
    d.add_argument("--junit", nargs="*", default=None, help="JUnit reports to read in addition")
    d.add_argument("--out", default=None, help="write the table here (default: print it)")
    d.set_defaults(fn=cmd_durations)
    p = sub.add_parser("plan")
    p.add_argument("--sha", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--branch", default=None)
    p.add_argument("--files", nargs="+", required=True, help="tree-relative test files")
    p.add_argument("--durations", nargs="*", default=None, help="duration tables, later ones override (baseline first, observed last)")
    p.add_argument("--tasks", type=int, default=None, help="force K (capped at the number of files)")
    p.add_argument("--budget-s", type=int, default=BUDGET_S_DEFAULT)
    p.add_argument("--target-s", type=int, default=TARGET_S_DEFAULT)
    p.add_argument("--task-flags", default=None, help="gate_task flags every task gets (default: --local-venv --hide-slurm; '' for none)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_plan)
    c = sub.add_parser("collect")
    c.add_argument("--plan", required=True)
    c.add_argument("--records", required=True, help="the label directory: <results>/<sha>/reruns/<label>/ (or a copy of it)")
    c.add_argument("--out-prefix", required=True, help="writes <prefix>.verdict.json, <prefix>.junit.xml, <prefix>.log")
    c.add_argument("--sacct", default=None, help="JSON {task index: slurm state} for the missing-record classification")
    c.add_argument("--array-job-id", default=None, help="the array the submission recorded: every record's slurm_array_job_id must be it (else identity)")
    c.add_argument("--venv-job-id", default=None)
    c.add_argument("--observed", default=None, help="a duration table to update with this battery's per-file seconds")
    c.set_defaults(fn=cmd_collect)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
