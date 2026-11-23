#!/usr/bin/env python3
"""The scope of an EXPLICIT RUN LIST for the StrongREJECT re-judge: one JOB per listed run, its units read from the run's own transcript
stores, for runs that carry no Claude judgment (or whose Claude judgment is not the selection rule). Stdlib only, like
:mod:`enumerate_units`, so it runs where the stores are (a cluster node with a bare ``python3``) as well as on the grading host.

    python3 scope_runs.py --runs RUNS.tsv --goals GOALS.json --out SCOPE [--registry DB | --no-registry] [--site NAME=PREFIX ...]
        [--scope-note TEXT] [--path-rewrite FROM=TO ...] [--allow-follow-up]
    python3 scope_runs.py --verify-jobs SCOPE/jobs.jsonl [--registry DB] [--path-rewrite FROM=TO ...] [--allow-follow-up]
        # the registry half only (no store is read)

RUNS.tsv (tab-separated, a header row): ``run_id``, ``eval_id`` and ``store`` are required; ``campaign``, ``kind``, ``haiku_judge_id``,
``haiku_judgment_id``, ``haiku_n_items_judged``, ``model_id``, ``object_id``, ``lane_source_legacy`` and ``pass_label`` are optional.
``store`` is the CELL directory of a static run (``shard*/responses.jsonl`` under it), of a multi-turn or PAIR run (the same layout, one
reply per row), or a best-of-N run's ``bon_responses.jsonl`` (or the directory holding it). Two tokens take the store from the registry
(registry mode only): ``@judgment`` the store the run's complete Haiku judgment graded (``judgment.store_uri``), so a re-judge of a
Haiku-judged run reads exactly the texts that judgment read; ``@run`` the store the run itself records (``run.store_uri``, filed when
the run's generation closed). A run listed on several rows has several stores (one unit each), which only ``--no-registry`` admits.

THE UNITS, from the store layout (the shapes :mod:`scope` builds from its maps):
  static  ``{type: static, path: <cell>}``; refused when the cell holds no ``shard*/responses.jsonl``
  mt      actorattack, crescendo and pair: ``{type: mt, path: <cell>, files: sorted <cell>/shard*/responses.jsonl, drop_goals}``, where
          ``drop_goals`` are the goals whose LATEST progress outcome (each file's sibling ``progress.jsonl``, read in the files' sorted
          order, a later row of a goal replacing an earlier one) starts ``error`` or ``refused`` -- the rule the paper-row maps were built with
          (``jailbreak_artifacts.exclude_errored_goals``); refused when the cell holds no reply file
  bon     strongreject_bon: ``{type: bon, path: <bon_responses.jsonl>, files: [it]}``, the native single pass; refused when the store's
          directory also holds another ``bon_responses*.jsonl`` (a climb or a rung of a folded reading, which this scope does not build)
``fold`` is null; ``mapping_method`` is ``explicit_run_list``. A store's SITE is the first ``--site NAME=PREFIX`` its path starts with
(``local`` = the home directory, always declared); a store under no declared prefix is refused (it would never be enumerated).

THE RUBRIC (``kind``: ``sr`` = v1, ``sr2`` = v2). With the registry, the kind is the rubric of the judge the run's INSTRUMENT names
(``run.instrument_id`` -> ``instrument.judge_id``: the Haiku v1 or v2 judge), because a best-of-N instrument can carry either; a run
whose instrument names neither is refused. Without it, the list's ``kind`` column, else ``sr2`` for strongreject_bon and ``sr``
otherwise; ``--verify-jobs`` then holds each job's kind to its run's instrument.

THE REGISTRY (read only, short indexed reads; default ``~/registry/registry.sqlite``): every listed run must exist on the listed eval;
``campaign``, ``model_id``, ``object_id``, ``lane_source_legacy`` and ``pass_label`` are taken from it (a TSV value that disagrees
refuses, and so does a ``kind``); the run's complete Haiku judgment, if any, gives ``haiku_judge_id`` / ``haiku_judgment_id`` (more than
one refuses, and so does a Haiku rubric other than the run's kind).
THE STORE IS THE RUN'S. A listed store must be the ONE store the registry ties to the run, after ``--path-rewrite`` (a mirror): the
store its complete Haiku judgment graded when that judgment names one, else ``run.store_uri`` (a best-of-N store is compared by its
directory, so the file and the directory holding it are one store). A store of another run is refused however valid its layout, and so
is a run the registry records no store for (its generation has not closed, so nothing says which texts are its own).
A run that already carries a MiniMax-M3 judgment is refused unless ``--allow-follow-up`` (a run is filed once; a second filing would be
a follow-up that cannot remove the first one's rows). ``--no-registry`` trusts the TSV (``campaign`` required; Haiku ids from the TSV
or null): it is for the cluster side, and ``--verify-jobs`` then runs the registry half over the jobs it wrote, on the grading host --
the run fields, the Haiku ids, the kind, and every unit's path against the registry's store AS STRINGS (the grading host need not hold
the store; ``--path-rewrite`` applies to the registry's path). The enumeration reads a multi-turn or best-of-N unit by its ``files``,
not its ``path``, so every file must lie under the unit's path (best-of-N: be it), and a job carries no ``fold``. The filing runs the
same check before it stages explicit-run-list jobs under the judge rows on record (``file_records.py --expect-pinned-judges``), so a
scope that was never verified cannot be filed.

OUTPUT (``--out``): ``jobs.jsonl`` (one job per run, sorted by run id: ``run_id, campaign, eval_id, category, kind, haiku_judge_id,
haiku_judgment_id, haiku_n_items_judged, lane_source_legacy, pass_label, model_id, object_id, units, fold, sites, mapping_method,
scope``), ``goals.json`` (a byte copy of ``--goals``, or ``--goals`` itself when it is already ``<out>/goals.json``:
:mod:`build_manifests` reads ``jobs.jsonl`` beside it) and
``scope_summary.json``. Nothing is written when any run is refused (``scope_summary.json`` alone names the problems; exit 2), and a
directory that already holds a ``jobs.jsonl`` is refused (a scope is built once, into a fresh directory).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shutil
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common.runlist import (DEFAULT_REGISTRY, REJUDGE_MODEL, Refused, Rules, connect_ro, parse_rules, read_tsv,  # noqa: E402
                                    recorded_store, rewrite)
from rejudge.strongreject.scope import CATEGORY, HAIKU_V1, HAIKU_V2, KIND_OF_JUDGE, site_of  # noqa: E402

HOME = os.path.expanduser("~")
MAPPING_METHOD = "explicit_run_list"
REQUIRED_COLUMNS = ("run_id", "eval_id", "store")
OPTIONAL_COLUMNS = ("campaign", "kind", "haiku_judge_id", "haiku_judgment_id", "haiku_n_items_judged", "model_id", "object_id",
                    "lane_source_legacy", "pass_label")
#: the run fields the registry states (a TSV value must agree)
RUN_FIELDS = ("campaign", "model_id", "object_id", "lane_source_legacy", "pass_label")
#: the ``store`` tokens that take the store from the registry: the one the run's Haiku judgment graded, the one the run records
FROM_JUDGMENT = "@judgment"
FROM_RUN = "@run"
KINDS = ("sr", "sr2")
BON_FILE = "bon_responses.jsonl"


def kind_of_eval(eval_id: str) -> str:
    """The rubric an eval's NAME suggests: the default where no registry says which judge the run's instrument names."""
    return "sr2" if eval_id == "strongreject_bon" else "sr"


# --------------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------------
def read_runs_tsv(path: str) -> Dict[str, Dict[str, Any]]:
    """run_id -> ``{eval_id, stores: [...], <optional columns>}`` in file order; a run on several rows gathers their stores (the other
    columns must agree)."""
    out: Dict[str, Dict[str, Any]] = {}
    for i, row in read_tsv(path, REQUIRED_COLUMNS, OPTIONAL_COLUMNS):
        rid = row["run_id"]
        if not rid or not row["eval_id"]:
            raise Refused(f"{path}:{i}: run_id and eval_id are required")
        if row.get("kind") is not None and row["kind"] not in KINDS:
            raise Refused(f"{path}:{i}: kind {row['kind']!r} is not one of {list(KINDS)}")
        cur = out.setdefault(rid, {"eval_id": row["eval_id"], "stores": []})
        for k in ("eval_id",) + OPTIONAL_COLUMNS:
            v = row.get(k)
            if v is None:
                continue
            if cur.get(k) is not None and cur[k] != v:
                raise Refused(f"{path}:{i}: run {rid} lists {k} {cur[k]!r} and {v!r}")
            cur[k] = v
        if row["store"]:
            cur["stores"].append(row["store"])
    return out


def registry_facts(db: str, run_ids: Sequence[str]) -> Tuple[Dict[str, dict], Dict[str, List[dict]], Dict[str, List[dict]]]:
    """``(runs, haiku judgments, re-judge judgments)`` of these runs, read only (primary-key / index reads in chunks of 500, each its own
    short read). A run row carries its ``store_uri`` and the judge its instrument names (``instrument_judge_id``)."""
    con = connect_ro(db)
    runs: Dict[str, dict] = {}
    haiku: Dict[str, List[dict]] = collections.defaultdict(list)
    rejudged: Dict[str, List[dict]] = collections.defaultdict(list)
    try:
        ids = sorted(set(run_ids))
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            marks = ",".join("?" * len(part))
            for r in con.execute(f"SELECT r.run_id, r.campaign, r.eval_id, r.model_id, r.object_id, r.lane_source_legacy, r.pass_label, "
                                 f"r.trial_index, r.token_cap_tokens, r.store_uri, r.instrument_id, i.judge_id AS instrument_judge_id "
                                 f"FROM run r LEFT JOIN instrument i ON i.instrument_id = r.instrument_id WHERE r.run_id IN ({marks})", part):
                runs[r["run_id"]] = dict(r)
            for j in con.execute(f"SELECT g.judgment_id, g.run_id, g.judge_id, g.status, g.finished_utc, g.n_items_judged, g.store_uri, "
                                 f"j.judge_model FROM judgment g JOIN judge j ON j.judge_id = g.judge_id WHERE g.run_id IN ({marks})", part):
                d = dict(j)
                if d["judge_id"] in KIND_OF_JUDGE:
                    haiku[d["run_id"]].append(d)
                if d["judge_model"] == REJUDGE_MODEL:
                    rejudged[d["run_id"]].append(d)
    finally:
        con.close()
    return runs, haiku, rejudged


# --------------------------------------------------------------------------------------------
# units from the store layout
# --------------------------------------------------------------------------------------------
def _read_jsonl(path: str) -> Iterable[Any]:
    with open(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def drop_goals(files: Sequence[str]) -> List[str]:
    """Goals whose LATEST progress outcome over the files' sibling ``progress.jsonl`` (read in the files' sorted order) is an error or a
    refusal (the rule the paper-row maps were built with)."""
    progress: Dict[str, str] = {}
    for fp in sorted(files):
        pp = os.path.join(os.path.dirname(fp), "progress.jsonl")
        if not os.path.exists(pp):
            continue
        for r in _read_jsonl(pp):
            if isinstance(r, dict) and "goal_id" in r:
                progress[str(r["goal_id"])] = str(r.get("outcome", ""))
    return sorted(g for g, o in progress.items() if o.startswith("error") or o.startswith("refused"))


def shard_files(cell: str) -> List[str]:
    return sorted(glob.glob(os.path.join(cell, "shard*", "responses.jsonl")))


def unit_of(category: str, store: str) -> Dict[str, Any]:
    """One store -> its unit (module docstring, THE UNITS); :class:`Refused` for a store that does not hold that layout."""
    store = os.path.abspath(os.path.expanduser(store))
    if category == "static":
        if not os.path.isdir(store) or not shard_files(store):
            raise Refused(f"static store {store} holds no shard*/responses.jsonl")
        return {"type": "static", "path": store}
    if category == "mt":
        files = shard_files(store) if os.path.isdir(store) else []
        if not files:
            raise Refused(f"multi-turn store {store} holds no shard*/responses.jsonl")
        return {"type": "mt", "path": store, "files": files, "drop_goals": drop_goals(files)}
    path = os.path.join(store, BON_FILE) if os.path.isdir(store) else store
    if os.path.basename(path) != BON_FILE or not os.path.isfile(path):
        raise Refused(f"best-of-N store {store}: no {BON_FILE}")
    others = sorted(p for p in glob.glob(os.path.join(os.path.dirname(path), "bon_responses*.jsonl"))
                    if os.path.basename(p) != BON_FILE and "error" not in os.path.basename(p))
    if others:
        raise Refused(f"best-of-N store {path} sits beside {others}: a folded (store + climb) reading, which this scope does not build")
    return {"type": "bon", "path": path, "files": [path], "drop_goals": []}


# --------------------------------------------------------------------------------------------
# the store the registry ties to a run
# --------------------------------------------------------------------------------------------
def store_dir(category: str, path: str) -> str:
    """What two paths of one store are compared by: a best-of-N store's DIRECTORY (a list, a unit or the registry may name the reply
    file or the directory holding it), else the path itself."""
    return os.path.dirname(path) if category == "bon" and os.path.basename(path) == BON_FILE else path


def bound_store(run: Mapping[str, Any], hj: Optional[Mapping[str, Any]]) -> Tuple[Optional[str], str, str]:
    """``(store_uri, what names it, the list token that takes it)``: the store the run's complete Haiku judgment graded when that
    judgment names one, else the store the run records; the uri is None when the registry records neither."""
    if hj is not None and hj.get("store_uri"):
        return hj["store_uri"], f"the one its Haiku judgment {hj['judgment_id']} graded", FROM_JUDGMENT
    return run.get("store_uri"), "the one the registry records for the run (run.store_uri)", FROM_RUN


def no_store_problem(run_id: str) -> str:
    return (f"{run_id}: the registry records no store for it (no run.store_uri, and no Haiku judgment that names one): its generation "
            f"has not closed, so nothing says which texts are its own")


def unit_file_problems(job: Mapping[str, Any]) -> List[str]:
    """Why what the enumeration READS of a built job is not inside its units' paths (empty = it is), AS STRINGS. A static unit is read
    by its ``path`` alone; a multi-turn or best-of-N unit is read by its ``files`` (never by its ``path``), so every file must lie
    under the unit's path (best-of-N: BE the path, the reply file) and there must be one; a ``fold`` is read instead of the units, and
    an explicit-run-list job has none."""
    rid, cat = job["run_id"], str(job.get("category"))
    out: List[str] = []
    if job.get("fold"):
        out.append(f"{rid}: it carries a fold (stores read instead of its units), which an explicit-run-list job never has")
    for u in job.get("units") or []:
        path = os.path.normpath(str(u.get("path")))
        files = [os.path.normpath(str(f)) for f in u.get("files") or []]
        outside = [f for f in files if (f != path if cat == "bon" else not f.startswith(path + os.sep))]
        if outside:
            out.append(f"{rid}: its unit reads {len(outside)} file(s) that are not "
                       + ("its path" if cat == "bon" else "under its path") + f" {path} (e.g. {outside[:2]})")
        elif cat in ("mt", "bon") and not files:
            out.append(f"{rid}: its unit of path {path} names no file (a {cat} unit is read by its files)")
    return out


def unit_store_problems(job: Mapping[str, Any], run: Mapping[str, Any], hj: Optional[Mapping[str, Any]], rules: Rules) -> List[str]:
    """Why what a built job reads is not the run's store (empty = it is), compared AS STRINGS with the registry's ``store_uri`` after
    ``rules``: every unit's ``path`` is that store, and every file the enumeration reads lies inside it (:func:`unit_file_problems`).
    The check ``--verify-jobs`` (and the filing of such jobs) runs where the store itself may not be mounted."""
    uri, named, _token = bound_store(run, hj)
    want = recorded_store(uri, rules)
    if want is None:
        return [no_store_problem(job["run_id"])]
    cat = str(job.get("category"))
    paths = [str(u.get("path")) for u in job.get("units") or []]
    out: List[str] = []
    if [store_dir(cat, os.path.normpath(p)) for p in paths] != [store_dir(cat, want)]:
        out.append(f"{job['run_id']}: its unit path(s) {paths} are not {named} ({uri}"
                   + (f", read here as {want}" if want != uri else "") + ")")
    return out + unit_file_problems(job)


# --------------------------------------------------------------------------------------------
# the jobs
# --------------------------------------------------------------------------------------------
def _text(v: Any) -> Optional[str]:
    return None if v is None else str(v)


def registry_problems(run_id: str, want: Mapping[str, Any], reg: Mapping[str, dict], haiku: Mapping[str, List[dict]],
                      rejudged: Mapping[str, List[dict]], allow_follow_up: bool,
                      exact: bool = False) -> Tuple[List[str], Optional[dict], Optional[str]]:
    """Why the registry refuses this run (empty = it does not), its one complete Haiku judgment (or None) and its rubric kind (the
    judge its instrument names; None when that is neither Haiku StrongREJECT judge, which refuses). A run field or a kind the list
    leaves empty is not compared, unless ``exact`` (a built job must state every one as the registry does)."""
    out: List[str] = []
    r = reg.get(run_id)
    if r is None:
        return [f"{run_id}: not a run in the registry"], None, None
    if r["eval_id"] != want["eval_id"]:
        out.append(f"{run_id}: the registry's eval is {r['eval_id']!r}, the list says {want['eval_id']!r}")
    for k in RUN_FIELDS:
        if (want.get(k) is not None or exact) and _text(want.get(k)) != _text(r[k]):
            out.append(f"{run_id}: the registry's {k} is {r[k]!r}, the list says {want.get(k)!r}")
    kind = KIND_OF_JUDGE.get(r.get("instrument_judge_id"))
    if kind is None:
        out.append(f"{run_id}: its instrument {str(r.get('instrument_id'))[:12]} names judge {r.get('instrument_judge_id')!r}, not the "
                   f"Haiku v1 / v2 StrongREJECT judge ({HAIKU_V1[:12]}, {HAIKU_V2[:12]}): the run's rubric cannot be derived")
    elif (want.get("kind") is not None or exact) and want.get("kind") != kind:
        out.append(f"{run_id}: the run's instrument is rubric {kind}, the list says kind {want.get('kind')!r}")
    hs = [j for j in haiku.get(run_id, []) if j["status"] == "complete"]
    hj = None
    if len(hs) > 1:
        out.append(f"{run_id}: {len(hs)} complete Haiku judgments")
    elif hs:
        hj = hs[0]
        if kind is not None and KIND_OF_JUDGE[hj["judge_id"]] != kind:
            out.append(f"{run_id}: its Haiku judgment {hj['judgment_id']} is rubric {KIND_OF_JUDGE[hj['judge_id']]}, the run's "
                       f"instrument is rubric {kind}")
    for k, v in (("haiku_judge_id", hj and hj["judge_id"]), ("haiku_judgment_id", hj and hj["judgment_id"])):
        if want.get(k) is not None and want[k] != v:
            out.append(f"{run_id}: the list's {k} {want[k]!r} is not the registry's complete Haiku judgment ({v!r})")
    if rejudged.get(run_id) and not allow_follow_up:
        out.append(f"{run_id}: already judged by {REJUDGE_MODEL} ({[(j['judgment_id'], j['status']) for j in rejudged[run_id]]}); a second "
                   f"filing would be a follow-up (--allow-follow-up)")
    return out, hj, kind


def build_jobs(runs: Mapping[str, Dict[str, Any]], sites: Sequence[Tuple[str, str]], rules: Rules, registry: Optional[str],
               scope_note: Optional[str], allow_follow_up: bool) -> Tuple[List[dict], List[str], Dict[str, int]]:
    """The jobs, every problem (a run with a problem gets no job), and how many jobs' stores each source vouches for (``judgment``: the
    store the Haiku judgment graded; ``run``: ``run.store_uri``; ``list``: nothing but the list, without the registry)."""
    problems: List[str] = []
    reg: Dict[str, dict] = {}
    haiku: Dict[str, List[dict]] = {}
    rejudged: Dict[str, List[dict]] = {}
    if registry:
        reg, haiku, rejudged = registry_facts(registry, list(runs))
    jobs: List[dict] = []
    bound: collections.Counter = collections.Counter()
    for run_id in sorted(runs):
        want = runs[run_id]
        ev = want["eval_id"]
        if ev not in CATEGORY:
            problems.append(f"{run_id}: eval {ev!r} is not one of the StrongREJECT evals {sorted(CATEGORY)}")
            continue
        cat = CATEGORY[ev]
        hj = None
        fields = {k: want.get(k) for k in RUN_FIELDS}
        kind = want.get("kind") or kind_of_eval(ev)
        uri, named, token = None, "", ""
        if registry:
            probs, hj, reg_kind = registry_problems(run_id, want, reg, haiku, rejudged, allow_follow_up)
            if probs:
                problems.extend(probs)
                continue
            kind = str(reg_kind)
            fields = {k: reg[run_id][k] for k in RUN_FIELDS}
            uri, named, token = bound_store(reg[run_id], hj)
        elif not want.get("campaign"):
            problems.append(f"{run_id}: --no-registry needs the list's campaign column")
            continue
        # the stores the registry can stand in for (registry mode only)
        tokens = {FROM_JUDGMENT: hj["store_uri"] if hj is not None and hj.get("store_uri") else None,
                  FROM_RUN: reg[run_id].get("store_uri") if registry else None}
        needs = {FROM_JUDGMENT: "a complete Haiku judgment with a store_uri", FROM_RUN: "a store_uri on the run (its generation closed)"}
        missing = sorted({s for s in want["stores"] if s in tokens and not tokens[s]})
        if missing:
            problems.extend(f"{run_id}: store {s} needs the registry and {needs[s]}" for s in missing)
            continue
        stores = [recorded_store(tokens[s], rules) if s in tokens else rewrite(os.path.abspath(os.path.expanduser(s)), rules)
                  for s in want["stores"]]
        if not stores:
            problems.append(f"{run_id}: no store")
            continue
        if registry:
            want_store = recorded_store(uri, rules)
            if want_store is None:
                problems.append(no_store_problem(run_id))
                continue
            if [os.path.realpath(store_dir(cat, s)) for s in stores] != [os.path.realpath(store_dir(cat, want_store))]:
                problems.append(f"{run_id}: the list's store(s) {stores} are not {named} ({uri}); list {token}, or --path-rewrite a mirror")
                continue
        try:
            units = [unit_of(cat, s) for s in stores]
        except Refused as e:
            problems.append(f"{run_id}: {e}")
            continue
        if cat == "bon" and len(units) != 1:
            problems.append(f"{run_id}: a best-of-N run reads one store, the list names {len(units)}")
            continue
        site_names = sorted({site_of(u["path"], sites) for u in units})
        if "unknown" in site_names:
            problems.append(f"{run_id}: a store lies under no declared site ({[u['path'] for u in units]}; sites {[n for n, _ in sites]})")
            continue
        if len(site_names) != 1:
            problems.append(f"{run_id}: its stores lie on several sites {site_names}")
            continue
        n_items = want.get("haiku_n_items_judged")
        job = {"run_id": run_id, "campaign": fields["campaign"], "eval_id": ev, "category": cat, "kind": kind,
               "haiku_judge_id": hj["judge_id"] if hj else want.get("haiku_judge_id"),
               "haiku_judgment_id": hj["judgment_id"] if hj else want.get("haiku_judgment_id"),
               "haiku_n_items_judged": hj["n_items_judged"] if hj else (int(n_items) if n_items is not None else None),
               "lane_source_legacy": fields["lane_source_legacy"], "pass_label": fields["pass_label"], "model_id": fields["model_id"],
               "object_id": fields["object_id"], "units": units, "fold": None, "sites": site_names, "mapping_method": MAPPING_METHOD,
               "scope": scope_note or f"explicit run list (campaign {fields['campaign']})"}
        if (job["haiku_judge_id"] is None) != (job["haiku_judgment_id"] is None):
            problems.append(f"{run_id}: a Haiku judge id without its judgment id (or the reverse)")
            continue
        if job["haiku_judge_id"] is not None and job["haiku_judge_id"] not in KIND_OF_JUDGE:
            problems.append(f"{run_id}: haiku_judge_id {job['haiku_judge_id']} is not the Haiku v1 / v2 judge ({HAIKU_V1[:12]}, {HAIKU_V2[:12]})")
            continue
        jobs.append(job)
        bound[{FROM_JUDGMENT: "judgment", FROM_RUN: "run"}.get(token, "list")] += 1
    return jobs, problems, dict(bound)


def verify_jobs(jobs_path: str, registry: str, allow_follow_up: bool, rules: Rules = ()) -> List[str]:
    """:func:`verify_job_rows` over a ``jobs.jsonl`` (``--verify-jobs``)."""
    with open(jobs_path) as fh:
        return verify_job_rows([json.loads(l) for l in fh if l.strip()], registry, allow_follow_up, rules)


def verify_job_rows(jobs: Sequence[Mapping[str, Any]], registry: str, allow_follow_up: bool, rules: Rules = ()) -> List[str]:
    """The registry half over built jobs (no store is read; empty = they verify): every job's run exists on its eval with its campaign
    and run fields, its kind is its instrument's rubric, its Haiku ids are the registry's complete Haiku judgment (or null when there
    is none), what it reads is the store the registry ties to the run (:func:`unit_store_problems`: unit paths and files, as strings),
    and no MiniMax-M3 judgment exists unless ``allow_follow_up``. It is what ``--verify-jobs`` runs over a scope built with
    ``--no-registry``, and what the filing runs (follow-ups allowed) before it stages such jobs under the judge rows on record."""
    reg, haiku, rejudged = registry_facts(registry, [j["run_id"] for j in jobs])
    problems: List[str] = []
    for j in jobs:
        want = {k: j.get(k) for k in ("eval_id", "kind", "haiku_judge_id", "haiku_judgment_id") + RUN_FIELDS}
        probs, hj, _kind = registry_problems(j["run_id"], want, reg, haiku, rejudged, allow_follow_up, exact=True)
        problems.extend(probs)
        if hj is not None and j.get("haiku_judgment_id") is None:
            problems.append(f"{j['run_id']}: the registry holds a complete Haiku judgment {hj['judgment_id']} the job does not name")
        if j.get("mapping_method") != MAPPING_METHOD or j.get("category") != CATEGORY.get(j["eval_id"]):
            problems.append(f"{j['run_id']}: not an explicit-run-list job of category {CATEGORY.get(j['eval_id'])}")
        if j["run_id"] in reg:
            problems.extend(unit_store_problems(j, reg[j["run_id"]], hj, rules))
    return problems


def check_goals(path: str) -> None:
    with open(path) as fh:
        d = json.load(fh)
    if not isinstance(d.get("goals"), dict) or not isinstance(d.get("goal_sets"), dict) or set(d["goal_sets"]) < {"static", "mt", "bon"}:
        raise Refused(f"{path}: not a goals.json (goals + goal_sets static / mt / bon)")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", help="the run list (TSV, module docstring)")
    ap.add_argument("--goals", help="goals.json (goals: dataset, revision, text; goal_sets) -- copied beside jobs.jsonl")
    ap.add_argument("--out", help="the scope directory")
    ap.add_argument("--verify-jobs", help="instead of building: the registry checks over this jobs.jsonl (no store read)")
    ap.add_argument("--registry", default=DEFAULT_REGISTRY, help="read only (default ~/registry/registry.sqlite)")
    ap.add_argument("--no-registry", action="store_true", help="trust the list (the cluster side); --verify-jobs checks it later")
    ap.add_argument("--site", action="append", default=[], help="NAME=PREFIX: stores under PREFIX are read on site NAME")
    ap.add_argument("--path-rewrite", action="append", default=[], help="FROM=TO prefix rewrite of a store path (a mirror): of the list's "
                    "stores and of the store the registry records")
    ap.add_argument("--scope-note", help="the judgments' scope text (default: explicit run list (campaign <campaign>))")
    ap.add_argument("--allow-follow-up", action="store_true", help="admit runs that already carry a MiniMax-M3 judgment")
    args = ap.parse_args(argv)
    rules = parse_rules(args.path_rewrite)
    if args.verify_jobs:
        problems = verify_jobs(args.verify_jobs, args.registry, args.allow_follow_up, rules)
        print(json.dumps({"jobs": args.verify_jobs, "problems": problems}, indent=1))
        return 0 if not problems else 2
    if not (args.runs and args.goals and args.out):
        ap.error("--runs, --goals and --out are required (or --verify-jobs)")
    sites = [("local", HOME)] + parse_rules(args.site, "--site")
    registry = None if args.no_registry else args.registry
    if registry and not os.path.exists(registry):
        raise SystemExit(f"REFUSED: no registry at {registry} (pass --no-registry on a host without one, then --verify-jobs where it is)")
    if os.path.exists(os.path.join(args.out, "jobs.jsonl")):
        raise SystemExit(f"REFUSED: {args.out} already holds a jobs.jsonl: a scope is built once, into a fresh directory")
    try:
        runs = read_runs_tsv(args.runs)
        check_goals(args.goals)
    except Refused as e:
        raise SystemExit(f"REFUSED: {e}")
    jobs, problems, bound = build_jobs(runs, sites, rules, registry, args.scope_note, args.allow_follow_up)
    os.makedirs(args.out, exist_ok=True)
    os.chmod(args.out, 0o700)
    summary = {"runs_listed": len(runs), "jobs": len(jobs), "problems": problems, "registry": registry, "mapping_method": MAPPING_METHOD,
               "by_category": dict(collections.Counter(j["category"] for j in jobs)),
               "by_kind": dict(collections.Counter(j["kind"] for j in jobs)),
               "by_eval": dict(collections.Counter(j["eval_id"] for j in jobs)),
               "by_sites": dict(collections.Counter("+".join(j["sites"]) for j in jobs)),
               "with_haiku_judgment": sum(1 for j in jobs if j["haiku_judgment_id"]),
               # what vouches for each job's store: the Haiku judgment's store, run.store_uri, or (--no-registry) the list alone
               "stores_bound_by": bound,
               "dropped_goals": sum(len(u.get("drop_goals") or []) for j in jobs for u in j["units"])}

    def write_summary() -> None:
        with open(os.path.join(args.out, "scope_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1)

    if problems:
        write_summary()
        print(f"REFUSED: {len(problems)} problem(s); nothing written but scope_summary.json")
        for p in problems[:30]:
            print("  ", p)
        return 2
    dest = os.path.join(args.out, "goals.json")
    if not (os.path.exists(dest) and os.path.samefile(args.goals, dest)):   # goals.py may have written it there already
        shutil.copyfile(args.goals, dest)
    tmp = os.path.join(args.out, ".jobs.jsonl.tmp")
    with open(tmp, "w") as fh:
        for j in jobs:
            fh.write(json.dumps(j, sort_keys=True) + "\n")
    os.replace(tmp, os.path.join(args.out, "jobs.jsonl"))
    write_summary()
    print(json.dumps({k: v for k, v in summary.items() if k != "problems"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
