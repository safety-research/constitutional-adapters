#!/usr/bin/env python3
"""The scope of an EXPLICIT RUN LIST for the OR-Bench hard / XSTest C/P/I re-judge: every pinned prompt of the run's task x the judged
rung's generation, for runs that carry no Sonnet 5 judgment (whose ``judge_items.jsonl`` :mod:`scope_cpi` reads to know the items).

    HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts python scripts/rejudge/terra/scope_cpi_runs.py --runs RUNS.tsv --out SCOPE \
        [--registry ~/registry/registry.sqlite | --no-registry] [--path-rewrite FROM=TO ...] [--scope-note TEXT] [--allow-follow-up]

RUNS.tsv (tab-separated, a header row): ``run_id``, ``eval_id`` (or_bench_hard | xstest), ``cell`` (the judged rung's cell directory:
``private/gens_<task>.jsonl`` under it) and ``cap`` (that rung's ``max_new_tokens``) are required; ``campaign``, ``trial`` (default 0)
are optional. The rung is chosen by whoever writes the list (the lane's rule: XSTest at its 2,048-token rung, OR-Bench at its final
rung), and the cell must say it is that rung: its spool header's ``gen_cfg.max_new_tokens`` must equal ``cap`` and its ``task`` the
eval's task.

WHICH ITEMS. Every pinned row of the task (``judged.or_bench_rows`` 1,319; ``judged.xstest_rows`` 450 = 250 safe + 200 contrast, BOTH
halves) over ``corpora.load_pins()``, offline from the HF cache. Every pinned item must have a generation in the spool (first write wins;
a duplicate with different text refuses, as ``GenSpool`` does; the spool is read only) and the spool must hold no item outside the pinned
set; either refuses the run. The request key is the Sonnet judge's ``JudgeRequest(...).key()`` for (rubric, prompt, response)
(:func:`rejudge.terra.cpi.request_key`), so a unit's key is the key the batch track's bodies carry.

THE REGISTRY (read only, short indexed reads): every listed run must exist on the listed eval (its campaign, model and object are taken
from it); a complete Sonnet 5 C/P/I judgment, if one exists, is recorded as the judgment re-measured (more than one refuses); a run that
already carries a MiniMax-M3 judgment is refused unless ``--allow-follow-up``. ``--no-registry`` trusts the list (``campaign`` required).

THE CELL IS THE RUN'S (registry mode). The run's ``store_uri`` in the registry is its cell at the rung its generation closed on:
``<root>/controls_cap<C>/<target>/cells/<object>__controls__<key>__<rung hash>``, where ``<key>`` is ``t<trial>__a<n>__<8 hex>`` and
``<rung hash>`` is 8 hex. The judged rung may be ANOTHER rung of the same ladder, and along one run's ladder only the rung directory
and the rung hash vary. So those two are free, and the listed cell must sit under the same ``<root>``, the same ``<target>``
directory, and name the same ``<object>`` and the same ``<key>`` (after ``--path-rewrite FROM=TO``, which maps a store path -- the
list's and the registry's -- to the mirror read here). The cell's name must match that grammar IN FULL: a moved-aside sibling
(``<cell>.failed_<job>``) is not a cell. A valid cell of another run (another target, another object, another root, another trial)
is refused, and so are a run the registry records no store for and a path that is not a controls cell.

OUTPUT (``--out``), the shapes of :mod:`scope_cpi` (the build and the filing read them unchanged):
  units.tsv.gz   run_id, item_id, trial, sample (0), turn (-1), custom_id, rubric, eval_id, orig_label (''), orig_status (''), note
                 {cell: 0, subset (xstest)}
  texts.jsonl.gz one row per distinct custom_id: {k, rubric, item_id, prompt, response}
  runs.jsonl     one row per run: ``sonnet_*`` null unless the registry holds a Sonnet judgment; ``sources`` [{trial, cell, cap,
                 spool, spool_sha256, origin}], ``judged_store`` (the cell as read here), ``judged_store_origin`` (the same cell where
                 it was generated: under the registry's root, so a cluster path when the cell was read through a mirror; without the
                 registry, the cell as the list named it), ``judged_spool_sha256``, ``judged_cap``, ``scope`` (the judgments' scope
                 text)
  report.json    totals, per-eval counts, every problem by name
Nothing but ``report.json`` is written when any run is refused (exit 2); a directory that already holds a scope is refused.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import re
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common.runlist import (DEFAULT_REGISTRY, REJUDGE_MODEL, Refused, Rules, connect_ro, parse_rules, read_tsv,  # noqa: E402
                                    recorded_store, rewrite)
from rejudge.terra import cpi as C  # noqa: E402
from rejudge.terra.scope_cpi import read_spool, sha256_file  # noqa: E402

REQUIRED_COLUMNS = ("run_id", "eval_id", "cell", "cap")
OPTIONAL_COLUMNS = ("campaign", "trial")
SOURCES_HOW = "explicit run list: the judged rung's cell"
#: a controls cell: ``<root>/controls_cap<rung>/<target>/cells/<object>__controls__<key>__<rung hash>``, ``<key>`` =
#: ``t<trial>__a<n>__<8 hex>``. Along one run's ladder the rung directory and the rung hash vary; the root, the target directory, the
#: object and the key do not. The name is matched IN FULL (``\Z``), so a suffixed sibling (``<cell>.failed_<job>``) is no cell
CELL_RE = re.compile(r"^(?P<root>.+)/controls_cap(?P<rung>\d+)/(?P<target>[^/]+)/cells/(?P<object>[^/]+?)__controls__"
                     r"(?P<key>t\d+__a\d+__[0-9a-f]{8})__(?P<rung_hash>[0-9a-f]{8})\Z")
CELL_SHAPE = "<root>/controls_cap<C>/<target>/cells/<object>__controls__t<T>__a<N>__<8 hex>__<8 hex>"
#: the parts of a cell's path that must be the registry's, as a refusal names them
BOUND_PARTS = {"root": "root", "target": "target", "object": "object", "key": "trial key"}


def read_runs_tsv(path: str, rules: Rules = ()) -> Dict[str, Dict[str, Any]]:
    """run_id -> its row: ``cell`` the path read here (the listed path made absolute, then ``rules``), ``cell_listed`` the path as the
    list named it."""
    out: Dict[str, Dict[str, Any]] = {}
    for i, row in read_tsv(path, REQUIRED_COLUMNS, OPTIONAL_COLUMNS):
        if any(row.get(k) is None for k in REQUIRED_COLUMNS):
            raise Refused(f"{path}:{i}: {list(REQUIRED_COLUMNS)} are required")
        if row["run_id"] in out:
            raise Refused(f"{path}:{i}: run {row['run_id']} is listed twice (one judged cell per run)")
        if row["eval_id"] not in C.RUBRIC_OF_EVAL:
            raise Refused(f"{path}:{i}: eval {row['eval_id']!r} is not one of {sorted(C.RUBRIC_OF_EVAL)}")
        try:
            row["cap"] = int(row["cap"])
            row["trial"] = int(row.get("trial") or 0)
        except ValueError:
            raise Refused(f"{path}:{i}: cap and trial are integers")
        row["cell_listed"] = os.path.abspath(os.path.expanduser(row["cell"]))
        row["cell"] = rewrite(row["cell_listed"], rules)
        out[row["run_id"]] = row
    return out


def pinned_rows() -> Dict[str, Dict[str, dict]]:
    """eval -> {item_id: pinned row}: the frozen prompts (offline from the HF cache)."""
    from sisteer.capeval import corpora
    from sisteer.capeval.tiers import judged
    pins = corpora.load_pins()
    return {"or_bench_hard": {r["item_id"]: r for r in judged.or_bench_rows(pins)},
            "xstest": {r["item_id"]: r for r in judged.xstest_rows(pins)}}


def spool_header(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        first = fh.readline()
    try:
        row = json.loads(first)
    except ValueError:
        return {}
    return dict(row.get("__spool_header__") or {}) if isinstance(row, dict) else {}


def registry_facts(db: str, run_ids: List[str]) -> Tuple[Dict[str, dict], Dict[str, List[dict]], Dict[str, str]]:
    """``(runs, judgments by run with their judge model, eval -> the Sonnet 5 C/P/I judge id)``, read only; a run row carries its
    ``store_uri``."""
    con = connect_ro(db)
    runs: Dict[str, dict] = {}
    judg: Dict[str, List[dict]] = collections.defaultdict(list)
    try:
        sonnet: Dict[str, str] = {}
        for rubric, ev in C.EVAL_OF_RUBRIC.items():
            rows = con.execute("SELECT judge_id FROM judge WHERE rubric_name = ? AND judge_model = ?", (rubric, C.ORIGINAL_MODEL)).fetchall()
            if len(rows) == 1:
                sonnet[ev] = rows[0][0]
        ids = sorted(set(run_ids))
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            marks = ",".join("?" * len(part))
            for r in con.execute(f"SELECT run_id, campaign, eval_id, model_id, object_id, lane_source_legacy, store_uri FROM run "
                                 f"WHERE run_id IN ({marks})", part):
                runs[r["run_id"]] = dict(r)
            for j in con.execute(f"SELECT g.judgment_id, g.run_id, g.judge_id, g.status, g.n_items_judged, g.n_judge_refusals, g.finished_utc, "
                                 f"j.judge_model FROM judgment g JOIN judge j ON j.judge_id = g.judge_id WHERE g.run_id IN ({marks})", part):
                judg[j["run_id"]].append(dict(j))
    finally:
        con.close()
    return runs, judg, sonnet


def cell_parts(path: str) -> Optional[Dict[str, str]]:
    """``{root, rung, target, object, key, rung_hash}`` of a controls cell path (:data:`CELL_RE`, the name matched in full), else None.
    String work only."""
    m = CELL_RE.match(os.path.normpath(path))
    return m.groupdict() if m else None


def bound_cell(cell: str, store_uri: Optional[str], rules: Rules = ()) -> str:
    """The judged ``cell`` (as read here) held to the run's ``store_uri`` (module docstring, THE CELL IS THE RUN'S); returns the cell's
    ORIGIN, the same rung's cell under the registry's own root. :class:`Refused` when the registry records no store, either path is not
    a controls cell (its name matched in full), or the root, the target directory, the object or the trial key differ."""
    if not store_uri:
        raise Refused("the registry records no store for it (run.store_uri is empty): its generation has not closed, so nothing says "
                      "which cell is its own")
    reg = cell_parts(recorded_store(store_uri) or "")
    want = cell_parts(recorded_store(store_uri, rules) or "")
    got = cell_parts(cell)
    if reg is None or want is None:
        raise Refused(f"the registry's store {store_uri} is not a controls cell ({CELL_SHAPE})"
                      + ("" if reg is None else " once --path-rewrite is applied"))
    if got is None:
        raise Refused(f"cell {cell} is not a controls cell ({CELL_SHAPE}; the name must match in full, so a moved-aside copy such as "
                      f"<cell>.failed_<job> is not one)")
    same = {"root": os.path.realpath(got["root"]) == os.path.realpath(want["root"]),      # one directory under two spellings is one root
            "target": got["target"] == want["target"], "object": got["object"] == want["object"], "key": got["key"] == want["key"]}
    differ = [f"{BOUND_PARTS[k]} {got[k]!r} (the registry's: {want[k]!r})" for k, ok in same.items() if not ok]
    if differ:
        raise Refused(f"cell {cell} is not a rung of the run's own store {store_uri}: its {'; its '.join(differ)}"
                      + ("" if same["root"] else ". A mirror of the store needs --path-rewrite FROM=TO"))
    return f"{reg['root']}/controls_cap{got['rung']}/{got['target']}/cells/{got['object']}__controls__{got['key']}__{got['rung_hash']}"


def scope_run(row: Mapping[str, Any], pinned: Mapping[str, Mapping[str, dict]], texts: Dict[str, dict]) -> Tuple[List[List[str]], Dict[str, Any]]:
    """One run's unit rows and its source record; :class:`Refused` naming the first problem."""
    ev, cell, cap, trial = row["eval_id"], row["cell"], row["cap"], row["trial"]
    task = C.TASK_OF_EVAL[ev]
    rubric = C.RUBRIC_OF_EVAL[ev]
    spool = os.path.join(cell, "private", f"gens_{task.split('/', 1)[1]}.jsonl")
    if not os.path.isfile(spool):
        raise Refused(f"cell {cell} has no spool {os.path.basename(spool)}")
    hdr = spool_header(spool)
    have_cap = (hdr.get("gen_cfg") or {}).get("max_new_tokens")
    if hdr.get("task") != task:
        raise Refused(f"{spool}: the spool header's task is {hdr.get('task')!r}, not {task!r}")
    if have_cap is None or int(have_cap) != int(cap):
        raise Refused(f"{spool}: the spool header's max_new_tokens is {have_cap!r}, the list says the judged rung is {cap}")
    try:
        gens = read_spool(spool)
    except ValueError as e:
        raise Refused(str(e))
    pins = pinned[ev]
    extra = sorted(set(gens) - set(pins))
    if extra:
        raise Refused(f"{spool}: {len(extra)} items are not pinned {ev} rows (e.g. {extra[:3]})")
    missing = sorted(set(pins) - set(gens))
    if missing:
        raise Refused(f"{spool}: {len(missing)} pinned {ev} items have no generation (e.g. {missing[:3]})")
    rows: List[List[str]] = []
    cnt: collections.Counter = collections.Counter()
    for iid, pr in pins.items():
        resp = gens[iid]
        key = C.request_key(rubric, pr["prompt"], resp)
        if key not in texts:
            texts[key] = {"k": key, "rubric": rubric, "item_id": iid, "prompt": pr["prompt"], "response": resp}
        note: Dict[str, Any] = {"cell": 0}
        if ev == "xstest":
            note["subset"] = pr.get("subset")
            cnt["subset_" + str(pr.get("subset"))] += 1
        rows.append([row["run_id"], iid, str(trial), "0", "-1", key, rubric, ev, "", "", json.dumps(note, separators=(",", ":"))])
        cnt["units"] += 1
    src = {"trial": trial, "cell": cell, "cap": int(cap), "spool": spool, "spool_sha256": sha256_file(spool),
           "origin": row.get("origin") or row.get("cell_listed") or cell}
    return rows, {"counts": dict(cnt), "source": src}


def main(argv: Optional[List[str]] = None, pinned_fn: Callable[[], Dict[str, Dict[str, dict]]] = pinned_rows) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--no-registry", action="store_true")
    ap.add_argument("--path-rewrite", action="append", default=[], help="FROM=TO prefix rewrite of a store path (a mirror): of the list's "
                    "cells and of the store the registry records")
    ap.add_argument("--scope-note", help="the judgments' scope text (default: explicit run list (campaign <campaign>))")
    ap.add_argument("--allow-follow-up", action="store_true", help="admit runs that already carry a MiniMax-M3 judgment")
    args = ap.parse_args(argv)
    rules = parse_rules(args.path_rewrite)
    if any(os.path.exists(os.path.join(args.out, f)) for f in ("units.tsv.gz", "runs.jsonl", "texts.jsonl.gz")):
        raise SystemExit(f"REFUSED: {args.out} already holds a scope: a scope is built once, into a fresh directory")
    os.makedirs(args.out, exist_ok=True)
    os.chmod(args.out, 0o700)
    try:
        runs = read_runs_tsv(args.runs, rules)
    except Refused as e:
        raise SystemExit(f"REFUSED: {e}")
    registry = None if args.no_registry else args.registry
    if registry and not os.path.exists(registry):
        raise SystemExit(f"REFUSED: no registry at {registry} (pass --no-registry to trust the list)")
    reg: Dict[str, dict] = {}
    judg: Dict[str, List[dict]] = {}
    sonnet: Dict[str, str] = {}
    if registry:
        reg, judg, sonnet = registry_facts(registry, list(runs))
    pinned = pinned_fn()
    problems: List[Dict[str, str]] = []
    texts: Dict[str, dict] = {}
    unit_rows: List[List[str]] = []
    recs: List[Dict[str, Any]] = []
    by_eval: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for run_id in sorted(runs):
        row = dict(runs[run_id])
        ev = row["eval_id"]
        sj = None
        meta = {"campaign": row.get("campaign"), "model_id": None, "object_id": None, "lane_source_legacy": None}
        if registry:
            r = reg.get(run_id)
            if r is None:
                problems.append({"run_id": run_id, "problem": "not a run in the registry"})
                continue
            if r["eval_id"] != ev:
                problems.append({"run_id": run_id, "problem": f"the registry's eval is {r['eval_id']!r}, the list says {ev!r}"})
                continue
            if row.get("campaign") is not None and row["campaign"] != r["campaign"]:
                problems.append({"run_id": run_id, "problem": f"the registry's campaign is {r['campaign']!r}, the list says {row['campaign']!r}"})
                continue
            meta = {k: r[k] for k in meta}
            mm = [j for j in judg.get(run_id, []) if j["judge_model"] == REJUDGE_MODEL]
            if mm and not args.allow_follow_up:
                problems.append({"run_id": run_id, "problem": f"already judged by {REJUDGE_MODEL} ({[(j['judgment_id'], j['status']) for j in mm]}); "
                                                              f"a second filing would be a follow-up (--allow-follow-up)"})
                continue
            ss = [j for j in judg.get(run_id, []) if j["judge_id"] == sonnet.get(ev) and j["status"] == "complete"]
            if len(ss) > 1:
                problems.append({"run_id": run_id, "problem": f"{len(ss)} complete Sonnet judgments"})
                continue
            sj = ss[0] if ss else None
            try:
                row["origin"] = bound_cell(row["cell"], r.get("store_uri"), rules)
            except Refused as e:
                problems.append({"run_id": run_id, "problem": str(e)})
                continue
        elif not row.get("campaign"):
            problems.append({"run_id": run_id, "problem": "--no-registry needs the list's campaign column"})
            continue
        try:
            urows, info = scope_run(row, pinned, texts)
        except Refused as e:
            problems.append({"run_id": run_id, "problem": str(e)})
            continue
        unit_rows.extend(urows)
        src = info["source"]
        recs.append({"run_id": run_id, "eval_id": ev, "campaign": meta["campaign"], "lane": meta["lane_source_legacy"],
                     "rubric": C.RUBRIC_OF_EVAL[ev], "sonnet_judge_id": sj["judge_id"] if sj else None,
                     "sonnet_judgment_id": sj["judgment_id"] if sj else None, "sonnet_n_items_judged": sj["n_items_judged"] if sj else None,
                     "sonnet_n_judge_refusals": sj["n_judge_refusals"] if sj else None, "sources_how": SOURCES_HOW, "sources": [src],
                     "judged_store": src["cell"], "judged_store_origin": src["origin"], "judged_spool_sha256": src["spool_sha256"],
                     "judged_cap": src["cap"], "store_bound_by": "run" if registry else "list",
                     "scope": args.scope_note or f"explicit run list (campaign {meta['campaign']})", "counts": info["counts"],
                     "model_id": meta["model_id"], "object_id": meta["object_id"], "selection_state": None})
        be = by_eval[ev]
        be["runs"] += 1
        be["units"] += info["counts"]["units"]
    distinct = collections.Counter(t["rubric"] for t in texts.values())
    report = {"runs_listed": len(runs), "runs": len(recs), "units": len(unit_rows), "distinct_requests": len(texts),
              "distinct_by_rubric": dict(distinct), "xstest_subset": "all", "cells_read": len(recs), "registry": registry,
              # what vouches for each run's cell: run.store_uri, or (--no-registry) the list alone
              "stores_bound_by": dict(collections.Counter(r["store_bound_by"] for r in recs)),
              "by_eval": {k: dict(v) for k, v in sorted(by_eval.items())}, "n_problems": len(problems), "problems": problems,
              "rubric_sha16": {ev: C.RUBRICS[C.RUBRIC_OF_EVAL[ev]].sha for ev in C.RUBRIC_OF_EVAL},
              "protocol_sha16": {ev: C.judging.protocol_sha(C.RUBRICS[C.RUBRIC_OF_EVAL[ev]]) for ev in C.RUBRIC_OF_EVAL},
              "sonnet_judges": sonnet}
    if problems:
        with open(os.path.join(args.out, "report.json"), "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"REFUSED: {len(problems)} problem(s); nothing written but report.json")
        for p in problems[:30]:
            print("  PROBLEM", p["run_id"], p["problem"])
        return 2
    with gzip.open(os.path.join(args.out, "units.tsv.gz"), "wt") as fh:
        fh.write("run_id\titem_id\ttrial\tsample\tturn\tcustom_id\trubric\teval_id\torig_label\torig_status\tnote\n")
        for r in unit_rows:
            fh.write("\t".join(r) + "\n")
    with gzip.open(os.path.join(args.out, "texts.jsonl.gz"), "wt") as fh:
        for k in sorted(texts):
            fh.write(json.dumps(texts[k], ensure_ascii=True) + "\n")
    with open(os.path.join(args.out, "runs.jsonl"), "w") as fh:
        for rec in recs:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "problems"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
