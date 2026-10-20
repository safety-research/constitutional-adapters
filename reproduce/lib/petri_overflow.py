#!/usr/bin/env python3
"""reproduce/lib/petri_overflow.py — the Petri CONTEXT-OVERFLOW RE-RUN PASS's preflight for 06_eval_waves.sh --pass overflow
(petri-completion, 2026-10-13).

A first-pass Petri drive that leaves legs INCOMPLETE on context-overflow ledgers exits 3 and the grading daemon records the attempt
`incomplete` with the drive's result (`verdict: overflow_pending`, a per-leg `legs.detail`). Finishing those legs needs a SECOND proxy at
the doubled window (`petri_proxy.sbatch <key> <view> <sha> overflow-rerun <cells>`) and a drive under F22_OVERFLOW_RERUN=1 — the pass 06
submits and the daemon drives. This module decides, from the authoritative records alone, WHAT that proxy must serve and declare:

  overflow_cells(state_dir, model, view, *, python, repo, models_table=None) -> dict | raises Refused(name, text)

The EVENT it keys on: the daemon's record `<state>/grade/petri_drive.<tag>.json` holds, for the first-pass unit's CURRENT bound job
(`stage.binding_state` over submissions.jsonl, `<key>/<view>/petri/proxy`), a settled attempt in state `incomplete` whose result file
reads `verdict == "overflow_pending"`. Refusals by name, each the one thing that is not yet so:
  petri_overflow_first_pass_unbound   the first-pass unit is unbound / mid re-run: submit the first pass first
  petri_overflow_first_pass_cluster_unknown
                                      the first-pass binding names no cluster (a torn or hand-written binding): the rerun proxy's
                                      cluster cannot be read; re-bind the first pass (06 --family petri --resubmit)
  petri_overflow_no_record            no daemon record, or no attempt for the bound job: the daemon has not driven this proxy
  petri_overflow_not_pending          the last attempt for that job is running (`claimed|started`), `completed` (nothing to re-run),
                                      `failed|partial|blocked` (re-drive the first pass: 06 --resubmit), or `incomplete` without a readable
                                      `overflow_pending` result
  petri_overflow_no_detail            the result carries no per-leg `legs.detail` (a drive older than this contract): re-drive on this tree
  petri_overflow_ledgers_moved        the LIVE reading (`petri_run.py legs-status` over the result's out root — the drive's own done-ness
                                      predicate) disagrees with the verdict: a named leg neither done nor still ledgered, or refused; an
                                      unnamed leg ledgered
  petri_overflow_complete             (a bound overflow unit) every named leg is done: nothing left, nothing to re-mint
  petri_overflow_nothing_ledgered     (no overflow unit yet) every named leg is already done without an overflow proxy
  petri_overflow_control_parent       a ledgered control leg whose `control_of` is not a frozen canonical id
  petri_overflow_untranslatable       a ledgered leg's harness cell id matches no cell of the view's config
  petri_overflow_declaration_short    a bound overflow proxy's declaration does not cover a still-ledgered leg (a hand-submitted proxy)
  petri_overflow_transition           the overflow unit is mid re-run (a transition token): the funnel reconciles it first — 06 --pass overflow
                                      runs `submit_twin.py reconcile-token` on the unit BEFORE this preflight (a real call; a dry run says so and
                                      composes nothing), so this refusal is reached only by a direct call or when the funnel left the token
                                      standing (an ambiguous launch a human decides). The preflight runs before any funnel call, so without
                                      06's step an interrupted `--pass overflow` submit refused here forever

THE LEGS: the first-pass result's `legs.detail` entries with `status == "incomplete_overflow"` are the NAMED set; `legs-status` must read
each as EITHER still ledgered (`done: false`, `refused: false`, `overflow_unresolved > 0`) OR done (completed since the verdict, by an
earlier overflow proxy's drive), and no leg outside the named set may carry an overflow entry. The STILL-LEDGERED subset is what
the pass drives (the drive's own `ledgered` selector reads the same predicate per leg at start).

THE CELLS (positional 5 of the sbatch; what the rerun proxy SERVES and DECLARES): for an UNBOUND overflow unit, derived from the
still-ledgered legs — a canonical leg (`control_of` null) -> the config cell whose harness translation (registry_adapter.harness_cell_id,
the import's own translation) equals the leg's `cell_id`; a CONTROL leg (`kind == "none"`, `control_of` set; the native
`none__lora0` twin has `control_of: none`) -> the config cell translating to its `control_of` parent, which is `none` — declaring `none`
is the ONE way the drive's enumeration keeps the controls twin (petri_run.restrict_to_requested) and the standup verification admits
it. The declaration is ordered as the config's explicit_cells. For a BOUND overflow unit (a `--resubmit` after a partial pass) the
declaration is the bound attempt's `spec.args[4]` VERBATIM (attempts.jsonl): a recomputation would drop the cells an earlier
overflow drive completed and register a different run set under the same producer key; carried verbatim, the re-minted proxy registers
the same run ids and the lifecycle is monotonic.

THE CLUSTER: the rerun proxy runs on the cluster the FIRST PASS is bound on (`first_cluster`, the binding's `cluster`) —
05 --clusters stages the object bank and builds the bank view per cluster, and the proxy's standup requires both (petri_proxy.sbatch
f22_bank_view); a row placed on `*` expanded to every configured label and the funnel's first label took it, so a first pass on a
non-primary cluster got its rerun proxy where nothing was staged. 06 pins the plan's row to `first_cluster` (stage.py plan
--overflow-cluster), mirrors and submits there alone, and refuses an explicit --clusters that excludes it. A bound overflow unit's own
cluster is reported as `overflow_cluster` (a pre-fix placement elsewhere is moved back by the funnel's cross-cluster replacement, 24d).

Read-only: nothing here writes under the state dir, the registry or the cluster. The CLI prints the decision as one JSON document on
stdout (exit 0) or `REFUSED: <name>: <text>` on stderr (exit 3).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DEFAULT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import stage as ST  # noqa: E402

FIRST_PASS_UNIT = "proxy"
OVERFLOW_UNIT = ST.PETRI_OVERFLOW_UNIT
LEG_STATUS_OVERFLOW = "incomplete_overflow"
PETRI_RUN_ENV = "SISTEER_PETRI_RUN"        # the test seam the funnel's resume proof uses too (submit_twin.PETRI_RUN_ENV): the legs-status script


class Refused(Exception):
    def __init__(self, name: str, text: str):
        super().__init__(f"{name}: {text}")
        self.name = name
        self.text = text


def _load(path: str) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def record_path(state_dir: str, tag: str) -> str:
    """grade_daemon.petri_drive_record_path's rule (spelled here so the lib stays stdlib-only)."""
    return os.path.join(state_dir, "grade", f"petri_drive.{tag}.json")


def tag_of(model: str, view: str) -> str:
    return model if view == "main" else f"{model}_{view}"


def _attempts_for_job(rec: Mapping[str, Any], job_id: str) -> List[dict]:
    return [a for a in (rec.get("attempts") or []) if isinstance(a, dict) and str((a.get("identity") or {}).get("job_id")) == str(job_id)]


def legs_status(python: str, repo: str, manifest: str, out_root: str) -> dict:
    """`scripts/petri_run.py legs-status --out-root OUT` under the site's interpreter (PYTHONPATH=<repo>/src) — the drive's own done-ness
    reading; Refused(petri_overflow_ledgers_moved) when it cannot run or refuses the record (nothing is provably ledgered then)."""
    script = os.environ.get(PETRI_RUN_ENV) or os.path.join(repo, "scripts", "petri_run.py")
    cmd = [python, script, "--manifest", manifest, "legs-status", "--out-root", out_root]
    env = dict(os.environ, PYTHONPATH=os.path.join(repo, "src") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    except (OSError, subprocess.SubprocessError) as exc:
        raise Refused("petri_overflow_ledgers_moved", f"legs-status could not run ({type(exc).__name__}: {exc}) — the interpreter must import the petri dependency group (08's contract)")
    if r.returncode != 0:
        tail = " | ".join((r.stderr or r.stdout or "").strip().splitlines()[-3:])[:600]
        raise Refused("petri_overflow_ledgers_moved", f"legs-status refused over {out_root} (rc {r.returncode}: {tail}) — the drive's own done-ness reading is unavailable; re-drive the first pass")
    try:
        doc = json.loads(r.stdout)
        list(doc["legs"])
    except (ValueError, KeyError, TypeError):
        raise Refused("petri_overflow_ledgers_moved", f"legs-status printed no JSON document ({(r.stdout or '')[:200]!r})")
    if doc.get("record_refused"):
        raise Refused("petri_overflow_ledgers_moved", f"the resumed drive would refuse every leg ({doc['record_refused']}); re-drive the first pass")
    return doc


def _harness_translation(cells: Sequence[str]) -> Dict[str, str]:
    """{harness cell id: config cell} for the view's config cells — the import's own translation (registry_adapter.harness_cell_id)."""
    src = os.path.join(REPO_DEFAULT, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer.capeval.petri import registry_adapter as ra
    out: Dict[str, str] = {}
    for c in cells:
        try:
            out[ra.harness_cell_id(c)] = c
        except Exception as exc:  # noqa: BLE001 — a config cell the harness cannot spell is said by name when a ledgered leg needs it
            out[f"__untranslatable__{c}"] = f"{c} ({type(exc).__name__}: {exc})"
    return out


def derive_cells(still_ledgered: Sequence[Mapping[str, Any]], frozen_canonical_ids: Sequence[str], config_cells: Sequence[str]) -> List[str]:
    """The config cells the rerun proxy must serve and declare for `still_ledgered` legs (legs-status entries): a canonical leg's own cell; a
    control leg's `control_of` parent (`none`); ordered as `config_cells`; refusals by name (the module docstring). Pure over its inputs."""
    harness_to_cfg = _harness_translation(config_cells)
    wanted_harness: List[str] = []
    for leg in still_ledgered:
        cid = str(leg.get("cell_id"))
        if leg.get("control_of"):
            parent = str(leg["control_of"])
            if parent not in frozen_canonical_ids:
                raise Refused("petri_overflow_control_parent", f"control leg {leg.get('idx')} {cid!r} names control_of {parent!r}, not a frozen canonical leg {list(frozen_canonical_ids)} — "
                                                              "the controls twin cannot be served without its parent; re-drive the first pass")
            wanted_harness.append(parent)
        else:
            wanted_harness.append(cid)
    cells: List[str] = []
    for h in wanted_harness:
        cfg = harness_to_cfg.get(h)
        if cfg is None:
            raise Refused("petri_overflow_untranslatable", f"ledgered leg {h!r} is the harness id of no cell of the view's config (explicit_cells {list(config_cells)}) — "
                                                          "the rerun proxy could not serve it; fix the roster / config, never route around it")
        if cfg not in cells:
            cells.append(cfg)
    return [c for c in config_cells if c in cells]


def bound_declaration(state_dir: str, binding: Mapping[str, Any]) -> List[str]:
    """The bound overflow attempt's positional 5 (`spec.args[4]`) from attempts.jsonl, VERBATIM — split on commas."""
    ledger = os.path.join(state_dir, "attempts.jsonl")
    try:
        with open(ledger, encoding="utf-8") as fh:
            for ln in fh:
                try:
                    e = json.loads(ln)
                except ValueError:
                    continue
                if e.get("event") == "attempt" and e.get("attempt_id") == binding.get("attempt_id"):
                    args = [str(x) for x in ((e.get("spec") or {}).get("args") or [])]
                    if len(args) >= 5 and args[4]:
                        return [c for c in args[4].split(",") if c]
                    raise Refused("petri_overflow_declaration_short", f"the bound overflow attempt {binding.get('attempt_id')} (job {binding.get('job_id')}) carries no cells positional "
                                                                       f"(spec.args {args}) — a hand-submitted or whole-baseline rerun proxy; nothing re-minted")
    except OSError:
        pass
    raise Refused("petri_overflow_declaration_short", f"no attempt record for the bound overflow attempt {binding.get('attempt_id')} in {ledger} — its declaration cannot be read")


def overflow_cells(state_dir: str, model: str, view: str, *, python: str, repo: str = REPO_DEFAULT, models_table: Optional[str] = None) -> dict:
    tag = tag_of(model, view)
    subs = os.path.join(state_dir, "submissions.jsonl")
    first_lwid = ST.logical_work_id(model, view, "petri", FIRST_PASS_UNIT)
    over_lwid = ST.logical_work_id(model, view, "petri", OVERFLOW_UNIT)
    b1 = ST.binding_state(subs, first_lwid)
    if b1["state"] not in ("submitted", "registered"):
        raise Refused("petri_overflow_first_pass_unbound", f"the first-pass unit {first_lwid} is {b1['state']} — submit the first pass first (06 --family petri); "
                                                           "the overflow pass runs only after its drive settled incomplete")
    job1 = str(b1.get("job_id"))
    first_cluster = str(b1.get("cluster") or "")
    if not first_cluster:
        raise Refused("petri_overflow_first_pass_cluster_unknown", f"the first-pass binding of {first_lwid} (job {job1}, {b1['state']}) names no cluster — the rerun proxy runs "
                                                                   "where the first pass ran (its object bank and bank view are there) and that cannot be read; re-bind the first pass (06 --family petri --resubmit)")
    rec_path = record_path(state_dir, tag)
    rec = _load(rec_path)
    if not isinstance(rec, dict):
        raise Refused("petri_overflow_no_record", f"no daemon record at {rec_path} — the daemon has not driven this proxy (08_grade.sh --daemon)")
    atts = _attempts_for_job(rec, job1)
    if not atts:
        raise Refused("petri_overflow_no_record", f"{rec_path} holds no attempt for the first-pass unit's bound job {job1} — the daemon has not driven this proxy (08_grade.sh --daemon)")
    last = atts[-1]
    st = str(last.get("state"))
    if st in ("claimed", "started"):
        raise Refused("petri_overflow_not_pending", f"a drive of job {job1} is running (attempt {last.get('n')} {st}): the overflow pass starts after it settles")
    if st == "completed":
        raise Refused("petri_overflow_not_pending", f"the drive of job {job1} completed every leg (attempt {last.get('n')}): nothing to re-run")
    if st != "incomplete":
        raise Refused("petri_overflow_not_pending", f"the last drive of job {job1} is {st} (attempt {last.get('n')}: {last.get('reason')}) — re-drive the first pass (06 --family petri --resubmit); "
                                                    "the overflow pass runs only after a drive settled `incomplete`")
    result = _load(str(last.get("result_file") or ""))
    if not isinstance(result, dict) or result.get("verdict") != "overflow_pending":
        raise Refused("petri_overflow_not_pending", f"attempt {last.get('n')} of job {job1} is incomplete but its result {last.get('result_file')} is unreadable or not `overflow_pending` "
                                                    f"({(result or {}).get('verdict')!r}) — re-drive the first pass")
    detail = ((result.get("legs") or {}).get("detail"))
    paths = result.get("paths") or {}
    if not isinstance(detail, list) or not detail or not paths.get("out_root"):
        raise Refused("petri_overflow_no_detail", f"the drive's result {last.get('result_file')} carries no per-leg `legs.detail` / `paths.out_root` (a drive older than this contract) — "
                                                  "re-drive the first pass on this tree (06 --family petri --resubmit)")
    named = {int(d["idx"]) for d in detail if isinstance(d, dict) and d.get("status") == LEG_STATUS_OVERFLOW}
    if not named:
        raise Refused("petri_overflow_not_pending", f"the drive's result names no leg `{LEG_STATUS_OVERFLOW}` though its verdict is overflow_pending — torn result; re-drive the first pass")
    manifest = os.path.join(repo, "configs", "capeval", "manifests", f"petri_full22_{tag}.yaml")
    if not os.path.isfile(manifest):
        raise Refused("petri_overflow_ledgers_moved", f"no Petri manifest at {manifest} on this tree — the drive's done-ness reading needs the audit's manifest")
    out_root = str(paths["out_root"])
    ls = legs_status(python, repo, manifest, out_root)
    legs = {int(l["idx"]): l for l in ls["legs"]}
    problems: List[str] = []
    for i in sorted(named):
        l = legs.get(i)
        if l is None:
            problems.append(f"leg {i} (named by the verdict) is not in the frozen enumeration now"); continue
        if l.get("refused"):
            problems.append(f"leg {i} {l.get('cell_id')}: the resumed drive would refuse it ({l.get('reason')})"); continue
        if not l.get("done") and not int(l.get("overflow_unresolved") or 0) > 0:
            problems.append(f"leg {i} {l.get('cell_id')}: neither done nor still on an overflow ledger (unresolved {l.get('unresolved')}, overflow {l.get('overflow_unresolved')})")
    for i, l in sorted(legs.items()):
        if i not in named and not l.get("done") and int(l.get("overflow_unresolved") or 0) > 0:
            problems.append(f"leg {i} {l.get('cell_id')}: on an overflow ledger ({l.get('overflow_unresolved')} entries) but the verdict did not name it")
    if problems:
        raise Refused("petri_overflow_ledgers_moved", "the ledgers differ from the drive's verdict: " + "; ".join(problems) + " — re-drive the first pass (06 --family petri --resubmit)")
    still = [legs[i] for i in sorted(named) if not legs[i].get("done")]
    done = [legs[i] for i in sorted(named) if legs[i].get("done")]
    b2 = ST.binding_state(subs, over_lwid)
    if b2["state"] == "transition":
        raise Refused("petri_overflow_transition", f"the overflow unit {over_lwid} is mid re-run (a transition token since {(b2.get('transition') or {}).get('utc')}): "
                                                   f"the funnel reconciles it first — `submit_twin.py --config <site> --model {model} reconcile-token --lwid {over_lwid}` "
                                                   "(06 --pass overflow runs it before this preflight; a token it leaves standing is a human's decision); nothing decided")
    bound = b2["state"] in ("submitted", "registered")
    if not still:
        if bound:
            raise Refused("petri_overflow_complete", f"every ledgered leg is done ({', '.join(f'{d['idx']}:{d['cell_id']}' for d in done)}) — the overflow pass has nothing left; "
                                                     f"the daemon settles its record from the drive's result (bound job {b2.get('job_id')}); nothing to re-mint")
        raise Refused("petri_overflow_nothing_ledgered", f"the first-pass verdict named legs {sorted(named)} but every one is done now ({', '.join(f'{d['idx']}:{d['cell_id']}' for d in done)}) "
                                                         "without an overflow proxy — nothing to re-run")
    frozen_canonical = [str(l.get("cell_id")) for l in ls["legs"] if not l.get("control_of")]
    config_cells = ST.view_cells_of(model, view, table=models_table)
    if bound:
        cells = bound_declaration(state_dir, b2)
        need = derive_cells(still, frozen_canonical, config_cells)
        short = [c for c in need if c not in cells]
        if short:
            raise Refused("petri_overflow_declaration_short", f"the bound overflow proxy (job {b2.get('job_id')}) declares {cells} but the still-ledgered legs need {need} "
                                                               f"(missing {short}) — a hand-submitted proxy; nothing re-minted")
        source = "bound-attempt"
    else:
        cells = derive_cells(still, frozen_canonical, config_cells)
        source = "computed"
    return {"model": model, "view": view, "tag": tag, "first_lwid": first_lwid, "first_job_id": job1, "first_attempt": last.get("n"), "first_cluster": first_cluster,
            "record": rec_path, "result": last.get("result_file"), "out_root": out_root, "named_legs": sorted(named),
            "still_ledgered": [{k: l.get(k) for k in ("idx", "cell_id", "control_of", "kind", "petri_run_key", "overflow_unresolved")} for l in still],
            "done_legs": [{k: l.get(k) for k in ("idx", "cell_id", "petri_run_key")} for l in done],
            "overflow_lwid": over_lwid, "overflow_bound": bound, "overflow_job_id": b2.get("job_id") if bound else None,
            "overflow_cluster": (str(b2.get("cluster")) if b2.get("cluster") else None) if bound else None,
            "declaration_source": source, "cells": cells}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state-root", required=True, help="the model's state dir ($SISTEER_STATE_DIR/<model>: submissions.jsonl, attempts.jsonl, grade/)")
    ap.add_argument("--model", required=True); ap.add_argument("--view", required=True)
    ap.add_argument("--python", default=sys.executable, help="the interpreter for petri_run.py legs-status (the site's SISTEER_PYTHON)")
    ap.add_argument("--repo", default=REPO_DEFAULT); ap.add_argument("--models-table", default=None)
    a = ap.parse_args(argv)
    try:
        doc = overflow_cells(a.state_root, a.model, a.view, python=a.python, repo=a.repo, models_table=a.models_table)
    except Refused as exc:
        print(f"REFUSED: {exc.name}: {exc.text}", file=sys.stderr, flush=True)
        return 3
    except ValueError as exc:
        print(f"REFUSED: petri_overflow_plan: {exc}", file=sys.stderr, flush=True)
        return 3
    print(json.dumps(doc, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
