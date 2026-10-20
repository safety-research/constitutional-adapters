#!/usr/bin/env python3
"""qwen-seedrep-native-battery — the controls JUDGE unit's registry run identity.

Under main #207 a multi-eval store's `run` records open at the FIRST ROUTED ROW (``hooks.StoreRecorder.on_row`` → ``_open``), and
the row-filer needs ``$SISTEER_REGISTRY_RUN_IDENTITY`` (``hooks.default_store_recorder``). The wave's controls GENERATE child
files no controls rows (the judged rows are the judge's), so the controls `run` records depend on the JUDGE resolving the child's
identity for the store it grades. This module resolves it from the ACCEPTED ATTEMPT'S OWN RECORD — never a hand-supplied string:

* the container writes, per driver child, ``<attempt root>/children/<eval_leg>__<arm>__s<s>.argv.json`` =
  ``{"argv": [...], "env_registry": {SISTEER_REGISTRY_RUN_IDENTITY: <plans path on the volume>, SISTEER_REGISTRY_SPOOL: ...}}``
  (``seedrep_served.py``, the spawn loop; the pull lands ``children/`` beside the stores). The controls child's argv carries
  ``--cell <cell>`` and ``--out-root <attempt root>/exp/sampled/controls_cap<cap>__s<s>``; those two facts bind the record to ONE store.
* the identity file it names lives under ``<...>/registry/plans/<rel>`` on the volume (``seedrep_launch_plan.PLANS_REL``); the SAME
  file is on the grading host under ``RS/registry/plans/<rel>`` — ``registry_plans.py --out RS/registry/plans`` wrote it and
  ``--stage-plans`` copied it to the volume byte-for-byte (the S0 IDENTITY gate holds the two equal).

The judge unit then exports that path as ``SISTEER_REGISTRY_RUN_IDENTITY`` (``grade_battery.sh`` controls phase, the judge step's
env), so the first judged row opens the `run` record with the SAME identity the generate child carried. A record that cannot be
bound — no ``children/`` record for the store, two of them, no identity in the env, a path outside ``registry/plans/``, a file
missing under RS (launch_report F-C: the plan's controls children name ``registry/plans/controls/…`` where the producer writes
``_stores/controls/…`` — until reconciled every controls child refuses here BY NAME), or a map whose route lacks its ``run_id``
(the store would mint a different id) — REFUSES; nothing is guessed and no unit is graded around it (the runbook's rule).

Stdlib only (the runbook's planner imports it beside ``final_rung`` / ``own_cache``).

The GRADING-TIME record (3d): what identity the judge unit exported for a store is persisted BESIDE the judgment, in the store,
BEFORE the judge runs (``REGISTRY_IDENTITY.json``: the identity file + its sha256, or its absence + ``provisional: true`` + why), and
``post`` reads THAT — never a re-resolution at post time (a record / identity file recovered AFTER a provisional grade would otherwise clear
the gap although no registry run was ever opened; the judge skips already-judged controls and re-emits from cache, so re-running it opens
nothing). The lane has no registry backfill operation for a judged store: the gap clears ONLY by a re-grade from scratch (the judgment
artifacts removed, the unit re-run with the identity bound), and the record refuses to be rewritten to `bound` over an existing
provisional grade while its judgment artifacts stand.

    judge_run_identity.py record <store dir> <identity file|-> <bound|provisional> <why|-> <stamp> <unit>
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

RUN_IDENTITY_ENV = "SISTEER_REGISTRY_RUN_IDENTITY"      # sisteer.registry.hooks.RUN_IDENTITY_ENV (restated: stdlib module)
SPOOL_ENV = "SISTEER_REGISTRY_SPOOL"                    # sisteer.registry.emit.SPOOL_ENV
PLANS_REL = "registry/plans"                            # seedrep_launch_plan.PLANS_REL: <RS>/registry/plans on the grading host, <LANE>/<sha7>/registry/plans on the volume
CHILDREN_DIR = "children"                               # seedrep_served.py: <attempt root>/children/<eval_leg>__<arm>__s<s>.argv.json
ARGV_SUFFIX = ".argv.json"
CONTROLS_STEM_PREFIX = "controls__"                     # child_log_name({"eval_leg": "controls", ...}) -> controls__<arm>__s<s>


class IdentityRefused(ValueError):
    """The store's run identity cannot be bound from the accepted attempt's record; the message names why."""


def _flag(argv: Sequence[Any], name: str) -> Optional[str]:
    """The value after ``name`` in an argv (``--cell X``), None when absent or last."""
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return str(argv[i + 1])
    return None


def has_child_records(attempt_root: Path) -> bool:
    """Does the attempt root carry the per-child records at all (a ``children/`` directory)? A root without one was not produced by
    the wave's spawn loop (a fixture, or a pull that predates the records): it can bind no store to a run and the runbook says so on the
    plan instead of refusing the grading; a root WITH records but none / a wrong one for a store is a producer contradiction and REFUSES."""
    return (Path(attempt_root) / CHILDREN_DIR).is_dir()


def child_records(attempt_root: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """Every ``children/*.argv.json`` of the attempt root, parsed; a record that is not ``{"argv": [...], "env_registry": {...}}`` refuses."""
    d = Path(attempt_root) / CHILDREN_DIR
    out: List[Tuple[Path, Dict[str, Any]]] = []
    if not d.is_dir():
        return out
    for p in sorted(d.glob(f"*{ARGV_SUFFIX}")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise IdentityRefused(f"{p}: unreadable child record ({e})") from None
        if not isinstance(rec, dict) or not isinstance(rec.get("argv"), list) or not isinstance(rec.get("env_registry"), dict):
            raise IdentityRefused(f"{p}: not a child record ({{argv: [...], env_registry: {{...}}}})")
        out.append((p, rec))
    return out


def plans_relpath(identity_path: str) -> str:
    """``<anything>/registry/plans/<rel>`` → ``<rel>``; a path not under ``registry/plans/`` refuses."""
    marker = f"/{PLANS_REL}/"
    s = str(identity_path)
    if marker not in s:
        raise IdentityRefused(f"the child's {RUN_IDENTITY_ENV} {s!r} is not under {PLANS_REL}/ (not a planned-run identity file)")
    rel = s.split(marker, 1)[1]
    if not rel or rel.endswith("/"):
        raise IdentityRefused(f"the child's {RUN_IDENTITY_ENV} {s!r} names no file under {PLANS_REL}/")
    return rel


def match_controls_child(attempt_root: Path, store_path: Path, cell_id: str) -> Tuple[Path, Dict[str, Any]]:
    """The ONE controls child record of the attempt whose ``--cell`` is the store's cell id and whose ``--out-root`` directory
    (``controls_cap<cap>__s<s>``) is a segment of the store's path under the attempt root. None or several refuse by name."""
    root = Path(attempt_root)
    try:
        rel_parts = Path(store_path).resolve().relative_to(root.resolve()).parts
    except ValueError:
        raise IdentityRefused(f"store {store_path} is not under the attempt root {root}") from None
    recs = child_records(root)
    if not recs:
        raise IdentityRefused(f"{root / CHILDREN_DIR}: no child record (*{ARGV_SUFFIX}) — the attempt root carries no per-child identity "
                              "(the pull lands children/ beside the stores; an attempt without it cannot bind its stores to a run)")
    hits = []
    for p, rec in recs:
        if not p.name.startswith(CONTROLS_STEM_PREFIX):
            continue
        argv = rec["argv"]
        out_root = _flag(argv, "--out-root")
        if _flag(argv, "--cell") == str(cell_id) and out_root and Path(out_root).name in rel_parts:
            hits.append((p, rec))
    if len(hits) != 1:
        raise IdentityRefused(f"{root / CHILDREN_DIR}: {len(hits)} controls child record(s) name --cell {cell_id!r} with an --out-root segment of "
                              f"{'/'.join(rel_parts)} (need exactly one): {[str(h[0].name) for h in hits]}")
    return hits[0]


def resolve_controls_identity(attempt_root: Path, store_path: Path, cell_id: str, plans_root: Path) -> Dict[str, Any]:
    """The judge unit's registry identity for ONE controls store: ``{"identity_path": <RS plans file>, "child_record": <argv.json>,
    "child_identity": <the path the child exported>, "routes": {<task>: <run_id>}}``. Refuses (``IdentityRefused``) by name."""
    rec_path, rec = match_controls_child(attempt_root, store_path, cell_id)
    ident = rec["env_registry"].get(RUN_IDENTITY_ENV)
    if not ident:
        raise IdentityRefused(f"{rec_path}: env_registry carries no {RUN_IDENTITY_ENV} (the child exported no identity; its rows joined no run)")
    rel = plans_relpath(ident)
    local = Path(plans_root) / rel
    if not local.is_file():
        raise IdentityRefused(f"{local}: the identity file the child exported ({ident}) is not under {plans_root} — the plan's child path and "
                              "the producer's file disagree (launch_report F-C: `registry/plans/controls/<arm>__s<s>.json` vs the producer's "
                              "`_stores/controls/<arm>__s<s>.json`); reconcile the spelling and regenerate, never guess a file")
    try:
        content = json.loads(local.read_text(encoding="utf-8"))
    except ValueError as e:
        raise IdentityRefused(f"{local}: not JSON ({e})") from None
    if not isinstance(content, dict) or not content or not all(isinstance(v, dict) for v in content.values()):
        raise IdentityRefused(f"{local}: a controls store's identity is a per-task MAP {{<writer task>: identity}} (WS4-H `_stores/controls`), "
                              f"got {type(content).__name__}" + (f" with keys {sorted(content)[:6]}" if isinstance(content, dict) else ""))
    routes: Dict[str, str] = {}
    for task, one in content.items():
        rid = one.get("run_id")
        if not isinstance(rid, str) or not rid:
            raise IdentityRefused(f"{local}: route {task!r} carries no run_id — the store would mint a different id from its own key")
        routes[str(task)] = rid
    return {"identity_path": str(local), "child_record": str(rec_path), "child_identity": str(ident), "routes": routes}


# --------------------------------------------------------------------------- the grading-time record
GRADING_IDENTITY_FILE = "REGISTRY_IDENTITY.json"          # <store>/REGISTRY_IDENTITY.json -- written by the judge unit BEFORE its judge step
JUDGE_ARTIFACTS: Tuple[str, ...] = ("judge_summary.json", "judge_items.jsonl")   # served_capability_judge.py SUMMARY_NAME / ITEMS_NAME
REGRADE_HINT = ("the gap clears only by a re-grade from scratch: remove the store's judge_summary.json / judge_items.jsonl and its private judge "
                "cache, then re-run the controls unit with the identity bound (this lane has no registry backfill operation for a judged store)")


def judge_artifacts_present(store: Path) -> List[str]:
    return [n for n in JUDGE_ARTIFACTS if (Path(store) / n).is_file()]


def grading_identity_path(store: Path) -> Path:
    return Path(store) / GRADING_IDENTITY_FILE


def record_grading_identity(store: Path, *, identity_path: Optional[str], provisional: bool, unresolved: Optional[str], stamp: str, unit: str) -> str:
    """Write ``<store>/REGISTRY_IDENTITY.json`` for the judge step that follows: ``{bound, identity_path, identity_sha256, provisional,
    unresolved, recorded_at, stamp, unit}``. Fail-closed rules: a store whose judgment artifacts stand keeps its EXISTING record (the
    judgment was made under that identity state) -- rewriting a `provisional` record to `bound` there REFUSES by name (re-grade from
    scratch), and a bound record's identity file may not change under standing judgments either; a store with no judgment artifacts
    (fresh, or re-graded from scratch) gets a fresh record."""
    store = Path(store)
    if not store.is_dir():
        raise IdentityRefused(f"{store}: not a store directory")
    bound = identity_path is not None and not provisional
    if bound:
        ip = Path(identity_path)
        if not ip.is_file():
            raise IdentityRefused(f"{ip}: the identity file to record does not exist")
        sha = hashlib.sha256(ip.read_bytes()).hexdigest()
    else:
        sha = None
    rec = {"bound": bound, "identity_path": str(identity_path) if bound else None, "identity_sha256": sha, "provisional": bool(provisional),
           "unresolved": None if bound else (unresolved or "no registry run identity"), "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "stamp": stamp, "unit": unit, "regrade_hint": None if bound else REGRADE_HINT}
    out = grading_identity_path(store)
    standing = judge_artifacts_present(store)
    if standing and not out.is_file():
        # judgments stand with NO grading-time record (a store graded before the record existed, or outside the unit): the identity state they
        # were made under is unknown, and the judge's resume skips them -- recording ANY state now (bound OR provisional) would claim a
        # provenance no judgment carries, and a bound one would clear the gap with no registry run opened. Refuse; re-grade from scratch.
        raise IdentityRefused(f"{store}: judgment artifacts {standing} stand with no {GRADING_IDENTITY_FILE} beside them -- the identity state they "
                              f"were made under is unrecorded and the judge's resume would re-emit them; nothing is recorded now; {REGRADE_HINT}")
    if out.is_file() and standing:
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
        except ValueError as e:
            raise IdentityRefused(f"{out}: unreadable grading-time record ({e}) beside standing judgments {standing}; {REGRADE_HINT}")
        if not isinstance(prev, dict) or "bound" not in prev or "provisional" not in prev:
            raise IdentityRefused(f"{out}: not a grading-time identity record (lacks bound / provisional) beside standing judgments {standing}; {REGRADE_HINT}")
        if prev.get("provisional") and bound:
            raise IdentityRefused(f"{out}: the store was graded PROVISIONALLY (no registry run identity; unit {prev.get('unit')}) and its judgment "
                                  f"artifacts {standing} stand -- recording a bound identity now would clear the gap with no registry run ever "
                                  f"opened; {REGRADE_HINT}")
        if prev.get("bound") and bound and prev.get("identity_sha256") != sha:
            raise IdentityRefused(f"{out}: the standing judgments were made under identity {prev.get('identity_path')} (sha {str(prev.get('identity_sha256'))[:12]}), "
                                  f"not {identity_path} (sha {sha[:12]}); {REGRADE_HINT}")
        return f"[judge-identity] kept: {out} ({'bound' if prev.get('bound') else 'PROVISIONAL'}; judgments stand: {standing})"
    out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return f"[judge-identity] recorded: {out} ({'bound to ' + str(identity_path) if bound else 'PROVISIONAL: ' + str(rec['unresolved'])[:120]})"


def read_grading_identity(store: Path) -> Dict[str, Any]:
    """What `post` publishes for a store: ``{"identity_unresolved": None | why, "provisional": bool, "grading_identity": <record or None>}``, read
    from the grading-time record ONLY. No record -> unresolved ("no grading-time identity record"; a store graded before this record existed, or
    outside the unit -- re-grade from scratch); a provisional record -> unresolved with its why + the re-grade hint; a bound record -> None."""
    store = Path(store)
    out_p = grading_identity_path(store)
    if not out_p.is_file():
        return {"identity_unresolved": f"no grading-time identity record ({GRADING_IDENTITY_FILE}) beside the judgment -- the store was graded outside the "
                                       f"controls unit or before the record existed; {REGRADE_HINT}", "provisional": None, "grading_identity": None}
    try:
        rec = json.loads(out_p.read_text(encoding="utf-8"))
    except ValueError as e:
        return {"identity_unresolved": f"{out_p}: unreadable grading-time identity record ({e}); {REGRADE_HINT}", "provisional": None, "grading_identity": None}
    if rec.get("bound") and rec.get("identity_path"):
        return {"identity_unresolved": None, "provisional": False, "grading_identity": rec}
    return {"identity_unresolved": f"graded PROVISIONALLY with no registry run identity (unit {rec.get('unit')}, {rec.get('recorded_at')}): "
                                   f"{rec.get('unresolved')}; {REGRADE_HINT}", "provisional": True, "grading_identity": rec}


def main(argv: Optional[Sequence[str]] = None) -> None:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 7 or a[0] != "record" or a[3] not in ("bound", "provisional"):
        raise SystemExit("usage: judge_run_identity.py record <store dir> <identity file|-> <bound|provisional> <why|-> <stamp> <unit>")
    try:
        print(record_grading_identity(Path(a[1]), identity_path=None if a[2] == "-" else a[2], provisional=(a[3] == "provisional"),
                                      unresolved=None if a[4] == "-" else a[4], stamp=a[5], unit=a[6]))
    except IdentityRefused as e:
        raise SystemExit(f"REFUSED: {e}")


if __name__ == "__main__":
    main()
