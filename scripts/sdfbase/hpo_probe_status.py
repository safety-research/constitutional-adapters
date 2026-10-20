"""hpo_probe_status — what the serving probe's record says the LoRA search's probe phase should do next.

The serving probe (scripts/sdfbase/serving_probe.py, the serving lane) checkpoints its launch list in
``probe_progress.json`` after every launch, exits ``EXIT_INCOMPLETE`` (4) when its time budget stops it before every
decision-relevant launch ran, self-resubmits with ``--resume`` on the SAME record, and writes ``allowed_pairs.json``
ONLY when complete. So a missing ``allowed_pairs.json`` is NOT "not started": ``reproduce/03_hpo.sh --phase probe``
reads the checkpoint through this module and reports INCOMPLETE + the successor job, or refuses to proceed — it never
re-submits a fresh probe over a partial record (the coordinator's item of 2026-09-29).

    printf '%s' "$(cat probe_progress.json)" | python hpo_probe_status.py --job-active 1   # -> "<STATE>\\t<detail>"

States: READY (a COMPLETE checkpoint AND a verdict + menu tied to the record's CURRENT result generation — the only state
in which 03 may fetch the menu), NOT_STARTED (no checkpoint, no job), RUNNING (no checkpoint yet, a job is queued or
running), INCOMPLETE_RUNNING (a partial checkpoint and a job — the probe's own successor — is queued/running),
INCOMPLETE_STALLED (a partial checkpoint and NO job: the attempt cap or a STOP sentinel — a human resumes it),
VERDICT_REFUSED (a complete checkpoint but no menu: the verdict refused certification — read probe_verdict.json),
MENU_STALE (a menu exists but the checkpoint is not complete or the menu's result generation is not the record's — a
result was reopened after it was written; NEVER fetch it), REOPEN_IN_PROGRESS (a revocation receipt of the record is
consumed and its certifying rewrite has not completed — a reverdict / receipted resume running, or one that died; whatever
else stands — a menu that is then the PRE-reopen selection, a complete checkpoint — the record is unreadable: NEVER fetch,
never resume afresh; `serving_probe.py reverdict --record <dir> --resume-interrupted` recovers it under that receipt, one
recovery at a time; Sol on 2ad8b3923), PARITY_STALE (the menu IS the record's current result
generation, but the PARITY-PASS record it read adapter parity from — `parity_record` in the verdict and the menu, D163 — is
not what was stamped: absent, its checkpoint not complete, its live result generation another than the stamped one (a
reference redo after the main verdict was written), its plan another checkpoint's / another scale table, or the verdict
and the menu name different parity records; NEVER fetch it — Sol on b497680d2, D191), MENU_UNVERIFIED (stdin mode only: a menu exists but this
mode cannot verify its generation — run `--record <dir>`), FAILED_START (`--record` mode: the carrier wrote its attempt
record `probe_attempt.json` — or a plan / RESULT stands — but no `probe_progress.json` exists and no job is active: the
attempt died between the two files (venv bootstrap, an import failure); the detail names the attempt — job id, host,
started — and says whether a plan exists (resume it) or not (a fresh start is required). With a job active the same record
reads RUNNING. 03 handles FAILED_START explicitly; SUBMIT_STATES and FETCH_STATES are unchanged.

THE STATUS CONTRACT (Sol's stage-E review, 2026-09-30). Menu presence alone never means READY. `--record <dir>` reads
the record itself (probe_progress.json, probe_verdict.json, allowed_pairs.json, launches/*.json) and prints
`<STATE>\t<detail>\t<json>` where the json carries: complete, menu_present, verdict_present, generation (the record's
current result generation = sha256 over sorted "name:sha256(bytes)" of launches/*.json — the same algorithm as
sisteer.capeval.serving.geometry.record_result_generation), n_results, verdict_generation, menu_generation,
generation_check (stamped_match | recomputed_match | mismatch | unverifiable | absent), replan_required, and the CROSS-RECORD
check: parity_check (none = the verdict read no parity record | current | absent | plan_mismatch | in_progress | incomplete |
mismatch), parity_why, parity_record {dir, stamped_generation, live_generation, stamped_n_results, live_n_results, complete,
attempt, hold, own_verdict_present, reference_check_present}. `in_progress` (Sol on 6a09ec13c): the parity checkpoint carries a
driver's attempt (a reference-only re-run marks it so BEFORE its HF phase and a dead worker leaves it), or the record lacks its own
probe_verdict.json / reference_check.json (archived first by that re-run), or its own verdict is stamped for another generation —
refused whatever the (unmoved) result generation says. READY iff
complete AND menu_present AND verdict_present AND generation_check in (stamped_match, recomputed_match) AND parity_check in
(none, current) — the main record's own stamps say nothing about the parity record's results, so a parity-reference redo that
leaves the main generation unchanged is caught HERE, by reading the parity record the stamp names (D191).
`recomputed_match`: an UNSTAMPED verdict (written before the stamp existed) is verified by recomputing from the result
set it LISTS — every launch it names with a status other than not_run must exist in the record with that status and
no other result may exist — so a verdict a running carrier writes without the stamp is still verifiable.

    python3 hpo_probe_status.py --record /path/to/record --job-active 0|1     # the front door's call (login host: stdlib only)
    printf '%s' "$(cat probe_progress.json)" | python hpo_probe_status.py --job-active 1   # the checkpoint-only mode (never READY)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

READY = "READY"
NOT_STARTED = "NOT_STARTED"
RUNNING = "RUNNING"
INCOMPLETE_RUNNING = "INCOMPLETE_RUNNING"
INCOMPLETE_STALLED = "INCOMPLETE_STALLED"
VERDICT_REFUSED = "VERDICT_REFUSED"
MENU_STALE = "MENU_STALE"
MENU_UNVERIFIED = "MENU_UNVERIFIED"
FAILED_START = "FAILED_START"
PARITY_STALE = "PARITY_STALE"
REOPEN_IN_PROGRESS = "REOPEN_IN_PROGRESS"
STATES = (READY, NOT_STARTED, RUNNING, INCOMPLETE_RUNNING, INCOMPLETE_STALLED, VERDICT_REFUSED, MENU_STALE, MENU_UNVERIFIED, FAILED_START,
          PARITY_STALE, REOPEN_IN_PROGRESS)
#: the states in which 03_hpo.sh may submit a probe job: exactly one — nothing recorded yet
SUBMIT_STATES = (NOT_STARTED,)
#: the ONE state in which 03_hpo.sh may fetch the menu / the geometry record
FETCH_STATES = (READY,)
PROGRESS_FILE = "probe_progress.json"
VERDICT_FILE = "probe_verdict.json"
MENU_FILE = "allowed_pairs.json"
ATTEMPT_FILE = "probe_attempt.json"     # the carrier's attempt record, written at START (serving_probe.sbatch)
PLAN_FILE = "probe_plan.json"
RESULT_FILE = "RESULT"
LAUNCHES_DIR = "launches"
GENERATION_KEY = "result_generation"
NOT_RUN = "not_run"
RECEIPTS_DIR = "reopen_receipts"
#: a consumed receipt whose reopen's certifying rewrite has not completed: `<gen>.<nonce>.inprogress.<tag>.json` (serving_probe
#: consume_reopen_receipt / finish_reopen_receipt) — a record showing one beside a complete checkpoint and no menu
#: is an INTERRUPTED reopen, recovered by `serving_probe.py reverdict --record <dir> --resume-interrupted`, never by a new revocation
INPROGRESS_RECEIPT_GLOB = "*.inprogress.*.json"
PARITY_RECORD_KEY = "parity_record"          # the verdict's / menu's stamp of the parity-pass record assertion 4 was read from
REFERENCE_CHECK_FILE = "reference_check.json"   # the record's own reference-phase record — archived first by a reference-only re-run
#: the parity_check values under which a menu may be READY: no parity record was read, or the one read is exactly as stamped
PARITY_CHECK_CURRENT = ("none", "current")


def record_generation(record_dir: Path) -> Optional[Dict[str, Any]]:
    """The record's CURRENT result generation — the same algorithm as ``geometry.record_result_generation`` (kept
    stdlib-only here so a cluster login host can run it): sha256 over the sorted ``"<name>:<sha256(bytes)>"`` lines of
    ``launches/*.json``; None when the record has no launches directory."""
    d = Path(record_dir) / LAUNCHES_DIR
    if not d.is_dir():
        return None
    lines = [f"{f.name}:{hashlib.sha256(f.read_bytes()).hexdigest()}" for f in sorted(d.glob("*.json")) if f.is_file()]
    return {"generation": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(), "n_results": len(lines)}


def _read_json(p: Path) -> Optional[Any]:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"unparseable": True}


def _result_statuses(record_dir: Path) -> Dict[str, str]:
    """``{launch_id: status}`` over the record's result files (a torn file counts as present with status ``unreadable``)."""
    out: Dict[str, str] = {}
    d = Path(record_dir) / LAUNCHES_DIR
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        if not f.is_file():
            continue
        doc = _read_json(f)
        if isinstance(doc, dict) and doc.get("launch_id") and not doc.get("unparseable"):
            out[str(doc["launch_id"])] = str(doc.get("status"))
        else:
            out[f.stem] = "unreadable"
    return out


def verify_parity_record(record_dir: Path, verdict: Optional[Any], menu: Optional[Any]) -> Dict[str, Any]:
    """The CROSS-RECORD check: the verdict + menu stamp the parity-pass record they read adapter parity
    from (``parity_record``: dir, result_generation, n_results, plan_launches, windows, synthetic_scale_by_set); the main record's
    own generation stamps do not move when THAT record's results do (a parity-reference redo), so READY must read the parity
    record itself. Returns ``{parity_check, parity_why, parity_record}``: ``none`` (no parity record was read — nothing to
    hold), ``current`` (a complete checkpoint, a plan of the same checkpoint / scale table / launch count / windows, and a live
    result generation equal to the stamped one, its checkpoint stamped the same), else ``absent`` / ``plan_mismatch`` /
    ``incomplete`` / ``mismatch`` with a named reason."""
    vpr = verdict.get(PARITY_RECORD_KEY) if isinstance(verdict, dict) else None
    mpr = menu.get(PARITY_RECORD_KEY) if isinstance(menu, dict) else None
    info: Dict[str, Any] = {"parity_check": "none", "parity_why": None, "parity_record": None}
    if vpr is None and mpr is None:
        return info
    if not isinstance(vpr, dict) or not isinstance(mpr, dict) or str(mpr.get("dir")) != str(vpr.get("dir")) \
            or mpr.get(GENERATION_KEY) != vpr.get(GENERATION_KEY):
        info["parity_check"] = "mismatch"
        info["parity_why"] = (f"the verdict and the menu do not name the same parity record (verdict {vpr if not isinstance(vpr, dict) else str(vpr.get('dir'))} @ "
                              f"{str((vpr or {}).get(GENERATION_KEY) if isinstance(vpr, dict) else None)[:12]}, menu "
                              f"{mpr if not isinstance(mpr, dict) else str(mpr.get('dir'))} @ {str((mpr or {}).get(GENERATION_KEY) if isinstance(mpr, dict) else None)[:12]})"
                              " — one of them was rewritten without the other")
        return info
    pdir = Path(str(vpr.get("dir") or ""))
    rec: Dict[str, Any] = {"dir": str(pdir), "stamped_generation": vpr.get(GENERATION_KEY), "stamped_n_results": vpr.get("n_results"),
                           "live_generation": None, "live_n_results": None, "complete": None}
    info["parity_record"] = rec
    pplan = _read_json(pdir / PLAN_FILE) if vpr.get("dir") else None
    if not isinstance(pplan, dict) or pplan.get("unparseable"):
        info["parity_check"] = "absent"
        info["parity_why"] = (f"the parity record {pdir} the verdict read adapter parity from is absent or unreadable (no readable "
                              f"{PLAN_FILE}) — its evidence cannot be re-read")
        return info
    problems = []
    if pplan.get("parity_pass") is not True:
        problems.append(f"its plan is not a parity pass (parity_pass={pplan.get('parity_pass')!r})")
    mplan = _read_json(Path(record_dir) / PLAN_FILE)
    if isinstance(mplan, dict) and not mplan.get("unparseable"):
        for k in ("model_key", "model_id", "revision"):
            if str(pplan.get(k)) != str(mplan.get(k)):
                problems.append(f"its {k} {pplan.get(k)!r} is not this record's {mplan.get(k)!r}")
    if dict(pplan.get("synthetic_scale_by_set") or {}) != dict(vpr.get("synthetic_scale_by_set") or {}):
        problems.append(f"its scale table {pplan.get('synthetic_scale_by_set')} is not the stamped {vpr.get('synthetic_scale_by_set')}")
    if vpr.get("plan_launches") is not None and pplan.get("n_launches") != vpr.get("plan_launches"):
        problems.append(f"it plans {pplan.get('n_launches')} launches, the stamp says {vpr.get('plan_launches')}")
    if vpr.get("windows") is not None and [int(w) for w in (pplan.get("windows") or [])] != [int(w) for w in vpr.get("windows")]:
        problems.append(f"its windows {pplan.get('windows')} are not the stamped {vpr.get('windows')}")
    # the parity record's OWN state is read before any verdict on it: the front door's PARITY_STALE remedy
    # — the receipted `reverdict --parity-record` — may only go out over a parity record that is COMPLETE and at rest, and it
    # judges that from this block's `complete` / `attempt` / `hold` / `live_generation`; a `plan_mismatch` (the parity record was
    # re-planned — the very case the remedy exists for) used to return before reading them, so `complete` was None and the
    # remedy could never be routed
    pprog = _read_json(pdir / PROGRESS_FILE)
    rec["complete"] = bool(isinstance(pprog, dict) and pprog.get("complete") is True)
    # a reference-only re-run on a COMPLETE parity record marks its checkpoint in progress (attempt + hold,
    # complete=false) BEFORE its HF phase and archives the record's own derived artifacts first; its result generation does not
    # move until the first per-launch rewrite — so an unchanged generation says nothing, and a driver's hold, or the absence of
    # the record's own verdict / reference_check.json, refuses whatever the generation says (a dead worker leaves both behind)
    attempt = pprog.get("attempt") if isinstance(pprog, dict) else None
    rec["attempt"] = attempt if isinstance(attempt, dict) else None
    rec["hold"] = pprog.get("hold") if isinstance(pprog, dict) else None
    live = record_generation(pdir)
    rec["live_generation"] = live["generation"] if live else None
    rec["live_n_results"] = live["n_results"] if live else None
    # AT REST: `complete` alone is written by run_launches.record() on the LAST result, BEFORE the separate
    # reference and verdict phases — a re-planned parity pass read in that window, or one whose own verdict was archived by a crash,
    # must never be presented to the PARITY_STALE remedy as eligible. At rest = a complete checkpoint, no driver's attempt / hold,
    # the checkpoint's stamped generation equal to the live one (its final write happened AFTER the last result), and the record's
    # OWN verdict and reference artefacts present. The front door's remedy tests THIS, never `complete`.
    pprog_gen = pprog.get(GENERATION_KEY) if isinstance(pprog, dict) else None
    rec["checkpoint_generation"] = pprog_gen
    rec["checkpoint_generation_matches_live"] = bool(live and pprog_gen and pprog_gen == live["generation"])
    rec["own_verdict_present"] = (pdir / VERDICT_FILE).is_file()
    rec["own_reference_check_present"] = (pdir / REFERENCE_CHECK_FILE).is_file()
    rec["at_rest"] = bool(rec["complete"] and rec["attempt"] is None and rec["hold"] in (None, "")
                          and rec["checkpoint_generation_matches_live"] and rec["own_verdict_present"] and rec["own_reference_check_present"])
    if problems:
        info["parity_check"] = "plan_mismatch"
        info["parity_why"] = f"the parity record {pdir} is not the one the verdict read: " + "; ".join(problems) + " — it was re-planned since"
        return info
    if isinstance(attempt, dict) and attempt:
        info["parity_check"] = "in_progress"
        info["parity_why"] = (f"the parity record {pdir}'s checkpoint says a driver holds it (attempt started {attempt.get('started_utc') or '?'}, job "
                              f"{attempt.get('job_id') or '?'} on {attempt.get('host') or '?'}"
                              + (", a reference-only re-run" if attempt.get("reference_only") else "")
                              + (f", interrupted {attempt.get('interrupted_utc')}" if attempt.get("interrupted_utc") else "")
                              + f"; hold {rec['hold']!r}) — its parity evidence is being redone (or that attempt died before its completing "
                              "write); an unchanged result generation certifies nothing while it stands")
        return info
    if not rec["complete"]:
        info["parity_check"] = "incomplete"
        info["parity_why"] = (f"the parity record {pdir} has no complete checkpoint ({PROGRESS_FILE} "
                              f"{'absent' if pprog is None else 'unreadable' if not isinstance(pprog, dict) or pprog.get('unparseable') else 'complete=false'}"
                              + (f", stopped: {pprog.get('stopped_reason')}" if isinstance(pprog, dict) and pprog.get("stopped_reason") else "") + ")"
                              " — a re-run of its launches or of its reference phase is under way or stalled; its parity evidence is in motion")
        return info
    if live is None or live["generation"] != vpr.get(GENERATION_KEY):
        info["parity_check"] = "mismatch"
        info["parity_why"] = (f"the parity record {pdir} is at result generation {str(rec['live_generation'])[:12]} ({rec['live_n_results']} result "
                              f"files), the verdict read it at {str(vpr.get(GENERATION_KEY))[:12]} ({vpr.get('n_results')}) — its results (a "
                              "reference redo, a re-run) moved after this record's verdict was written; the menu's adapter parity is not current")
        return info
    # the generation is the stamped one — which says nothing about a rewrite that has not reached its first per-launch write
    #: the record's own derived artifacts, archived first by a reference-only re-run, must be back, and its own
    # verdict must be of THIS generation
    pverdict = _read_json(pdir / VERDICT_FILE)
    rec["own_verdict_present"] = isinstance(pverdict, dict) and not pverdict.get("unparseable")
    rec["reference_check_present"] = (pdir / REFERENCE_CHECK_FILE).is_file()
    if not rec["own_verdict_present"] or not rec["reference_check_present"]:
        missing = [n for n, ok in ((VERDICT_FILE, rec["own_verdict_present"]), (REFERENCE_CHECK_FILE, rec["reference_check_present"])) if not ok]
        info["parity_check"] = "in_progress"
        info["parity_why"] = (f"the parity record {pdir} lacks its own derived artifact(s) {missing} — a reference-only re-run archives them "
                              "before it rewrites the results (and a worker that died before its completing write leaves them archived), or "
                              "its verdict never ran; a record at rest carries both")
        return info
    vg = pverdict.get(GENERATION_KEY)
    if vg and vg != live["generation"]:
        info["parity_check"] = "in_progress"
        info["parity_why"] = (f"the parity record {pdir}'s own verdict is stamped for result generation {str(vg)[:12]}, its result files are "
                              f"{live['generation'][:12]} — its results moved after its verdict; rewrite its verdict (the carrier's `verdict`) "
                              "before this record's")
        return info
    pg = pprog.get(GENERATION_KEY) if isinstance(pprog, dict) else None
    if pg and pg != live["generation"]:
        info["parity_check"] = "mismatch"
        info["parity_why"] = (f"the parity record {pdir}'s checkpoint is stamped for result generation {str(pg)[:12]}, its result files are "
                              f"{live['generation'][:12]} — the record is mid-rewrite")
        return info
    info["parity_check"] = "current"
    return info


def verify_menu(record_dir: Path) -> Dict[str, Any]:
    """Everything READY depends on, read from the record: the checkpoint's ``complete``, whether verdict / menu exist,
    the record's current generation and how the verdict + menu relate to it (``generation_check``), and — when the verdict
    read adapter parity from a parity-pass record — whether THAT record still is what was stamped (``parity_check``)."""
    record_dir = Path(record_dir)
    verdict = _read_json(record_dir / VERDICT_FILE)
    menu = _read_json(record_dir / MENU_FILE)
    info = _verify_menu_generation(record_dir, verdict, menu)
    # the menu's exclusions when a menu exists, else the verdict's (a REFUSED verdict writes no menu but still names every pair's reasons):
    # a parity record refused for want of reference evidence may ALSO carry a pair excluded by a failed engine launch,
    # which a reference-only re-run cannot repair
    info.update(serving_excluded(menu if isinstance(menu, dict) and not menu.get("unparseable") else verdict))
    if info["menu_present"] and info["verdict_present"]:
        info.update(verify_parity_record(record_dir, verdict, menu))
    else:
        info.update({"parity_check": "none", "parity_why": None, "parity_record": None})
    return info


#: the menu's exclusion reasons that are ADAPTER-PARITY verdicts (serving_probe.PARITY_REASON_PREFIXES + the no-evidence reason): a pair
#: excluded for any OTHER reason (a start-time OOM, a KV shortfall, an unrun ladder role, an undecided launch) has LAUNCHES TO RUN
PARITY_REASON_PREFIXES = ("reference_mismatch:", "reference_undecidable:", "reference_missing:")


def serving_excluded(menu: Optional[Any]) -> Dict[str, Any]:
    """``{serving_excluded: [[rank, set, reason]], serving_excluded_n}`` — the menu's pairs excluded for a SERVING reason (the first
    recorded reason is not an adapter-parity verdict) or left undecided. The front door's `--reopen` on a PARITY_STALE record RESUMES
    such a record (its launches are what the reopen is for; the resume's own verdict re-reads the parity record at its current
    generation) instead of routing the parity re-verdict alone."""
    rows: List[List[Any]] = []
    for e in (menu.get("excluded") or []) if isinstance(menu, dict) and not menu.get("unparseable") else []:
        if not isinstance(e, dict):
            continue
        reason = str(e.get("reason") or "")
        if e.get("decided") is False or not reason.startswith(PARITY_REASON_PREFIXES):
            rows.append([e.get("rank"), e.get("set"), reason[:120]])
    return {"serving_excluded": rows, "serving_excluded_n": len(rows)}


def _verify_menu_generation(record_dir: Path, verdict: Optional[Any], menu: Optional[Any]) -> Dict[str, Any]:
    """The SAME-record half of :func:`verify_menu`: the checkpoint, the artifacts' presence, the generation stamps."""
    progress = _read_json(record_dir / PROGRESS_FILE)
    gen = record_generation(record_dir)
    info: Dict[str, Any] = {
        "record": str(record_dir),
        "complete": bool(isinstance(progress, dict) and progress.get("complete") is True),
        "checkpoint_present": progress is not None,
        "verdict_present": isinstance(verdict, dict) and not verdict.get("unparseable"),
        "menu_present": isinstance(menu, dict) and not menu.get("unparseable"),
        "generation": gen["generation"] if gen else None, "n_results": gen["n_results"] if gen else 0,
        "verdict_generation": (verdict.get(GENERATION_KEY) if isinstance(verdict, dict) else None),
        "menu_generation": (menu.get(GENERATION_KEY) if isinstance(menu, dict) else None),
        "replan_required": list((progress or {}).get("replan_required") or []) if isinstance(progress, dict) else [],
        # the checkpoint's own driver marks — a reopen of THIS record (its reference re-run) needs it at rest, not merely READY
        "attempt": (progress.get("attempt") if isinstance(progress, dict) else None),
        "hold": (progress.get("hold") if isinstance(progress, dict) else None),
        "generation_check": "absent", "why": None}
    if not info["menu_present"] or not info["verdict_present"]:
        info["why"] = "no menu" if not info["menu_present"] else "menu without a verdict"
        return info
    if gen is None:
        info["generation_check"] = "unverifiable"
        info["why"] = f"the record has no {LAUNCHES_DIR}/ directory — nothing to verify the menu against"
        return info
    vg, mg = info["verdict_generation"], info["menu_generation"]
    if vg or mg:
        if vg == gen["generation"] and mg == gen["generation"]:
            info["generation_check"] = "stamped_match"
        else:
            info["generation_check"] = "mismatch"
            info["why"] = (f"verdict {str(vg)[:12]} / menu {str(mg)[:12]} stamped for another result generation than the record's "
                           f"{gen['generation'][:12]} ({gen['n_results']} result files) — a result was reopened or re-run after they were written")
        return info
    # unstamped (pre-stamp code): recompute from the result set the verdict LISTS
    listed = {str(l.get("id")): str(l.get("status")) for l in (verdict.get("launches") or [])
              if isinstance(l, dict) and l.get("status") not in (None, NOT_RUN)}
    actual = _result_statuses(record_dir)
    if listed and listed == actual:
        info["generation_check"] = "recomputed_match"
    else:
        info["generation_check"] = "mismatch"
        missing = sorted(set(listed) - set(actual))[:4]
        extra = sorted(set(actual) - set(listed))[:4]
        changed = sorted(k for k in set(listed) & set(actual) if listed[k] != actual[k])[:4]
        info["why"] = (f"unstamped verdict lists {len(listed)} results, the record holds {len(actual)}"
                       + (f"; missing {missing}" if missing else "") + (f"; new {extra}" if extra else "")
                       + (f"; status changed {changed}" if changed else "") + " — recomputation does not match")
    return info


def interrupted_reopen(record_dir: Path) -> Optional[Dict[str, Any]]:
    """The record's IN-PROGRESS reopen receipt, summarised WITHOUT its nonce or file name (never printed): the mint time, the
    consumer (what / action / started / pid / host / job) and how often it was resumed; None when no receipt is in progress."""
    d = Path(record_dir) / RECEIPTS_DIR
    if not d.is_dir():
        return None
    rows = []
    for f in sorted(d.glob(INPROGRESS_RECEIPT_GLOB)):
        doc = _read_json(f)
        c = (doc.get("consume") or {}) if isinstance(doc, dict) and not doc.get("unparseable") else {}
        rows.append({"minted_utc": (doc or {}).get("minted_utc") if isinstance(doc, dict) else None, "started_utc": c.get("started_utc"),
                     "what": c.get("what"), "action": c.get("action"), "pid": c.get("pid"), "host": c.get("host"), "job_id": c.get("job_id"),
                     "resumed": len(c.get("resumed") or []), "interrupted_by": c.get("interrupted_by")})
    if not rows:
        return None
    return {**rows[-1], "n_in_progress": len(rows)}


def probe_state_record(record_dir: Path, *, job_active: bool) -> tuple[str, str, Dict[str, Any]]:
    """``(state, detail, info)`` from the record itself (the front door's mode). READY requires a complete checkpoint AND a
    verdict + menu whose generation is the record's (stamped, or recomputed for an unstamped verdict)."""
    info = verify_menu(record_dir)
    progress = _read_json(Path(record_dir) / PROGRESS_FILE)
    info["interrupted_reopen"] = interrupted_reopen(record_dir)
    if info["interrupted_reopen"] and info["complete"]:
        # a reopen IN PROGRESS (a receipt consumed, its certifying rewrite not completed — a reverdict running, or one
        # that died) on a record whose checkpoint says COMPLETE makes the record UNREADABLE for every consumer whatever else stands: a
        # menu still present is the PRE-reopen selection (the consume precedes the archive); never READY / CURRENT, never fetched, never
        # resumed afresh. (A record whose checkpoint is NOT complete — a receipted resume or reference-only re-run under way, held by a
        # driver — keeps its INCOMPLETE_* state below, non-fetchable as well, with the receipt named: the resume of THAT run claims it.)
        ir = info["interrupted_reopen"]
        return REOPEN_IN_PROGRESS, (f"a reopen of this record is IN PROGRESS (or died mid-way) under the receipt minted {ir.get('minted_utc') or '?'} — "
                                    f"consumed {ir.get('started_utc') or '?'} for {ir.get('what') or '?'} by pid {ir.get('pid') or '?'} on {ir.get('host') or '?'}, "
                                    f"job {ir.get('job_id') or '-'}" + (f", resumed {ir['resumed']}x" if ir.get("resumed") else "")
                                    + f"; its certifying rewrite has not completed. NEVER fetch: {MENU_FILE} "
                                    + ("still stands but is the PRE-reopen selection" if info["menu_present"] else "is absent (archived by the reopen)")
                                    + f"; recover with `serving_probe.py reverdict --record {record_dir} --resume-interrupted` (the rewrite re-runs under "
                                    "that receipt's epoch, one recovery at a time under the record's reopen lock); NEVER a new revocation over it"), info
    if info["menu_present"]:
        if info["complete"] and info["generation_check"] in ("stamped_match", "recomputed_match"):
            if info["parity_check"] not in PARITY_CHECK_CURRENT:
                # this record's stamps agree with its own results, but the PARITY record they read adapter parity from moved
                # (or is gone / re-planned / incomplete): the menu's parity basis is not current — never fetched, never consumed
                pr = info.get("parity_record") or {}
                return PARITY_STALE, (f"{MENU_FILE} present and tied to this record's current result generation ({info['generation_check']}), "
                                      f"but its ADAPTER-PARITY basis is not current ({info['parity_check']}) — NEVER fetch it: {info['parity_why']}"
                                      + ("; a probe job is queued/running" if job_active else
                                         f"; no probe job — when the parity record {pr.get('dir') or '(unknown)'} is complete and at rest, rewrite this "
                                         "record's verdict against it (`serving_probe.py reverdict --parity-record <dir>` under a reopen receipt: the "
                                         "front door's --phase probe --reopen)")), info
            return READY, (f"{MENU_FILE} present and tied to the record's current result generation "
                           f"({info['generation_check']}, {info['n_results']} result files; checkpoint complete"
                           + (f"; parity record current at {str((info.get('parity_record') or {}).get('live_generation'))[:12]}"
                              if info["parity_check"] == "current" else "") + ")"), info
        why = info["why"] or ("the checkpoint is not complete" if not info["complete"] else info["generation_check"])
        detail = (f"{MENU_FILE} present but STALE — NEVER fetch it: {why}"
                  + ("" if info["complete"] else f"; checkpoint complete=false ({len(info['replan_required'])} replan_required)")
                  + ("; a probe job is queued/running (a re-probe is under way)" if job_active else
                     "; no probe job — resume the record (`run --resume --replan`) or rewrite its verdict (`serving_probe.py reverdict`)"))
        return MENU_STALE, detail, info
    if progress is None and not info["menu_present"]:
        # a carrier that died between its attempt record and the driver's first checkpoint:
        # without this the record read NOT_STARTED, the front door retried the unsuffixed unit and the funnel skipped it as
        # already submitted — stuck forever. With a job active it is simply RUNNING (the checkpoint is not written yet).
        attempt = _read_json(Path(record_dir) / ATTEMPT_FILE)
        plan_present = (Path(record_dir) / PLAN_FILE).is_file()
        result_present = (Path(record_dir) / RESULT_FILE).is_file()
        info["attempt"] = attempt if isinstance(attempt, dict) and not attempt.get("unparseable") else None
        info["plan_present"] = plan_present
        if (attempt is not None or plan_present or result_present) and not job_active:
            a = info["attempt"] or {}
            who = (f"attempt {a.get('attempt', '?')}/{a.get('max_attempts', '?')}, job {a.get('job_id') or '?'} ({a.get('job_name') or '?'}) on "
                   f"{a.get('host') or '?'}, started {a.get('written_utc') or '?'}, sha {str(a.get('sha') or '?')[:12]}"
                   if a else f"no {ATTEMPT_FILE}; " + ", ".join(n for n, p in ((PLAN_FILE, plan_present), (RESULT_FILE, result_present)) if p) + " present")
            nxt = ("a plan exists — resume the record (the carrier passes --resume --replan)" if plan_present
                   else "no plan exists — a FRESH start is required (nothing was planned)")
            return FAILED_START, (f"the carrier started ({who}) but the driver never wrote {PROGRESS_FILE} and no probe job is active: "
                                  f"the attempt died before its first checkpoint (venv bootstrap / import / sha-pin failure — read its log); {nxt}"), info
    state, detail = probe_state(progress, allowed_present=False, job_active=job_active)
    if state == VERDICT_REFUSED:
        # a refused verdict names WHY (its status and the pairs with no reference evidence) and — the menu being absent — the parity
        # record it read, with the same at-rest facts the PARITY_STALE remedy needs (the verdict stands in for the menu's stamp): the
        # front door's --reopen routes `reverdict --parity-record <dir>` for an `uncertified` / reference_missing verdict
        verdict = _read_json(Path(record_dir) / VERDICT_FILE)
        if isinstance(verdict, dict) and not verdict.get("unparseable"):
            info["verdict_status"] = verdict.get("status")
            info["unreferenced_pairs"] = list(verdict.get("unreferenced_pairs") or [])
            if isinstance(verdict.get(PARITY_RECORD_KEY), dict):
                info.update(verify_parity_record(record_dir, verdict, verdict))
    if info["interrupted_reopen"]:
        ir = info["interrupted_reopen"]
        detail += (f"; a revocation receipt of this record is IN PROGRESS (minted {ir.get('minted_utc') or '?'}, consumed {ir.get('started_utc') or '?'} for "
                   f"{ir.get('what') or '?'}): the run that holds this record consumed it and finishes it at its completing write; a resumed attempt "
                   "claims it — never a new revocation")
    return state, detail, info


def probe_state(progress: Optional[dict], *, allowed_present: bool, job_active: bool) -> tuple[str, str]:
    """``(state, detail)`` from the checkpoint (None when absent), whether ``allowed_pairs.json`` exists, and
    whether a probe job of this search is queued/running."""
    if allowed_present:
        # menu presence alone is no longer READY: the checkpoint-only mode cannot verify the menu's
        # result generation against the record — the front door runs `--record <dir>` for that
        return MENU_UNVERIFIED, (f"{MENU_FILE} present but its result generation cannot be verified from the checkpoint alone "
                                 f"— run `hpo_probe_status.py --record <dir>`; menu presence never means READY")
    if not isinstance(progress, dict):
        if job_active:
            return RUNNING, "no probe_progress.json yet; a probe job is queued/running"
        return NOT_STARTED, "no probe_progress.json and no probe job"
    n_done = int(progress.get("n_done") or 0)
    planned = int(progress.get("plan_launches") or 0)
    outstanding = progress.get("outstanding") or []
    decision = progress.get("decision_outstanding") or []
    stopped = progress.get("stopped_reason")
    if progress.get("complete"):
        return VERDICT_REFUSED, (f"probe_progress.json complete ({n_done}/{planned} launches) but no allowed_pairs.json: "
                                 "the verdict refused certification — read probe_verdict.json")
    detail = (f"{n_done}/{planned} launches done, {len(outstanding)} outstanding "
              f"({len(decision)} decision-relevant: {', '.join(str(x) for x in decision[:6])}"
              f"{'…' if len(decision) > 6 else ''}), stopped: {stopped or 'unknown'}")
    hold = progress.get("hold")
    if hold:
        # a results-rewriting phase (the reference-only re-run) holds the record; the resume is ITS resume
        a = progress.get("attempt") or {}
        detail += (f"; held by {hold} (attempt started {a.get('started_utc') or '?'}, job {a.get('job_id') or '?'}"
                   + (f", interrupted {a.get('interrupted_utc')}" if a.get("interrupted_utc") else "") + ") — "
                   + ("`run --resume --reference-only` resumes it (no new receipt: the record is not complete)" if a.get("reference_only") else "resume it"))
    replan = progress.get("replan_required") or []
    if replan:
        # start-time prefill OOMs above the record's measured batched-token budget — re-derived and re-run by
        # `run --resume --replan` before any verdict is read; the checkpoint is not complete while they stand
        detail += (f"; {len(replan)} launch(es) need `run --resume --replan` (prefill above the measured budget): "
                   f"{', '.join(str(x) for x in replan[:6])}{'…' if len(replan) > 6 else ''}")
    if job_active:
        return INCOMPLETE_RUNNING, detail + " — the probe's successor job continues the SAME record"
    return INCOMPLETE_STALLED, detail + " — NO successor job is queued (attempt cap or STOP sentinel)"


def main(argv=None, stdin=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job-active", default="0", help="1 when a probe job of this search is queued/running")
    ap.add_argument("--allowed-present", default="0", help="1 when allowed_pairs.json exists on the cluster")
    ap.add_argument("--record", help="the record dir (probe_progress.json, probe_verdict.json, allowed_pairs.json, launches/): "
                                     "the verifying mode — prints STATE, detail and a JSON info object; the ONLY mode that can say READY")
    a = ap.parse_args(argv)
    if a.record:
        state, detail, info = probe_state_record(Path(a.record), job_active=a.job_active == "1")
        print(f"{state}\t{detail}\t{json.dumps(info, sort_keys=True)}")
        return 0
    raw = (stdin if stdin is not None else sys.stdin).read()
    progress: Optional[dict] = None
    if raw.strip():
        try:
            doc = json.loads(raw)
            progress = doc if isinstance(doc, dict) else {"unparseable": True}
        except ValueError:
            progress = {"unparseable": True, "complete": False}
    state, detail = probe_state(progress, allowed_present=a.allowed_present == "1", job_active=a.job_active == "1")
    print(f"{state}\t{detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
