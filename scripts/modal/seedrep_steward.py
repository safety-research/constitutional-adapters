#!/usr/bin/env python
"""seedrep_steward.py -- the wave's RETRY STEWARD: ONE bounded, single-owner process on the grading host
that polls the CERTIFIED wave's terminal calls and invokes the retry policy (`seedrep_served.py --reattempt`, itself gated by the
launch certificate and idempotent over the wave's durable `reattempts` Dict) at most ONCE per failed call, stops when the wave is DONE
(every required accepted key covered) or when everything still uncovered is HELD, and dies at its own deadline. It is launched WITH the
production fan-out as a `systemd-run --user` transient unit (the EM / judge-unit idiom: `--collect`, `--setenv=PATH`, `-p WorkingDirectory`,
`-p RuntimeMaxSec`), so it survives the launching shell and the terminal that ran `--go` / `--legs`.

    seedrep_steward.py launch --rs RS --plan RS/launch_plan.json --certificate RS/launch_certificate.json --sha <W> --app <app name> \\
        [--pool-urls RS/pool_urls.json] [--poll 120] [--deadline-h 36]                    # the systemd-run unit (called by the app after fan-out)
    seedrep_steward.py run    --rs RS --plan ... --certificate ... --pass-cmd '<json argv>' [--poll 120] [--deadline-h 36]   # the loop (inside the unit)

Events, one line each: STEWARD_STARTED, STEWARD_PASS (a policy pass invoked, naming the failed calls that triggered it), STEWARD_DONE (WAVE_DONE),
STEWARD_STOP (HOLD: nothing uncovered is retryable | certificate: the certificate stopped validating), STEWARD_TIMEOUT, STEWARD_ALREADY_OWNED.
Single owner: the unit name `seedrep-steward-<W7>-<wave8>` (systemd refuses a second) AND an `flock` on `RS/steward.lock` (a second process exits
at once). The pass command is the ONLY thing that spawns; the steward itself never touches Modal state except to read it.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seedrep_launch_plan as lp  # noqa: E402

EVENTS = ("STEWARD_STARTED", "STEWARD_FANOUT", "STEWARD_MARKS", "STEWARD_PASS", "STEWARD_DONE", "STEWARD_STOP", "STEWARD_TIMEOUT", "STEWARD_ALREADY_OWNED")
LOCK_FILE = "steward.lock"
LOG_DIR = "logs"
DEFAULT_POLL_S = 120
DEFAULT_DEADLINE_H = 36.0                 # > 2 x the longest leg timeout (16 h) + one re-attempt; the unit's RuntimeMaxSec is the same bound
MIN_POLL_S, MAX_POLL_S = 30, 300


def unit_name(W7: str, wave_uuid: str) -> str:
    return f"seedrep-steward-{W7}-{lp.wave_uuid8(wave_uuid)}"


def event(name: str, payload: Mapping[str, Any]) -> str:
    assert name in EVENTS, name
    return f"{name} " + json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)


def pass_command(*, python: str, app_script: Path, app_name: str, sha: str, plan: Path, rs: Path, certificate: Path,
                 pool_urls: Optional[Path] = None) -> List[str]:
    """The ONE retry pass: `modal run --detach -n <app> seedrep_served.py --sha W --plan P --rs RS --reattempt --certificate C [--pool-urls]` --
    the same command the operator would type; `--detach` keeps the spawned successor alive after the pass exits."""
    argv = [python, "-m", "modal", "run", "--detach", "-n", app_name, str(app_script), "--sha", sha, "--plan", str(plan), "--rs", str(rs),
            "--reattempt", "--certificate", str(certificate)]
    if pool_urls:
        argv += ["--pool-urls", str(pool_urls)]
    return argv


def fanout_command(*, python: str, app_script: Path, app_name: str, sha: str, plan: Path, rs: Path, certificate: Path, start_token: str,
                   pool_urls: Optional[Path] = None) -> List[str]:
    """The FAN-OUT pass: `modal run --detach -n <app> seedrep_served.py --sha W --plan P --rs RS --fanout --start-token T
    --certificate C [--pool-urls]` -- the app spawns every row of `RS/production_launch.json` that carries no call yet, idempotently, under the
    steward that the intent names (the token) and only while a steward owns the wave."""
    argv = [python, "-m", "modal", "run", "--detach", "-n", app_name, str(app_script), "--sha", sha, "--plan", str(plan), "--rs", str(rs),
            "--fanout", "--start-token", str(start_token), "--certificate", str(certificate)]
    if pool_urls:
        argv += ["--pool-urls", str(pool_urls)]
    return argv


def new_start_token() -> str:
    import secrets
    return secrets.token_hex(8)


def steward_argv(*, unit: str, log: Path, tree: Path, python: str, rs: Path, plan: Path, certificate: Path, pass_cmd: Sequence[str],
                 poll_s: int, deadline_h: float, path_env: Optional[str] = None, start_token: str = "", fanout_cmd: Optional[Sequence[str]] = None) -> List[str]:
    """The `systemd-run --user` invocation (the repo's judge / grade-unit idiom): `--collect --quiet`, PATH forwarded with ~/.local/bin ahead,
    the tree as the working directory, `RuntimeMaxSec` = the steward's own deadline (the unit dies even if the loop hangs), the loop appended
    to its log."""
    deadline_s = int(float(deadline_h) * 3600)
    inner = [python, str(Path(__file__).resolve()), "run", "--rs", str(rs), "--plan", str(plan), "--certificate", str(certificate),
             "--pass-cmd", json.dumps(list(pass_cmd)), "--poll", str(int(poll_s)), "--deadline-h", str(float(deadline_h))]
    if start_token:
        inner += ["--start-token", str(start_token)]
    if fanout_cmd:
        inner += ["--fanout-cmd", json.dumps(list(fanout_cmd))]
    path_value = path_env if path_env is not None else f"{Path.home()}/.local/bin:{os.environ.get('PATH', '')}"
    return ["systemd-run", "--user", f"--unit={unit}", "--collect", "--quiet", f"--setenv=PATH={path_value}", "-p", f"WorkingDirectory={tree}",
            "-p", f"RuntimeMaxSec={deadline_s}", "/bin/bash", "-c", 'exec "$@" >> "$0" 2>&1', str(log), *inner]


def launch_steward(*, rs: Path, plan: Mapping[str, Any], plan_path: Path, certificate: Path, app_name: str, sha: str, tree: Path,
                   app_script: Path, python: Optional[str] = None, pool_urls: Optional[Path] = None, poll_s: int = DEFAULT_POLL_S,
                   deadline_h: float = DEFAULT_DEADLINE_H, run: Callable[..., Any] = subprocess.run, start_token: str = "") -> Dict[str, Any]:
    """Start the steward unit for THIS certified wave (the app's launcher calls it BEFORE any spawn; the steward performs the fan-out from
    `RS/production_launch.json`). `systemd-run` is resolved on PATH (a test plants a fake); a unit of this name already running makes systemd
    refuse -- reported, never a second owner. ``start_token`` (the intent's) rides the unit's argv and comes back in STEWARD_STARTED. Returns
    `{unit, log, argv, rc, launched, start_token}`."""
    py = python or sys.executable
    unit = unit_name(plan["W7"], plan["wave_uuid"])
    log = Path(rs) / LOG_DIR / f"{unit}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = pass_command(python=py, app_script=Path(app_script), app_name=app_name, sha=sha, plan=Path(plan_path), rs=Path(rs), certificate=Path(certificate),
                       pool_urls=pool_urls)
    fan = (fanout_command(python=py, app_script=Path(app_script), app_name=app_name, sha=sha, plan=Path(plan_path), rs=Path(rs), certificate=Path(certificate),
                          start_token=start_token, pool_urls=pool_urls) if start_token else None)
    argv = steward_argv(unit=unit, log=log, tree=Path(tree), python=py, rs=Path(rs), plan=Path(plan_path), certificate=Path(certificate), pass_cmd=cmd,
                        poll_s=poll_s, deadline_h=deadline_h, start_token=start_token, fanout_cmd=fan)
    if shutil.which("systemd-run") is None:
        return {"unit": unit, "log": str(log), "argv": argv, "rc": None, "launched": False, "why": "systemd-run is not on PATH", "start_token": start_token}
    r = run(argv, capture_output=True, text=True)
    out = {"unit": unit, "log": str(log), "argv": argv, "rc": int(r.returncode), "launched": r.returncode == 0, "start_token": start_token}
    if r.returncode != 0:
        out["why"] = (getattr(r, "stderr", "") or getattr(r, "stdout", "") or "").strip()[-400:] or "systemd-run refused (a unit of this name is already running?)"
    return out


def wait_started(log: Path, *, token: str, timeout_s: float = 90.0, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Any] = time.sleep) -> Optional[Dict[str, Any]]:
    """The launcher's hand-off: block until the unit's APPEND-ONLY log carries a `STEWARD_STARTED {...}` whose
    `start_token` is THIS invocation's ``token`` (a historical STARTED line of an earlier steward is never an acknowledgement), or a
    `STEWARD_ALREADY_OWNED` / `STEWARD_STOP` carrying the token; None on the deadline or a refusal -- the launcher then hands nothing over."""
    if not token:
        raise ValueError("wait_started needs the invocation's start token (the intent's steward.start_token)")
    t0 = clock()
    while True:
        text = Path(log).read_text() if Path(log).is_file() else ""
        for line in text.splitlines():
            if not line.startswith(("STEWARD_STARTED ", "STEWARD_ALREADY_OWNED ", "STEWARD_STOP ")):
                continue
            try:
                payload = json.loads(line.split(" ", 1)[1])
            except ValueError:
                continue
            if payload.get("start_token") != token:
                continue                                                  # another invocation's line (stale / a sibling); not ours
            return payload if line.startswith("STEWARD_STARTED ") else None
        if clock() - t0 >= timeout_s:
            return None
        sleep(1.0)


class StewardState:
    def __init__(self) -> None:
        self.passed_for: set = set()          # failed call ids a policy pass was already invoked for (at most ONE pass per failed call)
        self.n_passes = 0
        self.n_fanouts = 0                    # fan-out passes run from the intent
        self.n_mark_passes = 0                # per-tick mark reconciliations run (unbounded: until every claim has an observed winner)
        self.closed_queues: set = set()       # ladder roots whose CLOSED.json the reconciler has seen (never reconciled again)


def plan_queue_specs(L, plan: Mapping[str, Any]) -> List[Any]:
    """`QueueSpec`s for every ladder queue the plan declares (`plan["queues"]`: producers = the feeder rows' shards), the Petri re-run
    proxy excluded (its "queue" is the context-overflow ledger, not a ladder root)."""
    out = []
    for q in plan.get("queues") or []:
        if q["cap_next"] == lp.STAGE_PETRI_RERUN or not q.get("producers"):
            continue
        kind = lp.LADDER_KIND_OF_LEG.get(q["leg"])
        if kind is None:
            continue
        shards = [str(p["shard"]) for p in q["producers"]]
        out.append(L.QueueSpec.build(plan["W7"], plan["wave_uuid"], q["leg"], q["engine"], kind, int(q["cap_next"]), shards))
    return out


def reconcile_plan_queues(L, plan: Mapping[str, Any], *, vol, accepted_kv, state: StewardState, claims_kv=None, results_kv=None, lease_state=None,
                          log: Callable[[str], Any] = print) -> Dict[str, Any]:
    """ONE reconciler pass over every not-yet-closed ladder queue of the plan (the grading host closure loop, run from the steward's tick):
    `reconcile_queue` with `attempt_root_of` = THIS app's producer-root spelling (`lp.producer_attempt_root_of`; its default spells
    the shard bare and would verify nothing). Returns {closed: [...], open: {root: producer states}}."""
    default_root_of = lp.producer_attempt_root_of(plan["results_root"])
    import seedrep_carry_over as co  # noqa: E402
    out: Dict[str, Any] = {"closed": [], "open": {}}
    for spec in plan_queue_specs(L, plan):
        if spec.ladder_root in state.closed_queues:
            continue
        # fj4 (fj3 replay): a CARRIED producer's re-keyed inventory + stub live under `<ladder root>/carried/<shard>__<aid>/`
        # (`seedrep_carry_over.carried_producer_root`) -- resolved there when the stub exists, so a queue mixing a carried and a fresh producer
        # closes once the fresh one is accepted; every other producer resolves through the app's own spelling as before
        root_of = co.carry_aware_root_of(L, vol, spec, default_root_of)
        try:
            status = L.reconcile_queue(vol, accepted_kv, spec, claims_kv=claims_kv, results_kv=results_kv, lease_state=lease_state, attempt_root_of=root_of)
        except Exception as e:  # noqa: BLE001 -- a refusal on one queue never stops the others; it is logged by name
            log(f"[steward] reconcile {spec.ladder_root}: {type(e).__name__}: {e}")
            out["open"][spec.ladder_root] = {"error": f"{type(e).__name__}: {e}"}
            continue
        if L.read_closed(vol, spec) is not None:
            state.closed_queues.add(spec.ladder_root)
            out["closed"].append(spec.ladder_root)
        else:
            out["open"][spec.ladder_root] = {p: s.get("state") for p, s in (status.get("producers") or {}).items()}
    return out


def required_keys_all_legs(plan: Mapping[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for leg in lp.STANDUP_LEGS:
        try:
            out.update(lp.required_accepted_keys(plan, leg, "all"))
        except ValueError:
            continue
    return out


def steward_tick(*, plan: Mapping[str, Any], calls: List[dict], snapshot: Mapping[str, Any], holds: Mapping[str, Mapping[str, Any]],
                 state: StewardState, required: Mapping[str, str], carried: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """ONE poll's decision, pure: `{stop: None | 'done' | 'hold', invoke: [failed call ids], covered: n, required: n, held: [...], carried: n}`.
    ``carried`` (fj3): `{accepted_key: attempt_id}` of the CARRIED attempts (`RS/carry_over.json`) -- a carried row has no call
    of this wave, so it is covered by its provenance record, never by a call; nothing here re-runs or relabels it."""
    covered = {c["accepted_key"] for c in calls if (snapshot.get(c["call_id"]) or {}).get("accepted") and c.get("accepted_key") in required}
    carried_covered = {k for k in (carried or {}) if k in required}
    covered |= carried_covered
    missing = sorted(set(required) - covered)
    if required and not missing:
        return {"stop": "done", "invoke": [], "covered": len(covered), "required": len(required), "held": sorted(holds), "carried": len(carried_covered)}
    held_sids = {sid for sid, h in holds.items() if h.get("held")}
    missing_sids = {required[k] for k in missing}
    if missing and missing_sids <= held_sids:
        return {"stop": "hold", "invoke": [], "covered": len(covered), "required": len(required), "held": sorted(held_sids), "carried": len(carried_covered)}
    invoke = []
    for c in calls:
        cid = c["call_id"]
        s = snapshot.get(cid) or {}
        if not s.get("done") or s.get("accepted") or cid in state.passed_for or c["standup_id"] in held_sids:
            continue
        summ = s.get("summary") if isinstance(s.get("summary"), Mapping) else {}
        status = summ.get("status")
        if status in ("REATTEMPT_DUPLICATE", lp.FANOUT_DUPLICATE):
            continue                                                  # a duplicate container that exited at its claim is not the row's execution, never a failure
        failed = bool(s.get("error")) or (status not in (None, "DONE")) or ("rc=1" in str(s.get("result_line") or ""))
        if failed:
            invoke.append(cid)
    return {"stop": None, "invoke": sorted(invoke), "covered": len(covered), "required": len(required), "held": sorted(held_sids), "carried": len(carried_covered)}


def fanout_due(intent_path: Optional[Path]) -> List[str]:
    """The intent's rows with no call yet (the steward's job: the initial fan-out and every row a crashed fan-out pass left behind)."""
    if intent_path is None:
        return []
    intent = lp.read_production_intent(Path(intent_path))
    return lp.production_intent_unspawned(intent) if intent else []


def marks_pending(intent_path: Optional[Path]) -> List[str]:
    """Claimed rows whose executing container's mark has not been observed yet (the steward reconciles them EVERY tick, in-process, until every
    claim has an observed winner or the loop ends on DONE / HOLD / the deadline -- never an arbitrary pass count)."""
    if intent_path is None:
        return []
    intent = lp.read_production_intent(Path(intent_path))
    return (lp.production_intent_marks_pending(intent) + lp.production_intent_claimed_unspawned(intent)) if intent else []


def run_steward(*, rs: Path, plan: Mapping[str, Any], certificate_path: Path, run_pass: Callable[[], int], poll_s: int = DEFAULT_POLL_S,
                deadline_s: float = DEFAULT_DEADLINE_H * 3600, backend=None, registry_sha256: Optional[str] = None, clock: Callable[[], float] = time.time,
                sleep: Callable[[float], Any] = time.sleep, load_calls: Optional[Callable[[], List[dict]]] = None,
                emit: Callable[[str], Any] = lambda line: print(line, flush=True), reconcile: Optional[Callable[[StewardState], Dict[str, Any]]] = None,
                start_token: str = "", intent_path: Optional[Path] = None, run_fanout: Optional[Callable[[], int]] = None,
                reconcile_marks: Optional[Callable[[], Dict[str, Any]]] = None) -> int:
    """The loop. Exit 0 = DONE or HOLD-stop or another owner holds the lock; 2 = the certificate stopped validating; 3 = the deadline.
    ``reconcile`` (default: :func:`reconcile_plan_queues` over the Volume API + the wave's Dicts when that module is on the tree) runs once per
    tick BEFORE the retry decision -- the queues close from here, so a rung consumer never waits on a reconciler nobody runs.
    The steward OWNS the fan-out -- right after STEWARD_STARTED and on every tick, every row of ``intent_path``
    (`RS/production_launch.json`) with no call yet is spawned by ``run_fanout`` (the app's `--fanout` pass, idempotent under the intent lock);
    ``start_token`` is echoed in STEWARD_STARTED / ALREADY_OWNED / STOP so the launcher correlates its own invocation."""
    rs = Path(rs)
    rs.mkdir(parents=True, exist_ok=True)
    ident = {"wave_uuid": plan["wave_uuid"], "start_token": start_token}
    lock_fd = os.open(rs / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        emit(event("STEWARD_ALREADY_OWNED", {"lock": str(rs / LOCK_FILE), **ident}))
        os.close(lock_fd)
        return 0
    try:
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, f"{os.getpid()} {plan['wave_uuid']} {plan.get('launch_plan_sha')} {start_token}\n".encode())
        t0 = clock()
        state = StewardState()
        required = required_keys_all_legs(plan)
        # fj3: the CARRIED attempts (`RS/carry_over.json`) count as covered by provenance -- they have no call of this wave
        import seedrep_carry_over as co  # noqa: E402
        carry_rec = co.read_carry_over(rs) or {}
        carried = co.carried_accepted_entries(carry_rec)
        # fj7 (B): a row the carry REFUSED although a completed prior attempt exists is HELD by name -- surfaced as `carry_refused_hold`, never re-attempted
        carry_holds = {sid: {"held": True, "reason": co.CARRY_REFUSED_HOLD, **h} for sid, h in co.carry_refused_holds(carry_rec).items()}
        emit(event("STEWARD_STARTED", {"launch_plan_sha": plan.get("launch_plan_sha"), "required_keys": len(required), "carried_keys": len(carried),
                                       "carry_refused_holds": sorted(carry_holds), "poll_s": poll_s,
                                       "deadline_s": deadline_s, "pid": os.getpid(), "intent": (str(intent_path) if intent_path else None), **ident}))

        def fanout_if_due() -> None:
            due = fanout_due(intent_path)
            if due and run_fanout is not None:
                rc = run_fanout()
                state.n_fanouts += 1
                emit(event("STEWARD_FANOUT", {"n": state.n_fanouts, "due_before": len(due), "rows": due[:8], "rc": rc, "due_after": len(fanout_due(intent_path)), **ident}))
            elif due:
                emit(event("STEWARD_FANOUT", {"n": state.n_fanouts, "due_before": len(due), "rows": due[:8], "rc": None, "why": "no fan-out command", **ident}))
            # every claimed row's executing WINNER is looked for on EVERY tick (in-process, the wave's Dict) until observed and in
            # calls.jsonl -- a winner that starts late (after the recorded re-spawn exited FANOUT_DUPLICATE) is adopted whenever it appears
            pending = marks_pending(intent_path)
            if pending and reconcile_marks is not None:
                res = reconcile_marks()
                state.n_mark_passes += 1
                emit(event("STEWARD_MARKS", {"n": state.n_mark_passes, "pending_before": len(pending), "observed": list(res.get("observed") or [])[:8],
                                             "adopted": list(res.get("adopted") or [])[:8], "pending_after": len(marks_pending(intent_path)), **ident}))
        fanout_if_due()
        if backend is None:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import seedrep_watch as sw  # noqa: E402
            backend = sw.ModalBackend(plan)
            import seedrep_served as ss  # noqa: E402
            load_calls = load_calls or (lambda: ss.wave_calls(rs, plan))
            if reconcile is None:
                reconcile = _volume_reconciler(plan, emit)
            if reconcile_marks is None and intent_path is not None:
                import modal  # noqa: PLC0415
                led = modal.Dict.from_name(plan["dicts"]["reattempts"], create_if_missing=True)
                reconcile_marks = lambda: ss.reconcile_fanout_marks_file(rs, plan, Path(intent_path), led)  # noqa: E731
        first = True
        while True:
            if not first:
                fanout_if_due()                                           # a fan-out pass that crashed mid-way leaves rows due: they are spawned here, never by a launcher
            first = False
            if reconcile is not None:
                rec = reconcile(state)
                if rec.get("closed"):
                    emit(event("STEWARD_PASS", {"n": state.n_passes, "queues_closed": rec["closed"], "wave_uuid": plan["wave_uuid"]}))
            cert = json.loads(Path(certificate_path).read_text()) if Path(certificate_path).is_file() else None
            probs = lp.go_preconditions(plan, certificate=cert, registry_sha256=registry_sha256)
            if probs:
                emit(event("STEWARD_STOP", {"why": "certificate", "problems": probs[:5], **ident}))
                return 2
            calls = (load_calls or (lambda: []))()
            snap = backend.snapshot(calls) if calls else {}
            holds = backend.holds(calls) if (calls and hasattr(backend, "holds")) else {}
            holds = {**carry_holds, **holds}                          # fj7 (B): the carry's holds stand beside the re-attempt ledger's
            tick = steward_tick(plan=plan, calls=calls, snapshot=snap, holds=holds, state=state, required=required, carried=carried)
            if tick["stop"] == "done":
                emit(event("STEWARD_DONE", {"why": "WAVE_DONE", "required": tick["required"], "accepted": tick["covered"], "passes": state.n_passes, "wave_uuid": plan["wave_uuid"]}))
                return 0
            if tick["stop"] == "hold":
                emit(event("STEWARD_STOP", {"why": "HOLD", "held": tick["held"], "required": tick["required"], "accepted": tick["covered"], "passes": state.n_passes,
                                            "wave_uuid": plan["wave_uuid"]}))
                return 0
            if tick["invoke"]:
                rc = run_pass()
                state.n_passes += 1
                state.passed_for.update(tick["invoke"])
                emit(event("STEWARD_PASS", {"n": state.n_passes, "failed_calls": tick["invoke"], "rc": rc, "wave_uuid": plan["wave_uuid"]}))
            if clock() - t0 >= deadline_s:
                emit(event("STEWARD_TIMEOUT", {"deadline_s": deadline_s, "required": tick["required"], "accepted": tick["covered"], "passes": state.n_passes,
                                               "wave_uuid": plan["wave_uuid"]}))
                return 3
            sleep(poll_s)
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _volume_reconciler(plan: Mapping[str, Any], emit: Callable[[str], Any]) -> Optional[Callable[[StewardState], Dict[str, Any]]]:
    """The production reconciler: the ladder module from THIS tree (`scripts/modal/seedrep_ladder.py`), the Volume API, the wave's accepted /
    claims / results Dicts, `modal_lease_state` for stale claims; None (no reconciling) when the tree lacks the module -- logged once."""
    lib = Path(__file__).resolve().parent / "seedrep_ladder.py"
    if not lib.is_file():
        emit(event("STEWARD_PASS", {"n": 0, "reconcile": f"disabled: {lib} is not on this tree (WS2-E's protocol module)", "wave_uuid": plan["wave_uuid"]}))
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("seedrep_ladder", lib)
    L = importlib.util.module_from_spec(spec)
    sys.modules["seedrep_ladder"] = L
    spec.loader.exec_module(L)
    vol = L.RemoteVolume.from_name(lp.MODAL_VOLUME_NAME)
    kv = {p: L.ModalDictKV.from_name(plan["dicts"][p]) for p in ("accepted", "claims", "results")}
    return lambda state: reconcile_plan_queues(L, plan, vol=vol, accepted_kv=kv["accepted"], claims_kv=kv["claims"], results_kv=kv["results"],
                                               lease_state=L.modal_lease_state, state=state, log=lambda s: emit(s))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("launch", "run"):
        c = sub.add_parser(name)
        c.add_argument("--rs", type=Path, required=True)
        c.add_argument("--plan", type=Path, required=True, help="RS/launch_plan.json (FINAL)")
        c.add_argument("--certificate", type=Path, required=True, help="RS/launch_certificate.json")
        c.add_argument("--poll", type=int, default=DEFAULT_POLL_S)
        c.add_argument("--deadline-h", type=float, default=DEFAULT_DEADLINE_H)
    c = sub.choices["launch"]
    c.add_argument("--sha", required=True)
    c.add_argument("--app", required=True, help="the wave's app name (seedrep-served-<W7>)")
    c.add_argument("--pool-urls", type=Path, default=None)
    c.add_argument("--tree", type=Path, default=Path(__file__).resolve().parents[2])
    c = sub.choices["run"]
    c.add_argument("--pass-cmd", required=True, help="JSON argv of the retry pass (modal run --detach ... --reattempt --certificate ...)")
    c.add_argument("--start-token", default="", help="this invocation's start token (the intent's steward.start_token; echoed in STEWARD_STARTED)")
    c.add_argument("--fanout-cmd", default=None, help="JSON argv of the fan-out pass (modal run --detach ... --fanout --start-token T --certificate ...)")
    a = ap.parse_args(argv)
    if a.poll < MIN_POLL_S or a.poll > MAX_POLL_S:
        raise SystemExit(f"--poll must be within [{MIN_POLL_S}, {MAX_POLL_S}] s")
    plan = json.loads(a.plan.read_text())
    if a.cmd == "launch":
        out = launch_steward(rs=a.rs, plan=plan, plan_path=a.plan, certificate=a.certificate, app_name=a.app, sha=a.sha, tree=a.tree,
                             app_script=Path(__file__).resolve().with_name("seedrep_served.py"), pool_urls=a.pool_urls, poll_s=a.poll, deadline_h=a.deadline_h)
        print(json.dumps(out, indent=1))
        return 0 if out["launched"] else 1
    cmd = json.loads(a.pass_cmd)
    fan = json.loads(a.fanout_cmd) if a.fanout_cmd else None
    reg = a.tree / lp.DECODE_REGISTRY_REL if hasattr(a, "tree") else Path(__file__).resolve().parents[2] / lp.DECODE_REGISTRY_REL
    return run_steward(rs=a.rs, plan=plan, certificate_path=a.certificate, run_pass=lambda: subprocess.run(cmd).returncode, poll_s=a.poll,
                       deadline_s=a.deadline_h * 3600, registry_sha256=(lp.sha256_file(reg) if reg.is_file() else None), start_token=a.start_token,
                       intent_path=a.rs / lp.PRODUCTION_LAUNCH_FILE, run_fanout=(lambda: subprocess.run(fan).returncode) if fan else None)


if __name__ == "__main__":
    sys.exit(main())
