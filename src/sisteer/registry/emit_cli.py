"""The producer-side command line — STDLIB ONLY, runnable on a compute node BEFORE ``uv sync``::

    PYTHONPATH=<checkout>/src python3 -S -m sisteer.registry.emit_cli <command> …

Commands (every one prints what it did; ``state`` and ``push`` ALWAYS exit 0 — a lost record must
never fail a job; ``register`` exits 2 when the plan is refused, so a submit wrapper can refuse to
submit):

``register --plan FILE [--spool DIR] [--job-id ID] [--job-name N] [--sha-pin SHA] [--pending-dir DIR] [--check-only] [--no-push]``
    Submit-time registration: validate the planned run STRICTLY (an
    eval named without its ``item_set_id`` / ``instrument_id`` is refused naming the field), write
    ``run`` (queued) + ``run_state queued`` (+ a ``run_patch`` binding ``cluster_job_id`` /
    ``job_name``), copy the plan with its ``run_id`` to ``<pending-dir>/<job-id>.json`` (the job
    finds its own identity by ``$SLURM_JOB_ID``), push. Prints the run id.

``register --from-config CFG --model-key KEY --evals E1,E2 --serving-path P --campaign C --job-id ID --pending-dir DIR [--job-name N] [--roster-selection main] [--roster FILE] [--pins FILE] [--seeds DIR] [--code-sha SHA] [--substrate S] [--spool DIR] [--no-push] [--producer-key K] [--token-cap N --rung-index I --ladder L] [--pass-label P] [--trial-index T] [--pairs E:O,…] [--positions-mode M]``
    JOB-START self-registration (the default path since no submit wrapper is needed): the job builds
    its own plans from the config it resolved + the registry seeds (``sisteer.registry.jobplan``: one
    run per declared eval x declared object, the canonical seeded instrument per eval, object ids from
    the roster + pins / the SI texts), writes every ``run`` (lifecycle ``running``,
    ``extra.registered_at = job_start``), stores the pending file with a ``runs`` list — each entry
    carrying its routing facts (object, the eval's producer spellings) so ``hooks.identity_from_env``
    routes every store row onto its run — and pushes. Idempotent: an existing pending file for the job
    is kept and nothing is re-registered. Needs the node's plain ``python3`` (PyYAML), never ``-S``.
    Exits 0 with the pending file written (or already present); 2 when the registration is REFUSED — no spool, a missing fact, the
    planner's refusal (an unknown model key, an inconsistent rung, a ``--pairs`` pair outside the declared evals x objects), every pair
    skipped, a refused substrate — and 3 on an unexpected failure. static-reg-fix-2: a job whose registration is refused must
    not run, so ``common.sh f22_registry_register`` treats a missing pending file as fatal (exit 1 before any venv / standup); ``--pairs``
    registers ONLY the (eval, object) pairs a phase's per-pair gate selected (``jobplan.selected_pairs``); ``--positions-mode`` names the
    applied-positions mode the job will serve, and the instrument per eval is then the seeded one carrying that mode
    (``jobplan.canonical_instrument``; unset: the canonical instrument, as before).

``state --state running|generated|failed|cancelled (--run-id ID | --pending-dir DIR --job ID) [--node H] [--gpu-count N] [--gpu-type T] [--scheduler-state S] [--evidence-kind K] [--evidence-uri U] [--late-register FILE] [--final]``
    Job-side lifecycle: a ``run_state`` record from the job's own environment (+ a ``run_patch``
    with node / job / GPU facts on ``running``), then a push (``--final`` forces it past the
    debounce), for EVERY run the pending file names (its ``runs`` list, else its one ``run_id``). GPU
    facts not given on the command line are captured once per job from its environment:
    ``SLURM_GPUS_ON_NODE`` / ``SLURM_JOB_GPUS`` / ``SLURM_STEP_GPUS`` /
    ``CUDA_VISIBLE_DEVICES`` for the count, ``nvidia-smi --query-gpu=name`` for the type, and
    ``sacct -X -n -P -o AllocTRES -j <job>`` as the fallback for either; each source is named in the
    ``running`` state's note (``emit.hardware_from_env``), and the facts ride every target run's patch. A
    job with no pending file and ``--late-register <plan.json>`` emits the ``run`` itself first
    (``extra.registered_at = job_start``). No pending file and no plan: one notice, exit 0.

``push (--run-dir DIR | --pending-dir DIR --job ID) [--final]``
    Call the push script on one run's record directory — or on every run directory the job's pending
    file names (debounced unless ``--final``).

Nothing here reads a credential; the push script alone touches the key.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import emit, identity
from .records import utc_now

TAG = "[registry]"
EXIT_REFUSED = 2   # register --from-config: the registration was refused by name (nothing written) — the shell stops the job
EXIT_FAILED = 3    # register --from-config: an unexpected failure (nothing written) — likewise fatal for the job


def _say(msg: str) -> None:
    print(f"{TAG} {msg}", file=sys.stderr, flush=True)


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a JSON object")
    return data


def _plan_of(data: Mapping[str, object]) -> Dict[str, object]:
    """A pending file is ``{"run_id", "plan": {...}}``; a plan file is the plan itself."""
    if isinstance(data.get("plan"), dict):
        return dict(data["plan"])  # type: ignore[arg-type]
    return dict(data)


def _runs_of(data: Mapping[str, object]) -> List[Tuple[str, Dict[str, object]]]:
    """``[(run_id, plan)]`` a pending file names: its ``runs`` list (job-start self-registration: one entry per
    declared eval x object), else its one top-level ``run_id`` + ``plan``."""
    out: List[Tuple[str, Dict[str, object]]] = []
    runs = data.get("runs")
    if isinstance(runs, list):
        for r in runs:
            if isinstance(r, dict) and r.get("run_id"):
                out.append((str(r["run_id"]), _plan_of(r)))
    if not out and data.get("run_id"):
        out.append((str(data["run_id"]), _plan_of(data)))
    return out


def _spool(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    root = emit.spool_root()
    if root:
        return root
    if os.environ.get("REGISTRY_EMIT", "").strip().lower() in ("1", "true", "yes", "on"):
        return os.path.join(os.path.expanduser(emit.HOME_REGISTRY), "records")
    return None


def _push_script() -> Optional[str]:
    return os.environ.get(emit.PUSH_SCRIPT_ENV)


# --------------------------------------------------------------------------------------------
# register
# --------------------------------------------------------------------------------------------


def register(plan_path: str, spool: Optional[str] = None, job_id: Optional[str] = None, job_name: Optional[str] = None,
             pending_dir: Optional[str] = None, check_only: bool = False, submit_sha_pin: Optional[str] = None,
             script: str = "sisteer.registry.emit_cli register", push: bool = True, session: Optional[str] = None) -> int:
    try:
        plan = _plan_of(_load_json(plan_path))
    except (OSError, ValueError) as exc:
        _say(f"register: cannot read plan {plan_path!r}: {exc}")
        return 2
    if submit_sha_pin:
        plan.setdefault("submit_sha_pin", submit_sha_pin)
    if job_id:
        plan.setdefault("submit_evidence_uri", f"slurm://job/{job_id}")
    producer = emit.producer_block(script, session=session)
    try:
        rid = emit.check_planned_run(plan, producer)
    except emit.PlanRefused as exc:
        _say(f"register REFUSED: {exc}")
        _say("register: the harness does not launch an eval whose instrument is not fully resolved; the plan needs "
             + ", ".join(sorted(set(k for k in ("eval_id", "item_set_id", "instrument_id") if not plan.get(k))) or {"the named field"}))
        return 2
    if check_only:
        print(rid)
        return 0
    spool_path = _spool(spool)
    if not spool_path:
        _say("register: no spool resolves (set SISTEER_REGISTRY_SPOOL or --spool); nothing written")
        return 2
    paths = emit.write_planned_run(spool_path, plan, script, session=session, strict=False)
    w = emit.RecordWriter(spool_path, str(plan["campaign"]), rid, producer, push_script=_push_script())
    fields: Dict[str, object] = {}
    if job_id:
        fields["cluster_job_id"] = str(job_id)
    if job_name:
        fields["job_name"] = str(job_name)
    if fields:
        p = w.emit_patch(fields=fields)
        if p:
            paths.append(p)
    if pending_dir and job_id:
        os.makedirs(pending_dir, exist_ok=True)
        pend = os.path.join(pending_dir, f"{job_id}.json")
        tmp = pend + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"run_id": rid, "job_id": str(job_id), "job_name": job_name, "registered_utc": utc_now(),
                       "plan": dict(plan, run_id=rid)}, fh, indent=1, sort_keys=True)
        os.replace(tmp, pend)
        _say(f"register: pending file {pend}")
    if push:
        w.push(final=True)
    for p in paths:
        print(p)                      # the record paths on stdout (the `registry register-run` contract)
    _say(f"register: run_id={rid}")
    return 0


#: The twin-submit election's handles: a job submitted under an ATTEMPT carries
#: ``SISTEER_ATTEMPT_ID`` (+ ``SISTEER_ATTEMPTS_ROOT``) in its environment; the front door's ledger binds ``logical_work_id → job id →
#: run id(s)``, so every run this job registers stamps ``extra.attempt_id`` and ``extra.logical_work_id`` (from ``SISTEER_LOGICAL_WORK_ID``,
#: else the one-line file ``<attempts root>/<attempt id>/logical_work_id`` the submitter writes) and the pending file carries them too.
ATTEMPT_ID_ENV = "SISTEER_ATTEMPT_ID"
ATTEMPTS_ROOT_ENV = "SISTEER_ATTEMPTS_ROOT"
LOGICAL_WORK_ID_ENV = "SISTEER_LOGICAL_WORK_ID"


def attempt_stamp(env: Optional[Mapping[str, str]] = None) -> Dict[str, object]:
    """``{"attempt_id", "logical_work_id", "attempts_root"}`` from the job's environment (only the keys that resolve; {} outside an attempt)."""
    env = os.environ if env is None else env
    out: Dict[str, object] = {}
    aid = (env.get(ATTEMPT_ID_ENV) or "").strip()
    if not aid:
        return out
    out["attempt_id"] = aid
    root = (env.get(ATTEMPTS_ROOT_ENV) or "").strip()
    if root:
        out["attempts_root"] = root
    lw = (env.get(LOGICAL_WORK_ID_ENV) or "").strip()
    if not lw and root:
        try:
            with open(os.path.join(root, aid, "logical_work_id"), "r", encoding="utf-8") as fh:
                lw = fh.read().strip()
        except OSError:
            lw = ""
    if lw:
        out["logical_work_id"] = lw
    return out


def register_from_config(config: str, *, model_key: str, evals: Sequence[str], serving_path: str, campaign: str, job_id: str,
                         pending_dir: str, job_name: Optional[str] = None, roster_selection: str = "main", roster: Optional[str] = None,
                         pins: Optional[str] = None, seeds: Optional[str] = None, code_sha: Optional[str] = None,
                         substrate: Optional[str] = None, spool: Optional[str] = None, push: bool = True,
                         session: Optional[str] = None, script: str = "sisteer.registry.emit_cli register",
                         producer_key: Optional[str] = None, token_cap: Optional[int] = None, token_cap_rung_index: Optional[int] = None,
                         ladder: Optional[Sequence[int]] = None, pass_label: Optional[str] = None, trial_index: Optional[int] = None,
                         pairs: Optional[Sequence[str]] = None, serving_jobs: Optional[Sequence[str]] = None,
                         positions_mode: Optional[str] = None) -> int:
    """Job-start self-registration (module docstring). Exit 0 with the pending file written (or already present); EXIT_REFUSED (2) when the
    registration is refused by name and nothing is written; EXIT_FAILED (3) on an unexpected failure. static-reg-fix-2: the
    pending file is the job's run identity and a job without one must not run — the shell (common.sh f22_registry_register) is fatal on a
    missing pending file, so these exits stop the job before any GPU work. ``pairs``: the gate-selected (eval, object) pairs, the ONLY runs
    registered (jobplan.selected_pairs; a pair outside the declared evals x objects is refused)."""
    pend = os.path.join(pending_dir, f"{job_id}.json")
    if os.path.isfile(pend):
        _say(f"register: pending file {pend} already exists (a submit wrapper registered this job); kept, nothing re-registered")
        print(pend)
        return 0
    spool_path = _spool(spool)
    if not spool_path:
        _say("register: no spool resolves (set SISTEER_REGISTRY_SPOOL or --spool); nothing written (REFUSED, exit 2 — the job must not run unregistered)")
        return EXIT_REFUSED
    from . import jobplan as JP   # lazy: PyYAML + the seeds loader live behind it
    extra: Dict[str, object] = dict(attempt_stamp())
    if serving_jobs is not None:
        # the store's serving jobs the registering job KNOWS at start — its own id and the sibling shard
        # groups' live jobs the funnel handed it (never a store's routing tables) — sorted, de-duplicated; a non-numeric id refuses by name.
        ids = sorted({str(x).strip() for x in serving_jobs if str(x).strip()} | {str(job_id)})
        bad = [x for x in ids if not x.isdigit()]
        if bad:
            _say(f"register: --serving-jobs carries non-numeric job id(s) {bad} — a serving job is a scheduler job id; nothing written (REFUSED, exit 2)")
            return EXIT_REFUSED
        extra["serving_jobs"] = ids
        extra["serving_jobs_source"] = "funnel plan D1918"
    try:
        stamp = extra
        bundle = JP.build_plans(config_path=config, model_key=model_key, evals=list(evals), serving_path=serving_path, campaign=campaign,
                                job_id=str(job_id), job_name=job_name, roster_selection=roster_selection, seeds_dir=seeds, roster_path=roster,
                                pins_path=pins, code_sha=code_sha, substrate=substrate, lifecycle_state="running", extra=stamp or None,
                                producer_key=producer_key, token_cap=token_cap, token_cap_rung_index=token_cap_rung_index, ladder=ladder, pass_label=pass_label,
                                trial_index=trial_index, pairs=pairs, positions_mode=positions_mode)
    except (JP.JobPlanRefused, OSError, ValueError) as exc:
        _say(f"register: self-registration refused ({type(exc).__name__}: {exc}); nothing written (REFUSED, exit 2 — the job must not run unregistered)")
        return EXIT_REFUSED
    except Exception as exc:  # noqa: BLE001 — exit 3: the shell stops the job; PyYAML missing lands here too
        _say(f"register: self-registration failed ({type(exc).__name__}: {exc}); nothing written (FAILED, exit 3 — the job must not run unregistered)")
        return EXIT_FAILED
    for what, why in bundle.skipped:
        _say(f"register: skipped {what}: {why}")
    if not bundle.runs:
        _say("register: no run could be registered for this job (every declared eval x object was skipped, see above); nothing written (REFUSED, exit 2)")
        return EXIT_REFUSED
    producer = emit.producer_block(script, code_sha=code_sha, session=session, substrate=substrate)
    try:
        emit.check_substrate(producer.get("substrate"))    # what every writer below would refuse per record: refused ONCE here, by name
    except emit.SubstrateRefused as exc:
        _say(f"register: self-registration refused ({type(exc).__name__}: {exc}); nothing written (REFUSED, exit 2 — the job must not run unregistered)")
        return EXIT_REFUSED
    paths: List[str] = []
    writers: List[emit.RecordWriter] = []
    # dimension records the job registers beside its runs (jobplan.PlanBundle.dimensions: the padded wave's per-level instrument):
    # written FIRST, one record directory per dimension row, so the harvester folds the parent before the runs that name it
    for rt, payload in bundle.dimensions:
        wd = emit.RecordWriter(spool_path, str(campaign), f"{rt}__{identity.sha16(JP.dimension_id(rt, payload) or rt)}", producer,
                               push_script=_push_script())
        p = wd.write(rt, payload)
        if p:
            paths.append(p)
        writers.append(wd)
    for r in bundle.runs:
        _ids, run_payload, _state = emit.planned_run_records(r.plan, producer)
        run_payload["lifecycle_state"] = "running"
        run_payload.setdefault("started_utc", run_payload["queued_utc"])
        w = emit.RecordWriter(spool_path, str(r.plan["campaign"]), r.run_id, producer, push_script=_push_script())
        p = w.emit_run(run_payload)
        if p:
            paths.append(p)
        writers.append(w)
    os.makedirs(pending_dir, exist_ok=True)
    tmp = pend + ".tmp"
    pending_doc = bundle.as_pending(str(job_id), job_name)
    if stamp:
        pending_doc.update({k: stamp[k] for k in ("attempt_id", "logical_work_id") if k in stamp})   # the election's handles at the top level too
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(pending_doc, fh, indent=1, sort_keys=True)
    os.replace(tmp, pend)
    if push:
        for w in writers:
            w.push()
    for p in paths:
        print(p)
    for n in bundle.notes:
        _say(f"register: {n}")
    if bundle.dimensions:
        _say(f"register: {len(bundle.dimensions)} dimension record(s) registered beside the runs: "
             + ", ".join(f"{rt} {str(JP.dimension_id(rt, payload) or '')[:12]}" for rt, payload in bundle.dimensions))
    _say(f"register: {len(bundle.runs)} run(s) registered at job start for job {job_id}; pending file {pend}")
    return 0


# --------------------------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------------------------

_EVIDENCE_DEFAULT = {"running": "job_log", "generated": "completion_artifact", "failed": "scheduler_state", "cancelled": "scheduler_state"}


def _pending_file(pending_dir: Optional[str], job: Optional[str]) -> Optional[str]:
    if not pending_dir or not job:
        return None
    p = os.path.join(pending_dir, f"{job}.json")
    return p if os.path.isfile(p) else None


def state(state_: str, run_id: Optional[str] = None, pending_dir: Optional[str] = None, job: Optional[str] = None,
          node: Optional[str] = None, gpu_count: Optional[int] = None, gpu_type: Optional[str] = None,
          scheduler_state: Optional[str] = None, evidence_kind: Optional[str] = None, evidence_uri: str = "",
          late_register: Optional[str] = None, final: bool = False, spool: Optional[str] = None,
          script: str = "sisteer.registry.emit_cli state", session: Optional[str] = None, note: Optional[str] = None) -> int:
    spool_path = _spool(spool)
    if not spool_path:
        _say(f"state {state_}: no spool resolves; nothing written (job continues)")
        return 0
    producer = emit.producer_block(script, session=session)
    campaign: Optional[str] = None
    plan: Optional[Dict[str, object]] = None
    targets: List[Tuple[str, Optional[str]]] = []      # (run_id, campaign) — every run this state applies to
    pend = _pending_file(pending_dir, job)
    if pend:
        try:
            data = _load_json(pend)
            for rid, pl in _runs_of(data):
                targets.append((rid, str(pl.get("campaign") or "") or None))
            if targets and not run_id:
                run_id, campaign = targets[0]
                plan = _plan_of(data)
        except (OSError, ValueError) as exc:
            _say(f"state {state_}: pending file {pend} unreadable ({exc})")
    late_paths: List[str] = []
    if not run_id and late_register:
        try:
            plan = _plan_of(_load_json(late_register))
            plan.setdefault("extra", {})
            if isinstance(plan["extra"], dict):
                plan["extra"]["registered_at"] = "job_start"   # type: ignore[index]
            plan.setdefault("lifecycle_state", "running")
            rid = emit.check_planned_run(plan, producer)
            ids, run_payload, _ = emit.planned_run_records(plan, producer)
            run_payload["lifecycle_state"] = "running"
            run_payload["started_utc"] = utc_now()
            run_payload["queued_utc"] = run_payload["started_utc"]   # a reached state's instant; never observed → equal to the start
            w0 = emit.RecordWriter(spool_path, str(plan["campaign"]), ids, producer, push_script=_push_script())
            p = w0.emit_run(run_payload)
            if p:
                late_paths.append(p)
            run_id, campaign = ids, str(plan["campaign"])
            _say(f"state {state_}: late registration of {run_id} from {late_register} (registered_at=job_start)")
        except (OSError, ValueError, emit.PlanRefused) as exc:
            _say(f"state {state_}: late registration refused ({exc}); nothing written")
            return 0
    if not run_id:
        where = f"{pending_dir}/{job}.json" if pending_dir and job else "(no --pending-dir/--job)"
        _say(f"state {state_}: no run to attach to — no pending file at {where} and no --late-register plan; nothing written (job continues)")
        return 0
    if not any(t[0] == run_id for t in targets):
        targets.insert(0, (run_id, campaign))
    if state_ == "running" and (gpu_count is None or not gpu_type):
        # the job states its own hardware — SLURM_GPUS_ON_NODE / SLURM_JOB_GPUS / CUDA_VISIBLE_DEVICES and
        # nvidia-smi, sacct AllocTRES as the fallback — unless the caller stated it; the sources ride the state's note.
        # Captured ONCE per job (the environment is the job's, not a run's) and applied to every target run below.
        hw = emit.hardware_from_env(job_id=job)
        if gpu_count is None and hw.get("gpu_count") is not None:
            gpu_count = int(hw["gpu_count"])       # type: ignore[arg-type]
        else:
            hw["gpu_count"] = None
        if not gpu_type and hw.get("gpu_type"):
            gpu_type = str(hw["gpu_type"])
        else:
            hw["gpu_type"] = None
        hw_note = emit.hardware_note(hw)
        if hw_note:
            note = f"{note}; {hw_note}" if note else hw_note
            _say(f"state {state_}: {hw_note}")
    fields: Dict[str, object] = {}
    if state_ == "running":
        if node:
            fields["node_hostname"] = node
        if job:
            fields["cluster_job_id"] = str(job)
        if os.environ.get("SLURM_JOB_NAME"):
            fields["job_name"] = os.environ["SLURM_JOB_NAME"]
        if gpu_count is not None:
            fields["gpu_count"] = int(gpu_count)
        if gpu_type:
            fields["gpu_type"] = gpu_type
    paths = list(late_paths)
    for rid, camp in targets:
        if not camp:
            try:
                camp = identity.parse_run_id(rid)["campaign"]
            except ValueError:
                _say(f"state {state_}: {rid!r} is not a registry run id; nothing written for it")
                continue
        w = emit.RecordWriter(spool_path, camp, rid, producer, push_script=_push_script())
        p = w.emit_state(state_, evidence_kind or _EVIDENCE_DEFAULT.get(state_, "job_log"), evidence_uri, scheduler_state=scheduler_state,
                         note=note)
        if p:
            paths.append(p)
        if fields:
            p = w.emit_patch(fields=fields)
            if p:
                paths.append(p)
        w.push(final=final or state_ in ("failed", "cancelled", "generated"))
    for p in paths:
        _say(f"state {state_}: wrote {p}")
    if len(targets) > 1:
        _say(f"state {state_}: applied to {len(targets)} runs of job {job}")
    print(run_id)
    return 0


def push(run_dir: Optional[str] = None, final: bool = False, pending_dir: Optional[str] = None, job: Optional[str] = None,
         spool: Optional[str] = None) -> int:
    """Push one run's record directory: ``--run-dir`` explicitly, or resolved from the job's pending file
    (``<spool>/<campaign>/<run_dir_name(run_id)>``). Nothing to push (no script, no dir) → exit 0 quietly."""
    script = _push_script()
    if not script:
        return 0
    run_dirs: List[str] = [run_dir] if run_dir else []
    if not run_dirs:
        pend = _pending_file(pending_dir, job)
        spool_path = _spool(spool)
        if not pend or not spool_path:
            return 0
        try:
            data = _load_json(pend)
            for rid, plan in _runs_of(data):
                run_dirs.append(os.path.join(spool_path, str(plan.get("campaign") or ""), identity.run_dir_name(rid)))
        except (OSError, ValueError) as exc:
            _say(f"push: pending file {pend} unreadable ({exc})")
            return 0
    for d in run_dirs:
        if not os.path.isdir(d):
            continue
        w = emit.RecordWriter(os.path.dirname(os.path.dirname(d)), os.path.basename(os.path.dirname(d)),
                              os.path.basename(d), emit.producer_block("sisteer.registry.emit_cli push"), push_script=script)
        w.dir = d
        rc = w.push(final=final)
        _say(f"push: {d} (final={final}) rc={rc}")
    return 0


# --------------------------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python3 -S -m sisteer.registry.emit_cli", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--plan", help="a planned-run JSON (submit-time registration)")
    src.add_argument("--from-config", metavar="CFG", help="the job's resolved config: self-registration from it + the seeds")
    r.add_argument("--spool")
    r.add_argument("--job-id")
    r.add_argument("--job-name")
    r.add_argument("--sha-pin")
    r.add_argument("--pending-dir")
    r.add_argument("--check-only", action="store_true")
    r.add_argument("--no-push", action="store_true")
    r.add_argument("--session")
    # --from-config facts (the shell's resolved config + scheduler environment)
    r.add_argument("--model-key", help="the harness model key (+ view suffix), a legacy_spellings.yaml model_keys spelling")
    r.add_argument("--evals", help="comma-separated eval ids this job runs (its declared evals)")
    r.add_argument("--serving-path", help="lora_native | lens | hf | passthrough — the path this job resolved")
    r.add_argument("--campaign", help="the registry campaign (legacy_spellings.yaml campaigns)")
    r.add_argument("--roster-selection", default="main", help="the roster's object selection under the target (default main)")
    r.add_argument("--roster", help="the roster YAML (default: configs/sdfbase_full22_roster.yaml of the config's checkout)")
    r.add_argument("--pins", help="the object pins JSON (default: configs/sdfbase_full22_pins.json of the config's checkout)")
    r.add_argument("--seeds", help="the registry seeds directory (default: configs/registry of the config's checkout)")
    r.add_argument("--code-sha", help="the producing checkout's sha (default: git rev-parse HEAD of the cwd)")
    r.add_argument("--substrate", help="cluster_a | cluster_b | modal | runpod | grading_host (default: derived from the environment)")
    # static-reg-fix: a static phase registers its runs AT THE RUNG IT DRAWS — the producer key is the rung's private run name, the
    # cap the drawn one, with its index in the ladder; all three cap facts together or none (jobplan.rung_facts refuses otherwise)
    r.add_argument("--producer-key", help="the run key (`sdfbase_full22_<tag>_sampled_t<T>[_cap<N>]`); default <job_name>@<job_id>")
    r.add_argument("--serving-jobs", help="D1918: comma list of the store's serving job ids known at job start (the sibling shard groups' live jobs; "
                   "this job's own id is added) -> extra.serving_jobs, sorted, with extra.serving_jobs_source")
    r.add_argument("--token-cap", type=int, help="the DRAWN cap of this phase (token_cap_tokens on the run; the instrument keeps its read-at cap)")
    r.add_argument("--rung-index", type=int, help="the cap's index in --ladder (4096 -> 0, 8192 -> 1, 16384 -> 2)")
    r.add_argument("--ladder", help="the cap ladder, comma-separated (4096,8192,16384)")
    r.add_argument("--pass-label", help="base_cap (rung 0) | ladder_rung")
    r.add_argument("--trial-index", type=int, help="the sampled trial (t0 -> 0, t1 -> 1)")
    # static-reg-fix-2: a phase whose per-pair gate selects what it draws registers ONLY those pairs
    r.add_argument("--positions-mode", help="the applied-positions mode this job will serve (vocab.APPLIED_POSITIONS_MODES, e.g. non_template): "
                                            "the instrument per eval is then the seeded one carrying that mode; unset: the canonical instrument")
    r.add_argument("--pairs", help="eval:object,… — register ONLY these (eval, object) pairs of the declared evals x explicit_cells (the phase's "
                                   "per-pair gate selection); a pair outside them is refused (exit 2)")
    s = sub.add_parser("state")
    s.add_argument("--state", required=True, choices=("running", "generated", "failed", "cancelled"))
    s.add_argument("--run-id")
    s.add_argument("--pending-dir")
    s.add_argument("--job")
    s.add_argument("--node")
    s.add_argument("--gpu-count", type=int)
    s.add_argument("--gpu-type")
    s.add_argument("--scheduler-state")
    s.add_argument("--evidence-kind")
    s.add_argument("--evidence-uri", default="")
    s.add_argument("--late-register")
    s.add_argument("--note")
    s.add_argument("--final", action="store_true")
    s.add_argument("--spool")
    s.add_argument("--session")
    u = sub.add_parser("push")
    u.add_argument("--run-dir")
    u.add_argument("--pending-dir")
    u.add_argument("--job")
    u.add_argument("--spool")
    u.add_argument("--final", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "register" and args.from_config:
        missing = [k for k in ("model_key", "evals", "serving_path", "campaign", "job_id", "pending_dir") if not getattr(args, k)]
        if missing:
            _say(f"register --from-config: missing {', '.join('--' + m.replace('_', '-') for m in missing)}; nothing written (REFUSED, exit 2)")
            return EXIT_REFUSED
        try:
            return register_from_config(args.from_config, model_key=args.model_key, evals=[e for e in args.evals.split(",") if e.strip()],
                                        serving_path=args.serving_path, campaign=args.campaign, job_id=args.job_id, pending_dir=args.pending_dir,
                                        job_name=args.job_name, roster_selection=args.roster_selection, roster=args.roster, pins=args.pins,
                                        seeds=args.seeds, code_sha=args.code_sha, substrate=args.substrate, spool=args.spool,
                                        push=not args.no_push, session=args.session, producer_key=args.producer_key, token_cap=args.token_cap,
                                        token_cap_rung_index=args.rung_index, ladder=([int(x) for x in args.ladder.split(",") if x.strip()] if args.ladder else None),
                                        pass_label=args.pass_label, trial_index=args.trial_index,
                                        pairs=([args.pairs] if args.pairs is not None else None),
                                        serving_jobs=([x for x in args.serving_jobs.split(",")] if getattr(args, "serving_jobs", None) is not None else None),
                                        positions_mode=args.positions_mode)
        except Exception as exc:  # noqa: BLE001 — exit 3: a job never runs on a registration that did not happen
            _say(f"register --from-config: failed ({type(exc).__name__}: {exc}); nothing written (FAILED, exit 3 — the job must not run unregistered)")
            return EXIT_FAILED
    if args.cmd == "register":
        return register(args.plan, spool=args.spool, job_id=args.job_id, job_name=args.job_name, pending_dir=args.pending_dir,
                        check_only=args.check_only, submit_sha_pin=args.sha_pin, push=not args.no_push, session=args.session)
    if args.cmd == "state":
        try:
            return state(args.state, run_id=args.run_id, pending_dir=args.pending_dir, job=args.job, node=args.node,
                         gpu_count=args.gpu_count, gpu_type=args.gpu_type or None, scheduler_state=args.scheduler_state or None,
                         evidence_kind=args.evidence_kind, evidence_uri=args.evidence_uri, late_register=args.late_register,
                         final=args.final, spool=args.spool, session=args.session, note=args.note)
        except Exception as exc:  # noqa: BLE001 — a job never fails on its registry step
            _say(f"state {args.state}: failed ({type(exc).__name__}: {exc}); job continues")
            return 0
    if args.cmd == "push":
        try:
            return push(args.run_dir, final=args.final, pending_dir=args.pending_dir, job=args.job, spool=args.spool)
        except Exception as exc:  # noqa: BLE001
            _say(f"push: failed ({type(exc).__name__}: {exc}); job continues")
            return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
