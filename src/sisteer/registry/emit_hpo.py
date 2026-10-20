"""NATIVE ``fit_search`` / ``fit_trial`` emission for the hyperparameter search drivers — STDLIB ONLY.

The search driver (``scripts/tune_arm.py``) records itself at run time through the producer writer, the way every other producer
does (``emit.RecordWriter``; ``fit_search`` and ``fit_trial`` are record types the reducer already folds — ``records.RECORD_TYPE_TABLE``):

    sid = open_search(campaign=..., model_id=..., object_kind="lora", method="sobol", search_space={...}, n_trials=16, ...)
    record_trial(sid, "r1_t03", params={...}, status="complete", objective_value=2.31, clip_fraction=0.02, ...)
    close_search(sid, winner_params={...}, winner_trial_id="r1_t03", winner_objective=2.31, argmin_interior=True, ..., n_trials_completed=16)

* ``search_id`` = ``"hpo-native:" + sha16("<campaign>|<model_id>|<object_kind>|<started_utc>")`` — a prefix DISTINCT from the post-hoc
  reader's ``"hpo:" + sha16(label|model|arm|position|si|key)`` (``adapters/hpo_logs.py``), so the two writers never mint one key
  (no new adapter, no collision; the reader keeps reading the OLD sweeps).
* ``fit_trial`` is emitted ONCE, at the trial's TERMINAL outcome — ``status`` ∈ ``complete | failed | pruned`` is a required CHECK
  column (``schema_def.py``) and the validator refuses an omitted status and ``running``. In-flight progress rides
  ``fit_search.extra.trials_in_flight`` through a superseding search record (:func:`update_search`); a trial that resumes from its
  checkpoint emits nothing until it finishes; one that is abandoned emits ``status = failed`` with ``extra.reason``.
* ``close_search`` writes a SUPERSEDING ``fit_search`` record under the same ``search_id`` (the fold keeps the latest by order key):
  the winner, the fit diagnostics (interiority, the quadratic fit, recentre rounds, the confirmation trial, the noise estimate) and
  ``n_trials_completed``; the record log is append-only, so the opening record stays as history.
* The named params: ``lr``, ``lora_rank``, ``lora_modules`` (a SORTED list), ``per_expert_rank`` (``{"shared_expert": r,
  "routed_expert": max(1, r // 8)}``), ``warmup_frac``, ``max_grad_norm``, ``init_scheme``, ``trainable_params``, ``lora_set_id`` — in
  ``search_space_json`` / ``params_json`` / ``winner_params_json``; the reconnaissance clip rides ``search_space_json.max_grad_norm`` +
  ``extra.clip_percentile`` + ``grad_norm_reconnaissance_uri``.
* Every record carries the central stamps ``emit.RecordWriter`` adds (the run-overrides file, a third cluster's label), so
  ``extra_json.run_overrides.id`` is on every search and trial of an overridden run.

The records land under ``<spool>/<campaign>/fit_search__<sha16>/`` (one directory per search, like ``object__<sha16>/``), pushed after
each write; a lost record never costs a trial (the writer's contract), and a refusal here — an unknown method / status / direction,
an unset substrate — is a ``ValueError`` naming the field, raised BEFORE anything is written.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import emit, identity
from .records import utc_now
from .vocab import FIT_SEARCH_SUBSTRATES, OBJECTIVE_DIRECTIONS, SEARCH_METHODS, TRIAL_STATUSES

SEARCH_ID_PREFIX = "hpo-native:"
#: the axis names a search space / trial params dict may carry; a caller may add others, these are the documented ones
NAMED_PARAMS = ("lr", "lora_rank", "lora_modules", "per_expert_rank", "warmup_frac", "max_grad_norm", "init_scheme", "trainable_params",
                "lora_set_id")


class HpoEmitRefused(ValueError):
    """A search or trial record that would not validate: the refusal names the field (nothing is written)."""


def search_id_for(campaign: str, model_id: str, object_kind: str, started_utc: str) -> str:
    """``hpo-native:<sha16>`` over ``campaign|model_id|object_kind|started_utc`` — distinct from the post-hoc reader's ``hpo:`` keys."""
    return SEARCH_ID_PREFIX + identity.sha16("|".join([str(campaign), str(model_id), str(object_kind), str(started_utc)]))


def run_dir_for(search_id: str) -> str:
    """The record directory name of one search under ``<spool>/<campaign>/``."""
    return f"fit_search__{identity.sha16(search_id)}"


#: The last ``written_utc`` this process stamped on a search's records, by (spool, campaign, search run dir). Every call here builds its
#: own ``RecordWriter`` (a different producer block per call), and a writer's stamp is strictly monotone only WITHIN the writer, so
#: ``open_search`` -> ``update_search`` -> ``close_search`` inside one millisecond stamped the same instant and the three ``fit_search``
#: records -- one key, folded by the latest stamp -- fell to sha order: the closing record lost to the progress record now and then
#: (tests/test_emit_hpo.py's flake). Each writer now starts strictly above the search's last stamp (``RecordWriter.not_before``).
_LAST_STAMP: Dict[Tuple[str, str, str], str] = {}


def _stamp_key(spool_path: Optional[str], campaign: str, search_id: str) -> Tuple[str, str, str]:
    return (str(spool_path), str(campaign), run_dir_for(search_id))


def _writer(campaign: str, search_id: str, *, spool: Optional[str], script: str, code_sha: Optional[str], session: Optional[str],
            substrate: Optional[str], push_script: Optional[str]) -> emit.RecordWriter:
    spool_path = spool or emit.spool_root()
    producer = emit.producer_block(script, code_sha=code_sha, session=session, substrate=substrate)
    w = emit.RecordWriter(spool_path, campaign, run_dir_for(search_id), producer, push_script=push_script)
    w.not_before(_LAST_STAMP.get(_stamp_key(spool_path, campaign, search_id), ""))
    return w


def _write(w: emit.RecordWriter, campaign: str, search_id: str, record_type: str, payload: Mapping[str, object]) -> Optional[str]:
    """``w.write`` and then the search's watermark advanced to the stamp it issued, so the next writer of this search stays above it."""
    path = w.write(record_type, payload)
    if path is not None:
        key = _stamp_key(w.spool, campaign, search_id)
        _LAST_STAMP[key] = max(_LAST_STAMP.get(key, ""), w.last_written_utc)
    return path


def _sorted_modules(params: Mapping[str, object]) -> Dict[str, object]:
    out = dict(params)
    mods = out.get("lora_modules")
    if isinstance(mods, (list, tuple, set, frozenset)):
        out["lora_modules"] = sorted(str(m) for m in mods)
    return out


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_or_unverified(value, *, field: str = "config_sha256", where: str = "fit_search", fc: Optional[Dict[str, object]] = None):
    """The registry's own rule for the sha columns (schema_def SHA256_OR_UNVERIFIED: a 64-hex sha256 or the literal
    ``unverified``; nullable). None stays None (the column is nullable). A 64-hex value is kept, lower-cased. Anything
    else (the emitter used to put a 16-hex prefix here and the harvester refused EVERY fit_search record, leaving
    every fit_trial blocked on a missing parent) becomes ``unverified`` with a confidence declaration in ``fc`` — the
    schema's own escape hatch — and a stderr notice, instead of a refused record nobody sees."""
    if value is None:
        return None
    v = str(value).strip()
    if v == "unverified":
        # the literal sentinel is admitted only WITH a confidence declaration (validate.check_column_value refuses it bare);
        # a producer that already declared one keeps it, otherwise one is supplied here
        if fc is not None:
            cur = fc.get(field)
            if not (isinstance(cur, Mapping) and str(cur.get("reason") or "").strip()):
                fc[field] = {"confidence": "unverified",
                             "reason": f"{where}.{field}: recorded as unverified by the producer (no config file sha available)"}
        return v
    if _SHA256_RE.match(v.lower()):
        return v.lower()
    reason = f"{where}.{field}: {v[:24]!r} is not a 64-hex sha256 (len {len(v)}); recorded as unverified"
    if fc is not None:
        fc[field] = {"confidence": "unverified", "reason": reason}
    print(f"[emit_hpo] {reason}", file=sys.stderr)
    return "unverified"


def open_search(*, campaign: str, model_id: str, object_kind: str, method: str, search_space: Mapping[str, object], n_trials: int,
                objective: str = "best_heldout_loss", objective_direction: str = "minimize", substrate: Optional[str] = None,
                code_sha: Optional[str] = None, config_sha256: Optional[str] = None, extra: Optional[Mapping[str, object]] = None,
                started_utc: Optional[str] = None, search_id: Optional[str] = None, target_object_class: Optional[Mapping[str, object]] = None,
                job_ids: Optional[Sequence[str]] = None, wandb_sweep_url: Optional[str] = None, trials_uri: Optional[str] = None,
                grad_norm_reconnaissance_uri: Optional[str] = None, spool: Optional[str] = None, script: str = "scripts/tune_arm.py",
                session: Optional[str] = None, push_script: Optional[str] = None, notes: Optional[str] = None) -> str:
    """Write the OPENING ``fit_search`` record; returns the ``search_id``. ``method`` ∈ SEARCH_METHODS, ``objective_direction`` ∈
    OBJECTIVE_DIRECTIONS, ``substrate`` (default: the producer's) ∈ FIT_SEARCH_SUBSTRATES — refused by name otherwise."""
    if method not in SEARCH_METHODS:
        raise HpoEmitRefused(f"method {method!r} is not one of {SEARCH_METHODS}")
    if objective_direction not in OBJECTIVE_DIRECTIONS:
        raise HpoEmitRefused(f"objective_direction {objective_direction!r} is not one of {OBJECTIVE_DIRECTIONS}")
    if int(n_trials) <= 0:
        raise HpoEmitRefused(f"n_trials {n_trials!r} must be a positive integer")
    if substrate is not None and substrate not in FIT_SEARCH_SUBSTRATES:
        raise HpoEmitRefused(f"substrate {substrate!r} is not one of {FIT_SEARCH_SUBSTRATES} (a third Slurm cluster is `slurm` + "
                             f"{emit.CLUSTER_LABEL_ENV}; never aliased to grading_host)")
    started = started_utc or utc_now()
    sid = search_id or search_id_for(campaign, model_id, object_kind, started)
    # `unverified` is admitted on fit_search.substrate alone: the PRODUCER block still names a real substrate
    w = _writer(campaign, sid, spool=spool, script=script, code_sha=code_sha, session=session,
                substrate=(None if substrate == "unverified" else substrate), push_script=push_script)
    sub = substrate or str(w.producer.get("substrate") or "")
    if sub not in FIT_SEARCH_SUBSTRATES:
        raise HpoEmitRefused(f"substrate {sub!r} is not one of {FIT_SEARCH_SUBSTRATES}")
    _fc: Dict[str, object] = {}
    payload: Dict[str, object] = {
        "search_id": sid, "campaign": campaign,
        "target_object_class_json": dict(target_object_class or {}, model_id=model_id, object_kind=object_kind),
        "method": method, "n_trials": int(n_trials), "n_trials_completed": 0,
        "search_space_json": _sorted_modules(search_space), "objective": objective, "objective_direction": objective_direction,
        "recentre_rounds": 0, "substrate": sub, "job_ids_json": [str(j) for j in (job_ids or [])],
        "code_sha": code_sha or w.producer.get("code_sha") or "unverified",
        "config_sha256": sha256_or_unverified(config_sha256, fc=_fc), "started_utc": started,
        "wandb_sweep_url": wandb_sweep_url, "trials_uri": trials_uri, "grad_norm_reconnaissance_uri": grad_norm_reconnaissance_uri,
        "producer_session": w.producer.get("session"), "notes": notes,
        "extra": dict({"model_id": model_id, "object_kind": object_kind, "trials_in_flight": [], "phase": "open"}, **(dict(extra) if extra else {})),
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    if _fc:
        payload["field_confidence_json"] = {**dict(payload.get("field_confidence_json") or {}), **_fc}
    if _write(w, campaign, sid, "fit_search", payload) is None and not w.disabled:
        raise HpoEmitRefused("fit_search record not written")
    w.push()
    return sid


def record_trial(search_id: str, trial_id: str, params: Mapping[str, object], *, status: str, campaign: str,
                 objective_value: Optional[float] = None, clip_fraction: Optional[float] = None, epochs_seen: Optional[int] = None,
                 wandb_run_id: Optional[str] = None, started_utc: Optional[str] = None, finished_utc: Optional[str] = None,
                 extra: Optional[Mapping[str, object]] = None, spool: Optional[str] = None, script: str = "scripts/tune_arm.py",
                 code_sha: Optional[str] = None, session: Optional[str] = None, substrate: Optional[str] = None,
                 push_script: Optional[str] = None) -> Optional[str]:
    """Write ONE ``fit_trial`` record at the trial's TERMINAL outcome (``status`` ∈ TRIAL_STATUSES; ``complete`` needs an objective
    value). Returns the record path (None when emission is off)."""
    if status not in TRIAL_STATUSES:
        raise HpoEmitRefused(f"trial {trial_id!r}: status {status!r} is not one of {TRIAL_STATUSES} (a trial is recorded once, at its terminal outcome)")
    if status == "complete" and objective_value is None:
        raise HpoEmitRefused(f"trial {trial_id!r}: a complete trial needs its objective_value")
    w = _writer(campaign, search_id, spool=spool, script=script, code_sha=code_sha, session=session, substrate=substrate, push_script=push_script)
    payload: Dict[str, object] = {
        "search_id": search_id, "trial_id": str(trial_id), "params_json": _sorted_modules(params), "status": status,
        "objective_value": objective_value, "clip_fraction": clip_fraction, "epochs_seen": epochs_seen, "wandb_run_id": wandb_run_id,
        "started_utc": started_utc, "finished_utc": finished_utc or utc_now(), "extra": dict(extra) if extra else None,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    p = _write(w, campaign, search_id, "fit_trial", payload)
    w.push()
    return p


def update_search(search_id: str, *, campaign: str, opening: Mapping[str, object], n_trials_completed: int,
                  trials_in_flight: Optional[Sequence[str]] = None, job_ids: Optional[Sequence[str]] = None,
                  extra: Optional[Mapping[str, object]] = None, spool: Optional[str] = None, script: str = "scripts/tune_arm.py",
                  code_sha: Optional[str] = None, session: Optional[str] = None, substrate: Optional[str] = None,
                  push_script: Optional[str] = None) -> Dict[str, object]:
    """A SUPERSEDING ``fit_search`` record carrying progress: ``n_trials_completed``, the trials in flight (``extra.trials_in_flight``),
    the job ids so far. ``opening`` is the payload the previous search record carried (the caller chains what this returns); the
    fold keeps the latest record. Returns the payload written (the caller's next ``opening``)."""
    w = _writer(campaign, search_id, spool=spool, script=script, code_sha=code_sha, session=session, substrate=substrate, push_script=push_script)
    payload = _superseding(opening, search_id, n_trials_completed, phase="running", trials_in_flight=trials_in_flight, job_ids=job_ids, extra=extra)
    _write(w, campaign, search_id, "fit_search", payload)
    w.push()
    return payload


def _superseding(opening: Mapping[str, object], search_id: str, n_trials_completed: int, *, phase: str,
                 trials_in_flight: Optional[Sequence[str]], job_ids: Optional[Sequence[str]], extra: Optional[Mapping[str, object]]) -> Dict[str, object]:
    """The next ``fit_search`` payload chained from ``opening`` (what :func:`update_search` and :func:`close_search` share): the id and
    the completed count, the job ids merged with the opening's, and ``extra`` carrying ``phase`` + ``trials_in_flight`` under the
    caller's own ``extra`` keys."""
    payload = dict(opening)
    payload["search_id"] = search_id
    payload["n_trials_completed"] = int(n_trials_completed)
    if payload.get("config_sha256") is not None:          # A superseding record chained from a legacy opening is normalised too
        _fc: Dict[str, object] = dict(payload.get("field_confidence_json") or {})   # seeded with the opening's declarations (kept)
        payload["config_sha256"] = sha256_or_unverified(payload["config_sha256"], fc=_fc)
        if _fc:
            payload["field_confidence_json"] = _fc
    if job_ids is not None:
        payload["job_ids_json"] = sorted(set(str(j) for j in job_ids) | set(str(j) for j in (opening.get("job_ids_json") or [])))
    ex = dict(opening.get("extra") or {})
    ex["trials_in_flight"] = [str(t) for t in (trials_in_flight or ())]
    ex["phase"] = phase
    if extra:
        ex.update(dict(extra))
    payload["extra"] = ex
    return payload


def close_search(search_id: str, *, campaign: str, opening: Mapping[str, object], winner_params: Mapping[str, object], winner_trial_id: str,
                 winner_objective: float, n_trials_completed: int, argmin_interior: Optional[bool] = None,
                 quadratic_fit_positive_definite: Optional[bool] = None, quadratic_fit_params: Optional[Mapping[str, object]] = None,
                 recentre_rounds: int = 0, confirmation_trial_id: Optional[str] = None, confirmation_within_noise: Optional[bool] = None,
                 noise_estimate: Optional[float] = None, grad_norm_reconnaissance_uri: Optional[str] = None, trials_uri: Optional[str] = None,
                 job_ids: Optional[Sequence[str]] = None, finished_utc: Optional[str] = None, extra: Optional[Mapping[str, object]] = None,
                 spool: Optional[str] = None, script: str = "scripts/tune_arm.py", code_sha: Optional[str] = None, session: Optional[str] = None,
                 substrate: Optional[str] = None, push_script: Optional[str] = None) -> Dict[str, object]:
    """The CLOSING ``fit_search`` record (superseding, same ``search_id``): the winner, the fit diagnostics, the completed count. Booleans
    are stored as 0 / 1 (the schema's ``B`` columns); ``extra.phase = closed`` and no trial is left in flight. Returns the payload written."""
    w = _writer(campaign, search_id, spool=spool, script=script, code_sha=code_sha, session=session, substrate=substrate, push_script=push_script)
    payload = _superseding(opening, search_id, n_trials_completed, phase="closed", trials_in_flight=(), job_ids=job_ids, extra=extra)
    payload.update({
        "winner_params_json": _sorted_modules(winner_params), "winner_trial_id": str(winner_trial_id), "winner_objective": float(winner_objective),
        "argmin_interior": None if argmin_interior is None else int(bool(argmin_interior)),
        "quadratic_fit_positive_definite": None if quadratic_fit_positive_definite is None else int(bool(quadratic_fit_positive_definite)),
        "quadratic_fit_params_json": dict(quadratic_fit_params) if quadratic_fit_params is not None else None,
        "recentre_rounds": int(recentre_rounds), "confirmation_trial_id": confirmation_trial_id,
        "confirmation_within_noise": None if confirmation_within_noise is None else int(bool(confirmation_within_noise)),
        "noise_estimate": noise_estimate, "finished_utc": finished_utc or utc_now(),
    })
    if grad_norm_reconnaissance_uri is not None:
        payload["grad_norm_reconnaissance_uri"] = grad_norm_reconnaissance_uri
    if trials_uri is not None:
        payload["trials_uri"] = trials_uri
    payload = {k: v for k, v in payload.items() if v is not None}
    _write(w, campaign, search_id, "fit_search", payload)
    w.push(final=True)
    return payload


def per_expert_rank(rank: int, active_experts: int = 8) -> Dict[str, int]:
    """FD 6's MoE convention: the shared expert at the full rank, every routed expert at ``max(1, rank // active_experts)``."""
    r = int(rank)
    return {"shared_expert": r, "routed_expert": max(1, r // int(active_experts))}


__all__ = ["HpoEmitRefused", "NAMED_PARAMS", "SEARCH_ID_PREFIX", "close_search", "open_search", "per_expert_rank", "record_trial",
           "run_dir_for", "search_id_for", "update_search"]
