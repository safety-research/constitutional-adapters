#!/usr/bin/env python3
"""smoke_producer.py — one SYNTHETIC run's complete lifecycle through the harness's own registry hooks.

STDLIB ONLY, like the emitter it drives: runs on a compute node before ``uv sync``, as
``PYTHONPATH=<checkout>/src python3 -S scripts/registry/smoke_producer.py …``.

The staged live drills need a producer that writes records
exactly the way a real job does, without a GPU, a model or a store, so the transport (push key, inbox,
harvester, the cluster pull) and the lifecycle fold are exercised end to end under a campaign nobody can
mistake for a measurement. Every id it mints is prefixed ``drill_`` (the campaign MUST be
``drill_<stamp>``; anything else is refused), the dimension rows it seeds are its own (a ``drill_``
model pair, eval, item set, judge, instrument, and one fitted object per campaign — never the shared
``none`` object, never a real model or eval), and every record says so in ``extra.synthetic``.

The lifecycle, step by step, through the SAME code the harness runs (nothing here writes a record
file by hand):

1. ``register``  — the submit wrapper's step: ``emit_cli.register`` validates the planned run STRICTLY,
   writes ``run`` (queued) + ``run_state queued`` + the job binding ``run_patch``, stores the pending file
   (``<pending-dir>/<job-id>.json``) and pushes. (The Modal launcher's registration path left this drill with the
   Modal transport; ``hooks.queued_modal`` itself stays for the frozen launchers outside the registry.)
2. ``running``   — the job shell's step (``s2_init``): ``emit_cli.state --state running`` through the
   pending file, with the node / job / GPU facts.
3. the store     — ``hooks.StoreRecorder``, the recorder a ``CellStore`` calls: ``on_manifest`` (the
   store's own ``run`` + ``run_state running``), ``on_row`` × N (buffered ``item_result`` + ``item_flags``),
   ``on_generation_complete`` (a ``run_patch``), ``on_results`` (``run_state generated``, the completion
   ``run_patch``, ONE ``artifact`` pointer — the completion artifact; the drill writes no ``rows.jsonl``, so
   there is exactly one — and the ``not_measured`` degeneration rows, then a final push).
4. judging       — ``hooks.judgment_scope``: the claim (``judgment in_progress`` → the run folds to
   ``judging``), N judge ``item_result`` rows, ONE ``run_metric``, then ``judgment complete`` +
   ``run_state judged`` + the completion ``run_patch`` + ``not_measured`` rows + a final push.

The harvester then moves the run to ``ingested`` (S6: a complete judgment with coverage and a completion
artifact), so the proof of a finished drill is ``run_state_event`` = queued, running ×2 (job_log +
completion_artifact), generated, judged, and ``run.lifecycle_state = ingested`` with ``judged_utc`` set.

Expected rows per run of N items (the summary the script prints on stdout as ``SMOKE_JSON {…}`` carries
the EXACT per-type record counts read back from the files it wrote): ``item_result`` 2N (N store rows
under metric ``response_chars`` with ``judgment_id = ''`` + N judge rows under ``asr``), ``item_flags`` N,
``run_metric`` 1, ``artifact`` 1, ``judgment`` 1 (folded in_progress → complete), ``run_degeneration`` 8
(4 declared measures × the store's and the judgment's ``not_measured`` rows), ``run_state_event`` 5.

Usage::

  smoke_producer.py --campaign drill_<utcstamp> [--spool DIR] [--push-script PATH | --no-push]
                    [--pending-dir DIR] [--store-dir DIR] [--run-key KEY] [--n-items N]
                    [--job-id ID] [--job-name NAME] [--node HOST] [--code-sha 40hex]
                    [--dims | --no-dims] [--pushlog FILE] [--json]

``--spool`` defaults to the resolved spool (``$SISTEER_REGISTRY_SPOOL``, else the shared root's
``registry/records``); the push script to ``$SISTEER_REGISTRY_PUSH``
(``source $SHARED/scripts/registry/registry_env.sh`` sets both on a cluster). ``--pushlog`` appends every
record path the drill wrote, relative to the spool, to that file (an operator's index of what one drill
wrote). Exit 0 on success, 2 on a refused argument, 3 when a step
raised (the exception is printed; a REAL job never fails on these steps — every hook is guarded there).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import tempfile
import time
import traceback
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from sisteer.registry import emit, emit_cli, hooks, identity
from sisteer.registry import records as R
from sisteer.registry import synth

SCRIPT = "scripts/registry/smoke_producer.py"
SUMMARY_SENTINEL = "SMOKE_JSON "
CAMPAIGN_RE = re.compile(r"^drill_[A-Za-z0-9][A-Za-z0-9_.-]*$")

# -- the drill's own dimension identities (every id says `drill`; none collides with a real row) -----------
MODEL_BASE = "drill_model_base"
MODEL_IT = "drill_model_it"
MODEL_REV = hashlib.sha1(b"drill model revision").hexdigest()
EVAL_ID = "drill_eval"
ITEM_SET_ID = "drill_eval_v1"
JUDGE_MODEL = "drill-judge-rule"
RUBRIC_SHA = hashlib.sha256(b"drill rubric: value >= 0.5 is harmful").hexdigest()
PROTOCOL_SHA = hashlib.sha256(b"drill protocol: synthetic rule, no API call").hexdigest()
THRESHOLD_RULE = "score > 0.5"
JUDGE_ID = identity.judge_id(JUDGE_MODEL, RUBRIC_SHA, PROTOCOL_SHA, THRESHOLD_RULE)
INSTRUMENT: Dict[str, object] = synth.instrument_fields(eval_id=EVAL_ID, item_set_id=ITEM_SET_ID, judge_id=JUDGE_ID,
                                                        harness_config_hash=hashlib.sha256(b"drill harness config").hexdigest()[:16])
INSTRUMENT_ID = str(INSTRUMENT["instrument_id"])
ITEMS_SHA = hashlib.sha256(b"drill items: q00000..").hexdigest()
CHAT_TEMPLATE_SHA = hashlib.sha256(b"drill chat template").hexdigest()
GEN_METRIC = "response_chars"
JUDGE_METRIC = "asr"
OBJECT_LABEL = "constitution_lora"
SYNTHETIC_TAG = "drill"


def object_id_for(campaign: str) -> str:
    """One fitted object per campaign (its ``campaign`` column is the drill's), so two drills never rewrite
    each other's object row and no drill touches the shared ``none`` object."""
    return hashlib.sha256(f"drill object {campaign}".encode("utf-8")).hexdigest()


def intervention_hash_for(campaign: str, run_key: str) -> str:
    return identity.sha16(f"{campaign}|{run_key}|intervention")


def object_key_for(campaign: str, run_key: str) -> str:
    """The store's object discriminator, exactly as ``StoreRecorder.object_key_of`` spells it."""
    return f"{OBJECT_LABEL}@{intervention_hash_for(campaign, run_key)[:8]}"


def run_id_for(campaign: str, run_key: str) -> str:
    return identity.run_id(campaign, run_key, object_key_for(campaign, run_key), EVAL_ID, INSTRUMENT_ID)


def dimension_payloads(campaign: str) -> List[Tuple[str, Dict[str, object]]]:
    """The dimension rows a drill run needs, all its own: two models, the eval, its item set, the judge, the
    instrument and the per-campaign object. Pure; identical for the same campaign."""
    oid = object_id_for(campaign)
    model_common = {"family": "qwen35", "size_label": "4b", "chat_template_sha256": CHAT_TEMPLATE_SHA, "generation_config_present": 0,
                    "thinking_channel": "not_available", "paper_status": "reference", "extra": {"synthetic": SYNTHETIC_TAG}}
    return [
        ("model", dict(model_common, model_id=MODEL_BASE, checkpoint_kind="base", hf_repo="synthetic/drill-base-4b",
                       hf_revision=hashlib.sha1(b"drill base").hexdigest())),
        ("model", dict(model_common, model_id=MODEL_IT, checkpoint_kind="instruct", hf_repo="synthetic/drill-instruct-4b",
                       hf_revision=MODEL_REV)),
        ("eval", {"eval_id": EVAL_ID, "eval_category": "static_attack", "headline_metric_id": JUDGE_METRIC,
                  "metric_direction": "lower_is_better", "attacker_required": 0, "paper_status": "dropped",
                  "description": "synthetic drill eval (results-registry live drills); never a measurement",
                  "extra": {"synthetic": SYNTHETIC_TAG}}),
        ("item_set", {"item_set_id": ITEM_SET_ID, "eval_id": EVAL_ID, "dataset_version": "synthetic/drill@v1", "items_sha256": ITEMS_SHA,
                      "source_uri": "synthetic://drill/items.jsonl", "item_id_scheme": "q<index>", "subsample_rule": "full",
                      "extra": {"synthetic": SYNTHETIC_TAG}}),
        ("judge", {"judge_id": JUDGE_ID, "judge_model": JUDGE_MODEL, "judge_role": "judge", "rubric_name": "drill_rule",
                   "rubric_version": "1", "rubric_sha256": RUBRIC_SHA, "protocol_sha256": PROTOCOL_SHA, "score_scale": "0-1",
                   "threshold_rule": THRESHOLD_RULE, "judge_backend": "live", "extra": {"synthetic": SYNTHETIC_TAG}}),
        ("instrument", dict(INSTRUMENT, extra={"synthetic": SYNTHETIC_TAG})),
        ("object", {"object_id": oid, "object_kind": "lora", "paper_label": OBJECT_LABEL, "campaign": campaign, "object_role": "constitution",
                    "fit_target_kind": "corpus_document_fit", "fit_corpus_id": "synthetic-drill-corpus", "fit_corpus_sha256": ITEMS_SHA,
                    "fit_model_id": MODEL_BASE, "fit_thinking_render": "off", "fit_objective": "ce", "fit_positions": "all",
                    "fit_layers_json": [4, 5, 6], "fit_layer_index_convention": "zero_based_decoder_block", "fit_layer_count": 3,
                    "fit_layer_selection_rule": "listed_in_config", "fit_rank": 4, "fit_alpha": 8.0, "fit_seed": 0,
                    "fit_seed_confidence": "recorded", "fit_lr": 1e-3, "fit_lr_selected_by": "manual", "derivation": "none",
                    "artifact_sha256": oid, "artifact_bytes": 1234, "status": "reference", "extra": {"synthetic": SYNTHETIC_TAG}}),
    ]


def planned_run(campaign: str, run_key: str, n_items: int, code_sha: str) -> Dict[str, object]:
    """The planned run the submit wrapper registers — every required ``run`` field resolved,
    the object / instrument / model ids from :func:`dimension_payloads`. Pure and deterministic."""
    return {
        "campaign": campaign, "producer_run_key_legacy": run_key, "producer_object_key_legacy": object_key_for(campaign, run_key),
        "model_id": MODEL_IT, "model_revision": MODEL_REV, "object_id": object_id_for(campaign), "transfer_kind": "base_to_instruct",
        "eval_id": EVAL_ID, "item_set_id": ITEM_SET_ID, "instrument_id": INSTRUMENT_ID, "instrument_match": "match",
        "harness": "bespoke_sisteer", "harness_version": "0.1", "scoring_rule": "judge_threshold",
        "prompt_template_sha256": str(INSTRUMENT["prompt_template_sha256"]), "decode_seed_rule": "none", "decode_source": "greedy",
        "token_cap_tokens": 512, "thinking_mode": "off", "serving_path": "lora_native", "applied_positions_mode": "all",
        "multi_turn_steering_scope": "single_turn", "code_sha": code_sha, "n_items_planned": int(n_items),
        "intervention_hash": intervention_hash_for(campaign, run_key), "harness_config_hash": str(INSTRUMENT["harness_config_hash"]),
        "env_digest": identity.sha16("drill env digest"),
        "strength_coefficient": 1.0, "strength_nominal": 1.0, "strength_nominal_source": "filed", "is_operating_point": 1,
        "extra": {"synthetic": SYNTHETIC_TAG, "smoke_producer": SCRIPT},
    }


def resolved_spec(campaign: str, run_key: str, job_id: Optional[str]) -> Dict[str, object]:
    """What ``capeval.runtime.resolve_cell`` would hand the store: the keys ``StoreRecorder.run_payload`` reads."""
    return {"sweep": campaign, "run_id": run_key, "model_id": MODEL_IT, "model_revision": MODEL_REV, "arm_name": OBJECT_LABEL,
            "kind": "lora", "coeff": 1.0, "intervention_hash": intervention_hash_for(campaign, run_key), "engine": "lora_native",
            "trial_id": 0, "seed": 0, "attempt": 0, "harness_config_hash": str(INSTRUMENT["harness_config_hash"]),
            "env_digest": identity.sha16("drill env digest"), "slurm_job_id": job_id, "tiers": [1], "cell_index": 0}


def store_row(item_index: int, created_at: float) -> Dict[str, object]:
    """One generation row as ``CellStore.append_row`` records it (a ``ResultRow`` as a dict); every fifth
    item is at the token cap so ``item_flags`` rows are exercised too."""
    at_cap = item_index % 5 == 0
    return {"task": EVAL_ID, "task_version": "v1", "domain": "safety", "tier": 1, "item_id": f"q{item_index:05d}", "trial_id": 0,
            "seed": 0, "attempt": 0, "position_order": None, "metric": GEN_METRIC, "value": float(120 + 7 * (item_index % 11)),
            "extra_json": json.dumps({"at_cap": at_cap, "finish_reason": "length" if at_cap else "stop", "response_tokens": 512 if at_cap else 64}),
            "created_at": created_at, "wall_clock_s": 0.01}


def judge_rows(n_items: int) -> List[Dict[str, object]]:
    return [{"item_id": f"q{i:05d}", "metric_id": JUDGE_METRIC, "value": float(i % 2), "label": "harmful" if i % 2 else "safe"}
            for i in range(n_items)]


# --------------------------------------------------------------------------------------------
# the lifecycle
# --------------------------------------------------------------------------------------------


def _write_json(path: str, obj: Mapping[str, object]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path


def emit_dimensions(spool: str, campaign: str, code_sha: str, push_script: Optional[str], session: Optional[str]) -> Tuple[str, List[str]]:
    """Seed the drill's dimension rows under ``<spool>/<campaign>/dimensions/`` (one record file per type, the
    emitter's own sequence numbers) and push the directory. Idempotent in the fold: the same campaign yields
    the same payloads, so a re-run re-states the same rows."""
    w = emit.RecordWriter(spool, campaign, "dimensions", emit.producer_block(SCRIPT + " dimensions", code_sha, session),
                          push_script=push_script)
    paths = []
    for rt, payload in dimension_payloads(campaign):
        p = w.write(rt, payload)
        if p:
            paths.append(p)
    w.push(final=True)
    return w.dir or "", paths


def run_lifecycle(campaign: str, run_key: str, spool: str, *, n_items: int, code_sha: str, job_id: str, job_name: str, node: str,
                  pending_dir: str, store_dir: str, push_script: Optional[str], session: Optional[str]) -> Dict[str, object]:
    """Steps 1–4 for one run; returns the run id, the run directory, the pending file and the step timings."""
    t: Dict[str, float] = {}
    plan = planned_run(campaign, run_key, n_items, code_sha)
    rid = run_id_for(campaign, run_key)
    plan_path = _write_json(os.path.join(pending_dir, f"plan-{run_key}.json"), plan)
    pending_file = os.path.join(pending_dir, f"{job_id}.json")

    # 1. register (the submit wrapper's step)
    t0 = time.monotonic()
    rc = emit_cli.register(plan_path, spool=spool, job_id=job_id, job_name=job_name, pending_dir=pending_dir,
                           script=SCRIPT + " register", push=bool(push_script), session=session)
    if rc != 0:
        raise RuntimeError(f"emit_cli.register exited {rc} (the plan was refused — see the [registry] lines)")
    t["register_s"] = round(time.monotonic() - t0, 3)

    # 2. running (the job shell: s2_init's exact call)
    t0 = time.monotonic()
    rc = emit_cli.state("running", pending_dir=pending_dir, job=job_id, node=node, gpu_count=0, gpu_type=None,
                        scheduler_state="RUNNING", spool=spool, script=SCRIPT + " state", session=session)
    if rc != 0:
        raise RuntimeError(f"emit_cli.state running exited {rc}")
    t["running_s"] = round(time.monotonic() - t0, 3)

    # 3. the store (the recorder a CellStore calls, hook by hook)
    t0 = time.monotonic()
    spec = resolved_spec(campaign, run_key, job_id)
    ident = dict(plan, run_id=rid)
    rec = hooks.StoreRecorder(run_key, spec, {"*": ident}, spool=spool, push_script=push_script, script=SCRIPT + " store",
                              code_sha=code_sha, session=session)
    os.makedirs(store_dir, exist_ok=True)
    manifest = _write_json(os.path.join(store_dir, "manifest.json"), dict(spec, synthetic=SYNTHETIC_TAG))
    rec.on_manifest(spec, manifest)
    started = time.time()
    rec.on_row({"domain": "_meta", "metric": "ts_job_start", "value": started, "task": EVAL_ID, "item_id": "_meta"})
    for i in range(n_items):
        rec.on_row(store_row(i, started + 0.001 * i))
    marker = _write_json(os.path.join(store_dir, "generation_complete.json"), {"n": n_items, "synthetic": SYNTHETIC_TAG})
    rec.on_generation_complete(marker)
    rec.on_row({"domain": "_meta", "metric": "ts_cell_end", "value": started + 0.001 * n_items + 1.0, "task": EVAL_ID, "item_id": "_meta"})
    summary = {"n": n_items, "metric": GEN_METRIC, "synthetic": SYNTHETIC_TAG, "run_key": run_key, "campaign": campaign}
    results = _write_json(os.path.join(store_dir, "results.json"), summary)
    rec.on_results(summary, results)
    t["store_s"] = round(time.monotonic() - t0, 3)

    # 4. judging (the 3–5 lines a grading driver adds)
    t0 = time.monotonic()
    scope = hooks.judgment_scope(SCRIPT + " judge", {"run_id": rid, "campaign": campaign},
                                 {"judge_id": JUDGE_ID, "rubric_sha256": RUBRIC_SHA}, api_pool=None, spool=spool,
                                 push_script=push_script, registry_emit=True, store_uri=store_dir, session=session)
    if scope is None:
        raise RuntimeError("judgment_scope returned None (emission off)")
    rows = judge_rows(n_items)
    k = sum(1 for r in rows if r["value"] > 0.5)
    with scope:
        scope.items(rows)
        scope.metrics([{"metric_id": JUDGE_METRIC, "threshold": 0.5, "value": (k / n_items) if n_items else 0.0, "k": k, "n": n_items}])
    t["judging_s"] = round(time.monotonic() - t0, 3)

    run_dir = os.path.join(spool, campaign, identity.run_dir_name(rid))
    return {"run_id": rid, "run_dir": run_dir, "pending_file": pending_file, "plan_file": plan_path, "timings_s": t}


def count_records(paths: Sequence[str]) -> Dict[str, int]:
    """Records per type, read back from the files (the authoritative expected counts for verification)."""
    out: Dict[str, int] = {}
    for p in paths:
        for _, rec, _ in R.read_record_file(p):
            rt = str(rec.get("record_type"))
            out[rt] = out.get(rt, 0) + 1
    return dict(sorted(out.items()))


def expected_rows(counts: Mapping[str, int], n_items: int) -> Dict[str, int]:
    """What the database must hold for this run after the fold: item-scale rows are one per record, run-scale
    rows fold to one per natural key."""
    return {"item_result": counts.get("item_result", 0), "item_flags": counts.get("item_flags", 0),
            "run_degeneration": counts.get("run_degeneration", 0), "run_metric": counts.get("run_metric", 0),
            "artifact": counts.get("artifact", 0), "judgment": 1, "run": 1,
            "run_state_event": counts.get("run_state", 0), "n_items": int(n_items)}


def append_pushlog(pushlog: str, spool: str, paths: Sequence[str]) -> int:
    """One line per record path RELATIVE to the spool (an operator's index of what one drill wrote)."""
    os.makedirs(os.path.dirname(pushlog) or ".", exist_ok=True)
    rels = sorted(os.path.relpath(p, spool) for p in paths)
    with open(pushlog, "a", encoding="utf-8") as fh:
        for rel in rels:
            fh.write(rel + "\n")
    return len(rels)


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="smoke_producer.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--campaign", required=True, help="drill_<utcstamp>; refused unless it starts with drill_")
    p.add_argument("--spool", help="record spool (default: the resolved one — $SISTEER_REGISTRY_SPOOL, the shared root)")
    p.add_argument("--push-script", help="push_records.sh (default: $SISTEER_REGISTRY_PUSH)")
    p.add_argument("--no-push", action="store_true", help="never call the push script (the grading host's cluster pull is the backstop)")
    p.add_argument("--pending-dir", help="where the pending file goes (default: <spool>/../drills/<campaign>/pending)")
    p.add_argument("--store-dir", help="the synthetic store directory (default: <spool>/../drills/<campaign>/store/<run-key>)")
    p.add_argument("--run-key", default="drill_run_000", help="the producer's own execution key (one run per key)")
    p.add_argument("--n-items", type=int, default=3)
    p.add_argument("--job-id", help="the scheduler job id (default: $SLURM_JOB_ID, else drill-<pid>)")
    p.add_argument("--job-name", help="default: $SLURM_JOB_NAME, else smoke_producer")
    p.add_argument("--node", help="default: $SLURMD_NODENAME, else the hostname")
    p.add_argument("--code-sha", help="the 40-hex sha this copy came from (default: git rev-parse HEAD of the cwd, else unverified)")
    p.add_argument("--session", help="producer.session (default: $SISTEER_SESSION_NAME)")
    dims = p.add_mutually_exclusive_group()
    dims.add_argument("--dims", dest="dims", action="store_true", default=True, help="seed the drill's dimension rows (default)")
    dims.add_argument("--no-dims", dest="dims", action="store_false", help="skip the dimension rows (another producer seeded them)")
    p.add_argument("--pushlog", help="append every written record path (relative to the spool) to this file")
    p.add_argument("--json", action="store_true", help=f"print the summary as one line '{SUMMARY_SENTINEL}{{…}}' on stdout")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not CAMPAIGN_RE.match(args.campaign):
        print(f"smoke_producer: refusing campaign {args.campaign!r}: a drill campaign is drill_<stamp> (never a real campaign name)",
              file=sys.stderr)
        return 2
    if (os.environ.get(emit.DISABLED_ENV) or "").strip().lower() in ("1", "true", "yes", "on"):
        print(f"smoke_producer: {emit.DISABLED_ENV} is set — emission would be a no-op; unset it to drill", file=sys.stderr)
        return 2
    if args.n_items < 1:
        print("smoke_producer: --n-items must be >= 1", file=sys.stderr)
        return 2
    spool = args.spool or emit.spool_root()
    if not spool:
        print("smoke_producer: no spool resolves (pass --spool or set SISTEER_REGISTRY_SPOOL)", file=sys.stderr)
        return 2
    spool = os.path.abspath(spool)
    push_script: Optional[str] = None if args.no_push else (args.push_script or os.environ.get(emit.PUSH_SCRIPT_ENV))
    code_sha = args.code_sha or emit.git_sha()
    if not re.match(r"^[0-9a-f]{40}$", code_sha):
        print(f"smoke_producer: --code-sha must be 40 hex (got {code_sha!r}); the run record needs a real sha or a confidence entry",
              file=sys.stderr)
        return 2
    prev_push = os.environ.get(emit.PUSH_SCRIPT_ENV)
    prev_git_sha = emit.git_sha
    # the push script: explicit, else the environment; --no-push clears it for EVERY writer (emit_cli reads the env).
    # Every writer's producer block names the sha this copy came from: the drill runs from a bare file copy on the
    # cluster, where `git rev-parse HEAD` has no checkout to read. Both are restored on exit
    # (an in-process caller — the tests — keeps its module state).
    if push_script:
        os.environ[emit.PUSH_SCRIPT_ENV] = push_script
    else:
        os.environ.pop(emit.PUSH_SCRIPT_ENV, None)
    emit.git_sha = lambda cwd=None: code_sha  # type: ignore[assignment]
    try:
        return _run(args, spool, push_script, code_sha)
    finally:
        emit.git_sha = prev_git_sha  # type: ignore[assignment]
        if prev_push is None:
            os.environ.pop(emit.PUSH_SCRIPT_ENV, None)
        else:
            os.environ[emit.PUSH_SCRIPT_ENV] = prev_push


def _run(args: argparse.Namespace, spool: str, push_script: Optional[str], code_sha: str) -> int:
    job_id = args.job_id or os.environ.get("SLURM_JOB_ID") or f"drill-{os.getpid()}"
    job_name = args.job_name or os.environ.get("SLURM_JOB_NAME") or "smoke_producer"
    node = args.node or os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    session = args.session if args.session is not None else emit.session_from_env()
    drills_root = os.path.join(os.path.dirname(spool), "drills", args.campaign)
    pending_dir = os.path.abspath(args.pending_dir or os.path.join(drills_root, "pending"))
    store_dir = os.path.abspath(args.store_dir or os.path.join(drills_root, "store", args.run_key))

    # the store's own `run` record reads the job name from the environment exactly as a real job does (SLURM_JOB_NAME
    # is always set inside an allocation); a drill given --job-name outside one exports it for the duration so the
    # store's record agrees with the wrapper's binding — a later full `run` record's null would otherwise overwrite it
    prev_job_name = os.environ.get("SLURM_JOB_NAME")
    os.environ["SLURM_JOB_NAME"] = job_name
    try:
        return _lifecycle(args, spool, push_script, code_sha, job_id, job_name, node, session, pending_dir, store_dir)
    finally:
        if prev_job_name is None:
            os.environ.pop("SLURM_JOB_NAME", None)
        else:
            os.environ["SLURM_JOB_NAME"] = prev_job_name


def _lifecycle(args: argparse.Namespace, spool: str, push_script: Optional[str], code_sha: str, job_id: str, job_name: str, node: str,
               session: Optional[str], pending_dir: str, store_dir: str) -> int:
    started_utc = R.utc_now()
    t_all = time.monotonic()
    out: Dict[str, object] = {"campaign": args.campaign, "run_key": args.run_key, "spool": spool, "n_items": args.n_items,
                              "code_sha": code_sha, "job_id": job_id, "node": node, "host": socket.gethostname(),
                              "substrate": emit.substrate_from_env(), "push": bool(push_script), "push_script": push_script,
                              "started_utc": started_utc, "register": "slurm"}
    try:
        written: List[str] = []
        if args.dims:
            dims_dir, dim_paths = emit_dimensions(spool, args.campaign, code_sha, push_script, session)
            out["dims_dir"] = dims_dir
            out["dimension_records"] = count_records(dim_paths)
            written += dim_paths
        life = run_lifecycle(args.campaign, args.run_key, spool, n_items=args.n_items, code_sha=code_sha, job_id=job_id,
                             job_name=job_name, node=node, pending_dir=pending_dir, store_dir=store_dir, push_script=push_script,
                             session=session)
        out.update(life)
        run_paths = list(R.iter_record_files(str(life["run_dir"])))
        written += run_paths
        counts = count_records(run_paths)
        out["records"] = counts
        out["expected_rows"] = expected_rows(counts, args.n_items)
        out["files"] = sorted(os.path.relpath(p, spool) for p in written)
        out["n_files"] = len(written)
        out["bytes"] = sum(os.path.getsize(p) for p in written)
        if args.pushlog:
            out["pushlog"] = args.pushlog
            out["pushlog_lines"] = append_pushlog(args.pushlog, spool, written)
        out["store_dir"] = store_dir
        out["finished_utc"] = R.utc_now()
        out["elapsed_s"] = round(time.monotonic() - t_all, 3)
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001 — a DRILL reports the failure; a real job's hooks are guarded and never raise
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc().splitlines()[-6:]
        out["finished_utc"] = R.utc_now()
        out["elapsed_s"] = round(time.monotonic() - t_all, 3)
        print(f"smoke_producer: FAILED — {out['error']}", file=sys.stderr)
        if args.json:
            print(SUMMARY_SENTINEL + json.dumps(out, sort_keys=True, default=str), flush=True)
        else:
            print(json.dumps(out, indent=1, sort_keys=True, default=str), flush=True)
        return 3
    if args.json:
        print(SUMMARY_SENTINEL + json.dumps(out, sort_keys=True, default=str), flush=True)
    else:
        print(json.dumps(out, indent=1, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
