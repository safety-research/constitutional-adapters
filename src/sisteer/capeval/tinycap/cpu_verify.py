"""BigCodeBench verification as a CPU phase — the two-phase contract's file layout,
the three scoring mechanisms, the BINDING and the ``verify`` orchestration
(landing 2).

The GPU row (``served_capability_run.py generate --verification-phase generate``)
generates BigCodeBench WITHOUT scoring — ``eval_async(score=False)`` over a task
built ``generation_only`` (its sandbox dropped after every check ran) — and leaves
three things: the UNSCORED ``.eval`` in the store's task log dir
(``contracts.inspect_log_dir``), and in the ROW dir ``<row_dir>/generation/``:

* the generation SIDECAR ``<cell>__<task>__<store>.json`` (:data:`GENERATION_SIDECAR_FIELDS`;
  ``<store>`` = the store root's basename, so the in-run repeat ``…-rep2`` of the same
  ``--cell`` never collides with the cell's own):
  every generation-side fact the filing tail needs — the regime fields read off
  the live provider, the client record, the ledger's rejected count at generation
  end, the plan's facts (``inspect_served.plan_facts``), the target's
  ``model_usage`` key, the log's ids and their digest, the driver's concurrency;
* the PENDING MARKER ``pending_<cell>__<task>__<store>.json`` — EXACTLY
  ``contracts.PENDING_MARKER_FIELDS``; while it exists the task is NOT done and is
  never re-generated (a coupled run over it refuses by name).

The CPU verifier (``served_capability_run.py verify``, :func:`verify_store`) then
declares its own sandbox owner ``(run_id, cell, THIS CPU FunctionCall id)``, writes
its pointer ``<row_dir>/watch/attempt_<fcid>.json`` (``phase verify``, one cell —
the landing-1 watcher's ``--row-dir`` mode classifies and sweeps it per its one
cell), scores the stored log by ONE of three mechanisms (:data:`contracts.
CPU_VERIFY_MECHANISMS`; the G5 probe ``scripts/modal/tinycap_cpu_verify_probe.py``
decides which) and BINDS:

    the LOG OF RECORD is the GPU phase's log with, per sample, ``scores`` attached,
    keyed by ``(sample.id, sample.epoch)``. For every sample the vehicle's sample
    must exist exactly once, its ``output.completion`` byte-equal, its
    ``output.model`` equal and its final assistant message text equal; the
    original's ``model_usage``, ``output``, ``messages``, ``metadata`` and ``events``
    are kept untouched. A missing, duplicate, foreign or mismatched sample REFUSES
    the task by name — nothing is filed and the pending marker stays.

The original is archived to ``<row_dir>/generation/<cell>__<task>__<store>.eval``, the
merged log written in place, and ``results_from_log(merged, target_model=<the
sidecar's target_model>)`` feeds the SAME filing tail the coupled path runs
(``inspect_served._file_task_results``). ``manifest.row_engine.verification_pass``
becomes the ``verified`` block (EXACTLY ``contracts.VERIFICATION_PASS_VERIFIED_
FIELDS``), the marker is removed and ``RESULT`` gains
``result=COMPLETE status=OK … verified_by=<fcid>``.

Mechanisms:

* ``M1_score_async`` — ``score_async(copy, [verify()])``: predicted to raise (the
  stock scorer calls ``sandbox().exec`` and a re-score has no sandbox);
* ``M1b_score_async_sandboxenv`` — ``score_async`` with each sample yielded from
  inside ``sandboxenv_context(task_name, spec, max_sandboxes, True, sample)``
  (``inspect_ai._eval.task.sandbox``, internal), so the scorer finds an owned
  sandbox; the mutation is in place and the identical binding assertions run;
* ``M2_replay_vehicle`` — the public-API vehicle: ``eval_async(coupled task, model=
  replay/<name>)`` where the ``replay`` provider (:func:`replay_model`) returns each
  sample's STORED ``ModelOutput`` by ``(id, epoch)``; the vehicle's log is used ONLY
  to obtain scores, never as the log of record.

Module-level imports are stdlib + contracts + ``completion`` (the tinycap store side's
ONE JSON writer) + ``sandbox_scope`` only: the GPU side (``inspect_served``) imports this
module for the sidecar writer, and every inspect_ai use is lazy, inside the function
that needs it.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import socket
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import sandbox_scope as scope

__all__ = [
    "CpuVerifyError", "GenerationPhase", "GENERATION_DIR", "PENDING_MARKER_PREFIX", "GENERATION_SIDECAR_FIELDS",
    "VERIFY_POINTER_FIELDS",
    "REPLAY_PROVIDER", "DEFAULT_MECHANISM", "MAX_SANDBOXES_CEILING", "RESULT_TAG",
    "SCORER_IDENTITY_FIELDS", "executed_scorer_identity", "generation_scorer_identity", "scorer_identity_problems",
    "generation_sidecar_path", "pending_marker_path", "archived_log_path", "verify_pointer_path",
    "verify_sandbox_record_path", "generation_pointer_path", "write_generation_pending", "read_generation_sidecar",
    "read_generation_sidecar_if_present", "read_pending_marker", "pending_marker_exists", "write_verify_pointer",
    "write_generation_sidecar",
    "verified_block", "pending_block", "result_line", "append_result", "sha256_file", "default_max_sandboxes",
    "sample_key", "final_assistant_text", "unscored_problems", "strip_scores", "binding_problems", "bind_scores",
    "write_merged_log", "restore_unscored_log", "replay_model", "run_mechanism", "build_coupled_bcb_task",
    "verify_store", "one_cell_dir", "inspect_ai_version", "VERIFY_STEPS", "VERIFY_STEP_HOOK", "last_result_line",
    "SCORING_RECEIPT_FIELDS", "SCORING_RECEIPT_FIELDS_V1", "SCORING_RECEIPT_FIELD_SETS", "scoring_receipt_path", "verdict_digest", "write_scoring_receipt", "read_scoring_receipt",
    "row_disagreements", "VERIFY_CLAIM_FIELDS", "VERIFY_SUPERSEDES_ENV", "verify_claim_path", "read_verify_claim",
    "claim_verifier",
]


class CpuVerifyError(RuntimeError):
    """A refusal of the CPU verify phase — names the file, the sample or the field
    that departs; the driver turns it into a ``[served-cap] verify:`` exit. Nothing
    is filed after one, and the pending marker stays."""


@dataclass(frozen=True)
class GenerationPhase:
    """The row's generation-phase facts the GPU driver hands ``run_one_task``:
    the row dir (``--row-engine-record``'s ``row_dir``) the sidecar and the
    pending marker are written to, the cell, the GPU FunctionCall id (the pending
    marker's ``generation_fcid``; None on a hand run) and the driver's concurrency
    (already S // K on a K-concurrent row — the verifier's ``max_sandboxes`` basis)."""

    row_dir: Path
    cell: str
    generation_fcid: Optional[str]
    concurrency: int
    #: the store root's basename (``Path(--out-root).name``) — keys the row dir's generation files per STORE,
    #: since the in-run repeat ``<suffix>-rep2`` runs the same ``--cell`` into another store (worker B, 2026-09-29)
    store: str


# --------------------------------------------------------------------------- #
# the row dir's generation/ and watch/ files                                     #
# --------------------------------------------------------------------------- #

GENERATION_DIR = "generation"
WATCH_DIR = "watch"
#: the pending marker's file-name prefix under ``generation/`` — the harness globs ``pending_*.json`` by it
PENDING_MARKER_PREFIX = "pending_"
#: the verifier's default mechanism until the G5 probe records another: if M1b and M2 both pass,
#: M2 (public API only) unless M1b is measurably simpler to bind
DEFAULT_MECHANISM = "M2_replay_vehicle"
#: ``max_sandboxes = min(S // K, 8)`` — the ceiling
MAX_SANDBOXES_CEILING = 8
assert DEFAULT_MECHANISM in C.CPU_VERIFY_MECHANISMS

#: EXACTLY the generation sidecar's keys
GENERATION_SIDECAR_FIELDS: Tuple[str, ...] = (
    "task", "cell", "store", "log", "log_sha256", "n_items", "n_expected_full", "sample_ids", "pin_ids_sha256", "generated_at",
    "generation_fcid", "n_requests", "n_retried_requests", "generated_calls", "wall_clock_s", "logs_reused", "regimes", "client",
    "n_http_rejected_retried", "route_virtual_model", "target_model", "sampling_ref", "thinking", "reasoning_parser",
    "limits", "out_of_support", "row_identity", "attempt_id", "concurrency", "max_model_len", "plan",
    "producer_tree_sha", "inspect_ai_version")

#: the verifier's attempt pointer: the harness's ``POINTER_FIELDS`` (``scripts/modal/tinycap_served.py``) +
#: ``phase`` (``verify``) — NO ``cells`` list and no ``row_attempt`` block, so the watcher's ``--row-dir`` mode reads
#: it as a ONE-cell pointer (``pointer_owners`` -> the one owner) and never as a row pointer. Pinned against the
#: harness's tuple by ``tests/test_inspect_served_verification_phase.py`` (a source read, no Modal import).
VERIFY_POINTER_FIELDS: Tuple[str, ...] = (
    "run_id", "cell", "function_call_id", "name", "model", "mode", "out_dir", "controls",
    "tree_sha", "sandbox_record", "sandbox_app_name", "started_at", "phase")

#: the RESULT line's leading tag — the harness's (``tinycap_served.result_stamp_line``), so one reader parses both
RESULT_TAG = "tinycap-modal"
REC_STATUS_OK = "OK"

#: the checkpoints of :func:`verify_store`, in order, each reported to :data:`VERIFY_STEP_HOOK` the moment it is
#: done — ``scored`` (the bind done, nothing written yet) and then every store- or row-dir-mutating step (the two
#: swap steps from inside :func:`write_merged_log`) — a TEST seam (landing-2 review):
#: ``tests/test_served_capability_verification_phase.py`` kills the verifier at every one and proves the next
#: ``verify`` converges to the uninterrupted store, and runs a second verifier from ``scored`` to prove the claim.
#: ``None`` in production: the hook costs a name lookup per step.
VERIFY_STEPS: Tuple[str, ...] = ("scored", "receipt_written", "archive_written", "merged_written", "task_filed",
                                 "gen_stats_written", "manifest_written", "controls_merged", "result_appended",
                                 "marker_removed", "claim_released")
VERIFY_STEP_HOOK: Optional[Callable[[str], None]] = None

#: the CLAIM — the single-writer belt (the harness owns the authoritative rule: it JOINS a live
#: verifier's call instead of spawning a second, its verifiers record's call id being the ownership token):
#: ``<row_dir>/generation/<cell>__<task>__<store>.claim.json``, created atomically at entry BEFORE any read of the
#: log, re-read before every store mutation, removed LAST (a completed store has no claim — the row dir carries
#: +1 inode per cell only while a verifier is in flight). A claim naming ANOTHER fcid refuses by name unless the
#: dispatcher set :data:`VERIFY_SUPERSEDES_ENV` to exactly that fcid — which it does ONLY for a call it found
#: terminal or lost, never for a live one.
VERIFY_CLAIM_FIELDS: Tuple[str, ...] = ("function_call_id", "run_id", "started_at", "host")
VERIFY_SUPERSEDES_ENV = "TINYCAP_VERIFY_SUPERSEDES"


def _step(name: str) -> None:
    assert name in VERIFY_STEPS, name
    if VERIFY_STEP_HOOK is not None:
        VERIFY_STEP_HOOK(name)


def _task_fs(task: str) -> str:
    """``inspect/bigcodebench_full`` -> ``inspect__bigcodebench_full`` (the marker / summary file stem)."""
    name = C.marker_filename(task)
    assert name.endswith(".json"), name
    return name[: -len(".json")]


def _store_key(store: str) -> str:
    if not isinstance(store, str) or not store or "/" in store:
        raise CpuVerifyError(f"the store key must be the store root's basename (non-empty, no '/'), got {store!r}")
    return store


def generation_sidecar_path(row_dir: str | Path, cell: str, task: str, store: str) -> Path:
    """``<row_dir>/generation/<cell>__<task>__<store>.json`` (``<store>`` = the store root's basename)."""
    return Path(row_dir) / GENERATION_DIR / f"{cell}__{_task_fs(task)}__{_store_key(store)}.json"


def pending_marker_path(row_dir: str | Path, cell: str, task: str, store: str) -> Path:
    """``<row_dir>/generation/pending_<cell>__<task>__<store>.json`` — the harness globs ``pending_*.json``."""
    return Path(row_dir) / GENERATION_DIR / f"{PENDING_MARKER_PREFIX}{cell}__{_task_fs(task)}__{_store_key(store)}.json"


def archived_log_path(row_dir: str | Path, cell: str, task: str, store: str) -> Path:
    """``<row_dir>/generation/<cell>__<task>__<store>.eval`` — the ORIGINAL unscored log, archived before the merge."""
    return Path(row_dir) / GENERATION_DIR / f"{cell}__{_task_fs(task)}__{_store_key(store)}.eval"


def verify_pointer_path(row_dir: str | Path, function_call_id: str) -> Path:
    """``<row_dir>/watch/attempt_<fcid>.json`` — the verifier's attempt pointer (the same name the GPU phase's row
    pointer wears under ITS fcid; the watcher reads every ``attempt_*.json`` under the row dir)."""
    return Path(row_dir) / WATCH_DIR / f"attempt_{function_call_id}.json"


def generation_pointer_path(row_dir: str | Path, function_call_id: str) -> Path:
    """The GPU phase's row pointer under the row dir — the same file name as :func:`verify_pointer_path`."""
    return verify_pointer_path(row_dir, function_call_id)


def verify_sandbox_record_path(row_dir: str | Path, function_call_id: str, cell: str) -> Path:
    """``<row_dir>/watch/sandboxes_<fcid>__<cell>.jsonl`` — the verifier's provider record, the harness's spelling
    (it exports the same path as ``TINYCAP_SANDBOX_RECORD`` for the verifier process; an exported path wins). In
    the ROW dir, not the cell store: the store is bounded at exactly 40 inodes (inode rule 2), and every
    verification artefact stays in the shared row-level location."""
    return Path(row_dir) / WATCH_DIR / f"sandboxes_{function_call_id}__{cell}.jsonl"


def _read_json(path: Path, *, what: str) -> Dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CpuVerifyError(f"{what} {path} is unreadable ({type(e).__name__}: {e})") from None
    if not isinstance(doc, dict):
        raise CpuVerifyError(f"{what} {path}: expected a JSON object, got {type(doc).__name__}")
    return doc


def _exact_keys(doc: Mapping[str, Any], want: Sequence[str], *, what: str) -> None:
    if set(doc) != set(want):
        missing = sorted(set(want) - set(doc))
        extra = sorted(set(doc) - set(want))
        raise CpuVerifyError(f"{what}: keys are not exactly {list(want)}"
                             + (f" (missing {missing})" if missing else "")
                             + (f" (unknown {extra})" if extra else ""))


def _check_marker_agrees(sidecar: Mapping[str, Any], marker: Mapping[str, Any]) -> None:
    """The pending marker is the sidecar's projection: every ``contracts.PENDING_MARKER_FIELDS`` value equal."""
    for k in C.PENDING_MARKER_FIELDS:
        if sidecar[k] != marker[k]:
            raise CpuVerifyError(f"generation sidecar and pending marker disagree on {k}: {sidecar[k]!r} != {marker[k]!r}")


def inspect_ai_version() -> str:
    """This process's ``inspect_ai.__version__`` ("" when the client is not importable): the sidecar records the
    GPU phase's, the verifier holds its own to it, the G5 probe records it."""
    try:
        import inspect_ai
    except ImportError:
        return ""
    return str(getattr(inspect_ai, "__version__", ""))


def _check_sidecar(sidecar: Mapping[str, Any]) -> Tuple[str, str, str]:
    """EXACTLY :data:`GENERATION_SIDECAR_FIELDS`, non-empty task / cell, JSON without non-finite values;
    returns ``(task, cell, store key)``."""
    _exact_keys(sidecar, GENERATION_SIDECAR_FIELDS, what="generation sidecar")
    if not isinstance(sidecar["task"], str) or not sidecar["task"] or not isinstance(sidecar["cell"], str) or not sidecar["cell"]:
        raise CpuVerifyError(f"generation sidecar: task / cell must be non-empty strings, got "
                             f"{sidecar['task']!r} / {sidecar['cell']!r}")
    json.dumps(sidecar, allow_nan=False)      # stricter than the writer (no default=str): a non-JSON value refuses here
    return str(sidecar["task"]), str(sidecar["cell"]), _store_key(str(sidecar["store"]))


def write_generation_sidecar(row_dir: str | Path, *, sidecar: Mapping[str, Any]) -> Path:
    """The sidecar ALONE, atomically — what a re-run of ``generate`` that REUSED the retained unscored log
    writes (``logs_reused`` 1) beside the marker it found standing and held equal (the door: the
    marker is never rewritten for a generation that did not happen)."""
    task, cell, store = _check_sidecar(sidecar)
    return completion.atomic_write_json(generation_sidecar_path(row_dir, cell, task, store), sidecar)


def write_generation_pending(row_dir: str | Path, *, sidecar: Mapping[str, Any],
                             marker: Mapping[str, Any]) -> Tuple[Path, Path]:
    """The GPU phase's two files, the sidecar FIRST (so a marker never
    exists without its sidecar): ``sidecar`` must carry EXACTLY
    :data:`GENERATION_SIDECAR_FIELDS`, ``marker`` EXACTLY
    ``contracts.PENDING_MARKER_FIELDS``, and the two must agree on ``task``,
    ``cell``, ``log``, ``n_items``, ``n_expected_full``, ``generated_at`` and
    ``generation_fcid``. Atomic per file (``completion.atomic_write_json``: tmp +
    fsync + replace, a non-finite value refused). Returns ``(sidecar path, marker path)``."""
    _exact_keys(sidecar, GENERATION_SIDECAR_FIELDS, what="generation sidecar")
    _exact_keys(marker, C.PENDING_MARKER_FIELDS, what="pending marker")
    _check_marker_agrees(sidecar, marker)
    task, cell, store = _check_sidecar(sidecar)
    sc = write_generation_sidecar(row_dir, sidecar=sidecar)
    mk = completion.atomic_write_json(pending_marker_path(row_dir, cell, task, store), marker)
    return sc, mk


def read_generation_sidecar(row_dir: str | Path, cell: str, task: str, store: str) -> Dict[str, Any]:
    """The sidecar, held to EXACTLY :data:`GENERATION_SIDECAR_FIELDS` and to naming ``(cell, task, store)``."""
    p = generation_sidecar_path(row_dir, cell, task, store)
    if not p.is_file():
        raise CpuVerifyError(f"no generation sidecar at {p} — the GPU phase did not GENERATE {task} for {cell} "
                             f"into store {store!r} (or wrote it elsewhere)")
    doc = _read_json(p, what="generation sidecar")
    _exact_keys(doc, GENERATION_SIDECAR_FIELDS, what=f"generation sidecar {p}")
    if doc["task"] != task or doc["cell"] != cell or doc["store"] != store:
        raise CpuVerifyError(f"generation sidecar {p} names ({doc['cell']!r}, {doc['task']!r}, {doc['store']!r}), "
                             f"not ({cell!r}, {task!r}, {store!r})")
    return doc


def read_generation_sidecar_if_present(row_dir: str | Path, cell: str, task: str, store: str
                                       ) -> Optional[Dict[str, Any]]:
    """:func:`read_generation_sidecar` when the file exists, else None (a torn or foreign file still refuses)."""
    if not generation_sidecar_path(row_dir, cell, task, store).is_file():
        return None
    return read_generation_sidecar(row_dir, cell, task, store)


def read_pending_marker(row_dir: str | Path, cell: str, task: str, store: str) -> Dict[str, Any]:
    """The pending marker, held to EXACTLY ``contracts.PENDING_MARKER_FIELDS`` and to naming ``(cell, task)``."""
    p = pending_marker_path(row_dir, cell, task, store)
    if not p.is_file():
        raise CpuVerifyError(f"no pending marker at {p} — {task} for {cell} (store {store!r}) is not awaiting "
                             "verification (never generated in the two-phase mode, or already verified)")
    doc = _read_json(p, what="pending marker")
    _exact_keys(doc, C.PENDING_MARKER_FIELDS, what=f"pending marker {p}")
    if doc["task"] != task or doc["cell"] != cell:
        raise CpuVerifyError(f"pending marker {p} names ({doc['cell']!r}, {doc['task']!r}), not ({cell!r}, {task!r})")
    return doc


def pending_marker_exists(row_dir: str | Path, cell: str, task: str, store: str) -> bool:
    return pending_marker_path(row_dir, cell, task, store).is_file()


def write_verify_pointer(row_dir: str | Path, *, owner: scope.Owner, name: str, model: str, mode: str,
                         out_dir: str, controls: Sequence[str], tree_sha: str,
                         sandbox_record: Optional[str | Path] = None) -> Path:
    """The verifier's attempt pointer: the harness's pointer
    fields + ``phase: verify``, ``cell`` = the ONE cell, ``sandbox_record`` = the
    verifier's record in the row dir (``sandbox_record`` when the environment named
    one, else :func:`verify_sandbox_record_path`). Written BEFORE any sandbox is
    created, so a hard death of the CPU function is sweepable by the watcher."""
    record = (str(sandbox_record) if sandbox_record
              else str(verify_sandbox_record_path(row_dir, owner.function_call_id, owner.cell)))
    doc = {"run_id": str(owner.run_id), "cell": str(owner.cell), "function_call_id": str(owner.function_call_id),
           "name": str(name), "model": str(model), "mode": str(mode), "out_dir": str(out_dir),
           "controls": [str(c) for c in controls], "tree_sha": str(tree_sha),
           "sandbox_record": record,
           "sandbox_app_name": C.BCB_SANDBOX_APP_NAME, "started_at": time.time(),
           "phase": C.VERIFICATION_PHASE_VERIFY}
    assert set(doc) == set(VERIFY_POINTER_FIELDS)
    p = verify_pointer_path(row_dir, owner.function_call_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=True))
    return p


def pending_block(tasks: Sequence[str]) -> Dict[str, Any]:
    """``manifest.row_engine.verification_pass`` after the GPU phase: ``{"state": "pending", "tasks": [...]}``."""
    tasks = [str(t) for t in tasks]
    if not tasks or len(set(tasks)) != len(tasks):
        raise CpuVerifyError(f"a pending verification_pass names a non-empty list of distinct tasks, got {tasks!r}")
    return {"state": C.VERIFICATION_PASS_PENDING, "tasks": tasks}


def verified_block(*, mechanism: str, function_call_id: str, started_at: float, finished_at: float,
                   sandbox_facts: Mapping[str, Any], max_sandboxes: int, n_items: int,
                   vehicle_log_sha256: str, verifier_tree_sha: Optional[str] = None,
                   generation_tree_sha: Optional[str] = None, executed_image: Optional[str] = None,
                   executed_verify_timeout_s: Optional[int] = None, binding_version: Optional[str] = None) -> Dict[str, Any]:
    """``manifest.row_engine.verification_pass`` after the CPU phase — EXACTLY
    ``contracts.VERIFICATION_PASS_VERIFIED_FIELDS``. ``verifier_tree_sha`` is the ``TREE_SHA`` of
    the tree that FILES this block (the ``--tree`` stamp; None on a hand run without one),
    ``generation_tree_sha`` the store's producer tree (the GPU pointer's / the manifest's) —
    two RECORDED FACTS, never a gate (tinycap-verify-rate-limit, 2026-09-30): a verifier from a
    later landed tree files a store an earlier tree generated, and the block says which was which.
    ``executed_image`` / ``executed_verify_timeout_s`` / ``binding_version``: the scorer
    identity that ACTUALLY executed this scoring — held byte-equal to the generation's before scoring
    (:func:`scorer_identity_problems`); None on a resumed filing of an earlier attempt's receipt."""
    if mechanism not in C.CPU_VERIFY_MECHANISMS:
        raise CpuVerifyError(f"mechanism {mechanism!r} is not one of {list(C.CPU_VERIFY_MECHANISMS)}")
    doc = {"state": C.VERIFICATION_PASS_VERIFIED, "mechanism": str(mechanism), "function_call_id": str(function_call_id),
           "started_at": float(started_at), "finished_at": float(finished_at), "sandbox_facts": dict(sandbox_facts),
           "max_sandboxes": int(max_sandboxes), "n_items": int(n_items), "vehicle_log_sha256": str(vehicle_log_sha256),
           "verifier_tree_sha": (str(verifier_tree_sha) if verifier_tree_sha else None),
           "generation_tree_sha": (str(generation_tree_sha) if generation_tree_sha else None),
           "executed_image": (str(executed_image) if executed_image else None),
           "executed_verify_timeout_s": (int(executed_verify_timeout_s) if executed_verify_timeout_s is not None else None),
           "binding_version": (str(binding_version) if binding_version is not None else None)}
    assert tuple(doc) == C.VERIFICATION_PASS_VERIFIED_FIELDS
    return doc


# --------------------------------------------------------------------------- #
# the scorer identity: what the verifier WILL execute must be what the         #
# generation was hashed under                                                 #
# --------------------------------------------------------------------------- #

#: the scorer identity's three facts — the pinned sandbox image (hashed into the store identity), the verify timeout
#: (hashed), the binding contract version
SCORER_IDENTITY_FIELDS: Tuple[str, ...] = ("image", "verify_timeout_s", "binding_version")


def _compose_images(config: Any) -> List[str]:
    """The ``image`` of every service in a compose file, ``[]`` when ``config`` is not a readable compose YAML."""
    try:
        path = Path(str(config))
        if not path.is_file():
            return []
        import yaml  # the compose is YAML; PyYAML rides inspect_ai
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — not a compose file: the caller falls back to the contract, and says so
        return []
    services = doc.get("services") if isinstance(doc, Mapping) else None
    return [str(svc["image"]) for svc in (services or {}).values() if isinstance(svc, Mapping) and svc.get("image")]


def executed_scorer_identity(task: Any) -> Dict[str, Any]:
    """The scorer identity the RECONSTRUCTED coupled task will execute: the image read off the task's own sandbox
    compose (``task.sandbox.config``; ``image_source`` = ``compose``) — else, for a task without a compose (a local
    sandbox in a test, a bare spec), this tree's pinned :data:`contracts.BCB_SANDBOX_IMAGE` (``image_source`` =
    ``contract``, which is what the leg's builder bakes); the verify timeout the builder passes
    (:data:`contracts.BCB_VERIFY_TIMEOUT_S`, not introspectable off the scorer); and this tree's
    :data:`contracts.CPU_VERIFY_BINDING_VERSION`. A compose naming several distinct images refuses by name."""
    spec = getattr(task, "sandbox", None)
    images = _compose_images(getattr(spec, "config", None)) if spec is not None else []
    distinct = sorted(set(images))
    if len(distinct) > 1:
        raise CpuVerifyError(f"the reconstructed task's compose names {len(distinct)} images {distinct} — one scorer image is the contract")
    image = distinct[0] if distinct else str(C.BCB_SANDBOX_IMAGE)
    return {"image": image, "verify_timeout_s": int(C.BCB_VERIFY_TIMEOUT_S), "binding_version": str(C.CPU_VERIFY_BINDING_VERSION),
            "image_source": ("compose" if distinct else "contract")}


def generation_scorer_identity(plan_sandbox: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The scorer identity the GENERATION was hashed under, off the sidecar plan's ``sandbox`` facts (the generation
    tree's ``contracts.bcb_sandbox_facts()``): ``image``, ``verify_timeout_s``, and ``binding_version`` — declared since
    this change, :data:`contracts.CPU_VERIFY_BINDING_VERSION_LEGACY` for a sidecar that predates the key (every earlier
    generation ran under binding contract 1, the only one there was). Missing image / timeout refuses by name."""
    if not isinstance(plan_sandbox, Mapping):
        raise CpuVerifyError(f"the sidecar plan carries no sandbox facts ({plan_sandbox!r}) — the generation's scorer identity is unknown; refusing")
    missing = [k for k in ("image", "verify_timeout_s") if plan_sandbox.get(k) in (None, "")]
    if missing:
        raise CpuVerifyError(f"the sidecar plan's sandbox facts lack {missing} — the generation's scorer identity is unknown; refusing")
    return {"image": str(plan_sandbox["image"]), "verify_timeout_s": int(plan_sandbox["verify_timeout_s"]),
            "binding_version": str(plan_sandbox.get("binding_version") or C.CPU_VERIFY_BINDING_VERSION_LEGACY)}


def scorer_identity_problems(executed: Mapping[str, Any], generated: Mapping[str, Any]) -> List[str]:
    """Every field of :data:`SCORER_IDENTITY_FIELDS` on which the identity about to execute departs from the generation's
    — byte-equal or refused; an empty list admits the verifier (same tree or not)."""
    out: List[str] = []
    for k in SCORER_IDENTITY_FIELDS:
        e, g = executed.get(k), generated.get(k)
        if str(e) != str(g):
            out.append(f"{k}: this verifier would execute {e!r}, the store was generated under {g!r}")
    return out


def result_line(*, name: str, status: str, tree_sha: Optional[str], rc: int = 0,
                extra: Optional[Mapping[str, str]] = None) -> str:
    """One ``RESULT`` line in the harness's grammar (``tinycap_served.result_stamp_line``):
    ``tinycap-modal name=… rc=… result=… status=… sha=… at=…`` + this phase's trailing
    tokens (``phase=generate tasks=a,b`` / ``verified_by=<fcid>``). ``result`` is
    ``COMPLETE`` iff ``status`` is an OK status, else the status word itself (so a
    GENERATED store reads ``result=GENERATED status=GENERATED``)."""
    word = C.RESULT_COMPLETE if status in C.REC_OK_STATUSES else str(status)
    tail = "".join(f" {k}={v}" for k, v in (extra or {}).items())
    return (f"{RESULT_TAG} name={name} rc={rc} result={word} status={status} sha={tree_sha} "
            f"at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}{tail}")


def append_result(out_root: str | Path, line: str) -> Path:
    """Append one line to ``<out_root>/RESULT`` (the store root's completion stamp)."""
    p = Path(out_root) / C.RESULT_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip("\n") + "\n")
    return p


def last_result_line(out_root: str | Path) -> Optional[str]:
    """The store root's LAST ``RESULT`` line (what the harness's readers judge the store by), or None."""
    p = Path(out_root) / C.RESULT_FILENAME
    if not p.is_file():
        return None
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines[-1] if lines else None


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_with_sha256(src: Path, dst: Path) -> str:
    """Copy ``src`` to ``dst`` in 1 MiB chunks, hashing the bytes as they pass — the source read ONCE (a BCB log is
    50–250 MB on a network volume). Returns the source's digest."""
    h = hashlib.sha256()
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        for chunk in iter(lambda: fi.read(1 << 20), b""):
            h.update(chunk)
            fo.write(chunk)
    return h.hexdigest()


def default_max_sandboxes(concurrency: int) -> int:
    """``min(S // K, 8)`` — ``concurrency`` is the driver's, already S // K on a K-concurrent row."""
    n = int(concurrency)
    if n < 1:
        raise CpuVerifyError(f"max_sandboxes: the generation concurrency must be >= 1, got {concurrency!r}")
    return min(n, MAX_SANDBOXES_CEILING)


# --------------------------------------------------------------------------- #
# the binding — inspect_ai imported lazily                                      #
# --------------------------------------------------------------------------- #


def sample_key(sample: Any) -> Tuple[str, int]:
    """``(str(sample.id), int(sample.epoch))`` — the binding key (BigCodeBench's ids are ints in the log)."""
    return str(getattr(sample, "id")), int(getattr(sample, "epoch", 1) or 1)


def final_assistant_text(messages: Any) -> Optional[str]:
    """The text of the LAST assistant message, or None when there is none."""
    for m in reversed(list(messages or [])):
        if getattr(m, "role", None) == "assistant":
            return str(getattr(m, "text", ""))
    return None


def _output_completion(sample: Any) -> Optional[str]:
    out = getattr(sample, "output", None)
    return None if out is None else str(getattr(out, "completion", ""))


def _output_model(sample: Any) -> Optional[str]:
    out = getattr(sample, "output", None)
    return None if out is None else getattr(out, "model", None)


def unscored_problems(log: Any) -> List[str]:
    """Why ``log`` is NOT an unscored generation log (empty = it is): a sample carrying
    any score, a status other than ``success``, an errored sample, no samples."""
    problems: List[str] = []
    status = getattr(log, "status", None)
    if status != "success":
        problems.append(f"status {status!r}, not 'success'")
    samples = list(getattr(log, "samples", None) or [])
    if not samples:
        problems.append("no samples")
    scored = [str(getattr(s, "id", "?")) for s in samples if getattr(s, "scores", None)]
    if scored:
        problems.append(f"{len(scored)} sample(s) already carry scores ({scored[:5]}{'…' if len(scored) > 5 else ''})")
    errored = [str(getattr(s, "id", "?")) for s in samples if getattr(s, "error", None) is not None]
    if errored:
        problems.append(f"{len(errored)} errored sample(s) ({errored[:5]})")
    return problems


def strip_scores(log: Any) -> Any:
    """A deep copy of ``log`` with every sample's ``scores`` emptied and the header's
    score metrics / reductions cleared — what the G5 probe runs the mechanisms over
    (a scores-stripped COPY of a mirrored store's ``.eval``), and what an unscored
    generation log looks like."""
    out = log.model_copy(deep=True)
    for s in out.samples or []:
        s.scores = {}
    if out.results is not None:
        out.results.scores = []
    out.reductions = None
    return out


def binding_problems(original: Any, vehicle: Any, *, primary_scorer: str = "verify") -> List[str]:
    """Every reason the vehicle's scores may NOT be bound onto the original (empty =
    bind): a vehicle status other than ``success``; a duplicate ``(id, epoch)`` on
    either side; an original sample the vehicle lacks; a vehicle sample the
    original lacks (foreign); per matched pair a differing ``output.completion``
    (byte-equal required), a differing ``output.model``, a differing final assistant
    message text, an errored vehicle sample, or no ``primary_scorer`` score on it.
    Each names the sample."""
    problems: List[str] = []
    v_status = getattr(vehicle, "status", None)
    if v_status != "success":
        problems.append(f"vehicle log status {v_status!r}, not 'success' ({getattr(vehicle, 'error', None)})")
    orig = list(getattr(original, "samples", None) or [])
    veh = list(getattr(vehicle, "samples", None) or [])
    if not orig:
        problems.append("the original log carries no samples")
    o_keys = Counter(sample_key(s) for s in orig)
    v_keys = Counter(sample_key(s) for s in veh)
    for k, n in sorted(o_keys.items()):
        if n > 1:
            problems.append(f"original sample {k} appears {n} times")
    for k, n in sorted(v_keys.items()):
        if n > 1:
            problems.append(f"vehicle sample {k} appears {n} times (duplicate)")
    missing = sorted(set(o_keys) - set(v_keys))
    foreign = sorted(set(v_keys) - set(o_keys))
    for k in missing:
        problems.append(f"original sample {k} is missing from the vehicle")
    for k in foreign:
        problems.append(f"vehicle sample {k} is not in the original (foreign)")
    by_v = {sample_key(s): s for s in veh}
    for o in orig:
        k = sample_key(o)
        v = by_v.get(k)
        if v is None or o_keys[k] > 1 or v_keys[k] > 1:
            continue
        if getattr(v, "error", None) is not None:
            problems.append(f"vehicle sample {k} errored: {getattr(v.error, 'message', v.error)!s:.120}")
            continue
        oc, vc = _output_completion(o), _output_completion(v)
        if oc != vc:
            problems.append(f"sample {k}: output.completion differs (original {len(oc or '')} chars, "
                            f"vehicle {len(vc or '')} chars)")
        om, vm = _output_model(o), _output_model(v)
        if om != vm:
            problems.append(f"sample {k}: output.model differs ({om!r} != {vm!r})")
        if final_assistant_text(getattr(o, "messages", None)) != final_assistant_text(getattr(v, "messages", None)):
            problems.append(f"sample {k}: the final assistant message text differs")
        scores = getattr(v, "scores", None) or {}
        if not scores:
            problems.append(f"sample {k}: the vehicle carries no score")
        elif primary_scorer not in scores:
            problems.append(f"sample {k}: the vehicle carries scores {sorted(scores)} but no {primary_scorer!r}")
    return problems


def bind_scores(original: Any, vehicle: Any, *, primary_scorer: str = "verify") -> Any:
    """The LOG OF RECORD: a deep copy of ``original`` with each sample's
    ``scores`` set to its vehicle twin's (matched by :func:`sample_key`), the
    header's ``results.scores`` / ``reductions`` / ``eval.scorers`` taken from the
    vehicle (the metrics the scorer computed), and EVERYTHING ELSE — ``model_usage``,
    ``output``, ``messages``, ``metadata``, ``events``, ``eval``, ``stats``,
    ``plan`` — the original's, untouched. Refuses through :func:`binding_problems`
    first: a missing, duplicate, foreign or mismatched sample binds nothing."""
    problems = binding_problems(original, vehicle, primary_scorer=primary_scorer)
    if problems:
        shown = "; ".join(problems[:8]) + (f"; … {len(problems) - 8} more" if len(problems) > 8 else "")
        raise CpuVerifyError(f"refusing to bind the vehicle's scores onto the original log: {shown}")
    merged = original.model_copy(deep=True)
    by_v = {sample_key(s): s for s in vehicle.samples}
    for s in merged.samples:
        v = by_v[sample_key(s)]
        s.scores = {str(k): sc.model_copy(deep=True) for k, sc in (v.scores or {}).items()}
    if vehicle.results is not None:
        if merged.results is None:
            merged.results = vehicle.results.model_copy(deep=True)
            merged.results.total_samples = len(merged.samples)
            merged.results.completed_samples = len(merged.samples)
        else:
            merged.results.scores = [x.model_copy(deep=True) for x in vehicle.results.scores]
    merged.reductions = ([r.model_copy(deep=True) for r in vehicle.reductions]
                         if getattr(vehicle, "reductions", None) else None)
    if getattr(vehicle.eval, "scorers", None):
        merged.eval.scorers = [x.model_copy(deep=True) for x in vehicle.eval.scorers]
    return merged


def write_merged_log(merged: Any, *, log_path: str | Path, archive_path: str | Path) -> Dict[str, str]:
    """Archive the ORIGINAL at ``log_path`` to ``archive_path`` (byte copy, sha verified;
    an existing archive with the same bytes is fine, a different one refuses), then
    write ``merged`` over ``log_path`` atomically (a ``.part`` sibling + ``os.replace``,
    so a torn write never leaves a half log under the ``.eval`` name). Returns the two
    digests."""
    from inspect_ai.log import write_eval_log

    log_path, archive_path = Path(log_path), Path(archive_path)
    if not log_path.is_file():
        raise CpuVerifyError(f"the unscored log {log_path} is not a file")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        original_sha = sha256_file(log_path)
        if sha256_file(archive_path) != original_sha:
            raise CpuVerifyError(f"the archive {archive_path} exists with different bytes than {log_path} — a "
                                 "previous verify attempt archived another log; refusing to overwrite either")
    else:
        tmp = archive_path.with_name(archive_path.name + ".part")
        original_sha = _copy_with_sha256(log_path, tmp)      # copied and hashed in one read of the source
        if sha256_file(tmp) != original_sha:
            tmp.unlink(missing_ok=True)
            raise CpuVerifyError(f"archiving {log_path} to {archive_path}: the copy's bytes differ")
        os.replace(tmp, archive_path)
    _step("archive_written")
    part = log_path.with_name(log_path.name + ".part")
    write_eval_log(merged, str(part), format="eval")
    os.replace(part, log_path)
    _step("merged_written")
    return {"original_sha256": original_sha, "merged_sha256": sha256_file(log_path),
            "archive": str(archive_path), "log": str(log_path)}


def restore_unscored_log(archive_path: str | Path, log_path: str | Path, *, want_sha256: str) -> None:
    """Put the ARCHIVED original back under the store's log name: the archive is copied to a ``.part``
    sibling, its bytes held to ``want_sha256`` (the sidecar's ``log_sha256`` — the log the GPU phase left), then
    ``os.replace``d over ``log_path``, so the store's log is at every instant either the merged log an earlier
    attempt wrote or the original, never a torn file. The merged log it replaces is derivable — archive +
    re-score — which is exactly what the caller does next."""
    archive_path, log_path = Path(archive_path), Path(log_path)
    part = log_path.with_name(log_path.name + ".part")
    got = _copy_with_sha256(archive_path, part)
    if got != want_sha256:
        part.unlink(missing_ok=True)
        raise CpuVerifyError(f"restoring {archive_path} over {log_path}: the archive's bytes ({got[:12]}) are not the "
                             f"generated log's ({want_sha256[:12]}) — refusing to restore")
    os.replace(part, log_path)


# --------------------------------------------------------------------------- #
# the replay provider and the three mechanisms                                  #
# --------------------------------------------------------------------------- #

REPLAY_PROVIDER = "replay"
_REPLAY_API: Any = None


def _replay_api_class() -> Any:
    """Define and register ``@modelapi("replay")`` once, on first use (inspect_ai lazily)."""
    global _REPLAY_API
    if _REPLAY_API is not None:
        return _REPLAY_API
    from inspect_ai.log._samples import sample_active
    from inspect_ai.model import GenerateConfig, ModelAPI, modelapi

    @modelapi(name=REPLAY_PROVIDER)
    class ReplayAPI(ModelAPI):
        """Returns each sample's STORED ``ModelOutput`` by ``(id, epoch)`` — the replay
        vehicle of M2. It generates nothing: a sample it has no output for errors,
        and ``fail_on_error`` then fails the vehicle, which refuses the task."""

        def __init__(self, model_name: str, base_url: Optional[str] = None, api_key: Optional[str] = None,
                     api_key_vars: Optional[List[str]] = None, config: Any = None,
                     outputs: Optional[Mapping[Tuple[str, int], Any]] = None, **kw: Any) -> None:
            super().__init__(model_name, base_url, api_key, list(api_key_vars or []),
                             config if config is not None else GenerateConfig())
            self.outputs: Dict[Tuple[str, int], Any] = dict(outputs or {})
            self.served: Counter = Counter()

        async def generate(self, input: Any, tools: Any, tool_choice: Any, config: Any) -> Any:
            active = sample_active()
            if active is None:
                raise CpuVerifyError("replay: no active sample — the replay provider serves only inside an eval")
            key = (str(active.sample.id), int(active.epoch))
            try:
                out = self.outputs[key]
            except KeyError:
                raise CpuVerifyError(f"replay: no stored output for sample {key} — the vehicle asked for a sample the "
                                     "original log does not hold") from None
            self.served[key] += 1
            return out.model_copy(deep=True)

        def connection_key(self) -> str:
            return REPLAY_PROVIDER

    _REPLAY_API = ReplayAPI
    return ReplayAPI


def replay_model(original: Any, *, name: Optional[str] = None) -> Any:
    """A ``replay/<name>`` inspect ``Model`` over the original log's stored outputs
    (``name`` defaults to the original's model name after its provider prefix, so
    ``tinycap/petri__base`` replays as ``replay/petri__base``)."""
    from inspect_ai.model import GenerateConfig, Model

    api_cls = _replay_api_class()
    outputs = {}
    for s in original.samples or []:
        if getattr(s, "output", None) is None:
            raise CpuVerifyError(f"replay: original sample {sample_key(s)} carries no output")
        outputs[sample_key(s)] = s.output
    if name is None:
        full = str(getattr(original.eval, "model", "") or "")
        name = full.split("/", 1)[1] if "/" in full else (full or "model")
    return Model(api_cls(str(name), outputs=outputs), GenerateConfig())


def _task_scorers(task: Any) -> List[Any]:
    scorers = getattr(task, "scorer", None)
    if not scorers:
        raise CpuVerifyError("the coupled task carries no scorer — nothing to verify with")
    return list(scorers)


async def run_mechanism(mechanism: str, *, original: Any, coupled_task: Any, max_sandboxes: Optional[int],
                        log_dir: str | Path, max_connections: Optional[int] = None,
                        sample_shuffle: Optional[int | bool] = None, score_model: str = "mockllm/model",
                        task_name: Optional[str] = None) -> Tuple[Any, Path]:
    """Score ``original``'s samples by ``mechanism`` (:data:`contracts.CPU_VERIFY_MECHANISMS`)
    and return ``(vehicle log, vehicle log path)`` — the vehicle is what
    :func:`bind_scores` reads scores off, never the log of record.

    ``coupled_task`` is the COUPLED thin task (``sandbox`` = the owned provider, the
    stock scorer) over exactly the original's ids. ``score_model`` (M1 / M1b) names
    the model ``score_async`` reconstructs the task context with — the stock scorer
    calls no model, and the log header's own model would need the tinycap provider
    and a live proxy. ``max_connections`` / ``sample_shuffle`` reach the M2 vehicle
    (the binding must hold whatever order and concurrency the vehicle ran at)."""
    from inspect_ai.log import write_eval_log

    if mechanism not in C.CPU_VERIFY_MECHANISMS:
        raise CpuVerifyError(f"mechanism {mechanism!r} is not one of {list(C.CPU_VERIFY_MECHANISMS)}")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if mechanism in ("M1_score_async", "M1b_score_async_sandboxenv"):
        from inspect_ai._eval.score import score_async

        scorers = _task_scorers(coupled_task)
        work = original.model_copy(deep=True)
        if mechanism == "M1_score_async":
            scored = await score_async(work, scorers, model=score_model, copy=False)
        else:
            from inspect_ai._eval.task.sandbox import sandboxenv_context
            from inspect_ai.dataset import Sample

            spec = getattr(coupled_task, "sandbox", None)
            if spec is None:
                raise CpuVerifyError("M1b: the coupled task carries no sandbox spec to open per sample")
            tname = task_name or getattr(coupled_task, "name", None) or "bigcodebench"
            samples = list(work.samples or [])

            @contextlib.asynccontextmanager
            async def _with_sandbox(idx: int):
                es = samples[idx]
                ds = Sample(input=es.input, target=es.target, id=es.id, metadata=dict(es.metadata or {}),
                            sandbox=es.sandbox, files=es.files, setup=es.setup)
                async with sandboxenv_context(tname, spec, max_sandboxes, True, ds):
                    yield es

            scored = await score_async(work, scorers, model=score_model, copy=False, samples=_with_sandbox)
        path = log_dir / f"vehicle_{mechanism}.eval"
        write_eval_log(scored, str(path), format="eval")
        return scored, path
    # M2_replay_vehicle: the public-API vehicle
    from inspect_ai import eval_async

    model = replay_model(original)
    kwargs: Dict[str, Any] = {}
    if max_sandboxes is not None:
        kwargs["max_sandboxes"] = int(max_sandboxes)
        kwargs["sandbox_cleanup"] = True
    if max_connections is not None:
        kwargs["max_connections"] = int(max_connections)
    if sample_shuffle is not None:
        kwargs["sample_shuffle"] = sample_shuffle
    logs = await eval_async(coupled_task, model=model, log_dir=str(log_dir), log_format="eval",
                            fail_on_error=True, retry_on_error=0, log_realtime=False, **kwargs)
    vehicle = logs[0]
    return vehicle, Path(str(vehicle.location))


def build_coupled_bcb_task(name: str, *, log_ids: Sequence[str], repo_root: str | Path) -> Any:
    """The COUPLED thin BigCodeBench task (``sandbox`` = the owned provider, the stock
    ``verify`` scorer) over exactly ``log_ids`` (LOG vocabulary) — the leg's builder
    over its VERIFIED pin, the ids cross-checked."""
    from sisteer.capeval.tiers import inspect_served as isv

    if name not in isv.SANDBOXED_TASK_NAMES:
        raise CpuVerifyError(f"{name!r} is not a sandboxed BigCodeBench leg ({list(isv.SANDBOXED_TASK_NAMES)})")
    build = isv.load_repo_task(name)
    root = Path(repo_root)
    pin = C.load_bcb_full_pin(root) if name == C.CONTROL_BIGCODEBENCH_FULL else C.load_bcb_pin(root)
    ids = [str(i) for i in log_ids]
    task = isv._build_bcb_task(build, log_ids=ids, repo_root=root, pin=pin, generation_only=False)
    got = [str(s.id) for s in task.dataset]
    if got != ids:
        raise CpuVerifyError(f"{name}: the coupled task built over ids {got[:5]}… != the sidecar's {ids[:5]}…")
    return task


# --------------------------------------------------------------------------- #
# the verify orchestration                                                      #
# --------------------------------------------------------------------------- #


def one_cell_dir(out_root: str | Path) -> Path:
    """The ONE ``<out_root>/cells/<key>/`` store under a cell root — none or several refuse by name."""
    cells = Path(out_root) / "cells"
    dirs = sorted(p for p in cells.iterdir() if p.is_dir()) if cells.is_dir() else []
    if len(dirs) != 1:
        raise CpuVerifyError(f"{out_root}: expected exactly one cells/<key>/ store, found {[d.name for d in dirs]}")
    return dirs[0]


def _same_path(a: str | Path, b: str | Path) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a) == str(b)


#: the SCORING RECEIPT — ``<row_dir>/generation/<cell>__<task>__<store>.scoring.json``: the identity
#: of ONE attempt's scoring, written after the bind and BEFORE the swap. Once a merged log is on disk, a restart files
#: THAT scoring from the disk log (the receipt's mechanism / fcid / vehicle sha / max_sandboxes go on the verified
#: block) and never re-scores: a sandbox that timed out on the first attempt and passes on the retry would otherwise
#: leave the first attempt's verdicts in the rows the emitter skips and the second's in the log, summary and outcomes.
#: ``verdict_digest`` binds the receipt to the merged log's per-sample verdicts.
SCORING_RECEIPT_FIELDS_V1: Tuple[str, ...] = (
    "task", "cell", "store", "log_sha256", "primary_scorer", "mechanism", "function_call_id", "started_at",
    "max_sandboxes", "n_items", "vehicle_log_sha256", "verdict_digest", "written_at")
#: the scorer identity that EXECUTED the scoring rides the receipt too — a resumed filing
#: (a verifier killed after its swap, finished by the next `verify`) puts it on the verified block exactly as an
#: uninterrupted run would; a receipt a tree before this change wrote is EXACTLY the V1 set (read as None)
SCORING_RECEIPT_FIELDS: Tuple[str, ...] = SCORING_RECEIPT_FIELDS_V1 + ("executed_image", "executed_verify_timeout_s", "binding_version")
SCORING_RECEIPT_FIELD_SETS: Tuple[Tuple[str, ...], ...] = (SCORING_RECEIPT_FIELDS, SCORING_RECEIPT_FIELDS_V1)


def scoring_receipt_path(row_dir: str | Path, cell: str, task: str, store: str) -> Path:
    """``<row_dir>/generation/<cell>__<task>__<store>.scoring.json`` — the attempt's scoring identity."""
    return Path(row_dir) / GENERATION_DIR / f"{cell}__{_task_fs(task)}__{_store_key(store)}.scoring.json"


def verdict_digest(log: Any, *, primary_scorer: str) -> str:
    """sha256 over every sample's ``(id, epoch, scorer, value)`` — the scoring's identity, independent of the bytes
    the scorer's explanation carries (a unittest timing). A sample without the primary scorer's score refuses: the
    log is not a scored log of record."""
    lines: List[str] = []
    for s in sorted(list(getattr(log, "samples", None) or []), key=sample_key):
        scores = getattr(s, "scores", None) or {}
        if primary_scorer not in scores:
            raise CpuVerifyError(f"sample {sample_key(s)} carries no {primary_scorer!r} score — not a scored log of record")
        for name in sorted(scores):
            value = json.dumps(getattr(scores[name], "value", None), sort_keys=True, default=str)
            lines.append(f"{s.id}\t{int(getattr(s, 'epoch', 1) or 1)}\t{name}\t{value}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def write_scoring_receipt(row_dir: str | Path, receipt: Mapping[str, Any]) -> Path:
    """The receipt, EXACTLY :data:`SCORING_RECEIPT_FIELDS`, atomically (``completion.atomic_write_json``)."""
    _exact_keys(receipt, SCORING_RECEIPT_FIELDS, what="scoring receipt")
    task, cell, store = str(receipt["task"]), str(receipt["cell"]), _store_key(str(receipt["store"]))
    return completion.atomic_write_json(scoring_receipt_path(row_dir, cell, task, store), receipt)


def read_scoring_receipt(row_dir: str | Path, cell: str, task: str, store: str) -> Dict[str, Any]:
    """The receipt, held to EXACTLY :data:`SCORING_RECEIPT_FIELDS` and to naming ``(cell, task, store)``; absent
    refuses by name (a merged log on disk without its receipt is an attempt this tree did not write)."""
    p = scoring_receipt_path(row_dir, cell, task, store)
    if not p.is_file():
        raise CpuVerifyError(f"no scoring receipt at {p} — the store's log is a merged (scored) log but no attempt of "
                             "this tree recorded the scoring it holds; refusing to file a scoring of unknown identity "
                             "(never re-scored over it: the rows an earlier attempt filed would be skipped, not compared)")
    doc = _read_json(p, what="scoring receipt")
    if not any(set(doc) == set(fields) for fields in SCORING_RECEIPT_FIELD_SETS):
        _exact_keys(doc, SCORING_RECEIPT_FIELDS, what=f"scoring receipt {p}")
    for k in SCORING_RECEIPT_FIELDS:
        doc.setdefault(k, None)                       # a V1 receipt: the executed identity is unknown
    if doc["task"] != task or doc["cell"] != cell or doc["store"] != store:
        raise CpuVerifyError(f"scoring receipt {p} names ({doc['cell']!r}, {doc['task']!r}, {doc['store']!r}), "
                             f"not ({cell!r}, {task!r}, {store!r})")
    return doc


def row_disagreements(existing_rows: Sequence[Mapping[str, Any]], results: Sequence[Any], *, task: str,
                      is_unscored: Callable[[Any], bool]) -> List[str]:
    """Every way the rows ALREADY in ``rows.jsonl`` for ``task`` disagree with the scoring about to be filed
    (``results`` = the one pass's ``ItemResult`` list): per item the ``acc`` / ``unscored`` row's value, its
    ``scores``, ``sandbox_status`` and ``code_source``; the two aggregate rows' values; a row for an item or a
    metric the scoring does not file. Empty = every pre-existing row is this scoring's, so the emitter's skip of an
    existing key files nothing foreign. The filing FAILS CLOSED on any entry: a mixed store is
    never written."""
    expected: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in results:
        metric = C.METRIC_UNSCORED if r.unscored else C.METRIC_ACC
        facts = dict(getattr(r, "row_facts", None) or {})
        expected[(str(r.item_id), metric)] = {
            "value": (1.0 if r.unscored else float(r.value)),
            "scores": {str(k): (None if is_unscored(v) else v) for k, v in (getattr(r, "scores", None) or {}).items()},
            "sandbox_status": facts.get("sandbox_status"), "code_source": facts.get("code_source")}
    scored = [float(r.value) for r in results if not r.unscored]
    expected[(C.AGG_ITEM_ID, C.METRIC_ACC_AGGREGATE)] = {"value": (sum(scored) / len(scored)) if scored else 0.0}
    expected[(C.AGG_ITEM_ID, C.METRIC_N)] = {"value": float(len(results))}
    problems: List[str] = []
    for row in existing_rows:
        if row.get("task") != task:
            continue
        key = (str(row.get("item_id")), str(row.get("metric")))
        want = expected.get(key)
        if want is None:
            problems.append(f"row {key}: filed by an earlier attempt, this scoring files no such row")
            continue
        have_value = row.get("value")
        if have_value != want["value"]:
            problems.append(f"item {key[0]!r} {key[1]}: an earlier attempt filed {have_value}, this scoring says {want['value']}")
        if key[0] == C.AGG_ITEM_ID:
            continue
        try:
            extra = json.loads(row.get("extra_json") or "{}") if isinstance(row.get("extra_json"), str) else dict(row.get("extra") or {})
        except ValueError:
            problems.append(f"item {key[0]!r}: the earlier attempt's row carries unreadable extra_json")
            continue
        for k in ("scores", "sandbox_status", "code_source"):
            if k in extra and extra[k] != want[k]:
                problems.append(f"item {key[0]!r} {k}: an earlier attempt filed {extra[k]!r}, this scoring says {want[k]!r}")
    return problems


def verify_claim_path(row_dir: str | Path, cell: str, task: str, store: str) -> Path:
    """``<row_dir>/generation/<cell>__<task>__<store>.claim.json`` — which verifier holds this store."""
    return Path(row_dir) / GENERATION_DIR / f"{cell}__{_task_fs(task)}__{_store_key(store)}.claim.json"


def _iso(t: Any) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(t)))
    except (TypeError, ValueError, OverflowError):
        return repr(t)


def read_verify_claim(row_dir: str | Path, cell: str, task: str, store: str) -> Optional[Dict[str, Any]]:
    """The claim, EXACTLY :data:`VERIFY_CLAIM_FIELDS`, or None when no verifier holds the store. A claim that
    exists but cannot be read (another verifier between its create and its write, or a torn file) refuses: it is
    not free."""
    p = verify_claim_path(row_dir, cell, task, store)
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CpuVerifyError(f"the claim {p} exists but is unreadable ({type(e).__name__}: {e}) — another verifier "
                             "is writing it, or it is torn; the store is not free") from None
    if not isinstance(doc, dict):
        raise CpuVerifyError(f"the claim {p} is not a JSON object")
    _exact_keys(doc, VERIFY_CLAIM_FIELDS, what=f"claim {p}")
    return doc


def _held_by_message(held: Mapping[str, Any], *, task: str, cell: str, store: str) -> str:
    return (f"verifier {held['function_call_id']} holds the claim on {task} for {cell} (store {store!r}) since "
            f"{_iso(held['started_at'])} (run {held['run_id']}, host {held['host']}); a resume must join or supersede "
            f"it through the harness ({VERIFY_SUPERSEDES_ENV}=<that fcid>, set by the dispatcher only for a call it "
            "found terminal or lost) — nothing read, nothing filed")


def claim_verifier(row_dir: str | Path, cell: str, task: str, store: str, *, function_call_id: str,
                   run_id: Optional[str], supersedes: Optional[str]) -> Path:
    """Take the store for ``function_call_id`` (the belt against two live verifiers): an ATOMIC CREATE
    (``O_CREAT | O_EXCL``) of the claim. When one exists: naming this fcid it is ours (a re-entry of the same
    call); naming ``supersedes`` it is taken over (the dispatcher found that call terminal or lost) and rewritten
    atomically; naming any other fcid it refuses by name. Returns the claim path."""
    p = verify_claim_path(row_dir, cell, task, store)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"function_call_id": str(function_call_id), "run_id": (str(run_id) if run_id else None),
           "started_at": time.time(), "host": socket.gethostname()}
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        held = read_verify_claim(row_dir, cell, task, store)
        if held is None:
            # created and released between our two looks: the store is free now — take it
            completion.atomic_write_json(p, doc)
            return p
        if str(held["function_call_id"]) == str(function_call_id):
            return p                                  # ours already
        if supersedes and str(supersedes) == str(held["function_call_id"]):
            completion.atomic_write_json(p, doc)
            print(f"[served-cap] verify: superseding verifier {held['function_call_id']} on {task} for {cell} (its "
                  f"claim since {_iso(held['started_at'])}; the dispatcher found that call terminal or lost — "
                  f"{VERIFY_SUPERSEDES_ENV}); this verifier {function_call_id} holds the store now")
            return p
        raise CpuVerifyError(_held_by_message(held, task=task, cell=cell, store=store))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, indent=1, sort_keys=True))
        fh.flush()
        os.fsync(fh.fileno())
    return p


def _check_claim(row_dir: Path, cell: str, task: str, store: str, function_call_id: str, *, before: str) -> None:
    """Re-read the claim immediately before a mutation: it must still name THIS verifier, else refuse before
    anything moves ("superseded by <fcid> — nothing filed"). A released claim names the receipt's verifier (the
    one that took the store and finished) when a receipt exists."""
    held = read_verify_claim(row_dir, cell, task, store)
    if held is not None and str(held["function_call_id"]) == str(function_call_id):
        return
    other = str(held["function_call_id"]) if held is not None else None
    if other is None:
        rp = scoring_receipt_path(row_dir, cell, task, store)
        if rp.is_file():
            try:
                other = str(_read_json(rp, what="scoring receipt").get("function_call_id"))
            except CpuVerifyError:
                other = None
    raise CpuVerifyError(
        f"superseded by {other or 'another verifier'} — the claim on {task} for {cell} (store {store!r}) "
        f"{('is held by ' + other) if held is not None else 'was released (the store was taken over and finished)'}; "
        f"this verifier ({function_call_id}) stops before {before}: nothing filed")


def _resolve_run_id(run_id: Optional[str], *, marker: Mapping[str, Any], row_dir: Path
                    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[scope.Owner]]:
    """``run_id`` from the argument, else the environment (the harness exports the owner env; there is no
    ``--run-id`` on its argv), else the GPU phase's pointer named by the marker's ``generation_fcid``. Returns
    ``(run_id, the GPU pointer or None, the environment's owner or None)``."""
    env_owner = scope.owner_from_env()
    gen_pointer: Optional[Dict[str, Any]] = None
    if marker["generation_fcid"]:
        gp = generation_pointer_path(row_dir, str(marker["generation_fcid"]))
        if gp.is_file():
            gen_pointer = _read_json(gp, what="the GPU phase's attempt pointer")
    if run_id is None:
        run_id = env_owner.run_id if env_owner is not None else (str(gen_pointer["run_id"]) if gen_pointer else None)
    if not run_id:
        raise CpuVerifyError("no run id for the verifier's owner tuple: pass --run-id, export the TINYCAP_OWNER_* "
                             "environment, or leave the GPU phase's attempt pointer under the row dir")
    return str(run_id), gen_pointer, env_owner


def _declare_verifier(*, row_dir: Path, cell: str, function_call_id: str, run_id: str,
                      gen_pointer: Optional[Mapping[str, Any]], env_owner: Optional[scope.Owner],
                      manifest: Mapping[str, Any], sidecar: Mapping[str, Any],
                      tree: Optional[str | Path], out_root: Path, controls: Sequence[str], name: str
                      ) -> Tuple[scope.Owner, Path, str, str, Path, Optional[str]]:
    """Step 3 of :func:`verify_store`: the owner ``(run_id, cell, function_call_id)`` (``run_id`` resolved by
    :func:`_resolve_run_id`), the verifier's tree READ off ``--tree``'s ``TREE_SHA`` stamp and RECORDED beside the
    store's producer tree (never held equal — tinycap-verify-rate-limit, 2026-09-30: before, a stamp that was not the
    producer's refused, so a row generated at one landed tree could not be verified from the next; the binding the
    filing rests on is the generation ``.eval``'s sha, ``(id, epoch)``, the object digests and the pinned scorer image,
    none of which is the tree), the verifier's pointer written (``phase verify``) and the owner exported for the
    sandbox provider. Returns ``(owner, pointer path, pointer name, generation tree sha, record path, verifier tree
    sha or None)``."""
    owner = scope.Owner(str(run_id), str(cell), str(function_call_id))
    if env_owner is not None and env_owner != owner:
        raise CpuVerifyError(f"the environment declares owner {env_owner}, this verifier is {owner} — a wiring error; "
                             "the sandbox owner must be the verifier's own tuple")
    tree_sha = str((gen_pointer or {}).get("tree_sha") or (manifest.get(C.MANIFEST_KEY_PRODUCER) or {}).get("tree_sha") or "")
    verifier_tree_sha: Optional[str] = None
    if tree is not None:
        stamp = Path(tree) / C.TREE_SHA_FILENAME
        if stamp.is_file():
            verifier_tree_sha = stamp.read_text(encoding="utf-8").strip() or None
            if tree_sha and verifier_tree_sha and not (verifier_tree_sha.startswith(tree_sha) or tree_sha.startswith(verifier_tree_sha)):
                print(f"[served-cap] verify: this verifier runs tree {verifier_tree_sha[:12]} ({tree}); the store was generated "
                      f"at {tree_sha[:12]} — recorded as verification_pass.verifier_tree_sha / generation_tree_sha (the filing "
                      "binds on the generation log's sha, (id, epoch) and the object digests, not on the tree)")
    pointer_name = str((gen_pointer or {}).get("name") or f"{C.VERIFICATION_PHASE_VERIFY}:{cell}")
    pointer_model = str((gen_pointer or {}).get("model") or manifest.get("model_id") or "")
    pointer_mode = str((gen_pointer or {}).get("mode") or f"thinking_{sidecar.get('thinking') or 'off'}")
    # the provider's durable record: the path the harness exported for this verifier process when it did
    # (`scope.owner_env(owner, <row dir>/watch/sandboxes_<fcid>__<cell>.jsonl)`), else the same spelling here
    record_path = Path(os.environ.get(scope.SANDBOX_RECORD_ENV) or verify_sandbox_record_path(row_dir, function_call_id, cell))
    pointer = write_verify_pointer(row_dir, owner=owner, name=pointer_name, model=pointer_model, mode=pointer_mode,
                                   out_dir=str(out_root), controls=[str(c) for c in (controls or [name])],
                                   tree_sha=tree_sha, sandbox_record=record_path)
    os.environ.update(scope.owner_env(owner, record_path))
    print(f"[served-cap] verify: owner {owner} declared; pointer {pointer}; sandbox record {record_path}")
    return owner, pointer, pointer_name, tree_sha, record_path, verifier_tree_sha


def _finish_tail(*, out_root: Path, cell_dir: Path, limits: Mapping[str, Any], marker_path: Path, pointer_name: str,
                 tree_sha: str, verified_by: str, claim_path: Optional[Path] = None,
                 standup_id: Optional[str] = None) -> Tuple[Optional[Path], str]:
    """The last steps of :func:`verify_store`, each idempotent so a verifier that died inside them is finished
    by the next one: ``controls_summary.json`` merged when the run had no limits (a byte-identical
    re-offer from the standing store is accepted), the ``RESULT`` line appended unless the store's last line
    already IS this verification's, then the marker removed (so a store with no marker has its COMPLETE line by
    construction), then the CLAIM released — LAST of all: a completed store has no claim. Returns ``(controls
    summary path, the RESULT line)``. ``standup_id`` (the manifest's ``row_engine.standup_id``; the registry's
    landing-2 refusal, 2026-09-30): the re-stamp carries ``standup=<id>`` like the row runner's earlier line — the
    registry reads the LAST line's ``standup=`` and refuses a store whose last line disagrees with (or, before this,
    lacked) the manifest's — and ``verified_by=`` stays the LAST token (the idempotence check below reads it)."""
    merged_summary = completion.merge_controls_summary(cell_dir) if not limits else None
    _step("controls_merged")
    extra: Dict[str, str] = {}
    if standup_id:
        extra["standup"] = str(standup_id)
    extra["verified_by"] = str(verified_by)
    line = result_line(name=pointer_name, status=REC_STATUS_OK, tree_sha=tree_sha or None, extra=extra)
    last = last_result_line(out_root)
    if last is not None and f" result={C.RESULT_COMPLETE} " in f"{last} " and last.endswith(f" verified_by={verified_by}"):
        line = last                                   # an earlier attempt appended it before it died
    else:
        append_result(out_root, line)
    _step("result_appended")
    marker_path.unlink(missing_ok=True)
    _step("marker_removed")
    if claim_path is not None:
        claim_path.unlink(missing_ok=True)
    _step("claim_released")
    return merged_summary, line


def verify_store(*, out_root: str | Path, cell: str, task: str, row_dir: str | Path,
                 row_engine_record: str | Path, mechanism: str, function_call_id: str,
                 run_id: Optional[str] = None, max_sandboxes: Optional[int] = None,
                 tree: Optional[str | Path] = None, controls: Sequence[str] = (), repo_root: str | Path,
                 quality_fn: Optional[Callable[[Mapping[str, Any]], Dict[str, Any]]] = None,
                 supersedes: Optional[str] = None) -> Dict[str, Any]:
    """``served_capability_run.py verify`` in one function, every
    refusal a :class:`CpuVerifyError` naming what departs, BEFORE any filing.

    ONE verifier at a time on a store (a belt; the harness's rule is
    to JOIN a live call): after step 2 and BEFORE any read of the log this verifier
    CLAIMS the store (:func:`claim_verifier`; :data:`VERIFY_CLAIM_FIELDS`) — a claim
    naming another fcid refuses by name unless ``supersedes`` (default: the
    :data:`VERIFY_SUPERSEDES_ENV` environment, set by the dispatcher only for a call it
    found terminal or lost) names exactly that fcid; the claim is re-read before the
    swap, before the filing and before the manifest / RESULT / marker tail
    (:func:`_check_claim` — "superseded by <fcid> — nothing filed"), and released
    LAST, after the marker. The already-verified and tail-completion paths honour the
    same check.

    1. the row-engine record (``contracts.read_row_engine_record``) — its ``row_dir``
       is ``row_dir``; the store is the ONE ``<out_root>/cells/<key>/``; its manifest
       names ``cell``, carries the record's ``row_engine`` block (every field but
       ``verification_pass`` equal) with ``verification_pass`` PENDING naming ``task``
       — or VERIFIED, the door's other side: with no marker left the
       store is done and nothing moves (the report reads ``already_verified``); with
       the marker still standing an earlier verifier died between its manifest write
       and its marker removal, and only the tail below is completed;
    2. the sidecar and the pending marker (exact keys, agreeing, naming the log under
       the store's task log dir); the sidecar's route is the manifest's virtual model;
       the log's PROVENANCE: the store's log is the log the GPU phase left
       (the sidecar's ``log_sha256``) — or, after an earlier attempt's swap, that
       attempt's merged log while the archive holds the original (its sha the
       sidecar's), in which case that attempt's SCORING is filed from the merged log
       on disk under its receipt (step 4', below) and nothing is re-scored;
       anything else is a torn or foreign store and refuses;
    3. the owner ``(run_id, cell, function_call_id)`` — ``run_id`` from the argument,
       else the environment (the harness exports the owner env; there is no ``--run-id``
       on its argv), else the GPU phase's pointer named by the marker's
       ``generation_fcid`` — the verifier's pointer is written (``phase verify``) and
       the owner exported for the sandbox provider (the environment's
       ``TINYCAP_SANDBOX_RECORD`` kept when the harness set one);
    4. the unscored log (status success, no score on any sample, the sidecar's ids
       and model key); the coupled task over its ids; the plan from the sidecar's
       facts (``inspect_served.plan_from_facts``) — or, 4', when an earlier attempt's
       merged log is on disk: its receipt (:data:`SCORING_RECEIPT_FIELDS`; absent
       refuses), the merged log read from disk and held to the receipt's
       ``verdict_digest``, the sidecar's ids and model key, the plan's count; the
       receipt's mechanism / fcid / start / ``max_sandboxes`` / vehicle sha become the
       scoring of record and steps 5 and 7 are skipped;
    5. the mechanism (``max_sandboxes`` = the argument, else ``min(concurrency, 8)``);
       the BINDING (:func:`bind_scores`) — the merged log held in memory;
    6. over the merged log, before the swap, everything that can still refuse:
       ``results_from_log(merged, target_model=<sidecar target_model>)``, the coupled
       path's window guard (``assert_pass_fits_window`` with the sidecar's
       ``max_model_len``), the plan's item count, the sidecar's
       ``n_requests`` against the items + its retried count, the manifest's
       ``row_constants``, and — fail closed against a mixed store —
       every row ``rows.jsonl`` already holds for the task held equal to this scoring
       (:func:`row_disagreements`; any disagreement refuses naming the item);
    7. the scoring receipt (durable before the swap), the archive + the merged log
       in place (atomic), re-read;
    8. the SAME filing tail (``inspect_served._file_task_results`` — idempotent: the
       emitter dedupes rows already in ``rows.jsonl``, the summary / outcome record /
       done marker are overwritten), ``gen_stats.json``, the manifest's ``quality``
       (``quality_fn``, the driver's ``_cell_quality``) and its ``verification_pass``
       -> the verified block (naming the verifier whose SCORING it records), then
       :func:`_finish_tail`: ``controls_summary.json`` merged when the run had no
       limits, the RESULT line (``verified_by`` = that verifier), the marker removed
       LAST.

    Every store-mutating step reports to :data:`VERIFY_STEP_HOOK` (:data:`VERIFY_STEPS`)
    once done; the exec-spill is installed on the owned provider before any
    sandbox runs (every mechanism but M1, which opens none); the tests stand in for
    the task builder and the mechanism by patching this module's names."""
    from inspect_ai.log import read_eval_log

    from sisteer.capeval import store as cs
    from sisteer.capeval.tiers import inspect_served as isv
    from sisteer.capeval.tiers import tier0

    out_root, row_dir = Path(out_root), Path(row_dir)
    store_key = _store_key(out_root.name)
    started_at = time.time()
    if supersedes is None:
        supersedes = os.environ.get(VERIFY_SUPERSEDES_ENV) or None
    if mechanism not in C.CPU_VERIFY_MECHANISMS:
        raise CpuVerifyError(f"--mechanism {mechanism!r} is not one of {list(C.CPU_VERIFY_MECHANISMS)}")
    name = isv._task_name(task)
    if name not in isv.SANDBOXED_TASK_NAMES:
        raise CpuVerifyError(f"{task} is not a sandboxed BigCodeBench leg ({list(isv.SANDBOXED_TASK_NAMES)}) — the "
                             "verify phase files BigCodeBench alone")
    # 1. the record, the store, the manifest
    try:
        record = C.read_row_engine_record(row_engine_record)
    except C.RowEngineRecordError as e:
        raise CpuVerifyError(f"{C.FLAG_ROW_ENGINE_RECORD}: {e}") from None
    if not _same_path(record["row_dir"], row_dir):
        raise CpuVerifyError(f"the row-engine record's row_dir {record['row_dir']!r} is not --row-dir {str(row_dir)!r}")
    cell_dir = one_cell_dir(out_root)
    manifest = _read_json(cell_dir / "manifest.json", what="the store manifest")
    if manifest.get("cell_id") != cell:
        raise CpuVerifyError(f"the store {cell_dir} is cell {manifest.get('cell_id')!r}, not --cell {cell!r}")
    block = manifest.get(C.MANIFEST_KEY_ROW_ENGINE)
    if not isinstance(block, Mapping):
        raise CpuVerifyError(f"the store manifest carries no {C.MANIFEST_KEY_ROW_ENGINE!r} block — not a row-engine store")
    for k in C.ROW_ENGINE_MANIFEST_FIELDS:
        if k == "verification_pass":
            continue
        if block.get(k) != record[k]:
            raise CpuVerifyError(f"the store manifest's row_engine.{k} {block.get(k)!r} != the record's {record[k]!r} — "
                                 "the record is not this store's")
    vp = block.get("verification_pass")
    marker_path = pending_marker_path(row_dir, cell, task, store_key)
    claim_path = verify_claim_path(row_dir, cell, task, store_key)
    if isinstance(vp, Mapping) and vp.get("state") == C.VERIFICATION_PASS_VERIFIED:
        # the door's other side: this store's BigCodeBench task was verified already
        keys_problem = C.verified_block_keys_problem(vp)
        if keys_problem is not None:
            raise CpuVerifyError(f"the store's verification_pass reads verified but {keys_problem} — a torn verified block; refusing")
        report = {"task": task, "cell": cell, "mechanism": str(vp["mechanism"]), "n_items": int(vp["n_items"]),
                  "max_sandboxes": int(vp["max_sandboxes"]), "summary": completion.read_task_summary(cell_dir, task),
                  "digests": None, "vehicle_log_sha256": str(vp["vehicle_log_sha256"]), "pointer": None,
                  "controls_summary": None, "result_line": last_result_line(out_root), "verification_pass": dict(vp),
                  "already_verified": True, "tail_completed": False}
        if not marker_path.is_file():
            # the same single-writer check: a claim by another live verifier is not ours to clear; ours, or one
            # the dispatcher superseded, is stale (the tail passed the marker and died before the release) — cleared,
            # the one row-dir write of this path (a completed store has no claim)
            held = read_verify_claim(row_dir, cell, task, store_key)
            if held is not None and str(held["function_call_id"]) not in (str(function_call_id), str(supersedes or "")):
                raise CpuVerifyError(_held_by_message(held, task=task, cell=cell, store=store_key)
                                     + " (the store reads verified; that verifier is finishing or died before its release)")
            if held is not None:
                claim_path.unlink(missing_ok=True)
            print(f"[served-cap] verify: {task} for {cell} is already verified (by {vp['function_call_id']} via "
                  f"{vp['mechanism']} over {vp['n_items']} items); nothing to do — the store is untouched")
            return report
        # the marker still stands: an earlier verifier died after its manifest write; finish ITS tail (the
        # verified block is that verifier's, so is the RESULT line's verified_by) — no scoring, no sandbox
        sidecar = read_generation_sidecar(row_dir, cell, task, store_key)
        marker = read_pending_marker(row_dir, cell, task, store_key)
        _check_marker_agrees(sidecar, marker)
        run_id, gen_pointer, env_owner = _resolve_run_id(run_id, marker=marker, row_dir=row_dir)
        claim_verifier(row_dir, cell, task, store_key, function_call_id=function_call_id, run_id=run_id, supersedes=supersedes)
        _owner, pointer, pointer_name, tree_sha, _rec, _verifier_tree = _declare_verifier(
            row_dir=row_dir, cell=cell, function_call_id=function_call_id, run_id=run_id, gen_pointer=gen_pointer,
            env_owner=env_owner, manifest=manifest, sidecar=sidecar, tree=tree, out_root=out_root, controls=controls,
            name=name)
        print(f"[served-cap] verify: {task} for {cell} was verified by {vp['function_call_id']} whose run died before "
              f"removing the marker — completing its tail (controls summary, RESULT, marker); nothing re-scored")
        _check_claim(row_dir, cell, task, store_key, function_call_id, before="the tail")
        merged_summary, line = _finish_tail(out_root=out_root, cell_dir=cell_dir, limits=dict(sidecar["limits"] or {}),
                                            marker_path=marker_path, pointer_name=pointer_name, tree_sha=tree_sha,
                                            verified_by=str(vp["function_call_id"]), claim_path=claim_path,
                                            standup_id=(str(block.get("standup_id")) if block.get("standup_id") else None))
        report.update(pointer=str(pointer), controls_summary=(str(merged_summary) if merged_summary else None),
                      result_line=line, tail_completed=True)
        return report
    if not isinstance(vp, Mapping) or vp.get("state") != C.VERIFICATION_PASS_PENDING:
        raise CpuVerifyError(f"the store's verification_pass is {vp!r}, not a PENDING block — nothing to verify "
                             "(never generated in the two-phase mode)")
    if list(vp.get("tasks") or []) != [task]:
        raise CpuVerifyError(f"the store's pending verification_pass names tasks {vp.get('tasks')!r}; the verify phase "
                             f"files exactly {task!r} (the verified block records ONE task)")
    # 2. the sidecar and the marker (keyed by cell, task AND this store's basename)
    sidecar = read_generation_sidecar(row_dir, cell, task, store_key)
    marker = read_pending_marker(row_dir, cell, task, store_key)
    _check_marker_agrees(sidecar, marker)
    vmodel = manifest.get("virtual_model")
    if sidecar["route_virtual_model"] is not None and vmodel is not None and sidecar["route_virtual_model"] != vmodel:
        raise CpuVerifyError(f"the sidecar's route_virtual_model {sidecar['route_virtual_model']!r} is not the store's "
                             f"served virtual_model {vmodel!r} — the sidecar is another cell's")
    log_path = Path(str(marker["log"]))
    want_dir = C.inspect_log_dir(cell_dir, task, 0)
    if not log_path.is_file():
        raise CpuVerifyError(f"the pending marker names the log {log_path}, which is not a file")
    if not _same_path(log_path.parent, want_dir):
        raise CpuVerifyError(f"the pending marker's log {log_path} is not under the store's task log dir {want_dir}")
    version = sidecar.get("inspect_ai_version")
    have = inspect_ai_version()
    if version and have and str(version) != have:
        raise CpuVerifyError(f"the GPU phase generated under inspect_ai {version}, this verifier runs {have} — the two "
                             "phases must share the pinned client (the log format and the scorer are the pin's)")
    # 2b. the CLAIM — BEFORE any read of the log: one verifier at a time on this store
    run_id, gen_pointer, env_owner = _resolve_run_id(run_id, marker=marker, row_dir=row_dir)
    claim_verifier(row_dir, cell, task, store_key, function_call_id=function_call_id, run_id=run_id, supersedes=supersedes)
    # the log's PROVENANCE: the store's log is the one the GPU phase left, or an earlier verify attempt's
    # MERGED log with the original in the archive (the swap archives first, atomically) — nothing else
    log_sha256 = str(sidecar["log_sha256"])
    archive_path = archived_log_path(row_dir, cell, task, store_key)
    have_sha = sha256_file(log_path)
    resume_scoring = False
    if have_sha != log_sha256:
        if archive_path.is_file() and sha256_file(archive_path) == log_sha256:
            resume_scoring = True
        else:
            raise CpuVerifyError(
                f"the store's log {log_path} (sha256 {have_sha[:12]}) is not the log the GPU phase generated (the "
                f"sidecar's log_sha256 {log_sha256[:12]}) and no archive at {archive_path} holds that log — a torn or "
                "foreign store; refusing (an unfinished earlier verify would have left the original in the archive)")
    # 3. the owner, the pointer, the provider environment (the verifier's tree recorded, the producer's kept)
    _owner, pointer, pointer_name, tree_sha, _rec, verifier_tree_sha = _declare_verifier(
        row_dir=row_dir, cell=cell, function_call_id=function_call_id, run_id=run_id, gen_pointer=gen_pointer,
        env_owner=env_owner, manifest=manifest, sidecar=sidecar, tree=tree, out_root=out_root, controls=controls,
        name=name)
    target_model = str(sidecar["target_model"])
    if resume_scoring:
        # 4'. An earlier attempt SCORED this store and died after its swap. Its merged log on disk
        #     IS the log of record and its receipt names the scoring; that scoring is filed — never re-scored (a
        #     sandbox that failed an item then and passes now would put two attempts' verdicts in one store)
        receipt = read_scoring_receipt(row_dir, cell, task, store_key)
        if str(receipt["log_sha256"]) != log_sha256:
            raise CpuVerifyError(f"the scoring receipt binds another generation (log_sha256 {str(receipt['log_sha256'])[:12]}, "
                                 f"the sidecar's {log_sha256[:12]}); refusing")
        merged = read_eval_log(str(log_path))
        ids = isv.log_sample_ids(merged)
        why = isv.id_multiset_mismatch(ids, list(sidecar["sample_ids"]))
        if why is not None:
            raise CpuVerifyError(f"the merged log's ids are not the sidecar's ({why})")
        eval_model = str(getattr(merged.eval, "model", "") or "")
        if eval_model != target_model:
            raise CpuVerifyError(f"the merged log's model key {eval_model!r} is not the sidecar's target_model {target_model!r}")
        primary_scorer = str(receipt["primary_scorer"])
        digest = verdict_digest(merged, primary_scorer=primary_scorer)
        if digest != str(receipt["verdict_digest"]):
            raise CpuVerifyError(f"the merged log on disk is not the scoring the receipt records (verdict digest "
                                 f"{digest[:12]} != {str(receipt['verdict_digest'])[:12]}) — refusing to file a scoring "
                                 "of unknown identity")
        if int(receipt["n_items"]) != len(ids):
            raise CpuVerifyError(f"the scoring receipt records {receipt['n_items']} items, the merged log holds {len(ids)}")
        coupled_task = build_coupled_bcb_task(name, log_ids=ids, repo_root=Path(repo_root))
        plan = isv.plan_from_facts(sidecar["plan"], task=coupled_task)
        if not plan.generation_only or plan.row_task != task:
            raise CpuVerifyError(f"the sidecar's plan is not a generation-only plan of {task} (generation_only="
                                 f"{plan.generation_only}, row_task={plan.row_task!r})")
        plan = replace(plan, generation_only=False)
        if str(plan.primary_scorer) != primary_scorer:
            raise CpuVerifyError(f"the sidecar's plan names primary scorer {plan.primary_scorer!r}, the receipt {primary_scorer!r}")
        n_expect = plan.n_expected_run
        if n_expect is not None and len(ids) != n_expect:
            raise CpuVerifyError(f"the merged log holds {len(ids)} samples, the plan expects {n_expect}")
        mechanism_of_record, n_sandboxes = str(receipt["mechanism"]), int(receipt["max_sandboxes"])
        vehicle_sha, scoring_fcid = str(receipt["vehicle_log_sha256"]), str(receipt["function_call_id"])
        scoring_started_at = float(receipt["started_at"])
        # nothing executes here: the identity that scored is the RECEIPT's (None on a receipt a tree before it wrote)
        executed_identity = ({"image": receipt["executed_image"], "verify_timeout_s": receipt["executed_verify_timeout_s"],
                              "binding_version": receipt["binding_version"]} if receipt.get("executed_image") else None)
        digests = {"original_sha256": log_sha256, "merged_sha256": have_sha, "archive": str(archive_path), "log": str(log_path)}
        print(f"[served-cap] verify: an earlier verify attempt ({scoring_fcid}, {mechanism_of_record}) scored {log_path} "
              f"and died before filing it whole — resuming the filing of THAT scoring from the merged log on disk "
              "(nothing re-scored; the verified block names the verifier that scored)")
    else:
        # 4. the log, the task, the plan
        original = read_eval_log(str(log_path))
        problems = unscored_problems(original)
        if problems:
            raise CpuVerifyError(f"the retained log {log_path} is not an unscored generation log: " + "; ".join(problems))
        ids = isv.log_sample_ids(original)
        why = isv.id_multiset_mismatch(ids, list(sidecar["sample_ids"]))
        if why is not None:
            raise CpuVerifyError(f"the log's ids are not the sidecar's ({why})")
        eval_model = str(getattr(original.eval, "model", "") or "")
        if eval_model != target_model:
            raise CpuVerifyError(f"the log's model key {eval_model!r} is not the sidecar's target_model {target_model!r}")
        coupled_task = build_coupled_bcb_task(name, log_ids=ids, repo_root=Path(repo_root))
        plan = isv.plan_from_facts(sidecar["plan"], task=coupled_task)
        if not plan.generation_only or plan.row_task != task:
            raise CpuVerifyError(f"the sidecar's plan is not a generation-only plan of {task} (generation_only="
                                 f"{plan.generation_only}, row_task={plan.row_task!r})")
        plan = replace(plan, generation_only=False)
        if plan.primary_scorer is None:
            raise CpuVerifyError(f"the sidecar's plan names no primary scorer for {task}")
        n_expect = plan.n_expected_run
        if n_expect is not None and len(ids) != n_expect:
            raise CpuVerifyError(f"the log holds {len(ids)} samples, the plan expects {n_expect}")
        if mechanism != "M1_score_async":
            # the verifier's sandbox exec rides the spill channel above ARG_MAX, exactly as the coupled path's
            from sisteer.capeval.tiers import sandbox_owned as _SO
            from sisteer.capeval.tinycap import verifier_transport as _VT
            # the ledger in the ROW dir (inode rule 2: the store stays at exactly 40), store-keyed like the markers
            _VT.install_exec_spill(_SO.OwnedModalSandboxEnvironment, ledger_path=_VT.row_spill_ledger_path(row_dir, cell, store_key))
        # 4b. the SCORER IDENTITY: what this tree's coupled task will execute — its compose
        #     image, the verify timeout the builder bakes, the binding contract — must be byte-equal to what the
        #     generation was hashed under (the sidecar plan's sandbox facts); a later tree with another pinned image
        #     would otherwise score under the new image while the block claimed the old one. Refused BEFORE any sandbox
        #     (the marker stands, the store is untouched), the mark the harness names its REC by.
        executed_identity = executed_scorer_identity(coupled_task)
        generated_identity = generation_scorer_identity(plan.sandbox)
        identity_problems = scorer_identity_problems(executed_identity, generated_identity)
        if identity_problems:
            raise CpuVerifyError(f"{C.SCORER_IDENTITY_MISMATCH_MARK}: the scorer this tree would run is not the one the store was "
                                 f"generated under — " + "; ".join(identity_problems) + " (a cross-tree verifier is admitted only "
                                 "when the scorer identity is byte-equal; nothing scored, the marker stands)")
        print(f"[served-cap] verify: scorer identity held equal to the generation's — image {executed_identity['image'][-20:]} "
              f"({executed_identity['image_source']}), verify timeout {executed_identity['verify_timeout_s']} s, binding "
              f"{executed_identity['binding_version']}")
        # 5. the mechanism, the binding
        n_sandboxes = int(max_sandboxes) if max_sandboxes is not None else default_max_sandboxes(int(sidecar["concurrency"]))
        if n_sandboxes < 1:
            raise CpuVerifyError(f"--max-sandboxes must be >= 1, got {max_sandboxes!r}")
        tmp_dir = Path(tempfile.mkdtemp(prefix="tinycap-verify-"))
        try:
            try:
                vehicle, vehicle_path = asyncio.run(run_mechanism(mechanism, original=original, coupled_task=coupled_task,
                                                                  max_sandboxes=n_sandboxes, log_dir=tmp_dir))
            except CpuVerifyError:
                raise
            except Exception as e:  # noqa: BLE001 — the mechanism's own failure, named; nothing filed, the marker stays
                raise CpuVerifyError(f"mechanism {mechanism} failed over {len(ids)} samples: {type(e).__name__}: "
                                     f"{str(e)[:400]}") from e
            vehicle_sha = sha256_file(vehicle_path) if Path(vehicle_path).is_file() else ""
            if not vehicle_sha:
                raise CpuVerifyError(f"the mechanism {mechanism} left no vehicle log at {vehicle_path}")
            merged = bind_scores(original, vehicle, primary_scorer=str(plan.primary_scorer))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        # a full BigCodeBench log parses to hundreds of MB and the verifier's container has 8 GiB: each parsed log is
        # released as soon as nothing reads it (the archive below is a byte copy of the FILE, not of the object)
        del original, vehicle
        mechanism_of_record, scoring_fcid, scoring_started_at = mechanism, str(function_call_id), started_at
        digests = None
        _step("scored")
    # 6. EVERYTHING that can still refuse runs over the merged log IN MEMORY, before the swap: a refusal here
    #    leaves the store's unscored log and the marker exactly as the GPU phase left them. The claim first:
    #    a superseded verifier stops here, before it reads the store's rows against ITS scoring
    _check_claim(row_dir, cell, task, store_key, function_call_id, before="the pre-swap checks")
    per_sample = {0: isv.results_from_log(merged, sample_idx=0, primary_scorer=plan.primary_scorer,
                                          paired_ids=plan.paired_ids, attempt=1, max_tokens_used=plan.max_tokens,
                                          target_model=target_model, **isv._plan_readers(plan))}
    # the one-pass fit: the coupled path runs this guard right after results_from_log — a
    # cap-bound sample the served window clamped BELOW the cap is a false at-cap reading, and one whose prompt
    # was not measured cannot be told from it; the split path runs the SAME guard with the window the sidecar
    # recorded (the BCB OFF cap 4,096 pooled into the 4,608 gen_long window is exactly the case). The coupled
    # path's condition is an attributable target usage (route_virtual_model named); the sidecar's target_model
    # IS that key, held non-empty at generation end
    if target_model:
        isv.assert_pass_fits_window(task, per_sample[0], max_tokens=plan.max_tokens,
                                    max_model_len=sidecar.get("max_model_len"))
    if n_expect is not None and len(per_sample[0]) != n_expect:
        raise CpuVerifyError(f"{task}: {len(per_sample[0])} items != the {n_expect} the plan selected — refusing a "
                             "partial task rather than filing an aggregate over a short set")
    # the request accounting is the GENERATION side's: the coupled tail files n_requests = the pass's items + the
    # truncation retry's requests, and the sidecar carries both facts — held consistent here, so a verified store's
    # summary can never differ from a coupled store's on the field sets or the values
    n_retried_requests = int(sidecar["n_retried_requests"])
    if int(sidecar["n_requests"]) != len(per_sample[0]) + n_retried_requests:
        raise CpuVerifyError(f"{task}: the sidecar's n_requests {sidecar['n_requests']} != the log's {len(per_sample[0])} "
                             f"items + its n_retried_requests {n_retried_requests} — the generation side's request "
                             "accounting does not describe this log; refusing to file it")
    template = manifest.get("row_constants")
    if not isinstance(template, Mapping):
        raise CpuVerifyError("the store manifest carries no row_constants — the rows' constant columns are unknown")
    store = cs.CellStore(out_root, cell_dir.name, row_constants=dict(template))
    # FAIL CLOSED against a mixed store: every row an earlier attempt already filed for this task
    # must be THIS scoring's (the emitter skips an existing key, so a disagreeing row would otherwise stand under
    # this scoring's log, summary and outcomes); named per item, nothing moved
    existing_rows = store.rows_of_task(task)
    if existing_rows:
        problems = row_disagreements(existing_rows, per_sample[0], task=task, is_unscored=isv.is_unscored)
        if problems:
            shown = "; ".join(problems[:8]) + (f"; … {len(problems) - 8} more" if len(problems) > 8 else "")
            raise CpuVerifyError(f"{task}: {len(existing_rows)} row(s) an earlier verify attempt filed disagree with the "
                                 f"scoring about to be filed — {shown}. A store never holds two attempts' verdicts "
                                 "(mixed rows under one log); refusing. Repair = remove the task's rows (and any "
                                 "summary / outcomes / done record) and re-run verify")
        print(f"[served-cap] verify: {len(existing_rows)} row(s) an earlier attempt filed for {task} agree with this "
              "scoring — the filing below completes them")
    if not resume_scoring:
        # the claim re-read immediately BEFORE the swap: superseded -> nothing of ours is written
        _check_claim(row_dir, cell, task, store_key, function_call_id, before="the swap")
        # the scoring's identity, durable BEFORE the swap: a restart files THIS scoring from the merged log, never another
        write_scoring_receipt(row_dir, {
            "task": task, "cell": cell, "store": store_key, "log_sha256": log_sha256,
            "primary_scorer": str(plan.primary_scorer), "mechanism": mechanism_of_record,
            "function_call_id": scoring_fcid, "started_at": float(scoring_started_at), "max_sandboxes": n_sandboxes,
            "n_items": len(ids), "vehicle_log_sha256": vehicle_sha,
            "verdict_digest": verdict_digest(merged, primary_scorer=str(plan.primary_scorer)), "written_at": time.time(),
            "executed_image": executed_identity["image"], "executed_verify_timeout_s": executed_identity["verify_timeout_s"],
            "binding_version": executed_identity["binding_version"]})
        _step("receipt_written")
        # 7. the swap: the original archived (sha verified), the merged log written over it atomically, re-read
        digests = write_merged_log(merged, log_path=log_path, archive_path=archive_path)
        del merged                                  # the on-disk bytes are the log of record from here on
        merged_disk = read_eval_log(str(log_path))
        if isv.id_multiset_mismatch(isv.log_sample_ids(merged_disk), ids) is not None:
            raise CpuVerifyError(f"the merged log on disk does not hold the original's ids — refusing to file it")
        del merged_disk
    else:
        del merged
    # 8. the SAME filing tail (idempotent: rows deduped by the emitter — every existing row proven this scoring's
    #    above — and the records overwritten)
    em = tier0._Emitter(store, dict(template), dict(C.TINYCAP_DOMAIN_OF), C.INSPECT_TASK_VERSION)
    limits = dict(sidecar["limits"] or {})
    _check_claim(row_dir, cell, task, store_key, function_call_id, before="the filing")
    summary = isv._file_task_results(
        plan, store=store, task=task, per_sample=per_sample, em=em, thinking=str(sidecar["thinking"]),
        ident=dict(sidecar["row_identity"] or {}), sampling_ref=sidecar["sampling_ref"],
        reused=int(sidecar["logs_reused"]), n_retried_requests=n_retried_requests, t0=None, model=None,
        route_virtual_model=sidecar["route_virtual_model"], reasoning_parser=sidecar["reasoning_parser"],
        limits=limits, out_of_support=bool(sidecar["out_of_support"]), ledger_path=None, attempt_scope=None,
        generated={"n": int(sidecar["generated_calls"])}, generation=sidecar)
    _step("task_filed")
    finished_at = time.time()
    # gen_stats.json: the task's block becomes its summary, as the coupled driver files it
    gs_path = cell_dir / "gen_stats.json"
    gen_stats = _read_json(gs_path, what="gen_stats.json") if gs_path.is_file() else {}
    gen_stats[task] = summary
    gs_path.write_text(json.dumps(gen_stats, indent=2, default=str))
    _step("gen_stats_written")
    # the manifest: quality re-pooled, verification_pass -> verified; through the store writer (identity re-checked)
    if quality_fn is not None:
        manifest[C.MANIFEST_KEY_QUALITY] = quality_fn(gen_stats)
    block_out = dict(block)
    # the block names the verifier whose SCORING it records (a resumed filing keeps the receipt's fcid and start)
    # (the two tree shas are RECORDED: this filer's `--tree` stamp and the store's producer — a resumed filing names
    #  the filer's tree while `function_call_id` stays the scorer's; neither is a gate)
    block_out["verification_pass"] = verified_block(
        mechanism=mechanism_of_record, function_call_id=scoring_fcid, started_at=scoring_started_at,
        finished_at=finished_at, sandbox_facts=dict(plan.sandbox or C.bcb_sandbox_facts()), max_sandboxes=n_sandboxes,
        n_items=len(ids), vehicle_log_sha256=vehicle_sha, verifier_tree_sha=verifier_tree_sha,
        generation_tree_sha=(tree_sha or None),
        executed_image=(executed_identity["image"] if executed_identity else None),
        executed_verify_timeout_s=(executed_identity["verify_timeout_s"] if executed_identity else None),
        binding_version=(executed_identity["binding_version"] if executed_identity else None))
    manifest[C.MANIFEST_KEY_ROW_ENGINE] = C.validate_row_engine_record(block_out, where="verification_pass rewrite")
    _check_claim(row_dir, cell, task, store_key, function_call_id, before="the manifest / RESULT / marker tail")
    store.write_cell_manifest(manifest)
    _step("manifest_written")
    merged_summary, line = _finish_tail(out_root=out_root, cell_dir=store.cell_dir, limits=limits, marker_path=marker_path,
                                        pointer_name=pointer_name, tree_sha=tree_sha, verified_by=scoring_fcid,
                                        claim_path=claim_path,
                                        standup_id=(str(block.get("standup_id")) if block.get("standup_id") else None))
    print(f"[served-cap] verify: {task} filed for {cell} via {mechanism_of_record} over {len(ids)} items "
          f"(acc {summary.get(C.METRIC_ACC)}); verification_pass verified; marker removed; RESULT: {line}")
    return {"task": task, "cell": cell, "mechanism": mechanism_of_record, "n_items": len(ids), "max_sandboxes": n_sandboxes,
            "summary": summary, "digests": digests, "vehicle_log_sha256": vehicle_sha, "pointer": str(pointer),
            "controls_summary": (str(merged_summary) if merged_summary else None), "result_line": line,
            "verification_pass": manifest[C.MANIFEST_KEY_ROW_ENGINE]["verification_pass"],
            "already_verified": False, "tail_completed": False, "resumed_scoring": resume_scoring,
            "scoring_function_call_id": scoring_fcid}
