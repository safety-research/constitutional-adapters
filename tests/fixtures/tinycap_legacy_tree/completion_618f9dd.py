"""VERBATIM ``src/sisteer/capeval/tinycap/completion.py`` at 618f9dd (tinycap
W6) — the tree the two prompted stores were generated at, and the LAST tree
whose ``merge_controls_summary`` writes ``controls_summary.json`` with NO
``sources`` block (PW7 / f8b541b added the discipline). A test fixture, never
imported by the package: ``tests/test_evalplus_served.py`` assembles a
synthetic staged tree around it (``_legacy_tree``) so the launcher's two-tree
merge (PW10 / E17 A1) is exercised against the REAL pre-discipline code —
its ``complete_task_summaries`` + ``slot_payloads`` emit the payload, and its
``merge_controls_summary`` is what the pre-W10 planner used to run against
the canonical file (the reproduction). Do not modernise it: its value is
being what that tree was.
"""


from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sisteer.capeval.tinycap import contracts as C

__all__ = [
    "CompletionError",
    "active_limits",
    "atomic_write_json",
    "complete_task_summaries",
    "done_marker_path",
    "finish_task",
    "is_done",
    "merge_controls_summary",
    "partial_marker_path",
    "read_task_summary",
    "slot_payloads",
    "task_summary_path",
    "write_done_marker",
    "write_partial_marker",
    "write_task_summary",
]


class CompletionError(RuntimeError):
    """A completion-state refusal (a malformed marker, a done task without
    its summary, a merge without provenance)."""


# --------------------------------------------------------------------------- #
# limits                                                                       #
# --------------------------------------------------------------------------- #


def active_limits(**kinds: int | None) -> dict[str, int]:
    """``{kind: value}`` for the ACTIVE driver limits — the truthy, positive
    values among :data:`contracts.LIMIT_KINDS` (``limit`` / ``inspect_limit``
    / ``max_goals``); ``{}`` when none is active. An unknown kind refuses:
    the set is the identity's vocabulary (``resolved_config.tinycap.limits``)."""
    out: dict[str, int] = {}
    for kind, value in kinds.items():
        if kind not in C.LIMIT_KINDS:
            raise CompletionError(f"unknown limit kind {kind!r}; kinds: {list(C.LIMIT_KINDS)}")
        if value is None or isinstance(value, bool):
            continue
        if int(value) < 0:
            raise CompletionError(f"limit {kind}={value!r} must be >= 0")
        if int(value):
            out[kind] = int(value)
    return out


# --------------------------------------------------------------------------- #
# paths + atomic writers                                                       #
# --------------------------------------------------------------------------- #


def done_marker_path(cell_dir: str | Path, task: str) -> Path:
    return Path(cell_dir) / C.DONE_DIR / C.marker_filename(task)


def partial_marker_path(cell_dir: str | Path, task: str) -> Path:
    return Path(cell_dir) / C.PARTIAL_DIR / C.marker_filename(task)


def task_summary_path(cell_dir: str | Path, task: str) -> Path:
    return Path(cell_dir) / C.SUMMARY_DIR / C.marker_filename(task)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """tmp + fsync + replace, ``allow_nan=False``: a marker's existence must
    mean fully published, and a NaN count is a scorer failure, not a fact.
    The one JSON writer of the tinycap store side (markers, summaries, the
    EvalPlus runner's ``gen_meta`` / results, the ledger's endpoint record)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        blob = json.dumps(dict(payload), indent=1, sort_keys=True, allow_nan=False, default=str)
    except ValueError as e:
        raise CompletionError(f"{path}: non-finite value in payload ({e})") from e
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def _check_marker(task: str, payload: Mapping[str, Any], fields: Sequence[str],
                  kind: str) -> None:
    """Both markers' shared contract: EXACTLY ``fields``, naming ``task``."""
    if set(payload) != set(fields):
        raise CompletionError(
            f"{kind} marker for {task!r} carries {sorted(payload)}, not the contract's "
            f"{sorted(fields)}")
    if payload["task"] != task:
        raise CompletionError(f"{kind} marker task {payload['task']!r} != {task!r}")


def write_done_marker(cell_dir: str | Path, task: str, payload: Mapping[str, Any]) -> Path:
    """``done/<task>.json`` with EXACTLY :data:`contracts.DONE_MARKER_FIELDS`
    (P34 — one schema across lanes). Only :func:`finish_task` should reach
    this on a complete set; it is public for the tests and the orchestrator."""
    _check_marker(task, payload, C.DONE_MARKER_FIELDS, "done")
    return atomic_write_json(done_marker_path(cell_dir, task), payload)


def write_partial_marker(cell_dir: str | Path, task: str, payload: Mapping[str, Any]) -> Path:
    """``partial/<task>.json`` with EXACTLY :data:`contracts.PARTIAL_MARKER_FIELDS`."""
    _check_marker(task, payload, C.PARTIAL_MARKER_FIELDS, "partial")
    if not isinstance(payload["limits"], Mapping):
        raise CompletionError(f"partial marker for {task!r}: limits must be a mapping")
    return atomic_write_json(partial_marker_path(cell_dir, task), payload)


def write_task_summary(cell_dir: str | Path, task: str, payload: Mapping[str, Any]) -> Path:
    """``summary/<task>.json`` — the counts-only payload that becomes the
    task's ``controls_summary.json`` entry once the task is DONE. JSON-clean
    (no NaN) is asserted here; the counts-only vocabulary is asserted by the
    shared writer at merge time (the one place that rule lives)."""
    if not isinstance(payload, Mapping) or not payload:
        raise CompletionError(f"summary for {task!r} must be a non-empty mapping")
    return atomic_write_json(task_summary_path(cell_dir, task), payload)


def read_task_summary(cell_dir: str | Path, task: str) -> dict[str, Any] | None:
    p = task_summary_path(cell_dir, task)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def is_done(cell_dir: str | Path, task: str) -> bool:
    return done_marker_path(cell_dir, task).is_file()


# --------------------------------------------------------------------------- #
# the ONE completion decision                                                  #
# --------------------------------------------------------------------------- #


def finish_task(cell_dir: str | Path, *, task: str, n_items: int,
                n_expected_full: int | None, limits: Mapping[str, int] | None,
                n_rows: int, n_requests: int, n_rejected: int, wall_clock_s: float,
                summary: Mapping[str, Any]) -> dict[str, Any]:
    """Record a finished task: the summary always; then the DONE marker iff
    the run was unlimited (``limits`` empty) and the scored set is the frozen
    full set (``n_items == n_expected_full``, with ``n_expected_full`` known);
    else the PARTIAL marker naming the limits and the full count (R2).

    Returns ``{"complete": bool, "marker": path}``. A done marker that
    already exists for a task that now looks limited, or vice versa, is a
    store mix-up and refuses — the limits are part of the store identity,
    so within one store a task's completion status cannot flip."""
    lim = dict(limits or {})
    for kind in lim:
        if kind not in C.LIMIT_KINDS:
            raise CompletionError(f"unknown limit kind {kind!r}; kinds: {list(C.LIMIT_KINDS)}")
    n_items = int(n_items)
    complete = (not lim and n_expected_full is not None and n_items == int(n_expected_full))
    other = partial_marker_path(cell_dir, task) if complete else done_marker_path(cell_dir, task)
    if other.is_file():
        raise CompletionError(
            f"{task}: a {'partial' if complete else 'done'} marker already exists at {other} "
            f"but this run is {'complete' if complete else 'limited'} — the limits are store "
            "identity, so one store cannot hold both; refusing")
    write_task_summary(cell_dir, task, summary)
    base = {"task": task, "n_items": n_items, "n_rows": int(n_rows),
            "n_requests": int(n_requests), "n_rejected": int(n_rejected),
            "wall_clock_s": float(wall_clock_s), "finished_at": time.time()}
    if complete:
        marker = write_done_marker(cell_dir, task, base)
    else:
        marker = write_partial_marker(
            cell_dir, task,
            {**base, "n_expected_full": (int(n_expected_full) if n_expected_full is not None else None),
             "limits": lim})
    return {"complete": complete, "marker": str(marker)}


# --------------------------------------------------------------------------- #
# controls_summary.json — the merge (R3)                                        #
# --------------------------------------------------------------------------- #


def complete_task_summaries(cell_dir: str | Path) -> dict[str, dict[str, Any]]:
    """``{task: summary payload}`` for every task with a DONE marker under
    ``cell_dir``. The task name is read off the marker (never inverted from
    the file name) and cross-checked against it; a done task without its
    summary file is a torn or foreign store and refuses."""
    cell_dir = Path(cell_dir)
    done_dir = cell_dir / C.DONE_DIR
    out: dict[str, dict[str, Any]] = {}
    if not done_dir.is_dir():
        return out
    for p in sorted(done_dir.glob("*.json")):
        marker = json.loads(p.read_text(encoding="utf-8"))
        task = marker.get("task")
        if not isinstance(task, str) or C.marker_filename(task) != p.name:
            raise CompletionError(f"{p}: done marker names task {task!r}, which does not map to "
                                  f"this file name")
        summary = read_task_summary(cell_dir, task)
        if summary is None:
            raise CompletionError(
                f"{cell_dir}: {task} has a done marker but no {C.SUMMARY_DIR}/ payload — "
                "a store written by a runner that predates the summary discipline, or a torn "
                "write; re-run the task rather than merging a marker without its numbers")
        out[task] = summary
    return out


def slot_payloads(task_payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The two-SI control SLOT entries (``contracts.SLOT_OF_TASK``) derived
    from the DONE tinycap task summaries: today only the standard GSM8K —
    ``inspect/gsm8k`` -> slot ``gsm8k`` (PW3). The slot payload is the
    task summary projected onto ``contracts.SLOT_PAYLOAD_FIELDS_OF[slot]``;
    it carries ``task`` and ``item_set`` beside ``acc``/``n`` so a reader
    can never take it for the retired instrument's payload. A summary that
    lacks a slot field, names another task, or is the paired-200 bridge
    (never the headline) refuses instead of filling the slot."""
    out: dict[str, dict[str, Any]] = {}
    for task, slot in C.SLOT_OF_TASK.items():
        summary = task_payloads.get(task)
        if summary is None:
            continue
        fields = C.SLOT_PAYLOAD_FIELDS_OF[slot]
        missing = [f for f in fields if f not in summary]
        if missing:
            raise CompletionError(
                f"{task}: summary lacks the slot field(s) {missing} required by the "
                f"{slot!r} control slot ({list(fields)}) — a runner that predates the "
                "instrument-naming payload; re-run the task")
        if summary["task"] != task:
            raise CompletionError(
                f"{task}: summary names task {summary['task']!r} — refusing to fill the "
                f"{slot!r} slot from another instrument's numbers")
        # the item set the contracts table says this task produces (only the
        # FULL set fills a slot; the paired bridge never does)
        want_item_set = C.GSM8K_ITEM_SET_OF_TASK.get(task)
        if want_item_set is not None and summary["item_set"] != want_item_set:
            raise CompletionError(
                f"{task}: item_set {summary['item_set']!r} — only the FULL "
                f"({C.GSM8K_ITEM_SET_FULL}) item set fills the {slot!r} slot; the "
                f"{C.GSM8K_ITEM_SET_PAIRED} bridge is a comparison, never the headline")
        out[slot] = {f: summary[f] for f in fields}
    return out


class _CellId:
    """The one attribute ``cell_controls.write_controls_summary`` reads off a
    ``DefenseCell`` — so the orchestrator can merge with the manifest's
    ``cell_id`` alone, no registry needed."""

    __slots__ = ("id",)

    def __init__(self, cell_id: str) -> None:
        self.id = cell_id


def merge_controls_summary(cell_dir: str | Path) -> Path | None:
    """Merge every DONE tinycap task's summary under ``cell_dir`` into
    ``<out_root>/<cell_id>/controls_summary.json`` (``out_root`` =
    ``<cell_dir>/../..``, the ``<sweep>/cells/<key>`` layout) under the cell
    manifest's ``provenance`` block, through the shared atomic
    ``cell_controls.write_controls_summary`` (which refuses a file already
    written under a DIFFERENT provenance). Returns the path, or ``None`` —
    writing nothing — when no task is done (a limited store, or a run that
    has not finished)."""
    cell_dir = Path(cell_dir)
    manifest_path = cell_dir / "manifest.json"
    if not manifest_path.is_file():
        raise CompletionError(f"{cell_dir}: no manifest.json — not a cell store")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cell_id = manifest.get("cell_id")
    provenance = manifest.get(C.MANIFEST_KEY_PROVENANCE)
    if not isinstance(cell_id, str) or not cell_id:
        raise CompletionError(f"{manifest_path}: no cell_id")
    if not isinstance(provenance, Mapping) or not provenance:
        raise CompletionError(
            f"{manifest_path}: no {C.MANIFEST_KEY_PROVENANCE!r} block — the served driver "
            "writes it for every tinycap run (the same controls_provenance stamp as the HF "
            "driver); a summary without provenance is exactly the un-fillable artifact "
            "cell_controls refuses (r1 #4)")
    if cell_dir.parent.name != "cells":
        raise CompletionError(
            f"{cell_dir} is not a <sweep>/cells/<key> cell directory — cannot locate out_root")
    payloads = complete_task_summaries(cell_dir)
    if not payloads:
        return None
    # the two-SI control slots a done task fills (the standard GSM8K -> `gsm8k`)
    payloads.update(slot_payloads(payloads))
    try:
        from sisteer.eval.cell_controls import write_controls_summary
    except ImportError as e:  # the orchestrator image today: no torch
        raise CompletionError(
            "merge_controls_summary needs sisteer.eval.cell_controls (the shared "
            "write_controls_summary), which this environment cannot import "
            f"({type(e).__name__}: {e}). Run the merge from the served driver's environment "
            "(it runs automatically after a complete run), or give the caller's image the "
            "sisteer.eval import chain (torch) — the writer is deliberately not duplicated here") from e
    out_root = cell_dir.parent.parent
    return Path(write_controls_summary(out_root, _CellId(cell_id), payloads, dict(provenance)))
