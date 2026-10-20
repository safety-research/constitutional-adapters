"""Per-task completion state for the tinycap families.

ONE discipline for ``inspect/*`` (``tiers/inspect_served``) and
``evalplus/*`` (``tiers/evalplus_runner``):

* ``<cell_dir>/summary/<task>.json`` — the task's counts-only summary
  payload, written whenever a task FINISHES (complete or limited), so a
  resumed run and the merge below read the same numbers the run printed.
* ``<cell_dir>/done/<task>.json`` — written ONLY when the run was
  unlimited (no :data:`contracts.LIMIT_KINDS` flag active) AND the scored
  item set equals the frozen full set (P34). Its presence is what resume
  and the launcher trust.
* ``<cell_dir>/partial/<task>.json`` — written INSTEAD of the done marker
  by a limited run (R2), naming the active limits and the full count, so a
  ``--limit 5`` smoke can never be mistaken for a full pass and the
  launcher's resume never trusts it.

``controls_summary.json`` (R3) has ONE writer for tinycap tasks:
:func:`merge_controls_summary` — it reads the cell manifest's immutable
``provenance`` block (the same ``cell_controls.controls_provenance`` stamp
the HF driver records), collects every DONE task's summary and merges them
through ``cell_controls.write_controls_summary`` (read-modify-write with
provenance equality asserted, atomic replace). It is idempotent: the served
driver calls it after a complete run (served_ll + Inspect), the EvalPlus
sandbox orchestrator calls it again after conversion and execution land the
``evalplus/*`` done markers, and the payloads already present are simply
re-merged byte-identical. A limited store never has a done marker, so the
merge writes NOTHING for it (and never creates an empty summary). Every
entry it writes also names the STORE it came from (:func:`merge_source`,
recorded in the summary's ``sources`` block — PW7 / E14 A), which is what
keeps two stores of one cell (the same cell at another geometry pass shares
the summary file) from overwriting each other's numbers unnoticed. The pass
a source records is the HASHED one (``resolved_config.tinycap.pass``, PW8 /
E16 A2), cross-checked against the manifest's mutable top-level
``geometry.pass``; a store whose two disagree, or whose identity predates
the pass field, is refused rather than stamped with a pass it may not have run.

THE LAUNCHER'S TWO-TREE MERGE (PW10 / E17 A1). The launcher merges stores
produced by MANY trees, and a tree from before the source discipline has a
``write_controls_summary`` that overlays payloads and rewrites the document
with NO ``sources`` block — run against today's canonical file it replaced a
newer pass's standing value and erased every source (reproduced cross-version
by the served-path owner). So no producing tree's writer ever touches the
canonical file: the PRODUCING tree only EMITS a normalized payload — its own
:func:`complete_task_summaries` + :func:`slot_payloads`, the API every tree
since the summary discipline exposes — to a scratch path
(``contracts.MERGE_PAYLOAD_FIELDS``; :func:`normalized_merge_payload` is this
tree's copy of that recipe), and the CURRENT tree's
:func:`merge_emitted_payload` performs the one atomic, sourced write, building
the source record from the store's manifest itself: the hashed pass when the
identity carries one, else the top-level ``geometry.pass`` marked
``pass_source: legacy_top_level`` (``merge_source(legacy_pass=True)``). A
producing tree that cannot emit (its ``completion`` predates the summary
discipline) is refused, named — never merged through another tree's reading.

PW12 (E19). A ``legacy_top_level`` pass is RECORDED but never ORDERS: the
top-level field is rewritten by every resume and the old driver could relabel
pass-1 rows ``geometry.pass: 2``, so ``cell_controls._pass_decision`` treats
any claim involving one as ambiguous (a differing payload needs
``--replace-source``; an EQUAL payload may still adopt). The launcher's
``served_geometry.json`` is no immutable witness either — it is rewritten
into the same out_dir by every run of the configuration. The writer half also
records WHICH tree wrote (``writer_tree_sha``, attested beside the writer's
imported ``completion.__file__``), and when SEVERAL stores of one plan exactly
witness an unrecorded standing payload, :func:`equal_witness_plan` picks the
adopter BEFORE any write by one recorded deterministic rule
(``contracts.EQUAL_WITNESS_RULE``) and lists the others as ``equal_witnesses``.

Import rule: this module needs no torch, inspect_ai or evalplus —
the sandbox orchestrator (a stdlib + pyyaml image with ``sisteer`` mounted)
imports it through ``evalplus_runner`` for the markers. Only
:func:`merge_controls_summary` reaches ``sisteer.eval.cell_controls`` (which
imports ``sisteer.methods`` → torch), lazily, and refuses with the remedy
when that import is unavailable rather than re-implementing the writer.
"""

from __future__ import annotations

import json
import os
import re
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
    "done_marker_path", "marker_filename",
    "equal_witness_plan",
    "finish_task",
    "geometry_pass",
    "hashed_pass",
    "is_done",
    "merge_controls_summary",
    "merge_emitted_payload",
    "merge_source",
    "normalized_merge_payload",
    "partial_marker_path",
    "read_store_manifest",
    "read_task_outcomes",
    "read_task_summary",
    "slot_payloads",
    "store_is_out_of_support",
    "task_outcomes_path",
    "task_outcomes_reason",
    "task_summary_path",
    "verify_outcomes_against_summary",
    "write_done_marker",
    "write_partial_marker",
    "write_task_outcomes",
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


def marker_filename(task: str) -> str:
    """``contracts.marker_filename`` (``<task with "/" -> "__">.json``) — a pure delegate. The contracts' one rule
    spells every row task, ``inspect/ifeval`` included."""
    return C.marker_filename(task)


def done_marker_path(cell_dir: str | Path, task: str) -> Path:
    return Path(cell_dir) / C.DONE_DIR / marker_filename(task)


def partial_marker_path(cell_dir: str | Path, task: str) -> Path:
    return Path(cell_dir) / C.PARTIAL_DIR / marker_filename(task)


def task_summary_path(cell_dir: str | Path, task: str) -> Path:
    return Path(cell_dir) / C.SUMMARY_DIR / marker_filename(task)


def canonical_json_text(payload: Mapping[str, Any]) -> str:
    """The EXACT text :func:`atomic_write_json` writes for ``payload`` — the
    one ``json.dumps`` call of the tinycap store side, in one place.

    Two callers compare documents in BYTES rather than as parsed dicts, because
    a dict holds ``1 == 1.0`` and ``True == 1`` and a comparison whose job is
    "this changes no number" must see a type that moved (E63): the sandbox's
    re-bind gate (``tinycap_sandbox._json_text``) and the merge writer's
    re-bind predicate (``cell_controls``). They call THIS rather than each
    keeping a copy of the call, so a rendering can never drift from the bytes
    the store actually holds. ``ValueError`` on a non-finite value, which the
    writer turns into a :class:`CompletionError` naming the path."""
    return json.dumps(dict(payload), indent=1, sort_keys=True, allow_nan=False, default=str)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """tmp + fsync + replace, ``allow_nan=False``: a marker's existence must
    mean fully published, and a NaN count is a scorer failure, not a fact.
    The one JSON writer of the tinycap store side (markers, summaries, the
    EvalPlus runner's ``gen_meta`` / results, the ledger's endpoint record);
    the text it writes is :func:`canonical_json_text`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        blob = canonical_json_text(payload)
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
                  kind: str, optional: Sequence[str] = ()) -> None:
    """Both markers' shared contract: EXACTLY ``fields``, plus at most
    ``optional``, naming ``task``.

    The REQUIRED set stays exact (P34 — one schema across lanes, and the
    launcher's store audit compares it). ``optional`` is how a later contract
    adds a fact without marking every marker already on the volume untrusted
    (E55 #2: the EvalPlus samples digest)."""
    extra = sorted(set(payload) - set(fields) - set(optional))
    if set(fields) - set(payload) or extra:
        raise CompletionError(
            f"{kind} marker for {task!r} carries {sorted(payload)}, not the contract's "
            f"{sorted(fields)}" + (f" (optionally {sorted(optional)})" if optional else ""))
    if payload["task"] != task:
        raise CompletionError(f"{kind} marker task {payload['task']!r} != {task!r}")


def _draw_marker_fields() -> tuple[str, ...]:
    """The draw-subset contract's two optional marker keys (``n_draws``, ``n_expected_draws``) — present on
    every marker ``finish_task`` writes for a run that declared ``n_expected_draws``, absent on every other."""
    from sisteer.capeval.tinycap import ifeval_contracts as IC
    return tuple(IC.DONE_MARKER_DRAW_FIELDS)


def write_done_marker(cell_dir: str | Path, task: str, payload: Mapping[str, Any]) -> Path:
    """``done/<task>.json`` with EXACTLY :data:`contracts.DONE_MARKER_FIELDS`
    (P34 — one schema across lanes), optionally plus
    :data:`contracts.DONE_MARKER_OPTIONAL_FIELDS`. Only :func:`finish_task`
    should reach this on a complete set; it is public for the tests, the
    orchestrator, and the orchestrator's additive re-bind mode (E60), which
    re-writes an existing marker with its samples digest and a ``rebound_at``
    stamp and needs exactly this admission check."""
    _check_marker(task, payload, C.DONE_MARKER_FIELDS, "done",
                  (*C.DONE_MARKER_OPTIONAL_FIELDS, *_draw_marker_fields()))
    return atomic_write_json(done_marker_path(cell_dir, task), payload)


def write_partial_marker(cell_dir: str | Path, task: str, payload: Mapping[str, Any]) -> Path:
    """``partial/<task>.json`` with EXACTLY
    :data:`contracts.PARTIAL_MARKER_FIELDS`, optionally plus
    :data:`contracts.DONE_MARKER_SUMMARY_LIFTED_FIELDS` (a limited store's task
    binds its samples the same way a complete one does). NOT the whole optional
    set: ``rebound_at`` is admitted on a DONE marker only, because the re-bind
    mode (E60) targets done markers and nothing else writes the key."""
    _check_marker(task, payload, C.PARTIAL_MARKER_FIELDS, "partial",
                  (*C.DONE_MARKER_SUMMARY_LIFTED_FIELDS, *_draw_marker_fields()))
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


def read_store_manifest(cell_dir: str | Path) -> dict[str, Any] | None:
    """``<cell_dir>/manifest.json`` parsed, or ``None`` when it is absent,
    unreadable or not a JSON object — the ONE reader of a store's OWN manifest
    that the launcher's marker gate, the driver's provenance stamp and the
    analysis share. Never validated here: ``untrusted_store_reason`` is the
    validator."""
    p = Path(cell_dir) / "manifest.json"
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _store_tinycap_block(cell_dir: str | Path) -> Mapping[str, Any] | None:
    """``manifest.tinycap`` of the store's own manifest as a mapping, or ``None``
    when the manifest is absent, unreadable or the block is not a mapping —
    the ONE read the three marker predicates below share."""
    manifest = read_store_manifest(cell_dir)
    tinycap = manifest.get(C.MANIFEST_KEY_TINYCAP) if manifest else None
    return tinycap if isinstance(tinycap, Mapping) else None


def _out_of_support_marker(tinycap: Mapping[str, Any] | None) -> bool:
    return tinycap is not None and tinycap.get(C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING) is True


def _cap_override_marker(tinycap: Mapping[str, Any] | None) -> bool:
    if tinycap is None or C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING in tinycap:
        return False
    caps = tinycap.get(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK)
    return isinstance(caps, Mapping) and len(caps) > 0


def store_is_out_of_support(cell_dir: str | Path) -> bool:
    """The store's own POSITIVE marker: ``manifest.tinycap.out_of_support_thinking``
    is exactly ``True``. Absent, unreadable or any other value -> ``False`` — the
    sidecar requirement is gated on a positive producer marker, never on a
    missing key."""
    return _out_of_support_marker(_store_tinycap_block(cell_dir))


def store_has_cap_override(cell_dir: str | Path) -> bool:
    """The comparator store's own POSITIVE marker:
    ``manifest.tinycap.max_tokens_by_task`` is a NON-EMPTY mapping AND the
    block lacks ``out_of_support_thinking`` — an out-of-support store (which
    carries the cap key beside its marker) is :func:`store_is_out_of_support`'s,
    not this. Absent, unreadable, empty, null or any other shape -> ``False``:
    the discriminator is a present hashed key with a value, never a missing one."""
    return _cap_override_marker(_store_tinycap_block(cell_dir))


def store_requires_outcomes(cell_dir: str | Path) -> bool:
    """Whether a DONE task of this store owes a valid ``outcomes/<task>.json``
    beside its marker: EITHER positive producer marker — an out-of-support
    store (:func:`store_is_out_of_support`) or a comparator store under the
    off-cap override (:func:`store_has_cap_override`), both read off ONE parse
    of the manifest. The ONE predicate the launcher's
    ``invalid_done_marker_reason`` and its pre-standup torn-store check read;
    the driver derives the same scope from its flags (``held_to_record``).
    In-support stores (no key) -> False."""
    tinycap = _store_tinycap_block(cell_dir)
    return _out_of_support_marker(tinycap) or _cap_override_marker(tinycap)


def is_done(cell_dir: str | Path, task: str, *, require_outcomes: bool = False) -> bool:
    """The driver's skip predicate: the done marker is present — and, with
    ``require_outcomes`` (the driver passes ``held_to_record``: either
    override), a valid
    ``outcomes/<task>.json`` sits beside it (:func:`task_outcomes_reason` is
    ``None``). A done marker WITHOUT one is not skipped: the driver rebuilds the
    record from the task's retained ``.eval`` logs, or refuses by name
    (``inspect_served.repair_task_outcomes``) — never a re-run into the store.
    Without the flag: the bare marker test, byte for byte."""
    if not done_marker_path(cell_dir, task).is_file():
        return False
    return not require_outcomes or task_outcomes_reason(cell_dir, task) is None


# --------------------------------------------------------------------------- #
# the all-attempted outcome record (fit-support transfer)                      #
# --------------------------------------------------------------------------- #


def task_outcomes_path(cell_dir: str | Path, task: str) -> Path:
    """``<cell_dir>/outcomes/<task>.json`` (``contracts.TASK_OUTCOMES_DIR``)."""
    return Path(cell_dir) / C.TASK_OUTCOMES_DIR / marker_filename(task)


def _check_outcomes_shape(task: str, outcomes: Mapping[str, Any]) -> None:
    """EXACTLY ``contracts.TASK_OUTCOMES_HEADER_FIELDS`` at the top, EXACTLY
    ``contracts.TASK_OUTCOME_ITEM_FIELDS`` per record (or one of the other admitted
    shapes — each with or without ``TASK_OUTCOME_ITEM_BINDING_FIELDS``), naming ``task``;
    ``(item_id, sample_idx)`` unique, every ``sample_idx`` in
    ``[0, samples_per_item)``, ``len(items) == n_attempted_samples``, distinct
    ``item_id`` count ``== n_attempted_items``. The writer and the reader share
    this, so a record that fails it never reaches the store and one on the
    store that fails it is refused, named."""
    if not isinstance(outcomes, Mapping):
        raise CompletionError(f"outcomes for {task!r} must be a mapping")
    want = set(C.TASK_OUTCOMES_HEADER_FIELDS)
    have = set(outcomes)
    if have != want:
        raise CompletionError(
            f"outcomes for {task!r} carry {sorted(have)}, not the contract's {sorted(want)} "
            f"(missing {sorted(want - have)}, extra {sorted(have - want)})")
    if outcomes["task"] != task:
        raise CompletionError(f"outcomes task {outcomes['task']!r} != {task!r}")
    items = outcomes["items"]
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise CompletionError(f"outcomes for {task!r}: items must be a list of records")
    k = outcomes["samples_per_item"]
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise CompletionError(f"outcomes for {task!r}: samples_per_item {k!r} is not a positive int")
    # EXACTLY one of the admitted per-item shapes, decided by the FIRST record and
    # held by every other — a record mixing two shapes is refused, named. The
    # contracts' TASK_OUTCOME_ITEM_FIELD_SETS lists version 1, version 2 (with
    # `sampling_ref`) AND the IFEval shape; the IFEval shape is TASK-SCOPED here
    # (the IFEval task's items also carry the
    # instrument's row facts — the four metrics, the instruction count, the ids —
    # admitted for THAT task only, so a GSM8K record carrying them is refused;
    # every other task's record is byte-identical).
    # ... and, the llama batch (2026-09-13), the IFEval shape of a NO-SWITCH family (the row facts + the `none`
    # split's close-tag stray): TASK-SCOPED with it — contracts.TASK_OUTCOME_ITEM_FIELD_SETS_IFEVAL names both
    ifeval_shapes = [set(fs) for fs in C.TASK_OUTCOME_ITEM_FIELD_SETS_IFEVAL]
    admitted = [set(fs) for fs in C.TASK_OUTCOME_ITEM_FIELD_SETS if set(fs) not in ifeval_shapes]
    admitted.extend(_task_specific_item_field_sets(task))
    # Every admitted shape ALSO in its BOUND form — the two attempt-binding fields
    # (contracts.TASK_OUTCOME_ITEM_BINDING_FIELDS) the writer stamps on every item of a record whose run recorded
    # request evidence; exact either way, so a record mixing bound and unbound items, or one carrying only one of
    # the pair, is still refused by name. A default-off record matches the unbound shapes exactly as before.
    admitted.extend([s | set(C.TASK_OUTCOME_ITEM_BINDING_FIELDS) for s in list(admitted)])
    item_fields: set[str] | None = None
    seen: set[tuple[str, int]] = set()
    ids: set[str] = set()
    for i, rec in enumerate(items):
        if item_fields is None and isinstance(rec, Mapping) and set(rec) in admitted:
            item_fields = set(rec)
        if not isinstance(rec, Mapping) or set(rec) != item_fields:
            got = sorted(rec) if isinstance(rec, Mapping) else type(rec).__name__
            want = sorted(item_fields) if item_fields is not None else [sorted(fs) for fs in admitted]
            raise CompletionError(
                f"outcomes for {task!r}: record {i} carries {got}, not the contract's "
                f"{want}" + (" (every item of one record shares one shape)" if item_fields is not None and i else ""))
        item_id, sidx = rec["item_id"], rec["sample_idx"]
        if not isinstance(item_id, str) or not item_id:
            raise CompletionError(f"outcomes for {task!r}: record {i} item_id {item_id!r}")
        if isinstance(sidx, bool) or not isinstance(sidx, int) or not 0 <= sidx < k:
            raise CompletionError(
                f"outcomes for {task!r}: record {i} ({item_id}) sample_idx {sidx!r} is not in "
                f"[0, {k}) (samples_per_item)")
        key = (item_id, sidx)
        if key in seen:
            raise CompletionError(
                f"outcomes for {task!r}: duplicate (item_id, sample_idx) {key} — one record per "
                "attempted draw")
        seen.add(key)
        ids.add(item_id)
    n_samples, n_items = outcomes["n_attempted_samples"], outcomes["n_attempted_items"]
    if len(items) != n_samples:
        raise CompletionError(
            f"outcomes for {task!r}: {len(items)} records != n_attempted_samples {n_samples!r}")
    if len(ids) != n_items:
        raise CompletionError(
            f"outcomes for {task!r}: {len(ids)} distinct item ids != n_attempted_items {n_items!r}")


def _task_specific_item_field_sets(task: str) -> list[set[str]]:
    """The extra per-item shapes ONE task admits beside the contract's generic
    ones: the IFEval task's items carry ``ifeval_contracts.IFEVAL_OUTCOME_ITEM_FIELDS``
    (the v2 set + the row facts) or, on a NO-SWITCH family's store (llama),
    ``IFEVAL_OUTCOME_ITEM_FIELDS_NO_SWITCH`` (+ the ``none`` split's close-tag
    stray) — ``IFEVAL_OUTCOME_ITEM_FIELD_SETS``, both. Lazy import:
    ``ifeval_contracts`` imports the contracts, never this module."""
    from sisteer.capeval.tinycap import ifeval_contracts as IC
    return [set(fs) for fs in IC.IFEVAL_OUTCOME_ITEM_FIELD_SETS] if task == IC.IFEVAL_TASK else []


def write_task_outcomes(cell_dir: str | Path, task: str, outcomes: Mapping[str, Any]) -> Path:
    """``outcomes/<task>.json`` — the ALL-ATTEMPTED per-(item, sample) record
    (``contracts.TASK_OUTCOMES_HEADER_FIELDS`` / ``TASK_OUTCOME_ITEM_FIELDS``,
    exact sets asserted), atomic. Only :func:`finish_task` should reach this,
    BETWEEN the summary and the marker; public for the tests."""
    _check_outcomes_shape(task, outcomes)
    return atomic_write_json(task_outcomes_path(cell_dir, task), outcomes)


def read_task_outcomes(cell_dir: str | Path, task: str) -> dict[str, Any] | None:
    """The parsed, SHAPE-CHECKED outcome record, or ``None`` when the file is
    absent. A file that is present but unreadable or off-contract refuses
    (:class:`CompletionError`) — never read around."""
    p = task_outcomes_path(cell_dir, task)
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CompletionError(f"{p}: unreadable ({type(e).__name__}: {str(e)[:120]})") from e
    _check_outcomes_shape(task, doc)
    return doc


def verify_outcomes_against_summary(outcomes: Mapping[str, Any],
                                    summary: Mapping[str, Any], *,
                                    n_draws: int | None = None) -> None:
    """The ONE verifier the writer, both done-task admission sites and the
    analysis share (plan S2.7): the record's shape (:func:`_check_outcomes_shape`)
    and its agreement with the task's summary payload field by field —
    ``n_attempted_items == summary["n"]`` (``contracts.METRIC_N``),
    ``n_attempted_samples == summary["n_scored_samples"] +
    summary["n_unscored_samples"]``, and every same-named field
    (``contracts.TASK_OUTCOMES_SUMMARY_FIELDS``) equal.
    Refuses (:class:`CompletionError`) naming the first field that disagrees
    as ``<field> <outcomes value> != <summary value>``.

    The draw-subset contract: when the summary ALSO carries
    ``n_attempted_samples`` it must equal the record's,
    and ``n_draws`` (the run's own draw count, the marker's ``n_draws``) when
    given must equal both — so the summary, the record and the marker can never
    name three draw counts. A summary without the key and a call without
    ``n_draws`` (every pre-existing caller) verify exactly as before."""
    if not isinstance(summary, Mapping) or not summary:
        raise CompletionError("verify_outcomes_against_summary: summary must be a non-empty mapping")
    task = outcomes.get("task") if isinstance(outcomes, Mapping) else None
    if not isinstance(task, str) or not task:
        raise CompletionError("outcomes carry no task name")
    _check_outcomes_shape(task, outcomes)

    def disagree(field: str, a: Any, b: Any) -> CompletionError:
        return CompletionError(
            f"outcomes/{marker_filename(task)} disagrees with "
            f"summary/{marker_filename(task)}: {field} {a!r} != {b!r}")

    if C.METRIC_N not in summary:
        raise CompletionError(f"summary for {task!r} carries no {C.METRIC_N!r}")
    if outcomes["n_attempted_items"] != summary[C.METRIC_N]:
        raise disagree("n_attempted_items", outcomes["n_attempted_items"], summary[C.METRIC_N])
    for f in ("n_scored_samples", "n_unscored_samples"):
        if f not in summary:
            raise CompletionError(f"summary for {task!r} carries no {f!r}")
    draws = summary["n_scored_samples"] + summary["n_unscored_samples"]
    if outcomes["n_attempted_samples"] != draws:
        raise disagree("n_attempted_samples", outcomes["n_attempted_samples"], draws)
    if "n_attempted_samples" in summary and outcomes["n_attempted_samples"] != summary["n_attempted_samples"]:
        raise disagree("n_attempted_samples", outcomes["n_attempted_samples"], summary["n_attempted_samples"])
    if n_draws is not None and outcomes["n_attempted_samples"] != int(n_draws):
        raise CompletionError(
            f"outcomes/{marker_filename(task)}: n_attempted_samples {outcomes['n_attempted_samples']!r} != the "
            f"run's draw count n_draws {int(n_draws)} (the draw-subset contract, plan §5.4)")
    for f in C.TASK_OUTCOMES_SUMMARY_FIELDS:
        if f not in summary:
            raise CompletionError(f"summary for {task!r} carries no {f!r}")
        if outcomes[f] != summary[f]:
            raise disagree(f, outcomes[f], summary[f])


def task_outcomes_reason(cell_dir: str | Path, task: str) -> str | None:
    """Why ``task``'s outcome record does NOT stand beside its done marker under
    ``cell_dir`` — ``None`` when ``outcomes/<task>.json`` exists, parses, and
    passes :func:`verify_outcomes_against_summary` against
    ``summary/<task>.json``. The ONE reason function BOTH admission sites call
    (plan S2.8): the launcher's ``invalid_done_marker_reason`` (for a store
    :func:`store_requires_outcomes` positively marks — ``out_of_support_thinking``
    or the off-cap override's ``max_tokens_by_task``) and the driver's
    :func:`is_done` (``require_outcomes``). Three reason strings."""
    name = marker_filename(task)
    if not task_outcomes_path(cell_dir, task).is_file():
        return (f"no {C.TASK_OUTCOMES_DIR}/{name} beside the done marker of a store under an "
                f"authorised override ({C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING} or "
                f"{C.IDENTITY_KEY_MAX_TOKENS_BY_TASK}; a torn store; the analysis would refuse it — "
                "the driver rebuilds it from the retained .eval logs, or refuses by name)")
    try:
        outcomes = read_task_outcomes(cell_dir, task)
    except CompletionError as e:
        return f"{C.TASK_OUTCOMES_DIR}/{name} {e}"
    summary = read_task_summary(cell_dir, task)
    if summary is None:
        return (f"{C.TASK_OUTCOMES_DIR}/{name} has no {C.SUMMARY_DIR}/{name} to verify against "
                "(a torn store)")
    try:
        verify_outcomes_against_summary(outcomes, summary)
    except CompletionError as e:
        return str(e)
    return None


# --------------------------------------------------------------------------- #
# the ONE completion decision                                                  #
# --------------------------------------------------------------------------- #


def finish_task(cell_dir: str | Path, *, task: str, n_items: int,
                n_expected_full: int | None, limits: Mapping[str, int] | None,
                n_rows: int, n_requests: int, n_rejected: int, wall_clock_s: float,
                summary: Mapping[str, Any],
                outcomes: Mapping[str, Any] | None = None,
                n_draws: int | None = None,
                n_expected_draws: int | None = None,
                n_planned: int | None = None) -> dict[str, Any]:
    """Record a finished task: the summary always; then the DONE marker iff
    the run was unlimited (``limits`` empty) and the scored set is the frozen
    full set (``n_items == n_expected_full``, with ``n_expected_full`` known);
    else the PARTIAL marker naming the limits and the full count (R2).

    ``outcomes``: the task's all-attempted record
    (``contracts.TASK_OUTCOMES_HEADER_FIELDS``); when given it is verified
    against ``summary`` BEFORE anything is written and filed to
    ``outcomes/<task>.json`` BETWEEN the summary and the marker, so a marker
    this tree writes never exists without its record. Only the Inspect runner
    passes it.

    Returns ``{"complete": bool, "marker": path}``. A done marker that
    already exists for a task that now looks limited, or vice versa, is a
    store mix-up and refuses — the limits are part of the store identity,
    so within one store a task's completion status cannot flip.

    ONE end-to-end DRAW-subset
    contract: ``n_expected_draws`` (the run's expected draw count: ``items x
    trials`` at pass 1, the subset's ``n_draws`` on a rung) makes completeness
    a DRAW fact too — ``complete`` iff also ``n_draws == n_expected_draws`` —
    and the marker records ``n_draws`` + ``n_expected_draws``
    (``ifeval_contracts.DONE_MARKER_DRAW_FIELDS``); ``n_draws`` is also held
    to the outcome record's ``n_attempted_samples``. Every existing caller
    passes neither and writes a byte-identical marker.

    qwen-seedrep-native-battery F-AU: ``n_planned`` (the count the plan
    selected -- the Inspect runner passes ``plan.expected_n``) rides the DONE
    marker as :data:`contracts.DONE_MARKER_N_PLANNED_FIELD` when given, beside
    ``n_items`` (the items FILED, one row per attempted item since F-AU) and
    ``n_rows``; a caller that passes None writes a byte-identical marker. The
    partial marker already names the plan's count as ``n_expected_full``."""
    if n_expected_draws is not None and n_draws is None:
        raise CompletionError(f"{task}: n_expected_draws {int(n_expected_draws)} given without n_draws — the draw-subset "
                              "contract needs the run's own draw count beside the expected one")
    if outcomes is not None:
        verify_outcomes_against_summary(outcomes, summary, n_draws=n_draws)
    lim = dict(limits or {})
    for kind in lim:
        if kind not in C.LIMIT_KINDS:
            raise CompletionError(f"unknown limit kind {kind!r}; kinds: {list(C.LIMIT_KINDS)}")
    n_items = int(n_items)
    complete = (not lim and n_expected_full is not None and n_items == int(n_expected_full)
                and (n_expected_draws is None or int(n_draws) == int(n_expected_draws)))
    other = partial_marker_path(cell_dir, task) if complete else done_marker_path(cell_dir, task)
    if other.is_file():
        raise CompletionError(
            f"{task}: a {'partial' if complete else 'done'} marker already exists at {other} "
            f"but this run is {'complete' if complete else 'limited'} — the limits are store "
            "identity, so one store cannot hold both; refusing")
    write_task_summary(cell_dir, task, summary)
    if outcomes is not None:
        # BETWEEN the summary and the marker (plan S2.7): a marker written by
        # this tree never exists without its outcome record
        write_task_outcomes(cell_dir, task, outcomes)
    base = {"task": task, "n_items": n_items, "n_rows": int(n_rows),
            "n_requests": int(n_requests), "n_rejected": int(n_rejected),
            "wall_clock_s": float(wall_clock_s), "finished_at": time.time()}
    # E55 #2: the marker repeats the summary's samples binding when there is
    # one — LIFTED from the summary rather than passed in, so the two can never
    # name different bytes. A summary without it (any non-EvalPlus task, and
    # every EvalPlus task whose store predates the digest) leaves the marker
    # exactly as it was. Only the LIFTED subset is copied, never every optional
    # key (E60): `rebound_at` is written by the re-bind mode alone, and a run
    # that just FINISHED a task has not re-bound it.
    for f in C.DONE_MARKER_SUMMARY_LIFTED_FIELDS:
        if f in summary:
            base[f] = summary[f]
    if n_expected_draws is not None:
        # the draw-subset contract's two counts, on the marker ONLY when the run declared an expected draw count
        from sisteer.capeval.tinycap import ifeval_contracts as IC
        base[IC.DONE_MARKER_DRAW_FIELDS[0]] = int(n_draws)
        base[IC.DONE_MARKER_DRAW_FIELDS[1]] = int(n_expected_draws)
    if complete:
        if n_planned is not None:
            # F-AU: the plan's count beside the filed count, on the DONE marker only (the partial marker's is `n_expected_full`)
            base[C.DONE_MARKER_N_PLANNED_FIELD] = int(n_planned)
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
        if not isinstance(task, str) or marker_filename(task) != p.name:
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
    ``inspect/gsm8k_paired200`` -> slot ``gsm8k`` (PW3; the PAIRED subset
    since U4(a) / PW17 — the full set ``inspect/gsm8k`` is the in-depth
    ``gsm8k_full`` and fills no slot). The slot payload is the task summary
    projected onto ``contracts.SLOT_PAYLOAD_FIELDS_OF[slot]``; it carries
    ``task`` and ``item_set`` beside ``acc``/``n`` so a reader can never take
    it for another instrument's payload. A summary that lacks a slot field,
    names another task, carries another item set, or fails the slot's
    acceptance rule (``contracts.GSM8K_SLOT_ACCEPTANCE``: the frozen matched
    count, the pinned framework) refuses instead of filling the slot — the
    loader would otherwise file that number as legacy."""
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
        # PAIRED set fills the slot since U4(a); the full set is in-depth)
        want_item_set = C.GSM8K_ITEM_SET_OF_TASK.get(task)
        if want_item_set is not None and summary["item_set"] != want_item_set:
            raise CompletionError(
                f"{task}: item_set {summary['item_set']!r} — only the {want_item_set!r} item "
                f"set fills the {slot!r} slot (the standard GSM8K is the paired subset, U4(a)); "
                f"the {C.GSM8K_ITEM_SET_FULL} set is the in-depth {C.CONTROL_GSM8K_FULL} "
                f"({C.GSM8K_FULL_TASK}), never the headline")
        payload = {f: summary[f] for f in fields}
        if slot == C.GSM8K_SLOT and not C.is_standard_gsm8k_slot_payload(payload):
            have = {k: payload.get(k) for k in C.GSM8K_SLOT_ACCEPTANCE}
            raise CompletionError(
                f"{task}: the projected {slot!r} slot payload {have} is not the standard "
                f"instrument's ({dict(C.GSM8K_SLOT_ACCEPTANCE)}) — refusing to fill the slot "
                "with a number every loader would then file as legacy")
        out[slot] = payload
    return out


class _CellId:
    """The one attribute ``cell_controls.write_controls_summary`` reads off a
    ``DefenseCell`` — so the orchestrator can merge with the manifest's
    ``cell_id`` alone, no registry needed."""

    __slots__ = ("id",)

    def __init__(self, cell_id: str) -> None:
        self.id = cell_id


#: :func:`hashed_pass`'s answer for a manifest whose hashed identity carries
#: NO pass field at all (a store from before PW8) — distinct from ``None``,
#: which is a recorded "hand run, no launcher record".
PASS_UNRECORDED = object()


def _pass_value(value: Any) -> Any:
    """A pass field's value as the contract reads it: an int (never a bool),
    or ``None`` for a hand run; anything else is malformed and refused by
    the caller."""
    return value if value is None or (isinstance(value, int) and not isinstance(value, bool)) else PASS_UNRECORDED


def hashed_pass(manifest: Mapping[str, Any]) -> Any:
    """The geometry pass in the HASHED identity —
    ``resolved_config.tinycap.pass`` (PW8 / E16 A2): an int, ``None`` for a
    hand run that resolved none, or :data:`PASS_UNRECORDED` when the
    identity carries no such field (a store written before the pass was
    identity; nothing in its hash vouches for a pass) or the field is
    malformed."""
    resolved = manifest.get("resolved_config")
    tinycap = resolved.get(C.MANIFEST_KEY_TINYCAP) if isinstance(resolved, Mapping) else None
    if not isinstance(tinycap, Mapping) or C.GEOMETRY_PASS_FIELD not in tinycap:
        return PASS_UNRECORDED
    return _pass_value(tinycap[C.GEOMETRY_PASS_FIELD])


def geometry_pass(manifest: Mapping[str, Any]) -> Any:
    """The manifest's TOP-LEVEL ``geometry.pass`` — what the driver copies
    from the launcher's ``served_geometry.json`` into the served-geometry
    record; NOT identity (the block is rewritten by every resume), so it is
    only ever a cross-check on :func:`hashed_pass`. :data:`PASS_UNRECORDED`
    when the block or field is absent or malformed."""
    geometry = manifest.get(C.MANIFEST_KEY_GEOMETRY)
    if not isinstance(geometry, Mapping) or C.GEOMETRY_PASS_FIELD not in geometry:
        return PASS_UNRECORDED
    return _pass_value(geometry[C.GEOMETRY_PASS_FIELD])


_SHA40 = re.compile(r"^[0-9a-f]{40}$")


def merge_source(cell_dir: str | Path, manifest: Mapping[str, Any], *,
                 legacy_pass: bool = False,
                 writer_tree_sha: str | None = None) -> dict[str, Any]:
    """The store's MERGE SOURCE (``contracts.MERGE_SOURCE_FIELDS``; PW7,
    E14 A) — the directory key, the identity hash the store was written
    under, its geometry pass and the tree that produced it
    (``manifest.producer.tree_sha``; null for a store from before the
    producer discipline). Recorded per entry in ``controls_summary.json``,
    so a number can always name the store it came from and a second store
    cannot silently overwrite it.

    The pass is the HASHED one (PW8 / E16 A2, :func:`hashed_pass`): the
    top-level ``geometry.pass`` is mutable — a resume rewrites the block —
    so a source built from it could stamp rows with a pass they never ran
    under. A manifest whose hashed identity carries no pass field refuses
    (a store from before the pass was identity is merged through the tree
    that produced it, never through this one), and so does a top-level
    ``geometry.pass`` that disagrees with the hashed one (a hand-edited or
    relabelled store).

    ``legacy_pass=True`` (PW10 / E17 A1 — the launcher's two-tree merge,
    which writes for stores of EVERY producing tree while their own tree
    only emits the payload): a hashed identity WITHOUT the pass field falls
    back to the store's own top-level ``geometry.pass`` (``None`` when the
    block records none), and the record carries
    ``pass_source: legacy_top_level`` (``contracts.MERGE_SOURCE_PASS_SOURCE_FIELD``)
    saying so — exactly the pass the store's producing tree recorded for it,
    marked as the weaker reading. The disagreement refusal is unchanged. PW12
    (E19 A1-1): that marked pass carries NO pass-order authority in the
    writer (``cell_controls._pass_decision``) — it is recorded, not trusted.

    ``writer_tree_sha`` (PW12 / E19 A1-2): the full sha of the tree that
    WRITES this record — the launcher's two-tree merge passes its own
    (attested) tree; the driver's in-process merge passes none, its writer
    being the producer. Recorded as ``writer_tree_sha``
    (``contracts.MERGE_SOURCE_WRITER_TREE_SHA_FIELD``); a malformed value
    refuses."""
    cell_dir = Path(cell_dir)
    if writer_tree_sha is not None and not (
            isinstance(writer_tree_sha, str) and _SHA40.match(writer_tree_sha)):
        raise CompletionError(
            f"{cell_dir}: writer_tree_sha must be the full 40-hex commit of the WRITING tree, "
            f"got {writer_tree_sha!r} (PW12)")
    chash = manifest.get("harness_config_hash")
    if not isinstance(chash, str) or not chash:
        raise CompletionError(
            f"{cell_dir}/manifest.json: no harness_config_hash — the merge cannot record which "
            "identity produced these numbers (contracts.MERGE_SOURCE_FIELDS); not a cell store")
    pass_no = hashed_pass(manifest)
    top = geometry_pass(manifest)
    annotation: dict[str, Any] = {}
    if pass_no is PASS_UNRECORDED:
        if not legacy_pass:
            raise CompletionError(
                f"{cell_dir}/manifest.json: the hashed identity (resolved_config."
                f"{C.MANIFEST_KEY_TINYCAP}) carries no {C.GEOMETRY_PASS_FIELD!r} — a store from "
                "before the geometry pass was identity (PW8 / E16 A2), or a malformed field; "
                "nothing in its hash vouches for a pass, so the merge will not record one. Merge "
                "it through the launcher (its producing tree emits, this tree writes — PW10)")
        # the pass ITS tree recorded — the top-level block — marked as such
        pass_no = None if top is PASS_UNRECORDED else top
        annotation = {C.MERGE_SOURCE_PASS_SOURCE_FIELD: C.PASS_SOURCE_LEGACY_TOP_LEVEL}
    elif top is not PASS_UNRECORDED and top != pass_no:
        raise CompletionError(
            f"{cell_dir}/manifest.json: {C.MANIFEST_KEY_GEOMETRY}.{C.GEOMETRY_PASS_FIELD} {top!r} "
            f"disagrees with the hashed identity's {pass_no!r} — the top-level field is not "
            "identity and a resume rewrites it, so the hashed one is the store's pass; a "
            "disagreement is a relabelled or hand-edited manifest, refusing (PW8 / E16 A2)")
    producer = manifest.get(C.MANIFEST_KEY_PRODUCER)
    tree_sha = producer.get("tree_sha") if isinstance(producer, Mapping) else None
    if writer_tree_sha is not None:
        annotation[C.MERGE_SOURCE_WRITER_TREE_SHA_FIELD] = writer_tree_sha
    # P1-1: the store's template-check class rides on its source record, read
    # by the ONE reader at this consuming boundary (sweep is torch-free); a
    # store whose class cannot be read, or whose fit witness / fit_provenance
    # block disagree with it, is not a merge candidate — its numbers cannot
    # say what check they were produced under
    from sisteer.eval import sweep as _sweep
    problems = _sweep.store_provenance_problems(manifest)
    if problems:
        raise CompletionError(
            f"{cell_dir}/manifest.json: {'; '.join(problems)} — the store's provenance does "
            "not read consistently, so it is not a merge candidate (P1-1)")
    annotation[C.MERGE_SOURCE_TEMPLATE_CHECK_FIELD] = _sweep.template_check_of_manifest(manifest)
    return {"cell_key": cell_dir.name, "harness_config_hash": chash, "pass": pass_no,
            "producer_tree_sha": tree_sha if isinstance(tree_sha, str) and tree_sha else None,
            **annotation}


def merge_controls_summary(cell_dir: str | Path, *,
                           replace_source: Mapping[str, str] | None = None) -> Path | None:
    """Merge every DONE tinycap task's summary under ``cell_dir`` into
    ``<out_root>/<cell_id>/controls_summary.json`` (``out_root`` =
    ``<cell_dir>/../..``, the ``<sweep>/cells/<key>`` layout) under the cell
    manifest's ``provenance`` block, through the shared atomic
    ``cell_controls.write_controls_summary`` (which refuses a file already
    written under a DIFFERENT provenance). Returns the path, or ``None`` —
    writing nothing — when no task is done (a limited store, or a run that
    has not finished).

    Every entry written carries this store's :func:`merge_source` (PW7,
    E14 A), so several stores of one cell — the same cell at another
    geometry pass merges into the SAME summary — can never overwrite each
    other unnoticed: a higher pass replaces a lower one and records it as
    superseded, a lower pass leaves the higher one standing, and two stores
    claiming one entry at the same pass REFUSE. ``replace_source`` (the
    launcher's / driver's ``--replace-source``, parsed by
    ``contracts.parse_replace_source``) is ``{entry: winning cell_key}``:
    on that refusal, an entry it names is taken by the named store — this
    one, or the standing one — deliberately and recorded; an entry it does
    not name still refuses (PW8 / E16 A3). PW16 (E24): a re-offer from the
    STANDING store itself whose ``summary/<task>.json`` changed is refused,
    never overlaid, unless ``replace_source`` names that same store for the
    entry — the entry is then refreshed and the refresh recorded.

    This is the DRIVER's in-process merge of its own store; the launcher's
    merge of stores from arbitrary producing trees is
    :func:`merge_emitted_payload` (PW10 / E17 A1)."""
    cell_dir = Path(cell_dir)
    manifest = _merge_target(cell_dir)
    payloads = normalized_merge_payload(cell_dir)["payloads"]
    return _write_merged(cell_dir, manifest, payloads, replace_source=replace_source,
                         legacy_pass=False)


def _merge_target(cell_dir: Path) -> dict[str, Any]:
    """The cell manifest of a store the merge may write for — refusing a
    directory without one, a manifest without ``cell_id`` or the
    ``provenance`` block, and a directory outside the ``<sweep>/cells/<key>``
    layout (the summary lives at ``<cell_dir>/../../<cell_id>/``)."""
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
    return manifest


def normalized_merge_payload(cell_dir: str | Path) -> dict[str, Any]:
    """The NORMALIZED payload of a store for the merge
    (``contracts.MERGE_PAYLOAD_FIELDS``): ``{cell_dir, payloads}`` where
    ``payloads`` is every DONE task's summary (:func:`complete_task_summaries`)
    plus the two-SI slots those tasks fill (:func:`slot_payloads`) — ``{}``
    for a store with nothing done. This is exactly the recipe the launcher's
    two-tree merge (PW10 / E17 A1) runs INSIDE the producing tree
    (``tinycap_served.EMIT_CODE`` inlines the two calls, since a tree older
    than this function still has both of them), so a store's numbers are
    always read by the tree that wrote them, and only the CURRENT tree
    writes the canonical file (:func:`merge_emitted_payload`)."""
    cell_dir = Path(cell_dir)
    payloads = complete_task_summaries(cell_dir)
    if payloads:
        # the two-SI control slots a done task fills (the standard paired GSM8K -> `gsm8k`)
        payloads.update(slot_payloads(payloads))
    return {"cell_dir": str(cell_dir), "payloads": payloads}


def _checked_payloads(cell_dir: Path, payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The ``payloads`` of an EMITTED merge payload, once it is the contract's
    shape (``contracts.MERGE_PAYLOAD_FIELDS``) and describes ``cell_dir``."""
    if not isinstance(payload, Mapping) or set(payload) != set(C.MERGE_PAYLOAD_FIELDS):
        raise CompletionError(
            f"{cell_dir}: the emitted merge payload carries "
            f"{sorted(payload) if isinstance(payload, Mapping) else type(payload).__name__}, not "
            f"the contract's {list(C.MERGE_PAYLOAD_FIELDS)} (contracts.MERGE_PAYLOAD_FIELDS)")
    named = payload["cell_dir"]
    if not isinstance(named, str) or Path(named).resolve() != cell_dir.resolve():
        raise CompletionError(
            f"{cell_dir}: the emitted merge payload describes {named!r}, not this store — "
            "refusing to file another directory's numbers under this store's source")
    payloads = payload["payloads"]
    if not isinstance(payloads, Mapping) or any(
            not isinstance(k, str) or not isinstance(v, Mapping) for k, v in payloads.items()):
        raise CompletionError(
            f"{cell_dir}: the emitted merge payload's 'payloads' must map entry names to "
            "counts-only payload mappings")
    return {k: dict(v) for k, v in payloads.items()}


def _witness_rank(source: Mapping[str, Any]) -> tuple:
    """:data:`contracts.EQUAL_WITNESS_RULE` — a hashed-pass source before a
    ``legacy_top_level`` one, then the lexically smallest ``cell_key``."""
    legacy = source.get(C.MERGE_SOURCE_PASS_SOURCE_FIELD) == C.PASS_SOURCE_LEGACY_TOP_LEVEL
    return (1 if legacy else 0, str(source["cell_key"]))


def equal_witness_plan(emitted: Sequence[tuple[str | Path, Mapping[str, Any]]]
                       ) -> dict[str, dict[str, Any]]:
    """PW12 (E19, PW10-3 constraint ii) — decided BEFORE any write, over EVERY
    store of one merge plan: which store ADOPTS an unrecorded standing entry
    when more than one of them exactly witnesses it. ``emitted`` is
    ``[(cell_dir, emitted payload), ...]`` (the scratch payloads the producing
    trees wrote). Returns, per cell dir (as given), ``{"witnesses": {entry:
    [the OTHER equal stores' four-field records]}, "yield": [entries this store
    leaves to the winner]}``; the writer half passes each store its record
    (:func:`merge_emitted_payload` ``plan=``). The rule
    (``contracts.EQUAL_WITNESS_RULE``): a hashed-pass source before a
    ``legacy_top_level`` one, then the lexically smallest ``cell_key`` — pure,
    so whichever order the stores are then written in, and whether or not a
    later store's OTHER entries fail their merge, the file ends the same. Before
    this, directory order picked the first witness and the attribution stuck
    even when a later equal candidate failed the overall merge. Only
    ADOPTION candidates are planned (an UNRECORDED standing source, or no
    ``sources`` block, with an EQUAL offered payload — equality under the
    writer's own normalization, ``contracts.with_evalplus_metric_def``, so an
    EvalPlus payload naming no definition equals one stamped ``plus_only``,
    PW13); a differing payload, or a recorded standing source, is the writer's
    pass rule / ambiguity as before.
    A store whose manifest or payload is not a merge candidate is left out of
    the plan (the writer refuses it, named, on its own turn); an unreadable
    summary plans nothing for its group (every write into it refuses)."""
    per_dir: dict[str, dict[str, Any]] = {}
    groups: dict[Path, list[tuple[str, dict[str, Any], dict[str, dict[str, Any]]]]] = {}
    for cell_dir, payload in emitted:
        key = str(cell_dir)
        if key in per_dir:
            continue
        per_dir[key] = {"witnesses": {}, "yield": []}
        d = Path(cell_dir)
        try:
            manifest = _merge_target(d)
            payloads = _checked_payloads(d, payload)
            source = merge_source(d, manifest, legacy_pass=True)
        except CompletionError:
            continue        # not a candidate; its own write refuses, named
        summary = d.parent.parent / manifest["cell_id"] / C.CONTROLS_SUMMARY_FILENAME
        groups.setdefault(summary.resolve(), []).append((key, source, payloads))
    for summary, stores in groups.items():
        if len(stores) < 2 or not summary.is_file():
            continue
        try:
            doc = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue        # torn: every write into it refuses; nothing to plan
        controls = doc.get("controls") if isinstance(doc, Mapping) else None
        recorded = doc.get(C.SUMMARY_KEY_SOURCES) if isinstance(doc, Mapping) else None
        controls = controls if isinstance(controls, Mapping) else {}
        recorded = recorded if isinstance(recorded, Mapping) else {}
        unrecorded = dict(C.UNRECORDED_MERGE_SOURCE)
        for entry, standing in controls.items():
            if not C.requires_merge_source(entry, standing):
                continue
            rec = recorded.get(entry)
            core = ({f: rec.get(f) for f in C.MERGE_SOURCE_FIELDS} if isinstance(rec, Mapping)
                    else unrecorded)
            if core != unrecorded:
                continue    # a recorded source: the pass rule's business, not adoption's
            # the SAME normalization the writer compares under (PW13): an absent EvalPlus
            # definition IS plus_only on both sides, so a legacy tree's re-offer of a payload the
            # writer already stamped still witnesses it — and two such stores are still planned
            norm_standing = (C.with_evalplus_metric_def(entry, standing)
                             if isinstance(standing, Mapping) else standing)
            witnesses = [(key, source) for key, source, payloads in stores
                         if entry in payloads
                         and C.with_evalplus_metric_def(entry, payloads[entry]) == norm_standing]
            if len(witnesses) < 2:
                continue
            # P1-1: stores of DIFFERENT template-check classes are never equal
            # witnesses of one number — a skipped check is not equivalent to a
            # performed one, and an unrecorded class equals nothing — so no
            # adoption is planned for the entry; each store's own write then
            # meets the writer's ambiguity refusal and the operator names the
            # winner deliberately (contracts.MERGE_SOURCE_TEMPLATE_CHECK_FIELD)
            classes = {src.get(C.MERGE_SOURCE_TEMPLATE_CHECK_FIELD) for _k, src in witnesses}
            if len(classes) > 1:
                # CONTESTED: recorded on EVERY candidate's plan, not merely left
                # unplanned — an empty plan does not prohibit adoption, and
                # merge_emitted_payload would otherwise let the first store to
                # write adopt by order of arrival (fourth review). The writer
                # half refuses the whole store write for a contested entry
                # unless --replace-source names the winner among these keys.
                by_key = {src["cell_key"]: src.get(C.MERGE_SOURCE_TEMPLATE_CHECK_FIELD)
                          for _k, src in witnesses}
                for k, _src in witnesses:
                    per_dir[k].setdefault("contested", {})[entry] = dict(by_key)
                continue
            witnesses.sort(key=lambda ks: _witness_rank(ks[1]))
            winner_key, _winner = witnesses[0]
            # the four identity fields plus the class the plan grouped on, so the
            # writer can refuse a cross-class record it did not plan (P1-1); the
            # recorded equal-witness entries keep their four-field + digest shape
            losers = [{**{f: src[f] for f in C.MERGE_SOURCE_FIELDS},
                       C.MERGE_SOURCE_TEMPLATE_CHECK_FIELD: src.get(C.MERGE_SOURCE_TEMPLATE_CHECK_FIELD)}
                      for _k, src in witnesses[1:]]
            per_dir[winner_key]["witnesses"][entry] = losers
            for loser_key, _src in witnesses[1:]:
                per_dir[loser_key]["yield"].append(entry)
    return per_dir


def merge_emitted_payload(cell_dir: str | Path, payload: Mapping[str, Any], *,
                          replace_source: Mapping[str, str] | None = None,
                          writer_tree_sha: str | None = None,
                          plan: Mapping[str, Any] | None = None) -> Path | None:
    """PW10 (E17 A1) — the CURRENT tree's half of the launcher's two-tree
    merge: write ``payload`` — the normalized payload the store's PRODUCING
    tree emitted (:func:`normalized_merge_payload`'s shape,
    ``contracts.MERGE_PAYLOAD_FIELDS``; a scratch file, never the canonical
    one) — into ``<out_root>/<cell_id>/controls_summary.json`` through the
    shared sourced writer, exactly as :func:`merge_controls_summary` would
    for this tree's own store. The source record is built HERE from the
    store's manifest (``merge_source(legacy_pass=True)``): the hashed pass
    when the identity carries one, else the top-level ``geometry.pass``
    marked ``pass_source: legacy_top_level``. So a pre-source-discipline
    producing tree can never overlay the file or drop its ``sources`` block —
    the pass rule, the ambiguity refusal, ``replace_source`` and (PW16) the
    standing-source refresh discipline all apply to its numbers like any other
    store's. Returns the path, or ``None`` —
    writing nothing — when the payload names no done task. Refuses a payload
    that is not the contract's shape or describes another directory.

    PW12 (E19): ``writer_tree_sha`` is the WRITING tree's full sha, recorded
    on every source this call writes; ``plan`` is this store's record from
    :func:`equal_witness_plan` (computed over the whole merge plan before any
    write) — the entries it ``yield``s to another equal witness are left out
    of this write, and the ``witnesses`` it wins are recorded beside the
    adopted entry."""
    cell_dir = Path(cell_dir)
    manifest = _merge_target(cell_dir)
    payloads = _checked_payloads(cell_dir, payload)
    witnesses: dict[str, list] = {}
    if plan is not None:
        for entry in plan.get("yield") or ():
            payloads.pop(entry, None)      # an equal witness the rule gave to another store
        witnesses = {k: list(v) for k, v in (plan.get("witnesses") or {}).items()}
        # CONTESTED entries (P1-1, fourth review): identical payloads over an
        # unrecorded standing entry from stores of DIFFERENT template-check
        # classes are never adopted by order of arrival. With no replace_source
        # naming a winner among the candidates this call REFUSES the whole
        # store write — nothing written, the summary byte-identical, in either
        # order; a named winner adopts and every other candidate leaves the
        # entry out of its write.
        for entry, classes in (plan.get("contested") or {}).items():
            named = (replace_source or {}).get(entry)
            if named is None:
                raise CompletionError(
                    f"{cell_dir}: {entry!r} is CONTESTED — stores of different template-check "
                    f"classes offer identical payloads over an unrecorded standing entry "
                    f"({classes}); a skipped template check is not equivalent to a performed one, "
                    "so no store may adopt it by order of arrival. Name the winner with "
                    f"{C.REPLACE_SOURCE_FLAG} {entry}{C.REPLACE_SOURCE_ASSIGN}<cell_key> (one of "
                    f"{sorted(classes)}); nothing written for this store (P1-1)")
            if named not in classes:
                raise CompletionError(
                    f"{cell_dir}: {C.REPLACE_SOURCE_FLAG} names {named!r} for the contested "
                    f"{entry!r}, which is none of the contesting stores {sorted(classes)}; refusing")
            if named != cell_dir.name:
                payloads.pop(entry, None)  # yielded to the named winner
    return _write_merged(cell_dir, manifest, payloads, replace_source=replace_source,
                         legacy_pass=True, writer_tree_sha=writer_tree_sha,
                         equal_witnesses=witnesses)


def _write_merged(cell_dir: Path, manifest: Mapping[str, Any],
                  payloads: Mapping[str, Mapping[str, Any]], *,
                  replace_source: Mapping[str, str] | None, legacy_pass: bool,
                  writer_tree_sha: str | None = None,
                  equal_witnesses: Mapping[str, Sequence[Mapping[str, Any]]] | None = None
                  ) -> Path | None:
    """The ONE write: every entry sourced (:func:`merge_source`), through
    ``cell_controls.write_controls_summary``. ``None`` when nothing is done."""
    if not payloads:
        return None
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
    source = merge_source(cell_dir, manifest, legacy_pass=legacy_pass,
                          writer_tree_sha=writer_tree_sha)
    sources = {k: dict(source) for k, p in payloads.items() if C.requires_merge_source(k, p)}
    return Path(write_controls_summary(out_root, _CellId(manifest["cell_id"]), dict(payloads),
                                       dict(manifest[C.MANIFEST_KEY_PROVENANCE]),
                                       sources=sources, replace_source=dict(replace_source or {}),
                                       equal_witnesses={k: [dict(w) for w in v]
                                                        for k, v in (equal_witnesses or {}).items()}))
