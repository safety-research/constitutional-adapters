#!/usr/bin/env python
"""Truncation ladder for bcb-bfcl-200: the
detector, ``select`` and ``merge`` — PURE (files in, files out; no Modal, no
network, no GPU). Everything it reads is the PRODUCER's own record — the
summary's ``n_at_cap`` / ``n_stop_reason``, the outcome record's per-item
``at_cap`` (``stop_reason == "max_tokens"``), the rows' ``stop_reason`` — and
every reading is cross-checked against the others; a disagreement REFUSES.

Sub-commands::

    detect --rung1 <pulled rung-1 dir> [--cells a,b]
        per (cell, leg): n_at_cap / n_stop_reason from summary/<task>.json,
        cross-checked with outcomes/<task>.json and rows.jsonl; a leg FIRES iff
        ANY cell has n_at_cap / 200 > 0.01 (3/200 fires, 2/200 does not);
        `model_length` (a context overflow, NOT at-cap) counted separately.

    select --rung1 <dir> --out ladder_selection.json
        the detector, then for every FIRED leg the UNION of the capped items
        across both cells (canonical PINNED ids, pin order) with per-source
        provenance (store key, manifest / rows / summary / outcomes sha256,
        the store's hashed identity, the cap, n_at_cap, the capped ids + their
        sha256) and the per-task selection block {ids, n, n_selected,
        ids_sha256, cap_from, cap_to: 16384, max_prompt_tokens_rung1}; the
        file's `selection_sha256` is sha256(canonical_json(selection)) over
        the FULL block — a tampered prompt bound changes the digest. The
        launcher's `--item-selection` consumes this file; the meter
        (`bcb_bfcl_200_meter.py --stage-kind ladder-probe --selection <file>
        --unit probe:<cell> --model <key> --fitted-cell <cell>`, then
        `--stage-kind ladder-score --unit score --certificate <rung_probe.<cell>.json>`
        per fitted cell) reads `n_selected` per fired task.

    merge --rung1 <dir> --rung2 <dir> --selection ladder_selection.json
          --certificate rung_probe.json --out <dir>
        recompute, never trust: re-hash every rung-1 source; RECOMPUTE each
        source's capped ids and the per-task union and require byte equality
        with the selection; the certificate must bind the same selection
        digest and cap map; every rung-2 store must be a validly admitted
        ladder PARTIAL (limits == {"item_selection": 1}, n_items == n,
        n_expected_full == 200, item_selection / cap_overrides_by_task equal,
        outcomes ids == the selection exactly, `served` identical across the
        rung-2 cells and equal to the certificate's accepted rung); rung-2 rows
        take precedence for the selected items (stamped attempt 2 / cap_rung 2;
        their max_tokens_used must equal cap_to); exactly 200 unique items per
        (cell, task) and identical sets across cells; writes
        <out>/<cell>/<task>.rows.jsonl and <out>/merged_summary.json with
        n_at_cap_rung1 / n_at_cap_residual per (cell, task).

Layout of a pulled rung dir (what `bcb_bfcl_200_analysis.py --pull` lays down;
the volume's ``<VOL_GEN_ROOT>/<model>/<cell>/thinking_off/<suffix>/`` per cell)::

    <rung dir>/<cell>/cells/<key>/{manifest.json, rows.jsonl, summary/, outcomes/, done/ | partial/}
    <rung dir>/_ladder/rung_probe.json            (the run-level certificate, rung 2 only)

The two row tasks are ``inspect/bigcodebench_200`` (log item ids "17", pinned
ids "BigCodeBench/17") and ``inspect/bfcl_200`` (gorilla record ids, the same
string in both vocabularies); the pins under ``configs/capeval/tasks/`` are the
id vocabulary and the join, read ONLY through the contracts' VERIFIED
loaders (`contracts.load_bcb_pin` / `load_bfcl_pin`: a pin whose ids no longer
regenerate from its recorded rule refuses).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:      # `python scripts/tinycap_ladder.py` without PYTHONPATH=src
    sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval.manifest import canonical_json   # noqa: E402
from sisteer.capeval.tinycap import completion   # noqa: E402
from sisteer.capeval.tinycap import contracts as C   # noqa: E402
from sisteer.capeval.tinycap import rung_probe as RP   # noqa: E402
from sisteer.capeval.tinycap import widen_roster as W   # noqa: E402

# --------------------------------------------------------------------------- #
# constants                                                                    #
# --------------------------------------------------------------------------- #

LADDER_SCHEMA = 1
#: The frozen full item count of both legs (`BCB_SUBSET_N = BFCL_SUBSET_N = 200`).
FULL_N = 200
#: The truncation rule: a leg fires when MORE than 1 % of a
#: cell's responses stopped at the cap — 3/200 fires, 2/200 does not.
FIRE_THRESHOLD = 0.01
#: The ladder's second rung.
CAP_TO = 16_384
#: The two row tasks (`INSPECT_TASK_NAMES += ("bigcodebench_200", "bfcl_200")`).
BCB_TASK = C.inspect_task("bigcodebench_200")
BFCL_TASK = C.inspect_task("bfcl_200")
LADDER_TASKS: Tuple[str, ...] = (BCB_TASK, BFCL_TASK)
#: the lane roster's leg atoms (`widen_roster.LEGS`, the launcher's `--controls`) → the ladder's row tasks
TASK_OF_LEG: Mapping[str, str] = {C.CONTROL_BIGCODEBENCH_200: BCB_TASK, C.CONTROL_BFCL_200: BFCL_TASK}
#: the rung-1 analysis record `--analyze` writes beside the stores (`bcb_bfcl_200_analysis.ANALYSIS_FILENAME`): its `tasks`
#: are the wave's runnable legs at analysis time
ANALYSIS_FILENAME = "analysis.json"
PIN_REL_OF_TASK: Mapping[str, str] = {BCB_TASK: C.BCB_PIN_PATH, BFCL_TASK: C.BFCL_PIN_PATH}
#: The contracts' VERIFIED pin loaders (seed / n / of, regeneration from the recorded rule).
PIN_LOADER_OF_TASK: Mapping[str, Any] = {BCB_TASK: C.load_bcb_pin, BFCL_TASK: C.load_bfcl_pin}
#: inspect_ai's spelling of the provider's ``finish_reason: "length"`` — the ONE
#: at-cap definition (`inspect_served.STOP_REASON_AT_CAP`; asserted equal below
#: when the provider module is importable, so this script cannot drift from it).
STOP_REASON_AT_CAP = "max_tokens"
#: A CONTEXT overflow — a different failure, reported separately, never at-cap.
STOP_REASON_MODEL_LENGTH = "model_length"
#: The integer-valued limits contract of a selection run.
LIMITS_LADDER: Mapping[str, int] = {"item_selection": 1}
#: The rung-1 ↔ rung-2 identity contract: the hashed `tinycap`
#: blocks may differ in exactly these keys — and `served` only in its window, its
#: S and (the launcher sets the driver's width to the resolved S, by construction)
#: a `concurrency` that EQUALS `max_num_seqs` on each rung and the certified S.
ADMITTED_IDENTITY_DIFFS: Tuple[str, ...] = ("cap_overrides_by_task", "item_selection", "limits", "served")
ADMITTED_SERVED_DIFFS: Tuple[str, ...] = ("max_model_len", "max_num_seqs", "concurrency")
#: The launcher composes `--tool-calls` (and the agent-loop entry) iff BFCL is REQUESTED, so these two
#: served facts follow the legs a configuration files: a rerun that files no BFCL may serve without them.
LEG_DERIVED_SERVED_KEYS: Tuple[str, ...] = ("tool_calls", C.AGENT_LOOP_ENTRY_KEY)
#: bcb-bfcl-200-widen S2 (Slice A's contracts names): the hashed per-task EXECUTED sampling map,
#: present only under the `protocol` mode.
IDENTITY_KEY_SAMPLING_BY_TASK: str = C.IDENTITY_KEY_SAMPLING_BY_TASK
SAMPLING_MODE_PROTOCOL: str = C.SAMPLING_PROTOCOL
#: The normal launch files BOTH legs in ONE configuration per cell, while a rerun of only the
#: fired leg legitimately files one — so these per-task identity maps are compared on the
#: PROJECTION onto the merged tasks (rung 2's tasks must be a subset of rung 1's containing
#: every merged task), and a leg's own hashed fields are required iff rung 2 files that leg.
#: bcb-bfcl-200-widen: `sampling_by_task` joins the maps; the ONE grouped
#: projection rule set is :func:`project_identity`, shared with the analysis.
PROJECTED_IDENTITY_KEYS: Tuple[str, ...] = ("tasks", C.IDENTITY_KEY_THINKING_BY_TASK, C.IDENTITY_KEY_SAMPLES_BY_TASK,
                                            IDENTITY_KEY_SAMPLING_BY_TASK)
#: The hashed `tinycap` keys that are OPTIONAL and CONDITIONAL on the sampling mode: present iff
#: the run is `protocol` (rule 5 — a legacy `card` store pair stays valid; a protocol/card mix refuses).
CONDITIONAL_IDENTITY_KEYS: Mapping[str, str] = {IDENTITY_KEY_SAMPLING_BY_TASK: SAMPLING_MODE_PROTOCOL}


def row_tasks_of(identity: Mapping[str, Any], *, where: str = "identity") -> List[str]:
    """The ROW tasks a store files, from its hashed ``tinycap.tasks`` — which the
    producer writes as CONTROL ATOMS (``["bfcl_200", "bigcodebench_200"]``,
    `served_capability_run`'s ``sorted(atoms)``) — normalised through the
    contracts' canonical mapping (`contracts.TASKS_OF_CONTROL`). The stored
    identity is never rewritten; every projection and membership check here
    compares row names to row names. An entry outside the mapping refuses."""
    atoms = identity.get("tasks")
    if not isinstance(atoms, list) or not atoms:
        raise LadderError(f"{where}: tinycap.tasks {atoms!r} is not the producer's non-empty list of control atoms")
    out: List[str] = []
    for atom in atoms:
        rows = C.TASKS_OF_CONTROL.get(atom)
        if rows is None:
            raise LadderError(f"{where}: tinycap.tasks entry {atom!r} is not a control atom of the contracts "
                              f"(`TASKS_OF_CONTROL`); the producer writes control atoms, e.g. {C.CONTROL_BFCL_200!r}")
        out.extend(rows)
    return out
LEG_IDENTITY_FIELDS: Mapping[str, Tuple[str, ...]] = {BCB_TASK: tuple(C.TINYCAP_IDENTITY_BCB_FIELDS),
                                                       BFCL_TASK: tuple(C.TINYCAP_IDENTITY_BFCL200_FIELDS)}
#: The EXPERIMENT identity a rung-2 store must share with its rung-1 source
#: beyond the hashed `tinycap` block: the model, its revision, the
#: intervention, the served object digests, the producing tree, the sampling /
#: thinking blocks and EvalPlus's sandbox block (null on these runs). Paths into
#: the manifest; `harness_config_hash` and `geometry` move with the ladder by
#: construction and are bound through the `tinycap` contract instead.
RUNG_EXPERIMENT_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("model_id",), ("model_revision",), ("engine",), ("intervention_hash",), ("arm_name",), ("kind",),
    (C.MANIFEST_KEY_PRODUCER, "tree_sha"),
    (C.MANIFEST_KEY_ROUTE, "serving_identity"), (C.MANIFEST_KEY_ROUTE, "tool_calls"),
    (C.MANIFEST_KEY_ROUTE, C.AGENT_LOOP_ENTRY_KEY), (C.MANIFEST_KEY_ROUTE, C.REDUCTION_MODE_FIELD),
    (C.MANIFEST_KEY_ROUTE, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256),
    (C.MANIFEST_KEY_ROUTE, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL),
    (C.MANIFEST_KEY_SANDBOX,))
#: The manifest's `sampling` / `thinking` blocks carry per-task maps beside their scalars; the
#: scalars must be equal across rungs, each map equal on the merged tasks (a single-leg rerun drops
#: the other leg's entry by construction). bcb-bfcl-200-widen: `sampling.by_task` (the protocol
#: mode's executed-vs-protocol record per task) joins; its per-task `source` is PROVENANCE compared
#: per task, and the run-level joined `sampling.source` is EXCLUDED (a derived join over the filed
#: tasks, which a single-leg rerun legitimately changes).
PROJECTED_MANIFEST_BLOCKS: Tuple[Tuple[str, str], ...] = ((C.MANIFEST_KEY_SAMPLING, "samples_by_task"),
                                                          (C.MANIFEST_KEY_SAMPLING, "by_task"),
                                                          (C.MANIFEST_KEY_THINKING, "by_task"))
#: Per manifest block: the scalars EXCLUDED from comparison (derived joins) and the per-task maps
#: that are CONDITIONAL on the sampling mode (present iff protocol).
EXCLUDED_MANIFEST_SCALARS: Mapping[str, Tuple[str, ...]] = {C.MANIFEST_KEY_SAMPLING: ("source",)}
CONDITIONAL_MANIFEST_MAPS: Mapping[Tuple[str, str], str] = {(C.MANIFEST_KEY_SAMPLING, "by_task"): SAMPLING_MODE_PROTOCOL}
#: The certificate's rung sources (`rung_probe.RUNG_SOURCES`).
RUNG_SOURCES: Tuple[str, ...] = ("identity", "probe", "operator")
SEQS_SOURCES: Tuple[str, ...] = ("model_budget", "override")
#: bcb-bfcl-200-widen S6: schema 2 — the `ladder` block (the model's rungs the walk was on) and
#: `reprobed_from` (the re-probe at the wave's common footprint) join the exact key set.
CERTIFICATE_KEYS: Tuple[str, ...] = (
    "schema", "suffix", "accepted", "walked", "tree_sha", "model_slug", "hf", "rev", "cfg_path",
    "cfg_sha256", "object_digests", "selection_sha256", "cap_overrides_by_task", "budget", "ladder",
    "probed_cell", "probe_run_id", "standup_s", "reprobed_from", "at")
CERTIFICATE_SCHEMA = RP.CERTIFICATE_SCHEMA
ACCEPTED_KEYS: Tuple[str, ...] = ("max_model_len", "max_num_seqs", "seqs_source", "rung_source")
#: The four files whose sha256 a source record carries (the keys of `Store.shas`).
SHA_KEYS: Tuple[str, ...] = ("manifest_sha256", "rows_sha256", "summary_sha256", "outcomes_sha256")
AGG_ITEM_ID = C.AGG_ITEM_ID
METRIC_ACC = C.METRIC_ACC
MERGED_SUMMARY_FILENAME = "merged_summary.json"
#: `merge --admit-tree <sha> --admit-tree-why <text>` admits EXACTLY that sha as
#: the rung-2 `producer.tree_sha` while rung 1's differs (the verifier-transport fix changed the
#: producing tree between rungs; every other RUNG_EXPERIMENT_PATHS path and projected block must
#: still be equal). Recorded in the merged summary as `tree_admission`; None when the trees are equal.
TREE_PATH: Tuple[str, ...] = (C.MANIFEST_KEY_PRODUCER, "tree_sha")
ADMIT_TREE_FLAG = "--admit-tree"
ADMIT_TREE_WHY_FLAG = "--admit-tree-why"
TREE_ADMISSION_FIELDS: Tuple[str, ...] = ("rung1_tree_sha", "rung2_tree_sha", "why", "flag")
#: ``<task with "/" -> "__">.json`` under summary/ outcomes/ done/ partial/ — the contracts' rule.
marker_filename = C.marker_filename

try:   # the provider module names the literal; never let this script drift from it
    from sisteer.capeval.tiers import inspect_served as _I   # noqa: E402
    assert _I.STOP_REASON_AT_CAP == STOP_REASON_AT_CAP, _I.STOP_REASON_AT_CAP
except ImportError:   # pragma: no cover - a venv without inspect_ai
    pass


class LadderError(RuntimeError):
    """A refusal of this script: a store that lacks a producer fact, a
    reading that disagrees with another, a selection that does not recompute."""


# --------------------------------------------------------------------------- #
# small helpers                                                                #
# --------------------------------------------------------------------------- #


def sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_canonical(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def read_json(path: Union[str, Path]) -> Any:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise LadderError(f"{p}: unreadable: {e}") from e
    except ValueError as e:
        raise LadderError(f"{p}: not JSON: {e}") from e


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path)
    out: List[Dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise LadderError(f"{p}: unreadable: {e}") from e
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as e:
            raise LadderError(f"{p}:{i}: not JSON: {e}") from e
        if not isinstance(row, Mapping):
            raise LadderError(f"{p}:{i}: row is a {type(row).__name__}, not an object")
        out.append(dict(row))
    return out


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _positive_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _dig(d: Mapping[str, Any], path: Sequence[str]) -> Any:
    cur: Any = d
    for key in path:
        cur = cur.get(key) if isinstance(cur, Mapping) else None
    return cur


#: Where a store's producer records the object it SERVED, in the order a reader believes them
#: (shared with `bcb_bfcl_200_analysis.store_object_sha256`): the
#: standup's attested route digest, the per-cell map, then the driver's provenance sha
#: (`cell_controls.artifact_sha256`, the sha over the object bytes). The first lane's REAL
#: manifests carry BOTH route digests null — the standup attested none — and the provenance
#: filled, so a reader bound to the route alone refuses every real fitted store.
OBJECT_SHA_SOURCES: Tuple[str, ...] = (
    f"{C.MANIFEST_KEY_ROUTE}.{C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256}",
    f"{C.MANIFEST_KEY_ROUTE}.{C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL}",
    f"{C.MANIFEST_KEY_PROVENANCE}.artifact_sha256")


def served_object_sha256(manifest: Mapping[str, Any], *, cell: str) -> Tuple[Optional[str], Optional[str]]:
    """``(sha, source)`` of the object a store served — the first non-null of
    :data:`OBJECT_SHA_SOURCES` — or ``(None, None)`` when every source is null (an
    object-free cell, or a fitted store whose producer recorded nothing: the CALLER
    decides which). The sources that ARE recorded must agree: a route digest beside a
    different provenance sha is two objects claimed for one store, refused here."""
    route = manifest.get(C.MANIFEST_KEY_ROUTE) or {}
    by_cell = route.get(C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL) or {}
    candidates = (
        (route.get(C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256), OBJECT_SHA_SOURCES[0]),
        (by_cell.get(cell) if isinstance(by_cell, Mapping) else None, f"{OBJECT_SHA_SOURCES[1]}[{cell}]"),
        (_dig(manifest, (C.MANIFEST_KEY_PROVENANCE, "artifact_sha256")), OBJECT_SHA_SOURCES[2]))
    present = [(str(v), src) for v, src in candidates if isinstance(v, str) and v]
    if not present:
        return None, None
    if len({v for v, _ in present}) != 1:
        raise LadderError(f"cell {cell!r}: the manifest's served-object sources disagree — "
                          + "; ".join(f"{src} = {v[:12]}…" for v, src in present)
                          + " — two objects claimed for one store")
    return present[0]


# --------------------------------------------------------------------------- #
# pins and the id vocabularies                                                 #
# --------------------------------------------------------------------------- #


def to_pinned_id(task: str, log_id: str) -> str:
    """The row / log id (``"17"``) as the canonical PINNED id (``"BigCodeBench/17"``,
    `contracts.bcb_pinned_id`), refusing anything that does not round-trip — a
    ``"017"`` is not an id this lane knows. BFCL ids are the same string in both
    vocabularies."""
    if task != BCB_TASK:
        return str(log_id)
    try:
        pid = C.bcb_pinned_id(log_id)
    except ValueError as e:
        raise LadderError(str(e)) from e
    if C.bcb_log_id(pid) != log_id:
        raise LadderError(f"BigCodeBench id {log_id!r} is not canonical (the log id of {pid} is {C.bcb_log_id(pid)!r})")
    return pid


def to_log_id(task: str, pinned_id: str) -> str:
    """The inverse of :func:`to_pinned_id` (`contracts.bcb_log_id`), with the same round-trip strictness."""
    if task != BCB_TASK:
        return str(pinned_id)
    try:
        lid = C.bcb_log_id(pinned_id)
    except ValueError as e:
        raise LadderError(str(e)) from e
    if C.bcb_pinned_id(lid) != pinned_id:
        raise LadderError(f"BigCodeBench id {pinned_id!r} is not canonical (the pinned id of {lid!r} is {C.bcb_pinned_id(lid)})")
    return lid


def pin_path(repo_root: Union[str, Path], task: str) -> Path:
    if task not in PIN_REL_OF_TASK:
        raise LadderError(f"{task!r} is not a ladder leg; legs: {list(LADDER_TASKS)}")
    return Path(repo_root) / PIN_REL_OF_TASK[task]


def load_pin(repo_root: Union[str, Path], task: str) -> Dict[str, Any]:
    """The pin file of ``task`` through the contracts' VERIFIED loader (schema,
    seed / n / of, 200 distinct ids that regenerate from the recorded rule;
    `SubsetPinError` → refusal), plus the join facts this script reads: BCB ids
    in the canonical `BigCodeBench/<int>` vocabulary with `libs_by_id` keyed by
    exactly those ids; BFCL `population` a {category: [ids]} map in which every
    pinned id appears exactly once."""
    p = pin_path(repo_root, task)
    try:
        pin = PIN_LOADER_OF_TASK[task](repo_root)
    except C.SubsetPinError as e:
        raise LadderError(f"{p}: {e}") from e
    ids = pin["ids"]
    if task == BCB_TASK:
        for i in ids:
            if to_pinned_id(task, to_log_id(task, i)) != i:
                raise LadderError(f"{p}: id {i!r} is not in the pinned `BigCodeBench/<int>` vocabulary")
        libs = pin.get("libs_by_id")
        if not isinstance(libs, Mapping) or set(libs) != set(ids):
            raise LadderError(f"{p}: libs_by_id must be keyed by exactly the {FULL_N} pinned ids")
    else:
        pop = pin["population"]
        if not isinstance(pop, Mapping) or not all(isinstance(v, list) for v in pop.values()):
            raise LadderError(f"{p}: population must be {{category_name: [ids]}}")
        seen: Dict[str, str] = {}
        for cat, members in pop.items():
            for m in members:
                if m in seen:
                    raise LadderError(f"{p}: id {m!r} in categories {seen[m]!r} and {cat!r}")
                seen[m] = cat
        missing = [i for i in ids if i not in seen]
        if missing:
            raise LadderError(f"{p}: {len(missing)} pinned ids in no category, e.g. {missing[:3]}")
    return dict(pin)


def pin_order(pin: Mapping[str, Any]) -> Dict[str, int]:
    return {i: k for k, i in enumerate(pin["ids"])}


# --------------------------------------------------------------------------- #
# stores                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class Store:
    """One (cell, task) store as this script reads it."""

    cell: str
    task: str
    cell_dir: Path
    manifest: Dict[str, Any]
    summary: Dict[str, Any]
    outcomes: Dict[str, Any]
    rows: List[Dict[str, Any]]            # every rows.jsonl row (all tasks)
    marker_kind: str                      # "done" | "partial"
    marker: Dict[str, Any]
    shas: Dict[str, str] = field(default_factory=dict)      # SHA_KEYS -> sha256 of the four files
    root: Optional[Path] = None           # the configuration's out root the store was admitted under (`store_root_of`)

    @property
    def key(self) -> str:
        return self.cell_dir.name

    @property
    def identity(self) -> Dict[str, Any]:
        return self.manifest[C.MANIFEST_KEY_TINYCAP]

    def acc_rows(self) -> Dict[str, Dict[str, Any]]:
        """The per-item ``acc`` rows of this task, keyed by item id (log
        vocabulary), each with its ``extra_json`` PARSED under ``extra``;
        refuses a duplicate item or a row without ``extra_json``."""
        return acc_rows_of(self.rows, self.task, where=str(self.cell_dir / "rows.jsonl"))


def acc_rows_of(rows: Iterable[Mapping[str, Any]], task: str, *, where: str = "rows.jsonl") -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r.get("task") != task or r.get("metric") != METRIC_ACC or r.get("item_id") == AGG_ITEM_ID:
            continue
        item = r.get("item_id")
        if not isinstance(item, str) or not item:
            raise LadderError(f"{where}: an acc row of {task} carries item_id {item!r}")
        if item in out:
            raise LadderError(f"{where}: two acc rows for {task} item {item!r} — a store cannot hold two measurements")
        ej = r.get("extra_json")
        if not isinstance(ej, str):
            raise LadderError(f"{where}: acc row {task} {item!r} carries no extra_json string")
        try:
            extra = json.loads(ej)
        except ValueError as e:
            raise LadderError(f"{where}: acc row {task} {item!r} extra_json is not JSON: {e}") from e
        if not isinstance(extra, Mapping):
            raise LadderError(f"{where}: acc row {task} {item!r} extra_json is not an object")
        out[item] = {**r, "extra": dict(extra)}
    return out


def cells_of(rung_dir: Union[str, Path]) -> List[str]:
    """Every ``<rung dir>/<cell>/`` that holds a ``cells/`` directory, sorted;
    ``_ladder`` and dot-dirs are not cells."""
    root = Path(rung_dir).expanduser()
    if not root.is_dir():
        raise LadderError(f"rung dir {root} is not a directory")
    out = sorted(p.name for p in root.iterdir()
                 if p.is_dir() and not p.name.startswith((".", "_")) and (p / "cells").is_dir())
    if not out:
        raise LadderError(f"{root}: no <cell>/cells/ directory found (is this a pulled rung dir?)")
    return out


def store_dir_for(rung_dir: Union[str, Path], cell: str, task: str) -> Path:
    """The ONE ``<rung dir>/<cell>/cells/<key>/`` that filed a summary for
    ``task``; zero or several refuse by name (a store is never guessed)."""
    root = Path(rung_dir).expanduser() / cell / "cells"
    if not root.is_dir():
        raise LadderError(f"{root}: no cells/ directory for cell {cell!r}")
    hits = sorted(d for d in root.iterdir()
                  if d.is_dir() and (d / C.SUMMARY_DIR / marker_filename(task)).is_file())
    if not hits:
        raise LadderError(f"{root}: no store carries {C.SUMMARY_DIR}/{marker_filename(task)} for cell {cell!r}")
    if len(hits) > 1:
        raise LadderError(f"{root}: {len(hits)} stores carry a summary for {task}: {[h.name for h in hits]} — "
                          "one (cell, task) store per rung dir")
    return hits[0]


def store_root_of(cell_dir: Union[str, Path]) -> Path:
    """The configuration's OUT ROOT a store lives under — ``<root>/cells/<key>/`` →
    ``<root>``, where the launcher files ``lane_roster.json``, ``watch/`` and the
    durable sampling-validation failure; a store dir outside a ``cells/`` folder
    names its parent (a hand-laid fixture), so a root is always checked."""
    d = Path(cell_dir)
    return d.parent.parent if d.parent.name == "cells" else d.parent


def root_admissibility_reason(root: Union[str, Path]) -> Optional[str]:
    """Why a configuration's out root is NOT protocol evidence — None when it is.
    The ONE root-level rule shared by :func:`read_store` (hence every store the
    ladder or the analysis loads, rung 1 and rung 2 alike), :func:`merge` and the
    analysis's own admission: the worker's DURABLE sampling-validation failure
    ``sampling_validation_failed.json`` (`tinycap_served.write_sampling_validation_failure`
    — the clamp override did not hold and the BFCL rows executed at 0.01 behind
    the protocol label), which the launcher's own resume (`untrusted_store_reason`)
    refuses too. An unreadable or non-object marker is a failure, never "absent":
    whatever the store's DONE / PARTIAL markers say, nothing under such a root is
    protocol evidence — the wave is relaunched into a new ``--out-suffix``."""
    p = Path(root) / C.SAMPLING_VALIDATION_FAILED_FILENAME
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        doc = {"status": C.REC_SAMPLING_VALIDATION_FAILED, "error": f"unreadable ({e})"}
    if not isinstance(doc, Mapping):
        doc = {"status": C.REC_SAMPLING_VALIDATION_FAILED, "error": f"not an object ({type(doc).__name__})"}
    return (f"{Path(root)}: carries {C.SAMPLING_VALIDATION_FAILED_FILENAME} ({doc.get('status')}: {str(doc.get('error'))[:200]}; "
            f"attempt {doc.get('function_call_id')!r}) -- its outcomes and Inspect logs are not protocol evidence; the wave must be "
            "relaunched into a new --out-suffix")


def check_root_admissible(root: Union[str, Path]) -> None:
    reason = root_admissibility_reason(root)
    if reason:
        raise LadderError(reason)


def read_store(cell_dir: Union[str, Path], task: str, *, cell: str) -> Store:
    """Read one store for ``task``: its ROOT admitted first (:func:`root_admissibility_reason`
    — a root carrying the worker's sampling-validation failure refuses before a byte
    of the store is read), then manifest (must carry the hashed `tinycap` block),
    summary, outcomes (shape-checked and verified against the summary by the store
    side's own verifier), rows, and exactly one of done/partial marker; the sha256
    of the four files is recorded, the root remembered."""
    d = Path(cell_dir)
    root = store_root_of(d)
    check_root_admissible(root)
    mpath = d / "manifest.json"
    if not mpath.is_file():
        raise LadderError(f"{d}: no manifest.json")
    manifest = read_json(mpath)
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get(C.MANIFEST_KEY_TINYCAP), Mapping):
        raise LadderError(f"{mpath}: no `{C.MANIFEST_KEY_TINYCAP}` identity block — not a tinycap store")
    spath = d / C.SUMMARY_DIR / marker_filename(task)
    if not spath.is_file():
        raise LadderError(f"{d}: no {C.SUMMARY_DIR}/{marker_filename(task)}")
    summary = read_json(spath)
    opath = d / C.TASK_OUTCOMES_DIR / marker_filename(task)
    if not opath.is_file():
        raise LadderError(f"{d}: no {C.TASK_OUTCOMES_DIR}/{marker_filename(task)} — the all-attempted record is "
                          "what the ladder reads; a store without one is torn")
    outcomes = read_json(opath)
    try:
        completion.verify_outcomes_against_summary(outcomes, summary)
    except completion.CompletionError as e:
        raise LadderError(f"{d}: {e}") from e
    rpath = d / "rows.jsonl"
    if not rpath.is_file():
        raise LadderError(f"{d}: no rows.jsonl")
    rows = read_jsonl(rpath)
    done = d / C.DONE_DIR / marker_filename(task)
    partial = d / C.PARTIAL_DIR / marker_filename(task)
    if done.is_file() and partial.is_file():
        raise LadderError(f"{d}: both a done and a partial marker for {task}")
    if done.is_file():
        kind, marker = "done", read_json(done)
    elif partial.is_file():
        kind, marker = "partial", read_json(partial)
    else:
        raise LadderError(f"{d}: no done/ or partial/ marker for {task} — the task did not finish")
    shas = dict(zip(SHA_KEYS, (sha256_file(p) for p in (mpath, rpath, spath, opath))))
    return Store(cell=cell, task=task, cell_dir=d, manifest=dict(manifest), summary=dict(summary),
                 outcomes=dict(outcomes), rows=rows, marker_kind=kind, marker=dict(marker), shas=shas, root=root)


def load_store(rung_dir: Union[str, Path], cell: str, task: str) -> Store:
    return read_store(store_dir_for(rung_dir, cell, task), task, cell=cell)


# ---------------------------------------------------------------------------- #
# the detector                                                                 #
# ---------------------------------------------------------------------------- #


def at_cap_reading(store: Store) -> Dict[str, Any]:
    """The producer's at-cap facts of one (cell, task), cross-checked:
    summary ``n_at_cap`` / ``n_stop_reason`` (REFUSE None; REFUSE
    ``n_stop_reason != 200`` — every response must carry a stop reason for
    the rule to be decidable) must equal the outcome record's ``at_cap`` count
    and the rows' ``stop_reason == "max_tokens"`` count (REFUSE a disagreement).
    ``n_model_length`` (context overflow) is reported beside, never counted."""
    s, task, where = store.summary, store.task, str(store.cell_dir)
    n_at_cap, n_stop = s.get("n_at_cap"), s.get("n_stop_reason")
    if n_at_cap is None or n_stop is None:
        raise LadderError(f"{where}: summary of {task} records n_at_cap={n_at_cap!r} n_stop_reason={n_stop!r} — "
                          "the producer did not measure at-cap; the rule cannot be decided")
    if not isinstance(n_at_cap, int) or not isinstance(n_stop, int) or isinstance(n_at_cap, bool):
        raise LadderError(f"{where}: summary of {task} n_at_cap / n_stop_reason are not ints")
    if n_stop != FULL_N:
        raise LadderError(f"{where}: summary of {task} n_stop_reason {n_stop} != {FULL_N} — not every response "
                          "carries a stop reason (or the item set is short); refusing to decide the rule")
    if s.get("n") != FULL_N:
        raise LadderError(f"{where}: summary of {task} n {s.get('n')!r} != {FULL_N}")
    rows, by_item = validated_items(store)
    if len(rows) != FULL_N:
        raise LadderError(f"{where}: {task} has {len(rows)} acc rows, not {FULL_N}")
    r_at_cap = sum(1 for r in rows.values() if r["extra"].get("stop_reason") == STOP_REASON_AT_CAP)
    r_stop = sum(1 for r in rows.values() if r["extra"].get("stop_reason") is not None)
    if n_at_cap != r_at_cap:
        raise LadderError(f"{where}: {task} at-cap disagreement — summary n_at_cap {n_at_cap}, rows {r_at_cap}")
    if n_stop != r_stop:
        raise LadderError(f"{where}: {task} stop-reason disagreement — summary n_stop_reason {n_stop}, rows {r_stop}")
    return {"n": FULL_N, "n_at_cap": n_at_cap, "n_stop_reason": n_stop, "at_cap_rate": n_at_cap / n_stop,
            "n_model_length": sum(1 for it in by_item.values() if it.get("stop_reason") == STOP_REASON_MODEL_LENGTH),
            "cap": s.get("max_tokens")}


def validated_items(store: Store) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """The store's per-item ``acc`` rows and its outcome records JOINED by item
    identity (k = 1: one record per item at ``sample_idx`` 0), refusing any
    disagreement PER ITEM: the two must name the same items, and
    for every item the row's ``stop_reason`` must equal the record's, the
    record's ``at_cap`` must equal ``stop_reason == "max_tokens"``, and the two
    ``value``s must agree. Returns ``(rows_by_item, outcomes_by_item)``; the
    selection is then recomputed from the ROWS (the contract's source), which
    this join has shown to agree with the record."""
    where, task = str(store.cell_dir), store.task
    if store.outcomes.get("samples_per_item") != 1:
        raise LadderError(f"{where}: {task} samples_per_item {store.outcomes.get('samples_per_item')!r} != 1 — "
                          "this lane is k = 1")
    by_item: Dict[str, Dict[str, Any]] = {}
    for it in store.outcomes["items"]:
        if it.get("sample_idx") != 0:
            raise LadderError(f"{where}: outcomes of {task}: item {it.get('item_id')!r} at sample_idx {it.get('sample_idx')!r} "
                              "— this lane is k = 1")
        if it["item_id"] in by_item:
            raise LadderError(f"{where}: outcomes of {task}: two records for item {it['item_id']!r}")
        by_item[it["item_id"]] = it
    rows = store.acc_rows()
    if set(rows) != set(by_item):
        only_rows, only_out = sorted(set(rows) - set(by_item)), sorted(set(by_item) - set(rows))
        raise LadderError(f"{where}: {task} rows and outcomes name different items (rows only {only_rows[:5]}, outcomes only "
                          f"{only_out[:5]})")
    bad: List[str] = []
    for item, r in rows.items():
        it, e = by_item[item], r["extra"]
        row_reason, out_reason = e.get("stop_reason"), it.get("stop_reason")
        # a fully unscored item has `value: null` and files NO acc row (PW-nanfix) — so a null value never
        # validates a row that exists: the joined record must carry a numeric value equal to the row's
        value_ok = (isinstance(it.get("value"), (int, float)) and not isinstance(it.get("value"), bool)
                    and float(it["value"]) == float(r["value"]))
        agree = row_reason == out_reason and bool(it.get("at_cap")) == (out_reason == STOP_REASON_AT_CAP) and value_ok
        if not agree:
            bad.append(f"{item} (row value {r['value']!r} stop_reason {row_reason!r}; record value {it.get('value')!r} "
                       f"stop_reason {out_reason!r} at_cap {it.get('at_cap')!r})")
    if bad:
        raise LadderError(f"{where}: {task} at-cap disagreement between rows and outcomes on {len(bad)} item(s), e.g. "
                          f"{bad[:3]} — the two producer records must agree per item before any selection is drawn")
    return rows, by_item


def leg_fires(readings: Mapping[str, Mapping[str, Any]]) -> bool:
    """A leg FIRES iff ANY cell's ``n_at_cap / n_stop_reason > FIRE_THRESHOLD``
    (strictly greater: 2/200 = 1.0 % does not fire, 3/200 = 1.5 % does)."""
    return any(r["n_at_cap"] / r["n_stop_reason"] > FIRE_THRESHOLD for r in readings.values())


DEFAULT_LEGS_SOURCE = "default: both legs (no --roster/--wave given and no analysis.json beside the stores)"


def ladder_tasks(rung_dir: Union[str, Path], *, roster: Optional[Union[str, Path]] = None,
                 wave: Optional[str] = None) -> Tuple[List[str], str]:
    """The wave's leg set for the detector and the selector — ``(tasks in LADDER_TASKS order, source)``:
    the lane roster's RUNNABLE legs for ``wave`` when ``--roster/--wave`` are given (`widen_roster.runnable_legs`;
    a `blocked` leg — the llama waves' BFCL, skipped; nemotron's until its live parser PASS of
    2026-09-23 — is not a ladder leg), else the runnable legs the
    rung-1 ``analysis.json`` beside the stores records (`--analyze` wrote them from the roster of its day), else
    BOTH legs (the two-leg waves' default, unchanged). Previously the detector walked `LADDER_TASKS` unconditionally
    and refused a BCB-only wave for its missing BFCL store."""
    if (roster is None) != (wave is None):
        raise LadderError("--roster and --wave come together (the wave whose runnable legs are the ladder's leg set)")
    if roster is not None:
        try:
            doc = W.load_roster(Path(roster).expanduser())
            legs = W.runnable_legs(doc, str(wave))
        except (W.RosterError, OSError, ValueError) as e:
            raise LadderError(f"--roster {roster} / --wave {wave!r}: {e}") from e
        tasks = [TASK_OF_LEG[leg] for leg in legs if leg in TASK_OF_LEG]
        if not tasks:
            raise LadderError(f"--roster {roster} / --wave {wave!r}: no runnable leg among {list(TASK_OF_LEG)} (legs {legs}) — nothing to detect")
        return [t for t in LADDER_TASKS if t in tasks], f"roster {Path(roster).expanduser()} wave {wave} (runnable legs {legs})"
    rec = Path(rung_dir).expanduser() / ANALYSIS_FILENAME
    if rec.is_file():
        doc = read_json(rec)
        tasks = doc.get("tasks") if isinstance(doc, Mapping) else None
        if not isinstance(tasks, list) or not tasks or any(t not in LADDER_TASKS for t in tasks):
            raise LadderError(f"{rec}: `tasks` {tasks!r} is not a non-empty list of ladder legs {list(LADDER_TASKS)} — not a rung-1 analysis record")
        return [t for t in LADDER_TASKS if t in tasks], f"{ANALYSIS_FILENAME} beside the stores (the roster's runnable legs at analysis time: {tasks})"
    return list(LADDER_TASKS), DEFAULT_LEGS_SOURCE


def stores_carrying(rung_dir: Union[str, Path], cell: str, task: str) -> List[Path]:
    """Every ``<rung dir>/<cell>/cells/<key>/`` that filed a summary for ``task`` (zero, one or several)."""
    root = Path(rung_dir).expanduser() / cell / "cells"
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / C.SUMMARY_DIR / marker_filename(task)).is_file())


def detect(rung_dir: Union[str, Path], *, cells: Optional[Sequence[str]] = None, tasks: Optional[Sequence[str]] = None,
           tasks_source: str = DEFAULT_LEGS_SOURCE) -> Dict[str, Any]:
    """The detector over every (cell, leg) store under ``rung_dir`` for the wave's leg set ``tasks``
    (:func:`ladder_tasks`; default both legs). A leg in the set without a store refuses (naming the leg set and
    its source); a store carrying a leg OUTSIDE the set refuses as a mismatch between the stores and the wave's legs."""
    cells = list(cells) if cells else cells_of(rung_dir)
    if len(cells) < 2:
        raise LadderError(f"{rung_dir}: the paired design needs two cells, found {cells}")
    legs = [t for t in LADDER_TASKS if t in (list(tasks) if tasks else LADDER_TASKS)]
    if not legs or (tasks and any(t not in LADDER_TASKS for t in tasks)):
        raise LadderError(f"{rung_dir}: leg set {list(tasks or [])!r} is not a non-empty subset of the ladder legs {list(LADDER_TASKS)}")
    not_run = [t for t in LADDER_TASKS if t not in legs]
    for task in not_run:
        for cell in cells:
            hits = stores_carrying(rung_dir, cell, task)
            if hits:
                raise LadderError(f"{Path(rung_dir).expanduser() / cell}: carries a store for {task} ({[h.name for h in hits]}) — the roster "
                                  f"does not run that leg (leg set {legs} from {tasks_source}); the stores and the wave's legs disagree")
    per_task: Dict[str, Any] = {}
    for task in legs:
        try:
            readings = {cell: at_cap_reading(load_store(rung_dir, cell, task)) for cell in cells}
        except LadderError as e:
            hint = ("; pass --roster/--wave for the wave's runnable legs, or run --analyze first so analysis.json beside the stores names them"
                    if tasks_source == DEFAULT_LEGS_SOURCE else "")
            raise LadderError(f"{e} (leg set {legs} from {tasks_source}{hint})") from e
        per_task[task] = {"fired": leg_fires(readings), "cells": readings}
    return {"rung_dir": str(Path(rung_dir).expanduser()), "cells": cells, "threshold": FIRE_THRESHOLD,
            "full_n": FULL_N, "fired_any": any(v["fired"] for v in per_task.values()), "tasks": per_task,
            "legs": legs, "legs_not_run": not_run, "legs_source": tasks_source}


# --------------------------------------------------------------------------- #
# select — selection with provenance                                          #
# --------------------------------------------------------------------------- #


def capped_pinned_ids(store: Store, pin: Mapping[str, Any]) -> List[str]:
    """The items of ``store`` that stopped at the cap — read from the ROWS after
    :func:`validated_items` has shown them to agree with the outcome record item
    by item — as canonical PINNED ids in pin order; an id outside the pin
    refuses (a foreign store)."""
    order = pin_order(pin)
    rows, _ = validated_items(store)
    out: List[str] = []
    for item, r in rows.items():
        if r["extra"].get("stop_reason") == STOP_REASON_AT_CAP:
            pid = to_pinned_id(store.task, item)
            if pid not in order:
                raise LadderError(f"{store.cell_dir}: capped item {item!r} ({pid}) is not in the {store.task} pin — a foreign store")
            out.append(pid)
    return sorted(out, key=order.__getitem__)


def source_record(store: Store, pin: Mapping[str, Any], reading: Mapping[str, Any]) -> Dict[str, Any]:
    cap = store.summary.get("max_tokens")
    if not _positive_int(cap):
        raise LadderError(f"{store.cell_dir}: summary of {store.task} records max_tokens {cap!r} — the rung-1 cap "
                          "must be a positive int")
    ids = capped_pinned_ids(store, pin)
    return {"cell": store.cell, "task": store.task, "store_key": store.key, **store.shas,
            "identity": dict(store.identity), "cap": int(cap), "n_at_cap": int(reading["n_at_cap"]),
            "capped_ids": ids, "capped_ids_sha256": C.ids_sha256(ids)}


def union_selection(task: str, sources: Sequence[Mapping[str, Any]], pin: Mapping[str, Any],
                    stores: Mapping[str, Store]) -> Dict[str, Any]:
    """The per-task selection block: the UNION of the sources' capped ids in pin
    order, ``n`` (= ``n_selected``), ``ids_sha256``, ``cap_from`` (every source
    agrees), ``cap_to`` and ``max_prompt_tokens_rung1`` — the largest rung-1
    prompt among the selected items across both cells (the fit guard's input;
    a selected item without a recorded prompt_tokens refuses)."""
    srcs = [s for s in sources if s["task"] == task]
    if not srcs:
        raise LadderError(f"no sources for {task}")
    caps = {s["cap"] for s in srcs}
    if len(caps) != 1:
        raise LadderError(f"{task}: the rung-1 sources disagree on the cap: {sorted(caps)}")
    order = pin_order(pin)
    ids = sorted({i for s in srcs for i in s["capped_ids"]}, key=order.__getitem__)
    if not ids:
        raise LadderError(f"{task}: fired but no capped ids")
    log_ids = {to_log_id(task, i) for i in ids}
    prompts: List[int] = []
    for s in srcs:
        st = stores[s["cell"]]
        by_item = {it["item_id"]: it for it in st.outcomes["items"]}
        for lid in log_ids:
            it = by_item.get(lid)
            if it is None:
                raise LadderError(f"{st.cell_dir}: selected item {lid!r} has no outcome record in cell {s['cell']}")
            pt = it.get("prompt_tokens")
            if not _positive_int(pt):
                raise LadderError(f"{st.cell_dir}: selected item {lid!r} records prompt_tokens {pt!r} — the fit guard "
                                  "needs the rung-1 prompt length of every selected item")
            prompts.append(int(pt))
    return {"ids": ids, "n": len(ids), "n_selected": len(ids), "ids_sha256": C.ids_sha256(ids),
            "cap_from": int(caps.pop()), "cap_to": CAP_TO, "max_prompt_tokens_rung1": max(prompts)}


def selection_digest(selection: Mapping[str, Mapping[str, Any]]) -> str:
    """``sha256(canonical_json(selection))`` over the FULL per-task blocks —
    ids, n, n_selected, ids_sha256, cap_from, cap_to, max_prompt_tokens_rung1 —
    so a tampered prompt bound changes the digest. The certificate
    (`rung_probe.json.selection_sha256`) binds exactly this."""
    return sha256_canonical({k: dict(v) for k, v in selection.items()})


def build_selection(rung_dir: Union[str, Path], *, repo_root: Union[str, Path] = REPO,
                    cells: Optional[Sequence[str]] = None, tasks: Optional[Sequence[str]] = None,
                    tasks_source: str = DEFAULT_LEGS_SOURCE) -> Dict[str, Any]:
    """The `ladder_selection.json` document: the detector's
    readings for every leg OF THE WAVE'S LEG SET (:func:`ladder_tasks`),
    per-source provenance and the union for every FIRED leg, and the
    selection digest; `legs` / `legs_source` record the set and where it came from."""
    det = detect(rung_dir, cells=cells, tasks=tasks, tasks_source=tasks_source)
    sources: List[Dict[str, Any]] = []
    selection: Dict[str, Any] = {}
    for task in det["legs"]:
        if not det["tasks"][task]["fired"]:
            continue
        pin = load_pin(repo_root, task)
        stores = {cell: load_store(rung_dir, cell, task) for cell in det["cells"]}
        task_sources = [source_record(stores[cell], pin, det["tasks"][task]["cells"][cell]) for cell in det["cells"]]
        sources.extend(task_sources)
        selection[task] = union_selection(task, task_sources, pin, stores)
    return {"schema": LADDER_SCHEMA, "generated_by": "scripts/tinycap_ladder.py select", "generated_at": _now_iso(),
            "from_suffix": Path(rung_dir).expanduser().resolve().name, "threshold": FIRE_THRESHOLD, "full_n": FULL_N,
            "cells": det["cells"], "detector": det["tasks"], "fired_any": det["fired_any"],
            "legs": det["legs"], "legs_not_run": det["legs_not_run"], "legs_source": det["legs_source"],
            "sources": sources, "selection": selection, "selection_sha256": selection_digest(selection)}


def load_selection(path: Union[str, Path]) -> Dict[str, Any]:
    """A selection file through the contracts' ONE reader
    (`rung_probe.read_ladder_selection`: exact block keys, ladder row tasks,
    ``n == n_selected == len(ids)`` distinct ids in the task's pinned vocabulary,
    ``ids_sha256`` and the FULL digest recomputed, the fixed ``cap_to``,
    ``cap_from < cap_to``, a positive prompt bound; an EMPTY selection refuses —
    nothing fired, no ladder) plus this file's own provenance keys."""
    try:
        sel = RP.read_ladder_selection(path)
    except RP.LadderError as e:
        raise LadderError(str(e)) from e
    for key in ("cells", "detector", "sources"):
        if key not in sel:
            raise LadderError(f"{path}: missing {key!r}")
    if sel["schema"] != LADDER_SCHEMA:
        raise LadderError(f"{path}: schema {sel['schema']!r} != {LADDER_SCHEMA}")
    if not isinstance(sel["cells"], list) or len(sel["cells"]) < 2:
        raise LadderError(f"{path}: cells must name the paired design's cells, got {sel['cells']!r}")
    return dict(sel)


def write_selection(doc: Mapping[str, Any], out: Union[str, Path]) -> Path:
    p = Path(out).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# the certificate (read here for its bindings; validated by rung_probe)        #
# --------------------------------------------------------------------------- #


def certificate_serving_engine(cert: Mapping[str, Any]) -> str:
    """The engine a certificate's recorded ``ladder`` block names (native-serving-default):
    ``lora_native`` when the block carries the native keys' ``serving_engine``, else ``lens`` (the four-key lens
    block names none — by era, every lens certificate). Any other spelling refuses by name."""
    lad = cert.get("ladder")
    engine = lad.get("serving_engine", RP.SERVING_ENGINE_LENS) if isinstance(lad, Mapping) else RP.SERVING_ENGINE_LENS
    if engine not in (RP.SERVING_ENGINE_LENS, RP.SERVING_ENGINE_NATIVE):
        raise LadderError(f"the certificate's ladder block names serving_engine {engine!r}, neither "
                          f"{RP.SERVING_ENGINE_LENS!r} nor {RP.SERVING_ENGINE_NATIVE!r}")
    return str(engine)


def store_serving_engine(store: Store) -> str:
    """The ENGINE a rung store's manifest states — ``lens`` or ``lora_native`` — through THE ONE reader,
    :func:`contracts.read_store_serving_engine` (every spelling the producer stamps, held to the serving identity
    the record carries; a lens store by era on positive evidence). A rung store is a SERVED store, so a manifest that
    states nothing (an HF reference's shape), a torn or relabelled one, or an unknown engine REFUSES by name: a
    certificate is evidence about ONE engine's footprint and binds to no store whose engine cannot be read."""
    reading = C.read_store_serving_engine(store.manifest, where=str(store.cell_dir / "manifest.json"))
    if reading.refusal is not None:
        raise LadderError(f"{store.cell_dir}: the store's serving engine cannot be read — {reading.refusal}")
    if reading.engine is None:
        raise LadderError(f"{store.cell_dir}: the store's manifest states no serving engine ({reading.why}) — a rung store "
                          "is a served store and must read as one engine; a certificate binds to no store of unknown engine")
    return str(reading.engine)


def spec_for_certificate(model_key: str, cert: Mapping[str, Any]) -> Tuple["RP.LadderSpec", int, str]:
    """``(spec, batched_token_budget, budget_source)`` a certificate is judged against: the SHARED table's ladder
    (`rung_probe.ladder_spec`, its measured row) when the table sizes the model — ON THE ENGINE the certificate's own
    ``ladder`` block names (a native certificate is held to the ENGINE-SERVED ladder `rung_probe.native_ladder_spec`
    derives from the table's, never to the lens rungs it could not have been walked on; a lens certificate to the lens
    ladder, byte-identical) — else the ladder the certificate itself RECORDED (its `ladder` block, read through
    `rung_probe.spec_from_block`: the lens block's four keys, or the native block RE-DERIVED from its requested
    rungs) with the budget it recorded. The landed tree keeps main's geometry rows, under which llama8b's row is
    provisional and `ladder_spec` refuses — yet the llama ladders ran under a measured row and are finished;
    refusing to READ their certificates would make the merged waves unreproducible from the landed code. The
    fallback is explicit (a table refusal, never a silent default) and holds the certificate to its own recorded
    ladder; a native ladder the engine cannot size for a TABLE-sized model (no pinned width) is refused by name,
    never judged by the lens ladder; every other rule of `certificate_is_valid` (walk legality, cap map, selection
    digest, fit guard) runs unchanged."""
    engine = certificate_serving_engine(cert)
    try:
        spec = RP.ladder_spec(model_key)
    except RP.LadderError as table_refusal:
        lad, bud = cert.get("ladder"), cert.get("budget")
        if not isinstance(lad, Mapping) or not isinstance(bud, Mapping):
            raise LadderError(f"the shared geometry table refuses the ladder of {model_key!r} ({table_refusal}) and the certificate "
                              "records no `ladder` / `budget` block to judge it by") from table_refusal
        if lad.get("model_key") != model_key:
            raise LadderError(f"the certificate's recorded ladder names model {lad.get('model_key')!r}, not {model_key!r}") from table_refusal
        from sisteer.capeval.serving import geometry as _sg
        try:
            spec = RP.spec_from_block(lad, _sg.BUDGET_BY_MODEL.get(model_key))
        except RP.LadderError as e:
            raise LadderError(f"the shared geometry table refuses the ladder of {model_key!r} ({table_refusal}) and the certificate's "
                              f"recorded ladder block cannot be read: {e}") from table_refusal
        return spec, int(bud["batched_token_budget"]), str(bud["budget_source"])
    if engine == RP.SERVING_ENGINE_NATIVE:
        try:
            spec = RP.native_ladder_spec(spec)
        except RP.LadderError as e:
            raise LadderError(f"a {RP.SERVING_ENGINE_NATIVE} certificate of {model_key!r} cannot be judged: {e}") from None
    budget = RP.ladder_budget(spec)
    return spec, int(budget.batched_token_budget), str(budget.source)


def validate_certificate(cert: Mapping[str, Any], selection: Mapping[str, Mapping[str, Any]]) -> None:
    """The launcher's gate, re-run here (`rung_probe.certificate_is_valid`):
    window 32,768, S on the ladder, source consistency, the measured budget
    and ``S x L`` within it, a LEGAL walk whose skips agree with the recorded
    refusals, the cap map == the selection's ``cap_to`` per task, the selection
    digest over the FULL block, the fit guard. The launch-side provenance
    (tree, model, config, object digests) is taken from the certificate ITSELF
    — a reader of stores cannot know the launch — and is bound to the rung-2
    manifests by :func:`bind_certificate_to_store` instead; an operator rung is
    admitted at its own accepted pair (the analysis names it)."""
    caps = {task: int(block["cap_to"]) for task, block in selection.items()}
    # bcb-bfcl-200-widen S6: the MODEL's ladder, read off the certificate's own row (never Gemma's
    # numbers for a Nemotron or qwen certificate); a slug without a measured budget refuses there
    try:
        model_key = C.model_row(str(cert["model_slug"])).model_key
    except C.UnknownModel as e:
        raise LadderError(f"certificate model_slug {cert.get('model_slug')!r}: {e}") from None
    spec, budget_tokens, budget_source = spec_for_certificate(model_key, cert)
    prov = RP.Provenance(tree_sha=cert["tree_sha"], model_slug=cert["model_slug"], hf=cert["hf"], rev=cert["rev"],
                         cfg_path=cert["cfg_path"], cfg_sha256=cert["cfg_sha256"], object_digests=dict(cert["object_digests"]),
                         selection={t: dict(b) for t, b in selection.items()}, cap_overrides_by_task=caps,
                         batched_token_budget=budget_tokens, budget_source=budget_source,
                         suffix=str(cert["suffix"]), probed_cell=str(cert["probed_cell"]), ladder=spec)
    acc = cert.get("accepted") or {}
    operator = ((int(acc["max_model_len"]), int(acc["max_num_seqs"]))
                if isinstance(acc, Mapping) and acc.get("rung_source") == RP.RUNG_SOURCE_OPERATOR else None)
    reason = RP.certificate_is_valid(cert, prov, operator_rung=operator)
    if reason:
        raise LadderError(f"certificate refused: {reason}")


def load_certificate(path: Union[str, Path], *, selection: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """`rung_probe.json` (the certificate): exact key set, the
    accepted rung's four keys with positive int dims and admitted sources; with
    ``selection`` given, the launcher's full gate (:func:`validate_certificate`)
    — the merge and the meter always pass one."""
    cert = read_json(path)
    if not isinstance(cert, Mapping):
        raise LadderError(f"{path}: not an object")
    have, want = set(cert), set(CERTIFICATE_KEYS)
    if have != want:
        raise LadderError(f"{path}: certificate keys {sorted(have)} != {sorted(want)} (missing {sorted(want - have)}, "
                          f"extra {sorted(have - want)})")
    if cert["schema"] != CERTIFICATE_SCHEMA:
        raise LadderError(f"{path}: schema {cert['schema']!r} != {CERTIFICATE_SCHEMA}")
    acc = cert["accepted"]
    if not isinstance(acc, Mapping) or set(acc) != set(ACCEPTED_KEYS):
        raise LadderError(f"{path}: accepted must carry exactly {list(ACCEPTED_KEYS)}")
    if not _positive_int(acc["max_model_len"]) or not _positive_int(acc["max_num_seqs"]):
        raise LadderError(f"{path}: accepted dims must be positive ints, got {acc!r}")
    if acc["rung_source"] not in RUNG_SOURCES:
        raise LadderError(f"{path}: rung_source {acc['rung_source']!r} not in {list(RUNG_SOURCES)}")
    if acc["seqs_source"] not in SEQS_SOURCES:
        raise LadderError(f"{path}: seqs_source {acc['seqs_source']!r} not in {list(SEQS_SOURCES)}")
    if not isinstance(cert["walked"], list) or not isinstance(cert["cap_overrides_by_task"], Mapping):
        raise LadderError(f"{path}: walked must be a list and cap_overrides_by_task an object")
    if not isinstance(cert["selection_sha256"], str) or len(cert["selection_sha256"]) != 64:
        raise LadderError(f"{path}: selection_sha256 is not a sha256 hex")
    if selection is not None:
        validate_certificate(cert, selection)
    return dict(cert)


# --------------------------------------------------------------------------- #
# merge — recompute, never trust                                              #
# --------------------------------------------------------------------------- #


def _identity_diff_keys(a: Mapping[str, Any], b: Mapping[str, Any]) -> List[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


# ---------------------------------------------------------------------------- #
# bcb-bfcl-200-widen S6: the ONE grouped projection                             #
# rule set, shared by the ladder and the analysis                               #
# ---------------------------------------------------------------------------- #

#: What :func:`project_identity` projects: the hashed `tinycap` identity block, or the manifest's
#: `sampling` / `thinking` block. Per kind: the TASK-MAP fields, the per-task PROVENANCE fields
#: inside a map's entries, the run-level SCALARS excluded (derived joins), the CONDITIONAL maps.
PROJECTION_KINDS: Mapping[str, Dict[str, Any]] = {
    "identity": {"maps": tuple(k for k in PROJECTED_IDENTITY_KEYS if k != "tasks"), "provenance": {},
                 "excluded_scalars": (), "conditional": dict(CONDITIONAL_IDENTITY_KEYS)},
    C.MANIFEST_KEY_SAMPLING: {"maps": ("samples_by_task", "by_task"), "provenance": {"by_task": ("source",)},
                              "excluded_scalars": EXCLUDED_MANIFEST_SCALARS[C.MANIFEST_KEY_SAMPLING],
                              "conditional": {"by_task": SAMPLING_MODE_PROTOCOL}},
    C.MANIFEST_KEY_THINKING: {"maps": ("by_task",), "provenance": {}, "excluded_scalars": (), "conditional": {}},
}


def project_identity(block: Mapping[str, Any], retained_tasks: Sequence[str], *, kind: str = "identity",
                     mode: Optional[str] = None) -> Dict[str, Any]:
    """ONE projection of an identity / manifest block onto ``retained_tasks`` (the rerun's own
    row tasks ∩ the merged tasks), the same rules for the ladder and the analysis:

    1. every field is a SCALAR, a TASK MAP (``PROJECTION_KINDS[kind]["maps"]``) or PER-TASK
       PROVENANCE inside a map entry (``sampling.by_task[task].source``);
    2. ``scalars`` = the block with EVERY task map and every excluded run-level join stripped at
       once (``sampling.source`` — a derived join over the filed tasks — is excluded);
    3. ``maps[key]`` = the map RESTRICTED to the retained tasks, refusing a retained task the map
       lacks (a rerun that dropped a task it files is another instrument); the map's own cover
       (``maps_cover[key]``) is reported for the "covers exactly its tasks" rule;
    4. ``provenance[key][task]`` = the per-task provenance fields, compared per retained task;
       the entries in ``maps`` carry the remaining (non-provenance) fields;
    5. ``optional`` = which CONDITIONAL maps the block carries; with ``mode`` given, a conditional
       map must be present iff its mode is the run's (a protocol store without ``by_task`` / a
       card store carrying it refuses here).
    """
    rules = PROJECTION_KINDS.get(kind)
    if rules is None:
        raise LadderError(f"project_identity: unknown block kind {kind!r} (one of {list(PROJECTION_KINDS)})")
    if not isinstance(block, Mapping):
        raise LadderError(f"project_identity({kind}): the block is {type(block).__name__}, not an object")
    retained = [str(t) for t in retained_tasks]
    maps: Dict[str, Dict[str, Any]] = {}
    cover: Dict[str, List[str]] = {}
    prov: Dict[str, Dict[str, Any]] = {}
    optional: Dict[str, bool] = {}
    for key in rules["maps"]:
        present = key in block
        if key in rules["conditional"]:
            optional[key] = present
            if mode is not None and present != (mode == rules["conditional"][key]):
                raise LadderError(f"project_identity({kind}): {key!r} is {'present' if present else 'absent'} but the "
                                  f"sampling mode is {mode!r} — {key!r} is present iff the mode is "
                                  f"{rules['conditional'][key]!r}")
        if not present:
            if key in rules["conditional"]:
                continue
            # code review: an ABSENT non-conditional per-task map used to be skipped (the compare
            # loop then `continue`d on None), so a rung-2 identity wholly missing thinking_by_task /
            # samples_by_task was no longer refused — it is another instrument
            raise LadderError(f"project_identity({kind}): {key!r} is absent — every {kind} block carries this "
                              f"per-task map (present: {sorted(block)})")
        m = block.get(key)
        if not isinstance(m, Mapping):
            raise LadderError(f"project_identity({kind}): {key!r} is {type(m).__name__}, not a per-task map")
        cover[key] = sorted(str(t) for t in m)
        missing = [t for t in retained if t not in m]
        if missing:
            raise LadderError(f"project_identity({kind}): {key!r} lacks the retained task(s) {missing} (covers {cover[key]})")
        pfields = tuple(rules["provenance"].get(key, ()))
        entries: Dict[str, Any] = {}
        pentries: Dict[str, Any] = {}
        for t in retained:
            v = m[t]
            if pfields and isinstance(v, Mapping):
                entries[t] = {k: v[k] for k in v if k not in pfields}
                pentries[t] = {k: v.get(k) for k in pfields}
            else:
                entries[t] = v
        maps[key] = entries
        if pfields:
            prov[key] = pentries
    strip = set(rules["maps"]) | set(rules["excluded_scalars"])
    scalars = {k: v for k, v in block.items() if k not in strip}
    return {"kind": kind, "retained": retained, "scalars": scalars, "maps": maps, "maps_cover": cover,
            "provenance": prov, "optional": optional}


def projection_differences(p1: Mapping[str, Any], p2: Mapping[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    """Every difference between two projections of the SAME kind and retained tasks: the scalars
    (key by key), the optional-key set, each map per retained task, each provenance per task."""
    if p1["kind"] != p2["kind"] or p1["retained"] != p2["retained"]:
        raise LadderError(f"projection_differences: incomparable projections ({p1['kind']}/{p1['retained']} vs "
                          f"{p2['kind']}/{p2['retained']})")
    bad: Dict[str, Tuple[Any, Any]] = {}
    for k in sorted(set(p1["scalars"]) | set(p2["scalars"])):
        if p1["scalars"].get(k) != p2["scalars"].get(k):
            bad[k] = (p1["scalars"].get(k), p2["scalars"].get(k))
    if p1["optional"] != p2["optional"]:
        bad["<optional keys>"] = (p1["optional"], p2["optional"])
    for key in sorted(set(p1["maps"]) | set(p2["maps"])):
        m1, m2 = p1["maps"].get(key), p2["maps"].get(key)
        if m1 is None or m2 is None:
            if key not in p1["optional"] and key not in p2["optional"]:
                bad[key] = (m1, m2)
            continue
        for t in p1["retained"]:
            if m1.get(t) != m2.get(t):
                bad[f"{key}[{t}]"] = (m1.get(t), m2.get(t))
    for key in sorted(set(p1["provenance"]) | set(p2["provenance"])):
        v1, v2 = p1["provenance"].get(key) or {}, p2["provenance"].get(key) or {}
        for t in p1["retained"]:
            if v1.get(t) != v2.get(t):
                bad[f"{key}[{t}].provenance"] = (v1.get(t), v2.get(t))
    return bad


def check_rung_identity_contract(rung1: Mapping[str, Any], rung2: Mapping[str, Any], *, where: str,
                                 accepted: Mapping[str, Any], merged_tasks: Sequence[str], task: Optional[str] = None) -> None:
    """The rung-1 ↔ rung-2 identity contract: the two hashed
    `tinycap` blocks differ in exactly (a subset of) :data:`ADMITTED_IDENTITY_DIFFS`,
    with `served` differing only in its window, its S and — because the launcher
    sets the driver's ``concurrency`` to the resolved S by construction — a
    concurrency that equals ``max_num_seqs`` on EACH rung and the certificate's
    accepted S on rung 2; any other concurrency move refuses.

    The per-task maps (:data:`PROJECTED_IDENTITY_KEYS`) and the legs' own hashed
    fields (:data:`LEG_IDENTITY_FIELDS`) are compared on the PROJECTION onto
    ``merged_tasks``: the normal launch files both legs in one configuration
    while a rerun of the fired leg files one, so rung 2's ``tasks`` must be a
    subset of rung 1's that contains every merged task, each map must equal
    rung 1's on the merged tasks and cover exactly rung 2's tasks, and a leg's
    fields are present and equal iff rung 2 files that leg.
    ``task`` (the store's task being merged) must be among rung 2's tasks; when
    both legs fire and the rerun is launched per leg, each rung-2 store files
    one of the merged tasks and is projected onto the merged tasks IT files."""
    leg_fields = {f for fields in LEG_IDENTITY_FIELDS.values() for f in fields}
    diff = _identity_diff_keys(rung1, rung2)
    illegal = [k for k in diff if k not in ADMITTED_IDENTITY_DIFFS and k not in PROJECTED_IDENTITY_KEYS and k not in leg_fields]
    if illegal:
        raise LadderError(f"{where}: rung-2 identity differs from rung 1 in {illegal} — only "
                          f"{list(ADMITTED_IDENTITY_DIFFS)} (and the per-task maps projected onto the merged tasks) may differ")
    # control atoms in the store, row names everywhere else: normalise before any membership test
    t1, t2 = row_tasks_of(rung1, where=f"{where} rung 1"), row_tasks_of(rung2, where=f"{where} rung 2")
    if not set(t2) <= set(t1) or (task is not None and task not in t2) or not (set(t2) & set(merged_tasks)):
        raise LadderError(f"{where}: rung-2 tasks {t2} must be a subset of rung 1's {t1} filing the merged task"
                          f"{'' if task is None else ' ' + task} (merged {list(merged_tasks)})")
    projected = [t for t in merged_tasks if t in t2]
    # bcb-bfcl-200-widen: the grouped rules — each map restricted to the retained tasks, the
    # optional map present on both stores or neither (a protocol/card mix refuses)
    try:
        p2 = project_identity(rung2, projected, kind="identity")
        p1 = project_identity(rung1, projected, kind="identity")
    except LadderError as e:
        raise LadderError(f"{where}: {e}") from None
    for key, cov in p2["maps_cover"].items():
        if set(cov) != set(t2):
            raise LadderError(f"{where}: rung-2 {key} covers {sorted(cov)}, not its tasks {sorted(t2)}")
    if p1["optional"] != p2["optional"]:
        raise LadderError(f"{where}: rung-2 carries the optional identity key(s) {p2['optional']} against rung 1's "
                          f"{p1['optional']} — a protocol store never merges with a card store")
    for key in PROJECTED_IDENTITY_KEYS[1:]:
        m1, m2 = p1["maps"].get(key), p2["maps"].get(key)
        if m1 is None or m2 is None:
            continue                                   # an optional map absent on both (checked above)
        bad = [t for t in projected if m1.get(t) != m2.get(t)]
        if bad:
            raise LadderError(f"{where}: rung-2 {key} differs from rung 1 on the merged task(s) {bad}: "
                              + "; ".join(f"{t} rung1={m1.get(t)!r} rung2={m2.get(t)!r}" for t in bad))
    for task, fields in LEG_IDENTITY_FIELDS.items():
        for f in fields:
            if task in t2:
                if f not in rung2 or f not in rung1 or rung1[f] != rung2[f]:
                    raise LadderError(f"{where}: rung-2 files {task} but its {f} ({rung2.get(f)!r}) != rung 1's ({rung1.get(f)!r})")
            elif f in rung2:
                raise LadderError(f"{where}: rung 2 carries {f} though it files no {task}")
    s1, s2 = rung1.get("served"), rung2.get("served")
    if not isinstance(s1, Mapping) or not isinstance(s2, Mapping):
        raise LadderError(f"{where}: both rungs must carry a `served` block")
    sdiff = _identity_diff_keys(s1, s2)
    # the tool-call facts follow the legs: they may move only when rung 2 files no BFCL
    files_bfcl = BFCL_TASK in t2
    allowed = set(ADMITTED_SERVED_DIFFS) | (set() if files_bfcl else set(LEG_DERIVED_SERVED_KEYS))
    bad = [k for k in sdiff if k not in allowed]
    if bad:
        raise LadderError(f"{where}: rung-2 `served` differs from rung 1 in {bad} — only {sorted(allowed)} may")
    # the launcher's rule holds on EVERY rung, moved or not: a width stuck at rung 1's S on rung 2 is another regime
    for rung, blk in (("rung 1", s1), ("rung 2", s2)):
        if blk.get("concurrency") != blk.get("max_num_seqs"):
            raise LadderError(f"{where}: {rung} served concurrency {blk.get('concurrency')!r} != its max_num_seqs "
                              f"{blk.get('max_num_seqs')!r} — the launcher sets the driver's width to the resolved S; a "
                              "concurrency that did not follow S is another regime")
    if s2.get("concurrency") != accepted["max_num_seqs"]:
        raise LadderError(f"{where}: rung-2 concurrency {s2.get('concurrency')!r} != the certificate's accepted S "
                          f"{accepted['max_num_seqs']}")


def check_rung_experiment_identity(m1: Mapping[str, Any], m2: Mapping[str, Any], *, where: str,
                                    merged_tasks: Sequence[str], tasks2: Optional[Sequence[str]] = None,
                                    admit_tree_sha: Optional[str] = None) -> None:
    """The FULL experiment identity across rungs: every path of
    :data:`RUNG_EXPERIMENT_PATHS` equal between the rung-1 and rung-2 manifests
    — the model and its revision, the intervention hash, the served object
    digests, the producing tree — and the `sampling` / `thinking` blocks equal
    in their scalars with their per-task map equal on ``merged_tasks``
    (:data:`PROJECTED_MANIFEST_BLOCKS`). A rung-2 store from another object
    build, model or tree refuses — except a rung-2
    ``producer.tree_sha`` equal to EXACTLY ``admit_tree_sha`` when one is given:
    the one difference the operator admitted by name; every other path still
    refuses, and a rung-2 tree that is neither rung 1's nor the admitted sha
    refuses as before."""
    projected = [t for t in merged_tasks if tasks2 is None or t in tasks2]
    # the route's tool-call facts follow the legs a configuration files (see LEG_DERIVED_SERVED_KEYS)
    files_bfcl = tasks2 is None or BFCL_TASK in tasks2
    paths = [path for path in RUNG_EXPERIMENT_PATHS
             if files_bfcl or not (path[0] == C.MANIFEST_KEY_ROUTE and path[1] in LEG_DERIVED_SERVED_KEYS)]
    bad: Dict[str, Tuple[Any, Any]] = {}
    for path in paths:
        a, b = _dig(m1, path), _dig(m2, path)
        if a == b:
            continue
        if tuple(path) == TREE_PATH and admit_tree_sha is not None and b == admit_tree_sha:
            continue        # The ONE admitted difference — rung 2 at exactly the named tree
        bad[".".join(path)] = (a, b)
    # bcb-bfcl-200-widen: the grouped rules per block — scalars with EVERY task map and
    # the derived run-level `sampling.source` stripped; each map on the retained tasks; the
    # per-task provenance per task; the conditional `by_task` present on both or neither
    for block in sorted({b for b, _ in PROJECTED_MANIFEST_BLOCKS}):
        b1, b2 = m1.get(block) or {}, m2.get(block) or {}
        mode1, mode2 = (b1.get("mode"), b2.get("mode")) if block == C.MANIFEST_KEY_SAMPLING else (None, None)
        try:
            q1 = project_identity(b1, projected, kind=block, mode=mode1)
            q2 = project_identity(b2, projected, kind=block, mode=mode2)
        except LadderError as e:
            raise LadderError(f"{where}: {block}: {e}") from None
        for k, pair in projection_differences(q1, q2).items():
            bad[f"{block}.{k}" if k != "<optional keys>" else f"{block}.<optional keys>"] = pair
    if bad:
        raise LadderError(f"{where}: rung-2 experiment identity differs from rung 1 in {sorted(bad)}: " +
                          "; ".join(f"{k} rung1={a!r} rung2={b!r}" for k, (a, b) in list(bad.items())[:4]))


def bind_certificate_to_store(cert: Mapping[str, Any], store: Store) -> None:
    """Every provenance field the certificate carries that a store manifest can
    answer for: ``tree_sha`` == the producer's tree, ``rev`` == the
    model revision, ``hf`` == the model id, ``object_digests[cell]`` == the
    store's SERVED object digest (:func:`served_object_sha256`: the route's attested
    digest, the per-cell map, else the driver's ``provenance.artifact_sha256`` — a
    cell absent from the map must serve none);
    the selection digest is bound through the hashed ``item_selection`` (per
    task) and `merge`'s byte equality; ``cfg_sha256`` has no manifest leaf and
    is validated at launch by `rung_probe.certificate_is_valid`.

    And the ENGINE: the certificate's own
    engine (:func:`certificate_serving_engine` — its ``ladder`` block's) must equal
    the store's canonical reading (:func:`store_serving_engine`). Before this a
    valid NATIVE gemma certificate at (32,768, 14) merged LENS stores at that
    rung silently — the certificate was validated against its own engine's
    ladder, the store against the certificate's numbers, and nothing held the
    two engines to each other; rung-to-rung agreement cannot catch it when
    both rungs sit on the other engine, and the analysis then labelled the
    ladder's engine from the certificate. A cross-engine certificate binds
    nothing; a store whose engine cannot be read binds nothing."""
    m, where = store.manifest, str(store.cell_dir)
    cert_engine, store_engine = certificate_serving_engine(cert), store_serving_engine(store)
    if cert_engine != store_engine:
        raise LadderError(f"{where}: certificate serving engine {cert_engine!r} != the rung-2 store's {store_engine!r} — the "
                          "certificate was issued on the other engine's ladder (its footprint is that engine's, the "
                          "store's is this one's); a cross-engine certificate binds nothing")
    pairs = (("tree_sha", _dig(m, (C.MANIFEST_KEY_PRODUCER, "tree_sha"))),
             ("rev", m.get("model_revision")), ("hf", m.get("model_id")))
    for key, have in pairs:
        if cert[key] != have:
            raise LadderError(f"{where}: certificate {key} {cert[key]!r} != the rung-2 store's {have!r} — the certificate "
                              "was issued for another run")
    want_digest = dict(cert["object_digests"]).get(store.cell)
    # the served object is read where the producer recorded it — route digest, per-cell
    # map, then `provenance.artifact_sha256` (the earliest real manifests carry the route
    # digests null); the recorded sources must agree, and a fitted cell recorded NOWHERE is refused
    # by the inequality below (the certificate names a digest the store cannot vouch for)
    have_digest, have_source = served_object_sha256(m, cell=store.cell)
    if want_digest != have_digest:
        # bound to a local first: a replacement field split across lines is 3.12-only syntax
        source_text = have_source or "no recorded source: route digests and provenance.artifact_sha256 all null"
        raise LadderError(f"{where}: certificate object digest for cell {store.cell!r} {want_digest!r} != the store's served "
                          f"object digest {have_digest!r} (from {source_text}) — another object build")


def check_rung2_store(store: Store, *, selection_block: Mapping[str, Any], cert: Mapping[str, Any]) -> None:
    """A rung-2 store is a validly admitted ladder PARTIAL
    (`ladder_partial_reason`): a PARTIAL marker with exactly `PARTIAL_MARKER_FIELDS`,
    `limits == {"item_selection": 1}`, `n_items == n`, `n_expected_full == 200`;
    the summary's `n == n_items` and `max_tokens == cap_to`; the manifest's
    `item_selection[task] == {n, ids_sha256}`, `cap_overrides_by_task` == the
    certificate's, `served` window/S == the accepted rung; the outcomes' ids ==
    the selection exactly (log vocabulary); every rung-2 draw at `cap_to`."""
    where, task = str(store.cell_dir), store.task
    n, ids = int(selection_block["n"]), list(selection_block["ids"])
    if store.marker_kind != "partial":
        raise LadderError(f"{where}: rung-2 store of {task} carries a {store.marker_kind} marker; a selection run ends "
                          "PARTIAL by contract")
    m = store.marker
    if set(m) != set(C.PARTIAL_MARKER_FIELDS):
        raise LadderError(f"{where}: rung-2 partial marker keys {sorted(m)} != {sorted(C.PARTIAL_MARKER_FIELDS)}")
    if dict(m["limits"]) != dict(LIMITS_LADDER):
        raise LadderError(f"{where}: rung-2 limits {m['limits']!r} != {dict(LIMITS_LADDER)} (a smoke PARTIAL is not a "
                          "ladder store)")
    if m["n_items"] != n or m["n_expected_full"] != FULL_N:
        raise LadderError(f"{where}: rung-2 marker n_items {m['n_items']!r} / n_expected_full {m['n_expected_full']!r} "
                          f"!= {n} / {FULL_N}")
    s = store.summary
    if s.get("n") != n:
        raise LadderError(f"{where}: rung-2 summary n {s.get('n')!r} != {n}")
    if s.get("max_tokens") != selection_block["cap_to"]:
        raise LadderError(f"{where}: rung-2 summary max_tokens {s.get('max_tokens')!r} != cap_to {selection_block['cap_to']}")
    ident = store.identity
    if dict(ident.get("limits") or {}) != dict(LIMITS_LADDER):
        raise LadderError(f"{where}: rung-2 identity limits {ident.get('limits')!r} != {dict(LIMITS_LADDER)}")
    isel = ident.get("item_selection")
    want = {"n": n, "ids_sha256": selection_block["ids_sha256"]}
    if not isinstance(isel, Mapping) or dict(isel.get(task) or {}) != want:
        raise LadderError(f"{where}: rung-2 identity item_selection[{task!r}] {isel.get(task) if isinstance(isel, Mapping) else isel!r} "
                          f"!= {want} — the store was not generated on this selection")
    if dict(ident.get("cap_overrides_by_task") or {}) != dict(cert["cap_overrides_by_task"]):
        raise LadderError(f"{where}: rung-2 identity cap_overrides_by_task {ident.get('cap_overrides_by_task')!r} != the "
                          f"certificate's {dict(cert['cap_overrides_by_task'])}")
    if dict(cert["cap_overrides_by_task"]).get(task) != selection_block["cap_to"]:
        raise LadderError(f"{where}: the certificate's cap for {task} is {dict(cert['cap_overrides_by_task']).get(task)!r}, "
                          f"not cap_to {selection_block['cap_to']}")
    served = ident.get("served") or {}
    acc = cert["accepted"]
    if served.get("max_model_len") != acc["max_model_len"] or served.get("max_num_seqs") != acc["max_num_seqs"]:
        raise LadderError(f"{where}: rung-2 served ({served.get('max_model_len')}, {served.get('max_num_seqs')}) != the "
                          f"certificate's accepted ({acc['max_model_len']}, {acc['max_num_seqs']})")
    want_log = {to_log_id(task, i) for i in ids}
    got_log = {it["item_id"] for it in store.outcomes["items"]}
    if got_log != want_log:
        missing, extra = sorted(want_log - got_log), sorted(got_log - want_log)
        raise LadderError(f"{where}: rung-2 outcomes of {task} != the selection (missing {missing[:5]}, extra {extra[:5]})")
    # the rows must agree with the outcome record ITEM BY ITEM (value, stop reason, at-cap) before they
    # take precedence over rung 1 — an edited rung-2 row is refused here, not merged
    rows, _ = validated_items(store)
    if set(rows) != want_log:
        missing, extra = sorted(want_log - set(rows)), sorted(set(rows) - want_log)
        raise LadderError(f"{where}: rung-2 acc rows of {task} != the selection (missing {missing[:5]}, extra {extra[:5]})")
    bad_cap = [i for i, r in rows.items() if r["extra"].get("max_tokens_used") != selection_block["cap_to"]]
    if bad_cap:
        raise LadderError(f"{where}: {len(bad_cap)} rung-2 rows of {task} were not generated at cap_to "
                          f"{selection_block['cap_to']}, e.g. {bad_cap[:3]}")


def _restamp(row: Mapping[str, Any], **extra_updates: Any) -> Dict[str, Any]:
    r = {k: v for k, v in row.items() if k != "extra"}
    extra = dict(row["extra"])
    extra.update(extra_updates)
    r["extra_json"] = canonical_json(extra)
    return r


def merge(rung1_dir: Union[str, Path], rung2_dir: Union[str, Path], selection_path: Union[str, Path],
          certificate_path: Union[str, Path], out_dir: Union[str, Path], *,
          repo_root: Union[str, Path] = REPO, admit_tree: Optional[str] = None,
          admit_tree_why: Optional[str] = None) -> Dict[str, Any]:
    """`ladder merge` — see the module docstring. Every rung-1 and rung-2 store is
    read through :func:`read_store`, whose root admission refuses a root carrying the
    worker's sampling-validation failure (a poisoned rung-2 PARTIAL never merges);
    the roots are recorded per cell. Returns the merged summary.

    ``admit_tree`` / ``admit_tree_why`` (the CLI's
    :data:`ADMIT_TREE_FLAG` / :data:`ADMIT_TREE_WHY_FLAG`, required together): admit
    EXACTLY ``admit_tree`` (40-hex) as the rung-2 stores' ``producer.tree_sha`` while
    rung 1's differs — the case where the producing tree changed between rungs for a
    named reason (the verifier-transport fix) and nothing else did. Every other
    identity path, every projected block and the certificate binding are checked
    unchanged (the certificate is still issued at the rung-2 tree). Refused: a sha
    that is not the rung-2 stores' actual tree, a sha equal to rung 1's (nothing to
    admit), rung-1 stores that disagree on their own tree, a flag without its
    reason. The summary records ``tree_admission`` (None when not admitted) and, per
    cell, ``experiment[cell].tree_sha`` = rung 1's with ``rung2_tree_sha`` beside it."""
    if (admit_tree is None) != (admit_tree_why is None):
        raise LadderError(f"{ADMIT_TREE_FLAG} and {ADMIT_TREE_WHY_FLAG} are required together — an admitted tree "
                          "without its reason, or a reason without the sha it admits, records nothing usable")
    if admit_tree is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", admit_tree):
            raise LadderError(f"{ADMIT_TREE_FLAG} {admit_tree!r} is not a 40-hex commit sha")
        if not str(admit_tree_why).strip():
            raise LadderError(f"{ADMIT_TREE_WHY_FLAG} is empty — say why rung 2 was produced at {admit_tree[:8]}")
    sel = load_selection(selection_path)
    cert = load_certificate(certificate_path)
    if cert["selection_sha256"] != sel["selection_sha256"]:
        raise LadderError(f"certificate selection_sha256 {cert['selection_sha256'][:12]}… != the selection's "
                          f"{sel['selection_sha256'][:12]}… — rung 2 was certified on another selection")
    validate_certificate(cert, sel["selection"])
    if cert["probed_cell"] not in sel["cells"]:
        raise LadderError(f"certificate probed_cell {cert['probed_cell']!r} is not one of this ladder's cells {list(sel['cells'])}")
    want_caps = {task: block["cap_to"] for task, block in sel["selection"].items()}
    if dict(cert["cap_overrides_by_task"]) != want_caps:
        raise LadderError(f"certificate cap_overrides_by_task {dict(cert['cap_overrides_by_task'])} != {want_caps}")
    cells = list(sel["cells"])
    # the ladder is SINGLE-ENGINE — the rung-1 sources, the rung-2 stores and the
    # certificate name one engine (a comparison across engines measures the engine as well as the
    # intervention). Each rung-1 source is held to the certificate's engine here, through the canonical store reader;
    # each rung-2 store at the binding boundary (`bind_certificate_to_store`)
    cert_engine = certificate_serving_engine(cert)
    # 1. re-hash every rung-1 source and recompute its capped ids
    stores1: Dict[Tuple[str, str], Store] = {}
    pins: Dict[str, Dict[str, Any]] = {}
    recomputed_sources: List[Dict[str, Any]] = []
    for src in sel["sources"]:
        cell, task = src["cell"], src["task"]
        st = read_store(Path(rung1_dir).expanduser() / cell / "cells" / src["store_key"], task, cell=cell)
        engine1 = store_serving_engine(st)
        if engine1 != cert_engine:
            raise LadderError(f"{st.cell_dir}: the rung-1 source's serving engine {engine1!r} != the certificate's "
                              f"{cert_engine!r} — the ladder is single-engine: its rung-1 sources, its rung-2 stores and "
                              "its certificate all name one engine (a cross-engine merge measures the engine, not the cap)")
        stores1[(cell, task)] = st
        for k, sha in st.shas.items():
            if sha != src[k]:
                raise LadderError(f"{st.cell_dir}: {k} {sha[:12]}… != the selection's {str(src[k])[:12]}… — the "
                                  "rung-1 source drifted since `select`")
        if task not in pins:
            pins[task] = load_pin(repo_root, task)
        rec = source_record(st, pins[task], at_cap_reading(st))
        if rec["capped_ids"] != list(src["capped_ids"]) or rec["capped_ids_sha256"] != src["capped_ids_sha256"]:
            raise LadderError(f"{st.cell_dir}: the recomputed capped ids of {task} differ from the selection's source "
                              f"record (recomputed {len(rec['capped_ids'])}, recorded {len(src['capped_ids'])})")
        if rec["identity"] != src["identity"]:
            raise LadderError(f"{st.cell_dir}: the store's identity differs from the selection's source record")
        recomputed_sources.append(rec)
    # 2. recompute the union → byte equality with the selection block
    for task, block in sel["selection"].items():
        if task not in pins:
            raise LadderError(f"selection names {task} but records no sources for it")
        stores_t = {c: stores1[(c, task)] for c in cells if (c, task) in stores1}
        if set(stores_t) != set(cells):
            raise LadderError(f"{task}: sources cover cells {sorted(stores_t)}, not {cells}")
        want = union_selection(task, recomputed_sources, pins[task], stores_t)
        if canonical_json(want) != canonical_json(dict(block)):
            raise LadderError(f"{task}: the recomputed union does not equal the selection block byte for byte "
                              f"(recomputed n {want['n']}, recorded n {block['n']}; recomputed max_prompt_tokens_rung1 "
                              f"{want['max_prompt_tokens_rung1']}, recorded {block['max_prompt_tokens_rung1']})")
    # 3. rung-2 stores: admitted ladder PARTIALs on THIS selection, one `served` across cells
    merged_tasks = sorted(sel["selection"])
    stores2: Dict[Tuple[str, str], Store] = {}
    for task, block in sel["selection"].items():
        for cell in cells:
            st2 = load_store(rung2_dir, cell, task)
            check_rung2_store(st2, selection_block=block, cert=cert)
            check_rung_identity_contract(stores1[(cell, task)].identity, st2.identity, where=str(st2.cell_dir),
                                         accepted=cert["accepted"], merged_tasks=merged_tasks, task=task)
            check_rung_experiment_identity(stores1[(cell, task)].manifest, st2.manifest, where=str(st2.cell_dir),
                                           merged_tasks=merged_tasks, tasks2=row_tasks_of(st2.identity, where=str(st2.cell_dir)),
                                           admit_tree_sha=admit_tree)
            bind_certificate_to_store(cert, st2)
            stores2[(cell, task)] = st2
    # the admitted tree, held to what the stores actually record — rung 2 at EXACTLY the named sha, rung 1
    # at one other sha (the identity check above already refused a rung-2 tree that is neither)
    trees1 = {_dig(st.manifest, TREE_PATH) for st in stores1.values()}
    trees2 = {_dig(st.manifest, TREE_PATH) for st in stores2.values()}
    tree_admission: Optional[Dict[str, Any]] = None
    if admit_tree is not None:
        if len(trees1) != 1:
            raise LadderError(f"{ADMIT_TREE_FLAG}: the rung-1 stores record {len(trees1)} producer trees "
                              f"{sorted(map(str, trees1))} — one rung-1 tree is needed to record the admission against")
        if trees2 != {admit_tree}:
            raise LadderError(f"{ADMIT_TREE_FLAG} {admit_tree[:8]} is not the rung-2 stores' tree "
                              f"{sorted(str(t)[:8] for t in trees2)} — the admission names the sha rung 2 was ACTUALLY produced at")
        (tree1,) = trees1
        if tree1 == admit_tree:
            raise LadderError(f"{ADMIT_TREE_FLAG} {admit_tree[:8]} equals rung 1's tree — the trees agree, nothing to admit; "
                              "drop the flag")
        tree_admission = {"rung1_tree_sha": tree1, "rung2_tree_sha": admit_tree, "why": str(admit_tree_why).strip(),
                          "flag": ADMIT_TREE_FLAG}
    # ONE geometry for every rung-2 store (the served block minus the tool-call facts, which follow the legs a
    # store files), and per merged task the SAME full block in both cells
    geometries = {canonical_json({k: v for k, v in st.identity["served"].items() if k not in LEG_DERIVED_SERVED_KEYS})
                  for st in stores2.values()}
    if len(geometries) != 1:
        raise LadderError(f"the rung-2 cells were served under {len(geometries)} different `served` geometries — one "
                          "geometry for both cells (plan §S2b)")
    for task in merged_tasks:
        per_task = {canonical_json(stores2[(cell, task)].identity["served"]) for cell in cells}
        if len(per_task) != 1:
            raise LadderError(f"{task}: the rung-2 cells were served under different `served` blocks")
    # 4. merge rows: rung-1 for the unselected items, rung-2 (precedence) for the selected
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "schema": LADDER_SCHEMA, "generated_by": "scripts/tinycap_ladder.py merge", "generated_at": _now_iso(),
        "selection_sha256": sel["selection_sha256"], "cells": cells, "tasks": sorted(sel["selection"]),
        "certificate": {"accepted": dict(cert["accepted"]), "rung_source": cert["accepted"]["rung_source"],
                        "n_walked": len(cert["walked"]), "suffix": cert["suffix"], "tree_sha": cert["tree_sha"],
                        "rev": cert["rev"], "hf": cert["hf"], "cfg_sha256": cert["cfg_sha256"],
                        "object_digests": dict(cert["object_digests"])},
        "experiment": {cell: {"model_id": stores2[(cell, t)].manifest.get("model_id"),
                              "model_revision": stores2[(cell, t)].manifest.get("model_revision"),
                              "intervention_hash": stores2[(cell, t)].manifest.get("intervention_hash"),
                              # the producer block keeps rung 1's tree, rung 2's beside it (equal unless admitted)
                              "tree_sha": _dig(stores1[(cell, t)].manifest, TREE_PATH),
                              "rung2_tree_sha": _dig(stores2[(cell, t)].manifest, TREE_PATH),
                              "object_sha256": served_object_sha256(stores2[(cell, t)].manifest, cell=cell)[0],
                              "object_sha256_source": served_object_sha256(stores2[(cell, t)].manifest, cell=cell)[1]}
                       for (cell, t) in stores2 if t == sorted(sel["selection"])[0]},
        "tree_admission": tree_admission,
        "per_cell": {}}
    merged_sets: Dict[str, Dict[str, set]] = {}
    for task, block in sel["selection"].items():
        selected_log = {to_log_id(task, i) for i in block["ids"]}
        order = [to_log_id(task, i) for i in pins[task]["ids"]]        # the pin's order, log vocabulary
        rows_name = marker_filename(task)[:-len(".json")] + ".rows.jsonl"
        for cell in cells:
            st1, st2 = stores1[(cell, task)], stores2[(cell, task)]
            rows1, rows2 = st1.acc_rows(), st2.acc_rows()
            if set(rows1) != set(order):
                raise LadderError(f"{st1.cell_dir}: rung-1 acc rows of {task} are not exactly the {FULL_N} pinned items")
            merged: Dict[str, Dict[str, Any]] = {}
            for item, r in rows1.items():
                if item in selected_log:
                    merged[item] = _restamp(rows2[item], attempt=2, cap_rung=2)
                else:
                    merged[item] = _restamp(r, cap_rung=1)
            if len(merged) != FULL_N:
                raise LadderError(f"{cell}/{task}: merged {len(merged)} items, not {FULL_N}")
            merged_sets.setdefault(task, {})[cell] = set(merged)
            cell_out = out / cell
            cell_out.mkdir(parents=True, exist_ok=True)
            path = cell_out / rows_name
            with open(path, "w", encoding="utf-8") as f:
                for item in order:
                    f.write(json.dumps(merged[item], sort_keys=True, allow_nan=False) + "\n")
            k1 = sum(1 for r in rows1.values() if float(r["value"]) == 1.0)
            km = sum(1 for r in merged.values() if float(r["value"]) == 1.0)
            residual = sum(1 for r in rows2.values() if r["extra"].get("stop_reason") == STOP_REASON_AT_CAP)
            summary["per_cell"].setdefault(cell, {})[task] = {
                "n": FULL_N, "n_selected": block["n"], "cap_from": block["cap_from"], "cap_to": block["cap_to"],
                "k_rung1": k1, "acc_rung1": k1 / FULL_N, "k_merged": km, "acc_merged": km / FULL_N,
                "n_at_cap_rung1": int(st1.summary["n_at_cap"]), "n_at_cap_residual": residual,
                "n_model_length_rung2": sum(1 for r in rows2.values()
                                            if r["extra"].get("stop_reason") == STOP_REASON_MODEL_LENGTH),
                "rung1_store_key": st1.key, "rung2_store_key": st2.key,
                "rung1_shas": dict(st1.shas), "rung2_shas": dict(st2.shas),
                # the roots both stores were admitted under (`root_admissibility_reason`, re-checked by the analysis)
                "rung1_root": str(st1.root), "rung2_root": str(st2.root),
                "merged_rows": str(path), "merged_rows_sha256": sha256_file(path)}
    for task, by_cell in merged_sets.items():
        sets = list(by_cell.values())
        if any(s != sets[0] for s in sets[1:]):
            raise LadderError(f"{task}: the merged item sets differ across cells")
    (out / MERGED_SUMMARY_FILENAME).write_text(json.dumps(summary, indent=1, sort_keys=True, allow_nan=False) + "\n",
                                                encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect", help="the truncation detector over a pulled rung-1 dir")
    d.add_argument("--rung1", required=True)
    d.add_argument("--cells", default=None, help="comma-separated; default: every <cell>/cells/ under --rung1")
    d.add_argument("--roster", default=None, help="the lane roster: the wave's RUNNABLE legs are the leg set (with --wave)")
    d.add_argument("--wave", default=None, help="the roster wave of --rung1 (its runnable legs; a blocked leg is not a ladder leg)")
    s = sub.add_parser("select", help="write ladder_selection.json for the fired legs")
    s.add_argument("--rung1", required=True)
    s.add_argument("--cells", default=None)
    s.add_argument("--out", required=True)
    s.add_argument("--repo-root", default=str(REPO))
    s.add_argument("--roster", default=None, help="the lane roster: the wave's RUNNABLE legs are the leg set (with --wave)")
    s.add_argument("--wave", default=None, help="the roster wave of --rung1; without --roster/--wave the leg set is analysis.json's beside the stores, else both legs")
    m = sub.add_parser("merge", help="recompute + merge rung 1 and rung 2")
    m.add_argument("--rung1", required=True)
    m.add_argument("--rung2", required=True)
    m.add_argument("--selection", required=True)
    m.add_argument("--certificate", required=True, help="the rung-2 run's rung_probe.json")
    m.add_argument("--out", required=True)
    m.add_argument("--repo-root", default=str(REPO))
    m.add_argument(ADMIT_TREE_FLAG, default=None, metavar="SHA40",
                   help="D73: admit EXACTLY this sha as the rung-2 producer.tree_sha while rung 1's differs (every "
                        f"other identity path must still be equal); requires {ADMIT_TREE_WHY_FLAG}")
    m.add_argument(ADMIT_TREE_WHY_FLAG, default=None, metavar="TEXT",
                   help=f"why rung 2 was produced at {ADMIT_TREE_FLAG}'s sha; recorded in merged_summary.json")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    if a.cmd == "merge" and (a.admit_tree is None) != (a.admit_tree_why is None):
        p.error(f"{ADMIT_TREE_FLAG} and {ADMIT_TREE_WHY_FLAG} are required together")
    cells = [c for c in a.cells.split(",") if c] if getattr(a, "cells", None) else None
    if a.cmd in ("detect", "select"):
        tasks, tasks_source = ladder_tasks(a.rung1, roster=a.roster, wave=a.wave)
        print(f"[ladder] leg set {[t.split('/')[-1] for t in tasks]} — {tasks_source}")
    if a.cmd == "detect":
        det = detect(a.rung1, cells=cells, tasks=tasks, tasks_source=tasks_source)
        print(json.dumps(det, indent=1, sort_keys=True))
        print(f"[ladder] {'FIRED' if det['fired_any'] else 'no leg fired'}: " + ", ".join(
            f"{t.split('/')[-1]}={'fired' if v['fired'] else 'ok'} (" +
            "; ".join(f"{c} {r['n_at_cap']}/{r['n_stop_reason']}" for c, r in v["cells"].items()) + ")"
            for t, v in det["tasks"].items()))
        return 0
    if a.cmd == "select":
        doc = build_selection(a.rung1, repo_root=a.repo_root, cells=cells, tasks=tasks, tasks_source=tasks_source)
        path = write_selection(doc, a.out)
        if doc["selection"]:
            print(f"[ladder] selection → {path}: " + ", ".join(
                f"{t.split('/')[-1]} n_selected={b['n_selected']} cap {b['cap_from']}→{b['cap_to']} "
                f"max_prompt_tokens_rung1={b['max_prompt_tokens_rung1']}" for t, b in doc["selection"].items())
                + f"; selection_sha256 {doc['selection_sha256'][:12]}…")
        else:
            print(f"[ladder] no leg fired; selection written empty → {path}")
        return 0
    if a.cmd == "merge":
        summ = merge(a.rung1, a.rung2, a.selection, a.certificate, a.out, repo_root=a.repo_root,
                     admit_tree=a.admit_tree, admit_tree_why=a.admit_tree_why)
        if summ["tree_admission"] is not None:
            ta = summ["tree_admission"]
            print(f"[ladder] tree admission ({ta['flag']}): rung 2 produced at {ta['rung2_tree_sha'][:8]} "
                  f"(rung 1 at {ta['rung1_tree_sha'][:8]}) — {ta['why']}")
        for cell, tasks in summ["per_cell"].items():
            for task, r in tasks.items():
                print(f"[ladder] {cell} {task.split('/')[-1]}: acc rung-1 {r['k_rung1']}/{r['n']} → merged "
                      f"{r['k_merged']}/{r['n']}; at-cap rung-1 {r['n_at_cap_rung1']}, residual {r['n_at_cap_residual']}")
        print(f"[ladder] merged summary → {Path(a.out).expanduser() / MERGED_SUMMARY_FILENAME}")
        return 0
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LadderError as e:
        print(f"[ladder] REFUSED: {e}", file=sys.stderr)
        sys.exit(3)
