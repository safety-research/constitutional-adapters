"""Result rows, cell keys, resume, consolidation, config/env hashing (§2.3).

Stage-0 FROZEN interfaces. Bodies are ``NotImplementedError`` until the
implementation lands; ROW_COLUMNS / ResultRow / cell-key format /
CellStore layout are the binding contract, pinned by
``tests/test_capeval_protocol.py``.

Cell layout on shared storage (OUT = .../out/capeval/<sweep>/):
  cells/<cell_key>/rows.jsonl        append-only per-item rows (resume)
  cells/<cell_key>/results.json      summary, written LAST via
                                     tmp+fsync+replace — its existence is
                                     the ONLY cell-done resume marker
  cells/<cell_key>/manifest.json     resolved cell config, full hashes,
                                     resolved intervention spec
  cells/<cell_key>/generation_complete.json   EvalPlus cells only (F8)
  cache/klatoms/<digest16>.safetensors        Tier-0 base atom caches (§4.3)
  rows.parquet                       consolidate(), run by the report step
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Set, Tuple, Union

from sisteer.capeval.manifest import canonical_json, sha256_hex16


class StoreError(ValueError):
    """A result-store refusal (resume mismatch, malformed rows, …)."""


# Reserved (non-registry) values for the `domain` column: `_meta` carries
# the F7 timestamp rows and other bookkeeping; `target` carries the target
# -plugin rows (the target metric is its own §2.1 axis, never a capability
# domain). Aggregation reads registry domains only and refuses anything
# else outside this set (registered-metric rule, DESIGN §3).
META_DOMAIN = "_meta"
TARGET_DOMAIN = "target"
RESERVED_DOMAINS = (META_DOMAIN, TARGET_DOMAIN)

# --- Row schema (frozen; §2.3) ----------------------------------------------

# One row per (cell, item, metric). Parquet columns, in this exact order —
# all present, nullable where the type below says Optional.
ROW_COLUMNS = (
    "sweep",
    "run_id",
    "model_id",
    "model_revision",
    "arm_name",
    "kind",
    "coeff",
    "intervention_hash",
    "engine",
    "tier",
    "task",
    "task_version",
    "domain",
    "item_id",
    "trial_id",
    "seed",
    "attempt",
    "position_order",
    "metric",
    "value",
    "extra_json",
    "harness_config_hash",
    "env_digest",
    "slurm_job_id",
    "created_at",
    "wall_clock_s",
)


@dataclass
class ResultRow:
    """One (cell, item, metric) row. Field order == ROW_COLUMNS.

    No field has a default: every writer states every column explicitly
    (nullable columns take an explicit None). Timestamps are first-class
    rows (F7): metric = ts_metric(event), value = epoch seconds.
    NLL tasks emit per-window nll_sum + token_count as separate metric
    rows so aggregates recompute Σnum/Σden from rows (F14).
    """

    sweep: str
    run_id: str
    model_id: str
    model_revision: str
    arm_name: str
    kind: str
    coeff: Optional[float]  # None for kind "none"
    intervention_hash: str
    engine: str  # "hf" | "vllm" — fingerprint refuses cross-engine pairs
    tier: int
    task: str
    task_version: str
    domain: str
    item_id: str
    trial_id: int
    seed: int
    attempt: int  # 0 = first run; 1 = confirmation rerun (F16)
    position_order: Optional[int]  # item's index in this run's order (§7.1)
    metric: str
    value: float
    extra_json: Optional[str]  # canonical-JSON blob for metric extras
    harness_config_hash: str
    env_digest: str
    slurm_job_id: Optional[str]
    created_at: float  # epoch seconds
    wall_clock_s: Optional[float]


# --- Cell keys ---------------------------------------------------------------


def cell_key(
    arm: str,
    tier_or_task: str,
    trial: int,
    attempt: int,
    intervention_hash: str,
    harness_config_hash: str,
) -> str:
    """The frozen cell-dir name (§2.3, F8):

        f"{arm}__{tier_or_task}__t{trial}__a{attempt}__{ihash8}__{chash8}"

    where ihash8/chash8 are the 8-hex prefixes of the two hashes — a
    changed prompt/config/task version can never resume onto an old cell
    dir."""
    for label, val in (("arm", arm), ("tier_or_task", tier_or_task)):
        if not isinstance(val, str) or not val:
            raise StoreError(f"cell_key: {label} must be a non-empty string")
        if "__" in val or "/" in val:
            raise StoreError(f"cell_key: {label} {val!r} contains a reserved "
                             "separator ('__' or '/')")
    for label, val in (("trial", trial), ("attempt", attempt)):
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            raise StoreError(f"cell_key: {label} must be a non-negative int, "
                             f"got {val!r}")
    for label, h in (("intervention_hash", intervention_hash),
                     ("harness_config_hash", harness_config_hash)):
        if (not isinstance(h, str) or len(h) < 8
                or any(c not in "0123456789abcdef" for c in h[:8].lower())):
            raise StoreError(f"cell_key: {label} must be >=8 hex chars, "
                             f"got {h!r}")
    return (f"{arm}__{tier_or_task}__t{trial}__a{attempt}"
            f"__{intervention_hash[:8]}__{harness_config_hash[:8]}")


def ts_metric(event: str) -> str:
    """Metric name for a first-class timestamp row (F7): f"ts_{event}".
    Events per cell: job_start, model_ready, per-tier start/end, cell_end
    (+ the sandbox wave's start/provision/exec/transfer stamps, §10).
    Timestamp rows use the reserved META_DOMAIN in their `domain` column."""
    if not isinstance(event, str) or not event or any(c.isspace() for c in event):
        raise StoreError(f"ts_metric: event must be a non-empty token, "
                         f"got {event!r}")
    return f"ts_{event}"


# --- Per-cell store -----------------------------------------------------------


#: pass as ``recorder=`` to build a ``CellStore`` with NO recorder and no default
#: resolution (a caller that resolved its own and got None — ``runtime.resolve_cell``).
NO_RECORDER = object()


def _default_recorder(key: str, row_constants: Optional[Mapping[str, Any]]) -> Optional[Any]:
    """The recorder a ``CellStore`` resolves when handed none: the registry
    hooks' default (stdlib-only import), inside the same guard every hook
    call runs under — a failure here is one stderr line, never a store
    that fails to open."""
    try:
        from sisteer.registry import hooks as _registry_hooks
        return _registry_hooks.default_store_recorder(key, row_constants)
    except Exception as e:  # noqa: BLE001 — by contract: a lost record never costs a run
        import sys
        print(f"[registry] store {key}: default recorder not built "
              f"({type(e).__name__}: {e}); the run continues", file=sys.stderr, flush=True)
        return None


class CellStore:
    """Owns one cells/<cell_key>/ directory: append-only rows, resume
    validation, and the atomic done-marker discipline."""

    #: the four identity fields the resume gate compares (§2.3). Every
    #: caller's resolved spec must carry all four, so this tuple can only
    #: grow by a change to every driver — see CONDITIONAL_RESUME_KEYS for a
    #: field only some drivers record.
    RESUME_KEYS = ("intervention_hash", "harness_config_hash",
                   "model_revision", "engine")

    #: Immutable provenance the resume gate compares WHEN THE EXISTING
    #: MANIFEST RECORDS IT (E14 B, tinycap PW7). ``producer`` names the tree
    #: the served driver ran from (``contracts.MANIFEST_KEY_PRODUCER``:
    #: ``{tree_sha, driver}``); it is NOT part of the hashed identity, so a
    #: store keys the same at two revisions and the driver would otherwise
    #: append rows from revision B onto revision A's and atomically rewrite
    #: the manifest with the NEW producer — one store holding two revisions,
    #: labelled as the newest, which the EvalPlus merge's producer gate then
    #: trusts. Keyed on PRESENCE in the recorded manifest, not on the tuple
    #: alone: a manifest without the block is a legacy or HF-path store and
    #: validates exactly as before (no driver but the served one writes one),
    #: while a store that DOES name its tree may only be resumed by that
    #: tree, or by a spec that names the same one. The converse (PW8, E16 B):
    #: a spec that NAMES a producer may not resume a NON-EMPTY store whose
    #: manifest records none — the rows of a producer-less partial store
    #: come from a revision nobody recorded, and rewriting its manifest with
    #: this run's block would label them as this tree's (the direct driver
    #: never passes the launcher's store audit); only an EMPTY store (a
    #: manifest and nothing else) may gain the block. HF-path callers name
    #: no producer and are untouched by either direction.
    CONDITIONAL_RESUME_KEYS = ("producer",)

    def __init__(self, sweep_dir: Union[str, Path], key: str,
                 row_constants: Optional[Mapping[str, Any]] = None,
                 recorder: Optional[Any] = None) -> None:
        """``row_constants`` (integration seam, capeval_run.py docstring):
        the cell-constant ResultRow columns (the tier0 _TEMPLATE_KEYS set
        — sweep/run_id/model_id/model_revision/arm_name/kind/coeff/
        intervention_hash/engine/trial_id/seed/attempt/
        harness_config_hash/env_digest/slurm_job_id). Required for
        ``append_ts``; runtime.resolve_cell supplies it and also archives
        it in the cell manifest.json as "row_constants" (the EvalPlus
        exec wave builds its execution rows from that copy).

        ``recorder`` (results-registry native emission, additive): an
        object with ``on_manifest(resolved_spec, manifest_path)``,
        ``on_row(row_dict)``, ``on_generation_complete(path)`` and
        ``on_results(summary, results_path)`` — normally a
        ``sisteer.registry.hooks.StoreRecorder`` built by
        ``runtime.resolve_cell``. Every call happens AFTER the store's own
        write and inside a guard that prints one line and never raises, so
        the store's bytes, paths and exceptions are identical with and
        without one (``tests/test_registry_emit_store.py`` proves it
        byte-for-byte).

        None (the default) RESOLVES a recorder (every producer emits
        without opting in):
        ``sisteer.registry.hooks.default_store_recorder`` — the one switch
        (``SISTEER_REGISTRY_DISABLED`` / ``REGISTRY_EMIT`` / a resolving
        spool) and the job's own pending identity
        (``$SISTEER_REGISTRY_RUN_IDENTITY``, written by the job at start);
        when either is absent the result is None and the store is exactly
        the old one. The wave drivers (``petri_run``, ``cell_controls_run``,
        ``served_capability_run``, ``misalign.rows``) therefore record
        without a line of change. A caller that already resolved (or
        deliberately declined) a recorder passes it, or :data:`NO_RECORDER`
        to skip the default — ``runtime.resolve_cell`` does both."""
        self.sweep_dir = Path(sweep_dir)
        self.key = key
        self.row_constants = dict(row_constants) if row_constants is not None else None
        if recorder is NO_RECORDER:
            recorder = None
        elif recorder is None:
            recorder = _default_recorder(key, self.row_constants)
        self.recorder = recorder

    def _record(self, hook: str, *args: Any) -> None:
        """Call one recorder hook after a store write; a failing recorder
        costs one stderr line, never the run (registry contract)."""
        rec = self.recorder
        if rec is None:
            return
        try:
            getattr(rec, hook)(*args)
        except Exception as e:  # noqa: BLE001 — by contract: a lost record never costs a run
            import sys
            print(f"[registry] store {self.key}: {hook} failed "
                  f"({type(e).__name__}: {e}); the run continues", file=sys.stderr, flush=True)

    @property
    def cell_dir(self) -> Path:
        return self.sweep_dir / "cells" / self.key

    @property
    def rows_path(self) -> Path:
        return self.cell_dir / "rows.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.cell_dir / "manifest.json"

    @property
    def results_path(self) -> Path:
        return self.cell_dir / "results.json"

    def append_row(self, row: ResultRow) -> None:
        """Append one row to rows.jsonl (flush + fsync per row — per-item
        resume must survive a SIGKILL mid-cell). NaN/inf values are a
        refusal at the WRITE (allow_nan=False): a non-finite metric would
        otherwise flow through pairing into a bootstrap whose quantiles
        are NaN and classify as a silent SCREEN-PASS."""
        if not isinstance(row, ResultRow):
            raise StoreError(f"append_row takes a ResultRow, got "
                             f"{type(row).__name__}")
        self.cell_dir.mkdir(parents=True, exist_ok=True)
        try:
            line = json.dumps(dataclasses.asdict(row), sort_keys=True,
                              allow_nan=False)
        except ValueError as e:
            raise StoreError(
                f"append_row: non-finite value in row (task={row.task!r}, "
                f"item={row.item_id!r}, metric={row.metric!r}): {e} — a "
                "NaN/inf measurement is a scorer failure to refuse, not a "
                "row to store") from e
        with open(self.rows_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        if self.recorder is not None:
            self._record("on_row", json.loads(line))

    def _parsed_rows(self) -> list:
        """Parsed rows.jsonl dicts. A torn FINAL line (a write cut mid-way
        by preemption — the normal case on `low` QoS) is truncated away,
        with a printed notice, so the interrupted item simply reruns; a
        corrupt NON-final line means the file is damaged and is a refusal."""
        if not self.rows_path.exists():
            return []
        text = self.rows_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        rows, torn_at = [], None
        for i, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1 and not text.endswith("\n"):
                    torn_at = i
                    break
                raise StoreError(
                    f"{self.rows_path}: corrupt row at line {i + 1} (not the "
                    "final line — this is file damage, not a torn append; "
                    "refusing to resume)")
        if torn_at is not None:
            keep = "".join(l + "\n" for l in lines[:torn_at])
            tmp = self.rows_path.with_name(self.rows_path.name + ".tmp")
            tmp.write_text(keep, encoding="utf-8")
            _fsync_replace(tmp, self.rows_path)
            print(f"[capeval.store] {self.rows_path}: truncated a torn final "
                  f"line (interrupted append); item will rerun")
        return rows

    def completed_keys(self) -> Set[Tuple[str, str, str]]:
        """(task, item_id, metric) triples already filed in rows.jsonl —
        completion identity is per task AND metric, so this is the ONLY
        safe primitive for resume skips."""
        return {(r["task"], r["item_id"], r["metric"])
                for r in self._parsed_rows()}

    def rows_of_task(self, task: str) -> list:
        """The parsed rows.jsonl dicts filed under ``task``, in file order —
        every column, not just the completion key — for a caller that must
        VALIDATE the existing rows before it skips them (the EvalPlus
        converter checks their ``task_version`` / ``metric_def`` against the
        definition it writes, PW13 / E21 M2: a resume that dedupes on
        ``(task, item_id, metric)`` alone would leave rows under one
        definition and a summary under another). Read-only."""
        return [r for r in self._parsed_rows() if r.get("task") == task]

    def completed_item_ids(self, task: Optional[str] = None,
                           required_metrics: Optional[Sequence[str]] = None
                           ) -> Set[str]:
        """item_ids safe to skip on restart, AFTER validate_resume().

        The resume form is ``completed_item_ids(task, required_metrics)``:
        an item counts as complete only when EVERY metric in the task's
        required set is present — a bare item_id is ambiguous across tasks
        (tinyMMLU/"0" vs tinyArc/"0") and across companion metrics
        (nll_sum filed, token_count lost to a crash). The zero-argument
        form returns the cell-global bare-id set for bookkeeping/diagnostics
        ONLY — never gate a resume on it.
        """
        if task is None and required_metrics is None:
            return {r["item_id"] for r in self._parsed_rows()}
        if task is None or not required_metrics:
            raise StoreError(
                "completed_item_ids: the scoped form takes BOTH a task and "
                "its non-empty required metric set — completion identity is "
                "(task, item, metric), review r1 stats-core #3")
        keys = self.completed_keys()
        by_item: dict = {}
        for t, item, metric in keys:
            if t == task:
                by_item.setdefault(item, set()).add(metric)
        need = set(required_metrics)
        return {item for item, have in by_item.items() if need <= have}

    #: Files that are never store CONTENT: a writer's temp file (tmp + fsync
    #: + replace leaves ``*.tmp`` behind only when it died mid-write) and an
    #: editor backup. Everything else under the cell dir is a durable
    #: artifact a resume could reuse or relabel.
    _NON_CONTENT_SUFFIXES = (".tmp", "~")

    def _content_artifacts(self) -> list:
        """Every DURABLE artifact under the cell dir other than
        ``manifest.json`` — relative POSIX paths, sorted (E17 B, PW10).
        Content is not just rows and ``results.json``: ``done/`` /
        ``partial/`` / ``summary/`` markers, ``generation_complete.json``,
        ``results_pending.json``, the EvalPlus ``evalplus/<dataset>/``
        sidecars (``generations.jsonl`` / ``samples.jsonl`` /
        ``gen_meta.json``), Inspect logs under ``private/``,
        ``evidence.json``, ``gen_stats.json`` — a resume REUSES every one of
        them (the EvalPlus runner drops every id it finds in the sidecar from
        its todo), so adopting a store on the strength of "no rows" attributes
        another revision's generations to this one. Enumerated by walking the
        directory, excluding only the manifest, temp files
        (:data:`_NON_CONTENT_SUFFIXES`) and a torn final ``rows.jsonl`` line
        (repaired first; a rows file the repair emptied is not content). A
        manifest by itself is an EMPTY store: nothing in it can be mislabelled.

        THE CHECK SEES THE READER'S WORLD (fifth review, 2026-09-07). This used
        to walk with ``Path.rglob`` and keep only ``is_file()`` entries: a
        DIRECTORY symlink (a partial restore whose ``private/`` points at
        retained spools) is not descended by rglob and is not a file, so a
        POPULATED store read as empty here while the generation reader — which
        follows the link — read it as full, and the empty-store recovery
        carved out for "the run never reached its evidence" converted an
        invalid populated store into an apparently valid one with nothing
        regenerated. Now the walk never follows directory links but COUNTS
        them as content (``<rel>/``), counts a DANGLING link as content (it
        names something the reader would try to read), and surfaces a
        directory it cannot traverse instead of silently skipping it — the
        check fails CLOSED wherever it cannot see what the reader sees."""
        if not self.cell_dir.is_dir():
            return []
        self._parsed_rows()   # repairs a torn final line: alone it is not content
        root = self.cell_dir

        def _surface(err: OSError) -> None:
            raise StoreError(
                f"{root}: cannot enumerate the store's contents "
                f"({type(err).__name__}: {err}) — a store whose contents cannot be seen "
                "is not known to be empty; refusing to classify it") from err

        out = []
        for dirpath, dirnames, filenames in os.walk(root, onerror=_surface, followlinks=False):
            here = Path(dirpath)
            for d in list(dirnames):
                p = here / d
                if p.is_symlink():
                    # a linked directory: not descended, never invisible — the
                    # reader follows it, so it is content (or a dangling link
                    # the reader would trip over — content either way)
                    out.append(p.relative_to(root).as_posix() + "/")
                    dirnames.remove(d)
            for f in filenames:
                p = here / f
                rel = p.relative_to(root).as_posix()
                if rel == self.manifest_path.name or p.name.endswith(self._NON_CONTENT_SUFFIXES):
                    continue
                if p.is_symlink() and not p.exists():
                    out.append(rel)   # dangling: the reader would try it; fail closed
                    continue
                if p == self.rows_path and p.stat().st_size == 0:
                    continue          # the torn-line repair left it empty
                out.append(rel)
        return sorted(out)

    def _has_content(self) -> bool:
        """True iff the store holds ANY durable artifact other than its
        manifest (:meth:`_content_artifacts`)."""
        return bool(self._content_artifacts())

    def validate_resume(self, resolved_spec: Mapping[str, Any]) -> None:
        """Refuse-on-mismatch resume gate (§2.3): the FULL
        intervention_hash, harness_config_hash, model revision, and engine
        recorded in the cell manifest.json must equal the resolved run
        spec before anything is skipped; any mismatch is a refusal naming
        both values. Plus CONDITIONAL_RESUME_KEYS — immutable provenance
        (``producer``) compared when the RECORDED manifest carries it (E14
        B), and refused when the recorded manifest LACKS it while the spec
        names one and the store is non-empty (E16 B); a legacy or HF-path
        store whose spec names none validates exactly as before."""
        missing = [k for k in self.RESUME_KEYS if k not in resolved_spec]
        if missing:
            raise StoreError(f"validate_resume: resolved spec missing "
                             f"{missing} — cannot validate a resume without "
                             "the full identity fields")
        if not self.manifest_path.exists():
            artifacts = self._content_artifacts()
            if artifacts:
                raise StoreError(
                    f"{self.cell_dir}: store artifacts exist "
                    f"({_some(artifacts)}) but manifest.json "
                    "is missing — cannot validate their provenance; refusing "
                    "to resume onto them (a resume would reuse them under this "
                    "run's identity; E17 B)")
            return  # fresh cell — nothing to skip, nothing to validate
        recorded = json.loads(self.manifest_path.read_text())
        for k in self.RESUME_KEYS:
            if k not in recorded:
                raise StoreError(f"{self.manifest_path}: missing identity "
                                 f"field {k!r} — refusing to resume")
            if recorded[k] != resolved_spec[k]:
                raise StoreError(
                    f"{self.cell_dir}: resume mismatch on {k!r}: cell "
                    f"manifest records {recorded[k]!r} but the resolved run "
                    f"spec says {resolved_spec[k]!r} — refusing to resume "
                    "(F8)")
        for k in self.CONDITIONAL_RESUME_KEYS:
            if k not in recorded:
                artifacts = self._content_artifacts() if k in resolved_spec else []
                if artifacts:
                    raise StoreError(
                        f"{self.cell_dir}: the cell manifest records no {k!r} "
                        f"but this run names {resolved_spec[k]!r} and the "
                        f"store already holds artifacts ({_some(artifacts)}) "
                        "— they were produced by a revision nobody recorded, "
                        "and resuming would rewrite the manifest with this "
                        f"run's {k!r}, reuse them (persisted generations, "
                        "markers, rows) and label them as its own; refusing. "
                        "Generate into a fresh out root (E16 B / E17 B)")
                continue        # a legacy / HF-path store, or an empty one: validated as before
            if k not in resolved_spec:
                raise StoreError(
                    f"{self.cell_dir}: the cell manifest records {k!r} "
                    f"{recorded[k]!r} but this run's resolved spec names none "
                    f"— refusing to resume (a driver that cannot say which "
                    f"tree it runs from would rewrite the manifest's {k!r} "
                    "and hide that the rows come from two revisions; E14 B)")
            if recorded[k] != resolved_spec[k]:
                raise StoreError(
                    f"{self.cell_dir}: resume mismatch on {k!r}: cell "
                    f"manifest records {recorded[k]!r} but this run is "
                    f"{resolved_spec[k]!r} — refusing to resume; the rows in "
                    "this store were produced by another revision and it is "
                    "not identity, so nothing else would have caught it. Run "
                    "from the tree that produced them, or generate into a "
                    "fresh out root (E14 B)")
        self._parsed_rows()  # repairs a torn tail before any skip decision

    def write_cell_manifest(self, resolved_spec: Mapping[str, Any]) -> None:
        """Write manifest.json: resolved cell config incl. full hashes and
        the resolved intervention spec archived verbatim (§2.1). If a
        manifest already exists it must agree on the identity fields
        (validate_resume semantics) — overwriting a different identity is
        exactly the stale-reuse F8 exists to prevent."""
        missing = [k for k in self.RESUME_KEYS if k not in resolved_spec]
        if missing:
            raise StoreError(f"write_cell_manifest: resolved spec missing "
                             f"identity field(s) {missing}")
        if self.manifest_path.exists():
            self.validate_resume(resolved_spec)
        _write_json_atomic(self.manifest_path, dict(resolved_spec))
        self._record("on_manifest", dict(resolved_spec), str(self.manifest_path))

    def write_generation_complete(self) -> None:
        """EvalPlus cells only (F8): atomic marker after generation;
        results.json still only appears after execution rows return."""
        _write_json_atomic(self.cell_dir / "generation_complete.json",
                           {"generation_complete": True,
                            "written_at": time.time()})
        self._record("on_generation_complete",
                     str(self.cell_dir / "generation_complete.json"))

    def generation_complete(self) -> bool:
        return (self.cell_dir / "generation_complete.json").exists()

    def note_generated(self, task: str, item_ids: Sequence[str], uri: str) -> None:
        """A judged task's GENERATED items (their raw generations sit in a private
        spool, ``uri``; the rows come later, from the judge) told to the recorder:
        the results registry files ONE null-valued ``item_result`` per generated
        item so the run's shard union holds the plan's item count at completion --
        a judged route that opens no run at generation left the registry with
        serving + completion patches alone and every such run REFUSED ("no
        item_result rows and no run_state"). Writes NO store byte: the store is
        byte-identical with a recorder or without one."""
        self._record("on_generated", str(task), [str(i) for i in item_ids], str(uri))

    def write_results(self, summary: Mapping[str, Any]) -> None:
        """Write results.json LAST via tmp+fsync+replace (the stage2_run
        _write_json pattern). Its existence is the ONLY cell-done marker."""
        _write_json_atomic(self.results_path, dict(summary))
        self._record("on_results", dict(summary), str(self.results_path))

    def is_done(self) -> bool:
        """True iff results.json exists — never any other heuristic."""
        return self.results_path.exists()

    def append_ts(self, event: str, epoch_s: float) -> None:
        """File an F7 first-class timestamp row (metric=ts_<event>,
        value=epoch seconds, domain=META_DOMAIN) with the resolved cell's
        row-constant fields — the driver seam (capeval_run.py docstring).
        Refuses when the store was built without row_constants."""
        if self.row_constants is None:
            raise StoreError(
                "append_ts: this CellStore has no row_constants — build it "
                "through runtime.resolve_cell (integration seam), or pass "
                "row_constants to the constructor")
        self.append_row(ResultRow(
            **self.row_constants,
            tier=-1,  # cell-level bookkeeping, not a tier's measurement
            task="_cell",
            task_version="_meta",
            domain=META_DOMAIN,
            item_id="-",
            position_order=None,
            metric=ts_metric(event),
            value=float(epoch_s),
            extra_json=None,
            created_at=time.time(),
            wall_clock_s=None,
        ))


def _some(names: Sequence[str], n: int = 6) -> str:
    """``a, b, c (+4 more)`` — the first few artifact names for a refusal."""
    names = list(names)
    head = ", ".join(names[:n])
    return head if len(names) <= n else f"{head} (+{len(names) - n} more)"


def _fsync_replace(tmp: Path, dst: Path) -> None:
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    # fsync the directory so the rename itself is durable
    dirfd = os.open(dst.parent, os.O_RDONLY)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """The stage2_run `_write_json` pattern: temp + fsync + os.replace —
    existence must mean fully-published (resume-marker discipline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str),
                   encoding="utf-8")
    _fsync_replace(tmp, path)


# --- Sweep-level -------------------------------------------------------------


# Explicit arrow types per column — never inferred, so an all-null
# nullable column can't flip type between sweeps.
_STRING_COLS = frozenset({
    "sweep", "run_id", "model_id", "model_revision", "arm_name", "kind",
    "intervention_hash", "engine", "task", "task_version", "domain",
    "item_id", "metric", "extra_json", "harness_config_hash", "env_digest",
    "slurm_job_id",
})
_INT_COLS = frozenset({"tier", "trial_id", "seed", "attempt", "position_order"})
_FLOAT_COLS = frozenset({"coeff", "value", "created_at", "wall_clock_s"})


def consolidate(
    sweep_dir: Union[str, Path],
    skip_incomplete: bool = False,
    exclude_keys_by_cell: Optional[
        Mapping[str, Set[Tuple[str, str, str]]]
    ] = None,
) -> Path:
    """Fold every DONE cell's rows.jsonl into <sweep_dir>/rows.parquet
    with exactly ROW_COLUMNS (nullable columns nullable). Run by the
    report step; returns the parquet path.

    results.json is the ONLY cell-done marker (§2.3), so a cell without
    one — preempted, failed, or stale-partial — must never contribute
    rows to consolidated results (partial-cell rows either bias the
    fingerprint when the legs happen to share a prefix, or blow up
    pairing at report time after a clean rerun). The default is a
    REFUSAL naming every incomplete cell; ``skip_incomplete=True``
    excludes them with a printed notice instead (for mid-campaign
    inspection, never the report path).

    ``exclude_keys_by_cell`` is an explicit item-tombstone mapping used by
    Petri terminal auditor refusals. Matching stored rows are omitted from the
    consolidated score frame; their append-only source evidence remains in the
    cell store and the refusal sidecar. A tombstone is never inferred.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    sweep_dir = Path(sweep_dir)
    cells_dir = sweep_dir / "cells"
    if not cells_dir.is_dir():
        raise StoreError(f"consolidate: no cells/ under {sweep_dir}")
    cell_dirs = sorted(p for p in cells_dir.iterdir() if p.is_dir())
    incomplete = [p.name for p in cell_dirs
                  if not CellStore(sweep_dir, p.name).is_done()]
    if incomplete:
        if not skip_incomplete:
            raise StoreError(
                f"consolidate: {len(incomplete)} cell(s) under {cells_dir} "
                "have no results.json — the only cell-done marker (§2.3); "
                "rerun them (or pass skip_incomplete=True to exclude them "
                f"from an inspection build): {incomplete}")
        for name in incomplete:
            print(f"[capeval.store] consolidate: excluding incomplete cell "
                  f"{name} (no results.json)")
    tombstones = {str(cell): set(keys) for cell, keys in
                  (exclude_keys_by_cell or {}).items()}
    unknown_tombstone_cells = sorted(
        set(tombstones) - {p.name for p in cell_dirs})
    if unknown_tombstone_cells:
        raise StoreError(
            "consolidate: tombstones name unknown cell(s) "
            f"{unknown_tombstone_cells} — refusing to hide rows by typo")
    all_rows: list = []
    for cell in cell_dirs:
        if cell.name in set(incomplete):
            continue
        store = CellStore(sweep_dir, cell.name)
        excluded = tombstones.get(cell.name, set())
        for r in store._parsed_rows():
            extra = sorted(set(r) - set(ROW_COLUMNS))
            missing = sorted(set(ROW_COLUMNS) - set(r))
            if extra or missing:
                raise StoreError(
                    f"{store.rows_path}: row schema drift — extra {extra}, "
                    f"missing {missing} (frozen ROW_COLUMNS)")
            key = (str(r["task"]), str(r["item_id"]), str(r["metric"]))
            if key in excluded:
                continue
            all_rows.append(r)
    if not all_rows:
        raise StoreError(f"consolidate: no rows found under {cells_dir} — "
                         "nothing to consolidate (wrong sweep dir?)")

    fields = []
    for col in ROW_COLUMNS:
        if col in _STRING_COLS:
            typ = pa.string()
        elif col in _INT_COLS:
            typ = pa.int64()
        elif col in _FLOAT_COLS:
            typ = pa.float64()
        else:  # pragma: no cover - ROW_COLUMNS is frozen
            raise StoreError(f"consolidate: column {col!r} has no arrow type")
        fields.append(pa.field(col, typ))
    schema = pa.schema(fields)
    table = pa.Table.from_pylist(all_rows, schema=schema)
    out = sweep_dir / "rows.parquet"
    tmp = sweep_dir / "rows.parquet.tmp"
    pq.write_table(table, tmp)
    _fsync_replace(tmp, out)
    return out


def harness_config_hash(resolved_run_config: Mapping[str, Any]) -> str:
    """16-hex sha256 over canonical_json of the resolved run config
    (§2.1): rendered task configs, gen params, engine flags, lm-eval
    version, evalplus image digest, dataset revisions,
    chat_template_bytes(model) — the tokenizer's rendered template string
    for this revision."""
    if not isinstance(resolved_run_config, Mapping) or not resolved_run_config:
        raise StoreError("harness_config_hash takes a non-empty mapping "
                         "(the resolved run config)")
    return sha256_hex16(canonical_json(dict(resolved_run_config)).encode("utf-8"))


def eval_config_digest(task_versions: Mapping[str, str], *,
                       template_hash: Optional[str],
                       gen_params: Optional[Mapping[str, Any]]) -> str:
    """16-hex sha256 of the EVALUATION-CONFIG identity two legs (and a
    calibration match) must share: the per-task task_version map, the
    rendered chat-template hash, and the generation params.

    Deliberately NARROWER than harness_config_hash, which hashes the whole
    manifest document — sweep name, arms, margins — and therefore differs
    between a calibration manifest and a sweep manifest by construction.
    This digest is the instrument-only identity: it agrees across
    manifests running the same tasks/templates/generation config, and
    drifts exactly when prompt rendering, task versions, or generation
    params change under the same item IDs.

    ``template_hash``/``gen_params`` are REQUIRED keywords that may be
    None: an absent component is a stated input recorded as null in the
    hashed payload, never a silent omission. The runtime archives the
    digest in each cell's manifest.json via the resolved spec (the cell
    manifest archives the resolved spec verbatim)."""
    if not isinstance(task_versions, Mapping) or not task_versions:
        raise StoreError("eval_config_digest: task_versions must be a "
                         "non-empty {task: task_version} mapping")
    for t, v in task_versions.items():
        if not isinstance(t, str) or not t or not isinstance(v, str) or not v:
            raise StoreError(f"eval_config_digest: bad task_versions entry "
                             f"{t!r}: {v!r} (non-empty strings only)")
    payload = {
        "task_versions": dict(task_versions),
        "template_hash": template_hash,
        "gen_params": dict(gen_params) if gen_params is not None else None,
    }
    return sha256_hex16(canonical_json(payload).encode("utf-8"))


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


#: the SDPA flag state every production entry point establishes before it
#: records a digest (models.configure_fast_sdpa / fitmon.pin_attention_backends
#: on an otherwise-default torch): cuDNN SDPA off — it re-plans per input
#: shape on CPU and selects different kernels, so it changes both throughput
#: and numerics — with torch's other backend defaults untouched.
PINNED_SDPA_STATE = {
    "cudnn_sdp": False,
    "flash_sdp": True,
    "mem_efficient_sdp": True,
    "math_sdp": True,
}


def attention_backend_state() -> Optional[dict]:
    """The LIVE torch SDPA backend-availability flags, read at call time.

    Deliberately derived from ``torch.backends.cuda`` state rather than
    from whether ``configure_fast_sdpa()`` was called: a fingerprint that
    records the INTENDED state is worse than none (e.g. under
    ``SISTEER_KEEP_CUDNN_SDP=1`` the pin call is a no-op and cuDNN stays
    live — exactly the deviation this exists to expose). Returns None when
    torch (or this torch's flag API) is absent — then no SDPA kernel can
    run, and the digest's ``torch`` field already separates such envs."""
    try:
        from torch.backends import cuda as _tbc
        return {
            "cudnn_sdp": bool(_tbc.cudnn_sdp_enabled()),
            "flash_sdp": bool(_tbc.flash_sdp_enabled()),
            "mem_efficient_sdp": bool(_tbc.mem_efficient_sdp_enabled()),
            "math_sdp": bool(_tbc.math_sdp_enabled()),
        }
    except Exception:
        return None


def env_digest_fields() -> dict:
    """The raw field dict env_digest() hashes — best-effort fields
    present, absent ones recorded as null (never silently skipped).

    ONE deliberate exception to "always present": ``attention_backend``
    (the live SDPA flag state) folds into the payload ONLY when it
    deviates from :data:`PINNED_SDPA_STATE` — the parity_n pattern: the
    default stays out of the hash entirely, so every digest recorded at
    the project-pinned backend (the entire existing corpus; env_digest is
    provenance-only, never key-bearing) stays byte-identical, and the
    digest changes exactly when the backend genuinely differs."""
    import platform

    fields: dict = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "lm_eval", "vllm", "evalplus"):
        fields[pkg] = _pkg_version(pkg.replace("_", "-")) or _pkg_version(pkg)

    cuda_runtime = gpu_name = cuda_driver = None
    try:
        import torch
        cuda_runtime = torch.version.cuda
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        drv = Path("/proc/driver/nvidia/version")
        if drv.exists():
            cuda_driver = drv.read_text().splitlines()[0].strip()
    except Exception:
        pass
    fields["cuda_runtime"] = cuda_runtime
    fields["cuda_driver"] = cuda_driver
    fields["gpu_name"] = gpu_name

    backend = attention_backend_state()
    if backend is not None and backend != PINNED_SDPA_STATE:
        fields["attention_backend"] = backend
    return fields


def env_digest() -> str:
    """16-hex sha256 over {python, torch, transformers, lm_eval?, vllm?,
    evalplus?, cuda driver, GPU name} — best-effort fields present, absent
    ones recorded as null (never silently skipped) (§2.1) — plus the live
    SDPA backend flags when (and only when) they deviate from the pinned
    default (env_digest_fields docstring)."""
    return sha256_hex16(canonical_json(env_digest_fields()).encode("utf-8"))
