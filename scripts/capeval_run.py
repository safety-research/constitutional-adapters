"""capeval driver CLI — one process per invocation.

Subcommands (all take ``--manifest``):

  print-cells   emit the cell enumeration, its count, and ready-to-paste
                sbatch ``--array`` ranges (F15: array widths are NEVER
                handwritten — every submission uses this output).
  run           execute ONE cell (``--cell-index N`` for sbatch arrays, or
                ``--arm NAME [--trial K]``), with per-item resume (§2.3).
  consolidate   fold cells/*/rows.jsonl -> rows.parquet (store.consolidate).
  fingerprint   per-arm domain statuses + screen status (stats/fingerprint).
  power         simulation-based power table (stats/power).
  report        build the HTML/W&B/figures report (report/build.py).

CELL ENUMERATION — THE CANONICAL COPY
=====================================
``enumerate_cells()`` below is the one canonical enumeration; the sbatch
side reads the SAME ordering through the ``CAPEVAL_*`` constants exported
by ``scripts/sbatch/capeval_common.sh``, and
``tests/test_capeval_driver.py`` pins the two against each other (the
mirrored-constants test) plus the expected counts (controls: 15). A
manifest edit that changes any width fails that test rather than silently
truncating an array. The rules, by manifest ``mode``:

* ``calibration`` — exactly one ``kind: none`` ablation; one cell per
  trial ``0..k-1`` where ``k = ablation.params["trials"]`` (default 6,
  §7.1; grown in batches of 3, max 12). Trial 0 is the
  designated reference and sole KL-cache builder: it must run ALONE
  before anything else (§11.2 order — the F5 race eliminator).
* ``control`` — per ablation, in manifest order: if the ablation's
  ``params.tier1_coeffs`` is ABSENT, one cell per coeff at the manifest's
  full tier set; if PRESENT (the §8 ladder split), one Tier-0-only cell
  per coeff (cheap rung, time_class ``t0only``) followed by one
  Tier-1-only cell per coeff listed in ``tier1_coeffs`` (time_class
  ``full``). ``tier1_coeffs ⊄ coeffs`` is a refusal.
* ``sweep`` — one cell per (ablation, coeff), full tier set; the target
  plugin runs inside each cell when the manifest declares a target.
  Index 0 is the pilot cell by convention (§9: list the pilot arm first).
  A MULTI-document manifest file (the s2_rediscovery convention — one
  document per cell so each cell binds its own target, F20; loaded via
  ``load_manifests``) enumerates as the concatenation of its
  documents' cells, indices continuous in document order — document
  index == array index for one-arm-one-coeff documents.

INTEGRATION NOTE: the calibration
trial count rides in the ``kind: none`` ablation's ``params: {trials: 6}``
— the frozen ``Manifest`` schema has no top-level trial field, and
``AblationSpec.params`` is the one schema-blessed extension point.
``params.tier1_coeffs`` on control arms is the same kind of convention.

RE-EXEC MACHINERY
=================
Weight-mutating arms (``WEIGHT_MUTATING_KINDS``) never execute in the
driver's own process: the parent re-execs ``sys.executable <this file>
... --reexec-child`` so weight restoration can never leak into a
neighboring cell even on crash. The SAME machinery serves confirmation
reruns (§4.4/§7.3): when attempt 0 reports a confirmation trigger — the
in-process return value's ``confirmation_needed``, or the child's
distinguished exit code ``EXIT_CONFIRMATION_NEEDED`` — the parent re-execs
the cell as ``--attempt 1`` (fresh subprocess = fresh model load, fresh
seed via ``cell_seed()``, fresh item order, same job allocation). A
``--reexec-child`` process NEVER spawns further processes.

SEAM CONTRACT (the driver's fake-seam tests pin the
orchestration order, not these signatures):
  make_engine(manifest)                          -> serving Engine (loaded)
  resolve_cell(manifest, cell, attempt, out_root, seed)
      -> (store, resolved: Mapping)  # hashes resolved, cell manifest
         written, resume validated (§2.3). ``store`` carries the frozen
         CellStore surface (key/is_done/write_results/...) PLUS
         ``append_ts(event, epoch_s)``, which files the F7 first-class
         timestamp row (metric=ts_<event>, value=epoch seconds) with the
         resolved cell's row-constant fields.
  run_tier0(engine, manifest, cell, store, seed)  -> dict
  run_tier1(engine, manifest, cell, store, seed)  -> dict
  run_target(engine, manifest, cell, store, seed) -> dict
Each phase dict may carry ``confirmation_needed: [reasons...]``; the
driver unions them into results.json and triggers the F16 rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval import manifest as cm  # noqa: E402  (frozen skeleton)

# --- Frozen driver constants (mirrored in capeval_common.sh; test-pinned) ---

# Arms whose ctx mutates model WEIGHTS (restore-on-exit): always isolated
# one-arm-per-process. Hook-only arms are not listed.
WEIGHT_MUTATING_KINDS = ("control:noise_lora", "control:weight_noise")
DEFAULT_CALIB_TRIALS = 6
TIME_CLASS_FULL = "full"      # gets the script's full --time
TIME_CLASS_T0 = "t0only"      # Tier-0-only ladder rungs (0-02:00:00, §11.2)
# Distinguished exit code from a --reexec-child whose cell completed but
# filed a confirmation trigger (F16). Any other nonzero rc is a failure.
# The exit code alone is NEVER trusted: it is honored only when the cell
# dir carries the persisted CONFIRMATION_MARKER the child wrote (an
# unrelated child death with rc 42 must not masquerade as a trigger —
# review tier1 #5).
EXIT_CONFIRMATION_NEEDED = 42
# Atomic per-cell marker persisting attempt 0's confirmation triggers
# ({"probes": [reasons...]}). Written BEFORE results.json, so a
# preemption between them re-triggers attempt 1 on resume instead of
# silently skipping the confirmation (review tier1 #5). Never deleted:
# attempt 1's own results.json makes the re-trigger a no-op.
CONFIRMATION_MARKER = "confirmation_required.json"
# fingerprint.json / power.json wire format version (report/build.py
# checks it — simplify-pass #6). v2: the screen no longer consumes
# KL flags, and each arm entry carries the kl_coarse_lb annotation.
FINGERPRINT_SCHEMA_VERSION = 2
# Seed convention (ResultRow.seed files this):
# calibration trial t -> seed t; other cells -> 100 + cell index; a
# confirmation rerun (attempt 1) adds 10000 (fresh seed AND fresh order —
# item order must derive from the seed).
ATTEMPT_SEED_OFFSET = 10000


@dataclass(frozen=True)
class Cell:
    """One enumerated cell — the unit an sbatch array index decodes to."""

    index: int
    arm: str
    kind: str
    coeff: Optional[float]
    trial: int
    tiers: Tuple[int, ...]
    time_class: str


def cell_seed(cell: Cell, attempt: int) -> int:
    base = cell.trial if _is_calibration_kind(cell) else 100 + cell.index
    return base + (ATTEMPT_SEED_OFFSET if attempt else 0)


def _is_calibration_kind(cell: Cell) -> bool:
    return cell.kind == cm.NONE_KIND and cell.trial >= 0 and cell.coeff is None


def enumerate_cells(manifest: cm.Manifest) -> List[Cell]:
    """The canonical cell enumeration — see the module docstring for the
    per-mode rules. Deterministic in manifest order; refusals are loud."""
    cells: List[Cell] = []
    tiers = tuple(manifest.tiers)

    def add(arm, kind, coeff, trial, cell_tiers, time_class):
        cells.append(Cell(len(cells), arm, kind, coeff, trial, tuple(cell_tiers), time_class))

    if manifest.mode == "calibration":
        if len(manifest.ablations) != 1 or manifest.ablations[0].kind != cm.NONE_KIND:
            raise ValueError(
                "mode: calibration requires exactly one kind: none ablation; "
                f"got {[(a.name, a.kind) for a in manifest.ablations]}")
        ab = manifest.ablations[0]
        trials = int(ab.params.get("trials", DEFAULT_CALIB_TRIALS))
        if not 1 <= trials <= 12:
            raise ValueError(f"calibration trials must be 1..12 (D5/D10 cap), got {trials}")
        for t in range(trials):
            add(ab.name, ab.kind, None, t, tiers, TIME_CLASS_FULL)
    elif manifest.mode == "control":
        for ab in manifest.ablations:
            coeffs = list(ab.coeffs) or [None]
            t1c = ab.params.get("tier1_coeffs")
            if t1c is None:
                for c in coeffs:
                    add(ab.name, ab.kind, c, 0, tiers, TIME_CLASS_FULL)
            else:
                missing = [c for c in t1c if c not in coeffs]
                if missing:
                    raise ValueError(
                        f"arm {ab.name}: tier1_coeffs {missing} not in coeffs {coeffs}")
                for c in coeffs:  # Tier-0 at every rung (§8)
                    add(ab.name, ab.kind, c, 0, (0,), TIME_CLASS_T0)
                for c in coeffs:  # Tier-1 only at the named rungs
                    if c in t1c:
                        add(ab.name, ab.kind, c, 0, (1,), TIME_CLASS_FULL)
    elif manifest.mode == "sweep":
        for ab in manifest.ablations:
            for c in list(ab.coeffs) or [None]:
                add(ab.name, ab.kind, c, 0, tiers, TIME_CLASS_FULL)
    else:  # pragma: no cover - load_manifest refuses unknown modes first
        raise ValueError(f"unknown manifest mode {manifest.mode!r}")
    return cells


def load_all_manifests(path: Path) -> List[cm.Manifest]:
    """Load a manifest file as a list of documents (usually length 1;
    the multi-doc s2_rediscovery convention gives one per cell)."""
    return list(cm.load_manifests(path))


def enumerate_all(manifests: Sequence[cm.Manifest]) -> List[Tuple[Cell, cm.Manifest]]:
    """Concatenated enumeration across manifest documents, indices
    continuous in document order. Multi-document is a sweep-mode-only
    convention (each doc binds its own target, F20)."""
    import dataclasses

    modes = {m.mode for m in manifests}
    if len(modes) != 1:
        raise ValueError(f"manifest documents disagree on mode: {sorted(modes)}")
    if len(manifests) > 1 and manifests[0].mode != "sweep":
        raise ValueError("multi-document manifests are a sweep-mode convention "
                         f"(got mode {manifests[0].mode!r})")
    pairs: List[Tuple[Cell, cm.Manifest]] = []
    for m in manifests:
        for c in enumerate_cells(m):
            pairs.append((dataclasses.replace(c, index=len(pairs)), m))
    return pairs


def array_ranges(indices: Sequence[int]) -> str:
    """Compress sorted indices into an sbatch --array string ("2-5,8-11")."""
    if not indices:
        return ""
    out, run_start, prev = [], indices[0], indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append(f"{run_start}-{prev}" if prev > run_start else f"{run_start}")
        run_start = prev = i
    out.append(f"{run_start}-{prev}" if prev > run_start else f"{run_start}")
    return ",".join(out)


def format_print_cells(manifest: cm.Manifest, cells: List[Cell]) -> str:
    """The print-cells wire format (consumed by §11.2 submissions and by
    the mirrored-constants test). KEY=VALUE lines are the machine seam."""
    lines = [
        f"# capeval cells: sweep={manifest.sweep} mode={manifest.mode} engine={manifest.engine}",
        "INDEX\tARM\tKIND\tCOEFF\tTRIAL\tTIERS\tTIME_CLASS",
    ]
    for c in cells:
        coeff = "-" if c.coeff is None else repr(c.coeff)
        lines.append(
            f"{c.index}\t{c.arm}\t{c.kind}\t{coeff}\t{c.trial}\t"
            f"{','.join(map(str, c.tiers))}\t{c.time_class}")
    lines.append(f"COUNT={len(cells)}")
    lines.append(f"ARRAY_ALL={array_ranges([c.index for c in cells])}")
    full = [c.index for c in cells if c.time_class == TIME_CLASS_FULL]
    t0 = [c.index for c in cells if c.time_class == TIME_CLASS_T0]
    if t0:
        lines.append(f"ARRAY_FULL={array_ranges(full)}")
        lines.append(f"ARRAY_T0ONLY={array_ranges(t0)}")
    if manifest.mode == "calibration":
        lines.append("ARRAY_REFERENCE=0  # trial 0 runs ALONE first (F5)")
        lines.append(f"ARRAY_REST={array_ranges([c.index for c in cells if c.index > 0])}")
    if manifest.mode == "sweep":
        lines.append("ARRAY_PILOT=0  # §9: pilot before releasing the rest")
        lines.append(f"ARRAY_REST={array_ranges([c.index for c in cells if c.index > 0])}")
    return "\n".join(lines)


def select_cell(cells: List[Cell], *, index: Optional[int] = None,
                arm: Optional[str] = None, trial: Optional[int] = None,
                coeff: Optional[float] = None) -> Cell:
    if index is not None:
        if not 0 <= index < len(cells):
            raise ValueError(
                f"--cell-index {index} out of range: this manifest enumerates "
                f"{len(cells)} cells (0-{len(cells) - 1}); array width and "
                "enumeration must both come from print-cells (F15)")
        return cells[index]
    matches = [c for c in cells
               if c.arm == arm
               and (trial is None or c.trial == trial)
               and (coeff is None or c.coeff == coeff)]
    if len(matches) != 1:
        raise ValueError(
            f"--arm {arm!r} --trial {trial} --coeff {coeff} matches "
            f"{len(matches)} cells; disambiguate (print-cells lists all)")
    return matches[0]


# --- run: one cell per invocation --------------------------------------------


def _out_root(args: argparse.Namespace, manifest: cm.Manifest) -> Path:
    """--out-root when given, else the manifest's default sweep dir."""
    return Path(args.out_root) if args.out_root else _default_out_root(manifest)


def _default_out_root(manifest: cm.Manifest) -> Path:
    env = os.environ.get("CAPEVAL_OUT_ROOT")
    if env:
        return Path(env) / manifest.sweep
    # on a cluster the job environment names its shared root (SISTEER_SHARED_ROOT, from the cluster's site block)
    if os.environ.get("SISTEER_SHARED_ROOT"):
        project = os.environ.get("SISTEER_PROJECT_ROOT")
        if not project:
            raise SystemExit("set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)")
        return Path(project) / "out" / "capeval" / manifest.sweep
    raise SystemExit(
        "cannot resolve the output root: not on the cluster and neither "
        "--out-root nor CAPEVAL_OUT_ROOT is set (plan §2.3)")


def _load_seams(engine_name: str) -> Dict[str, Any]:
    """Resolve the phase runners (lazy — heavy deps must never
    import in the base lane). Loud, named error when a phase runner's
    module has not landed. Tests monkeypatch the returned dict."""
    try:
        from sisteer.capeval import runtime as rt  # integration module (WS merge)
    except ImportError as e:
        raise RuntimeError(
            "phase runners unavailable: sisteer.capeval.runtime has not been "
            "wired yet (integration step; WS-A store/corpora + WS-B "
            "tier0/hf_engine + WS-C tier1 land there). Structural driver "
            f"paths (print-cells, re-exec, decode) work without it. [{e}]"
        ) from e
    return {
        "make_engine": rt.make_engine,
        "resolve_cell": rt.resolve_cell,
        "run_tier0": rt.run_tier0,
        "run_tier1": rt.run_tier1,
        "run_target": rt.run_target,
    }


def _wandb_start(manifest: cm.Manifest, run_name: str):
    """W&B lifecycle: offline on cluster (stage2_common.sh
    exports WANDB_MODE=offline), online on the grading host report step. Set
    CAPEVAL_WANDB=disabled to skip (tests). Import failure is LOUD."""
    if os.environ.get("CAPEVAL_WANDB", "") == "disabled":
        return None
    import wandb  # base dep

    return wandb.init(project="capeval", name=run_name,
                      config={"sweep": manifest.sweep, "mode": manifest.mode,
                              "base": manifest.base, "engine": manifest.engine})


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """tmp + fsync + replace: existence must mean fully-published (the
    store's own marker discipline, applied to the driver's files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=1, default=str))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _attempt0_marker_path(store: Any) -> Path:
    """Attempt 0's confirmation marker for this cell: the cell key differs
    across attempts ONLY in its ``__a<attempt>__`` segment (cell_key
    format, store §2.3), so the a0 sibling dir is derived exactly."""
    arm, tier_tok, trial_tok, _attempt_tok, ih8, ch8 = store.key.split("__")
    a0_key = "__".join((arm, tier_tok, trial_tok, "a0", ih8, ch8))
    return Path(store.sweep_dir) / "cells" / a0_key / CONFIRMATION_MARKER


def _probe_names(entries: Sequence[Any]) -> List[str]:
    """Normalize confirmation_needed entries to probe NAMES: tier0 files
    dicts ({"probe": "canary"/"dist_shift", ...} — its CONFIRMABLE_PROBES
    vocabulary); plain strings pass through. The names are what attempt 1
    hands back as ``confirm_probes`` (frozen contract)."""
    names = []
    for e in entries:
        names.append(e["probe"] if isinstance(e, Mapping) and "probe" in e
                     else str(e))
    return sorted(set(names))


def _read_confirm_probes(store: Any) -> List[str]:
    marker = _attempt0_marker_path(store)
    if not marker.is_file():
        return []
    return list(json.loads(marker.read_text()).get("probes", []))


def _confirmation_marker_paths(out_root: Path, cell: Cell) -> List[Path]:
    """Attempt-0 confirmation markers matching this cell, found WITHOUT
    resolving hashes (the parent of a weight-mutating cell never loads
    the seams): arm/trial/attempt are literal cell-key segments and arm
    names cannot contain '__' (store.cell_key refuses), so the glob is
    exact on those fields."""
    pattern = f"{cell.arm}__*__t{cell.trial}__a0__*"
    return sorted((out_root / "cells").glob(f"{pattern}/{CONFIRMATION_MARKER}"))


def _execute_cell(manifest: cm.Manifest, cell: Cell, attempt: int,
                  out_root: Path, seams: Mapping[str, Any]) -> Dict[str, Any]:
    """Run one cell in THIS process. Ordering discipline (F7/F8, §2.3):
    ts_job_start -> engine load -> ts_model_ready -> per-tier
    ts_tier<k>_start/end -> target -> ts_cell_end -> confirmation marker
    (when triggered) -> results.json LAST (or results_pending.json +
    generation_complete.json for EvalPlus cells).
    Returns the results summary (incl. any confirmation_needed)."""
    # Pin the SDPA backend BEFORE resolve_cell records env_digest() into the
    # row constants and cell manifest: the engine load below makes the first
    # configure_fast_sdpa call otherwise, so a digest recorded here would
    # fingerprint the torch-default backend state while every scored token
    # ran on the pinned one (cell_controls_run.cmd_generate's discipline).
    # Idempotent; SISTEER_KEEP_CUDNN_SDP=1 still wins — and then the digest
    # records that deviation, which is the point.
    from sisteer.models import configure_fast_sdpa
    configure_fast_sdpa()
    seed = cell_seed(cell, attempt)
    store, resolved = seams["resolve_cell"](manifest, cell, attempt, out_root, seed)
    if store.is_done():
        # Confirmation persistence (review tier1 #5): a done attempt-0
        # cell whose marker exists still owes the attempt-1 rerun — a
        # preemption between results.json and the parent's re-exec must
        # re-trigger on resume, never silently skip the confirmation.
        marker = Path(store.cell_dir) / CONFIRMATION_MARKER
        if attempt == 0 and marker.is_file():
            probes = list(json.loads(marker.read_text()).get("probes", []))
            print(f"[capeval] cell done but {CONFIRMATION_MARKER} present — "
                  f"re-triggering the attempt-1 confirmation ({probes})")
            return {"resumed_done": True, "confirmation_needed": probes}
        print(f"[capeval] cell already done (results.json present): {store.key}")
        return {"resumed_done": True, "confirmation_needed": []}

    wb = _wandb_start(manifest, run_name=store.key)
    ts = lambda event: store.append_ts(event, time.time())  # noqa: E731
    results: Dict[str, Any] = {"cell": cell.__dict__, "attempt": attempt,
                               "seed": seed, "phases": {},
                               "confirmation_needed": []}
    try:
        ts("job_start")
        engine = seams["make_engine"](manifest)
        ts("model_ready")
        for tier in cell.tiers:
            ts(f"tier{tier}_start")
            if tier == 0:
                # attempt 1 re-checks exactly the persisted attempt-0
                # probes (frozen kwarg name — orchestrator decision).
                kwargs = ({"confirm_probes": frozenset(_read_confirm_probes(store))}
                          if attempt == 1 else {})
                phase = seams["run_tier0"](engine, manifest, cell, store, seed,
                                           **kwargs) or {}
            else:
                phase = seams["run_tier1"](engine, manifest, cell, store, seed) or {}
            results["phases"][f"tier{tier}"] = {k: v for k, v in phase.items()
                                                if k != "confirmation_needed"}
            results["confirmation_needed"] += list(phase.get("confirmation_needed", []))
            ts(f"tier{tier}_end")
        if manifest.mode == "sweep" and manifest.target is not None:
            ts("target_start")
            phase = seams["run_target"](engine, manifest, cell, store, seed) or {}
            results["phases"]["target"] = {k: v for k, v in phase.items()
                                           if k != "confirmation_needed"}
            results["confirmation_needed"] += list(phase.get("confirmation_needed", []))
            ts("target_end")
        ts("cell_end")
        if results["confirmation_needed"] and attempt == 0:
            # persisted BEFORE any done/pending marker (review tier1 #5):
            # the trigger survives preemption and gates the exit-42 path.
            # "probes" carries the normalized NAMES attempt 1 feeds back
            # as confirm_probes; "entries" archives the full records.
            _write_json_atomic(Path(store.cell_dir) / CONFIRMATION_MARKER,
                               {"probes": _probe_names(results["confirmation_needed"]),
                                "entries": results["confirmation_needed"],
                                "written_at": time.time()})
        awaiting = any(p.get("awaiting_execution") for p in results["phases"].values()
                       if isinstance(p, Mapping))
        if awaiting:
            # EvalPlus generation cell (§5.5/F8): the summary parks in
            # results_pending.json (NOT a done marker), committed
            # ATOMICALLY, and only THEN generation_complete.json — so the
            # execution wave can never observe the marker while the local
            # phases/pending summary are missing (review tier1 #3). The
            # wave requires BOTH files; results.json stays the wave's.
            pending = Path(store.cell_dir) / "results_pending.json"
            _write_json_atomic(pending, results)
            store.write_generation_complete()
            print(f"[capeval] awaiting sandbox execution — wrote {pending} "
                  "then generation_complete.json; results.json is the exec "
                  "wave's to write (F8)")
        else:
            store.write_results(results)  # LAST — the only cell-done marker
    finally:
        if wb is not None:
            wb.finish()
    return results


def _reexec_argv(args: argparse.Namespace, attempt: int) -> List[str]:
    """argv for the fresh-subprocess re-exec (F16): same manifest + cell,
    explicit --attempt, --reexec-child so the child never recurses."""
    argv = [sys.executable, str(Path(__file__).resolve()),
            "--manifest", str(args.manifest), "run",
            "--attempt", str(attempt), "--reexec-child"]
    if args.cell_index is not None:
        argv += ["--cell-index", str(args.cell_index)]
    else:
        argv += ["--arm", args.arm]
        if args.trial is not None:
            argv += ["--trial", str(args.trial)]
        if args.coeff is not None:
            argv += ["--coeff", repr(args.coeff)]
    if args.out_root:
        argv += ["--out-root", str(args.out_root)]
    return argv


def _spawn(argv: List[str]) -> int:
    print(f"[capeval] re-exec (fresh subprocess): {' '.join(argv)}", flush=True)
    return subprocess.run(argv).returncode


def cmd_run(args: argparse.Namespace, manifests: Sequence[cm.Manifest]) -> int:
    pairs = enumerate_all(manifests)
    cell = select_cell([c for c, _ in pairs], index=args.cell_index,
                       arm=args.arm, trial=args.trial, coeff=args.coeff)
    manifest = pairs[cell.index][1]  # the cell's OWNING document (its target)
    out_root = _out_root(args, manifest)
    print(f"[capeval] cell {cell.index}: arm={cell.arm} kind={cell.kind} "
          f"coeff={cell.coeff} trial={cell.trial} tiers={cell.tiers} "
          f"attempt={args.attempt}")

    if args.reexec_child:
        # Child: run in-process, NEVER spawn. A confirmation trigger is
        # reported to the parent via the distinguished exit code.
        res = _execute_cell(manifest, cell, args.attempt, out_root,
                            _load_seams(manifest.engine))
        return EXIT_CONFIRMATION_NEEDED if res.get("confirmation_needed") else 0

    needs_isolation = cell.kind in WEIGHT_MUTATING_KINDS
    if needs_isolation:
        rc = _spawn(_reexec_argv(args, attempt=args.attempt))
        confirmation = rc == EXIT_CONFIRMATION_NEEDED
        if confirmation and not _confirmation_marker_paths(out_root, cell):
            # exit 42 without the persisted marker = an unrelated child
            # death that happens to exit 42, NOT a confirmation trigger —
            # honoring it would mask a failed attempt 0 (review tier1 #5).
            print(f"[capeval] child exited {EXIT_CONFIRMATION_NEEDED} but no "
                  f"{CONFIRMATION_MARKER} exists in the cell dir — treating "
                  "as a FAILURE, not a confirmation trigger")
            return rc
        if rc not in (0, EXIT_CONFIRMATION_NEEDED):
            return rc
    else:
        res = _execute_cell(manifest, cell, args.attempt, out_root,
                            _load_seams(manifest.engine))
        confirmation = bool(res.get("confirmation_needed"))

    if confirmation and args.attempt == 0 and not args.no_confirm:
        print("[capeval] confirmation trigger filed on attempt 0 -> automatic "
              "fresh-subprocess rerun as attempt 1 (F16: fresh model load, "
              "fresh seed, fresh item order, same allocation)")
        rc1 = _spawn(_reexec_argv(args, attempt=1))
        # attempt 1's own trigger just means the flag is CONFIRMED — that is
        # a recorded result, not a failure; both attempts' rows persist.
        return 0 if rc1 in (0, EXIT_CONFIRMATION_NEEDED) else rc1
    return 0


# --- thin CPU-step wrappers ---------------------------------------------------


def cmd_consolidate(args, manifests) -> int:
    from sisteer.capeval import store as cs

    path = cs.consolidate(_out_root(args, manifests[0]))
    print(f"[capeval] consolidated -> {path}")
    return 0


def _calibration_inputs(args, manifest):
    """(reference trial-0 DataFrame, Registry, Calibration, calib_root) —
    the shared assembly for fingerprint/power/controls-accept: consolidate
    the calibration sweep, select the trial-0 reference leg, load + check
    calibration.json against this manifest's identity."""
    import pandas as pd

    from sisteer.capeval import store as cs
    from sisteer.capeval.domains import Registry
    from sisteer.capeval.stats import calibrate as cal

    out_root = _out_root(args, manifest)
    calib_root = (Path(args.calibration_root) if args.calibration_root
                  else out_root.parent / "calibration")
    ref = pd.read_parquet(cs.consolidate(calib_root))
    ref = ref[(ref["trial_id"] == 0) & (ref["attempt"] == 0)
              & (ref["kind"] == "none") & (ref["engine"] == manifest.engine)]
    if ref.empty:
        raise SystemExit(f"no calibration trial-0 reference rows under "
                         f"{calib_root} — run the calibration sweep first "
                         "(§11.2 order)")
    registry = Registry.load(REPO / "configs" / "capeval" / "domains.yaml")
    calib = cal.load_calibration(calib_root / "calibration.json")
    cal.check_calibration(calib, model_revision=manifest.base_revision,
                          engine=manifest.engine,
                          harness_version=cal.harness_version(registry))
    return ref, registry, calib, calib_root


def _arm_exclusions(doc) -> List[str]:
    return list(doc.target.off_target_exclusions) if doc.target else []


def cmd_fingerprint(args, manifests) -> int:
    """Write <out-root>/fingerprint.json: per-arm §7.2/§7.3 domain
    statuses + screen, assembled from the consolidated sweep rows, the
    calibration trial-0 reference leg, and the calibration null, through
    stats.fingerprint.build_fingerprint (the real per-arm API — the
    previous call shape did not exist; simplify-pass #1).

    Wire format (schema_version 1; report/build.py checks it):
      {"schema_version": 1, "sweep": ..., "arms": {arm: build_fingerprint
       output — domains carry the writer's upper_bound/lower_bound}}"""
    import pandas as pd

    from sisteer.capeval import store as cs
    from sisteer.capeval.stats import fingerprint as fp

    manifest = manifests[0]
    out_root = _out_root(args, manifest)
    ref, registry, calib, _ = _calibration_inputs(args, manifest)
    df = pd.read_parquet(cs.consolidate(out_root))

    pairs = enumerate_all(manifests)
    docs_by_arm: Dict[str, Any] = {}
    for cell, doc in pairs:
        docs_by_arm.setdefault(cell.arm, doc)
    arms_out: Dict[str, Any] = {}
    for arm, doc in docs_by_arm.items():
        arm_df = df[(df["arm_name"] == arm) & (df["attempt"] == 0)]
        if arm_df.empty:
            print(f"[capeval] fingerprint: no rows for arm {arm!r} — skipped "
                  "(report reconciles expected vs filed arms)")
            continue
        # a Tier-0 flag is CONFIRMED iff tier0 filed flag_canary_confirmed
        # — the row its state machine emits only when the REQUESTED F16
        # rerun reproduced the canary failure, the same read
        # controls.observations_from_store uses. The canary is the ONLY
        # screen-feeding Tier-0 flag; KL-derived rows (the older stored
        # flag_kl_gt_kappa included) are annotations and never reach the
        # screen. This also retires the old both-attempts (task, metric)
        # intersection, which no canary confirmation could ever satisfy
        # (its flag row exists only at attempt 1) — it only ever matched
        # on the now-demoted KL flags.
        arm_flags = df[df["arm_name"] == arm]
        confirmed = bool(
            (arm_flags["metric"] == "flag_canary_confirmed").any())
        target = None
        if doc.target is not None:
            legs = []
            for primary in doc.target.primaries:
                task = f"target/{primary.plugin}"
                sub = arm_df[arm_df["task"] == task]
                a = dict(zip(sub[sub["metric"] == f"{primary.metric}/arm"]["item_id"],
                             sub[sub["metric"] == f"{primary.metric}/arm"]["value"]))
                b = dict(zip(sub[sub["metric"] == f"{primary.metric}/base"]["item_id"],
                             sub[sub["metric"] == f"{primary.metric}/base"]["value"]))
                if a and b:
                    # item-keyed mappings — target_status pairs by id and
                    # refuses set mismatches itself (stats r1 #7)
                    legs.append((primary, b, a))
            if legs:
                target = fp.target_status(legs, b=args.b)
        arms_out[arm] = fp.build_fingerprint(
            arm_df, ref, registry, calib, model=manifest.base, arm_name=arm,
            margins=dict(doc.margins), exclusions=_arm_exclusions(doc),
            tier0_flag_confirmed=confirmed, target=target, b=args.b)

    payload = {"schema_version": FINGERPRINT_SCHEMA_VERSION,
               "sweep": manifest.sweep, "arms": arms_out}
    path = out_root / "fingerprint.json"
    _write_json_atomic(path, payload)
    print(f"[capeval] fingerprint -> {path} ({len(arms_out)} arm(s))")
    return 0


def cmd_power(args, manifests) -> int:
    """Write <out-root>/power.json from stats.power.power_table (the real
    API — the previous build_power_table call did not exist;
    simplify-pass #1). One table per distinct exclusion set across the
    manifest documents, so every cell's own gating family size is
    represented."""
    import dataclasses as _dc

    from sisteer.capeval.stats import power as pw

    manifest = manifests[0]
    out_root = _out_root(args, manifest)
    _, registry, calib, calib_root = _calibration_inputs(args, manifest)

    tables = []
    seen_exclusions = []
    for doc in manifests:
        excl = sorted(_arm_exclusions(doc))
        if excl in seen_exclusions:
            continue
        seen_exclusions.append(excl)
        table = pw.power_table(
            calib, registry, model=manifest.base, exclusions=excl,
            nll_discretization_scale=args.nll_discretization_scale)
        tables.append({
            "exclusions": excl,
            "entries": [_dc.asdict(r) for (_, _dom), r in sorted(table.items())],
        })
    payload = {"schema_version": FINGERPRINT_SCHEMA_VERSION,
               "sweep": manifest.sweep, "model": manifest.base,
               "calibration": str(calib_root / "calibration.json"),
               "nll_discretization_scale": args.nll_discretization_scale,
               "tables": tables}
    path = out_root / "power.json"
    _write_json_atomic(path, payload)
    n = sum(len(t["entries"]) for t in tables)
    print(f"[capeval] power table -> {path} ({n} domain entries, "
          f"{len(tables)} exclusion set(s))")
    return 0


def cmd_report(args, manifests) -> int:
    from sisteer.capeval.report.build import build as report_build

    paths = report_build(
        sweep_dir=_out_root(args, manifests[0]), sweep_name=manifests[0].sweep,
        figs_dir=Path(args.figs_dir) if args.figs_dir else None,
        wandb_mode=args.wandb_mode)
    print(f"[capeval] report -> {paths.html}")
    return 0


def cmd_controls_accept(args, manifests) -> int:
    """Assemble ControlObservations from a mode:control sweep's store and
    write controls_acceptance.json in the frozen shape.

    Consumes: the control sweep's cells (consolidated here), the
    calibration sweep's trial-0 reference rows (same consolidation), and
    calibration.json (null bands + identity gate). Writes
    <out-root>/controls_acceptance.json ALWAYS (a failing acceptance is a
    recorded result), then exits 0 on pass, 2 on fail — a NO-OP failure
    (hard_stop) indicts the harness and must stop the sweep loudly."""
    import pandas as pd

    from sisteer.capeval import store as cs
    from sisteer.capeval.domains import Registry
    from sisteer.capeval.stats import calibrate as cal
    from sisteer.capeval.stats import controls as cc

    manifest = manifests[0]
    if manifest.mode != "control":
        raise SystemExit(f"controls-accept needs a mode:control manifest, "
                         f"got mode {manifest.mode!r}")
    out_root = _out_root(args, manifest)
    calib_root = (Path(args.calibration_root) if args.calibration_root
                  else out_root.parent / "calibration")

    df = pd.read_parquet(cs.consolidate(out_root))
    ref = pd.read_parquet(cs.consolidate(calib_root))
    ref = ref[(ref["trial_id"] == 0) & (ref["attempt"] == 0)
              & (ref["kind"] == "none") & (ref["engine"] == manifest.engine)]
    if ref.empty:
        raise SystemExit(f"no calibration trial-0 reference rows under "
                         f"{calib_root} — run the calibration sweep first "
                         "(§11.2 order)")
    registry = Registry.load(REPO / "configs" / "capeval" / "domains.yaml")
    calib = cal.load_calibration(calib_root / "calibration.json")
    cal.check_calibration(calib, model_revision=manifest.base_revision,
                          engine=manifest.engine,
                          harness_version=cal.harness_version(registry))

    # Tier-0 evidence domains: named EXPLICITLY from the registry —
    # the domains whose every task is a tier0/ probe.
    tier0_domains = [d.name for d in registry.domains
                     if d.kind in ("accuracy", "nll") and d.tasks
                     and all(t.startswith("tier0/") for t in d.tasks)]
    # Top rungs pinned from the MANIFEST, never re-derived from whatever
    # cells happen to exist.
    top_rungs = {ab.kind: max(ab.coeffs) for ab in manifest.ablations
                 if ab.kind in cc.DAMAGE_KINDS and ab.coeffs}
    obs = cc.observations_from_store(df, ref, registry, model=manifest.base,
                                     tier0_domains=tier0_domains, b=args.b)
    report = cc.check_controls(obs, cc.null_bands_from_calibration(calib),
                               top_rungs)
    payload = cc.controls_acceptance_payload(obs, report)
    path = out_root / "controls_acceptance.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"[capeval] controls acceptance -> {path}")
    for name, entry in payload["arms"].items():
        print(f"[capeval]   {name}: {entry['acceptance']} ({entry['role']})")
    if payload["hard_stop"]:
        print("[capeval] HARD STOP: a no-op control failed — the harness "
              "indicts itself; do NOT proceed to the S2 sweep (§8)")
    return 0 if payload["passed"] else 2


def cmd_print_cells(args, manifests) -> int:
    pairs = enumerate_all(manifests)
    print(format_print_cells(manifests[0], [c for c, _ in pairs]))
    return 0


# --- CLI ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--manifest", required=True, type=Path)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="execute one cell")
    run.add_argument("--cell-index", type=int, default=None)
    run.add_argument("--arm", default=None)
    run.add_argument("--trial", type=int, default=None)
    run.add_argument("--coeff", type=float, default=None)
    run.add_argument("--attempt", type=int, default=0, choices=(0, 1))
    run.add_argument("--out-root", default=None)
    run.add_argument("--no-confirm", action="store_true",
                     help="suppress the automatic F16 confirmation rerun")
    run.add_argument("--reexec-child", action="store_true",
                     help=argparse.SUPPRESS)  # internal (F16 re-exec)

    for name in ("consolidate", "fingerprint", "power", "print-cells"):
        p = sub.add_parser(name)
        p.add_argument("--out-root", default=None)
        if name in ("fingerprint", "power"):
            p.add_argument("--calibration-root", default=None,
                           help="calibration sweep dir "
                                "(default: <out-root>/../calibration)")
            p.add_argument("--b", type=int, default=10_000,
                           help="bootstrap draws for the §7.2 bounds")
        if name == "power":
            p.add_argument("--nll-discretization-scale", type=float,
                           default=None,
                           help="±deviation (nats/window) simulated at the "
                                "rule-of-three rate when an NLL domain's "
                                "null spread is exactly zero (deterministic "
                                "replay); required then, ignored otherwise "
                                "(stats/power.simulate_power_nll)")
    ca = sub.add_parser("controls-accept",
                        help="write controls_acceptance.json (spec acceptance #1)")
    ca.add_argument("--out-root", default=None)
    ca.add_argument("--calibration-root", default=None,
                    help="calibration sweep dir (default: <out-root>/../calibration)")
    ca.add_argument("--b", type=int, default=1000,
                    help="bootstrap draws for the point-estimate machinery")
    rep = sub.add_parser("report")
    rep.add_argument("--out-root", default=None)
    rep.add_argument("--figs-dir", default=None)
    rep.add_argument("--wandb-mode", default="online",
                     choices=("online", "offline", "disabled"),
                     help="report W&B run mode (plan §11.2: online from the "
                          "grading host — the standing offline+sync convention is "
                          "for cluster nodes, which never run `report`)")
    return ap


COMMANDS = {
    "run": cmd_run,
    "controls-accept": cmd_controls_accept,
    "consolidate": cmd_consolidate,
    "fingerprint": cmd_fingerprint,
    "power": cmd_power,
    "report": cmd_report,
    "print-cells": cmd_print_cells,
}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifests = load_all_manifests(args.manifest)
    if args.cmd == "run" and args.cell_index is None and args.arm is None:
        raise SystemExit("run needs --cell-index or --arm")
    return COMMANDS[args.cmd](args, manifests)


if __name__ == "__main__":
    sys.exit(main())
