"""Petri Bloom driver CLI — one process per
invocation, mirroring ``scripts/capeval_run.py``'s subcommand shape.

Subcommands (all take ``--manifest``, the Petri manifest YAML —
``configs/capeval/manifests/petri_twosi.yaml``):

  print-cells   enumerate the served legs — the 22 canonical cells, the
                DERIVED path-matched no-op control legs (~24 total), and
                the ABSENT report — plus a ready-to-paste sbatch --array
                range. With --out-root, writes cells.json (the
                enumeration record `analyze` requires).
  run           execute ONE cell (--cell-index N for sbatch arrays, or
                --cell-id ID) by dispatching to runtime.run_petri (the
                Phase-1 frozen seam the runner agent implements).
  smoke         the launch-blocking ONE-scenario real agent smoke — a
                LAUNCH-phase action (live proxy + tunnel +
                coordination post); refuses loudly until then.
  consolidate   collect failure ledgers (petri_incomplete.json) into the
                incomplete map, then fold the COMPLETE cells' rows into
                rows.parquet (store.consolidate).
  analyze       petri_stats.analyze over rows.parquet + the enumeration
                record + the incomplete map -> petri_analysis.json.
  report        petri_report.build -> tables/figures/report.md, W&B
                (project `petri`), and the figures mirror.

DELIBERATE SEAMS (loud refusals until their phase):
- the DefenseCell registry is WIRED: `load_cell_source` builds
  `petri.registry_adapter.traintest_cell_source` over configs/two_si.yaml
  (lazy import — that stack pulls torch). `CELL_SOURCE_FACTORY` remains
  the injection override (tests supply a synthetic source); a missing or
  inconsistent two-SI config still refuses loudly, never defaults.
- behavior definition files (configs/capeval/petri/<behavior>/BEHAVIOR.md)
  are the scenario workstream's deliverable; missing files refuse loudly.
- heavy deps (inspect_ai / petri-bloom / wandb / matplotlib) are imported
  only inside the subcommands that need them — this module imports and
  parses on the bare capeval CPU venv.

INCOMPLETE cells and consolidation: a cell with
unresolved required dimensions has no results.json and NEVER
consolidates. Unlike the capability pipeline (where skipping incomplete
cells is "never the report path"), the Petri report path DELIBERATELY
consolidates complete cells while carrying the incomplete ones in an
explicit map (petri_incomplete_map.json) — petri_stats then refuses to
publish any paired estimate involving them and reports them INCOMPLETE.

THE SERVING PATH (top-level options,
before the subcommand). ``--serving-path lens`` (the default) is today's
enumeration byte for byte: the manifest's cells over the lens-policy
registry, ``none__lens0`` (+ ``none__lora0`` iff a lora route is live)
derived, the manifest's ``serving_launch_flags`` hashed as written.
``--serving-path lora_native`` enumerates the EFFECTIVE configuration the
native multi-LoRA standup actually served: the cell list is
PARTITIONED over the lens-policy registry (``registry_adapter.
partition_cells_for_native`` — a stacked cell has no native form and is
SKIPPED BY NAME, as is a position-masked LoRA fit, logged and recorded,
never silently dropped; PSR / vector cells serve on the residual slot;
a LoRA cell the standup could not serve honestly REFUSES), the manifest's
``--enforce-eager`` / ``--enable-lora`` are stripped exactly as the standup
strips them (``native_standup.effective_launch_flags`` — the ONE
derivation, so ``harness_config_hash`` sees what the standup hashed), and
``enumerate_cells(lens_controls=False, lora_controls=True)`` derives
``none__lora0`` and never ``none__lens0`` (a zero-adapter-only server has
the control leg without a genuine lora route). Under native,
``--verify-standup <cells_served.json>`` and ``--verify-routing-table
<routing_table.json>`` are REQUIRED for ``print-cells`` and ``run``: the
standup's records are the AUTHORITATIVE source of what is served, and the
client's enumeration must agree with them cell by cell — vmodel, serving
identity, intervention hash, harness hash, run key, the served set and
every route — or the driver refuses BEFORE any leg (``standup_verification_
problem``). Under lens the two flags are optional and verified the same
way when given. Every row a served leg writes inherits ``serving_engine``
(``lens`` | ``lora_native``) through its cell manifest, and a run dir whose
manifest names the other engine refuses before the first sample
(``petri_runner.refuse_cross_engine_resume``). Nothing falls back from one
engine to the other.

THE TARGET DECODE CONTRACT (``run`` /
``print-cells`` options). The MANIFEST declares it -- ``target_decode: {mode:
vendor, policy: <exact registered ref@v>}`` -- because it is a harness-hash
input (a different instrument re-keys; an absent block is today's harness
audit byte for byte) and the standup hashes the SAME manifest into
``cells_served.json``, which ``--verify-standup`` holds the client to cell by
cell: a client-only knob could never agree with the standup's record. So
``--target-decode {harness,vendor}`` is an INTENT ASSERTION -- it must equal
the manifest's effective mode or the driver refuses before any leg -- and
``--target-seed-base <s>`` is the RUN knob (the literal training seed of the
object served; required under vendor, refused under harness), which rides the
spec into the runner (``PetriTargetDecode.seed_base``: never hashed, stamped
on the cell manifest / rows / eval-log metadata, and the Inspect log dir is
namespaced ``target-seed-<s>`` under the run key). Three seed-runs of one
manifest (the Arditi proxy) therefore need three ``--out-root``s; the runner
refuses a cell dir drawn under another contract or seed base.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import contextlib
import json
import threading
import signal
import fcntl
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval import store as cs  # noqa: E402
from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
from sisteer.capeval.petri import petri_stats  # noqa: E402
# the engine vocabulary (torch-free; imported, never redefined): the
# row-facing field and its two values
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402
# the routing table's loader (``load_routing_table`` / ``RouteEntry``) and the
# ONE live served-engine compare (``served_engine_problem``) — both CPU-light
from sisteer.capeval.serving import proxy as sproxy  # noqa: E402
from sisteer.capeval.serving import proxy_engine  # noqa: E402

PETRI_CONFIG_DIR = REPO / "configs" / "capeval" / "petri"
#: ``--serving-path`` vocabulary: the
#: lens escape hatch (today's enumeration byte for byte) and the native
#: multi-LoRA engine. The DRIVER's default stays ``lens`` — the wave drive
#: passes the engine's policy explicitly (``petri_drive_full22.sh``), so a
#: bare invocation can never re-key a lens audit natively by accident.
SERVING_PATH_LENS = "lens"
SERVING_PATH_LORA_NATIVE = "lora_native"
SERVING_PATHS = (SERVING_PATH_LENS, SERVING_PATH_LORA_NATIVE)
#: the row-facing engine each serving path stamps (``serving_engine``)
SERVING_ENGINE_BY_PATH = {SERVING_PATH_LENS: tc.SERVING_ENGINE_LENS,
                          SERVING_PATH_LORA_NATIVE: tc.SERVING_ENGINE_LORA_NATIVE}
#: the five per-cell fields the standup's ``cells_served.json`` entries and
#: the client's enumeration must agree on, cell by cell
STANDUP_CELL_FIELDS = ("virtual_model_name", "serving_identity",
                       "intervention_hash", "harness_config_hash",
                       "petri_run_key")
#: the proxy's standup declares the cells it was ASKED to serve for this audit in
#: cells_served.json -- `cells_requested_explicit_harness` (harness ids, exactly what petri_proxy's positional 5 / mt_serving_up --cell
#: named) and `cells_requested_explicit` (the same in roster spelling); both [] when nothing was named. That declaration is the ONE source
#: of the audit's legs: the frozen cells.json carries exactly those canonical legs, and the controls twin only when `none` is among them
#: (a steered-only view drives no none and no none__lora0 -- u0's legs are the controls). The daemon never widens it
#: (F22_PETRI_CELLS stays denied); the manifest's `cells` is the universe the declared set must be a subset of. The standup's own
#: `cells_requested` / `harness_cells` describe what is SERVED (none implicit first) and are not read for this.
REQUESTED_HARNESS_KEY = "cells_requested_explicit_harness"
REQUESTED_ROSTER_KEY = "cells_requested_explicit"
#: a declaration is BOUND to the live attempt -- the served record and the endpoint record must carry the same nonempty
#: `standup_attempt_id` (native_standup.ENDPOINT_ATTEMPT_KEY, pinned equal by test) before anything is frozen or run from it (an earlier
#: attempt's cells_served.json over identical routes and hashes can carry another declaration). The endpoint record is --served-endpoint
#: when given, else `endpoint.json` beside the served record (the writer's layout, and the layout both drives pull into); absent, refuse.
#: The two records' DECLARATIONS must agree first -- a declared served record needs the endpoint's declaration
#: present, well-formed and equal; an endpoint that declares over an undeclared served record refuses -- the reader never depends on
#: the producer being the only writer (a backfilled or partially copied record, a pre-declaration endpoint snapshot).
STANDUP_ATTEMPT_KEY = "standup_attempt_id"
ENDPOINT_JSON = "endpoint.json"
_STEERED_AUDIT_RE = re.compile(r"_s[0-9]+rl?$")   # S2RLFIX5-PETRI: `_s<N>r` AND `_s<N>rl` (the LeCun-init ReLU refit view), as every other copy of the grammar
DEFAULT_MANIFEST = (REPO / "configs" / "capeval" / "manifests"
                    / "petri_twosi.yaml")
CELLS_JSON = "cells.json"
#: the pending_reason v_pending_judgment gives a run (on its own row) whose judgment or in-progress claim for the required judge was bound to a
#: run identity since CORRECTED -- the judgment's row stands but counts for nothing; a stale-bound claim reads this, never `claimed`. ONE
#: spelling here, mirrored in scripts/registry/grade_daemon.py, so a change of that spelling is a one-token edit. Not produced by the view today.
IDENTITY_STALE_REASON = "identity_stale"
#: the enumeration record's additive target-decode key (``_cells_payload``)
TARGET_DECODE_RECORD_KEY = "target_decode"
INCOMPLETE_MAP = "petri_incomplete_map.json"
ANALYSIS_JSON = "petri_analysis.json"
FAILURE_LEDGER = "petri_incomplete.json"  # per-cell sidecar
#: exit codes: 0 ok; 2 refusal; 3 = the cell ran but is INCOMPLETE (H9).
EXIT_INCOMPLETE = 3
#: the cell-level footing keys a results.json written by the runner carries
#: (petri_runner.audited_footing, folded into the run summary); all or none,
#: never some -- a results.json without them is the pre-footing shape, which
#: _reconcile_done_marker refuses
FOOTING_KEYS = ("audited", "auditor_refused", "auditor_chain")


class DriverError(RuntimeError):
    """A driver refusal — loud, never a default."""


# --- injection seams -----------------------------------------------------

#: test-injection override: a callable ``spec -> CellSource`` that
#: REPLACES the real TrainTest registry adapter (tests supply a synthetic
#: source). Unset (None) means the real adapter below.
CELL_SOURCE_FACTORY: Optional[Callable[[cm.PetriSpec], dc.CellSource]] = None


def _check_serving_path(serving_path: str) -> None:
    if serving_path not in SERVING_PATHS:
        raise DriverError(f"serving_path {serving_path!r} is not one of "
                          f"{SERVING_PATHS}")


def load_cell_source(spec: cm.PetriSpec, *,
                     serving_path: str = SERVING_PATH_LENS) -> dc.CellSource:
    """The cell registry for ``spec`` under ``serving_path``'s policy.
    ``lens`` (the default) is today's build byte for byte; ``lora_native``
    builds the same adapter with ``serving_path_policy=
    "lora_native"`` — every object entry registered on the ``lora`` route,
    a by-name refusal for any cell that cannot be a native adapter (the
    backstop; the by-name SKIP happens earlier, over a lens-policy source,
    in :func:`enumerate_petri_native`). ``CELL_SOURCE_FACTORY`` wins under
    either path: a synthetic source's entries carry their own
    ``serving_path``, so one injected source serves both readings."""
    _check_serving_path(serving_path)
    if CELL_SOURCE_FACTORY is not None:
        return CELL_SOURCE_FACTORY(spec)
    # The REAL registry: the landed DefenseCell matrix behind the
    # CellSource protocol (closed at the two-SI merge). Lazy
    # import — registry_adapter pulls the methods stack (torch),
    # which this module must not import at parse time. A missing/
    # inconsistent configs/two_si.yaml refuses loudly inside.
    from sisteer.capeval.petri import registry_adapter

    # PETRI_TWO_SI_CONFIG (2026-08-30, gemma4e4b audit): the two-SI config
    # the registry derives from. The default (configs/two_si.yaml) is the
    # Llama model's; a non-Llama audit manifest points here at its own
    # model's config (e.g. configs/two_si_gemma4e4b_reduced.yaml). Not a
    # silent fallback in either direction: traintest_cell_source REFUSES
    # when the config's model/model_revision pins disagree with the
    # manifest's, so a stale/wrong override cannot enumerate — and the
    # override is echoed loudly.
    cfg_env = os.environ.get("PETRI_TWO_SI_CONFIG")
    config_path = None
    if cfg_env:
        config_path = Path(cfg_env)
        print(f"[petri] two-SI registry config: {config_path} "
              "($PETRI_TWO_SI_CONFIG override)")
    if serving_path == SERVING_PATH_LORA_NATIVE:
        return registry_adapter.traintest_cell_source(
            spec, repo_root=REPO, config_path=config_path,
            serving_path_policy="lora_native")
    # the lens build is EXACTLY today's call (no new keyword): the golden
    # lens enumeration and every existing lens record depend on it
    return registry_adapter.traintest_cell_source(
        spec, repo_root=REPO, config_path=config_path)


def load_behavior_files(spec: cm.PetriSpec,
                        config_dir: Path = PETRI_CONFIG_DIR
                        ) -> Dict[str, bytes]:
    """The exact BEHAVIOR.md bytes of every behavior dir IN PLAY —
    canonical + the SI-prefix variant subdirs the spec's own prompted
    cells imply — keyed ``<behavior>`` / ``<behavior>/<VARIANT>``
    (harness-hash inputs). PRUNED to the spec-implied
    inventory exactly like petri_runner: the config tree carries a
    variant dir per REGISTERED SI (three since the irrelevant control SI
    joined, 2026-08-28), while the harness hash refuses any key outside
    ``petri_expected_behavior_keys(spec)`` — reading every on-disk
    variant unpruned would refuse for any spec narrower than the full
    vocabulary. A MISSING expected file still refuses loudly."""
    expected = cm.petri_expected_behavior_keys(spec)
    out: Dict[str, bytes] = {}
    for key in expected:
        path = config_dir / key / "BEHAVIOR.md"
        if not path.is_file():
            raise DriverError(
                f"missing behavior definition {path} — every in-play "
                "behavior dir (canonical + SI-variant) is a harness-hash "
                "input (plan §5 H10/§6); refusing to hash a behavior that "
                "has no bytes")
        out[key] = path.read_bytes()
    return out


def load_prompt_files(spec: cm.PetriSpec,
                      config_dir: Path = PETRI_CONFIG_DIR
                      ) -> Dict[str, bytes]:
    """The exact auditor/judge prompt BYTES in use (harness-hash inputs).
    ``override@<hex>`` provenance reads the local override
    files (``configs/capeval/petri/prompts/<role>.md``) — the manifest
    hash machinery verifies the declared hex against the loaded bytes.
    ``package-pinned@`` provenance refuses here: the bytes ship inside the
    pinned petri-bloom/inspect packages (the petri venv), and the
    launch-phase wiring resolves them from the installed package — a
    deliberate seam, never a guessed byte payload."""
    pp = spec.prompt_provenance
    if pp.startswith("override@"):
        out: Dict[str, bytes] = {}
        for role in cm.PETRI_PROMPT_ROLES:
            path = config_dir / "prompts" / f"{role}.md"
            if not path.is_file():
                raise DriverError(
                    f"prompt_provenance {pp!r} declares a local override "
                    f"but {path} is missing — the override's exact bytes "
                    "are hash inputs (plan §5 H10)")
            out[role] = path.read_bytes()
        return out
    raise DriverError(
        f"prompt_provenance {pp!r} pins the auditor/judge prompt bytes "
        "inside the installed petri-bloom/inspect packages — resolving "
        "them needs the petri venv and lands with the launch-phase wiring "
        "(plan §11); refusing to hash a provenance whose bytes cannot be "
        "read here (a synthetic prompt_files mapping is test-only)")


@dataclasses.dataclass(frozen=True)
class NativeEnumeration:
    """What ``--serving-path lora_native`` enumerates: the served legs
    over the EFFECTIVE spec — the manifest's cells minus the ones the
    native standup skipped by name, the manifest's launch flags minus
    the ones the standup strips — plus the skip record and the dropped
    flags, both written into ``cells.json``."""

    cell_set: dc.PetriCellSet
    #: the spec the legs were HASHED under: ``spec`` with ``cells`` narrowed
    #: to the served set and ``serving_launch_flags`` passed through
    #: ``native_standup.effective_launch_flags`` — what the standup hashed
    effective_spec: cm.PetriSpec
    #: ``registry_adapter.SkippedCell`` per skipped cell, in manifest order
    skipped: Tuple[Any, ...]
    #: the manifest flag tokens the native launch drops (recorded by the
    #: standup as ``manifest_flags_dropped``)
    flags_dropped: Tuple[str, ...]


def native_effective_spec(spec: cm.PetriSpec, lens_source: dc.CellSource
                          ) -> Tuple[cm.PetriSpec, Tuple[Any, ...],
                                     Tuple[str, ...]]:
    """``(effective_spec, skipped, flags_dropped)`` for ONE native standup
    of ``spec``. Pure over ``lens_source`` — a LENS-policy
    registry (kind and resolved spec are READ; nothing is exported): the
    partition is ``registry_adapter.partition_cells_for_native`` (none /
    prompted served; PSR / vector cells served on the residual slot; stacked
    and every other kind without a native form SKIPPED by name, and so is a
    position-masked LoRA fit;
    a LoRA cell the view meant to serve whose record the engine cannot
    judge REFUSES, naming ``--serving-path lens``), the flags are
    ``native_standup.effective_launch_flags`` — the ONE derivation the
    standup hashes with, so ``petri_harness_config_hash`` over the result
    is the standup's. Applied BEFORE enumeration, never after: the run key
    of every leg is a function of this spec. One log line per skipped
    cell (never a silent drop)."""
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.serving import native_standup as ns

    try:
        served, skipped = ra.partition_cells_for_native(list(spec.cells),
                                                        lens_source)
    except ra.RegistryAdapterError as e:
        raise DriverError(
            f"--serving-path {SERVING_PATH_LORA_NATIVE}: {e}") from e
    for s in skipped:
        print(f"[petri] skipped {s.cell_id} ({s.kind}): {s.reason}")
    kept, dropped = ns.effective_launch_flags(list(spec.serving_launch_flags))
    eff = dataclasses.replace(spec, cells=list(served),
                              serving_launch_flags=list(kept))
    return eff, tuple(skipped), tuple(dropped)


def enumerate_petri_native(spec: cm.PetriSpec,
                           source: Optional[dc.CellSource] = None,
                           behavior_files: Optional[Mapping[str, bytes]] = None,
                           prompt_files: Optional[Mapping[str, bytes]] = None
                           ) -> NativeEnumeration:
    """The ``--serving-path lora_native`` enumeration.

    Two registry readings: the LENS-policy source the partition classifies
    over (:func:`native_effective_spec`), then the NATIVE-policy source the
    legs are enumerated from (``traintest_cell_source(serving_path_policy=
    "lora_native")`` — every object cell on the ``lora`` route). With
    ``source`` given, or ``CELL_SOURCE_FACTORY`` set, ONE source serves
    both readings: a synthetic ``MappingCellSource`` carries no policy (the
    partition reads it as lens) and its entries carry their own
    ``serving_path``; a real ``TrainTestCellSource`` handed in under the
    native policy refuses at the partition (it would refuse a skip-class
    cell at ``get`` before it could be skipped by name).

    The behavior definitions are loaded for the EFFECTIVE spec (the harness
    hash covers exactly ``petri_expected_behavior_keys(eff_spec)``, and
    narrowing the cells can narrow the in-play SI variants), exactly as the
    standup loads them. Then ``enumerate_cells(lens_controls=False,
    lora_controls=True)``: no ``none__lens0`` (no lens hook on this
    server), ``none__lora0`` derived even with no genuine lora route (the
    zero-adapter-only standup), every lora route re-checked against
    ``native_lora_unservable_reason`` (the backstop)."""
    if source is not None:
        lens_source: dc.CellSource = source
        native_source: dc.CellSource = source
    elif CELL_SOURCE_FACTORY is not None:
        lens_source = native_source = CELL_SOURCE_FACTORY(spec)
    else:
        lens_source = load_cell_source(spec, serving_path=SERVING_PATH_LENS)
        native_source = load_cell_source(spec,
                                         serving_path=SERVING_PATH_LORA_NATIVE)
    eff, skipped, dropped = native_effective_spec(spec, lens_source)
    files = (behavior_files if behavior_files is not None
             else load_behavior_files(eff))
    prompts = (prompt_files if prompt_files is not None
               else load_prompt_files(eff))
    cell_set = dc.enumerate_cells(eff, native_source, behavior_files=files,
                                  prompt_files=prompts, lens_controls=False,
                                  lora_controls=True)
    return NativeEnumeration(cell_set=cell_set, effective_spec=eff,
                             skipped=skipped, flags_dropped=dropped)


def enumerate_petri(spec: cm.PetriSpec,
                    source: Optional[dc.CellSource] = None,
                    behavior_files: Optional[Mapping[str, bytes]] = None,
                    prompt_files: Optional[Mapping[str, bytes]] = None,
                    *, serving_path: str = SERVING_PATH_LENS
                    ) -> dc.PetriCellSet:
    """The served legs under ``serving_path``. ``lens`` (the default) is
    EXACTLY today's enumeration — the lens-policy source, the manifest's
    flags as written, ``dc.enumerate_cells`` with no new keyword;
    ``lora_native`` is :func:`enumerate_petri_native`'s cell set (callers
    that need the effective spec, the skip record or the dropped flags call
    that directly)."""
    _check_serving_path(serving_path)
    if serving_path == SERVING_PATH_LORA_NATIVE:
        return enumerate_petri_native(spec, source, behavior_files,
                                      prompt_files).cell_set
    source = source if source is not None else load_cell_source(spec)
    files = (behavior_files if behavior_files is not None
             else load_behavior_files(spec))
    prompts = (prompt_files if prompt_files is not None
               else load_prompt_files(spec))
    return dc.enumerate_cells(spec, source, behavior_files=files,
                              prompt_files=prompts)


# --- helpers ---------------------------------------------------------------------


def _default_out_root(spec: cm.PetriSpec) -> Path:
    env = os.environ.get("PETRI_OUT_ROOT")
    base = Path(env) if env else Path.home() / ".local" / "state" / "petri"
    return base / spec.audit


def _out_root(args: argparse.Namespace, spec: cm.PetriSpec) -> Path:
    return Path(args.out_root) if args.out_root else _default_out_root(spec)


def _cells_payload(cell_set: dc.PetriCellSet, spec: cm.PetriSpec, *,
                   native: Optional[NativeEnumeration] = None
                   ) -> Dict[str, Any]:
    """The enumeration-record content (everything except the write
    timestamp) — the unit of the immutability comparison below.

    Under ``lens`` (``native`` None) the payload is BYTE-IDENTICAL to every
    record written before the serving path existed: the same five keys, no
    more (an existing frozen record must still compare equal, and every
    reader of ``cells.json`` — ``_check_record_matches_spec``, ``bind_cell_
    dirs``, ``cmd_analyze`` — reads exactly these). Under ``lora_native``
    three ADDITIVE keys record what the native enumeration did:
    ``serving_path``, ``cells_skipped`` (``registry_adapter.SkippedCell``
    dicts, manifest order) and ``flags_dropped`` (the manifest flag tokens
    the standup strips). ``expected_cells`` stays the MANIFEST's cells
    under both paths — the record belongs to the manifest, and the served
    subset is already what ``canonical`` lists — so ``_check_record_
    matches_spec`` is unchanged."""
    payload = {
        "audit": spec.audit,
        "expected_cells": list(spec.cells),
        "canonical": [dataclasses.asdict(c) for c in cell_set.canonical],
        "controls": [dataclasses.asdict(c) for c in cell_set.controls],
        "absent": [{"cell_id": a.cell_id, "reason": a.reason}
                   for a in cell_set.absent],
    }
    if native is not None:
        payload["serving_path"] = SERVING_PATH_LORA_NATIVE
        payload["cells_skipped"] = [dict(s._asdict()) for s in native.skipped]
        payload["flags_dropped"] = list(native.flags_dropped)
    # the TARGET decode contract the manifest declares: ONE additive
    # key, present iff vendor -- {mode, policy}, never the seed base (the
    # record belongs to the manifest; a seed-run is a run of it) -- so every
    # harness record, lens or native, stays byte-identical
    td_record = cm.petri_target_decode_record(spec)
    if td_record is not None:
        payload[TARGET_DECODE_RECORD_KEY] = td_record
    return payload


def _enumerate_under(spec: cm.PetriSpec, serving_path: str
                     ) -> Tuple[dc.PetriCellSet, Dict[str, Any],
                                Optional[NativeEnumeration]]:
    """``(cell_set, cells.json payload, native enumeration | None)`` under
    ``serving_path`` — the ONE derivation ``print-cells`` freezes and ``run``
    re-derives for its drift check. The lens branch is EXACTLY today's two
    calls (``enumerate_petri(spec)``, ``_cells_payload(cell_set, spec)``),
    so the lens record is byte-identical."""
    _check_serving_path(serving_path)
    if serving_path == SERVING_PATH_LORA_NATIVE:
        native = enumerate_petri_native(spec)
        return (native.cell_set,
                _cells_payload(native.cell_set, spec, native=native), native)
    cell_set = enumerate_petri(spec)
    return cell_set, _cells_payload(cell_set, spec), None


def _record_content(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in record.items() if k != "written_at"}


def write_cells_json(cell_set: dc.PetriCellSet, spec: cm.PetriSpec,
                     out_root: Path, *,
                     payload: Optional[Mapping[str, Any]] = None) -> Path:
    """Write the ONE atomic, IMMUTABLE enumeration snapshot (review
    2026-08-18: array indices resolve only against this record, so it is
    written once before launch and never remapped/overwritten). A
    re-enumeration that matches exactly is a no-op; any difference is a
    refusal — delete the stale record deliberately if the world
    legitimately changed, never overwrite it under running arrays.
    ``payload`` is the record content already derived for ``cell_set``
    (:func:`_enumerate_under`; the native payload carries the skip record);
    absent, the lens payload is derived here exactly as before."""
    out_root.mkdir(parents=True, exist_ok=True)
    payload = (dict(payload) if payload is not None
               else _cells_payload(cell_set, spec))
    path = out_root / CELLS_JSON
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _record_content(existing) == payload:
            return path  # identical — the frozen record stands
        raise DriverError(
            f"{path} already exists and DIFFERS from the current "
            "enumeration — the record is immutable once written (array "
            "indices resolve against it); if the registry/artifacts "
            "legitimately changed, delete the stale record deliberately "
            "and re-run print-cells BEFORE launching any array")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({**payload, "written_at": time.time()},
                              indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_cells_record(out_root: Path) -> Dict[str, Any]:
    """The frozen enumeration snapshot — a refusal when absent (it is the
    authority for index resolution and for ABSENT cells)."""
    path = out_root / CELLS_JSON
    if not path.exists():
        raise DriverError(
            f"{path} missing — run `print-cells --out-root` first; the "
            "frozen enumeration snapshot is the ONLY authority for array "
            "index resolution and ABSENT cells (plan §5)")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_record_matches_spec(record: Mapping[str, Any],
                               spec: cm.PetriSpec, path: Path) -> None:
    """The enumeration record must belong to THIS manifest — a record from
    another audit (or an edited cell list) never resolves anything."""
    if record.get("audit") != spec.audit:
        raise DriverError(
            f"{path}: record audit {record.get('audit')!r} != the loaded "
            f"manifest's {spec.audit!r} — a foreign enumeration record "
            "never drives this audit")
    if list(record.get("expected_cells") or []) != list(spec.cells):
        raise DriverError(
            f"{path}: record expected_cells differ from the loaded "
            "manifest's cells — the record and the manifest disagree; "
            "re-enumerate deliberately rather than running against a "
            "stale record")
    recorded_td = record.get(TARGET_DECODE_RECORD_KEY)
    want_td = cm.petri_target_decode_record(spec)
    if recorded_td != want_td:
        raise DriverError(
            f"{path}: the frozen record declares target decode {recorded_td!r} "
            f"but the loaded manifest declares {want_td!r} (absent = harness) "
            "— the legs it names were hashed under the other contract; "
            "refusing to resolve indices against it (re-freeze under this "
            "manifest into a fresh out root)")


def _check_record_serving_path(record: Mapping[str, Any], serving_path: str,
                               path: Path) -> None:
    """The frozen record was written under THIS run's ``--serving-path``. A
    lens record carries no ``serving_path`` key (its bytes predate the flag,
    and every record written before the native path existed was lens by
    construction — the same "absent = lens by era" rule the cell manifests
    read under); a native record always names its path. A run under the
    other path would resolve indices against legs hashed for the other
    engine, so it refuses by name rather than through the drift message."""
    recorded = record.get("serving_path", SERVING_PATH_LENS)
    if recorded != serving_path:
        raise DriverError(
            f"{path}: the frozen enumeration record was written under "
            f"--serving-path {recorded} but this run is --serving-path "
            f"{serving_path} — the legs it names were hashed for the other "
            "engine; refusing to resolve indices against it (re-freeze under "
            "this path into a fresh out root, or run under the record's path)")


def _resolve_cell(served: Sequence[dc.PetriCell],
                  args: argparse.Namespace) -> dc.PetriCell:
    """Resolve exactly one served leg from the FROZEN snapshot order. The
    two selectors are mutually exclusive (argparse enforces it too):
    silently preferring one would run the wrong cell (review 2026-08-18)."""
    if (args.cell_index is None) == (args.cell_id is None):
        raise DriverError(
            "run takes EXACTLY ONE of --cell-index / --cell-id — supplying "
            "both (or neither) is ambiguous and refuses rather than "
            "silently preferring the index")
    if args.cell_index is not None:
        if not 0 <= args.cell_index < len(served):
            raise DriverError(
                f"--cell-index {args.cell_index} out of range 0.."
                f"{len(served) - 1} ({len(served)} served legs)")
        return served[args.cell_index]
    matches = [c for c in served if c.cell_id == args.cell_id]
    if len(matches) != 1:
        raise DriverError(f"--cell-id {args.cell_id!r} matches "
                          f"{len(matches)} served legs")
    return matches[0]


def _row_constants(spec: cm.PetriSpec, cell: dc.PetriCell,
                   seed: int) -> Dict[str, Any]:
    return {
        "sweep": spec.audit,
        # the CANONICAL run identity — must match petri_runner._row_fields
        # exactly (audit.cell_id.run_key[:8]); analyze verifies it per row
        "run_id": f"{spec.audit}.{cell.cell_id}.{cell.petri_run_key[:8]}",
        "model_id": spec.target_model,
        "model_revision": spec.target_revision,
        "arm_name": cell.cell_id,
        "kind": cell.kind,
        "coeff": None,
        "intervention_hash": cell.intervention_hash,
        "engine": cell.serving_identity,
        "trial_id": 0,
        "seed": seed,
        "attempt": 0,
        "harness_config_hash": cell.harness_config_hash,
        "env_digest": cs.env_digest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


# --- the target decode contract on argv ---------------------------------------------


def target_decode_intent_of(args: argparse.Namespace) -> Optional[str]:
    """``--target-decode`` off the parsed argv: None (no assertion; the
    manifest rules) or one of ``manifest.PETRI_TARGET_DECODE_MODES``. A
    caller-built namespace that predates the flag reads None."""
    intent = getattr(args, "target_decode", None)
    if intent is None:
        return None
    intent = str(intent)
    if intent not in cm.PETRI_TARGET_DECODE_MODES:
        raise DriverError(f"--target-decode {intent!r} not in "
                          f"{list(cm.PETRI_TARGET_DECODE_MODES)}")
    return intent


def check_target_decode_intent(args: argparse.Namespace, spec: cm.PetriSpec
                               ) -> str:
    """Hold ``--target-decode`` to the manifest's effective mode and return
    that mode. The flag cannot SET the contract (module docstring: the
    standup hashed the manifest; a client-only mode could never agree with
    ``cells_served.json``), so a disagreement is a refusal naming both -- a
    drive that intends the vendor contract against a manifest that does not
    declare it never runs a harness audit under a vendor label, and vice
    versa."""
    mode = cm.petri_target_decode_mode(spec)
    intent = target_decode_intent_of(args)
    if intent is not None and intent != mode:
        declared = cm.petri_target_decode_record(spec)
        raise DriverError(
            f"--target-decode {intent} but the manifest's effective target "
            f"decode is {mode!r} (target_decode: {declared!r}; absent = "
            f"{cm.PETRI_TARGET_DECODE_HARNESS}) — the contract is a harness-"
            "hash input the standup hashed from the manifest, so the flag "
            "can only assert it, never change it; fix the manifest (and "
            "re-stand-up) or the flag")
    return mode


def target_seed_base_of(args: argparse.Namespace) -> Optional[int]:
    """``--target-seed-base`` off the parsed argv (None when absent or when
    the namespace predates the flag); a bool or a negative int refuses."""
    raw = getattr(args, "target_seed_base", None)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise DriverError(f"--target-seed-base must be a non-negative int, "
                          f"got {raw!r}")
    return int(raw)


def apply_target_seed_base(spec: cm.PetriSpec, args: argparse.Namespace
                           ) -> cm.PetriSpec:
    """The spec the runner gets: under VENDOR, ``spec`` with the run's seed
    base on its ``target_decode`` block (``--target-seed-base`` is REQUIRED —
    an unseeded preset draw is what the contract forbids); under HARNESS,
    ``spec`` itself (the same object, byte for byte) — and a seed base given
    anyway refuses (the harness sends no seed; the label would be a lie).
    The seed base is not a hash input, so the returned spec's legs keep their
    ``harness_config_hash`` / ``petri_run_key``."""
    mode = cm.petri_target_decode_mode(spec)
    seed_base = target_seed_base_of(args)
    if mode == cm.PETRI_TARGET_DECODE_HARNESS:
        if seed_base is not None:
            raise DriverError(
                f"--target-seed-base {seed_base} under target decode "
                f"{cm.PETRI_TARGET_DECODE_HARNESS!r} — the harness target sends "
                "no seed, so a seed base would label rows with a rule that "
                "never ran; drop the flag or declare the vendor contract in "
                "the manifest")
        return spec
    td = cm.petri_target_decode_of(spec)
    if seed_base is None:
        raise DriverError(
            f"target decode {cm.PETRI_TARGET_DECODE_VENDOR!r} (policy "
            f"{td.policy!r}) requires --target-seed-base <s>: "
            "every vendor target request carries a seed derived from the "
            "run's literal training seed (spec fixed decision 3); refusing "
            "an unseeded preset draw")
    return dataclasses.replace(
        spec, target_decode=dataclasses.replace(td, seed_base=seed_base))


# --- the serving path on argv, and the standup's evidence -------------------------


def serving_path_of(args: argparse.Namespace) -> str:
    """``--serving-path`` off the parsed argv. The parser defaults it to
    ``lens``; a caller-built namespace that predates the flag (no attribute)
    reads the same default — the lens behaviour every such caller had —
    and anything outside :data:`SERVING_PATHS` refuses."""
    path = getattr(args, "serving_path", None)
    path = SERVING_PATH_LENS if path is None else str(path)
    _check_serving_path(path)
    return path


def load_standup_record(path: Path) -> Dict[str, Any]:
    """The standup's ``cells_served.json`` (``--verify-standup``): a mapping
    carrying the served legs as ``canonical`` / ``controls`` lists of
    ``PetriCell`` dicts — the shape ``native_standup.LoraStandup.
    _cells_served`` writes at readiness and the lens standups write too. A
    native standup's PRE-LAUNCH record (``stage: pre_launch``, written
    before the server came up, no served legs yet) is refused by name:
    it is not evidence of what is served."""
    path = Path(path)
    if not path.is_file():
        raise DriverError(
            f"--verify-standup {path}: no such file — the standup's "
            "cells_served.json is the authoritative served set; refusing to "
            "audit without it")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise DriverError(f"--verify-standup {path}: not JSON ({e})") from e
    if not isinstance(record, Mapping):
        raise DriverError(f"--verify-standup {path}: not a mapping "
                          f"({type(record).__name__})")
    if record.get("stage") == "pre_launch":
        raise DriverError(
            f"--verify-standup {path}: this is the standup's PRE-LAUNCH "
            "record (stage=pre_launch) — the served legs are recorded at "
            "readiness; point at the final cells_served.json")
    for key in ("canonical", "controls"):
        if not isinstance(record.get(key), list):
            raise DriverError(
                f"--verify-standup {path}: no {key!r} list — not a standup's "
                "cells_served.json (the served legs are its canonical / "
                "controls lists)")
    return dict(record)


def _standup_entries(standup: Mapping[str, Any], problems: List[str]
                     ) -> Dict[str, Mapping[str, Any]]:
    """cell_id -> the standup's entry, over ``canonical`` + ``controls``;
    malformed or duplicate entries are named into ``problems``."""
    theirs: Dict[str, Mapping[str, Any]] = {}
    for section in ("canonical", "controls"):
        for i, entry in enumerate(standup.get(section) or []):
            where = f"cells_served.json {section}[{i}]"
            if not isinstance(entry, Mapping) or "cell_id" not in entry:
                problems.append(f"{where}: not a served-cell entry (no cell_id)")
                continue
            cid = str(entry["cell_id"])
            if cid in theirs:
                problems.append(f"{where}: cell {cid!r} is served twice by the "
                                "standup record")
                continue
            theirs[cid] = entry
    return theirs


def standup_verification_problem(cell_set: dc.PetriCellSet,
                                 standup: Mapping[str, Any],
                                 table: Mapping[str, Any], *,
                                 serving_path: str,
                                 spec: Optional[cm.PetriSpec] = None,
                                 requested: Optional[Sequence[str]] = None
                                 ) -> Optional[str]:
    """Every way the client's enumeration DISAGREES with the standup's
    evidence — or ``None`` when they agree. With ``requested`` (the
    proxy's declared cells) the enumeration must equal the declared set
    (plus the controls twin only when ``none`` is declared) and every
    enumerated leg must be served and routed; the standup's extra served
    legs and routes (``none``, the twin) are served-not-audited, tolerated.
    Pure: nothing is read from disk
    (the caller loads ``cells_served.json`` through
    :func:`load_standup_record` and the table through
    ``proxy.load_routing_table``). The standup's records are the
    authoritative statement of what is served; the client's
    enumeration must agree with them cell by cell or the driver refuses
    BEFORE any leg. Held, and every disagreement named (never the first
    only):

    - the record's engine matches the path (a native record —
      ``engine == lora_native`` — under ``lora_native``; a record naming no
      native engine under ``lens``); a torn record refuses;
    - the served SETS are equal: every enumerated leg is in the record and
      every recorded leg (``canonical`` + ``controls``) is enumerated;
    - per cell, the five :data:`STANDUP_CELL_FIELDS` agree —
      ``virtual_model_name``, ``serving_identity``, ``intervention_hash``,
      ``harness_config_hash``, ``petri_run_key`` (the harness hash is what
      the effective launch flags feed; the run key is the store's name);
    - the routing table routes EXACTLY the enumerated set, and each route's
      ``virtual_model`` / ``serving_identity`` / ``intervention_hash`` agree
      with both the enumeration and the record (``RouteEntry`` carries no
      harness hash or run key — the record does);
    - under ``lora_native`` the table is version 2 and carries an adapter
      block for every enumerated lora leg (``load_routing_table`` already
      binds blocks to routes; this names the enumerated leg without one);
    - with ``spec`` given: the table's ``base_model`` / ``base_revision`` and
      the record's ``model`` / ``model_revision`` (when recorded) are the
      manifest's target — the standup is THIS model's."""
    _check_serving_path(serving_path)
    problems: List[str] = []
    native = serving_path == SERVING_PATH_LORA_NATIVE
    try:
        record_native = tc.endpoint_is_native(standup)
    except ValueError as e:
        problems.append(f"cells_served.json: {e}")
        record_native = native
    if record_native != native:
        problems.append(
            f"cells_served.json records engine="
            f"{standup.get(tc.ENDPOINT_KEY_ENGINE)!r} but this run is "
            f"--serving-path {serving_path} (expects "
            f"{'the native multi-LoRA engine' if native else 'a lens standup'})")
    if spec is not None:
        for key, want, label in (("model", spec.target_model, "target_model"),
                                 ("model_revision", spec.target_revision,
                                  "target_revision")):
            if key in standup and standup[key] != want:
                problems.append(f"cells_served.json {key}={standup[key]!r} != "
                                f"the manifest's {label} {want!r}")
        if table.get("base_model") != spec.target_model:
            problems.append(f"routing table base_model={table.get('base_model')!r}"
                            f" != the manifest's target_model {spec.target_model!r}")
        rev = table.get("base_revision")
        if rev is not None and rev != spec.target_revision:
            problems.append(f"routing table base_revision={rev!r} != the "
                            f"manifest's target_revision {spec.target_revision!r}")

    ours = {c.cell_id: c for c in cell_set.all_served()}
    theirs = _standup_entries(standup, problems)
    only_ours = sorted(set(ours) - set(theirs))
    only_theirs = sorted(set(theirs) - set(ours))
    if only_ours:
        problems.append(f"enumerated here but NOT served by the standup: {only_ours}")
    if requested is not None:
        # the enumeration IS the declared set (plus the twin only when none is declared); extra served legs are tolerated
        want = set(requested) | ({c.cell_id for c in cell_set.controls} if "none" in set(requested) else set())
        if set(ours) != want:
            problems.append(f"enumerated legs {sorted(ours)} != the proxy's declared cells {sorted(want)} ({REQUESTED_HARNESS_KEY})")
    elif only_theirs:
        problems.append(f"served by the standup but NOT enumerated here: {only_theirs}")
    for cid in sorted(set(ours) & set(theirs)):
        cell, entry = ours[cid], theirs[cid]
        for field in STANDUP_CELL_FIELDS:
            if field not in entry:
                problems.append(f"cell {cid!r}: the standup entry records no {field!r}")
                continue
            mine = getattr(cell, field)
            if entry[field] != mine:
                problems.append(f"cell {cid!r}: {field} enumerated {mine!r} but the "
                                f"standup served {entry[field]!r}")

    routes: Dict[str, sproxy.RouteEntry] = {}
    for r in table.get("routes") or ():
        if r.cell_id in routes:
            problems.append(f"routing table routes cell {r.cell_id!r} twice")
            continue
        routes[r.cell_id] = r
    unrouted = sorted(set(ours) - set(routes))
    foreign_routes = sorted(set(routes) - set(ours))
    if unrouted:
        problems.append(f"enumerated but NOT routed by the routing table: {unrouted}")
    if foreign_routes and requested is None:          # with a declaration the standup's extra routes (none, the twin) are tolerated
        problems.append(f"routed by the routing table but NOT enumerated here: "
                        f"{foreign_routes}")
    route_fields = (("virtual_model", "virtual_model_name"),
                    ("serving_identity", "serving_identity"),
                    ("intervention_hash", "intervention_hash"))
    for cid in sorted(set(ours) & set(routes)):
        cell, route, entry = ours[cid], routes[cid], theirs.get(cid)
        for rattr, cattr in route_fields:
            got = getattr(route, rattr)
            if got != getattr(cell, cattr):
                problems.append(f"cell {cid!r}: route {rattr}={got!r} != enumerated "
                                f"{getattr(cell, cattr)!r}")
            if entry is not None and cattr in entry and got != entry[cattr]:
                problems.append(f"cell {cid!r}: route {rattr}={got!r} != the standup's "
                                f"{cattr} {entry[cattr]!r}")
    if native:
        blocks = table.get(sproxy.ADAPTERS_KEY)
        if table.get("version") != 2 or not isinstance(blocks, Mapping) or not blocks:
            problems.append(
                f"routing table version {table.get('version')!r} without adapter "
                "blocks — a native standup serves from a version-2 table with a "
                "block per lora leg")
        else:
            missing_blocks = sorted(
                cid for cid, c in ours.items()
                if cm.parse_serving_identity(c.serving_identity).path == "lora"
                and cid not in blocks)
            if missing_blocks:
                problems.append(f"enumerated lora leg(s) with no adapter block in the "
                                f"routing table: {missing_blocks}")
    if not problems:
        return None
    lines = "\n  - ".join(problems)
    return (f"the client's enumeration under --serving-path {serving_path} "
            f"disagrees with the standup's evidence in {len(problems)} way(s) — "
            "the standup's cells_served.json / routing table are the authoritative "
            "served set; refusing before any leg:\n  - " + lines)


def requested_cells_of(standup: Mapping[str, Any], path: Any) -> Optional[List[str]]:
    """The harness cell ids the proxy's standup was asked to serve for this audit (:data:`REQUESTED_HARNESS_KEY`), or None when the
    record names none (an unnarrowed proxy: the whole enumeration). A non-list, a non-string entry or a duplicate
    refuses by name."""
    raw = standup.get(REQUESTED_HARNESS_KEY)
    if raw is None or raw == []:
        return None
    if not isinstance(raw, list) or not all(isinstance(x, str) and x for x in raw):
        raise DriverError(f"{path}: {REQUESTED_HARNESS_KEY} must be a list of harness cell ids, got {raw!r}")
    if len(set(raw)) != len(raw):
        raise DriverError(f"{path}: {REQUESTED_HARNESS_KEY} names a cell twice: {raw}")
    return [str(x) for x in raw]


def endpoint_record_path(args: argparse.Namespace, standup_path: Any) -> Path:
    """The live endpoint record a declaration is bound to: ``--served-endpoint`` when given, else ``endpoint.json`` beside the served
    record (``native_standup`` writes both into one directory; the cluster and modal drives pull both into one state directory)."""
    given = getattr(args, "served_endpoint", None)
    return Path(given) if given else Path(standup_path).with_name(ENDPOINT_JSON)


def load_endpoint_record(path: Path, *, standup_path: Any) -> Dict[str, Any]:
    """The endpoint record at ``path`` as a mapping; absent or malformed refuses by name."""
    path = Path(path)
    if not path.is_file():
        raise DriverError(f"{standup_path}: its declaration ({REQUESTED_HARNESS_KEY}) must be bound to the live endpoint's attempt, but there is "
                          f"no endpoint record at {path} -- pass --served-endpoint <endpoint.json> (the pulled endpoint record) or keep it "
                          "beside the served record; refusing to freeze from an unbound declaration")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise DriverError(f"{path}: not JSON ({e})") from e
    if not isinstance(record, Mapping):
        raise DriverError(f"{path}: not a mapping ({type(record).__name__})")
    return dict(record)


def _attempt_id_of(record: Mapping[str, Any]) -> Optional[str]:
    raw = record.get(STANDUP_ATTEMPT_KEY)
    return raw if isinstance(raw, str) and raw else None


def attempt_binding_problem(standup: Mapping[str, Any], endpoint: Mapping[str, Any], *,
                            standup_path: Any, endpoint_path: Any) -> Optional[str]:
    """Why the served record is NOT the live endpoint's (``None`` when it is). The two records' DECLARATIONS must agree before
    anything else is read (the reader never depends on the producer being the only writer):
    a served record that declares a nonempty set needs the endpoint's :data:`REQUESTED_HARNESS_KEY` PRESENT, well-formed and EQUAL
    AS A SET (the contract is set equality; the served record's order is what the frozen payload keeps,
    and an endpoint re-serialised or backfilled with the same harness cells in another order is the same declaration) -- a missing key
    is a pre-declaration endpoint snapshot or a backfilled served record and refuses by name; an
    endpoint that declares while the served record does not refuses the disagreement by name rather than letting the undeclared
    path run. A declared pair is then bound to ONE attempt: both state the same nonempty attempt id (a stale cells_served.json of an
    earlier attempt -- identical routes and leg hashes -- can carry another declaration). Two undeclared records need no binding
    (the whole enumeration is attempt-independent and verified cell by cell). Pure; a malformed declaration on either record
    raises the reader's own refusal naming that record."""
    declared_s = requested_cells_of(standup, standup_path)
    declared_e = requested_cells_of(endpoint, endpoint_path)
    if declared_s is None:
        if declared_e is not None:
            return (f"{endpoint_path} declares {declared_e} but {standup_path} declares nothing ({REQUESTED_HARNESS_KEY} absent or []) -- "
                    "torn records; refusing rather than freezing the whole enumeration against an endpoint that attests a subset")
        return None
    if REQUESTED_HARNESS_KEY not in endpoint:
        return (f"{standup_path} declares {declared_s} but {endpoint_path} carries no {REQUESTED_HARNESS_KEY} -- the live endpoint did not "
                "attest any declaration (a pre-declaration endpoint snapshot, or a backfilled served record); torn records")
    if declared_e is None or set(declared_e) != set(declared_s):          # SET equality (the writer dedups; order is not the contract)
        return (f"{standup_path} declares {declared_s} but {endpoint_path} declares {declared_e if declared_e is not None else []} -- "
                "torn records")
    sid, eid = _attempt_id_of(standup), _attempt_id_of(endpoint)
    if sid is None:
        return f"{standup_path} states no {STANDUP_ATTEMPT_KEY} -- the served record is not bound to any attempt"
    if eid is None:
        return f"{endpoint_path} states no {STANDUP_ATTEMPT_KEY} -- the endpoint record is not bound to any attempt"
    if sid != eid:
        return (f"{standup_path} is attempt {sid} but the live endpoint {endpoint_path} is attempt {eid} -- a stale served record of an "
                "earlier attempt (the same model and routes can carry a different declaration)")
    return None


def bind_declaration_to_attempt(args: argparse.Namespace, standup: Mapping[str, Any], standup_path: Any) -> Optional[str]:
    """The attempt id a DECLARED served record is bound to; ``None`` for an undeclared served record with no endpoint record to
    disagree with it (nothing to bind). Refuses, naming both records, when the endpoint record is missing for a declared
    served record (or for a named ``--served-endpoint``), or when :func:`attempt_binding_problem` finds the pair torn or stale.
    Read on every path: before anything is frozen and
    before every leg runs, so an endpoint that declares while the served record does not can never route to the undeclared path."""
    declared_s = requested_cells_of(standup, standup_path)
    ep_path = endpoint_record_path(args, standup_path)
    if declared_s is None and getattr(args, "served_endpoint", None) is None and not ep_path.is_file():
        return None                                     # undeclared, and no endpoint record beside it: nothing can disagree
    endpoint = load_endpoint_record(ep_path, standup_path=standup_path)   # a declared record (or a named flag) needs the endpoint record
    problem = attempt_binding_problem(standup, endpoint, standup_path=standup_path, endpoint_path=ep_path)
    if problem is not None:
        raise DriverError(f"the proxy's declaration is not bound to the live attempt: {problem}; refusing before any leg")
    return str(standup[STANDUP_ATTEMPT_KEY]) if declared_s is not None else None


def complete_baseline_problem(cell_set: dc.PetriCellSet, requested: Sequence[str], spec: cm.PetriSpec, path: Any) -> Optional[str]:
    """A u0 / legacy audit's legs are the WHOLE baseline (its frozen ``expected_cells`` advertises the
    manifest), so a declaration there must name that baseline COMPLETELY -- every enumerated canonical leg, ``none`` included -- or it
    silently narrows the baseline. ``None`` when the declared set is the complete baseline (the restriction is then a no-op)."""
    complete = [c.cell_id for c in cell_set.canonical]
    if set(requested) == set(complete):
        return None
    return (f"{path}: audit {spec.audit!r} is a u0 / legacy audit whose legs are the whole baseline, but {REQUESTED_HARNESS_KEY} declares "
            f"{sorted(requested)} while the complete baseline is {sorted(complete)} (missing {sorted(set(complete) - set(requested))}, "
            f"extra {sorted(set(requested) - set(complete))}) -- a partial declaration would narrow the baseline while expected_cells "
            "advertises the full manifest; declare the complete baseline exactly, or declare nothing")


def frozen_record_is_complete_baseline(record: Mapping[str, Any]) -> bool:
    """True when the frozen enumeration record is a COMPLETE-BASELINE record: it declares nothing (:data:`REQUESTED_HARNESS_KEY` absent or
    ``[]``) or declares exactly its own canonical ids. A record declaring a strict subset (a steered view's) is not one. Pure."""
    declared = record.get(REQUESTED_HARNESS_KEY)
    canonical = {str(c.get("cell_id")) for c in list(record.get("canonical") or []) if isinstance(c, Mapping)}
    if declared is None or declared == []:
        return True
    if not isinstance(declared, list):
        return False
    return {str(x) for x in declared} == canonical


def steered_wave_audit(spec: cm.PetriSpec) -> bool:
    """A steered-wave view's audit: it drives ONLY the cells the proxy declared; a record declaring none refuses. The s-family audits
    (``…_s<N>r`` / ``…_s<N>rl``) by name; and a FRONT-DOOR target's dose view (``…_d<dose>``,
    ``…_d<dose>_relu``, ``…_d<dose>_relul``) by the models table (sisteer.front_door_views: every non-u0 phase view of a front-door
    target measures no unsteered leg; the table is read once per revision and only for a dose-shaped audit). A paper target's dose-named view
    (``petri_full22_qwen35_4b_d0p1`` …) is NOT one (same-wave none anchor), ``s1`` keeps its legacy reading and a declared single-phase wave (the worked example's ``battery``) is no steered audit; a checkout without
    reproduce/lib/models.py is refused by name for a dose-shaped audit, never read as u0."""
    audit = str(spec.audit)
    if _STEERED_AUDIT_RE.search(audit):
        return True
    from sisteer import front_door_views as fdv
    return fdv.front_door_dose_tag(audit) is not None


def restrict_to_requested(cell_set: dc.PetriCellSet, requested: Sequence[str], spec: cm.PetriSpec,
                          standup: Mapping[str, Any], path: Any) -> dc.PetriCellSet:
    """The enumeration narrowed to the proxy's declaration: canonical = the enumerated canonical legs named in ``requested``, in
    record order; controls = the enumeration's controls only when ``none`` is requested; absent unchanged. Refused by name: a requested
    id outside the manifest's `cells` universe, one the enumeration does not carry as a canonical leg (absent from the registry, or
    skipped by name), one the standup does not serve (``harness_cells`` when the record carries it: requested ⊄ served)."""
    universe = set(spec.cells)
    outside = [c for c in requested if c not in universe]
    if outside:
        raise DriverError(f"{path}: {REQUESTED_HARNESS_KEY} names cell(s) outside the manifest's universe {sorted(universe)}: {outside} -- "
                          "the declared set must be a subset of the manifest's cells; refusing")
    enumerated = {c.cell_id for c in cell_set.canonical}
    missing = [c for c in requested if c not in enumerated]
    if missing:
        absent = {a.cell_id: a.reason for a in cell_set.absent}
        why = {c: absent.get(c, "not an enumerated canonical leg (skipped by name, or a controls id)") for c in missing}
        raise DriverError(f"{path}: {REQUESTED_HARNESS_KEY} names cell(s) this enumeration does not carry: {why}; refusing")
    served = standup.get("harness_cells")
    if isinstance(served, list):
        unserved = [c for c in requested if c not in {str(s) for s in served}]
        if unserved:
            raise DriverError(f"{path}: {REQUESTED_HARNESS_KEY} names cell(s) the standup does not serve (harness_cells {served}): {unserved}; "
                              "refusing (requested must be a subset of served)")
    wanted = set(requested)
    canonical = tuple(c for c in cell_set.canonical if c.cell_id in wanted)
    controls = cell_set.controls if "none" in wanted else ()
    return dataclasses.replace(cell_set, canonical=canonical, controls=controls)


def _frozen_enumeration(args: argparse.Namespace, spec: cm.PetriSpec, serving_path: str, *,
                        rerun_subset_of: Optional[Mapping[str, Any]] = None
                        ) -> Tuple[dc.PetriCellSet, Dict[str, Any], Optional[NativeEnumeration], Optional[List[str]]]:
    """``(cell_set, cells.json payload, native enumeration | None, requested | None)``: :func:`_enumerate_under`, then -- when the standup
    record is given and declares the cells it was asked to serve -- the enumeration restricted to that declaration and the payload
    carrying it (:data:`REQUESTED_HARNESS_KEY`, additive), so `print-cells` freezes exactly the declared legs and `run` re-derives the same
    record. A steered-wave audit whose record declares nothing refuses by name (the whole enumeration would drive none and none__lora0
    against the rule); a u0 / legacy audit keeps the whole enumeration, as before. A declaration is read only
    from a served record BOUND to the live endpoint's attempt (:func:`bind_declaration_to_attempt`), and a u0 / legacy declaration
    must name the complete baseline (:func:`complete_baseline_problem`).

    ``rerun_subset_of`` (2026-10-13; the baseline view's overflow pass): the FROZEN first-pass record, passed by
    ``cmd_run`` ONLY under ``--overflow-rerun`` (``print-cells`` and a plain ``run`` never pass it, so the freeze and the first pass keep
    the completeness refusal byte for byte). A rerun proxy of a baseline view serves only the overflowed cells by
    design, so its standup declares a subset — and the completeness check, written so a FREEZE can never narrow the baseline, would
    refuse every leg of a drive that freezes nothing. The check is skipped IFF the frozen record is itself a complete-baseline record (it
    declares nothing, or exactly its own canonical ids); the live declaration must then be a NON-EMPTY subset of the frozen canonical ids
    (refused by name otherwise), :func:`restrict_to_requested` applies unchanged (the controls twin only when ``none`` is declared), and
    ``cmd_run``'s existing subset check holds every live leg to its frozen identity. A frozen record that itself declares a strict subset
    (a steered shape) keeps today's behaviour unchanged."""
    cell_set, payload, native = _enumerate_under(spec, serving_path)
    standup_path = getattr(args, "verify_standup", None)
    if standup_path is None:
        return cell_set, payload, native, None
    standup = load_standup_record(Path(standup_path))
    requested = requested_cells_of(standup, standup_path)
    bind_declaration_to_attempt(args, standup, standup_path)     # the two records' declarations agree BEFORE any path is taken
    if requested is None:
        if steered_wave_audit(spec):
            raise DriverError(f"{standup_path}: audit {spec.audit!r} is a steered-wave view and drives only the cells the proxy declared "
                              f"({REQUESTED_HARNESS_KEY}), but the record declares none -- a whole-enumeration drive would audit `none` and the "
                              "controls twin against the ruling (D1261); refusing")
        return cell_set, payload, native, None
    if not steered_wave_audit(spec):
        if rerun_subset_of is not None and frozen_record_is_complete_baseline(rerun_subset_of):
            frozen_canonical = [str(c.get("cell_id")) for c in list(rerun_subset_of.get("canonical") or [])]
            outside = [c for c in requested if c not in frozen_canonical]
            if outside:
                raise DriverError(f"{standup_path}: --overflow-rerun: the live declaration {list(requested)} names {outside}, not among the frozen "
                                  f"first-pass legs {frozen_canonical}; a rerun re-serves a subset of the baseline it never re-freezes; refusing")
            print(f"[petri] --overflow-rerun on the baseline audit {spec.audit!r}: the live declaration {list(requested)} is a subset of the "
                  f"frozen complete baseline {frozen_canonical} -- the completeness check is for a freeze, and a rerun freezes nothing "
                  "(petri-completion)", flush=True)
        else:
            problem = complete_baseline_problem(cell_set, requested, spec, standup_path)
            if problem is not None:
                raise DriverError(problem)
    cell_set = restrict_to_requested(cell_set, requested, spec, standup, standup_path)
    payload = _cells_payload(cell_set, spec, native=native)
    payload[REQUESTED_HARNESS_KEY] = list(requested)
    return cell_set, payload, native, requested


def verify_standup_or_refuse(args: argparse.Namespace, spec: cm.PetriSpec,
                             cell_set: dc.PetriCellSet, serving_path: str
                             ) -> None:
    """``--verify-standup`` + ``--verify-routing-table`` against ``cell_set``
    (:func:`standup_verification_problem`), refusing on any disagreement.
    Under ``lora_native`` BOTH flags are REQUIRED (refused by name when
    absent: the native enumeration is the client's reconstruction of what
    the standup served, and only the standup's records can confirm it);
    under ``lens`` they are optional and verified the same way when given
    (one without the other refuses — the table and the record are read
    together)."""
    standup_path = getattr(args, "verify_standup", None)
    table_path = getattr(args, "verify_routing_table", None)
    if serving_path == SERVING_PATH_LORA_NATIVE:
        missing = [flag for flag, value in (("--verify-standup", standup_path),
                                            ("--verify-routing-table", table_path))
                   if value is None]
        if missing:
            raise DriverError(
                f"--serving-path {SERVING_PATH_LORA_NATIVE} requires "
                f"{' and '.join(missing)}: the native enumeration must be held "
                "to the standup's cells_served.json and routing_table.json "
                "(the authoritative served set) before any leg runs")
    if standup_path is None and table_path is None:
        return
    if (standup_path is None) != (table_path is None):
        raise DriverError(
            "--verify-standup and --verify-routing-table go together (the "
            "record names the run keys, the table names the routes; each is "
            "held to the other) — pass both or neither")
    standup = load_standup_record(Path(standup_path))
    try:
        table = sproxy.load_routing_table(Path(table_path))
    except (sproxy.ProxyError, OSError, ValueError, KeyError, TypeError) as e:
        raise DriverError(f"--verify-routing-table {table_path}: {e}") from e
    requested = requested_cells_of(standup, standup_path)
    bind_declaration_to_attempt(args, standup, standup_path)         # both records agree (idempotent with _frozen_enumeration; run re-binds per leg)
    problem = standup_verification_problem(cell_set, standup, table,
                                           serving_path=serving_path, spec=spec, requested=requested)
    if problem is not None:
        raise DriverError(problem)
    print(f"[petri] standup evidence verified: {len(cell_set.all_served())} "
          f"served leg(s) agree with {standup_path} and {table_path} on "
          f"{list(STANDUP_CELL_FIELDS)} and every route"
          + (f"; audited legs = the proxy's declared cells {requested}" if requested else ""))


# --- subcommands -------------------------------------------------------------------


def cmd_print_cells(args: argparse.Namespace, spec: cm.PetriSpec) -> int:
    serving_path = serving_path_of(args)
    # the drive's stated contract must be the manifest's — held
    # before anything is enumerated or frozen
    check_target_decode_intent(args, spec)
    cell_set, payload, native, requested = _frozen_enumeration(args, spec, serving_path)   # the proxy's declared cells, when any
    # the standup's evidence is held BEFORE the record is frozen: a record
    # the standup disagrees with must never drive an array
    verify_standup_or_refuse(args, spec, cell_set, serving_path)
    if requested is not None:
        print(f"[petri] legs restricted to the proxy's declared cells ({REQUESTED_HARNESS_KEY}): {requested}"
              + ("" if "none" in requested else "; no `none`, no controls twin (the u0 legs are the controls)"))
    served = cell_set.all_served()
    print(f"[petri] audit {spec.audit}: {len(cell_set.canonical)} canonical "
          f"+ {len(cell_set.controls)} control leg(s) = {len(served)} served"
          f"; {len(cell_set.absent)} ABSENT")
    for i, cell in enumerate(served):
        tag = (f" control_of={cell.control_of}" if cell.control_of else "")
        print(f"  [{i:2d}] {cell.cell_id:34s} {cell.cell_class:8s} "
              f"{cell.serving_identity:32s} run_key={cell.petri_run_key}"
              f"{tag}")
    for a in cell_set.absent:
        print(f"  ABSENT {a.cell_id}: {a.reason}")
    if native is not None:
        print(f"[petri] --serving-path {serving_path}: {len(native.skipped)} "
              f"cell(s) skipped by name "
              f"{[s.cell_id for s in native.skipped]}; manifest launch flags "
              f"dropped {list(native.flags_dropped)}; effective flags "
              f"{list(native.effective_spec.serving_launch_flags)}")
    print(f"  sbatch --array=0-{len(served) - 1}")
    if args.out_root:
        path = write_cells_json(cell_set, spec, _out_root(args, spec),
                                payload=payload)
        print(f"[petri] wrote enumeration record {path}")
    return 0


def _already_done(store: cs.CellStore, cell: dc.PetriCell, epochs: int) -> bool:
    """`run`'s already-done path: :func:`leg_already_done` with its print (the leg log says why no GPU work ran)."""
    return leg_already_done(store, cell, epochs, say=True)


def leg_already_done(store: cs.CellStore, cell: dc.PetriCell, epochs: int, *, say: bool = True) -> bool:
    """True when the cell carries a results.json that its store bears out.
    The done marker is reconciled by the SAME routine every collector's
    walk runs (:func:`_reconcile_cell_dir`, through :func:`bind_cell_dirs`):
    the directory is bound to its manifest (cell id + run key -- this leg's
    -- the directory named by the run key, results.json naming its own
    cell and run) and the footing is recomputed from rows.jsonl and the
    refusal sidecar over an identity-bearing SETTLED ledger. Before this
    the already-done path accepted any results.json whose ledger merely
    lacked unresolved entries -- a bare ``{"unscored": 0}`` with no ledger,
    no manifest and no footing passed here and was refused by `consolidate`
    on the very same cell (merge review, 2026-09-07). One routine, one
    verdict."""
    if not store.is_done():
        return False
    b = _bind_cell_dir(store.cell_dir)
    for field, want in (("cell_id", cell.cell_id),
                        ("petri_run_key", cell.petri_run_key)):
        if getattr(b, field) != want:
            raise DriverError(
                f"{store.cell_dir}: manifest.json records {field}="
                f"{getattr(b, field)!r} but this leg is {field}={want!r} -- "
                "the directory is another leg's; refusing to treat its "
                "results.json as this cell's done marker")
    _reconcile_cell_dir(b, epochs)
    if say:
        print(f"[petri] cell {cell.cell_id} already done "
              f"({store.results_path}) — nothing to do")
    return True


LEGS_STATUS_RECORD_MISSING_RC = 2


def legs_status(out_root: Path, spec: cm.PetriSpec) -> Dict[str, Any]:
    """The drive's OWN done-ness reading of every leg of the frozen enumeration at ``out_root`` (petri-completion, 2026-10-13) — what a
    resumed drive would do per leg, read without GPU, proxy, standup evidence or a fresh enumeration: the frozen record through the SAME
    reader ``run`` uses (:func:`read_cells_record`) and the SAME record<->manifest check (:func:`_check_record_matches_spec`; a record
    the resumed drive would refuse is reported as ``record_refused`` and every leg ``refused``), then per leg of ``canonical + controls``
    in record order (the index ``petri_drive_full22.sh`` and ``run --cell-index`` use) :func:`leg_already_done` over the leg's store
    (``done``; a DriverError from the binding / reconcile is ``refused`` with its reason — an identity mismatch is a refusal, never
    "not done"), the unresolved ledger entries and the overflow-classified ones among them (what :func:`_arm_overflow_rerun` requires),
    and whether results.json is present. The funnel's --resubmit proof (reproduce/lib/submit_twin.py) and 06's overflow pass
    (reproduce/lib/petri_overflow.py) consume it; the drive's `ledgered` selector reads it too. Raises DriverError when there is no
    readable frozen record (nothing is provably done)."""
    path = out_root / CELLS_JSON
    if not path.exists():
        raise DriverError(f"no frozen enumeration at {path} -- the resumed drive would re-enumerate; nothing is provably done")
    try:
        record = read_cells_record(out_root)
    except ValueError as e:
        raise DriverError(f"{path}: not a JSON record ({e}) -- nothing is provably done") from e
    doc: Dict[str, Any] = {"out_root": str(out_root), "cells_json": str(path), "audit": spec.audit,
                           "serving_path": record.get("serving_path", SERVING_PATH_LENS), "record_refused": None, "legs": []}
    try:
        _check_record_matches_spec(record, spec, path)
    except DriverError as e:
        doc["record_refused"] = str(e)
    epochs = spec.bloom_options.epochs
    with contextlib.redirect_stdout(sys.stderr):      # the reconcile's own footing lines ([petri_runner] cell …: audited …) go to stderr: stdout stays the ONE JSON document the funnel and 06 parse
        for idx, raw in enumerate(list(record.get("canonical") or []) + list(record.get("controls") or [])):
            doc["legs"].append(_leg_status(out_root, raw, idx, epochs, doc["record_refused"]))
    return doc


def _leg_status(out_root: Path, raw: Mapping[str, Any], idx: int, epochs: int, record_refused: Optional[str]) -> Dict[str, Any]:
    """One leg of :func:`legs_status` (the module docstring there)."""
    cell = dc.PetriCell(**raw)
    store = cs.CellStore(out_root, cell.petri_run_key, recorder=cs.NO_RECORDER)
    leg: Dict[str, Any] = {"idx": idx, "cell_id": cell.cell_id, "control_of": cell.control_of, "kind": cell.kind,
                           "petri_run_key": cell.petri_run_key, "cell_dir": str(store.cell_dir), "done": False, "refused": False,
                           "reason": None, "results_present": store.is_done(), "unresolved": None, "overflow_unresolved": None}
    if record_refused is not None:
        leg.update(refused=True, reason=f"the frozen record is refused: {record_refused}")
    else:
        try:
            leg["done"] = leg_already_done(store, cell, epochs, say=False)
        except DriverError as e:
            leg.update(done=False, refused=True, reason=str(e))
    try:
        unresolved = _unresolved_ledger_entries(store.cell_dir)
        leg["unresolved"] = len(unresolved)
        leg["overflow_unresolved"] = len(overflow_ledger_entries(unresolved))
    except DriverError as e:
        leg.update(refused=True, reason=(leg["reason"] + "; " if leg["reason"] else "") + str(e))
    return leg


def cmd_legs_status(args: argparse.Namespace, spec: cm.PetriSpec) -> int:
    """`legs-status --out-root OUT`: :func:`legs_status` as ONE JSON document on stdout (nothing else is printed there); exit 0 when the
    record was read (the verdicts are inside), :data:`LEGS_STATUS_RECORD_MISSING_RC` with the reason on stderr when there is no readable
    frozen record."""
    out_root = _out_root(args, spec)
    try:
        doc = legs_status(out_root, spec)
    except DriverError as e:
        print(f"REFUSED: legs-status: {e}", file=sys.stderr, flush=True)
        return LEGS_STATUS_RECORD_MISSING_RC
    print(json.dumps(doc, indent=2, sort_keys=True), flush=True)
    return 0


#: the endpoint an offline engine handle points at: unreachable by construction, so a
#: sample that would still need the target errors at once instead of auditing
OFFLINE_BASE_URL = "http://127.0.0.1:9/v1"


def offline_engine_from_models_json(path: Path, cell: Any) -> Any:
    """A ``ProxyClientEngine`` handle built from a SAVED ``/v1/models`` response
    (the grading host drive's ``$STATE/models.json``, captured from the live proxy) with
    NO network call — the proxy-free conversion path (``--offline-models-json``).
    Validates the listing exactly as ``ProxyClientEngine.load()`` does (a non-empty
    ``data`` list of ``petri__`` ids owned by petri, a ``petri_proxy`` mapping) and
    additionally requires the cell's virtual model among the served ids, then sets
    ``served_models`` / ``proxy_identity`` as ``load()`` would. The handle's
    ``base_url`` is :data:`OFFLINE_BASE_URL` (or ``$PETRI_PROXY_BASE_URL`` when
    set), unreachable by construction: nothing is audited here."""
    try:
        listing = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise DriverError(f"--offline-models-json {path}: unreadable: {e}") from e
    data = listing.get("data") if isinstance(listing, Mapping) else None
    if not isinstance(data, list) or not data:
        raise DriverError(f"--offline-models-json {path}: no served models listed")
    ids: List[str] = []
    for m in data:
        mid = m.get("id") if isinstance(m, Mapping) else None
        if not isinstance(mid, str) or not mid.startswith(dc.VMODEL_PREFIX):
            raise DriverError(f"--offline-models-json {path}: {mid!r} is outside the "
                              f"{dc.VMODEL_PREFIX!r} namespace — not a petri proxy listing")
        owned = m.get("owned_by")
        if owned is not None and owned != "petri":
            raise DriverError(f"--offline-models-json {path}: {mid!r} owned by {owned!r}, not petri")
        ids.append(mid)
    meta = listing.get("petri_proxy")
    if not isinstance(meta, Mapping):
        raise DriverError(f"--offline-models-json {path}: no petri_proxy identity block — "
                          "the identity checks have nothing to read; refusing")
    if cell.virtual_model_name not in ids:
        raise DriverError(f"--offline-models-json {path}: the saved listing does not serve "
                          f"{cell.virtual_model_name!r} (served: {sorted(ids)}) — this record "
                          "is not the proxy that served this cell; refusing")
    engine = sproxy.ProxyClientEngine(
        os.environ.get(sproxy.PROXY_BASE_URL_ENV) or OFFLINE_BASE_URL,
        identity=cell.serving_identity)
    engine.proxy_identity = dict(meta)
    engine.served_models = tuple(sorted(ids))
    mtime = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(Path(path).stat().st_mtime))
    print(f"[petri] OFFLINE CONVERSION: engine handle from {path} (captured {mtime}); "
          f"no live proxy — every identity check reads the saved block, nothing is audited, "
          f"a sample still needing generation errors against {engine.base_url}", flush=True)
    return engine


def overflow_rerun_subset_problem(record: Mapping[str, Any], live_payload: Mapping[str, Any], requested: Sequence[str],
                                  path: Any) -> Optional[str]:
    """S2RLFIX6-PETRI: the problem with driving an overflow rerun whose LIVE declaration (``live_payload``, derived from the
    second proxy's cells_served.json) is not a subset of the FROZEN first-pass record, else None. A rerun re-serves some of the legs
    the first pass froze -- the overflowed ones; the completed ones stay closed on the first proxy's results -- so every live leg must
    be a frozen leg with the SAME identity (the whole cell dict: run key, harness hash, intervention hash, serving identity), every
    record key other than the leg lists and the declaration must agree, and the declaration itself must name frozen legs only. Nothing
    is remapped: the caller keeps resolving indices against the frozen record."""
    rec = _record_content(record)
    live = dict(live_payload)
    leg_keys = ("canonical", "controls", REQUESTED_HARNESS_KEY)
    for k in sorted(set(rec) | set(live)):
        if k in leg_keys:
            continue
        if rec.get(k) != live.get(k):
            return (f"{path}: --overflow-rerun: the live enumeration differs from the frozen first-pass record in {k!r} "
                    f"(frozen {rec.get(k)!r}, live {live.get(k)!r}) -- a rerun may only narrow the SET of legs; refusing")
    frozen_by_kind = {kind: {c["cell_id"]: c for c in list(rec.get(kind) or [])} for kind in ("canonical", "controls")}
    live_ids: List[str] = []
    for kind in ("canonical", "controls"):
        for c in list(live.get(kind) or []):
            frozen = frozen_by_kind[kind].get(c["cell_id"])
            if frozen is None:
                return (f"{path}: --overflow-rerun: live {kind} leg {c['cell_id']!r} is not in the frozen first-pass enumeration "
                        f"{sorted(frozen_by_kind[kind])} -- a rerun re-serves a subset of the frozen legs, never a new one; refusing")
            if frozen != c:
                moved = sorted(k for k in set(frozen) | set(c) if frozen.get(k) != c.get(k))
                return (f"{path}: --overflow-rerun: live {kind} leg {c['cell_id']!r} differs from its frozen identity in {moved} -- "
                        "the leg's identity or serving evidence moved since the first pass; refusing to resolve indices against it")
            live_ids.append(c["cell_id"])
    if not live_ids:
        return f"{path}: --overflow-rerun: the live declaration {list(requested)} selects no frozen leg; refusing"
    frozen_canonical = frozen_by_kind["canonical"]
    outside = [r for r in requested if r not in frozen_canonical]
    if outside:
        return (f"{path}: --overflow-rerun: the live declaration names {outside}, not among the frozen first-pass legs "
                f"{sorted(frozen_canonical)}; refusing")
    return None


def _wif_pool_label() -> str:
    """The org the leg's federated inspect clients are routed to (``sisteer.eval.anthropic_wif.inspect_wif_org``), for the typed
    terminal line of a raw SDK rejection; ``unknown`` under the key modes or when nothing is installed."""
    try:
        from sisteer.eval import anthropic_wif
        return anthropic_wif.inspect_wif_org() or "unknown"
    except Exception:  # noqa: BLE001 -- a label for a message, never a refusal
        return "unknown"


def cmd_run(args: argparse.Namespace, spec: cm.PetriSpec) -> int:
    from sisteer.capeval import runtime as rt  # deferred: torch-adjacent
    from sisteer.capeval.tiers import petri_runner as pr  # noqa: E402

    serving_path = serving_path_of(args)
    # the engine every row of this leg is stamped with (through the cell
    # manifest); ``lens`` today, ``lora_native`` on the native standup
    serving_engine = SERVING_ENGINE_BY_PATH[serving_path]
    out_root = _out_root(args, spec)
    # Array indices resolve ONLY from the frozen pre-launch snapshot
    # (review 2026-08-18): a fresh enumeration may serve only to VERIFY
    # exact agreement — any drift refuses, never remaps or overwrites.
    record = read_cells_record(out_root)
    _check_record_matches_spec(record, spec, out_root / CELLS_JSON)
    _check_record_serving_path(record, serving_path, out_root / CELLS_JSON)
    snapshot_served = [dc.PetriCell(**c) for c in
                       list(record.get("canonical") or [])
                       + list(record.get("controls") or [])]
    # re-enumerated under the SAME path the record was frozen under; under --overflow-rerun the frozen record rides along so a baseline
    # view's rerun proxy may declare a SUBSET of it (petri-completion: the completeness check is a freeze's, and a rerun freezes nothing)
    fresh, fresh_payload, native, _requested = _frozen_enumeration(args, spec, serving_path,   # the same restriction the record was frozen under
                                                                   rerun_subset_of=record if getattr(args, "overflow_rerun", False) else None)
    # THE SPEC THE RUNNER GETS: under ``lora_native`` the EFFECTIVE spec the
    # legs were hashed under — the manifest's cells minus the ones the standup
    # skipped by name, ``serving_launch_flags`` minus the tokens the native
    # launch strips (``--enforce-eager``, ``--enable-lora``) — never the
    # manifest spec. ``petri_runner._run`` recomputes the harness hash from the
    # spec it is handed and holds it to ``cell.harness_config_hash`` (H10); the
    # cells were enumerated over the effective spec, so the manifest spec fails
    # that check before the first sample on every native run whose manifest
    # pins a stripped flag (every wave manifest does). Under ``lens`` the
    # manifest spec, exactly as before.
    run_spec = native.effective_spec if native is not None else spec
    if _record_content(record) != fresh_payload:
        if getattr(args, "overflow_rerun", False) and _requested is not None:
            # S2RLFIX6-PETRI: an overflow rerun re-serves a SUBSET of the first pass's frozen
            # enumeration -- the overflowed legs; the completed ones stay closed on the first proxy's results -- so the live
            # declaration is held to the frozen record as a subset, leg by leg (identity + serving evidence), never as equality
            # with the snapshot. The indices stay the record's (``snapshot_served`` above); nothing is remapped or rewritten.
            # (s2r / s1 reruns declared the complete frozen set, so equality held there; s2rl is the first strict-subset rerun.)
            problem = overflow_rerun_subset_problem(record, fresh_payload, _requested, out_root / CELLS_JSON)
            if problem is not None:
                raise DriverError(problem)
            frozen_ids = [c["cell_id"] for c in list(record.get("canonical") or []) + list(record.get("controls") or [])]
            print(f"[petri] --overflow-rerun: the live declaration {list(_requested)} is a subset of the frozen enumeration "
                  f"{frozen_ids} -- indices resolve against the frozen record, the other legs stay as the first pass left them "
                  "(S2RLFIX6-PETRI)", flush=True)
        else:
            raise DriverError(
                f"enumeration drift: re-enumerating {spec.audit!r} no longer "
                f"matches the frozen snapshot {out_root / CELLS_JSON} — the "
                "registry/artifact world changed after launch; refusing to "
                "remap indices or overwrite the record (re-freeze deliberately "
                "with print-cells after deleting the stale record)")
    # the standup's evidence, BEFORE the leg (required under lora_native)
    verify_standup_or_refuse(args, spec, fresh, serving_path)
    # THE TARGET DECODE CONTRACT (module docstring): the flag is held
    # to the manifest, then the run's seed base rides the spec the runner
    # gets (under harness the spec object is unchanged)
    check_target_decode_intent(args, run_spec)
    run_spec = apply_target_seed_base(run_spec, args)
    target_run_identity = pr.target_decode_run_identity(run_spec)
    target_identity, target_seed_base = target_run_identity or (None, None)
    cell = _resolve_cell(snapshot_served, args)
    # The evaluation seed is MANIFEST-owned (it enters harness_config_hash),
    # so `run_petri` refuses a divergent seed. `--seed` is the analysis
    # bootstrap knob (cmd_analyze) and is meaningless here — refuse it loudly
    # rather than silently ignoring it.
    seed = spec.bloom_options.seed
    if args.seed is not None and args.seed != seed:
        raise DriverError(
            f"--seed {args.seed} does not apply to `run`: the evaluation seed "
            f"is manifest-owned (bloom_options.seed={seed}); it enters the "
            "harness hash and run_petri refuses any other value. (--seed is "
            "the bootstrap seed for `analyze`.)")

    # ``row_constants`` stays the EXACT ResultRow column template (store.
    # ROW_COLUMNS is frozen; CellStore.append_ts spreads these into a
    # ResultRow) — the engine rides the manifest top-level, beside ``engine``
    row_constants = _row_constants(spec, cell, seed)
    store = cs.CellStore(out_root, cell.petri_run_key,
                         row_constants=row_constants)
    resolved = {
        "cell_id": cell.cell_id,
        "virtual_model_name": cell.virtual_model_name,
        "control_of": cell.control_of,
        "si_prefix_hash": cell.si_prefix_hash,
        "petri_run_key": cell.petri_run_key,
        "intervention_hash": cell.intervention_hash,
        "harness_config_hash": cell.harness_config_hash,
        "model_revision": spec.target_revision,
        "engine": cell.serving_identity,
        # the served ENGINE (additive, unhashed): ``lens`` |
        # ``lora_native``. ``engine`` above is the serving IDENTITY, which a
        # passthrough leg shares across both engines — this key is what keeps
        # a native-era ``none`` run dir from resuming a lens-era one
        tc.SERVED_ENGINE_FIELD: serving_engine,
        # the TARGET decode contract + the run's seed base:
        # present iff vendor (absent = harness by era; additive, unhashed
        # beyond what the manifest's block already put in the run key)
        **pr.target_decode_stamp(target_run_identity),
        "row_constants": row_constants,
    }
    # a Petri store is single-engine: a dir whose manifest names (or, by
    # era, implies) the other engine refuses HERE — before validate_resume
    # reads it and before the manifest write below (the runner re-checks
    # from the live proxy identity before its first sample)
    pr.refuse_cross_engine_resume(store.cell_dir, serving_engine)
    # ... and single-contract / single-seed-base: the runner
    # re-checks before its first sample too
    pr.refuse_cross_target_decode_resume(store.cell_dir, target_identity,
                                         target_seed_base)
    store.validate_resume(resolved)  # refuse-on-mismatch before anything
    store.write_cell_manifest(resolved)

    armed_overflow: Optional[List[Any]] = None
    pre_rerolled = 0
    if getattr(args, "overflow_rerun", False):
        from sisteer.capeval.tiers import petri_runner as _pr
        # the proxy's ATTESTED window (the lens path on b5a920fdd): read off the native standup's cells_served.json
        # (--verify-standup) and/or the pulled endpoint.json (--served-endpoint, the lens drive's only record; both must agree); the
        # runner keys the pass's resume namespace by it and refuses when it disagrees with the requested re-run window -- an armed pass
        # without an attestation refuses here, never guesses
        served = served_window_from_evidence(getattr(args, "verify_standup", None), getattr(args, "served_endpoint", None))
        os.environ[_pr.PETRI_SERVED_MAX_MODEL_LEN_ENV] = str(served)
        armed_overflow = _arm_overflow_rerun(store.cell_dir, cell.cell_id, spec)
        pre_rerolled = len(_pr.read_judge_rerolled(store))     # the re-roll stamps before this pass (progress = a new one)
        pre_refused = len(_pr.read_auditor_refused(store))     # the auditor-terminal records before this pass (growth at
                                                               # zero rows is the zero-resolution floor, re-read here)

    if _already_done(store, cell, spec.bloom_options.epochs):
        return 0

    # the Phase-2 serving engine, through runtime's frozen dispatch seam
    # (serving/proxy.py — audit cells are always proxy_vllm-served by
    # construction, B1; the HF shim is gate-reference tooling only)
    offline = getattr(args, "offline_models_json", None)
    if offline is not None:
        engine = offline_engine_from_models_json(Path(offline), cell)
    else:
        engine = rt.make_engine(SimpleNamespace(
            engine=cell.serving_identity, base=spec.target_model,
            base_revision=spec.target_revision))
    if serving_path == SERVING_PATH_LORA_NATIVE:
        # the LIVE proxy is the native engine — held from its /v1/models
        # identity block (captured at load) BEFORE the first sample; under
        # lens no new live check (today's behaviour byte for byte: the
        # runner's lens-conditional agent-loop check stands)
        problem = proxy_engine.served_engine_problem(
            getattr(engine, "proxy_identity", None),
            expect=proxy_engine.ENGINE_EXPECT_LORA_NATIVE)
        if problem is not None:
            raise DriverError(
                f"--serving-path {SERVING_PATH_LORA_NATIVE}: the live proxy is "
                f"not the native multi-LoRA engine — {problem}; refusing "
                "before the first sample (the standup's evidence verified "
                "above describes a server this proxy is not)")
    # a credential the API rejected -- the runner's typed stop out of the federated inspect
    # clients, or the SDK's own 401 / 403 -- ends this leg with the typed text as its TERMINAL line (the drive's leg log, the
    # grading daemon's `credential` kind), never a traceback and never a results.json over it; every other failure passes as before
    from sisteer import vault as _vault
    with _vault.credential_exit(pool=_wif_pool_label, prefix="[petri] "):
        summary = rt.run_petri(engine, run_spec, cell, store, seed)

    # a pass that met REJECTED requests (the runner's hard errors) or withheld terminal records under the
    # zero-resolution floor is refused here by name, BEFORE any completion is considered — the runner recorded the
    # evidence beside the ledger; this is the driver's verdict on it (rc 2, the drive's FAILED surface)
    _refuse_rejected_pass(store.cell_dir, cell.cell_id)
    unscored = int(summary.get("unscored", 0))
    if armed_overflow is not None:
        # the SAME rule the runner evaluated before persisting anything (petri_runner.zero_resolution_floor), re-read from
        # the written evidence: new auditor-terminal records at zero rows — a runner that filed them anyway is refused here
        new_refused = len(pr.read_auditor_refused(store)) - pre_refused
        if pr.zero_resolution_floor(armed=True, rows_emitted=int(summary.get("rows_emitted", 0) or 0),
                                    new_auditor_terminal=new_refused):
            raise DriverError(
                f"--overflow-rerun: cell {cell.cell_id!r} filed 0 rows and {new_refused} NEW auditor-terminal record(s) — "
                "an armed pass that scored nothing resolved none of those casualties (the 2026-10-01 false completion: 41 "
                "rejected requests filed as sheds); the runner should have withheld them (petri_runner.zero_resolution_floor). "
                "Refusing before any completion; the cell stays as the ledger says (D696)")
    if unscored == 0:
        # never write the done marker over an unresolved ledger — the
        # ledger, not the runner's summary, is authoritative (H9; review
        # 2026-08-18: results.json must not hide unresolved entries)
        unresolved = _unresolved_ledger_entries(store.cell_dir)
        if unresolved:
            raise DriverError(
                f"cell {cell.cell_id}: run_petri reports zero unscored but "
                f"the failure ledger at {store.cell_dir / FAILURE_LEDGER} "
                f"still carries {len(unresolved)} unresolved "
                f"entr{'y' if len(unresolved) == 1 else 'ies'} — refusing "
                "to write results.json over an unresolved ledger (H9)")
        if not store.is_done():
            store.write_results(dict(summary))
        # the audited FOOTING beside the completion, by default (petri_runner
        # module docstring: auditor sheds are a first-class outcome; cells
        # are compared on their audited footing, and a refusal is not harm)
        print(f"[petri] cell {cell.cell_id} COMPLETE "
              f"({summary.get('rows_emitted', '?')} rows; "
              f"{pr.audited_footing_line(summary)})")
        return 0
    # An armed overflow re-run that resolved NOTHING must refuse here —
    # rc=3 with an unchanged casualty count reads like progress while the
    # actual state is an infinite no-op loop.
    if armed_overflow is not None:
        _assert_overflow_rerun_progress(store.cell_dir, cell.cell_id,
                                        armed_overflow, pre_rerolled=pre_rerolled)
    # H9: an unresolved required dimension leaves the cell INCOMPLETE —
    # results.json is NOT written; the failure ledger says why.
    print(f"[petri] cell {cell.cell_id} INCOMPLETE: {unscored} unresolved "
          f"ledger entr{'y' if unscored == 1 else 'ies'} "
          f"(see {store.cell_dir / FAILURE_LEDGER}) — no results.json "
          "written; the cell never consolidates until resolved; "
          f"{pr.audited_footing_line(summary)}")
    return EXIT_INCOMPLETE


def cmd_smoke(args: argparse.Namespace, spec: cm.PetriSpec) -> int:
    raise DriverError(
        "the real agent smoke is a LAUNCH-phase action (plan §10 H11): it "
        "needs the live proxy job, the SSH tunnel, and the coordination "
        "post already out (launch order §11 steps 1-3), then runs ONE real "
        "agent-modality scenario end-to-end and passes only if (1) the "
        "target emits a well-formed tool call, (2) the auditor's simulated "
        "tool response is consumed, (3) the assistant turn resumes, and "
        "(4) the judge output parses. Run it via petri_audit.sh --smoke at "
        "launch; it is deliberately not runnable from the CPU build phase")


def _unresolved_ledger_entries(cell_dir: Path) -> List[Any]:
    """The cell's unresolved failure-ledger entries. The CANONICAL ledger
    shape is the runner's: a dict whose ``"unresolved"`` key lists the
    open (behavior, scenario, epoch, dimension) records — anything else is
    a malformed ledger and refuses loudly (never a silent zero)."""
    ledger_path = cell_dir / FAILURE_LEDGER
    if not ledger_path.exists():
        return []
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(ledger, Mapping) or not isinstance(
            ledger.get("unresolved"), list):
        raise DriverError(
            f"{ledger_path}: malformed failure ledger — the canonical "
            "shape is a mapping with an 'unresolved' list "
            "(petri_runner.write_failure_ledger); refusing to guess")
    return list(ledger["unresolved"])


def _ledger_payload(cell_dir: Path) -> Dict[str, Any]:
    """The whole failure-ledger payload ({} when none), its canonical shape enforced."""
    _unresolved_ledger_entries(cell_dir)   # loud on a malformed shape
    ledger_path = cell_dir / FAILURE_LEDGER
    if not ledger_path.exists():
        return {}
    return dict(json.loads(ledger_path.read_text(encoding="utf-8")))


def _refuse_rejected_pass(cell_dir: Path, cell_id: str) -> None:
    """Refuse, by name, a pass whose ledger records REJECTED requests
    (``hard_errors``: the API answered the request itself with a non-retryable
    error — a 400 invalid_request_error, a 401/403, a 404 — so no ladder ran
    and nothing was audited) or terminal records WITHHELD under the armed
    pass's zero-resolution floor (``terminal_withheld``). Either is a
    pass-level failure to fix and re-drive, never progress and never a
    replay: the casualties stand byte-identical in ``unresolved`` and the
    diagnosis names the HTTP error so the next action is a fix to the
    REQUEST (the model, the beta header, the extra body), not a quarantine
    of resume dirs and not a geometry change."""
    from sisteer.capeval.tiers import petri_runner as pr
    payload = _ledger_payload(cell_dir)
    hard = payload.get(pr.HARD_ERRORS_KEY) or []
    if hard:
        labels: Dict[str, int] = {}
        for rec in hard:
            he = rec.get("hard_error") or {}
            label = (f"HTTP {he.get('status')} {he.get('error_type') or ''}".strip()
                     if he.get("status") is not None else "no status, no transport evidence")
            labels[label] = labels.get(label, 0) + 1
        first = (hard[0].get("hard_error") or {})
        raise DriverError(
            f"cell {cell_id!r}: {len(hard)} relaunched scenario(s) were REJECTED by the API "
            f"({json.dumps(labels, sort_keys=True)}; e.g. {first.get('model')} as {first.get('role')}: "
            f"{str(first.get('error', ''))[:300]}) — a HARD error about the REQUEST, not an auditor shed and not a "
            f"replay: no ladder ran, nothing was audited, nothing is terminal. The casualties stand unchanged in "
            f"{cell_dir / FAILURE_LEDGER} ('unresolved'; the rejections under '{pr.HARD_ERRORS_KEY}'). Fix the request "
            "(the model, the beta header, the extra body) and re-drive; not a geometry problem (D696)")
    withheld = payload.get(pr.TERMINAL_WITHHELD_KEY) or {}
    if withheld:
        n_a = len(withheld.get("auditor_refused") or [])
        raise DriverError(
            f"--overflow-rerun: cell {cell_id!r} resolved NOTHING — the pass filed 0 rows, judged no finished transcript, "
            f"and its only new outcomes were {n_a} auditor shed(s); the runner WITHHELD those records "
            f"('{pr.TERMINAL_WITHHELD_KEY}' in {cell_dir / FAILURE_LEDGER}) and the casualties stand unchanged. A relaunch "
            "of scenarios that had full transcripts at the smaller window ending in nothing but auditor sheds is a failure "
            "of the pass (the auditor, the request), not of the scenarios: read the withheld evidence, fix the pass, "
            "re-drive (D696)")


def overflow_ledger_entries(entries: Sequence[Any]) -> List[Any]:
    """The subset of unresolved ledger entries the converter classified
    as context-overflow casualties (reason prefix
    ``petri_runner.OVERFLOW_REASON_PREFIX`` — the machine-checkable
    contract between the converter and this driver)."""
    from sisteer.capeval.tiers import petri_runner as pr

    return [e for e in entries
            if isinstance(e, Mapping)
            and str(e.get("reason", "")).startswith(pr.OVERFLOW_REASON_PREFIX)]


#: the re-run pass's RELAUNCH RECORD per cell: the window and factor served, the overflowed scenario ids and the
#: judge-refused ids re-rolled under the declared judge refusal fallback, per behaviour -- what the pass relaunched
#: and why, read into the results footing (`rerun_pass`) and the registry judgment's notes so it reaches the artifact as the
#: designed overflow pass, never a contract change
RERUN_RELAUNCH_RECORD = "petri_rerun_relaunch.json"   # == petri_runner.RERUN_RELAUNCH_RECORD (asserted at arming); the runner restricts
                                                       # the pass's launch to this record's union and writes back `launched` / `left_unresolved`


def _rerollable_refused(cell_dir: Path, spec: Any) -> List[Dict[str, Any]]:
    """The judge-refused sidecar records the pass will re-roll (petri_runner.rerollable_judge_refusals: the default judge's,
    never re-rolled before, under a declared judge refusal fallback), as dicts; [] without a sidecar or a fallback."""
    from types import SimpleNamespace
    from sisteer.capeval.tiers import petri_runner as pr
    store = SimpleNamespace(cell_dir=Path(cell_dir))      # the sidecar readers use the store's cell_dir only
    if not pr.judge_refused_path(store).is_file():
        return []
    _standing, rerolled = pr.rerollable_judge_refusals(pr.read_judge_refused(store), spec, pr.settled_ledger(store))
    return [{"behavior": r.behavior, "scenario_id": r.scenario_id, "epoch": r.epoch, "judge_model": r.judge_model} for r in rerolled]


def served_window_from_standup(standup_path: Any) -> int:
    """The window the proxy ATTESTS (`attested_max_model_len` in the standup's final cells_served.json, which the drive pulls beside the
    endpoint and hands in as --verify-standup); the armed pass refuses without it."""
    if not standup_path:
        raise DriverError("--overflow-rerun needs --verify-standup <cells_served.json>: the pass keys its resume namespace by the window "
                          "the proxy ATTESTS, never by the locally requested factor")
    p = Path(standup_path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise DriverError(f"--overflow-rerun: cannot read the standup evidence at {p}: {e}") from None
    att = doc.get("attested_max_model_len") if isinstance(doc, Mapping) else None
    if not isinstance(att, int) or isinstance(att, bool) or att < 16:
        raise DriverError(f"--overflow-rerun: {p} carries no attested_max_model_len (got {att!r}) — the workers did not attest the "
                          "served window; refusing to arm the pass against an unattested geometry")
    return att


def served_window_from_endpoint(endpoint_path: Any) -> int:
    """The window the proxy ATTESTS as the drive's pulled endpoint.json carries it (`attested_max_model_len`, written by the standup for
    either engine); the lens serving path has no cells_served.json, so this is the armed pass's attestation there."""
    if not endpoint_path:
        raise DriverError("--overflow-rerun needs the proxy's attestation: --verify-standup <cells_served.json> (the native standup's record) "
                          "or --served-endpoint <endpoint.json> (the pulled endpoint record, either engine); the pass keys its resume "
                          "namespace by the window the proxy ATTESTS, never by the locally requested factor")
    p = Path(endpoint_path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise DriverError(f"--overflow-rerun: cannot read the endpoint record at {p}: {e}") from None
    att = doc.get("attested_max_model_len") if isinstance(doc, Mapping) else None
    if not isinstance(att, int) or isinstance(att, bool) or att < 16:
        raise DriverError(f"--overflow-rerun: {p} carries no attested_max_model_len (got {att!r}) — the standup did not attest the served "
                          "window on this record; refusing to arm the pass against an unattested geometry")
    return att


def served_window_from_evidence(standup_path: Any, endpoint_path: Any) -> int:
    """The attested window an armed pass is keyed by, from whichever evidence the drive handed in: the native standup's cells_served.json
    (--verify-standup), the pulled endpoint.json (--served-endpoint; the lens path's only record), or both — which must agree (the lens
    drive passes no --verify-standup, so an armed lens pass refused before sampling). Neither refuses by name."""
    if not standup_path and not endpoint_path:
        return served_window_from_endpoint(None)                      # the refusal naming both flags
    windows = {}
    if standup_path:
        windows["--verify-standup"] = served_window_from_standup(standup_path)
    if endpoint_path:
        windows["--served-endpoint"] = served_window_from_endpoint(endpoint_path)
    if len(set(windows.values())) > 1:
        raise DriverError("--overflow-rerun: the two attestations disagree — "
                          + ", ".join(f"{k} attests max_model_len={v}" for k, v in windows.items())
                          + " — the records are not the same standup's; refusing")
    return next(iter(windows.values()))


def _arm_overflow_rerun(cell_dir: Path, cell_id: str, spec: Any = None) -> List[Any]:
    """Arm the context-overflow re-run pass for one cell: refuse unless
    this cell's failure ledger actually holds overflow-classified
    unresolved entries (the flag asserts a targeted re-run, never a
    fresh audit in disguise), then set ``PETRI_OVERFLOW_RERUN`` so the
    runner caps its eval width to the re-run serving geometry, and say
    loudly what the serving stack must have been relaunched at. WHICH
    scenarios re-run is not this flag's business — the resume machinery
    relaunches the ledger's unresolved scenarios, PROVIDED nothing
    replays them: a caught-and-converted overflow casualty COMPLETED at
    the inspect level (the judge scored the truncated transcript; only
    the converter withholds the rows), so a resume sub-namespace log dir
    already holding those completed samples makes ``eval_set`` return
    them verbatim instead of relaunching. ``cmd_run`` therefore verifies
    AFTERWARDS that the pass resolved something
    (:func:`_assert_overflow_rerun_progress`) — the armed casualty
    entries are returned for exactly that comparison."""
    from sisteer.capeval.tiers import petri_runner as pr

    unresolved = _unresolved_ledger_entries(cell_dir)
    casualties = overflow_ledger_entries(unresolved)
    if not casualties:
        raise DriverError(
            f"--overflow-rerun: cell {cell_id!r} has no unresolved "
            f"context-overflow ledger entries under {cell_dir} "
            f"({len(unresolved)} unresolved total) — nothing to re-run at "
            "a larger window; run without the flag (a plain resume) or "
            "pick the right cell")
    geo = pr.rerun_serving_geometry()
    scenarios = sorted({e.get("scenario_id") for e in casualties})
    # the judge-refused items re-rolled under the declared fallback join the relaunch set, marked REFUSED (not
    # overflowed); the record names both sets per behaviour with the window served, so the pass's selection effect is explicit
    refused = _rerollable_refused(cell_dir, spec) if spec is not None else []
    record = {
        "version": 1, "cell_id": cell_id, "armed_at": time.time(),
        "window": geo["max_model_len"], "max_num_seqs": geo["max_num_seqs"], "scale": pr.rerun_window_scale(),
        "overflowed": sorted({(str(e.get("behavior")), str(e.get("scenario_id"))) for e in casualties}),
        "refused": sorted({(str(r["behavior"]), str(r["scenario_id"])) for r in refused}),
        "judge_refusal_fallback_models": list(getattr(spec, "judge_refusal_fallback_models", []) or []),
        "auditor_refusal_fallback_models": list(getattr(spec, "auditor_refusal_fallback_models", []) or []),
    }
    record["overflowed"] = [list(x) for x in record["overflowed"]]; record["refused"] = [list(x) for x in record["refused"]]
    assert RERUN_RELAUNCH_RECORD == pr.RERUN_RELAUNCH_RECORD, "the relaunch record's name must be the runner's"
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / RERUN_RELAUNCH_RECORD).write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    if refused:
        print(f"[petri] overflow re-run for cell {cell_id} also RE-ROLLS {len(refused)} judge-refused item(s) under "
              f"judge_refusal_fallback_models={record['judge_refusal_fallback_models']} (D475(4)): "
              + ", ".join(f"{r['behavior']}/{r['scenario_id']}" for r in refused))
    print(f"[petri] overflow re-run armed for cell {cell_id}: "
          f"{len(casualties)} unresolved overflow entr"
          f"{'y' if len(casualties) == 1 else 'ies'} across scenarios "
          f"{scenarios}. The serving stack MUST be up at "
          f"max_model_len={geo['max_model_len']} "
          f"max_num_seqs={geo['max_num_seqs']} "
          "(petri_runner.rerun_serving_geometry — same token product and "
          "KV ceiling as the certified standard geometry); eval width is "
          "capped to match. Scenarios re-run FROM TURN 1 at the same "
          "1/16-of-default-window response cap: petri_bloom cannot seed "
          "a partial transcript, and nothing here depends on exact "
          "replay — recorded turns are never regenerated.")
    os.environ[pr.PETRI_OVERFLOW_RERUN_ENV] = "1"
    return casualties


def _ledger_key(entry: Mapping[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """The identity of one failure-ledger record — the LedgerEntry key
    (behavior, scenario_id, epoch, dimension) in its JSON form."""
    return (entry.get("behavior"), entry.get("scenario_id"),
            entry.get("epoch"), entry.get("dimension"))


def _assert_overflow_rerun_progress(cell_dir: Path, cell_id: str,
                                    armed: Sequence[Any], *, pre_rerolled: int = 0) -> None:
    """Refuse when an armed ``--overflow-rerun`` pass resolved NONE of
    its casualties: returning rc=3 with an
    unchanged count reads like ordinary INCOMPLETE progress, while the
    state it hides is an infinite no-op loop that no number of further
    passes can leave.

    The loop: the overflow protocol relies on resume to relaunch the
    casualties, but a casualty's sample COMPLETED at the inspect level
    (petri surfaces the 400 to the auditor, the sample finishes, the
    judge scores the truncated transcript — only our converter withholds
    the rows), so once a resume sub-namespace log dir
    (``resume-<sha16(scenario subset)>`` under the cell's inspect log
    dir) holds those completed samples, ``eval_set`` returns them
    verbatim on every later pass — seconds of wall clock, zero fresh
    target calls — and the converter re-withholds the identical
    casualties. Observed live on the gemma4e4b audit (2026-09-01): three
    passes, 27 -> 27 -> 27, byte-identical model-usage totals.

    Zero-progress has two distinguishable causes, told apart by the
    surviving evidence:

    - every armed casualty survives with a BYTE-IDENTICAL reason ->
      nothing re-executed (the replay above). Remedy: quarantine — move
      aside, never delete — the stale ``resume-*`` dirs whose completed
      samples predate the wide-window serving stack, then re-run.
    - reasons CHANGED -> the scenarios genuinely re-ran and STILL
      overflowed at the re-run window; another pass at this geometry
      cannot converge.
    """
    # The rejected-pass check comes FIRST: a pass whose relaunched requests were REJECTED (or whose terminal records were
    # withheld under the zero-resolution floor) leaves every casualty byte-identical by design — the replay diagnosis below
    # would send the operator to quarantine resume dirs for a request error
    _refuse_rejected_pass(cell_dir, cell_id)
    after = {_ledger_key(e): str(e.get("reason", ""))
             for e in overflow_ledger_entries(
                 _unresolved_ledger_entries(cell_dir))}
    armed_map = {_ledger_key(e): str(e.get("reason", ""))
                 for e in armed if isinstance(e, Mapping)}
    resolved = set(armed_map) - set(after)
    if resolved:
        return  # real progress — the ordinary INCOMPLETE path stands
    # a re-rolled judge-refused item counts as progress ONLY when the runner stamped it from post-launch evidence (the
    # sidecar's `rerolled` list grew: the item was scored under the fallback or re-filed as a terminal refusal) -- the pre-run
    # record's `refused` list alone proves nothing (a replayed resume leaves every item unchanged)
    from types import SimpleNamespace
    from sisteer.capeval.tiers import petri_runner as pr
    post_rerolled = len(pr.read_judge_rerolled(SimpleNamespace(cell_dir=cell_dir)))
    if post_rerolled > int(pre_rerolled):
        return
    if all(after.get(k) == reason for k, reason in armed_map.items()):
        raise DriverError(
            f"--overflow-rerun: cell {cell_id!r} resolved NONE of its "
            f"{len(armed_map)} armed overflow casualties and every "
            "surviving ledger reason is byte-identical to the pre-run "
            "ledger — nothing re-executed. The casualties' samples "
            "completed at the inspect level, so eval_set replays them "
            "from the resume sub-namespace log dirs (resume-<sha16> "
            "under this cell's inspect log dir) instead of relaunching; "
            "every further pass is the same no-op. Quarantine (move "
            "aside — never delete) the stale resume-* dirs holding "
            "completed samples from the previous serving window, then "
            "re-run --overflow-rerun against the wide-window stack.")
    raise DriverError(
        f"--overflow-rerun: cell {cell_id!r} resolved NONE of its "
        f"{len(armed_map)} armed overflow casualties, but the recorded "
        "evidence changed — the scenarios genuinely re-ran and STILL "
        "overflowed at the re-run window (rerun_serving_geometry()); "
        "another pass at this geometry cannot converge. The window (or "
        "the audit's turn budget) needs a rethink, not a retry.")


def _served_run_keys(record: Mapping[str, Any]) -> Dict[str, str]:
    """cell_id -> petri_run_key for every SERVED cell (canonical + control)
    in the frozen enumeration record — the ONE run each cell may have."""
    keys: Dict[str, str] = {}
    for c in (list(record.get("canonical") or [])
              + list(record.get("controls") or [])):
        keys[str(c["cell_id"])] = str(c["petri_run_key"])
    return keys


@dataclasses.dataclass(frozen=True)
class BoundCellDir:
    """One ``cells/<run_key>/`` directory bound to its cell, its run and
    the frozen enumeration (:func:`bind_cell_dirs`)."""
    cell_id: str
    petri_run_key: str
    path: Path
    manifest: Dict[str, Any]
    results: Optional[Dict[str, Any]]  # None: no done marker (incomplete)


def _exact_int(mapping: Any, key: str, where: str) -> int:
    """One footing count from a results.json — present and an EXACT int
    (ordinary ``==`` lets ``0.0`` and ``False`` stand for ``0``)."""
    if not isinstance(mapping, Mapping) or key not in mapping:
        raise DriverError(f"{where}: {key!r} is missing — a footing count is "
                          "never defaulted")
    value = mapping[key]
    if type(value) is not int:
        raise DriverError(
            f"{where}: {key!r} = {value!r} is not an exact integer (bool, "
            "float, string and null are refused — a count is never coerced)")
    return value


def _exact_int_tally(mapping: Any, keys: Sequence[str],
                     where: str) -> Dict[str, int]:
    """A tally with EXACTLY ``keys``, every value an exact int."""
    if not isinstance(mapping, Mapping):
        raise DriverError(f"{where}: not a mapping")
    names, expected = set(map(str, mapping)), set(keys)
    if names != expected:
        raise DriverError(f"{where}: buckets {sorted(names)} are not the "
                          f"recognized {sorted(expected)}")
    return {k: _exact_int(mapping, k, where) for k in keys}


def _bound_ledger(b: BoundCellDir) -> Optional[Dict[str, Any]]:
    """The cell's failure ledger, REQUIRED to name this cell and run — or
    None when no ledger file exists. Checking the identity only when the
    key was present let an identity-free
    ``{"version": 1, "unresolved": []}`` satisfy the settled-ledger
    requirement, so the canonical shape (a mapping with an
    ``unresolved`` list) is enforced first; then ``cell_id`` and
    ``petri_run_key`` must both be recorded and equal the bound cell's —
    petri_runner.write_failure_ledger always writes them, so a ledger
    without them is not the producer's."""
    ledger_path = b.path / FAILURE_LEDGER
    if not ledger_path.exists():
        return None
    _unresolved_ledger_entries(b.path)  # loud on a malformed shape
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    for key, expected in (("cell_id", b.cell_id),
                          ("petri_run_key", b.petri_run_key)):
        if key not in ledger:
            raise DriverError(
                f"{ledger_path}: no {key} recorded — a failure ledger names "
                "its cell and run (petri_runner.write_failure_ledger); an "
                "identity-free ledger is no evidence for any cell")
        if str(ledger[key]) != expected:
            raise DriverError(
                f"{ledger_path}: {key} {ledger[key]!r} != {expected!r} — the "
                "ledger is another cell's or another run's")
    return dict(ledger)


def _reconcile_done_marker(b: BoundCellDir, epochs: int) -> None:
    """A results.json is a done marker ONLY if the cell store bears it out
    (a directory holding just a manifest and a fully-refused results.json
    — no rows, no refusal sidecar, no ledger — was COMPLETE with score
    None; matching run-key strings prove agreement between files, not
    that the cell ran).

    Evidence, in the order the runner leaves it: the failure ledger must
    exist and be settled (``unresolved: []``); the results must carry the
    footing (``audited`` / ``auditor_refused`` / ``behaviors``); the refusal
    sidecar, when present, names this cell and run; and per behavior the
    footing is RECOMPUTED from the store — ``audited`` from rows.jsonl (items
    with every required dimension filed, refused items excluded),
    ``auditor_refused`` counts, kinds and identities from the sidecar,
    ``judge_refused`` count and identities from ``petri_judge_refused.json``
    (the judge twin: a FINISHED transcript the pinned judge declined to
    score — petri_runner's judge-refusals comment), ``n_items`` as the sum
    of all three — and must match exactly. A results.json WITHOUT the
    ``judge_refused`` key (the runner writes it iff the cell has a judge
    refusal; cells filed before the key existed never carry it) reconciles
    ONLY with an absent or empty judge sidecar — absence means zero, and a
    filed refusal the results do not report would be a silent N/N. Rows or
    refusals for a behavior the
    results do not name, or partially filed items, are a contradiction for
    a complete cell. Every mismatch is a refusal."""
    from sisteer.capeval.tiers import petri_runner as pr  # deferred, heavy
    assert b.results is not None
    where = f"{b.path / 'results.json'}"
    ledger = _bound_ledger(b)  # names this cell and run, or refuses
    if ledger is None:
        raise DriverError(
            f"{where}: no {FAILURE_LEDGER} beside the done marker — a "
            "complete cell records its settled ledger (unresolved: []); a "
            "results.json with no supporting evidence is no done marker")
    unresolved = list(ledger["unresolved"])
    if unresolved:
        raise DriverError(
            f"{b.path}: results.json (the done marker) coexists with "
            f"{len(unresolved)} unresolved failure-ledger "
            f"entr{'y' if len(unresolved) == 1 else 'ies'} — the ledger is "
            "authoritative (H9); integrity refusal, never a silent "
            "consolidation")
    for key in ("audited", "auditor_refused", "behaviors"):
        if not isinstance(b.results.get(key), Mapping):
            raise DriverError(
                f"{where}: no {key!r} footing — a results.json without the "
                "audited footing and its per-behavior block cannot be "
                "reconciled with the store and is no done marker (no "
                "versioned legacy shape exists)")
    store = cs.CellStore(b.path.parent.parent, b.path.name)
    sidecar_path = pr.auditor_refused_path(store)
    if sidecar_path.is_file():
        raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
        for key, expected in (("cell_id", b.cell_id),
                              ("petri_run_key", b.petri_run_key)):
            if not isinstance(raw, Mapping) or str(raw.get(key)) != expected:
                raise DriverError(
                    f"{sidecar_path}: {key} does not name this cell and run "
                    f"({expected!r}) — the refusal sidecar is another run's")
    try:
        refusals = pr.read_auditor_refused(store)
    except pr.PetriRunnerError as e:
        raise DriverError(str(e)) from e
    refused_by_b: Dict[str, list] = {}
    for r in refusals:
        refused_by_b.setdefault(r.behavior, []).append(r)
    tallies = pr.auditor_refused_by_behavior(refusals)
    # the judge twin: petri_judge_refused.json names this cell and run, or
    # it is another run's; a malformed sidecar refuses (never a silent zero)
    judge_sidecar_path = pr.judge_refused_path(store)
    if judge_sidecar_path.is_file():
        raw = json.loads(judge_sidecar_path.read_text(encoding="utf-8"))
        # ours, or the candidate-si-refit lineage's (same filename, another
        # record shape)? Named as such FIRST — a file another implementation
        # wrote correctly is never "malformed" here
        try:
            pr.assert_judge_refused_schema(
                raw.get("schema") if isinstance(raw, Mapping) else None,
                judge_sidecar_path)
        except pr.PetriRunnerError as e:
            raise DriverError(str(e)) from e
        for key, expected in (("cell_id", b.cell_id),
                              ("petri_run_key", b.petri_run_key)):
            if not isinstance(raw, Mapping) or str(raw.get(key)) != expected:
                raise DriverError(
                    f"{judge_sidecar_path}: {key} does not name this cell "
                    f"and run ({expected!r}) — the judge-refusal sidecar is "
                    "another run's")
    try:
        judge_refusals = pr.read_judge_refused(store)
    except pr.PetriRunnerError as e:
        raise DriverError(str(e)) from e
    judge_by_b: Dict[str, list] = {}
    for r in judge_refusals:
        judge_by_b.setdefault(r.behavior, []).append(r)
    rows = store._parsed_rows()
    row_behaviors = {str(r.get("task"))[len(pr.PETRI_TASK_PREFIX):]
                     for r in rows
                     if str(r.get("task", "")).startswith(pr.PETRI_TASK_PREFIX)}
    declared = b.results["behaviors"]
    evidence_only = sorted((row_behaviors | set(refused_by_b) | set(judge_by_b))
                           - set(declared))
    if evidence_only:
        raise DriverError(
            f"{where}: the store holds rows or refusals for behavior(s) "
            f"{evidence_only} the results do not name — the done marker "
            "does not describe this store")
    evidence_items = evidence_expected = 0
    cell_chain = pr._zero_chain_tally()  # this lane: summed per behavior
    for behavior in sorted(declared):
        entry = declared[behavior]
        bwhere = f"{where} behavior {behavior!r}"
        if not isinstance(entry, Mapping):
            raise DriverError(f"{bwhere}: per-behavior footing is not a mapping")
        refused_here = refused_by_b.get(behavior, [])
        refused_ids = {pr._item_id(r.scenario_id, r.epoch, epochs)
                       for r in refused_here}
        judge_here = judge_by_b.get(behavior, [])
        judge_ids = {pr._item_id(r.scenario_id, r.epoch, epochs)
                     for r in judge_here}
        # rows of a terminally refused item (either kind) are tombstoned:
        # they never count toward `audited`
        excluded_ids = refused_ids | judge_ids
        required = set(pr.required_metrics(behavior))
        task = pr.PETRI_TASK_PREFIX + behavior
        filed: Dict[str, set] = {}
        chain_by_item: Dict[str, Any] = {}
        for r in rows:
            item = str(r.get("item_id"))
            if r.get("task") == task and r.get("metric") in required \
                    and item != pr.AGGREGATE_ITEM_ID and item not in excluded_ids:
                filed.setdefault(item, set()).add(str(r["metric"]))
                # this lane: the item's auditor-chain provenance rides every
                # score dimension's extra_json and must agree across them
                try:
                    extra = json.loads(r.get("extra_json") or "{}")
                except (TypeError, ValueError) as e:
                    raise DriverError(f"{bwhere}: item {item!r} carries "
                                      f"malformed extra_json ({e})") from e
                chain_value = (extra.get("auditor_chain")
                               if isinstance(extra, Mapping) else None)
                if item in chain_by_item and chain_by_item[item] != chain_value:
                    raise DriverError(
                        f"{bwhere}: item {item!r} has inconsistent "
                        "auditor_chain provenance across its score dimensions")
                chain_by_item[item] = chain_value
        partial = sorted(i for i, ms in filed.items() if not required <= ms)
        if partial:
            raise DriverError(
                f"{bwhere}: item(s) {partial} have partially filed "
                "dimensions in rows.jsonl yet the ledger is settled — a "
                "complete cell has no such item; refusing the done marker")
        audited_from_rows = len(filed)
        refused_from_sidecar = len(refused_here)
        judge_from_sidecar = len(judge_here)
        evidence_items += audited_from_rows
        evidence_expected += (audited_from_rows + refused_from_sidecar
                              + judge_from_sidecar)
        recorded_audited = _exact_int(entry, "audited", bwhere)
        if recorded_audited != audited_from_rows:
            raise DriverError(
                f"{bwhere}: results record audited={recorded_audited!r} but "
                f"rows.jsonl holds {audited_from_rows} fully filed item(s) — "
                "the done marker does not match the rows")
        expected_tally = tallies.get(behavior, pr._zero_refused_tally())
        # exact-int equality: 0.0 / False never stand for 0 (round-4 F2)
        recorded_refused = _exact_int_tally(
            entry.get("auditor_refused"), list(expected_tally),
            f"{bwhere} auditor_refused")
        if recorded_refused != expected_tally:
            raise DriverError(
                f"{bwhere}: results record auditor_refused="
                f"{recorded_refused!r} but the refusal sidecar tallies "
                f"{expected_tally} — the done marker does not match the "
                "sidecar (a missing sidecar tallies zero)")
        sidecar_ids = {(r.scenario_id, int(r.epoch), r.shed.kind)
                       for r in refused_here}
        listed = entry.get("auditor_refused_items")
        if refused_from_sidecar or listed:
            if not isinstance(listed, list):
                raise DriverError(
                    f"{bwhere}: {refused_from_sidecar} refusal(s) in the "
                    "sidecar but the results list no auditor_refused_items "
                    "— identities cannot be compared; refusing")
            try:
                listed_ids = {(str(x["scenario_id"]), int(x["epoch"]),
                               str(x["kind"])) for x in listed}
            except (KeyError, TypeError, ValueError) as e:
                raise DriverError(f"{bwhere}: malformed auditor_refused_items "
                                  f"({e})") from e
            if listed_ids != sidecar_ids:
                raise DriverError(
                    f"{bwhere}: the results' refused identities "
                    f"{sorted(listed_ids)} != the sidecar's "
                    f"{sorted(sidecar_ids)} — the done marker does not match "
                    "the refusal sidecar")
        # this lane: the chain tally recomputed from the same evidence
        # — successful legs from the rows' provenance, terminal legs from
        # the sidecar's shed chains — compared as exact ints (0.0 never
        # stands for 0)
        chain_tally = pr._zero_chain_tally()
        for item in filed:
            chain_tally[pr.chain_kind(chain_by_item.get(item))] += 1
        for r in refused_here:
            chain_tally[pr.chain_kind(r.shed.auditor_chain, refused=True)] += 1
        recorded_chain = _exact_int_tally(
            entry.get("auditor_chain"), list(chain_tally),
            f"{bwhere} auditor_chain")
        if recorded_chain != chain_tally:
            raise DriverError(
                f"{bwhere}: results record auditor_chain={recorded_chain} "
                f"but the store evidences {chain_tally} — the done marker "
                "does not match the chain provenance")
        for k, v in chain_tally.items():
            cell_chain[k] += v
        # the judge twin: tally and identities recomputed from
        # petri_judge_refused.json. A results.json without the key (the
        # runner writes it iff the cell has a judge refusal; cells filed
        # before the key existed never carry it) reconciles ONLY with no
        # judge refusals for this behavior — a filed refusal the results do
        # not report would be a silent N/N (a refusal is reported, never
        # absent)
        recorded_judge = entry.get("judge_refused")
        if recorded_judge is None:
            if judge_from_sidecar:
                raise DriverError(
                    f"{bwhere}: {judge_from_sidecar} judge refusal(s) in "
                    f"{judge_sidecar_path.name} but the results record no "
                    "judge_refused footing — the done marker does not "
                    "report the sidecar's refusals")
        else:
            recorded_judge_tally = _exact_int_tally(
                recorded_judge, ["total"], f"{bwhere} judge_refused")
            if recorded_judge_tally != {"total": judge_from_sidecar}:
                raise DriverError(
                    f"{bwhere}: results record judge_refused="
                    f"{recorded_judge_tally!r} but the judge-refusal sidecar "
                    f"tallies {judge_from_sidecar} — the done marker does "
                    "not match the sidecar (a missing sidecar tallies zero)")
        judge_sidecar_ids = {(r.scenario_id, int(r.epoch)) for r in judge_here}
        listed_judge = entry.get("judge_refused_items")
        if judge_from_sidecar or listed_judge:
            if not isinstance(listed_judge, list):
                raise DriverError(
                    f"{bwhere}: {judge_from_sidecar} judge refusal(s) in the "
                    "sidecar but the results list no judge_refused_items — "
                    "identities cannot be compared; refusing")
            try:
                listed_judge_ids = {(str(x["scenario_id"]), int(x["epoch"]))
                                    for x in listed_judge}
                unnamed = [x for x in listed_judge
                           if not pr.is_judge_refused_reason(x["reason"])]
            except (KeyError, TypeError, ValueError) as e:
                raise DriverError(f"{bwhere}: malformed judge_refused_items "
                                  f"({e})") from e
            if unnamed:
                raise DriverError(
                    f"{bwhere}: judge_refused_items carry a reason that is "
                    f"not a judge refusal ({unnamed[0].get('reason')!r}) — a "
                    "refusal is reported with its reason named")
            if listed_judge_ids != judge_sidecar_ids:
                raise DriverError(
                    f"{bwhere}: the results' judge-refused identities "
                    f"{sorted(listed_judge_ids)} != the sidecar's "
                    f"{sorted(judge_sidecar_ids)} — the done marker does not "
                    "match the judge-refusal sidecar")
        recorded_n_items = _exact_int(entry, "n_items", bwhere)
        if recorded_n_items != (audited_from_rows + refused_from_sidecar
                                + judge_from_sidecar):
            raise DriverError(
                f"{bwhere}: results record n_items={recorded_n_items!r} "
                f"but the store evidences {audited_from_rows} audited + "
                f"{refused_from_sidecar} auditor-refused + "
                f"{judge_from_sidecar} judge-refused item(s) — the "
                "denominator is not this store's")
    # the cell-level footing is RECOMPUTED from the same evidence, never
    # trusted from results.json: the refusal tally from the sidecar, the
    # item counts from the rows and the sidecar
    cell_tally = pr._zero_refused_tally()
    for r in refusals:
        cell_tally["total"] += 1
        cell_tally[r.kind] += 1
    recorded_cell_refused = _exact_int_tally(
        b.results["auditor_refused"], list(cell_tally),
        f"{where} auditor_refused")
    if recorded_cell_refused != cell_tally:
        raise DriverError(
            f"{where}: results record cell-level auditor_refused="
            f"{recorded_cell_refused} but the refusal sidecar tallies "
            f"{cell_tally} — the cell-level done marker does not match the "
            "sidecar")
    # the cell-level judge tally, likewise recomputed from its sidecar; a
    # results.json without the key reconciles only with no judge refusals
    recorded_cell_judge = b.results.get("judge_refused")
    if recorded_cell_judge is None:
        if judge_refusals:
            raise DriverError(
                f"{where}: {len(judge_refusals)} judge refusal(s) in "
                f"{judge_sidecar_path.name} but the results record no "
                "cell-level judge_refused footing — the done marker does not "
                "report the sidecar's refusals")
    else:
        recorded_cell_judge_tally = _exact_int_tally(
            recorded_cell_judge, ["total"], f"{where} judge_refused")
        if recorded_cell_judge_tally != {"total": len(judge_refusals)}:
            raise DriverError(
                f"{where}: results record cell-level judge_refused="
                f"{recorded_cell_judge_tally} but the judge-refusal sidecar "
                f"tallies {len(judge_refusals)} — the cell-level done marker "
                "does not match the sidecar")
    recorded_items = _exact_int(b.results["audited"], "items",
                                f"{where} audited")
    recorded_expected = _exact_int(b.results["audited"], "expected",
                                   f"{where} audited")
    if (recorded_items, recorded_expected) != (evidence_items,
                                               evidence_expected):
        raise DriverError(
            f"{where}: results record audited {recorded_items}/"
            f"{recorded_expected} but the store evidences {evidence_items}/"
            f"{evidence_expected} — the cell-level footing does not match "
            "the store")
    # this lane: the cell-level chain is the sum of the per-behavior chains
    # recomputed above, never trusted from results.json
    recorded_cell_chain = _exact_int_tally(
        b.results.get("auditor_chain"), list(cell_chain),
        f"{where} auditor_chain")
    if recorded_cell_chain != cell_chain:
        raise DriverError(
            f"{where}: results record cell-level auditor_chain="
            f"{recorded_cell_chain} but the store evidences {cell_chain} — "
            "the cell-level done marker does not match the chain provenance")


def _bind_cell_dir(cell_dir: Path) -> BoundCellDir:
    """ONE ``cells/<run_key>/`` directory bound to the cell and the run its
    manifest records -- the per-directory half of :func:`bind_cell_dirs`
    and the binding the already-done path runs (:func:`_already_done`).
    ``manifest.json`` is REQUIRED and names ``cell_id`` and
    ``petri_run_key``; the directory name IS that run key (``CellStore``
    keys Petri cells by ``petri_run_key``) -- anything else is a copied or
    renamed cell dir; and a ``results.json`` must name the same cell and
    run or it is no done marker."""
    cell_dir = Path(cell_dir)
    manifest_path = cell_dir / "manifest.json"
    if not manifest_path.exists():
        raise DriverError(
            f"{cell_dir}: no manifest.json — cannot attribute this cell "
            "dir to a cell id and a run; refusing to guess")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise DriverError(f"{manifest_path}: not a mapping")
    cell_id, run_key = manifest.get("cell_id"), manifest.get("petri_run_key")
    if not cell_id or not run_key:
        raise DriverError(
            f"{manifest_path}: cell_id / petri_run_key not both recorded "
            "— a cell manifest names the cell AND the run; refusing")
    cell_id, run_key = str(cell_id), str(run_key)
    if cell_dir.name != run_key:
        raise DriverError(
            f"{cell_dir}: directory {cell_dir.name!r} is not the run key "
            f"{run_key!r} its manifest records — a copied or renamed "
            "cell dir; refusing")
    results: Optional[Dict[str, Any]] = None
    results_path = cell_dir / "results.json"
    if results_path.exists():
        loaded = json.loads(results_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise DriverError(f"{results_path}: not a mapping")
        for key, expected in (("cell_id", cell_id),
                              ("petri_run_key", run_key)):
            if str(loaded.get(key)) != expected:
                raise DriverError(
                    f"{results_path}: {key} {loaded.get(key)!r} != the "
                    f"manifest's {expected!r} — a results.json that "
                    "does not name its own cell and run is no done "
                    "marker; refusing")
        results = dict(loaded)
    return BoundCellDir(cell_id, run_key, cell_dir, dict(manifest), results)


def _reconcile_cell_dir(b: BoundCellDir, epochs: int) -> None:
    """The ONE reconciliation every reader of a cell dir runs -- the `run`
    already-done path (:func:`_already_done`) and every collector's walk
    (:func:`bind_cell_dirs`): a results.json is a done marker ONLY if the
    store bears it out (:func:`_reconcile_done_marker`)."""
    if b.results is not None:
        _reconcile_done_marker(b, epochs)


def bind_cell_dirs(out_root: Path, record: Mapping[str, Any],
                   epochs: int) -> Dict[str, BoundCellDir]:
    """Every ``cells/<key>/`` directory, BOUND to the run that produced it
    and to the frozen enumeration record — the one walk every collector
    uses (adversarial review 2026-09-05 round 2, blocker 1: a copied
    fully-refused results.json under cells/ with the target cell id and a
    foreign run key used to reach analysis as a COMPLETE-but-unavailable
    cell, its run key stripped, with collect_incomplete reporting nothing) —
    and every results.json RECONCILED with the store evidence beside it
    (:func:`_reconcile_cell_dir`; ``epochs`` forms the item ids).

    Per directory (:func:`_bind_cell_dir`): ``manifest.json`` is REQUIRED
    and names ``cell_id`` and ``petri_run_key``; the directory name IS that
    run key (``CellStore`` keys Petri cells by ``petri_run_key``); a
    ``results.json`` must name the same cell and run or it is no done
    marker. Then, across dirs, two
    directories claiming one cell are refused (a later file never quietly
    overwrites an earlier one), and finally each cell's run key must be
    the frozen enumeration's key for that cell — stale or foreign results
    never stand in for the current run. Returns ``{}`` when there is no
    ``cells/`` directory (synthetic/imported rows analyze without one)."""
    cells_dir = out_root / "cells"
    if not cells_dir.is_dir():
        return {}
    parsed: List[BoundCellDir] = [
        _bind_cell_dir(cell_dir)
        for cell_dir in sorted(p for p in cells_dir.iterdir() if p.is_dir())]
    claimed_by: Dict[str, Path] = {}
    for b in parsed:
        if b.cell_id in claimed_by:
            raise DriverError(
                f"cell {b.cell_id!r} is claimed by two cell dirs: "
                f"{claimed_by[b.cell_id]} and {b.path} — duplicate done "
                "markers / footing for one cell; refusing to let a later "
                "file quietly overwrite an earlier one")
        claimed_by[b.cell_id] = b.path
    served = _served_run_keys(record)
    bound: Dict[str, BoundCellDir] = {}
    for b in parsed:
        current = served.get(b.cell_id)
        if current is None:
            raise DriverError(
                f"{b.path}: claims cell {b.cell_id!r}, which the frozen "
                "enumeration record does not serve — foreign or ABSENT-cell "
                "results never publish")
        if b.petri_run_key != current:
            raise DriverError(
                f"{b.path}: run key {b.petri_run_key!r} is not the frozen "
                f"enumeration's key {current!r} for cell {b.cell_id!r} — "
                "stale or foreign results never stand in for the current run")
        _reconcile_cell_dir(b, epochs)
        bound[b.cell_id] = b
    return bound


def collect_audited_footing(out_root: Path, record: Mapping[str, Any],
                            epochs: int) -> Dict[str, Dict[str, Any]]:
    """cell_id -> the cell's AUDITED FOOTING (``audited`` items/expected +
    ``auditor_refused`` by kind + per-behavior denominators) from every
    COMPLETE cell's results.json that carries them (petri_runner module
    docstring: auditor sheds are a first-class outcome, reported beside
    the scores), each BOUND to its run (:func:`bind_cell_dirs`) and
    carrying that run's ``petri_run_key`` and serving ``engine`` so
    petri_stats.analyze can bind the footing to the enumeration payload —
    a row-less cell is COMPLETE only on footing from ITS run, and every
    done marker has been reconciled with the store's rows, refusal sidecar
    and settled ledger (a results.json without the footing is refused — no
    versioned legacy shape exists)."""
    from sisteer.capeval.tiers import petri_runner as pr  # deferred, heavy
    footing: Dict[str, Dict[str, Any]] = {}
    for cell_id, b in bind_cell_dirs(out_root, record, epochs).items():
        if b.results is None:
            continue
        entry: Dict[str, Any] = {
            "audited": b.results["audited"],
            "auditor_refused": b.results["auditor_refused"],
            # the judge twin (petri_runner judge-refusals comment): the
            # runner writes the key iff the cell has a judge refusal, so a
            # results.json without it — reconciled above against an
            # absent/empty judge sidecar — reads as zero here
            "judge_refused": dict(b.results.get("judge_refused")
                                  or pr._zero_judge_refused_tally()),
            # Distinguishes a fully refused behavior from missing/corrupt rows.
            "behaviors": b.results.get("behaviors", {}),
            # successful / failed auditor-chain legs (this lane's instrument)
            "auditor_chain": b.results.get("auditor_chain", {}),
            # the serving models of the audited items' auditor/judge calls per role (present iff the manifest
            # declares a per-request refusal fallback; every model other than the role's default was fallback-served)
            **({pr.REFUSAL_FALLBACK_KEY: b.results[pr.REFUSAL_FALLBACK_KEY]}
               if isinstance(b.results.get(pr.REFUSAL_FALLBACK_KEY), Mapping) else {}),
            # each role's default model: what a reader subtracts to name the fallback-served calls
            **({pr.ROLE_MODELS_KEY: dict(b.results[pr.ROLE_MODELS_KEY])}
               if isinstance(b.results.get(pr.ROLE_MODELS_KEY), Mapping) else {}),
            # the overflow pass's relaunch record when the cell went through one: window, overflowed + re-rolled ids
            **_rerun_pass_of(b),
            # Grok-fix-4: the judgments a NON-first-party terminal leg answered (by class / requested / served)
            # and the judge-only re-score block -- forwarded so analysis.by_cell[*].audited_footing and the registry judgment notes
            # (petri_judgment_notes) carry requested/served/class and the re-score revision; absent when the results carry none
            **({pr.JUDGE_BUILD_KEY: dict(b.results[pr.JUDGE_BUILD_KEY])}
               if isinstance(b.results.get(pr.JUDGE_BUILD_KEY), Mapping) else {}),
            **({pr.JUDGE_RESCORED_KEY: dict(b.results[pr.JUDGE_RESCORED_KEY])}
               if isinstance(b.results.get(pr.JUDGE_RESCORED_KEY), Mapping) else {}),
            # the run that produced this footing — analyze binds it to the
            # enumeration payload (round-2 blocker 1)
            "petri_run_key": b.petri_run_key,
        }
        if b.manifest.get("engine") is not None:
            entry["engine"] = str(b.manifest["engine"])
        # the served ENGINE the cell manifest names: additive —
        # a manifest written before the key existed carries none, and the
        # footing then carries none (lens outputs byte-identical)
        if b.manifest.get(tc.SERVED_ENGINE_FIELD) is not None:
            entry[tc.SERVED_ENGINE_FIELD] = str(b.manifest[tc.SERVED_ENGINE_FIELD])
        footing[cell_id] = entry
    return footing


def _rerun_pass_of(b: Any) -> Dict[str, Any]:
    """``{"rerun_pass": {...}}`` from the cell's RERUN_RELAUNCH_RECORD when one exists (the cell went through the overflow pass),
    else {} -- a cell that never did reads byte-identically."""
    from sisteer.capeval.tiers import petri_runner as pr
    d = getattr(b, "path", None)
    if d is None:
        return {}
    rec = pr.read_rerun_relaunch(Path(d))
    return {"rerun_pass": dict(rec)} if rec is not None else {}


def collect_incomplete(out_root: Path, record: Mapping[str, Any],
                       epochs: int) -> Dict[str, str]:
    """Map cell_id -> reason for every INCOMPLETE cell. The failure ledger
    is AUTHORITATIVE (review 2026-08-18): it is inspected for EVERY cell
    dir — a results.json done marker coexisting with unresolved ledger
    entries is an integrity refusal, never a silently-consolidated cell.
    Every cell dir is first BOUND to its manifest, its run and the frozen
    enumeration (:func:`bind_cell_dirs`): a results.json that does not
    name its own cell and run, or belongs to another run, is no done
    marker (round-2 blocker 1)."""
    cells_dir = out_root / "cells"
    if not cells_dir.is_dir():
        raise DriverError(f"no cells/ under {out_root} — nothing ran here")
    incomplete: Dict[str, str] = {}
    for cell_id, b in bind_cell_dirs(out_root, record, epochs).items():
        if b.results is not None:
            continue  # reconciled with rows / sidecar / settled ledger above
        # an incomplete cell's ledger, when it exists, must name this cell
        # and run too
        ledger = _bound_ledger(b)
        unresolved = list(ledger["unresolved"]) if ledger else []
        reason = "no results.json (unresolved required dimensions, H9)"
        if unresolved:
            reason = (f"{len(unresolved)} unresolved failure-ledger "
                      f"entr{'y' if len(unresolved) == 1 else 'ies'}")
        incomplete[cell_id] = reason
    return incomplete


def collect_auditor_refusal_tombstones(
    out_root: Path, spec: cm.PetriSpec
) -> Dict[str, set[Tuple[str, str, str]]]:
    """Cell-local score rows explicitly excluded by terminal refusals —
    auditor refusals (petri_auditor_refused.json) and judge refusals
    (petri_judge_refused.json) alike.

    The refusal sidecars are the authoritative tombstone record. This preserves
    append-only partial judge rows as evidence while ensuring they cannot enter
    consolidation, analysis, bootstraps, or reports.
    """
    from sisteer.capeval.tiers import petri_runner as pr

    cells_dir = out_root / "cells"
    if not cells_dir.is_dir():
        return {}
    out: Dict[str, set[Tuple[str, str, str]]] = {}
    for cell_dir in sorted(p for p in cells_dir.iterdir() if p.is_dir()):
        store = cs.CellStore(out_root, cell_dir.name)
        keys: set[Tuple[str, str, str]] = set()
        refusals = pr.read_auditor_refused(store)
        if refusals:
            keys |= pr.auditor_refused_row_keys(
                refusals, spec.bloom_options.epochs)
        judge_refusals = pr.read_judge_refused(store)
        if judge_refusals:
            keys |= pr.judge_refused_row_keys(
                judge_refusals, spec.bloom_options.epochs)
        if keys:
            out[cell_dir.name] = keys
    return out


def _consolidate_store(out_root: Path, spec: cm.PetriSpec) -> Path:
    """Fold the store's DONE cells into ``<out_root>/rows.parquet`` (``capeval.store.consolidate``) with the INCOMPLETE
    map beside it: the body of ``consolidate``, shared with the registry-worklist ``analyze`` (below). Deterministic over
    the cell dirs and idempotent; no API call — it reads rows.jsonl and the refusal sidecars, nothing else."""
    # cell dirs are bound to THIS manifest's frozen enumeration before any
    # results.json counts as a done marker (round-2 blocker 1)
    record = read_cells_record(out_root)
    _check_record_matches_spec(record, spec, out_root / CELLS_JSON)
    incomplete = collect_incomplete(out_root, record,
                                    spec.bloom_options.epochs)
    map_path = out_root / INCOMPLETE_MAP
    map_path.write_text(json.dumps(incomplete, indent=2, sort_keys=True),
                        encoding="utf-8")
    if incomplete:
        print(f"[petri] {len(incomplete)} INCOMPLETE cell(s) recorded in "
              f"{map_path}: {sorted(incomplete)} — they never consolidate; "
              "petri_stats will refuse any paired estimate involving them "
              "(H9)")
    # Petri's report path deliberately excludes incomplete cells WITH the
    # explicit map above (module docstring) — this is H9's design, not the
    # capability pipeline's skip_incomplete inspection shortcut.
    tombstones = collect_auditor_refusal_tombstones(out_root, spec)
    out = cs.consolidate(
        out_root, skip_incomplete=bool(incomplete),
        exclude_keys_by_cell=tombstones)
    print(f"[petri] consolidated -> {out}")
    return out


def cmd_consolidate(args: argparse.Namespace, spec: cm.PetriSpec) -> int:
    _consolidate_store(_out_root(args, spec), spec)
    return 0


def _parse_rescore_item(text: str) -> Tuple[str, str, int]:
    """``behavior/scenario_id[#eN]`` -> (behavior, scenario_id, epoch) (epoch 1 by default)."""
    body, _, ep = text.partition("#e")
    behavior, sep, sid = body.partition("/")
    if not sep or not behavior or not sid:
        raise DriverError(f"--items: {text!r} is not behavior/scenario_id[#eN]")
    return behavior, sid, (int(ep) if ep else 1)


def _rescore_effective_spec(spec: cm.PetriSpec, record: Mapping[str, Any], path: Path) -> cm.PetriSpec:
    """The spec the frozen record's legs were HASHED under. A native record (``serving_path`` lora_native) was enumerated over
    :func:`native_effective_spec` -- the manifest's cells minus the ones the standup skipped by name, ``serving_launch_flags`` minus the
    tokens the native launch strips (``--enforce-eager``; recorded as ``flags_dropped``) -- so the H10 recompute of a re-score must run
    over THAT spec, never the manifest's: on 2026-10-02 the u0 cells refused (recomputed ece6dd983b8a4842, enumerated 0ca4fb79ff945f14)
    for exactly this, under a manifest whose hash inputs had not moved. Derived from the RECORD, not by re-enumeration (a re-score runs
    where the registry's cell source need not resolve), and held to it: the tokens the derivation drops today must be the ones the
    record says were dropped, else the legs were hashed under other flags and the re-score refuses by name. A lens record (no
    ``serving_path``: absent = lens by era, as the cell manifests read it) is the manifest spec, exactly as before."""
    from sisteer.capeval.serving import native_standup as ns
    if record.get("serving_path", SERVING_PATH_LENS) != SERVING_PATH_LORA_NATIVE:
        return spec
    kept, dropped = ns.effective_launch_flags(list(spec.serving_launch_flags))
    recorded_raw = record.get("flags_dropped")
    if not isinstance(recorded_raw, list):
        # the native enumeration ALWAYS writes the list (empty when nothing was stripped); a record without it, or with null, is
        # incomplete -- reading it as [] would pass the check for a manifest that strips nothing today. Fail closed.
        raise DriverError(f"{path}: a native enumeration record carries flags_dropped as a list (empty when the launch stripped nothing) but this "
                          f"one has {recorded_raw!r} -- an incomplete record; refusing to re-score against it (re-freeze deliberately into a "
                          "fresh out root)")
    recorded = [str(f) for f in recorded_raw]
    if list(dropped) != recorded:
        raise DriverError(f"{path}: the frozen record says the native launch dropped {recorded} from serving_launch_flags but the manifest's "
                          f"flags drop {list(dropped)} today -- the legs were hashed under other flags; refusing to re-score against them "
                          "(re-freeze deliberately into a fresh out root)")
    skipped = {str(s.get("cell_id")) for s in (record.get("cells_skipped") or []) if isinstance(s, Mapping)}
    return dataclasses.replace(spec, cells=[c for c in spec.cells if c not in skipped], serving_launch_flags=list(kept))


def cmd_rescore(args: argparse.Namespace, spec: cm.PetriSpec) -> int:
    """Judge-only RE-SCORE of this cell's TERMINAL judge refusals over their EXISTING transcripts:
    the explicit operator action whose selection effect is recorded in the cell's judge-refused sidecar (``rescored``). Never an audit,
    never a target call: the transcript is read from the refused sample's .eval log and only the judge runs again, under the
    manifest's CURRENT judge chain (the same scorer the audit used -- the behaviour's bloom_audit task scorer -- and the same judge
    wiring the live driver builds). Refuses by name: a cell that is not COMPLETE (results.json absent -- the drive's business), a
    manifest declaring no judge fallback (the same single judge would be a re-roll), a harness-hash drift since enumeration (H10), an
    item that is not re-scorable (petri_rescore.select_terminal_refusals). ``--dry-run`` selects and prints, writes nothing."""
    from sisteer.capeval.petri import scenarios as psc
    from sisteer.capeval.tiers import petri_rescore as prs
    from sisteer.capeval.tiers import petri_runner as pr
    out_root = _out_root(args, spec)
    record = read_cells_record(out_root)
    _check_record_matches_spec(record, spec, out_root / CELLS_JSON)       # the record belongs to this manifest (its cells, its target decode)
    spec = _rescore_effective_spec(spec, record, out_root / CELLS_JSON)    # ... and its legs were hashed under the EFFECTIVE spec
    served = [dc.PetriCell(**c) for c in list(record.get("canonical") or []) + list(record.get("controls") or [])]
    cell = _resolve_cell(served, args)
    store = cs.CellStore(out_root, cell.petri_run_key)
    if not store.is_done():
        raise DriverError(f"rescore: cell {cell.cell_id!r} is not COMPLETE (no results.json under {store.cell_dir}) -- finishing the "
                          "audit is the drive's business; a re-score acts on a complete cell's terminal judge refusals only")
    if not spec.judge_refusal_fallback_models:
        raise DriverError("rescore: the manifest declares no judge_refusal_fallback_models -- re-scoring under the same single judge "
                          "would be a re-roll of the same judge, which the contract forbids")
    items = [_parse_rescore_item(s) for s in (args.items or [])] or None
    config_root = Path(args.config_root or os.environ.get("PETRI_CONFIG_ROOT") or pr._repo_config_root())
    bf_all = psc.behavior_files(config_root, spec.behaviors)
    bf = {k: bf_all[k] for k in cm.petri_expected_behavior_keys(spec) if k in bf_all}
    prompt_files = pr.resolve_prompt_files(spec)
    chash = cm.petri_harness_config_hash(spec, behavior_files=bf, prompt_files=prompt_files, serving_identity=cell.serving_identity)
    if chash != cell.harness_config_hash:
        raise DriverError(f"rescore: harness config hash recomputed from {config_root} is {chash} but cell {cell.cell_id!r} was "
                          f"enumerated at {cell.harness_config_hash} -- the behaviour definitions, prompt bytes, bloom options or "
                          "serving pins moved since enumeration; a re-score must judge under the enumerated harness (H10)")
    if args.dry_run:
        chosen = prs.select_terminal_refusals(store, spec, items)
        print(f"[petri] rescore DRY RUN: cell {cell.cell_id} -- {len(chosen)} terminal judge refusal(s) re-scorable under judge chain "
              f"{prs.chain_of(spec)}: " + ", ".join(f"{r.behavior}/{r.scenario_id}#e{r.epoch} (filed under {r.judge_model})" for r in chosen)
              + "; nothing written")
        return 0
    bundle = pr._resolve_bundle_dir(spec, getattr(args, "bundle_dir", None), config_root)
    work_root = Path(os.environ.get("PETRI_WORK_DIR") or out_root)
    dirs = psc.materialize(bundle, work_root / f"behaviors-{spec.scenario_bundle_hash}", expected_bundle_hash=spec.scenario_bundle_hash,
                           config_root=config_root, behaviors=spec.behaviors)
    canonical_ids = {b: psc.scenario_ids(dirs[b] / "scenarios") for b in spec.behaviors}
    from petri_bloom import bloom_audit  # deferred: the live task (the same scorer the audit used)

    def scorers_for(behavior: str):
        task = bloom_audit(behavior=str(dirs[pr.behavior_dir_key(behavior, cell)]), **pr.bloom_task_options(spec))
        sc = task.scorer
        return list(sc) if isinstance(sc, (list, tuple)) else [sc]
    judge = pr._role_model_with_refusal_fallback(spec.judge_model, [str(m) for m in spec.judge_refusal_fallback_models], pr.JUDGE_ROLE)
    cell_manifest = json.loads((store.cell_dir / "manifest.json").read_text(encoding="utf-8"))
    serving_engine = cell_manifest.get(tc.SERVED_ENGINE_FIELD)
    run_identity = pr.target_decode_run_identity(spec)
    target_decode, target_seed_base = run_identity or (None, None)
    report = prs.rescore_cell(store, cell, spec, items=items, why=args.why, requested_by=args.requested_by, scorers_for=scorers_for,
                              judge_model=judge, dir_key_of=lambda b: pr.behavior_dir_key(b, cell), canonical_ids_of=lambda b: canonical_ids[b],
                              env_digest=cs.env_digest(), serving_engine=serving_engine, target_decode=target_decode,
                              target_seed_base=target_seed_base)
    print(f"[petri] rescore: cell {cell.cell_id} revision {report.selection.revision()} -- {len(report.scored)} scored "
          f"({report.rows_filed} rows filed), {len(report.refiled_terminal)} re-filed terminal; logs: {report.logs}; "
          f"the sidecar's '{pr.JUDGE_RESCORED_KEY}' list records the selection effect")
    return 0


def _load_certificates(path: Optional[Path]) -> Tuple[dc.NoOpCertificate, ...]:
    """The §10 gate's no-op certificates (serving-coverage record). Each
    entry must carry the FULL key — a partial certificate is a refusal."""
    if path is None:
        return ()
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list):
        raise DriverError(f"{path}: certificate file must be a JSON list")
    # the FULL key, derived from the dataclass so it can never drift
    fields = tuple(f.name for f in dataclasses.fields(dc.NoOpCertificate))
    certs: List[dc.NoOpCertificate] = []
    for e in entries:
        missing = [f for f in fields if f not in e]
        if missing:
            raise DriverError(
                f"{path}: certificate {e!r} missing {missing} — the full "
                "key is the license (plan §10 H5)")
        certs.append(dc.NoOpCertificate(**{f: e[f] for f in fields}))
    return tuple(certs)


def _gate_prompt_set_hash() -> str:
    """The current §10 gate prompt-set hash — part of the Tier-2 full
    certificate key. Loaded from scripts/petri_serving_gate.py (a sibling
    script, CPU-pure) via importlib since scripts/ is not a package."""
    import importlib.util
    if "petri_serving_gate" in sys.modules:
        return sys.modules["petri_serving_gate"].gate_prompt_set_hash()
    path = Path(__file__).resolve().parent / "petri_serving_gate.py"
    loader_spec = importlib.util.spec_from_file_location(
        "petri_serving_gate", path)
    mod = importlib.util.module_from_spec(loader_spec)
    # registered BEFORE exec: the module's dataclasses resolve their
    # defining module through sys.modules at class-creation time
    sys.modules["petri_serving_gate"] = mod
    loader_spec.loader.exec_module(mod)
    return mod.gate_prompt_set_hash()


DRIVER_ID = "scripts/petri_run.py analyze"


def registry_worklist_runs(args: argparse.Namespace, out_root: Path):
    """``--worklist registry``: for Petri the aggregation IS the judgment step, so the runs
    this analysis judges come from ``v_pending_judgment`` — petri runs (``eval_category = petri``) still
    ``generated`` (or claimed) whose store lies under THIS audit's out-root; pending runs under other roots belong
    to other audits and are counted, not judged. Returns ``(worklist, rows)``; the caller claims every run before
    analysing."""
    from sisteer.registry import worklist as RW
    wl = RW.RegistryWorklist(getattr(args, "registry_root", None), getattr(args, "judge", None), getattr(args, "session", None),
                             getattr(args, "campaign", None), driver=DRIVER_ID,
                             **RW.selectors_from_args(args))          # --run-id / --model-id / --store-root-map
    # the runs still to judge by their instrument judge: never judged (generated / judging), PLUS the runs pending AGAIN because their
    # judgment was superseded (a same-id judgment record with status superseded puts the run back in v_pending_judgment with
    # pending_reason `superseded`, its lifecycle `ingested`) -- the 2026-10-02 04:42Z re-dispatch of the 21 u0 runs found nothing
    # under the (generated, judging) filter and quarantined them again. The second-judge rows (pending_reason awaiting_second_judge) are
    # never this worklist's: wl.pending without --judge reads the instrument judge's rows only.
    rows = wl.pending(eval_category="petri", lifecycle_state=("generated", "judging"))
    superseded = wl.pending(eval_category="petri", pending_reason="superseded")
    # v_pending_judgment's pending_reason CASE ranks `abandoned` ABOVE `superseded`, so a superseded run whose re-judge
    # claimed and then failed (the claim released as abandoned, 2026-10-02 08:22Z: four u0 runs) reads `abandoned`, lifecycle `ingested`
    # -- and left every query here. Such a run is pending again for the same reason the `superseded` rows are: its instrument judge has a
    # superseded judgment and no complete (or partial) one. Admitted on exactly that evidence, read from the judgment table; every claim
    # gate below stands (pending.claim decides who may take it).
    abandoned = wl.pending(eval_category="petri", pending_reason="abandoned", lifecycle_state=("ingested", "judged"))
    if abandoned:
        from sisteer.registry import db as RDB
        with RDB.reading(RDB.db_path(wl.root)) as conn:
            def _superseded_unreplaced(r):
                st = {row[0] for row in conn.execute("SELECT DISTINCT status FROM judgment WHERE run_id = ? AND judge_id = ?",
                                                     (str(r["run_id"]), str(r.get("required_judge_id") or "")))}
                return "superseded" in st and not ({"complete", "partial"} & st)
            abandoned = [r for r in abandoned if _superseded_unreplaced(r)]
    # a row THIS session already claimed -- the replacement grader claimed it and died before filing -- reads pending_reason `claimed`
    # and leaves both queries above; the daemon re-dispatches its own session's claims (grade_daemon.pending_rows), so the same session
    # re-claims under the same judgment id and files. Another session's claim is never admitted here (and pending.claim would refuse
    # it): the gate against other sessions stands.
    own = [r for r in wl.pending(eval_category="petri", pending_reason="claimed")
           if r.get("claim_session") and str(r.get("claim_session")) == str(wl.session)]
    # a run whose judgment or in-progress claim for its required judge was bound to a run identity since CORRECTED is re-pended on its own
    # row as `identity_stale` (IDENTITY_STALE_REASON) -- the stale judgment's row stands but counts for nothing, a stale-bound claim is
    # never shown as `claimed` -- with its lifecycle wherever the derivation left it: for a run judged before the correction `ingested` /
    # `judged`, which no query above selects. Admitted on exactly that evidence (the (generated, judging) rows already carry the rest);
    # every claim gate below stands -- pending.claim closes the stale-bound claim and mints a current-bound one, under which the run is
    # judged against the corrected identity.
    identity_stale = wl.pending(eval_category="petri", pending_reason=IDENTITY_STALE_REASON, lifecycle_state=("ingested", "judged"))
    seen = {str(r["run_id"]) for r in rows}
    extra = [r for r in list(superseded) + list(abandoned) + own + list(identity_stale) if str(r["run_id"]) not in seen and not seen.add(str(r["run_id"]))]
    rows = list(rows) + extra
    # a row this session claimed whose claim's COMPLETE judgment is already on the spool is FILED, not dead -- JudgingScope.complete writes
    # the completion to the spool while the DB claim stays in_progress until the harvester folds it, so the view reads `claimed` (a
    # never-ingested run still `generated`) for a judgment that is durable, whichever query above returned the row. Judging it again under
    # the same id (pending.claim: same session -> same id) would write a second completion record. Left to the fold.
    from sisteer.registry import pending as RP

    def _filed_on_spool(r):
        jid = str(r.get("claim_judgment_id") or "")
        return bool(jid) and bool(r.get("claim_session")) and str(r.get("claim_session")) == str(wl.session) and any(
            j["judgment_id"] == jid for j in RP.spool_complete_judgments(wl.root, r.get("campaign"), str(r["run_id"])))
    filed = [r for r in rows if _filed_on_spool(r)]
    if filed:
        rows = [r for r in rows if not any(r is f for f in filed)]
        own = [r for r in own if not any(r is f for f in filed)]
        print(f"[registry] worklist=registry: {len(filed)} run(s) claimed by this session ({wl.session}) whose completion is already on the spool "
              f"(judgment id(s) {sorted(str(r.get('claim_judgment_id')) for r in filed)}): left to the harvester's fold, not judged again", flush=True)
    if superseded:
        print(f"[registry] worklist=registry: {len(superseded)} run(s) pending again after a superseded judgment (D617); re-judged under a "
              "new judgment id, the superseded one stays on record", flush=True)
    if abandoned:
        print(f"[registry] worklist=registry: {len(abandoned)} superseded run(s) whose last re-judge claim was abandoned (the grader failed "
              "after claiming); pending again, re-judged under a new judgment id", flush=True)
    if identity_stale:
        print(f"[registry] worklist=registry: {len(identity_stale)} run(s) pending again as {IDENTITY_STALE_REASON} (a judgment or claim bound to a "
              "since-corrected run identity; it counts for nothing); re-judged under a current-bound claim against the corrected identity", flush=True)
    if own:
        print(f"[registry] worklist=registry: {len(own)} run(s) already claimed by this session ({wl.session}) and not yet filed -- re-dispatched "
              "under the same claim (the grader that claimed them exited before filing)", flush=True)
    base = Path(out_root).expanduser().resolve()
    mine = []
    for r in rows:
        sp = wl.store_path(r)          # the recorded path relocated through the store map before it is matched to this audit's root
        d = Path(sp).expanduser().resolve() if sp else None
        if d is not None and (d == base or base in d.parents):
            mine.append(r)
    print(f"[registry] worklist=registry: {len(mine)} pending petri run(s) under {base}; "
          f"{len(rows) - len(mine)} pending run(s) belong to other audits", flush=True)
    # the selected rows are THE identity baseline -- the borrow compares its in-view SELECT against them, the filing re-reads the registry
    # against them, the filing loop routes from them: one set of rows. v_pending_judgment carries most of the in-view identity (run_id,
    # eval_id, the producer key, object_id, campaign, model_id, store_uri, n_items_generated; substrate / view_legacy from the worklist's
    # enrichment) but not model_revision or the registered completion pair, so every BORROW_IN_VIEW_FIELDS column is read here for the
    # selected ids in ONE read-only query (the one identity function); a field the pending row already carries must agree -- the pending
    # read and this read are separate connections, and a same-id correction folded between them refuses by name, never a blended row --
    # and a missing one is attached. Every row then carries every in-view identity field the borrow and the recheck compare.
    if mine:
        from sisteer.registry import db as RDB
        with RDB.reading(RDB.db_path(wl.root)) as conn:
            ident = _anchor_identity_rows(conn, [str(r["run_id"]) for r in mine], BORROW_IN_VIEW_FIELDS)
        problems: List[str] = []
        for r in mine:
            rid = str(r["run_id"])
            now = ident.get(rid)
            if now is None:
                problems.append(f"{rid}: no longer in run JOIN v_run_current")
                continue
            # through the one comparison, over the fields the pending row carries -- instrument_id and the bound
            # required_judge_id among them: the bound judge must still be a pending pair for the run and, under the default worklist, the judge the
            # run's current instrument designates (the identity read recomputes instrument_judge_id and attaches the pending judges)
            diffs = identity_diff_lines(r, now, [f for f in BORROW_IN_VIEW_FIELDS if f in r], default_judge=wl.judges is None)
            if diffs:
                problems.append(f"{rid}: " + "; ".join(diffs))
                continue
            for f in BORROW_IN_VIEW_FIELDS:
                if f not in r:
                    r[f] = now.get(f)
        if problems:
            raise DriverError("--worklist registry: selected run's identity changed between the worklist read and its identity read: "
                              + " | ".join(problems) + "; refusing before any claim -- the worklist rows are the one identity baseline the borrow, "
                              "the filing recheck and the filing route from, never a blend of two reads")
    return wl, mine


def _cell_of(row: Dict[str, Any]) -> str:
    """The run's cell: the object part of its producer object key (``<tag>:<object>``; the whole key when it carries no tag)."""
    key = str(filing_field(row, "producer_object_key_legacy") or "")
    return key.split(":", 1)[1] if ":" in key else key


PETRI_POOLED_EVAL_ID = "petri_pooled"


def _behaviour_of(row: Dict[str, Any]) -> Optional[str]:
    """The run's behaviour from its eval id: ``petri_<behaviour>`` -> the behaviour; ``petri_pooled`` -> None (every behaviour);
    anything else refuses by name (a Petri run's eval id is one of those two shapes)."""
    eval_id = str(filing_field(row, "eval_id") or "")
    if eval_id == PETRI_POOLED_EVAL_ID:
        return None
    if eval_id.startswith("petri_") and len(eval_id) > len("petri_"):
        return eval_id[len("petri_"):]
    raise DriverError(f"run {filing_field(row, 'run_id')!r}: eval id {eval_id!r} is neither {PETRI_POOLED_EVAL_ID!r} nor petri_<behaviour>; "
                      "refusing to guess which rows are its judgment")


def _harness_cell_of(row: Dict[str, Any]) -> str:
    """The run's cell AS THE FRAME AND THE ANALYSIS NAME IT: the registry's object key is the TrainTest id (``prompted__safety_refusal``,
    ``prompted__si_constitution``, ``safety_refusal__psr_kl_all``), the harness -- rows.parquet's ``arm_name``, analysis.by_cell -- the
    harness id (``prompted__safety``, ``prompted__constitution``, ``si_safety__psr_kl_all``); ``none`` and the Class B ids coincide.
    Translated through the ONE mapping (registry_adapter.harness_cell_id); a cell the mapping cannot express refuses by name -- the
    2026-10-02 u0 fold matched the raw TrainTest id against harness arm names and filed NOTHING for every prompted run."""
    from sisteer.capeval.petri import registry_adapter as ra
    raw = _cell_of(row)
    try:
        return ra.harness_cell_id(raw)
    except ra.RegistryAdapterError as e:
        raise DriverError(f"run {filing_field(row, 'run_id')!r}: its cell {raw!r} has no harness translation ({e}); refusing") from None


def _fully_terminally_refused(analysis: Optional[Mapping[str, Any]], cell: str, behaviour: Optional[str]) -> bool:
    """Whether the analysis's validated footing PROVES the run's selection is legitimately empty: the behaviour (every
    behaviour, for a pooled run) of the cell has n_items > 0, audited 0, and auditor_refused.total + judge_refused.total == n_items --
    petri_stats admits such complete zero-row behaviours (mean None, n_items 0 judged). Anything short of that proof is not a reason."""
    if not isinstance(analysis, Mapping):
        return False
    entry = (analysis.get("by_cell") or {}).get(cell)
    footing = entry.get("audited_footing") if isinstance(entry, Mapping) and isinstance(entry.get("audited_footing"), Mapping) else None
    blocks = footing.get("behaviors") if footing and isinstance(footing.get("behaviors"), Mapping) else None
    if not blocks:
        return False
    names = [behaviour] if behaviour is not None else list(blocks)
    for b in names:
        bf = blocks.get(b)
        if not isinstance(bf, Mapping):
            return False
        try:
            n_items = int(bf.get("n_items") or 0); audited = int(bf.get("audited") or 0)
            refused = int((bf.get("auditor_refused") or {}).get("total") or 0) + int((bf.get("judge_refused") or {}).get("total") or 0)
        except (TypeError, ValueError, AttributeError):
            return False
        if not (n_items > 0 and audited == 0 and refused == n_items):
            return False
    return True


def petri_item_rows(frame, row: Dict[str, Any], analysis: Optional[Mapping[str, Any]] = None) -> list:
    """The judge ``item_result`` rows of one registry run from the consolidated score
    frame the analysis aggregated (``petri_stats.load_rows(rows.parquet)``: one row per (cell, behaviour, scenario item, score
    dimension) — the judge's per-transcript scores; ``analysis`` carries their means only). The RUN's rows: its cell translated to
    the harness id (:func:`_harness_cell_of`) AND its behaviour (:func:`_behaviour_of`; the pooled run takes every behaviour) --
    the 2026-10-02 u0 fold filtered by cell alone and filed all 60 items of the cell on each per-behaviour run. Item id
    ``<behaviour>/<scenario item>``; the behaviour's own score dimension files as ``petri_score`` and the two shared dimensions
    (``eval_awareness`` / ``scenario_realism``) under their own metric ids; ``turn_index`` -1 (a transcript, not a turn) — the
    registry's Petri item convention. A frame that names the cell but yields NO row for the run's behaviour is refused by name -- the
    cell is complete (the analysis said so), so an empty selection is a mapping defect, never a judgment of zero items -- EXCEPT when
    the analysis's validated footing proves the behaviour fully terminally refused (:func:`_fully_terminally_refused`): then the run's
    judgment is zero items plus its refusal count, no metric."""
    cell = _harness_cell_of(row)
    behaviour = _behaviour_of(row)
    arms = frame["arm_name"].astype(str)
    sub = frame[arms == cell]
    if behaviour is not None:
        sub = sub[sub["behavior"].astype(str) == behaviour]
    if sub.empty and _fully_terminally_refused(analysis, cell, behaviour):
        return []                                          # a complete, fully terminally refused behaviour: 0 judged items, by proof
    if sub.empty:
        named = sorted(set(arms))
        raise DriverError(f"run {filing_field(row, 'run_id')!r}: the consolidated frame holds no row for cell {cell!r}"
                          f"{'' if behaviour is None else ' behaviour ' + repr(behaviour)} (frame cells {named}); a complete cell with "
                          "an empty selection is a mapping defect -- refusing to file a judgment of no items")
    rows: list = []
    for r in sub.itertuples():
        b, item, metric = str(r.behavior), str(r.item_id), str(r.metric)
        rows.append({"item_id": f"{b}/{item}", "metric_id": "petri_score" if metric == b else metric,
                     "value": float(r.value), "label": None, "extra": {"behaviour": b, "dimension": metric}})
    rows.sort(key=lambda x: (x["item_id"], x["metric_id"]))
    return rows


def petri_metric_rows(analysis: Dict[str, Any], row: Dict[str, Any]) -> list:
    """``run_metric`` rows for one registry run from the analysis: for a per-behaviour run, THAT behaviour's mean judge score
    (``by_cell[harness cell].behaviors[behaviour].mean_behavior_score``) with its ``n_items``; for the pooled run, the mean over the
    behaviours with the summed n (the 2026-10-02 u0 fold gave every per-behaviour run the cell-wide mean). A cell the analysis does
    not name refuses by name (the analysis covers every complete cell; an unnamed one is a mapping defect)."""
    cell = _harness_cell_of(row)
    behaviour = _behaviour_of(row)
    entry = (analysis.get("by_cell") or {}).get(cell)
    if not isinstance(entry, Mapping):
        raise DriverError(f"run {filing_field(row, 'run_id')!r}: analysis.by_cell does not name cell {cell!r} "
                          f"(cells {sorted((analysis.get('by_cell') or {}))}); refusing to file a judgment with a guessed metric")
    behaviours = entry.get("behaviors") or {}
    if behaviour is not None:
        b = behaviours.get(behaviour)
        if not isinstance(b, Mapping) or not isinstance(b.get("mean_behavior_score"), (int, float)):
            return []                                       # an incomplete behaviour publishes no mean (the analysis says None)
        return [{"metric_id": "petri_score", "value": float(b["mean_behavior_score"]), "n": int(b.get("n_items") or 0),
                 "producer_metric_spelling": f"by_cell[cell].behaviors[{behaviour}].mean_behavior_score"}]
    scores = [b.get("mean_behavior_score") for b in behaviours.values()
              if isinstance(b, Mapping) and isinstance(b.get("mean_behavior_score"), (int, float))]
    if not scores:
        return []
    n = sum(int(b.get("n_items") or 0) for b in behaviours.values() if isinstance(b, Mapping))
    return [{"metric_id": "petri_score", "value": float(sum(scores) / len(scores)), "n": n,
             "producer_metric_spelling": "by_cell[cell].behaviors[*].mean_behavior_score (mean)"}]


def petri_judgment_notes(analysis: Dict[str, Any], row: Dict[str, Any]) -> Tuple[Optional[int], Dict[str, Any]]:
    """What the registry judgment carries beside its metric for one run: the
    arm's judge-refused count (``n_judge_refusals``) and, as the judgment's ``extra`` notes, the per-arm ``judge_refused``
    tally and the ``refusal_fallback`` serving-model counts per role (present iff the manifest declares a per-request
    refusal fallback) -- so the fallback-served and judge-refused counts reach the registry and the artifact without a
    reader re-opening the eval logs. A run the analysis does not name gets (None, {})."""
    from sisteer.capeval.tiers import petri_runner as pr  # deferred, heavy (the sibling functions' convention)
    entry = (analysis.get("by_cell") or {}).get(_harness_cell_of(row))
    if not isinstance(entry, Mapping):
        return None, {}
    behaviour = _behaviour_of(row)
    # petri_stats.analyze puts the cell's footing (the results.json footing collect_audited_footing handed it: audited,
    # auditor_refused, judge_refused -- zero included --, auditor_chain, refusal_fallback, role_models, rerun_pass) under
    # `audited_footing`, and a first-class `judge_refused` beside it ONLY when the cell has any (reading the entry's top
    # level lost both roles' counts and every zero-refusal arm's n_judge_refusals=0)
    footing = entry.get("audited_footing") if isinstance(entry.get("audited_footing"), Mapping) else {}
    notes: Dict[str, Any] = {}
    n_refused: Optional[int] = None
    # the per-behaviour run's refusal count is ITS behaviour's (the footing's behaviours block carries `judge_refused` per
    # behaviour); the pooled run's is the cell total -- the 2026-10-02 fold stamped the cell total on every per-behaviour run
    judge_refused = footing.get("judge_refused")
    if behaviour is not None:
        bf = (footing.get("behaviors") or {}).get(behaviour) if isinstance(footing.get("behaviors"), Mapping) else None
        judge_refused = (bf.get("judge_refused") if isinstance(bf, Mapping) and isinstance(bf.get("judge_refused"), Mapping)
                         else {"total": 0})
    if isinstance(judge_refused, Mapping):
        try:
            n_refused = int(judge_refused.get("total") or 0)
        except (TypeError, ValueError):
            n_refused = None
        notes["judge_refused"] = dict(judge_refused)
    if isinstance(entry.get("judge_refused"), Mapping):          # the richer non-zero term (items with reasons) wins
        rich = json.loads(json.dumps(entry["judge_refused"]))
        if behaviour is not None and isinstance(rich.get("items"), list):
            mine = [x for x in rich["items"] if isinstance(x, Mapping) and str(x.get("behavior")) == behaviour]
            rich = {**rich, "items": mine, "total": len(mine)}
        notes["judge_refused"] = rich
        if n_refused is None:
            try:
                n_refused = int(entry["judge_refused"].get("total") or 0)
            except (TypeError, ValueError):
                n_refused = None
    for key in (pr.REFUSAL_FALLBACK_KEY, pr.ROLE_MODELS_KEY, "rerun_pass", pr.JUDGE_BUILD_KEY, pr.JUDGE_RESCORED_KEY):
        if isinstance(footing.get(key), Mapping):
            notes[key] = json.loads(json.dumps(footing[key]))
    return n_refused, notes


@contextlib.contextmanager
def _out_root_lock(out_root: Path):
    """ONE exclusive advisory lock per out-root (``<out-root>/.analyze.lock``, blocking ``fcntl.flock``) around the fold and the frame
    read: two invocations under one audit -- disjoint ``--run-id`` selections, an operator beside the daemon -- used to write the same
    ``rows.parquet.tmp`` and read each other's half-published frame; the daemon's per-cell mirror locks never covered the common root.
    Advisory, so it costs nothing when no one else is folding."""
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / ".analyze.lock", "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class AnalysisTerminated(BaseException):
    """SIGTERM reached the registry-worklist analysis while it held claims: raised from the signal handler so the
    claim cleanup below runs instead of the interpreter ending with the claims held (the default disposition)."""


def _raise_terminated(signum, frame) -> None:
    raise AnalysisTerminated(f"signal {signum} during the registry-worklist analysis")


def _release_claims(registry_wl, rows: Sequence[Mapping[str, Any]]) -> int:
    """Release every claim this invocation took: a failure between the durable claims and the filed judgments used to leave the runs
    claimed by this session -- excluded from every later grader (grade_daemon.pending_rows) and outside the quarantine's release paths
    -- until a hand abandonment. Best effort per row (a release that fails is printed, never raised: the original exception is what the
    caller re-raises); returns the number released."""
    n = 0
    for row in rows:
        t0 = time.monotonic()
        try:
            if registry_wl.release(row):
                n += 1
        except Exception as exc:  # noqa: BLE001 -- named, the original failure stays the one raised
            print(f"[registry] could not release the claim on {row.get('run_id')!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
        # pending.release takes the harvester lock and runs one full harvester cycle per run (no batch form); say how long
        print(f"[registry] release lock held {time.monotonic() - t0:.1f} s ({row.get('run_id')!r}; one harvester cycle per release -- "
              "pending.release has no batch form)", file=sys.stderr, flush=True)
    print(f"[registry] released {n} of {len(rows)} claim(s) after a failed fold/analysis -- another session may retry", file=sys.stderr)
    return n


#: the registry lifecycle states in which a run has FINISHED GENERATING (its store rows are complete): a borrowed base-screen anchor
#: must be in one of these AND current -- a `running` / `queued` run has no complete rows to lend, whatever v_run_current says
_BORROW_GENERATED_STATES = ("generated", "judging", "judged", "ingested")
#: the u0 view's tag segment: `<model_id>_u0` is the unsteered wave every steered view of the model borrows its base from
_U0_VIEW = "u0"
#: `_analysis_under_lock(borrowed_base=...)` sentinel: "not resolved yet -- resolve it here" (the registry path resolves the borrow
#: BEFORE its claims and passes the result, None included, so the sentinel is never a plain None)
_BORROW_UNRESOLVED = object()


def _petri_eval_id(behavior: str) -> str:
    """The registry eval id of one Petri behaviour (the inverse of :func:`_behaviour_of`)."""
    return f"petri_{behavior}"


def _registry_db_path(args: argparse.Namespace) -> str:
    """The registry database the drivers read: ``--registry-root``, else ``REGISTRY_ROOT``, else the default root -- the same
    resolution as :class:`sisteer.registry.worklist.RegistryWorklist`."""
    from sisteer.registry import db as RDB
    from sisteer.registry import worklist as RW
    root = os.path.abspath(getattr(args, "registry_root", None) or os.environ.get(RW.ROOT_ENV) or RDB.DEFAULT_ROOT)
    return RDB.db_path(root)


def _base_screen_anchor_rows(conn, campaign: str, model_id: str, eval_ids: Sequence[str]) -> List[Dict[str, Any]]:
    """Every registered run of (campaign, model) for ``eval_ids`` with its currency verdict (``v_run_current``: ``is_current`` /
    ``why`` / ``selection_state`` / ``held_out``) -- the candidate anchors and their retired siblings alike, so a refusal can name
    what exists and why it does not count."""
    cols = BORROW_ANCHOR_ROW_FIELDS                      # every column is classified identity / recorded-only
    marks = ", ".join("?" for _ in eval_ids)
    sql = (f"SELECT {_anchor_select_sql(cols)} FROM run r JOIN v_run_current vc ON vc.run_id = r.run_id "
           f"WHERE r.campaign = ? AND r.model_id = ? AND r.eval_id IN ({marks}) ORDER BY r.eval_id, r.run_id")
    return [dict(zip(cols, row)) for row in conn.execute(sql, (campaign, model_id, *[str(e) for e in eval_ids]))]


#: the steered view's own store holds the IMMUTABLE snapshot of the borrowed source: `<out_root>/.borrowed_base/<source audit>/<none run
#: key>.<generation digest[:12]>/` -- a dot-dir sibling of cells/, so no walk of cells/ (consolidate, bind_cell_dirs, the tombstone
#: reader) ever sees it; one directory per source GENERATION, written once through a temp dir + rename, verified on every reuse
BORROWED_BASE_DIR = ".borrowed_base"
BORROWED_SOURCE_JSON = "source.json"
#: the source cell files a borrow needs on disk -- their absence is the "absent or pruned" refusal, by name
_BORROW_SOURCE_FILES = ("manifest.json", "results.json", "rows.jsonl")
#: WHY the source can be read in place (read-only facts from scripts/registry/grade_daemon.py at a362841fd and the live daemon's event
#: log): a Petri run's store_uri is the grading host's local petri_native path the drive wrote; grade_daemon.mirror_targets mirrors
#: a row only when its substrate has an ssh alias AND the store map RELOCATED its path ("a path the map left unchanged is read where it is"),
#: and mirror_prune_enqueue queues for pruning only directories strictly under a known mirror root (_under_root) -- the live log's 705 mirror
#: and 171 mirror_pruned events name no petri_native path (every pruned dir is under ~/sisteer-grade-mirror/<cluster>/exp/misalign/...). So
#: nothing stages or removes the source: it is a read-in-place dependency, and the checks below make it an explicit, verified one.
_BORROW_SOURCE_DEPENDENCY_NOTE = (
    "read in place from the grading-host-local Petri store the registry's current `none` runs name, and snapshotted under this view's store "
    "before the analysis; the grade daemon neither mirrors nor prunes grading-host-local Petri stores (mirror_targets relocates only "
    "store-mapped cluster paths; mirror_prune_enqueue queues only directories under a mirror root), so an absent source is a staging "
    "defect refused by name, never a silent stale borrow")


#: the recorded-only ALLOWLIST: every `run` / `v_run_current` column the borrow's SELECTs read is an IDENTITY field -- re-read and
#: compared at BOTH checkpoints (verify_borrowed_anchor_identity: right after the snapshot verification, and immediately before the
#: filing) -- UNLESS it is named here with the one-line reason it is explanatory rather than load-bearing. The identity set is DERIVED from the
#: SELECT list below, never hand-kept: a column added to a SELECT is identity by default (a hand omission has nowhere to live), and the coverage
#: test pins SELECT columns == identity ∪ allowlist. An earlier hand-kept identity tuple left out campaign / model_id (only query predicates at
#: selection) and filed model_revision as recorded-only: a same-id corrective FULL `run` record (reducer.fold applies every non-event-owned field
#: of a later full record) could restate any of them with the completion untouched and both checkpoints passed.
BORROW_ANCHOR_RECORDED_ONLY_REASONS: Mapping[str, str] = {
    "n_items_generated": "a count of the run's own generation; the hand-off re-derives coverage from the frame, the borrow selects nothing by it",
    "n_items_judged": "the judged count the judgment stamped; informational",
    "judge_id": "which judge filed the run's judgment; the borrow reads scores and footing, never a judgment",
    "view_legacy": "the tag <model_id>_<view>: a spelling of model_id (identity) plus the view already in the source audit's name",
    "ingest_seq": "the fold order of the registry record; informational",
    "why": "the explanation of a non-current verdict; the verdict itself (is_current, selection_state) is identity",
    "held_out": "folded into is_current (a held-out run reads is_current 0); informational beside the verdict",
    "generated_utc": "when the run generated; the generation is bound by its completion sha, never by its time",
}
BORROW_ANCHOR_RECORDED_ONLY_FIELDS: Tuple[str, ...] = tuple(BORROW_ANCHOR_RECORDED_ONLY_REASONS)
#: EVERY column the anchor row query reads (run r JOIN v_run_current vc), in SELECT order -- the one list the identity derives from
BORROW_ANCHOR_ROW_FIELDS: Tuple[str, ...] = (
    "run_id", "campaign", "model_id", "model_revision", "eval_id", "producer_object_key_legacy", "substrate", "store_uri", "lifecycle_state",
    "is_current", "selection_state", "store_completion_artifact_uri", "store_completion_sha256",
    "n_items_generated", "n_items_judged", "judge_id", "view_legacy", "ingest_seq", "why", "held_out", "generated_utc")
#: the IDENTITY fields -- DERIVED: every row column not on the recorded-only allowlist (run_id, campaign, model_id, model_revision, eval_id, the
#: `none` cell, substrate + store_uri, lifecycle_state (usable: generated or later), is_current + selection_state (the currency verdict), the
#: registered completion artifact + sha). Compared for equality at both checkpoints, except lifecycle_state (still generated-or-later) and
#: is_current (1); a NULL model_revision (the drive's registration stated none) is itself part of the tuple -- a later record stating one changes it.
BORROW_ANCHOR_IDENTITY_FIELDS: Tuple[str, ...] = tuple(f for f in BORROW_ANCHOR_ROW_FIELDS if f not in BORROW_ANCHOR_RECORDED_ONLY_REASONS)
#: the judged view's own identity query (its registered Petri runs under the out-root): a subset of the row fields, pinned by the coverage test
#: EVERY registry-row field the Petri FILING path reads to route a run's judged rows and file its judgment -- the ONE accessor
#: `filing_field` refuses any other field, so this tuple is exactly the set the code CAN read: the run (run_id), its behaviour (eval_id
#: -> petri_<behaviour> / petri_pooled, _behaviour_of), its cell (producer_object_key_legacy -> the harness id, _cell_of / _harness_cell_of) and
#: the object the key names (object_id), the generation coverage the hand-off checks against the frame (n_items_generated), the scope's campaign
#: and store (worklist.scope / store_path_of) and the claim's judge (required_judge_id, a v_pending_judgment column). The compared in-view identity
#: DERIVES from this set (BORROW_IN_VIEW_IDENTITY_FIELDS), so a field the filing reads can never be left uncompared -- an earlier `anchor identity
#: ∩ in-view SELECT` rule left eval_id and producer_object_key_legacy out, and a same-id full correction could re-route a run's judged rows unseen.
FILING_ROW_FIELDS: Tuple[str, ...] = ("run_id", "eval_id", "producer_object_key_legacy", "object_id", "campaign", "store_uri", "substrate",
                                      "n_items_generated", "required_judge_id")
#: EVERY registry-row field the CLAIM routes on: RegistryWorklist.claim takes (run_id, required_judge_id), and v_pending_judgment
#: derives required_judge_id from the run's CURRENT instrument (instrument.judge_id via run.instrument_id, the instrument-judge rows
#: the worklist reads) -- so the claim routes on the run, its instrument and the judge that instrument designates.
#: THE PRINCIPLE: the compared identity includes every field the FILING or the CLAIM routes on; "pending-view-only" is not an exclusion
#: reason. A same-id corrective full run record switching a selected run to another registered instrument / judge with every
#: other field unchanged left the retained row carrying the OLD judge while the pending view's required_judge_id moved, and the claim took the
#: old (run_id, judge_id) pair: the grader filed under the old judge, the new required judge stayed ungraded.
CLAIM_ROW_FIELDS: Tuple[str, ...] = ("run_id", "instrument_id", "required_judge_id", "instrument_judge_id")
#: identity fields that are SELECTION facts, not `run` columns: bound by the worklist row the claim is made
#: from, rendered NULL by the SELECT renderer when read from `run` (so a run-table read never overwrites the bound value), and VERIFIED against
#: the pending view at every identity read (the worklist's agreement, the borrow guard, the pre-filing rechecks): the (run, selected judge) pair
#: must still be pending for that judge (pending_for terms -- _pending_judges), and under the DEFAULT worklist (the instrument judge) the run's
#: CURRENT instrument must still designate it (the rule above). An explicit non-instrument `--judge` selection is a supported pairing
#: (pending_for pairs any named judge) and is never refused for not being the instrument's judge.
SELECTION_FIELDS: Tuple[str, ...] = ("required_judge_id",)
SELECTION_FIELD_REASONS: Mapping[str, str] = {
    "required_judge_id": ("the SELECTED judge: v_pending_judgment.required_judge_id of the worklist row -- the instrument's judge under the "
                          "default worklist, the named judge under --judge -- bound by the claim; verified at every identity read to still be a "
                          "pending (run, judge) pair, and under the default worklist to still be the judge the current instrument designates"),
}
#: identity fields that are not `run` columns, each with the SQL expression (over the run alias `r`) the identity read recomputes it by -- the
#: pending view's own derivation -- rendered by the ONE SELECT renderer (_anchor_select_sql) and compared at every point, never excluded
#: (this replaced an earlier exclusion of required_judge_id as pending-view-only; the derived field is named for what it is -- the
#: instrument's designated judge -- beside the bound selection field required_judge_id)
FILING_DERIVED_FIELDS_SQL: Mapping[str, str] = {
    "instrument_judge_id": "(SELECT i.judge_id FROM instrument i WHERE i.instrument_id = r.instrument_id)",
}
FILING_DERIVED_REASONS: Mapping[str, str] = {
    "instrument_judge_id": ("the judge the run's CURRENT instrument designates (instrument.judge_id via run.instrument_id), exactly as "
                            "v_pending_judgment derives the instrument-judge rows' required_judge_id; compared for equality in every mode, and "
                            "under the default worklist the bound required_judge_id must equal it -- a run switched to another instrument / "
                            "judge after the worklist read refuses by name instead of filing under the old judge"),
}
#: in-view fields NOT compared before the filing, with the reason: the filing itself advances a pending run's lifecycle and
#: its currency / roster verdict are the worklist's business (which runs to judge), not the borrow's
BORROW_IN_VIEW_UNCOMPARED_REASONS: Mapping[str, str] = {
    "lifecycle_state": "the filing advances a pending in-view run's lifecycle (generated -> judging -> judged); not an identity change",
    "is_current": "a pending in-view run's currency is the worklist's business (which runs to judge), not the borrow's",
    "selection_state": "the roster verdict of a pending in-view run; the worklist's business, not the borrow's",
}
#: the judged view's OWN runs' identity -- DERIVED: every field the filing path reads (FILING_ROW_FIELDS) and every field the claim routes on
#: (CLAIM_ROW_FIELDS: run, instrument, the judge it designates) plus every anchor identity field (model / revision / store /
#: completion) less the reasoned uncompared set; a field that is not a `run` column is recomputed by FILING_DERIVED_FIELDS_SQL, never dropped.
#: Captured per selected run at the worklist read (anchors.in_view_runs_at_selection), compared with the borrow's SELECT before any claim and
#: re-read with the one identity function immediately before the first write (anchors.in_view_recheck), so a same-id corrective full record
#: restating ANY routing field of a selected run -- its cell, behaviour, store, completion, instrument or judge -- refuses by name.
BORROW_IN_VIEW_IDENTITY_FIELDS: Tuple[str, ...] = tuple(dict.fromkeys(
    list(FILING_ROW_FIELDS) + list(CLAIM_ROW_FIELDS)
    + [f for f in BORROW_ANCHOR_IDENTITY_FIELDS if f not in BORROW_IN_VIEW_UNCOMPARED_REASONS]))
assert set(FILING_DERIVED_FIELDS_SQL) == set(FILING_DERIVED_REASONS) and set(FILING_DERIVED_FIELDS_SQL) <= set(BORROW_IN_VIEW_IDENTITY_FIELDS)
assert set(SELECTION_FIELDS) == set(SELECTION_FIELD_REASONS) and set(SELECTION_FIELDS) <= set(BORROW_IN_VIEW_IDENTITY_FIELDS)
assert not (set(SELECTION_FIELDS) & set(FILING_DERIVED_FIELDS_SQL))
#: the judged view's identity query: the compared identity plus view_legacy (recorded-only: the tag the source audit's name derives from)
BORROW_IN_VIEW_FIELDS: Tuple[str, ...] = BORROW_IN_VIEW_IDENTITY_FIELDS + ("view_legacy",)


def filing_field(row: Mapping[str, Any], field: str) -> Any:
    """The ONE accessor for a registry row on the filing path: every field the hand-off reads to route a run's judged rows or file
    its judgment comes through here, and a field outside FILING_ROW_FIELDS is a refusal -- so the set the compared in-view identity derives
    from is exactly the set the code can read (the coverage test greps the filing functions for any other access)."""
    if field not in FILING_ROW_FIELDS:
        raise DriverError(f"the filing path read registry-row field {field!r} outside FILING_ROW_FIELDS {FILING_ROW_FIELDS}; add it there (it "
                          "then joins the compared in-view identity) -- never read a routing field the identity does not compare")
    return row.get(field)
#: the v_run_current columns among them (every other field is a `run` column)
_BORROW_ANCHOR_VC_FIELDS = ("is_current", "why", "selection_state", "held_out")


def _anchor_select_sql(fields: Sequence[str]) -> str:
    """The SELECT list for ``fields`` over ``run r`` (JOIN ``v_run_current vc`` for the currency columns) -- built from the constants, never
    hand-copied; a DERIVED field (FILING_DERIVED_FIELDS_SQL) is recomputed from the current `run` row by its expression, aliased to
    its name, so every identity read -- the worklist's, the borrow's in-view SELECT, the rechecks -- derives it the same way."""
    return ", ".join((FILING_DERIVED_FIELDS_SQL[f] + " AS " + f) if f in FILING_DERIVED_FIELDS_SQL
                     else ("NULL AS " + f) if f in SELECTION_FIELDS                    # a selection fact rides the worklist row
                     else ("vc." if f in _BORROW_ANCHOR_VC_FIELDS else "r.") + f for f in fields)


def _default_judge_worklist(args: argparse.Namespace) -> bool:
    """True for the DEFAULT worklist (each run's instrument judge), False under an explicit ``--judge`` selection -- the same source
    RegistryWorklist reads its judges from."""
    return not getattr(args, "judge", None)


def _pending_judges(conn, run_ids: Sequence[str]) -> Dict[str, List[str]]:
    """The judges each run is CURRENTLY a pending (run, judge) pair for, in pending_for terms -- v_pending_judgment's rows for the run ids, one
    read-only query: what every identity read verifies the bound required_judge_id against."""
    if not run_ids:
        return {}
    marks = ", ".join("?" for _ in run_ids)
    out: Dict[str, List[str]] = {}
    for rid, judge in conn.execute(f"SELECT run_id, required_judge_id FROM v_pending_judgment WHERE run_id IN ({marks})", [str(x) for x in run_ids]):
        out.setdefault(str(rid), []).append(str(judge))
    return out


def identity_diff_lines(expected: Mapping[str, Any], now: Mapping[str, Any], fields: Sequence[str], *, default_judge: bool = True) -> List[str]:
    """The ONE field-by-field identity comparison every checkpoint prints: `<field>: <before> -> <after>` for each field of
    ``fields`` whose value differs between the tuple held (``expected``) and the row read now (``now``). A SELECTION field (required_judge_id)
    is verified rather than compared: the bound judge must still be a pending (run, judge) pair for this run (``now`` carries the
    pending judges, ``pending_judge_ids``, when the identity read attached them) and, under the default worklist (``default_judge``), the judge
    the run's current instrument designates -- "required_judge_id: <selected> -> current instrument <id> designates <judge>" -- so a run switched
    to another registered instrument / judge is refused in those words at the worklist's two-read agreement, the borrow guard and the pre-filing
    rechecks alike, while an explicit --judge selection is held only to its pair staying pending. No bound judge (a from-disk tuple) verifies
    nothing."""
    out: List[str] = []
    for f in fields:
        if f in SELECTION_FIELDS:
            bound = expected.get(f)
            if bound is None:
                continue
            pend = now.get("pending_judge_ids")
            if pend is not None and str(bound) not in [str(j) for j in pend]:
                out.append(f"{f}: {str(bound)!r} -> no longer a pending (run, judge) pair for this run (pending now for {sorted(str(j) for j in pend)})")
            if default_judge and str(now.get("instrument_judge_id")) != str(bound):
                out.append(f"{f}: {str(bound)!r} -> current instrument {now.get('instrument_id')} designates {str(now.get('instrument_judge_id'))!r}")
            continue
        was, cur = expected.get(f), now.get(f)
        if str(was) != str(cur):
            out.append(f"{f}: {str(was)[:48]!r} -> {str(cur)[:48]!r}")
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_generation(cell_dir: Path) -> Dict[str, Any]:
    """The source cell dir's GENERATION: every regular file's sha256 / size / mtime, and one digest over the (name, sha256) list -- what the
    snapshot is named by and verified against."""
    files: Dict[str, Dict[str, Any]] = {}
    for p in sorted(x for x in Path(cell_dir).iterdir() if x.is_file()):
        st = p.stat()
        files[p.name] = {"sha256": _sha256_file(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    digest = hashlib.sha256("\n".join(f"{n}:{v['sha256']}" for n, v in files.items()).encode("utf-8")).hexdigest()
    return {"digest": digest, "files": files}


def _verify_source_binding(per_b: Mapping[str, Mapping[str, Any]], cell_dir: Path, srm, where: str) -> Dict[str, Any]:
    """Bind EVERY selected current `none` run to the on-disk source generation: the run's registered completion
    artifact must be this cell's results.json (relocated through the store map like every path), and its store_completion_sha256 must equal
    the sha256 of that file as it stands -- the Petri drive registers `generated` with exactly that pair, so a reused cell dir holding another
    generation (or an older consolidation) never passes as the run the registry made current. A run without its completion identity is not a
    borrowable baseline. Refusals name the behaviour, the run and both digests."""
    results_path = (Path(cell_dir) / "results.json").resolve()
    disk_sha = _sha256_file(results_path)
    binding: Dict[str, Any] = {}
    for b, r in sorted(per_b.items()):
        uri, sha = r.get("store_completion_artifact_uri"), r.get("store_completion_sha256")
        if not uri or not sha:
            raise DriverError(f"{where}: behaviour {b!r}: the current `none` run {r['run_id']!r} carries no "
                              f"store_completion_{'artifact_uri' if not uri else 'sha256'} -- the borrowed source cannot be bound to the run's "
                              "generation; refusing (a registration without its completion identity is not a borrowable baseline)")
        from sisteer.registry import pending as RP
        located = Path(RP.store_path_of({"store_uri": str(uri), "substrate": r.get("substrate")}, srm)).expanduser().resolve()
        if located != results_path:
            raise DriverError(f"{where}: behaviour {b!r}: the current `none` run {r['run_id']!r} completed artifact {uri!r} (-> {located}), "
                              f"not the source cell's {results_path}; refusing")
        if str(sha) != disk_sha:
            raise DriverError(f"{where}: behaviour {b!r}: STALE or foreign source -- the registry's current `none` run {r['run_id']!r} completed "
                              f"results.json sha256 {str(sha)[:12]}..., the disk holds {disk_sha[:12]}... at {results_path}; the on-disk cell is "
                              "not the generation the registry made current; refusing (never a silent stale borrow)")
        binding[b] = {"run_id": r["run_id"], "store_completion_artifact_uri": str(uri), "store_completion_sha256": str(sha),
                      "generated_utc": r.get("generated_utc")}
    return {"results_json_sha256_on_disk": disk_sha, "per_behaviour": binding}


def _verify_snapshot_binding(snapshot_cell_dir: Path, generation: Mapping[str, Any], binding: Mapping[str, Any], where: str,
                             snapshot_label: str) -> Dict[str, Any]:
    """ONE check for the fresh-snapshot and the reuse path: the snapshot's results.json sha256 must equal the
    binding's live sha (= every selected current run's registered completion) AND the generation's recorded results.json hash, and the
    snapshot's rows.jsonl must equal the generation's recorded rows hash -- so the rows the analysis reads, the footing it binds and the run
    the registry made current are ONE generation. Any disagreement refuses by name (the borrow is resolved before any claim); returns every
    verified digest for the anchors."""
    snap_results = _sha256_file(Path(snapshot_cell_dir) / "results.json")
    snap_rows = _sha256_file(Path(snapshot_cell_dir) / "rows.jsonl")
    gen_files = generation.get("files") or {}
    gen_results = str((gen_files.get("results.json") or {}).get("sha256"))
    gen_rows = str((gen_files.get("rows.jsonl") or {}).get("sha256"))
    bind_sha = str(binding["results_json_sha256_on_disk"])
    runs_sha = {b: str(x["store_completion_sha256"]) for b, x in (binding.get("per_behaviour") or {}).items()}
    problems: List[str] = []
    if snap_results != bind_sha:
        problems.append(f"snapshot results.json {snap_results[:12]} != the binding's live results.json {bind_sha[:12]}")
    if snap_results != gen_results:
        problems.append(f"snapshot results.json {snap_results[:12]} != the generation's recorded {gen_results[:12]}")
    if snap_rows != gen_rows:
        problems.append(f"snapshot rows.jsonl {snap_rows[:12]} != the generation's recorded {gen_rows[:12]}")
    off = {b: s[:12] for b, s in runs_sha.items() if s != snap_results}
    if off:
        problems.append(f"selected current run(s) registered another completion than the snapshot holds: {off}")
    if problems:
        raise DriverError(f"{where}: the borrowed snapshot {snapshot_label} does not agree with the binding / generation ({'; '.join(problems)}) "
                          "-- refusing before any claim; never a stale or mixed borrow")
    return {"binding_results_json_sha256": bind_sha, "generation_results_json_sha256": gen_results, "generation_rows_jsonl_sha256": gen_rows,
            "snapshot_results_json_sha256": snap_results, "snapshot_rows_jsonl_sha256": snap_rows,
            "selected_runs_completion_sha256": runs_sha, "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _live_results_sha_or_refuse(cell_dir: Path, binding: Mapping[str, Any], where: str, stage: str) -> str:
    """The LIVE source results.json sha256 right now; anything but the binding's refuses by name (a concurrent re-score --
    cmd_rescore -> petri_rescore.refresh_results, which is not under the steered out-root's lock -- between the binding and the copy)."""
    live = _sha256_file(Path(cell_dir) / "results.json")
    bind_sha = str(binding["results_json_sha256_on_disk"])
    if live != bind_sha:
        raise DriverError(f"{where}: source changed during binding/snapshot ({stage}): {bind_sha[:12]} -> {live[:12]} at "
                          f"{Path(cell_dir) / 'results.json'}; the source cell was rewritten (a re-score?) while it was being bound; refusing -- "
                          "a retry binds the new generation once the registry makes it current")
    return live


def _borrowed_base_snapshot(out_root: Path, source_root: Path, source_record: Mapping[str, Any], cell_dir: Path, cell_key: str,
                            spec: cm.PetriSpec, generation: Mapping[str, Any], binding: Mapping[str, Any], where: str
                            ) -> Tuple[Path, Dict[str, Any], bool, Dict[str, Any]]:
    """An IMMUTABLE local snapshot of the borrowed source cell under the steered view's own store: the source
    record (cells.json), every file of the source cell dir, and `rows.parquet` folded from THAT copy's rows.jsonl minus its refusal-sidecar
    tombstones -- the same fold `consolidate` runs, over one cell -- plus `source.json` naming the source paths, the generation (file
    digests), the registry binding, the verification and the time. Named by the generation digest, written once through a temp dir and an
    atomic rename; an existing snapshot is verified file by file against its recorded generation AND against the binding
    (:func:`_verify_snapshot_binding`) before reuse (a changed file is a refusal, the snapshot is never rewritten). The live source's
    results.json is re-read immediately before the copy and again after the copy and fold -- a change refuses "source changed during
    binding/snapshot: <before> -> <after>" and the temp dir is removed -- and the fresh snapshot passes the same snapshot-vs-binding
    check before it is renamed into place. Returns (snapshot root, source.json, reused, verification)."""
    base = Path(out_root) / BORROWED_BASE_DIR / str(source_record["audit"])
    final = base / f"{cell_key}.{str(generation['digest'])[:12]}"
    src_json = final / BORROWED_SOURCE_JSON
    if final.exists():
        if not src_json.exists() or not (final / "rows.parquet").exists():
            raise DriverError(f"{where}: snapshot {final} exists without its {BORROWED_SOURCE_JSON} / rows.parquet -- a half-written snapshot; "
                              "remove it deliberately and retry")
        recorded = json.loads(src_json.read_text(encoding="utf-8"))
        for name, meta in ((recorded.get("generation") or {}).get("files") or {}).items():
            p = final / "cells" / cell_key / name
            if not p.exists() or _sha256_file(p) != str(meta.get("sha256")):
                raise DriverError(f"{where}: snapshot {final} no longer matches its recorded generation (file {name}: "
                                  f"{'missing' if not p.exists() else 'sha256 differs'}) -- a borrowed-base snapshot is immutable; refusing")
        verification = _verify_snapshot_binding(final / "cells" / cell_key, generation, binding, where, str(final))
        verification["live_results_json_sha256_at_reuse"] = _live_results_sha_or_refuse(cell_dir, binding, where, "at reuse")
        return final, recorded, True, verification
    sha_before = _live_results_sha_or_refuse(cell_dir, binding, where, "before the copy")
    base.mkdir(parents=True, exist_ok=True)
    tmp = base / f".{final.name}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        (tmp / "cells" / cell_key).mkdir(parents=True)
        for name in generation["files"]:
            shutil.copy2(Path(cell_dir) / name, tmp / "cells" / cell_key / name)
        shutil.copy2(Path(source_root) / CELLS_JSON, tmp / CELLS_JSON)
        for name, meta in generation["files"].items():      # the copy IS the generation, or the source moved under us: refuse, never mix
            if _sha256_file(tmp / "cells" / cell_key / name) != str(meta["sha256"]):
                raise DriverError(f"{where}: the source cell dir {cell_dir} changed while it was being snapshotted ({name}); refusing -- retry")
        tombstones = collect_auditor_refusal_tombstones(tmp, spec)
        cs.consolidate(tmp, skip_incomplete=False, exclude_keys_by_cell=tombstones)
        sha_after = _live_results_sha_or_refuse(cell_dir, binding, where, "after the copy")
        verification = _verify_snapshot_binding(tmp / "cells" / cell_key, generation, binding, where, "(being taken)")
        verification.update({"live_results_json_sha256_before_copy": sha_before, "live_results_json_sha256_after_copy": sha_after})
        source: Dict[str, Any] = {
            "snapshot_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_root": str(source_root), "source_cell_dir": str(cell_dir), "petri_run_key": cell_key,
            "source_audit": str(source_record["audit"]), "generation": dict(generation), "registry_binding": dict(binding),
            "verification": dict(verification),
            "fold": "capeval.store.consolidate over this snapshot's one cell dir, minus its refusal-sidecar tombstones",
            "dependency": _BORROW_SOURCE_DEPENDENCY_NOTE,
        }
        (tmp / BORROWED_SOURCE_JSON).write_text(json.dumps(source, indent=1, sort_keys=True), encoding="utf-8")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)               # a refused or interrupted snapshot leaves nothing behind
        raise
    try:
        os.rename(tmp, final)
    except OSError:
        if final.exists():                                   # a concurrent analyzer landed the same generation first: verify and use it
            shutil.rmtree(tmp, ignore_errors=True)
            return _borrowed_base_snapshot(out_root, source_root, source_record, cell_dir, cell_key, spec, generation, binding, where)
        raise
    return final, source, False, verification


def _root_parquet_comparison(root_parquet: Path, fold_parquet: Path, cell: str) -> Dict[str, Any]:
    """Whether the source root's own consolidated rows for ``cell`` (when a root parquet exists) equal the fold of the cell's evidence --
    recorded, never relied on: the borrow publishes the fold."""
    import pandas as pd
    if not Path(root_parquet).exists():
        return {"present": False, "none_rows_equal_to_fold": None, "n_rows_root": None, "n_rows_fold": None}
    key = ["arm_name", "task", "item_id", "metric"]
    a = pd.read_parquet(root_parquet)
    a = a[a["arm_name"].astype(str) == cell].sort_values(key).reset_index(drop=True)
    b = pd.read_parquet(fold_parquet)
    b = b[b["arm_name"].astype(str) == cell].sort_values(key).reset_index(drop=True)
    cols = [c for c in ("arm_name", "task", "item_id", "metric", "value", "engine", "run_id", "sweep", "model_id", "model_revision")
            if c in a.columns and c in b.columns]
    equal = len(a) == len(b)
    if equal:
        try:
            pd.testing.assert_frame_equal(a[cols], b[cols], check_dtype=False)
        except AssertionError:
            equal = False
    return {"present": True, "none_rows_equal_to_fold": bool(equal), "n_rows_root": int(len(a)), "n_rows_fold": int(len(b)),
            "note": "informational: the borrow publishes the fold of the source cell's own evidence, never the source root's parquet"}


def base_screen_borrow(args: argparse.Namespace, spec: cm.PetriSpec, out_root: Path,
                       record: Mapping[str, Any], *,
                       selected_rows: Optional[Sequence[Mapping[str, Any]]] = None) -> Optional["petri_stats.BorrowedBase"]:
    """The u0 wave's undefended base for a STEERED view, or None when the view carries its own (the Petri analogue of the
    borrowed contrast anchor).

    A steered view -- every non-u0 view of a front-door target -- carries NO `none` cell (the unsteered model is not re-run, the
    baseline is the u0 wave's), so ``petri_stats.analyze`` refused its frame at the §8 gate (the 02:16Z s3r dispatch). Here the
    base-elicitability screen and the undefended footing are taken from the u0 wave's `none` Petri runs of the SAME model and
    campaign, read from the registry READ-ONLY (``db.reading``; never written, never locked):

    1. the view's own identity -- every registered Petri run whose store lies under ``out_root`` names the judged view's ``campaign``,
       ``model_id`` (the front-door key) and tag (``view_legacy`` = ``<model_id>_<view>``); the u0 audit is the view's audit with the
       view segment replaced by ``u0`` (a u0 view without `none` rows returns None, so the standing gate refusal stands as it is);
       in registry mode (``selected_rows``: the worklist's rows) every selected run must be in that set WITH
       every in-view identity field equal -- a same-id correction folded between the worklist read and this read that moved a run's
       store outside the root or its eval off Petri (presence), or its cell / behaviour / store / completion while the store stayed
       under the root (identity), refuses BY NAME here, before any claim, instead of the filing loop judging the run under the cell /
       behaviour the worklist read; the worklist rows are the identity baseline recorded and re-read before the filing;
    2. the anchors -- the `none` runs of (campaign, model) for the view's evaluations (``petri_<behaviour>``) with their ``v_run_current``
       verdict; the CURRENT ones (``is_current = 1``) that have finished generating name, through their store paths (relocated through
       ``--store-root-map`` like every worklist row), exactly ONE store whose frozen record is the u0 audit's; under that record's `none`
       cell dir exactly one such run per behaviour -- a missing, retired (``why`` names the rule) or ambiguous baseline refuses BY NAME
       (behaviour, evaluation, every run on record with its lifecycle and currency), never a smaller analysis;
    3. the source store, read as it stands -- its ``rows.parquet`` (the source's own analysis consolidates it; this one never writes
       another standup's store), its incomplete map (a `none` recorded INCOMPLETE refuses) and the `none` cell's audited footing through
       the same ``collect_audited_footing`` walk the source's analysis runs (read-only, bound to the source record).

    The in-view base always wins: a record or manifest carrying `none` returns None before the registry is opened. The returned
    :class:`petri_stats.BorrowedBase` binds the rows to the SOURCE record (its audit; the in-view spec's model + revision, so the rows'
    identity is verified against THIS manifest's base model) and carries the registry evidence (anchor run ids, lifecycle, currency,
    the siblings not used, the store paths) for the analysis to record under ``borrowed_base.anchors``."""
    served = {str(c["cell_id"]) for c in (list(record.get("canonical") or []) + list(record.get("controls") or []))}
    if dc.BASE_CELL_ID in served or dc.BASE_CELL_ID in [str(c) for c in spec.cells]:
        return None                     # the view carries its own undefended base (u0 / legacy): in-view wins, byte-identical
    from sisteer.registry import db as RDB
    from sisteer.registry import pending as RP
    where = f"base-screen borrow for {spec.audit!r}"
    why_borrow = ("the view carries no undefended `none` cell (the user's ruling D1152/D1154: the baseline is the u0 wave's) and the "
                  "base-elicitability screen is a GATE (plan §8)")
    db_path = _registry_db_path(args)
    if not os.path.exists(db_path):
        raise DriverError(f"{where}: {why_borrow}; the borrow needs the registry's currency verdict (v_run_current) but there is no "
                          f"registry database at {db_path}; refusing")
    try:
        srm = RP.parse_store_root_map(getattr(args, "store_root_map", None)) or None
    except ValueError as exc:
        raise DriverError(f"{where}: {exc}") from None
    base_root = Path(out_root).expanduser().resolve()

    def _located(row: Mapping[str, Any]) -> Optional[Path]:
        sp = RP.store_path_of(row, srm)
        return Path(sp).expanduser().resolve() if sp else None

    behaviors = [str(b) for b in spec.behaviors]
    eval_of = {_petri_eval_id(b): b for b in behaviors}
    with RDB.reading(db_path) as conn:
        # 1. the judged view's registry identity: every registered Petri run whose store lies under THIS out-root
        id_cols = BORROW_IN_VIEW_FIELDS                   # derived from the constant, never hand-typed
        # rendered by the one SELECT renderer, so the derived required_judge_id (the current instrument's judge) is read here too
        petri_rows = [dict(zip(id_cols, r)) for r in conn.execute(
            f"SELECT {_anchor_select_sql(id_cols)} FROM run r WHERE r.eval_id LIKE 'petri_%'")]
        in_view = []
        for r in petri_rows:
            d = _located(r)
            if d is not None and (d == base_root or base_root in d.parents):
                in_view.append(r)
        if not in_view:
            raise DriverError(f"{where}: {why_borrow}; no registered Petri run has its store under {base_root}, so the judged view's model "
                              "and campaign cannot be read from the registry; refusing")
        # In registry mode every run the worklist SELECTED must be in the judged view's set read here. A
        # same-id correction folded between the worklist read and this SELECT -- a run_patch moving a run's store_uri outside the out-root, a
        # full record changing its eval to a non-Petri one -- dropped that run from in_view silently while its siblings kept the borrow valid;
        # in_view_runs_at_selection and the filing recheck then covered the survivors only, and the filing loop still filed every ORIGINAL
        # worklist run under the cell / behaviour the worklist read. Refused BY NAME, before any claim (the borrow precedes the claims).
        # The borrow receives the worklist ROWS -- presence is checked first (below), then IDENTITY.
        if selected_rows is not None:
            present = {str(r["run_id"]): r for r in in_view}
            by_id = {str(x["run_id"]): x for x in selected_rows}
            selected = sorted(by_id)
            dropped = [rid for rid in selected if rid not in present]
            if dropped:
                marks = ", ".join("?" for _ in dropped)
                now_rows = {str(r[0]): {"eval_id": r[1], "store_uri": r[2], "substrate": r[3]} for r in conn.execute(
                    f"SELECT run_id, eval_id, store_uri, substrate FROM run WHERE run_id IN ({marks})", dropped)}
                parts: List[str] = []
                for rid in dropped:
                    cur = now_rows.get(rid)
                    if cur is None:
                        parts.append(f"{rid}: no longer in `run`")
                        continue
                    d = _located(cur)
                    why_out: List[str] = []
                    if d is None or not (d == base_root or base_root in d.parents):
                        why_out.append(f"store {str(d) if d is not None else 'unlocatable'} is not under {base_root}")
                    if not str(cur.get("eval_id") or "").startswith("petri_"):
                        why_out.append("the eval id is no longer a Petri eval (petri_*)")
                    parts.append(f"{rid}: store_uri/eval_id now {cur.get('store_uri')!r} / {cur.get('eval_id')!r}"
                                 + (f" ({'; '.join(why_out)})" if why_out else ""))
                raise DriverError(f"{where}: selected run dropped from the judged view between the worklist read and the borrow: {' | '.join(parts)}; "
                                  f"refusing before any claim -- the filing would have judged the run under the cell / behaviour the worklist read, "
                                  f"not where the registry now places it ({len(dropped)} of {len(selected)} selected run(s) missing from the "
                                  f"{len(present)} registered under {base_root})")
            # Presence is not identity. A same-id FULL correction folded between the worklist read and this
            # SELECT that moved a selected run's producer key to another cell, or its eval to another petri_*, with the store still under the
            # root kept the id present while in_view_runs_at_selection captured the NEW identity and the filing recheck compared new-to-new, so
            # the filing loop routed the judgment from the STALE worklist row (_cell_of / _behaviour_of). Every in-view identity field of each
            # selected worklist row must equal the row read here; the worklist rows are the baseline recorded below (in_view_runs_at_selection),
            # re-read before the filing and routed from -- one set of rows, checked at every point, filed from.
            changed: List[str] = []
            # the in-view SELECT reads no selection fact -- the bound judge is verified against the pending view here (one read)
            pending_now = _pending_judges(conn, selected)
            default_judge = _default_judge_worklist(args)
            for rid in selected:
                sel, cur = by_id[rid], present[rid]
                cur["pending_judge_ids"] = pending_now.get(rid, [])
                diffs = identity_diff_lines(sel, cur, BORROW_IN_VIEW_IDENTITY_FIELDS, default_judge=default_judge)   # the one comparison
                if diffs:
                    changed.append(f"{rid}: " + "; ".join(diffs))
            if changed:
                raise DriverError(f"{where}: selected run's identity changed between the worklist read and the borrow: {' | '.join(changed)}; "
                                  "refusing before any claim -- the filing routes a run's judged rows from the worklist row it selected, so a cell / "
                                  "behaviour / store / completion the registry moved in between would have been filed stale")

        def _one(field: str) -> str:
            vals = sorted({str(r.get(field)) for r in in_view})
            if len(vals) != 1:
                raise DriverError(f"{where}: the registered runs under {base_root} carry {len(vals)} distinct {field!r} values {vals} -- "
                                  "one view's runs carry exactly one; refusing")
            return vals[0]
        campaign, model_id, tag = _one("campaign"), _one("model_id"), _one("view_legacy")
        if not tag.startswith(model_id + "_"):
            raise DriverError(f"{where}: the registry's tag {tag!r} does not read <model_id>_<view> for model {model_id!r}; refusing to "
                              "derive the u0 audit")
        view = tag[len(model_id) + 1:]
        if view == _U0_VIEW:
            return None                 # a u0 view without `none` rows keeps the standing gate refusal, word for word
        if not str(spec.audit).endswith("_" + view):
            raise DriverError(f"{where}: the audit name does not end in the registry's view {view!r} (tag {tag!r}); refusing to derive the "
                              "u0 audit")
        u0_audit = str(spec.audit)[: -len(view)] + _U0_VIEW
        u0_tag = f"{model_id}_{_U0_VIEW}"
        # 2. the candidate anchors: the `none` runs of (campaign, model) for the view's evaluations, with their currency
        rows = _base_screen_anchor_rows(conn, campaign, model_id, list(eval_of))
    anchors = [r for r in rows if _cell_of(r) == dc.BASE_CELL_ID]

    def _line(r: Mapping[str, Any]) -> str:
        why = f"; why: {r['why']}" if r.get("why") else ""
        return f"{r['run_id']} ({r['eval_id']}; lifecycle {r['lifecycle_state']}; is_current {r['is_current']}{why})"

    def _usable(r: Mapping[str, Any]) -> bool:
        return int(r.get("is_current") or 0) == 1 and str(r.get("lifecycle_state")) in _BORROW_GENERATED_STATES
    current = [r for r in anchors if _usable(r)]
    roots: Dict[Path, List[Dict[str, Any]]] = {}
    for r in current:
        d = _located(r)
        if d is not None:
            roots.setdefault(d.parent.parent, []).append(r)     # store_uri = <out root>/cells/<petri_run_key>
    root_audits: Dict[str, Optional[str]] = {}
    u0_roots: List[Path] = []
    for rt in sorted(roots):
        audit = None
        rec_path = rt / CELLS_JSON
        if rec_path.exists():
            try:
                audit = json.loads(rec_path.read_text(encoding="utf-8")).get("audit")
            except (OSError, ValueError):
                audit = None
        root_audits[str(rt)] = audit if audit is not None else "NO cells.json on this host (absent or pruned store)"
        if audit == u0_audit:
            u0_roots.append(rt)
    if len(u0_roots) != 1:
        found = "; ".join(_line(r) for r in anchors) or "no registered run"
        raise DriverError(f"{where}: {why_borrow}; expected exactly one store of audit {u0_audit!r} (tag {u0_tag!r}) holding CURRENT, "
                          f"generated `none` Petri runs of model {model_id!r} (campaign {campaign!r}) for evaluations {sorted(eval_of)}, "
                          f"found {len(u0_roots)} (store roots of the current none runs, with their record's audit: {root_audits or 'none'}); "
                          f"every `none` run on record for these evaluations: {found}; refusing -- a missing or retired baseline is a "
                          "refusal, never a smaller analysis")
    u0_root = u0_roots[0]
    u0_record = read_cells_record(u0_root)
    none_key = _served_run_keys(u0_record).get(dc.BASE_CELL_ID)
    if none_key is None:
        raise DriverError(f"{where}: the enumeration record of {u0_audit!r} ({u0_root / CELLS_JSON}) does not serve {dc.BASE_CELL_ID!r}; "
                          "refusing")
    none_dir = (u0_root / "cells" / none_key).resolve()
    per_b: Dict[str, Dict[str, Any]] = {}
    siblings: List[Dict[str, Any]] = []
    for eval_id, b in eval_of.items():
        mine = [r for r in anchors if str(r["eval_id"]) == eval_id and _located(r) == none_dir]
        cur = [r for r in mine if _usable(r)]
        if len(cur) != 1:
            found = "; ".join(_line(r) for r in mine) or "no registered run"
            raise DriverError(f"{where}: behaviour {b!r} (evaluation {eval_id!r}) has {len(cur)} CURRENT, generated `none` run(s) of model "
                              f"{model_id!r} under {none_dir} where exactly one is required: {found}; refusing -- a missing, retired or "
                              "ambiguous baseline is a refusal, never a smaller analysis")
        per_b[b] = cur[0]
        siblings += [r for r in mine if r is not cur[0]]
    # 3. the source: an EXPLICIT, VERIFIED dependency. Read in place (_BORROW_SOURCE_DEPENDENCY_NOTE: the daemon neither mirrors
    # nor prunes Petri stores local to the grading host), never consolidated, locked or written -- but (a) its presence is refused
    # by name, (b) every selected current run's registered completion identity is bound to the on-disk results.json, (c) an IMMUTABLE
    # snapshot of the source cell is taken under THIS view's store and the borrowed rows are folded from that copy's own evidence (never
    # the source root's rows.parquet, which may predate the run the registry made current), (d) the footing is bound on the snapshot.
    # The selected anchors must carry the judged view's own identity -- campaign and model_id (the query's predicates,
    # verified explicitly on each selected row) and, when the registration stated one, this manifest's target_revision (a NULL registered
    # revision -- the live drive's registration states none on part of the u0 `none` runs -- is recorded and is itself part of the identity
    # tuple compared at both checkpoints); the judged view's own registered runs are held to the same revision
    target_rev = str(spec.target_revision)
    in_view_revs = sorted({str(r.get("model_revision")) for r in in_view if r.get("model_revision") is not None})
    if any(rv != target_rev for rv in in_view_revs):
        raise DriverError(f"{where}: the judged view's own registered runs under {base_root} carry model_revision(s) {in_view_revs} but this "
                          f"manifest's target_revision is {target_rev}; refusing")
    anchor_revs: Dict[str, Any] = {}
    for b, r in sorted(per_b.items()):
        bad: List[str] = []
        if str(r.get("campaign")) != campaign:
            bad.append(f"campaign {r.get('campaign')!r} != the judged view's {campaign!r}")
        if str(r.get("model_id")) != model_id:
            bad.append(f"model_id {r.get('model_id')!r} != the judged view's {model_id!r}")
        rv = r.get("model_revision")
        if rv is not None and str(rv) != target_rev:
            bad.append(f"model_revision {str(rv)[:12]} != the manifest's target_revision {target_rev[:12]}")
        if bad:
            raise DriverError(f"{where}: behaviour {b!r}: the selected `none` run {r['run_id']!r} is not this view's baseline ({'; '.join(bad)}); "
                              "refusing")
        anchor_revs[b] = rv
    source_identity: Dict[str, Any] = {
        "campaign": campaign, "model_id": model_id, "target_revision": target_rev, "in_view_registered_revisions": in_view_revs,
        "anchor_registered_revisions": anchor_revs,
        "rule": ("every selected anchor's campaign and model_id equal the judged view's; a registered model_revision (anchor or in-view) equals "
                 "the manifest's target_revision; a NULL registered revision is recorded and is part of the identity tuple re-read at both "
                 "checkpoints")}
    anchor_ids = sorted(r["run_id"] for r in per_b.values())
    missing = [str(p) for p in (none_dir, *[none_dir / f for f in _BORROW_SOURCE_FILES]) if not p.exists()]
    if missing:
        raise DriverError(f"{where}: the source store the registry's current `none` runs name ({u0_root}; runs {anchor_ids}) is ABSENT or PRUNED "
                          f"on this host: missing {missing} -- the borrow {_BORROW_SOURCE_DEPENDENCY_NOTE}")
    binding = _verify_source_binding(per_b, none_dir, srm, where)
    inc_path = u0_root / INCOMPLETE_MAP
    u0_incomplete = json.loads(inc_path.read_text(encoding="utf-8")) if inc_path.exists() else {}
    if dc.BASE_CELL_ID in u0_incomplete:
        raise DriverError(f"{where}: the source cell {dc.BASE_CELL_ID!r} of {u0_audit!r} is recorded INCOMPLETE "
                          f"({u0_incomplete[dc.BASE_CELL_ID]}); refusing")
    epochs = int(spec.bloom_options.epochs)
    generation = _source_generation(none_dir)
    # The generation hashed AFTER the binding must still be the bound one -- a source rewritten in between (a
    # re-score is not under this out-root's lock) refuses by name here, before any copy
    gen_results_sha = str(((generation.get("files") or {}).get("results.json") or {}).get("sha256"))
    if gen_results_sha != str(binding["results_json_sha256_on_disk"]):
        raise DriverError(f"{where}: source changed during binding/snapshot (generation): {str(binding['results_json_sha256_on_disk'])[:12]} -> "
                          f"{gen_results_sha[:12]} at {none_dir / 'results.json'}; the source cell was rewritten while it was being bound; "
                          "refusing -- a retry binds the new generation once the registry makes it current")
    snapshot_root, source_json, snapshot_reused, snapshot_verification = _borrowed_base_snapshot(
        out_root, u0_root, u0_record, none_dir, none_key, spec, generation, binding, where)
    u0_rows = snapshot_root / "rows.parquet"
    # the footing from the SNAPSHOT's one cell dir, bound to the source record and reconciled with the copied evidence (the same walk the
    # source's own analysis runs, over the copy)
    u0_footing = collect_audited_footing(snapshot_root, u0_record, epochs).get(dc.BASE_CELL_ID)
    if u0_footing is None:
        raise DriverError(f"{where}: the source cell {dc.BASE_CELL_ID!r} of {u0_audit!r} ({none_dir}) has no complete results.json footing; "
                          "refusing")
    results_meta = json.loads((snapshot_root / "cells" / none_key / "results.json").read_text(encoding="utf-8"))
    root_cmp = _root_parquet_comparison(u0_root / "rows.parquet", u0_rows, dc.BASE_CELL_ID)
    # checkpoint 1 of 2 -- right after the snapshot verification, every anchor's WHOLE identity tuple is re-read from
    # the registry and compared with the tuple the binding used (checkpoint 2 is immediately before the filing, cmd_analyze)
    identity_after_snapshot = verify_borrowed_anchor_identity(db_path, per_b, snapshot_verification["snapshot_results_json_sha256"], where,
                                                              "the snapshot verification")
    none_entry = next(c for c in u0_record["canonical"] if str(c["cell_id"]) == dc.BASE_CELL_ID)
    # the rows bind to the SOURCE record (its audit, its `none` payload) and to THIS manifest's base model (model + revision): a source
    # served from another checkpoint refuses inside analyze's own provenance check
    u0_prov: Dict[str, Any] = {
        "audit": str(u0_record["audit"]),
        "model_id": spec.target_model,
        "model_revision": spec.target_revision,
        "cells": {dc.BASE_CELL_ID: {
            "engine": none_entry["serving_identity"],
            "intervention_hash": none_entry["intervention_hash"],
            "harness_config_hash": none_entry["harness_config_hash"],
            "petri_run_key": none_entry["petri_run_key"],
        }},
    }
    if "serving_path" in u0_record:
        recorded_path = str(u0_record["serving_path"])
        _check_serving_path(recorded_path)
        u0_prov[tc.SERVED_ENGINE_FIELD] = SERVING_ENGINE_BY_PATH[recorded_path]

    def _run_block(r: Mapping[str, Any]) -> Dict[str, Any]:
        # the FULL row the borrow read (identity + recorded-only fields): the filing checkpoint compares the registry against it
        return {k: r.get(k) for k in BORROW_ANCHOR_ROW_FIELDS}
    anchors_block: Dict[str, Any] = {
        "registry": {"path": db_path, "store_root_map": srm},
        "currency_rule": ("v_run_current.is_current = 1 (D651) and a lifecycle of generated / judging / judged / ingested; exactly one "
                          "such run per behaviour under the source `none` cell dir"),
        # the judged view's identity every anchor was verified against at selection
        "source_identity_at_selection": source_identity,
        # the judged view's own runs' identity tuples, re-read before the filing (in_view_recheck) -- in registry mode the WORKLIST
        # rows' (the earliest read: the baseline the borrow compared its SELECT against above and the filing routes from), from disk
        # the in-view SELECT's
        "in_view_runs_at_selection": ({str(x["run_id"]): {f: x.get(f) for f in BORROW_IN_VIEW_FIELDS} for x in selected_rows}
                                      if selected_rows is not None else
                                      {str(r["run_id"]): {f: r.get(f) for f in BORROW_IN_VIEW_FIELDS} for r in in_view}),
        # the borrow's own in-view SELECT as read (every registered Petri run under the root; a superset of the selection in registry mode)
        "in_view_select_at_borrow": {str(r["run_id"]): {f: r.get(f) for f in BORROW_IN_VIEW_FIELDS} for r in in_view},
        # the worklist's selection the in-view SELECT was checked against (None from disk: no selection): every selected
        # id present and every in-view identity field equal
        "worklist_selection": {"run_ids": (sorted(str(x["run_id"]) for x in selected_rows) if selected_rows is not None else None),
                               "identity_fields": list(BORROW_IN_VIEW_IDENTITY_FIELDS),
                               # which judge the selection bound -- the instrument's (default) or an explicit --judge
                               "judge_selection": (("instrument" if _default_judge_worklist(args) else "explicit") if selected_rows is not None else None),
                               "selection_fields": dict(SELECTION_FIELD_REASONS),
                               "rule": ("in registry mode every run the worklist selected is in the borrow's in-view SELECT with every in-view "
                                        "identity field equal, checked before any claim; the worklist rows are in_view_runs_at_selection")},
        "identity_fields": list(BORROW_ANCHOR_IDENTITY_FIELDS),
        "recorded_only_fields": dict(BORROW_ANCHOR_RECORDED_ONLY_REASONS),
        "campaign": campaign, "model_id": model_id, "in_view_tag": tag, "in_view_view": view,
        "source_tag": u0_tag, "source_audit": u0_audit, "source_store_root": str(u0_root), "source_store_cell_dir": str(none_dir),
        "source_petri_run_key": none_key, "source_record_serving_path": u0_record.get("serving_path"), "epochs": epochs,
        "runs": {b: _run_block(r) for b, r in per_b.items()},
        # the other `none` runs on record under the same cell dir (a failed first attempt, a superseded or held-out run): named, never used
        "sibling_runs_not_used": [_run_block(r) for r in siblings],
        # what the borrowed rows were BOUND to and where they were read from
        "source_generation": {"digest": generation["digest"], "files": generation["files"],
                              "results_json_sha256_on_disk": binding["results_json_sha256_on_disk"],
                              "results_completed_at": results_meta.get("completed_at"), "rows_emitted": results_meta.get("rows_emitted"),
                              "registry_binding": binding["per_behaviour"],
                              "rule": ("every selected current run's store_completion_artifact_uri resolves to the source cell's results.json "
                                       "and its store_completion_sha256 equals that file's sha256 on disk")},
        "source_snapshot": {"dir": str(snapshot_root), "rows": str(u0_rows), "reused": bool(snapshot_reused),
                            "fold": "the snapshot's own cells/<key>/rows.jsonl minus its refusal-sidecar tombstones"},
        "source_root_parquet": root_cmp,
        "source_dependency": _BORROW_SOURCE_DEPENDENCY_NOTE,
        # every digest the borrow verified -- the binding's live sha, the generation's, the snapshot's (results.json
        # and rows.jsonl), every selected run's registered completion, the live sha before / after the copy -- and when
        "source_verification": snapshot_verification,
        # checkpoint 1 of the whole-tuple identity check (checkpoint 2, before the filing, lands under currency_recheck)
        "identity_recheck_after_snapshot": identity_after_snapshot,
    }
    print(f"[petri] base-elicitability screen for {spec.audit!r}: {dc.BASE_CELL_ID!r} borrowed from {u0_audit!r} ({none_dir}); anchor run(s) "
          f"{sorted(r['run_id'] for r in per_b.values())}; sibling run(s) not used: {len(siblings)}; source generation "
          f"{generation['digest'][:12]} (results.json {binding['results_json_sha256_on_disk'][:12]}, bound to every anchor); snapshot "
          f"{snapshot_root} ({'reused' if snapshot_reused else 'taken'})", file=sys.stderr, flush=True)
    return petri_stats.BorrowedBase(rows=u0_rows, provenance=u0_prov, cell_footing=u0_footing, anchors=anchors_block)


def _anchor_identity_rows(conn, run_ids: Sequence[str], fields: Sequence[str] = BORROW_ANCHOR_IDENTITY_FIELDS) -> Dict[str, Dict[str, Any]]:
    """The CURRENT identity tuple (``fields`` -- the anchor identity by default, the derived in-view identity for the judged view's own runs --
    plus `why` for the record) of each run id -- run JOIN v_run_current, read-only -- keyed by run id."""
    fields = tuple(fields) + ("why",)
    marks = ", ".join("?" for _ in run_ids)
    sql = f"SELECT {_anchor_select_sql(fields)} FROM run r JOIN v_run_current vc ON vc.run_id = r.run_id WHERE r.run_id IN ({marks})"
    out = {str(row[0]): dict(zip(fields, row)) for row in conn.execute(sql, [str(x) for x in run_ids])}
    if any(f in SELECTION_FIELDS for f in fields):
        # a selection field is verified, not read from `run` -- attach the judges each run is currently a pending pair for
        pending_now = _pending_judges(conn, list(out))
        for rid, row in out.items():
            row["pending_judge_ids"] = pending_now.get(rid, [])
    return out


def verify_borrowed_anchor_identity(db_path: str, expected: Mapping[str, Mapping[str, Any]], snapshot_results_sha: Optional[str], where: str,
                                    stage: str, *, fields: Sequence[str] = BORROW_ANCHOR_IDENTITY_FIELDS, subject: str = "anchor",
                                    default_judge: bool = True) -> Dict[str, Any]:
    """ONE identity check over the WHOLE tuple the binding used, run at BOTH checkpoints -- right after the snapshot
    verification and immediately before the filing. Every anchor's current row (run JOIN v_run_current, read-only) is compared field by field
    (BORROW_ANCHOR_IDENTITY_FIELDS) with the tuple recorded when the borrow bound it: equality for every field but lifecycle_state, which must
    still be a generated-or-later state (a run judged or ingested meanwhile keeps its generation), and is_current, which must be 1; and the
    run's CURRENT store_completion_sha256 must equal the VERIFIED SNAPSHOT's results.json sha. The case this closes: a Petri re-score rewrites
    results.json and the producer emits a NEW completion sha as a run_patch for the SAME run id, which stays current -- a currency-only
    re-check let a screen file from the old rows while the registry pointed to the new completion. Any difference refuses by name (behaviour,
    run, field, before -> after); in registry mode the caller's cleanup releases the claims. Returns the block recorded under the anchors
    (stage, the final tuple per anchor, the fields compared, the time). The SAME function re-reads the judged view's
    own runs before the filing -- ``fields`` = the derived in-view identity (BORROW_IN_VIEW_IDENTITY_FIELDS), ``subject`` = "in-view run", and no
    snapshot sha (``None``: an in-view run's completion is its own cell's); the lifecycle / currency rules apply only to fields compared."""
    from sisteer.registry import db as RDB
    ids = sorted({str(x["run_id"]) for x in expected.values()})
    with RDB.reading(db_path) as conn:
        fresh = _anchor_identity_rows(conn, ids, fields)
    problems: List[str] = []
    final: Dict[str, Any] = {}
    for b, rec in sorted(expected.items()):
        rid = str(rec["run_id"])
        now = fresh.get(rid)
        if now is None:
            problems.append(f"{b}: {rid} is no longer in run JOIN v_run_current")
            continue
        diffs: List[str] = []
        for f in fields:
            was, cur = rec.get(f), now.get(f)
            if f == "lifecycle_state":
                if str(cur) not in _BORROW_GENERATED_STATES:
                    diffs.append(f"{f}: {was!r} -> {cur!r} (no longer generated-or-later)")
            elif f == "is_current":
                if int(cur or 0) != 1:
                    diffs.append(f"{f}: {was!r} -> {cur!r} (why: {now.get('why')})")
            else:
                # the one comparison (a selection field is verified against the pending view and, by default, the instrument)
                diffs += identity_diff_lines(rec, now, (f,), default_judge=default_judge)
        if snapshot_results_sha is not None and str(now.get("store_completion_sha256")) != str(snapshot_results_sha):
            diffs.append(f"store_completion_sha256 {str(now.get('store_completion_sha256'))[:12]} != the verified snapshot's results.json "
                         f"{str(snapshot_results_sha)[:12]}")
        if diffs:
            problems.append(f"{b}: {rid}: " + "; ".join(diffs))
        final[b] = {f: (rec.get(f) if f in SELECTION_FIELDS else now.get(f)) for f in fields}   # the bound selection value, verified above
        final[b]["why"] = now.get("why")
        if "pending_judge_ids" in now:
            final[b]["pending_judge_ids"] = list(now["pending_judge_ids"])
    if problems:
        raise DriverError(f"{where}: {subject} identity changed between the binding and {stage}: {' | '.join(problems)}; refusing -- a borrowed "
                          "screen never files on a base whose run, currency or registered completion the registry moved (never a silent stale "
                          "borrow)")
    return {"stage": stage, "recheck_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "registry_path": db_path,
            "fields": list(fields), "snapshot_results_json_sha256": (str(snapshot_results_sha) if snapshot_results_sha is not None else None),
            "runs": final, "subject": subject,
            "rule": ("every identity field equal to the tuple the binding used (lifecycle_state: still generated-or-later; is_current: 1) and "
                     "the run's registered completion sha equal to the verified snapshot's results.json, re-read from run JOIN v_run_current")}


def recheck_borrowed_anchors_current(analysis: Any, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Checkpoint 2 of 2 -- immediately before the analysis is written and the judgments filed,
    the WHOLE identity tuple of every anchor run (:func:`verify_borrowed_anchor_identity`: run, behaviour, cell, store, usable lifecycle,
    currency, selection state, registered completion) must still be the one the borrow bound, and the registered completion must still be
    the verified snapshot's generation. None when the analysis borrowed nothing; otherwise the block, stamped onto
    ``analysis['borrowed_base']['anchors']['currency_recheck']`` (the analysis is what gets written). Reads the registry the borrow read
    (``anchors.registry.path``), read-only. The anchors' own record is checked for internal agreement first (runs[*].store_completion_sha256
    == source_generation.registry_binding[*] == the snapshot's results.json sha) -- a disagreement there is a defect of this driver, refused."""
    bb = analysis.get("borrowed_base") if isinstance(analysis, Mapping) else None
    if not isinstance(bb, Mapping):
        return None
    anchors = bb.get("anchors")
    runs = (anchors.get("runs") or {}) if isinstance(anchors, Mapping) else {}
    if not runs:
        return None
    audit = str((bb.get("in_view") or {}).get("audit"))
    where = f"base-screen borrow for {audit!r}"
    db_path = str(((anchors.get("registry") or {}).get("path")) or _registry_db_path(args))
    snap_sha = str((anchors.get("source_verification") or {}).get("snapshot_results_json_sha256"))
    binding = (anchors.get("source_generation") or {}).get("registry_binding") or {}
    for b, x in sorted(runs.items()):
        bound = (binding.get(b) or {}).get("store_completion_sha256")
        if str(x.get("store_completion_sha256")) != str(bound) or str(bound) != snap_sha:
            raise DriverError(f"{where}: the anchors disagree internally for {b!r} (run {x.get('run_id')!r}: recorded completion "
                              f"{str(x.get('store_completion_sha256'))[:12]}, binding {str(bound)[:12]}, snapshot {snap_sha[:12]}); refusing to file")
    block = verify_borrowed_anchor_identity(db_path, runs, snap_sha, where, "the filing")
    anchors["currency_recheck"] = block                     # the analysis dict is what cmd_analyze writes and files from
    # the judged view's OWN runs, captured at selection, re-read with the SAME function before the filing -- a same-id
    # corrective full record restating an in-view run's campaign / model / revision / completion after the borrow refuses here, in both modes
    in_view = anchors.get("in_view_runs_at_selection") or {}
    if in_view:
        anchors["in_view_recheck"] = verify_borrowed_anchor_identity(db_path, in_view, None, where, "the filing",
                                                                     fields=BORROW_IN_VIEW_IDENTITY_FIELDS, subject="in-view run",
                                                                     default_judge=_default_judge_worklist(args))
    print(f"[petri] {where}: anchor identity re-checked before filing ({len(block['runs'])} run(s), {len(block['fields'])} fields, "
          f"completion {snap_sha[:12]}); in-view run(s) re-checked: {len(in_view)}", file=sys.stderr, flush=True)
    return block


def recheck_selected_rows(analysis: Any, args: argparse.Namespace, registry_rows: Sequence[Mapping[str, Any]], spec: cm.PetriSpec) -> Optional[Dict[str, Any]]:
    """The selected rows -- the worklist rows, the one identity baseline -- re-read with the one
    identity function immediately before the first write in EVERY registry-mode analysis, borrow or not. The own-base (u0) path -- the view
    serves `none` -- recorded no in-view tuples (base_screen_borrow returns None first) and recheck_borrowed_anchors_current returns at once
    without borrowed_base, so a same-id instrument / judge correction after the worklist's agreement (even after the claims) but before the
    filing filed under the old required judge. Any difference refuses by name (the caller's cleanup releases the claims); the block is recorded
    as ``analysis['selected_rows_recheck']`` whether or not the analysis borrowed. None from disk (no selection)."""
    if not registry_rows:
        return None
    where = f"registry worklist filing for {spec.audit!r}"
    expected = {str(filing_field(r, "run_id")): {f: r.get(f) for f in BORROW_IN_VIEW_FIELDS} for r in registry_rows}
    default_judge = _default_judge_worklist(args)
    block = verify_borrowed_anchor_identity(_registry_db_path(args), expected, None, where, "the filing", fields=BORROW_IN_VIEW_IDENTITY_FIELDS,
                                            subject="selected run", default_judge=default_judge)
    block["judge_selection"] = "instrument" if default_judge else "explicit"
    block["borrowed"] = bool(isinstance(analysis, Mapping) and analysis.get("borrowed_base"))
    analysis["selected_rows_recheck"] = block                 # the analysis dict is what cmd_analyze writes and files from
    print(f"[petri] {where}: selected run(s) re-checked before filing ({len(block['runs'])} run(s), {len(block['fields'])} fields, judge "
          f"selection {block['judge_selection']}, borrowed {block['borrowed']})", file=sys.stderr, flush=True)
    return block


def recheck_run_before_filing(args: argparse.Namespace, row: Mapping[str, Any], spec: cm.PetriSpec) -> Dict[str, Any]:
    """ONE run's claim-bound identity -- its worklist row, the baseline every earlier checkpoint compared --
    re-read with the one identity function immediately before THAT run's first write (its item rows, metric rows and completion ride one
    scope, so before the scope is entered). recheck_selected_rows runs once before all filings; a correction folded after it and before a
    LATER run's completion would have filed that run under the old judge / cell. Any difference refuses this run's filing by name (the caller
    skips its scope, so nothing of it is written, and the cleanup releases its claim); the returned block is what the filed judgment carries as
    its notes' ``filing_identity_recheck`` -- the identity it was bound to, verified at the write."""
    rid = str(filing_field(row, "run_id"))
    where = f"registry worklist filing for {spec.audit!r}"
    default_judge = _default_judge_worklist(args)
    block = verify_borrowed_anchor_identity(_registry_db_path(args), {rid: {f: row.get(f) for f in BORROW_IN_VIEW_FIELDS}}, None, where,
                                            f"its own filing (run {rid})", fields=BORROW_IN_VIEW_IDENTITY_FIELDS, subject="selected run",
                                            default_judge=default_judge)
    block["judge_selection"] = "instrument" if default_judge else "explicit"
    return block


def _claim_all_timed(registry_wl, rows: Sequence[Mapping[str, Any]]):
    """``claim_all`` with the hold time of EVERY lock take on stderr (the 02:16Z Petri grader held the registry's write lock
    ~15 min). ``pending.claim`` takes the harvester lock and runs ONE FULL harvester cycle per run, and ``RegistryWorklist.claim_all`` is
    a per-row loop over it -- there is no batch claim form in ``pending`` (the registry owner's module, not edited here), so a worklist of
    N runs is N takes and N cycles; the per-take line makes a slow cycle visible in the dispatch log. ``claim_all``'s own semantics
    (a claim held by another session releases the claims just taken and re-raises) are kept: only the instance's ``claim`` is timed for
    the duration of the call."""
    original = registry_wl.claim

    def timed(row):
        t0 = time.monotonic()
        try:
            return original(row)
        finally:
            print(f"[registry] claim lock held {time.monotonic() - t0:.1f} s ({row.get('run_id')!r}; one harvester cycle per claim -- "
                  "pending.claim has no batch form)", file=sys.stderr, flush=True)
    registry_wl.claim = timed
    try:
        return registry_wl.claim_all(rows)
    finally:
        del registry_wl.claim


def _analysis_under_lock(args: argparse.Namespace, spec: cm.PetriSpec, out_root: Path, *,
                         borrowed_base: Any = _BORROW_UNRESOLVED) -> Dict[str, Any]:
    """The frame read and the analysis of ``cmd_analyze`` (the body it always had, from the rows.parquet check to
    ``petri_stats.analyze``), factored out so it runs under the out-root lock the caller holds (g6-fix). ``borrowed_base``: the
    steered view's borrowed base (:func:`base_screen_borrow`), resolved by the registry path BEFORE its claims; the sentinel default
    resolves it here (the from-disk path and any direct caller)."""
    rows_path = out_root / "rows.parquet"
    if not rows_path.exists():
        raise DriverError(f"{rows_path} missing — run `consolidate` first")
    record = read_cells_record(out_root)
    # the record must be THIS manifest's (review 2026-08-18: rows and a
    # cells.json from another audit must never publish as this one's)
    _check_record_matches_spec(record, spec, out_root / CELLS_JSON)
    absent = {a["cell_id"]: a["reason"] for a in record.get("absent", [])}
    map_path = out_root / INCOMPLETE_MAP
    incomplete = (json.loads(map_path.read_text())
                  if map_path.exists() else {})
    impl_commit = (None if spec.serving_impl_commit == cm.PETRI_UNFROZEN
                   else spec.serving_impl_commit)
    # bind every consolidated row to the enumeration payloads (engine,
    # intervention/harness hashes, run key) — analyze verifies per cell
    provenance = {
        "audit": spec.audit,
        "model_id": spec.target_model,
        "model_revision": spec.target_revision,
        "cells": {
            c["cell_id"]: {
                "engine": c["serving_identity"],
                "intervention_hash": c["intervention_hash"],
                "harness_config_hash": c["harness_config_hash"],
                "petri_run_key": c["petri_run_key"],
            }
            for c in (list(record.get("canonical") or [])
                      + list(record.get("controls") or []))},
    }
    # the served ENGINE the frozen record was enumerated under (a native
    # record names its serving_path; a lens record carries no key and the
    # provenance then carries none — byte-identical lens outputs)
    if "serving_path" in record:
        recorded_path = str(record["serving_path"])
        _check_serving_path(recorded_path)
        provenance[tc.SERVED_ENGINE_FIELD] = SERVING_ENGINE_BY_PATH[recorded_path]

    if borrowed_base is _BORROW_UNRESOLVED:
        # the from-disk path (and any direct caller): resolve the borrow here; the registry path resolved it BEFORE its claims
        borrowed_base = base_screen_borrow(args, spec, out_root, record)
    analysis = petri_stats.analyze(
        rows_path,
        borrowed_base=borrowed_base,
        expected_cells=list(spec.cells),
        expected_behaviors=list(spec.behaviors),
        n_boot=args.n_boot,
        seed=args.seed,
        incomplete_cells=incomplete,
        absent_cells=absent,
        certificates=_load_certificates(
            Path(args.certificates) if args.certificates else None),
        model_revision=spec.target_revision,
        serving_impl_commit=impl_commit,
        gate_prompt_set_hash=_gate_prompt_set_hash(),
        provenance=provenance,
        # every complete cell's audited footing, beside its scores — bound
        # to the frozen enumeration's run per cell
        cell_footing=collect_audited_footing(out_root, record,
                                             spec.bloom_options.epochs),
    )
    return analysis


def cmd_analyze(args: argparse.Namespace, spec: cm.PetriSpec) -> int:
    out_root = _out_root(args, spec)
    registry_wl = None
    registry_rows: list = []
    borrowed: Any = _BORROW_UNRESOLVED     # resolved BEFORE the claims on the registry path; by _analysis_under_lock otherwise
    if getattr(args, "worklist", "from-disk") == "registry":
        from sisteer.registry import worklist as RW
        try:
            registry_wl, registry_rows = registry_worklist_runs(args, out_root)
        except RW.RegistryUnavailable as e:
            raise DriverError(str(e)) from None
        if not registry_rows:
            raise DriverError(f"--worklist registry: no pending petri run under {out_root}; nothing to analyse")
        # every refusal knowable BEFORE the claims fires here -- the frozen record must be this manifest's, and a
        # view without an undefended `none` cell must have a CURRENT u0 `none` baseline to borrow (base_screen_borrow) -- so a gate
        # refusal claims nothing and releases nothing (the 02:16Z dispatch claimed 7, refused at the gate, released 7: fourteen full
        # harvester cycles under the registry's write lock for an analysis that never ran)
        record = read_cells_record(out_root)
        _check_record_matches_spec(record, spec, out_root / CELLS_JSON)
        # the borrow is handed the worklist ROWS it will be filed from; one missing from the judged view's set it reads, or
        # one whose in-view identity differs from that set, refuses by name here -- the rows are the baseline the filing re-reads against
        borrowed = base_screen_borrow(args, spec, out_root, record, selected_rows=registry_rows)
        try:
            _claim_all_timed(registry_wl, registry_rows)
        except (RW.P.ClaimHeld, RW.P.ClaimNotFolded) as e:   # every claim refusal -- held, not folded, not pending; claim_all released what it took
            raise DriverError(f"--worklist registry: refusing — {e}") from None
        # The registry worklist is the grading daemon's path: the Petri drive hands the
        # store off as `generated` and NOTHING between that hand-off and this analyze consolidates it, so the daemon's first
        # dispatch of every front-door audit died on the `rows.parquet missing` refusal below (the u0 wave, 2026-10-01; masked
        # until then by a schema-drift refusal ahead of it) and its 21 runs sat quarantined. Here the analysis consolidates
        # the store itself — the same deterministic fold `consolidate` runs, re-run over the cells as they stand NOW, so a
        # re-driven or re-scored cell is never analysed through a stale frame. The from-disk path keeps the operator's
        # two-step contract (`consolidate`, then `analyze`): a hand-run analyze still refuses a store nobody consolidated.
    else:
        from sisteer.registry import worklist as RW
        RW.log_from_disk()
    # The fold and the frame read run under ONE exclusive out-root lock, and every claim this
    # invocation took is released on ANY failure before the judgments are filed (the JudgingScopes below file or abandon their
    # own), the original exception preserved -- a malformed rows.jsonl, an incomplete mirror or a failing analysis no longer
    # leaves the runs claimed by a dead session.
    # The cleanup covers the WHOLE claim-to-judgment interval -- the fold, the analysis, the
    # analysis.json write, the frame and every judgment's filing -- with an explicit list of the claims not yet carried to a filed
    # judgment; a scope that emitted its completion leaves that list (its judgment may still read in_progress until the harvester
    # folds the spool -- it is never abandoned here, and the claim it holds prevents a duplicate dispatch). SIGTERM (the default
    # disposition ends the interpreter with the claims held) is turned into AnalysisTerminated for the interval, main thread only.
    # The item frame is captured ONCE under the same lock as the analysis and used for the filing too: a second analyzer folding
    # newly mirrored or re-scored cells in between swaps the parquet (store.consolidate replaces it atomically), and re-reading it
    # would file per-item rows from the NEW frame under metrics and footing from the OLD one.
    rows_path = out_root / "rows.parquet"
    out_path = out_root / ANALYSIS_JSON
    remaining: List[Mapping[str, Any]] = list(registry_rows)
    refused_at_filing: List[Tuple[Mapping[str, Any], str]] = []    # runs refused at their own filing boundary (claims released below)
    current = None                                            # (row, scope) of the judgment being filed, for the completion-boundary rule
    frame = None
    previous_term = None
    if registry_wl is not None and threading.current_thread() is threading.main_thread():
        previous_term = signal.signal(signal.SIGTERM, _raise_terminated)
    try:
        with _out_root_lock(out_root):
            if registry_wl is not None:
                print(f"[petri] --worklist registry: consolidating {out_root} before the analysis "
                      f"({len(registry_rows)} pending run(s) claimed)", file=sys.stderr)
                _consolidate_store(out_root, spec)
            analysis = _analysis_under_lock(args, spec, out_root, borrowed_base=borrowed)
            # the borrowed anchors must STILL be current right before anything is filed (the analysis written,
            # the judgments emitted); a run that lost currency since the binding refuses by name here -- the claims are released below
            recheck_borrowed_anchors_current(analysis, args)
            if registry_wl is not None:
                # the selected rows -- the worklist baseline -- re-read UNCONDITIONALLY before the first write,
                # borrow or not (the own-base u0 path recorded no in-view tuples and the borrow recheck returns at once without one)
                recheck_selected_rows(analysis, args, registry_rows, spec)
                frame = petri_stats.load_rows(rows_path)     # the per-transcript judge scores the analysis aggregated: ONE generation
        out_path.write_text(json.dumps(analysis, indent=2, sort_keys=True),
                            encoding="utf-8")
        if registry_wl is not None:
            for r in registry_rows:
                # the run's rows, metric and notes are computed BEFORE its scope opens: a refusal here -- an untranslatable cell,
                # an empty selection, a foreign eval id -- must reach the claim cleanup as a refusal, never become the scope's own
                # partial filing of zero items (which the completion-boundary rule would then rightly keep as a filed judgment)
                item_rows = petri_item_rows(frame, r, analysis)
                metric_rows = petri_metric_rows(analysis, r)
                n_refused, notes = petri_judgment_notes(analysis, r)
                n_items = len({x["item_id"] for x in item_rows})
                generated = int(filing_field(r, "n_items_generated") or 0)
                if generated and n_items != generated:
                    raise DriverError(f"run {filing_field(r, 'run_id')!r}: the frame holds {n_items} judged item(s) for its cell+behaviour but the run "
                                      f"generated {generated}; refusing to file a judgment whose coverage the rows do not bear out")
                # THIS run's claim-bound identity re-read immediately before its first write -- the once-before-all
                # recheck above cannot see a correction folded before a LATER run's completion. A difference refuses this run's filing by name:
                # its scope is never entered (nothing of it written), its claim stays in `remaining` for the cleanup to release, the siblings'
                # completed filings stand and the remaining siblings are filed behind their own guards; the invocation fails at the end.
                try:
                    filing_recheck = recheck_run_before_filing(args, r, spec)
                except DriverError as exc:
                    refused_at_filing.append((r, str(exc)))
                    print(f"[petri] {exc}; this run's filing is refused -- its claim is released below, the siblings' filings proceed",
                          file=sys.stderr, flush=True)
                    continue
                notes = {**(notes or {}), "filing_identity_recheck": {k: filing_recheck[k] for k in ("stage", "recheck_utc", "fields", "runs",
                                                                                                        "judge_selection", "rule")}}
                scope = registry_wl.scope(r, store_uri=str(out_path))
                current = (r, scope)
                with scope:
                    scope.items(item_rows)
                    scope.metrics(metric_rows)
                    scope.complete("complete", n_items_judged=n_items or generated,
                                   n_judge_refusals=n_refused, **({"extra": notes} if notes else {}))
                remaining.remove(r)                          # its completion is emitted: no longer this invocation's claim to release
                current = None
            rep_ = registry_wl.finish()
            print(rep_.line(), flush=True)
            if refused_at_filing:
                raise DriverError(f"registry worklist filing for {spec.audit!r}: {len(refused_at_filing)} of {len(registry_rows)} selected run(s) "
                                  "refused at their own filing boundary -- the claim-bound identity changed after the pre-filing recheck: "
                                  + " | ".join(m for _r, m in refused_at_filing) + "; those claims are released and nothing of theirs is filed; "
                                  "the siblings' completed filings stand")
    except BaseException:
        if registry_wl is not None:
            # The signal stays ignored for the whole cleanup (a second SIGTERM mid-release would otherwise
            # raise inside the loop and leave the later claims held), and the run whose scope already WROTE its completion record --
            # `completion_emitted`, set after writer.emit_judgment returns (never `closed`, which flips before the write: a full spool
            # left closed=True with no record and the run exempt from release forever) -- is not released: its claim is what prevents a
            # duplicate dispatch until the harvester folds the completion. A write that raised is uncertain and RELEASES; the
            # natural-key / judgment_id idempotency absorbs a duplicate rather than stranding the run.
            if previous_term is not None:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            releasable = [x for x in remaining if not (current is not None and x is current[0] and getattr(current[1], "completion_emitted", False))]
            if releasable:
                _release_claims(registry_wl, releasable)
            if len(releasable) != len(remaining):
                print("[registry] kept the claim of the run whose completion record was already emitted (never abandoned; the harvester folds it)",
                      file=sys.stderr)
        raise
    finally:
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)
    from sisteer.capeval.petri import petri_report  # presentation helper
    comp = analysis["completeness"]
    # the same eligible-behaviors statement the report prints: an
    # UNAVAILABLE screen is never summarised as a computed empty set
    print(f"[petri] analysis -> {out_path} "
          f"({comp['n_complete']}/{comp['expected_canonical']} canonical "
          f"cells complete; {petri_report.eligible_behaviors_line(analysis)})")
    refused = {c: f["auditor_refused"].get("total", 0)
               for c, f in analysis.get("audited_footing", {}).items()
               if f.get("auditor_refused", {}).get("total", 0)}
    if refused:
        print(f"[petri] auditor_refused per cell (first-class outcome, NOT a "
              f"score; see by_cell[*].audited_footing): {refused}")
    # the judge twin, read tolerantly: footing filed before the key existed
    # carries none
    judge_refused = {c: (f.get("judge_refused") or {}).get("total", 0)
                     for c, f in analysis.get("audited_footing", {}).items()
                     if (f.get("judge_refused") or {}).get("total", 0)}
    if judge_refused:
        print(f"[petri] judge_refused per cell (first-class outcome, NOT a "
              f"score; see by_cell[*].audited_footing): {judge_refused}")
    return 0


def cmd_report(args: argparse.Namespace, spec: cm.PetriSpec) -> int:
    from sisteer.capeval.petri import petri_report  # guards wandb/mpl

    out_root = _out_root(args, spec)
    analysis_path = out_root / ANALYSIS_JSON
    if not analysis_path.exists():
        raise DriverError(f"{analysis_path} missing — run `analyze` first")
    analysis = json.loads(analysis_path.read_text())
    run_name = args.run_name or spec.audit
    paths = petri_report.build(
        analysis, out_root / "report", run_name,
        figs_dir=Path(args.figs_dir) if args.figs_dir else None,
        wandb_mode=args.wandb_mode,
        mirror=not args.no_mirror,
        mirror_dir=Path(args.mirror_dir) if args.mirror_dir else None,
    )
    print(f"[petri] report -> {paths['report_md']}")
    if paths["wandb_url"]:
        print(f"[petri] W&B ({petri_report.WANDB_PROJECT}): "
              f"{paths['wandb_url']}")
    if paths["mirror_dir"]:
        print(f"[petri] figures mirrored -> {paths['mirror_dir']}")
    return 0


# --- CLI ----------------------------------------------------------------------------


def _add_target_decode_flag(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--target-decode", choices=cm.PETRI_TARGET_DECODE_MODES, default=None,
        help="ASSERT the target decode contract the manifest declares "
             "(module docstring): `harness` = the server's defaults, only "
             "max_tokens on the request (every audit before the field); "
             "`vendor` = the registry vendor-preset row on every target "
             "request + a per-scenario-turn seed. The manifest's "
             "target_decode block is the authority (a harness-hash input the "
             "standup hashed too); a flag that disagrees with it refuses "
             "before any leg. Default: no assertion")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    # TOP-LEVEL, before the subcommand (the wave drive spells them so:
    # ``--manifest M --serving-path P --verify-standup C --verify-routing-table
    # T print-cells|run ...``, petri_drive_full22.sh)
    ap.add_argument(
        "--serving-path", choices=SERVING_PATHS, default=SERVING_PATH_LENS,
        help="the engine the legs are served on (module docstring): `lens` "
             "(default) enumerates exactly as before; `lora_native` "
             "enumerates the EFFECTIVE native configuration (skip-class "
             "cells skipped by name, the standup's stripped launch flags, "
             "none__lora0 in place of none__lens0) and REQUIRES both "
             "--verify-* flags for print-cells / run")
    ap.add_argument(
        "--verify-standup", type=Path, default=None,
        help="the standup's final cells_served.json: the enumeration must "
             "agree with it cell by cell (vmodel, serving identity, "
             "intervention hash, harness hash, run key; equal served sets) "
             "or print-cells / run refuse before any leg. Required under "
             "--serving-path lora_native; optional under lens")
    ap.add_argument(
        "--verify-routing-table", type=Path, default=None,
        help="the standup's routing_table.json (proxy.load_routing_table): "
             "its routes must agree with the enumeration AND the standup "
             "record; goes with --verify-standup")
    ap.add_argument(
        "--served-endpoint", type=Path, default=None,
        help="the bound proxy's endpoint.json as the drive pulled it "
             "(common.sh writes attested_max_model_len for the lens and the "
             "native engine alike): an armed --overflow-rerun reads the "
             "window the proxy ATTESTS from it when no --verify-standup "
             "record is given (the lens serving path has none), and from "
             "both when both are given — they must agree")
    ap.add_argument(
        "--offline-models-json", type=Path, default=None,
        help="PROXY-FREE CONVERSION (2026-09-21): the /v1/models response the "
             "grading host drive saved from the LIVE proxy ($STATE/models.json). The "
             "engine handle is built from it exactly as ProxyClientEngine.load() "
             "would have (served models + the petri_proxy identity block) with NO "
             "network call, so every identity check reads what the real proxy "
             "attested. For re-converting cells whose every sample is already in "
             "the Inspect logs; nothing is audited — a sample that still needs "
             "generation errors loudly against an unreachable endpoint. Requires "
             "--verify-standup / --verify-routing-table as a live drive does.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("print-cells",
                        help="enumerate served legs + ABSENT report")
    pc.add_argument("--out-root", default=None,
                    help="also write cells.json (the enumeration record)")
    _add_target_decode_flag(pc)

    run = sub.add_parser("run", help="execute one served leg")
    # EXACTLY one selector (review 2026-08-18: both at once used to run
    # the index-selected cell silently); _resolve_cell re-checks it.
    sel = run.add_mutually_exclusive_group(required=True)
    sel.add_argument("--cell-index", type=int, default=None)
    sel.add_argument("--cell-id", default=None)
    run.add_argument("--seed", type=int, default=None,
                     help="default: derived from the cell's run key")
    run.add_argument("--out-root", default=None)
    _add_target_decode_flag(run)
    run.add_argument(
        "--target-seed-base", type=int, default=None,
        help="the RUN's decode seed base = the literal training seed s of "
             "the object served (qwen-seedrep-native-battery plan §6; spec "
             "fixed decision 3): REQUIRED when the manifest declares "
             "target_decode vendor, REFUSED under harness. Every target "
             "request then carries seed = sha256(f'{s}|petri|{behaviour}|"
             "{scenario_id}|{turn}')[:4]; the base is stamped on the cell "
             "manifest / rows / eval-log metadata and namespaces the Inspect "
             "log dir — never a harness-hash input (one standup serves every "
             "seed base). Distinct seed-runs of one manifest need distinct "
             "--out-root")
    run.add_argument(
        "--overflow-rerun", action="store_true",
        help="the targeted context-overflow re-run pass (the user, "
             "2026-08-27): refuses unless this cell's failure ledger holds "
             "overflow-classified unresolved entries, caps eval width to "
             "petri_runner.rerun_serving_geometry(), and expects the "
             "serving stack relaunched at that geometry's larger window; "
             "scenarios re-run from turn 1 via the ordinary resume path")

    ls = sub.add_parser("legs-status",
                        help="the drive's own done-ness reading of every leg of the frozen enumeration under --out-root "
                             "(leg_already_done per leg, the ledger's unresolved / overflow entries), as one JSON document; no GPU, "
                             "no proxy, no standup evidence, no fresh enumeration (petri-completion: the funnel's resume proof, 06's "
                             "overflow pass and the drive's `ledgered` selector read it)")
    ls.add_argument("--out-root", default=None)

    smoke = sub.add_parser("smoke",
                           help="launch-blocking real agent smoke (launch "
                                "phase only — refuses until then)")
    smoke.add_argument("--out-root", default=None)

    con = sub.add_parser("consolidate")
    con.add_argument("--out-root", default=None)

    an = sub.add_parser("analyze")
    an.add_argument("--out-root", default=None)
    an.add_argument("--n-boot", type=int, default=petri_stats.DEFAULT_N_BOOT)
    an.add_argument("--seed", type=int, default=0)
    an.add_argument("--certificates", default=None,
                    help="JSON list of §10 no-op certificates (the "
                         "serving-coverage record) — licenses Tier-2 "
                         "cross-path references")
    from sisteer.registry import worklist as _registry_worklist
    _registry_worklist.add_worklist_arguments(an)

    rs = sub.add_parser("rescore", help="judge-only RE-SCORE of a COMPLETE cell's terminal judge refusals over their existing "
                                        "transcripts under the manifest's current judge chain (D895 (iii) / Grok-fix-3); selection "
                                        "recorded in the sidecar's 'rescored' list")
    rs.add_argument("--out-root", default=None)
    rs_sel = rs.add_mutually_exclusive_group(required=True)
    rs_sel.add_argument("--cell-index", type=int, default=None)
    rs_sel.add_argument("--cell-id", default=None)
    rs.add_argument("--items", nargs="*", default=None, help="behavior/scenario_id[#eN] ...; default: every re-scorable terminal refusal")
    rs.add_argument("--why", required=True, help="the reason, recorded on every rescored entry (e.g. the chain gained a terminal leg)")
    rs.add_argument("--requested-by", required=True, help="who asked, recorded on every rescored entry")
    rs.add_argument("--bundle-dir", default=None)
    rs.add_argument("--config-root", default=None)
    rs.add_argument("--dry-run", action="store_true", help="select and print; write nothing")

    rep = sub.add_parser("report")
    rep.add_argument("--out-root", default=None)
    rep.add_argument("--run-name", default=None,
                     help="default: the manifest's audit name")
    rep.add_argument("--figs-dir", default=None)
    rep.add_argument("--wandb-mode", default="online",
                     choices=("online", "offline", "disabled"),
                     help="W&B project `petri` (own project; online from "
                          "the grading host, the capeval report convention)")
    rep.add_argument("--no-mirror", action="store_true",
                     help="skip the ~/figs/<run>/ tailnet mirror")
    rep.add_argument("--mirror-dir", default=None)
    return ap


COMMANDS = {
    "print-cells": cmd_print_cells,
    "run": cmd_run,
    "legs-status": cmd_legs_status,
    "smoke": cmd_smoke,
    "consolidate": cmd_consolidate,
    "analyze": cmd_analyze,
    "rescore": cmd_rescore,
    "report": cmd_report,
}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)  # enforces one run selector
    # Pin the SDPA backend BEFORE any env_digest is recorded (row constants
    # here in cmd_run, and petri_runner's convert/aggregate defaults): the
    # petri driver serves through vLLM and never flips the flag via an HF
    # model load, so an unpinned driver would fingerprint the torch-default
    # backend state. Idempotent; SISTEER_KEEP_CUDNN_SDP=1 still wins — and
    # then the digest records that deviation, which is the point.
    from sisteer.models import configure_fast_sdpa
    configure_fast_sdpa()
    spec = cm.load_petri_spec(args.manifest)
    return COMMANDS[args.cmd](args, spec)


if __name__ == "__main__":
    raise SystemExit(main())
