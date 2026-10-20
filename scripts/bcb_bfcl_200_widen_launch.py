#!/usr/bin/env python
"""bcb-bfcl-200-widen — the LANE's launch wrapper (plan S3 consumer (1), S6, S7): one roster
wave = one `modal run` of the tinycap launcher at ONE sha, every runnable cell (`none` FIRST),
every runnable leg, under the benchmarks' `protocol` decoding — metered, watched, and bound to
the roster AS LAUNCHED.

    python scripts/bcb_bfcl_200_widen_launch.py --wave gemma4e4b_it --sha <40-hex> --stage-kind smoke \\
        --forecast-h 0.55 --inspect-limit 20 --tag smoke [--dry-run]
    python scripts/bcb_bfcl_200_widen_launch.py --wave gemma4e4b_it --sha <sha> --stage-kind wave --forecast-h 1.2
    python scripts/bcb_bfcl_200_widen_launch.py --wave gemma4e4b_it --sha <sha> --ladder-rerun \\
        --cap-override inspect/bfcl_200=16384 --item-selection <ladder_selection.json> --tag cap16k [--certify | --cell X]

What it does, in order (nothing else):
  1. READS the roster through the one reader (`widen_roster`) and REFUSES a roster failing its
     own `validate`, a wave with no runnable cell, a blocked leg asked for, an unmet prerequisite
     (printed by name — the orchestrator records it in `prerequisites_met`), a `--cell` that is
     not runnable, `--ladder-rerun` without its tuple, a `--stage-kind` that disagrees with the
     launch's step, and `glm46vflash` anywhere.
  2. COMPOSES the launch: app `tinycap-bcbw-<wave>-<sha7>-<cell|wave>-<hex6>` (unique per invocation — the
     nonce is the invocation id's, so abandoning one invocation can never stop a sibling's app),
     suffix `bcbw-<wave>-<sha7>[-<tag>]`,
     `--pf --pass1 --sampling protocol --model <row> --cells <runnable, none first> --modes off
     --serving-path lora_native --controls <runnable legs> --out-suffix <suffix>` (`--serving-path` is the
     lane's ONE serving-path authority, `widen_roster.SERVING_PATH`, which the roster header declares and
     `validate` holds it to — before this the wrapper composed none and native was never ESTABLISHED for
     the lane; the same flag rides on the sandbox watcher so it reads the `native/` attempt pointers);
     `--cell X[,Y]` (repeatable) NARROWS a plain smoke / wave to the named runnable cells in
     the given order, `none` first when present (the canary gate — `none`, then ONE arm, then the rest —
     used to be impossible because `--cell` was a ladder-rerun flag only); a ladder rerun adds the ladder tuple
     (`--cap-override` + `--item-selection`), the CERTIFIED WAVE's cells (`--ladder-wave-cells`,
     every runnable cell — the membership every certificate and the wave record are bound to,
     stated separately from the cells THIS launch scores), and `--ladder-certify` (step 1: every
     fitted cell's probe, the re-probes at the common rung, the wave record) or `--cell X
     [--ladder-probe-cell X]` (step 2: ONE cell's scoring at the common rung; a base / prompted
     cell states no probe cell and is admitted on the wave record).
  3. RENDERS the launch-time roster snapshot (`--roster-snapshot`): one block per launched cell
     — roster sha256, wave, entry, cell, object generation + superseding key + sha, launched_at,
     digest — which `run_configuration` writes into the cell's store as `lane_roster.json`
     (a fitted store's generation label is bound to the roster as launched, never the current
     one; immutable once filed) and whose object shas the preflight holds the served artifacts to.
  4. METERS the stage in the lane's ledger through `scripts/bcb_bfcl_200_meter.py` — a
     reservation per (wave, suffix, unit) with the stage KIND separate — and REFUSES
     the launch on the meter's refusal. An admission is the launch's CLAIM: a per-cell
     scoring launch claims `score:<cell>` (the whole-wave `score` stays for a whole-wave scoring),
     and whenever this invocation admitted units and then does NOT launch — a later unit's
     admission refused, the launcher gone before a run id, NO RUN ID — every claim it admitted is
     RELEASED (`--release "<why>"` per unit), both outcomes recorded in the refusal. A ladder SCORING stage is metered from the wave's
     per-cell certificates, which the meter reads LOCALLY: the wrapper first fetches every
     fitted cell's `rung_probe.<cell>.json` and the wave record `rung_admission.json` from the
     volume into a local cache and verifies each certificate's sha256 against the record's, the
     record's suffix and cell set against the launch's — for a base / prompted scoring cell too.
  5. LAUNCHES `modal run --detach` from `--tree` (a clean detached checkout at `--sha`), captures
     `[tinycap] run id:` from the BYTES this invocation appended to ITS OWN launch log, and ARMS
     the sandbox watcher in the same step (`scripts/tinycap_sandbox_watch.py`). Every local file
     (snapshot, launch log, watcher log, dry-run record, certificate cache) is keyed by an
     invocation id — suffix + scope + time + token — so concurrent per-cell scoring wrappers
     share nothing here. No run id within the wait terminates the local client, reconciles the
     remote app BY NAME (list, stop by id, list again) and releases the meter claims as "never
     ran" ONLY when the app is confirmed gone and no GPU fan-out was printed; otherwise the
     claims are RETAINED and the exact reconcile commands are printed.
     `--dry-run` composes and prints everything, runs the launcher's own CPU dry run (its whole
     output, never a tail), and touches neither the ledger nor Modal.

ROW MODE: `--row <model row>` launches EVERY runnable cell of
every roster wave whose `model_row` is the row in ONE container — one standup, per-cell drivers and stores —
so the qwen s1 row is `qwen35_4b_s1` + `qwen35_4b_s1_arditi` (9 cells, `none` deduplicated, the Arditi
cell's snapshot AND suffix from ITS wave: `bcbw-qwen35_4b_s1_arditi-<sha7>-shared` beside
`bcbw-qwen35_4b_s1-<sha7>-shared`, so the solo reader's `launch_tag_of` rule holds unchanged).
`--cells a,b` is the row-scoped SUBSET (a relaunch of parked cells under `--tag shared-r<N>`, §1.6.4; `none`
may be included); `--tag` is `shared`, `shared-r<N>`, `shared-negctl` (the five-item negative-identity smoke,
§1.3) or `warm` (the compile-cache warm probe, §1.5: suffix `…-warm`) — the tag decides the stage kind (only the smoke is `smoke`),
the store suffixes and the meter identity. The row's argv is today's `compose` argv + `--row`,
`--repeat-cells none`, (`--negative-control-cell X` | `--warm-cache-only`), `--concurrency-k K`,
`--cache-expectation warm|cold`, (`--no-cache-persist`), `--row-forecast-h H` (= `--forecast-h`, the meter
claim; the harness refuses ROW_TOO_LONG at plan time), (`--verification-phase generate`:
BigCodeBench generated unscored on the GPU and verified by ONE CPU container per cell after the row returns;
`coupled` is the default and spells nothing, so a landing-1 argv is byte-identical; `verify` is the CPU
verifier's own subcommand and refuses here) and `--out-suffix-by-cell <json>` — every spelling the
contracts' (`C.FLAG_ROW` …), which the harness's `cli_parser()` / `main` bind to as well. `--concurrency-k K`
is 1..3: K cell drivers at once on the one engine, each at client width S // K. The meter gets
ONE claim per row — `--unit row:<row>` under the row's primary suffix (§1.8: K overlapping drivers would
otherwise sum the same GPU minute K times) — and the watcher gets `--row-dir <gen root>/rows/<row>/<primary
suffix>` (§1.10), the ONE row-level place the harness files its attempt pointers and sweep certificates.

Pure functions (`refusals`, `compose`, `row_refusals`, `compose_row`, `snapshot_map`, `meter_argvs`, `score_unit`,
`release_argvs`, `watcher_argv`, `certificate_remote_paths`, `verify_certificate_cache`, `run_id_from_log`) are what
the tests pin; `main` wires them to subprocesses (`fetch_certificates`, `wait_for_run_id`,
`abandon_launch`, `release_claims` take the runner / process as arguments so the tests drive them
with doubles).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.capeval.tinycap import rung_probe as RP  # noqa: E402
from sisteer.capeval.tinycap import widen_roster as W  # noqa: E402

LAUNCHER_REL = "scripts/modal/tinycap_served.py"
METER_REL = "scripts/bcb_bfcl_200_meter.py"
WATCHER_REL = "scripts/tinycap_sandbox_watch.py"
MODAL_PY_DEFAULT = "~/venvs/modal/bin/python"
LEDGER_DEFAULT = "~/.local/state/bcb-bfcl-200-widen/gpu_ledger.jsonl"
LOG_DIR_DEFAULT = "$WORK/bcb-bfcl-200-widen/logs"   # under the site's working-data root; expanded through sisteer.site
#: the local, digest-verified copy of a wave's ladder certificates the meter reads
CERT_CACHE_DEFAULT = "~/.local/state/bcb-bfcl-200-widen/certificates"
APP_PREFIX = "tinycap-bcbw"
#: Modal app names: at most 64 characters of ``[A-Za-z0-9._-]``
MODAL_APP_NAME_MAX = 64
_APP_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")
#: a Modal app id as the launcher's ``modal run`` output prints it (``https://modal.com/apps/…/ap-<22>``)
APP_ID_RE = re.compile(r"\b(ap-[A-Za-z0-9]{22})\b")
SUFFIX_PREFIX = "bcbw"
#: the lane's decoding mode — the contracts' spelling, never a literal
SAMPLING_MODE = C.SAMPLING_PROTOCOL
#: the lane's SERVING PATH: the roster reader's ONE constant, composed as
#: `--serving-path` on the launcher argv AND the watcher argv, recorded on the Launch; the roster header
#: declares the same spelling and `widen_roster.validate` refuses a disagreement
SERVING_PATH = W.SERVING_PATH
SERVING_PATH_FLAG = C.FLAG_SERVING_PATH
PLAIN_STAGE_KINDS = ("smoke", "wave")
LADDER_STAGE_KINDS = ("ladder-probe", "ladder-score")
STAGE_KINDS = PLAIN_STAGE_KINDS + LADDER_STAGE_KINDS
RUN_ID_RE = re.compile(r"run id: ([A-Za-z0-9-]+)")
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
#: where the tinycap volume is mounted inside a container (`tinycap_served.VOL_PATH`): the
#: `modal volume get` CLI takes paths RELATIVE to the volume root, so every `/vol/...` path the
#: launcher records is translated by stripping this mount
VOLUME_MOUNT = "/vol"
#: how long the wrapper waits for the launcher to print its run id (the preflight runs first)
RUN_ID_WAIT_S = 20 * 60
RUN_ID_POLL_S = 5.0
#: the meter's scoring units (`bcb_bfcl_200_meter.SCORE_CELL_PREFIX`): the wave's whole scoring, or ONE cell's
SCORE_UNIT = "score"
SCORE_CELL_PREFIX = "score:"
#: the key the wave record travels under in :func:`certificate_remote_paths`
ADMISSION_KEY = "__admission__"
#: the launcher's line that marks the GPU fan-out (printed AFTER `run id:`): its presence in this
#: invocation's output is the evidence that billable work started (`gpu_work_started`)
GPU_FANOUT_MARKER = "[tinycap] launching "
#: modal's `app list --json` states that mean the app is not (yet) stopped
APP_LIVE_STATES = ("ephemeral", "ephemeral (detached)", "initializing...", "stopping...", "deployed", "unknown")
# --------------------------------------------------------------------------- #
# ROW MODE                                                                     #
# --------------------------------------------------------------------------- #
#: the meter unit of a row launch: ONE claim per (row, primary suffix) — `row:<model row>` (no per-cell claims)
ROW_UNIT_PREFIX = "row:"
#: the row dir's head under the gen root: `<VOL_GEN_ROOT>/rows/<row>/<primary suffix>` (OUTSIDE `native/`, so neither the
#: mirror's store walk nor the reader's `native/<row>/<cell>/thinking_off/<suffix>` binding reads it as a store; mirrored as a RECORD)
ROW_DIR_HEAD = "rows"
#: the row lane's tags (the only new suffix segments are `-shared`, the relaunch `-r<N>` after it, and the sidecars:
#: `-negctl` for the negative-identity smoke (§1.3) and `-warm` for the compile-cache warm probe (§1.5 / §3.1: `bcbw-<wave>-<sha7>-warm`, the
#: tag `warm` — module C's schedule contract); `-rep2` is the HARNESS's repeat suffix)
ROW_TAG_DEFAULT = "shared"
ROW_TAG_NEGCTL = "shared-negctl"
ROW_TAG_WARM = "warm"
#: a warm probe RE-RUN files under `warm<N>` (N ≥ 2: `warm2`, `warm3`, …) — the same probe kind under its own suffix, so a second
#: probe against the same tree neither collides with the done `-warm` row nor is read as it (two re-probes after the cache-key fix)
ROW_TAG_RE = re.compile(r"^(?:shared(?:-r(?P<relaunch>[1-9]\d*)|-negctl)?|warm(?P<warm_n>[2-9]|[1-9]\d+)?)$")


def is_warm_tag(tag: str) -> bool:
    """True for the compile-cache warm probe's tags: ``warm`` and its re-runs ``warm<N>`` (N ≥ 2)."""
    m = ROW_TAG_RE.match(str(tag or ""))
    return bool(m) and str(tag).startswith(ROW_TAG_WARM)
#: the harness's repeat-cell store suffix (plan S10 / §1.12: the in-run repeat `none` files under `<suffix>-rep2`)
REPEAT_SUFFIX_TEMPLATE = "{suffix}-rep{n}"
#: a roster wave whose name ends so is a rest wave of its seed view (the Arditi cell keeps ITS wave's suffix)
ARDITI_WAVE_SUFFIX = "_arditi"
CACHE_EXPECTATIONS = ("warm", "cold")
#: the harness's row-mode flags (module A implements them in `scripts/modal/tinycap_served.py`): ONE spelling, the contracts' — the
#: harness's `ROW_*_FLAG` names and its `cli_parser()` bind to the same constants (integration, 2026-09-28), so the argv composed here
#: is the argv the harness parses (`tests/test_tinycap_row_launch_grammar.py` parses a composed row argv through the harness's parser)
FLAG_ROW = C.FLAG_ROW
FLAG_REPEAT_CELLS = C.FLAG_REPEAT_CELLS
FLAG_NEGATIVE_CONTROL_CELL = C.FLAG_NEGATIVE_CONTROL_CELL
FLAG_WARM_CACHE_ONLY = C.FLAG_WARM_CACHE_ONLY
FLAG_CONCURRENCY_K = C.FLAG_CONCURRENCY_K
FLAG_CACHE_EXPECTATION = C.FLAG_CACHE_EXPECTATION
FLAG_NO_CACHE_PERSIST = C.FLAG_NO_CACHE_PERSIST
FLAG_OUT_SUFFIX_BY_CELL = C.FLAG_OUT_SUFFIX_BY_CELL
#: the row's forecast rides to the harness too (ROW_TOO_LONG refuses a forecast above 0.8 x the function timeout at PLAN
#: time, before any container) — the SAME number as the launcher's ONE `row:<row>` meter claim (`--forecast-h`), never a second estimate
FLAG_ROW_FORECAST_H = C.FLAG_ROW_FORECAST_H
#: the row launch's verification phase — the contracts' spelling; a ROW launch runs `coupled`
#: or `generate` (`verify` is the CPU verifier's subcommand, refused at compose time)
FLAG_VERIFICATION_PHASE = C.FLAG_VERIFICATION_PHASE
ROW_VERIFICATION_PHASES = (C.VERIFICATION_PHASE_COUPLED, C.VERIFICATION_PHASE_GENERATE)
#: landing 2, code review L2 H3 (b): the harness's CPU-only resume of a generate row (`tinycap_served.ROW_VERIFY_PENDING_FLAG`;
#: the contracts surface is frozen for this lane, so both spell it — pinned equal by test)
FLAG_VERIFY_PENDING = "--verify-pending"
#: tinycap-verify-rate-limit (2026-09-30): the CPU verifiers' dispatch WIDTH and per-verifier sandbox OVERRIDE — the contracts'
#: spelling, composed through to the harness only when given (the harness's defaults otherwise: width `C.VERIFY_WIDTH_DEFAULT`, the
#: marker's `min(S // K, 8)`); Modal creates at most `C.SANDBOX_CREATE_RATE_LIMIT_PER_S` sandboxes/s per workspace
FLAG_VERIFY_WIDTH = C.FLAG_VERIFY_WIDTH
FLAG_VERIFY_MAX_SANDBOXES = C.FLAG_VERIFY_MAX_SANDBOXES
#: the verifier's sandbox ceiling (`cpu_verify.MAX_SANDBOXES_CEILING`); spelled here so the launcher imports no driver module
VERIFY_MAX_SANDBOXES_CEILING = 8
#: the widest K the launcher composes (the harness refuses the same bound by name)
ROW_CONCURRENCY_K_MAX = 3
#: the watcher's row-dir mode flag (implemented in `scripts/tinycap_sandbox_watch.py`)
WATCHER_FLAG_ROW_DIR = "--row-dir"


class LaunchError(SystemExit):
    """A refusal, before anything is metered or launched."""


class Launch(NamedTuple):
    wave: str
    row: str
    sha: str
    app: str
    suffix: str
    cells: Tuple[str, ...]              # the cells THIS launch runs / scores (`none` first; one under --cell)
    controls: Tuple[str, ...]
    launcher_argv: Tuple[str, ...]      # after `modal run --detach scripts/modal/tinycap_served.py`
    env: Dict[str, str]
    snapshot: Dict[str, Dict[str, Any]]
    ladder: bool
    certify: bool
    watcher_cells: Tuple[str, ...]
    #: the WAVE's runnable cells (what a ladder certifies over; == `cells` on a plain launch)
    wave_cells: Tuple[str, ...]
    #: the ONE stage kind this launch is metered as (derived in `compose`, honoured by `main`)
    stage_kind: str
    #: the serving path REQUESTED on the launcher and the watcher argv (== `SERVING_PATH`). A request, never a
    #: fact: what actually served is the store's `route.serving_engine` / `serving_path` in its manifest, which
    #: the canary gate reads back and compares to this
    serving_path_requested: str
    # ---- ROW MODE (defaults keep every per-wave Launch byte-identical) ----
    #: True for a `compose_row` launch: one container for the row, ONE meter claim, the watcher on the row dir
    row_mode: bool = False
    #: the lane tag (`shared`, `shared-r<N>`, `shared-negctl`, `warm`); "" on a per-wave launch
    tag: str = ""
    #: every roster wave of the row, the PRIMARY (seed) wave first; == (wave,) on a per-wave launch
    waves: Tuple[str, ...] = ()
    #: the per-cell store suffix (`out_suffix(wave_of_cell, sha, tag)`): the Arditi cell under its own wave's tag; None per wave
    suffix_by_cell: Optional[Dict[str, str]] = None
    #: `<VOL_GEN_ROOT>/rows/<row>/<primary suffix>` — the row-level dir the watcher reads (`--row-dir`); None per wave
    row_dir: Optional[str] = None
    #: the cells the harness serves TWICE on the row engine (the in-run repeat floor); files under `<suffix>-rep2`
    repeat_cells: Tuple[str, ...] = ()
    #: the negative identity control (§1.3): the ONE fitted cell whose expected digest the harness flips, so exactly it refuses
    negative_control_cell: Optional[str] = None
    #: the warm probe (§1.5): standup → readiness → persist the compile cache → teardown; no driver
    warm_only: bool = False
    #: K concurrent cell drivers per engine (landing 1: 1; landing 2: 2 at `--concurrency S // K`)
    concurrency_k: int = 1
    #: what this launch DECLARES about the compile cache (`warm` after the same-shape probe, else `cold`), recorded so a cold reading is never a miss
    cache_expectation: Optional[str] = None
    #: whether the harness may persist the compile cache after a cold, attested readiness (never for a smoke / --inspect-limit run)
    cache_persist: bool = True
    #: the row's forecast H200-hours as passed to the harness (`--row-forecast-h`; = the meter claim); None per wave / when not given
    row_forecast_h: Optional[float] = None
    #: landing 2: `coupled` (the default; spelled nowhere — a landing-1 argv is byte-identical) or `generate`
    #: (BigCodeBench generated unscored on the GPU, verified on CPU containers after the row returns); per wave: coupled
    verification_phase: str = C.VERIFICATION_PHASE_COUPLED
    #: landing 2 (code review H3 b): the CPU-only resume of a generate row — the harness reads the row dir and spawns the
    #: verifiers for the still-pending stores; NO GPU container, so nothing is metered as a row claim
    verify_pending: bool = False
    #: tinycap-verify-rate-limit: verifiers in flight at once (None = the harness's default) and the per-verifier
    #: max_sandboxes override (None = the marker's min(S // K, 8)); both ride the argv only when given
    verify_width: Optional[int] = None
    verify_max_sandboxes: Optional[int] = None

    @property
    def row_unit(self) -> str:
        """The row's ONE meter unit (`row:<model row>`)."""
        return f"{ROW_UNIT_PREFIX}{self.row}"

    def modal_argv(self, modal_py: str) -> List[str]:
        return [os.path.expanduser(modal_py), "-m", "modal", "run", "--detach", LAUNCHER_REL, *self.launcher_argv]

    @property
    def fitted_wave_cells(self) -> Tuple[str, ...]:
        return tuple(c for c in self.wave_cells if C.parse_launcher_cell(c).kind == C.CELL_KIND_FITTED)


def cell_list(cell: Any) -> Optional[Tuple[str, ...]]:
    """``--cell`` as the wrapper reads it: None when absent; else the named cells, repeatable and / or
    comma-separated, blanks dropped, duplicates refused, in the GIVEN order."""
    if cell is None:
        return None
    parts = [cell] if isinstance(cell, str) else list(cell)
    out: List[str] = []
    for part in parts:
        for c in str(part).split(","):
            c = c.strip()
            if not c:
                continue
            if c in out:
                raise LaunchError(f"--cell names {c!r} twice")
            out.append(c)
    if not out:
        raise LaunchError("--cell names no cell")
    return tuple(out)


def refusals(roster: Mapping[str, Any], pins: Mapping[str, Any], models: Mapping[str, Any], *, wave: str,
             cell: Any = None, ladder: bool = False, tree: Optional[str] = None) -> List[str]:
    """Every reason NOT to launch ``wave`` (empty = go): the roster fails its own rules; the wave
    is unknown or has no runnable cell; a prerequisite is unmet; a ``cell`` (one, or a list)
    is not one of the wave's runnable cells; a ladder rerun of a base / prompted cell alone is fine
    (the wave record admits it), a ladder rerun without a fitted cell in the WAVE is meaningless."""
    out = W.validate(roster, pins, models, tree=tree)          # the launched tree reads its own configs + live verdict
    if out:
        return [f"roster: {p}" for p in out]
    if wave not in W.waves(roster):
        return [f"wave {wave!r} is not in the roster ({W.waves(roster)})"]
    cells = W.runnable_cells(roster, wave)
    if not cells:
        return [f"wave {wave!r} has no runnable cell — every entry is queued / blocked: "
                + "; ".join(f"{e}: {q['status']} ({q['reason']}; unblock: {q['unblock']})"
                            for e, q in W.queued_or_blocked(roster, wave).items())]
    w = W.wave(roster, wave)
    if not w.get("model_row"):
        return [f"wave {wave!r} names no model_row"]
    for p in W.prerequisites(roster, wave):
        out.append(f"wave {wave!r}: prerequisite NOT met — {p!r} (the orchestrator records it in "
                   "`prerequisites_met` once it holds)")
    if not W.runnable_legs(roster, wave):
        out.append(f"wave {wave!r} has no runnable leg")
    for c in (cell_list(cell) or ()):
        if c not in cells:
            out.append(f"--cell {c!r} is not a runnable cell of wave {wave!r} ({cells})")
    if ladder and not any(C.parse_launcher_cell(c).kind == C.CELL_KIND_FITTED for c in cells):
        out.append(f"wave {wave!r}: a ladder rerun needs a fitted cell to certify the footprint on")
    return out


def app_name(wave: str, sha: str, *, scope: Optional[str] = None, token: Optional[str] = None) -> str:
    """``tinycap-bcbw-<wave>-<sha7>`` and, given the invocation's ``token``,
    ``-<scope>-<token>`` (the remote app name used to identify only (wave, sha), so
    abandoning one per-cell scoring invocation stopped every sibling app sharing them). The scope
    (the ONE cell scored, else ``wave``) keeps the name descriptive and is shortened to fit Modal's
    64-character limit; the token keeps it unique. Refuses a base too long to carry a token."""
    base = f"{APP_PREFIX}-{wave}-{sha[:7]}"
    if token is None:
        return base
    tail = f"-{token}"
    room = MODAL_APP_NAME_MAX - len(base) - len(tail) - 1
    if room < 1:
        raise LaunchError(f"app name {base!r} leaves no room for the invocation tail within {MODAL_APP_NAME_MAX} characters")
    scope_tag = _APP_NAME_CHARS.sub("-", str(scope or "wave")).strip("-")[:room].rstrip("-") or "x"
    name = f"{base}-{scope_tag}{tail}"
    assert len(name) <= MODAL_APP_NAME_MAX and not _APP_NAME_CHARS.search(name), name
    return name


def out_suffix(wave: str, sha: str, tag: str = "") -> str:
    tag = str(tag or "").strip("-")
    return f"{SUFFIX_PREFIX}-{wave}-{sha[:7]}" + (f"-{tag}" if tag else "")


def snapshot_map(roster: Mapping[str, Any], wave: str, cells: Sequence[str], *, roster_sha256: str,
                 launched_at: str) -> Dict[str, Dict[str, Any]]:
    """The launch-time roster snapshot: ``{cell_id: block}`` for every launched cell."""
    return {c: W.snapshot(roster, wave, W.entry_of_cell(roster, wave, c), roster_sha256=roster_sha256,
                          launched_at=launched_at) for c in cells}


def resolve_stage_kind(requested: Optional[str], *, ladder: bool, certify: bool, cell: Optional[str]) -> str:
    """The ONE stage kind a launch is metered as. A plain launch is ``wave`` unless ``smoke`` is
    asked; a ladder rerun is ``ladder-probe`` (certify, or no ``--cell``) or ``ladder-score``
    (``--cell X``) — an explicit ``--stage-kind`` that disagrees with the launch's own step is
    REFUSED rather than silently overridden (code review: the ``ladder-*`` choices were accepted
    and ignored), and a ``ladder-*`` kind without a ladder rerun means nothing."""
    if requested is not None and requested not in STAGE_KINDS:
        raise LaunchError(f"--stage-kind {requested!r} not in {STAGE_KINDS}")
    if not ladder:
        # a plain launch narrowed by --cell is STILL a smoke / wave (the ladder-score derivation is the ladder's)
        if requested in LADDER_STAGE_KINDS:
            raise LaunchError(f"--stage-kind {requested} is a ladder rerun's stage (--ladder-rerun --cap-override … "
                              "--item-selection …); a plain launch is smoke or wave")
        return requested or "wave"
    derived = "ladder-probe" if (certify or cell is None) else "ladder-score"
    if requested is not None and requested != derived:
        step = "--certify" if certify else (f"--cell {cell}" if cell is not None else "whole-wave probe")
        raise LaunchError(f"--stage-kind {requested} disagrees with this rerun's step {derived} ({step})")
    return derived


def compose(roster: Mapping[str, Any], *, wave: str, sha: str, roster_sha256: str, launched_at: str,
            stage_kind: Optional[str] = None, inspect_limit: int = 0, tag: str = "", cell: Any = None,
            cap_override: str = "", item_selection: str = "", certify: bool = False, ladder_rerun: bool = False,
            ladder_reprobe: bool = False, snapshot_path: Optional[str] = None, extra: Sequence[str] = (),
            app_token: Optional[str] = None, legs: Any = None) -> Launch:
    """The launch, composed from the roster (refusals checked by :func:`refusals` first).
    ``app_token`` (the invocation's nonce) makes the remote app name unique to this invocation.
    ``cell``: on a PLAIN smoke / wave, one cell or a list (repeatable / comma-separated) of the
    wave's runnable cells — the launch runs exactly those, in the given order with ``none`` first
    when present, and the snapshot, the meter units and the watcher cells follow; on a ladder rerun
    exactly ONE cell (the scoring step), as before."""
    if not _SHA40.match(str(sha)):
        raise LaunchError(f"--sha {sha!r} is not a 40-hex commit sha")
    w = W.wave(roster, wave)
    row = str(w["model_row"])
    wave_cells = tuple(W.runnable_cells(roster, wave))
    controls = tuple(W.runnable_legs(roster, wave))
    ladder = bool(ladder_rerun or cap_override or item_selection)
    cells = cell_list(cell)
    if ladder and not (cap_override and item_selection):
        # code review: `--ladder-rerun` ALONE used to compose a PLAIN full-wave launch
        raise LaunchError("--cap-override and --item-selection travel together (the ladder's rung 2); "
                          "--ladder-rerun without them launches nothing")
    if certify and not ladder:
        raise LaunchError("--certify is a ladder-rerun flag (--ladder-rerun --cap-override … --item-selection …)")
    if certify and cells is not None:
        raise LaunchError("--certify certifies the whole wave's footprint; --cell scores one cell after it — not both")
    if ladder and cells is not None and len(cells) != 1:
        raise LaunchError(f"a ladder scoring step scores ONE cell at the common rung; --cell named {len(cells)}: {list(cells)}")
    if ladder_reprobe and not ladder:
        raise LaunchError(f"{C.FLAG_LADDER_REPROBE} qualifies a ladder rerun")
    one = cells[0] if cells is not None and len(cells) == 1 else None
    kind = resolve_stage_kind(stage_kind, ladder=ladder, certify=certify, cell=one)
    if cells is None:
        launched = wave_cells
    else:
        for c in cells:
            if c not in wave_cells:
                raise LaunchError(f"--cell {c!r} is not a runnable cell of wave {wave!r} ({list(wave_cells)})")
        # `none` FIRST when present (the same-wave anchor every contrast pairs against), the rest as given
        launched = tuple([c for c in cells if c == C.ROUTE_NONE_CELL] + [c for c in cells if c != C.ROUTE_NONE_CELL])
    if ladder:
        fired = sorted(C.parse_cap_overrides(cap_override))
        # a rerun serves the FIRED legs alone (tinycap_served.ladder_configurations refuses otherwise)
        controls = tuple(leg for leg in controls if C.inspect_task(leg) in fired)
        if not controls:
            raise LaunchError(f"--cap-override names {fired}, none of which is a runnable leg of {wave!r}")
    if legs is not None:
        # a plain launch may serve a SUBSET of the wave's runnable legs (one cell on one leg —
        # qwen35_4b_it difference LoRA on BFCL alone); a leg the roster does not run on the wave refuses by name
        if ladder:
            raise LaunchError("--legs is a plain-launch flag; a ladder rerun's legs are the fired caps (--cap-override)")
        wanted = tuple(cell_list(legs))
        for leg in wanted:
            if leg not in controls:
                raise LaunchError(f"--legs {leg!r} is not a runnable leg of wave {wave!r} ({list(controls)})")
        controls = tuple(leg for leg in controls if leg in wanted)      # the roster's order, never the flag's
    suffix = out_suffix(wave, sha, tag)
    app = app_name(wave, sha, scope=launch_scope(cells), token=app_token)
    snap = snapshot_map(roster, wave, launched, roster_sha256=roster_sha256, launched_at=launched_at)
    argv: List[str] = ["--sha", sha, "--app-name", app, "--pf", "--pass1", C.FLAG_SAMPLING, SAMPLING_MODE,
                       "--model", row, "--cells", ",".join(launched), "--modes", C.THINKING_OFF,
                       SERVING_PATH_FLAG, SERVING_PATH,
                       "--controls", ",".join(controls), "--out-suffix", suffix]
    if inspect_limit:
        argv += [C.FLAG_INSPECT_LIMIT, str(int(inspect_limit))]
    argv += [C.FLAG_ROSTER_SNAPSHOT, snapshot_path or json.dumps(snap, sort_keys=True, separators=(",", ":"))]
    if ladder:
        argv += [C.FLAG_CAP_OVERRIDE, cap_override, C.FLAG_ITEM_SELECTION, item_selection,
                 # the CERTIFIED WAVE's membership, separate from the cells this launch scores
                 C.FLAG_LADDER_WAVE_CELLS, ",".join(wave_cells)]
        if ladder_reprobe:
            argv.append(C.FLAG_LADDER_REPROBE)
        if certify:
            argv.append(C.FLAG_LADDER_CERTIFY)
        elif one is not None and C.parse_launcher_cell(one).kind == C.CELL_KIND_FITTED:
            argv += [C.FLAG_LADDER_PROBE_CELL, one]
    argv += list(extra)
    return Launch(wave=wave, row=row, sha=sha, app=app, suffix=suffix, cells=launched, controls=controls,
                  launcher_argv=tuple(argv), env={C.MODAL_APP_NAME_ENV: app}, snapshot=snap, ladder=ladder,
                  certify=bool(certify), watcher_cells=launched, wave_cells=wave_cells, stage_kind=kind,
                  serving_path_requested=SERVING_PATH)


def launch_scope(cells: Optional[Sequence[str]]) -> str:
    """The launch's SCOPE for the app name and the invocation id: ``wave`` for the whole wave, the ONE
    cell when one is named, ``<n>cells`` for a list of cells (the invocation nonce keeps the name unique)."""
    if not cells:
        return "wave"
    return cells[0] if len(cells) == 1 else f"{len(cells)}cells"


# --------------------------------------------------------------------------- #
# ROW MODE: one container per model row                                        #
# --------------------------------------------------------------------------- #


def row_waves(roster: Mapping[str, Any], row: str) -> Tuple[str, ...]:
    """Every roster wave whose ``model_row`` is ``row``, the PRIMARY wave first (the one plain seed / instruct wave), then the rest
    waves in roster order (qwen s1: ``qwen35_4b_s1``, ``qwen35_4b_s1_arditi``). Empty when no wave names the row."""
    named = [w for w in W.waves(roster) if str(W.wave(roster, w).get("model_row") or "") == row]
    plain = [w for w in named if not w.endswith(ARDITI_WAVE_SUFFIX)]
    rest = [w for w in named if w.endswith(ARDITI_WAVE_SUFFIX)]
    return tuple(plain + rest)


def primary_wave(waves: Sequence[str], *, row: str) -> str:
    """The row's PRIMARY wave: exactly one plain (non-Arditi) wave — its tag keys the row dir, the meter identity and the app name."""
    plain = [w for w in waves if not w.endswith(ARDITI_WAVE_SUFFIX)]
    if len(plain) != 1:
        raise LaunchError(f"row {row!r} has {len(plain)} plain (non-{ARDITI_WAVE_SUFFIX}) waves {plain} — the primary wave must be exactly one")
    return plain[0]


def row_cell_map(roster: Mapping[str, Any], waves: Sequence[str]) -> Dict[str, str]:
    """``{cell: wave}`` over the row's runnable cells in roster order — the primary wave's cells first, then each rest wave's NEW cells —
    with ``none`` first and deduplicated (the Arditi wave lists ``none`` too; the row serves it once, under the primary suffix)."""
    out: Dict[str, str] = {}
    for w in waves:
        for c in W.runnable_cells(roster, w):
            out.setdefault(c, w)
    ordered = [c for c in out if c == C.ROUTE_NONE_CELL] + [c for c in out if c != C.ROUTE_NONE_CELL]
    return {c: out[c] for c in ordered}


def row_tag_problem(tag: str) -> Optional[str]:
    """Why ``tag`` is not a row-lane tag (None = fine): ``shared``, ``shared-r<N>`` (N ≥ 1), ``shared-negctl``, ``warm``, ``warm<N>`` (N ≥ 2)."""
    if not ROW_TAG_RE.match(str(tag or "")):
        return (f"--tag {tag!r} is not a row-lane tag: shared | shared-r<N> (a row-scoped subset relaunch, N >= 1) | {ROW_TAG_NEGCTL} "
                f"(the negative-identity smoke) | {ROW_TAG_WARM} | {ROW_TAG_WARM}<N> (the compile-cache warm probe and its re-runs, N >= 2)")
    return None


def row_relaunch_n(tag: str) -> Optional[int]:
    """The relaunch number of a ``shared-r<N>`` tag, else None."""
    m = ROW_TAG_RE.match(str(tag or ""))
    return int(m.group("relaunch")) if m and m.group("relaunch") else None


def row_stage_kind(tag: str) -> str:
    """The stage kind a row launch is metered as: the negative-control smoke is ``smoke``; a row, a relaunch AND the warm probe are ``wave``
    (module C's meter contract: the probe is a real standup under its own ``-warm`` suffix; the schedule's items — same wave, same
    ``row:<row>`` unit — are told apart by their `tag`, which the wrapper passes to `bcbw_gate.py envelope --about-tag`)."""
    return "smoke" if tag == ROW_TAG_NEGCTL else "wave"


def row_dir_for(row: str, primary_suffix: str, *, gen_root: str = C.VOL_GEN_ROOT) -> str:
    """``<gen root>/rows/<row>/<primary suffix>`` — the harness's row dir; the watcher's ``--row-dir``."""
    return f"{gen_root.rstrip('/')}/{ROW_DIR_HEAD}/{row}/{primary_suffix}"


def repeat_suffix(suffix: str, n: int = 2) -> str:
    """The store suffix of the ``n``-th serving of a repeated cell on the row engine (the harness's spelling, plan S10)."""
    return REPEAT_SUFFIX_TEMPLATE.format(suffix=suffix, n=int(n))


def row_unit_of(row: str) -> str:
    return f"{ROW_UNIT_PREFIX}{row}"


def row_refusals(roster: Mapping[str, Any], pins: Mapping[str, Any], models: Mapping[str, Any], *, row: str, cells: Any = None,
                 tag: str = ROW_TAG_DEFAULT, tree: Optional[str] = None) -> List[str]:
    """Every reason NOT to launch ``row`` as one container (empty = go): the roster fails its own rules; no wave names the row, or
    its plain wave is not exactly one; the waves' runnable legs disagree; a wave's prerequisite is unmet; a ``cells`` subset names a
    cell no wave of the row runs; the tag is outside the lane's grammar."""
    out = W.validate(roster, pins, models, tree=tree)
    if out:
        return [f"roster: {p}" for p in out]
    waves = row_waves(roster, row)
    if not waves:
        return [f"row {row!r} is named by no wave of the roster (model_row of {W.waves(roster)})"]
    try:
        prim = primary_wave(waves, row=row)
    except LaunchError as e:
        return [str(e)]
    legs = tuple(W.runnable_legs(roster, prim))
    if not legs:
        out.append(f"row {row!r}: its primary wave {prim!r} has no runnable leg")
    for w in waves:
        if not W.runnable_cells(roster, w):
            out.append(f"row {row!r}: wave {w!r} has no runnable cell — " + "; ".join(
                f"{e}: {q['status']} ({q['reason']}; unblock: {q['unblock']})" for e, q in W.queued_or_blocked(roster, w).items()))
        if tuple(W.runnable_legs(roster, w)) != legs:
            out.append(f"row {row!r}: wave {w!r} runs legs {W.runnable_legs(roster, w)} but the primary wave {prim!r} runs {list(legs)} — "
                       "a row engine serves ONE leg set (D309: a one-leg store is non-comparable); block the odd wave or launch it alone")
        for p in W.prerequisites(roster, w):
            out.append(f"row {row!r}: wave {w!r} prerequisite NOT met — {p!r}")
    cell_map = row_cell_map(roster, waves)
    for c in (cell_list(cells) or ()):
        if c not in cell_map:
            out.append(f"--cells {c!r} is not a runnable cell of row {row!r} ({list(cell_map)})")
    problem = row_tag_problem(tag)
    if problem:
        out.append(problem)
    return out


def compose_row(roster: Mapping[str, Any], *, row: str, sha: str, roster_sha256: str, launched_at: str, tag: str = ROW_TAG_DEFAULT,
                legs: Any = None, cells: Any = None, repeat_cells: Any = None, negative_control_cell: Optional[str] = None,
                warm_only: bool = False, concurrency_k: int = 1, cache_expectation: str = "cold", cache_persist: Optional[bool] = None,
                stage_kind: Optional[str] = None, inspect_limit: int = 0, snapshot_path: Optional[str] = None, extra: Sequence[str] = (),
                app_token: Optional[str] = None, row_forecast_h: Optional[float] = None,
                verification_phase: str = C.VERIFICATION_PHASE_COUPLED, verify_pending: bool = False,
                verify_width: Optional[int] = None, verify_max_sandboxes: Optional[int] = None) -> Launch:
    """The ROW launch, composed from the roster (refusals checked by :func:`row_refusals` first): every runnable cell of
    every wave of the row — or the ``cells`` SUBSET — in ONE container at ONE sha, ``none`` first, the per-cell suffix keeping
    today's launch-tag grammar (``out_suffix(wave_of_cell, sha, tag)``), the row dir keyed by the primary wave's suffix, ONE stage kind
    derived from the tag. ``repeat_cells`` (default ``none``) are served twice for the in-run floor; ``negative_control_cell`` (a fitted
    launched cell; tag ``shared-negctl``; needs ``inspect_limit`` — the five-item smoke) flips that cell's expected digest;
    ``warm_only`` (tag ``warm``; the whole row's served set, no subset, no repeat) stands the engine up and persists the cache.
    ``cache_persist`` defaults to True and is forced False for a smoke (``inspect_limit`` / the negative control). ``row_forecast_h``
    (the launcher's ``--forecast-h``, the row's ONE meter claim) rides to the harness as ``--row-forecast-h`` so its plan-time
    ROW_TOO_LONG bound reads the same number the meter admits; None (a compose without a claim) spells no flag.
    ``verification_phase`` (landing 2): ``coupled`` (the default) spells NOTHING — a landing-1 argv is byte-identical;
    ``generate`` rides as ``--verification-phase generate`` before the suffix map (BigCodeBench generated unscored, the GPU released,
    the harness spawns ONE CPU verifier per BCB cell); ``verify`` is the CPU verifier's subcommand and refuses here. ``concurrency_k``
    is 1..ROW_CONCURRENCY_K_MAX concurrent cell drivers per engine at client width S // K."""
    if not _SHA40.match(str(sha)):
        raise LaunchError(f"--sha {sha!r} is not a 40-hex commit sha")
    if row_forecast_h is not None and not (float(row_forecast_h) > 0):
        raise LaunchError(f"--forecast-h {row_forecast_h!r} is not a positive number of H200-hours (a row launch is metered as ONE row:<row> claim)")
    problem = row_tag_problem(tag)
    if problem:
        raise LaunchError(problem)
    waves = row_waves(roster, row)
    if not waves:
        raise LaunchError(f"row {row!r} is named by no wave of the roster")
    prim = primary_wave(waves, row=row)
    cell_map = row_cell_map(roster, waves)
    wave_cells = tuple(cell_map)
    controls = tuple(W.runnable_legs(roster, prim))
    for w in waves:
        if tuple(W.runnable_legs(roster, w)) != controls:
            raise LaunchError(f"row {row!r}: wave {w!r} runs legs {W.runnable_legs(roster, w)} but {prim!r} runs {list(controls)}")
    if legs is not None:
        wanted = tuple(cell_list(legs))
        for leg in wanted:
            if leg not in controls:
                raise LaunchError(f"--legs {leg!r} is not a runnable leg of row {row!r} ({list(controls)})")
        controls = tuple(leg for leg in controls if leg in wanted)
    subset = cell_list(cells)
    if subset is None:
        launched = wave_cells
    else:
        for c in subset:
            if c not in cell_map:
                raise LaunchError(f"--cells {c!r} is not a runnable cell of row {row!r} ({list(cell_map)})")
        launched = tuple([c for c in subset if c == C.ROUTE_NONE_CELL] + [c for c in subset if c != C.ROUTE_NONE_CELL])
    kind = row_stage_kind(tag)
    if stage_kind is not None and stage_kind != kind:
        raise LaunchError(f"--stage-kind {stage_kind} disagrees with the row tag {tag!r}, which is metered as {kind}")
    k = int(concurrency_k)
    if k < 1:
        raise LaunchError(f"--concurrency-k {concurrency_k!r} must be >= 1")
    if k > ROW_CONCURRENCY_K_MAX:
        raise LaunchError(f"--concurrency-k {concurrency_k!r}: at most {ROW_CONCURRENCY_K_MAX} concurrent cell drivers per row engine (plan §2.1 / A5)")
    if verification_phase == C.VERIFICATION_PHASE_VERIFY:
        raise LaunchError(f"{FLAG_VERIFICATION_PHASE} {C.VERIFICATION_PHASE_VERIFY} is the CPU verifier's subcommand (served_capability_run.py verify, "
                          f"spawned by the harness after the row returns) — a row launch runs {list(ROW_VERIFICATION_PHASES)}")
    if verification_phase not in ROW_VERIFICATION_PHASES:
        raise LaunchError(f"{FLAG_VERIFICATION_PHASE} {verification_phase!r} not in {list(ROW_VERIFICATION_PHASES)}")
    if verification_phase == C.VERIFICATION_PHASE_GENERATE and warm_only:
        raise LaunchError(f"--warm-cache-only drives no cell; {FLAG_VERIFICATION_PHASE} {C.VERIFICATION_PHASE_GENERATE} means nothing beside it")
    if verify_pending:
        # landing 2 (code review H3 b): the CPU-only resume of a GENERATED row — the same row / cells / suffix (the row dir is keyed
        # by them), the generate phase spelled, no warm probe / negative control / GPU claim
        if verification_phase != C.VERIFICATION_PHASE_GENERATE:
            raise LaunchError(f"{FLAG_VERIFY_PENDING} resumes a {C.VERIFICATION_PHASE_GENERATE} row's CPU verification — pass "
                              f"{FLAG_VERIFICATION_PHASE} {C.VERIFICATION_PHASE_GENERATE} beside it (got {verification_phase!r})")
        if warm_only or negative_control_cell is not None:
            raise LaunchError(f"{FLAG_VERIFY_PENDING} launches no row: --warm-cache-only / --negative-control-cell mean nothing beside it")
        if row_forecast_h is not None:
            raise LaunchError(f"{FLAG_VERIFY_PENDING} spends no GPU: --forecast-h (the row's meter claim) means nothing beside it")
    if verify_width is not None or verify_max_sandboxes is not None:
        # tinycap-verify-rate-limit: the verifiers' knobs belong to a generate row (or its resume) — refused beside coupled
        if verification_phase != C.VERIFICATION_PHASE_GENERATE:
            raise LaunchError(f"{FLAG_VERIFY_WIDTH} / {FLAG_VERIFY_MAX_SANDBOXES} tune the CPU verifiers of a "
                              f"{C.VERIFICATION_PHASE_GENERATE} row — they mean nothing beside {FLAG_VERIFICATION_PHASE} {verification_phase!r}")
        if verify_width is not None and int(verify_width) < 1:
            raise LaunchError(f"{FLAG_VERIFY_WIDTH} must be >= 1 (verifiers in flight at once), got {verify_width!r}")
        if verify_max_sandboxes is not None and not 1 <= int(verify_max_sandboxes) <= VERIFY_MAX_SANDBOXES_CEILING:
            raise LaunchError(f"{FLAG_VERIFY_MAX_SANDBOXES} must be in 1..{VERIFY_MAX_SANDBOXES_CEILING}, got {verify_max_sandboxes!r}")
    if cache_expectation not in CACHE_EXPECTATIONS:
        raise LaunchError(f"--cache-expectation {cache_expectation!r} not in {CACHE_EXPECTATIONS}")
    # the smoke / the warm probe: the tag names the mode and the mode names the tag, so the suffix, the schedule item and the meter identity agree
    if negative_control_cell is not None and warm_only:
        raise LaunchError("--negative-control-cell and --warm-cache-only are two different launches (the smoke serves; the probe only stands up)")
    if negative_control_cell is not None:
        if tag != ROW_TAG_NEGCTL:
            raise LaunchError(f"a negative-control smoke files under --tag {ROW_TAG_NEGCTL} (plan §1.3: never mirrored, pruned at close-wave), not {tag!r}")
        if negative_control_cell not in launched:
            raise LaunchError(f"--negative-control-cell {negative_control_cell!r} is not one of the launched cells {list(launched)}")
        if C.parse_launcher_cell(negative_control_cell).kind != C.CELL_KIND_FITTED:
            raise LaunchError(f"--negative-control-cell {negative_control_cell!r} is not a fitted cell — only a fitted object has an expected digest to flip (E40 H1)")
        if not inspect_limit:
            raise LaunchError("a negative-control run is a SMOKE: pass --inspect-limit N (the plan's five items); it can never gate PASS and must not spend a full row")
    if warm_only:
        if not is_warm_tag(tag):
            raise LaunchError(f"the warm probe files under --tag {ROW_TAG_WARM} or {ROW_TAG_WARM}<N> (plan §1.5: its own row:<row> claim under a -warm / -warm<N> suffix), not {tag!r}")
        if subset is not None:
            raise LaunchError("--warm-cache-only stands up the row's WHOLE served set (the cache key folds in max_loras = the row's adapter count); --cells would warm another shape")
        if inspect_limit:
            raise LaunchError("--warm-cache-only runs no driver; --inspect-limit means nothing here")
    elif is_warm_tag(tag):
        raise LaunchError(f"--tag {tag} is the warm probe's tag; pass --warm-cache-only with it")
    if negative_control_cell is None and tag == ROW_TAG_NEGCTL:
        raise LaunchError(f"--tag {ROW_TAG_NEGCTL} is the negative-control smoke's tag; pass --negative-control-cell X with it")
    # the repeat: `none` by default WHEN the launch serves it (a subset relaunch without `none` repeats nothing — §1.6.4: its cells are
    # cross_standup comparisons and the r1 `none` stays the floor); an explicit list must name launched cells; "" / () = no repeat
    repeats: Tuple[str, ...] = ()
    if repeat_cells is None:
        repeats = (C.ROUTE_NONE_CELL,) if (C.ROUTE_NONE_CELL in launched and not warm_only) else ()
    elif not warm_only and repeat_cells not in ("", (), []):
        repeats = tuple(cell_list(list(repeat_cells) if isinstance(repeat_cells, (tuple, list)) else repeat_cells) or ())
    for c in repeats:
        if c not in launched:
            raise LaunchError(f"--repeat-cells {c!r} is not one of the launched cells {list(launched)} — a repeat rides the same engine")
    persist = (True if cache_persist is None else bool(cache_persist)) and not inspect_limit and negative_control_cell is None
    suffix_by_cell = {c: out_suffix(cell_map[c], sha, tag) for c in launched}
    suffix = out_suffix(prim, sha, tag)
    row_dir = row_dir_for(row, suffix)
    # a warm probe's app scope carries ITS tag (`row-warm`, `row-warm2`, …) so a re-probe reads as one in `modal app list`
    scope = f"row-{tag}" if warm_only else "row-negctl" if negative_control_cell is not None else ("row" if subset is None else f"row-{len(launched)}cells")
    app = app_name(prim, sha, scope=scope, token=app_token)
    snap = {c: W.snapshot(roster, cell_map[c], W.entry_of_cell(roster, cell_map[c], c), roster_sha256=roster_sha256, launched_at=launched_at)
            for c in launched}
    argv: List[str] = ["--sha", sha, "--app-name", app, "--pf", "--pass1", C.FLAG_SAMPLING, SAMPLING_MODE,
                       "--model", row, "--cells", ",".join(launched), "--modes", C.THINKING_OFF,
                       SERVING_PATH_FLAG, SERVING_PATH,
                       "--controls", ",".join(controls), "--out-suffix", suffix]
    if inspect_limit:
        argv += [C.FLAG_INSPECT_LIMIT, str(int(inspect_limit))]
    argv += [C.FLAG_ROSTER_SNAPSHOT, snapshot_path or json.dumps(snap, sort_keys=True, separators=(",", ":"))]
    argv.append(FLAG_ROW)
    if repeats:
        argv += [FLAG_REPEAT_CELLS, ",".join(repeats)]
    if negative_control_cell is not None:
        argv += [FLAG_NEGATIVE_CONTROL_CELL, negative_control_cell]
    if warm_only:
        argv.append(FLAG_WARM_CACHE_ONLY)
    argv += [FLAG_CONCURRENCY_K, str(k), FLAG_CACHE_EXPECTATION, cache_expectation]
    if not persist:
        argv.append(FLAG_NO_CACHE_PERSIST)
    if row_forecast_h is not None:
        argv += [FLAG_ROW_FORECAST_H, f"{float(row_forecast_h):g}"]
    if verification_phase != C.VERIFICATION_PHASE_COUPLED:
        # landing 2 (§2.2.3): the phase rides only when it is not the coupled default, so a landing-1 argv is byte-identical
        argv += [FLAG_VERIFICATION_PHASE, str(verification_phase)]
    if verify_pending:
        argv.append(FLAG_VERIFY_PENDING)
    # tinycap-verify-rate-limit: the verifiers' width / sandbox override ride right after the phase (and the resume flag),
    # each only when given — the harness's `launch_argv` spells them in the same place, so a solo argv is byte-identical
    if verify_width is not None:
        argv += [FLAG_VERIFY_WIDTH, str(int(verify_width))]
    if verify_max_sandboxes is not None:
        argv += [FLAG_VERIFY_MAX_SANDBOXES, str(int(verify_max_sandboxes))]
    argv += [FLAG_OUT_SUFFIX_BY_CELL, json.dumps(suffix_by_cell, sort_keys=True, separators=(",", ":"))]
    argv += list(extra)
    return Launch(wave=prim, row=row, sha=sha, app=app, suffix=suffix, cells=launched, controls=controls, launcher_argv=tuple(argv),
                  env={C.MODAL_APP_NAME_ENV: app}, snapshot=snap, ladder=False, certify=False, watcher_cells=launched, wave_cells=wave_cells,
                  stage_kind=kind, serving_path_requested=SERVING_PATH,
                  row_mode=True, tag=str(tag), waves=waves, suffix_by_cell=suffix_by_cell, row_dir=row_dir, repeat_cells=repeats,
                  negative_control_cell=negative_control_cell, warm_only=bool(warm_only), concurrency_k=k, cache_expectation=cache_expectation,
                  cache_persist=persist, row_forecast_h=(float(row_forecast_h) if row_forecast_h is not None else None),
                  verification_phase=str(verification_phase), verify_pending=bool(verify_pending),
                  verify_width=(int(verify_width) if verify_width is not None else None),
                  verify_max_sandboxes=(int(verify_max_sandboxes) if verify_max_sandboxes is not None else None))


# --------------------------------------------------------------------------- #
# the wave's ladder certificates, read LOCALLY by the meter (code review B-P1) #
# --------------------------------------------------------------------------- #


def invocation_id(suffix: str, *, scope: str, launched_at: str, token: Optional[str] = None) -> str:
    """The name every LOCAL file of one wrapper invocation is keyed by: the
    suffix (the volume's store / certificate relationship), the launch's scope (the ONE cell a
    scoring launch runs, else ``wave``), the launch time and a random token — so two wrappers
    scoring different cells of one suffix concurrently never share a snapshot, launch log,
    watcher log or certificate cache (B's single-cell snapshot used to overwrite A's before the
    launcher read it)."""
    stamp = re.sub(r"[^0-9T]", "", str(launched_at)[:19])          # YYYYmmddTHHMMSS (the UTC offset dropped)
    return f"{suffix}.{scope}.{stamp}-{token or secrets.token_hex(3)}"


def invocation_paths(log_dir: Path, cache_root: str, inv: str) -> Dict[str, Path]:
    """Every local path of the invocation ``inv``: the roster snapshot the launcher reads, the
    launch log the run id is read from, the watcher log, the dry-run record, the certificate cache."""
    log_dir = Path(log_dir)
    return {"snapshot": log_dir / f"roster_snapshot_{inv}.json", "launch_log": log_dir / f"{inv}.log",
            "watch_log": log_dir / f"{inv}_watch.log", "dry_log": log_dir / f"{inv}_dryrun.log",
            "cert_cache": Path(cache_root).expanduser() / inv}


def volume_relative(path: str) -> str:
    """A container path under :data:`VOLUME_MOUNT` as the ``modal volume`` CLI wants it."""
    p = str(path)
    if not p.startswith(VOLUME_MOUNT + "/"):
        raise LaunchError(f"{p!r} is not under the volume mount {VOLUME_MOUNT}/")
    return p[len(VOLUME_MOUNT) + 1:]


def certificate_remote_paths(launch: Launch) -> Dict[str, str]:
    """``{cell: container path}`` of every fitted WAVE cell's certificate plus the wave record
    (key :data:`ADMISSION_KEY`) — what the launcher filed under
    ``<VOL_GEN_ROOT>/<row>/_ladder/<suffix>/``."""
    out = {c: str(RP.certificate_path(C.VOL_GEN_ROOT, launch.row, launch.suffix, c)) for c in launch.fitted_wave_cells}
    out[ADMISSION_KEY] = str(RP.admission_path(C.VOL_GEN_ROOT, launch.row, launch.suffix))
    return out


def verify_certificate_cache(cache_dir: Path, launch: Launch) -> Dict[str, str]:
    """The fetched wave record + certificates, VERIFIED (never trusted because they were
    downloaded): the record parses and names this launch's suffix; its cell set == the wave's
    fitted cells; every certificate parses, is about the cell it is filed under, and hashes to
    the record's ``certificate_sha256`` for that cell; every certificate's accepted rung == the
    record's common rung. Returns ``{cell: local certificate path}`` for the meter."""
    fitted = launch.fitted_wave_cells
    if not fitted:
        raise LaunchError(f"wave {launch.wave!r} has no fitted cell — nothing certifies a ladder footprint")
    apath = cache_dir / RP.ADMISSION_FILENAME
    adoc, aerr = RP.read_admission(apath)
    if adoc is None:
        raise LaunchError(f"wave record {apath}: {aerr or 'absent'} — certify the wave first (--certify)")
    if str(adoc.get("suffix")) != launch.suffix:
        raise LaunchError(f"wave record {apath} is for suffix {adoc.get('suffix')!r}, not {launch.suffix!r}")
    recorded = adoc.get("cells") if isinstance(adoc.get("cells"), Mapping) else {}
    if set(recorded) != set(fitted):
        raise LaunchError(f"wave record {apath} names cells {sorted(recorded)} but the wave's fitted cells are "
                          f"{sorted(fitted)} — the record is not this wave's")
    common = dict(adoc.get("common") or {})
    local: Dict[str, str] = {}
    for cell in fitted:
        cpath = cache_dir / RP.certificate_filename(cell)
        cert, cerr = RP.read_certificate(cpath)
        if cert is None:
            raise LaunchError(f"certificate {cpath}: {cerr or 'absent'}")
        if str(cert.get("probed_cell")) != cell:
            raise LaunchError(f"certificate {cpath} is about {cert.get('probed_cell')!r}, not {cell!r}")
        want = str(recorded[cell].get("certificate_sha256")) if isinstance(recorded[cell], Mapping) else None
        have = RP.certificate_sha256(cert)
        if have != want:
            raise LaunchError(f"certificate {cpath} sha256 {have[:12]}… != the wave record's {str(want)[:12]}… — "
                              "not the certificate the record was written over (superseded or edited)")
        if dict(cert.get("accepted") or {}) != common:
            raise LaunchError(f"certificate {cpath} accepted {cert.get('accepted')!r} != the record's common rung {common!r}")
        local[cell] = str(cpath)
    return local


def fetch_certificates(launch: Launch, *, cache_dir: Path, modal_py: str,
                       runner: Optional[Callable[..., "subprocess.CompletedProcess[str]"]] = None) -> Dict[str, str]:
    """Fetch the wave's certificate set into ``cache_dir`` (this invocation's own — never shared
    with a concurrent wrapper; ``modal volume get --force``, one file at a time, every failure
    named) and :func:`verify_certificate_cache` it."""
    runner = subprocess.run if runner is None else runner          # resolved at CALL time (a test's double reaches it)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for key, remote in certificate_remote_paths(launch).items():
        local = cache_dir / (RP.ADMISSION_FILENAME if key == ADMISSION_KEY else RP.certificate_filename(key))
        argv = [os.path.expanduser(modal_py), "-m", "modal", "volume", "get", "--force", C.MODAL_VOLUME,
                volume_relative(remote), str(local)]
        r = runner(argv, text=True, capture_output=True)
        if r.returncode != 0:
            raise LaunchError(f"[bcbw] could not fetch {remote} from volume {C.MODAL_VOLUME} (rc {r.returncode}): "
                              f"{(r.stderr or r.stdout or '')[-1500:]}\n  {' '.join(shlex.quote(x) for x in argv)}")
        if not local.is_file():
            raise LaunchError(f"[bcbw] `modal volume get` exited 0 but wrote nothing at {local}")
    return verify_certificate_cache(cache_dir, launch)


def meter_argvs(launch: Launch, *, ledger: str, forecast_h: Optional[float], model_key: str,
                item_selection: str = "", certificate_paths: Optional[Mapping[str, str]] = None,
                python: str = sys.executable) -> List[List[str]]:
    """The meter invocations that ADMIT this launch (one reservation per (wave, suffix,
    unit) with the stage KIND separate — ``launch.stage_kind``): a smoke / wave reserves per
    launched cell at ``forecast_h / n_cells`` each; a ladder certification reserves one
    ``probe:<cell>`` per fitted WAVE cell from the actual selection (`--model` sizes the rungs);
    a scoring rerun — of a fitted, base OR prompted cell — reserves ``score`` with every fitted
    wave cell's LOCAL certificate (``certificate_paths``, the verified cache), which the meter
    reads from this filesystem."""
    base = [python, METER_REL, "--ledger", ledger, "--wave", launch.wave, "--suffix", launch.suffix]
    out: List[List[str]] = []
    stage_kind = launch.stage_kind
    if launch.row_mode:
        # ONE claim per row — `row:<row>` under the row's primary suffix, the WHOLE row's forecast (standup + Σ per-cell
        # drivers + the repeat) as one number. Never per cell: with K > 1 the per-cell actuals overlap and `committed_hours` would sum the
        # same GPU minute K times; a warm probe / smoke / relaunch is its own suffix, so its own reservation identity.
        if forecast_h is None:
            raise LaunchError(f"a row launch ({stage_kind}) needs --forecast-h (the WHOLE row's H200-hours: standup + every cell's driver + the repeat)")
        return [base + ["--unit", launch.row_unit, "--stage-kind", stage_kind, "--forecast-h", f"{float(forecast_h):.3f}"]]
    if stage_kind in PLAIN_STAGE_KINDS:
        if forecast_h is None:
            raise LaunchError(f"--stage-kind {stage_kind} needs --forecast-h (the whole stage's H200-hours)")
        per = float(forecast_h) / max(1, len(launch.cells))
        for cell in launch.cells:
            out.append(base + ["--unit", cell, "--stage-kind", stage_kind, "--forecast-h", f"{per:.3f}"])
        return out
    if not item_selection:
        raise LaunchError("a ladder stage is metered from the actual selection (--item-selection)")
    fitted = list(launch.fitted_wave_cells)
    if stage_kind == "ladder-probe":
        for cell in fitted:
            out.append(base + ["--unit", f"probe:{cell}", "--stage-kind", "ladder-probe", "--model", model_key,
                               "--selection", item_selection, "--fitted-cell", cell])
        return out
    if stage_kind == "ladder-score":
        paths = dict(certificate_paths or {})
        if set(paths) != set(fitted):
            raise LaunchError(f"a scoring stage is metered from the wave's per-cell certificates — have {sorted(paths)}, "
                              f"the wave's fitted cells are {sorted(fitted)} (fetch_certificates first)")
        argv = base + ["--unit", score_unit(launch.cells), "--stage-kind", "ladder-score", "--model", model_key,
                       "--selection", item_selection]
        for cell in fitted:
            argv += ["--certificate", paths[cell]]
        out.append(argv)
        return out
    raise LaunchError(f"--stage-kind {stage_kind!r} not in {STAGE_KINDS}")


def score_unit(cells: Sequence[str]) -> str:
    """The scoring claim's unit (Fixer C's contract §2): a ``--cell X`` launch claims ``score:<cell>``
    — its own claim, so the wave's other cells' scoring launches are not refused against it — and a
    whole-wave scoring launch claims ``score``."""
    return f"{SCORE_CELL_PREFIX}{cells[0]}" if len(cells) == 1 else SCORE_UNIT


def unit_of(meter_argv: Sequence[str]) -> str:
    """The ``--unit`` a meter invocation claims."""
    argv = list(meter_argv)
    return argv[argv.index("--unit") + 1]


def release_argvs(launch: Launch, *, ledger: str, units: Sequence[str], why: str,
                  python: str = sys.executable) -> List[List[str]]:
    """One ``--release`` per admitted unit (Fixer C's contract §1): closes the claim this invocation
    made and never launched, so the unit's retry is not refused against a dangling claim."""
    return [[python, METER_REL, "--ledger", ledger, "--release", why, "--wave", launch.wave, "--suffix", launch.suffix,
             "--unit", unit] for unit in units]


def release_claims(launch: Launch, *, ledger: str, units: Sequence[str], why: str, python: str = sys.executable,
                   runner: Optional[Callable[..., "subprocess.CompletedProcess[str]"]] = None) -> List[Dict[str, Any]]:
    """Run every release and RECORD each outcome (a failed release is reported in the refusal,
    never raised over it — the refusal is already the outcome, and the operator needs both)."""
    runner = subprocess.run if runner is None else runner
    out: List[Dict[str, Any]] = []
    for argv in release_argvs(launch, ledger=ledger, units=units, why=why, python=python):
        r = runner(argv, text=True, capture_output=True)
        out.append({"unit": unit_of(argv), "rc": r.returncode, "argv": argv,
                    "output": ((r.stdout or "") + (r.stderr or ""))[-800:]})
    return out


def _releases_text(released: Sequence[Mapping[str, Any]]) -> str:
    if not released:
        return "no meter claim to release"
    return "released " + ", ".join(f"{r['unit']} (rc {r['rc']})" for r in released)


def watcher_argv(run_id: str, launch: Launch, *, modal_py: str = MODAL_PY_DEFAULT, poll_s: int = 120,
                 max_hours: float = 6.0) -> List[str]:
    # the watcher reads the attempt pointers under the engine's out root (`native/` for lora_native;
    # `tinycap_sandbox_watch.cell_out_dir` through `contracts.gen_root_for_engine`) — without the flag it
    # looked under the LENS root for a native launch and would have watched nothing
    argv = [os.path.expanduser(modal_py), WATCHER_REL, "--run-id", run_id, "--suffix", launch.suffix,
            "--model", launch.row, "--cells", ",".join(dict.fromkeys(launch.watcher_cells)),
            SERVING_PATH_FLAG, launch.serving_path_requested]
    if launch.row_mode:
        # a row container's cells SHARE one FunctionCall id, which the per-cell mode refuses by design; the harness files ONE
        # declared row pointer (+ per-cell sweep certificates) under the row dir, and the watcher's `--row-dir` mode reads every pointer there
        argv += [WATCHER_FLAG_ROW_DIR, str(launch.row_dir)]
    return argv + ["--poll", str(int(poll_s)), "--max-hours", f"{max_hours:g}"]


def run_id_from_log(text: str, start: int = 0) -> Optional[str]:
    """The first ``run id:`` in ``text`` at or after CHARACTER offset ``start`` (text semantics;
    the launch log is read by BYTE offset through :func:`appended_text`)."""
    m = RUN_ID_RE.search((text or "")[start:])
    return m.group(1) if m else None


def appended_text(log: Path, start: int) -> str:
    """What THIS invocation appended to ``log``: the bytes from offset ``start`` (``fh.tell()``
    after the header), decoded on their own. Round 2: applying a byte offset to
    ``read_text()`` skipped past the run-id line whenever multibyte output preceded it, and a
    valid retry read as NO RUN ID."""
    return log.read_bytes()[start:].decode("utf-8", errors="replace")


def gpu_work_started(text: str) -> bool:
    """The launcher printed its GPU fan-out line (after ``run id:``): billable work began."""
    return GPU_FANOUT_MARKER in (text or "")


def wait_for_run_id(log: Path, proc, *, start: int, wait_s: Optional[float] = None, poll_s: Optional[float] = None,
                    clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep) -> Optional[str]:
    """Poll the launch log from byte offset ``start`` until a run id appears, the launcher exits,
    or ``wait_s`` (default :data:`RUN_ID_WAIT_S`, read at call time) passes (None then)."""
    wait_s = RUN_ID_WAIT_S if wait_s is None else wait_s
    poll_s = RUN_ID_POLL_S if poll_s is None else poll_s
    deadline = clock() + wait_s
    while True:
        run_id = run_id_from_log(appended_text(log, start))
        if run_id or proc.poll() is not None or clock() >= deadline:
            return run_id
        sleep(poll_s)


def live_apps_named(rows: Any, app: str, *, app_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """The ``modal app list --json`` rows that ARE this invocation's app and are not stopped: the
    description must equal the invocation-unique ``app`` name and, when the launcher's output named
    the app id, the row's ``app_id`` must equal it too (every app sharing (wave, sha) used to count
    as the failed invocation's)."""
    if not isinstance(rows, list):
        raise LaunchError(f"`modal app list --json` returned {type(rows).__name__}, not a list")
    return [r for r in rows if isinstance(r, Mapping) and r.get("description") == app
            and (app_id is None or r.get("app_id") == app_id)
            and str(r.get("state") or "unknown") in APP_LIVE_STATES]


def app_id_from_log(text: str) -> Optional[str]:
    """The Modal app id the launcher's ``modal run`` output names (its ``View run at …/ap-…`` line),
    or None — the EXACT remote object this invocation created."""
    m = APP_ID_RE.search(text or "")
    return m.group(1) if m else None


def abandon_launch(proc, launch: Launch, *, modal_py: str, appended: str = "",
                   runner: Optional[Callable[..., "subprocess.CompletedProcess[str]"]] = None) -> Dict[str, Any]:
    """No run id: the launch is NOT handed over unwatched (code review). The local ``modal run``
    client is terminated (then killed); the remote app is RECONCILED — ``modal app list --json``,
    ONLY the live row(s) that are THIS invocation's app (the invocation-unique description; and the
    app id parsed from the launcher's own output in ``appended`` when it printed one) are stopped
    BY ID, the list is read again — and ``remote_stopped`` is True ONLY when the list succeeded and
    no such live row remains (a non-zero stop used to be followed by an unconditional release,
    and a name shared by sibling invocations made the stop kill their apps).
    A parsed app id whose listed description is another app's is a refusal to stop anything
    (recorded), never a stop. Every outcome is recorded."""
    runner = subprocess.run if runner is None else runner
    py = os.path.expanduser(modal_py)
    app_id = app_id_from_log(appended)
    rec: Dict[str, Any] = {"client_rc": proc.poll(), "terminated": False, "killed": False, "app": launch.app,
                           "app_id": app_id, "stops": [], "remote_stopped": False, "live_before": None, "live_after": None,
                           "problems": []}
    if rec["client_rc"] is None:
        proc.terminate()
        rec["terminated"] = True
        try:
            rec["client_rc"] = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            rec["killed"] = True
            rec["client_rc"] = proc.wait(timeout=30)

    def listed() -> Optional[List[Dict[str, Any]]]:
        argv = [py, "-m", "modal", "app", "list", "--json"]
        r = runner(argv, text=True, capture_output=True)
        if r.returncode != 0:
            rec["problems"].append(f"`{' '.join(argv)}` rc {r.returncode}: {(r.stderr or r.stdout or '')[-600:]}")
            return None
        try:
            rows = json.loads(r.stdout or "[]")
            if app_id is not None:
                mine = [x for x in rows if isinstance(x, Mapping) and x.get("app_id") == app_id]
                if mine and mine[0].get("description") != launch.app:
                    rec["problems"].append(f"app id {app_id} (from the launcher's output) is listed as "
                                           f"{mine[0].get('description')!r}, not this invocation's {launch.app!r} — nothing stopped")
                    return None
            return live_apps_named(rows, launch.app, app_id=app_id)
        except (ValueError, LaunchError) as e:
            rec["problems"].append(f"`{' '.join(argv)}` output unreadable: {e}")
            return None

    live = listed()
    rec["live_before"] = live
    if live is None:
        return rec
    for row in live:
        argv = [py, "-m", "modal", "app", "stop", str(row.get("app_id"))]
        r = runner(argv, text=True, capture_output=True)
        rec["stops"].append({"app_id": row.get("app_id"), "state": row.get("state"), "argv": argv, "rc": r.returncode,
                             "output": ((r.stdout or "") + (r.stderr or ""))[-800:]})
    after = listed() if live else []
    rec["live_after"] = after
    rec["remote_stopped"] = after is not None and not after
    if after:
        rec["problems"].append("live app(s) remain after the stop: " + ", ".join(f"{r.get('app_id')} ({r.get('state')})" for r in after))
    return rec


def reconcile_commands(launch: Launch, *, ledger: str, units: Sequence[str], modal_py: str,
                       python: str = sys.executable) -> List[str]:
    """The exact commands the operator runs when the wrapper could NOT confirm the remote app is
    gone: list, stop by id, then release each retained claim."""
    py = os.path.expanduser(modal_py)
    out = [f"{py} -m modal app list --json   # find description == {launch.app!r} (this invocation's name alone)",
           f"{py} -m modal app stop <app_id>"]
    out += [" ".join(shlex.quote(x) for x in argv)
            for argv in release_argvs(launch, ledger=ledger, units=units, why="never ran (reconciled by hand)", python=python)]
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roster", default=str(W.ROSTER_PATH))
    ap.add_argument("--wave", default=None, help="a per-wave launch (today's path); exclusive with --row")
    ap.add_argument("--row", default=None,
                    help="ROW MODE (tinycap-shared-standup §1.6): every runnable cell of every wave whose model_row is this row, in ONE container "
                         "(one standup; per-cell drivers and stores); exclusive with --wave, --cell and the ladder flags")
    ap.add_argument("--cells", action="append", default=None,
                    help="row mode: the row-scoped SUBSET to launch (repeatable / comma-separated; a relaunch of parked cells under --tag shared-r<N>, §1.6.4)")
    ap.add_argument("--repeat-cells", default=None,
                    help="row mode: cells served TWICE on the row engine for the in-run floor (comma-separated; default: none when the launch serves it; "
                         "'' for no repeat)")
    ap.add_argument("--negative-control-cell", default=None,
                    help=f"row mode: the ONE fitted cell whose expected digest the harness flips (§1.3 smoke; needs --tag {ROW_TAG_NEGCTL} and --inspect-limit)")
    ap.add_argument("--warm-cache-only", action="store_true", help=f"row mode: the compile-cache warm probe (§1.5; needs --tag {ROW_TAG_WARM} or {ROW_TAG_WARM}<N> for a re-run, suffix …-warm / …-warm<N>); no driver runs")
    ap.add_argument("--concurrency-k", type=int, default=1,
                    help=f"row mode: K concurrent cell drivers per engine at client width S // K (1..{ROW_CONCURRENCY_K_MAX}; landing 2, plan §2.1)")
    ap.add_argument("--verification-phase", default=C.VERIFICATION_PHASE_COUPLED, choices=list(ROW_VERIFICATION_PHASES),
                    help="row mode (landing 2, plan §2.2): coupled (the default; BigCodeBench verified in the GPU container) or generate "
                         "(BCB generated unscored, the GPU released, ONE CPU verifier per BCB cell spawned by the harness)")
    ap.add_argument("--verify-pending", action="store_true",
                    help="row mode (landing 2, code review H3): the CPU-only RESUME of a GENERATED row — the harness reads the row dir "
                         "(row_rec.json + markers + manifests), spawns the CPU verifier for every still-pending store, skips the verified "
                         "ones and parks torn ones by name; NO GPU row (needs --verification-phase generate; no --forecast-h)")
    ap.add_argument("--verify-width", dest="verify_width", type=int, default=None,
                    help=f"row mode (generate / --verify-pending): CPU verifiers in flight at once (the harness's default {C.VERIFY_WIDTH_DEFAULT} when "
                         f"not given; Modal creates at most {C.SANDBOX_CREATE_RATE_LIMIT_PER_S} sandboxes/s per workspace and each verifier opens "
                         "up to max_sandboxes)")
    ap.add_argument("--verify-max-sandboxes", dest="verify_max_sandboxes", type=int, default=None,
                    help=f"row mode (generate / --verify-pending): override every verifier's max_sandboxes (the marker's min(S // K, 8)); "
                         f"1..{VERIFY_MAX_SANDBOXES_CEILING}; recorded on the spawn's pending payload and the verified block")
    ap.add_argument("--cache-expectation", default="cold", choices=list(CACHE_EXPECTATIONS),
                    help="row mode: what this launch declares about the compile cache (warm after the same-shape probe); recorded, never inferred")
    ap.add_argument("--no-cache-persist", action="store_true", help="row mode: never persist the compile cache (forced for a smoke / --inspect-limit run)")
    ap.add_argument("--standup-forecast-h", type=float, default=None,
                    help="row mode: the standup's share of --forecast-h (recorded in the launch log for the meter's reconcile; the claim is --forecast-h)")
    ap.add_argument("--sha", required=True, help="the 40-hex sha the tree at --tree is checked out at (staged on the volume)")
    ap.add_argument("--tree", default=None, help="the clean detached checkout to launch from (default: this repo)")
    ap.add_argument("--stage-kind", default=None, choices=list(STAGE_KINDS),
                    help="wave (default) or smoke for a plain launch; a ladder rerun derives ladder-probe / ladder-score "
                         "and refuses a kind that disagrees with its step")
    ap.add_argument("--forecast-h", type=float, default=None, help="the stage's H200-hour forecast (smoke / wave)")
    ap.add_argument("--inspect-limit", type=int, default=0, help="a smoke's per-task item cap (0 = the whole 200)")
    ap.add_argument("--tag", default="", help="suffix tag (bcbw-<wave>-<sha7>-<tag>); e.g. smoke, cap16k")
    ap.add_argument("--ledger", default=LEDGER_DEFAULT)
    ap.add_argument("--log-dir", default=LOG_DIR_DEFAULT)
    ap.add_argument("--cert-cache", default=CERT_CACHE_DEFAULT,
                    help="the local, digest-verified copy of the wave's ladder certificates the meter reads")
    ap.add_argument("--modal-py", default=MODAL_PY_DEFAULT)
    ap.add_argument("--ladder-rerun", action="store_true", help="the truncation ladder's rung 2 (with --cap-override + --item-selection)")
    ap.add_argument("--cap-override", default="")
    ap.add_argument("--item-selection", default="")
    ap.add_argument("--certify", action="store_true", help="ladder step 1: certify the wave's footprint (no scoring)")
    ap.add_argument("--cell", action="append", default=None,
                    help="a plain smoke / wave: run ONLY these runnable cells (repeatable / comma-separated; `none` first when "
                         "present — D90's canary gate); a ladder rerun: step 2, score ONE cell at the common rung")
    ap.add_argument("--ladder-reprobe", action="store_true")
    ap.add_argument("--legs", default=None,
                    help="a plain smoke / wave: serve ONLY these runnable legs of the wave (comma-separated leg atoms, e.g. bfcl_full) — D307's "
                         "one-leg cells; default every runnable leg")
    ap.add_argument("--no-watcher", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def row_mode_problems(a: argparse.Namespace) -> List[str]:
    """The CLI's row / wave exclusivity: exactly one of --row / --wave; --row takes none of the per-wave flags."""
    out: List[str] = []
    if bool(a.row) == bool(a.wave):
        out.append("exactly one of --row <model row> (row mode) and --wave <roster wave> (a per-wave launch) is required")
    if a.row:
        for flag, val in (("--cell", a.cell), ("--ladder-rerun", a.ladder_rerun), ("--cap-override", a.cap_override),
                          ("--item-selection", a.item_selection), ("--certify", a.certify), ("--ladder-reprobe", a.ladder_reprobe)):
            if val:
                out.append(f"{flag} is a per-wave flag; row mode takes --cells for a subset and no ladder rerun")
    else:
        for flag, val in (("--cells", a.cells), ("--negative-control-cell", a.negative_control_cell), ("--warm-cache-only", a.warm_cache_only),
                          ("--no-cache-persist", a.no_cache_persist), ("--standup-forecast-h", a.standup_forecast_h)):
            if val:
                out.append(f"{flag} is a row-mode flag (--row <model row>)")
        if a.concurrency_k != 1:
            out.append("--concurrency-k is a row-mode flag (--row <model row>)")
        if a.verification_phase != C.VERIFICATION_PHASE_COUPLED:
            out.append("--verification-phase is a row-mode flag (--row <model row>)")
        if a.verify_pending:
            out.append("--verify-pending is a row-mode flag (--row <model row>)")
        if a.verify_width is not None:
            out.append(f"{FLAG_VERIFY_WIDTH} is a row-mode flag (--row <model row>)")
        if a.verify_max_sandboxes is not None:
            out.append(f"{FLAG_VERIFY_MAX_SANDBOXES} is a row-mode flag (--row <model row>)")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    roster_path = Path(a.roster).expanduser()
    roster = W.load_roster(roster_path)
    pins = W.load_pins()
    problems = row_mode_problems(a)
    if problems:
        raise LaunchError("[bcbw] REFUSED:\n  - " + "\n  - ".join(problems))
    ladder = bool(a.ladder_rerun or a.cap_override or a.item_selection)
    if a.row:
        problems = row_refusals(roster, pins, C.MODELS, row=a.row, cells=a.cells, tag=a.tag or ROW_TAG_DEFAULT, tree=a.tree)
    else:
        problems = refusals(roster, pins, C.MODELS, wave=a.wave, cell=a.cell, ladder=ladder, tree=a.tree)
    if problems:
        raise LaunchError("[bcbw] REFUSED:\n  - " + "\n  - ".join(problems))
    launched_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    from sisteer import site as _site
    log_dir = _site.expand_path(a.log_dir, key="--log-dir")   # an unset $WORK refuses by name
    log_dir.mkdir(parents=True, exist_ok=True)
    roster_sha = W.roster_sha256(roster_path)
    # ONE nonce per invocation keys every LOCAL file (round 2) AND the remote app name (round 3): two
    # wrappers on one suffix share nothing here, and abandoning one can never stop the other's app
    token = secrets.token_hex(3)
    if a.row:
        tag = a.tag or ROW_TAG_DEFAULT
        waves = row_waves(roster, a.row)
        prim = primary_wave(waves, row=a.row)
        subset = cell_list(a.cells)
        scope = f"row-{tag}" if a.warm_cache_only else "row-negctl" if a.negative_control_cell else ("row" if subset is None else f"row-{len(subset)}cells")
        inv = invocation_id(out_suffix(prim, a.sha, tag), scope=scope, launched_at=launched_at, token=token)
        paths = invocation_paths(log_dir, a.cert_cache, inv)
        snap_path = paths["snapshot"]
        launch = compose_row(roster, row=a.row, sha=a.sha, roster_sha256=roster_sha, launched_at=launched_at, tag=tag, legs=a.legs, cells=a.cells,
                             repeat_cells=a.repeat_cells, negative_control_cell=a.negative_control_cell, warm_only=a.warm_cache_only,
                             concurrency_k=a.concurrency_k, cache_expectation=a.cache_expectation, cache_persist=not a.no_cache_persist,
                             stage_kind=a.stage_kind, inspect_limit=a.inspect_limit, snapshot_path=str(snap_path), app_token=token,
                             row_forecast_h=a.forecast_h, verification_phase=a.verification_phase, verify_pending=a.verify_pending,
                             verify_width=a.verify_width, verify_max_sandboxes=a.verify_max_sandboxes)
    else:
        inv = invocation_id(out_suffix(a.wave, a.sha, a.tag), scope=launch_scope(cell_list(a.cell)), launched_at=launched_at, token=token)
        paths = invocation_paths(log_dir, a.cert_cache, inv)
        snap_path = paths["snapshot"]
        launch = compose(roster, wave=a.wave, sha=a.sha, roster_sha256=roster_sha, launched_at=launched_at, legs=a.legs,
                         stage_kind=a.stage_kind, inspect_limit=a.inspect_limit, tag=a.tag, cell=a.cell,
                         cap_override=a.cap_override, item_selection=a.item_selection, certify=a.certify,
                         ladder_rerun=a.ladder_rerun, ladder_reprobe=a.ladder_reprobe, snapshot_path=str(snap_path),
                         app_token=token)
    snap_path.write_text(json.dumps(launch.snapshot, indent=1, sort_keys=True) + "\n")
    row = C.model_row(launch.row)
    tree = Path(a.tree).expanduser() if a.tree else REPO
    if launch.row_mode:
        print(f"[bcbw] ROW {launch.row} ({row.hf}@{row.rev[:12]}) = waves {list(launch.waves)} -> ONE container; app {launch.app}, primary suffix "
              f"{launch.suffix}, stage {launch.stage_kind} (tag {launch.tag}), invocation {inv}")
        print(f"[bcbw] row dir {launch.row_dir} (the watcher's --row-dir; pointers + sweep certificates live there, never in a cell store)")
        print(f"[bcbw] per-cell suffixes: " + ", ".join(f"{c}={s}" for c, s in launch.suffix_by_cell.items()))
        print(f"[bcbw] repeat {list(launch.repeat_cells) or 'none'} (-> <suffix>-rep2); negative control {launch.negative_control_cell}; "
              f"warm-only {launch.warm_only}; concurrency K={launch.concurrency_k}; cache expectation {launch.cache_expectation}; "
              f"cache persist {launch.cache_persist}; verification phase {launch.verification_phase}"
              + ("; CPU-only resume of the pending verification (--verify-pending: no GPU row)" if launch.verify_pending else "")
              + (f"; verifiers {launch.verify_width} in flight ({FLAG_VERIFY_WIDTH})" if launch.verify_width is not None else "")
              + (f"; max_sandboxes {launch.verify_max_sandboxes} per verifier ({FLAG_VERIFY_MAX_SANDBOXES})" if launch.verify_max_sandboxes is not None else "")
              + (f"; standup forecast {a.standup_forecast_h:g} h of the row's --forecast-h" if a.standup_forecast_h is not None else ""))
    else:
        print(f"[bcbw] wave {launch.wave} -> row {launch.row} ({row.hf}@{row.rev[:12]}), app {launch.app}, suffix {launch.suffix}, "
              f"stage {launch.stage_kind}, invocation {inv}")
    print(f"[bcbw] cells {list(launch.cells)} (wave {list(launch.wave_cells)}); legs {list(launch.controls)}; "
          f"sampling {SAMPLING_MODE}; thinking off; serving path REQUESTED {launch.serving_path_requested} (D86; the store's "
          "route.serving_engine is the fact, D94)")
    print(f"[bcbw] roster snapshot -> {snap_path} (roster sha256 {roster_sha[:12]}…)")
    for cell, block in launch.snapshot.items():
        print(f"[bcbw]   {cell}: generation {block['object_generation']} superseded_by {block['superseded_by_key']} "
              f"sha {str(block['object_sha256'])[:12] if block['object_sha256'] else None}")
    certificate_paths: Optional[Dict[str, str]] = None
    if launch.stage_kind == "ladder-score":
        remote = certificate_remote_paths(launch)
        print("[bcbw] scoring is metered from the wave's certificates, fetched and digest-verified locally first:")
        for key, path in remote.items():
            print(f"[bcbw]   {'wave record' if key == ADMISSION_KEY else key}: {path}")
        if a.dry_run:
            # the dry run touches no volume: the meter line below names the LOCAL cache paths it would read
            certificate_paths = {c: str(paths["cert_cache"] / RP.certificate_filename(c)) for c in launch.fitted_wave_cells}
        else:
            certificate_paths = fetch_certificates(launch, cache_dir=paths["cert_cache"], modal_py=a.modal_py)
            print(f"[bcbw] certificates verified against the wave record: {json.dumps(certificate_paths, sort_keys=True)}")
    meters = meter_argvs(launch, ledger=a.ledger, forecast_h=a.forecast_h, model_key=row.model_key,
                         item_selection=a.item_selection, certificate_paths=certificate_paths)
    for m in meters:
        print("[bcbw] meter: " + " ".join(shlex.quote(x) for x in m))
    modal_argv = launch.modal_argv(a.modal_py)
    print(f"[bcbw] launch (cwd {tree}): {C.MODAL_APP_NAME_ENV}={launch.app} " + " ".join(shlex.quote(x) for x in modal_argv))
    print("[bcbw] watcher (armed after the run id appears): " + " ".join(watcher_argv("<run id>", launch, modal_py=a.modal_py)))
    if a.dry_run:
        # the launcher's OWN CPU dry run, from the tree — plans, gates the geometry, prints the argv.
        # It runs the launcher's argparse `__main__` (not `modal run`), so it needs the interpreter
        # that has `modal` importable (the launcher imports it at module top) and must NOT carry
        # `--pf`: the preflight is a `modal run`-only local-entrypoint flag (a Modal CPU function);
        # the direct dry run plans without it. Its WHOLE output is the record (never a tail:
        # the early gate messages are the ones that matter when it refuses).
        dry = [os.path.expanduser(a.modal_py), LAUNCHER_REL,
               *[x for x in launch.launcher_argv if x != "--pf"], "--dry-run"]
        env = dict(os.environ, **launch.env, PYTHONPATH=f"{tree}/src")
        print("[bcbw] dry run: " + " ".join(shlex.quote(x) for x in dry))
        r = subprocess.run(dry, cwd=str(tree), env=env, text=True, capture_output=True)
        dry_log = paths["dry_log"]
        dry_log.write_text(f"== dry run {launched_at} {' '.join(dry)}\n== stdout\n{r.stdout}\n== stderr\n{r.stderr}\n"
                           f"== rc {r.returncode}\n")
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        print(f"[bcbw] dry run record -> {dry_log}")
        if r.returncode != 0:
            raise LaunchError(f"[bcbw] the launcher's dry run exited {r.returncode} — nothing metered, nothing launched")
        print("[bcbw] dry run: nothing metered, nothing launched")
        return 0
    admitted: List[str] = []                               # the claims THIS invocation made (released if it does not launch)
    for m in meters:
        r = subprocess.run(m, cwd=str(tree), text=True, capture_output=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        if r.returncode != 0:
            released = release_claims(launch, ledger=a.ledger, units=admitted,
                                      why=f"a later unit's admission was refused ({unit_of(m)}, rc {r.returncode}); nothing launched")
            raise LaunchError(f"[bcbw] METER REFUSED {unit_of(m)} (rc {r.returncode}) — nothing launched; "
                              f"{_releases_text(released)}")
        admitted.append(unit_of(m))
    log = paths["launch_log"]
    env = dict(os.environ, **launch.env, PYTHONPATH=f"{tree}/src")
    # code review R2-3: the claims above are admitted; a LOCAL failure between here and a live child (the log cannot
    # be opened, `--modal-py` is missing / not executable, Popen raises) is a spawn that PROVABLY did not happen —
    # no child exists — so every admitted unit is released and the release outcomes are reported with the refusal
    # (a failure AFTER Popen returned is the NO RUN ID path below, which needs the remote evidence)
    try:
        fh = log.open("a")
    except OSError as e:
        why = f"never ran: the launch log {log} could not be opened ({type(e).__name__}: {e}); nothing spawned"
        released = release_claims(launch, ledger=a.ledger, units=admitted, why=why)
        raise LaunchError(f"[bcbw] SPAWN FAILED before any child: {why}; meter claims: {_releases_text(released)}")
    with fh:
        fh.write(f"== launch {launched_at} {' '.join(modal_argv)}\n")
        fh.flush()
        start = fh.tell()                                   # THIS invocation's output begins here (a BYTE offset)
        try:
            proc = subprocess.Popen(modal_argv, cwd=str(tree), env=env, stdout=fh, stderr=subprocess.STDOUT)
        except (OSError, ValueError) as e:                  # FileNotFoundError / PermissionError / NotADirectoryError: no child
            why = (f"never ran: the launcher could not be spawned ({type(e).__name__}: {e}; {shlex.quote(modal_argv[0])} "
                   f"is {'missing' if not os.path.exists(modal_argv[0]) else 'not runnable'}); nothing spawned")
            fh.write(f"== spawn failed: {why}\n")
            released = release_claims(launch, ledger=a.ledger, units=admitted, why=why)
            raise LaunchError(f"[bcbw] SPAWN FAILED before any child: {why}; meter claims: {_releases_text(released)}; "
                              "nothing is left running or claimed")
    print(f"[bcbw] launcher pid {proc.pid}; log {log} (run id read from byte {start})")
    run_id = wait_for_run_id(log, proc, start=start)
    if not run_id:
        rc = proc.poll()
        state = f"launcher exited {rc}" if rc is not None else f"still running after {RUN_ID_WAIT_S // 60} min"
        appended = appended_text(log, start)
        rec = abandon_launch(proc, launch, modal_py=a.modal_py, appended=appended)
        started = gpu_work_started(appended)
        client = 'killed' if rec['killed'] else 'terminated' if rec['terminated'] else 'already gone'
        stops = "; ".join(f"stop {s['app_id']} ({s['state']}) -> rc {s['rc']}" for s in rec["stops"]) or "no live app listed by name"
        head = (f"[bcbw] NO RUN ID in {log} ({state}) — the local client was {client} (rc {rec['client_rc']}); remote app "
                f"{launch.app!r}: {stops}")
        # round 2: a claim is released as "never ran" ONLY with evidence — the remote app confirmed gone
        # (listed by name, stopped by id, no live row left) AND no GPU fan-out in this invocation's output (the
        # launcher prints `run id:` BEFORE `launching N configuration(s)`, so no run id means no GPU container
        # unless that line is present); otherwise the claims are RETAINED and the reconcile commands printed
        if rec["remote_stopped"] and not started:
            released = release_claims(launch, ledger=a.ledger, units=admitted,
                                      why=f"never ran: NO RUN ID ({state}), no GPU fan-out, app {launch.app} confirmed stopped")
            raise LaunchError(f"{head}; confirmed stopped; meter claims: {_releases_text(released)}; nothing is left running "
                              "unwatched or claimed")
        commands = "\n  ".join(reconcile_commands(launch, ledger=a.ledger, units=admitted, modal_py=a.modal_py))
        if started:
            why = ("GPU work STARTED in this invocation (the launcher's fan-out line is in the log) — its hours are billable: "
                   "read the app's logs, then `--record --actual-h <h>` each claim instead of releasing it")
        else:
            why = ("the remote app could NOT be confirmed stopped (" + "; ".join(rec["problems"] or ["see the stops above"])
                   + ") — it may still be running unwatched")
        raise LaunchError(f"{head}; meter claims RETAINED ({', '.join(admitted) or 'none'}): {why}. Reconcile by hand:\n  {commands}")
    print(f"[bcbw] run id {run_id}")
    if not a.no_watcher:
        wargv = watcher_argv(run_id, launch, modal_py=a.modal_py)
        wlog = paths["watch_log"]
        with wlog.open("a") as fh:
            wp = subprocess.Popen(wargv, cwd=str(tree), env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"[bcbw] watcher pid {wp.pid}; log {wlog}")
    print(f"[bcbw] LAUNCHED {launch.suffix} run_id={run_id} app={launch.app}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
