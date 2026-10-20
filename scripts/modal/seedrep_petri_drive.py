"""The seedrep battery's Petri DRIVE (the grading host, plain python, not a Modal app) — from
``scripts/modal/arditi_allpos_petri_drive.py`` (arditi-qwen-all-positions @8cab86fd, read-only), re-parameterised
for the Qwen seed-replicate native battery.

ONE PROCESS PER PROXY. The wave stands up FOUR Petri proxies in ``seedrep_served.py``: one view engine
per training seed (``s1``, ``s2``, ``s3``: the view's 5 fitted cells + ``none`` + the two prompted routes = 8
arm-runs each, every request under seed base s) and ONE Arditi engine (``ar_c0p3__arditi_allpos``, run at seed
bases 1, 2, 3 = 3 seed-runs). This driver drives ONE of them (``--view``), the orchestrator launches four, and
the watcher (``seedrep_watch.py --petri-drive``) follows each. Per proxy it does, in the cluster driver's order:

  1. PROVENANCE: worktree HEAD == --sha (W), clean under configs/scripts/src; the proxy's tree sha, when it
     differs, must be audit-side identical.
  2. THE PROXY: the tunnel record from the wave's sentinel Dict FIRST (``seedrep-<sha7>-<wave_uuid8>-sentinels``,
     six-axis key ``<sha7>/<wave_uuid>/petri/<view>/<base|petri-rerun>/<shard>/modal_tunnel.json``), the volume
     copy second; ``endpoint.json`` + ``routing_table.json`` + ``cells_served.json`` off the attempt root the
     record names EXPLICITLY (``attempt_root_on_volume`` + ``attempt_id`` are REQUIRED fields -- never derived
     from ``serving_dir_on_volume``, whose layout carries ``/exp/`` twice); asserts ``mode petri``
     / ``engine lora_native``; the container call must still be RUNNING. The record is BOUND to this drive
     before anything is read off the volume: its ``control_key`` must equal this drive's own
     six-axis key exactly and its attempt root must lie under ``/vol/seedrep/<sha7>/``, so a live tunnel of
     another wave or W found at a discovery location is refused by name; ``--serving-vol`` is only where to
     LOOK for the record's volume copy -- it must equal the record's ``serving_dir_on_volume`` and never
     replaces it.
  3. THE TUNNEL at LOOPBACK: ``petri_target.target_client_config`` refuses a non-loopback base URL by design,
     so a 127.0.0.1 TCP -> TLS relay (SNI = the modal.forward host) carries the legs.
  4. THE ENVIRONMENT: the auditor + judge key is HIGHPRIO ONLY through the vault's
     mechanism (:func:`auditor_env`: ``sisteer.vault.anthropic_api_key("highprio")`` -- the org's OWN env var
     else the secret-manager CLI; never a file, never a literal, never argv; any other org REFUSED, a key-shaped
     org REFUSED); ``PETRI_EVAL_MAX_CONNECTIONS`` = 3 per view arm-run, 4 per Arditi seed-run
     (72 + 12 = 84 concurrent auditor conversations <= the standing sonnet highprio ceiling 100); the chain's
     third auditor leg may point at a seedrep attacker pool (:func:`qwen_auditor_env`).
  5. THE SCHEDULE (:func:`petri_schedule`): the enumeration frozen under ``--serving-path lora_native`` and held
     to the standup's evidence. The native enumeration (``petri_run.enumerate_petri_native``) lists the
     manifest's cells PLUS the derived no-op control ``none__lora0`` (canonical then controls; ``--cell-index``
     indexes THAT order), so the lane's arms are SELECTED from the complete enumeration by the rows' own fields,
     never counted blind: a view proxy runs its 8 arms = every non-control row (5 fitted + ``none``
     + 2 prompted routes; the derived control is excluded and logged), the Arditi proxy runs ``arditi_allpos``
     alone (``none`` and the control excluded; a SECOND object on that standup is refused) x 3
     seed bases, each seed-run in its OWN out root (the run key is manifest-derived, so two seed bases in one
     root would share it). Fewer served arms than the lane expects is REFUSED with the count named (a 6-cell
     view is never audited as "the view"). Every leg: ``petri_run.py ... run --out-root ...
     --cell-index i --target-decode vendor --target-seed-base <s>`` (``i`` is the enumeration's
     own index), three attempts on EVERY pass -- only rc 0 / 3 end the loop, every other rc goes round again
     -- the width THROTTLED DOWN 3 -> 2 -> 1 on a 429-heavy attempt (never a pool hop), the
     per-leg return record bound to the drive id + the results bytes. The FIRST leg of every distinct
     out root launches at once; only a leg sharing a root with a live earlier leg waits for that root's
     materialised bundle (:func:`launch_legs`).
  6. HEARTBEAT every 5 min (armed BEFORE the legs launch) + the COMPLETION into the sentinel Dict (the container
     mirrors them into the proxy's $OUT): ONE atomic bundle -- the verdict record + the STOP content carrying
     ``attempt=<id>`` -- under the ATTEMPT-NAMESPACED key ``<K>/<attempt_id>/completion.json`` (:func:`completion_key`,
     :func:`publish_completion`), BOUND to the proxy attempt whose tunnel record is re-read at that moment (the served app
     accepts nothing unbound -- a counts-only STOP is its ``stopped_unbound``, rc 1; a replacement attempt installing
     its record between the read and the write cannot misread a bundle keyed by ANOTHER attempt id; the
     generic ``<K>/drive_verdict.json`` + ``<K>/STOP`` pair only under ``--legacy-generic-completion``). A missing or
     another attempt's record publishes NOTHING (rc 2: the stores stand locally; a drive bound
     to the live attempt continues them with ``--resume-from``). STOP only when no leg FAILED (the cluster driver's
     release rule, on every pass -- the re-run pass included); a FAILED leg HOLDS the proxy -- nothing published,
     the served app's wait would end on a verdict -- unless the orchestrator asks, by name, for
     ``--teardown-after-failure`` (:func:`release_decision`). ``--width`` may only NARROW a view below its
     planned width (the 84-of-100 ceiling is budgeted from the planned widths and re-checked over
     the EFFECTIVE widths).
  7. WAVE/ATTEMPT-SCOPED STATE: every writable root carries (sha7, wave_uuid8, attempt_id) -- the out
     roots ``~/.local/state/petri_native/petri_seedrep_<sha7>_<wave_uuid8>_<attempt_id>_<tag>`` and the attempt
     state dir ``~/.local/state/seedrep_battery/petri/<sha7>/<wave_uuid8>/<view>[_rerun]/attempts/<attempt_id>/``
     -- so a new wave (or a re-executed proxy attempt) at the same W starts from EMPTY roots and can never adopt
     another wave's stores. Continuing a prior wave/attempt's stores is EXPLICIT and provenance-preserving:
     ``--resume-from <prior attempt state dir>`` copies its out roots' ``cells/`` into this attempt's roots
     (never over an existing one) and records where they came from in ``RESUMED_FROM.json`` beside each root
     and in the drive record -- ATOMICALLY: every source and destination is preflighted, the stores are staged
     with their provenance in hidden sibling roots and published by rename, and any failure rolls the published
     roots back so no destination exists without its provenance. IDEMPOTENT RESUME within an
     attempt: a re-invocation skips a leg ONLY when the
     Petri driver's OWN store reconciliation (``petri_run._already_done`` under the leg python: manifest-bound
     cell dir, results naming its cell + run, a settled identity-bearing ledger, the footing recomputed from the
     store) bears its results.json out -- never on the file's existence -- re-runs the rest (petri_run resumes
     by run key inside a leg), and records itself as a NEW drive (``drive_<id>.json``).

Usage (the grading host; the driver python is the modal venv's, the LEG python a venv with the `petri` dependency group):

  ~/venvs/modal/bin/python scripts/modal/seedrep_petri_drive.py --sha <W 40hex> --wave-json RS/wave.json \
      --view s1 --python ~/venvs/seedrep-petri-drive/bin/python [--legs ALL] [--petri-pass overflow-rerun] \
      [--qwen-auditor-pool $STATE/sisteer/attacker-pool-modal/seedrep/endpoint_0.json] \
      [--resume-from $STATE/seedrep_battery/petri/<W-sha7>/<prior wave_uuid8>/s1/attempts/<prior attempt_id>] [--dry-run]
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
import posixpath
import re
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

try:
    LOCAL_REPO = Path(__file__).resolve().parents[2]
except IndexError:
    LOCAL_REPO = Path.cwd()
if (LOCAL_REPO / "src").is_dir() and str(LOCAL_REPO / "src") not in sys.path:
    sys.path.insert(0, str(LOCAL_REPO / "src"))

# the chain's third-auditor env names, owned by the reader (a stdlib-only module; imports from the modal venv too)
from sisteer.capeval.petri.auditor_fallback import QWEN_API_KEY_ENV, QWEN_BASE_URL_ENV  # noqa: E402
# the engine reader of a standup's endpoint.json (names only; CPU-light): the drive reads the engine THROUGH it, never a lens fact
from sisteer.capeval.tinycap.contracts import endpoint_is_native, served_reduction_mode  # noqa: E402

# the lane's generator (beside this file): the grading host LANE bank root + the bank-view derivation the stager uses (F-R) -- ONE spelling
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import seedrep_launch_plan as lp  # noqa: E402

LANE = "seedrep"
LEG = "petri"
VOL_NAME = "twosi-fits"
VOL_PATH = "/vol"
MODEL_KEY = "qwen35_4b"
if lp.MODEL_KEY != MODEL_KEY:
    raise RuntimeError(f"seedrep_launch_plan.MODEL_KEY {lp.MODEL_KEY!r} != this drive's {MODEL_KEY!r} -- one target per lane")
VIEWS: Tuple[str, ...] = ("s1", "s2", "s3", "arditi")
SEED_VIEWS: Tuple[str, ...] = ("s1", "s2", "s3")
ARDITI_ARM = "arditi_allpos"
ARDITI_SEED_BASES: Tuple[int, ...] = (1, 2, 3)
#: stage axis for the Petri proxies: the base proxy and the context-overflow re-run proxy
STAGES: Dict[str, str] = {"": "base", "overflow-rerun": "petri-rerun"}
STAGE_VALUES: Tuple[str, ...] = ("base", "2048", "8192", "16384", "petri-rerun")
DICT_PURPOSES: Tuple[str, ...] = ("accepted", "claims", "results", "heartbeats", "sentinels", "stop")
DEFAULT_SHARD = "0"
HEARTBEAT_NAME, STOP_NAME, TUNNEL_FILE = "driver_heartbeat", "STOP", "modal_tunnel.json"
#: the attempt-bound completion record the proxy accepts (its PETRI_VERDICT_NAME; `petri_completion_evidence` reads `attempt_id`,
#: `drive_id`, `verdict.{counts, nruns, release}`): published under this drive's six-axis key, BEFORE the STOP that carries `attempt=<id>`
VERDICT_NAME = "drive_verdict.json"
VERDICT_SCHEMA = "seedrep_petri_drive_verdict/1"
#: the ATTEMPT-NAMESPACED completion: ONE atomic bundle -- the verdict record + the STOP content -- under
#: ``<K>/<attempt_id>/completion.json`` (:func:`completion_key`). A generic ``<K>/...`` key can be read by a REPLACEMENT proxy attempt that
#: installed its record between this drive's binding read and its writes (the TOCTOU); a key carrying the attempt id cannot. The generic
#: ``<K>/drive_verdict.json`` + ``<K>/STOP`` pair is written ONLY under ``--legacy-generic-completion`` (a proxy that still watches them).
COMPLETION_NAME = "completion.json"
COMPLETION_SCHEMA = "seedrep_petri_drive_completion/1"
RESUME_SCHEMA = "seedrep_petri_drive_resume/1"
#: the tunnel record fields that identify the proxy ATTEMPT this drive is bound to; the record under the key is re-read at publish time
#: and must agree on every one of them, else the attempt is gone (missing) or replaced (stale) and nothing is published
BINDING_FIELDS = ("attempt_id", "attempt_root_on_volume", "control_key")
HEARTBEAT_PERIOD_S = 300
#: F-AR: the local relay port is a FREE OS-assigned loopback port (`local_port_of`), never proxy_port + an offset -- see there
LOCAL_PORT_MIN = 1024
#: the Petri auditor (sonnet-5) and judge (opus-5) run on HIGHPRIO -- the ONLY admissible org
AUDITOR_ORG = "highprio"
VIEW_WIDTH, ARDITI_WIDTH, AUDITOR_CEILING = 3, 4, 100
#: 5 fitted cells + `none` + prompted__si_constitution + prompted__safety_refusal
EXPECTED_VIEW_ARM_RUNS = 8
EXPECTED_ARDITI_CELLS = 1
BUNDLE_HASH = "8812ba083bb5473c"
ARDITI_CFG_REL = "configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep.yaml"
ARDITI_MANIFEST_REL = "configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep_manifest.yaml"
#: the container copies the tree + bank under its own root; the proxy's serve dir is under its exp tree
DEFAULT_CONTAINER_EXP_ROOT = "/workspace/seedrep/exp"
DEFAULT_RETRY_SLEEP_CAP_S = 300
#: sustained 429s in one attempt's log -> the next attempt runs one connection narrower (never a pool hop)
THROTTLE_429_THRESHOLD = 20
_429_RE = re.compile(r"\b429\b|rate.?limit", re.I)
AUDIT_SIDE_PATHS = ("configs", "src", "scripts/petri_run.py", "scripts/sdfbase/full22/make_bank_view.py", "scripts/sdfbase/full22/f22_roster.py")
TARGET_DECODE = "vendor"
#: the proxy container's attempt id (MODAL_TASK_ID + a uuid4 suffix, minted at function entry) as the tunnel
#: record carries it -- a path component of every writable root here, so it is held to a path-safe alphabet
ATTEMPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
#: what a tunnel record (``run_petri_proxy`` writes it) MUST carry for this drive to trust it. ``control_key`` is the
#: record's OWN six-axis key ``control_key(wave_uuid, "petri", view, stage, shard)`` (the Dict key it sits under, minus the
#: name): the drive compares it EXACTLY with its own, so a live tunnel found at a discovery location can never bind another
#: wave's or another W's proxy to this drive
TUNNEL_REQUIRED_KEYS = ("url", "function_call_id", "proxy_port", "control_key", "serving_dir_on_volume", "attempt_root_on_volume",
                        "attempt_id", "view", "stage")
RESUMED_FROM_FILE = "RESUMED_FROM.json"
#: three attempts per leg on EVERY pass; the sleeps between them (the cluster driver's)
LEG_ATTEMPTS, PROXY_UNREACHABLE_SLEEP_S, ATTEMPT_SLEEP_S = 3, 30, 20
#: a leg sharing an out root with a LIVE earlier leg waits for that root's materialised bundle at most this long
BUNDLE_WAIT_CAP_S = 900
LAUNCH_STAGGER_S = 2


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class DriveRefused(SystemExit):
    """A drive-time refusal (the cluster driver's FATAL): named, never a fallback."""

    def __init__(self, msg: str) -> None:
        super().__init__(f"[seedrep-petri-drive] {msg}")


# --------------------------------------------------------------------------- control-plane grammar (pinned by test)

def w7_of(sha: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", sha or ""):
        raise DriveRefused(f"--sha {sha!r} must be the full 40-hex sha")
    return sha[:7]


def wave_uuid8(wave_uuid: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", (wave_uuid or "").lower()):
        raise DriveRefused(f"wave_uuid {wave_uuid!r} is not a uuid4 string (RS/wave.json, minted at launch)")
    return wave_uuid.lower()[:8]


def dict_name(w7: str, wave_uuid8_: str, purpose: str) -> str:
    """``seedrep-<W-sha7>-<wave_uuid8>-<purpose>`` -- every persistent Dict of the wave."""
    if purpose not in DICT_PURPOSES:
        raise ValueError(f"Dict purpose {purpose!r} not in {DICT_PURPOSES}")
    if not re.fullmatch(r"[0-9a-f]{7}", w7) or not re.fullmatch(r"[0-9a-f]{8}", wave_uuid8_):
        raise ValueError(f"dict_name needs W7 (7 hex) and wave_uuid8 (8 hex), got {w7!r} {wave_uuid8_!r}")
    return f"{LANE}-{w7}-{wave_uuid8_}-{purpose}"


def control_key(w7: str, wave_uuid: str, leg: str, view: str, stage: str, shard: str, name: str = "") -> str:
    """The six-axis key ``(wave_uuid, leg, view|arditi, stage_or_cap, shard)`` joined by ``/`` (+ ``/<name>``
    for a named record under it). Every axis is required and slash-free, so a base attempt and its rung
    attempts never collide and a dead wave's state is unreachable by construction."""
    axes = {"W7": w7, "wave_uuid": wave_uuid, "leg": leg, "view": view, "stage": stage, "shard": shard}
    for k, v in axes.items():
        if not isinstance(v, str) or not v or "/" in v:
            raise ValueError(f"control_key axis {k}={v!r} must be a non-empty slash-free string")
    if not re.fullmatch(r"[0-9a-f]{7}", w7):
        raise ValueError(f"control_key W7 {w7!r} is not 7 hex")
    wave_uuid8(wave_uuid)
    if stage not in STAGE_VALUES:
        raise ValueError(f"control_key stage {stage!r} not in {STAGE_VALUES}")
    key = "/".join(axes.values())
    if name:
        if "/" in name:
            raise ValueError(f"control_key name {name!r} must be slash-free")
        key += "/" + name
    return key


# --------------------------------------------------------------------------- per-view parameters (pure, tested)

@dataclasses.dataclass(frozen=True)
class ViewParams:
    view: str
    seed_bases: Tuple[int, ...]
    width: int
    expected_cells: int                 # served cells the enumeration must hold (per proxy)
    cfg_rel: str
    manifest_rel: str
    bank_root_rel: str                  # the grading host bank ROOT the view lives under, relative to $HOME (the LANE bank; the Arditi ladder)
    bank_view_rel: str                  # the grading host bank view, relative to $HOME (``<bank_root_rel>/.../cells_view/<view>``)

    @property
    def is_arditi(self) -> bool:
        return self.view == "arditi"

    @property
    def expected_runs(self) -> int:
        return self.expected_cells * len(self.seed_bases)


def view_params(view: str) -> ViewParams:
    """A seed view's grading-host bank view is the LANE bank's (``lp.grading_host_bank_view_rel``: ``$STAGE/sdfbase/seedrep_battery/bank/
    <MODEL_KEY>/cells_view/<view>``, what ``00_build_views.sh`` builds and ``seedrep_served --stage-bank`` uploads) -- NEVER the peers'
    shared ``$STAGE/sdfbase/full22_objects``, which holds only their views and refused every battery view (F-R)."""
    if view in SEED_VIEWS:
        s = int(view[1:])
        return ViewParams(view=view, seed_bases=(s,), width=VIEW_WIDTH, expected_cells=EXPECTED_VIEW_ARM_RUNS,
                          cfg_rel=f"configs/sdfbase_full22_{MODEL_KEY}_{view}.yaml",
                          manifest_rel=f"configs/capeval/manifests/petri_full22_{MODEL_KEY}_{view}.yaml",
                          bank_root_rel=lp.GRADING_HOST_LANE_BANK_REL, bank_view_rel=lp.grading_host_bank_view_rel(view))
    if view == "arditi":
        return ViewParams(view=view, seed_bases=ARDITI_SEED_BASES, width=ARDITI_WIDTH, expected_cells=EXPECTED_ARDITI_CELLS,
                          cfg_rel=ARDITI_CFG_REL, manifest_rel=ARDITI_MANIFEST_REL,
                          bank_root_rel=lp.GRADING_HOST_ARDITI_LADDER_REL, bank_view_rel=f"{lp.GRADING_HOST_ARDITI_LADDER_REL}/cells_view/{lp.ARDITI_VIEW}")
    raise DriveRefused(f"--view {view!r} must be one of {VIEWS}")


def resolve_grading_host_bank_view(vp: ViewParams, home: Path) -> Path:
    """The grading host bank view directory a seed-view proxy's legs read (``TT_TWO_SI_DEFENSES_DIR``): ``<home>/<vp.bank_view_rel>`` under
    the LANE bank root. Refuses BY NAME when the view is absent -- naming the view, the root it was looked for under, and the operator
    step that builds it -- never a fallback to another bank (F-R: the old shared-bank path refused every battery view)."""
    root = Path(home) / vp.bank_root_rel
    p = Path(home) / vp.bank_view_rel
    if not p.is_dir():
        raise DriveRefused(f"no grading host bank view {vp.view!r} at {p} -- the lane bank {root} (lane_common.LANE_BANK; "
                           f"00_build_views.sh builds cells_view/{vp.view} there) has no such view")
    return p


def run_tag(view: str, seed_base: int) -> str:
    """The out-root tag of one (view, seed base): ``qwen35_4b_s<s>`` for a view engine (s is its own seed),
    ``qwen35_4b_arditi_s<s>`` for the Arditi engine's three seed-runs."""
    if view in SEED_VIEWS:
        if seed_base != int(view[1:]):
            raise ValueError(f"view {view} runs at its own seed base {view[1:]}, not {seed_base}")
        return f"{MODEL_KEY}_{view}"
    if view == "arditi":
        if seed_base not in ARDITI_SEED_BASES:
            raise ValueError(f"arditi seed base {seed_base} not in {ARDITI_SEED_BASES}")
        return f"{MODEL_KEY}_arditi_s{seed_base}"
    raise ValueError(f"unknown view {view!r}")


def _check_w7_wu8(w7: str, wave_uuid8_: str, where: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{7}", w7) or not re.fullmatch(r"[0-9a-f]{8}", wave_uuid8_):
        raise ValueError(f"{where} needs W7 (7 hex) and wave_uuid8 (8 hex), got {w7!r} {wave_uuid8_!r}")


def out_root_for(home: Path, w7: str, wave_uuid8_: str, attempt_id: str, tag: str) -> Path:
    """``$STATE/petri_native/petri_seedrep_<W-sha7>_<wave_uuid8>_<attempt_id>_<tag>`` -- the Petri store root of
    ONE (wave, proxy attempt, view/seed): a ``petri_seedrep_<W-sha7>_<tag>`` root widened by the wave and attempt
    axes, so a new wave -- or a re-executed proxy attempt -- at the same W starts EMPTY and can never
    adopt another wave's stores; ``wave_uuid8`` is the same 8-hex prefix the wave's Dict names carry, ``attempt_id``
    the tunnel record's. The sidecar producer reads the roots ``modal_drive.json`` / ``drive_<id>.json`` name."""
    _check_w7_wu8(w7, wave_uuid8_, "out_root_for")
    check_attempt_id(attempt_id)
    return Path(home) / ".local" / "state" / "petri_native" / f"petri_{LANE}_{w7}_{wave_uuid8_}_{attempt_id}_{tag}"


def state_dir_for(home: Path, w7: str, wave_uuid8_: str, view: str, petri_pass: str) -> Path:
    """The WAVE-scoped drive state ``$STATE/seedrep_battery/petri/<W-sha7>/<wave_uuid8>/<view>[_rerun]`` -- the
    drive log and the tunnel wait live here, before the proxy attempt is known; everything bound to the attempt
    goes under :func:`attempt_state_dir`."""
    _check_w7_wu8(w7, wave_uuid8_, "state_dir_for")
    suffix = {"": "", "overflow-rerun": "_rerun"}[petri_pass]
    return Path(home) / ".local" / "state" / f"{LANE}_battery" / LEG / w7 / wave_uuid8_ / f"{view}{suffix}"


def attempt_state_dir(state_dir: Path, attempt_id: str) -> Path:
    """``<wave state dir>/attempts/<attempt_id>``: the records of driving ONE proxy attempt (endpoint, tunnel,
    routing table, cells_served, models, per-leg logs + return records + locks, ``drive_<id>.json``,
    ``RESUMED_FROM.json``)."""
    return Path(state_dir) / "attempts" / check_attempt_id(attempt_id)


def resolve_width(vp: ViewParams, requested: Optional[int]) -> int:
    """``--width`` for one view's drive: ``None`` -> the view's planned width; a value must be >= 1 and AT MOST the
    view's planned width (the 84-of-100 ceiling is budgeted per view from the PLANNED widths, and a
    single drive cannot see the other three drives' overrides, so a wider width here could only be checked against
    a global total this process does not know -- widening is refused by name; narrowing is the throttle). A ``0``
    is refused rather than read as "default"."""
    if requested is None:
        return vp.width
    w = int(requested)
    if w < 1:
        raise DriveRefused(f"--width {requested} must be >= 1")
    if w > vp.width:
        raise DriveRefused(f"--width {w} exceeds view {vp.view}'s planned width {vp.width} ({vp.expected_runs} runs x {vp.width} = "
                           f"{vp.expected_runs * vp.width} of the {AUDITOR_CEILING}-conversation highprio ceiling; {vp.expected_runs} x {w} = "
                           f"{vp.expected_runs * w}) -- --width only narrows; widening a view needs the plan's constants changed, not a flag")
    return w


def total_auditor_width(views: Iterable[str] = VIEWS, widths: Optional[Mapping[str, int]] = None) -> int:
    """Concurrent auditor conversations across the drives: sum over proxies of runs x EFFECTIVE width -- at the planned
    widths 3 x 8 x 3 + 3 x 4 = 84. ``widths`` overrides a view's width (the drive passes its own ``--width``); a view
    name not in ``views`` is refused rather than silently ignored."""
    widths = dict(widths or {})
    unknown = sorted(set(widths) - set(views))
    if unknown:
        raise ValueError(f"widths for views not being summed: {unknown}")
    return sum(vp.expected_runs * int(widths.get(vp.view, vp.width)) for vp in map(view_params, views))


def assert_within_auditor_ceiling(views: Iterable[str] = VIEWS, ceiling: int = AUDITOR_CEILING,
                                  widths: Optional[Mapping[str, int]] = None) -> int:
    views = tuple(views)
    total = total_auditor_width(views, widths)
    if total > ceiling:
        eff = {v: int((widths or {}).get(v, view_params(v).width)) for v in views}
        raise DriveRefused(f"{total} concurrent auditor conversations exceed the highprio ceiling {ceiling} (effective widths {eff})")
    return total


# --------------------------------------------------------------------------- the schedule (pure, tested)

@dataclasses.dataclass(frozen=True)
class LegRun:
    view: str
    cell_index: int
    cell_id: str
    vmodel: str
    run_key: str
    seed_base: int
    width: int
    tag: str
    petri_pass: str = ""

    @property
    def record_name(self) -> str:
        return f"leg_{self.cell_index}_s{self.seed_base}"


def leg_rows(cells_json: Mapping[str, Any]) -> List[dict]:
    """The enumeration's served legs in ``petri_run``'s ``--cell-index`` order: canonical rows, then controls."""
    return list(cells_json.get("canonical") or []) + list(cells_json.get("controls") or [])


OBJECT_CELL_CLASSES = ("class_a", "class_b", "stacked")


def lane_arm_indices(cells_json: Mapping[str, Any], vp: ViewParams) -> Tuple[List[int], List[Tuple[int, str, str]]]:
    """SELECT the lane's arms from the COMPLETE native enumeration by the rows' OWN fields, keeping every row's
    ``--cell-index`` -> ``(arm indices, excluded rows as (index, cell_id, why))``.

    ``petri_run.enumerate_petri_native`` lists the manifest's cells PLUS the derived path-matched no-op control
    ``none__lora0`` (``cell_class == "control"``, ``control_of == "none"``; ``defense_cells.enumerate_cells(...,
    lora_controls=True)``), so a complete seed view enumerates 9 rows and the Arditi manifest (``none`` +
    ``arditi_allpos``) 3 -- counting served rows blind refused every real proxy. A view proxy's arms are every
    NON-control row (5 fitted + ``none`` + the 2 prompted routes; the control is excluded by its class, never by
    name); the Arditi proxy's arm is ``arditi_allpos`` alone -- ``none`` and the control are excluded, and any
    OTHER object cell on that standup is REFUSED (a multi-object Arditi standup skips the arm by
    name). A row without ``cell_id`` / ``cell_class`` is refused: a derived control cannot be told from an arm
    without them (the enumeration always writes both)."""
    kept: List[int] = []
    excluded: List[Tuple[int, str, str]] = []
    for idx, row in enumerate(leg_rows(cells_json)):
        cid, cls = str(row.get("cell_id") or ""), str(row.get("cell_class") or "")
        if not cid or not cls:
            raise DriveRefused(f"cells.json leg {idx} names no cell_id/cell_class -- cannot tell a derived control from a lane arm")
        if vp.is_arditi:
            if cid == ARDITI_ARM:
                kept.append(idx)
            elif cls in OBJECT_CELL_CLASSES:
                raise DriveRefused(f"view arditi: the enumeration carries a SECOND object cell {cid!r} ({cls}) beside {ARDITI_ARM} "
                                   "-- a multi-object Arditi standup skips the arm by name (plan §2.3, D2); refusing")
            else:
                excluded.append((idx, cid, f"not the Arditi arm ({cls})"))
        elif cls == "control":
            excluded.append((idx, cid, f"derived no-op control of {row.get('control_of')!r} -- not one of the lane's {vp.expected_cells} arm-runs"))
        else:
            kept.append(idx)
    return kept, excluded


def served_cells(cells_json: Mapping[str, Any], served_models: Sequence[str],
                 indices: Optional[Sequence[int]] = None) -> Tuple[List[int], List[Tuple[int, str]]]:
    """The cluster driver's rule over ``indices`` (default: every row): a leg whose vmodel the endpoint does not
    serve is SKIPPED (logged, never a 404 read as a failed audit); a leg with NO vmodel is refused under the
    native engine."""
    rows = leg_rows(cells_json)
    kept, skipped = [], []
    for idx in (range(len(rows)) if indices is None else indices):
        vm = rows[idx].get("virtual_model_name", "")
        if not vm:
            raise DriveRefused(f"cells.json leg {idx} names no virtual_model_name -- cannot hold it to served_models")
        if served_models and vm not in served_models:
            skipped.append((idx, vm))
        else:
            kept.append(idx)
    return kept, skipped


def petri_schedule(cells_json: Mapping[str, Any], served_models: Sequence[str], vp: ViewParams, *,
                   petri_pass: str = "", requested: Optional[Sequence[int]] = None) -> List[LegRun]:
    """The per-proxy schedule: every SERVED lane arm (:func:`lane_arm_indices`) x every seed base of the view,
    EXACTLY ``vp.expected_runs`` of them (8 for a view engine; 3 for Arditi), each run carrying the arm's
    enumeration index as its ``--cell-index``. Fewer served arms than expected is REFUSED with the count named
    -- a view proxy without its two prompted routes must not be audited as the view;
    more is refused too (an extra route is not this lane's arm set). The derived control counts neither way."""
    if cells_json.get("serving_path") != "lora_native":
        raise DriveRefused(f"cells.json was frozen under serving_path={cells_json.get('serving_path')!r}, not lora_native")
    rows = leg_rows(cells_json)
    arms, excluded = lane_arm_indices(cells_json, vp)
    kept, _ = served_cells(cells_json, served_models, arms)
    if len(kept) != vp.expected_cells:
        raise DriveRefused(
            f"view {vp.view}: the enumeration serves {len(kept)} lane arms, the lane expects {vp.expected_cells} "
            f"({'5 fitted + none + 2 prompted routes' if not vp.is_arditi else 'the one Arditi cell'}); "
            f"served arm vmodels: {[rows[i].get('virtual_model_name') for i in kept]}; "
            f"excluded rows: {[(i, c) for i, c, _w in excluded]}")
    if requested is not None:
        bad = [i for i in requested if i not in kept]
        if bad:
            raise DriveRefused(f"requested legs {bad} are not served lane arms of this proxy (served arms: {kept})")
        wanted = set(requested)
        kept = [i for i in kept if i in wanted]
    runs = []
    for idx in kept:
        row = rows[idx]
        for s in vp.seed_bases:
            runs.append(LegRun(view=vp.view, cell_index=idx, cell_id=str(row.get("cell_id", "")),
                               vmodel=str(row["virtual_model_name"]), run_key=str(row["petri_run_key"]),
                               seed_base=int(s), width=vp.width, tag=run_tag(vp.view, int(s)), petri_pass=petri_pass))
    if requested is None and len(runs) != vp.expected_runs:
        raise DriveRefused(f"view {vp.view}: {len(runs)} runs scheduled, expected {vp.expected_runs}")
    return runs


def leg_argv(py: str, petri_opts: Sequence[str], run: LegRun, out_root: Path) -> List[str]:
    """The exact ``petri_run.py`` argv for one leg: the top-level opts, then ``run`` with the out
    root, the cell index and the target-decode flags; ``--overflow-rerun`` on the re-run pass only."""
    argv = [py, "scripts/petri_run.py", *petri_opts, "run", "--out-root", str(out_root), "--cell-index", str(run.cell_index),
            "--target-decode", TARGET_DECODE, "--target-seed-base", str(run.seed_base)]
    if run.petri_pass == "overflow-rerun":
        argv.append("--overflow-rerun")
    return argv


def assert_lane_leg_argv(argv: Sequence[str], run: LegRun) -> None:
    """The lane profile every leg MUST carry (the runner contract): a ``run`` subcommand with
    ``--target-decode vendor`` (asserts the manifest's ``target_decode`` block -- a leg without it would audit
    at HARNESS decode under the lane's name) and ``--target-seed-base <the run's literal base>`` (a run knob
    outside the manifest hash, so each Arditi seed-run's base must be spelled on its own argv), after ``run``.
    Refused by name; never repaired."""
    a = list(argv)
    if "run" not in a:
        raise DriveRefused(f"{run.record_name}: leg argv has no `run` subcommand: {a}")
    tail = a[a.index("run") + 1:]

    def value_of(flag: str) -> Optional[str]:
        return tail[tail.index(flag) + 1] if flag in tail and tail.index(flag) + 1 < len(tail) else None

    if value_of("--target-decode") != TARGET_DECODE:
        raise DriveRefused(f"{run.record_name}: leg argv does not carry `--target-decode {TARGET_DECODE}` after `run` "
                           f"(got {value_of('--target-decode')!r}) -- the lane audits at the vendor row only; refusing to launch")
    if value_of("--target-seed-base") != str(run.seed_base):
        raise DriveRefused(f"{run.record_name}: leg argv carries `--target-seed-base {value_of('--target-seed-base')!r}`, "
                           f"this run's base is {run.seed_base}; refusing to launch")
    if value_of("--out-root") is None:
        raise DriveRefused(f"{run.record_name}: leg argv names no --out-root")


def width_after_attempt(width: int, log_text: str, *, threshold: int = THROTTLE_429_THRESHOLD) -> int:
    """Throttle DOWN on a 429-heavy attempt: one connection narrower (floor 1); a clean log keeps the width.
    Never a pool hop -- the org is fixed."""
    if width < 1:
        raise ValueError(f"width {width} < 1")
    hits = len(_429_RE.findall(log_text or ""))
    return max(1, width - 1) if hits >= threshold else width


def leg_attempts(run: LegRun, *, launch: Callable[[int], Tuple[Any, str]], proxy_up: Callable[[], bool],
                 sleep: Callable[[float], None] = time.sleep, note: Callable[[str], None] = print,
                 log: Callable[[str], None] = print, attempts: int = LEG_ATTEMPTS) -> Tuple[Any, List[dict]]:
    """One leg's attempt loop, the SAME on every pass (the re-run pass used to break on any rc after ONE
    attempt, so a transient transport / auditor / proxy failure there stopped the leg cold and skipped the throttle):
    up to ``attempts`` attempts; ``launch(width)`` runs the leg and returns ``(rc, that attempt's log text)``; the loop
    ENDS on rc 0 (complete) or 3 (incomplete on the overflow ledger) ONLY -- every other rc goes round again, the
    width first throttled DOWN by :func:`width_after_attempt` on a 429-heavy attempt (never a pool hop); an attempt
    finding the proxy unreachable sleeps and retries without launching. Returns ``(the last rc -- "unreachable" when
    no attempt launched, one event per attempt)``."""
    rc: Any = "unreachable"
    w = run.width
    events: List[dict] = []
    for attempt in range(1, int(attempts) + 1):
        note(f"== {run.record_name} attempt {attempt} width={w} {utc_now()} ==")
        if not proxy_up():
            note("proxy unreachable")
            events.append({"attempt": attempt, "width": w, "rc": None, "proxy_up": False})
            sleep(PROXY_UNREACHABLE_SLEEP_S)
            continue
        rc, attempt_text = launch(w)
        note(f"== {run.record_name} attempt {attempt} rc={rc} ==")
        events.append({"attempt": attempt, "width": w, "rc": rc, "proxy_up": True})
        if rc in (0, 3):
            break
        w_next = width_after_attempt(w, attempt_text)
        if w_next != w:
            log(f"{run.record_name}: 429-heavy attempt {attempt}; width {w} -> {w_next} (throttle down, never hop)")
            w = w_next
        sleep(ATTEMPT_SLEEP_S)
    return rc, events


def release_decision(verdict: Mapping[str, Any], *, teardown_after_failure: bool = False) -> Tuple[bool, str]:
    """Whether the drive writes STOP (releases the proxy) -> ``(write STOP, why)``. ``release``: every leg ended and
    none FAILED -- the cluster driver's release rule, on EVERY pass (the re-run pass used to STOP on any
    outcome). ``held``: a FAILED leg holds the proxy up so a re-invocation can resume it. ``teardown-after-failure``:
    STOP despite FAILED legs, ONLY under the explicit, named ``--teardown-after-failure`` -- the orchestrator's
    deliberate call, never the default."""
    if verdict["release"]:
        return True, "release"
    if teardown_after_failure:
        return True, "teardown-after-failure"
    return False, "held"


# --------------------------------------------------------------------------- the auditor key (pure given a resolver; tested)

_KEY_SHAPED = re.compile(r"^sk-|^[A-Za-z0-9_\-]{32,}$")


def auditor_env(org: str = AUDITOR_ORG, *, key_resolver: Optional[Callable[[str], str]] = None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """The leg's Anthropic environment for the auditor + judge, and the RECORD of which pool it uses.

    ``org`` must be ``highprio`` (a ``lowprio`` request is REFUSED here, not re-routed);
    an org that looks like a key is refused before anything is resolved. The key comes from the vault
    mechanism only -- ``key_resolver(org)`` defaults to :func:`sisteer.vault.anthropic_api_key`, which reads
    the org's OWN env var else queries the secret-manager CLI and raises ``PoolUnavailable`` otherwise -- and is
    placed in the env under ``ANTHROPIC_API_KEY`` with the base URL PINNED. The record carries the pool name and
    the key's sha256[:12], never the key."""
    if not isinstance(org, str) or not org:
        raise DriveRefused("auditor org must be a non-empty string")
    if _KEY_SHAPED.match(org):
        raise DriveRefused("the auditor org is a POOL NAME (highprio), never a key or a key-shaped literal")
    if org != AUDITOR_ORG:
        raise DriveRefused(f"the Petri auditor + judge run on {AUDITOR_ORG} only (spec fixed decision 9); "
                           f"refusing auditor org {org!r} -- throttle the width, never hop pools")
    if key_resolver is None:
        from sisteer.vault import anthropic_api_key as key_resolver  # type: ignore[assignment]
    from sisteer.vault import ANTHROPIC_BASE_URL, pool_stamp

    key = key_resolver(org)
    if not isinstance(key, str) or not key.strip():
        raise DriveRefused(f"the {org} key resolver returned nothing")
    key = key.strip()
    env = {
        "ANTHROPIC_API_KEY": key,
        "ANTHROPIC_BASE_URL": ANTHROPIC_BASE_URL,
        "SISTEER_AUDITOR_POOL": org,
        "F22_AUDITOR_ORG": org,
    }
    return env, pool_stamp(org, key)


REMOVED_QWEN_LEG_MSG = ("the Petri auditor chain is API-only (sonnet -> grok; D111, ReLU 3931984bf): the abliterated-Qwen third leg "
                        "is retired and no pool is stood up for it -- drop --qwen-auditor-pool / --qwen-auditor-url / "
                        "--allow-unreachable-qwen-auditor")


def qwen_auditor_env(pool_record: Mapping[str, Any], bearer: str) -> Dict[str, str]:
    """The chain's THIRD auditor leg (the abliterated Qwen3.6-27B) served by a seedrep attacker pool: the
    pool's URL + the lane bearer under the names ``auditor_fallback`` reads (``PETRI_QWEN_AUDITOR_BASE_URL``,
    ``PETRI_QWEN_AUDITOR_API_KEY``). The record must be a seedrep pool serving the attacker model."""
    if pool_record.get("lane") != LANE:
        raise DriveRefused(f"the Qwen-auditor pool record is not a {LANE} pool (lane={pool_record.get('lane')!r})")
    model = str(pool_record.get("model") or "")
    if not model.endswith("Huihui-Qwen3.6-27B-abliterated"):
        raise DriveRefused(f"the Qwen-auditor pool serves {model!r}, not the abliterated Qwen3.6-27B")
    url = str(pool_record.get("url") or "").rstrip("/")
    if not url.startswith("https://"):
        raise DriveRefused(f"the Qwen-auditor pool url {url!r} is not https")
    if not bearer:
        raise DriveRefused("the Qwen-auditor pool is bearer-guarded and no bearer was given")
    return {QWEN_BASE_URL_ENV: url + "/v1", QWEN_API_KEY_ENV: bearer}


def read_private_key_file(path: Path) -> str:
    """The pool's self-minted bearer, read ONLY from under ``$XDG_RUNTIME_DIR`` (the pool renders it in the tmpfs runtime dir
    and nowhere else); a path anywhere else -- the earlier ``$STATE`` location included -- is refused unread, as is a
    runtime dir that is unset, a file that is not 0600-private, or one that is empty."""
    rd = os.environ.get("XDG_RUNTIME_DIR")
    if not rd:
        raise DriveRefused(f"refusing bearer file {path}: XDG_RUNTIME_DIR is unset, so no tmpfs runtime dir can hold it (D77)")
    try:
        Path(path).resolve().relative_to(Path(rd).resolve())
    except ValueError:
        raise DriveRefused(f"refusing bearer file {path}: not under the tmpfs runtime dir {rd} (D77: the pool renders its bearer only there; "
                           f"a pre-D77 file is not read -- re-run the pool's --action endpoint with SEEDREP_ATTACKER_ROTATE_KEY=1)") from None
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise DriveRefused(f"refusing bearer file {path}: mode {oct(mode)} is not private")
    v = path.read_text().strip()
    if not v:
        raise DriveRefused(f"refusing empty bearer file {path}")
    return v


# --------------------------------------------------------------------------- records, verdict, resume (pure, tested)

def leg_status_line(rc, drive_id: str, results_path: Path) -> str:
    sha = hashlib.sha256(results_path.read_bytes()).hexdigest() if results_path.is_file() else "absent"
    return f"{rc} drive={drive_id} results={sha}"


def parse_leg_status(line: str) -> Tuple[str, str, str]:
    parts = line.strip().split()
    rc = parts[0] if parts else ""
    drive = next((p[len("drive="):] for p in parts[1:] if p.startswith("drive=")), "")
    results = next((p[len("results="):] for p in parts[1:] if p.startswith("results=")), "")
    return rc, drive, results


def unresolved_count(cell_dir: Path) -> int:
    try:
        d = json.loads((cell_dir / "petri_incomplete.json").read_text())
        u = d.get("unresolved")
        return len(u) if isinstance(u, list) else 0
    except (OSError, ValueError):
        return 0


ReconcileFn = Callable[[LegRun, Path], Tuple[bool, str]]


def completion_verdict(runs: Sequence[LegRun], out_roots: Mapping[str, Path], state_dir: Path, reconcile: ReconcileFn) -> dict:
    """The cluster driver's accounting per (cell, seed base): FAILED (last recorded return neither 0 nor 3),
    UNVERIFIED (0 over results bytes that changed since), UNRECONCILED (a results.json the Petri store does not
    bear out -- FAILED, held, never skipped), COMPLETE (results.json, the record agrees AND ``reconcile(run,
    cell_dir)`` -- the Petri driver's OWN store reconciliation, :func:`reconcile_cell_store` -> ``petri_run.
    _already_done`` -- bears the marker out), INCOMPLETE (an unresolved overflow ledger), else FAILED.
    A results file's existence is never the verdict. Release (STOP) only when NFAIL == 0."""
    n = {"complete": 0, "incomplete": 0, "failed": 0}
    detail = []
    for r in runs:
        rec_p = state_dir / f"{r.record_name}.rc"
        lrc = ldrive = lres = ""
        if rec_p.is_file():
            lrc, ldrive, lres = parse_leg_status(rec_p.read_text())
        cell_dir = Path(out_roots[r.tag]) / "cells" / r.run_key
        results = cell_dir / "results.json"
        if lrc and lrc not in ("0", "3"):
            n["failed"] += 1
            detail.append((r.record_name, r.run_key, f"failed_rc_{lrc}_drive_{ldrive or 'unrecorded'}"))
            continue
        if results.is_file():
            if lrc == "0" and lres and lres != "absent" and hashlib.sha256(results.read_bytes()).hexdigest() != lres:
                n["failed"] += 1
                detail.append((r.record_name, r.run_key, "unverified_results_changed_since_return"))
                continue
            ok, why = reconcile(r, cell_dir)
            if not ok:
                n["failed"] += 1
                detail.append((r.record_name, r.run_key, f"unreconciled: {why}"))
                continue
            n["complete"] += 1
            detail.append((r.record_name, r.run_key, "complete" + ("" if lrc else "_by_cell_dir_alone")))
        elif unresolved_count(cell_dir) > 0:
            n["incomplete"] += 1
            detail.append((r.record_name, r.run_key, "incomplete_on_overflow_ledger"))
        else:
            n["failed"] += 1
            detail.append((r.record_name, r.run_key, "failed_no_results"))
    return {"counts": n, "nruns": len(runs), "release": n["failed"] == 0, "detail": detail}


def legs_to_run(runs: Sequence[LegRun], verdict: Mapping[str, Any]) -> List[LegRun]:
    """IDEMPOTENT RESUME: the legs a (re-)invocation drives = every run whose verdict is not COMPLETE. A
    complete leg is never re-driven (its results bytes are the record); an incomplete, failed or unverified
    one is (petri_run resumes by run key inside the leg)."""
    complete = {name for name, _rk, why in verdict["detail"] if why.startswith("complete")}
    return [r for r in runs if r.record_name not in complete]


#: read the manifest's ``bloom_options.epochs`` through the Petri driver's own loader (the value the reconciliation forms item ids with)
EPOCHS_SNIPPET = r"""
import sys
from pathlib import Path
wt, manifest = sys.argv[1], sys.argv[2]
sys.path.insert(0, str(Path(wt) / "src"))
from sisteer.capeval import manifest as cm
print(cm.load_petri_spec(Path(wt) / manifest).bloom_options.epochs)
"""

#: the Petri driver's OWN done check, verbatim: ``petri_run._already_done`` over the leg's CellStore -- the routine
#: ``petri_run.py run`` itself skips a complete cell with (and refuses a marker the store does not bear out with)
RECONCILE_SNIPPET = r"""
import json, sys
from pathlib import Path
from types import SimpleNamespace
wt, cell_dir, cell_id, run_key, epochs = sys.argv[1:6]
sys.path.insert(0, str(Path(wt) / "scripts"))
sys.path.insert(0, str(Path(wt) / "src"))
import petri_run
from sisteer.capeval import store as cs
p = Path(cell_dir)
try:
    if p.name != run_key:
        raise petri_run.DriverError(f"{p}: directory {p.name!r} is not the run key {run_key!r}")
    store = cs.CellStore(p.parent.parent, run_key)
    done = petri_run._already_done(store, SimpleNamespace(cell_id=cell_id, petri_run_key=run_key), int(epochs))
    out = {"complete": bool(done), "reason": "" if done else "no results.json"}
except Exception as e:
    out = {"complete": False, "reason": f"{type(e).__name__}: {e}"}
sys.stdout.write("\n" + json.dumps(out) + "\n")
"""


def manifest_epochs(py: str, wt: Path, manifest_rel: str) -> int:
    """``bloom_options.epochs`` of the leg's manifest, through the LEG python and the driver's own loader
    (``manifest.load_petri_spec``) -- never a yaml read of our own."""
    r = subprocess.run([py, "-c", EPOCHS_SNIPPET, str(wt), manifest_rel], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise DriveRefused(f"cannot load {manifest_rel} through {py}: {(r.stdout + r.stderr)[-800:]}")
    return int(r.stdout.strip().splitlines()[-1])


def reconcile_cell_store(py: str, wt: Path, cell_dir: Path, cell_id: str, run_key: str, epochs: int) -> Tuple[bool, str]:
    """The AUTHORITATIVE done check for one leg: ``petri_run._already_done`` -- the very routine
    ``petri_run.py run`` skips a complete cell with -- run under the LEG python (the driver's imports live there):
    the ``cells/<run_key>/`` dir bound to its manifest (this cell id + run key; directory = run key), a results.json
    naming the same cell and run, the failure ledger settled and identity-bearing, the footing recomputed from
    rows.jsonl + the refusal sidecars. ``(True, "")`` when the store bears the marker out; ``(False, reason)`` for
    no marker, or a marker the store does not bear out (the reason is the driver's own sentence). Never a file's
    existence."""
    r = subprocess.run([py, "-c", RECONCILE_SNIPPET, str(wt), str(cell_dir), cell_id, run_key, str(int(epochs))],
                       capture_output=True, text=True, timeout=600)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    try:
        verdict = json.loads(lines[-1])
        return bool(verdict["complete"]), str(verdict.get("reason") or "")
    except (IndexError, ValueError, KeyError, TypeError):
        return False, f"reconciliation did not run (rc {r.returncode}): {(r.stdout + r.stderr)[-500:]}"


def newest_drive_record(state_dir: Path) -> Tuple[Path, dict]:
    """The newest ``drive_<id>.json`` (by ``started_at``, then name) under an attempt state dir."""
    recs = sorted(Path(state_dir).glob("drive_*.json"))
    if not recs:
        raise DriveRefused(f"no drive_<id>.json under {state_dir} -- not a drive's attempt state dir")

    def started(p: Path):
        try:
            return str(json.loads(p.read_text()).get("started_at") or ""), p.name
        except ValueError:
            return "", p.name
    p = max(recs, key=started)
    return p, json.loads(p.read_text())


def import_staging_dir(root: Path, drive_id: str) -> Path:
    """The hidden SIBLING an out root is staged in while an import is in flight -- ``.<root name>.import-<drive_id>``
    beside the root, so a ``petri_seedrep_<W7>_*`` glob never lists a half-built store."""
    root = Path(root)
    return root.parent / f".{root.name}.import-{drive_id}"


IMPORT_TXN_SCHEMA = "seedrep_petri_drive_import_txn/1"


def import_txn_marker(out_roots: Mapping[str, Path], w7: str, wave_uuid8_: str, attempt_id: str) -> Path:
    """The durable TRANSACTION MARKER of an import into this attempt's roots: a hidden file beside the roots,
    named by the ATTEMPT (stable across drive ids), on disk exactly while per-root renames are in flight -- written
    before the first ``publish``, removed after the last -- so a restart or SIGTERM between renames leaves a record
    the next invocation recovers from (:func:`recover_import_transaction`, which main runs before any import)."""
    parents = {Path(r).parent for r in out_roots.values()}
    if len(parents) != 1:
        raise ValueError(f"the out roots do not share one parent: {sorted(map(str, parents))}")
    _check_w7_wu8(w7, wave_uuid8_, "import_txn_marker")
    return parents.pop() / f".petri_{LANE}_{w7}_{wave_uuid8_}_{check_attempt_id(attempt_id)}.import-txn.json"


def write_import_txn_marker(marker: Path, txn: Mapping[str, Any]) -> None:
    """Durably: the bytes to a sibling temp file, fsync, then one rename onto the marker's name."""
    tmp = marker.with_name(marker.name + ".tmp")
    with tmp.open("w") as f:
        f.write(json.dumps(txn, indent=1))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, marker)


def recover_import_transaction(out_roots: Mapping[str, Path], *, w7: str, wave_uuid: str, attempt_id: str,
                               unpublish: Callable[[Path, Path], Any] = os.rename) -> Optional[dict]:
    """Run BEFORE any import or any use of the roots (main does): when an import transaction marker for this attempt
    exists, an earlier invocation died between per-root renames -- or after the last rename, before the marker went.
    Per root the state is one of published (its destination exists), staged (its sibling exists) or absent; rename is
    atomic, so BOTH existing is not a state this import leaves and is refused by name. Every root published -> the
    transaction had COMPLETED: the marker is removed and the roots stand (each carries its ``RESUMED_FROM.json``).
    Otherwise the partial publication is ROLLED BACK: published roots renamed back to their staged siblings, every
    staged sibling removed, then the marker -- the operator re-runs the import if they still want it. Any failure
    leaves the marker in place and raises naming what is still present; nothing is swallowed. Returns ``None`` (no
    open transaction) or the recovery record."""
    marker = import_txn_marker(out_roots, w7, wave_uuid8(wave_uuid), attempt_id)
    if not marker.exists():
        return None
    try:
        txn = json.loads(marker.read_text())
    except ValueError as e:
        raise DriveRefused(f"{marker}: unreadable import transaction marker ({e}) -- inspect and resolve by hand")
    if txn.get("schema") != IMPORT_TXN_SCHEMA or txn.get("into") != {"W7": w7, "wave_uuid": wave_uuid, "attempt_id": attempt_id}:
        raise DriveRefused(f"{marker}: not this attempt's import transaction (schema {txn.get('schema')!r}, into {txn.get('into')!r}) -- refusing to act on it")
    roots = [(str(r["tag"]), Path(r["staged"]), Path(r["dst"])) for r in (txn.get("roots") or [])]
    if not roots:
        raise DriveRefused(f"{marker}: names no roots -- refusing to guess; resolve by hand")
    states: Dict[str, str] = {}
    for tag, staged, dst in roots:
        s, p = staged.exists(), dst.exists()
        if s and p:
            raise DriveRefused(f"{marker}: {dst} AND its staged sibling {staged} both exist -- not a state this import leaves; refusing to guess, resolve by hand")
        states[tag] = "published" if p else ("staged" if s else "absent")
    base = {"marker": str(marker), "drive_id": txn.get("drive_id"), "from_drive_record": (txn.get("from") or {}).get("drive_record"), "states": states}
    if all(v == "published" for v in states.values()):
        marker.unlink()
        return {"action": "completed", **base}
    problems: List[str] = []
    unpublished: List[str] = []
    removed: List[str] = []
    for tag, staged, dst in reversed(roots):
        if states[tag] == "published":
            try:
                unpublish(dst, staged)
                unpublished.append(str(dst))
            except OSError as e:
                problems.append(f"could not un-publish {dst} -> {staged}: {e}")
    for _tag, staged, _dst in roots:
        if staged.exists():
            try:
                shutil.rmtree(staged)
                removed.append(str(staged))
            except OSError as e:
                problems.append(f"could not remove staged {staged}: {e}")
    residual = [str(dst) for _t, _s, dst in roots if dst.exists()] + [str(st) for _t, st, _d in roots if st.exists()]
    if problems or residual:
        raise DriveRefused(f"{marker}: recovery did NOT complete -- {'; '.join(problems) or 'no error raised'}; still present: {residual}; "
                           "the marker stays -- resolve by hand")
    marker.unlink()
    return {"action": "rolled_back", "unpublished": unpublished, "removed_staged": removed, **base}


def import_prior_stores(prior_state_dir: Path, out_roots: Mapping[str, Path], *, w7: str, wave_uuid: str, attempt_id: str,
                        view: str, drive_id: str, copytree: Callable[[Path, Path], Any] = shutil.copytree,
                        publish: Callable[[Path, Path], Any] = os.rename, unpublish: Callable[[Path, Path], Any] = os.rename) -> dict:
    """``--resume-from``: continue a PRIOR wave/attempt's Petri stores in THIS attempt's roots -- explicitly, with
    provenance, and ATOMICALLY per root: a per-tag validate-then-copy left an earlier tag's ``cells/`` imported
    with no ``RESUMED_FROM.json`` when a later tag was missing, occupied or failed. Reads the prior attempt state
    dir's newest ``drive_<id>.json`` (the same W and view required; itself refused), then in three phases:

    0. An OPEN transaction marker for these roots (:func:`import_txn_marker`) is refused before any other
       check -- a root it left published must point at the recovery, never read as merely occupied.
    1. PREFLIGHT, before ANY mutation: every tag has a recorded prior root that is a directory; every final
       destination root does not exist (an import lands only in a root nobody has written -- a re-invocation
       resumes by itself); no staged sibling of this drive's is lying about.
    2. STAGE: each prior root's ``cells/`` tree (the stores EXACTLY as the legs wrote them) is copied into the hidden
       sibling :func:`import_staging_dir` of its destination, and the ONE provenance record ``RESUMED_FROM.json`` is
       written into every staged root -- so provenance travels WITH the store.
    3. PUBLISH: the durable transaction marker (:func:`import_txn_marker`) is written FIRST, then each staged root is
       renamed onto its destination (``publish``, default ``os.rename`` -- one atomic step per root), then the marker
       is removed. Any failure at any point un-publishes the roots already published (renamed back, ``unpublish``) and
       removes every staged sibling, and then VERIFIES: nothing is swallowed -- a clean rollback is claimed only
       when no final destination and no staged sibling remains; otherwise an explicit error names what is still present
       and the marker stays for :func:`recover_import_transaction` on the next invocation.

    NOT copied: ``cells.json`` and ``modal_drive.json`` -- this attempt freezes its own enumeration against its own
    proxy's evidence, and ``petri_run`` judges every imported dir by its manifest + run key (the reconciliation
    above; a foreign dir refuses by name). Returns the record (the drive record carries it too)."""
    rec_p, prior = newest_drive_record(prior_state_dir)
    if prior.get("W7") != w7:
        raise DriveRefused(f"--resume-from {rec_p}: W7 {prior.get('W7')!r} != this drive's {w7!r} -- another W's stores never continue here")
    if prior.get("view") != view:
        raise DriveRefused(f"--resume-from {rec_p}: view {prior.get('view')!r} != {view!r}")
    prior_attempt = str(prior.get("attempt_id") or (prior.get("tunnel") or {}).get("attempt_id") or "")
    if (prior.get("wave_uuid"), prior_attempt) == (wave_uuid, attempt_id):
        raise DriveRefused(f"--resume-from {rec_p}: that is this very (wave, attempt) -- a re-invocation resumes by itself, without an import")
    prior_roots = dict(prior.get("out_roots") or {})
    # 0. an OPEN import transaction into these roots is refused FIRST: a root it left published would otherwise be refused below as
    # merely "already exists", hiding that the recovery -- not a fresh attempt -- is the remedy
    marker = import_txn_marker(out_roots, w7, wave_uuid8(wave_uuid), attempt_id)
    if marker.exists():
        raise DriveRefused(f"{marker} exists -- an import transaction into this attempt's roots is OPEN (an earlier invocation died between "
                           "renames); recover_import_transaction() rolls it back or completes it (main runs it before any import); refusing to stage over it")
    # 1. PREFLIGHT -- every source mapping and every final destination, before anything is written
    plan: List[Tuple[str, Path, Path, Path]] = []                      # (tag, prior root, destination root, staged sibling)
    for tag, root in out_roots.items():
        src = prior_roots.get(tag)
        if not src:
            raise DriveRefused(f"--resume-from {rec_p}: no out root recorded for {tag}")
        src_root, dst = Path(src), Path(root)
        if not src_root.is_dir():
            raise DriveRefused(f"--resume-from {rec_p}: the recorded out root for {tag}, {src_root}, is not a directory -- nothing to continue")
        if dst.exists():
            raise DriveRefused(f"{dst} already exists -- an import never writes over a store; use a fresh attempt or resume by itself")
        staged = import_staging_dir(dst, drive_id)
        if staged.exists():
            raise DriveRefused(f"{staged} already exists -- an earlier import into {dst} did not finish; inspect it and remove it by hand")
        plan.append((tag, src_root, dst, staged))
    copied: Dict[str, dict] = {}
    published: List[Tuple[Path, Path]] = []
    record: dict = {}

    def rollback() -> Tuple[List[str], List[str]]:
        """Un-publish every published root (renamed back to its staged sibling) and remove every staged sibling, swallowing
        NOTHING -> (the failures, what is still present: final destinations, then staged siblings)."""
        problems: List[str] = []
        for staged, dst in reversed(published):
            try:
                unpublish(dst, staged)
            except OSError as e:
                problems.append(f"could not un-publish {dst} -> {staged}: {e}")
        for _tag, _src, _dst, staged in plan:
            if staged.exists():
                try:
                    shutil.rmtree(staged)
                except OSError as e:
                    problems.append(f"could not remove staged {staged}: {e}")
        residual = [str(dst) for _t, _s, dst, _st in plan if dst.exists()] + [str(st) for _t, _s, _d, st in plan if st.exists()]
        return problems, residual

    def failed(e: BaseException) -> BaseException:
        """Roll back, VERIFY, and return what to raise: the original (a clean rollback under a KeyboardInterrupt / SystemExit), a
        named clean-rollback refusal, or the explicit residual error -- never a clean claim over a leftover root."""
        problems, residual = rollback()
        if not problems and not residual:
            if marker.exists():
                marker.unlink()                                                   # the transaction closed clean
            if isinstance(e, Exception):
                return DriveRefused(f"--resume-from {rec_p}: import failed ({type(e).__name__}: {e}); rolled back -- no root of this attempt "
                                    "was left behind (verified: no destination, no staged sibling)")
            return e
        where = (f"the transaction marker {marker} stays for recover_import_transaction() on the next invocation" if marker.exists()
                 else "no transaction marker had been written yet (the failure was in staging)")
        return DriveRefused(f"--resume-from {rec_p}: import failed ({type(e).__name__}: {e}) and the rollback did NOT complete -- "
                            f"{'; '.join(problems) or 'no error raised'}; still present: {residual}; {where} -- resolve before retrying")

    try:
        # 2. STAGE -- the stores, then the ONE provenance record into every staged root
        for tag, src_root, dst, staged in plan:
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.mkdir()
            src_cells = src_root / "cells"
            if src_cells.is_dir():
                copytree(src_cells, staged / "cells")
                dirs = sorted(p.name for p in (staged / "cells").iterdir() if p.is_dir())
            else:
                dirs = []
            copied[tag] = {"from": str(src_root), "to": str(dst), "cell_dirs": dirs}
        record = {
            "schema": RESUME_SCHEMA, "drive_id": drive_id, "imported_at": utc_now(),
            "into": {"W7": w7, "wave_uuid": wave_uuid, "attempt_id": attempt_id, "view": view},
            "from": {"state_dir": str(prior_state_dir), "drive_record": str(rec_p), "drive_id": prior.get("drive_id"), "W": prior.get("W"),
                     "W7": prior.get("W7"), "wave_uuid": prior.get("wave_uuid"), "attempt_id": prior_attempt, "out_roots": prior_roots},
            "copied": copied,
        }
        for _tag, _src, _dst, staged in plan:
            (staged / RESUMED_FROM_FILE).write_text(json.dumps(record, indent=1))
        # 3. PUBLISH -- the durable transaction marker FIRST (a restart between renames recovers from it), then one rename per root,
        # then the marker goes; a later failure un-publishes the earlier ones
        write_import_txn_marker(marker, {"schema": IMPORT_TXN_SCHEMA, "drive_id": drive_id, "written_at": utc_now(),
                                         "into": {"W7": w7, "wave_uuid": wave_uuid, "attempt_id": attempt_id},
                                         "from": {"drive_record": str(rec_p)},
                                         "roots": [{"tag": t, "staged": str(st), "dst": str(d)} for t, _s, d, st in plan]})
        for _tag, _src, dst, staged in plan:
            publish(staged, dst)
            published.append((staged, dst))
        marker.unlink()
    except BaseException as e:
        err = failed(e)
        if err is e:
            raise
        raise err from e
    return record



def _same_path(a: Any, b: Any) -> bool:
    def norm(p: Any) -> str:
        q = Path(os.path.expanduser(str(p)))
        return str(q.resolve()) if q.exists() else os.path.normpath(str(q))
    return norm(a) == norm(b)


def adopt_recovered_import(out_roots: Mapping[str, Path], *, requested: Optional[Path], recovery: Optional[Mapping[str, Any]], w7: str,
                           wave_uuid: str, attempt_id: str, view: str) -> dict:
    """After :func:`recover_import_transaction` returned ``completed`` (every root had been published; the earlier invocation died
    before removing the marker): the recovered roots ARE the import this drive was asked for (re-entering
    :func:`import_prior_stores` with the unchanged ``--resume-from`` argv refused the recovered roots as "already exists"). The ONE
    ``RESUMED_FROM.json`` every root carries is loaded, required IDENTICAL across the roots, bound to THIS attempt (``into``), to
    every out-root tag (``copied``) and to the marker's own source (``from.drive_record`` == the recovery's); when the current
    ``--resume-from`` names a DIFFERENT source than the recovered import came from, the drive refuses explicitly rather than
    silently continuing another import. Returns the record (it becomes this drive's ``resumed_from``)."""
    records: Dict[str, Any] = {}
    for tag, root in out_roots.items():
        rf = Path(root) / RESUMED_FROM_FILE
        if not rf.is_file():
            raise DriveRefused(f"recovered root {root} carries no {RESUMED_FROM_FILE} -- not an import this drive can adopt; resolve by hand")
        try:
            records[tag] = json.loads(rf.read_text())
        except ValueError as e:
            raise DriveRefused(f"{rf}: unreadable provenance ({e}) -- resolve by hand")
    distinct = {json.dumps(r, sort_keys=True) for r in records.values()}
    if len(distinct) != 1:
        raise DriveRefused(f"the recovered roots carry {len(distinct)} DIFFERENT {RESUMED_FROM_FILE} records ({sorted(records)}) -- not one import; "
                           "resolve by hand")
    rec = next(iter(records.values()))               # the roots' OWN record, key order as written -- the receipt re-serialises to the same bytes
    into = {"W7": w7, "wave_uuid": wave_uuid, "attempt_id": attempt_id, "view": view}
    if rec.get("schema") != RESUME_SCHEMA or rec.get("into") != into:
        raise DriveRefused(f"the recovered {RESUMED_FROM_FILE} is not this attempt's import (schema {rec.get('schema')!r}, into {rec.get('into')!r} "
                           f"!= {into}) -- refusing to adopt it")
    if set(rec.get("copied") or {}) != set(out_roots):
        raise DriveRefused(f"the recovered {RESUMED_FROM_FILE} covers tags {sorted(rec.get('copied') or {})}, this drive's roots are {sorted(out_roots)} "
                           "-- not the same import; resolve by hand")
    src = (rec.get("from") or {})
    if recovery and recovery.get("from_drive_record") and str(src.get("drive_record")) != str(recovery["from_drive_record"]):
        raise DriveRefused(f"the recovered roots' provenance names {src.get('drive_record')!r} but the transaction marker named "
                           f"{recovery['from_drive_record']!r} -- not the same import; resolve by hand")
    if requested is not None and not _same_path(requested, src.get("state_dir")):
        raise DriveRefused(f"--resume-from {requested} names a DIFFERENT source than the recovered import came from ({src.get('state_dir')}, drive record "
                           f"{src.get('drive_record')}) -- refusing; drop --resume-from (or name that source) to continue the recovered import, or "
                           "use a fresh attempt for the other one")
    return rec


def resolve_resume(out_roots: Mapping[str, Path], *, resume_from: Optional[Path], recovery: Optional[Mapping[str, Any]], w7: str, wave_uuid: str,
                   attempt_id: str, view: str, drive_id: str, log: Callable[[str], Any] = print) -> Tuple[Optional[dict], str]:
    """This drive's ``resumed_from`` -> ``(record or None, how)``, ``how`` in {adopted, imported, none}.

    The durable adoption predicate is the ROOTS' OWN PROVENANCE, not the transaction marker: ``RESUMED_FROM.json`` is written
    into every staged root BEFORE its rename, so it stands in a destination root iff the import published that root. When every root of
    this attempt already stands and any of them carries it, the import is ADOPTED from it (:func:`adopt_recovered_import`: all identical,
    bound to this attempt, the argv's source matching) -- whether the marker is still there (a COMPLETED recovery this invocation made),
    already gone (an earlier restart's recovery consumed it and then refused a wrong ``--resume-from`` or malformed provenance, or died
    before recording the adoption), or long gone (a plain re-invocation of an attempt whose import succeeded). Nothing in-memory carries
    between invocations; each derives the answer again from the marker, the roots and the argv. Else an explicit ``--resume-from``
    imports (:func:`import_prior_stores`); else nothing."""
    if recovery and recovery.get("action") == "completed":
        rec = adopt_recovered_import(out_roots, requested=resume_from, recovery=recovery, w7=w7, wave_uuid=wave_uuid, attempt_id=attempt_id, view=view)
        log(f"RESUME_ADOPTED: the recovered import (drive {rec.get('drive_id')}, from {rec['from']['drive_record']}) is this drive's --resume-from"
            + (f" (the argv named the same source, {resume_from})" if resume_from is not None else " (no --resume-from on the argv)"))
        return rec, "adopted"
    roots = [Path(r) for r in out_roots.values()]
    if roots and all(r.exists() for r in roots) and any((r / RESUMED_FROM_FILE).is_file() for r in roots):
        rec = adopt_recovered_import(out_roots, requested=resume_from, recovery=None, w7=w7, wave_uuid=wave_uuid, attempt_id=attempt_id, view=view)
        log(f"RESUME_ADOPTED: this attempt's roots already stand with their import's provenance (drive {rec.get('drive_id')}, from "
            f"{rec['from']['drive_record']}; no open transaction) -- adopted as this drive's --resume-from"
            + (f" (the argv named the same source, {resume_from})" if resume_from is not None else " (no --resume-from on the argv)"))
        return rec, "adopted"
    if resume_from is not None:
        return import_prior_stores(Path(resume_from), out_roots, w7=w7, wave_uuid=wave_uuid, attempt_id=attempt_id, view=view, drive_id=drive_id), "imported"
    return None, "none"


def recover_and_resolve_resume(out_roots: Mapping[str, Path], *, resume_from: Optional[Path], w7: str, wave_uuid: str, attempt_id: str, view: str,
                               drive_id: str, astate: Path, log: Callable[[str], Any] = print) -> Tuple[Optional[dict], Optional[dict], str]:
    """ONE invocation's resume step, exactly as ``main()`` runs it and as the tests replay it invocation by invocation: recover an
    OPEN import transaction (:func:`recover_import_transaction`), resolve the resume (:func:`resolve_resume`: adopt / import / none) and
    durably record the adoption receipt (``RESUMED_FROM.json`` in the attempt state dir). Returns ``(recovery, resumed_from, resume_how)``;
    nothing else survives to the next invocation."""
    recovery = recover_import_transaction(out_roots, w7=w7, wave_uuid=wave_uuid, attempt_id=attempt_id)
    if recovery:
        log(f"IMPORT_RECOVERY {recovery['action']}: marker {recovery['marker']} (drive {recovery['drive_id']}, from {recovery['from_drive_record']}); "
            f"root states {recovery['states']}"
            + (f"; un-published {recovery['unpublished']}, removed staged {recovery['removed_staged']} -- pass --resume-from again to import"
               if recovery["action"] == "rolled_back" else " -- the imported roots stand, each with its RESUMED_FROM.json; adopted as this drive's import"))
    resumed_from, resume_how = resolve_resume(out_roots, resume_from=resume_from, recovery=recovery, w7=w7, wave_uuid=wave_uuid, attempt_id=attempt_id,
                                              view=view, drive_id=drive_id, log=log)
    if resumed_from:
        astate.mkdir(parents=True, exist_ok=True)
        (astate / RESUMED_FROM_FILE).write_text(json.dumps(resumed_from, indent=1))
        log(f"RESUMED_FROM ({resume_how}) {resumed_from['from']['drive_record']} (wave {resumed_from['from']['wave_uuid']} attempt "
            f"{resumed_from['from']['attempt_id']}): " + ", ".join(f"{t}: {len(c['cell_dirs'])} cell dir(s)" for t, c in resumed_from["copied"].items()))
    return recovery, resumed_from, resume_how


# --------------------------------------------------------------------------- paths + transport helpers (pure where possible)

def volume_client_path(path: str) -> str:
    prefix = f"{VOL_PATH}/"
    if not path.startswith(prefix):
        raise ValueError(f"path is outside {VOL_PATH}: {path}")
    return path[len(VOL_PATH):]


def container_to_volume_path(p: str, container_exp_root: str, attempt_root_vol: str) -> str:
    """A path the container wrote into endpoint.json (under ITS exp root) -> the same file under the attempt's
    volume root the TreeSyncer ships that exp tree to. Anything outside the exp root is refused."""
    root = container_exp_root.rstrip("/") + "/"
    if not str(p).startswith(root):
        raise ValueError(f"{p} is not under the container's exp root {root}")
    return attempt_root_vol.rstrip("/") + "/" + str(p)[len(root):]


def check_attempt_id(attempt_id: Any) -> str:
    """The proxy attempt id as a path component: non-empty and path-safe (minted as ``<task id>-<uuid4>``)."""
    if not isinstance(attempt_id, str) or not ATTEMPT_ID_RE.match(attempt_id):
        raise DriveRefused(f"attempt_id {attempt_id!r} is not a path-safe attempt id ({ATTEMPT_ID_RE.pattern})")
    return attempt_id


def tunnel_record_problems(rec: Mapping[str, Any], *, view: str, stage: str, expected_control_key: str, w7: str) -> List[str]:
    """What the tunnel record (``run_petri_proxy`` writes it) must carry for this drive to trust it:
    :data:`TUNNEL_REQUIRED_KEYS`, every one present. ``control_key`` must EQUAL ``expected_control_key`` -- this
    drive's own six-axis key ``(W-sha7, wave_uuid, petri, view, stage, shard)`` -- exactly: a record found at a
    discovery location is otherwise a live tunnel of ANOTHER wave or W, and the results would land in roots
    labelled as this wave. ``attempt_root_on_volume`` is the attempt's root on the volume
    (``/vol/seedrep/<W-sha7>/exp/seedrep_battery_modal/petri/<view>/<shard>/attempts/<attempt_id>``) -- under
    THIS W's tree ``/vol/seedrep/<W-sha7>/``, no ``..``, no trailing slash -- and ``attempt_id`` its last
    component, both REQUIRED and never derived: the volume layout carries ``/exp/`` twice, so a
    ``serving_dir_on_volume.split("/exp/")`` fallback landed on ``/vol/seedrep/<W-sha7>`` and read a foreign
    ``routing_table.json``; ``serving_dir_on_volume`` must sit under ``<attempt root>/exp/`` (the TreeSyncer
    ships the container's exp tree there, which is what :func:`container_to_volume_path` maps onto)."""
    if not re.fullmatch(r"[0-9a-f]{7}", w7 or ""):
        raise ValueError(f"tunnel_record_problems needs W7 (7 hex), got {w7!r}")
    if not isinstance(expected_control_key, str) or expected_control_key.split("/")[0] != w7 or len(expected_control_key.split("/")) != 6:
        raise ValueError(f"expected_control_key {expected_control_key!r} is not this drive's six-axis key under W7 {w7}")
    problems = []
    for k in TUNNEL_REQUIRED_KEYS:
        if not rec.get(k):
            problems.append(f"missing {k}")
    if rec.get("view") and rec.get("view") != view:
        problems.append(f"view {rec.get('view')!r} != {view!r}")
    if rec.get("stage") and rec.get("stage") != stage:
        problems.append(f"stage {rec.get('stage')!r} != {stage!r}")
    ck = rec.get("control_key")
    if ck and ck != expected_control_key:
        problems.append(f"control_key {ck!r} != this drive's {expected_control_key!r} (W7/wave_uuid/leg/view/stage/shard) "
                        "-- another wave's or W's proxy")
    u = str(rec.get("url") or "")
    if u and not u.startswith("https://"):
        problems.append(f"url {u!r} is not https (modal.forward)")
    w_root = f"{VOL_PATH}/{LANE}/{w7}/"
    sd = str(rec.get("serving_dir_on_volume") or "")
    if sd and not sd.startswith(w_root):
        problems.append(f"serving_dir_on_volume {sd!r} is not under {w_root}")
    if sd and ".." in PurePosixPath(sd).parts:
        problems.append(f"serving_dir_on_volume {sd!r} carries a '..' component")
    ar = str(rec.get("attempt_root_on_volume") or "")
    if ar:
        if not ar.startswith(w_root):
            problems.append(f"attempt_root_on_volume {ar!r} is not under {w_root}")
        if ".." in PurePosixPath(ar).parts:
            problems.append(f"attempt_root_on_volume {ar!r} carries a '..' component")
        if ar.endswith("/"):
            problems.append(f"attempt_root_on_volume {ar!r} ends with '/'")
        if sd and not sd.startswith(ar.rstrip("/") + "/exp/"):
            problems.append(f"serving_dir_on_volume {sd!r} is not under the attempt's exp tree {ar.rstrip('/')}/exp/")
    aid = rec.get("attempt_id")
    if aid:
        if not isinstance(aid, str) or not ATTEMPT_ID_RE.match(aid):
            problems.append(f"attempt_id {aid!r} is not a path-safe attempt id")
        elif ar and PurePosixPath(ar.rstrip("/")).name != aid:
            problems.append(f"attempt_id {aid!r} != the attempt root's last component {PurePosixPath(ar.rstrip('/')).name!r}")
    return problems


def resolve_serving_dir(tunnel: Mapping[str, Any], discovery_dir: str = "") -> str:
    """The proxy's serving dir is the tunnel record's ``serving_dir_on_volume``, ALWAYS. ``--serving-vol`` is a
    DISCOVERY location only -- where to look for the record's volume copy when the Dict has none -- and when given,
    its normalised value must EQUAL the record's normalised value; it never replaces it (an override
    could point a validated record at another attempt's ``endpoint.json``). Returns the record's value verbatim."""
    recorded = str(tunnel.get("serving_dir_on_volume") or "")
    if not recorded:
        raise DriveRefused("the tunnel record names no serving_dir_on_volume")
    if discovery_dir:
        given, rec_norm = posixpath.normpath(discovery_dir), posixpath.normpath(recorded)
        if given != rec_norm:
            raise DriveRefused(f"--serving-vol {discovery_dir!r} (normalised {given!r}) != the tunnel record's serving_dir_on_volume "
                               f"{recorded!r} -- --serving-vol is a discovery location only and never replaces the record's serving dir; "
                               "drop it, or point it at the record's dir")
    return recorded


def endpoint_engine_problem(ep: Mapping[str, Any]) -> Optional[str]:
    """Why a standup ``endpoint.json`` is not the lane's native engine (None = it is). ENGINE FIRST, through main's one
    contract reader (``tests/test_tinycap_launcher_native.py`` admits exactly two engine-aware readers of the lens
    reduction-mode fact under ``scripts/modal`` and this drive is neither, so it reads NO lens fact of its own): a lens record
    (no engine key, or ``lens``) is refused by name; a record whose two engine spellings disagree is torn and refuses there.
    Then the WHOLE record through main's engine-aware ``served_reduction_mode`` reader: a native
    record that also states a lens fact is two engines on one record and refuses there, in the reader's words -- the refusal the
    lane carried before the reader refactor, restored without this drive spelling the lens key itself."""
    try:
        native = endpoint_is_native(ep)
    except ValueError as exc:                      # torn: two spellings of one engine that disagree
        return f"endpoint.json: {exc}"
    if not native or ep.get("engine") != "lora_native":
        return f"endpoint.json engine={ep.get('engine')} -- the lane requires lora_native"
    try:
        served_reduction_mode(ep, source="endpoint.json")
    except ValueError as exc:                      # a native record carrying a lens fact: two engines on one record
        return f"{exc} -- two engines on one record"
    return None


def attempt_paths(tunnel: Mapping[str, Any], ep: Mapping[str, Any], container_exp_root: str) -> Dict[str, str]:
    """The volume paths of the proxy's ``routing_table.json`` / ``cells_served.json`` under ITS attempt root:
    the tunnel record's EXPLICIT ``attempt_root_on_volume`` + ``/exp`` is where the TreeSyncer
    ships the container's exp tree, and the container paths ``endpoint.json`` names map onto it through
    :func:`container_to_volume_path`. No derivation from the serving dir, ever."""
    root = str(tunnel.get("attempt_root_on_volume") or "")
    if not root:
        raise DriveRefused("the tunnel record names no attempt_root_on_volume -- refusing to derive the attempt root from the serving dir")
    root = root.rstrip("/")
    attempt_exp = root + "/exp"
    return {"attempt_root": root, "attempt_exp": attempt_exp,
            "routing_table": container_to_volume_path(str(ep["routing_table"]), container_exp_root, attempt_exp),
            "cells_served": container_to_volume_path(str(ep["cells_served"]), container_exp_root, attempt_exp)}


def parse_legs_arg(legs: str, kept: Sequence[int]) -> Optional[List[int]]:
    """``ALL`` -> None (every served leg); else the listed cell indices, each refused unless served."""
    if legs.strip().upper() == "ALL":
        return None
    out = []
    for tok in re.split(r"[\s,]+", legs.strip()):
        if not tok:
            continue
        if not tok.isdigit() or int(tok) not in kept:
            raise DriveRefused(f"leg index {tok!r} is not a served leg ({list(kept)})")
        out.append(int(tok))
    return out


def free_local_port() -> int:
    """A loopback port the OS just handed out free (bind 0, read it back, release)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def local_port_of(proxy_port: int) -> int:
    """The drive's local relay port (F-AR): a FREE OS-assigned loopback port in ``LOCAL_PORT_MIN..65535``, distinct from the
    proxy's own -- never ``proxy_port + 10000``. A proxy published proxy_port 57204; 57204 + 10000 = 67204 > 65535 and
    `local_port_held` raised OverflowError at bind time, the drive died and the READY proxy idled to its cap. This is the
    grading host's side of F-AL (`relay_port_for` in seedrep_served.py picks the container's relay port the same way). No
    cross-drive stability is needed: only THIS drive's own legs read the port (PETRI_PROXY_BASE_URL in their env), and every
    record of it (DRIVE_START, the drive record's local_relay) is written from the port actually bound."""
    for _ in range(64):
        p = free_local_port()
        if p != int(proxy_port) and LOCAL_PORT_MIN <= p <= 65535:
            return p
    raise DriveRefused(f"no free local relay port distinct from the proxy port {proxy_port}")


def local_port_held(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", int(port)))
            return False
        except OSError:
            return True


def open_local_relay(tunnel: Mapping[str, Any], tunnel_host: str) -> Tuple[int, dict]:
    """THE TUNNEL at loopback: choose the local relay port from the tunnel record's proxy_port (F-AR: a free port, see
    `local_port_of`), refuse one something else holds, start the TLS relay on it. Returns ``(local_port, relay)``; the relay's
    ``listen`` names the same port, which is what DRIVE_START and the drive record carry."""
    local_port = local_port_of(int(tunnel["proxy_port"]))
    if local_port_held(local_port):
        raise DriveRefused(f"local port {local_port} is held by something else -- free it first")
    relay = start_local_tls_relay(local_port, tunnel_host)
    return local_port, relay


def materialised_bundle_dir(out_root: Path, bundle_hash: str = BUNDLE_HASH) -> Path:
    return out_root / f"behaviors-{bundle_hash}"


def launch_legs(to_run: Sequence[LegRun], out_roots: Mapping[str, Path], *, start: Callable[[LegRun], Any],
                bundle_ready: Callable[[str], bool], now: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                log: Callable[[str], None] = print, wait_cap_s: float = BUNDLE_WAIT_CAP_S, stagger_s: float = LAUNCH_STAGGER_S,
                poll_s: float = 5.0) -> List[dict]:
    """Launch the legs. The FIRST leg of every distinct out root launches at once -- a fresh root's
    bundle can only be materialised by a leg writing into it, so waiting for it there waited on nothing (the
    three fresh Arditi roots launched at 0 / 902 / 1,804 s). A later leg on the SAME root waits, at most
    ``wait_cap_s``, for that root's materialised bundle while the root's live writer is alive (two legs must not
    materialise one root at once); a root whose writer died is claimed by the next leg. ``start(run)`` launches
    the leg and returns a handle with ``is_alive()`` and ``name``; ``bundle_ready(tag)`` reads the root. Returns
    one event per launch -- ``{record_name, tag, t, waited_s, waited_on}`` in ``now()``'s clock."""
    writer: Dict[str, Any] = {}
    events: List[dict] = []
    for run in to_run:
        waited_s, waited_on = 0.0, None
        w = writer.get(run.tag)
        if w is not None and w.is_alive() and not bundle_ready(run.tag):
            t0 = now()
            while not bundle_ready(run.tag) and w.is_alive() and now() - t0 < wait_cap_s:
                sleep(poll_s)
            waited_s, waited_on = now() - t0, w.name
            log(f"{run.record_name}: bundle under {out_roots[run.tag]} materialised={bundle_ready(run.tag)} "
                f"(waited {int(waited_s)} s on {w.name}{'' if w.is_alive() else ', which ended'})")
        h = start(run)
        if w is None or not w.is_alive():
            writer[run.tag] = h
        events.append({"record_name": run.record_name, "tag": run.tag, "t": now(), "waited_s": waited_s, "waited_on": waited_on})
        sleep(stagger_s)
    return events


def sentinel_content(kind: str, drive_id: str, extra: str = "") -> bytes:
    """A timestamp + the drive id (+ counts for STOP) so the container's content-keyed mirror sees every touch."""
    return f"{kind} {utc_now()} drive={drive_id}{(' ' + extra) if extra else ''}\n".encode()


def attempt_binding(bound: Mapping[str, Any], current: Any) -> Tuple[str, str]:
    """Whether the tunnel record NOW under this drive's key is the one it bound to at start -> ``(state, why)``, state in
    {bound, missing, unreadable, stale} (a completion is evidence only about the attempt whose record
    the proxy is waiting under). ``missing``: nothing under the key -- the proxy pops its record when its wait ends (a
    timeout, the wave's stop, an unbound STOP), so the attempt this drive drove is GONE; ``unreadable``: not a JSON object;
    ``stale``: a record of ANOTHER attempt (a re-executed proxy cleared the key's sentinels and published its own) -- every
    differing :data:`BINDING_FIELDS` entry named. Only ``bound`` publishes."""
    if current is None:
        return "missing", (f"no {TUNNEL_FILE} under the key -- proxy attempt {bound['attempt_id']} ended (WS2-D pops the record when its wait "
                           "ends) or its record was never published")
    try:
        cur = json.loads(current.decode() if isinstance(current, (bytes, bytearray)) else current) \
            if isinstance(current, (str, bytes, bytearray)) else current
    except ValueError as e:
        return "unreadable", f"{TUNNEL_FILE} under the key is not JSON ({e})"
    if not isinstance(cur, dict):
        return "unreadable", f"{TUNNEL_FILE} under the key is not a JSON object ({type(cur).__name__})"
    diffs = [f"{k}: record {cur.get(k)!r} != bound {bound[k]!r}" for k in BINDING_FIELDS if cur.get(k) != bound[k]]
    if diffs:
        return "stale", "another attempt's record now holds the key -- " + "; ".join(diffs)
    return "bound", f"the record under the key names this drive's attempt {bound['attempt_id']}"


def drive_verdict_record(*, bound: Mapping[str, Any], drive_id: str, verdict: Mapping[str, Any], identity: Mapping[str, Any],
                         stop_why: str) -> dict:
    """The attempt-bound ``drive_verdict.json`` that ``petri_completion_evidence`` reads: it requires ``attempt_id`` (== the
    proxy's), ``drive_id`` and ``verdict.{counts, nruns, release}`` -- ``complete`` iff ``release`` is true and ``counts.failed``
    == 0, ``incomplete`` otherwise (a --teardown-after-failure). Everything else is provenance for the audit."""
    return {"schema": VERDICT_SCHEMA, "attempt_id": check_attempt_id(bound["attempt_id"]), "attempt_root_on_volume": bound["attempt_root_on_volume"],
            "control_key": bound["control_key"], "drive_id": drive_id,
            **{k: identity[k] for k in ("W", "W7", "wave_uuid", "view", "stage", "shard", "petri_pass")},
            "verdict": {"counts": {k: int(verdict["counts"][k]) for k in ("complete", "incomplete", "failed")}, "nruns": int(verdict["nruns"]),
                        "release": bool(verdict["release"]), "detail": [list(x) for x in verdict.get("detail", [])]},
            "stop_why": stop_why, "written_at": utc_now()}


def stop_sentinel_extra(counts: Mapping[str, int], attempt_id: str, stop_why: str) -> str:
    """The STOP sentinel's tail as ``parse_stop_sentinel`` reads it (whitespace-separated ``k=v`` tokens):
    ``complete=N incomplete=N failed=N attempt=<id>[ reason=<why>]``. ``attempt=`` BINDS the STOP to the proxy attempt -- a
    STOP without it stops the proxy but is NOT completion evidence (``stopped_unbound``, rc 1: the state every
    successful drive used to land in)."""
    if not stop_why or any(c.isspace() for c in stop_why):
        raise ValueError(f"stop_why must be one whitespace-free token, got {stop_why!r}")
    return (f"complete={int(counts['complete'])} incomplete={int(counts['incomplete'])} failed={int(counts['failed'])} "
            f"attempt={check_attempt_id(attempt_id)}" + ("" if stop_why == "release" else f" reason={stop_why}"))


def completion_key(key: str, attempt_id: str) -> str:
    """``<K>/<attempt_id>/completion.json`` -- the completion bundle's ATTEMPT-NAMESPACED key under the six-axis key ``K``. The proxy
    derives it from ITS attempt id and watches only that key, so a bundle of an earlier attempt can never be read as this attempt's."""
    if not isinstance(key, str) or not key or key.endswith("/"):
        raise ValueError(f"completion_key needs the six-axis key, got {key!r}")
    return f"{key}/{check_attempt_id(attempt_id)}/{COMPLETION_NAME}"


def completion_bundle(*, bound: Mapping[str, Any], drive_id: str, verdict: Mapping[str, Any], identity: Mapping[str, Any], stop_why: str) -> dict:
    """The ONE atomic completion record: :func:`drive_verdict_record`'s fields (so the proxy's verdict reader -- ``attempt_id``, ``drive_id``,
    ``verdict.release``, ``verdict.counts.failed`` -- reads it unchanged) + ``stop`` = the STOP sentinel's exact content and reason."""
    rec = drive_verdict_record(bound=bound, drive_id=drive_id, verdict=verdict, identity=identity, stop_why=stop_why)
    content = sentinel_content("STOP", drive_id, stop_sentinel_extra(verdict["counts"], bound["attempt_id"], stop_why)).decode()
    return {**rec, "schema": COMPLETION_SCHEMA, "stop": {"content": content, "why": stop_why}}


def read_sentinel_now(sentinels: Any, key: str, *, tries: int = 3, sleep_s: float = 10.0, log: Callable[[str], Any] = print,
                      sleep: Callable[[float], Any] = time.sleep) -> Any:
    """The value under ``key`` NOW (``None`` when absent). A Dict read that RAISES is retried ``tries`` times and then re-raised:
    a transport failure must never read as "missing" (which would refuse to publish a real completion)."""
    for i in range(tries):
        try:
            return sentinels.get(key)
        except Exception as e:  # noqa: BLE001
            log(f"sentinel read {key} failed ({type(e).__name__}: {e}) -- try {i + 1}/{tries}")
            if i + 1 == tries:
                raise
            sleep(sleep_s)
    raise AssertionError("unreachable")


def publish_completion(sentinels: Any, *, current_tunnel: Any, key: str, bound: Mapping[str, Any], drive_id: str, verdict: Mapping[str, Any],
                       identity: Mapping[str, Any], write_stop: bool, stop_why: str, legacy_generic: bool = False,
                       log: Callable[[str], Any] = print) -> dict:
    """Publish this drive's completion to the proxy as ONE atomic bundle under the ATTEMPT-NAMESPACED key
    :func:`completion_key` -- or refuse by name. ``key`` is the six-axis key ``K`` (== the bound record's ``control_key``). The proxy's wait
    ends on ANY non-waiting evidence, so a HELD drive (no STOP decided) publishes NOTHING: a verdict now would end the proxy this hold
    keeps up for the re-drive. A decided STOP publishes only when the tunnel record now under the key is the one this drive bound to
    (:func:`attempt_binding`): a missing or stale record means the attempt is gone or replaced, and a verdict bound to it would be
    evidence about nothing the proxy is waiting on -- the stores stand locally and a drive against the live attempt continues them with
    ``--resume-from``. The binding read is NOT atomic with the writes (a replacement attempt may install its record in between);
    the attempt id in the key is what makes that harmless: the bundle lands under THIS attempt's key, which no other
    attempt watches. ``legacy_generic`` ALSO writes the generic ``<K>/drive_verdict.json`` then ``<K>/STOP`` (the pre-namespace
    contract) -- for a proxy that still watches them, and re-opening the race for any such reader."""
    if key != bound["control_key"]:
        raise ValueError(f"publish_completion: key {key!r} is not the bound record's control_key {bound['control_key']!r}")
    state, why = attempt_binding(bound, current_tunnel)
    out: dict = {"binding": state, "binding_why": why, "write_stop": bool(write_stop), "stop_why": stop_why, "published": [],
                 "completion_key": None, "bundle": None, "verdict_record": None, "stop_content": None, "legacy_generic": bool(legacy_generic)}
    if not write_stop:
        log(f"COMPLETION_HELD: no {COMPLETION_NAME} published (binding={state}) -- the proxy stays up for the re-drive")
        return out
    if state != "bound":
        log(f"COMPLETION_UNPUBLISHED binding={state}: {why} -- no {COMPLETION_NAME} written under this drive's attempt key; the Petri stores of "
            f"attempt {bound['attempt_id']} stand locally; a drive bound to the live attempt continues them with --resume-from")
        return out
    ck = completion_key(key, bound["attempt_id"])
    bundle = completion_bundle(bound=bound, drive_id=drive_id, verdict=verdict, identity=identity, stop_why=stop_why)
    sentinels[ck] = json.dumps(bundle)                                            # ONE write: the verdict and the STOP land together
    out.update(published=[ck], completion_key=ck, bundle=bundle, stop_content=bundle["stop"]["content"],
               verdict_record={**{k: v for k, v in bundle.items() if k != "stop"}, "schema": VERDICT_SCHEMA})
    if legacy_generic:
        sentinels[f"{key}/{VERDICT_NAME}"] = json.dumps(out["verdict_record"])
        out["published"].append(f"{key}/{VERDICT_NAME}")
        sentinels[f"{key}/{STOP_NAME}"] = bundle["stop"]["content"]
        out["published"].append(f"{key}/{STOP_NAME}")
    return out


def drive_rc(verdict: Mapping[str, Any], completion: Mapping[str, Any]) -> int:
    """The drive's exit code. 0: the verdict RELEASED and the completion was published bound to the live attempt (the consumer
    accepts the proxy). 2: released, but NOT publishable -- the tunnel record under the key was missing, unreadable or another
    attempt's; the stores are complete locally, nothing was published. 1: legs FAILED (held, or torn down by
    --teardown-after-failure)."""
    if not verdict["release"]:
        return 1
    return 0 if completion["binding"] == "bound" and completion["published"] else 2


def audit_side_identical(wt: Path, sha_a: str, sha_b: str) -> bool:
    if sha_a == sha_b:
        return True
    r = subprocess.run(["git", "diff", "--quiet", sha_a, sha_b, "--", *AUDIT_SIDE_PATHS], cwd=str(wt))
    return r.returncode == 0


def non_audit_diff(wt: Path, sha_a: str, sha_b: str) -> list:
    if sha_a == sha_b:
        return []
    r = subprocess.run(["git", "diff", "--name-only", sha_a, sha_b], cwd=str(wt), capture_output=True, text=True)
    return [f for f in r.stdout.split() if not any(f == a or f.startswith(a.rstrip("/") + "/") for a in AUDIT_SIDE_PATHS)]


def arm_grading_host_bank_view(arms: dict, key: str, arm: str, home: Path) -> Tuple[Path, str, str, str]:
    """The arms table is the authority for the Arditi bank root / subdir / cell / operating-point view."""
    a = arms["arms"][arm]
    view = (((arms.get("targets") or {}).get(key) or {}).get(arm) or {}).get("bank_view") or ""
    if not view:
        raise DriveRefused(f"the arms table has no {arm} operating point for {key}")
    root = a["bank_root"]
    root = root[len("exp/objects/"):] if root.startswith("exp/objects/") else root
    sub = a.get("bank_subdir") or ""
    p = home / "arditi-stage" / "objects" / root / key / (sub if sub else "") / "cells_view" / view
    return Path(os.path.normpath(p)), a["cell"], view, a.get("roster") or ""


def start_local_tls_relay(listen_port: int, host: str, port: int = 443) -> dict:
    """127.0.0.1:listen_port -> TLS(host:port) with SNI=host, byte for byte (the Modal edge routes by SNI)."""
    import asyncio
    counters = {"connections": 0, "open": 0, "errors": 0}
    loop = asyncio.new_event_loop()
    ctx = ssl.create_default_context()
    ref: dict = {}

    async def pipe(r, w):
        try:
            while True:
                data = await r.read(65536)
                if not data:
                    break
                w.write(data)
                await w.drain()
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass

    async def handle(cr, cw):
        counters["connections"] += 1
        counters["open"] += 1
        try:
            ur, uw = await asyncio.open_connection(host, port, ssl=ctx, server_hostname=host)
        except Exception:  # noqa: BLE001
            counters["errors"] += 1
            counters["open"] -= 1
            cw.close()
            return
        try:
            await asyncio.gather(pipe(cr, uw), pipe(ur, cw))
        finally:
            counters["open"] -= 1

    async def serve():
        ref["server"] = await asyncio.start_server(handle, "127.0.0.1", listen_port)
        async with ref["server"]:
            await ref["server"].serve_forever()

    def run():
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(serve())
        except Exception as e:  # noqa: BLE001
            counters["fatal"] = f"{type(e).__name__}: {e}"

    th = threading.Thread(target=run, daemon=True, name="local-tls-relay")
    th.start()
    time.sleep(0.5)

    def stop():
        srv = ref.get("server")
        if srv is not None:
            loop.call_soon_threadsafe(srv.close)
        loop.call_soon_threadsafe(loop.stop)
    return {"listen": f"127.0.0.1:{listen_port}", "upstream": f"tls://{host}:{port}", "counters": counters, "stop": stop, "thread": th}


def _vol():
    import modal
    return modal.Volume.from_name(VOL_NAME, create_if_missing=False)


def vol_read(vol, vol_path: str) -> Optional[bytes]:
    try:
        return b"".join(vol.read_file(volume_client_path(vol_path)))
    except Exception:  # noqa: BLE001 -- absent (or not yet synced)
        return None


def _sentinels(name: str):
    import modal
    return modal.Dict.from_name(name, create_if_missing=True)


def call_running(call_id: str) -> Tuple[bool, str]:
    """The analogue of ``squeue -j $JOBID == RUNNING`` for the proxy's container call."""
    import modal
    from modal.exception import OutputExpiredError
    try:
        fc = modal.FunctionCall.from_id(call_id)
        fc.get(timeout=0)
        return False, "the container call has RETURNED (the proxy is gone)"
    except OutputExpiredError:
        return False, "the container call's output expired (it ended long ago)"
    except TimeoutError:
        return True, "running"
    except Exception as e:  # noqa: BLE001
        return False, f"the container call raised: {type(e).__name__}: {e}"


def http_json(url: str, timeout: float = 15.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def probe_petri_models(base: str, timeout: float = 10.0) -> Tuple[Optional[dict], bool]:
    """``GET <base>/v1/models`` through the relay: ``(the listing, it serves a petri__ vmodel)``; ``(None, False)``
    when the proxy is unreachable or the listing is malformed."""
    try:
        models = http_json(f"{base}/v1/models", timeout=timeout)
        return models, any("petri__" in str(m.get("id", "")) for m in models.get("data", []))
    except Exception:  # noqa: BLE001 -- not up yet / gone
        return None, False


def load_wave(wave_json: Path) -> Tuple[str, str]:
    """``RS/wave.json`` -> (wave_uuid, launch_plan_sha or ''); the uuid is the wave's identity."""
    try:
        d = json.loads(Path(wave_json).read_text())
    except (OSError, ValueError) as e:
        raise DriveRefused(f"cannot read {wave_json}: {e}")
    wu = str(d.get("wave_uuid") or "")
    wave_uuid8(wu)
    return wu, str(d.get("launch_plan_sha") or "")


# --------------------------------------------------------------------------- the drive

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--sha", required=True, help="W: the FULL 40-hex sha this worktree is at (clean)")
    ap.add_argument("--proxy-sha", default="", help="the tree sha the PROXY container runs (default: --sha)")
    ap.add_argument("--wave-json", required=True, help="RS/wave.json (wave_uuid minted at launch)")
    ap.add_argument("--view", required=True, choices=VIEWS)
    ap.add_argument("--shard", default=DEFAULT_SHARD)
    ap.add_argument("--petri-pass", default="", choices=list(STAGES))
    ap.add_argument("--legs", default="ALL", help='ALL | "0 1" (cell indices of the enumeration)')
    ap.add_argument("--width", type=int, default=None, help="PETRI_EVAL_MAX_CONNECTIONS per run (default: 3 view / 4 Arditi)")
    # the org is NOT a choice: the Petri auditor + judge run on highprio; auditor_env refuses anything else
    ap.add_argument("--auditor-org", default=AUDITOR_ORG, choices=[AUDITOR_ORG])
    ap.add_argument("--python", default="", help="the LEG python (a venv with the `petri` dependency group)")
    # the auditor chain is API-only (sonnet -> grok); the abliterated-Qwen third leg is GONE and no pool is stood up
    # for it. The removed flags are REFUSED by name below so the old command line cannot run silently.
    ap.add_argument("--qwen-auditor-pool", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--qwen-auditor-url", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--allow-unreachable-qwen-auditor", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--retry-sleep-cap-s", type=int, default=DEFAULT_RETRY_SLEEP_CAP_S)
    ap.add_argument("--container-exp-root", default=DEFAULT_CONTAINER_EXP_ROOT)
    ap.add_argument("--serving-vol", default="", help="DISCOVERY only: a volume dir to look for the tunnel record's copy in when the Dict has "
                    "none; it must equal the record's serving_dir_on_volume and never replaces it")
    ap.add_argument("--wait-endpoint-min", type=float, default=45.0)
    ap.add_argument("--teardown-after-failure", action="store_true",
                    help="write STOP (tear the proxy down) even when legs FAILED -- an explicit, named teardown; by default a FAILED leg HOLDS "
                         "the proxy up so a re-invocation can resume it, on every pass (the re-run pass included)")
    ap.add_argument("--legacy-generic-completion", action="store_true",
                    help="ALSO mirror the completion to the generic six-axis keys <K>/drive_verdict.json + <K>/STOP (the pre-attempt-namespace "
                         "contract) for a proxy that still watches them; default off -- a generic key can be read by a REPLACEMENT proxy attempt "
                         "(the fourth review's P2-2), the attempt-namespaced <K>/<attempt_id>/completion.json cannot")
    ap.add_argument("--resume-from", default="", help="EXPLICIT import of a PRIOR wave/attempt's Petri stores: that drive's attempt state dir "
                    "(.../<view>/attempts/<attempt_id>); its out roots' cells/ are copied into this attempt's fresh roots with provenance "
                    "(RESUMED_FROM.json). Never implicit -- a new wave at the same W starts empty")
    ap.add_argument("--dry-run", action="store_true", help="stop after freezing the enumeration and the schedule (no leg runs, no sentinel)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sha = args.sha
    w7 = w7_of(sha)
    wave_uuid, launch_plan_sha = load_wave(Path(args.wave_json).expanduser())
    wu8 = wave_uuid8(wave_uuid)
    stage = STAGES[args.petri_pass]
    vp = view_params(args.view)
    # --width only NARROWS this view; the ceiling is checked over the EFFECTIVE widths, this view's override included
    width = resolve_width(vp, args.width)
    effective_widths = {v: (width if v == args.view else view_params(v).width) for v in VIEWS}
    assert_within_auditor_ceiling(widths=effective_widths)
    wt = LOCAL_REPO
    py = args.python or str(wt / ".venv" / "bin" / "python")
    if not os.access(py, os.X_OK):
        raise DriveRefused(f"no leg python at {py} (uv sync --extra cpu --group petri into a venv and pass --python)")
    home = Path.home()
    state = state_dir_for(home, w7, wu8, args.view, args.petri_pass)     # WAVE-scoped: the log + the tunnel wait
    state.mkdir(parents=True, exist_ok=True)
    log_p = state / "drive.log"
    drive_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.{os.getpid()}"

    def log(msg: str) -> None:
        line = f"[{utc_now()}] {msg}"
        print(line, flush=True)
        with log_p.open("a") as f:
            f.write(line + "\n")

    # 1. PROVENANCE + the grading-host-side inputs
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(wt), capture_output=True, text=True).stdout.strip()
    if head != sha:
        raise DriveRefused(f"worktree HEAD {head[:12]} != launched {sha[:12]} -- git checkout --detach {sha}")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "configs", "scripts", "src"], cwd=str(wt)).returncode != 0:
        raise DriveRefused("worktree dirty under configs/scripts/src")
    proxy_sha = args.proxy_sha or sha
    w7_of(proxy_sha)
    if not audit_side_identical(wt, proxy_sha, sha):
        raise DriveRefused(f"the audit-side paths {AUDIT_SIDE_PATHS} differ between the proxy's tree {proxy_sha[:12]} and this driver's {sha[:12]}")
    for rel_ in (vp.cfg_rel, vp.manifest_rel, "configs/sdfbase_full22_arms.yaml", "scripts/petri_run.py"):
        if not (wt / rel_).is_file():
            raise DriveRefused(f"missing {rel_}")
    # the grading host bank view: a seed view's from the LANE bank root (F-R; refused by name when absent), Arditi's from the arms table
    if vp.is_arditi:
        import yaml
        arms = yaml.safe_load((wt / "configs/sdfbase_full22_arms.yaml").read_text())
        bank_view, arm_cell, _arm_view, _roster = arm_grading_host_bank_view(arms, MODEL_KEY, ARDITI_ARM, home)
        if not (bank_view / arm_cell / "object.pt").is_file():
            raise DriveRefused(f"no grading host bank view for arm {ARDITI_ARM} at {bank_view}/{arm_cell}/object.pt")
        grading_host_bank = {"source": "arms_table", "root": str(home / vp.bank_root_rel), "root_rel": vp.bank_root_rel, "view_dir": str(bank_view)}
    else:
        bank_view = resolve_grading_host_bank_view(vp, home)
        grading_host_bank = {"source": "lane_bank", "root": str(home / vp.bank_root_rel), "root_rel": vp.bank_root_rel, "view_dir": str(bank_view)}
    bundle = home / ".local" / "state" / "petri" / "scenarios" / BUNDLE_HASH
    if not bundle.is_dir():
        raise DriveRefused(f"no bundle copy at {bundle}")
    r = subprocess.run([py, "-c", "import inspect_ai, petri_bloom"], capture_output=True, text=True)
    if r.returncode != 0:
        raise DriveRefused(f"the leg venv lacks inspect_ai/petri_bloom: {(r.stdout + r.stderr)[-500:]}")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise DriveRefused("OPENROUTER_API_KEY is unset -- the auditor chain's grok leg needs it (export it in the drive's environment)")

    # the auditor + judge key: HIGHPRIO through the vault mechanism (never a file, never argv); the record carries the sha12 only
    key_env, key_stamp = auditor_env(args.auditor_org)
    # the auditor chain is sonnet -> grok, API-only; a command line still naming the retired Qwen third leg is refused by name
    if args.qwen_auditor_pool is not None or args.qwen_auditor_url is not None or args.allow_unreachable_qwen_auditor:
        raise DriveRefused(f"{REMOVED_QWEN_LEG_MSG}")
    qwen_env: Dict[str, str] = {}
    qwen_ok = False

    # 2. THE PROXY: the tunnel record from the Dict first, the volume second
    vol = _vol()
    sentinel_dict = dict_name(w7, wu8, "sentinels")
    sentinels = _sentinels(sentinel_dict)

    def key_for(name: str) -> str:            # every record of this drive sits under the same six axes
        return control_key(w7, wave_uuid, LEG, args.view, stage, args.shard, name)

    tkey = key_for(TUNNEL_FILE)
    deadline = time.time() + args.wait_endpoint_min * 60
    tunnel = None
    while time.time() < deadline and tunnel is None:
        raw = None
        try:
            raw = sentinels.get(tkey)
        except Exception:  # noqa: BLE001
            raw = None
        if raw is None and args.serving_vol:
            b = vol_read(vol, f"{args.serving_vol.rstrip('/')}/{TUNNEL_FILE}")
            raw = b.decode() if b else None
        if raw:
            try:
                tunnel = json.loads(raw)
            except ValueError:
                tunnel = None
        if tunnel is None:
            time.sleep(20)
    if tunnel is None:
        raise DriveRefused(f"no {TUNNEL_FILE} under Dict key {tkey} (nor {args.serving_vol or 'no --serving-vol'}) within {args.wait_endpoint_min} min")
    # the record is bound to THIS drive's six axes (its control_key == key_for("")) and to THIS W's volume tree before anything is
    # read off it; --serving-vol was only where to LOOK for it
    problems = tunnel_record_problems(tunnel, view=args.view, stage=stage, expected_control_key=key_for(""), w7=w7)
    if problems:
        raise DriveRefused(f"tunnel record at {tkey}: " + "; ".join(problems))
    attempt_id = check_attempt_id(tunnel["attempt_id"])
    astate = attempt_state_dir(state, attempt_id)                      # ATTEMPT-scoped: every record bound to this proxy attempt
    astate.mkdir(parents=True, exist_ok=True)
    serving_vol = resolve_serving_dir(tunnel, args.serving_vol)
    ep_b = vol_read(vol, f"{serving_vol.rstrip('/')}/endpoint.json")
    if not ep_b:
        raise DriveRefused(f"no endpoint.json under {serving_vol}")
    ep = json.loads(ep_b)
    (astate / "endpoint.json").write_text(json.dumps(ep, indent=1))
    (astate / TUNNEL_FILE).write_text(json.dumps(tunnel, indent=1))
    if ep.get("mode") != "petri":
        raise DriveRefused(f"endpoint mode={ep.get('mode')}, not petri")
    problem = endpoint_engine_problem(ep)
    if problem is not None:
        raise DriveRefused(problem)
    exp_root = str(tunnel.get("container_exp_root") or args.container_exp_root)
    # the attempt root is the record's EXPLICIT field; the container paths endpoint.json names map under <attempt root>/exp
    paths = attempt_paths(tunnel, ep, exp_root)
    attempt_root_vol = paths["attempt_root"]
    rt_b = vol_read(vol, paths["routing_table"])
    cs_b = vol_read(vol, paths["cells_served"])
    if not (rt_b and cs_b):
        raise DriveRefused(f"routing table / cells_served named by endpoint.json not on the volume under {paths['attempt_exp']}: "
                           f"{ep['routing_table']} {ep['cells_served']}")
    (astate / "routing_table.json").write_bytes(rt_b)
    (astate / "cells_served.json").write_bytes(cs_b)
    rt, cs = json.loads(rt_b), json.loads(cs_b)
    if not (rt.get("version") == 2 and rt.get("adapters")):
        raise DriveRefused("the pulled routing table is not a version-2 table with adapter blocks")
    if not (cs.get("engine") == "lora_native" and cs.get("stage") != "pre_launch"):
        raise DriveRefused("cells_served.json is not the FINAL native record")
    served_models = list(ep.get("served_models") or [])
    running, note = call_running(tunnel["function_call_id"])
    if not running:
        raise DriveRefused(f"proxy container {tunnel['function_call_id']}: {note}")

    # 3. THE TUNNEL at loopback
    tunnel_host = urlparse(tunnel["url"]).hostname
    local_port, relay = open_local_relay(tunnel, tunnel_host)   # a free port, never proxy_port + 10000 (which overflowed to 67204)
    base = f"http://127.0.0.1:{local_port}"
    models = None
    for _ in range(30):
        models, ok = probe_petri_models(base)
        if ok:
            break
        time.sleep(3)
    if not models:
        raise DriveRefused(f"{base}/v1/models unreachable through the tunnel")
    (astate / "models.json").write_text(json.dumps(models, indent=1))
    pxk = ((models.get("petri_proxy") or {}).get("engine") or {}).get("kind")
    if pxk != "lora_native":
        raise DriveRefused(f"/v1/models petri_proxy.engine.kind={pxk}, not lora_native -- refusing to audit")

    # 4. THE ENVIRONMENT the legs run under
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY_LOWPRIO", None)   # the legs see ONE Anthropic key: highprio's, under its own name
    env.update(key_env)
    env.update(qwen_env)
    env.update({
        "PETRI_PROXY_BASE_URL": f"{base}/v1",
        "PETRI_TWO_SI_CONFIG": vp.cfg_rel,
        "TT_TWO_SI_DEFENSES_DIR": str(bank_view),
        "PETRI_PROMPT_OVERRIDE_DIR": str(wt / "configs/capeval/petri/prompts"),
        "PETRI_BUNDLE_DIR": str(bundle),
        "PETRI_AUDITOR_RETRY_SLEEP_CAP_S": str(int(args.retry_sleep_cap_s)),
        "F22_SERVING_PATH": "lora_native",
    })
    log(f"DRIVE_START {LANE} view={args.view} stage={stage} shard={args.shard} drive_id={drive_id} W={sha} W7={w7} wave_uuid={wave_uuid} "
        f"launch_plan_sha={launch_plan_sha or '-'} container_call={tunnel['function_call_id']} task={tunnel.get('task_id')} "
        f"tunnel={tunnel_host} relay={relay['listen']}->{relay['upstream']} base={base} proxy_port={tunnel.get('proxy_port')} proxy_sha={proxy_sha} "
        f"serving_dir={serving_vol} attempt_id={attempt_id} attempt_root={attempt_root_vol} state={astate} width={width} "
        f"auditor_org={key_stamp['pool']} key_sha256_12={key_stamp['key_sha256_12']} "
        f"qwen_auditor={qwen_env.get(QWEN_BASE_URL_ENV, '-')} qwen_auditor_reachable={qwen_ok} "
        f"target_decode={TARGET_DECODE} seed_bases={list(vp.seed_bases)} retry_sleep_cap_s={args.retry_sleep_cap_s}")

    # 5. FREEZE the enumeration per out root (one per seed base), held to the standup's evidence; build the schedule
    petri_opts = ["--manifest", vp.manifest_rel, "--serving-path", "lora_native",
                  "--verify-standup", str(astate / "cells_served.json"), "--verify-routing-table", str(astate / "routing_table.json")]
    # every out root carries (wave_uuid8, attempt_id): a new wave / a re-executed proxy attempt starts EMPTY
    out_roots: Dict[str, Path] = {run_tag(args.view, s): out_root_for(home, w7, wu8, attempt_id, run_tag(args.view, s)) for s in vp.seed_bases}
    # ONE step, replayable per invocation: an import transaction an earlier invocation left OPEN is recovered BEFORE any root is
    # touched (completed / rolled back / refused); then the resume is resolved -- roots that already stand with their import's
    # provenance are ADOPTED as the requested import whether or not a marker was there, else --resume-from imports -- and
    # the adoption receipt is recorded durably in the attempt state dir
    recovery, resumed_from, resume_how = recover_and_resolve_resume(out_roots, resume_from=Path(args.resume_from).expanduser() if args.resume_from else None,
                                                                    w7=w7, wave_uuid=wave_uuid, attempt_id=attempt_id, view=args.view, drive_id=drive_id,
                                                                    astate=astate, log=log)
    cells_json = None
    for tag, out in out_roots.items():
        cells_p = out / "cells.json"
        if cells_p.is_file():
            this = json.loads(cells_p.read_text()) or {}
            if this.get("serving_path") != "lora_native":
                raise DriveRefused(f"{cells_p} was frozen under --serving-path {this.get('serving_path')}; refusing")
        else:
            out.mkdir(parents=True, exist_ok=True)
            with log_p.open("a") as lf:
                r = subprocess.run([py, "scripts/petri_run.py", *petri_opts, "print-cells", "--out-root", str(out), "--target-decode", TARGET_DECODE],
                                   cwd=str(wt), env=env, stdout=lf, stderr=subprocess.STDOUT)
            if r.returncode != 0 or not cells_p.is_file():
                raise DriveRefused(f"print-cells failed for {tag} (see drive.log)")
            this = json.loads(cells_p.read_text())
        if cells_json is None:
            cells_json = this
        elif [row.get("petri_run_key") for row in leg_rows(this)] != [row.get("petri_run_key") for row in leg_rows(cells_json)]:
            raise DriveRefused(f"the enumeration under {out} differs from the first out root's -- one proxy, one enumeration")
    arms, excluded = lane_arm_indices(cells_json, vp)
    for idx, cid, why in excluded:
        log(f"leg {idx} ({cid}) is not one of this proxy's lane arms: {why} -- excluded")
    kept, skipped = served_cells(cells_json, served_models, arms)
    for idx, vm in skipped:
        log(f"leg {idx} ({vm}) is NOT served by this proxy (served_models: {','.join(served_models)}) -- skipped")
    requested = parse_legs_arg(args.legs, kept)
    schedule = petri_schedule(cells_json, served_models, dataclasses.replace(vp, width=width), petri_pass=args.petri_pass, requested=requested)
    epochs = manifest_epochs(py, wt, vp.manifest_rel)

    def reconcile(run: LegRun, cell_dir: Path) -> Tuple[bool, str]:      # the Petri driver's own done check, under the leg python
        return reconcile_cell_store(py, wt, cell_dir, run.cell_id, run.run_key, epochs)

    verdict0 = completion_verdict(schedule, out_roots, astate, reconcile)
    to_run = legs_to_run(schedule, verdict0)
    for name, rk, why in verdict0["detail"]:
        if why.startswith("unreconciled"):
            log(f"{name} ({rk}) has a results.json the store does not bear out -- {why}; re-driven, never skipped")
    log(f"schedule: {len(schedule)} runs ({vp.expected_cells} cells x {len(vp.seed_bases)} seed bases); already complete: "
        f"{len(schedule) - len(to_run)}; to run: {[r.record_name for r in to_run]}")
    drive_rec = {
        "schema": "seedrep_petri_drive/1", "drive_id": drive_id, "substrate": "modal", "lane": LANE, "leg": LEG, "view": args.view,
        "stage": stage, "shard": args.shard, "W": sha, "W7": w7, "wave_uuid": wave_uuid, "launch_plan_sha": launch_plan_sha,
        "sentinel_dict": sentinel_dict, "tunnel_key": tkey, "tunnel": tunnel, "serving_dir_on_volume": serving_vol,
        "attempt_id": attempt_id, "attempt_root_on_volume": attempt_root_vol,
        "local_relay": {"listen": relay["listen"], "upstream": relay["upstream"]}, "proxy_sha": proxy_sha,
        "non_audit_diff_vs_proxy_sha": non_audit_diff(wt, proxy_sha, sha), "auditor": key_stamp, "width": width,
        "effective_widths": effective_widths, "serving_vol_discovery": args.serving_vol or None,
        "teardown_after_failure": bool(args.teardown_after_failure),
        "qwen_auditor_url": None, "qwen_auditor_reachable_at_start": False, "auditor_chain": "sonnet->grok (D111: the Qwen third leg is retired)",
        "target_decode": TARGET_DECODE, "grading_host_bank": grading_host_bank, "schedule": [dataclasses.asdict(r) for r in schedule],
        "skipped_not_served": skipped, "excluded_rows": excluded, "already_complete": [r.record_name for r in schedule if r not in to_run],
        "out_roots": {k: str(v) for k, v in out_roots.items()}, "state_dir": str(astate), "wave_state_dir": str(state),
        "epochs": epochs, "resumed_from": resumed_from, "resume_how": resume_how, "import_recovery": recovery, "verdict_at_start": verdict0,
        "started_at": utc_now(),
    }

    def write_drive_rec() -> None:
        body = json.dumps(drive_rec, indent=1)
        (astate / f"drive_{drive_id}.json").write_text(body)
        for out in out_roots.values():
            (out / "modal_drive.json").write_text(body)

    write_drive_rec()
    if args.dry_run:
        log("DRY RUN: enumeration frozen, schedule built, nothing driven")
        relay["stop"]()
        return 0

    # 6. HEARTBEAT into the sentinel Dict -- armed BEFORE the legs launch (it used to start after the launch loop's waits)
    hb_stop = threading.Event()
    hb_count = [0]
    hb_key = key_for(HEARTBEAT_NAME)

    def heartbeat():
        while True:
            try:
                sentinels[hb_key] = sentinel_content("heartbeat", drive_id).decode()
                hb_count[0] += 1
            except Exception as e:  # noqa: BLE001
                log(f"heartbeat put WARNING: {type(e).__name__}: {e}")
            if hb_stop.wait(HEARTBEAT_PERIOD_S):
                break
    hb_t = threading.Thread(target=heartbeat, name="heartbeat", daemon=True)
    hb_t.start()

    # 7. THE LEGS, concurrently, three attempts each, the width throttled down on a 429-heavy attempt
    def run_leg(run: LegRun) -> None:
        out = out_roots[run.tag]
        clog = astate / f"{run.record_name}.log"
        lock_p = astate / f".{run.record_name}.lock"

        def note(text: str) -> None:
            with clog.open("a") as f:
                f.write(text + "\n")

        with lock_p.open("w") as lk:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                note(f"{run.record_name}: already locked by another drive, skip")
                return
            def launch(w: int) -> Tuple[int, str]:                  # ONE attempt at width w -> (rc, that attempt's log text)
                leg_env = {**env, "PETRI_EVAL_MAX_CONNECTIONS": str(w), "PETRI_OUT_ROOT": str(out.parent), "PETRI_WORK_DIR": str(out),
                           "PETRI_LOG_STORE": str(out / "inspect_logs")}
                mark = clog.stat().st_size
                argv = leg_argv(py, petri_opts, run, out)
                assert_lane_leg_argv(argv, run)
                with clog.open("a") as f:
                    r = subprocess.run(argv, cwd=str(wt), env=leg_env, stdout=f, stderr=subprocess.STDOUT)
                with clog.open("rb") as f:
                    f.seek(mark)
                    return r.returncode, f.read().decode(errors="replace")

            # the same three-attempt + throttle path on EVERY pass; only rc 0 / 3 end it
            rc, attempts_made = leg_attempts(run, launch=launch, proxy_up=lambda: probe_petri_models(base)[1], note=note, log=log)
            drive_rec.setdefault("leg_attempts", {})[run.record_name] = attempts_made
            cell_dir = out / "cells" / run.run_key
            (astate / f"{run.record_name}.rc").write_text(leg_status_line(rc, drive_id, cell_dir / "results.json") + "\n")
            if rc == 0 and (cell_dir / "results.json").is_file():
                log(f"LEG[{run.record_name}] COMPLETE {run.run_key} (drive {drive_id})")
            elif rc in (0, 3):
                log(f"LEG[{run.record_name}] INCOMPLETE {run.run_key} (drive {drive_id})")
            else:
                log(f"LEG[{run.record_name}] FAILED rc={rc} {run.run_key} (drive {drive_id})")

    threads: List[threading.Thread] = []

    def start(run: LegRun) -> threading.Thread:
        t = threading.Thread(target=run_leg, args=(run,), name=run.record_name, daemon=True)
        t.start()
        threads.append(t)
        log(f"launched {run.record_name} (cell {run.cell_id} vmodel {run.vmodel} seed_base {run.seed_base} width {run.width} out_root {out_roots[run.tag]})")
        return t

    launches = launch_legs(to_run, out_roots, start=start, bundle_ready=lambda tag: materialised_bundle_dir(out_roots[tag]).is_dir(), log=log)
    drive_rec["launches"] = launches
    for t in threads:
        t.join()
    hb_stop.set()
    hb_t.join(timeout=60)
    log(f"DRIVE_DONE heartbeats_put={hb_count[0]}")

    # 8. RELEASE the proxy when every leg ENDED and none FAILED
    verdict = completion_verdict(schedule, out_roots, astate, reconcile)
    for name, rk, why in verdict["detail"]:
        if not (why.startswith("complete") or why.startswith("incomplete")):
            log(f"{name} ({rk}) {why} -- held")
    n = verdict["counts"]
    counts = f"complete={n['complete']} incomplete={n['incomplete']} failed={n['failed']}"
    # STOP only when the verdict RELEASES (no FAILED leg), on every pass; a failure holds the proxy unless the orchestrator asked, by
    # name, for --teardown-after-failure (the re-run pass used to STOP on any outcome)
    write_stop, stop_why = release_decision(verdict, teardown_after_failure=args.teardown_after_failure)
    # the completion is published BOUND to the proxy attempt: the tunnel record is re-read NOW and drive_verdict.json +
    # a STOP carrying attempt=<id> go out only while it is still THIS drive's attempt; a missing / stale record publishes nothing (rc 2), a
    # held drive publishes nothing (the consumer's wait would end on the verdict)
    bound = {k: tunnel[k] for k in BINDING_FIELDS}
    completion = publish_completion(sentinels, current_tunnel=read_sentinel_now(sentinels, tkey, log=log), key=key_for(""), bound=bound,
                                    drive_id=drive_id, verdict=verdict,
                                    identity={"W": sha, "W7": w7, "wave_uuid": wave_uuid, "view": args.view, "stage": stage, "shard": args.shard,
                                              "petri_pass": args.petri_pass},
                                    write_stop=write_stop, stop_why=stop_why, legacy_generic=args.legacy_generic_completion, log=log)
    if completion["bundle"]:
        (astate / COMPLETION_NAME).write_text(json.dumps(completion["bundle"], indent=1))
    if completion["stop_content"]:
        where = (f"completion bundle (verdict + STOP attempt={attempt_id}) written to {sentinel_dict} at {completion['completion_key']}"
                 + (f" and mirrored to the legacy generic {VERDICT_NAME} + STOP under {key_for('')}" if args.legacy_generic_completion else ""))
        if stop_why == "release":
            log(f"ALL {verdict['nruns']} RUNS ENDED ({n['complete']} COMPLETE, {n['incomplete']} INCOMPLETE on overflow ledgers, {n['failed']} FAILED) -- {where}")
        else:
            log(f"TEARDOWN_AFTER_FAILURE: {n['failed']} FAILED of {verdict['nruns']} runs ({counts}) -- {where} by explicit --teardown-after-failure; "
                "the failed legs are NOT complete")
    elif not write_stop:
        log(f"runs remain ({n['failed']} FAILED) -- proxy left up (re-run this drive; it resumes the incomplete legs only; "
            "--teardown-after-failure writes STOP regardless)")
    drive_rec.update(finished_at=utc_now(), verdict=verdict, heartbeats_put=hb_count[0], stop_written=bool(completion["stop_content"]),
                     stop_why=stop_why, completion=completion, local_relay_counters=dict(relay["counters"]))
    relay["stop"]()
    write_drive_rec()
    return drive_rc(verdict, completion)


if __name__ == "__main__":
    raise SystemExit(main())
