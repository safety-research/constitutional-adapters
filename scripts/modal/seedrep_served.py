"""seedrep-served: the seed-replicate battery wave on Modal H200s — ONE app for the wave.

EVERY container EXECUTES ONE ROW of `RS/launch_plan.json` (`scripts/modal/seedrep_launch_plan.py`, the one generator):
the row carries the standup argv composed through the real front-end contract (`mt_serving_up.py --serving-path
lora_native` over `lora_serving_up.bring_up_lora`; S derived by the core and lowered to the LoRA kernel ceiling, never
typed), the geometry it must ATTEST, the route set, the object pins, the per-cell adapter-reuse expectations, the driver
children with their registry identity files, the queue producers it feeds and its six-axis control key. The container
adds only what cannot be known at plan time: the ATTEMPT (its id, root, spool), the live proxy URL / routing table, the
concurrency derived from the attested S, and the evidence of the hardware it ran on.

    /vol/seedrep/                                       the lane root (§4.1); never the shared /vol/repos (pruned by peers)
      repos/sisteer-<W>                                 the staged tree (git-verified: HEAD == W, clean)
      objects/sdfbase/full22/qwen35_4b/{cells,cells_view/s1..s3}    the 15 objects + the three views (VIEW_MANIFEST.json)
      objects/arditi_allpos/qwen35_4b/ladder/{cells,cells_view/ar_c0p3}   Scale's ar_c0p3 (read-only the grading host copy)
      objects/STAGED.json                               {sha per staged id, W, utc} — every GPU container re-verifies the 16
      multilora_adapters/qwen35_4b/                     the content-addressed adapter bank (SISTEER_ADAPTER_BANK_ROOT)
      cache/{vllm,flashinfer}                           compile caches (seeded into the container, persisted back)
      <W7>/registry/plans/<leg>/<arm>__s<s>.json        the planned-run identity files (--stage-plans)
      <W7>/exp/seedrep_battery_modal/<leg>/<view|arditi>/<shard>/attempts/<attempt_id>/     ONE execution attempt (§4.6)
      <W7>/exp/seedrep_battery_modal/<leg>/<view|arditi>/rungs/<cap>/<shard>/attempts/<attempt_id>/   a rung attempt
      <W7>/<wave_uuid>/ladder/...                       the ladder's queues / materialised rows (the producers' CLOSED markers)
      <W7>/logs/                                        the container logs

THE ATTEMPT (§4.6): attempt_id = <MODAL_TASK_ID>-<uuid4[:12]>, minted at function entry; its root is IMMUTABLE and never
resumed onto; ATTEMPT.json is written first, COMPLETE.json LAST; the store bytes are shipped by a TreeSyncer every 60 s
(+ at exit) and committed; then in ORDER: COMPLETE.json -> queue_inventory.json -> ONE `accepted.put(key, attempt_id,
skip_if_exists=True)` -> the producer's CLOSED marker. A losing attempt marks itself `superseded_by` and publishes
nothing (its registry spool `<root>/registry/records` is private; `registry_publish.py` copies ACCEPTED attempts only).

Run (the grading host; `-n seedrep-served-<W7>` is REQUIRED; `$STAGE/sdfbase/seedrep_battery/` (the site's staging directory) holds the operator steps):

  M=~/venvs/modal/bin/python; W=<40 hex>; W7=${W:0:7}; RS=~/.local/state/romp/candidate-si-refit/run-state/full22/seedrep_battery
  P=${SISTEER_PYTHON:-.venv/bin/python}; "$P" -c 'import sisteer'   # the repository's interpreter, as the front door picks it (SISTEER_PYTHON, else <repo>/.venv); it must pass before the first step
  $M -m modal run -n seedrep-served-$W7 scripts/modal/seedrep_served.py --sha $W --build
  $M -m modal run -n seedrep-served-$W7 scripts/modal/seedrep_served.py --sha $W --stage --stage-bank --stage-plans $RS/registry/plans
  $M -m modal run -n seedrep-served-$W7 scripts/modal/seedrep_served.py --sha $W --pf --plan $RS/launch_plan_draft.json
  $M -m modal run --detach -n seedrep-served-$W7 scripts/modal/seedrep_served.py --sha $W --plan $RS/launch_plan_draft.json --smoke pasttense__s1__base__shard00 --smoke-id S1 --smoke-max-children 5
  $M -m modal run --detach -n seedrep-served-$W7 scripts/modal/seedrep_served.py --sha $W --plan $RS/launch_plan.json --go <standup_id>
  $M -m modal run --detach -n seedrep-served-$W7 scripts/modal/seedrep_served.py --sha $W --plan $RS/launch_plan.json --legs pasttense,bon
  scripts/modal/seedrep_watch.py --rs $RS --leg pasttense              # the watcher (one per leg; Monitor persistent)
  $M -m modal run -n seedrep-served-$W7 scripts/modal/seedrep_served.py --sha $W --plan $RS/launch_plan.json --pull pasttense --pull-stages all --pull-to $RS/stores
  # the results registry, AFTER certify + pull (the grading host; the attempt spools `<attempt root>/registry/records` ride the pull):
  #   the accepted map = the wave's accepted Dict (`launch_plan.json["dicts"]["accepted"]`) exported as {six-axis key: attempt_id}
  $M -c 'import json,modal,sys; json.dump(dict(modal.Dict.from_name(sys.argv[1]).items()), open(sys.argv[2], "w"), indent=1)' <dicts.accepted> $RS/accepted.json
  for f in $RS/registry/plans/<leg>/*.json; do "$P" scripts/sdfbase/seedrep/logical_run_complete.py --plan $f \
      --accepted $RS/accepted.json --attempts $RS/stores --spool ~/registry/records --verdict-dir $RS/registry/union --launch-plan $RS/launch_plan.json; done
  "$P" scripts/sdfbase/seedrep/registry_publish.py --accepted $RS/accepted.json --attempts $RS/stores --union-dir $RS/registry/union
  #   (ACCEPTED, ADMITTED attempts' spools -> ~/registry/inbox/seedrep-modal/, merged into ~/registry/records by the harvester; --dry-run first)

SMOKE MODE (Astra P2-14): `--smoke <standup_id> --smoke-id <S1|S1b|S2|S3|S4|S5> [--smoke-max-children 5] [--smoke-timeout-h 4]` runs ONE
DRAFT row under an ISOLATED smoke wave (its own uuid -> its own Dicts, ladder root and results root `..._smoke/<id>`), bounded to the
first N children and the timeout; production fan-out (`--go` / `--legs`) admits the FINAL plan only.
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
import faulthandler
import fcntl
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import uuid
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import modal

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def grading_host_repo_src(here: Path = _HERE) -> Optional[Path]:
    """`<repo>/src` when THIS file sits in a `<repo>/scripts/modal/` layout (the grading host's launcher; the same layout rule
    `lp.registry_plans_module` applies), else None (a container: the file is copied beside the plan module, no repo). The root cause
    on the grading host's side: the launcher's `_registry_queued` imports `sisteer.registry.hooks` (stdlib-only) to register every
    spawn; under `~/venvs/modal` nothing had put `<repo>/src` on `sys.path`, the ImportError was returned on silently, and NO spawn of
    any wave was ever registered (zero seedrep runs in the registry after S1-S5)."""
    try:
        src = (here.parents[1] / "src") if (here.name == "modal" and here.parent.name == "scripts") else None
    except IndexError:            # the image installs this module FLAT (/root/seedrep_served.py): no parents[1], no repo
        return None
    if src is not None and (src / "sisteer" / "registry" / "hooks.py").is_file():
        return src
    return None


_REPO_SRC = grading_host_repo_src()
if _REPO_SRC is not None and str(_REPO_SRC) not in sys.path:
    sys.path.append(str(_REPO_SRC))        # APPEND: an installed / PYTHONPATH sisteer keeps precedence; the modal venv carries none
import seedrep_launch_plan as lp  # noqa: E402  (the generator's constants + composers; shipped into the image beside this file)

DEFAULT_APP_NAME = "seedrep-served"
app = modal.App(DEFAULT_APP_NAME)
VOL_NAME = lp.MODAL_VOLUME_NAME                                # ONE spelling: the consumer argv's --volume names the same Volume
VOL = modal.Volume.from_name(VOL_NAME, create_if_missing=False)
VOL_PATH = lp.VOL_PATH
LANE, REPOS, OBJECTS_VOL, ADAPTER_BANK_ROOT, CACHE_VOL, HF_VOL = lp.LANE, lp.REPOS, lp.OBJECTS_VOL, lp.ADAPTER_BANK_ROOT, lp.CACHE_VOL, lp.HF_VOL
BANK_VOL = f"{OBJECTS_VOL}/sdfbase/full22"                    # <bank>/<target>/{cells,cells_view/<view>}
ARDITI_BANK_VOL = f"{OBJECTS_VOL}/arditi_allpos/{lp.MODEL_KEY}/ladder"
STAGED_JSON = f"{OBJECTS_VOL}/STAGED.json"
CORPORA_VOL = f"{VOL_PATH}/heldout_corpora"
#: the frozen replay corpora as the DRIVERS see them -- `_build_shared` symlinks `<SHARED_LOCAL>/exp/heldout_corpora`
#: -> CORPORA_VOL, and every child gets `SISTEER_HELDOUT_CORPUS_DIR` pointing there (renellm reads `<dir>/renellm.jsonl`;
#: target_static.sbatch exports the same variable from `$SHARED/exp/heldout_corpora` and f22_fatals without it -- the app used to
#: bypass the sbatch, so renellm fell to `$HOME/.local/state/...` under HOME=/root, loud_skip'd, and every child exited 0 having
#: written nothing)
HELDOUT_CORPUS_ENV = "SISTEER_HELDOUT_CORPUS_DIR"
#: `datasets` reads a PREPARED arrow cache under HF_DATASETS_CACHE when offline -- the hub snapshot alone does not serve it
#: (g8k_common.sh g8k_init / g8k_datasets_guard); the cluster puts it at $SHARED/hf_datasets, the lane under /vol/hf
HF_DATASETS_VOL = f"{HF_VOL}/datasets"
HELDOUT_CORPUS_LOCAL = f"{lp.SHARED_LOCAL}/exp/heldout_corpora"
CONTAINER_USER = lp.CONTAINER_USER
SHARED_LOCAL = lp.SHARED_LOCAL
VENVS_LOCAL = lp.VENVS_LOCAL
UV_CACHE = "/opt/uv-cache"
UV_PYTHONS = "/opt/uv-pythons"
UV_PIN = "0.12.1"
PY_VERSION = "3.12"
VLLM_REQS_REL = lp.VLLM_REQS_REL
DEPENDENCY_FILES = ("pyproject.toml", "uv.lock", VLLM_REQS_REL)
COMPILE_CACHES = {"vllm": "/root/.cache/vllm", "flashinfer": "/root/.cache/flashinfer"}
EVIDENCE_NAME = "modal_evidence.json"
ATTEMPT_JSON = "ATTEMPT.json"
COMPLETE_JSON = "COMPLETE.json"
INVENTORY_JSON = "queue_inventory.json"
LADDER_ENQUEUED_JSON = "ladder_enqueued.json"                 # the tailer's EXPLICIT inventory; REQUIRED on a producer row
RESULT_FILE = "RESULT"
SERVE_DIR = lp.SERVE_DIR_REL                                  # exp/serve (under the exp tree the drive maps)
#: the attempt root's files everything downstream reads (§4.6): the stores under exp/ EXACTLY as the drivers write them
ATTEMPT_ROOT_LAYOUT: Tuple[str, ...] = (
    ATTEMPT_JSON, COMPLETE_JSON, INVENTORY_JSON, LADDER_ENQUEUED_JSON, RESULT_FILE, EVIDENCE_NAME,
    "usage_ledger.jsonl", "request_evidence.jsonl", "gpu_dmon.log", "vllm_metrics.jsonl", "driver_inflight.jsonl",
    "determinism.json", f"{SERVE_DIR}/endpoint.json", f"{SERVE_DIR}/cells_served.json", f"{SERVE_DIR}/serving_attestation.json",
    f"{SERVE_DIR}/routing_table.json", f"{SERVE_DIR}/standup_jobs.jsonl", f"{SERVE_DIR}/vllm_server.log.gz",
    f"{SERVE_DIR}/cells.json", f"{SERVE_DIR}/registry_runs.json", "exp/", f"{lp.SPOOL_REL}/", "children/",
)
#: the container-local attempt root and its exp tree (what the TreeSyncer ships to `<attempt root on volume>/`)
CONTAINER_ATTEMPT_ROOT = "/workspace/attempt"
CONTAINER_EXP_ROOT = f"{CONTAINER_ATTEMPT_ROOT}/exp"
#: the durable per-result streams whose ROWS are the attempt's progress (§7.4) -- counted under <root>/exp
PROGRESS_STREAM_NAMES: Tuple[str, ...] = ("responses.jsonl", "controls_events.jsonl", "progress.jsonl")
#: the Petri drive's completion evidence: a STOP sentinel bound to THIS attempt, or the verdict record
PETRI_VERDICT_NAME = "drive_verdict.json"
PETRI_STALE_SENTINELS = ("STOP", "driver_heartbeat", "modal_tunnel.json", PETRI_VERDICT_NAME)
TAILER_POLL_S = 15
#: the belt: the post-RESULT tail (drain -> seal -> inventory -> acceptance) runs under a bounded deadline; past it the row logs
#: POST_RESULT_HANG with a thread dump, records the hang under modal_evidence.json["post_result"] and RETURNS (the container exits;
#: the RESULT already written is never changed). Sized from the batched drain's real cost -- at most one driver-venv helper per
#: (stream, control) plus one `si-text` per prompted arm, ~10-20 s each cold, 8 streams -> a few minutes -- an order of magnitude
#: above that and ~30x under the production container timeout (16 h) that the controls rows would otherwise have idled to.
POST_RESULT_EXIT_S = 1800
POST_RESULT_EXIT_CTX = "post_result_exit_s"                  # spawn_ctx override (a test's short deadline)
POST_RESULT_HANG = "POST_RESULT_HANG"
#: the EXPLICIT terminal status of an attempt whose post-RESULT tail timed out BEFORE its acceptance election --
#: rides in the returned dict / the watcher's snapshot summary / modal_evidence.json["post_result"]["terminal_status"] (never RESULT,
#: which records the children's verdict); `call_failure`, the watcher's FAILED event and `steward_tick` classify it as a failure, and
#: `reattempt_decision` as INFRA (`lp.D73_INFRA_STATUSES`: <= 1 automatic re-attempt, then HOLD).
POST_RESULT_TIMEOUT = "POST_RESULT_TIMEOUT"
MISALIGN_MIN_ZLIB = 0.2                                       # runner.RERUN_MIN_ZLIB_RATIO: below it a capped response is a loop, not a rung item
STATE_DIR = Path.home() / ".local" / "state" / "seedrep-modal"
#: the battery's run-state root on the grading host: the full22 run-state tree's seedrep_battery/, under the state root ($STATE),
#: expanded where the local entrypoint starts (:func:`site_run_state`)
RS_DEFAULT = "$STATE/romp/candidate-si-refit/run-state/full22/seedrep_battery"


def site_run_state(value: str) -> str:
    """``--rs`` with the site's ``$STATE`` (the state root) expanded through scripts/site_paths.py, refused by name when it is unset;
    a path without a placeholder comes back with ``~`` expanded."""
    if "$" not in str(value):
        return str(Path(value).expanduser())
    scripts = str(Path(__file__).resolve().parent.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from site_paths import site_path
    return str(site_path(value, "--rs"))
PETRI_TUNNEL_FILE = "modal_tunnel.json"
PETRI_SENTINELS = ("STOP", "driver_heartbeat", PETRI_VERDICT_NAME)
RELAY_OFFSET = 10000
HEARTBEAT_S = 60
SYNC_EVERY_S = 60
DMON_CADENCE_S = 10
STAGED_IDS_N = 16                                             # 15 lane objects + the Arditi carrier

try:
    LOCAL_REPO = Path(__file__).resolve().parents[2]
except IndexError:  # Modal re-imports this module flat at /root/seedrep_served.py
    LOCAL_REPO = Path("/repo")

if modal.is_local():
    from cpu_suite import shallow_clone_tarball  # noqa: E402  (the cpu_suite staging helper, never re-implemented)


# --------------------------------------------------------------------------- #
# pure helpers (tested on the CPU under a stub `modal`)                         #
# --------------------------------------------------------------------------- #

def sha256_file(path) -> str:
    return lp.sha256_file(Path(path))


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def volume_client_path(path: str) -> str:
    """``/vol/...`` -> the Volume client's ``/...``."""
    if path == VOL_PATH:
        return "/"
    if not path.startswith(VOL_PATH + "/"):
        raise ValueError(f"path is outside {VOL_PATH}: {path}")
    return path[len(VOL_PATH):]


def expected_app_name(sha: str) -> str:
    return f"seedrep-served-{sha[:7]}"


def dependency_hashes(tree) -> dict:
    return {name: sha256_file(Path(tree) / name) for name in DEPENDENCY_FILES}


def serving_venv_names(reqs_sha256: str) -> Tuple[str, str, str]:
    """common.sh f22_serving_venv: venv, sentinel, sentinel value (strength's image builds it at these names)."""
    venv = f"{VENVS_LOCAL}/sisteer-petri-vllm-{reqs_sha256[:8]}"
    return venv, f"{VENVS_LOCAL}/.sisteer-petri-vllm-{reqs_sha256[:8]}.vllm-bootstrap", reqs_sha256[:16]


def driver_venv_names(W: str) -> Tuple[str, str]:
    """common.sh f22_driver_venv: ``/home/$USER/venvs/f22-<sha12>`` + its ``.synced`` sentinel holding the sha."""
    venv = f"{VENVS_LOCAL}/f22-{W[:12]}"
    return venv, f"{VENVS_LOCAL}/.f22-{W[:12]}.synced"


def mint_attempt_id(task_id: Optional[str], nonce: Optional[str] = None) -> str:
    """``<MODAL_TASK_ID>-<uuid4[:12]>`` (§4.6): two executions of the SAME call and two spawns of the same row always get
    DISTINCT ids. A missing task id (a local test) reads ``local``."""
    tid = (task_id or "").replace("ta-", "") or "local"
    return f"{tid}-{(nonce or uuid.uuid4().hex)[:12]}"


def results_root(W7: str) -> str:
    return f"{LANE}/{W7}/exp/{lp.RESULTS_ROOT_NAME}"


def attempt_root_for(root: str, row: Mapping[str, Any], attempt_id: str) -> str:
    """The attempt's IMMUTABLE root: base rows at ``<root>/<leg>/<view>/<shard>/attempts/<attempt_id>`` (§4.6 verbatim);
    rung rows at ``<root>/<leg>/<view>/rungs/<cap>/<shard>/attempts/<attempt_id>`` -- ONE spelling with the watcher's (`lp.attempt_root_for`)."""
    return lp.attempt_root_for(root, leg=row["leg"], view=row["engine"], stage=row["stage"], shard=row["shard"], attempt_id=attempt_id, kind=row.get("kind"))


def venv_env(env: Mapping[str, str], python: str) -> Dict[str, str]:
    """The environment for a subprocess launched with a VENV's python by absolute path: that venv's ``bin`` FIRST on ``PATH`` and
    ``VIRTUAL_ENV`` set, so the tools the script shells out to by name (``vllm`` from the serving venv, ``inspect``/``uv`` from the
    client and driver venvs) resolve (the S1 standup died ``'vllm' not on PATH`` because ``container_env`` dropped
    ``VIRTUAL_ENV`` and never added the venv's bin — a python invoked by absolute path does not put its bin/ on PATH)."""
    out = dict(env)
    bin_dir = str(Path(python).absolute().parent)        # the INVOKED path: a venv python is a symlink to the base interpreter -- never dereference
    prior = [p for p in out.get("PATH", "").split(":") if p and p != bin_dir]
    out["PATH"] = ":".join([bin_dir, *prior])
    out["VIRTUAL_ENV"] = str(Path(bin_dir).parent)
    return out


# --------------------------------------------------------------------------- #
# the lane's attacker-pool bearer in the DRIVER containers                      #
# --------------------------------------------------------------------------- #

#: == sisteer.eval.attacks.attacker_pool.POOL_BEARER_ENV (this module cannot import sisteer inside its container -- only
#: seedrep_launch_plan.py ships beside it -- so it spells the literal; tests/test_seedrep_fw_pool_bearer.py pins the equality).
#: Distinct from $VLLM_API_KEY, which in a driver container is the TARGET proxy's bearer (when any): multi-turn children that
#: sent that one to the pools got 1296 x 401, zero target forwards, rc 0.
POOL_BEARER_ENV = "SEEDREP_ATTACKER_POOL_KEY"
#: the pool app's spelling (`seedrep_attacker_pool._runtime_key_dir`): $XDG_RUNTIME_DIR / these / <lane> / api_key
POOL_BEARER_REL = ("seedrep-battery", "attacker-pool")
ATTACKER_ENDPOINT_FLAG = "--attacker-endpoint"
#: the lane NAME the pool app renders its bearer under (== `seedrep_attacker_pool.LANE`; spelled here the way POOL_BEARER_ENV is, that
#: module being un-importable in-container; tests/test_seedrep_fy_w19_app_side.py pins the equality). NOT `LANE` above: that is the
#: lane's VOLUME PATH (`lp.LANE` == /vol/seedrep). Joining the path here let pathlib collapse
#: $XDG_RUNTIME_DIR/seedrep-battery/attacker-pool / "/vol/seedrep" / api_key onto /vol/seedrep/api_key, and the spawn gate refused
#: every pool-pinned row for a bearer that sat, 0600, exactly where the pool app had rendered it.
POOL_LANE = "seedrep"


def local_pool_bearer_path(lane: str = POOL_LANE, environ: Mapping[str, str] = os.environ) -> Optional[Path]:
    """``$XDG_RUNTIME_DIR/seedrep-battery/attacker-pool/<lane>/api_key`` (the pool app's `_local_key_path`), None with the runtime dir
    unset. ``lane`` is a bare NAME: a path-like lane (a `/` in it, or `.`/`..`) REFUSES by name -- joined into the runtime-dir path it
    silently relocates the key, and a refusal here is the only way that class cannot recur quietly."""
    if not lane or "/" in lane or os.sep in lane or lane in (".", ".."):
        raise ValueError(f"the pool lane must be the pool app's bare lane NAME ({POOL_LANE!r}), never a path: got {lane!r} -- joining a path "
                         "into the runtime-dir key path collapses it onto that path (W19 F-AB, D102)")
    rd = environ.get("XDG_RUNTIME_DIR")
    return (Path(rd) / POOL_BEARER_REL[0] / POOL_BEARER_REL[1] / lane / "api_key") if rd else None


def read_local_pool_bearer(lane: str = POOL_LANE, environ: Mapping[str, str] = os.environ) -> Tuple[str, str]:
    """READ-ONLY: ``(bearer, problem)``. The pool app's self-minted lane bearer from the tmpfs runtime dir -- never
    minted here (a missing key means the pool app's `--action endpoint` has not run in this session; the spawn gate refuses by
    name when a multi-turn row needs it), never printed. ``problem`` says why the bearer is "" (runtime dir unset, file absent /
    not under the runtime dir once resolved / not private / empty), so the refusal can say it."""
    p = local_pool_bearer_path(lane, environ)
    if p is None:
        return "", "XDG_RUNTIME_DIR is unset (D77: the bearer is rendered ONLY under the tmpfs runtime dir)"
    if not p.is_file():
        return "", f"no bearer file at {p} (the pool app's --action endpoint renders it)"
    try:
        p.resolve().relative_to(Path(environ["XDG_RUNTIME_DIR"]).resolve())
    except ValueError:
        return "", f"bearer file {p} resolves outside the runtime dir"
    mode = p.stat().st_mode & 0o777
    if mode & 0o077:
        return "", f"bearer file {p} mode {oct(mode)} is not private"
    v = p.read_text().strip()
    if not v:
        return "", f"bearer file {p} is empty"
    return v, ""


def pool_bearer_secrets(bearer: str) -> list:
    """The Modal secret that carries the lane bearer into the GPU functions' containers as ``$POOL_BEARER_ENV`` -- the SAME
    transport the pool app uses for its own $VLLM_API_KEY (`seedrep_attacker_pool.pool_secret`). An empty bearer builds an EMPTY
    secret (no env var at all), so a container spawned without one fails its pool children by name
    (`child_env_with_pool_bearer`) instead of exporting "". A stub `modal` without `Secret` (the CPU tests) attaches nothing."""
    secret = getattr(modal, "Secret", None)
    if secret is None:
        return []
    return [secret.from_dict({POOL_BEARER_ENV: bearer} if bearer else {})]


if modal.is_local():
    POOL_BEARER, POOL_BEARER_PROBLEM = read_local_pool_bearer()
else:
    POOL_BEARER, POOL_BEARER_PROBLEM = os.environ.get(POOL_BEARER_ENV, ""), ""
POOL_BEARER_SECRETS = pool_bearer_secrets(POOL_BEARER)


def pool_accepts_bearer(url: str, bearer: str, *, timeout: float = 10.0) -> bool:
    """``GET <url>/v1/models`` WITH the bearer -> 200: the pool app's own authenticated probe (`endpoint_health`), run at spawn
    time so a bearer the pools reject refuses the spawn (naming the pool, never the bearer) instead of 401-ing every driver call."""
    # through the app's ONE HTTP read (`_http_get_text`, F-O): a non-2xx raises HTTPError, a transport failure URLError -- both "rejects"
    try:
        _http_get_text(url.rstrip("/") + "/v1/models", timeout=timeout, headers={"Authorization": f"Bearer {bearer}"})
        return True
    except Exception:  # noqa: BLE001 - a 401/403 or a down pool is "rejects"; the health probe before this one names a down pool
        return False


def child_env_with_pool_bearer(argv: Sequence[str], env: Mapping[str, str], container_environ: Mapping[str, str]) -> Dict[str, str]:
    """A driver child's FINAL env: the lane pool bearer copied from the container's environ (the Modal secret) into the child's
    env ONLY when its argv carries `--attacker-endpoint` (a multi-turn driver on the deployed pool); every other child gets the
    env WITHOUT it (`container_env` scrubbed the name from the base). A pool child in a container that carries no bearer is a
    LookupError -- the spawn gate should have refused; the row fails by name rather than launch a child destined for 401s."""
    out = dict(env)
    if not lp._flag_value(list(argv), ATTACKER_ENDPOINT_FLAG):
        out.pop(POOL_BEARER_ENV, None)
        return out
    bearer = container_environ.get(POOL_BEARER_ENV, "")
    if not bearer:
        raise LookupError(f"this container carries no ${POOL_BEARER_ENV} (the lane attacker-pool bearer, a Modal secret read from the grading host's "
                          f"$XDG_RUNTIME_DIR at spawn) but the child's argv names {ATTACKER_ENDPOINT_FLAG}: it would reach the pool with no "
                          "bearer and every goal would 401 (D99)")
    out[POOL_BEARER_ENV] = bearer
    return out


def container_env(*, bank_view: str, tree: str, thinking: bool, petri: bool = False,
                  base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The client-side env of every GPU function (§4.2): the inherited env SCRUBBED of the five keys strength scrubs,
    then the lane's values. ``HF_HOME`` / ``HF_HUB_OFFLINE`` are set for the SERVING venv (the engine reads its weights
    off /vol/hf); the drivers see the same env plus their per-child registry wiring (`child_env`)."""
    env = dict(os.environ if base is None else base)
    for k in ("HF_HOME", "HF_HUB_OFFLINE", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV", "PYTHONPATH"):
        env.pop(k, None)
    env.pop(POOL_BEARER_ENV, None)      # The lane pool bearer reaches ONLY the --attacker-endpoint children (child_env_with_pool_bearer)
    env.update({"CUDA_VISIBLE_DEVICES": "0", "HOME": "/root", "USER": CONTAINER_USER, "LOGNAME": CONTAINER_USER,
                lp.__dict__.get("ADAPTER_BANK_ROOT_ENV", "SISTEER_ADAPTER_BANK_ROOT"): ADAPTER_BANK_ROOT,
                "TT_TWO_SI_DEFENSES_DIR": bank_view, "F22_LOG_REQUESTS": "1",
                "F22_SERVER_DEFAULT_POLICY": lp.SERVER_DEFAULT_POLICY, "HF_HOME": HF_VOL, "HF_HUB_OFFLINE": "1",
                "PYTHONPATH": f"{tree}/src:{tree}/scripts", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
                HELDOUT_CORPUS_ENV: HELDOUT_CORPUS_LOCAL,                 # renellm's frozen corpus, as the cluster sbatch exports it
                "HF_DATASETS_CACHE": HF_DATASETS_VOL, "HF_DATASETS_OFFLINE": "1",   # the PREPARED `datasets` cache the offline readers need (g8k_common.sh)
                "F22_THINKING_ON": "1" if thinking else ""})
    if not thinking:
        env.pop("F22_THINKING_ON", None)
    if petri:
        env["PETRI_EVAL_MAX_CONNECTIONS"] = env.get("PETRI_EVAL_MAX_CONNECTIONS", "3")
    return env


concurrency_for = lp.concurrency_for                          # ONE rule, in the generator


def child_argv_env(child: Mapping[str, Any], *, attempt_root: str, plans_root: str, proxy_url: str, routing_table: str,
                   driver_python: str, S: int, ncells: int, bank_view: str, base_env: Mapping[str, str],
                   pool_urls: Optional[Mapping[str, str]] = None, attempt_id: str = "", standup_id: str = "", call_id: str = "",
                   attempt_root_rel: str = "", child_subs: Optional[Mapping[str, Mapping[str, str]]] = None, client_python: str = "") -> Tuple[List[str], Dict[str, str]]:
    """ONE driver child -> the FINAL argv + env: `lp.resolve_spawn` over a `lp.SpawnContext` (the same function the container's
    spawn loop calls for every child; kept as the keyword entry point the tests and the app share). A GSM8K child needs its per-child ledger
    substitutions (`child_subs[lp.child_key(child)]`, what `gsm8k_child_up` returns) or it refuses."""
    ctx = lp.SpawnContext(attempt_root=str(attempt_root), attempt_id=attempt_id, standup_id=standup_id, plans_root=plans_root, proxy_url=proxy_url,
                          routing_table=routing_table, driver_python=driver_python, S=int(S), ncells=int(ncells), bank_view=bank_view, base_env=base_env,
                          pool_urls=pool_urls, call_id=call_id, attempt_root_rel=attempt_root_rel, child_subs=child_subs, client_python=client_python)
    return lp.resolve_spawn(child, ctx)


#: the readiness contract reads the REAL native attestation (pinned by tests/fixtures/seedrep_f_l -- the failed S1's bytes),
#: not the double it was written against. (a) vLLM 0.25.1 reports `cudagraph_mode: FULL_AND_PIECEWISE`: FULL decode graphs for every capture
#: size <= max_num_seqs (`attested_cudagraph_full_capture_sizes`, max = S), piecewise beyond, `cudagraph_specialize_lora`. The property that
#: matters is the FULL capture surface -- max == S and S among the FULL sizes; a mode with no FULL surface (PIECEWISE alone, NONE, eager)
#: refuses. The literal `FULL` failed every real standup (S1, S1b, S2, S3). (b) the residual slot is a TOP-LEVEL block
#: (`serving_attestation.RESIDUAL_SLOT_KEY`: armed, rpc_method, n_workers, expected{adapters{<harness id>: kind, op, layers, boundaries,
#: position_mode, residual_fingerprint, lora_int_id}}, worker_reports), never a key inside `adapters[<id>]`. (c) the object-digest map the
#: standup asserted is written under either name from ONE dict (`native_standup`: `expected_object_sha256_by_cell` and
#: `object_artifact_sha256_by_cell`); the lane's endpoint.json carries the latter (`expected_digests_source: cli`).
CUDAGRAPH_MODES_WITH_FULL_CAPTURE: Tuple[str, ...] = lp.CUDAGRAPH_MODES_WITH_FULL_CAPTURE     # ONE definition (the plan module); the reference admission shares it
RESIDUAL_SLOT_KEY = "residual_slot"
#: endpoint.json carries the slot as a SUMMARY -- `{armed, cells: {<harness id>: {kind, position_mode, residual_fingerprint}}}` (the cells the
#: standup served THROUGH the slot); serving_attestation.json beside it carries the full block (`expected.adapters[<id>]` with kind / op /
#: layers / boundaries / residual_fingerprint, `worker_reports`, `n_workers`, `rpc_method`). The contract reads each file's own shape.
RESIDUAL_CELL_PAYLOAD_KEYS: Tuple[str, ...] = ("kind", "residual_fingerprint")
RESIDUAL_ATTESTED_PAYLOAD_KEYS: Tuple[str, ...] = ("kind", "op", "residual_fingerprint")
#: (c) the writer: cells_served.json carries `expected_object_sha256_by_cell` (the pins the standup was TOLD, `expected_digests_source: cli`),
#: endpoint.json carries `object_artifact_sha256_by_cell` (the artifacts it LOADED); the contract reads the first where it lives and holds the
#: second equal to it -- an earlier reader took `expected_object_sha256_by_cell` off the endpoint, where it never was.
EXPECTED_DIGESTS_KEY = "expected_object_sha256_by_cell"          # cells_served.json
ARTIFACT_DIGESTS_KEY = "object_artifact_sha256_by_cell"          # endpoint.json
#: L1: the readiness checks that speak about the SERVED GEOMETRY (a mismatch there is GEOMETRY_MISMATCH); every other refusal is a contract
#: mismatch between the row's expectations and the attestation (READINESS_CONTRACT) -- the old "attested" substring filed a cudagraph or
#: adapter failure as a geometry failure, so the RESULT line named the wrong cause
GEOMETRY_PROBLEM_PREFIXES: Tuple[str, ...] = ("attested_max_model_len ", "attested_max_num_seqs ", "kernel adjustment ", "unexpected adjustments ",
                                              "geometry.window_source ")
READINESS_CONTRACT = "READINESS_CONTRACT"


def readiness_failure_status(problems: Sequence[str]) -> str:
    """GEOMETRY_MISMATCH when any problem is about the served geometry itself; READINESS_CONTRACT otherwise (L1)."""
    return "GEOMETRY_MISMATCH" if any(p.startswith(GEOMETRY_PROBLEM_PREFIXES) for p in problems) else READINESS_CONTRACT


def readiness_problems(endpoint: Mapping[str, Any], cells_served: Mapping[str, Any], row: Mapping[str, Any],
                       attestation: Optional[Mapping[str, Any]] = None) -> List[str]:
    """Every assertion of §4.3 on the READY records, against the row's expectations; ``[]`` = the engine is the one the
    plan describes. A mismatch is GEOMETRY_MISMATCH / ADAPTER_NOT_REUSED for the container (fatal, teardown).

    TWO NAMESPACES MEET HERE: the row speaks PLAN ids (route_set, objects, adapters, the PSR cells); the
    native records speak HARNESS ids (`cells_served` canonical + controls rows, `expected_object_sha256_by_cell`,
    `adapters_reused`, `adapters`), and list the DERIVED zero-adapter control `none__lora0` beside the route set. Every
    comparison converts at this boundary through `lp.harness_id`; the plan's expectation is `row["cells_served_expected"]`."""
    want, geom = row["readiness"], row["geometry"]
    out: List[str] = []
    eng = endpoint.get("engine")
    eng = eng.get("kind") if isinstance(eng, dict) else eng
    if (eng or endpoint.get("serving_engine")) != "lora_native":
        out.append(f"engine {eng!r} != lora_native")
    att_L, att_S = endpoint.get("attested_max_model_len"), endpoint.get("attested_max_num_seqs")
    if att_L != want["attested_max_model_len"]:
        out.append(f"attested_max_model_len {att_L} != {want['attested_max_model_len']}")
    if att_S != want["attested_max_num_seqs"]:
        out.append(f"attested_max_num_seqs {att_S} != {want['attested_max_num_seqs']} (= min(identity, 524,287 // L))")
    g = endpoint.get("geometry") or {}
    adjustments = g.get("adjustments") or []
    exp_adj = want.get("adjustment")
    if exp_adj is not None:
        kernel = [a for a in adjustments if isinstance(a, Mapping) and a.get("kind") == exp_adj["kind"]]
        if not kernel or int(kernel[-1].get("to_S", -1)) != int(exp_adj["to_S"]) or int(kernel[-1].get("from_S", -1)) != int(exp_adj["from_S"]):
            out.append(f"kernel adjustment {kernel!r} != expected {exp_adj}")
    elif adjustments:
        out.append(f"unexpected adjustments {adjustments!r} (identity S already within the kernel ceiling)")
    if g.get("window_source") != geom["expected_window_source"]:
        out.append(f"geometry.window_source {g.get('window_source')!r} != {geom['expected_window_source']!r} (main #200: a table "
                   "window keeps `table` through the lowering; a non-table window reads `override`)")
    # the engine's REAL mode, and the PROPERTY -- FULL decode graphs captured exactly up to the served S
    S_want = int(want["capture_surface_ge"])
    # every attestation the row MANDATES is REQUIRED with exact contents -- an absent field used to pass because each
    # check ran only when the field was present (fail-open); absence is now a readiness failure, named per field
    # the core's own rule (serving_attestation.cudagraph_coverage_problems): the FULL surface's max covers S, and the sizes' max IS that max
    # the mode / capture-max predicate is the plan module's (`lp.cudagraph_full_capture_problems`), shared with the reference admission
    cap_max = endpoint.get("attested_cudagraph_full_capture_max")
    sizes = endpoint.get("attested_cudagraph_full_capture_sizes")
    out += lp.cudagraph_full_capture_problems(attested_mode=endpoint.get("attested_cudagraph_mode"), capture_max=cap_max, S=S_want)
    if not isinstance(sizes, (list, tuple)) or not sizes:
        out.append(f"attested_cudagraph_full_capture_sizes ABSENT or empty (a mandated attestation: the FULL capture sizes whose max is the surface)")
    elif cap_max is not None and max(int(x) for x in sizes) != int(cap_max):
        out.append(f"cudagraph FULL capture sizes max {max(int(x) for x in sizes)} != full_capture_max {cap_max} (surface drift: the dispatcher's keys and the max disagree)")
    lc = endpoint.get("lora_config") or {}
    if not lc:
        out.append(f"lora_config ABSENT (a mandated attestation: rank {want['max_lora_rank']} / max_loras {want['max_loras']})")
    elif int(lc.get("max_lora_rank", -1)) != int(want["max_lora_rank"]) or int(lc.get("max_loras", -1)) != int(want["max_loras"]):
        out.append(f"lora_config {lc} != rank {want['max_lora_rank']} / max_loras {want['max_loras']}")
    if want.get("max_logprobs") is not None and endpoint.get("max_logprobs") != want["max_logprobs"]:
        out.append(f"max_logprobs {endpoint.get('max_logprobs')!r} != {want['max_logprobs']} (MMLU top_logprobs 64 would be refused)")
    if str(endpoint.get("thinking_mode") or "off") != want["thinking_mode"]:
        out.append(f"thinking_mode {endpoint.get('thinking_mode')!r} != {want['thinking_mode']}")
    sdp = endpoint.get("server_default_policy")
    if sdp is None:
        out.append(f"server_default_policy ABSENT (a mandated attestation: {want['server_default_policy']})")
    elif sdp != want["server_default_policy"]:
        out.append(f"server_default_policy {sdp!r} != {want['server_default_policy']}")
    # --- the served cells, in the record's own (harness) namespace: the route set translated + the derived control -----
    expect = row.get("cells_served_expected") or lp.expected_cells_served(row["engine"])
    canonical = [c.get("cell_id") for c in (cells_served.get("canonical") or [])]
    controls = [c.get("cell_id") for c in (cells_served.get("controls") or [])]
    if sorted(canonical) != sorted(expect["canonical"]):
        out.append(f"cells_served.canonical {sorted(canonical)} != the route set in harness ids {sorted(expect['canonical'])}")
    if sorted(controls) != sorted(expect["controls"]):
        out.append(f"cells_served.controls {sorted(controls)} != {sorted(expect['controls'])} (the derived zero-adapter control is served on every lora_native engine)")
    bad_class = [c.get("cell_id") for c in (cells_served.get("controls") or []) if c.get("cell_class") not in (None, "control")]
    if bad_class:
        out.append(f"cells_served.controls rows {bad_class} are not cell_class control")
    reused = endpoint.get("adapters_reused")
    if reused is None:
        out.append("adapters_reused ABSENT (a mandated attestation: the adapter-reuse map, gate 10)")
    else:
        for cell, a in row["adapters"].items():
            h = cell if cell == lp.NONE_LORA0 else lp.harness_id(cell)
            if not a.get("expected_reused"):
                continue
            if h not in reused:
                out.append(f"adapters_reused has no entry for {cell} ({h}) -- an expected re-use that was never attested (gate 10)")
            elif reused[h] is not True:
                out.append(f"adapter for {cell} ({h}) was EXPORTED (reused={reused[h]!r}), not re-used — a race or a drift (gate 10)")
    # the residual slot is a TOP-LEVEL block, never a key inside `adapters[<id>]`: endpoint.json's SUMMARY names the cells served
    # through the slot with their payload (kind, fingerprint); the attestation beside it, when given, adds the workers' own reports and the
    # per-adapter expected payload (kind, op, fingerprint). A PSR / vector cell missing from either, or carrying a null payload, is a plain LoRA.
    armed_cells = list(want.get("residual_slot_armed") or [])
    if armed_cells:
        rs = endpoint.get(RESIDUAL_SLOT_KEY)
        if not isinstance(rs, Mapping):
            out.append(f"{RESIDUAL_SLOT_KEY} ABSENT (a mandated TOP-LEVEL attestation block for a PSR / vector cell: serving_attestation.RESIDUAL_SLOT_KEY)")
        else:
            if rs.get("armed") is not True:
                out.append(f"{RESIDUAL_SLOT_KEY}.armed {rs.get('armed')!r} != True (the residual op is not armed on the workers)")
            served_cells = rs.get("cells") if isinstance(rs.get("cells"), Mapping) else {}
            for cell in armed_cells:
                h = lp.harness_id(cell)
                e = served_cells.get(h)
                if not isinstance(e, Mapping):
                    out.append(f"{RESIDUAL_SLOT_KEY}.cells has no entry for {cell} ({h}) (a mandated attestation for a PSR / vector cell: the cells served THROUGH the slot)")
                elif any(not e.get(k) for k in RESIDUAL_CELL_PAYLOAD_KEYS):
                    out.append(f"{RESIDUAL_SLOT_KEY}.cells[{h}] carries no residual payload (kind {e.get('kind')!r}, fingerprint {str(e.get('residual_fingerprint'))[:12]!r}) "
                               f"-- {cell} would be served as a plain LoRA, not through the slot")
        if attestation is not None:
            ra = attestation.get(RESIDUAL_SLOT_KEY)
            if not isinstance(ra, Mapping):
                out.append(f"serving_attestation.{RESIDUAL_SLOT_KEY} ABSENT (schema-2 records carry the block: REQUIRED_TOP_KEYS_V2)")
            else:
                if ra.get("armed") is not True:
                    out.append(f"serving_attestation.{RESIDUAL_SLOT_KEY}.armed {ra.get('armed')!r} != True")
                reports = ra.get("worker_reports")
                if not isinstance(reports, list) or not reports:
                    out.append(f"serving_attestation.{RESIDUAL_SLOT_KEY}.worker_reports ABSENT or empty (no worker attested the residual op)")
                elif any(not (isinstance(r, Mapping) and r.get("armed") is True) for r in reports):
                    out.append(f"serving_attestation.{RESIDUAL_SLOT_KEY}.worker_reports: {sum(1 for r in reports if not (isinstance(r, Mapping) and r.get('armed') is True))} "
                               f"of {len(reports)} worker(s) not armed")
                exp_adapters = ((ra.get("expected") or {}).get("adapters") or {}) if isinstance(ra.get("expected"), Mapping) else {}
                for cell in armed_cells:
                    h = lp.harness_id(cell)
                    e = exp_adapters.get(h)
                    if not isinstance(e, Mapping):
                        out.append(f"serving_attestation.{RESIDUAL_SLOT_KEY}.expected.adapters has no entry for {cell} ({h})")
                    elif any(not e.get(k) for k in RESIDUAL_ATTESTED_PAYLOAD_KEYS):
                        out.append(f"serving_attestation.{RESIDUAL_SLOT_KEY}.expected.adapters[{h}] carries no residual payload (kind {e.get('kind')!r}, op {e.get('op')!r}, "
                                   f"fingerprint {str(e.get('residual_fingerprint'))[:12]!r})")
    # the pins the standup was TOLD live in cells_served.json (`expected_object_sha256_by_cell`); the artifacts it LOADED in
    # endpoint.json (`object_artifact_sha256_by_cell`). Both mandated, held equal, and each pin the plan carries compared by name.
    expected_map = cells_served.get(EXPECTED_DIGESTS_KEY)
    artifact_map = endpoint.get(ARTIFACT_DIGESTS_KEY)
    if not isinstance(expected_map, Mapping):
        out.append(f"cells_served.{EXPECTED_DIGESTS_KEY} ABSENT (a mandated attestation: the object-digest map the standup was told, expected_digests_source cli)")
    if not isinstance(artifact_map, Mapping):
        out.append(f"endpoint.{ARTIFACT_DIGESTS_KEY} ABSENT (a mandated attestation: the object artifacts the standup loaded)")
    if isinstance(expected_map, Mapping) and isinstance(artifact_map, Mapping) and dict(expected_map) != dict(artifact_map):
        out.append(f"endpoint.{ARTIFACT_DIGESTS_KEY} != cells_served.{EXPECTED_DIGESTS_KEY} (the standup loaded artifacts other than the pins it was told)")
    digests = expected_map if isinstance(expected_map, Mapping) else (artifact_map if isinstance(artifact_map, Mapping) else None)
    if digests is not None:
        key = EXPECTED_DIGESTS_KEY if isinstance(expected_map, Mapping) else ARTIFACT_DIGESTS_KEY
        for cell, obj in row["objects"].items():
            h = lp.harness_id(cell)
            if h not in digests:
                out.append(f"{key} has no entry for {h} (cell {cell}, pin {obj['object_sha256'][:12]}…)")
            elif digests.get(h) != obj["object_sha256"]:
                out.append(f"{key}[{h}] {digests.get(h)!r} != pin {obj['object_sha256'][:12]}… (cell {cell})")
    return out


# --- the metrics scrape (F-O): the ENGINE's Prometheus page THROUGH THE PROXY, verified once before it feeds §9.4 ----

VLLM_METRIC_FAMILIES = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:generation_tokens_total",
                        "vllm:prompt_tokens_total", "vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc")

#: the routing proxy's read-only pass-through of the engine's /metrics (`sisteer.capeval.serving.proxy.METRICS_ROUTE`; named
#: here because this file imports no sisteer at module level)
METRICS_ROUTE = "/metrics"


def metrics_url_from_endpoint(endpoint: Mapping[str, Any]) -> str:
    """The vLLM `/metrics` URL from the native endpoint contract -- the ROUTING PROXY's pass-through on `proxy_port`
    (`endpoint.json` `proxy_port`, else the port of `proxy_url`), NOT the engine's `vllm_port` (ten smokes once all
    passed readiness and then ended METRICS_UNSCRAPABLE on HTTP 401). The engine's own `/metrics` sits
    behind `vllm_auth.SisteerTokenAuthMiddleware` like every non-liveness path, and its per-launch bearer lives ONLY inside
    the standup process -- `endpoint.json` records `proxy_token: None` and the attempt root is synced to shared storage, so
    the token can never travel to this app through a file. The proxy's `/metrics` (`proxy.RoutingProxy.handle_metrics`)
    forwards to the engine with the launch's `auth_headers` and returns the body verbatim, unauthenticated like the proxy's
    other routes when the endpoint names no `proxy_token`."""
    port = endpoint.get("proxy_port")
    if not (isinstance(port, int) and not isinstance(port, bool) and port > 0):
        url = endpoint.get("proxy_url")
        try:
            port = int(str(url).rsplit(":", 1)[1].split("/")[0]) if isinstance(url, str) and url else 0
        except (ValueError, IndexError):
            port = 0
        if port <= 0:
            raise ValueError(f"endpoint.json carries no integer proxy_port ({endpoint.get('proxy_port')!r}) and no proxy_url with a port "
                             f"({url!r}); the metrics scrape reads the proxy's {METRICS_ROUTE} pass-through, never the guarded engine")
    return f"http://127.0.0.1:{port}{METRICS_ROUTE}"


def metrics_headers_from_endpoint(endpoint: Mapping[str, Any]) -> Dict[str, str]:
    """The headers the metrics scrape sends the proxy: the endpoint's `proxy_token` as a bearer when it names one (the
    proxy then guards `/metrics` exactly like `/v1/models`), else none -- the endpoint's `proxy_token: None` means the proxy
    checks no token. Never the ENGINE's token: this app never holds it."""
    token = endpoint.get("proxy_token")
    if isinstance(token, str) and token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def parse_vllm_metrics(text: str) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for w in VLLM_METRIC_FAMILIES:
            if line.startswith(w):
                try:
                    row[w] = float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    pass
    return row


def verify_metrics_scrape(url: str, fetch: Callable[[str], str]) -> Tuple[Dict[str, float], List[str]]:
    """ONE verified scrape before the sampler starts: the body must parse as Prometheus text carrying the vLLM request
    gauges (running / waiting) — a proxy 404 page, an HTML error or a connection refusal is a NAMED problem (fatal for
    the row: these scrapes are the ONLY inputs to §9.4)."""
    try:
        text = fetch(url)
    except Exception as e:  # noqa: BLE001
        return {}, [f"metrics scrape {url} failed: {type(e).__name__}: {e}"]
    row = parse_vllm_metrics(text)
    problems = []
    for must in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        if must not in row:
            problems.append(f"metrics scrape {url}: no {must} sample in {len(text)} bytes (not a vLLM /metrics body)")
    return row, problems


def _http_get_text(url: str, timeout: float = 5.0, headers: Optional[Mapping[str, str]] = None) -> str:
    """ONE GET, the body as text; the one HTTP read the metrics scrape (`verify_metrics_scrape`) and the in-container
    sampler (`_start_samplers`) share, so the target and its headers are decided in one place (F-O)."""
    import urllib.request
    req = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(errors="replace")


# --- progress from the attempt's OWN outputs ----------------------------------------------------------------------

def count_progress(root: Path) -> Dict[str, int]:
    """The attempt-local output counters the heartbeat carries: rows across every durable per-result stream under
    `<root>/exp` (§7.4's streams by file name), the number of such files, and the driver exits recorded in
    `driver_inflight.jsonl`. A watcher reading `rows_written` sees the store grow, not a constant zero."""
    root = Path(root)
    rows = files = 0
    exp = root / "exp"
    if exp.is_dir():
        for p in exp.rglob("*.jsonl"):
            if p.name not in PROGRESS_STREAM_NAMES or not p.is_file():
                continue
            files += 1
            try:
                with p.open("rb") as fh:
                    rows += sum(1 for _ in fh)
            except OSError:
                continue
    exits = 0
    inflight = root / "driver_inflight.jsonl"
    if inflight.is_file():
        try:
            exits = sum(1 for line in inflight.read_text().splitlines() if '"event": "exit"' in line)
        except OSError:
            exits = 0
    return {"rows_written": rows, "stream_files": files, "children_exited": exits}


def heartbeat_payload(*, attempt_id: str, phase: Optional[str], running: Optional[int], progress: Mapping[str, int],
                      extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    return {"attempt_id": attempt_id, "t": utc(), "rows_written": int(progress.get("rows_written", 0)),
            "stream_files": int(progress.get("stream_files", 0)), "children_exited": int(progress.get("children_exited", 0)),
            "running": running, "phase": phase, **dict(extra or {})}


# --- the acceptance state machine (§4.6 rev 3) ------------------------------------------------------------------------

class AttemptState:
    PARTIAL, COMPLETE, INVENTORIED, ACCEPTED, CLOSED, SUPERSEDED = "partial", "complete", "inventoried", "accepted", "closed", "superseded"


class PublicationFailed(RuntimeError):
    """A durable publication (copy to the volume + commit) FAILED: nothing after it may run, least of all the acceptance
    (the best-effort sync swallowed every error and an attempt could win the election with COMPLETE.json
    and its inventory unpublished)."""


class LadderInventoryIncomplete(RuntimeError):
    """A producer row reached its finish without an explicit, complete `ladder_enqueued.json`: the attempt
    FAILS instead of being accepted with an empty queue."""


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def attempt_record(*, attempt_id: str, call_id: str, task_id: str, W: str, wave_uuid: str, row: Mapping[str, Any],
                   prior_attempts: Sequence[Mapping[str, Any]], production_wave_uuid: Optional[str] = None,
                   launch_plan_sha: Optional[str] = None) -> Dict[str, Any]:
    """ATTEMPT.json. `wave_uuid` IS the execution wave (the wave this attempt ran in -- what `read_attempt_root` binds to; it must
    equal the control key's); `execution_wave_uuid` repeats it by name and `production_wave_uuid` + `launch_plan_sha` name the plan the
    attempt is certified against -- additive fields, the original keys and their meaning unchanged."""
    return {"schema": "seedrep_attempt/1", "attempt_id": attempt_id, "call_id": call_id, "task_id": task_id, "W": W,
            "wave_uuid": wave_uuid, "execution_wave_uuid": wave_uuid, "production_wave_uuid": production_wave_uuid or wave_uuid,
            "launch_plan_sha": launch_plan_sha, "launch_plan_row": row["standup_id"], "control_key": row["control_key"],
            "accepted_key": row.get("accepted_key"), "prior_attempts_seen": list(prior_attempts), "started_utc": utc(),
            "resumed_from": None, "superseded_by": None}


def prior_attempts_of(attempts_dir: Path) -> List[Dict[str, Any]]:
    """Every sibling attempt root + its state (evidence for ATTEMPT.json; nothing is ever resumed onto one)."""
    out = []
    if attempts_dir.is_dir():
        for d in sorted(p for p in attempts_dir.iterdir() if p.is_dir()):
            out.append({"attempt_id": d.name, "state": attempt_state(d)})
    return out


def attempt_state(root: Path, closed_marker: Optional[Path] = None) -> str:
    root = Path(root)
    att = {}
    if (root / ATTEMPT_JSON).is_file():
        try:
            att = json.loads((root / ATTEMPT_JSON).read_text())
        except ValueError:
            att = {}
    if att.get("superseded_by"):
        return AttemptState.SUPERSEDED
    if closed_marker is not None and closed_marker.is_file():
        return AttemptState.CLOSED
    if att.get("accepted"):
        return AttemptState.ACCEPTED
    if (root / INVENTORY_JSON).is_file():
        return AttemptState.INVENTORIED
    if (root / COMPLETE_JSON).is_file():
        return AttemptState.COMPLETE
    return AttemptState.PARTIAL


def write_ladder_enqueued(root: Path, *, attempt_id: str, control_key: str, accepted_key: str, ladder: Mapping[str, Any],
                          entries: Sequence[Mapping[str, Any]], streams: Sequence[Mapping[str, Any]], complete: bool,
                          problems: Sequence[str] = ()) -> Dict[str, Any]:
    """`<root>/ladder_enqueued.json` (`seedrep_ladder_enqueued/1`): the EXPLICIT inventory of every queue file THIS attempt
    enqueued — the producer's SEALED entry set after the tailer drained its streams — plus what was read (per stream: rows,
    at-cap rows, enqueued). `complete` is False when a stream could not be read to its end or a record refused; a
    producer row is REQUIRED to carry a complete one before its acceptance."""
    doc = {"schema": "seedrep_ladder_enqueued/1", "attempt_id": attempt_id, "control_key": control_key, "accepted_key": accepted_key,
           "ladder": dict(ladder), "n": len(entries), "entries": [dict(e) for e in entries], "streams": [dict(x) for x in streams],
           "complete": bool(complete), "problems": list(problems), "written_utc": utc()}
    write_json(Path(root) / LADDER_ENQUEUED_JSON, doc)
    return doc


def require_ladder_inventory(root: Path, *, expected_entries: Optional[Sequence[Mapping[str, Any]]] = None,
                             require_rows: bool = False) -> Dict[str, Any]:
    """A producer row's finish REQUIRES `ladder_enqueued.json`: present, `complete`, `n == len(entries)`, every entry with an
    `item_key` / `path` / `sha256`, and — when the caller holds the producer's own sealed set — the SAME set of paths.
    Anything less raises :class:`LadderInventoryIncomplete` (the attempt FAILS). ``require_rows`` (a BASE producer row):
    an EMPTY inventory whose streams read NO rows is the silent-failure case -- children that exited 0 having written nothing (renellm
    without its corpus) -- and is refused by name; an empty inventory over streams that DID carry rows (nothing hit the cap) stands."""
    p = Path(root) / LADDER_ENQUEUED_JSON
    if not p.is_file():
        raise LadderInventoryIncomplete(f"{p.name} missing: a producer row must carry its tailer's explicit inventory before acceptance")
    doc = json.loads(p.read_text())
    if doc.get("schema") != "seedrep_ladder_enqueued/1":
        raise LadderInventoryIncomplete(f"{p.name}: schema {doc.get('schema')!r} is not seedrep_ladder_enqueued/1")
    if not doc.get("complete"):
        raise LadderInventoryIncomplete(f"{p.name}: complete=False — {doc.get('problems')}")
    entries = doc.get("entries")
    if not isinstance(entries, list) or doc.get("n") != len(entries):
        raise LadderInventoryIncomplete(f"{p.name}: n {doc.get('n')!r} != {len(entries) if isinstance(entries, list) else 'no'} entries")
    for e in entries:
        if not (isinstance(e, Mapping) and e.get("item_key") and e.get("path") and e.get("sha256")):
            raise LadderInventoryIncomplete(f"{p.name}: entry {e!r} lacks item_key / path / sha256")
    if require_rows and not entries:
        streams = doc.get("streams") or []
        rows_read = sum(int((st or {}).get("rows_read") or 0) for st in streams if isinstance(st, Mapping))
        if not streams or rows_read == 0:
            raise LadderInventoryIncomplete(f"{p.name}: an empty inventory over {len(streams)} stream(s) that carried no rows -- the children "
                                            "wrote nothing (a driver that loud_skipped and exited 0: the D26 face); never accepted with []")
    if expected_entries is not None:
        want = sorted(str(e["path"]) for e in expected_entries)
        got = sorted(str(e["path"]) for e in entries)
        if want != got:
            raise LadderInventoryIncomplete(f"{p.name}: entries {len(got)} != the producer's sealed set {len(want)}")
    return doc


def finish_attempt(*, root: Path, key: str, attempt_id: str, inventory: Sequence[Mapping[str, Any]], accepted: Any,
                   commit: Callable[[], None], closed_marker: Optional[Path] = None, summary: Mapping[str, Any],
                   producer: Any = None, ladder_required: bool = False, require_rows: bool = False) -> Dict[str, Any]:
    """The ORDER of publication, each step PUBLISHED before the next through ``commit`` — a THROWING barrier: a
    copy or commit failure raises and NOTHING after it runs, least of all the acceptance. (1) COMPLETE.json; (2) the
    inventory — the ``producer``'s `queue_inventory.json` (its own schema, sealed entry set; its volume commits ARE
    this barrier) or an empty one for a non-ladder row; a producer row FIRST requires the tailer's explicit
    `ladder_enqueued.json` (:func:`require_ladder_inventory`) — missing / incomplete raises and the attempt FAILS; (3) the
    acceptance = ONE ``accepted.put(key, attempt_id, skip_if_exists=True)`` under the producer's key spelling; (4) the
    accepted attempt's producer CLOSED marker (`producer.write_closed_marker`). A loser reads the holder, marks itself
    ``superseded_by`` in its OWN ATTEMPT.json and publishes nothing."""
    root = Path(root)
    if ladder_required:
        require_ladder_inventory(root, expected_entries=(producer.seal() if producer is not None else inventory), require_rows=require_rows)
    write_json(root / COMPLETE_JSON, {"attempt_id": attempt_id, "control_key": key, "completed_utc": utc(), **dict(summary)})
    commit()
    if producer is not None:
        inv_doc = producer.write_inventory()                     # its own schema; seals; commits through the barrier
        inventory = list(inv_doc["entries"])
    else:
        write_json(root / INVENTORY_JSON, {"attempt_id": attempt_id, "control_key": key, "n": len(inventory),
                                            "entries": [dict(e) for e in inventory], "written_utc": utc()})
        commit()
    won = bool(accepted.put(key, attempt_id, skip_if_exists=True))
    att_path = root / ATTEMPT_JSON
    att = json.loads(att_path.read_text()) if att_path.is_file() else {"attempt_id": attempt_id}
    if not won:
        holder = accepted.get(key)
        att["superseded_by"] = holder
        att["superseded_utc"] = utc()
        write_json(att_path, att)
        commit()
        return {"state": AttemptState.SUPERSEDED, "accepted_attempt": holder, "attempt_id": attempt_id}
    att["accepted"] = True
    att["accepted_utc"] = utc()
    write_json(att_path, att)
    commit()
    if producer is not None:
        producer.write_closed_marker(accepted)
        return {"state": AttemptState.CLOSED, "accepted_attempt": attempt_id, "attempt_id": attempt_id, "n_enqueued": len(inventory)}
    if closed_marker is not None:
        keys = sorted(str(e.get("item_key")) for e in inventory)
        write_json(closed_marker, {"attempt_id": attempt_id, "control_key": key, "n_enqueued": len(inventory),
                                   "item_keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(), "closed_utc": utc()})
        commit()
        return {"state": AttemptState.CLOSED, "accepted_attempt": attempt_id, "attempt_id": attempt_id}
    return {"state": AttemptState.ACCEPTED, "accepted_attempt": attempt_id, "attempt_id": attempt_id}


def reconciler_closable(root: Path, *, accepted_attempt: Optional[str], volume_has: Callable[[str], Optional[str]]) -> Tuple[bool, str]:
    """The grading host reconciler's authority to SYNTHESISE a CLOSED record: the attempt is the ACCEPTED one,
    its root shows COMPLETE.json AND queue_inventory.json, and every inventory entry's attempt-unique file is on the volume
    with the recorded sha (``volume_has(path) -> sha256 | None``). Both inventory shapes are read: this module's (an
    `attempt_id` per entry) and the producer's `seedrep_queue_inventory/1` (the attempt id at the top). A superseded
    attempt's later publications cannot disturb this: no two attempts share a queue path."""
    root = Path(root)
    att_id = root.name
    if accepted_attempt != att_id:
        return False, f"{att_id} is not the accepted attempt ({accepted_attempt!r})"
    if not (root / COMPLETE_JSON).is_file():
        return False, "no COMPLETE.json"
    if not (root / INVENTORY_JSON).is_file():
        return False, "no queue_inventory.json"
    inv = json.loads((root / INVENTORY_JSON).read_text())
    if inv.get("attempt_id") not in (None, att_id):
        return False, f"inventory names attempt {inv.get('attempt_id')!r}, not {att_id}"
    for e in inv.get("entries", []):
        if e.get("attempt_id", inv.get("attempt_id")) != att_id:
            return False, f"inventory entry {e.get('item_key')} names attempt {e.get('attempt_id')!r}, not {att_id}"
        got = volume_has(str(e["path"]))
        if got != e.get("sha256"):
            return False, f"inventory file {e['path']} sha {got!r} != {e.get('sha256')!r}"
    return True, "accepted + COMPLETE + inventory verified"


#: when the ACCEPTED rung attempt adopted another same-call execution's elected results, that execution's request evidence /
#: ledger rows / engine-log evidence is copied into `<accepted root>/adopted_from/<source attempt>/` as a verified dependency.
#: The pull downloads the whole accepted root (so the directory rides along), lists every adopted attempt in the pull row, manifests
#: its files, and REFUSES an adopted directory declared with no files -- a dependency with no evidence.
ADOPTED_FROM_DIR = "adopted_from"


def adopted_from_attempts(pulled: Path) -> Tuple[List[str], List[str]]:
    """`(adopted attempt ids, the ones whose directory holds no file)` under a pulled accepted root."""
    d = Path(pulled) / ADOPTED_FROM_DIR
    if not d.is_dir():
        return [], []
    adopted = sorted(p.name for p in d.iterdir() if p.is_dir())
    empty = [a for a in adopted if not any(q.is_file() for q in (d / a).rglob("*"))]
    return adopted, empty


def pull_manifest_rows(accepted_attempt: str, files: Iterable[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """RS/pull_manifest.jsonl rows: the attempt id per pulled file (sha-manifested)."""
    return [{"attempt_id": accepted_attempt, "path": p, "sha256": s} for p, s in files]


class NoAcceptedAttempt(RuntimeError):
    """A shard whose accepted Dict entry is absent: nothing may be pulled, published or read for it."""


def accepted_attempt_for(accepted: Any, key: str) -> str:
    """The ONE attempt the pull / the ladder producers / the audits / the registry publication read for a control key —
    the holder of the `accepted` Dict entry (keyed as `row["accepted_key"]`). Absent = REFUSED (never
    the newest root, never a guess)."""
    att = accepted.get(key)
    if not att:
        raise NoAcceptedAttempt(f"no accepted attempt for {key}: the shard has not completed (or every attempt died before acceptance)")
    return str(att)


# --------------------------------------------------------------------------- #
# the ladder's PRODUCER side in the base container                              #
# --------------------------------------------------------------------------- #

LADDER_LIB_REL = "scripts/modal/seedrep_ladder.py"            # the ladder protocol module, loaded BY PATH from the staged tree


class LadderUnavailable(RuntimeError):
    """A producer row on a tree without `seedrep_ladder.py`, or a stream this tailer cannot turn into queue records:
    the attempt FAILS by name before any GPU minute is spent (never an empty inventory)."""


def load_module_by_path(name: str, path: Path):
    """impl_ws2e §2's recipe: register in sys.modules BEFORE exec (dataclasses + `from __future__ import annotations`)."""
    import importlib.util
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    path = Path(path)
    if not path.is_file():
        raise LadderUnavailable(f"{path} is absent: WS2-E's protocol module is required on a producer row's tree")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def ensure_tree_src_importable(tree: Path) -> str:
    """The staged tree's ``src`` at the HEAD of the APP process's ``sys.path`` (launch F-G). The ladder module is loaded by
    path INTO the Modal function's own interpreter (the image's system python: uv + pyyaml + httpx), and its producer side
    imports stdlib-only `sisteer` leaves lazily (`sampler_keys` -> `sisteer.eval.attacks.bon_served.SAMPLER_KEYS`,
    `redraw_messages` -> `sisteer.eval.system_prefix`). `PYTHONPATH=<tree>/src` reaches only the CHILD processes
    (`container_env`), never this interpreter, so without this line the first at-cap row raised ModuleNotFoundError inside
    the tailer and the row ended LADDER_TAILER_ERROR after every GPU-hour was spent. Idempotent; returns the inserted path."""
    src = str(Path(tree) / "src")
    if src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)
    return src


def load_ladder_lib(tree: Path):
    return load_module_by_path("seedrep_ladder", Path(tree) / LADDER_LIB_REL)


def volume_rel(path: str) -> str:
    """`/vol/...` -> the volume-RELATIVE path `LocalVolume(Path('/vol'))` addresses (no leading slash)."""
    return volume_client_path(path).lstrip("/")


def container_path_to_volume_rel(local_path: Path, container_root: Path, attempt_root_vol: str) -> str:
    """A file under the container-local attempt root -> its volume-relative path once the TreeSyncer has shipped it."""
    rel = Path(local_path).relative_to(Path(container_root))
    return f"{volume_rel(attempt_root_vol)}/{rel.as_posix()}"


def sampling_from_decode(decode: Mapping[str, Any]) -> Dict[str, Any]:
    """The static row's `decode` block minus its bookkeeping = the sampler preset the re-draw re-issues."""
    return {k: v for k, v in decode.items() if k not in ("do_sample", "cap", "arm", "seed_base", "trial")}


def zlib_ratio(text: str) -> float:
    raw = text.encode("utf-8", errors="replace")
    return (len(zlib.compress(raw)) / len(raw)) if raw else 1.0


def read_new_jsonl_rows(path: Path, offset: int) -> Tuple[List[dict], int, List[dict]]:
    """Rows appended since ``offset`` (COMPLETE lines only; an unterminated FINAL line is left for the next pass -- the one
    deferral). Returns (rows, new_offset, malformed) where ``malformed`` names every complete line that is not JSON, with its
    byte offset in the file, its sha256 and its head (a malformed complete line is a scientific record that is
    MISSING from the stream; the tailer records it as a fatal problem, never as a counter)."""
    rows, malformed = [], []
    with path.open("rb") as fh:
        fh.seek(offset)
        data = fh.read()
    if not data:
        return rows, offset, []
    end = data.rfind(b"\n")
    if end < 0:
        return rows, offset, []
    pos = offset
    for line in data[: end + 1].split(b"\n"):
        start = pos
        pos += len(line) + 1
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            malformed.append({"offset": start, "length": len(line), "sha256": hashlib.sha256(line).hexdigest(), "head": line[:80].decode(errors="replace")})
    return rows, offset + end + 1, malformed


class StreamAdapter:
    """ONE ladder kind's view of the attempt's durable per-result streams: which files, which rows are AT CAP (and
    eligible), and the queue record for one. ``live`` adapters enqueue as rows land; the others BUFFER at-cap rows
    and build their records at DRAIN, when the fields the record needs (the misalign request messages, a control's
    prompt) can be resolved in one pass."""

    kind = ""
    live = True

    def streams(self, root: Path) -> List[Path]:
        raise NotImplementedError

    def at_cap(self, row: Mapping[str, Any]) -> bool:
        raise NotImplementedError

    def records(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> List[dict]:
        raise NotImplementedError


class StaticStreamAdapter(StreamAdapter):
    """`exp/transcripts_results/<private_run>/<leg>/<defense>/shard<NN>/responses.jsonl` (jailbreak_run's `_gated_shard_dir`):
    the arm is the `<defense>` path component, the seed base the row's `decode.seed_base`, the re-draw = the row's
    `attack_prompt` under the arm's SI (a prompted arm) at the row's sampler preset against the arm's virtual model."""

    kind = "static"
    live = True

    def __init__(self, L, spec, ref, attempt_id: str, *, leg: str, container_root: Path, attempt_root_vol: str,
                 vmodel_of: Callable[[str], str], system_of: Callable[[str], Optional[str]]) -> None:
        self.L, self.spec, self.ref, self.attempt_id, self.leg = L, spec, ref, attempt_id, leg
        self.container_root, self.attempt_root_vol = Path(container_root), attempt_root_vol
        self.vmodel_of, self.system_of = vmodel_of, system_of

    def streams(self, root: Path) -> List[Path]:
        return sorted(Path(root).glob(f"exp/transcripts_results/*/{self.leg}/*/shard*/responses.jsonl"))

    def at_cap(self, row: Mapping[str, Any]) -> bool:
        return row.get("finish_reason") == "length" or row.get("at_cap") is True

    def records(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> List[dict]:
        arm = path.parts[-3]
        base_store = container_path_to_volume_rel(path, self.container_root, self.attempt_root_vol)
        out = []
        for row in rows:
            dec = row.get("decode") or {}
            if "seed_base" not in dec:
                raise LadderUnavailable(f"{path}: a sampled static row without decode.seed_base cannot name its seed run")
            out.append(self.L.static_queue_record(self.spec, arm=arm, s=int(dec["seed_base"]), family=self.leg, row=row, base_store=base_store,
                                                  producer=self.ref, producer_attempt_id=self.attempt_id, vmodel=self.vmodel_of(arm),
                                                  sampling=sampling_from_decode(dec), system=self.system_of(arm)))
        return out


class MisalignStreamAdapter(StreamAdapter):
    """`exp/misalign/agentic__s<s>__<behaviour>/<cell dir>/responses.jsonl` (the per-(seed, behaviour) roots):
    at cap AND not a repetition loop (`runner.truncated_selection`'s zlib rule, ratio >= 0.2); the re-draw needs the runner's
    exact request messages, resolved at DRAIN for every buffered (cell, behaviour, item) through ``messages_of`` (the
    driver-venv helper `seedrep_ladder_helpers.py misalign-messages`), the sampler row from the cell dir's `meta.json`."""

    kind = "misalign"
    live = False

    def __init__(self, L, spec, ref, attempt_id: str, *, container_root: Path, attempt_root_vol: str,
                 messages_of: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Sequence[Mapping[str, str]]]],
                 min_zlib: float = MISALIGN_MIN_ZLIB) -> None:
        self.L, self.spec, self.ref, self.attempt_id = L, spec, ref, attempt_id
        self.container_root, self.attempt_root_vol, self.messages_of, self.min_zlib = Path(container_root), attempt_root_vol, messages_of, min_zlib

    def streams(self, root: Path) -> List[Path]:
        return sorted(Path(root).glob("exp/misalign/agentic__s*__*/*/responses.jsonl"))

    def at_cap(self, row: Mapping[str, Any]) -> bool:
        return row.get("finish_reason") == "length" and zlib_ratio(str(row.get("response") or "")) >= self.min_zlib

    @staticmethod
    def request_key(cell_id: str, behavior: str, item_id: str) -> str:
        return f"{cell_id}|{behavior}|{item_id}"

    def records(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> List[dict]:
        m = re.match(r"agentic__s(\d+)__(.+)$", path.parts[-3])
        if not m:
            raise LadderUnavailable(f"{path}: not a per-(seed, behaviour) misalign root")
        s = int(m.group(1))
        meta_p = path.parent / "meta.json"
        sampling = (json.loads(meta_p.read_text()).get("sampling") if meta_p.is_file() else None) or {}
        if not sampling:
            raise LadderUnavailable(f"{meta_p}: no sampling row; the re-draw needs the runner's resolved sampler")
        base_store = container_path_to_volume_rel(path, self.container_root, self.attempt_root_vol)
        wanted = [{"cell_id": r["cell_id"], "behavior": r.get("behavior"), "item_id": str(r["item_id"])} for r in rows]
        messages = self.messages_of(wanted)
        out = []
        for r in rows:
            key = self.request_key(r["cell_id"], str(r.get("behavior")), str(r["item_id"]))
            if key not in messages:
                raise LadderUnavailable(f"{path}: no request messages resolved for {key}")
            out.append(self.L.misalign_queue_record(self.spec, arm=lp.traintest_id(r["cell_id"]), s=s, row=r, messages=messages[key],
                                                    base_store=base_store, producer=self.ref, producer_attempt_id=self.attempt_id, sampling=sampling))
        return out


CONTROLS_ROOT_RE = re.compile(r"^(?:controls|gsm8k_thinking_on)_cap(?P<cap>\d+)__s(?P<s>\d+)$")


def controls_seed_of_store(path: Path) -> int:
    """The seed base a controls / GSM8K store belongs to, read off its seed-owned root `<...>_cap<N>__s<s>` (lp.CONTROLS_OUT_REL /
    GSM8K_OUT_REL): the Arditi engine's three seed children no longer share a root, so the attribution is the path's,
    never a guess from the row's seed list. Refuses a path with no seed-owned component."""
    for part in reversed(Path(path).parts):
        m = CONTROLS_ROOT_RE.match(part)
        if m:
            return int(m.group("s"))
    raise LadderUnavailable(f"{path}: no seed-owned root component (`controls_cap<N>__s<s>` / `gsm8k_thinking_on_cap<N>__s<s>`) -- the store "
                            "cannot be attributed to one seed base")


def controls_prompts_via_helper(driver_python: str, tree: Path, control: str, item_ids: Sequence[str], *, env: Mapping[str, str]) -> Dict[str, str]:
    """`seedrep_ladder_helpers.py controls-prompts` under the DRIVER venv: the judged corpora's pinned rows (`judged.xstest_rows` /
    `or_bench_rows`, the loaders the driver itself used) -> {item_id: prompt} for the ids asked."""
    with tempfile.TemporaryDirectory(prefix="seedrep-controls-prompts-") as td:
        req, out = Path(td) / "items.json", Path(td) / "prompts.json"
        req.write_text(json.dumps(sorted(set(str(i) for i in item_ids))))
        r = subprocess.run([driver_python, str(Path(tree) / "scripts/modal/seedrep_ladder_helpers.py"), "controls-prompts", "--control", control,
                            "--items", str(req), "--out", str(out)], capture_output=True, text=True, env=dict(env), cwd=str(tree), timeout=1800)
        if r.returncode != 0:
            raise LadderUnavailable(f"controls-prompts helper rc {r.returncode} for {control}: {(r.stdout + r.stderr)[-800:]}")
        return json.loads(out.read_text())


def controls_sampling_via_helper(driver_python: str, tree: Path, config_rel: str, *, env: Mapping[str, str]) -> Dict[str, Any]:
    """`seedrep_ladder_helpers.py controls-sampling` under the DRIVER venv: the child's config's `controls_gen_config` -- the SAME
    resolver the served controls driver keys its stores and spool headers on (the policy row the sampled block names)."""
    with tempfile.TemporaryDirectory(prefix="seedrep-controls-sampling-") as td:
        out = Path(td) / "sampling.json"
        r = subprocess.run([driver_python, str(Path(tree) / "scripts/modal/seedrep_ladder_helpers.py"), "controls-sampling", "--config", config_rel,
                            "--out", str(out)], capture_output=True, text=True, env=dict(env), cwd=str(tree), timeout=600)
        if r.returncode != 0:
            raise LadderUnavailable(f"controls-sampling helper rc {r.returncode} for {config_rel}: {(r.stdout + r.stderr)[-800:]}")
        return json.loads(out.read_text())


class ControlsStreamAdapter(StreamAdapter):
    """`controls_events.jsonl` (per store: `item_id, at_cap, finish_reason, sampling_seed, text_sha256, request_id,
    completion_id, cell_id, control, error`): at cap and error-free. The record needs the item's PROMPT and the cell's system
    text, which the event does not carry: ``prompt_of(control, item_id, cell)`` resolves them at DRAIN; without a resolver
    this adapter REFUSES BY NAME (the attempt fails; nothing is enqueued silently)."""

    kind = "controls"
    live = False

    def __init__(self, L, spec, ref, attempt_id: str, *, container_root: Path, attempt_root_vol: str,
                 seed_of_store: Callable[[Path], int], vmodel_of: Callable[[str], str], sampling: Optional[Mapping[str, Any]] = None,
                 prompt_of: Optional[Callable[[str, str, str], Tuple[str, Optional[str]]]] = None,
                 sampling_of: Optional[Callable[[str, int], Mapping[str, Any]]] = None,
                 prompts_of: Optional[Callable[[str, Sequence[str]], Mapping[str, str]]] = None,
                 system_of: Optional[Callable[[str], Optional[str]]] = None) -> None:
        self.L, self.spec, self.ref, self.attempt_id = L, spec, ref, attempt_id
        self.container_root, self.attempt_root_vol = Path(container_root), attempt_root_vol
        self.seed_of_store, self.vmodel_of, self.prompt_of = seed_of_store, vmodel_of, prompt_of
        self.sampling, self.sampling_of = (dict(sampling) if sampling is not None else None), sampling_of
        #: F-AM: the BATCH resolvers -- `prompts_of(control, item_ids) -> {item_id: prompt}` ONCE per (stream, control) and
        #: `system_of(arm)` ONCE per arm per stream. The per-item `prompt_of` stays as the fallback (a test's injected resolver);
        #: `prompts_of` wins when both are wired.
        self.prompts_of, self.system_of = prompts_of, system_of

    def _resolver(self, rows: Sequence[Mapping[str, Any]]) -> Callable[[str, str, str], Tuple[str, Optional[str]]]:
        """The (control, item_id, arm) -> (prompt, system) resolver for ONE stream's rows. With `prompts_of` wired every distinct
        (control, item id) of the stream is resolved in ONE call per control BEFORE the per-event loop, and the arm's system text once
        per arm (controls rows used to run one driver-venv helper subprocess per DISTINCT at-cap (control, item) -- the cache
        only ever held the ids already asked one at a time -- plus one `si-text` subprocess PER at-cap event of a prompted arm, all at
        drain with the engine already torn down: 8 streams x up to 1,769 draws, hours of idle H200 and a container that never
        returned; measured: 120 events -> 80 + 120 interpreter launches, now 6 + 1). Without it the per-item
        `prompt_of` is called per event."""
        if self.prompts_of is None:
            if self.prompt_of is None:
                raise LadderUnavailable("controls: the controls_events stream carries no item prompt and no prompt resolver was wired "
                                        "(WS1-C: expose the judged specs' item prompts, or carry `prompt` + `system_sha256` on the event)")
            return self.prompt_of
        wanted: Dict[str, List[str]] = {}
        for ev in rows:
            wanted.setdefault(str(ev["control"]), []).append(str(ev["item_id"]))
        prompts = {control: dict(self.prompts_of(control, sorted(set(ids)))) for control, ids in sorted(wanted.items())}
        systems: Dict[str, Optional[str]] = {}

        def resolve(control: str, item_id: str, arm: str) -> Tuple[str, Optional[str]]:
            prompt = prompts.get(control, {}).get(item_id)
            if prompt is None:
                raise LadderUnavailable(f"controls: no pinned {control} row carries item {item_id} (the judged corpora loaders)")
            if arm not in systems:
                systems[arm] = self.system_of(arm) if self.system_of is not None else None
            return prompt, systems[arm]
        return resolve

    def streams(self, root: Path) -> List[Path]:
        return sorted(Path(root).glob("exp/sampled/controls_cap*/**/controls_events.jsonl"))

    def at_cap(self, row: Mapping[str, Any]) -> bool:
        return row.get("at_cap") is True and row.get("error") is None

    def records(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> List[dict]:
        resolve = self._resolver(rows)                                   # F-AM: one helper call per (stream, control), never one per event
        s = self.seed_of_store(path)
        base_store = container_path_to_volume_rel(path, self.container_root, self.attempt_root_vol)
        out = []
        for ev in rows:
            cell = str(ev.get("cell_id"))
            arm = lp.ladder_arm_of_cell(cell)                            # F-AK: the SAME translation the wiring indexes the row's children by
            prompt, system = resolve(str(ev["control"]), str(ev["item_id"]), arm)
            if self.sampling_of is not None:
                sampling = dict(self.sampling_of(arm, s))                # the store's OWNER's config row: (arm, seed base) -- an Arditi seed child has its own
            elif self.sampling is not None:
                sampling = self.sampling
            else:
                raise LadderUnavailable("controls: no sampling row -- neither the arm's config (controls_gen_config) nor an explicit sampling was wired")
            if not sampling:
                raise LadderUnavailable(f"controls: an EMPTY sampling row for arm {arm} -- the re-draw needs the policy row the driver sent")
            out.append(self.L.controls_queue_record(self.spec, arm=arm, s=s, control=str(ev["control"]), event=ev, prompt=prompt, system=system,
                                                    base_store=base_store, producer=self.ref, producer_attempt_id=self.attempt_id,
                                                    vmodel=self.vmodel_of(arm), sampling=sampling))
        return out


class Gsm8kStreamAdapter(StreamAdapter):
    """Inspect files `outcomes/inspect__gsm8k_paired200.json` at the END of a run (`items[]` with `item_id, sample_idx,
    stop_reason, at_cap, ...`, `inspect_rung.ITEM_FACT_FIELDS`): a JSON document, not a jsonl stream -- read whole at
    DRAIN; at cap = `at_cap` AND `stop_reason == max_tokens` (the builder re-checks). The arm / seed come from the child that
    owns the store (``owner_of(path) -> (arm, s)``)."""

    kind = "gsm8k"
    live = False

    def __init__(self, L, spec, ref, attempt_id: str, *, container_root: Path, attempt_root_vol: str,
                 owner_of: Callable[[Path], Tuple[str, int]]) -> None:
        self.L, self.spec, self.ref, self.attempt_id = L, spec, ref, attempt_id
        self.container_root, self.attempt_root_vol, self.owner_of = Path(container_root), attempt_root_vol, owner_of

    def streams(self, root: Path) -> List[Path]:
        return sorted(Path(root).glob("exp/sampled/gsm8k_thinking_on_cap*/**/outcomes/inspect__gsm8k_paired200.json"))

    def at_cap(self, row: Mapping[str, Any]) -> bool:
        return bool(row.get("at_cap")) and row.get("stop_reason") == "max_tokens"

    def records(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> List[dict]:
        arm, s = self.owner_of(path)
        base_store = container_path_to_volume_rel(path, self.container_root, self.attempt_root_vol)
        return [self.L.gsm8k_queue_record(self.spec, arm=arm, s=s, item=it, base_store=base_store, producer=self.ref,
                                          producer_attempt_id=self.attempt_id) for it in rows]


def _read_outcomes_items(path: Path) -> List[dict]:
    doc = json.loads(path.read_text())
    items = doc.get("items") if isinstance(doc, Mapping) else None
    if not isinstance(items, list):
        raise LadderUnavailable(f"{path}: no items[] list (inspect_rung's all-attempted outcome record)")
    return items


class StreamTailer:
    """Follows the adapter's streams from the attempt root and ENQUEUES every at-cap item through the ladder's Producer on a
    tailer thread: per pass, the new complete rows of every stream; a live adapter builds records at once, a buffering one
    at drain. `drain()` runs a final pass after every driver exited, flushes the buffer, joins the thread and returns the
    per-stream summary; a builder refusal is recorded and re-raised at drain so the attempt FAILS instead of sealing a
    silently short inventory."""

    def __init__(self, root: Path, adapter: StreamAdapter, producer: Any, *, poll_s: float = TAILER_POLL_S,
                 log: Callable[[str], Any] = print) -> None:
        self.root, self.adapter, self.producer, self.poll_s, self.log = Path(root), adapter, producer, poll_s, log
        self.offsets: Dict[str, int] = {}
        self.stats: Dict[str, Dict[str, int]] = {}
        self.buffer: Dict[str, List[dict]] = {}
        self.problems: List[str] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._draining = False
        self.enqueued = 0

    def _stat(self, path: Path) -> Dict[str, int]:
        return self.stats.setdefault(str(path), {"rows_read": 0, "at_cap": 0, "enqueued": 0, "bad_lines": 0})

    def _enqueue(self, path: Path, rows: List[dict]) -> None:
        if not rows:
            return
        try:
            for rec in self.adapter.records(path, rows):
                self.producer.enqueue(rec)
                self._stat(path)["enqueued"] += 1
                self.enqueued += 1
        except Exception as e:  # noqa: BLE001 -- recorded, re-raised at drain
            self.problems.append(f"{path}: {type(e).__name__}: {e}")
            self.log(f"[tailer] REFUSED {path}: {type(e).__name__}: {e}")

    def pass_once(self) -> None:
        with self._lock:
            for path in self.adapter.streams(self.root):
                key = str(path)
                st = self._stat(path)
                try:
                    if path.suffix == ".json":
                        if not self._draining or self.offsets.get(key):
                            continue                      # a whole DOCUMENT (written at the end): read exactly once, at drain
                        rows = _read_outcomes_items(path)
                        self.offsets[key] = path.stat().st_size
                        bad = []                                     # a whole document parsed at once has no malformed LINES
                    else:
                        rows, new_off, bad = read_new_jsonl_rows(path, self.offsets.get(key, 0))
                        self.offsets[key] = new_off
                except Exception as e:  # noqa: BLE001
                    self.problems.append(f"{path}: {type(e).__name__}: {e}")
                    continue
                st["rows_read"] += len(rows)
                st["bad_lines"] += len(bad)
                for b in bad:
                    # a COMPLETE line that is not JSON: a record the driver meant to write is missing from the stream -- fatal for the
                    # attempt (drain raises, nothing is sealed or accepted); location and digest kept so the line can be found again
                    self.problems.append(f"{path}: malformed COMPLETE JSONL line at byte offset {b['offset']} (length {b['length']}, sha256 "
                                         f"{b['sha256'][:16]}..., head {b['head']!r}) -- a scientific record is missing; acceptance blocked")
                    self.log(f"[tailer] MALFORMED {path} @{b['offset']} sha256 {b['sha256'][:16]}")
                capped = [r for r in rows if self.adapter.at_cap(r)]
                st["at_cap"] += len(capped)
                if not capped:
                    continue
                if self.adapter.live:
                    self._enqueue(path, capped)
                else:
                    self.buffer.setdefault(key, []).extend(capped)

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self.pass_once()
            except Exception as e:  # noqa: BLE001
                self.problems.append(f"pass: {type(e).__name__}: {e}")

    def start(self) -> "StreamTailer":
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ladder-tailer")
        self._thread.start()
        return self

    def drain(self) -> Dict[str, Any]:
        """The final pass over every stream (the drivers have exited), the buffered records built and enqueued, the
        thread joined. Raises :class:`LadderUnavailable` when any refusal was recorded."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=600)
        self._draining = True
        self.pass_once()                                  # a stream that only appears now (a driver's last file); the .json documents
        with self._lock:
            for key, rows in list(self.buffer.items()):
                self._enqueue(Path(key), rows)
            self.buffer.clear()
        summary = {"streams": [{"path": k, **v} for k, v in sorted(self.stats.items())], "enqueued": self.enqueued,
                   "problems": list(self.problems), "complete": not self.problems}
        if self.problems:
            raise LadderUnavailable(f"the ladder tailer recorded {len(self.problems)} refusal(s): {self.problems[:3]}")
        return summary


def ladder_producer_for_row(L, row: Mapping[str, Any], ctx: Mapping[str, Any], *, vol, attempt_id: str, attempt_root_vol: str):
    """The `Producer` for a base row's declared queue (`row["ladder"]`: kind, cap_next, this shard, every declared shard) --
    the spec's producers are the base shards at stage `base`; `attempt_root` is volume-relative (the inventory lands there)."""
    lad = row["ladder"]
    W7, wave_uuid = ctx["W"][:7], ctx["wave_uuid"]
    spec = L.QueueSpec.build(W7, wave_uuid, row["leg"], row["engine"], lad["kind"], int(lad["cap_next"]), list(lad["shards"]))
    ref = next(p for p in spec.producers if p.shard == str(lad["shard"]))
    return L.Producer(vol, spec, ref, attempt_id, volume_rel(attempt_root_vol))


def rung_next_producer(L, row: Mapping[str, Any], ctx: Mapping[str, Any], *, vol, attempt_id: str, attempt_root_vol: str,
                       entries: Sequence[Mapping[str, Any]]):
    """The rung container's OWN producer side for the queue above it, REBUILT from the attempt-unique queue files its consumer
    wrote (the consumer ran WITHOUT `--finish-producer`, so the app owns COMPLETE -> inventory -> acceptance -> marker): a
    `Producer` for `spec.next_spec([this rung's shard])` whose sealed entry set is the enumerated files."""
    lad = row["ladder"]
    W7, wave_uuid = ctx["W"][:7], ctx["wave_uuid"]
    this_spec = L.QueueSpec.build(W7, wave_uuid, row["leg"], row["engine"], lad["kind"], int(lad["consumes"]), list(lad["feeder_shards"]))
    nxt = this_spec.next_spec(list(lad["shards"]))
    if nxt is None:
        return None
    ref = next(p for p in nxt.producers if p.shard == str(lad["shard"]))
    prod = L.Producer(vol, nxt, ref, attempt_id, volume_rel(attempt_root_vol))
    prod.entries.extend(dict(e) for e in entries)
    return prod


def rung_entries_from_volume(vol_root: Path, ladder_root_rel: str, attempt_id: str) -> List[Dict[str, Any]]:
    """Every `queue/<arm>__s<s>/<item_key>/<attempt_id>.json` under the NEXT queue's root that THIS attempt wrote (attempt-unique
    names) -> inventory entries `{item_key, arm_s, slot, path, sha256}` in the ladder's shape."""
    q = Path(vol_root) / ladder_root_rel / "queue"
    out = []
    if q.is_dir():
        for p in sorted(q.glob(f"*/*/{attempt_id}.json")):
            arm_s, item_key = p.parts[-3], p.parts[-2]
            rel = p.relative_to(Path(vol_root)).as_posix()
            out.append({"item_key": item_key, "arm_s": arm_s, "slot": f"{arm_s}/{item_key}", "path": rel, "sha256": sha256_file(p)})
    return out


RUNG_CONSUMER_ARM = "consumer"                                # the rung consumer child's ARM label (path-safe; never a glob character)
#: fx1: the env var naming an interpreter that CARRIES the Modal SDK, set on the consumer child to THIS process's python. The
#: consumer runs under the DRIVER venv (`uv sync --frozen --extra cu128`: no `modal` distribution) with PYTHONPATH=<tree>/src:<tree>/scripts,
#: where a bare `import modal` answers with the repo's scripts/modal namespace package -- all 11 seed-1 rungs died `AttributeError: module
#: 'modal' has no attribute 'Volume'` in `_consume_volume` (2026-09-21). The consumer reaches the wave's Dicts and the mount's commit /
#: reload through ONE helper process under this interpreter (`rung_redraw.SdkBridge` / `modal_sdk_bridge.py`). == `seedrep_ladder.SDK_PYTHON_ENV`
#: (spelled here because the plan module ships alone into the image; tests/test_seedrep_fx1_rung_sdk_bridge.py pins the equality).
RUNG_SDK_PYTHON_ENV = "SEEDREP_MODAL_SDK_PYTHON"
#: fx1b: inside a Modal container the SDK is NOT in the interpreter's site-packages -- modal 1.5.4 mounts it at /pkg and
#: /pkg/sitecustomize.py puts /__modal/deps on sys.path -- so a helper run `-I` (no PYTHONPATH, no site customisation) sees neither. THIS process
#: has them on its sys.path; `sdk_import_roots` derives them from the live interpreter and the child env carries them here, os.pathsep-joined,
#: for the helper to prepend before its `import modal` (`modal_sdk_bridge.SDK_PATH_ENV` spells the same literal; pinned by test).
RUNG_SDK_PATH_ENV = "SEEDREP_MODAL_SDK_PATH"


def sdk_import_exclusions(*, tree: str = "", cwd: Optional[str] = None, here: Path = _HERE) -> List[str]:
    """The path prefixes :func:`sdk_import_roots` drops (fx1c): the staged ``tree`` (the driver venv's PYTHONPATH roots,
    src / scripts, which must never shadow anything in the helper), the CWD, and THIS app's own repo checkout + its ``scripts/`` -- the last two
    ONLY when the file really sits in a ``<repo>/scripts/modal/`` layout (`grading_host_repo_src`'s rule: the grading host launcher). In a container the image
    installs this module FLAT at /root/seedrep_served.py, where ``here.parent.parent`` is ``/`` -- fx1b excluded it, a prefix every absolute path
    is under, so the roots came out EMPTY and the helper refused `hello`. Hence two hard rules: never exclude ``/`` or any
    single-segment prefix (``/root``, ``/pkg``), and compare paths path-aware (`Path.is_relative_to`), never by string prefix."""
    cands: List[str] = [tree, cwd if cwd is not None else os.getcwd()]
    if grading_host_repo_src(here) is not None:                       # the <repo>/scripts/modal/ layout only: the checkout and its scripts/
        try:
            cands += [str(here.parents[1]), str(here.parent)]
        except IndexError:                                      # the image installs this module FLAT (/root/seedrep_served.py): no parents[1] (fc11l's guard shape)
            pass
    out: List[str] = []
    for x in cands:
        if not x:
            continue
        r = Path(x).resolve()
        if len(r.parts) < 3:                                    # "/" (1 part) or a single segment such as /root (2 parts): never a prefix exclusion
            continue
        if str(r) not in out:
            out.append(str(r))
    return out


def sdk_import_roots(*, tree: str = "", modal_module: Any = None, sys_path: Optional[Sequence[str]] = None, cwd: Optional[str] = None,
                     exists: Callable[[str], bool] = os.path.isdir, here: Path = _HERE) -> List[str]:
    """The import roots the consumer's SDK helper needs (fx1b): FIRST the parent of the ``modal`` package directory as THIS process imported it
    (in a container `/pkg`), THEN every sys.path entry of this process, in order, that is an existing directory and is not under one of
    :func:`sdk_import_exclusions` (the staged tree, the CWD, and -- on the grading host layout only -- this app's checkout), not empty; deduplicated.
    The helper prepends them in this order, so its `import modal` resolves exactly as the app's did (`/pkg`, then `/__modal/deps` and the rest).
    An EMPTY result is a named failure at the call site (`rung_consumer_child` logs the sys.path and the exclusions), never a silent one."""
    mod = sys.modules.get("modal") if modal_module is None else modal_module
    entries: List[str] = []
    f = getattr(mod, "__file__", None)
    if isinstance(f, str) and f:
        entries.append(str(Path(f).resolve().parent.parent))
    entries += [str(p) for p in (sys.path if sys_path is None else sys_path)]
    excluded = [Path(x) for x in sdk_import_exclusions(tree=tree, cwd=cwd, here=here)]
    out: List[str] = []
    for e in entries:
        if not e or e == ".":
            continue
        r = Path(e).resolve()
        if any(r == x or r.is_relative_to(x) for x in excluded):
            continue
        if not exists(str(r)) or str(r) in out:
            continue
        out.append(str(r))
    return out


def child_log_name(kid: Mapping[str, Any]) -> str:
    """ONE encoder for a child's log / argv file stem under `<attempt root>/children/` (`<eval_leg>__<arm>__s<s>`, path-safe): the
    spawn WRITES it and the rung tally lookup READS it (the spawn wrote `<leg>__*__s0.log` while the lookup asked for
    `<leg>_____s0.log`, so no producer rung ever found its tally).

    A GOAL-SHARDED child (`_static_children`: `goal_shard` k of `goal_nshards` n > 1, the `--shard k --nshards n` drivers of one static
    arm-run) gets `__shard<k>` appended: previously the 5 shard children of one static standup shared ONE `<leg>__<arm>__s<s>` stem,
    so their logs interleaved in one file and the `.argv.json` was the last writer's. An unsharded child (n == 1, or no shard axis:
    controls, BoN, misalign, multi-turn, the rung consumer) keeps the plain stem, so every existing reader's path is unchanged."""
    stem = f"{kid['eval_leg']}__{kid['arm']}__s{kid['s']}"
    nsh = kid.get("goal_nshards")
    if nsh is not None and int(nsh) > 1:
        stem += f"__shard{int(kid['goal_shard'])}"
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem)


def rung_consumer_child(row: Mapping[str, Any], consumer_argv: Sequence[str], *, attempt_root: str = "", attempt_id: str = "",
                        tree: str = "") -> Dict[str, Any]:
    """The ONE child a rung container runs: the consumer (its placeholders already resolved), labelled :data:`RUNG_CONSUMER_ARM`,
    spawned with the request-evidence directory and the attempt spool under THIS attempt root (`lp.rung_child_env`)
    and the rung's arm -> seed-base map beside it, so every draw's evidence and identity land under the attempt root."""
    env = (lp.rung_child_env(attempt_root=attempt_root, attempt_id=attempt_id, standup_id=row["standup_id"], leg=row["leg"], seed_bases=list(row.get("seed_bases") or []))
           if attempt_root else {})
    env[RUNG_SDK_PYTHON_ENV] = sys.executable          # fx1: the container's SDK-bearing interpreter (this one), for the consumer's bridge
    roots = sdk_import_roots(tree=tree)                # fx1b: ... and the import roots its `-I` helper must prepend to reach the SDK
    if not roots:                                      # fx1c: never silent -- the helper will refuse `hello` by name; say here WHY nothing was derived
        print(f"[rung-consumer] WARNING: no Modal SDK import roots derived for the consumer's helper ({RUNG_SDK_PATH_ENV} empty): "
              f"modal={getattr(sys.modules.get('modal'), '__file__', None)!r} sys.path={list(sys.path)!r} "
              f"excluded={sdk_import_exclusions(tree=tree)!r} here={str(_HERE)!r} tree={tree!r}", flush=True)
    env[RUNG_SDK_PATH_ENV] = os.pathsep.join(roots)
    return {"eval_leg": row["eval_legs"][0], "arm": RUNG_CONSUMER_ARM, "s": 0, "argv": list(consumer_argv), "concurrency_rule": "S", "env": env,
            "registry_identity_rel": None, "arm_runs": [dict(a) for a in (row.get("arm_runs") or [])]}


def teardown_siblings(procs: Sequence[Tuple[dict, Any, Any]], pending: List[Any], *, kill: Optional[Callable[[Any], Any]] = None) -> List[str]:
    """SIGTERM every live sibling's process group and drop every pending child; returns the names signalled (the caller's poll
    loop collects their exits and the row fails by name -- never a container idling to its deadline)."""
    kill = kill or (lambda p: os.killpg(p.pid, signal.SIGTERM))
    names = []
    for k, p, _ in list(procs):
        try:
            kill(p)
        except ProcessLookupError:
            pass
        names.append(child_log_name(k))
    pending.clear()
    return names


REATTEMPT_LEDGER_SCHEMA = "seedrep_reattempt_ledger/2"          # /2: entries and intents carry launch_plan_sha
#: a re-attempt is published in THREE durable steps on the wave's `reattempts` Dict, the ONE atomic primitive Modal gives
#: (`put(skip_if_exists=True)`): (1) the INTENT for (standup, failed call) is CLAIMED -- exactly one pass wins the put, a
#: concurrent pass skips; (2) the standup entry's counter is written; (3) `.spawn()`, then the intent is reconciled with the spawned call
#: id. A pass that dies between (1) and (3) leaves an UNSPAWNED intent whose lease expires; the next pass RECOVERS it (pop + re-put: one
#: winner) and spawns. The container spawned for an intent claims `<intent>|executing` at entry, so even a duplicate spawn of one intent
#: (a crash AFTER `.spawn()` but before (3)) executes at most once -- the second exits `REATTEMPT_DUPLICATE` before any GPU minute.
REATTEMPT_INTENT_LEASE_S = 900
REATTEMPT_INTENT_PREFIX = "intent"
REATTEMPT_DUPLICATE = "REATTEMPT_DUPLICATE"                       # the status a duplicate execution of one intent exits with; never a standup failure


def reattempt_intent_key(standup_id: str, failed_call_id: str) -> str:
    return f"{REATTEMPT_INTENT_PREFIX}|{standup_id}|{failed_call_id}"


def reattempt_executing_key(intent_key: str) -> str:
    return f"{intent_key}|executing"


def _ledger_json(raw: Any) -> Dict[str, Any]:
    return json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else dict(raw)


def reattempt_ledger_get(ledger: Any, standup_id: str) -> Dict[str, Any]:
    """The standup's durable reattempt entry (the wave's `reattempts` Dict): the counter, the HOLD, every call judged (idempotent passes), the plan
    it belongs to (`launch_plan_sha`, None until the first write)."""
    raw = ledger.get(standup_id)
    if raw is None:
        return {"schema": REATTEMPT_LEDGER_SCHEMA, "standup_id": standup_id, "launch_plan_sha": None, "reattempts": 0, "held": False, "hold_reason": None,
                "held_at": None, "judged_calls": [], "history": []}
    return _ledger_json(raw)


def reattempt_ledger_put(ledger: Any, entry: Mapping[str, Any]) -> None:
    ledger.put(entry["standup_id"], json.dumps(entry))


def claim_reattempt_intent(ledger: Any, *, standup_id: str, failed_call_id: str, launch_plan_sha: str, wave_uuid: str, reattempt_n: int,
                           pass_id: str, now: float) -> Optional[Dict[str, Any]]:
    """Step (1): ONE `put(skip_if_exists)` on the intent key -- the winner gets the intent back, a concurrent pass None (it spawns nothing)."""
    key = reattempt_intent_key(standup_id, failed_call_id)
    intent = {"schema": REATTEMPT_LEDGER_SCHEMA, "kind": "intent", "key": key, "standup_id": standup_id, "failed_call_id": failed_call_id,
              "launch_plan_sha": launch_plan_sha, "wave_uuid": wave_uuid, "reattempt_n": int(reattempt_n), "pass_id": pass_id,
              "claimed_at": float(now), "lease_until": float(now) + REATTEMPT_INTENT_LEASE_S, "spawned_call_id": None, "spawned_at": None,
              "recoveries": 0, "recovered_from": None}
    return intent if ledger.put(key, json.dumps(intent), skip_if_exists=True) else None


def record_spawned_intent(ledger: Any, intent: Dict[str, Any], call_id: str) -> None:
    """Step (3): reconcile the intent with the call it spawned (an overwrite: the intent key is already ours)."""
    intent.update(spawned_call_id=call_id, spawned_at=utc())
    ledger.put(intent["key"], json.dumps(intent))


def stale_unspawned_intents(ledger: Any, *, launch_plan_sha: str, now: float) -> List[Dict[str, Any]]:
    """Every intent of THIS plan whose pass died between claim and spawn: no `spawned_call_id` and the lease has run out. An intent inside its
    lease belongs to a live pass mid-spawn and is left alone."""
    out = []
    for k, raw in list(ledger.items()):
        if not str(k).startswith(REATTEMPT_INTENT_PREFIX + "|") or str(k).endswith("|executing"):
            continue
        try:
            it = _ledger_json(raw)
        except (ValueError, TypeError):
            continue
        if it.get("kind") == "intent" and it.get("spawned_call_id") is None and it.get("launch_plan_sha") == launch_plan_sha and float(it.get("lease_until") or 0) < now:
            out.append(it)
    return sorted(out, key=lambda it: it["key"])


def recover_intent(ledger: Any, intent: Mapping[str, Any], *, pass_id: str, now: float) -> Optional[Dict[str, Any]]:
    """Take over ONE stale intent: `pop` (atomic -- exactly one recoverer gets the value) then re-`put(skip_if_exists)` under this pass's lease;
    None when another recoverer won either step."""
    raw = ledger.pop(intent["key"], None)
    if raw is None:
        return None
    it = _ledger_json(raw)
    if it.get("spawned_call_id") is not None:                     # spawned meanwhile: put it back untouched, nothing to recover
        ledger.put(it["key"], json.dumps(it), skip_if_exists=True)
        return None
    it.update(recovered_from=it.get("pass_id"), pass_id=pass_id, claimed_at=float(now), lease_until=float(now) + REATTEMPT_INTENT_LEASE_S,
              recoveries=int(it.get("recoveries") or 0) + 1)
    return it if ledger.put(it["key"], json.dumps(it), skip_if_exists=True) else None


def fanout_execution_claim(ledger: Any, standup_id: str, nonce: str, *, attempt_id: str, call_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """The CONTAINER's guard for its row's fan-out claim: ONE `put(skip_if_exists)` on `fanout|<standup>|<nonce>|executing`
    carrying THIS container's call id + attempt id -> (won, the executing mark when lost). A duplicate spawn of one claimed row exits
    FANOUT_DUPLICATE before standing anything up; the mark is the durable call list the launcher reconciles against."""
    key = lp.fanout_executing_key(standup_id, nonce)
    mark = {"attempt_id": attempt_id, "call_id": call_id, "utc": utc()}
    if ledger.put(key, json.dumps(mark), skip_if_exists=True):
        return True, None
    return False, fanout_mark(ledger, standup_id, nonce)


def fanout_mark(ledger: Any, standup_id: str, nonce: str) -> Optional[Dict[str, Any]]:
    v = ledger.get(lp.fanout_executing_key(standup_id, nonce))
    if v is None:
        return None
    try:
        return json.loads(v) if isinstance(v, str) else dict(v)
    except (ValueError, TypeError):
        return {"call_id": None, "raw": str(v)}


def reattempt_execution_claim(ledger: Any, intent_key: str, attempt_id: str) -> Tuple[bool, Optional[str]]:
    """The CONTAINER's guard at function entry: ONE `put(skip_if_exists)` on `<intent>|executing` -> (won, the executing attempt when lost).
    A duplicate spawn of one intent exits `REATTEMPT_DUPLICATE` before any GPU minute."""
    key = reattempt_executing_key(intent_key)
    if ledger.put(key, attempt_id, skip_if_exists=True):
        return True, None
    holder = ledger.get(key)
    return False, (str(holder) if holder is not None else None)


def call_failure(snapshot_rec: Mapping[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
    """`(failed, status, error)` of ONE terminal call from the watcher's snapshot record (the same rule the watcher's FAILED event uses)."""
    summ = snapshot_rec.get("summary") if isinstance(snapshot_rec.get("summary"), Mapping) else {}
    err = snapshot_rec.get("error")
    res = str(snapshot_rec.get("result_line") or "")
    status = summ.get("status")
    if status is None and res:
        m = re.search(r"status=(\S+)", res)
        status = m.group(1) if m else None
    failed = bool(err) or (status not in (None, "DONE")) or ("rc=1" in res)
    return failed, status, (str(err) if err else None)


def _judge(entry: Dict[str, Any], cid: str, *, status: Optional[str], err: Optional[str], d: Mapping[str, Any]) -> None:
    if cid not in entry["judged_calls"]:
        entry["judged_calls"].append(cid)
        entry["history"].append({"call_id": cid, "status": status, "error": (err or "")[:200], "class": d.get("class"), "action": d.get("action"), "at": utc()})


def apply_reattempt_policy(plan: Mapping[str, Any], rs: Path, *, calls: Sequence[Mapping[str, Any]], snapshot: Mapping[str, Any],
                           rows_by_sid: Mapping[str, Mapping[str, Any]], ledger: Any, spawn: Callable[[Mapping[str, Any], Mapping[str, Any]], str],
                           log: Callable[[str], Any] = print, now: Optional[float] = None, pass_id: Optional[str] = None,
                           registry_spool: Optional[str] = None) -> List[Dict[str, Any]]:
    """The reattempt policy, one pass, PLAN-BOUND, RACE-SAFE and CRASH-RECOVERABLE. First RECOVER every stale unspawned intent of THIS plan
    (a pass that died between claim and spawn) -- pop + re-put, one winner, then spawn. Then, for every TERMINAL, FAILED call of THIS plan
    (`launch_plan_sha` == the plan's; a call of another plan of the same wave is refused, never retried by a regenerated plan) whose accepted
    key holds no accepted attempt: consult the standup's DURABLE ledger entry (its `launch_plan_sha` must be this plan's or unset), decide
    (`lp.reattempt_decision`), and either CLAIM the intent (one `put(skip_if_exists)`: a concurrent pass finds it held and spawns nothing),
    write the counter, spawn ONCE through ``spawn(row, intent)`` and reconcile the intent with the call id -- or HOLD. A duplicate execution's
    `REATTEMPT_DUPLICATE` exit is never a standup failure. Returns the decisions taken."""
    plan_sha = str(plan.get("launch_plan_sha") or "")
    wave = str(plan["wave_uuid"])
    now = time.time() if now is None else float(now)
    pass_id = pass_id or f"pass-{uuid.uuid4().hex[:12]}"
    decisions: List[Dict[str, Any]] = []
    if not plan_sha:
        log("[reattempt] the plan carries no launch_plan_sha; nothing is retried")
        return decisions
    # --- (A) recovery: intents a dead pass claimed and never spawned ---------------------------------------------------------------
    for stale in stale_unspawned_intents(ledger, launch_plan_sha=plan_sha, now=now):
        sid = stale["standup_id"]
        if sid not in rows_by_sid:
            log(f"[reattempt] stale intent {stale['key']} names a standup not in this plan; left for the operator")
            continue
        it = recover_intent(ledger, stale, pass_id=pass_id, now=now)
        if it is None:
            continue                                                # another recoverer won
        entry = reattempt_ledger_get(ledger, sid)
        entry["launch_plan_sha"] = entry.get("launch_plan_sha") or plan_sha
        entry["reattempts"] = max(int(entry.get("reattempts") or 0), int(it["reattempt_n"]))
        _judge(entry, it["failed_call_id"], status=None, err="recovered unspawned intent", d={"class": "infra", "action": "reattempt"})
        reattempt_ledger_put(ledger, entry)
        new_cid = spawn(rows_by_sid[sid], it)
        record_spawned_intent(ledger, it, new_cid)
        entry["history"][-1]["reattempt_call_id"] = new_cid
        reattempt_ledger_put(ledger, entry)
        _legs_log(rs, f"REATTEMPT {sid} RECOVERED intent {it['key']} (pass {it['recovered_from']} died before spawning) -> call={new_cid} (D73: {it['reattempt_n']}/{lp.D73_MAX_AUTO_REATTEMPTS})")
        decisions.append({"standup_id": sid, "call_id": it["failed_call_id"], "action": "reattempt", "class": "infra", "reattempts": it["reattempt_n"],
                          "recovered": True, "reattempt_call_id": new_cid, "intent": it["key"]})
    # --- (B) the failed calls of THIS plan ------------------------------------------------------------------------------------------
    for c in calls:
        cid, sid = c["call_id"], c["standup_id"]
        if c.get("launch_plan_sha") != plan_sha:
            log(f"[reattempt] {sid}: call {cid} was spawned from plan {c.get('launch_plan_sha')!r}, not this plan {plan_sha!r}; a regenerated plan never retries another plan's failure")
            decisions.append({"standup_id": sid, "call_id": cid, "action": "refused", "class": None, "reattempts": None, "reason": "call of another plan"})
            continue
        s = snapshot.get(cid) or {}
        if not s.get("done"):
            continue
        failed, status, err = call_failure(s)
        if not failed or s.get("accepted"):
            continue
        if str(status or "").upper() == REATTEMPT_DUPLICATE:
            continue                                                # a duplicate execution of one intent exited before any GPU minute; not a failure
        entry = reattempt_ledger_get(ledger, sid)
        if cid in entry["judged_calls"]:
            if entry.get("held") and not entry.get(LEDGER_TERMINAL_KEY) and sid in rows_by_sid and cid == (entry["judged_calls"] or [cid])[-1]:
                # fr6 recovery: the HOLD was decided and written but its terminal states were not (a pass that died between the two
                # ledger writes) -- emit them now, onto the same `held_at`, and record it
                entry[LEDGER_TERMINAL_KEY] = _hold_terminal(rows_by_sid[sid], rs=rs, plan=plan, entry=entry, call_id=cid, snap=s, spool=registry_spool, log=log)
                reattempt_ledger_put(ledger, entry)
            continue
        if sid not in rows_by_sid:
            log(f"[reattempt] {sid}: call {cid} names a standup not in this plan; ignored")
            continue
        if entry.get("launch_plan_sha") not in (None, plan_sha):
            log(f"[reattempt] {sid}: the durable ledger entry belongs to plan {entry.get('launch_plan_sha')!r}, not {plan_sha!r}; refusing to act on it")
            decisions.append({"standup_id": sid, "call_id": cid, "action": "refused", "class": None, "reattempts": entry.get("reattempts"), "reason": "ledger entry of another plan"})
            continue
        d = lp.reattempt_decision(entry, status=status, error=err)
        entry["launch_plan_sha"] = plan_sha
        if d["action"] == "reattempt":
            intent = claim_reattempt_intent(ledger, standup_id=sid, failed_call_id=cid, launch_plan_sha=plan_sha, wave_uuid=wave, reattempt_n=int(d["reattempts"]),
                                            pass_id=pass_id, now=now)
            if intent is None:
                log(f"[reattempt] {sid}: call {cid}'s re-attempt is already claimed by another pass; nothing spawned here")
                continue
            _judge(entry, cid, status=status, err=err, d=d)
            entry["reattempts"] = int(d["reattempts"])
            reattempt_ledger_put(ledger, entry)                                # DURABLE before the spawn
            new_cid = spawn(rows_by_sid[sid], intent)
            record_spawned_intent(ledger, intent, new_cid)
            entry["history"][-1]["reattempt_call_id"] = new_cid
            reattempt_ledger_put(ledger, entry)
            _legs_log(rs, f"REATTEMPT {sid} after call={cid} ({d['class']} {status or err}) -> call={new_cid} (D73: {d['reattempts']}/{lp.D73_MAX_AUTO_REATTEMPTS}; intent {intent['key']})")
            log(f"[reattempt] {sid}: {d['reason']} -> call {new_cid}")
            decisions.append({"standup_id": sid, "call_id": cid, **d, "reattempt_call_id": new_cid, "intent": intent["key"]})
        else:
            _judge(entry, cid, status=status, err=err, d=d)
            entry.update(held=True, hold_reason=d["reason"], held_at=utc())
            reattempt_ledger_put(ledger, entry)                                # the HOLD is durable FIRST (the authority for what follows)
            _legs_log(rs, f"HOLD {sid} call={cid} {d['reason']}")
            log(f"[reattempt] HOLD {sid}: {d['reason']}")
            # fr6: the LOGICAL runs of a HELD standup end here, on the grading host, metadata only -- the failed attempt's
            # private spool never reaches the registry; a re-attempt that succeeded would have ended them `generated` instead
            entry[LEDGER_TERMINAL_KEY] = _hold_terminal(rows_by_sid[sid], rs=rs, plan=plan, entry=entry, call_id=cid, snap=s, spool=registry_spool, log=log)
            reattempt_ledger_put(ledger, entry)
            decisions.append({"standup_id": sid, "call_id": cid, **d, LEDGER_TERMINAL_KEY: entry[LEDGER_TERMINAL_KEY]})
    return decisions


def _hold_terminal(row: Mapping[str, Any], *, rs: Path, plan: Mapping[str, Any], entry: Mapping[str, Any], call_id: str, snap: Mapping[str, Any],
                   spool: Optional[str], log: Callable[[str], Any]) -> Dict[str, Any]:
    """`registry_hold_terminal_states` for one HELD standup, never raising into the policy pass; ONE line says what was (not) filed."""
    failed, status, err = call_failure(snap)
    try:
        res = registry_hold_terminal_states(row, rs=rs, plan=plan, entry=entry, call_id=call_id, status=status, error=err,
                                            summary=snap.get("summary") if isinstance(snap.get("summary"), Mapping) else None, spool=spool)
    except Exception as e:  # noqa: BLE001 -- the HOLD stands; the registry gap is said
        res = {"state": hold_terminal_state(status), "call_id": call_id, "runs": 0, "records": [], "problems": [f"{type(e).__name__}: {e}"], "spool": spool}
    line = f"[registry] HOLD {row.get('standup_id')}: {res['runs']} run(s) -> run_state {res['state']} ({res.get('spool')})"
    if res["problems"]:
        line += f"; {len(res['problems'])} NOT filed: {res['problems'][:2]}"
    log(line)
    return res


def wave_calls(rs: Path, plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every production call of THIS wave AND THIS plan (`launch_plan_sha`) in RS/calls.jsonl (smoke calls excluded; a call spawned from another
    plan of the same wave is excluded -- a regenerated same-wave plan never retries an old failure)."""
    p = Path(rs) / "calls.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        if c.get("wave_uuid") == plan["wave_uuid"] and not c.get("smoke") and c.get("launch_plan_sha") == plan.get("launch_plan_sha"):
            out.append(c)
    return out


def reattempt_pass(plan: Mapping[str, Any], rs: Path, app_name: str, pool_urls: Optional[Mapping[str, str]],
                   certificate: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """The grading host's `--reattempt` step -- gated by the launch certificate in `main`: the watcher's live snapshot of every
    call of this plan -> the policy pass over the wave's durable `reattempts` Dict, spawning through :func:`spawn_row` with the claimed intent
    (a NEW attempt under the existing attempt rules; the container claims the intent's `executing` key at entry)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import seedrep_watch as sw  # noqa: E402
    calls = wave_calls(rs, plan)
    backend = sw.ModalBackend(plan)
    snap = backend.snapshot(calls)
    ledger = modal.Dict.from_name(plan["dicts"]["reattempts"], create_if_missing=True)
    rows_by_sid = {r["standup_id"]: r for r in list(plan["standups"]) + list(plan["rungs"])}
    def guarded_spawn(row, intent):
        if certificate is not None:
            certified_launch_guard(certificate, [row["standup_id"]], where="re-attempt spawn")   # fc9b: the same guard at this spawn boundary
        return spawn_row(row, plan, app_name, rs, pool_urls, reattempt_intent=intent["key"])
    return apply_reattempt_policy(plan, rs, calls=calls, snapshot=snap, rows_by_sid=rows_by_sid, ledger=ledger, spawn=guarded_spawn)


def request_evidence_files(root: Path) -> List[Dict[str, Any]]:
    """The evidence files a rung consumer wrote AT the attempt root (two streams at the literal root, beside the proxy's
    `usage_ledger.jsonl` and the engine log): path (root-relative), rows, sha256 -- recorded in the rung attempt's `ladder_enqueued.json`
    and COMPLETE summary so the pull / audit read them with the inventory."""
    d = Path(root)
    out = []
    for name in lp.REQUEST_EVIDENCE_FILES:
        p = d / name
        if p.is_file():
            data = p.read_bytes()
            out.append({"path": name, "rows": sum(1 for line in data.splitlines() if line.strip()),
                        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    return out


def consumer_tally_from_log(log_path: Path) -> Optional[dict]:
    """The consumer prints its tally as the LAST JSON object on stdout (`rung_redraw.main`); read it back off the child log."""
    if not Path(log_path).is_file():
        return None
    text = Path(log_path).read_text(errors="replace")
    i = text.rfind("\n{")
    if i < 0:
        i = 0 if text.startswith("{") else -1
    if i < 0:
        return None
    try:
        return json.loads(text[i:].strip())
    except ValueError:
        return None


def misalign_messages_via_helper(driver_python: str, tree: Path, cells_json: Path, requests: Sequence[Mapping[str, Any]],
                                 env: Mapping[str, str]) -> Dict[str, Any]:
    """The driver-venv helper (`scripts/modal/seedrep_ladder_helpers.py misalign-messages`): the runner's own
    `load_items` / `si_text_for` / `build_messages` for every requested (cell, behaviour, item) -> `{key: messages}`."""
    with tempfile.TemporaryDirectory(prefix="seedrep-msgs-") as td:
        req = Path(td) / "requests.json"
        out = Path(td) / "messages.json"
        req.write_text(json.dumps(list(requests)))
        r = subprocess.run([driver_python, str(Path(tree) / "scripts/modal/seedrep_ladder_helpers.py"), "misalign-messages",
                            "--family", "agentic", "--cells", str(cells_json), "--requests", str(req), "--out", str(out)],
                           capture_output=True, text=True, env=dict(env), cwd=str(tree), timeout=1800)
        if r.returncode != 0:
            raise LadderUnavailable(f"misalign-messages helper rc {r.returncode}: {(r.stdout + r.stderr)[-800:]}")
        return json.loads(out.read_text())


# --------------------------------------------------------------------------- #
# the Petri proxy's completion evidence (the drive contract)                     #
# --------------------------------------------------------------------------- #

STOP_SENTINEL_RE = re.compile(r"^STOP\s+(?P<utc>\S+)\s+drive=(?P<drive>\S+)(?P<extra>.*)$", re.S)


def petri_tunnel_record(*, url: str, host: str, port: int, proxy_port: int, relay_port: int, call_id: str, task_id: str,
                        attempt_id: str, attempt_root_vol: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    """`modal_tunnel.json` as `seedrep_petri_drive` REQUIRES it (`TUNNEL_REQUIRED_KEYS`, `tunnel_record_problems`,
    `attempt_paths`): `url` (https, modal.forward), `function_call_id`, `proxy_port`, `serving_dir_on_volume` (under
    `<attempt root>/exp/`), `attempt_root_on_volume` (under /vol/seedrep/, no trailing slash), `attempt_id` (path-safe, == the
    root's last component), `view`, `stage`, plus `control_key` (the six axes in the drive's `/` spelling -- its `key_for("")`),
    `task_id` and `container_exp_root` (the container path the endpoint's file paths sit under, mapped onto `<root>/exp`)."""
    root = attempt_root_vol.rstrip("/")
    if Path(root).name != attempt_id:
        raise ValueError(f"attempt root {root} does not end in the attempt id {attempt_id}")
    return {"url": url, "host": host, "port": port, "function_call_id": call_id, "task_id": task_id, "proxy_port": int(proxy_port),
            "relay_port": int(relay_port), "serving_dir_on_volume": f"{root}/{SERVE_DIR}", "attempt_root_on_volume": root,
            "attempt_id": attempt_id, "container_exp_root": CONTAINER_EXP_ROOT, "view": row["engine"], "stage": row["stage"],
            "shard": str(row["shard"]), "control_key": row["control_key"], "standup_id": row["standup_id"], "written_at": utc()}


STOP_COUNT_KEYS = ("complete", "incomplete", "failed")       # the ONLY k=v tokens read as integers; ids and reasons stay strings


def parse_stop_sentinel(content: Any) -> Optional[Dict[str, Any]]:
    """`sentinel_content("STOP", drive_id, "complete=N incomplete=N failed=N attempt=<id>[ reason=<why>]")` (the
    completion contract). Only the three COUNT keys are coerced to int; `attempt=`, `reason=` and `drive=` stay strings, so an all-digit
    attempt id can never be int-coerced into a mismatch with the proxy's str id."""
    text = content.decode(errors="replace") if isinstance(content, (bytes, bytearray)) else str(content or "")
    m = STOP_SENTINEL_RE.match(text.strip())
    if not m:
        return None
    out: Dict[str, Any] = {"utc": m.group("utc"), "drive_id": m.group("drive")}
    for tok in m.group("extra").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = int(v) if (k in STOP_COUNT_KEYS and v.isdigit()) else v
    return out


#: the drive's completion contract, REVISED: ONE atomic bundle at `<K>/<attempt_id>/completion.json`
#: (`seedrep_petri_drive.completion_key`), schema `seedrep_petri_drive_completion/1` = every round-3 verdict field + `stop: {content, why}`
#: (the exact STOP sentinel). The generic `<K>/drive_verdict.json` + `<K>/STOP` pair is written ONLY under `--legacy-generic-completion`
#: (default OFF): a proxy watching only those keys would wait until its deadline. THIS attempt watches its own key and nothing else; a
#: bundle of another attempt sits under another key and is never evidence for this one; anything under the generic keys is ignorable.
PETRI_COMPLETION_NAME = "completion.json"
PETRI_COMPLETION_SCHEMA = "seedrep_petri_drive_completion/1"


def petri_completion_key(key: str, attempt_id: str) -> str:
    """`<K>/<attempt_id>/completion.json` -- the ONE key this attempt watches (== the drive's `completion_key(K, attempt_id)`, pinned by test)."""
    if not isinstance(key, str) or not key or key.endswith("/"):
        raise ValueError(f"petri_completion_key needs the six-axis key, got {key!r}")
    if not isinstance(attempt_id, str) or not attempt_id or "/" in attempt_id:
        raise ValueError(f"petri_completion_key needs a path-safe attempt id, got {attempt_id!r}")
    return f"{key}/{attempt_id}/{PETRI_COMPLETION_NAME}"


def petri_bound_completion(sentinels: Any, key: str, attempt_id: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """`(the completion bundle bound to THIS attempt | None, noise)`: the bundle under this attempt's key, when it is JSON, carries the
    schema, names this attempt and this control key. Anything else -- a non-JSON value, a bundle naming another attempt or key, or the
    generic (legacy) pair -- is NOISE, never terminal."""
    ck = petri_completion_key(key, attempt_id)
    noise: List[str] = []
    raw = sentinels.get(ck)
    if raw is None:
        for name in (PETRI_VERDICT_NAME, "STOP"):
            if sentinels.get(f"{key}/{name}") is not None:
                noise.append(f"{key}/{name}: a generic (legacy) completion key -- not this attempt's evidence (WS2-F 1e132e41: only {ck} is read); ignored")
        return None, noise
    try:
        bundle = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else dict(raw)
    except (ValueError, TypeError):
        return None, [f"{ck}: the completion bundle is not JSON (ignored)"]
    if not isinstance(bundle, dict):
        return None, [f"{ck}: the completion bundle is not an object (ignored)"]
    problems = []
    if bundle.get("schema") != PETRI_COMPLETION_SCHEMA:
        problems.append(f"schema {bundle.get('schema')!r} != {PETRI_COMPLETION_SCHEMA}")
    if bundle.get("attempt_id") != attempt_id:
        problems.append(f"names attempt {bundle.get('attempt_id')!r}, not {attempt_id}")
    if bundle.get("control_key") != key:
        problems.append(f"control_key {bundle.get('control_key')!r} != {key}")
    if problems:
        return None, [f"{ck}: the bundle under this attempt's key is not this attempt's completion ({'; '.join(problems)}); ignored"]
    return bundle, noise


def petri_completion_evidence(sentinels: Any, key: str, attempt_id: str) -> Dict[str, Any]:
    """Success evidence for a Petri proxy attempt = the EXTERNAL drive's reconciled completion bundle under THIS attempt's key
    (`petri_completion_key`): `verdict.release` true with `verdict.counts.failed == 0` -> `complete`; any other bound bundle (a
    `--teardown-after-failure`, a withheld release) -> `incomplete`; no bound bundle -> `waiting`, with every ignored thing named under
    `noise` (another attempt's evidence must never tear this proxy down). Never a timeout, never a cancellation."""
    ck = petri_completion_key(key, attempt_id)
    bundle, noise = petri_bound_completion(sentinels, key, attempt_id)
    if bundle is not None:
        verdict = bundle.get("verdict") or {}
        counts = verdict.get("counts") or {}
        if verdict.get("release") is True and int(counts.get("failed", 1)) == 0:
            return {"state": "complete", "why": "completion bundle bound to this attempt: release with zero failed", "evidence": bundle, "kind": "completion",
                    "key": ck, "noise": noise}
        return {"state": "incomplete", "why": f"completion bundle bound to this attempt: counts {counts}, release {verdict.get('release')!r}, "
                                              f"stop_why {bundle.get('stop_why')!r}", "evidence": bundle, "kind": "completion", "key": ck, "noise": noise}
    return {"state": "waiting", "why": (f"no {PETRI_COMPLETION_NAME} under {ck} yet" if not noise else f"nothing terminal under {ck}: {noise[-1]}"),
            "evidence": None, "noise": noise}


def mirror_bound_completion(sentinels: Any, serve_dir: Path, key: str, attempt_id: str) -> List[str]:
    """Mirror into the serve dir ONLY the completion bound to THIS attempt: the bundle as `completion.json` FIRST, then `STOP` = the bundle's
    exact sentinel content (the verdict before the STOP); another attempt's records are never copied. Returns the
    file names written, in order."""
    bundle, _ = petri_bound_completion(sentinels, key, attempt_id)
    if bundle is None:
        return []
    (Path(serve_dir) / PETRI_COMPLETION_NAME).write_text(json.dumps(bundle, indent=1))
    written = [PETRI_COMPLETION_NAME]
    content = (bundle.get("stop") or {}).get("content")
    if content:
        (Path(serve_dir) / "STOP").write_text(str(content))
        written.append("STOP")
    return written


def clear_stale_sentinels(sentinels: Any, key: str, names: Sequence[str] = PETRI_STALE_SENTINELS) -> List[str]:
    """Before this attempt publishes its tunnel record: pop every sentinel a PRIOR attempt of this row left under the key
    (a stale STOP would otherwise end the new proxy at once), so any STOP that appears afterwards follows a drive that read
    THIS record. Returns the names popped."""
    popped = []
    for name in names:
        try:
            if sentinels.get(f"{key}/{name}") is not None:
                sentinels.pop(f"{key}/{name}")
                popped.append(name)
        except Exception:  # noqa: BLE001
            continue
    return popped


# --------------------------------------------------------------------------- #
# the image (strength's two-venv image, byte for byte in its recipe)            #
# --------------------------------------------------------------------------- #

BUILD = "/opt/seedrep-build"
if modal.is_local():
    LOCAL_DEP_HASHES = dependency_hashes(LOCAL_REPO)
    _REQS_SHA = LOCAL_DEP_HASHES[VLLM_REQS_REL]
else:
    LOCAL_DEP_HASHES = {}
    _REQS_SHA = "0" * 64
SERVE_VENV, SERVE_SENTINEL, SERVE_SENTINEL_VALUE = serving_venv_names(_REQS_SHA)
SERVING_VENV_CMD = (
    f"mkdir -p {VENVS_LOCAL} && uv venv --python {PY_VERSION} {SERVE_VENV} && "
    f"uv pip install --python {SERVE_VENV}/bin/python --index-strategy unsafe-best-match "
    f"--exclude-newer 2026-08-03T00:00:00Z -r {BUILD}/vllm-requirements.txt && "
    f"echo {SERVE_SENTINEL_VALUE} > {SERVE_SENTINEL}"
)
DRIVER_PREWARM_CMD = (
    f"mkdir -p {BUILD}/proj/src/sisteer && printf '__version__ = \"0.1.0\"\\n' > {BUILD}/proj/src/sisteer/__init__.py && "
    f"cd {BUILD}/proj && UV_PROJECT_ENVIRONMENT=/opt/prewarm-driver uv sync --frozen --extra cu128 && "
    "/opt/prewarm-driver/bin/python -c 'import torch, pyrit, yaml; print(\"driver prewarm OK\", torch.__version__, pyrit.__version__)' && "
    "rm -rf /opt/prewarm-driver"
)


def serving_venv_check_cmd(venv: str, sentinel: str) -> str:
    return (f"{venv}/bin/python -c \"import vllm, torch, transformers; "
            "assert vllm.__version__.startswith('0.25.1'), vllm.__version__; "
            "assert torch.__version__ == '2.11.0+cu129', torch.__version__; "
            "assert transformers.__version__ == '5.14.1', transformers.__version__; "
            f"print('SERVING VENV OK', vllm.__version__, torch.__version__)\" && test -x {venv}/bin/vllm && cat {sentinel}")


#: the images' own pip deps are pinned to the EXACT versions the repo's uv.lock resolves (never a floating latest); the
#: pins are part of the image provenance (`IMAGE_PIP_PINS` rides `RS/modal_image.json` and `--pf` re-asserts the installed versions)
IMAGE_PIP_PINS: Dict[str, str] = {"pyyaml": "6.0.3", "httpx": "0.28.1"}
_PIN = lambda name: f"{name}=={IMAGE_PIP_PINS[name]}"  # noqa: E731

cpu_image = (
    modal.Image.debian_slim(python_version=PY_VERSION)
    .apt_install("git")
    .pip_install(_PIN("pyyaml"))
    .run_commands("git config --global --add safe.directory '*'")
    .env({"USER": CONTAINER_USER, "LOGNAME": CONTAINER_USER})
    .add_local_file(str(_HERE / "seedrep_launch_plan.py"), "/root/seedrep_launch_plan.py", copy=True)
    .add_local_file(str(_HERE / "seedrep_carry_over.py"), "/root/seedrep_carry_over.py", copy=True)      # fj2: importable in-container beside the plan module
)
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python=PY_VERSION)
    .apt_install("git", "build-essential", "curl", "unixodbc", "util-linux", "procps")
    .env({"CUDA_HOME": "/usr/local/cuda", "USER": CONTAINER_USER, "LOGNAME": CONTAINER_USER,
          "UV_CACHE_DIR": UV_CACHE, "UV_LINK_MODE": "copy", "UV_PYTHON_INSTALL_DIR": UV_PYTHONS,
          "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    .pip_install(f"uv=={UV_PIN}", _PIN("pyyaml"), _PIN("httpx"))
    .run_commands(f"uv python install {PY_VERSION}", "git config --global --add safe.directory '*'")
    .add_local_file(str(LOCAL_REPO / "pyproject.toml"), f"{BUILD}/proj/pyproject.toml", copy=True)
    .add_local_file(str(LOCAL_REPO / "uv.lock"), f"{BUILD}/proj/uv.lock", copy=True)
    .add_local_file(str(LOCAL_REPO / VLLM_REQS_REL), f"{BUILD}/vllm-requirements.txt", copy=True)
    .run_commands(DRIVER_PREWARM_CMD)
    .run_commands(SERVING_VENV_CMD, serving_venv_check_cmd(SERVE_VENV, SERVE_SENTINEL))
    .add_local_file(str(_HERE / "seedrep_launch_plan.py"), "/root/seedrep_launch_plan.py", copy=True)
    .add_local_file(str(_HERE / "seedrep_carry_over.py"), "/root/seedrep_carry_over.py", copy=True)      # fj2: importable in-container beside the plan module
)


# --------------------------------------------------------------------------- #
# staging (CPU): the tree, the bank, the identity plans, the preflight           #
# --------------------------------------------------------------------------- #

def _git(repo, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", f"safe.directory={Path(repo).resolve()}", *args], cwd=str(repo),
                          capture_output=True, text=True, timeout=900)


def checkout_facts(repo, sha: str) -> dict:
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    clean = bool(head) and _git(repo, "diff", "--quiet", "HEAD", "--").returncode == 0
    ok = bool(head) and head == sha and clean
    why = None if ok else ("not a git checkout" if not head else f"HEAD {head} != {sha}" if head != sha else "modified tracked files")
    return {"ok": ok, "head": head or None, "clean": clean, "why": why}


def bank_entries(bank_root) -> List[dict]:
    """Every view entry under ``<bank>/cells_view/<view>/VIEW_MANIFEST.json`` with the on-disk facts (strength's reader)."""
    out = []
    for vm in sorted(Path(bank_root).glob("cells_view/*/VIEW_MANIFEST.json")):
        m = json.loads(vm.read_text())
        for e in m["entries"]:
            p = vm.parent / e["cell_id"]
            rec = {"target": m["target"], "view": m["view"], "cell_id": e["cell_id"], "staged_id": e["staged_id"],
                   "manifest_sha256": e["object_sha256"], "is_symlink": p.is_symlink(),
                   "link_target": os.readlink(p) if p.is_symlink() else None, "resolves": (p / "object.pt").is_file()}
            rec["object_sha256"] = sha256_file(p / "object.pt") if rec["resolves"] else None
            rec["sha_ok"] = rec["object_sha256"] == e["object_sha256"]
            out.append(rec)
    return out


def verify_staged_objects(bank_root: str, arditi_root: str, pins: Mapping[str, Any], arditi_pins: Mapping[str, Any],
                          extra_views: Optional[Mapping[str, Mapping[str, str]]] = None, extra_pins: Optional[Mapping[str, str]] = None) -> dict:
    """The 16 shas -- the three s-views' 15 entries against configs/sdfbase_full22_pins.json and the Arditi view's one against
    configs/arditi_allpos_pins.json -- PLUS every extra view the PLAN references (`extra_views` = {view: {cell: staged_id}} from
    `lp.bank_views_for_plan`: the S5 probe rows' views, the dormant path's dose views) whose
    objects are pinned by the PLAN (`extra_pins` = {staged_id: object_sha256} from `lp.bank_pins_for_plan`; the tree's pin file never
    carries a peer object -- a tree pin that disagrees with the plan's refuses): symlink resolves, sha256(object.pt) == VIEW_MANIFEST
    == pin, every declared cell present under its staged id. FATAL for a container on any mismatch (§4.1). Returns ``{ok, entries,
    problems, shas}``; ``shas`` has 16 + the extra objects."""
    problems, shas = [], {}
    extra = {str(v): {str(c): str(sid) for c, sid in cells.items()} for v, cells in (extra_views or {}).items()}
    want_views = tuple(lp.VIEWS) + tuple(sorted(extra))
    entries = [e for e in bank_entries(f"{bank_root}/{lp.MODEL_KEY}") if e["view"] in want_views]
    tp = dict(pins.get(lp.MODEL_KEY) or {})
    for sid, sha in sorted((extra_pins or {}).items()):
        tree_pin = (tp.get(sid) or {}).get("object_sha256")
        if tree_pin is not None and tree_pin != sha:
            problems.append({"staged_id": sid, "tree_pin": tree_pin, "plan_pin": sha, "why": "the tree's pin file and the plan pin this object under two shas"})
        tp.setdefault(sid, {"object_sha256": sha})
    for e in entries:
        pin = (tp.get(e["staged_id"]) or {}).get("object_sha256")
        ok = e["is_symlink"] and e["resolves"] and e["sha_ok"] and pin == e["object_sha256"]
        if not ok:
            problems.append({k: e.get(k) for k in ("view", "cell_id", "staged_id", "is_symlink", "resolves", "sha_ok")} | {"pin": pin})
        shas[e["staged_id"]] = e["object_sha256"]
    for v in lp.VIEWS:
        got = sorted(e["cell_id"] for e in entries if e["view"] == v)
        if got != sorted(lp.FITTED_CELLS):
            problems.append({"view": v, "cells": got, "want": sorted(lp.FITTED_CELLS)})
    for v, cells in sorted(extra.items()):
        got = {e["cell_id"]: e["staged_id"] for e in entries if e["view"] == v}
        if got != cells:
            problems.append({"view": v, "cells": got, "want": cells, "why": "a plan-referenced dose view is absent from the bank or serves other objects"})
    ar = [e for e in bank_entries(arditi_root) if e["view"] == lp.ARDITI_VIEW and e["cell_id"] == lp.ARDITI_CELL]
    apin = ((arditi_pins.get(lp.MODEL_KEY) or {}).get(lp.ARDITI_STAGED_ID) or {}).get("object_sha256")
    if len(ar) != 1 or not (ar[0]["is_symlink"] and ar[0]["resolves"] and ar[0]["sha_ok"] and ar[0]["object_sha256"] == apin):
        problems.append({"view": lp.ARDITI_VIEW, "arditi": [e.get("object_sha256") for e in ar], "pin": apin})
    else:
        shas[lp.ARDITI_STAGED_ID] = ar[0]["object_sha256"]
    want_n = STAGED_IDS_N + sum(len(c) for c in extra.values())
    if len(shas) != want_n:
        problems.append({"n_staged": len(shas), "want": want_n})
    return {"ok": not problems, "n_entries": len(entries) + len(ar), "problems": problems, "shas": shas}


@app.function(image=image, timeout=1800, cpu=2, memory=8192)
def image_check(reqs_sha: str) -> dict:
    venv, sentinel, value = serving_venv_names(reqs_sha)
    r = subprocess.run(["bash", "-c", serving_venv_check_cmd(venv, sentinel)], capture_output=True, text=True, timeout=600)
    cache_entries = sum(1 for _ in Path(UV_CACHE).iterdir()) if Path(UV_CACHE).is_dir() else 0
    sentinel_value = Path(sentinel).read_text().strip() if Path(sentinel).is_file() else None
    return {"serving_venv": venv, "sentinel": sentinel_value, "sentinel_expected": value, "check_rc": r.returncode,
            "check_out": (r.stdout + r.stderr)[-1500:], "uv_cache_entries": cache_entries,
            "image_id": os.environ.get("MODAL_IMAGE_ID"),
            "tools": {t: shutil.which(t) for t in ("nvidia-smi", "curl", "git", "bash", "uv")},
            "ok": r.returncode == 0 and cache_entries > 0 and sentinel_value == value}


@app.function(image=cpu_image, volumes={VOL_PATH: VOL}, timeout=1800, cpu=4, memory=16384)
def unpack_tree(sha: str, force: bool = False) -> dict:
    """Install the uploaded shallow checkout under the LANE's repos/ (never /vol/repos, pruned at 12 h) + a staged stamp."""
    VOL.reload()
    dst, tar_path = Path(REPOS) / f"sisteer-{sha}", Path(REPOS) / f"sisteer-{sha}.tar.gz"
    out = {"sha": sha, "tree": str(dst)}
    if dst.exists() and not force:
        checked = checkout_facts(dst, sha)
        out.update(existing=True, verify=checked)
        tar_path.unlink(missing_ok=True)
        if not checked["ok"]:
            out["error"] = f"existing checkout does not verify: {checked['why']} (use --restage)"
        VOL.commit()
        return out
    if not tar_path.is_file():
        return {**out, "error": f"no uploaded tarball at {tar_path}"}
    tmp = Path(REPOS) / f".seedrep-unpack-{sha}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tmp, filter="fully_trusted")
        extracted = tmp / f"sisteer-{sha}"
        if not (extracted / ".git").exists():
            return {**out, "error": "uploaded checkout has no .git directory"}
        if dst.exists():
            shutil.rmtree(dst)
        extracted.rename(dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tar_path.unlink(missing_ok=True)
    checked = checkout_facts(dst, sha)
    out.update(existing=False, verify=checked)
    if checked["ok"]:
        Path(f"{REPOS}/sisteer-{sha}.staged.json").write_text(json.dumps({"W": sha, "utc": utc(), **checked}, indent=1))
    else:
        out["error"] = f"unpacked checkout does not verify: {checked['why']}"
    VOL.commit()
    return out


@app.function(image=cpu_image, volumes={VOL_PATH: VOL}, timeout=1800, cpu=4, memory=16384)
def unpack_bank(tar_name: str, sha: str, pins: dict, arditi_pins: dict, extra_views: Optional[dict] = None, extra_pins: Optional[dict] = None) -> dict:
    """Extract the uploaded bank tarball (full22/<target>/{cells,cells_view/s1..s3 + every plan-referenced extra view} +
    arditi_allpos/<target>/ladder/{cells, cells_view/ar_c0p3}) under <LANE>/objects, re-verify the 16 + extra shas SERVER-SIDE (the
    extra objects under the PLAN's pins), write objects/STAGED.json."""
    VOL.reload()
    tar_path = Path(LANE) / tar_name
    if not tar_path.is_file():
        return {"error": f"no uploaded tarball at {tar_path}"}
    tmp = Path(LANE) / f".unpack-bank-{int(time.time())}"
    tmp.mkdir(parents=True)
    moved = []
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tmp, filter="fully_trusted")
        for src in sorted(tmp.rglob("*")):
            if src.is_dir() and src.name in ("cells", "cells_view") and src.parent != tmp:
                rel = src.relative_to(tmp)
                for entry in sorted(src.iterdir()):
                    dst = Path(OBJECTS_VOL) / rel / entry.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.is_symlink() or dst.exists():
                        shutil.rmtree(dst) if dst.is_dir() and not dst.is_symlink() else dst.unlink()
                    shutil.move(str(entry), str(dst))
                    moved.append(str(rel / entry.name))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        tar_path.unlink(missing_ok=True)
    rep = verify_staged_objects(BANK_VOL, ARDITI_BANK_VOL, pins, arditi_pins, extra_views=extra_views, extra_pins=extra_pins)
    tree_pins_path = Path(REPOS) / f"sisteer-{sha}" / lp.PINS_REL
    if tree_pins_path.is_file():
        tp = json.loads(tree_pins_path.read_text()).get(lp.MODEL_KEY, {})
        for sid, sha_ in rep["shas"].items():
            if sid in tp and tp[sid].get("object_sha256") != sha_:
                rep["problems"].append({"staged_id": sid, "tree_pin": tp[sid].get("object_sha256"), "bank": sha_})
                rep["ok"] = False
    if rep["ok"]:
        Path(STAGED_JSON).write_text(json.dumps({"schema": "seedrep_staged/1", "W": sha, "utc": utc(), "shas": rep["shas"],
                                                 "extra_views": {v: dict(c) for v, c in (extra_views or {}).items()}}, indent=1))
    VOL.commit()
    return {"n_moved": len(moved), **rep, "staged_json": STAGED_JSON if rep["ok"] else None}


ARM_CELL_CHECK_REL = "scripts/sdfbase/full22/arm_cell_check.py"


def arms_paper_view(tree: Path, target: str = lp.MODEL_KEY) -> str:
    """`targets.<target>.view` of the arms table on the tree at W -- the PAPER view `arm_cell_check.resolve()` holds a job's
    `--view` to (launch_report F-A: the app passed `lp.ARDITI_VIEW` = `ar_c0p3`, the Arditi RUNG whose bank view dir the
    objects live under, and the checker refused it deterministically as 'not the paper view'). Read from the table, never
    a literal; a table without the row refuses by name."""
    import yaml  # PyYAML: the generator's own dependency (seedrep_launch_plan reads the roster with it)
    arms = yaml.safe_load((Path(tree) / lp.ARMS_REL).read_text(encoding="utf-8")) or {}
    try:
        view = arms["targets"][target]["view"]
    except (KeyError, TypeError):
        raise ValueError(f"{lp.ARMS_REL}: no targets.{target}.view (the paper view arm_cell_check holds --view to)") from None
    if not isinstance(view, str) or not view:
        raise ValueError(f"{lp.ARMS_REL}: targets.{target}.view is {view!r}, not a view name")
    return view


def arditi_resolve_argv(*, py: str, tree: Path) -> List[str]:
    """`arm_cell_check.py --resolve` for the Arditi arm, as the preflight runs it: `--view` = the arms table's paper view for the
    target (`arms_paper_view`), `--defenses-dir` = the staged bank's `cells_view/<rung>` (the RUNG names the DIRECTORY, not the view)."""
    tree = Path(tree)
    return [py, str(tree / ARM_CELL_CHECK_REL), "--arms", str(tree / lp.ARMS_REL), "--target", lp.MODEL_KEY, "--arm", "arditi_allpos",
            "--view", arms_paper_view(tree), "--resolve", "--defenses-dir", f"{ARDITI_BANK_VOL}/cells_view/{lp.ARDITI_VIEW}",
            "--pins", str(tree / lp.ARDITI_PINS_REL)]


#: what each LEG's drivers read under HF_HUB_OFFLINE=1 / HF_HOME=/vol/hf, by name -- the static / bon /
#: multi-turn legs' confirmatory goal set (jailbreak_run.py: HarmBench's exact test parquet + JBB harmful, `sisteer.eval.jailbreak_data`
#: pins), renellm's frozen replay corpus (`/vol/heldout_corpora`), the controls' pinned corpora and the GSM8K Platinum split
#: (`configs/capeval/pins.yaml` entries, resolved through `sisteer.capeval.corpora`'s OWN file rule), and the PREPARED `datasets` caches
#: the offline `load_dataset` readers need (GSM8K: openai/gsm8k AND gsm8k-platinum, `g8k_cells.py`'s literals; MMLU: cais/mmlu `all`).
#: Derived from the PLAN's legs, never a literal list; a leg this table does not name reads nothing off the volume (misalign / petri
#: carry their prompts in the tree). Directory presence is NECESSARY, not sufficient: `preflight_offline_readers` runs
#: the actual readers under the driver venv and the tinycap CLIENT venv (the GSM8K task-planning probe from g8k_common.sh).
CORPORA_PINS_REL = "configs/capeval/pins.yaml"
CLIENT_LOCK_REL = "configs/capeval/tinycap-client-requirements.txt"       # f22_client_venv's lock: inspect_ai + inspect_evals + the driver's deps
G8K_CELLS_REL = "scripts/sdfbase/gsm8k/g8k_cells.py"                     # DATASETS_REQUIRED / DATASETS_CONFIG_DIRS (stdlib module)
HELDOUT_CORPUS_FILES = ("renellm.jsonl", "renellm.jsonl.manifest.json")
CONTROLS_PIN_NAMES: Tuple[str, ...] = ("xstest_v2", "or_bench", "mmlu")
GSM8K_PIN_NAMES: Tuple[str, ...] = ("gsm8k_platinum",)
#: the offline `datasets` config dir the MMLU reader's signature names -- parity.mmlu_parity_rows: load_dataset("cais/mmlu", "all", ...)
MMLU_DATASETS_DIR: Tuple[str, str] = ("cais___mmlu", "all")
GOAL_SET_LEGS_EXTRA: Tuple[str, ...] = ("bon",)


def corpora_needed_for_plan(plan: Mapping[str, Any]) -> Dict[str, Any]:
    """``{legs, goal_sets: bool, heldout: [files], pins: [pins.yaml names], datasets: [dataset dirs]}`` for the legs the plan stands up."""
    legs = {str(r.get("leg")) for r in list(plan.get("standups") or []) + list(plan.get("s5_probe_rows") or [])}
    goal_set_legs = set(lp.STATIC_LEGS) | set(lp.MULTITURN_LEGS) | set(GOAL_SET_LEGS_EXTRA)
    need: Dict[str, Any] = {"legs": sorted(legs), "goal_sets": bool(legs & goal_set_legs), "heldout": [], "pins": [], "datasets": []}
    if "renellm" in legs:
        need["heldout"] = list(HELDOUT_CORPUS_FILES)
    if "controls" in legs:
        need["pins"] += list(CONTROLS_PIN_NAMES)
        need["datasets"].append(MMLU_DATASETS_DIR[0])
    if "gsm8k" in legs:
        need["pins"] += list(GSM8K_PIN_NAMES)
        need["datasets"].append("gsm8k")            # both of g8k_cells.DATASETS_REQUIRED (resolved from the tree by `datasets_cache_dirs_needed`)
    return need


def dataset_snapshot_dir(hf_vol: str, repo_id: str, revision: str) -> Path:
    return Path(hf_vol) / "hub" / f"datasets--{repo_id.replace('/', '--')}" / "snapshots" / revision


def _tree_src_on_path(tree: Path) -> None:
    src = str(Path(tree) / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def pinned_snapshot_payload(snap: Path, entry: Mapping[str, Any]) -> List[str]:
    """The files a pins.yaml entry READS from its snapshot -- `sisteer.capeval.corpora.pinned_snapshot_files`' own local rule (the verify
    format's suffixes, then the entry's glob patterns on the snapshot-relative path), sans its `snapshot_download`: a README or
    .gitattributes never counts (ANY file used to pass). Empty = the reader would refuse this snapshot."""
    import fnmatch
    from sisteer.capeval import corpora as CP  # the tree's own rule table (yaml-only import)
    suffixes = CP._VERIFY_SUFFIXES[entry["verify"]]
    patterns = entry.get("patterns")
    out = []
    for q in sorted(Path(snap).rglob("*")):
        if not q.is_file():
            continue
        rel = q.relative_to(snap).as_posix()
        if not any(rel.endswith(sfx) for sfx in suffixes):
            continue
        if patterns and not any(fnmatch.fnmatch(rel, pat) for pat in patterns):
            continue
        out.append(rel)
    return out


def g8k_cells_module(tree: Path):
    """`scripts/sdfbase/gsm8k/g8k_cells.py` from the STAGED tree, by path (stdlib-only; its sibling helpers resolve from its own dir)."""
    import importlib.util
    path = Path(tree) / G8K_CELLS_REL
    if not path.is_file():
        raise FileNotFoundError(f"{G8K_CELLS_REL} is absent from the staged tree: the GSM8K datasets literals cannot be read")
    d = str(path.parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("seedrep_pf_g8k_cells", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def datasets_cache_dirs_needed(tree: Path, need: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    """``{<dataset dir>: <offline config dir>}`` the plan's offline `load_dataset` readers look for under HF_DATASETS_CACHE, from the
    tree's own literals: g8k_cells.DATASETS_REQUIRED / DATASETS_CONFIG_DIRS for GSM8K (openai/gsm8k is looked up offline as
    `default-data_dir=main`), `cais___mmlu/all` for the MMLU control."""
    out: Dict[str, Optional[str]] = {}
    if "gsm8k" in need.get("datasets", []):
        g8k = g8k_cells_module(tree)
        for d in g8k.DATASETS_REQUIRED:
            out[str(d)] = str(g8k.DATASETS_CONFIG_DIRS[d])
    if MMLU_DATASETS_DIR[0] in need.get("datasets", []):
        out[MMLU_DATASETS_DIR[0]] = MMLU_DATASETS_DIR[1]
    return out


def datasets_cache_problems(cache_root: str, dirs: Mapping[str, Optional[str]]) -> List[str]:
    """g8k_common.sh `g8k_datasets_guard`, ported: each dataset dir present under HF_DATASETS_CACHE AND its offline CONFIG dir (a
    symlink alias is followed; a dangling one refuses by name with the repair recipe). Necessary, not sufficient -- the reader probe rules."""
    problems = []
    for d, cfg in dirs.items():
        p = Path(cache_root) / d
        if not p.is_dir():
            problems.append(f"HF_DATASETS_CACHE={cache_root} carries no prepared {d} (the offline readers load it from a PREPARED arrow cache, "
                            "never the hub snapshot: prepare once by an ONLINE load at the readers' exact signatures; g8k_common.sh D2/D18)")
            continue
        if cfg:
            q = p / cfg
            if not q.is_dir():
                how = "a dangling symlink" if q.is_symlink() else "absent"
                problems.append(f"HF_DATASETS_CACHE={cache_root} carries no offline config dir {d}/{cfg} ({how}): `datasets` looks OFFLINE for the "
                                f"config dir named by the load signature while an ONLINE prepare may name it by a kwargs hash -- alias it: "
                                f"ln -s <hash-named dir> {cache_root}/{d}/{cfg} (D18)")
    return problems


def preflight_corpora_check(tree: Path, plan: Mapping[str, Any], *, hf_vol: str = HF_VOL, corpora_vol: str = CORPORA_VOL,
                            datasets_cache: str = HF_DATASETS_VOL) -> dict:
    """Preflight check 8 (strength's `corpora_pinned_snapshots_present`, ported): every corpus / dataset the PLAN's legs
    read under HF_HUB_OFFLINE=1 is present on the volume at its PINNED revision -- HarmBench's EXACT test parquet and the JBB harmful file
    (the staged tree's `jailbreak_data` pins), each pins.yaml entry's PAYLOAD files by the reader's own rule (`pinned_snapshot_payload`;
    a snapshot dir holding only a README refuses), renellm's frozen corpus + manifest under `/vol/heldout_corpora` (F-H), and the PREPARED
    `datasets` cache dirs the offline `load_dataset` readers need (`datasets_cache_problems`). FATAL by name. The actual readers then run
    under their venvs in `preflight_offline_readers` -- this check is the fast, static half."""
    import yaml  # PyYAML: the cpu image's own dependency
    tree = Path(tree)
    need = corpora_needed_for_plan(plan)
    found: Dict[str, Any] = {"legs": need["legs"]}
    if need["goal_sets"]:
        _tree_src_on_path(tree)
        from sisteer.eval import jailbreak_data as jd  # type: ignore  # the staged tree's pins; its module top is stdlib-only
        for label, repo, rev, file in (("harmbench", jd.HARMBENCH_REPO, jd.HARMBENCH_REVISION, jd.HARMBENCH_FILES["test"]),
                                       ("jbb", jd.JBB_REPO, jd.JBB_REVISION, jd.JBB_FILES["harmful"])):
            snap = dataset_snapshot_dir(hf_vol, repo, rev)
            assert snap.is_dir(), f"{label}: no snapshot {snap}"
            assert (snap / file).is_file(), f"{label}: {file} missing under {snap} (the file the reader downloads by name)"
            found[label] = {"repo_id": repo, "revision": rev[:12], "file": file}
    if need["pins"]:
        _tree_src_on_path(tree)
        pins = (yaml.safe_load((tree / CORPORA_PINS_REL).read_text(encoding="utf-8")) or {}).get("datasets") or {}
        for name in need["pins"]:
            e = pins.get(name)
            assert isinstance(e, Mapping) and e.get("repo_id") and e.get("revision") and e.get("verify"), f"{CORPORA_PINS_REL}: no complete pin named {name!r}"
            snap = dataset_snapshot_dir(hf_vol, str(e["repo_id"]), str(e["revision"]))
            assert snap.is_dir(), f"{name}: no snapshot {snap}"
            payload = pinned_snapshot_payload(snap, e)
            assert payload, (f"{name}: no payload matching patterns={e.get('patterns')!r} / verify={e['verify']!r} under {snap} "
                             f"(present: {sorted(q.relative_to(snap).as_posix() for q in snap.rglob('*') if q.is_file())[:6]}) -- the reader would refuse it")
            found[name] = {"repo_id": e["repo_id"], "revision": str(e["revision"])[:12], "files": payload}
    if need["heldout"]:
        for f in need["heldout"]:
            assert (Path(corpora_vol) / f).is_file(), f"heldout corpus {f} missing under {corpora_vol} (renellm replays it; F-H)"
        found["heldout_corpora"] = sorted(q.name for q in Path(corpora_vol).iterdir())
    if need["datasets"]:
        dirs = datasets_cache_dirs_needed(tree, need)
        problems = datasets_cache_problems(datasets_cache, dirs)
        assert not problems, problems
        found["datasets_cache"] = {"root": datasets_cache, "dirs": dict(dirs)}
    return found


# --- the actual readers, under the venvs the drivers use (presence is not readability) ----------------------------------------

def client_lock_split(lock_text: str) -> Tuple[List[str], List[str]]:
    """f22_client_venv's split of the tinycap client lock: the hashed requirement lines (`--require-hashes`) and the VCS pin
    (`instruction-following-eval @ git+...`, installed `--no-deps` on its own)."""
    vcs_re = re.compile(r"^(instruction-following-eval) @ ")
    hashed = [ln for ln in lock_text.splitlines() if not vcs_re.match(ln)]
    vcs = [ln.rstrip().rstrip("\\").rstrip() for ln in lock_text.splitlines() if vcs_re.match(ln)]
    return hashed, vcs


def client_venv_names(lock_sha256: str) -> Tuple[str, str, str]:
    """common.sh f22_client_venv: `/home/$USER/venvs/tinycap-client-<sha8>` + its `.client-bootstrap` sentinel holding sha16."""
    venv = f"{VENVS_LOCAL}/tinycap-client-{lock_sha256[:8]}"
    return venv, f"{VENVS_LOCAL}/.tinycap-client-{lock_sha256[:8]}.client-bootstrap", lock_sha256[:16]


def client_venv_python(tree: Path) -> str:
    return f"{client_venv_names(sha256_file(Path(tree) / CLIENT_LOCK_REL))[0]}/bin/python"


CLIENT_VENV_GUARD = ("import inspect_ai, inspect_evals, sisteer.capeval.tinycap.contracts, sisteer.capeval.serving.usage_ledger; "
                     "print('client venv OK', inspect_ai.__version__)")


def ensure_client_venv(tree: Path, *, run: Callable[..., Any] = subprocess.run) -> dict:
    """common.sh `f22_client_venv`, in-container: the tinycap CLIENT venv (inspect_ai + inspect_evals + the driver's deps; CPU-only) from
    the tree's `configs/capeval/tinycap-client-requirements.txt` -- `uv venv`, the hashed lines `--no-deps --require-hashes`, the VCS
    pin `--no-deps`, then the import guard; idempotent on its sentinel. The GSM8K task-planning probe (and, on the cluster, every
    Inspect-side GSM8K step) runs under THIS venv, not the driver venv, whose `uv sync --extra cu128` carries no inspect_ai."""
    tree = Path(tree)
    lock = tree / CLIENT_LOCK_REL
    if not lock.is_file():
        raise FileNotFoundError(f"no client lock at {lock}")
    venv, sentinel, want = client_venv_names(sha256_file(lock))
    py = f"{venv}/bin/python"
    if Path(sentinel).is_file() and Path(sentinel).read_text().strip() == want and Path(py).exists():
        return {"venv": venv, "built": False}
    t0 = time.time()
    Path(VENVS_LOCAL).mkdir(parents=True, exist_ok=True)
    hashed, vcs = client_lock_split(lock.read_text(encoding="utf-8"))
    hashed_path = Path(VENVS_LOCAL) / f".{Path(venv).name}.hashed.txt"
    hashed_path.write_text("\n".join(hashed) + "\n", encoding="utf-8")
    env = {**os.environ, "UV_CACHE_DIR": UV_CACHE, "UV_LINK_MODE": "copy"}
    env.pop("VIRTUAL_ENV", None)
    cutoff = time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime(time.time() - 7 * 86400))    # the cluster's rolling `-7 days` (hashes pin the bytes)
    steps = []
    if not Path(py).exists():
        steps.append(["uv", "venv", "--python", PY_VERSION, venv])
    # --no-config, as common.sh's: run from the tree, uv would read pyproject's exclude-newer-package entries, which a command-line
    # --exclude-newer does not replace (the anthropic entry refuses the lock's anthropic 1.1.0); the lock is compiled with --no-config
    steps.append(["uv", "pip", "install", "--no-config", "--python", py, "--no-deps", "--require-hashes", "--exclude-newer", cutoff,
                  "-r", str(hashed_path)])
    steps += [["uv", "pip", "install", "--python", py, "--no-deps", v] for v in vcs]
    for argv in steps:
        r = run(argv, cwd=str(tree), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"client venv step failed rc {r.returncode}: {' '.join(argv[:6])}...: {(r.stdout + r.stderr)[-1200:]}")
    g = run([py, "-c", CLIENT_VENV_GUARD], cwd=str(tree), env={**env, "PYTHONPATH": f"{tree}/src:{tree}/scripts"}, capture_output=True, text=True)
    if g.returncode != 0:
        raise RuntimeError(f"client venv guard: inspect_ai / inspect_evals / the tinycap contracts / the usage ledger do not import: {(g.stdout + g.stderr)[-800:]}")
    Path(sentinel).write_text(want)
    return {"venv": venv, "built": True, "s": round(time.time() - t0, 1), "guard": g.stdout.strip()[-120:], "n_vcs": len(vcs)}


#: the reader probes: `name -> (venv, script, argv_tail)`; each script is run `python - <argv>` with the probe env (offline, the volume
#: caches) from the tree, and PRINTS one `readers OK:` line or raises. The GSM8K probe is g8k_common.sh's `g8k_datasets_probe` verbatim.
READER_STATIC_GOAL_SETS = """import sys
from sisteer.eval.jailbreak_data import load_harmbench, load_jbb
hb = load_harmbench(split="test")
jb = load_jbb("harmful")
assert hb and jb, (len(hb), len(jb))
print(f"readers OK: harmbench test {len(hb)} behaviours, jbb harmful {len(jb)} (offline, by the drivers' own loaders)")
"""
READER_HELDOUT_RENELLM = """import os
from sisteer.eval.attacks import renellm
assert os.environ.get("SISTEER_HELDOUT_CORPUS_DIR"), "SISTEER_HELDOUT_CORPUS_DIR is unset in the probe env"
c = renellm._corpus()
assert c is not None, (f"renellm frozen corpus not readable at {renellm.replay_path()}: neither the volume's copy {renellm._corpus_path()} "
                       "nor the tree's shipped copy exists (loud_skip -> every child exits 0 writing nothing)")
print(f"readers OK: renellm corpus {c.content_hash[:16]} at {renellm.replay_path()}")
"""
READER_CONTROLS_JUDGED = """import sys
from sisteer.capeval.corpora import load_pins
from sisteer.capeval.tiers import judged
pins = load_pins(sys.argv[1])
xs = judged.xstest_rows(pins)
ob = judged.or_bench_rows(pins)
assert len(xs) == judged.XSTEST_N_SAFE + judged.XSTEST_N_UNSAFE, len(xs)
assert len(ob) == judged.ORBENCH_N, len(ob)
print(f"readers OK: xstest_v2 {len(xs)} rows, or_bench {len(ob)} rows (offline, the judged loaders)")
"""
READER_CONTROLS_MMLU = """import sys
from sisteer.capeval.corpora import load_pins
from sisteer.capeval.tiers.parity import mmlu_parity_rows
rows = mmlu_parity_rows(load_pins(sys.argv[1]), n=None)
assert rows, "mmlu: no rows"
print(f"readers OK: mmlu {len(rows)} rows OFFLINE from HF_DATASETS_CACHE (load_dataset cais/mmlu all)")
"""
READER_GSM8K_PLATINUM = """import sys
from sisteer.capeval.corpora import load_pins
from sisteer.capeval.tiers.parity import gsm8k_parity_rows
rows = gsm8k_parity_rows(load_pins(sys.argv[1]), n=None)
assert rows, "gsm8k_platinum: no rows"
print(f"readers OK: gsm8k-platinum {len(rows)} rows OFFLINE from HF_DATASETS_CACHE (load_dataset madrylab/gsm8k-platinum main)")
"""
READER_GSM8K_PLAN_TASK = """import sys
from pathlib import Path
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tiers import inspect_served as isv
task, n = sys.argv[1], int(C.GSM8K_PAIRED_N)
plan = isv.plan_task(task, grader_model=None, thinking="off", repo_root=Path.cwd())
ids = list(plan.sample_ids or [])
assert len(ids) == n, f"{task} planned {len(ids)} items offline, not {n}"
print(f"datasets probe OK: {task} plans {len(ids)} paired items OFFLINE from HF_DATASETS_CACHE (both datasets resolved)")
"""


def reader_probe_scripts(need: Mapping[str, Any]) -> Dict[str, Tuple[str, str, Tuple[str, ...]]]:
    """The probes the plan's legs require: ``name -> (venv, script, argv_tail)`` with ``venv`` in {driver, client}; `{pins}` in an argv
    tail is the staged tree's pins.yaml path (substituted by the runner)."""
    out: Dict[str, Tuple[str, str, Tuple[str, ...]]] = {}
    if need.get("goal_sets"):
        out["static_goal_sets"] = ("driver", READER_STATIC_GOAL_SETS, ())
    if need.get("heldout"):
        out["heldout_renellm"] = ("driver", READER_HELDOUT_RENELLM, ())
    if "xstest_v2" in need.get("pins", []) or "or_bench" in need.get("pins", []):
        out["controls_judged"] = ("driver", READER_CONTROLS_JUDGED, ("{pins}",))
    if "mmlu" in need.get("pins", []):
        out["controls_mmlu"] = ("driver", READER_CONTROLS_MMLU, ("{pins}",))
    if "gsm8k_platinum" in need.get("pins", []):
        out["gsm8k_platinum"] = ("driver", READER_GSM8K_PLATINUM, ("{pins}",))
        out["gsm8k_plan_task"] = ("client", READER_GSM8K_PLAN_TASK, (lp.GSM8K_INSPECT_TASK,))
    return out


def reader_probe_env(tree: Path, *, hf_vol: str = HF_VOL, datasets_cache: str = HF_DATASETS_VOL, corpus_dir: str = HELDOUT_CORPUS_LOCAL,
                     base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The drivers' env (`container_env`) with the offline switches the readers must honour and NO GPU."""
    env = container_env(bank_view="", tree=str(tree), thinking=False, base=base)
    env.update({"HF_HOME": hf_vol, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "HF_DATASETS_CACHE": datasets_cache,
                HELDOUT_CORPUS_ENV: corpus_dir, "CUDA_VISIBLE_DEVICES": ""})
    return env


def run_reader_probe(script: str, argv_tail: Sequence[str], *, python: str, tree: Path, env: Mapping[str, str], timeout: int = 1200,
                     run: Callable[..., Any] = subprocess.run) -> dict:
    """`<python> - <argv>` with the script on stdin, from the tree; `{pins}` resolved to the tree's pins.yaml. Raises on a non-zero rc."""
    tail = [str(Path(tree) / CORPORA_PINS_REL) if a == "{pins}" else a for a in argv_tail]
    r = run([python, "-", *tail], input=script, cwd=str(tree), env=dict(env), capture_output=True, text=True, timeout=timeout)
    out = (r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"reader probe rc {r.returncode} under {python}: {out[-1500:]}")
    return {"python": python, "out": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""}


@app.function(image=image, volumes={VOL_PATH: VOL}, timeout=3600, cpu=8, memory=32768)
def preflight_offline_readers(sha: str, plan: dict) -> dict:
    """Preflight check 8's AUTHORITATIVE half: the tree built the way a leg builds it, the driver venv and the tinycap
    CLIENT venv, then every reader the plan's legs use, run under its venv with the drivers' offline env against the volume's caches --
    the static goal sets, renellm's corpus, the judged controls, MMLU and GSM8K-Platinum through `load_dataset`, and g8k_common.sh's
    task-planning probe (`inspect_served.plan_task` planning exactly GSM8K_PAIRED_N items offline, which resolves openai/gsm8k AND
    gsm8k-platinum from the PREPARED cache). A hub snapshot without its prepared cache fails HERE, by the reader's own error."""
    VOL.reload()
    rep: dict = {"sha": sha, "ok": True, "checks": {}}

    def chk(name, fn):
        try:
            rep["checks"][name] = {"ok": True, **(fn() or {})}
        except BaseException as e:  # noqa: BLE001 -- every failure named, none hidden
            rep["checks"][name] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:1200]}"}
            rep["ok"] = False

    box: Dict[str, Any] = {}

    def build():
        built = _build_shared(sha)
        box["tree"] = Path(built["repo_dir"])
        return {"tree": str(box["tree"])}
    chk("tree_built_as_a_leg_builds_it", build)
    if "tree" not in box:
        return rep
    tree = box["tree"]
    need = corpora_needed_for_plan(plan)
    probes = reader_probe_scripts(need)
    rep["needs"] = need
    chk("driver_venv", lambda: ensure_driver_venv(tree, sha))
    if any(kind == "client" for kind, _, _ in probes.values()):
        chk("client_venv", lambda: ensure_client_venv(tree))
    env = reader_probe_env(tree)
    for name, (kind, script, tail) in probes.items():
        py = lp.driver_venv_python(sha) if kind == "driver" else client_venv_python(tree)
        if not Path(py).exists():
            rep["checks"][f"reader_{name}"] = {"ok": False, "err": f"no {kind} venv python at {py}"}
            rep["ok"] = False
            continue
        chk(f"reader_{name}", lambda script=script, tail=tail, py=py: {"venv": kind, **run_reader_probe(script, tail, python=py, tree=tree, env=env)})
    return rep


def preflight_objects_check(tree: Path, plan: Mapping[str, Any], *, bank_root: str = BANK_VOL, arditi_root: str = ARDITI_BANK_VOL,
                            staged_json: str = STAGED_JSON) -> dict:
    """The preflight's object check (`sixteen_object_shas`): the bank verifies against the tree's pins PLUS the PLAN's extra views and pins --
    `lp.bank_views_for_plan(plan)` / `lp.bank_pins_for_plan(plan)`, the SAME derivation `do_stage_bank` / `unpack_bank` wrote STAGED.json
    from (the check called verify_staged_objects without them, so a plan with `s5_probe_rows` -- 16 + 2 staged -- read
    "STAGED.json disagrees with the bank" at every W). Still FAILS on any sha mismatch, on an entry STAGED.json carries that the plan does
    not pin (a plan with no probe rows against an 18-entry STAGED.json), on a plan-referenced view the bank lacks, and on a STAGED.json
    whose recorded extra views are not the plan's."""
    pins = json.loads((Path(tree) / lp.PINS_REL).read_text())
    apins = json.loads((Path(tree) / lp.ARDITI_PINS_REL).read_text())
    extra_views = lp.bank_views_for_plan(plan)
    extra_pins = lp.bank_pins_for_plan(plan)
    rep_ = verify_staged_objects(bank_root, arditi_root, pins, apins, extra_views=extra_views, extra_pins=extra_pins)
    assert rep_["ok"], rep_["problems"]
    staged = json.loads(Path(staged_json).read_text())
    assert staged["shas"] == rep_["shas"], "STAGED.json disagrees with the bank"
    recorded = {str(v): {str(c): str(sid) for c, sid in cells.items()} for v, cells in (staged.get("extra_views") or {}).items()}
    want = {str(v): {str(c): str(sid) for c, sid in cells.items()} for v, cells in extra_views.items()}
    assert recorded == want, f"STAGED.json's extra views {sorted(recorded)} are not the plan's {sorted(want)}"
    return {"n": len(rep_["shas"]), "n_extra": sum(len(c) for c in extra_views.values()), "extra_views": sorted(extra_views)}


@app.function(image=cpu_image, volumes={VOL_PATH: VOL}, timeout=900, cpu=2, memory=8192)
def preflight(sha: str, plan: dict, expected_dep_hashes: dict) -> dict:
    """--pf (§4.1): every non-GPU precondition from the volume — the staged tree at W (exact HEAD, clean, the image's
    dependency files), the staged object shas (the 16 + the plan's extra views), the model snapshot, the identity plan files every child names (strict JSON),
    the three vendor decode rows through `PolicyFile.by_ref` + not-superseded, `make_full22_configs.py --check`,
    `render_arditi_seedrep_configs.py --check`, `arm_cell_check.py --resolve` for ar_c0p3, the plan's own
    validation through the tree's geometry table, and (check 8) every corpus / dataset the plan's legs read under
    HF_HUB_OFFLINE=1 present on the volume at its pin. No GPU minute before it passes."""
    VOL.reload()
    rep: dict = {"sha": sha, "ok": True, "checks": {}}

    def chk(name, fn):
        try:
            rep["checks"][name] = {"ok": True, **(fn() or {})}
        except BaseException as e:  # noqa: BLE001 -- every failure named, none hidden
            rep["checks"][name] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:800]}"}
            rep["ok"] = False

    tree = Path(REPOS) / f"sisteer-{sha}"

    def tree_ok():
        facts = checkout_facts(tree, sha)
        assert facts["ok"], facts
        got = dependency_hashes(tree)
        assert got == expected_dep_hashes, f"staged dependency files differ from the image's: {got} != {expected_dep_hashes}"
        return {"tree": str(tree), **facts}
    chk("tree_staged_clean_and_image_matches", tree_ok)

    def objects():
        return preflight_objects_check(tree, plan)
    chk("sixteen_object_shas", objects)          # the name is referenced by the certificate / S0 records; the set is the PLAN's

    def model_snapshot():
        snap = Path(HF_VOL) / "hub" / f"models--{lp.MODEL.replace('/', '--')}" / "snapshots" / lp.MODEL_REVISION
        assert snap.is_dir(), f"no snapshot at {snap}"
        st = [p for p in snap.iterdir() if p.name.endswith(".safetensors")]
        assert st and (snap / "config.json").is_file(), sorted(p.name for p in snap.iterdir())
        return {"snapshot": str(snap), "n_safetensors": len(st)}
    chk("model_snapshot_at_pinned_revision", model_snapshot)

    def plans():
        root = Path(plan["plans_root"])
        missing, bad = [], []
        for r in plan["standups"] + plan["rungs"]:
            for k in r["children"]:
                p = root / k["registry_identity_rel"][len(lp.PLANS_REL) + 1:]     # <leg>/<arm>__s<s>.json under the plans root
                if not p.is_file():
                    missing.append(str(p))
                    continue
                try:
                    d = json.loads(p.read_text())
                except ValueError as e:
                    bad.append(f"{p}: {e}")
                    continue
                single = isinstance(d, dict) and ("eval_id" in d or "run_id" in d or (isinstance(d.get("plan"), dict)))
                task_map = isinstance(d, dict) and d and all(isinstance(v, dict) for v in d.values()) and not single
                if not (single or task_map):
                    bad.append(f"{p}: neither a single identity nor a per-task map")
        assert not missing and not bad, {"missing": missing[:10], "n_missing": len(missing), "bad": bad[:10]}
        return {"n_children": sum(len(r["children"]) for r in plan["standups"] + plan["rungs"])}
    chk("registry_identity_files_strict", plans)

    def decode_rows():
        sys.path.insert(0, str(tree / "src"))
        from sisteer.registry.decode_policy import PolicyFile  # type: ignore
        pf = PolicyFile.load(tree / "configs/registry/decode_policy.json")
        rows = {}
        for mode in ("bon", "multi_turn", "petri"):
            ref = f"{pf.model_id(lp.MODEL_KEY)}|{mode}|off|vendor_preset@v1"
            pol = pf.by_ref(ref)
            assert pol.superseded_by_policy_id is None, f"{ref} is superseded by {pol.superseded_by_policy_id}"
            rows[ref] = {"is_default": getattr(pol, "is_default", None)}
        return {"rows": rows}
    chk("three_vendor_decode_rows_by_ref", decode_rows)

    def generators():
        py = sys.executable
        out = {}
        for rel, extra in (("scripts/sdfbase/make_full22_configs.py", ["--check"]),
                           ("scripts/sdfbase/seedrep/render_arditi_seedrep_configs.py", ["--check"])):
            if not (tree / rel).is_file():
                out[rel] = "ABSENT"
                continue
            r = subprocess.run([py, str(tree / rel), *extra], cwd=str(tree), capture_output=True, text=True, timeout=600,
                               env={**os.environ, "PYTHONPATH": f"{tree}/src:{tree}/scripts"})
            out[rel] = r.returncode
            assert r.returncode == 0, f"{rel} --check rc {r.returncode}: {(r.stdout + r.stderr)[-600:]}"
        r = subprocess.run(arditi_resolve_argv(py=py, tree=tree), cwd=str(tree), capture_output=True, text=True, timeout=600,
                           env={**os.environ, "PYTHONPATH": f"{tree}/src:{tree}/scripts"})
        out["arm_cell_check --resolve"] = r.returncode
        assert r.returncode == 0, f"arm_cell_check rc {r.returncode}: {(r.stdout + r.stderr)[-600:]}"
        return out
    chk("generators_check_and_arditi_resolve", generators)

    def plan_validates():
        sys.path.insert(0, str(tree / "src"))
        from sisteer.capeval.serving import geometry as sg  # type: ignore
        problems = lp.validate_plan(plan, table=sg)
        assert not problems, problems[:20]
        return {"n_standups": len(plan["standups"]), "n_rungs": len(plan["rungs"])}
    chk("launch_plan_validates_against_table", plan_validates)

    def corpora():
        return preflight_corpora_check(tree, plan)
    chk("corpora_pinned_snapshots_present", corpora)     # check 8: what the legs read offline is on the volume, at its pin
    return rep


@app.function(image=cpu_image, gpu="H200", timeout=600, cpu=2, memory=8192)
def gpu_probe(hold_s: int = 60) -> dict:
    t0 = time.time()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    time.sleep(hold_s)
    return {"task_id": os.environ.get("MODAL_TASK_ID"), "gpu": gpu, "start": t0, "end": time.time()}


def probe_concurrency(n: int, hold_s: int = 60) -> dict:
    """S0's quota preflight: N probes together -> the peak alive at any instant (strength's gpu_probe)."""
    calls = [gpu_probe.spawn(hold_s) for _ in range(n)]
    results = [c.get() for c in calls]
    starts, ends = sorted(r["start"] for r in results), sorted(r["end"] for r in results)
    peak = alive = i = j = 0
    while i < len(starts):
        if j < len(ends) and ends[j] <= starts[i]:
            alive -= 1; j += 1
        else:
            alive += 1; i += 1; peak = max(peak, alive)
    rep = {"n_spawned": n, "peak_concurrent": peak, "distinct_gpus": len({r["gpu"] for r in results})}
    print(json.dumps(rep, indent=1))
    return rep


# --------------------------------------------------------------------------- #
# the container: run ONE plan row                                               #
# --------------------------------------------------------------------------- #

class TreeSyncer:
    """Incremental copy of a local tree to a volume path (strength's): a file is re-shipped only when (size, mtime_ns) differs."""

    def __init__(self, src, dest) -> None:
        self.src, self.dest = Path(src), Path(dest)
        self.seen: Dict[str, Tuple[int, int]] = {}

    def copy_changed(self) -> int:
        n = 0
        if not self.src.is_dir():
            return 0
        for p in self.src.rglob("*"):
            if p.is_symlink() or not p.is_file():
                continue
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            rel = str(p.relative_to(self.src))
            sig = (st.st_size, st.st_mtime_ns)
            if self.seen.get(rel) == sig:
                continue
            q = self.dest / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(p, q)
            except FileNotFoundError:
                continue
            self.seen[rel] = sig
            n += 1
        return n


def _sync_all(syncers, lock, label: str) -> None:
    with lock:
        try:
            n = sum(s.copy_changed() for s in syncers)
            VOL.commit()
            print(f"[sync {label}] {n} file(s) committed", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[sync {label}] WARNING: {type(e).__name__}: {e}", flush=True)


def _gpu_facts() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        name, uuid_, drv, mem = [s.strip() for s in out.split(",", 3)]
        return {"name": name, "uuid": uuid_, "driver_version": drv, "memory_total": mem}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


SEED_COMPILE_CACHES_ENV = "SEEDREP_SEED_COMPILE_CACHES"           # "0" / unset (default) = do not seed the local compile caches from the volume;
#                                                                  "1" = seed, bounded by SEED_MAX_FILES / SEED_MAX_BYTES; anything else = off, recorded
# F-Y (launch_report 10:03Z): the seed copytree runs in every container BEFORE the venv check, object verification and standup. Once 14
# rows had persisted 17,515 files / 5,994 MB the next containers each copied for ~44 min (398 files/min, 86 MB/min measured in-container)
# before standing anything up, against a cold compile of a few minutes; rows that seeded nothing attested READY in ~15-18 min. The bound is
# what a seed may cost when it is turned on: ~5 min of copy at the measured rate, well under a cold compile's savings.
SEED_MAX_FILES = 2000
SEED_MAX_BYTES = 512 * 1024 * 1024


def _count_tree(root: Path) -> tuple[int, int]:
    """(files, bytes) under ``root`` from ONE os.scandir walk -- the volume is a FUSE mount, so the metadata pass happens once, before any copy.
    Symlinks count as one file of the link's own size and are never followed (copytree(symlinks=True) copies them AS links); directories are
    not counted."""
    files = nbytes = 0
    stack = [str(root)]
    while stack:
        with os.scandir(stack.pop()) as it:
            for ent in it:
                if ent.is_dir(follow_symlinks=False):
                    stack.append(ent.path)
                    continue
                files += 1
                try:
                    nbytes += ent.stat(follow_symlinks=False).st_size
                except OSError:
                    pass
    return files, nbytes


def seed_compile_caches(env: Mapping[str, str] = None) -> dict:
    """The `compile_cache_seed` evidence of an attempt, one record per COMPILE_CACHES entry: ``{"volume", "local", "seeded", "setting", ...}``.
    OFF by default: ``$SEEDREP_SEED_COMPILE_CACHES`` must be "1" for the local cache to be seeded from the volume, and then the
    source is counted first and refused over the bound. The decision is RECORDED, never silent -- ``skipped`` is ``by_default`` (unset),
    ``by_env`` (set to anything but "1"), ``no_source`` (on, but the volume holds nothing) or ``over_bound`` (on, source over SEED_MAX_FILES /
    SEED_MAX_BYTES; ``source_files`` / ``source_bytes`` / ``count_s`` / ``bound`` say by how much); a copy records ``seed_s`` and the counts as
    before. ``env`` is the launcher's setting as spawn_ctx carried it (the remote worker's environ is the image's, like the persist switch);
    the raw value rides so the evidence can tell an unset default from an explicit off."""
    e = os.environ if env is None else env
    raw = str(e.get(SEED_COMPILE_CACHES_ENV, "") or "").strip()
    setting = {SEED_COMPILE_CACHES_ENV: raw or "0", "source": "env" if raw else "default"}
    out = {}
    for name, local in COMPILE_CACHES.items():
        src, dst = Path(CACHE_VOL) / name, Path(local)
        rec = {"volume": str(src), "local": local, "seeded": False, "setting": setting}
        if raw != "1":
            rec["skipped"] = "by_env" if raw else "by_default"
            print(f"[compile-cache] seed {name}: skipped ({rec['skipped']}: {SEED_COMPILE_CACHES_ENV}={raw or 'unset'}); {src} -> {local} not copied",
                  flush=True)
        elif not src.is_dir():
            rec["skipped"] = "no_source"
            print(f"[compile-cache] seed {name}: skipped (no_source: {src} is missing)", flush=True)
        else:
            t0 = time.time()
            files, nbytes = _count_tree(src)
            rec.update(source_files=files, source_bytes=nbytes, count_s=round(time.time() - t0, 1))
            if files == 0:                                       # the count is the emptiness test (a tree of empty dirs holds nothing to seed)
                rec["skipped"] = "no_source"
                print(f"[compile-cache] seed {name}: skipped (no_source: {src} holds no files)", flush=True)
            elif files > SEED_MAX_FILES or nbytes > SEED_MAX_BYTES:
                rec.update(skipped="over_bound", bound={"max_files": SEED_MAX_FILES, "max_bytes": SEED_MAX_BYTES})
                print(f"[compile-cache] seed {name}: skipped (over_bound: {files} files / {nbytes} B at {src}, bound {SEED_MAX_FILES} files / "
                      f"{SEED_MAX_BYTES} B; counted in {rec['count_s']}s); {local} not seeded", flush=True)
            else:
                t1 = time.time()
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst, symlinks=True)
                rec.update(seeded=True, seed_s=round(time.time() - t1, 1))
                print(f"[compile-cache] seed {name}: copied {files} files / {nbytes} B {src} -> {local} in {rec['seed_s']}s "
                      f"(counted in {rec['count_s']}s)", flush=True)
        out[name] = rec
    return out


PERSIST_COMPILE_CACHES_ENV = "SEEDREP_PERSIST_COMPILE_CACHES"     # "1" = persist (OPT-IN); unset (default) / anything else = skip, recorded
COMPILE_CACHE_SKIPPED = "skipped_by_env"                          # set, but not "1"
COMPILE_CACHE_SKIPPED_DEFAULT = "skipped_by_default"              # unset (the default is OFF -- fx1, 2026-09-21)
COMPILE_CACHE_SKIPPED_NOT_DONE = "skipped_not_done"               # on, but this container's RESULT is not DONE
RESULT_STATUS_DONE = "DONE"                                       # the RESULT status a persist requires


def compile_cache_persist_evidence(persist: Callable[[], dict] = None, env: Mapping[str, str] = None, *, status: Optional[str] = None) -> Any:
    """The `compile_cache_persist` evidence of an attempt: the persist's own record, or a marker saying why it did not run. The persist
    copies the container's vLLM + FlashInfer compile caches file-for-file onto the volume (10k-35k files on a volume with ~40k inodes
    free -- inode_footprint.md), so it is OFF unless asked for AND the container finished: ``$SEEDREP_PERSIST_COMPILE_CACHES``
    must be "1" (``skipped_by_default`` when unset, ``skipped_by_env`` when set to anything else -- the seed switch's rule), and
    ``status`` must be ``DONE`` (``skipped_not_done`` otherwise, ``None`` included: a caller that does not say is not DONE). fx1,
    2026-09-21: the 11 DRIVERS_FAILED seed-1 rungs each persisted a full cache tree at exit (+22,834 inodes) under the old "1" default.
    A skip is RECORDED in the attempt's evidence the way every other skipped sub-step is, never silent."""
    e = os.environ if env is None else env
    raw = str(e.get(PERSIST_COMPILE_CACHES_ENV, "") or "").strip()
    if raw != "1":
        marker = COMPILE_CACHE_SKIPPED if raw else COMPILE_CACHE_SKIPPED_DEFAULT
        print(f"[compile-cache] persist skipped ({marker}: {PERSIST_COMPILE_CACHES_ENV}={raw or 'unset'})", flush=True)
        return marker
    if status != RESULT_STATUS_DONE:
        print(f"[compile-cache] persist skipped ({COMPILE_CACHE_SKIPPED_NOT_DONE}: status={status!r}); nothing copied to the volume", flush=True)
        return COMPILE_CACHE_SKIPPED_NOT_DONE
    return (persist or persist_compile_caches)()


def persist_compile_caches() -> dict:
    out: dict = {}
    for name, local in COMPILE_CACHES.items():
        src, dst = Path(local), Path(CACHE_VOL) / name
        n = 0
        if src.is_dir():
            for p in src.rglob("*"):
                if p.is_symlink() or not p.is_file():
                    continue
                q = dst / p.relative_to(src)
                try:
                    if q.is_file() and q.stat().st_size == p.stat().st_size:
                        continue
                    q.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, q)
                    n += 1
                except OSError as e:
                    out.setdefault("errors", []).append(f"{name}/{p.relative_to(src)}: {e}")
        out[name] = {"shipped": n}
    try:
        VOL.commit()
    except Exception as e:  # noqa: BLE001
        out.setdefault("errors", []).append(f"commit: {type(e).__name__}: {e}")
    return out


def _build_shared(W: str) -> dict:
    """/workspace/seedrep: a COPY of the staged tree (git-verified) + a COPY of the two banks (symlinks preserved) — the
    strength `_build_shared` shape (`git diff --quiet HEAD` over the mounted volume took > 120 s)."""
    shared = Path(SHARED_LOCAL)
    if not Path(CORPORA_VOL).is_dir():
        # the frozen replay corpora are a precondition of the drivers (renellm loud_skips without them and exits 0
        # having written nothing); a container REFUSES rather than building a tree whose children would silently produce no rows
        raise RuntimeError(f"{CORPORA_VOL} is absent on the volume: the frozen replay corpora (renellm.jsonl) must be staged before any "
                           f"tree is built -- refusing (the preflight's corpora check names what is missing)")
    shutil.rmtree(shared, ignore_errors=True)
    (shared / "scripts").mkdir(parents=True)
    (shared / "scripts" / "hf_env.sh").write_text(f"#!/bin/bash\nexport HF_HOME={HF_VOL}\nexport HF_TOKEN_PATH={SHARED_LOCAL}/hf_token\n")
    tree_src = Path(REPOS) / f"sisteer-{W}"
    repo_dir = Path(lp.tree_local(W))
    if not (tree_src / ".git").exists():
        raise RuntimeError(f"{tree_src} is not a staged checkout (run --stage)")
    shutil.copytree(tree_src, repo_dir, symlinks=True)
    facts = checkout_facts(repo_dir, W)
    if not facts["ok"]:
        raise RuntimeError(f"copied tree does not verify: {facts}")
    exp = shared / "exp"
    exp.mkdir()
    shutil.copytree(Path(OBJECTS_VOL), exp / "objects", symlinks=True)
    os.symlink(CORPORA_VOL, exp / "heldout_corpora", target_is_directory=True)     # == HELDOUT_CORPUS_LOCAL, the drivers' $SISTEER_HELDOUT_CORPUS_DIR
    (shared / ".cache").mkdir()
    os.symlink(UV_CACHE, shared / ".cache" / "uv", target_is_directory=True)
    Path(VENVS_LOCAL).mkdir(parents=True, exist_ok=True)
    return {"repo_dir": str(repo_dir), "tree": facts}


def ensure_driver_venv(tree: Path, W: str) -> dict:
    """common.sh f22_driver_venv's line, in-container: `UV_PROJECT_ENVIRONMENT=<venv> uv sync --frozen --extra cu128` in the
    staged tree from the warm cache (so `sisteer` resolves INSIDE the sha tree), idempotent on its sentinel."""
    venv, sentinel = driver_venv_names(W)
    if Path(sentinel).is_file() and Path(sentinel).read_text().strip() == W and Path(venv, "bin", "python").exists():
        return {"venv": venv, "built": False}
    t0 = time.time()
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": venv, "UV_CACHE_DIR": UV_CACHE, "UV_LINK_MODE": "copy"}
    env.pop("VIRTUAL_ENV", None)
    last = None
    for attempt in range(3):
        r = subprocess.run(["uv", "sync", "--frozen", "--extra", "cu128"], cwd=str(tree), env=env, capture_output=True, text=True)
        last = r
        if r.returncode == 0 and Path(venv, "bin", "python").exists():
            Path(sentinel).write_text(W)
            return {"venv": venv, "built": True, "s": round(time.time() - t0, 1), "attempts": attempt + 1}
        time.sleep(30)
    raise RuntimeError(f"uv sync failed 3 times: {(last.stdout + last.stderr)[-1500:] if last else ''}")


class Samplers:
    """The handle `_start_samplers` returns. `stop(timeout)` ends the metrics scrape and JOINS its thread -- called by `teardown_standup`
    BEFORE the engine goes away (the scrape used to outlive teardown and append `{"t", "error"}` rows to
    vllm_metrics.jsonl for the whole producer-evidence phase, and `utilisation_table.read_metrics` refuses any row without the running
    key, so every bars / utilisation reading over a finished root refused). `n_error_rows` / `n_rows` count what was written while the
    engine was up; the metrics evidence records both. No row of any kind is written once the sampler is stopped."""

    def __init__(self, thread: threading.Thread, own_stop: threading.Event, counts: Dict[str, int]) -> None:
        self.thread, self._own_stop, self._counts = thread, own_stop, counts

    def stop(self, timeout: float) -> bool:
        """Set the sampler's own stop, join; True when the thread ended within ``timeout``."""
        self._own_stop.set()
        self.thread.join(timeout)
        return not self.thread.is_alive()

    @property
    def n_error_rows(self) -> int:
        return int(self._counts["errors"])

    @property
    def n_rows(self) -> int:
        return int(self._counts["rows"])


def _start_samplers(serve_dir: Path, root: Path, stop: threading.Event, proxy_url_ref: dict) -> Samplers:
    """The two in-container SAMPLERS: `nvidia-smi dmon -i 0 -s um -d 10` -> gpu_dmon.log, and a 10 s
    scrape of vLLM /metrics -> vllm_metrics.jsonl (running / waiting / generation & prompt tokens / KV usage). The scrape
    reads `proxy_url_ref["metrics_url"]` with `proxy_url_ref["metrics_headers"]` through `_http_get_text` -- the SAME read
    `verify_metrics_scrape` verified once before this thread started (F-O: the proxy's pass-through, never a bare GET on the
    guarded engine) -- and parses it with `parse_vllm_metrics`. Ends on the row's shared ``stop`` (a `fail` path) or on the
    returned handle's own stop (`Samplers.stop`, teardown); an error row is written ONLY while neither is set (F-AG)."""
    dmon_log = (root / "gpu_dmon.log").open("ab")
    try:
        dmon = subprocess.Popen(["nvidia-smi", "dmon", "-i", "0", "-s", "um", "-d", str(DMON_CADENCE_S)], stdout=dmon_log,
                                stderr=subprocess.STDOUT)
    except FileNotFoundError:
        dmon = None
    own_stop = threading.Event()
    counts = {"errors": 0, "rows": 0}

    def stopped() -> bool:
        return stop.is_set() or own_stop.is_set()

    def scrape():
        with (root / "vllm_metrics.jsonl").open("a") as fh:
            while not own_stop.wait(DMON_CADENCE_S):
                if stop.is_set():
                    break
                url = proxy_url_ref.get("metrics_url")
                if not url:
                    continue
                try:
                    text = _http_get_text(url, timeout=5, headers=proxy_url_ref.get("metrics_headers"))
                except Exception as e:  # noqa: BLE001
                    if stopped():                      # the engine is going away, not failing: never an error row after stop (F-AG)
                        break
                    counts["errors"] += 1
                    fh.write(json.dumps({"t": utc(), "error": f"{type(e).__name__}: {e}"}) + "\n"); fh.flush()
                    continue
                if stopped():
                    break
                row: Dict[str, Any] = {"t": utc(), **parse_vllm_metrics(text)}
                counts["rows"] += 1
                fh.write(json.dumps(row) + "\n"); fh.flush()
        if dmon is not None:
            dmon.terminate()
        dmon_log.close()
    thread = threading.Thread(target=scrape, daemon=True, name="metrics-scrape")
    thread.start()
    return Samplers(thread, own_stop, counts)


# ---------------- the standup's own job record + its endpoint record surviving teardown ----------------
STANDUP_JOBS_FILE = "standup_jobs.jsonl"                                    # <root>/exp/serve/standup_jobs.jsonl (ATTEMPT_ROOT_LAYOUT)


def standup_job_line(endpoint: Mapping[str, Any], cells_served: Mapping[str, Any], row: Mapping[str, Any], *, attempt_id: str,
                     call_id: Optional[str], W: str, node: str, at: Optional[str] = None) -> Dict[str, Any]:
    """The ONE job line of this attempt's ``exp/serve/standup_jobs.jsonl``, in the cluster sbatch's shape (`common.sh` f22 "target up"
    printf; `reconstruct_prev_endpoint.COPIED` names the fields readers copy: job_id / node / key / view / mode / eval / sha /
    lens_reduction_mode / max_num_seqs / max_model_len / attested_* / batched_tokens / cluster; `engine_identity_table` holds `sha` equal
    to ATTEMPT.json's W). The Modal path never wrote it, so `audit_common.read_standup` refused every in-container audit one layer past
    F-U. The job id is the ATTEMPT id (a Modal container has no Slurm job); the values are the standup's own records
    (endpoint.json, cells_served.json) and the plan row -- nothing an operator authors."""
    geom = endpoint.get("geometry") if isinstance(endpoint.get("geometry"), Mapping) else {}
    asked = row.get("geometry") if isinstance(row.get("geometry"), Mapping) else {}
    att_s, att_l = endpoint.get("attested_max_num_seqs"), endpoint.get("attested_max_model_len")
    S, L = asked.get("S", att_s), asked.get("L", att_l)
    eval_legs = row.get("eval_legs") or (getattr(lp, "EVAL_LEGS_OF", {}) or {}).get(row.get("leg"), ()) or (row.get("leg"),)
    return {"job_id": attempt_id, "node": node, "key": endpoint.get("model_key") or lp.MODEL_KEY, "view": row.get("view") or row.get("engine"),
            "mode": row.get("leg"), "eval": ",".join(str(e) for e in eval_legs), "sha": W, "engine": endpoint.get("engine") or "lora_native",
            "serving_path": endpoint.get("serving_path") or "lora_native", "cells_skipped": list(cells_served.get("cells_skipped") or []),
            "geometry_adjustments": geom.get("adjustments"), "lens_reduction_mode": endpoint.get("lens_reduction_mode"),
            "max_num_seqs": S, "max_model_len": L, "attested_max_num_seqs": att_s, "attested_max_model_len": att_l,
            "batched_tokens": (int(S) * int(L)) if isinstance(S, int) and isinstance(L, int) else None, "budget_fraction": None,
            "model_facts": endpoint.get("model_facts"), "cluster": "modal", "gpu_mem_util": endpoint.get("gpu_memory_utilization"),
            "mrope_attestation": None, "thinking_mode": endpoint.get("thinking_mode", endpoint.get("thinking")),
            "thinking_certification_verdict": endpoint.get("thinking_certification_verdict"), "upstream_timeout_s": endpoint.get("upstream_timeout_s"),
            "serving_path_selection": None, "written_at": at or utc(),
            # the Modal path's own identity beside the cluster fields
            "substrate": "modal", "attempt_id": attempt_id, "call_id": call_id, "standup_id": row.get("standup_id"), "engine_id": row.get("engine"),
            "stage": row.get("stage"), "shard": row.get("shard"), "kind": row.get("kind")}


def write_standup_jobs(serve_dir: Path, line: Mapping[str, Any]) -> Path:
    """``<serve dir>/standup_jobs.jsonl`` = exactly the one line (one standup per attempt root; ids are unique per attempt)."""
    p = Path(serve_dir) / STANDUP_JOBS_FILE
    p.write_text(json.dumps(dict(line), default=str) + "\n", encoding="utf-8")
    return p


def restore_endpoint_record(ep: Path, ep_bytes: bytes) -> Optional[Dict[str, Any]]:
    """`mt_serving_up.py` UNLINKS ``endpoint.json`` at teardown (its pollers' down signal, `:1669`). The attempt root keeps the record the
    standup published at READY, byte-identical -- the TreeSyncer shipped exactly those bytes at "ready" and never deletes, so the volume
    already holds them; restoring in-container makes the audits that run after teardown (`request_evidence_audit`, `producer_evidence`)
    read the same tree the volume does. Returns the evidence block when it restored, None when the file was still there."""
    if ep.exists():
        return None
    ep.write_bytes(ep_bytes)
    return {"bytes": len(ep_bytes), "sha256": hashlib.sha256(ep_bytes).hexdigest(), "at": utc(),
            "why": "mt_serving_up.py unlinks endpoint.json at teardown; the READY record is kept for the post-teardown audits (W19 F-AA)"}


#: the rung modes this app executes: the consumer (static / controls / misalign) and the GSM8K queue-derived Inspect assembly
EXECUTABLE_RUNG_MODES = (lp.RUNG_MODE_CONSUME, lp.RUNG_MODE_GSM8K_SUBSET)
GSM8K_ASSEMBLY_VERDICT = "gsm8k_assembly.json"
GSM8K_ASSEMBLY_POLL_S = 30
GSM8K_ASSEMBLY_MARGIN_S = 1800


PRODUCER_RECORDER_REL = lp.PRODUCER_WRITERS["resolved_sampler"]               # scripts/sdfbase/record_resolved_sampler.py (on this tree)
EXPECTED_SAMPLERS_HELPER_REL = "scripts/modal/seedrep_ladder_helpers.py"        # `expected-samplers` -> <root>/expected_samplers.json (driver venv)
PRODUCER_AUDIT_REL = lp.PRODUCER_WRITERS["request_evidence_audit"]            # scripts/sdfbase/seedrep/request_evidence_audit.py (on the tree at W)


def _run_tool(argv: Sequence[str], log: Path, env: Mapping[str, str], tree: Path, run: Callable[..., Any]) -> Optional[int]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as lf:
        lf.write((" ".join(str(x) for x in argv) + "\n").encode())
        try:
            r = run([str(x) for x in argv], env={**dict(env), "CUDA_VISIBLE_DEVICES": ""}, stdout=lf, stderr=subprocess.STDOUT, cwd=str(tree))
        except Exception as e:  # noqa: BLE001 -- the tool refusing to start is recorded, never a container crash
            lf.write(f"[producer-evidence] {type(e).__name__}: {e}\n".encode())
            return None
    return int(r.returncode)


def _bound_file(path: Path, *, writer: str, W: str, rc: Optional[int], why: Optional[str]) -> Dict[str, Any]:
    if path.is_file():
        return {"sha256": sha256_file(path), "writer": writer, "tree_sha": W, "rc": rc}
    return {"absent": True, "why": why or f"{path.name} was not written (tool rc {rc})", "writer": writer, "tree_sha": W, "rc": rc}


DETERMINISM_REL = "scripts/sdfbase/seedrep/determinism_check.py"          # the determinism spot-check (on the tree at W)
DETERMINISM_OUT = "determinism.json"                                        # <root>/determinism.json (ATTEMPT_ROOT_LAYOUT) -- a single-leg container's record
DETERMINISM_COPY_REL = "audits/determinism"                                 # <root>/audits/determinism/<table leg>.json -- what battery_tables reads (RS/audits/determinism/<leg>.json)


def determinism_table_legs(container_leg: str, *, stage: Any = lp.STAGE_BASE) -> Tuple[str, ...]:
    """The TABLE legs one container leg's standup served, in the check's own vocabulary (`audit_common.LEGS`, its `--leg`
    choices): the controls container's three are `CONTROLS_ALL` (`xstest`, `or_bench_hard_1k`, `mmlu` -- the store writers'
    spelling the check discovers stores by; `EVAL_LEGS_OF["controls"]` spells the registry EVAL id `or_bench_hard`, which the
    check's parser refuses) on the BASE row, and the two `CONTROLS_LADDER` legs on a RUNG (F-T: a rung store holds no MMLU, so a
    check asked for it there would find no store); every other container leg's table legs are its eval legs (misalign: the two behaviours)."""
    if container_leg == "controls":
        return lp.audit_table_legs("controls", stage=stage)
    return tuple(lp.EVAL_LEGS_OF.get(container_leg, (container_leg,)))


def determinism_check_argv(*, driver_python: str, tree: Path, table_leg: str, root: Path, proxy_url: str, metrics_url: str, out: Path,
                           copy_to: Optional[Path] = None, petri_behaviours: Sequence[str] = (), n: int = lp.DETERMINISM_N_PLANNED,
                           item_budget: Optional[int] = None) -> List[str]:
    """The exact argv of `determinism_check.py` for ONE table leg of this attempt root (the app used to pass
    `--attempt-root` / `--routing-table`, neither of which the parser takes, and no `--metrics-url`, which it requires -- so the
    check exited on argparse and no record was ever written). `--root` = the attempt root (its ATTEMPT.json, endpoint.json, stores,
    request evidence and routing table are all discovered UNDER it; the table is not a parser input); `--proxy-url` = the live
    RoutingProxy /v1 base; `--metrics-url` = the engine's /metrics THROUGH THE PROXY's pass-through (`metrics_url_from_endpoint`:
    the drained-state read; F-O -- the engine's own page is bearer-guarded and the check holds no engine token). No
    `--tokenize-url`: the proxy's /tokenize pass-through takes chat `messages`, not the `{prompt}` body the check sends, so the
    token basis is the check's declared WHITESPACE basis. The proxy bearer, when any, rides the env (`$VLLM_API_KEY`), never argv.
    `--n` = the items the check selects per arm: the plan's 20 on every production row, `min(20, <the row's item budget>)` on a
    BUDGET-LIMITED smoke row (an S1-S4 BoN / misalign row under `--limit-goals 10` / `--limit-items 10`, whose store files 10 items by
    construction -- the S3 bon smoke's check refused "files 10 item(s), fewer than the 20", so S3's PASS could never derive at the smoke
    budget), with `--item-budget <budget>` naming the budget so the record says `budget_limited: true`. Both come from the ROW
    (`lp.determinism_spot_check_n`), never from the store's size: a production row's short store must still refuse."""
    argv = [driver_python, str(Path(tree) / DETERMINISM_REL), "--root", str(root), "--leg", table_leg, "--proxy-url", proxy_url,
            "--metrics-url", metrics_url, "--out", str(out), "--n", str(int(n))]
    if item_budget is not None:
        argv += ["--item-budget", str(int(item_budget))]
    if copy_to is not None:
        argv += ["--copy-to", str(copy_to)]
    if table_leg == "petri" and petri_behaviours:
        argv += ["--petri-behaviours", ",".join(str(b) for b in petri_behaviours)]
    return argv


def determinism_evidence(root: Path, *, tree: Path, W: str, row: Mapping[str, Any], driver_python: str, env: Mapping[str, str],
                         proxy_url: str, metrics_url: str, run: Callable[..., Any] = subprocess.run) -> Dict[str, Any]:
    """The §9.2 spot-check, run IN-CONTAINER against the live standup before teardown, once per TABLE leg of the container, its rc
    CONSULTED and its record BOUND the way `producer_evidence` binds the audit: `modal_evidence.json["determinism"][<table leg>]`
    = `_bound_file` (sha256 + writer + tree sha + the tool's rc, or ABSENT by name with the rc) over `<root>/audits/determinism/<leg>.json`
    (a single-leg container also gets the layout's `<root>/determinism.json`, the same bytes via `--copy-to`). A tool absent from the
    tree is recorded ABSENT by name; a non-zero rc (2 = the check REFUSED: engine not drained, a body without its wire bytes, no
    store) leaves the file unwritten and the record says so -- the determinism audit then finds no record, loudly, instead of a
    spot-check that silently never ran."""
    out: Dict[str, Any] = {}
    det = Path(tree) / DETERMINISM_REL
    legs = determinism_table_legs(row["leg"], stage=row["stage"])
    logs = root / "children"
    # the selection count from the ROW's budget -- 20 on a production row (and on every S5 full-shape / non-budgeted leg),
    # min(20, budget) with the budget named on an S1-S4 BoN / misalign smoke row; the check never reads the store's size for this
    n_select, item_budget = lp.determinism_spot_check_n(row)
    for leg in legs:
        rec_path = root / DETERMINISM_COPY_REL / f"{leg}.json"
        copy_to = (root / DETERMINISM_OUT) if len(legs) == 1 else None
        rc: Optional[int] = None
        argv: Optional[List[str]] = None
        if det.is_file():
            argv = determinism_check_argv(driver_python=driver_python, tree=tree, table_leg=leg, root=root, proxy_url=proxy_url,
                                          metrics_url=metrics_url, out=rec_path, copy_to=copy_to,
                                          petri_behaviours=[str(b) for b in (row.get("behaviours") or [])], n=n_select, item_budget=item_budget)
            rc = _run_tool(argv, logs / "determinism.log", env, tree, run)
        bound = _bound_file(rec_path, writer=DETERMINISM_REL, W=W, rc=rc,
                            why=None if det.is_file() else f"{DETERMINISM_REL} is not on the tree at {W[:12]} (WS3-G)")
        bound["argv"] = argv
        if copy_to is not None:
            bound["copy"] = str(copy_to.relative_to(root)) if copy_to.is_file() else None
        out[leg] = bound
        print(f"[determinism] {row['leg']}/{leg}: n={n_select}{' (budget-limited, item budget ' + str(item_budget) + ', D122)' if item_budget is not None else ''} "
              f"rc={rc} {'record ' + str(rec_path.relative_to(root)) if rec_path.is_file() else 'NO RECORD (' + str(bound.get('why')) + ')'}",
              flush=True)
    return out


#: F-AO: the audit's WRITER-PRODUCED index of the verdicts it filed (`request_evidence_audit.INDEX_NAME`, held equal by
#: tests/test_seedrep_fp3_verdict_binding.py) -- `<root>/audits/request_log_index.json`
AUDIT_INDEX_NAME = "request_log_index.json"


def _bind_child_verdict(root: Path, *, entries: Optional[List[Mapping[str, Any]]], idx_problem: Optional[str], vmodels: Optional[Mapping[str, Any]],
                        vm_problem: Optional[str], leg: str, cell: str, plan_arm: str, s: int, stage: str, shard_dir: str, plan_rel: str, W: str,
                        rc: Optional[int], tool_why: Optional[str]) -> Dict[str, Any]:
    """ONE (plan child, table leg) -> the verdict FILE the audit filed for it, through the audit's index (F-AO). The plan child's CELL (the
    TrainTest id the drivers were handed; `arditi_allpos` for the Arditi row, whose plan ARM is the label `ar_c0p3__arditi_allpos`) matches an
    index entry in EITHER spelling (`traintest_id` / `harness_id`) for the same (leg, s, stage, shard); exactly one must match. The file is then
    held to the index (its stamped `arm_binding`, its sha), the binding's vmodel to the standup's own `endpoint.json.vmodels`, and the binding's
    cell to the plan child's. None, two, or a disagreement is ABSENT BY NAME (the certificate refuses that check) -- never a guess. Without an
    index (an audit that refused before writing, or one older than F-AO) only the plan's own name can bind, its cross-check recorded as
    not available."""
    base: Dict[str, Any] = {"writer": PRODUCER_AUDIT_REL, "tree_sha": W, "rc": rc, "plan_arm": plan_arm, "plan_cell": cell}

    def absent(why: str) -> Dict[str, Any]:
        return {"absent": True, "why": why, **base}

    def _s(m: Mapping[str, Any], key: str) -> Optional[str]:
        return str(m.get(key)) if m.get(key) is not None else None

    if tool_why:
        return absent(tool_why)
    if entries is None:
        p = root / plan_rel
        if p.is_file():
            return {"sha256": sha256_file(p), **base, "bound_rel": plan_rel,
                    "arm_binding_check": f"not_available: {idx_problem} -- bound by the plan's own name without the writer's binding"}
        return absent(f"{Path(plan_rel).name} was not written (tool rc {rc}); {idx_problem}, so no other spelling can be resolved")
    hits = [e for e in entries if isinstance(e, Mapping) and e.get("leg") == leg and str(e.get("s")) == str(s) and str(e.get("stage")) == stage
            and str(e.get("shard")) == shard_dir and cell in ({_s(e, "traintest_id"), _s(e, "harness_id")} - {None})]
    if not hits:
        others = sorted({str(e.get("arm")) for e in entries if isinstance(e, Mapping) and e.get("leg") == leg})
        return absent(f"no verdict for ({leg}, cell {cell!r} (plan arm {plan_arm!r}), s{s}, {stage}, {shard_dir}) under either spelling in "
                      f"{AUDIT_INDEX_NAME} (tool rc {rc}; the audit filed {len(others)} {leg} verdict(s): {others[:6]})")
    if len(hits) > 1:
        return absent(f"{len(hits)} verdicts in {AUDIT_INDEX_NAME} bind ({leg}, cell {cell!r}, s{s}, {stage}, {shard_dir}): "
                      f"{[e.get('rel') for e in hits]} -- ambiguous, never resolved by picking one")
    e = hits[0]
    rel = str(e.get("rel") or "")
    bound_rel = f"{lp.AUDIT_OUT_REL}/{rel}"
    p = root / bound_rel
    if not rel or not p.is_file():
        return absent(f"{AUDIT_INDEX_NAME} names {rel!r} for ({leg}, cell {cell!r}, s{s}) but no such file is under {lp.AUDIT_OUT_REL}/")
    try:
        doc = json.loads(p.read_text())
    except ValueError as ex:
        return absent(f"{bound_rel} is not JSON ({ex})")
    ab = doc.get("arm_binding") if isinstance(doc, dict) else None
    if not isinstance(ab, Mapping):
        return absent(f"{bound_rel} carries no arm_binding -- a verdict written before F-AN; the store's arm cannot be bound to cell {cell!r}")
    problems: List[str] = []
    for key in ("harness_id", "traintest_id", "vmodel", "arm"):
        if _s(ab, key) != _s(e, key):
            problems.append(f"arm_binding.{key} {ab.get(key)!r} != index {e.get(key)!r}")
    if doc.get("arm") != e.get("arm"):
        problems.append(f"verdict arm {doc.get('arm')!r} != index arm {e.get('arm')!r}")
    sha = sha256_file(p)
    if e.get("sha256") and e.get("sha256") != sha:
        problems.append(f"file sha {sha[:12]} != index sha {str(e.get('sha256'))[:12]} (rewritten since the audit filed it)")
    expected_cell = _s(e, "traintest_id") if _s(e, "traintest_id") is not None else _s(e, "harness_id")
    if expected_cell != cell:
        problems.append(f"the binding's cell {expected_cell!r} is not the plan child's {cell!r}")
    if vmodels is None:
        problems.append(str(vm_problem))
    elif vmodels.get(e.get("harness_id")) != e.get("vmodel"):
        problems.append(f"{SERVE_DIR}/endpoint.json.vmodels[{e.get('harness_id')!r}] = {vmodels.get(e.get('harness_id'))!r} != the binding's vmodel {e.get('vmodel')!r}")
    if problems:
        return absent(f"{bound_rel}: the verdict's arm_binding does not bind cell {cell!r} (plan arm {plan_arm!r}): " + "; ".join(problems))
    id_kind = "TrainTest" if _s(e, "traintest_id") is not None else "harness"
    return {"sha256": sha, **base, "bound_rel": bound_rel, "store_arm": e.get("arm"), "harness_id": e.get("harness_id"), "traintest_id": e.get("traintest_id"),
            "vmodel": e.get("vmodel"), "spelling": e.get("spelling"),
            "arm_binding_check": (f"agrees: verdict arm_binding == {AUDIT_INDEX_NAME} entry; {SERVE_DIR}/endpoint.json.vmodels[{e.get('harness_id')!r}] == "
                                  f"{e.get('vmodel')!r}; plan cell {cell!r} == the binding's {id_kind} id")}


def bind_audit_verdicts(root: Path, *, row: Mapping[str, Any], W: str, rc_a: Optional[int], aud_present: bool) -> Dict[str, Any]:
    """Every (plan child, table leg) verdict binding of a row, keyed by the PLAN's rel (`lp.audit_verdict_rels(k["eval_leg"], k["arm"], …)` -- the
    key the certificate and the carry-over look up), each record naming the FILE the audit actually filed (`bound_rel`) and the checks that bound
    it (F-AO; :func:`_bind_child_verdict`). The standup's `endpoint.json.vmodels` is read once as the independent cross-check of every binding's
    vmodel."""
    out: Dict[str, Any] = {}
    out_dir = root / lp.AUDIT_OUT_REL
    idx_p = out_dir / AUDIT_INDEX_NAME
    entries: Optional[List[Mapping[str, Any]]] = None
    idx_problem: Optional[str] = None
    if idx_p.is_file():
        try:
            idx = json.loads(idx_p.read_text())
            ents = idx.get("verdicts") if isinstance(idx, dict) else None
            if isinstance(ents, list):
                entries = ents
            else:
                idx_problem = f"{AUDIT_INDEX_NAME} carries no `verdicts` list"
        except ValueError as e:
            idx_problem = f"{AUDIT_INDEX_NAME} is not JSON ({e})"
    else:
        idx_problem = f"no {lp.AUDIT_OUT_REL}/{AUDIT_INDEX_NAME} (the audit wrote no index: it refused before writing, or predates F-AO)"
    vmodels: Optional[Mapping[str, Any]] = None
    vm_problem: Optional[str] = None
    ep_p = root / SERVE_DIR / "endpoint.json"
    try:
        ep = json.loads(ep_p.read_text())
        vm = ep.get("vmodels") if isinstance(ep, dict) else None
        if isinstance(vm, dict) and vm:
            vmodels = vm
        else:
            vm_problem = f"{SERVE_DIR}/endpoint.json carries no `vmodels` map -- the binding's vmodel has no standup record to be held to"
    except (OSError, ValueError) as e:
        vm_problem = f"{SERVE_DIR}/endpoint.json unreadable ({type(e).__name__}: {e}) -- the binding's vmodel has no standup record to be held to"
    tool_why = None if aud_present else f"{PRODUCER_AUDIT_REL} is not on the tree at {W[:12]} (WS3-G)"
    shard_dir = f"shard{int(row['shard']):02d}"
    for k in row.get("children") or []:
        cell = str(k.get("cell") or k["arm"])          # the SERVED cell the drivers were handed; an older child shape carries the arm only
        for leg, plan_rel in lp.audit_verdict_rels(k["eval_leg"], k["arm"], k["s"], stage=row["stage"], shard=row["shard"]):
            out[plan_rel] = _bind_child_verdict(root, entries=entries, idx_problem=idx_problem, vmodels=vmodels, vm_problem=vm_problem, leg=leg, cell=cell,
                                                plan_arm=str(k["arm"]), s=int(k["s"]), stage=str(row["stage"]), shard_dir=shard_dir, plan_rel=plan_rel,
                                                W=W, rc=rc_a, tool_why=tool_why)
    return out


def producer_evidence(root: Path, *, tree: Path, W: str, row: Mapping[str, Any], serve_dir: Path, driver_python: str, env: Mapping[str, str],
                      run: Callable[..., Any] = subprocess.run) -> Dict[str, Any]:
    """The PRODUCER writes and BINDS the smoke evidence under its own attempt root, after the drivers and before the log is
    packed: `record_resolved_sampler.py --require-wired` over the standup's vLLM log -> `<root>/resolved_sampler.json`;
    `request_evidence_audit.py --roots <root> --out <root>/audits` -> one verdict per child store (`audits/request_log/<leg>/<arm>__s<s>...json`);
    the container's own `exp/serve/cells.json`. Each file's sha256 + writer + tree sha (+ the tool's rc) is recorded under
    `modal_evidence.json["producer_evidence"]`; a tool absent from the tree, or a file it did not write, is recorded ABSENT by name -- the
    certificate then refuses that check (nothing an operator supplies can stand in)."""
    out: Dict[str, Any] = {}
    logs = root / "children"
    rec = Path(tree) / PRODUCER_RECORDER_REL
    rs_out = root / lp.RESOLVED_SAMPLER_REL
    rc = None
    if rec.is_file():
        argv: List[str] = [driver_python, str(rec), "--vllm-log", str(serve_dir / "vllm_server.log"), "--out", str(rs_out), "--require-wired"]
        # the CLIENT class's expected sampler PER TASK -- a controls row sends two samplers by design (the over-refusal tasks under
        # the vendor row + MMLU greedy at max_tokens 1), a single reference does not exist, and every controls / bon row read UNCOMPARED
        # until this. The driver-venv helper resolves each child task's reference from the drivers' own authorities (the config's controls
        # row, the registry's rows by exact ref) into `<root>/expected_samplers.json`; the recorder compares each task against ITS reference
        # and records a verdict per task. A helper that fails leaves the recorder without expectations: it then reads UNCOMPARED (loud),
        # never AGREES.
        exp_out = root / lp.EXPECTED_SAMPLERS_REL
        helper = Path(tree) / EXPECTED_SAMPLERS_HELPER_REL
        rc_e: Optional[int] = None
        if helper.is_file():
            row_p = logs / "plan_row.json"
            logs.mkdir(parents=True, exist_ok=True)
            row_p.write_text(json.dumps(dict(row), default=str))
            rc_e = _run_tool([driver_python, str(helper), "expected-samplers", "--row", str(row_p), "--tree", str(tree), "--out", str(exp_out)],
                             logs / "resolved_sampler.log", env, tree, run)
        if rc_e == 0 and exp_out.is_file():
            argv += ["--expect-tasks", str(exp_out)]
            # fc4: a gsm8k row's RENDER PROBE (cap 512, decoded under the thinking card) is classified by the driver's own tag and compared against the
            # pinned card row -- `--probe-expect kinds --probe-tags <each child's request_evidence.jsonl> --expect <card>` -- all derived from the plan
            # row by the helper (`gsm8k_probe_block`); rows without the block (controls / static / bon) keep the default probe reference unchanged
            try:
                probe = (json.loads(exp_out.read_text()) or {}).get("probe")
            except ValueError:
                probe = None
            if isinstance(probe, Mapping) and probe.get("probe_expect"):
                argv += ["--probe-expect", str(probe["probe_expect"])]
                for rel in probe.get("tags_rel") or []:
                    argv += ["--probe-tags", str(root / rel)]
                for k, v in sorted((probe.get("expect") or {}).items()):
                    argv += ["--expect", f"{k}={v}"]
                if str(probe["probe_expect"]) == "kinds":
                    # fg1 / F-AV: the standup's OWN untagged probe kinds, declared from THIS root's records (exp/serve/endpoint.json + positive_controls.json)
                    argv += declare_kind_argv(probe_declaration_from_root(root), client_caps=_expected_client_caps(exp_out))
        else:
            with (logs / "resolved_sampler.log").open("a") as lf:
                lf.write(f"[producer-evidence] no per-task sampler expectations ({EXPECTED_SAMPLERS_HELPER_REL} {'absent on the tree' if not helper.is_file() else f'rc {rc_e}'}): "
                         "the recorder runs without --expect-tasks and reads UNCOMPARED on a multi-sampler row, never AGREES\n")
            for k, v in sorted((row.get("frozen_sampler") or {}).items()):             # the client class's expected row when the plan pins one
                argv += ["--expect", f"{k}={v}"]
        rc = _run_tool(argv, logs / "resolved_sampler.log", env, tree, run)
        out[lp.EXPECTED_SAMPLERS_REL] = _bound_file(exp_out, writer=EXPECTED_SAMPLERS_HELPER_REL, W=W, rc=rc_e,
                                                    why=(f"{EXPECTED_SAMPLERS_HELPER_REL} is not on the tree at {W[:12]}" if not helper.is_file() else None))
    out[lp.RESOLVED_SAMPLER_REL] = _bound_file(rs_out, writer=PRODUCER_RECORDER_REL, W=W, rc=rc, why=None if rec.is_file() else f"{PRODUCER_RECORDER_REL} is not on the tree at {W[:12]}")
    aud = Path(tree) / PRODUCER_AUDIT_REL
    rc_a = None
    if aud.is_file():
        # the audit's `--legs` vocabulary is the TABLE legs (`or_bench_hard_1k`, not EVAL_LEGS_OF's eval id `or_bench_hard`, which it refuses)
        argv = [driver_python, str(aud), "--roots", str(root), "--out", str(root / lp.AUDIT_OUT_REL), "--legs", ",".join(determinism_table_legs(row["leg"], stage=row["stage"]))]
        if row["leg"] == "petri" and row.get("behaviours"):
            argv += ["--petri-behaviours", ",".join(str(b) for b in row["behaviours"])]
        rc_a = _run_tool(argv, logs / "request_evidence_audit.log", env, tree, run)
    # one binding per verdict the audit WRITES: per TABLE leg (a controls child has three; F4), the FILE resolved through the audit's own index
    # (F-AO: the plan child's cell in either spelling -> the verdict the audit filed under the STORE's spelling) -- never a caller-composed name
    out.update(bind_audit_verdicts(root, row=row, W=W, rc_a=rc_a, aud_present=aud.is_file()))
    if row.get("kind") == lp.S5_PROBE_KIND:
        # the window MEASUREMENT per probe child -- `seedrep_s5_measure.py` over the child's COMPLETE
        # multi-turn store + the served target's per-call `target_timing.jsonl` -> `<root>/audits/s5_window/<leg>/<cell>__s<s>.json`, bound here
        # exactly like the audit (sha256 + writer + tree sha + rc, or ABSENT by name); `seedrep_s5_record.py window` reads THIS file off the accepted
        # attempt and nothing an operator authors
        mea_rel = lp.PRODUCER_WRITERS["s5_window_measurement"]
        mea = Path(tree) / mea_rel
        for k, rel in lp.s5_window_rels(row):
            rc_m = None
            if mea.is_file():
                argv = lp.s5_measure_argv(driver_python=driver_python, tree=tree, root=root, row=row, child=k, out=root / rel)
                rc_m = _run_tool(argv, logs / "s5_measure.log", env, tree, run)
            out[rel] = _bound_file(root / rel, writer=mea_rel, W=W, rc=rc_m, why=None if mea.is_file() else f"{mea_rel} is not on the tree at {W[:12]} (W16s5)")
            print(f"[s5-measure] {row['standup_id']} {k['eval_leg']}/{k['cell']} s{k['s']}: rc={rc_m} {'record ' + rel if (root / rel).is_file() else 'NO RECORD (' + str(out[rel].get('why')) + ')'}",
                  flush=True)
    if row.get("pool"):
        # F-M: a pool-dependent row SNAPSHOTS its pool's live endpoint record (the pool Server's endpoint.json on the shared volume)
        # byte-for-byte into this attempt root and binds it, so the certificate keeps the attacker's deployment shape -- model revision, cards,
        # allocator config, window, S, app / server / container identity -- and refuses changed bytes; the plan's pool row carries only a PATH
        src = Path(lp.pool_endpoint_record(row["pool"], vol_path=VOL_PATH))
        snap = root / lp.POOL_ENDPOINT_SNAPSHOT_REL
        if src.is_file():
            snap.write_bytes(src.read_bytes())
        out[lp.POOL_ENDPOINT_SNAPSHOT_REL] = _bound_file(snap, writer=lp.PRODUCER_WRITERS["pool_endpoint"], W=W, rc=0 if src.is_file() else None,
                                                          why=None if src.is_file() else f"pool {row['pool']}'s endpoint record is absent at {src}")
    out[lp.CELLS_JSON_REL] = _bound_file(root / lp.CELLS_JSON_REL, writer="scripts/modal/seedrep_served.py", W=W, rc=0, why="the standup never enumerated its cells")
    return out


def gsm8k_child_serve_dir(root: Path, child: Mapping[str, Any], *, serve_dir: Path) -> Path:
    """The child's OWN copy of the standup record beside its routing table (a ledger is bound by the record `ledger_endpoint.json` BESIDE
    `routing_table.json`, and the driver holds `endpoint.routing_table` to `--routing-table`; eight children share one standup here, so each gets
    `<root>/exp/serve_gsm8k/<cell>__s<s>/{routing_table.json (the bytes), endpoint.json (routing_table rewritten to the copy)}`)."""
    d = root / child["ledger"]["serve_dir_rel"]
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(serve_dir / "routing_table.json", d / "routing_table.json")
    ep = json.loads((serve_dir / "endpoint.json").read_text())
    ep["routing_table"] = str(d / "routing_table.json")
    ep["ledger_serve_copy_of"] = str(serve_dir / "endpoint.json")
    write_json(d / "endpoint.json", ep)
    return d


def gsm8k_ledger_argv(child: Mapping[str, Any], *, root: Path, driver_python: str, proxy_url: str, port: int, attempt_id: str) -> List[str]:
    """The `st_ledger_up` line: `python -m sisteer.capeval.serving.usage_ledger --upstream <proxy> --port <free> --host 127.0.0.1 --ledger-path
    <out root>/usage_ledger.jsonl --endpoint-json <serve dir>/ledger_endpoint.json --attempt-id a<k>`."""
    led = child["ledger"]
    return [driver_python, "-m", "sisteer.capeval.serving.usage_ledger", "--upstream", proxy_url, "--port", str(int(port)), "--host", "127.0.0.1",
            "--ledger-path", str(root / led["ledger_path_rel"]), "--endpoint-json", str(root / led["endpoint_json_rel"]), "--attempt-id", attempt_id]


def gsm8k_attempt_begin(out_root: Path, facts: Mapping[str, Any]) -> str:
    """The `st_attempt_begin` step: `<out root>/ladder_state.json` (`g8k_ladder_state/1`, attempts[], accepted_generation_attempt)
    -> the minted `a<k>` (k = the attempts recorded on THIS store + 1; a fresh immutable attempt root mints a1)."""
    path = Path(out_root) / lp.GSM8K_LADDER_STATE_FILENAME
    doc = json.loads(path.read_text()) if path.is_file() else {"schema": "g8k_ladder_state/1", "attempts": [], "accepted_generation_attempt": None}
    attempts = doc.setdefault("attempts", [])
    aid = f"a{len(attempts) + 1}"
    attempts.append({"attempt_id": aid, "started": utc(), "ended": None, "rc": None, "result": None, **dict(facts)})
    write_json(path, doc)
    return aid


def gsm8k_attempt_end(out_root: Path, attempt_id: str, *, rc: int, result: str) -> Dict[str, Any]:
    """The `st_attempt_end` step: the attempt's rc / result, and on rc 0 `accepted_generation_attempt` = the store's manifest's own
    (`cells/*/manifest.json`, the driver's `--attempt-id` publication) when it names one, else this attempt."""
    path = Path(out_root) / lp.GSM8K_LADDER_STATE_FILENAME
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text())
    for a in doc.get("attempts", []):
        if a.get("attempt_id") == attempt_id:
            a.update(ended=utc(), rc=int(rc), result=str(result))
    if rc == 0:
        accepted, source = attempt_id, "launcher (the driver's manifest named no accepted_generation_attempt)"
        for m in sorted(Path(out_root).glob("cells/*/manifest.json")):
            try:
                named = json.loads(m.read_text()).get("accepted_generation_attempt")
            except (OSError, ValueError):
                named = None
            if named:
                accepted, source = str(named), str(m)
                break
        doc["accepted_generation_attempt"] = accepted
        doc["accepted_generation_attempt_source"] = source
    write_json(path, doc)
    return doc


def relay_port_for(bound: int) -> int:
    """The Petri relay's listen port (F-AL): a FREE OS-assigned port that is not the proxy's own -- never ``bound + RELAY_OFFSET``,
    which exceeded 65535 whenever the proxy drew a high ephemeral port (a petri row drew 71000 -> `modal.forward` InvalidError, the
    container died with no RESULT). Always in 1..65535 by construction."""
    for _ in range(64):
        p = free_port()
        if p != int(bound) and 0 < p <= 65535:
            return p
    raise RuntimeError(f"no free relay port distinct from the proxy port {bound}")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


def gsm8k_child_up(child: Mapping[str, Any], *, root: Path, serve_dir: Path, proxy_url: str, driver_python: str, tree: Path, env: Mapping[str, str],
                   popen: Callable[..., Any] = subprocess.Popen, wait_s: float = lp.GSM8K_LEDGER_WAIT_S, sleep: Callable[[float], Any] = time.sleep) -> Dict[str, Any]:
    """The per-store pre-driver seam for ONE GSM8K child: the attempt id minted into the store's ladder_state.json, the child's serve-dir copy,
    its usage ledger started with that attempt id and waited for (its endpoint record, bounded), and the per-child spawn substitutions
    (`{ledger_url}`, `{child_routing_table}`, `{gsm8k_attempt_id}`). Raises RuntimeError when the ledger never writes its record (LEDGER_FAILED)."""
    out_root = root / child["out_rel"]
    out_root.mkdir(parents=True, exist_ok=True)
    aid = gsm8k_attempt_begin(out_root, {"node": socket.gethostname(), "substrate": "modal", "cell": child["cell"], "s": child["s"], "cap": child["cap"]})
    d = gsm8k_child_serve_dir(root, child, serve_dir=serve_dir)
    ep_path = root / child["ledger"]["endpoint_json_rel"]
    ep_path.unlink(missing_ok=True)
    argv = gsm8k_ledger_argv(child, root=root, driver_python=driver_python, proxy_url=proxy_url, port=free_port(), attempt_id=aid)
    log = (d / "ledger.log").open("ab")
    proc = popen(argv, env={**dict(env), "CUDA_VISIBLE_DEVICES": ""}, stdout=log, stderr=subprocess.STDOUT, cwd=str(tree), start_new_session=True)
    state = {"attempt_id": aid, "ledger_proc": proc, "ledger_log": log, "serve_dir": str(d), "out_root": str(out_root)}
    try:
        deadline = time.monotonic() + float(wait_s)
        while not ep_path.is_file():
            if proc.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError(f"LEDGER_FAILED {lp.child_key(child)}: no {ep_path} after {wait_s}s (ledger rc {proc.poll()})")
            sleep(1.0)
        try:
            rec = json.loads(ep_path.read_text())
        except ValueError as e:
            raise RuntimeError(f"LEDGER_FAILED {lp.child_key(child)}: {ep_path} is not JSON ({e})")
        url = rec.get("ledger_url") if isinstance(rec, Mapping) else None
        if not isinstance(url, str) or not url:
            raise RuntimeError(f"LEDGER_FAILED {lp.child_key(child)}: {ep_path} carries no ledger_url")
    except BaseException:
        # a ledger that never wrote a valid record is not left running, and the attempt it was minted for is ENDED
        # (st_attempt_end with the LEDGER_FAILED word) -- the next attempt on this store mints a<k+1> against a closed a<k>
        gsm8k_child_down(state, rc=1, result="LEDGER_FAILED")
        raise
    return {**state, "ledger_url": url,
            "subs": {lp.LEDGER_URL_PLACEHOLDER: url, lp.CHILD_ROUTING_TABLE_PLACEHOLDER: str(d / "routing_table.json"), lp.GSM8K_ATTEMPT_ID_PLACEHOLDER: aid}}


def gsm8k_child_down(state: Mapping[str, Any], *, rc: Optional[int], result: str) -> None:
    """The child's ledger TERM'd (st_ledger_down) and its attempt ended in ladder_state.json."""
    proc = state.get("ledger_proc")
    if proc is not None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, AttributeError):
            pass
        try:
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
    if state.get("ledger_log") is not None:
        try:
            state["ledger_log"].close()
        except Exception:  # noqa: BLE001
            pass
    gsm8k_attempt_end(Path(state["out_root"]), state["attempt_id"], rc=(int(rc) if rc is not None else 1), result=result)


def gsm8k_budget_prestep(kids: Sequence[Mapping[str, Any]], *, root: Path, driver_python: str, tree: Path, env: Mapping[str, str], W: str,
                         geometry_eval: str, run: Callable[..., Any] = subprocess.run, client_python: str = "") -> List[str]:
    """G-FIT before any GSM8K child (`g8k_budget_artifact`): ONE prompt-budget run per distinct (cap, cell) the children name
    (`kid["budget_argv"]`, placeholders resolved), then the `.fit.json` sidecar held to `lp.gsm8k_budget_problems` (fits, cap, entry, arm on,
    tree_sha == W). Returns the problems (empty = every artifact fits); a non-fit or a failed tool run names the (cap, cell)."""
    problems: List[str] = []
    done: Dict[str, bool] = {}
    for k in kids:
        rel = k.get("budget_rel")
        if not rel or rel in done:
            continue
        done[rel] = True
        raw = list(k.get("budget_argv") or [])
        if not raw:
            problems.append(f"{k['arm']} s{k['s']}: no budget_argv on the child")
            continue
        if any(lp.CLIENT_PY_PLACEHOLDER in a for a in raw) and not client_python:
            # the G-FIT tool imports inspect_ai -- it runs under the tinycap CLIENT venv, never the driver venv
            problems.append(f"({k['cap']}, {k['cell']}): budget_argv needs the tinycap CLIENT venv python ({lp.CLIENT_PY_PLACEHOLDER}) but none was built")
            continue
        argv = [a.replace(lp.CLIENT_PY_PLACEHOLDER, client_python).replace(lp.DRIVER_PY_PLACEHOLDER, driver_python).replace(lp.ATTEMPT_ROOT_PLACEHOLDER, str(root)) for a in raw]
        out = root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with (out.parent / "budget.log").open("ab") as lf:
            r = run(argv, env=dict(env), stdout=lf, stderr=subprocess.STDOUT, cwd=str(tree))
        side = Path(str(out) + ".fit.json")
        sidecar = json.loads(side.read_text()) if side.is_file() else None
        probs = lp.gsm8k_budget_problems(sidecar, cap=int(k["cap"]), geometry_eval=geometry_eval, W=W)
        if r.returncode != 0 or probs:
            problems.append(f"({k['cap']}, {k['cell']}): budget tool rc {r.returncode}; {'; '.join(probs) or 'sidecar ok'}")
    return problems


def gsm8k_assembly_argv(row: Mapping[str, Any], ctx: Mapping[str, Any], *, tree: Path, client_python: str, root: Path, deadline_s: float) -> List[str]:
    """The assembly helper's argv (`scripts/modal/seedrep_gsm8k_assembly.py` under the tinycap CLIENT venv -- the helper is
    Inspect-dependent, `inspect_rung.classify_store` -> `store_texts` -> `inspect_served.reusable_log` imports `inspect_ai.log` BEFORE any escalation
    decision, and the driver venv carries no inspect_ai): this queue's identity from the ROW, the two Dicts, the slots from the row's assembly
    block, the verdict under the attempt root. Refuses by name without a client python (the driver venv is not a stand-in)."""
    if not client_python:
        raise ValueError(f"{row.get('standup_id')}: the GSM8K assembly helper runs under the tinycap CLIENT venv (inspect_ai / inspect_evals) but no "
                         "client python was built for this row -- the driver venv is not a stand-in (launch F-I)")
    asm = row["rung_assembly"]
    return [client_python, str(Path(tree) / lp.GSM8K_ASSEMBLY_HELPER_REL), "--tree", str(tree), "--vol-root", VOL_PATH, "--results-root", ctx["results_root"],
            "--W7", ctx["W"][:7], "--wave-uuid", ctx["wave_uuid"], "--leg", row["leg"], "--view", row["engine"], "--cap-next", str(int(row["stage"])),
            "--producers", ",".join(asm["feeder_shards"]), "--accepted-dict", ctx["dicts"]["accepted"], "--results-dict", ctx["dicts"]["results"],
            "--attempt-root", str(root), "--slots", json.dumps(asm["slots"]), "--out", str(root / GSM8K_ASSEMBLY_VERDICT),
            "--poll-s", str(GSM8K_ASSEMBLY_POLL_S), "--deadline-s", str(int(max(60, deadline_s)))]


def run_gsm8k_assembly(row: Mapping[str, Any], ctx: Mapping[str, Any], *, tree: Path, client_python: str, root: Path, env: Mapping[str, str],
                       deadline_s: float, run: Callable[..., Any] = subprocess.run) -> Tuple[Optional[Dict[str, Any]], int]:
    """Run the helper UNDER THE CLIENT VENV; returns (the verdict document | None when it wrote none, the rc)."""
    argv = gsm8k_assembly_argv(row, ctx, tree=tree, client_python=client_python, root=root, deadline_s=deadline_s)
    (root / "children").mkdir(parents=True, exist_ok=True)
    with (root / "children" / "gsm8k_assembly.log").open("ab") as lf:
        r = run(argv, env=dict(env), stdout=lf, stderr=subprocess.STDOUT, cwd=str(tree))
    vp = root / GSM8K_ASSEMBLY_VERDICT
    return (json.loads(vp.read_text()) if vp.is_file() else None), int(r.returncode)


def gsm8k_children_to_spawn(row: Mapping[str, Any], verdict: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Exactly the children whose slot ESCALATED (the verdict's `escalate` slots), in the row's order."""
    esc = {(s["arm"], int(s["s"]), s["cell"]) for s in verdict.get("slots") or [] if s.get("escalate")}
    return [k for k in row["children"] if (k["arm"], int(k["s"]), k["cell"]) in esc]


#: EVERY subprocess the app launches from a tree, and every plan composer, by the interpreter it
#: runs under. `driver` = `lp.driver_venv_python(W)` (uv sync --frozen --extra cu128: no inspect_ai / inspect_evals); `client` = the tinycap CLIENT
#: venv (`ensure_client_venv`: inspect_ai + inspect_evals + the driver's deps, CPU torch); `serving` = the image's vLLM venv. The test computes each
#: launched script's import closure: one that reaches `inspect_served` / `inspect_rung` (or imports inspect_* itself) must be `client` unless the
#: reach is provably unexecuted on that path (the two `driver` exceptions the test asserts by their guards), and every entry's module-level
#: third-party imports must be in the chosen venv's lock. (name, launched script | "-m module", interpreter)
APP_SUBPROCESS_TABLE: Tuple[Tuple[str, str, str], ...] = (
    ("gsm8k_assembly_argv", lp.GSM8K_ASSEMBLY_HELPER_REL, "client"),
    ("gsm8k_budget_prestep", "scripts/tinycap_oos_prompt_budget.py", "client"),
    ("lp._gsm8k_children", "scripts/served_capability_run.py", "client"),
    ("determinism_check_argv", DETERMINISM_REL, "driver"),
    ("producer_evidence:resolved_sampler", lp.PRODUCER_WRITERS["resolved_sampler"], "driver"),
    ("producer_evidence:request_evidence_audit", lp.PRODUCER_WRITERS["request_evidence_audit"], "driver"),
    ("registry_row_preflight:probe", "scripts/sdfbase/seedrep/registry_probe.py", "driver"),      # The registry emission probe (imports sisteer.registry.hooks)
    ("gsm8k_ledger_argv", "-m sisteer.capeval.serving.usage_ledger", "driver"),
    ("controls_prompts_via_helper", "scripts/modal/seedrep_ladder_helpers.py", "driver"),
    ("controls_sampling_via_helper", "scripts/modal/seedrep_ladder_helpers.py", "driver"),
    ("misalign_messages_via_helper", "scripts/modal/seedrep_ladder_helpers.py", "driver"),
    ("ladder_adapter_for_row:si-text", "scripts/modal/seedrep_ladder_helpers.py", "driver"),
    ("lp._static_children", "scripts/jailbreak_run.py", "driver"),
    ("lp._multiturn_children", "scripts/jailbreak_run.py", "driver"),
    ("lp._bon_children", "scripts/sdfbase/full22/bon_served_run.py", "driver"),
    ("lp._controls_children", "scripts/served_capability_run.py", "driver"),
    ("lp._misalign_children", "-m sisteer.capeval.misalign.runner", "driver"),
    ("lp.rung_consumer_argv", "scripts/sdfbase/seedrep/rung_redraw.py", "driver"),
    ("lp.standup_argv", "scripts/sdfbase/mt_serving_up.py", "serving"),
)


def _run_standup_row(row: dict, ctx: dict) -> dict:
    """The body of run_view_leg / run_arditi_leg / run_rung / run_petri_proxy: seed caches -> verify the 16 shas -> build the
    tree -> the driver venv -> the standup from the ROW's argv -> readiness assertions (both namespaces) -> a VERIFIED
    metrics scrape -> the ladder tailer (a producer row) -> the children at concurrency = the attested S (a rung: the
    consumer; Petri: the proxy wait for the drive's bound completion evidence) -> determinism spot-check hook -> teardown ->
    drain + seal + the explicit ladder inventory -> the acceptance state machine behind a THROWING publication barrier."""
    t0 = time.time()
    W, W7, wave_uuid = ctx["W"], ctx["W"][:7], ctx["wave_uuid"]
    task_id = os.environ.get("MODAL_TASK_ID", "")
    call_id = modal.current_function_call_id() or ""
    attempt_id = mint_attempt_id(task_id)
    hostname = socket.gethostname()
    VOL.reload()
    res_root = ctx.get("results_root") or results_root(W7)
    root_vol = attempt_root_for(res_root, row, attempt_id)
    root = Path(CONTAINER_ATTEMPT_ROOT)                # container-local; synced to root_vol
    shutil.rmtree(root, ignore_errors=True)
    (root / SERVE_DIR).mkdir(parents=True)
    prior = prior_attempts_of(Path(root_vol).parent)
    write_json(root / ATTEMPT_JSON, attempt_record(attempt_id=attempt_id, call_id=call_id, task_id=task_id, W=W, wave_uuid=wave_uuid,
                                                   row=row, prior_attempts=prior, production_wave_uuid=ctx.get("production_wave_uuid"),
                                                   launch_plan_sha=ctx.get("launch_plan_sha")))
    heartbeats = modal.Dict.from_name(ctx["dicts"]["heartbeats"], create_if_missing=True)
    stop_dict = modal.Dict.from_name(ctx["dicts"]["stop"], create_if_missing=True)
    accepted = modal.Dict.from_name(ctx["dicts"]["accepted"], create_if_missing=True)
    key, akey = row["control_key"], row["accepted_key"]
    syncers = [TreeSyncer(root, root_vol)]
    lock, stop = threading.Lock(), threading.Event()
    barrier = make_publication_barrier(syncers, lock)        # THROWS; the periodic sync stays best-effort
    is_petri = row["leg"] == "petri"
    is_rung = lp.is_rung_row(row) and not is_petri            # fc9d: the plan module's ONE rung definition (kind, stage, rung_mode, consumer binding)
    producer_row = bool(row.get("ladder")) and (row.get("ladder") or {}).get("cap_next") is not None
    _sync_all(syncers, lock, "attempt")
    print(f"[row] {row['standup_id']} attempt={attempt_id} call={call_id} task={task_id} host={hostname} root={root_vol}", flush=True)

    def fail(status: str, **more) -> dict:
        stop.set()
        # fr5 (4): a refused row's runs end `failed` in the registry (job_log -> this attempt's RESULT on the volume), never
        # `queued` / `running` forever -- except a duplicate execution's (NO_TERMINAL_STATUSES): those runs belong to the container
        # that holds the intent / claim. Before the tree is built the app has no `sisteer`; that is said, not swallowed.
        if status not in NO_TERMINAL_STATUSES:
            # who holds the accepted key? finish_attempt() may have ELECTED this attempt and then failed to publish its closing
            # marker -- that is closure trouble to recover from, never an evaluation failure; the attempt records evidence, never a state
            try:
                holder = accepted.get(akey)
            except Exception as e:  # noqa: BLE001
                holder = None
                print(f"[registry] {row['standup_id']}: accepted.get({akey!r}) unreadable while refusing {status} ({type(e).__name__}: {e})", flush=True)
            try:
                more["registry_attempt_terminal"] = registry_attempt_terminal(row, plans_root=ctx["plans_root"], attempt_root=str(root), status=status,
                                                                              attempt_id=attempt_id, evidence_uri=f"{root_vol}/{RESULT_FILE}",
                                                                              code_sha=str(ctx["W"]), standup_id=row["standup_id"], accepted_holder=holder,
                                                                              note=f"row refused: {status}")
            except Exception as e:  # noqa: BLE001 -- a lost record never costs the failure path
                more["registry_attempt_terminal"] = {"status": status, "runs": 0, "problems": [f"{type(e).__name__}: {e}"]}
                print(f"[registry] {row['standup_id']}: attempt-terminal evidence NOT filed for {status} ({type(e).__name__}: {e})", flush=True)
        final_status = terminal_publish(root, status=status, attempt_id=attempt_id, standup_id=row["standup_id"], barrier=barrier)
        return {"status": final_status, "attempt_id": attempt_id, "attempt_root": root_vol, "standup_id": row["standup_id"],
                **({"status_before_publication": status} if final_status != status else {}), **more}

    # a re-attempt executes its intent at most ONCE -- the second spawn of one intent (a pass that died after `.spawn()`) exits here
    if ctx.get("reattempt_intent"):
        reatt = modal.Dict.from_name(ctx["dicts"]["reattempts"], create_if_missing=True)
        won, holder = reattempt_execution_claim(reatt, str(ctx["reattempt_intent"]), attempt_id)
        if not won:
            return fail(REATTEMPT_DUPLICATE, intent=ctx["reattempt_intent"], executing_attempt=holder,
                        why="another execution already holds this re-attempt intent; a duplicate spawn never runs a second standup")
    # a claimed fan-out row executes at most ONCE -- the container claims `fanout|<row>|<nonce>|executing` with its call id;
    # a duplicate spawn of the same claim (a pass that died between .spawn() and its record, then re-spawned) exits here before any standup
    if ctx.get("fanout_claim"):
        fc = ctx["fanout_claim"]
        led = modal.Dict.from_name(ctx["dicts"]["reattempts"], create_if_missing=True)
        won, holder = fanout_execution_claim(led, str(fc.get("standup_id")), str(fc.get("nonce")), attempt_id=attempt_id, call_id=str(call_id))
        if not won:
            return fail(lp.FANOUT_DUPLICATE, claim=dict(fc), executing=holder,
                        why="another container already executes this row's fan-out claim; a duplicate spawn never stands anything up")
    # a rung row in a mode this app cannot execute fails BY NAME before any GPU minute (the consumer for static / controls / misalign; the
    # queue-derived Inspect assembly for GSM8K)
    is_gsm8k_rung = is_rung and row.get("rung_mode") == lp.RUNG_MODE_GSM8K_SUBSET
    if is_rung and row.get("rung_mode") not in EXECUTABLE_RUNG_MODES:
        return fail("RUNG_MODE_UNSUPPORTED", rung_mode=row.get("rung_mode"), why=f"executable rung modes: {sorted(EXECUTABLE_RUNG_MODES)}")

    evidence = {"schema": "seedrep_served/modal_evidence/1", "hardware": "modal", "attempt_id": attempt_id,
                "modal": {"app_name": ctx.get("app_name"), "function_call_id": call_id, "task_id": task_id, "hostname": hostname,
                          "image_id": os.environ.get("MODAL_IMAGE_ID"), "gpu": _gpu_facts(), **row["modal"]},
                "W": W, "wave_uuid": wave_uuid, "execution_wave_uuid": wave_uuid, "production_wave_uuid": ctx.get("production_wave_uuid"),
                "launch_plan_sha": ctx.get("launch_plan_sha"), "reattempt_intent": ctx.get("reattempt_intent"),
                "standup_id": row["standup_id"], "control_key": key, "accepted_key": akey,
                "smoke": row.get("smoke"), "started_at": utc(),
                # the seed decides on the launcher's $SEEDREP_SEED_COMPILE_CACHES as spawn_ctx carried it ("" when the launcher predates
                # the key -> off); the record per cache names the decision and the counts
                "compile_cache_seed": seed_compile_caches(env={SEED_COMPILE_CACHES_ENV: str(ctx.get(SEED_COMPILE_CACHES_CTX, ""))}),
                # the EFFECTIVE compile-cache persist setting, as it reached this worker (spawn_ctx from the launcher's environment; "" = unset =
                # OFF, also when the launcher predates the key -- fx1) -- what `compile_cache_persist` below was decided on, with the row's status
                "compile_cache_persist_setting": {PERSIST_COMPILE_CACHES_ENV: str(ctx.get(PERSIST_COMPILE_CACHES_CTX, "")),
                                                  "source": "spawn_ctx" if PERSIST_COMPILE_CACHES_CTX in ctx else "default (spawn_ctx carried no setting: off)"}}
    write_json(root / EVIDENCE_NAME, evidence)

    # the image's serving venv + the 16 shas BEFORE any tree is built
    serve_venv, serve_sentinel, serve_value = serving_venv_names(ctx["dep_hashes"][VLLM_REQS_REL])
    if not (Path(serve_venv, "bin", "vllm").exists() and Path(serve_sentinel).is_file()
            and Path(serve_sentinel).read_text().strip() == serve_value):
        return fail("IMAGE_UNPREPARED", error=f"serving venv {serve_venv} not as the image build promised")
    pins = json.loads((Path(REPOS) / f"sisteer-{W}" / lp.PINS_REL).read_text())
    apins = json.loads((Path(REPOS) / f"sisteer-{W}" / lp.ARDITI_PINS_REL).read_text())
    # the fixed 16 PLUS this row's own extra view + objects (an S5 probe row serves a peer view the tree's pin file never carries; the
    # plan's objects table pins them)
    row_extra = lp.bank_views_for_plan({"standups": [row]})
    row_pins = lp.bank_pins_for_plan({"standups": [row]})
    verify = verify_staged_objects(BANK_VOL, ARDITI_BANK_VOL, pins, apins, extra_views=row_extra, extra_pins=row_pins)
    evidence["object_verification"] = {"ok": verify["ok"], "n": len(verify["shas"]), "problems": verify["problems"], "extra_views": row_extra}
    if not verify["ok"]:
        return fail("OBJECT_SHA_MISMATCH", problems=verify["problems"])
    try:
        built = _build_shared(W)
    except Exception as e:  # noqa: BLE001
        return fail("TREE_UNBUILDABLE", error=f"{type(e).__name__}: {e}")
    tree = Path(built["repo_dir"])
    if dependency_hashes(tree) != ctx["dep_hashes"]:
        return fail("DEPENDENCY_MISMATCH")
    # the staged tree's src on THIS interpreter's sys.path for EVERY row, before anything imports `sisteer` in the app
    # process -- the registry preflight, the served-runs file and the completion records all import `sisteer.registry.hooks` HERE (not in
    # a driver child, whose PYTHONPATH never reaches this interpreter). Conditional on `producer_row or is_rung` it left the 50 BoN /
    # multi-turn / Petri rows refusing REGISTRY_EMISSION_OFF on a cold container with ModuleNotFoundError while their driver probe said ON.
    # It sits here for the ladder's lazy `sisteer` imports; the ladder LIBRARY itself stays conditional below.
    evidence["app_sys_path_src"] = ensure_tree_src_importable(tree)
    # the ladder protocol module BEFORE the standup: a producer / rung row without it fails by name here, not after the engine is up
    L = None
    if producer_row or is_rung:
        try:
            L = load_ladder_lib(tree)
        except LadderUnavailable as e:
            return fail("LADDER_LIBRARY_MISSING", error=str(e))
    try:
        dv = ensure_driver_venv(tree, W)
    except Exception as e:  # noqa: BLE001
        return fail("DRIVER_VENV_FAILED", error=str(e)[-800:])
    evidence["driver_venv"] = dv
    driver_python = lp.driver_venv_python(W)
    # a row whose children or presteps import inspect_ai (an `inspect/` child, a G-FIT budget_argv) gets the tinycap CLIENT venv
    # (common.sh f22_client_venv, the cluster's $CPY) -- the driver venv's `uv sync --extra cu128` carries no inspect_ai / inspect_evals
    client_python = ""
    if lp.row_needs_client_venv(row):
        try:
            evidence["client_venv"] = ensure_client_venv(tree)
        except Exception as e:  # noqa: BLE001
            return fail("CLIENT_VENV_FAILED", error=str(e)[-800:])
        client_python = client_venv_python(tree)
    bank_view = lp._flag_value(row["argv"], "--defenses-dir")
    env = container_env(bank_view=bank_view, tree=str(tree), thinking=bool(row["geometry"]["thinking"]), petri=is_petri)
    # the row records itself natively or it does not run -- every child's identity file at its in-container path, the REAL hooks
    # under the composed child env (driver venv) ON with the ATTEMPT spool; the block is modal_evidence.json["registry"]
    reg_block, reg_problems = registry_row_preflight(row, plans_root=ctx["plans_root"], attempt_root=str(root), base_env=env,
                                                     probe=lambda e: registry_probe(driver_python, e, tree=tree))
    evidence[REGISTRY_EVIDENCE_KEY] = reg_block
    write_json(root / EVIDENCE_NAME, evidence)
    if reg_problems:
        return fail(REGISTRY_EMISSION_OFF, problems=reg_problems)

    def sync_loop():
        while not stop.wait(SYNC_EVERY_S):
            _sync_all(syncers, lock, "periodic")
    threading.Thread(target=sync_loop, daemon=True, name="tree-sync").start()

    hb_state: Dict[str, Any] = {"running": None, "phase": "setup", "extra": {}}

    def heartbeat_loop():
        while not stop.wait(HEARTBEAT_S):
            try:
                progress = count_progress(root)                      # attempt-local output counters
                heartbeats[key] = json.dumps(heartbeat_payload(attempt_id=attempt_id, phase=hb_state.get("phase"), running=hb_state["running"],
                                                               progress=progress, extra=hb_state.get("extra")))
            except Exception as e:  # noqa: BLE001
                print(f"[heartbeat] WARNING {type(e).__name__}: {e}", flush=True)
    threading.Thread(target=heartbeat_loop, daemon=True, name="heartbeat").start()

    # --- the standup, from the ROW's argv (the ONE placeholder resolved) ---------------------------------------------
    serve_dir = root / SERVE_DIR
    argv = [a.replace(lp.ATTEMPT_ROOT_PLACEHOLDER, str(root)) for a in row["argv"]]
    argv[0] = f"{serve_venv}/bin/python"                         # the image's serving venv (the plan names the same path)
    evidence["standup_argv"] = argv
    hb_state["phase"] = "standup"
    standup_log = (serve_dir / "standup.log").open("ab")
    # the standup's serving facts join the ATTEMPT spool (never the shared /vol/registry/records the in-container probes would resolve),
    # and the runs it serves are named beside endpoint.json (hooks.RUNS_FILE) so write_endpoint's hook records them
    try:
        evidence[REGISTRY_EVIDENCE_KEY]["serving_runs"] = write_registry_runs(row, plans_root=ctx["plans_root"], serve_dir=serve_dir)
    except Exception as e:  # noqa: BLE001 -- refused by name, never a standup whose serving facts go unrecorded
        return fail(REGISTRY_EMISSION_OFF, problems=[f"{REGISTRY_RUNS_FILE} not written: {type(e).__name__}: {e}"])
    write_json(root / EVIDENCE_NAME, evidence)
    standup_env = dict(env, **{lp.SPOOL_ENV: f"{root}/{lp.SPOOL_REL}", lp.SUBSTRATE_ENV: lp.SUBSTRATE})   # fr5 (3): every in-container writer says modal
    standup = subprocess.Popen(argv, env=venv_env(standup_env, argv[0]), stdout=standup_log, stderr=subprocess.STDOUT, cwd=str(tree), start_new_session=True)
    ep = serve_dir / "endpoint.json"
    ready_deadline = time.time() + lp.READY_TIMEOUT_S + 300
    while time.time() < ready_deadline:
        if ep.exists() or standup.poll() is not None:
            break
        time.sleep(5)
    if not ep.exists():
        tail = (serve_dir / "standup.log").read_text(errors="replace")[-4000:] if (serve_dir / "standup.log").exists() else ""
        return fail("STANDUP_FAILED", standup_tail=tail, standup_rc=standup.poll())
    ep_bytes = ep.read_bytes()                                      # the READY record's bytes, kept for `restore_endpoint_record` (F-AA)
    endpoint = json.loads(ep_bytes)
    att_p = serve_dir / "serving_attestation.json"                   # the workers' own residual-slot reports live here, beside endpoint.json
    attestation = json.loads(att_p.read_text()) if att_p.is_file() else None
    stop_file = serve_dir / "STOP"
    samplers: Optional[Samplers] = None                              # bound by `_start_samplers` below; teardown stops it first (F-AG)

    def teardown_standup(timeout: int = 300) -> None:
        # F-AG: the metrics scrape is stopped and JOINED before the engine goes away -- no error row is ever written after this
        if samplers is not None:
            evidence.setdefault("metrics", {}).update(sampler_stopped=samplers.stop(timeout=3 * DMON_CADENCE_S), n_error_rows=samplers.n_error_rows,
                                                      n_rows=samplers.n_rows)
        stop_file.touch()
        try:
            standup.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(standup.pid, signal.SIGKILL)
        restored = restore_endpoint_record(ep, ep_bytes)             # F-AA: the READY endpoint.json outlives the standup's own unlink
        if restored is not None:
            evidence["endpoint_json_restored_after_teardown"] = restored

    if not (serve_dir / "cells_served.json").is_file():
        # L2: the served-cells record is a mandated output of the standup; a missing file used to read as `{}` and fail open into
        # cells_served.canonical [] != ... -- refused by name instead
        teardown_standup()
        return fail(READINESS_CONTRACT, problems=[f"{serve_dir / 'cells_served.json'} ABSENT (the standup wrote endpoint.json but no cells_served.json: the served-cells record is mandated)"])
    cells_served = json.loads((serve_dir / "cells_served.json").read_text())
    # F-AA: the standup's own job record, the cluster sbatch's shape, ONE line -- `audit_common.read_standup` requires it
    evidence["standup_jobs"] = str(write_standup_jobs(serve_dir, standup_job_line(endpoint, cells_served, row, attempt_id=attempt_id, call_id=call_id,
                                                                                    W=ctx["W"], node=hostname)).relative_to(root))
    problems = readiness_problems(endpoint, cells_served, row, attestation=attestation)
    evidence["readiness"] = {"problems": problems, "attested_max_model_len": endpoint.get("attested_max_model_len"),
                             "attested_max_num_seqs": endpoint.get("attested_max_num_seqs"),
                             "window_source": (endpoint.get("geometry") or {}).get("window_source"),
                             "adjustments": (endpoint.get("geometry") or {}).get("adjustments")}
    write_json(root / EVIDENCE_NAME, evidence)
    if problems:
        teardown_standup()
        return fail(readiness_failure_status(problems),             # L1: GEOMETRY_MISMATCH only for the served geometry; the contract otherwise
                    problems=problems)
    S = int(endpoint["attested_max_num_seqs"])
    proxy_url = endpoint["proxy_url"].rstrip("/")
    table = endpoint["routing_table"]
    # the metrics scrape: the engine's page THROUGH THE PROXY's /metrics pass-through (proxy_port; F-O -- the engine's own
    # /metrics is bearer-guarded and this app never holds that token), verified once before it feeds §9.4
    try:
        metrics_url = metrics_url_from_endpoint(endpoint)
    except ValueError as e:
        teardown_standup()
        return fail("METRICS_UNSCRAPABLE", error=str(e))
    metrics_headers = metrics_headers_from_endpoint(endpoint)
    first_scrape, m_problems = verify_metrics_scrape(metrics_url, lambda u: _http_get_text(u, headers=metrics_headers))
    evidence["metrics"] = {"url": metrics_url, "first_scrape": first_scrape, "problems": m_problems}
    if m_problems:
        teardown_standup()
        return fail("METRICS_UNSCRAPABLE", problems=m_problems)
    samplers = _start_samplers(serve_dir, root, stop, {"metrics_url": metrics_url, "metrics_headers": metrics_headers})
    # cells.json (the misalign runner's contract) from the standup's own enumeration -- HARNESS ids, as served
    served = set(endpoint.get("served_models") or [])
    rows_ = []
    vmodel_by_harness: Dict[str, str] = {}
    si_by_harness: Dict[str, Optional[str]] = {}
    for c in list(cells_served.get("canonical") or []) + list(cells_served.get("controls") or []):
        vmodel_by_harness[c["cell_id"]] = c.get("virtual_model_name")
        si_by_harness[c["cell_id"]] = c.get("si_id") if c.get("cell_class") == "prompted" else None
        if c["virtual_model_name"] in served:
            rows_.append({"cell_id": c["cell_id"], "virtual_model": c["virtual_model_name"], "serving_identity": c.get("serving_identity"),
                          "intervention_hash": c.get("intervention_hash"), "si_id": c.get("si_id") if c.get("cell_class") == "prompted" else None,
                          "kind": c.get("kind")})
    write_json(serve_dir / "cells.json", {"cells": rows_, "serving": {"substrate": "modal", "attempt_id": attempt_id, "standup_id": row["standup_id"],
                                                                        "attested_max_num_seqs": S, "attested_max_model_len": endpoint.get("attested_max_model_len"),
                                                                        "engine": "lora_native", "app": ctx.get("app_name")}})
    _sync_all(syncers, lock, "ready")

    # --- the ladder's producer side (a base row that feeds a queue): the ladder Producer + this module's tailer --------
    lvol = producer = tailer = None
    if L is not None:
        lvol = L.LocalVolume(Path(VOL_PATH), commit=barrier, reload=VOL.reload)     # its commits ARE the throwing barrier
    if producer_row and not is_rung:
        try:
            producer = ladder_producer_for_row(L, row, ctx, vol=lvol, attempt_id=attempt_id, attempt_root_vol=root_vol)
            adapter = ladder_adapter_for_row(L, row, ctx, producer=producer, root=root, attempt_root_vol=root_vol, attempt_id=attempt_id,
                                             vmodel_by_harness=vmodel_by_harness, si_by_harness=si_by_harness,
                                             driver_python=driver_python, tree=tree, env=env)
            tailer = StreamTailer(root, adapter, producer).start()
        except (LadderUnavailable, Exception) as e:  # noqa: BLE001
            teardown_standup()
            return fail("LADDER_TAILER_UNAVAILABLE", error=f"{type(e).__name__}: {e}")

    # --- the children (a rung: the consumer; Petri: no driver here) ------------------------------------------------
    hb_state["phase"] = "drivers"
    children_out: List[dict] = []
    ncells = len(row["route_set"])
    inflight_log = (root / "driver_inflight.jsonl").open("a")
    if is_petri:
        kids = []
    elif is_gsm8k_rung:
        # the queue-derived Inspect assembly -- wait for the queue to CLOSE, classify the ACCEPTED base stores, write the subsets,
        # then spawn ONE Inspect child per slot that ESCALATED (> 1 % at cap); a refusal fails the row by name, never a partial rung
        hb_state["phase"] = "gsm8k-assembly"
        verdict, arc = run_gsm8k_assembly(row, ctx, tree=tree, client_python=client_python, root=root, env=env,      # the CLIENT venv
                                          deadline_s=row["modal"]["timeout_s"] - (time.time() - t0) - GSM8K_ASSEMBLY_MARGIN_S)
        evidence["gsm8k_assembly"] = {"rc": arc, "verdict": verdict}
        write_json(root / EVIDENCE_NAME, evidence)
        if verdict is None or not verdict.get("closed"):
            teardown_standup()
            return fail("GSM8K_QUEUE_NOT_CLOSED", rc=arc, why=(verdict or {}).get("why") or "the assembly helper wrote no verdict")
        if not verdict.get("ok"):
            teardown_standup()
            return fail("GSM8K_ASSEMBLY_REFUSED", rc=arc, problems=[s.get("refused") for s in verdict.get("slots") or [] if s.get("refused")])
        kids = gsm8k_children_to_spawn(row, verdict)
        print(f"[row] gsm8k assembly: {verdict['n_escalate']} slot(s) escalate, {verdict['n_not_triggered']} below the trigger -> {len(kids)} child(ren)", flush=True)
    elif is_rung:
        if not lp.is_ladder_consumer_row(row):                   # fc9d: only a LADDER CONSUMER reaches the consumer child; a contradictory row is refused, never run
            return fail("RUNG_ROW_SHAPE_REFUSED", problems=[f"{row['standup_id']}: kind {row.get('kind')!r} rung_mode {row.get('rung_mode')!r} stage {row.get('stage')!r} "
                                                             f"consumer {'present' if row.get('rung_consumer') else 'absent'} is not a ladder-consumer rung"])
        # the consumer child carries the plan's RAW consumer argv; `lp.resolve_spawn` resolves its placeholders ({attempt_id}, {call_id},
        # {proxy_url}, {attempt_root_rel}, {driver_python}, {attempt_root} = the container-LOCAL root for --attempt-root-local)
        kids = [rung_consumer_child(row, row["rung_consumer"], attempt_root=str(root), attempt_id=attempt_id, tree=str(tree))]
        env_problems = lp.rung_child_env_problems(kids[0]["env"], attempt_root=str(root), attempt_id=attempt_id, seed_bases=list(row.get("seed_bases") or []))
        if env_problems:
            return fail("RUNG_CHILD_ENV_MISSING", problems=env_problems)    # before the server: a consumer with no / a mis-pinned evidence env never draws
        root.mkdir(parents=True, exist_ok=True)                                 # §R2.2: the evidence lands AT the attempt root
    else:
        kids = row["children"]
    if row["leg"] == "gsm8k" and kids:
        # G-FIT (g8k_budget_artifact): the prompt-budget artifact + fit sidecar per (cap, cell) BEFORE any Inspect child
        hb_state["phase"] = "gsm8k-budget"
        b_problems = gsm8k_budget_prestep(kids, root=root, driver_python=driver_python, tree=tree, env=env, W=W, geometry_eval=row["geometry"]["eval"],
                                          client_python=client_python)
        evidence["gsm8k_budget"] = {"problems": b_problems, "n_artifacts": len({k.get("budget_rel") for k in kids})}
        if b_problems:
            teardown_standup()
            return fail("GSM8K_BUDGET_UNFIT", problems=b_problems)
    # ONE spawn resolution for every child -- the generated child + this context -> the FINAL argv / env Popen receives
    child_subs: Dict[str, Dict[str, str]] = {}
    gsm8k_states: Dict[str, Dict[str, Any]] = {}
    spawn_context = lp.SpawnContext(attempt_root=str(root), attempt_id=attempt_id, standup_id=row["standup_id"], plans_root=ctx["plans_root"], proxy_url=proxy_url,
                                    routing_table=table, driver_python=driver_python, S=S, ncells=ncells, bank_view=bank_view, base_env=env,
                                    pool_urls=ctx.get("pool_urls"), call_id=call_id or attempt_id, attempt_root_rel=volume_rel(root_vol), child_subs=child_subs,
                                    client_python=client_python)
    procs: List[Tuple[dict, subprocess.Popen, Any]] = []
    child_registry: Dict[str, Tuple[Dict[str, str], str]] = {}        # Child name -> (its registry env, its store root) for the completion record
    drained_by_stop = False                                            # fr5 (4): a stop-Dict drain ends the runs `cancelled`, a failure `failed`
    max_live = S if row["leg"] in lp.STATIC_LEGS else len(kids)
    pending = list(kids)
    (root / "children").mkdir(exist_ok=True)
    while pending or procs:
        while pending and len(procs) < max_live:
            k = pending.pop(0)
            if row["leg"] == "gsm8k":
                # the per-store seam BEFORE the driver: the attempt id into ladder_state.json, the child's routing-table copy,
                # its usage ledger up with that attempt id; the per-child substitutions the argv's {ledger_url} / {child_routing_table} / {gsm8k_attempt_id} need
                try:
                    st = gsm8k_child_up(k, root=root, serve_dir=serve_dir, proxy_url=proxy_url, driver_python=driver_python, tree=tree, env=env)
                except Exception as e:  # noqa: BLE001
                    teardown_siblings(procs, pending)
                    for st_ in gsm8k_states.values():
                        gsm8k_child_down(st_, rc=1, result="LEDGER_FAILED")
                    teardown_standup()
                    return fail("GSM8K_LEDGER_FAILED", error=f"{type(e).__name__}: {e}", child=child_log_name(k))
                gsm8k_states[lp.child_key(k)] = st
                child_subs[lp.child_key(k)] = st["subs"]
            try:
                k_argv, k_env = lp.resolve_spawn(k, spawn_context)
            except ValueError as e:
                teardown_siblings(procs, pending)
                for st_ in gsm8k_states.values():
                    gsm8k_child_down(st_, rc=1, result="CHILD_UNRESOLVABLE")
                teardown_standup()
                return fail("CHILD_UNRESOLVABLE", error=str(e), child=child_log_name(k))
            try:
                k_env = child_env_with_pool_bearer(k_argv, k_env, os.environ)     # The lane pool bearer, --attacker-endpoint children only
            except LookupError as e:
                teardown_siblings(procs, pending)
                for st_ in gsm8k_states.values():
                    gsm8k_child_down(st_, rc=1, result="POOL_BEARER_MISSING")
                teardown_standup()
                return fail("POOL_BEARER_MISSING", error=str(e), child=child_log_name(k))
            name = child_log_name(k)                            # the ONE encoder the tally lookup shares
            try:
                k_store_root = lp.child_store_root(k, k_argv)
            except ValueError:
                k_store_root = ""
            child_registry[name] = ({e: k_env.get(e, "") for e in (lp.RUN_IDENTITY_ENV, lp.SPOOL_ENV, lp.SUBSTRATE_ENV)}, k_store_root)
            log = (root / "children" / f"{name}.log").open("ab")
            (root / "children" / f"{name}.argv.json").write_text(json.dumps({"argv": k_argv, "env_registry": {e: k_env.get(e) for e in (lp.RUN_IDENTITY_ENV, lp.SPOOL_ENV)}}, indent=1))
            p = subprocess.Popen(k_argv, env=venv_env(k_env, k_argv[0]), stdout=log, stderr=subprocess.STDOUT, cwd=str(tree), start_new_session=True)
            procs.append((k, p, log))
            inflight_log.write(json.dumps({"t": utc(), "event": "start", "child": name, "live": len(procs)}) + "\n"); inflight_log.flush()
        hb_state["running"] = len(procs)
        time.sleep(5)
        for k, p, log in list(procs):
            rc = p.poll()
            if rc is not None:
                log.close()
                procs.remove((k, p, log))
                children_out.append({"eval_leg": k["eval_leg"], "arm": k["arm"], "s": k["s"], "rc": rc})
                # the generation of a store-less child is recorded by the container, through the real hook under the child's env
                reg_env, reg_store = child_registry.get(child_log_name(k), ({}, ""))
                n_reg = len(child_registry_completion(k, reg_env, rc=rc, store_root=reg_store, attempt_id=attempt_id, log_name=child_log_name(k),
                                                      code_sha=str(ctx["W"])))
                evidence[REGISTRY_EVIDENCE_KEY].setdefault("completion_patches", {})[child_log_name(k)] = n_reg
                if lp.child_key(k) in gsm8k_states:                 # st_attempt_end + st_ledger_down for this store
                    gsm8k_child_down(gsm8k_states.pop(lp.child_key(k)), rc=rc, result=("OK" if rc == 0 else "DRIVER_FAILED"))
                inflight_log.write(json.dumps({"t": utc(), "event": "exit", "child": child_log_name(k), "rc": rc, "live": len(procs)}) + "\n"); inflight_log.flush()
                if rc != 0 and lp.D73_TEARDOWN_AFTER_FAILURE and evidence.get("teardown_after_failure") is None:
                    # the FIRST failed child tears its siblings down -- no container waits past its deadline on a row that cannot be accepted
                    killed = teardown_siblings(procs, pending)
                    evidence["teardown_after_failure"] = {"first_failed": child_log_name(k), "rc": rc, "siblings_killed": killed, "pending_dropped": True, "at": utc()}
                    print(f"[row] child {child_log_name(k)} exited rc={rc}: tearing the standup down (D73: --teardown-after-failure semantics; {len(killed)} sibling(s) signalled)", flush=True)
                    inflight_log.write(json.dumps({"t": utc(), "event": "teardown_after_failure", "first_failed": child_log_name(k), "killed": killed}) + "\n"); inflight_log.flush()
        try:
            if stop_dict.get(key) is not None:
                print("[row] STOP sentinel present in the wave's stop Dict — draining", flush=True)
                drained_by_stop = True
                for _, p, _ in procs:
                    os.killpg(p.pid, signal.SIGTERM)
                pending.clear()
        except Exception:  # noqa: BLE001
            pass
        if time.time() - t0 > row["modal"]["timeout_s"] - 1200:
            print("[row] approaching the container timeout — draining", flush=True)
            for _, p, _ in procs:
                os.killpg(p.pid, signal.SIGTERM)
            pending.clear()
    inflight_log.close()
    for st_ in list(gsm8k_states.values()):                          # a ledger whose child was torn down / drained
        gsm8k_child_down(st_, rc=1, result="TORN_DOWN")
    petri_outcome = None
    if is_petri:
        petri_outcome = _petri_proxy_wait(row, ctx, root, serve_dir, endpoint, stop_dict, key, hb_state, attempt_id=attempt_id,
                                          attempt_root_vol=root_vol, call_id=call_id, task_id=task_id)
        children_out = [petri_outcome["child"]]

    # --- determinism spot-check hook (determinism_check.py runs in-container before teardown, once per table leg) ---
    # integration F2: the argv is the check's REAL contract (--root / --leg / --proxy-url / --metrics-url / --out), its rc is consulted
    # and the record is bound under modal_evidence.json["determinism"] (producer_evidence's idiom) -- never a fire-and-forget subprocess
    if row["kind"] == "base" and row["engine"] == "s1":
        hb_state["phase"] = "determinism"
        evidence["determinism"] = determinism_evidence(root, tree=tree, W=ctx["W"], row=row, driver_python=driver_python, env=env,
                                                       proxy_url=proxy_url, metrics_url=metrics_url)

    # --- teardown + records -------------------------------------------------------------------------------------------
    hb_state["phase"] = "teardown"
    teardown_standup(timeout=600)
    standup_log.close()
    # the producer's evidence (the resolved-sampler record, the audit verdicts, the served cells) written and BOUND here,
    # over the complete server log, before it is packed -- the certificate reads these files and nothing an operator points at
    hb_state["phase"] = "producer-evidence"
    evidence[lp.PRODUCER_EVIDENCE_KEY] = producer_evidence(root, tree=tree, W=ctx["W"], row=row, serve_dir=serve_dir, driver_python=driver_python, env=env)
    vlog = serve_dir / "vllm_server.log"
    if vlog.exists():                                            # the WHOLE log, gzipped, never truncated
        with vlog.open("rb") as fi, gzip.open(serve_dir / "vllm_server.log.gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
        vlog.unlink()
    stop.set()
    evidence.setdefault(REGISTRY_EVIDENCE_KEY, {})["records_written"] = count_spool_records(root / lp.SPOOL_REL)   # 0 is said, by name
    failed = [c for c in children_out if c.get("rc") not in (0, None)]
    status = "DONE" if not failed else ("PETRI_" + petri_outcome["state"].upper() if is_petri else "DRIVERS_FAILED")
    evidence.update(children=children_out, elapsed_s=round(time.time() - t0, 1), finished_at=utc(),
                    # fx1: decided on the ctx value ("" when the launcher predates the key / left it unset -> OFF) AND this row's status
                    compile_cache_persist=compile_cache_persist_evidence(env={PERSIST_COMPILE_CACHES_ENV: str(ctx.get(PERSIST_COMPILE_CACHES_CTX, ""))}, status=status),
                    attested_max_num_seqs=S, petri=petri_outcome)
    write_json(root / EVIDENCE_NAME, evidence)
    (root / RESULT_FILE).write_text(f"{utc()} status={status} rc={0 if not failed else 1} attempt={attempt_id} standup={row['standup_id']} "
                                     f"children={len(children_out)} failed={len(failed)} S={S} elapsed_s={evidence['elapsed_s']}\n")
    if failed:
        # a row that is not DONE records RECOVERY EVIDENCE for every child run (a run_patch: this attempt, its status,
        # the RESULT on the volume, whether the wave's STOP drained it) -- never a logical run_state, which is the grading host's to mint
        # when the standup is HELD; filed BEFORE the final sync so the spool ships it. Nothing is elected at RESULT time (accepted_holder None).
        try:
            evidence[REGISTRY_EVIDENCE_KEY]["attempt_terminal"] = registry_attempt_terminal(
                row, plans_root=ctx["plans_root"], attempt_root=str(root), status=status, attempt_id=attempt_id, evidence_uri=f"{root_vol}/{RESULT_FILE}",
                code_sha=str(ctx["W"]), standup_id=row["standup_id"], accepted_holder=None, cancelled=drained_by_stop,
                note=f"row {status}: {len(failed)} of {len(children_out)} children failed" + ("; drained by the wave's STOP" if drained_by_stop else ""))
        except Exception as e:  # noqa: BLE001
            evidence[REGISTRY_EVIDENCE_KEY]["attempt_terminal"] = {"runs": 0, "problems": [f"{type(e).__name__}: {e}"]}
            print(f"[registry] {row['standup_id']}: attempt-terminal evidence NOT filed ({type(e).__name__}: {e})", flush=True)
        evidence[REGISTRY_EVIDENCE_KEY]["records_written"] = count_spool_records(root / lp.SPOOL_REL)
        write_json(root / EVIDENCE_NAME, evidence)
    _sync_all(syncers, lock, "final")
    if failed:
        return {"status": status, "attempt_id": attempt_id, "attempt_root": root_vol, "standup_id": row["standup_id"], "failed": failed,
                "petri": petri_outcome}

    # --- drain + seal + the EXPLICIT ladder inventory, then the state machine behind the barrier ----------------------------
    # F-AM: the whole post-RESULT tail runs under `run_post_result_tail`'s bounded deadline (POST_RESULT_EXIT_S; spawn_ctx may
    # override) -- past it the row logs POST_RESULT_HANG + a thread dump, records the hang in modal_evidence.json["post_result"]
    # and RETURNS with the RESULT already written untouched (no acceptance: the pull refuses by name), instead of idling an H200
    # to the container timeout.
    hb_state["phase"] = "acceptance"

    def acceptance_tail() -> Dict[str, Any]:
        nonlocal producer
        inventory: List[dict] = []
        if tailer is not None:
            hb_state["phase"] = "ladder-drain"
            summary = tailer.drain()                              # raises on any recorded refusal
            hb_state["phase"] = "ladder-seal"
            entries = producer.seal()
            write_ladder_enqueued(root, attempt_id=attempt_id, control_key=key, accepted_key=akey, ladder=row["ladder"], entries=entries,
                                  streams=summary["streams"], complete=True)
            inventory = entries
        elif is_gsm8k_rung:
            # the ceiling: nothing is queued above 16,384; the explicit ladder record carries the assembly verdict (which slots escalated / were below
            # the trigger, the source attempts) so the pull and the at-cap table read WHY a slot has no rung store
            v = (evidence.get("gsm8k_assembly") or {}).get("verdict") or {}
            write_ladder_enqueued(root, attempt_id=attempt_id, control_key=key, accepted_key=akey, ladder=row["ladder"], entries=[],
                                  streams=[{"path": GSM8K_ASSEMBLY_VERDICT, "rows_read": len(v.get("slots") or []), "at_cap": int(v.get("n_escalate") or 0), "enqueued": 0,
                                            "gsm8k_assembly": {"n_escalate": v.get("n_escalate"), "n_not_triggered": v.get("n_not_triggered"),
                                                               "slots": [{k_: s.get(k_) for k_ in ("arm", "s", "cell", "exit", "escalate", "n_at_cap", "n", "source_attempt_id")}
                                                                         for s in v.get("slots") or []]}}],
                                  complete=True)
            inventory = []
        elif is_rung:
            lad = row["ladder"]
            entries: List[dict] = []
            if lad and lad.get("cap_next") is not None:
                next_root_rel = L.ladder_root_rel(W7, wave_uuid, row["leg"], row["engine"], int(lad["cap_next"]))
                entries = rung_entries_from_volume(Path(VOL_PATH), next_root_rel, attempt_id)
                tally = consumer_tally_from_log(root / "children" / f"{child_log_name(kids[0])}.log")
                declared = None if tally is None else tally.get("enqueued_next")
                complete = declared is not None and int(declared) == len(entries)
                evidence_files = request_evidence_files(root)
                drawn = int((tally or {}).get("drawn") or 0)
                ev_problems = ([] if (not drawn or evidence_files) else
                               [f"the consumer drew {drawn} item(s) but wrote no request evidence at the attempt root (WS1-D′'s client; WS2-E finding 2, §R2.2)"])
                write_ladder_enqueued(root, attempt_id=attempt_id, control_key=key, accepted_key=akey, ladder=lad, entries=entries,
                                      streams=[{"path": "consumer tally", "rows_read": drawn, "at_cap": (tally or {}).get("at_ceiling", 0) + len(entries),
                                                "enqueued": len(entries), "declared_enqueued_next": declared}] + evidence_files,
                                      complete=(complete and not ev_problems),
                                      problems=([] if complete else [f"consumer tally enqueued_next={declared!r} != {len(entries)} queue files on the volume"]) + ev_problems)
                producer = rung_next_producer(L, row, ctx, vol=lvol, attempt_id=attempt_id, attempt_root_vol=root_vol, entries=entries)
                if producer is not None:                          # the adopted-evidence index reaches the NEXT queue's inventory
                    producer.adopted = list((tally or {}).get("adopted") or [])
                inventory = entries
        hb_state["phase"] = "acceptance"
        return finish_attempt(root=root, key=akey, attempt_id=attempt_id, inventory=inventory, accepted=accepted, commit=barrier,
                              closed_marker=None, summary={"children": len(children_out), "S": S, "elapsed_s": evidence["elapsed_s"]},
                              producer=producer, ladder_required=(producer is not None),
                              require_rows=bool(producer_row and not is_rung))       # F-H: a base producer whose children wrote nothing is refused

    try:
        tail = run_post_result_tail(acceptance_tail, deadline_s=float(ctx.get(POST_RESULT_EXIT_CTX, POST_RESULT_EXIT_S)),
                                    phase_of=lambda: hb_state.get("phase"))
    except LadderInventoryIncomplete as e:
        return fail("LADDER_INVENTORY_INCOMPLETE", error=str(e))
    except LadderUnavailable as e:
        return fail("LADDER_TAILER_ERROR", error=str(e))
    except PublicationFailed as e:
        return fail("PUBLICATION_FAILED", error=str(e))
    holder = None
    if tail["hang"]:
        # fo3: did the hung tail get PAST the acceptance election? The wave's accepted Dict is the authority; unreadable ->
        # not known accepted -> the explicit timeout status (fail closed: a missing attempt must enter recovery, never wait 36 h as DONE)
        try:
            holder = accepted.get(akey)
        except Exception as e:  # noqa: BLE001
            print(f"[row] WARNING accepted.get({akey!r}) after the post-RESULT hang: {type(e).__name__}: {e}", flush=True)
    verdict = post_result_outcome(tail, attempt_id=attempt_id, accepted_holder=holder)
    evidence["post_result"] = verdict["post_result"]
    write_json(root / EVIDENCE_NAME, evidence)
    done = {"status": verdict["status"], "attempt_id": attempt_id, "attempt_root": root_vol, "standup_id": row["standup_id"], "acceptance": verdict["acceptance"],
            "elapsed_s": evidence["elapsed_s"], "S": S, "petri": petri_outcome, "post_result": evidence["post_result"],
            **({"error": verdict["error"]} if verdict.get("error") else {})}
    if tail["hang"]:
        print(f"[row] {verdict['status']} after the post-RESULT hang (phase={tail.get('phase')} accepted holder={holder!r}); RESULT left as written", flush=True)
        # the hang record to the volume, itself bounded: the hung tail may hold the publication lock (a commit that never returned)
        run_post_result_tail(lambda: _sync_all(syncers, lock, "post-result-hang"), deadline_s=120.0, phase_of=lambda: "post-result-hang publish",
                             dump=lambda: None)
    return done


def ladder_adapter_for_row(L, row: Mapping[str, Any], ctx: Mapping[str, Any], *, producer, root: Path, attempt_root_vol: str, attempt_id: str,
                           vmodel_by_harness: Mapping[str, str], si_by_harness: Mapping[str, Optional[str]], driver_python: str, tree: Path,
                           env: Mapping[str, str]) -> StreamAdapter:
    """The tailer's adapter for the row's ladder kind, wired to the container's records: vmodels from `cells_served`, a
    prompted arm's SI text through the misalign runner's `si_text_for` (the runner's own harness SI ids), misalign request
    messages through the driver-venv helper, GSM8K store ownership from the children's out roots."""
    kind = row["ladder"]["kind"]
    spec, ref = producer.spec, producer.ref
    common = dict(container_root=root, attempt_root_vol=attempt_root_vol)

    def vmodel_of(arm: str) -> str:
        h = lp.harness_id(arm)
        vm = vmodel_by_harness.get(h)
        if not vm:
            raise LadderUnavailable(f"no virtual model served for arm {arm} ({h}) in cells_served.json")
        return vm

    system_cache: Dict[str, Optional[str]] = {}

    def system_of(arm: str) -> Optional[str]:
        si = si_by_harness.get(lp.harness_id(arm))
        if not si:
            return None
        if arm in system_cache:                                   # F-AM: ONE `si-text` helper per arm, never one per at-cap row
            return system_cache[arm]
        r = subprocess.run([driver_python, str(Path(tree) / "scripts/modal/seedrep_ladder_helpers.py"), "si-text", "--si-id", si],
                           capture_output=True, text=True, env=dict(env), cwd=str(tree), timeout=300)
        if r.returncode != 0:
            raise LadderUnavailable(f"si-text helper rc {r.returncode} for {si}: {(r.stdout + r.stderr)[-400:]}")
        system_cache[arm] = r.stdout.rstrip("\n")
        return system_cache[arm]
    if kind == "static":
        return StaticStreamAdapter(L, spec, ref, attempt_id, leg=row["leg"], vmodel_of=vmodel_of, system_of=system_of, **common)
    if kind == "misalign":
        cells_json = root / SERVE_DIR / "cells.json"
        return MisalignStreamAdapter(L, spec, ref, attempt_id,
                                     messages_of=lambda reqs: misalign_messages_via_helper(driver_python, tree, cells_json, reqs, env), **common)
    if kind == "controls":
        # the seed is READ OFF the seed-owned root (every controls child owns `controls_cap<N>__s<s>`); the item prompt
        # and the arm's sampling row are resolved IN-CONTAINER through the driver-venv helper (the driver's own corpora loaders and
        # its `controls_gen_config`); a prompted arm's system text is the same `si-text` the static adapter uses. A test may inject
        # `controls_prompt_resolver` / `controls_sampling` through ctx; production needs neither.
        # F-AK: the store's owner is the child whose CELL names the store (the driver's --cell = the event's cell_id) at the store's
        # seed base (the seed-owned root) -- keyed through `lp.ladder_arm_of_cell`, the SAME translation `records()` applies to the event.
        # Indexing the children by their arm LABEL (`ar_c0p3__arditi_allpos`) and looking the Arditi store up by `arditi_allpos` finds no child.
        config_of_child: Dict[Tuple[str, int], str] = {}
        for k in row["children"]:
            owner = (lp.ladder_arm_of_cell(str(k["cell"])), int(k["s"]))
            if owner in config_of_child:
                raise LadderUnavailable(f"controls: two children of this row own the store for arm {owner[0]} at seed base {owner[1]} -- the plan's "
                                        "seed-owned roots must be unique, as validate_plan requires")
            config_of_child[owner] = str(k.get("config") or "")
        prompt_cache: Dict[Tuple[str, str], str] = {}
        sampling_cache: Dict[str, Dict[str, Any]] = {}

        def prompts_of(control: str, item_ids: Sequence[str]) -> Mapping[str, str]:
            # F-AM: ONE `controls-prompts` helper per (stream, control) for the ids not yet resolved -- running one per DISTINCT at-cap
            # (control, item) (the cache filled one asked id at a time) after the engine was torn down costs hours of idle H200
            missing = sorted({str(i) for i in item_ids if (control, str(i)) not in prompt_cache})
            if missing:
                prompt_cache.update({(control, i): p for i, p in controls_prompts_via_helper(driver_python, tree, control, missing, env=env).items()})
            return {str(i): prompt_cache[(control, str(i))] for i in item_ids if (control, str(i)) in prompt_cache}

        def prompt_of(control: str, item_id: str, arm: str) -> Tuple[str, Optional[str]]:
            prompt = prompts_of(control, [item_id]).get(item_id)
            if prompt is None:
                raise LadderUnavailable(f"controls: no pinned {control} row carries item {item_id} (the judged corpora loaders)")
            return prompt, system_of(arm)

        def sampling_of(arm: str, s: int) -> Mapping[str, Any]:
            cfg_rel = config_of_child.get((arm, int(s)))
            if not cfg_rel:
                raise LadderUnavailable(f"controls: no child of this row owns the store for arm {arm} at seed base {s} (children own "
                                        f"{sorted(config_of_child)}; labels {sorted({str(k['arm']) for k in row['children']})}) -- its sampling row "
                                        "cannot be resolved (F-AK: the owner is the child whose CELL names the store, never its arm label)")
            if cfg_rel not in sampling_cache:
                sampling_cache[cfg_rel] = dict(controls_sampling_via_helper(driver_python, tree, cfg_rel, env=env))
            return sampling_cache[cfg_rel]
        injected_sampling = ctx.get("controls_sampling")
        injected_prompt = ctx.get("controls_prompt_resolver")
        return ControlsStreamAdapter(L, spec, ref, attempt_id, seed_of_store=controls_seed_of_store, vmodel_of=vmodel_of,
                                     sampling=(dict(injected_sampling) if injected_sampling else None),
                                     sampling_of=(None if injected_sampling else sampling_of),
                                     prompt_of=injected_prompt or prompt_of,
                                     prompts_of=(None if injected_prompt else prompts_of),      # the batch path is production's (F-AM)
                                     system_of=(None if injected_prompt else system_of), **common)
    if kind == "gsm8k":
        owners = {}
        for k in row["children"]:
            owners[k["cell"]] = (k["arm"], int(k["s"]))

        def owner_of(path: Path) -> Tuple[str, int]:
            for part in reversed(path.parts):
                for cell, own in owners.items():
                    if part == cell or part.startswith(cell):
                        return own
            raise LadderUnavailable(f"{path}: no GSM8K child owns this outcomes store")
        return Gsm8kStreamAdapter(L, spec, ref, attempt_id, owner_of=owner_of, **common)
    raise LadderUnavailable(f"no tailer adapter for ladder kind {kind!r}")


PUBLICATION_FAILED = "PUBLICATION_FAILED"


def terminal_publish(root: Path, *, status: str, attempt_id: str, standup_id: str, barrier: Callable[[], None]) -> str:
    """Every terminal path that claims a durable verdict -- a failure incl. the Petri proxy's `PETRI_<state>` on --teardown-after-failure --
    writes RESULT and then runs the SYNCHRONOUS THROWING Volume barrier (the best-effort `_sync_all("final")` swallowed
    `VOL.commit()` failures, so a proxy could exit with terminal Dict evidence and no durable Volume copy). A barrier failure rewrites RESULT as
    PUBLICATION_FAILED (the caller's status kept in the line) and that is the status returned; a second barrier attempt is best-effort."""
    (root / RESULT_FILE).write_text(f"{utc()} status={status} rc=1 attempt={attempt_id} standup={standup_id}\n")
    try:
        barrier()
    except PublicationFailed as e:
        (root / RESULT_FILE).write_text(f"{utc()} status={PUBLICATION_FAILED} rc=1 attempt={attempt_id} standup={standup_id} before={status} why={str(e)[:200]!r}\n")
        try:
            barrier()
        except PublicationFailed:
            pass
        return PUBLICATION_FAILED
    return status


def thread_stack(thread: threading.Thread) -> List[str]:
    """The formatted stack of a live thread (what it is blocked in), [] once it has ended."""
    frame = sys._current_frames().get(thread.ident) if thread.ident is not None else None
    return traceback.format_stack(frame) if frame is not None else []


def dump_all_threads(log: Callable[[str], Any] = print) -> None:
    """Every live thread's stack through ``log`` (thread NAMES included), plus faulthandler's own dump to fd 2 when stderr has one."""
    names = {t.ident: t.name for t in threading.enumerate()}
    for tid, frame in sys._current_frames().items():
        log(f"[row] thread {names.get(tid, '?')} ({tid}):\n{''.join(traceback.format_stack(frame))}")
    try:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    except Exception:  # noqa: BLE001 -- a stderr without a fileno (a test's capture): the dump above already carries every thread
        pass


def run_post_result_tail(work: Callable[[], Any], *, deadline_s: float, phase_of: Callable[[], Any] = lambda: None,
                         log: Callable[[str], Any] = print, dump: Callable[[], Any] = dump_all_threads) -> Dict[str, Any]:
    """F-AM's belt: run ``work`` (the post-RESULT tail: drain -> seal -> inventory -> acceptance) in a daemon thread and wait at most
    ``deadline_s``. Returned: ``{"hang": False, "elapsed_s", "result"}`` when it returned (its exception re-raised here, so the caller's
    `fail(...)` classification is unchanged); ``{"hang": True, "phase", "deadline_s", "elapsed_s", "blocking", "at"}`` when it did not --
    the blocking phase and the tail thread's stack are logged as POST_RESULT_HANG with a dump of every thread, and the CALLER returns
    (the container exits; the tail thread is a daemon). The S3 and S5 controls rows sat 50-110 min after `[sync final]`
    (an idle H200 each, 16 h to the production container timeout) because that tail never returned; nothing bounded it."""
    out: Dict[str, Any] = {}

    def run() -> None:
        try:
            out["result"] = work()
        except BaseException as e:  # noqa: BLE001 -- re-raised in the caller's thread
            out["error"] = e
    t0 = time.time()
    t = threading.Thread(target=run, daemon=True, name="post-result-tail")
    t.start()
    t.join(max(0.0, float(deadline_s)))
    elapsed = round(time.time() - t0, 1)
    if t.is_alive():
        phase, stack = phase_of(), thread_stack(t)
        log(f"[row] {POST_RESULT_HANG} phase={phase} deadline_s={deadline_s} elapsed_s={elapsed}: the post-RESULT tail did not return; "
            f"returning with the RESULT already written. Blocking call:\n{''.join(stack[-3:])}")
        dump()
        return {"hang": True, "phase": phase, "deadline_s": float(deadline_s), "elapsed_s": elapsed, "blocking": stack, "at": utc()}
    if "error" in out:
        raise out["error"]
    return {"hang": False, "elapsed_s": elapsed, "result": out.get("result")}


def post_result_outcome(tail: Mapping[str, Any], *, attempt_id: str, accepted_holder: Optional[str]) -> Dict[str, Any]:
    """The row's TERMINAL verdict from the bounded tail's record: ``{status, acceptance, post_result[, error]}``.
    - the tail RETURNED: DONE, acceptance = its `finish_attempt` result.
    - it hung AFTER the acceptance election (the wave's accepted Dict names THIS attempt): DONE -- the row is covered, the acceptance is
      recorded from the Dict (the CLOSED marker / closing commits may not have landed: the reconciler's domain); recovery never re-runs
      completed evaluation work.
    - it hung BEFORE the election (no holder, or another attempt's): POST_RESULT_TIMEOUT, an EXPLICIT terminal failure with the hang's
      diagnostics (phase, elapsed, blocking frames) -- what `call_failure`, the watcher's FAILED event, `steward_tick` and
      `reattempt_decision` (infra: <= 1 automatic re-attempt, then HOLD) read. fo2 returned `DONE, acceptance None` here, which every
      classifier read as a healthy call: the attempt stayed missing (PRODUCER_MISSING, no FAILED), dependent queues could not close and
      the steward waited to its 36 h deadline. The RESULT file is never rewritten: it records what the children did; this status records
      what the tail did not."""
    post = {k: v for k, v in tail.items() if k != "result"}
    if not tail.get("hang"):
        return {"status": "DONE", "acceptance": tail.get("result"), "post_result": post}
    post["accepted_holder"] = accepted_holder
    if accepted_holder is not None and str(accepted_holder) == str(attempt_id):
        post["hang_after_acceptance"] = True
        return {"status": "DONE", "post_result": post,
                "acceptance": {"state": AttemptState.ACCEPTED, "accepted_attempt": attempt_id, "attempt_id": attempt_id,
                               "note": "the post-RESULT tail hung after the acceptance election; the CLOSED marker is the reconciler's to verify"}}
    post["hang_after_acceptance"] = False
    post["terminal_status"] = POST_RESULT_TIMEOUT
    return {"status": POST_RESULT_TIMEOUT, "acceptance": None, "post_result": post,
            "error": (f"{POST_RESULT_TIMEOUT}: the post-RESULT tail did not return within {tail.get('deadline_s')} s (phase {tail.get('phase')}); "
                      f"the attempt is not accepted (holder {accepted_holder!r}); RESULT left as written")}


def make_publication_barrier(syncers: Sequence[TreeSyncer], lock: threading.Lock) -> Callable[[], None]:
    """The SYNCHRONOUS, THROWING publication step every acceptance-path write goes through: copy every changed
    file of the attempt tree to the volume and `VOL.commit()`; any exception is re-raised as :class:`PublicationFailed` so the
    acceptance never follows an unpublished COMPLETE.json / inventory. The periodic `_sync_all` stays best-effort."""
    def barrier() -> None:
        with lock:
            try:
                n = sum(s.copy_changed() for s in syncers)
                VOL.commit()
            except Exception as e:  # noqa: BLE001
                raise PublicationFailed(f"publication failed: {type(e).__name__}: {e}") from e
            print(f"[publish] {n} file(s) committed", flush=True)
    return barrier


def _petri_proxy_wait(row, ctx, root: Path, serve_dir: Path, endpoint: dict, stop_dict, key: str, hb_state: dict, *, attempt_id: str,
                      attempt_root_vol: str, call_id: str, task_id: str) -> Dict[str, Any]:
    """The Petri proxy: a byte-for-byte TCP relay in front of the loopback proxy, `modal.forward` tunnel, the tunnel
    record (its required fields) in the sentinels Dict AFTER the key's stale sentinels are cleared, heartbeat + STOP +
    verdict mirrored from the Dict; ends ONLY on (a) the drive's completion evidence BOUND to this attempt -> `complete`
    (rc 0, eligible for acceptance), (b) an unbound STOP -> `stopped_unbound` (rc 1), (c) the wave's stop Dict ->
    `cancelled` (rc 1), (d) the container deadline -> `timeout` (rc 1). A timeout or a cancellation is never a success."""
    sentinels = modal.Dict.from_name(ctx["dicts"]["sentinels"], create_if_missing=True)
    bound = int(endpoint.get("proxy_port") or endpoint["proxy_url"].rsplit(":", 1)[1].split("/")[0])
    relay_port = relay_port_for(bound)
    relay = start_tcp_relay(relay_port, bound)
    hb_state["phase"] = "petri-proxy"
    popped = clear_stale_sentinels(sentinels, key)
    outcome: Dict[str, Any] = {"state": "timeout", "why": "the container deadline passed with no bound completion evidence",
                               "stale_sentinels_cleared": popped, "evidence": None}
    with modal.forward(relay_port) as tunnel:   # F-AL: relay_port is a free port <= 65535 by construction (relay_port_for)
        rec = petri_tunnel_record(url=tunnel.url, host=tunnel.host, port=tunnel.port, proxy_port=bound, relay_port=relay_port, call_id=call_id,
                                  task_id=task_id, attempt_id=attempt_id, attempt_root_vol=attempt_root_vol, row=row)
        write_json(serve_dir / PETRI_TUNNEL_FILE, rec)
        sentinels[f"{key}/{PETRI_TUNNEL_FILE}"] = json.dumps(rec)
        print(f"[petri] tunnel up: {tunnel.url} -> {relay['listen']} -> {relay['target']} (key {key}, attempt {attempt_id})", flush=True)
        last_seen: Dict[str, bytes] = {}
        noise_seen: set = set()
        try:
            deadline = time.time() + row["modal"]["timeout_s"] - 1200
            while time.time() < deadline:
                try:
                    content = sentinels.get(f"{key}/driver_heartbeat")
                except Exception:  # noqa: BLE001
                    content = None
                if content is not None:
                    b = content.encode() if isinstance(content, str) else bytes(content)
                    if last_seen.get("driver_heartbeat") != b:
                        last_seen["driver_heartbeat"] = b
                        (serve_dir / "driver_heartbeat").write_bytes(b)
                        hb_state["extra"] = {"drive_heartbeat": b.decode(errors="replace")[:120]}
                ev = petri_completion_evidence(sentinels, key, attempt_id)
                for n_ in ev.get("noise") or []:
                    if n_ not in noise_seen:
                        noise_seen.add(n_)
                        print(f"[petri] IGNORED (not this attempt's evidence): {n_}", flush=True)
                if ev["state"] != "waiting":
                    # mirror ONLY evidence bound to this attempt, the verdict BEFORE the STOP
                    mirror_bound_completion(sentinels, serve_dir, key, attempt_id)
                    outcome.update(ev)
                    break
                try:
                    if stop_dict.get(key) is not None:
                        outcome.update(state="cancelled", why="the wave's stop Dict names this key", evidence=None)
                        break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(30)
        finally:
            try:
                sentinels.pop(f"{key}/{PETRI_TUNNEL_FILE}")
            except Exception:  # noqa: BLE001
                pass
    relay["stop"]()
    rc = 0 if outcome["state"] == "complete" else 1
    outcome["child"] = {"eval_leg": "petri", "arm": "*", "s": 0, "rc": rc, "relay": dict(relay["counters"]), "state": outcome["state"]}
    outcome["tunnel_record"] = rec
    return outcome


def start_tcp_relay(listen_port: int, target_port: int, target_host: str = "127.0.0.1") -> dict:
    """arditi_allpos_served's byte-for-byte TCP relay (asyncio, its own thread): the forwarder reaches the relay, the relay
    reaches the loopback-only proxy."""
    import asyncio
    counters = {"connections": 0, "open": 0, "errors": 0}
    loop = asyncio.new_event_loop()
    server_ref: dict = {}

    async def pipe(reader, writer):
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    async def handle(client_r, client_w):
        counters["connections"] += 1
        counters["open"] += 1
        try:
            up_r, up_w = await asyncio.open_connection(target_host, target_port)
        except Exception:  # noqa: BLE001
            counters["errors"] += 1
            counters["open"] -= 1
            client_w.close()
            return
        try:
            await asyncio.gather(pipe(client_r, up_w), pipe(up_r, client_w))
        finally:
            counters["open"] -= 1

    async def serve():
        server_ref["server"] = await asyncio.start_server(handle, "0.0.0.0", listen_port)
        async with server_ref["server"]:
            await server_ref["server"].serve_forever()

    def run():
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(serve())
        except Exception as e:  # noqa: BLE001
            counters["fatal"] = f"{type(e).__name__}: {e}"

    th = threading.Thread(target=run, daemon=True, name="tcp-relay")
    th.start()
    time.sleep(0.5)

    def stop():
        srv = server_ref.get("server")
        if srv is not None:
            loop.call_soon_threadsafe(srv.close)
        loop.call_soon_threadsafe(loop.stop)
    return {"listen": f"0.0.0.0:{listen_port}", "target": f"{target_host}:{target_port}", "counters": counters, "stop": stop}


# The three functions that run DRIVER children carry the lane pool bearer as a Modal secret (POOL_BEARER_SECRETS; the pool
# app's own transport) -- `container_env` scrubs it from every child's base env and `child_env_with_pool_bearer` hands it to the
# --attacker-endpoint children only. run_petri_proxy runs no in-container driver (the grading host drive) and carries none.
@app.function(image=image, volumes={VOL_PATH: VOL}, gpu="H200", cpu=lp.LEG_CPU, memory=lp.LEG_MEMORY_MIB, timeout=16 * 3600, retries=0,
              secrets=POOL_BEARER_SECRETS)
def run_view_leg(row: dict, ctx: dict) -> dict:
    return _run_standup_row(row, ctx)


@app.function(image=image, volumes={VOL_PATH: VOL}, gpu="H200", cpu=lp.LEG_CPU, memory=lp.LEG_MEMORY_MIB, timeout=16 * 3600, retries=0,
              secrets=POOL_BEARER_SECRETS)
def run_arditi_leg(row: dict, ctx: dict) -> dict:
    return _run_standup_row(row, ctx)


@app.function(image=image, volumes={VOL_PATH: VOL}, gpu="H200", cpu=lp.LEG_CPU, memory=lp.LEG_MEMORY_MIB, timeout=8 * 3600, retries=0,
              secrets=POOL_BEARER_SECRETS)
def run_rung(row: dict, ctx: dict) -> dict:
    return _run_standup_row(row, ctx)


@app.function(image=image, volumes={VOL_PATH: VOL}, gpu="H200", cpu=lp.PETRI_CPU, memory=lp.PETRI_MEMORY_MIB, timeout=12 * 3600, retries=0)
def run_petri_proxy(row: dict, ctx: dict) -> dict:
    return _run_standup_row(row, ctx)


FUNCTIONS = {"run_view_leg": run_view_leg, "run_arditi_leg": run_arditi_leg, "run_rung": run_rung, "run_petri_proxy": run_petri_proxy}


# --------------------------------------------------------------------------- #
# local side: staging, plan-driven spawn, pull                                  #
# --------------------------------------------------------------------------- #

def _record_call(rs: Path, rec: dict) -> None:
    rs.mkdir(parents=True, exist_ok=True)
    with (rs / "calls.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _legs_log(rs: Path, line: str) -> None:
    rs.mkdir(parents=True, exist_ok=True)
    with (rs / "legs.log").open("a") as f:
        f.write(f"{utc()} {line}\n")
    print(line, flush=True)


PERSIST_COMPILE_CACHES_CTX = "persist_compile_caches"           # the spawn_ctx key the launcher's $SEEDREP_PERSIST_COMPILE_CACHES rides to the worker
SEED_COMPILE_CACHES_CTX = "seed_compile_caches"                 # ... and the one $SEEDREP_SEED_COMPILE_CACHES rides (F-Y; raw value, "" = unset)


def spawn_ctx(plan: Mapping[str, Any], app_name: str, pool_urls: Optional[Mapping[str, str]], *, smoke: Optional[Mapping[str, Any]] = None,
              env: Optional[Mapping[str, str]] = None) -> dict:
    """The per-container context: the wave's identity, Dicts, roots and dependency hashes -- or the SMOKE wave's (an isolated
    identity) when ``smoke`` is given. Operator settings read in THIS (launcher / steward) process ride here explicitly --
    the remote worker's environ is the image's, not the operator's: ``persist_compile_caches`` = ``$SEEDREP_PERSIST_COMPILE_CACHES``
    verbatim ("" when unset = the OFF default; "1" persists -- and only from a DONE container; the inode gate), from ``env``
    (default ``os.environ``); ``seed_compile_caches`` =
    ``$SEEDREP_SEED_COMPILE_CACHES`` verbatim ("" when unset = the OFF default; "1" seeds, bounded), carried raw so the worker's
    evidence can tell an unset default from an explicit off."""
    e = os.environ if env is None else env
    ctx = {"W": plan["W"], "wave_uuid": plan["wave_uuid"], "dicts": plan["dicts"], "plans_root": plan["plans_root"],
           PERSIST_COMPILE_CACHES_CTX: str(e.get(PERSIST_COMPILE_CACHES_ENV, "") or "").strip(),
           SEED_COMPILE_CACHES_CTX: str(e.get(SEED_COMPILE_CACHES_ENV, "") or "").strip(),
           "ladder_root": plan["ladder_root"], "results_root": plan["results_root"], "dep_hashes": LOCAL_DEP_HASHES, "app_name": app_name,
           "pool_urls": dict(pool_urls or {}), "smoke": None,
           # the two identities: `wave_uuid` / `execution_wave_uuid` = the wave THIS container runs in (Dicts, ladder, attempt
           # root, ATTEMPT.json, stores, records); `production_wave_uuid` + `launch_plan_sha` = the plan the run is certified against
           "execution_wave_uuid": plan["wave_uuid"], "production_wave_uuid": plan["wave_uuid"], "launch_plan_sha": plan.get("launch_plan_sha")}
    if smoke is not None:
        ctx.update(wave_uuid=smoke["wave_uuid"], execution_wave_uuid=smoke["wave_uuid"], dicts=smoke["dicts"], ladder_root=smoke["ladder_root"],
                   results_root=smoke["results_root"], smoke={"smoke_id": smoke["smoke_id"], "production_wave_uuid": plan["wave_uuid"]})
    return ctx


class LaunchRefused(SystemExit):
    """A launch the entrypoint refuses by name (draft fan-out, an unsupported rung, a multi-turn row without pools, …)."""


pool_is_healthy = lp.pool_is_healthy                        # the ONE health rule (lp.POOL_HEALTH_RULE), shared with the watcher


def select_launch_rows(plan_d: Mapping[str, Any], *, go: str = "", legs: str = "", smoke: str = "", smoke_id: str = "",
                       smoke_wave_uuid: Optional[str] = None, smoke_max_children: int = 5, smoke_timeout_h: float = 4.0,
                       pool_urls: Optional[Mapping[str, str]] = None,
                       pool_health: Callable[[str], bool] = pool_is_healthy,
                       pool_bearer: Optional[str] = None, pool_accepts: Optional[Callable[[str, str], bool]] = None,
                       carried: Optional[Mapping[str, Mapping[str, Any]]] = None,
                       carry_holds: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Tuple[List[dict], Optional[dict]]:
    """The entrypoint's launch guard, pure: PRODUCTION fan-out (`--go` / `--legs`) admits FINAL plans only; SMOKE mode
    (`--smoke <standup_id> --smoke-id <tag>`) admits DRAFT rows only, re-keyed into an isolated smoke wave and
    bounded (children, timeout). A multi-turn row without pool URLs, an unknown row, an unsupported rung mode refuse by name.
    A multi-turn row also needs the lane attacker-pool bearer the driver containers will carry (``pool_bearer``; None =
    the one this process read from $XDG_RUNTIME_DIR at import, `POOL_BEARER`) and every pinned pool must ACCEPT it
    (``pool_accepts(url, bearer)``, default `pool_accepts_bearer`: GET /v1/models -> 200) -- else refuse by name, never a
    silent 401 later. Returns (rows, smoke identity | None)."""
    if sum(1 for x in (go, legs, smoke) if x) != 1:
        raise LaunchRefused("[seedrep] pass exactly one of --go <standup_id>, --legs <leg,...>, --smoke <standup_id>")
    rows_all = list(plan_d["standups"]) + list(plan_d["rungs"])
    identity = None
    if smoke:
        rows_all = rows_all + list(plan_d.get("s5_probe_rows") or [])      # the probe rows run only under --smoke
        if plan_d["materialisation"] != "draft":
            raise LaunchRefused("[seedrep] --smoke runs DRAFT rows only (S1–S5 precede the FINAL plan); production fan-out is --go / --legs on the final plan")
        if not smoke_id:
            raise LaunchRefused("[seedrep] --smoke needs --smoke-id <tag> (S1, S1b, S2, S3, S4, S5, …)")
        picked = [r for r in rows_all if r["standup_id"] == smoke]
        if not picked:
            raise LaunchRefused(f"[seedrep] no draft plan row matches --smoke {smoke!r}")
        identity = lp.smoke_identity(plan_d, smoke_id, smoke_wave_uuid or str(uuid.uuid4()))
        try:
            picked = [lp.smoke_row(r, identity, max_children=int(smoke_max_children), max_timeout_s=int(float(smoke_timeout_h) * 3600)) for r in picked]
        except ValueError as e:
            # smoke_row refuses an unknown --smoke-id / a smoke without a declared adapter role (lp.SMOKE_ADAPTER_ROLES) -- surfaced
            # as this function's own refusal, by name, like every other launch refusal here
            raise LaunchRefused(f"[seedrep] --smoke {smoke!r} --smoke-id {smoke_id!r}: {e}") from None
    else:
        if plan_d["materialisation"] != "final":
            raise LaunchRefused("[seedrep] only the FINAL plan is launched (draft plans feed S0 / S5 and --smoke)")
        picked = [r for r in rows_all if r["standup_id"] == go] if go else \
                 [r for r in plan_d["standups"] if r["leg"] in {x.strip() for x in legs.split(",") if x.strip()}]
        if not picked:
            raise LaunchRefused(f"[seedrep] no plan row matches --go {go!r} / --legs {legs!r}")
        if carried:
            # fj2: a row whose attempt is CARRIED from a prior tree (the certificate's carry-over block; `seedrep_carry_over`) is skipped BY
            # NAME -- its accepted key already names the carried attempt in this wave's Dict, so it is never spawned
            skipped = [r for r in picked if r["standup_id"] in carried]
            for r in skipped:
                cf = carried[r["standup_id"]]
                print(f"[seedrep] CARRIED {r['standup_id']} from {cf.get('W7')}/{cf.get('attempt_id')} (D112; prior tree {str(cf.get('tree_sha'))[:12]}, "
                      f"source {cf.get('source')}) -- not spawned", flush=True)
            picked = [r for r in picked if r["standup_id"] not in carried]
            if not picked:
                raise LaunchRefused(f"[seedrep] every selected row is CARRIED from a prior tree ({sorted(r['standup_id'] for r in skipped)}) -- nothing to spawn")
        if carry_holds:
            # fj7 (B): a row whose prior source holds a COMPLETED attempt that the carry evaluator REFUSED is a HOLD by name --
            # never a spawn candidate. A schema-only refusal (a verdict without arm_binding) is not permission to regenerate canonical work.
            import seedrep_carry_over as co  # noqa: E402
            held = [r for r in picked if r["standup_id"] in carry_holds]
            for r in held:
                h = carry_holds[r["standup_id"]]
                print(f"[seedrep] HOLD {co.CARRY_REFUSED_HOLD} {r['standup_id']}: a completed attempt {h.get('attempt_id')} of {h.get('prior')} exists and the carry "
                      f"refused it ({h.get('clause')}: {str(h.get('why'))[:200]}) -- held, never re-spawned", flush=True)
            picked = [r for r in picked if r["standup_id"] not in carry_holds]
            if held and not picked:
                raise LaunchRefused(f"[seedrep] every selected row is HELD ({co.CARRY_REFUSED_HOLD}: {sorted(r['standup_id'] for r in held)}) -- a completed prior attempt "
                                    "exists for each and the carry refused it; fix the carry (re-audit / clauses), never re-spawn")
    bad = [r["standup_id"] for r in picked if r["kind"] == "rung" and r["leg"] != "petri" and r.get("rung_mode") not in EXECUTABLE_RUNG_MODES]
    if bad:
        raise LaunchRefused(f"[seedrep] rung rows {bad} carry a rung mode this app cannot execute ({sorted(EXECUTABLE_RUNG_MODES)})")
    mt = [r for r in picked if r["leg"] in lp.MULTITURN_LEGS]
    if mt and not pool_urls:
        raise LaunchRefused("[seedrep] multi-turn rows need --pool-urls <json> (the deployed pools' endpoints; pools are UP before any target)")
    for r in mt:
        if r.get("pool") and r["pool"] not in (pool_urls or {}):
            raise LaunchRefused(f"[seedrep] {r['standup_id']} pins pool {r['pool']!r} but --pool-urls has no URL for it")
    # main #211 (3353a3a5): a multi-turn target refuses instantly when no attacker was ever submitted -- the pool must be UP before
    # its targets spawn (the plan's launch_order); one health probe per distinct pinned pool, at spawn time
    probed: Dict[str, bool] = {}
    for r in mt:
        name = r.get("pool")
        if name not in probed:
            probed[name] = bool(pool_health(pool_urls[name]))
        if not probed[name]:
            raise LaunchRefused(f"[seedrep] {r['standup_id']}: attacker pool {name!r} at {pool_urls[name]} is not healthy (/health must answer 200|401 "
                                f"before any multi-turn target spawns -- {lp.ATTACKER_PRESENCE_RULE})")
    if mt:
        # the drivers reach the pools with the lane bearer (a Modal secret from the grading host's runtime dir); none read, or one the
        # pools reject, refuses HERE -- otherwise the multi-turn children run to rc 0 on 1296 x 401
        bearer = POOL_BEARER if pool_bearer is None else pool_bearer
        if not bearer:
            why = (POOL_BEARER_PROBLEM or "the bearer read at import is empty") if pool_bearer is None else "no bearer was given"
            raise LaunchRefused(f"[seedrep] multi-turn rows {[r['standup_id'] for r in mt]} need the lane attacker-pool bearer in the driver "
                                f"containers (${POOL_BEARER_ENV}, D99) and none was read: {why} -- run the pool app's --action endpoint (it "
                                "renders the bearer under $XDG_RUNTIME_DIR, D77) in THIS session before any multi-turn target spawns")
        sha12 = hashlib.sha256(bearer.encode()).hexdigest()[:12]
        accepts = pool_accepts or pool_accepts_bearer
        for name in probed:
            if not accepts(pool_urls[name], bearer):
                raise LaunchRefused(f"[seedrep] attacker pool {name!r} at {pool_urls[name]} rejects the lane bearer (sha256:{sha12}; GET /v1/models "
                                    "with it must answer 200) -- the pools were deployed under another bearer (rotated? a stale runtime dir?): "
                                    "redeploy them or re-render the bearer before any multi-turn target spawns (D99)")
        print(f"[seedrep] lane attacker-pool bearer sha256:{sha12} accepted by {sorted(probed)}; it rides the driver containers as "
              f"${POOL_BEARER_ENV} (a Modal secret; never argv, never the plan, never a volume file)", flush=True)
    return picked, identity


def smoke_wave_for(rs: Path, smoke_id: str) -> str:
    """The smoke's EXECUTION wave uuid, minted ONCE per smoke id and persisted at `RS/smokes/<id>/wave.json` (`lp.smoke_wave_path`; resume by
    row; the S5 record writer and the certificate read the same file)."""
    p = lp.smoke_wave_path(rs, smoke_id)
    if p.is_file():
        return lp.check_wave_uuid(json.loads(p.read_text())["wave_uuid"])
    wu = str(uuid.uuid4())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"smoke_id": smoke_id, "wave_uuid": wu, "minted_at": utc()}, indent=1) + "\n")
    return wu


def spawn_row(row: Mapping[str, Any], plan: Mapping[str, Any], app_name: str, rs: Path, pool_urls: Optional[Mapping[str, str]],
              *, smoke: Optional[Mapping[str, Any]] = None, reattempt_intent: Optional[str] = None, fanout_claim: Optional[Mapping[str, str]] = None) -> str:
    """`.spawn` ONE plan row on its declared function; registry queued-run emission per child (hooks.queued_modal) when the
    plans resolve locally; the call recorded in RS/calls.jsonl + legs.log (with its accepted key, the watcher's unit). A re-attempt
    carries its claimed ``reattempt_intent`` key in the ctx (the container claims `<intent>|executing` at entry)."""
    fn = FUNCTIONS[row["modal"]["function"]]
    ctx = spawn_ctx(plan, app_name, pool_urls, smoke=smoke)
    if reattempt_intent:
        ctx["reattempt_intent"] = str(reattempt_intent)
    if fanout_claim:
        ctx["fanout_claim"] = {"standup_id": str(fanout_claim["standup_id"]), "nonce": str(fanout_claim["nonce"])}   # delta 3 (4): the row's durable claim rides along
    fc = fn.spawn(dict(row), ctx)
    rec = {"call_id": fc.object_id, "app": app_name, "W": plan["W"], "wave_uuid": ctx["wave_uuid"], "standup_id": row["standup_id"],
           "leg": row["leg"], "engine": row["engine"], "stage": row["stage"], "shard": row["shard"], "control_key": row["control_key"],
           "accepted_key": row["accepted_key"], "kind": row["kind"], "function": row["modal"]["function"], "spawned_at": utc(),
           "smoke": (dict(smoke) if smoke else None),
           # the two identities + the plan this call was spawned from (`wave_calls` / the retry policy filter on it)
           "execution_wave_uuid": ctx["execution_wave_uuid"], "production_wave_uuid": ctx["production_wave_uuid"], "launch_plan_sha": ctx["launch_plan_sha"],
           "reattempt_intent": ctx.get("reattempt_intent"), "fanout_claim": ctx.get("fanout_claim")}
    _record_call(rs, rec)
    _legs_log(rs, f"SPAWNED {row['standup_id']} call={fc.object_id} key={row['control_key']}")
    _registry_queued(row, plan, fc.object_id, app_name, rs)
    return fc.object_id


# --------------------------------------------------------------------------- #
# the battery's rows record themselves in the results registry NATIVELY         #
# --------------------------------------------------------------------------- #
#
# The design was in place with none of the records: `resolve_spawn` composed every child's `SISTEER_REGISTRY_RUN_IDENTITY` (the staged
# identity file's in-container path) and `SISTEER_REGISTRY_SPOOL` (`<attempt root>/registry/records`) correctly -- the pasttense children's
# `.argv.json` records show both -- yet the attempt root on the volume had no `registry/` at all and the grading host's registry held zero
# seedrep runs. Two silences, one per side:
#   (container) the static / BoN / multi-turn drivers (`jailbreak_run.py generate`, `bon_served_run.py`) build no CellStore and call no
#     registry hook, so the identity env reached a process that never read it; nothing was ever written under the spool, and nothing
#     said so (`[registry]` appears in no child log). The CellStore drivers (controls, GSM8K) have the default recorder; the store-less
#     ones now get their generation recorded HERE, by the container, through the REAL hook under the child's own env
#     (`child_registry_completion`), and every row asserts BEFORE its standup that the hooks under a composed child env are ON with the
#     attempt spool (`registry_row_preflight`; refused by name `REGISTRY_EMISSION_OFF` otherwise -- a silent "not recorded" is the defect).
#   (the grading host) `_registry_queued` imported `sisteer.registry.hooks` under $HOME/venvs/modal, which carries no `sisteer`, and
#     returned on the ImportError silently (`grading_host_repo_src` puts <repo>/src on sys.path; the import failure is now SAID).
#   The standup's serving facts had a third: `write_endpoint`'s hook found no `registry_runs.json` beside `endpoint.json` ("no run
#   identities for the served keys ... serving facts not recorded") and would have resolved the SHARED /vol/registry/records as its spool;
#   the container now writes the served runs file (`write_registry_runs`) and hands the standup the attempt spool.

REGISTRY_EVIDENCE_KEY = "registry"                    # modal_evidence.json["registry"]: spool, identity_paths, enabled, probe, serving_runs, records_written
REGISTRY_EMISSION_OFF = "REGISTRY_EMISSION_OFF"       # the row's refusal status: it cannot record itself
REGISTRY_RUNS_FILE = "registry_runs.json"             # hooks.RUNS_FILE -- <serve dir>/registry_runs.json, read by native_standup.write_endpoint's hook
REGISTRY_SCRIPT = "scripts/modal/seedrep_served.py"
#: the probe a composed child env is asserted with: THE REAL hooks, under THAT env, in the DRIVER venv -- a TREE SCRIPT (fr4: every
#: app-side launch is a row of APP_SUBPROCESS_TABLE by its script), printing `{enabled, spool, identity_routes}` as one JSON line
REGISTRY_PROBE_REL = "scripts/sdfbase/seedrep/registry_probe.py"


def _hooks():
    """`sisteer.registry.hooks` (stdlib-only) -- raised, never swallowed: the caller says what it could not record."""
    from sisteer.registry import hooks as rh  # type: ignore
    return rh


def registry_identity_files(row: Mapping[str, Any], *, plans_root: str) -> Dict[str, str]:
    """`{child name: identity file path}` for every child of `row` naming a registry identity (`registry_identity_rel`), resolved under
    `plans_root` exactly as `lp.child_registry_env` resolves it into the child's env (the in-container volume path on a worker, RS on the
    grading host) -- the same spelling rule, so the path asserted is the path the child exports."""
    out: Dict[str, str] = {}
    for k in row.get("children") or []:
        rel = k.get("registry_identity_rel")
        if rel:
            out[child_log_name(k)] = f"{plans_root}/{lp.identity_rel_under_plans(rel)}"
    return out


def load_identity_file(path: str) -> Dict[str, Dict[str, object]]:
    """A staged identity file -> `{route: identity}` through `hooks.normalise_identities` (ONE identity routes `*`; a per-task map routes by
    the store writers' task names); refuses a file with no identity or a route without its `run_id` (the store would mint another id)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    idents = _hooks().normalise_identities(data if isinstance(data, dict) else None)
    if not idents:
        raise ValueError(f"{path}: no registry identity (neither one identity nor a {{task: identity}} map)")
    for route, ident in idents.items():
        if not ident.get("run_id"):
            raise ValueError(f"{path}: route {route!r} carries no run_id")
    return idents


def registry_probe(driver_python: str, env: Mapping[str, str], *, tree: Path, timeout_s: float = 180.0) -> Dict[str, Any]:
    """The REAL hooks evaluated under a child's env in the DRIVER venv: `{enabled, spool, identity_routes}` -- the staged tree's
    :data:`REGISTRY_PROBE_REL` (`registry_probe.py`), the one probe launch the subprocess table names."""
    r = subprocess.run([driver_python, str(Path(tree) / "scripts/sdfbase/seedrep/registry_probe.py")], env=dict(env), capture_output=True, text=True,
                       timeout=timeout_s)
    if r.returncode != 0:
        raise RuntimeError(f"registry probe rc={r.returncode}: {(r.stderr or r.stdout)[-600:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def registry_row_preflight(row: Mapping[str, Any], *, plans_root: str, attempt_root: str, base_env: Mapping[str, str],
                           probe: Callable[[Mapping[str, str]], Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    """BEFORE the standup (no GPU minute on a row that cannot record itself): every identity-bearing child's file EXISTS at its
    in-container path and loads (`load_identity_file`), and the REAL hooks under a composed child env -- `lp.child_registry_env` over the
    container env, the registry half of what `resolve_spawn` hands Popen -- report emission ON with the ATTEMPT spool resolved and the
    identity's routes read. A row with no identity-bearing child (a rung consumer) is probed under the spool alone. Returns
    (`modal_evidence.json["registry"]`, problems); problems -> :data:`REGISTRY_EMISSION_OFF`."""
    files = registry_identity_files(row, plans_root=plans_root)
    spool = f"{attempt_root}/{lp.SPOOL_REL}"
    block: Dict[str, Any] = {"spool": spool, "identity_paths": files, "n_children_with_identity": len(files), "enabled": None, "probe": None}
    problems: List[str] = []
    for name, path in files.items():
        if not os.path.isfile(path):
            problems.append(f"{name}: identity file absent at {path} (staged under {plans_root}? --stage-plans)")
            continue
        try:
            load_identity_file(path)
        except Exception as e:  # noqa: BLE001 -- named, never guessed
            problems.append(f"{name}: {type(e).__name__}: {e}")
    rel = next((k.get("registry_identity_rel") for k in (row.get("children") or []) if k.get("registry_identity_rel")), None)
    env = dict(base_env)
    if rel:
        env.update(lp.child_registry_env(plans_root=plans_root, registry_identity_rel=rel, attempt_root=attempt_root))
    else:
        env[lp.SPOOL_ENV] = spool
    block["probe_env"] = {k: env.get(k) for k in (lp.RUN_IDENTITY_ENV, lp.SPOOL_ENV)}
    try:
        res: Dict[str, Any] = dict(probe(env))
    except Exception as e:  # noqa: BLE001
        problems.append(f"registry probe failed under the child env: {type(e).__name__}: {e}")
        res = {}
    block["probe"] = res
    block["enabled"] = bool(res.get("enabled")) if res else False
    if res:
        if not res.get("enabled"):
            problems.append(f"emission is OFF under the composed child env ({lp.SPOOL_ENV}={spool}): SISTEER_REGISTRY_DISABLED / REGISTRY_EMIT=0 "
                            "reached the container env?")
        if res.get("spool") != spool:
            problems.append(f"the hooks resolve spool {res.get('spool')!r} under the child env, not the attempt spool {spool}")
        if rel and not res.get("identity_routes"):
            problems.append(f"the hooks read no identity routes from {env.get(lp.RUN_IDENTITY_ENV)}")
    return block, problems


def registry_runs_for_row(row: Mapping[str, Any], *, plans_root: str) -> Dict[str, Any]:
    """`{served_key: run | [run, ...]}` -- `hooks.load_runs_file`'s shape (`run = {run_id, campaign, expected: {serving_path}}`, one per
    served key) with the ADDITIVE list form for a served route several registered runs measure (the controls store's three evals, Petri's
    behaviours). THE KEY IS THE ENDPOINT'S OWN SPELLING: the identity's `extra.served_key` is the arm's route in the PLAN
    namespace (TrainTest: `constv2-mix-full__lora_r4_ce`, `prompted__si_constitution`, the Arditi ARM LABEL `arditi_allpos` -- never its
    staged id `ar_c0p3__arditi_allpos`), while `endpoint["adapters"]` / `served_positions` speak HARNESS ids
    (`si_constv2_mix_full__lora_r4_ce`, `prompted__constitution`); the hook reads each key as the served key to select the adapter block,
    so a plan-spelled key silently dropped every fitted run's adapter fingerprint, positions and intervention-hash check (all five fitted
    routes on the S1 endpoint). Translated here through the lane's ONE table (`lp.harness_id`, pinned equal to
    `registry_adapter.harness_cell_id`); an id outside it REFUSES by name, never a silent rename. Run ids are untouched. Never a run id
    as key (`emit_serving_facts_fanout`'s rule). Deduped by run id: the shard children of one static arm share one identity file."""
    rh = _hooks()
    by_key: Dict[str, List[Dict[str, Any]]] = {}
    seen: set = set()
    for _name, path in registry_identity_files(row, plans_root=plans_root).items():
        for route, ident in load_identity_file(path).items():
            rid = str(ident["run_id"])
            if rid in seen:
                continue
            seen.add(rid)
            plan_key = str((ident.get("extra") or {}).get("served_key") or "")
            if not plan_key:
                raise ValueError(f"{path}: route {route!r} records no extra.served_key")
            try:
                served_key = lp.harness_id(plan_key)                      # PLAN spelling -> the endpoint's HARNESS key
            except ValueError as e:
                raise ValueError(f"{path}: route {route!r}: {e}") from None
            try:
                rh.identity.parse_run_id(served_key)
            except ValueError:
                pass
            else:
                raise ValueError(f"{path}: extra.served_key {plan_key!r} translates to a run id, not a served key")
            by_key.setdefault(served_key, []).append({"run_id": rid, "campaign": str(ident["campaign"]),
                                                      "expected": {"serving_path": str(ident.get("serving_path") or "lora_native")}})
    return {k: (v[0] if len(v) == 1 else v) for k, v in by_key.items()}


def write_registry_runs(row: Mapping[str, Any], *, plans_root: str, serve_dir: Path) -> Dict[str, Any]:
    """`<serve dir>/registry_runs.json` (hooks.RUNS_FILE) BEFORE the standup, so `native_standup.write_endpoint`'s hook records the serving
    facts of every run this engine serves instead of noticing that it has no run identities. A row with no identity-bearing child writes
    nothing (n_runs 0)."""
    runs = registry_runs_for_row(row, plans_root=plans_root)
    n_runs = sum(len(v) if isinstance(v, list) else 1 for v in runs.values())
    out: Dict[str, Any] = {"path": None, "n_served_keys": len(runs), "n_runs": n_runs}
    if runs:
        p = Path(serve_dir) / REGISTRY_RUNS_FILE
        write_json(p, runs)
        out["path"] = str(p)
    return out


@contextlib.contextmanager
def _env_overlay(pairs: Mapping[str, Optional[str]]):
    prior = {k: os.environ.get(k) for k in pairs}
    try:
        for k, v in pairs.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def child_tasks(child: Mapping[str, Any]) -> List[str]:
    """The task spellings a child's rows carry: a string (`static/<leg>`, `strongreject_bon`, `agentic/<behaviour>`) or the controls
    child's `{control: task}` map (its values are the store writers' task names, the per-task identity map's routes)."""
    t = child.get("task")
    if isinstance(t, Mapping):
        return [str(v) for v in t.values()]
    return [str(t)] if t else []


def child_registry_completion(child: Mapping[str, Any], child_env: Mapping[str, str], *, rc: Optional[int], store_root: str,
                              attempt_id: str, log_name: str, code_sha: Optional[str] = None) -> List[str]:
    """After a driver child EXITS rc 0: one `run_patch` (generation complete; the child's store root as its uri; the shard facts under
    `extra_json`) per registered run the child's identity file claims for its tasks -- THE REAL hook (`hooks.emit_generation_complete`)
    under the child's own registry env (`SISTEER_REGISTRY_RUN_IDENTITY` / `SISTEER_REGISTRY_SPOOL`, as `resolve_spawn` composed them), so
    the ATTEMPT spool carries a record for EVERY child, the store-less drivers included (`jailbreak_run.py generate`, `bon_served_run.py`:
    no CellStore, no hook of their own -- the in-container silence). A failed child records nothing (its store is no generation);
    the hook is guarded (a refusal is one `[registry]` line and an empty list), never a row failure."""
    if rc != 0 or not child_env.get(lp.RUN_IDENTITY_ENV):
        return []
    rh = _hooks()
    extra = {"container_child": log_name, "attempt_id": attempt_id, "rc": rc, "eval_leg": child.get("eval_leg"), "arm": child.get("arm"),
             "s": child.get("s"), "goal_shard": child.get("goal_shard"), "goal_nshards": child.get("goal_nshards")}
    with _env_overlay({lp.RUN_IDENTITY_ENV: child_env.get(lp.RUN_IDENTITY_ENV), lp.SPOOL_ENV: child_env.get(lp.SPOOL_ENV),
                       lp.SUBSTRATE_ENV: child_env.get(lp.SUBSTRATE_ENV) or lp.SUBSTRATE}):
        # fr5 (5): the producer names the TREE sha the container runs (the app process sits outside any checkout: git says "unverified")
        paths = rh.emit_generation_complete(str(child.get("arm")), child_tasks(child), str(store_root), script=f"{REGISTRY_SCRIPT} child",
                                            extra={k: v for k, v in extra.items() if v is not None}, code_sha=code_sha)
    return list(paths or [])


#: fr5 (4): the refusals that must NOT end the row's runs -- another execution of the SAME intent / claim owns them
NO_TERMINAL_STATUSES: Tuple[str, ...] = (REATTEMPT_DUPLICATE, lp.FANOUT_DUPLICATE)
TERMINAL_STATES: Tuple[str, ...] = ("failed", "cancelled")
#: fr6: the LOGICAL terminal state of a run is decided on the grading host by the retry / hold decision, never by an attempt --
#: an attempt eligible for a re-attempt cannot know it is the last. `registry_hold_terminal_states` mints it when a standup is HELD.
HOLD_TERMINAL_SCRIPT = f"{REGISTRY_SCRIPT} hold"
LEDGER_TERMINAL_KEY = "terminal_states"                # the re-attempt ledger entry's record of the emission (additive, /2 schema)


def registry_utc_ms(stamp: Optional[str]) -> Optional[str]:
    """The registry's stamp grammar (UTC_MS, `YYYY-MM-DDTHH:MM:SS.mmmZ`) for one of this app's second-precision `utc()` stamps
    (`YYYY-MM-DDTHH:MM:SSZ` -> `.000Z`, deterministically -- a re-emission folds onto the same row); an ms stamp is kept; anything
    else is None (the caller stamps now)."""
    s = str(stamp or "")
    if re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", s):
        return s
    if re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", s):
        return s[:-1] + ".000Z"
    return None


def hold_terminal_state(status: Optional[str]) -> str:
    """`cancelled` when the failed call's status spells a cancellation (a drained / cancelled wave), else `failed`."""
    return "cancelled" if "CANCEL" in str(status or "").upper() else "failed"


def registry_hold_terminal_states(row: Mapping[str, Any], *, rs: Path, plan: Mapping[str, Any], entry: Mapping[str, Any], call_id: str,
                                  status: Optional[str], error: Optional[str], summary: Optional[Mapping[str, Any]],
                                  spool: Optional[str] = None) -> Dict[str, Any]:
    """The separately authorized, METADATA-ONLY publication of a HELD standup's terminal lifecycle -- ONE
    `run_state failed|cancelled` per run the standup's children carry, written on the grading host into ITS spool (`~/registry/records`,
    ingested directly: no pull, no accepted-only publication, no item result ever), with evidence `job_log` -> the failed attempt's
    RESULT on the volume (the watcher's summary names the attempt root; an attempt that left no verdict has none, and the note says so).
    The AUTHORITY is the decision recorded in the durable re-attempt ledger (`entry.held`): after the <= 1 re-attempt is decided the
    LOGICAL run's terminal state is known -- a failed attempt's own private spool never reaches the registry (registry_publish copies
    ACCEPTED, COMPLETE attempts only), which is why the failed gsm8k runs sat `queued`. `event_utc` = the entry's `held_at`, so a
    re-emission (a pass that died between the two ledger writes and recovered) folds onto the SAME run_state row. Identity files are
    read under `rs/registry/plans`; a child whose file cannot be read is a named problem, the others are still filed."""
    rh = _hooks()
    spool_path = spool or rh.resolve_spool()
    event_utc = registry_utc_ms(entry.get("held_at")) or rh.utc_now()
    out: Dict[str, Any] = {"state": hold_terminal_state(status), "call_id": call_id, "runs": 0, "records": [], "problems": [], "spool": spool_path,
                           "event_utc": event_utc, "evidence_uri": ""}
    if not spool_path:
        out["problems"].append("no registry spool resolves on this host (set SISTEER_REGISTRY_SPOOL); terminal states not recorded")
        return out
    summ = dict(summary) if isinstance(summary, Mapping) else {}
    root_vol = str(summ.get("attempt_root") or "")
    out["evidence_uri"] = f"{root_vol}/{RESULT_FILE}" if root_vol else ""
    note = f"D73 HOLD after call {call_id}: {entry.get('hold_reason')}" + (f"; status {status}" if status else "") + (f"; error {str(error)[:160]}" if error else "")
    if not root_vol:
        note += "; the attempt left no RESULT (no attempt root in the watcher's summary)"
    producer = rh.emit.producer_block(HOLD_TERMINAL_SCRIPT, str(plan.get("W") or "") or None)
    seen: set = set()
    for name, path in registry_identity_files(row, plans_root=str(Path(rs) / lp.PLANS_REL)).items():
        try:
            idents = load_identity_file(path)
        except Exception as e:  # noqa: BLE001
            out["problems"].append(f"{name}: {type(e).__name__}: {e}")
            continue
        for _route, ident in idents.items():
            rid = str(ident["run_id"])
            if rid in seen:
                continue
            seen.add(rid)
            w = rh.emit.RecordWriter(spool_path, str(ident["campaign"]), rid, producer)
            p = w.emit_state(out["state"], "job_log", out["evidence_uri"], event_utc=event_utc, note=note)
            w.push()
            if p:
                out["records"].append(p)
                out["runs"] += 1
    return out


def registry_attempt_terminal(row: Mapping[str, Any], *, plans_root: str, attempt_root: str, status: str, attempt_id: str, evidence_uri: str,
                              code_sha: str, standup_id: str, accepted_holder: Optional[str], cancelled: bool = False,
                              note: Optional[str] = None) -> Dict[str, Any]:
    """An ATTEMPT that is not DONE records, for every run its children carry, ONE `run_patch`
    of RECOVERY EVIDENCE -- `extra_json.attempt_terminal = {attempt_id, status, state_hint, evidence_uri, accepted_holder,
    closure_trouble}` -- and NEVER a logical `run_state`: an attempt eligible for a re-attempt cannot know it is the last, and an
    attempt that finish_attempt() already ELECTED (the accepted holder) and then failed to publish its closing marker
    (`PUBLICATION_FAILED`) is not an evaluation failure at all -- an earlier revision filed `run_state failed` there, registry_publish
    shipped it with the accepted attempt, and the reducer kept the false finished_utc past the later `generated`. The LOGICAL terminal
    state is the grading host's, minted by the HOLD (`registry_hold_terminal_states`). `state_hint` (`cancelled` for a stop-Dict drain,
    else `failed`) is what this attempt would have said, kept as evidence; `closure_trouble` is True when THIS attempt is the accepted
    holder. ONE record per run (shard children share an identity); a child whose identity cannot be read is a named problem, the rest
    are filed."""
    rh = _hooks()
    spool = f"{attempt_root}/{lp.SPOOL_REL}"
    producer = rh.emit.producer_block(f"{REGISTRY_SCRIPT} attempt-terminal", code_sha, substrate=lp.SUBSTRATE)
    closure_trouble = bool(accepted_holder) and str(accepted_holder) == str(attempt_id)
    evidence = {"attempt_id": attempt_id, "status": status, "state_hint": ("cancelled" if cancelled else "failed"), "evidence_uri": evidence_uri,
                "accepted_holder": accepted_holder, "closure_trouble": closure_trouble, "recorded_utc": rh.utc_now(), "standup_id": standup_id,
                **({"note": note} if note else {})}
    out: Dict[str, Any] = {"status": status, "state_hint": evidence["state_hint"], "closure_trouble": closure_trouble, "accepted_holder": accepted_holder,
                           "evidence_uri": evidence_uri, "standup_id": standup_id, "runs": 0, "records": [], "problems": []}
    seen: set = set()
    for name, path in registry_identity_files(row, plans_root=plans_root).items():
        try:
            idents = load_identity_file(path)
        except Exception as e:  # noqa: BLE001 -- named, the rest still filed
            out["problems"].append(f"{name}: {type(e).__name__}: {e}")
            continue
        for _route, ident in idents.items():
            rid = str(ident["run_id"])
            if rid in seen:
                continue
            seen.add(rid)
            w = rh.emit.RecordWriter(spool, str(ident["campaign"]), rid, producer)
            p = w.emit_patch(fields={"extra_json": {"attempt_terminal": dict(evidence)}})
            w.push()
            if p:
                out["records"].append(p)
                out["runs"] += 1
    return out


def count_spool_records(spool) -> int:
    """The record files under an attempt spool (`modal_evidence.json["registry"]["records_written"]`; 0 = nothing recorded, by name)."""
    root = Path(spool)
    return sum(1 for p in root.rglob("*") if p.is_file()) if root.is_dir() else 0


def _registry_queued(row, plan, call_id: str, app_name: str, rs: Path) -> Dict[str, Any]:
    """hooks.queued_modal ONCE per registered run of the row -- the identity files under `RS/registry/plans` the children name (the shard
    children of one static arm share one file: ONE queued run, not six) -- into the grading host's spool (`~/registry/records`), then ONE
    `[registry] <standup>: queued <n> run(s) -> <spool>` line. A lost record never costs a launch, but it is SAID, never swallowed: the
    earlier spawns registered nothing because `sisteer` was unimportable under the modal venv and the ImportError was returned on silently
    (`grading_host_repo_src`). Returns the summary (queued / records / spool / refused / skipped) the launcher prints."""
    sid = row.get("standup_id")
    summary: Dict[str, Any] = {"standup_id": sid, "queued": 0, "records": 0, "spool": None, "refused": [], "skipped": []}
    try:
        rh = _hooks()
    except Exception as e:  # noqa: BLE001
        summary["refused"].append(f"sisteer.registry.hooks unimportable: {type(e).__name__}: {e}")
        print(f"[registry] {sid}: spawn NOT registered -- {summary['refused'][0]} (is <repo>/src on this launcher's sys.path?)", flush=True)
        return summary
    summary["spool"] = rh.resolve_spool()
    seen: set = set()
    for k in row.get("children") or []:
        rel = k.get("registry_identity_rel")
        if not rel:
            summary["skipped"].append(f"{child_log_name(k)}: no registry_identity_rel")
            continue
        p = Path(rs) / rel
        if not p.is_file():
            summary["refused"].append(f"{child_log_name(k)}: identity file absent under RS: {p}")
            continue
        try:
            for _route, ident in load_identity_file(str(p)).items():
                rid = str(ident["run_id"])
                if rid in seen:
                    continue
                seen.add(rid)
                planned = {kk: v for kk, v in ident.items() if kk != "run_id"}      # the hook mints the id from the fields (queued_modal_for_plan)
                paths = rh.queued_modal(planned, modal_app_id=str(app_name), modal_call_id=call_id, script=REGISTRY_SCRIPT)
                if paths is None:
                    summary["refused"].append(f"{rid}: refused by queued_modal (see the [registry] notice above)")
                else:
                    summary["queued"] += 1
                    summary["records"] += len(paths)
        except Exception as e:  # noqa: BLE001
            summary["refused"].append(f"{child_log_name(k)}: {type(e).__name__}: {e}")
    line = f"[registry] {sid}: queued {summary['queued']} run(s) ({summary['records']} record(s)) -> {summary['spool']}"
    if summary["refused"]:
        line += f"; {len(summary['refused'])} NOT registered: {summary['refused'][:3]}"
    print(line, flush=True)
    return summary


def do_stage_tree(sha: str, restage: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix="seedrep-tree-") as td:
        tar = shallow_clone_tarball(sha, Path(td))
        print(f"[stage] pushing {tar.name} ({tar.stat().st_size // 1_000_000} MB) -> {REPOS}/", flush=True)
        with VOL.batch_upload(force=True) as batch:
            batch.put_file(str(tar), f"{volume_client_path(REPOS)}/{tar.name}")
        staged = unpack_tree.remote(sha, force=restage)
    print(json.dumps(staged, indent=1))
    if staged.get("error"):
        raise SystemExit(f"[stage] tree staging failed: {staged['error']}")
    return staged


def local_bank_problems(bank_local: Path, arditi_local: Path, pins: Mapping[str, Any], arditi_pins: Mapping[str, Any],
                        extra_views: Optional[Mapping[str, Mapping[str, str]]] = None, extra_pins: Optional[Mapping[str, str]] = None) -> dict:
    """The grading host's bank BEFORE upload: the lane bank (cells through the shared-bank symlink + the views make_bank_view.py built: s1..s3 and
    every plan-referenced extra view -- the S5 probe rows' included) and the Arditi stage's `objects/arditi_allpos/qwen35_4b/ladder/`
    (a read-only copy) -- every sha verified locally first, the extra objects under the plan's pins."""
    return verify_staged_objects(str(bank_local), str(arditi_local), pins, arditi_pins, extra_views=extra_views, extra_pins=extra_pins)


def stage_bank_arcnames(shas: Mapping[str, str], extra_views: Mapping[str, Mapping[str, str]]) -> List[Tuple[str, str]]:
    """What the bank tarball carries, as (bank-relative source, arcname): every verified object under `cells/`, the three s-views, the
    Arditi carrier + view, and EVERY plan-referenced extra view. Pure, so a test can hold it to the plan."""
    out: List[Tuple[str, str]] = []
    for sid in sorted(shas):
        if sid == lp.ARDITI_STAGED_ID:
            out.append((f"arditi:cells/{sid}", f"arditi_allpos/{lp.MODEL_KEY}/ladder/cells/{sid}"))
        else:
            out.append((f"bank:{lp.MODEL_KEY}/cells/{sid}", f"sdfbase/full22/{lp.MODEL_KEY}/cells/{sid}"))
    for v in list(lp.VIEWS) + sorted(extra_views):
        out.append((f"bank:{lp.bank_view_subpath(v)}", f"sdfbase/full22/{lp.bank_view_subpath(v)}"))
    out.append((f"arditi:cells_view/{lp.ARDITI_VIEW}", f"arditi_allpos/{lp.MODEL_KEY}/ladder/cells_view/{lp.ARDITI_VIEW}"))
    return out


def do_stage_bank(sha: str, bank_local: Path, arditi_local: Path, plan_d: Optional[Mapping[str, Any]] = None) -> dict:
    """Stage the bank the PLAN references: the 16 lane objects + every extra view and its objects that ANY row the app may execute names
    (`lp.bank_views_for_plan` over the executable rows AND the smoke-only `s5_probe_rows`; the objects pinned by the plan,
    `lp.bank_pins_for_plan` -- the probe objects 86fa8e9c / 0522a255 were never packaged). A plan-less call stages the
    fixed 16 only -- pass the draft plan."""
    pins = json.loads((LOCAL_REPO / lp.PINS_REL).read_text())
    apins = json.loads((LOCAL_REPO / lp.ARDITI_PINS_REL).read_text())
    extra = lp.bank_views_for_plan(plan_d) if plan_d else {}
    extra_pins = lp.bank_pins_for_plan(plan_d) if plan_d else {}
    rep = local_bank_problems(bank_local, arditi_local, pins, apins, extra_views=extra, extra_pins=extra_pins)
    if not rep["ok"]:
        raise SystemExit(f"[stage-bank] the grading host bank refuses before upload: {rep['problems']}")
    print(f"[stage-bank] local bank OK: {rep['n_entries']} entries, {len(rep['shas'])} shas == VIEW_MANIFEST == pins (views {list(lp.VIEWS) + sorted(extra)}; "
          f"{len(extra_pins)} plan-pinned extra object(s))", flush=True)
    tar_name = f"bank-{int(time.time())}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="seedrep-bank-") as td:
        tar = Path(td) / tar_name
        with tarfile.open(tar, "w:gz") as tf:
            # full22/<target>/{cells/<every verified object>, cells_view/s1..s3 + every plan-referenced dose view}; arditi_allpos/<target>/ladder/{cells/ar_c0p3__*, cells_view/ar_c0p3}
            for src, arcname in stage_bank_arcnames(rep["shas"], extra):
                kind, rel = src.split(":", 1)
                tf.add((arditi_local if kind == "arditi" else bank_local) / rel, arcname=arcname)
        print(f"[stage-bank] pushing {tar.name} ({tar.stat().st_size // 1_000_000} MB) -> {LANE}/", flush=True)
        with VOL.batch_upload(force=True) as batch:
            batch.put_file(str(tar), f"{volume_client_path(LANE)}/{tar_name}")
    out = unpack_bank.remote(tar_name, sha, pins, apins, extra, extra_pins)
    print(json.dumps(out, indent=1))
    if not out.get("ok"):
        raise SystemExit("[stage-bank] the bank on the volume does not verify")
    return out


def do_stage_plans(plans_local: Path, plans_root: str) -> int:
    """Stage identity files `<leg>/<arm>__s<s>.json` from `plans_local` to `plans_root` on the volume."""
    files = sorted(p for p in Path(plans_local).rglob("*.json") if p.is_file())
    if not files:
        raise SystemExit(f"[stage-plans] no identity files under {plans_local}")
    with VOL.batch_upload(force=True) as batch:
        for p in files:
            batch.put_file(str(p), f"{volume_client_path(plans_root)}/{p.relative_to(plans_local)}")
    print(f"[stage-plans] {len(files)} identity files -> {plans_root}", flush=True)
    return len(files)


class PullRefused(RuntimeError):
    """A pull that must not be recorded as a success: a download error, a root without COMPLETE.json, a producer root
    without its inventory, or a shard with no accepted attempt."""


RS_DETERMINISM_REL = "audits/determinism"                      # RS/audits/determinism/<table leg>__<attempt_id>.json (battery_tables.load_determinism reads it)


class DeterminismCollectRefused(RuntimeError):
    """A pulled determinism record that cannot be promoted: not the attempt's own, malformed, or colliding with different bytes."""


def validate_determinism_records(pulled: Path, rs: Path, *, attempt_id: str) -> List[Tuple[Path, Path]]:
    """Phase 1 of the promotion: EVERY `audits/determinism/*.json` under the pulled accepted attempt root read and held (JSON; names THIS attempt;
    `leg` == the file stem) and every destination `RS/audits/determinism/<leg>__<attempt_id>.json` checked for a conflict (present with DIFFERENT
    bytes) BEFORE anything is written -- one bad record refuses the whole shard with nothing promoted. Returns the (src, dst) plan."""
    src_dir = Path(pulled) / DETERMINISM_COPY_REL
    if not src_dir.is_dir():
        return []
    dst_dir = Path(rs) / RS_DETERMINISM_REL
    plan: List[Tuple[Path, Path]] = []
    for p in sorted(src_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            raise DeterminismCollectRefused(f"{p}: not JSON ({e})")
        if not isinstance(d, dict) or str(d.get("attempt_id") or "") != str(attempt_id):
            raise DeterminismCollectRefused(f"{p}: determinism record names attempt {d.get('attempt_id') if isinstance(d, dict) else '?'!r}, not the pulled accepted attempt {attempt_id!r}")
        if str(d.get("leg") or "") != p.stem:
            raise DeterminismCollectRefused(f"{p}: determinism record is of leg {d.get('leg')!r} but the file is named {p.stem!r}")
        dst = dst_dir / f"{p.stem}__{attempt_id}.json"
        if dst.is_file() and dst.read_bytes() != p.read_bytes():
            raise DeterminismCollectRefused(f"{dst}: already holds a DIFFERENT determinism record for leg {p.stem} / attempt {attempt_id}; not overwritten")
        plan.append((p, dst))
    return plan


def collect_determinism_records(pulled: Path, rs: Path, *, attempt_id: str) -> List[str]:
    """PROMOTE the pulled accepted attempt's `audits/determinism/<table leg>.json` records (written in-container by `determinism_evidence`, one
    per table leg) to `RS/audits/determinism/<table leg>__<attempt_id>.json` -- the flat directory `battery_tables.load_determinism` reads,
    keyed by (leg, attempt) so a leg served by several s1 shards keeps every shard's record (the reader COMBINES them; the same attempt twice
    refuses there). ATOMIC with the shard's admission: phase 1 validates every record and destination (`validate_determinism_records`),
    phase 2 copies; a copy failure ROLLS BACK the files this call created and refuses -- the caller writes the shard's manifest rows only after
    this returns, so a refused shard is never admitted and never leaves a new record visible. A byte-identical re-pull is a no-op. Returns the
    RS-relative paths written or confirmed."""
    plan = validate_determinism_records(pulled, rs, attempt_id=attempt_id)
    todo = [(src, dst) for src, dst in plan if not dst.is_file()]
    tmps: List[Tuple[Path, Path]] = []                                   # (tmp, dst) -- tmp = `.<name>.tmp` beside dst, outside the reader's *.json namespace
    published: List[Path] = []
    try:
        for src, dst in todo:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.tmp")
            tmps.append((tmp, dst))                                      # registered BEFORE the copy: a copy that creates then raises is still cleaned
            shutil.copyfile(src, tmp)
        for tmp, dst in tmps:                                            # every copy succeeded: publish by rename (atomic per file, same directory)
            os.replace(tmp, dst)
            published.append(dst)
    except OSError as e:
        leftovers = _remove_quiet([t for t, _d in tmps] + published)
        rolled = f"{len(tmps)} temp file(s) and {len(published)} published record(s) removed" if not leftovers else \
                 f"CLEANUP FAILED for {[str(x) for x in leftovers]} -- remove them by hand before re-pulling"
        raise DeterminismCollectRefused(f"promotion of {Path(pulled).name}'s determinism records failed ({type(e).__name__}: {e}); {rolled}; nothing promoted")
    return [str(dst.relative_to(rs)) for _src, dst in plan]


def _remove_quiet(paths: Sequence[Path]) -> List[Path]:
    """Remove every path that exists; return the ones that could NOT be removed (a rollback is claimed only when this is empty)."""
    failed: List[Path] = []
    for p in paths:
        try:
            if p.exists() or p.is_symlink():
                os.unlink(str(p))
        except OSError:
            failed.append(p)
    return failed


def verify_pulled_root(pulled: Path, *, producer_row: bool) -> List[str]:
    """What a pulled ACCEPTED attempt root must show LOCALLY before its manifest rows are written: the root
    exists, `COMPLETE.json` names this attempt, and -- for a producer row -- `queue_inventory.json` and `ladder_enqueued.json`
    are present, complete and agree on the entry count."""
    problems = []
    if not pulled.is_dir():
        return [f"{pulled}: not pulled (absent)"]
    comp = pulled / COMPLETE_JSON
    if not comp.is_file():
        problems.append(f"{pulled.name}: no {COMPLETE_JSON}")
    else:
        try:
            if json.loads(comp.read_text()).get("attempt_id") != pulled.name:
                problems.append(f"{pulled.name}: {COMPLETE_JSON} names another attempt")
        except ValueError:
            problems.append(f"{pulled.name}: {COMPLETE_JSON} is not JSON")
    if producer_row:
        inv, enq = pulled / INVENTORY_JSON, pulled / LADDER_ENQUEUED_JSON
        if not inv.is_file():
            problems.append(f"{pulled.name}: no {INVENTORY_JSON}")
        if not enq.is_file():
            problems.append(f"{pulled.name}: no {LADDER_ENQUEUED_JSON}")
        if inv.is_file() and enq.is_file():
            try:
                i, e = json.loads(inv.read_text()), json.loads(enq.read_text())
                if not e.get("complete"):
                    problems.append(f"{pulled.name}: {LADDER_ENQUEUED_JSON} complete=False")
                if int(i.get("n", -1)) != int(e.get("n", -2)):
                    problems.append(f"{pulled.name}: inventory n {i.get('n')} != ladder_enqueued n {e.get('n')}")
            except ValueError:
                problems.append(f"{pulled.name}: inventory / ladder_enqueued not JSON")
    return problems


def pull_leg(plan: Mapping[str, Any], leg: str, dest: Path, rs: Path, *, stages: str = "base", accepted: Any = None,
             run_get: Optional[Callable[[str, Path], None]] = None) -> Dict[str, Any]:
    """RS/stores/<leg>/<view>/<shard>/ (base) and RS/stores/<leg>/<view>/rungs/<cap>/<shard>/ (rungs) by `modal volume get` of
    the ACCEPTED attempt root only (the wave's accepted Dict, its key spelling), STAGE-AWARE (`base` | `rungs` | `all`);
    a download error FAILS the shard (never an empty manifest); every pulled root is verified locally (COMPLETE.json + the
    producer inventories) BEFORE its sha-manifest rows are recorded; a shard with no accepted attempt is REFUSED."""
    accepted = accepted if accepted is not None else modal.Dict.from_name(plan["dicts"]["accepted"], create_if_missing=True)
    rows = list(lp.required_accepted_keys(plan, leg, stages).items())
    by_id = {r["standup_id"]: r for r in list(plan["standups"]) + list(plan["rungs"])}
    # fj3: a CARRIED attempt's root is the one the carry record names under the PRIOR tree's results root -- never derived under
    # this plan's, where it does not exist; every other attempt resolves through the app's own spelling as before
    import seedrep_carry_over as co  # noqa: E402
    resolve_root = co.attempt_root_resolver(co.read_carry_over(Path(rs)), attempt_root_for)

    run_get = run_get or volume_get
    Path(rs).mkdir(parents=True, exist_ok=True)
    refused: List[str] = []
    pulled_ok: List[dict] = []
    for akey, sid in rows:
        r = by_id[sid]
        try:
            att = accepted_attempt_for(accepted, akey)
        except NoAcceptedAttempt as e:
            print(f"[pull] {e}", flush=True)
            refused.append(f"{sid}: no accepted attempt")
            continue
        src = resolve_root(plan["results_root"], r, att)
        out = dest / leg / r["engine"] / (f"shard{int(r['shard']):02d}" if r["stage"] == lp.STAGE_BASE else f"rungs/{r['stage']}/shard{int(r['shard']):02d}")
        out.mkdir(parents=True, exist_ok=True)
        print(f"[pull] {src} -> {out}", flush=True)
        try:
            run_get(src, out)
        except Exception as e:  # noqa: BLE001
            refused.append(f"{sid}: download failed: {type(e).__name__}: {e}")
            continue
        pulled = out / att
        problems = verify_pulled_root(pulled, producer_row=bool(r.get("ladder") and (r.get("ladder") or {}).get("cap_next") is not None))
        adopted, empty_adopted = adopted_from_attempts(pulled)
        if empty_adopted:
            problems.append(f"{ADOPTED_FROM_DIR}/{empty_adopted}: adopted attempt(s) declared with no evidence files")
        if problems:
            refused.append(f"{sid}: {problems}")
            continue
        if adopted:
            print(f"[pull] {sid}: accepted attempt {att} adopted evidence from {adopted}", flush=True)
        # the accepted attempt's determinism records (§9.2, `audits/determinism/<table leg>.json`) PROMOTED to RS/audits/determinism/ where the
        # tables read them -- keyed by (leg, attempt), never one <leg>.json overwriting another shard's. BEFORE the manifest rows: the manifest
        # is prepare's admission authority, so a shard whose records refuse is never admitted, and every record is validated
        # before any is promoted (no partial promotion)
        try:
            collected = collect_determinism_records(pulled, rs, attempt_id=att)
        except DeterminismCollectRefused as e:
            refused.append(f"{sid}: {e}")
            continue
        files = [(str(p.relative_to(out)), sha256_file(p)) for p in sorted(pulled.rglob("*")) if p.is_file()]
        with (rs / "pull_manifest.jsonl").open("a") as f:
            for row_ in pull_manifest_rows(att, files):
                f.write(json.dumps({**row_, "leg": leg, "engine": r["engine"], "stage": r["stage"], "shard": r["shard"], "standup_id": sid}) + "\n")
        pulled_ok.append({"standup_id": sid, "attempt_id": att, "n_files": len(files), "adopted_from": adopted, "determinism_collected": collected})
    if refused:
        raise PullRefused(f"[pull] REFUSED {len(refused)} of {len(rows)} rows: {refused}")
    return {"leg": leg, "stages": stages, "pulled": pulled_ok}


LAUNCH_LOCK_FILE = "launch.lock"
STEWARD_START_TIMEOUT_S = 90.0


def certificate_read_text(rs: Path) -> Callable[[str], Optional[str]]:
    """The reader `--go` / `--legs` / `--reattempt` re-hash the derived records' sources with: a `/vol/<rel>` source (an accepted
    attempt's COMPLETE.json / RESULT / modal_evidence.json) through the Modal Volume API -- the grading host has no /vol mount -- and every other source
    (a check output under RS) locally. The Volume handle is opened on the first /vol/ read only."""
    box: Dict[str, Any] = {}

    def read(path: str) -> Optional[str]:
        if not str(path).startswith(lp.VOL_PATH + "/"):
            return None                                               # delta 3 (1): no local / operator path is ever evidence
        if "vol" not in box:
            import seedrep_watch as sw  # noqa: E402
            box["sw"] = sw
            box["vol"] = modal.Volume.from_name(lp.MODAL_VOLUME_NAME, create_if_missing=False)
        return box["sw"].volume_read_text(box["vol"], str(path))
    return read


def live_smoke_accepted_getters(rs: Path, plan: Mapping[str, Any]) -> Callable[[str], Callable[[str], Any]]:
    """`accepted_get_for(smoke_id) -> get(accepted key)` over each smoke wave's LIVE accepted Dict: at --go the accepted
    attempt of every smoke row is what the Dict names NOW -- never a snapshot, which could name a successful-but-superseded attempt."""
    box: Dict[str, Any] = {}

    def for_smoke(sid: str) -> Callable[[str], Any]:
        if sid not in box:
            import seedrep_carry_over as co  # noqa: E402
            if sid == co.PRODUCTION_ACCEPTED_ID:
                # fj2: the certificate's carry-over block is held to THIS wave's LIVE accepted Dict (the carried keys must name the attempts NOW)
                d = modal.Dict.from_name(plan["dicts"]["accepted"], create_if_missing=False)
                box[sid] = lp.RecordingGetter(plan["dicts"]["accepted"], d.get)
                return box[sid]
            wave = lp.smoke_execution_wave(Path(rs), sid)
            if wave is None:
                box[sid] = lp.RecordingGetter(f"(no minted wave for {sid})", lambda key: None)
            else:
                name = lp.smoke_identity(plan, sid, wave)["dicts"]["accepted"]
                d = modal.Dict.from_name(name, create_if_missing=False)
                box[sid] = lp.RecordingGetter(name, d.get)            # delta 3 (2): every read RECORDED (the certificate carries them)
        return box[sid]
    for_smoke.getters = box  # type: ignore[attr-defined]
    return for_smoke


def live_accepted_dict_getters() -> Callable[[str], Callable[[str], Any]]:
    """fc11b: `accepted_get_for_dict(dict name) -> get(accepted key)` over the PRIOR trees' smoke Dicts BY NAME (a prior-tree PASS record names the Dict
    each row resolved through) -- what --go / --legs / --reattempt hold such a record's rows to NOW; one recording getter per Dict."""
    box: Dict[str, lp.RecordingGetter] = {}

    def for_dict(name: str) -> Callable[[str], Any]:
        if name not in box:
            d = modal.Dict.from_name(name, create_if_missing=False)
            box[name] = lp.RecordingGetter(name, d.get)
        return box[name]
    for_dict.getters = box  # type: ignore[attr-defined]
    return for_dict


def certified_launch_guard(cert: Mapping[str, Any], standup_ids: Sequence[str], *, where: str) -> None:
    """The ONE check every spawn boundary runs -- the launcher's selection, the fan-out's intent and each row it spawns
    (fresh or recovered), the re-attempt policy's spawn (and, through its `guarded_spawn`, the stale-intent recovery) -- every standup id must be in
    the certificate's `launch_set` (None = every planned row); a row outside it is refused by name, never spawned under a certificate whose S5
    coverage was derived without it. A row HELD by the certificate's SEALED carry-over block (`carry_refused_holds` -- every
    hold kind the helper reports, `carry_refused_hold` and `launch_history_unknown` alike) is refused by name and hold kind at the same
    boundaries: `select_launch_rows` read the holds, but the re-attempt pass and the stale-intent recovery spawned every same-plan call without
    them. The holds are read from the CERTIFICATE (sealed at certify), never from RS."""
    probs = lp.launched_ids_problems(cert, standup_ids)
    import seedrep_carry_over as co  # noqa: PLC0415
    holds = co.carry_refused_holds(cert.get("carry_over") if isinstance(cert, Mapping) else None)
    for sid in standup_ids:
        h = holds.get(str(sid))
        if h:
            probs.append(f"row {sid} is HELD by the certificate's carry-over block (hold {h.get('reason')}; clause {h.get('clause')}; prior attempt {h.get('attempt_id')}"
                         f"{' under ' + str(h.get('prior_W'))[:7] if h.get('prior_W') else ''}: {str(h.get('why'))[:160]}) -- a held row is never spawned")
    if probs:
        raise LaunchRefused(f"[seedrep] {where} REFUSED (D146; the certified launch set): " + "; ".join(probs))     # the hold problems name their kind themselves


def require_certificate(plan_d: Mapping[str, Any], *, plan_path: Path, rs: Path, cert_p: Path,
                        accepted_get_for: Optional[Callable[[str], Callable[[str], Any]]] = None, launched_ids: Optional[Sequence[str]] = None,
                        accepted_get_for_dict: Optional[Callable[[str], Callable[[str], Any]]] = None) -> dict:
    """A production launch, the fan-out and a re-attempt consume the LAUNCH CERTIFICATE and
    nothing else -- S0, S1-S5, the W gate, the decode proof, the utilisation verdict, draft == final, all re-validated on their own content, the
    derived records' sources RE-HASHED and every check RE-RUN, each smoke row's accepted attempt held to the LIVE accepted Dict, and the plan
    held by its BYTES (the content digest recomputed + the plan file's sha256 against the certificate's)."""
    cert = json.loads(cert_p.read_text()) if cert_p.is_file() else None
    reg = LOCAL_REPO / lp.DECODE_REGISTRY_REL
    probs = lp.go_preconditions(plan_d, certificate=cert, registry_sha256=(sha256_file(reg) if reg.is_file() else None), plan_path=Path(plan_path),
                                read_text=certificate_read_text(Path(rs)), accepted_get_for=(accepted_get_for or live_smoke_accepted_getters(Path(rs), plan_d)),
                                launched_ids=launched_ids,                    # fc9: every row named is in the certified launch set
                                accepted_get_for_dict=(accepted_get_for_dict or live_accepted_dict_getters()))   # fc11b: the prior smoke Dicts, LIVE, by name
    if probs:
        raise LaunchRefused("[seedrep] production launch REFUSED (D73 (4); the launch certificate): " + "; ".join(probs))
    return cert


def volume_get(src: str, out: Path) -> None:
    """`modal volume get --force <vol> <client path> <out dir>` -- the pull's ONE download primitive (`pull_leg`'s default; the re-audit's too)."""
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", VOL_NAME, volume_client_path(src), str(out)], check=True)


CARRIED_EVIDENCE_DIR = "carried"


def carried_evidence_root_rel(W7: str, wave_uuid: str, standup_id: str, attempt_id: str) -> str:
    """The carried entry's OWN evidence dir on the volume: `<seedrep root>/<wave>/carried/<standup_id>__<attempt_id>/evidence` -- the
    re-audit's verdicts + index and a copy of the standup's endpoint.json live here; the historical attempt root is never written."""
    return f"seedrep/{W7}/{wave_uuid}/{CARRIED_EVIDENCE_DIR}/{standup_id}__{attempt_id}/evidence"


def probe_declaration_from_root(pulled: Path) -> Dict[str, Any]:
    """The prior standup's PROBE traffic, named from the root's OWN records (never today's plan): `exp/serve/endpoint.json.probe_max_tokens` (the
    scaffolding's greedy probes, 48) and `exp/serve/positive_controls.json` -- its `salt_isolation` control fires ONE-TOKEN greedy requests per route
    (`n_routes` x `probe_set.n`; the Arditi root: 3 x 11 = 33 at `max_tokens=1`, `temperature=0.0`, neutral defaults), a probe kind the recorder's
    default class (PROBE_MAX_TOKENS alone) does not know, so they read UNCLASSIFIED_REQUESTS. Returns the caps to declare and the kind's expected count."""
    out: Dict[str, Any] = {"probe_max_tokens": None, "declared": {}, "source": {}}
    ep_p, pc_p = pulled / SERVE_DIR / "endpoint.json", pulled / SERVE_DIR / "positive_controls.json"
    try:
        ep = json.loads(ep_p.read_text()) if ep_p.is_file() else {}
        pc = json.loads(pc_p.read_text()) if pc_p.is_file() else {}
    except ValueError:
        return out
    if isinstance(ep, Mapping) and isinstance(ep.get("probe_max_tokens"), int):
        out["probe_max_tokens"] = int(ep["probe_max_tokens"])
        out["source"]["probe_max_tokens"] = f"{SERVE_DIR}/endpoint.json"
    si = ((pc.get("controls") or {}).get("salt_isolation") if isinstance(pc, Mapping) else None) or {}
    ps = (pc.get("probe_set") if isinstance(pc, Mapping) else None) or {}
    if isinstance(si, Mapping) and isinstance(si.get("n_routes"), int) and isinstance(ps, Mapping) and isinstance(ps.get("n"), int) and si.get("passed") is not None:
        # the record states the COUNT, not the cap: an Arditi (residual-slot) standup sends them as ONE-TOKEN greedy requests (measured: 3 x 11 = 33 at
        # max_tokens=1), an s-view standup at the probe set's own cap (measured: 9 x 11 = 99 inside its 609 probes at 48, indistinguishable without
        # tags). So cap 1 is declared as an ADDITIONAL greedy probe cap (harmless when the log holds none) and the count is held only when one-token
        # requests are present (`conditional`); anything left unclassified still refuses
        out["declared"]["salt_isolation"] = {"cap": 1, "expected_count": int(si["n_routes"]) * int(ps["n"]), "sampling": ps.get("sampling"), "conditional": True,
                                             "cap_rule": "one-token greedy when the standup sends them (residual-slot / Arditi engines); else at probe_max_tokens, inside the probe set's class",
                                             "source": f"{SERVE_DIR}/positive_controls.json: controls.salt_isolation.n_routes ({si['n_routes']}) x probe_set.n ({ps['n']}) greedy requests"}
    # fg1 / F-AV: the standup's own POSITIVE CONTROLS (`native_standup.run_positive_controls` via `LoraStandup.greedy_ids`) -- UNTAGGED greedy requests
    # at the standup's `probe_max_tokens` (48 by default): noop_certificate (the base and the zero adapter over the probe set), steering_bites (per lora
    # cell: n_probes on the cell + its scale-0 twin), selection_honored (its mixed-batch requests + at most (genuine + 2) x probe_set.n solo references).
    # The gsm8k smoke logged 609 of them at cap 48 beside the drivers' 40 tagged `bites` at the same cap, and the recorder's KINDS mode (fc4) read
    # every one UNCLASSIFIED against the tag's exact count. Declared as the recorder's `standup_controls` kind at the record's cap with a MAXIMUM
    # count summed from the record's own stated counts (memoisation only lowers the real number; more than the bound still refuses by name) --
    # only when the record states the cap, the greedy sampling and the controls' counts; a record that does not leaves the requests unclassified
    # (refused by name), never folded.
    pmt = out["probe_max_tokens"]
    if pmt is not None and isinstance(ps, Mapping) and ps.get("sampling") == "greedy" and ps.get("max_tokens") == pmt and isinstance(ps.get("n"), int):
        bound, basis = standup_controls_request_bound(pc)
        if bound is not None:
            out["declared"][STANDUP_CONTROLS_KIND] = {"cap": int(pmt), "expected_count": int(bound), "count_rule": "max", "sampling": "greedy", "conditional": False,
                                                      "basis": basis,
                                                      "cap_rule": "the standup's own probe_max_tokens (endpoint.json.probe_max_tokens == positive_controls.json.probe_set.max_tokens)",
                                                      "source": f"{SERVE_DIR}/positive_controls.json: the positive controls' stated request counts, summed as an upper bound"}
        else:
            out.setdefault("problems", []).append(f"{STANDUP_CONTROLS_KIND}: {basis}")
    return out


#: fg1 / F-AV: the recorder's kind for the native standup's own positive controls (record_resolved_sampler.PROBE_KIND_STANDUP_CONTROLS /
#: contracts.PROBE_KIND_STANDUP_CONTROLS, pinned equal by test); the ONLY kind `probe_declaration_from_root` declares to the recorder's kinds mode
#: (fc11g's `salt_isolation` is a cap-1 COUNT the default mode carries through `--probe-cap`; it is no table kind)
STANDUP_CONTROLS_KIND = "standup_controls"
DECLARABLE_KINDS: Tuple[str, ...] = (STANDUP_CONTROLS_KIND,)


def standup_controls_request_bound(pc: Mapping[str, Any]) -> Tuple[Optional[int], Any]:
    """An UPPER BOUND on the greedy requests the standup's positive controls fired at its probe cap, summed from `positive_controls.json`'s
    own stated counts: noop_certificate `n_prompts` x 2 (base + zero adapter); steering_bites per cell `n_probes` x 2 (the cell + its twin; a cell
    without the count -- the logprob statistic's text divergence report -- counts the probe set twice); selection_honored `n_requests` (its mixed
    batches) + (genuine vmodels + 2) x probe_set.n solo references (memoised: fewer are sent); boundary_refusal one per restricted cell. The
    controls FULL root (5 lora cells, probe set 11): 22 + 50 + 539 + 77 = 688 against 609 logged. Returns ``(bound, basis)`` or
    ``(None, why)`` when the record states no counts."""
    ps = pc.get("probe_set") if isinstance(pc, Mapping) else None
    controls = pc.get("controls") if isinstance(pc, Mapping) else None
    if not isinstance(ps, Mapping) or not isinstance(ps.get("n"), int) or not isinstance(controls, Mapping):
        return None, "positive_controls.json states no probe_set.n / controls"
    n_set = int(ps["n"])
    noop = controls.get("noop_certificate") if isinstance(controls.get("noop_certificate"), Mapping) else {}
    n_noop = 2 * int(noop["n_prompts"]) if isinstance(noop.get("n_prompts"), int) else 0
    bites = controls.get("steering_bites") if isinstance(controls.get("steering_bites"), Mapping) else {}
    cells = bites.get("cells") if isinstance(bites.get("cells"), Mapping) else {}
    n_bites = sum((2 * int(c["n_probes"])) if isinstance(c.get("n_probes"), int) else 2 * n_set for c in cells.values() if isinstance(c, Mapping))
    sel = controls.get("selection_honored") if isinstance(controls.get("selection_honored"), Mapping) else {}
    n_sel = int(sel["n_requests"]) if isinstance(sel.get("n_requests"), int) else 0
    n_solo = (len(pc.get("genuine_vmodels") or []) + 2) * n_set if n_sel else 0
    boundary = controls.get("boundary_refusal") if isinstance(controls.get("boundary_refusal"), Mapping) else {}
    n_boundary = len(boundary.get("restricted_cells")) if isinstance(boundary.get("restricted_cells"), list) else 0
    total = n_noop + n_bites + n_sel + n_solo + n_boundary
    basis = {"probe_set_n": n_set, "noop_certificate": n_noop, "steering_bites": n_bites, "selection_honored_mixed": n_sel, "selection_honored_solo_max": n_solo,
             "boundary_refusal": n_boundary,
             "rule": "the sum of the record's stated request counts -- an upper bound (solo references are memoised, so fewer are sent); count=N is a MAXIMUM to the recorder"}
    if total <= 0:
        return None, "positive_controls.json states no positive-control request counts (no control ran)"
    return total, basis


def declare_kind_argv(probe_decl: Mapping[str, Any], *, client_caps: set) -> List[str]:
    """fg1 / F-AV: the `--declare-kind` arguments the recorder's KINDS mode takes for the standup's own untagged probe kinds
    (`probe_declaration_from_root`): the table kinds only (:data:`DECLARABLE_KINDS`), never at a client cap (the recorder would fold client
    requests into the probe class). A `count_rule: max` declaration is `count=N` (a maximum: fewer passes, more is UNCLASSIFIED by name), any
    other `count=N!` (exact); a declaration without a count names the cap alone (cap + the verified greedy shape)."""
    out: List[str] = []
    caps = {int(c) for c in client_caps}
    for kind, d in sorted((probe_decl.get("declared") or {}).items()):
        if kind not in DECLARABLE_KINDS or not isinstance(d, Mapping) or int(d.get("cap", -1)) in caps:
            continue
        spec = f"{kind}:cap={int(d['cap'])}"
        n = d.get("expected_count")
        if isinstance(n, int) and not isinstance(n, bool):
            spec += f":count={n}" + ("" if d.get("count_rule") == "max" else "!")
        out += ["--declare-kind", spec]
    return out


def _expected_client_caps(exp_out: Path) -> set:
    """The CLIENT caps `expected_samplers.json` names (every child task's caps) -- a declared probe cap that is also a client cap is never declared."""
    caps: set = set()
    try:
        if exp_out.is_file():
            for t in ((json.loads(exp_out.read_text()) or {}).get("tasks") or {}).values():
                caps |= {int(c) for c in (t.get("caps") or []) if isinstance(c, int)}
    except (ValueError, AttributeError, TypeError):
        pass
    return caps


def rederive_sampler_verdict(*, pulled: Path, evidence: Path, work: Path, tree: Path, this_sha: str, row: Mapping[str, Any], python: str,
                             run_tool: Optional[Callable[..., int]] = None) -> Dict[str, Any]:
    """fc11c: the sampler verdict of a PRIOR root RE-DERIVED under THIS tree, exactly as `producer_evidence` derives it for a current row:
    this tree's `expected-samplers` helper over the PRIOR plan's row (its children's tasks -> each task's reference off this tree's authorities) into
    `<evidence>/expected_samplers.json`, then this tree's `record_resolved_sampler.py --require-wired --expect-tasks` over the pulled root's engine
    log into `<evidence>/resolved_sampler.json` -- never into the historical root, whose stored verdict (the F-AR-era recorder: UNCOMPARED on every
    two-sampler row) is recorded beside it as `historical_verdict`. Returns the producer bindings (writer / sha / tree_sha = this sha / rc) and the
    per-row `sampler` record; a helper that fails leaves the recorder on the row's frozen sampler (`--expect`), as `producer_evidence` does."""
    rec, helper = Path(tree) / PRODUCER_RECORDER_REL, Path(tree) / EXPECTED_SAMPLERS_HELPER_REL
    hist_p = pulled / lp.RESOLVED_SAMPLER_REL
    historical = None
    if hist_p.is_file():
        try:
            historical = json.loads(hist_p.read_text()).get("verdict")
        except ValueError:
            historical = "(not JSON)"
    vllm_log = pulled / SERVE_DIR / "vllm_server.log"
    out, exp_out = evidence / lp.RESOLVED_SAMPLER_REL, evidence / lp.EXPECTED_SAMPLERS_REL
    log = work / "resolved_sampler.log"
    env = {**os.environ, "PYTHONPATH": f"{tree}/src:{tree}/scripts:{tree}/scripts/sdfbase/seedrep"}

    def _run(argv: Sequence[Any]) -> Optional[int]:
        if run_tool is not None:
            return int(run_tool([str(x) for x in argv], cwd=pulled, log=log))
        return _run_tool(argv, log, env, tree, subprocess.run)
    rc_e: Optional[int] = None
    if helper.is_file():
        row_p = work / "plan_row.json"
        row_p.parent.mkdir(parents=True, exist_ok=True)
        row_p.write_text(json.dumps(dict(row), default=str))
        rc_e = _run([python, helper, "expected-samplers", "--row", row_p, "--tree", tree, "--out", exp_out])
    argv: List[Any] = [python, rec, "--vllm-log", vllm_log, "--out", out, "--require-wired"]
    # fc11g: the standup's own probe kinds, declared from the ROOT's records -- a cap that is also a client cap (a controls row's MMLU scoring at 1) is
    # never declared a probe cap (the recorder would fold client requests into the probe class)
    probe_decl = probe_declaration_from_root(pulled)
    client_caps: set = _expected_client_caps(exp_out) if rc_e == 0 else set()
    probe: Any = None
    if rc_e == 0 and exp_out.is_file():
        try:
            probe = (json.loads(exp_out.read_text()) or {}).get("probe")
        except ValueError:
            probe = None
    kinds_mode = isinstance(probe, Mapping) and str(probe.get("probe_expect")) == "kinds"
    # fc11g's one-token cap and the probe cap ride `--probe-cap` in the recorder's DEFAULT (greedy) mode only: under `kinds` (fc4: a gsm8k row)
    # `--probe-cap` would REPLACE the kind table's caps (the render probe at 512 would read as client traffic); there the standup's own kinds are
    # DECLARED instead (fg1 / F-AV: `--declare-kind standup_controls:cap=<probe_max_tokens>:count=<max>`), and the `standup_controls` cap is the
    # probe cap the default class already holds, so it is never a `--probe-cap` of its own
    probe_caps = sorted({int(d["cap"]) for k, d in probe_decl["declared"].items() if k != STANDUP_CONTROLS_KIND and int(d["cap"]) not in client_caps})
    if probe_caps and probe_decl["probe_max_tokens"] is not None and not kinds_mode:
        for cap in sorted({probe_decl["probe_max_tokens"], *probe_caps}):
            argv += ["--probe-cap", str(cap)]
    declared_argv: List[str] = []
    if rc_e == 0 and exp_out.is_file():
        argv += ["--expect-tasks", exp_out]
        if isinstance(probe, Mapping) and probe.get("probe_expect"):
            argv += ["--probe-expect", str(probe["probe_expect"])]
            for rel in probe.get("tags_rel") or []:
                argv += ["--probe-tags", str(pulled / rel)]
            for k, v in sorted((probe.get("expect") or {}).items()):
                argv += ["--expect", f"{k}={v}"]
            if kinds_mode:
                declared_argv = declare_kind_argv(probe_decl, client_caps=client_caps)
                argv += declared_argv
    else:
        for k, v in sorted((row.get("frozen_sampler") or {}).items()):
            argv += ["--expect", f"{k}={v}"]
    rc: Optional[int] = None
    why = None
    if not rec.is_file():
        why = f"{PRODUCER_RECORDER_REL} is not on the tree at {this_sha[:12]}"
    elif not vllm_log.is_file():
        why = f"the pulled root carries no {SERVE_DIR}/vllm_server.log -- nothing to re-derive the sampler verdict from"
    else:
        evidence.mkdir(parents=True, exist_ok=True)
        rc = _run(argv)
    rederived = None
    probe_rec: Dict[str, Any] = {"declared": probe_decl["declared"], "probe_caps_argv": probe_caps if probe_decl["probe_max_tokens"] is not None else [],
                                 "probe_max_tokens": probe_decl["probe_max_tokens"], "client_caps": sorted(client_caps), "n_probe_by_cap": None, "n_unclassified": None,
                                 # fg1 / F-AV: what the KINDS-mode argv declared (the kinds the certificate holds per kind), the recorder's per-kind counts and verdicts
                                 "declared_argv": declared_argv, "declared_argv_kinds": sorted(a.split(":", 1)[0] for a in declared_argv if not a.startswith("--")),
                                 "n_by_kind": None, "verdict_by_kind": None, "problems": list(probe_decl.get("problems") or [])}
    if out.is_file():
        try:
            fresh_doc = json.loads(out.read_text())
            rederived = fresh_doc.get("verdict")
            probe_cls = (fresh_doc.get("classes") or {}).get("probe") or {}
            probe_rec["n_probe_by_cap"] = {str(x.get("max_tokens")): int(x.get("n")) for x in (probe_cls.get("distinct_max_tokens") or []) if isinstance(x, Mapping)}
            probe_rec["n_unclassified"] = fresh_doc.get("n_unclassified_requests")
            probe_rec["n_by_kind"] = fresh_doc.get("n_probe_requests_by_kind") if isinstance(fresh_doc.get("n_probe_requests_by_kind"), Mapping) else None
            probe_rec["verdict_by_kind"] = fresh_doc.get("probe_verdict_by_kind") if isinstance(fresh_doc.get("probe_verdict_by_kind"), Mapping) else None
        except (ValueError, AttributeError, TypeError):
            rederived = "(not JSON)"
    bindings = {lp.RESOLVED_SAMPLER_REL: _bound_file(out, writer=PRODUCER_RECORDER_REL, W=this_sha, rc=rc, why=why),
                lp.EXPECTED_SAMPLERS_REL: _bound_file(exp_out, writer=EXPECTED_SAMPLERS_HELPER_REL, W=this_sha, rc=rc_e,
                                                      why=(f"{EXPECTED_SAMPLERS_HELPER_REL} is not on the tree at {this_sha[:12]}" if not helper.is_file() else None))}
    record = {"historical_verdict": historical, "rederived_verdict": rederived, "rederived_by": this_sha, "recorder_rc": rc, "expected_samplers_rc": rc_e,
              "vllm_log": str(vllm_log), "expect_tasks": bool(rc_e == 0 and exp_out.is_file()), "why": why, "basis": lp.SMOKE_PASS_SAMPLER_BASIS,
              "probe": probe_rec}
    return {"bindings": bindings, "record": record}


def pulled_root_stores(pulled: Path, *, row: Mapping[str, Any], tree: Path = LOCAL_REPO) -> Dict[str, Any]:
    """fc11f / fc11i: the stores a PULLED prior root actually holds, by IDENTITY (table leg, seed, arm, store path) -- `audit_common.inventory_stores`
    (the audit's own reader per table leg; every candidate store file no leg claims is a NAMED problem). A SMOKE row ran with `--smoke-max-children N`
    -- its root holds the stores of the arms that ran (the smoke row: the `none` arm's five children), never the plan row's full arm list -- so this
    inventory is the scope the re-audit's verdicts are held to for a prior SMOKE row, one bound verdict per identity; a carried PRODUCTION row keeps
    the full-arm requirement (`seedrep_carry_over.complete_verdict`)."""
    ac = lp.load_audit_common(Path(tree))          # fc11m: THIS tree's reader by explicit path; a foreign / cached copy refused by name
    legs = determinism_table_legs(str(row["leg"]), stage=row.get("stage", lp.STAGE_BASE))
    inv = ac.inventory_stores(Path(pulled), legs, petri_behaviours=[str(b) for b in (row.get("behaviours") or [])])
    problems: Dict[str, Any] = dict(inv["problems"])
    if inv["unclassified"]:
        problems["unclassified"] = list(inv["unclassified"])
    return {"stores_present": list(inv["stores"]), "stores_problems": problems, "table_legs": list(legs)}


def scope_row_to_stores(row: Mapping[str, Any], stores_present: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    """The row with its children narrowed to the arms whose stores the pulled root holds (a child's cell, or its harness id, equals a present store's
    arm) -- for the sampler step's task set; when a present store maps to no plan child the row is left whole (the audit's binding decides there)."""
    arms = {str(e.get("arm")) for e in (stores_present or []) if isinstance(e, Mapping)} if isinstance(stores_present, list) \
        else {str(a) for legs in (stores_present or {}).values() for a in legs}
    if not arms:
        return dict(row)
    def _matches(k: Mapping[str, Any]) -> bool:
        cell = str(k.get("cell") or k.get("arm"))
        return cell in arms or lp.harness_id(cell) in arms or str(k.get("arm")) in arms
    kids = [k for k in (row.get("children") or []) if _matches(k)]
    covered = {str(k.get("cell") or k.get("arm")) for k in kids} | {lp.harness_id(str(k.get("cell") or k.get("arm"))) for k in kids} | {str(k.get("arm")) for k in kids}
    if not kids or not arms <= covered:
        return dict(row)
    return {**dict(row), "children": kids}


def reaudit_carried_root(*, prev_row: Mapping[str, Any], this_row: Mapping[str, Any], prior: Mapping[str, Any], aid: str, root: str, this_plan: Mapping[str, Any],
                         rs: Path, this_sha: str, tree: Path = LOCAL_REPO, run_get: Optional[Callable[[str, Path], None]] = None,
                         put_file: Optional[Callable[[str, bytes], None]] = None, run_tool: Optional[Callable[..., int]] = None,
                         python: Optional[str] = None) -> Dict[str, Any]:
    """fj7 (clause (c) under THIS tree's readers): re-run THIS tree's `request_evidence_audit.py` over a CARRIED root's stores.

    1. pull the historical attempt root (immutable; read only) under `RS/carry_over/reaudit/<sid>/<aid>/pulled/<volume-relative root>`.
    2. run the audit from the pull dir with a RELATIVE root (so every verdict's `store_dir` spells the historical root, which the certificate
       holds it to), the SAME argv the app's in-container audit uses (`--roots`, `--out`, `--legs`, the Petri behaviours; no `--plan` -- the
       leg contract decides the ledger topology) plus `--audited-at <this sha>` (the verdicts' W is the auditor's tree; the store's tree
       stays under `store_W`), into `<work>/evidence/audits/`.
    3. copy the standup's `exp/serve/endpoint.json` beside it (the binding's vmodel cross-check reads it) and bind every plan child's verdict
       through the writer's index (`bind_audit_verdicts`, F-AO) -- `producer_evidence` keyed by the plan's rel with `bound_rel`, sha, writer,
       tree_sha = this sha, rc.
    4. upload the evidence dir to the carried entry's own volume root (:func:`carried_evidence_root_rel`).
    Returns the re-audit record the evaluator threads into clause (c) and the carry entry (`reaudited_by`)."""
    import shutil
    sid = str(this_row["standup_id"])
    work = Path(rs) / "carry_over" / "reaudit" / sid / aid
    pull_dir = work / "pulled"
    rel_root = root[len(VOL_PATH) + 1:] if root.startswith(VOL_PATH + "/") else root.lstrip("/")
    dest_parent = pull_dir / Path(rel_root).parent
    dest_parent.mkdir(parents=True, exist_ok=True)
    (run_get or volume_get)(root, dest_parent)
    pulled = pull_dir / rel_root
    if not (pulled / "ATTEMPT.json").is_file():
        raise LaunchRefused(f"[seedrep] re-audit: the pulled root {pulled} carries no ATTEMPT.json -- the historical attempt root was not retrieved")
    evidence = work / "evidence"
    out_dir = evidence / lp.AUDIT_OUT_REL
    if out_dir.exists():
        shutil.rmtree(out_dir)                       # the tool refuses an existing request_log tree; every evaluation re-audits afresh
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    legs = determinism_table_legs(this_row["leg"], stage=this_row["stage"])
    # fc11d: the sampler verdict is re-derived FIRST (this tree's recorder over the pulled engine log, into the evidence dir -- `rederive_sampler_verdict`)
    # and handed to the audit as an EXPLICIT input (`--resolved-sampler`): the audit used to read the historical root's resolved_sampler.json (UNCOMPARED
    # on every two-sampler row) and embed it as `resolved_sampler_verdict` in every verdict, which the certificate then refused
    # fc11f: what the pulled root actually holds, per table leg -- the scope a prior SMOKE row's verdicts are held to (never the plan row's full arm list)
    stores = pulled_root_stores(pulled, row=this_row, tree=Path(tree))
    sampler_row = scope_row_to_stores(prev_row, stores["stores_present"])
    sampler = rederive_sampler_verdict(pulled=pulled, evidence=evidence, work=work, tree=Path(tree), this_sha=this_sha, row=sampler_row, python=python or sys.executable,
                                       run_tool=run_tool)
    fresh_rs = evidence / lp.RESOLVED_SAMPLER_REL
    argv = [python or sys.executable, str(Path(tree) / PRODUCER_AUDIT_REL), "--roots", rel_root, "--out", str(out_dir), "--legs", ",".join(legs),
            "--audited-at", this_sha] + (["--resolved-sampler", str(fresh_rs)] if fresh_rs.is_file() else [])
    if this_row["leg"] == "petri" and this_row.get("behaviours"):
        argv += ["--petri-behaviours", ",".join(str(b) for b in this_row["behaviours"])]
    log = work / "request_evidence_audit.log"
    if run_tool is not None:
        rc = int(run_tool(argv, cwd=pull_dir, log=log))
    else:
        with open(log, "w") as fh:
            rc = subprocess.run(argv, cwd=str(pull_dir), stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": f"{tree}/src:{tree}/scripts:{tree}/scripts/sdfbase/seedrep"}).returncode
    ep_src = pulled / SERVE_DIR / "endpoint.json"
    if ep_src.is_file():
        (evidence / SERVE_DIR).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ep_src, evidence / SERVE_DIR / "endpoint.json")
    bindings = bind_audit_verdicts(evidence, row=this_row, W=this_sha, rc_a=rc, aud_present=(Path(tree) / PRODUCER_AUDIT_REL).is_file())
    bindings.update(sampler["bindings"])          # fc11c: the re-derived sampler verdict's bindings beside the audit's verdicts
    ev_rel = carried_evidence_root_rel(this_plan["W7"], this_plan["wave_uuid"], sid, aid)
    if put_file is None:
        import seedrep_ladder as L  # noqa: E402
        vol = L.RemoteVolume.from_name(lp.MODAL_VOLUME_NAME)
        put_file = vol.write_bytes
    files = sorted(p for p in evidence.rglob("*") if p.is_file())
    for p in files:
        put_file(f"{ev_rel}/{p.relative_to(evidence).as_posix()}", p.read_bytes())
    idx = out_dir / AUDIT_INDEX_NAME
    return {"reaudited_by": this_sha, "evidence_root": f"{VOL_PATH}/{ev_rel}", "local_evidence_root": str(evidence), "store_root": root, "producer_evidence": bindings,
            "rc": rc, "argv": argv, "tool": PRODUCER_AUDIT_REL, "n_files": len(files), "index_sha256": (sha256_file(idx) if idx.is_file() else None),
            "log": str(log), "sampler": sampler["record"],
            # fc11f: the pulled root's stores per table leg (+ legs whose stores refused to read) and the plan row's arms -- the scope rule's inputs
            "stores_present": stores["stores_present"], "stores_problems": stores["stores_problems"], "table_legs": stores["table_legs"],
            "pulled_root": str(pulled),          # fc11i: where the identity inventory was taken (re-enumerated by the validator when still on this machine)
            "arms_planned": sorted({str(k.get("cell") or k.get("arm")) for k in (this_row.get("children") or [])})}


def local_read_text(path: str) -> Optional[str]:
    """A grading-host-local file's text, None when absent -- `seedrep_carry_over.default_local_read_text` (the ONE spelling; kept under this name for callers)."""
    import seedrep_carry_over as co  # noqa: E402
    return co.default_local_read_text(path)


def load_criterion3_waivers(spec: str) -> List[Dict[str, Any]]:
    """`--criterion3-waiver <json | file>`: one waiver document or a list of them, each validated by `seedrep_carry_over.criterion3_waiver_problems`
    -- a malformed document refuses the launch step, never silently widens or narrows a scope."""
    import seedrep_carry_over as co  # noqa: E402
    if not spec or not spec.strip():
        return []
    text = spec
    if not spec.lstrip().startswith(("{", "[")):
        p = Path(spec).expanduser()
        if not p.is_file():
            raise LaunchRefused(f"[seedrep] --criterion3-waiver {spec!r} is neither a JSON document nor a file")
        text = p.read_text()
    doc = json.loads(text)
    docs = doc if isinstance(doc, list) else [doc]
    for i, w in enumerate(docs):
        probs = co.criterion3_waiver_problems(w, f"waiver[{i}]")
        if probs:
            raise LaunchRefused(f"[seedrep] --criterion3-waiver: {probs}")
    return [dict(w) for w in docs]


def carry_over_evaluate(plan_d: Mapping[str, Any], *, rs: Path, sha: str, sources: Sequence[str],
                        accepted_get_for_dict: Optional[Callable[[str], Callable[[str], Any]]] = None, read_text: Optional[Callable[[str], Optional[str]]] = None,
                        git: Optional[Callable[..., Any]] = None, accepted_put: Optional[Callable[..., bool]] = None, accepted_get: Optional[Callable[[str], Any]] = None,
                        readiness: Optional[Callable[..., List[str]]] = None, replay: Optional[Callable[..., Dict[str, Any]]] = None,
                        petri_proof: Optional[Callable[..., Dict[str, Any]]] = None, criterion3_waivers: Sequence[Mapping[str, Any]] = (),
                        petri_read_text: Optional[Callable[[str], Optional[str]]] = None, petri_home: Optional[Path] = None,
                        reaudit: Union[None, bool, Callable[..., Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The grading host `--carry-over <prior>[,<prior>]` step (fj2), run on the FINAL plan BEFORE certify: every production row judged against the
    prior trees' candidates (`seedrep_carry_over.evaluate_carry_over`: the five clauses, the git byte-identity check on THIS repo, this tree's
    readiness contract), the record written to `RS/carry_over.json` (the certificate ingests it), and every CARRIED attempt seeded into THIS
    wave's accepted Dict under THIS wave's key with its provenance. Every Modal / git read is the app's real one unless injected (tests)."""
    import seedrep_carry_over as co  # noqa: E402
    rs = Path(rs)
    priors = [co.load_prior_source(Path(p)) for p in sources]
    git = git or co.default_git(LOCAL_REPO)
    if replay is None:
        # the ladder-queue REPLAY over the live volume + THIS wave's accepted Dict (seedrep_ladder's own adapters; the app's root spelling)
        def replay(**kw: Any) -> Dict[str, Any]:
            import seedrep_ladder as L  # noqa: E402
            vol = L.RemoteVolume.from_name(lp.MODAL_VOLUME_NAME)
            acc = L.ModalDictKV.from_name(plan_d["dicts"]["accepted"], create_if_missing=True)
            return co.replay_ladder_queue(L=L, vol=vol, accepted_kv=acc, default_root_of=lp.producer_attempt_root_of(plan_d["results_root"]), **kw)
    if petri_proof is None:
        # fj3: the sidecar proof over the grading host's local Petri stores the completion bundle names; the judge chain is n/a in a prior tree that
        # has none. fj5: EVERY local input is read through the `read_text` the evaluator hands in -- its RecordingReader over
        # `petri_read_text` (default the grading host's filesystem) -- so results.json and both refusal ledgers (absent ones as absent) enter the
        # entry's evidence_inventory_local and are re-read at certification; nothing is read around the reader
        def petri_proof(*, prev_row: Mapping[str, Any], prior: Mapping[str, Any], aid: str, evidence: Mapping[str, Any],
                        read_text: Callable[[str], Optional[str]]) -> Dict[str, Any]:
            rc, text = git(["show", f"{prior['W']}:src/sisteer/capeval/petri/auditor_fallback.py"])
            judge_in_prior = rc == 0 and "judge_fallback" in text
            return co.petri_sidecar_proof(co.petri_cell_dirs(evidence, home=petri_home), read_text, judge_chain_in_prior=judge_in_prior)

    if reaudit is False:
        # an EXPLICIT opt-out (tests / diagnostics only): the historical verdicts are read as they are -- under fp3's strict arm_binding check
        # every pre-F-AN row then refuses and is HELD (carry_refused_hold), never spawned; the default (None) is the real re-audit
        print("[seedrep] carry-over: re-audit DISABLED by the caller -- historical verdicts read as they are (diagnostics only)", flush=True)
        reaudit = None
    elif reaudit is None:
        # fj7: THIS tree's request-evidence audit over every carried root (older verdicts predate F-AN's arm_binding; the fitted cells have
        # none) -- into the carried entry's own evidence root, bound through the writer's index
        def reaudit(**kw: Any) -> Dict[str, Any]:
            return reaudit_carried_root(rs=rs, this_sha=sha, **kw)

    def _dict_get(name: str) -> Callable[[str], Any]:
        try:
            d = modal.Dict.from_name(name, create_if_missing=False)
        except Exception as e:  # noqa: BLE001 -- a prior wave whose Dict is gone has no accepted attempt to carry (recorded by name per row)
            print(f"[seedrep] carry-over: accepted Dict {name} unreachable ({type(e).__name__}) -- no attempt of that wave can be carried", flush=True)
            return lambda key: None
        return d.get
    rec = co.evaluate_carry_over(this_plan=plan_d, this_sha=sha, priors=priors, read_text=read_text or certificate_read_text(rs),
                                 accepted_get_for_dict=accepted_get_for_dict or _dict_get, git=git, readiness_problems=readiness or readiness_problems,
                                 replay=replay, petri_proof=petri_proof, criterion3_waivers=list(criterion3_waivers), local_read_text=petri_read_text, reaudit=reaudit,
                                 generation_proof_doc=co.rs_generation_proof_reader(rs))          # fj30: RS/carry_over/proofs/<sid>__<aid>.json
    path = co.write_carry_over(rs, rec)
    if accepted_put is None or accepted_get is None:
        acc = modal.Dict.from_name(plan_d["dicts"]["accepted"], create_if_missing=True)
        accepted_put, accepted_get = acc.put, acc.get
    seeded = co.seed_accepted(rec, accepted_put, accepted_get)
    for sid, e in sorted(rec["carried"].items()):
        cf = e["carried_from"]
        rp = ((e.get("clauses") or {}).get("ladder_queue") or {}).get("replay") or {}
        _legs_log(rs, f"CARRIED {sid} attempt={e['attempt_id']} from {cf['W7']}/{cf['source']} prior_tree={cf['tree_sha'][:12]} key={e['accepted_key']} (D112; seeded)"
                      + (f" reaudited_by={str(e.get('reaudited_by'))[:12]}" if e.get("reaudited_by") else "")
                      + (f" replay={rp.get('n_records')} records queue_closed={rp.get('queue_closed')}" if rp else "")
                      + (f" criterion3_waiver={e['criterion3_waiver'].get('decision')}" if e.get("criterion3_waiver") else ""))
    for sid, e in sorted(rec["refused"].items()):
        _legs_log(rs, f"CARRY-REFUSED {sid} clause={e.get('clause')} why={str(e.get('why'))[:240]}")
    return {"record": str(path), "n_carried": rec["n_carried"], "n_refused": rec["n_refused"], "carried": sorted(rec["carried"]), "seeded": seeded,
            "refused": {sid: {"clause": e.get("clause"), "why": e.get("why")} for sid, e in rec["refused"].items()}}


def _probe_free(lock_path: Path) -> bool:
    """True when nobody holds ``lock_path`` (the probe takes and releases it at once; never held)."""
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return True
        except OSError:
            return False
    finally:
        os.close(fd)


class _IntentLock:
    """The intent file's read-modify-write lock (`RS/production_launch.lock`): the launcher and the steward's fan-out pass both write it."""

    def __init__(self, rs: Path) -> None:
        self.path = Path(rs) / lp.PRODUCTION_LAUNCH_LOCK
        self.fd = -1

    def __enter__(self) -> "_IntentLock":
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)


def launch_production(plan_d: Mapping[str, Any], *, plan_path: Path, rs: Path, app_name: str, sha: str, pools: Optional[Mapping[str, str]],
                      pool_urls_path: Optional[Path], cert_p: Path, go: str = "", legs: str = "", steward_deadline_h: float = 36.0,
                      steward_launch: Optional[Callable[..., Dict[str, Any]]] = None, wait_started: Optional[Callable[..., Any]] = None,
                      select: Callable[..., Any] = None, accepted_get_for: Optional[Callable[[str], Callable[[str], Any]]] = None,
                      accepted_get_for_dict: Optional[Callable[[str], Callable[[str], Any]]] = None,
                      new_token: Optional[Callable[[], str]] = None, carry_over: Sequence[Any] = ()) -> Dict[str, Any]:
    """The production launch: the launcher spawns NOTHING. In this order and no other: (1) the certificate (the
    plan's bytes bound, every derived record re-derived, the accepted attempts read from the LIVE Dicts); (2) the launcher lock `RS/launch.lock`;
    (3) the selection validated (`select_launch_rows`: FINAL plan, executable rung modes, pools healthy); (4) the DURABLE FAN-OUT INTENT
    `RS/production_launch.json` -- new (certificate sha, plan digest + file sha, wave, the full required-key set, per-row spawn state) or the one
    already there for THIS certificate (rows not yet in it are added; rows already in it are never spawned twice; an intent of another certificate
    or plan refuses -- never a second wave); (5) the wave's `RS/steward.lock` probed: held = a steward owns the wave = it will fan out any rows added
    (nothing else to do; with no new rows: "production already launched" is refused); free = the steward unit is launched with a fresh START TOKEN
    written into the intent FIRST and its `STEWARD_STARTED {start_token}` awaited (not launched / never reported -> LaunchRefused; the intent
    stays, a later --go resumes it). The STEWARD performs and completes every spawn from the intent (`--fanout`)."""
    import seedrep_steward as st  # noqa: E402
    steward_launch = steward_launch or st.launch_steward
    wait_started = wait_started or st.wait_started
    select = select or select_launch_rows
    new_token = new_token or st.new_start_token
    rs = Path(rs)
    rs.mkdir(parents=True, exist_ok=True)
    cert = require_certificate(plan_d, plan_path=plan_path, rs=rs, cert_p=cert_p, accepted_get_for=accepted_get_for, accepted_get_for_dict=accepted_get_for_dict)
    # fj2: the CARRIED rows come from the CERTIFICATE's carry-over block (evaluated by `--carry-over` BEFORE certify, re-validated by
    # require_certificate: provenance, clauses, roots still COMPLETE, this wave's live Dict naming them); `--carry-over` at --go only cross-checks
    # that the certified record is the one on disk for exactly the priors named -- never a second evaluation past the certificate
    import seedrep_carry_over as co  # noqa: E402
    co_rec = cert.get("carry_over") if isinstance(cert, Mapping) else None
    carried = co.carried_standup_ids(co_rec)
    carry_holds = co.carry_refused_holds(co_rec)                    # fj7 (B): refused carries over COMPLETED prior attempts are HELD, never spawned
    for sid, h in sorted(carry_holds.items()):
        _legs_log(rs, f"HOLD {co.CARRY_REFUSED_HOLD} {sid} attempt={h.get('attempt_id')} prior={h.get('prior')} clause={h.get('clause')} why={str(h.get('why'))[:200]}")
    if carry_over:
        if not isinstance(co_rec, Mapping):
            raise LaunchRefused("[seedrep] production launch REFUSED: --carry-over named prior trees but the certificate carries no carry-over block -- run "
                                "`--carry-over <prior>[,<prior>] --plan <FINAL>` BEFORE certify (the certificate lists every carried attempt), then certify, then --go")
        want = sorted(str(Path(p)) for p in carry_over)
        have = sorted(str(Path(p["path"])) for p in (co_rec.get("priors") or []) if isinstance(p, Mapping) and p.get("path"))
        if want != have:
            raise LaunchRefused(f"[seedrep] production launch REFUSED: --carry-over names {want}, the certified carry-over record was evaluated over {have}")
        disk = co.read_carry_over(rs)
        if disk is None or co.record_digest(disk) != co.record_digest(co_rec):
            raise LaunchRefused(f"[seedrep] production launch REFUSED: {rs / co.CARRY_OVER_FILE} is not the certified carry-over record (re-evaluated after certify?)")
    lock_fd = os.open(rs / LAUNCH_LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise LaunchRefused(f"[seedrep] production launch REFUSED: another launcher holds {rs / LAUNCH_LOCK_FILE} for this wave")
        picked, _identity = select(plan_d, go=go, legs=legs, pool_urls=pools, **({"carried": carried} if carried else {}), **({"carry_holds": carry_holds} if carry_holds else {}))
        certified_launch_guard(cert, [r["standup_id"] for r in picked], where="production launch")     # fc9: the selection vs the certified launch set
        for sid, cf in sorted(carried.items()):
            _legs_log(rs, f"CARRIED {sid} attempt={cf.get('attempt_id')} from {cf.get('W7')}/{cf.get('source')} prior_tree={str(cf.get('tree_sha'))[:12]} key={cf.get('accepted_key')} (D112; not spawned)")
        intent_p = rs / lp.PRODUCTION_LAUNCH_FILE
        steward_lock = rs / st.LOCK_FILE
        with _IntentLock(rs):
            intent = lp.read_production_intent(intent_p)
            if intent is None:
                intent = lp.production_intent_new(plan=plan_d, certificate=cert, rows=picked, selection={"go": go, "legs": legs})
                mode, added = "new", sorted(r["standup_id"] for r in picked)
            else:
                probs = lp.production_intent_problems(intent, plan=plan_d, certificate=cert)
                if probs:
                    raise LaunchRefused(f"[seedrep] production launch REFUSED: {intent_p} is not this certified plan's fan-out intent -- never a second wave: " + "; ".join(probs))
                added = lp.production_intent_extend(intent, picked, selection={"go": go, "legs": legs})
                owned = not _probe_free(steward_lock)
                unspawned = lp.production_intent_unspawned(intent)
                if not added and (owned or not unspawned):
                    n_all = len(intent["rows"])
                    raise LaunchRefused(f"[seedrep] production launch REFUSED: production is already launched for certificate {cert['certificate_sha256'][:12]} "
                                        f"({intent_p}: {n_all - len(unspawned)}/{n_all} rows spawned{'; the steward owns the wave' if owned else ''}); "
                                        "re-attempts go through the steward, never a second --go")
                mode = "extended" if added else "resumed"
            owned = not _probe_free(steward_lock)
            if owned:
                if mode == "new":
                    raise LaunchRefused(f"[seedrep] production launch REFUSED: a steward already owns wave {plan_d['wave_uuid']} ({steward_lock}) with no fan-out intent on record")
                lp.write_production_intent(intent_p, intent)
                _legs_log(rs, f"INTENT {mode} rows_added={len(added)} steward=owned unit={(intent.get('steward') or {}).get('unit')} (the running steward fans them out)")
                return {"mode": mode, "intent": str(intent_p), "added": added, "steward": None, "started": None, "spawned_by_launcher": []}
            token = new_token()
            unit = st.unit_name(plan_d["W7"], plan_d["wave_uuid"])
            lp.production_intent_set_steward(intent, unit=unit, start_token=token)
            lp.write_production_intent(intent_p, intent)                 # the durable claim, BEFORE the steward and before any spawn
        _legs_log(rs, f"INTENT {mode} rows={len(intent['rows'])} added={len(added)} unspawned={len(lp.production_intent_unspawned(intent))} token={token} -> {intent_p}")
        rep = steward_launch(rs=rs, plan=plan_d, plan_path=Path(plan_path), certificate=cert_p, app_name=app_name, sha=sha, tree=LOCAL_REPO,
                             app_script=Path(__file__).resolve(), pool_urls=pool_urls_path, deadline_h=float(steward_deadline_h), start_token=token)
        _legs_log(rs, f"STEWARD {'launched' if rep.get('launched') else 'NOT launched'} unit={rep.get('unit')} token={token} log={rep.get('log')}" + ("" if rep.get("launched") else f" why={rep.get('why')}"))
        if not rep.get("launched"):
            raise LaunchRefused(f"[seedrep] production launch REFUSED: the steward did not start ({rep.get('why')}) -- nothing was spawned; the intent {intent_p} stays "
                                "(a later --go resumes it); production never runs unattended")
        started = wait_started(Path(rep["log"]), token=token, timeout_s=STEWARD_START_TIMEOUT_S)
        if not started:
            raise LaunchRefused(f"[seedrep] production launch REFUSED: the steward unit {rep.get('unit')} never reported STEWARD_STARTED for token {token} within "
                                f"{STEWARD_START_TIMEOUT_S:.0f} s (log {rep.get('log')}) -- nothing was spawned; the intent {intent_p} stays")
        _legs_log(rs, f"STEWARD STARTED unit={rep.get('unit')} pid={started.get('pid')} token={token} required_keys={started.get('required_keys')} -- the steward fans out {len(lp.production_intent_unspawned(intent))} row(s)")
        return {"mode": mode, "intent": str(intent_p), "added": added, "steward": rep, "started": started, "spawned_by_launcher": []}
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def reconcile_fanout_marks(rs: Path, plan_d: Mapping[str, Any], intent: Dict[str, Any], ledger: Any) -> Dict[str, List[str]]:
    """For every claimed row with a recorded call: read the container's executing mark (`fanout|<row>|<nonce>|executing`, the durable call list);
    record the EXECUTING call id in the intent and -- when it is not the recorded one (the first spawn ran, the re-spawn exited FANOUT_DUPLICATE)
    -- append it to RS/calls.jsonl so the watcher and the steward follow the container that runs. Mutates ``intent`` (the caller writes it)."""
    observed, adopted = [], []
    for sid, r in (intent.get("rows") or {}).items():
        if not (isinstance(r, Mapping) and r.get("claim")):
            continue
        mark = fanout_mark(ledger, sid, r["claim"])
        if not mark or not mark.get("call_id"):
            continue
        winner = str(mark["call_id"])
        # the WINNER is appended to RS/calls.jsonl (durably) BEFORE the row counts as reconciled -- the steward and the watcher read
        # calls.jsonl, so an execution known only from its mark would be unmonitored; done wherever the mark is discovered, whatever was recorded
        if _ensure_call_recorded(rs, plan_d, sid, r, winner, mark):
            adopted.append(sid)
        if r.get("executing_call_id") != winner:
            r["executing_call_id"] = winner
            observed.append(sid)
        if not r.get("call_id"):
            r.update(call_id=winner, spawned_utc=r.get("spawned_utc") or utc(), spawned_by=r.get("spawned_by") or "adopted-from-mark")
    return {"observed": observed, "adopted": adopted}


def _ensure_call_recorded(rs: Path, plan_d: Mapping[str, Any], sid: str, r: Mapping[str, Any], winner: str, mark: Mapping[str, Any]) -> bool:
    """Append the executing container's call to RS/calls.jsonl unless it is there already (the steward's `wave_calls` is calls.jsonl). Returns True
    when a record was appended."""
    calls = {c["call_id"]: c for c in wave_calls(rs, plan_d)}
    if winner in calls:
        return False
    recorded = r.get("call_id")
    base = dict(calls.get(recorded) or {"app": None, "W": plan_d["W"], "wave_uuid": plan_d["wave_uuid"], "standup_id": sid, "leg": r["leg"], "engine": r["engine"],
                                        "stage": r["stage"], "shard": r["shard"], "control_key": r["control_key"], "accepted_key": r["accepted_key"], "kind": r["kind"],
                                        "smoke": None, "execution_wave_uuid": plan_d["wave_uuid"], "production_wave_uuid": plan_d["wave_uuid"],
                                        "launch_plan_sha": plan_d.get("launch_plan_sha")})
    _record_call(rs, {**base, "call_id": winner, "spawned_at": utc(), "adopted_from_mark": recorded, "executing_attempt": mark.get("attempt_id"),
                      "fanout_claim": {"standup_id": sid, "nonce": r["claim"]}})
    _legs_log(rs, f"FANOUT {sid} executing call={winner} adopted from the container's mark into calls.jsonl (recorded {recorded!r})")
    return True


def reconcile_fanout_marks_file(rs: Path, plan_d: Mapping[str, Any], intent_p: Path, ledger: Any) -> Dict[str, Any]:
    """The steward's per-tick reconciliation: under the intent lock, every claimed row's executing mark is read, its winner
    appended to calls.jsonl and recorded in the intent; runs EVERY tick while any claim has no observed winner -- never an arbitrary pass count."""
    with _IntentLock(rs):
        intent = lp.read_production_intent(intent_p)
        if intent is None:
            return {"observed": [], "adopted": [], "pending": []}
        out = reconcile_fanout_marks(rs, plan_d, intent, ledger)
        lp.write_production_intent(intent_p, intent)
        out["pending"] = lp.production_intent_marks_pending(intent) + lp.production_intent_claimed_unspawned(intent)
        return out


def run_fanout(plan_d: Mapping[str, Any], *, plan_path: Path, rs: Path, app_name: str, pools: Optional[Mapping[str, str]], cert_p: Path, start_token: str,
               spawn: Callable[..., str] = None, accepted_get_for: Optional[Callable[[str], Callable[[str], Any]]] = None, by: str = "steward",
               accepted_get_for_dict: Optional[Callable[[str], Callable[[str], Any]]] = None,
               ledger: Any = None, new_nonce: Optional[Callable[[], str]] = None) -> Dict[str, Any]:
    """The steward's fan-out pass (`--fanout --start-token T`): under the certificate, the intent
    `RS/production_launch.json` (this certified plan's; its `steward.start_token` == T) and ONLY while a steward owns the wave, every row with no
    call yet is executed EXACTLY ONCE across the Modal spawn boundary: (1) a DURABLE per-row claim (a nonce written into the intent) BEFORE
    `.spawn()`; (2) the spawn carries the claim and the container claims `fanout|<row>|<nonce>|executing` at entry (a duplicate exits
    FANOUT_DUPLICATE); (3) the call id recorded. A row found CLAIMED with no call (a pass died between (1) and (3)) is 'spawn outcome unknown':
    the container's executing mark is read first (adopt its call id), then RS/calls.jsonl (adopt), and only then is the row re-spawned under the
    SAME nonce -- whichever container claims first runs, the other exits. Every claimed row's mark is reconciled afterwards. (Modal offers no
    FunctionCall listing by tag; the executing mark in the wave's Dict is the call list this app owns.)"""
    import seedrep_steward as st  # noqa: E402
    spawn = spawn or spawn_row
    new_nonce = new_nonce or lp.new_claim_nonce
    rs = Path(rs)
    cert = require_certificate(plan_d, plan_path=plan_path, rs=rs, cert_p=cert_p, accepted_get_for=accepted_get_for, accepted_get_for_dict=accepted_get_for_dict)
    intent_p = rs / lp.PRODUCTION_LAUNCH_FILE
    if _probe_free(rs / st.LOCK_FILE):
        raise LaunchRefused(f"[seedrep] fan-out REFUSED: no steward owns wave {plan_d['wave_uuid']} ({rs / st.LOCK_FILE} is free) -- the fan-out runs under the steward only")
    if ledger is None:
        ledger = modal.Dict.from_name(plan_d["dicts"]["reattempts"], create_if_missing=True)
    spawned, adopted, recovered = [], [], []
    while True:
        with _IntentLock(rs):
            intent = lp.read_production_intent(intent_p)
            if intent is None:
                raise LaunchRefused(f"[seedrep] fan-out REFUSED: no fan-out intent at {intent_p} (the launcher writes it before the steward starts)")
            probs = lp.production_intent_problems(intent, plan=plan_d, certificate=cert)
            if probs:
                raise LaunchRefused("[seedrep] fan-out REFUSED: the intent is not this certified plan's: " + "; ".join(probs))
            certified_launch_guard(cert, list(intent["rows"]), where="fan-out")                    # fc9b: the intent's rows vs the certified launch set
            if not start_token or (intent.get("steward") or {}).get("start_token") != start_token:
                raise LaunchRefused(f"[seedrep] fan-out REFUSED: start token {start_token!r} is not the intent's {(intent.get('steward') or {}).get('start_token')!r} -- "
                                    "the fan-out runs under the steward this intent names")
            due = lp.production_intent_unspawned(intent)
            if not due:
                marks = reconcile_fanout_marks(rs, plan_d, intent, ledger)
                lp.write_production_intent(intent_p, intent)
                break
            sid = due[0]
            row = lp.plan_row_by_id(plan_d, sid)
            r = intent["rows"][sid]
            how = "spawned"
            if r.get("claim"):
                # the spawn outcome is UNKNOWN: reconcile before any second spawn
                nonce = r["claim"]
                mark = fanout_mark(ledger, sid, nonce)
                prior = [c for c in wave_calls(rs, plan_d) if c.get("standup_id") == sid and (c.get("fanout_claim") or {}).get("nonce") == nonce]
                if mark and mark.get("call_id"):
                    cid, how = str(mark["call_id"]), "adopted-from-mark"
                    _ensure_call_recorded(rs, plan_d, sid, r, cid, mark)          # delta 4 (3): the winner is in calls.jsonl BEFORE the row counts as reconciled
                    r["executing_call_id"] = cid
                    adopted.append(sid)
                elif prior:
                    cid, how = prior[0]["call_id"], "adopted-from-calls"
                    adopted.append(sid)
                else:
                    certified_launch_guard(cert, [sid], where="fan-out re-spawn")                  # fc9b: at the spawn boundary itself
                    cid, how = spawn(row, plan_d, app_name, rs, pools, fanout_claim={"standup_id": sid, "nonce": nonce}), "re-spawned-under-claim"
                    recovered.append(sid)
            else:
                certified_launch_guard(cert, [sid], where="fan-out spawn")                         # fc9b: at the spawn boundary itself
                nonce = new_nonce()
                lp.production_intent_claim(intent, sid, nonce)
                lp.write_production_intent(intent_p, intent)                 # the DURABLE claim, before .spawn()
                cid = spawn(row, plan_d, app_name, rs, pools, fanout_claim={"standup_id": sid, "nonce": nonce})
                spawned.append(sid)
            lp.production_intent_mark_spawned(intent, sid, cid, by=f"{by}:{start_token}")
            lp.write_production_intent(intent_p, intent)
            _legs_log(rs, f"FANOUT {sid} call={cid} {how} claim={nonce} token={start_token}")
    remaining = lp.production_intent_unspawned(intent)
    _legs_log(rs, f"FANOUT done spawned={len(spawned)} adopted={len(adopted)} recovered={len(recovered)} marks_observed={len(marks['observed'])} "
                  f"marks_adopted={len(marks['adopted'])} remaining={len(remaining)} rows={len(intent['rows'])} token={start_token}")
    return {"spawned": spawned, "adopted": adopted, "recovered": recovered, "marks": marks, "remaining": remaining, "rows": len(intent["rows"])}


@app.local_entrypoint()
def main(sha: str = "", plan: str = "", build: bool = False, stage: bool = False, restage: bool = False, stage_bank: bool = False,
         bank_local: str = str(Path.home() / lp.GRADING_HOST_LANE_BANK_REL),          # the LANE bank (lane_common.LANE_BANK); the Petri drive reads the same root (F-R)
         arditi_local: str = str(Path.home() / lp.GRADING_HOST_ARDITI_LADDER_REL),
         stage_plans: str = "", pf: bool = False, go: str = "", legs: str = "", smoke: str = "", smoke_id: str = "",
         smoke_max_children: int = 5, smoke_timeout_h: float = 4.0, rs: str = RS_DEFAULT,
         pool_urls: str = "", pull: str = "", pull_stages: str = "base", pull_to: str = "", probe_concurrency_n: int = 0,
         reattempt: bool = False, certificate: str = "", steward_deadline_h: float = 36.0, fanout: bool = False, start_token: str = "",
         carry_over: str = "", criterion3_waiver: str = ""):
    if len(sha) != 40:
        raise SystemExit("[seedrep] pass the full 40-hex --sha of the wave tree W")
    want = expected_app_name(sha)
    run_name = app.description or DEFAULT_APP_NAME
    if run_name != want:
        raise SystemExit(f"[seedrep] run with `modal run -n {want} ...`; got app name {run_name!r}")
    rs = site_run_state(rs)
    rs_p = Path(rs)
    plan_d = json.loads(Path(plan).read_text()) if plan else None
    if plan_d and plan_d["W"] != sha:
        raise SystemExit(f"[seedrep] the plan is for W {plan_d['W']}, not --sha {sha}")
    print(f"[seedrep] app={run_name} W={sha} plan={plan or '-'} dep_hashes={ {k: v[:12] for k, v in LOCAL_DEP_HASHES.items()} }", flush=True)
    if probe_concurrency_n:
        probe_concurrency(int(probe_concurrency_n))
    if build:
        rep = image_check.remote(LOCAL_DEP_HASHES[VLLM_REQS_REL])
        print(json.dumps(rep, indent=1, default=str))
        if not rep.get("ok"):
            raise SystemExit("[seedrep] the GPU image's venv check failed")
        (rs_p / "modal_image.json").parent.mkdir(parents=True, exist_ok=True)
        (rs_p / "modal_image.json").write_text(json.dumps({"image_id": rep.get("image_id"), "W": sha, "utc": utc()}, indent=1))
    if stage:
        do_stage_tree(sha, restage)
    if stage_bank:
        do_stage_bank(sha, Path(bank_local), Path(arditi_local), plan_d)         # the plan names the dose views to stage
    if stage_plans:
        if not plan_d:
            raise SystemExit("[seedrep] --stage-plans needs --plan (its plans_root)")
        do_stage_plans(Path(stage_plans), plan_d["plans_root"])
    if pf:
        if not plan_d:
            raise SystemExit("[seedrep] --pf needs --plan")
        rep = preflight.remote(sha, plan_d, LOCAL_DEP_HASHES)
        print(json.dumps(rep, indent=1, default=str))
        if not rep.get("ok"):
            raise SystemExit("[seedrep] PREFLIGHT FAILED -- fix every item above before any GPU")
        # check 8's authoritative half: the actual offline readers under the driver / client venv, after the static checks pass
        rep2 = preflight_offline_readers.remote(sha, plan_d)
        print(json.dumps({"offline_readers": rep2}, indent=1, default=str))
        if not rep2.get("ok"):
            raise SystemExit("[seedrep] PREFLIGHT FAILED -- the offline readers under the driver / client venv refused (see offline_readers above); no GPU")
    pools = json.loads(Path(pool_urls).read_text()) if pool_urls else None
    cert_p = Path(certificate) if certificate else rs_p / lp.CERTIFICATE_FILE
    # fj2: `--carry-over <prior RS or plan>[,<...>]` (comma-separated: modal's entrypoint takes one string per flag) -- alone: evaluate + write
    # RS/carry_over.json + seed this wave's accepted Dict (BEFORE certify); with --go / --legs: the cross-check against the certified record
    carry_sources = [s.strip() for s in carry_over.split(",") if s.strip()] if carry_over else []
    if carry_sources and not (go or legs):
        if not plan_d:
            raise SystemExit("[seedrep] --carry-over needs --plan (the FINAL plan the carried attempts are admitted into)")
        out = carry_over_evaluate(plan_d, rs=rs_p, sha=sha, sources=carry_sources, criterion3_waivers=load_criterion3_waivers(criterion3_waiver))
        print(json.dumps({"carry_over": out}, indent=1, default=str))
    if go or legs or smoke:
        if not plan_d:
            raise SystemExit("[seedrep] --go / --legs / --smoke need --plan")
        if go or legs:
            # the production launch = the certificate (bound to the plan's BYTES) -> the launcher lock -> the durable
            # fan-out INTENT -> the steward FIRST (its STEWARD_STARTED for THIS token); the STEWARD spawns from the intent -- the launcher spawns nothing
            out = launch_production(plan_d, plan_path=Path(plan), rs=rs_p, app_name=run_name, sha=sha, pools=pools, pool_urls_path=(Path(pool_urls) if pool_urls else None),
                                    cert_p=cert_p, go=go, legs=legs, steward_deadline_h=float(steward_deadline_h), carry_over=carry_sources)
            print(json.dumps({"production_launch": out}, indent=1, default=str))
        else:
            smoke_wu = smoke_wave_for(rs_p, smoke_id) if (smoke and smoke_id) else None
            picked, identity = select_launch_rows(plan_d, go=go, legs=legs, smoke=smoke, smoke_id=smoke_id, smoke_wave_uuid=smoke_wu,
                                                  smoke_max_children=smoke_max_children, smoke_timeout_h=smoke_timeout_h, pool_urls=pools)
            for r in picked:
                spawn_row(r, plan_d, run_name, rs_p, pools, smoke=identity)
    if fanout:
        if not plan_d:
            raise SystemExit("[seedrep] --fanout needs --plan")
        out = run_fanout(plan_d, plan_path=Path(plan), rs=rs_p, app_name=run_name, pools=pools, cert_p=cert_p, start_token=start_token)
        print(json.dumps({"fanout": out}, indent=1, default=str))
    if reattempt:
        if not plan_d:
            raise SystemExit("[seedrep] --reattempt needs --plan")
        cert = require_certificate(plan_d, plan_path=Path(plan), rs=rs_p, cert_p=cert_p)      # a re-attempt is a production spawn -- the same certificate gates it
        print(json.dumps({"reattempt_pass": reattempt_pass(plan_d, rs_p, run_name, pools, certificate=cert)}, indent=1, default=str))
    if pull:
        if not plan_d:
            raise SystemExit("[seedrep] --pull needs --plan")
        try:
            out = pull_leg(plan_d, pull, Path(pull_to or (rs_p / "stores")), rs_p, stages=pull_stages)
        except PullRefused as e:
            raise SystemExit(str(e))
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    raise SystemExit("run with `python -m modal run -n seedrep-served-<W7> scripts/modal/seedrep_served.py ...`")
