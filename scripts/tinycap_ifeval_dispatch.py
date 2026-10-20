#!/usr/bin/env python
"""IFEval canonical battery: the THIN wave-wide dispatcher.

Why a dispatcher beside the launcher's ``main``: the launcher runs ONE model per
invocation and fans every planned configuration out at once behind the
per-invocation ``gpu_container_budget`` trip-wire, while the IFEval battery is 42
launch rows over seven slugs on two images whose ceiling is the AGGREGATE GPU
concurrency (``contracts.MAX_GPU_CONTAINERS``). So this module walks the frozen
matrix, plans ONE configuration per row through the launcher's own planner, spawns
each through ``launch_configurations`` with exactly one configuration per call (the
budget guard never sees more than one) and keeps at most eight FunctionCalls
outstanding across every slug and both image functions. It is not an orchestrator:
no DAG, no ladder decision, no GPU-hour ledger; the ladder planner names the rows
to promote (``--rows``) and the rung (``--rung``), this module dispatches them.

State: ``dispatch_state.json`` is rewritten atomically after EVERY transition and a
per-configuration record lands at the row's ``dispatch_paths[cap]`` under the local
wave root, so a restart re-adopts every ACTIVE FunctionCall by id instead of
re-spawning it, never re-spawns DONE or REFUSED (a refusal reproduces itself at the
price of a standup), re-queues FAILED (a raised ``.get()`` or an error REC is
retryable) and refuses to start under a standing STOP sentinel.

The spawn boundary: a remote spawn is persisted as a LAUNCH RESERVATION
(``launching``, keyed by an idempotent ``launch_key`` the configuration carries into the
launcher's ``spawned.jsonl`` line) BEFORE Modal is asked for anything, and flipped to
``active`` with the FunctionCall id after. Before this the spawn preceded the durable
``queued -> active`` transition, so a dispatcher dying after Modal accepted the call but
before the id was persisted left a RUNNING call the state called queued, and a restart
spawned the configuration again (reproduced: 9 remote calls, 8 tracked, one configuration
twice — two writers on one out root). A restart now RECONCILES every reservation through the
journal: a line carrying its key names the accepted call, which is adopted; no line means
the spawn's fate is unknown and the reservation BLOCKS re-spawning that configuration until
the operator reconciles it (``--adopt <cid>=<fc_id>`` after finding the call, or
``--release <cid>`` after verifying none runs). While one is pending the
restart exits FAILED AT ONCE — before the preflights, before any adoption or spawn — because an
ambiguous reservation may name a running GPU call and the ceiling below cannot be held around it
(3 active + 1 ambiguous used to fill five more slots: nine calls outstanding, the block reported
only after the queue drained). And every reservation counts against the concurrency ceiling
(``outstanding``) by the DISTINCT calls its journal names, at least one: a reservation is not
assumed to be one call, and two journal lines carrying one call id are one call, not two.

The remote work's LIFETIME is independent of this process (fix lane X):
the loop runs inside ``app.run(detach=True)`` — a DETACHED Modal app keeps its
running FunctionCalls alive after the client disconnects, whereas an ephemeral
``app.run()`` (``detach=False``, the default) has the client's disconnect TERMINATE
every running task of the app (``modal/runner.py``: "Terminates all running tasks
for ephemeral apps"), so exiting on the tick / deadline guard or an interrupt would
have cancelled up to eight GPU containers while the state file called them
adoptable. The app id of every ``run`` is persisted in the state (``app_id`` = the
current one, ``app_ids`` = every one this wave used, and on each active entry), so
``--stop`` stops EVERY recorded app by id (the sentinel first, so a live loop spawns
nothing more) and a clean completion — nothing queued, nothing active — stops the
wave's own apps (``cleanup``); a guard exit stops nothing and prints the adopt line.
STOP is the ONLY path that cancels running calls, and it is explicit.

THE STATUS BOUNDARY (CONFIRMED HIGH; re-check Z18): ``FunctionCall.get(timeout=0)`` answers one of
THREE things — the call finished (a REC, or the remote exception Modal deserialised as the call's result), the call is
still running (a bare ``TimeoutError``), or NOTHING, because the client could not obtain the status: a ``ConnectionError``
after Modal's own retries, a grpclib stream error, a closed client, or the server REJECTING the poll RPC — Modal 1.5.4's
``grpc_error_converter`` (``modal/_grpc_client.py``) turns every gRPC status into a ``modal.exception`` class:
``ResourceExhaustedError`` for a server-side rate limit, ``PermissionDeniedError``, ``AuthError``, ``ServiceError``,
``InternalError``, ``NotFoundError``, … The third is a failure to OBTAIN the status, not a status: the remote call keeps
running. ``_poll_active`` used to file every exception other than the timeout as a TERMINAL failure, so one transport
error moved a running call to ``failed``, freed its slot, ``_fill_slots`` refilled it (reproduced: 8 unfinished calls, one
poll raising ``ConnectionError`` -> 9 calls at Modal while the dispatcher counted 8), and the forgotten call was then
cancellable by the clean-completion cleanup or duplicated by the FAILED re-queue. The first fix listed the transport
classes and kept ``terminal`` as the default; the re-check reproduced the same defect with the installed SDK's converter
(``ResourceExhaustedError`` / ``PermissionDeniedError`` matched neither the names nor the modules and read as terminal: a
rate-limited poll freed a running call's slot). So the rule is inverted — TERMINAL REQUIRES REMOTE EVIDENCE, EVERYTHING
ELSE IS UNKNOWN: ``poll_outcome`` files ``OUTCOME_TERMINAL`` only on affirmative evidence of the REMOTE function's own
result — the remote traceback Modal attaches to a deserialised remote exception (``__line_cache__`` / ``<ta-…>`` frames),
or one of the SDK's own result classes raised from a decoded ``GenericResult`` of this call (``FunctionTimeoutError``,
``OutputExpiredError``, ``InternalFailure``, ``RemoteError`` from the ``modal`` package) — and ``OUTCOME_UNKNOWN`` for
anything else the client raises, known class or not. On an
UNKNOWN outcome the entry STAYS active, keeps its slot (``outstanding`` counts it), records ``status_unknown_since`` /
``status_unknown_error`` / ``status_unknown_polls`` (state + dispatch record), and is re-polled with a doubling
tick backoff (1, 2, 4, 8 ticks); the first status obtained afterwards clears the marks (a REC files as before; a
running call is simply running). A call whose status stays unknown past ``--status-unknown-max-s`` (default 30 min)
exits the wave FAILED — fail closed, WITHOUT ``cleanup_apps`` (nothing is cancelled; every active call stays adoptable) —
and the remedies are the reservation path's: re-run with the same ``--state`` once the client obtains statuses again
(the call is adopted by id and polled again), or, having looked the call up, ``--release <cid>`` (verified: no call runs;
re-queued next start) / ``--adopt <cid>=<fc_id>`` (re-pointed). A configuration is re-queued ONLY after a terminal status
is established; an unknown-status call is never counted as free. The same rule guards adoption (``FunctionCall.from_id``
opens the client and decodes no result): every exception there is a status not obtained — a ``NotFoundError`` for an id
the server does not know included (not the call's result; the operator settles a wrong id with ``--release`` after
looking it up) — so the entry stays active without a handle and the loop retries the adoption. The price of the inverted
default is deliberate and loud: a finished call whose result the client cannot DECODE (Modal's ``ExecutionError`` out of
``_process_result``) reads as unknown on every poll and reaches the budget exit, where the operator settles it by name,
rather than being misfiled while a running call's slot is refilled.

ADMISSION (fix lane X): nothing spawns until the launcher's preflight
(``--pf``, the CPU function on Modal that reads the volume) has vouched for the
ACTUAL planned configurations — per (slug, cap) the queued rows' cells at THIS
dispatch's cap (``--off-cap-by-task ifeval=<cap>`` above pass 1), so the plan the
preflight vouches for has the same prompt class (the cell) and geometry entry
(``capability:ifeval[_longsi]_<cap>``) as the one the container will serve, and its
``nemotron_backend_attested`` check reads the same-GPU duty certification for that
entry and tree. Admission is REQUIRED (``--no-preflight`` is the explicit, logged
opt-out for a smoke on a stub); a failed slug prevents EVERY spawn; the admitted
(slug, cap) set is persisted in the state and every spawn is held to it; the
container re-derives the Nemotron certification itself before its standup
(``tinycap_served._run_configuration``, fail closed), so a spawn that slipped past
the operator-side gate still cannot pay for an unattested standup.

``store_complete``: the production default reads the dispatch record this module
wrote and treats a REC with one of the launcher's ``REC_OK_STATUSES`` (``OK`` = the
driver finished with nothing left, ``DONE`` = every task validly done before the
standup) as complete. The dispatcher runs on the operator's machine and cannot read
the volume, so the launcher's done-marker audit (``resume_report``, which needs the
served identity) is out of reach; the container re-runs that audit on a re-spawn,
so the least-assumption local reading is "a recorded OK/DONE REC".

Production: ``~/venvs/modal/bin/python scripts/tinycap_ifeval_dispatch.py --matrix
configs/sdfbase/ifeval/matrix.json --sha <tree sha>``. The launcher names its Modal
app at import from ``TINYCAP_APP_NAME``, so the name is exported BEFORE ``import
tinycap_served``; the loop runs inside ``with L.app.run(detach=True):``. Tests inject a stub
launcher, fake FunctionCalls, a fake clock and sleep; nothing here imports Modal."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import posixpath
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "src", _REPO / "scripts" / "modal"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sisteer import site  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.capeval.tinycap.completion import atomic_write_json  # noqa: E402

IFEVAL_DISPATCH_STATE = C.IFEVAL_DISPATCH_STATE
STOP_SENTINEL = "STOP"
STATE_SCHEMA = "ifeval_dispatch_state/1"
RECORD_SCHEMA = "ifeval_dispatch_record/1"
PASS1_CAP = int(C.IFEVAL_MAX_TOKENS_OFF)
PREFLIGHT_FLAG = "--pf"
DEFAULT_POLL_S = 30.0
DEFAULT_MAX_HOURS = 48.0
REC_OK_FALLBACK: Tuple[str, ...] = ("OK", "DONE")
EXIT_OK, EXIT_FAILED, EXIT_STOPPED = 0, 3, 4
TERMINAL = ("done", "refused", "failed")
#: the reservation bucket — a configuration whose spawn is persisted BEFORE Modal is asked; neither queued nor active
LAUNCHING = "launching"
TAG = "[ifeval-dispatch]"
ENTRY_KEYS = ("fc_id", "function", "name", "spawned_at", "status", "error", "rec", "t", "app_id", "launch_key", "reserved_at",
              "journal_fc_ids", "status_unknown_since", "status_unknown_error", "status_unknown_polls")
#: the status boundary (module docstring): what one ``FunctionCall.get(timeout=0)`` told us
OUTCOME_RUNNING, OUTCOME_TERMINAL, OUTCOME_UNKNOWN = "running", "terminal", "unknown"
#: TERMINAL REQUIRES REMOTE EVIDENCE, EVERYTHING ELSE IS UNKNOWN (re-check Z18). The two forms of evidence:
#: (1) Modal attaches the REMOTE traceback to a deserialised remote exception (``modal._traceback.append_modal_tb``, called
#: from ``_process_result``: the frames' filenames start ``<ta-…>:`` and the exception gains ``__line_cache__``) — the mark
#: of a call's own RESULT whatever the class (a container's own ``ConnectionError`` is a terminal remote failure);
#: (2) the SDK's own RESULT classes, raised by ``_process_result`` / ``poll_function`` from a decoded ``GenericResult`` of
#: THIS call — ``FunctionTimeoutError`` (GENERIC_STATUS_TIMEOUT), ``InternalFailure`` (GENERIC_STATUS_INTERNAL_FAILURE),
#: ``RemoteError`` (a failure with no serialised exception), ``OutputExpiredError`` (no outputs and no unfinished inputs)
#: — matched by class name AND the ``modal`` package. Everything else the client raises is a status NOT obtained: the
#: builtin ``ConnectionError`` family, grpclib's errors, every ``modal.exception`` class ``grpc_error_converter`` translates
#: a rejected RPC into (``ResourceExhaustedError`` = a server-side rate limit, ``PermissionDeniedError``, ``AuthError``,
#: ``ServiceError``, ``InternalError``, ``NotFoundError``, …), ``ClientClosed``, Modal's ``ConnectionError``, an
#: ``ExecutionError``, an unmarked exception of any class. There is deliberately NO allow-list of transport classes: the
#: first fix had one, and the installed SDK's converter raised two classes it did not name.
REMOTE_TRACEBACK_ATTR = "__line_cache__"
REMOTE_TRACEBACK_FRAME_PREFIX = "<ta-"
RESULT_EXCEPTION_NAMES: Tuple[str, ...] = ("FunctionTimeoutError", "OutputExpiredError", "InternalFailure", "RemoteError")
RESULT_EXCEPTION_PACKAGE = "modal"
#: how long one call's status may stay unknown before the wave fails closed (state persisted, nothing cancelled)
DEFAULT_STATUS_UNKNOWN_MAX_S = 1800.0
#: the re-poll backoff of an unknown-status call, in TICKS: 1, 2, 4, 8, 8, … (8 ticks = 4 min at the default poll)
STATUS_UNKNOWN_MAX_BACKOFF_TICKS = 8
#: the key the configuration carries into the launcher's ``spawned.jsonl`` line (the idempotent submission key)
LAUNCH_KEY_FIELD = "launch_key"
#: every Modal CLI the dispatcher runs goes through THIS interpreter (``python -m modal``), never a bare
#: ``modal`` executable — under the documented absolute venv interpreter the venv's ``bin`` is not on ``PATH``, so
#: ``shutil.which("modal")`` is None, the first preflight died with ``FileNotFoundError`` and ``--stop`` could not run
MODAL_ARGV: Tuple[str, ...] = (sys.executable, "-m", "modal")
#: the ``modal app stop <id> --yes`` of one app, injectable (tests); the CLI aborts without a TTY, hence ``--yes``
STOP_APP_ARGV: Tuple[str, ...] = (*MODAL_ARGV, "app", "stop")
#: the launcher script the preflight runs, by ABSOLUTE repository path (the launcher's own argv spells it relative to
#: the repository root, which is only right when that root is the working directory)
LAUNCHER_SCRIPT = _REPO / "scripts" / "modal" / "tinycap_served.py"


def wave_root_for(sha: str) -> str:
    """The wave's root ON THE VOLUME (out roots and item subsets are container paths)."""
    return f"{C.VOL_TINYCAP_ROOT}/ifeval_canonical/{str(sha)[:7]}"


def local_root_for(sha: str) -> Path:
    """The wave's root on the operator's machine (state, spawned log, dispatch records). ``LOCAL_OUT_ROOT`` is expanded
    through ``sisteer.site`` (placeholders, then ``~``): a placeholder that does not resolve here refuses by name, never a
    literal directory."""
    return site.expand_path(C.LOCAL_OUT_ROOT, key="LOCAL_OUT_ROOT") / "ifeval_canonical" / str(sha)[:7]


def app_name_for(sha: str) -> str:
    return f"ifeval-canonical-{str(sha)[:7]}"


def mint_run_id(sha: str) -> str:
    return f"{str(sha)[:7]}-{uuid.uuid4().hex[:8]}"


def mint_launch_key(run_id: str, cid: str) -> str:
    """The idempotent submission key of ONE spawn attempt of ``cid`` in this wave run — unique per attempt, so a re-queued
    FAILED row's next spawn is another key and the journal can tell the two apart."""
    return f"{run_id}/{cid}/{uuid.uuid4().hex[:12]}"


def journal_lines_for(spawned_log: Path, launch_key: str) -> List[dict]:
    """Every ``spawned.jsonl`` line the launcher appended for ``launch_key`` (normally 0 or 1; 2 would be the very bug the
    reservation exists to prevent, and is reported as ambiguous)."""
    p = Path(spawned_log)
    if not p.is_file():
        return []
    out: List[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue                # a torn trailing line is not a journal entry
        if isinstance(rec, Mapping) and rec.get(LAUNCH_KEY_FIELD) == launch_key:
            out.append(dict(rec))
    return out


def journal_calls_for(spawned_log: Path, launch_key: str) -> List[str]:
    """The DISTINCT FunctionCall ids the journal records for ``launch_key``, in first-seen order: one
    id on two lines is ONE call (a line written twice, not a spawn made twice); two ids are two spawns of one
    reservation; a line without an id records no call. What the reconciliation and the concurrency ceiling count."""
    return list(dict.fromkeys(str(ln["fc_id"]) for ln in journal_lines_for(spawned_log, launch_key) if ln.get("fc_id")))


def _by_cap(row: Optional[Mapping], field: str, cap: int):
    """A cap-keyed matrix field; JSON spells the keys as strings."""
    m = (row or {}).get(field) or {}
    return m.get(str(cap), m.get(cap))


def configuration_id(row: Mapping, cap: int) -> str:
    cid = _by_cap(row, "configuration_ids", cap)
    if not cid:
        raise ValueError(f"row {row.get('slug')} x {row.get('cell')} names no configuration id at cap {cap}")
    return str(cid)


def load_matrix(path: str | Path) -> dict:
    doc = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(doc, Mapping) or not isinstance(doc.get("rows"), list) or not doc["rows"]:
        raise ValueError(f"{path}: not an IFEval matrix (no rows)")
    return dict(doc)


def select_rows(matrix: Mapping, *, cap: int = PASS1_CAP, rows: Optional[Sequence[str]] = None
                ) -> Tuple[List[dict], List[dict]]:
    """``(launch rows in matrix order, held rows)`` at ``cap``. A held row never
    enters the queue, whatever ``--rows`` says; neither does a DEFERRED row (an
    AlphaSteer / Arditi design slot whose object sha is not filled — it is
    returned among the held rows, so the state file lists it under ``held`` with
    its record path as the reason); an unknown status, a cap off the row's
    ladder, a rung above pass 1 without its subset / out root, and a ``--rows``
    id that is held, deferred or unknown are refused rather than skipped."""
    launch, held = [], []
    for r in matrix["rows"]:
        status = r.get("status")
        if status not in ("launch", "held", "deferred"):
            raise ValueError(f"row {r.get('slug')} x {r.get('cell')}: status {status!r} is not launch | held | deferred")
        if status == "deferred":
            if r.get("object_sha256"):
                raise ValueError(f"row {r.get('slug')} x {r.get('cell')}: status deferred but an object_sha256 is filled — "
                                 "regenerate the matrix (a filled slot is a launch row), never launch a deferred one")
            rec = dict(r)
            rec["held_reason"] = (f"DEFERRED: a design slot with no object sha; filled from "
                                  f"{(r.get('deferred') or {}).get('record')} — never launched")
            held.append(rec)
            continue
        (held if status == "held" else launch).append(dict(r))
    for r in launch:
        if int(cap) not in [int(c) for c in (r.get("caps") or ())]:
            raise ValueError(f"{r.get('slug')} x {r.get('cell')}: cap {cap} is not on the row's ladder {r.get('caps')}")
        if int(cap) != PASS1_CAP and not (_by_cap(r, "subset_paths", cap) and _by_cap(r, "rung_out_roots", cap)):
            raise ValueError(f"{configuration_id(r, cap)}: rung {cap} has no subset path / rung out root in the matrix")
    if rows:
        by_id = {configuration_id(r, cap): r for r in launch}
        held_ids = {configuration_id(r, cap) for r in held}
        wanted = list(dict.fromkeys(rows))
        bad_held = [w for w in wanted if w in held_ids]
        unknown = [w for w in wanted if w not in by_id and w not in held_ids]
        if bad_held or unknown:
            raise ValueError(f"--rows: held or deferred {bad_held}, unknown {unknown} at cap {cap}")
        launch = [r for r in launch if configuration_id(r, cap) in set(wanted)]
    return launch, held


def row_paths(row: Mapping, cap: int, *, wave_root: str) -> Tuple[str, Optional[str]]:
    """``(out_dir, item_subset)`` on the volume: pass 1 runs the full item set into
    ``out_root``; a higher rung runs ``subset_paths[cap]`` into ``rung_out_roots[cap]``."""
    if int(cap) == PASS1_CAP:
        return posixpath.join(wave_root, row["out_root"]), None
    return (posixpath.join(wave_root, _by_cap(row, "rung_out_roots", cap)),
            posixpath.join(wave_root, _by_cap(row, "subset_paths", cap)))


#: the serving path a matrix ROW stamps (`serving_path`, the generator's document-level
#: declaration copied onto every row). A row that stamps `lora_native` is planned with `serving_path=lora_native` — the launcher
#: then composes `--adapter-bank /vol/tinycap/multilora_adapters/<model key>` and marks the configuration natively; a row that
#: stamps nothing (the frozen 40-row matrix) is planned exactly as before (the launcher's LIBRARY default, lens).
ROW_SERVING_PATH_KEY = "serving_path"
SERVING_PATH_LORA_NATIVE = "lora_native"
SERVING_PATH_LENS = "lens"
SERVING_PATHS = (SERVING_PATH_LORA_NATIVE, SERVING_PATH_LENS)
#: the dispatch record's two serving-path fields: what the matrix ROW asked for (written from
#: the `queued` state on) and what the COMPLETED store's manifest.json ATTESTS (written only at `done`, from the store, never
#: from the row or the plan); a disagreement fails the row (REC_SERVING_PATH_MISMATCH) and the wave exits non-zero
RECORD_KEY_REQUESTED_SERVING_PATH = "requested_serving_path"
RECORD_KEY_ATTESTED_SERVING_PATH = "serving_path_attested"
RECORD_KEY_ATTESTED_SOURCE = "serving_path_attested_source"
REC_SERVING_PATH_MISMATCH = "SERVING_PATH_MISMATCH"
def attest_store_serving_path(manifests: Sequence[Tuple[str, Mapping]]) -> Tuple[str, str]:
    """The ONE serving path a completed store's manifests attest (``[(manifest path, document), ...]``, every
    ``<out_dir>/cells/*/manifest.json``), each read through the contracts' one reader (``read_store_serving_engine``: every
    spelling present must agree and match the serving identity; a refusal is a torn / relabelled record). A manifest that
    STAMPS no engine under any spelling refuses here — a store this wave produced always stamps it, so the reader's
    lens-BY-ERA resolution is never taken for it; every manifest under the root must agree; none is a refusal."""
    if not manifests:
        raise ValueError("the store has no cells/*/manifest.json to attest a serving path from")
    seen: Dict[str, List[str]] = {}
    for path, doc in manifests:
        r = C.read_store_serving_engine(doc, where=str(path))
        if r.refusal:
            raise ValueError(r.refusal)
        if r.stated is None:
            raise ValueError(f"{path}: stamps no serving engine under any spelling ({r.why}) — never read as lens by era for a "
                             "store this wave produced")
        seen.setdefault(str(r.engine), []).append(r.why)
    if len(seen) > 1:
        raise ValueError(f"the store's manifests attest more than one serving path: {seen!r}")
    ((engine, sources),) = seen.items()
    return engine, "; ".join(sources)


def default_store_manifests(out_dir: str, *, volume=None) -> List[Tuple[str, Mapping]]:
    """The production reader of a completed store's manifests OFF THE VOLUME, in-process through the modal SDK (no cold
    ``python -m modal`` subprocess per row): ``Volume.listdir(<out_dir>/cells)`` for the hashed cell dirs, ``read_file`` of each
    ``manifest.json``. ``out_dir`` is the container path (``/vol/...``); the volume path drops the mount prefix. ``volume`` is
    injectable (tests hand a fake with ``listdir`` / ``read_file``); by default ``modal.Volume.from_name(C.MODAL_VOLUME)``,
    imported lazily so the dispatcher's tests run without the SDK. A listing or a read that fails raises — the attestation is
    then a refusal, never a guessed engine."""
    if volume is None:
        import modal                                  # the modal venv is the dispatcher's production interpreter
        volume = modal.Volume.from_name(C.MODAL_VOLUME)
    rel = volume_relative(out_dir)
    manifests: List[Tuple[str, Mapping]] = []
    for entry in volume.listdir(f"{rel}/cells"):
        if getattr(getattr(entry, "type", None), "name", str(getattr(entry, "type", ""))) not in ("DIRECTORY", "dir"):
            continue
        path = f"{entry.path}/manifest.json"
        manifests.append((f"/{path}", json.loads(b"".join(volume.read_file(path)).decode("utf-8"))))
    return manifests


def volume_relative(container_path: str) -> str:
    """``/vol/tinycap/...`` -> ``tinycap/...`` (the volume's own path for a container mount path); a path outside the mount
    refuses by name."""
    mount = C.VOL_TINYCAP_ROOT.rsplit("/", 1)[0]            # "/vol"
    p = str(container_path)
    if not p.startswith(mount + "/"):
        raise ValueError(f"{p!r} is not under the volume mount {mount}/")
    return p[len(mount) + 1:]


def row_serving_path(row: Mapping) -> Optional[str]:
    """The serving path ``row`` stamps, or None (an unstamped row = the launcher's library default, lens). Anything but the
    launcher's two spellings refuses by name — a typo would otherwise plan the other engine silently."""
    sp = row.get(ROW_SERVING_PATH_KEY)
    if sp is None:
        return None
    if sp not in (SERVING_PATH_LORA_NATIVE, SERVING_PATH_LENS):
        raise ValueError(f"row {row.get('slug')} x {row.get('cell')}: {ROW_SERVING_PATH_KEY} {sp!r} is not "
                         f"{SERVING_PATH_LORA_NATIVE!r} | {SERVING_PATH_LENS!r}")
    return str(sp)


def rows_serving_path(rows: Sequence[Mapping]) -> Optional[str]:
    """ONE serving path for a set of rows (the preflight plans a slug's cells together): every row's stamp must agree; a
    mixed set refuses by name; an unstamped set is None."""
    paths = {row_serving_path(r) for r in rows}
    if len(paths) > 1:
        raise ValueError(f"the rows stamp more than one serving path {sorted(str(p) for p in paths)} — one engine per plan")
    return next(iter(paths)) if paths else None


def reduction_mode_for(serving_path: Optional[str]) -> str:
    """The lens reduction mode a plan / GPU kwargs / preflight argv carry for ``serving_path``: ``stacked`` under the lens
    (the frozen matrix's rows, as before; None = the library default, lens), the launcher's DEFAULT under native (the
    launcher's own dry run of 2026-09-22: "--reduction-mode 'stacked' is a lens fact the native engine cannot attest;
    under --serving-path lora_native only the default is accepted", so a native container handed ``stacked`` would refuse
    after its standup was planned)."""
    return C.REDUCTION_MODE_LAUNCHER_DEFAULT if serving_path == SERVING_PATH_LORA_NATIVE else C.REDUCTION_MODE_STACKED


def row_reduction_mode(row: Mapping) -> str:
    return reduction_mode_for(row_serving_path(row))


def plan_row(launcher, row: Mapping, cap: int, *, wave_root: str) -> dict:
    """ONE launcher configuration for the row at ``cap``: the planner is the
    launcher's own, so the container name, standup extra and identity are never a
    second spelling. A row stamping ``serving_path`` plans under it — with
    the launcher's default reduction mode, the only one the native engine admits
    (:func:`row_reduction_mode`); an unstamped row plans exactly as before (no
    keyword passed, ``stacked``)."""
    out_dir, subset = row_paths(row, cap, wave_root=wave_root)
    sp = row_serving_path(row)
    cfgs = launcher.plan_configurations(
        cells=[row["cell"]], modes=[C.THINKING_OFF], controls=[C.CONTROL_IFEVAL], model=row["slug"],
        sampling_mode=C.SAMPLING_VENDOR_PRESET, samples=C.IFEVAL_SAMPLES_PER_ITEM,
        reduction_mode=row_reduction_mode(row), pass_no=1,
        off_cap_by_task=({C.IFEVAL_TASK_NAME: int(cap)} if int(cap) != PASS1_CAP else None),
        item_subset=subset, bank_view=row["view"], out_dir=out_dir,
        **({"serving_path": sp} if sp is not None else {}))
    if len(cfgs) != 1:
        raise ValueError(f"{configuration_id(row, cap)}: the planner returned {len(cfgs)} configurations, not 1")
    cfg = dict(cfgs[0])
    if sp is not None and cfg.get(ROW_SERVING_PATH_KEY, SERVING_PATH_LENS) != sp:
        # the planner marks a native configuration with the key (present iff native); a lens plan carries none
        raise ValueError(f"{configuration_id(row, cap)}: the row stamps {ROW_SERVING_PATH_KEY} {sp!r} but the planner returned "
                         f"{cfg.get(ROW_SERVING_PATH_KEY, SERVING_PATH_LENS)!r} — the launcher did not honour the serving path")
    return cfg


def object_digests_for(row: Mapping) -> Optional[Dict[str, Dict[str, object]]]:
    """The ``object_digests`` a lens container reads (``object_digest_for``: keyed
    by cell, ``artifact_sha256`` the digest the standup is held to): the frozen
    matrix's ``object_sha256``; ``None`` for a base / prompted row."""
    digest = row.get("object_sha256")
    if not digest:
        return None
    if not C.is_sha256_hex(digest):
        raise ValueError(f"{row.get('slug')} x {row.get('cell')}: object_sha256 {digest!r} is not a sha256 hex")
    return {row["cell"]: {"artifact_sha256": digest, "intervention_hash": None, "expected": None,
                          "lens_route_intervention_hash": None}}


def kwargs_for(row: Mapping, *, sha: str, run_id: str) -> Dict[str, object]:
    """The GPU function's kwargs: the launcher ``main``'s own ``fan_out`` set (the reduction mode per the row's engine —
    :func:`row_reduction_mode`; the container refuses ``stacked`` under native)."""
    return {"sha": sha, "limit": 0, "inspect_limit": 0, "samples": C.IFEVAL_SAMPLES_PER_ITEM, "pass_no": 1,
            "controls": [C.CONTROL_IFEVAL], "reduction_mode": row_reduction_mode(row),
            "object_digests": object_digests_for(row), "oos_prompt_budget": None, "run_id": run_id, "ladder": None}


def function_for(launcher, row: Mapping):
    return launcher.configuration_function(C.model_row(row["slug"]))


def _fn_name(fn) -> str:
    return next((v for v in (getattr(fn, "__name__", None), getattr(fn, "tag", None)) if isinstance(v, str) and v),
                type(fn).__name__)


def _still_running(exc: BaseException) -> bool:
    """``fc.get(timeout=0)`` on an unfinished call raises a bare ``TimeoutError``
    (Modal's poll raises the builtin; its subclasses ``FunctionTimeoutError`` /
    ``OutputExpiredError`` are real failures and are NOT matched)."""
    return isinstance(exc, TimeoutError) or type(exc).__name__ == "TimeoutError"


def remote_origin(exc: BaseException) -> bool:
    """Whether Modal handed ``exc`` back as the call's RESULT: a deserialised remote exception carries the remote
    traceback (``append_modal_tb``: ``__line_cache__`` on the exception, frames whose filename starts ``<ta-``)."""
    if hasattr(exc, REMOTE_TRACEBACK_ATTR):
        return True
    tb = exc.__traceback__
    while tb is not None:
        if str(tb.tb_frame.f_code.co_filename).startswith(REMOTE_TRACEBACK_FRAME_PREFIX):
            return True
        tb = tb.tb_next
    return False


def sdk_result(exc: BaseException) -> bool:
    """Whether ``exc`` is one of the SDK's own RESULT classes (:data:`RESULT_EXCEPTION_NAMES`, from the ``modal``
    package): raised by ``_process_result`` / ``poll_function`` from a decoded ``GenericResult`` of this call — the
    server's verdict on the call, carrying no remote traceback. A same-named class from any other package is not."""
    cls = type(exc)
    module = str(getattr(cls, "__module__", "") or "")
    return cls.__name__ in RESULT_EXCEPTION_NAMES and module.split(".")[0] == RESULT_EXCEPTION_PACKAGE


def poll_outcome(exc: BaseException) -> str:
    """What an exception out of ``FunctionCall.get(timeout=0)`` (or ``FunctionCall.from_id``) established about the call
    (module docstring, the status boundary). TERMINAL REQUIRES REMOTE EVIDENCE, EVERYTHING ELSE IS UNKNOWN: ``running``
    for the bare timeout; ``terminal`` only on affirmative evidence of the REMOTE function's own result — the remote
    traceback mark (:func:`remote_origin`) or one of the SDK's result classes (:func:`sdk_result`); ``unknown`` for
    everything else the client raised — a transport error, a rejected RPC translated by ``grpc_error_converter``
    (``ResourceExhaustedError``, ``PermissionDeniedError``, ``NotFoundError``, …), a closed client, an
    ``ExecutionError``, an unmarked exception of any class. An unknown outcome NEVER frees the slot."""
    if _still_running(exc):
        return OUTCOME_RUNNING
    if remote_origin(exc) or sdk_result(exc):
        return OUTCOME_TERMINAL
    return OUTCOME_UNKNOWN


def _exc_text(exc: BaseException, n: int = 300) -> str:
    return f"{type(exc).__name__}: {str(exc)[:n]}"


def default_store_complete(local_root: Path, cap: int, rows_by_id: Mapping[str, Mapping],
                           ok_statuses: Sequence[str]) -> Callable[[Mapping], bool]:
    """The production ``store_complete``: the configuration's dispatch record carries
    a REC with an OK status (the module docstring says why not the volume's markers)."""
    def _complete(cfg: Mapping) -> bool:
        rel = _by_cap(rows_by_id.get(str(cfg.get("configuration_id") or "")), "dispatch_paths", cap)
        p = local_root / rel if rel else None
        if p is None or not p.is_file():
            return False
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return False
        return rec.get("state") == "done" and str(rec.get("status")) in tuple(ok_statuses)
    return _complete


def default_adopt(fc_id: str):
    import modal  # the production dispatcher runs under the modal venv; tests inject
    return modal.FunctionCall.from_id(fc_id)


class Dispatcher:
    """The loop: poll the active set, fill free slots from the queue (at most
    ``max_outstanding`` outstanding across everything), persist after every
    transition; exit when nothing is queued or active, on a STOP sentinel, or at
    the tick / deadline guard (state persisted, active calls left adoptable)."""

    def __init__(self, launcher, matrix: Mapping, *, sha: str, cap: int = PASS1_CAP,
                 rows: Optional[Sequence[str]] = None, app_name: str = "", wave_root: str = "",
                 local_root: Optional[str | Path] = None, state_path: Optional[str | Path] = None,
                 run_id: str = "", max_outstanding: int = C.MAX_GPU_CONTAINERS,
                 poll_s: float = DEFAULT_POLL_S, max_hours: float = DEFAULT_MAX_HOURS,
                 max_ticks: Optional[int] = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.time, adopt: Optional[Callable[[str], object]] = None,
                 store_complete: Optional[Callable[[Mapping], bool]] = None,
                 log: Callable[[str], None] = lambda s: print(s, flush=True),
                 app_id: Optional[str] = None, stop_app: Optional[Callable[[str], int]] = None,
                 admitted: Optional[Mapping[str, Mapping]] = None,
                 status_unknown_max_s: float = DEFAULT_STATUS_UNKNOWN_MAX_S,
                 store_manifests: Optional[Callable[[str], Sequence[Tuple[str, Mapping]]]] = None):
        self.launcher, self.sha, self.cap = launcher, str(sha), int(cap)
        #: how a COMPLETED store's manifests are read for the serving path it attests (``out_dir`` ->
        #: ``[(manifest path, document), ...]``); the production reader pulls them off the volume; tests inject one
        self.store_manifests = store_manifests or default_store_manifests
        #: the status boundary: how long one active call's status may stay unknown before the wave fails closed
        self.status_unknown_max_s = float(status_unknown_max_s)
        #: the loop's tick counter (the unit of the unknown-status re-poll backoff)
        self.tick = 0
        #: the slugs the preflight ADMITTED at this cap (``run_preflights``' verdicts, ``{slug: record}``);
        #: None = admission not enforced by this object (a direct construction — the tests' loop harness; ``main``
        #: always passes the verdicts, or an explicit ``--no-preflight`` opt-out record)
        self.admitted = dict(admitted) if admitted is not None else None
        #: the DETACHED Modal app this process spawns into (``app.run(detach=True)``'s app id; None under a stub
        #: launcher / a dry run) and how an app is stopped (``modal app stop <id> --yes``; injectable)
        self.app_id = str(app_id) if app_id else None
        self.stop_app = stop_app or default_stop_app
        self.app_name = app_name or app_name_for(sha)
        self.wave_root = wave_root or wave_root_for(sha)
        self.local_root = Path(local_root).expanduser() if local_root else local_root_for(sha)
        self.state_path = Path(state_path).expanduser() if state_path else self.local_root / IFEVAL_DISPATCH_STATE
        self.stop_path = self.state_path.parent / STOP_SENTINEL
        self.spawned_log = self.local_root / getattr(launcher, "SPAWNED_LOG_DEFAULT", "spawned.jsonl")
        self.max_outstanding, self.poll_s, self.max_hours, self.max_ticks = int(max_outstanding), poll_s, max_hours, max_ticks
        self.sleep, self.clock, self.log, self.adopt = sleep, clock, log, adopt or default_adopt
        self.ok_statuses = tuple(getattr(launcher, "REC_OK_STATUSES", REC_OK_FALLBACK))
        launch_all, held = select_rows(matrix, cap=self.cap)
        self.all_rows_by_id = {configuration_id(r, self.cap): r for r in launch_all}
        self.rows_by_id = {configuration_id(r, self.cap): r for r in select_rows(matrix, cap=self.cap, rows=rows)[0]}
        self.held = {configuration_id(r, self.cap): r.get("held_reason") for r in held}
        self.store_complete = store_complete or default_store_complete(
            self.local_root, self.cap, self.all_rows_by_id, self.ok_statuses)
        self.handles: Dict[str, object] = {}
        self._cfgs: Dict[str, dict] = {}
        self.state = self._load_or_init(run_id)

    # -- state ---------------------------------------------------------------
    def _load_or_init(self, run_id: str) -> dict:
        st: Optional[dict] = None
        if self.state_path.is_file():
            st = json.loads(self.state_path.read_text(encoding="utf-8"))
            for key, want in (("schema", STATE_SCHEMA), ("sha", self.sha), ("cap", self.cap)):
                if st.get(key) != want:
                    raise ValueError(f"{self.state_path}: {key} {st.get(key)!r} != {want!r}; a different wave's state")
            if run_id and st.get("run_id") != run_id:
                raise ValueError(f"{self.state_path}: run_id {st.get('run_id')!r} != --run-id {run_id!r}")
        if st is None:
            st = {"schema": STATE_SCHEMA, "run_id": run_id or mint_run_id(self.sha), "sha": self.sha, "cap": self.cap,
                  "app_name": self.app_name, "wave_root": self.wave_root, "max_outstanding": self.max_outstanding,
                  "queued": [], LAUNCHING: {}, "active": {}, "done": {}, "refused": {}, "failed": {}, "retried": {}, "held": {},
                  "app_id": None, "app_ids": [], "updated_at": None}
        # a state written before the reservation bucket existed reads as having none
        st.setdefault(LAUNCHING, {})
        st.setdefault("reconciled", {})
        # the status boundary: an unknown-status mark from a previous process rides into this one (its budget clock keeps
        # running — a restart is not a fresh budget), but that process's re-poll TICK is meaningless here: poll at once
        for ent in st["active"].values():
            ent.pop("status_unknown_retry_tick", None)
        # the detached app THIS process spawns into, and every app the wave has used (a restart runs a NEW
        # detached app while the adopted calls keep running under the old one — both must be stoppable by id)
        st["app_id"] = self.app_id
        st["app_ids"] = list(dict.fromkeys([*(st.get("app_ids") or []), *([self.app_id] if self.app_id else [])]))
        if self.admitted is not None:
            st["admission"] = {"cap": self.cap, "slugs": {k: dict(v) for k, v in self.admitted.items()}}
        st["held"] = dict(self.held)
        st["queued"] = []
        for cid in self.rows_by_id:
            # a RESERVATION is neither queued nor active: it is reconciled at start (`_reconcile_reservations`), and an
            # ambiguous one blocks the configuration until the operator settles it — never silently re-queued
            if cid in st["done"] or cid in st["refused"] or cid in st["active"] or cid in st[LAUNCHING]:
                continue
            prev = st["failed"].pop(cid, None)
            if prev is not None:
                st.setdefault("retried", {})[cid] = prev
                self.log(f"{TAG} {cid}: re-queued after FAILED ({prev.get('error')})")
            st["queued"].append(cid)
        return st

    @property
    def run_id(self) -> str:
        return str(self.state["run_id"])

    def _persist(self) -> None:
        self.state["updated_at"] = self.clock()
        atomic_write_json(self.state_path, self.state)

    def _row(self, cid: str) -> Mapping:
        return self.rows_by_id.get(cid) or self.all_rows_by_id[cid]

    def _cfg(self, cid: str) -> dict:
        if cid not in self._cfgs:
            self._cfgs[cid] = {**plan_row(self.launcher, self._row(cid), self.cap, wave_root=self.wave_root),
                               "configuration_id": cid}
        return self._cfgs[cid]

    def _write_record(self, cid: str, state: str, entry: Mapping) -> None:
        row = self._row(cid)
        rel = _by_cap(row, "dispatch_paths", self.cap)
        if not rel:
            raise ValueError(f"{cid}: the matrix names no dispatch path at cap {self.cap}")
        out_dir, subset = row_paths(row, self.cap, wave_root=self.wave_root)
        requested = row_serving_path(row)
        atomic_write_json(self.local_root / rel, {
            "schema": RECORD_SCHEMA, "configuration_id": cid, "state": state, "cap": self.cap, "run_id": self.run_id,
            "app_name": self.app_name, "slug": row["slug"], "cell": row["cell"], "view": row.get("view"),
            "image": row.get("image"), "object_sha256": row.get("object_sha256"), "out_dir": out_dir,
            "item_subset": subset, "geometry_eval": _by_cap(row, "geometry_eval_by_cap", self.cap),
            # what the ROW asked for (present from `queued` on; the FINAL out_dir above is the
            # one the container writes under — an explicit out root is left as named by the launcher, never re-rooted), and
            # what the COMPLETED store attests (written at `done` ONLY, read off its manifest.json; None until then)
            **({RECORD_KEY_REQUESTED_SERVING_PATH: requested} if requested is not None else {}),
            **({RECORD_KEY_ATTESTED_SERVING_PATH: entry.get(RECORD_KEY_ATTESTED_SERVING_PATH),
                RECORD_KEY_ATTESTED_SOURCE: entry.get(RECORD_KEY_ATTESTED_SOURCE)}
               if requested is not None or entry.get(RECORD_KEY_ATTESTED_SERVING_PATH) is not None else {}),
            **{k: entry.get(k) for k in ENTRY_KEYS}})

    def _state_of(self, cid: str) -> str:
        return next((s for s in ("queued", LAUNCHING, "active") + TERMINAL if cid in self.state[s]), "new")

    def _transition(self, cid: str, to: str, *, rec: Optional[Mapping] = None, error: str = "",
                    entry: Optional[Mapping] = None, note: str = "", extra: Optional[Mapping] = None) -> None:
        prev = self._state_of(cid)
        # the entry moves out of whichever live bucket holds it (a reservation's fields ride into the active entry);
        # `extra` (the serving-path attestation) merges onto the selected entry, whichever it was
        reserved = self.state[LAUNCHING].pop(cid, None)
        ent = dict(entry or self.state["active"].pop(cid, None) or reserved or {})
        if entry is not None:
            self.state["active"].pop(cid, None)
        ent.update(dict(extra or {}))
        if cid in self.state["queued"]:
            self.state["queued"].remove(cid)
        ent["t"] = self.clock()
        if rec is not None:
            ent["status"], ent["rec"] = rec.get("status"), {k: v for k, v in rec.items() if k != "evidence"}
        if error:
            ent["error"] = error
        self.state[to][cid] = ent
        if to in TERMINAL:
            self.handles.pop(cid, None)
        self._persist()
        self._write_record(cid, to, ent)
        detail = " | ".join(x for x in (note, f"status {ent.get('status')}" if rec is not None else "",
                                        f"error {error}" if error else "") if x)
        self.log(f"{TAG} {cid}: {prev} -> {to}" + (f" ({detail})" if detail else ""))

    # -- the loop ------------------------------------------------------------
    def _reservation_calls(self, ent: Mapping) -> List[str]:
        """The distinct calls the journal names for one reservation (``journal_calls_for``); none without a key."""
        key = ent.get(LAUNCH_KEY_FIELD)
        return journal_calls_for(self.spawned_log, str(key)) if key else []

    def ambiguous_reservations(self) -> Dict[str, List[str]]:
        """``{cid: distinct journal call ids}`` for every reservation the journal does NOT settle — zero ids (the
        process died between Modal's acceptance and the journal write: the spawn's fate is unknown) or two or more
        (two spawns of one reservation). READ-ONLY, so ``main`` asks before the preflights and before entering the
        detached app; ``_reconcile_reservations`` applies the same classification."""
        return {cid: ids for cid, ent in self.state[LAUNCHING].items()
                if len(ids := self._reservation_calls(ent)) != 1}

    def _ambiguity_line(self, cid: str, ent: Mapping, fc_ids: Sequence[str]) -> str:
        key = ent.get(LAUNCH_KEY_FIELD)
        app = ent.get("app_id") or self.app_name
        if not fc_ids:
            why, remedy = ("no journal line carries its launch key — the spawn's fate is unknown",
                           f"find the call (the Modal dashboard / `modal app logs`, app {app}), then either "
                           f"`--adopt {cid}=<fc_id>` or, having verified that no call runs, `--release {cid}`")
        else:
            why, remedy = (f"{len(fc_ids)} calls carry its launch key ({list(fc_ids)}) — {len(fc_ids)} spawns of one reservation",
                           f"stop all but one of them (`modal app stop` / the dashboard, app {app}) and `--adopt {cid}=<the "
                           f"survivor's fc_id>`, or stop them all and `--release {cid}`")
        return f"{TAG} {cid}: AMBIGUOUS reservation ({why}; key {key}); NOT re-spawned. Reconcile it: {remedy}"

    def _reconcile_reservations(self) -> Dict[str, List[str]]:
        """Every reservation left by a previous process is settled through the DURABLE journal before anything is
        adopted or spawned. Exactly one DISTINCT call id on the ``spawned.jsonl`` lines carrying the reservation's key
        names the call Modal accepted -> the entry becomes ACTIVE with that id (adopted below like any other; a line
        written twice for one call is one call). No id -> the spawn's fate is UNKNOWN (the process died between Modal
        accepting the call and the launcher's fsync'd journal write); two or more -> two spawns of one reservation. Either
        way the reservation STAYS, stamped with the ids the journal names (``journal_fc_ids``, what the ceiling counts it
        as), the configuration is never re-spawned, the log names the remedies, and the ambiguous map is RETURNED so
        ``run`` fails closed at once."""
        ambiguous = self.ambiguous_reservations()
        for cid, ent in list(self.state[LAUNCHING].items()):
            if cid in ambiguous:
                ent["journal_fc_ids"] = list(ambiguous[cid])
                self.log(self._ambiguity_line(cid, ent, ambiguous[cid]))
                continue
            (fc_id,) = self._reservation_calls(ent)
            lines = journal_lines_for(self.spawned_log, str(ent.get(LAUNCH_KEY_FIELD)))
            self.state.setdefault("reconciled", {})[cid] = {"launch_key": ent.get(LAUNCH_KEY_FIELD), "fc_id": fc_id,
                                                            "how": "spawned.jsonl", "t": self.clock()}
            self._transition(cid, "active", entry={**ent, "fc_id": fc_id, "spawned_at": lines[0].get("t"), "journal_fc_ids": [fc_id]},
                             note=f"reservation reconciled from {self.spawned_log.name}: FunctionCall {fc_id}"
                                  + (f" ({len(lines)} journal lines, one call)" if len(lines) > 1 else ""))
        if ambiguous:
            self._persist()
        return ambiguous

    def outstanding(self) -> int:
        """The remote calls this process must assume are RUNNING — every active entry plus every
        reservation, a reservation weighted by the distinct calls its journal names and never less than one (an
        unjournaled spawn may be running). ``_fill_slots`` holds THIS under ``max_outstanding``; ``len(active)`` alone
        let 3 active + 1 ambiguous reservation fill five more slots (nine calls at Modal)."""
        return len(self.state["active"]) + sum(max(1, len(ent.get("journal_fc_ids") or ()))
                                               for ent in self.state[LAUNCHING].values())

    def _fail_closed(self, ambiguous: Mapping[str, Sequence[str]]) -> int:
        """An ambiguous reservation may name a running GPU call, so the restart stops HERE — nothing
        adopted, nothing preflighted, nothing spawned, no app stopped — and exits FAILED with the remedies named; the
        active calls keep running under their detached apps and stay adoptable on the next start."""
        self.log(f"{TAG} {len(ambiguous)} ambiguous reservation(s) pending {sorted(ambiguous)}: nothing adopted, nothing "
                 "spawned — the wave exits FAILED until each is reconciled (`--adopt <cid>=<fc_id>` after finding the call, "
                 "or `--release <cid>` after verifying none runs), then re-run with the same --state")
        self._persist()
        self.print_summary()
        return EXIT_FAILED

    def _adopt_one(self, cid: str, ent: Mapping):
        """The handle of one active entry's call, adopted by id — or None when the adoption could not be settled: an
        entry without an id is a terminal record failure; any exception out of ``FunctionCall.from_id`` (it opens the
        client and decodes no result) is a status NOT obtained — a transport error, a rejected RPC and a ``NotFoundError``
        for an id the server does not know alike — so the entry stays active (marked, re-tried by the loop under the
        backoff) and a wrong id is the operator's to settle with ``--release`` after looking it up. Only remote evidence
        (``poll_outcome`` terminal: the remote-traceback mark or an SDK result class) files the record's failure."""
        fc_id = ent.get("fc_id")
        if not fc_id:
            self._transition(cid, "failed", error="active entry without a FunctionCall id")
            return None
        try:
            fc = self.adopt(str(fc_id))
        except KeyboardInterrupt:
            raise
        except BaseException as e:  # noqa: BLE001 — classified below; terminal only on remote evidence
            if poll_outcome(e) == OUTCOME_TERMINAL:
                self._transition(cid, "failed", error=f"adopt {fc_id}: {_exc_text(e)}")
                return None
            self._status_unknown(cid, e, what=f"adopt {fc_id}")
            return None
        self.handles[cid] = fc
        self.log(f"{TAG} {cid}: adopted FunctionCall {fc_id} ({ent.get('function')})")
        return fc

    def _adopt_active(self) -> None:
        for cid, ent in list(self.state["active"].items()):
            self._adopt_one(cid, ent)

    # -- the status boundary (module docstring) --------------------------------
    def _status_unknown(self, cid: str, exc: BaseException, *, what: str) -> None:
        """A poll (or adoption) that could NOT obtain the call's status: the entry stays ACTIVE and keeps its slot, the
        failure is persisted on it (state + dispatch record), and its next poll is deferred by a doubling tick backoff.
        Never a transition: a status-unknown call is assumed RUNNING until a status says otherwise."""
        ent = self.state["active"][cid]
        now = self.clock()
        n = int(ent.get("status_unknown_polls") or 0) + 1
        since = ent.get("status_unknown_since")
        if since is None:
            since = ent["status_unknown_since"] = now
        ent["status_unknown_polls"] = n
        ent["status_unknown_error"] = f"{what}: {_exc_text(exc)}"
        wait = min(2 ** (n - 1), STATUS_UNKNOWN_MAX_BACKOFF_TICKS)
        ent["status_unknown_retry_tick"] = self.tick + wait
        self._persist()
        self._write_record(cid, "active", ent)
        self.log(f"{TAG} {cid}: status UNKNOWN ({ent['status_unknown_error']}) — the call is assumed RUNNING and keeps its "
                 f"slot; failed poll {n}, next poll in {wait} tick(s); unknown for {max(0.0, now - float(since)):.0f} s of the "
                 f"{self.status_unknown_max_s:.0f} s budget")

    def _status_known(self, cid: str) -> None:
        """A status WAS obtained for an entry that had unknown marks: clear them (state + record) and say so."""
        ent = self.state["active"].get(cid)
        if ent is None or ent.get("status_unknown_since") is None:
            return
        n, since = ent.get("status_unknown_polls"), float(ent.get("status_unknown_since") or self.clock())
        for k in ("status_unknown_since", "status_unknown_error", "status_unknown_polls", "status_unknown_retry_tick"):
            ent.pop(k, None)
        self._persist()
        self._write_record(cid, "active", ent)
        self.log(f"{TAG} {cid}: status re-obtained after {n} failed poll(s) ({max(0.0, self.clock() - since):.0f} s unknown)")

    def _poll_due(self, ent: Mapping) -> bool:
        return int(ent.get("status_unknown_retry_tick") or 0) <= self.tick

    def status_unknown_expired(self) -> Dict[str, Mapping]:
        """The active entries whose status has been unknown for at least the budget — what fails the wave closed."""
        now = self.clock()
        return {cid: ent for cid, ent in self.state["active"].items()
                if ent.get("status_unknown_since") is not None and now - float(ent["status_unknown_since"]) >= self.status_unknown_max_s}

    def _fail_closed_status_unknown(self, expired: Mapping[str, Mapping]) -> int:
        """The status of a call stayed unknown past the budget: exit FAILED with the state persisted, NOTHING cancelled
        (no ``cleanup_apps`` — the calls may be running under their detached apps and stay adoptable), and the remedies
        named — the reservation path's vocabulary."""
        for cid, ent in expired.items():
            self.log(f"{TAG} {cid}: status UNKNOWN for {self.clock() - float(ent['status_unknown_since']):.0f} s (>= the "
                     f"{self.status_unknown_max_s:.0f} s budget; last: {ent.get('status_unknown_error')}); FunctionCall "
                     f"{ent.get('fc_id')} under app {ent.get('app_id') or self.app_name}. NOT re-queued, NOT cancelled. Either re-run "
                     f"with the same --state once the Modal client obtains statuses again — the transport back, the rate limit "
                     f"lifted — (the call is adopted by id and polled again), or look the call up (the Modal dashboard / `modal app "
                     f"logs`) and `--release {cid}` (verified: no call runs; re-queued next start) or `--adopt {cid}=<fc_id>` "
                     f"(re-pointed)")
        self.log(f"{TAG} {len(expired)} call(s) with status unknown past the budget {sorted(expired)}: the wave exits FAILED, "
                 f"state persisted, {len(self.state['active'])} active call(s) left RUNNING and adoptable, nothing spawned further")
        self._persist()
        self.print_summary()
        return EXIT_FAILED

    def _attest_serving_path(self, cid: str, rec: Mapping) -> Optional[Dict[str, str]]:
        """For a row that REQUESTED a serving path, read the completed store's manifests off its out_dir
        and hold the attested path to the request. Returns the two attestation fields to ride the done entry, or None after
        filing the row FAILED (``REC_SERVING_PATH_MISMATCH``: a mismatch, an unreadable store, a torn manifest) — the wave
        then exits non-zero. A row that requested nothing (the frozen matrix) is filed as before."""
        row = self._row(cid)
        requested = row_serving_path(row)
        if requested is None:
            return {}
        out_dir, _subset = row_paths(row, self.cap, wave_root=self.wave_root)
        try:
            attested, source = attest_store_serving_path(list(self.store_manifests(out_dir)))
        except Exception as e:  # noqa: BLE001 — every failure to attest is a refusal by name
            self._transition(cid, "failed", rec={**dict(rec), "status": REC_SERVING_PATH_MISMATCH},
                             error=f"{REC_SERVING_PATH_MISMATCH}: requested {requested!r}; the store at {out_dir} attests nothing readable — {e}")
            return None
        if attested != requested:
            self._transition(cid, "failed", rec={**dict(rec), "status": REC_SERVING_PATH_MISMATCH},
                             extra={RECORD_KEY_ATTESTED_SERVING_PATH: attested, RECORD_KEY_ATTESTED_SOURCE: source},
                             error=f"{REC_SERVING_PATH_MISMATCH}: requested {requested!r}, the completed store at {out_dir} attests "
                                   f"{attested!r} ({source}) — the row is not the measurement it was planned as")
            return None
        return {RECORD_KEY_ATTESTED_SERVING_PATH: attested, RECORD_KEY_ATTESTED_SOURCE: source}

    def _file_rec(self, cid: str, rec: object) -> None:
        status = rec.get("status") if isinstance(rec, Mapping) else None
        if status in self.ok_statuses:
            attested = self._attest_serving_path(cid, rec)
            if attested is None:
                return                                      # filed FAILED by the attestation
            self._transition(cid, "done", rec=rec, extra=attested)
        elif C.is_refusal_status(status):
            self._transition(cid, "refused", rec=rec, note="never retried on GPU")
        else:
            self._transition(cid, "failed", rec=rec if isinstance(rec, Mapping) else {"status": repr(rec)},
                             error=f"REC status {status!r}")

    def _poll_active(self) -> None:
        for cid in list(self.state["active"]):
            ent = self.state["active"][cid]
            if not self._poll_due(ent):
                continue                                    # backing off after an unknown-status poll
            fc = self.handles.get(cid)
            if fc is None:
                # no handle yet (an adoption a transport error deferred): retry it under the same classification
                fc = self._adopt_one(cid, ent)
                if fc is None:
                    continue
            try:
                rec = fc.get(timeout=0)
            except KeyboardInterrupt:
                raise
            except BaseException as e:  # noqa: BLE001 — classified; terminal only on remote evidence, never assumed
                outcome = poll_outcome(e)
                if outcome == OUTCOME_RUNNING:
                    self._status_known(cid)
                elif outcome == OUTCOME_UNKNOWN:
                    # the status boundary: NOT a failure of the call — the entry keeps its slot, the error rides the entry
                    self._status_unknown(cid, e, what="poll")
                else:
                    # remote evidence: the call's own result (a marked remote exception or an SDK result class)
                    self._transition(cid, "failed", error=f"{type(e).__name__}: {str(e)[:600]}")
                continue
            self._status_known(cid)
            self._file_rec(cid, rec)

    def _fill_slots(self) -> None:
        # The ceiling is on OUTSTANDING calls — active entries AND reservations (by the calls their journal
        # names) — never on the active bucket alone
        while self.state["queued"] and self.outstanding() < self.max_outstanding:
            cid = self.state["queued"][0]
            row, cfg = self._row(cid), self._cfg(cid)
            if self.store_complete(cfg):
                self._transition(cid, "done", rec={"status": "DONE"}, note="store validated; not spawned")
                continue
            if self.admitted is not None and not (self.admitted.get(row["slug"]) or {}).get("ok"):
                # No GPU spawn without the preflight's verdict for THIS slug at THIS cap
                raise RuntimeError(f"{TAG} {cid}: slug {row['slug']!r} was not admitted at cap {self.cap} "
                                   f"(admitted: {sorted(k for k, v in self.admitted.items() if v.get('ok'))}) — "
                                   "the preflight vouches for the planned configurations before any spawn; nothing spawned")
            fn = function_for(self.launcher, row)
            # The RESERVATION is persisted (state + dispatch record) BEFORE Modal is asked; the configuration
            # carries the idempotent key into the launcher's spawned.jsonl line, which is what a restart reconciles by
            key = mint_launch_key(self.run_id, cid)
            reserved = {"function": _fn_name(fn), "name": cfg.get("name"), "slug": row["slug"], "cell": row["cell"],
                        "app_id": self.app_id, LAUNCH_KEY_FIELD: key, "reserved_at": self.clock(), "fc_id": None}
            self._transition(cid, LAUNCHING, entry=reserved, note="reservation persisted before the spawn")
            handles = self.launcher.launch_configurations(
                fn, [{**cfg, LAUNCH_KEY_FIELD: key}], kwargs_for(row, sha=self.sha, run_id=self.run_id),
                spawned_log=self.spawned_log, wait=False)
            if len(handles) != 1:
                raise RuntimeError(f"{cid}: launch_configurations returned {len(handles)} handles for one configuration")
            h = handles[0][1]
            ent = {**reserved, "fc_id": getattr(h, "object_id", None), "spawned_at": self.clock()}
            if isinstance(h, Mapping):          # a function without .spawn ran inline: its REC is in hand
                self._transition(cid, "active", entry=ent, note="ran inline")
                self._file_rec(cid, h)
                continue
            self.handles[cid] = h
            self._transition(cid, "active", entry=ent, note=f"FunctionCall {ent['fc_id']} on {ent['function']}")

    def _stop(self, why: str, rc: int) -> int:
        """Leave: the state persisted, every active call RUNNING under its detached app (adoptable by id on the next
        start — nothing is cancelled here; ``--stop`` is the one explicit cancellation) and the adopt line printed."""
        self.log(f"{TAG} {why}: {len(self.state['active'])} active, {len(self.state['queued'])} queued; "
                 "state persisted, active calls keep running under their detached app(s) "
                 f"{sorted({str(e.get('app_id')) for e in self.state['active'].values() if e.get('app_id')})} "
                 "and stay adoptable — re-run with the same --state to adopt them, or --stop to cancel them")
        self._persist()
        self.print_summary()
        return rc

    def cleanup_apps(self) -> Dict[str, int]:
        """A CLEAN completion — nothing queued, nothing active, so no call can be cancelled — stops every detached app
        this wave recorded (``app_ids``), so finished waves do not accumulate idle detached apps. Returns the CLI rc per
        app id; a non-zero rc is logged, never fatal (the work is done; the app is a shell)."""
        out: Dict[str, int] = {}
        if self.state["queued"] or self.state["active"]:
            raise RuntimeError(f"{TAG} cleanup_apps with {len(self.state['queued'])} queued / "
                               f"{len(self.state['active'])} active — stopping an app now would cancel running calls")
        for app_id in list(self.state.get("app_ids") or []):
            rc = int(self.stop_app(str(app_id)))
            out[str(app_id)] = rc
            self.log(f"{TAG} cleanup: modal app stop {app_id} -> rc {rc}" + ("" if rc == 0 else " (left running; stop it by hand)"))
        self.state["apps_stopped"] = {**(self.state.get("apps_stopped") or {}), **out}
        self._persist()
        return out

    def run(self) -> int:
        if self.stop_path.is_file():
            self.log(f"{TAG} STOP sentinel {self.stop_path} present: remove it to dispatch; nothing spawned")
            return EXIT_STOPPED
        ambiguous = self._reconcile_reservations()
        if ambiguous:
            # Fail closed AT ONCE — before adopting, before the first spawn
            return self._fail_closed(ambiguous)
        self._adopt_active()
        self._persist()
        deadline, self.tick = self.clock() + self.max_hours * 3600.0, 0
        while True:
            if self.stop_path.is_file():
                return self._stop("STOP sentinel: spawning nothing more", EXIT_STOPPED)
            self._poll_active()
            expired = self.status_unknown_expired()
            if expired:
                # the status boundary: fail closed BEFORE filling a slot — nothing cancelled, nothing re-queued
                return self._fail_closed_status_unknown(expired)
            self._fill_slots()
            if not self.state["queued"] and not self.state["active"]:
                break
            self.tick += 1
            if (self.max_ticks is not None and self.tick >= self.max_ticks) or self.clock() >= deadline:
                return self._stop(f"guard: tick {self.tick} / deadline reached", EXIT_FAILED)
            self.sleep(self.poll_s)
        self._persist()
        self.print_summary()
        if self.state[LAUNCHING]:
            # A reservation left at the end may name a RUNNING call — the apps stay up, the wave is not clean.
            # (Unreachable given the fail-closed exit above: a reservation this process makes is flipped to active in
            # the same _fill_slots step; kept as the backstop for the invariant, never as the reporting path)
            self.log(f"{TAG} {len(self.state[LAUNCHING])} reservation(s) left pending "
                     f"{sorted(self.state[LAUNCHING])}: reconcile with --adopt / --release before the wave can complete")
            return EXIT_FAILED
        self.cleanup_apps()
        return EXIT_OK if not (self.state["refused"] or self.state["failed"]) else EXIT_FAILED

    # -- reporting -----------------------------------------------------------
    def queue_lines(self) -> List[str]:
        st = self.state
        lines = [f"{TAG} queue at cap {self.cap} (app {self.app_name}, run {self.run_id}, <= {self.max_outstanding} "
                 f"outstanding): {len(st['queued'])} to spawn, {len(st[LAUNCHING])} reserved, {len(st['active'])} active, "
                 f"{len(st['done'])} done, {len(st['refused'])} refused, {len(self.held)} held"]
        for cid in st["queued"]:
            row, cfg = self._row(cid), self._cfg(cid)
            out_dir, subset = row_paths(row, self.cap, wave_root=self.wave_root)
            lines.append(f"  {row['slug']} x {row['cell']} -> {cid} | {_fn_name(function_for(self.launcher, row))} | "
                         f"{out_dir} | subset {subset or '-'} | {cfg.get('name')}")
        return lines + [f"  HELD {cid}: {why}" for cid, why in self.held.items()]

    def summary_lines(self) -> List[str]:
        st = self.state
        lines = [f"{TAG} summary run {self.run_id} cap {self.cap}: " + ", ".join(
            f"{s} {len(st[s])}" for s in ("done", "refused", "failed", "active", LAUNCHING, "queued")) + f", held {len(self.held)}"]
        for s in ("done", "refused", "failed", "active"):
            lines += [f"  {s:<8} {cid:<50} {e.get('status') or e.get('error') or ''} fc {e.get('fc_id') or '-'}"
                      + (f" | status UNKNOWN since {e.get('status_unknown_since')} ({e.get('status_unknown_polls')} failed polls: "
                         f"{e.get('status_unknown_error')})" if e.get("status_unknown_since") is not None else "")
                      for cid, e in st[s].items()]
        lines += [f"  {'AMBIGUOUS':<8} {cid:<50} reservation {e.get(LAUNCH_KEY_FIELD)} — --adopt {cid}=<fc_id> | --release {cid}"
                  for cid, e in st[LAUNCHING].items()]
        lines += [f"  {'queued':<8} {cid}" for cid in st["queued"]]
        return lines + [f"  {'held':<8} {cid:<50} {str(why)[:90]}" for cid, why in self.held.items()]

    def print_summary(self) -> None:
        for line in self.summary_lines():
            self.log(line)


def dispatch(launcher, matrix: Mapping, **kw) -> int:
    return Dispatcher(launcher, matrix, **kw).run()


# -- STOP / preflight / production wiring ------------------------------------
def default_stop_app(app_id: str, *, run_subprocess=subprocess.run) -> int:
    """``modal app stop <app id> --yes`` — the one cancellation of a wave's running calls."""
    return int(getattr(run_subprocess([*STOP_APP_ARGV, str(app_id), "--yes"], check=False), "returncode", 1))


def recorded_app_ids(state_path: Path) -> List[str]:
    """Every Modal app id the wave's state file recorded (``app_ids``, then the active entries', then ``app_id``);
    ``[]`` when there is no state or it records none (a wave dispatched before the ids were persisted)."""
    p = Path(state_path)
    if not p.is_file():
        return []
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return []
    ids = [*(st.get("app_ids") or []), *(e.get("app_id") for e in (st.get("active") or {}).values()), st.get("app_id")]
    return [str(i) for i in dict.fromkeys(i for i in ids if i)]


def request_stop(state_path: Path, app_name: str, *, run_subprocess=subprocess.run,
                 log: Callable[[str], None] = print) -> int:
    """The EXPLICIT stop: write the sentinel FIRST (a live loop then spawns nothing more), then cancel the
    running calls by stopping EVERY detached app the state recorded (``recorded_app_ids``: a restarted wave has one
    app per run) through the CLI — by id, the name being shared by every run of the wave; the name alone when the
    state records no id (a wave from before the ids were persisted). ``--yes`` because the CLI aborts without a TTY."""
    stop = Path(state_path).parent / STOP_SENTINEL
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.write_text(f"stop requested at {time.strftime('%Y-%m-%dT%H:%M:%S')} for app {app_name}\n", encoding="utf-8")
    targets = recorded_app_ids(state_path) or [app_name]
    rc = EXIT_OK
    for target in targets:
        argv = [*STOP_APP_ARGV, target, "--yes"]
        log(f"{TAG} wrote {stop}; running: {' '.join(argv)}")
        if getattr(run_subprocess(argv, check=False), "returncode", 0) != 0:
            rc = EXIT_FAILED
    return rc


def reconcile_reservation(state_path: Path, cid: str, *, fc_id: Optional[str] = None, release: bool = False,
                          log: Callable[[str], None] = print) -> int:
    """The operator settles ONE ambiguous reservation on the state file — ``--adopt <cid>=<fc_id>`` moves it to
    ``active`` under the call the operator found (adopted by id on the next start, never re-spawned); ``--release <cid>``
    (the operator VERIFIED that no call runs) drops it so the next start re-queues the configuration. The status boundary
    (module docstring) takes the SAME two remedies for an ACTIVE entry whose status is UNKNOWN (``status_unknown_since``
    set): ``--release`` after verifying the call is gone or finished (re-queued next start), ``--adopt`` to re-point it
    (the marks cleared; polled afresh). A cid that is neither reserved nor status-unknown refuses by name — an active
    call whose status is being obtained is nobody's to settle; the settlement is recorded under ``reconciled``."""
    p = Path(state_path)
    if not p.is_file():
        log(f"{TAG} no state file at {p}")
        return EXIT_FAILED
    st = json.loads(p.read_text(encoding="utf-8"))
    st.setdefault(LAUNCHING, {})
    st.setdefault("active", {})
    bucket = LAUNCHING
    ent = st[LAUNCHING].get(cid)
    if ent is None:
        unknown = {c: e for c, e in st["active"].items() if e.get("status_unknown_since") is not None}
        if cid in unknown:
            ent, bucket = unknown[cid], "active"
        else:
            log(f"{TAG} {cid} is neither a pending reservation (pending: {sorted(st[LAUNCHING])}) nor an active call with "
                f"status unknown (unknown: {sorted(unknown)}) — nothing to settle")
            return EXIT_FAILED
    if bool(fc_id) == bool(release):
        log(f"{TAG} exactly one of --adopt <cid>=<fc_id> / --release <cid>")
        return EXIT_FAILED
    ent = dict(st[bucket].pop(cid))
    was_unknown = ent.get("status_unknown_since")
    for k in ("status_unknown_since", "status_unknown_error", "status_unknown_polls", "status_unknown_retry_tick"):
        ent.pop(k, None)
    ent["t"] = time.time()
    tail = f" (status unknown since {was_unknown})" if was_unknown is not None else ""
    if fc_id:
        st["active"][cid] = {**ent, "fc_id": str(fc_id), "spawned_at": ent.get("spawned_at") or ent.get("reserved_at")}
        how = f"operator --adopt {fc_id}{tail}"
    else:
        st.setdefault("released", {})[cid] = ent
        how = f"operator --release (verified: no running call){tail}"
    st.setdefault("reconciled", {})[cid] = {"launch_key": ent.get(LAUNCH_KEY_FIELD), "fc_id": fc_id, "how": how, "t": ent["t"]}
    atomic_write_json(p, st)
    log(f"{TAG} {cid}: reservation settled — {how}; re-run with the same --state to continue")
    return EXIT_OK


def preflight_argv(launcher, *, sha: str, app_name: str, slug: str, cells: Sequence[str], cap: int = PASS1_CAP,
                   serving_path: Optional[str] = None) -> Tuple[List[str], Dict[str, str]]:
    """One slug's preflight-only invocation: the launcher's own argv composition
    plus ``--pf`` (no ``--pass1``, so nothing launches), the ``TINYCAP_APP_NAME=``
    prefix moved from argv into the environment — and the dispatch's CAP as
    the OFF setter above pass 1, so the preflight plans the cells at the geometry
    entry the containers will serve (``capability:ifeval[_longsi]_<cap>``: the cap
    from the setter, the prompt class from the cell) and its Nemotron check reads
    THAT entry's certification, never the 2,048 default's. ``serving_path``: the rows' stamp, handed to the
    launcher's argv composer so the preflight plans on the engine the containers will serve (the composer spells the flag
    whenever it is not the entry-point default); None = the composer's own default, as before."""
    kw = {"off_cap_by_task": f"{C.IFEVAL_TASK_NAME}={int(cap)}"} if int(cap) != PASS1_CAP else {}
    if serving_path is not None:
        kw["serving_path"] = str(serving_path)
    argv = list(launcher.launch_argv(sha, cells=",".join(cells), modes=C.THINKING_OFF, controls=C.CONTROL_IFEVAL,
                                     samples=C.IFEVAL_SAMPLES_PER_ITEM, model=slug, app_name=app_name,
                                     reduction_mode=reduction_mode_for(serving_path), **kw))
    prefix = launcher.app_name_command(app_name)
    if argv and argv[0] == prefix:
        argv.pop(0)
    key, _, val = prefix.partition("=")
    return modal_run_argv(argv) + [PREFLIGHT_FLAG], {**os.environ, key: val}


def modal_run_argv(argv: Sequence[str]) -> List[str]:
    """The launcher's printed ``modal run <relative script> …`` re-spelled for a SUBPROCESS: this interpreter's
    ``-m modal`` in place of a bare ``modal`` executable (absent from ``PATH`` under an absolute venv interpreter) and the
    launcher script by its absolute repository path (the printed argv is relative to the repository root, the operator's
    copy-paste convention, not a working-directory guarantee). An argv of another shape is refused by name — the
    dispatcher never guesses which token is the executable."""
    argv = list(argv)
    if len(argv) < 3 or argv[0] != "modal" or argv[1] != "run" or not str(argv[2]).endswith("tinycap_served.py"):
        raise ValueError(f"{TAG} the launcher's argv is not `modal run <launcher script> …`: {argv[:3]} — cannot compose "
                         "the preflight subprocess")
    return [*MODAL_ARGV, "run", str(LAUNCHER_SCRIPT), *argv[3:]]


def run_preflights(launcher, rows: Sequence[Mapping], *, sha: str, app_name: str, cap: int = PASS1_CAP,
                   run_subprocess=subprocess.run, log: Callable[[str], None] = print, clock: Callable[[], float] = time.time
                   ) -> Dict[str, Dict[str, object]]:
    """Per slug, in matrix order, the queued rows' cells at ``cap``: ``{slug: {"ok": bool, "rc": int, "cap": cap,
    "cells": [...], "argv": [...], "t": ...}}`` — the ADMISSION record the dispatcher persists and holds every spawn to.
    A slug whose preflight exited non-zero is not admitted (``ok`` False)."""
    cells_by_slug: Dict[str, List[str]] = {}
    rows_by_slug: Dict[str, List[Mapping]] = {}
    for r in rows:
        cells_by_slug.setdefault(r["slug"], []).append(r["cell"])
        rows_by_slug.setdefault(r["slug"], []).append(r)
    out: Dict[str, Dict[str, object]] = {}
    for slug, cells in cells_by_slug.items():
        sp = rows_serving_path(rows_by_slug[slug])
        argv, env = preflight_argv(launcher, sha=sha, app_name=app_name, slug=slug, cells=cells, cap=cap, serving_path=sp)
        log(f"{TAG} preflight {slug} @ cap {cap}: {' '.join(argv)}")
        rc = int(getattr(run_subprocess(argv, env=env, check=False), "returncode", 1))
        out[slug] = {"ok": rc == 0, "rc": rc, "cap": int(cap), "cells": list(cells), "argv": list(argv), "t": clock(),
                     **({ROW_SERVING_PATH_KEY: sp} if sp is not None else {})}
    return out


def failed_preflights(admission: Mapping[str, Mapping[str, object]]) -> List[str]:
    return [slug for slug, v in admission.items() if not v.get("ok")]


def import_launcher(app_name: str):
    """The production launcher, its Modal app named from the environment at import."""
    os.environ[C.MODAL_APP_NAME_ENV] = app_name
    import tinycap_served as L  # scripts/modal is on sys.path above
    if getattr(L, "APP_NAME", app_name) != app_name:
        raise SystemExit(f"{TAG} launcher app {L.APP_NAME!r} != {app_name!r}")
    return L


def _app_context(launcher):
    """The Modal app the loop spawns into — DETACHED: the calls outlive this process; an ephemeral
    ``app.run()`` would terminate every running task when the client disconnects. A launcher without an ``app``
    (a test stub) runs in a null context."""
    app = getattr(launcher, "app", None)
    return app.run(detach=True) if app is not None and hasattr(app, "run") else contextlib.nullcontext()


def app_id_of(running_app) -> Optional[str]:
    """The live app's id off the object ``app.run`` yields (``modal.App.app_id``); None for a stub / null context."""
    app_id = getattr(running_app, "app_id", None)
    return str(app_id) if app_id else None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="IFEval canonical battery: thin wave-wide dispatcher (<= 8 outstanding)")
    ap.add_argument("--matrix", required=True, help="the frozen matrix JSON")
    ap.add_argument("--sha", required=True, help="the staged tree sha the containers run")
    ap.add_argument("--rung", type=int, default=PASS1_CAP, help=f"the cap to dispatch (default {PASS1_CAP} = pass 1)")
    ap.add_argument("--rows", default="", help="comma-separated configuration ids to narrow to (ladder promotions)")
    ap.add_argument("--app-name", default="", help="Modal app (default ifeval-canonical-<sha7>)")
    ap.add_argument("--wave-root", default="", help=f"wave root on the volume (default {C.VOL_TINYCAP_ROOT}/ifeval_canonical/<sha7>)")
    ap.add_argument("--local-root", default="", help=f"local wave root (default {C.LOCAL_OUT_ROOT}/ifeval_canonical/<sha7>)")
    ap.add_argument("--state", default="", help=f"state file (default <local root>/{IFEVAL_DISPATCH_STATE})")
    ap.add_argument("--run-id", default="", help="per-wave run id (minted and persisted when absent)")
    ap.add_argument("--poll-s", type=float, default=DEFAULT_POLL_S)
    ap.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    ap.add_argument("--max-ticks", type=int, default=None)
    ap.add_argument("--status-unknown-max-s", type=float, default=DEFAULT_STATUS_UNKNOWN_MAX_S,
                    help="how long one active call's status may stay UNKNOWN (no status obtained on any poll: a transport error, a "
                         "rejected RPC such as a rate limit, an undecodable result) before the wave exits FAILED with nothing "
                         "cancelled (default %(default)s s); the call stays adoptable, --release / --adopt settle it")
    ap.add_argument("--resume", action="store_true", help="require an existing state file (re-running with the same --state resumes anyway)")
    ap.add_argument("--preflight", action="store_true",
                    help="(the default since D27 B3; kept for old command lines) run the launcher's --pf per (slug, cap) "
                         "before dispatching")
    ap.add_argument("--no-preflight", action="store_true",
                    help="EXPLICIT opt-out of the admission (a smoke on a stub launcher only): spawns without the "
                         "preflight's verdict; the container's own fail-closed certification check still applies")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop", action="store_true", help="write the STOP sentinel and `modal app stop <app> --yes`")
    ap.add_argument("--adopt", default="", metavar="CID=FC_ID",
                    help="settle an AMBIGUOUS reservation (or re-point an active call whose status stayed UNKNOWN): the call the "
                         "operator found for the configuration (adopted by id on the next start, never re-spawned)")
    ap.add_argument("--release", default="", metavar="CID",
                    help="settle an AMBIGUOUS reservation (or an active call whose status stayed UNKNOWN) after VERIFYING no "
                         "call runs: the configuration is re-queued on the next start")
    return ap


def main(argv: Optional[Sequence[str]] = None, *, launcher=None, sleep=time.sleep, clock=time.time, adopt=None,
         store_complete=None, run_subprocess=subprocess.run, log: Callable[[str], None] = lambda s: print(s, flush=True)
         ) -> int:
    a = build_parser().parse_args(list(argv) if argv is not None else None)
    app_name = a.app_name or app_name_for(a.sha)
    local_root = Path(a.local_root).expanduser() if a.local_root else local_root_for(a.sha)
    state_path = Path(a.state).expanduser() if a.state else local_root / IFEVAL_DISPATCH_STATE
    if a.stop:
        return request_stop(state_path, app_name, run_subprocess=run_subprocess, log=log)
    if a.adopt or a.release:
        if a.adopt:
            cid, _, fc_id = a.adopt.partition("=")
            return reconcile_reservation(state_path, cid, fc_id=fc_id, log=log)
        return reconcile_reservation(state_path, a.release, release=True, log=log)
    if a.resume and not state_path.is_file():
        raise SystemExit(f"{TAG} --resume: no state file at {state_path}")
    matrix = load_matrix(a.matrix)
    launcher = launcher if launcher is not None else import_launcher(app_name)

    def stop_app(app_id: str) -> int:
        return default_stop_app(app_id, run_subprocess=run_subprocess)

    def build(app_id: Optional[str]) -> Dispatcher:
        return Dispatcher(launcher, matrix, sha=a.sha, cap=a.rung, rows=[r for r in a.rows.split(",") if r] or None,
                          app_name=app_name, wave_root=a.wave_root, local_root=local_root, state_path=state_path,
                          run_id=a.run_id, poll_s=a.poll_s, max_hours=a.max_hours, max_ticks=a.max_ticks, sleep=sleep,
                          clock=clock, adopt=adopt, store_complete=store_complete, log=log, app_id=app_id,
                          stop_app=stop_app, status_unknown_max_s=a.status_unknown_max_s)
    if a.dry_run:
        d = build(None)
        for line in d.queue_lines() + [f"{TAG} dry run: nothing spawned"]:
            log(line)
        return EXIT_OK
    # ADMISSION before any spawn — the preflight per (slug, cap) over the queued rows' cells (the actual planned
    # configurations); a failed slug prevents EVERY spawn; the verdicts are persisted and every spawn is held to them
    probe = build(None)
    # An AMBIGUOUS reservation (the journal settles none / several calls for it) fails the restart
    # closed HERE — before a preflight is paid for, before the detached app is entered, before anything is adopted or
    # spawned; the remedies are the two settlement commands above
    ambiguous = probe.ambiguous_reservations()
    if ambiguous:
        for cid, fc_ids in ambiguous.items():
            log(probe._ambiguity_line(cid, probe.state[LAUNCHING][cid], fc_ids))
        log(f"{TAG} {len(ambiguous)} ambiguous reservation(s) pending {sorted(ambiguous)}: nothing preflighted, nothing "
            "adopted, nothing spawned — exit FAILED; settle each with --adopt / --release, then re-run with the same --state")
        return EXIT_FAILED
    queued_rows = [probe.rows_by_id[c] for c in probe.state["queued"]]
    if a.no_preflight:
        log(f"{TAG} --no-preflight: spawning WITHOUT the preflight's admission (explicit opt-out; the container's own "
            "fail-closed certification check still applies)")
        admission = {r["slug"]: {"ok": True, "rc": None, "cap": int(a.rung), "cells": None, "argv": None,
                                 "opt_out": "--no-preflight", "t": clock()} for r in queued_rows}
    else:
        admission = run_preflights(launcher, queued_rows, sha=a.sha, app_name=app_name, cap=a.rung,
                                   run_subprocess=run_subprocess, log=log, clock=clock)
        bad = failed_preflights(admission)
        if bad:
            log(f"{TAG} preflight failed for {bad} at cap {a.rung}; nothing spawned")
            return EXIT_FAILED
    # The DETACHED app is entered first, so the dispatcher records ITS id (state + every active entry) before
    # the first spawn; the calls it spawns outlive this process and are adopted by id on the next start
    with _app_context(launcher) as running_app:
        d = build(app_id_of(running_app))
        d.admitted = dict(admission)
        d.state["admission"] = {"cap": d.cap, "slugs": {k: dict(v) for k, v in admission.items()}}
        if d.app_id:
            log(f"{TAG} detached app {d.app_id} ({app_name}): running calls survive this process; "
                f"`modal app stop {d.app_id} --yes` (or --stop) cancels them")
        return d.run()


if __name__ == "__main__":
    sys.exit(main())
